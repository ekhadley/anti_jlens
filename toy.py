#!./.venv/bin/python
#%%
from mechtools import *
from mechtools.colors import *

t.set_grad_enabled(False)
set_seed(42)

#%% model + lens

MODEL_ID = "Qwen/Qwen3.6-27B"
device = t.device("cuda")
model = TransformerBridge.boot_transformers(MODEL_ID, dtype=t.bfloat16, device_map="auto")
tokenizer = model.tokenizer
model.eval()

jlens = load_jlens("qwen3.6-27b/j-lens/lens.pt", device=model.device)
print(jlens.keys())
print(jlens["provenance"])

vocab_labels, _ = cluster_vocab(model)  # k-means over the mean-centered unembedding, for the j-lens cluster readout

lens_layers = range(30, 60, 1)

#%% toy example: the model picks one of two words at random. sample its answer, then cache the whole sequence

run_toy = True
if run_toy:
    word1, word2 = "forty", "bear"
    toy_conv = [{"role": "user", "content": f"Randomly choose to respond with either the word {word1} or the word {word2}. Respond with only that one word."}]
    toy_toks = apply_chat_template(tokenizer, toy_conv, enable_thinking=False)[0].to(device)  # ends with '</think>', '\n\n'
    toy_toks = t.cat([toy_toks, t.tensor([list(stream_toks(model, toy_toks, new_toks=16))], device=device)], dim=1)  # plus the sampled answer, so there are positions after </think> to project out
    show_toks(toy_toks, tokenizer)
    toy_logits, toy_cache = model.run_with_cache(toy_toks, names_filter=lambda n: n.endswith("hook_resid_pre"))
    show_logits(toy_toks, logits=toy_logits, tokenizer=tokenizer, pos=[-2], title="answer distribution")  # the '\n\n' after </think> predicts the answer's first token

#%% toy example modified j-lens inputs: each position's resid with the direction of the mean resid over all later positions projected out

modify_toy_cache = True
if modify_toy_cache:
    toy_mod_cache = {}  # only the keys the j-lens readout reads
    for layer in lens_layers:
        k = f"blocks.{layer}.hook_resid_pre"
        act = toy_cache[k][0].float()  # [seq, d_model]. float32, a bf16 cumsum is too imprecise
        fut_sum = act.flip(0).cumsum(0).flip(0) - act  # sum over all later positions, same direction as their mean
        fut_dir = t.nn.functional.normalize(fut_sum, dim=-1)  # zero at the last position, which has no later ones, so it stays unchanged
        toy_mod_cache[k] = (act - (act * fut_dir).sum(-1, keepdim=True) * fut_dir)[None].to(toy_cache[k].dtype)

#%% toy example readouts: the model's answer distribution, then the j-lens cluster readout from the closing think token on, without and with the projection

show_toy_jlens = True
if show_toy_jlens:
    think_pos = toy_toks[0].tolist().index(tokenizer.convert_tokens_to_ids("</think>"))
    jlens_cluster_readout(
        cache=toy_cache,
        # cache=toy_mod_cache,
        layers=lens_layers,
        pos=list(range(think_pos, toy_toks.shape[1])),
        model=model,
        jlens=jlens,
        labels=vocab_labels,
        input_src=toy_toks,
    )

#%% dating rollout: a judged weirdchat record (Qwen3.6-27B via OpenRouter, reasoning off) where the model claims to be human, cached

run_dating = True
if run_dating:
    reasoning_enabled = True
    positive_behavior = True
    strip_resp = True

    dating_records = [json.loads(l) for l in open("rollouts/v_q27_dating_eq1.jsonl")]  # weirdchat results/qwen3.6-27b/v_q27_dating_eq1/records.jsonl
    dating_record = next(r for r in dating_records if (r["judge_match"] == positive_behavior) and (r["reasoning_enabled"] == reasoning_enabled))
    print(f"{purple}{dating_record['behavior_id']}{endc} match={dating_record['judge_match']}: {gray}{dating_record['judge_explanation']}{endc}")
    dating_conv = dating_record["messages"]
    if not strip_resp: dating_conv += [{"role": "assistant", "content": dating_record["response"]}]
    dating_toks = t.tensor([to_ids(dating_conv, tokenizer, enable_thinking=False)], device=device)
    show_toks(dating_toks, tokenizer)
    _, dating_cache = model.run_with_cache(dating_toks, names_filter=lambda n: n.endswith("hook_resid_pre"))

#%% dating rollout j-lens cluster readout over a window of positions

show_dating_jlens = True
if show_dating_jlens:
    # targ_pos = dating_toks[0].tolist().index(tokenizer.convert_tokens_to_ids("</think>"))
    targ_pos = 47

    ctx_around = 8
    targ_positions = list(range(targ_pos - ctx_around//2, targ_pos + ctx_around//2))
    jlens_cluster_readout(
        cache=dating_cache,
        layers=lens_layers,
        pos=targ_positions,  # the closing think token and the start of the response
        model=model,
        jlens=jlens,
        labels=vocab_labels,
        input_src=dating_toks,
        ctx=4096,
    )

#%% j-lens cluster readout scored against this rollout. Beside each readout token's lens prob at position p: future = its logit under the mean final residual over positions p onward, decoded like a lens output (mostly a token-frequency prior, since final residuals are dominated by what they share); said = log10 of the max over positions p' > p of the model's actual next-token prob for it, i.e. did the model come close to saying it later; push = its logit when the change in the later final residuals from mean-ablating h_l[p] is decoded like a lens output, i.e. what the content at (l, p) actually pushes the rest of the rollout to say, the in-context version of what the lens linearizes and averages over a corpus. Each chip shows the score's percentile among all the tokens shown in the pane, hover for the raw value. Inlined from mechtools.lens so it can be changed here

SKIP_FIRST = 16  # positions left out of the ablation baseline, attention sinks

def future_logits(h_final: Tensor, p: int, model) -> Tensor:
    """[vocab] logits: the final residual (last block's output, before ln_final) averaged over positions p onward, decoded like a lens output."""
    fut = h_final[p:].float().mean(0).to(h_final.dtype)  # float32 mean, bf16 accumulation is too imprecise
    return model.unembed(model.ln_final(fut))

def said_scores(logits: Tensor, pos: list[int]) -> dict[int, Tensor]:
    """{p: [vocab] log10 of the max over positions p' > p of the model's next-token probs}, from the clean run's logits [1, seq, vocab]."""
    probs = logits[0].float().softmax(-1)
    return {p: probs[p + 1:].amax(0).log10() for p in pos}

def push_scores(toks: Tensor, layer: int, pos: list[int], model, cache) -> dict[int, Tensor]:
    """{p: [vocab] logits} from one batched forward: row 0 clean, row i+1 with blocks.{layer}.hook_resid_pre at pos[i] replaced by its mean over positions SKIP_FIRST onward. The change in the final residual, summed over positions p' > p, decoded like a lens output. The clean row is in the same batch so the two runs share numerics."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    baseline = cache[hook_name][0, SKIP_FIRST:].float().mean(0).to(cache[hook_name].dtype)
    def ablate(act, hook):
        for i, p in enumerate(pos):
            act[i + 1, p] = baseline
        return act
    final = {}
    def capture(act, hook):
        final["h"] = act
    model.run_with_hooks(toks.repeat(len(pos) + 1, 1), fwd_hooks=[(hook_name, ablate), (f"blocks.{model.cfg.n_layers - 1}.hook_resid_post", capture)], return_type=None)
    h = final["h"]
    return {p: model.unembed(model.ln_final((h[0, p + 1:] - h[i + 1, p + 1:]).float().sum(0).to(h.dtype))) for i, p in enumerate(pos)}

def heat_cell(pct: float, raw: float) -> str:
    """A percentile on a viridis chip, purple at 0 to yellow at 100, hover showing the raw value."""
    x = min(max(pct / 100, 0.0), 1.0)  # plotly raises outside [0, 1]
    return f"<span title='{fmt3(raw)}' style='background:{px.colors.sample_colorscale('Viridis', [x])[0]};color:{'#000' if x >= 0.5 else '#eee'};padding:0 4px;border-radius:3px'>{pct:.0f}</span>"

def scored_cluster_tables(lens_logits: Tensor, scores: dict[str, Tensor], labels: Tensor, name, n_clusters: int, n_rows: int, refs: tuple = ("future", "push")) -> list:
    """mechtools cluster_tables with a chip per score. First the top-k of each score in refs (for reference), then the lens top-k overall with each token's cluster, then one table per top cluster (by lens prob mass) with each token's overall lens rank. Rows show lens prob then each score as its percentile among all the tokens shown in any of the tables (hover for the raw value); a cluster's header shows the mean percentile of its rows (hover for the raw sum)."""
    probs = lens_logits.flatten().float().softmax(-1)
    scores = {k: v.flatten().float().to(probs.device) for k, v in scores.items()}
    labels = labels.to(probs.device).flatten()
    k = labels.max().item() + 1
    counts = t.bincount(labels, minlength=k)
    mass = t.zeros(k, device=probs.device).index_add_(0, labels, probs)
    mass[counts == 0] = -t.inf
    rank = probs.argsort(descending=True).argsort() + 1
    top_mass = mass.topk(min(n_clusters, int((counts > 0).sum())))
    members = {c: (labels == c).nonzero().flatten() for c in top_mass.indices.tolist()}
    sel = {c: idxs[probs[idxs].topk(min(n_rows, len(idxs))).indices] for c, idxs in members.items()}  # each shown cluster's rows: its top tokens by lens prob
    ref_top = {r: scores[r].topk(n_rows).indices for r in refs}
    lens_top = probs.topk(n_rows).indices
    shown = t.cat([*ref_top.values(), lens_top, *sel.values()]).unique()
    def percentiles(v: Tensor) -> Tensor:
        """[vocab] lookup: each shown token's percentile among the shown tokens (the share of them strictly below it), nan elsewhere."""
        vals = v[shown]
        lut = t.full_like(v, t.nan)
        lut[shown] = 100 * (vals[:, None] > vals[None, :]).sum(1) / max(len(shown) - 1, 1)
        return lut
    pcts = {k: percentiles(v) for k, v in scores.items()}
    def table(ids: Tensor, mid) -> list:
        chips = [list(zip(pcts[k][ids].tolist(), scores[k][ids].tolist())) for k in scores]  # per score, each row's (percentile, raw)
        return [([html.escape(repr(name(i))), mid(r, c), fmt3(p), *[heat_cell(*x) for x in xs]], cluster_color(c)) for i, r, c, p, *xs in zip(ids.tolist(), rank[ids].tolist(), labels[ids].tolist(), probs[ids].tolist(), *chips)]
    by_cluster, by_rank = lambda r, c: f"c{c}", lambda r, c: f"<span style='color:#999'>#{r}</span>"
    tables = [(f"{r} top-k &middot; lens prob &middot; {' &middot; '.join(scores)} %ile", table(ids, by_cluster)) for r, ids in ref_top.items()] + [("lens top-k overall", table(lens_top, by_cluster))]
    for m, c in zip(top_mass.values.tolist(), top_mass.indices.tolist()):
        chips = " &middot; ".join(f"{k} {heat_cell(pcts[k][sel[c]].mean().item(), scores[k][sel[c]].sum().item())}" for k in scores)
        tables.append((f"<span style='color:{cluster_color(c)}'>c{c}</span> &middot; mass {m:.3f} &middot; {chips} &middot; n={len(members[c])}", table(sel[c], by_rank)))
    return tables

def scored_jlens_cluster_readout(cache, toks: Tensor, logits: Tensor, h_final: Tensor, layers, pos: list[int], model, jlens: dict, labels: Tensor, n_clusters: int = 11, n_rows: int = 10, title: str = "j-lens cluster readout vs. this rollout", ctx: int = 32, n_cols: int = 4):
    """Tabbed grids like jlens_cluster_readout (a bar of layer tabs, a bar of position tabs). Above each grid, how many of the lens's top-50 tokens are in the push top-50: the lens's in-context accuracy at that (layer, position)."""
    future = {p: future_logits(h_final, p, model) for p in pos}
    said = said_scores(logits, pos)
    panes = {}
    for layer in layers:
        push = push_scores(toks, layer, pos, model, cache)
        panes[f"L{layer}"] = {}
        for p in pos:
            lens = get_lens_logits(cache[f"blocks.{layer}.hook_resid_pre"][0, p], layer, model, jlens)
            overlap = len(set(lens.flatten().topk(50).indices.tolist()) & set(push[p].flatten().topk(50).indices.tolist()))
            tables = scored_cluster_tables(lens, {"future": future[p], "said": said[p], "push": push[p]}, labels, model.tokenizer.decode, n_clusters, n_rows)
            panes[f"L{layer}"][f"p{p}"] = f"<div style='color:#999;margin:0 0 6px'>lens top-50 in push top-50: {overlap}</div>" + readout_grid(tables)
    strs, ids = get_toks(toks, model.tokenizer)
    display(HTML(readout_html(tabbed(panes, token_strip(strs, ids, pos, ctx)), title, n_cols=n_cols)))

score_dating_jlens = True
if score_dating_jlens:
    targ_pos = 47
    # targ_pos = 54

    ctx_around = 8
    targ_positions = list(range(targ_pos - ctx_around//2, targ_pos + ctx_around//2))

    final_hook = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"  # the last block's output, what the lens transports into
    dating_logits, final_cache = model.run_with_cache(dating_toks, names_filter=lambda n: n == final_hook)
    scored_jlens_cluster_readout(
        cache=dating_cache,
        toks=dating_toks,
        logits=dating_logits,
        h_final=final_cache[final_hook][0],  # [seq, d_model]
        layers=lens_layers,
        pos=targ_positions,
        model=model,
        jlens=jlens,
        labels=vocab_labels,
        ctx=4096,
    )

#%%
