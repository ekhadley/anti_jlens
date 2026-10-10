#!./.venv/bin/python
#%%
from pathlib import Path
from safetensors import safe_open
from mechtools import *
from mechtools.colors import *

t.set_grad_enabled(False)
set_seed(42)

#%% model + NLA. The verbalizer is two LoRAs on the model's own weights (the warm-start LoRA, 3.7 GB, and the RL LoRA on top of it), loaded before the Bridge boots (peft cannot inject into the Bridge's wrappers) and disabled by default, so sampling and caching below are the plain model's; nla_show enables them around a read

MODEL_ID = "Qwen/Qwen3.6-27B"
device = t.device("cuda")
hf = load_hf_model(MODEL_ID, device_map="auto")
peft, nla_meta = load_nla(hf)  # ceselder/qwen3.6-27b-nla-rl, RL step 400: reads layer 42 only, the output of block 42
peft.base_model.set_adapter(nla_meta["adapters"])  # both adapters active together; loading alone leaves only the warm-start one active, which is neither the base nor the verbalizer
peft.base_model.disable_adapter_layers()
model = boot_bridge(hf)
tokenizer = model.tokenizer

nla_hook = f"blocks.{nla_meta['layer']}.hook_resid_post"  # the one activation the checkpoint reads

def nla_show(cache: dict, pos: list[int], toks: Tensor, n: int = 2, **kw) -> dict:
    """nla_readout of a hook_resid_post cache at `pos`, the adapters enabled for the read only: a tab per position, n samples each; {pos: [explanation per sample]}. kw goes to nla_readout (ctx, do_sample=False for greedy, enable_thinking=True for the training render, ...)."""
    peft.base_model.enable_adapter_layers()
    out = nla_readout(cache, pos, peft, tokenizer, nla_meta, n=n, input_src=toks, **kw)
    peft.base_model.disable_adapter_layers()
    return out

#%% toy example: the model picks one of two words at random. sample its answer, cache the whole sequence, then read the NLA from the closing think token on

run_toy = True
if run_toy:
    word1, word2 = "forty", "bear"
    toy_conv = [{"role": "user", "content": f"Randomly choose to respond with either the word {word1} or the word {word2}. Respond with only that one word."}]
    toy_toks = apply_chat_template(tokenizer, toy_conv, enable_thinking=False)[0].to(device)  # ends with '</think>', '\n\n'
    toy_toks = t.cat([toy_toks, t.tensor([list(stream_toks(model, toy_toks, new_toks=16))], device=device)], dim=1)  # plus the sampled answer
    show_toks(toy_toks, tokenizer)
    toy_logits, toy_cache = model.run_with_cache(toy_toks, names_filter=lambda n: n == nla_hook)
    show_logits(toy_toks, logits=toy_logits, tokenizer=tokenizer, pos=[-2], title="answer distribution")  # the '\n\n' after </think> predicts the answer's first token
    think_pos = toy_toks[0].tolist().index(tokenizer.convert_tokens_to_ids("</think>"))
    toy_nla = nla_show(toy_cache, list(range(think_pos, toy_toks.shape[1])), toy_toks)

#%% dating rollout: a judged weirdchat record (Qwen3.6-27B via OpenRouter, reasoning off) where the model claims to be human, cached and read over a window of positions

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
    _, dating_cache = model.run_with_cache(dating_toks, names_filter=lambda n: n == nla_hook)

    targ_pos = 47
    ctx_around = 8
    targ_positions = list(range(targ_pos - ctx_around//2, targ_pos + ctx_around//2))
    dating_nla = nla_show(dating_cache, targ_positions, dating_toks, ctx=4096)

#%% a saved direction from the ortho project (its data/vectors/Qwen3.6-27B, uploaded to the box): row 42 of the [n_layers, d_model] vector is resid_post 42, the one layer the NLA reads, so it goes in as is. The injection is norm-matched, so only the row's direction matters, not its norm (0.04). Both signs: the rows are cheat minus clean means, so -v is clean minus cheat

run_vector = True
if run_vector:
    VECTOR = Path("/workspace/vectors/Qwen3.6-27B/grader_parity_cheat_vs_clean.safetensors")  # the json sidecar is next to it
    with safe_open(VECTOR, "pt") as f:
        v = f.get_tensor("v")  # [64, 5120] float32
    n = 4
    signs = {"cheat - clean": 1.0, "clean - cheat": -1.0}
    peft.base_model.enable_adapter_layers()
    vec_nla = dict(zip(signs, nla_read(peft, tokenizer, t.stack([s * v[nla_meta["layer"]] for s in signs.values()]).to(device), nla_meta, n=n)))  # one batch of 2 x n rows, sign-major; {label: [explanation per sample]}
    peft.base_model.disable_adapter_layers()
    for label, samples in vec_nla.items():
        print(f"{purple}L{nla_meta['layer']}{endc} {cyan}{label}{endc}")
        for s in samples: print(f"    {gray}-{endc} {s}")
    show_table(["sign"] + [f"sample {j}" for j in range(1, n + 1)], [(label, *samples) for label, samples in vec_nla.items()], title=f"NLA on {VECTOR.stem}")

#%%
