#!./.venv/bin/python
#%%
from mechtools import *
from mechtools.colors import *

t.set_grad_enabled(False)
set_seed(42)

#%% model + oracle lens. The lens is a LoRA on the model's own weights, loaded before the Bridge boots (peft cannot inject into the Bridge's wrappers) and disabled by default, so sampling and caching below are the plain model's; olens_show enables it around a read

MODEL_ID = "Qwen/Qwen3.6-27B"
device = t.device("cuda")
hf = load_hf_model(MODEL_ID, device_map="auto")
peft, olens_contract = load_olens(hf)  # agu18dec/olens_and_ar: reads layers 20..60 step 4, the output of block L
peft.base_model.disable_adapter_layers()
model = boot_bridge(hf)
tokenizer = model.tokenizer

olens_layers = [28, 36, 44, 52, 60]  # each layer is one batched generation over all positions read, so fewer layers is faster

def olens_show(cache: dict, pos: list[int], toks: Tensor, n: int = 2, **kw) -> list[dict]:
    """olens_readout of a hook_resid_post cache at `pos`, the lens enabled for the read only: a tab per layer and per position, n samples each; its records (layer, pos, i, j, h, text, ids, stopped, bullets). kw goes to olens_readout (ctx, do_sample=False for greedy, ...)."""
    peft.base_model.enable_adapter_layers()
    out = olens_readout(cache, olens_layers, pos, peft, tokenizer, olens_contract, n=n, input_src=toks, **kw)
    peft.base_model.disable_adapter_layers()
    return out

#%% toy example: the model picks one of two words at random. sample its answer, cache the whole sequence, then read the lens from the closing think token on

run_toy = True
if run_toy:
    word1, word2 = "forty", "bear"
    toy_conv = [{"role": "user", "content": f"Randomly choose to respond with either the word {word1} or the word {word2}. Respond with only that one word."}]
    toy_toks = apply_chat_template(tokenizer, toy_conv, enable_thinking=False)[0].to(device)  # ends with '</think>', '\n\n'
    toy_toks = t.cat([toy_toks, t.tensor([list(stream_toks(model, toy_toks, new_toks=16))], device=device)], dim=1)  # plus the sampled answer
    show_toks(toy_toks, tokenizer)
    toy_logits, toy_cache = model.run_with_cache(toy_toks, names_filter=lambda n: n.endswith("hook_resid_post"))
    show_logits(toy_toks, logits=toy_logits, tokenizer=tokenizer, pos=[-2], title="answer distribution")  # the '\n\n' after </think> predicts the answer's first token
    think_pos = toy_toks[0].tolist().index(tokenizer.convert_tokens_to_ids("</think>"))
    toy_olens = olens_show(toy_cache, list(range(think_pos, toy_toks.shape[1])), toy_toks)

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
    _, dating_cache = model.run_with_cache(dating_toks, names_filter=lambda n: n.endswith("hook_resid_post"))

    targ_pos = 47
    ctx_around = 8
    targ_positions = list(range(targ_pos - ctx_around//2, targ_pos + ctx_around//2))
    dating_olens = olens_show(dating_cache, targ_positions, dating_toks, ctx=4096)

#%%
