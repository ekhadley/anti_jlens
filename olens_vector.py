#!./.venv/bin/python
#%% a saved direction from the ortho project through the oracle lens. The vector is [n_layers, d_model] with row L at resid_post L, which is the lens's "layer L", so row L goes in as is (the j-lens readout in ortho shifts it to resid_pre layout; this one does not). The lens replaces the marker's embedding with alpha x the unit vector, so the rows' norms (0.003 to 0.1) do not matter, only their directions. Both signs are read: the rows are cheat minus clean means, so -v is clean minus cheat
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer
from mechtools import *
from mechtools.colors import *

t.set_grad_enabled(False)
set_seed(42)

VECTOR = Path("/workspace/vectors/Qwen3.6-27B/grader_parity_cheat_vs_clean.safetensors")  # ortho data/vectors/Qwen3.6-27B, uploaded 2026-10-09; the json sidecar is next to it

#%% model + oracle lens. No Bridge: nothing is cached here, the lens only generates. Skip this cell in a kernel where olens.py's model cell already ran (it defines the same peft, tokenizer and olens_contract)

MODEL_ID = "Qwen/Qwen3.6-27B"
device = t.device("cuda")
hf = load_hf_model(MODEL_ID, device_map="auto")
peft, olens_contract = load_olens(hf)  # agu18dec/olens_and_ar: layers 20..60 step 4
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

#%% read every lens layer, both signs, n samples each: one batched generation per layer over the 2 x n rows

with safe_open(VECTOR, "pt") as f:
    v = f.get_tensor("v")  # [64, 5120] float32
n = 4
signs = {"cheat - clean": 1.0, "clean - cheat": -1.0}
peft.base_model.enable_adapter_layers()  # a no-op right after load_olens; needed in olens.py's kernel, where the lens is disabled between reads
olens_vec = {}
for L in olens_contract["layers"]:
    rows = olens_read(peft, tokenizer, t.stack([s * v[L] for s in signs.values()]).to(device), L, olens_contract, n=n)  # [2, n] lists of bullets, sign-major
    for label, items in zip(signs, rows):
        olens_vec[(label, L)] = items
        print(f"{purple}L{L}{endc} {cyan}{label}{endc}")
        for bullets in items: print(f"    {gray}-{endc} " + f" {gray}|{endc} ".join(bullets))
peft.base_model.disable_adapter_layers()
show_table(["layer", "sign"] + [f"sample {j}" for j in range(1, n + 1)], [(f"L{L}", label, *["; ".join(b) for b in items]) for (label, L), items in olens_vec.items()], title=f"oracle lens on {VECTOR.stem}")

#%%
