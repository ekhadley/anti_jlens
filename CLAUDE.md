# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Early-stage interpretability research on the j-lens (a learned per-layer linear lens: `J [n_layers, d_model, d_model]` maps a residual into final-layer space, then `ln_final` + `unembed` give logits). Current work is in `toy.py`: sample a model's answer to a "pick one of two words" prompt, cache `hook_resid_pre`, then compare the j-lens cluster readout on the original residuals against residuals with the direction of the sum of all later positions' residuals projected out.

## Environment

- uv, Python 3.12. `uv sync` to set up; run with `.venv/bin/python` (also the shebang in `toy.py`).
- No tests, linter, or build step.
- Needs CUDA. The model is `Qwen/Qwen3.6-27B` in bf16 with `device_map="auto"`, so loading is slow - don't run `toy.py` end to end casually.

## Code structure

- `toy.py` is a `#%%` cell script meant to be run interactively (IPython / Jupyter kernel). Each cell after model loading is gated by a `run_x = True` / `modify_x = True` / `show_x = True` flag. The model, lens, and caches persist across cells, so later cells assume earlier ones ran.
- Almost all functionality comes from `mechtools`, the user's own library, installed as a git dependency (`ssh://git@github.com/ekhadley/mechtools`) and star-imported (`from mechtools import *`). It provides `t` (torch), `TransformerBridge`, `load_jlens`, `cluster_vocab`, `jlens_cluster_readout`, `apply_chat_template`, `stream_toks`, `show_toks`, `show_logits`, etc. To see what a helper does, read the installed source in `.venv/lib/python3.12/site-packages/mechtools/` (lens code is in `lens.py`). To pick up new mechtools commits: `uv lock --upgrade-package mechtools && uv sync`.
- Importing mechtools enables IPython autoreload and loads the nearest `.env` walking up from the cwd.
- Lenses are fetched with `hf_hub_download` from the HF repo `camilablank/workspace-lenses`; the path passed to `load_jlens` (e.g. `qwen3.6-27b/j-lens/lens.pt`) is relative to that repo, not this directory.
- `jlens_cluster_readout` only reads `cache[f"blocks.{layer}.hook_resid_pre"][0, pos]`, so a modified cache can be a plain dict holding just those keys for the lens layers (as `toy_mod_cache` does).
