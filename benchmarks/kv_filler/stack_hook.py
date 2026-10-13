"""The encoded stack (`benchmarks/vllm_encoded/stack.py`, its arm `stack` or the plain `pstack`) in this directory's
engines (`serve.py`, `step_profile.py`): installed at the end of vLLM's `load_model`, as `vllm_encoded/serve.py` installs
it, with its gates (every linear, the lm_head's decode, a step's receipts against the driver's) before the engine
profiles memory. The stack captures its own decode sizes (`CAPTURE`, up to 128 rows); a larger batch runs eager, its
linears decoded into the stack's ephemeral scratch.

Both directories have a `bench.py` and a `serve.py`: vllm_encoded's serve.py loads as `vllm_encoded_serve`, and the
stack's `bench` (codec_tensor's) only for the stack's own imports; this directory's modules keep their names.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

VE = Path(__file__).resolve().parent.parent / "vllm_encoded"


def install(arm: str, k: int, wt_lib: str, reader_lib: str, paged_lib: str, select_trials: int = 5, new_stream=None):
    """Hooks vLLM's load_model to install the arm's stack; returns (the stack module, the install, its capture sizes).
    `new_stream` makes the stack's side streams (`stack.new_stream`), on an SM partition's step context."""
    ours = {m: sys.modules.pop(m) for m in ("bench", "serve") if m in sys.modules}
    sys.path.insert(0, str(VE))
    try:
        spec = importlib.util.spec_from_file_location("vllm_encoded_serve", VE / "serve.py")
        vs = importlib.util.module_from_spec(spec)
        sys.modules["vllm_encoded_serve"] = vs
        spec.loader.exec_module(vs)
        import stack as S
    finally:
        sys.modules.pop("bench", None)
        sys.modules.update(ours)
    if new_stream is not None:
        S.new_stream = new_stream
    from verity_vllm.engine.hooks import Hooks

    args = argparse.Namespace(arm=arm, k=k, wt_lib=wt_lib, reader_lib=reader_lib, paged_lib=paged_lib,
                              select_trials=select_trials)
    inst = vs.Install(args)
    vs.hook_runner(Hooks(tag="ra-vllm-encoded-stack/kv-filler"), inst)
    return S, inst, list(S.CAPTURE)


def weight_bytes(inst, model) -> int:
    """The bytes a decode step reads of weights: the stack's linears' and lm_head's stores (their words, or the plain
    fragments and rows), and the model's other parameters but the embedding."""
    st = inst.stack
    rest = sum(p.numel() * p.element_size() for n, p in model.named_parameters() if "embed_tokens" not in n)
    return rest + sum(lin.nbytes for lin in st.linears)
