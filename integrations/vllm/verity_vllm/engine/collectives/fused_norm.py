"""`fused_allreduce_rms_norm` as one recorded TP collective, for the Match recorder (`rank_worker.tp2_coll_install`).

DeepSeek-V3.2's decoder layer (GLM-5.3 on CUDA, vLLM d9105ea80: `vllm/models/deepseek_v32/nvidia/model.py`) runs its o_proj and MoE
with ``reduce_results=False`` and reduces each output in the RMSNorm after it: `fused_allreduce_rms_norm(hidden_states, residual,
norm)` (`vllm/models/common/ops/fused_allreduce_rms_norm.py`).  Within FlashInfer's token budget the call is one fused kernel
(all-reduce, residual add, RMSNorm) that writes the new residual into ``hidden_states`` in place and never calls ``norm``;
past it, the module's own `tensor_model_parallel_all_reduce` and then ``norm``.  Neither path goes through a name the linear,
embedding or MoE wraps patch, so without this the recorder sees none of a layer's all-reduces.

Each call is one record of kind `KIND` at the norm's parent module (the decoder layer, or the model for its final norm: the
Build's `tp_links` site), in call order.  Its ``part`` is ``stack([hidden_states, residual])`` copied before the call runs, its
``out`` the new residual, and ``path`` is `FUSED` (``norm`` did not run) or `UNFUSED`.  `expected_residual` is what the fused
path's new residual must be (`program.registry.tp_allreduce`, MNNVL); the Match checks it per element (`pipeline/tp/match.py`).
The normed half is the Commit's replay's.  Torch is imported lazily.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterator

FAR_MODULE = "vllm.models.common.ops.fused_allreduce_rms_norm"
FAR_NAME = "fused_allreduce_rms_norm"
KIND = "fused_allreduce_rms_norm"
FUSED, UNFUSED = "F", "N"
F32_MIN_NORMAL = 2.0 ** -126


def targets(model: Any) -> list[tuple[Any, str]]:
    """Every binding of the function to patch, as (python module, name): its own module and each module of the model's classes
    that imported it.  Empty when vLLM has none, or no class of ``model`` calls it."""
    home = sys.modules.get(FAR_MODULE)
    fn = getattr(home, FAR_NAME, None) if home is not None else None
    if fn is None:
        return []
    users = [sys.modules[n] for n in sorted({type(m).__module__ for m in model.modules()}) if n in sys.modules and n != FAR_MODULE]
    users = [m for m in users if getattr(m, FAR_NAME, None) is fn]
    return [(home, FAR_NAME), *((m, FAR_NAME) for m in users)] if users else []


@contextmanager
def silenced(rec: Any) -> Iterator[None]:
    """The aten observer's recorder ``rec`` (None: none runs) off for the block, so the recorder's own copies stay out of the log."""
    was = rec.active if rec is not None else None
    if rec is not None:
        rec.active = False
    try:
        yield
    finally:
        if rec is not None:
            rec.active = was


def recording(orig: Callable, model: Any, st: dict, record: Callable[[Callable, str], Callable], quiet: Callable) -> Callable:
    """``orig`` wrapped so each call is one `KIND` record: ``record(call, KIND)`` is the recorder's own wrap (it runs ``call`` on
    the part and keeps part, out and digests); ``quiet()`` hides this wrapper's copies from the run's aten observer."""
    import torch

    names = {id(m): n for n, m in model.named_modules()}
    ran: dict[int, bool] = {}

    def mark(m: Any, args: Any) -> None:
        ran[id(m)] = True

    def fused_allreduce_rms_norm(hidden_states, residual, norm, *a, **k):
        if id(norm) not in ran:
            st["handles"].append(norm.register_forward_pre_hook(mark))
        ran[id(norm)] = False
        got: dict[str, Any] = {}

        def call(_part):
            got["out"] = orig(hidden_states, residual, norm, *a, **k)
            return got["out"][1]

        with quiet():
            part = torch.stack([hidden_states.detach().clone(), residual.detach().clone()])
        prev, st["cur"] = st["cur"], {"name": names.get(id(norm), "").rpartition(".")[0] or None, "cls": type(norm).__name__}
        try:
            record(call, KIND)(part)
        finally:
            st["cur"] = prev
        st["records"][-1]["path"] = UNFUSED if ran[id(norm)] else FUSED
        return got["out"]

    return fused_allreduce_rms_norm


def expected_residual(part: Any, order: Any = None) -> Any:
    """The fused path's new residual from every rank's ``stack([partial, residual])`` (rank order): the MNNVL all-reduce, an f32 sum
    from +0 over ranks 0, 1, ..., WORLD-1 with -0 payloads written as +0 and FADD.FTZ, rounded to bf16, then ``__hadd`` with the
    residual (`tp_allreduce.AllReduceRmsNormMnnvl_v1`'s residual output).  ``order`` is ignored: MNNVL reads ranks ascending."""
    import torch

    def ftz(x):
        return torch.where(x.abs() < F32_MIN_NORMAL, torch.zeros_like(x).copysign(x), x)

    acc = torch.zeros(part[0][0].shape, dtype=torch.float32)
    for p in part:
        f = p[0].float()
        acc = ftz(acc + ftz(torch.where(f == 0, torch.zeros_like(f), f)))
    return (acc.to(torch.bfloat16).float() + part[0][1].float()).to(torch.bfloat16)
