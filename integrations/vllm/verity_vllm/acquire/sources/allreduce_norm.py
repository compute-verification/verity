"""fused_allreduce_rms_norm's calls on one TP rank, bound for the TP partial source (`partial_source.TPPartialSource`).

Some models leave a row-parallel output un-reduced and fuse its all-reduce into the RMSNorm after it: vLLM's
`fused_allreduce_rms_norm(hidden_states, residual, norm)` (vllm/models/common/ops/fused_allreduce_rms_norm.py, imported by
DeepSeek-V3.2's model module and others).  With more than one rank each call takes one of two paths: FlashInfer's fused
all-reduce + residual add + RMSNorm, which writes the new residual into `hidden_states` in place, returns
`(norm_out, hidden_states)` and never calls `norm`; or `tensor_model_parallel_all_reduce` from that module, then
`norm(reduced, residual)`.  Neither goes through a linear, so the partial source's collective wraps never see the partial.

On every call this binds the rank's partial `hidden_states` as `<norm>/part_rank<r>` (the partial source's name for a collective
input, attributed to the norm the call passes) and, under the gather-mode committer, copies it before the call runs.  When `norm`
did not run (the fused path) it binds the returns as `<norm>/0` and `<norm>/1`, the names the module hook gives them on the
other path, so a step binds the same names whichever path it took, and copies them at once.  The path of each call is recorded
per step, in call order (`F` fused, `N` all-reduce then the norm module).

The sites are the norms the first step passes: which norms a model hands over depends on the layer (DeepSeek-V3.2's layer 0
normalizes the embedding directly) and cannot be read before a forward runs.  When the model's classes call the function and
no call has been seen, the sites are one placeholder that no step binds, so `identity_check` fails every step.
"""
from __future__ import annotations

import sys
from typing import Any

FAR_MODULE = "vllm.models.common.ops.fused_allreduce_rms_norm"
FAR_NAME = "fused_allreduce_rms_norm"
NO_CALL_SITE = "<fused_allreduce_rms_norm: no call seen>"
FUSED, UNFUSED = "F", "N"
PATH_STEPS_KEPT = 256


def targets(model: Any) -> tuple[list[tuple[Any, str]], bool]:
    """Every binding of the function to patch, as (python module, name): its own module and each module of the model's classes
    that imported it; and whether any of the model's classes did.  A model that imports it has loaded its module already."""
    home = sys.modules.get(FAR_MODULE)
    fn = getattr(home, FAR_NAME, None) if home is not None else None
    if fn is None:
        return [], False
    users = [sys.modules[n] for n in sorted({type(m).__module__ for m in model.modules()}) if n in sys.modules and n != FAR_MODULE]
    users = [m for m in users if getattr(m, FAR_NAME, None) is fn]
    return [(home, FAR_NAME), *((m, FAR_NAME) for m in users)], bool(users)


def _flush(c: Any) -> None:
    nc = getattr(c, "_nc", None)
    if getattr(c, "gather", False) and nc is not None and hasattr(nc, "flush"):
        nc.flush()


class AllReduceNorm:
    def __init__(self, src: Any, model: Any) -> None:
        self.src = src
        self.names = {id(m): n for n, m in model.named_modules()}
        self.targets, self.used = targets(model)
        users = {m.__name__ for m, _a in self.targets[1:]}
        mods = dict(model.named_modules())
        # an upper bound of the sites for the arena: the norms (a weight and a variance_epsilon) held by a module whose class calls it
        self.candidates = [n for n, m in mods.items() if users and n and hasattr(m, "variance_epsilon") and hasattr(m, "weight")
                           and type(mods.get(n.rpartition(".")[0])).__module__ in users]
        self._bytes_per_token = sum(int(mods[n].weight.numel()) * int(mods[n].weight.element_size()) for n in self.candidates)
        self.first: list[str] | None = None
        self._calls: list[tuple[str, str]] = []
        self._step: int | None = None
        self.paths: list[tuple[int, str]] = []
        self._ran = False
        self._hooked: dict[int, Any] = {}
        self.stats: dict[str, int] = {"calls": 0, FUSED: 0, UNFUSED: 0, "returns_bound": 0, "unattributed": 0}

    # -- expected population ------------------------------------------------------------------------------------------------
    def site_names(self) -> list[str]:
        if not self.used or self.src.world <= 1:
            return []
        first = self.first if self.first is not None else sorted({n for n, _p in self._calls})
        return first or [NO_CALL_SITE]

    def extra_step_bytes(self) -> tuple[int, int]:
        """(bytes per token, tensors per step) of the partials, over the candidate sites; the fused path's returns replace the
        norm module's outputs, which the module table already counts."""
        if not self.used or self.src.world <= 1:
            return 0, 0
        return self._bytes_per_token, len(self.candidates)

    # -- install / steps ------------------------------------------------------------------------------------------------------
    def uninstall(self) -> None:
        for h in self._hooked.values():
            h.remove()
        self._hooked = {}

    def close_step(self) -> None:
        if self._calls:
            if self.first is None:
                self.first = sorted({n for n, _p in self._calls})
            if len(self.paths) < PATH_STEPS_KEPT:
                self.paths.append((int(self._step or 0), "".join(p for _n, p in self._calls)))
        self._calls = []
        self._step = None

    def _mark(self, mod: Any, args: Any) -> None:
        self._ran = True

    def wrap(self, orig):
        me = self

        def fused_allreduce_rms_norm(hidden_states, residual, norm, *a, **k):
            c = me.src.c
            job = getattr(c, "_cur", None)
            if job is None or me.src.world <= 1:
                return orig(hidden_states, residual, norm, *a, **k)
            name = me.names.get(id(norm))
            if name is None:
                me.stats["unattributed"] += 1
                c.errors.append("allreduce norm: fused_allreduce_rms_norm called with a norm that is not a module of the model"[:200])
                c.x03_expected = getattr(c, "x03_expected", 0) + 1
                return orig(hidden_states, residual, norm, *a, **k)
            if id(norm) not in me._hooked:
                me._hooked[id(norm)] = norm.register_forward_pre_hook(me._mark)
            if me._step is None:
                me._step = int(getattr(job, "step", 0))
            prev, me.src.cur = me.src.cur, (name, FAR_NAME)
            try:
                me.src._bind("all_reduce", hidden_states)
            finally:
                me.src.cur = prev
            _flush(c)                                    # the fused path writes the new residual into hidden_states
            me._ran = False
            out = orig(hidden_states, residual, norm, *a, **k)
            path = UNFUSED if me._ran else FUSED
            if path == FUSED:
                me._bind_returns(name, out, job)
            me._calls.append((name, path))
            me.stats["calls"] += 1
            me.stats[path] += 1
            return out

        return fused_allreduce_rms_norm

    def _bind_returns(self, name: str, out: Any, job: Any) -> None:
        """The fused path's (normed, new residual) under the module hook's names (`_make_hook`, with its row-4 residual copy when
        the committer makes one), copied at once: the new residual is the partial's buffer."""
        c = self.src.c
        for k, o in enumerate(out):
            if hasattr(o, "device"):
                c._acquire(f"{name}/{k}", o, job)
                self.stats["returns_bound"] += 1
        if getattr(c, "regen_row4", False) and len(out) == 2 and str(getattr(out[1], "dtype", "")).endswith("bfloat16"):
            c._acquire(f"{name}/row4_f32", out[1].float(), job)
        _flush(c)

    def record(self) -> dict[str, Any]:
        return {"targets": [m.__name__ for m, _a in self.targets], "model_calls_it": self.used, "sites": self.site_names(),
                "candidates": len(self.candidates), "stats": dict(self.stats), "paths_by_step": self.paths[:64],
                "rule": (f"one <norm>/part_rank<r> per site per step, copied before the call; on the fused path ({FUSED}) the returns as "
                         f"<norm>/0, <norm>/1; on the other ({UNFUSED}) the norm module runs and its hook binds them")}


__all__ = ["AllReduceNorm", "targets", "FAR_MODULE", "FAR_NAME", "NO_CALL_SITE", "FUSED", "UNFUSED"]
