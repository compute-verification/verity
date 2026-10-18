"""The block-FP8 linears' input quantization on one rank, as an acquisition source: `<linear module>/x_q` (e4m3 [M, K]) and
`<linear module>/x_s` (f32 [M, K/128], row-major), the q and s members of `DeepGemmFp8Linear_v1` (`registry.deepgemm`).

vLLM's `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` quantizes a linear's input itself only on DeepGEMM's branch (M >= 32, or
every M under `VLLM_BATCH_INVARIANT`): `per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=True)` ->
`torch.ops._C.per_token_group_fp8_quant(x, x_q, x_s, 128, ...)`, then `fp8_gemm_nt((x_q, x_s), ...)`.  FlashInfer's branch (M < 32)
quantizes inside its GEMM and nothing leaves the op.  A forward pre-hook marks the linear module that runs; inside it, during a
committed step, the wrapper of the `_C` op binds what the launch wrote, the words DeepGEMM reads: x_q as it is and x_s, a
column-major view, as its row-major copy, both copied at once (x_q is freed when the linear returns).  The source evaluates nothing:
the replay holds both to `Fp8GroupQuantUe8m0_v1` of the committed input and y to the GEMM of them.  One launch per linear call;
a second inside one call is a collector error.
"""
from __future__ import annotations

from typing import Any, Mapping

from verity_vllm.acquire.sources.allreduce_norm import _flush
from verity_vllm.engine import hooks

GROUP_QUANT_OP = "per_token_group_fp8_quant"
GROUP = 128
MEMBERS = ("x_q", "x_s")
FAMILY = "DeepGemmFp8Linear_v1"


def sites(model: Any) -> list[str]:
    """The block-FP8 linear modules: a quant method and a `weight_scale_inv` (the fused-MoE experts hold `w13_weight_scale_inv`)."""
    return [n for n, m in model.named_modules() if n and hasattr(m, "quant_method") and hasattr(m, "weight_scale_inv")]


def manifest_names_it(manifest: Mapping[str, Any] | None) -> bool:
    """Whether a rank's manifest rows name a `DeepGemmFp8Linear_v1` Call."""
    return any(str(r.get("spec") or "").split("{", 1)[0] == FAMILY for r in (manifest or {}).get("identities") or ())


class LinearQuantSource:
    short_name = "linear_quant"

    def __init__(self, committer: Any, model: Any, names: list[str] | None = None) -> None:
        import torch

        self.torch, self.c, self.model = torch, committer, model
        self.sites = list(sites(model) if names is None else names)
        if not self.sites:
            raise ValueError("linear quant: the model has no block-FP8 linear module")
        self._hooks = hooks.Hooks()
        self._handles: list[Any] = []
        self._scope: list[list[Any]] = []
        self.stats: dict[str, Any] = {"calls": 0, "bytes": 0, "rows": 0, "sites": len(self.sites), "errors": 0}
        self.active = False

    def install(self) -> None:
        mods = dict(self.model.named_modules())
        for op in self.sites:
            self._handles.append(mods[op].register_forward_pre_hook(self._enter(op)))
            self._handles.append(mods[op].register_forward_hook(self._leave(op), always_call=True))
        ops_c = self.torch.ops._C
        if not hasattr(ops_c, GROUP_QUANT_OP):
            raise RuntimeError(f"linear quant: torch.ops._C has no {GROUP_QUANT_OP}")
        self._hooks.patch(ops_c, GROUP_QUANT_OP, self.wrap(getattr(ops_c, GROUP_QUANT_OP)))
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._hooks.uninstall()
        self._scope.clear()
        self.active = False

    def on_step_begin(self) -> None:
        self._scope.clear()

    def _enter(self, op: str) -> Any:
        def pre(_module: Any, _args: tuple) -> None:
            self._scope.append([op, 0])

        return pre

    def _leave(self, op: str) -> Any:
        def post(_module: Any, _args: tuple, _out: Any) -> None:
            if self._scope and self._scope[-1][0] == op:
                self._scope.pop()

        return post

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def wrap(self, orig: Any) -> Any:
        me = self

        def per_token_group_fp8_quant(input: Any, output_q: Any, output_s: Any, group_size: Any, *a: Any, **k: Any) -> Any:
            r = orig(input, output_q, output_s, group_size, *a, **k)
            job = getattr(me.c, "_cur", None)
            if job is not None and me._scope:
                me.bind(me._scope[-1], output_q, output_s, int(group_size), job)
            return r

        return per_token_group_fp8_quant

    def bind(self, frame: list[Any], x_q: Any, x_s: Any, group_size: int, job: Any) -> None:
        op = frame[0]
        frame[1] += 1
        if frame[1] > 1:
            return self.error(f"linear quant: {op} launched {GROUP_QUANT_OP} {frame[1]} times in one call")
        M, K = (tuple(x_q.shape) + (0, 0))[:2]
        if group_size != GROUP or x_q.dim() != 2 or tuple(x_s.shape) != (M, K // GROUP):
            return self.error(f"linear quant: {op} quantized {tuple(x_q.shape)} into scales {tuple(x_s.shape)} at group {group_size}; "
                              f"DeepGemmFp8Linear_v1 reads [M, K] e4m3 and [M, K/{GROUP}] f32")
        s = x_s if x_s.is_contiguous() else x_s.contiguous()
        self.c._acquire(f"{op}/{MEMBERS[0]}", x_q, job)
        self.c._acquire(f"{op}/{MEMBERS[1]}", s, job)
        _flush(self.c)
        self.stats["calls"] += 1
        self.stats["rows"] += int(M)
        self.stats["bytes"] += int(x_q.numel()) * int(x_q.element_size()) + int(s.numel()) * 4

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        mods = dict(model.named_modules())
        per_token = sum(int(getattr(mods[n], "input_size_per_partition", 0) or 0) * (1 + 4 / GROUP) for n in self.sites if n in mods)
        return {"per_token_bytes": int(per_token), "per_seq_bytes": 0, "tensors_per_step": 2 * len(self.sites)}

    def declaration_classes(self) -> list[dict]:
        return []


def attach(com: Any, model: Any, manifest: Mapping[str, Any] | None) -> LinearQuantSource | None:
    """Attach the source to `com` when the rank's manifest rows name a `DeepGemmFp8Linear_v1` Call; None otherwise."""
    if not manifest_names_it(manifest):
        return None
    src = LinearQuantSource(com, model)
    com.attach_source(src)
    return src


__all__ = ["LinearQuantSource", "sites", "manifest_names_it", "attach", "MEMBERS", "FAMILY", "GROUP_QUANT_OP"]
