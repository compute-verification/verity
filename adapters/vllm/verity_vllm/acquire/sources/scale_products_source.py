"""The blockwise FP8 GEMM's scale products (`query.scale_products`) as an acquisition source: per linear module and step, the committed
products of every token row, `<linear module>/scale_products` (f32 [rows, N/G * K/G], weight block nb major, k block kb minor), in the
same step stream and run root as the module's other values.  Attached whenever the capture plan names `scale_products` classes
(a Program built under `fp8_scale_construction = shared-per-block`), refused on a plan that names none.

The sites are the capture plan's `scale_products` classes (`program.predict.capture_plan.of_build`, taken as data): the linear module
and the products' width, N/G * K/G.  A forward pre-hook marks the site module that runs; inside it, during a
committed step, the wrapper of `torch.ops._C.cutlass_scaled_mm` reads the launch's `a_scales` (the [M, K/G] group scales
`per_token_group_fp8_quant` wrote, x_s) and `b_scales` (the transposed view of the module's [N/G, K/G] `weight_scale_inv`, w_s), computes
`F32Mul_v1(x_s[m][kb], w_s[nb][kb])` on the host through the circuit (`numerics.fp8_twins.scale_products`), acquires the words on the launch's
device, then runs the installed kernel unchanged.  No kernel changes.  The block geometry is the launch's own (`b_scales` is
[K/G, N/G]); a launch whose scales are not one geometry, or whose products are not the plan's width, is a collector error.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from verity_vllm.acquire.sources.call_boundary_plan import SCALE_PRODUCTS
from verity_vllm.definitions.numerics import fp8_twins
from verity_vllm.engine import hooks
from verity_vllm.log import init_logger
from verity_vllm.query import scale_products as SP

logger = init_logger(__name__)


def products(torch: Any, a_scales: Any, b_scales: Any) -> Any:
    """f32 [M, N/G * K/G] on `a_scales`' device: `fp8_twins.scale_products` of the launch's [M, K/G] activation scales and the [N/G, K/G]
    block scales `b_scales` is the transposed view of, carried as 32-bit words end to end."""
    xs = a_scales.contiguous().view(torch.int32).cpu().numpy().view(np.uint32)
    ws = b_scales.t().contiguous().view(torch.int32).cpu().numpy().view(np.uint32)
    out = fp8_twins.scale_products(xs, ws).reshape(xs.shape[0], -1)
    return torch.from_numpy(out.view(np.int32)).to(a_scales.device).view(torch.float32)


class ScaleProductsSource:
    short_name = "scale_products"

    def __init__(self, committer: Any, model: Any, sites: Mapping[str, int]) -> None:
        """`sites`: linear module op path -> the products' width, N/G * K/G (`sites`)."""
        import torch

        if not sites:
            raise ValueError("scale products: no linear module to tap")
        self.torch, self.c, self.model = torch, committer, model
        self.sites = {str(k): int(v) for k, v in sites.items()}
        self._hooks = hooks.Hooks()
        self._handles: list[Any] = []
        self._scope: list[str] = []
        self._held: list[Any] = []
        self.stats: dict[str, Any] = {"tensors": 0, "bytes": 0, "rows": 0, "sites": len(self.sites), "errors": 0}
        self.active = False

    def install(self) -> None:
        mods = dict(self.model.named_modules())
        missing = sorted(set(self.sites) - set(mods))
        if missing:
            raise RuntimeError(f"scale products: the capture plan's linear modules are not modules of the model: {missing[:4]}")
        for op in self.sites:
            self._handles.append(mods[op].register_forward_pre_hook(self._enter(op)))
            self._handles.append(mods[op].register_forward_hook(self._leave(op), always_call=True))
        self._hooks.patch(self.torch.ops._C, "cutlass_scaled_mm", self._wrap(self.torch.ops._C.cutlass_scaled_mm))
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._hooks.uninstall()
        self._scope.clear()
        self._held.clear()
        self.active = False

    def on_step_begin(self) -> None:
        self._held.clear()

    def _enter(self, op: str) -> Any:
        def pre(_module: Any, _args: tuple) -> None:
            self._scope.append(op)

        return pre

    def _leave(self, op: str) -> Any:
        def post(_module: Any, _args: tuple, _out: Any) -> None:
            if self._scope and self._scope[-1] == op:
                self._scope.pop()

        return post

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def _wrap(self, orig: Any) -> Any:
        me, torch = self, self.torch

        def cutlass_scaled_mm(out: Any, a: Any, b: Any, a_scales: Any, b_scales: Any, bias: Any = None) -> None:
            job = me.c._cur
            if job is not None and me._scope:
                op = me._scope[-1]
                width = me.sites[op]
                if a_scales.dim() != 2 or b_scales.dim() != 2 or int(a_scales.shape[1]) != int(b_scales.shape[0]) or \
                        int(b_scales.shape[0]) * int(b_scales.shape[1]) != width:
                    me.error(f"scale products: {op} launched cutlass_scaled_mm with a_scales {tuple(a_scales.shape)}, b_scales "
                             f"{tuple(b_scales.shape)}; the capture plan's products are {width} wide ([M, K/G] x [K/G, N/G])")
                else:
                    t0 = time.perf_counter()
                    v = products(torch, a_scales, b_scales)
                    me._held.append(v)
                    me.c._acquire(f"{op}/{SP.MEMBER}", v, job)
                    n = int(v.numel()) * 4
                    me.stats["tensors"] += 1
                    me.stats["bytes"] += n
                    me.stats["rows"] += int(v.shape[0])
                    spans = getattr(me.c, "spans", None)
                    if spans is not None:
                        spans.add("acquisition", time.perf_counter() - t0, step=job.step, nbytes=n)
            return orig(out, a, b, a_scales, b_scales, bias)

        return cutlass_scaled_mm

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        per_token = sum(4 * w for w in self.sites.values())
        return {"per_token_bytes": per_token, "per_seq_bytes": 0, "tensors_per_step": len(self.sites)}

    def declaration_classes(self) -> list[dict]:
        return [{"class": SCALE_PRODUCTS, "how": "bound-as-leaf", "rule": None, "count": self.stats["tensors"], "bytes": self.stats["bytes"],
                 "tensors": self.stats["tensors"], "rows": self.stats["rows"], "errors": self.stats["errors"],
                 "evidence": (f"scale-products source: `<linear module>/{SP.MEMBER}` of {len(self.sites)} sites, F32Mul_v1 of the launch's "
                              "activation scales and the module's weight_scale_inv through the IR (numerics.fp8_twins.scale_products; no kernel change)")}]


def sites(plan: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Linear module op path -> the products' width, of every `scale_products` class of the capture plan.  ValueError, naming the
    class, for a member other than `scale_products` or a module named at two widths."""
    out: dict[str, int] = {}
    for c in plan:
        if c["capture"] != SCALE_PRODUCTS:
            continue
        op, w = str(c["op_path"]), int(c["width"])
        if c["member"] != SP.MEMBER:
            raise ValueError(f"scale products: {op}/{c['member']} is a scale_products class and not the member {SP.MEMBER}")
        if out.setdefault(op, w) != w:
            raise ValueError(f"scale products: {op} is named at two widths ({out[op]}, {w})")
    return out


def attach(com: Any, enabled: bool, plan: Sequence[Mapping[str, Any]], required_classes: tuple[str, ...], *,
           explicit_classes: bool = False, model: Any = None) -> tuple[tuple[str, ...], ScaleProductsSource | None]:
    """Attach the source to `com` when `enabled`; the capture plan must name `scale_products` classes, each this source's, else the
    Commit is refused (exit 3).  Returns the required classes with `scale_products` added unless they were given explicitly."""
    if not enabled:
        return required_classes, None
    try:
        at = sites(plan)
    except ValueError as e:
        logger.error("[delta] REFUSE: %s", e)
        raise SystemExit(3) from None
    if not at:
        logger.error("[delta] REFUSE: the scale-products source but the capture plan names no scale_products class (a Program without "
                     "ScaledMmFp8BlockSharedScale Calls)")
        raise SystemExit(3)
    src = ScaleProductsSource(com, model if model is not None else getattr(com, "model", None), at)
    com.attach_source(src)
    logger.info("[delta] scale-products source attached: %d sites -> committer %s", len(at), com.name)
    if not explicit_classes and SCALE_PRODUCTS not in required_classes:
        return (*required_classes, SCALE_PRODUCTS), src
    return required_classes, src


__all__ = ["ScaleProductsSource", "products", "sites", "attach"]
