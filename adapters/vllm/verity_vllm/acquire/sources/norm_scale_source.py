"""The norm scales (`query.norm_scales`) as an acquisition source: one f32 per norm Call per row, committed as `<norm module>/norm_scale` in
the same step stream and run root as the module's outputs.  On by default (commit_delta `--norm-tap-so`, passed when the row's manifest
names norm scales); NORM_TAP=0 leaves it off, for A/B only.

The modules are the required manifest's `norm_scales` op paths.  A forward pre-hook marks the one that runs (the scale tensor takes its
input's leading shape: tokens, or tokens x heads for a q/k norm) and a forward hook unmarks it.  Inside it the scale is taken where the
norm computes it (`query.norm_scales.site_of`):

* `cuda` (`RMSNormFusedCuda_v2`): `torch.ops._C.fused_add_rms_norm` is replaced by the tapped op of `ops/pod_norm_tap.sh`,
  `verity_norm_tap.fused_add_rms_norm_scale`: the same kernel, launched the same way, whose thread 0 also stores `s_variance`;
* `triton` (`RMSNormTriton_v1`): vLLM's `_rms_norm_kernel` is replaced by the copy in `norm_tap_src.rms_norm_tap`, which also stores
  `inv_rms`, once the installed kernel's source is checked to be the pinned one;
* `aten` (`RsqrtF32_v1`, an ATen-chain norm): the output of `torch.rsqrt` inside the module.

Outside a committed step, or outside the manifest's norm modules, the installed kernels run.  The tapped kernels compute exactly the
installed kernels' outputs and write every row's scale, which equals the circuit's (`properties.norm_tap_exactness`).  Every patch goes
through `engine.hooks` and `uninstall` undoes it.  A norm kernel of another kind inside a norm module is a collector error.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import time
from typing import Any, Mapping

from verity_vllm.acquire.norm_tap_src.build_ext import NAME as TAP_NS
from verity_vllm.acquire.norm_tap_src.build_ext import TAP_VERSION
from verity_vllm.engine import hooks
from verity_vllm.query import norm_scales as NS


def kernel_source_sha256(kernel: Any) -> str:
    """sha256 of a jitted Triton kernel's Python source (`inspect.getsource` of the function it wraps)."""
    return hashlib.sha256(inspect.getsource(getattr(kernel, "fn", kernel)).encode()).hexdigest()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TritonTap:
    """Stands in for vLLM's `_rms_norm_kernel` while the source is installed: `kernel[grid](*args, **kw)` launches the tapped copy with a
    scale buffer inside a manifest norm module during a committed step, else the installed kernel.  Any other attribute is the installed
    kernel's."""

    def __init__(self, source: NormScaleSource, installed: Any, tapped: Any) -> None:
        self.source, self.installed, self.tapped = source, installed, tapped

    def __getitem__(self, grid: Any) -> Any:
        return lambda *args, **kw: self.launch(grid, args, kw)

    def launch(self, grid: Any, args: tuple, kw: dict) -> Any:
        src = self.source
        where = src.where("triton")
        if where is None:
            return self.installed[grid](*args, **kw)
        if len(args) != 7:
            src.error(f"norm scales: _rms_norm_kernel launched with {len(args)} positional arguments inside {where[1]} (the pinned launch has 7)")
            return self.installed[grid](*args, **kw)
        t0 = time.perf_counter()
        job, op, lead = where
        scale = src.torch.empty(int(args[0].shape[0]), dtype=src.torch.float32, device=args[0].device)
        res = self.tapped[grid](args[0], args[1], args[2], scale, *args[3:], **kw)
        src.take(op, lead, scale, job, t0)
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self.installed, name)


class NormScaleSource:
    short_name = "norm_scale"

    def __init__(self, committer: Any, model: Any, modules: Mapping[str, str], so_path: str | None = None, *,
                 words_per_token: int | None = None) -> None:
        """`modules`: op path -> norm Definition family (`query.norm_scales.manifest_modules`); `so_path`: the tap build (needed when a
        module's norm is the fused kernel); `words_per_token`: the manifest's scale words per token row (the arena bound)."""
        import torch

        self.torch, self.c, self.model = torch, committer, model
        self.modules = {op: NS.site_of(fam) for op, fam in modules.items()}
        bad = sorted(op for op, kind in self.modules.items() if kind is None)
        if not self.modules or bad:
            raise ValueError(f"norm scales: no norm module to tap ({len(self.modules)}) or modules of no norm family: {bad[:4]}")
        self.kinds = sorted(set(self.modules.values()))
        self.so_path, self.ns = so_path, None
        if "cuda" in self.kinds:
            if not so_path:
                raise RuntimeError("norm scales: the manifest names fused norms and no tap build was given (--norm-tap-so, ops/pod_norm_tap.sh)")
            torch.ops.load_library(so_path)
            self.ns = getattr(torch.ops, TAP_NS)
            version = int(self.ns.tap_version())
            if version != TAP_VERSION:
                raise RuntimeError(f"norm scales: {so_path} is tap version {version}, this source launches version {TAP_VERSION}")
        self.words_per_token = int(words_per_token or len(self.modules))
        self._hooks = hooks.Hooks()
        self._handles: list[Any] = []
        self._scope: list[tuple[str, tuple[int, ...]]] = []
        self._held: list[Any] = []
        self.stats: dict[str, Any] = {"tensors": 0, "bytes": 0, "by_kind": {k: 0 for k in self.kinds}, "errors": 0, "modules": len(self.modules),
                                      "so": so_path, "so_sha256": _file_sha256(so_path) if self.ns is not None else None, "triton_source": None}
        self.active = False

    # ------------------------------------------------------------------------------------------------------------ install / uninstall
    def install(self) -> None:
        torch = self.torch
        mods = dict(self.model.named_modules())
        missing = sorted(set(self.modules) - set(mods))
        if missing:
            raise RuntimeError(f"norm scales: the manifest's norm modules are not modules of the model: {missing[:4]}")
        if "cuda" in self.kinds or "triton" in self.kinds:
            from vllm import envs
            if not envs.VLLM_BATCH_INVARIANT:
                raise RuntimeError("norm scales: the tapped kernels are vLLM's batch-invariant ones and this engine runs without VLLM_BATCH_INVARIANT")
        for op in self.modules:
            self._handles.append(mods[op].register_forward_pre_hook(self._enter(op)))
            self._handles.append(mods[op].register_forward_hook(self._leave(op), always_call=True))
        if "cuda" in self.kinds:
            self._hooks.patch(torch.ops._C, "fused_add_rms_norm", self._fused(torch.ops._C.fused_add_rms_norm))
        if "triton" in self.kinds:
            from vllm.model_executor.determinism import batch_invariant

            from verity_vllm.acquire.norm_tap_src import rms_norm_tap

            got = kernel_source_sha256(batch_invariant._rms_norm_kernel)
            if got != rms_norm_tap.PINNED_SHA256:
                raise RuntimeError(f"norm scales: vLLM's _rms_norm_kernel source is {got}, not the pinned {rms_norm_tap.PINNED_SHA256}: the tapped "
                                   "copy would not be this kernel")
            self.stats["triton_source"] = got
            self._hooks.patch(batch_invariant, "_rms_norm_kernel", TritonTap(self, batch_invariant._rms_norm_kernel, rms_norm_tap._rms_norm_kernel))
        if "aten" in self.kinds:
            self._hooks.patch(torch, "rsqrt", self._rsqrt(torch.rsqrt))
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

    # ------------------------------------------------------------------------------------------------------------ scope and acquisition
    def _enter(self, op: str) -> Any:
        scope = self._scope

        def pre(_module: Any, args: tuple) -> None:
            x = args[0] if args else None
            scope.append((op, tuple(int(d) for d in x.shape[:-1]) if hasattr(x, "shape") else ()))

        return pre

    def _leave(self, op: str) -> Any:
        scope = self._scope

        def post(_module: Any, _args: tuple, _out: Any) -> None:
            if scope and scope[-1][0] == op:
                scope.pop()

        return post

    def where(self, kind: str, *, strict: bool = True) -> tuple[Any, str, tuple[int, ...]] | None:
        """(the committed step's job, the norm module, its input's leading shape) when a `kind` scale site runs inside a manifest norm
        module during a committed step; None otherwise.  `strict`: a site of another kind inside a norm module is a collector error."""
        job = self.c._cur
        if job is None or not self._scope:
            return None
        op, lead = self._scope[-1]
        if self.modules[op] != kind:
            if strict:
                self.error(f"norm scales: a {kind} norm kernel ran inside {op}, whose manifest norm is taken at {self.modules[op]}")
            return None
        return job, op, lead

    def take(self, op: str, lead: tuple[int, ...], scale: Any, job: Any, t0: float) -> None:
        t = scale.view(lead) if lead and math.prod(lead) == scale.numel() else scale
        self._held.append(t)
        self.c._acquire(f"{op}/{NS.MEMBER}", t, job)
        n = int(t.numel()) * int(t.element_size())
        self.stats["tensors"] += 1
        self.stats["bytes"] += n
        self.stats["by_kind"][self.modules[op]] += 1
        spans = getattr(self.c, "spans", None)
        if spans is not None:
            spans.add("acquisition", time.perf_counter() - t0, step=job.step, nbytes=n)

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def _fused(self, orig: Any) -> Any:
        me, torch = self, self.torch

        def fused_add_rms_norm(input: Any, residual: Any, weight: Any, epsilon: float) -> None:
            where = me.where("cuda")
            if where is None:
                return orig(input, residual, weight, epsilon)
            t0 = time.perf_counter()
            job, op, lead = where
            scale = torch.empty(input.numel() // input.shape[-1], dtype=torch.float32, device=input.device)
            me.ns.fused_add_rms_norm_scale(input, residual, weight, epsilon, scale)
            me.take(op, lead, scale, job, t0)
            return None

        return fused_add_rms_norm

    def _rsqrt(self, orig: Any) -> Any:
        me = self

        def rsqrt(*args: Any, **kw: Any) -> Any:
            out = orig(*args, **kw)
            where = me.where("aten", strict=False)
            if where is not None:
                job, op, lead = where
                me.take(op, lead, out, job, time.perf_counter())
            return out

        return rsqrt

    # ------------------------------------------------------------------------------------------------------------ the committer's view
    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        return {"per_token_bytes": 4 * self.words_per_token, "per_seq_bytes": 0, "tensors_per_step": len(self.modules)}

    def declaration_classes(self) -> list[dict]:
        return [{"class": NS.FAMILY, "how": "bound-as-leaf", "rule": None, "count": self.stats["tensors"], "bytes": self.stats["bytes"],
                 "tensors": self.stats["tensors"], "kinds": dict(self.stats["by_kind"]), "errors": self.stats["errors"],
                 "evidence": (f"norm-scale source: `<norm module>/{NS.MEMBER}` of {len(self.modules)} norm modules (sites {', '.join(self.kinds)}); "
                              f"tap build {self.so_path} (sha256 {self.stats['so_sha256']}), Triton source {self.stats['triton_source']}; outputs "
                              "bit-identical to the installed kernels and every scale == the IR model's: verity_vllm.properties.norm_tap_exactness")}]


def attach(com: Any, so: str | None, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], *,
           explicit_classes: bool = False, model: Any = None) -> tuple[tuple[str, ...], NormScaleSource | None]:
    """Attach the source to `com` when `so` (the tap build) is given; the required manifest must name `norm_scales` identities, else the
    Commit is refused (exit 3).  Returns the required classes with `norm_scales` added unless they were given explicitly."""
    if not so:
        return required_classes, None
    modules = NS.manifest_modules(manifest)
    if not modules:
        print("[delta] REFUSE: --norm-tap-so but the required-value manifest names no norm_scales identity (manifest build --norm-scales)", flush=True)
        raise SystemExit(3)
    src = NormScaleSource(com, model if model is not None else getattr(com, "model", None), modules, so,
                          words_per_token=NS.words_per_token(manifest) or None)
    com.attach_source(src)
    print(f"[delta] norm-scale source attached: {len(modules)} norm modules (sites {', '.join(src.kinds)}), {src.words_per_token} words per "
          f"token, tap {so} -> committer {com.name}", flush=True)
    if not explicit_classes and NS.FAMILY not in required_classes:
        return (*required_classes, NS.FAMILY), src
    return required_classes, src
