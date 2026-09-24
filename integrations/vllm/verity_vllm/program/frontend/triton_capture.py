"""Generic interception of DIRECT Triton launches during `torch.export` tracing.

vLLM's batch-invariant code path launches some Triton kernels straight from Python (`_rms_norm_kernel[grid](...)` in
`vllm/model_executor/determinism/batch_invariant.py`) rather than through a registered torch op. Under FakeTensor tracing
such a launch reads `.data_ptr()` and export refuses ("erroneously tracing into a custom kernel"). This module makes every
such launch an *opaque graph node* carrying exactly what the eager pattern matchers use to identify a kernel: the kernel's
qualified name, its constexpr/scalar arguments, the grid, and the tensor operands in signature order.

    verity_cba::triton_launch(str kernel, Tensor(a!)[] tensors, str meta_json) -> ()

Kernel-agnostic (no kernel is named here); a derive rule per *kernel* then binds the node to a registered Definition
(`rules_vllm.py::TritonRmsNormRule`). Only launches whose tensor arguments are FakeTensors are intercepted; real launches
are untouched.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import Any

import torch

from verity_vllm.program.frontend.rules.triton_iface import triton_launch as op  # the op definition lives with the rules that bind it

try:
    import triton as _triton
    _TRITON_VERSION = getattr(_triton, "__version__", "?")
except Exception:  # noqa: BLE001
    _TRITON_VERSION = None


def _is_fake(t) -> bool:
    return isinstance(t, torch.Tensor) and (t.device.type == "meta" or type(t).__name__ in ("FakeTensor", "FunctionalTensor")
                                             or torch._subclasses.fake_tensor.is_fake(t))


def _jsonable(v: Any):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "__name__"):
        return f"<{type(v).__name__}:{v.__name__}>"
    if type(v).__name__ == "dtype" and isinstance(getattr(v, "name", None), str):  # a `tl.dtype` constexpr (e.g. fused_moe's compute_type)
        return f"<dtype:{v.name}>"
    return f"<{type(v).__name__}>"


class TritonLaunchLog:
    """Every intercepted launch (kernel, scalar args, tensor arg names/shapes) — evidence for the REPORT."""

    def __init__(self) -> None:
        self.launches: list[dict] = []


def _make_run(orig_run, log: TritonLaunchLog):
    def run(self, *args, grid=None, warmup=False, **kwargs):
        if not any(_is_fake(a) for a in list(args) + list(kwargs.values())):
            return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        names = list(getattr(self, "arg_names", []))
        bound: dict[str, Any] = {}
        for i, a in enumerate(args):
            bound[names[i] if i < len(names) else f"arg{i}"] = a
        bound.update(kwargs)
        tensor_names = [n for n, v in bound.items() if isinstance(v, torch.Tensor)]
        tensors = [bound[n] for n in tensor_names]
        scalars = {n: _jsonable(v) for n, v in bound.items() if not isinstance(v, torch.Tensor)}
        fn = getattr(self, "fn", None)
        kernel = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', getattr(self, '__name__', '?'))}"
        # implementation identity beyond the name: sha256 of the kernel's Python source (what the eager capture's
        # triton_adapter records as `src_sha`); effective launch geometry (explicit kwargs, else Triton's defaults)
        try:
            src_sha = hashlib.sha256(str(getattr(self, "src", "")).encode("utf-8")).hexdigest()
        except Exception:  # noqa: BLE001
            src_sha = None
        try:
            cnames = sorted(p.name for p in self.params if p.is_constexpr)
        except Exception:  # noqa: BLE001
            cnames = sorted(getattr(self, "constexprs", []) if isinstance(getattr(self, "constexprs", None), (list, tuple, set)) else [])
        num_warps = kwargs.get("num_warps", 4)  # triton default
        num_stages = kwargs.get("num_stages", 3)  # triton default (sm80+)
        geometry = {"num_warps": num_warps, "num_warps_explicit": "num_warps" in kwargs, "num_stages": num_stages,
                    "num_stages_explicit": "num_stages" in kwargs, "num_ctas": kwargs.get("num_ctas", 1)}
        try:
            g = tuple(grid) if isinstance(grid, (tuple, list)) else (grid if isinstance(grid, int) else None)
            grid_json = _jsonable(list(g)) if g is not None else "<callable>"
        except Exception:  # noqa: BLE001
            grid_json = "<unknown>"
        meta = {"kernel": kernel, "src_sha256": src_sha, "triton": _TRITON_VERSION, "grid": grid_json, "scalars": scalars, "geometry": geometry,
                "tensor_args": tensor_names, "constexprs": cnames,
                "tensor_meta": [{"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype)} for t in tensors]}
        log.launches.append(dict(meta))
        op(kernel, tensors, json.dumps(meta, sort_keys=True))
        return None

    return run


@contextmanager
def intercept_triton_launches(log: TritonLaunchLog | None = None):
    """Patch `triton.runtime.jit.JITFunction.run` for the duration of an export."""
    log = log if log is not None else TritonLaunchLog()
    try:
        from triton.runtime.jit import JITFunction
    except Exception:  # noqa: BLE001 — no triton: nothing to intercept
        yield log
        return
    orig = JITFunction.run
    JITFunction.run = _make_run(orig, log)
    try:
        yield log
    finally:
        JITFunction.run = orig


__all__ = ["op", "intercept_triton_launches", "TritonLaunchLog"]
