"""Launch proxies on vLLM's top-p split kernels, which record the split count S each launch ran with (`pipeline.topp_pin_check`
holds it to the registered one, `topp_geometry.registered_splits`)."""
from __future__ import annotations

import importlib
import sys
from numbers import Integral
from typing import Any

from verity_vllm.engine import hooks

__all__ = ["install_split_kernel_proxies"]

#: vLLM's split-row top-p pipeline (`vllm.v1.sample.ops.topk_topp_triton._apply_topp_split`): three Triton JITFunctions
#: looked up from the module's globals at every launch, each taking the split count as the constexpr kwarg `S` over grid (batch_size * S,).
TOPP_SPLIT_MODULE = "vllm.v1.sample.ops.topk_topp_triton"
TOPP_SPLIT_KERNELS = ("_topp_sb_stats_kernel", "_topp_sb_step_kernel", "_topp_sb_mask_kernel")


def _grid_rows(grid: Any) -> int | None:
    """The first dimension of a launch grid given as an integer or as a tuple or list led by one; None for any other grid (vLLM
    launches on a literal tuple; a callable grid is left unread)."""
    head = grid[0] if isinstance(grid, (tuple, list)) and grid else grid
    return int(head) if isinstance(head, Integral) else None


class _SplitKernelProxy:
    """Launch proxy for one `_topp_sb_*_kernel` JITFunction: `proxy[grid](*args, **kw)` appends (kernel name, grid, kw['S']) to `sink`
    and then performs the ORIGINAL launch unchanged.  The recorded S is the constexpr the kernel was compiled and ran with, never
    recomputed from the batch.  Attribute access falls through to the wrapped JITFunction."""

    def __init__(self, kernel: Any, name: str, sink: list) -> None:
        self._kernel, self._name, self._sink = kernel, name, sink

    def __getitem__(self, grid: Any):
        launch = self._kernel[grid]
        name, sink = self._name, self._sink

        def run(*args: Any, **kw: Any):
            s = kw.get("S")
            sink.append((name, _grid_rows(grid), int(s) if isinstance(s, int) else None))
            return launch(*args, **kw)
        return run

    def __getattr__(self, item: str) -> Any:
        return getattr(self._kernel, item)


def install_split_kernel_proxies(sink: list) -> dict[str, hooks.Patch]:
    """Replace the three split kernels in vLLM's module namespace with recording proxies; returns {name: patch}, each of which its
    caller uninstalls (empty when the module has none of them: a vLLM whose top-p has no split kernels records no launch).  A vLLM
    without the module is an ImportError: there is no launch to read a split count from."""
    mod = sys.modules.get(TOPP_SPLIT_MODULE) or importlib.import_module(TOPP_SPLIT_MODULE)
    patches: dict[str, hooks.Patch] = {}
    for nm in TOPP_SPLIT_KERNELS:
        k = getattr(mod, nm, None)
        if k is None or isinstance(k, _SplitKernelProxy):
            continue
        patches[nm] = hooks.patch(mod, nm, _SplitKernelProxy(k, nm, sink))
    return patches

