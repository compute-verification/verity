"""Launch proxies on vLLM's top-p split kernels, which record the split count S each launch ran with (`pipeline.topp_pin_check`
holds it to the registered one, `topp_geometry.registered_splits`)."""
from __future__ import annotations

import sys
from typing import Any

from verity_vllm.engine import hooks

#: vLLM's split-row top-p pipeline (`vllm.v1.sample.ops.topk_topp_triton._apply_topp_split`): three Triton JITFunctions
#: looked up from the module's globals at every launch, each taking the split count as the constexpr kwarg `S` over grid (batch_size * S,).
TOPP_SPLIT_MODULE = "vllm.v1.sample.ops.topk_topp_triton"
TOPP_SPLIT_KERNELS = ("_topp_sb_stats_kernel", "_topp_sb_step_kernel", "_topp_sb_mask_kernel")


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
            try:                                              # vLLM launches on a literal tuple grid; a callable (meta) grid is left unread
                g0 = None if callable(grid) else (int(grid[0]) if isinstance(grid, (tuple, list)) else int(grid))
            except (TypeError, ValueError, IndexError):
                g0 = None
            s = kw.get("S")
            sink.append((name, g0, int(s) if isinstance(s, int) else None))
            return launch(*args, **kw)
        return run

    def __getattr__(self, item: str) -> Any:
        return getattr(self._kernel, item)


def install_split_kernel_proxies(sink: list) -> dict[str, hooks.Patch]:
    """Replace the three split kernels in vLLM's module namespace with recording proxies; returns {name: patch} for uninstall (empty
    when the module / kernels are absent: an older vLLM without the split pipeline records no launch)."""
    try:
        mod = sys.modules.get(TOPP_SPLIT_MODULE) or __import__(TOPP_SPLIT_MODULE, fromlist=["_"])
    except Exception:  # noqa: BLE001
        return {}
    patches: dict[str, hooks.Patch] = {}
    for nm in TOPP_SPLIT_KERNELS:
        k = getattr(mod, nm, None)
        if k is None or isinstance(k, _SplitKernelProxy):
            continue
        patches[nm] = hooks.patch(mod, nm, _SplitKernelProxy(k, nm, sink))
    return patches


def uninstall_split_kernel_proxies(patches: dict[str, hooks.Patch]) -> None:
    for p in reversed(list(patches.values())):
        p.uninstall()
