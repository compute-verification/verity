"""The collector's binding of vLLM's top-p split count S (`runner.sampler/splits`, manifest family `sampler_splits`):
launch proxies on the split kernels, the per-step S of their launches, and the prescribed S of a step on which vLLM ran no split kernel."""
from __future__ import annotations

import sys
from typing import Any, Callable

from verity_vllm.engine import hooks

#: vLLM's split-row top-p pipeline (`vllm.v1.sample.ops.topk_topp_triton._apply_topp_split`): three Triton JITFunctions
#: looked up from the module's globals at every launch, each taking the split count as the constexpr kwarg `S` over grid (batch_size * S,).
TOPP_SPLIT_MODULE = "vllm.v1.sample.ops.topk_topp_triton"
TOPP_SPLIT_KERNELS = ("_topp_sb_stats_kernel", "_topp_sb_step_kernel", "_topp_sb_mask_kernel")
SPLITS_IDENTITY = "runner.sampler/splits"


class _SplitKernelProxy:
    """Launch proxy for one `_topp_sb_*_kernel` JITFunction: `proxy[grid](*args, **kw)` appends (kernel name, grid, kw['S']) to `sink`
    and then performs the ORIGINAL launch unchanged.  The recorded S is the constexpr the kernel was compiled and ran with -- the value
    of record for `runner.sampler/splits`; the collector never recomputes it from the batch (that would be `SplitsFor` observed post hoc,
    not the kernel's own S).  Attribute access falls through to the wrapped JITFunction."""

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
    when the module / kernels are absent -- an older vLLM without the split pipeline binds no splits, and the manifest's `sampler_splits`
    rows then fail coverage closed, visibly)."""
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


def split_count_of_launches(launches: list[tuple], n_rows: int) -> tuple[int | None, str | None]:
    """The split count S ONE sampling step's split-kernel launches ran with, or (None, why).  Every launch of the step must carry the same
    S and a grid of exactly n_rows * S (the sampled rows): a disagreement is a finding (bind nothing -> the `sampler_splits` rows fail
    coverage closed), never averaged or taken from the last launch."""
    if not launches:
        return None, "no split-kernel launch this step"
    Ss = {s for _n, _g, s in launches}
    if len(Ss) != 1 or None in Ss:
        return None, f"split kernels disagree on S within one step: {sorted(str(s) for s in Ss)}"
    S = Ss.pop()
    if S < 1 or S & (S - 1):
        return None, f"kernel S={S} is not a power of two"
    bad = [(n, g) for n, g, _s in launches if g != n_rows * S]
    if bad:
        return None, f"split-kernel grid != rows*S ({n_rows}*{S}): {bad[:3]}"
    if not any(n == "_topp_sb_mask_kernel" for n, _g, _s in launches):
        return None, "the mask kernel did not launch this step"
    return S, None



def prescribed_idle_splits(splits_for: Callable[[int, int], int], top_p_rows: Any, n_rows: int, num_sms: int) -> tuple[int | None, str | None]:
    """The split count a derived-S Program prescribes for a sampling step on which no split kernel launched, or (None, why).

    vLLM d9105ea80 `SamplingStates.get_top_k_top_p` skips top-p for the whole batch when every row's top_p is 1 (no kernel, logits
    untouched), and `GumbelTopPTokenSelect`'s mask returns early at p >= 1 keeping every lane for any S: the select's output does not
    depend on S there, so the Program's input is its prescribed value `splits_for(rows, num_SMs)` (SplitsFor_v1), which the boundary
    linkage checks.  A row with top_p < 1 and no launch is a finding: nothing is bound (coverage fails closed)."""
    if top_p_rows is None:
        return None, "no per-row top_p to show the batch skipped top-p"
    ps = [float(p) for p in top_p_rows]
    if len(ps) != n_rows:
        return None, f"{len(ps)} top_p rows for {n_rows} sampled rows"
    if any(p != 1.0 for p in ps):
        return None, "a row has top_p < 1 but no split kernel launched this step"
    return int(splits_for(n_rows, num_sms)), None


def batch_top_p(runner: Any, a: tuple, kw: dict) -> Any:
    """The step's per-row top_p as the served sampler reads it (`sampling_states.top_p.np[input_batch.idx_mapping_np]`), or None."""
    ib = a[1] if len(a) > 1 else kw.get("input_batch")
    tp = getattr(getattr(getattr(runner, "sampler", None), "sampling_states", None), "top_p", None)
    idx = getattr(ib, "idx_mapping_np", None)
    return tp.np[idx] if (idx is not None and hasattr(tp, "np")) else None


def step_splits(launches: list[tuple], n_rows: int, splits_for: Callable[[int, int], int] | None, top_p_rows: Callable[[], Any],
                num_sms: Callable[[], int]) -> tuple[int | None, str | None, str]:
    """One sampling step's S and the counter it bumps: the launches' S when a split kernel ran; else, for a derived-S Program
    (`splits_for` set), the prescribed S (`prescribed_idle_splits`); else nothing.  `why` names a step that should bind and cannot."""
    if launches:
        S, why = split_count_of_launches(launches, n_rows)
        return S, why, "splits_bound_steps"
    if splits_for is None:
        return None, None, ""
    S, why = prescribed_idle_splits(splits_for, top_p_rows(), n_rows, num_sms())
    return S, why, "splits_prescribed_steps"
