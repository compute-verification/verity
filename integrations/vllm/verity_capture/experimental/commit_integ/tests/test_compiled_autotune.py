"""compiled_autotune: the autotune-equivalence record is fail-closed and never raises into the kernel check (CPU-only tests; the GPU
demonstration -- every candidate config of Inductor's pointwise heuristic reproducing the committed silu_mul outputs bit for bit -- ran on
cp-dev2 against the Qwen run-9 generated kernel: candidates {XBLOCK 1024/4 warps, XBLOCK 512/8 warps} == the two configs the arms executed,
13.44 M elements, 0 differing)."""
from __future__ import annotations

from verity_capture.experimental.commit_integ import compiled_autotune as ca


class _Cfg:
    def __init__(self, kwargs, num_warps, num_stages):
        self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages


def test_cfg_dict_and_label_are_canonical():
    d = ca.cfg_dict(_Cfg({"XBLOCK": 512, "R0_BLOCK": 2048, "junk": "x"}, 8, 1))
    assert d == {"kwargs": {"R0_BLOCK": 2048, "XBLOCK": 512}, "num_warps": 8, "num_stages": 1}
    assert ca.cfg_label(d) == "R0_BLOCK=2048 XBLOCK=512 warps=8 stages=1"


def test_unknown_kernel_is_not_shown_and_does_not_raise():
    rec = ca.replay_under_candidates("no_such_kernel_" + "z" * 6, [None], {}, {0: "x", 1: "out"}, {"out"})
    assert rec["shown"] is False and rec["why"] == "no autotuner bound to this kernel in the process"
    assert rec["candidates"] == [] and rec["executed"] == [] and rec["results"] == []
    assert ca.candidates("no_such_kernel_" + "z" * 6) is None
