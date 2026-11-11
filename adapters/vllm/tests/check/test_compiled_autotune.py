"""compiled_autotune: the autotune-equivalence record is fail-closed and never raises into the kernel check (CPU-only tests; the GPU
demonstration -- every candidate config of Inductor's pointwise heuristic reproducing the committed silu_mul outputs bit for bit -- ran on
cp-dev2 against the Qwen run-9 generated kernel: candidates {XBLOCK 1024/4 warps, XBLOCK 512/8 warps} == the two configs the arms executed,
13.44 M elements, 0 differing)."""
from __future__ import annotations

import sys
import types

from verity_vllm.check import compiled_autotune as ca


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
    assert ca.candidates("no_such_kernel_" + "z" * 6) == (None, ca.NO_AUTOTUNER)


def test_an_unimportable_autotuner_is_named_in_the_record(monkeypatch):
    monkeypatch.setattr(ca, "_AUT", None)
    monkeypatch.setattr(ca, "_SCAN_ERROR", None)
    monkeypatch.setitem(sys.modules, "torch._inductor.codecache", None)
    rec = ca.replay_under_candidates("triton_poi_k", [None], {}, {0: "x", 1: "out"}, {"out"})
    assert rec["shown"] is False
    assert rec["why"].startswith(f"{ca.NO_AUTOTUNER}: Inductor's autotuner is not importable (ModuleNotFoundError"), rec["why"]


def test_an_unreadable_pointwise_decorator_makes_the_candidates_not_reconstructable(tmp_path, monkeypatch):
    """Defaults in place of the decorator's hints could offer another candidate set than the process's heuristic did."""
    from torch._inductor.runtime.hints import HeuristicType

    code = types.SimpleNamespace(co_filename=str(tmp_path / "gone.py"))
    fn = types.SimpleNamespace(__name__="triton_poi_k", fn=types.SimpleNamespace(__code__=code))
    aut = types.SimpleNamespace(heuristic_type=HeuristicType.POINTWISE, inductor_meta={}, triton_meta={}, size_hints={"x": 8},
                                device_props=None, fn=fn, filename="gone.py", launchers=[])
    monkeypatch.setattr(ca, "_AUT", {"triton_poi_k": aut})
    monkeypatch.setattr(ca, "_CANDS", {})
    res, why = ca.candidates("triton_poi_k")
    assert res is None and why.startswith("reconstructing the heuristic's candidates raised: FileNotFoundError"), why
    rec = ca.replay_under_candidates("triton_poi_k", [None], {}, {0: "x", 1: "out"}, {"out"})
    assert rec["shown"] is False and rec["why"] == f"candidate configs not reconstructable: {why}"
