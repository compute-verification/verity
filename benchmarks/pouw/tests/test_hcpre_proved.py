"""HC-PREFILL's proved arm (benchmarks/pouw/filler/hcpre_proved.py) meets the filler arm contract and its record's limits,
declares Theorem K's noise, builds pfill.cuh's co tile without the phase filler, and places the filler on pfill-k's decode
splits and co for prefill."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

FILLER = Path(__file__).resolve().parents[1] / "filler"


def _module(name):
    """The scorer's module as test_filler_score.py registered it, else loaded here: an arm file imports `arms` from
    sys.modules, so one copy must serve every test in the process."""
    mod = sys.modules.get(name)
    if mod is not None and Path(getattr(mod, "__file__", "")).resolve() == FILLER / f"{name}.py":
        return mod
    spec = importlib.util.spec_from_file_location(name, FILLER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pfill_k_co_arm(monkeypatch):
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "")
    monkeypatch.syspath_prepend(str(FILLER))
    A, S = _module("arms"), _module("scoring")
    arm = A.load(str(FILLER / "hcpre_proved.py") + ":PFillKCo", record=True)
    assert A.check_arm(arm, record=True) == []
    import os
    assert os.environ["NVCC_APPEND_FLAGS"].split() == ["-DPFILL_BITS", "-DPFILL_CO", "-DPFILL_NO_PHASE"]
    assert arm.filler.noise == "bits01" and arm.filler.W == 4 and arm.filler.NC == 4
    assert "bits01" in S.FILLER_ENTRIES["theorem-k"]["noise"]
    split = _module("pfill_arm").POWER_SPLIT
    for ph, b, l in S.points(robust=True):
        p = A.placed(arm, ph, b, l)
        if ph == "prefill":
            assert p == A.Placement("co")
        else:
            assert p == A.Placement("split", A.point_lookup(split, b, l))
    assert set(arm.sources) >= {"benchmarks/pouw/decode_capacity/pfill.cuh", "benchmarks/pouw/filler/hcpre_proved.py"}


def test_co_tile_is_the_4_warp_tile():
    """The co tile's receipts are dfill_ref's for a tile of PF_W = 4 warps (B 32 x 512), the width it builds at
    unless -DPF_CO_W says otherwise."""
    src = (FILLER.parent / "decode_capacity" / "pfill.cuh").read_text()
    co = src[src.index("#ifdef PFILL_CO"):src.index("#else", src.index("#ifdef PFILL_CO"))]
    assert "#define PF_CO_W 4" in co
    assert "#define PF_W PF_CO_W" in co
    assert "#define PF_TN (128 * PF_W)" in src
