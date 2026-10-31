"""FILL's fused arm (benchmarks/pouw/filler/fused_arm.py) against what runs it: the filler-arm/v1 contract and the
ledger's record limits, its placements, fused_score.py's fscore arguments in fscore.cu's order, and every table row's
flin and attention filler configurations among those flin.cuh builds (a row naming an unbuilt one fails on the GPU only
after the build and the lease)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

FILLER = Path(__file__).resolve().parents[1]
DCAP = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, FILLER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


F = _load("fused_arm")
FS = _load("fused_score")
A = F.A  # the arms module fused_arm imported (the one test_filler_score loaded, when it ran first): one Placement class
ARMS = [F.Fused(), F.FusedV2(), F.FusedPump(), F.FusedPumpK(), F.FusedWarpK(), F.FusedPumpBU(), F.FusedPumpBUXin13(),
        F.FusedPumpBULi()]
V1 = [a for a in ARMS if a.nvcc_flags == "-DDFILL_V1"]


def _built() -> tuple[dict[str, set], set]:
    src = (DCAP / "flin.cuh").read_text()
    body = re.search(r"static const char\* flin_built\(int M\) \{(.*?)\n\}", src, re.S).group(1)
    lists = re.findall(r'"([^"]+)"', body)
    flin = dict(zip(["32", "64", "128", "160/192", "256"], (set(s.split(", ")) for s in lists)))
    fatt = set(re.search(r'fatt_built\(\) \{ return "([^"]+)"', src).group(1).split(", "))
    return flin, fatt


def _flin_key(b):
    m = 32 if b <= 32 else 64 if b <= 64 else 128 if b <= 128 else 160 if b <= 192 else 256
    return "160/192" if m == 160 else str(m)


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: a.name)
def test_contract_and_record(arm):
    assert A.check_arm(arm, record=True) == []
    if arm in V1:  # FILL v1's chains, and flin's and attention's filler warps; dcap.cu's filler is pfill otherwise
        assert arm.filler.noise == F.CO_PREFILL.noise == "chained"
    else:  # pfill-k's {0,1} tile or pfill-bU's, the settings filler-scorer/v3's Theorem K entry reads
        kind = {"-DPFILL_BITS": "bits01", "-DPFILL_BU": "bitsU"}[arm.nvcc_flags.split()[0]]
        assert arm.filler.noise == kind
        assert arm.filler.W == 8 and arm.filler.S % 32 == 0  # pfill's 8-warp blocks; S a multiple of 32
        # the fold filler-scorer/v5 reads, declared as the NC and build make it (pf_absorb): duo at NC 2; at NC 4 ram,
        # or the r9 XOR-in (13, 19) or LOP3+IMAD
        flags = arm.nvcc_flags.split()
        r9 = [f for f in ("-DPFILL_XIN13", "-DPFILL_LI") if f in flags]
        assert len(r9) <= 1 and "-DPFILL_XIN" not in flags  # pfill.cuh refuses both pairs (#error)
        nc4 = {"-DPFILL_XIN13": "xor-in-13", "-DPFILL_LI": "lop3-imad"}[r9[0]] if r9 else "ram"
        assert arm.filler.fold == {2: "duo", 4: nc4}[arm.filler.NC]
        assert kind == "bits01" or arm.filler.fold != "ram"  # bU's pump arms bind every word


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: a.name)
def test_placements(arm):
    pre = A.Placement("co") if arm in V1 else A.Placement("split", 4)
    assert A.placed(arm, "prefill", 8192, 0) == pre
    assert arm.fused("prefill", 8192, 0) is None
    assert arm.filler_at("prefill", 8192, 0) == (F.CO_PREFILL if arm in V1 else arm.filler)
    for (b, l), (sms, q) in arm.table.items():
        assert A.placed(arm, "decode", b, l) == A.Placement("split", sms)
        assert arm.fused("decode", b, l) == q and arm.filler_at("decode", b, l) == arm.filler


def test_fscore_arguments_follow_its_usage():
    use = re.search(r'const char\* use = "fscore ([^"]+)"', (FILLER / "fscore.cu").read_text()).group(1).split()
    fields = [f.name for f in F.FusedStep.__dataclass_fields__.values()]
    assert use[use.index("S") + 1:use.index("ROUNDS")] + ["ST"] == fields  # ST reaches fscore as FLIN_STOP
    assert "FLIN_STOP=str(q.ST)" in (FILLER / "fused_score.py").read_text()
    q = F.FusedStep(FW=1, FNC=2, AFW=3, ANC=4, APS=5, ALZ=6, LT=7, LM=8, ST=9)
    assert FS.fused_args(q) == [str(i) for i in range(1, 9)]


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: a.name)
def test_table_configurations_are_built(arm):
    flin, fatt = _built()
    for (b, l), (_, q) in arm.table.items():
        if q.LT != 15:
            assert q.FW == 0 or f"{q.FW}:{q.FNC}" in flin[_flin_key(b)], (b, l, q)
        assert q.AFW == 0 or f"{q.AFW}:{q.ANC}" in fatt, (b, l, q)
        assert q.LM == 0 or q.LT != 15, (b, l, q)  # lm_head through flin needs a linear through flin before it


def test_pump_arm_has_no_fixed_work():
    """filler-scorer/v3 reads no theorem for a filler with an arm-side driver's fixed work, so FusedPump's rows carry no
    filler warps."""
    assert all(q.FW == 0 and q.AFW == 0 for _, q in F.FusedPump().table.values())
    assert all(a().table == F.FusedPump().table
               for a in (F.FusedPumpK, F.FusedPumpBU, F.FusedPumpBUXin13, F.FusedPumpBULi))


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: a.name)
def test_pump_only_build_runs_no_filler_warp(arm):
    """-DFSCORE_PUMP_ONLY builds fscore with no flin or attention filler warps (fscore.cu sets FLIN_PUMP_ONLY before
    dcap.cu brings in flin.cuh), so an arm with it has none in any row, and filler-scorer/v5 reads only its pump's fold."""
    if "-DFSCORE_PUMP_ONLY" not in arm.nvcc_flags.split():
        return
    assert all(q.FW == 0 and q.AFW == 0 for _, q in arm.table.values())
    src = (FILLER / "fscore.cu").read_text()
    assert re.search(r"#ifdef FSCORE_PUMP_ONLY\n#define FLIN_PUMP_ONLY\n#endif\n#define DCAP_FLIN\n", src)
    assert src.index("#define FLIN_PUMP_ONLY") < src.index('#include "dcap.cu"')


def test_warp_arm_fills_through_dfill_k_warp():
    """FusedWarpK's filler warps are flin's alone, each stopped before its block's end, and on a pfill build flin's filler
    step is the `__noinline__` dfill_k_warp the scorer's check reads (coins, tile id and step its only inputs)."""
    rows = F.FusedWarpK().table
    assert all(q.AFW == 0 and (q.FW == 0) == (q.ST == 0) for _, q in rows.values())
    assert sum(q.FW > 0 for _, q in rows.values()) >= 16
    assert all(rows[k] == F.TABLE_PUMP[k] for k, (_, q) in rows.items() if q.FW == 0)
    src = (DCAP / "flin.cuh").read_text()
    assert re.search(r"__device__ __noinline__ uint64_t dfill_k_warp\(uint64_t c0, uint64_t c1, unsigned long long tile, "
                     r"unsigned long long step\)", src)
    assert re.search(r"h \^= dfill_k_warp\(rs\.c0, rs\.c1, tile, \(unsigned long long\)steps\);", src)
