"""The proved-filler scorer arms (benchmarks/pouw/filler/pfill_arm.py) meet the filler arm contract, and place the
filler as FILL v1 does at every suite point; pfill_proved.py's readings are the scorer's arithmetic with only the
filler's certified fraction changed."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

FILLER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FILLER.parent / "harness"))
import ledger  # noqa: E402


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


def _load_file(name):
    spec = importlib.util.spec_from_file_location(f"hcproved_{name}", FILLER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ARMS = (("PFillRam", 4, "", 1), ("PFillDuo", 2, "", 1), ("PFillE2m1Ram", 4, "-DPFILL_E2M1", 2))


def test_pfill_arms_meet_the_contract_and_place_as_fill_v1(monkeypatch):
    """Each arm sets the tile kind score.py's build inherits (NVCC_APPEND_FLAGS), says its products a slot, and has a
    lineage score.py's ledger record takes (scored, then refused a 249-character hypothesis)."""
    A, S = _module("arms"), _module("scoring")
    P = _load_file("pfill_arm")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "unset")
    v1 = A.load("fill-v1")
    for attr, nc, flags, per_slot in ARMS:
        arm = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{attr}")
        assert A.check_arm(arm) == []
        assert ledger.one_line(arm.lineage["hypothesis"], "the hypothesis")
        assert __import__("os").environ["NVCC_APPEND_FLAGS"] == flags
        assert P.products_per_slot(arm.lineage) == per_slot
        assert arm.filler == A.Filler(W=8, NC=nc, S=256, chunk_log2=19, noise="pm1")
        assert arm.filler.S % 32 == 0  # pfill's dfill_make refuses any other S
        assert "benchmarks/pouw/filler/pfill.cuh" in arm.sources
        for ph, b, l in S.points():
            assert A.placed(arm, ph, b, l) == A.placed(v1, ph, b, l)


@pytest.mark.parametrize("attr, flags, k", [("PFillPowerSplit", "", False), ("PFillK", "-DPFILL_BITS", True)])
def test_power_split_arms_have_a_split_at_every_scored_point(monkeypatch, attr, flags, k):
    """pfill-power-split and pfill-k are pfill-ram's filler (pfill-k's on {0,1} bits) on the power-split table: an exact
    entry at every scored decode point, and 4 filler SMs during prefill. Each declares the noise filler-scorer/v3 reads:
    pfill-k bits01, the noise of Theorem K's entry, the ±1 tiles pm1, Theorem E's."""
    A, S = _module("arms"), _module("scoring")
    P = _load_file("pfill_arm")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "unset")
    arm = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{attr}")
    env = __import__("os").environ["NVCC_APPEND_FLAGS"]
    ram = A.load("benchmarks/pouw/filler/pfill_arm.py:PFillRam")
    assert A.check_arm(arm) == []
    assert ledger.one_line(arm.lineage["hypothesis"], "the hypothesis")
    noise = "bits01" if k else "pm1"
    assert (arm.filler, arm.sources, env) == (dataclasses.replace(ram.filler, noise=noise), ram.sources, flags)
    assert noise in S.FILLER_ENTRIES["theorem-k" if k else "theorem-e"]["noise"]
    assert (P.theorem_k_setting(arm.lineage), P.products_per_slot(arm.lineage)) == (k, 1)
    for ph, b, l in S.points(robust=False):
        p = A.placed(arm, ph, b, l)
        assert p == (A.Placement("split", 4) if ph == "prefill" else A.Placement("split", P.POWER_SPLIT[(b, l)]))
    for b, l in S.ROBUST:
        assert A.placed(arm, "decode", b, l) == A.Placement("split", A.point_lookup(P.POWER_SPLIT, b, l))


def test_k_phase_arm_runs_pfill_k_with_a_phase_filler_of_the_builds_width(monkeypatch):
    """pfill-k-phase is pfill-k's pump with an exact table row at every suite point; each phase filler is a shape pscore
    takes (8 | S > 8, tiles >= 1) at the width its build fixes (-DPF_PW, which plaunch holds dfill_p to), on a split
    (psched_score refuses a phase point without one); prefill keeps pfill-k's 4-SM split and no phase filler."""
    A, S = _module("arms"), _module("scoring")
    P = _load_file("pfill_arm")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "unset")
    k = A.load("benchmarks/pouw/filler/pfill_arm.py:PFillK")
    arm = A.load("benchmarks/pouw/filler/pfill_arm.py:PFillKPhase")
    flags = __import__("os").environ["NVCC_APPEND_FLAGS"]
    assert A.check_arm(arm) == []
    assert ledger.one_line(arm.lineage["hypothesis"], "the hypothesis")
    assert flags.split()[0] == "-DPFILL_BITS" and P.theorem_k_setting(arm.lineage)
    pw = int(flags.split("-DPF_PW=")[1].split()[0])
    assert arm.filler == k.filler
    phased = 0
    for ph, b, l in S.points(robust=True):
        q = arm.phase(ph, b, l)
        if ph == "prefill":
            assert q is None and A.placed(arm, ph, b, l) == A.Placement("split", 4)
            continue
        assert (b, l) in P.PHASE_K
        assert A.placed(arm, ph, b, l).place == "split"
        if q is not None:
            phased += 1
            assert (q.W, q.NC) == (pw, 4) and q.S > 8 and q.S % 8 == 0 and q.tiles >= 1
    assert phased


@pytest.mark.parametrize("attr, table", [("PFillKFused", "FUSED_K"), ("PFillKFused2", "FUSED_K2"),
                                         ("PFillKFused3", "FUSED_K3"), ("PFillKFused4", "FUSED_K4")])
def test_k_fused_arm_runs_pfill_k_beside_fills_step_with_no_filler_warps(monkeypatch, attr, table):
    """pfill-k-fused is pfill-k's pump (bits01, the build's -DPFILL_BITS) with an exact table row at every suite point; a
    fused point runs FILL's step with no filler warps (FW = AFW = 0: fscore counts no fixed work, which filler-scorer/v3's
    Theorem K entry refuses), always on a split; prefill keeps pfill-k's 4-SM split; its sources carry fscore's build."""
    A, S = _module("arms"), _module("scoring")
    P = _load_file("pfill_arm")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "unset")
    k = A.load("benchmarks/pouw/filler/pfill_arm.py:PFillK")
    arm = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{attr}")
    assert arm.table == getattr(P, table)
    assert A.check_arm(arm) == [] and A.record_problems(arm) == []
    assert __import__("os").environ["NVCC_APPEND_FLAGS"] == "-DPFILL_BITS"
    assert arm.filler == k.filler and arm.filler.noise == "bits01" and P.theorem_k_setting(arm.lineage)
    assert {"benchmarks/pouw/filler/flin.cuh", "benchmarks/pouw/filler/fscore.cu"} <= set(arm.sources)
    fused = 0
    for ph, b, l in S.points(robust=True):
        q = arm.fused(ph, b, l)
        if ph == "prefill":
            assert q is None and A.placed(arm, ph, b, l) == A.Placement("split", 4)
            continue
        assert (b, l) in getattr(P, table) and A.placed(arm, ph, b, l).place == "split"
        if q is not None:
            fused += 1
            assert (q.FW, q.AFW) == (0, 0)
        else:
            assert A.placed(arm, ph, b, l).fill_sms == A.point_lookup(P.POWER_SPLIT, b, l)
    assert fused


@pytest.mark.parametrize(("attr", "fold"), [("PFillBUDuoFused4", "PFillBUDuo")])
def test_fused4_fold_arms_are_fused4_with_only_the_filler_changed(monkeypatch, attr, fold):
    """Each fold arm on fused4's placement is pfill-k-fused4's table, fused steps, sources and prefill split with the
    fold arm's filler, build flags and receipt, so v4 reads the tile it pumps and pfill_proved its fold share."""
    A, S = _module("arms"), _module("scoring")
    P = _load_file("pfill_arm")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "unset")
    f4 = A.load("benchmarks/pouw/filler/pfill_arm.py:PFillKFused4")
    tile = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{fold}")
    want = __import__("os").environ["NVCC_APPEND_FLAGS"]
    arm = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{attr}")
    assert A.check_arm(arm, record=True) == [] and A.record_problems(arm) == []
    assert __import__("os").environ["NVCC_APPEND_FLAGS"] == want
    assert (arm.filler, arm.sources, arm.table) == (tile.filler, f4.sources, P.FUSED_K4)
    assert [t for t in arm.lineage["mechanism"] if t.startswith(("noise:", "receipt:"))] == [
        t for t in tile.lineage["mechanism"] if t.startswith(("noise:", "receipt:"))]
    assert P.fold_share(arm.lineage) == 1.0 and P.theorem_k_setting(arm.lineage)
    for ph, b, l in S.points(robust=True):
        assert A.placed(arm, ph, b, l) == A.placed(f4, ph, b, l) and arm.fused(ph, b, l) == f4.fused(ph, b, l)


@pytest.mark.parametrize(("attr", "base", "nc", "tiles"), [("PFillBUDuoFPhase", "PFillBUDuoFused4", 2, {2048: 5, 3072: 8}),
                                                          ("PFillBUDuoFPhase2", "PFillBUDuoFused4", 2, {2048: 5, 3072: 6}),
                                                          ("PFillBUDuoFPhase3", "PFillBUDuoFused4", 2, {2048: 5, 3072: 7})])
def test_fphase_arms_add_a_phase_filler_at_b256_only(monkeypatch, attr, base, nc, tiles):
    """An fphase arm is its fused4 fold arm with a phase filler of the build's fold (NC) and width (-DPF_PW=4) at 256/2K
    and 256/3K, on the split there and with no fused step; every other point is the fused4 arm's."""
    A, S = _module("arms"), _module("scoring")
    P = _load_file("pfill_arm")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "unset")
    f4 = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{base}")
    want = __import__("os").environ["NVCC_APPEND_FLAGS"].split()
    arm = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{attr}")
    assert A.check_arm(arm, record=True) == [] and A.record_problems(arm) == []
    assert sorted(__import__("os").environ["NVCC_APPEND_FLAGS"].split()) == sorted(want + ["-DPF_PW=4"])
    assert arm.filler == f4.filler and P.fold_share(arm.lineage) == 1.0
    assert {"benchmarks/pouw/filler/pscore.cu", "benchmarks/pouw/filler/pfill_fp_score.py"} <= set(arm.sources)
    for ph, b, l in S.points(robust=True):
        q = arm.phase(ph, b, l)
        assert A.placed(arm, ph, b, l) == A.placed(f4, ph, b, l)
        if b == 256 and l in tiles:
            assert arm.fused(ph, b, l) is None and A.placed(arm, ph, b, l).place == "split"
            assert (q.W, q.NC, q.tiles) == (4, nc, tiles[l])
        else:
            assert q is None and arm.fused(ph, b, l) == f4.fused(ph, b, l)


def test_bu_arm_is_power_split_on_the_bu_tile(monkeypatch):
    """pfill-bu is pfill-power-split's filler and placement with only the tile's kind changed (-DPFILL_BU), in Theorem
    K's setting and no other arm."""
    A, S = _module("arms"), _module("scoring")
    P = _load_file("pfill_arm")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "unset")
    split = A.load("benchmarks/pouw/filler/pfill_arm.py:PFillPowerSplit")
    arm = A.load("benchmarks/pouw/filler/pfill_arm.py:PFillBU")
    assert A.check_arm(arm) == []
    assert ledger.one_line(arm.lineage["hypothesis"], "the hypothesis")
    assert __import__("os").environ["NVCC_APPEND_FLAGS"].split() == ["-DPFILL_BU", "-Xcompiler", "-mpopcnt"]
    assert (replace(arm.filler, noise=split.filler.noise), arm.sources, arm.lineage["approach"]) == (
        split.filler, split.sources, "pouw/filler-proved")
    assert arm.filler.noise == "bitsU"
    assert P.products_per_slot(arm.lineage) == 1 and P.theorem_k_setting(arm.lineage)
    assert not any(P.theorem_k_setting(getattr(P, c).__dataclass_fields__["lineage"].default_factory())
                   for c in ("PFillRam", "PFillDuo", "PFillPowerSplit", "PFillE2m1Ram"))
    for ph, b, l in S.points():
        assert A.placed(arm, ph, b, l) == A.placed(split, ph, b, l)


def _row(kind, b, l, alone_ms, arm_ms):
    """A minimal `dscore` row with filler: 4 ABBA blocks, 3 reps each side, 10 chunks of 1e9 MACs a window."""
    blocks = [{"order": "alone,arm", "alone": {"window_ms": 3 * alone_ms, "reps": [[alone_ms, 2000.0, 300.0]] * 3},
               "arm": {"window_ms": 3 * arm_ms, "reps": [[arm_ms, 2000.0, 300.0]] * 3, "chunks": 10}}
              for _ in range(4)]
    return {"mode": "dscore", "kind": kind, "B": b, "L": l, "place": "split", "fill_sms": 60, "chunk_macs": 1e9,
            "alone_fill": {"mac_per_s": 1e12}, "rec_checked": 16, "rec_bad": 0, "setup_ms": 1.0, "spread": {},
            "kv_wrapped": False, "blocks": blocks}


def test_proved_readings_rescore_with_only_the_filler_fraction(tmp_path):
    S = _module("scoring")
    P = _load_file("pfill_proved")
    rows = []
    for ph, b, l in S.points():
        alone = 10.0 if ph == "decode" else 160.0
        rows.append(_row("step" if ph == "decode" else "prefill", b, l, alone, alone * 1.02))
    (tmp_path / "dscore.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = P.readings(tmp_path)
    assert res["provisional"]["SCORE"] == S.score(P.stats_of(tmp_path))["SCORE"]
    order = sorted(res, key=lambda k: res[k]["filler_fraction"])
    scores = [res[k]["SCORE"] for k in order]
    assert scores == sorted(scores) and scores[0] < scores[-1]
    assert res["proved"]["filler_fraction"] == 0.225
    assert res["in-model"]["status"] == "in-model"


def test_e2m1_arm_adds_only_the_conditional_l64_reading(tmp_path):
    """An e2m1 run (its arm.json's lineage) keeps every s8 reading, the proved 0.225 a slot included (at 8 slots a head
    both tiles' floors are a quarter of a slot), and adds the L = 64 priced reading at twice it, marked conditional."""
    S = _module("scoring")
    P = _load_file("pfill_proved")
    A = _load_file("pfill_arm")
    rows = [_row("step" if ph == "decode" else "prefill", b, l, 10.0, 10.2) for ph, b, l in S.points()]
    (tmp_path / "dscore.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "arm.json").write_text(json.dumps({"lineage": A.PFillRam.__dataclass_fields__["lineage"]
                                                   .default_factory()}))
    s8 = P.readings(tmp_path)
    assert "priced-l64" not in s8
    (tmp_path / "arm.json").write_text(json.dumps({"lineage": A.PFillE2m1Ram.__dataclass_fields__["lineage"]
                                                   .default_factory()}))
    f4 = P.readings(tmp_path)
    assert {k: f4[k] for k in s8} == s8
    assert f4["priced-l64"]["filler_fraction"] == 2 * s8["proved"]["filler_fraction"]
    assert f4["priced-l64"]["status"] == "conditional"
    assert f4["priced-l64"]["SCORE"] > s8["proved"]["SCORE"]
    assert "theorem-k" not in f4


def test_k_arms_add_only_the_theorem_k_reading(tmp_path):
    """A pfill-k or pfill-bU run keeps every reading of a ±1 run, E's proved 0.225 included, and adds Theorem K's c = 1
    at φ = 0.9: proved for pfill-k (with its referee's caveats), projected for pfill-bU (not in the Lean lock yet),
    each at ram's fold share ½; a ±1 run never gets it."""
    S = _module("scoring")
    P = _load_file("pfill_proved")
    A = _load_file("pfill_arm")
    rows = [_row("step" if ph == "decode" else "prefill", b, l, 10.0, 10.2) for ph, b, l in S.points()]
    (tmp_path / "dscore.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def lineage(cls):
        return cls.__dataclass_fields__["lineage"].default_factory()

    (tmp_path / "arm.json").write_text(json.dumps({"lineage": lineage(A.PFillPowerSplit)}))
    pm1 = P.readings(tmp_path)
    assert "theorem-k" not in pm1
    for cls, status in ((A.PFillK, "proved"), (A.PFillBU, "projected")):
        (tmp_path / "arm.json").write_text(json.dumps({"lineage": lineage(cls)}))
        k = P.readings(tmp_path)
        assert {n: k[n] for n in pm1} == pm1 and set(k) - set(pm1) == {"theorem-k"}
        assert (k["theorem-k"]["filler_fraction"], k["theorem-k"]["status"]) == (0.45, status)
        assert k["theorem-k"]["SCORE"] > k["proved"]["SCORE"]


@pytest.mark.parametrize(("attr", "base", "nc", "flag", "noise"), [("PFillBUDuo", "PFillBU", 2, None, "bitsU"),
                                                                   ("PFillBULi", "PFillBU", 4, "-DPFILL_LI", "bitsU"),
                                                                   ("PFillBUXin13", "PFillBU", 4, "-DPFILL_XIN13", "bitsU"),
                                                                   ("PFillKDuo", "PFillK", 2, None, "bits01")])
def test_fold_arms_change_only_the_receipt_and_read_every_word(monkeypatch, tmp_path, attr, base, nc, flag, noise):
    """pfill-bU's and pfill-k's duo (NC 2) arms and pfill-bU's r9 arms (NC 4 built with -DPFILL_LI or -DPFILL_XIN13) are
    their ram arm with only the fold changed; ram's receipt certifies word pairs (Q = a0 d0 + d1), so its count readings
    are half theirs: E 0.1125 against 0.225, K 0.45 against 0.9. The r9
    folds' readings are marked provisional while they are refereed."""
    A, S = _module("arms"), _module("scoring")
    P = _load_file("pfill_proved")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "unset")
    ram = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{base}")
    flags = __import__("os").environ["NVCC_APPEND_FLAGS"]
    arm = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{attr}")
    assert A.check_arm(arm, record=True) == []
    got = __import__("os").environ["NVCC_APPEND_FLAGS"].split()
    assert sorted(got) == sorted(flags.split() + ([flag] if flag else []))
    assert arm.filler == replace(ram.filler, NC=nc) and arm.filler.noise == noise
    assert arm.sources == ram.sources and arm.lineage["parent"] == ram.name
    for ph, b, l in S.points():
        assert A.placed(arm, ph, b, l) == A.placed(ram, ph, b, l)
    rows = [_row("step" if ph == "decode" else "prefill", b, l, 10.0, 10.2) for ph, b, l in S.points()]
    (tmp_path / "dscore.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = {}
    for a in (ram, arm):
        (tmp_path / "arm.json").write_text(json.dumps({"lineage": a.lineage}))
        got[a.name] = P.readings(tmp_path)
    r, f = got[ram.name], got[arm.name]
    assert (f["proved"]["filler_fraction"], f["theorem-k"]["filler_fraction"]) == (0.225, 0.9)
    assert (r["proved"]["filler_fraction"], r["theorem-k"]["filler_fraction"]) == (0.1125, 0.45)
    assert r["provisional"] == f["provisional"] and f["theorem-k"]["SCORE"] > r["theorem-k"]["SCORE"]
    assert f["proved"]["status"] == ("proved, fold provisional" if flag else "proved") and r["proved"]["status"] == "proved"
    assert f["provisional"]["status"] == "provisional"
