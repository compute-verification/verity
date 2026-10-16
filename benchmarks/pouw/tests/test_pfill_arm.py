"""HC-PROVED's scorer arms (benchmarks/pouw/filler/pfill_arm.py) meet the filler arm contract, and place the filler as
FILL v1 does at every suite point; pfill_proved.py's readings are the scorer's arithmetic with only the filler's
certified fraction changed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

FILLER = Path(__file__).resolve().parents[1] / "filler"
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
    lineage score.py's ledger record takes (r20261016-070822-5f55 scored, then refused a 249-character hypothesis)."""
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
        assert arm.filler == A.Filler(W=8, NC=nc, S=256, chunk_log2=19)
        assert arm.filler.S % 32 == 0  # pfill's dfill_make refuses any other S
        assert "benchmarks/pouw/decode_capacity/pfill.cuh" in arm.sources
        for ph, b, l in S.points():
            assert A.placed(arm, ph, b, l) == A.placed(v1, ph, b, l)


def test_power_split_arm_has_a_split_at_every_scored_point(monkeypatch):
    """pfill-power-split is pfill-ram's filler on HC-POWER's table: an exact entry at every scored decode point, and 4
    filler SMs during prefill."""
    A, S = _module("arms"), _module("scoring")
    P = _load_file("pfill_arm")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "unset")
    arm = A.load("benchmarks/pouw/filler/pfill_arm.py:PFillPowerSplit")
    ram = A.load("benchmarks/pouw/filler/pfill_arm.py:PFillRam")
    assert A.check_arm(arm) == []
    assert ledger.one_line(arm.lineage["hypothesis"], "the hypothesis")
    assert (arm.filler, arm.sources, __import__("os").environ["NVCC_APPEND_FLAGS"]) == (ram.filler, ram.sources, "")
    for ph, b, l in S.points(robust=False):
        p = A.placed(arm, ph, b, l)
        assert p == (A.Placement("split", 4) if ph == "prefill" else A.Placement("split", P.POWER_SPLIT[(b, l)]))
    for b, l in S.ROBUST:
        assert A.placed(arm, "decode", b, l) == A.Placement("split", A.point_lookup(P.POWER_SPLIT, b, l))


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
    s8 = P.readings(tmp_path)
    assert "priced-l64" not in s8
    (tmp_path / "arm.json").write_text(json.dumps({"lineage": A.PFillE2m1Ram.__dataclass_fields__["lineage"]
                                                   .default_factory()}))
    f4 = P.readings(tmp_path)
    assert {k: f4[k] for k in s8} == s8
    assert f4["priced-l64"]["filler_fraction"] == 2 * s8["proved"]["filler_fraction"]
    assert f4["priced-l64"]["status"] == "conditional"
    assert f4["priced-l64"]["SCORE"] > s8["proved"]["SCORE"]
