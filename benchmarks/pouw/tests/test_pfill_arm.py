"""HC-PROVED's scorer arms (benchmarks/pouw/filler/pfill_arm.py) meet the filler arm contract, and place the filler as
FILL v1 does at every suite point; pfill_proved.py's readings are the scorer's arithmetic with only the filler's
certified fraction changed."""

from __future__ import annotations

import importlib.util
import json
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


def _load_file(name):
    spec = importlib.util.spec_from_file_location(f"hcproved_{name}", FILLER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pfill_arms_meet_the_contract_and_place_as_fill_v1():
    A, S = _module("arms"), _module("scoring")
    v1 = A.load("fill-v1")
    for attr, nc in (("PFillRam", 4), ("PFillDuo", 2)):
        arm = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{attr}")
        assert A.check_arm(arm) == []
        assert arm.filler == A.Filler(W=8, NC=nc, S=256, chunk_log2=19)
        assert arm.filler.S % 32 == 0  # pfill's dfill_make refuses any other S
        assert "benchmarks/pouw/decode_capacity/pfill.cuh" in arm.sources
        for ph, b, l in S.points():
            assert A.placed(arm, ph, b, l) == A.placed(v1, ph, b, l)


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
