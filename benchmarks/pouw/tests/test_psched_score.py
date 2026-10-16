"""HC-SCHED's arm-side runner (benchmarks/pouw/filler/psched_score.py) on synthetic out dirs: the locked `score.py report`
scores pscore's rows (the phase filler's work counted in FILL chunks) unchanged, the runner's two added gates (the phase
filler's DRAM share against the same counter's control, and its receipts and tile counts) void the SCORE when they fail,
and the arm (psched_arm.py) places a phase filler only beside a split, as a pure function of the point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filler"))
import psched_score as P  # noqa: E402

A, S, SC = P.A, P.S, P.SC
ARM = "benchmarks/pouw/filler/psched_arm.py:PhaseSched"


@pytest.fixture(autouse=True)
def _one_arms_module(monkeypatch):
    # test_filler_score loads its own copies of arms and scoring into sys.modules; the arm module imports `arms` when
    # A.load runs it, and its Filler must be the class check_arm tests against
    monkeypatch.setitem(sys.modules, "arms", A)
    monkeypatch.setitem(sys.modules, "scoring", S)


def ncu_csv(kernel, rows):
    head = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'
    body = [f'"{i}","{kernel}(args)","{m}","{u}","{v}"' for i, (m, u, v) in enumerate(rows)]
    return "==PROF== Connected\n" + "\n".join([head, *body]) + "\n"


def block(alone_ms, arm_ms, chunks):
    arm = {"window_ms": 3 * arm_ms, "reps": [[arm_ms, 2000.0, 300.0]] * 3}
    if chunks is not None:
        arm["chunks"] = chunks
    return {"order": "alone,arm", "alone": {"window_ms": 3 * alone_ms, "reps": [[alone_ms, 2000.0, 300.0]] * 3},
            "arm": arm}


def write_run(tmp: Path, *, phase_dram=("Kbyte", "20"), tiles_short=0, phase_bad=0) -> tuple[Path, object]:
    arm = A.load(ARM)
    out = tmp / arm.name
    out.mkdir()
    (out / "arm.json").write_text(json.dumps({**SC.arm_doc(arm), "spec": ARM}))
    (out / "coins.txt").write_text("ab" * 16)
    (out / "commit.txt").write_text("c" * 40)
    rows = []
    for ph, b, l in S.points():
        p, q = A.placed(arm, ph, b, l), P.phase_of(arm, ph, b, l)
        alone = 10.0 if ph == "decode" else 160.0
        fill = p.place != "none"
        r = {"mode": "dscore", "kind": "step" if ph == "decode" else "prefill", "B": b, "L": l, "place": p.place,
             "fill_sms": p.fill_sms, "chunk_macs": 1e9, "alone_fill": {"mac_per_s": 1e12 if fill else 0.0},
             "rec_checked": 32 if fill else 0, "rec_bad": 0, "setup_ms": 1.0, "spread": {}, "kv_wrapped": False,
             "blocks": [block(alone, alone * 1.02, (20.5 if q else 10) if fill else None) for _ in range(4)]}
        if q:
            r |= {"driver": "pscore", "rec_bad": phase_bad,
                  "phase": {"W": q.W, "NC": q.NC, "S": q.S, "tiles": q.tiles, "tiles_short": tiles_short}}
        rows.append(r)
    (out / "dscore.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "sgate.jsonl").write_text(json.dumps({"mode": "sgate", "ok": True}) + "\n")
    (out / "dgate.jsonl").write_text(json.dumps({"mode": "dgate", "NC": 4, "S": 2048, "ok": True}) + "\n" + json.dumps(
        {"mode": "pgate", "NC": 4, "S": 128, "mismatches": 0, "control_equal": 0, "ok": True}) + "\n")
    (out / "gate.jsonl").write_text(json.dumps({"mode": "gate", "ok": True}) + "\n")
    (out / "ncu-control.csv").write_text(ncu_csv("bw_read", [("dram__bytes.sum", "Gbyte", "1.08"),
                                                             ("gpu__time_duration.sum", "msecond", "0.72")]))
    (out / "ncu-filler.csv").write_text(ncu_csv("void dfill_k<4>", [("dram__bytes.sum", "Kbyte", "200"),
                                                                    ("gpu__time_duration.sum", "msecond", "1.12")]))
    (out / "ncu-phase.csv").write_text(ncu_csv("void dfill_p<4>", [("dram__bytes.sum", *phase_dram),
                                                                   ("gpu__time_duration.sum", "msecond", "0.5")]))
    (out / "smi.csv").write_text("2092, 3090, 410.5, 600.00, 2430, 2430, 0x0000000000000000\n")
    return out, arm


def test_locked_report_scores_the_phase_rows(tmp_path):
    out, arm = write_run(tmp_path)
    res = P.report(out, arm)
    locked = json.loads((out / "score-locked.json").read_text())
    assert res["scored"] and locked["scored"]
    assert res["SCORE"] == pytest.approx(locked["SCORE"])
    assert res["points"]["decode 256/2048"]["fill_rate"] == pytest.approx(20.5 * 1e9 / 30.6e-3)
    assert res["points"]["decode 1/2048"]["fill_rate"] == pytest.approx(10 * 1e9 / 30.6e-3)
    n_phase = sum(1 for ph, b, l in S.points() if P.phase_of(arm, ph, b, l))
    assert res["gates"]["phase_receipts"] == {"ok": True, "points": n_phase, "checked": 32 * n_phase, "bad": 0,
                                              "tiles_short": 0}
    assert res["gates"]["phase_dram"]["ok"]
    assert res["gates"]["phase_dram"]["share_of_hbm"] == pytest.approx((20e3 / 0.5e-3) / (1.08e9 / 0.72e-3))


@pytest.mark.parametrize("kw, gate", [({"phase_dram": ("Mbyte", "100")}, "phase_dram"),
                                      ({"tiles_short": 1}, "phase_receipts"), ({"phase_bad": 1}, "phase_receipts")])
def test_phase_gates_void_the_score(tmp_path, kw, gate):
    out, arm = write_run(tmp_path, **kw)
    res = P.report(out, arm)
    assert not res["gates"][gate]["ok"] and not res["gates"]["ok"]
    assert res["SCORE"] is None and not res["scored"] and res["score_if_gated"] is not None


def test_arm_places_a_phase_filler_only_beside_a_split():
    arm = A.load(ARM)
    for ph, b, l in S.points():
        p, q = A.placed(arm, ph, b, l), P.phase_of(arm, ph, b, l)
        assert (p, q) == (A.placed(arm, ph, b, l), P.phase_of(arm, ph, b, l))
        if q is not None:
            assert p.place == "split" and q.S % 8 == 0 and q.S > 8 and q.tiles >= 1 and q.NC in (1, 2, 4)
    assert A.placed(arm, "prefill", 8192, 0) == A.NONE and arm.phase("prefill", 8192, 0) is None
    assert arm.phase("decode", 1, 2048) is None and A.placed(arm, "decode", 1, 2048) == A.Placement("split", 108)
    assert S.honesty(arm, {}, S.points())["ok"]
