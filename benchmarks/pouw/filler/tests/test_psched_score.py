"""psched's arm-side runner (benchmarks/pouw/filler/psched_score.py) on synthetic out dirs: the locked `score.py report`
scores pscore's rows (the phase filler's work counted in FILL chunks) unchanged, the runner's two added gates (the phase
filler's DRAM share against the same counter's control, and its receipts and tile counts) void the SCORE when they fail,
and the arm (psched_arm.py) places a phase filler only beside a split, as a pure function of the point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psched_score as P

A, S, SC = P.A, P.S, P.SC
ARM = "benchmarks/pouw/filler/psched_arm.py:PhaseSched"
PBIT = 1 << 62  # dcap.cu's: the phase filler's tile ids carry it, its rounds' tag


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


def block(alone_ms, arm_ms, chunks, solo=None):
    arm = {"window_ms": 3 * arm_ms, "reps": [[arm_ms, 2000.0, 300.0]] * 3}
    b = {"order": "alone,arm,solo", "alone": {"window_ms": 3 * alone_ms, "reps": [[alone_ms, 2000.0, 300.0]] * 3},
         "arm": arm}
    if chunks is not None:
        arm["chunks"] = chunks
        b["solo"] = {"window_ms": 3 * arm_ms, "chunks": chunks if solo is None else solo, "reps": []}
    return b


def a_round(point, slots, slot, tag=0):
    """A round record as rounds.cuh prints it: the scorer's range for the slot, its own coins."""
    rid = point * slots + slot
    lo = rid << S.RANGE_LOG2
    c0, c1 = S.smix(2 * rid + 1 + (tag >> 40)), S.smix(2 * rid + 2 + (tag >> 40))
    rd = {"id": rid, "slot": slot, "range": [lo, (rid + 1) << S.RANGE_LOG2], "tiles": [lo, lo + 4000],
          "coins": [f"{c0:016x}", f"{c1:016x}"], "seed0": f"{S.dseed(c0, c1, tag | lo):016x}", "checked": [lo + 3000]}
    return {**rd, "tag": f"{tag:016x}"} if tag else rd


def v2_rounds(r, phase):
    """dscore's rounds, or pscore's with the phase filler's tagged ones beside the split's, POINT_SLOTS (256) a point."""
    n, pt = len(r["blocks"]), r["point"]
    r["rounds"] = n
    r["slots"] = 256
    r["alone_fill"]["round"] = a_round(pt, r["slots"], 0)
    if phase:
        r["alone_fill"]["phase_round"] = a_round(pt, r["slots"], 1, PBIT)
    for i, b in enumerate(r["blocks"]):
        off = 2 if phase else 1
        b["arm"]["round"] = a_round(pt, r["slots"], off + i)
        b["solo"]["round"] = a_round(pt, r["slots"], off + (2 if phase else 1) * n + i)
        if phase:
            b["arm"]["phase_round"] = a_round(pt, r["slots"], 2 + n + i, PBIT)
            b["solo"]["phase_round"] = a_round(pt, r["slots"], 2 + 3 * n + i, PBIT)


def write_run(tmp: Path, *, phase_dram=("Kbyte", "20"), tiles_short=0, phase_bad=0, spec=ARM,
              sgates=None) -> tuple[Path, object]:
    arm = A.load(spec)
    out = tmp / arm.name
    out.mkdir(parents=True)
    (out / "arm.json").write_text(json.dumps({**SC.arm_doc(arm), "spec": ARM}))
    (out / "coins.txt").write_text("ab" * 16)
    (out / "commit.txt").write_text("c" * 40)
    rows = []
    for i, (ph, b, l) in enumerate(S.points()):
        p, q = A.placed(arm, ph, b, l), P.phase_of(arm, ph, b, l)
        alone = 10.0 if ph == "decode" else 160.0
        fill = p.place != "none"
        r = {"mode": "dscore", "kind": "step" if ph == "decode" else "prefill", "B": b, "L": l, "place": p.place,
             "fill_sms": p.fill_sms, "chunk_macs": 1e9, "alone_fill": {"mac_per_s": 1e12 if fill else 0.0},
             "rec_checked": 32 if fill else 0, "rec_bad": 0, "setup_ms": 1.0, "spread": {}, "kv_wrapped": False,
             "point": i, "blocks": [block(alone, alone * 1.02, (20.5 if q else 10) if fill else None) for _ in range(4)]}
        if q:
            r |= {"driver": "pscore", "rec_bad": phase_bad,
                  "phase": {"W": q.W, "NC": q.NC, "S": q.S, "tiles": q.tiles, "tiles_short": tiles_short}}
        if fill:
            v2_rounds(r, q is not None)
        rows.append(r)
    (out / "dscore.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ctl = {"mode": "dscore", "kind": "step", "B": 32, "L": 8192, "place": "split", "fill_sms": 124, "chunk_macs": 1e9,
           "alone_fill": {"mac_per_s": 1e12}, "rec_checked": 32, "rec_bad": 0, "setup_ms": 1.0, "spread": {},
           "kv_wrapped": False, "S": 2048, "cheat_S": S.cheat_s(2048), "point": len(rows),
           "blocks": [block(10.0, 10.2, 11, solo=10) for _ in range(4)]}
    (out / "control.jsonl").write_text(json.dumps(ctl) + "\n")
    fillers = [arm.filler] + sorted({P.filler_of(arm, ph, b, l) for ph, b, l in S.points()
                                     if A.placed(arm, ph, b, l).place != "none"} - {arm.filler}, key=str)
    sg = [{"mode": "sgate", "W": f.W, "NC": f.NC, "S": f.S, "tiles": (1 << f.chunk_log2) // (f.W * f.NC * f.S), "ok": True}
          for f in fillers] if sgates is None else sgates
    (out / "sgate.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sg))
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
    assert res["gates"]["phase_receipts"] == {"ok": True, "unrounded": [], "points": n_phase, "checked": 32 * n_phase,
                                              "bad": 0, "tiles_short": 0}
    assert res["gates"]["replay"]["ok"] and res["gates"]["honest"]["control"]["caught"]
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
    assert arm.phase("decode", 1, 2048) is None and A.placed(arm, "decode", 1, 2048) == A.Colocation("split", 108)
    assert S.honesty(arm, {}, S.points())["ok"]


@pytest.mark.parametrize("cls, co_tiles", [("PhaseSchedCo", (8, 4, 128, 8)), ("PhaseSchedCo2", (4, 4, 128, 32))])
def test_co_arm_runs_its_prefill_shape_and_needs_an_sgate_for_it(tmp_path, cls, co_tiles):
    spec = f"benchmarks/pouw/filler/psched_arm.py:{cls}"
    arm = A.load(spec)
    assert not A.check_arm(arm)
    assert A.placed(arm, "prefill", 8192, 0) == A.Colocation("co")
    co = P.filler_of(arm, "prefill", 8192, 0)
    assert co != arm.filler and co.chunk_log2 < arm.filler.chunk_log2
    assert all(P.filler_of(arm, ph, b, l) == arm.filler for ph, b, l in S.points() if P.phase_of(arm, ph, b, l))
    res = P.report(*write_run(tmp_path / "two", spec=spec))
    assert res["scored"] and res["gates"]["sgates"] == {"ok": True, "rows": 2, "shapes": 2,
                                                        "ran": sorted({(8, 4, 2048, 8), co_tiles}, key=str)}
    res = P.report(*write_run(tmp_path / "one", spec=spec, sgates=[{"mode": "sgate", "W": 8, "NC": 4, "S": 2048,
                                                                       "tiles": 8, "ok": True}]))
    assert not res["gates"]["sgates"]["ok"] and res["SCORE"] is None and res["score_if_gated"] is not None
