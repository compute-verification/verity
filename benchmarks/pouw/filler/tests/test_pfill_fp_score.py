"""The combined runner (benchmarks/pouw/filler/pfill_fp_score.py) on synthetic out dirs: each point has one driver
(fscore at a fused step, pscore at a phase filler, the locked dscore elsewhere, never two), the report is the locked
report with both runners' gates, and a failure in either runner's gates voids the SCORE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pfill_fp_score as F

A, S, SC = F.A, F.S, F.SC
ARM = "benchmarks/pouw/filler/pfill_arm.py:PFillBUDuoFPhase3"
PBIT = 1 << 62  # dcap.cu's: the phase filler's tile ids carry it, its rounds' tag


@pytest.fixture(autouse=True)
def _one_arms_module(monkeypatch):
    # as test_psched_score: the arm module imports `arms` when A.load runs it, and its Filler must be check_arm's
    monkeypatch.setitem(sys.modules, "arms", A)
    monkeypatch.setitem(sys.modules, "scoring", S)
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "unset")


def ncu_csv(kernel, rows):
    head = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'
    body = [f'"{i}","{kernel}(args)","{m}","{u}","{v}"' for i, (m, u, v) in enumerate(rows)]
    return "==PROF== Connected\n" + "\n".join([head, *body]) + "\n"


def block(alone_ms, arm_ms, chunks):
    arm = {"window_ms": 3 * arm_ms, "reps": [[arm_ms, 2000.0, 300.0]] * 3, "chunks": chunks}
    return {"order": "alone,arm,solo", "alone": {"window_ms": 3 * alone_ms, "reps": [[alone_ms, 2000.0, 300.0]] * 3},
            "arm": arm, "solo": {"window_ms": 3 * arm_ms, "chunks": chunks, "reps": []}}


def a_round(point, slots, slot, tag=0):
    rid = point * slots + slot
    lo = rid << S.RANGE_LOG2
    c0, c1 = S.smix(2 * rid + 1 + (tag >> 40)), S.smix(2 * rid + 2 + (tag >> 40))
    rd = {"id": rid, "slot": slot, "range": [lo, (rid + 1) << S.RANGE_LOG2], "tiles": [lo, lo + 4000],
          "coins": [f"{c0:016x}", f"{c1:016x}"], "seed0": f"{S.dseed(c0, c1, tag | lo):016x}", "checked": [lo + 3000]}
    return {**rd, "tag": f"{tag:016x}"} if tag else rd


def rounds(r, phase):
    n, pt = len(r["blocks"]), r["point"]
    r["rounds"], r["slots"] = n, 256
    r["alone_fill"]["round"] = a_round(pt, 256, 0)
    if phase:
        r["alone_fill"]["phase_round"] = a_round(pt, 256, 1, PBIT)
    off = 2 if phase else 1
    for i, b in enumerate(r["blocks"]):
        b["arm"]["round"] = a_round(pt, 256, off + i)
        b["solo"]["round"] = a_round(pt, 256, off + (2 if phase else 1) * n + i)
        if phase:
            b["arm"]["phase_round"] = a_round(pt, 256, 2 + n + i, PBIT)
            b["solo"]["phase_round"] = a_round(pt, 256, 2 + 3 * n + i, PBIT)


def write_run(tmp: Path, *, fgate_ok=True, tiles_short=0) -> tuple[Path, object]:
    arm = A.load(ARM)
    out = tmp / arm.name
    out.mkdir(parents=True)
    (out / "arm.json").write_text(json.dumps({**SC.arm_doc(arm), "spec": ARM}))
    (out / "coins.txt").write_text("ab" * 16)
    (out / "commit.txt").write_text("c" * 40)
    by = F.drivers(arm, S.points())
    rows = []
    for i, (ph, b, l) in enumerate(S.points()):
        p, (d, q) = A.placed(arm, ph, b, l), by[(ph, b, l)]
        alone = 10.0 if ph == "decode" else 160.0
        r = {"mode": "dscore", "kind": "step" if ph == "decode" else "prefill", "B": b, "L": l, "place": p.place,
             "fill_sms": p.fill_sms, "chunk_macs": 1e9, "alone_fill": {"mac_per_s": 1e12}, "rec_checked": 32,
             "rec_bad": 0, "setup_ms": 1.0, "spread": {}, "kv_wrapped": False, "point": i,
             "blocks": [block(alone, alone * 1.02, 20.5 if d == "pscore" else 10) for _ in range(4)]}
        if d == "pscore":
            r |= {"driver": "pscore", "phase": {"W": q.W, "NC": q.NC, "S": q.S, "tiles": q.tiles,
                                                "tiles_short": tiles_short}}
        elif d == "fscore":
            r |= {"driver": "fscore", "step_sms": 188 - p.fill_sms,
                  "fused": {"slots_a_step": 0, "steps_bad": 0, "out_mismatches": 0, "ctl_equal": 0, "out_rows": 4}}
        rounds(r, d == "pscore")
        rows.append(r)
    (out / "dscore.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ctl = {"mode": "dscore", "kind": "step", "B": 32, "L": 8192, "place": "split", "fill_sms": 124, "chunk_macs": 1e9,
           "alone_fill": {"mac_per_s": 1e12}, "rec_checked": 32, "rec_bad": 0, "setup_ms": 1.0, "spread": {},
           "kv_wrapped": False, "S": 2048, "cheat_S": S.cheat_s(2048), "point": len(rows),
           "blocks": [{**block(10.0, 10.2, 11), "solo": {"window_ms": 30.6, "chunks": 10, "reps": []}}
                      for _ in range(4)]}
    (out / "control.jsonl").write_text(json.dumps(ctl) + "\n")
    f = arm.filler
    (out / "sgate.jsonl").write_text(json.dumps({"mode": "sgate", "W": f.W, "NC": f.NC, "S": f.S,
                                                 "tiles": (1 << f.chunk_log2) // (f.W * f.NC * f.S), "ok": True}) + "\n")
    (out / "dgate.jsonl").write_text(json.dumps({"mode": "dgate", "NC": 4, "S": 2048, "ok": True}) + "\n" + json.dumps(
        {"mode": "pgate", "NC": 4, "S": 64, "mismatches": 0, "control_equal": 0, "ok": True}) + "\n")
    (out / "gate.jsonl").write_text(json.dumps({"mode": "gate", "ok": True}) + "\n"
                                    + json.dumps({"mode": "fgate", "M": 32, "linear": "qkv", "FW": 0, "ok": fgate_ok}) + "\n")
    (out / "ncu-control.csv").write_text(ncu_csv("bw_read", [("dram__bytes.sum", "Gbyte", "1.08"),
                                                             ("gpu__time_duration.sum", "msecond", "0.72")]))
    (out / "ncu-filler.csv").write_text(ncu_csv("void dfill_k<4>", [("dram__bytes.sum", "Kbyte", "200"),
                                                                    ("gpu__time_duration.sum", "msecond", "1.12")]))
    (out / "ncu-phase.csv").write_text(ncu_csv("void dfill_p<4>", [("dram__bytes.sum", "Kbyte", "20"),
                                                                   ("gpu__time_duration.sum", "msecond", "0.5")]))
    (out / "ncu-fused.csv").write_text(ncu_csv("void flin_k<32>", [("dram__bytes.sum", "Kbyte", "500")] * 64
                                               + [("gpu__time_duration.sum", "msecond", "0.02")] * 64))
    (out / "smi.csv").write_text("2092, 3090, 410.5, 600.00, 2430, 2430, 0x0000000000000000\n")
    return out, arm


def test_each_point_has_one_driver_and_the_report_has_both_runners_gates(tmp_path):
    out, arm = write_run(tmp_path)
    by = F.drivers(arm, S.points())
    assert {d for d, _ in by.values()} == {"fscore", "pscore", "dscore"}
    assert sorted(k for k, (d, _) in by.items() if d == "pscore") == [("decode", 256, 2048), ("decode", 256, 3072)]
    res = F.report(out, arm)
    locked = json.loads((out / "score-locked.json").read_text())
    assert res["scored"] and locked["scored"] and res["SCORE"] == pytest.approx(locked["SCORE"])
    g = res["gates"]
    assert all(g[k]["ok"] for k in F.FUSED_GATES + F.PHASE_GATES) and g["ok"]
    assert g["fused_receipts"]["points"] == sum(d == "fscore" for d, _ in by.values())
    assert g["phase_receipts"]["points"] == 2
    assert res["points"]["decode 256/2048"]["fill_rate"] == pytest.approx(20.5 * 1e9 / 30.6e-3)
    assert json.loads((out / "score.json").read_text())["SCORE"] == res["SCORE"]


@pytest.mark.parametrize("kw, gate", [({"fgate_ok": False}, "fused_exact"), ({"tiles_short": 1}, "phase_receipts")])
def test_either_runners_gate_voids_the_score(tmp_path, kw, gate):
    out, arm = write_run(tmp_path, **kw)
    res = F.report(out, arm)
    assert not res["gates"][gate]["ok"] and not res["gates"]["ok"]
    assert res["SCORE"] is None and not res["scored"] and res["score_if_gated"] is not None


def test_a_point_with_both_a_fused_step_and_a_phase_filler_is_refused():
    arm = A.load(ARM)
    step = arm.fused("decode", 128, 2048)
    arm.fused = lambda ph, b, l: step if ph == "decode" else None
    with pytest.raises(A.ArmError, match="both a fused step and a phase filler"):
        F.drivers(arm, S.points())
