"""dbw.py: the prover's calibration beside Dither serving (`registered`), and the report's rows from it."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

DITHER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DITHER))
import dbw  # noqa: E402

MS = 10 ** 6


class FakeSide:
    """A side whose `ready` comes once `serve` has run `after` times (or whose `expect` fails at once)."""

    def __init__(self, after: int = 3, error: str | None = None) -> None:
        self.after, self.error, self.served = after, error, 0
        self.enough = threading.Event()

    def expect(self, kind: str) -> dict:
        assert kind == "ready"
        if self.error:
            raise dbw.SideError(self.error)
        self.enough.wait()
        return {"event": "ready", "t_ns": 5}

    def serve(self) -> dict:
        self.served += 1
        if self.served >= self.after:
            self.enough.set()
        return {"lo": self.served, "hi": self.served + 1}


def test_registered_serves_until_ready():
    side = FakeSide(after=3)
    ready, blocks = dbw.registered(side, side.serve)
    assert ready == {"event": "ready", "t_ns": 5}
    assert len(blocks) == side.served >= 3
    assert blocks[0] == {"lo": 1, "hi": 2}


def test_registered_raises_the_sides_error():
    side = FakeSide(error="the side exited (3)")
    with pytest.raises(dbw.SideError, match="exited"):
        dbw.registered(side, side.serve)


def side_doc(i: int, t: int, rows: list[dict]) -> dict:
    """One side's doc over two reps, its on blocks inside their sessions' passes, the dither arm 1.5 times slower on."""
    blocks = []
    sessions = {0: {"passes_ns": [[t, t + 20 * MS]], "accepted": True, "failure": None, "pace": None,
                    "memory": {"share": 1.0}, "paced": {}}}
    for r in range(2):
        for phase, k in (("off", None), ("on", 1 + r)):
            lo = t
            for arm, base in (("exact", 7.0), ("dither-x", 18.0)):
                tp = base * (1.5 if phase == "on" and arm == "dither-x" else 1.0)
                rows.append({"op": "decode", "arm": arm, "rep": r, "bw": phase, "side": i, "tpot_b32_ms": tp,
                             "spans_ns": [t, t + 50 * MS, t + 100 * MS]})
                t += 100 * MS
            blocks.append({"rep": r, "phase": phase, "lo": lo, "hi": t, "k": k})
            if k is not None:
                sessions[k] = {"passes_ns": [[lo - 10 * MS, lo + 90 * MS], [lo + 90 * MS, t + 10 * MS]],
                               "accepted": True, "failure": None, "pace": None, "memory": {"share": 1.0}, "paced": {}}
            t += 10 * MS
    sessions[3] = {"passes_ns": [[t, t + 20 * MS]], "accepted": False, "failure": "receipt", "pace": None,
                   "memory": None, "paced": None}
    return {"i": i, "prover_args": "", "plan": dbw.plan(2), "blocks": blocks, "sessions": sessions,
            "ready": {"gib": 1, "blocks": 8, "allowance_ns": 150 * MS, "t_ns": 0,
                      "calibration": {"pass_ns": {"p50": 40 * MS, "max": 90 * MS}}}}


def test_report_calibration_rows_are_neither_timed_nor_uncovered():
    rows: list[dict] = []
    side = side_doc(0, 10 ** 12, rows)
    plain = dbw.report(rows, [side], ["exact", "dither-x"])
    cal_blocks, lo = [], 5 * 10 ** 12
    for j in range(3):
        a = lo + j * 120 * MS
        rows.append({"op": "decode", "arm": "dither-x", "rep": 0, "bw": "cal", "side": 0, "tpot_b32_ms": 30.0 + j,
                     "spans_ns": [a + 10 * MS, a + 60 * MS, a + 110 * MS]})
        cal_blocks.append({"rep": 0, "phase": "cal", "lo": a, "hi": a + 120 * MS})
    side["calibrated_beside"] = {"arms": ["dither-x"], "blocks": cal_blocks}
    side["ready"]["t_ns"] = lo + 300 * MS
    got = dbw.report(rows, [side], ["exact", "dither-x"])
    s = got["sides"][0]
    assert got["tpot_b32_ms"] == plain["tpot_b32_ms"]
    assert s["tpot_b32_ms"]["dither-x"]["on_over_off"] == 1.5
    assert s["uncovered"] == {}
    assert s["honest_accepted"] and s["tampered_refused"]
    assert s["calibrated_beside"] == {"arms": ["dither-x"], "ops": 3, "serving_s": 0.36, "ready_at_s": 0.3,
                                      "decoding": round(300 / 360, 4), "tpot_b32_ms": {"dither-x": 31.0}}
    assert plain["sides"][0]["calibrated_beside"] is None
