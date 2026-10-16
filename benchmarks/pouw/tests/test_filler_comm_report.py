"""filler_comm/report.py: a time is the median over reps of the slowest rank's rep, shares are of one GPU's int8 peak at the
clock read, the fill without partial-tile credit scales by the finished tiles' steps, NCCL's log lines between the records
are skipped, and a point whose KV cache wrapped is flagged."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("tpcomm_report",
                                               Path(__file__).resolve().parents[1] / "filler_comm" / "report.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

MHZ = 2000.0
PEAK = 1024 * 188 * MHZ * 1e6


def _rec(rank, arm, reps, served=0.0, fill_steps=0.0, full_steps=None):
    r = {"mode": "tp", "tp": 2, "rank": rank, "B": 32, "L": 2048, "arm": arm, "mhz": MHZ, "served_macs": served,
         "fill_steps_rep": fill_steps, "fill_macs_rep": fill_steps * 4 * 4096.0, "reps": [[ms, MHZ, 300] for ms in reps]}
    if arm == "phase":
        r.update(rec_checked=16, rec_bad=0)
        if full_steps is not None:
            r["fill_steps_full_rep"] = full_steps
    return json.dumps(r)


def _write(tmp_path, wrapped):
    d = tmp_path / "set0"
    d.mkdir()
    (d / "settings.txt").write_text("NCCL_CGA_CLUSTER_SIZE=1 TPC_RESERVE=8\n")
    served = 1e11
    # rank 1 is the slower rank on two of base's three reps: the median of per-rep maxima is 41, either rank's alone 40
    ranks = {
        0: [_rec(0, "base", [40, 40, 40], served), _rec(0, "nocomm", [30, 30, 30], served), _rec(0, "comm", [8, 8, 8]),
            _rec(0, "phase", [42, 42, 42], served, 1e7, 4e6)],
        1: [_rec(1, "base", [41, 41, 39], served), _rec(1, "nocomm", [30, 30, 30], served), _rec(1, "comm", [8, 9, 8]),
            _rec(1, "phase", [42, 42, 42], served, 1e7, 4e6)],
    }
    for r, lines in ranks.items():
        body = ["host:1:1 [0] NCCL INFO Channel 00 : 0[4] -> 1[5] via SHM/direct",
                json.dumps({"mode": "pf_gate", "rank": r, "ok": True})] + lines
        body.append(json.dumps({"mode": "point", "tp": 2, "rank": r, "B": 32, "L": 2048, "kv_wrapped": wrapped}))
        (d / f"rank{r}.jsonl").write_text("\n".join(body) + "\n")
    return served


def test_shares_from_the_slowest_rank(tmp_path):
    served = _write(tmp_path, wrapped=False)
    (row,) = R.rows(str(tmp_path))
    assert row["base_ms"] == 41 and row["nocomm_ms"] == 30 and row["comm_ms"] == 8 and row["phase_ms"] == 42
    assert row["served"] == pytest.approx(served / 41e-3 / PEAK)
    assert row["idle"] == pytest.approx(1 - row["served"])
    assert row["comm_exposed"] == pytest.approx(11 / 41)
    assert row["slowdown"] == pytest.approx(42 / 41)
    assert row["fill"] == pytest.approx(1e7 * 4 * 4096 / 42e-3 / PEAK)
    assert row["fill_finished"] == pytest.approx(row["fill"] * 0.4)
    assert row["rec_bad"] == 0 and row["rec_checked"] == 32
    assert row["gates_ok"] is True and row["kv_wrapped"] is False


def test_wrapped_kv_is_flagged(tmp_path, capsys):
    _write(tmp_path, wrapped=True)
    (row,) = R.rows(str(tmp_path))
    assert row["kv_wrapped"] is True
    sys.argv = ["report.py", str(tmp_path)]
    R.main()
    assert "2048 (KV wrapped)" in capsys.readouterr().out
