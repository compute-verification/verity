"""comm/report.py: a time is the median over reps of the slowest rank's rep, shares are of one GPU's int8 peak at the
clock read, the fill without partial-tile credit scales by the finished tiles' steps, NCCL's log lines between the records
are skipped, and a point whose KV cache wrapped is flagged."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("tpcomm_report",
                                               Path(__file__).resolve().parents[1] / "comm" / "report.py")
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


def _write_pfill(tmp_path, *, mma="IMMA.16832.S8.S8", app="2430", slow=41):
    """A pfill-k run: MMA-counted fill, every run-level gate's file."""
    d = tmp_path / "set0"
    d.mkdir()
    (d / "settings.txt").write_text("NCCL_CGA_CLUSTER_SIZE=1 TPC_S=64\n")
    served = 1e11
    for r in (0, 1):
        def rec(arm, reps, **kw):
            x = {"mode": "tp", "tp": 2, "rank": r, "B": 32, "L": 2048, "arm": arm, "mhz": MHZ, "served_macs": served,
                 "reps": [[ms, MHZ, 300] for ms in reps], **kw}
            return json.dumps(x)
        lines = [json.dumps({"mode": "pf_gate", "kind": k, "rank": r, "ok": True}) for k in ("fixed", "up", "cut")]
        lines += [rec("base", [40, 40, 40]), rec("nocomm", [30, 30, 30]), rec("comm", [8, 8, 8]),
                  rec("phase", [slow] * 3, fill_mmas_rep=4e7, fill_macs_rep=4e7 * 4096.0, fill_mmas_full_rep=1e7,
                      rec_checked=16, rec_bad=0, whole_checked=8, warps_checked=8, shape_bad=0, cut_warps=2000,
                      fold_nc=2, fold_s=64, fold_xin=0)]
        (d / f"rank{r}.jsonl").write_text("\n".join(lines) + "\n")
    (tmp_path / "sass.txt").write_text(f"     16 {mma}\n")
    (tmp_path / "alone.jsonl").write_text(json.dumps({"mode": "alone", "ok": True, "share_gpu": 0.75,
                                                      "share_fill_sms": 0.78}) + "\n")
    hdr = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    (tmp_path / "ncu-filler.csv").write_text(hdr + "".join(
        f'"{i}","void kfill_k<4>(...)","dram__bytes.sum","Kbyte","20"\n'
        f'"{i}","void kfill_k<4>(...)","gpu__time_duration.sum","msecond","2.4"\n' for i in range(3)))
    (tmp_path / "ncu-control.csv").write_text(hdr + "".join(
        f'"{i}","bw_read(...)","dram__bytes.sum","Gbyte","1.075"\n'
        f'"{i}","bw_read(...)","gpu__time_duration.sum","msecond","0.7"\n' for i in range(3)))
    (tmp_path / "gpu.txt").write_text(f"4, GPU-x, 2092, 2430, 300, 600, {app}, 2430, 0x0\n" * 2)
    return served


def test_pfill_fill_counts_mmas_and_reads_k(tmp_path):
    _write_pfill(tmp_path)
    (row,) = R.rows(str(tmp_path))
    assert row["fill"] == pytest.approx(4e7 * 4096 / 41e-3 / PEAK)
    assert row["fold"] == "duo"
    assert row["fill_k"] == pytest.approx(0.9 * row["fill"])  # duo binds every word (provisional: proved up to 4 words)
    assert row["coverage_base"] == pytest.approx(0.225 * row["served"])
    assert row["coverage"] == pytest.approx(0.225 * row["served"] * 40 / 41 + row["fill_k"])
    assert row["fill_finished"] == pytest.approx(0.25 * row["fill"])
    assert row["idle_off_comm"] == pytest.approx(1 - row["served"] - 10 / 40)
    g = R.gates(str(tmp_path), [row])
    assert g["all_ok"], g
    assert g["dram"]["share_of_hbm"] == pytest.approx(20e3 / 2.4e-3 / (1.075e9 / 0.7e-3))


@pytest.mark.parametrize("kw, gate", [({"mma": "HMMA.16816.F32"}, "sass"), ({"app": "1800"}, "clocks"),
                                      ({"slow": 43}, "slowdown")])
def test_pfill_gates_fail(tmp_path, kw, gate):
    _write_pfill(tmp_path, **kw)
    g = R.gates(str(tmp_path), list(R.rows(str(tmp_path))))
    assert not g[gate]["ok"] and not g["all_ok"]


def test_proved_fraction_by_fold():
    # duo binds every word (provisional: its binding is proved for runs of one to four words,
    # DuoQuadNoKernel; five or more words and the served receipt's projected form are open), ram binds word pairs only
    # (half); r9's XOR family (LI, XIN13) reads at ram's proved half, provisional, until the select-charging lemma lands
    assert R.proved_fraction(2, False, 64) == pytest.approx(0.9)
    assert R.proved_fraction(4, False, 64) == pytest.approx(0.45)             # round 3's rem fill K was 2x this
    assert R.proved_fraction(4, True, 64) == pytest.approx(0.45)
    assert R.proved_fraction(None, False, None) == pytest.approx(0.9)         # no fold recorded: word-binding reading
    assert R.fold_label(2, False) == "duo"
    assert R.fold_label(4, False) == "ram"
    assert R.fold_label(4, True) == "xin13/li(prov)"


def _write_defer(tmp_path, nc, xin, store_bytes=25e9):
    d = tmp_path / "set0"
    d.mkdir()
    (d / "settings.txt").write_text("NCCL_CGA_CLUSTER_SIZE=1 TPC_S=64 TPC_DT=8192\n")
    served = 1e11
    for r in (0, 1):
        def rec(arm, reps, **kw):
            return json.dumps({"mode": "tp", "tp": 2, "rank": r, "B": 256, "L": 2048, "arm": arm, "mhz": MHZ,
                               "served_macs": served, "reps": [[ms, MHZ, 300] for ms in reps], **kw})
        lines = [rec("base", [80, 80, 80]),
                 rec("defer", [88, 88, 88], fill_macs_rep=1e7, defer_demand_macs=served, defer_gap_macs_rep=served,
                     defer_drain_macs_rep=0.0, dcap=10, dSd=64, dT=8192, defer_store_bytes=store_bytes,
                     fold_nc=nc, fold_s=64, fold_xin=(1 if xin else 0),
                     dp_whole_checked=32, dp_whole_bad=0, dp_whole_control_equal=0)]
        (d / f"rank{r}.jsonl").write_text("\n".join(lines) + "\n")
    return served


def test_defer_fill_k_and_storage_duo(tmp_path):
    _write_defer(tmp_path, nc=2, xin=False)
    (row,) = R.rows(str(tmp_path))
    assert row["defer_fold"] == "duo"
    assert row["defer_fill_k"] == pytest.approx(row["defer_fill"] * 0.9)
    assert row["defer_gap_share"] == pytest.approx(1.0) and row["defer_overflow_share"] == pytest.approx(0.0)
    assert row["defer_store_gib"] == pytest.approx(25e9 / 2**30)
    assert row["defer_slowdown"] == pytest.approx(88 / 80)


def test_defer_fill_k_xin_is_provisional(tmp_path):
    _write_defer(tmp_path, nc=4, xin=True)
    (row,) = R.rows(str(tmp_path))
    assert row["defer_fold"] == "xin13/li(prov)"
    assert row["defer_fill_k"] == pytest.approx(row["defer_fill"] * 0.45)
