"""comm/report.py: a time is the median over reps of the slowest rank's rep, shares are of one GPU's int8 peak at the
clock read, the fill without partial-tile credit scales by the finished tiles' steps, NCCL's log lines between the records
are skipped, and a point whose KV cache wrapped is flagged."""

from __future__ import annotations

import functools
import json
import sys

import pytest
from conftest import load

R = load("comm/report.py")

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


def _tp(rank, B, served, arm, reps, **kw):
    """A rank's tp record of one arm at batch B, as its line: the arm's reps and what else the run wrote into it."""
    return json.dumps({"mode": "tp", "tp": 2, "rank": rank, "B": B, "L": 2048, "arm": arm, "mhz": MHZ,
                       "served_macs": served, "reps": [[ms, MHZ, 300] for ms in reps], **kw})


SMI = "4, GPU-x, 2092, 2430, 300, 600, {}, 2430, 0x0\n"  # an nvidia-smi row: its application clock, then the default


def _ncu(kernel, unit, *launches, ms):
    """ncu's CSV of a kernel's launches: each one's DRAM bytes, in `unit`, and its `ms` on the GPU."""
    return '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n' + "".join(
        f'"{i}","{kernel}","dram__bytes.sum","{unit}","{b}"\n'
        f'"{i}","{kernel}","gpu__time_duration.sum","msecond","{ms}"\n' for i, b in enumerate(launches))


def _write_pfill(tmp_path, *, mma="IMMA.16832.S8.S8", app="2430", slow=41):
    """A pfill-k run: MMA-counted fill, every run-level gate's file."""
    d = tmp_path / "set0"
    d.mkdir()
    (d / "settings.txt").write_text("NCCL_CGA_CLUSTER_SIZE=1 TPC_S=64\n")
    served = 1e11
    for r in (0, 1):
        rec = functools.partial(_tp, r, 32, served)
        lines = [json.dumps({"mode": "pf_gate", "kind": k, "rank": r, "ok": True}) for k in ("fixed", "up", "cut")]
        lines += [rec("base", [40, 40, 40]), rec("nocomm", [30, 30, 30]), rec("comm", [8, 8, 8]),
                  rec("phase", [slow] * 3, fill_mmas_rep=4e7, fill_macs_rep=4e7 * 4096.0, fill_mmas_full_rep=1e7,
                      rec_checked=16, rec_bad=0, whole_checked=8, warps_checked=8, shape_bad=0, cut_warps=2000,
                      fold_nc=2, fold_s=64, fold_xin=0)]
        (d / f"rank{r}.jsonl").write_text("\n".join(lines) + "\n")
    (tmp_path / "sass.txt").write_text(f"     16 {mma}\n")
    (tmp_path / "alone.jsonl").write_text(json.dumps({"mode": "alone", "ok": True, "share_gpu": 0.75,
                                                      "share_fill_sms": 0.78}) + "\n")
    (tmp_path / "ncu-filler.csv").write_text(_ncu("void kfill_k<4>(...)", "Kbyte", 20, 20, 20, ms=2.4))
    (tmp_path / "ncu-control.csv").write_text(_ncu("bw_read(...)", "Gbyte", 1.075, 1.075, 1.075, ms=0.7))
    (tmp_path / "gpu.txt").write_text(SMI.format(app) * 2)
    return served


def test_each_point_has_its_own_row(tmp_path):
    for name, settings, points in (("set0", "TPC_S=64", {32: 40, 64: 80}), ("set1", "TPC_S=32", {32: 50})):
        d = tmp_path / name
        d.mkdir()
        (d / "settings.txt").write_text(settings + "\n")
        for r in (0, 1):
            lines = [_tp(r, B, 1e11, arm, [ms] * 3)
                     for B, base in points.items() for arm, ms in (("base", base), ("nocomm", base - 10))]
            (d / f"rank{r}.jsonl").write_text("\n".join(lines) + "\n")
    got = [(row["set"], row["settings"], row["B"], row["base_ms"], row["nocomm_ms"]) for row in R.rows(str(tmp_path))]
    assert got == [("set0", "TPC_S=64", 32, 40, 30), ("set0", "TPC_S=64", 64, 80, 70), ("set1", "TPC_S=32", 32, 50, 40)]


def _write_set(tmp_path, ranks):
    """A set whose rank r wrote the lines `ranks[r]`."""
    d = tmp_path / "set0"
    d.mkdir()
    (d / "settings.txt").write_text("TPC_S=64\n")
    for r, lines in enumerate(ranks):
        (d / f"rank{r}.jsonl").write_text("\n".join(lines) + "\n")


def test_a_time_and_a_field_are_medians_over_an_arms_repeats(tmp_path):
    # four repeats of base, three reps each: neither median over the repeats is a repeat's own value
    repeats = ((40, 2000.0), (52, 1900.0), (44, 1960.0), (46, 1920.0))
    _write_set(tmp_path, [[_tp(0, 32, 1e11, "base", [ms + 2, ms, ms - 1], mhz=mhz) for ms, mhz in repeats]])
    (row,) = R.rows(str(tmp_path))
    assert row["base_ms"] == 45
    assert row["served"] == pytest.approx(1e11 / 45e-3 / (1024 * 188 * 1940.0 * 1e6))
    assert row["gates_ok"] is None  # no rank wrote a gate's verdict


def test_a_field_is_the_median_over_every_ranks_record_of_every_repeat(tmp_path):
    # two repeats of base on two ranks, four clocks: their median, 1930, is no rank's, no repeat's and not their mean
    clocks = ((2000.0, 1900.0), (1960.0, 1700.0))  # each rank's, one a repeat
    _write_set(tmp_path, [[_tp(r, 32, 1e11, "base", [40] * 3, mhz=mhz) for mhz in of_rank]
                          for r, of_rank in enumerate(clocks)])
    (row,) = R.rows(str(tmp_path))
    assert row["served"] == pytest.approx(1e11 / 40e-3 / (1024 * 188 * 1930.0 * 1e6))


def test_a_rank_cut_short_times_the_reps_every_rank_has(tmp_path):
    _write_set(tmp_path, [[_tp(0, 32, 1e11, "base", [40, 42, 90])], [_tp(1, 32, 1e11, "base", [41, 43])]])
    (row,) = R.rows(str(tmp_path))
    assert row["base_ms"] == 42  # the median of 41 and 43, the slower rank's time of each rep both ranks have


@pytest.mark.parametrize("failed", [0, 1])
def test_one_failed_gate_fails_the_set_and_only_the_wrapped_point_is_flagged(tmp_path, failed):
    # the wrapped point, and three that each differ from it in one part of (tp, B, L)
    points = ((2, 64, 2048), (2, 32, 2048), (2, 64, 4096), (4, 64, 2048))
    _write_set(tmp_path, [[json.dumps({"mode": "pf_gate", "rank": r, "ok": r != failed}),  # one rank's gate fails
                           *(_tp(r, B, 1e11, "base", [40] * 3, tp=tp, L=L) for tp, B, L in points),
                           json.dumps({"mode": "point", "tp": 2, "rank": r, "B": 64, "L": 2048, "kv_wrapped": True})]
                          for r in (0, 1)])
    got = [(row["tp"], row["B"], row["L"], row["gates_ok"], row["kv_wrapped"]) for row in R.rows(str(tmp_path))]
    assert got == [(*key, False, key == points[0]) for key in sorted(points)]
    assert R.gates(str(tmp_path), [])["pf_gate"]["ok"] is False  # and it fails the run's gate, whichever rank's it is


def test_the_fillers_arms_are_read_at_their_own_clock_time_and_fold(tmp_path):
    # the filler's arms ran at other clocks than base's 2000 MHz: phase at 1800 under the ram fold, commphase at 1900;
    # the point at B 64 has a commphase arm and no comm arm to hold it against
    _write_set(tmp_path, [[_tp(r, 32, 1e11, "base", [40] * 3), _tp(r, 32, 0.0, "comm", [8] * 3),
                           _tp(r, 32, 1e11, "phase", [41] * 3, mhz=1800.0, fill_macs_rep=4e10, fold_nc=4, fold_xin=0),
                           _tp(r, 32, 0.0, "commphase", [10] * 3, mhz=1900.0, fill_macs_rep=2e9),
                           _tp(r, 64, 0.0, "commphase", [10] * 3, fill_macs_rep=2e9)] for r in (0, 1)])
    row, lone = R.rows(str(tmp_path))
    assert row["fill"] == pytest.approx(4e10 / 41e-3 / (0.9 * PEAK))
    assert row["served_phase"] == pytest.approx(1e11 / 41e-3 / (0.9 * PEAK))
    assert (row["fold"], row["fill_k"]) == ("ram", pytest.approx(0.45 * row["fill"]))
    assert row["coverage"] == pytest.approx(0.225 * row["served_phase"] + 0.45 * row["fill"])
    assert row["comm_slowdown"] == pytest.approx(10 / 8)
    assert row["fill_in_comm"] == pytest.approx(2e9 / 10e-3 / (0.95 * PEAK))
    assert "comm_slowdown" not in lone and "fill_in_comm" not in lone


COUNTS = ("rec_bad", "rec_checked", "whole_checked", "warps_checked", "shape_bad")


def test_a_points_receipt_counts_are_sums_over_every_record_of_the_phase(tmp_path):
    # two repeats of phase on two ranks: the four records hold 1, 2, 3 and 4 of each count, and as many cut warps
    def phase(r, n):
        return _tp(r, 32, 1e11, "phase", [41] * 3, fill_macs_rep=4e10, cut_warps=n, **dict.fromkeys(COUNTS, n))
    _write_set(tmp_path, [[_tp(r, 32, 1e11, "base", [40] * 3), phase(r, 1 + r), phase(r, 3 + r)] for r in (0, 1)])
    (row,) = R.rows(str(tmp_path))
    assert [row[k] for k in COUNTS] == [10] * 5 and row["cut_warps"] == 2.5


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


def _failing(tmp_path, rows):
    """The run's gates that are not ok for the points' `rows`, by name; all_ok says whether there is one."""
    g = R.gates(str(tmp_path), rows)
    failing = [k for k, v in g.items() if k != "all_ok" and not v["ok"]]
    assert g["all_ok"] == (not failing)
    return failing


def test_a_run_that_wrote_nothing_passes_no_gate(tmp_path):
    assert _failing(tmp_path, []) == ["pf_gate", "receipts", "slowdown", "sass", "alone", "dram", "clocks"]


SPOILED = [  # the gate that fails, if one does; a file of _write_pfill's run; what the file holds in its place
    ("sass", "sass.txt", "     16 IMMA.16832.S8.S8\n      2 HMMA.16816.F32\n"),  # a second MMA opcode
    ("sass", "sass.txt", "     16 IMMA.16832.S8.S8\nnon-RZ IMMA: 0\nnon-RZ IMMA: 3\n"),  # the second count is not zero
    ("alone", "alone.jsonl", '{"ok": true, "share_gpu": 0.75, "share_fill_sms": 0.78}\n{"ok": false}\n'),
    ("dram", "ncu-control.csv", _ncu("bw_read(...)", "Gbyte", 1.3, 1.3, 1.3, ms=0.7)),  # the control over its 1 GiB
    ("dram", "ncu-control.csv", _ncu("bw_read(...)", "Gbyte", 0.9, 0.9, 0.9, ms=0.7)),  # and under it
    (None, "ncu-control.csv", _ncu("bw_read(...)", "Gbyte", 1.075, 1.3, 1.075, ms=0.7)),  # its median is in range
    ("dram", "ncu-filler.csv", _ncu("void kfill_k<4>(...)", "Kbyte", 20, 80000, 20, ms=2.4)),  # one launch over
    ("clocks", "gpu.txt", SMI.format(2430) + SMI.format(1800)),  # one row's application clock is not its default
]


@pytest.mark.parametrize("gate, file, text", SPOILED)
def test_a_spoiled_file_fails_its_own_gate_and_no_other(tmp_path, gate, file, text):
    _write_pfill(tmp_path)
    (tmp_path / file).write_text(text)
    assert _failing(tmp_path, list(R.rows(str(tmp_path)))) == ([gate] if gate else [])


def test_one_points_bad_receipts_or_slowdown_fail_the_runs_gate(tmp_path):
    _write_pfill(tmp_path)
    (row,) = R.rows(str(tmp_path))
    assert _failing(tmp_path, [row]) == []
    for gate, bad in (("receipts", {"rec_bad": 1}), ("receipts", {"shape_bad": 1}), ("receipts", {"rec_checked": 0}),
                      ("slowdown", {"slowdown": 1.06})):
        assert _failing(tmp_path, [row, {**row, **bad}]) == [gate]  # a second point, one field of its row gone bad


def test_proved_fraction_by_fold():
    # duo binds every word (provisional: its binding is proved for runs of one to four words,
    # DuoQuadNoKernel; five or more words and the served receipt's projected form are open), ram binds word pairs only
    # (half); r9's XOR family (LI, XIN13) reads at ram's proved half, provisional, until the select-charging lemma lands
    assert R.proved_fraction(2, False, 64) == pytest.approx(0.9)
    assert R.proved_fraction(4, False, 64) == pytest.approx(0.45)             # half of duo's: ram binds a pair of words
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
        rec = functools.partial(_tp, r, 256, served)
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


def test_the_deferred_pass_is_read_from_every_record_of_the_defer_arm(tmp_path):
    # two repeats of defer on two ranks at 1800 MHz, beside base at 2000 and dalone: 0.6 of the demand ran in the gaps
    # and 0.3 drained after, nothing was stored, and the four records hold 1, 2, 3 and 4 of each count
    def defer(r, n):
        return _tp(r, 32, 1e11, "defer", [88] * 3, mhz=1800.0, fill_macs_rep=4e10, defer_demand_macs=1e11,
                   defer_gap_macs_rep=6e10, defer_drain_macs_rep=3e10, defer_store_bytes=0, dcap=10, dSd=64, dT=8192,
                   dp_whole_bad=n, dp_whole_control_equal=10 * n, dp_whole_checked=100 * n, fill_shape_bad=n)
    _write_set(tmp_path, [[_tp(r, 32, 1e11, "base", [80] * 3), defer(r, 1 + r), defer(r, 3 + r),
                           _tp(r, 32, 1e11, "dalone", [60] * 3)] for r in (0, 1)])
    (row,) = R.rows(str(tmp_path))
    assert (row["defer_gap_share"], row["defer_overflow_share"]) == pytest.approx((0.6, 0.3))
    assert row["defer_fill"] == pytest.approx(4e10 / 88e-3 / (0.9 * PEAK))
    assert row["defer_store_gib"] is None
    assert (row["dp_whole_bad"], row["dp_whole_checked"], row["defer_fill_bad"]) == (110, 1000, 10)
    assert (row["dcap"], row["dSd"], row["dT"]) == (10, 64, 8192)
    assert (row["dalone_ms"], row["dalone_cost"]) == (60, pytest.approx(60 / 80))


def test_the_deferred_pass_has_two_gates_of_its_own(tmp_path):
    _write_defer(tmp_path, nc=2, xin=False)
    for r in (0, 1):  # rank 1's dp_gate fails: one control came out equal to its receipt
        gate = {"mode": "dp_gate", "rank": r, "ok": r == 0, "bad": 0, "control_equal": r, "q_bad": 0}
        with (tmp_path / "set0" / f"rank{r}.jsonl").open("a") as f:
            f.write(json.dumps(gate) + "\n")
    (row,) = R.rows(str(tmp_path))
    assert row["defer_fill_bad"] == 2  # neither rank's defer record says its fill's shape was checked
    good = {**row, "defer_fill_bad": 0}
    g = R.gates(str(tmp_path), [good])
    assert g["dp_gate"] == {"ok": False, "records": 2, "bad": 1} and g["defer_receipts"]["ok"]
    for bad in ({"dp_whole_bad": 1}, {"defer_fill_bad": 1}):
        assert not R.gates(str(tmp_path), [good, {**good, **bad}])["defer_receipts"]["ok"]
    assert not R.gates(str(tmp_path), [])["defer_receipts"]["ok"]  # no point's deferred receipts were checked
