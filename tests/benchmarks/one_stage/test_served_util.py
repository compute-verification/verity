"""`served_util.py` on a toy Nsight Systems export and step trace: the steady state is the run of full steps, its Tensor
Active mean is the tensor-busy share of the whole step (idle time counting as zero), and the losses (kernels by category,
idle split by the copy running) sum to 100 less that mean."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "one_stage"))
import served_util as U  # noqa: E402  (a script beside the suite, importable only once its dir is on the path)

Z = 1_000 * 10**9                                      # the report's timeline zero, UTC ns
MS = 10**6


def _report(p: Path) -> Path:
    """Six 10 ms spans; in each, an int8 GEMM for 6 ms at 90% tensor, an RMS norm for 2 ms at 0%, then 2 ms with no kernel,
    a device-to-host copy over its first half. Samples every 0.5 ms."""
    db = sqlite3.connect(p)
    db.executescript("""
        CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER);
        CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, metricId INTEGER, metricName TEXT);
        CREATE TABLE GPU_METRICS (timestamp INTEGER, typeId INTEGER, metricId INTEGER, value REAL);
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER, demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER);
    """)
    db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?)", (Z,))
    db.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (0, ?, ?)", [(1, "Tensor Active [Throughput %]"), (2, "SM Active [Throughput %]")])
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "int8_gemm_kernel"), (2, "rms_norm_kernel"), (3, "device_kernel"),
                                                           (4, "void cutlass::device_kernel<Sm90GemmTile>()")])
    for s in range(6):
        b = 10 * s * MS
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 1, 1)", (b, b + 6 * MS))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 2, 2)", (b + 6 * MS, b + 8 * MS))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, 2, ?)", (b + 8 * MS, b + 9 * MS, 10**9))
    for j in range(120):
        t = j * MS // 2 + MS // 4
        off = t % (10 * MS)
        db.execute("INSERT INTO GPU_METRICS VALUES (?, 0, 1, ?)", (t, 90.0 if off < 6 * MS else 0.0))
        db.execute("INSERT INTO GPU_METRICS VALUES (?, 0, 2, ?)", (t, 100.0 if off < 8 * MS else 0.0))
    db.commit()
    return p


def _trace(p: Path, T: int = 8) -> Path:
    """Step k ends at 10 (k + 1) ms; step 0 is not full, steps 1 to 4 are, step 5 is not; one arrival before them."""
    recs = [{"kind": "header", "schema": "verity-vllm/step-trace/v1"}, {"kind": "arrive", "request": "a", "t": (Z + 5 * MS) / 1e9}]
    recs += [{"kind": "step", "step": k, "T": T if 1 <= k <= 4 else T // 2, "t": (Z + 10 * (k + 1) * MS) / 1e9} for k in range(6)]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def test_steady_state_mean_and_losses(tmp_path):
    got = U.analyse(_report(tmp_path / "prof.sqlite"), _trace(tmp_path / "advice_trace.jsonl"), 8, 1830.0, sent_s=(Z + 5 * MS) / 1e9)
    w = got["queued"]
    assert (w["steps"], w["first_step"], w["last_step"]) == (4, 1, 4)
    assert w["step_wall_ms"]["mean"] == pytest.approx(10.0, abs=1e-3)
    assert w["tensor_active_pct"] == pytest.approx(54.0)
    assert w["gpu_busy_pct"] == pytest.approx(80.0)
    assert w["copies"]["d2h"]["GB"] == pytest.approx(4.0) and w["copies"]["d2h"]["busy_pct"] == pytest.approx(10.0)
    loss = {r["label"]: r["loss_pts"] for r in w["losses"]}
    assert loss == pytest.approx({"elementwise": 20.0, "idle, D2H copy": 10.0, "idle, no GPU work": 10.0, "gemm": 6.0})
    assert sum(loss.values()) == pytest.approx(100 - w["tensor_active_pct"])
    assert w["other_metrics_mean"]["SM Active [Throughput %]"] == pytest.approx(80.0)
    gaps = {g["us"]: (g["gaps"], g["time_pct"]) for g in w["idle_gaps"]}
    assert gaps["2000-20000"] == (4, pytest.approx(20.0)) and gaps["0-20"] == (0, 0.0)
    assert got["all"]["steps"] == 4
    assert (got["queued_body"]["steps"], got["queued_body"]["last_step"]) == (3, 3)


def test_queued_starts_once_the_last_request_is_sent(tmp_path):
    got = U.analyse(_report(tmp_path / "prof.sqlite"), _trace(tmp_path / "advice_trace.jsonl"), 8, 1830.0, sent_s=(Z + 25 * MS) / 1e9)
    assert (got["queued"]["steps"], got["queued"]["first_step"]) == (2, 3)
    assert "queued" not in U.analyse(tmp_path / "prof.sqlite", tmp_path / "advice_trace.jsonl", 8, 1830.0)
    timings = tmp_path / "main.timings.json"
    timings.write_text(json.dumps({"start_unix": 1000.0, "requests": [{"send_s": 0.002}, {"send_s": 0.025}, {"send_s": None}]}))
    assert U.last_send(timings) == pytest.approx(1000.025)


def test_generic_short_names_go_by_their_demangled_name(tmp_path):
    db = sqlite3.connect(_report(tmp_path / "prof.sqlite"))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 3, 4)", (70 * MS, 71 * MS))
    db.commit()
    names = {n for _, _, n in U.kernels(db)}
    assert "void cutlass::device_kernel<Sm90GemmTile>()" in names
    assert U.category("void cutlass::device_kernel<Sm90GemmTile>()") == "gemm"
    fa3 = "void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<"
    assert [U.category(n) for n in ("sha512_rows_kernel", "h1_noise_fold", "fa3_matReqG_tap", fa3, "reshape_and_cache_flash_kernel")] \
        == ["commit", "pouw", "tap", "attention", "copy"]


def test_tensor_active_at_the_reference_clock(tmp_path):
    """The GPC clock comes in Hz under a [MHz] label; each tensor sample is scaled by the clock sample at or before it, so
    a window at 1,830 MHz and 915 MHz in turn reads half its tensor-busy cycles at the reference in the slow half."""
    db = sqlite3.connect(_report(tmp_path / "prof.sqlite"))
    assert U.analyse(tmp_path / "prof.sqlite", _trace(tmp_path / "advice_trace.jsonl"), 8, 1830.0)["all"]["sm_clock_mhz"] is None
    db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (0, 3, 'GPC Clock Frequency [MHz]')")
    for j in range(120):
        t = j * MS // 2 + MS // 4
        db.execute("INSERT INTO GPU_METRICS VALUES (?, 0, 3, ?)", (t, 1830e6 if t < 30 * MS else 915e6))
    db.commit()
    w = U.analyse(tmp_path / "prof.sqlite", _trace(tmp_path / "advice_trace.jsonl"), 8, 1830.0)["all"]
    assert w["tensor_active_pct"] == pytest.approx(54.0)
    assert w["sm_clock_mhz"] == pytest.approx((1830 + 915) / 2, abs=0.1)
    assert w["tensor_active_pct_at_ref"] == pytest.approx(54.0 * 0.75)
    assert U.mhz(1599.3) == 1599.3


def test_the_steady_cut_drops_steps_over_one_and_a_half_medians(tmp_path):
    """Full steps 1 to 4 of 10, 5, 20 and 3 ms: the median is 7.5 ms, so the 20 ms step goes, and Tensor Active is read over
    the kept steps' samples only (12 of 20, 10 of 10 and 2 of 6 at 90%)."""
    ends = [10, 20, 25, 45, 48, 55]
    recs = [{"kind": "header", "schema": "verity-vllm/step-trace/v1"}]
    recs += [{"kind": "step", "step": k, "T": 8 if 1 <= k <= 4 else 4, "t": (Z + e * MS) / 1e9} for k, e in enumerate(ends)]
    (trace := tmp_path / "advice_trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    s = U.analyse(_report(tmp_path / "prof.sqlite"), trace, 8, 1830.0)["all"]["steady"]
    assert (s["steps"], s["dropped"]) == (3, 1)
    assert s["step_wall_ms"] == {"mean": pytest.approx(6.0), "median": pytest.approx(5.0)}
    assert s["tensor_active_pct"] == pytest.approx(60.0)
    assert s["sm_clock_mhz"] is None


@pytest.mark.parametrize("arm, served", [("stock", True), ("noseal", True), ("graphs", False), ("async", False),
                                         ("seal-async", False)])
def test_an_arm_off_the_served_stack_says_so(tmp_path, monkeypatch, capsys, arm, served):
    out = tmp_path / "util.json"
    monkeypatch.setattr(sys, "argv", ["served_util.py", "--sqlite", str(_report(tmp_path / "prof.sqlite")), "--trace",
                                      str(_trace(tmp_path / "advice_trace.jsonl")), "--T", "8", "--arm", arm, "--card", "h200-sxm-141gb", "--out", str(out)])
    assert U.main() == 0
    assert json.loads(out.read_text())["served_stack"] is served
    assert ("(not the served stack)" in capsys.readouterr().out) is not served


@pytest.mark.parametrize("row, served", [({"async_scheduling": True, "source": "w.json"}, True),
                                         ({"async_scheduling": False, "source": "w.json"}, False), (None, False)])
def test_a_commit_arm_served_async_is_the_served_stack_when_its_row_declares_async(tmp_path, monkeypatch, row, served):
    """``<arm>-async`` is the served stack when serve-openai held its engine to a bench row that declares async scheduling
    (serve.json's ``row_scheduling``); not when the row declares sync (served_util.sh's bypass) or serve.json names none."""
    (serve := tmp_path / "serve").mkdir()
    (serve / "serve.json").write_text(json.dumps({"row_scheduling": row}))
    out = tmp_path / "util.json"
    monkeypatch.setattr(sys, "argv", ["served_util.py", "--sqlite", str(_report(tmp_path / "prof.sqlite")), "--trace",
                                      str(_trace(tmp_path / "advice_trace.jsonl")), "--T", "8", "--arm", "noseal-async",
                                      "--card", "h200-sxm-141gb", "--serve-dir", str(serve), "--out", str(out)])
    assert U.main() == 0
    assert json.loads(out.read_text())["served_stack"] is served


def test_too_few_full_steps_is_said(tmp_path):
    got = U.analyse(_report(tmp_path / "prof.sqlite"), _trace(tmp_path / "advice_trace.jsonl", T=16), 8, 1830.0)
    assert got["all"]["steps"] == 0 and "fewer than two" in got["all"]["note"]


SECRET = "SECRET-request-text"


def test_util_json_and_util_txt_hold_the_serves_aggregates_alone(tmp_path, monkeypatch, capsys):
    """util.json's serve part is counts, sizes and seconds over the windows, the engine's steps and whether it failed, and
    H1's release counts: no window's own row, error text, pause or release summary leaves through it or util.txt."""
    (serve := tmp_path / "serve").mkdir()
    (serve / "windows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (
        {"window": 7, "steps": 4, "committed_bytes": 2e9, "passed": True, "seconds": {"open": 1.0}, "error": SECRET},
        {"window": 8, "steps": 4, "committed_bytes": 4e9, "passed": False, "seconds": {"open": 3.0},
         "pouw_h1": {"error": SECRET, "seconds": 2.0}})))
    (serve / "summary.json").write_text(json.dumps({
        "served": 9, "engine_steps": 12, "engine_error": SECRET, "stopped_by": SECRET, "pauses": [SECRET], "release": {"x": SECRET},
        "sealer": {"x": SECRET}, "commit": {"seals": [SECRET], "pouw_h1": {"windows": 2, "tied": 1, "untied": 1, "failed": 0,
                                                                           "gate_passed_rows": 3, "first": SECRET}}}))
    out = tmp_path / "util.json"
    monkeypatch.setattr(sys, "argv", ["served_util.py", "--sqlite", str(_report(tmp_path / "prof.sqlite")), "--trace",
                                      str(_trace(tmp_path / "advice_trace.jsonl")), "--T", "8", "--arm", "h1", "--card",
                                      "h200-sxm-141gb", "--serve-dir", str(serve), "--out", str(out)])
    assert U.main() == 0
    doc, said = json.loads(out.read_text()), capsys.readouterr().out
    assert doc["serve"]["windows"] == {"closed": 2, "passed": 1, "h1_failed": 1}
    assert doc["serve"]["summary"] == {"engine_steps": 12, "engine_failed": True,
                                       "pouw_h1": {"windows": 2, "tied": 1, "untied": 1, "failed": 0, "gate_passed_rows": 3}}
    assert SECRET not in out.read_text() and SECRET not in said, said


def test_a_stub_runs_out_and_log_hold_no_trace_timings_summary_or_serve_options(tmp_path):
    """served_util.sh's leased part, under stand-ins for nsys, nvidia-smi, the venv's python and sleep, whose serve writes
    the step trace, summary, pauses, serve options and the loads' timings into the serve dir and scratch: none of those
    files, the GPU's identity, nsys's output or the profile is in $OUT, and none of their contents is in $OUT or the run's
    stdout and stderr; $OUT holds steps.txt alone, its lines with no clock."""
    bin_, venv, cuda, root, out = (tmp_path / x for x in ("bin", "venv/bin", "cuda", "root", "out"))
    for d in (bin_, venv, cuda / "bin", root / "scratch" / "local" / "serve"):
        d.mkdir(parents=True)
    stubs = {
        bin_ / "nsys": ('case $1 in launch) while [ $# -gt 0 ]; do [ "$1" = --out ] && d=$2; shift; done; mkdir -p "$d"\n'
                        f'  for f in serve.json summary.json advice_trace.jsonl pauses.jsonl; do echo {SECRET} > "$d/$f"; done\n'
                        '  echo "SERVE-OPENAI listening" ;;\nesac'),
        bin_ / "nvidia-smi": f"echo name, {SECRET}-uuid",
        # the serve opens its log only once it runs: each wait is for its listening line, not a clock
        bin_ / "sleep": f'until grep -q "SERVE-OPENAI listening" "{root}/scratch/local/serve.log" 2>/dev/null; do :; done',
        venv / "python": ('case "$*" in *serve_load.py*) while [ $# -gt 0 ]; do [ "$1" = --out ] && o=$2; shift; done\n'
                          f'  echo {SECRET} > "$o"; echo {SECRET} > "${{o%.jsonl}}.timings.json" ;;\nesac'),
    }
    for path, body in stubs.items():
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "ARM": "stock", "CARD": "h200-sxm-141gb", "T": "1024",
           "N": "1", "UTIL_ROOT": str(root), "VENV": str(venv.parent), "CUDA_HOME": str(cuda), "NSYS": str(bin_ / "nsys"),
           "OUT": str(out), "TREE": str(HERE.parents[2]), "ROW": str(tmp_path / "row")}
    run = subprocess.run(["bash", str(HERE.parents[2] / "benchmarks" / "one_stage" / "served_util.sh"), "leased"],
                         env=env, capture_output=True, text=True, check=False)
    log = run.stdout + run.stderr
    assert "SERVE SERVE-OPENAI listening" in log, log
    held = {p.name for p in out.rglob("*")}
    assert not held & {"serve.json", "summary.json", "advice_trace.jsonl", "pauses.jsonl", "warm.timings.json",
                       "main.timings.json"}, held
    assert SECRET not in log and not [p for p in out.rglob("*") if p.is_file() and SECRET in p.read_text(errors="replace")]
    assert held == {"steps.txt"}, held
    assert not [x for x in (out / "steps.txt").read_text().splitlines() if re.match(r"\d\d:\d\d:\d\dZ ", x)]
