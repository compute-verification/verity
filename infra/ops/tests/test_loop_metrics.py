"""The control pod's loops' numbers (infra/control/loop_metrics.py): all compute's dollars an hour from the fleet loop's newest
sample and the budget guard's last poll, against the cap, or null with its why; each sample sent once and a null at most every
ten minutes while none comes; a store index pass's length or its exit code's meaning, from the record store_index.sh writes.
The sender is a stand-in and the clock is injected. One test runs store_index.sh itself, its paths moved under tmp_path and a
stand-in `python3.12` on PATH: the record it writes is the one the script reads, and a send that hangs doesn't hold the loop."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / "infra" / "control"
spec = importlib.util.spec_from_file_location("loop_metrics", CONTROL / "loop_metrics.py")
M = sys.modules["loop_metrics"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

T0 = 1_800_000_000


def _guard(tmp_path, monkeypatch, polled: float | None, rate: float = 42.32) -> Path:
    monkeypatch.setenv("RESEARCH_GUARD_DIR", str(tmp_path / "guard"))
    state = M.guard.paths()["state"]
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"schema": "research/budgets-guard/v1", "last_ok_poll": polled, "rate_now": rate}))
    return state


def _rows(t: float, *rates: float | None) -> list[dict]:
    return [{"t": M.fleet.utc(t), "name": f"vy-nebius-cpu-{i}", "rate_usd_per_hour": r} for i, r in enumerate(rates, 1)]


def test_the_hourly_total_is_the_samples_rates_and_the_guards_runpod_rate_against_the_cap(tmp_path, monkeypatch):
    _guard(tmp_path, monkeypatch, T0 - 65)
    assert M.compute_reading(_rows(T0, 6.34, 6.34, 0.82), T0) == {
        "metric": "infra/compute-usd-per-hour", "value": 55.82, "mark": "measured", "time": M.fleet.utc(T0), "target": 150}
    assert M.compute_reading([], T0)["value"] == 42.32, "no Nebius machine runs: RunPod's alone"


def test_the_hourly_total_is_null_with_its_why_when_a_machine_has_no_rate_or_the_guards_poll_is_stale(tmp_path, monkeypatch):
    state = _guard(tmp_path, monkeypatch, T0 - 700)
    r = M.compute_reading(_rows(T0, 6.34, None, None), T0)
    assert r["value"] is None and r["target"] == 150 and r["why"] == (
        "vy-nebius-cpu-2, vy-nebius-cpu-3 have no rate in the machines registry; the budget guard's last poll was 12 min "
        "before the sample, over 10")
    _guard(tmp_path, monkeypatch, None)
    assert M.compute_reading(_rows(T0, 6.34), T0)["why"] == "the budget guard has completed no poll"
    state.write_text("{cut")
    assert M.compute_reading(_rows(T0, 6.34), T0)["why"] == f"the budget guard's state {state} is unreadable (JSONDecodeError)"
    state.unlink()
    assert M.compute_reading(_rows(T0, 6.34), T0)["why"] == f"the budget guard has left no state at {state}"


def test_the_loop_sends_each_sample_once_and_a_null_at_most_every_ten_minutes_while_none_comes(tmp_path, monkeypatch):
    _guard(tmp_path, monkeypatch, T0 - 30, rate=0.0)
    monkeypatch.setattr(M.fleet, "FLEET_DIR", tmp_path / "fleet")
    monkeypatch.setattr(M.metrics, "deployed_commit", lambda start: "c" * 40)
    (tmp_path / "fleet").mkdir()
    samples = tmp_path / "fleet" / f"samples-{M.fleet.utc(T0)[:10].replace('-', '')}.jsonl"
    samples.write_text("".join(json.dumps(r) + "\n" for r in _rows(T0, 6.34, 6.34)))
    later = T0 + 1450
    passes = iter([61, 601, 661, 1200, 1201, 1500, 1560])
    now = [T0 + 1]
    sent = []

    def sleep(s: float) -> None:
        now[0] = T0 + next(passes)
        if now[0] == T0 + 1500:
            _guard(tmp_path, monkeypatch, later - 40, rate=0.0)
            with samples.open("a") as f:
                f.write(json.dumps(_rows(later, 6.34)[0]) + "\n")

    def send(readings, *, loop, commit, dry_run):
        sent.append((now[0] - T0, readings[0]["value"], readings[0]["time"], loop, commit, dry_run))
    assert M.compute(every=60, clock=lambda: now[0], sleep=sleep, send=send, passes=8) == 0
    assert [s[:3] for s in sent] == [(1, 12.68, M.fleet.utc(T0)), (601, None, M.fleet.utc(T0 + 601)),
                                     (1201, None, M.fleet.utc(T0 + 1201)), (1500, 6.34, M.fleet.utc(later))], (
        "each sample once at its own time; while none is newer than 10 min, a null at most every 10 min")
    assert {s[3:] for s in sent} == {("loop-metrics", "c" * 40, False)}


def test_the_null_for_a_missing_sample_says_why(tmp_path, monkeypatch):
    monkeypatch.setattr(M.fleet, "FLEET_DIR", tmp_path)
    sent = []
    assert M.compute(clock=lambda: T0, send=lambda r, **k: sent.append(r[0])) == 0
    assert sent == [{"metric": "infra/compute-usd-per-hour", "value": None, "mark": "measured", "time": M.fleet.utc(T0),
                     "target": 150, "why": "the fleet loop has written no sample in 10 min: it is down, or no Nebius "
                                           "machine runs"}]


def test_a_failed_send_is_one_line_and_the_loop_goes_on(tmp_path, monkeypatch, capsys):
    _guard(tmp_path, monkeypatch, T0)
    monkeypatch.setattr(M.fleet, "FLEET_DIR", tmp_path)
    (tmp_path / f"samples-{M.fleet.utc(T0)[:10].replace('-', '')}.jsonl").write_text(json.dumps(_rows(T0, 1.0)[0]) + "\n{cut")
    assert M.compute(clock=lambda: T0 + 5, send=lambda *a, **k: None) == 1
    assert capsys.readouterr().err == f"{M.fleet.utc(T0 + 5)} infra/compute-usd-per-hour: not read: JSONDecodeError: " \
        "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)\n"
    (tmp_path / f"samples-{M.fleet.utc(T0)[:10].replace('-', '')}.jsonl").write_text(json.dumps(_rows(T0, 1.0)[0]) + "\n")
    assert M.compute(clock=lambda: T0 + 5, send=lambda *a, **k: "POST https://x/api/metrics: HTTP 503 down") == 1
    assert capsys.readouterr().err == f"{M.fleet.utc(T0 + 5)} POST https://x/api/metrics: HTTP 503 down\n"


def test_a_pass_is_its_length_at_its_end_or_null_with_its_exit_codes_meaning():
    assert M.index_reading({"start": T0, "end": T0 + 365, "rc": 0}) == {
        "metric": "infra/store-index-pass-seconds", "value": 365, "mark": "measured", "time": M.fleet.utc(T0 + 365)}
    whys = {rc: M.index_reading({"start": T0, "end": T0 + 9, "rc": rc}) for rc in (1, 124, 137, 143)}
    assert all(r["value"] is None for r in whys.values())
    assert [whys[rc]["why"] for rc in (1, 124, 137, 143)] == [
        "the pass failed with exit 1: store-index.log's lines before it say why",
        "the pass ran into store_index.sh's timeout (exit 124)",
        "the pass was killed by signal 9 (exit 137), as an out-of-memory kill is",
        "the pass was killed by signal 15 (exit 143)"]


def test_store_index_sh_records_each_pass_for_the_script_and_never_waits_on_its_send(tmp_path):
    """The loop's record and the script's reader agree, and a send that hangs leaves the loop free: the stand-in send blocks on a
    FIFO the test opens only after the loop's pass has ended (its `sleep 300`, here `exit 0`)."""
    text = (CONTROL / "store_index.sh").read_text()
    moves = {"/data/infra/": f"{tmp_path}/infra/", "/root/": f"{tmp_path}/root/", "/data/watchers/src": f"{tmp_path}/watchers",
             "  sleep 300\n": "  exit 0\n"}
    for a, b in moves.items():
        assert a in text, f"store_index.sh no longer has {a!r}: move this test's stand-in paths with it"
        text = text.replace(a, b)
    for d in ("infra/store-index-src", "root/.config/verity", "bin"):
        (tmp_path / d).mkdir(parents=True)
    for f in ("r2.env", "store-index.env"):
        (tmp_path / "root" / ".config" / "verity" / f).write_text("")
    (tmp_path / "infra" / "store-index-src" / "COMMIT").write_text("c1\n")
    os.mkfifo(fifo := tmp_path / "release")
    py = tmp_path / "bin" / "python3.12"
    py.write_text('#!/bin/sh\ncase "$*" in\n  *"-m research data index"*) exit 3 ;;\n'
                  f'  *loop_metrics.py*) echo "send started: $*"; cat {fifo} > /dev/null; exit 0 ;;\nesac\nexit 99\n')
    py.chmod(0o755)
    (script := tmp_path / "store_index.sh").write_text(text)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    p = subprocess.run(["sh", str(script)], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-500:]
    rec = json.loads((tmp_path / "infra" / "store-index-pass.json").read_text())
    assert rec["rc"] == 3 and rec["commit"] == "c1" and rec["end"] >= rec["start"]
    with open(fifo, "w"):
        pass
    log = (tmp_path / "infra" / "store-index.log").read_text()
    assert " pass rc=3 c1\n" in log
    assert f"send started: -s {tmp_path}/watchers/infra/control/loop_metrics.py index --pass {tmp_path}/infra/store-index-pass.json" in log
    sent = []
    assert M.index(tmp_path / "infra" / "store-index-pass.json", send=lambda r, **k: sent.append((r, k))) == 0
    assert sent == [([M.index_reading(rec)], {"loop": "store-index", "commit": "c1", "dry_run": False})]
