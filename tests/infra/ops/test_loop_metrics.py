"""The control pod's loops' numbers (infra/control/loop_metrics.py): all compute's dollars an hour, every Nebius machine of the
fleet's last pass (none when none runs) and every RunPod pod the budget guard read, against the row's limit, or null with its
why; the guard's poll aged as the reading is sent; each fleet pass sent once, across a restart too, and a null at most every
ten minutes while no pass comes; a send that raises or fails is one line, said once while the door answers the same way; a
second copy of the loop refuses; the rows name the loops that send them, and each reading names its script and holds only
the keys the door takes. A store index pass's length or its exit code's meaning, from the record store_index.sh writes.
The sender is a stand-in and the clock is injected. One test runs store_index.sh itself for two passes, its paths moved under
tmp_path and stand-ins for `date`, `sleep` and `python3.12` first on PATH, so it reads no clock."""
from __future__ import annotations

import fcntl
import functools
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / "infra" / "control"
spec = importlib.util.spec_from_file_location("loop_metrics", CONTROL / "loop_metrics.py")
M = sys.modules["loop_metrics"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

T0 = 1_800_000_000
ROWS = {m["id"]: m for m in tomllib.loads((CONTROL / "metrics.toml").read_text())["metric"]}
LIMIT = ROWS[M.COMPUTE]["floor"]


@pytest.fixture(autouse=True)
def _no_deployed_commit(tmp_path, monkeypatch):
    """A checkout has no COMMIT, and the one above it is outside the suite's inputs."""
    monkeypatch.setattr(M, "COMMIT_FILE", tmp_path / "COMMIT", raising=False)
    monkeypatch.setattr(M.fleet, "FLEET_DIR", tmp_path / "fleet")


def _guard(tmp_path, monkeypatch, polled: float | None, rate: float = 42.32, unrated=(), **extra) -> Path:
    """The guard's state: `rate_now` (its scope) is less than `rate_all` (every pod), which is the one counted."""
    monkeypatch.setenv("RESEARCH_GUARD_DIR", str(tmp_path / "guard"))
    state = M.guard.paths()["state"]
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"schema": "research/budgets-guard/v1", "last_ok_poll": polled, "rate_now": rate - 2.0,
                                 "rate_all": rate, "unrated": list(unrated), **extra}))
    return state


def _rows(t: float, *rates: float | None) -> list[dict]:
    return [{"t": M.fleet.utc(t), "name": f"vy-nebius-cpu-{i}", "rate_usd_per_hour": r} for i, r in enumerate(rates, 1)]


def _pass(t: float, *rates: float | None) -> None:
    M.fleet.FLEET_DIR.mkdir(exist_ok=True)
    (M.fleet.FLEET_DIR / "last-sample.json").write_text(json.dumps({"t": M.fleet.utc(t), "rows": _rows(t, *rates)}))


def test_the_hourly_total_is_every_nebius_machine_and_every_runpod_pod_against_the_rows_limit(tmp_path, monkeypatch):
    _guard(tmp_path, monkeypatch, T0 - 65)
    assert M.limit() == LIMIT
    assert M.compute_reading(_rows(T0, 6.34, 6.34, 0.82), T0, T0 + 5, M.limit()) == {
        "metric": "infra/compute-usd-per-hour", "value": 55.82, "mark": "measured", "time": M.fleet.utc(T0), "target": LIMIT}


def test_with_no_nebius_machine_running_the_loop_sends_the_runpod_rate_alone(tmp_path, monkeypatch):
    _guard(tmp_path, monkeypatch, T0 - 30)
    _pass(T0)
    sent = []
    assert M.compute(clock=lambda: T0 + 5, send=lambda r, **k: sent.append(r[0])) == 0
    assert sent == [{"metric": "infra/compute-usd-per-hour", "value": 42.32, "mark": "measured", "time": M.fleet.utc(T0),
                     "target": LIMIT}]


def test_the_guards_poll_is_aged_as_the_reading_is_sent_not_at_the_fleets_pass(tmp_path, monkeypatch):
    _guard(tmp_path, monkeypatch, T0 - 100)
    _pass(T0, 6.34)
    sent = []
    assert M.compute(clock=lambda: T0 + 590, send=lambda r, **k: sent.append(r[0])) == 0
    assert sent[0]["value"] is None
    assert sent[0]["why"] == "the budget guard's last poll was 11.5 min ago, over 10"


def test_a_rate_that_cant_be_read_makes_the_total_null_with_its_why_never_a_smaller_sum(tmp_path, monkeypatch):
    def why(rows):
        r = M.compute_reading(rows, T0, T0 + 5, LIMIT)
        assert r["value"] is None and r["target"] == LIMIT
        return r["why"]
    state = _guard(tmp_path, monkeypatch, T0 - 700)
    assert why(_rows(T0, 6.34, None, None)) == (
        "vy-nebius-cpu-2, vy-nebius-cpu-3 have no rate in the machines registry; the budget guard's last poll was 11.8 min "
        "ago, over 10")
    _guard(tmp_path, monkeypatch, T0, unrated=["pod-a", "pod-b"])
    assert why(_rows(T0, 6.34)) == "RunPod pods pod-a, pod-b list no costPerHr that is a number"
    state.write_text(json.dumps({"last_ok_poll": T0, "rate_now": 40.0}))
    assert why(_rows(T0, 6.34)) == "the budget guard's state has no `rate_all`: its guard predates it"
    _guard(tmp_path, monkeypatch, None)
    assert why(_rows(T0, 6.34)) == "the budget guard has completed no poll"
    state.write_text("{cut")
    assert why(_rows(T0, 6.34)) == f"the budget guard's state {state} is unreadable (JSONDecodeError)"
    state.unlink()
    assert why(_rows(T0, 6.34)) == f"the budget guard has left no state at {state}"


def test_the_loop_sends_each_pass_once_across_a_restart_and_a_null_at_most_every_ten_minutes(tmp_path, monkeypatch):
    _guard(tmp_path, monkeypatch, T0 - 30, rate=0.0)
    (tmp_path / "COMMIT").write_text("c" * 40 + "\n")
    kept = tmp_path / "loop-metrics.json"
    _pass(T0, 6.34, 6.34)
    later = T0 + 1450
    passes = iter([61, 601, 661, 1200, 1201, 1500, 1560])
    now = [T0 + 1]
    sent = []

    def sleep(s: float) -> None:
        now[0] = T0 + next(passes)
        if now[0] == T0 + 1500:
            _guard(tmp_path, monkeypatch, later - 40, rate=0.0)
            _pass(later, 6.34)

    def send(readings, *, command, commit, dry_run):
        sent.append((now[0] - T0, readings[0]["value"], readings[0]["time"], command, commit, dry_run))
    assert M.compute(every=60, state=kept, clock=lambda: now[0], sleep=sleep, send=send, passes=8) == 0
    assert [s[:3] for s in sent] == [(1, 12.68, M.fleet.utc(T0)), (601, None, M.fleet.utc(T0 + 601)),
                                     (1201, None, M.fleet.utc(T0 + 1201)), (1500, 6.34, M.fleet.utc(later))], (
        "each fleet pass once at its own time; while none is newer than 10 min, a null at most every 10 min")
    assert {s[3:] for s in sent} == {("infra/control/loop_metrics.py", "c" * 40, False)}
    assert json.loads(kept.read_text()) == {"sent": M.fleet.utc(later), "said": T0 + 1500}
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".loop-metrics")) == [], "replaced, no temporary left"
    del sent[:]
    now[0] = T0 + 1570
    assert M.compute(every=60, state=kept, clock=lambda: now[0], sleep=sleep, send=send, passes=1) == 0
    assert sent == [], "a restarted loop doesn't send the pass it sent before"
    _pass(T0 + 1565, 6.34)
    assert M.compute(state=kept, dry_run=True, clock=lambda: now[0], send=send) == 0 and sent[-1][1] == 6.34
    assert json.loads(kept.read_text())["sent"] == M.fleet.utc(later), "a dry run keeps no state"


def test_a_send_that_raises_or_fails_is_said_once_per_reason_and_the_loop_goes_on(tmp_path, monkeypatch, capsys):
    """Until the door is live it answers each pass the same way: that is said once, not every pass, and so is the first send
    it takes after."""
    _guard(tmp_path, monkeypatch, T0)
    _pass(T0, 1.0)
    now = [T0 + 5]
    outcomes = iter([RuntimeError("IncompleteRead(0 bytes read)"), *["POST https://x/api/metrics: HTTP 404"] * 4, None])

    def send(readings, **k):
        o = next(outcomes)
        if isinstance(o, Exception):
            raise o
        return o

    def sleep(s):
        now[0] += 400
        _pass(now[0], 1.0)
    kept = tmp_path / "loop-metrics.json"
    assert M.compute(every=60, state=kept, clock=lambda: now[0], sleep=sleep, send=send, passes=4) == 1
    once = " (said once while the door answers so)\n"
    assert capsys.readouterr().err == (
        f"{M.fleet.utc(T0 + 5)} infra/compute-usd-per-hour: the send raised RuntimeError: IncompleteRead(0 bytes read){once}"
        f"{M.fleet.utc(T0 + 405)} POST https://x/api/metrics: HTTP 404{once}")
    assert json.loads(kept.read_text())["failing"] == "POST https://x/api/metrics: HTTP 404"
    assert M.compute(every=60, state=kept, clock=lambda: now[0], sleep=sleep, send=send, passes=3) == 1
    assert capsys.readouterr().err == f"{M.fleet.utc(T0 + 2005)} infra/compute-usd-per-hour: the door took it again\n", (
        "a restart doesn't say the refusal again; the first send the door takes is said once")
    assert "failing" not in json.loads(kept.read_text())
    (M.fleet.FLEET_DIR / "last-sample.json").write_text("{cut")
    assert M.compute(clock=lambda: T0 + 5, send=send) == 1
    assert capsys.readouterr().err.startswith(f"{M.fleet.utc(T0 + 5)} infra/compute-usd-per-hour: not read: JSONDecodeError")


def test_a_second_copy_of_the_loop_refuses_and_sends_nothing(tmp_path, capsys):
    kept = tmp_path / "loop-metrics.json"
    fd = os.open(tmp_path / "loop-metrics.json.lock", os.O_CREAT | os.O_WRONLY)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def send(*a, **k):
        raise AssertionError("nothing is sent")
    assert M.compute(every=60, state=kept, clock=lambda: T0, send=send) == 0
    assert capsys.readouterr().err == f"{M.fleet.utc(T0)} infra/compute-usd-per-hour: nothing sent: another copy holds {kept}.lock\n"
    os.close(fd)
    off = tmp_path / "no-such-dir" / "loop-metrics.json"
    assert M.compute(every=60, state=off, clock=lambda: T0, send=send) == 1
    assert capsys.readouterr().err.endswith(f"its lock {off}.lock can't be opened (FileNotFoundError): not the pod's loop\n")


def test_each_row_names_the_loop_that_sends_it():
    watchers = tomllib.loads((CONTROL / "watchers.toml").read_text())
    argv = next(lp["argv"] for lp in watchers["loop"] if lp["name"] == "loop-metrics")
    on_pod = [a.replace("{src}/", "").replace("{dir}", watchers["dir"]) for a in argv]
    assert ROWS[M.COMPUTE]["command"] == " ".join(on_pod), "the compute row is the watchers loop `loop-metrics`, as it runs"
    assert ROWS[M.INDEX]["command"] == "sh infra/control/store_index.sh", "the index row is the store index loop"


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


def test_store_index_sh_replaces_its_record_and_reaps_and_logs_each_send_which_never_holds_the_lock(tmp_path, monkeypatch):
    """Two passes of store_index.sh with stand-ins first on PATH:
    - `date +%s` reads 1000 before a pass has indexed and 1365 after, so the record's length is 365 only when the start is read
      before the pass; `date -u` is one fixed time. No clock is read.
    - `python3.12`: the pass exits 3; the first pass's send reports whether it holds the loop's lock descriptor (fd 9), then
      blocks on a FIFO and exits 124, as `timeout 60` does when it kills a hung send; the second's exits 0.
    - `sleep`: the first opens the FIFO, which the loop can only reach if it didn't wait on its send; the second fails, which
      ends the loop (`sleep 300 || { wait; exit 0; }`).
    The first pass's send is reaped by the second pass, whose `wait` logs its 124. A hard link to the record from before holds
    its old bytes: the record was replaced, not written in place."""
    text = (CONTROL / "store_index.sh").read_text()
    moves = {"/data/infra/": f"{tmp_path}/infra/", "/root/": f"{tmp_path}/root/", "/data/watchers/src": f"{tmp_path}/watchers",
             "  sleep 300\n": "  sleep 300 || { wait; exit 0; }\n"}
    for a, b in moves.items():
        assert a in text, f"store_index.sh no longer has {a!r}: move this test's stand-in paths with it"
        text = text.replace(a, b)
    for d in ("infra/store-index-src", "root/.config/verity", "bin", "marks"):
        (tmp_path / d).mkdir(parents=True)
    for f in ("r2.env", "store-index.env"):
        (tmp_path / "root" / ".config" / "verity" / f).write_text("")
    (tmp_path / "infra" / "store-index-src" / "COMMIT").write_text("c1\n")
    record = tmp_path / "infra" / "store-index-pass.json"
    record.write_text("old\n")
    os.link(record, before := tmp_path / "record-before")
    os.mkfifo(fifo := tmp_path / "release")
    marks = tmp_path / "marks"
    stand_ins = {
        "date": f'case "$1" in\n  +%s) [ -e {marks}/indexed ] && echo 1365 || echo 1000 ;;\n  *) echo 2027-01-15T08:00:00Z ;;\nesac\n',
        "python3.12": f'case "$*" in\n  *"-m research data index"*) touch {marks}/indexed; exit 3 ;;\n'
                      f'  *loop_metrics.py*) [ -e /proc/self/fd/9 ] && echo "send holds fd 9"; echo "send started: $*"\n'
                      f'    [ -e {marks}/sent ] && exit 0; touch {marks}/sent; cat {fifo} > /dev/null; exit 124 ;;\nesac\nexit 99\n',
        "sleep": f'rm -f {marks}/indexed\n[ -e {marks}/slept ] && exit 1\ntouch {marks}/slept; echo go > {fifo}\n'}
    for name, body in stand_ins.items():
        (p := tmp_path / "bin" / name).write_text("#!/bin/sh\n" + body)
        p.chmod(0o755)
    (script := tmp_path / "store_index.sh").write_text(text)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"), prepend=os.pathsep)
    p = subprocess.run(["sh", str(script)], stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-500:]
    rec = json.loads(record.read_text())
    assert rec == {"start": 1000, "end": 1365, "rc": 3, "commit": "c1"}, "the start is read before the pass, the end after"
    assert before.read_text() == "old\n" and not record.with_name(record.name + ".tmp").exists(), "replaced, not written in place"
    log = (tmp_path / "infra" / "store-index.log").read_text()
    assert log.count("2027-01-15T08:00:00Z pass rc=3 c1\n") == 2
    assert "send holds fd 9" not in log, "the send runs without the loop's lock descriptor"
    assert log.count("send started: ") == 2
    assert log.count("metrics send rc=") == 1 and "2027-01-15T08:00:00Z metrics send rc=124\n" in log, (
        "the next pass reaps the send and logs the timeout's exit")
    sent = []
    assert M.index(record, send=lambda r, **k: sent.append((r, k))) == 0
    assert sent == [([M.index_reading(rec)], {"command": "infra/control/store_index.sh", "commit": "c1", "dry_run": False})]


def test_a_store_index_send_the_door_refuses_is_said_once_and_exits_1_only_then(tmp_path, capsys):
    record, sent = tmp_path / "pass.json", tmp_path / "send.json"
    record.write_text(json.dumps({"start": T0, "end": T0 + 365, "rc": 0, "commit": "c1d2e3f"}))
    answers = iter(["POST x: HTTP 404", "POST x: HTTP 404", "POST x: HTTP 503", None, None])
    rcs = [M.index(record, send=lambda r, **k: next(answers), clock=lambda: T0, state=sent) for _ in range(5)]
    assert rcs == [1, 0, 1, 0, 0], "store_index.sh logs `metrics send rc=1` only beside a reason it hasn't said"
    at = M.fleet.utc(T0)
    assert capsys.readouterr().err == (f"{at} POST x: HTTP 404 (said once while the door answers so)\n"
                                       f"{at} POST x: HTTP 503 (said once while the door answers so)\n"
                                       f"{at} infra/store-index-pass-seconds: the door took it again\n")
    assert json.loads(sent.read_text()) == {}
    assert M.index(record, send=lambda r, **k: "POST x: HTTP 404", clock=lambda: T0) == 1, "with no --state, each is said"


def test_every_reading_names_its_rows_script_and_holds_only_the_keys_the_door_takes(tmp_path, monkeypatch):
    """Through the real sender: the door refuses a whole request for one key it doesn't know (`loop` was one)."""
    door = {"metric", "time", "value", "mark", "unit", "ci95", "target", "verdict", "why", "period", "over", "member", "run",
            "commit", "command"}
    (key := tmp_path / "panels.key").write_text("k")
    (tmp_path / "COMMIT").write_text("c" * 40 + "\n")
    bodies = []
    send = functools.partial(M.metrics.send, key=key, post=lambda method, url, body, k, what: bodies.append(body))
    _guard(tmp_path, monkeypatch, T0 - 30)
    _pass(T0, 6.34)
    assert M.compute(clock=lambda: T0 + 5, send=send) == 0
    assert M.compute(clock=lambda: T0 + 5000, send=send) == 0
    (record := tmp_path / "pass.json").write_text(json.dumps({"start": T0, "end": T0 + 365, "rc": 0, "commit": "c1d2e3f"}))
    assert M.index(record, send=send, clock=lambda: T0) == 0
    readings = [r for b in bodies for r in b["readings"]]
    assert [r["metric"] for r in readings] == [M.COMPUTE, M.COMPUTE, M.INDEX] and [r["value"] for r in readings][1] is None
    assert all(set(r) <= door for r in readings), [sorted(set(r) - door) for r in readings]
    for r in readings:
        assert r["command"] in ROWS[r["metric"]]["command"].split(), "the reading names the script its row runs"
    assert [r["commit"] for r in readings] == ["c" * 40, "c" * 40, "c1d2e3f"]
