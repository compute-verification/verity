"""The monitors (`research/monitors.py`): the runner opens each incident once and clears it once, through the alert dir and an
optional webhook; each check reads its source the way the node writes it; node 2 is read in one session per pass, behind a
guard that fails closed; the relay posts every new event once; and the files deploy.toml installs on node 1 run every
node-side check."""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from research import monitors as M

T0 = 1791060000.0           # 2026-10-03T20:40:00Z
SRC = Path(__file__).resolve().parents[1] / "src"
REPO = Path(__file__).resolve().parents[3]
UNIT = SRC / "research/pods/nebius/monitoring/vy-monitors.service"


def iso(t: float) -> str:
    return M.iso(t)


class Run:
    """A fake subprocess.run: `answers` maps a predicate over the argv (and stdin) to (rc, stdout, stderr)."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, argv, input=None, capture_output=True, text=True, timeout=None, env=None):
        self.calls.append((list(argv), input))
        for pred, (rc, out, err) in self.answers:
            if pred(argv, input):
                return subprocess.CompletedProcess(argv, rc, out, err)
        raise AssertionError(f"unexpected command {argv}")


def has(word):
    return lambda argv, _input: any(word in a for a in argv)


def mon(check, **kw) -> M.Monitor:
    return M.Monitor(kw.pop("id", "m"), kw.pop("owner", "infra"), kw.pop("severity", "warn"), "node", check, "do the thing", **kw)


def found(*keys, **kw):
    return lambda ctx: [M.Finding(k, f"*{k} is wrong*", **kw) for k in keys]


def events(alerts: Path) -> list[dict]:
    p = alerts / "monitors.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


def passes(state, m, results, alerts, t0=T0, every=60, **kw):
    """Run `m` once a minute with each check result in turn (a list of keys, or an exception to raise)."""
    out = []
    for i, r in enumerate(results):
        def check(ctx, r=r):
            if isinstance(r, Exception):
                raise r
            return [M.Finding(k, f"*{k} is wrong*") for k in r]
        mm = M.Monitor(m.id, m.owner, m.severity, m.where, check, m.action, every_s=m.every_s, for_s=m.for_s, clear_s=m.clear_s,
                       events=m.events, unread_s=m.unread_s)
        out.append(M.step(state, [mm], lambda memo: M.Ctx(now=t0 + i * every, memo=memo), t0 + i * every, M.AlertDir(alerts),
                          by="test", **kw))
    return out


# ---- the runner ------------------------------------------------------------------------------------------------------------------

def test_an_incident_opens_once_with_a_note_and_clears_once(tmp_path):
    state, alerts = M.new_state(), tmp_path / "alerts"
    out = passes(state, mon(found()), [["a"], ["a"], ["a", "b"], ["b"], []], alerts)
    assert [[(e["event"], e["key"]) for e in evs] for evs in out] == [
        [("open", "m|a")], [], [("open", "m|b")], [("clear", "m|a")], [("clear", "m|b")]]
    evs = events(alerts)
    assert [e["event"] for e in evs] == ["open", "open", "clear", "clear"]
    assert evs[0]["schema"] == "research/monitors-event/v1" and evs[0]["owners"] == ["infra"] and evs[0]["by"] == "test"
    assert evs[2]["text"] == "*Cleared:* a is wrong" and evs[2]["since"] == iso(T0)
    notes = sorted((alerts / "outbox").iterdir())
    assert len(notes) == 2 and all(n.name.startswith("20261003T") and "-alert-m-" in n.name for n in notes)
    assert state["open"] == {} and state["last_pass"] == T0 + 240


def test_the_note_has_alert_sinks_front_matter_and_names_its_monitor(tmp_path):
    passes(M.new_state(), mon(found()), [["a"]], tmp_path)
    text = next((tmp_path / "outbox").iterdir()).read_text()
    fm = M.front_matter(text)
    assert {k: fm[k] for k in ("campaign", "lane", "kind", "status", "repo")} == {
        "campaign": "nebius-monitoring", "lane": "infra", "kind": "finding", "status": "open", "repo": "danielreuter/verity"}
    assert fm["monitor"] == "m" and fm["key"] == '"m|a"' and fm["owners"] == "[infra]" and fm["id"] + ".md" == events(tmp_path)[0]["note"]
    assert "# a is wrong" in text and "What to do: do the thing" in text


def test_a_finding_opens_only_once_seen_for_its_for_s_and_a_gap_starts_it_again(tmp_path):
    state = M.new_state()
    out = passes(state, mon(found(), for_s=120), [["a"], ["a"], [], ["a"], ["a"], ["a"]], tmp_path)
    assert [len(evs) for evs in out] == [0, 0, 0, 0, 0, 1]


def test_an_incident_clears_only_once_gone_for_its_clear_s(tmp_path):
    state = M.new_state()
    out = passes(state, mon(found(), clear_s=120), [["a"], [], ["a"], [], [], []], tmp_path)
    assert [[e["event"] for e in evs] for evs in out] == [["open"], [], [], [], [], ["clear"]]


def test_a_timers_unit_that_fails_every_run_is_one_incident(tmp_path):
    """vy-custody.service fails each 5-minute run and is out of `systemctl --failed` while the next one runs: one open, no clears."""
    [m] = [m for m in M.MONITORS if m.id == "units-n1"]
    out = passes(M.new_state(), m, [["system:vy-custody.service"], []] * 6, tmp_path, every=300)
    assert [e["event"] for evs in out for e in evs] == ["open"]


def test_a_check_that_isnt_due_doesnt_run(tmp_path):
    state, ran = M.new_state(), []
    m = mon(lambda ctx: ran.append(ctx.now) or [], every_s=600)
    for t in (T0, T0 + 60, T0 + 596, T0 + 1200):
        M.step(state, [m], lambda memo, t=t: M.Ctx(now=t, memo=memo), t, M.AlertDir(tmp_path), by="test")
    assert ran == [T0, T0 + 596, T0 + 1200]


def test_a_skip_changes_nothing_and_a_failed_read_keeps_incidents_then_opens_unreadable(tmp_path):
    state = M.new_state()
    m = mon(found(), unread_s=180)
    out = passes(state, m, [["a"], M.Skip("timed window"), RuntimeError("ssh: timeout"), RuntimeError("ssh: timeout"),
                            RuntimeError("ssh: timeout"), ["a"]], tmp_path)
    assert [[(e["event"], e["key"]) for e in evs] for evs in out] == [
        [("open", "m|a")], [], [], [], [("open", "m|unreadable")], [("clear", "m|unreadable")]]
    unread = events(tmp_path)[1]
    assert unread["owners"] == ["infra"] and "ssh: timeout" in unread["text"] and "for 3 minutes" in unread["text"]
    assert state["monitors"]["m"]["error"] is None and "m|a" in state["open"]


def test_a_monitor_that_never_read_opens_unreadable_after_unread_s_from_its_first_pass(tmp_path):
    out = passes(M.new_state(), mon(found(), unread_s=120), [OSError("no file")] * 3, tmp_path)
    assert [[e["key"] for e in evs] for evs in out] == [[], [], ["m|unreadable"]]


def test_an_events_monitor_remembers_its_first_pass_and_posts_each_new_finding_once(tmp_path):
    state = M.new_state()
    out = passes(state, mon(found(), events=True), [["old"], ["old", "new"], ["new", "newer"], []], tmp_path)
    assert [[(e["event"], e["key"]) for e in evs] for evs in out] == [[], [("event", "m|new")], [("event", "m|newer")], []]
    assert state["open"] == {}


def test_a_write_to_the_alert_dir_that_fails_opens_nothing_and_the_next_pass_tries_again(tmp_path):
    (tmp_path / "alerts").write_text("not a directory")
    state = M.new_state()
    assert passes(state, mon(found()), [["a"]], tmp_path / "alerts") == [[]]
    assert state["open"] == {} and "sink_error" in state
    (tmp_path / "alerts").unlink()
    assert [e["key"] for e in passes(state, mon(found()), [["a"]], tmp_path / "alerts", t0=T0 + 60)[0]] == ["m|a"]
    assert "sink_error" not in state


def test_the_alert_dir_leaves_out_an_open_or_clear_its_key_already_ends_on_and_an_event_it_has(tmp_path):
    def ev(kind, key):
        return {"event": kind, "key": key, "monitor": "m", "severity": "warn", "owners": ["infra"], "text": "*x*", "action": "a",
                "by": "t"}

    seq = [("open", "m|a"), ("open", "m|a"), ("clear", "m|a"), ("clear", "m|a"), ("open", "m|a"), ("event", "m|e"),
           ("event", "m|e"), ("clear", "m|b")]
    d = M.AlertDir(tmp_path)
    assert [d.write(ev(k, key), T0) for k, key in seq] == [True, False, True, False, True, True, False, True]


def test_a_pass_run_again_from_the_state_before_it_writes_nothing_twice_and_the_relay_posts_it_once(tmp_path):
    alerts, keys = tmp_path / "alerts", ["old"]
    e = mon(lambda ctx: [M.Finding(k, f"*{k}*") for k in keys], id="held-idle", events=True)
    st = M.new_state()
    M.step(st, [e], lambda memo: M.Ctx(now=T0, memo=memo), T0, M.AlertDir(alerts), by="test")
    before = json.dumps(st)
    keys.append("19:00Z|node 1|circuits")
    for now in (T0 + 65, T0 + 130):                                         # the first is killed before it saves its state
        st = json.loads(before)
        M.step(st, [mon(found("a")), e], lambda memo, now=now: M.Ctx(now=now, memo=memo), now, M.AlertDir(alerts), by="test")
    lines = (alerts / "monitors.jsonl").read_text().splitlines()
    assert [(x["event"], x["key"]) for x in map(json.loads, lines)] == [("open", "m|a"), ("event", "held-idle|19:00Z|node 1|circuits")]
    assert len(list((alerts / "outbox").iterdir())) == 2
    posted, src, rstate = [], FakeSource("node1"), {}
    M.relay_pass(rstate, [src], lambda ev, th: posted.append(ev["key"]) or {"ts": "1"}, T0)
    src.all = lines
    M.relay_pass(rstate, [src], lambda ev, th: posted.append(ev["key"]) or {"ts": "1"}, T0 + 200)
    assert posted == ["m|a", "held-idle|19:00Z|node 1|circuits"]


def test_the_state_is_saved_after_each_monitor_so_a_pass_killed_midway_keeps_what_it_wrote(tmp_path):
    path, st = tmp_path / "state.json", M.new_state()

    def killed(ctx):
        raise KeyboardInterrupt                                             # systemd's kill, as far as the pass can tell

    with pytest.raises(KeyboardInterrupt):
        M.step(st, [mon(found("a")), mon(killed, id="k")], lambda memo: M.Ctx(now=T0, memo=memo), T0, M.AlertDir(tmp_path / "a"),
               by="test", checkpoint=lambda: M.save(path, st))
    assert list(M.load(path)["open"]) == ["m|a"]


def test_checks_still_due_at_the_passs_deadline_are_late_and_run_next_pass(tmp_path):
    t, ran, state = [0.0], [], M.new_state()
    slow = mon(lambda ctx: t.__setitem__(0, t[0] + M.PASS_S - 2) or [], id="slow", every_s=600)
    after = mon(lambda ctx: ran.append(ctx.now) or [], id="after", every_s=600)
    for now in (T0, T0 + 60):
        M.step(state, [slow, after], lambda memo, now=now: M.Ctx(now=now, memo=memo), now, M.AlertDir(tmp_path), by="test",
               clock=lambda: t[0])
        t[0] += 1000
    assert ran == [T0 + 60] and state["monitors"]["after"]["skipped"] is None
    late = M.Ctx(now=T0, run=Run(), pass_=M.Pass(deadline=20.0, clock=lambda: 0.0))
    with pytest.raises(M.Late, match="too little for node 2's session"):
        late.node2()
    run = session()
    M.Ctx(now=T0, run=run, pass_=M.Pass(deadline=50.0, clock=lambda: 0.0)).node2()
    assert "timeout -k 5 40 " in run.calls[0][0][-1] and '"budget": 25.0' in run.calls[0][1]


def test_the_webhook_gets_each_event_retries_until_slack_takes_it_and_never_logs_its_url(tmp_path):
    import urllib.error

    url = "https://hooks.example.invalid/services/T0/" + "B0/x"
    sent, fail = [], [True]

    def post(u, text):
        if fail[0]:
            raise urllib.error.HTTPError(u, 500, "boom " + u, {}, None)
        sent.append((u, text))

    state = M.new_state()
    passes(state, mon(found()), [["a"]], tmp_path, webhook=url, post=post)
    assert sent == [] and len(state["retry"]) == 1 and state["webhook_error"].endswith("HTTP 500") and url not in json.dumps(state)
    fail[0] = False
    passes(state, mon(found()), [["a"], []], tmp_path, t0=T0 + 60, webhook=url, post=post)
    assert [t.splitlines()[0] for _, t in sent] == ["@infra *a is wrong*", "*Cleared:* a is wrong"] and state["retry"] == []
    assert all(e["webhook"] for e in events(tmp_path)) and "webhook_error" not in state


def test_run_refuses_while_another_runner_holds_the_lock(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(M.WEBHOOK_ENV, raising=False)
    state = tmp_path / "state.json"
    fd = M.lock(state)
    assert M.main(["run", "--by", "test", "--state", str(state), "--alerts", str(tmp_path / "a"), "--only", "held-idle"]) == 3
    assert "another runner holds" in capsys.readouterr().err
    os.close(fd)


def test_run_exits_1_while_its_alert_dir_takes_no_writes_and_node_1s_heartbeat_pages_on_it(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(M.WEBHOOK_ENV, raising=False)
    monkeypatch.setattr(M, "HELD_IDLE", str(tmp_path / "none.jsonl"))       # held-idle-stale finds `stale`
    (tmp_path / "alerts").write_text("not a directory")

    def run(state, alerts):
        return M.main(["run", "--by", "test", "--state", str(state), "--alerts", str(alerts), "--only", "held-idle-stale"])

    assert run(tmp_path / "state.json", tmp_path / "alerts") == 1 and "takes no writes" in capsys.readouterr().err
    saved = json.loads((tmp_path / "state.json").read_text())
    out = f"{saved['last_pass']}\n{json.dumps(saved['sink_error'])}\n"
    (f,) = M.node_heartbeat(M.Ctx(now=saved["last_pass"] + 60, run=Run((has("n1"), (0, out, "")))))
    assert f.key == "sink" and "can't write their alert dir" in f.text and saved["sink_error"] in f.text
    assert run(tmp_path / "fresh.json", tmp_path / "a") == 0
    assert list(json.loads((tmp_path / "fresh.json").read_text())["open"]) == ["held-idle-stale|stale"]


def test_a_runner_that_crashes_fails_its_unit():
    """vy-monitors.service counts SuccessExitStatus as a pass: only 3 (the lock refused) may be there, never 1, a traceback's."""
    ok = re.search(r"^SuccessExitStatus=(.*)$", UNIT.read_text(), re.M)[1].split()
    assert ok == ["3"]


def test_a_traceback_exits_1(tmp_path):
    (tmp_path / "state.json").write_text("{not json")
    r = subprocess.run([sys.executable, "-m", "research.monitors", "run", "--by", "test", "--state", str(tmp_path / "state.json"),
                        "--alerts", str(tmp_path / "a"), "--only", "held-idle-stale"], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(SRC)})
    assert r.returncode == 1 and "Traceback" in r.stderr


def test_the_command_line_lists_and_rejects_unknown_ids(capsys):
    assert M.main(["list", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "research/monitors-list/v1" and {m["id"] for m in doc["monitors"]} == set(M.BY_ID)
    assert M.main(["check", "no-such-monitor"]) == 2


def test_every_monitor_names_an_owner_a_severity_an_action_and_what_it_checks():
    from research import slack

    handles = {h["handle"] for h in slack.load_registry(slack.registry_path())["handles"]}
    assert set(M.HANDLES) <= handles
    for m in M.MONITORS:
        assert m.owner in handles and m.severity in ("warn", "page") and m.where in ("node", "relay", "machine"), m.id
        assert m.action and m.doc and m.every_s >= 60, m.id


# ---- node 2: one session per pass, behind its guard ---------------------------------------------------------------------------

def status(age=5, *, timed=False, window=False, free=2, n=8, now=T0):
    return (f"{iso(now - age)} free GPUs {free}/{n}, kept free -, waiters -, timed {timed}, window waiting {window}, "
            "backfill False, queued 0 (gpu 0, cpu 0)")


def guard(tmp_path, line, now=T0) -> str:
    """What N2_GUARD prints under sh, then `echo ran`, with `line` as node 2's fill status and `date +%s` saying `now`."""
    f, bin_ = tmp_path / "status.txt", tmp_path / "bin"
    if line is not None:
        f.write_text(line + "\n  running x\n")
    bin_.mkdir(exist_ok=True)
    (bin_ / "date").write_text(f'#!/bin/sh\nif [ "$1" = "+%s" ]; then echo {int(now)}; else exec {shutil.which("date")} "$@"; fi\n')
    (bin_ / "date").chmod(0o755)
    return subprocess.run(["sh", "-c", M.N2_GUARD.replace(M.FILL_STATUS, str(f)) + "echo ran"], capture_output=True, text=True,
                          env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"}).stdout


@pytest.mark.parametrize("line, why", [
    (status(5), None),
    (status(60), None),
    (status(61), "61 s old"),                       # fill_runner rewrites it every 10 s: a minute without, it has stopped
    (status(-3600), "3600 s ahead of its clock"),
    (status(5, timed=True), "isn't quiet"),
    (status(5, window=True), "isn't quiet"),        # the drain just before a timed run
    ("free GPUs 2/8, kept free -, timed False, window waiting False", "has no time"),
    (None, "has no fill status"),
])
def test_node_2s_guard_lets_a_read_run_only_under_a_quiet_status_at_most_a_minute_old(tmp_path, line, why):
    out = guard(tmp_path, line)
    if why is None:
        assert out == "ran\n"
    else:
        assert out.startswith(M.SKIPPED) and not out.endswith("ran\n") and why in str(M.refused(out, T0)), out


def test_a_refused_read_is_a_skip_and_what_runs_on_node_2_is_bounded_and_niced_there():
    run = Run((has("vy-n2"), (0, f"{M.SKIPPED} {int(T0)} 5 {status(timed=True)}\n", "")))
    with pytest.raises(M.Refused, match="isn't quiet") as e:
        M.Ctx(now=T0, run=run).n2("systemctl --failed; echo ---")
    assert (e.value.status, e.value.now, e.value.age) == (status(timed=True), T0, 5)
    argv = run.calls[0][0]
    assert argv[:len(M.SSH) + 1] == [*M.SSH, "vy-n2"] and argv[-1].startswith(M.N2_GUARD)
    assert argv[-1][len(M.N2_GUARD):] == "timeout -k 5 80 nice -n 19 ionice -c3 sh -c 'systemctl --failed; echo ---'"


def session(line=None, *, now=T0, windows=None, **parts):
    """Node 2 over ssh, its guard letting the session through: each part's output, and what it read of fill/windows."""
    doc = {"status": line or status(), "now": now, "windows": windows,
           "parts": {k: {"rc": 0, "out": v, "err": ""} for k, v in parts.items()}}
    return Run((has("vy-n2"), (0, json.dumps(doc), "")))


def refusing(line, *, now=T0, age=5):
    return Run((has("vy-n2"), (0, f"{M.SKIPPED} {int(now)} {age} {line}\n", "")))


N2 = [m for m in M.MONITORS if m.id.endswith("-n2")]


def n2_pass(state, run, now, alerts):
    return M.step(state, N2, lambda memo: M.Ctx(now=now, memo=memo, run=run), now, M.AlertDir(alerts), by="test")


def test_a_pass_logs_into_node_2_once_reading_every_part_its_due_checks_need(tmp_path):
    state, run = M.new_state(), session(probe=json.dumps(reading()), waiters="[]", preempted="", units="")
    n2_pass(state, run, T0, tmp_path)
    (argv, driver), = run.calls
    assert argv[-1].startswith(M.N2_GUARD) and argv[-1].endswith("sh -c 'python3 -'")
    assert "def evaluate(" in driver and M.UNITS in driver and "/run/gpu-lease/wait." in driver and "preempted.log" in driver
    assert all(st["error"] is None and st["skipped"] is None for st in state["monitors"].values()) and state["node2"]["read_t"] == T0
    n2_pass(state, run, T0 + 65, tmp_path)                                  # disk-n2 and units-n2 aren't due
    (argv, driver), = run.calls[1:]
    assert "def evaluate(" not in driver and M.UNITS not in driver and "/run/gpu-lease/wait." in driver


def test_while_node_2_isnt_quiet_each_pass_logs_in_once_runs_nothing_there_and_every_node_2_check_skips(tmp_path):
    state = M.new_state()
    for i in range(3):
        now = T0 + 65 * i
        run = refusing(status(timed=True, now=now), now=now)
        n2_pass(state, run, now, tmp_path)
        assert len(run.calls) == 1
    sts = state["monitors"]
    assert all("isn't quiet" in sts[m.id]["skipped"] for m in N2 if m.id != "fill-status-n2")
    assert sts["fill-status-n2"]["skipped"] is None and sts["fill-status-n2"]["error"] is None


def test_no_pass_logs_into_node_2_while_a_window_the_last_session_read_as_booked_runs(tmp_path):
    state = M.new_state()
    windows = f"# when, minutes, who\n{iso(T0 + 600)[:16]}Z 30 # circuits timed run\nnot a window\n2026-10-01T10:00Z 30\n"
    n2_pass(state, session(windows=windows, probe=json.dumps(reading()), waiters="[]", preempted="", units=""), T0, tmp_path)
    assert state["node2"]["windows"] == [[T0 + 600, T0 + 2400, "circuits timed run"]]
    run = Run()
    n2_pass(state, run, T0 + 900, tmp_path)
    assert run.calls == [] and all("booked window until 21:20Z (circuits timed run)" in st["skipped"]
                                   for st in state["monitors"].values())
    run = session(waiters="[]", preempted="")
    n2_pass(state, run, T0 + 2400, tmp_path)
    assert len(run.calls) == 1


def test_node_2s_session_reads_each_part_and_stops_once_the_status_isnt_quiet(tmp_path):
    f, w = tmp_path / "status.txt", tmp_path / "windows"
    f.write_text("2000-01-01T00:00:00Z free GPUs 8/8, timed False, window waiting False\n")
    w.write_text("2026-10-04T10:00Z 30 # x\n")
    parts = {"a": [["cat"], "from stdin"],
             "b": [["sh", "-c", f"echo B; echo '2000-01-01T00:00:00Z free GPUs 0/8, timed True, window waiting False' > {f}"], None],
             "c": [["sh", "-c", "echo C"], None]}
    driver = M.n2_driver(parts, 30, status=str(f), windows=str(w), go_s=1e12)      # a status from 2000 counts as fresh
    doc = json.loads(subprocess.run([sys.executable, "-"], input=driver, capture_output=True, text=True).stdout)
    assert doc["parts"]["a"]["out"] == "from stdin" and doc["parts"]["b"]["out"] == "B\n" and "c" not in doc["parts"]
    assert doc["status"].endswith("timed False, window waiting False") and doc["stopped"].endswith("timed True, window waiting False")
    assert doc["windows"] == "2026-10-04T10:00Z 30 # x\n"


def unit_timeout() -> float:
    return float(re.search(r"^TimeoutStartSec=(\d+)$", UNIT.read_text(), re.M)[1])


def test_a_pass_with_node_2_hung_waits_on_it_once_and_ends_well_inside_the_units_timeout(tmp_path):
    """Node 2 takes the connection, then never answers: each ssh runs to its own timeout."""
    waited = []

    def hung(argv, input=None, capture_output=True, text=True, timeout=None, env=None):
        waited.append(timeout)
        raise subprocess.TimeoutExpired(argv, timeout)

    state = M.new_state()
    n2_pass(state, hung, T0, tmp_path)
    assert waited == [M.N2_S] and M.N2_S < M.PASS_S <= unit_timeout() - 30
    assert all("TimeoutExpired" in state["monitors"][m.id]["error"] for m in N2)


def test_a_part_that_fails_is_its_own_checks_error_and_one_the_session_didnt_reach_is_a_skip():
    doc = {"status": status(), "now": T0, "windows": None, "stopped": status(timed=True),
           "parts": {"units": {"rc": 1, "out": "", "err": "Failed to connect to bus"}, "preempted": {"error": "timed out after 3 s"}}}
    run = Run((has("vy-n2"), (0, json.dumps(doc), "")))
    ctx = M.Ctx(now=T0, run=run, pass_=M.Pass(n2_parts=frozenset({"units", "preempted", "waiters"})))
    with pytest.raises(RuntimeError, match="units exited 1: Failed to connect to bus"):
        ctx.n2_part("units")
    with pytest.raises(RuntimeError, match="timed out after 3 s"):
        ctx.n2_part("preempted")
    with pytest.raises(M.Skip, match="stopped being quiet"):
        ctx.n2_part("waiters")
    assert len(run.calls) == 1


# ---- the watermarks -------------------------------------------------------------------------------------------------------------

GB = 10**9


def reading(*, root_free_gb=150.0, ws_pct=70.0, iused=7_000_000, t=T0, errors=()):
    return {"schema": "resource-probe/v1", "node": "n1", "t": t, "errors": list(errors),
            "disk": {"root": {"path": "/", "free_b": root_free_gb * GB, "pct": 40.0, "ipct": 5.0, "iused": 1_000_000, "t": t},
                     "workspace": {"path": "/workspace", "free_b": 1000 * GB, "pct": ws_pct, "ipct": 35.0, "inodes": 20_000_000,
                                   "iused": iused, "t": t}},
            "mem": {"total_b": 1800 * GB, "avail_b": 900 * GB, "avail_pct": 50.0}, "ramlock": {"reserved_gb": 0},
            "replay_queue": {"queued": 0, "gb": 0}, "idle_gpus": [{"gpu": 3, "min_gib": 40.0}], "unpreserved": {"old": 0}}


def test_the_watermarks_key_breaches_as_the_stewards_tick_did_and_hard_ones_page():
    assert M.norm("root free 52 GB < 60 GB") == "root free # GB < # GB"
    assert M.norm("GPU 3 holds 40 GiB at 0% for 15 min") == "GPU<3> holds # GiB at #% for # min"
    backlog = "11 finished runs older than 1 h lack custody (oldest {})"
    assert {M.norm(backlog.format("r20261002-044616-9ba7, r20261002-052646-06e1")),
            M.norm(backlog.format("r20261002-052646-06e1, r20261002-054607-7b26"))} == {"# finished runs older than # h lack custody"}
    run = Run((has("--reading"), (0, json.dumps(reading(root_free_gb=40, errors=["idle_gpus: URLError: down"])), "")))
    ctx = M.Ctx(now=T0, run=run)
    got = {f.key: f for f in M.watermarks("n1")(ctx)}
    assert set(got) == {"HARD root free # GB < # GB: no new check starts", "GPU<3> holds # GiB at #% for # min", "error:idle_gpus"}
    assert got["HARD root free # GB < # GB: no new check starts"].severity == "page"
    assert got["error:idle_gpus"].for_s == 1200 and got["GPU<3> holds # GiB at #% for # min"].severity is None
    assert ctx.memo["rate_base"]["t"] == T0 and run.calls[0][0][:5] == M.NICE


def test_the_inode_rate_compares_with_the_base_reading_the_memo_keeps():
    memo: dict = {}
    for t, iused in ((T0, 7_000_000), (T0 + 1000, 7_200_000)):
        run = Run((has("--reading"), (0, json.dumps(reading(iused=iused, t=t)), "")))
        keys = [f.key for f in M.watermarks("n1")(M.Ctx(now=t, memo=memo, run=run))]
    assert any("inodes/h" in k for k in keys) and memo["rate_base"]["t"] == T0


def test_node_2s_probe_runs_in_its_session_and_its_reading_is_judged_on_node_1():
    run = session(probe=json.dumps(reading(root_free_gb=40)))
    keys = [f.key for f in M.watermarks("n2")(M.Ctx(now=T0, run=run))]
    (argv, driver), = run.calls
    assert "HARD root free # GB < # GB: no new check starts" in keys and "def evaluate(" in driver and "--reading" in driver


def test_the_probes_reading_flag_prints_a_reading_and_writes_nothing(tmp_path, monkeypatch, capsys):
    rp = M._probe()
    monkeypatch.setattr(rp, "read", lambda node, **kw: reading())
    assert rp.main(["--node", "n2", "--reading", "--state", str(tmp_path / "s" / "probe.json")]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "resource-probe/v1" and "breaches" not in doc and not any(tmp_path.iterdir())


# ---- node 1's stalls, the held-idle panel --------------------------------------------------------------------------------------

def test_stalls_are_n1_alerts_incidents_each_to_its_owner():
    snap = {"now": T0, "blocked": {"/run/gpu-lease-circuits": "pid 41 ran on GPU 3"}, "pane": [f"{iso(T0 - 30)} tick {{}}"], "log": None}
    run = Run((lambda argv, stdin: argv[-1] == "-" and "capture-pane" in stdin, (0, json.dumps(snap), "")))
    (f,) = M.stalls_n1(M.Ctx(now=T0, run=run))
    assert f.key == "blocked:/run/gpu-lease-circuits" and f.owners == ("circuits",) and "GPU 3" in f.text and "Cleared" in f.cleared


def held_row(hour, lane="circuits", pct=80.0, target=50.0, change=-2.0, **kw):
    return {"hour": hour, "node": "node 1", "lane": lane, "held_gpu_h": 4.0, "held_idle_gpu_h": 3.2, "held_idle_pct": pct,
            "target_pct": target, "change_pts": change, **kw}


def write_rows(root: Path, rows) -> None:
    p = root / M.HELD_IDLE.lstrip("/")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_held_idle_alerts_a_lane_above_target_that_isnt_falling_to_that_lane(tmp_path):
    write_rows(tmp_path, [held_row("2026-10-03T18:00Z"), held_row("2026-10-03T18:00Z", lane="circuits-tp8"),
                          held_row("2026-10-03T19:00Z", change=-6.0), held_row("2026-10-03T19:00Z", lane="proofs", pct=40.0),
                          held_row("2026-10-03T19:00Z", lane="console", target=None)])
    got = M.held_idle(M.Ctx(now=T0, root=tmp_path))
    assert [(f.key, f.owners) for f in got] == [("2026-10-03T18:00Z|node 1|circuits", ("circuits",)),
                                                ("2026-10-03T18:00Z|node 1|circuits-tp8", ("infra",))]
    assert "80.0% of 4.0 held GPU-hours" in got[0].text and "-2.0 points" in got[0].text
    write_rows(tmp_path, [held_row("2026-10-02T19:00Z"), held_row("2026-10-02T20:00Z", lane="proofs")])
    assert [f.key for f in M.held_idle(M.Ctx(now=T0, root=tmp_path))] == ["2026-10-02T20:00Z|node 1|proofs"]


def test_held_idle_is_stale_70_minutes_after_the_newest_hour_ends(tmp_path):
    write_rows(tmp_path, [held_row("2026-10-03T19:00Z")])
    assert M.held_idle_stale(M.Ctx(now=M.epoch("2026-10-03T21:10:00Z"), root=tmp_path)) == []
    (f,) = M.held_idle_stale(M.Ctx(now=M.epoch("2026-10-03T21:11:00Z"), root=tmp_path))
    assert f.key == "stale" and "71 minutes" in f.text and "19:00Z" in f.text
    assert M.held_idle_stale(M.Ctx(now=T0, root=tmp_path / "none"))[0].key == "stale"


# ---- a GPU idle while a waiter waits --------------------------------------------------------------------------------------------

def lease_state(root: Path, d="/run/gpu-lease", *, t=T0, gpus=(), waiting=(), blocked=None, name="node1"):
    p = root / d.lstrip("/") / "n1-lease.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"t": iso(t), "node": "vy-nebius-1", "mode": "local", "pool": len(gpus), "pool_name": name,
                             "blocked": blocked, "pending_holders": [], "waiting": list(waiting), "gpus": list(gpus),
                             "events": []}))


def gpu(i, holder=None, lease=None, fence=()):
    return {"index": i, "uuid": f"GPU-{i}", "holder": holder, "fence_pids": list(fence), **({"lease": lease} if lease else {})}


def waiter(since, n=1, **kw):
    return {"file": f"wait.{int(since)}.1", "pid": "1", "n": n, "who": "adhoc:ubuntu", "run": "r1", "since": iso(since), **kw}


def pool_doc(root: Path, gpus, t=T0):
    p = root / M.POOL_DOC.lstrip("/")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"generated_at": iso(t), "nodes": {"n1": {"gpus": gpus}}}))


def test_a_lease_waiter_with_a_free_pool_gpu_or_an_unheld_one_is_held_back(tmp_path):
    lease = {"who": "x", "since": iso(T0 - 600)}
    lease_state(tmp_path, gpus=[gpu(0, fence=[9]), gpu(1, "gpu-pool-1", lease), gpu(2, "gpu-pool-2", lease)],
                waiting=[waiter(T0 - 30), waiter(T0 - 200, on="5")])
    ctx = M.Ctx(now=T0, root=tmp_path)
    assert M.idle_while_waiting_n1(ctx) == []                                   # nothing idle; the newest waiter too new anyway
    pool_doc(tmp_path, [{"index": 0, "busy_5m": 0.0, "holder": None, "class": "free"},
                        {"index": 1, "busy_5m": 0.9, "holder": "lease:x", "class": "verity"}])
    assert M.idle_while_waiting_n1(ctx) == []                                   # GPU 0 idle, but the waiter is pinned to 5
    lease_state(tmp_path, gpus=[gpu(0, fence=[9]), gpu(1, "gpu-pool-1", lease), gpu(2, "gpu-pool-2")],
                waiting=[waiter(T0 - 30), waiter(T0 - 200, n=2)])
    (f,) = M.idle_while_waiting_n1(ctx)
    assert f.key == "node1" and f.for_s == 300 and "GPUs 0, 2 idle" in f.text and "Free in the pool: [2]" in f.text
    assert "held by nobody: [0]" in f.text and "waited 3 min" in f.text


def test_idle_while_waiting_skips_a_blocked_pool_and_the_kueue_hold_and_flags_a_stale_controller(tmp_path):
    lease_state(tmp_path, gpus=[gpu(2, "gpu-pool-2")], waiting=[waiter(T0 - 200)], blocked="pid 4 on GPU 2")
    assert M.idle_while_waiting_n1(M.Ctx(now=T0, root=tmp_path)) == []
    with pytest.raises(M.Skip):
        M.idle_while_waiting_n1(M.Ctx(now=M.epoch("2026-10-03T12:45:00Z"), root=tmp_path))
    lease_state(tmp_path, "/run/gpu-lease-circuits", t=T0 - 600, name="circuits")
    (f,) = M.idle_while_waiting_n1(M.Ctx(now=T0, root=tmp_path))
    assert f.key == "circuits:stale" and "10 minutes" in f.text


def test_a_node_2_waiter_with_enough_free_gpus_is_held_back_unless_its_gpu_is_unknown():
    w = f"pid=7 n=2 run=- who=circuits-tp8 since={iso(T0 - 400)}"

    def check(free, *ws):
        return M.idle_while_waiting_n2(M.Ctx(now=T0, run=session(status(free=free), waiters=json.dumps(ws))))

    (f,) = check(3, w)
    assert f.key == "idle" and "3 of 8 GPUs free" in f.text and "circuits-tp8" in f.text
    assert check(1, w) == []
    pinned = f"pid=7 n=1 on=5 run=- who=x since={iso(T0 - 400)}"
    assert check(3, pinned) == [] and check(8, pinned)[0].key == "idle"
    compile(M.N2_WAITERS, "n2-waiters", "exec")


def test_node_2s_fill_status_is_stale_after_5_minutes_on_node_2s_clock_whether_the_guard_refused_or_not():
    def check(age):                                          # node 1's clock an hour ahead of node 2's
        return M.fill_status_n2(M.Ctx(now=T0 + 3600, run=session(status(age), now=T0)))
    assert check(299) == []
    assert "6 minutes old" in check(360)[0].text
    (f,) = M.fill_status_n2(M.Ctx(now=T0, run=refusing(status(400), age=400)))
    assert f.key == "stale" and "7 minutes old" in f.text


def test_node_2_refusing_for_2_hours_past_its_booked_window_opens_one_unwatched_and_its_checks_read_nothing_meanwhile(tmp_path):
    """A timed lease never released (a window that crashed, a stuck lease file): every pass finds a fresh `timed True`."""
    state, opened = M.new_state(), []
    good = dict(probe=json.dumps(reading()), waiters="[]", preempted="", units="")
    n2_pass(state, session(windows=f"{iso(T0 + 600)[:16]}Z 60 # proofs timed run\n", **good), T0, tmp_path)
    for i in range(1, 41):                                  # every 5 minutes: a refusal, the window (no login), then refusals
        now = T0 + 300 * i
        opened += [(now - T0, e["key"]) for e in n2_pass(state, refusing(status(timed=True, now=now), now=now), now, tmp_path)]
    assert opened == [(4200 + 7200 + 300, "fill-status-n2|unwatched")]       # the window ends at T0 + 4200; 2 h on, the next pass
    (f,) = [e for e in events(tmp_path) if e["key"] == "fill-status-n2|unwatched"]
    assert "refused every read since 20:45Z*, 3.2 hours" in f["text"] and "units-n2" in f["text"]
    assert state["monitors"]["disk-n2"]["last_ok"] == T0 + 3600 and state["monitors"]["units-n2"]["last_ok"] == T0 + 3900  # in the window
    now = T0 + 300 * 42                                     # disk-n2, every 10 minutes, is due again
    garbled = Run((has("vy-n2"), (0, "not json", "")))
    assert {e["key"] for e in n2_pass(state, garbled, now, tmp_path)} == {
        f"{m.id}|unreadable" for m in N2 if m.id != "fill-status-n2"}                      # unread since the window, not now
    assert {e["key"] for e in n2_pass(state, session(now=now + 600, **good), now + 600, tmp_path) if e["event"] == "clear"} == {
        "fill-status-n2|unwatched", *(f"{m.id}|unreadable" for m in N2 if m.id != "fill-status-n2")}


# ---- a lease preempted within 60 s of its grant ----------------------------------------------------------------------------------

def preempted(t, since, gpu=6, sig="SIGTERM", pid=41390):
    return (f"{iso(t)} {sig} GPU {gpu} for pid 43581 (n=1, who=adhoc:ubuntu): pid={pid} run=r20261003-150938-8bc9 who=adhoc:ubuntu "
            f"uuid=GPU-a98a since={iso(since)} until={iso(since + 3600)} mem=48G preempt=1 scope=gpu-lease-{pid} cmd=bash run.sh --seed 68")


def test_fast_preemptions_are_leases_preempted_within_60_s_of_their_grant_in_the_last_30_minutes():
    lines = [preempted(T0 - 2000, T0 - 2010), preempted(T0 - 600, T0 - 608), preempted(T0 - 590, T0 - 608, sig="SIGKILL"),
             preempted(T0 - 300, T0 - 600, gpu=5, pid=28062), "garbage", preempted(T0 - 100, T0 - 140, gpu=2, pid=5)]
    got = M.fast_preemptions(lines, T0)
    assert [(x["gpu"], x["pid"], x["after_s"]) for x in got] == [(6, "41390", 8), (2, "5", 40)]


def test_fast_preempt_reads_each_node_1_pool_and_node_2_behind_its_guard(tmp_path):
    p = tmp_path / "run/gpu-lease-circuits/preempted.log"
    p.parent.mkdir(parents=True)
    p.write_text(preempted(T0 - 60, T0 - 70) + "\n")
    (f,) = M.fast_preempt_n1(M.Ctx(now=T0, root=tmp_path))
    assert f.key == "/run/gpu-lease-circuits" and "1 lease preempted within 60 s of its grant" in f.text and "10 s after" in f.text
    run = session(preempted="partial line\n" + preempted(T0 - 60, T0 - 70) + "\n")
    assert M.fast_preempt_n2(M.Ctx(now=T0, run=run))[0].key == "/run/gpu-lease"


# ---- failed units -----------------------------------------------------------------------------------------------------------------

FAILED = ("nvidia-fabricmanager.service loaded failed failed NVIDIA fabric manager service\n"
          "systemd-ask-password-console.path loaded failed failed Dispatch Password Requests to Console Directory Watch\n")


def test_failed_units_leave_out_the_expected_ones_and_tell_user_units_apart():
    assert M.failed_units(FAILED + "---\nn2-custody.service loaded failed failed n2 custody\n") == [
        ("system", "systemd-ask-password-console.path"), ("user", "n2-custody.service")]
    assert all(reason for reason in M.EXPECTED_FAILED.values())


def test_units_on_node_1_tolerate_a_user_manager_that_isnt_there_and_node_2_reads_both_behind_the_guard():
    run = Run((lambda a, _: a[:2] == ["systemctl", "--user"], (1, "", "Failed to connect to bus: No medium found")),
              (lambda a, _: a[0] == "systemctl", (0, FAILED, "")))
    assert [f.key for f in M.units("n1")(M.Ctx(now=T0, run=run))] == ["system:systemd-ask-password-console.path"]
    run = session(units=FAILED.splitlines()[0] + "\n---\n")
    assert M.units("n2")(M.Ctx(now=T0, run=run)) == [] and M.UNITS in run.calls[0][1]


def test_failed_login_session_scopes_are_one_incident_spanning_them():
    scopes = "".join(f"session-{n}.scope loaded failed failed Session {n} of User research\n" for n in (8334, 8212, 8299))
    run = Run((lambda a, _: a[:2] == ["systemctl", "--user"], (0, "", "")),
              (lambda a, _: a[0] == "systemctl", (0, FAILED + scopes, "")))
    other, s = M.units("n1")(M.Ctx(now=T0, run=run))
    assert other.key == "system:systemd-ask-password-console.path"
    assert s.key == "system:session-*.scope"
    assert "3 login session scopes have failed* (`session-8212.scope` to `session-8334.scope`)" in s.text
    one = Run((lambda a, _: a[:2] == ["systemctl", "--user"], (0, "", "")),
              (lambda a, _: a[0] == "systemctl", (0, scopes.splitlines()[0] + "\n", "")))
    assert "1 login session scope has failed* (`session-8334.scope`)" in M.units("n1")(M.Ctx(now=T0, run=one))[0].text


# ---- relay-side ----------------------------------------------------------------------------------------------------------------

def test_drift_is_one_incident_per_node_and_kind_and_an_unreadable_node_fails_the_check():
    rep = {"vy-nebius-1": {"commit": "c", "strays": ["/usr/local/bin/vy-old.bak-1"], "entries": [
        {"path": "/a", "state": "stale", "restart": None}, {"path": "/b", "state": "stale", "restart": None},
        {"path": "/c", "state": "ok", "restart": "a process started before"}, {"path": "/t/x", "state": "missing", "tree": "/t", "restart": None}]}}
    run = Run((has("drift"), (1, json.dumps(rep), "")))
    got = {f.key: f.text for f in M.deploy_drift(M.Ctx(now=T0, run=run))}
    assert set(got) == {"vy-nebius-1:stale", "vy-nebius-1:restart", "vy-nebius-1:stray"} and "2 stale" in got["vy-nebius-1:stale"]
    with pytest.raises(RuntimeError, match="vy-nebius-2"):
        M.deploy_drift(M.Ctx(now=T0, run=Run((has("drift"), (1, json.dumps({**rep, "vy-nebius-2": {"error": "ssh"}}), "")))))


def test_unpreserved_asks_for_runs_over_30_minutes_old_and_says_when_the_flag_is_missing():
    rep = {"schema": "research/pods-unpreserved/v0.1", "unpreserved": ["a1", {"id": "a2"}, "a3", "a4"]}
    run = Run((has("vy-nebius-1"), (1, json.dumps(rep), "")), (has("vy-nebius-2"), (0, json.dumps({"unpreserved": []}), "")))
    (f,) = M.unpreserved(M.Ctx(now=T0, run=run))
    assert f.key == "vy-nebius-1" and "4 runs" in f.text and "a1, a2, a3 and 1 more" in f.text
    assert ["--older-min", "30", "--json"] == run.calls[0][0][-3:]
    old = Run((has("unpreserved"), (2, "", "research: error: unrecognized arguments: --older-min 30")))
    with pytest.raises(RuntimeError, match="node-custody-unit"):
        M.unpreserved(M.Ctx(now=T0, run=old))


def test_an_unanswered_ask_goes_to_whoever_it_asked():
    rep = {"inboxes": [], "dropped": [], "unanswered": [{"ts": "1791050000.000100", "by": "circuits", "to": ["infra", "proofs"]}]}
    (f,) = M.asks(M.Ctx(now=T0, run=Run((has("traffic"), (0, json.dumps(rep), "")))))
    assert f.key == "1791050000.000100" and f.owners == ("infra", "proofs") and "@circuits" in f.text


def test_node_1s_heartbeat_is_stale_after_10_minutes_without_a_pass():
    def ctx(last):
        return M.Ctx(now=T0, run=Run((has("n1"), (0, f"{last}\n", ""))))
    assert M.node_heartbeat(ctx(T0 - 500)) == []
    assert "haven't run for 25 minutes" in M.node_heartbeat(ctx(T0 - 1500))[0].text
    (f,) = M.node_heartbeat(ctx("missing"))
    assert f.key == "stale" and "never finished a pass" in f.text


# ---- the relay -----------------------------------------------------------------------------------------------------------------

class FakeSource(M.Source):
    def __init__(self, name, lines=(), notes=None):
        super().__init__(name, "/x")
        self.all, self.files = list(lines), dict(notes or {})

    def lines(self, after):
        return len(self.all), self.all[after:]

    def notes(self):
        return sorted(self.files)

    def head(self, name):
        return self.files[name]


def ev(kind, key, **kw):
    return json.dumps({"schema": "research/monitors-event/v1", "event": kind, "monitor": "m", "key": key, "owners": ["infra"],
                       "severity": "warn", "text": f"{kind} {key}", "action": "a", "by": "t", **kw})


def test_the_relay_posts_what_is_still_open_first_then_each_new_event_once_with_clears_in_their_thread():
    posts = []

    def post(e, thread):
        posts.append((e["event"], e["key"], (thread or {}).get("ts")))
        return {"ts": f"ts-{e['key']}"}

    src = FakeSource("node1", [ev("open", "m|a"), ev("clear", "m|a"), ev("open", "m|b")])
    rstate: dict = {}
    assert M.relay_pass(rstate, [src], post, T0) == 1 and posts == [("open", "m|b", None)]
    src.all += [ev("open", "m|b"), ev("open", "m|c", webhook=True), ev("clear", "m|b"), ev("clear", "m|zz"), ev("event", "m|e")]
    M.relay_pass(rstate, [src], post, T0)
    assert posts[1:] == [("clear", "m|b", "ts-m|b"), ("event", "m|e", None)]
    assert M.relay_pass(rstate, [src], post, T0) == 0 and rstate["sources"]["node1"]["lines"] == 8


def test_a_post_that_fails_is_its_sources_first_next_pass_and_the_other_sources_go_on():
    node1, relay, rstate, fail, logged = FakeSource("node1"), FakeSource("relay"), {}, [True], []
    M.relay_pass(rstate, [node1, relay], lambda e, t: {}, T0)
    node1.all += [ev("open", "m|a"), ev("open", "m|b"), ev("open", "m|c")]
    relay.all += [ev("open", "node-heartbeat|stale", severity="page")]

    def post(e, thread):
        if e["key"] == "m|b" and fail[0]:
            raise RuntimeError("chat.postMessage: HTTP 503")
        return {"ts": e["key"]}

    assert M.relay_pass(rstate, [node1, relay], post, T0, log=logged.append) == 2
    assert rstate["sources"]["node1"]["lines"] == 1 and rstate["sources"]["relay"]["lines"] == 1
    assert logged == ["node1|open|m|b| not posted, node1 stops here until the next pass: RuntimeError: chat.postMessage: HTTP 503"]
    fail[0] = False
    assert M.relay_pass(rstate, [node1, relay], post, T0 + 60) == 2 and rstate["failing"] == {}


def test_a_post_slack_refuses_for_good_or_that_fails_for_an_hour_is_dropped_with_a_log_line():
    src, rstate, logged = FakeSource("node1"), {}, []
    M.relay_pass(rstate, [src], lambda e, t: {"ts": "1"}, T0)
    src.all += [ev("open", "m|a"), ev("open", "m|b")]
    M.relay_pass(rstate, [src], lambda e, t: {"ts": e["key"]}, T0)
    src.all += [ev("clear", "m|a"), ev("event", "m|e"), ev("clear", "m|b")]
    posted = []

    def post(e, thread):
        if e["key"] == "m|a":
            raise RuntimeError("conversations.replies: thread_not_found")         # its alert was deleted in #agent-alerts
        if e["key"] == "m|e":
            raise RuntimeError("chat.postMessage: HTTP 500")
        posted.append((e["event"], e["key"]))
        return {}

    for t in (T0 + 60, T0 + 120, T0 + 60 + M.RELAY_GIVE_UP_S):
        M.relay_pass(rstate, [src], post, t, log=logged.append)
    assert posted == [("clear", "m|b")] and rstate["threads"] == {} and rstate["failing"] == {}
    assert [ln.split(":")[0] for ln in logged] == ["dropped node1|clear|m|a|", "node1|event|m|e| not posted, node1 stops here until the next pass",
                                                  "node1|event|m|e| not posted, node1 stops here until the next pass",
                                                  "dropped node1|event|m|e|"]


def test_a_source_that_cant_be_read_is_logged_and_the_next_source_still_posts():
    class Down(FakeSource):
        def lines(self, after):
            raise RuntimeError("node1: exited 255: ssh: connect to host n1 port 22: Connection timed out")

    relay, rstate, logged, posted = FakeSource("relay"), {}, [], []
    M.relay_pass(rstate, [Down("node1"), relay], lambda e, t: {}, T0, log=logged.append)
    relay.all.append(ev("open", "node-heartbeat|stale", severity="page"))
    M.relay_pass(rstate, [Down("node1"), relay], lambda e, t: posted.append(e["key"]) or {}, T0 + 60, log=logged.append)
    assert posted == ["node-heartbeat|stale"] and len(logged) == 2 and logged[0].startswith("node1 unread: RuntimeError: node1: exited 255")


def fake_slack(monkeypatch, fail=lambda ch, thread: None, permalink=None, handles=()) -> list:
    """slack_post's Slack: `_post` records (channel, the text's first line, thread) unless `fail(channel, thread)` returns an
    error to raise; chat.getPermalink raises `permalink` when given. The registry holds `handles`."""
    from research import slack as S
    posts = []

    class Api:
        dry_run = False

        def __init__(self, *a, **kw):
            pass

        def call(self, method, **kw):
            if permalink:
                raise permalink
            return {"ok": True, "permalink": "https://slack.example/p"}

    def _post(api, ch, handle, text, thread=None, **kw):
        if e := fail(ch, thread):
            raise e
        posts.append((ch, text.splitlines()[0], thread))
        return f"ts{len(posts)}"

    monkeypatch.setattr(S, "registry_path", lambda *a: Path("registry.json"))
    monkeypatch.setattr(S, "load_registry", lambda p: {"handles": list(handles), "channels": {}})
    monkeypatch.setattr(S, "channel_id_for", lambda reg, key: {"agent-alerts": "CALERTS", S.FRONT_DOOR: "CFRONT"}[key])
    monkeypatch.setattr(S, "Routes", lambda *a, **kw: None)
    monkeypatch.setattr(S, "Slack", Api)
    monkeypatch.setattr(S, "_post", _post)
    monkeypatch.setattr(S, "entry", lambda reg, h: {"handle": h})
    return posts


def test_an_alert_whose_pointer_fails_is_posted_once_and_its_pointer_once_slack_takes_it(monkeypatch):
    from research import slack as S
    left = [2]

    def fail(ch, thread):
        if ch == "CFRONT" and left[0]:
            left[0] -= 1
            return S.SlackError("chat.postMessage: HTTP 429 (rate limited; retry after 1 s)")

    posts = fake_slack(monkeypatch, fail, permalink=S.SlackError("chat.getPermalink: HTTP 429"))
    src, rstate = FakeSource("node1"), {}
    M.relay_pass(rstate, [src], M.slack_post(False), T0)
    src.all.append(ev("open", "stalls-n1|blocked", owners=["proofs"]))
    for i in range(1, 4):
        M.relay_pass(rstate, [src], M.slack_post(False), T0 + 60 * i, log=lambda s: None)
    assert [(ch, thread) for ch, _, thread in posts] == [("CALERTS", None), ("CFRONT", None)]
    assert posts[1][1] == "@proofs open stalls-n1|blocked (alert: #agent-alerts)"
    assert rstate["threads"]["node1|stalls-n1|blocked"] == {"t": T0 + 60, "ts": "ts1", "pointer": "ts2", "done": True}


def test_an_alert_about_a_run_tags_the_handle_of_the_agent_that_sent_it_and_points_it_at_the_alert(monkeypatch):
    posts = fake_slack(monkeypatch, handles=[
        {"handle": "infra", "holder": "the infra coordinator (bc-17cc41f1-6227-5fab-bc64-3fa2f224558b)"},
        {"handle": "proofs", "holder": "the proofs coordinator (bc-8416bc72-c4cc); the research coordinator (bc-8ece7cde) too"},
        {"handle": "circuits", "holder": "the circuits coordinator (bc-b8aaadaa-aefd-503e-9863-5eabb3a28c79)"}])
    src, rstate = FakeSource("node1"), {}
    M.relay_pass(rstate, [src], M.slack_post(False), T0)
    src.all += [ev("event", "oom-kills|1", text="*n1: the OOM killer took lake of run r1, sent by "
                                                "https://cursor.com/agents/bc-8ece7cde-1111-2222*"),
                ev("event", "oom-kills|2", text="*n1: the OOM killer took lake of run r2, sent by https://cursor.com/agents/bc-00000000*")]
    M.relay_pass(rstate, [src], M.slack_post(False), T0 + 60)
    assert [(ch, " ".join(w for w in text.split() if w.startswith("@"))) for ch, text, _ in posts] == [("CALERTS", "@infra @proofs"), ("CFRONT", "@proofs"),
                                                                    ("CALERTS", "@infra")]


def test_a_clear_into_a_deleted_thread_is_a_post_of_its_own(monkeypatch, capsys):
    from research import slack as S
    posts = fake_slack(monkeypatch, lambda ch, thread: thread and S.SlackError("conversations.replies: thread_not_found"))
    src, rstate = FakeSource("node1"), {}
    M.relay_pass(rstate, [src], M.slack_post(False), T0)
    src.all += [ev("open", "units-n1|system:x"), ev("clear", "units-n1|system:x", text="*Cleared:* x")]
    M.relay_pass(rstate, [src], M.slack_post(False), T0 + 60)
    assert posts == [("CALERTS", "@infra open units-n1|system:x", None), ("CALERTS", "*Cleared:* x", None)]
    assert "its thread is gone" in capsys.readouterr().out and rstate["threads"] == {}


def test_the_relay_posts_grafanas_new_notes_but_not_the_monitors_own():
    old = "---\nid: x\nlane: infra\n---\n\n# old\n"
    src = FakeSource("node1", notes={"20261003T1000Z-alert-x-1.md": old})
    posts = []
    M.relay_pass({}, [src], lambda e, t: posts.append(e) or {}, T0)
    rstate: dict = {}
    M.relay_pass(rstate, [src], lambda e, t: posts.append(e) or {}, T0)
    src.files["20261003T2000Z-alert-gpustall-2.md"] = "---\nid: y\nlane: infra\norigin: grafana on vy-nebius-1\n---\n\n# GpuStall: 1 firing\n- GPU 3\n"
    src.files["20261003T2000Z-alert-m-3.md"] = "---\nid: z\nmonitor: m\n---\n\n# mine\n"
    M.relay_pass(rstate, [src], lambda e, t: posts.append(e) or {}, T0)
    assert [(p["monitor"], p["owners"], p["text"].splitlines()[0]) for p in posts] == [("grafana", ["infra"], "# GpuStall: 1 firing")]


def test_a_notes_name_reaches_the_shell_quoted(tmp_path):
    name = "x;touch pwned;.md"
    (tmp_path / "a b" / "outbox").mkdir(parents=True)
    (tmp_path / "a b" / "outbox" / name).write_text("---\nid: x\n---\n")
    src = M.Source("relay", str(tmp_path / "a b"), run=lambda argv, **kw: subprocess.run(argv, cwd=tmp_path, **kw))
    assert src.head(name) == "---\nid: x\n---\n" and src.lines(0) is None
    assert not (tmp_path / "pwned").exists()


def test_a_relay_post_tags_its_owners_and_says_what_to_do():
    e = json.loads(ev("open", "m|a", owners=["circuits", "infra"], severity="page"))
    text = M.message(e, lambda h: f"<{h}>")
    assert text.startswith("<circuits> <infra> [page] open m|a\n_What to do:_ a")


def test_what_a_finding_quotes_cant_mention_anyone_in_slack(monkeypatch):
    line = (f"{iso(T0 - 60)} SIGTERM GPU 7 for pid 2 (n=1, who=<!channel>): pid=1 run=r1 who=<!channel> uuid=GPU-x "
            f"since={iso(T0 - 65)} until={iso(T0 + 3600)} mem=48G preempt=1 scope=gpu-lease-1 cmd=a&b")
    f = M._fast_finding("/run/gpu-lease", "Node 1", M.fast_preemptions([line], T0))
    e = {"event": "open", "owners": ["proofs"], "severity": "warn", "text": f.text, "action": M.FAST, "monitor": "fast-preempt-n1",
         "by": "<!here>"}
    text = M.message(e, lambda h: f"<!subteam^S1|@{h}>")
    assert text.startswith("<!subteam^S1|@proofs> *Node 1:") and "<!" not in text[len("<!subteam^S1|@proofs>"):]
    assert "(&lt;!channel&gt;, run r1)" in text and text.endswith(", &lt;!here&gt;_")
    assert M.message({**e, "event": "clear", "text": "*Cleared:* <!channel> & co"}) == "*Cleared:* &lt;!channel&gt; &amp; co"
    posts = fake_slack(monkeypatch)
    M.relay_pass(rstate := {}, [src := FakeSource("node1")], M.slack_post(False), T0)
    src.all.append(json.dumps({**e, "key": "fast-preempt-n1|/run/gpu-lease", "text": "*a <!channel> & b*"}))
    M.relay_pass(rstate, [src], M.slack_post(False), T0 + 60)
    assert posts[1][1] == "@proofs a !channel &amp; b (alert: https://slack.example/p)"


def test_a_finding_with_a_thread_is_one_reply_in_that_thread_of_the_front_door_tagging_its_owners(monkeypatch, tmp_path):
    runs, st, alerts = ["r1"], {}, tmp_path / "alerts"
    m = M.Monitor("failed-checks", "infra", "warn", "relay", lambda ctx: [
        M.Finding(f"n1:{r}", f"*Check `{r}` failed*", owners=("circuits",), thread="1790.5") for r in runs], "fix it", events=True)
    M.step(st, [m], lambda memo: M.Ctx(now=T0, memo=memo), T0, M.AlertDir(alerts), by="test")
    runs.append("r2")
    M.step(st, [m], lambda memo: M.Ctx(now=T0 + 60, memo=memo), T0 + 60, M.AlertDir(alerts), by="test")
    (e,) = events(alerts)
    assert e["thread"] == "1790.5" and e["event"] == "event"
    posts = fake_slack(monkeypatch)
    src, rstate = FakeSource("relay"), {}
    M.relay_pass(rstate, [src], M.slack_post(False), T0)
    src.all.append(json.dumps(e))
    M.relay_pass(rstate, [src], M.slack_post(False), T0 + 60)
    assert posts == [("CFRONT", "@circuits *Check `r2` failed*", "1790.5")]


# ---- failed checks ---------------------------------------------------------------------------------------------------------------

LANDER_ID = "bc-8ece7cde-78d8-5ed9-84b0-a0a81b19f628"
REG = {"handles": [{"handle": "infra", "holder": "the infra coordinator (bc-17cc41f1-6227-5fab-bc64-3fa2f224558b)"},
                   {"handle": "circuits", "holder": "the circuits coordinator (bc-b8aaadaa-aefd-503e-9863-5eabb3a28c79)"},
                   {"handle": M.LANDER, "holder": f"the lander: the research coordinator ({LANDER_ID})"}]}


def write_run(root: Path, rid: str, *, command, state="failed", rc=1, steps=(), logs=None, failure=None, by=""):
    d, req = root / "runs" / rid, root / "requests" / rid
    d.mkdir(parents=True)
    req.mkdir(parents=True)
    (req / "request.json").write_text(json.dumps({"command": command, "submitted_by": by, "cwd": str(root / "src"),
                                                  "remote": {"source_sha": "abc123"}}))
    (d / "status.json").write_text(json.dumps({"state": state, "transitions": [
        {"state": "running"}, {"state": state, "rc": rc, "ended_utc": "2026-10-03T20:30:00.000Z"}]}))
    (d / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps) + '{"step": "half-writ')
    for name, text in (logs or {}).items():
        (d / name).write_text(text)
    if failure:
        (d / "failure.json").write_text(json.dumps(failure))


def test_the_reader_lists_slot_checks_with_a_failed_step_or_a_bad_end_and_each_ones_first_failing_line(tmp_path):
    slot = ["python3", "tools/verity/check/slot.py", "--", "uv", "run", "python", "tools/verity/check/check.py"]
    log = "collected 3\nE   AssertionError: x\nsomething failed here\nFAILED tests/test_a.py::test_b - AssertionError\n"
    write_run(tmp_path, "r20261003-200000-aaaa", command=slot, state="running", rc=None,
              steps=[{"step": "lints", "status": "passed"}, {"step": "pytest", "status": "failed", "rc": 1, "log": "pytest.log",
                                                             "ended_utc": "2026-10-03T20:20:00Z"}], logs={"pytest.log": log})
    write_run(tmp_path, "r20261003-200000-bbbb", command=slot, state="failed", rc=143,
              failure={"class": "CANCELLED_DEADMAN", "why": "no heartbeat for 10 min"})
    write_run(tmp_path, "r20261003-200000-cccc", command=slot, state="done", rc=0, steps=[{"step": "pytest", "status": "passed"}])
    write_run(tmp_path, "r20261003-200000-dddd", command=["python3", "tools/verity/check/check.py"], state="failed")
    write_run(tmp_path, "r20261003-200000-1111", command=slot, state="failed", rc=M.NOT_STARTED)  # a twin that left its line
    write_run(tmp_path, "r20261003-200000-eeee", command=slot, state="running", rc=None,
              steps=[{"step": "lean-audit", "status": "failed", "log": "lean-audit.log"}],
              logs={"lean-audit.log": "AUDIT x: FAIL; built\n  lean: failed\n    error: dependency 'p' not in manifest\n"})
    write_run(tmp_path, "r20261002-100000-ffff", command=slot)                   # a day the reader doesn't list
    old = tmp_path / "runs" / "r20261003-100000-ffff"
    write_run(tmp_path, old.name, command=slot)
    for f in old.iterdir():
        os.utime(f, (old.stat().st_mtime - M.FAILED_RECENT_S - 60,) * 2)
    line = tmp_path / "check-line"
    line.mkdir()
    (line / "1791-1").write_text(json.dumps({"pid": 1, "run": "r20261003-203000-wait", "by": "lander"}))
    (line / "1792-2").write_text(json.dumps({"pid": 2, "run": "r20261003-203100-dead"}))      # nobody holds it: it died
    held = os.open(line / "1791-1", os.O_RDONLY)
    fcntl.flock(held, fcntl.LOCK_EX)
    script = M.failed_reader(T0, runs=str(tmp_path / "runs"), requests=str(tmp_path / "requests"), line=str(line))
    got = json.loads(subprocess.run([sys.executable, "-"], input=script, capture_output=True, text=True, check=True).stdout)
    os.close(held)
    assert got["waiting"] == [{"run": "r20261003-203000-wait", "by": "lander"}]
    rows = {r["run"][-4:]: r for r in got["rows"]}
    assert set(rows) == {"aaaa", "bbbb", "eeee"}
    assert rows["aaaa"]["step"] == "pytest" and rows["aaaa"]["line"] == "FAILED tests/test_a.py::test_b - AssertionError"
    assert rows["eeee"]["line"] == "error: dependency 'p' not in manifest"
    assert rows["bbbb"].get("step") is None and rows["bbbb"]["class"] == "CANCELLED_DEADMAN" and rows["bbbb"]["rc"] == 143


def test_the_reader_names_a_suites_runs_failed_suite_as_it_happens_from_its_stdout(tmp_path):
    """A suites.py run writes no steps.jsonl: its first failed suite, and that suite's first failing line, come from stdout.log
    while it still runs (25ea, 8 Oct: verity-ops failed at 23:54Z and nobody heard until the run ended at 01:01Z)."""
    slot = ["python3", "tools/verity/check/slot.py", "--", "bash", "-c", "uv run python tools/verity/check/suites.py --changed x"]
    out = ("suites: the change from x touches 3\npassed  verity-numerical   1093 passed in 39 s\n"
           "failed  verity-ops         61 passed, 1 failed in 5 s\n        E       assert False\n"
           "        FAILED tests/test_keeper.py::test_a - assert False\n        log: /tmp/x.log\n"
           "passed  research           1955 passed in 35 s\nfailed  verity-later   1 failed\n")
    write_run(tmp_path, "r20261003-200000-aaaa", command=slot, state="running", rc=None, logs={"stdout.log": out})
    (tmp_path / "runs" / "r20261003-200000-aaaa" / "steps.jsonl").unlink()
    write_run(tmp_path, "r20261003-200000-bbbb", command=slot, state="running", rc=None,
              logs={"stdout.log": "passed  verity-numerical   1093 passed in 39 s\n"})
    script = M.failed_reader(T0, runs=str(tmp_path / "runs"), requests=str(tmp_path / "requests"), line=str(tmp_path / "none"))
    got = json.loads(subprocess.run([sys.executable, "-"], input=script, capture_output=True, text=True, check=True).stdout)
    (row,) = got["rows"]
    assert row["run"].endswith("aaaa") and row["step"] == "suite verity-ops" and row["suite"]
    assert row["line"] == "FAILED tests/test_keeper.py::test_a - assert False" and row["state"] == "running"


def test_a_trains_prs_are_its_merges_from_the_head_and_a_merge_of_main_between_them():
    assert M.train_prs(["Merge main (2546d8362)", "Merge #1497 (7a4b5b0a2)", "Merge #1541 (fbf9d25b2) onto 0928: x",
                        "Merge 0058fab8ec7f (0058fab8)", "Merge #1462 (c3a6dd31)"]) == [1497, 1541]
    assert M.train_prs(["fix the test", "Merge #1 (ab)"]) == []
    assert M.lane_of(REG, "cursor/node-lockout-558b") == "infra" and M.lane_of(REG, "cursor/x-81c4") is None
    assert M.lane_of(REG, "main") is None


def test_a_failed_check_rings_the_lane_of_each_open_pr_and_the_lander_in_the_merge_thread_once_per_run(monkeypatch):
    monkeypatch.setattr(M, "_registry", lambda: REG)
    train = ["Merge #1466 (ea2f79691)", "Merge #1460 (759ec3fb2)", "Merge #1561 (c5bb5d4ed)", "Merge 0058fab8ec7f (0058fab8)"]
    rows = {"n1": [{"run": "r1", "state": "running", "step": "pytest", "step_ended": "2026-10-03T20:20:00Z", "line": "FAILED `t`",
                    "subjects": train, "sha": "705ca7269313", "by": "someone"},
                   {"run": "r2", "state": "failed", "rc": 143, "class": "CANCELLED_MANUAL", "subjects": train, "by": LANDER_ID},
                   {"run": "r3", "state": "failed", "rc": 1, "step": "pytest", "subjects": ["a lane's fix"], "sha": "f00",
                    "by": "https://cursor.com/agents/bc-b8aaadaa-aefd-503e-9863-5eabb3a28c79"}],
            "vy-n2": [{"run": "r4", "state": "failed", "rc": 137, "class": "CGROUP_OOM", "why": "the OOM killer took pytest",
                       "ended": "2026-10-03T20:30:00.000Z", "subjects": ["a PR's head"], "sha": "beef", "by": LANDER_ID}]}
    prs = {"1466": ("OPEN", "cursor/node-lockout-558b"), "1460": ("OPEN", "cursor/kueue-shadow-n1-558b"),
           "1561": ("MERGED", "cursor/x-8c79")}
    calls = []

    def answer(argv, input=None, **kw):
        calls.append(argv)
        out = (255, "", "ssh: connect to host cpu1: Connection timed out") if argv[-2] == "vy-nebius-cpu-1" else \
            (0, rows[argv[-2]] if isinstance(rows[argv[-2]], str) else json.dumps({"rows": rows[argv[-2]], "waiting": []}), "") \
            if argv[0] == "ssh" else \
            (0, json.dumps(dict(zip(("state", "headRefName"), prs[argv[5]]))), "") if argv[:5] == ["gh", "-R", M.REPO_SLUG, "pr", "view"] else \
            (0, json.dumps([{"number": 1570, "headRefName": "cursor/c4-8c79"}]), "")
        assert argv[0] == "ssh" and argv[-1].endswith("nice -n 19 ionice -c3 python3 -") or argv[:4] == ["gh", "-R", M.REPO_SLUG, "pr"]
        return subprocess.CompletedProcess(argv, *out)

    memo: dict = {}
    got = {f.key: f for f in M.failed_checks(M.Ctx(now=T0, memo=memo, run=answer))}
    assert set(got) == {"vy-nebius-1:r1", "vy-nebius-2:r4"}
    f = got["vy-nebius-1:r1"]
    assert f.owners == ("infra", M.LANDER) and f.thread == M.MERGE_THREAD
    assert f.text == ("*Check `r1` failed step `pytest` on vy-nebius-1* at 20:20Z (the run goes on): its log's first failing line "
                      "is `FAILED 't'`. Train at `705ca7269`: #1466 `cursor/node-lockout-558b`, #1460 `cursor/kueue-shadow-n1-558b`.")
    f = got["vy-nebius-2:r4"]
    assert f.owners == ("circuits", M.LANDER, "infra")
    assert f.text.startswith("*Check `r4` was killed on vy-nebius-2* at 20:30Z, before any step failed (CGROUP_OOM): the OOM killer")
    assert "PR at `beef`: #1570 `cursor/c4-8c79`." in f.text
    n = len(calls)
    assert set(f.key for f in M.failed_checks(M.Ctx(now=T0 + 60, memo=memo, run=answer))) == set(got)
    assert [c[0] for c in calls[n:]] == ["ssh"] * 3
    for n in ("n1", "vy-n2"):
        rows[n] = "not json"
    with pytest.raises(RuntimeError, match="vy-nebius-cpu-1: RuntimeError"):
        M.failed_checks(M.Ctx(now=T0 + 120, memo=memo, run=answer))


# ---- what node 1 runs ---------------------------------------------------------------------------------------------------------

def test_the_files_deploy_toml_installs_on_node_1_run_every_node_side_check(tmp_path):
    """Catches a node-side check importing a module deploy.toml doesn't install: on node 1 that is a pass that fails every
    minute, which only the relay's heartbeat would notice."""
    manifest = tomllib.loads((SRC / "research/pods/nebius/deploy.toml").read_text())
    lib = "/usr/local/lib/vy-monitoring/"
    files = {e["path"]: e["src"] for e in manifest["file"] if e["path"].startswith(lib + "research/")}
    assert lib + "research/monitors.py" in files
    for path, src in files.items():
        dest = tmp_path / path[len(lib):]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / src).read_bytes())
    code = ("import research.monitors as M; from research.pods.nebius import n1_alerts; M._probe(); "
            "print(','.join(m.id for m in M.MONITORS if m.where == 'node'))")
    r = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, cwd=tmp_path,
                       env={"PYTHONPATH": str(tmp_path), "PATH": os.environ.get("PATH", "")})
    assert r.returncode == 0, r.stderr
    assert set(r.stdout.strip().split(",")) == {m.id for m in M.MONITORS if m.where == "node"}
    unit = (REPO / files[lib + "research/monitors.py"]).parent / "pods/nebius/monitoring/vy-monitors.service"
    text = unit.read_text()
    assert f"Environment=PYTHONPATH={lib.rstrip('/')}" in text and "-m research.monitors run --by vy-monitors@vy-nebius-1 --where node" in text
    for name in ("vy-monitors.service", "vy-monitors.timer"):
        assert any(e["path"] == f"/etc/systemd/system/{name}" for e in manifest["file"])


# ---- every machine: memory, OOM kills, peers, lost checks -----------------------------------------------------------------------

def machine_root(tmp_path, scopes: dict[str, tuple[str | None, int]], *, full=0.0, avail_kb=900 * 2**20, total_kb=1000 * 2**20):
    """A fake machine under `tmp_path`: PSI, meminfo, and USER_SLICE's session scopes, each {name: (run id or None, bytes)}."""
    (tmp_path / "proc/pressure").mkdir(parents=True, exist_ok=True)
    (tmp_path / "proc/pressure/memory").write_text(f"some avg10=1.00 avg60={full:.2f} avg300=0.00 total=1\n"
                                                   f"full avg10=0.00 avg60={full:.2f} avg300=0.00 total=1\n")
    (tmp_path / "proc/meminfo").write_text(f"MemTotal:       {total_kb} kB\nMemFree:        1 kB\nMemAvailable:   {avail_kb} kB\n")
    for i, (name, (rid, nbytes)) in enumerate(scopes.items()):
        d = tmp_path / M.USER_SLICE.lstrip("/") / name
        d.mkdir(parents=True, exist_ok=True)
        pid = str(1000 + i)
        (d / "cgroup.procs").write_text(pid + "\n")
        (d / "memory.current").write_text(f"{nbytes}\n")
        argv = ["python3", "-m", "research", "run", "--runs-dir", "/r", "--id", rid] if rid else ["tmux", "new-session"]
        (tmp_path / "proc" / pid).mkdir(parents=True, exist_ok=True)
        (tmp_path / "proc" / pid / "cmdline").write_text("\0".join(argv) + "\0")
    return tmp_path


def mctx(root, run=None, memo=None, now=T0, **kw) -> M.Ctx:
    return M.Ctx(now=now, memo={} if memo is None else memo, run=run or Run(), root=root, machine="vy-nebius-1", **kw)


def test_memory_pressure_names_the_biggest_runs_once_every_task_stalls_or_little_is_left(tmp_path):
    scopes = {"session-1.scope": (None, 0), "session-2.scope": ("r20261008-000440-0c41", 551 * 2**30),
              "session-3.scope": ("r20261008-002727-b62e", 132 * 2**30)}
    assert M.memory_pressure(mctx(machine_root(tmp_path, scopes, full=2))) == []
    (f,) = M.memory_pressure(mctx(machine_root(tmp_path, scopes, full=25)))
    assert f.key == "memory" and "25% of the last minute" in f.text
    assert f.text.index("r20261008-000440-0c41 551G") < f.text.index("r20261008-002727-b62e 132G")
    (f,) = M.memory_pressure(mctx(machine_root(tmp_path, scopes, full=0, avail_kb=2 * 2**20)))
    assert "0.2% of its 1000G available" in f.text


def test_an_oom_kill_is_named_with_the_run_its_scope_held_and_who_sent_it_even_after_the_scope_is_gone(tmp_path):
    rid = "r20261008-000440-0c41"
    root = machine_root(tmp_path, {"session-5807.scope": (rid, 500 * 2**30)})
    req = root / M.REQUESTS.lstrip("/") / rid
    req.mkdir(parents=True)
    (req / "request.json").write_text(json.dumps({"command": ["python3", "tools/verity/check/slot.py", "--by",
                                                              "https://cursor.com/agents/bc-x", "--", "uv"]}))
    quiet = Run((has("journalctl"), (0, "1791420000.1 host kernel: Linux version\n", "")))
    memo: dict = {}
    assert M.oom_kills(mctx(root, quiet, memo)) == []
    shutil.rmtree(root / M.USER_SLICE.lstrip("/") / "session-5807.scope")
    line = ("1791421183.523 host kernel: oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=/,mems_allowed=0-1,"
            "oom_memcg=/user.slice/user-1001.slice/session-5807.scope,task_memcg=/user.slice/user-1001.slice/session-5807.scope,"
            "task=python3,pid=4132925,uid=1001\n")
    (f,) = M.oom_kills(mctx(root, Run((has("journalctl"), (0, line, ""))), memo, now=T0 + 60))
    assert f.key == "1791421183:4132925"
    assert f"run {rid}, sent by https://cursor.com/agents/bc-x" in f.text and "session-5807.scope" in f.text
    blind = Run((has("journalctl"), (0, "", "Hint: You are currently not seeing messages from other users and the system.")))
    with pytest.raises(RuntimeError, match="can't read the kernel's log"):
        M.oom_kills(mctx(root, blind, memo))


def test_disk_forecast_pages_when_the_fitted_line_reaches_the_watermark_within_6_hours_and_names_the_biggest_live_runs(
        tmp_path, monkeypatch):
    rid, quiet = "r20261008-000440-0c41", "r20261008-002727-b62e"
    root = machine_root(tmp_path, {"session-1.scope": (rid, 1), "session-2.scope": (quiet, 1), "session-3.scope": (None, 1)})
    for r in (rid, quiet):
        (root / M.RUNS.lstrip("/") / r).mkdir(parents=True)
    req = root / M.REQUESTS.lstrip("/") / rid
    req.mkdir(parents=True)
    (req / "request.json").write_text(json.dumps({"submitted_by": "https://cursor.com/agents/bc-x"}))
    use = {"pct": 70.0, "ipct": 50.0, "size_b": 5 * 2**40}
    monkeypatch.setattr(M, "disk_use", lambda path: dict(use))
    du = Run((has("du"), (124, f"412\t/r/{rid}\n3\t/r/{quiet}\n", "du: cut off")))
    memo: dict = {}
    for i in range(3):                                   # 3 readings over 20 minutes: too few for a line
        use["pct"] = 70.0 + 2 * i
        assert M.disk_forecast(mctx(root, du, memo, now=T0 + 600 * i)) == []
    use["pct"] = 76.0                                    # 12 points/h: 85% in 0.75 h
    (f,) = M.disk_forecast(mctx(root, du, memo, now=T0 + 1800))
    assert f.key == "disk" and "disk use reaches 85% in about 0.8 h*: 76.0% now" in f.text and "+12.00 points/h (+614 GiB/h)" in f.text
    assert f.text.index(f"{rid} 412G (sent by https://cursor.com/agents/bc-x)") < f.text.index(f"{quiet} 3G")
    (argv, _), = [c for c in du.calls if "du" in c[0]]
    assert argv[:2] == ["timeout", "55"] and "nice" in argv and str(root / M.RUNS.lstrip("/") / rid) in argv
    use["pct"] = 76.0                                    # flat for 3 hours: the old rise falls out of the fit
    for k in range(1, 19):
        found = M.disk_forecast(mctx(root, du, memo, now=T0 + 1800 + 600 * k))
    assert found == [] and len(memo["readings"]) == 19             # 3 hours of 10-minute readings, both ends kept
    use["ipct"] = 81.0
    (f,) = M.disk_forecast(mctx(root, du, memo, now=T0 + 20000))
    assert f.key == "inode" and "inode use is past 80%" in f.text


MANIFEST = """
[[file]]
path = "/etc/systemd/system/vy-a.service"
nodes = ["vy-nebius-1", "vy-nebius-cpu-1"]
run = ["systemd:vy-a", "systemd:vy-b.timer"]
[[file]]
path = "/home/research/.config/systemd/user/u.service"
nodes = ["vy-nebius-1"]
run = ["user-systemd:u", "tmux:x"]
[[file]]
path = "/etc/systemd/system/vy-held.service"
nodes = ["vy-nebius-1"]
run = ["systemd:vy-held"]
hold = "waits for its owner"
[[tree]]
path = "/opt/t"
nodes = ["vy-nebius-2"]
run = ["systemd:vy-n2-only"]
"""


def show(*units: tuple[str, str, str]) -> str:
    return "\n\n".join(f"LoadState={load}\nUnitFileState={uf}\nActiveState={act}\nSubState={act}" for load, uf, act in units) + "\n"


def test_the_units_deploy_toml_runs_on_a_machine_are_its_run_entries_less_held_ones_tmux_and_expected_off(monkeypatch):
    assert M.expected_units("vy-nebius-1", MANIFEST) == ["systemd:vy-a.service", "systemd:vy-b.timer", "user-systemd:u.service"]
    assert M.expected_units("vy-nebius-2", MANIFEST) == ["systemd:vy-n2-only.service"]
    monkeypatch.setitem(M.EXPECTED_OFF, ("vy-nebius-1", "systemd:vy-b.timer"), "stopped by design")
    assert M.expected_units("vy-nebius-1", MANIFEST) == ["systemd:vy-a.service", "user-systemd:u.service"]
    assert M.expected_units("vy-nebius-cpu-1", MANIFEST) == ["systemd:vy-a.service", "systemd:vy-b.timer"]
    real = set(M.expected_units("vy-nebius-1", M.MANIFEST.read_text()))
    assert real >= {"systemd:vy-watchdog.timer", "systemd:vy-lease.service", "systemd:vy-monitors.timer"}
    assert "systemd:vy-n2-offload.service" not in real and all(M.EXPECTED_OFF.values())


def test_units_expected_flags_a_unit_disabled_stopped_or_never_installed(tmp_path, monkeypatch):
    manifest = tmp_path / "deploy.toml"
    manifest.write_text(MANIFEST)
    monkeypatch.setattr(M, "MANIFEST", manifest)
    system = show(("loaded", "disabled", "active"), ("loaded", "enabled", "active"))
    user = show(("loaded", "enabled", "inactive"))
    run = Run((lambda a, _i: "--user" in a and a[-1] == "u.service", (0, user, "")),
              (lambda a, _i: a[:2] == ["systemctl", "show"] and a[-2:] == ["vy-a.service", "vy-b.timer"], (0, system, "")))
    a, u = M.units_expected(mctx(tmp_path, run))
    assert a.key == "systemd:vy-a.service" and "unit `vy-a.service` isn't enabled (disabled), so it won't start at boot*" in a.text
    assert u.key == "user-systemd:u.service" and "the user unit `u.service` isn't running (inactive/inactive)*" in u.text
    gone = Run((lambda a, _i: "--user" in a, (0, show(("loaded", "enabled", "active")), "")),
               (has("show"), (0, show(("loaded", "enabled", "active"), ("not-found", "", "inactive")), "")))
    [g] = M.units_expected(mctx(tmp_path, gone))
    assert g.key == "systemd:vy-b.timer" and "unit `vy-b.timer` was never installed (systemd doesn't know it)*" in g.text
    assert "isn't running" not in g.text
    ok = Run((lambda a, _i: "--user" in a, (0, show(("loaded", "enabled", "active")), "")),
             (has("show"), (0, show(("loaded", "static", "active"), ("loaded", "enabled", "active")), "")))
    assert M.units_expected(mctx(tmp_path, ok)) == []
    with pytest.raises(RuntimeError, match="exited 1"):
        M.units_expected(mctx(tmp_path, Run((has("show"), (1, "", "Failed to connect to bus")))))


def test_a_peer_that_doesnt_answer_ssh_is_a_finding_and_one_that_does_isnt(tmp_path):
    run = Run((lambda a, _i: a[0] == "ssh" and "vy-nebius-cpu-1" in a, (255, "", "ssh: connect to host: Connection timed out")),
              (lambda a, _i: a[0] == "ssh", (0, "", "")))
    (f,) = M.answering(mctx(tmp_path, run, peers=("vy-nebius-cpu-1", "vy-nebius-cpu-2")))
    assert f.key == "vy-nebius-cpu-1" and "Connection timed out" in f.text and "isn't answering vy-nebius-1" in f.text


def test_the_daily_line_counts_yesterdays_full_checks_infrastructure_ended_and_pages_past_the_target(tmp_path):
    now = 1791441000.0                     # 2026-10-08T06:30Z: the line is 2026-10-07's
    root = machine_root(tmp_path, {"session-9.scope": ("r20261007-230000-0004", 1)})
    def run(rid, cls, state, cmd):
        d = root / M.RUNS.lstrip("/") / rid
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"transitions": [{"state": "running"}] + ([{"state": state}] if state else [])}))
        if cls:
            (d / "failure.json").write_text(json.dumps({"class": cls}))
        q = root / M.REQUESTS.lstrip("/") / rid
        q.mkdir(parents=True)
        (q / "request.json").write_text(json.dumps({"command": cmd, "harness_args": []}))
    check = ["python3", "tools/verity/check/slot.py", "--by", "x", "--", "uv", "run"]
    run("r20261007-100000-0001", "SUCCESS", "done", check)
    run("r20261007-110000-0002", "CGROUP_OOM_ATTRIBUTED", "failed", check)
    run("r20261007-120000-0003", None, None, check)                       # its machine went down under it
    run("r20261007-230000-0004", None, None, check)                       # still running in session-9
    run("r20261007-130000-0005", "CGROUP_OOM_ATTRIBUTED", "failed", ["bash", "-c", "pytest"])
    run("r20261008-010000-0006", "CGROUP_OOM_ATTRIBUTED", "failed", check)
    (f,) = M.lost_checks(mctx(root, now=now))
    assert f.key == "2026-10-07" and f.severity == "page"
    assert "2 of 4 full checks lost to infrastructure (50.0%" in f.text
    assert "r20261007-110000-0002 (CGROUP_OOM_ATTRIBUTED)" in f.text and "r20261007-120000-0003 (never ended)" in f.text


def test_every_nebius_machine_runs_the_watchdog_from_the_monitoring_lib_with_its_own_name_and_peers():
    manifest = tomllib.loads((SRC / "research/pods/nebius/deploy.toml").read_text())
    rows = {e["path"]: e for e in manifest["file"]}
    for path in ("/etc/systemd/system/vy-watchdog.service", "/etc/systemd/system/vy-watchdog.timer",
                 "/usr/local/lib/vy-monitoring/research/monitors.py", "/usr/local/lib/vy-monitoring/research/__init__.py",
                 "/usr/local/lib/vy-monitoring/research/pods/nebius/deploy.toml"):
        assert {"vy-nebius-1", "vy-nebius-cpu-1"} <= set(rows[path]["nodes"]), path
    text = (SRC / "research/pods/nebius/monitoring/vy-watchdog.service").read_text()
    assert "-m research.monitors run --where machine $VY_WATCHDOG_ARGS" in text
    assert "EnvironmentFile=/etc/vy/watchdog.env" in text and "SupplementaryGroups=systemd-journal" in text
    assert re.search(r"^OOMScoreAdjust=-\d+$", text, re.M)
    assert {m.id for m in M.MONITORS if m.where == "machine"} == {"memory-pressure", "oom-kills", "answering", "disk-forecast",
                                                                    "units-expected", "lost-checks"}


def test_a_check_whose_step_failed_is_cancelled_once_a_ticket_waits_for_a_slot_on_its_machine_and_the_post_says_so(monkeypatch):
    monkeypatch.setattr(M, "_registry", lambda: REG)
    row = {"run": "r1", "state": "running", "step": "lean-audit", "step_ended": "2026-10-03T20:20:00Z", "line": "error: x",
           "subjects": ["Merge #1466 (ea2f79691)"], "sha": "705ca7269313", "by": LANDER_ID}
    waiting, calls = [], []

    def answer(argv, input=None, **kw):
        calls.append(argv)
        if argv[0] == "ssh":
            return subprocess.CompletedProcess(argv, 0, json.dumps({"rows": [row] if argv[-2] == "n1" else [],
                                                                    "waiting": waiting if argv[-2] == "n1" else []}), "")
        if argv[:2] == ["gh", "-R"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps({"state": "OPEN", "headRefName": "cursor/x-558b"}), "")
        assert argv[3:7] == ["cancel", "r1", "--on", "vy-nebius-1"] and "--by" in argv
        return subprocess.CompletedProcess(argv, 0, "research cancel: r1: SIGTERM", "")

    memo: dict = {}
    (f,) = M.failed_checks(M.Ctx(now=T0, memo=memo, run=answer, act=True))
    assert "Cancelled" not in f.text and not any("cancel" in c for c in calls)
    waiting.append({"run": "r20261003-203000-5415", "by": "lander"})
    assert len(M.failed_checks(M.Ctx(now=T0 + 60, memo={**memo}, run=answer))) == 1      # `check` acts on nothing
    assert not any("cancel" in c for c in calls)
    got = M.failed_checks(M.Ctx(now=T0 + 120, memo=memo, run=answer, act=True))
    assert [g.key for g in got] == ["vy-nebius-1:r1", "vy-nebius-1:r1:cancelled"] and got[0].text == f.text
    assert got[1].text == ("*Check `r1` on vy-nebius-1*: Cancelled at 20:42Z, since r20261003-203000-5415 waited for a check slot "
                           "on vy-nebius-1.") and got[1].owners == ("infra", M.LANDER)
    assert [c for c in calls if "cancel" in c][0][-1].startswith("failed-checks: step lean-audit failed while 1 waited")
    n = len(calls)
    assert len(M.failed_checks(M.Ctx(now=T0 + 180, memo=memo, run=answer, act=True))) == 2 and not any("cancel" in c for c in calls[n:])
    (f,) = M.failed_checks(M.Ctx(now=T0, memo={}, run=answer, act=True))
    assert f.text.endswith(" Cancelled at 20:40Z, since r20261003-203000-5415 waited for a check slot on vy-nebius-1.")
    row["suite"], calls[:] = True, []                     # a suites.py run's failed suite rings, and the lander decides
    (f,) = M.failed_checks(M.Ctx(now=T0, memo={}, run=answer, act=True))
    assert "Cancelled" not in f.text and not any("cancel" in c for c in calls)
