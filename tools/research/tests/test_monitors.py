"""The monitors (`research/monitors.py`): the runner opens each incident once and clears it once, through the alert dir and an
optional webhook; each check reads its source the way the node writes it; node 2 is read in one session per pass, behind a
guard that fails closed; the relay posts every new event once; and the files deploy.toml installs on node 1 run every
node-side check."""
from __future__ import annotations

import json
import os
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
        assert m.owner in handles and m.severity in ("warn", "page") and m.where in ("node", "relay"), m.id
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


def test_a_post_that_fails_is_the_relays_first_next_pass():
    src, rstate, fail = FakeSource("node1"), {}, [True]
    M.relay_pass(rstate, [src], lambda e, t: {}, T0)
    src.all += [ev("open", "m|a"), ev("open", "m|b")]

    def post(e, thread):
        if e["key"] == "m|b" and fail[0]:
            raise RuntimeError("slack down")
        return {"ts": e["key"]}

    with pytest.raises(RuntimeError):
        M.relay_pass(rstate, [src], post, T0)
    assert rstate["sources"]["node1"]["lines"] == 1
    fail[0] = False
    assert M.relay_pass(rstate, [src], post, T0) == 1


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


def test_a_relay_post_tags_its_owners_and_says_what_to_do():
    e = json.loads(ev("open", "m|a", owners=["circuits", "infra"], severity="page"))
    text = M.message(e, lambda h: f"<{h}>")
    assert text.startswith("<circuits> <infra> [page] open m|a\n_What to do:_ a")


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
