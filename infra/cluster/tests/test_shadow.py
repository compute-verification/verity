"""The shadow against vy-nebius-2's own logs: a slice of 30 Sep (07:20-07:58Z, three timed windows and fifteen
preemptions) replayed through the adapter reproduces what `gpu-lease` did, and the pass bar sees what it must."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from cluster import ledger, model, nebius2, shadow
from cluster.ledger import epoch
from cluster.plan import Start, Wait

DESC = Path(__file__).parents[1] / "descriptions" / "nebius.toml"
FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def c() -> model.Cluster:
    return model.load(DESC)


def rows() -> list[dict]:
    return [json.loads(x) for x in gzip.open(FIX / "node2-util-0720-0758.jsonl.gz", "rt")]


def preempted() -> list[str]:
    return (FIX / "node2-preempted-0720-0758.log").read_text().splitlines()


def test_replaying_the_nodes_logs_reproduces_every_lease_and_every_preemption(c, tmp_path):
    rs, pre = rows(), preempted()
    shadow.replay(c, rs, pre, tmp_path)
    records = ledger.read(tmp_path / "ledger.jsonl")
    assert ledger.verify(records) == []

    seen: dict[str, list[int]] = {}
    prev: set[str] = set()
    for r in rs:
        obs = nebius2.from_sampler(r)
        cur = {a.id for a in nebius2.allocations(c, obs)}
        for aid in cur - prev:
            seen.setdefault(aid, [obs.t, None])
        for aid in prev - cur:
            if seen[aid][1] is None:
                seen[aid][1] = obs.t
        prev = cur
    started = [r["alloc"]["id"] for r in records if r["kind"] == "start"]
    assert len(started) == len(set(started)), "a stale owner line (a lease being taken) never restarts an ended lease"
    starts = {r["alloc"]["id"]: r["t"] for r in records if r["kind"] == "start"}
    ends = {r["alloc"]: (r["t"], r["kind"], r.get("by")) for r in records if r["kind"] in ("end", "evict")}
    first = {aid: t0 for aid, (t0, _) in seen.items()}
    assert {aid: t for aid, t in starts.items()} == first, "each lease starts in the ledger when the node first shows it"
    for aid, (_, t1) in seen.items():
        if t1 is not None and aid in starts:
            assert ends[aid][0] == t1, aid

    victims = {}
    for p in map(nebius2.preemption, pre):
        v = p["victim"]
        victims[f"gl-{v['pid']}@{epoch(v['since'])}"] = f"gl-{p['by']}"
    assert len(victims) == 15
    for aid, by in victims.items():
        assert ends[aid][1:] == ("evict", by), f"{aid} was stopped for {by}"

    last = nebius2.from_sampler(rs[-1])
    s = ledger.state(records)
    assert sorted(s.allocations) == sorted(a.id for a in nebius2.allocations(c, last))
    assert ledger.state_from_dict(json.loads((tmp_path / "state.json").read_text())) == s


def test_the_replay_passes_the_bar_with_every_window_started_within_five_seconds(c, tmp_path):
    shadow.replay(c, rows(), preempted(), tmp_path)
    ev = shadow.evaluate(tmp_path, [])
    assert ev["pass"], ev["checks"]
    assert ev["evidence"]["windows"] == 3 and not ev["evidence"]["enough"]
    assert ev["window_lag_s"]["n"] == 3 and ev["window_lag_s"]["max"] <= shadow.WINDOW_LAG_S
    assert ev["safety"] == [] and ev["design"], "design divergences are listed for review"


def _line(pid: int, since: str, extra: str = "") -> str:
    return f"pid={pid} run=- who=w{pid} since={since} {extra}".strip()


def test_a_one_gpu_timed_lease_beside_a_session_nobody_preempts_is_planned_as_gpu_lease_ran_it(c, tmp_path):
    """Until the clamp, a partial `--timed` lease starts beside what it needn't displace, as `gpu-lease`
    starts it, so the shadow agrees with `gpu-lease` and the node breaks no rule."""
    sh = shadow.Shadow(c, tmp_path)
    t = epoch("2026-09-30T10:00:00Z")
    sess = _line(11, "2026-09-30T09:59:00Z")
    sh.step(nebius2.Observation(t, {0: sess}))
    sh.step(nebius2.Observation(t + 10, {0: sess, 1: _line(12, "2026-09-30T10:00:05Z", "until=2026-09-30T10:05:05Z timed=1")}))
    div = tmp_path / "divergences.jsonl"
    assert not div.exists() or div.read_text() == ""


def test_a_request_left_waiting_beside_free_gpus_is_a_safety_divergence(c, tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "plan", lambda cl, node, allocs, queue, now: [Wait(j.id, "no reason") for j in queue])
    sh = shadow.Shadow(c, tmp_path)
    t = epoch("2026-09-30T10:00:00Z")
    w = (("wait.1.21", "pid=21 n=2 run=- who=w since=2026-09-30T10:00:00Z"),)
    for dt in (0, 300, 601, 700):
        sh.step(nebius2.Observation(t + dt, {}, w))
    div = [json.loads(x) for x in (tmp_path / "divergences.jsonl").read_text().splitlines()]
    assert [(d["class"], d["job"]) for d in div] == [("safety", "gl-21")]


def test_a_plan_that_breaks_a_rule_is_a_safety_divergence(c, tmp_path, monkeypatch):
    t = epoch("2026-09-30T10:00:00Z")
    held = _line(11, "2026-09-30T09:59:00Z")

    def bad(cl, node, allocs, queue, now):
        return [Start(model.Allocation(id=f"{j.id}@{now}", job=j, node=node, gpus=(0,), cpus=frozenset(), memory_gib=1,
                                       started=now)) for j in queue]
    monkeypatch.setattr(shadow, "plan", bad)
    sh = shadow.Shadow(c, tmp_path)
    sh.step(nebius2.Observation(t, {0: held}, (("wait.1.21", "pid=21 n=1 run=- who=w since=2026-09-30T10:00:00Z"),)))
    div = [json.loads(x) for x in (tmp_path / "divergences.jsonl").read_text().splitlines()]
    assert div and div[0]["class"] == "safety" and div[0]["msg"].startswith("the plan breaks a rule: vy-nebius-2: GPU 0")


def test_quiet_agreement_flags_a_pause_outside_every_window_and_a_late_resume():
    wins = [{"alloc": "gl-5@100", "pid": 5, "t0": 100, "t1": 400}]
    ok = [{"t": "1970-01-01T00:01:41Z", "event": "pause", "pgid": 9, "holder": 5},
          {"t": "1970-01-01T00:06:45Z", "event": "resume", "pgid": 9, "why": "window over"}]
    assert shadow.quiet_agreement(wins, ok, 30) == []
    late = ok[:1] + [{"t": "1970-01-01T00:08:00Z", "event": "resume", "pgid": 9, "why": "window over"}]
    stray = [{"t": "1970-01-01T00:20:00Z", "event": "pause", "pgid": 3, "holder": 6}]
    assert len(shadow.quiet_agreement(wins, late + stray, 30)) == 2


def test_a_windows_quiet_record_is_found_by_its_run_id_and_names_what_ran_beside_it(c, tmp_path, capsys):
    """PoUW's conditions: quiet is recorded per window and looked up by the window's run id, and a session a timed lease
    started beside is named (holder, GPUs, pid in its allocation id), so the panel can mark that window's rows noisy."""
    from cluster import cli
    sh = shadow.Shadow(c, tmp_path)
    t = epoch("2026-09-30T10:00:00Z")
    sess = _line(11, "2026-09-30T09:59:00Z").replace("run=-", "run=r-sess")
    win = _line(12, "2026-09-30T10:00:05Z", "until=2026-09-30T10:05:05Z timed=1").replace("run=-", "run=r-win")
    sh.step(nebius2.Observation(t, {0: sess}))
    sh.step(nebius2.Observation(t + 10, {0: sess, 1: win}))
    sh.step(nebius2.Observation(t + 20, {0: sess}))
    assert cli.main(["ledger", "quiet", str(tmp_path / "ledger.jsonl"), "r-win"]) == 2
    rep = json.loads(capsys.readouterr().out)
    assert rep["run"] == "r-win" and rep["gpus"] == [1] and not rep["quiet"]
    [o] = [o for o in rep["overlaps"] if o["state"] == "running"]
    assert (o["alloc"].split("@")[0], o["who"], o["gpus"], o["run"], o["preempt"]) == ("gl-11", "w11", [0], "r-sess", "never")
    assert cli.main(["ledger", "quiet", str(tmp_path / "ledger.jsonl"), "r-none"]) == 1
