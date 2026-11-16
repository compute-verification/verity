import json

import pytest
from _build import T0, alloc, job, one

from cluster import ledger
from cluster.plan import Idle, Start, apply, plan


def _window_day(path):
    """A CPU job, then a quiet window that freezes it, then the thaw; returns the ledger's records and the allocation ids."""
    c = one()
    lg = ledger.Ledger(path)
    cpu = job("o-1", preempt="freeze", max_s=3600)
    acts = plan(c, "n1", [], [cpu], T0)
    lg.actions(T0, acts)
    live = apply([], acts, T0)
    acts = plan(c, "n1", live, [job("w-1", gpus=8, priority="timed", quiet=True, max_s=600)], T0 + 100)
    lg.actions(T0 + 100, acts)
    live = apply(live, acts, T0 + 100)
    win = next(a for a in live if a.job.quiet)
    lg.append(T0 + 400, "end", alloc=win.id, rc=0, run="r20260930-000000-0000")
    live = [a for a in live if a.id != win.id]
    lg.actions(T0 + 400, plan(c, "n1", live, [], T0 + 400))
    return ledger.read(path), live[0].id, win.id


def test_chain_verifies_and_every_tamper_is_named(tmp_path):
    p = tmp_path / "ledger.jsonl"
    records, _, _ = _window_day(p)
    assert ledger.verify(records) == [] and [r["seq"] for r in records] == list(range(1, len(records) + 1))
    edited = [dict(r) for r in records]
    edited[1]["by"] = "someone-else"
    assert ledger.verify(edited) == ["record 2: hash does not match its content"]
    edited[1]["hash"] = ledger.digest(edited[1])
    assert ledger.verify(edited) == ["record 3: prev does not match record 2's hash"]
    assert ledger.verify(records[:1] + records[2:])[0] == "record 2: seq is 3"
    lines = p.read_text().splitlines()
    lines[0] = lines[0].replace('"t":1790000000', '"t":1790000001')
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="record 1: hash does not match"):
        ledger.Ledger(p)


def test_time_never_goes_back(tmp_path):
    lg = ledger.Ledger(tmp_path / "l.jsonl")
    lg.append(T0, "admin-begin", node="n1", who="nebius-owner", why="driver")
    with pytest.raises(ValueError, match="older than the ledger's last"):
        lg.append(T0 - 1, "admin-end", node="n1")


def test_quiet_report_certifies_a_window_and_catches_what_broke_it(tmp_path):
    records, cpu, win = _window_day(tmp_path / "l.jsonl")
    rep = ledger.quiet_report(records, win)
    assert rep["quiet"] and rep["running"] == [(T0 + 100, T0 + 400)]
    assert rep["overlaps"] == [{"alloc": cpu, "state": "frozen", "t0": T0 + 100, "t1": T0 + 400, "quiet_safe": False, "who": "o",
                               "gpus": [], "run": None, "preempt": "freeze"}]
    hist, _ = ledger.history(records)
    assert hist[cpu].intervals == [["running", T0, T0 + 100], ["frozen", T0 + 100, T0 + 400], ["running", T0 + 400, None]]
    lg = ledger.Ledger(tmp_path / "l.jsonl")
    lg.append(T0 + 500, "admin-begin", node="n1", who="nebius-owner", why="clocks")
    rep = ledger.quiet_report(ledger.read(tmp_path / "l.jsonl"), win)
    assert rep["quiet"]
    busy = [dict(r) for r in records if not (r["kind"] == "freeze")]
    assert not ledger.quiet_report(busy, win)["quiet"]


def test_admin_session_during_a_window_voids_its_quiet(tmp_path):
    lg = ledger.Ledger(tmp_path / "l.jsonl")
    w = alloc(job("w-1", gpus=8, priority="timed", quiet=True), tuple(range(8)), "0-1")
    lg.actions(T0, [Start(w)])
    lg.append(T0 + 60, "admin-begin", node="n1", who="nebius-owner", why="repair")
    lg.append(T0 + 90, "admin-end", node="n1")
    rep = ledger.quiet_report(ledger.read(tmp_path / "l.jsonl"), w.id, until=T0 + 600)
    assert not rep["quiet"] and rep["admin"][0]["who"] == "nebius-owner"


def test_failures_feed_names_job_submitter_and_node(tmp_path):
    lg = ledger.Ledger(tmp_path / "l.jsonl")
    a = alloc(job("a-1", gpus=1), (0,), "0-1")
    lg.actions(T0, plan(one(), "n1", [], [a.job], T0))
    aid = f"a-1@{T0}"
    lg.actions(T0 + 700, [Idle(aid, "a", 700)])
    lg.append(T0 + 800, "end", alloc=aid, rc=3, run="r1")
    lg.actions(T0 + 900, plan(one(), "n1", [], [job("b-1", max_s=None)], T0 + 900))
    feed = ledger.failures(ledger.read(tmp_path / "l.jsonl"))
    assert [f["what"] for f in feed][:2] == ["refused: every job but a service declares max_s, the time it is killed at", "exit 3"]
    assert feed[1] == {"t": T0 + 800, "what": "exit 3", "run": "r1", "job": "a-1", "submitter": "a", "node": "n1"}
    assert feed[2]["what"] == "GPUs idle 700 s"
    assert ledger.failures(ledger.read(tmp_path / "l.jsonl"), since=T0 + 850)[0]["job"] == "b-1"
    assert json.loads(json.dumps(feed)) == feed


def test_gpu_usage_counts_unsampled_time_as_held_not_idle():
    samples = [(T0 + 10 * i, i < 6) for i in range(30)]
    assert ledger.gpu_usage(T0, T0 + 600, samples, 10) == {"held_s": 600, "busy_s": 60, "sampled_s": 300}
    assert ledger.gpu_usage(T0 + 5, T0 + 25, samples, 10) == {"held_s": 20, "busy_s": 20, "sampled_s": 20}


def test_usage_ranks_holders_by_busy_share(tmp_path):
    lg = ledger.Ledger(tmp_path / "l.jsonl")
    for k, (who, busy) in enumerate((("a", 60), ("b", 540), ("a", 0))):
        t = T0 + 1000 * k
        a = alloc(job(f"{who}-{busy}", gpus=1), (0,), "0-1", started=t)
        lg.actions(t, [Start(a)])
        samples = [(t + 10 * i, 10 * i < busy) for i in range(60)]
        lg.append(t + 600, "end", alloc=a.id, rc=0, gpus=[{"index": 0, "uuid": "", **ledger.gpu_usage(t, t + 600, samples, 10)}])
    rows = ledger.usage(ledger.read(tmp_path / "l.jsonl"))
    assert [(r["submitter"], r["leases"], r["busy_share"]) for r in rows] == [("a", 2, 0.05), ("b", 1, 0.9)]


# a gpu-lease/usage/v1 record as node 2's gpu-lease writes it (infra/nebius 1ebbd156): busy and sampled count whole 10 s
# readings, so they can pass held_s by up to one cadence
USAGE = {"schema": "gpu-lease/usage/v1", "host": "vy-nebius-2", "pid": 2235138, "scope": "gpu-lease-2235138", "who": "bc-ccd30e80",
         "run": "r20260930-190711-c584", "timed": False, "preemptible": True, "since": "2026-09-30T19:07:33Z",
         "end": "2026-09-30T19:22:33Z", "rc": 0, "cmd": "timeout 900 bash window.sh",
         "gpus": [{"index": 3, "uuid": "GPU-9f1f172d", "held_s": 900, "busy_s": 905, "sampled_s": 910}]}


def test_usage_v1_records_become_end_records_and_feed_the_usage_report(tmp_path):
    assert ledger.end_fields(USAGE) == {"rc": 0, "run": "r20260930-190711-c584",
                                        "gpus": [{"index": 3, "uuid": "GPU-9f1f172d", "held_s": 900, "busy_s": 900, "sampled_s": 900}]}
    assert ledger.usage_key(USAGE) == (2235138, ledger.epoch("2026-09-30T19:07:33Z"))
    with pytest.raises(ValueError, match="not a gpu-lease/usage/v1 record"):
        ledger.end_fields({"schema": "other"})
    lg = ledger.Ledger(tmp_path / "l.jsonl")
    j = job("l-1", gpus=1, cpus=0, submitter="bc-ccd30e80")
    acts = plan(one(), "n1", [], [j], T0)
    lg.actions(T0, acts, source="gpu-lease")
    lg.end(T0 + 900, acts[0].alloc.id, usage=USAGE)
    records = ledger.read(tmp_path / "l.jsonl")
    assert records[0]["source"] == "gpu-lease" and records[1]["rc"] == 0 and records[1]["run"] == USAGE["run"]
    (row,) = ledger.usage(records)
    assert row["submitter"] == "bc-ccd30e80" and row["busy_share"] == 1.0


def test_state_comes_from_the_ledger_alone_and_state_json_round_trips(tmp_path):
    records, cpu, win = _window_day(tmp_path / "l.jsonl")
    st = ledger.state(records)
    assert list(st.allocations) == [cpu] and st.allocations[cpu].state == "running" and st.allocations[cpu].frozen_s == 300
    assert st.seq == len(records) and st.head == records[-1]["hash"] and st.queue == {}
    half = ledger.state(records[:3])
    assert ledger.state(records[3:], half) == st
    lg = ledger.Ledger(tmp_path / "l.jsonl")
    lg.submit(T0 + 500, job("q-1", gpus=1), "n1", handle="/run/gpu-lease/wait.1.42")
    lg.submit(T0 + 501, job("q-2", gpus=1), "n1")
    lg.withdraw(T0 + 502, "q-2", "gave up")
    st = ledger.state(ledger.read(tmp_path / "l.jsonl"))
    assert [(q.job.id, q.handle, q.since) for q in st.queue.values()] == [("q-1", "/run/gpu-lease/wait.1.42", T0 + 500)]
    ledger.write_state(tmp_path / "state.json", st)
    assert ledger.state_from_dict(json.loads((tmp_path / "state.json").read_text())) == st
    assert [p.name for p in tmp_path.iterdir()] and not list(tmp_path.glob(".state.json.*"))
