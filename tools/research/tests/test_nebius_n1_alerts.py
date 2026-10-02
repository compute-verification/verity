"""Node 1's stall alerts (`pods/nebius/n1_alerts.py`): which reads of node 1 are incidents, and that each is posted once to its
owner and cleared once in its thread."""
from __future__ import annotations

import importlib.util
from pathlib import Path

NEBIUS = Path(__file__).resolve().parents[1] / "src/research/pods/nebius"
_spec = importlib.util.spec_from_file_location("nebius_n1_alerts", NEBIUS / "n1_alerts.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

T0 = 1790900000.0   # 2026-10-02T00:13:20Z
OK = "2026-10-02T00:13:00Z tick {\"gpu_util_10m\": 4.6}"


def fail(stamp: str) -> str:
    return f"2026-10-02T{stamp}Z tick failed: kubectl create: jobs.batch \"nd-x\" already exists"


def snap(ticks=(OK,), blocked=None, now=T0, log=None) -> dict:
    return {"now": now, "blocked": blocked or {}, "pane": None if ticks is None else list(ticks), "log": log}


def test_the_remote_read_compiles_and_matches_the_dispatchers_tick_lines():
    compile(A.REMOTE, "remote", "exec")
    assert A.TICK_RE.match(OK)[2] is None
    assert A.TICK_RE.match(fail("00:12:00"))[2].startswith("kubectl create")
    assert not A.TICK_RE.match("submitted nd-vllm-epoch-run-f6f9d7d87d-replay-0 (vllm-epoch-run/cov-gm294 task 2 try 0)")


def test_incidents_are_a_blocked_pool_three_failed_ticks_a_stalled_or_unseen_dispatcher():
    assert A.incidents(snap()) == {}
    two = A.incidents(snap([OK, fail("00:12:00"), fail("00:13:00")]))
    assert two == {}
    three = A.incidents(snap([fail("00:11:00"), fail("00:12:00"), fail("00:13:00")]))
    who, text, _ = three["dispatch"]
    assert who == ["circuits", "infra"] and "00:11:00Z to 2026-10-02T00:13:00Z" in text and "already exists" in text
    assert "hasn't ticked since 2026-10-02T00:13:00Z" in A.incidents(snap(now=T0 + 601))["dispatch"][1]
    assert "can't be seen" in A.incidents(snap(ticks=None))["dispatch"][1]
    b = A.incidents(snap(blocked={"/run/gpu-lease-circuits": "t pid 41 (circuits-x) ran on GPU 3 without a lease"}))
    who, text, _ = b["blocked:/run/gpu-lease-circuits"]
    assert who == ["circuits"] and "pid 41" in text and "GPU 3" in text
    assert A.incidents(snap(blocked={"/run/gpu-lease": "t"}))["blocked:/run/gpu-lease"][0] == ["proofs"]


def test_the_dispatchers_ticks_are_the_pane_and_loop_log_together():
    late = T0 + 601
    blank_pane = snap(ticks=[OK], now=late, log=["2026-10-02T00:22:00Z tick {}", "2026-10-02T00:23:00Z tick {}"])
    assert A.incidents(blank_pane) == {}
    assert "since 2026-10-02T00:23:00Z" in A.incidents(snap(ticks=None, now=T0 + 1300, log=blank_pane["log"]))["dispatch"][1]
    failed_in_pane = snap([fail("00:11:00"), fail("00:12:00"), fail("00:13:00")], log=["2026-10-02T00:10:00Z tick {}"])
    assert "failed its last 3 ticks" in A.incidents(failed_in_pane)["dispatch"][1]
    assert A.last_ticks(snap([OK], log=[OK])) == [OK]


def test_each_incident_is_posted_once_and_cleared_once_in_its_thread():
    posts: list[tuple] = []

    def post(who, text, thread):
        posts.append((tuple(who), text.split("*")[1], thread))
        return f"ts{len(posts)}"

    state = {"open": {}}
    blocked = {"/run/gpu-lease": "t pid 7 (provers-y) ran on GPU 0 without a lease"}
    A.step(state, snap(blocked=blocked), T0, post)
    A.step(state, snap(blocked=blocked), T0 + 60, post)
    assert posts == [(("proofs",), "Node 1's lease pool `/run/gpu-lease` is blocked", None)]
    A.step(state, None, T0 + 120, post)            # a failed read clears nothing
    assert len(posts) == 1
    A.step(state, snap(now=T0 + 180), T0 + 180, post)
    assert posts[1] == (("proofs",), "Cleared:", "ts1") and state["open"] == {}


def test_node_1_unreadable_for_ten_minutes_is_infras_incident_until_a_read_works():
    posts: list[tuple] = []
    state = {"open": {}}
    A.step(state, snap(), T0, lambda who, text, thread: posts.append((who, text, thread)))
    for dt in (60, 300, 600):
        A.step(state, None, T0 + dt, lambda who, text, thread: posts.append((who, text, thread)))
    assert posts == []
    A.step(state, None, T0 + 660, lambda who, text, thread: posts.append((who, text, thread)))
    A.step(state, None, T0 + 720, lambda who, text, thread: posts.append((who, text, thread)))
    assert [(w, t[:44]) for w, t, _ in posts] == [(["infra"], "*Infra's node 1 alert relay hasn't read node")]
    A.step(state, snap(["2026-10-02T00:26:00Z tick {}"], now=T0 + 780), T0 + 780, lambda who, text, thread: posts.append((who, text, thread)))
    assert posts[-1][1].startswith("*Cleared:* the relay reads node 1 again") and state["open"] == {}


def test_a_post_that_fails_is_made_again_at_the_next_read():
    calls: list[str] = []

    def flaky(who, text, thread):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("slack down")
        return "ts"

    state = {"open": {}}
    bad = snap([fail("00:11:00"), fail("00:12:00"), fail("00:13:00")])
    try:
        A.step(state, bad, T0, flaky)
    except RuntimeError:
        pass
    assert state["open"] == {}
    A.step(state, bad, T0 + 60, flaky)
    assert len(calls) == 2 and set(state["open"]) == {"dispatch"}
