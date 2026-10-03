"""Node 1's stall alerts (`pods/nebius/n1_alerts.py`): which reads of node 1 are incidents, and that each is posted once to its
owner and cleared once in its thread; and that a change to a Kueue ClusterQueue's quotas is posted and logged once."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

NEBIUS = Path(__file__).resolve().parents[1] / "src/research/pods/nebius"
_spec = importlib.util.spec_from_file_location("nebius_n1_alerts", NEBIUS / "n1_alerts.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

T0 = 1790900000.0   # 2026-10-02T00:13:20Z
OK = "2026-10-02T00:13:00Z tick {\"gpu_util_10m\": 4.6}"
# node 1's `kubectl get clusterqueues -o json --show-managed-fields` at 16:41Z on 3 Oct, trimmed to the names, managedFields
# and spec.resourceGroups
QUEUES = json.loads((Path(__file__).parent / "fixtures/nebius/clusterqueues-managed.json").read_text())["items"]
GPU = "resourceGroups[0].flavors[vy-nebius-1].resources[nvidia.com/gpu]"


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


def queues(**edits) -> list:
    """QUEUES with `edits[name](queue)` applied to a copy of each named ClusterQueue (one that returns None is deleted)."""
    out = []
    for q in copy.deepcopy(QUEUES):
        q = edits.get(q["metadata"]["name"], lambda q: q)(q)
        if q is not None:
            out.append(q)
    return out


def resource(q: dict, name: str) -> dict:
    return next(r for r in q["spec"]["resourceGroups"][0]["flavors"][0]["resources"] if r["name"] == name)


def patched(q: dict, name: str, field: str, value, at: str) -> dict:
    """`kubectl patch` of one resource's field at `at`, as the API server records it: kubectl-patch owns resourceGroups."""
    resource(q, name)[field] = value
    for f in q["metadata"]["managedFields"]:
        if f["manager"] == "kubectl-patch":
            f["time"], f["fieldsV1"]["f:spec"]["f:resourceGroups"] = at, {}
    return q


class Watch:
    """watch_quotas over a list of reads (None: node 1 unread; a str: kubectl's error), recording posts and log appends."""

    def __init__(self):
        self.state, self.posts, self.log, self.said = {"open": {}}, [], [], []

    def tick(self, read, now=T0) -> None:
        s = None if read is None else {**snap(now=now), **({"queues": None, "queues_error": read} if isinstance(read, str)
                                                          else {"queues": read})}
        self.said.append(A.watch_quotas(self.state, s, now, lambda who, text, thread: self.posts.append((who, text, thread)),
                                        self.log.extend))


def test_the_fixture_reads_as_node_1s_quotas_and_who_owns_them():
    q = A.quotas(QUEUES)
    assert sorted(q) == ["circuits", "deployments-cpu", "deployments-gpu", "provers"]
    assert q["provers"]["groups"][f"{GPU}.nominalQuota"] == "6" and q["provers"]["groups"][f"{GPU}.borrowingLimit"] == "6"
    assert q["provers"]["by"] == ("kubectl-patch", "2026-10-03T15:07:46Z")
    # the quiet hour's later stopPolicy patch is kubectl-patch's, but circuits' resourceGroups are still apply's
    assert q["circuits"]["by"] == ("kubectl-client-side-apply", "2026-09-30T16:14:48Z")
    assert q["circuits"]["groups"]["resourceGroups[0].coveredResources"] == "[nvidia.com/gpu, cpu, memory]"


def test_the_first_read_records_the_baseline_and_an_unchanged_read_posts_nothing():
    w = Watch()
    w.tick(QUEUES)
    assert w.posts == [] and w.log == [] and w.said == ["baseline of 4 ClusterQueues"]
    assert set(w.state["quotas"]) == {"circuits", "deployments-cpu", "deployments-gpu", "provers"}
    w.state = json.loads(json.dumps(w.state))           # the loop's state file between ticks
    w.tick(queues(provers=lambda q: (resource(q, "nvidia.com/gpu").update(nominalQuota="6"), q)[1]), T0 + 60)
    assert w.posts == [] and w.log == [] and w.said[-1] == "unchanged"   # 6 as `apply` writes it, "6" as `patch` does


def test_a_nominal_quota_change_is_posted_to_infra_and_logged_with_the_field_old_new_and_manager():
    w = Watch()
    w.tick(QUEUES)
    w.tick(queues(provers=lambda q: patched(q, "nvidia.com/gpu", "nominalQuota", 2, "2026-10-02T00:13:41Z")), T0 + 60)
    line = (f"ClusterQueue `provers`: `{GPU}.nominalQuota` 6 -> 2 "
            "(spec.resourceGroups owner: kubectl-patch, last write 2026-10-02T00:13:41Z)")
    assert w.posts == [(["infra"], line, None)]
    assert w.log == [f"2026-10-02T00:14:20Z n1-alerts: {line}"]
    w.tick(queues(provers=lambda q: patched(q, "nvidia.com/gpu", "nominalQuota", 2, "2026-10-02T00:13:41Z")), T0 + 120)
    assert len(w.posts) == 1 and len(w.log) == 1


def test_a_borrowing_limit_change_is_posted_and_logged_likewise():
    w = Watch()
    w.tick(QUEUES)
    w.tick(queues(**{"deployments-gpu": lambda q: patched(q, "cpu", "borrowingLimit", "8", "2026-10-02T00:13:50Z")}), T0 + 60)
    line = ("ClusterQueue `deployments-gpu`: `resourceGroups[0].flavors[vy-nebius-1].resources[cpu].borrowingLimit` 72 -> 8 "
            "(spec.resourceGroups owner: kubectl-patch, last write 2026-10-02T00:13:50Z)")
    assert [t for _, t, _ in w.posts] == [line] and w.log == [f"2026-10-02T00:14:20Z n1-alerts: {line}"]


def test_one_read_with_several_changes_is_one_post_of_one_line_each():
    w = Watch()
    w.tick(QUEUES)

    def swap(gpus):
        return lambda q: patched(q, "nvidia.com/gpu", "nominalQuota", gpus, "2026-10-02T00:14:00Z")

    w.tick(queues(provers=swap(2), **{"deployments-gpu": swap(6)}), T0 + 60)
    assert len(w.posts) == 1 and w.posts[0][1].splitlines() == [ln.split(" n1-alerts: ", 1)[1] for ln in w.log]
    assert [ln.split("`")[1] for ln in w.log] == ["deployments-gpu", "provers"]
    assert "2 -> 6" in w.log[0] and "6 -> 2" in w.log[1]


def test_a_new_or_deleted_cluster_queue_is_reported():
    w = Watch()
    w.tick(QUEUES)
    backfill = copy.deepcopy(next(q for q in QUEUES if q["metadata"]["name"] == "circuits"))
    backfill["metadata"]["name"] = "backfill"
    w.tick(queues() + [backfill], T0 + 60)
    w.tick(queues(**{"deployments-cpu": lambda q: None}) + [backfill], T0 + 120)
    new, gone = (t for _, t, _ in w.posts)
    assert new.startswith(f"ClusterQueue `backfill` is new: `resourceGroups[0].coveredResources` [nvidia.com/gpu, cpu, memory], "
                          f"`{GPU}.nominalQuota` 0,")
    assert new.endswith("(spec.resourceGroups owner: kubectl-client-side-apply, last write 2026-09-30T16:14:48Z)")
    assert gone.startswith("ClusterQueue `deployments-cpu` was deleted; it had ") and f"`{GPU}.borrowingLimit` 0" in gone
    assert len(w.log) == 2 and set(w.state["quotas"]) == {"backfill", "circuits", "deployments-gpu", "provers"}


def test_a_kubectl_failure_posts_nothing_and_keeps_the_baseline():
    w = Watch()
    w.tick("The connection to the server 127.0.0.1:6443 was refused")
    w.tick(None, T0 + 30)
    assert "quotas" not in w.state and w.posts == []         # no baseline from a failed read
    w.tick(QUEUES, T0 + 60)
    baseline = copy.deepcopy(w.state["quotas"])
    w.tick("The connection to the server 127.0.0.1:6443 was refused", T0 + 120)
    w.tick(None, T0 + 180)
    assert w.posts == [] and w.log == [] and w.state["quotas"] == baseline
    assert w.said[-2] == "kubectl failed (The connection to the server 127.0.0.1:6443 was refused); kept the baseline"
    w.tick(queues(provers=lambda q: patched(q, "nvidia.com/gpu", "nominalQuota", 2, "2026-10-02T00:16:00Z")), T0 + 240)
    assert len(w.posts) == 1 and "6 -> 2" in w.posts[0][1]


def test_a_failed_post_or_append_is_made_again_at_the_next_read_and_the_other_not():
    state, posts, log, slack_up, ssh_up = {"open": {}}, [], [], [False], [True]

    def post(who, text, thread):
        if not slack_up[0]:
            raise RuntimeError("slack down")
        posts.append(text)

    def append(lines):
        if not ssh_up[0]:
            raise RuntimeError("ssh exited 255")
        log.extend(lines)

    def tick(read, now):
        return A.watch_quotas(state, {**snap(now=now), "queues": read}, now, post, append)

    tick(QUEUES, T0)
    said = tick(queues(provers=lambda q: patched(q, "nvidia.com/gpu", "nominalQuota", 2, "2026-10-02T00:14:00Z")), T0 + 60)
    assert said == "1 changed; post failed, retried at the next read: RuntimeError: slack down"
    assert len(log) == 1 and posts == []
    slack_up[0], ssh_up[0] = True, False
    said = tick(queues(provers=lambda q: patched(q, "nvidia.com/gpu", "nominalQuota", 3, "2026-10-02T00:15:00Z")), T0 + 120)
    assert said.startswith("1 changed; log failed")
    assert [p.count("\n") for p in posts] == [1] and "6 -> 2" in posts[0] and "2 -> 3" in posts[0]
    ssh_up[0] = True
    tick(queues(provers=lambda q: patched(q, "nvidia.com/gpu", "nominalQuota", 3, "2026-10-02T00:15:00Z")), T0 + 180)
    assert len(posts) == 1 and [ln.split()[0] for ln in log] == ["2026-10-02T00:14:20Z", "2026-10-02T00:15:20Z"]
    assert state["quota_unsent"] == {"log": [], "post": []}


def test_the_remote_read_trims_kubectls_output_to_what_the_watch_reads(monkeypatch):
    """REMOTE runs on node 1; here its kubectl is faked with the fixture plus a status-subresource entry it must drop, and
    it sees no pool, pane or loop.log."""
    import os
    import subprocess

    def no_file(*a, **kw):
        raise OSError("not on node 1")

    full = copy.deepcopy(QUEUES)
    full[0]["metadata"]["managedFields"].append({"manager": "kueue", "operation": "Update", "subresource": "status",
                                                 "time": "2026-10-03T16:23:18Z", "fieldsV1": {"f:status": {}}})
    full[0]["status"] = {"admittedWorkloads": 0}

    class Done:
        def __init__(self, argv):
            self.returncode = 0 if argv[:4] == A.KUBECTL else 1
            self.stdout, self.stderr = (json.dumps({"items": full}) if argv[:4] == A.KUBECTL else ""), "no tmux"

    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: Done(argv))
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    printed: list[str] = []
    exec(compile(A.REMOTE, "remote", "exec"), {"print": printed.append, "open": no_file})
    out = json.loads(printed[0])
    assert out["blocked"] == {} and out["pane"] is None and out["log"] is None
    assert A.quotas(out["queues"]) == A.quotas(QUEUES)
    assert all(not f.get("subresource") for q in out["queues"] for f in q["metadata"]["managedFields"])
    assert "status" not in out["queues"][0]
