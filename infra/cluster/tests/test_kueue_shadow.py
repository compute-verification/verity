"""Node 1 with Kueue as its only scheduler, in shadow: the queues it mirrors, the demand it reads, and the divergences."""
from __future__ import annotations

import json

from cluster import kueue_shadow as K
from cluster.nebius2 import Observation

CQS = [{"metadata": {"name": "provers"}, "spec": {
    "cohortName": "nebius", "namespaceSelector": {}, "queueingStrategy": "BestEffortFIFO", "stopPolicy": "None",
    "preemption": {"reclaimWithinCohort": "Any"},
    "resourceGroups": [{"coveredResources": ["cpu", "memory", "nvidia.com/gpu"], "flavors": [{"name": "vy-nebius-1", "resources": [
        {"name": "cpu", "nominalQuota": 64}, {"name": "memory", "nominalQuota": "384Gi"},
        {"name": "nvidia.com/gpu", "nominalQuota": 2, "borrowingLimit": 6}]}]}]}}]
SLOTS = {"slots": [{"name": "c", "cpus": "8-31", "trains": False, "holder": None},
                   {"name": "a", "cpus": "32-63", "trains": True, "holder": {"pid": 11, "since": "2026-10-07T08:00Z", "run": "r-train"}},
                   {"name": "b", "cpus": "64-95", "trains": False, "holder": {"pid": 12, "since": "2026-10-07T08:05Z"}}],
         "waiters": [{"since": "2026-10-07T08:10Z", "priority": False, "pid": 13, "run": "r-lane"}]}
LEAN = {"pools": {"check": {"n": 2, "slots": [{"index": 0, "holder": {"pid": 21, "since": "2026-10-07T08:00Z"}},
                                              {"index": 1, "holder": None}], "waiters": []},
                  "audit": {"n": 1, "slots": [{"index": 0, "holder": None}], "waiters": [{"since": "2026-10-07T08:12Z", "pid": 22}]}}}
WPC = {"dev": 100, "circuits-gpu": 600}


def wl(name: str, queue: str = "provers", admitted: bool = True, pod_labels: dict | None = None, finished: bool = False) -> dict:
    conds = [{"type": "QuotaReserved", "status": "True" if admitted else "False", "message": "couldn't assign flavors"}]
    if finished:
        conds.append({"type": "Finished", "status": "True"})
    return {"metadata": {"name": name, "creationTimestamp": "2026-10-07T07:00:00Z"},
            "spec": {"queueName": queue, "priority": 300,
                     "priorityClassRef": {"group": "kueue.x-k8s.io", "kind": "WorkloadPriorityClass", "name": "prover-bench"},
                     "podSets": [{"name": "main", "count": 1, "template": {"metadata": {"labels": pod_labels or {}}, "spec": {"containers": [
                         {"name": "c", "resources": {"requests": {"cpu": "4"}}}]}}}]},
            "status": {"conditions": conds}}


LEASES = {"node1": Observation(0, {0: "pid=100 run=r1 who=w since=2026-10-07T08:01:00Z cmd=x",
                                    1: "pid=100 run=r1 who=w since=2026-10-07T08:01:00Z cmd=x",
                                    2: "pid=5 who=kubernetes since=2026-10-07T00:00:00Z cmd=fence"},
                                 (("wait.1.200", "pid=200 n=1 run=r2 who=w since=2026-10-07T08:02:00Z"),)),
          "circuits": Observation(0, {})}


def test_queues_copy_the_nodes_into_their_own_cohort() -> None:
    objs = K.queues(CQS, SLOTS, LEAN)
    cqs = {o["metadata"]["name"]: o for o in objs if o["kind"] == "ClusterQueue"}
    assert set(cqs) == {"shadow-provers", "shadow-checks", "shadow-lean"}
    p = cqs["shadow-provers"]["spec"]
    assert p["cohortName"] == "vy-shadow" and p["namespaceSelector"] == {"matchLabels": {"vy.dev/shadow": "true"}}
    assert p["resourceGroups"] == CQS[0]["spec"]["resourceGroups"] and p["preemption"] == {"reclaimWithinCohort": "Any"}
    assert all(o["metadata"]["labels"] == {"vy.dev/shadow": "true"} for o in objs)
    checks = cqs["shadow-checks"]["spec"]["resourceGroups"][0]["flavors"]
    assert [(f["name"], f["resources"][0]["nominalQuota"]) for f in checks] == [("vy-shadow-slots", 2), ("vy-shadow-train-slots", 1)]
    taints = {o["metadata"]["name"]: o["spec"] for o in objs if o["kind"] == "ResourceFlavor"}
    assert taints == {"vy-shadow-slots": {"nodeLabels": {K.SLOT_KIND: "any"}},
                      "vy-shadow-train-slots": {"nodeLabels": {K.SLOT_KIND: "trains"}, "nodeTaints": [K.TRAINS_TAINT]}}
    lean = cqs["shadow-lean"]["spec"]["resourceGroups"][0]
    assert lean["coveredResources"] == ["vy.dev/lean-audit", "vy.dev/lean-check"]
    lqs = {o["metadata"]["name"]: o["spec"]["clusterQueue"] for o in objs if o["kind"] == "LocalQueue"}
    assert lqs == {n: n for n in cqs}


def test_no_slot_files_no_slot_queues() -> None:
    assert {o["metadata"]["name"] for o in K.queues(CQS, None, None) if o["kind"] == "ClusterQueue"} == {"shadow-provers"}


def test_demand_reads_every_scheduler() -> None:
    items = {i.key: i for i in K.demand(
        [wl("w-run"), wl("w-wait", admitted=False), wl("w-done", finished=True),
         wl("holder", pod_labels={"verity.dev/gpu-pool": "node1"}), wl("elsewhere", queue="unknown")],
        {"provers": "provers"}, LEASES, WPC, SLOTS, LEAN)}
    assert set(items) == {"k8s/w-run", "k8s/w-wait", "lease/node1/gl-100", "lease/node1/gl-200",
                          "slot/r-train", "slot/12", "slot/r-lane", "lean/check/21", "lean/audit/22"}
    assert items["k8s/w-run"].running and not items["k8s/w-wait"].running
    assert items["k8s/w-run"].priority == 300 and items["k8s/w-run"].priority_class == "prover-bench"
    assert items["k8s/w-run"].queue == "shadow-provers"
    lease = items["lease/node1/gl-100"]
    assert lease.running and lease.priority == 100 and lease.priority_class == "dev"
    assert lease.pod_sets[0]["template"]["spec"]["containers"][0]["resources"]["requests"] == {"nvidia.com/gpu": "2"}
    assert not items["lease/node1/gl-200"].running
    train = items["slot/r-train"]
    assert train.trains and train.priority == 1 and train.pod_sets[0]["template"]["spec"]["tolerations"][0]["key"] == "vy.dev/trains"
    assert "tolerations" not in items["slot/r-lane"].pod_sets[0]["template"]["spec"] and not items["slot/r-lane"].running
    # a holder is pinned to the kind of slot it holds; a waiter may take either kind it tolerates
    assert train.pod_sets[0]["template"]["spec"]["nodeSelector"] == {K.SLOT_KIND: "trains"}
    assert items["slot/12"].pod_sets[0]["template"]["spec"]["nodeSelector"] == {K.SLOT_KIND: "any"}
    assert "nodeSelector" not in items["slot/r-lane"].pod_sets[0]["template"]["spec"]
    assert items["lean/audit/22"].pod_sets[0]["template"]["spec"]["containers"][0]["resources"]["requests"] == {"vy.dev/lean-audit": "1"}


def test_workload_is_bare_and_named_by_key() -> None:
    i = K.Item(key="lease/node1/gl-100", queue="shadow-provers", pod_sets=(K._pod_set({"nvidia.com/gpu": "1"}),),
               priority=100, running=True, since="t", priority_class="dev")
    w = K.workload(i)
    assert w["metadata"]["name"] == K.workload_name(i.key) and w["metadata"]["namespace"] == "vy-shadow"
    assert w["metadata"]["annotations"]["vy.dev/shadow-key"] == i.key
    assert w["spec"]["queueName"] == "shadow-provers" and w["spec"]["priorityClassRef"]["name"] == "dev"
    assert json.loads(json.dumps(w)) == w


def item(key: str, running: bool, since: str = "2026-10-07T08:00Z") -> K.Item:
    return K.Item(key=key, queue="shadow-checks", pod_sets=(K._pod_set({K.CHECK_SLOT: "1"}),), priority=0, running=running,
                  since=since)


def test_reconcile_creates_what_runs_before_waiters_and_deletes_the_gone() -> None:
    gone = {"metadata": {"name": K.workload_name("old")}}
    create, delete = K.reconcile([item("w", False, "08:00"), item("r2", True, "08:02"), item("r1", True, "08:01")], [gone])
    assert [w["metadata"]["annotations"]["vy.dev/shadow-key"] for w in create] == ["r1", "r2"]
    assert delete == [K.workload_name("old")]
    have = [{"metadata": {"name": K.workload_name(k)}} for k in ("r1", "r2")]
    create, delete = K.reconcile([item("w", False), item("r1", True), item("r2", True)], have)
    assert [w["metadata"]["annotations"]["vy.dev/shadow-key"] for w in create] == ["w"] and delete == []


def shadow(key: str, admitted: bool, since: str = "2026-10-07T08:00Z") -> dict:
    return {"metadata": {"name": K.workload_name(key), "annotations": {"vy.dev/shadow-since": since}}, "status": {"conditions": [
        {"type": "QuotaReserved", "status": "True" if admitted else "False", "message": "insufficient quota"}]}}


def test_compare_names_both_ways_the_node_and_kueue_differ() -> None:
    items = [item("agree-run", True), item("agree-wait", False), item("node-ran", True), item("kueue-ran", False),
             item("new", False)]
    ws = [shadow("agree-run", True), shadow("agree-wait", False), shadow("node-ran", False), shadow("kueue-ran", True)]
    first = K.compare("2026-10-07T08:30:00Z", items, ws)
    assert first["divergent"] == 0 and first["disagree"] == ["kueue-ran", "node-ran"]
    rec = K.compare("2026-10-07T08:31:00Z", items, ws, frozenset(first["disagree"]))
    assert rec["schema"] == "cluster/kueue-shadow/v1" and rec["items"] == 5 and rec["compared"] == 4
    assert [(d["key"], d["kind"]) for d in rec["divergences"]] == [("kueue-ran", "kueue-ran"), ("node-ran", "node-ran")]
    assert rec["divergences"][1]["kueue"] == "insufficient quota"


def test_a_holder_that_asks_again_is_a_new_request() -> None:
    """A run that let its slot go and queued again keeps its key; its admitted Workload is the old request's, so the shadow
    deletes it rather than counting the old admission against whoever the node gave the slot to."""
    again = item("lean/build/r1", False, "2026-10-07T10:33Z")
    old = shadow("lean/build/r1", True, "2026-10-07T10:04Z")
    rec = K.compare("2026-10-07T10:34:00Z", [again], [old], frozenset({"lean/build/r1"}))
    assert rec["compared"] == 0 and rec["divergent"] == 0
    create, delete = K.reconcile([again], [old])
    assert create == [] and delete == [K.workload_name("lean/build/r1")]
    gone = old | {"metadata": old["metadata"] | {"deletionTimestamp": "2026-10-07T10:34:01Z"}}
    assert K.compare("2026-10-07T10:35:00Z", [again], [gone])["compared"] == 0
    assert K.reconcile([again], [shadow("lean/build/r1", False, "2026-10-07T10:33Z")]) == ([], [])


def test_last_disagree_reads_the_records_last_tick(tmp_path) -> None:
    out = tmp_path / "rec.jsonl"
    assert K.last_disagree(out) == frozenset()
    out.write_text(json.dumps({"disagree": ["a"]}) + "\n" + json.dumps({"error": "x"}) + "\n")
    assert K.last_disagree(out) == frozenset()
    out.write_text(json.dumps({"disagree": ["a"]}) + "\n" + json.dumps({"disagree": ["b"]}) + "\n")
    assert K.last_disagree(out) == {"b"}


def test_tick_writes_only_shadow_objects(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    answers = {"clusterqueues": {"items": CQS}, "localqueues": {"items": [{"metadata": {"name": "provers"},
                                                                            "spec": {"clusterQueue": "provers"}}]},
               "workloadpriorityclasses": {"items": [{"metadata": {"name": "dev"}, "value": 100}]},
               "workloads": {"items": [wl("w-run")]}}

    def run(args: list[str], stdin: str | None = None) -> str:
        calls.append(args)
        if args[0] == "get":
            if "-n" in args and args[args.index("-n") + 1] == "vy-shadow":
                return json.dumps({"items": []})
            return json.dumps(answers[args[1].split(".")[0]])
        return ""

    applied: list[dict] = []
    monkeypatch.setattr(K, "_status", lambda script: None)
    pools = {}
    for pool, (_, cq, pc) in K.POOLS.items():
        (tmp_path / "leases" / pool).mkdir(parents=True)
        pools[pool] = (str(tmp_path / "leases" / pool), cq, pc)
    monkeypatch.setattr(K, "POOLS", pools)              # node 1's /run/gpu-lease holds live leases (744f, 9 Oct: created 2)
    rec = K.tick(tmp_path / "rec.jsonl", run=run, apply=applied.extend, delete=lambda names: None)
    assert all(c[0] == "get" for c in calls)
    assert all(o["metadata"]["labels"] == {"vy.dev/shadow": "true"} for o in applied)
    assert {o["metadata"].get("namespace") for o in applied if o["kind"] in ("Workload", "LocalQueue")} == {"vy-shadow"}
    assert rec["created"] == 1 and rec["compared"] == 0
    assert json.loads((tmp_path / "rec.jsonl").read_text())["items"] == 1
