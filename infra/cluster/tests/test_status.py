"""`cluster status`: node 1's Kueue objects and node 2's ledger state as rows by owner and key, GPU-h ready, holds, disk and
release.py's bundle count, from snapshots in the shapes the live nodes return (30 Sep / 1 Oct 2026)."""
from __future__ import annotations

import json
import re

from _build import job

from cluster import cli, ledger, model, nebius1, nebius2, status
from cluster.ledger import epoch

T = epoch("2026-10-01T00:30:00Z")
DF = ("Filesystem     1024-blocks       Used  Available Capacity Mounted on\n"
      "/dev/vdc        5259290696 3638812520 1620461792      70% /workspace\n")
BUNDLE_A = "/workspace/jobs/probe-jit/cfgtp2-deferred-smol2/D/sweep/smollm2-135m__tp1__b1__i256__o32/commit/replay_bundle_p0"
BUNDLE_B = "/workspace/jobs/cov/cov-g176/sweep/qwen3-4b__tp1__b8__i256__o32/commit/replay_bundle_p0"
DU = f"561566596\t{BUNDLE_A}\n13000000000\t{BUNDLE_B}\n"


def kjob(name: str, item: str, **labels: str) -> dict:
    ws = item.split("/")[0]
    return {"kind": "Job", "metadata": {"name": name, "labels": {"verity.dev/workstream": ws, **labels},
                                        "annotations": {"verity.dev/item": item}}, "spec": {}}


def cond(type_: str, status_: str, at: str, reason: str = "") -> dict:
    return {"type": type_, "status": status_, "lastTransitionTime": at, "reason": reason}


def wl(owner: str, queue: str, *, gpus: int = 0, cpu: str = "4", memory: str = "170G", row: str | None = None,
       active: bool = True, conds: tuple = (), created: str = "2026-10-01T00:00:00Z", deadline: int | None = None,
       prio: str = "circuits-gpu") -> dict:
    req = {"cpu": cpu, "memory": memory, **({"nvidia.com/gpu": str(gpus)} if gpus else {})}
    env = [{"name": "OMP_NUM_THREADS", "value": cpu}, *([{"name": "ROW", "value": row}] if row else [])]
    spec = {"containers": [{"env": env, "resources": {"requests": req}}], **({"activeDeadlineSeconds": deadline} if deadline else {})}
    return {"kind": "Workload",
            "metadata": {"name": f"job-{owner}-x", "creationTimestamp": created, "labels": {"kueue.x-k8s.io/job-uid": "u"},
                         "ownerReferences": [{"kind": "Job", "name": owner}]},
            "spec": {"active": active, "queueName": queue, "priorityClassRef": {"name": prio},
                     "podSets": [{"count": 1, "template": {"metadata": {"labels": {}}, "spec": spec}}]},
            "status": {"conditions": list(conds)}}


def pod(name: str, job_name: str | None, start: str, *, gpus: int = 0, cpu: str = "4", phase: str = "Running",
        labels: dict | None = None) -> dict:
    meta = {"name": name, "labels": labels or {}}
    if job_name:
        meta["ownerReferences"] = [{"kind": "Job", "name": job_name}]
    req = {"cpu": cpu, "memory": "8G", **({"nvidia.com/gpu": str(gpus)} if gpus else {})}
    return {"kind": "Pod", "metadata": meta, "spec": {"containers": [{"resources": {"requests": req}}]},
            "status": {"phase": phase, "startTime": start}}


def queue(kind: str, name: str, stop: str | None) -> dict:
    return {"kind": kind, "metadata": {"name": name}, "spec": {**({"stopPolicy": stop} if stop else {})}}


G176 = "nd-vllm-epoch-run-2a25cdab54-gpu-0"
BF16 = "nd-proofs-bf16-hi-4ed9fadd78-prover-b-0"
FLOCK = "nd-proofs-flock-f-f75266031d-prover-b-0"
TP2 = "nd-vllm-epoch-run-cc6666e339-gpu-0"
N128 = "nd-vllm-epoch-run-07d3232f90-gpu-0"
G055 = "nd-n2-build-1cfb93a181-gpu-0"
DONE = "nd-vllm-epoch-run-0000000000-build-0"
SKY = "sky-jobs-controller-29a45b65-29a45b65-head"


def node1_snapshot() -> dict:
    items = [
        kjob(G176, "vllm-epoch-run/cov-g176"),
        wl(G176, "deployments-gpu", gpus=1, row="qwen3-4b__bf16__rtxpro6000__tp1__b8__i256__o32__mixed__greedy__bi-eager",
           conds=(cond("QuotaReserved", "True", "2026-10-01T00:20:00Z"), cond("Admitted", "True", "2026-10-01T00:20:00Z"))),
        pod(f"{G176}-2fzth", G176, "2026-10-01T00:21:56Z", gpus=1),
        kjob(BF16, "proofs-bf16-hill/bf16-hill-k8192-s0"),
        wl(BF16, "provers", gpus=1, cpu="16", prio="prover-bench",
           conds=(cond("QuotaReserved", "True", "2026-10-01T00:25:00Z"),)),
        pod(f"{BF16}-pend", BF16, "2026-10-01T00:25:01Z", gpus=1, cpu="16", phase="Pending"),
        kjob(FLOCK, "proofs-flock-fp/fp-hill-nvf4-k2048"),
        wl(FLOCK, "provers", gpus=1, cpu="16", prio="prover-bench", created="2026-10-01T00:10:00Z",
           conds=(cond("QuotaReserved", "False", "2026-10-01T00:22:00Z", "Pending"),
                  cond("Requeued", "True", "2026-10-01T00:22:00Z", "Preempted"))),
        kjob(TP2, "vllm-epoch-run/cov-p104"),
        wl(TP2, "deployments-gpu", gpus=2, cpu="8", deadline=7200, created="2026-10-01T00:28:00Z",
           row="qwen25-7b__bf16__rtxpro6000__tp2__b1__i256__o32__mixed__greedy__bi-eager"),
        kjob(N128, "vllm-epoch-run/cov-n128"),
        wl(N128, "deployments-gpu", gpus=1, active=False, created="2026-09-30T22:48:07Z",
           row="qwen25-15b__bf16__rtxpro6000__tp1__b32__i256__o32__mixed__greedy__bi-eager",
           conds=(cond("QuotaReserved", "False", "2026-09-30T23:00:00Z", "Inadmissible"),
                  cond("Evicted", "True", "2026-09-30T23:00:00Z", "Deactivated"))),
        kjob(G055, "n2-build/cov-g055", **{"verity.dev/owner": "circuits"}),
        wl(G055, "deployments-gpu", gpus=1, active=False, created="2026-09-30T22:32:59Z",
           row="llama32-1b__bf16__rtxpro6000__tp1__b32__i1024__o128__mixed__greedy__bi-eager"),
        kjob(DONE, "vllm-epoch-run/cov-g001"),
        wl(DONE, "deployments-cpu", conds=(cond("Finished", "True", "2026-09-30T20:00:00Z"),)),
        pod(SKY, None, "2026-09-30T05:30:24Z", cpu="2", labels={"skypilot-cluster": "sky-jobs-controller-29a45b65"}),
        pod("nd-old-pod", DONE, "2026-09-30T19:00:00Z", phase="Succeeded"),
        queue("ClusterQueue", "deployments-gpu", "Hold"),
        queue("ClusterQueue", "provers", "None"),
        queue("ClusterQueue", "backfill", None),
        queue("LocalQueue", "circuits", "HoldAndDrain"),
    ]
    return {"node": nebius1.NODE, "t": T, "k8s": items, "df": DF, "bundles": DU}


def by_key(rows: list[dict]) -> dict[str, dict]:
    return {r["key"]: r for r in rows}


def test_node1_rows_are_kueues_unfinished_workloads_by_owner_and_key_with_phase_batch_resources_and_time():
    rep = status.node1(node1_snapshot(), T)
    rows = by_key(rep["jobs"])
    assert set(rows) == {"vllm-epoch-run/cov-g176", "proofs-bf16-hill/bf16-hill-k8192-s0", "proofs-flock-fp/fp-hill-nvf4-k2048",
                         "vllm-epoch-run/cov-p104", SKY}
    g176 = rows["vllm-epoch-run/cov-g176"]
    assert (g176["owner"], g176["phase"], g176["batch"], g176["state"], g176["gpus"], g176["cpus"], g176["queue"]) == \
        ("vllm-epoch-run", "gpu", 8, "running", 1, 4, "deployments-gpu")
    assert g176["since"] == epoch("2026-10-01T00:21:56Z") and g176["for_s"] == T - g176["since"]
    bf16 = rows["proofs-bf16-hill/bf16-hill-k8192-s0"]
    assert (bf16["state"], bf16["phase"], bf16["since"], bf16["cpus"]) == ("admitted", "prover-b", epoch("2026-10-01T00:25:00Z"), 16)
    flock = rows["proofs-flock-fp/fp-hill-nvf4-k2048"]
    assert (flock["state"], flock["since"]) == ("queued", epoch("2026-10-01T00:22:00Z"))
    tp2 = rows["vllm-epoch-run/cov-p104"]
    assert (tp2["state"], tp2["gpus"], tp2["batch"], tp2["length_s"], tp2["length_is"]) == ("queued", 2, 1, 7200, "max")
    sky = rows[SKY]
    assert (sky["owner"], sky["state"], sky["gpus"], sky["cpus"], sky["queue"]) == ("skypilot", "running", 0, 2, None)
    assert [r["state"] for r in rep["jobs"] if r["owner"] == "vllm-epoch-run"] == ["running", "queued"]


def test_node1_ready_counts_queued_gpus_and_only_the_gpu_hours_a_length_declares():
    ready = status.node1(node1_snapshot(), T)["ready"]
    assert ready == {"jobs": 2, "gpus": 3, "gpu_jobs": 2, "known": 1, "gpu_h": 4.0}


def test_node1_holds_are_stopped_queues_and_deactivated_workloads_with_their_owner_label():
    holds = status.node1(node1_snapshot(), T)["holds"]
    assert holds["queues"] == [{"kind": "ClusterQueue", "name": "deployments-gpu", "stop_policy": "Hold"},
                               {"kind": "LocalQueue", "name": "circuits", "stop_policy": "HoldAndDrain"}]
    held = by_key(holds["workloads"])
    assert set(held) == {"vllm-epoch-run/cov-n128", "n2-build/cov-g055"}
    n128, g055 = held["vllm-epoch-run/cov-n128"], held["n2-build/cov-g055"]
    assert (n128["owner"], n128["owner_label"], n128["since"], n128["batch"]) == \
        ("vllm-epoch-run", None, epoch("2026-09-30T23:00:00Z"), 32)
    assert (g055["owner"], g055["owner_label"], g055["since"]) == ("circuits", "circuits", epoch("2026-09-30T22:32:59Z"))


def test_disk_is_dfs_capacity_and_bundles_are_release_pys_total_in_gb():
    rep = status.node1(node1_snapshot(), T)
    assert rep["disk"] == {"path": "/workspace", "use_pct": 70}
    b = rep["bundles"]
    assert (b["count"], b["gb"], b["roots"]) == (2, round((561566596 + 13000000000) / 1e9, 1), list(status.BUNDLE_ROOTS))
    assert [i["path"] for i in b["items"]] == [BUNDLE_B, BUNDLE_A]
    assert status.bundles("") == {"rule": "release.py bundles_gb", "roots": list(status.BUNDLE_ROOTS), "count": 0, "gb": 0.0,
                                  "items": []}


def test_the_bundle_command_is_release_pys_rule_and_fails_rather_than_reading_zero_without_sudo():
    cmd = status.COMMANDS["bundles"]
    assert cmd.startswith("sudo -n true && ")
    assert "find /workspace/jobs/cov /workspace/jobs/probe-jit -xdev -name 'replay_bundle_p*' -prune -print0" in cmd
    assert "xargs -0r du -sb" in cmd
    assert "if [ -x /usr/local/sbin/vy-sudo ]; then sudo -n /usr/local/sbin/vy-sudo du-bundles" in cmd


def test_owner_filters_jobs_and_held_workloads_but_not_the_nodes_holds_disk_or_bundles():
    rep = status.node1(node1_snapshot(), T, owner="vllm-epoch-run")
    assert [r["key"] for r in rep["jobs"]] == ["vllm-epoch-run/cov-g176", "vllm-epoch-run/cov-p104"]
    assert [r["key"] for r in rep["holds"]["workloads"]] == ["vllm-epoch-run/cov-n128"]
    assert len(rep["holds"]["queues"]) == 2 and rep["disk"]["use_pct"] == 70 and rep["bundles"]["count"] == 2
    assert rep["ready"] == {"jobs": 1, "gpus": 2, "gpu_jobs": 1, "known": 1, "gpu_h": 4.0}
    assert status.node1(node1_snapshot(), T, owner="circuits")["holds"]["workloads"][0]["key"] == "n2-build/cov-g055"


def node2_snapshot() -> dict:
    fill = job("gl-43527", "pous", submitter="fill:bc-7442ca43", priority="fill", gpus=1, cpus=0, max_s=480,
               node=nebius2.NODE, gpu_ids=(1,))
    sess = job("gl-16021", "pous", submitter="verity:bc-698052e1-cov-g217-proof", priority="session", kind="session",
               gpus=1, cpus=0, max_s=None, expected_s=1800, node=nebius2.NODE, run="r20260930-235745-a3d0")
    waiter = job("gl-57723", "pous", submitter="fill:bc-dbc19788", priority="fill", gpus=2, cpus=0, max_s=900,
                 node=nebius2.NODE)
    cpu = job("v-1", "verity", submitter="circuits", cpus=8, max_s=None, kind="session", node=nebius2.NODE)
    allocs = {a.id: a for a in (
        model.Allocation(id=f"gl-43527@{T - 120}", job=fill, node=nebius2.NODE, gpus=(1,), cpus=frozenset(), memory_gib=193,
                         started=T - 120),
        model.Allocation(id=f"gl-16021@{T - 600}", job=sess, node=nebius2.NODE, gpus=(2,), cpus=frozenset(), memory_gib=1,
                         started=T - 600, state="frozen", frozen_by="w", frozen_at=T - 60))}
    queue = {waiter.id: ledger.Queued(waiter, nebius2.NODE, "wait.17908.57723", T - 30),
             cpu.id: ledger.Queued(cpu, nebius2.NODE, None, T - 5)}
    st = ledger.State(allocs, queue, seq=479, head="h", t=T - 2)
    return {"node": nebius2.NODE, "t": T, "df": DF.replace("70%", "41%"), "state": ledger.state_to_dict(st),
            "state_path": "/workspace/pouw/infra/cluster/live/20260930T2320Z/state.json"}


def test_node2_rows_are_the_ledgers_leases_and_waiters_by_submitter_and_run():
    rep = status.node2(node2_snapshot(), T)
    rows = {r["name"]: r for r in rep["jobs"]}
    lease = rows[f"gl-43527@{T - 120}"]
    assert (lease["owner"], lease["key"], lease["state"], lease["gpus"], lease["cpus"], lease["for_s"], lease["cls"]) == \
        ("fill:bc-7442ca43", "gl-43527", "running", 1, 0, 120, "fill")
    sess = rows[f"gl-16021@{T - 600}"]
    assert (sess["key"], sess["state"], sess["length_s"], sess["length_is"]) == ("r20260930-235745-a3d0", "frozen", 1800, "expected")
    wait = rows["wait.17908.57723"]
    assert (wait["state"], wait["gpus"], wait["for_s"], wait["length_s"]) == ("queued", 2, 30, 900)
    assert rep["ready"] == {"jobs": 2, "gpus": 2, "gpu_jobs": 1, "known": 1, "gpu_h": 0.5}
    assert rep["holds"] == {"queues": [], "workloads": []} and "bundles" not in rep and rep["disk"]["use_pct"] == 41
    assert rep["ledger"] == {"path": "/workspace/pouw/infra/cluster/live/20260930T2320Z/state.json", "seq": 479, "t": T - 2,
                             "age_s": 2, "window": False, "stale": False}
    assert status.node2(node2_snapshot(), T + status.STALE_S + 10)["ledger"]["stale"]
    assert [r["owner"] for r in status.node2(node2_snapshot(), T, owner="circuits")["jobs"]] == ["circuits"]


def test_node2_state_held_back_by_a_timed_window_is_stale_only_after_the_window_should_have_ended():
    timed = job("gl-314342", "pous", submitter="bc-dd22acf8", priority="timed", quiet=True, gpus=8, cpus=0, max_s=1200,
                node=nebius2.NODE, run="r20260930-221231-3dd1")
    a = model.Allocation(id=f"gl-314342@{T}", job=timed, node=nebius2.NODE, gpus=tuple(range(8)), cpus=frozenset(),
                         memory_gib=1544, started=T)
    snap = {**node2_snapshot(), "state": ledger.state_to_dict(ledger.State({a.id: a}, {}, seq=522, head="h", t=T))}
    during = status.node2(snap, T + 900)["ledger"]
    assert (during["window"], during["stale"], during["age_s"]) == (True, False, 900)
    assert status.node2(snap, T + 1200 + status.STALE_S + 1)["ledger"]["stale"]
    assert "seq 522, 15m old (a timed window: the agent writes no state until it ends)" in status.render([status.node2(snap, T + 900)])


def test_render_prints_each_nodes_header_then_its_jobs_and_held_workloads_by_owner():
    text = status.render([status.node1(node1_snapshot(), T), status.node2(node2_snapshot(), T),
                          {"node": "vy-nebius-3", "t": T, "error": "ReadError: no route"}])
    assert "vy-nebius-1 at 2026-10-01T00:30:00Z\n  disk: /workspace 70% used (df)\n" in text
    assert "  unreplayed Commit bundles: 13.6 GB in 2 (release.py's rule: jobs/cov, jobs/probe-jit)\n" in text
    assert "  ready: 2 queued, 3 GPUs asked; at least 4.00 GPU-h (1 of 2 GPU jobs declare a length)\n" in text
    assert ("  holds: ClusterQueue deployments-gpu stopPolicy=Hold, LocalQueue circuits stopPolicy=HoldAndDrain; "
            "2 Workloads deactivated (spec.active=false), 1 with a verity.dev/owner label\n") in text
    lines = text.splitlines()
    n1 = lines[lines.index("  holds: ClusterQueue deployments-gpu stopPolicy=Hold, LocalQueue circuits stopPolicy=HoldAndDrain; "
                           "2 Workloads deactivated (spec.active=false), 1 with a verity.dev/owner label") + 1:]
    assert n1[0].split() == ["STATE", "OWNER", "KEY", "PHASE", "MODEL", "B", "GPU", "CPU", "FOR"]
    assert n1[1].split() == ["admitted", "proofs-bf16-hill", "bf16-hill-k8192-s0", "prover-b", "-", "-", "1", "16", "5m"]
    assert n1[4].split() == ["running", "vllm-epoch-run", "cov-g176", "gpu", "qwen3-4b", "8", "1", "4", "8m"]
    assert n1[6].split() == ["held", "circuits", "n2-build/cov-g055", "gpu", "llama32-1b", "32", "1", "4", "1h57m"]
    assert "  ledger state: seq 479, 2s old (/workspace/pouw/infra/cluster/live/20260930T2320Z/state.json)\n" in text
    assert "  ready: 2 queued, 2 GPUs asked; 0.50 GPU-h\n  holds: none (no Kueue on node 2)\n" in text
    n2 = lines[lines.index("  holds: none (no Kueue on node 2)") + 1:]
    assert n2[0].split() == ["STATE", "OWNER", "KEY", "GPU", "CPU", "FOR"]
    assert "vy-nebius-3: could not read: ReadError: no route" in text


def test_cli_status_reads_every_node_with_read_only_commands_and_prints_json(capsys, monkeypatch):
    sent: list[tuple[str, str]] = []
    out = {nebius1.NODE: {"kueue": json.dumps({"items": node1_snapshot()["k8s"]}), "df": DF, "bundles": DU},
           nebius2.NODE: {"state": "/live/state.json\n" + json.dumps(node2_snapshot()["state"]), "df": DF}}

    def runner(node):
        def run(cmd):
            sent.append((node, cmd))
            return out[node][next(k for k, v in status.COMMANDS.items() if v == cmd)]
        return run

    monkeypatch.setattr(status, "clock", lambda: T)
    monkeypatch.setattr(status, "main", lambda nodes, owner, as_json, runner_=runner, main=status.main:
                        main(nodes, owner, as_json, runner=runner_))
    assert cli.main(["status", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "cluster/status/v1" and doc["t"] == T
    assert [n["node"] for n in doc["nodes"]] == [nebius1.NODE, nebius2.NODE]
    assert doc["nodes"][0]["bundles"]["count"] == 2 and doc["nodes"][1]["ledger"]["path"] == "/live/state.json"
    assert sorted(sent) == sorted([(nebius1.NODE, status.COMMANDS[k]) for k in ("kueue", "df", "bundles")]
                                  + [(nebius2.NODE, status.COMMANDS[k]) for k in ("state", "df")])
    writes = re.compile(r"\b(patch|delete|apply|create|edit|label|annotate|scale|cordon|drain|rm|mv|kill|tee|chmod|chown)\b|>(?!/dev/null)")
    for _, cmd in sent:
        assert not writes.search(cmd), cmd


def test_cli_status_reports_a_node_it_cannot_read_and_exits_1(capsys, monkeypatch):
    def runner(node):
        def run(cmd):
            raise status.ReadError(f"{node}: ssh: connect to host: Connection timed out")
        return run

    monkeypatch.setattr(status, "clock", lambda: T)
    monkeypatch.setattr(status, "main", lambda nodes, owner, as_json, main=status.main: main(nodes, owner, as_json, runner=runner))
    assert cli.main(["status", "--node", "nebius-2"]) == 1
    assert capsys.readouterr().out == "vy-nebius-2: could not read: ReadError: vy-nebius-2: ssh: connect to host: Connection timed out\n"
