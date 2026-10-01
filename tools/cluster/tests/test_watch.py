"""`cluster watch`: which jobs made no progress for the window and whose they are, from the snapshots `cluster status` reads
and the progress a node's probe reports (files, the workload's CPU, GPUs), and the probe itself run here on a made-up node."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import pytest
from _build import job
from test_status import BF16, FLOCK, G176, T, node1_snapshot, pod

from cluster import cli, ledger, model, nebius1, nebius2, status, watch

W = 300
HANDLES = [{"handle": "infra", "usergroup": "S0INFRA", "lanes": ["infra", "skypilot"], "agents": ["bc-17cc41f1"]},
           {"handle": "circuits", "usergroup": "S0CIRC", "lanes": ["circuits", "circuits-*", "vllm-*"], "agents": ["bc-698052e1"]},
           {"handle": "proofs", "usergroup": "S0PROOF", "lanes": ["proofs-*", "proofs-flock-*"]},
           {"handle": "compute-accounting", "usergroup": "S0PUW", "agents": ["bc-2aa33ad8"]}]
G176_POD = f"{G176}-2fzth"


def test_an_owner_routes_by_its_parts_last_first_to_an_agent_prefix_or_the_longest_lane_pattern():
    assert watch.route("vllm-epoch-run", HANDLES) == ("circuits", "vllm-epoch-run")
    assert watch.route("circuits", HANDLES) == ("circuits", "circuits")
    assert watch.route("proofs-flock-fp", HANDLES) == ("proofs", "proofs-flock-fp")
    assert watch.route("fill:bc-2aa33ad8", HANDLES) == ("compute-accounting", "compute-accounting")
    assert watch.route("verity:bc-698052e1-cov-g217-proof", HANDLES) == ("circuits", "circuits")
    assert watch.route("lean-audit:circuits-w", HANDLES) == ("circuits", "circuits-w")
    assert watch.route("circuits-kind:nobody", HANDLES) == ("circuits", "circuits-kind"), "a later part nobody claims falls through"
    for nobody in ("-", "research", "skypilotx", "bc-99999999"):
        assert watch.route(nobody, HANDLES) == ("infra", "infra")
    assert watch.usergroup("circuits", HANDLES) == "S0CIRC" and watch.usergroup("gone", HANDLES) == ""


def test_the_repositorys_registry_routes_the_owners_the_nodes_run_today():
    handles = watch.load_registry(watch.REGISTRY)
    for owner, want in (("vllm-epoch-run", "circuits"), ("n2-build", "circuits"), ("proofs-flock-fp", "proofs"),
                        ("proofs-verify-overlap", "proofs"), ("backend-sweep", "proofs"), ("fill:bc-2aa33ad8", "compute-accounting"),
                        ("fill:bc-dbc19788-573d-5ba4-b3b2-d6551c1c60ef", "compute-accounting"), ("bc-7442ca43", "compute-accounting"),
                        ("verity:bc-698052e1-cov-g217-proof", "circuits"), ("lean-audit:proofs-w", "proofs"), ("skypilot", "infra"),
                        ("-", "infra"), ("research", "infra")):
        assert watch.route(owner, handles)[0] == want, owner
    assert all(re.fullmatch(r"S[0-9A-Z]+", watch.usergroup(h["handle"], handles)) for h in handles if h.get("lanes"))


def node1() -> dict:
    snap = node1_snapshot()
    for it in snap["k8s"]:
        if it["kind"] == "Job" and it["metadata"]["name"] == FLOCK:
            it["metadata"]["labels"]["verity.dev/goal"] = "cov-g217"
    snap["k8s"].append(pod("sky-gpu-1-head", None, "2026-10-01T00:00:00Z", gpus=1, labels={"skypilot-cluster": "gpu-1"}))
    return snap


def progress(**kw) -> dict:
    run = {"dir": "/workspace/jobs/runs/r1", "newest": T - 400, "complete": True, "cpu": {"cpu_s": 0.5, "over_s": W}}
    run.update(kw.pop("run", {}))
    return {"t": T, "runs": {"r1": run}, "pods": {G176_POD: ["r1"]}, "gpu": {G176_POD: 0.0, "sky-gpu-1-head": 0.0},
            "gpu_error": None, **kw}


def test_node1s_candidates_are_long_running_pods_admitted_workloads_and_waiting_goal_jobs_not_held_or_controllers():
    c = watch.candidates(node1(), T, W)
    got = {j["job"]: (j["state"], j["pods"], j["goal"]) for j in c["jobs"]}
    assert got == {"vllm-epoch-run/cov-g176": ("running", [G176_POD], None),
                   "proofs-bf16-hill/bf16-hill-k8192-s0": ("admitted", [], None),
                   "proofs-flock-fp/fp-hill-nvf4-k2048": ("queued", [], "cov-g217"),
                   "sky-gpu-1-head": ("running", ["sky-gpu-1-head"], None)}
    assert not c["window"] and not c["stale"]
    a = watch.probe_args(c, W)
    assert a["pods"] == sorted([(G176_POD, ledger.epoch("2026-10-01T00:21:56Z")), ("sky-gpu-1-head", ledger.epoch("2026-10-01T00:00:00Z"))])
    assert a["prom"] == nebius1.PROM and a["roots"] == list(watch.ROOTS) and "runs" not in a
    assert watch.candidates(node1(), T, 3600)["jobs"] == [], "nothing has run an hour but the controller, which isn't watched"


def test_a_job_stalls_only_when_its_files_cpu_and_gpus_all_were_idle_for_the_window():
    c = watch.candidates(node1(), T, W)
    rep = watch.judge(c, progress(), HANDLES, W)
    by = {f["job"]: f for f in rep["findings"]}
    g = by["vllm-epoch-run/cov-g176"]
    assert (g["handle"], g["usergroup"], g["lane"], g["run"], g["quiet_s"], g["gpus"]) == ("circuits", "S0CIRC", "vllm-epoch-run", "r1", 400, 1)
    assert g["why"] == "no run file changed, and its processes and GPUs were idle for 5 min"
    assert by["sky-gpu-1-head"]["why"] == "no run found for it, and its GPUs were idle for 5 min"
    assert (by["sky-gpu-1-head"]["handle"], by["sky-gpu-1-head"]["run"]) == ("infra", "-")
    assert by["proofs-bf16-hill/bf16-hill-k8192-s0"]["why"] == "admitted 5 min ago and no pod is running"
    w = by["proofs-flock-fp/fp-hill-nvf4-k2048"]
    assert (w["why"], w["goal"], w["handle"]) == ("a goal job waiting 5 min for a grant", "cov-g217", "proofs")
    assert rep["watched"] == 4 and rep["unwatched"] == []
    for moved in (progress(run={"newest": T - 100}), progress(run={"cpu": {"cpu_s": 7.0, "over_s": W}}),
                  progress(gpu={G176_POD: 40.0, "sky-gpu-1-head": 0.0})):
        assert "vllm-epoch-run/cov-g176" not in {f["job"] for f in watch.judge(c, moved, HANDLES, W)["findings"]}


def test_a_signal_that_cant_be_read_is_named_and_a_job_with_none_is_unwatched_not_stalled():
    c = watch.candidates(node1(), T, W)

    def g176(p):
        rep = watch.judge(c, p, HANDLES, W)
        return next((f["why"] for f in rep["findings"] if f["job"] == "vllm-epoch-run/cov-g176"), None), rep["unwatched"]

    assert g176(progress(gpu_error="URLError: refused"))[0] == \
        "no run file changed, and its processes were idle for 5 min; its GPUs couldn't be read"
    assert g176(progress(run={"cpu": None}))[0] == "no run file changed, and its GPUs were idle for 5 min; its CPU use couldn't be read"
    assert g176(progress(run={"cpu": None}, gpu={}))[0] == "no run file changed for 5 min; its CPU use and GPUs couldn't be read"
    why, unwatched = g176(progress(run={"complete": False}))
    assert why is None and unwatched[0]["why"] == "its run dir has too many files to scan for a change"
    why, unwatched = g176(progress(pods={}, gpu={}, gpu_error="URLError: refused"))
    assert why is None and [u["job"] for u in unwatched] == ["sky-gpu-1-head", "vllm-epoch-run/cov-g176"]
    assert unwatched[0]["why"] == "no run found for it, and no GPU reading (URLError: refused)"


def node2(*, timed: bool = False, t: int = T) -> dict:
    lease = job("gl-77", "pous", submitter="verity:bc-698052e1-cov-g217", priority="session", kind="session", gpus=1, cpus=0,
                max_s=None, expected_s=1800, node=nebius2.NODE, run="r20261001-000000-aaaa", goal="cov-g217", quiet=timed)
    fill = job("gl-88", "pous", submitter="fill:bc-2aa33ad8", priority="fill", gpus=1, cpus=0, max_s=3600, node=nebius2.NODE)
    goal = job("gl-99", "pous", submitter="bc-2aa33ad8", priority="session", gpus=1, cpus=0, max_s=600, node=nebius2.NODE,
               goal="pouw-g3")
    plain = job("gl-55", "pous", submitter="bc-2aa33ad8", priority="session", gpus=1, cpus=0, max_s=600, node=nebius2.NODE)
    allocs = {a.id: a for a in (
        model.Allocation(id=f"gl-77@{T - 3600}", job=lease, node=nebius2.NODE, gpus=(2,), cpus=frozenset(), memory_gib=1,
                         started=T - 3600),
        model.Allocation(id=f"gl-88@{T - 3600}", job=fill, node=nebius2.NODE, gpus=(3,), cpus=frozenset(), memory_gib=1,
                         started=T - 3600))}
    queue = {goal.id: ledger.Queued(goal, nebius2.NODE, "wait.1.99", T - 1900), plain.id: ledger.Queued(plain, nebius2.NODE, "wait.1.55", T - 1900)}
    st = ledger.State(allocs, queue, seq=9, head="h", t=t)
    return {"node": nebius2.NODE, "t": T, "df": "x\n/dev/vdc 1 1 1 41% /workspace\n", "state": ledger.state_to_dict(st),
            "state_path": "/live/state.json"}


def test_node2s_leases_are_watched_by_run_and_pid_and_a_goal_waiter_by_its_wait():
    c = watch.candidates(node2(), T, 1800)
    assert [(j["name"], j["pid"], j["run"], j["goal"]) for j in c["jobs"]] == [
        ("wait.1.99", None, None, "pouw-g3"), (f"gl-88@{T - 3600}", "88", None, None),
        (f"gl-77@{T - 3600}", "77", "r20261001-000000-aaaa", "cov-g217")]
    a = watch.probe_args(c, 1800)
    assert (a["runs"], a["pids"], a["util_dir"]) == (["r20261001-000000-aaaa"], ["77", "88"], watch.UTIL_DIR) and "prom" not in a
    p = {"t": T, "runs": {"r20261001-000000-aaaa": {"dir": "/workspace/research/runs/r20261001-000000-aaaa", "newest": T - 3000,
                                                     "complete": True, "cpu": {"cpu_s": 0.0, "over_s": 1800}}},
         "pods": {}, "gpu": {"77": 0.0, "88": 80.0}, "gpu_error": None}
    rep = watch.judge(c, p, HANDLES, 1800)
    assert [(f["name"], f["handle"], f["lane"], f["goal"], f["why"]) for f in rep["findings"]] == [
        ("wait.1.99", "compute-accounting", "compute-accounting", "pouw-g3", "a goal job waiting 30 min for a grant"),
        (f"gl-77@{T - 3600}", "circuits", "circuits", "cov-g217", "no run file changed, and its processes and GPUs were idle for 30 min")]


def test_node2s_leases_go_unread_through_a_timed_window_and_a_stale_ledger_is_an_error():
    c = watch.candidates(node2(timed=True), T, 1800)
    assert c["window"] and [j["state"] for j in c["jobs"]] == ["queued"] and watch.probe_args(c, 1800) is None
    sent = []

    def run(cmd):
        sent.append(cmd)
        return "/live/state.json\n" + json.dumps(node2(t=T - 600)["state"]) if cmd == status.COMMANDS["state"] else "x\n/dev/vdc 1 1 1 41% /workspace\n"

    with pytest.raises(status.ReadError, match="stale"):
        watch.watch(nebius2.NODE, run, T, HANDLES, 1800)
    assert sorted(sent) == sorted([status.COMMANDS["state"], status.COMMANDS["df"]])


def test_the_cli_watches_every_node_read_only_writes_the_exporters_file_and_prints_each_finding(tmp_path, capsys, monkeypatch):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema": "verity/slack-registry/v1", "handles": HANDLES}))
    sent: list[tuple[str, str]] = []
    canned = {nebius1.NODE: {status.COMMANDS["kueue"]: json.dumps({"items": node1()["k8s"]}), status.COMMANDS["df"]: node1()["df"]},
              nebius2.NODE: {status.COMMANDS["state"]: "/live/state.json\n" + json.dumps(node2()["state"]),
                             status.COMMANDS["df"]: node2()["df"]}}
    probes = {nebius1.NODE: progress(), nebius2.NODE: {"t": T, "runs": {}, "pods": {}, "gpu": {"77": 50.0, "88": 50.0}, "gpu_error": None}}

    def runner(node):
        def run(cmd):
            sent.append((node, cmd))
            if cmd.startswith("python3 -c "):
                return json.dumps(probes[node])
            return canned[node].get(cmd, "")
        return run

    monkeypatch.setattr(watch, "clock", lambda: T)
    monkeypatch.setattr(watch, "main", lambda nodes, as_json, main=watch.main, **kw: main(nodes, as_json, **{**kw, "runner": runner}))
    out = tmp_path / "stall-watch.json"
    assert cli.main(["watch", "--window-min", "5", "--registry", str(reg), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.splitlines()[0] == "vy-nebius-1: 4 stalled of 4 watched"
    assert ("  STALLED @circuits vllm-epoch-run/cov-g176 (owner vllm-epoch-run; run r1, 1 GPU, running 8m, quiet 6m): "
            "no run file changed, and its processes and GPUs were idle for 5 min") in text
    assert ("vy-nebius-2: 1 stalled of 3 watched\n  STALLED @compute-accounting gl-99 (owner bc-2aa33ad8; goal pouw-g3, 1 GPU, "
            "queued 31m): a goal job waiting 5 min for a grant\n") in text
    doc = json.loads(out.read_text())
    assert doc["schema"] == "cluster/watch/v1" and doc["t"] == T and doc["window_s"] == 300
    assert [len(n["findings"]) for n in doc["nodes"]] == [4, 1]
    assert not any(cmd == status.COMMANDS["bundles"] for _, cmd in sent), "the watch doesn't walk the bundles"
    writes = re.compile(r"\b(patch|delete|apply|create|edit|label|annotate|scale|cordon|drain|rm|mv|kill|tee|chmod|chown)\b|>(?!/dev/null)")
    for _, cmd in sent:
        if not cmd.startswith("python3 -c "):
            assert not writes.search(cmd), cmd
    assert not re.search(r"\.write|open\([^)]*['\"][wa]|os\.(remove|unlink|rename|replace|kill)|subprocess", watch.PROBE)


def test_an_unreadable_registry_or_node_exits_1_and_still_writes_the_file(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(watch, "clock", lambda: T)
    out = tmp_path / "w.json"
    assert watch.main([nebius1.NODE], False, registry=tmp_path / "none.json", out=out, runner=lambda n: None) == 1
    assert "registry" in json.loads(out.read_text())["error"] and capsys.readouterr().out.startswith("could not watch: registry")

    def down(node):
        def run(cmd):
            raise status.ReadError(f"{node}: ssh: connect to host: Connection timed out")
        return run

    reg = tmp_path / "r.json"
    reg.write_text(json.dumps({"handles": HANDLES}))
    assert watch.main([nebius2.NODE], False, registry=reg, out=out, runner=down) == 1
    assert json.loads(out.read_text())["nodes"] == [{"node": nebius2.NODE, "error": "ReadError: vy-nebius-2: ssh: connect to host: Connection timed out"}]
    assert capsys.readouterr().out == "vy-nebius-2: could not watch: ReadError: vy-nebius-2: ssh: connect to host: Connection timed out\n"


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def test_the_probe_reads_a_nodes_run_dirs_workload_cpu_and_sampler_and_finds_a_pods_runs(tmp_path):
    """The probe runs as written here, on a made-up node: run dirs, resources.jsonl and the sampler's day files."""
    now = T
    root = tmp_path / "jobs" / "runs"
    quiet_run, busy_run, other = root / "r20261001-000500-aaaa", root / "r20261001-001000-bbbb", root / "r20261001-001100-cccc"
    for d in (quiet_run, busy_run, other):
        d.mkdir(parents=True)
    (quiet_run / "events.jsonl").write_text(json.dumps({"kind": "started", "host": "pod-a"}) + "\n")
    (busy_run / "events.jsonl").write_text(json.dumps({"kind": "started", "host": "pod-b"}) + "\n")
    (other / "events.jsonl").write_text(json.dumps({"kind": "started", "host": "pod-z"}) + "\n")
    (quiet_run / "stdout.log").write_text("x")
    (quiet_run / "sampler.json").write_text("{}")
    with open(quiet_run / "resources.jsonl", "w") as f:
        for i in range(0, 3600, 10):
            t = now - 3600 + i
            f.write(json.dumps({"kind": "cgroup", "t_utc": _iso(t), "pad": "y" * 4000}) + "\n")
            f.write(json.dumps({"kind": "process", "t_utc": _iso(t), "processes": [
                {"pid": 9, "starttime_ticks": 1, "cpu_s": 100.0 + (i if i < 1500 else 1500) * 0.5}]}) + "\n")
    computing = root / "r20261001-000100-dddd"
    computing.mkdir()
    with open(computing / "resources.jsonl", "w") as f:
        for i in range(0, 3600, 10):
            f.write(json.dumps({"kind": "process", "t_utc": _iso(now - 3600 + i), "pad": "y" * 4000, "processes": [
                {"pid": 9, "starttime_ticks": 1, "cpu_s": i * 0.5}, *([{"pid": 12, "starttime_ticks": 5, "cpu_s": (i - 3000) * 1.0}]
                                                                      if i >= 3000 else [])]}) + "\n")
    (computing / "stdout.log").write_text("built nothing yet")
    os.utime(computing / "stdout.log", (now - 3500, now - 3500))
    (busy_run / "out").mkdir()
    (busy_run / "out" / "part-1").write_text("y")
    for path in quiet_run.iterdir():
        os.utime(path, (now - 2000, now - 2000))
    os.utime(quiet_run / "resources.jsonl", (now, now))
    os.utime(quiet_run / "sampler.json", (now, now))
    os.utime(busy_run / "events.jsonl", (now - 2000, now - 2000))
    os.utime(busy_run / "out" / "part-1", (now - 30, now - 30))
    util = tmp_path / "util"
    util.mkdir()
    day = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
    with open(util / f"{day}.jsonl", "w") as f:
        for t in range(now - 2400, now + 1, 10):
            f.write(json.dumps({"t": _iso(t)[:19] + "Z", "gpus": [
                {"i": 0, "util": 90 if t < now - 1900 else 0, "lease": "pid=77 run=r1 who=w since=x"},
                {"i": 1, "util": 3, "lease": "pid=88 run=- who=fill:w since=x"}, {"i": 2, "util": 99, "lease": ""}]}) + "\n")
    starts = {"pod-a": ledger.epoch("2026-10-01T00:05:00Z") - 30, "pod-b": ledger.epoch("2026-10-01T00:10:00Z")}
    args = {"now": now, "window_s": 1800, "quiet": watch.QUIET, "roots": [str(tmp_path / "research" / "runs"), str(root)],
            "pods": sorted(starts.items()), "runs": [computing.name], "pids": ["77", "88"], "util_dir": str(util), "tail_bytes": 1 << 20}
    r = subprocess.run([sys.executable, "-c", watch.PROBE, json.dumps(args)], capture_output=True, text=True, check=True)
    out = json.loads(r.stdout)
    assert out["t"] == now and out["gpu_error"] is None
    assert out["pods"] == {"pod-a": [quiet_run.name], "pod-b": [busy_run.name]}
    q, b = out["runs"][quiet_run.name], out["runs"][busy_run.name]
    assert (q["dir"], q["newest"], q["complete"]) == (str(quiet_run), now - 2000, True), "resources.jsonl and sampler.json don't count"
    assert q["cpu"]["over_s"] >= 1780 and q["cpu"]["cpu_s"] <= 0.5 * 10, "its CPU stopped 35 min ago"
    assert (b["newest"], b["cpu"]) == (now - 30, None)
    cpu = out["runs"][computing.name]["cpu"]
    assert out["runs"][computing.name]["newest"] == now - 3500 and abs(cpu["cpu_s"] - (0.5 * cpu["over_s"] + 590)) <= 10, \
        "its root's CPU over the window, and all of a child's that started in it"
    assert out["gpu"] == {"77": 0.0, "88": 3.0}, "lease 77's GPU went idle 31 min ago; only the window counts"
    stale = subprocess.run([sys.executable, "-c", watch.PROBE, json.dumps({**args, "now": now + 600, "pods": []})],
                           capture_output=True, text=True, check=True)
    assert json.loads(stale.stdout)["gpu_error"] == "the sampler wrote no line in the last 120 s"
