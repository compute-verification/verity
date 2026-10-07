"""Node 1's dispatcher spills a GPU Job that Kueue has held 2 min to node 2's fill queue when node 2's runner would start it now
(proofs' NVF4 K=16384 step 1 waited behind eight held node-1 GPUs at 07:44Z, 1 Oct, while node 2's GPUs 0-3 sat empty). The
spill deactivates the Kueue workload, a node-2 exit 0 ends the task on node 1, and a failure or a second stop sends it back.
The node-2 probe, the writer and the generated fill script run here for real, against a node-2 tree in tmp_path."""
from __future__ import annotations

import calendar
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from research.pods import registry

pytest.importorskip("yaml")
NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
KEY = "proofs-flock-fp/fp-hill-nvf4-k16384-step1-cde1c7a"
STATUS = "2026-10-01T07:57:36Z free GPUs 1/8, kept free [7], waiters -, timed False, window waiting False, backfill False"
T0 = calendar.timegm(time.strptime("2026-10-01T08:00:00Z", "%Y-%m-%dT%H:%M:%SZ"))


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("VY_DISPATCH_ROOT", str(tmp_path / "dispatch"))
    monkeypatch.setenv("VY_READY", str(tmp_path / "ready"))
    monkeypatch.setenv("VY_DISPATCH_DEPTH", "provers=2")
    monkeypatch.delenv("PACK_SPOOL", raising=False)
    monkeypatch.delenv("PACK_COMMITS", raising=False)
    monkeypatch.delenv("VY_N2_SPILL_FLAG", raising=False)
    spec = importlib.util.spec_from_file_location("dispatch", NEBIUS / "dispatch.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "dispatch", mod)
    spec.loader.exec_module(mod)
    n2 = tmp_path / "n2"
    for d in ("fill/queue", "fill/running", "fill/done", "fill/failed", "leases", "spill"):
        (n2 / d).mkdir(parents=True)
    (n2 / "fill/keep-free").write_text("7 # memory accounting's until 17:00Z\n")
    (n2 / "fill/windows").write_text("# start UTC, minutes\n")
    (n2 / "fill/status.txt").write_text(STATUS + "\n  running x.sh\n")
    for i in range(8):
        (n2 / f"leases/{i}.lock").touch()
    mod.N2_FILL, mod.N2_LEASES, mod.N2_SPILL = str(n2 / "fill"), str(n2 / "leases"), str(n2 / "spill")
    mod.n2_calls = []

    def n2_python(code, *args, stdin=None):
        mod.n2_calls.append("probe" if code == mod.N2_PROBE else "write")
        if getattr(mod, "n2_fail", None) == mod.n2_calls[-1]:
            raise RuntimeError("ssh vy-n2: rc 255: Connection refused")
        r = subprocess.run([sys.executable, "-c", code, *args], input=stdin, capture_output=True, text=True, check=False)
        if r.returncode:
            raise RuntimeError(r.stderr.strip())
        return r.stdout

    monkeypatch.setattr(mod, "n2_python", n2_python)
    monkeypatch.setattr(mod, "held", lambda t=None: False)
    mod.jobs, mod.wls, mod.calls, mod.patch_fails = [], [], [], False

    def kubectl(*args, stdin=None):
        mod.calls.append(args)
        if args[:2] == ("get", "jobs"):
            return json.dumps({"items": mod.jobs})
        if args[:2] == ("get", "pods"):
            return json.dumps({"items": []})
        if args[:2] == ("get", "workloads"):
            return json.dumps({"items": mod.wls})
        if args[:2] == ("get", "workload"):
            return json.dumps(next(w for w in mod.wls if w["metadata"]["name"] == args[2]))
        if args[0] == "patch" and mod.patch_fails:
            raise RuntimeError("kubectl patch workload: Operation cannot be fulfilled: the object has been modified")
        return ""

    monkeypatch.setattr(mod, "kubectl", kubectl)
    mod.n2 = n2
    return mod


def _item(tree: Path, **n2) -> dict:
    return {"id": "nvf4", "template": "prover-bench", "tree": str(tree), "queue": "provers",
            "env": {"CMD": "bash verity/ml/flock/pod/74-gemm-hill.sh DTYPE=nvf4 K=16384 CPUS=16", "QUESTION": "q"},
            "resources": {"prover-bench": {"cpus": 16, "memory": 96, "gpus": 1}},
            "n2": {"cmd": "true", "owner": "bc-8416bc72", "max_min": 25, "cpus": 16, "mem_gb": 96, **n2}}


def _held(dispatch, item: dict, key: str = KEY, created: str = "2000-01-01T00:00:00Z") -> tuple[dict, dict]:
    """A suspended Job of `item` and its unadmitted workload, added to what kubectl lists."""
    job = {**dispatch.render(dispatch.pin(item), key, 0, 0), "status": {}}
    wl = {"metadata": {"name": f"job-{job['metadata']['name']}-1a2b3", "creationTimestamp": created, "resourceVersion": "17",
                       "ownerReferences": [{"kind": "Job", "name": job["metadata"]["name"]}]},
          "spec": {"queueName": "provers", "active": True},
          "status": {"conditions": [{"type": "QuotaReserved", "status": "False", "reason": "Pending"}]}}
    dispatch.jobs.append(job)
    dispatch.wls.append(wl)
    return job, wl


def _log(dispatch, ev: str) -> list[dict]:
    p = dispatch.ROOT / "log.jsonl"
    return [r for r in map(json.loads, p.read_text().splitlines()) if r.get("ev") == ev] if p.exists() else []


def _patches(dispatch) -> list[tuple[str, dict]]:
    return [(a[2], json.loads(a[-1])) for a in dispatch.calls if a[0] == "patch"]


def _run_fill(dispatch, name: str, env: dict | None = None) -> int:
    """Node 2's fill runner starting the queued script once."""
    p = dispatch.n2 / "fill/queue" / f"{name}.sh"
    return subprocess.run(["bash", str(p)], env={**os.environ, "FILL_JOB": f"{name}.sh", **(env or {})}, capture_output=True,
                          text=True, check=False).returncode


def test_an_n2_block_needs_cmd_owner_and_a_max_min_within_the_runners_cap(dispatch, tmp_path):
    b = dispatch.n2_block(_item(tmp_path))
    assert (b["gpus"], b["prio"], b["max_min"], b["cpus"]) == (1, 0, 25, 16)
    for bad, why in (({"cmd": ""}, "no n2 block"), ({"max_min": 45}, "max_min 45"), ({"gpus": 2}, "gpus 2"),
                     ({"pool": "circuits"}, "pool"), ({"owner": "a b"}, "owner")):
        with pytest.raises(ValueError, match=why):
            dispatch.n2_block(_item(tmp_path, **bad))
    with pytest.raises(ValueError):
        dispatch.n2_block({k: v for k, v in _item(tmp_path).items() if k != "n2"})


def test_the_probe_reads_node_2s_locks_keep_free_queue_and_spill_results(dispatch):
    (dispatch.n2 / "fill/queue/pn2h-a-q0.sh").write_text("#!/usr/bin/env bash\n# fill: owner=bc-84 gpus=1 project=verity on=4-7\n")
    (dispatch.n2 / "fill/queue/.half.sh.tmp").write_text("# fill: owner=x gpus=1\n")
    (dispatch.n2 / "fill/running/n2spill-a.sh").touch()
    (dispatch.n2 / "spill/n2spill-b").mkdir()
    (dispatch.n2 / "spill/n2spill-b/result.json").write_text('{"rc": 0}\n')
    with open(dispatch.n2 / "leases/3.lock") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        probe = json.loads(dispatch.n2_python(dispatch.N2_PROBE, dispatch.N2_FILL, dispatch.N2_LEASES, dispatch.N2_SPILL,
                                              "n2spill-a", "n2spill-b", "n2spill-c"))
    assert probe["gpus"] == list(range(8)) and probe["busy"] == [3] and probe["status"] == STATUS
    assert dispatch.gpu_list(probe["keep_free"]) == {7} and probe["queued"] == [["pn2h-a-q0.sh", "owner=bc-84 gpus=1 project=verity on=4-7"]]
    assert probe["spills"] == {"n2spill-a": {"where": "running", "result": None}, "n2spill-b": {"where": None, "result": '{"rc": 0}\n'},
                               "n2spill-c": {"where": None, "result": None}}


def _probe(busy=(), keep="7", queued=(), windows="", status=STATUS) -> dict:
    return {"gpus": list(range(8)), "busy": list(busy), "keep_free": keep, "queued": [list(q) for q in queued], "windows": windows,
            "status": status}


@pytest.mark.parametrize("probe, n2, idle, why", [
    (_probe(busy=[4, 5, 6]), {}, [0, 1, 2, 3], None),
    (_probe(busy=range(7)), {}, [], "1 GPU free"),                               # only the kept-free GPU 7
    (_probe(busy=[0, 1, 2, 3, 4, 5]), {}, [6], None),                            # 6 and 7 free, 7 kept free
    (_probe(busy=[0, 1, 2, 3, 4]), {"on": "0-3"}, [], "go to 0 queued"),
    (_probe(busy=[4, 5, 6], queued=[("verity-commit-x.sh", "owner=c gpus=2 project=verity")]), {}, [0, 1, 2, 3], None),
    (_probe(busy=[4, 5, 6], queued=[("a.sh", "owner=p gpus=1"), ("b.sh", "owner=p gpus=1"), ("c.sh", "owner=p gpus=1"),
                                     ("d.sh", "owner=p gpus=1")]), {}, [], "4 queued GPU job(s) ahead"),
    (_probe(busy=[4, 5, 6], queued=[("pn2h.sh", f"owner=p gpus=1 project=verity prio={p}") for p in (0, 0, 0, 0)]), {"prio": 5},
     [0, 1, 2, 3], None),
    (_probe(busy=[4, 5, 6], queued=[("cpu.sh", "owner=p gpus=0 project=verity")] * 9), {}, [0, 1, 2, 3], None),
    (_probe(busy=[4, 5, 6], windows="2026-10-01T08:20Z 30 # Pearl\n"), {}, [], "window at 08:20Z"),
    (_probe(busy=[4, 5, 6], windows="2026-10-01T08:28Z 30\n2026-10-01T07:00Z 30\n"), {"max_min": 25}, [0, 1, 2, 3], None),
    (_probe(busy=[4, 5, 6], status=STATUS.replace("timed False", "timed True")), {}, [], "timed True"),
])
def test_node_2_has_room_only_where_its_runner_would_start_the_job_now(dispatch, tmp_path, probe, n2, idle, why):
    got, reason = dispatch.n2_room(probe, dispatch.n2_block(_item(tmp_path, **n2)), T0)
    if why is None:
        assert (got, reason) == (idle, None)
    else:
        assert got == [] and why in reason, reason


def test_with_the_flag_off_or_no_held_job_nothing_reaches_node_2(dispatch, tmp_path):
    _held(dispatch, _item(tmp_path))
    dispatch.tick()
    assert not dispatch.n2_calls and not _patches(dispatch)
    dispatch.SPILL_FLAG.parent.mkdir(parents=True, exist_ok=True)
    dispatch.SPILL_FLAG.touch()
    job, wl = dispatch.jobs[0], dispatch.wls[0]
    t = dispatch.ts(wl["metadata"]["creationTimestamp"]) + 119
    assert dispatch.n2_spill(dispatch.jobs, dispatch.wls, t=t) == set() and not dispatch.n2_calls
    dispatch.jobs[0] = {**job, "status": {"startTime": "2026-10-01T07:40:00Z"}}       # preempted after it ran: never spilled
    dispatch.tick()
    dispatch.jobs[0] = {**job, "metadata": {**job["metadata"], "annotations": {
        **job["metadata"]["annotations"], "verity.dev/item-json": json.dumps({k: v for k, v in _item(tmp_path).items() if k != "n2"})}}}
    dispatch.tick()
    assert not dispatch.n2_calls and not _patches(dispatch)


def test_one_held_job_spills_per_tick_and_its_queue_tops_up_behind_it(dispatch, tmp_path):
    dispatch.SPILL_FLAG.parent.mkdir(parents=True, exist_ok=True)
    dispatch.SPILL_FLAG.touch()
    job, wl = _held(dispatch, _item(tmp_path, pool="provers", cpuset="160-175", prio=5))
    _held(dispatch, _item(tmp_path), key=KEY + "-rerun", created="2000-01-01T00:01:00Z")
    (tmp_path / "ready/proofs-arch").mkdir(parents=True)
    (tmp_path / "ready/proofs-arch/next.json").write_text(json.dumps({k: v for k, v in _item(tmp_path).items() if k not in ("n2", "id")}))
    dispatch.tick()
    name = f"n2spill-{job['metadata']['name']}"
    assert _patches(dispatch) == [(wl["metadata"]["name"], {"metadata": {"resourceVersion": "17"}, "spec": {"active": False}})]
    assert sorted(p.name for p in (dispatch.n2 / "fill/queue").iterdir()) == [f"{name}.sh"]
    script = (dispatch.n2 / "fill/queue" / f"{name}.sh").read_text()
    assert "# fill: owner=bc-8416bc72 gpus=1 max_min=25 cpus=16 project=verity prio=5 mem_gb=96\n" in script
    assert "/usr/local/bin/vy-provers bash -c true" in script and "VY_PROVERS_CPUS=160-175" in script
    assert json.loads((dispatch.n2 / "spill" / name / "item.json").read_text())["env"]["CMD"].endswith("CPUS=16")
    (ev,) = _log(dispatch, "n2-spill")
    assert (ev["job"], ev["key"], ev["queue"], ev["n2"], ev["idle"]) == (job["metadata"]["name"], KEY, "provers", name, [0, 1, 2, 3, 4, 5, 6])
    assert json.loads((dispatch.SPILL / f"{job['metadata']['name']}.json").read_text())["workload"] == wl["metadata"]["name"]
    # the spilled workload no longer counts against provers' depth of 2, so the ready item went out beside the other held one
    assert [a[0] for a in dispatch.calls if a[0] == "create"] == ["create"]


def _spilled(dispatch, tmp_path, **n2) -> tuple[dict, dict, str]:
    dispatch.SPILL_FLAG.parent.mkdir(parents=True, exist_ok=True)
    dispatch.SPILL_FLAG.touch()
    job, wl = _held(dispatch, _item(tmp_path, **n2))
    dispatch.tick()
    dispatch.calls.clear()
    return job, wl, f"n2spill-{job['metadata']['name']}"


def _done(dispatch) -> list[dict]:
    p = dispatch.ROOT / "done.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def test_a_spilled_task_that_exits_0_on_node_2_ends_on_node_1(dispatch, tmp_path):
    job, wl, name = _spilled(dispatch, tmp_path, cmd=f"echo $N2_SPILL_KEY > {tmp_path}/ran")
    assert _run_fill(dispatch, name) == 0 and (tmp_path / "ran").read_text().strip() == KEY
    dispatch.tick()
    assert ("delete", "job", job["metadata"]["name"], "--ignore-not-found", "--wait=false") in dispatch.calls
    assert not _patches(dispatch)
    assert _done(dispatch)[-1] == {**_done(dispatch)[-1], "key": KEY, "state": "succeeded", "rc": 0, "n2": name}
    (end,) = _log(dispatch, "end")
    assert (end["n2"], end["fill_job"], end["tries"]) == (name, f"{name}.sh", 1)
    assert not list(dispatch.SPILL.glob("*.json")) and (dispatch.SPILL / "done" / f"{job['metadata']['name']}.json").exists()
    assert _run_fill(dispatch, name) == 0 and len((tmp_path / "ran").read_text().split()) == 1    # a rerun only repeats its exit


@pytest.mark.parametrize("runs, why", [
    ([3], "node 2 rc 3 after 1 start(s)"),
    ([143, 143, 1], "node 2 rc None after 3 start(s) (stopped twice)"),
])
def test_a_failed_or_twice_stopped_run_goes_back_to_node_1_once(dispatch, tmp_path, runs, why):
    count = tmp_path / "count"
    job, wl, name = _spilled(dispatch, tmp_path, cmd=f'echo x >> {count}; exit "$(cat {tmp_path}/rc)"')
    for rc in runs:
        (tmp_path / "rc").write_text(str(rc))
        assert _run_fill(dispatch, name) == rc
        if rc == 143:
            dispatch.tick()
            assert not _patches(dispatch)                       # a stop requeues on node 2; node 1 waits
    assert len(count.read_text().split()) == min(len(runs), 2)  # the third start runs nothing
    dispatch.tick()
    assert _patches(dispatch) == [(wl["metadata"]["name"], {"spec": {"active": True}})]
    (back,) = _log(dispatch, "n2-spill-back")
    assert back["why"] == why and back["outcome"] == "back" and not _done(dispatch)
    dispatch.calls.clear()
    dispatch.tick()
    assert not _patches(dispatch) and len(_log(dispatch, "n2-spill")) == 1


def test_a_fill_job_gone_from_node_2_without_a_result_comes_back_on_the_second_tick(dispatch, tmp_path):
    job, wl, name = _spilled(dispatch, tmp_path)
    (dispatch.n2 / "fill/queue" / f"{name}.sh").rename(dispatch.n2 / "fill/failed" / f"{name}.sh")
    dispatch.tick()
    assert not _patches(dispatch)
    dispatch.tick()
    assert _patches(dispatch) == [(wl["metadata"]["name"], {"spec": {"active": True}})]
    assert "its fill job is in failed" in _log(dispatch, "n2-spill-back")[0]["why"]


def test_node_2_unreachable_or_a_changed_workload_leaves_the_job_on_node_1(dispatch, tmp_path):
    dispatch.SPILL_FLAG.parent.mkdir(parents=True, exist_ok=True)
    dispatch.SPILL_FLAG.touch()
    job, wl = _held(dispatch, _item(tmp_path))
    dispatch.n2_fail = "write"
    dispatch.tick()
    assert [p[1]["spec"] for p in _patches(dispatch)] == [{"active": False}, {"active": True}]
    assert "node 2 took no fill job" in _log(dispatch, "n2-spill-back")[0]["why"] and not list(dispatch.SPILL.glob("*.json"))
    job2, wl2 = _held(dispatch, _item(tmp_path), key=KEY + "-b")
    dispatch.n2_fail, dispatch.patch_fails = None, True
    dispatch.calls.clear()
    dispatch.tick()
    assert not list(dispatch.SPILL.glob("*.json")) and not list((dispatch.n2 / "fill/queue").iterdir()) and not _log(dispatch, "n2-spill")
    dispatch.n2_fail = "probe"
    dispatch.patch_fails = False
    dispatch.calls.clear()
    dispatch.tick()
    assert not _patches(dispatch)


def test_pool_provers_runs_the_command_through_vy_provers_with_the_item(dispatch, tmp_path):
    fake = tmp_path / "vy-provers"
    fake.write_text(f'#!/usr/bin/env bash\necho "$VY_PROVERS_CPUS $*" > {tmp_path}/argv\nexec "$@"\n')
    fake.chmod(0o755)
    dispatch.VY_PROVERS = str(fake)
    job, wl, name = _spilled(dispatch, tmp_path, pool="provers", cpuset="144-159", cmd=f"cp $N2_SPILL_ITEM {tmp_path}/item.json")
    assert _run_fill(dispatch, name) == 0
    assert (tmp_path / "argv").read_text().split() == ["144-159", "bash", "-c", "cp", "$N2_SPILL_ITEM", f"{tmp_path}/item.json"]
    assert json.loads((tmp_path / "item.json").read_text())["n2"]["pool"] == "provers"
