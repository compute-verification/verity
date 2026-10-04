"""Node 1's MPS Commit packing: dispatch.py routes only eligible Commits to the spool (and only under PACK_COMMITS=1), reads
commit-pack results back into the chain (a fault requeues the Commit once, unpacked), and the pilot (sky/commit_pack.py) runs
spooled Commits under a pod-local MPS daemon, stopping the pod's other Commits when one faults."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from research.pods import registry

pytest.importorskip("yaml")
NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
LLAMA_B8 = "llama32-1b__bf16__rtxpro6000__tp1__b8__i256__o32__mixed__greedy__bi-eager"


def _load(path: Path, name: str, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)   # dataclasses look their module up by name
    spec.loader.exec_module(mod)
    return mod


def _tree(root: Path, deferred: bool = True) -> Path:
    tools = root / "integrations/vllm/verity_vllm/pipeline/research_tools.py"
    tools.parent.mkdir(parents=True)
    tools.write_text("VLLM_REPLAY = 'vllm.replay'\n" if deferred else "VLLM_COMMIT = 'vllm.commit'\n")
    return root


def _item(tree: Path, row: str = LLAMA_B8, **kw) -> dict:
    return {"id": "c1", "template": "config-run", "tree": str(tree),
            "env": {"ROW": row, "ROLE": "LLAMA32_1B", "REPO": "unsloth/Llama-3.2-1B", "REVISION": "9535bd9b"}, **kw}


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("VY_DISPATCH_ROOT", str(tmp_path / "dispatch"))
    monkeypatch.setenv("VY_READY", str(tmp_path / "ready"))
    monkeypatch.delenv("PACK_SPOOL", raising=False)
    monkeypatch.delenv("PACK_COMMITS", raising=False)
    monkeypatch.setenv("PACK_DISK_PATH", str(tmp_path))
    monkeypatch.setenv("PACK_DISK_PAUSE", "1.01")
    monkeypatch.setenv("PACK_ALERT_DIR", str(tmp_path / "alerts"))
    mod = _load(NEBIUS / "dispatch.py", "dispatch", monkeypatch)
    mod.calls = []

    def kubectl(*args, stdin=None):
        mod.calls.append((args, json.loads(stdin) if stdin else None))
        return "{}"

    monkeypatch.setattr(mod, "kubectl", kubectl)
    monkeypatch.setattr(mod, "held", lambda t=None: False)
    monkeypatch.setattr(mod, "draining", lambda t=None: False)
    return mod


def test_commit_pack_renders_one_gpu_workload_in_deployments_gpu(dispatch):
    job = dispatch.render({"id": "p", "template": "commit-pack", "tree": str(NEBIUS), "env": {}}, "commit-pack/p1", 0, 0)
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "deployments-gpu"
    pod = job["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert c["resources"]["limits"] == {"nvidia.com/gpu": "1"} and c["resources"]["requests"]["cpu"] == "4"
    assert pod["runtimeClassName"] == "nvidia"
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["SRC"] == str(NEBIUS) and env["PACK_MEM_LIMIT"] == "0=28G" and env["PACK_GPU_UTIL"] == "0.20"
    assert "sky/commit_pack.py" in c["command"][-1] and (NEBIUS / "sky" / "commit_pack.py").exists()
    # vLLM's budget is GPU_UTIL of the RTX PRO 6000's 94.97 GiB; the Commit stages ~6 GiB beside it (at 0.28 its 2.3 GiB window
    # ring found 946 MiB) and the client's CUDA context takes ~0.5 GiB, all inside the MPS limit
    limit_gib = float(env["PACK_MEM_LIMIT"].split("=")[1].rstrip("G"))
    assert float(env["PACK_GPU_UTIL"]) * 94.97 + 6 + 0.5 <= limit_gib - 1
    assert int(env["PACK_SLOTS"]) * limit_gib <= 94.97


def test_a_class_sizes_each_of_config_runs_three_tasks(dispatch, tmp_path):
    """A `class` item on config-run's three tasks (build, gpu, replay) used to raise in render, which failed every dispatcher tick."""
    item = _item(_tree(tmp_path / "t"), "llama32-1b__bf16__rtxpro6000__tp2__b1__i256__o32__mixed__greedy__bi-eager", **{"class": "tp2"})
    build, gpu, n = dispatch.classes()["tp2"]
    want = [(None, build, 0), (4 * n, gpu, n), (8, 64, 0)]
    for idx, (cpus, mem, gpus) in enumerate(want):
        c = dispatch.render(item, "lane/tp2", idx, 0)["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
        assert c["memory"] == f"{mem}G" and int(c.get("nvidia.com/gpu", 0)) == gpus, (idx, c)
        assert cpus is None or c["cpu"] == str(cpus), (idx, c)


@pytest.mark.parametrize("row, why", [
    (LLAMA_B8, None),
    ("smollm2-135m__bf16__rtxpro6000__tp1__b1__i1024__o128__mixed__greedy__bi-eager", None),
    ("llama32-3b__bf16__rtxpro6000__tp1__b8__i256__o32__mixed__greedy__bi-eager", None),
    ("qwen25-05b-instruct__bf16__rtxpro6000__tp1__b1__i1024__o128__mixed__greedy__bi-eager", None),
    ("r1-distill-qwen-15b__bf16__rtxpro6000__tp1__b8__i256__o32__mixed__greedy__bi-eager", None),
    ("llama31-8b__bf16__rtxpro6000__tp1__b8__i256__o32__mixed__greedy__bi-eager", "packing list"),
    ("llama32-1b__bf16__rtxpro6000__tp1__b16__i256__o32__mixed__greedy__bi-eager", "batch 16"),
    ("llama32-1b__bf16__rtxpro6000__tp2__b8__i256__o32__mixed__greedy__bi-eager", "not TP1"),
    ("gemma2-2b__bf16__rtxpro6000__tp1__b8__i256__o32__mixed__greedy__bi-eager", "packing list"),
    ("pythia-160m__bf16__rtxpro6000__tp1__b8__i256__o32__mixed__greedy__bi-eager", "packing list"),
    ("mistral-7b__bf16__rtxpro6000__tp1__b8__i256__o32__mixed__greedy__bi-eager", "packing list"),
])
def test_only_small_tp1_commits_pack(dispatch, tmp_path, row, why):
    got = dispatch.packable(_item(_tree(tmp_path / "t"), row), "lane/c1", 1)
    assert got is None if why is None else why in got


def test_the_build_an_unpacked_or_leased_item_and_a_tree_without_deferred_replay_never_pack(dispatch, tmp_path):
    tree = _tree(tmp_path / "t")
    assert "gpu task" in dispatch.packable(_item(tree), "lane/c1", 0)
    assert "unpacked" in dispatch.packable(_item(tree, unpacked="p: fault"), "lane/c1", 1)
    assert "leased" in dispatch.packable(_item(tree, lease="wrap"), "lane/c1", 1)
    assert "deferred" in dispatch.packable(_item(_tree(tmp_path / "old", deferred=False)), "lane/c1", 1)


def test_a_leased_commit_mounts_its_queues_lease_pool_and_requests_no_gpu(dispatch, tmp_path, monkeypatch):
    tree = _tree(tmp_path / "t")
    monkeypatch.setattr(dispatch, "LEASE_HOSTDIRS", {"deployments-gpu": "/run/gpu-lease-circuits"})
    job = dispatch.render(_item(tree, lease="self"), "lane/c1", 1, 0)
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "deployments-gpu"
    pod = job["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert "nvidia.com/gpu" not in c["resources"]["requests"] and pod["hostPID"] and pod["runtimeClassName"] == "nvidia"
    vol = next(v for v in pod["volumes"] if v["name"] == "gpu-lease")
    assert vol["hostPath"]["path"] == "/run/gpu-lease-circuits"
    assert next(m for m in c["volumeMounts"] if m["name"] == "gpu-lease")["mountPath"] == "/run/gpu-lease"
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["GPU_LEASE_DIR"] == "/run/gpu-lease" and env["NVIDIA_VISIBLE_DEVICES"] == "all"
    monkeypatch.setattr(dispatch, "LEASE_HOSTDIRS", {})
    pod = dispatch.render(_item(tree, lease="self"), "lane/c1", 1, 0)["spec"]["template"]["spec"]
    assert next(v for v in pod["volumes"] if v["name"] == "gpu-lease")["hostPath"]["path"] == "/run/gpu-lease"


def test_a_node_2_build_resubmits_its_commit_with_the_items_lease_and_pinned_template(dispatch, tmp_path, monkeypatch):
    """n2_build.sh's Commit submit dropped the item's `lease` (4 of 10 leased items came back unleased, Oct 1) and its
    `template_sha`; its arguments must give dispatch.py's `submit` the item's own fields."""
    code = re.search(r"python3 - \"\$item\" << 'EOF'\n(.*?)\nEOF\n", (NEBIUS / "n2_build.sh").read_text(), re.S).group(1)
    tree, sha = _tree(tmp_path / "t"), dispatch.load_template("config-run")[1]
    item = {**_item(tree, lease="self", template_sha=sha, queue="circuits"),
            "n2": {"src": str(tree), "key": "c1", "mem_gb": 64, "cpus": 16}}
    (tmp_path / "item.json").write_text(json.dumps(item))
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "item.json")], capture_output=True, text=True,
                         check=True).stdout
    args = shlex.split(next(ln for ln in out.splitlines() if ln.startswith("COMMIT_ARGS=("))[len("COMMIT_ARGS=("):-1])
    got = []
    monkeypatch.setattr(dispatch, "submit", lambda it, key, idx=0, tries=0, dry=False: got.append((it, key, idx)))
    dispatch.main(["submit", "config-run", "n2-build/c1", "--tree", str(tree), "--task", "1", *args])
    (sent, key, idx), = got
    assert (key, idx) == ("n2-build/c1", 1) and sent["env"] == item["env"]
    assert (sent.get("lease"), sent.get("template_sha"), sent.get("queue")) == ("self", sha, "circuits")


def test_a_one_off_invocation_takes_the_loops_settings_from_dispatch_env(dispatch, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ.pop("VY_LEASE_HOSTDIRS", None)
    os.environ.pop("VY_DISPATCH_CPUS", None)
    os.environ["VY_PROVER_CPUS"] = "128-191"
    dispatch.ROOT.mkdir(parents=True, exist_ok=True)
    (dispatch.ROOT / "dispatch.env").write_text("# node 1's loop\nVY_LEASE_HOSTDIRS=deployments-gpu=/run/gpu-lease-circuits\n"
                                                "VY_DISPATCH_CPUS=96-127\nVY_PROVER_CPUS=128-175\n")
    mod = _load(NEBIUS / "dispatch.py", "dispatch", monkeypatch)
    assert mod.LEASE_HOSTDIRS == {"deployments-gpu": "/run/gpu-lease-circuits"}
    assert (mod.CPUS, mod.PROVER_CPUS) == ("96-127", "128-191")
    pod = mod.render(_item(_tree(tmp_path / "t"), lease="self"), "lane/c1", 1, 0)["spec"]["template"]["spec"]
    assert next(v for v in pod["volumes"] if v["name"] == "gpu-lease")["hostPath"]["path"] == "/run/gpu-lease-circuits"


def test_route_spools_only_under_the_flag_and_below_the_disk_pause(dispatch, tmp_path):
    item = _item(_tree(tmp_path / "t"))
    dispatch.route(item, "lane/c1", 1)
    assert [a[0] for a, _ in dispatch.calls] == ["create"] and not list(dispatch.PACK.glob("queue/*"))
    dispatch.calls.clear()
    dispatch.PACK_COMMITS = True
    dispatch.PACK_DISK_PAUSE = 0.0
    dispatch.route(item, "lane/c1", 1)
    assert [a[0] for a, _ in dispatch.calls] == ["create"] and not list(dispatch.PACK.glob("queue/*"))
    dispatch.calls.clear()
    dispatch.PACK_DISK_PAUSE = 1.01
    name = dispatch.route(item, "lane/c1", 1)
    entry = json.loads((dispatch.PACK / "queue" / f"{name}.json").read_text())
    assert not dispatch.calls and entry["key"] == "lane/c1" and entry["idx"] == 1
    assert entry["env"]["ROW"] == LLAMA_B8 and "row stage commit" in entry["script"]
    assert all((dispatch.PACK / d).stat().st_mode & 0o777 == 0o777 for d in ("", "queue", "claimed", "results/done", "logs", "stats"))
    assert dispatch.pack_keys() == {"lane/c1"}


def _result(dispatch, entry: dict, **res) -> None:
    (dispatch.PACK / "results").mkdir(parents=True, exist_ok=True)
    (dispatch.PACK / "results" / f"{entry['job']}.json").write_text(json.dumps({"entry": entry, "pod": "pod-a", **res}))


def test_results_continue_the_chain_or_requeue_once_unpacked(dispatch, tmp_path):
    dispatch.PACK_COMMITS = True
    item = _item(_tree(tmp_path / "t"))
    name = dispatch.spool(item, "lane/c1", 1)
    entry = json.loads((dispatch.PACK / "queue" / f"{name}.json").read_text())
    _result(dispatch, entry, rc=0, fault=False)
    dispatch.pack_results([])
    (_, job), = [(a, j) for a, j in dispatch.calls if a[0] == "create"]
    assert job["metadata"]["labels"]["verity.dev/task"] == "2"            # the CPU replay, as a normal Job
    assert json.loads(job["metadata"]["annotations"]["verity.dev/item-json"])["packed"] == "pod-a"   # so tick judges it
    dispatch.calls.clear()
    _result(dispatch, {**entry, "job": "x2"}, rc=None, fault=True, reason="stopped: pod fault")
    dispatch.pack_results([])
    (_, job), = [(a, j) for a, j in dispatch.calls if a[0] == "create"]
    assert job["metadata"]["labels"]["verity.dev/task"] == "1" and job["metadata"]["labels"]["verity.dev/try"] == "1"
    assert "unpacked" in json.loads(job["metadata"]["annotations"]["verity.dev/item-json"])
    dispatch.calls.clear()
    _result(dispatch, {**entry, "job": "x3", "item": {**item, "unpacked": "pod-a: fault"}}, rc=1, fault=False)
    dispatch.pack_results([])
    assert not dispatch.calls
    assert json.loads((dispatch.ROOT / "done.jsonl").read_text().splitlines()[-1])["state"] == "failed"


def test_an_entry_claimed_by_a_pod_that_is_gone_requeues_unpacked(dispatch, tmp_path):
    name = dispatch.spool(_item(_tree(tmp_path / "t")), "lane/c1", 1)
    claimed = dispatch.PACK / "claimed" / "pod-a"
    claimed.mkdir(parents=True)
    (dispatch.PACK / "queue" / f"{name}.json").rename(claimed / f"{name}.json")
    dispatch.pack_results([{"metadata": {"name": "pod-a"}, "status": {"phase": "Running"}}])
    assert not dispatch.calls and (claimed / f"{name}.json").exists()
    dispatch.pack_results([])
    (_, job), = [(a, j) for a, j in dispatch.calls if a[0] == "create"]
    assert "pod-a ended without a result" in json.loads(job["metadata"]["annotations"]["verity.dev/item-json"])["unpacked"]
    assert not claimed.exists()


def _log(dispatch, ev: str) -> list[dict]:
    return [r for r in map(json.loads, (dispatch.ROOT / "log.jsonl").read_text().splitlines()) if r.get("ev") == ev]


def test_a_stopped_model_is_refused_with_its_reason_until_circuits_lifts_it(dispatch, tmp_path):
    item = _item(_tree(tmp_path / "t"))
    other = _item(_tree(tmp_path / "u"), LLAMA_B8.replace("llama32-1b", "qwen25-05b"))
    dispatch.stop_model("llama32-1b", "lane/c1", "r-replay", "replay 459/460 equal", commit_run="r-commit", pod="pod-a", job="j")
    assert "stopped packing: replay 459/460 equal" in dispatch.packable(item, "lane/c2", 1)
    assert dispatch.packable(other, "lane/c3", 1) is None
    dispatch.PACK_COMMITS = True
    dispatch.route(item, "lane/c2", 1)
    assert [a[0] for a, _ in dispatch.calls] == ["create"] and not list(dispatch.PACK.glob("queue/*"))
    (stop,) = _log(dispatch, "pack-stop")
    assert (stop["model"], stop["key"], stop["run"], stop["reason"]) == ("llama32-1b", "lane/c1", "r-replay", "replay 459/460 equal")
    (note,) = (tmp_path / "alerts").glob("*-alert-pack-stop-llama32-1b-*.md")
    text = note.read_text()
    assert "lane: node1-dispatcher" in text and "r-commit" in text and "pack-lift llama32-1b" in text
    with pytest.raises(SystemExit, match="only circuits"):
        dispatch.lift_model("llama32-1b", "mps-pack", "looks fine")
    dispatch.lift_model("llama32-1b", "circuits", "a scheduling fault, not MPS")
    assert dispatch.packable(item, "lane/c2", 1) is None
    entries = json.loads(dispatch.stops_file().read_text())
    assert [(e["model"], bool(e.get("lift"))) for e in entries] == [("llama32-1b", False), ("llama32-1b", True)]
    assert all(e.get("t") for e in entries) and entries[0]["run"] == "r-replay" and entries[0]["key"] == "lane/c1"


def test_an_unreadable_stop_list_stops_every_model(dispatch, tmp_path):
    dispatch.pack_tree()
    dispatch.stops_file().write_text("{not json")
    assert "can't be read" in dispatch.packable(_item(_tree(tmp_path / "t")), "lane/c1", 1)


def _ended_replay(dispatch, item: dict, key: str, *, rc: int = 0, started: str = "2000-01-01T00:00:00Z") -> dict:
    job = dispatch.render(dispatch.pin(item), key, 2, 0)
    job["status"] = {"startTime": started, **({"succeeded": 1} if rc == 0 else {"failed": 1})}
    return job


def _tick(dispatch, jobs: list[dict], pods: list[dict] = ()) -> None:
    def kubectl(*args, stdin=None):
        dispatch.calls.append((args, json.loads(stdin) if stdin else None))
        if args[:2] == ("get", "jobs"):
            return json.dumps({"items": jobs})
        if args[:2] == ("get", "pods"):
            return json.dumps({"items": list(pods)})
        if args[:2] == ("get", "workloads"):
            return json.dumps({"items": []})
        if args[0] == "logs":
            return "research: run r-replay -> /workspace/jobs/runs/r-replay\n"
        return ""
    dispatch.kubectl = kubectl
    dispatch.tick()


def _record(sweep: Path, row: str, equal: int, evaluated: int = 460, k: int = 460, outcome: str = "PASS", **rp) -> None:
    d = sweep / row
    d.mkdir(parents=True, exist_ok=True)
    (d / "pipeline-commit-run").write_text("r-commit\n")
    (d / "config_record.json").write_text(json.dumps({"outcome": outcome, "replay": {"k": k, "evaluated": evaluated, "equal": equal,
                                                                                      "mismatch_n": evaluated - equal, **rp}}))


@pytest.mark.parametrize("record, rc, started, stopped", [
    ({"equal": 460}, 0, "2000-01-01T00:00:00Z", None),
    ({"equal": 459}, 0, "2000-01-01T00:00:00Z", "replay 459/460 equal"),
    ({"equal": 400, "evaluated": 400}, 0, "2000-01-01T00:00:00Z", "replay 400/400 equal of k=460"),
    ({"equal": 460, "not_evaluated_n": 3}, 0, "2000-01-01T00:00:00Z", "not evaluated 3"),   # a PASS on a named gap
    ({"equal": 460, "outcome": "FAIL"}, 0, "2000-01-01T00:00:00Z", "outcome FAIL"),
    ({"equal": 460}, 12, "2000-01-01T00:00:00Z", "the replay task failed (rc 12)"),
    ({"equal": 460}, 0, "2100-01-01T00:00:00Z", "predates the replay Job"),                  # a record of an earlier run
])
def test_a_packed_commits_replay_passes_only_at_460_of_460_or_stops_its_model(dispatch, tmp_path, record, rc, started, stopped):
    sweep = tmp_path / "sweep"
    item = _item(_tree(tmp_path / "t"), packed="pod-a")
    item["env"]["SWEEP_DIR"] = str(sweep)
    _record(sweep, LLAMA_B8, **record)
    job = _ended_replay(dispatch, item, "lane/c1", rc=rc, started=started)
    _tick(dispatch, [job], [{"metadata": {"name": "pod-r", "labels": {"job-name": job["metadata"]["name"]}},
                             "status": {"containerStatuses": [{"state": {"terminated": {"exitCode": rc}}}]}}])
    done = json.loads((dispatch.ROOT / "done.jsonl").read_text().splitlines()[-1])
    (verdict,) = _log(dispatch, "packed-replay")
    assert done["packed"] == "pod-a" and verdict["pod"] == "pod-a"
    if stopped is None:
        assert done["state"] == "succeeded" and verdict["verdict"] == "pass" and verdict["equal"] == 460
        assert not _log(dispatch, "pack-stop") and dispatch.packable(_item(_tree(tmp_path / "u")), "lane/c2", 1) is None
    else:
        assert done["state"] == "failed" and stopped in done["why"] and verdict["verdict"] == "fail"
        (stop,) = _log(dispatch, "pack-stop")
        assert (stop["model"], stop["run"], stop["commit_run"]) == ("llama32-1b", "r-replay", "r-commit") and stopped in stop["reason"]
        assert "stopped packing" in dispatch.packable(_item(_tree(tmp_path / "u")), "lane/c2", 1)


def test_an_unpacked_items_replay_is_not_judged(dispatch, tmp_path):
    sweep = tmp_path / "sweep"
    item = _item(_tree(tmp_path / "t"))
    item["env"]["SWEEP_DIR"] = str(sweep)
    _record(sweep, LLAMA_B8, equal=459)
    _tick(dispatch, [_ended_replay(dispatch, item, "lane/c1")])
    assert json.loads((dispatch.ROOT / "done.jsonl").read_text())["state"] == "succeeded"
    assert not _log(dispatch, "packed-replay") and not dispatch.stops_file().exists()


def test_the_pack_hold_sends_the_spool_out_deletes_unstarted_pods_and_drains_at_1233(dispatch, tmp_path):
    item = _item(_tree(tmp_path / "t"))
    dispatch.PACK_COMMITS = True
    dispatch.spool(item, "lane/c1", 1)
    pod_job = dispatch.render({"id": "p", "template": "commit-pack", "tree": str(NEBIUS), "env": {}}, "commit-pack/p1", 0, 0)
    waiting = {**pod_job, "metadata": {**pod_job["metadata"], "name": "pack-waiting"}, "status": {}}
    running = {**pod_job, "metadata": {**pod_job["metadata"], "name": "pack-running"}, "status": {"startTime": "2026-10-01T12:00:00Z"}}
    dispatch.held = lambda t=None: True
    dispatch.route(item, "lane/c2", 1)
    dispatch.pack_pods([])
    assert [a[0] for a, _ in dispatch.calls] == ["create"] and len(list(dispatch.PACK.glob("queue/*"))) == 1
    dispatch.calls.clear()
    dispatch.pack_hold([waiting, running])
    created = [j for a, j in dispatch.calls if a[0] == "create"]
    assert len(created) == 1 and created[0]["metadata"]["labels"]["verity.dev/task"] == "1"   # the Commit, as a normal Job
    assert "unpacked" not in json.loads(created[0]["metadata"]["annotations"]["verity.dev/item-json"])
    assert [a for a, _ in dispatch.calls if a[0] == "delete"] == [("delete", "job", "pack-waiting", "--wait=false")]
    assert not list(dispatch.PACK.glob("queue/*")) and not dispatch.pack_keys()
    dispatch.calls.clear()
    dispatch.draining = lambda t=None: True
    dispatch.pack_hold([running])
    assert [a for a, _ in dispatch.calls if a[0] == "delete"] == [("delete", "job", "pack-running", "--wait=false")]


def test_a_pack_pod_starts_only_when_a_queued_entry_fits_the_bundle_cap(dispatch, tmp_path, monkeypatch):
    """A pod that can claim nothing idles 3 min and exits: 66 pack pods (181 GPU-min) ran no Commit on Oct 1."""
    monkeypatch.setenv("PACK_BUNDLE_GLOBS", f"{tmp_path}/cov/*/commit/replay_bundle_p*")
    monkeypatch.setenv("PACK_BUNDLE_CAP_GB", "100")
    monkeypatch.setenv("PACK_BUNDLE_EST_GB", "60")
    monkeypatch.setattr(dispatch, "quiet", lambda t=None: False)
    dispatch.PACK_COMMITS = True
    tree = _tree(tmp_path / "t")

    def pods() -> int:
        n = sum(1 for a, j in dispatch.calls
                if a[0] == "create" and j["metadata"]["annotations"]["verity.dev/template"].startswith("commit-pack@"))
        dispatch.calls.clear()
        return n

    dispatch.spool(_item(tree, LLAMA_B8.replace("__i256__o32__", "__i1024__o128__")), "lane/c1", 1)
    dispatch.pack_pods([])
    assert pods() == 1
    entry = next(dispatch.PACK.glob("queue/*.json"))
    (dispatch.PACK / "claimed" / "pod-gone").mkdir(parents=True)
    (dispatch.PACK / "claimed" / "pod-gone" / "e.json").write_text(entry.read_text())
    dispatch.pack_pods([])
    assert pods() == 0
    dispatch.spool(_item(tree, LLAMA_B8.replace("__b8__", "__b1__")), "lane/c2", 1)
    dispatch.pack_pods([])
    assert pods() == 1


@pytest.mark.parametrize("path, drain", [("dispatch.py", (12, 33)), ("sky/commit_pack.py", (12, 28))])
def test_no_pack_pod_starts_or_claims_from_1210z_and_none_runs_past_1235z(monkeypatch, path, drain):
    before = dict(os.environ)
    mod = _load(NEBIUS / path, f"windows_{drain[1]}", monkeypatch)
    assert dict(os.environ) == before                    # on node 1 its dispatch.env reached the next tests (#925's train)
    day = 1790812800                                     # 2026-10-01T00:00:00Z

    def at(h: int, m: int) -> float:
        return day + h * 3600 + m * 60
    assert not mod.held(at(12, 9)) and mod.held(at(12, 10)) and mod.held(at(13, 29)) and not mod.held(at(13, 30))
    assert not mod.draining(at(*drain) - 60) and mod.draining(at(*drain)) and not mod.draining(at(13, 30))
    assert drain < (12, 35)


FAKE_MPS = """#!{py}
import os, sys
pipe = os.environ["CUDA_MPS_PIPE_DIRECTORY"]
if sys.argv[1:] == ["-d"]:
    if os.fork():
        sys.exit(0)
    os.setsid()
    os.mkfifo(os.path.join(pipe, "control"))
    with open(os.path.join(pipe, "control")) as f:
        f.read()
else:
    with open(os.path.join(pipe, "control"), "w") as f:
        f.write(sys.stdin.read())
"""


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    fake = bin_ / "nvidia-cuda-mps-control"
    fake.write_text(FAKE_MPS.format(py=os.path.realpath(sys.executable)))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("HOSTNAME", "pod-a")
    for k, v in {"PACK_SPOOL": tmp_path / "spool", "PACK_POLL_S": 0.01, "PACK_IDLE_S": 0, "PACK_STATS_S": 0,
                 "PACK_STOP_GRACE_S": 5, "PACK_DISK_PATH": tmp_path, "PACK_DISK_PAUSE": 1.01,
                 "PACK_BUNDLE_GLOBS": f"{tmp_path}/cov/*/commit/replay_bundle_p*"}.items():
        monkeypatch.setenv(k, str(v))
    mod = _load(NEBIUS / "sky" / "commit_pack.py", "commit_pack", monkeypatch)
    monkeypatch.setattr(mod, "held", lambda t: False)
    monkeypatch.setattr(mod, "draining", lambda t: False)
    umask = os.umask(0o022)
    yield mod
    os.umask(umask)


def _queue(spool: Path, name: str, script: str, row: str = "") -> None:
    (spool / "queue").mkdir(parents=True, exist_ok=True)
    (spool / "queue" / f"{name}.json").write_text(json.dumps({"key": f"lane/{name}", "idx": 1, "tries": 0, "job": name,
                                                               "env": {"ROW": row}, "script": script}))


def test_the_pilot_runs_commits_as_mps_clients_and_cleans_up(pilot, tmp_path):
    spool = tmp_path / "spool"
    _queue(spool, "ok", 'test "$CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" = 0=28G && test "$GPU_UTIL" = 0.20 && '
                        'test "$REPLAY_DEFERRED" = 1 && test "$(umask)" = 0022 && '
                        'test -S "$CUDA_MPS_PIPE_DIRECTORY/control" -o -p "$CUDA_MPS_PIPE_DIRECTORY/control"')
    assert pilot.main() == 0
    res = json.loads((spool / "results" / "ok.json").read_text())
    assert res["rc"] == 0 and res["fault"] is False and res["pod"] == "pod-a"
    for p in (spool / "claimed" / "pod-a", spool / "results", spool / "results" / "ok.json", spool / "logs" / "pod-a"):
        assert p.stat().st_mode & 0o002, p        # the dispatcher runs as another uid
    assert not list((spool / "claimed" / "pod-a").iterdir()) and not (tmp_path / "tmp" / "commit-pack-pod-a").exists()


def test_a_fault_in_one_client_stops_the_others_and_claims_nothing_more(pilot, tmp_path):
    spool = tmp_path / "spool"
    os.mkfifo(tmp_path / "never")
    _queue(spool, "a-stuck", f"read x < {tmp_path / 'never'}")
    _queue(spool, "b-fault", "echo 'CUDA error: an illegal memory access was encountered'; exit 1")
    _queue(spool, "c-later", "exit 0")
    os.utime(spool / "queue" / "c-later.json", (1e10, 1e10))
    pilot.SLOTS = 2
    assert pilot.main() == pilot.FAULT_EXIT
    b = json.loads((spool / "results" / "b-fault.json").read_text())
    a = json.loads((spool / "results" / "a-stuck.json").read_text())
    assert b["fault"] and "illegal memory access" in b["reason"] and a["fault"] and a["rc"] is None
    assert (spool / "queue" / "c-later.json").exists()


def test_no_claims_at_the_disk_pause_and_no_b8_while_bundles_pass_the_cap(pilot, tmp_path):
    spool = tmp_path / "spool"
    _queue(spool, "b8", "exit 0", LLAMA_B8)
    _queue(spool, "b1", "exit 0", LLAMA_B8.replace("__b8__", "__b1__"))
    pilot.DISK_PAUSE = 0.0
    assert pilot.main() == 0 and not (spool / "results").exists()
    pilot.DISK_PAUSE = 1.01
    bundle = tmp_path / "cov" / "row" / "commit" / "replay_bundle_p1"
    bundle.mkdir(parents=True)
    (bundle / "part").write_bytes(b"x" * 10)
    pilot.BUNDLE_CAP = 5
    assert pilot.main() == 0
    assert (spool / "results" / "b1.json").exists() and (spool / "queue" / "b8.json").exists()


def test_the_bundle_cap_counts_the_commits_running_and_the_candidate(pilot):
    def e(row):
        return {"env": {"ROW": row}}
    long_b8 = e(LLAMA_B8.replace("__i256__o32__", "__i1024__o128__"))
    short_b8, b1 = e(LLAMA_B8), e(LLAMA_B8.replace("__b8__", "__b1__"))
    assert pilot.bundle_est(long_b8) == pilot.BUNDLE_EST and pilot.bundle_est(short_b8) == pilot.BUNDLE_EST / 4
    n = int(pilot.BUNDLE_CAP // pilot.BUNDLE_EST)
    assert pilot.bundle_ok(long_b8, 0, [long_b8] * (n - 1)) and not pilot.bundle_ok(long_b8, 0, [long_b8] * n)
    assert pilot.bundle_ok(short_b8, 0, [short_b8, short_b8]) and not pilot.bundle_ok(short_b8, int(pilot.BUNDLE_CAP), [])
    assert pilot.bundle_ok(b1, int(pilot.BUNDLE_CAP) * 2, [long_b8, long_b8])


def test_a_full_pod_of_b8_1k_commits_fits_the_cap_with_each_models_measured_bundle(pilot):
    """At 150 GB and 120 GB for every model, one B8 1k Commit ran at a time: the grid's bundles measured 25-57 GB, and the cap
    held small Commits for 70-110 min with 6 of node 1's 8 GPUs empty (circuits, Oct 1)."""
    def e(model):
        return {"env": {"ROW": f"{model}__bf16__rtxpro6000__tp1__b8__i1024__o128__mixed__greedy__bi-eager"}}
    small, phi3 = e("qwen3-06b"), e("phi3-mini")
    assert pilot.bundle_est(small) >= 57e9 and pilot.bundle_est(phi3) >= 113e9
    for e8 in (small, phi3):
        assert pilot.bundle_ok(e8, 10**10, [e8] * (pilot.SLOTS - 1))


def test_the_default_bundle_globs_count_every_sweep_on_node_1_but_not_published_attempts(monkeypatch):
    monkeypatch.delenv("PACK_BUNDLE_GLOBS", raising=False)
    mod = _load(NEBIUS / "sky" / "commit_pack.py", "commit_pack_defaults", monkeypatch)
    phi3 = LLAMA_B8.replace("llama32-1b", "phi3-mini").replace("__i256__o32__", "__i1024__o128__")
    waiting = [f"/workspace/jobs/cov/cov-g217/{LLAMA_B8}/commit/replay_bundle_p1",
               f"/workspace/jobs/probe-jit/cfgtp2-deferred-phi3b8g/D/sweep/{phi3}/commit/replay_bundle_p0",
               f"/workspace/research/runs/cfgtp2-cpu/C3/sweep/{LLAMA_B8}/commit/replay_bundle_p0",
               f"/workspace/research/runs/cfgtp2-cpu/C5/{LLAMA_B8}/commit/replay_bundle_p0"]
    for p in waiting:
        assert any(Path(p).match(g) for g in mod.BUNDLE_GLOBS), p
    published = "/workspace/jobs/runs/r20260930-224845-a7df/outputs/verdict/commit/replay_bundle_p0"
    assert not any(Path(published).match(g) for g in mod.BUNDLE_GLOBS)


def test_a_bundle_linked_into_another_sweep_counts_once(pilot, tmp_path):
    bundle = tmp_path / "cov" / "a" / "commit" / "replay_bundle_p0"
    bundle.mkdir(parents=True)
    (bundle / "part").write_bytes(b"x" * 10)
    (tmp_path / "cov" / "b" / "commit").mkdir(parents=True)
    (tmp_path / "cov" / "b" / "commit" / "replay_bundle_p0").symlink_to(bundle)
    (tmp_path / "cov" / "c" / "commit" / "replay_bundle_p0").mkdir(parents=True)
    os.link(bundle / "part", tmp_path / "cov" / "c" / "commit" / "replay_bundle_p0" / "part")
    assert pilot.bundle_bytes(pilot.BUNDLE_GLOBS) == 10


def test_a_client_exiting_is_no_mps_fault_but_a_server_fault_is(pilot, tmp_path):
    mps = pilot.Mps(tmp_path / "scratch")
    mps.logs.mkdir(parents=True)
    (mps.logs / "control.log").write_text(
        "[2026-10-01 00:01:15.101 Control   790] NEW CLIENT 814 from user 1000: Server already exists\n"
        "[2026-10-01 00:01:15.102 Control   790] User did not send valid credentials\n"
        "[2026-10-01 00:01:22.590 Control   790] Server 805 has 2 active worker threads. Server will not shutdown.\n"
        "[2026-10-01 00:01:22.591 Control   790] Cmd:set_default_active_thread_percentage 100\n")
    (mps.logs / "server.log").write_text(
        "[2026-10-01 00:01:22.584 Server   805] Server failed to recevie command with status 806, "
        "assuming client {PID: 814, Context ID: 0} exit\n")
    assert mps.log_fault() is None
    with open(mps.logs / "server.log", "a") as f:
        f.write("[2026-10-01 00:02:00.000 Server   805] Client {PID: 815, Context ID: 0} encountered a fatal GPU fault\n")
    assert "fatal GPU fault" in mps.log_fault()
    with open(mps.logs / "control.log", "a") as f:
        f.write("[2026-10-01 00:02:01.000 Control   790] Server 805 hit a fatal exception, shutting down\n")
    assert "fatal exception" in mps.log_fault()


def test_in_the_hold_the_pilot_claims_nothing_and_exits(pilot, tmp_path):
    spool = tmp_path / "spool"
    _queue(spool, "later", "exit 0")
    pilot.held = lambda t: True
    assert pilot.main() == 0
    assert (spool / "queue" / "later.json").exists() and not (spool / "results").exists()


def test_at_the_drain_the_pilot_stops_its_commits_as_drained_not_faulted_and_exits(pilot, tmp_path):
    spool = tmp_path / "spool"
    os.mkfifo(tmp_path / "never")
    _queue(spool, "a", f"read x < {tmp_path / 'never'}")
    _queue(spool, "b", f"read x < {tmp_path / 'never'}")
    pilot.draining = lambda t: True                      # checked only once Commits run: the first pass claims both
    assert pilot.main() == 0
    for name in ("a", "b"):
        res = json.loads((spool / "results" / f"{name}.json").read_text())
        assert res["rc"] is None and res["fault"] is False and res["reason"].startswith("drained")


FAKE_RESEARCH = """import json, os, sys
open(os.environ["LABEL_ARGV"], "w").write(json.dumps(sys.argv[1:] + [os.environ.get("RESEARCH_STORE", "")]))
print("research data label: on r2://bucket", file=sys.stderr)
"""


def test_the_pilot_labels_each_packed_commits_attempt(pilot, tmp_path, monkeypatch):
    src, sweep, row = tmp_path / "tree", tmp_path / "sweep", LLAMA_B8
    (src / "tools/research/src/research").mkdir(parents=True)
    (src / "tools/research/src/research/__init__.py").write_text("")
    (src / "tools/research/src/research/__main__.py").write_text(FAKE_RESEARCH)
    monkeypatch.setenv("LABEL_ARGV", str(tmp_path / "argv.json"))
    env = {"SRC": str(src), "SWEEP_DIR": str(sweep), "ROW": row, "RESEARCH_STORE": "/workspace/jobs/store"}
    c = pilot.Client("n", {"key": "lane/c1", "env": env}, None, tmp_path / "log", 0, started=1.0)
    assert pilot.label_packed(c).startswith("unlabelled: ")                      # the Commit never wrote its run id
    (sweep / row).mkdir(parents=True)
    (sweep / row / "pipeline-commit-run").write_text("r20261001-070000-abcd\n")
    assert pilot.label_packed(c) == "labelled r20261001-070000-abcd: research data label: on r2://bucket"
    assert json.loads((tmp_path / "argv.json").read_text()) == [
        "data", "label", "r20261001-070000-abcd", "ov.mps_packed", "true", "--by", "mps-pack", "--off-vocab", "/workspace/jobs/store"]
    late = pilot.Client("n", {"key": "lane/c1", "env": env}, None, tmp_path / "log", 0, started=4e9)
    assert "predates this Commit" in pilot.label_packed(late)                    # an earlier try's run id


def test_claim_has_one_winner(pilot, tmp_path):
    spool = tmp_path / "spool"
    _queue(spool, "only", "exit 0")
    assert pilot.claim(spool, "pod-a")[0] == "only" and pilot.claim(spool, "pod-b") is None
