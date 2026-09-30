"""Node 1's MPS Commit packing: dispatch.py routes only eligible Commits to the spool (and only under PACK_COMMITS=1), reads
commit-pack results back into the chain (a fault requeues the Commit once, unpacked), and the pilot (sky/commit_pack.py) runs
spooled Commits under a pod-local MPS daemon, stopping the pod's other Commits when one faults."""
from __future__ import annotations

import importlib.util
import json
import os
import stat
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
    mod = _load(NEBIUS / "dispatch.py", "dispatch", monkeypatch)
    mod.calls = []

    def kubectl(*args, stdin=None):
        mod.calls.append((args, json.loads(stdin) if stdin else None))
        return "{}"

    monkeypatch.setattr(mod, "kubectl", kubectl)
    return mod


def test_commit_pack_renders_one_gpu_workload_in_deployments_gpu(dispatch):
    job = dispatch.render({"id": "p", "template": "commit-pack", "tree": str(NEBIUS), "env": {}}, "commit-pack/p1", 0, 0)
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "deployments-gpu"
    pod = job["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert c["resources"]["limits"] == {"nvidia.com/gpu": "1"} and c["resources"]["requests"]["cpu"] == "4"
    assert pod["runtimeClassName"] == "nvidia"
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["SRC"] == str(NEBIUS) and env["PACK_MEM_LIMIT"] == "0=28G" and env["PACK_GPU_UTIL"] == "0.28"
    assert "sky/commit_pack.py" in c["command"][-1] and (NEBIUS / "sky" / "commit_pack.py").exists()


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
    monkeypatch.setattr(mod, "quiet", lambda t: False)
    umask = os.umask(0o022)
    yield mod
    os.umask(umask)


def _queue(spool: Path, name: str, script: str, row: str = "") -> None:
    (spool / "queue").mkdir(parents=True, exist_ok=True)
    (spool / "queue" / f"{name}.json").write_text(json.dumps({"key": f"lane/{name}", "idx": 1, "tries": 0, "job": name,
                                                               "env": {"ROW": row}, "script": script}))


def test_the_pilot_runs_commits_as_mps_clients_and_cleans_up(pilot, tmp_path):
    spool = tmp_path / "spool"
    _queue(spool, "ok", 'test "$CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" = 0=28G && test "$GPU_UTIL" = 0.28 && '
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
    assert pilot.bundle_ok(long_b8, 0, []) and not pilot.bundle_ok(long_b8, 0, [long_b8])
    assert pilot.bundle_ok(short_b8, 0, [short_b8, short_b8]) and not pilot.bundle_ok(short_b8, int(pilot.BUNDLE_CAP), [])
    assert pilot.bundle_ok(b1, int(pilot.BUNDLE_CAP) * 2, [long_b8, long_b8])


def test_claim_has_one_winner(pilot, tmp_path):
    spool = tmp_path / "spool"
    _queue(spool, "only", "exit 0")
    assert pilot.claim(spool, "pod-a")[0] == "only" and pilot.claim(spool, "pod-b") is None
