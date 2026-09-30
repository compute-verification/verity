"""vy-nebius-1's dispatcher (pods/nebius/dispatch.py): a template task renders as a Kueue-labelled batch/v1 Job pinned off the check
slots, a two-task class sets each task's memory, and ready files are taken in rank order with the workstreams taking turns."""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

from research.pods import registry

pytest.importorskip("yaml")
DISPATCH = Path(registry.__file__).resolve().parent / "nebius" / "dispatch.py"
CELL = {"ROW": "smollm2-135m__x", "ROLE": "B0", "REPO": "HuggingFaceTB/SmolLM2-135M", "REVISION": "93efa2f0"}


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("VY_DISPATCH_ROOT", str(tmp_path / "dispatch"))
    monkeypatch.setenv("VY_READY", str(tmp_path / "ready"))
    spec = importlib.util.spec_from_file_location("dispatch", DISPATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_run_renders_two_chained_jobs(dispatch):
    item = {"id": "c1", "template": "config-run", "tree": "/t", "class": "small", "env": CELL}
    build, gpu = (dispatch.render(item, "lane/c1", i, 0) for i in (0, 1))
    assert build["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "deployments-cpu"
    assert gpu["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "deployments-gpu"
    for job in (build, gpu):
        assert job["spec"]["suspend"] is True and job["spec"]["backoffLimit"] == 0
        pod = job["spec"]["template"]
        assert not any(k.startswith("kueue.x-k8s.io/") for k in pod["metadata"]["labels"])
        c = pod["spec"]["containers"][0]
        assert c["command"][:3] == ["taskset", "-c", "96-127"]
        env = {e["name"]: e["value"] for e in c["env"]}
        assert env["SRC"] == "/t" and env["ROW"] == CELL["ROW"] and env["OMP_NUM_THREADS"] == c["resources"]["requests"]["cpu"]
        assert c["resources"]["requests"]["memory"] == "64G"
    assert "nvidia.com/gpu" not in build["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
    assert gpu["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"] == {"nvidia.com/gpu": "1"}
    assert gpu["spec"]["template"]["spec"]["runtimeClassName"] == "nvidia"
    assert build["metadata"]["name"] != gpu["metadata"]["name"] and len(gpu["metadata"]["name"]) <= 63
    tp2 = dispatch.render({**item, "class": "tp2"}, "lane/c1", 1, 0)["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert tp2["requests"]["cpu"] == "8" and tp2["limits"] == {"nvidia.com/gpu": "2"}


def test_backfill_item_gets_backfill_priority_and_missing_env_is_refused(dispatch):
    item = {"id": "c1", "template": "config-run", "tree": "/t", "queue": "backfill", "env": CELL}
    labels = dispatch.render(item, "lane/c1", 1, 0)["metadata"]["labels"]
    assert labels["kueue.x-k8s.io/queue-name"] == "backfill" and labels["kueue.x-k8s.io/priority-class"] == "backfill"
    with pytest.raises(ValueError, match="ROW"):
        dispatch.render({**item, "env": {k: v for k, v in CELL.items() if k != "ROW"}}, "lane/c1", 0, 0)


def test_ready_files_rank_first_then_workstreams_take_turns(dispatch, tmp_path):
    t = time.time()
    for n, (ws, name, rank) in enumerate([("a", "a1", 0), ("a", "a2", 0), ("a", "a3", 0), ("b", "b1", 0), ("b", "b2", 5)]):
        f = tmp_path / "ready" / ws / f"{name}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f'{{"template": "prover-dev", "tree": "/t", "rank": {rank}}}')
        os.utime(f, (t + n, t + n))
    assert [k for k, _, _ in dispatch.ready_items()] == ["b/b2", "a/a1", "b/b1", "a/a2", "a/a3"]
    key, _, f = dispatch.ready_items()[0]
    dispatch.set_aside(f, key, "taken")
    assert not f.exists() and len(list((tmp_path / "dispatch" / "taken" / "b").glob("b2.*.json"))) == 1
