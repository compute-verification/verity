"""Node 1's dispatcher counts a task's Job that another path already created as submitted. On Oct 1 a Commit Job created
before its Build's end was routed made every tick fail on `create` for 2 h 20 min, so no Job after it was handled."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from research.pods import registry

pytest.importorskip("yaml")
NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
TEMPLATE = ("name: config-run\n---\nname: build\nresources:\n  cpus: 4\n  memory: 8\n  labels:\n    kueue.x-k8s.io/queue-name: q\n"
            "run: echo build\n---\nname: gpu\nresources:\n  cpus: 8\n  memory: 64\n  labels:\n    kueue.x-k8s.io/queue-name: q\n"
            "run: echo gpu\n")
KEY = "vllm-epoch-run/cov-gm390"


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("VY_DISPATCH_ROOT", str(tmp_path / "dispatch"))
    monkeypatch.setenv("VY_READY", str(tmp_path / "ready"))
    monkeypatch.delenv("PACK_SPOOL", raising=False)
    monkeypatch.delenv("PACK_COMMITS", raising=False)
    spec = importlib.util.spec_from_file_location("dispatch", NEBIUS / "dispatch.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "dispatch", mod)
    spec.loader.exec_module(mod)
    (tmp_path / "sky" / "jobs").mkdir(parents=True)
    (tmp_path / "sky" / "jobs" / "config-run.yaml").write_text(TEMPLATE)
    monkeypatch.setattr(mod, "TEMPLATES", tmp_path / "sky" / "jobs")
    monkeypatch.setattr(mod, "PLACE_FLAG", tmp_path / "place.on")
    mod.cluster = {}

    def kubectl(*args, stdin=None):
        if args[0] == "create":
            job = json.loads(stdin)
            name = job["metadata"]["name"]
            if name in mod.cluster:
                raise RuntimeError(f'kubectl create -f -: Error from server (AlreadyExists): jobs.batch "{name}" already exists')
            mod.cluster[name] = job
            return ""
        if args[:2] == ("get", "job"):
            if args[2] not in mod.cluster:
                raise RuntimeError(f'kubectl get job {args[2]}: Error from server (NotFound): jobs.batch "{args[2]}" not found')
            return json.dumps(mod.cluster[args[2]])
        raise AssertionError(args)

    monkeypatch.setattr(mod, "kubectl", kubectl)
    return mod


def _log(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "dispatch" / "log.jsonl").read_text().splitlines()]


def _item(tmp_path) -> dict:
    return {"template": "config-run", "tree": str(tmp_path), "queue": "circuits", "env": {"ROW": "r"}}


def test_a_task_whose_job_already_exists_as_that_task_and_try_counts_as_submitted(dispatch, tmp_path):
    name = dispatch.submit(_item(tmp_path), KEY, 1)
    assert dispatch.submit(_item(tmp_path), KEY, 1) == name
    assert [(r["ev"], r["job"], r["task"], r["try"]) for r in _log(tmp_path)] == [("submit", name, 1, 0), ("exists", name, 1, 0)]
    assert list(dispatch.cluster) == [name]


def test_a_name_clash_with_another_items_job_still_fails(dispatch, tmp_path):
    name = dispatch.submit(_item(tmp_path), KEY, 1)
    dispatch.cluster[name]["metadata"]["annotations"]["verity.dev/item"] = "vllm-epoch-run/cov-gm391"
    with pytest.raises(RuntimeError, match="AlreadyExists"):
        dispatch.submit(_item(tmp_path), KEY, 1)
    dispatch.cluster[name]["metadata"]["annotations"]["verity.dev/item"] = KEY
    dispatch.cluster[name]["metadata"]["labels"]["verity.dev/try"] = "1"
    with pytest.raises(RuntimeError, match="AlreadyExists"):
        dispatch.submit(_item(tmp_path), KEY, 1)
    assert [r["ev"] for r in _log(tmp_path)] == ["submit"]
