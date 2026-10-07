"""Node 1's dispatcher records the queue's placement of each Job it creates, with the item's research question (option A of
the `--queue` switch, top-level 1 Oct 5:14 AM PDT; circuits' yes at 5:15 on one condition: no change to a rendered Job, its env
or its tree). The queue is a stub `cluster` package here: this suite's inputs don't include infra/cluster."""
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
KINDS = """
[kinds.goldens]
keys = ["*-plan", "*-plan[0-9]*", "*-pk"]
phase = "gpu1"
max_wall_min = 30
question = "Does the golden twin reproduce its original?"

[kinds.grid-models]
keys = ["cov-gm*"]
phase = "gpu1"
max_wall_min = 30
question = "Does the grid row replay equal?"
"""
STUB = """import json, os, sys
a = sys.argv[1:]
with open(os.environ["STUB_ARGV"], "a") as f:
    f.write(json.dumps(a) + "\\n")
mode = open(os.environ["STUB_MODE"]).read().strip()
q = a[a.index("--question") + 1]
if mode == "refuse":
    print(json.dumps({"refused": "vy-nebius-1: too big"}))
    sys.exit(3)
if mode == "place":
    print(json.dumps({"node": "vy-nebius-1", "why": "starts now", "question": q or "the kind's", "warnings": []}))
"""


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
    q = tmp_path / "queue"
    (q / "src" / "cluster").mkdir(parents=True)
    (q / "src" / "cluster" / "__init__.py").write_text("")
    (q / "src" / "cluster" / "__main__.py").write_text(STUB)
    (q / "kinds").mkdir()
    (q / "kinds" / "circuits.toml").write_text(KINDS)
    monkeypatch.setattr(mod, "QUEUE_SRC", q)
    monkeypatch.setattr(mod, "PLACED", tmp_path / "placed.jsonl")
    monkeypatch.setattr(mod, "PLACE_FLAG", tmp_path / "place.on")
    monkeypatch.setenv("STUB_ARGV", str(tmp_path / "argv.jsonl"))
    monkeypatch.setenv("STUB_MODE", str(tmp_path / "mode"))
    (tmp_path / "mode").write_text("place")
    mod.created = []

    def kubectl(*args, stdin=None):
        assert args[0] == "create", args
        mod.created.append(json.loads(stdin))
        return ""

    monkeypatch.setattr(mod, "kubectl", kubectl)
    return mod


def _item(tmp_path, **kw) -> dict:
    return {"template": "config-run", "tree": str(tmp_path), "queue": "circuits", "env": {"ROW": "r"}, **kw}


def _rows(tmp_path) -> list[dict]:
    p = tmp_path / "placed.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def _argv(tmp_path) -> list[list[str]]:
    p = tmp_path / "argv.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def _without_times(job: dict) -> dict:
    return json.loads(json.dumps(job).replace(job["metadata"]["name"], "NAME"))


def test_the_flag_off_asks_no_queue_and_the_job_is_the_same_either_way(dispatch, tmp_path):
    dispatch.submit(_item(tmp_path), "vllm-epoch-run/cov-gm149", 1)
    assert _argv(tmp_path) == [] and _rows(tmp_path) == []
    (tmp_path / "place.on").touch()
    dispatch.submit(_item(tmp_path), "vllm-epoch-run/cov-gm149", 1)
    off, on = dispatch.created
    assert _without_times(on) == _without_times(off)
    (row,) = _rows(tmp_path)
    assert row | {"t": None} == {"t": None, "kind": "dispatch", "key": "vllm-epoch-run/cov-gm149", "job": on["metadata"]["name"],
                                 "family": "grid-models", "queue": on["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
                                 "question": "Does the grid row replay equal?", "executor": "kueue", "node": "vy-nebius-1",
                                 "why": "starts now"}
    (a,) = _argv(tmp_path)
    assert a[a.index("--kind") + 1] == "grid-models" and a[a.index("--node") + 1] == "vy-nebius-1"
    assert a[a.index("--cpus") + 1] == "8" and a[a.index("--mem-gb") + 1] == "64" and a[a.index("--submitter") + 1] == "circuits"


def test_an_items_question_overrides_its_family_and_the_first_matching_family_wins(dispatch, tmp_path):
    (tmp_path / "place.on").touch()
    dispatch.submit(_item(tmp_path, question="Is gm006's plan2 twin equal?"), "vllm-epoch-run/cov-gm006-plan2", 1)
    dispatch.submit(_item(tmp_path), "vllm-epoch-run/cov-gm006-plan2", 1)
    dispatch.submit(_item(tmp_path), "vllm-epoch-run/cov-x001", 1)
    own, golden, none = _rows(tmp_path)
    assert (own["family"], own["question"]) == ("goldens", "Is gm006's plan2 twin equal?")
    assert (golden["family"], golden["question"]) == ("goldens", "Does the golden twin reproduce its original?")
    assert none["family"] is None and none["question"] == "the kind's" and "--kind" not in _argv(tmp_path)[-1]
    dispatch.submit(_item(tmp_path), "proofs-verify-overlap/pvo-1", 1)
    assert len(dispatch.created) == 4 and len(_rows(tmp_path)) == 3


def test_a_refusal_or_no_answer_is_recorded_and_the_job_is_still_created(dispatch, tmp_path):
    (tmp_path / "place.on").touch()
    (tmp_path / "mode").write_text("refuse")
    dispatch.submit(_item(tmp_path), "vllm-epoch-run/cov-gm149", 1)
    (tmp_path / "mode").write_text("silent")
    dispatch.submit(_item(tmp_path), "vllm-epoch-run/cov-gm150", 1)
    assert len(dispatch.created) == 2
    refused, silent = _rows(tmp_path)
    assert refused["refused"] == "vy-nebius-1: too big" and "node" not in refused
    assert silent["error"].startswith("the queue gave no placement") and "node" not in silent
