"""Node 1's dispatcher handles each finished Job on its own. An exception while handling one (routing its next task, labelling it
seen) is that Job's: an `item-failed` event is logged, the Job stays unseen for the next tick, and the tick goes on to the other
Jobs and the ready files. The loop's tick line then says `tick failed`, so the steward's failed-tick alert still counts it.
On 1-2 Oct three AlreadyExists stalls in 10 h each stopped every tick, so no Job after the failing one was handled."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from research.pods import registry

NEBIUS = Path(registry.__file__).resolve().parent / "nebius"


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("VY_DISPATCH_ROOT", str(tmp_path / "dispatch"))
    monkeypatch.setenv("VY_READY", str(tmp_path / "ready"))
    for k in ("PACK_SPOOL", "PACK_COMMITS", "VY_DISPATCH_DEPTH"):
        monkeypatch.delenv(k, raising=False)
    spec = importlib.util.spec_from_file_location("dispatch_tick", NEBIUS / "dispatch.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "dispatch_tick", mod)
    spec.loader.exec_module(mod)
    mod.jobs, mod.labelled, mod.routed, mod.submitted = {}, [], [], []

    def kubectl(*args, stdin=None):
        if args[:2] in (("get", "jobs"), ("get", "pods"), ("get", "workloads")):
            return json.dumps({"items": list(mod.jobs.values()) if args[1] == "jobs" else []})
        if args[:2] == ("label", "job"):
            mod.labelled.append(args[2])
            return ""
        raise AssertionError(args)

    def route(item, key, idx, tries=0, dry=False):
        if key == "vllm-epoch-run/cov-a":
            raise RuntimeError('kubectl create -f -: Error from server (Forbidden): jobs.batch "x" is forbidden')
        mod.routed.append(key)

    monkeypatch.setattr(mod, "kubectl", kubectl)
    monkeypatch.setattr(mod, "route", route)
    monkeypatch.setattr(mod, "load_template", lambda name, sha=None: ([{"name": "build"}, {"name": "gpu"}], "s"))
    monkeypatch.setattr(mod, "render", lambda item, key, idx, tries: {"metadata": {"name": "job-c", "labels": {
        "kueue.x-k8s.io/queue-name": "circuits"}}})
    monkeypatch.setattr(mod, "submit", lambda item, key, idx=0, tries=0, dry=False: mod.submitted.append(key))
    for f in ("pack_results", "pack_hold"):
        monkeypatch.setattr(mod, f, lambda *a, **k: None)
    monkeypatch.setattr(mod, "n2_spill", lambda *a, **k: set())
    return mod


def finished_build(mod, name: str, key: str) -> None:
    item = {"id": key.split("/")[1], "template": "config-run", "tree": "/t", "template_sha": "s", "env": {}}
    mod.jobs[name] = {"metadata": {"name": name, "labels": {"verity.dev/task": "0", "verity.dev/try": "0"},
                                   "annotations": {"verity.dev/item": key, "verity.dev/item-json": json.dumps(item),
                                                   "verity.dev/template": "config-run@s"}},
                      "spec": {}, "status": {"succeeded": 1}}


def test_one_jobs_error_leaves_it_unseen_and_the_tick_goes_on_to_the_other_jobs_and_the_ready_files(dispatch, tmp_path, capsys):
    finished_build(dispatch, "job-a", "vllm-epoch-run/cov-a")      # routing its Commit raises
    finished_build(dispatch, "job-b", "vllm-epoch-run/cov-b")
    (tmp_path / "ready" / "vllm-epoch-run").mkdir(parents=True)
    (tmp_path / "ready" / "vllm-epoch-run" / "cov-c.json").write_text(json.dumps(
        {"template": "config-run", "tree": str(tmp_path), "queue": "circuits"}))

    failed = dispatch.tick()
    assert dispatch.routed == ["vllm-epoch-run/cov-b"] and dispatch.labelled == ["job-b"]    # job-a stays unseen
    assert dispatch.submitted == ["vllm-epoch-run/cov-c"]                                     # and the ready files still go
    assert len(failed) == 1 and failed[0].startswith("job-a: RuntimeError") and "Forbidden" in failed[0]
    log = [json.loads(line) for line in (tmp_path / "dispatch" / "log.jsonl").read_text().splitlines()]
    assert [(r["ev"], r["job"]) for r in log if r["ev"] == "item-failed"] == [("item-failed", "job-a")]
    assert "item-failed job-a (vllm-epoch-run/cov-a): RuntimeError" in capsys.readouterr().err

    assert dispatch.main(["loop", "--once"]) == dispatch.TICK_FAILED     # job-a fails again: the alert's `tick failed` line
    err = capsys.readouterr().err
    assert " tick failed: 1 Job(s) left unseen: job-a: RuntimeError" in err
