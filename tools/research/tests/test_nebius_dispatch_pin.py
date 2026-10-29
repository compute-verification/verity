"""Node 1's dispatcher chains an item on the template content it started on: a template refreshed mid-chain (config-run going from
two tasks to three, 2:15 PM PDT Sep 30) applies only to items submitted after it. Twelve Commits dispatched on the two-task template
had their replay chained on the three-task one, which looked for a bundle the old template never sealed and exited 12."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from research.pods import registry

pytest.importorskip("yaml")
NEBIUS = Path(registry.__file__).resolve().parent / "nebius"


def _template(*tasks: str) -> str:
    return "name: config-run\n" + "".join(
        f"---\nname: {t}\nresources:\n  cpus: 4\n  memory: 8\n  labels:\n    kueue.x-k8s.io/queue-name: q\nrun: echo {t}\n"
        for t in tasks)


TWO, THREE = _template("build", "gpu"), _template("build", "gpu", "replay")


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
    monkeypatch.setattr(mod, "TEMPLATES", tmp_path / "sky" / "jobs")
    mod.jobs, mod.pods, mod.created, mod.labelled = [], [], [], []

    def kubectl(*args, stdin=None):
        if args[0] == "create":
            mod.created.append(json.loads(stdin))
            return ""
        if args[0] == "label":
            mod.labelled.append(args[2])
            return ""
        if args[:2] == ("get", "jobs"):
            return json.dumps({"items": mod.jobs})
        if args[:2] == ("get", "pods"):
            return json.dumps({"items": mod.pods})
        if args[:2] == ("get", "workloads"):
            return json.dumps({"items": []})
        raise AssertionError(args)

    monkeypatch.setattr(mod, "kubectl", kubectl)
    return mod


def _live(dispatch, text: str) -> str:
    (dispatch.TEMPLATES / "config-run.yaml").write_text(text)
    return dispatch.template_sha(text)


def _end(dispatch, job: dict, rc: int = 0) -> list[dict]:
    """Tick once with `job` ended (rc 0 succeeded, else failed with that exit code); the Jobs the tick created."""
    job = {**job, "status": {"succeeded": 1} if rc == 0 else {"failed": 1}}
    dispatch.jobs = [job]
    dispatch.pods = [] if rc == 0 else [{"metadata": {"labels": {"job-name": job["metadata"]["name"]}},
                                         "status": {"containerStatuses": [{"state": {"terminated": {"exitCode": rc}}}]}}]
    dispatch.created.clear()
    dispatch.tick()
    return list(dispatch.created)


def _done(dispatch) -> list[dict]:
    p = dispatch.ROOT / "done.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def _item(tmp_path) -> dict:
    return {"id": "a", "template": "config-run", "tree": str(tmp_path), "env": {}}


def test_a_chain_started_on_two_tasks_ends_after_its_commit_when_the_template_gains_a_replay(dispatch, tmp_path):
    two = _live(dispatch, TWO)
    dispatch.submit(_item(tmp_path), "lane/a")
    (build,) = dispatch.created
    assert build["metadata"]["annotations"]["verity.dev/template"] == f"config-run@{two}"
    assert (dispatch.PINNED / two / "config-run.yaml").read_text() == TWO
    three = _live(dispatch, THREE)
    (gpu,) = _end(dispatch, build)
    assert gpu["metadata"]["annotations"]["verity.dev/template"] == f"config-run@{two}"
    assert gpu["metadata"]["labels"]["verity.dev/task"] == "1"
    assert _end(dispatch, gpu) == []
    assert _done(dispatch)[-1] | {"t": None} == {"t": None, "key": "lane/a", "state": "succeeded", "rc": 0, "task": 1,
                                                  "job": gpu["metadata"]["name"]}
    dispatch.submit({**_item(tmp_path), "id": "b"}, "lane/b")
    assert dispatch.created[-1]["metadata"]["annotations"]["verity.dev/template"] == f"config-run@{three}"


def test_a_job_rendered_before_pinning_chains_on_its_annotation_and_is_held_without_that_content(dispatch, tmp_path):
    two = _live(dispatch, TWO)
    dispatch.submit(_item(tmp_path), "lane/a", 1, dry=True)
    job = dispatch.render(_item(tmp_path), "lane/a", 1, 0)
    item = json.loads(job["metadata"]["annotations"]["verity.dev/item-json"])
    assert "template_sha" not in item and job["metadata"]["annotations"]["verity.dev/template"] == f"config-run@{two}"
    assert not dispatch.PINNED.exists()                    # a dry run keeps no copy
    _live(dispatch, THREE)
    assert _end(dispatch, job) == [] and dispatch.labelled == []
    assert _done(dispatch) == [] and not (dispatch.ROOT / "log.jsonl").exists()
    dispatch.PINNED.joinpath(two).mkdir(parents=True)
    (dispatch.PINNED / two / "config-run.yaml").write_text(TWO)
    assert _end(dispatch, job) == [] and dispatch.labelled == [job["metadata"]["name"]]
    assert _done(dispatch)[-1]["state"] == "succeeded" and _done(dispatch)[-1]["task"] == 1


def test_a_resubmission_after_exit_99_renders_from_the_pinned_template(dispatch, tmp_path):
    two = _live(dispatch, TWO)
    dispatch.submit(_item(tmp_path), "lane/a", 1)
    (gpu,) = dispatch.created
    _live(dispatch, THREE)
    (again,) = _end(dispatch, gpu, rc=99)
    assert again["metadata"]["labels"]["verity.dev/try"] == "1"
    assert again["metadata"]["annotations"]["verity.dev/template"] == f"config-run@{two}"


def test_a_pinned_copy_with_other_content_is_refused(dispatch, tmp_path):
    two = _live(dispatch, TWO)
    dispatch.PINNED.joinpath(two).mkdir(parents=True)
    (dispatch.PINNED / two / "config-run.yaml").write_text(THREE)
    _live(dispatch, THREE)
    with pytest.raises(ValueError, match=f"config-run@{two}"):
        dispatch.render({**_item(tmp_path), "template_sha": two}, "lane/a", 0, 0)


def test_status_counts_an_item_by_its_last_done_line(dispatch):
    dispatch.append("done.jsonl", {"key": "lane/a", "state": "failed", "rc": 12, "task": 2})
    dispatch.append("done.jsonl", {"key": "lane/b", "state": "failed", "rc": 1, "task": 1})
    dispatch.append("done.jsonl", {"key": "lane/a", "state": "succeeded", "rc": 0, "task": 1, "corrects": {"rc": 12}})
    assert {k: r["state"] for k, r in dispatch.outcomes().items()} == {"lane/a": "succeeded", "lane/b": "failed"}


def test_a_provers_task_runs_on_the_prover_slices_and_every_other_task_off_them(dispatch, tmp_path):
    """vLLM jobs ran unpinned across the provers' 16-core slices and skewed the timed CPU verifier."""
    _live(dispatch, TWO)
    dispatch.submit(_item(tmp_path), "lane/a")
    dispatch.submit({**_item(tmp_path), "id": "p", "queue": "provers"}, "lane/p")
    vllm, prover = (j["spec"]["template"]["spec"]["containers"][0]["command"][:3] for j in dispatch.created)
    assert vllm == ["taskset", "-c", "96-127,176-191"] and prover == ["taskset", "-c", "128-175"]


def test_a_preempted_job_starts_its_next_pod_only_after_the_old_one_ended(dispatch, tmp_path):
    """A Commit Kueue reclaimed kept running through its grace while the resumed Job ran a second copy in the same row (00:17Z)."""
    _live(dispatch, TWO)
    dispatch.submit(_item(tmp_path), "lane/a")
    assert dispatch.created[0]["spec"]["podReplacementPolicy"] == "Failed"
