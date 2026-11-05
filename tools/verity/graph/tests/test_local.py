"""The reference executor and `runners.execute` on the toy checkout, with stand-in runners: order, reads placed at their
producers' paths, keys from the reads' digests, and invariant 6: an error is retried and never a result, a fail never is."""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
from pathlib import Path

import pytest
from conftest import LISTERS

from verity_graph import cli, contract, families, graph, local, runners


def launcher(doc: dict, behave: dict, calls: list[str]):
    """A launch that runs each task through `runners.execute` with a stand-in runner: ``behave[id]`` (default: write its
    outputs and pass) given (task, work)."""
    def write(task: dict, work: Path) -> dict:
        for o in task["outputs"]:
            (work / o["path"]).parent.mkdir(parents=True, exist_ok=True)
            (work / o["path"]).write_text(f"{task['id']} {o['name']}\n")
        return {"status": "pass"}

    def launch(task: dict, work: Path) -> dict:
        calls.append(task["id"])
        return runners.execute(task, doc, work, runner=behave.get(task["id"], write))
    return launch


def run(tree: Path, tmp_path: Path, behave: dict | None = None, retries: int = 1):
    doc = graph.listing(tree, listers=LISTERS)
    calls: list[str] = []
    events = io.StringIO()
    out = local.check(tree, None, tmp_path / "x", launcher(doc, behave or {}, calls), retries, events, io.StringIO(), doc)
    return out, calls, [json.loads(x) for x in events.getvalue().splitlines()], doc


def test_tasks_run_in_order_with_their_reads_in_place(tree, tmp_path):
    seen = {}

    def reader(task: dict, work: Path) -> dict:
        seen["a"] = (work / "out/a").read_text()
        (work / "out/b").parent.mkdir(parents=True, exist_ok=True)
        (work / "out/b").write_text("b\n")
        return {"status": "pass", "detail": {"read": True}}

    out, calls, events, doc = run(tree, tmp_path, {"toy-b:b": reader})
    assert out["ok"] and calls == ["other:x", "toy:a", "toy-b:b"] and seen["a"] == "toy:a o\n"
    a, b = out["results"]["toy:a"], out["results"]["toy-b:b"]
    assert a["key"] == next(t["key"] for t in doc["tasks"] if t["id"] == "toy:a")
    task_b = next(t for t in doc["tasks"] if t["id"] == "toy-b:b")
    assert b["key"] == contract.key(task_b, doc["files"], {"toy:a": a["outputs"]}, doc["envs"]["py"]["digest"])
    assert b["outputs"] == {"o": contract.digest(b"b\n")} and b["detail"] == {"read": True}
    assert contract.result_errors(b, task_b) == []
    assert [e["event"] for e in events if e["task"] == "toy-b:b"] == ["queued", "started", "finished"]
    assert all(e["schema"] == "verity-events/v1" and "owner" not in e for e in events)


def test_a_fail_is_a_result_and_never_retried(tree, tmp_path):
    out, calls, events, _ = run(tree, tmp_path, {"toy:a": lambda task, work: {"status": "fail"}}, retries=3)
    assert calls.count("toy:a") == 1 and out["results"]["toy:a"]["status"] == "fail"
    assert out["cancelled"] == ["toy-b:b"] and not out["ok"]
    assert [e["event"] for e in events if e["task"] == "toy-b:b"] == ["queued", "cancelled"]


def test_an_error_is_retried_and_never_a_result(tree, tmp_path):
    def broken(task, work):
        raise OSError("disk full")
    out, calls, events, _ = run(tree, tmp_path, {"toy:a": broken}, retries=2)
    assert calls.count("toy:a") == 3
    assert "toy:a" not in out["results"] and out["errors"]["toy:a"]["status"] == "error"
    assert "disk full" in out["errors"]["toy:a"]["detail"]["error"]
    assert [e["event"] for e in events if e["task"] == "toy:a"] == ["queued"] + ["started", "errored"] * 3


def test_an_error_that_clears_on_retry_gives_the_result(tree, tmp_path):
    tries = []

    def flaky(task, work):
        tries.append(1)
        if len(tries) == 1:
            return {"status": "error", "detail": {"error": "worker died"}}
        (work / "out").mkdir(exist_ok=True)
        (work / "out/a").write_text("a\n")
        return {"status": "pass"}
    out, calls, _, _ = run(tree, tmp_path, {"toy:a": flaky})
    assert out["ok"] and calls.count("toy:a") == 2


def test_an_undeclared_read_is_refused(tree, tmp_path):
    out, _, _, _ = run(tree, tmp_path, {"other:x": lambda task, work: {"status": "pass", "undeclared_reads": ["README"]}})
    assert out["errors"]["other:x"]["undeclared_reads"] == ["README"] and not out["ok"]
    assert "other:x" not in out["results"]


def test_check_local_writes_its_events_beside_its_work_unless_told_where(tree, tmp_path, monkeypatch, capsys):
    doc = graph.listing(tree, listers=LISTERS)
    monkeypatch.setattr(cli, "ROOT", tree)
    monkeypatch.setattr(families, "listers", lambda: LISTERS)
    monkeypatch.setattr(local, "launcher", lambda root, listed: launcher(doc, {}, []))
    for argv, where in (([], tmp_path / "w" / "events.jsonl"), (["--events", str(tmp_path / "e.jsonl")], tmp_path / "e.jsonl")):
        assert cli.main(["check", "--local", "--work", str(tmp_path / "w"), *argv]) == 0
        assert json.loads(capsys.readouterr().out)["events"] == str(where)
        events = [json.loads(x) for x in where.read_text().splitlines()]
        assert {e["purpose"] for e in events} == {"adhoc"} and [e["event"] for e in events].count("finished") == 3
    results = json.loads((tmp_path / "w" / "results.json").read_text())
    assert sorted(results) == ["other:x", "toy-b:b", "toy:a"] and {r["status"] for r in results.values()} == {"pass"}


def write_outputs(task: dict, work: Path) -> dict:
    for o in task["outputs"]:
        (work / o["path"]).parent.mkdir(parents=True, exist_ok=True)
        (work / o["path"]).write_text(f"{task['id']} {o['name']}\n")
    return {"status": "pass"}


def test_a_run_given_its_part_runs_no_lister(tree, tmp_path, monkeypatch, capsys):
    doc = graph.listing(tree, listers=LISTERS)
    f = tmp_path / "graph.json"

    def lister():
        raise AssertionError("a run given its part of the listing lists nothing")
    monkeypatch.setattr(cli, "ROOT", tree)
    monkeypatch.setattr(families, "listers", lister)
    monkeypatch.setattr(runners, "run", write_outputs)
    for n, (tid, given) in enumerate((("toy:a", doc), ("toy:a", graph.part(doc, "toy:a")),
                                      ("toy-b:b", graph.part(doc, "toy-b:b")))):
        f.write_bytes(contract.canonical(given))
        work = tmp_path / "w" / str(n)
        (work / "out").mkdir(parents=True)
        (work / "out/a").write_text("toy:a o\n")
        assert cli.main(["run", tid, "--work", str(work), "--graph", str(f)]) == 0
        res = json.loads(capsys.readouterr().out)
        task = next(t for t in doc["tasks"] if t["id"] == tid)
        assert res["status"] == "pass" and res["key"] == contract.key(task, doc["files"], {"toy:a": {
            "o": contract.digest(b"toy:a o\n")}}, doc["envs"]["py"]["digest"]), "the key of the listing's task"
    (tree / "src/b.txt").write_text("changed\n")
    assert cli.main(["run", "toy-b:b", "--work", str(tmp_path / "w2"), "--graph", str(f)]) == 2, \
        "a refusal is neither a fail nor a run"
    assert "src/b.txt differs here" in capsys.readouterr().err


def in_process(argv: list[str], cwd: Path, **_) -> subprocess.CompletedProcess:
    """A spawn for `local.launcher` that runs `verity graph ...` in this process: what the run does is what the test sees."""
    assert argv[1] == "graph" and Path(cwd) == cli.ROOT, argv
    out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    with contextlib.redirect_stdout(out):
        rc = cli.main(argv[2:])
    out.flush()
    return subprocess.CompletedProcess(argv, rc, out.buffer.getvalue().decode(), "")


@pytest.mark.parametrize("change", [{}, {"src/a.txt": "changed\n"}, {"new.txt": "untracked\n"}],
                         ids=["clean", "an-input-changed", "an-untracked-file"])
def test_the_reference_executor_lists_once_and_no_task_s_run_lists_again(tree, tmp_path, monkeypatch, capsys, change):
    for p, text in change.items():
        (tree / p).write_text(text)
    calls: list[str] = []

    def counted(name, lister):
        def tasks(root, files):
            calls.append(name)
            return lister(root, files)
        return tasks
    spawned: list[list[str]] = []
    real = local.launcher
    monkeypatch.setattr(cli, "ROOT", tree)
    monkeypatch.setattr(families, "listers", lambda: {n: counted(n, f) for n, f in LISTERS.items()})
    monkeypatch.setattr(local, "launcher", lambda root, doc: real(root, doc, lambda argv, **kw: (
        spawned.append(argv), in_process(argv, **kw))[1]))
    monkeypatch.setattr(runners, "run", write_outputs)
    assert cli.main(["check", "--local", "--work", str(tmp_path / "w")]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert calls == ["other", "toy"], "the executor lists each family once, and no task's run lists again"
    assert len(spawned) == 3 and all(a[-2] == "--graph" and Path(a[-1]).is_file() for a in spawned)
    results = json.loads((tmp_path / "w" / "results.json").read_text())
    doc = graph.listing(tree, listers=LISTERS)
    assert results["toy:a"]["key"] == next(t["key"] for t in doc["tasks"] if t["id"] == "toy:a")
    assert doc["files"]["src/a.txt"] == contract.digest(change.get("src/a.txt", "a\n").encode())


def test_what_the_executor_takes_from_a_launch(tree, tmp_path):
    doc = graph.listing(tree, listers=LISTERS)
    a = next(t for t in doc["tasks"] if t["id"] == "toy:a")
    work = tmp_path / "w"
    (work / "out").mkdir(parents=True)
    (work / "out/a").write_text("a\n")
    good = runners.execute(a, doc, work, runner=lambda task, w: {"status": "pass"})
    assert local.judged(a, a["key"], good, work) == good
    assert "isn't the executor's" in local.judged(a, "sha512:00", good, work)["detail"]["error"]
    (work / "out/a").write_text("other\n")
    assert "isn't the bytes its digest names" in local.judged(a, a["key"], good, work)["detail"]["error"]
    assert local.judged(a, a["key"], {"error": "no output"}, work)["detail"]["error"] == "no output"
    assert local.judged(a, a["key"], {**good, "owner": "x"}, work)["status"] == "error"


def test_execute_turns_a_bad_run_into_a_fail_or_an_error(tree, tmp_path):
    doc = graph.listing(tree, listers=LISTERS)
    a = next(t for t in doc["tasks"] if t["id"] == "toy:a")
    b = next(t for t in doc["tasks"] if t["id"] == "toy-b:b")
    res = runners.execute(a, doc, tmp_path / "1", runner=lambda task, w: {"status": "pass"})
    assert res["status"] == "fail" and "without its outputs o" in res["detail"]["error"]
    res = runners.execute(a, doc, tmp_path / "2", runner=lambda task, w: {"status": "pass", "key": "sha512:00"})
    assert res["status"] == "error" and "the core's" in res["detail"]["error"]
    res = runners.execute(b, doc, tmp_path / "3", runner=lambda task, w: {"status": "pass"})
    assert res["status"] == "error" and res["key"] is None and "reads missing" in res["detail"]["error"]

    def late(task, work):
        assert runners.call(["true"], work, work) == 0
        runners._deadline[-1] = float("-inf")
        return {"status": "pass" if runners.call(["true"], work, work) == 0 else "fail"}
    res = runners.execute({**a, "outputs": []}, doc, tmp_path / "4", runner=late)
    assert res["status"] == "error" and res["detail"]["timeout_s"] == 60 and "outlived" in res["detail"]["error"], \
        "a timeout is no verdict: the key leaves timeout_s out, so a cached fail would outlive a longer limit"
    assert (tmp_path / "4" / graph.LOG).read_text().startswith("$ true")
