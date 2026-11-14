"""golden.py (protected): a corpus log absent from the checkout is fetched, and one the fetch can't give is skipped with why; with
the fold left out, what `check` makes of each entry, what `record` writes, and what `main` prints and exits with. The frozen
corpus itself, whose logs are over a tracked file's limit, is `test_golden_of_record`."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.properties import golden

REPO = Path(__file__).resolve().parents[2]
#: entry name -> what its log folds to now: the two recorded values or others, or no fold at all
FOLDS = {"same": ("digest-same", "sha-same"), "digest": ("moved", "sha-digest"), "attribution": ("digest-attribution", "moved"),
         "both": ("moved", "moved"), "raises": None}
WHY_ABSENT = "log not present in this checkout; fetch-fixtures rc 1: MISSING"


def test_an_absent_log_is_fetched_and_one_the_fetch_cannot_give_is_skipped_with_why(tmp_path, monkeypatch):
    entry = {"name": "x", "log": "data/logs/no-such.jsonl.gz", "profile": "p", "digest": "d", "attribution_sha": "a"}
    (tmp_path / "corpus.json").write_text(json.dumps({"schema": "verity-gen/golden/v1", "entries": [entry]}))
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 1, "", "MISSING  data/logs/no-such.jsonl.gz\n")
    monkeypatch.setattr(golden.subprocess, "run", run)
    rep = golden.check(tmp_path / "corpus.json")
    assert calls == [([sys.executable, "-m", "research", "data", "fetch-fixtures", "--repo", str(REPO.parent.parent),
                       str(REPO / entry["log"])], {"capture_output": True, "text": True})]
    assert rep["status"] == "skipped"
    assert rep["entries"][0]["reason"] == ("log not present in this checkout; fetch-fixtures rc 1: "
                                           "MISSING  data/logs/no-such.jsonl.gz")


def test_a_log_that_is_here_is_not_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(golden.subprocess, "run", lambda *a, **kw: pytest.fail("fetched a log that is here"))
    (tmp_path / "here.jsonl.gz").write_bytes(b"")
    assert golden._fetch(tmp_path / "here.jsonl.gz") is None


def folded(instances: list[dict], report: dict, nodes: int | None = 3) -> SimpleNamespace:
    """A fold's result as `attribution` reads it: the instances, the report and the Program's body (none when `nodes` is)."""
    return SimpleNamespace(instances=instances, report=SimpleNamespace(to_dict=lambda: report),
                           body=None if nodes is None else SimpleNamespace(nodes=[None] * nodes))


def test_the_attribution_is_each_instances_step_event_pattern_and_specialization_in_order():
    """With the fold's seven counts and the number of nodes, and nothing else of the instance or the report."""
    inst: list[dict[str, Any]] = [{"step": 0, "event": 4, "pattern": "gemm", "spec": "Gemm{N=4}", "hint": "layers.0"},
                                  {"step": 0, "event": 9, "pattern": None, "spec": "Add{N=4}"},
                                  {"step": 1, "event": 4, "pattern": "gemm", "spec": "Gemm{N=4}"}]
    counts: dict[str, Any] = {"steps": 2, "computes": 3, "transparent": 1, "unsupported": 0, "by_pattern": {"gemm": 2}, "by_spec": {"Gemm{N=4}": 2},
                              "unsupported_ops": {}}
    sha, info = golden.attribution(folded(inst, {**counts, "seconds": 1.5}))
    assert info == {"n_instances": 3, "n_steps": 2, "patterns": ["", "gemm"], "specs": 2}
    doc = {"sequence": [[0, 4, "gemm", "Gemm{N=4}"], [0, 9, "", "Add{N=4}"], [1, 4, "gemm", "Gemm{N=4}"]], "report": counts, "n_nodes": 3}
    assert sha == hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert golden.attribution(folded([{**i, "hint": "elsewhere"} for i in inst], {**counts, "seconds": 9.0}))[0] == sha
    others = [[{**inst[0], key: value}, *inst[1:]] for key, value in (("step", 2), ("event", 5), ("pattern", "gemm2"), ("spec", "Gemm{N=8}"))]
    for other in (*others, [inst[1], inst[0], inst[2]], inst[:2]):
        assert golden.attribution(folded(other, counts))[0] != sha
    for key in counts:
        assert golden.attribution(folded(inst, {**counts, key: "changed"}))[0] != sha, key
    assert golden.attribution(folded(inst, counts, nodes=4))[0] != sha
    assert golden.attribution(folded(inst, counts, nodes=None)) == golden.attribution(folded(inst, counts, nodes=0))


def measured(digest: str, sha: str) -> dict:
    return {"digest": digest, "attribution_sha": sha, "unsupported": 0, "unresolved": 0, "seconds": 0.25, "n_instances": 3,
            "n_steps": 2, "patterns": ["gemm"], "specs": 1}


@pytest.fixture
def corpus(tmp_path, monkeypatch) -> Path:
    """A corpus of six entries with the fetch and the fold left out: `same` folds to its record, `digest`, `attribution` and
    `both` to another, `raises` cannot be folded, and the log of `absent` is not here."""
    entries = [{"name": name, "log": f"data/logs/{name}.jsonl.gz", "profile": f"profile-{name}", "digest": f"digest-{name}",
                "attribution_sha": f"sha-{name}", "n_instances": 4} for name in (*FOLDS, "absent")]
    (tmp_path / "corpus.json").write_text(json.dumps({"schema": "verity-gen/golden/v1", "entries": entries}))

    def measure(log, profile_name):
        name = Path(log).name.removesuffix(".jsonl.gz")
        assert log == REPO / "data" / "logs" / f"{name}.jsonl.gz" and profile_name == f"profile-{name}"
        fold = FOLDS[name]
        if fold is None:
            raise RuntimeError("no such pattern")
        return measured(*fold)
    monkeypatch.setattr(golden, "measure", measure)
    monkeypatch.setattr(golden, "_fetch", lambda log: WHY_ABSENT if log.name == "absent.jsonl.gz" else None)
    return tmp_path / "corpus.json"


def test_check_passes_an_entry_that_folds_to_its_record_and_says_why_another_does_not(corpus):
    rep = golden.check(corpus)
    assert set(rep) == {"schema", "status", "utc", "corpus", "entries", "n_pass", "n_fail", "n_skipped", "ok", "digest"}
    assert rep["schema"] == "verity-gen/golden-check/v1" and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", rep["utc"])
    by = {e["name"]: e for e in rep["entries"]}
    assert list(by) == [*FOLDS, "absent"] and rep["corpus"] == str(corpus)
    assert {name: (e["status"], e.get("reason")) for name, e in by.items()} == {
        "same": ("pass", None), "digest": ("fail", "digest changed"), "attribution": ("fail", " attribution changed"),
        "both": ("fail", "digest changed attribution changed"), "raises": ("fail", "RuntimeError: no such pattern"),
        "absent": ("skipped", WHY_ABSENT)}
    assert by["same"] == {"name": "same", "log": "data/logs/same.jsonl.gz", "profile": "profile-same",
                          "expected_digest": "digest-same", "expected_attribution_sha": "sha-same", "got_digest": "digest-same",
                          "got_attribution_sha": "sha-same", "n_instances": 3, "seconds": 0.25, "status": "pass"}
    assert (by["both"]["got_digest"], by["both"]["got_attribution_sha"], by["both"]["expected_n_instances"]) == ("moved", "moved", 4)
    assert "got_digest" not in by["raises"] and "got_digest" not in by["absent"]
    assert (rep["status"], rep["ok"], rep["n_pass"], rep["n_fail"], rep["n_skipped"]) == ("fail", False, 1, 4, 1)


@pytest.mark.parametrize("only, status, counts", [(["same"], "pass", (1, 0, 0)), (["same", "absent"], "pass", (1, 0, 1)),
                                                  (["absent"], "skipped", (0, 0, 1)), (["no such entry"], "skipped", (0, 0, 0)),
                                                  (["absent", "raises"], "fail", (0, 1, 1))])
def test_check_of_some_entries_fails_on_a_failure_and_passes_on_a_pass(corpus, only, status, counts):
    rep = golden.check(corpus, only=only)
    assert [e["name"] for e in rep["entries"]] == [name for name in (*FOLDS, "absent") if name in only]
    assert (rep["status"], rep["ok"]) == (status, status == "pass") and (rep["n_pass"], rep["n_fail"], rep["n_skipped"]) == counts


def test_main_prints_a_line_an_entry_and_exits_1_on_a_failure(corpus, tmp_path, monkeypatch, capsys):
    check = golden.check
    monkeypatch.setattr(golden, "check", lambda only=None: check(corpus, only))
    report = tmp_path / "reports" / "golden" / "golden.json"
    assert golden.main(cli.parse("golden", ["--out", str(report)], {})) == 1
    rep = json.loads(report.read_text())
    assert rep["status"] == "fail" and [e["name"] for e in rep["entries"]] == [*FOLDS, "absent"]
    assert report.read_text() == json.dumps(rep, indent=1) + "\n"
    out = capsys.readouterr()
    assert out.out == "" and out.err.split("\n") == [
        "[golden] pass    same:  (3 instances, 0.25 s)",
        "[golden] fail    digest: digest changed (3 instances, 0.25 s)",
        "[golden] fail    attribution:  attribution changed (3 instances, 0.25 s)",
        "[golden] fail    both: digest changed attribution changed (3 instances, 0.25 s)",
        "[golden] fail    raises: RuntimeError: no such pattern (? instances, ? s)",
        f"[golden] skipped absent: {WHY_ABSENT} (? instances, ? s)",
        "[golden] fail: 1 pass, 4 fail, 1 skipped", ""]
    assert golden.main(cli.parse("golden", ["--only", "same", "--out", str(report)], {})) == 0
    assert json.loads(report.read_text())["status"] == "pass"


@pytest.mark.parametrize("only, last", [("same,absent", "[golden] pass: 1 pass, 0 fail, 1 skipped"),
                                        ("absent", "[golden] skipped: 0 pass, 0 fail, 1 skipped")])
def test_main_exits_0_where_nothing_failed_and_writes_no_report_it_is_not_asked_for(corpus, tmp_path, monkeypatch, capsys, only, last):
    check = golden.check
    monkeypatch.setattr(golden, "check", lambda only=None: check(corpus, only))
    assert golden.main(cli.parse("golden", ["--only", only], {})) == 0
    assert capsys.readouterr().err.split("\n")[-2:] == [last, ""]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["corpus.json"]


def test_record_writes_the_entry_in_place_of_the_one_of_its_name(tmp_path, monkeypatch):
    """A log named from the checkout's root is folded there and recorded under the name given; one named by its whole path is
    recorded from the root. The corpus file and its directory are made, and an entry recorded again moves to the end."""
    given = []

    def measure(log, profile_name):
        given.append((log, profile_name))
        return measured("d1", "a1")
    monkeypatch.setattr(golden, "measure", measure)
    path = tmp_path / "golden" / "of" / "record" / "corpus.json"
    first = golden.record("one", "data/logs/one.jsonl.gz", "prof", decision="D-7", corpus_path=path)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", first.pop("recorded_utc"))
    assert first == {"name": "one", "log": "data/logs/one.jsonl.gz", "profile": "prof", "digest": "d1", "attribution_sha": "a1",
                     "n_instances": 3, "n_steps": 2, "patterns": ["gemm"], "decision": "D-7"}
    golden.record("two", str(REPO / "data" / "logs" / "two.jsonl.gz"), "prof", corpus_path=path)
    assert given == [(REPO / "data" / "logs" / "one.jsonl.gz", "prof"), (str(REPO / "data" / "logs" / "two.jsonl.gz"), "prof")]
    monkeypatch.setattr(golden, "measure", lambda log, profile_name: measured("d2", "a2"))
    golden.record("one", "data/logs/one.jsonl.gz", "prof", corpus_path=path)
    doc = json.loads(path.read_text())
    assert doc["schema"] == "verity-gen/golden/v1" and path.read_text() == json.dumps(doc, indent=1) + "\n"
    assert [(e["name"], e["log"], e["digest"], e["decision"]) for e in doc["entries"]] == [
        ("two", "data/logs/two.jsonl.gz", "d1", None), ("one", "data/logs/one.jsonl.gz", "d2", None)]


def test_main_records_with_a_log_and_a_profile_and_prints_the_entry(monkeypatch, capsys):
    calls = []

    def record(*a, **kw):
        calls.append((a, kw))
        return {"name": "one", "digest": "d1"}
    monkeypatch.setattr(golden, "record", record)
    for argv in (["--record", "one"], ["--record", "one", "--log", "one.jsonl.gz"], ["--record", "one", "--profile", "prof"]):
        with pytest.raises(UsageError) as refused:
            golden.main(cli.parse("golden", argv, {}))
        assert str(refused.value) == "--record needs --log and --profile"
    assert calls == []
    assert golden.main(cli.parse("golden", ["--record", "one", "--log", "one.jsonl.gz", "--profile", "prof", "--decision", "D-7"], {})) == 0
    assert calls == [(("one", "one.jsonl.gz", "prof"), {"decision": "D-7"})]
    out = capsys.readouterr()
    assert out.out == json.dumps({"name": "one", "digest": "d1"}, indent=1) + "\n" and out.err == ""
