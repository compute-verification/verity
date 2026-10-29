"""A workload's bytes are a fixture in the evidence store (`config.workload_file`): every one is listed by its directory's SHA256SUMS
and registered in fixtures/artifacts.json, an absent one is fetched by its art id, a changed one is refused, an unlisted file passes
through, and `verity-vllm workload-file` makes them present for the shell scripts."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from verity_vllm import config as C
from verity_vllm.pipeline import cli

EXAMPLE = C.REPO / "examples" / "vllm" / "workloads"


def _stored(d: Path, docs: dict[str, dict]) -> dict[str, bytes]:
    d.mkdir(parents=True, exist_ok=True)
    data = {n: json.dumps(doc).encode() for n, doc in docs.items()}
    (d / C.DIGESTS).write_text("".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in sorted(data.items())))
    return data


def _fetch_from(data: dict[str, bytes], calls: list):
    def fetch(paths):
        calls.append([Path(p).name for p in paths])
        for p in paths:
            Path(p).write_bytes(data[Path(p).name])
    return fetch


def test_every_workload_is_a_fixture_its_directory_lists_and_the_registry_holds():
    """The tree keeps a workload's name and digest (SHA256SUMS) and its art id (fixtures/artifacts.json), never its bytes
    (tests/test_repository.py holds the registered paths gitignored)."""
    reg = json.loads((C.REPO / "fixtures" / "artifacts.json").read_text())
    dirs = {os.path.relpath(d, C.REPO) for d in (C.WORKLOADS, EXAMPLE)}
    registered = {v for k, v in reg.items() if k.startswith("art:") and isinstance(v, str) and os.path.dirname(v) in dirs}
    listed = {os.path.relpath(d / n, C.REPO) for d in (C.WORKLOADS, EXAMPLE) for n in C.stored_workloads(d)}
    assert registered == listed, (f"in fixtures/artifacts.json, not in a SHA256SUMS: {sorted(registered - listed)}; "
                                  f"in a SHA256SUMS, not registered: {sorted(listed - registered)}")
    assert len(C.stored_workloads(EXAMPLE)) == 1
    loose = sorted(p.name for p in C.WORKLOADS.glob("*.json") if p.name not in C.stored_workloads())
    assert not loose, (f"{loose}: workloads no SHA256SUMS lists. A workload is a fixture: `research data put --kind fixture/v1 --file "
                       "integrations/vllm/workloads/NAME --preserve`, its id and path in fixtures/artifacts.json, the path in .gitignore, "
                       "and `sha256sum NAME >> SHA256SUMS` in integrations/vllm/workloads/")


def test_a_present_workload_is_checked_against_its_digest_and_an_absent_one_fetched(tmp_path, monkeypatch):
    data = _stored(tmp_path, {"a.json": {"x": 1}, "b.json": {"x": 2}})
    calls: list = []
    monkeypatch.setattr(C, "_fetch", _fetch_from(data, calls))
    (tmp_path / "a.json").write_bytes(data["a.json"])
    assert C.workload_file(tmp_path / "a.json") == tmp_path / "a.json" and calls == []
    assert C.workload_file(str(tmp_path / "b.json")).read_bytes() == data["b.json"] and calls == [["b.json"]]
    assert C.workload_file(tmp_path / "b.json") and calls == [["b.json"]]   # present now: never refetched
    (tmp_path / "a.json").write_text(json.dumps({"x": 3}))
    with pytest.raises(RuntimeError, match=r"a\.json: SHA-256 [0-9a-f]{64} is not the [0-9a-f]{64} its SHA256SUMS lists: a changed "
                                           r"workload is a new fixture"):
        C.workload_file(tmp_path / "a.json")


def test_an_unlisted_workload_passes_through_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_fetch", lambda paths: pytest.fail(f"fetched {paths}"))
    (tmp_path / "w.json").write_text("{}")
    assert C.workload_file(tmp_path / "w.json") == tmp_path / "w.json"
    assert not C.workload_file(tmp_path / "absent.json").exists()
    _stored(tmp_path, {"other.json": {}})
    assert C.workload_file(tmp_path / "w.json") == tmp_path / "w.json"


def test_workload_files_lists_stored_and_present_ones_and_fetches_the_absent_in_one_call(tmp_path, monkeypatch):
    data = _stored(tmp_path, {"a.json": {"x": 1}, "b.json": {"x": 2}, "c.json": {"x": 3}})
    calls: list = []
    monkeypatch.setattr(C, "_fetch", _fetch_from(data, calls))
    (tmp_path / "b.json").write_bytes(data["b.json"])
    (tmp_path / "scratch.json").write_text("{}")
    assert [p.name for p in C.workload_files(tmp_path)] == ["a.json", "b.json", "c.json", "scratch.json"]
    assert calls == [["a.json", "c.json"]]


def test_a_failed_fetch_names_the_workload_and_the_fix(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "fixtures").mkdir(parents=True)
    (repo / "fixtures" / "artifacts.json").write_text("{}")
    _stored(repo / "w", {"a.json": {}})
    monkeypatch.chdir(repo / "w")
    seen = []

    def run(argv, **kw):
        seen.append((argv, kw))
        return subprocess.CompletedProcess(argv, 1, "", "no source configured")
    monkeypatch.setattr(C.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=r"w/a\.json: stored workloads, not here, and `research data fetch-fixtures` could not fetch "
                                           r"them \(rc 1: no source configured\)"):
        C.workload_file(repo / "w" / "a.json")
    (argv, kw), = seen
    assert argv[2:] == ["research", "data", "fetch-fixtures", "--repo", str(repo.resolve()), "--json", "w/a.json"]
    # fetch-fixtures reads a relative PATH against its cwd, not --repo
    assert (Path(kw.get("cwd") or os.getcwd()) / argv[-1]).resolve() == (repo / "w" / "a.json").resolve()
    assert kw["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(repo.resolve() / "tools" / "research" / "src")


def test_row_file_finds_a_stored_row_that_is_not_here(tmp_path, monkeypatch):
    row = "stored-elsewhere"
    data = _stored(tmp_path, {f"{row}.json": {"x": 1}})
    calls: list = []
    monkeypatch.setattr(C, "_fetch", _fetch_from(data, calls))
    monkeypatch.setenv("VERITY_WORKLOADS", str(tmp_path))
    assert C.row_file(row) == (tmp_path / f"{row}.json").resolve() and calls == [[f"{row}.json"]]


def test_the_workload_file_command_makes_each_present_and_refuses_a_changed_one(tmp_path, monkeypatch, capsys):
    data = _stored(tmp_path, {"a.json": {"x": 1}})
    monkeypatch.setattr(C, "_fetch", _fetch_from(data, []))
    assert cli.main(["workload-file", str(tmp_path / "a.json")]) == 0
    assert capsys.readouterr().out == f"{tmp_path / 'a.json'}\n"
    (tmp_path / "a.json").write_text("{}")
    with pytest.raises(RuntimeError, match="is not the"):
        cli.main(["workload-file", str(tmp_path / "a.json")])


def test_an_option_naming_a_changed_workload_is_a_usage_error(tmp_path, monkeypatch, capsys):
    data = _stored(tmp_path, {"a.json": {"x": 1}})
    (tmp_path / "a.json").write_bytes(data["a.json"] + b" ")
    with pytest.raises(SystemExit) as e:
        cli.main(["served-match", "--workload", str(tmp_path / "a.json")])
    assert e.value.code == 2 and "is not the" in capsys.readouterr().err
