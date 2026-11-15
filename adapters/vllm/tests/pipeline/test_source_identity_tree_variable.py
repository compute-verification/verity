"""`verity_vllm.pipeline.source_identity`'s reads of the environment, in this process.  ``VERITY_TREE``: the requested tree is
checked against the variable, and the manifest's record carries the variable as the environment holds it.  ``RESEARCH_SOURCE_SHA``:
it declares the commit where the command line names none, and the manifest carries it and ``PYTHONPATH`` as the environment holds
them.  The command's own tests (`test_source_identity`) run it in a child process, with the packages it resolves, the commits git
names and the run store's generic half; here all three are stood in, so no test starts a program or needs the run store, which
the public release leaves out.  The same stand-ins hold, in this process, what the command and the precheck do with the
generic half's verdict (the record written before the refusal, the exit code and the lines each prints), the declared commit
given to a package or a requested tree that names none, the expected search path outside a workspace, and the code identity's
refusals."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.config import UsageError
from verity_vllm.pipeline import research_tools
from verity_vllm.pipeline import source_identity as sid

INTEGRATION = Path(__file__).resolve().parents[2]  # <repo>/adapters/vllm
REPO = INTEGRATION.parents[1]
#: what the environment holds for the variable in each case; None is a variable that is not set
HELD = {"unset": None, "empty": "", "the repository": str(REPO), "the integration": str(INTEGRATION)}
#: a commit as the command line names one, and as the research runner's variable does
GIVEN, SHIPPED = "a" * 40, "f" * 40
#: a key a record leaves out
ABSENT = object()


def _hold(monkeypatch: pytest.MonkeyPatch, value: str | None, name: str = "VERITY_TREE") -> None:
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


class Refused(Exception):
    """Stands in for the generic half's refusal, and keeps what the module raises it with."""

    def __init__(self, mismatches: list[dict[str, Any]], requested_root: str) -> None:
        super().__init__(f"{len(mismatches)} mismatch(es) against {requested_root}")
        self.mismatches, self.requested_root = mismatches, requested_root


FAIL_CLASS = "the stand-in's fail class"


@pytest.fixture
def generic(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The run store's generic half (`sid.GENERIC`), stood in where the module loads it by name: no commit is asked of git, no
    package is checked, and the document is the root and the mismatches the module hands it."""
    half = SimpleNamespace(
        FAIL_CLASS=FAIL_CLASS, SourceIdentityError=Refused,
        tree_identity=lambda root: {"git_sha": None, "git_sha_source": "not asked", "dirty_patch_sha256": None, "branch": None},
        check=lambda ids, root, *, requested_sha=None: [],
        document=lambda ids, root, mm: {"requested_root": root, "ok": not mm, "fail_class": FAIL_CLASS if mm else None, "mismatches": mm,
                                        "packages": ids})

    def run_store(module: str, entry: str) -> SimpleNamespace:
        assert module == sid.GENERIC, f"{entry} loads {module}, which this test does not stand in"
        return half

    monkeypatch.setattr(sid, "run_store", run_store)
    return half


@pytest.fixture
def no_packages(monkeypatch: pytest.MonkeyPatch, generic: SimpleNamespace) -> None:
    """A manifest of no package: what is left is what `manifest` itself does."""
    monkeypatch.setattr(sid, "resolve", lambda root, *, sha=None: {})


@pytest.mark.parametrize("case", sorted(HELD))
def test_the_requested_tree_is_this_files_where_the_variable_is_unset_empty_or_names_it(monkeypatch, case):
    _hold(monkeypatch, HELD[case])
    assert sid.requested_tree() == str(REPO)
    assert sid.requested_tree(str(INTEGRATION)) == str(REPO)


def test_a_variable_that_names_another_tree_is_refused_by_name(monkeypatch, generic, tmp_path):
    _hold(monkeypatch, str(tmp_path))
    with pytest.raises(generic.SourceIdentityError) as refusal:
        sid.requested_tree()
    (mismatch,) = refusal.value.mismatches
    other = os.path.realpath(tmp_path)
    assert (mismatch["package"], mismatch["resolved_path"], mismatch["repo_root"]) == ("VERITY_TREE", other, other)
    assert mismatch["why"] == f"$VERITY_TREE names another tree than the code that runs ({REPO})"
    assert refusal.value.requested_root == str(REPO)


@pytest.mark.parametrize("case", sorted(HELD))
def test_the_manifest_records_the_variable_as_the_environment_holds_it(monkeypatch, no_packages, case):
    _hold(monkeypatch, HELD[case])
    doc = sid.manifest()
    assert doc["VERITY_TREE"] == HELD[case] and doc["requested_root"] == str(REPO)
    assert doc["ok"] is True and doc["mismatches"] == []


def test_the_manifest_of_a_variable_that_names_another_tree_fails_and_records_it(monkeypatch, no_packages, tmp_path):
    _hold(monkeypatch, str(tmp_path))
    doc = sid.manifest()
    assert doc["VERITY_TREE"] == str(tmp_path) and doc["requested_root"] == str(REPO)
    assert doc["ok"] is False and doc["fail_class"] == FAIL_CLASS and [m["package"] for m in doc["mismatches"]] == ["VERITY_TREE"]
    unchecked = sid.manifest(strict_env=False)
    assert unchecked["VERITY_TREE"] == str(tmp_path) and unchecked["ok"] is True


@pytest.mark.parametrize("held", [None, "", SHIPPED])
def test_a_commit_the_command_line_names_is_the_declared_one_whatever_the_variable_holds(monkeypatch, held):
    _hold(monkeypatch, held, sid.ENV_SOURCE_SHA)
    assert sid.declared_sha(GIVEN) == (GIVEN, "--sha")


@pytest.mark.parametrize("held", [None, ""])
def test_no_commit_is_declared_where_the_command_line_names_none_and_the_variable_is_unset_or_empty(monkeypatch, held):
    _hold(monkeypatch, held, sid.ENV_SOURCE_SHA)
    assert sid.declared_sha() == (None, None) and sid.declared_sha("") == (None, None)


def test_the_variable_declares_the_commit_where_the_command_line_names_none(monkeypatch):
    _hold(monkeypatch, SHIPPED, sid.ENV_SOURCE_SHA)
    assert sid.ENV_SOURCE_SHA == "RESEARCH_SOURCE_SHA"
    assert sid.declared_sha() == (SHIPPED, "RESEARCH_SOURCE_SHA") and sid.declared_sha("") == (SHIPPED, "RESEARCH_SOURCE_SHA")


@pytest.mark.parametrize("name", ["PYTHONPATH", "RESEARCH_SOURCE_SHA"])
@pytest.mark.parametrize("held", [None, "", "a value"])
def test_the_manifest_records_the_search_path_and_the_runners_commit_as_the_environment_holds_them(monkeypatch, no_packages, name, held):
    _hold(monkeypatch, None)
    _hold(monkeypatch, held, name)
    doc = sid.manifest()
    assert doc[name] == held and name in doc
    assert doc["ok"] is True and doc["mismatches"] == []


#: the packages a manifest of `resolve` stood in resolves: one with a commit of its own, one whose tree names none
RESOLVED = {"verity": {"git_sha": "b" * 40}, "research": {"git_sha": None}}


@pytest.fixture
def resolved(monkeypatch: pytest.MonkeyPatch, generic: SimpleNamespace) -> None:
    monkeypatch.setattr(sid, "resolve", lambda root, *, sha=None: RESOLVED)


@pytest.mark.usefixtures("resolved")
def test_the_precheck_writes_the_record_with_its_stage_and_returns_it(monkeypatch, tmp_path):
    _hold(monkeypatch, None)
    out = tmp_path / "not" / "made" / "source_identity.json"
    doc = sid.precheck("commit", out=out)
    assert doc["stage"] == "commit" and doc["ok"] is True and doc["packages"] == RESOLVED
    assert out.read_text() == json.dumps(doc, indent=1, default=str) + "\n"


@pytest.mark.usefixtures("resolved")
def test_a_refused_precheck_writes_its_record_before_it_refuses(monkeypatch, tmp_path):
    _hold(monkeypatch, str(tmp_path / "other"))
    out = tmp_path / "record" / "source_identity.json"
    with pytest.raises(Refused) as refusal:
        sid.precheck("commit", out=out)
    record = json.loads(out.read_text())
    assert record["stage"] == "commit" and record["ok"] is False
    assert refusal.value.mismatches == record["mismatches"] and [m["package"] for m in record["mismatches"]] == ["VERITY_TREE"]
    assert refusal.value.requested_root == record["requested_root"] == str(REPO)


@pytest.mark.usefixtures("resolved")
@pytest.mark.parametrize("shown", [False, True])
def test_the_command_says_each_package_s_commit_and_the_tree_and_exits_0(monkeypatch, capsys, tmp_path, shown):
    _hold(monkeypatch, None)
    out = tmp_path / "source_identity.json"
    assert sid.main(sid.Options(out=str(out), print=shown)) == 0
    said = capsys.readouterr()
    record = json.dumps(json.loads(out.read_text()), indent=1, default=str) + "\n"
    assert said.out == (record if shown else "") + f"SOURCE-IDENTITY-OK verity={'b' * 9} research=- tree={REPO}\n"
    assert said.err == ""


@pytest.mark.usefixtures("resolved")
@pytest.mark.parametrize("written", [False, True])
def test_a_refused_command_says_the_refusal_and_where_its_record_is_and_exits_3(monkeypatch, capsys, tmp_path, written):
    _hold(monkeypatch, str(tmp_path / "other"))
    out = tmp_path / "source_identity.json"
    assert sid.main(sid.Options(out=str(out) if written else None, print=True)) == 3
    said = capsys.readouterr()
    assert said.out == "" and out.exists() is written
    assert said.err == f"1 mismatch(es) against {REPO}\n" + (f"[source_identity] record: {out}\n" if written else "")


def _resolving(monkeypatch: pytest.MonkeyPatch, generic: SimpleNamespace, records: dict[str, dict[str, Any]], core: list[str] | None) -> None:
    """`resolve` with the generic half and `research.pythonpath` stood in: each import name resolves to its record of `records`,
    and the workspace ships core from `core`'s directories (None: the tree is not a workspace)."""
    def package_dirs(root: str, name: str) -> list[str]:
        assert (root, name) == (str(REPO), "verity")
        if core is None:
            raise LookupError(root)
        return core

    def resolve_module(name: str, markers: tuple[str, ...]) -> dict[str, Any]:
        assert markers == sid.ROOT_MARKERS
        return dict(records[name])

    halves = {sid.GENERIC: SimpleNamespace(resolve_module=resolve_module), "research.pythonpath": SimpleNamespace(package_dirs=package_dirs)}
    monkeypatch.setattr(sid, "run_store", lambda module, entry: halves[module])


@pytest.mark.parametrize(("declared", "held", "source"), [(GIVEN, None, "--sha"), (None, SHIPPED, "RESEARCH_SOURCE_SHA"),
                                                          (GIVEN, SHIPPED, "--sha")])
def test_a_package_of_the_requested_tree_that_names_no_commit_takes_the_declared_one(monkeypatch, generic, tmp_path, declared, held, source):
    _hold(monkeypatch, held, sid.ENV_SOURCE_SHA)
    _resolving(monkeypatch, generic, {"verity": {"repo_root": str(REPO), "git_sha": None},
                                      "verity_vllm": {"repo_root": str(REPO), "git_sha": "b" * 40},
                                      "research": {"repo_root": str(tmp_path), "git_sha": None}}, [])
    ids = sid.resolve(sha=declared)
    assert (ids["verity"]["git_sha"], ids["verity"]["git_sha_source"]) == (declared or held, source)
    assert ids["verity_vllm"]["git_sha"] == "b" * 40 and "git_sha_source" not in ids["verity_vllm"]
    assert ids["research"]["git_sha"] is None and "git_sha_source" not in ids["research"]


@pytest.mark.parametrize("held", [None, ""])
def test_no_package_takes_a_commit_where_none_is_declared(monkeypatch, generic, held):
    _hold(monkeypatch, held, sid.ENV_SOURCE_SHA)
    _resolving(monkeypatch, generic, dict.fromkeys(sid.PACKAGES, {"repo_root": str(REPO), "git_sha": None}), [])
    assert all(rec["git_sha"] is None and "git_sha_source" not in rec for rec in sid.resolve().values())


@pytest.mark.parametrize(("core", "entries"), [(None, []), (["verity"], [""]), (["a/verity", "b/c/verity"], ["a", "b/c"])])
def test_core_is_expected_from_the_workspace_s_directories_and_from_none_outside_a_workspace(monkeypatch, generic, core, entries):
    _hold(monkeypatch, None, sid.ENV_SOURCE_SHA)
    _resolving(monkeypatch, generic, dict.fromkeys(sid.PACKAGES, {}), core)
    ids = sid.resolve()
    assert ids["verity"]["expected_sys_path_entry"] == os.pathsep.join(os.path.normpath(os.path.join(str(REPO), e)) for e in entries)


@pytest.mark.usefixtures("no_packages")
@pytest.mark.parametrize(("own", "declared", "held", "recorded"), [
    (None, GIVEN, None, (GIVEN, "--sha")), (None, None, SHIPPED, (SHIPPED, "RESEARCH_SOURCE_SHA")),
    (None, None, None, (None, ABSENT)), ("c" * 40, GIVEN, SHIPPED, ("c" * 40, ABSENT))])
def test_a_requested_tree_that_names_no_commit_is_recorded_with_the_declared_one(monkeypatch, generic, own, declared, held, recorded):
    _hold(monkeypatch, None)
    _hold(monkeypatch, held, sid.ENV_SOURCE_SHA)
    monkeypatch.setattr(generic, "document", lambda ids, root, mm: {"requested": {"git_sha": own}, "ok": True, "mismatches": mm})
    requested = sid.manifest(sha=declared)["requested"]
    assert (requested["git_sha"], requested.get("git_sha_source", ABSENT)) == recorded


def _closure(monkeypatch: pytest.MonkeyPatch, files: dict[str, str]) -> list[str]:
    """`research_tools.VLLM_BUILD` stood in (a name the module keeps once set): its closure is the integration and one directory of
    core, and its manifest is `files`.  Returns the roots it was asked for."""
    asked: list[str] = []

    def closure_manifest(root: str) -> dict[str, str]:
        asked.append(root)
        return files

    build = SimpleNamespace(closure=["adapters/vllm/**", "verity/**"], closure_manifest=closure_manifest)
    monkeypatch.setattr(research_tools, "VLLM_BUILD", build, raising=False)
    return asked


def test_the_code_identity_is_the_sha256_of_the_closure_s_canonical_manifest(monkeypatch, tmp_path):
    files = {"verity/b.py": "2" * 64, "adapters/vllm/a.py": "1" * 64}
    asked = _closure(monkeypatch, files)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    assert sid.code_identity(tmp_path) == {"sha256": hashlib.sha256(canonical).hexdigest(), "files": 2}
    assert asked == [tmp_path]


@pytest.mark.parametrize(("files", "missing"), [({"adapters/vllm/a.py": "1"}, "verity/"), ({"verity/b.py": "2"}, "adapters/vllm/"),
                                                ({"adapters/vllm/a.py": "1", "verityx/b.py": "2"}, "verity/")])
def test_a_root_with_no_file_under_a_directory_of_the_closure_is_refused_by_it(monkeypatch, tmp_path, files, missing):
    _closure(monkeypatch, files)
    with pytest.raises(FileNotFoundError) as refusal:
        sid.code_identity(tmp_path)
    assert str(refusal.value) == f"code identity: no {missing} file under {str(tmp_path)!r} (not a repository root)"


def test_a_code_identity_refused_for_want_of_the_run_store_names_the_entry_and_other_refusals_pass_as_they_are(monkeypatch, tmp_path):
    def code_identity(root):
        raise refusals.pop(0)

    refusals: list[Exception] = [UsageError("the run store is not here"), FileNotFoundError("not a repository root")]
    monkeypatch.setattr(sid, "code_identity", code_identity)
    with pytest.raises(UsageError) as wrapped:
        sid.code_identity_of("the Commit", tmp_path)
    assert str(wrapped.value) == "the Commit: the run store is not here"
    with pytest.raises(FileNotFoundError, match="^not a repository root$"):
        sid.code_identity_of("the Commit", tmp_path)
    monkeypatch.setattr(sid, "code_identity", lambda root: {"sha256": "s", "files": 1, "root": root})
    assert sid.code_identity_of("the Commit", tmp_path) == {"sha256": "s", "files": 1, "root": tmp_path}
