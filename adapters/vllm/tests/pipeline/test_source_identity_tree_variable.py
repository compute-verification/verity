"""`verity_vllm.pipeline.source_identity`'s reads of the environment, in this process.  ``VERITY_TREE``: the requested tree is
checked against the variable, and the manifest's record carries the variable as the environment holds it.  ``RESEARCH_SOURCE_SHA``:
it declares the commit where the command line names none, and the manifest carries it and ``PYTHONPATH`` as the environment holds
them.  The command's own tests (`test_source_identity`) run it in a child process, with the packages it resolves, the commits git
names and the run store's generic half; here all three are stood in, so no test starts a program or needs the run store, which
the public release leaves out."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.pipeline import source_identity as sid

INTEGRATION = Path(__file__).resolve().parents[2]  # <repo>/adapters/vllm
REPO = INTEGRATION.parents[1]
#: what the environment holds for the variable in each case; None is a variable that is not set
HELD = {"unset": None, "empty": "", "the repository": str(REPO), "the integration": str(INTEGRATION)}
#: a commit as the command line names one, and as the research runner's variable does
GIVEN, SHIPPED = "a" * 40, "f" * 40


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
        document=lambda ids, root, mm: {"requested_root": root, "ok": not mm, "fail_class": FAIL_CLASS if mm else None, "mismatches": mm})

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
