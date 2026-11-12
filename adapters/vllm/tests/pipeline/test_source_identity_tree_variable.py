"""`verity_vllm.pipeline.source_identity`'s two reads of ``VERITY_TREE``, in this process: the requested tree is checked against the variable, and the
manifest's record carries the variable as the environment holds it.  The command's own tests (`test_source_identity`) run it in a
child process, with the packages it resolves and the commits git names; here both are stood in, so no test starts a program."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from research.telemetry import source_identity as gsid

from verity_vllm.pipeline import source_identity as sid

INTEGRATION = Path(__file__).resolve().parents[2]  # <repo>/adapters/vllm
REPO = INTEGRATION.parents[1]
#: what the environment holds for the variable in each case; None is a variable that is not set
HELD = {"unset": None, "empty": "", "the repository": str(REPO), "the integration": str(INTEGRATION)}


def _hold(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("VERITY_TREE", raising=False)
    else:
        monkeypatch.setenv("VERITY_TREE", value)


@pytest.fixture
def no_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    """A manifest of no package, whose requested tree's commit is not asked of git: what is left is what `manifest` itself does."""
    monkeypatch.setattr(sid, "resolve", lambda root, *, sha=None: {})
    monkeypatch.setattr(gsid, "tree_identity", lambda root: {"git_sha": None, "git_sha_source": "not asked", "dirty_patch_sha256": None,
                                                             "branch": None})


@pytest.mark.parametrize("case", sorted(HELD))
def test_the_requested_tree_is_this_files_where_the_variable_is_unset_empty_or_names_it(monkeypatch, case):
    _hold(monkeypatch, HELD[case])
    assert sid.requested_tree() == str(REPO)
    assert sid.requested_tree(str(INTEGRATION)) == str(REPO)


def test_a_variable_that_names_another_tree_is_refused_by_name(monkeypatch, tmp_path):
    _hold(monkeypatch, str(tmp_path))
    with pytest.raises(gsid.SourceIdentityError, match="PRECHECK_FAIL_SOURCE_IDENTITY") as refusal:
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
    assert doc["ok"] is False and doc["fail_class"] == gsid.FAIL_CLASS and [m["package"] for m in doc["mismatches"]] == ["VERITY_TREE"]
    unchecked = sid.manifest(strict_env=False)
    assert unchecked["VERITY_TREE"] == str(tmp_path) and unchecked["ok"] is True
