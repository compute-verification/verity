"""Every merge driver `.gitattributes` names is set where drive and the gates merge (`research.clone.RECORD_DRIVER`): a clone's
own config holds none of them, and without its driver git merges the file as text.  An ignore file merges as git's own
`union`: two branches that each add a line at one place merge, and no line either side holds is lost."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from research.clone import RECORD_DRIVER

ROOT = Path(__file__).resolve().parents[1]
#: git's own merge drivers, which need no config
GIT_OWN = frozenset({"text", "binary", "union"})


def test_every_merge_driver_the_attributes_name_is_set_in_drives_clone_and_its_script_exists():
    named = {m for line in (ROOT / ".gitattributes").read_text().splitlines() if not line.lstrip().startswith("#")
             for m in re.findall(r"\bmerge=([\w-]+)", line)} - GIT_OWN
    assert {"lean-audit", "fixture-map", "vllm-allowlist"} <= named
    missing = sorted(n for n in named if f"merge.{n}.driver" not in RECORD_DRIVER)
    assert not missing, ("`.gitattributes` names merge drivers that drive's clone doesn't set, so it merges their files as "
                         "text: add merge.<name>.name and merge.<name>.driver to RECORD_DRIVER in "
                         f"tools/research/src/research/clone.py: {missing}")
    absent = sorted(n for n in named if not any((ROOT / p).is_file()
                                                for p in re.findall(r"[\w./-]+\.py", RECORD_DRIVER[f"merge.{n}.driver"])))
    assert not absent, f"RECORD_DRIVER runs a script that isn't in the tree for: {absent}"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "HOME": str(repo.parent),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=check, capture_output=True, text=True,
                          env=env)


def _merged_ignore(tmp_path: Path, base: str, ours: str, theirs: str) -> str | None:
    """A `.gitignore` after git merges two branches that made ``ours`` and ``theirs`` of ``base``, under the tree's
    `.gitattributes`; None when git leaves a conflict."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".gitattributes").write_text((ROOT / ".gitattributes").read_text())
    for name, text in (("main", base), ("ours", ours), ("theirs", theirs)):
        if name != "main":
            _git(repo, "checkout", "-q", "-b", name, "main")
        (repo / ".gitignore").write_text(text)
        _git(repo, "add", ".gitattributes", ".gitignore")
        _git(repo, "commit", "-q", "-m", name)
    _git(repo, "checkout", "-q", "ours")
    return None if _git(repo, "merge", "-q", "--no-edit", "theirs", check=False).returncode else (repo / ".gitignore").read_text()


def test_two_branches_that_each_add_an_ignore_line_at_one_place_merge(tmp_path):
    base = (ROOT / ".gitignore").read_text()
    assert _merged_ignore(tmp_path, base, base + "ours-only/\n", base + "theirs-only/\n") == base + "ours-only/\ntheirs-only/\n", (
        "two branches that each add a line at the end of `.gitignore` must merge with both lines: keep "
        "`.gitignore merge=union` in .gitattributes")


def test_a_union_of_ignore_lines_loses_none_and_keeps_a_re_include_below_its_pattern(tmp_path):
    """One branch renames two ignored paths and the other adds a line right below them, which git's text merge calls a
    conflict.  The union holds every line of both, so nothing either branch ignores stops being ignored; the two old lines
    come back beside their new names, which `tests/test_repository.py` shows only if they cover a tracked file."""
    head = "out/\n!kept/out/\n"
    ours, theirs = head + "new-a/\nnew-b/\n", head + "old-a/\nold-b/\nmore/\n"
    merged = _merged_ignore(tmp_path, head + "old-a/\nold-b/\n", ours, theirs)
    assert merged is not None, "`.gitignore` must merge as a union: keep `.gitignore merge=union` in .gitattributes"
    lines = merged.splitlines()
    assert set(ours.splitlines()) | set(theirs.splitlines()) <= set(lines)
    assert lines.index("out/") < lines.index("!kept/out/")
