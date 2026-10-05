"""protected.py (protected): the diff of a real git repository laid out like this one, where git names every path from the
repository root and the globs name it from the package root."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from verity_vllm.properties import protected

PKG = "integrations/vllm"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                           *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def _commit(root: Path, files: dict[str, str], message: str) -> str:
    _write(root, files)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    base = _commit(root, {f"{PKG}/verity_vllm/check/gates.py": "GATES = 1\n",
                          f"{PKG}/manifests/checkpoints.json": json.dumps({"checkpoints": [{"repo": "a", "revision": "0" * 40}]}),
                          "verity/README.md": "core\n"}, "base")
    return root, base


def test_a_root_relative_diff_touching_a_protected_file_is_not_sealed(tmp_path):
    root, base = _repo(tmp_path)
    head = _commit(root, {f"{PKG}/verity_vllm/check/gates.py": "GATES = 2\n", "verity/README.md": "core, edited\n"}, "lane")
    rep = protected.provenance(base, head, cwd=root / PKG)
    assert not rep["sealed"] and not rep["ok"]
    assert rep["protected_files_modified"] == [{"status": "M", "path": "verity_vllm/check/gates.py", "rule": "protected"}]
    assert {c["path"] for c in rep["changed"]} == {"verity_vllm/check/gates.py", "../../verity/README.md"}


def test_lane_writable_paths_and_paths_outside_the_package_seal_clean(tmp_path):
    root, base = _repo(tmp_path)
    head = _commit(root, {f"{PKG}/verity_vllm/program/registry/quarantine/q.py": "Q = 1\n",
                          f"{PKG}/tests/observe/test_new.py": "\n", "verity/README.md": "core, edited\n"}, "lane")
    rep = protected.provenance(base, head, cwd=root / PKG)
    assert rep["sealed"] and rep["protected_files_modified"] == [] and rep["n_changed"] == 3


def test_a_changed_frozen_manifest_entry_is_not_sealed(tmp_path):
    root, base = _repo(tmp_path)
    head = _commit(root, {f"{PKG}/manifests/checkpoints.json": json.dumps({"checkpoints": [{"repo": "b", "revision": "0" * 40}]})},
                   "lane")
    rep = protected.provenance(base, head, cwd=root / PKG)
    assert not rep["sealed"]
    assert [p["rule"] for p in rep["protected_files_modified"]] == ["frozen manifest entry: a@00000000 (#0): changed"]


def test_renaming_a_protected_file_into_a_lane_writable_tree_is_not_sealed(tmp_path):
    root, base = _repo(tmp_path)
    (root / PKG / "verity_vllm/program/registry/quarantine").mkdir(parents=True)
    _git(root, "mv", f"{PKG}/verity_vllm/check/gates.py", f"{PKG}/verity_vllm/program/registry/quarantine/gates.py")
    head = _commit(root, {}, "lane")
    rep = protected.provenance(base, head, cwd=root / PKG)
    assert not rep["sealed"]
    assert rep["protected_files_modified"] == [{"status": "D", "path": "verity_vllm/check/gates.py", "rule": "protected"}]
