"""`tools/verity/lean/lean_changed.py`'s trees: a package's closure and the order packages build in (what each requires
first), the tracked files a tree holds beside each package's `.lake`, and the sandbox its builds run in, which keeps the
dependencies read-only."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("check_lean_changed", ROOT / "tools" / "verity" / "lean" / "lean_changed.py")
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)


def _tree(tmp: Path, monkeypatch) -> tuple[Path, list[str]]:
    """A checkout with package `a` (a git dependency, and `p` by path), `p`, and `c`, which needs nothing."""
    root = tmp / "repo"
    files = {
        "a/lakefile.toml": "", "a/lean-toolchain": "v1\n", "a/lean-audit.json": "{}", "a/Aa.lean": "import Aa.One\n",
        "a/Aa/One.lean": "def one := 1\n",
        "a/lake-manifest.json": json.dumps({"packages": [{"name": "dep", "type": "git", "rev": "abc"},
                                                         {"name": "p", "type": "path", "dir": "../p"}]}),
        "p/lakefile.toml": "", "p/lean-toolchain": "v1\n", "p/lean-audit.json": "{}", "p/Pp.lean": "def p := 1\n",
        "c/lakefile.toml": "", "c/lean-toolchain": "v1\n", "c/lean-audit.json": "{}", "c/C.lean": "def c := 1\n",
    }
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    for m in (LC, LC.LA):
        monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda r=None: sorted(files))
    monkeypatch.setattr(LC.LA.COMMON, "cores", lambda: 2)
    return root, sorted(files)


def test_a_package_builds_after_those_it_requires_and_a_spec_after_the_packages_proving_its_guarantees(tmp_path, monkeypatch):
    """`s`'s guarantees are proved in `sp` (`proved_in`), which requires `s`: each one's closure holds the other."""
    root, _files = _tree(tmp_path, monkeypatch)
    assert LC.closure("a") == ["a", "p"]
    more = {"s/lakefile.toml": "", "s/lean-audit.json": json.dumps({"proved_in": ["../sp"]}), "s/S.lean": "",
            "sp/lakefile.toml": "", "sp/lean-audit.json": "{}", "sp/Sp.lean": "",
            "sp/lake-manifest.json": json.dumps({"packages": [{"name": "s", "type": "path", "dir": "../s"}]})}
    for f, text in more.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    assert sorted(["a", "c", "p", "s", "sp"], key=lambda p: (LC.rank(p), p)) == ["c", "p", "a", "sp", "s"]
    assert LC.closure("s") == ["s", "sp"] and LC.closure("sp") == ["sp", "s"]


def test_the_tree_holds_the_tracked_files_and_keeps_each_lake(tmp_path, monkeypatch):
    root, files = _tree(tmp_path, monkeypatch)
    tree = tmp_path / "tree"
    assert LC.sync(["a", "p"], files, tree) == 10
    assert (tree / "a" / "Aa" / "One.lean").read_text() == "def one := 1\n" and not (tree / "c").exists()
    (tree / "a" / ".lake" / "build").mkdir(parents=True)
    (tree / "a" / ".lake" / "build" / "One.olean").write_text("built")
    (tree / "a" / "Gone.lean").write_text("untracked")
    (root / "a" / "Aa" / "One.lean").write_text("def one := 2\n")
    assert LC.sync(["a", "p"], files, tree) == 1, "only what changed is written, so Lake's traces of the rest still hold"
    assert (tree / "a" / "Aa" / "One.lean").read_text() == "def one := 2\n" and not (tree / "a" / "Gone.lean").exists()
    assert (tree / "a" / ".lake" / "build" / "One.olean").read_text() == "built"
    assert LC.sync(["a", "p"], files, tree) == 0


def test_the_sandbox_keeps_the_dependencies_read_only_and_the_build_writable(tmp_path):
    """The audit's own sandbox (`tools/verity/lean/sandbox.sh`): what a build may write is its `.lake` but not its
    `.lake/packages`, shown on a file there that is a hard link to one kept outside the tree."""
    if not LC.sandboxed():
        pytest.skip("no unprivileged user namespaces here (`unshare`)")
    pkg = tmp_path / "tree" / "a"
    store = tmp_path / "deps" / "k"
    (store / "dep").mkdir(parents=True)
    (store / "dep" / "Dep.olean").write_text("kept")
    (pkg / ".lake" / "packages" / "dep").mkdir(parents=True)
    os.link(store / "dep" / "Dep.olean", pkg / ".lake" / "packages" / "dep" / "Dep.olean")
    lake = tmp_path / "lake"
    lake.write_text("#!/usr/bin/env bash\nmkdir -p .lake/build && echo built > .lake/build/A.olean\n"
                    "echo tampered > .lake/packages/dep/Dep.olean 2>/dev/null && echo WROTE-DEPS\nexit 0\n")
    lake.chmod(0o755)
    rc, text = LC.build("a", tmp_path / "tree", str(lake))
    assert rc == 0 and "WROTE-DEPS" not in text, text
    assert (pkg / ".lake" / "build" / "A.olean").read_text() == "built\n"
    assert (store / "dep" / "Dep.olean").read_text() == "kept"
