"""`anchors.py`: a move that leaves a `Path(__file__)` join reaching somewhere else is listed, one that keeps every target is
not."""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("anchors_under_test", HERE / "anchors.py")
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)

LIB = '''from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "table.json"
BUILD = ROOT / "pkg" / "target" / "release"
HERE = Path(__file__).parent


def fixture(name):
    return HERE / "fixtures" / name
'''


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    for rel, text in {"pkg/src/lib.py": LIB, "pkg/src/fixtures/a.json": "{}", "pkg/Cargo.toml": "", "data/table.json": "{}"}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def _move(root: Path, *renames: tuple[str, str], edit: dict[str, str] | None = None) -> list[str]:
    base = _git(root, "rev-parse", "HEAD")
    for old, new in renames:
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        _git(root, "mv", old, new)
    for rel, text in (edit or {}).items():
        (root / rel).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "move")
    return A.run(base, "HEAD", cwd=root)


def test_the_evaluator_reads_each_anchored_form():
    names: dict[str, str] = {}

    def at(src: str) -> str | None:
        return A.anchor(ast.parse(src, mode="eval").body, "a/b/c.py", names)

    assert at("Path(__file__)") == "a/b/c.py"
    assert at("Path(__file__).resolve().parents[1]") == "a"
    assert at("Path(__file__).parents[2]") == ""
    assert at("Path(__file__).parents[3]") == A.OUTSIDE
    assert at("Path(__file__).parent / '..' / 'x' / 'y.json'") == "a/x/y.json"
    assert at("Path(__file__).parent.joinpath('d', 'e')") == "a/b/d/e"
    assert at("Path(__file__).with_name('k.toml')") == "a/b/k.toml"
    assert at("Path(__file__).parents[2] / 'tools/x'") == "tools/x"
    assert at("Path(p)") is None and at("ROOT / 'x'") is None
    names["ROOT"] = ""
    assert at("ROOT / 'x'") == "x"
    assert A.joins("a/b/c.py", LIB) == [(2, ""), (3, "data/table.json"), (4, "pkg/target/release"), (5, "a/b"),
                                        (9, "a/b/fixtures")]
    assert A.joins("a/b/c.py", "X = 1\n") == []


def test_a_move_that_keeps_depth_and_neighbours_passes_once_its_paths_are_rewritten(tmp_path):
    root = _repo(tmp_path)
    assert _move(root, ("pkg", "core"), edit={"core/src/lib.py": LIB.replace('"pkg"', '"core"')}) == []


def test_an_untracked_tail_follows_its_tracked_prefix(tmp_path):
    root = _repo(tmp_path)
    assert _move(root, ("pkg", "core")) == ["pkg/src/lib.py:4 (now core/src/lib.py): reached pkg/target/release, which went "
                                            "to core/target/release; no join there reaches it"]


def test_a_file_moved_to_another_depth_is_listed(tmp_path):
    root = _repo(tmp_path)
    got = _move(root, ("pkg/src/lib.py", "pkg/lib.py"), ("pkg/src/fixtures/a.json", "pkg/fixtures/a.json"))
    assert got == ["pkg/src/lib.py:2 (now pkg/lib.py): reached ., which went to .; no join there reaches it",
                   "pkg/src/lib.py:3 (now pkg/lib.py): reached data/table.json, which went to data/table.json; no join there "
                   "reaches it",
                   "pkg/src/lib.py:4 (now pkg/lib.py): reached pkg/target/release, which went to pkg/target/release; no join "
                   "there reaches it"]


def test_a_rewritten_depth_passes_and_a_file_left_behind_is_listed(tmp_path):
    root = _repo(tmp_path)
    moved = LIB.replace("parents[2]", "parents[1]")
    assert _move(root, ("pkg/src/lib.py", "pkg/lib.py"), ("pkg/src/fixtures/a.json", "pkg/fixtures/a.json"),
                 edit={"pkg/lib.py": moved}) == []
    got = _move(root, ("pkg/lib.py", "pkg/core/lib.py"), edit={"pkg/core/lib.py": moved.replace("parents[1]", "parents[2]")})
    assert got == ["pkg/lib.py:9 (now pkg/core/lib.py): reached pkg/fixtures, which went to pkg/fixtures; no join there "
                   "reaches it"]


def test_a_directory_reference_follows_its_files_and_a_split_one_may_go_either_way(tmp_path):
    root = _repo(tmp_path)
    lib = LIB.replace('ROOT / "data" / "table.json"', 'ROOT / "data"')
    (root / "pkg/src/lib.py").write_text(lib)
    (root / "data/other.json").write_text("{}")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "a directory")
    assert _move(root, ("data", "tables"), edit={"pkg/src/lib.py": lib.replace('"data"', '"tables"')}) == []
    assert _move(root, ("tables/table.json", "t1/table.json"), ("tables/other.json", "t2/other.json"),
                 edit={"pkg/src/lib.py": lib.replace('"data"', '"t2"')}) == []
    assert _move(root, ("t2/other.json", "t3/other.json")) == [
        "pkg/src/lib.py:3 (now pkg/src/lib.py): reached t2, which went to t3; no join there reaches it"]


SH = '''#!/usr/bin/env bash
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)   # pkg/src -> the root
python3 "$(dirname "$0")/lib.py"
cat "${ROOT}/data/table.json" "$HERE/fixtures/a.json"  # "$ROOT/not/read"
cd "$ROOT" && git -C "$HERE" status
'''


def test_a_shell_script_reaches_from_its_directory(tmp_path):
    assert A.shell_joins("pkg/src/run.sh", SH) == [(3, "pkg/src/lib.py"), (4, "data/table.json"),
                                                  (4, "pkg/src/fixtures/a.json"), (5, ""), (5, "pkg/src")]
    assert A.shell_joins("d/x.sh", 'here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)\n"$here/a"\n') == [(2, "a")]
    root = _repo(tmp_path)
    (root / "pkg/src/run.sh").write_text(SH)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "a script")
    assert _move(root, ("pkg/src", "pkg/bin")) == []
    assert _move(root, ("pkg/bin/run.sh", "run.sh")) == [
        "pkg/bin/run.sh:3 (now run.sh): reached pkg/bin/lib.py, which went to pkg/bin/lib.py; no join there reaches it",
        "pkg/bin/run.sh:4 (now run.sh): reached data/table.json, which went to data/table.json; no join there reaches it",
        "pkg/bin/run.sh:4 (now run.sh): reached pkg/bin/fixtures/a.json, which went to pkg/bin/fixtures/a.json; no join there "
        "reaches it",
        "pkg/bin/run.sh:5 (now run.sh): reached pkg/bin, which went to pkg/bin; no join there reaches it"]


def test_the_cli_exits_one_on_a_listed_join(tmp_path, capsys, monkeypatch):
    root = _repo(tmp_path)
    _move(root, ("pkg/src/lib.py", "pkg/lib.py"))
    monkeypatch.setattr(A, "ROOT", root)
    assert A.main(["HEAD~1", "HEAD"]) == 1
    assert "reached data/table.json" in capsys.readouterr().out
    assert A.main(["HEAD", "HEAD"]) == 0
    assert A.main(["nope", "HEAD"]) == 2
