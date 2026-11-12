"""`../pod/lake_closure.py` on toy packages: what a package's closure is, what its key covers, and where a copy puts each directory."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

POD = Path(__file__).resolve().parents[2] / "pod"
spec = importlib.util.spec_from_file_location("lake_closure_for_tests", POD / "lake_closure.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def tree(root: Path) -> Path:
    """`service/flock`, which requires `primitives/base` and `primitives/circuits` by path (the second through the first's
    manifest too), beside `primitives/other`, which nothing requires."""
    for d, entries in (("service/flock", [("base", "../../primitives/base"), ("circuits", "../../primitives/circuits")]),
                       ("primitives/circuits", [("base", "../base")]), ("primitives/base", []), ("primitives/other", [])):
        (root / d / ".lake").mkdir(parents=True)
        (root / d / ".lake" / "built").write_text("x")
        (root / d / "A.lean").write_text(f"-- {d}\n")
        (root / d / "lake-manifest.json").write_text(json.dumps(
            {"packages": [{"name": n, "type": "path", "dir": p} for n, p in entries]
             + [{"name": "mathlib", "type": "git", "url": "u", "rev": "r"}]}))
    return root / "service" / "flock"


def test_the_closure_is_the_package_and_its_path_dependencies(tmp_path):
    package = tree(tmp_path)
    assert C.closure(package) == [package, tmp_path / "primitives/base", tmp_path / "primitives/circuits"]
    assert C.closure(tmp_path / "primitives/base") == [tmp_path / "primitives/base"]


def test_the_key_changes_with_a_required_package_and_not_with_a_build(tmp_path):
    package = tree(tmp_path)
    before = C.key(package, [])
    (package / ".lake" / "more").write_text("y")
    (tmp_path / "primitives/other/A.lean").write_text("-- changed\n")
    assert C.key(package, []) == before
    (tmp_path / "primitives/base/A.lean").write_text("-- changed\n")
    changed = C.key(package, [])
    assert changed != before and len(changed) == 16
    extra = tmp_path / "Extra.lean"
    extra.write_text("-- extra\n")
    assert C.key(package, [extra]) != changed


def test_a_copy_keeps_each_package_at_its_relative_place(tmp_path):
    package = tree(tmp_path / "src")
    at = C.copy(package, tmp_path / "dest")
    assert at == tmp_path / "dest/service/flock"
    got = sorted(p.relative_to(tmp_path / "dest").as_posix() for p in (tmp_path / "dest").rglob("*") if p.is_file())
    assert got == ["primitives/base/A.lean", "primitives/base/lake-manifest.json", "primitives/circuits/A.lean",
                   "primitives/circuits/lake-manifest.json", "service/flock/A.lean", "service/flock/lake-manifest.json"]
    assert (at / "../../primitives/base/A.lean").resolve().read_text() == "-- primitives/base\n"


def test_a_package_that_requires_nothing_copies_to_its_own_name(tmp_path):
    tree(tmp_path / "src")
    at = C.copy(tmp_path / "src/primitives/base", tmp_path / "dest")
    assert at == tmp_path / "dest/base" and (at / "A.lean").is_file() and not (at / ".lake").exists()
