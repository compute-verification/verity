"""`layout_case.py`: the paths of a moved tree that differ only by case, on toy trees and toy maps."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MOVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOVE))
spec = importlib.util.spec_from_file_location("layout_case_for_tests", MOVE / "layout_case.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def layout(paths: dict[str, str]):
    return C.L.Layout({"roots": {"old": {}, "new": {}}, "paths": paths, "modules": {}})


def test_two_directories_that_differ_by_case_are_one_finding():
    paths = ["v/VStar/Run.lean", "v/VStar/Trace.lean", "v/vstar/lakefile.toml", "v/vstar/tests/test_x.py", "v/other.py"]
    assert C.case_sets(paths) == [["v/VStar", "v/vstar"]]


def test_files_that_differ_by_case_and_none_that_do_not():
    assert C.case_sets(["a/README.md", "a/Readme.md", "a/b.py"]) == [["a/README.md", "a/Readme.md"]]
    assert C.case_sets(["a/x.py", "b/X.py", "a/y/x.py"]) == []
    assert C.case_sets(["a/Straße.md", "a/STRASSE.md"]) == [["a/STRASSE.md", "a/Straße.md"]]


def test_a_file_and_a_directory_collide_too():
    assert C.case_sets(["a/Lean", "a/lean/x.lean"]) == [["a/Lean", "a/lean"]]


def test_the_maps_run_in_order_each_on_the_tree_before_it():
    tracked = ["core/vstar/main.lean", "core/flock/VStar/Run.lean", "core/flock/lakefile.toml", "tools/t.py"]
    first = layout({"core/flock/": "flock/verifier/"})
    second = layout({"core/vstar/": "flock/verifier/vstar/"})
    final = C.moved(tracked, [first, second])
    assert final == {"core/vstar/main.lean": "flock/verifier/vstar/main.lean",
                     "core/flock/VStar/Run.lean": "flock/verifier/VStar/Run.lean",
                     "core/flock/lakefile.toml": "flock/verifier/lakefile.toml", "tools/t.py": "tools/t.py"}
    assert C.case_sets(final.values()) == [["flock/verifier/VStar", "flock/verifier/vstar"]]
    renamed = C.moved(tracked, [first, layout({"core/vstar/": "flock/verifier/vstar_cli/"})])
    assert C.case_sets(renamed.values()) == []
    chained = C.moved(["a/x.py"], [layout({"a/": "b/"}), layout({"b/": "c/"})])
    assert chained == {"a/x.py": "c/x.py"}
