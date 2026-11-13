"""h1v3_ref.py under a pinned tree of an older layout: it runs that tree's h1tile wherever the tree keeps it, and a tree
with none, or one whose h1tile fails on an import of its own, fails saying which."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "h1v3_ref.py"
STUB = "def wrap_bits(b, c, u):\n    return (b + c + u) & 1\n"


def older() -> list[tuple[str, ...]]:
    """`OLDER` as the script writes it: the older layouts are its data, read here rather than written a second time."""
    for node in ast.parse(SCRIPT.read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["OLDER"]:
            return list(ast.literal_eval(node.value))
    raise AssertionError("h1v3_ref.py assigns no OLDER")


def tree_with(root: Path, parts: tuple[str, ...] | None, source: str = STUB) -> Path:
    """A tree whose `verity/` holds h1tile at ``parts`` alone (none: no h1tile), every directory on the way a package."""
    (root / "verity").mkdir(parents=True)
    (root / "verity/__init__.py").write_text("")
    if parts:
        d = root
        for p in parts[:-1]:
            d = d / p
            d.mkdir(exist_ok=True)
            (d / "__init__.py").write_text("")
        (d / f"{parts[-1]}.py").write_text(source)
    return root


def run(tree: Path, tmp_path: Path) -> tuple[subprocess.CompletedProcess, Path]:
    src, dst = tmp_path / "in.json", tmp_path / "out.json"
    src.write_text(json.dumps({"wrap_table": True}))
    # the tree first, as a launcher puts it, then what this process imports from, so the script starts as the suite does
    path = os.pathsep.join([str(tree), *filter(None, sys.path)])
    r = subprocess.run([sys.executable, str(SCRIPT), str(src), str(dst)], env={**os.environ, "PYTHONPATH": path},
                       capture_output=True, text=True)
    return r, dst


def test_the_script_names_two_older_layouts_as_parts() -> None:
    layouts = older()
    assert len(layouts) == 2 and len(set(layouts)) == 2
    assert all(p[0] == "verity" and p[-1] == "h1tile" and all(w.isidentifier() for w in p) for p in layouts)


@pytest.mark.parametrize("which", [0, 1])
def test_a_tree_of_an_older_layout_runs_its_own_h1tile(tmp_path: Path, which: int) -> None:
    parts = older()[which]
    tree = tree_with(tmp_path / "tree", parts)
    r, dst = run(tree, tmp_path)
    assert r.returncode == 0, r.stderr
    out = json.loads(dst.read_text())
    assert Path(out["h1tile"]) == tree.joinpath(*parts[:-1], "h1tile.py") and out["peeled"] is False
    assert len(out["wrap"]) == 2 * 256 * 256 and out["wrap"][:3] == [0, 1, 0]


def test_a_tree_with_no_h1tile_fails_with_every_name_it_tried(tmp_path: Path) -> None:
    r, dst = run(tree_with(tmp_path / "tree", None), tmp_path)
    assert r.returncode != 0 and not dst.exists()
    last = r.stderr.strip().splitlines()[-1]
    assert "no h1tile under the tree on PYTHONPATH" in last
    assert all(".".join(p[:2]) in last for p in older())


def test_an_older_h1tile_that_fails_on_its_own_import_is_not_passed_over(tmp_path: Path) -> None:
    tree = tree_with(tmp_path / "tree", older()[0], "import a_module_no_tree_has\n")
    r, dst = run(tree, tmp_path)
    assert r.returncode != 0 and not dst.exists()
    assert "No module named 'a_module_no_tree_has'" in r.stderr.strip().splitlines()[-1]
