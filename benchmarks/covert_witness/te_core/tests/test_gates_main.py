"""te_core.gates_main: te_core's gate table and `main`'s are compared as code."""
import os
from pathlib import Path

import pytest

from te_core import VERSION, gates_main
from te_core.gates_main import compared


def renamed(tmp_path, edit=lambda name, source: source):
    """te_core's own files under training_slowdown's name, as `main` holds them, each through `edit`."""
    for name in (*gates_main.FILES, "__init__.py"):
        source = (gates_main.HERE / name).read_text().replace(gates_main.OURS, gates_main.THEIRS)
        (tmp_path / name).write_text(edit(name, source))
    return tmp_path


def test_the_same_statements_under_the_other_name_are_the_same(tmp_path):
    out = compared(renamed(tmp_path, lambda n, s: s.replace('"""', '"""Its own first line. ', 1) + "\n# and a comment\n"))
    assert out["same"] and out["versions"] == {"te_core": VERSION, "training_slowdown": VERSION}
    assert all(f["te_core"] != f["training_slowdown"] and not f["differ"] for f in out["files"].values())


def test_a_changed_count_or_a_missing_price_differs(tmp_path, capsys):
    out = compared(renamed(tmp_path, lambda n, s: s.replace("6524", "6525") if n == "gates.py" else s))
    assert not out["same"] and out["files"]["gates.py"]["differ"] == ["DEFINITIONS"]
    assert out["files"]["opnames.py"]["differ"] == []
    out = compared(renamed(tmp_path, lambda n, s: s.replace('    "gemm/tf32": (("HopperTf32WgmmaDot8_v2", 8),),\n', "")))
    assert out["files"]["gates.py"]["differ"] == ["PRICES"]
    assert gates_main.main([str(tmp_path)]) == 1 and '"same": false' in capsys.readouterr().out


def test_mains_gate_table_is_this_one():
    """`TRAINING_SLOWDOWN`: a checkout of `main`'s benchmarks/training_slowdown/training_slowdown."""
    if not (path := os.environ.get("TRAINING_SLOWDOWN")):
        pytest.skip("TRAINING_SLOWDOWN names no checkout of main's training_slowdown")
    out = compared(Path(path))
    assert out["same"], out
    assert out["versions"]["training_slowdown"] == VERSION
