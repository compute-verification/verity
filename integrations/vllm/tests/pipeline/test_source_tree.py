"""`source_tree`: a namespace package is recorded with every portion it spans, and a portion from another tree is a mismatch."""
from __future__ import annotations

import json
from pathlib import Path

from verity_vllm.pipeline import source_tree as sid


def _portion(tree: Path, root: str, name: str, module: str) -> str:
    (tree / root / name / module).mkdir(parents=True)
    (tree / root / name / module / "__init__.py").write_text("")
    (tree / "EXPORT.json").write_text(json.dumps({"commit": tree.name * 8}))
    return str(tree / root)


def test_a_namespace_with_a_portion_from_another_tree_is_a_mismatch(tmp_path):
    """Recording only the first portion passed a namespace whose second portion came from another checkout."""
    ours, other = tmp_path / "ours", tmp_path / "other"
    src, lib, foreign = _portion(ours, "src", "nsprobe", "ir"), _portion(ours, "lib", "nsprobe", "ml"), _portion(other, "src", "nsprobe", "ml")
    both = sid.resolve_module("nsprobe", path=[src, lib])
    assert both["repo_root"] == str(ours.resolve()) and [p["repo_root"] for p in both["portions"]] == [str(ours.resolve())] * 2
    assert sid.check({"nsprobe": both}, str(ours)) == []
    mixed = sid.resolve_module("nsprobe", path=[src, foreign])
    assert mixed["resolved_path"] == str((ours / "src" / "nsprobe").resolve()) and mixed["repo_root"] is None
    [m] = sid.check({"nsprobe": mixed}, str(ours))
    assert "namespace portion(s) outside the requested tree" in m["why"] and str((other / "src" / "nsprobe").resolve()) in m["why"]
