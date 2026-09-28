"""`suites.py` in a tree without git, as `research run --source` ships one: suites are found, and keyed from their files."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _suites(monkeypatch, root: Path):
    spec = importlib.util.spec_from_file_location("suites_under_test", HERE / "suites.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)           # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(mod, "GIT", False)
    return mod


def test_a_shipped_tree_is_discovered_and_keyed_from_its_files(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pkg"]\n')
    (tmp_path / "pkg" / "tests").mkdir(parents=True)
    (tmp_path / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (tmp_path / "pkg" / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"1")
    S = _suites(monkeypatch, tmp_path)

    [suite] = S.discover()
    assert (suite.name, suite.path) == ("pkg", "pkg")
    assert S.metadata() == ["pkg/pyproject.toml", "pyproject.toml", "uv.lock"]
    before = S.object_ids(None, ["pkg", "uv.lock"], S.digests())
    assert before["uv.lock"] == "absent"

    (tmp_path / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"2")      # what a machine adds is not source
    assert S.object_ids(None, ["pkg", "uv.lock"], S.digests()) == before
    (tmp_path / "pkg" / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    assert S.object_ids(None, ["pkg"], S.digests())["pkg"] != before["pkg"]


def test_a_shipped_tree_reports_writes_as_git_would(tmp_path, monkeypatch):
    (tmp_path / ".gitignore").write_text("**/target/\n")
    (tmp_path / "pkg" / "target").mkdir(parents=True)
    (tmp_path / "pkg" / "target" / "kept.bin").write_bytes(b"a tracked file the rules cover")
    (tmp_path / "pkg" / "a.py").write_text("A = 1\n")
    S = _suites(monkeypatch, tmp_path)
    before = S.digests()
    assert "pkg/target/kept.bin" in before

    (tmp_path / "pkg" / "target" / "build.o").write_bytes(b"x")        # a build output: ignored, as in a checkout
    assert S.changed(before) == []
    (tmp_path / "pkg" / "a.py").write_text("A = 2\n")
    (tmp_path / "pkg" / "new.json").write_text("{}\n")
    (tmp_path / "pkg" / "target" / "kept.bin").unlink()
    assert S.changed(before) == ["pkg/a.py", "pkg/new.json", "pkg/target/kept.bin"]
