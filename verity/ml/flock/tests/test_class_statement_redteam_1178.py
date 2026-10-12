"""Red-team 1178: `_source_digest` over a namespace package's portions must never give one digest to two sources Python imports
differently (a false hit reuses a staged statement built by other code)."""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verity_flock.class_statement")
from verity_flock import class_statement as CS  # noqa: E402


def _digest(monkeypatch, locs):
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: SimpleNamespace(submodule_search_locations=locs)
                        if name == "verity" else find_spec(name, *a))
    return CS._source_digest()


def _imports(tmp_path, order, stmt):
    pp = ":".join(str(tmp_path / o) for o in order)
    r = subprocess.run([sys.executable, "-B", "-c", stmt], env={"PYTHONPATH": pp}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_the_digest_is_imported_from_this_tree():
    assert Path(CS.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[4])


def test_a_portion_holding_only_a_namespace_subpackage_is_hashed(monkeypatch, tmp_path):
    """Portion b holds only ns/sub/mod.py, no __init__.py anywhere: pkgutil.iter_modules lists nothing in it, but Python imports
    ns.sub.mod from it.  Editing it must move the digest."""
    (tmp_path / "a/ns").mkdir(parents=True)
    (tmp_path / "a/ns/m.py").write_text("X = 1\n")
    (tmp_path / "b/ns/sub").mkdir(parents=True)
    (tmp_path / "b/ns/sub/mod.py").write_text("Y = 1\n")
    locs = [str(tmp_path / "a/ns"), str(tmp_path / "b/ns")]
    assert _imports(tmp_path, ("a", "b"), "import ns.sub.mod as m; print(m.Y)") == "1"
    d0 = _digest(monkeypatch, locs)
    (tmp_path / "b/ns/sub/mod.py").write_text("Y = 2  # edited\n")
    assert _imports(tmp_path, ("a", "b"), "import ns.sub.mod as m; print(m.Y)") == "2"
    assert _digest(monkeypatch, locs) != d0


def test_portions_that_overlap_never_share_a_digest_across_orders(monkeypatch, tmp_path):
    """Portions a and b both hold ns/x.py: sys.path order picks which one Python imports.  The two orders must not get one digest;
    refusing the overlap is the fail-closed answer."""
    for p, v in (("a", 1), ("b", 2)):
        (tmp_path / p / "ns").mkdir(parents=True)
        (tmp_path / p / "ns" / "x.py").write_text(f"V = {v}\n")
    imported = {o: _imports(tmp_path, o, "import ns.x; print(ns.x.V)") for o in (("a", "b"), ("b", "a"))}
    assert imported[("a", "b")] != imported[("b", "a")]
    digests = {}
    for o in imported:
        try:
            digests[o] = _digest(monkeypatch, [str(tmp_path / p / "ns") for p in o])
        except ValueError:
            return
    assert digests[("a", "b")] != digests[("b", "a")]


def test_todays_digest_is_the_one_portion_walk():
    """On today's layout every package has one portion, and the digest is the old per-portion walk, byte for byte, less the
    files under a name holding a dot."""
    h = hashlib.sha256()
    for name in ("verity", "verity.ml.circuits", "verity.ml.kernels", "verity.core", "verity_flock", "circuit_check", "verity_vllm"):
        spec = importlib.util.find_spec(name)
        assert spec is None or len(spec.submodule_search_locations or []) <= 1, (name, spec.submodule_search_locations)
        for root in (spec.submodule_search_locations or []) if spec else []:
            for f in sorted(Path(root).rglob("*.py")):
                if not any("." in p for p in f.relative_to(root).with_suffix("").parts):
                    h.update(f"{name}/{f.relative_to(root).as_posix()}\0".encode() + hashlib.sha256(f.read_bytes()).digest())
    assert h.hexdigest() == CS._source_digest()


def test_a_name_holding_a_dot_is_no_module_and_is_not_hashed(monkeypatch, tmp_path):
    """A built Lake package keeps Mathlib's Python scripts under `.lake/`: Python imports nothing from a directory or file whose
    name holds a dot, so editing one leaves the digest alone, while editing an importable module beside it still moves it."""
    for rel in ("ns/m.py", "ns/lean/.lake/packages/mathlib/scripts/lint.py", "ns/lean/a.b.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("X = 1\n")
    locs = [str(tmp_path / "ns")]
    d0 = _digest(monkeypatch, locs)
    for rel in ("ns/lean/.lake/packages/mathlib/scripts/lint.py", "ns/lean/a.b.py"):
        (tmp_path / rel).write_text("X = 2  # edited\n")
    assert _digest(monkeypatch, locs) == d0
    (tmp_path / "ns/m.py").write_text("X = 2  # edited\n")
    assert _digest(monkeypatch, locs) != d0
