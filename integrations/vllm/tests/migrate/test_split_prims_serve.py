"""`tools/migrate/split_prims_serve.py`: the split on a synthetic family (what it moves, how it rewrites importers, that a rerun
changes nothing, and that it refuses by name), and on this tree (the split is applied and every importer settled)."""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
SCRIPT = ROOT / "tools" / "migrate" / "split_prims_serve.py"


def _load():
    spec = importlib.util.spec_from_file_location("split_prims_serve", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # its dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


S = _load()
REG = S.REG

DEFS = '''"""A Definition module."""
from __future__ import annotations

import math
from functools import lru_cache

X = 1


def body(a):
    return math.floor(a) + X


# the twin, which leaves
@lru_cache(maxsize=None)
def twin():
    return body(2) + math.ceil(0.5)


TWIN_CONST = 2  # a trailing comment that
# goes on below


__all__ = ["X", "body", "twin", "TWIN_CONST"]
'''

VOCAB = '''"""Not a Definition module: it leaves whole."""
from verity_vllm.program.registry import defs as D

THING = D.X
'''

USER = '''"""An importer."""
from verity_vllm.program.registry import defs as D
from verity_vllm.program.registry.defs import X, twin
from verity_vllm.program.registry import vocab_mod
from verity_vllm.program.registry.vocab_mod import THING

NOTE = "data a digest reads: registry.vocab_mod"
PATH = "verity_vllm.program.registry.vocab_mod"


def f():
    """Reads `verity_vllm.program.registry.defs.twin` through registry.vocab_mod."""
    from verity_vllm.program.registry import defs as D2
    return D2.twin() + D2.X  # see registry.vocab_mod


def g():
    from verity_vllm.program.registry import defs as D3
    return D3.TWIN_CONST


VALUES = (D.X, D.twin(), D.TWIN_CONST, X, twin(), vocab_mod.THING, THING, f(), g())

PROGRAM = """
from verity_vllm.program.registry import defs, vocab_mod as V
from verity_vllm.program.registry.defs import X, twin
print(V.THING, twin(), X)
"""
'''

INSIDE = '''from . import defs
from .vocab_mod import THING

Y = defs.twin() + THING
'''


@pytest.fixture
def family(tmp_path, monkeypatch):
    """A tree with one Definition module (`defs`: `twin` and `TWIN_CONST` leave for `twins.defs`), one module that leaves
    whole (`vocab_mod` -> `twins.vocab_mod`), and importers outside and inside the registry."""
    pkg = tmp_path / "integrations/vllm/verity_vllm"
    files = {
        "__init__.py": "", "program/__init__.py": "", "program/registry/__init__.py": '"""The registry."""\n',
        "program/registry/defs.py": DEFS, "program/registry/vocab_mod.py": VOCAB, "program/registry/inside.py": INSIDE,
        "user.py": USER,
    }
    for rel, text in files.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(text)
    (tmp_path / "README.md").write_text("See `registry.vocab_mod` and `verity_vllm.program.registry.defs.twin`.\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(S, "PACKAGES", {"verity_vllm.program.twins": '"""Twins."""\n'})
    monkeypatch.setattr(S, "SPLITS", (S.Split(f"{REG}.defs", "verity_vllm.program.twins.defs", ("twin", "TWIN_CONST"),
                                              "defs_twins", '"""The twins of defs."""\n'),))
    monkeypatch.setattr(S, "MOVES", (S.Move(f"{REG}.vocab_mod", "verity_vllm.program.twins.vocab_mod",
                                            ("registry.vocab_mod", "twins.vocab_mod")),))
    monkeypatch.setattr(S, "DEFINITION_MODULES", (f"{REG}.defs", f"{REG}.inside", REG))
    monkeypatch.setattr(S, "DEFINITION_IMPORTS", (f"{REG}.defs", f"{REG}.inside", REG, "verity_vllm.program.twins.vocab_mod",
                                                  "verity_vllm.program.twins.defs"))
    monkeypatch.setattr(S, "KEEP", {})
    return tmp_path


def _values(root: Path) -> str:
    """`user.VALUES`, `inside.Y` and what `user.PROGRAM` prints, computed by a Python whose `verity_vllm` is the synthetic
    tree's."""
    env = dict(os.environ, PYTHONPATH=str(root / "integrations/vllm"))
    code = "import verity_vllm.user as u, verity_vllm.program.registry.inside as i; print(u.VALUES, i.Y); exec(u.PROGRAM)"
    return subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True, cwd=root).stdout


def _read(root: Path, rel: str) -> str:
    return (root / "integrations/vllm/verity_vllm" / rel).read_text()


def test_split_moves_the_names_and_every_importer_follows(family):
    before = _values(family)
    log: list[str] = []
    changed = S.run(family, log=log.append)
    assert _values(family) == before
    assert "integrations/vllm/verity_vllm/program/registry/vocab_mod.py" in changed
    assert not (family / "integrations/vllm/verity_vllm/program/registry/vocab_mod.py").exists()
    defs, twins = _read(family, "program/registry/defs.py"), _read(family, "program/twins/defs.py")
    assert "def twin" not in defs and "TWIN_CONST" not in defs and "lru_cache" not in defs and "import math" in defs
    assert '__all__ = ["X", "body"]' in defs
    assert twins.startswith('"""The twins of defs."""\n')
    assert "# the twin, which leaves\n@lru_cache(maxsize=None)\ndef twin" in twins
    assert "TWIN_CONST = 2  # a trailing comment that\n# goes on below\n" in twins
    assert "from verity_vllm.program.registry.defs import body" in twins and "from functools import lru_cache" in twins
    user = _read(family, "user.py")
    assert "from verity_vllm.program.twins.defs import twin\nfrom verity_vllm.program.registry.defs import X" in user
    assert "from verity_vllm.program.twins import vocab_mod" in user
    assert "from verity_vllm.program.twins.vocab_mod import THING" in user
    assert "D.X, defs_twins.twin(), defs_twins.TWIN_CONST" in user
    assert "from verity_vllm.program.registry import defs as D\nfrom verity_vllm.program.twins import defs as defs_twins" in user
    # a function's alias read only for moved names is replaced, one read for anything else keeps it
    assert "    from verity_vllm.program.twins import defs as defs_twins\n    return defs_twins.TWIN_CONST" in user
    assert ("    from verity_vllm.program.registry import defs as D2\n    from verity_vllm.program.twins import defs as defs_twins\n"
            "    return defs_twins.twin() + D2.X  # see twins.vocab_mod") in user
    assert "`verity_vllm.program.twins.defs.twin` through twins.vocab_mod" in user
    assert 'PATH = "verity_vllm.program.twins.vocab_mod"' in user
    assert ('PROGRAM = """\nfrom verity_vllm.program.twins import vocab_mod as V\nfrom verity_vllm.program.registry import defs\n'
            "from verity_vllm.program.twins.defs import twin\nfrom verity_vllm.program.registry.defs import X\n") in user
    inside = _read(family, "program/registry/inside.py")
    assert "from verity_vllm.program.twins.vocab_mod import THING" in inside and "defs_twins.twin()" in inside
    assert (family / "README.md").read_text() == "See `twins.vocab_mod` and `verity_vllm.program.twins.defs.twin`.\n"


def test_a_string_that_is_data_is_left_and_reported(family):
    log: list[str] = []
    S.run(family, log=log.append)
    assert 'NOTE = "data a digest reads: registry.vocab_mod"' in _read(family, "user.py")
    assert any(line.startswith("left") and "user.py:" in line and "data a digest reads" in line for line in log)


def test_a_rerun_changes_nothing_and_check_writes_nothing(family):
    assert S.run(family, check=True, log=lambda _m: None)
    assert (family / "integrations/vllm/verity_vllm/program/registry/vocab_mod.py").exists()
    first = S.run(family, log=lambda _m: None)
    assert first
    log: list[str] = []
    assert S.run(family, log=log.append) == []
    assert all(line.startswith(("already", "left")) for line in log)


def test_refuses_what_stays_reading_a_moved_name(family):
    p = family / "integrations/vllm/verity_vllm/program/registry/defs.py"
    p.write_text(p.read_text().replace("__all__", "Z = twin()\n\n\n__all__"))
    with pytest.raises(S.Refused, match=r"still reads \['twin'\]"):
        S.run(family, log=lambda _m: None)


def test_refuses_a_name_the_module_no_longer_defines(family):
    p = family / "integrations/vllm/verity_vllm/program/registry/defs.py"
    p.write_text(p.read_text().replace("TWIN_CONST = 2", "OTHER = 2"))
    with pytest.raises(S.Refused, match=r"does not define \['TWIN_CONST'\]"):
        S.run(family, log=lambda _m: None)


def test_refuses_a_definition_module_importing_outside_the_definitions(family, monkeypatch):
    monkeypatch.setattr(S, "DEFINITION_IMPORTS", (f"{REG}.defs", f"{REG}.inside", REG))
    with pytest.raises(S.Refused, match=r"inside line 2: imports verity_vllm.program.twins.vocab_mod"):
        S.run(family, log=lambda _m: None)


def test_refuses_a_program_in_a_string_whose_source_is_not_its_value(family):
    value = "from verity_vllm.program.registry import vocab_mod\n"
    assert S._embedded_program("x.py", '"""' + value + '"""', value) == '"""from verity_vllm.program.twins import vocab_mod\n"""'
    with pytest.raises(S.Refused, match="source text is not its value"):
        S._embedded_program("x.py", repr(value), value)


def test_refuses_an_alias_the_importer_binds_already(family):
    p = family / "integrations/vllm/verity_vllm/user.py"
    p.write_text(p.read_text().replace("NOTE =", "defs_twins = 0\nNOTE ="))
    with pytest.raises(S.Refused, match="binds `defs_twins` already"):
        S.run(family, log=lambda _m: None)


def test_the_allowlist_entries_follow_their_symbols():
    sp = next(s for s in S.SPLITS if s.src.endswith(".b1"))
    moved = '  {"file": "verity_vllm/program/registry/b1.py", "kind": "table", "symbol": "SERVE_FAMILIES", "pattern": "x"},\n'
    kept = '  {"file": "verity_vllm/program/registry/b1.py", "kind": "model-name", "symbol": "<module>", "pattern": "y"},\n'
    out = S.rewrite_allowlist(moved + kept)
    assert out == moved.replace("registry/b1.py", "vocabulary/serve.py") + kept
    assert "SERVE_FAMILIES" in sp.names


def test_the_ratchet_entries_follow_what_moved(family):
    """A symbol's entry follows the symbol, an import edge follows the import, a module's size comes down to its size now
    (and goes under the limit), and two entries that land on one key add their counts."""
    S.run(family, log=lambda _m: None)
    tree = S.Tree(family)
    entries = [
        {"file": "verity_vllm/program/registry/defs.py", "kind": "doc-round", "symbol": "twin", "detail": "R1N"},
        {"file": "verity_vllm/program/twins/defs.py", "kind": "doc-round", "symbol": "twin", "detail": "R1N", "count": 2},
        {"file": "verity_vllm/program/registry/defs.py", "kind": "doc-round", "symbol": "body", "detail": "R1N"},
        {"file": "verity_vllm/program/registry/defs.py", "kind": "layer", "symbol": "<module>",
         "detail": "program.registry.defs -> functools"},
        {"file": "verity_vllm/program/registry/defs.py", "kind": "layer", "symbol": "<module>", "detail": "program.registry.defs -> math"},
        {"file": "verity_vllm/program/registry/defs.py", "kind": "module", "symbol": "<module>", "detail": "lines", "count": 900},
    ]
    text = '{\n  "rule": "x",\n  "entries": [\n' + ",\n".join("    " + S.json.dumps(e) for e in entries) + "\n  ]\n}\n"
    out = S.json.loads(S.rewrite_ratchet(tree, text))["entries"]
    assert out == [
        {"file": "verity_vllm/program/twins/defs.py", "kind": "doc-round", "symbol": "twin", "detail": "R1N", "count": 3},
        entries[2],
        {"file": "verity_vllm/program/twins/defs.py", "kind": "layer", "symbol": "<module>", "detail": "program.twins.defs -> functools"},
        entries[4],
    ]


def test_an_importer_grown_past_its_size_cap_joins_its_imports():
    rel = "integrations/vllm/verity_vllm/x.py"
    before = "from a.b import c\nfrom a.b import d as D\n\nX = 1\n"
    after = "from a.b import c\nfrom a.b import d as D\nfrom e import f\n\nX = 1\n"
    assert S.fit_size(rel, before, after, {"verity_vllm/x.py": 4}) == "from a.b import c, d as D\nfrom e import f\n\nX = 1\n"
    assert S.fit_size(rel, before, after, {"verity_vllm/x.py": 5}) == after
    with pytest.raises(S.Refused, match="over its P10 cap of 3"):
        S.fit_size(rel, before, after, {"verity_vllm/x.py": 3})


def test_this_tree_has_the_split_applied():
    """Every moved name is in its new module and not in its old one, every moved module is gone from the registry, and no
    import in the integration reaches a moved name through its old module (what `--check` would rewrite; the suite reads
    only its own inputs, and `--check` covers the rest of the tree)."""
    tree = S.Tree(REPO)
    for sp in S.SPLITS:
        src = {n for s in ast.parse(tree.read(S.path_of(sp.src))).body for n in S._bound(s)}
        dst = {n for s in ast.parse(tree.read(S.path_of(sp.dst))).body for n in S._bound(s)}
        assert not set(sp.names) & src, f"{sp.src} still defines {sorted(set(sp.names) & src)}"
        assert set(sp.names) <= dst, f"{sp.dst} lacks {sorted(set(sp.names) - dst)}"
    for m in S.MOVES:
        assert not tree.exists(S.path_of(m.src)) and tree.exists(S.path_of(m.dst)), m
    assert S.leftovers(tree, [rel for rel in tree.files() if rel.startswith(f"{S.VLLM}/")]) == []
    assert S.definition_imports(tree) == []
