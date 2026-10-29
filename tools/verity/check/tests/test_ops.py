"""The ops seam: the check reaches research only through `ops.py`, which names exactly what it takes, and its configuration is read
from a root it is given, at the home research's queue and drive read too, so moving the check out of verity changes paths alone."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    return mod


OPS = _module("ops_under_test", HERE / "ops.py")
D = _module("declaration_for_ops", HERE / "declaration.py")
#: what loads a module by a name it is given
IMPORTERS = frozenset({"import_module", "__import__", "importorskip"})


def _files() -> dict[str, ast.Module]:
    return {p.relative_to(HERE).as_posix(): ast.parse(p.read_text(), str(p)) for p in sorted(HERE.rglob("*.py"))
            if "__pycache__" not in p.parts}


def _research(name: str) -> bool:
    return name == "research" or name.startswith("research.")


def research_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Each import of research in ``tree``: an import statement, a name handed to `IMPORTERS`, or a `module:attr` string."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(n.lineno, a.name) for a in n.names if _research(a.name)]
        elif isinstance(n, ast.ImportFrom) and not n.level and _research(n.module or ""):
            out.append((n.lineno, n.module))
        elif isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            f = n.func
            if (f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None) in IMPORTERS \
                    and _research(n.args[0].value):
                out.append((n.lineno, n.args[0].value))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and ":" in n.value and " " not in n.value \
                and _research(n.value.partition(":")[0]):
            out.append((n.lineno, n.value))
    return out


def _is_ops(e: ast.expr) -> bool:
    """Whether ``e`` is the ops module as a file here holds it: `OPS`, `_ops` or `ops`, `X.OPS`, or a load of `ops.py`."""
    if isinstance(e, ast.Name):
        return e.id in ("OPS", "_ops", "ops")
    if isinstance(e, ast.Attribute):
        return e.attr == "OPS"
    return isinstance(e, ast.Call) and any(isinstance(c, ast.Constant) and c.value == "ops.py" for c in ast.walk(e))


def ops_reads(tree: ast.Module) -> set[str]:
    """The names ``tree`` reads of ops: not its tables or its dunders, which only this file reads."""
    return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and _is_ops(n.value)
            and n.attr not in ("USES", "TESTS") and not n.attr.startswith("__")}


def test_ops_is_the_only_importer_of_research_under_the_check():
    found = [f"{f}:{line}: {name}" for f, tree in _files().items() if f not in ("ops.py", "tests/test_ops.py")      # its examples
             for line, name in research_imports(tree)]
    assert not found, ("these reach research past ops.py; read the name from OPS (`OPS = _load(\"ops\")` in check.py), "
                       "naming it in ops.py's USES, or TESTS for a test's:\n" + "\n".join(found))


def test_the_scan_sees_each_way_of_importing_research():
    code = ("import research\nimport research.store.local as L\nfrom research.merge import rule_touches\n"
            "pytest.importorskip('research.store.remote')\nimportlib.import_module('research.runs')\nX = 'research.queue:Rules'\n"
            "from .research import x\nimport researcher\nY = 'see research.mem: its reservations'\n")
    assert [n for _, n in research_imports(ast.parse(code))] == [
        "research", "research.store.local", "research.merge", "research.store.remote", "research.runs", "research.queue:Rules"]


def test_ops_names_exactly_what_the_check_and_its_tests_read_of_it():
    """A name the code reads is in USES and a name only tests read is in TESTS, and each is read where its table says."""
    files = _files()
    code = set().union(*(ops_reads(t) for f, t in files.items() if not f.startswith("tests/")))
    tests = set().union(*(ops_reads(t) for f, t in files.items() if f.startswith("tests/")))
    assert not set(OPS.USES) & set(OPS.TESTS)
    assert code == set(OPS.USES), f"USES lists {sorted(set(OPS.USES) - code)} that no code reads; code reads {sorted(code - set(OPS.USES))}"
    assert tests - code == set(OPS.TESTS), (f"TESTS lists {sorted(set(OPS.TESTS) - tests)} that no test reads; tests read "
                                            f"{sorted(tests - code - set(OPS.TESTS))} besides USES")


def test_each_name_ops_lists_resolves_in_research_and_one_it_doesnt_is_refused():
    for name, where in {**OPS.USES, **OPS.TESTS}.items():
        assert getattr(OPS, name) is not None, where
    try:
        OPS.__getattr__("not_a_name")
    except AttributeError as e:
        assert "name it in ops.py's USES" in str(e)
    else:
        raise AssertionError("ops resolved a name it doesn't list")


def test_a_name_ops_lists_is_read_from_research_each_time_so_a_patch_reaches_the_check(monkeypatch):
    monkeypatch.setattr(OPS.store_local, "LocalStore", "patched")
    assert OPS.LocalStore == "patched"


def test_the_check_and_research_read_its_configuration_at_one_home():
    """ci.toml, drive.toml and queue.toml sit in one directory, named once in the check (`declaration.HOME`) and once in
    research (`queue.CHECK_HOME`): moving them is a change of those two constants."""
    assert D.HOME == OPS.queue.CHECK_HOME
    assert D.CI == OPS.CI_JOBS == f"{D.HOME}/ci.toml"
    assert OPS.DRIVE_FLAGS == f"{D.HOME}/drive.toml" and OPS.RULES_BEFORE_MOVE == f"{D.HOME}/queue.toml"
    assert all((D.ROOT / p).is_file() for p in (D.CI, OPS.DRIVE_FLAGS, OPS.RULES_BEFORE_MOVE))


def test_the_declaration_is_read_from_the_root_it_is_given_not_the_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "verity"
    (root / D.CI).parent.mkdir(parents=True)
    (root / D.CI).write_text('schema = "ci/v0"\n[[job]]\nname = "pytest"\nowner = "ci"\ngroup = 1\n'
                             '[[job]]\nname = "lean"\nowner = "ci"\ngroup = 2\n[[plan]]\nshards = [["pytest"], ["lean"]]\n')
    (root / D.REGISTRY).parent.mkdir(parents=True)
    (root / D.REGISTRY).write_text(json.dumps({"handles": [{"handle": "ci"}]}))
    monkeypatch.chdir(tmp_path)
    assert [j.name for j in D.load(root)] == ["pytest", "lean"]
    assert D.plans(root) == {2: (("pytest",), ("lean",))}
    assert [j.name for j in D.load()] != ["pytest", "lean"], "with no root, the tree this declaration.py is in"
