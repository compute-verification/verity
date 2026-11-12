"""How the suite gets the benchmark's scripts (``conftest.py``): what the tree keeps for one ``sys.path`` to serve every
test, and `load`."""
from __future__ import annotations

import ast
import sys
import tomllib
from collections import Counter
from pathlib import Path

import conftest as CT
import report

POUW = CT.POUW
TESTS = Path(__file__).resolve().parent
#: (directory, name): one script of the directory may still import NAME by name beside its own ``NAME.py``, though under
#: the suite NAME is another directory's (``run`` is the front door's), so a test that loads the script puts the
#: script's own NAME in ``sys.modules`` first. Like a lint's list, an entry may stand before its script is in the tree,
#: and it goes in the commit that makes the script load the file
EXCUSED = frozenset({("pearl_c_sm90", "run")})


def imported(path: Path) -> set[str]:
    """The top-level module names `path` imports by name, anywhere in it."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.partition(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names.add(node.module.partition(".")[0])
    return names


def test_a_name_two_directories_hold_is_imported_by_name_from_one_of_them_at_most():
    """With every script directory on one path, ``import NAME`` is the first holder's module whoever asks. So only the
    first holder's own scripts and the tests ask by name: a script beside another ``NAME.py`` never does, nor one in no
    holder's directory, which means whichever holder it put on the path itself. Nobody asks by name for a NAME that
    `AHEAD` gives to no directory, and `AHEAD` names no directory it need not. `EXCUSED` lets one script of a directory
    ask beside its own file, whether or not `AHEAD` gives the name a directory, and no second one."""
    shared = {name: dirs for name, dirs in CT.holders().items() if len(dirs) > 1}
    ahead = [POUW / d for d in CT.AHEAD]
    wrong, used, excused = [], set(), set()
    for path in [*CT.scripts(), *sorted(TESTS.glob("*.py"))]:
        for name in sorted(imported(path) & shared.keys()):
            first, beside = shared[name][0], path.parent in shared[name]
            where = ", ".join(str(d.relative_to(POUW)) for d in shared[name])
            entry = (path.parent.relative_to(POUW).as_posix(), name)
            if beside and path.parent != first and entry in EXCUSED - excused:
                excused.add(entry)
            elif first not in ahead:
                wrong.append(f"{path.relative_to(POUW)} imports {name}, which {where} hold and conftest.AHEAD gives to "
                             f"none: put the directory whose scripts import it by name in AHEAD, or load() the file")
            elif beside and path.parent != first:
                wrong.append(f"{path.relative_to(POUW)} imports {name} beside its own {name}.py, and under the suite "
                             f"that is {first.relative_to(POUW)}'s: load it from its file")
            elif not beside and path.parent != TESTS:
                wrong.append(f"{path.relative_to(POUW)} imports {name} by name from outside {where}: under the suite "
                             f"that is {first.relative_to(POUW)}'s, so the script loads the file it means")
            else:
                used.add(first)
    assert not wrong, "\n".join(wrong)
    unused = [d.name for d in ahead if d not in used]
    assert not unused, f"nothing imports by name a module {unused} share with another directory: take them out of AHEAD"


def test_no_script_is_another_suites_unless_this_suites_inputs_hold_it():
    """A directory with its own ``pyproject.toml`` is a suite of its own. The read guard fails this suite when a test reads
    a file of one that this suite's ``inputs`` do not hold, and the test above reads every script: so `NOT_SCRIPTS` names
    each such directory."""
    inputs = tomllib.loads((POUW / "pyproject.toml").read_text())["tool"]["verity"]["tests"]["inputs"]
    held = {POUW.parents[1] / i for i in inputs}
    theirs = sorted({own.parent.name for own in POUW.glob("*/pyproject.toml")
                     if own.parent not in held and any(own.parent in p.parents for p in CT.scripts())})
    assert not theirs, f"{theirs} are suites of their own that this suite's inputs do not hold: name them in NOT_SCRIPTS"


def test_load_names_each_script_apart():
    names = Counter(CT.load_name(p.relative_to(POUW)) for p in CT.scripts())
    assert not [name for name, n in names.items() if n > 1]
    assert not names.keys() & CT.holders().keys()


def test_load_is_the_files_module_under_its_own_name_and_runs_it_once():
    theirs = CT.load("gpu_kit/sass_gate.py")
    assert theirs.__name__ == "gpu_kit_sass_gate" and Path(theirs.__file__) == POUW / "gpu_kit" / "sass_gate.py"
    assert sys.modules["gpu_kit_sass_gate"] is theirs is CT.load("gpu_kit/sass_gate.py")


def test_a_name_ahead_gives_a_directory_is_that_directorys_by_name():
    assert len(CT.holders()["report"]) > 1 and Path(report.__file__) == POUW / "harness" / "report.py"


def _first_import(name: str) -> Path:
    """The file a first ``import NAME`` runs: what the import system's finders give, asked in their order."""
    for finder in sys.meta_path:
        if (spec := finder.find_spec(name, None)) is not None:
            return Path(spec.origin)
    raise ModuleNotFoundError(name)


def test_a_name_two_directories_hold_stays_its_first_holders_whatever_is_first_on_the_path(monkeypatch):
    """A script a launcher starts has its own directory first on ``sys.path``, and many put it there themselves, so under
    the suite the path's order does not last past the first script a test imports. With any other holder's directory put
    first, a first ``import NAME`` is still the first holder's file. Without `FirstHolder` it would be the other's, which
    is what the finder is for."""
    shared = {name: dirs for name, dirs in CT.holders().items() if len(dirs) > 1}
    assert {"derive", "report", "run"} <= shared.keys()
    for name, (first, *others) in shared.items():
        for other in others:
            monkeypatch.setattr(sys, "path", [str(other), *sys.path])
            assert _first_import(name) == first / f"{name}.py"
            with monkeypatch.context() as without:
                without.setattr(sys, "meta_path", [f for f in sys.meta_path if not isinstance(f, CT.FirstHolder)])
                assert _first_import(name) == other / f"{name}.py"
    # and it answers for nothing else: a name one directory holds, another package's, or a submodule's
    [finder] = [f for f in sys.meta_path if isinstance(f, CT.FirstHolder)]
    for name in ("conftest", "bench", "json", "verity"):
        assert finder.find_spec(name) is None
    assert finder.find_spec("report", [str(POUW)]) is None


def test_a_file_that_fails_to_load_leaves_no_module_behind(tmp_path, monkeypatch):
    (tmp_path / "broken.py").write_text("raise RuntimeError('at import')\n")
    monkeypatch.setattr(CT, "POUW", tmp_path)
    for _ in range(2):
        try:
            CT.load("broken.py")
        except RuntimeError as ex:
            assert str(ex) == "at import"
        else:
            raise AssertionError("a file that raises loaded")
        assert CT.load_name("broken.py") not in sys.modules
