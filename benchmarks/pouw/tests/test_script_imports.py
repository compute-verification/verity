"""How the suite gets the benchmark's scripts (``conftest.py``): what the tree keeps for one ``sys.path`` to serve every
test, and `load`."""
from __future__ import annotations

import ast
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import conftest as CT
import report

POUW = CT.POUW
TESTS = Path(__file__).resolve().parent


def holders() -> dict[str, list[Path]]:
    """Each script's module name with the directories that hold it, in the path's order."""
    held, order = defaultdict(set), CT.script_dirs()
    for p in CT.scripts():
        held[p.stem].add(p.parent)
    return {name: [d for d in order if d in dirs] for name, dirs in held.items()}


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
    `AHEAD` gives to no directory, and `AHEAD` names no directory it need not."""
    shared = {name: dirs for name, dirs in holders().items() if len(dirs) > 1}
    ahead = [POUW / d for d in CT.AHEAD]
    wrong, used = [], set()
    for path in [*CT.scripts(), *sorted(TESTS.glob("*.py"))]:
        for name in sorted(imported(path) & shared.keys()):
            first, beside = shared[name][0], path.parent in shared[name]
            where = ", ".join(str(d.relative_to(POUW)) for d in shared[name])
            if first not in ahead:
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
    assert not names.keys() & holders().keys()


def test_load_is_the_files_module_under_its_own_name_and_runs_it_once():
    theirs = CT.load("gpu_kit/sass_gate.py")
    assert theirs.__name__ == "gpu_kit_sass_gate" and Path(theirs.__file__) == POUW / "gpu_kit" / "sass_gate.py"
    assert sys.modules["gpu_kit_sass_gate"] is theirs is CT.load("gpu_kit/sass_gate.py")


def test_a_name_ahead_gives_a_directory_is_that_directorys_by_name():
    assert len(holders()["report"]) > 1 and Path(report.__file__) == POUW / "harness" / "report.py"


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
        assert "broken" not in sys.modules
