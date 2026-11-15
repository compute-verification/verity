"""Every opt-in test a cold check runs by name (check.py's `OPT_IN`: the nightly's) is in its suite and carries a mark that reads
its variable: a test renamed or deleted, or a mark that stops reading the variable, fails here in every check rather than in the
nightly alone, where the test would otherwise run nowhere."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "tools" / "verity" / "check" / "check.py"


def opt_in() -> dict[str, dict[str, tuple[str, ...]]]:
    """check.py's `OPT_IN` (suite -> variable -> tests), read from its source: a literal, so the table needs no import of
    check."""
    [value] = [n.value for n in ast.parse(CHECK.read_text()).body
               if isinstance(n, ast.Assign) and [getattr(t, "id", None) for t in n.targets] == ["OPT_IN"]]
    return ast.literal_eval(value)


def marks(path: Path, test: str) -> str | None:
    """The source of ``test``'s decorators in ``path`` and of the module's definitions they name (`_opt_in`, `OPT_IN`), or None
    when the module defines no such test."""
    src = path.read_text() if path.is_file() else ""
    body = ast.parse(src).body
    tests = {n.name: n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    named = {**tests, **{t.id: n for n in body if isinstance(n, (ast.Assign, ast.AnnAssign))
                         for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}}
    if test not in tests:
        return None
    used = {x.id for d in tests[test].decorator_list for x in ast.walk(d) if isinstance(x, ast.Name)} & set(named)
    return "\n".join([*(ast.get_source_segment(src, d) or "" for d in tests[test].decorator_list),
                      *(ast.get_source_segment(src, named[n]) or "" for n in sorted(used))])


def test_every_opt_in_test_a_cold_check_names_is_in_its_suite_and_marked_by_its_variable():
    table = opt_in()
    assert table, "check.py's OPT_IN is empty"
    wrong = []
    for suite, var, tests in ((suite, var, tests) for suite, by_var in table.items() for var, tests in by_var.items()):
        for test in tests:
            file, _, name = test.partition("::")
            seen = marks(ROOT / suite / file, name)
            if seen is None:
                wrong.append(f"{suite}: {test} isn't in its suite: rename or drop it in check.py's OPT_IN")
            elif var not in seen:
                wrong.append(f"{suite}: {test}: none of its marks reads {var}: mark it to skip unless {var}=1, or name in "
                             "OPT_IN the variable its mark reads")
    assert not wrong, "\n".join(wrong)
