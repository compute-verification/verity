"""No test writes a measurement into a research run's result. Under `research run` a process inherits `RESEARCH_RUN_DIR`
and `OUT`, and the harness's kit (`verity_bench.kit.writer`, `kit.report`) writes into `$RESEARCH_RUN_DIR/result.json`
whatever directory its caller names, so a test run there by plain pytest wrote its fixture's figures as the run's
measurements, and the Metrics tab drew them as a benchmark's points. Inside pytest the kit refuses the run the
process started under (`kit.refuse_the_started_run`), and a command a test starts, which doesn't run pytest, writes it.
`suites.py` drops `RESEARCH_*` for its suites; a plain pytest doesn't, so each suite that can reach the kit unsets both
in an autouse fixture of its own conftest, which every pytest of its tests loads. A suite reaches the kit where its
inputs, as `suites.py` keys and guards them (`inputs_of`: its directory, its dependencies' and what it declares), hold
the kit: the read guard fails one whose tests import the kit outside them."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: the variables that name where a run's result and outputs are written
PLACES = ("RESEARCH_RUN_DIR", "OUT")
#: an input that holds the kit's writer, `benchmarks/harness/verity_bench/kit.py`, and its report
KIT = ("benchmarks/harness/verity_bench/kit.py", "benchmarks/harness/verity_bench/kit_report.py")
FIX = '''@pytest.fixture(autouse=True)
def _no_run_result(monkeypatch):
    """No test writes into a research run's result (`tests/test_tests_write_no_measurement.py`)."""
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)'''


def holds(inputs: list[str]) -> bool:
    """Whether one of a suite's inputs is the kit or a directory above it."""
    return any(k == i or k.startswith(i.rstrip("/") + "/") for k in KIT for i in inputs if i != ".")


def reaching() -> list[str]:
    """The directory of each suite's conftest whose inputs reach the kit's writer: `tests` for the repository's suite."""
    sys.path.insert(0, str(ROOT / "tools" / "check"))
    try:
        import suites
    finally:
        sys.path.remove(str(ROOT / "tools" / "check"))
    pkgs = suites.members()
    return sorted("tests" if s.path == "." else s.path for s in suites.discover()
                  if s.path == "." or holds(suites.inputs_of(s, pkgs)))


def unsets_both(source: str) -> bool:
    """Whether a conftest holds an autouse fixture that deletes both `PLACES` from the environment."""
    for f in ast.walk(ast.parse(source)):
        if not isinstance(f, ast.FunctionDef):
            continue
        autouse = any(isinstance(d, ast.Call) and any(k.arg == "autouse" and isinstance(k.value, ast.Constant)
                                                      and k.value.value is True for k in d.keywords)
                      for d in f.decorator_list)
        names = {n.value for n in ast.walk(f) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        deletes = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "delenv"
                      for c in ast.walk(f))
        if autouse and deletes and set(PLACES) <= names:
            return True
    return False


def test_every_suite_that_reaches_the_kit_starts_each_test_outside_a_run():
    missing = [s for s in reaching()
               if not (ROOT / s / "conftest.py").is_file() or not unsets_both((ROOT / s / "conftest.py").read_text())]
    assert not missing, (f"these suites can reach the kit's writer and don't unset {' and '.join(PLACES)} before each "
                         f"test, so under `research run` a test of theirs writes the run's measurements: "
                         f"{', '.join(missing)}. Put this in each one's conftest.py (with `import pytest`):\n\n{FIX}")


def test_the_rule_finds_a_suite_by_its_inputs_and_a_fixture_by_what_it_deletes():
    assert holds(["benchmarks/harness"]) and holds(["benchmarks/harness/verity_bench/kit.py"])
    assert not holds(["benchmarks/harness/specs", "benchmarks/harness/README.md", "verity", "."])
    assert {"tests", "tests/benchmarks/harness", "tests/benchmarks/compute"} <= set(reaching())
    assert unsets_both("import pytest\n\n" + FIX)
    assert not unsets_both("import pytest\n\n" + FIX.replace("(autouse=True)", ""))
    assert not unsets_both("import pytest\n\n" + FIX.replace('"OUT"', '"ELSE"'))
    assert not unsets_both("import pytest\n\n" + FIX.replace("delenv", "setenv"))
