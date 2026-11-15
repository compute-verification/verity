"""The suite's swap guard (`conftest.py`): a test that leaves a loaded `verity` module swapped for another object, in
`sys.modules` or as its package's attribute, is failed by name and the module is put back for the tests after it; a test whose
`monkeypatch` undoes its swap is not failed, in a plain run or under the plugin `suites.py` loads into every suite."""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

CASES = "tests/tools/swap_guard_cases.py"
#: what `suites.py` hands its plugin about this run, which a pytest started here must not act on: no report written over this
#: run's, no test left out as another part's or as a reused pass
RUNNER = ("VERITY_SUITE_REPORT", "VERITY_SUITE_PART", "VERITY_SUITE_REUSE", "VERITY_SUITE_HEAVY")


@pytest.mark.parametrize("plugin", [False, True], ids=["plain", "under-the-suite-guard"])
def test_the_guard_fails_a_test_that_leaves_a_module_swapped_and_no_test_whose_monkeypatch_undoes_it(
        tmp_path, monkeypatch, plugin) -> None:
    """The cases of `swap_guard_cases.py` in a pytest of their own.  Under the suite's plugin too, loaded by name as `suites.py`
    loads it and from where it loaded it into this run (skipped in a run it didn't start, which has none): its autouse fixture
    takes `monkeypatch` before any fixture of this suite is set up, so without `conftest.py`'s own `monkeypatch` the guard would
    look before the swap is undone."""
    for name in RUNNER:
        monkeypatch.delenv(name, raising=False)
    loaded = sys.modules.get("verity_suite_guard")
    if plugin and loaded is not None:
        monkeypatch.setenv("PYTHONPATH", str(Path(inspect.getfile(loaded)).parent), prepend=os.pathsep)
    elif plugin:
        pytest.skip("`suites.py` loads its plugin into the suites it runs, and it didn't start this run")
    r = subprocess.run([sys.executable, "-m", "pytest", CASES, "-rA", "-p", "no:cacheprovider", f"--basetemp={tmp_path / 'b'}",
                        *(["-p", "verity_suite_guard"] if plugin else [])], cwd=Path(__file__).resolve().parents[2],
                       capture_output=True, text=True)
    lines = r.stdout.splitlines()
    left = [f"{CASES}::test_a_loaded_module_left_swapped_is_failed_by_name",
            f"{CASES}::test_a_package_attribute_left_swapped_is_failed_by_name"]
    assert sorted(line for line in lines if line.startswith(("PASSED ", "FAILED ", "ERROR "))) == sorted([
        f"PASSED {CASES}::test_a_stand_in_module_set_with_monkeypatch_is_not_left",
        f"PASSED {left[0]}", f"ERROR {left[0]}",
        f"PASSED {CASES}::test_the_test_after_it_reads_the_module_that_was_loaded",
        f"PASSED {left[1]}", f"ERROR {left[1]}",
        f"PASSED {CASES}::test_the_test_after_that_reads_the_attribute_that_was_there"]), r.stdout + r.stderr
    assert r.returncode == 1
    for case in left:
        assert any(f"ERROR at teardown of {case.rpartition('::')[2]}" in line for line in lines)
    assert r.stdout.count("swapped for another copy (sys.modules or package attribute): ['verity.circuits.defs']") == 2
