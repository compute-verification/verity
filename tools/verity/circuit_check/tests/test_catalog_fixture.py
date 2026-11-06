"""conftest.py's catalog rules, in a pytest of their own over a copy of it: a process builds circuit-check's catalog only for a
selected test that asks for it, such a test is marked slow (so `--quick` builds none), and without it `targets.suite` and
`targets.catalog` fail by name rather than build one; handed records (`CIRCUIT_CHECK_RECORDS`) that don't count fail the session
by name rather than build one."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PYPROJECT = """[tool.pytest.ini_options]
markers = ["slow: x", "circuit_suite: x"]

[tool.verity.tests]
worker_gb = 32
"""

TESTS = '''
import conftest
import pytest

from circuit_check import targets as TG


def test_unmarked(suite):
    pass


def test_quick():
    assert not conftest._BUILT
    with pytest.raises(pytest.fail.Exception, match="give this test one, and mark it slow"):
        TG.suite()
    with pytest.raises(pytest.fail.Exception, match="give this test one, and mark it slow"):
        TG.catalog()
'''


#: a catalog test over records `units.plan` finds none of, or whose one binding's record misses (the registries unloaded)
HANDED = '''
import pytest

from circuit_check import units as UN

{plan}


@pytest.mark.slow
def test_marked(suite):
    pass
'''

PLANS = {
    "none": ("UN.plan = lambda cache, repo, *a, **k: None", "they hold no record of this circuit-check's version"),
    "missed": ('UN.plan = lambda cache, repo, *a, **k: UN._lookup(cache, repo, UN.Entry(("syn#_x_roots",), "_x_roots", '
               '"_x_roots", UN.key(("syn#_x_roots",), 0), []))', "syn#_x_roots's record doesn't count"),
}


def _session(tmp_path: Path, tests_py: str, env: dict | None = None) -> subprocess.CompletedProcess:
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_text((Path(__file__).parent / "conftest.py").read_text())
    (tests / "test_it.py").write_text(tests_py)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=tmp_path,
                          check=False, capture_output=True, text=True, env=env)


def test_a_test_that_asks_for_the_catalog_without_the_slow_mark_builds_none(tmp_path):
    r = _session(tmp_path, TESTS)
    assert "1 passed, 1 error" in r.stdout, r.stdout[-3000:]
    assert "mark them slow: ['tests/test_it.py::test_unmarked']" in r.stdout, r.stdout[-3000:]


@pytest.mark.parametrize("plan", sorted(PLANS))
def test_handed_records_that_dont_count_fail_the_session_by_name_before_any_test(tmp_path, plan):
    stub, why = PLANS[plan]
    records = tmp_path / "records"
    records.mkdir()
    r = _session(tmp_path, HANDED.format(plan=stub), {**os.environ, "CIRCUIT_CHECK_RECORDS": str(records)})
    assert r.returncode == pytest.ExitCode.USAGE_ERROR, (r.stdout[-3000:], r.stderr[-3000:])
    assert f"CIRCUIT_CHECK_RECORDS={records} was handed, and {why}" in r.stderr, r.stderr[-3000:]
    assert "passed" not in r.stdout, r.stdout[-3000:]
