"""conftest.py's catalog rules, in a pytest of their own over a copy of it: a process builds circuit-check's catalog only for a
selected test that asks for it, such a test is marked slow (so `--quick` builds none), and without it `targets.suite` and
`targets.catalog` fail by name rather than build one."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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


def test_a_test_that_asks_for_the_catalog_without_the_slow_mark_builds_none(tmp_path):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_text((Path(__file__).parent / "conftest.py").read_text())
    (tests / "test_it.py").write_text(TESTS)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=tmp_path,
                       check=False, capture_output=True, text=True)
    assert "1 passed, 1 error" in r.stdout, r.stdout[-3000:]
    assert "mark them slow: ['tests/test_it.py::test_unmarked']" in r.stdout, r.stdout[-3000:]
