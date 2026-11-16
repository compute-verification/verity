"""`conftest.py`'s `BOOKKEEPING`: every test it names exists, and a named test's failed check warns and passes while an
unnamed test's fails and a named test that crashes still fails."""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFTEST = Path(__file__).resolve().parent / "conftest.py"


def test_every_listed_test_exists() -> None:
    spec = importlib.util.spec_from_file_location("bookkeeping_conftest", CONFTEST)
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    gone = [f"{f}::{t}" for f, ts in bk.BOOKKEEPING.items() for t in ts
            if not (ROOT / f).is_file() or not re.search(rf"^def {t}\(", (ROOT / f).read_text(encoding="utf-8"), re.M)]
    assert not gone, ("tests/conftest.py's BOOKKEEPING names tests that don't exist (renamed or deleted): "
                      f"{', '.join(gone)}. Rename or drop each entry")


def test_a_listed_failure_warns_and_the_rest_fail(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text(textwrap.dedent(f"""
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location("bookkeeping_conftest", {str(CONFTEST)!r})
        bk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bk)
        bk.ROOT = Path(__file__).resolve().parent
        bk.BOOKKEEPING = {{"test_x.py": ("test_listed", "test_listed_crash")}}
        pytest_configure, pytest_collection_modifyitems, pytest_pyfunc_call = (
            bk.pytest_configure, bk.pytest_collection_modifyitems, bk.pytest_pyfunc_call)
    """))
    (tmp_path / "test_x.py").write_text(textwrap.dedent("""
        import pytest

        def test_listed():
            assert False, "found a stale entry"

        def test_listed_crash():
            raise RuntimeError("the scan broke")

        def test_unlisted():
            pytest.fail("a real failure")
    """))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", str(tmp_path)],
                       cwd=tmp_path, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "PASSED test_x.py::test_listed" in out and "found a stale entry" in out and "Bookkeeping" in out, out
    assert "FAILED test_x.py::test_listed_crash" in out and "FAILED test_x.py::test_unlisted" in out, out
