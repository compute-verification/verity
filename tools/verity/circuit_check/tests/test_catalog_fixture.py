"""conftest.py's catalog rules, in a pytest of their own over a copy of it: a process builds circuit-check's catalog only for a
selected test that asks for it, such a test is marked slow (so `--quick` builds none), and without it `targets.suite` and
`targets.catalog` fail by name rather than build one; records that don't count, handed (`CIRCUIT_CHECK_RECORDS`) or not, end
the plan at the first one, so that this process builds no binding, and the session's one build of the records, in a process
of its own and once for every xdist worker, stores them where the session then takes them; and where that build stored none
a session can take, the catalog is built without records and its line says why."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PYPROJECT = """[tool.pytest.ini_options]
markers = ["slow: x"]

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


#: stands in for ``circuit-check --all --records-only``: notes in the records' directory that it ran, and with which other
#: arguments, then reports and exits as ``builder.json`` beside it says
BUILDER = '''
import json
import sys
from pathlib import Path

cache, out = Path(sys.argv[1]), Path(sys.argv[2])
does = json.loads((Path(__file__).parent / "builder.json").read_text())
cache.mkdir(parents=True, exist_ok=True)
with open(cache / "runs", "a") as f:
    f.write(" ".join(sys.argv[3:]) + "\\n")
if does["report"] is not None:
    out.write_text(json.dumps(does["report"]))
sys.stderr.write(does["said"])
sys.exit(does["rc"])
'''

#: two catalog tests over records of one binding (the registries unloaded), as ``want.json`` beside them says: its record
#: counts from the start (``counts``), or only once `BUILDER` ran and exited 0; before that `units.plan` finds no record of
#: this version (``plan`` ``none``) or one that misses.  A plan that goes on past the miss is where `units.plan` builds that
#: binding in this process; the catalog without records is a stub that says it made one, and why
RECORDS = '''
import json
import sys
import types
from pathlib import Path

import conftest
import pytest

from circuit_check import units as UN

HERE = Path(__file__).parent
ENTRY = UN.Entry(("syn#_x_roots",), "_x_roots", "_x_roots", UN.key(("syn#_x_roots",), 0), [], record="r")
PLAN = types.SimpleNamespace(entries=[ENTRY], calls=set(), derived=types.SimpleNamespace(ids=[], units={}, nodes={}))
WANT = json.loads((HERE / "want.json").read_text())


def _stored(cache):
    return (Path(cache) / "runs").exists() and json.loads((HERE / "builder.json").read_text())["rc"] == 0


def _lookup(cache, repo, e):
    return ({"nodes": {}}, {}) if WANT["counts"] or _stored(cache) else None


def _plan(cache, repo, *a, **k):
    if WANT["plan"] == "none" and not _stored(cache):
        return None
    if UN._lookup(cache, repo, ENTRY) is None:
        raise AssertionError("the plan went on past a record that doesn't count, to build its binding in this process")
    return PLAN


UN._lookup, UN.plan = _lookup, _plan
UN.listing = lambda cache, entries, roots=None: {"listed": [e.name for e in entries]}
conftest._records_argv = lambda cache, jobs, mem_gb, out, listed: [
    sys.executable, str(HERE / "builder.py"), str(cache), str(out), *(["--records-manifest", listed] if listed else [])]
conftest._apart = lambda shared, jobs, why=None: conftest.Catalog(conftest.Suite([], lambda fid: None), {}, lambda fid: None,
                                                                  f"built apart (a stub): {why}", set())


@pytest.mark.slow
def test_marked(suite):
    assert conftest._BUILT[0].how == WANT["how"]


@pytest.mark.slow
def test_marked_too(suite):
    assert conftest._BUILT[0].how == WANT["how"]
'''

#: what `BUILDER` reports when it built the one binding and stored its record
STORED = {"reused": 0, "rebuilt": ["syn#_x_roots"], "skipped": ["syn#_x_roots"], "stored": 1, "seconds": 1.5}
BUILT = ("from records: 0 of 1 bindings reused, 1 built and stored in 1.5s by one process for the session (1 of them held a "
         "record that did not count): syn#_x_roots")


def _session(tmp_path: Path, tests_py: str, env: dict | None = None, args: tuple[str, ...] = (),
             files: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    # the copy reads its variables through circuit-check's registry, which it finds beside the pyproject.toml above it
    (tmp_path / "settings.py").write_text((Path(__file__).parents[1] / "settings.py").read_text())
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_text((Path(__file__).parent / "conftest.py").read_text())
    (tests / "test_it.py").write_text(tests_py)
    for name, text in (files or {}).items():
        (tests / name).write_text(text)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args, "tests"], cwd=tmp_path,
                          check=False, capture_output=True, text=True, env=env)


def _records_session(tmp_path: Path, how: str, *, counts: bool = False, plan: str = "missed", report: dict | None = None,
                     rc: int = 0, said: str = "", env: dict | None = None,
                     args: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """`RECORDS`' session, whose two tests hold the catalog's line to ``how``, over the home directory ``tmp_path``."""
    home = {k: v for k, v in os.environ.items() if not k.startswith("CIRCUIT_CHECK_RECORDS")}
    return _session(tmp_path, RECORDS, {**home, "HOME": str(tmp_path), **(env or {})}, args,
                    {"builder.py": BUILDER, "builder.json": json.dumps({"report": report, "rc": rc, "said": said}),
                     "want.json": json.dumps({"how": how, "counts": counts, "plan": plan})})


def test_a_test_that_asks_for_the_catalog_without_the_slow_mark_builds_none(tmp_path):
    r = _session(tmp_path, TESTS)
    assert "1 passed, 1 error" in r.stdout, r.stdout[-3000:]
    assert "mark them slow: ['tests/test_it.py::test_unmarked']" in r.stdout, r.stdout[-3000:]


@pytest.mark.parametrize("plan", ["missed", "none"])
@pytest.mark.parametrize("handed", [False, True])
def test_records_that_dont_count_are_built_and_stored_by_one_process_and_then_taken(tmp_path, plan, handed):
    """A record that doesn't count, or none of this version, in the directory `check` keeps them in or a handed one, is no
    error: the plan ends before this process builds its binding, the session's build stores it there, and the session takes
    it; the catalog's line counts it."""
    records = tmp_path / "handed" if handed else tmp_path / ".cache" / "verity-check" / "circuit-check"
    r = _records_session(tmp_path, BUILT, plan=plan, report=STORED,
                         env={"CIRCUIT_CHECK_RECORDS": str(records)} if handed else None)
    assert r.returncode == 0 and "2 passed" in r.stdout, (r.stdout[-3000:], r.stderr[-3000:])
    assert (records / "runs").read_text() == "\n"


def test_under_xdist_one_worker_starts_the_build_of_the_records_and_every_worker_takes_them(tmp_path):
    """Two workers, each of which makes the catalog before its tests: the records are built once, and the controller shows
    the catalog's line, which no worker's own output reaches."""
    r = _records_session(tmp_path, BUILT, report=STORED, args=("-n", "2"))
    assert r.returncode == 0 and "2 passed" in r.stdout, (r.stdout[-3000:], r.stderr[-3000:])
    assert (tmp_path / ".cache" / "verity-check" / "circuit-check" / "runs").read_text() == "\n"
    assert f"circuit-check's catalog: {BUILT}, 0 targets (in 2 xdist worker(s))" in r.stdout, r.stdout[-3000:]


def test_records_that_all_count_start_no_build_and_the_session_lists_them(tmp_path):
    listed = tmp_path / "out" / "listed.json"
    r = _records_session(tmp_path, "from records: 1 of 1 bindings reused, 0 built", counts=True,
                         env={"CIRCUIT_CHECK_RECORDS_MANIFEST": str(listed)})
    assert r.returncode == 0 and "2 passed" in r.stdout, (r.stdout[-3000:], r.stderr[-3000:])
    assert not (tmp_path / ".cache").exists()
    assert json.loads(listed.read_text()) == {"listed": ["syn#_x_roots"]}


def test_the_build_of_the_records_lists_them_where_the_session_was_told_to(tmp_path):
    """With a record to build, the listing is the build's, which knows which records it built: the session hands it the
    path and writes none of its own."""
    listed = tmp_path / "listed.json"
    r = _records_session(tmp_path, BUILT, report=STORED, env={"CIRCUIT_CHECK_RECORDS_MANIFEST": str(listed)})
    assert r.returncode == 0 and "2 passed" in r.stdout, (r.stdout[-3000:], r.stderr[-3000:])
    assert (tmp_path / ".cache" / "verity-check" / "circuit-check" / "runs").read_text() == f"--records-manifest {listed}\n"
    assert not listed.exists()


@pytest.mark.parametrize("report,rc,said,why", [
    ({"reused": 0, "rebuilt": ["syn#_x_roots"], "stored": 0, "why": "the tracer could not see what it did (os.system true)"},
     1, "", "the tracer could not see what it did (os.system true)"),
    (None, 3, "MemoryError: boom\n", "`circuit-check --all --records-only` exited 3: MemoryError: boom"),
])
def test_a_build_that_stored_no_record_leaves_the_catalog_built_without_records_and_its_line_says_why(tmp_path, report, rc, said,
                                                                                                    why):
    r = _records_session(tmp_path, f"built apart (a stub): {why}", report=report, rc=rc, said=said)
    assert r.returncode == 0 and "2 passed" in r.stdout, (r.stdout[-3000:], r.stderr[-3000:])
    assert (tmp_path / ".cache" / "verity-check" / "circuit-check" / "runs").read_text() == "\n"
