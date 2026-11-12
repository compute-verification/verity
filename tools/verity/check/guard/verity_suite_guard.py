"""The pytest plugin `suites.py` loads into every suite it runs (`-p verity_suite_guard`, from this directory, which holds
nothing else, so putting it on `sys.path` shadows no module).

It records every repository file the tests open outside the suite's inputs (`VERITY_SUITE_ALLOW`, paths relative to
`VERITY_SUITE_ROOT`, less the nested suites `VERITY_SUITE_DENY` names), every test whose call took over 5 s without the `slow`
mark, and the outcome counts, and writes them to
`VERITY_SUITE_REPORT` when the session ends. `suites.py` fails a suite that read outside its inputs or wrote no report: a
cached pass is only sound when everything the tests read is part of the key. Opens go through Python's audit hook (`open`
events), installed when pytest loads this module, before any conftest: it sees every file Python opens, imports included, but
nothing a subprocess opens. A Python a test starts traces its own opens (`child/`), and `suites.py` fails the suite on those
outside the inputs too, naming the test (`VERITY_SUITE_TEST`, set while one runs).

It also records what `suites.py` keeps beside a pass and checks before reusing it: which of the metadata files in
`VERITY_SUITE_WATCH` the tests opened, the installed distributions whose modules were imported (from site-packages), and the
top-level imports that failed with no installed distribution providing the name (`import` events).

Under pytest-xdist every worker loads it too, since xdist gives each worker the controller's command line and `sys.path`.
A worker sends its record to the controller when its session ends (`workeroutput`), and the controller counts outcomes from
the reports workers send it. A worker that sends no record fails the suite (`unguarded`).

For `suites.py`'s per-test verdicts it also records each test that passed (every phase, and not an xfail) with its call's
seconds, each test that failed, and which of the suite's `test_*.py` files were opened while a test module was collected or
its tests ran (under `"*"` when neither: a session fixture, a conftest). It deselects the tests `VERITY_SUITE_REUSE` names,
whose passes `suites.py` reuses, and in an xdist worker it moves the longest tests (`VERITY_SUITE_HEAVY`) to the heads of the
workers' shares: an order only, which parallel suites declare their outcomes don't depend on. Under `suites.py --files` it
collects only the test files of the run's part (`VERITY_SUITE_PART`, `in_part`), or of a part by test id deselects the tests
whose node id hashes to another part (`id_part`), and reports how many it left out (`part_out`). Under `suites.py --changed
--quick` it marks the tests of the files `--verity-whole` names `verity_whole`, so that run's `-m` keeps them whole. It sets `VERITY_SUITE_MODULE` while a test module is collected or run, so the Pythons a test starts
trace their own reads (`child/`), and in a child this process forks it writes each such read at once to `VERITY_SUITE_READS`
(`child/verity_suite_reads.py`): a fork's memory never comes back, and one that ends through `os._exit` flushes nothing.

It traces, too, what a per-test key narrowed to the files the tests use holds (`suites.py`, `Traced`): each repository file its
process opens or loads and each repository directory it lists (`traced`), and, per test module, each program started that no
tracer sees into (`untraced`), as `child/verity_suite_reads.py` classifies them; its children write theirs to the reads file.

Two kinds of skip fail instead: one for an unreachable evidence store under a recorded check (`STORE_UNREACHABLE`), and a
`pytest.importorskip` whose missing module is this checkout's (`importorskip`), which means it moved.

It keeps the environment's values out of a failure's text: a suite runs in its caller's environment, credentials included,
and a run's log is published. While a suite runs, `os.environ`'s repr names its variables and holds no value
(`_environ_names`, from this module's load until the session ends). A copy of the environment (`dict(os.environ)`) still
prints as a dict does, and so does the environment of a Python a test starts (the forms: `STILL` in `tests/test_guard.py`).
To see a value, run the test alone in an emptied environment.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
import sysconfig
import tempfile

import pytest

SLOW_S = 5.0
#: the reason a test that needs the evidence store gives when it skips because it can't reach it (the testing skill). Under a
#: recorded check (`VERITY_STORE_REQUIRED=1`, which `check.py` gives its steps) that skip is a failure, so a check never passes
#: without the tests that need the store.
STORE_UNREACHABLE = "the evidence store is not reachable"
_STORE_REQUIRED = os.environ.get("VERITY_STORE_REQUIRED") == "1"
_KEY = "verity_suite_guard"
_REPORT = os.environ.get("VERITY_SUITE_REPORT")
_ACTIVE = bool(os.environ.get("VERITY_SUITE_ROOT") and _REPORT)

_root = os.path.realpath(os.environ.get("VERITY_SUITE_ROOT", ""))
_watch = frozenset(os.path.normpath(os.path.join(_root, p)) for p in json.loads(os.environ.get("VERITY_SUITE_WATCH", "[]")))
#: file read outside the inputs (a cached module as its source) -> the test module reading when it was first opened ("*": a
#: conftest or session fixture), so a failure names what was read and by whom, and drive can attribute it to a PR
_outside: dict[str, str] = {}
_watched: set[str] = set()
_tried: set[str] = set()
_record: dict[str, set] = {"distributions": set(), "missing": set()}      # from workers, in the controller
_unguarded: list[str] = []
_counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
_slow: list[tuple[str, float]] = []
_suite = os.path.join(os.path.realpath(os.getcwd()), "")         # suites.py runs pytest from the suite's directory
_current = "*"                                                   # the test module being collected or run
_test = ""                                                       # the node id of the test running, if one is
_reads: dict[str, set[str]] = {}                                 # test module (or "*") -> other test modules it opened
#: node id -> seconds of its setup and call, None once a phase did not pass. Setup counts: a module-scoped fixture's setup goes
#: to the first test that needs it, and ordering by call alone started those late (a module with 138 s of calls behind
#: 1,270-1,475 s of fixture setup per worker). A test enters only once its call passed: one stopped mid-call never counts
_outcomes: dict[str, float | None] = {}
_setup_s: dict[str, float] = {}                                  # node id -> seconds of its passed setup, until its call reports
_failed: set[str] = set()
_traced: set[str] = set()                                        # "F<tab>file" / "L<tab>directory" this process read
_untraced: dict[str, set[str]] = {}                              # test module (or "*") -> programs no tracer sees into
_workers_untraced: dict[str, set[str]] | None = None             # in an xdist controller: its workers', which start its tests' programs
_pid = os.getpid()


def _child_reads():
    spec = importlib.util.spec_from_file_location("verity_suite_reads", os.path.join(os.path.dirname(os.path.abspath(__file__)), "child",
                                                                                     "verity_suite_reads.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_reads_mod = _child_reads() if _ACTIVE else None
_forked = _reads_mod.Reads() if _reads_mod is not None and os.environ.get("VERITY_SUITE_READS") else None
_tracer = _reads_mod.Reads() if _reads_mod is not None else None


def _audit(event: str, args: tuple) -> None:
    if os.getpid() != _pid:                          # a fork of this process: what it reads must reach the file now
        if _forked is not None:
            _forked.seen(event, args, _current, _test)
        return
    try:
        hits = _tracer.trace(event, args) if _tracer is not None else []
    except (TypeError, ValueError):
        hits = []
    for kind, rel in hits:
        if kind == "X":
            _untraced.setdefault(_current, set()).add(rel)
        elif kind == "O":
            _outside.setdefault(rel, _current)
        else:
            _traced.add(f"{kind}\t{rel}")
    if event == "import":
        if args and isinstance(args[0], str):
            _tried.add(args[0].partition(".")[0])
        return
    if event != "open" or not args or isinstance(args[0], int):
        return
    try:
        path = os.fsdecode(args[0])
    except TypeError:
        return
    path = os.path.normpath(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))
    if path in _watch:
        _watched.add(os.path.relpath(path, _root))
    if _tracer.outside(path):
        _outside.setdefault(os.path.relpath(_reads_mod._source(path), _root), _current)
    name = os.path.basename(path)
    if path.startswith(_suite) and name.startswith("test_") and name.endswith(".py"):
        rel = os.path.relpath(path, _root)
        if rel != _current:
            _reads.setdefault(_current, set()).add(rel)


def _site() -> tuple[str, ...]:
    return tuple({os.path.join(os.path.realpath(sysconfig.get_paths()[k]), "") for k in ("purelib", "platlib")})


def _used() -> dict[str, list[str]]:
    """The installed distributions this process imported, or tried to import, modules of; and the top-level imports that
    failed with no installed distribution providing them."""
    site = _site()
    top = {n.partition(".")[0] for n, m in list(sys.modules.items())
           if isinstance(getattr(m, "__file__", None), str) and os.path.realpath(m.__file__).startswith(site)}
    owners = importlib.metadata.packages_distributions()
    failed = {n for n in _tried if n and n not in sys.modules}
    return {"distributions": sorted({d for t in top | (failed & owners.keys()) for d in owners.get(t, ())}),
            "missing": sorted(failed - owners.keys())}


if _ACTIVE:
    sys.addaudithook(_audit)


def _module_of(path) -> str:
    return os.path.relpath(os.path.realpath(str(path)), _root)


def _enter(module: str) -> str:
    """Make ``module`` the one whose tests are running, here and in every child started meanwhile; the one before it."""
    global _current
    before, _current = _current, module
    if _ACTIVE:
        os.environ["VERITY_SUITE_MODULE"] = module
    return before


def _leave(before: str) -> None:
    global _current
    _current = before
    if not _ACTIVE:
        return
    if before == "*":
        os.environ.pop("VERITY_SUITE_MODULE", None)
    else:
        os.environ["VERITY_SUITE_MODULE"] = before


#: the machine's own files a test would read in place of its fixtures (variable: what it would read), each pointed at a path in
#: the test's tmp dir that doesn't exist: cpu-1's run-via-slot rule, a Nebius host's CPU and GPU split, its peers and its lease
#: deadline, and a slotted machine's slots file
HOST_FILES = {"RUNS_VIA_SLOT_FILE": "/etc/vy/runs-via-slot", "DIRECT_CPUS_FILE": "/etc/vy/direct-cpus",
              "GPU_LEASE_ALLOWED_FILE": "/etc/vy/direct-gpus", "CHECK_PEERS": "/etc/vy/peers",
              "LEASE_DEADLINE_FILE": "/etc/research/deadline", "CHECK_SLOTS": "/workspace/research/locks/slots"}
#: the check's caches (variable: default), each an empty directory of the test's own, so no test reads a pass, a report or a
#: build some earlier check on the machine kept (node 1's real caches failed test_placement and test_shards)
HOST_CACHES = {"CHECK_CACHE": "~/.cache/verity-check", "VERITY_TEST_CACHE": "~/.cache/verity/tests"}
#: what a check slot gives the processes it runs (slot.py), which a test run inside one would otherwise take as its own
HOST_SLOT = ("CHECK_SLOT",)


def _without_host(d: str, env=os.environ) -> None:
    for k in HOST_FILES:
        env[k] = os.path.join(d, f"absent-{k.lower()}")
    for k in HOST_CACHES:
        os.makedirs(os.path.join(d, k.lower()), exist_ok=True)
        env[k] = os.path.join(d, k.lower())
    for k in HOST_SLOT:
        env.pop(k, None)


if _ACTIVE:
    # before any test module is imported: a module reads some of these into constants then (slot.py's PEERS, suites.py's CACHE)
    _host = tempfile.mkdtemp(prefix="verity-suite-host-")
    atexit.register(shutil.rmtree, _host, True)
    _without_host(_host)


@pytest.fixture(autouse=True)
def _host_isolated(monkeypatch, tmp_path_factory):
    """Every test in every suite starts without the machine it runs on: `HOST_FILES` absent, `HOST_CACHES` empty and `HOST_SLOT`
    unset, here and in every program it starts; a test or conftest that wants one sets its own. Which cores the process may use
    can't be set here, so the quick tier runs the suites off the lowest one, under a group-writable umask (`quick_tier.confined`)."""
    env: dict[str, str] = {}
    _without_host(str(tmp_path_factory.mktemp("host")), env)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for k in HOST_SLOT:
        monkeypatch.delenv(k, raising=False)


#: under `suites.py --changed --quick`, the tests of each test file `WHOLE_OPT` names (from the suite's directory) carry
#: `WHOLE_MARK`, which that run's `-m` keeps beside the tests not marked slow (`suites.whole_args`)
WHOLE_MARK, WHOLE_OPT = "verity_whole", "--verity-whole"
_whole: frozenset = frozenset()


def pytest_addoption(parser) -> None:
    parser.addoption(WHOLE_OPT, action="append", default=[], metavar="FILE",
                     help=f"mark this test file's tests {WHOLE_MARK} (suites.py --changed --quick runs them whole)")


def pytest_configure(config) -> None:
    global _whole
    _whole = frozenset(config.getoption(WHOLE_OPT))
    config.addinivalue_line("markers", f"{WHOLE_MARK}: a test of a file the change reaches, run with its slow tests")


def pytest_itemcollected(item) -> None:
    if _whole and os.path.relpath(os.path.realpath(item.path), _suite).replace(os.sep, "/") in _whole:
        item.add_marker(WHOLE_MARK)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """Opens while a module is collected (its import) are that module's."""
    if not isinstance(collector, pytest.Module):
        yield
        return
    before = _enter(_module_of(collector.path))
    try:
        yield
    finally:
        _leave(before)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _test
    before, _test = _enter(_module_of(item.path)), item.nodeid
    if _ACTIVE:
        os.environ["VERITY_SUITE_TEST"] = item.nodeid
    try:
        yield
    finally:
        _test = ""
        if _ACTIVE:
            os.environ.pop("VERITY_SUITE_TEST", None)
        _leave(before)


def heavy_first(nodeids: list[str], heavy: list[str], workers: int) -> list[int]:
    """The order (indices into ``nodeids``) that starts the first ``workers`` of the ``heavy`` tests (longest first) found in
    ``nodeids`` each at the head of its own worker's first share under xdist's `worksteal`, the other tests in collection order. `worksteal`
    splits the collection into contiguous shares, one per worker, and steals from a share's tail, so a long test deep in a share
    starts late and ends the run late."""
    starts, at = [], 0
    for i in range(workers):
        n = (len(nodeids) - at) // (workers - i)
        if n:
            starts.append(at)
        at += n
    where = {n: i for i, n in enumerate(nodeids)}
    first = [where[n] for n in dict.fromkeys(heavy) if n in where][:len(starts)]
    placed = set(first)
    rest = iter(i for i in range(len(nodeids)) if i not in placed)
    heads = dict(zip(starts, first))
    return [heads[p] if p in heads else next(rest) for p in range(len(nodeids))]


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items) -> None:
    """Deselect the tests of the other parts of a part by test id (`id_part`), then those `VERITY_SUITE_REUSE` names; in an xdist
    worker, start the longest of those left at the heads of the workers' shares (`VERITY_SUITE_HEAVY`, longest first; every
    worker orders its collection alike, as xdist requires)."""
    global _part_out
    if _id_part is not None and (out := [i for i in items if id_part(i.nodeid, _id_part[1]) not in _id_part[0]]):
        _part_out = len(out)
        config.hook.pytest_deselected(items=out)
        items[:] = [i for i in items if id_part(i.nodeid, _id_part[1]) in _id_part[0]]
    if reuse := os.environ.get("VERITY_SUITE_REUSE"):
        with open(reuse) as f:
            ids = set(json.load(f))
        gone = [i for i in items if i.nodeid in ids]
        if gone:
            config.hook.pytest_deselected(items=gone)
            items[:] = [i for i in items if i.nodeid not in ids]
    workers = (getattr(config, "workerinput", None) or {}).get("workercount", 1)
    if (heavy := os.environ.get("VERITY_SUITE_HEAVY")) and workers > 1:
        with open(heavy) as f:
            order = heavy_first([i.nodeid for i in items], json.load(f), workers)
        items[:] = [items[i] for i in order]


def _load_part() -> dict | None:
    if not (_ACTIVE and (f := os.environ.get("VERITY_SUITE_PART"))):
        return None
    with open(f) as fh:
        return json.load(fh)


_part_doc = _load_part()
#: a part by test file: (the suite's directory, `in_part`'s arguments)
_part = ((_part_doc["dir"], (set(_part_doc["ks"]), _part_doc["m"], frozenset(_part_doc["mine"]), frozenset(_part_doc["every"])))
         if _part_doc is not None and _part_doc.get("by") != "id" else None)
#: a part by test id: (its ks, m)
_id_part = (set(_part_doc["ks"]), _part_doc["m"]) if _part_doc is not None and _part_doc.get("by") == "id" else None
_part_out = 0                                                    # the collected tests of the other parts by id, deselected


def id_part(nodeid: str, m: int) -> int:
    """The part, 1..``m``, of a suite in parts by test id (`suites.id_part_of`) that runs test ``nodeid`` (from the suite's
    directory): by its sha256, as circuit-check shards its targets, so every test the suite's collection finds runs in exactly
    one part on any machine."""
    return 1 + int.from_bytes(hashlib.sha256(nodeid.encode()).digest()[:8], "big") % m


def in_part(rel: str, ks: set[int], m: int, mine: frozenset, every: frozenset) -> bool:
    """Whether test file ``rel`` (from the suite's directory) is in parts ``ks`` of ``m``: by `suites.partition`'s assignment
    (``mine`` of ``every``), and a file pytest collects that it doesn't list by its path's hash, so every file the suite's own
    collection finds runs in exactly one part."""
    if rel in every:
        return rel in mine
    return int(hashlib.sha256(rel.encode()).hexdigest()[:8], 16) % m + 1 in ks


def pytest_ignore_collect(collection_path, config):
    """Under `suites.py --files` (`VERITY_SUITE_PART`): a test file outside the run's part by file isn't collected, nor imported
    (a part by test id collects every file)."""
    name = collection_path.name
    if _part is None or not name.endswith(".py") or not (name.startswith("test_") or name.endswith("_test.py")):
        return None
    rel = os.path.relpath(collection_path, _part[0]).replace(os.sep, "/")
    return None if in_part(rel, *_part[1]) else True


def _ours(name: str) -> bool:
    """Whether module ``name``'s top-level package is this checkout's: it resolves inside `VERITY_SUITE_ROOT` and outside
    site-packages, which a workspace's venv may put inside the tree."""
    try:
        spec = importlib.util.find_spec(name.partition(".")[0])
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    where = [*(spec.submodule_search_locations or ()), *([spec.origin] if spec.has_location else [])]
    root, site = os.path.join(_root, ""), _site()
    return any((p := os.path.realpath(w)).startswith(root) and not p.startswith(site) for w in where)


_pytest_importorskip = pytest.importorskip


def importorskip(modname: str, minversion: str | None = None, reason: str | None = None, *, exc_type=None):
    """`pytest.importorskip`, failing where the module that is missing is this checkout's (`_ours`). A skip is for what an install
    may lack (torch, vllm, a GPU library); a module of ours that isn't there has moved or gone, and pytest skips on exactly the
    `ModuleNotFoundError` a move raises, so the tests of a moved module would pass as skips. A module of ours that needs a missing
    third-party package still skips."""
    try:
        importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name and _ours(e.name):
            pytest.fail(f"pytest.importorskip({modname!r}): {e.name} is not importable, and {e.name.partition('.')[0]} is this "
                        "checkout's, so it moved or is gone. Import it from where it is now, plainly: importorskip is for a "
                        "third-party package an install may lack", pytrace=False)
    except Exception:  # noqa: BLE001  (any other failure is pytest's importorskip's to report, below)
        pass
    return _pytest_importorskip(modname, minversion, reason, exc_type=exc_type)


if _ACTIVE:
    pytest.importorskip = importorskip


#: how to read a value the guard withholds from a failure's text
SEE_VALUES = "to see them, run the test alone in an emptied environment"
_Environ = type(os.environ)
_environ_repr = _Environ.__repr__


def _environ_names(env) -> str:
    """`os.environ`'s repr while a suite runs: its names and no value. It reads the names as the repr it stands in for does,
    from the mapping's own table, so a subclass that records each read of the environment records none here."""
    names = sorted(os.fsdecode(k) for k in getattr(env, "_data", env))
    return f"<os.environ: values withheld ({SEE_VALUES}); names: {' '.join(names)}>"


if _ACTIVE:
    _Environ.__repr__ = _environ_names             # as the module loads, so an assert in a conftest's import prints no value either


def pytest_unconfigure(config) -> None:
    """The session is over: the environment's repr is the one this module found, for a process that goes on after
    `pytest.main` returns. A fork the tests made keeps the names-only one for as long as it lives."""
    if _Environ.__repr__ is _environ_names:
        _Environ.__repr__ = _environ_repr


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Under `VERITY_STORE_REQUIRED=1`, a skip whose reason says the store is unreachable (`STORE_UNREACHABLE`) fails instead."""
    report = (yield).get_result()
    if not (_STORE_REQUIRED and report.skipped) or getattr(report, "wasxfail", None) is not None:
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    if STORE_UNREACHABLE in reason:
        report.outcome = "failed"
        report.longrepr = (f"{reason}\nVERITY_STORE_REQUIRED=1 (a recorded check): a test that needs the evidence store fails "
                           "where it can't reach it, rather than skip")


def pytest_runtest_logreport(report) -> None:
    if report.outcome != "passed" or getattr(report, "wasxfail", None) is not None:
        _outcomes[report.nodeid] = None
        if report.outcome == "failed":
            _failed.add(report.nodeid)
    elif report.when == "setup":
        _setup_s[report.nodeid] = report.duration
    elif report.when == "call" and _outcomes.get(report.nodeid, 0.0) is not None:
        _outcomes[report.nodeid] = _setup_s.pop(report.nodeid, 0.0) + report.duration
    if report.when == "call":
        _counts[report.outcome] += 1
        if report.duration > SLOW_S and "slow" not in report.keywords:
            _slow.append((report.nodeid, round(report.duration, 1)))
    elif report.outcome == "failed":
        _counts["error"] += 1
    elif report.outcome == "skipped" and report.when == "setup":
        _counts["skipped"] += 1


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    """In the controller, for each xdist worker as it finishes: take the files it read outside the inputs."""
    global _workers_untraced, _part_out
    record = (getattr(node, "workeroutput", None) or {}).get(_KEY)
    if record is None:
        _unguarded.append(node.gateway.id)
    else:
        _part_out = max(_part_out, record.get("part_out", 0))
        for p, m in record["outside"].items():
            _outside.setdefault(p, m)
        _watched.update(record["watched"])
        for k in _record:
            _record[k].update(record[k])
        for m, files in record.get("reads", {}).items():
            _reads.setdefault(m, set()).update(files)
        _traced.update(record.get("traced", []))
        _workers_untraced = _workers_untraced or {}
        for m, progs in record.get("untraced", {}).items():
            _workers_untraced.setdefault(m, set()).update(progs)


def pytest_sessionfinish(session, exitstatus) -> None:
    if not _ACTIVE:
        return
    config = session.config
    used = _used()
    reads = {m: sorted(files) for m, files in sorted(_reads.items())}
    if hasattr(config, "workerinput"):
        config.workeroutput[_KEY] = {"outside": dict(sorted(_outside.items())), "watched": sorted(_watched), "reads": reads, **used,
                                     "traced": sorted(_traced), "untraced": {m: sorted(v) for m, v in sorted(_untraced.items())},
                                     "part_out": _part_out}
        return
    used = {k: sorted(set(v) | _record[k]) for k, v in used.items()}
    passed = {n: round(s, 3) for n, s in sorted(_outcomes.items()) if s is not None}
    untraced = _untraced if _workers_untraced is None else _workers_untraced     # a controller's own programs are its workers
    payload = json.dumps({"counts": _counts, "outside": dict(sorted(_outside.items())), "unguarded": _unguarded, "unmarked_slow": _slow,
                          "watched": sorted(_watched), "passed": passed, "failed": sorted(_failed), "reads": reads, **used,
                          "traced": sorted(_traced), "untraced": {m: sorted(v) for m, v in sorted(untraced.items())},
                          "part_out": _part_out},
                         indent=1)
    with open(_REPORT, "w") as f:
        f.write(payload)
