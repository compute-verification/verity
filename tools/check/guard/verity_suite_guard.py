"""The pytest plugin `suites.py` loads into every suite it runs (`-p verity_suite_guard`, from this directory, which holds
nothing else, so putting it on `sys.path` shadows no module).

It records every repository file the tests open outside the suite's inputs (`VERITY_SUITE_ALLOW`, paths relative to
`VERITY_SUITE_ROOT`), every test whose call took over 5 s without the `slow` mark, and the outcome counts, and writes them to
`VERITY_SUITE_REPORT` when the session ends. `suites.py` fails a suite that read outside its inputs or wrote no report: a
cached pass is only sound when everything the tests read is part of the key. Opens go through Python's audit hook (`open`
events), installed when pytest loads this module, before any conftest: it sees every file Python opens, imports included, but
nothing a subprocess opens.

It also records what `suites.py` keeps beside a pass and checks before reusing it: which of the metadata files in
`VERITY_SUITE_WATCH` the tests opened, the installed distributions whose modules were imported (from site-packages), and the
top-level imports that failed with no installed distribution providing the name (`import` events).

Under pytest-xdist every worker loads it too, since xdist gives each worker the controller's command line and `sys.path`.
A worker sends its record to the controller when its session ends (`workeroutput`), and the controller counts outcomes from
the reports workers send it. A worker that sends no record fails the suite (`unguarded`).
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import sysconfig

import pytest

SLOW_S = 5.0
_KEY = "verity_suite_guard"
_ACTIVE = bool(os.environ.get("VERITY_SUITE_ROOT") and os.environ.get("VERITY_SUITE_REPORT"))

_root = os.path.realpath(os.environ.get("VERITY_SUITE_ROOT", ""))
_paths = [os.path.normpath(os.path.join(_root, p)) for p in json.loads(os.environ.get("VERITY_SUITE_ALLOW", "[]"))]
_allowed = tuple(os.path.join(p, "") for p in _paths if os.path.isdir(p))
_allowed += tuple({os.path.join(os.path.realpath(sys.prefix), ""), os.path.join(os.path.abspath(sys.prefix), "")})
_allowed_files = frozenset(p for p in _paths if not os.path.isdir(p))
_inside = os.path.join(_root, "")
_watch = frozenset(os.path.normpath(os.path.join(_root, p)) for p in json.loads(os.environ.get("VERITY_SUITE_WATCH", "[]")))
_outside: set[str] = set()
_watched: set[str] = set()
_tried: set[str] = set()
_record: dict[str, set] = {"distributions": set(), "missing": set()}      # from workers, in the controller
_unguarded: list[str] = []
_counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
_slow: list[tuple[str, float]] = []


def _audit(event: str, args: tuple) -> None:
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
    if path.startswith(_inside) and not path.startswith(_allowed) and path not in _allowed_files:
        _outside.add(os.path.relpath(path, _root))


def _used() -> dict[str, list[str]]:
    """The installed distributions this process imported, or tried to import, modules of; and the top-level imports that
    failed with no installed distribution providing them."""
    site = tuple({os.path.join(os.path.realpath(sysconfig.get_paths()[k]), "") for k in ("purelib", "platlib")})
    top = {n.partition(".")[0] for n, m in list(sys.modules.items())
           if isinstance(getattr(m, "__file__", None), str) and os.path.realpath(m.__file__).startswith(site)}
    owners = importlib.metadata.packages_distributions()
    failed = {n for n in _tried if n and n not in sys.modules}
    return {"distributions": sorted({d for t in top | (failed & owners.keys()) for d in owners.get(t, ())}),
            "missing": sorted(failed - owners.keys())}


if _ACTIVE:
    sys.addaudithook(_audit)


def pytest_runtest_logreport(report) -> None:
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
    record = (getattr(node, "workeroutput", None) or {}).get(_KEY)
    if record is None:
        _unguarded.append(node.gateway.id)
    else:
        _outside.update(record["outside"])
        _watched.update(record["watched"])
        for k in _record:
            _record[k].update(record[k])


def pytest_sessionfinish(session, exitstatus) -> None:
    if not _ACTIVE:
        return
    config = session.config
    used = _used()
    if hasattr(config, "workerinput"):
        config.workeroutput[_KEY] = {"outside": sorted(_outside), "watched": sorted(_watched), **used}
        return
    used = {k: sorted(set(v) | _record[k]) for k, v in used.items()}
    payload = json.dumps({"counts": _counts, "outside": sorted(_outside), "unguarded": _unguarded, "unmarked_slow": _slow,
                          "watched": sorted(_watched), **used}, indent=1)
    with open(os.environ["VERITY_SUITE_REPORT"], "w") as f:
        f.write(payload)
