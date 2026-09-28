"""The pytest plugin `suites.py` loads into every suite it runs (`-p suite_guard`).

It records every repository file the tests open outside the suite's inputs (`VERITY_SUITE_ALLOW`, paths relative to
`VERITY_SUITE_ROOT`), every test whose call took over 5 s without the `slow` mark, and the outcome counts, and writes them to
`VERITY_SUITE_REPORT` when the session ends. `suites.py` fails a suite that read outside its inputs: a cached pass is only
sound when everything the tests read is part of the key. Opens go through Python's audit hook (`open` events), which sees
every file Python opens, imports included, but nothing a subprocess opens.
"""
from __future__ import annotations

import json
import os
import sys

SLOW_S = 5.0

_root = os.path.realpath(os.environ.get("VERITY_SUITE_ROOT", ""))
_paths = [os.path.normpath(os.path.join(_root, p)) for p in json.loads(os.environ.get("VERITY_SUITE_ALLOW", "[]"))]
_allowed = tuple(os.path.join(p, "") for p in _paths if os.path.isdir(p))
_allowed += tuple({os.path.join(os.path.realpath(sys.prefix), ""), os.path.join(os.path.abspath(sys.prefix), "")})
_allowed_files = frozenset(p for p in _paths if not os.path.isdir(p))
_inside = os.path.join(_root, "")
_outside: set[str] = set()
_counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
_slow: list[tuple[str, float]] = []


def _audit(event: str, args: tuple) -> None:
    if event != "open" or not args or isinstance(args[0], int):
        return
    try:
        path = os.fsdecode(args[0])
    except TypeError:
        return
    path = os.path.normpath(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))
    if path.startswith(_inside) and not path.startswith(_allowed) and path not in _allowed_files:
        _outside.add(os.path.relpath(path, _root))


def pytest_configure(config) -> None:
    if _root and os.environ.get("VERITY_SUITE_REPORT"):
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


def pytest_sessionfinish(session, exitstatus) -> None:
    path = os.environ.get("VERITY_SUITE_REPORT")
    if path:
        payload = json.dumps({"counts": _counts, "outside": sorted(_outside), "unmarked_slow": _slow}, indent=1)
        with open(path, "w") as f:
            f.write(payload)
