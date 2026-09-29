"""No test depends on wall-clock time (AGENTS.md "Tests"). `check` runs every suite at once on a loaded pod, so a test that
sleeps, compares clock readings, or gives up after a timeout passes or fails with the scheduler. An AST scan over every
suite's test tree rejects them: wait on the event itself (a process's exit, a pipe or FIFO the other side writes or holds,
a line of its output), inject the clock or the timer into the code that measures time, and give a fake server an event
to block on instead of a sleep.

What may stay is listed in ``ALLOWED``, one line of reason each: a whole file (``path``) or one top-level function, class
or assigned name in it (``path::name``). An entry that no longer matches anything fails too."""

from __future__ import annotations

import ast
import re

from test_repository import ROOT, _testpaths, _tracked

#: What each module offers that reads a clock or waits on one.
CLOCKS = {
    "time": {"sleep", "time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns", "process_time",
             "process_time_ns", "thread_time", "thread_time_ns", "clock_gettime", "clock_gettime_ns"},
    "threading": {"Timer"},
    "signal": {"alarm", "setitimer"},
    "asyncio": {"sleep", "wait_for"},
}
#: `datetime.now()`, `dt.datetime.utcnow()`, `date.today()`: the attribute, on anything named `datetime` or `date`.
NOW = {"now", "utcnow", "today"}
#: Python source inside a string (a `python -c` workload) that sleeps, or a `sleep` command in an argv: the test races the
#: workload. Block the workload on a FIFO or a pipe the test holds instead.
SLEEP_IN_SOURCE = re.compile(r"\bsleep\(")

ALLOWED: dict[str, str] = {
    "backends/numerical/tests/reference/exhaustive_driver.py": "the WP1 pod driver (a recorded run, not a test) reports its steps' wall time",
    "integrations/vllm/tests/check/adversarial.py::run_checkers": "the mutation harness reports each run's seconds; no test reads them",
    "integrations/vllm/tests/check/adversarial.py::run_one": "the mutation harness reports each fold's seconds; no test reads them",
    "integrations/vllm/tests/commit/test_commit_fail_closed.py::_WATCHDOG_PROG":
        "drives the native collector's dead-watchdog, a C++ timer with no injectable clock; CUDA only, never in check",
    "integrations/vllm/tests/program/beyond_gemm.py": "a GPU probe script (not collected) that reports its own wall time",
    "integrations/vllm/tests/program/crosscheck.py::main": "the cross-check script's summary line reports its wall time; no test runs main",
    "integrations/vllm/tests/program/padded_commit_tiny.py": "records build/commit timings in its row; test_padded_commit_tiny asserts none of them",
    "integrations/vllm/tests/properties/router_tap_exactness_gpu.py::check_config": "the GPU run records the IR pass's seconds; nothing reads them",
    "integrations/vllm/tests/properties/router_tap_exactness_gpu.py::check_live": "the GPU run records the IR pass's seconds; nothing reads them",
    "tests/test_no_wall_clock.py::test_the_scan_sees_each_form": "the scan's own fixture: source it must flag, held in a string",
}


def _dotted(node: ast.AST) -> str:
    return f"{_dotted(node.value)}.{node.attr}" if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else ""


def _number(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def offenses(source: str) -> list[tuple[str, int, str]]:
    """(top-level name or "<module>", line, what) for every use of real time in `source`."""
    tree = ast.parse(source)
    modules = {m: m for m in CLOCKS}           # local name -> module, by `import time` / `import time as t`
    names: dict[str, str] = {}                 # local name -> "time.sleep", by `from time import sleep`
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update({a.asname or a.name: a.name for a in node.names if a.name in CLOCKS})
        elif isinstance(node, ast.ImportFrom) and node.module in CLOCKS:
            names.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names if a.name in CLOCKS[node.module]})
    out = []
    for top in tree.body:
        where = (top.name if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 else top.targets[0].id if isinstance(top, ast.Assign) and isinstance(top.targets[0], ast.Name) else "<module>")
        for node in ast.walk(top):
            what = None
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in modules \
                    and node.attr in CLOCKS[modules[node.value.id]]:
                what = f"{modules[node.value.id]}.{node.attr}"
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
                what = names[node.id]
            elif isinstance(node, ast.Attribute) and node.attr in NOW and _dotted(node.value).split(".")[-1] in ("datetime", "date"):
                what = f"{_dotted(node)}"
            elif isinstance(node, ast.keyword) and node.arg == "timeout" and not isinstance(node.value, ast.Name) \
                    and not (isinstance(node.value, ast.Constant) and node.value.value is None):
                what = "a timeout= argument"               # a name is a fake passing the real signature's timeout through
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("wait", "join") \
                    and any(map(_number, node.args)):
                what = f".{node.func.attr}() with a timeout"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and SLEEP_IN_SOURCE.search(node.value):
                what = "a sleep in Python source held in a string"
            elif isinstance(node, (ast.List, ast.Tuple)) and any(isinstance(e, ast.Constant) and e.value == "sleep" for e in node.elts):
                what = "a sleep command"
            if what:
                out.append((where, getattr(node, "lineno", top.lineno), what))
    return sorted(out, key=lambda o: o[1])


def _test_sources() -> list[str]:
    tracked = _tracked()
    testpaths = _testpaths(tracked)
    return [p for p in tracked if p.endswith(".py") and any(t == "." or p.startswith(t + "/") for t in testpaths)]


def test_tests_do_not_depend_on_wall_clock_time() -> None:
    offenders, used = [], set()
    for path in _test_sources():
        for where, line, what in offenses((ROOT / path).read_text(encoding="utf-8")):
            key = next((k for k in (f"{path}::{where}", path) if k in ALLOWED), None)
            if key:
                used.add(key)
            else:
                offenders.append(f"{path}:{line} ({where}): {what}")
    assert not offenders, ("tests must not depend on wall-clock time: wait on the event, inject the clock (see this module's "
                           "docstring), or add an ALLOWED entry with its reason:\n" + "\n".join(offenders))
    assert not set(ALLOWED) - used, f"ALLOWED entries that match nothing any more: {sorted(set(ALLOWED) - used)}"


def test_the_scan_sees_each_form() -> None:
    src = '''
import time as t, datetime as dt
from time import monotonic as mono
import subprocess, threading
PROG = "import time; time.sleep(2)"

def f(p, e):
    t.sleep(1)
    mono()
    dt.datetime.now(dt.timezone.utc)
    subprocess.run(["x"], timeout=5)
    e.wait(0.5)
    threading.Timer(1, print)
    subprocess.run(["python", "-c", "import time; time.sleep(9)"])
    subprocess.Popen(["sleep", "4"])

def g(clock, p, monkeypatch, mod, real, timeout):
    clock.now += 5
    ", ".join(["a"])
    p.wait()
    real(p, timeout=timeout)
    p.communicate(timeout=None)
    t.strftime("%Y", t.gmtime(0))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
'''
    assert [(w, what) for w, _, what in offenses(src)] == [
        ("PROG", "a sleep in Python source held in a string"), ("f", "time.sleep"), ("f", "time.monotonic"), ("f", "dt.datetime.now"), ("f", "a timeout= argument"),
        ("f", ".wait() with a timeout"), ("f", "threading.Timer"), ("f", "a sleep in Python source held in a string"),
        ("f", "a sleep command")]
