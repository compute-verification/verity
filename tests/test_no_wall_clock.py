"""No test depends on wall-clock time (AGENTS.md "Tests"). `check` runs every suite at once on a loaded pod, so a test that
sleeps, compares clock readings, or gives up after a timeout passes or fails with the scheduler. An AST scan over every
suite's test tree rejects them: wait on the event itself (a process's exit, a pipe or FIFO the other side writes or holds,
a line of its output), inject the clock or the timer into the code that measures time, and give a fake server an event
to block on instead of a sleep.

What may stay is listed in ``ALLOWED``, one line of reason each: a whole file (``path``) or one top-level function, class
or assigned name in it (``path::name``). An entry that no longer matches anything fails too.

The same holds for build order: a test may run or skip on what `check` builds concurrently (the Lean toolchain and the Lake
build outputs) only in a suite that `check` runs after those steps (`test_only_the_suites_check_runs_after_its_lean_steps_...`)."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import tomllib

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
    "benchmarks/numerical/tests/reference/exhaustive_driver.py": "the WP1 pod driver (a recorded run, not a test) reports its steps' wall time",
    "integrations/vllm/tests/commit/test_commit_fail_closed.py::_WATCHDOG_PROG":
        "drives the native collector's dead-watchdog, a C++ timer with no injectable clock; CUDA only, never in check",
    "integrations/vllm/tests/engine/int8_decode_split_gpu.py":
        "a GPU probe script (not collected) that measures the int8 LM head's share of decode time; no test reads it",
    "integrations/vllm/tests/engine/int8_attention_tpot_gpu.py::generate":
        "a GPU timing script (not collected) that measures int8_exact decode TPOT, batched against the reference; no test reads it",
    "integrations/vllm/tests/program/beyond_gemm.py": "a GPU probe script (not collected) that reports its own wall time",
    "integrations/vllm/tests/program/crosscheck.py::main": "the cross-check script's summary line reports its wall time; no test runs main",
    "integrations/vllm/tests/protocols/local_gate.py::_now":
        "the by-hand local gate (not collected) gives infra's auditor.py the times its signed entries record, as on vy-challenger-1",
    "integrations/vllm/tests/protocols/pouw_h1_rehearsal.py::_poll_end":
        "the by-hand live rehearsal (not collected) polls the live gate's status for a session's end entry: it has no wait verb",
    "integrations/vllm/tests/protocols/pous_gpu.py":
        "the PoUS option's GPU smoke (not collected) records its load, install, decode and step seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_gpu.py::run": "the GPU run records its install, serve and compile seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_gpu.py::compile_from": "the host compile records its seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_gpu.py::fill": "the GPU run records its serve and step seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_gpu.py::period_program":
        "the GPU run records the period Program's declare and digest seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_gpu.py::await_end":
        "the live run polls the live gate's status for a window's session end entry before reserving the next: it has no wait verb",
    "integrations/vllm/tests/protocols/pouw_h1_gpu.py::await_listener":
        "the live run waits for the challenger's session port to listen (ss -ltn) once a second; a probe connect would take its one client",
    "integrations/vllm/tests/protocols/pouw_h1_gpu.py::served_measures":
        "the GPU run records its step seconds and when serving ended; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_gpu.py::hand_off": "the GPU run records its suspend's seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_gpu.py::fill_close": "the close stage records its seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_gpu.py::credit_check": "the credit check records its seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_ranks_gpu.py::serve":
        "the GPU run records its build, serve and release seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_ranks_gpu.py::commit_window":
        "the GPU run records each Commit window's install, serve and release seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_rehearsal.py::build_program":
        "the by-hand rehearsal (not collected) records the declared Program's build and encode seconds; nothing reads them",
    "integrations/vllm/tests/protocols/pouw_h1_rehearsal.py::serve":
        "the by-hand rehearsal (not collected) records when each stage exited, how long the GPU lease outlived serving",
    "integrations/vllm/tests/properties/router_tap_exactness_gpu.py::check_config": "the GPU run records the IR pass's seconds; nothing reads them",
    "integrations/vllm/tests/properties/router_tap_exactness_gpu.py::check_live": "the GPU run records the IR pass's seconds; nothing reads them",
    "tests/test_no_wall_clock.py::test_the_scan_sees_each_form": "the scan's own fixture: source it must flag, held in a string",
    "tools/verity/circuit_check/tests/fixtures/ambient_builders.py::_clocked":
        "a builder that reads a clock so that circuit-check --ambient refuses it; nothing compares or waits on the reading",
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


def lean_gates(source: str) -> list[int]:
    """The lines of skip conditions (`skipif(...)`, `if ...: pytest.skip(...)`) that probe what `check`'s Lean group builds or
    installs while the other suites run: elan's `lake` (a "lake" string, as in `shutil.which("lake")`) or a Lake build output (a
    path through `.lake`), in the condition itself or in a function or module-level name it uses."""
    tree = ast.parse(source)
    defs: dict[str, ast.AST] = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    defs |= {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}

    def probes(node: ast.AST, seen: frozenset[str] = frozenset()) -> bool:
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and (n.value == "lake" or ".lake" in n.value):
                return True
            if isinstance(n, ast.Name) and n.id in defs and n.id not in seen and probes(defs[n.id], seen | {n.id}):
                return True
        return False

    lines = []
    for n in ast.walk(tree):
        cond = None
        if isinstance(n, ast.Call) and _dotted(n.func).endswith("skipif"):
            cond = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "condition"), None)
        elif isinstance(n, ast.If) and any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                                           and _dotted(s.value.func) == "pytest.skip" for s in n.body):
            cond = n.test
        if cond is not None and probes(cond):
            lines.append(n.lineno)
    return sorted(lines)


def _test_sources() -> list[str]:
    tracked = _tracked()
    testpaths = _testpaths(tracked)
    return [p for p in tracked if p.endswith(".py") and any(t == "." or p.startswith(t + "/") for t in testpaths)]


def test_only_the_suites_check_runs_after_its_lean_steps_skip_on_lean() -> None:
    """`check` builds the Lean verifier and every Lake package, and installs elan when `lake` is missing, while the pytest step
    runs: a test elsewhere that runs or skips on `lake` or a `.lake/` build output passes or fails with the build order.
    Such tests belong to a suite `check` runs after its Lean steps (`AFTER_LEAN`), and
    that suite declares the build outputs it uses under `[tool.verity.tests] artifacts`, which key its cached pass."""
    spec = importlib.util.spec_from_file_location("repo_check", ROOT / "tools" / "verity" / "check" / "check.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    names = {}
    for p in _tracked():
        if os.path.basename(p) == "pyproject.toml":
            data = tomllib.loads((ROOT / p).read_text())
            if "pytest" in data.get("tool", {}):
                name = re.sub(r"[-_.]+", "-", data["project"]["name"]).lower() if "project" in data else os.path.dirname(p)
                names[name] = (os.path.dirname(p), data)
    after = [names[s][0] + "/" for s in check.AFTER_LEAN]
    offenders = [f"{path}:{line}" for path in _test_sources() if not path.startswith(tuple(after))
                 for line in lean_gates((ROOT / path).read_text(encoding="utf-8"))]
    assert not offenders, "tests that skip on what check's Lean steps build, outside the suites it runs after them:\n" + "\n".join(offenders)
    flock = names["verity-flock"][1]["tool"]["verity"]["tests"]
    assert "verity/core/service/flock/.lake/build/bin/flock-verify" in flock.get("artifacts", []), "verity-flock runs the verifier"


def test_the_lean_gate_scan_sees_each_form() -> None:
    src = '''
import shutil, pytest
from pathlib import Path
BIN = Path("lean") / ".lake" / "build" / "bin" / "v"

def _built():
    return BIN.exists()

@pytest.mark.skipif(shutil.which("lake") is None, reason="x")
def test_a():
    pass

@pytest.mark.skipif(not _built(), reason="x")
def test_b():
    pass

def test_c():
    if not (Path("p") / ".lake").is_dir():
        pytest.skip("x")

@pytest.mark.skipif(shutil.which("cargo") is None, reason="x")
def test_d():
    pass
'''
    assert lean_gates(src) == [9, 13, 18]


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
