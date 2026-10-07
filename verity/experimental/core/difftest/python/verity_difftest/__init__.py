"""The differential test of a Lean port against today's Python (`README.md` in this distribution's directory).

A `Port` names a function served by a Lean executable (`Difftest.serve`, `lean/Difftest.lean`), the Python reference it must
agree with, and a Hypothesis strategy of inputs, invalid ones included. `run` keeps one server process open, feeds both sides
every input the strategy draws, and fails with a `Mismatch` on the first difference, shrunk by Hypothesis to a minimal input:

* both give an output, and the outputs differ as canonical JSON (`canonical`: compact, keys sorted, integers only);
* one rejects and the other doesn't (reject parity: the reasons need not match);
* the reference raises something that is not one of the port's declared rejects, or the server faults or dies.

A run is deterministic: Hypothesis draws from `seed` (`derandomize`, no example database, no deadline), so `run(port,
examples=N, seed=S)` finds the same minimal input again, and `replay(port, value)` compares one input. Nothing here reads a
clock or times out (`tests/test_no_wall_clock.py`): a port whose Lean never answers hangs its test.

The harness never skips: when `lake` is missing or the build fails it raises. A port's test module skips on `lake` itself
(`pytestmark = pytest.mark.skipif(shutil.which("lake") is None, ...)`), which puts its suite among those `check` runs after
its Lean steps (`AFTER_LEAN` in `tools/verity/check/check.py`, held by `tests/test_no_wall_clock.py`).
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import seed as use_seed
from hypothesis.strategies import SearchStrategy


class Reject(Exception):
    """The reference refuses its input (the message is the reason). A port may declare other exception types as rejects."""


class ProtocolError(Exception):
    """The server broke the protocol (`lean/Difftest.lean`): an `error` response, a line that isn't canonical JSON, or no line."""


class BuildError(Exception):
    """`lake build` of a port's executable failed, or `lake` is not on the PATH."""


class Mismatch(AssertionError):
    """Lean and Python differ on `input` (the minimal one Hypothesis found): `lean` and `python` are each side's outcome,
    `("output", canonical text)`, `("reject", reason)` or `("error", message)`."""

    def __init__(self, fn: str, value: Any, lean: tuple[str, str], python: tuple[str, str]):
        self.fn, self.input, self.lean, self.python = fn, value, lean, python
        self.report = ""
        super().__init__(f"{fn}: Lean {_show(lean)}, Python {_show(python)} on {canonical(value)}")

    def __str__(self) -> str:
        return self.report or super().__str__()


class Vacuous(AssertionError):
    """A run passed without exercising both outcomes: no input gave an output, or (unless the port waives it) none was
    rejected. A strategy that only reaches one side of the port's domain checks nothing on the other."""


def canonical(value: Any) -> str:
    """The canonical JSON text of `value` (as `Flock.canon` writes it): compact, object keys sorted, strings in UTF-8 with
    only `"`, `\\` and control characters escaped. Its domain is null, booleans, integers, strings, lists (tuples) and objects
    with string keys; anything else, a float among them, raises `TypeError`."""
    stack = [value]
    while stack:
        x = stack.pop()
        if x is None or isinstance(x, (bool, int)):
            continue
        if isinstance(x, str):
            try:
                x.encode("utf-8")
            except UnicodeEncodeError as e:
                raise TypeError(f"not canonical JSON: a string that is not UTF-8 ({e.reason})") from None
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, dict):
            if not all(isinstance(k, str) for k in x):
                raise TypeError("not canonical JSON: an object key that is not a string")
            stack.extend(x)
            stack.extend(x.values())
        else:
            raise TypeError(f"not canonical JSON: a {type(x).__name__}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@functools.cache
def build(package: Path, exe: str) -> Path:
    """`lake build EXE` in the Lake package `package`, once per process; the built executable's path."""
    try:
        p = subprocess.run(["lake", "build", exe], cwd=package, capture_output=True, text=True)
    except FileNotFoundError:
        raise BuildError(f"lake is not on the PATH, so {exe} in {package} cannot be built") from None
    if p.returncode != 0:
        raise BuildError(f"lake build {exe} in {package} failed ({p.returncode}):\n{p.stdout}{p.stderr}")
    return package / ".lake" / "build" / "bin" / exe


@dataclass(frozen=True)
class Port:
    """One ported function. `exe` is the server executable's path, or with `package` the name of a `lean_exe` of that Lake
    package, which `run` builds. `reference(value)` returns the expected output (canonical JSON's domain) or raises one of
    `rejects`. `inputs` draws inputs, invalid ones included; `need_rejects=False` waives the check that some are rejected."""

    exe: str | Path
    fn: str
    reference: Callable[[Any], Any]
    inputs: SearchStrategy
    package: Path | None = None
    rejects: tuple[type[BaseException], ...] = (Reject,)
    need_rejects: bool = True

    def argv(self) -> list[str]:
        return [str(build(Path(self.package), str(self.exe)) if self.package is not None else Path(self.exe))]


class Server:
    """One server process, started at the first call and again after it dies; requests are answered in order."""

    def __init__(self, argv: Sequence[str]):
        self.argv = list(argv)
        self._proc: subprocess.Popen | None = None
        self._stderr = None

    def _start(self) -> subprocess.Popen:
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                      text=True, encoding="utf-8")
        return self._proc

    def call(self, fn: str, value: Any) -> tuple[str, str]:
        """`("output", canonical text)` or `("reject", reason)`; raises `ProtocolError` for anything else."""
        request = canonical({"fn": fn, "input": value})
        proc = self._proc or self._start()
        try:
            proc.stdin.write(request + "\n")
            proc.stdin.flush()
            reply = proc.stdout.readline()
        except BrokenPipeError:
            reply = ""
        if not reply.endswith("\n"):
            raise ProtocolError(f"{self.argv[0]} ended on {request}: {self._dead()}")
        reply = reply[:-1]
        try:
            msg = json.loads(reply)
            ok = isinstance(msg, dict) and len(msg) == 1 and canonical(msg) == reply
        except (ValueError, TypeError):
            ok = False
        if not ok:
            proc.kill()
            self._dead()
            raise ProtocolError(f"{self.argv[0]} answered {request} with a line that is not one canonical JSON object "
                                f"(stdout is the protocol's: write anything else to stderr): {reply[:500]}")
        (kind, body), = msg.items()
        if kind == "output":
            return "output", canonical(body)
        if kind == "reject" and isinstance(body, str):
            return "reject", body
        raise ProtocolError(f"{self.argv[0]} on {request}: {body if kind == 'error' else reply}")

    def _dead(self) -> str:
        proc, self._proc = self._proc, None
        if proc.stdin:
            proc.stdin.close()
        code = proc.wait()
        self._stderr.seek(0)
        err = self._stderr.read().decode("utf-8", "replace")
        self._stderr.close()
        return f"exit {code}, stderr:\n{err[-2000:]}"

    def close(self) -> None:
        if self._proc is not None:
            self._dead()

    def __enter__(self) -> Server:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _show(outcome: tuple[str, str]) -> str:
    kind, body = outcome
    return f"{kind} {body}" if kind == "output" else f"{kind} {body!r}"


def _python(port: Port, value: Any) -> tuple[str, str]:
    try:
        out = port.reference(copy.deepcopy(value))
    except port.rejects as e:
        return "reject", str(e)
    except Exception as e:
        return "error", f"the reference raised {type(e).__name__}: {e}, not one of {[r.__name__ for r in port.rejects]}"
    try:
        return "output", canonical(out)
    except TypeError as e:
        return "error", f"the reference's output is {e}"


def _compare(port: Port, server: Server, value: Any, seen: dict[str, int]) -> tuple[str, str]:
    canonical(value)
    python = _python(port, value)
    try:
        lean = server.call(port.fn, value)
    except ProtocolError as e:
        lean = ("error", str(e))
    seen[python[0]] = seen.get(python[0], 0) + 1
    if "error" in (lean[0], python[0]) or lean != python and (lean[0], python[0]) != ("reject", "reject"):
        raise Mismatch(port.fn, value, lean, python)
    return lean


def replay(port: Port, value: Any, server: Server | None = None) -> tuple[str, str]:
    """Compare one input, as `run` does: Lean's outcome when the two agree, else `Mismatch`."""
    with Server(port.argv()) if server is None else contextlib.nullcontext(server) as s:
        return _compare(port, s, value, {})


@dataclass(frozen=True)
class Stats:
    examples: int
    outputs: int
    rejects: int


def run(port: Port, examples: int = 100, seed: int = 0, server: Server | None = None) -> Stats:
    """Draw `examples` inputs from `port.inputs` (deterministically, from `seed`) and compare Lean and Python on each. Raises
    `Mismatch` with a report of the minimal input Hypothesis shrank to and how to reproduce it, or `Vacuous`."""
    seen: dict[str, int] = {}

    @use_seed(seed)
    @settings(derandomize=True, database=None, deadline=None, max_examples=examples,
              report_multiple_bugs=False, suppress_health_check=[HealthCheck.too_slow])
    @given(port.inputs)
    def differential(value: Any) -> None:
        _compare(port, s, value, seen)

    with Server(port.argv()) if server is None else contextlib.nullcontext(server) as s:
        try:
            differential()
        except Mismatch as e:
            e.report = "\n".join([
                f"difftest of {port.fn!r} ({s.argv[0]}): Lean and Python differ; the minimal input Hypothesis found:",
                f"  input:  {canonical(e.input)}",
                f"  lean:   {_show(e.lean)}",
                f"  python: {_show(e.python)}",
                f"reproduce: verity_difftest.replay(port, {e.input!r}), or the whole run: "
                f"verity_difftest.run(port, examples={examples}, seed={seed})"])
            raise
    total = sum(seen.values())
    if not seen.get("output"):
        raise Vacuous(f"difftest of {port.fn!r}: none of {total} inputs gave an output; the strategy reaches only rejects")
    if port.need_rejects and not seen.get("reject"):
        raise Vacuous(f"difftest of {port.fn!r}: none of {total} inputs was rejected; draw invalid inputs too, or set "
                      f"need_rejects=False for a total function")
    return Stats(total, seen.get("output", 0), seen.get("reject", 0))


def server_fixture(port_or_argv: Port | Sequence[str], scope: str = "module"):
    """A pytest fixture holding one `Server` open across the tests of its scope: `server = server_fixture(PORT)` at a test
    module's top level, then `run(PORT, server=server)` in each test."""
    import pytest

    @pytest.fixture(scope=scope)
    def server() -> Iterator[Server]:
        with Server(port_or_argv.argv() if isinstance(port_or_argv, Port) else port_or_argv) as s:
            yield s

    return server
