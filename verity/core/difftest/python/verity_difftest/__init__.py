"""The differential test of core's Lean, the verifier of record, against the Python that mirrors it (`README.md` in this
distribution's directory).

A port is a Lean executable that answers one canonical JSON line per request, the Python it must agree with (`reference`),
and a Hypothesis strategy of inputs, invalid ones included. There are two kinds, sharing the server, the run and the shrinking:

* `Port`: one function behind `Difftest.serve` (`lean/Difftest.lean`), requests `{"fn", "input"}`. Lean and Python must
  give the same output as canonical JSON (`canonical`: compact, keys sorted, integers only), or both reject (reject parity:
  the reasons need not match).
* `ProtocolPort`: a core protocol's executable in `serve` mode (`Core.Protocol.main`, the interface of
  `verity/core/protocols/interface`), one case `{"commitment", "evidence", "params"}` per line and one outcome
  per case. Lean and Python must agree on `verdict`, `commitment`, `public`, `profile` and `protocol` exactly; `reason` is
  not compared. `verity_difftest.protocol` is the interface's case framing in Python, for references to build on.

`run` keeps one server process open, feeds both sides every input the strategy draws, and fails with a `Mismatch` on the
first difference, shrunk by Hypothesis to a minimal input; a reference that raises something it doesn't declare, and a
server that faults or dies, are differences too. A run is deterministic: Hypothesis draws from `seed` (`derandomize`, no
example database, no deadline) and from its own constants alone (`_no_local_constants`), so `run(port, examples=N,
seed=S)` finds the same minimal input again whatever modules the process has loaded, and `replay(port, value)` compares
one input. Nothing here reads a clock or times out (`tests/test_no_wall_clock.py`): a port whose Lean
never answers hangs its test.

A port names its verifier of record (`of_record`): ``"python"`` until its protocol switches to its Lean by the 8 Oct
procedure, ``"lean"`` after (PoUW's switch is `verity/core/protocols/pouw/verifier.py`). Either way the two sides must
agree; the side of record is the verdict a `Mismatch` holds the other to, and what a run's `Stats` count.

The harness never skips: when `lake` is missing or the build fails it raises. A port's test module skips on `lake` itself
(`pytestmark = pytest.mark.skipif(shutil.which("lake") is None, ...)`), which puts its suite among those `check` runs after
its Lean steps (`AFTER_LEAN` in `tools/verity/check/check.py`, held by `tests/test_no_wall_clock.py`).

Only `run` imports Hypothesis: `build`, `Server` and `canonical` need the stdlib alone, so a port's Lean can serve at run
time (the memory challenger's `verdict.py` in the PoUS harness) where no test dependency is installed.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import functools
import json
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

#: One side's answer to one input: `(kind, text)`. A `Port`'s kinds are `output` (the canonical text), `reject` (the
#: reason) and `error`; a `ProtocolPort`'s are the verdicts (the outcome's canonical text) and `error`.
Outcome = tuple[str, str]


class Reject(Exception):
    """The reference refuses its input (the message is the reason). A port may declare other exception types as rejects."""


class ProtocolError(Exception):
    """The server broke the protocol: a fault it reports (`{"error"}`), a line that isn't one canonical JSON object of the
    expected form, or no line at all."""


class BuildError(Exception):
    """`lake build` of a port's executable failed, or `lake` is not on the PATH."""


class Mismatch(AssertionError):
    """Lean and Python differ on `input` (the minimal one Hypothesis found): `lean` and `python` are each side's
    `Outcome`, and `of_record` is the port's side of record, whose verdict the message holds the other to."""

    def __init__(self, label: str, value: Any, lean: Outcome, python: Outcome, of_record: str = "python"):
        self.label, self.input, self.lean, self.python, self.of_record = label, value, lean, python, of_record
        self.report = ""
        if of_record == "lean":
            super().__init__(f"{label}: Python {_show(python)} where Lean, of record, gives {_show(lean)}, on "
                             f"{canonical(value)}")
        else:
            super().__init__(f"{label}: Lean {_show(lean)}, Python {_show(python)} on {canonical(value)}")

    def __str__(self) -> str:
        return self.report or super().__str__()


class Vacuous(AssertionError):
    """A run passed without exercising both sides: no input was accepted, or (unless the port waives it) none was refused.
    A strategy that only reaches one side of the port's domain checks nothing on the other."""


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


#: Lake builds a path require in place, and ports share theirs (`lean/` here, the core protocol interface), so two ports'
#: builds at once, as two suites in one `check` run them, compile one module's C in one directory together, and clang
#: reads it while the other rewrites it
BUILD_LOCK = Path(__file__).resolve().parents[2] / "lean" / ".lake" / "harness-build.lock"


@functools.cache
def build(package: Path, exe: str) -> Path:
    """`lake build EXE` in the Lake package `package`, once per process and one at a time across processes (`BUILD_LOCK`);
    the built executable's path."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(["lake", "build", exe], cwd=package, capture_output=True, text=True)
        except FileNotFoundError:
            raise BuildError(f"lake is not on the PATH, so {exe} in {package} cannot be built") from None
    if p.returncode != 0:
        raise BuildError(f"lake build {exe} in {package} failed ({p.returncode}):\n{p.stdout}{p.stderr}")
    return package / ".lake" / "build" / "bin" / exe


class Server:
    """One server process, started at the first request and again after it dies; requests are answered in order."""

    def __init__(self, argv: Sequence[str]):
        self.argv = list(argv)
        self._proc: subprocess.Popen | None = None
        self._stderr = None

    def _start(self) -> subprocess.Popen:
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                      text=True, encoding="utf-8")
        return self._proc

    def exchange(self, request: Any) -> dict:
        """Send `request` as one canonical JSON line and read one line back, which must be one canonical JSON object."""
        line = canonical(request)
        proc = self._proc or self._start()
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
            reply = proc.stdout.readline()
        except BrokenPipeError:
            reply = ""
        if not reply.endswith("\n"):
            raise ProtocolError(f"{self.argv[0]} ended on {line}: {self._dead()}")
        reply = reply[:-1]
        try:
            msg = json.loads(reply)
            why = "" if isinstance(msg, dict) and canonical(msg) == reply else "not canonical"
        except (ValueError, TypeError) as e:
            why = f"{type(e).__name__}: {e}"
        if why:
            proc.kill()
            self._dead()
            raise ProtocolError(f"{self.argv[0]} answered {line} with a line that is not one canonical JSON object ({why}; "
                                f"stdout is the protocol's: write anything else to stderr): {reply[:500]}")
        return msg

    def call(self, fn: str, value: Any) -> Outcome:
        """`Difftest.serve`'s answer to `{"fn": fn, "input": value}`: `("output", canonical text)` or `("reject", reason)`;
        anything else raises `ProtocolError`."""
        msg = self.exchange({"fn": fn, "input": value})
        if len(msg) == 1:
            (kind, body), = msg.items()
            if kind == "output":
                return "output", canonical(body)
            if kind == "reject" and isinstance(body, str):
                return "reject", body
            if kind == "error":
                raise ProtocolError(f"{self.argv[0]} on {fn}: {body}")
        raise ProtocolError(f"{self.argv[0]} on {fn}: not an output, a reject or an error: {canonical(msg)[:500]}")

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


def _exe(exe: str | Path, package: Path | None) -> str:
    return str(build(Path(package), str(exe)) if package is not None else Path(exe))


def _reference(port: Port | ProtocolPort, value: Any) -> tuple[str, Any]:
    """The reference on a copy of `value`: `("returned", its output)`, `("reject", reason)` for one of the port's
    `rejects`, or `("error", what)` for anything else it raises, which every comparison counts as a difference."""
    try:
        return "returned", port.reference(copy.deepcopy(value))
    except port.rejects as e:
        return "reject", str(e)
    except Exception as e:  # noqa: BLE001 -- whatever else the reference raises is its outcome: an error, a difference
        return "error", f"the reference raised {type(e).__name__}: {e}, not one of {[r.__name__ for r in port.rejects]}"


#: A port's sides that can be of record.
RECORDS = ("python", "lean")


def _check_record(port: Port | ProtocolPort) -> None:
    if port.of_record not in RECORDS:
        raise ValueError(f"of_record is {port.of_record!r}, not one of {list(RECORDS)}")


@dataclass(frozen=True)
class Port:
    """One ported function. `exe` is the server executable's path, or with `package` the name of a `lean_exe` of that Lake
    package, which `run` builds. `reference(value)` returns the expected output (canonical JSON's domain) or raises one of
    `rejects`. `inputs` draws inputs, invalid ones included; `need_rejects=False` waives the check that some are rejected.
    `of_record` is the side of record (`RECORDS`)."""

    exe: str | Path
    fn: str
    reference: Callable[[Any], Any]
    inputs: SearchStrategy
    package: Path | None = None
    rejects: tuple[type[BaseException], ...] = (Reject,)
    need_rejects: bool = True
    of_record: str = "python"
    accepted = "output"

    def __post_init__(self) -> None:
        _check_record(self)

    @property
    def label(self) -> str:
        return self.fn

    def argv(self) -> list[str]:
        return [_exe(self.exe, self.package)]

    def lean(self, server: Server, value: Any) -> Outcome:
        return server.call(self.fn, value)

    def python(self, value: Any) -> Outcome:
        kind, out = _reference(self, value)
        if kind != "returned":
            return kind, out
        try:
            return "output", canonical(out)
        except TypeError as e:
            return "error", f"the reference's output is {e}"

    def key(self, outcome: Outcome) -> Any:
        """What must be equal: the output, or only the fact of a rejection."""
        return outcome[:1] if outcome[0] == "reject" else outcome


#: The fields of a protocol's outcome line (`Core.Answer.toJson`), and its verdicts.
OUTCOME_FIELDS = frozenset({"commitment", "profile", "protocol", "public", "reason", "verdict"})
VERDICTS = frozenset({"accept", "reject", "undecided"})


def _outcome(msg: Any, who: str) -> Outcome:
    """`(verdict, canonical text)` of an outcome object, or `ValueError` saying what about it is not one."""
    if not isinstance(msg, dict) or set(msg) != OUTCOME_FIELDS:
        got = sorted(msg) if isinstance(msg, dict) else type(msg).__name__
        raise ValueError(f"{who} outcome has the fields {got}, not {sorted(OUTCOME_FIELDS)}")
    if msg["verdict"] not in VERDICTS:
        raise ValueError(f"{who} outcome's verdict {msg['verdict']!r} is not one of {sorted(VERDICTS)}")
    return msg["verdict"], canonical(msg)


@dataclass(frozen=True)
class ProtocolPort:
    """A core protocol's port: its executable (`exe`, or a `lean_exe` of the Lake package `package`) run as `EXE serve`.
    `reference(case)` returns the outcome dict, all six fields (`verity_difftest.protocol.answer` builds one as
    `Core.Protocol.case` does). `inputs` draws cases, `{"commitment": hex, "evidence": hex, "params": JSON}` and malformed
    ones (bad hex, params, evidence, missing fields); `need_rejects=False` waives the check that some aren't accepted.
    `of_record` is the side of record (`RECORDS`)."""

    exe: str | Path
    reference: Callable[[Any], dict]
    inputs: SearchStrategy
    package: Path | None = None
    need_rejects: bool = True
    serve: tuple[str, ...] = ("serve",)
    of_record: str = "python"
    accepted = "accept"
    rejects: ClassVar[tuple[type[BaseException], ...]] = ()

    def __post_init__(self) -> None:
        _check_record(self)

    @property
    def label(self) -> str:
        return f"{Path(str(self.exe)).name} serve"

    def argv(self) -> list[str]:
        return [_exe(self.exe, self.package), *self.serve]

    def lean(self, server: Server, case: Any) -> Outcome:
        try:
            return _outcome(server.exchange(case), "Lean's")
        except ValueError as e:
            raise ProtocolError(f"{server.argv[0]}: {e}") from None

    def python(self, case: Any) -> Outcome:
        kind, out = _reference(self, case)
        if kind != "returned":
            return kind, out
        try:
            return _outcome(out, "the reference's")
        except (ValueError, TypeError) as e:
            return "error", str(e)

    def key(self, outcome: Outcome) -> Any:
        """Everything but the reason."""
        if outcome[0] == "error":
            return outcome
        return {k: v for k, v in json.loads(outcome[1]).items() if k != "reason"}


def _show(outcome: Outcome) -> str:
    kind, body = outcome
    return f"{kind} {body}" if kind in ("output", *VERDICTS) else f"{kind} {body!r}"


def _compare(port: Port | ProtocolPort, server: Server, value: Any, seen: dict[str, int]) -> Outcome:
    canonical(value)
    python = port.python(value)
    try:
        lean = port.lean(server, value)
    except ProtocolError as e:
        lean = ("error", str(e))
    kind = (lean if port.of_record == "lean" else python)[0]
    seen[kind] = seen.get(kind, 0) + 1
    if "error" in (lean[0], python[0]) or port.key(lean) != port.key(python):
        raise Mismatch(port.label, value, lean, python, port.of_record)
    return lean


def replay(port: Port | ProtocolPort, value: Any, server: Server | None = None) -> Outcome:
    """Compare one input, as `run` does: Lean's outcome when the two agree, else `Mismatch`."""
    with Server(port.argv()) if server is None else contextlib.nullcontext(server) as s:
        return _compare(port, s, value, {})


@dataclass(frozen=True)
class Stats:
    examples: int
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def outputs(self) -> int:
        return self.counts.get("output", 0)

    @property
    def rejects(self) -> int:
        return self.counts.get("reject", 0)


@contextlib.contextmanager
def _no_local_constants() -> Iterator[None]:
    """Hypothesis also draws, at p = 0.05 per draw, constants it scrapes from every local module the process has loaded
    (`providers._get_local_constants`, cached per constraint in `CONSTANTS_CACHE`), so a seeded run's draws moved when a
    module the reference imports gained a literal, or a test imported one first. Within this, it draws from its own
    constants alone. `tests/test_constants.py` pins the hook, so a Hypothesis upgrade that moves it fails there."""
    from hypothesis.internal.conjecture import providers
    from hypothesis.internal.constants_ast import Constants

    scrape = providers._get_local_constants
    providers._get_local_constants = Constants
    providers.CONSTANTS_CACHE.cache.clear()
    try:
        yield
    finally:
        providers._get_local_constants = scrape
        providers.CONSTANTS_CACHE.cache.clear()


def run(port: Port | ProtocolPort, examples: int = 100, seed: int = 0, server: Server | None = None) -> Stats:
    """Draw `examples` inputs from `port.inputs` (deterministically, from `seed`) and compare Lean and Python on each. Raises
    `Mismatch` with a report of the minimal input Hypothesis shrank to and how to reproduce it, or `Vacuous`. The `Stats`
    count the outcomes of the side of record by kind.  Hypothesis is imported here, so the server, `build` and `replay` run
    without it."""
    from hypothesis import HealthCheck, given, settings
    from hypothesis import seed as use_seed

    seen: dict[str, int] = {}

    @use_seed(seed)
    @settings(derandomize=True, database=None, deadline=None, max_examples=examples, report_multiple_bugs=False,
              suppress_health_check=[HealthCheck.too_slow])
    @given(port.inputs)
    def differential(value: Any) -> None:
        _compare(port, s, value, seen)

    with Server(port.argv()) if server is None else contextlib.nullcontext(server) as s, _no_local_constants():
        try:
            differential()
        except Mismatch as e:
            lean_of_record = port.of_record == "lean"
            e.report = "\n".join([
                f"difftest of {port.label!r} ({s.argv[0]}): "
                + ("Python differs from the Lean, the verifier of record" if lean_of_record else "Lean and Python differ")
                + "; the minimal input Hypothesis found:",
                f"  input:  {canonical(e.input)}",
                f"  lean:   {_show(e.lean)}" + ("  (of record)" if lean_of_record else ""),
                f"  python: {_show(e.python)}",
                f"reproduce: verity_difftest.replay(port, {e.input!r}), or the whole run: "
                f"verity_difftest.run(port, examples={examples}, seed={seed})"])
            raise
    total = sum(seen.values())
    if not seen.get(port.accepted):
        raise Vacuous(f"difftest of {port.label!r}: none of {total} inputs gave {port.accepted!r}; the strategy reaches "
                      f"only refusals")
    if port.need_rejects and total == seen[port.accepted]:
        raise Vacuous(f"difftest of {port.label!r}: none of {total} inputs was rejected; draw invalid inputs too, or set "
                      f"need_rejects=False for a total function")
    return Stats(total, dict(seen))


def server_fixture(port_or_argv: Port | ProtocolPort | Sequence[str], scope: str = "module"):
    """A pytest fixture holding one `Server` open across the tests of its scope: `server = server_fixture(PORT)` at a test
    module's top level, then `run(PORT, server=server)` in each test."""
    import pytest

    @pytest.fixture(scope=scope)
    def server() -> Iterator[Server]:
        with Server(port_or_argv.argv() if isinstance(port_or_argv, (Port, ProtocolPort)) else port_or_argv) as s:
            yield s

    return server
