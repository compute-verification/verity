"""PoUW's verifier. Lean is the definition and the verifier (AGENTS.md "Design
rules"), and PoUW has switched to it: the served execution phase's checks
are its Lean's, `lean/check` (`pouw-check`), `lean/commit` (`pouw-commit`) and `lean/window` (`pouw-window`), whose
verdicts `benchmarks/pouw/served_zk/of_record.py` publishes. The Python beside them (`core_check`, `coins`, `window`) is kept
as the reference, and the difftests hold it to the Lean: a disagreement fails with the Lean's verdict. A served H1
execution phase's statements are judged here too: `credit.window` and `credit.period_credit` take each sampled proof block's outcome from
`lean("credit-verified", …)` (`PouwCheck.Verified`), with `credit._verified` the reference, and each round's live
outcome from `lean("credit-live-verified", …)` (`PouwCheck.LiveVerified`), with `credit._live_verified` the reference,
on the circuit files its credit period's open committed, `lean("credit-committed-files", …)` (`PouwCheck.CommittedFiles`),
with `credit.committed_files` the reference. The verdicts acted on are the Lean's too (`of_record`): `credit.window`'s
and `credit.period_credit`'s, `lean("window", …)` and `lean("period", …)` (`PouwCheck.Credit.window`,
`Credit.periodCredit`) on those outcomes, and the challenger's salt check's, `lean("salt", …)` (`PouwCheck.saltCheck`,
`core_check.salt_of_record`), each with its Python twin beside it, and a disagreement between them is a refusal. An
execution phase's and a credit period's verdicts are `pouw-check serve`'s and `pouw-period serve`'s (`serve_of_record`,
`core_check.case`, `period_check.case`), and a verdict whose Lean no interface here reaches yet raises
(`not_written`) rather than fall back on its twin; the README's "Verdicts of record" lists each.

`RECORD_VERDICT` is the switch, and setting it to ``"python"`` is the one line that undoes it.

`coins` is the challenger's entropy: bytes the Lean verifier reads from the operating system (`pouw-check-difftest`'s
``coins``, `IO.getRandomBytes`, A3's `uniform/io-getrandombytes`), which a served `coins.Challenger` and
`window.live_challenger` flip every coin and hm96 salt from. Python never reads them from the operating system itself, and
`RECORD_VERDICT` doesn't switch them. Building takes the lock `verity_difftest.build` takes (its `BUILD_LOCK`): Lake builds
the path requires in place, and two builds at once would rewrite one module's C under each other.

This module imports only the standard library, so what decides an execution phase never depends on `verity.ml`."""
from __future__ import annotations

import atexit
import fcntl
import functools
import json
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, NoReturn

RECORD_VERDICT = "lean"
#: the refusal code of an answer of record the Lean and its Python twin disagree on, or the Lean can't give
DISAGREE = "disagree"

PACKAGE = Path(__file__).resolve().parent / "lean" / "check"
EXE = "pouw-check-difftest"
BUILD_LOCK = Path(__file__).resolve().parents[5] / "verity/circuits/difftest" / ".lake" / "harness-build.lock"
#: The most bytes one `coins` request takes (the Lean's `COINS_MAX`).
COINS_MAX = 1 << 24


class LeanError(RuntimeError):
    """The Lean verifier can't be built or run, or answered outside its protocol."""


#: the core protocols' executables of `PACKAGE` whose `serve` answers a case (`Core.Protocol.main`): one execution
#: phase's check (`verity/pouw/v1`) and a credit period's (`verity/pouw/period/v1`)
SERVE_EXES = ("pouw-check", "pouw-period")


@functools.cache
def executable(exe: str = EXE) -> Path:
    """`lake build EXE` (`pouw-check-difftest` by default), once per process and executable, under `BUILD_LOCK`; the
    executable's path."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(["lake", "build", exe], cwd=PACKAGE, capture_output=True, text=True)
        except FileNotFoundError:
            raise LeanError(f"lake is not on the PATH, so {exe} in {PACKAGE} cannot be built") from None
    if p.returncode != 0:
        raise LeanError(f"lake build {exe} in {PACKAGE} failed ({p.returncode}):\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return PACKAGE / ".lake" / "build" / "bin" / exe


def _canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


class _Server:
    """One process of `exe` with `args` per Python process (a request line, an answer line), started at the first call
    and again after it dies: `pouw-check-difftest` by default, or a core protocol's `serve`."""

    def __init__(self, exe: str = EXE, args: tuple[str, ...] = ()) -> None:
        self.exe, self.args = exe, args
        self._proc: subprocess.Popen | None = None
        self._stderr: Any = None
        self._lock = threading.Lock()

    def exchange(self, request: Any, what: str) -> Any:
        """The JSON line `exe` answers to `request`'s canonical line; raises `LeanError` where it ends or answers a line
        that is not JSON."""
        line = _canonical(request)
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._stderr = tempfile.TemporaryFile()
                self._proc = subprocess.Popen([str(executable(self.exe)), *self.args], stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE, stderr=self._stderr, text=True, encoding="utf-8")
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
                reply = self._proc.stdout.readline()
            except BrokenPipeError:
                reply = ""
            if not reply.endswith("\n"):
                self._proc.kill()
                self._proc.wait()
                self._stderr.seek(0)
                err = self._stderr.read().decode("utf-8", "replace")[-2000:]
                self._proc = None
                raise LeanError(f"{self.exe} ended on {what}: {err}")
        try:
            return json.loads(reply)
        except ValueError:
            raise LeanError(f"{self.exe} on {what} answered a line that is not JSON: {reply[:500]}") from None

    def call(self, fn: str, value: Any) -> tuple[str, Any]:
        """`("output", JSON)` or `("reject", reason)`; anything else raises `LeanError`."""
        msg = self.exchange({"fn": fn, "input": value}, fn)
        if isinstance(msg, dict) and len(msg) == 1:
            (kind, body), = msg.items()
            if kind == "output" or (kind == "reject" and isinstance(body, str)):
                return kind, body
        raise LeanError(f"{self.exe} on {fn}: {_said(msg)}")

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.stdin.close()
                self._proc.wait()
            self._proc = None


_server = _Server()
atexit.register(_server.close)
_serves = {exe: _Server(exe, ("serve",)) for exe in SERVE_EXES}
for _s in _serves.values():
    atexit.register(_s.close)


def lean(fn: str, value: Any) -> Any:
    """`pouw-check-difftest`'s answer to function `fn` on `value`, a check of record whose verdict the caller takes (a
    refusal is part of that answer, never a reject). Raises `LeanError` where the Lean can't run or rejects the
    input."""
    kind, body = _server.call(fn, value)
    if kind != "output":
        raise LeanError(f"{EXE} rejected {fn}'s input: {body}")
    return body


def _said(x: Any) -> str:
    s = json.dumps(x, sort_keys=True, separators=(",", ":"))
    return s if len(s) <= 600 else s[:600] + "…"


def of_record(fn: str, value: Any, python: Any) -> Any:
    """The answer of record to `fn` on `value` (`pouw-check-difftest`'s inputs), whose Python twin answered `python`
    in the difftest's JSON form: `python` under ``RECORD_VERDICT = "python"``; else the Lean's (`lean`) when it is
    that same answer, and ``{"refused": [DISAGREE, why]}`` naming both when it isn't or the Lean can't give one. The
    Python is the twin the Lean is held to, never its fallback."""
    if RECORD_VERDICT != "lean":
        return python
    try:
        got = lean(fn, value)
    except (LeanError, TypeError, ValueError) as e:
        return {"refused": [DISAGREE, f"the Lean verifier of record gave no answer to {fn}, so it stands refused: {e}"]}
    if got != json.loads(json.dumps(python)):
        return {"refused": [DISAGREE, f"the Lean verifier of record and its Python twin disagree on {fn}, so it "
                                      f"stands refused: the Lean answered {_said(got)}, the Python {_said(python)}"]}
    return got


def serve_of_record(exe: str, case: Any, python: dict) -> dict:
    """The outcome of record (`Core.Answer`'s JSON) of core protocol `exe`'s `serve` on `case`, whose Python twin
    answered `python`: `python` under ``RECORD_VERDICT = "python"``; else the Lean's when it is that same outcome, its
    reason included, and a rejection whose reason starts ``disagree:`` naming both when it isn't or the Lean can't give
    one. The Python is the twin the Lean is held to, never its fallback."""
    if RECORD_VERDICT != "lean":
        return python
    refuse = {"protocol": python.get("protocol"), "verdict": "reject", "commitment": python.get("commitment"),
              "public": None, "profile": None}
    try:
        got = _serves[exe].exchange(case, "a serve case")
    except (LeanError, TypeError, ValueError) as e:
        return {**refuse, "reason": f"{DISAGREE}: the Lean verifier of record ({exe} serve) gave no answer, so the "
                                    f"case stands refused: {e}"}
    if _canonical(got) != _canonical(python):
        return {**refuse, "reason": f"{DISAGREE}: the Lean verifier of record ({exe} serve) and its Python twin "
                                    f"disagree, so the case stands refused: the Lean answered {_said(got)}, the "
                                    f"Python {_said(python)}"}
    return got


#: the goal under which a check `not_written` stands for gets its Lean interface
GOAL = "compute-accounting/pouw-verdict-from-main"


def not_written(fn: str, lean: str) -> NoReturn:
    """The verdict of record of `fn`, the Lean `lean`'s, which no interface here reaches yet: raises
    `NotImplementedError` naming `GOAL`. The Python twin beside `fn` answers the difftest only, never in its place."""
    raise NotImplementedError(GOAL)


def coins(n: int) -> bytes:
    """`n` bytes of the operating system's randomness, read by the Lean verifier (`IO.getRandomBytes`, A3's
    `uniform/io-getrandombytes`): a served challenger's entropy. Raises `LeanError` where the Lean can't run, refuses or
    answers other than `n` bytes: no coins, so no commitment and no sample."""
    if not (type(n) is int and 0 <= n <= COINS_MAX):
        raise ValueError(f"coins are 0 to {COINS_MAX} bytes a request, not {n!r}")
    kind, body = _server.call("coins", n)
    if kind != "output":
        raise LeanError(f"{EXE} refused {n} coin bytes: {body}")
    if not (isinstance(body, str) and len(body) == 2 * n and all(c in "0123456789abcdef" for c in body)):
        raise LeanError(f"{EXE} answered other than {n} coin bytes in hex")
    return bytes.fromhex(body)


#: the claim instance naming `coins`'s bytes, which a served challenger reads (`spot.challenger.coin_claim`)
CLAIM = "uniform/io-getrandombytes"
coins.claim = CLAIM  # type: ignore[attr-defined]


__all__ = ["CLAIM", "COINS_MAX", "DISAGREE", "GOAL", "RECORD_VERDICT", "SERVE_EXES", "LeanError", "coins", "executable",
           "lean", "not_written", "of_record", "serve_of_record"]
