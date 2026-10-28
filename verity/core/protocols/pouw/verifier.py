"""PoUW's verifier of record. Lean is the definition and the verifier of record (AGENTS.md "Design
rules"), and PoUW has switched to it: the served window's checks
are its Lean's, `lean/check` (`pouw-check`), `lean/commit` (`pouw-commit`) and `lean/window` (`pouw-window`), whose
verdicts `benchmarks/pouw/served_zk/of_record.py` publishes. The Python beside them (`core_check`, `coins`, `window`) is kept
as the reference, and the difftests hold it to the Lean: a disagreement fails with the Lean's verdict. A served H1
window's statements are judged here too: `credit.window` and `credit.period_credit` take each drawn unit's outcome from
`lean("credit-verified", …)` (`PouwCheck.Verified`), with `credit._verified` the reference, and each round's live
outcome from `lean("credit-live-verified", …)` (`PouwCheck.LiveVerified`), with `credit._live_verified` the reference,
on the circuit files its period's open committed, `lean("credit-committed-files", …)` (`PouwCheck.CommittedFiles`),
with `credit.committed_files` the reference.

`RECORD_VERDICT` is the switch, and setting it to ``"python"`` is the one line that undoes it.

`coins` is the challenger's entropy: bytes the Lean verifier of record reads from the operating system (`pouw-check-difftest`'s
``coins``, `IO.getRandomBytes`, A3's `uniform/io-getrandombytes`), which a served `coins.Challenger` and
`window.live_challenger` draw every coin and hm96 salt from. Python never reads them from the operating system itself, and
`RECORD_VERDICT` doesn't switch them. Building takes the lock `verity_difftest.build` takes (its `BUILD_LOCK`): Lake builds
the path requires in place, and two builds at once would rewrite one module's C under each other.

This module imports only the standard library, so what decides a window never depends on `verity.ml`."""
from __future__ import annotations

import atexit
import fcntl
import functools
import json
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

RECORD_VERDICT = "lean"

PACKAGE = Path(__file__).resolve().parent / "lean" / "check"
EXE = "pouw-check-difftest"
BUILD_LOCK = Path(__file__).resolve().parents[2] / "difftest" / "lean" / ".lake" / "harness-build.lock"
#: The most bytes one `coins` request takes (the Lean's `COINS_MAX`).
COINS_MAX = 1 << 24


class LeanError(RuntimeError):
    """The Lean verifier can't be built or run, or answered outside its protocol."""


@functools.cache
def executable() -> Path:
    """`lake build pouw-check-difftest`, once per process, under `BUILD_LOCK`; the executable's path."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(["lake", "build", EXE], cwd=PACKAGE, capture_output=True, text=True)
        except FileNotFoundError:
            raise LeanError(f"lake is not on the PATH, so {EXE} in {PACKAGE} cannot be built") from None
    if p.returncode != 0:
        raise LeanError(f"lake build {EXE} in {PACKAGE} failed ({p.returncode}):\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return PACKAGE / ".lake" / "build" / "bin" / EXE


class _Server:
    """One `pouw-check-difftest` process per Python process (a request line, an answer line), started at the first call
    and again after it dies."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._stderr: Any = None
        self._lock = threading.Lock()

    def call(self, fn: str, value: Any) -> tuple[str, Any]:
        """`("output", JSON)` or `("reject", reason)`; anything else raises `LeanError`."""
        line = json.dumps({"fn": fn, "input": value}, separators=(",", ":"))
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._stderr = tempfile.TemporaryFile()
                self._proc = subprocess.Popen([str(executable())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                              stderr=self._stderr, text=True, encoding="utf-8")
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
                raise LeanError(f"{EXE} ended on {fn}: {err}")
        try:
            msg = json.loads(reply)
        except ValueError:
            raise LeanError(f"{EXE} on {fn} answered a line that is not JSON: {reply[:500]}") from None
        if isinstance(msg, dict) and len(msg) == 1:
            (kind, body), = msg.items()
            if kind == "output" or (kind == "reject" and isinstance(body, str)):
                return kind, body
        raise LeanError(f"{EXE} on {fn}: {reply[:500]}")

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.stdin.close()
                self._proc.wait()
            self._proc = None


_server = _Server()
atexit.register(_server.close)


def lean(fn: str, value: Any) -> Any:
    """`pouw-check-difftest`'s answer to function `fn` on `value`, a check of record whose verdict the caller takes (a
    refusal is part of that answer, never a reject). Raises `LeanError` where the Lean can't run or rejects the
    input."""
    kind, body = _server.call(fn, value)
    if kind != "output":
        raise LeanError(f"{EXE} rejected {fn}'s input: {body}")
    return body


def coins(n: int) -> bytes:
    """`n` bytes of the operating system's randomness, read by the Lean verifier of record (`IO.getRandomBytes`, A3's
    `uniform/io-getrandombytes`): a served challenger's entropy. Raises `LeanError` where the Lean can't run, refuses or
    answers other than `n` bytes: no coins, so no commitment and no draw."""
    if not (type(n) is int and 0 <= n <= COINS_MAX):
        raise ValueError(f"coins are 0 to {COINS_MAX} bytes a request, not {n!r}")
    kind, body = _server.call("coins", n)
    if kind != "output":
        raise LeanError(f"{EXE} refused {n} coin bytes: {body}")
    if not (isinstance(body, str) and len(body) == 2 * n and all(c in "0123456789abcdef" for c in body)):
        raise LeanError(f"{EXE} answered other than {n} coin bytes in hex")
    return bytes.fromhex(body)


#: the claim instance naming `coins`'s bytes, which a served challenger reads (`sampled_proofs.challenger.coin_claim`)
CLAIM = "uniform/io-getrandombytes"
coins.claim = CLAIM  # type: ignore[attr-defined]
