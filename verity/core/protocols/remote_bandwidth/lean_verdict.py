"""The remote audit's Lean verdicts, called from `audit` (`post_seed`): `remote-audit-verdict` in `lean/`, built once a
process with `lake build` and kept open behind the difftest harness's line protocol (`Difftest.serve`: a request line
`{"fn", "input"}`, an answer line `{"output": ...}` or `{"reject": reason}`)."""
from __future__ import annotations

import fcntl
import functools
import json
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent / "lean"
EXE = "remote-audit-verdict"
BUILD_LOCK = PACKAGE / ".lake" / "verdict-build.lock"


class LeanError(RuntimeError):
    """The Lean verdict can't be built or run, or answered outside its protocol."""


@functools.cache
def executable() -> Path:
    """`lake build remote-audit-verdict`, once per process, under `BUILD_LOCK`; the executable's path."""
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
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._stderr: Any = None
        self._lock = threading.Lock()

    def call(self, fn: str, value: Any) -> tuple[str, Any]:
        line = json.dumps({"fn": fn, "input": value}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
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
        msg = json.loads(reply)
        if isinstance(msg, dict) and len(msg) == 1:
            (kind, body), = msg.items()
            if kind == "output" or (kind == "reject" and isinstance(body, str)):
                return kind, body
        raise LeanError(f"{EXE} on {fn}: {reply[:500]}")


_SERVER = _Server()


def call(fn: str, value: Any) -> tuple[str, Any]:
    """`("output", JSON)` or `("reject", reason)` from the Lean's `fn` on `value`; anything else raises `LeanError`."""
    return _SERVER.call(fn, value)


__all__ = ["EXE", "LeanError", "PACKAGE", "call", "executable"]
