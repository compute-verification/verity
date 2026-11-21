"""The Lean verifier of a P2 session, the process `poub-verifier` (`Poub.Verifier`'s `p2_*`
requests, `Poub.P2Session`, in `lean/`), as `p2_parties.py` runs it beside `audit.AuditRun` over the P2 codec:
one request a JSON line on its stdin, a line's or a file's bytes right after it, each answered by one line on its
stdout, `{"answer": JSON}` or `{"error": why}`. The standard library only, so the party runs as the auditor's gate
starts it (`python3 -S -E`)."""

from __future__ import annotations

import fcntl
import functools
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from verity import lean_process as LC

PACKAGE = Path(__file__).resolve().parent / "lean"
EXE = "poub-verifier"
BUILD_LOCK = Path(__file__).resolve().parents[6] / "verity/circuits/difftest/.lake/harness-build.lock"
"""`verity_difftest.BUILD_LOCK`: Lake builds the package's path requires in place, so a build here and the difftest's
build of `poub-difftest` take turns."""


class LeanError(RuntimeError):
    """The process answered a request with an error, or ended."""


@functools.cache
def build(package: Path = PACKAGE) -> Path:
    """`lake build poub-verifier` in PoUB's Lake package, once a process, and the executable's
    path."""
    if shutil.which("lake") is None:
        raise LeanError(f"lake is not on the PATH, so {EXE} cannot be built: put elan's bin on the PATH "
                        f"(tools/verity/lean/setup.sh), or name a built one (--lean-exe)")
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = subprocess.run(["lake", "build", EXE], cwd=package, capture_output=True, text=True)
    if p.returncode:
        raise LeanError(f"lake build {EXE} in {package} failed ({p.returncode}):\n{p.stdout[-4000:]}{p.stderr[-4000:]}")
    return package / ".lake" / "build" / "bin" / EXE


def exe(given: Path | None = None) -> Path:
    """The executable: `given` (the caller's, `p2_parties --lean-exe`), else built here (`build`), or, where `lake` isn't
    on the PATH, the one already built here: an auditor gate's party, whose pinned tree, never changed after,
    `vy-verifier-pin` built it in."""
    if given is not None:
        return given
    built = PACKAGE / ".lake" / "build" / "bin" / EXE
    return built if shutil.which("lake") is None and built.is_file() else build()


class Lean:
    """One `poub-verifier` process, its requests answered in the order they are made: `send` writes one (with
    `body` after its line), `read` returns the next answer (`LeanError` for an error or the process gone), and `call`
    is the two. A request's body is written whole before `send` returns, so a caller may compute while the process
    checks it and `read` after."""

    def __init__(self, path: Path) -> None:
        self.stderr = tempfile.TemporaryFile("a+b")
        self.proc = subprocess.Popen([str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                     env=LC.lean_env())
        self.unsent: OSError | None = None

    def send(self, request: dict[str, Any], body: bytes = b"") -> None:
        try:
            self.proc.stdin.write(json.dumps(request, separators=(",", ":")).encode() + b"\n" + body)
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise LeanError(f"{LC.exit_why(self.proc.wait()) or 'the process is gone'}: {e}") from None

    def read(self) -> Any:
        line = self.proc.stdout.readline()
        self.stderr.seek(0)
        err = self.stderr.read().decode("utf-8", "replace")
        if (panic := LC.panic_line(err)) is not None:
            self.proc.kill()
            raise LeanError(f"{LC.LEAN_PANIC}: {panic}")
        if not line:
            code = self.proc.wait()
            raise LeanError(f"{LC.exit_why(code) or f'the process ended ({code})'}: {err[-2000:]}")
        msg = json.loads(line)
        if "error" in msg:
            raise LeanError(msg["error"])
        return msg["answer"]

    def call(self, request: dict[str, Any], body: bytes = b"") -> Any:
        self.send(request, body)
        return self.read()

    def close(self, check: bool = True) -> None:
        """Ends the process: its input closed, then its end waited for. One that has exited (a request crashed it) can't
        take what a failed `send` left in its stdin's buffer: `unsent` is the error closing it raised then, and the
        process is waited for all the same. With ``check``, `LeanError` by name (`LC.LEAN_PANIC`, `LC.LEAN_EXIT`) where
        it wrote a panic line while it ended or ended with a status but 0: an answer read from it is not used."""
        try:
            self.proc.stdin.close()
        except OSError as e:
            self.unsent = e
        self.proc.stdout.read()
        code = self.proc.wait()
        self.stderr.seek(0)
        err = self.stderr.read().decode("utf-8", "replace")
        self.stderr.close()
        if check and (why := (f"{LC.LEAN_PANIC}: {panic}" if (panic := LC.panic_line(err)) else LC.exit_why(code))):
            raise LeanError(why)

    def __enter__(self) -> Lean:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close(check=exc[0] is None)


__all__ = ["BUILD_LOCK", "PACKAGE", "build"]
