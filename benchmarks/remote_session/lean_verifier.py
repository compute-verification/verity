"""The remote audit's verifier of record, the Lean process `remote-audit-verifier` (`RemoteAudit.Verifier`, in
verity/core/protocols/remote_bandwidth/lean), as `verifier.py` runs it beside `audit.Session`: one request a JSON line
on its stdin (an openings frame's bytes right after its line), each answered by one line on its stdout; a window's
check runs on the process's threads and its verdict comes on a line of its own, handed to `done` as it arrives.

And the JSON the two verifiers are compared in: each event as the Lean reads it (`Difftest.eventOf`), and what
`audit.Session` returns for it as the Lean writes its result (`Difftest.resultJ`), as the difftest's
`ra_cases.apply` writes it."""
from __future__ import annotations

import fcntl
import functools
import json
import queue
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity.core.protocols.remote_bandwidth import audit, codec, receipts

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "verity/core/protocols/remote_bandwidth/lean"
EXE = "remote-audit-verifier"
REFUSED = {"refused": True}
BUILD_LOCK = ROOT / "verity/core/difftest/lean/.lake/harness-build.lock"
"""`verity_difftest.BUILD_LOCK`: Lake builds the package's path requires in place, so a build here and the difftest's
build of `remote-audit-difftest` (a suite beside this one) take turns."""


class LeanError(RuntimeError):
    """The process answered a request with an error, or ended."""


@functools.cache
def build(package: Path = PACKAGE) -> Path:
    """`lake build remote-audit-verifier` in the remote audit's Lake package, once a process, and the executable's
    path."""
    if shutil.which("lake") is None:
        raise LeanError("lake is not on the PATH, so remote-audit-verifier cannot be built: put elan's bin on the PATH "
                        "(tools/verity/lean/setup.sh), pass --lean-exe, or run with --verdict python")
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = subprocess.run(["lake", "build", EXE], cwd=package, capture_output=True, text=True)
    if p.returncode:
        raise LeanError(f"lake build {EXE} in {package} failed ({p.returncode}):\n{p.stdout[-4000:]}{p.stderr[-4000:]}")
    return package / ".lake" / "build" / "bin" / EXE


class Lean:
    """One `remote-audit-verifier` process: `call` sends a request and returns its answer (`LeanError` for an error),
    `done({"done": id, "accepted": bool})` is called on the reader's thread as each window's check ends, and
    `done({"done": None, ...})` once if the process ends: every check still running has no verdict."""

    def __init__(self, exe: Path, done: Callable[[dict], None]) -> None:
        self.proc = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.done, self.answers, self.lock = done, queue.Queue(), threading.Lock()
        self.ended: str | None = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            msg = json.loads(line)
            if "done" in msg:
                self.done(msg)
            else:
                self.answers.put(msg)
        self.ended = f"the process ended ({self.proc.wait()})"
        self.done({"done": None, "accepted": False})
        self.answers.put({"error": self.ended})

    def call(self, request: dict, body: bytes = b"") -> Any:
        with self.lock:
            if self.ended is not None:
                raise LeanError(self.ended)
            try:
                self.proc.stdin.write(json.dumps(request, separators=(",", ":")).encode() + b"\n" + body)
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise LeanError(f"the process is gone: {e}") from None
            msg = self.answers.get()
        if "error" in msg:
            raise LeanError(msg["error"])
        return msg["answer"]

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.wait()


def codec_json(c: codec.Codec) -> dict:
    return {"field": c.field.name, "masked": c.masked, "rounds": c.rounds}


def registration_json(r: audit.Registration) -> dict:
    """A registration as the difftest's case names it (`Difftest.registrationOf`)."""
    return {"bind": r.bind, "blocks": r.blocks, "codec": codec_json(r.codec), "codec_seed": r.codec_seed.hex(),
            "layout": r.layout, "node": r.node.hex(), "payload_root": r.payload_root.hex(), "receipt": r.receipt,
            "rows": r.rows}


def register_request(node: bytes, blocks: int, payload_root: bytes, c: codec.Codec, receipt: str, rows: int | None,
                     nonce: bytes) -> dict:
    return {"op": "register", "node": node.hex(), "blocks": blocks, "payload_root": payload_root.hex(),
            "codec": codec_json(c), "receipt": receipt, "rows": receipts.sketch_rows(receipt, rows),
            "layout": receipts.LAYOUT, "nonce": nonce.hex()}


def params_json(p: audit.Params) -> dict:
    return {"deadline_ns": p.deadline_ns, "windows": p.windows, "samples": p.samples, "min_passes": p.min_passes,
            "window_ns": p.window_ns, "lifetime": p.lifetime, "deferred": p.deferred}


def _ints(xs: Any, what: str) -> list[list[int]]:
    """Runs `[first, count]` as both verifiers read them, or ValueError: a message the driver can't read."""
    if not isinstance(xs, (list, tuple)) or any(not isinstance(r, (list, tuple)) or len(r) != 2
                                                or any(type(x) is not int for x in r) for r in xs):
        raise ValueError(f"{what} are runs [first, count] of integers")
    return [list(r) for r in xs]


def release_event(released: Any, withdrawn: Any) -> dict:
    return {"kind": "release", "runs": _ints(released, "released blocks"), "withdrawn": _ints(withdrawn, "withdrawn blocks")}


def receive_event(t: Any, i: Any, root: bytes, change: Any, step: Any) -> dict:
    """A root as the event the Lean reads (the driver adds `now`); ValueError where the message isn't one (a step
    that isn't an integer is none, which both verifiers reject as `step`)."""
    if type(t) is not int or type(i) is not int:
        raise ValueError("a root names its window and pass by integers")
    if change is not None:
        if not isinstance(change, (list, tuple)) or len(change) != 2:
            raise ValueError("a change is [added, removed]")
        change = [_ints(change[0], "added blocks"), _ints(change[1], "removed blocks")]
    return {"kind": "receive", "t": t, "i": i, "root": root.hex(), "runs": None, "change": change,
            "step": step if type(step) is int else None}


def verdict_json(v: audit.Verdict) -> dict:
    return {"accepted": v.window} if v.accepted else {
        "at": None if v.at is None else list(v.at), "failure": v.failure, "window": v.window}


def done_json(out: audit.Verdict | None) -> dict:
    return {"ok": True} if out is None else verdict_json(out)


def released_json(c) -> dict:
    return {"released": None if c is None else c.index}


def appended_json(out) -> dict:
    return verdict_json(out) if isinstance(out, audit.Verdict) else {"coin": out[1].hex(), "first": out[0]}


def drew_json(out) -> dict:
    return verdict_json(out) if isinstance(out, audit.Verdict) else {"draws": [list(d) for d in out]}


def summary_json(rec: audit.Record) -> dict:
    """An accepted session's record as the Lean writes it (`Difftest.summaryJ`, `ra_cases.summary`)."""
    return {"appended": rec.appended, "appends": None if rec.appends is None else rec.appends.hex(),
            "registration": rec.registration.hex(), "released": rec.released,
            "windows": [{"deadline_ns": w.deadline_ns, "drawn": w.profile.drawn["receipt"], "lengths": list(w.lengths),
                         "opened_ns": w.opened_ns, "post_seed": list(w.post_seed), "released_ns": w.released_ns,
                         "steps": list(w.steps)} for w in rec.windows]}


class Tape:
    """The challenger's generator, drawing from `entropy` and keeping what it drew: a session's committed coins
    (`coins`, each drawn before its salt) and its append coins (`appends`), as the Lean's session takes them."""

    def __init__(self, count: int, entropy: Callable[[int], bytes]) -> None:
        self.count, self.entropy, self.drawn = 2 * count, entropy, []

    def __call__(self, n: int) -> bytes:
        out = self.entropy(n)
        self.drawn.append(out)
        return out

    @property
    def coins(self) -> list[bytes]:
        return self.drawn[:self.count:2]

    @property
    def appends(self) -> list[bytes]:
        return self.drawn[self.count:]
