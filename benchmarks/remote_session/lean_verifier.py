"""The remote audit's verifier, the Lean process `remote-audit-verifier` (`RemoteAudit.Verifier`, in
verity/core/protocols/remote_bandwidth/lean), as `verifier.py` runs it beside `audit.AuditRun`: one request a JSON line
on its stdin (an openings frame's bytes right after its line), each answered by one line on its stdout; a window's
check runs on the process's threads and its verdict comes on a line of its own, handed to `done` as it arrives.

And the JSON the two verifiers are compared in: each event as the Lean reads it (`Difftest.eventOf`), and what
`audit.AuditRun` returns for it as the Lean writes its result (`Difftest.resultJ`), as the difftest's
`ra_cases.apply` writes it; an accepted session's record as the Lean writes it (`summary_json`), and that record read
back as the `audit.Record` the ledger reads (`record_of`)."""
from __future__ import annotations

import collections
import fcntl
import functools
import hashlib
import json
import queue
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts

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
                        "(tools/verity/lean/setup.sh), or pass --lean-exe")
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = subprocess.run(["lake", "build", EXE], cwd=package, capture_output=True, text=True)
    if p.returncode:
        raise LeanError(f"lake build {EXE} in {package} failed ({p.returncode}):\n{p.stdout[-4000:]}{p.stderr[-4000:]}")
    return package / ".lake" / "build" / "bin" / EXE


class Lean:
    """One `remote-audit-verifier` process, its requests taken in the order they are made and written on a thread of
    their own, so that whoever makes one never waits on the pipe: `send` hands a request over and has `answered` called
    with its answer on the reader's thread, and `call` waits for the answer and returns it (`LeanError` for an error).
    `done({"done": id, "accepted": bool})` is called on the reader's thread as each window's check ends, and
    `done({"done": None, ...})` once if the process ends: every check still running has no verdict, and every request
    not yet answered is answered with the error."""

    def __init__(self, exe: Path, done: Callable[[dict], None]) -> None:
        self.proc = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.done, self.requests, self.lock = done, queue.SimpleQueue(), threading.Lock()
        self.pending: collections.deque[Callable[[dict], None] | None] = collections.deque()
        self.ended: str | None = None
        threading.Thread(target=self._write, daemon=True).start()
        threading.Thread(target=self._read, daemon=True).start()

    def _end(self, why: str) -> None:
        with self.lock:
            self.ended = self.ended or why
            pending, self.pending = list(self.pending), collections.deque()
        for answered in pending:
            if answered is not None:
                answered({"error": self.ended})

    def _write(self) -> None:
        while (item := self.requests.get()) is not None:
            data, answered = item
            with self.lock:
                ended = self.ended
                if ended is None:
                    self.pending.append(answered)
            if ended is not None:
                if answered is not None:
                    answered({"error": ended})
                continue
            try:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as e:
                self._end(f"the process is gone: {e}")
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def _read(self) -> None:
        for line in self.proc.stdout:
            msg = json.loads(line)
            if "done" in msg:
                self.done(msg)
                continue
            with self.lock:
                answered = self.pending.popleft() if self.pending else None
            if answered is not None:
                answered(msg)
        self._end(f"the process ended ({self.proc.wait()})")
        self.done({"done": None, "accepted": False})

    def _put(self, request: dict, body: bytes, answered: Callable[[dict], None] | None) -> None:
        self.requests.put((json.dumps(request, separators=(",", ":")).encode() + b"\n" + body, answered))

    def send(self, request: dict, body: bytes = b"", answered: Callable[[Any], None] | None = None) -> None:
        """`request` (with `body` after its line) to the process behind every request made before it; `answered`, if
        given, called on the reader's thread with its answer, or `{"error": why}` where `call` would raise."""
        self._put(request, body, None if answered is None else
                  lambda msg: answered(msg["answer"] if "answer" in msg else {"error": msg["error"]}))

    def call(self, request: dict, body: bytes = b"") -> Any:
        box: queue.SimpleQueue = queue.SimpleQueue()
        self._put(request, body, box.put)
        msg = box.get()
        if "error" in msg:
            raise LeanError(msg["error"])
        return msg["answer"]

    def close(self) -> None:
        self.requests.put(None)
        self.proc.wait()


def codec_json(c: codec.Codec) -> dict:
    return {"field": c.field.name, "masked": c.masked, "rounds": c.rounds}


def registration_json(r: audit.Registration) -> dict:
    """A registration as the difftest's case names it (`Difftest.registrationOf`)."""
    return {"bind": r.bind, "blocks": r.blocks, "codec": codec_json(r.codec), "codec_seed": r.codec_seed.hex(),
            "layout": r.layout, "node": r.node.hex(), "payload_root": r.payload_root.hex(), "receipt": r.receipt,
            "rows": r.rows}


def register_request(node: bytes, blocks: int, payload_root: bytes, c: codec.Codec, receipt: str, rows: int | None,
                     nonce: bytes, bind: bool, site: dict) -> dict:
    """`Verifier.lean`'s register request: the node held to the Site of `site`, `{"log", "auditor"}`, which the Lean
    checks itself (`Difftest.siteOf`)."""
    return {"op": "register", "site": site, "node": node.hex(), "blocks": blocks, "payload_root": payload_root.hex(),
            "codec": codec_json(c), "receipt": receipt, "bind": bind, "rows": receipts.sketch_rows(receipt, rows),
            "layout": receipts.LAYOUT, "nonce": nonce.hex()}


def params_json(p: audit.Params) -> dict:
    """A session's parameters as the Lean reads them (`Difftest.sessionOf`), naming `groups_per_coin` only above 1."""
    shared = {} if p.groups_per_coin == 1 else {"groups_per_coin": p.groups_per_coin}
    return {"deadline_ns": p.deadline_ns, "windows": p.windows, "samples": p.samples, "min_passes": p.min_passes,
            "window_ns": p.window_ns, "lifetime": p.lifetime, "deferred": p.deferred,
            "pace": None if p.pace is None else p.pace.to_json()} | shared


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
    """An accepted session's record as the Lean writes it (`Difftest.recordJ`, `ra_cases.summary`): all of it but the
    coin commitment's root, the block size, the registration's fields its digest binds, and of each window's profile
    more than its draws; `groups_per_coin` only above 1."""
    shared = {} if rec.groups_per_coin == 1 else {"groups_per_coin": rec.groups_per_coin}
    return shared | {"appended": rec.appended, "appends": None if rec.appends is None else rec.appends.hex(),
            "blocks": rec.blocks, "lifetime": rec.lifetime, "pace": None if rec.pace is None else rec.pace.to_json(),
            "registration": rec.registration.hex(), "released": rec.released,
            "windows": [{"arrived_ns": list(w.arrived_ns), "deadline_ns": w.deadline_ns,
                         "drawn": w.profile.drawn["receipt"], "heads": [None if h is None else h.hex() for h in w.heads],
                         "lengths": list(w.lengths), "opened_ns": w.opened_ns, "post_seed": list(w.post_seed),
                         "released_ns": w.released_ns, "stale": list(w.stale), "steps": list(w.steps)}
                        for w in rec.windows]}


def record_of(j: dict, registration: audit.Registration, coins: bytes, delta: float, samples: int) -> audit.Record:
    """The record the Lean gave an accepted session (`summary_json`'s form) as `audit.Record`: the rest from the
    registration its digest names, the coin commitment's root `coins`, each window's openings law at `delta` over
    the Lean's lengths and draws (`audit.window_profile`), and the session's `samples`, which each window's draws
    must be min(samples, U_t) at (`audit.Record`)."""
    if j["registration"] != registration.digest.hex():
        raise ValueError(f"the Lean's record is of registration {j['registration'][:16]}…, not "
                         f"{registration.digest.hex()[:16]}…")
    pace = j["pace"]
    return audit.Record(
        registration.digest, coins, j["blocks"], audit.BLOCK_BYTES,
        tuple(audit.WindowRecord(t, tuple(w["lengths"]), w["released_ns"], w["deadline_ns"], w["opened_ns"],
                                 audit.window_profile(w["lengths"], w["drawn"], delta), tuple(w["post_seed"]),
                                 tuple(w["steps"]), tuple(None if h is None else bytes.fromhex(h) for h in w["heads"]),
                                 tuple(w["stale"]), tuple(w["arrived_ns"])) for t, w in enumerate(j["windows"])),
        registration.receipt, registration.bind, registration.codec, registration.layout, registration.rows,
        j["appended"], j["released"], j["lifetime"], None if j["appends"] is None else bytes.fromhex(j["appends"]),
        None if pace is None else audit.Pace(pace["k"], pace["window_ns"], pace.get("in_flight", 2)),
        j.get("groups_per_coin", 1), samples=samples)


def released_coins(c: coins.SessionCoins) -> tuple[list[bytes], list[bytes]]:
    """The session's coins as the Lean's session takes them: its W + ⌈W/g⌉ committed coins (window t's challenge at t,
    draw d's openings coin at W + d, g windows a draw), each one the challenger never released 32 zero bytes, which no
    event reads, and the append coins in the order released."""
    zero, w, d = bytes(coins.COIN_BYTES), c.windows, coins.openings_coins(c.windows, c.per_coin)
    out = [coins.coin(o) for o in c.opened(coins.CHALLENGE)], [coins.coin(o) for o in c.opened(coins.OPENINGS)]
    return ([*out[0], *[zero] * (w - len(out[0])), *out[1], *[zero] * (d - len(out[1]))],
            [coins.coin(o) for o in c.opened(coins.APPEND)])


class DifftestCase:
    """A session's difftest case (`ra_cases`: what `replay.py transcript` replays through both verifiers), written to
    `path` as the session runs, on a thread of its own so the session's loop never waits on the disk: its registration,
    parameters and `not_before_ns`, then each event in the order the verifiers took it (`event`, as the Lean read it),
    a window's openings frames as the wire carried them (`verify`, `frames`) where its verdict was set, and the coins the
    challenger released (`close`, `released_coins`, which returns the file's path, bytes and sha256). Both verifiers
    read the frames as the live ones do (`Wire.openingsOf`, `wire.openings_of`): a frame that doesn't decode leaves the
    window no openings, so its replay rejects it as the session did. Every released coin is kept, never its salt: the
    case replays each under a zero salt, and nothing a verifier answers reads the commitment. Frames are written only
    once the session's passes are over (`passes_ended`, or the close): a 60 s window's are about 0.6 GB of hex, and
    writing them beside a timed pass would take the verifier's process from the loop whose stamps the memory line
    reads."""

    def __init__(self, path: Path, registration: audit.Registration, params: audit.Params, not_before_ns: int) -> None:
        self.path, self.items, self.result = path, queue.Queue(), None
        self.passes = threading.Event()
        head = {"registration": registration_json(registration), "params": params_json(params),
                "not_before_ns": not_before_ns}
        self.thread = threading.Thread(target=self._write, args=(head,), daemon=True)
        self.thread.start()

    def event(self, ev: dict) -> None:
        self.items.put(("event", json.dumps(ev, separators=(",", ":"))))

    def verify(self, t: int, frames: list[bytes]) -> None:
        self.items.put(("verify", t, frames))

    def passes_ended(self) -> None:
        """The session's passes are over: openings held back until now may be written."""
        self.passes.set()

    def close(self, released: tuple[list[bytes], list[bytes]]) -> dict:
        self.passes.set()
        self.items.put(("close", released))
        self.thread.join()
        return self.result

    def _write(self, head: dict) -> None:
        h, n, events, closed = hashlib.sha256(), 0, 0, False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                def put(s: str) -> None:
                    nonlocal n
                    b = s.encode()
                    f.write(b)
                    h.update(b)
                    n += len(b)

                put(json.dumps(head, separators=(",", ":"))[:-1] + ',"events":[')
                while True:
                    item = self.items.get()
                    if item[0] == "close":
                        closed = True
                        put('],"coins":' + json.dumps([c.hex() for c in item[1][0]]) + ',"append_coins":'
                            + json.dumps([c.hex() for c in item[1][1]]) + "}")
                        break
                    put("," if events else "")
                    events += 1
                    if item[0] == "event":
                        put(item[1])
                        continue
                    self.passes.wait()
                    put(f'{{"kind":"verify","t":{item[1]},"frames":[')
                    for k, body in enumerate(item[2]):
                        put(("," if k else "") + '"' + body.hex() + '"')
                    put("]}")
            self.result = {"path": str(self.path), "bytes": n, "sha256": h.hexdigest(), "events": events}
        except OSError as e:
            self.result = {"path": str(self.path), "error": f"{type(e).__name__}: {e}"}
            while not closed:
                closed = self.items.get()[0] == "close"
