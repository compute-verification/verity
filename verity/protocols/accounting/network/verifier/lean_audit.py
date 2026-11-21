"""The records check's launcher: every verdict is the Lean check's, `verity/network-records/v3` (`NetworkCheck.Records`,
the `network-records` executable of `verity/protocols/accounting/network/verifier/lean`). Each call is one unit
of the Lean protocol, sent to one `network-records verdicts` process kept open across calls and audits:

- `check_link`: egress links `["", link]` (`"clock"` when `link` is empty) over windows 0 … len(syncs) − 1, the first
  link carrying no frames and the spatial unit's clock as its syncs (`Audit.syncs`, each window's set by the first call
  that checked it), since the Lean check reads the unit's sync from the first egress link; the
  commitment holds this call's records of `link`, an attested one only if it is egress. With `grid` True it is the
  Lean's `verdicts`; with `grid` False (N2's head check holds the grid) `verdicts-no-grid`, every test of `verdicts`
  but the egress grid comparison (`Records.evaluateNoGrid`);
- `check_ingress_link`: the ingress link `[link]` alone, with this call's records of `link`, an attested one only if it is
  ingress.

The grid parameters, Σ_sync, key and session the call doesn't read (the ingress grid of an egress call, say, or the key
and the session when no attested record of the link is in it) are placeholders. Anything the Lean protocol can't
state (a key other than an `LmsKey` for an attested record, a raw grid that isn't a tuple of tuples of bytes or `None`,
a window or a grid field outside its integer width, a frame of other than B bytes, a session id not 32 bytes) raises
`ValueError`; the exe missing, or `lake` to build it, raises `FileNotFoundError`. Nothing is computed in Python instead.
`VERITY_NETWORK_RECORDS_EXE` names the executable, and then nothing but `verity` and the stdlib is needed besides it; by
default the first request runs `lake build network-records` in that package and starts what it builds. A process forked
after a request starts its own executable at its first; `SERVER`'s is closed at exit. The same process answers
`link-window` (`Server.call`), N2's verdict on one execution phase's proof, which no caller here sends,
`certificates`, the client's check of one window's served attestations (`certificate_refusals`), and `load`, the check of a run's load's registrations
(`relay_check.check_load`, `NetworkVerifier.Load.loadCheck`), with `transfer-registration`, its read of one
registration, which the load's agreement test holds to `transfer_rows.check_registration`.

`encode` is the commitment's encoding of one published record (`NetworkCheck/Records.lean`'s module docstring).
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import struct
import subprocess
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.commitments.crypto import lms
from verity.protocols.proof.spot.verifier.one_stage import lean_child as LC
from verity.protocols.accounting.network.verifier.certificate import (STATUS_CODES, Certificate, CertifiedRecord,
                                                        Direction, LmsKey, PublicKey)
from verity.protocols.accounting.network.verifier.commitment import CommittedRecord
from verity.protocols.accounting.network.verifier.grid import GridParams, BlockGrid, Status
from verity.protocols.accounting.network.verifier.schedule import ClockSync, LogicalFrame, SyncSet

PACKAGE = Path(__file__).resolve().parents[0] / "lean"
EXE = PACKAGE / ".lake" / "build" / "bin" / "network-records"
# `verity_difftest.build`'s lock: no two `lake build`s of a package overlap, across processes and suites.
BUILD_LOCK = Path(__file__).resolve().parents[5] / "verity/circuits/difftest" / ".lake" / "harness-build.lock"
RAW, COMMITTED, CERTIFIED = 0, 1, 2
_KEY = LmsKey(lms.Params(8, 8, 1, toy=True), bytes(24))
_SESSION = bytes(lms.SESSION_BYTES)
_SYNC_SET = SyncSet((0,), (1,))

Published = BlockGrid | CommittedRecord | CertifiedRecord


@dataclass(frozen=True, slots=True)
class Verdict:
    link: str
    window: int
    # None when accepted; otherwise the Lean check's violation: "status:<reason>", "advice", "mismatch", "duplicate",
    # "no-verdict" or "signature"
    violation: str | None

    @property
    def accepted(self) -> bool:
        return self.violation is None


def _text(s: str, what: str) -> bytes:
    b = s.encode("utf-8")
    if len(b) >= 1 << 16:
        raise ValueError(f"{what} is at most 65,535 UTF-8 bytes")
    return struct.pack(">H", len(b)) + b


def _u(x: int, width: int, what: str) -> bytes:
    if type(x) is not int or not 0 <= x < 1 << 8 * width:
        raise ValueError(f"{what} is an unsigned {8 * width}-bit integer, not {x!r}")
    return x.to_bytes(width, "big")


def _status(s: Status) -> bytes:
    return bytes([STATUS_CODES[s.reason]])


def _grid(grid) -> bytes:
    if type(grid) is not tuple or any(type(row) is not tuple for row in grid):
        raise ValueError("outside the Lean audit's domain: a raw record's grid is a tuple of tuples")
    out = [_u(len(grid), 4, "a grid's rows")]
    for row in grid:
        out.append(_u(len(row), 4, "a row's slots"))
        for slot in row:
            if slot is None:
                out.append(b"\0")
            elif isinstance(slot, (bytes, bytearray)):
                out += [b"\1", _u(len(slot), 4, "a frame's length"), bytes(slot)]
            else:
                raise ValueError(f"outside the Lean audit's domain: a slot is bytes or None, not {type(slot).__name__}")
    return b"".join(out)


def encode(record: Published) -> bytes:
    """`u8 kind ‖ u64be len(body) ‖ body`: raw, committed or attested (`service.encode`'s bytes)."""
    if isinstance(record, CertifiedRecord):
        c = record.certificate
        kind, body = CERTIFIED, b"".join([
            record.head.encode(), record.salt, _u(len(c.key_id), 2, "len(I)"), c.key_id,
            _text(c.link, "a certificate's link"),
            _u(c.direction.code, 1, "direction"), _u(c.window, 8, "window"), c.digest,
            _u(len(c.signature), 4, "len(signature)"), c.signature])
    elif isinstance(record, CommittedRecord):
        kind, body = COMMITTED, b"".join([_text(record.link, "link"), _u(record.window, 8, "window"),
                                          _status(record.status), _text(record.digest, "digest")])
    elif isinstance(record, BlockGrid):
        kind, body = RAW, b"".join([_text(record.link, "link"), _u(record.window, 8, "window"), _status(record.status),
                                    _grid(record.grid)])
    else:
        raise ValueError(f"not a published record: {type(record).__name__}")
    return bytes([kind]) + _u(len(body), 8, "len(body)") + body


def _grid_json(g: GridParams) -> dict:
    return {"allocation": g.allocation, "bucket_ms": g.bucket_ms, "buckets": g.buckets, "frame_bytes": g.frame_bytes,
            "queue_capacity": g.queue_capacity}


def _sync_json(s: ClockSync) -> list[int]:
    return [s.delta, s.rho.numerator, s.rho.denominator]


def case(egress: Sequence[tuple[str, Sequence[LogicalFrame], Sequence[ClockSync]]], ingress: Sequence[str], windows: int,
         grid: GridParams, ingress_grid: GridParams, sync_set: SyncSet, key: LmsKey, session: bytes,
         records: Sequence[Published]) -> dict:
    """A `serve` case of `verity/network-records/v3`: the unit's egress links with their frames and syncs, its ingress
    links, the attester's key and session, and the records."""
    p = key.params
    params = {"egress": {"grid": _grid_json(grid), "links": [l for l, _, _ in egress]},
              "ingress": {"grid": _grid_json(ingress_grid), "links": list(ingress)},
              "key": key.public.hex(), "lms": {"H": p.H, "N": p.N, "W": p.W, "toy": p.toy},
              "session": bytes(session).hex(),
              "sync_set": {"deltas": list(sync_set.deltas),
                           "rhos": [[r.numerator, r.denominator] for r in sync_set.rhos]},
              "windows": windows}
    evidence = {"links": [{"frames": [[f.payload.hex(), f.anchor, f.index] for f in frames],
                           "syncs": [_sync_json(s) for s in syncs]} for _, frames, syncs in egress]}
    return {"commitment": b"".join(encode(r) for r in records).hex(),
            "evidence": json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode().hex(), "params": params}


def build() -> Path:
    """`lake build network-records` in its package, so the executable is the source's; the executable's path."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(["lake", "build", "network-records"], cwd=PACKAGE, capture_output=True, text=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"lake is not on the PATH, so network-records in {PACKAGE} cannot be built; "
                                    "set VERITY_NETWORK_RECORDS_EXE to a built one") from None
    if p.returncode != 0:
        raise RuntimeError(f"lake build network-records in {PACKAGE} failed ({p.returncode}):\n{p.stdout}{p.stderr}")
    return EXE


class Server:
    """One `network-records verdicts` process, started at the first request and answering in order. The executable is
    `exe`, else `VERITY_NETWORK_RECORDS_EXE`, else the package's, built (`build`) before the first start."""

    def __init__(self, exe: Path | str | None = None):
        named = exe or os.environ.get("VERITY_NETWORK_RECORDS_EXE")
        self.exe = Path(named) if named else None
        self._pid = os.getpid()
        self._proc: subprocess.Popen | None = None
        self._stderr = None
        self._lock = threading.Lock()

    def _own(self) -> None:
        """In a process forked from the one that started the executable, forget the parent's: its pipe is the parent's."""
        if self._pid != os.getpid():
            if self._proc is not None:
                self._proc.stdin.close()
                self._proc.stdout.close()
            self._pid, self._proc, self._lock = os.getpid(), None, threading.Lock()

    def verdicts(self, case: dict, fn: str = "verdicts") -> list[dict]:
        """Every execution phase's verdict on `case` (`{"direction", "link", "verdict", "window"}`, in the audit's
        order) by `fn`, `verdicts` or `verdicts-no-grid`; `ValueError` when the Lean check refuses the case."""
        return self.call(fn, case)["link_windows"]

    def call(self, fn: str, case: dict) -> Any:
        """`fn`'s output on `case` (`verdicts`, `verdicts-no-grid`, `link-window`: `NetworkCheck.TransferBlocks`'s verdict
        of record on one execution phase, `certificates`, `load` or `transfer-registration`); `ValueError` when the Lean
        refuses the case. The process runs in `lean_child.lean_env`; `lean_child.LeanPanic` where it wrote a panic line,
        whatever it answered, and `lean_child.LeanExit` where it ended before answering."""
        self._own()
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                if self.exe is None:
                    self.exe = build()
                if not self.exe.is_file():
                    raise FileNotFoundError(f"{self.exe}: build it with `lake build network-records` in "
                                            f"verity/protocols/accounting/network/verifier/lean, or set VERITY_NETWORK_RECORDS_EXE")
                self._stderr = tempfile.TemporaryFile()
                self._proc = subprocess.Popen([str(self.exe), "verdicts"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                              stderr=self._stderr, text=True, encoding="utf-8", env=LC.lean_env())
            proc = self._proc
            try:
                proc.stdin.write(json.dumps({"fn": fn, "input": case}, separators=(",", ":")) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except BrokenPipeError:
                line = ""
            # the child writes its panic line before its answer, so it is in the file once the answer is read
            self._stderr.seek(0)
            panic = LC.panic_line(self._stderr.read())
            if panic is not None or not line.endswith("\n"):
                self._proc = None
                if proc.poll() is None:
                    proc.kill()
                code = proc.wait()
                if panic is not None:
                    raise LC.LeanPanic(f"{LC.LEAN_PANIC}: {panic}")
                raise LC.LeanExit(LC.exit_why(code) or f"{LC.LEAN_EXIT}: {self.exe} verdicts ended (exit {code}) "
                                  "before it answered")
        msg = json.loads(line)
        if "reject" in msg:
            raise ValueError(f"outside the Lean audit's domain: {msg['reject']}")
        if "output" not in msg:
            raise RuntimeError(f"{self.exe} verdicts: {msg}")
        return msg["output"]

    def close(self) -> None:
        self._own()
        with self._lock:
            if self._proc is not None:
                self._proc.stdin.close()
                self._proc.wait()
                self._proc = None


SERVER = Server()
atexit.register(SERVER.close)


def _need_key(records: Sequence[Published], key: PublicKey | None, session: bytes | None) -> None:
    if (key is None or session is None) and any(isinstance(r, CertifiedRecord) for r in records):
        raise ValueError("certified records are checked against the network attester's public key in its session: "
                         "pass key and session")


def _mine(link: str, records: Sequence[Published], direction: Direction) -> list[Published]:
    return [r for r in records if r.link == link and not (isinstance(r, CertifiedRecord) and r.direction is not direction)]


def _lean_key(records: Sequence[Published], key: PublicKey | None, session: bytes | None) -> tuple[LmsKey, bytes]:
    if not any(isinstance(r, CertifiedRecord) for r in records):
        return _KEY, _SESSION
    if not isinstance(key, LmsKey):
        raise ValueError(f"outside the Lean audit's domain: certified records under a {type(key).__name__}, not an LmsKey")
    return key, session


def _verdicts(out: list[dict], direction: Direction, link: str, windows: int) -> list[Verdict]:
    vs = [Verdict(link, v["window"], None if v["verdict"] == "accept" else v["verdict"]) for v in out
          if v["direction"] == direction.value and v["link"] == link]
    if [v.window for v in vs] != list(range(windows)):
        raise RuntimeError(f"the Lean check gave {link}'s windows {[v.window for v in vs]}, not 0 … {windows - 1}")
    return vs


@dataclass
class Audit:
    """Every verdict of a run, each the Lean check's, and the spatial unit's clock: each window's sync, set by the first
    egress link checked for that window. The run is accepted only if every execution phase is."""

    verdicts: list[Verdict] = field(default_factory=list)
    syncs: dict[int, ClockSync] = field(default_factory=dict)
    server: Server = field(default=SERVER, repr=False, compare=False)

    def check_link(self, link: str, records: Sequence[Published], frames: Sequence[LogicalFrame],
                   syncs: Sequence[ClockSync], sync_set: SyncSet, params: GridParams,
                   key: PublicKey | None = None, session: bytes | None = None, grid: bool = True) -> list[Verdict]:
        """One egress link's verdicts for windows 0 … len(syncs) − 1. With `grid` False it is the Lean's
        `verdicts-no-grid`, which computes no Program grid (N2 gives no `frames`); a raw or committed record of `link`
        then raises `ValueError` here, with or without windows, as `verdicts-no-grid` refuses."""
        _need_key(records, key, session)
        frames, syncs = list(frames), list(syncs)
        mine = _mine(link, records, Direction.EGRESS)
        if not grid and any(not isinstance(r, CertifiedRecord) for r in mine):
            raise ValueError("only a certified record's grid is held by its head check: compare the others' (grid=True)")
        lean = []
        if syncs:
            clock = [self.syncs.get(w, s) for w, s in enumerate(syncs)]
            unit = "clock" if link == "" else ""
            out = self.server.verdicts(case([(unit, [], clock), (link, frames, syncs)], [], len(syncs), params, params,
                                            sync_set, *_lean_key(mine, key, session), mine),
                                       "verdicts" if grid else "verdicts-no-grid")
            lean = _verdicts(out, Direction.EGRESS, link, len(syncs))
        for w, s in enumerate(syncs):
            self.syncs.setdefault(w, s)
        self.verdicts.extend(lean)
        return lean

    def check_ingress_link(self, link: str, records: Sequence[Published], windows: int, key: PublicKey | None = None,
                           params: GridParams | None = None, session: bytes | None = None) -> list[Verdict]:
        _need_key(records, key, session)
        mine = _mine(link, records, Direction.INGRESS)
        if params is None and any(isinstance(r, CertifiedRecord) for r in mine):
            raise ValueError("certified ingress records are checked against the grid parameters: pass params")
        lean = []
        if windows > 0:
            grid = params or GridParams(1, 1, 1)
            out = self.server.verdicts(case([], [link], windows, grid, grid, _SYNC_SET, *_lean_key(mine, key, session),
                                            mine))
            lean = _verdicts(out, Direction.INGRESS, link, windows)
        self.verdicts.extend(lean)
        return lean

    @property
    def accepted(self) -> bool:
        return bool(self.verdicts) and all(v.accepted for v in self.verdicts)

    @property
    def violations(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.accepted]


def _attestation_json(c: Certificate) -> dict:
    if not (type(c.key_id) is bytes and type(c.link) is str and isinstance(c.direction, Direction)
            and type(c.window) is int and c.window >= 0 and type(c.digest) is bytes and type(c.signature) is bytes):
        raise ValueError(f"outside the Lean audit's domain: a certificate of ill-typed fields: {c!r}")
    return {"key_id": c.key_id.hex(), "link": c.link, "direction": c.direction.code, "window": c.window,
            "digest": c.digest.hex(), "signature": c.signature.hex()}


def certificate_refusals(key: PublicKey, session: bytes, link: str, window: int, certificates: Sequence[Certificate],
                         clocks: Sequence[Certificate], seen: Sequence[bytes] = (), server: Server | None = None
                         ) -> tuple[list[str | None], list[str | None]]:
    """Each attestation's refusal in one window's answer, then each clock attestation's, None where the client keeps it:
    the Lean's (`NetworkCheck.TransferBlocks.refusals`, `network-records verdicts`' ``certificates``). A key other than
    an `LmsKey`, an attestation of ill-typed fields, or anything else the Lean refuses to read raises `ValueError`."""
    if not isinstance(key, LmsKey):
        raise ValueError(f"outside the Lean audit's domain: certificates under a {type(key).__name__}, not an LmsKey")
    p = key.params
    case = {"lms": {"H": p.H, "N": p.N, "W": p.W, "toy": p.toy}, "key": key.public.hex(),
            "session": bytes(session).hex(), "link": link, "window": window,
            "certificates": [_attestation_json(c) for c in certificates],
            "clocks": [_attestation_json(c) for c in clocks], "seen": [bytes(d).hex() for d in seen]}
    out = (server or SERVER).call("certificates", case)
    return out["certificates"], out["clocks"]
