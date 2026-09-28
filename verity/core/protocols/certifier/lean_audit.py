"""`audit.Audit` with every verdict computed by the Lean check, `verity/network-records/v1` (`NetworkCheck.Records`, the
`network-records` executable of `verity/core/protocols/network/lean`), which `audit.py` is the reference for.

`Audit` here has `audit.Audit`'s interface and returns its `Verdict`s, so a caller switches by its import line alone.
`audit.Audit` keeps running beside the Lean check as its reference, on every call with the same arguments: the verdicts
are the Lean's, the Python's are kept (`Audit.reference`), a link-window where the two differ is a `disagreement` violation,
and a call that one answers and the other raises on raises. Each call is one unit of the Lean protocol, sent to one
`network-records verdicts` process kept open across calls and audits:

- `check_link`: egress links `["", link]` (`"clock"` when `link` is empty) over windows 0 … len(syncs) − 1, the first
  link carrying no frames and the spatial unit's clock as its syncs (`Audit.syncs`, each window's set by the first call
  that checked it, as `audit.Audit` sets it), since the Lean check reads the unit's sync from the first egress link; the
  commitment holds this call's records of `link`, a certified one only if it is egress;
- `check_ingress_link`: the ingress link `[link]` alone, with this call's records of `link`, a certified one only if it is
  ingress.

The grid parameters, Σ_sync and key the call doesn't read (the ingress grid of an egress call, say, or the key when no
certified record of the link is in it) are placeholders. Anything the Lean protocol can't state (a key other than an
`LmsKey` for a certified record, a raw grid that isn't a tuple of tuples of bytes or `None`, a window or a grid field
outside its integer width, a frame of other than B bytes) raises `ValueError`, as does a call `audit.Audit` refuses; the
exe missing, or `lake` to build it, raises `FileNotFoundError`. Nothing is computed in Python instead.
`VERITY_NETWORK_RECORDS_EXE` names the executable, and then nothing but `verity` and the stdlib is needed besides it; by
default the first request runs `lake build network-records` in that package and starts what it builds. A process forked
after a request starts its own executable at its first; `SERVER`'s is closed at exit.

`encode` is the commitment's encoding of one published record (`NetworkCheck/Records.lean`'s module docstring).
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import struct
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from verity.ml.crypto import lms
from verity.core.protocols.certifier import audit
from verity.core.protocols.certifier.audit import Published, Verdict
from verity.core.protocols.certifier.certificate import STATUS_CODES, CertifiedRecord, Direction, LmsKey, PublicKey
from verity.core.protocols.certifier.commitment import CommittedRecord
from verity.core.protocols.certifier.grid import GridParams, Record, Status
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet

PACKAGE = Path(__file__).resolve().parents[1] / "network" / "lean"
EXE = PACKAGE / ".lake" / "build" / "bin" / "network-records"
# `verity_difftest.build`'s lock: no two `lake build`s of a package overlap, across processes and suites.
BUILD_LOCK = Path(__file__).resolve().parents[2] / "difftest" / "lean" / ".lake" / "harness-build.lock"
RAW, COMMITTED, CERTIFIED = 0, 1, 2
DISAGREEMENT = "disagreement"
_KEY = LmsKey(lms.Params(8, 8, 1, toy=True), bytes(24))
_SYNC_SET = SyncSet((0,), (1,))


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
    """`u8 kind ‖ u64be len(body) ‖ body`: raw, committed or certified (`service.encode`'s bytes)."""
    if isinstance(record, CertifiedRecord):
        c = record.certificate
        kind, body = CERTIFIED, b"".join([
            record.head.encode(), _u(len(c.key_id), 2, "len(I)"), c.key_id, _text(c.link, "a certificate's link"),
            _u(c.direction.code, 1, "direction"), _u(c.window, 8, "window"), c.digest,
            _u(len(c.signature), 4, "len(signature)"), c.signature])
    elif isinstance(record, CommittedRecord):
        kind, body = COMMITTED, b"".join([_text(record.link, "link"), _u(record.window, 8, "window"),
                                          _status(record.status), _text(record.digest, "digest")])
    elif isinstance(record, Record):
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
         grid: GridParams, ingress_grid: GridParams, sync_set: SyncSet, key: LmsKey,
         records: Sequence[Published]) -> dict:
    """A `serve` case of `verity/network-records/v1`: the unit's egress links with their frames and syncs, its ingress
    links, and the records."""
    p = key.params
    params = {"egress": {"grid": _grid_json(grid), "links": [l for l, _, _ in egress]},
              "ingress": {"grid": _grid_json(ingress_grid), "links": list(ingress)},
              "key": key.public.hex(), "lms": {"H": p.H, "N": p.N, "W": p.W, "toy": p.toy},
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
        self._lock = threading.Lock()

    def _own(self) -> None:
        """In a process forked from the one that started the executable, forget the parent's: its pipe is the parent's."""
        if self._pid != os.getpid():
            if self._proc is not None:
                self._proc.stdin.close()
                self._proc.stdout.close()
            self._pid, self._proc, self._lock = os.getpid(), None, threading.Lock()

    def verdicts(self, case: dict) -> list[dict]:
        """Every link-window's verdict on `case` (`{"direction", "link", "verdict", "window"}`, in the audit's order);
        `ValueError` when the Lean check refuses the case."""
        self._own()
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                if self.exe is None:
                    self.exe = build()
                if not self.exe.is_file():
                    raise FileNotFoundError(f"{self.exe}: build it with `lake build network-records` in "
                                            f"verity/core/protocols/network/lean, or set VERITY_NETWORK_RECORDS_EXE")
                self._proc = subprocess.Popen([str(self.exe), "verdicts"], stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE, text=True, encoding="utf-8")
            proc = self._proc
            try:
                proc.stdin.write(json.dumps({"fn": "verdicts", "input": case}, separators=(",", ":")) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except BrokenPipeError:
                line = ""
            if not line.endswith("\n"):
                self._proc = None
                raise RuntimeError(f"{self.exe} verdicts ended (exit {proc.wait()})")
        msg = json.loads(line)
        if "reject" in msg:
            raise ValueError(f"outside the Lean audit's domain: {msg['reject']}")
        if "output" not in msg:
            raise RuntimeError(f"{self.exe} verdicts: {msg}")
        return msg["output"]["link_windows"]

    def close(self) -> None:
        self._own()
        with self._lock:
            if self._proc is not None:
                self._proc.stdin.close()
                self._proc.wait()
                self._proc = None


SERVER = Server()
atexit.register(SERVER.close)


def _need_key(records: Sequence[Published], key: PublicKey | None) -> None:
    if key is None and any(isinstance(r, CertifiedRecord) for r in records):
        raise ValueError("certified records are checked against the network certifier's public key: pass key")


def _mine(link: str, records: Sequence[Published], direction: Direction) -> list[Published]:
    return [r for r in records if r.link == link and not (isinstance(r, CertifiedRecord) and r.direction is not direction)]


def _lean_key(records: Sequence[Published], key: PublicKey | None) -> LmsKey:
    if not any(isinstance(r, CertifiedRecord) for r in records):
        return _KEY
    if not isinstance(key, LmsKey):
        raise ValueError(f"outside the Lean audit's domain: certified records under a {type(key).__name__}, not an LmsKey")
    return key


def _verdicts(out: list[dict], direction: Direction, link: str, windows: int) -> list[Verdict]:
    vs = [Verdict(link, v["window"], None if v["verdict"] == "accept" else v["verdict"]) for v in out
          if v["direction"] == direction.value and v["link"] == link]
    if [v.window for v in vs] != list(range(windows)):
        raise RuntimeError(f"the Lean check gave {link}'s windows {[v.window for v in vs]}, not 0 … {windows - 1}")
    return vs


@dataclass
class Audit:
    """`audit.Audit`, each verdict the Lean check's, with `audit.Audit` (`python`) run beside it on every call as the
    reference: `lean` holds the Lean check's verdicts and `reference` the Python's, and a link-window where they differ is
    a `disagreement` violation in `verdicts`. A call where one raises and the other answers raises."""

    verdicts: list[Verdict] = field(default_factory=list)
    syncs: dict[int, ClockSync] = field(default_factory=dict)
    server: Server = field(default=SERVER, repr=False, compare=False)
    python: audit.Audit = field(default_factory=audit.Audit, repr=False, compare=False)
    lean: list[Verdict] = field(default_factory=list, repr=False, compare=False)
    reference: list[Verdict] = field(default_factory=list, repr=False, compare=False)

    def _beside(self, lean: list[Verdict], reference: Callable[[], list[Verdict]]) -> list[Verdict]:
        try:
            ref = reference()
        except Exception as e:
            raise RuntimeError(f"the reference audit raised where the Lean check answered: {e!r}") from e
        if [(v.link, v.window) for v in ref] != [(v.link, v.window) for v in lean]:
            raise RuntimeError(f"the reference audit gave link-windows {[(v.link, v.window) for v in ref]}, "
                               f"the Lean check {[(v.link, v.window) for v in lean]}")
        vs = [l if l == r else Verdict(l.link, l.window, DISAGREEMENT) for l, r in zip(lean, ref)]
        self.lean.extend(lean)
        self.reference.extend(ref)
        self.verdicts.extend(vs)
        return vs

    def check_link(self, link: str, records: Sequence[Published], frames: Sequence[LogicalFrame],
                   syncs: Sequence[ClockSync], sync_set: SyncSet, params: GridParams,
                   key: PublicKey | None = None) -> list[Verdict]:
        _need_key(records, key)
        frames, syncs = list(frames), list(syncs)
        lean = []
        if syncs:
            mine = _mine(link, records, Direction.EGRESS)
            clock = [self.syncs.get(w, s) for w, s in enumerate(syncs)]
            unit = "clock" if link == "" else ""
            out = self.server.verdicts(case([(unit, [], clock), (link, frames, syncs)], [], len(syncs), params, params,
                                            sync_set, _lean_key(mine, key), mine))
            lean = _verdicts(out, Direction.EGRESS, link, len(syncs))
        vs = self._beside(lean, lambda: self.python.check_link(link, records, frames, syncs, sync_set, params, key))
        for w, s in enumerate(syncs):
            self.syncs.setdefault(w, s)
        return vs

    def check_ingress_link(self, link: str, records: Sequence[Published], windows: int, key: PublicKey | None = None,
                           params: GridParams | None = None) -> list[Verdict]:
        _need_key(records, key)
        mine = _mine(link, records, Direction.INGRESS)
        if params is None and any(isinstance(r, CertifiedRecord) for r in mine):
            raise ValueError("certified ingress records are checked against the grid parameters: pass params")
        lean = []
        if windows > 0:
            grid = params or GridParams(1, 1, 1)
            out = self.server.verdicts(case([], [link], windows, grid, grid, _SYNC_SET, _lean_key(mine, key), mine))
            lean = _verdicts(out, Direction.INGRESS, link, windows)
        return self._beside(lean, lambda: self.python.check_ingress_link(link, records, windows, key, params))

    @property
    def accepted(self) -> bool:
        return bool(self.verdicts) and all(v.accepted for v in self.verdicts)

    @property
    def violations(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.accepted]
