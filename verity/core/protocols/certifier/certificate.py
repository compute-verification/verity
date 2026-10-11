"""Network certificates: the network certifier's signed record of each link-window, on SHA-512 (record v2).

The network certifier is the auditor-trusted device on a node's uplink that shapes each link into the grid
(`grid.Certifier`). Once per link-window it signs one **network certificate**: a signature over the SHA-512 digest of the window's record. A
certificate is witness: it enters proofs and never reaches the auditor in the clear, so its hashes are unsalted.

**The record (v2).** The record is still exactly the grid and one status (W5), encoded with padding elided:

    R = u16be len(link) ‖ link ‖ u8 direction ‖ u64be window
      ‖ u32be T ‖ u32be r ‖ u32be B ‖ u32be τ_b
      ‖ u8 status ‖ u8 deadline ‖ parameters (64 bytes)
      ‖ u32be n_0 ‖ … ‖ u32be n_{T−1} ‖ root (64 bytes)
    record digest = tagged_sha512("verity-network-certifier/record/v2", R)

`direction` is 0 for egress and 1 for ingress, `status` is `STATUS_CODES`'s code, `deadline` is 1 when the certifier
settled the record by the verdict deadline, and `parameters` is `parameters_digest`. The counts n_t are the occupancy,
which the schedule check reads exhaustively. `root` is a Merkle root over the window's N = Σ n_t real frames, in
the order `grid.fill` places them, each of B bytes:

    leaf(f)    = SHA-512( block("verity-network-certifier/frame/v2\\0" ‖ u32be B) ‖ f )
    E_0        = SHA-512( block("verity-network-certifier/empty/v2\\0") )
    node(a, b) = SHA-512( block("verity-network-certifier/node/v2\\0") ‖ a ‖ b ),   E_{l+1} = node(E_l, E_l)

`block(x)` is x zero-padded to one 128-byte SHA-512 block, so every message starts from a constant midstate. The tree
has the fixed depth D = ⌈log2(T·r)⌉: leaf k is frame k for k < N and E_0 after, so a padding slot never enters it.
Opening one frame (`open_frame`, `verify_frame`) costs D siblings, the same at every occupancy.

**The certificate** signs a 64-byte message that binds the record digest to the key, the certifier session and the
link-window, so a certificate can't be replayed onto another window, link, direction, key or session. It is LMS's
network-certificate role digest (`verity.ml.crypto.lms.certificate_message`), each field but I and the window prefixed
with its length as a u64be:

    message = SHA-512( [tag] ‖ I ‖ [ [session] ‖ [link] ‖ [direction] ‖ u64be window ] ‖ [record digest] ),
              tag = "verity/network-certificate-message/v2", direction "egress" or "ingress", [x] = u64be len(x) ‖ x

`session` is the certifier session's id, 32 bytes (`lms.SESSION_BYTES`) that the challenger draws for the session and
issues to the certifier; the audit takes it from its parameters, as it takes the key, so the record head and the
certificate's encoding don't carry it. The tag is the message's own, distinct from the ids of the checks that read it
(`verity/network-certificate/v3`, `verity/network-records/v2`).

The scheme is LMS over SHA-512 (`lms.DEPLOYED`: n = 64, w = 1, h = 20, the whole SHA-512 output, not truncated), whose
unforgeability is ``euf-cma/lms-sha512-n64``. The certifier's signing step (`Issuer`) is handed it as a `Signer`
(`LmsSigner`), and the audit as a `PublicKey` (`LmsKey`). A signer whose leaf state is missing, behind or held by another
signer refuses (`SignerRefused`), and then the link-window has no certificate: it fails closed, as a `no-verdict`. So
does a signer asked to start a session its state has already started (`Signer.start`): the issuer starts its session
before its first signature, and a refused start leaves every link-window of the run without a certificate.
"""

from __future__ import annotations

import enum
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha512
from verity.ml.crypto import lms
from verity.core.protocols.certifier.grid import Grid, GridParams, Reason, Record, Status, row_left_packed
from verity.core.protocols.certifier.schedule import SyncSet

RECORD_TAG = "verity-network-certifier/record/v2"
PARAMETERS_TAG = "verity-network-certifier/parameters/v1"
FRAME_TAG = b"verity-network-certifier/frame/v2\0"
EMPTY_TAG = b"verity-network-certifier/empty/v2\0"
NODE_TAG = b"verity-network-certifier/node/v2\0"
DIGEST_BYTES = 64

STATUS_CODES: dict[Reason | None, int] = {
    None: 0,
    Reason.MONITOR_UNAVAILABLE: 1,
    Reason.OVERFLOW: 2,
    Reason.MALFORMED: 3,
    Reason.DROPPED: 4,
    Reason.DELIVERY_REFUSED: 5,
    Reason.COVERAGE_NEVER_ESTABLISHED: 6,
}


class Direction(enum.Enum):
    EGRESS = "egress"
    INGRESS = "ingress"

    @property
    def code(self) -> int:
        return 0 if self is Direction.EGRESS else 1


def _block(*parts: bytes) -> bytes:
    data = b"".join(parts)
    if len(data) > 128:
        raise ValueError("a tag block is one 128-byte SHA-512 block")
    return data + bytes(128 - len(data))


_NODE_BLOCK = _block(NODE_TAG)


def _u(value: int, width: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value < 1 << (8 * width):
        raise ValueError(f"{name} is an unsigned {8 * width}-bit integer, not {value!r}")
    return value.to_bytes(width, "big")


def _text(value: str, name: str) -> bytes:
    data = value.encode("utf-8")
    return _u(len(data), 2, f"len({name})") + data


def frame_leaf(frame: bytes, frame_bytes: int) -> bytes:
    if len(frame) != frame_bytes:
        raise ValueError(f"frames are {frame_bytes} bytes, not {len(frame)}")
    return hashlib.sha512(_block(FRAME_TAG, _u(frame_bytes, 4, "B")) + frame).digest()


def node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha512(_NODE_BLOCK + left + right).digest()


@lru_cache(maxsize=64)
def empty(level: int) -> bytes:
    """E_level: the root of a subtree of 2^level empty leaves."""
    if level == 0:
        return hashlib.sha512(_block(EMPTY_TAG)).digest()
    below = empty(level - 1)
    return node(below, below)


def depth(params: GridParams) -> int:
    """D = ⌈log2(T·r)⌉, the tree's fixed depth: room for every slot of the window."""
    return (params.buckets * params.allocation - 1).bit_length()


def real_frames(grid: Grid, params: GridParams) -> list[bytes]:
    """The window's real frames in the order `grid.fill` places them; padding is elided. A grid that isn't left-packed
    (W4) has no encoding."""
    if len(grid) != params.buckets:
        raise ValueError(f"a grid has {params.buckets} rows, not {len(grid)}")
    out: list[bytes] = []
    for row in grid:
        if len(row) != params.allocation:
            raise ValueError(f"a row has {params.allocation} slots, not {len(row)}")
        if not row_left_packed(row):
            raise ValueError("a row that isn't left-packed has no encoding (W4)")
        out += [f for f in row if f is not None]
    return out


def _levels(frames: Sequence[bytes], params: GridParams) -> list[list[bytes]]:
    """The tree's non-empty prefix at each level, leaves first; every node to its right is the level's E."""
    if len(frames) > params.buckets * params.allocation:
        raise ValueError("more frames than the window has slots")
    level = [frame_leaf(f, params.frame_bytes) for f in frames]
    levels = [level]
    for lv in range(depth(params)):
        if len(level) % 2:
            level = level + [empty(lv)]
        level = [node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        levels.append(level)
    return levels


def frames_root(frames: Sequence[bytes], params: GridParams) -> bytes:
    top = _levels(frames, params)[-1]
    return top[0] if top else empty(depth(params))


def open_frame(frames: Sequence[bytes], params: GridParams, k: int) -> tuple[bytes, ...]:
    """Frame k's authentication path: D siblings, leaf level first."""
    if not 0 <= k < len(frames):
        raise ValueError(f"frame {k} is not among the window's {len(frames)}")
    path = []
    for lv, level in enumerate(_levels(frames, params)[:-1]):
        sibling = k ^ 1
        path.append(level[sibling] if sibling < len(level) else empty(lv))
        k >>= 1
    return tuple(path)


def verify_frame(root: bytes, params: GridParams, k: int, frame: bytes, path: Sequence[bytes]) -> bool:
    """Whether `frame` is leaf k under `root`. A verifier also checks k < N = Σ n_t from the record's counts, since a
    leaf past N is E_0, which no frame hashes to."""
    if len(path) != depth(params) or not 0 <= k < 1 << len(path) or len(frame) != params.frame_bytes:
        return False
    digest = frame_leaf(frame, params.frame_bytes)
    for sibling in path:
        digest = node(digest, sibling) if k % 2 == 0 else node(sibling, digest)
        k >>= 1
    return digest == root


def parameters_digest(params: GridParams, sync_set: SyncSet | None = None) -> bytes:
    """The SHA-512 digest of what the protocol fixes before the window (W1): every grid parameter, Q included, and
    Σ_sync where a clock applies (egress)."""
    manifest = {
        "buckets": params.buckets, "allocation": params.allocation, "frame_bytes": params.frame_bytes,
        "bucket_ms": params.bucket_ms, "queue_capacity": params.queue_capacity,
        "sync_set": None if sync_set is None else {"deltas": list(sync_set.deltas),
                                                   "rhos": [str(r) for r in sync_set.rhos]},
    }
    return tagged_sha512(PARAMETERS_TAG, canonical_json_bytes(manifest))


@dataclass(frozen=True, slots=True)
class RecordHead:
    """A v2 record with its frames replaced by their root: everything the record digest reads, and what a proof opens
    to read the counts and the status. Like the certificate, it is witness."""

    link: str
    direction: Direction
    window: int
    buckets: int
    allocation: int
    frame_bytes: int
    bucket_ms: int
    status: Status
    deadline: bool
    parameters: bytes
    counts: tuple[int, ...]
    root: bytes

    def __post_init__(self) -> None:
        if len(self.counts) != self.buckets or any(not 0 <= n <= self.allocation for n in self.counts):
            raise ValueError(f"a record has {self.buckets} counts of 0 … {self.allocation}")
        if len(self.parameters) != DIGEST_BYTES or len(self.root) != DIGEST_BYTES:
            raise ValueError("the parameters digest and the root are 64 bytes")

    def encode(self) -> bytes:
        return b"".join([
            _text(self.link, "link"), _u(self.direction.code, 1, "direction"), _u(self.window, 8, "window"),
            _u(self.buckets, 4, "T"), _u(self.allocation, 4, "r"), _u(self.frame_bytes, 4, "B"),
            _u(self.bucket_ms, 4, "τ_b"),
            _u(STATUS_CODES[self.status.reason], 1, "status"), _u(int(self.deadline), 1, "deadline"), self.parameters,
            *(_u(n, 4, "count") for n in self.counts), self.root,
        ])

    def digest(self) -> bytes:
        return tagged_sha512(RECORD_TAG, self.encode())


def record_head(link: str, direction: Direction, window: int, grid: Grid, status: Status, params: GridParams,
                parameters: bytes, *, deadline: bool = True) -> RecordHead:
    frames = real_frames(grid, params)
    return RecordHead(link, direction, window, params.buckets, params.allocation, params.frame_bytes, params.bucket_ms,
                      status, deadline, parameters, tuple(sum(f is not None for f in row) for row in grid),
                      frames_root(frames, params))


def certificate_message(key_id: bytes, session: bytes, link: str, direction: Direction, window: int,
                        digest: bytes) -> bytes:
    if len(digest) != DIGEST_BYTES:
        raise ValueError("a record digest is 64 bytes")
    return lms.certificate_message(key_id, session, link, direction.value, window, digest)


class SignerRefused(Exception):
    """The signer's one-time-key state is missing, behind or held by another signer, so it signs nothing rather than risk
    reusing a key; or the session it is asked to start is already started."""


class Signer(Protocol):
    """A stateful hash-based signer. `start` records a session id as started before returning, and raises
    `SignerRefused` when its state has started it already or this signer has started another. `sign` uses a fresh
    one-time key and records it as used before returning, and raises `SignerRefused` when its state is missing or
    behind."""

    @property
    def key_id(self) -> bytes: ...

    def start(self, session: bytes) -> None: ...

    def sign(self, message: bytes) -> bytes: ...


class PublicKey(Protocol):
    @property
    def key_id(self) -> bytes: ...

    def verify(self, message: bytes, signature: bytes) -> bool: ...


@dataclass(frozen=True, slots=True)
class LmsKey:
    """A certifier's LMS public key, ``I ‖ T[1]``, under one parameter set."""

    params: lms.Params
    public: bytes

    @property
    def key_id(self) -> bytes:
        return self.public[:16]

    def verify(self, message: bytes, signature: bytes) -> bool:
        try:
            sig = lms.Signature.decode(self.params, signature)
        except ValueError:
            return False
        return lms.verify(self.params, self.public, message, sig)


class LmsSigner:
    """A certifier's LMS key and its leaf state (`lms.LeafState` at `state`), held from the first signature on, so a second
    signer on the same state is refused. ``floor`` is the next leaf known unused, such as 1 + the last q published."""

    def __init__(self, key: lms.PrivateKey, state: str | Path, *, floor: int = 0) -> None:
        self.key, self.state, self.floor = key, Path(state), floor
        self._leaves: lms.LeafState | None = None

    @classmethod
    def create(cls, key: lms.PrivateKey, state: str | Path) -> LmsSigner:
        """A new key's signer, its leaf state created at leaf 0."""
        lms.LeafState.create(state, lms.public_key(key), key.params.H)
        return cls(key, state)

    @property
    def key_id(self) -> bytes:
        return self.key.I

    def public_key(self) -> LmsKey:
        return LmsKey(self.key.params, lms.public_key(self.key))

    def _state(self) -> lms.LeafState:
        if self._leaves is None:
            self._leaves = lms.LeafState(self.state, lms.public_key(self.key), self.key.params.H, floor=self.floor)
        return self._leaves

    def start(self, session: bytes) -> None:
        try:
            self._state().start(session)
        except lms.StateError as e:
            raise SignerRefused(str(e)) from None

    def sign(self, message: bytes) -> bytes:
        try:
            return lms.sign(self.key, self._state(), message).encode()
        except lms.StateError as e:
            raise SignerRefused(str(e)) from None

    def close(self) -> None:
        if self._leaves is not None:
            self._leaves.close()
            self._leaves = None


@dataclass(frozen=True, slots=True)
class Certificate:
    key_id: bytes
    link: str
    direction: Direction
    window: int
    digest: bytes
    signature: bytes

    def message(self, session: bytes) -> bytes:
        """The message the certificate signs in certifier session ``session``."""
        return certificate_message(self.key_id, session, self.link, self.direction, self.window, self.digest)


@dataclass(frozen=True, slots=True)
class CertifiedRecord:
    """What the certifier hands over per link-window: the certificate and the record head it signs."""

    head: RecordHead
    certificate: Certificate

    @property
    def link(self) -> str:
        return self.head.link

    @property
    def direction(self) -> Direction:
        return self.head.direction

    @property
    def window(self) -> int:
        return self.head.window

    @property
    def status(self) -> Status:
        return self.head.status

    def verify(self, key: PublicKey, session: bytes) -> bool:
        """The certificate is signed by `key` in certifier session `session`, names the head's link-window and signs the
        head's digest."""
        c, h = self.certificate, self.head
        if not (c.key_id == key.key_id and (c.link, c.direction, c.window) == (h.link, h.direction, h.window)
                and c.digest == h.digest()):
            return False
        try:
            message = c.message(session)
        except ValueError:
            return False
        return key.verify(message, c.signature)


@dataclass
class Issuer:
    """The network certifier's signing step, beside its grid (`grid.Certifier`): one certificate per closed link-window,
    in certifier session `session`. Before its first signature it starts the session on its signer, which refuses a
    session its state has already started. A refusal, of the start or of a leaf, leaves the link-window without a
    certificate (fail-closed) and is listed in `refused`; a refused start is asked again at the next link-window, and
    refused again."""

    signer: Signer
    session: bytes
    refused: list[tuple[str, Direction, int]] = field(default_factory=list)
    started: bool = False

    def certify(self, record: Record, direction: Direction, params: GridParams, parameters: bytes, *,
                deadline: bool = True) -> CertifiedRecord | None:
        head = record_head(record.link, direction, record.window, record.grid, record.status, params, parameters,
                           deadline=deadline)
        digest = head.digest()
        key_id = self.signer.key_id
        message = certificate_message(key_id, self.session, record.link, direction, record.window, digest)
        try:
            if not self.started:
                self.signer.start(self.session)
                self.started = True
            signature = self.signer.sign(message)
        except SignerRefused:
            self.refused.append((record.link, direction, record.window))
            return None
        return CertifiedRecord(head, Certificate(key_id, record.link, direction, record.window, digest, signature))
