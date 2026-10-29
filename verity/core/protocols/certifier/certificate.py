"""Network certificates: the network certifier's signed record of each link-window, on SHA-512 (record v2).

The network certifier is the auditor-trusted device on a node's uplink that shapes each link into the grid
(`grid.Certifier`). Once per link-window whose record is complete it signs one **network certificate**: a signature
over a hiding digest of the window's record, salted by the certifier. A link-window whose record isn't complete gets
none (`Issuer`), so it fails closed. The record head and the salt are witness: they enter proofs and never reach
the auditor in the clear, so the record's own hashes are unsalted, and what the certificate signs is the salted digest.

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

**The signed digest** hides the record digest R under `salt_c`, 64 bytes the certifier draws from its own randomness,
fresh for each record (`Issuer`'s `coins`), never the prover's:

    signed digest = SHA-512( "verity/net-cert/v1" ‖ salt_c ‖ R )     (`certificate_digest`)

The certifier hands `salt_c` to the prover with the head, as witness only (`CertifiedRecord`). N2's head check
(`head_check`) links the two: R is the digest of the head it opened, and the salted digest of R is the one the
certificate signs. So the signature check reads only the certificate's public fields and its signed digest
(`Certificate.verify`), never a hash of witness. Until N2's unit is proved, its verifier of record checks that
link on the host too, over the opening (`CertifiedRecord.verify`).

**The certificate** signs a 64-byte message that binds the signed digest to the key, the certifier session and the
link-window, so a certificate can't be replayed onto another window, link, direction, key or session. It is LMS's
network-certificate role digest (`verity.ml.crypto.lms.certificate_message`), each field but I and the window prefixed
with its length as a u64be:

    message = SHA-512( [tag] ‖ I ‖ [ [session] ‖ [link] ‖ [direction] ‖ u64be window ] ‖ [signed digest] ),
              tag = "verity/network-certificate-message/v2", direction "egress" or "ingress", [x] = u64be len(x) ‖ x

`session` is the certifier session's id, 32 bytes (`lms.SESSION_BYTES`) that the challenger draws for the session and
issues to the certifier; the audit takes it from its parameters, as it takes the key, so the record head and the
certificate's encoding don't carry it. The tag is the message's own, distinct from the ids of the checks that read it
(`verity/network-certificate/v4`, `verity/network-records/v3`).

**The clock certificate** (`CertifiedClock`, `Issuer.salted_clock`): beside each egress record the certifier signs the
sync its shaper ran the window under (`schedule.ClockSync`, δ and ρ = p/q in lowest terms), hidden as a record digest
is, under the same message, so it names the same key, session and link-window:

    clock digest  = tagged_sha512("verity-network-certifier/clock/v1", u64be δ ‖ u64be p ‖ u64be q)   (`clock_digest`)
    signed digest = SHA-512( "verity/net-cert/v1" ‖ salt_c ‖ clock digest )                         (`clock_commitment`)

`salt_c` is 64 fresh bytes of the certifier's own coins, as a record's, and never one an earlier clock certificate of
the issuer has: under one salt two windows of one sync would sign one digest, which shows that their syncs are equal.
What the certifier signs is witness: the sync and its salt never reach the auditor in the clear, and only the
certificate, a signature over a hiding commitment, is public. N2's advised verifier reads each window's sync only
through its opening against the salted digest its certificate signs (`CertifiedClock.refusal`,
`advised.SignedClocks`). A record certificate signs a record digest salted and a clock certificate a clock digest,
each under its own tag, so one opens as the other only on a SHA-512 collision. The Lean device's issuer
(`CertifierDevice.Issuer.clock`) signs the same, and its difftest holds it to `Issuer.salted_clock`.

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
import secrets
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha512
from verity.ml.crypto import lms
from verity.core.protocols.certifier.grid import Grid, GridParams, Reason, Record, Status, row_left_packed
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet

RECORD_TAG = "verity-network-certifier/record/v2"
PARAMETERS_TAG = "verity-network-certifier/parameters/v1"
CLOCK_TAG = "verity-network-certifier/clock/v1"
FRAME_TAG = b"verity-network-certifier/frame/v2\0"
EMPTY_TAG = b"verity-network-certifier/empty/v2\0"
NODE_TAG = b"verity-network-certifier/node/v2\0"
CERTIFICATE_TAG = b"verity/net-cert/v1"
DIGEST_BYTES = 64
SALT_BYTES = 64

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
    to read the counts and the status. Like the salt the certifier hands with it, it is witness."""

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


def certificate_digest(salt: bytes, record: bytes) -> bytes:
    """The digest a certificate signs: SHA-512("verity/net-cert/v1" ‖ salt_c ‖ R) of the record digest R under the
    certifier's salt."""
    return hashlib.sha512(CERTIFICATE_TAG + salt + record).digest()


def certificate_message(key_id: bytes, session: bytes, link: str, direction: Direction, window: int,
                        digest: bytes) -> bytes:
    if len(digest) != DIGEST_BYTES:
        raise ValueError("a signed digest is 64 bytes")
    return lms.certificate_message(key_id, session, link, direction.value, window, digest)


def clock_digest(sync: ClockSync) -> bytes:
    """The digest a clock certificate signs: tagged_sha512("verity-network-certifier/clock/v1", u64be δ ‖ u64be p ‖
    u64be q) of the sync (δ, ρ = p/q in lowest terms)."""
    if type(sync) is not ClockSync:
        raise ValueError(f"a clock digest is of a schedule.ClockSync, not a {type(sync).__name__}")
    return tagged_sha512(CLOCK_TAG, _u(sync.delta, 8, "δ") + _u(sync.rho.numerator, 8, "p")
                         + _u(sync.rho.denominator, 8, "q"))


def clock_commitment(salt: bytes, sync: ClockSync) -> bytes:
    """The digest a salted clock certificate signs: the sync's clock digest hidden under the certifier's salt as a
    record digest is, SHA-512("verity/net-cert/v1" ‖ salt_c ‖ clock digest) (`certificate_digest`)."""
    if type(salt) is not bytes or len(salt) != SALT_BYTES:
        raise ValueError(f"a clock certificate's salt is {SALT_BYTES} bytes")
    return certificate_digest(salt, clock_digest(sync))


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
    signer on the same state is refused. ``floor`` is the next leaf known unused, such as 1 + the last q published. One
    signer may be shared across threads (an `Issuer` on the proxy's loop, its reveal endpoint off it): it signs one
    message at a time."""

    def __init__(self, key: lms.PrivateKey, state: str | Path, *, floor: int = 0) -> None:
        self.key, self.state, self.floor = key, Path(state), floor
        self._leaves: lms.LeafState | None = None
        self._lock = threading.Lock()  # the leaf state reads, then advances, the next leaf: never two at once

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
        with self._lock:
            try:
                self._state().start(session)
            except lms.StateError as e:
                raise SignerRefused(str(e)) from None

    def sign(self, message: bytes) -> bytes:
        with self._lock:
            try:
                return lms.sign(self.key, self._state(), message).encode()
            except lms.StateError as e:
                raise SignerRefused(str(e)) from None

    def close(self) -> None:
        with self._lock:
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

    def verify(self, key: PublicKey, session: bytes) -> bool:
        """The host's signature check: the certificate is `key`'s and its signature verifies over its message in
        certifier session `session`. It reads the certificate's public fields and its signed digest, nothing of the
        head."""
        if self.key_id != key.key_id:
            return False
        try:
            message = self.message(session)
        except ValueError:
            return False
        return key.verify(message, self.signature)


@dataclass(frozen=True, slots=True)
class CertifiedRecord:
    """What the certifier hands over per link-window: the record head, the salt it drew for it (`salt_c`, witness like
    the head) and the certificate, which signs the head's digest salted by it (`certificate_digest`)."""

    head: RecordHead
    salt: bytes
    certificate: Certificate

    def __post_init__(self) -> None:
        if len(self.salt) != SALT_BYTES:
            raise ValueError(f"a certificate salt is {SALT_BYTES} bytes, not {len(self.salt)}")

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
        """The certificate names the head's link-window, signs the head's digest under the record's salt, and verifies
        under `key` in certifier session `session` (`Certificate.verify`)."""
        c, h = self.certificate, self.head
        if (c.link, c.direction, c.window) != (h.link, h.direction, h.window):
            return False
        try:
            record = h.digest()
        except ValueError:
            return False
        return c.digest == certificate_digest(self.salt, record) and c.verify(key, session)


@dataclass(frozen=True, slots=True)
class CertifiedClock:
    """What the certifier hands over per egress link-window beside its record: the sync its shaper ran the window under
    and the salt it drew for it (`salt_c`), both witness as a head and its salt are, and the certificate, the one public
    part, which signs the sync's clock digest salted by it (`clock_commitment`)."""

    sync: ClockSync
    salt: bytes
    certificate: Certificate

    def __post_init__(self) -> None:
        if type(self.salt) is not bytes or len(self.salt) != SALT_BYTES:
            raise ValueError(f"a clock certificate's salt is {SALT_BYTES} bytes")

    @property
    def link(self) -> str:
        return self.certificate.link

    @property
    def window(self) -> int:
        return self.certificate.window

    def refusal(self, key: PublicKey, session: bytes) -> str | None:
        """Why the opening fails under `key` in certifier session `session`, or None when it holds: the certificate is
        an egress one of `key`'s, the sync and the salt open the digest it signs, and its signature verifies over its
        message in the session (`Certificate.verify`), which binds the link and the window."""
        c = self.certificate
        if type(c) is not Certificate or c.direction is not Direction.EGRESS:
            return "it isn't an egress certificate, and a clock is an egress window's"
        if c.key_id != key.key_id:
            return f"another key's, not {key.key_id.hex()[:16]}…"
        try:
            opened = clock_commitment(self.salt, self.sync)
        except ValueError:
            opened = None
        if c.digest != opened:
            return "its sync and salt don't open the salted digest it signs"
        if not c.verify(key, session):
            return "its signature doesn't verify under the key in its session for the link-window it names"
        return None

    def verify(self, key: PublicKey, session: bytes) -> bool:
        return self.refusal(key, session) is None


@dataclass
class Issuer:
    """The network certifier's signing step, beside its grid (`grid.Certifier`): one certificate per closed link-window,
    in certifier session `session`. Before its first signature it starts the session on its signer, which refuses a
    session its state has already started. A refusal, of the start or of a leaf, leaves the link-window without a
    certificate (fail-closed) and is listed in `refused`; a refused start is asked again at the next link-window, and
    refused again. **It certifies only a complete record**: a record whose status isn't complete is refused the same
    way, before the session's start or a leaf, so a link-window that isn't covered has no certificate and fails closed
    (`no-verdict`), and a certificate that verifies under the key is of a complete record unless it is a forgery. Each
    record's salt and each clock's is `SALT_BYTES` fresh bytes of `coins`, the certifier's own
    randomness (the operating system's by default; a test injects its own), and `clock_salts` are those its signed
    clock certificates have taken, none of which it takes again."""

    signer: Signer
    session: bytes
    refused: list[tuple[str, Direction, int]] = field(default_factory=list)
    started: bool = False
    coins: Callable[[int], bytes] = field(default=secrets.token_bytes, repr=False)
    clock_salts: set[bytes] = field(default_factory=set, repr=False)

    def certify(self, record: Record, direction: Direction, params: GridParams, parameters: bytes, *,
                deadline: bool = True) -> CertifiedRecord | None:
        salt = self.coins(SALT_BYTES)
        if len(salt) != SALT_BYTES:
            raise ValueError(f"the certifier's coins gave {len(salt)} bytes for a {SALT_BYTES}-byte salt")
        head = record_head(record.link, direction, record.window, record.grid, record.status, params, parameters,
                           deadline=deadline)
        digest = certificate_digest(salt, head.digest())
        if not record.status.complete:
            # formed as `sign` forms it, so what `sign` raises on (a session id not 32 bytes) raises here too
            certificate_message(self.signer.key_id, self.session, record.link, direction, record.window, digest)
            self.refused.append((record.link, direction, record.window))
            return None
        certificate = self.sign(record.link, direction, record.window, digest)
        return None if certificate is None else CertifiedRecord(head, salt, certificate)

    def salted_clock(self, link: str, window: int, sync: ClockSync) -> CertifiedClock | None:
        """The clock certificate of the egress link-window, `sync` being the one the certifier's shaper ran it under:
        the sync's clock digest salted by `SALT_BYTES` fresh bytes of `coins` (`clock_commitment`), or None when the
        signer refuses. Raises `ValueError` on coins of another length and on a salt they gave an earlier signed clock
        certificate of this issuer, under which two windows of one sync would sign one digest. A salt is taken once
        signed under: a refused one signed nothing, so nothing public shares it."""
        salt = self.coins(SALT_BYTES)
        if len(salt) != SALT_BYTES:
            raise ValueError(f"the certifier's coins gave {len(salt)} bytes for a {SALT_BYTES}-byte salt")
        if salt in self.clock_salts:
            raise ValueError("the certifier's coins gave an earlier clock certificate's salt again, under which two "
                             "windows of one sync would sign one digest, so it signs no clock under it")
        certificate = self.sign(link, Direction.EGRESS, window, clock_commitment(salt, sync))
        if certificate is None:
            return None
        self.clock_salts.add(salt)
        return CertifiedClock(sync, salt, certificate)

    def sign(self, link: str, direction: Direction, window: int, digest: bytes) -> Certificate | None:
        """A certificate of the 64-byte `digest` for the link-window in this session, or None when the signer refuses
        (the link-window is then listed in `refused`). `certify` signs a record's salted digest with it,
        `salted_clock` a window's salted sync, and the warden's cell rows their head's digest (`warden_rows`)."""
        key_id = self.signer.key_id
        message = certificate_message(key_id, self.session, link, direction, window, digest)
        try:
            if not self.started:
                self.signer.start(self.session)
                self.started = True
            signature = self.signer.sign(message)
        except SignerRefused:
            self.refused.append((link, direction, window))
            return None
        return Certificate(key_id, link, direction, window, digest, signature)
