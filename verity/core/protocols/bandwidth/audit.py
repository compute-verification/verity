"""The bandwidth audit v0 over a PoUS store `E` (P2's `C*`), one window at a time.  Each device's store is set up under
a salt with its identity bound in (`draw_salt`, its GPU's 16-byte UUID, `device_id`, which the on-node challenger reads
from the driver, `read_device`), so devices holding one `W` hold different stores, and the verifier refuses a store
whose salt is not its device's (`Verifier`).

The verifier (`Verifier`, `Window`, `ChallengerState`) is the auditor's hardware, on a machine the prover doesn't
control; whatever runs on the GPU's node (`read_device`, the tagger's host) is the prover's, and every bound holds if it
is adversarial, except that the device id it reads is the prover's word (`PROTOCOL.md` §2).

1. The verifier samples a fresh nonce from its own randomness and records it in the store's challenger state, on disk
   (`Window.open`); then it derives the window's key from its own secret, bound to `vk`, the tag and the nonce, and
   sends the seed `σ` (`Window.start`, which starts its clock).
2. Before `T = deadline_ns` (plus `rtt_allowance_ns`) the prover computes `y_j = Tag(σ, j, E_j)` for every block and
   returns the root of the tags' tree (`tag_pass`, then `Window.commit`, which stops the clock).
3. Only then does the verifier draw `s = samples` indices, uniform with replacement, from the same key.  A late
   commitment closes the window rejected and draws none.
4. The prover opens each: the block with its audit path to `vk`'s root, and the tags of its leaf with their path to the
   committed root (`TagCommitment.open`).  The verifier checks both paths and recomputes the tag (`Window.verdict`).
   Openings are not timed: the commitment already binds every tag.

A window is used once: a second start or commitment is refused, a nonce is never sampled twice, and nothing is resampled.
One window over a store is open at a time, from its record to its commitment or until `T + RTT` after its seed without
one: the verifier refuses to open another before then, so no read serves two windows' ingress.  Both rules hold for every
verifier of a store, whatever its tag, and across restarts, because all of them share the store's one
`ChallengerState`, a file named by the store in the verifier's state directory that fails closed.  The verdict
publishes only whether the window was accepted (`Verdict.public`).  `Params` refuses `T + RTT` at or past
`regen_floor_ns`, the fastest cooperating P2 root (`2 · ROOT_CALL_NS`), so no block is rebuilt from `W` within a window
(SeqRoot).  What an accepted window certifies, and on what, is `PROTOCOL.md`; `bound` computes it, for a certified tag
only (`tags.Tag.certified`).
"""

from __future__ import annotations

import fcntl
import math
import os
import re
import secrets
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from verity.core.protocols.pous.codec import Geometry
from verity.core.protocols.pous.params import SALT_BITS
from verity.core.protocols.pous.schemes.p2 import ROOT_CALL_NS
from verity.core.protocols.pous.verifier import MerkleTree, VerifierKey, leaf_hash, verify_path
from verity.ml.randomness import Key, derive

from . import tags
from .tags import SEED_BYTES

WINDOW_DOMAIN = "verity/bandwidth/window/v2"
SALT_DOMAIN = "verity/bandwidth/salt/v2"
DEVICE_BYTES = 16
NONCE_BYTES = 32
STATE_FORMAT = "verity/bandwidth/challenger-state/v1"

REGEN_FLOOR_NS = 2 * ROOT_CALL_NS
"""The fastest 16,448-bit root on the allowed hardware, 8 cores of one chiplet cooperating, 1,708,058 ns
(`pous.schemes.p2.ROOT_CALL_NS` is half of it).  Rebuilding a block of `C*` from `W` takes two roots in sequence, so a
window shorter than one cannot rebuild any block.  The fastest root measured on the RTX PRO 6000 is slower, 18.53 ms
(floor 2.26 ms)."""

FAILURES = ("late", "malformed", "store-path", "tag-path", "tag-mismatch")


@dataclass(frozen=True, slots=True)
class Params:
    """A window's parameters: the tag construction, `s`, `T`, the round-trip allowance and the rebuild floor."""

    tag: str
    samples: int
    deadline_ns: int
    rtt_allowance_ns: int = 0
    regen_floor_ns: int = REGEN_FLOOR_NS

    def __post_init__(self) -> None:
        tags.get(self.tag)
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError("a window draws at least one sample")
        if self.deadline_ns <= 0 or self.rtt_allowance_ns < 0:
            raise ValueError("the deadline is positive and the round-trip allowance nonnegative")
        if self.limit_ns >= self.regen_floor_ns:
            raise ValueError(f"T + RTT = {self.limit_ns} ns reaches the rebuild floor of {self.regen_floor_ns} ns: "
                             "a prover could rebuild a block from W within the window")

    @property
    def limit_ns(self) -> int:
        """`T` plus the round-trip allowance: the longest a prover may take, which the bound charges it with."""
        return self.deadline_ns + self.rtt_allowance_ns


_GPU_UUID = re.compile(r"GPU-([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})")


def device_id(uuid: str) -> bytes:
    """A GPU's identity: the 16 bytes of the UUID its driver reports (`cudaDeviceProp.uuid`), from the text nvidia-smi
    prints, `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` in lowercase hex.  Every other spelling is refused, so one GPU has
    one identity and one salt."""
    m = _GPU_UUID.fullmatch(uuid) if type(uuid) is str else None
    if m is None:
        raise ValueError("a GPU's id is its UUID as nvidia-smi prints it, GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    return bytes.fromhex("".join(m.groups()))


MIG_OFF = ("Disabled", "[N/A]")
"""nvidia-smi's `mig.mode.current` for a GPU that is not split into MIG slices: off, or a GPU without MIG."""


def nvidia_smi(args: Sequence[str]) -> str:
    """nvidia-smi's output for `args`, from the driver of the node this runs on."""
    return subprocess.run(["nvidia-smi", *args], check=True, capture_output=True, text=True).stdout


def read_device(gpu: str, *, smi: Callable[[Sequence[str]], str] | None = None) -> bytes:
    """The id of GPU `gpu` as the on-node challenger reads it on the GPU's node, the prover's word (`PROTOCOL.md` §2):
    the UUID the driver reports, `GPU-…` as nvidia-smi prints it (`smi`, by default `nvidia_smi`), through `device_id`.
    `gpu` is what nvidia-smi's `-i` takes: an index, the GPU's UUID or its PCI bus id.  A MIG slice is refused, whether
    `gpu` names one or the GPU is in MIG mode, whose CUDA devices are its slices, each with a UUID of its own: a slice
    is not a device."""
    if type(gpu) is not str or not gpu or gpu.startswith("MIG-"):
        raise ValueError(f"{gpu!r} names no whole GPU")
    rows = (smi or nvidia_smi)(["-i", gpu, "--query-gpu=uuid,mig.mode.current", "--format=csv,noheader"]).splitlines()
    if len(rows) != 1:
        raise ValueError(f"nvidia-smi gives {len(rows)} GPUs for {gpu!r}, not one")
    fields = rows[0].split(", ")
    if len(fields) != 2:
        raise ValueError(f"nvidia-smi's row for {gpu!r} is not a UUID and a MIG mode: {rows[0]!r}")
    uuid, mig = fields
    if mig not in MIG_OFF:
        raise ValueError(f"{uuid} is in MIG mode ({mig}): its CUDA devices are slices, and a slice is not a device")
    return device_id(uuid)


def draw_salt(source: bytes, commitment: bytes, device: bytes) -> bytes:
    """The salt of `device`'s store, which PoUS's setup (`pous.setup`) takes in place of its own `draw_salt`: sampled after
    the commitment to `W` from the verifier's secret (or a later beacon round) as PoUS's is, with the device's 16-byte
    id bound in (`device_id`).  Devices holding one `W` then hold different stores, so their `vk`, seeds and tags
    differ."""
    if type(device) is not bytes or len(device) != DEVICE_BYTES:
        raise ValueError(f"a device's id is its GPU's {DEVICE_BYTES}-byte UUID (device_id)")
    return derive(source, SALT_DOMAIN, {"w": commitment, "device": device}).stream("salt", SALT_BITS // 8)


def window_key(source: bytes, vk: VerifierKey, tag: str, nonce: bytes) -> Key:
    """The key of a window of `tag` over `vk`'s store.  `source` is the verifier's own secret: from a public one the
    samples are known before the commitment, and a prover tags only the blocks they name.  `nonce` is sampled fresh for
    the window from the verifier's own randomness, never counted: a key that a window number fixes repeats in every
    verifier, tag or restart that counts from the same place, and a window whose samples are known is passed by tagging
    only them."""
    if type(nonce) is not bytes or len(nonce) != NONCE_BYTES:
        raise ValueError(f"a window's nonce is {NONCE_BYTES} bytes")
    return derive(source, WINDOW_DOMAIN, {"root": vk.root, "salt": vk.salt, "tag": tags.get(tag).name, "nonce": nonce})


def seed(key: Key) -> bytes:
    """`σ`, sent at the window's start."""
    return key.stream("seed", SEED_BYTES)


def samples(key: Key, blocks: int, s: int) -> tuple[int, ...]:
    """The `s` sampled indices, uniform on `0..blocks-1` with replacement, drawn only after the commitment."""
    return tuple(key.uniform(blocks, ("sample", t)) for t in range(s))


class Store:
    """The prover's `E`: the blocks of a PoUS store with their tree under `vk`'s root (`pous.verifier`)."""

    def __init__(self, params: Geometry, salt: bytes, blocks: Sequence[bytes]) -> None:
        if len(blocks) != params.blocks or any(len(b) != params.block_bytes for b in blocks):
            raise ValueError(f"E is {params.blocks} blocks of {params.block_bytes} bytes")
        self.params, self.salt, self.blocks = params, salt, tuple(bytes(b) for b in blocks)
        self.tree = MerkleTree([leaf_hash(params, salt, i, b) for i, b in enumerate(self.blocks)])

    @classmethod
    def from_segments(cls, params: Geometry, salt: bytes, segments: Iterable[Sequence[bytes]]) -> Store:
        return cls(params, salt, [b for segment in segments for b in segment])

    @property
    def root(self) -> bytes:
        return self.tree.root


@dataclass(frozen=True, slots=True)
class Opening:
    """Sample `index` opened: its block and audit path to `vk`'s root, and its leaf's tags with their path."""

    index: int
    block: bytes
    store_path: tuple[bytes, ...]
    leaf_tags: tuple[bytes, ...]
    tag_path: tuple[bytes, ...]


class TagCommitment:
    """The prover's tags for one window, and the tree whose `root` it commits."""

    def __init__(self, store: Store, tag: str, seed: bytes, ys: Sequence[bytes]) -> None:
        self.store, self.tag, self.seed, self.ys = store, tags.get(tag), seed, tuple(ys)
        if len(self.ys) != store.params.blocks:
            raise ValueError("one tag per block")
        self.tree = MerkleTree(tags.leaves(self.tag, seed, self.ys))

    @property
    def root(self) -> bytes:
        return self.tree.root

    def open(self, j: int) -> Opening:
        per = tags.tags_per_leaf(self.tag)
        c = j // per
        return Opening(j, self.store.blocks[j], self.store.tree.path(j), self.ys[c * per:(c + 1) * per],
                       self.tree.path(c))


def tag_pass(store: Store, tag: str, seed: bytes) -> TagCommitment:
    """The honest prover's pass: every block's tag, committed.  The GPU tagger (`benchmarks/bandwidth`) computes the
    same root."""
    t = tags.get(tag)
    t.check_block(store.params.block_bytes)
    keyed = t.keyed(seed)
    return TagCommitment(store, tag, seed, [keyed(j, b) for j, b in enumerate(store.blocks)])


@dataclass(frozen=True, slots=True)
class Verdict:
    """A window's outcome.  `failure` (one of `FAILURES`), the failing sample's `index` and the commitment's
    `elapsed_ns` stay with the verifier; only `public()` leaves it."""

    accepted: bool
    elapsed_ns: int | None = None
    failure: str | None = None
    index: int | None = None

    def public(self) -> dict[str, bool]:
        return {"accepted": self.accepted}


class StateError(RuntimeError):
    """A refusal to open a window: the store's challenger state is missing, unreadable, another store's, not the
    store's one state, held by another challenger, changed under this one, or closed."""


_NONCE = re.compile(rb"[0-9a-f]{64}")


def _head(root: bytes, salt: bytes) -> bytes:
    return f"{STATE_FORMAT} {root.hex()} {salt.hex()}".encode()


def state_name(vk: VerifierKey) -> str:
    """The name of `vk`'s store's challenger state in a challenger's state directory: the store's root and salt, so a
    directory has one place for each store's state."""
    return f"{vk.root.hex()}-{vk.salt.hex()}.state"


def _survey(directory: Path) -> None:
    """Refuse a state directory that holds anything but states, each named by the store its first line names
    (`state_name`): a copy or link of a state under another name, a link in place of one, a subdirectory or a stray
    file.  The directory then holds at most one state per store, and its listing shows which stores have one."""
    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except OSError as e:
        raise StateError(f"{directory}: unreadable as a state directory ({type(e).__name__})") from None
    for e in entries:
        try:
            if not e.is_file(follow_symlinks=False):
                raise StateError(f"{e.path}: a state directory holds only states")
            with open(e.path, "rb") as f:
                head = f.readline(len(STATE_FORMAT) + 512)
        except OSError as x:
            raise StateError(f"{e.path}: unreadable ({type(x).__name__})") from None
        fields = head.removesuffix(b"\n").split(b" ")
        if (len(fields) != 3 or fields[0] != STATE_FORMAT.encode()
                or e.name.encode() != fields[1] + b"-" + fields[2] + b".state"):
            raise StateError(f"{e.path} is not a state named by its store: a store has one challenger state, the file "
                             "its root and salt name")


class ChallengerState:
    """The challenger's state for one store, shared by every verifier of the store (every tag's) and kept across
    restarts, in a file that fails closed: in the verifier's state directory, passed explicitly, on the verifier's
    machine (the auditor's, never the GPU's node), never a signer's key directory.  The file is named by the store
    (`state_name`: its root and salt), and the directory holds nothing but such files (`_survey`), so a store has one
    state: `create` refuses a second, and both `create` and opening refuse a directory that holds a copy or link of a
    state under another name, or any other file.  The file is `STATE_FORMAT`'s line with the store's root and salt, then
    one record a line: `open NONCE TAG` when a window takes its nonce, appended and fsynced before its `σ` leaves, and
    `close NONCE` when that window closes (its commitment; `T + RTT` without one, recorded when the next window opens;
    or its verifier stopping first).

    Opening takes the file's lock for the object's life, so one verifier holds a store's state at a time, and refuses a
    file that is missing, a link, unreadable (a first line that is not the format's, a record that is neither, a nonce
    opened twice, a close that is not the open window's, a last line cut short) or another store's.  A window the file
    shows open was its last verifier's, which no longer runs, so opening closes it: no live verifier can accept it.
    Each window's nonce comes from `nonces`, the verifier's own randomness (`secrets.token_bytes`), and a nonce the
    file holds is refused, so no two windows over the store share a key.  Before each record the file must be as this
    object left it.  Nothing here rewrites or truncates the file.  A state restored from a snapshot, or copied to another
    directory or host, is past what the directory can see; that the verifier keeps one state directory and never rolls
    back or forks a state rests on its host (`trusted-host`), and neither repeats a key while the nonces are
    fresh (PROTOCOL.md §5)."""

    def __init__(self, directory: str | os.PathLike, vk: VerifierKey, *,
                 nonces: Callable[[int], bytes] = secrets.token_bytes) -> None:
        self.directory, self.store, self._nonces = Path(directory), (vk.root, vk.salt), nonces
        self.path = self.directory / state_name(vk)
        self._open: Window | None = None
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise StateError(f"{self.path}: missing") from None
        except OSError as e:
            raise StateError(f"{self.path}: a link or unwritable ({type(e).__name__})") from None
        self._file = os.fdopen(fd, "ab", buffering=0)
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file.close()
            raise StateError(f"{self.path} is held by another challenger") from None
        try:
            self._used, dangling = self._read()
            _survey(self.directory)
            if dangling is not None:
                self._append(f"close {dangling.hex()}\n")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def create(directory: str | os.PathLike, vk: VerifierKey) -> None:
        """A new store's state in the challenger's state directory, with no window; refused when the store has one, and
        when the directory holds anything but states named by their stores: a store's state is never made again."""
        d = Path(directory)
        _survey(d)
        p = d / state_name(vk)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            raise StateError(f"{p} exists: a store's challenger state is never made again") from None
        try:
            os.write(fd, _head(vk.root, vk.salt) + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        d = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(d)
        finally:
            os.close(d)

    def _read(self) -> tuple[set[bytes], bytes | None]:
        """The nonces used and the nonce of a window open at the end of the file, from the locked file itself."""
        fd = self._file.fileno()
        try:
            self._stat = os.fstat(fd)
            data = os.pread(fd, self._stat.st_size, 0)
        except OSError as e:
            raise StateError(f"{self.path}: unreadable ({type(e).__name__})") from None
        bad = StateError(f"{self.path}: unreadable (not a {STATE_FORMAT} state of whole records)")
        if len(data) != self._stat.st_size or not data.endswith(b"\n"):
            raise bad
        head, *records = data[:-1].split(b"\n")
        fields = head.split(b" ")
        if len(fields) != 3 or fields[0] != STATE_FORMAT.encode():
            raise bad
        if head != _head(*self.store):
            raise StateError(f"{self.path} is another store's")
        used: set[bytes] = set()
        open_: bytes | None = None
        for r in records:
            f = r.split(b" ")
            if len(f) == 3 and f[0] == b"open" and _NONCE.fullmatch(f[1]) and f[2].decode("ascii", "replace") in tags.TAGS:
                n = bytes.fromhex(f[1].decode())
                if open_ is not None or n in used:
                    raise bad
                used.add(n)
                open_ = n
            elif len(f) == 2 and f[0] == b"close" and open_ is not None and f[1] == open_.hex().encode():
                open_ = None
            else:
                raise bad
        return used, open_

    def _check(self) -> None:
        """The file is the one this object locked, as it left it."""
        if self._file.closed:
            raise StateError(f"{self.path}: closed")
        try:
            now, at = os.fstat(self._file.fileno()), os.stat(self.path)
        except OSError as e:
            raise StateError(f"{self.path}: unreadable ({type(e).__name__})") from None
        if now.st_size != self._stat.st_size or (now.st_dev, now.st_ino) != (at.st_dev, at.st_ino):
            raise StateError(f"{self.path}: changed under this challenger")

    def _append(self, record: str) -> None:
        line = record.encode()
        try:
            written = self._file.write(line)
            os.fsync(self._file.fileno())
            self._stat = os.fstat(self._file.fileno())
        except OSError as e:
            raise StateError(f"{self.path}: unwritable ({type(e).__name__})") from None
        if written != len(line):
            raise StateError(f"{self.path}: a short write")

    def _close(self, w: Window) -> None:
        self._open = None
        self._append(f"close {w._nonce.hex()}\n")

    def _claim(self, w: Window) -> bytes:
        """A fresh nonce for `w`, recorded open and fsynced before this returns; refused while another window over the
        store is open."""
        o = self._open
        if o is not None and o._live():
            raise RuntimeError("another window over this store is open: it closes at its commitment or at T + RTT "
                               "after its seed")
        self._check()
        if o is not None:
            self._close(o)
        nonce = self._nonces(NONCE_BYTES)
        if type(nonce) is not bytes or len(nonce) != NONCE_BYTES:
            raise StateError(f"the challenger's randomness gave no {NONCE_BYTES}-byte nonce")
        if nonce in self._used:
            raise StateError(f"nonce {nonce.hex()} is in {self.path}: a window's nonce is never drawn twice")
        self._append(f"open {nonce.hex()} {w._v.params.tag}\n")
        self._used.add(nonce)
        self._open = w
        return nonce

    def _hold(self, w: Window) -> None:
        """`w`, open, still holds the store."""
        if self._file.closed or self._open is not w:
            raise StateError(f"{self.path}: closed while the window was open")

    def _release(self, w: Window) -> None:
        """`w` has its commitment: its record closes, unless it closed already (past `T + RTT`)."""
        if self._file.closed:
            raise StateError(f"{self.path}: closed while the window was open")
        if self._open is w:
            self._check()
            self._close(w)

    def close(self) -> None:
        """Release the file; a window still open closes with it, and its commitment is refused."""
        if self._file.closed:
            return
        try:
            if self._open is not None:
                self._close(self._open)
        finally:
            fcntl.flock(self._file, fcntl.LOCK_UN)
            self._file.close()

    def __enter__(self) -> ChallengerState:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Verifier:
    """The verifier of one tag's windows over one store `vk`, from its own secret `source`.  The store must be the one
    `device` holds: its salt is `draw_salt(salt_source, commitment, device)`, or the verifier refuses it, so a store set
    up under any other salt (PoUS's own, another device's) is never audited as this device's.  Every verifier of a store
    takes the store's one `state`, which samples each window's nonce, records it before `σ` leaves, and keeps one window
    over the store open at a time, whichever verifier opens it."""

    def __init__(self, vk: VerifierKey, params: Params, source: bytes, state: ChallengerState, *, device: bytes,
                 commitment: bytes, salt_source: bytes, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        tags.get(params.tag).check_block(vk.params.block_bytes)
        if vk.salt != draw_salt(salt_source, commitment, device):
            raise ValueError("the store's salt is not draw_salt's for this device and commitment: refused")
        if not isinstance(state, ChallengerState) or state.store != (vk.root, vk.salt):
            raise ValueError("a verifier takes its store's challenger state")
        self.vk, self.params, self.source, self.state, self.clock = vk, params, source, state, clock

    def window(self) -> Window:
        return Window(self)


class Window:
    """One window: `open()` records it, `start()` sends `σ` (opening it first if it is not open), `commit(root)` takes
    the prover's root and returns the samples, and `verdict(openings)` checks them, each once and in that order."""

    def __init__(self, verifier: Verifier) -> None:
        self._v = verifier
        self._nonce: bytes | None = None
        self._key: Key | None = None
        self._started: int | None = None
        self._root: bytes | None = None
        self._elapsed: int | None = None
        self._samples: tuple[int, ...] | None = None
        self._verdict: Verdict | None = None

    @property
    def params(self) -> Params:
        """The window's tag, samples and deadline, which the prover is told."""
        return self._v.params

    def _live(self) -> bool:
        """Whether the window holds its store: open, without a commitment, and unstarted or within `T + RTT` of its
        start on its clock."""
        return (self._nonce is not None and self._elapsed is None
                and (self._started is None or self._v.clock() - self._started <= self._v.params.limit_ns))

    def open(self) -> None:
        """Take the window's nonce and record it in the store's state, fsynced, before its `σ` exists: from here the
        window holds the store until its commitment, or until `T + RTT` after `start`.  The prover prepares between
        `open` and `start`, so the record's write is not in the gap it prepares for.  Refused, and the window left
        closed, while another over the store is open."""
        if self._nonce is not None:
            raise RuntimeError("the window is open")
        self._nonce = self._v.state._claim(self)

    def start(self) -> bytes:
        """`σ`, and the clock starts; the window opens first if it is not open."""
        if self._key is not None:
            raise RuntimeError("the window has started")
        if self._nonce is None:
            self.open()
        else:
            self._v.state._hold(self)
        self._key = window_key(self._v.source, self._v.vk, self._v.params.tag, self._nonce)
        s = seed(self._key)
        self._started = self._v.clock()
        return s

    def commit(self, root: bytes) -> tuple[int, ...]:
        """Stop the clock on the prover's root; the samples if it is on time, else none and the window is rejected."""
        stop = self._v.clock()
        if self._started is None:
            raise RuntimeError("the window has not started")
        if self._elapsed is not None:
            raise RuntimeError("the window has a commitment")
        self._v.state._release(self)
        self._elapsed = stop - self._started
        if type(root) is not bytes or len(root) != tags.DIGEST_BYTES:
            self._verdict = Verdict(False, self._elapsed, "malformed")
            return ()
        if self._elapsed > self._v.params.limit_ns:
            self._verdict = Verdict(False, self._elapsed, "late")
            return ()
        self._root = root
        self._samples = samples(self._key, self._v.vk.params.blocks, self._v.params.samples)
        return self._samples

    def verdict(self, openings: Sequence[Opening] = ()) -> Verdict:
        if self._elapsed is None:
            raise RuntimeError("the window has no commitment")
        if self._verdict is None:
            self._verdict = self._check(openings)
        return self._verdict

    def _check(self, openings: Sequence[Opening]) -> Verdict:
        vk, params, elapsed = self._v.vk, self._v.params, self._elapsed
        tag = tags.get(params.tag)
        if len(openings) != len(self._samples):
            return Verdict(False, elapsed, "malformed")
        keyed, head = tag.keyed(seed(self._key)), tags.leaf_head(tag, seed(self._key))
        per, blocks = tags.tags_per_leaf(tag), vk.params.blocks
        count = tags.leaf_count(tag, blocks)
        for i, o in zip(self._samples, openings):
            c = i // per
            size = min(per, blocks - c * per)
            if (not isinstance(o, Opening) or o.index != i or type(o.block) is not bytes
                    or len(o.leaf_tags) != size
                    or any(type(y) is not bytes or len(y) != tag.tag_bytes for y in o.leaf_tags)):
                return Verdict(False, elapsed, "malformed", i)
            if not vk.check_path(i, o.block, o.store_path):
                return Verdict(False, elapsed, "store-path", i)
            if not verify_path(self._root, count, c, tags.leaf_hash(head, c, o.leaf_tags), o.tag_path):
                return Verdict(False, elapsed, "tag-path", i)
            if keyed(i, o.block) != o.leaf_tags[i - c * per]:
                return Verdict(False, elapsed, "tag-mismatch", i)
        return Verdict(True, elapsed)


EPSILON = Fraction(1, 64)
SAMPLES = 1849
QUERIES_LOG2 = 64
SLACK_LOG2 = 41
"""The proof's deployed point: `ε_s = 1/64`, `s = 1,849`, at most `2^64` oracle queries in a window, and
`λ = 64·k + 41`, which together keep the error under `2^-40` at `N = 2^19`.  `2^64` is past what any fleet computes in
a window (an RTX PRO 6000 makes about `2^26` BLAKE3 compressions in 1.7 ms), so the query bound needs no assumption."""


def certified_blocks(blocks: int, epsilon: Fraction = EPSILON) -> int:
    """`k = ⌈(1 − ε_s)·N⌉`."""
    return math.ceil((1 - Fraction(epsilon)) * blocks)


def samples_for(blocks: int, error_log2: int, epsilon: Fraction = EPSILON) -> int:
    """The fewest `s` with `((k − 1)/N)^s ≤ 2^error_log2`, the sampling term of the error."""
    k = certified_blocks(blocks, epsilon)
    return math.ceil(error_log2 * math.log(2) / math.log((k - 1) / blocks))


@dataclass(frozen=True, slots=True)
class Bound:
    """`Bandwidth.Guarantees.AuditRO` for one device and window: an accepted window read at least `ingress_bytes` of
    device memory, except with probability at most `error`, the sum of its three terms."""

    blocks: int
    certified_blocks: int
    block_bytes: int
    chip_bytes: int
    side_bytes: int
    slack_bits: int
    ingress_bytes: int
    error_samples: float
    error_guess: float
    error_compression_log2: int

    @property
    def error(self) -> float:
        return self.error_samples + self.error_guess + 2.0 ** self.error_compression_log2


def bound(params: Params, blocks: int, block_bytes: int, *, chip_bytes: int, side_bytes_per_s: float,
          epsilon: Fraction = EPSILON, queries_log2: int = QUERIES_LOG2, slack_log2: int = SLACK_LOG2) -> Bound:
    """`ingress ≥ k·b − S − c − G·b − λ` bits with `k = ⌈(1 − ε_s)·N⌉`, except with probability
    `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ` (`Bandwidth.Guarantees.AuditRO`, PROTOCOL.md §1).  `S` is the chip's storage,
    `c = cap_side·(T + RTT)`, `G = 0` because `Params` refuses a window as long as a rebuild, `q ≤ 2^queries_log2` and
    `λ = queries_log2·k + slack_log2`, so `C(q, k)/2^λ ≤ q^k/2^λ = 2^-slack_log2`.  Only a certified tag has a bound."""
    tag = tags.get(params.tag)
    if not tag.certified:
        raise ValueError(f"{tag.name} is an uncertified fast check: nothing bounds a device's ingress through it")
    k = certified_blocks(blocks, epsilon)
    lam = queries_log2 * k + slack_log2
    side = math.ceil(side_bytes_per_s * params.limit_ns / 1e9)
    ingress_bits = k * 8 * block_bytes - 8 * chip_bytes - 8 * side - lam
    error_samples = math.exp(params.samples * math.log((k - 1) / blocks)) if k > 1 else 0.0
    return Bound(blocks, k, block_bytes, chip_bytes, side, lam, max(0, ingress_bits // 8), error_samples,
                 float(params.samples * tag.guess(block_bytes)), queries_log2 * k - lam)
