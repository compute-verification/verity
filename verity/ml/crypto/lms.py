"""A stateful hash-based signature: LMS with LM-OTS (RFC 8554) over SHA-512, in plain Python, the scheme of the network
certifier's certificates and the memory challenger's records, and the reference of their in-proof verifier
(`verity.ml.commitments.gates.lms`). Its security is the claim ``euf-cma/lms-sha512-n64`` (`verity.core.claims`).

**Use.** One key per device per role. ``keygen(DEPLOYED, seed)`` (seed: N = 64 secret bytes the device keeps), then once
``LeafState.create(path, public_key(key), DEPLOYED.H)``; to sign, ``sign(key, state, message)`` with ``state = LeafState(path,
public_key(key), DEPLOYED.H)``; to check, ``verify(DEPLOYED, public, message, sig)``. The message is a role's digest
(`certificate_message`, `memory_challenger_message`), never hand-rolled. A network certifier first starts its session on
the state (``state.start(session)``), which refuses a session id the key has already started.

**The hash.** ``H(x) = SHA-512(x)``, the whole output: N = 64, nothing truncated. The one other N, `TOY_N` = 8 (the
first word of SHA-512's output, asked for by name with ``toy=True``), is circuit-check's binding of the verifier and is never
deployed. Every hash input is RFC 8554's, with its domain separators: a chain step ``H(I ||
u32(q) || u16(i) || u8(j) || tmp)``, the message ``Q = H(I || u32(q) || u16(D_MESG) || C || message)``, the one-time public
key ``K = H(I || u32(q) || u16(D_PBLC) || z_0 || ... || z_{P-1})``, a leaf ``H(I || u32(r) || u16(D_LEAF) || K)`` and an
interior node ``H(I || u32(r) || u16(D_INTR) || left || right)``, node r of the tree numbered from the root r = 1 (a leaf q
is node 2^H + q). A message is any bytes; a role's digest is 64, the length the in-proof verifier takes.

**Parameters.** (N, W, H): N the hash length, W in {1, 2, 4, 8} the Winternitz width, H the tree height (2^H one-time keys).
Deployed: ``DEPLOYED = Params(64, 1, 20)``, W = 1 the fewest compressions to verify. P = U + V chains, U = ceil(8N / W) message digits and V checksum digits, and the
checksum shifted left LS = 16 - V W bits (RFC 8554 §4.1, Appendix B).

**Differences from RFC 8554.** No type fields: a verifier holds one parameter set, so a signature is ``u32(q) || C || y_0 ||
... || y_{P-1} || path_0 || ... || path_{H-1}`` and a public key ``I || T[1]``. Keys are deterministic from a seed (the RFC's
Appendix A): ``x_q[i] = H(I || u32(q) || u16(i) || u8(0xff) || SEED)``, ``C = H(I || u32(q) || u16(0xfffd) || u8(0xff) ||
SEED)`` and ``I = SHA-512(b"verity/lms/v0/I" || SEED)[:16]``. With the hash swapped for SHA-256 (`Params.hash`), the tests
reproduce RFC 8554's Test Case 2.

**Cost.** Keygen hashes about 2^H (P 2^W + 70) times: about 1.37G compressions at the deployed parameters, about 16
minutes on one core, divided by ``workers``; a signature at most P (2^W - 1) + P + 1 hashes; a verification 827 compressions
at the deployed parameters (P (2^W - 1) chain steps, the one-time key's ceil((P N + 39) / 128) blocks, two for the message,
one for the leaf and two per interior node). A signature is 34,756 bytes, a public key 80. Standard library only.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["D_INTR", "D_LEAF", "D_MESG", "D_PBLC", "DEPLOYED", "DIRECTIONS", "LeafState", "Params", "PrivateKey", "ROLES",
           "SESSION_BYTES", "Signature", "StateError", "certificate_message", "digits", "key", "keygen", "memory_challenger_message",
           "public_key", "root_of", "sign", "sign_at", "verify", "TOY_N"]

D_PBLC, D_MESG, D_LEAF, D_INTR = 0x8080, 0x8181, 0x8282, 0x8383


def _u32(x: int) -> bytes:
    return x.to_bytes(4, "big")


def _u16(x: int) -> bytes:
    return x.to_bytes(2, "big")


#: The one N besides 64 that `Params` admits, and only with ``toy=True``: one SHA-512 word, the smallest value of the same byte
#: and word layout, for circuit-check's binding of the in-proof verifier. Never deployed, never a key's: 64 bits of hash is no
#: signature.
TOY_N = 8


@dataclass(frozen=True)
class Params:
    N: int
    W: int
    H: int
    toy: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        sizes = (TOY_N, 64) if self.toy else (64,)
        if self.N not in sizes or self.W not in (1, 2, 4, 8) or isinstance(self.H, bool) or not 1 <= self.H <= 30:
            raise ValueError(f"LMS parameters are N in {sizes}, W in (1, 2, 4, 8), 1 <= H <= 30, not {self}")

    @property
    def U(self) -> int:
        return -(-8 * self.N // self.W)

    @property
    def V(self) -> int:
        return -(-(((2 ** self.W - 1) * self.U).bit_length()) // self.W)

    @property
    def P(self) -> int:
        return self.U + self.V

    @property
    def LS(self) -> int:
        return 16 - self.V * self.W

    def hash(self, data: bytes) -> bytes:
        return hashlib.sha512(data).digest()[:self.N]

    def signature_bytes(self) -> int:
        return 4 + self.N * (1 + self.P + self.H)


DEPLOYED = Params(64, 1, 20)


@dataclass(frozen=True)
class PrivateKey:
    params: Params
    I: bytes
    seed: bytes
    tree: tuple[bytes, ...]  # node r at index r, 1 <= r < 2^(H+1)

    @property
    def root(self) -> bytes:
        return self.tree[1]


@dataclass(frozen=True)
class Signature:
    q: int
    C: bytes
    y: tuple[bytes, ...]
    path: tuple[bytes, ...]

    def encode(self) -> bytes:
        return _u32(self.q) + self.C + b"".join(self.y) + b"".join(self.path)

    @classmethod
    def decode(cls, params: Params, data: bytes) -> "Signature":
        if len(data) != params.signature_bytes():
            raise ValueError(f"an LMS signature for {params} is {params.signature_bytes()} bytes, not {len(data)}")
        n, q = params.N, int.from_bytes(data[:4], "big")
        cut = [data[4 + n * k:4 + n * (k + 1)] for k in range(1 + params.P + params.H)]
        return cls(q, cut[0], tuple(cut[1:1 + params.P]), tuple(cut[1 + params.P:]))


# --- the messages ------------------------------------------------------------------------------------------------------------

#: role -> its tag, which every digest of the role begins with. The network certificate's is the message's own,
#: distinct from the ids of the checks that read it (`verity/network-certificate/v4`, `verity/network-records/v3`).
ROLES = {"network-certificate": "verity/network-certificate-message/v2",
         "memory-challenger-record": "verity/memory-challenger-record/v1"}
DIRECTIONS = ("egress", "ingress")
#: the length of a certifier session id, which the challenger samples for each session of the certifier
SESSION_BYTES = 32


def _field(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def _digest(role: str, I: bytes, context: bytes, record: bytes) -> bytes:
    if len(I) != 16:
        raise ValueError(f"an LMS identifier is 16 bytes, not {len(I)}")
    return hashlib.sha512(_field(ROLES[role].encode()) + I + _field(context) + _field(record)).digest()


def certificate_message(I: bytes, session: bytes, link: str, direction: str, window: int, record: bytes) -> bytes:
    """The message a network certificate signs: ``SHA-512(tag || I || context || record)``, the tag
    ``verity/network-certificate-message/v2``, the context the certifier session's id (`SESSION_BYTES`), the link's name,
    the direction (``egress`` or ``ingress``) and the window's index, each field but I and the window prefixed with its
    length (u64)."""
    if direction not in DIRECTIONS or isinstance(window, bool) or not 0 <= window < 2 ** 64:
        raise ValueError(f"a certificate's direction is one of {DIRECTIONS} and its window a u64, not {direction!r}, {window!r}")
    if len(session) != SESSION_BYTES:
        raise ValueError(f"a certifier session id is {SESSION_BYTES} bytes, not {len(session)}")
    context = _field(bytes(session)) + _field(link.encode()) + _field(direction.encode()) + window.to_bytes(8, "big")
    return _digest("network-certificate", I, context, record)


def memory_challenger_message(I: bytes, session: str | bytes, record: bytes) -> bytes:
    """The message a memory challenger's session record signs: as `certificate_message`, the tag
    ``verity/memory-challenger-record/v1`` and the context the session id."""
    sid = session.encode() if isinstance(session, str) else bytes(session)
    return _digest("memory-challenger-record", I, _field(sid), record)


# --- the scheme --------------------------------------------------------------------------------------------------------------

def _coef(s: bytes, i: int, w: int) -> int:
    return ((2 ** w - 1) & (s[i * w // 8] >> (8 - (w * (i % (8 // w)) + w))))


def digits(params: Params, Q: bytes) -> list[int]:
    """The P chain lengths a signature reveals: Q's U digits, then its checksum's V (RFC 8554 §4.4, ``coef(Q || Cksm(Q))``)."""
    w = params.W
    cksm = sum(2 ** w - 1 - _coef(Q, i, w) for i in range(params.U)) << params.LS
    s = Q + _u16(cksm)
    return [_coef(s, i, w) for i in range(params.P)]


def _chain(params: Params, I: bytes, q: int, i: int, tmp: bytes, start: int, stop: int) -> bytes:
    for j in range(start, stop):
        tmp = params.hash(I + _u32(q) + _u16(i) + bytes([j]) + tmp)
    return tmp


def _x(params: Params, I: bytes, seed: bytes, q: int, i: int) -> bytes:
    return params.hash(I + _u32(q) + _u16(i) + b"\xff" + seed)


def _ots_public(params: Params, I: bytes, q: int, z: list[bytes]) -> bytes:
    return params.hash(I + _u32(q) + _u16(D_PBLC) + b"".join(z))


def _leaf(params: Params, I: bytes, q: int, K: bytes) -> bytes:
    return params.hash(I + _u32(2 ** params.H + q) + _u16(D_LEAF) + K)


def _interior(params: Params, I: bytes, r: int, left: bytes, right: bytes) -> bytes:
    return params.hash(I + _u32(r) + _u16(D_INTR) + left + right)


def _leaves(params: Params, I: bytes, seed: bytes, lo: int, hi: int) -> list[bytes]:
    e = 2 ** params.W - 1
    out = []
    for q in range(lo, hi):
        z = [_chain(params, I, q, i, _x(params, I, seed, q, i), 0, e) for i in range(params.P)]
        out.append(_leaf(params, I, q, _ots_public(params, I, q, z)))
    return out


def keygen(params: Params, seed: bytes, workers: int = 1) -> PrivateKey:
    """The key with 2^H one-time keys that ``seed`` (N secret bytes) determines, its whole tree computed in ``workers``
    processes."""
    if len(seed) != params.N:
        raise ValueError(f"an LMS seed is {params.N} bytes, not {len(seed)}")
    return key(params, hashlib.sha512(b"verity/lms/v0/I" + seed).digest()[:16], seed, workers)


def key(params: Params, I: bytes, seed: bytes, workers: int = 1) -> PrivateKey:
    """The key of identifier ``I`` (16 bytes) whose one-time keys ``seed`` derives (RFC 8554 Appendix A)."""
    if len(I) != 16:
        raise ValueError(f"an LMS identifier is 16 bytes, not {len(I)}")
    top = 2 ** params.H
    if workers <= 1:
        leaves = _leaves(params, I, seed, 0, top)
    else:
        step = -(-top // (4 * workers))
        with ProcessPoolExecutor(workers) as pool:
            parts = [pool.submit(_leaves, params, I, seed, lo, min(lo + step, top)) for lo in range(0, top, step)]
            leaves = [x for f in parts for x in f.result()]
    tree: list[bytes] = [b""] * top + leaves
    for r in range(top - 1, 0, -1):
        tree[r] = _interior(params, I, r, tree[2 * r], tree[2 * r + 1])
    return PrivateKey(params, I, seed, tuple(tree))


def public_key(key: PrivateKey) -> bytes:
    """``I || T[1]``."""
    return key.I + key.root


def _message_hash(params: Params, I: bytes, q: int, C: bytes, message: bytes) -> bytes:
    return params.hash(I + _u32(q) + _u16(D_MESG) + C + message)


def sign_at(key: PrivateKey, q: int, message: bytes) -> Signature:
    """The signature of ``message`` with one-time key q, with no state: for vectors and tests. A signer uses `sign`, since two
    messages signed at one q give the key away."""
    p = key.params
    if not 0 <= q < 2 ** p.H:
        raise ValueError(f"q is below 2^{p.H}, not {q}")
    C = p.hash(key.I + _u32(q) + _u16(0xFFFD) + b"\xff" + key.seed)
    a = digits(p, _message_hash(p, key.I, q, C, message))
    y = tuple(_chain(p, key.I, q, i, _x(p, key.I, key.seed, q, i), 0, a[i]) for i in range(p.P))
    r, path = 2 ** p.H + q, []
    while r > 1:
        path.append(key.tree[r ^ 1])
        r //= 2
    return Signature(q, C, y, tuple(path))


def sign(key: PrivateKey, state: "LeafState", message: bytes) -> Signature:
    """The signature of ``message`` at the next unused leaf, which ``state`` has advanced past on disk before this returns."""
    if state.public != public_key(key) or state.H != key.params.H:
        raise StateError("the leaf state is another key's")
    return sign_at(key, state.reserve(), message)


def root_of(params: Params, I: bytes, message: bytes, sig: Signature) -> bytes:
    """The root the signature's one-time key and path give (RFC 8554 Algorithms 4b and 6a)."""
    if not 0 <= sig.q < 2 ** params.H or len(sig.y) != params.P or len(sig.path) != params.H:
        raise ValueError(f"a signature for {params} has q below 2^H, P chain values and H path nodes")
    e = 2 ** params.W - 1
    a = digits(params, _message_hash(params, I, sig.q, sig.C, message))
    z = [_chain(params, I, sig.q, i, sig.y[i], a[i], e) for i in range(params.P)]
    tmp, r = _leaf(params, I, sig.q, _ots_public(params, I, sig.q, z)), 2 ** params.H + sig.q
    for sib in sig.path:
        tmp = _interior(params, I, r // 2, sib, tmp) if r & 1 else _interior(params, I, r // 2, tmp, sib)
        r //= 2
    return tmp


def verify(params: Params, public: bytes, message: bytes, sig: Signature) -> bool:
    """Whether ``sig`` signs ``message`` under ``public`` (``I || T[1]``)."""
    if len(public) != 16 + params.N:
        raise ValueError(f"an LMS public key is {16 + params.N} bytes, not {len(public)}")
    try:
        return root_of(params, public[:16], message, sig) == public[16:]
    except ValueError:
        return False


# --- the leaf state ----------------------------------------------------------------------------------------------------------

class StateError(RuntimeError):
    """A refusal to sign: the leaf state is missing, unreadable, another key's, behind a leaf already used, held by another
    signer, or used up; or a refusal to start a session already started."""


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class LeafState:
    """The next unused leaf of one key, and the session ids it has started, in a file that fails closed.

    ``path`` holds ``{"public", "H", "next", "sessions", "check"}`` as JSON (``sessions`` the ids `start` has recorded, in
    hex, and ``check`` the first 16 bytes of SHA-512 over the rest), and ``path.journal`` every leaf reserved, one per line.
    Opening takes ``path.lock`` exclusively for the object's life: a second signer on the same state is refused, or with
    ``wait`` waits on the lock until the first closes the state, so signers that each open it only to sign take turns.
    Opening refuses a state that is missing, unreadable, another key's, or behind its own journal or ``floor`` (the
    caller's next leaf known unused, such as 1 + the last q it published). `reserve` writes the advanced state and its
    journal line and fsyncs both before it returns the leaf, so a crash never reuses one: at worst the journal misses the
    one leaf the state already skipped. `start` records a session id as started the same way, and refuses one already
    recorded, so no two signers of the key run one session. A restore of both files from one snapshot is past what a file
    can see; ``floor`` is its guard for leaves, and for session ids there is none: both rest on its host never restoring
    an earlier copy of the state.
    """

    def __init__(self, path: str | os.PathLike, public: bytes, H: int, *, floor: int = 0, wait: bool = False) -> None:
        self.path, self.public, self.H = Path(path), bytes(public), H
        self._journal = self.path.with_name(self.path.name + ".journal")
        self._lock = open(self.path.with_name(self.path.name + ".lock"), "a+")
        self._session: bytes | None = None
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise StateError(f"{self.path} is held by another signer") from None
        try:
            self._next, self._sessions = self._read()
            if self._next < floor:
                raise StateError(f"{self.path}: next leaf {self._next} is behind the floor {floor}")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def create(path: str | os.PathLike, public: bytes, H: int) -> None:
        """A new key's state, at leaf 0; refused when the state or its journal exists."""
        p = Path(path)
        j = p.with_name(p.name + ".journal")
        if p.exists() or j.exists():
            raise StateError(f"{p} already has a leaf state")
        with open(j, "x") as f:
            os.fsync(f.fileno())
        LeafState._write(p, bytes(public), H, 0, ())

    @staticmethod
    def _body(public: bytes, H: int, nxt: int, sessions: tuple[bytes, ...]) -> dict:
        b = {"public": public.hex(), "H": H, "next": nxt, "sessions": [s.hex() for s in sessions]}
        b["check"] = hashlib.sha512(json.dumps(b, sort_keys=True).encode()).hexdigest()[:32]
        return b

    @staticmethod
    def _write(p: Path, public: bytes, H: int, nxt: int, sessions: tuple[bytes, ...]) -> None:
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(LeafState._body(public, H, nxt, sessions), f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        _fsync_dir(p.parent)

    def _read(self) -> tuple[int, tuple[bytes, ...]]:
        try:
            body = json.loads(self.path.read_text())
            used = [int(x) for x in self._journal.read_text().splitlines()]
            sessions = tuple(bytes.fromhex(s) for s in body["sessions"]) if isinstance(body, dict) else ()
            intact = (isinstance(body, dict) and isinstance(body.get("next"), int)
                      and body == self._body(bytes.fromhex(str(body.get("public"))), body.get("H"), body["next"],
                                             sessions))
        except FileNotFoundError as e:
            raise StateError(f"{e.filename}: missing") from None
        except (OSError, ValueError, TypeError, KeyError) as e:
            raise StateError(f"{self.path}: unreadable ({type(e).__name__})") from None
        if not intact:
            raise StateError(f"{self.path}: unreadable (its check does not match)")
        if body["public"] != self.public.hex() or body["H"] != self.H:
            raise StateError(f"{self.path} is another key's")
        nxt = body["next"]
        if any(b <= a for a, b in zip(used, used[1:])):
            raise StateError(f"{self._journal}: unreadable (not increasing)")
        last = used[-1] if used else -1
        if last >= nxt:
            raise StateError(f"{self.path}: next leaf {nxt} is behind leaf {last}, already used")
        if last < nxt - 2:
            raise StateError(f"{self._journal}: behind its state (last leaf {last}, next {nxt})")
        return nxt, sessions

    def _current(self) -> tuple[int, tuple[bytes, ...]]:
        if self._lock.closed:
            raise StateError(f"{self.path}: closed")
        nxt, sessions = self._read()
        if (nxt, sessions) != (self._next, self._sessions):
            raise StateError(f"{self.path}: changed under this signer")
        return nxt, sessions

    def start(self, session: bytes) -> None:
        """Start this signer's one session, ``session``, after the state on disk records it; refused when the state has
        already started it, unless this signer did, or this signer has started another session."""
        session = bytes(session)
        nxt, sessions = self._current()
        if self._session == session:
            return
        if self._session is not None:
            raise StateError(f"{self.path}: this signer has already started a session")
        if session in sessions:
            raise StateError(f"{self.path}: session {session.hex()} is already started")
        self._write(self.path, self.public, self.H, nxt, sessions + (session,))
        self._sessions, self._session = sessions + (session,), session

    def reserve(self) -> int:
        """The next unused leaf, after the state on disk is past it."""
        nxt, sessions = self._current()
        if nxt >= 2 ** self.H:
            raise StateError(f"{self.path}: all 2^{self.H} leaves are used")
        self._write(self.path, self.public, self.H, nxt + 1, sessions)
        with open(self._journal, "a") as f:
            f.write(f"{nxt}\n")
            f.flush()
            os.fsync(f.fileno())
        self._next = nxt + 1
        return nxt

    def close(self) -> None:
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    def __enter__(self) -> "LeafState":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
