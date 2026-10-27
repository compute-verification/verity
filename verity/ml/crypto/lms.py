"""A stateful hash-based signature: LMS with LM-OTS (RFC 8554) over SHA-512, in plain Python, the scheme of the network
certifier's certificates and the memory challenger's records, and the reference of their in-proof verifier
(`verity.ml.commitments.gates.lms`). Its security is the claim ``euf-cma/lms-sha512-n64`` (`verity.core.claims`).

**Use.** One key per device per role. ``keygen(DEPLOYED, seed)`` (seed: N = 64 secret bytes the device keeps), then once
``LeafState.create(path, public_key(key), DEPLOYED.H)``; to sign, ``sign(key, state, message)`` with ``state = LeafState(path,
public_key(key), DEPLOYED.H)``; to check, ``verify(DEPLOYED, public, message, sig)``. The message is a role's digest
(`certificate_message`, `memory_challenger_message`), never hand-rolled. A network certifier first starts its session on
the state (``state.start(session)``), which refuses a session id the key has already started.

**HSS: a device's identity and its parties' keys** (RFC 8554 §6 at two levels; NIST SP 800-208). A device's long-lived
top tree is its identity (`Identity.init`, made on the device from its own coins; its public key is the device's key in
the deployment configuration). Each party on the device gets a short-lived lower tree (`Identity.issue`), whose public key
the identity signs once (`party_key_message`, binding the deployment's hash and the party's name). The lower key signs
the party's messages (`PartyKey`): one root per certification period (`period_root_message`), which goes to the prover
only, and one public liveness record per sampling block (`liveness_message`). An `HssSignature` is the lower key, its
certificate and the signature; `hss_verify` checks both levels, and `verify_liveness` a liveness record in the clear. Both
levels are at ``DEPLOYED`` and both trees' states are `LeafState`s, so neither ever signs twice at a leaf, as long as its
host never restores an earlier copy of the state (`faithful`, `verity.core.claims`).

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

__all__ = ["D_INTR", "D_LEAF", "D_MESG", "D_PBLC", "DEPLOYED", "DEPLOYMENT_BYTES", "DIRECTIONS", "HssSignature", "Identity",
           "LeafState", "Params", "PartyKey", "PrivateKey", "ROLES", "SESSION_BYTES", "Signature", "StateError",
           "certificate_message", "digits", "hss_verify", "key", "keygen", "liveness_message", "memory_challenger_message",
           "party_key_message", "period_root_message", "public_key", "root_of", "sign", "sign_at", "verify",
           "verify_certificate", "verify_liveness", "TOY_N"]

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
#: distinct from the ids of the checks that read it (`verity/network-certificate/v4`, `verity/network-records/v3`). The
#: last three are HSS's (`party_key_message`, `period_root_message`, `liveness_message`).
ROLES = {"network-certificate": "verity/network-certificate-message/v2",
         "memory-challenger-record": "verity/memory-challenger-record/v1",
         "party-key": "verity/party-key/v1",
         "period-root": "verity/period-root/v1",
         "liveness": "verity/liveness/v1"}
DIRECTIONS = ("egress", "ingress")
#: the length of a certifier session id, which the challenger draws for each session of the certifier
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


#: the length of a deployment configuration's hash, SHA-512 of the file
DEPLOYMENT_BYTES = 64


def _u64(x: int, what: str) -> bytes:
    if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2 ** 64:
        raise ValueError(f"{what} is a u64, not {x!r}")
    return x.to_bytes(8, "big")


def _party_context(deployment: bytes, party: str) -> bytes:
    if len(deployment) != DEPLOYMENT_BYTES:
        raise ValueError(f"a deployment's hash is {DEPLOYMENT_BYTES} bytes, not {len(deployment)}")
    return _field(bytes(deployment)) + _field(party.encode())


def _period_context(deployment: bytes, party: str, attachment: bytes, link: str, period: int) -> bytes:
    return (_party_context(deployment, party) + _field(bytes(attachment)) + _field(link.encode())
            + _u64(period, "a certification period's index"))


def party_key_message(I: bytes, deployment: bytes, party: str, lower: bytes) -> bytes:
    """The message a device's identity (the top tree of its HSS key, identifier ``I``) signs to certify one party's lower
    key: the tag ``verity/party-key/v1``, the context the deployment's hash and the party's name, the record the lower
    key's public key (``I || T[1]``), each field but I prefixed with its length (u64)."""
    return _digest("party-key", I, _party_context(deployment, party), bytes(lower))


def period_root_message(I: bytes, deployment: bytes, party: str, attachment: bytes, link: str, period: int, count: int,
                        root: bytes) -> bytes:
    """The message a party's lower key (identifier ``I``) signs once per certification period: the tag
    ``verity/period-root/v1``, the context the deployment's hash, the party's name, the attachment id the challenger drew,
    the link's name (for a party on no link, its device role), the period's index and the number of records under the root
    (u64 each, unprefixed), the record the root of the period's records tree (`verity.core.attest`)."""
    context = _period_context(deployment, party, attachment, link, period) + _u64(count, "a period's record count")
    return _digest("period-root", I, context, bytes(root))


def liveness_message(I: bytes, deployment: bytes, party: str, attachment: bytes, link: str, period: int,
                     block: int) -> bytes:
    """The message of a party's one public record per sampling block (or per run, for a party that runs in runs): "block
    ``block`` of the period was certified", with no content. The tag ``verity/liveness/v1``, the context as
    `period_root_message`'s with the block's index (u64) in the place of the count, the record empty."""
    return _digest("liveness", I, _period_context(deployment, party, attachment, link, period)
                   + _u64(block, "a sampling block's index"), b"")


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


# --- HSS: a device's identity and its parties' keys --------------------------------------------------------------------

@dataclass(frozen=True)
class HssSignature:
    """A two-level HSS signature (RFC 8554 §6 at L = 2, without its type fields and its count of levels): the party's lower
    public key (``I || T[1]``), the identity's signature of the lower key's `party_key_message` (its certificate), and the
    lower key's signature of the message. Encoded ``lower || certificate || signature``, 16 + N + 2 `signature_bytes`."""

    lower: bytes
    certificate: Signature
    signature: Signature

    def encode(self) -> bytes:
        return self.lower + self.certificate.encode() + self.signature.encode()

    @classmethod
    def decode(cls, params: Params, data: bytes) -> "HssSignature":
        n, s = 16 + params.N, params.signature_bytes()
        if len(data) != n + 2 * s:
            raise ValueError(f"an HSS signature for {params} is {n + 2 * s} bytes, not {len(data)}")
        return cls(bytes(data[:n]), Signature.decode(params, data[n:n + s]), Signature.decode(params, data[n + s:]))


def verify_certificate(params: Params, identity: bytes, deployment: bytes, party: str, lower: bytes,
                       certificate: Signature) -> bool:
    """Whether ``certificate`` is the identity's signature over ``lower``'s `party_key_message` for ``party`` in the
    deployment; False for a lower key of another length."""
    if len(identity) != 16 + params.N:
        raise ValueError(f"an LMS public key is {16 + params.N} bytes, not {len(identity)}")
    if len(lower) != 16 + params.N:
        return False
    return verify(params, identity, party_key_message(identity[:16], deployment, party, lower), certificate)


def hss_verify(params: Params, identity: bytes, deployment: bytes, party: str, sig: HssSignature, message: bytes) -> bool:
    """Whether ``sig`` signs ``message`` under the device ``identity`` for ``party`` in the deployment: its lower key's
    certificate verifies under the identity, and its signature under the lower key. ``message`` is a lower key's role
    digest (`period_root_message`, `liveness_message`), made with the lower key's identifier ``sig.lower[:16]``."""
    return (verify_certificate(params, identity, deployment, party, sig.lower, sig.certificate)
            and verify(params, sig.lower, message, sig.signature))


def verify_liveness(params: Params, identity: bytes, deployment: bytes, party: str, attachment: bytes, link: str,
                    period: int, block: int, sig: HssSignature) -> bool:
    """Whether ``sig`` is the party's public liveness record of sampling block ``block`` of the period: checked in the
    clear, as the verifier reads it."""
    if len(sig.lower) != 16 + params.N:
        return False
    message = liveness_message(sig.lower[:16], deployment, party, attachment, link, period, block)
    return hss_verify(params, identity, deployment, party, sig, message)


IDENTITY_SEED, IDENTITY_STATE, IDENTITY_PUBLIC = "identity.seed", "identity.state", "identity.public"
LOWER_SEED, LOWER_STATE, LOWER_CERTIFICATE = "lower.seed", "lower.state", "lower.certificate"


def _write_secret(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _read_secret(path: Path, n: int) -> bytes:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise StateError(f"{path}: missing") from None
    if len(data) != n:
        raise StateError(f"{path}: a seed is {n} bytes, not {len(data)}")
    return data


def _party_dir(deployment: bytes, party: str) -> str:
    if not party or not all(c.isascii() and (c.isalnum() or c in "-_.") for c in party) or party[0] == ".":
        raise ValueError(f"a party's name is ASCII letters, digits, '-', '_' and '.', not starting with '.', not {party!r}")
    if len(deployment) != DEPLOYMENT_BYTES:
        raise ValueError(f"a deployment's hash is {DEPLOYMENT_BYTES} bytes, not {len(deployment)}")
    return f"{bytes(deployment).hex()}-{party}"


class Identity:
    """A device's identity: the top tree of its two-level HSS key (RFC 8554 §6; NIST SP 800-208), made on the device from
    its own coins (`os.urandom`) by `init`, whose public key (``I || T[1]``, in ``identity.public``) is the device's key in a
    deployment configuration. Its seed (``identity.seed``, mode 0600 in a directory of mode 0700) never leaves the device
    and is never written twice. It signs only `party_key_message`s, one leaf each under its `LeafState`
    (``identity.state``): each certifies one lower tree, made by `issue` for one party of one deployment under
    ``parties/<deployment hex>-<party>/<n>/``, ``n`` counting the party's lower trees from 0, so a party whose tree runs out
    is issued the next. Opening the identity rebuilds its tree from the seed (`keygen`; about 16 minutes on one core at
    `DEPLOYED`, divided by ``workers``)."""

    def __init__(self, directory: str | os.PathLike, params: Params = DEPLOYED, workers: int = 1, *,
                 _key: PrivateKey | None = None) -> None:
        self.directory, self.params = Path(directory), params
        self.key = _key or keygen(params, _read_secret(self.directory / IDENTITY_SEED, params.N), workers)
        self.public = public_key(self.key)
        recorded = (self.directory / IDENTITY_PUBLIC).read_text().strip()
        if recorded != self.public.hex():
            raise StateError(f"{self.directory / IDENTITY_PUBLIC} is another key's")

    @classmethod
    def init(cls, directory: str | os.PathLike, params: Params = DEPLOYED, workers: int = 1) -> "Identity":
        """A new identity in ``directory``; refused when the directory already holds one."""
        d = Path(directory)
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_secret(d / IDENTITY_SEED, os.urandom(params.N))
        key = keygen(params, _read_secret(d / IDENTITY_SEED, params.N), workers)
        LeafState.create(d / IDENTITY_STATE, public_key(key), params.H)
        (d / IDENTITY_PUBLIC).write_text(public_key(key).hex() + "\n")
        return cls(d, params, workers, _key=key)

    def _issued(self, deployment: bytes, party: str) -> list[Path]:
        base = self.directory / "parties" / _party_dir(deployment, party)
        return sorted((p for p in base.glob("*") if p.name.isdigit()), key=lambda p: int(p.name)) if base.is_dir() else []

    def issue(self, deployment: bytes, party: str, workers: int = 1) -> "PartyKey":
        """The next lower tree of ``party`` in the deployment: its seed from the device's coins, its leaf state, and the
        identity's certificate of its public key, signed at the identity's next leaf."""
        n = len(self._issued(deployment, party))
        d = self.directory / "parties" / _party_dir(deployment, party) / str(n)
        d.mkdir(mode=0o700, parents=True, exist_ok=False)
        _write_secret(d / LOWER_SEED, os.urandom(self.params.N))
        key = keygen(self.params, _read_secret(d / LOWER_SEED, self.params.N), workers)
        lower = public_key(key)
        LeafState.create(d / LOWER_STATE, lower, self.params.H)
        with LeafState(self.directory / IDENTITY_STATE, self.public, self.params.H, wait=True) as state:
            cert = sign(self.key, state, party_key_message(self.key.I, deployment, party, lower))
        tmp = d / (LOWER_CERTIFICATE + ".tmp")
        tmp.write_bytes(cert.encode())
        os.replace(tmp, d / LOWER_CERTIFICATE)
        _fsync_dir(d)
        return PartyKey(d, self.public, deployment, party, self.params, workers, _key=key)

    def party(self, deployment: bytes, party: str, workers: int = 1) -> "PartyKey":
        """The party's newest lower tree in the deployment; refused when it has none."""
        issued = self._issued(deployment, party)
        if not issued:
            raise StateError(f"{self.directory}: no key issued to {party!r} in this deployment")
        return PartyKey(issued[-1], self.public, deployment, party, self.params, workers)


class PartyKey:
    """One party's lower tree on its device, certified by the device's identity for the party in one deployment: it signs
    the party's `period_root_message` once per certification period and its `liveness_message` once per sampling block,
    each at the next leaf of its `LeafState`, into an `HssSignature`. The signed roots go to the prover only, as witness;
    the liveness records, and with them the lower key and its certificate, are public.

    It signs for one attachment at a time, the one it last started (`start`), and starts each attachment once in the
    key's life: a restart within an attachment signs nothing more of it. That rests on no-state-rollback, the leaf
    state's own premise: a state restored from before the start would start it again."""

    def __init__(self, directory: str | os.PathLike, identity: bytes, deployment: bytes, party: str,
                 params: Params = DEPLOYED, workers: int = 1, *, _key: PrivateKey | None = None) -> None:
        self.directory, self.identity, self.params = Path(directory), bytes(identity), params
        self.deployment, self.party = bytes(deployment), party
        self.key = _key or keygen(params, _read_secret(self.directory / LOWER_SEED, params.N), workers)
        self.lower = public_key(self.key)
        self.certificate = Signature.decode(params, (self.directory / LOWER_CERTIFICATE).read_bytes())
        if not verify_certificate(params, self.identity, self.deployment, party, self.lower, self.certificate):
            raise StateError(f"{self.directory}: its certificate does not verify under the identity")
        self.attachment: bytes | None = None

    def start(self, attachment: bytes) -> None:
        """Start this key's run of ``attachment`` (the attachment id the challenger drew), after the key's leaf state on
        disk records it as started (`LeafState.start`), so that `period_root` and `liveness` sign for it; a no-op for the
        attachment already started here, and refused for one the state records already (this key's earlier run of it, or
        another signer's)."""
        attachment = bytes(attachment)
        if attachment == self.attachment:
            return
        with LeafState(self.directory / LOWER_STATE, self.lower, self.params.H, wait=True) as state:
            state.start(attachment)
        self.attachment = attachment

    def _started(self, attachment: bytes) -> None:
        if self.attachment is None or bytes(attachment) != self.attachment:
            raise StateError(f"{self.directory}: attachment {bytes(attachment).hex()} is not the one this key started")

    def sign(self, message: bytes) -> HssSignature:
        """``message`` (a role digest made with this key's identifier ``self.key.I``) signed at the next unused leaf."""
        with LeafState(self.directory / LOWER_STATE, self.lower, self.params.H, wait=True) as state:
            return HssSignature(self.lower, self.certificate, sign(self.key, state, message))

    def period_root(self, attachment: bytes, link: str, period: int, count: int, root: bytes) -> HssSignature:
        self._started(attachment)
        return self.sign(period_root_message(self.key.I, self.deployment, self.party, attachment, link, period, count,
                                             root))

    def liveness(self, attachment: bytes, link: str, period: int, block: int) -> HssSignature:
        self._started(attachment)
        return self.sign(liveness_message(self.key.I, self.deployment, self.party, attachment, link, period, block))
