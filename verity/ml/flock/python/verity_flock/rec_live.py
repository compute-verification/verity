"""``rec/inner-transcript/v1``: the inner session of a recursive session (hiding model architectures), run by the unmodified
C-Flock prover against a verifier that never sees the hidden statement or the prover's messages.

    python -m verity_flock.rec_live --listen 127.0.0.1:PORT --out DIR --schedule FILE [--sessions 1]
    python -m verity_flock.rec_live --schedule-for STATEMENT         a statement's schedule (flock-circuit statement's output)
    python -m verity_flock.rec_live --schedule-of PUBLIC_JSON        a record's schedule, to compare with the one it was held to

The prover (``flock-circuit prove --verifier ADDR`` under ``FC_COINS=os``) speaks the live protocol (``live/src/lib.rs``) to
:class:`Proxy`, which runs on the prover's side. The proxy is the firewall of the inner phase (live PROTOCOL.md §10.3): it
answers what the live server would, except that every coin comes from :class:`CoinServer`, the verifier's side, which is shown
only the session's public schedule and salted hashes:

* ``Hello``: the pinned schedule's (profile, reps, flavor, tables, the link's shape), without Σ (a digest of the hidden
  statement), once the prover's, Σ aside, is the same canonical JSON (`canonical`); never the prover's own text or object;
* ``Commit``: each table's level-0 cap (``root_B``) as its salted top (`commit_top`), answered with the link points; root_F
  (Σ) stays with the prover;
* each ``Round``: the salted top of each tree it commits (its ``BYTES`` frames that are caps), and ``c_k``, a hiding
  commitment to the round's free message, answered with the round's ``n`` coins.

**Salted tops.** A tree's top (a Ligerito cap of ``2^c`` nodes, 64 bytes each) is committed node by node, each node as a
registered row is (`top_rows`: its 32 u16 words and 32 zero words, one SHA-512 block; ``hm96-sha512/row/v1`` under its own
fresh 192-byte salt), so the coin server holds one ``b || c`` a node and V*'s openings each read their node as one row of the
registered value the verifier builds from them (`verity_flock.rec_outer`). The nodes and salts are the prover's
(``private.json``). Level 0's cap reaches the firewall three times (``Commit``, the rep's binding round, Ligerito's first
absorb): it is committed once, at ``Commit``, and the firewall refuses a binding round whose cap is not ``root_B``; a later
copy is counted (``level0``), never committed again. The inner trees' leaves need no salt of their own (the statement's
``flock-leaf/sha512-unsalted``, `verity_flock.circuit.LEAF_SCHEME_PLAIN`): nothing about a tree but its salted top leaves the
firewall.

**The schedule.** The proxy holds the session to a pinned schedule (`schedule_of`: the public ``Hello``, each table's root
node count, the number of ``Link``'s ``y`` elements, every stream's parent, position and round count, and per round its
stream, index, coin count, its salted tops' node counts, its copies of a level-0 top and its message-row count).
``--schedule`` gives it and is required: `schedule_for` derives it from the statement's public shape alone (``flock-circuit
statement``), never from a session the prover ran, and `serve` writes it to ``OUT/schedule.json`` before the first session. A
request that leaves the schedule, or a ``Link`` whose ``y`` is not the schedule's number of zero elements (the prover sends
zeros), is refused ``FIREWALL-REFUSED`` before the coin server sees it. A refusal, or a session that ends without ``Finish``,
is a **stop**: the coin server's record says so (``end``: ``stop``, not why), and `serve` writes that record and counts the
session.

**Live coins.** Each round's coins are drawn by the coin server when the round's commitments reach it, never before
(`RecursiveSound`'s game and `RecursiveZK` draw them so, and `Flock.Firewall.Contract`'s ``coins`` rule lets coins out only
after the commitments the schedule owes); `serve`'s challenger commits to none. A challenger may instead commit to every
coin before the session (`CoinServer.commit_coins`, from the schedule alone: one hm96-sha512 commitment a round, the link
points as round 0 of stream ``link``, each to `coin_message` under a fresh salt of the challenger's). A proxy given those
commitments (``commitments``) stops a session whose coins do not open their commitment or come without an opening, and a
proxy given none stops one whose coins come with an opening, as the contract decides under the public input's ``coins``.

A round's content is the transcript-v2 frames the prover absorbed since its last squeeze (PROTOCOL.md §6.1). `split_round`
sorts them: the domain and labels are public constants and are dropped; a ``BYTES`` frame is a cap (committed as its salted
top), a proof-of-work nonce (8 bytes, dropped: with live coins grinding buys nothing) or, in a rep's first round, the
statement digest (dropped: it is a function of the hidden statement); the observed field elements are the free message ``F_k``. ``F_k`` is
laid out as one row of whole :data:`ROW_WORDS`-word blocks (each element 8 words, its 16 bytes lo LE then hi LE; the last
block zero-padded) and committed as a registered row is (``verity.core.protocols.sampled_proofs.one_stage.registered``:
``hm96-sha512/row/v1`` under its own fresh 192-byte salt), so ``c_k`` is that row's ``b || c``: one commitment a round, as
one port of V*'s algebra reads it. A round with no free message has an empty ``c_k``. The row and salt are the prover's:
V*'s algebra opens each round's row against the registered value the verifier builds from that round's ``c_k`` of both reps
(`message_row`, `commit_rows`).

At ``Finish`` the proxy writes the session directory (``session.json``, a ``flock-live-session/v1`` record that
``flock-circuit replay`` and the Lean verifier check, and ``<table>.rep<r>.bin``), the coin server's record ``public.json``
(``rec/inner-transcript/v1``: everything the verifier saw and issued, ``tops`` naming how its tops are salted) and the
prover's ``private.json`` (every top's nodes and every round's rows, with their salts), and answers the prover with a verdict
that says nothing was verified: the inner proof is V*'s witness.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from verity.ml.commitments import hm96

from .rec_algebra import fast100

KIND = "rec/inner-transcript/v1"
SESSION_KIND = "flock-live-session/v1"
#: how a record's tree tops are salted: each 64-byte node a registered row (``hm96-sha512/row/v1``) under its own salt
#: (`commit_top`)
TOPS = "hm96-sha512/v1/node-row"

OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES = 1, 2, 3, 4, 5
KIND_SCALAR, KIND_SLICE = 1, 2
LABEL_R1CS = b"flock-r1cs-v0"
NONCE_BYTES = 8
CAP_NODE_BYTES = 64

REQ_HELLO, REQ_LINK, REQ_OPEN, REQ_ROUND, REQ_PROOF, REQ_FINISH, REQ_COMMIT, REQ_PRIME, REQ_PING, REQ_REGISTER = range(1, 11)
RESP_OK, RESP_COINS, RESP_REFUSED, RESP_VERDICT, RESP_WORDS, RESP_DRAW = range(1, 7)

#: u16 words per block of a message row: one SHA-512 block, 8 field elements
ROW_WORDS = 64
#: `Flock.Firewall.Contract.COINS_TAG`: the coin message's tag
COINS_TAG = b"verity/firewall-coins/v0\0"
#: the stream name of the link points' coins (``Commit``'s answer), round 0, in the challenger's commitments and the transcript
LINK_COINS = "link"
COINS_KIND = "rec/inner-coins/v0"
F_WORDS = 8
MASK64 = (1 << 64) - 1

#: a flock-circuit statement's session (`schedule_for`): its one table, fast100's reps, the link's points (one ``y``
#: element each, a table) and each point's free in-block coordinates (``circuit.rs`` ``PT_LOCAL``; ``nbl`` block ones follow)
TABLE, REPS, LINK_POINTS, PT_LOCAL = "circuit", 2, 2, 24
#: candidate 3's stacked table R (`stacked_rounds`)
STACKED = "stacked"
#: candidate 3's sparse phase, a stream a rep (`sparse_rounds`), each its rep's child at the rep's end
SPARSE = "sparse"


# --- wire codec (live/src/lib.rs `Req`, `Resp`) -------------------------------------------------------------------------------

class _Cur:
    def __init__(self, b: bytes, at: int = 1):
        self.b, self.at = b, at

    def u32(self) -> int:
        if self.at + 4 > len(self.b):
            raise ValueError("short frame")
        v = struct.unpack_from("<I", self.b, self.at)[0]
        self.at += 4
        return v

    def u64(self) -> int:
        if self.at + 8 > len(self.b):
            raise ValueError("short frame")
        v = struct.unpack_from("<Q", self.b, self.at)[0]
        self.at += 8
        return v

    def bytes(self) -> bytes:
        n = self.u32()
        if self.at + n > len(self.b):
            raise ValueError("short frame")
        v = self.b[self.at:self.at + n]
        self.at += n
        return v

    def str(self) -> str:
        return self.bytes().decode()


def decode_req(b: bytes) -> tuple:
    """``(tag, *fields)`` in ``Req``'s field order."""
    if not b:
        raise ValueError("empty frame")
    c, t = _Cur(b), b[0]
    if t == REQ_HELLO:
        return t, c.str()
    if t in (REQ_LINK, REQ_PING):
        return t, c.bytes()
    if t == REQ_OPEN:
        return t, c.str(), c.str(), c.bytes(), c.u64()
    if t == REQ_ROUND:
        return t, c.str(), c.bytes(), c.u32()
    if t == REQ_PROOF:
        return t, c.str(), c.u32(), c.bytes()
    if t in (REQ_FINISH, REQ_REGISTER):
        return (t,)
    if t == REQ_COMMIT:
        root_f = c.bytes()
        lists = []
        for _ in range(2):
            lists.append([(c.str(), c.bytes()) for _ in range(c.u32())])
        return t, root_f, lists[0], lists[1]
    if t == REQ_PRIME:
        return t, c.bytes(), c.u32()
    raise ValueError(f"unknown request tag {t}")


def _put_bytes(b: bytes) -> bytes:
    return struct.pack("<I", len(b)) + b


def encode_req(t: int, *f) -> bytes:
    """``Req::encode`` (tests drive the proxy with it)."""
    if t == REQ_HELLO:
        return bytes([t]) + _put_bytes(f[0].encode())
    if t in (REQ_LINK, REQ_PING):
        return bytes([t]) + _put_bytes(f[0])
    if t == REQ_OPEN:
        return bytes([t]) + _put_bytes(f[0].encode()) + _put_bytes(f[1].encode()) + _put_bytes(f[2]) + struct.pack("<Q", f[3])
    if t == REQ_ROUND:
        return bytes([t]) + _put_bytes(f[0].encode()) + _put_bytes(f[1]) + struct.pack("<I", f[2])
    if t == REQ_PROOF:
        return bytes([t]) + _put_bytes(f[0].encode()) + struct.pack("<I", f[1]) + _put_bytes(f[2])
    if t in (REQ_FINISH, REQ_REGISTER):
        return bytes([t])
    if t == REQ_COMMIT:
        out = bytes([t]) + _put_bytes(f[0])
        for lst in (f[1], f[2]):
            out += struct.pack("<I", len(lst)) + b"".join(_put_bytes(n.encode()) + _put_bytes(v) for n, v in lst)
        return out
    raise ValueError(f"no encoder for request tag {t}")


def f128_bytes(v: int) -> bytes:
    return struct.pack("<QQ", v & MASK64, v >> 64)


def f128_hex(v: int) -> str:
    """The record's form: hi then lo, each 16 hex digits (``f128_hex`` in lib.rs)."""
    return f"{v >> 64:016x}{v & MASK64:016x}"


def resp_ok() -> bytes:
    return bytes([RESP_OK])


def resp_coins(coins: list[int]) -> bytes:
    return bytes([RESP_COINS]) + struct.pack("<I", len(coins)) + b"".join(f128_bytes(c) for c in coins)


def resp_str(tag: int, s: str) -> bytes:
    return bytes([tag]) + _put_bytes(s.encode())


def decode_resp(b: bytes) -> tuple:
    c, t = _Cur(b), b[0]
    if t == RESP_OK:
        return (t,)
    if t == RESP_COINS:
        n = c.u32()
        return t, [c.u64() | (c.u64() << 64) for _ in range(n)]
    if t in (RESP_REFUSED, RESP_VERDICT, RESP_DRAW):
        return t, c.str()
    raise ValueError(f"unknown response tag {t}")


def read_frame(f) -> bytes | None:
    """One length-prefixed frame, or None at a clean end of stream."""
    head = f.read(4)
    if not head:
        return None
    if len(head) < 4:
        raise ValueError("truncated frame length")
    n = struct.unpack("<I", head)[0]
    b = f.read(n)
    if len(b) != n:
        raise ValueError("truncated frame")
    return b


def write_frame(f, b: bytes) -> None:
    f.write(struct.pack("<I", len(b)) + b)
    f.flush()


# --- a round's content ----------------------------------------------------------------------------------------------------------

def frames(content: bytes) -> list[tuple[int, int, int, bytes]]:
    """The transcript-v2 frames of a round's content: ``(op, kind, len, payload)``; a squeeze's payload is empty."""
    out, at = [], 0
    while at < len(content):
        if at + 16 > len(content):
            raise ValueError("a truncated frame header")
        op, kind, n = content[at], content[at + 1], struct.unpack_from("<Q", content, at + 8)[0]
        at += 16
        size = 16 * n if op == OP_OBSERVE else 0 if op == OP_SQUEEZE else n
        if op not in (OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES):
            raise ValueError(f"unknown frame op {op}")
        if at + size > len(content):
            raise ValueError("a frame overruns its round")
        out.append((op, kind, n, content[at:at + size]))
        at += -(-size // 16) * 16
    return out


@dataclass
class Split:
    """A round sorted: its caps (public), its free message (field elements, the prover's), what was dropped, its coin count."""

    caps: list[bytes] = field(default_factory=list)
    free: list[int] = field(default_factory=list)
    nonces: list[bytes] = field(default_factory=list)
    digest: bytes | None = None
    n: int = 0


def split_round(content: bytes, first_of_rep: bool) -> Split:
    """`frames` sorted (the module docstring). ``first_of_rep``: the round binds the statement (a rep stream's first round:
    ``LABEL_R1CS``, the statement digest, then the level-0 cap)."""
    s, binding = Split(), None
    fr = frames(content)
    if not fr or fr[-1][0] != OP_SQUEEZE:
        raise ValueError("a round without its squeeze")
    for op, kind, n, p in fr:
        if op == OP_LABEL:
            if first_of_rep and p == LABEL_R1CS and binding is None:
                binding = 0
        elif op == OP_BYTES:
            if binding == 0:
                s.digest, binding = p, 1
            elif binding == 1:
                if len(p) % CAP_NODE_BYTES or not p:
                    raise ValueError(f"a level-0 cap of {len(p)} bytes")
                s.caps.append(p)
                binding = 2
            elif len(p) == NONCE_BYTES:
                s.nonces.append(p)
            elif p and len(p) % CAP_NODE_BYTES == 0:
                s.caps.append(p)
            else:
                raise ValueError(f"a BYTES frame of {len(p)} bytes is neither a cap nor a nonce")
        elif op == OP_OBSERVE:
            if (kind == KIND_SCALAR and n != 1) or kind not in (KIND_SCALAR, KIND_SLICE):
                raise ValueError(f"an observe of kind {kind} and length {n}")
            s.free += [int.from_bytes(p[16 * i:16 * i + 8], "little") | (int.from_bytes(p[16 * i + 8:16 * i + 16], "little") << 64)
                       for i in range(n)]
    if first_of_rep and binding != 2:
        raise ValueError("a rep's first round binds no statement digest and level-0 cap")
    s.n = fr[-1][2]
    return s


def message_rows(free: list[int]) -> list[list[int]]:
    """``F_k`` as its round's one row (none for an empty message): element ``i`` is words ``8 i ..``, its lo then hi 64 bits
    as little-endian u16 words, zero words up to whole blocks of :data:`ROW_WORDS`."""
    words: list[int] = []
    for v in free:
        words += [(v >> (16 * j)) & 0xFFFF for j in range(F_WORDS)]
    return [words + [0] * (-len(words) % ROW_WORDS)] if words else []


def row_elements(row: list[int]) -> list[int]:
    """`message_rows`' inverse on one row: its ``len(row) / 8`` field elements (the padding's zeros last)."""
    return [sum(row[F_WORDS * i + j] << (16 * j) for j in range(F_WORDS)) for i in range(len(row) // F_WORDS)]


def commit_rows(rows: list[list[int]], salts: list[bytes]) -> list[bytes]:
    """Each row's ``b || c`` as a registered row's (``hm96-sha512/row/v1`` over ``sha512/row/v1``, role x)."""
    from verity.ml.commitments.rowleaf import ROLE_X, sha512_row_digest

    inst = hm96.SHA512
    return [inst.commit_string(inst.default_key, sha512_row_digest(r, 16, ROLE_X), s) for r, s in zip(rows, salts, strict=True)]


def top_nodes(top: bytes) -> list[bytes]:
    if not top or len(top) % CAP_NODE_BYTES:
        raise ValueError(f"a tree top of {len(top)} bytes")
    return [top[i:i + CAP_NODE_BYTES] for i in range(0, len(top), CAP_NODE_BYTES)]


def top_rows(cap: bytes) -> list[list[int]]:
    """A tree's top as registered rows, one a node: its 64 bytes as 32 little-endian u16 words, then 32 zero words."""
    if not cap or len(cap) % CAP_NODE_BYTES:
        raise ValueError(f"a top of {len(cap)} bytes is not whole {CAP_NODE_BYTES}-byte nodes")
    return [[cap[i + j] | (cap[i + j + 1] << 8) for j in range(0, CAP_NODE_BYTES, 2)] + [0] * (ROW_WORDS - CAP_NODE_BYTES // 2)
            for i in range(0, len(cap), CAP_NODE_BYTES)]


def commit_top(cap: bytes, salts: list[bytes]) -> list[bytes]:
    """A top's salted commitment: each node's ``b || c`` as a registered row's (`top_rows`), one salt a node."""
    return commit_rows(top_rows(cap), salts)


def coin_message(stream: str, k: int, coins: list[int]) -> bytes:
    """What the challenger commits to for round ``k`` of ``stream``'s coins (`Flock.Firewall.Contract.coinMsg`): the tag,
    the stream's UTF-8 bytes after their length, the round, the coin count (each 4 bytes big-endian), then each coin's 16
    bytes (`f128_bytes`)."""
    s = stream.encode()
    return COINS_TAG + struct.pack(">I", len(s)) + s + struct.pack(">II", k, len(coins)) + b"".join(f128_bytes(c) for c in coins)


def coin_commitment(stream: str, k: int, coins: list[int], salt: bytes) -> bytes:
    """The challenger's commitment to a round's coins: hm96-sha512's ``b || c`` of SHA-512(`coin_message`) under ``salt``
    (`Flock.Firewall.Contract.commitBC`, the commit items' commitment); ``salt`` is its opening."""
    inst = hm96.SHA512
    return inst.commit_string(inst.default_key, hashlib.sha512(coin_message(stream, k, coins)).digest(), salt)


def refused(what: str) -> str:
    return f"FIREWALL-REFUSED: {what}"


def canonical(v) -> str:
    """``v`` as the firewall compares it with the pinned schedule: JSON with sorted keys and no spaces, so ``2`` and ``2.0``
    (or ``true`` and ``1``) differ."""
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def rep_rounds(m: int, k_log: int, claims: int) -> list[tuple[int, list[int], int, int]]:
    """One fast100 rep's rounds for a table of ``2^m`` packed elements, ``2^k_log`` a block, with ``claims`` claims (2 and the
    statement's extra ones): each round's coin count, the node counts of the tops it commits, its copies of level 0's top
    (committed once, at ``Commit``) and the number of field elements the prover absorbs before its squeeze (one message row
    when there are any: `message_rows`), in `rec_algebra.run`'s order of draws."""
    sch = fast100(m)
    lv, R = sch.levels, len(sch.levels) - 1
    cap = lambda level: [1 << lv[level].cap]
    # the zerocheck (the binding round, its point, round 1's ab and c, one round a variable), the lincheck, the ring switches
    out = [(6, [], 1, 0), (m - 13, [], 0, 0), (1, [], 0, 128)] + [(1, [], 0, 2)] * (m - 5) + [(1, [], 0, 0)]
    out += [(1, [], 0, 2)] * (k_log - 6) + [(1, [], 0, 64)] + [(7, [], 0, 128)] * claims + [(claims, [], 0, 0)]
    # Ligerito: level 0's cap again and its k0 challenges, then per level its OOD pair, queries and sumcheck challenges

    def level(i: int, split: int) -> list:
        q = lv[i].q
        return [(split, cap(i + 1), 0, 4), (1, [], 0, 5), (split, [], 0, 0), (1, [], 0, 5), (q, [], 0, 0),
                ((q - 1).bit_length(), [], 0, 0), (1, [], 0, 4)]

    split = sch.n - sch.k0 + 1
    out += [(sch.n, [], 1, 1), (1, [], 0, 1)] + [(2, [], 0, 4)] * sch.k0 + level(0, split)
    for i in range(R):
        if split < 4:
            raise ValueError(f"m = {m}: ligerito's split_dim < k")
        out += [(2, [], 0, 0)] + [(2, [], 0, 4)] * 3
        ext = split - 4
        if i + 1 < R:
            split = ext + 1
            out += level(i + 1, split)
        else:
            out += [(lv[R].q, [], 0, 4 + (2 << ext)), ((lv[R].q - 1).bit_length(), [], 0, 0), (1, [], 0, 0)]
    return out


def zp_round(m: int, k_log: int) -> int:
    """The index in `rep_rounds` of the round that absorbs z_partial (64 elements) and squeezes r′."""
    return m + k_log - 7


def holo_rounds(m: int, k_log: int, claims: int) -> list[tuple[int, list[int], int, int]]:
    """`rep_rounds` with hidden unit circuits (`rec_algebra.Shape.holo`): z_partial's round squeezes nothing (its frames end
    in a squeeze of no coins, `verity_flock.rec_session`; `rec_algebra.Coins` skips it), and a round of its own absorbs h (64
    elements) and squeezes r′ (`rec_algebra.h_round`)."""
    out, k = rep_rounds(m, k_log, claims), zp_round(m, k_log)
    if out[k] != (1, [], 0, 64):
        raise AssertionError(f"round {k} is not z_partial's")
    return out[:k] + [(0, [], 0, 64), (1, [], 0, 64)] + out[k + 1:]


def stacked_rounds(m: int) -> list[tuple[int, list[int], int, int]]:
    """The stream of the stacked table R (``2^m`` cells) that candidate 3's opening-only prover runs (``flock-circuit
    stacked``, its one rep): the binding round, which also absorbs the first claim's s_hat_v and squeezes its ring switch's
    coins, then `rep_rounds`' second ring switch, the gammas and Ligerito at fast100(m)."""
    if type(m) is not int or not 22 <= m <= 35:
        raise ValueError(f"the stacked table's m is 22 to 35, not {m}")
    return [(7, [], 1, 128)] + rep_rounds(m, 6, 2)[m + 1:]


def sparse_rounds(sh) -> list[tuple[int, list[int], int, int]]:
    """A rep's sparse phase at `verity_flock.rec_sparse`'s shape ``sh`` (`rec_sparse.prove`'s draws, each message one row): ζ_c,
    H2's rounds, v2's (no coins), H3's, u's (ζ_p), and with T > 1 vt's (ρ_t)."""
    from .rec_sparse import PRE_ROUNDS

    out = [(PRE_ROUNDS, [], 0, 0)] + [(1, [], 0, 2)] * PRE_ROUNDS + [(0, [], 0, 1)]
    out += [(1, [], 0, d) for d in sh.degrees] + [(sh.P_LOG, [], 0, sh.PP)]
    return out + ([(sh.T_LOG, [], 0, sh.T)] if sh.stacked else [])


def schedule_for(statement: dict, holo: bool = False, stacked: int | None = None, sparse=None) -> dict:
    """The schedule of a flock-circuit statement's inner session (one table, fast100's two reps, live OS coins) from the
    statement's public shape alone: ``dense_m``, ``k_log``, ``nbl`` and ``regions`` of ``flock-circuit statement``'s
    ``STATEMENT`` record (`statement_of`), the four numbers the schedule reveals anyway. Each region adds one extra claim a
    link point. ``holo``: hidden unit circuits' rounds (`holo_rounds`). ``sparse``: the bucket's `rec_sparse.Shape`; each
    rep's sparse stream (`sparse_rounds`), opened at its rep's end, follows both reps'. ``stacked``: the stacked table's
    ``m``; Hello names it after the circuit's table, Commit holds its root, and its stream (`stacked_rounds`) comes last."""
    m, k_log, nbl, regions = (statement.get(k) for k in ("dense_m", "k_log", "nbl", "regions"))
    if not all(type(v) is int and v >= 0 for v in (m, k_log, nbl, regions)):
        raise ValueError("a STATEMENT record names dense_m, k_log, nbl and regions")
    rep = (holo_rounds if holo else rep_rounds)(m, k_log, 2 + LINK_POINTS * regions)
    streams = {f"{TABLE}/rep{r}": rep for r in range(REPS)}
    at = {s: ("", 0) for s in streams}
    roots = {TABLE: 1 << fast100(m).levels[0].cap}
    if sparse is not None:
        for r in range(REPS):
            streams[f"{SPARSE}/rep{r}"], at[f"{SPARSE}/rep{r}"] = sparse_rounds(sparse), (f"{TABLE}/rep{r}", len(rep))
    if stacked is not None:
        streams[f"{STACKED}/rep0"], roots[STACKED] = stacked_rounds(stacked), 1 << fast100(stacked).levels[0].cap
        at[f"{STACKED}/rep0"] = ("", 0)
    return {"kind": KIND + "/schedule",
            "hello": {"digests": "sha512", "flavor": "rs", "link": {"m": PT_LOCAL + nbl, "points": LINK_POINTS}, "profile": "fast100",
                      "reps": REPS, "rounds": "bytes retained", "tables": list(roots)},
            "roots": roots, "link": {"y": LINK_POINTS},
            "streams": {s: {"parent": at[s][0], "at": at[s][1], "rounds": len(rs)} for s, rs in streams.items()},
            "rounds": [{"stream": s, "k": k, "n": n, "tops": tops, "level0": level0, "commits": int(e > 0)}
                       for s, rs in streams.items() for k, (n, tops, level0, e) in enumerate(rs)]}


def statement_of(text: str) -> dict:
    """The ``STATEMENT`` record in ``flock-circuit statement``'s output (its other lines ignored), or the text as one JSON
    object."""
    for line in text.splitlines():
        if line.startswith("STATEMENT\t"):
            return json.loads(line.split("\t", 1)[1])
    return json.loads(text)


def check_schedule(s) -> dict:
    """``s``, if it is a schedule a session can be held to (every field of the type `schedule_of` gives it); else ValueError."""
    nat = lambda v: type(v) is int and v >= 0
    try:
        h = s["hello"]
        ok = (s["kind"] == KIND + "/schedule" and isinstance(h, dict) and isinstance(h["link"], dict) and "sigma" not in h["link"]
              and nat(h["link"]["m"]) and nat(h["link"]["points"]) and nat(s["link"]["y"])
              and all(isinstance(t, str) and nat(v) for t, v in s["roots"].items())
              and all(isinstance(n, str) and isinstance(v["parent"], str) and nat(v["at"]) and nat(v["rounds"]) for n, v in s["streams"].items())
              and all(r["stream"] in s["streams"] and nat(r["k"]) and nat(r["n"]) and all(nat(c) for c in r["tops"]) and nat(r["level0"])
                      and nat(r["commits"]) for r in s["rounds"]))
    except (KeyError, TypeError, AttributeError):
        ok = False
    if not ok:
        raise ValueError(f"not a {KIND}/schedule")
    return s


def schedule_of(record: dict) -> dict:
    """What the pinned schedule fixes of a coin server's record (module docstring), to compare a session with the schedule it
    was held to: a salted top's node count is its number of ``b || c``."""
    return {"kind": KIND + "/schedule", "hello": record["hello"],
            "roots": {t: len(r) for t, r in sorted(record["link"]["roots"].items())}, "link": {"y": len(record["link"]["y"])},
            "streams": {s: {"parent": v["parent"], "at": v["at"], "rounds": v["rounds"]} for s, v in sorted(record["streams"].items())},
            "rounds": [{"stream": r["stream"], "k": r["k"], "n": r["n"], "tops": [len(t) for t in r["tops"]], "level0": r["level0"],
                        "commits": len(r["commits"])} for r in record["rounds"]]}


# --- the verifier's side ----------------------------------------------------------------------------------------------------------

class CoinServer:
    """The verifier's coin server: public data in, coins out, every exchange recorded (``record``). ``rand(n)`` returns n
    bytes (default the operating system's generator)."""

    def __init__(self, rand: Callable[[int], bytes] = os.urandom):
        self.rand = rand
        self.hello: dict | None = None
        self.link: dict | None = None
        self.streams: dict[str, dict] = {}
        self.rounds: list[dict] = []
        #: how the session ended: "finish", or "stop" (a refusal, or a session that ended without Finish)
        self.end: str | None = None
        #: the coins `commit_coins` drew before the session, by (stream, round), each with the salt that opens its commitment
        self.committed: dict[tuple[str, int], tuple[list[int], bytes]] | None = None
        #: `commit_coins`'s answer: the commitments in hex, per stream and round
        self.commitments: dict[str, list[str]] | None = None
        #: the opening sent with each round's coins, by (stream, round)
        self.openings: dict[tuple[str, int], bytes] = {}

    def _coins(self, n: int) -> list[int]:
        b = self.rand(16 * n)
        return [int.from_bytes(b[16 * i:16 * i + 8], "little") | (int.from_bytes(b[16 * i + 8:16 * i + 16], "little") << 64)
                for i in range(n)]

    def commit_coins(self, schedule: dict, fresh_salt: Callable[[], bytes] = hm96.SHA512.fresh_salt) -> dict[str, list[str]]:
        """Commit to every coin of the session before it starts (Goldreich–Kahan): draw the coins the pinned schedule needs,
        the link points then each round's in the schedule's order (so ``rand`` is read as the rounds would read it), and
        commit to each round's under a salt of its own (`coin_commitment`). Returns the commitments in hex, per stream and
        round in order: the verifier's public input's ``coins`` (`firewall_transcript.public`). Each round's coins then go
        out with their salt as the opening (``openings``), and a round whose coins were not committed is refused. A
        schedule's round that is not its stream's next (no session reaches it: the firewall holds its i-th round to the
        schedule's i-th) gets no commitment."""
        if self.committed is not None or self.hello is not None:
            raise ValueError("the challenger commits to its coins once, before Hello")
        link = check_schedule(schedule)["hello"]["link"]
        plan = [(LINK_COINS, 0, link["points"] * link["m"])] + [(r["stream"], r["k"], r["n"]) for r in schedule["rounds"]]
        self.committed, out = {}, {}
        for stream, k, n in plan:
            if len(out.setdefault(stream, [])) != k:
                continue
            coins, salt = self._coins(n), fresh_salt()
            self.committed[(stream, k)] = (coins, salt)
            out[stream].append(coin_commitment(stream, k, coins, salt).hex())
        self.commitments = out
        return out

    def commitment_record(self) -> dict:
        """``coins.json``: the commitments `commit_coins` made (``coins``) and the opening sent with each round's coins
        (``openings``, per stream in round order; a round the session never reached has none)."""
        openings: dict[str, list[str]] = {}
        for (s, _k), y in sorted(self.openings.items()):
            openings.setdefault(s, []).append(y.hex())
        return {"kind": COINS_KIND, "coins": self.commitments, "openings": openings}

    def _draw(self, stream: str, k: int, n: int) -> list[int]:
        """Round ``k`` of ``stream``'s ``n`` coins: the committed ones with their opening, or live ones if nothing was
        committed."""
        if self.committed is None:
            return self._coins(n)
        got = self.committed.get((stream, k))
        if got is None or len(got[0]) != n or (stream, k) in self.openings:
            raise ValueError(f"{stream} round {k}: {n} coins the challenger never committed to")
        self.openings[(stream, k)] = got[1]
        return list(got[0])

    def start(self, hello: dict) -> None:
        if self.hello is not None:
            raise ValueError("a second Hello")
        if "sigma" in hello.get("link", {}):
            raise ValueError("the coin server is never shown Σ")
        self.hello = hello

    def commit(self, roots: dict[str, list[bytes]]) -> list[int]:
        """Each table's level-0 top as its salted commitment (`commit_top`: a ``b || c`` a node) in; the link points out
        (``points × m`` coins)."""
        if self.hello is None or self.link is not None:
            raise ValueError("the link points are drawn once, after Hello")
        spec = self.hello["link"]
        pts = self._draw(LINK_COINS, 0, spec["points"] * spec["m"])
        self.link = {"roots": {t: [b.hex() for b in bcs] for t, bcs in sorted(roots.items())}, "points": [f128_hex(c) for c in pts],
                     "y": None}
        return pts

    def y(self, y: list[int]) -> None:
        if self.link is None or self.link["y"] is not None:
            raise ValueError("y follows the link points, once")
        self.link["y"] = [f128_hex(v) for v in y]

    def open(self, stream: str, parent: str, at: int) -> None:
        if stream in self.streams:
            raise ValueError(f"stream {stream} opened twice")
        self.streams[stream] = {"parent": parent, "at": at, "rounds": 0}

    def round(self, stream: str, tops: list[list[bytes]], commits: list[bytes], n: int, level0: int = 0) -> list[int]:
        """A round's salted tops (each a ``b || c`` a node) and ``c_k`` in, ``n`` coins out. ``level0``: how many copies of
        a level-0 top the round absorbed (committed at ``Commit``, not again)."""
        if self.link is None or self.link["y"] is None:
            raise ValueError("a coin before the link exchange")
        st = self.streams.get(stream)
        if st is None:
            raise ValueError(f"a round on stream {stream}, never opened")
        coins = self._draw(stream, st["rounds"], n)
        self.rounds.append({"stream": stream, "k": st["rounds"], "tops": [[b.hex() for b in t] for t in tops], "level0": level0,
                            "commits": [c.hex() for c in commits], "n": n, "coins": [f128_hex(c) for c in coins]})
        st["rounds"] += 1
        return coins

    def record(self) -> dict:
        """Everything the coin server saw and issued; its streams by name, as a schedule lists them, whatever order the
        prover opened them in. A stream is listed once it has a round; the firewall opens it here only at that round, so
        neither the record nor the coin server's calls show whether or when the prover sent its Open."""
        return {"kind": KIND, "tops": TOPS, "hello": self.hello, "link": self.link,
                "streams": {s: self.streams[s] for s in sorted(self.streams) if self.streams[s]["rounds"]},
                "rounds": self.rounds,
                "message_rows": sum(len(r["commits"]) for r in self.rounds), "row_words": ROW_WORDS, "end": self.end}


# --- the prover's side ------------------------------------------------------------------------------------------------------------

def _dg(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def _opening(cap: bytes, salts: list[bytes]) -> dict:
    """A salted top's opening as ``private.json`` keeps it: its nodes and their salts."""
    return {"nodes": [cap[i:i + CAP_NODE_BYTES].hex() for i in range(0, len(cap), CAP_NODE_BYTES)], "salts": [s.hex() for s in salts]}


class Proxy:
    """The live server's part on the prover's side: answers the unmodified prover, takes every coin from ``coins`` (shown only
    the schedule and salted hashes), holds the session to ``schedule`` (`check_schedule`), and keeps what the session record
    and V*'s witness need. ``commitments``, when the challenger committed to its coins before the session
    (`CoinServer.commit_coins`), are those commitments: every coin answer must then come with the salt that opens its
    round's commitment. Without them the coins are live, and a coin answer that comes with an opening stops the session
    (`Flock.Firewall.Contract`'s rule)."""

    def __init__(self, coins: CoinServer, fresh_salt: Callable[[], bytes] = hm96.SHA512.fresh_salt, *, schedule: dict,
                 commitments: dict[str, list[str]] | None = None):
        self.coins = coins
        self.fresh_salt = fresh_salt
        self.schedule = check_schedule(schedule)
        self.commitments = commitments
        self.hello: str | None = None
        self.hello_v: dict | None = None
        self.commit: tuple | None = None
        self.roots: dict[str, dict] = {}
        self.points: list[list[int]] = []
        self.y: bytes | None = None
        self.streams: dict[str, dict] = {}
        self.proofs: dict[tuple[str, int], bytes] = {}
        self.private: list[dict] = []
        self.stopped: str | None = None
        self.verdict: dict | None = None

    def _salted(self, cap: bytes) -> tuple[list[bytes], dict]:
        """A top under fresh salts, one a node: its ``b || c`` a node for the coin server, its opening for ``private.json``."""
        salts = [self.fresh_salt() for _ in top_rows(cap)]
        return commit_top(cap, salts), _opening(cap, salts)

    def _pinned(self, what: str, got, want) -> None:
        if canonical(got) != canonical(want):
            raise ValueError(refused(f"{what}: {canonical(got)} where the pinned schedule has {canonical(want)}"))

    def _opened(self, stream: str, k: int, coins: list[int]) -> None:
        """Round ``k`` of ``stream``'s coins open the challenger's commitment to them under the salt sent with them; live
        coins (no commitments) come with no opening."""
        y = self.coins.openings.get((stream, k))
        if not self.commitments:
            if y is not None:
                raise ValueError(refused(f"{stream} round {k}: an opening of coins the challenger never committed to"))
            return
        cs = self.commitments.get(stream, [])
        if k >= len(cs):
            raise ValueError(refused(f"{stream} round {k}: coins the challenger never committed to"))
        if y is None:
            raise ValueError(refused(f"{stream} round {k}: coins without the opening of the challenger's commitment"))
        if len(y) != hm96.SHA512.salt_bytes or coin_commitment(stream, k, coins, y).hex() != cs[k]:
            raise ValueError(refused(f"{stream} round {k}: coins that do not open the challenger's commitment"))

    def stop(self, why: str) -> None:
        """The session stops (its first reason is kept), and the coin server's record says so (``end``: ``stop``)."""
        self.stopped = self.stopped or why
        self.coins.end = "stop"

    def handle(self, req: bytes) -> bytes:
        """One request's answer. A refusal stops the session: every later request is refused for the same reason, and
        ``Finish`` answers a verdict that names it."""
        if self.stopped is not None and self.verdict is None and req[:1] != bytes([REQ_PING]):
            if req[:1] == bytes([REQ_FINISH]):
                self.verdict = {"accepted": False, "why": self.stopped}
                return resp_str(RESP_VERDICT, json.dumps(self.verdict))
            return resp_str(RESP_REFUSED, self.stopped)
        try:
            return self._handle(decode_req(req))
        except Exception as e:  # noqa: BLE001 - a request the firewall cannot handle stops its session, never the firewall
            why = f"rec proxy: {e}" if isinstance(e, ValueError) else \
                f"rec proxy: {refused(f'a request the firewall cannot read ({type(e).__name__}: {e})')}"
            self.stop(why)
            return resp_str(RESP_REFUSED, why)

    def _handle(self, r: tuple) -> bytes:
        t = r[0]
        if t == REQ_PING:
            return resp_ok()
        if self.verdict is not None:
            raise ValueError("session finished")
        if t == REQ_REGISTER:
            return resp_str(RESP_DRAW, "null")
        if t == REQ_HELLO:
            if self.hello is not None:
                raise ValueError("a second Hello")
            v = json.loads(r[1])
            if not isinstance(v, dict) or not isinstance(v.get("link"), dict) or not isinstance(v["link"].get("sigma"), str):
                raise ValueError(refused("Hello: not an object whose link names Σ"))
            if "coins" in v:
                raise ValueError("the inner session runs on live OS coins (FC_COINS=os): no seed, no coin tree")
            pub = {**v, "link": {k: x for k, x in v["link"].items() if k != "sigma"}}
            want = self.schedule["hello"]
            self._pinned("Hello", pub, want)
            # the coin server records the schedule's Hello, never the prover's spelling of it
            self.hello, self.hello_v = r[1], {**json.loads(canonical(want)), "link": {**want["link"], "sigma": v["link"]["sigma"]}}
            self.coins.start(json.loads(canonical(want)))
            return resp_ok()
        if t == REQ_COMMIT:
            # refused here, never by the coin server: a call it refuses would reach the auditor
            if self.hello is None or self.commit is not None:
                raise ValueError("Commit comes once, after Hello")
            _, root_f, roots, publics = r
            if publics and any(v for _, v in publics):
                raise ValueError("public words in an inner session")
            if len({name for name, _ in roots}) != len(roots):
                raise ValueError(refused("Commit names a table twice"))
            self._pinned("Commit's roots", {name: len(top_nodes(v)) for name, v in sorted(roots)}, self.schedule["roots"])
            self.commit = (root_f, roots, publics)
            bcs = {}
            for t, cap in roots:
                bcs[t], self.roots[t] = self._salted(cap)
            pts = self.coins.commit(bcs)
            self._opened(LINK_COINS, 0, pts)
            m = self.hello_v["link"]["m"]
            self.points = [pts[i:i + m] for i in range(0, len(pts), m)]
            return resp_coins(pts)
        if t == REQ_LINK:
            if self.commit is None or self.y is not None:
                raise ValueError("Link comes once, after Commit")
            if len(r[1]) != 16 * self.schedule["link"]["y"]:
                raise ValueError(refused(f"Link's y: {len(r[1])} bytes where the pinned schedule has {self.schedule['link']['y']} elements of 16"))
            if any(r[1]):
                raise ValueError(refused("Link's y is not zero"))
            self.y = r[1]
            self.coins.y([int.from_bytes(self.y[i:i + 8], "little") | (int.from_bytes(self.y[i + 8:i + 16], "little") << 64)
                          for i in range(0, len(self.y), 16)])
            return resp_ok()
        if t == REQ_OPEN:
            _, stream, parent, label, at = r
            want = self.schedule["streams"].get(stream)
            self._pinned(f"Open {stream}", {"parent": parent, "at": at}, want and {"parent": want["parent"], "at": want["at"]})
            if stream in self.streams:
                raise ValueError(f"stream {stream} opened twice")
            # held until the stream's first round: where the prover sent its Open must not reach the coin server
            self.streams[stream] = {"parent": parent, "label": label, "at": at, "rounds": [], "cap": None}
            return resp_ok()
        if t == REQ_ROUND:
            _, stream, content, n = r
            st = self.streams.get(stream)
            if st is None:
                raise ValueError(f"a round on stream {stream}, never opened")
            k = len(st["rounds"])
            s = split_round(content, k == 0 and not st["parent"])
            if s.n != n:
                raise ValueError(f"the round squeezes {s.n} coins, the prover asked for {n}")
            level0 = {bytes.fromhex("".join(o["nodes"])) for o in self.roots.values()}
            if k == 0 and not st["parent"]:
                root_b = self.roots.get(stream.rsplit("/", 1)[0])
                if root_b is None or s.caps[0].hex() != "".join(root_b["nodes"]):
                    raise ValueError(f"{stream}: the binding round's cap is not the committed root_B")
                st["cap"] = s.caps[0]
            tops = [self._salted(c) for c in s.caps if c not in level0]
            rows = message_rows(s.free)
            i, sched = len(self.coins.rounds), self.schedule["rounds"]
            self._pinned(f"round {i}", {"stream": stream, "k": k, "n": n, "tops": [len(top_nodes(c)) for c in s.caps if c not in level0],
                                        "level0": sum(c in level0 for c in s.caps), "commits": len(rows)},
                         sched[i] if i < len(sched) else None)
            salts = [self.fresh_salt() for _ in rows]
            bcs = commit_rows(rows, salts)
            if k == 0:
                if self.y is None:
                    raise ValueError("a coin before the link exchange")
                self.coins.open(stream, st["parent"], st["at"])
            coins = self.coins.round(stream, [t for t, _ in tops], bcs, n, level0=sum(c in level0 for c in s.caps))
            st["rounds"].append({"content": content, "n": n, "coins": coins})
            self.private.append({"stream": stream, "k": k, "tops": [o for _, o in tops], "elements": [f128_hex(v) for v in s.free],
                                 "rows": rows, "salts": [x.hex() for x in salts], "nonces": [x.hex() for x in s.nonces]})
            self._opened(stream, k, coins)
            return resp_coins(coins)
        if t == REQ_PROOF:
            _, table, rep, b = r
            self.proofs[(table, rep)] = b
            return resp_ok()
        if t == REQ_FINISH:
            if self.commit is None or self.y is None:
                raise ValueError("Finish before the link exchange")
            # the firewall's own counts: an early Finish must not make it read the coin server
            ran = {s: len(v["rounds"]) for s, v in sorted(self.streams.items()) if v["rounds"]}
            self._pinned("Finish's rounds", sum(ran.values()), len(self.schedule["rounds"]))
            self._pinned("Finish's streams", ran, {s: v["rounds"] for s, v in sorted(self.schedule["streams"].items())})
            self.coins.end = "finish"
            self.verdict = {"accepted": None, "why": f"{KIND}: the inner proof is V*'s witness; nothing verified here"}
            return resp_str(RESP_VERDICT, json.dumps(self.verdict))
        raise ValueError(f"request tag {t} has no place in an inner session")

    def session_record(self) -> dict:
        """The ``flock-live-session/v1`` record the live server would have written (``Server::record_json``): what
        ``Server::from_record`` replays."""
        root_f, roots, publics = self.commit
        link = {"mode": "exchange", "sigma": self.hello_v["link"]["sigma"], "points": [[f128_hex(c) for c in p] for p in self.points],
                "points_g": None, "y": [f128_hex(int.from_bytes(self.y[i:i + 8], "little") | (int.from_bytes(self.y[i + 8:i + 16], "little") << 64))
                                        for i in range(0, len(self.y), 16)],
                "root_f": root_f.hex(), "root_b": {t: v.hex() for t, v in roots}, "publics": {t: v.hex() for t, v in publics},
                "root_f_sha512": _dg(root_f), "root_b_sha512": {t: _dg(v) for t, v in roots},
                "publics_sha512": {t: _dg(v) for t, v in publics}}
        lb = root_f + b"".join(v for _, v in sorted(roots)) + b"".join(f128_bytes(c) for p in self.points for c in p) + self.y
        streams = []
        for name in sorted(self.streams):
            st = self.streams[name]
            streams.append({"stream": name, "parent": st["parent"], "label": st["label"].decode("utf-8", "replace"), "at": st["at"],
                            "rounds": [{"k": k, "g": None, "n": rd["n"], "msg_len": len(rd["content"]),
                                        "coins": [f128_hex(c) for c in rd["coins"]], "msg_sha512": _dg(rd["content"]),
                                        "msg": rd["content"].hex()} for k, rd in enumerate(st["rounds"])],
                            "root_sha512": _dg(st["cap"]) if not st["parent"] else None})
        h = self.hello_v
        return {"kind": SESSION_KIND, "hello": self.hello, "link": link, "link_mode": "exchange",
                "config": {"profile": h["profile"], "reps": h["reps"], "flavor": h["flavor"], "require_link": True, "prime_live": False,
                           "tables": [{"name": t, "binding": "R1cs (rec proxy: the statement digest is the prover's)"} for t in h["tables"]]},
                "unit_draw": None, "aborted": None, "verdict": self.verdict, "streams": streams, "prime": None, "coin_tree": None,
                "coin_seed": None, "link_sha512": _dg(lb),
                "proof_sha512": [{"stream": f"{t}/rep{r}", "bytes": len(b), "sha512": _dg(b)} for (t, r), b in sorted(self.proofs.items())]}

    def write(self, out: Path) -> Path:
        """The session directory (``session.json``, ``<table>.rep<r>.bin``), ``public.json``, ``private.json`` and, when the
        challenger committed to its coins, its commitments and openings, ``coins.json``."""
        out = Path(out)
        sess = out / "session"
        sess.mkdir(parents=True, exist_ok=True)
        (sess / "session.json").write_text(json.dumps(self.session_record(), indent=1))
        for (t, r), b in self.proofs.items():
            (sess / f"{t}.rep{r}.bin").write_bytes(b)
        (out / "public.json").write_text(json.dumps(self.coins.record(), indent=1))
        if self.commitments:
            (out / "coins.json").write_text(json.dumps(self.coins.commitment_record(), indent=1))
        (out / "private.json").write_text(json.dumps({"kind": KIND + "/private", "tops": TOPS, "roots": self.roots, "rounds": self.private}))
        return sess


def serve(listen: str, out: Path, rand: Callable[[int], bytes] = os.urandom, ready: Callable[[str], None] | None = None,
          sessions: int = 1, done: Callable[[dict], None] | None = None, *, schedule: dict, firewall=None,
          parts: list[str] | None = None, stream: str = "rec", budget: int | None = None, release: dict | None = None,
          now_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000) -> list[dict]:
    """Accept connections on ``listen`` until ``sessions`` of them have each carried a session, one connection each (several:
    a prover's ``--warm`` and ``--runs`` sessions), write each into ``out``, or ``out/s<k>`` when there are several, and
    return their summaries, each also passed to ``done`` as it ends. A session that finishes is written whole
    (`Proxy.write`); one that stops (a refusal, or a connection that ends without ``Finish``: closed, reset, or cut inside a
    frame) has only the coin server's record, ``public.json``, and counts as one of the ``sessions``. Each session's coins
    are live: the challenger commits to none. ``ready`` is called with the bound address. Every session is held to ``schedule`` (`schedule_for`'s, from public data alone), written to
    ``out/schedule.json`` before the first is served. With ``firewall`` (a `verity_flock.firewall.Firewall`), each session
    opens a recursive session on ``stream`` with its V* statements in their order (``parts``, public, from K: L0–L6 and the
    algebra at K = 4096), its id in the session directory's ``firewall.json``: a stop is that recursive session's stop, so
    the firewall serves none of its outer sessions, and a finished session lets them be served, in that order, each once.
    ``budget`` is the stream's failure budget (the challenger's stream's ``budget``); once the stream has stopped at it, a
    connection is closed before anything is served and ``serve`` returns. ``release`` (``{"windows_ms": [one a
    statement]}``, public) is each recursive session's release schedule, its ``start_ms`` the time (``now_ms``) its
    connection was accepted, so that no release time depends on how long the inner session took. A recursive session the
    firewall cannot open as given (other than `Refused`: the caller's fault, such as a schedule for other statements)
    closes the connection unserved, and ``serve`` raises."""
    host, port = listen.rsplit(":", 1)
    res: list[dict] = []
    if firewall is not None and parts is None:
        raise ValueError("serve with a firewall opens each recursive session with its statements in their order (parts)")
    check_schedule(schedule)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "schedule.json").write_text(json.dumps(schedule, indent=1))
    with socket.create_server((host, int(port))) as srv:
        h, p = srv.getsockname()[:2]
        addr = f"{h}:{p}"
        if ready:
            ready(addr)
        while len(res) < sessions:
            conn, _ = srv.accept()
            sid = None
            if firewall is not None:
                from .firewall import Refused
                try:
                    sid = firewall.open(stream, parts, budget, None if release is None else {"start_ms": now_ms(), **release})
                except Refused as e:
                    conn.close()
                    print("STOPPED\t" + json.dumps({"session": len(res), "refused": str(e)}), flush=True)
                    res.append({"session": None, "refused": str(e), "rounds": 0, "firewall": None})
                    if done:
                        done(res[-1])
                    break
                except BaseException:
                    conn.close()
                    raise
            proxy = Proxy(CoinServer(rand), schedule=schedule)
            try:
                with conn, conn.makefile("rb") as rf, conn.makefile("wb") as wf:
                    while (req := read_frame(rf)) is not None:
                        was = proxy.stopped
                        write_frame(wf, proxy.handle(req))
                        if was is None and proxy.stopped is not None:
                            print("STOPPED\t" + json.dumps({"session": len(res), "refused": proxy.stopped}), flush=True)
                        if proxy.verdict is not None:
                            break
            except (OSError, ValueError):
                pass
            if proxy.verdict is None and proxy.stopped is None:
                proxy.stop("rec proxy: the session ended without Finish")
                print("STOPPED\t" + json.dumps({"session": len(res), "refused": proxy.stopped}), flush=True)
            d = Path(out) / f"s{len(res)}" if sessions > 1 else Path(out)
            if proxy.stopped is not None:
                d.mkdir(parents=True, exist_ok=True)
                (d / "public.json").write_text(json.dumps(proxy.coins.record(), indent=1))
                res.append({"session": None, "refused": proxy.stopped, "rounds": len(proxy.coins.rounds)})
                if firewall is not None:
                    firewall.stop(sid, "inner", proxy.stopped)
            else:
                sess = proxy.write(d)
                rec = proxy.coins.record()
                res.append({"session": str(sess), "rounds": len(rec["rounds"]), "message_rows": rec["message_rows"],
                            "coins": sum(r["n"] for r in rec["rounds"]), "tops": sum(len(r["tops"]) for r in rec["rounds"]),
                            "top_nodes": sum(len(t) for r in rec["rounds"] for t in r["tops"]),
                            "proof_bytes": {f"{t}/rep{r}": len(b) for (t, r), b in proxy.proofs.items()}})
                if firewall is not None:
                    firewall.finish(sid)
            if firewall is not None:
                (d / "firewall.json").write_text(json.dumps({"kind": "firewall/v0/session", "session": sid}))
                res[-1]["firewall"] = sid
            if done:
                done(res[-1])
    return res


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:0")
    ap.add_argument("--out")
    ap.add_argument("--sessions", type=int, default=1, help="sessions to serve, one connection each (several: out/s<k>/)")
    ap.add_argument("--schedule", help="the schedule every session is held to (required to serve: --schedule-for's)")
    ap.add_argument("--schedule-for", metavar="STATEMENT",
                    help="print the schedule of a statement (flock-circuit statement's output, or its STATEMENT record) and exit")
    ap.add_argument("--holo", action="store_true", help="--schedule-for: hidden unit circuits' rounds (h after z_partial)")
    ap.add_argument("--stacked", type=int, metavar="M_R", help="--schedule-for: the stacked table R's stream, R of 2^M_R cells")
    ap.add_argument("--sparse", metavar="NU,ELL,S_LOG,T_LOG", help="--schedule-for: each rep's sparse stream at this RecSparse shape")
    ap.add_argument("--schedule-of", help="print a coin server record's (public.json) schedule and exit")
    ap.add_argument("--firewall", metavar="DIR", help="the firewall's state: each session opens a recursive session on it")
    ap.add_argument("--stream", default="rec", help="the stream the recursive sessions are on (--firewall)")
    ap.add_argument("--parts", metavar="L0,L1,...,alg",
                    help="each recursive session's V* statements in their order, the only order served (required with --firewall)")
    ap.add_argument("--budget", type=int, help="the stream's failure budget, the challenger's stream's (--firewall)")
    ap.add_argument("--release-windows-ms", metavar="W0,W1,...",
                    help="the release schedule's public window a statement, in ms: statement j leaves at the connection's time "
                         "plus W0 + ... + Wj (--firewall)")
    a = ap.parse_args(argv)
    if a.schedule_for:
        sparse = None
        if a.sparse:
            from .rec_sparse import Shape
            sparse = Shape(*(int(v) for v in a.sparse.split(",")))
        print(json.dumps(schedule_for(statement_of(Path(a.schedule_for).read_text()), a.holo, a.stacked, sparse), indent=1))
        return 0
    if a.schedule_of:
        print(json.dumps(schedule_of(json.loads(Path(a.schedule_of).read_text())), indent=1))
        return 0
    if not a.out or not a.schedule:
        ap.error("--out and --schedule are required (rec_live --schedule-for STATEMENT derives the schedule)")
    release = None if a.release_windows_ms is None else {"windows_ms": [int(w) for w in a.release_windows_ms.split(",")]}
    if (a.parts is not None or a.budget is not None or release is not None) and not a.firewall:
        ap.error("--parts, --budget and --release-windows-ms need --firewall")
    if a.firewall and a.parts is None:
        ap.error("--firewall needs --parts: each recursive session's V* statements in their order (L0–L6,alg at K = 4096)")
    fw = None
    if a.firewall:
        from .firewall import Firewall
        fw = Firewall(Path(a.firewall))
    serve(a.listen, Path(a.out), ready=lambda addr: print(f"SERVING {addr}", flush=True), sessions=a.sessions,
          done=lambda res: print("REC_INNER\t" + json.dumps(res), flush=True), schedule=json.loads(Path(a.schedule).read_text()),
          firewall=fw, parts=None if a.parts is None else a.parts.split(","), stream=a.stream, budget=a.budget, release=release)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
