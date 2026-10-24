"""The I/O tie (`PROTOCOL.md`, "The I/O tie"), a prototype on a fake upstream: the reference of one egress link-window's
tie unit and the Boolean parts it is built from. Nothing here is in a Spec or proved, and the shim and N3's template
still carry text: this module's id event is the design's, not the wire's yet.

For each of K live coins u the unit draws frame k = ⌊u·N / 2^64⌋ of the window's N real frames (N = Σ n_t of the
head it opens at leaf 2 + direction of N2's commitment, `head_check`'s opening), opens the frame under the head's
frames root (`NetworkFrameRead`), holds it to the canonical event of one token id (`IoTieEvent`), and opens the token
port's row that holds request ⌊k / F⌋'s output k mod F (`NetworkHidingRead`), whose column must be that id
(`IoTieDraw`). With F, C and 32 powers of two, F ≥ 32, the position is wiring: k's low 5 bits are the column, its
next log2(F) − 5 the row within the request, and the rest the request.

    NetworkFrameRead{B, D}(frame: Array<B, Byte>, pos: Array<D>, path: Array<D, Digest>, root: Digest) -> Value<1>
    IoTieEvent{B}(frame: Array<B, Byte>, id: Array<32>) -> Value<1>
    IoTieColumn(row: Array<1024>, column: Array<5>) -> Array<32>
    IoTieDraw{B, D, FOUT, C}(frames_root: Digest, k: Array<D>, frame: Array<B, Byte>, frame_path: Array<D, Digest>,
        port: Digest, port_domain: Digest, row: Array<128, Byte>, salt: Salt, row_path: Array<DT, Digest>) -> Value<1>
        DT = log2(C) + D − log2(FOUT), FOUT the port's F

**The token port** (`TokenPort`) is one value per link-window, committed before the window's coins: request g's F
sampled ids, then its P prompt ids, 32 u32be ids a row, request g's slot s at row g·C + ⌊s / 32⌋ and column s mod 32,
C ≥ (F + P) / 32 a power of two. Each row is an `hm96-sha512/row/v2` leaf under its own salt in one
`frame-v3-sha512` tree of 2^DT leaves; rows no request fills are zero rows, salted like the rest.

**Not in the skeleton:** the head's opening (`head_check.NetworkHidingRead`, the same leaf N2 reads), the draw's
k = ⌊u·N / 2^64⌋ and k < N with N the head's counts' sum (`IoTiePosition`), the K draws' AND, and the ingress side.
`reference` checks all of them in the clear.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ZERO, and_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.frame_v3 import root_ref
from verity.ml.commitments.gates.lms import LmsEqual, Sha512Compress
from verity.ml.commitments.gates.sha512 import midstate, pad_tail
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA, row_leaf
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier.active import CANONICAL_STREAM, DATA, EVENT_END, EVENT_HEAD, HEADER
from verity.core.protocols.certifier.certificate import FRAME_TAG, NODE_TAG, verify_frame
from verity.core.protocols.certifier.grid import GridParams

BIT, BYTE, WORD, DIGEST, SALT = HC.BIT, HC.BYTE, HC.WORD, HC.DIGEST, HC.SALT
ID_DIGITS = 10
ID_EVENT_HEAD = EVENT_HEAD[:EVENT_HEAD.index(b'"delta"')] + b'"token_id":'
ID_EVENT_TAIL = b',"finish_reason":null}]}' + EVENT_END
ID_EVENT_BYTES = len(ID_EVENT_HEAD) + ID_DIGITS + len(ID_EVENT_TAIL)
#: the id's first character in a frame: the frame's header, then the event's head
ID_AT = HEADER.size + len(ID_EVENT_HEAD)
ROW_IDS = 32
ROW_BYTES = 4 * ROW_IDS


# --- the id event ----------------------------------------------------------------------------------------------------

def id_event(token_id: int) -> bytes:
    """The canonical event of one token id: `ID_EVENT_HEAD`, the id right-aligned in 10 decimal characters,
    `ID_EVENT_TAIL`; `ID_EVENT_BYTES` long whatever the id."""
    if type(token_id) is not int or not 0 <= token_id < 1 << 32:
        raise ValueError(f"a token id is a u32, not {token_id!r}")
    return ID_EVENT_HEAD + b"%10d" % token_id + ID_EVENT_TAIL


def id_frame(token_id: int, frame_bytes: int) -> bytes:
    """The link frame of a token id's event: a `DATA` header on `CANONICAL_STREAM`, the event, zero fill."""
    if frame_bytes < ID_AT + ID_DIGITS + len(ID_EVENT_TAIL):
        raise ValueError(f"a {frame_bytes}-byte frame has no room for an id event")
    return (HEADER.pack(DATA, CANONICAL_STREAM, ID_EVENT_BYTES) + id_event(token_id)).ljust(frame_bytes, b"\0")


def frame_id(frame: bytes) -> int | None:
    """The token id whose canonical frame this is, or None: `IoTieEvent`'s reference, the one parse of a frame."""
    digits = frame[ID_AT:ID_AT + ID_DIGITS]
    if len(digits) != ID_DIGITS or not digits.lstrip(b" ").isdigit():
        return None
    token_id = int(digits)
    return token_id if token_id < 1 << 32 and frame == id_frame(token_id, len(frame)) else None


# --- the position and the port ---------------------------------------------------------------------------------------

def draw(coin: int, n: int) -> int:
    """The frame a 64-bit coin draws of n: ⌊coin·n / 2^64⌋."""
    if not 0 <= coin < 1 << 64 or n < 1:
        raise ValueError("a draw takes a 64-bit coin and at least one frame")
    return coin * n >> 64


def _log2(x: int, name: str) -> int:
    if type(x) is not int or x < 1 or x & (x - 1):
        raise ValueError(f"{name} is a power of two, not {x!r}")
    return x.bit_length() - 1


@dataclass(frozen=True)
class TokenPort:
    """The token port's shape for one link-window of a D-deep frame tree: F outputs and P prompt ids a request,
    C rows a request, 2^DT rows, DT = log2(C) + D − log2(F), so every request a window's frames can hold has its rows."""

    F: int
    P: int
    C: int
    D: int

    def __post_init__(self) -> None:
        if _log2(self.F, "F") < 5 or self.C * ROW_IDS < self.F + self.P or self.P < 0:
            raise ValueError(f"a port has F >= 32 and C·32 >= F + P, not F={self.F}, P={self.P}, C={self.C}")
        _log2(self.C, "C")
        if self.DT < 1:
            raise ValueError("a port has at least two rows")

    @property
    def DT(self) -> int:
        return _log2(self.C, "C") + self.D - _log2(self.F, "F")

    @property
    def leaves(self) -> int:
        return 1 << self.DT

    def slot(self, g: int, s: int) -> tuple[int, int]:
        """(row, column) of request g's slot s: output j at s = j, prompt id i at s = F + i."""
        return g * self.C + s // ROW_IDS, s % ROW_IDS

    def at(self, k: int) -> tuple[int, int]:
        """(row, column) of the window's egress frame k: request ⌊k / F⌋'s output k mod F."""
        return self.slot(k // self.F, k % self.F)

    def rows(self, requests: Sequence[tuple[Sequence[int], Sequence[int]]]) -> list[bytes]:
        """The port's rows for the window's requests in canonical order, each (outputs, prompt): F outputs and at
        most P prompt ids, slots past the prompt 2^32 − 1."""
        if len(requests) * self.C > self.leaves:
            raise ValueError(f"{len(requests)} requests don't fit {self.leaves} rows")
        ids = [0] * (self.leaves * ROW_IDS)
        for g, (outputs, prompt) in enumerate(requests):
            if len(outputs) != self.F or len(prompt) > self.P:
                raise ValueError(f"a request has {self.F} outputs and at most {self.P} prompt ids")
            for s, x in enumerate([*outputs, *prompt, *[0xFFFF_FFFF] * (self.P - len(prompt))]):
                row, col = self.slot(g, s)
                ids[row * ROW_IDS + col] = x
        return [struct.pack(f">{ROW_IDS}I", *ids[r * ROW_IDS:(r + 1) * ROW_IDS]) for r in range(self.leaves)]

    def domain(self, binding: bytes) -> CommitmentDomain:
        return CommitmentDomain(binding, -1, RangeIndexedDomain(0, self.leaves), hash="sha512")


def port_levels(domain: CommitmentDomain, rows: Sequence[bytes], salts: Sequence[bytes]) -> tuple[tuple[bytes, ...], ...]:
    """The port's tree: each row's `hm96-sha512/row/v2` leaf under its salt, as N2 commits its heads."""
    return tree_levels(domain, [row_leaf(domain, i, HC.hiding_value(r, s), SCHEMA)
                                for i, (r, s) in enumerate(zip(rows, salts, strict=True))])


def path(levels: Sequence[Sequence[bytes]], i: int) -> tuple[bytes, ...]:
    """Leaf i's siblings in a tree's levels, leaf level first."""
    return tuple(levels[lv][(i >> lv) ^ 1] for lv in range(len(levels) - 1))


# --- the reference ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Draw:
    """One draw's witness: the frame and its path in the frames tree, the port row, its salt and path."""

    frame: bytes
    frame_path: tuple[bytes, ...]
    row: bytes
    salt: bytes
    row_path: tuple[bytes, ...]


@dataclass(frozen=True)
class Instance:
    """One egress link-window's tie: the head's opening as N2's (`head_check.Instance`'s fields), the grid and the
    port's shape (statics), the port's root and domain id and the coins (public), and one `Draw` per coin."""

    record: bytes
    record_domain: bytes
    head: bytes
    salt: bytes
    head_path: tuple[bytes, bytes]
    link_bytes: int
    params: GridParams
    port: TokenPort
    port_root: bytes
    port_domain: bytes
    coins: tuple[int, ...]
    draws: tuple[Draw, ...]


def head_counts(head: bytes, L: int, T: int) -> tuple[tuple[int, ...], bytes]:
    """A head's T counts and its frames root (`RecordHead.encode`'s layout, `head_check.Shape`)."""
    sh = HC.Shape(L, T)
    if len(head) != sh.HL:
        raise ValueError(f"the head is {len(head)} bytes, the shape's {sh.HL}")
    return struct.unpack(f">{T}I", head[sh.CNT:sh.ROOT]), head[sh.ROOT:sh.HL]


def reference_draws(inst: Instance) -> list[bool]:
    """Each draw's outcome in the clear: frame k = `draw`(coin, N) opens under the head's frames root, it is the
    canonical event of an id, and the port row of request ⌊k / F⌋'s output k mod F opens under the port's root with
    that id in its column. Raises on a head that doesn't open at the egress head's leaf of N2's commitment."""
    if root_ref(inst.record_domain, SCHEMA, HC.hiding_value(inst.head, inst.salt), HC.HEADS, inst.head_path) != inst.record:
        raise ValueError("the head doesn't open at leaf 2 of N2's commitment")
    counts, frames_root = head_counts(inst.head, inst.link_bytes, inst.params.buckets)
    n = sum(counts)
    if len(inst.draws) != len(inst.coins):
        raise ValueError("one draw a coin")
    out = []
    for coin, d in zip(inst.coins, inst.draws):
        if n == 0:
            out.append(True)
            continue
        k = draw(coin, n)
        row_at, col = inst.port.at(k)
        opens = verify_frame(frames_root, inst.params, k, d.frame, d.frame_path)
        token = frame_id(d.frame)
        row_opens = (len(d.row_path) == inst.port.DT and len(d.row) == ROW_BYTES
                     and root_ref(inst.port_domain, SCHEMA, HC.hiding_value(d.row, d.salt), row_at, d.row_path)
                     == inst.port_root)
        committed = int.from_bytes(d.row[4 * col:4 * col + 4], "big") if row_opens else None
        out.append(bool(opens and row_opens and token is not None and token == committed))
    return out


def reference(inst: Instance) -> bool:
    """The unit's ok bit: every draw's."""
    return all(reference_draws(inst))


# --- the circuit -----------------------------------------------------------------------------------------------------

def _frame_block(frame_bytes: int) -> bytes:
    return (FRAME_TAG + frame_bytes.to_bytes(4, "big")).ljust(128, b"\0")


def _compress(B, state: Coll, data: Sequence[Sequence]) -> Coll:
    """SHA-512's compressions over bytes (8 bits each) whole blocks long, from a chaining value."""
    words = HC._words(data)
    for k in range(0, len(words), 16):
        state = B.call(Sha512Compress, state, HC._coll(Array(16, WORD), [x for w in words[k:k + 16] for x in w]))
    return state


def _cv(K, prefix: bytes) -> Coll:
    return HC._coll(DIGEST, [x for v in midstate(prefix) for x in K.bits(v, 64)])


def _sig_swap(S):
    return (("d", BIT), ("cur", DIGEST), ("sib", DIGEST)), Array(2, DIGEST)


def _build_swap(C, S, *e):
    d, cur, sib = e[0][0], e[1:9], e[9:17]
    t = [[C.AND(d, x ^ y) for x, y in zip(cw, sw)] for cw, sw in zip(cur, sib)]
    return ([[x ^ u for x, u in zip(cw, tw)] for cw, tw in zip(cur, t)]
            + [[y ^ u for y, u in zip(sw, tw)] for sw, tw in zip(sib, t)])


NetworkFrameSwap = HC._part(
    "NetworkFrameSwap", (), _sig_swap, _build_swap,
    "(cur, sib), swapped when d: a frames-tree level's children, cur the right one iff d; 512 ANDs")


def _check_frame(S) -> None:
    if type(S.B) is not int or S.B < 1 or type(S.D) is not int or not 1 <= S.D <= 64:
        raise ValueError(f"a frame read has B >= 1 and 1 <= D <= 64, not B={S.B!r}, D={S.D!r}")


def _sig_frame(S):
    _check_frame(S)
    return (("frame", Array(S.B, BYTE)), ("pos", Array(S.D, BIT)), ("path", Array(S.D, DIGEST)), ("root", DIGEST)), BIT


def _body_frame(B, S, frame, pos, path_, root):
    _check_frame(S)
    K = HC._Consts(B)
    fb = HC._refs(frame)
    cur = _compress(B, _cv(K, _frame_block(S.B)),
                    [fb[8 * i:8 * i + 8] for i in range(S.B)] + [K.bits(b, 8) for b in pad_tail(128 + S.B)])
    node, tail = _cv(K, NODE_TAG.ljust(128, b"\0")), [K.bits(b, 8) for b in pad_tail(256)]
    for k in range(S.D):
        pair = B.call(NetworkFrameSwap, pos[k], cur, path_[k])
        cur = _compress(B, B.call(Sha512Compress, node, pair.reshape(Array(16, WORD))), tail)
    return B.call(bind(LmsEqual, N=64), cur, root)


NetworkFrameRead = CompositeDefinition(
    "NetworkFrameRead", 1, ("B", "D"), _sig_frame, _body_frame, conformance="exact-model-tested",
    doc="1 iff the B-byte frame is leaf pos of the depth-D frames tree of root (`certificate.verify_frame`): its leaf "
        "SHA-512(block(frame tag ‖ u32be B) ‖ frame), each level SHA-512(block(node tag) ‖ l ‖ r) with (l, r) swapped "
        "by pos's bit, leaf level first")


def _sig_event(S):
    if type(S.B) is not int or S.B < ID_AT + ID_DIGITS + len(ID_EVENT_TAIL):
        raise ValueError(f"a {S.B!r}-byte frame has no room for an id event")
    return (("frame", Array(S.B, BYTE)), ("id", Array(32, BIT))), BIT


def _add(C, x: Sequence, y: Sequence) -> list:
    """x + y, both as wide as x, the carry out dropped: a ripple adder, 2 ANDs a bit."""
    out, c = [], ZERO
    for a, b in zip(x, y, strict=True):
        t = a ^ b
        out.append(t ^ c)
        c = C.AND(a, b) ^ C.AND(c, t)
    return out


def _build_event(C, S, *e):
    frame, token = e[:S.B], e[S.B]
    template = id_frame(0, S.B)
    oks = [HC._same(frame[i][b], template[i] >> b & 1)
           for i in range(S.B) if not ID_AT <= i < ID_AT + ID_DIGITS for b in range(8)]
    chars = frame[ID_AT:ID_AT + ID_DIGITS]
    space = [and_reduce(C, [HC._same(c[b], 0x20 >> b & 1) for b in range(8)]) for c in chars]
    # a digit is 0x30 ... 0x39: high nibble 0011, low nibble at most 9 (not b3 with b2 or b1)
    digit = [C.AND(and_reduce(C, [c[7].inv(), c[6].inv(), c[5], c[4]]), C.AND(c[3], C.AND(c[2].inv(), c[1].inv()).inv()).inv())
             for c in chars]
    zero = [and_reduce(C, [c[b].inv() for b in range(4)]) for c in chars]
    for i in range(ID_DIGITS):
        oks.append(space[i] ^ digit[i])
        if i:
            oks.append(C.AND(digit[i - 1], space[i]).inv())
        if i < ID_DIGITS - 1:
            lead = digit[i] if i == 0 else C.AND(digit[i], space[i - 1])
            oks.append(C.AND(lead, zero[i]).inv())
    oks.append(digit[-1])
    width = 34
    v = [ZERO] * width
    for c in chars:
        # v·10 + the character's low nibble (a space's is 0)
        v = _add(C, _add(C, [ZERO] * 3 + v[:-3], [ZERO] + v[:-1]), [*c[:4], *[ZERO] * (width - 4)])
    oks += [HC._same(v[j], token[j]) for j in range(32)] + [v[32].inv(), v[33].inv()]
    return [and_reduce(C, oks)]


IoTieEvent = HC._part(
    "IoTieEvent", ("B",), _sig_event, _build_event,
    "1 iff the B-byte frame is `id_frame(id, B)`: its other bytes the template's, and its 10 characters spaces then "
    "digits, the first digit nonzero unless it is the last, reading id (u32, least significant bit first)")


def _sig_column(S):
    return (("row", Array(8 * ROW_BYTES, BIT)), ("column", Array(5, BIT))), Array(32, BIT)


def _build_column(C, S, row, column):
    # id c is u32be at bytes 4c ... 4c + 3: its bit j is bit j % 8 of byte 4c + 3 - j // 8
    ids = [[row[8 * (4 * c + 3 - j // 8) + j % 8] for j in range(32)] for c in range(ROW_IDS)]
    for b in range(5):
        ids = [[x ^ C.AND(column[b], x ^ y) for x, y in zip(ids[2 * i], ids[2 * i + 1])] for i in range(len(ids) // 2)]
    return [ids[0]]


IoTieColumn = HC._part(
    "IoTieColumn", (), _sig_column, _build_column,
    "the u32be id at the column (5 bits, least significant first) of a port row of 32, least significant bit first: "
    "31 32-bit muxes, 992 ANDs")


def _draw_statics(S) -> int:
    port = TokenPort(S.FOUT, 0, S.C, S.D)
    if _log2(S.FOUT, "FOUT") > S.D:
        raise ValueError(f"FOUT = {S.FOUT} is more than a depth-{S.D} tree's leaves")
    return port.DT


def _sig_draw(S):
    DT = _draw_statics(S)
    return (("frames_root", DIGEST), ("k", Array(S.D, BIT)), ("frame", Array(S.B, BYTE)), ("frame_path", Array(S.D, DIGEST)),
            ("port", DIGEST), ("port_domain", DIGEST), ("row", Array(ROW_BYTES, BYTE)), ("salt", SALT),
            ("row_path", Array(DT, DIGEST))), BIT


def _body_draw(B, S, frames_root, k, frame, frame_path, port, port_domain, row, salt, row_path):
    DT = _draw_statics(S)
    K = HC._Consts(B)
    kb, lf = HC._refs(k), _log2(S.FOUT, "FOUT")
    # row g·C + (k mod F) >> 5 with g = k >> log2(F): k's bits 5 ... log2(F) − 1, zeros to log2(C), then g's bits
    pos = kb[5:lf] + K.bits(0, _log2(S.C, "C") - (lf - 5)) + kb[lf:]
    row_bits = row.reshape(Array(8 * ROW_BYTES, BIT))
    token = B.call(IoTieColumn, row_bits, HC._coll(Array(5, BIT), kb[:5]))
    oks = [B.call(bind(NetworkFrameRead, B=S.B, D=S.D), frame, k, frame_path, frames_root),
           B.call(bind(IoTieEvent, B=S.B), frame, token),
           B.call(bind(HC.NetworkHidingRead, N_BITS=8 * ROW_BYTES, D=DT), row_bits, salt, port, port_domain,
                  HC._coll(Array(DT, BIT), pos), row_path)]
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BIT), bits))


IoTieDraw = CompositeDefinition(
    "IoTieDraw", 1, ("B", "D", "FOUT", "C"), _sig_draw, _body_draw, conformance="exact-model-tested",
    doc="1 iff the frame is leaf k of the frames tree (`NetworkFrameRead`), it is the id event of an id "
        "(`IoTieEvent`), and the port row at g·C + (k mod F) >> 5, g = k >> log2(F), opens under the port's root and "
        "domain (`NetworkHidingRead`) with that id at column k mod 32 (`IoTieColumn`)")


# --- arguments and the fake upstream's evaluation ---------------------------------------------------------------------

def draw_arguments(frames_root: bytes, k: int, d: Draw, port_root: bytes, port_domain: bytes, D: int) -> list[list[int]]:
    """`IoTieDraw`'s arguments for one draw in its signature's order, each its leaves' bits."""
    bits = HC._digest_bits
    return [bits(frames_root), HC._int_bits(k, D), HC._bytes_bits(d.frame), [b for s in d.frame_path for b in bits(s)],
            bits(port_root), bits(port_domain), HC._bytes_bits(d.row), bits(d.salt),
            [b for s in d.row_path for b in bits(s)]]


def fake_salt(seed: bytes, i: int) -> bytes:
    """A deterministic hiding salt for a fake upstream's tests (a prover draws fresh ones)."""
    out = b""
    while len(out) < hm96.SHA512.salt_bytes:
        out += hashlib.sha512(seed + i.to_bytes(8, "big") + len(out).to_bytes(4, "big")).digest()
    return out[:hm96.SHA512.salt_bytes]


__all__ = ["Draw", "ID_AT", "ID_EVENT_BYTES", "Instance", "IoTieColumn", "IoTieDraw", "IoTieEvent", "NetworkFrameRead",
           "NetworkFrameSwap", "TokenPort", "draw", "draw_arguments", "fake_salt", "frame_id", "head_counts", "id_event",
           "id_frame", "path", "port_levels", "reference", "reference_draws"]
