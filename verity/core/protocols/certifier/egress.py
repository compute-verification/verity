"""N2, the egress side of the step-indexed scan (note:network-accounting/20261025T0830Z-report-net68-scan-router; the
cells as note:network-accounting/20261025T0720Z-draft-n2-egress-encoder renders them): each egress bucket's cells
computed from the scan's slot state before it and held, inside the unit, to the warden's registered row for that
bucket. Nothing here is in a Spec yet.

**One unit a link ℓ and bucket t** (`EgressBucket`). It reads two rows, both registered reads of input ports (which
`Flock.Registered.checkPort` accepts): ``state``, the egress view of the slot state S_t the scan's ingress step t
wrote, and ``row``, the warden's egress row t of link ℓ (`warden_rows.bucket_port`: unit `reads_from + t` reads row
t), whose bit 8k + i is bit i of byte k. Its one output is a u16 verdict whose bit 0 is 1 (`ACCEPT`) exactly when the
cells the state renders are the row bit for bit, and the other 15 bits are 0; a proof unit's outputs are at most 16
bits. Every switch is computed from the state, none is witness, so the unit adds nothing to c, and bucket t's egress
reads no ingress after bucket t: structural Fits (`NetTiming.Layout.Fits`: no egress gate of step t depends on an
ingress gate of a step t' > t) holds by construction. It reads S_t rather than S_{t−1} because the warden's egress
row of a connection's admission bucket already holds its lane, a `PAD` on its q.

**The state's egress view** (`Shape`'s offsets) is a row of u16 words, bit 16w + i bit i of word w, each field least
significant bit first: b, the bucket the state is read at (the state step t writes holds b = t), then NR request
slots of `Shape.slot_bits` each: live (bit 0 of a u16; its other bits unread), q
(the request's ingress ordinal on the wire, from 1), admitted (its admission bucket), n (its output count) and its K
output ids of IB bits, output k at bit 112 + k·IB, the slot zero-padded to whole words. The ingress step writes b,
live, q and admitted, the anchored Programs n and the ids (the scan note's §1).

**The map** (`bucket_map`): the live slots at b are the j with live_j, admitted_j ≤ b and b ≤ admitted_j + D +
⌊n_j / r⌋, ranked by (q, j) as ρ, lane ρ rank ρ's. Slot ρ·r + i of the row is a `PAD` on q_j before admitted_j + D;
else, with k = (b − admitted_j − D)·r + i, the `DATA` cell of output k's id on q_j for k < n_j, the `END` on q_j for
k = n_j and a `PAD` on q_j past it. The lanes past the live ones are B zero bytes (a `PAD` on 0). More live slots than
R / r lanes, or a `DATA` cell of an output k ≥ K, renders no row, and the verdict is 0.

**The gates**: one comparison of q a pair of request slots for the ranks; for each slot and cell its output k's id by
a mux over the slot's K ids, k computed from b and the slot's admitted; each lane's fields the XOR of the slots'
masked by "of that rank" (the live ranks are distinct, so this is exact and needs no mux); a cell a select of its
kind, every byte a constant of the kind but q and `DATA`'s ten id characters (`b"%10d" % id`, by double dabble); then
one AND a row bit for the equality.

    EgressBucket{B, R, r, NR, K, IB, D}(state: Array<Shape.state_words, u16>, row: Array<R·B, Byte>) -> u16

Every u16 and u32 is `Array<w, BIT>`, least significant bit first. B is even (a slot is whole u16 words) and at least
`io_tie.CELL_USED`; r divides R; K < 2^32 − 2; 1 <= IB <= 32; D < 2^32.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

from verity.ml.circuits.boolean.forms import ONE, OR, ZERO, add, and_reduce, maj, or_reduce
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates.frame_v3 import _part
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier.io_tie import (CELL_USED, EMPTY, END_ENTRY, HEADER, ID_AT, ID_DIGITS, _eq, _lt,
                                                    data_cell, end_cell, pad_cell)

BIT, BYTE = HC.BIT, HC.BYTE
U16 = Array(16, BIT)
#: the verdict word of a bucket whose row is the one its state renders
ACCEPT = 1
#: a slot's fields' bit offsets in the state: live (a u16), q, admitted and n (u32s), then the K ids
LIVE, Q, ADMITTED, N, IDS = 0, 16, 48, 80, 112


@dataclass(frozen=True)
class Shape:
    """An egress bucket: R slots of B-byte cells in lanes of r, the state's NR request slots of K output ids of IB
    bits each, and the warden's delay D."""

    B: int
    R: int
    r: int
    NR: int
    K: int
    IB: int
    D: int

    def __post_init__(self) -> None:
        B, R, r, NR, K, IB, D = self.B, self.R, self.r, self.NR, self.K, self.IB, self.D
        if type(B) is not int or B % 2 or not CELL_USED <= B < 1 << 32:
            raise ValueError(f"a cell is an even number of bytes, at least {CELL_USED}, not B={B!r}")
        if type(R) is not int or type(r) is not int or R < 1 or r < 1 or R % r:
            raise ValueError(f"a bucket's R >= 1 slots are lanes of r >= 1, not R={R!r}, r={r!r}")
        if type(NR) is not int or type(K) is not int or not 1 <= NR < 1 << 16 or not 1 <= K < END_ENTRY:
            raise ValueError(f"a state has 1 <= NR < 2^16 request slots of 1 <= K < 2^32 − 2 outputs, not NR={NR!r}, "
                             f"K={K!r}")
        if type(IB) is not int or not 1 <= IB <= 32 or type(D) is not int or not 0 <= D < 1 << 32:
            raise ValueError(f"an id is 1 <= IB <= 32 bits and the delay 0 <= D < 2^32, not IB={IB!r}, D={D!r}")

    @property
    def lanes(self) -> int:
        return self.R // self.r

    @property
    def slot_bits(self) -> int:
        return -(-(IDS + self.K * self.IB) // 16) * 16

    @property
    def state_bits(self) -> int:
        return 32 + self.NR * self.slot_bits

    @property
    def state_words(self) -> int:
        return self.state_bits // 16

    def slot_at(self, j: int) -> int:
        """Request slot j's first bit in the state."""
        return 32 + j * self.slot_bits


def _shape(S) -> Shape:
    return Shape(S.B, S.R, S.r, S.NR, S.K, S.IB, S.D)


def _u32(name: str, x: object) -> int:
    if type(x) is not int or not 0 <= x < 1 << 32:
        raise ValueError(f"{name} is a u32, not {x!r}")
    return x


# --- the reference ---------------------------------------------------------------------------------------------------

class Record(NamedTuple):
    """A slot's record: its stream q (0 for a slot no connection holds), its request slot j, and e: `EMPTY` for a
    `PAD`, `END_ENTRY` for the `END`, or the output k for `DATA`."""

    q: int
    j: int
    e: int


#: a lane no live request slot holds: B zero bytes, a `PAD` on stream 0
UNHELD = Record(0, 0, EMPTY)


class Request(NamedTuple):
    """A live request slot as the map reads it: its stream q (from 1), its admission bucket and its n outputs."""

    q: int
    admitted: int
    n: int


class Slot(NamedTuple):
    """A request slot of the state: live (bit 0 read), q, admitted, n and its output ids (zero past them, at most K)."""

    live: int
    q: int
    admitted: int
    n: int
    ids: tuple[int, ...]


class State(NamedTuple):
    """The state's egress view: b, the bucket it is read at, and its request slots (`FREE` past them)."""

    b: int
    slots: tuple[Slot, ...]


FREE = Slot(0, 0, 0, 0, ())


def bucket_map(b: int, slots: Sequence[Request | None], R: int, r: int, D: int) -> tuple[tuple[Record, ...], bool]:
    """Bucket b's records and whether it overflows: the live request slots at b, the j with a request admitted at or
    before b and b ≤ admitted_j + D + ⌊n_j / r⌋, ranked by (q, j) as ρ. Slot ρ·r + i is (q_j, j, `EMPTY`) before
    admitted_j + D; else, with k = (b − admitted_j − D)·r + i, (q_j, j, k) for k < n_j, (q_j, j, `END_ENTRY`) for
    k = n_j and (q_j, j, `EMPTY`) past it. The slots past the live ones are `UNHELD`. It overflows when more than
    R / r slots are live; the records are then the first R / r's."""
    if type(r) is not int or r < 1 or type(R) is not int or R % r or type(D) is not int or D < 0:
        raise ValueError(f"a bucket's R slots are lanes of r >= 1, and the delay D >= 0, not R={R!r}, r={r!r}, D={D!r}")
    live = sorted((s.q, j) for j, s in enumerate(slots)
                  if s is not None and s.admitted <= b <= s.admitted + D + s.n // r)
    out = [UNHELD] * R
    for rho, (q, j) in enumerate(live[:R // r]):
        s = slots[j]
        for i in range(r):
            k = (b - s.admitted - D) * r + i
            e = EMPTY if b < s.admitted + D or k > s.n else END_ENTRY if k == s.n else k
            out[rho * r + i] = Record(q, j, e)
    return tuple(out), len(live) * r > R


def _slots(sh: Shape, state: State) -> list[Slot]:
    if len(state.slots) > sh.NR:
        raise ValueError(f"a state has NR = {sh.NR} request slots, not {len(state.slots)}")
    out = []
    for s in list(state.slots) + [FREE] * (sh.NR - len(state.slots)):
        if type(s.live) is not int or not 0 <= s.live < 1 << 16:
            raise ValueError(f"live is a u16, not {s.live!r}")
        _u32("q", s.q), _u32("admitted", s.admitted), _u32("n", s.n)
        if len(s.ids) > sh.K or any(type(x) is not int or not 0 <= x < 1 << sh.IB for x in s.ids):
            raise ValueError(f"a request slot has at most K = {sh.K} ids of IB = {sh.IB} bits, not {s.ids!r}")
        out.append(s._replace(ids=tuple(s.ids) + (0,) * (sh.K - len(s.ids))))
    return out


def rendered(sh: Shape, state: State) -> bytes | None:
    """The row bucket b's cells are under the state (`bucket_map`'s records rendered: a `PAD` on q for `EMPTY`, the
    `END` on q, the `DATA` cell of request slot j's output e), or None when it renders none: more live slots than
    lanes, or a `DATA` record of an output e ≥ K."""
    slots = _slots(sh, state)
    live = [Request(s.q, s.admitted, s.n) if s.live & 1 else None for s in slots]
    records, over = bucket_map(_u32("b", state.b), live, sh.R, sh.r, sh.D)
    if over:
        return None
    cells = []
    for q, j, e in records:
        if e == EMPTY:
            cells.append(pad_cell(sh.B, q))
        elif e == END_ENTRY:
            cells.append(end_cell(q, sh.B))
        elif e >= sh.K:
            return None
        else:
            cells.append(data_cell(slots[j].ids[e], q, sh.B))
    return b"".join(cells)


def verdict(sh: Shape, state: State, row: bytes) -> int:
    """`EgressBucket`'s verdict word: `ACCEPT` when the row is the one the state renders, else 0."""
    if len(row) != sh.R * sh.B:
        raise ValueError(f"a row is R·B = {sh.R * sh.B} bytes, not {len(row)}")
    return ACCEPT if rendered(sh, state) == row else 0


def state_bits(sh: Shape, state: State) -> list[int]:
    """The state's egress view as its row's bits (the module docstring's layout)."""
    bits = HC._int_bits(_u32("b", state.b), 32)
    for s in _slots(sh, state):
        slot = (HC._int_bits(s.live, 16) + HC._int_bits(s.q, 32) + HC._int_bits(s.admitted, 32) + HC._int_bits(s.n, 32)
                + [x for v in s.ids for x in HC._int_bits(v, sh.IB)])
        bits += slot + [0] * (sh.slot_bits - len(slot))
    return bits


def state_words(sh: Shape, state: State) -> list[int]:
    """The state's row as u16 words (bit i of word w its bit 16w + i)."""
    bits = state_bits(sh, state)
    return [sum(b << i for i, b in enumerate(bits[w:w + 16])) for w in range(0, len(bits), 16)]


def bucket_arguments(sh: Shape, state: State, row: bytes) -> list[list[int]]:
    """`EgressBucket`'s arguments in its signature's order, each its leaves' bits: the state's row and the warden's."""
    if len(row) != sh.R * sh.B:
        raise ValueError(f"a row is R·B = {sh.R * sh.B} bytes, not {len(row)}")
    return [state_bits(sh, state), [x >> i & 1 for x in row for i in range(8)]]


# --- the circuit -----------------------------------------------------------------------------------------------------

def _sig_bucket(S):
    sh = _shape(S)
    return (("state", Array(sh.state_words, U16)), ("row", Array(sh.R * sh.B, BYTE))), U16


def _const(c: int, w: int) -> list:
    return [ONE if c >> b & 1 else ZERO for b in range(w)]


def _select(C, bits: Sequence, index: Sequence):
    """bits[index] for index below len(bits), by a mux tree over index's low bits: len(bits) − 1 ANDs at most."""
    level = list(bits)
    for b in range((len(level) - 1).bit_length()):
        level += [ZERO] * (len(level) % 2)
        level = [x ^ C.AND(index[b], x ^ y) for x, y in zip(level[::2], level[1::2])]
    return level[0]


def _decimal(C, x: Sequence) -> list[list]:
    """The 10 BCD digits of u32 x, units first, each 4 bits LSB first: double dabble, each digit that may be 5 or
    more taking 3 when it is before each shift (5 ANDs; constants fold the digits not yet reached)."""
    digits = [[ZERO] * 4 for _ in range(ID_DIGITS)]
    for bit in reversed(list(x)):
        for d in digits:
            # d ≥ 5 is d3, or d2 with d1 or d0; d3 excludes d2 for a digit of at most 9
            s = d[3] ^ C.AND(d[2], OR(C, d[1], d[0]))
            c1 = C.AND(d[0], s)
            c2 = maj(C, d[1], s, c1)
            d[:] = [d[0] ^ s, d[1] ^ s ^ c1, d[2] ^ c2, d[3] ^ C.AND(d[2], c2)]
        carry = bit
        for d in digits:
            carry, d[:] = d[3], [carry, *d[:3]]
    return digits


def _id_chars(C, tid: Sequence) -> list[list]:
    """`b"%10d" % id`'s characters, most significant first, each 8 bits LSB first: a digit 0x30 + d, a leading zero a
    space 0x20 (the units digit never)."""
    digits = _decimal(C, tid)[::-1]
    shown, chars = ZERO, []
    for i, d in enumerate(digits):
        shown = ONE if i == ID_DIGITS - 1 else OR(C, shown, or_reduce(C, d))
        chars.append([*d, shown, ONE, ZERO, ZERO])
    return chars


def _sub(C, x: Sequence, y: Sequence) -> list:
    """x − y mod 2^len(x), y as wide as x."""
    return add(C, x, [v.inv() for v in y], cin=ONE, width=len(x))


def _times(C, x: Sequence, c: int, w: int) -> list:
    """x·c mod 2^w for a constant c >= 1: an add for each set bit of c but one."""
    out = None
    for s in range(c.bit_length()):
        if c >> s & 1:
            term = ([ZERO] * s + list(x) + [ZERO] * w)[:w]
            out = term if out is None else add(C, out, term, width=w)
    return out


def _xor(xs: Sequence):
    out = ZERO
    for x in xs:
        out = out ^ x
    return out


def _build_bucket(C, S, *els):
    sh = _shape(S)
    r, NR, K, IB, D, B = sh.r, sh.NR, sh.K, sh.IB, sh.D, sh.B
    sb = [w[i] for w in els[:sh.state_words] for i in range(16)]
    row = els[sh.state_words:]
    b = sb[:32]
    W, w = 32 + r.bit_length(), max(1, (NR - 1).bit_length())
    q, live, cells = [], [], []
    for j in range(NR):
        at = sh.slot_at(j)
        lv, qj = sb[at + LIVE], sb[at + Q:at + Q + 32]
        a, nj = sb[at + ADMITTED:at + ADMITTED + 32], sb[at + N:at + N + 32]
        ids = [sb[at + IDS + k * IB:at + IDS + (k + 1) * IB] for k in range(K)]
        aD, b33 = add(C, a, _const(D, 32), width=33), b + [ZERO]
        pre = _lt(C, b33, aD)
        x = _sub(C, b33, aD)[:32]
        xr, nW = _times(C, x, r, W), nj + [ZERO] * (W - 32)
        # b <= admitted + D + floor(n / r) is x·r <= n, once b >= admitted + D
        live.append(C.AND(lv, C.AND(_lt(C, b, a).inv(), OR(C, pre, _lt(C, nW, xr).inv()))))
        q.append(qj)
        emits, mine = pre.inv(), []
        for i in range(r):
            k = add(C, xr, _const(i, W), width=W)
            data, end = C.AND(emits, _lt(C, k, nW)), C.AND(emits, _eq(C, k, nW))
            tid = [_select(C, [v[t] for v in ids], k) for t in range(IB)]
            mine.append((data, end, C.AND(data, _lt(C, k, _const(K, W)).inv()), tid))
        cells.append(mine)
    # before[i][j]: request slot i ranks before j, by (q, slot): for i < j, q_i <= q_j
    before = [[ZERO] * NR for _ in range(NR)]
    for j in range(NR):
        for i in range(j):
            c = _lt(C, q[j], q[i])
            before[j][i], before[i][j] = c, c.inv()
    rank = []
    for j in range(NR):
        k = [ZERO] * w
        for i in range(NR):
            if i != j:
                k = add(C, k, [C.AND(live[i], before[i][j])], width=w)
        rank.append(k)
    lanes, bad = sh.lanes, []
    if lanes < NR:
        bad += [C.AND(live[j], _lt(C, rank[j], _const(lanes, w)).inv()) for j in range(NR)]
    tmpl = (pad_cell(B), data_cell(0, 0, B), end_cell(0, B))
    digit_at = range(ID_AT, ID_AT + ID_DIGITS)
    assert all(tmpl[0][k] == tmpl[2][k] == 0 for k in digit_at) and HEADER.size == 9
    eqs = []
    for rho in range(lanes):
        sel = [C.AND(live[j], _eq(C, rank[j], _const(rho, w))) if rho < NR else ZERO for j in range(NR)]
        sq = [_xor(C.AND(s, v[t]) for s, v in zip(sel, q)) for t in range(32)]
        for i in range(r):
            data, end, past = (_xor(C.AND(s, cells[j][i][f]) for j, s in enumerate(sel)) for f in range(3))
            tid = [_xor(C.AND(s, cells[j][i][3][t]) for j, s in enumerate(sel)) for t in range(IB)]
            bad.append(past)
            pad = (data ^ end).inv()
            chars = _id_chars(C, tid + [ZERO] * (32 - IB))
            got = row[(rho * r + i) * B:(rho * r + i + 1) * B]
            for k in range(B):
                if 1 <= k <= 4:
                    # q is u32 big-endian at bytes 1 … 4: byte k's bit t is q's bit 8·(4 − k) + t
                    byte = [sq[8 * (4 - k) + t] for t in range(8)]
                elif k in digit_at:
                    byte = [C.AND(data, c) for c in chars[k - ID_AT]]
                else:
                    byte = [_xor(kind for kind, t in zip((pad, data, end), tmpl) if t[k] >> u & 1) for u in range(8)]
                eqs += [(v ^ g).inv() for v, g in zip(byte, got[k])]
    ok = and_reduce(C, eqs + [or_reduce(C, bad).inv()])
    return [ok] + [ZERO] * 15


EgressBucket = _part(
    "EgressBucket", ("B", "R", "r", "NR", "K", "IB", "D"), _sig_bucket, _build_bucket,
    "the verdict of one egress bucket: 1 in bit 0 exactly when its row of R B-byte cells is the one its state "
    "renders. The state, b then NR request slots of (live u16, q, admitted, n u32s, K ids of IB bits) each padded to "
    "whole u16s: the live slots at b, bit 0 of live 1, admitted <= b <= admitted + D + floor(n / r), ranked by (q, j) "
    "as rho; slot rho·r + i a PAD on q before admitted + D, else with k = (b − admitted − D)·r + i the DATA cell of id "
    "k for k < n, the END for k = n, a PAD past it; the lanes past the live ones B zero bytes; no row when more than "
    "R / r slots are live or a DATA cell has k >= K. u16 and u32 LSB first, the row's bit 8k + i bit i of byte k")


__all__ = ["ACCEPT", "ADMITTED", "EgressBucket", "FREE", "IDS", "LIVE", "N", "Q", "Record", "Request", "Shape",
           "Slot", "State", "UNHELD", "bucket_arguments", "bucket_map", "rendered", "state_bits", "state_words",
           "verdict"]
