"""N2, the egress encoder (note:network-accounting/20261025T0720Z-draft-n2-egress-encoder): each egress bucket's row
rendered inside the window's circuit, from the window's token ids and a private map, as the exact bytes of the
warden's registered row for that bucket. Nothing here is in a Spec. Until proofs' EVALCOMP makes causality semantic,
structural Fits refuses a window that holds these units under V3's Programs (`routed` reads every id structurally),
so they are built, circuit-checked and proved standalone.

**The row is an output** (`EgressBucket`, one unit a link ℓ and bucket t): bucket t's R slots of B-byte cells, `o0`,
whose bit 8k + i is bit i of byte k: the `hm96-sha512/row/v2` row of its R·B/2 little-endian u16 words
(`warden_rows.words`), which is the warden's row t of link ℓ (`warden_rows.window_rows`) bit for bit exactly when the
cells are honest. The verifier's registered read (`flock-verify --registered`, `warden_rows.bucket_port`: unit
`reads_from + t` reads row t) holds o0 to the warden's signed row, so no gate compares a cell to anything and no gate
hashes a row. `o1` is presence, bit 0 the constant 1: the warden's egress is constant-rate, every slot of every bucket
holds a cell, an unheld one B zero bytes.

**A slot's cell** (`cell`) is its record (q, j, e) and its routed entry (p, id): a `PAD` on q for e = `EMPTY` (q = 0
the slot no connection holds); `POISON`·B for a request slot j ≥ NR or one not ready; the `END` on q for e =
`END_ENTRY`; the `DATA` cell of id on q for e < K and p = j·K + e (p's low PB bits, PB the bit length of NR·K); else
`POISON`·B. 0xFF is no kind the warden writes, so a poisoned slot's row opens no registered row. q is the request's
ingress ordinal on the wire, from 1; j its request slot in the window; e an output index of request slot j. The
gates: a 4-way mux by kind, every byte a constant of the kind but q and DATA's ten id characters (`b"%10d" % id`, by
double dabble); a constant byte costs no AND, since the kinds' selects are one-hot.

**What a row reads.** The encoder reads id only for a `DATA` record of a ready slot whose routed index is its own, so
unit (ℓ, t)'s row is a function of its map, its ready bits and {id_{j,e} : ready[j]} (the note's §7 local lemma).
`ready`'s bit j mod 16 of word ⌊j/16⌋ is 1 when request slot j's ingress is complete at buckets ≤ t (N1's ingress
shell's); bits at NR and above are ignored.

**The router** (`EgressRoute`, one a link-window): the window's NR·K ids, entry p = j·K + k request slot j's output k
tagged with its constant index p, and filler entries tagged NR·K (no j·K + e with j < NR and e < K), through a Beneš
network (`io_tie.IoTieRoute`) whose switches are the prover's witness (`switches`, `route_perm`: entry j·K + e to the
position t·R + σ whose record is `DATA` (q, j, e), the other entries to the rest in order). Any setting is a
permutation and an index travels with its id, so the encoder's index check pins the id it renders to output e of
request slot j whatever the switches. Positions t·R … t·R + R − 1 are encoder unit t's `routed`; positions at T·R
and above aren't outputs. The router runs `IoTieRoute`'s network on its entries (the constant indices folded, the
positions past T·R pruned), so splitting it a unit a Beneš column is a change to `_build_route` alone.

**The map** (`bucket_map`) is `EgressMap_v1`'s reference (N2b, its Definition to follow): `io_tie.window_map`'s rule
with the order the identity and e an output index.

    EgressBucket{B, R, NR, K}(map: Array<R, Array<3, u32>>, routed: Array<R, Array<2, u32>>,
        ready: Array<⌈NR/16⌉, u16>) -> (o0: Array<R·B, Byte>, o1: u16)
    EgressRoute{NR, K, T, R}(ids: Array<NR·K, u32>, switches: Array<(2D − 1)·2^(D − 1), BIT>)
        -> Array<T·R, Array<2, u32>>            W = PB + 32 bits routed an entry, D = route_depth(T·R, NR·K)

Every u32 and u16 is `Array<w, BIT>`, least significant bit first. B is even (a slot is whole u16 words) and at least
`io_tie.CELL_USED`; K < 2^32 − 2 and NR·K < 2^32.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple, Sequence

from verity.ml.circuits.boolean.forms import ONE, OR, ZERO, add, and_reduce, maj, or_reduce
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments.gates.frame_v3 import _part
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier.io_tie import (CELL_USED, EMPTY, END_ENTRY, HEADER, ID_AT, ID_DIGITS, IoTieRoute,
                                                    _eq, _lt, benes, benes_apply, data_cell, end_cell, pad_cell,
                                                    route_depth)

BIT, BYTE = HC.BIT, HC.BYTE
U16, U32 = Array(16, BIT), Array(32, BIT)
#: the byte of a cell no warden row holds: 0xFF is no kind (PAD 0, DATA 1, SALT 3, END 0x20, FAIL 0x21, REFUSED 0x22)
POISON = 0xFF


def entry_bits(NR: int, K: int) -> int:
    """PB, the bits of an entry index: wide enough for the filler index NR·K."""
    return (NR * K).bit_length()


def ready_words(NR: int) -> int:
    return -(-NR // 16)


def _check(B: object, R: object, NR: object, K: object) -> None:
    if type(B) is not int or B % 2 or not CELL_USED <= B < 1 << 32:
        raise ValueError(f"a cell is an even number of bytes, at least {CELL_USED}, not B={B!r}")
    if type(R) is not int or R < 1:
        raise ValueError(f"a bucket has R >= 1 slots, not R={R!r}")
    if type(NR) is not int or type(K) is not int or NR < 1 or not 1 <= K < END_ENTRY or NR * K >= 1 << 32:
        raise ValueError(f"a window has NR >= 1 request slots of 1 <= K < 2^32 − 2 outputs, NR·K < 2^32, not "
                         f"NR={NR!r}, K={K!r}")


@dataclass(frozen=True)
class Shape:
    """An egress link-window's encoder units and router: T buckets of R slots of B-byte cells, and the window's NR
    request slots of K outputs each, entry j·K + k request slot j's output k, routed over M = 2^D positions."""

    B: int
    T: int
    R: int
    NR: int
    K: int

    def __post_init__(self) -> None:
        _check(self.B, self.R, self.NR, self.K)
        if type(self.T) is not int or self.T < 1 or self.T * self.R > 1 << 32:
            raise ValueError(f"a link-window has T >= 1 buckets, at most 2^32 slots, not T={self.T!r}")

    @property
    def PB(self) -> int:
        return entry_bits(self.NR, self.K)

    @property
    def D(self) -> int:
        return route_depth(self.T * self.R, self.NR * self.K)

    @property
    def M(self) -> int:
        return 1 << self.D

    @property
    def W(self) -> int:
        """A routed entry's bits: its index, then its id."""
        return self.PB + 32

    @property
    def switches(self) -> int:
        return (2 * self.D - 1) << (self.D - 1)


class Record(NamedTuple):
    """A slot's record: its stream q (the request's ingress ordinal on the wire, 0 for a slot no connection holds),
    its request slot j, and e: `EMPTY` for a `PAD`, `END_ENTRY` for the `END`, or an output k < K for `DATA`."""

    q: int
    j: int
    e: int


#: a slot no connection holds: B zero bytes, a `PAD` on stream 0
UNHELD = Record(0, 0, EMPTY)


def _u32(name: str, x: object) -> int:
    if type(x) is not int or not 0 <= x < 1 << 32:
        raise ValueError(f"{name} is a u32, not {x!r}")
    return x


# --- the reference ---------------------------------------------------------------------------------------------------

def cell(B: int, NR: int, K: int, record: Sequence[int], entry: Sequence[int], ready: Sequence[int]) -> bytes:
    """`EgressBucket`'s cell of one slot: its record (q, j, e), its routed entry (p, id) and the ready bits (bit j
    request slot j's)."""
    _check(B, 1, NR, K)
    q, j, e = (_u32(n, x) for n, x in zip("qje", record, strict=True))
    p, tid = (_u32(n, x) for n, x in zip(("p", "id"), entry, strict=True))
    if e == EMPTY:
        return pad_cell(B, q)
    if j >= NR or not ready[j]:
        return bytes([POISON]) * B
    if e == END_ENTRY:
        return end_cell(q, B)
    if e >= K or p % (1 << entry_bits(NR, K)) != j * K + e:
        return bytes([POISON]) * B
    return data_cell(tid, q, B)


def bucket_row(B: int, NR: int, K: int, records: Sequence[Sequence[int]], routed: Sequence[Sequence[int]],
               ready: Sequence[int]) -> bytes:
    """`EgressBucket`'s o0: the bucket's R cells, slot σ's at bytes σ·B … (σ + 1)·B − 1."""
    if len(records) != len(routed):
        raise ValueError(f"one routed entry a slot: {len(records)} records and {len(routed)} entries")
    return b"".join(cell(B, NR, K, m, x, ready) for m, x in zip(records, routed))


# --- the prover's side: the map, the ready bits and the routing ------------------------------------------------------

class Request(NamedTuple):
    """A link-window's request slot: its stream q (from 1), its admission bucket and its n outputs."""

    q: int
    admitted: int
    n: int


def bucket_map(b: int, slots: Sequence[Request | None], R: int, r: int, D: int) -> tuple[tuple[Record, ...], bool]:
    """Bucket b's records and its overflow bit (`EgressMap_v1`'s reference): the live request slots at b, the j with
    a request admitted at or before b and b ≤ admitted_j + D + ⌊n_j / r⌋, ranked by q as ρ. Slot ρ·r + i is
    (q_j, j, `EMPTY`) before admitted_j + D; else, with k = (b − admitted_j − D)·r + i, (q_j, j, k) for k < n_j,
    (q_j, j, `END_ENTRY`) for k = n_j and (q_j, j, `EMPTY`) past it. The slots past the live ones are `UNHELD`.
    Overflow is 1 when more than R / r slots are live; the records are then the first R / r's."""
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


def honest_ready(b: int, slots: Sequence[Request | None], NR: int) -> list[int]:
    """A fake upstream's ready bits at bucket b: request slot j's ingress is complete once it is admitted."""
    if len(slots) > NR:
        raise ValueError(f"a window has NR = {NR} request slots, not {len(slots)}")
    return [int(s is not None and s.admitted <= b) for s in slots] + [0] * (NR - len(slots))


def route_perm(shape: Shape, maps: Sequence[Sequence[Record]]) -> list[int]:
    """The window's routing, which the prover sets: entry j·K + e to the position t·R + σ whose record is `DATA`
    (q, j, e) (the first such), the other entries to the remaining positions in order."""
    sh = shape
    if len(maps) != sh.T or any(len(m) != sh.R for m in maps):
        raise ValueError(f"a window's map is T = {sh.T} buckets of R = {sh.R} records")
    perm, taken = [-1] * sh.M, set()
    for t, records in enumerate(maps):
        for s, m in enumerate(records):
            if m.j < sh.NR and m.e < sh.K and perm[m.j * sh.K + m.e] < 0:
                perm[m.j * sh.K + m.e] = t * sh.R + s
                taken.add(t * sh.R + s)
    rest = iter(i for i in range(sh.M) if i not in taken)
    return [x if x >= 0 else next(rest) for x in perm]


def switches(shape: Shape, maps: Sequence[Sequence[Record]]) -> list[int]:
    """The prover's switch bits (`route_perm`'s Beneš network, `io_tie.benes`)."""
    return benes(route_perm(shape, maps))


def entries(shape: Shape, ids: Sequence[int]) -> list[tuple[int, int]]:
    """The router's inputs as (index, id): entry p of the window's ids, and filler entries (NR·K, 0) past them."""
    n = shape.NR * shape.K
    if len(ids) != n:
        raise ValueError(f"a window has NR·K = {n} ids, not {len(ids)}")
    return [(p, _u32("an id", x)) for p, x in enumerate(ids)] + [(n, 0)] * (shape.M - n)


def routed(shape: Shape, ids: Sequence[int], bits: Sequence[int]) -> list[tuple[int, int]]:
    """`EgressRoute` in the clear: the (index, id) at each of the T·R positions under the switch bits."""
    return benes_apply(entries(shape, ids), bits)[:shape.T * shape.R]


# --- the circuits ----------------------------------------------------------------------------------------------------

def _sig_bucket(S):
    _check(S.B, S.R, S.NR, S.K)
    return ((("map", Array(S.R, Array(3, U32))), ("routed", Array(S.R, Array(2, U32))),
             ("ready", Array(ready_words(S.NR), U16))), Tuple(Array(S.R * S.B, BYTE), U16))


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


def _kinds(C, S, q, j, e, p, ready) -> tuple:
    """A slot's one-hot kind: (pad, data, end, poison)."""
    PB, kb = entry_bits(S.NR, S.K), (S.K - 1).bit_length()
    top = and_reduce(C, e[1:])
    pad, marked_end = C.AND(top, e[0]), C.AND(top, e[0].inv())
    held = C.AND(pad.inv(), C.AND(_lt(C, j, _const(S.NR, 32)), _select(C, ready[:S.NR], j)))
    # j·K + e mod 2^PB, e cut to the bits of K − 1: exact when j < NR and e < K, which a DATA cell needs
    index = list(e[:min(kb, PB)]) + [ZERO] * (PB - min(kb, PB))
    for s in range(PB):
        if S.K >> s & 1:
            index = add(C, index, [ZERO] * s + list(j[:PB - s]), width=PB)
    end = C.AND(held, marked_end)
    data = C.AND(C.AND(held, marked_end.inv()), C.AND(_lt(C, e, _const(S.K, 32)), _eq(C, p[:PB], index)))
    return pad, data, end, (pad ^ data ^ end).inv()


def _build_bucket(C, S, *els):
    R, B = S.R, S.B
    mp, rt = els[:3 * R], els[3 * R:5 * R]
    ready = [w[b] for w in els[5 * R:] for b in range(16)]
    tmpl = (pad_cell(B), data_cell(0, 0, B), end_cell(0, B), bytes([POISON]) * B)
    digit_at = range(ID_AT, ID_AT + ID_DIGITS)
    assert all(tmpl[0][k] == tmpl[2][k] == 0 for k in digit_at) and HEADER.size == 9
    out = []
    for s in range(R):
        q, j, e = mp[3 * s:3 * s + 3]
        p, tid = rt[2 * s:2 * s + 2]
        kinds = _kinds(C, S, q, j, e, p, ready)
        poison, data = kinds[3], kinds[1]
        chars = _id_chars(C, tid)
        for k in range(B):
            if 1 <= k <= 4:
                # q is u32 big-endian at bytes 1 … 4: byte k's bit i is q's bit 8·(4 − k) + i
                out.append([OR(C, q[8 * (4 - k) + i], poison) for i in range(8)])
            elif k in digit_at:
                out.append([C.AND(data, c) ^ poison for c in chars[k - ID_AT]])
            else:
                byte = []
                for i in range(8):
                    v = ZERO
                    for sel, t in zip(kinds, tmpl):
                        if t[k] >> i & 1:
                            v = v ^ sel
                    byte.append(v)
                out.append(byte)
    return [*out, [ONE] + [ZERO] * 15]


EgressBucket = _part(
    "EgressBucket", ("B", "R", "NR", "K"), _sig_bucket, _build_bucket,
    "(o0, o1) of one egress bucket: o0 its R B-byte cells, slot σ's from its record (q, j, e) and routed entry "
    "(p, id): a PAD on q for e = 2^32 − 1; 0xFF·B for j >= NR or ready bit j 0; the END on q for e = 2^32 − 2; the "
    "DATA cell of id on q for e < K and p's low PB bits j·K + e (PB the bit length of NR·K); else 0xFF·B; o1 "
    "presence, 1 in bit 0. u32 and u16 LSB first, o0's bit 8k + i bit i of byte k")


def _route_shape(S) -> Shape:
    """The router's shape; it reads no cell, so any B."""
    return Shape(CELL_USED + CELL_USED % 2, S.T, S.R, S.NR, S.K)


def _sig_route(S):
    sh = _route_shape(S)
    return ((("ids", Array(S.NR * S.K, U32)), ("switches", Array(sh.switches, BIT))), Array(S.T * S.R, Array(2, U32)))


def _build_route(C, S, *els):
    sh, n = _route_shape(S), S.NR * S.K
    xs = [_const(min(p, n), sh.PB) + (list(els[p]) if p < n else [ZERO] * 32) for p in range(sh.M)]
    out = IoTieRoute.build(C, SimpleNamespace(W=sh.W, D=sh.D), *xs, els[n])
    return [x for i in range(S.T * S.R) for x in (out[i][:sh.PB] + [ZERO] * (32 - sh.PB), out[i][sh.PB:])]


EgressRoute = _part(
    "EgressRoute", ("NR", "K", "T", "R"), _sig_route, _build_route,
    "an egress link-window's ids routed to its T·R slots: entry p < NR·K (request slot j's output k at p = j·K + k) "
    "tagged with p, and M − NR·K filler entries tagged NR·K with id 0, through `IoTieRoute{W=PB+32, D}`'s Beneš "
    "network on M = 2^D positions (D = route_depth(T·R, NR·K)) under the switches; position t·R + σ's (index "
    "zero-extended to 32 bits, id) for t·R + σ < T·R")


# --- arguments -------------------------------------------------------------------------------------------------------

def _u32_bits(xs: Sequence[int]) -> list[int]:
    return [b for x in xs for b in HC._int_bits(_u32("a u32", x), 32)]


def bucket_arguments(NR: int, records: Sequence[Sequence[int]], routed_: Sequence[Sequence[int]],
                     ready: Sequence[int]) -> list[list[int]]:
    """`EgressBucket`'s arguments in its signature's order, each its leaves' bits: the map, the routed entries and the
    ready words (bit j of the ready bits at bit j mod 16 of word ⌊j / 16⌋, zero past them)."""
    bits = [int(bool(x)) for x in ready[:NR]]
    return [_u32_bits([x for m in records for x in m]), _u32_bits([x for r in routed_ for x in r]),
            bits + [0] * (16 * ready_words(NR) - len(bits))]


def route_arguments(ids: Sequence[int], bits: Sequence[int]) -> list[list[int]]:
    """`EgressRoute`'s arguments in its signature's order, each its leaves' bits."""
    return [_u32_bits(ids), [int(b) for b in bits]]


__all__ = ["EgressBucket", "EgressRoute", "POISON", "Record", "Request", "Shape", "UNHELD", "bucket_arguments",
           "bucket_map", "bucket_row", "cell", "entries", "entry_bits", "honest_ready", "ready_words",
           "route_arguments", "route_perm", "routed", "switches"]
