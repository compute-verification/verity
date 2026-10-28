"""The ingress side of the step-indexed scan: one unit a certifier bucket t, `IngressScanBucket`, which reads bucket
t's ingress row, the carried state S_{t−1} and the Programs' row at t, and computes S_t from them and S_t's egress
view, its outputs beside an ok bit (the generator form); the first bucket's, `IngressScanFirst`, reads no state and
starts from the empty one. Every switch is computed from the state and the rows; none is witness. Nothing here is
in a Spec.

**The row** is `warden_rows.window_rows`' row t of the ingress link: S = lanes·r slots of B-byte cells, lane block ρ
(slots ρ·r … ρ·r + r − 1) the ρ-th live connection by ordinal and a block no connection holds r·B zero bytes. Its cells
are the warden's (`warden.read_cell`): `DATA` of P = B − 9 bytes, `END` with the request's last 1 … P bytes, `PAD` of
none, each on the connection's ordinal q. A connection's ingress is one warden-form id request
(`active.id_request(..., offset=0)`: no LENGTH, its length the content-length plus the head), cut by `warden.cells`, so
a bucket's cells of a connection are `DATA`, at most one `END`, then `PAD`s, and only `PAD`s once its `END` is in.

**Fixed slots a timestep.** A timestep is EVERY buckets,
its boundary the bucket t ≡ 0 mod EVERY of the scan, and the warden's window is whole timesteps (EVERY divides T,
`Class.serves`), so every link-window has its boundaries at the same buckets and one window circuit serves them all.
A request admitted at bucket s is anchored at A(s), the first boundary at or after s + WITHIN (`Class.anchor`), WITHIN
the class's ingress bound δ in buckets, and its Program reads its slot in S_{A−1}. Each boundary has SLOTS = lanes
Program slots, slot j the Program that reads the scan's slot j: a request holds its slot from s to at least
s + D + 1 > EVERY buckets, so the requests anchored at one boundary are in distinct slots, and SLOTS = lanes holds every
request the warden admits. Only EVERY and SLOTS are public: whether a slot carries a request, which, its arrival bucket
and its prompt's length are values. Traffic past the slots is the warden's to refuse: a connection that finds every lane
and queue place taken gets `REFUSED`, and the network certifier certifies no record of that window, so the period check
refuses it (`warden`'s "Admission").

**The state** (`layout`) is `now` (u32, the buckets the scan has taken), `phase` (`now` mod EVERY, u32) and `last`
(u32, the last ordinal admitted), then NR = lanes request slots, each `live`, `done` (its request's `END` is in), `q`,
`left` (its request's bytes still to come, u32), `cells` (received, CB = bit length of C bits), `fields` (prompt ids
received, u32), `max_tokens` and `seed` (u32), `temperature` and `top_p` (their seven digits' value, 24 bits),
`admitted` (`now` at its admission, u32), `anchor` (A(admitted), u32), `occ` (u32, the slot's occupancy word: 1
when the slot's Program starts at bucket `now`, the slot live, its `END` in and its anchor `now`, else 0; a reader
reads its bit 0), `n` and `out`, its Program's output count (u32) and KO output ids of IDB bits (both zero until its
anchor), and `ids`, C·K prompt ids of IDB bits: cell c's field k at c·K + k, so prompt position p is at `position(p)`.
Each field is least significant bit first, the fields in `layout`'s order, then `pad`, zeros to a whole 16-bit word
(S_t's zero, `sin`'s not read). The empty state (`empty`) is all zeros.

**The token-id row** (`token_row`): the Programs at boundary b read the scan's state row b − 1, o1 of bucket b − 1's
unit, committed under its hiding leaf and registered as `verity/certifier/scan-state/v1`'s row b − 1: slot j's `occ`,
1 where the slot's Program starts at b and 0 for an empty slot, its `fields`, the request's prompt length, and its
prompt ids, prompt position p at `ids[position(p)]` for p below `fields`, each IDB bits, beside the request's
`max_tokens`, `seed`, `temperature` and `top_p`. The ids are the scan's own reading of the certified cells, each the
decimal field of the warden's signed ingress row its request came in on, so the row is held to the certified bytes by
the scan's gates and needs no other tie. A Program that reads its slot as i32s (occupancy, prompt length and NP ids;
vLLM's `ScanSlots_v2`) reads `occ`, `fields` and the ids each at its bits in the state row (`slot_row`),
zero-extended, copies with no gate, through a read of the whole state row (`state_bits`, 9,353 words at w63's class).
An empty slot's `fields` and ids aren't zeroed, nor are the ids past `fields` (they can be a request's not
anchored at b, or an earlier request's), so such a Program masks its outputs by `occ` and reads a slot's first
`fields` ids only. A window circuit of the link-window from scan bucket t0 reads the window's rows as one value
(vLLM's `SCAN_STATE`, row a for boundary a): `window_state`, the scan's rows t0 − 1 … t0 + buckets − 2, a constant
offset t0 − 1, with row −1, the first link-window's, the empty state's, since no unit registers S_{−1}.

**The Programs' row** (`prog_encode`) at bucket t is NR entries, each `read` (1 bit), n (u32) and KO output ids of IDB
bits (`entry_layout`), then zeros to a whole word: entry j the outputs of the Program in slot j of the boundary t, its
`read` 1, and all zero for an empty slot (`occ` 0 in S_{t−1}): an empty slot's outputs are masked by its occupancy word
and never reach the egress. An occupied slot whose entry is absent (`read` 0) is refused: a request no Program reads
would reach the server with no Program's outputs counting it. It is the form the Programs' outputs are tied to: a
Program gives all its outputs at its anchor, and the slot holds them in `out` through its last egress bucket, past the
link-window's end where that is later (a group's scan carries its state across its link-windows). Here a registered
row the prover commits, standing in for the anchored Programs' outputs.

**The egress view** (`view`) of S_t is `egress.EgressBucket`'s state at b = t: b, then each slot's live, q, admitted,
n and its KO output ids at `egress`'s offsets (`egress.Shape(B, lanes·r, r, NR, KO, IDB, D)`): egress bucket t reads
S_t's view, 302 words at w63's class, where reading S_t itself would hash its 9,353.

**A unit** (`step`, its reference):
1. each block's q is its first cell's ordinal; a block is empty (q = 0, every byte zero) or every cell is on q and is
   `DATA`, `END` or `PAD`; the nonempty blocks come first, q increasing;
2. a block on a live slot's q continues that slot; a block on another q is new, its q above `last`; a slot is held
   while it is live and `now` is at most admitted + D + ⌊n / r⌋, its last egress bucket (`egress.bucket_map`'s),
   so its ids stay in place until its Program has read them and its outputs until the warden has sent them; a slot
   is free when no block continues it and it isn't held, and the i-th new block takes the i-th free slot (the lowest
   free slot), its `admitted` `now`, its anchor A(`now`) and n and `out` zero;
3. a block's cells in order: its data cells (`DATA` or `END`), none once its slot is `done`, then its `PAD`s (kind,
   length and payload zero); each data cell is the next frame of its slot's request (`io_request._reading`,
   a first frame from `left` = 0, whose head's fields go into the slot), `END` exactly when it holds the request's last
   byte, its length the request's bytes in it, its slot's `cells` below C, its slot's `fields` at most KI after it,
   and its ids, each below 2^IDB, written at the slot's cell `cells` (a new slot's ids from the slot's last request
   are overwritten as its request comes in, and no Program reads past a slot's `fields`);
4. `now` + 1, `phase` + 1 wrapping to 0 at EVERY (a `phase` of EVERY or more is refused), `last` the last new block's
   q, and a slot live iff a block continued or took it or it is held;
5. a slot `due` (live in S_{t−1}, its anchor `now`) takes its n and `out` from its entry of the Programs' row; where its
   `occ` in S_{t−1} is 1 the entry is present (`read` 1) and n is at most KO; every other entry, and the entry of a
   slot whose `occ` in S_{t−1} is 0 (a due slot's included), is zero, `read` included;
6. each slot's `occ` in S_t: 1 where it is live, `done` and its anchor is S_t's `now`, else 0.
The outputs are o0, a u16 whose bit 0, ok, is 1 iff every check holds (its other 15 bits 0), o1, S_t, and o2, S_t's
egress view at t. Where a check fails, o1 and o2 are what the gates compute (`reference` gives none), and o0 refuses
them.

**The warden it serves** (`Class.serves`): a class takes every run of a warden whose B, r, lanes and timestep `every`
are its own (B, r, lanes and EVERY; the warden's window is whole timesteps, `WardenParams`), whose `ingress_cells` is
at most C, whose `ingress_buckets` is at most WITHIN (a request's `END` is in S_{A−1}, or its ingress failed), whose
delay is D (at least WITHIN + EVERY − 1, so a request's anchor is at most admitted + D, where its egress starts) and
whose `egress_cells` is at most KO + 1 (a response's ids fit `out`). The warden frees a lane after its
connection's `END` cell, at admitted + D + ⌊n / r⌋, the bucket the slot's hold ends, so NR = lanes slots suffice.
A request the class can't hold is refused, never truncated: a prompt of more than KI ids (the warden bounds a
request's cells, not its ids) at the cell that takes its `fields` past KI, and a Program that gives more than KO
outputs (`SlotEntry_v1`'s n = KO + 1 for a request that doesn't fit its window circuit's class) at its anchor, so
the link-window fails visibly at the scan's unit.

**Structural Fits.** A unit reads only bucket t's row and S_{t−1}, so the scan's carried state is its only path from
one bucket's ingress to a later bucket's gates: each step reads the state the step before wrote, whatever the arrivals
and the prompt lengths, which stay private.

**Where it runs.** `row`, `sin` and `prog` are each a registered read, an input port `Registered.checkPort` accepts:
`row` the warden's signed row t (`warden_rows.bucket_port`), `sin` the scan's state row t − 1 and `prog` the Programs'
row t. o1 and o2 are gates, never inputs: the unit's statement commits o1 as the scan's state row t and o2 as its view
row t, each registered as a read of its row (`checkPort` accepts a hidden-output statement's output row, the
verifier's PROTOCOL.md §16.13), so bucket t + 1's `sin` and egress bucket t's `state`, which read those rows, read what
bucket t computed: the carried state and the view are wires through the registered roots. The first bucket's unit,
`IngressScanFirst`, has no `sin`: its S_{−1} is the empty state, constants of the circuit, so no row the prover commits
stands in for it. Only o0 is public: a unit proof's public output is at most 16 bits, and o1 and o2 are committed.

    IngressScanBucket{B, r, lanes, C, IDB, EVERY, WITHIN, D, KO, KI}(row: Array<lanes·r·B, Byte>, sin: Array<NS>,
        prog: Array<NP>) -> (u16, Array<NS>, Array<NV>)
    IngressScanFirst{B, r, lanes, C, IDB, EVERY, WITHIN, D, KO, KI}(row: Array<lanes·r·B, Byte>, prog: Array<NP>)
        -> (u16, Array<NS>, Array<NV>)
        NS = `state_bits`: 96 + lanes·(338 + CB + (KO + C·K)·IDB) rounded up to a multiple of 16, K = (B − 9) // 11
        NP = `prog_bits`: lanes·(33 + KO·IDB) rounded up to a multiple of 16
        NV = `view_bits`: 32 + lanes·(112 + KO·IDB rounded up to a multiple of 16)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Sequence

from verity.ml.circuits.boolean.forms import ONE, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments.gates.frame_v3 import _part
from verity.core.protocols.certifier import egress as EG
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier import io_ids as II
from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier.active import DATA, HEADER, ID_DIGITS, ID_FIELD_BYTES, PAD
from verity.core.protocols.certifier.warden import END, WardenParams

BIT, BYTE = HC.BIT, HC.BYTE
#: the state's head: `now`, `phase`, `last`
HEAD = (("now", 32), ("phase", 32), ("last", 32))
#: a ratio field's seven digits' value: below 10^7
RATIO_BITS = (10 ** 7 - 1).bit_length()
#: the class's fields, the Definitions' statics in order
STATICS = ("B", "r", "lanes", "C", "IDB", "EVERY", "WITHIN", "D", "KO", "KI")


@dataclass(frozen=True)
class Class:
    """The scan's public class: B-byte cells, r a connection a bucket, `lanes` connections at once (as many request
    slots), C cells a request at most, IDB bits a prompt or output id, a timestep of EVERY buckets (a boundary at each
    bucket t ≡ 0 mod EVERY of the scan, with SLOTS = lanes Program slots) and WITHIN, the buckets a request's ingress
    may take (δ): a request admitted at s is anchored at the first boundary at or after s + WITHIN, whose Programs
    read the state before it; the warden's delay D, at least WITHIN + EVERY − 1 so that a request's anchor is at most
    admitted + D, KO outputs a request at most and KI prompt ids a request at most, at most the ids a slot holds."""

    B: int
    r: int
    lanes: int
    C: int
    IDB: int = 17
    EVERY: int = 16
    WITHIN: int = 12
    D: int = 27
    KO: int = 64
    KI: int = 1024

    def __post_init__(self) -> None:
        for name in STATICS:
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise ValueError(f"{name} is a positive integer, not {v!r}")
        if self.IDB > 32 or self.C >= 1 << 16 or self.EVERY >= 1 << 31 or self.WITHIN >= 1 << 31:
            raise ValueError(f"IDB <= 32, C < 2^16 and EVERY, WITHIN < 2^31, not IDB={self.IDB}, C={self.C}, "
                             f"EVERY={self.EVERY}, WITHIN={self.WITHIN}")
        if not self.WITHIN + self.EVERY - 1 <= self.D < 1 << 31 or self.KO >= 1 << 16:
            raise ValueError(f"WITHIN + EVERY - 1 <= D < 2^31 and KO < 2^16, not D={self.D}, KO={self.KO}")
        Q.layout(self.B)
        self.egress
        if self.KI > self.prompt_room:
            raise ValueError(f"KI <= {self.prompt_room}, the prompt ids a slot of C = {self.C} cells holds, "
                             f"not {self.KI}")

    @cached_property
    def lay(self) -> Q.Layout:
        return Q.layout(self.B)

    @cached_property
    def egress(self) -> EG.Shape:
        """The egress bucket's shape (`egress.Shape`) whose state is this scan's egress view (`view`)."""
        return EG.Shape(self.B, self.S, self.r, self.NR, self.KO, self.IDB, self.D)

    @property
    def S(self) -> int:
        return self.lanes * self.r

    @property
    def NR(self) -> int:
        return self.lanes

    @property
    def SLOTS(self) -> int:
        """The Program slots a timestep: the scan's slots, one a lane."""
        return self.lanes

    @property
    def K(self) -> int:
        """A continuation cell's prompt fields, a grid row's."""
        return self.lay.room // ID_FIELD_BYTES

    @property
    def K0(self) -> int:
        """A first cell's prompt fields at most."""
        return (self.lay.room - self.lay.start) // ID_FIELD_BYTES

    @property
    def prompt_room(self) -> int:
        """The prompt ids a slot holds: a first cell's K0 and C − 1 continuations' K."""
        return self.C * self.K - self.K + self.K0

    @property
    def CB(self) -> int:
        return self.C.bit_length()

    def anchor(self, now: int, phase: int) -> int:
        """The first boundary at or after now + WITHIN, phase being now mod EVERY (the state's), mod 2^32."""
        x = phase + self.WITHIN % self.EVERY
        x -= self.EVERY if x >= self.EVERY else 0
        return (now + self.WITHIN + (self.EVERY - x if x else 0)) % (1 << 32)

    def serves(self, params: WardenParams) -> None:
        """ValueError, naming the fix, unless the scan takes every run of the warden at `params` (the module
        docstring's "The warden it serves")."""
        need = {"cell_bytes": (params.cell_bytes == self.B, f"= B = {self.B}"),
                "slots": (params.slots == self.r, f"= r = {self.r}"),
                "lanes": (params.lanes == self.lanes, f"= lanes = {self.lanes}"),
                "every": (params.every == self.EVERY, f"= EVERY = {self.EVERY}"),
                "ingress_cells": (params.ingress_cells <= self.C, f"<= C = {self.C}"),
                "ingress_buckets": (params.ingress_buckets <= self.WITHIN, f"<= WITHIN = {self.WITHIN}"),
                "delay": (params.delay == self.D, f"= D = {self.D}"),
                "egress_cells": (params.egress_cells <= self.KO + 1, f"<= KO + 1 = {self.KO + 1}")}
        bad = [f"{k} {want} (it is {getattr(params, k)})" for k, (ok, want) in need.items() if not ok]
        if bad:
            raise ValueError(f"{self} doesn't serve the warden at these parameters: set " + "; ".join(bad))


def slot_layout(cls: Class) -> tuple[tuple[str, int], ...]:
    return (("live", 1), ("done", 1), ("q", 32), ("left", 32), ("cells", cls.CB), ("fields", 32), ("max_tokens", 32),
            ("seed", 32), ("temperature", RATIO_BITS), ("top_p", RATIO_BITS), ("admitted", 32), ("anchor", 32),
            ("occ", 32), ("n", 32), ("out", cls.KO * cls.IDB), ("ids", cls.C * cls.K * cls.IDB))


def _fields_bits(cls: Class) -> int:
    return sum(w for _, w in HEAD) + cls.NR * sum(w for _, w in slot_layout(cls))


def layout(cls: Class) -> dict[str, tuple[int, int]]:
    """Each field's (offset, width) in the state row: `now`, `phase`, `last`, slot j's as ``"j.name"``, then `pad`."""
    out, at = {}, 0
    for name, w in HEAD:
        out[name], at = (at, w), at + w
    for j in range(cls.NR):
        for name, w in slot_layout(cls):
            out[f"{j}.{name}"], at = (at, w), at + w
    out["pad"] = (at, state_bits(cls) - at)
    return out


def state_bits(cls: Class) -> int:
    """The state row's bits: its fields', zero-padded to whole 16-bit words (a row a statement registers)."""
    return -(-_fields_bits(cls) // 16) * 16


def entry_layout(cls: Class) -> tuple[tuple[str, int], ...]:
    """An entry of the Programs' row: `read` (1 where a Program anchored at the bucket reads the slot), n (u32) and KO
    output ids of IDB bits."""
    return (("read", 1), ("n", 32), ("out", cls.KO * cls.IDB))


def prog_bits(cls: Class) -> int:
    """The Programs' row: NR entries (`entry_layout`), zero-padded to whole 16-bit words."""
    return -(-cls.NR * sum(w for _, w in entry_layout(cls)) // 16) * 16


def view_bits(cls: Class) -> int:
    """The egress view's bits (`egress.Shape.state_bits`)."""
    return cls.egress.state_bits


def position(cls: Class, p: int) -> int:
    """Prompt position p's index in a slot's `ids`: frame 0's fields first, then each continuation's K."""
    return p if p < cls.K0 else cls.K + p - cls.K0


@dataclass
class Slot:
    live: int = 0
    done: int = 0
    q: int = 0
    left: int = 0
    cells: int = 0
    fields: int = 0
    max_tokens: int = 0
    seed: int = 0
    temperature: int = 0
    top_p: int = 0
    admitted: int = 0
    anchor: int = 0
    occ: int = 0
    n: int = 0
    out: list[int] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)


@dataclass
class State:
    now: int
    phase: int
    last: int
    slots: list[Slot]


def empty(cls: Class) -> State:
    return State(0, 0, 0, [Slot(out=[0] * cls.KO, ids=[0] * (cls.C * cls.K)) for _ in range(cls.NR)])


def _put(bits: list[int], v: int, w: int) -> None:
    if not 0 <= v < 1 << w:
        raise ValueError(f"{v} is no {w}-bit field")
    bits.extend(v >> b & 1 for b in range(w))


class _Reader:
    def __init__(self, bits: Sequence[int]) -> None:
        self.bits, self.at = bits, 0

    def take(self, w: int) -> int:
        v = sum(int(self.bits[self.at + b]) << b for b in range(w))
        self.at += w
        return v


def encode(cls: Class, st: State) -> list[int]:
    bits: list[int] = []
    for v in (st.now, st.phase, st.last):
        _put(bits, v, 32)
    for s in st.slots:
        for name, w in slot_layout(cls)[:-2]:
            _put(bits, getattr(s, name), w)
        if len(s.out) != cls.KO or len(s.ids) != cls.C * cls.K:
            raise ValueError(f"a slot holds {cls.KO} outputs and {cls.C * cls.K} ids, not {len(s.out)}, {len(s.ids)}")
        for v in (*s.out, *s.ids):
            _put(bits, v, cls.IDB)
    return bits + [0] * (state_bits(cls) - len(bits))


def decode(cls: Class, bits: Sequence[int]) -> State:
    if len(bits) != state_bits(cls):
        raise ValueError(f"a state is {state_bits(cls)} bits, not {len(bits)}")
    rd = _Reader(bits)
    now, phase, last, slots = rd.take(32), rd.take(32), rd.take(32), []
    for _ in range(cls.NR):
        s = Slot(**{name: rd.take(w) for name, w in slot_layout(cls)[:-2]})
        s.out = [rd.take(cls.IDB) for _ in range(cls.KO)]
        s.ids = [rd.take(cls.IDB) for _ in range(cls.C * cls.K)]
        slots.append(s)
    return State(now, phase, last, slots)


#: an entry of the Programs' row (`entry_layout`): `read`, a request's n outputs and its KO output ids (zero past them)
Entry = tuple[int, int, tuple[int, ...]]
NONE: Entry = (0, 0, ())


def entry(n: int, ids: Sequence[int]) -> Entry:
    """The entry of a Program that reads its slot and outputs n ids `ids`."""
    return 1, n, tuple(ids)


def prog_encode(cls: Class, entries: Sequence[Entry]) -> list[int]:
    """The Programs' row at a bucket: entry j the outputs of the Program anchored there that reads slot j (`entry`),
    `NONE` for a slot no Program anchored there reads."""
    if len(entries) != cls.NR:
        raise ValueError(f"a Programs' row has NR = {cls.NR} entries, not {len(entries)}")
    bits: list[int] = []
    for r, n, ids in entries:
        if len(ids) > cls.KO:
            raise ValueError(f"an entry has at most KO = {cls.KO} ids, not {len(ids)}")
        _put(bits, r, 1)
        _put(bits, n, 32)
        for v in (*ids, *[0] * (cls.KO - len(ids))):
            _put(bits, v, cls.IDB)
    return bits + [0] * (prog_bits(cls) - len(bits))


def prog_decode(cls: Class, bits: Sequence[int]) -> list[Entry]:
    if len(bits) != prog_bits(cls):
        raise ValueError(f"a Programs' row is {prog_bits(cls)} bits, not {len(bits)}")
    rd = _Reader(bits)
    return [(rd.take(1), rd.take(32), tuple(rd.take(cls.IDB) for _ in range(cls.KO))) for _ in range(cls.NR)]


def view(cls: Class, st: State, b: int) -> EG.State:
    """S_t's egress view at bucket b = t (`egress.State`): each slot's live, q, admitted, n and output ids."""
    return EG.State(b, tuple(EG.Slot(s.live, s.q, s.admitted, s.n, tuple(s.out)) for s in st.slots))


def view_encode(cls: Class, st: State, b: int) -> list[int]:
    return EG.state_bits(cls.egress, view(cls, st, b))


def due(cls: Class, st: State) -> list[bool]:
    """The slots the Programs anchored at the bucket after `st` read: live, their anchor its bucket."""
    return [s.live == 1 and st.now == s.anchor for s in st.slots]


def held(cls: Class, st: State) -> list[bool]:
    """The slots held at the bucket after `st`: live, up to their last egress bucket admitted + D + ⌊n / r⌋ (the
    egress map's), at or after their anchor since D ≥ WITHIN + EVERY − 1."""
    return [s.live == 1 and st.now <= s.admitted + cls.D + s.n // cls.r for s in st.slots]


def _head_values(lay: Q.Layout, payload: bytes) -> dict[str, int]:
    """A first frame's max_tokens, temperature, top_p and seed (`io_request.FIELDS`' first four spans)."""
    (o0, w0, _), (o1, w1, _), (o2, w2, _), (o3, w3, _) = lay.spans[1:5]
    ratio = lambda f: int(f[:1] + f[2:])  # noqa: E731
    return {"max_tokens": int(payload[o0:o0 + w0]), "temperature": ratio(payload[o1:o1 + w1]),
            "top_p": ratio(payload[o2:o2 + w2]), "seed": int(payload[o3:o3 + w3])}


def step(cls: Class, row: bytes, st: State, prog: Sequence[Entry] | None = None) -> State | None:
    """S_t from bucket t's ingress row, S_{t−1} and the Programs' row at t (none: every entry `NONE`), the module
    docstring's "A unit", or None where a check fails."""
    B, r, lay = cls.B, cls.r, cls.lay
    if len(row) != cls.S * B:
        raise ValueError(f"a row is {cls.S * B} bytes, not {len(row)}")
    prog = [(x, n, tuple(ids) + (0,) * (cls.KO - len(ids))) for x, n, ids in (prog or [NONE] * cls.NR)]
    if len(prog) != cls.NR:
        raise ValueError(f"a Programs' row has NR = {cls.NR} entries, not {len(prog)}")
    if st.phase >= cls.EVERY:
        return None
    dues = due(cls, st)
    reads = [d and s.occ & 1 for d, s in zip(dues, st.slots)]
    if any(not x if o else x or n or any(ids) for o, (x, n, ids) in zip(reads, prog)):
        return None
    if any(n > cls.KO for _x, n, _ids in prog):
        return None
    blocks = [[row[(p * r + i) * B:(p * r + i + 1) * B] for i in range(r)] for p in range(cls.lanes)]
    qs = []
    for blk in blocks:
        q = HEADER.unpack_from(blk[0])[1]
        if q == 0 and any(any(c) for c in blk):
            return None
        if q and any(HEADER.unpack_from(c)[1] != q or c[0] not in (DATA, END, PAD) for c in blk):
            return None
        qs.append(q)
    for a, b in zip(qs, qs[1:]):
        if (not a and b) or (a and b and a >= b):
            return None
    match = [[q != 0 and s.live == 1 and s.q == q for s in st.slots] for q in qs]
    if any(sum(m) > 1 for m in match):
        return None
    new = [q != 0 and not any(m) for q, m in zip(qs, match)]
    if any(n and q <= st.last for n, q in zip(new, qs)):
        return None
    hold = held(cls, st)
    free = [j for j in range(cls.NR) if not any(m[j] for m in match) and not hold[j]]
    sel: list[int | None] = []
    for p in range(cls.lanes):
        if any(match[p]):
            sel.append(match[p].index(True))
        elif new[p]:
            k = sum(new[:p])
            if k >= len(free):
                return None
            sel.append(free[k])
        else:
            sel.append(None)
    phase = st.phase + 1 if st.phase + 1 < cls.EVERY else 0
    out = State((st.now + 1) % (1 << 32), phase, st.last,
                [replace(s, live=int(h), out=list(s.out), ids=list(s.ids)) for s, h in zip(st.slots, hold)])
    for p, q in enumerate(qs):
        if new[p]:
            out.last = q
    for p, j in enumerate(sel):
        if j is None:
            continue
        old = st.slots[j]
        cur = replace(old, out=list(old.out), ids=list(old.ids)) if not new[p] else \
            Slot(admitted=st.now, anchor=cls.anchor(st.now, st.phase), out=[0] * cls.KO, ids=list(old.ids))
        padded = False
        for c in blocks[p]:
            kind, _, n = HEADER.unpack_from(c)
            payload = c[HEADER.size:]
            if kind == PAD:
                if n or any(payload):
                    return None
                padded = True
                continue
            if padded or cur.done or cur.cells >= cls.C:
                return None
            first = cur.left == 0
            fr = Q.parse_frame(lay, payload, first, cur.left)
            if fr is None:
                return None
            rest = (fr.length if first else cur.left) - fr.used
            if (kind == END) != (rest == 0) or n != fr.used:
                return None
            base = lay.start if first else 0
            vals = [int(payload[base + ID_FIELD_BYTES * k + 1:base + ID_FIELD_BYTES * (k + 1)]) if k < fr.fields else 0
                    for k in range(cls.K)]
            if any(v >= 1 << cls.IDB for v in vals):
                return None
            if first:
                for name, v in _head_values(lay, payload).items():
                    setattr(cur, name, v)
                cur.fields = fr.fields
            else:
                cur.fields = (cur.fields + fr.fields) % (1 << 32)
            if cur.fields > cls.KI:
                return None
            cur.ids[cur.cells * cls.K:(cur.cells + 1) * cls.K] = vals
            cur.cells, cur.left, cur.done = cur.cells + 1, rest, int(rest == 0)
        cur.live, cur.q = 1, qs[p]
        out.slots[j] = cur
    for j, (d, (_x, n, ids)) in enumerate(zip(dues, prog)):
        if d:
            out.slots[j].n, out.slots[j].out = n, list(ids)
    for s in out.slots:
        s.occ = int(s.live == 1 and s.done == 1 and s.anchor == out.now)
    return out


def reference(cls: Class, row: bytes, sin: Sequence[int] | None, prog: Sequence[int]
              ) -> tuple[int, list[int] | None, list[int] | None]:
    """`IngressScanBucket`'s outputs in the clear (`IngressScanFirst`'s for sin None, the empty state): ok, and S_t
    and its egress view at t where ok is 1, None where it is 0."""
    before = empty(cls) if sin is None else decode(cls, sin)
    st = step(cls, row, before, prog_decode(cls, prog))
    if st is None:
        return 0, None, None
    return 1, encode(cls, st), view_encode(cls, st, before.now)


def programs(cls: Class, st: State, outputs: dict[int, Sequence[int]]) -> list[Entry]:
    """The Programs' row at the bucket after `st`, the honest one: each occupied slot read by its request's Program,
    whose outputs are `outputs[q]` (none for a q it lacks), `NONE` for an empty slot."""
    return [entry(len(outputs.get(s.q, ())), outputs.get(s.q, ())) if s.occ & 1 else NONE for s in st.slots]


def token_row(cls: Class, st: State) -> list[tuple[int, int, tuple[int, ...]]]:
    """The token-id row `st` gives the Programs at the bucket after it (the module docstring's): each slot's occupancy
    word, its prompt's length and its prompt ids in prompt order, (0, 0, ()) for an empty slot."""
    return [(1, s.fields, tuple(s.ids[position(cls, p)] for p in range(s.fields))) if s.occ & 1 else (0, 0, ())
            for s in st.slots]


def slot_row(cls: Class, j: int, NP: int) -> list[tuple[int, int]]:
    """Where slot j's row of NP + 2 i32s (as vLLM's `ScanSlots_v2` reads it: occupancy, prompt length, then the ids of prompt
    positions 0 … NP − 1) is in the state row: each i32's (offset, width) in bits (`layout`), its value those bits
    zero-extended, least significant first: `occ`, `fields`, then position p's id at `ids[position(p)]`."""
    if not 0 <= j < cls.NR:
        raise ValueError(f"slot {j}: the state has NR = {cls.NR} slots")
    top = cls.prompt_room
    if not 1 <= NP <= top:
        raise ValueError(f"NP = {NP}: a slot holds the ids of prompt positions 0 … {top - 1}")
    lay = layout(cls)
    at = lay[f"{j}.ids"][0]
    return [lay[f"{j}.occ"], lay[f"{j}.fields"], *((at + position(cls, p) * cls.IDB, cls.IDB) for p in range(NP))]


#: the registered value whose rows are the scan's state rows, as a window circuit reads them (`window_state`)
SCAN_STATE = "verity/certifier/scan-state/v1"


def window_state(cls: Class, rows: Sequence[Sequence[int]], t0: int, buckets: int) -> list[list[int]]:
    """The state rows a window circuit of the link-window of `buckets` buckets from scan bucket t0 reads (the module
    docstring's token-id row), row a the state boundary a's Programs read: the scan's rows t0 − 1 … t0 + buckets − 2
    of `rows` (S_0, S_1, … as the units register them), row −1 the empty state's. ValueError, naming the fix, unless
    the link-window is whole timesteps and `rows` reach its last."""
    if t0 < 0 or buckets < 1 or t0 % cls.EVERY or buckets % cls.EVERY:
        raise ValueError(f"a link-window starts at a boundary and is whole timesteps: t0 >= 0 and buckets >= 1, each "
                         f"a multiple of EVERY = {cls.EVERY}, not t0 = {t0}, buckets = {buckets}")
    if t0 + buckets - 1 > len(rows):
        raise ValueError(f"the link-window from {t0} reads state rows up to {t0 + buckets - 2}: register "
                         f"{t0 + buckets - 1} rows, not {len(rows)}")
    if any(len(rows[t]) != state_bits(cls) for t in range(max(t0 - 1, 0), t0 + buckets - 1)):
        raise ValueError(f"a state row is {state_bits(cls)} bits")
    return [encode(cls, empty(cls)) if t < 0 else list(rows[t]) for t in range(t0 - 1, t0 + buckets - 1)]


def scan(cls: Class, rows: Sequence[bytes], outputs: dict[int, Sequence[int]] | None = None,
         start: State | None = None) -> tuple[list[State], list[list[Entry]]] | None:
    """The states S_0 … S_{T−1} of the scan of `rows` from `start` (the empty state), each step's Programs' row
    `programs` of `outputs` (each request's output ids, by its ordinal), and those rows; None where a step refuses."""
    st, out, progs = start or empty(cls), [], []
    for row in rows:
        progs.append(programs(cls, st, outputs or {}))
        st = step(cls, row, st, progs[-1])
        if st is None:
            return None
        out.append(st)
    return out, progs


# --- the circuit --------------------------------------------------------------------------------------------------


def _cls(S) -> Class:
    return Class(*(getattr(S, k) for k in STATICS))


def _outs(cls: Class) -> Tuple:
    return Tuple(EG.U16, Array(state_bits(cls), BIT), Array(view_bits(cls), BIT))


def _sig(S):
    cls = _cls(S)
    return (("row", Array(cls.S * cls.B, BYTE)), ("sin", Array(state_bits(cls), BIT)),
            ("prog", Array(prog_bits(cls), BIT))), _outs(cls)


def _sig_first(S):
    cls = _cls(S)
    return (("row", Array(cls.S * cls.B, BYTE)), ("prog", Array(prog_bits(cls), BIT))), _outs(cls)


def _eq(C, x: Sequence, y: Sequence):
    return and_reduce(C, [(a ^ b).inv() for a, b in zip(x, y, strict=True)])


def _eqk(C, x: Sequence, k: int):
    return and_reduce(C, [HC._same(b, k >> j & 1) for j, b in enumerate(x)]) if k < 1 << len(x) else ZERO


def _lt(C, x: Sequence, y: Sequence):
    """x < y, unsigned, as wide as each other."""
    lt = ZERO
    for a, b in zip(x, y, strict=True):
        # from the least significant bit up: lt' = (¬a ∧ b) ∨ (¬(a ⊕ b) ∧ lt)
        lt = lt ^ C.AND(a ^ b, b ^ lt)
    return lt


def _pick(C, sels: Sequence, xs: Sequence[Sequence]) -> list:
    """Σ_i sels[i] ∧ xs[i] bitwise, the sels at most one: an AND a bit a candidate."""
    out = [ZERO] * len(xs[0])
    for s, x in zip(sels, xs, strict=True):
        out = [o ^ C.AND(s, b) for o, b in zip(out, x)]
    return out


def _split(fields: Sequence[tuple[str, int]], bits: Sequence) -> dict[str, list]:
    out, at = {}, 0
    for name, w in fields:
        out[name], at = list(bits[at:at + w]), at + w
    return out


def _cell(C, cls: Class, cell: Sequence, left: Sequence, q: Sequence):
    """One cell read against its slot's request: (its ordinal is q, data, pad, a data cell's checks (the reading, its
    kind and length, its ids below 2^IDB), ended, rest, count, the ids, the head's values, start)."""
    lay, H, room = cls.lay, HEADER.size, cls.lay.room
    kind, ordinal = cell[0], [cell[4 - j // 8][j % 8] for j in range(32)]
    length = [cell[8 - j // 8][j % 8] for j in range(32)]
    is_data, is_end, is_pad = _eqk(C, kind, DATA), _eqk(C, kind, END), _eqk(C, kind, PAD)
    data = is_data ^ is_end
    ch = Q._Chars(C, cell[H:])
    zero = [ch.eq(i, 0) for i in range(room)]
    start = and_reduce(C, [b.inv() for b in left])
    f_ok, f_end, f_rest, f_count, f_len = Q._reading(C, ch, zero, lay, True, left)
    c_ok, c_end, c_rest, c_count, _ = Q._reading(C, ch, zero, lay, False, left)
    rok, ended = mux(C, start, [f_ok, f_end], [c_ok, c_end])
    rest, count = mux(C, start, f_rest, c_rest), mux(C, start, f_count, c_count)
    used = mux(C, ended, mux(C, start, f_len, left), const_bits(room, 32))
    pad = and_reduce(C, [is_pad, *(b.inv() for b in length), *zero])
    dok = [rok, (is_end ^ ended).inv(), _eq(C, length, used)]
    ids = []
    for k in range(cls.K):
        nib = []
        for d in range(ID_DIGITS):
            cont = cell[H + ID_FIELD_BYTES * k + 1 + d][:4]
            o = H + lay.start + ID_FIELD_BYTES * k + 1 + d
            first = cell[o][:4] if lay.start + ID_FIELD_BYTES * (k + 1) <= room else [ZERO] * 4
            nib.append([C.AND(start, a ^ b) ^ b for a, b in zip(first, cont)])
        value = II._decimal(C, nib)[:32]
        here = Q._gt_const(C, count, k)
        dok.append(C.AND(here, or_reduce(C, value[cls.IDB:])).inv())
        ids.append([C.AND(here, b) for b in value[:cls.IDB]])
    (o0, w0, _), (o1, w1, _), (o2, w2, _), (o3, w3, _) = lay.spans[1:5]
    low = lambda o, w: [cell[H + o + i][:4] for i in range(w)]  # noqa: E731
    ratio = lambda o: [cell[H + o][:4]] + [cell[H + o + i][:4] for i in range(2, 8)]  # noqa: E731
    head = {"max_tokens": II._decimal(C, low(o0, w0))[:32], "temperature": II._decimal(C, ratio(o1)),
            "top_p": II._decimal(C, ratio(o2)), "seed": II._decimal(C, low(o3, w3))[:32]}
    return _eq(C, ordinal, q), data, pad, and_reduce(C, dok), ended, rest, count, ids, head, start


def _build(C, S, *e):
    cls = _cls(S)
    row, (sin, prog) = e[:cls.S * cls.B], (list(x) for x in e[cls.S * cls.B:])
    return _scan(C, cls, row, sin, prog)


def _build_first(C, S, *e):
    cls = _cls(S)
    return _scan(C, cls, e[:cls.S * cls.B], [ZERO] * state_bits(cls), list(e[cls.S * cls.B]))


def _scan(C, cls: Class, row: Sequence, sin: list, prog: list) -> list:
    B, r, L, NR, K, CB = cls.B, cls.r, cls.lanes, cls.NR, cls.K, cls.CB
    sf = slot_layout(cls)
    hw, sw = sum(w for _, w in HEAD), sum(w for _, w in sf)
    head = _split(HEAD, sin[:hw])
    slots = [_split(sf, sin[hw + j * sw:hw + (j + 1) * sw]) for j in range(NR)]
    oks = []
    blocks = [[row[(p * r + i) * B:(p * r + i + 1) * B] for i in range(r)] for p in range(L)]
    qs = [[blk[0][4 - j // 8][j % 8] for j in range(32)] for blk in blocks]
    empty_ = [and_reduce(C, [b.inv() for b in q]) for q in qs]
    for p in range(L - 1):
        oks.append(C.AND(empty_[p], empty_[p + 1].inv()).inv())
        both = C.AND(empty_[p].inv(), empty_[p + 1].inv())
        oks.append(C.AND(both, _lt(C, qs[p], qs[p + 1]).inv()).inv())
    now, phase = head["now"], head["phase"]
    oks.append(_lt(C, phase, const_bits(cls.EVERY, 32)))
    x = add(C, phase, const_bits(cls.WITHIN % cls.EVERY, 32), width=32)
    x = mux(C, _lt(C, x, const_bits(cls.EVERY, 32)), x, sub(C, x, const_bits(cls.EVERY, 32), 32)[0])
    gap = [C.AND(or_reduce(C, x), b) for b in sub(C, const_bits(cls.EVERY, 32), x, 32)[0]]
    anchor = add(C, add(C, now, const_bits(cls.WITHIN, 32), width=32), gap, width=32)
    # held: now <= admitted + D + floor(n / r), which is (now − admitted − D)·r <= n once now >= admitted + D
    W, now33 = 32 + r.bit_length(), now + [ZERO]
    held, due = [], []
    for j in range(NR):
        aD = add(C, slots[j]["admitted"], const_bits(cls.D, 32), width=33)
        pre = _lt(C, now33, aD)
        xr = EG._times(C, EG._sub(C, now33, aD)[:32], r, W)
        within = OR(C, pre, _lt(C, slots[j]["n"] + [ZERO] * (W - 32), xr).inv())
        held.append(C.AND(slots[j]["live"][0], within))
        due.append(C.AND(slots[j]["live"][0], _eq(C, now, slots[j]["anchor"])))
    entries = _split([(f"{j}.{n}", w) for j in range(NR) for n, w in entry_layout(cls)], prog)
    for j in range(NR):
        x, reads = entries[f"{j}.read"][0], C.AND(due[j], slots[j]["occ"][0])
        oks.append(C.AND(reads, x.inv()).inv())
        oks.append(C.AND(reads.inv(), or_reduce(C, [x] + entries[f"{j}.n"] + entries[f"{j}.out"])).inv())
        oks.append(_lt(C, const_bits(cls.KO, 32), entries[f"{j}.n"]).inv())
    match = [[and_reduce(C, [empty_[p].inv(), slots[j]["live"][0], _eq(C, slots[j]["q"], qs[p])]) for j in range(NR)]
             for p in range(L)]
    for p in range(L):
        for j in range(NR):
            for k in range(j + 1, NR):
                oks.append(C.AND(match[p][j], match[p][k]).inv())
    matched = [or_reduce(C, m) for m in match]
    new = [C.AND(empty_[p].inv(), matched[p].inv()) for p in range(L)]
    for p in range(L):
        oks.append(C.AND(new[p], _lt(C, head["last"], qs[p]).inv()).inv())
    free = [C.AND(or_reduce(C, [match[p][j] for p in range(L)]).inv(), held[j].inv()) for j in range(NR)]
    rw = max(L, NR).bit_length()
    new_rank, free_rank, acc = [], [], [ZERO] * rw
    for p in range(L):
        new_rank.append(acc)
        acc = add(C, acc, [new[p]], width=rw)
    acc = [ZERO] * rw
    for j in range(NR):
        free_rank.append(acc)
        acc = add(C, acc, [free[j]], width=rw)
    assign = [[and_reduce(C, [new[p], free[j], _eq(C, new_rank[p], free_rank[j])]) for j in range(NR)]
              for p in range(L)]
    for p in range(L):
        oks.append(C.AND(new[p], or_reduce(C, assign[p]).inv()).inv())
    sel = [[match[p][j] ^ assign[p][j] for j in range(NR)] for p in range(L)]
    last = head["last"]
    for p in range(L):
        last = mux(C, new[p], qs[p], last)
    gathered = ("done", "left", "cells", "fields", "max_tokens", "seed", "temperature", "top_p", "admitted", "anchor")
    finals, writes = [], []
    for p, blk in enumerate(blocks):
        cur = {n: _pick(C, match[p], [slots[j][n] for j in range(NR)]) for n in gathered}
        cur["admitted"] = [a ^ C.AND(new[p], b) for a, b in zip(cur["admitted"], now)]
        cur["anchor"] = [a ^ C.AND(new[p], b) for a, b in zip(cur["anchor"], anchor)]
        c0, padded, mask, vals = cur["cells"], ZERO, [], []
        for i, cell in enumerate(blk):
            on_q, data, pad, dok, ended, rest, count, ids, hv, start = _cell(C, cls, cell, cur["left"], qs[p])
            data = C.AND(data, empty_[p].inv())
            # an empty block's cells are B zero bytes: PAD on ordinal 0; a nonempty block's are on q, pads or data
            oks.append(C.AND(on_q, pad ^ C.AND(data, dok)))
            room = _lt(C, cur["cells"], const_bits(cls.C, CB))
            oks.append(C.AND(data, and_reduce(C, [padded.inv(), cur["done"][0].inv(), room]).inv()).inv())
            first = C.AND(data, start)
            for n, v in hv.items():
                cur[n] = mux(C, first, v, cur[n])
            total = add(C, cur["fields"], count, width=32)
            cur["fields"] = mux(C, data, mux(C, start, count, total), cur["fields"])
            oks.append(C.AND(data, _lt(C, const_bits(cls.KI, 32), cur["fields"])).inv())
            cur["cells"] = mux(C, data, add(C, cur["cells"], [ONE], width=CB), cur["cells"])
            cur["left"] = mux(C, data, rest, cur["left"])
            cur["done"] = [OR(C, cur["done"][0], C.AND(data, ended))]
            padded = OR(C, padded, pad)
            mask.append(data)
            vals.append([b for v in ids for b in v])
        finals.append(cur)
        writes.append((c0, mask, vals))
    out_slots = []
    for j in range(NR):
        s = slots[j]
        taken = or_reduce(C, [sel[p][j] for p in range(L)])
        o = {"live": [OR(C, taken, held[j])], "q": _pick(C, [sel[p][j] for p in range(L)], qs)}
        o["q"] = [a ^ C.AND(taken.inv(), b) for a, b in zip(o["q"], s["q"])]
        for n in gathered:
            got = _pick(C, [sel[p][j] for p in range(L)], [f[n] for f in finals])
            o[n] = [a ^ C.AND(taken.inv(), b) for a, b in zip(got, s[n])]
        o["ids"] = _append(C, cls, [sel[p][j] for p in range(L)], writes, s["ids"])
        kept = or_reduce(C, [assign[p][j] for p in range(L)]).inv()
        for n in ("n", "out"):
            o[n] = mux(C, due[j], entries[f"{j}.{n}"], [C.AND(kept, b) for b in s[n]])
        out_slots.append(o)
    nxt = add(C, now, [ONE], width=32)
    for o in out_slots:
        starts = and_reduce(C, [o["live"][0], o["done"][0], _eq(C, nxt, o["anchor"])])
        o["occ"] = [starts] + [ZERO] * 31
    up = add(C, phase, [ONE], width=32)
    wrap = _eqk(C, up, cls.EVERY)
    computed = nxt + [C.AND(wrap.inv(), b) for b in up] + last + [b for o in out_slots for n, _ in sf for b in o[n]]
    computed += [ZERO] * (state_bits(cls) - len(computed))
    sh = cls.egress
    seen = list(now)
    for o in out_slots:
        slot = o["live"] + [ZERO] * 15 + o["q"] + o["admitted"] + o["n"] + o["out"]
        seen += slot + [ZERO] * (sh.slot_bits - len(slot))
    return [[and_reduce(C, oks)] + [ZERO] * 15, computed, seen]


def _append(C, cls: Class, sels: Sequence, writes: Sequence, ids: Sequence) -> list:
    """Slot ids after its block's data cells are written at its cells count: the block routed to the slot (`sels`),
    its r rows shifted to the count by a barrel shifter, a row past C dropped, and each written row in place of the
    old one."""
    K, IDB, CB, Cn = cls.K, cls.IDB, cls.CB, cls.C
    w = K * IDB
    c = _pick(C, sels, [x[0] for x in writes])
    rows = [_pick(C, sels, [x[2][i] for x in writes]) for i in range(cls.r)]
    mask = [_pick(C, sels, [[x[1][i]] for x in writes])[0] for i in range(cls.r)]
    zero = [ZERO] * w
    for s in range(CB):
        sh, n = 1 << s, min(len(rows) + (1 << s), Cn)
        rows = [mux(C, c[s], rows[g - sh] if 0 <= g - sh < len(rows) else zero, rows[g] if g < len(rows) else zero)
                for g in range(n)]
        mask = [mux(C, c[s], [mask[g - sh] if 0 <= g - sh < len(mask) else ZERO],
                    [mask[g] if g < len(mask) else ZERO])[0] for g in range(n)]
    rows, mask = rows[:Cn], mask[:Cn]
    out = list(ids)
    for g in range(len(rows)):
        out[g * w:(g + 1) * w] = mux(C, mask[g], rows[g], ids[g * w:(g + 1) * w])
    return out


_STEP = ("each block continues the live slot on its ordinal or takes the lowest free slot (live and up to admitted + D "
         "+ floor(n / r) is held) as a new connection above `last`, its data cells are the next frames of the slot's "
         "warden-form id request and its ids (IDB bits each) are appended at the slot's cell count, below C, its prompt "
         "at most KI ids; a slot whose anchor is now and whose occupancy word occ has bit 0 set has its entry of prog "
         "read (its read bit 1) and takes its n, at most KO, and KO output ids from it, every other entry of prog "
         "zero; each slot's occ in S_t is 1 where it is live, its END is in and its anchor is now + 1, else 0; every "
         "switch is computed from the state, prog and the row")
IngressScanBucket = _part(
    "IngressScanBucket", STATICS, _sig, _build,
    "the scan's state S_t = `ingress_scan.step` of sin, bucket t's ingress row (lanes·r warden cells of B bytes, a "
    "block of r a live connection in ordinal order) and the Programs' row prog, and its egress view at t "
    "(`egress.EgressBucket`'s state), with 1 in bit 0 of the first output iff every check of the step holds: " + _STEP,
    version=2)
IngressScanBucket.gated = True  # type: ignore[attr-defined]  # circuit-check holds its digest pins
IngressScanFirst = _part(
    "IngressScanFirst", STATICS, _sig_first, _build_first,
    "the scan's first state S_0 = `ingress_scan.step` of the empty state (all zeros, constants), bucket 0's ingress "
    "row and the Programs' row prog, and its egress view at 0, with 1 in bit 0 of the first output iff every check of "
    "the step holds: " + _STEP, version=2)
IngressScanFirst.gated = True  # type: ignore[attr-defined]  # circuit-check holds its digest pins


def arguments(cls: Class, cases: Sequence[tuple[bytes, Sequence[int], Sequence[int]] | tuple[bytes, Sequence[int]]]
              ) -> list[list]:
    """The arguments for (row, sin, prog) cases (`IngressScanBucket`) or (row, prog) cases (`IngressScanFirst`):
    each row's bytes as bits, least significant first, and the bit rows as they are."""
    rows = [[[b >> i & 1 for i in range(8)] for b in c[0]] for c in cases]
    return [rows, *([list(c[k]) for c in cases] for k in range(1, len(cases[0])))]


__all__ = ["Class", "Entry", "IngressScanBucket", "IngressScanFirst", "NONE", "STATICS", "Slot", "State", "arguments",
           "decode", "due", "empty", "encode", "entry", "entry_layout", "held", "layout", "position", "prog_bits",
           "prog_decode", "prog_encode", "programs", "reference", "scan", "slot_row", "state_bits", "step", "token_row",
           "view", "view_bits", "view_encode", "window_state"]
