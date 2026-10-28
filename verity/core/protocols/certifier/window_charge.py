"""Advice v1's whole window charged: its decisions, the hits it sends past the pool bound and the arrival order's
deviations, every count and every value as `code_v1` codes them, with the carry chained from window to window
(`protocol/charge-chain.md`, "The whole window").

    ChargeCount_v2{N, H, O}  (book, carry, en[N], bit[N], hen[H], short[H], oen[O], p[O], r[O]) -> (ok, bits[CW], carry')
    ChargeLink_v2{N, H, O}   (book, carry_in[160], carry_out[160], en[N], bit[N], hen[H], short[H], oen[O], p[O], r[O])
                             -> (ok, bits[CW])
    carry = (n0[3], n1[3], run[2], pos[32], rank[25])

`ChargeCount_v1` (`arith_encode`) charges a window that sends only decisions: it codes `code_v1`'s four other counts
(n_hits, n_resumes, n_order, n_lag) as 0. Advice v1 never sends resumes or lag bits. The replay derives preemptions and
the certified O forces every lag row, and `v1.py` refuses a window that sends either. It does send two of the four:
past the pool bound, a shortfall for each lookup the replay reads (`hits`), and, where the arrivals deviate from release
order (the client's ingress), an order entry for each deviating arrival (`order`). This unit charges them. It counts
`code_v1._window`'s segment for the whole window with the counting encoder (`arith_encode`, "The counting variant"),
in `code_v1`'s order:
1. n_arrived, then each decision at the book's KT estimate for min(run, 2): `ChargeCount_v1`'s gates;
2. n_hits, then each hit's value (v1's sent shortfall, in hit units);
3. n_resumes, 0;
4. n_order, then each entry's position gap and rank step;
5. n_lag, 0;
6. the finish. Its count is the window's charge, which the caller holds to c_w.

**Slots.** N decision slots (en, bit), H hit slots (hen, and short, the value, 17 bits) and O order slots (oen; p, the
arrival's position, 32 bits; r, its rank, 25 bits). Each field's enabled slots hold its values in order, and a disabled
slot changes nothing. A field's count is its enabled slots. A field with no slots (H = 0, O = 0) has no inputs and
its count is 0.

**An integer** v ∈ [0, hi] is coded as `code.Binned.put(v, 0, hi)`. First its Exp-Golomb bin j = bitlen(v), the
symbol (Σ_{i<j} f_i, f_j, total) of its table. Then its offset u = v − a_j, uniform over the bin's
n_j = min(2^j − 1, hi) − a_j + 1 values (a_0 = 0, a_j = 2^(j−1)), as `code.put_uniform` codes it:
- nothing when n_j = 1;
- (u, 1, n_j) when 1 < n_j ≤ 2^16;
- (⌊u / 2^15⌋, 1, ⌊(n_j − 1) / 2^15⌋ + 1), then (u mod 2^15, 1, 2^15), when n_j > 2^16.

The circuit reads the bin by a one-hot of bitlen(v) and its (cum, freq) from per-window prefix sums of the public
table. n_j is a constant of the bin, so the one-hot selects the uniform step's total by XORs alone. Each value:
- a count of S slots: hi is the book's cap, at least 2^bitlen(S) − 1, so n_j = 2^(j−1) ≤ 2^15;
- a hit: hi = hit_units = 2^16;
- an order entry's gap g = p − pos: hi = gap = 2^24. pos is the last entry's position + 1, 0 before the session's
  first entry, so g ≥ 0 is p after it (`code_v1._State.step` + 1);
- its rank step z = zigzag(r − rank): hi = 2·index + 1 = 2^25 + 1. rank is the last entry's rank + 1, 0 before the
  first (`_State.index` + 1), and r ≤ index = 2^24. The entry's order_len, always 1 under v1, costs nothing.

**Refusals** (ok = 0): `ChargeCount_v1`'s; a carry-in rank past 2^24 + 1; an enabled hit past 2^16; an enabled entry
with p < pos, a gap past 2^24, r past 2^24 or p = 2^32 − 1 (its pos wouldn't fit); a selected cumulative frequency past
2^16; any symbol the coder refuses.

**The book** (`window_book`) is the registered book's tables at the bins a window reaches, under v1's caps
(`v1.TABLE_CAPS`): hit_units 2^16, gap and index 2^24, order_len 1, and count caps that cover each count's top bin.

**The carry** is `ChargeCount_v1`'s (the KT counts and run) and `code_v1._State`'s order step and index, as pos and
rank: 149 bits in a 160-bit row (`ROW_BITS`), chained as `charge_chain` chains v1's, under this module's own value
`VALUE`. After each hit, and after each order entry's gap and its rank step, the carried words are committed, as after
each decision, so the build stays linear in N + H + O.

**Cost** (`size`, ANDs): `ChargeCount_v2{N, 0, 0}` is `ChargeCount_v1{N}` and 24 for the carry-in's rank check: 19,560
at N = 2. A hit slot adds about 8,600: its bin step and one uniform step, about 4,000 each, and the bin's selection. An
order slot adds about 25,800: the gap's bin and two uniform steps, the rank step's three, and their selections. A count
over S ≥ 2 slots adds its offset step, as n_arrived does. `ChargeCount_v2{1, 1, 1}` is 53,287, {2, 2, 0} 39,803,
{2, 0, 2} 74,901 and {4, 2, 2} 100,586; `ChargeLink_v2` adds 160 for the link (100,746 at {4, 2, 2}). A lowering adds
the copies (`size`'s ``carry``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, reduce
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates import sha512 as SG
from verity.core.protocols.certifier import arith_encode as AE, charge_chain as CC
from verity.core.protocols.certifier.arith_encode import CW, K_MAX, MAX_TOTAL, RUNS, Tally
from verity.core.protocols.certifier.head_check import BIT, _part
from verity.core.protocols.sampled_proofs.one_stage import registered as G

VALUE = "window-charge"
#: v1's registered caps on a hit (hit units), an order entry's gap and its rank (`v1.TABLE_CAPS`)
HIT_CAP, GAP_CAP, INDEX_CAP = 1 << 16, 1 << 24, 1 << 24
ZIG_CAP = 2 * INDEX_CAP + 1
#: the widths of a hit's value, an order entry's position and its rank
HW, PW, RW = HIT_CAP.bit_length(), 32, INDEX_CAP.bit_length()
#: `code_v1._window`'s integer symbols in its order, each a table of the book
SYMBOLS = ("n_arrived", "n_hits", "hit", "n_resumes", "n_order", "order_gap", "order_index", "n_lag")
_CHUNK = 15


# --- the specification ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowBook:
    """`ChargeCount_v2`'s public tables (`window_book`): for each of `SYMBOLS`, its table's frequencies at the bins a
    window reaches and the table's total."""

    tables: tuple[tuple[tuple[int, ...], int], ...]

    def __getitem__(self, name: str) -> tuple[tuple[int, ...], int]:
        return self.tables[SYMBOLS.index(name)]


@dataclass(frozen=True)
class WindowCarry:
    """What `code_v1`'s windows carry from one to the next: `ChargeCount_v1`'s carry (each decision context's KT
    counts, and min(run, 2)), and `pos` and `rank`, the last order entry's position and rank, each + 1 (0 before the
    session's first entry)."""

    counts: tuple[tuple[int, int], ...] = ((0, 0),) * RUNS
    run: int = 0
    pos: int = 0
    rank: int = 0


@dataclass(frozen=True)
class Window:
    """What one sampling block of the network certifier is charged: advice v1's residual for the block, each field's
    values in the order its reader asks for them, as `code_v1` codes them. `arrived`, the replay's decisions; `hits`,
    each sent hit in hit units; `resumes`, the resumed rows' positions; `order`, each entry a step's position and its
    new requests' certified ingress indices; `lag`, async mode's bits. `reference_window` charges the residual v1
    sends, which has no resumes and no lag bits and one index an order entry, in slots: `arrived` in (en, bits),
    `hits` in (hen, short) and each entry (p, (r,)) in (oen, p, r). That residual is the spec's `Window` (`ds`,
    `hits`, `order`)."""

    arrived: tuple[bool, ...] = ()
    hits: tuple[int, ...] = ()
    resumes: tuple[int, ...] = ()
    order: tuple[tuple[int, tuple[int, ...]], ...] = ()
    lag: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "arrived", tuple(bool(b) for b in self.arrived))
        object.__setattr__(self, "hits", tuple(int(h) for h in self.hits))
        object.__setattr__(self, "resumes", tuple(int(p) for p in self.resumes))
        object.__setattr__(self, "order", tuple((int(s), tuple(int(i) for i in idx)) for s, idx in self.order))
        object.__setattr__(self, "lag", tuple(bool(b) for b in self.lag))


def _bins(N: int, H: int, O: int) -> list[tuple[str, int]]:
    """Each of `SYMBOLS` and the bins a window of N decisions, H hits and O order entries reaches."""
    return [("n_arrived", N.bit_length() + 1), ("n_hits", H.bit_length() + 1), ("hit", HW + 1), ("n_resumes", 1),
            ("n_order", O.bit_length() + 1), ("order_gap", GAP_CAP.bit_length() + 1),
            ("order_index", ZIG_CAP.bit_length() + 1), ("n_lag", 1)]


def first_carry(book: dict) -> WindowCarry:
    """The session's first carry: `arith_encode.first_carry`'s, and no order entry yet."""
    return WindowCarry(AE.first_carry(book).counts)


def window_book(book: dict, N: int, H: int, O: int) -> tuple[WindowBook, WindowCarry]:
    """`ChargeCount_v2{N, H, O}`'s tables from the registered book in `code_v1.Book.to_json()`'s form, and the first
    window's carry. Refuses (ValueError) caps other than v1's for the values, a count whose top bin its cap cuts short,
    and a model that isn't an Exp-Golomb table over every bin to its cap (`code_v1.fit_book`'s are)."""
    caps = book["caps"]
    fixed = {"hit_units": HIT_CAP, "gap": GAP_CAP, "index": INDEX_CAP, "order_len": 1}
    if any(caps.get(k) != v for k, v in fixed.items()):
        raise ValueError(f"the book's caps aren't v1's: {fixed}")
    for S, cap in ((N, "arrived"), (H, "hits"), (O, "order")):
        if not 0 <= S < 1 << 16 or (1 << S.bit_length()) - 1 > caps[cap]:
            raise ValueError(f"{S} slots: 0 ≤ S < 2^16, its top bin within the book's cap of {caps[cap]}")
    his = {"n_arrived": caps["arrived"], "n_hits": caps["hits"], "hit": HIT_CAP, "n_resumes": caps["resumes"],
           "n_order": caps["order"], "order_gap": GAP_CAP, "order_index": ZIG_CAP, "n_lag": caps["lag"]}
    tables = []
    for name, bins in _bins(N, H, O):
        top = his[name].bit_length()
        m = book["models"][name].get("binned")
        if m is None or m["lo"] != 0 or len(m["freqs"]) <= top:
            raise ValueError(f"the book's {name} isn't an Exp-Golomb table over bins 0 … {top}")
        freqs = [int(f) for f in m["freqs"]]
        tables.append((tuple(freqs[:bins]), sum(freqs[:top + 1])))
    return WindowBook(tuple(tables)), first_carry(book)


def _offset(j: int, hi: int) -> tuple[int, int]:
    """Bin j's first value a_j and its n_j values in [0, hi]."""
    a = 0 if j == 0 else 1 << (j - 1)
    return a, min((1 << j) - 1, hi) - a + 1


def _uniform(u: int, n: int) -> list[tuple[int, int, int]]:
    """`code.put_uniform(u, n)`'s symbols."""
    if n <= MAX_TOTAL:
        return [(u, 1, n)] if n > 1 else []
    return _uniform(u >> _CHUNK, ((n - 1) >> _CHUNK) + 1) + [(u & ((1 << _CHUNK) - 1), 1, 1 << _CHUNK)]


def binned_symbols(v: int, freqs: Sequence[int], total: int, hi: int) -> list[tuple[int, int, int]]:
    """`code.Binned.put(v, 0, hi)`'s symbols (cum, freq, total) for 0 ≤ v ≤ hi, the bin's from `freqs` (bins 0 … at
    least bitlen(v)) and `total`: the bin, then its offset uniform (the module's "An integer")."""
    j = v.bit_length()
    a, n = _offset(j, hi)
    return [(sum(freqs[:j]), freqs[j], total)] + _uniform(v - a, n)


def zigzag(v: int) -> int:
    return 2 * v if v >= 0 else -2 * v - 1


def _code(st: Tally, symbols) -> Tally:
    for s in symbols:
        st = AE.reference_count_step(K_MAX, 0, CW, st, 1, s)
    return st


def reference_window(N: int, H: int, O: int, book: WindowBook, carry: WindowCarry, en: Sequence[int],
                     bits: Sequence[int], hen: Sequence[int] = (), short: Sequence[int] = (), oen: Sequence[int] = (),
                     p: Sequence[int] = (), r: Sequence[int] = ()) -> tuple[int, int, WindowCarry]:
    """(ok, bits, carry') of one window, `code_v1._window`'s segment counted by the references: its N decision slots
    (en, bits), H hit slots (hen, short) and O order slots (oen, p, r); bits is the window's charge. A refusal returns
    ok 0, and then only ok is specified."""
    if (len(en), len(bits), len(hen), len(short), len(oen), len(p), len(r)) != (N, N, H, H, O, O, O):
        raise ValueError(f"N = {N}, H = {H} and O = {O} slots")
    if [len(t[0]) for t in book.tables] != [b for _n, b in _bins(N, H, O)]:
        raise ValueError(f"a book for N = {N}, H = {H} and O = {O}")
    refused = 0, 0, carry
    if carry.run >= RUNS or carry.rank > INDEX_CAP + 1:
        return refused

    def count(st: Tally, name: str, slots: Sequence[int]) -> Tally:
        return _code(st, binned_symbols(sum(map(int, slots)), *book[name], (1 << len(slots).bit_length()) - 1))

    st = count(Tally(), "n_arrived", en)
    ok, counts, run = 1, carry.counts, carry.run
    for e, b in zip(en, bits):
        good, f0, f1, counts = AE.reference_book_kt(RUNS, counts, e, run, b)
        ok &= good
        st = AE.reference_count_step(K_MAX, 1, CW, st, e, (b, f0, f1))
        if e:
            run = 0 if b else min(run + 1, RUNS - 1)
    st = count(st, "n_hits", hen)
    for e, s in zip(hen, short):
        if e:
            if s > HIT_CAP:
                return refused
            st = _code(st, binned_symbols(s, *book["hit"], HIT_CAP))
    st = count(st, "n_resumes", ())
    st = count(st, "n_order", oen)
    pos, rank = carry.pos, carry.rank
    for e, pp, rr in zip(oen, p, r):
        if e:
            if not pos <= pp < (1 << PW) - 1 or pp - pos > GAP_CAP or rr > INDEX_CAP:
                return refused
            st = _code(st, binned_symbols(pp - pos, *book["order_gap"], GAP_CAP))
            st = _code(st, binned_symbols(zigzag(rr - rank), *book["order_index"], ZIG_CAP))
            pos, rank = pp + 1, rr + 1
    st = count(st, "n_lag", ())
    good, length = AE.reference_count_finish(CW, st, (1 << CW) - 1)
    return ok & good, length, WindowCarry(counts, run, pos, rank)


# --- the row ----------------------------------------------------------------------------------------------------------

_CARRY = AE._CARRY + [("pos", 1, PW), ("rank", 1, RW)]
_CARRY_BITS = sum(n * w for _, n, w in _CARRY)
ROW_BITS = CC.WORD_BITS * -(-_CARRY_BITS // CC.WORD_BITS)


def encode(carry: WindowCarry) -> tuple[int, ...]:
    """The carry's `ROW_BITS` bits, `_CARRY`'s fields in order, each least significant bit first, then zeros;
    ValueError for a field that doesn't fit."""
    if len(carry.counts) != RUNS:
        raise ValueError(f"a carry is {RUNS} contexts' counts")
    vals = {"n0": [c[0] for c in carry.counts], "n1": [c[1] for c in carry.counts], "run": [carry.run],
            "pos": [carry.pos], "rank": [carry.rank]}
    bits = [b for f in AE._args(_CARRY, vals) for b in f]
    return tuple(bits + [0] * (ROW_BITS - len(bits)))


def carry_of(bits: Sequence[int]) -> WindowCarry:
    """The carry a row's bits hold, its padding ignored."""
    if len(bits) != ROW_BITS or any(b not in (0, 1) for b in bits):
        raise ValueError(f"a carry row is {ROW_BITS} bits")
    f = AE._fields(_CARRY, bits)
    return WindowCarry(tuple(zip(f["n0"], f["n1"])), f["run"][0], f["pos"][0], f["rank"][0])


def reference_link(N: int, H: int, O: int, book: WindowBook, row_in: Sequence[int], row_out: Sequence[int],
                   *slots: Sequence[int]) -> tuple[int, int]:
    """(ok, bits) of one window's charge from its carry rows: `reference_window` from `row_in`'s carry, and ok only if
    `row_out` is its carry' (`encode`)."""
    ok, length, nxt = reference_window(N, H, O, book, carry_of(row_in), *slots)
    return int(bool(ok) and tuple(row_out) == encode(nxt)), length


def session(N: int, H: int, O: int, book: dict, windows: Sequence[Sequence[Sequence[int]]]
            ) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    """The honest chain over a session's windows, each its slots (en, bits, hen, short, oen, p, r), under the
    registered book (its JSON): each window's (bits, row_in, row_out). ValueError for a window the charge refuses."""
    wb, carry = window_book(book, N, H, O)
    out = []
    for w, slots in enumerate(windows):
        ok, length, nxt = reference_window(N, H, O, wb, carry, *slots)
        if not ok:
            raise ValueError(f"window {w}'s charge is refused")
        out.append((length, encode(carry), encode(nxt)))
        carry = nxt
    return out


# --- the chain --------------------------------------------------------------------------------------------------------

def start_opening(program: str, book: dict) -> bytes:
    """b‖c of the first carry's row under its public salt (`charge_chain.start_salt` of `VALUE`)."""
    return CC._opening(CC.row(encode(first_carry(book))), CC.start_salt(program, VALUE))


def commit_window(program: str, book: dict, carry_out: WindowCarry, registration: str,
                  carry: G.Registered | None = None, salt: bytes | None = None) -> G.Registered:
    """A window's produced `VALUE`: its carry-in row and `carry_out`'s (`charge_chain.commit_rows`), the session's
    first carry-in the book's first carry."""
    return CC.commit_rows(program, VALUE, encode(first_carry(book)), encode(carry_out), registration, carry, salt)


def reads() -> dict[str, dict[str, int]]:
    """The charging unit's rows: `carry_in` 0 and `carry_out` 1."""
    return {VALUE: {"carry_in": 0, "carry_out": 1}}


def check_chain(windows: Sequence[Sequence[bytes]], program: str, book: dict) -> str | None:
    """`charge_chain.check_chain`'s CH-rows, CH-first and CH-carry over `VALUE`'s rows."""
    return CC.check_rows(windows, start_opening(program, book))


# --- the circuits ------------------------------------------------------------------------------------------------------

def book_fields(N: int, H: int, O: int) -> list[tuple[str, int, int]]:
    """The tables' public inputs, (name, elements, width): each of `SYMBOLS`' bins, then its total."""
    return [f for name, bins in _bins(N, H, O) for f in ((name, bins, 17), (name + "_total", 1, 17))]


def book_values(book: WindowBook) -> dict[str, list[int]]:
    return {k: v for name, (freqs, total) in zip(SYMBOLS, book.tables)
            for k, v in ((name, list(freqs)), (name + "_total", [total]))}


def _slots(S) -> list[tuple[str, int, int]]:
    return [("en", S.N, 1), ("bit", S.N, 1), ("hen", S.H, 1), ("short", S.H, HW), ("oen", S.O, 1), ("p", S.O, PW),
            ("r", S.O, RW)]


def _window_in(S) -> list[tuple[str, int, int]]:
    return [f for f in book_fields(S.N, S.H, S.O) + _CARRY + _slots(S) if f[1]]


def _window_out(S) -> list[tuple[str, int, int]]:
    return [("ok", 1, 1), ("bits", 1, CW)] + _CARRY


def _xor(bits):
    return reduce(lambda a, b: a ^ b, bits, ZERO)


def _prefix(C, table):
    """The table's cumulative frequencies Σ_{i<j} f_i at each bin j, in 22 bits."""
    out, run = [], [ZERO] * 22
    for fj in table:
        out.append(run)
        run = add(C, run, fj, width=22)
    return out


def _plan(j: int, hi: int) -> tuple[int, bool]:
    """Bin j's first uniform symbol's total, and whether a second (u mod 2^15, 1, 2^15) follows it."""
    _a, n = _offset(j, hi)
    if n <= MAX_TOTAL:
        return n, False
    return ((n - 1) >> _CHUNK) + 1, True


def _binned(C, phase, en, v, table, cums, total, hi, tally, oks):
    """`code.Binned.put(v, 0, hi)` for a slot's W-bit v ≤ hi, which the caller holds it to; `table` and `cums` at
    bins 0 … W. A selected cumulative frequency past 2^16 appends a refusal to `oks`."""
    W = len(v)
    C.phase = phase
    t = list(v) + [ZERO]
    for k in reversed(range(W)):
        t[k] = AE._or(C, v[k], t[k + 1])
    hot = [t[0].inv()] + [t[j - 1] ^ t[j] for j in range(1, W + 1)]
    cum, freq = [ZERO] * 22, [ZERO] * 17
    for j in range(W + 1):
        cum = [a ^ C.AND(hot[j], b) for a, b in zip(cum, cums[j])]
        freq = [a ^ C.AND(hot[j], b) for a, b in zip(freq, table[j])]
    oks.append(C.AND(en, AE._nonzero(C, cum[16:])).inv())
    tally = AE._build_count_step(C, AE._NUM, *tally, [en], cum[:16], freq, total)
    C.phase = phase
    plan = [_plan(j, hi) for j in range(W + 1)]
    u = [C.AND(v[i], hot[i + 1].inv()) for i in range(W)] + [ZERO] * (16 + _CHUNK)
    wide = _xor(hot[j] for j, (_n, w) in enumerate(plan) if w)
    first = C.AND(en, _xor(hot[j] for j, (n, _w) in enumerate(plan) if n > 1))
    totals = [_xor(hot[j] for j, (n, _w) in enumerate(plan) if (n >> k) & 1) for k in range(17)]
    if any(w for _n, w in plan):
        low = [u[k] ^ C.AND(wide, u[k] ^ u[k + _CHUNK]) for k in range(16)]
    else:
        low = u[:16]
    tally = AE._build_count_step(C, AE._NUM, *tally, [first], low, const_bits(1, 17), totals)
    if any(w for _n, w in plan):
        C.phase = phase
        tally = AE._build_count_step(C, AE._NUM, *tally, [C.AND(en, wide)], u[:_CHUNK] + [ZERO], const_bits(1, 17),
                                     const_bits(1 << _CHUNK, 17))
    return tally


def _commit(C, words):
    return [SG.commit(C, x) for x in words]


def _build_window(C, S, *elements):
    f = AE._split(_window_in(S), elements)

    def wires(name):
        return [e[0] for e in f.get(name, [])]

    oks = []
    tally = AE._count(C, "n_arrived", wires("en"), f["n_arrived"], f["n_arrived_total"][0], AE._fresh_tally(), oks)
    tally, n0, n1, run = AE._decide(C, wires("en"), wires("bit"), f["n0"], f["n1"], f["run"][0], tally, oks)
    tally = AE._count(C, "n_hits", wires("hen"), f["n_hits"], f["n_hits_total"][0], tally, oks)
    if S.H:
        C.phase = "book"
        cums = _prefix(C, f["hit"])
    for e, s in zip(wires("hen"), f.get("short", [])):
        C.phase = "hit"
        oks.append(C.AND(e, C.AND(s[HW - 1], AE._nonzero(C, s[:HW - 1]))).inv())
        tally = _binned(C, "hit", e, s, f["hit"], cums, f["hit_total"][0], HIT_CAP, tally, oks)
        C.phase = "carry"
        tally = _commit(C, tally)
    tally = AE._count(C, "n_resumes", [], f["n_resumes"], f["n_resumes_total"][0], tally, oks)
    tally = AE._count(C, "n_order", wires("oen"), f["n_order"], f["n_order_total"][0], tally, oks)
    pos, rank = f["pos"][0], f["rank"][0]
    C.phase = "order"
    oks.append(C.AND(rank[RW - 1], AE._nonzero(C, rank[1:RW - 1])).inv())
    if S.O:
        C.phase = "book"
        gcums, icums = _prefix(C, f["order_gap"]), _prefix(C, f["order_index"])
    for e, pp, rr in zip(wires("oen"), f.get("p", []), f.get("r", [])):
        C.phase = "order"
        g, borrow = sub(C, pp, pos, PW)
        over = AE._or(C, AE._nonzero(C, g[25:]), C.AND(g[24], AE._nonzero(C, g[:24])))
        bad = or_reduce(C, [borrow, over, C.AND(rr[RW - 1], AE._nonzero(C, rr[:RW - 1])), and_reduce(C, pp)])
        oks.append(C.AND(e, bad).inv())
        tally = _binned(C, "order", e, g[:25], f["order_gap"], gcums, f["order_gap_total"][0], GAP_CAP, tally, oks)
        C.phase = "carry"
        tally = _commit(C, tally)
        C.phase = "order"
        d = sub(C, rr + [ZERO], rank + [ZERO], RW + 1)[0]
        z = [d[RW]] + [d[k] ^ d[RW] for k in range(RW)]
        tally = _binned(C, "order", e, z, f["order_index"], icums, f["order_index_total"][0], ZIG_CAP, tally, oks)
        C.phase = "order"
        pos = mux(C, e, add(C, pp, [ONE], width=PW), pos)
        rank = mux(C, e, add(C, rr, [ONE], width=RW), rank)
        C.phase = "carry"
        tally, pos, rank = _commit(C, tally), SG.commit(C, pos), SG.commit(C, rank)
    tally = AE._count(C, "n_lag", [], f["n_lag"], f["n_lag_total"][0], tally, oks)
    fin = AE._build_count_finish(C, SimpleNamespace(CW=CW), *tally, const_bits((1 << CW) - 1, CW))
    C.phase = "count"
    ok = and_reduce(C, oks + [fin[0][0]])
    res = {"ok": [[ok]], "bits": [fin[1]], "n0": n0, "n1": n1, "run": [run], "pos": [pos], "rank": [rank]}
    return AE._flat(_window_out(S), res)


def _check_statics(S) -> None:
    if not all(0 <= x < 1 << 16 for x in (S.N, S.H, S.O)):
        raise ValueError(f"statics {dict(S.__dict__)}: 0 ≤ N, H, O < 2^16")


def _sig_window(S):
    _check_statics(S)
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _window_in(S)), AE._type(_window_out(S))


def _link_in(S) -> list[tuple[str, int, int]]:
    rows = [("carry_in", 1, ROW_BITS), ("carry_out", 1, ROW_BITS)]
    return book_fields(S.N, S.H, S.O) + rows + [f for f in _slots(S) if f[1]]


_LINK_OUT = [("ok", 1, 1), ("bits", 1, CW)]


def _carry_elements(bits) -> list[list]:
    out, at = [], 0
    for _name, n, w in _CARRY:
        out += [list(bits[at + e * w:at + (e + 1) * w]) for e in range(n)]
        at += n * w
    return out


def _build_link(C, S, *elements):
    f = AE._split(_link_in(S), elements)
    book = [e for name, _n, _w in book_fields(S.N, S.H, S.O) for e in f[name]]
    slots = [e for name, n, _w in _slots(S) if n for e in f[name]]
    out = _build_window(C, S, *book, *_carry_elements(f["carry_in"][0]), *slots)
    C.phase = "link"
    nxt = ([b for e in out[2:] for b in e] + [ZERO] * ROW_BITS)[:ROW_BITS]
    same = and_reduce(C, [(x ^ y).inv() for x, y in zip(nxt, f["carry_out"][0])])
    return [[C.AND(out[0][0], same)], list(out[1])]


def _sig_link(S):
    _check_statics(S)
    rows = {"carry_in", "carry_out"}
    return (tuple((name, Array(w, BIT) if name in rows else Array(n, Array(w, BIT))) for name, n, w in _link_in(S)),
            AE._type(_LINK_OUT))


ChargeCount = _part(
    "ChargeCount", ("N", "H", "O"), _sig_window, _build_window,
    "one window's charge under advice v1: code_v1's segment for the whole window counted, its decisions at the book's "
    "KT estimate for min(run, 2), its hits and its order entries' gaps and rank steps each an Exp-Golomb bin and a "
    "uniform offset, and every count, from the book's public tables, the KT counts, run and the last entry's position "
    "and rank carried across windows (`window_charge.reference_window`)", version=2)

ChargeLink = _part(
    "ChargeLink", ("N", "H", "O"), _sig_link, _build_link,
    "one window's charge under advice v1 with its carry chained: ChargeCount_v2{N, H, O} from the carry in the "
    "window's committed carry-in row, ok only if its carry' is the window's carry-out row, the next window's carry-in "
    "(`window_charge.reference_link`)", version=2)


def charge_unit(N: int, H: int, O: int):
    return bind(ChargeCount, N=N, H=H, O=O)


def link_unit(N: int, H: int, O: int):
    return bind(ChargeLink, N=N, H=H, O=O)


def build_link(C, N: int, H: int, O: int, book: Sequence, row_in: Sequence, row_out: Sequence, en: Sequence,
               bits: Sequence, hen: Sequence = (), short: Sequence = (), oen: Sequence = (), p: Sequence = (),
               r: Sequence = ()) -> tuple:
    """`ChargeLink_v2{N, H, O}` inline in a caller's circuit: `book`, the elements of `book_fields(N, H, O)` in order;
    `row_in` and `row_out`, the window's carry rows' `ROW_BITS` wires each (its reads of `VALUE`'s rows 0 and 1); `en`,
    `bits`, `hen` and `oen`, one wire per slot; `short`, `p` and `r`, one word per slot. Returns (ok, bits[CW]), bits
    the window's charge, which the caller holds to c_w."""
    S = SimpleNamespace(N=N, H=H, O=O)
    words = {"en": [[e] for e in en], "bit": [[b] for b in bits], "hen": [[e] for e in hen],
             "short": [list(s) for s in short], "oen": [[e] for e in oen], "p": [list(x) for x in p],
             "r": [list(x) for x in r]}
    slots = [e for name, n, _w in _slots(S) if n for e in words[name]]
    out = _build_link(C, S, *book, list(row_in), list(row_out), *slots)
    return out[0][0], out[1]


# --- arguments and evaluation -----------------------------------------------------------------------------------------

def _slot_values(slots: Sequence[Sequence[int]]) -> dict[str, list[int]]:
    return dict(zip(("en", "bit", "hen", "short", "oen", "p", "r"), (list(s) for s in slots)))


def window_arguments(N: int, H: int, O: int, book: WindowBook, carry: WindowCarry,
                     *slots: Sequence[int]) -> list[list[int]]:
    """`ChargeCount_v2{N, H, O}`'s arguments, each a flat bit list."""
    S = SimpleNamespace(N=N, H=H, O=O)
    vals = {**book_values(book), "n0": [c[0] for c in carry.counts], "n1": [c[1] for c in carry.counts],
            "run": [carry.run], "pos": [carry.pos], "rank": [carry.rank], **_slot_values(slots)}
    return AE._args(_window_in(S), vals)


def evaluate_windows(N: int, H: int, O: int, cases: Sequence[tuple]) -> list[tuple[int, int, WindowCarry]]:
    """`ChargeCount_v2{N, H, O}` on each (book, carry, en, bits, hen, short, oen, p, r), in one evaluation."""
    S = SimpleNamespace(N=N, H=H, O=O)
    args = [list(col) for col in zip(*(window_arguments(N, H, O, *c) for c in cases))]
    out = evaluate_bits(charge_unit(N, H, O), *args)
    res = []
    for row_ in out.reshape(len(cases), -1).tolist():
        f = AE._fields(_window_out(S), row_)
        res.append((f["ok"][0], f["bits"][0],
                    WindowCarry(tuple(zip(f["n0"], f["n1"])), f["run"][0], f["pos"][0], f["rank"][0])))
    return res


def link_arguments(N: int, H: int, O: int, book: WindowBook, row_in: Sequence[int], row_out: Sequence[int],
                   *slots: Sequence[int]) -> list[list[int]]:
    """`ChargeLink_v2{N, H, O}`'s arguments, each a flat bit list."""
    if len(row_in) != ROW_BITS or len(row_out) != ROW_BITS:
        raise ValueError(f"two {ROW_BITS}-bit rows")
    S = SimpleNamespace(N=N, H=H, O=O)
    flat = AE._args(book_fields(N, H, O), book_values(book)) + [list(row_in), list(row_out)]
    return flat + AE._args([f for f in _slots(S) if f[1]], _slot_values(slots))


def evaluate_links(N: int, H: int, O: int, cases: Sequence[tuple]) -> list[tuple[int, int]]:
    """`ChargeLink_v2{N, H, O}` on each (book, row_in, row_out, en, bits, hen, short, oen, p, r), in one evaluation."""
    args = [list(col) for col in zip(*(link_arguments(N, H, O, *c) for c in cases))]
    out = evaluate_bits(link_unit(N, H, O), *args).reshape(len(cases), -1).tolist()
    return [(int(o[0]), sum(int(b) << j for j, b in enumerate(o[1:1 + CW]))) for o in out]


# --- cost -------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _staged(kind: str, N: int, H: int, O: int) -> dict[str, int]:
    sig, build = {"charge": (_sig_window, _build_window), "link": (_sig_link, _build_link)}[kind]
    S = SimpleNamespace(N=N, H=H, O=O)
    C = Circuit(cse=True)
    params, _ = sig(S)
    ins = [C.input(f"p{i}e{j}", w) for i, (_n, t) in enumerate(params) for j, w in enumerate(BL.elements(t))]
    C.output("out", [b for e in build(C, S, *ins) for b in e])
    return C.phase_counts()


def size(kind: str = "charge", N: int = 1, H: int = 0, O: int = 0) -> dict[str, int]:
    """`ChargeCount_v2{N, H, O}`'s (``charge``) or `ChargeLink_v2{N, H, O}`'s (``link``) ANDs by stage, traced on a
    counting circuit. Its ``carry`` is no AND: the copies of the carried words, a row each in a lowering."""
    out = dict(_staged(kind, N, H, O))
    out["ands"] = sum(v for k, v in out.items() if k != "carry")
    return out


__all__ = ["ChargeCount", "ChargeLink", "GAP_CAP", "HIT_CAP", "INDEX_CAP", "ROW_BITS", "SYMBOLS", "VALUE", "Window",
           "WindowBook", "WindowCarry", "ZIG_CAP", "binned_symbols", "book_fields", "book_values", "build_link",
           "carry_of", "charge_unit", "check_chain", "commit_window", "encode", "evaluate_links", "evaluate_windows",
           "first_carry", "link_arguments", "link_unit", "reads", "reference_link", "reference_window", "session",
           "size", "start_opening", "window_arguments", "window_book", "zigzag"]
