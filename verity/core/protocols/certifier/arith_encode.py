"""Advice v1's arithmetic encoder as Boolean Definitions, run forward inside the compile unit alongside the replay scan:
counting the bits it would write, which is what the composition charges, or checking them against a segment.

    ArithCountStep_v1{K, BIN, CW}  (tally, en, symbol) -> tally'
    ArithCountFinish_v1{CW}        (tally, c) -> (ok, bits)
    tally = (low[32], high[32], used, ok, bits[CW])
    BookKT_v1{C}                   (counts[C], en, ctx, bit) -> (ok, f0, f1, counts')
    ChargeCount_v1{N}              (book, carry, en[N], bit[N]) -> (ok, bits[CW], carry')
    carry = (n0[3], n1[3], run[2])

    ArithEncodeStep_v1{K, BIN}     (coder, en, symbol, seg[K]) -> (coder', n)
    ArithEncodeFinish_v1           (coder, seg[2]) -> (ok, n)
    coder = (low[32], high[32], open, x, used, ok)
    KTContext_v1{C, NW}            (counts[C], en, ctx, bit) -> (ok, f0, f1, counts')

Under advice v1 each certifier window is one terminated segment of the
32-bit Witten–Neal–Cleary arithmetic coder (`benchmarks/network_e2e/encodings/code.py`, `ArithEncoder`), as `code_v1`
codes it: the window's counts, then its values (the replay's decisions, each a bit at the registered book's KT estimate
for its context, and the arrival order's deviations and the hits sent past the pool bound, integers). The compile unit
runs the encoder forward, since which value is read next, and so its window, is the replay's position.

**The counting variant** is what the compile slice composes. The committed advice is the values themselves, which the
replay scan reads, and the charge is their coded length: the encoder runs over the same symbols and counts the
segment bits it would write, and the window's finish refuses unless the count is at most the declared charge c. A
segment is `shifts + 2` bits, or none when nothing was used (code.py's `ArithEncoder`): each scaling puts one bit in it
(a resolved bit, or a pending one the next resolution or the finish writes), and the finish two more. So the step adds
its scalings to `bits`, and the finish adds 2 when `used`, then outputs ok (`ok` and bits ≤ c) and bits. Advice v1
sends no pool header: its KV pool is registered. Within one symbol code.py's loop takes p resolutions (E1, E2: low and
high agree on their top bit) and then u underflows (E3: low's top two bits 01 and high's 10), and never a resolution
after an underflow, which leaves low's top bit 0 and high's 1. In bit positions from the top, p is the first index
where low and high differ, and u the run after it where low is 1 and high 0. So the step takes n = p + u from a
thermometer over the top K + 2 positions (three ANDs a position), refuses n > K, and gets the next interval by one
shift: low becomes 0 followed by low's low 31 bits shifted up by n, with 0s in, and high 1 followed by its shifted up
with 1s in, a log shifter of bitlen(K) levels. It has no per-scaling slots, which the segment check needs. The count
is CW bits (`CW` = 20, a window's charge below 2^20), and a count past it is refused.

**What the circuit computes, never witness.** A decision's context is the replay's position (under `code_v1`
min(run, 2), with run the decisions since the last arrival), carried in the circuit. Its (f0, f1) are the model's, from
counts that start at the registered book's public priors and that each coded decision raises in the circuit. An
integer's (cum, freq, total) are read from the book's public tables at the value's bin. A context or a frequency the
prover supplied would pick the code per value and charge any decisions next to nothing.

**The bit-checking variant** (`ArithEncodeStep_v1`, `ArithEncodeFinish_v1`) is the alternative composition, where the
advice is the coded segments: the encoder's output must be the window's committed segment. `code.ArithDecoder.finish`
takes a segment only when it is the encoder's codeword for what it decodes to, so there the forward encoding is
`decode`. The rest of this docstring is that variant's, then the two models.

**The coder's state.** `low` and `high` are code.py's, 32 bits each. code.py also counts its pending bits (the
underflow scalings not yet resolved); the circuit doesn't, because it never writes the output. A pending run's bits are
the complement of the bit that resolves it, so the circuit checks the segment in the decoder's order, one segment bit per
scaling: `open` says a pending run is open, and `x` is the segment's bit at the run's first position. `used` is
code.py's (a symbol of frequency below its total was coded), and `ok` is 0 once anything was refused. A window starts at
`fresh()`: (0, 2^32 − 1, 0, 0, 0, 1).

**The step** codes one symbol when `en` is 1 (when 0, it passes the state through and consumes nothing):
- *The symbol.* With BIN = 1 it is a decision bit with its model's frequencies f0, f1 ≥ 1, f0 + f1 ≤ 2^16, coded as
  code.py's `encode(f0 if bit else 0, f1 if bit else f0, f0 + f1)` (`code_v1.Encoder.bit`). One product and one
  division serve both ends: with r = high − low + 1 and q = ⌊r·f0 / (f0 + f1)⌋, a 1 sets low to low + q and a 0 sets
  high to low + q − 1. With BIN = 0 it is code.py's general (cum, freq, total), with two products and two divisions;
  freq = total leaves the interval and `used` as they are, as code.py's early return does.
- *The scalings*, K slots of code.py's loop: E1 (high < 2^31) and E2 (low ≥ 2^31) resolve a bit (low's top bit), E3
  (2^30 ≤ low, high < 3·2^30) adds a pending bit, and each shifts the interval. A slot that doesn't scale leaves the
  state as it is, so every later slot doesn't either. After the update the range is at least ⌊(2^30 + 2)/2^16⌋ = 2^14,
  and each scaling doubles it until it exceeds 2^31, so a symbol takes at most 18 scalings and K = 18 never refuses
  one. A symbol that would need more than K is refused.
- *The segment check.* `seg` is the window's next K segment bits, at its position (the bits the decoder would read,
  one per scaling). The encoder writes position t when a run resolves: the run's first position gets the resolving bit
  b, and every other position of the run gets ¬b. So a resolution with no open run needs seg = b there. A resolution
  that closes an open run needs seg = ¬b there, and x = b. An E3 that opens a run sets x to its segment bit, and an E3
  inside an open run needs its segment bit to be ¬x. Every segment position is checked once, when it is consumed, and a
  run's first position once more, when the run resolves.
- *Outputs*: the next state and n, the scalings, which is the segment bits consumed. The window's reader moves by n.

**The finish** is code.py's `finish`: when `used`, one more pending bit, then b_f = 0 if low < 2^30 else 1 resolves the
run. In positions, that is the run's first position (the open run's, or the first new one) gets b_f, and the two new
positions get the rest. So it needs seg[0] = b_f ⊕ open and seg[1] = ¬b_f, and, if a run was open, x = b_f. It
consumes n = 2 bits, or none when nothing was used. Its window's segment must end there: n positions after the reader's.

**The book's model** (`BookKT_v1`) is `code_v1.KT`, the estimate advice v1 codes its bits at. C contexts each count
their 0s and 1s, in KW = 15 bits, from the registered book's priors (`Book.priors`, public). Context ctx gives
f = 2n + 1 for each count, and the bit then raises its count; once 2·(n0 + n1) + 2 would pass 2^16, that is once
n0 + n1 reaches 2^15, both are halved, rounding up. The book keeps n0 + n1 < 2^15, and the unit refuses counts past it
and a context ≥ C. With `en` = 0 it passes the counts through. v1's decisions take context min(run, 2), with run the
decisions since the last arrival, carried across windows (`code_v1._State.run`): a function of the bits alone, which the
composition keeps as a 2-bit counter (C = 3). Async's lag bits take one context (C = 1).

**A window's charge** (`ChargeCount_v1{N}`) is the one call the compile unit makes a window, for a window that sends
only decisions (no hits, resumes, order entries or lag bits: the compile slice's class). Its N slots are the window's
decisions in order, `en` (the slot holds one) and `bit` (the decision), and a slot with `en` 0 changes nothing. In the
replay's units they are the ones the window's load loads (`replay_chain.ReplayLoad`). It counts
`code_v1._window`'s segment for them:
- n_arrived, the enabled slots' count, as the book's Exp-Golomb bin, one general step at the bin's (cum, freq, total)
  read from the public table by a one-hot of the count's bit length, then, once n ≥ 2, its offset in the bin, one more
  step, uniform;
- each enabled slot's decision, through `BookKT_v1{3}` at min(run, 2) and the decision step;
- `ZERO_COUNTS`' 0s, each its table's bin 0, then the finish.

It outputs ok, the window's charge (bits, which the caller holds to c_w) and the carry: the three contexts' counts and
min(run, 2), which `code_v1` carries across windows, so the next window's call takes them. The book is public
(`charge_book`, from the registered book's JSON, whose digest is registered): the tables a window of at most N decisions
reaches, and the run's first carry, the priors and run 0 (`first_carry`). A later window's carry is a value, which
`charge_chain` commits from window to window (`ChargeLink_v1{N}`). The count, the bins, the contexts and the
frequencies are the circuit's. `charge_fields(N)` lays out the public inputs as a row's fields, and `build_charge`
makes the call inline. After each decision the carried words (the count's state, the counts and run) are committed,
`sha512.commit`'s copies: a row each in a lowering and no gate in the traced circuit, so no word's linear form carries
the decisions before it and the build is linear in N.

**The KT model** (`KTContext_v1`) is the estimate advice v1 used before `code_v1`, kept for the bit-checking
variant. C contexts each count their 0s and 1s (`replay._KT`, from the registered priors). Context ctx gives f = 2n + 1
for each count, halved with rounding up while f0 + f1 > 2^16 (`scaled`), and the decision then adds to its count.
After h ≥ 1 halvings f is ⌊n / 2^(h−1)⌋ + 1, so the circuit takes the first h whose two candidates fit. With `en` = 0 it
passes the counts through. It refuses a context ≥ C and a count that would overflow NW bits. In the compile unit
the counts start public, at the registered priors, and pass from model to model, and the context index is the
replay's (plain: min(passes, 2), C = 3): neither is witness.

**Statics** (public): K, the scalings a symbol may take (18 covers every symbol); BIN, binary (1) or general (0)
symbols; CW, the count's width; C, the contexts; NW, a count's width in the KT model; and N, a window's slots (1 ≤ N < 2^16; `charge_book` also refuses an N whose top bin the book's
cap cuts short). Fields are little-endian. Both steps are specified on the states a window reaches (low ≤ high),
where code.py's loop is the one above.

**Cost** (`size`, ANDs): ArithCountStep_v1{K=18, BIN=1, CW=20} is 2,243: the interval (one 33 × 16-bit product, one
non-restoring division with 32 quotient bits, and the two ends) 1,855, the scalings at once 366 and the count 22.
{K=18, BIN=0, CW=20} is 3,963. ArithCountFinish_v1{CW=20} is 41. BookKT_v1{C=3} is 337 and {C=1} 210. One
decision, counted, is the step and the book's
model: 2,580 ANDs, and the context counter's few. ChargeCount_v1{N} is about 2,553 a decision (the two, less what
they share, and the run counter) and about 15,000 a window (n_arrived's two steps, the four zero counts and the
finish): 19,536 at N = 2, 50,664 at N = 14 and 178,978 at N = 64. A lowering adds the copies, about 175 rows a
decision (`size`'s ``carry``). ArithEncodeStep_v1{K=18, BIN=1} is 3,244: the
interval 1,855, the 18 scaling slots 1,188 and the segment check 201. {K=18, BIN=0} is 4,964, and ArithEncodeFinish_v1
6. KTContext_v1{C=3, NW=24} is 1,202.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, add, and_reduce, const_bits, multiply, mux, or_reduce, sub
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments.gates import sha512 as SG
from verity.core.protocols.certifier.head_check import BIT, _part

PREC = 32
TOP = (1 << PREC) - 1
HALF = 1 << (PREC - 1)
Q1 = 1 << (PREC - 2)
Q3 = 3 * Q1
MAX_TOTAL = 1 << 16
#: the most scalings one symbol needs: its interval keeps at least 2^14 values, and each scaling doubles it past 2^31
K_MAX = 18
#: the count's width the composition binds: a window's charge below 2^20 bits
CW = 20


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


# --- the specification -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Coder:
    """One window's encoder state (the module's docstring)."""

    low: int = 0
    high: int = TOP
    open: int = 0
    x: int = 0
    used: int = 0
    ok: int = 1


def fresh() -> Coder:
    return Coder()


def _symbol(BIN: int, sym: Sequence[int]) -> tuple[bool, int, int, int]:
    """(valid, cum, freq, total) of a symbol: (bit, f0, f1) with BIN, else (cum, freq, total)."""
    if BIN:
        bit, f0, f1 = sym
        t = f0 + f1
        return f0 >= 1 and f1 >= 1 and t <= MAX_TOTAL, (f0 if bit else 0), (f1 if bit else f0), t
    cum, freq, total = sym
    return freq >= 1 and cum + freq <= total <= MAX_TOTAL, cum, freq, total


def reference_step(K: int, BIN: int, st: Coder, en: int, sym: Sequence[int], seg: Sequence[int]) -> tuple[Coder, int]:
    """One symbol through code.py's encoder, its output checked in the decoder's order against `seg`, the window's next
    K bits (the module's docstring). Returns the next state and the bits consumed; a refusal sets `ok` to 0, and then
    only `ok` is specified."""
    low, high, op, x, used, ok = st.low, st.high, st.open, st.x, st.used, st.ok
    if en:
        valid, cum, freq, total = _symbol(BIN, sym)
        if not valid:
            return replace(st, ok=0), 0
        r = high - low + 1
        high = low + (r * (cum + freq)) // total - 1
        low = low + (r * cum) // total
        used |= int(freq != total)
    n = 0
    while True:
        if high < HALF or low >= HALF:
            b = int(low >= HALF)
            if n >= K:
                return replace(st, ok=0), 0
            s = seg[n]
            if op:
                ok &= int(s != b and x == b)
            else:
                ok &= int(s == b)
            op = 0
            low, high = low & (HALF - 1), high & (HALF - 1)
        elif low >= Q1 and high < Q3:
            if n >= K:
                return replace(st, ok=0), 0
            s = seg[n]
            if op:
                ok &= int(s != x)
            else:
                x, op = s, 1
            low, high = low - Q1, high - Q1
        else:
            break
        low, high = 2 * low, 2 * high + 1
        n += 1
    return Coder(low, high, op, x, used, ok), n


def reference_finish(st: Coder, seg: Sequence[int]) -> tuple[int, int]:
    """code.py's `finish` checked against the window's last two bits: (ok, bits consumed)."""
    if not st.used:
        return st.ok, 0
    b = int(st.low >= Q1)
    ok = st.ok & int(seg[0] == b ^ st.open and seg[1] != b)
    if st.open:
        ok &= int(st.x == b)
    return ok, 2


def scaled(f0: int, f1: int) -> tuple[int, int]:
    """The replay's KT estimate (`replay._KT`) halved, rounding up, into the coder's total."""
    while f0 + f1 > MAX_TOTAL:
        f0, f1 = (f0 + 1) >> 1, (f1 + 1) >> 1
    return f0, f1


def reference_kt(C: int, NW: int, counts: Sequence[tuple[int, int]], en: int, ctx: int,
                 bit: int) -> tuple[int, int, int, tuple[tuple[int, int], ...]]:
    """(ok, f0, f1, counts') of one decision under the replay's KT counts per context, scaled as advice v1 does. A
    refusal returns ok 0, and then only ok is specified."""
    counts = tuple(tuple(c) for c in counts)
    if not en:
        return 1, 0, 0, counts
    if ctx >= C:
        return 0, 0, 0, counts
    n0, n1 = counts[ctx]
    f0, f1 = scaled(2 * n0 + 1, 2 * n1 + 1)
    m0, m1 = n0 + (not bit), n1 + bool(bit)
    if max(m0, m1) >> NW:
        return 0, 0, 0, counts
    return 1, f0, f1, counts[:ctx] + ((m0, m1),) + counts[ctx + 1:]


@dataclass(frozen=True)
class Tally:
    """The counting encoder's state: code.py's interval and `used`, `ok`, and `bits`, the segment bits counted."""

    low: int = 0
    high: int = TOP
    used: int = 0
    ok: int = 1
    bits: int = 0


def _scale(low: int, high: int) -> tuple[int, int, int]:
    """code.py's loop: the interval after it and its scalings."""
    n = 0
    while True:
        if high < HALF or low >= HALF:
            low, high = low & (HALF - 1), high & (HALF - 1)
        elif low >= Q1 and high < Q3:
            low, high = low - Q1, high - Q1
        else:
            return low, high, n
        low, high, n = 2 * low, 2 * high + 1, n + 1


def reference_count_step(K: int, BIN: int, CW: int, st: Tally, en: int, sym: Sequence[int]) -> Tally:
    """One symbol through code.py's encoder, its scalings added to `bits` (each puts one bit in the segment: a resolved
    bit, or a pending one the next resolution or the finish writes). A refusal (a symbol code.py refuses, more than K
    scalings, or `bits` past CW bits) sets `ok` to 0, and then only `ok` is specified."""
    low, high, used = st.low, st.high, st.used
    if en:
        valid, cum, freq, total = _symbol(BIN, sym)
        if not valid:
            return replace(st, ok=0)
        r = high - low + 1
        low, high = low + (r * cum) // total, low + (r * (cum + freq)) // total - 1
        used |= int(freq != total)
    low, high, n = _scale(low, high)
    if n > K or (st.bits + n) >> CW:
        return replace(st, ok=0)
    return Tally(low, high, used, st.ok, st.bits + n)


def reference_count_finish(CW: int, st: Tally, c: int) -> tuple[int, int]:
    """code.py's `finish` counted: (ok, bits), with bits the window's charge, the segment's length (two more when
    `used`), and ok 0 unless it is at most the declared charge c."""
    total = st.bits + 2 * st.used
    return st.ok & int(total <= c and not total >> CW), total & ((1 << CW) - 1)


def count_window(K: int, BIN: int, symbols: Sequence[Sequence[int]], CW: int = CW) -> int | None:
    """The window's segment length by the references (en = 1 each), None on a refusal."""
    st = Tally()
    for s in symbols:
        st = reference_count_step(K, BIN, CW, st, 1, s)
    ok, bits = reference_count_finish(CW, st, (1 << CW) - 1)
    return bits if ok else None


#: a count's width in `BookKT_v1`: the book keeps 2·(n0 + n1) + 2 ≤ 2^16, so n0 + n1 < 2^15
KW = 15


def reference_book_kt(C: int, counts: Sequence[tuple[int, int]], en: int, ctx: int,
                      bit: int) -> tuple[int, int, int, tuple[tuple[int, int], ...]]:
    """(ok, f0, f1, counts') of one bit under the registered book's KT model (`code_v1.KT`): f = 2n + 1 for each count
    of context ctx, then the bit's count raised and, once 2·(n0 + n1) + 2 would pass 2^16, both halved, rounding up. A
    refusal (ctx ≥ C, or counts the book can't hold) returns ok 0, and then only ok is specified."""
    counts = tuple(tuple(c) for c in counts)
    if not en:
        return 1, 0, 0, counts
    if ctx >= C:
        return 0, 0, 0, counts
    n0, n1 = counts[ctx]
    if n0 + n1 >> KW:
        return 0, 0, 0, counts
    m = [n0, n1]
    m[bool(bit)] += 1
    if (m[0] + m[1]) >> KW:
        m = [(m[0] + 1) >> 1, (m[1] + 1) >> 1]
    return 1, 2 * n0 + 1, 2 * n1 + 1, counts[:ctx] + (tuple(m),) + counts[ctx + 1:]


def check_window(K: int, BIN: int, symbols: Sequence[Sequence[int]], segment: Sequence[int]) -> bool:
    """The window's segment against its symbols (en = 1 each), by the reference: every step and the finish accept, and
    the segment ends where the finish does."""
    st, pos = fresh(), 0
    seg = list(segment)
    for s in symbols:
        st, n = reference_step(K, BIN, st, 1, s, (seg[pos:pos + K] + [0] * K)[:K])
        pos += n
        if not st.ok or pos > len(seg):
            return False
    ok, n = reference_finish(st, (seg[pos:pos + 2] + [0, 0])[:2])
    return bool(ok) and pos + n == len(seg)


#: the counts a window that sends only decisions codes after them, each 0, in `code_v1._window`'s order
ZERO_COUNTS = ("n_hits", "n_resumes", "n_order", "n_lag")
#: the decisions' contexts, min(run, 2)
RUNS = 3


@dataclass(frozen=True)
class ChargeBook:
    """What `ChargeCount_v1{N}` reads of the registered book, public (`charge_book`): `arrived`, n_arrived's bin
    frequencies at bins 0 … bitlen(N), and `total`, its table's total; `zeros`, each of `ZERO_COUNTS`' (freq, total) at
    bin 0."""

    arrived: tuple[int, ...]
    total: int
    zeros: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Carry:
    """What `code_v1`'s windows carry from one to the next for the decisions: each context's KT counts (n0, n1), and
    min(run, 2). The run's first window starts at the book's priors and run 0 (`charge_book`)."""

    counts: tuple[tuple[int, int], ...] = ((0, 0),) * RUNS
    run: int = 0


def charge_book(book: dict, N: int) -> tuple[ChargeBook, Carry]:
    """`ChargeCount_v1{N}`'s public inputs from the registered book in `code_v1.Book.to_json()`'s form, the JSON its
    digest names: the tables at the bins a window of at most N decisions reaches, and the first window's carry. Refuses
    (ValueError) an integer model that isn't an Exp-Golomb table over every bin to its cap (`code_v1.fit_book`'s are), and
    an N whose top bin the cap cuts short."""
    caps = book["caps"]

    def table(name: str) -> tuple[list[int], int]:
        top = caps[name[2:]].bit_length()
        m = book["models"][name].get("binned")
        if m is None or m["lo"] != 0 or len(m["freqs"]) <= top:
            raise ValueError(f"the book's {name} isn't an Exp-Golomb table over bins 0 … {top}")
        return [int(f) for f in m["freqs"]], sum(int(f) for f in m["freqs"][:top + 1])

    if not 1 <= N < 1 << 16 or (1 << N.bit_length()) - 1 > caps["arrived"]:
        raise ValueError(f"N = {N}: 1 ≤ N < 2^16, its top bin within the book's cap of {caps['arrived']} decisions")
    freqs, total = table("n_arrived")
    zeros = tuple((t[0][0], t[1]) for t in map(table, ZERO_COUNTS))
    return ChargeBook(tuple(freqs[:N.bit_length() + 1]), total, zeros), first_carry(book)


def first_carry(book: dict) -> Carry:
    """The session's first carry: the book's priors for the decisions' contexts (`code_v1.Book.priors`), and run 0."""
    return Carry(tuple(tuple(int(x) for x in book["priors"].get(f"arrived/{c}", (0, 0))) for c in range(RUNS)), 0)


def reference_charge(N: int, book: ChargeBook, carry: Carry, en: Sequence[int],
                     bits: Sequence[int]) -> tuple[int, int, Carry]:
    """(ok, bits, carry') of one window that sends only decisions, `code_v1`'s segment counted by the references:
    n_arrived (the enabled slots), each enabled slot's decision at the book's estimate for its context min(run, 2),
    `ZERO_COUNTS`' 0s, and the finish; bits is the window's charge. A refusal returns ok 0, and then only ok is
    specified."""
    if len(en) != N or len(bits) != N or len(book.arrived) != N.bit_length() + 1:
        raise ValueError(f"N = {N} slots and bins 0 … bitlen(N)")
    n = sum(map(int, en))
    j = n.bit_length()
    st = reference_count_step(K_MAX, 0, CW, Tally(), 1, (sum(book.arrived[:j]), book.arrived[j], book.total))
    if j >= 2:
        st = reference_count_step(K_MAX, 0, CW, st, 1, (n - (1 << (j - 1)), 1, 1 << (j - 1)))
    ok, counts, run = int(carry.run < RUNS), carry.counts, carry.run
    for e, b in zip(en, bits):
        good, f0, f1, counts = reference_book_kt(RUNS, counts, e, run, b)
        ok &= good
        st = reference_count_step(K_MAX, 1, CW, st, e, (b, f0, f1))
        if e:
            run = 0 if b else min(run + 1, RUNS - 1)
    for freq, total in book.zeros:
        st = reference_count_step(K_MAX, 0, CW, st, 1, (0, freq, total))
    good, length = reference_count_finish(CW, st, (1 << CW) - 1)
    return ok & good, length, Carry(counts, run)


# --- the circuits ----------------------------------------------------------------------------------------------------

_CODER = [("low", 1, PREC), ("high", 1, PREC), ("open", 1, 1), ("x", 1, 1), ("used", 1, 1), ("ok", 1, 1)]


def _step_in(S) -> list[tuple[str, int, int]]:
    sym = [("bit", 1, 1), ("f0", 1, 16), ("f1", 1, 16)] if S.BIN else [("cum", 1, 16), ("freq", 1, 17), ("total", 1, 17)]
    return _CODER + [("en", 1, 1)] + sym + [("seg", 1, S.K)]


def _step_out(S) -> list[tuple[str, int, int]]:
    return _CODER + [("n", 1, _bitlen(S.K))]


_FIN_IN = _CODER + [("seg", 1, 2)]
_FIN_OUT = [("ok", 1, 1), ("n", 1, 2)]


def _kt_in(S) -> list[tuple[str, int, int]]:
    return [("n0", S.C, S.NW), ("n1", S.C, S.NW), ("en", 1, 1), ("ctx", 1, _bitlen(S.C - 1)), ("bit", 1, 1)]


def _kt_out(S) -> list[tuple[str, int, int]]:
    return [("ok", 1, 1), ("f0", 1, 16), ("f1", 1, 16), ("n0", S.C, S.NW), ("n1", S.C, S.NW)]


def _tally(S) -> list[tuple[str, int, int]]:
    return [("low", 1, PREC), ("high", 1, PREC), ("used", 1, 1), ("ok", 1, 1), ("bits", 1, S.CW)]


def _count_in(S) -> list[tuple[str, int, int]]:
    return _tally(S) + _step_in(S)[len(_CODER):-1]


def _count_out(S) -> list[tuple[str, int, int]]:
    return _tally(S)


def _count_fin_in(S) -> list[tuple[str, int, int]]:
    return _tally(S) + [("c", 1, S.CW)]


def _count_fin_out(S) -> list[tuple[str, int, int]]:
    return [("ok", 1, 1), ("bits", 1, S.CW)]


def _bkt_in(S) -> list[tuple[str, int, int]]:
    return [("n0", S.C, KW), ("n1", S.C, KW), ("en", 1, 1), ("ctx", 1, _bitlen(S.C - 1)), ("bit", 1, 1)]


def _bkt_out(S) -> list[tuple[str, int, int]]:
    return [("ok", 1, 1), ("f0", 1, KW + 1), ("f1", 1, KW + 1), ("n0", S.C, KW), ("n1", S.C, KW)]


_CARRY = [("n0", RUNS, KW), ("n1", RUNS, KW), ("run", 1, 2)]


def _charge_in(S) -> list[tuple[str, int, int]]:
    z = len(ZERO_COUNTS)
    return ([("arrived", S.N.bit_length() + 1, 17), ("total", 1, 17), ("zfreq", z, 17), ("ztotal", z, 17)] + _CARRY
            + [("en", S.N, 1), ("bit", S.N, 1)])


def _charge_out(S) -> list[tuple[str, int, int]]:
    return [("ok", 1, 1), ("bits", 1, CW)] + _CARRY


def _type(fields) -> Tuple:
    return Tuple(*[Array(n, Array(w, BIT)) for _, n, w in fields])


def _split(fields, elements) -> dict:
    out, at = {}, 0
    for name, n, _w in fields:
        out[name] = [list(e) for e in elements[at:at + n]]
        at += n
    return out


def _flat(fields, res: dict) -> list:
    return [e for name, _n, _w in fields for e in res[name]]


def _or(C, a, b):
    return a ^ b ^ C.AND(a, b)


def _pad(x, w):
    return (list(x) + [ZERO] * w)[:w]


def _nonzero(C, x):
    return or_reduce(C, x)


def _divide(C, num, den, qbits):
    """⌊num / den⌋ in `qbits` bits for 1 ≤ den and num < den · 2^qbits, by non-restoring division: the partial remainder
    R stays in [−den, den), each step adds den when R < 0 and subtracts it otherwise, and quotient bit i is R ≥ 0 after
    step i, the restoring quotient's bit."""
    dw = len(den)
    W = dw + 2
    num = _pad(num, qbits + dw)
    rem, neg = _pad(num[qbits:], W), ZERO
    d = _pad(den, W)
    q = [ZERO] * qbits
    for i in reversed(range(qbits)):
        t = [num[i]] + rem[:W - 1]
        rem = add(C, t, [b ^ neg.inv() for b in d], cin=neg.inv(), width=W)
        neg = rem[W - 1]
        q[i] = neg.inv()
    return q


def _le_pow2(C, x, k):
    """x ≤ 2^k for a (k+1)-bit x."""
    return C.AND(x[k], _nonzero(C, x[:k])).inv()


def _interval(C, S, f, low, high, used, ok, en):
    """The symbol's subinterval (the module's "The step"): (low, high, used, ok) after it."""
    C.phase = "interval"
    r = add(C, sub(C, high, low, PREC)[0], [ONE], width=PREC + 1)
    if S.BIN:
        bit, f0, f1 = f["bit"][0][0], f["f0"][0], f["f1"][0]
        total = add(C, f0, f1, width=17)
        valid = and_reduce(C, [_nonzero(C, f0), _nonzero(C, f1), _le_pow2(C, total, 16)])
        q = _divide(C, multiply(C, r, f0), total, PREC)
        up = add(C, low, q, width=PREC)
        e1, e0 = C.AND(en, bit), C.AND(en, bit.inv())
        low = mux(C, e1, up, low)
        high = mux(C, e0, sub(C, up, [ONE], PREC)[0], high)
        used = _or(C, used, en)
    else:
        cum, freq, total = f["cum"][0], f["freq"][0], f["total"][0]
        cf = add(C, cum, freq, width=18)
        valid = and_reduce(C, [_nonzero(C, freq), sub(C, total + [ZERO], cf, 18)[1].inv(), _le_pow2(C, total, 16)])
        teff = mux(C, en, total, const_bits(1, 17))
        q_lo = _divide(C, multiply(C, r, [C.AND(en, b) for b in cum]), teff, PREC)
        q_hi = _divide(C, multiply(C, r, mux(C, en, cf[:17], const_bits(1, 17))), teff, PREC + 1)
        low_ = add(C, low, q_lo, width=PREC)
        high = sub(C, add(C, low, q_hi, width=PREC + 1), [ONE], PREC + 1)[0][:PREC]
        low = low_
        diff = or_reduce(C, [a ^ b for a, b in zip(freq, total)])
        used = _or(C, used, C.AND(en, diff))
    ok = C.AND(ok, C.AND(en, valid.inv()).inv())
    return low, high, used, ok


def _flags(C, low, high):
    """One scaling slot's case: e12 (E1 or E2 resolves a bit), e3 (an underflow) and s, whether it scales at all."""
    e12 = (low[31] ^ high[31]).inv()
    e3 = C.AND(C.AND(e12.inv(), low[30]), high[30].inv())
    return e12, e3, e12 ^ e3


def _shift(C, s, e3, low, high):
    l30, h30 = low[30], high[30]
    return mux(C, s, [ZERO] + low[:30] + [l30 ^ e3], low), mux(C, s, [ONE] + high[:30] + [h30 ^ e3], high)


def _more(C, low, high):
    """The interval would scale again: past the K slots, a refusal."""
    return (low[31] ^ high[31]).inv() ^ C.AND(C.AND((low[31] ^ high[31]), low[30]), high[30].inv())


def _scalings(therm, w):
    """The count of a thermometer (slot j scales only after slot j − 1 did) in w bits, by XORs alone."""
    n = [ZERO] * w
    for j, t in enumerate(therm):
        for k in range(w):
            if ((j >> k) ^ ((j + 1) >> k)) & 1:
                n[k] = n[k] ^ t
    return n


def _build_step(C, S, *elements):
    f = _split(_step_in(S), elements)
    low, high = f["low"][0], f["high"][0]
    op, x, used, ok = f["open"][0][0], f["x"][0][0], f["used"][0][0], f["ok"][0][0]
    en, seg = f["en"][0][0], f["seg"][0]
    low, high, used, ok = _interval(C, S, f, low, high, used, ok, en)

    therm, bad = [], []
    for j in range(S.K):
        C.phase = "scale"
        l31 = low[31]
        e12, e3, s = _flags(C, low, high)
        C.phase = "check"
        sj = seg[j]
        bad.append(C.AND(e12, _or(C, sj ^ l31 ^ op, C.AND(op, x ^ l31))))
        bad.append(C.AND(C.AND(e3, op), (sj ^ x).inv()))
        start = C.AND(e3, op.inv())
        x = x ^ C.AND(start, sj ^ x)
        op = _or(C, C.AND(op, e12.inv()), e3)
        C.phase = "scale"
        low, high = _shift(C, s, e3, low, high)
        therm.append(s)
    C.phase = "check"
    ok = C.AND(ok, _or(C, or_reduce(C, bad), _more(C, low, high)).inv())
    res = {"low": [low], "high": [high], "open": [[op]], "x": [[x]], "used": [[used]], "ok": [[ok]],
           "n": [_scalings(therm, _bitlen(S.K))]}
    return _flat(_step_out(S), res)


def _normalize(C, K, low, high):
    """code.py's loop at once (the module's "The counting variant"): the interval after it, n its scalings, and
    whether it takes more than K. Index i is bit 31 − i. F[i]: the first difference is at i; A: no difference through
    i; B: past the first difference, every index since it has low 1 and high 0. Each pair joined is disjoint, so XOR
    joins it."""
    C.phase = "scale"
    a, A, F = ONE, [], []
    for i in range(K + 1):
        F.append(C.AND(a, low[31 - i] ^ high[31 - i]))
        a = a ^ F[i]
        A.append(a)
    b, B = ZERO, [ZERO]
    for i in range(1, K + 2):
        b = C.AND(b ^ F[i - 1], C.AND(low[31 - i], high[31 - i].inv()))
        B.append(b)
    T = [A[j] ^ B[j + 1] for j in range(K + 1)]
    n = _scalings(T[:K], _bitlen(K))
    lo, hi = low[:31], high[:31]
    for k, s in enumerate(n):
        sh = 1 << k
        lo = mux(C, s, [ZERO] * sh + lo[:31 - sh], lo)
        hi = mux(C, s, [ONE] * sh + hi[:31 - sh], hi)
    return lo + [ZERO], hi + [ONE], n, T[K]


def _build_count_step(C, S, *elements):
    f = _split(_count_in(S), elements)
    low, high, used, ok = f["low"][0], f["high"][0], f["used"][0][0], f["ok"][0][0]
    en, bits = f["en"][0][0], f["bits"][0]
    low, high, used, ok = _interval(C, S, f, low, high, used, ok, en)
    low, high, n, more = _normalize(C, S.K, low, high)
    C.phase = "count"
    bits = add(C, bits, n, width=S.CW + 1)
    ok = C.AND(ok, _or(C, more, bits[S.CW]).inv())
    res = {"low": [low], "high": [high], "used": [[used]], "ok": [[ok]], "bits": [bits[:S.CW]]}
    return _flat(_count_out(S), res)


def _build_finish(C, S, *elements):
    f = _split(_FIN_IN, elements)
    low, op, x, used, ok, seg = f["low"][0], f["open"][0][0], f["x"][0][0], f["used"][0][0], f["ok"][0][0], f["seg"][0]
    b = _or(C, low[31], low[30])
    bad = or_reduce(C, [seg[0] ^ b ^ op, (seg[1] ^ b).inv(), C.AND(op, x ^ b)])
    ok = C.AND(ok, C.AND(used, bad).inv())
    return _flat(_FIN_OUT, {"ok": [[ok]], "n": [[ZERO, used]]})


def _build_count_finish(C, S, *elements):
    f = _split(_count_fin_in(S), elements)
    used, ok, bits, c = f["used"][0][0], f["ok"][0][0], f["bits"][0], f["c"][0]
    C.phase = "count"
    total = add(C, bits, [ZERO, used], width=S.CW + 1)
    over = _or(C, total[S.CW], sub(C, c, total[:S.CW], S.CW)[1])
    return _flat(_count_fin_out(S), {"ok": [[C.AND(ok, over.inv())]], "bits": [total[:S.CW]]})


def _select(C, ctx, n0, n1, nc, w):
    """The one-hot of ctx over nc contexts, whether ctx < nc, and context ctx's two counts."""
    hot = [and_reduce(C, [b if (c >> k) & 1 else b.inv() for k, b in enumerate(ctx)]) for c in range(nc)]
    in_range = or_reduce(C, hot)
    a0, a1 = [ZERO] * w, [ZERO] * w
    for c in range(nc):
        a0 = [u ^ C.AND(hot[c], v) for u, v in zip(a0, n0[c])]
        a1 = [u ^ C.AND(hot[c], v) for u, v in zip(a1, n1[c])]
    return hot, in_range, a0, a1


def _build_book_kt(C, S, *elements):
    f = _split(_bkt_in(S), elements)
    n0, n1, en, ctx, bit = f["n0"], f["n1"], f["en"][0][0], f["ctx"][0], f["bit"][0][0]
    C.phase = "select"
    hot, in_range, a0, a1 = _select(C, ctx, n0, n1, S.C, KW)
    C.phase = "update"
    s = add(C, a0, a1, width=KW + 1)
    full = and_reduce(C, s[:KW])
    m0 = add(C, a0, [bit.inv()], width=KW + 1)
    m1 = add(C, a1, [bit], width=KW + 1)
    o0 = mux(C, full, add(C, m0[1:], [m0[0]], width=KW), m0[:KW])
    o1 = mux(C, full, add(C, m1[1:], [m1[0]], width=KW), m1[:KW])
    ok = C.AND(en, _or(C, in_range.inv(), s[KW])).inv()
    out0, out1 = [], []
    for c in range(S.C):
        w = C.AND(en, hot[c])
        out0.append(mux(C, w, o0, n0[c]))
        out1.append(mux(C, w, o1, n1[c]))
    res = {"ok": [[ok]], "f0": [[en] + [C.AND(en, b) for b in a0]], "f1": [[en] + [C.AND(en, b) for b in a1]],
           "n0": out0, "n1": out1}
    return _flat(_bkt_out(S), res)


def _popcount(C, xs):
    """The number of 1s among xs, by a tree of adds."""
    nums = [[x] for x in xs]
    while len(nums) > 1:
        nxt = [add(C, a, b, width=len(a) + 1) for a, b in zip(nums[::2], nums[1::2])]
        nums = nxt + nums[len(nxt) * 2:]
    return nums[0]


_DEC, _NUM = SimpleNamespace(K=K_MAX, BIN=1, CW=CW), SimpleNamespace(K=K_MAX, BIN=0, CW=CW)


def _fresh_tally() -> list:
    return [const_bits(0, PREC), const_bits(TOP, PREC), [ZERO], [ONE], [ZERO] * CW]


def _count(C, phase, en, table, total, tally, oks):
    """The count of the enabled slots `en`, as `code_v1` codes a window's count: its Exp-Golomb bin at the table's
    (cum, freq, total), read by a one-hot of the count's bit length, then, once the count is 2 or more, its offset in
    the bin, uniform. `table`: the bins 0 … bitlen(len(en)); a selected cumulative frequency past 2^16 appends a
    refusal to `oks`. No slots: the count 0, bin 0."""
    if not en:
        return _build_count_step(C, _NUM, *tally, [ONE], [ZERO] * 16, table[0], total)
    nw = len(en).bit_length()
    C.phase = phase
    n = _pad(_popcount(C, en), nw)
    t = list(n) + [ZERO]
    for k in reversed(range(nw)):
        t[k] = _or(C, n[k], t[k + 1])
    hot = [t[0].inv()] + [t[j - 1] ^ t[j] for j in range(1, nw + 1)]
    cum, freq, run_cum = [ZERO] * 22, [ZERO] * 17, [ZERO] * 22
    for j, fj in enumerate(table):
        cum = [u ^ C.AND(hot[j], v) for u, v in zip(cum, run_cum)]
        freq = [u ^ C.AND(hot[j], v) for u, v in zip(freq, fj)]
        run_cum = add(C, run_cum, fj, width=22)
    oks.append(_nonzero(C, cum[16:]).inv())
    tally = _build_count_step(C, _NUM, *tally, [ONE], cum[:16], freq, total)
    if nw >= 2:
        C.phase = phase
        low = _pad([C.AND(n[i], hot[i + 1].inv()) for i in range(nw)], 16)
        tally = _build_count_step(C, _NUM, *tally, [t[1]], low, const_bits(1, 17), _pad(hot[1:], 17))
    return tally


def _decide(C, en, bit, n0, n1, run, tally, oks):
    """Each enabled slot's decision at the book's KT estimate for min(run, 2), the counts and run updated: (tally, n0,
    n1, run). A run past 2 appends a refusal to `oks`, as does each `BookKT_v1` refusal."""
    oks.append(C.AND(run[0], run[1]).inv())
    for e, b in zip(en, bit):
        kt = _build_book_kt(C, SimpleNamespace(C=RUNS), *n0, *n1, [e], run, [b])
        oks.append(kt[0][0])
        n0, n1 = kt[3:3 + RUNS], kt[3 + RUNS:3 + 2 * RUNS]
        tally = _build_count_step(C, _DEC, *tally, [e], [b], kt[1], kt[2])
        C.phase = "run"
        inc = [C.AND(run[0].inv(), run[1].inv()), _or(C, run[0], run[1])]
        run = mux(C, e, [C.AND(x, b.inv()) for x in inc], run)
        # committed once a decision, or each carried word's form grows with every decision before it; not `C.commit`,
        # which a gate trace (circuit-check, evaluation) takes as the word itself
        C.phase = "carry"
        tally, n0, n1 = [SG.commit(C, x) for x in tally], [SG.commit(C, x) for x in n0], [SG.commit(C, x) for x in n1]
        run = SG.commit(C, run)
    return tally, n0, n1, run


def _build_charge(C, S, *elements):
    f = _split(_charge_in(S), elements)
    en, bit = [e[0] for e in f["en"]], [e[0] for e in f["bit"]]
    oks = []
    tally = _count(C, "n_arrived", en, f["arrived"], f["total"][0], _fresh_tally(), oks)
    tally, n0, n1, run = _decide(C, en, bit, f["n0"], f["n1"], f["run"][0], tally, oks)
    for zf, zt in zip(f["zfreq"], f["ztotal"]):
        tally = _count(C, "count", [], [zf], zt, tally, oks)
    fin = _build_count_finish(C, SimpleNamespace(CW=CW), *tally, const_bits((1 << CW) - 1, CW))
    C.phase = "count"
    ok = and_reduce(C, oks + [fin[0][0]])
    return _flat(_charge_out(S), {"ok": [[ok]], "bits": [fin[1]], "n0": n0, "n1": n1, "run": [run]})


def _build_kt(C, S, *elements):
    f = _split(_kt_in(S), elements)
    n0, n1, en, ctx, bit = f["n0"], f["n1"], f["en"][0][0], f["ctx"][0], f["bit"][0][0]
    C.phase = "select"
    hot, in_range, a0, a1 = _select(C, ctx, n0, n1, S.C, S.NW)
    C.phase = "scale"
    H = max(0, S.NW - 13)
    cands, fits = [], []
    for h in range(H + 1):
        if h == 0:
            s = add(C, a0, a1, width=S.NW + 1)
            fit = sub(C, s, const_bits((1 << 15), S.NW + 1), S.NW + 1)[1]
            cand = ([ZERO] + a0, [ZERO] + a1)
        else:
            u0, u1 = a0[h - 1:], a1[h - 1:]
            w = len(u0) + 1
            s = add(C, u0, u1, width=w)
            fit = sub(C, s, const_bits(MAX_TOTAL - 1, max(w, 17)), max(w, 17))[1]
            cand = (u0, u1)
        fits.append(fit)
        cands.append((_pad(cand[0], 16), _pad(cand[1], 16)))
    g0, g1 = [ZERO] * 16, [ZERO] * 16
    prev = ZERO
    for h in range(H + 1):
        first = C.AND(fits[h], prev.inv())
        g0 = [u ^ C.AND(first, v) for u, v in zip(g0, cands[h][0])]
        g1 = [u ^ C.AND(first, v) for u, v in zip(g1, cands[h][1])]
        prev = _or(C, prev, fits[h])
    f0 = add(C, g0, [ONE], width=16)
    f1 = add(C, g1, [ONE], width=16)
    C.phase = "update"
    m0 = add(C, a0, [bit.inv()], width=S.NW + 1)
    m1 = add(C, a1, [bit], width=S.NW + 1)
    over = _or(C, m0[S.NW], m1[S.NW])
    ok = C.AND(en, _or(C, in_range.inv(), _or(C, over, prev.inv()))).inv()
    out0, out1 = [], []
    for c in range(S.C):
        w = C.AND(en, hot[c])
        out0.append(mux(C, w, m0[:S.NW], n0[c]))
        out1.append(mux(C, w, m1[:S.NW], n1[c]))
    res = {"ok": [[ok]], "f0": [[C.AND(en, b) for b in f0]], "f1": [[C.AND(en, b) for b in f1]], "n0": out0, "n1": out1}
    return _flat(_kt_out(S), res)


def _sig_step(S):
    if S.K < 1 or S.BIN not in (0, 1):
        raise ValueError(f"statics {dict(S.__dict__)}: K ≥ 1 and BIN a bit")
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _step_in(S)), _type(_step_out(S))


def _sig_finish(S):
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _FIN_IN), _type(_FIN_OUT)


def _sig_kt(S):
    if S.C < 1 or S.NW < 15:
        raise ValueError(f"statics {dict(S.__dict__)}: C ≥ 1 and NW ≥ 15")
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _kt_in(S)), _type(_kt_out(S))


def _sig_count(S):
    if not 1 <= S.K <= 30 or S.BIN not in (0, 1) or S.CW < _bitlen(S.K):
        raise ValueError(f"statics {dict(S.__dict__)}: 1 ≤ K ≤ 30, BIN a bit and CW ≥ bitlen(K)")
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _count_in(S)), _type(_count_out(S))


def _sig_count_finish(S):
    if S.CW < 2:
        raise ValueError(f"statics {dict(S.__dict__)}: CW ≥ 2")
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _count_fin_in(S)), _type(_count_fin_out(S))


def _sig_book_kt(S):
    if S.C < 1:
        raise ValueError(f"statics {dict(S.__dict__)}: C ≥ 1")
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _bkt_in(S)), _type(_bkt_out(S))


def _sig_charge(S):
    if not 1 <= S.N < 1 << 16:
        raise ValueError(f"statics {dict(S.__dict__)}: 1 ≤ N < 2^16")
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _charge_in(S)), _type(_charge_out(S))


ArithEncodeStep = _part(
    "ArithEncodeStep", ("K", "BIN"), _sig_step, _build_step,
    "one symbol through the 32-bit WNC arithmetic encoder, its output bits checked in the decoder's order against "
    "the window's next K segment bits; the next coder state and the bits consumed, the symbol's charge "
    "(`arith_encode.reference_step`)")

ArithEncodeFinish = _part(
    "ArithEncodeFinish", (), _sig_finish, _build_finish,
    "the arithmetic encoder's termination checked against the window's last two segment bits "
    "(`arith_encode.reference_finish`)")

KTContext = _part(
    "KTContext", ("C", "NW"), _sig_kt, _build_kt,
    "the replay's adaptive KT estimate for one decision's context, scaled into the coder's total as advice v1 did "
    "before code_v1, and its counts updated (`arith_encode.reference_kt`)")

ArithCountStep = _part(
    "ArithCountStep", ("K", "BIN", "CW"), _sig_count, _build_count_step,
    "one symbol through the 32-bit WNC arithmetic encoder, the bits it puts in the window's segment added to the "
    "window's count (`arith_encode.reference_count_step`)")

ArithCountFinish = _part(
    "ArithCountFinish", ("CW",), _sig_count_finish, _build_count_finish,
    "the arithmetic encoder's termination added to the window's count, held to the declared charge c "
    "(`arith_encode.reference_count_finish`)")

BookKT = _part(
    "BookKT", ("C",), _sig_book_kt, _build_book_kt,
    "the registered book's KT estimate for one bit symbol's context (`code_v1.KT`): f = 2n + 1, and its count raised, "
    "both halved once their estimate would pass 2^16 (`arith_encode.reference_book_kt`)")

ChargeCount = _part(
    "ChargeCount", ("N",), _sig_charge, _build_charge,
    "one window's charge under advice v1 when it sends only decisions: code_v1's segment counted, n_arrived from the "
    "enabled slots and its bin's frequency from the book's public table, each decision at the book's KT estimate for "
    "min(run, 2), both carried across windows, then the other fields' zero counts and the finish "
    "(`arith_encode.reference_charge`)")


def step_unit(K: int = K_MAX, BIN: int = 1):
    return bind(ArithEncodeStep, K=K, BIN=BIN)


def finish_unit():
    return bind(ArithEncodeFinish)


def kt_unit(C: int = 3, NW: int = 24):
    return bind(KTContext, C=C, NW=NW)


def count_unit(K: int = K_MAX, BIN: int = 1, CW: int = CW):
    return bind(ArithCountStep, K=K, BIN=BIN, CW=CW)


def count_finish_unit(CW: int = CW):
    return bind(ArithCountFinish, CW=CW)


def book_kt_unit(C: int = 3):
    return bind(BookKT, C=C)


def charge_unit(N: int):
    return bind(ChargeCount, N=N)


def charge_fields(N: int) -> list[tuple[str, int, int]]:
    """`ChargeCount_v1{N}`'s inputs before the slots, (name, elements, width) as a row's fields: the book's tables, then
    the carry (`_CARRY`: n0, n1 and run)."""
    return _charge_in(SimpleNamespace(N=N))[:-2]


def charge_values(book: ChargeBook, carry: Carry) -> dict[str, list[int]]:
    """`charge_fields`' values, by name."""
    return {"arrived": list(book.arrived), "total": [book.total], "zfreq": [f for f, _t in book.zeros],
            "ztotal": [t for _f, t in book.zeros], "n0": [c[0] for c in carry.counts],
            "n1": [c[1] for c in carry.counts], "run": [carry.run]}


def build_charge(C, N: int, public: Sequence, en: Sequence, bits: Sequence) -> tuple:
    """`ChargeCount_v1{N}` inline in a caller's circuit: `public`, the elements of `charge_fields(N)` in order; `en` and
    `bits`, one wire per slot. Returns (ok, bits[CW], carry'), carry' as `_CARRY`'s elements, the next window's."""
    out = _build_charge(C, SimpleNamespace(N=N), *public, *[[e] for e in en], *[[b] for b in bits])
    return out[0][0], out[1], out[2:]


# --- arguments and evaluation ----------------------------------------------------------------------------------------


def _bits(v: int, w: int) -> list[int]:
    if v < 0 or v >> w:
        raise ValueError(f"{v} doesn't fit {w} bits")
    return [(v >> j) & 1 for j in range(w)]


def _args(fields, vals: dict) -> list[list[int]]:
    return [[b for v in (vals[name] if isinstance(vals[name], (list, tuple)) else [vals[name]]) for b in _bits(v, w)]
            for name, _n, w in fields]


def _fields(fields, flat: Sequence[int]) -> dict:
    at, out = 0, {}
    for name, n, w in fields:
        out[name] = [sum(int(flat[at + e * w + j]) << j for j in range(w)) for e in range(n)]
        at += n * w
    return out


def _coder_vals(st: Coder) -> dict:
    return {"low": st.low, "high": st.high, "open": st.open, "x": st.x, "used": st.used, "ok": st.ok}


def step_arguments(K: int, BIN: int, st: Coder, en: int, sym: Sequence[int], seg: Sequence[int]) -> list[list[int]]:
    S = SimpleNamespace(K=K, BIN=BIN)
    names = ("bit", "f0", "f1") if BIN else ("cum", "freq", "total")
    vals = {**_coder_vals(st), "en": en, **dict(zip(names, sym)), "seg": sum(int(b) << j for j, b in enumerate(seg))}
    return _args(_step_in(S), vals)


def evaluate_steps(K: int, BIN: int, cases: Sequence[tuple[Coder, int, Sequence[int], Sequence[int]]]) -> list[tuple[Coder, int]]:
    """The step at (K, BIN) on each (state, en, symbol, seg), in one evaluation."""
    S = SimpleNamespace(K=K, BIN=BIN)
    args = [list(rows) for rows in zip(*(step_arguments(K, BIN, *c) for c in cases))]
    out = evaluate_bits(step_unit(K, BIN), *args)
    res = []
    for row in out.reshape(len(cases), -1).tolist():
        f = _fields(_step_out(S), row)
        res.append((Coder(f["low"][0], f["high"][0], f["open"][0], f["x"][0], f["used"][0], f["ok"][0]), f["n"][0]))
    return res


def evaluate_finishes(cases: Sequence[tuple[Coder, Sequence[int]]]) -> list[tuple[int, int]]:
    args = [list(rows) for rows in zip(*(_args(_FIN_IN, {**_coder_vals(st), "seg": seg[0] | seg[1] << 1})
                                         for st, seg in cases))]
    out = evaluate_bits(finish_unit(), *args)
    return [(f["ok"][0], f["n"][0]) for f in (_fields(_FIN_OUT, row) for row in out.reshape(len(cases), -1).tolist())]


def _tally_vals(st: Tally) -> dict:
    return {"low": st.low, "high": st.high, "used": st.used, "ok": st.ok, "bits": st.bits}


def evaluate_counts(K: int, BIN: int, CW: int, cases: Sequence[tuple[Tally, int, Sequence[int]]]) -> list[Tally]:
    """The counting step at (K, BIN, CW) on each (state, en, symbol), in one evaluation."""
    S = SimpleNamespace(K=K, BIN=BIN, CW=CW)
    names = ("bit", "f0", "f1") if BIN else ("cum", "freq", "total")
    args = [list(rows) for rows in zip(*(_args(_count_in(S), {**_tally_vals(st), "en": en, **dict(zip(names, sym))})
                                         for st, en, sym in cases))]
    out = evaluate_bits(count_unit(K, BIN, CW), *args)
    res = []
    for row in out.reshape(len(cases), -1).tolist():
        f = _fields(_count_out(S), row)
        res.append(Tally(f["low"][0], f["high"][0], f["used"][0], f["ok"][0], f["bits"][0]))
    return res


def evaluate_count_finishes(CW: int, cases: Sequence[tuple[Tally, int]]) -> list[tuple[int, int]]:
    """The counting finish at CW on each (state, c), in one evaluation."""
    S = SimpleNamespace(CW=CW)
    args = [list(rows) for rows in zip(*(_args(_count_fin_in(S), {**_tally_vals(st), "c": c}) for st, c in cases))]
    out = evaluate_bits(count_finish_unit(CW), *args)
    return [(f["ok"][0], f["bits"][0]) for f in (_fields(_count_fin_out(S), row)
                                                for row in out.reshape(len(cases), -1).tolist())]


def evaluate_book_kt(C: int, cases: Sequence[tuple[Sequence[tuple[int, int]], int, int, int]]
                     ) -> list[tuple[int, int, int, tuple[tuple[int, int], ...]]]:
    """`BookKT_v1{C}` on each (counts, en, ctx, bit), in one evaluation."""
    S = SimpleNamespace(C=C)
    args = [list(rows) for rows in zip(*(_args(_bkt_in(S), {"n0": [c[0] for c in counts], "n1": [c[1] for c in counts],
                                                            "en": en, "ctx": ctx, "bit": bit})
                                         for counts, en, ctx, bit in cases))]
    out = evaluate_bits(book_kt_unit(C), *args)
    res = []
    for row in out.reshape(len(cases), -1).tolist():
        f = _fields(_bkt_out(S), row)
        res.append((f["ok"][0], f["f0"][0], f["f1"][0], tuple(zip(f["n0"], f["n1"]))))
    return res


def evaluate_charges(N: int, cases: Sequence[tuple[ChargeBook, Carry, Sequence[int], Sequence[int]]]
                     ) -> list[tuple[int, int, Carry]]:
    """`ChargeCount_v1{N}` on each (book, carry, en, bits), in one evaluation."""
    S = SimpleNamespace(N=N)
    args = [list(rows) for rows in zip(*(_args(_charge_in(S), {**charge_values(b, c), "en": en, "bit": bits})
                                         for b, c, en, bits in cases))]
    out = evaluate_bits(charge_unit(N), *args)
    res = []
    for row in out.reshape(len(cases), -1).tolist():
        f = _fields(_charge_out(S), row)
        res.append((f["ok"][0], f["bits"][0], Carry(tuple(zip(f["n0"], f["n1"])), f["run"][0])))
    return res


def evaluate_kt(C: int, NW: int, cases: Sequence[tuple[Sequence[tuple[int, int]], int, int, int]]
                ) -> list[tuple[int, int, int, tuple[tuple[int, int], ...]]]:
    S = SimpleNamespace(C=C, NW=NW)

    def one(counts, en, ctx, bit):
        return _args(_kt_in(S), {"n0": [c[0] for c in counts], "n1": [c[1] for c in counts], "en": en, "ctx": ctx,
                                 "bit": bit})

    args = [list(rows) for rows in zip(*(one(*c) for c in cases))]
    out = evaluate_bits(kt_unit(C, NW), *args)
    res = []
    for row in out.reshape(len(cases), -1).tolist():
        f = _fields(_kt_out(S), row)
        res.append((f["ok"][0], f["f0"][0], f["f1"][0], tuple(zip(f["n0"], f["n1"]))))
    return res


# --- cost ------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _staged(kind: str, key: tuple) -> dict[str, int]:
    sig, build, statics = {"step": (_sig_step, _build_step, ("K", "BIN")), "finish": (_sig_finish, _build_finish, ()),
                           "kt": (_sig_kt, _build_kt, ("C", "NW")),
                           "count": (_sig_count, _build_count_step, ("K", "BIN", "CW")),
                           "count_finish": (_sig_count_finish, _build_count_finish, ("CW",)),
                           "book_kt": (_sig_book_kt, _build_book_kt, ("C",)),
                           "charge": (_sig_charge, _build_charge, ("N",))}[kind]
    S = SimpleNamespace(**dict(zip(statics, key)))
    C = Circuit(cse=True)
    params, _ = sig(S)
    ins = [C.input(f"p{i}e{j}", w) for i, (_n, t) in enumerate(params) for j, w in enumerate([t.elem.n] * t.n)]
    C.output("out", [b for e in build(C, S, *ins) for b in e])
    return C.phase_counts()


def size(kind: str = "step", **st) -> dict[str, int]:
    """ANDs by stage of the step (K, BIN), the finish, the model (C, NW), the counting step (K, BIN, CW), its finish
    (CW), the book's model (C) or a window's charge (N), traced on a counting circuit. The charge's ``carry`` is no AND:
    its copies of the carried words, once a decision, each a row of a lowering and no gate."""
    keys = {"step": ("K", "BIN"), "finish": (), "kt": ("C", "NW"), "count": ("K", "BIN", "CW"),
            "count_finish": ("CW",), "book_kt": ("C",), "charge": ("N",)}[kind]
    out = dict(_staged(kind, tuple(st[k] for k in keys)))
    out["ands"] = sum(v for k, v in out.items() if k != "carry")
    return out


__all__ = ["ArithCountFinish", "ArithCountStep", "ArithEncodeFinish", "ArithEncodeStep", "BookKT", "CW", "Carry",
           "ChargeBook", "ChargeCount", "Coder", "HALF", "KW", "K_MAX", "KTContext", "MAX_TOTAL", "Q1", "RUNS", "TOP",
           "Tally", "ZERO_COUNTS", "book_kt_unit", "build_charge", "charge_book", "charge_fields", "charge_unit",
           "charge_values", "check_window", "count_finish_unit", "count_unit", "count_window", "evaluate_book_kt",
           "evaluate_charges", "evaluate_count_finishes", "evaluate_counts", "evaluate_finishes", "evaluate_kt",
           "evaluate_steps", "finish_unit", "first_carry", "fresh", "kt_unit", "reference_book_kt", "reference_charge",
           "reference_count_finish", "reference_count_step", "reference_finish", "reference_kt", "reference_step",
           "scaled", "size", "step_arguments", "step_unit"]
