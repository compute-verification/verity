"""Offline memory checking (Blum, Evans, Gemmell, Kannan and Naor, 1994) as Boolean Definitions: a log of reads and writes to
word-addressed memories is consistent, every address, value and time in it a private value.

    MemAccess_v1{G, A, V, T}
        (alpha: F, gamma: F, R: F, W: F, ok: Value<1>, en: Value<1>, tag: Array<G, Value<1>>, a: Array<A, Value<1>>,
         v: Array<V, Value<1>>, t: Array<T, Value<1>>, w: Array<V, Value<1>>, now: Array<T, Value<1>>) -> (F, F, Value<1>)
    MemEntry_v1{G, A, V, T}
        (alpha, gamma, R, W, ok, first: Value<1>, pen: Value<1>, ptag: Array<G, Value<1>>, pa: Array<A, Value<1>>,
         en, tag, a, v0: Array<V, Value<1>>, v, t) -> (F, F, Value<1>)
    MemCheck_v1{N, M, G, A, V, T}
        (alpha, gamma, en: Array<N, Value<1>>, tag: Array<N, Array<G, Value<1>>>, a, v, t, w,
         l_en: Array<M, Value<1>>, l_tag: Array<M, Array<G, Value<1>>>, l_a, l_v0, l_v, l_t) -> Value<1>
        F = Array<128, Value<1>>, an element of GF(2^128) (`gf128`); every width at least 1

**The check.** Each access reads a record (a, v, t), the value v at address a and the time t it was written, and writes
(a, w, now). Each address the log touches has an initial record (a, v0, 0) and a final record, the last (a, v, t) written
there. The log is consistent iff every read's t is below its now and, as multisets,

    reads ∪ finals = writes ∪ initials

which the circuit checks as a fingerprint over F under α and γ, the verifier's live coins, drawn after the log is committed:

    R = ∏ (α + e(read)) · ∏ (α + e(final)),   W = ∏ (α + e(write)) · ∏ (α + e(initial)),   accept iff R = W

The nows are public wiring, distinct and above 0, and each v0 is a public function of its address, which the caller
computes (a sparse memory's is 0). A record is distinct per (tag, a, value, time), so two different multisets of at most n
records a side give different polynomials in (α, γ) of degree at most n·max(1, k − 1), and collide with probability at most
n·max(1, k − 1) / 2^128 (Schwartz–Zippel).

**Records.** A record is the bits tag ‖ a ‖ value ‖ time, least significant first, of widths G, A, V and T: the tag names
the memory, so memories of different shapes share one fingerprint, each with its address and value in the low bits of a
and v. Its k = ⌈(G + A + V + T)/128⌉ chunks of 128 bits c_0, …, c_{k−1} (the last zero-padded) give
e = Σ γ^i c_i, by Horner's rule: k − 1 products.

**The touched list.** The initial and final records come from a list of the addresses the log touches, sorted strictly by
the key (tag, a), the tag most significant, so no address has two initial records; its used entries are a prefix. A
position of the log or the list that isn't used (en = 0) holds zeros and leaves R and W as they are.

**MemAccess** is one access: ``R·(α + e(tag, a, v, t))`` and ``W·(α + e(tag, a, w, now))``, and ``ok ∧ t < now``; unused,
``(R, W, ok ∧ the access's fields are zero)``. **MemEntry** is one entry of the list after the entry (pen, ptag, pa):
``R·(α + e(final))``, ``W·(α + e(tag, a, v0, 0))`` and ``ok ∧ (first ∨ (pen ∧ (ptag, pa) < (tag, a)))``; unused,
``(R, W, ok ∧ its fields are zero)``. A stream chains them, carrying (R, W, ok) and the previous entry. **MemCheck** is a
whole log of N accesses at the times 1, …, N (so N < 2^T) and its list of M entries: 1 iff R = W and every ok holds, the
chain from R = W = 1. Each product is `gf128`'s, 2,187 ANDs or fewer on constant bits.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import ONE, OR, ZERO, const_bits, or_reduce, sub
from verity.circuits.boolean.gf128 import mul, mul_forms
from verity.circuits.types import Array, Tuple

__all__ = ["Access", "Entry", "Widths", "MemAccess", "MemCheck", "MemEntry", "access", "consistent", "element", "entry",
           "memcheck", "pack"]

BIT = BL.BIT
F = Array(128, BIT)
STATE = Tuple(F, F, BIT)


class Widths(NamedTuple):
    """A record's field widths: tag G, address A, value V, time T."""
    G: int
    A: int
    V: int
    T: int


class Access(NamedTuple):
    en: int
    tag: int
    a: int
    v: int
    t: int
    w: int
    now: int


class Entry(NamedTuple):
    en: int
    tag: int
    a: int
    v0: int
    v: int
    t: int


def _check(S) -> None:
    for k in ("G", "A", "V", "T"):
        x = getattr(S, k)
        if not isinstance(x, int) or isinstance(x, bool) or x < 1:
            raise ValueError(f"a memory check's width {k} is {x!r}, not a positive int")


# --- the references ----------------------------------------------------------------------------------------------------

def pack(S, tag: int, a: int, value: int, time: int) -> int:
    """A record's bits: tag ‖ a ‖ value ‖ time, least significant first."""
    for x, w in ((tag, S.G), (a, S.A), (value, S.V), (time, S.T)):
        if not 0 <= x < 1 << w:
            raise ValueError(f"a record field {x} out of its {w} bits")
    return tag | a << S.G | value << (S.G + S.A) | time << (S.G + S.A + S.V)


def element(S, gamma: int, tag: int, a: int, value: int, time: int) -> int:
    """``Σ γ^i c_i`` over the record's 128-bit chunks."""
    r, n = pack(S, tag, a, value, time), S.G + S.A + S.V + S.T
    cs = [(r >> (128 * i)) & ((1 << 128) - 1) for i in range(-(-n // 128))]
    e = cs[-1]
    for c in reversed(cs[:-1]):
        e = mul(e, gamma) ^ c
    return e


def access(S, alpha: int, gamma: int, R: int, W: int, ok: int, x: Access) -> tuple[int, int, int]:
    """`MemAccess`'s (R, W, ok) after the access ``x``."""
    if not x.en:
        return R, W, ok & int(not any((x.tag, x.a, x.v, x.t, x.w)))
    return (mul(R, alpha ^ element(S, gamma, x.tag, x.a, x.v, x.t)), mul(W, alpha ^ element(S, gamma, x.tag, x.a, x.w, x.now)),
            ok & int(x.t < x.now))


def entry(S, alpha: int, gamma: int, R: int, W: int, ok: int, first: int, prev: Entry, x: Entry) -> tuple[int, int, int]:
    """`MemEntry`'s (R, W, ok) after the list entry ``x``, which follows ``prev`` (its en, tag and a read)."""
    if not x.en:
        return R, W, ok & int(not any((x.tag, x.a, x.v0, x.v, x.t)))
    order = first or (prev.en and (prev.tag, prev.a) < (x.tag, x.a))
    return (mul(R, alpha ^ element(S, gamma, x.tag, x.a, x.v, x.t)), mul(W, alpha ^ element(S, gamma, x.tag, x.a, x.v0, 0)),
            ok & int(bool(order)))


def memcheck(S, alpha: int, gamma: int, accesses: Sequence[Access], entries: Sequence[Entry]) -> int:
    """`MemCheck`: 1 iff the chain from R = W = 1 ends with R = W and every ok; access j's now is j + 1."""
    R, W, ok = 1, 1, 1
    for j, x in enumerate(accesses):
        if x.now != j + 1:
            raise ValueError(f"MemCheck's access {j} is at time {j + 1}, not {x.now}")
        R, W, ok = access(S, alpha, gamma, R, W, ok, x)
    prev = Entry(0, 0, 0, 0, 0, 0)
    for i, x in enumerate(entries):
        R, W, ok = entry(S, alpha, gamma, R, W, ok, int(i == 0), prev, x)
        prev = x
    return int(bool(ok) and R == W)


def consistent(accesses: Sequence[Access], entries: Sequence[Entry]) -> bool:
    """What the fingerprint stands for, exactly: unused positions zero, distinct nows above 0 with every read's t below its
    now, the list's used entries a prefix sorted strictly by (tag, a), and reads ∪ finals = writes ∪ initials."""
    used = [x for x in accesses if x.en]
    if any(any(x[1:6]) for x in accesses if not x.en) or any(any(e[1:]) for e in entries if not e.en):
        return False
    nows = [x.now for x in used]
    if len(set(nows)) != len(nows) or any(x.now <= 0 or x.t >= x.now for x in used):
        return False
    keys = [(e.tag, e.a) for e in entries if e.en]
    if [e.en for e in entries] != [1] * len(keys) + [0] * (len(entries) - len(keys)) or keys != sorted(set(keys)):
        return False
    lhs = sorted([(x.tag, x.a, x.v, x.t) for x in used] + [(e.tag, e.a, e.v, e.t) for e in entries if e.en])
    rhs = sorted([(x.tag, x.a, x.w, x.now) for x in used] + [(e.tag, e.a, e.v0, 0) for e in entries if e.en])
    return lhs == rhs


# --- the circuits ------------------------------------------------------------------------------------------------------

_ONE128 = [ONE] + [ZERO] * 127


def _element_forms(C, gamma: list, bits: list) -> list:
    k = -(-len(bits) // 128)
    bits = list(bits) + [ZERO] * (128 * k - len(bits))
    cs = [bits[128 * i:128 * (i + 1)] for i in range(k)]
    e = cs[-1]
    for c in reversed(cs[:-1]):
        e = [x ^ y for x, y in zip(mul_forms(C, e, gamma), c)]
    return e


def _fold(C, alpha: list, gamma: list, acc: list, en, bits: list) -> list:
    """``acc · (α + e(bits))`` if en, else ``acc``: the factor is 1 when unused."""
    f = [x ^ y for x, y in zip(alpha, _element_forms(C, gamma, bits))]
    return mul_forms(C, acc, [o ^ C.AND(en, x ^ o) for x, o in zip(f, _ONE128)])


def _zero(C, *fields: list):
    return or_reduce(C, [x for f in fields for x in f]).inv()


def _verdict(C, ok, en, used, unused):
    """``ok ∧ (used if en else unused)``."""
    return C.AND(ok, unused ^ C.AND(en, used ^ unused))


def access_forms(C, S, alpha, gamma, R, W, ok, en, tag, a, v, t, w, now) -> list[list]:
    en, ok = en[0], ok[0]
    R2 = _fold(C, alpha, gamma, R, en, [*tag, *a, *v, *t])
    W2 = _fold(C, alpha, gamma, W, en, [*tag, *a, *w, *now])
    _, before = sub(C, list(t), list(now), S.T)
    return [R2, W2, [_verdict(C, ok, en, before, _zero(C, tag, a, v, t, w))]]


def entry_forms(C, S, alpha, gamma, R, W, ok, first, pen, ptag, pa, en, tag, a, v0, v, t) -> list[list]:
    en, ok, first, pen = en[0], ok[0], first[0], pen[0]
    R2 = _fold(C, alpha, gamma, R, en, [*tag, *a, *v, *t])
    W2 = _fold(C, alpha, gamma, W, en, [*tag, *a, *v0, *[ZERO] * S.T])
    _, less = sub(C, [*pa, *ptag], [*a, *tag], S.A + S.G)
    after = OR(C, first, C.AND(pen, less))
    return [R2, W2, [_verdict(C, ok, en, after, _zero(C, tag, a, v0, v, t))]]


def _split(e: Sequence, counts: Sequence[int]) -> list[list]:
    out, i = [], 0
    for n in counts:
        out.append(list(e[i:i + n]))
        i += n
    assert i == len(e)
    return out


def check_forms(C, S, *e) -> list:
    N, M = S.N, S.M
    alpha, gamma, en, tag, a, v, t, w, l_en, l_tag, l_a, l_v0, l_v, l_t = _split(
        e, (1, 1, 1, N, N, N, N, N, 1, M, M, M, M, M))
    alpha, gamma, en, l_en = alpha[0], gamma[0], en[0], l_en[0]
    R, W, ok = _ONE128, _ONE128, [ONE]
    for j in range(N):
        R, W, ok = access_forms(C, S, alpha, gamma, R, W, ok, [en[j]], tag[j], a[j], v[j], t[j], w[j],
                                const_bits(j + 1, S.T))
    prev = ([ZERO], [ZERO] * S.G, [ZERO] * S.A)
    for i in range(M):
        R, W, ok = entry_forms(C, S, alpha, gamma, R, W, ok, [ONE if i == 0 else ZERO], *prev, [l_en[i]], l_tag[i], l_a[i],
                               l_v0[i], l_v[i], l_t[i])
        prev = ([l_en[i]], l_tag[i], l_a[i])
    return [C.AND(ok[0], _zero(C, [x ^ y for x, y in zip(R, W)]))]


def _sig_access(S):
    _check(S)
    G, A, V, T = (Array(S.G, BIT), Array(S.A, BIT), Array(S.V, BIT), Array(S.T, BIT))
    return ((("alpha", F), ("gamma", F), ("R", F), ("W", F), ("ok", BIT), ("en", BIT), ("tag", G), ("a", A), ("v", V),
             ("t", T), ("w", V), ("now", T)), STATE)


def _sig_entry(S):
    _check(S)
    G, A, V, T = (Array(S.G, BIT), Array(S.A, BIT), Array(S.V, BIT), Array(S.T, BIT))
    return ((("alpha", F), ("gamma", F), ("R", F), ("W", F), ("ok", BIT), ("first", BIT), ("pen", BIT), ("ptag", G),
             ("pa", A), ("en", BIT), ("tag", G), ("a", A), ("v0", V), ("v", V), ("t", T)), STATE)


def _sig_check(S):
    _check(S)
    for k in ("N", "M"):
        x = getattr(S, k)
        if not isinstance(x, int) or isinstance(x, bool) or x < 1:
            raise ValueError(f"MemCheck's {k} is {x!r}, not a positive int")
    if S.N >= 1 << S.T:
        raise ValueError(f"MemCheck's times 1..{S.N} need more than T = {S.T} bits")
    G, A, V, T = (Array(S.G, BIT), Array(S.A, BIT), Array(S.V, BIT), Array(S.T, BIT))
    N, M = S.N, S.M
    return ((("alpha", F), ("gamma", F), ("en", Array(N, BIT)), ("tag", Array(N, G)), ("a", Array(N, A)), ("v", Array(N, V)),
             ("t", Array(N, T)), ("w", Array(N, V)), ("l_en", Array(M, BIT)), ("l_tag", Array(M, G)), ("l_a", Array(M, A)),
             ("l_v0", Array(M, V)), ("l_v", Array(M, V)), ("l_t", Array(M, T))), BIT)


MemAccess = TR.part(
    "MemAccess", ("G", "A", "V", "T"), _sig_access, access_forms,
    "One access of a memory-checked log: (R·(α + e(tag, a, v, t)), W·(α + e(tag, a, w, now)), ok ∧ t < now) if en, else "
    "(R, W, ok ∧ the fields are zero); e is a record's packing under γ over GF(2^128)")

MemEntry = TR.part(
    "MemEntry", ("G", "A", "V", "T"), _sig_entry, entry_forms,
    "One entry of a memory-checked log's touched list after (pen, ptag, pa): (R·(α + e(tag, a, v, t)), "
    "W·(α + e(tag, a, v0, 0)), ok ∧ (first ∨ (pen ∧ (ptag, pa) < (tag, a)))) if en, else (R, W, ok ∧ the fields are zero)")

MemCheck = TR.part(
    "MemCheck", ("N", "M", "G", "A", "V", "T"), _sig_check, check_forms,
    "A memory-checked log of N accesses at times 1..N and its touched list of M entries: 1 iff the fingerprints of "
    "reads ∪ finals and writes ∪ initials under (α, γ) agree, every read's time is below its now, the list is sorted and "
    "unused positions are zero")
