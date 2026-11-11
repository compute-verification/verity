"""`TokenSelect_v1{V}` (greedy sampling) as units of at most a Flock block each, a tree of partial argmaxes over the row's K
chunks of N = V / K logits: `plan` lists them with their wiring, `cut` counts them by Definition (`b1.CUTS`), `twin` computes
every unit's outputs and the token, and `GreedySelect_v1` is the tree as one Definition.

    GreedyChunk_v1{N,FIRST}   (logits: bf16[N]) -> (bf16, Value<16>): the chunk's best logit and its index in the chunk
    GreedyCombine_v2{V,N}     (logits: bf16[N], b0: bf16, i0: Value<16>, .., b{K-2}, i{K-2}) -> Array<2, Value<16>>: the
                              token's low and high 16 bits, from the last of the K = V / N chunks (its lanes) and the other
                              chunks' pairs
    GreedySelect_v1{V,N}      (logits: bf16[V]) -> Array<2, Value<16>>: the same two words; its body is the plan's units

**Every chunk is whole.**  N divides V, so each unit reads N lanes that start at a multiple of N, in one row and in several
rows end to end (a replay block of several tokens: `spot.verifier.second_stage` links a unit's argument to the block's logits
only at such a start), and the token's two words are one return field, which it links as one committed row.  The plan's K is
`UNITS` on a row it divides (`units`), and `plan` refuses a K that does not divide V; the Definitions take any such N.

The combine reads the last chunk's lanes rather than a pair for it so that its proof block composes: Flock's composer takes a
unit only with some row past 111 bytes (`windows.interface.check_proof_block`), and the pairs are 2-byte rows; a row whose
chunks have fewer than 56 lanes would need it to read more.

**The function** is `TokenSelect_v1{V}`'s: from (logits[0], 0), take (x_i, i) iff ``Bf16GtStrict_v1(x_i, best)``, so the lowest
index of the largest logit wins, a NaN is never taken, +0 and -0 are equal, and the token is 0 whenever logits[0] is a NaN.  The
row's first chunk (FIRST) is that scan over its N lanes.  A later chunk is the scan from (-inf, 0): its first maximum above -inf,
and (-inf, 0) when it has none; so is the last chunk's pair inside the combine (the first's when the row is one chunk).  The
combine starts from chunk 0's pair and takes chunk c's (b_c, cN + i_c) iff
``Bf16GtStrict_v1(b_c, best)``.  Its best after chunks 0 .. c-1 is the scan's: a NaN logits[0], which neither ever leaves, or a
number at least every number in those chunks, so the scan moves in chunk c iff some lane is above it, and then ends at the
chunk's first maximum, which is b_c's lane.  Ties keep the earlier chunk, as the scan keeps the earlier lane.

**Every output is one 16-bit word.**  A chunk's index is below N <= 2^16.  The token cN + i_c is below V <= 2^32: its low word
is the wrapping sum of ``cN mod 2^16`` and i_c (`U16Add_v1`), and its high word is ``cN >> 16`` plus that sum's carry.  The
carry is 0 at every i_c < N unless a multiple of 2^16 falls inside the chunk, at its lane ``t = 2^16 - cN mod 2^16`` with
0 < t < N (`carries`; no chunk has one when N is a power of two); there it is 1 iff i_c >= t, which the combine computes as
``Bf16GtStrict_v1(i_c, t - 1)``: on the words 0 .. 0x7F80 (+0, the positive numbers and +inf, no NaN among them) bf16's
order is the integers', and both words are below N, so a width some chunk carries at is at most 0x7F81 lanes
(`GreedyCombine_v2` refuses a wider one).  At N = 7,744 of V = 154,880 two chunks carry: chunk 8 from its lane 3,584 and chunk
16 from its lane 7,168.  Each select keeps the running value as its *then* branch under the comparison's `BitNot`, as
``TokenSelectChunkStep_v1``'s does, so its bits are the new value's XOR one AND each and a chunk lowers to a circuit linear in N.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from verity.circuits.defs import SpecializedDefinition, bind, composite
from verity.definitions.ops.argmax import U16, SelectU16, U16Add
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import BitNot
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import prims as P
from verity.definitions.registry.b1 import NEG_INF_BF16
from verity.definitions.registry.prims import BF16

#: the chunks a served row is cut into, a number: any divisor of V is a plan of the same Definitions (`plan`'s K).  More chunks
#: are smaller chunk blocks and a larger combine, which reads two rows a chunk, each an hm96 slot of its block
UNITS = 20
SPAN = 1 << 16
#: the largest word bf16 orders as the integers do: 0 .. 0x7F80 are +0, the positive numbers and +inf
ORDERED = 0x7F80
_CONF = "model-untested"


def units(V: int) -> int:
    """The chunks `plan` cuts a row of V into: `UNITS`, and on a row it does not divide the most below it that do."""
    if isinstance(V, bool) or not isinstance(V, int) or V < 1:
        raise ValueError(f"V is a row's logits, an int >= 1, not {V!r}")
    return next(k for k in range(min(UNITS, V), 0, -1) if V % k == 0)


def width(V: int) -> int:
    """A chunk's lanes on a row of V in `units` chunks."""
    return V // units(V)


def carries(V: int, N: int) -> dict[int, int]:
    """Each chunk c of a row of V in chunks of N that a multiple of 2^16 falls inside -> its lane t there: cN + i is in the
    next 2^16-wide span iff i >= t (0 < t < N), so the token's high word is ``cN >> 16`` plus [i >= t]."""
    return {c: SPAN - c * N % SPAN for c in range(1, V // N) if c * N % SPAN + N > SPAN}


def _lanes(N) -> None:
    if isinstance(N, bool) or not isinstance(N, int) or not 1 <= N <= SPAN:
        raise ValueError(f"N is a chunk's lanes, an int in [1, 2^16] (its index is a 16-bit word), not {N!r}")


def _chunk_sig(S):
    _lanes(S.N)
    if not isinstance(S.FIRST, bool):
        raise ValueError(f"FIRST says whether the chunk is the row's first, a bool (an id's `FIRST=True`), not {S.FIRST!r}")
    return (("logits", Array(S.N, BF16)),), Tuple(BF16, U16)


def _scan(B, logits, n: int, first: bool, last_best: bool = True):
    """The scan over a chunk's ``n`` logits -> (its best, that lane's index in the chunk): ``first``, from (logits[0], 0); else
    from (-inf, 0).  The last lane's best only with ``last_best``."""
    if first:
        best, at, start = logits[0], B.call(const(16, 0)), 1
    else:
        best, at, start = B.call(const(16, NEG_INF_BF16)), B.call(const(16, 0)), 0
    for i in range(start, n):
        keep = B.call(BitNot, B.call(P.Bf16GtStrict, logits[i], best))
        if i:
            at = B.call(SelectU16, keep, at, B.call(const(16, i)))
        if last_best or i < n - 1:
            best = B.call(P.SelectBf16, keep, best, logits[i])
    return best, at


@composite("GreedyChunk", 1, ["N", "FIRST"], _chunk_sig, conformance=_CONF)
def GreedyChunk(B, S, logits):
    """A chunk of N logits -> (its best, that lane's index in the chunk): FIRST, `TokenSelect_v1`'s scan over them from
    (logits[0], 0); else the scan from (-inf, 0), (-inf, 0) when no lane is above -inf (the module docstring)."""
    return tuple_of(*_scan(B, logits, S.N, S.FIRST))


def _whole(S) -> None:
    _lanes(S.N)
    if isinstance(S.V, bool) or not isinstance(S.V, int) or not S.N <= S.V <= SPAN * SPAN or S.V % S.N:
        raise ValueError(f"a row of V = {S.V} in chunks of N = {S.N}: N divides V (every chunk is whole) and V <= 2^32 (the "
                         "token's two words)")
    if S.N - 1 > ORDERED and carries(S.V, S.N):
        raise ValueError(f"a row of V = {S.V} in chunks of N = {S.N}: a multiple of 2^16 falls inside a chunk, whose carry "
                         f"compares indices as bf16 words, in the integers' order only up to {ORDERED:#x}: N <= {ORDERED + 1}, "
                         "or a power of two")


def _combine_sig(S):
    _whole(S)
    pairs = tuple(p for c in range(S.V // S.N - 1) for p in ((f"b{c}", BF16), (f"i{c}", U16)))
    return (("logits", Array(S.N, BF16)), *pairs), Array(2, U16)


@composite("GreedyCombine", 2, ["V", "N"], _combine_sig, conformance=_CONF)
def GreedyCombine(B, S, logits, *pairs):
    """The row's last chunk of logits and its other chunks' pairs (b_c, i_c) -> the token's (low, high) 16-bit words, one
    field: the last chunk's pair is `GreedyChunk_v1`'s, and from chunk 0's pair, chunk c's (b_c, cN + i_c) is taken iff
    ``Bf16GtStrict_v1(b_c, best)``; cN + i_c's high word is ``cN >> 16``, and one more from the chunk's lane `carries` names
    (the module docstring)."""
    n, carry = S.V // S.N, carries(S.V, S.N)
    pairs = (*pairs, *_scan(B, logits, S.N, n == 1, last_best=n > 1))
    best, lo, hi = pairs[0], pairs[1], B.call(const(16, 0))
    for c in range(1, n):
        b, i = pairs[2 * c], pairs[2 * c + 1]
        keep = B.call(BitNot, B.call(P.Bf16GtStrict, b, best))
        off = c * S.N
        span = B.call(const(16, off // SPAN))
        if c in carry:
            past = B.call(P.Bf16GtStrict, i, B.call(const(16, carry[c] - 1)))
            span = B.call(SelectU16, past, B.call(const(16, off // SPAN + 1)), span)
        lo = B.call(SelectU16, keep, lo, B.call(U16Add, i, B.call(const(16, off % SPAN))) if off % SPAN else i)
        hi = B.call(SelectU16, keep, hi, span)
        if c < n - 1:
            best = B.call(P.SelectBf16, keep, best, b)
    return array_of([lo, hi])


STAGE_DEFINITIONS = [GreedyChunk, GreedyCombine]


# --- the plan ------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """One unit of the cut, as `topp_stages.Unit`: ``key`` names it, ``defn`` its Definition, ``lanes`` the logits it reads,
    ``wires`` each wired parameter's (parameter, writer key, writer's output index)."""
    key: tuple
    defn: SpecializedDefinition
    lanes: tuple[int, int]
    wires: tuple[tuple[str, tuple, int], ...] = field(default=())


@cache
def plan(V: int, K: int | None = None) -> tuple[Unit, ...]:
    """The units of a row of V in K chunks (`units(V)` of them when None) in an order that writes each before its readers:
    its chunks of V / K lanes but the last, then the combine, which reads the last chunk's lanes and whose output is the
    token's words.  A K that does not divide V is refused: a narrower last chunk starts off the chunks' rows."""
    K = units(V) if K is None else K
    if isinstance(K, bool) or not isinstance(K, int) or K < 1 or V % K:
        raise ValueError(f"a row of V = {V} is not K = {K!r} whole chunks: K divides V")
    N = V // K
    _lanes(N)
    out = [Unit(("chunk", c), bind(GreedyChunk, N=N, FIRST=c == 0), (c * N, (c + 1) * N)) for c in range(K - 1)]
    wires = tuple(w for c in range(K - 1) for w in ((f"b{c}", ("chunk", c), 0), (f"i{c}", ("chunk", c), 1)))
    return (*out, Unit(("combine",), bind(GreedyCombine, V=V, N=N), ((K - 1) * N, V), wires))


def cut(t) -> list[tuple[SpecializedDefinition, int]]:
    """The units a binding of `TokenSelect_v1{V}` is cut into, each Definition bound and how many of it (`b1.CUTS`); a binding
    of `GreedySelect_v1{V,N}`'s are its body's, at its own chunks (`CUTS`)."""
    V, N = int(t.bindings["V"]), t.bindings.get("N")
    n = Counter(u.defn for u in plan(V, None if N is None else V // int(N)))
    return list(n.items())


def _select_sig(S):
    _whole(S)
    return (("logits", Array(S.V, BF16)),), Array(2, U16)


@composite("GreedySelect", 1, ["V", "N"], _select_sig, conformance=_CONF)
def GreedySelect(B, S, logits):
    """A row of V logits -> the token's (low, high) 16-bit words: `plan`'s units at V / N chunks, each called on its lanes
    and the combine on its chunks' pairs.  Every argument is one row of N lanes that starts at a multiple of N or one whole
    output of one unit, and the return is the combine's one field."""
    outs: dict[tuple, object] = {}
    for u in plan(S.V, S.V // S.N):
        wires = {p: outs[k][j] for p, k, j in u.wires}
        outs[u.key] = B.call(u.defn, *(logits[u.lanes[0]:u.lanes[1]] if p == "logits" else wires[p]
                                       for p, _ty in u.defn.signature()[0]))
    return outs[("combine",)]


def select(V: int, K: int | None = None) -> SpecializedDefinition:
    """`GreedySelect_v1` at `plan(V, K)`'s chunks."""
    return bind(GreedySelect, V=V, N=plan(V, K)[-1].defn.bindings["N"])


CUTS = {"GreedySelect_v1": cut}  # held to the block as its units, which its body calls (`checks.cut_of`)


# --- the twin ------------------------------------------------------------------------------------------------------------------

def _keys(x: np.ndarray) -> np.ndarray:
    """bf16 words as float64 with every NaN at -inf, so it is taken only where -inf is (never strictly above a best)."""
    f = (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return np.where(np.isnan(f), -np.inf, f)


def _first_max(x: np.ndarray) -> tuple[int, int]:
    """The scan from (-inf, 0) over bf16 words ``x``: (its first maximum above -inf, that lane), (-inf, 0) when there is none."""
    f = _keys(x)
    if not (f > -np.inf).any():
        return NEG_INF_BF16, 0
    i = int(np.argmax(f))
    return int(x[i]), i


def _nan(w: int) -> bool:
    return (w & 0x7F80) == 0x7F80 and bool(w & 0x7F)


def twin(V: int, logits, K: int | None = None) -> tuple[int, dict[tuple, list[np.ndarray]]]:
    """The token and every unit's outputs (`plan(V, K)`'s key -> its return fields, each a uint16 array: a chunk's best and
    index one word each, the combine's token its low and high words) on bf16 words ``logits``."""
    x = np.asarray(logits, dtype=np.uint16).reshape(-1)
    if x.size != V:
        raise ValueError(f"{x.size} logits, not V = {V}")
    words: dict[tuple, list[np.ndarray]] = {}
    us, pairs = plan(V, K), []
    N = V // len(us)
    for c, u in enumerate(us):
        lo, hi = u.lanes
        if c == 0:
            b, i = _first_max(x[1:hi])
            b, i = (int(x[0]), 0) if _nan(int(x[0])) or not _keys(np.array([b]))[0] > _keys(x[:1])[0] else (b, i + 1)
        else:
            b, i = _first_max(x[lo:hi])
        pairs.append((b, i))
        if c < len(us) - 1:
            words[u.key] = [np.array([b], np.uint16), np.array([i], np.uint16)]
    best, token = pairs[0][0], pairs[0][1]
    for c in range(1, len(pairs)):
        b, i = pairs[c]
        if not _nan(best) and _keys(np.array([b]))[0] > _keys(np.array([best]))[0]:
            best, token = b, c * N + i
    words[("combine",)] = [np.array([token % SPAN, token // SPAN], np.uint16)]
    return token, words


__all__ = ["UNITS", "GreedyChunk", "GreedyCombine", "GreedySelect", "STAGE_DEFINITIONS", "Unit", "carries", "cut", "plan",
           "select", "twin", "units", "width"]
