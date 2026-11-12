"""`TokenSelect_v1{V}` (greedy sampling) as units of at most a Flock block each, a tree of partial argmaxes over the row's chunks of
`CHUNK` logits (the last one the rest): `plan` lists them with their wiring, `cut` counts them by Definition (`b1.CUTS`), and
`twin` computes every unit's outputs and the token.

    GreedyChunk_v1{N,FIRST}   (logits: bf16[N]) -> (bf16, Value<16>): the chunk's best logit and its index in the chunk
    GreedyCombine_v1{V,N}     (logits: bf16[L], b0: bf16, i0: Value<16>, .., b{n-2}, i{n-2}) -> (Value<16>, Value<16>): the
                              token's low and high 16 bits, from the last of the n = ceil(V / N) chunks (its L lanes) and the
                              other chunks' pairs

The combine reads the last chunk's lanes rather than a pair for it so that its proof block composes: Flock's composer takes a
unit only with some row past 111 bytes (`windows.interface.check_proof_block`), and the pairs are 2-byte rows.  The last chunk
is the narrowest, so the combine's block stays below a full chunk's; a row whose last chunk has fewer than 56 lanes would
need it to read more.

**The function** is `TokenSelect_v1{V}`'s: from (logits[0], 0), take (x_i, i) iff ``Bf16GtStrict_v1(x_i, best)``, so the lowest
index of the largest logit wins, a NaN is never taken, +0 and -0 are equal, and the token is 0 whenever logits[0] is a NaN.  The
row's first chunk (FIRST) is that scan over its N lanes.  A later chunk is the scan from (-inf, 0): its first maximum above -inf,
and (-inf, 0) when it has none; so is the last chunk's pair inside the combine (the first's when the row is one chunk).  The
combine starts from chunk 0's pair and takes chunk c's (b_c, cN + i_c) iff
``Bf16GtStrict_v1(b_c, best)``.  Its best after chunks 0 .. c-1 is the scan's: a NaN logits[0], which neither ever leaves, or a
number at least every number in those chunks, so the scan moves in chunk c iff some lane is above it, and then ends at the
chunk's first maximum, which is b_c's lane.  Ties keep the earlier chunk, as the scan keeps the earlier lane.

**Every output is one 16-bit word.**  N is a power of two at most 2^16, so cN + i_c is ``cN mod 2^16 + i_c`` in its low word
(i_c < N adds below cN's lowest set bit) and ``cN >> 16`` in its high word: the row's i32 token as its two words, its 2^16-wide
span and its index there.  Each select keeps the running value as its *then* branch under the comparison's `BitNot`, as
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
from verity.circuits.refs import tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import prims as P
from verity.definitions.registry.b1 import NEG_INF_BF16, chunks
from verity.definitions.registry.prims import BF16

#: a full chunk's logits: a power of two, so its index's bits sit below the chunk's offset in the token's low word.  A narrower
#: one makes the combine the largest unit: it reads two rows a chunk, and each row takes an hm96 slot of its block
CHUNK = 16384
SPAN = 1 << 16
_CONF = "model-untested"


def width(V: int) -> int:
    """A full chunk's lanes on a row of V: `CHUNK`, or on a smaller row the power of two near a quarter of it, so a small row is
    a tree of a few chunks too."""
    return min(CHUNK, 1 << max(0, (V - 1).bit_length() - 2))


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


def _combine_sig(S):
    _lanes(S.N)
    if S.N & (S.N - 1) or isinstance(S.V, bool) or not isinstance(S.V, int) or not S.N <= S.V <= SPAN * SPAN:
        raise ValueError(f"a row of V = {S.V} in chunks of N = {S.N}: N a power of two, N <= V <= 2^32 (the token's two words)")
    *before, (lo, hi) = chunks(S.V, S.N)
    pairs = tuple(p for c in range(len(before)) for p in ((f"b{c}", BF16), (f"i{c}", U16)))
    return (("logits", Array(hi - lo, BF16)), *pairs), Tuple(U16, U16)


@composite("GreedyCombine", 1, ["V", "N"], _combine_sig, conformance=_CONF)
def GreedyCombine(B, S, logits, *pairs):
    """The row's last chunk of logits and its other chunks' pairs (b_c, i_c) -> the token's (low, high) 16-bit words: the last
    chunk's pair is `GreedyChunk_v1`'s, and from chunk 0's pair, chunk c's (b_c, cN + i_c) is taken iff
    ``Bf16GtStrict_v1(b_c, best)`` (the module docstring)."""
    n = len(pairs) // 2 + 1
    lo, hi = chunks(S.V, S.N)[-1]
    pairs = (*pairs, *_scan(B, logits, hi - lo, n == 1, last_best=n > 1))
    best, lo, hi = pairs[0], pairs[1], B.call(const(16, 0))
    for c in range(1, n):
        b, i = pairs[2 * c], pairs[2 * c + 1]
        keep = B.call(BitNot, B.call(P.Bf16GtStrict, b, best))
        off = c * S.N
        lo = B.call(SelectU16, keep, lo, B.call(U16Add, i, B.call(const(16, off % SPAN))) if off % SPAN else i)
        hi = B.call(SelectU16, keep, hi, B.call(const(16, off // SPAN)))
        if c < n - 1:
            best = B.call(P.SelectBf16, keep, best, b)
    return tuple_of(lo, hi)


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
def plan(V: int) -> tuple[Unit, ...]:
    """The units of a row of V in an order that writes each before its readers: its chunks of `width` lanes but the last, then
    the combine, which reads the last chunk's lanes and whose outputs are the token's words."""
    N = width(V)
    *cs, last = chunks(V, N)
    out = [Unit(("chunk", c), bind(GreedyChunk, N=hi - lo, FIRST=c == 0), (lo, hi)) for c, (lo, hi) in enumerate(cs)]
    wires = tuple(w for c in range(len(cs)) for w in ((f"b{c}", ("chunk", c), 0), (f"i{c}", ("chunk", c), 1)))
    return (*out, Unit(("combine",), bind(GreedyCombine, V=V, N=N), last, wires))


def cut(t) -> list[tuple[SpecializedDefinition, int]]:
    """The units a binding of `TokenSelect_v1{V}` is cut into, each Definition bound and how many of it (`b1.CUTS`)."""
    n = Counter(u.defn for u in plan(int(t.bindings["V"])))
    return list(n.items())


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


def twin(V: int, logits) -> tuple[int, dict[tuple, list[np.ndarray]]]:
    """The token and every unit's outputs (`plan`'s key -> its return fields, each a uint16 array of one word) on bf16 words
    ``logits``."""
    x = np.asarray(logits, dtype=np.uint16).reshape(-1)
    if x.size != V:
        raise ValueError(f"{x.size} logits, not V = {V}")
    words: dict[tuple, list[np.ndarray]] = {}
    N, pairs = width(V), []
    cs = chunks(V, N)
    for c, (lo, hi) in enumerate(cs):
        if c == 0:
            b, i = _first_max(x[1:hi])
            b, i = (int(x[0]), 0) if _nan(int(x[0])) or not _keys(np.array([b]))[0] > _keys(x[:1])[0] else (b, i + 1)
        else:
            b, i = _first_max(x[lo:hi])
        pairs.append((b, i))
        if c < len(cs) - 1:
            words[("chunk", c)] = [np.array([b], np.uint16), np.array([i], np.uint16)]
    best, token = pairs[0][0], pairs[0][1]
    for c in range(1, len(pairs)):
        b, i = pairs[c]
        if not _nan(best) and _keys(np.array([b]))[0] > _keys(np.array([best]))[0]:
            best, token = b, c * N + i
    words[("combine",)] = [np.array([token % SPAN], np.uint16), np.array([token // SPAN], np.uint16)]
    return token, words


__all__ = ["CHUNK", "GreedyChunk", "GreedyCombine", "STAGE_DEFINITIONS", "Unit", "cut", "plan", "twin", "width"]
