"""`TokenSelect_v1{V}` (greedy sampling) as units of at most a Flock block each, a tree of partial argmaxes over the row's K
chunks of N = V / K logits: `plan` lists them with their wiring, `cut` counts them by Definition (`b1.CUTS`), `twin` computes
every unit's outputs and the token, and `GreedySelect_v1` is the tree as one Definition.

    GreedyChunk_v2{N,OFF}     (logits: bf16[N]) -> Array<3, Value<16>>: the best logit of the chunk at lanes OFF .. OFF + N of
                              its row, then that lane's token index OFF + i, low word and high word
    GreedyCombine_v2{V,N}     (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}) -> Array<2, Value<16>>: the token's low
                              and high 16 bits, from the last of the K = V / N chunks (its lanes) and the other chunks' results
    GreedySelect_v1{V,N}      (logits: bf16[V]) -> Array<2, Value<16>>: the same two words; its body is the plan's units

**Every chunk is whole.**  N divides V, so each unit reads N lanes that start at a multiple of N, in one row and in several
rows end to end (a replay block of several tokens: `spot.verifier.second_stage` links a unit's argument to the block's logits
only at such a start), and a chunk's three words and the token's two are each one return field, which it wires or links as one
committed row.  The plan's K is `UNITS` on a row it divides (`units`), and `plan` refuses a K that does not divide V; the
Definitions take any such N, and `GreedyChunk_v2` any OFF that is a multiple of N.

**Each chunk position is its own Definition.**  OFF is a static, so the K - 1 chunks before the last are K - 1 templates,
and the combine reads each one's result through one parameter.  In the second stage's interior a wired parameter is held at its
writer's output port (`second_stage._held_port`), and a session two of whose ports are held at one port is a form Flock's
verifier refuses (`Flock.PortRoots.dupId`): so no two parameters of a unit read one template's output, and a chunk's result
is one field, not a field a word.

The combine reads the last chunk's lanes rather than a result for it so that its proof block composes: Flock's composer takes a
unit only with some row past 111 bytes (`windows.interface.check_proof_block`), and the results are 6-byte rows; a row whose
chunks have fewer than 56 lanes would need it to read more.

**The function** is `TokenSelect_v1{V}`'s: from (logits[0], 0), take (x_i, i) iff ``Bf16GtStrict_v1(x_i, best)``, so the lowest
index of the largest logit wins, a NaN is never taken, +0 and -0 are equal, and the token is 0 whenever logits[0] is a NaN.  The
row's first chunk (OFF = 0) is that scan over its N lanes.  A later chunk is the scan from (-inf, 0): its first maximum above
-inf, and (-inf, 0) when it has none; so is the last chunk's inside the combine (the first's when the row is one chunk).  The
combine starts from chunk 0's result and takes chunk c's (b_c, cN + i_c) iff ``Bf16GtStrict_v1(b_c, best)``.  Its best after
chunks 0 .. c-1 is the scan's: a NaN logits[0], which neither ever leaves, or a number at least every number in those chunks,
so the scan moves in chunk c iff some lane is above it, and then ends at the chunk's first maximum, which is b_c's lane.  Ties
keep the earlier chunk, as the scan keeps the earlier lane, the combine's own chunk being the last.

**Every output is one 16-bit word.**  A chunk's index i is below N <= 2^16.  Its token index OFF + i is below V <= 2^32: its low
word is the wrapping sum of ``OFF mod 2^16`` and i (`U16Add_v1`), and its high word is ``OFF >> 16`` plus that sum's carry.  The
carry is 0 at every i < N unless a multiple of 2^16 falls inside the chunk, at its lane ``t = 2^16 - OFF mod 2^16`` with
0 < t < N (`carry`; no chunk has one when N is a power of two, OFF being a multiple of N); there it is 1 iff i >= t, which
the chunk computes as ``Bf16GtStrict_v1(i, t - 1)``: on the words 0 .. 0x7F80 (+0, the positive numbers and +inf, no NaN among
them) bf16's order is the integers', and both words are below N, so a chunk with a carry is at most 0x7F81 lanes
(`GreedyChunk_v2` refuses a wider one, and `GreedyCombine_v2` one whose own chunk is).  At N = 7,744 of V = 154,880 two chunks
carry: chunk 8 (OFF = 61,952) from its lane 3,584 and chunk 16 (OFF = 123,904) from its lane 7,168.  Each select keeps the
running value as its *then* branch under the comparison's `BitNot`, as ``TokenSelectChunkStep_v1``'s does, so its bits are the
new value's XOR one AND each and a chunk lowers to a circuit linear in N.
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
from verity.circuits.refs import array_of
from verity.circuits.types import Array
from verity.definitions.registry import prims as P
from verity.definitions.registry.b1 import NEG_INF_BF16
from verity.definitions.registry.prims import BF16

#: the chunks a served row is cut into, a number: any divisor of V is a plan of the same Definitions (`plan`'s K).  More chunks
#: are smaller chunk blocks, a Definition more each, and a larger combine, which reads a row a chunk, an hm96 slot of its block
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


def carry(N: int, OFF: int) -> int | None:
    """The lane t of the chunk at lanes OFF .. OFF + N that a multiple of 2^16 falls at, 0 < t < N: OFF + i is in the next
    2^16-wide span iff i >= t, so its high word is ``OFF >> 16`` plus [i >= t].  None for a chunk inside one span."""
    t = SPAN - OFF % SPAN
    return t if t < N else None


def carries(V: int, N: int) -> dict[int, int]:
    """Each chunk c of a row of V in chunks of N that a multiple of 2^16 falls inside -> its lane there (`carry`)."""
    return {c: t for c in range(V // N) if (t := carry(N, c * N)) is not None}


def _lanes(N) -> None:
    if isinstance(N, bool) or not isinstance(N, int) or not 1 <= N <= SPAN:
        raise ValueError(f"N is a chunk's lanes, an int in [1, 2^16] (its index is a 16-bit word), not {N!r}")


def _at(N, OFF) -> None:
    """Refuse a chunk of N lanes at OFF that is no whole chunk of a row, or whose token index is not two words as computed."""
    _lanes(N)
    if isinstance(OFF, bool) or not isinstance(OFF, int) or OFF < 0 or OFF % N or OFF + N > SPAN * SPAN:
        raise ValueError(f"OFF = {OFF!r} is a chunk's first lane in its row: a multiple of N = {N} (every chunk is whole) with "
                         "OFF + N <= 2^32 (the token index's two words)")
    if N - 1 > ORDERED and carry(N, OFF) is not None:
        raise ValueError(f"a chunk of N = {N} lanes at OFF = {OFF}: a multiple of 2^16 falls inside it, and its carry compares "
                         f"indices as bf16 words, in the integers' order only up to {ORDERED:#x}: N <= {ORDERED + 1}, or a "
                         "power of two")


def _chunk_sig(S):
    _at(S.N, S.OFF)
    return (("logits", Array(S.N, BF16)),), Array(3, U16)


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


def _token(B, i, n: int, off: int):
    """The token index ``off + i`` of lane ``i`` of the chunk of ``n`` lanes at ``off`` -> its (low, high) 16-bit words: the
    wrapping sum, and ``off >> 16`` plus its carry, 1 from the chunk's lane `carry` names (the module docstring)."""
    lo = B.call(U16Add, i, B.call(const(16, off % SPAN))) if off % SPAN else i
    hi, t = B.call(const(16, off // SPAN)), carry(n, off)
    if t is not None:
        past = B.call(P.Bf16GtStrict, i, B.call(const(16, t - 1)))
        hi = B.call(SelectU16, past, B.call(const(16, off // SPAN + 1)), hi)
    return lo, hi


@composite("GreedyChunk", 2, ["N", "OFF"], _chunk_sig, conformance=_CONF)
def GreedyChunk(B, S, logits):
    """The chunk of N logits at lanes OFF .. OFF + N of its row -> one field of three words: its best logit, then that
    lane's token index OFF + i, low word and high word.  At OFF = 0 the scan is `TokenSelect_v1`'s over them from
    (logits[0], 0); else the scan from (-inf, 0), (-inf, OFF) when no lane is above -inf (the module docstring)."""
    best, i = _scan(B, logits, S.N, S.OFF == 0)
    return array_of([best, *_token(B, i, S.N, S.OFF)])


def _whole(S) -> None:
    _lanes(S.N)
    if isinstance(S.V, bool) or not isinstance(S.V, int) or not S.N <= S.V <= SPAN * SPAN or S.V % S.N:
        raise ValueError(f"a row of V = {S.V} in chunks of N = {S.N}: N divides V (every chunk is whole) and V <= 2^32 (the "
                         "token's two words)")


def _combine_sig(S):
    _whole(S)
    _at(S.N, S.V - S.N)
    return (("logits", Array(S.N, BF16)), *((f"r{c}", Array(3, U16)) for c in range(S.V // S.N - 1))), Array(2, U16)


@composite("GreedyCombine", 2, ["V", "N"], _combine_sig, conformance=_CONF)
def GreedyCombine(B, S, logits, *results):
    """The row's last chunk of logits and its other chunks' results (`GreedyChunk_v2`'s three words each) -> the token's
    (low, high) 16-bit words, one field: the last chunk's result is `GreedyChunk_v2`'s at OFF = V - N, and from chunk 0's
    result, chunk c's token index is taken iff ``Bf16GtStrict_v1(b_c, best)`` (the module docstring)."""
    n = S.V // S.N
    best, i = _scan(B, logits, S.N, n == 1, last_best=n > 1)
    results = (*results, (best, *_token(B, i, S.N, S.V - S.N)))
    best, lo, hi = results[0][0], results[0][1], results[0][2]
    for c in range(1, n):
        b = results[c][0]
        keep = B.call(BitNot, B.call(P.Bf16GtStrict, b, best))
        lo = B.call(SelectU16, keep, lo, results[c][1])
        hi = B.call(SelectU16, keep, hi, results[c][2])
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
    its chunks of V / K lanes but the last, each its own Definition (its OFF), then the combine, which reads the last chunk's
    lanes and whose output is the token's words.  A K that does not divide V is refused: a narrower last chunk starts off the
    chunks' rows."""
    K = units(V) if K is None else K
    if isinstance(K, bool) or not isinstance(K, int) or K < 1 or V % K:
        raise ValueError(f"a row of V = {V} is not K = {K!r} whole chunks: K divides V")
    N = V // K
    for c in range(K):
        _at(N, c * N)
    out = [Unit(("chunk", c), bind(GreedyChunk, N=N, OFF=c * N), (c * N, (c + 1) * N)) for c in range(K - 1)]
    wires = tuple((f"r{c}", ("chunk", c), 0) for c in range(K - 1))
    return (*out, Unit(("combine",), bind(GreedyCombine, V=V, N=N), ((K - 1) * N, V), wires))


def cut(t) -> list[tuple[SpecializedDefinition, int]]:
    """The units a binding of `TokenSelect_v1{V}` is cut into, each Definition bound and how many of it (`b1.CUTS`); a binding
    of `GreedySelect_v1{V,N}`'s are its body's, at its own chunks (`CUTS`)."""
    V, N = int(t.bindings["V"]), t.bindings.get("N")
    n = Counter(u.defn for u in plan(V, None if N is None else V // int(N)))
    return list(n.items())


def _select_sig(S):
    _whole(S)
    if S.N - 1 > ORDERED and carries(S.V, S.N):
        _at(S.N, min(carries(S.V, S.N)) * S.N)
    return (("logits", Array(S.V, BF16)),), Array(2, U16)


@composite("GreedySelect", 1, ["V", "N"], _select_sig, conformance=_CONF)
def GreedySelect(B, S, logits):
    """A row of V logits -> the token's (low, high) 16-bit words: `plan`'s units at V / N chunks, each called on its lanes
    and the combine on its chunks' results.  Every argument is one row of N lanes that starts at a multiple of N or the one
    whole output of one unit, no two of a unit's from one Definition, and the return is the combine's one field."""
    outs: dict[tuple, object] = {}
    for u in plan(S.V, S.V // S.N):
        wires = {p: outs[k] for p, k, _field in u.wires}
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
    """The token and every unit's outputs (`plan(V, K)`'s key -> its one return field, a uint16 array: a chunk's best and its
    token index's low and high words, the combine's token its low and high words) on bf16 words ``logits``."""
    x = np.asarray(logits, dtype=np.uint16).reshape(-1)
    if x.size != V:
        raise ValueError(f"{x.size} logits, not V = {V}")
    words: dict[tuple, list[np.ndarray]] = {}
    us, results = plan(V, K), []
    for c, u in enumerate(us):
        lo, hi = u.lanes
        if c == 0:
            b, i = _first_max(x[1:hi])
            b, i = (int(x[0]), 0) if _nan(int(x[0])) or not _keys(np.array([b]))[0] > _keys(x[:1])[0] else (b, i + 1)
        else:
            b, i = _first_max(x[lo:hi])
        results.append((b, lo + i))
        if c < len(us) - 1:
            words[u.key] = [np.array([b, (lo + i) % SPAN, (lo + i) // SPAN], np.uint16)]
    best, token = results[0]
    for b, at in results[1:]:
        if not _nan(best) and _keys(np.array([b]))[0] > _keys(np.array([best]))[0]:
            best, token = b, at
    words[("combine",)] = [np.array([token % SPAN, token // SPAN], np.uint16)]
    return token, words


__all__ = ["UNITS", "GreedyChunk", "GreedyCombine", "GreedySelect", "STAGE_DEFINITIONS", "Unit", "carries", "carry", "cut",
           "plan", "select", "twin", "units", "width"]
