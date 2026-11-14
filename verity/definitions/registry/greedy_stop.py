"""A request's length check as a unit of at most a Flock block: `StopCheck_v1` over the request's T output ids, and that
the last id is the greedy select's token over its last row of logits, recomputed by `greedy_stages.GreedyCombine_v2`.

    GreedyStopCheck_v1{V,N,T,STOP,MAX_TOKENS,PROMPT_LEN,MAX_MODEL_LEN,FINISH}
        (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}, ids: Array<T, i32>) -> bit

is 1 exactly when `StopCheck_v1{T,STOP,MAX_TOKENS,PROMPT_LEN,MAX_MODEL_LEN,FINISH}` over ``ids`` is 1 and ``ids[T - 1]``
is `GreedyCombine_v2{V,N}`'s token over ``logits`` and the results ``r<c>``, its low and high words.  `reference` is its
function.

**Why the combine is in it.**  Flock's composer takes a unit only with some row past 111 bytes, and `StopCheck_v1`'s rows
are its ids, four bytes each: a request of a few outputs is no block alone.  The combine's logits row is, as in
`greedy_stages` (the combine reads the last chunk's lanes for the same reason), and the token it recomputes is the last id
the check reads, so the logits are no input the check ignores.

**The words compare as their widenings.**  The token is two 16-bit words and the id one 32-bit word, and no primitive widens
the one to the other or compares 16-bit words.  `Bf16ToF32_v1` is a 16-bit word shifted up 16 bits, exactly, on every
word, so it is injective: the id's high half (`U32Hi16_v1`) and the token's high word are equal iff their widenings are
(`I32Eq_v1`), and the id's low half and the token's low word are iff the id doubled 16 times (`I32Add_v1` of a word and
itself, its low half moved up) equals the low word's widening.

Registered functions:
  GreedyStopCheck{V,N,T,STOP,MAX_TOKENS,PROMPT_LEN,MAX_MODEL_LEN,FINISH}  (logits: N bf16, r<c>: 3 u16, ids: T i32) -> bit
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from verity.circuits.defs import bind, composite
from verity.circuits.ops.scalar import BitAnd, I32Add, I32Eq, U32Hi16
from verity.circuits.types import Array
from verity.definitions.registry import greedy_stages as G
from verity.definitions.registry import stop_check as SC
from verity.definitions.registry.prims import BIT, I32, Bf16ToF32

STATICS = ["V", "N", "T", "STOP", "MAX_TOKENS", "PROMPT_LEN", "MAX_MODEL_LEN", "FINISH"]


def refuse(S: dict) -> str | None:
    """`stop_check.refuse` on the check's statics; the combine's are refused by its signature."""
    return SC.refuse(S)


def _sig(S):
    sig, _token = G._combine_sig(S)
    return (*sig, ("ids", Array(S.T, I32))), BIT


def reference(row: np.ndarray, ids: Sequence[int], V: int, N: int, STOP: tuple[int, ...], MAX_TOKENS: int, PROMPT_LEN: int,
              MAX_MODEL_LEN: int, FINISH: int) -> int:
    """`GreedyStopCheck_v1`'s function on the row's V logits (bf16 words), whose last N are ``logits`` and whose other
    chunks' results are the ``r<c>`` (`greedy_stages.twin`), and T = len(ids) i32 encodings."""
    token, _units = G.twin(V, row, V // N)
    return SC.reference(ids, STOP, MAX_TOKENS, PROMPT_LEN, MAX_MODEL_LEN, FINISH) & int((ids[-1] & SC.X.M32) == token)


@composite("GreedyStopCheck", 1, STATICS, _sig, refuse=refuse,
           conformance="model-untested: StopCheck_v1's, and GreedyCombine_v2's token against the last output id")
def GreedyStopCheck(B, S, logits, *rest):
    """1 exactly when `StopCheck_v1` over ``ids`` is 1 and the last id is the combine's token (the module docstring)."""
    *results, ids = rest
    token = B.call(bind(G.GreedyCombine, V=S.V, N=S.N), logits, *results)
    lo, hi = token[0], token[1]
    check = B.call(bind(SC.StopCheck, T=S.T, STOP=S.STOP, MAX_TOKENS=S.MAX_TOKENS, PROMPT_LEN=S.PROMPT_LEN,
                        MAX_MODEL_LEN=S.MAX_MODEL_LEN, FINISH=S.FINISH), ids)
    last = ids[S.T - 1]
    low = last
    for _ in range(16):
        low = B.call(I32Add, low, low)
    high = B.call(I32Eq, B.call(Bf16ToF32, B.call(U32Hi16, last)), B.call(Bf16ToF32, hi))
    return B.call(BitAnd, check, B.call(BitAnd, B.call(I32Eq, low, B.call(Bf16ToF32, lo)), high))


__all__ = ["STATICS", "GreedyStopCheck", "reference", "refuse"]
