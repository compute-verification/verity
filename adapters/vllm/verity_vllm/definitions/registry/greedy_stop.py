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

**A class's check.**  `GreedyStopCheckClass_v1{V,N,P,T,STOP,MAX_MODEL_LEN}` is the length check of a class's Program (P
prompt ids, the prompt's exact length, a power of two or not; T = O served ids), whose selects carry a bit and a count step
to step (`greedy_stages.GreedyCombineLive_v1`):

    (logits: bf16[N], r0, .., r{K-2}, ids: Array<T, i32>, request: Array<4, i32>, count: Array<16, bit>,
     live: Array<16, bit>) -> bit

with ``request`` the request row's first four fields (``lp``, ``max_tokens``, ``min_tokens``, ``ignore_eos``), the ones
every select reads, and ``count`` and ``live`` the last select's ``o2`` and ``o1``, a u16 word each
(`greedy_stages.WORD_BITS`), of which it reads the count's low log2(T) + 1 bits and the bit's bit 0.  It is 1 exactly when `StopRule_v1{T}` over ``ids``, ``max_tokens``, the stop ids STOP (none where ``ignore_eos``
is not 0: the one word `stop_check.NOT_AN_ID`), ``lp`` and MAX_MODEL_LEN gives (n, finish) with finish not 0 and n equal to
``count``, ``min_tokens`` is 0, ``lp`` is P, every id from the n-th on is 0, ``live`` is 0, and, where n is T, ``ids[T -
1]`` is the combine's token.  `class_reference` is its function.  Holding n equal to ``count`` ties the rule's n to the
count the selects carried, so the two cannot be two numbers; ``live`` 0 says the selects' chain has stopped, which it does
where the rule ends or at T.  Refused by its bit until their rules are written: a request the rule does not end within T, a
``min_tokens`` other than 0 (the rule doesn't read it) and a prompt of other than P ids.

Registered functions:
  GreedyStopCheck{V,N,T,STOP,MAX_TOKENS,PROMPT_LEN,MAX_MODEL_LEN,FINISH}  (logits: N bf16, r<c>: 3 u16, ids: T i32) -> bit
  GreedyStopCheckClass_v1{V,N,P,T,STOP,MAX_MODEL_LEN}  (logits: N bf16, r<c>: 3 u16, ids: T i32, request: 4 i32, count: 16 bits,
                                                   live: 16 bits) -> bit
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from verity.circuits.defs import bind, composite
from verity.circuits.ops.scalar import BitAnd, I32Add, I32Eq, U32Hi16
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array

from verity_vllm.definitions.registry import greedy_stages as G
from verity_vllm.definitions.registry import stop_check as SC
from verity_vllm.definitions.registry.prims import BIT, I32, Bf16ToF32

STATICS = ["V", "N", "T", "STOP", "MAX_TOKENS", "PROMPT_LEN", "MAX_MODEL_LEN", "FINISH"]
#: `GreedyStopCheckClass_v1`'s statics: the class's (P, T) and its stop ids and model length, no request's
CLASS_STATICS = ["V", "N", "P", "T", "STOP", "MAX_MODEL_LEN"]


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


def _is_token(B, last, token):
    """Whether the i32 id ``last`` is the token, its (low, high) words: both halves compared as their widenings (the module
    docstring)."""
    lo, hi = token[0], token[1]
    low = last
    for _ in range(16):
        low = B.call(I32Add, low, low)
    high = B.call(I32Eq, B.call(Bf16ToF32, B.call(U32Hi16, last)), B.call(Bf16ToF32, hi))
    return B.call(BitAnd, B.call(I32Eq, low, B.call(Bf16ToF32, lo)), high)


@composite("GreedyStopCheck", 1, STATICS, _sig, refuse=refuse,
           conformance="model-untested: StopCheck_v1's, and GreedyCombine_v2's token against the last output id")
def GreedyStopCheck(B, S, logits, *rest):
    """1 exactly when `StopCheck_v1` over ``ids`` is 1 and the last id is the combine's token (the module docstring)."""
    *results, ids = rest
    token = B.call(bind(G.GreedyCombine, V=S.V, N=S.N), logits, *results)
    check = B.call(bind(SC.StopCheck, T=S.T, STOP=S.STOP, MAX_TOKENS=S.MAX_TOKENS, PROMPT_LEN=S.PROMPT_LEN,
                        MAX_MODEL_LEN=S.MAX_MODEL_LEN, FINISH=S.FINISH), ids)
    return B.call(BitAnd, check, _is_token(B, ids[S.T - 1], token))


def refuse_class(S: dict) -> str | None:
    """A class's check with no rule: T as `greedy_stages.refuse_class` refuses O, P no prompt length (an int in
    [1, 2^30]: a class is keyed by its prompt's exact length, a power of two or not), and STOP and MAX_MODEL_LEN as
    `stop_check.refuse` refuses them.  Only the statics bound so far."""
    p = S.get("P", 1)
    if not (type(p) is int and 1 <= p <= 1 << 30):
        return f"P is a class's prompt length, an int in [1, 2^30] (got P={p!r})"
    return G.refuse_class({("O" if k == "T" else k): v for k, v in S.items() if k in ("T", "STOP", "MAX_MODEL_LEN")})


def _class_sig(S):
    sig, _token = G._combine_sig(S)
    return (*sig, ("ids", Array(S.T, I32)), ("request", Array(4, I32)), ("count", G.WORD_BITS), ("live", G.WORD_BITS)), BIT


def class_reference(row: np.ndarray, ids: Sequence[int], request: Sequence[int], count: int, live: int, V: int, N: int,
                    P: int, STOP: tuple[int, ...], MAX_MODEL_LEN: int) -> int:
    """`GreedyStopCheckClass_v1`'s function on the row's V logits (as `reference`), T = len(ids) i32 encodings, the i32 limits
    ``request``, and the last select's ``count`` and ``live``."""
    X = SC.X
    lp, max_tokens, min_tokens, ignore_eos = (int(x) & X.M32 for x in request)
    ids = [int(i) & X.M32 for i in ids]
    T = len(ids)
    n, finish = X.reference(ids, max_tokens, STOP if STOP and not ignore_eos else (SC.NOT_AN_ID,), lp, MAX_MODEL_LEN)
    token, _units = G.twin(V, row, V // N)
    return int(finish != 0 and n == count and min_tokens == 0 and lp == P and all(i == 0 for i in ids[n:]) and not live
               and (n < T or ids[T - 1] == token))


@composite("GreedyStopCheckClass", 1, CLASS_STATICS, _class_sig, refuse=refuse_class,
           conformance="model-untested: StopRule_v1's (n, finish) against the selects' count, and GreedyCombine_v2's token")
def GreedyStopCheckClass(B, S, logits, *rest):
    """1 exactly when `StopRule_v1` over the class's ids and the request's limits ends them, at the n the last select's
    ``count`` holds, with the ids past n 0, ``live`` 0, ``min_tokens`` 0, ``lp`` P, and the last id the combine's token
    where n is T (the module docstring)."""
    X = SC.X
    *results, ids, request, count, live = rest
    token = B.call(bind(G.GreedyCombine, V=S.V, N=S.N), logits, *results)
    o = X.word_ops(B)
    lp, max_tokens, min_tokens, ignore_eos = (request[i] for i in range(4))
    if S.STOP:
        keep = o.ieq(ignore_eos, o.word(0))
        words = [o.isel(keep, o.word(s), o.word(SC.NOT_AN_ID)) for s in S.STOP]
    else:
        words = [o.word(SC.NOT_AN_ID)]
    rule = B.call(bind(X.StopRule, T=S.T, NSTOP=len(words)), ids, max_tokens,
                  Coll(Array(len(words), I32), concat([w.refs for w in words])), lp, o.word(S.MAX_MODEL_LEN))
    n, finish = rule[0], rule[1]
    ok = o.band(o.bnot(o.ieq(finish, o.word(0))), o.ieq(n, G.count_word(o, count, G.count_bits(S.T))))
    ok = o.band(ok, o.ieq(min_tokens, o.word(0)))
    ok = o.band(ok, o.ieq(lp, o.word(S.P)))
    ok = o.band(ok, o.bnot(live[0]))
    within = {i: o.ile(n, o.word(i)) for i in range(1, S.T)}
    for i in range(1, S.T):
        ok = o.band(ok, o.bor(o.bnot(within[i]), o.ieq(ids[i], o.word(0))))
    return o.band(ok, o.bor(within[S.T - 1], _is_token(B, ids[S.T - 1], token)))


__all__ = ["CLASS_STATICS", "STATICS", "GreedyStopCheck", "GreedyStopCheckClass", "class_reference", "reference", "refuse",
           "refuse_class"]
