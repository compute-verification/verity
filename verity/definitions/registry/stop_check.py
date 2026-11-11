"""A request's length check: whether its outputs end where `stop_rule.StopRule_v1` computes, as one Call a Program returns as a
check (`adapters/vllm`'s `program.frontend.stop_rule`).

`StopCheck_v1{T, STOP, MAX_TOKENS, PROMPT_LEN, MAX_MODEL_LEN, FINISH}` (ids: Array<T, i32>) -> bit is 1 exactly when
`StopRule_v1{T, NSTOP}` over the T outputs ``ids``, the stop ids STOP and the constants MAX_TOKENS, PROMPT_LEN and
MAX_MODEL_LEN, gives (n, finish) = (T, FINISH): the outputs stop at their last with FINISH 1 (a stop id), 2 (``max_tokens``)
or 3 (the model's length), or run on past it with FINISH 0.  The rule is `StopRule_v1`'s; this compares its two words.

With no stop ids (STOP empty: a request under ignore_eos, or a model that states none) the rule reads the one word
`NOT_AN_ID`, since it takes at least one.  FINISH is never 1 then, so an output equal to that word can only make the check 0:
it accepts a subset of what the rule over no stop ids accepts, the same set on every stream of token ids.  `reference` is its
function.

Registered functions:
  StopCheck{T,STOP,MAX_TOKENS,PROMPT_LEN,MAX_MODEL_LEN,FINISH}  (ids: T i32) -> bit
"""
from __future__ import annotations

from collections.abc import Sequence

from verity.circuits.defs import bind, composite
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.definitions.registry import stop_rule as X
from verity.definitions.registry.prims import BIT, I32

#: the stop id the rule reads where a request has none: no token id, the i32 word -1
NOT_AN_ID = 0xFFFFFFFF


def refuse(S: dict) -> str | None:
    """A binding with no check: T outside `StopRule_v1`'s, STOP not increasing u32 ids, a constant outside [0, 2^31), or
    FINISH not a code of `stop_rule.FINISH` (1 only with a stop id).  Only the statics bound so far."""
    t, stop, fin = S.get("T", 1), S.get("STOP", ()), S.get("FINISH", 0)
    consts = [S.get(k, 0) for k in ("MAX_TOKENS", "PROMPT_LEN", "MAX_MODEL_LEN")]
    if not (type(t) is int and 1 <= t < (1 << 31) - 1 and isinstance(stop, tuple) and len(stop) < 1 << 16
            and all(type(s) is int and 0 <= s <= X.M32 for s in stop) and list(stop) == sorted(set(stop))
            and all(type(c) is int and 0 <= c < 1 << 31 for c in consts) and type(fin) is int and fin in X.FINISH
            and (fin != 1 or stop)):
        return (f"T must be in [1, 2^31 - 1), STOP increasing u32 ids, MAX_TOKENS, PROMPT_LEN and MAX_MODEL_LEN in [0, 2^31) and "
                f"FINISH one of {sorted(X.FINISH)}, 1 only with a stop id (got T={t!r}, STOP={stop!r}, constants {consts!r}, "
                f"FINISH={fin!r})")
    return None


def reference(ids: Sequence[int], STOP: tuple[int, ...], MAX_TOKENS: int, PROMPT_LEN: int, MAX_MODEL_LEN: int, FINISH: int) -> int:
    """`StopCheck_v1`'s function on i32 encodings, T = len(ids)."""
    return int(X.reference(ids, MAX_TOKENS, STOP or (NOT_AN_ID,), PROMPT_LEN, MAX_MODEL_LEN) == (len(ids), FINISH))


@composite("StopCheck", 1, ["T", "STOP", "MAX_TOKENS", "PROMPT_LEN", "MAX_MODEL_LEN", "FINISH"],
           lambda S: ((("ids", Array(S.T, I32)),), BIT), refuse=refuse,
           conformance="model-untested: StopRule_v1's (n, finish) against the Program's output count and its finish")
def StopCheck(B, S, ids):
    """1 exactly when `StopRule_v1` over ``ids`` and the statics' constants gives (T, FINISH) (the module docstring)."""
    o, stops = X.word_ops(B), S.STOP or (NOT_AN_ID,)
    words = [o.word(s) for s in stops]
    r = B.call(bind(X.StopRule, T=S.T, NSTOP=len(stops)), ids, o.word(S.MAX_TOKENS),
               Coll(Array(len(stops), I32), concat([w.refs for w in words])), o.word(S.PROMPT_LEN), o.word(S.MAX_MODEL_LEN))
    return o.band(o.ieq(r[0], o.word(S.T)), o.ieq(r[1], o.word(S.FINISH)))
