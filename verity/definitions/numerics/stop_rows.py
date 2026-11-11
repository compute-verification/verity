"""The replay row kernel of a request's length check, `StopCheck_v1` (`registry.stop_check`, a Call of `registry.stop_rule`'s
`StopRule_v1`): its Definition evaluated by `verity.circuits.evaluation.evaluate_batch` with the IR's reference on the row's T
committed ids.  The statics are read as the spec text renders them (STOP a JSON list, the rest integers); an id word past 32 bits
is declined, never wrapped onto a stop id.  Registers from `rows` (kernel "rows").
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from verity.definitions.numerics.kernel_registry import Declined

FAMILY = "StopCheck_v1"
EVALUATOR = ("verity.definitions.numerics.stop_rows.stop_check (StopCheck_v1{T,STOP,MAX_TOKENS,PROMPT_LEN,MAX_MODEL_LEN,FINISH}: "
             "the Definition by evaluate_batch's IR reference over the row's committed ids; registry.stop_check)")


def statics_of(statics: dict[str, Any]) -> dict[str, Any]:
    """`statics` as `bind` takes them: STOP a tuple and the rest ints, from the spec text's JSON or as a binding record holds them."""
    def one(v: Any) -> Any:
        v = json.loads(v) if isinstance(v, str) else v
        return tuple(int(s) for s in v) if isinstance(v, (list, tuple)) else int(v)
    return {k: one(statics[k]) for k in ("T", "STOP", "MAX_TOKENS", "PROMPT_LEN", "MAX_MODEL_LEN", "FINISH")}


def stop_check(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    from verity.circuits.defs import bind
    from verity.circuits.evaluation import evaluate_batch
    from verity.definitions.registry import stop_check as SC
    S = statics_of(statics)
    if len(ops) != 1:
        raise Declined(f"{FAMILY}: the row wires {len(ops)} args, the Definition takes 1")
    ids = np.asarray(ops[0]).reshape(-1)
    if ids.size != S["T"]:
        raise Declined(f"{FAMILY}: {ids.size} id words, T = {S['T']}")
    if ids.dtype.kind not in "iu":
        raise Declined(f"{FAMILY}: id words of {ids.dtype}, not integer words")
    if ids.dtype.itemsize > 4 and ids.size and (int(ids.min()) < 0 or int(ids.max()) >> 32):
        raise Declined(f"{FAMILY}: an id word past 32 bits, which no I32 output holds")
    words = (ids.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    out = evaluate_batch(bind(SC.StopCheck, **S), words.reshape(1, -1), kernel="reference").reshape(-1)
    return {"out": out.astype(np.uint8)}


def check_targets(rng) -> list:
    from verity.circuits.defs import bind
    from verity.definitions.registry import stop_check as SC
    return [bind(SC.StopCheck, T=4, STOP=stop, MAX_TOKENS=4, PROMPT_LEN=8, MAX_MODEL_LEN=64, FINISH=f)
            for stop, f in (((3, 7), 0), ((3, 7), 1), ((3, 7), 2), ((), 2))]


__all__ = ["EVALUATOR", "FAMILY", "check_targets", "statics_of", "stop_check"]
