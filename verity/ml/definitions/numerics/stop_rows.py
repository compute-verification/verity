"""The replay row kernel of the length rule, `StopRule_v1{L,STOP,FINISH}` (`registry.io_shell`): its Definition evaluated by
`verity.ml.circuits.evaluation.evaluate_batch` with the IR's reference on the row's L committed ids.  The statics are read as the
spec text renders them (STOP a JSON list, FINISH a JSON string); an id word past 32 bits is declined, never wrapped onto a stop id.
Registers from `rows` (kernel "rows").
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from verity.ml.definitions.numerics.kernel_registry import Declined

FAMILY = "StopRule_v1"
EVALUATOR = ("verity.ml.definitions.numerics.stop_rows.stop_rule (StopRule_v1{L,STOP,FINISH}: the Definition by evaluate_batch's IR "
             "reference over the row's committed ids; registry.io_shell)")


def statics_of(statics: dict[str, Any]) -> dict[str, Any]:
    """`statics` as `bind` takes them: STOP a tuple and FINISH a str, from the spec text's JSON or as a binding record holds them."""
    def one(v: Any) -> Any:
        return json.loads(v) if isinstance(v, str) and v[:1] in '["' else v
    stop, finish = one(statics["STOP"]), one(statics["FINISH"])
    return {"L": int(statics["L"]), "STOP": tuple(int(s) for s in stop), "FINISH": str(finish)}


def stop_rule(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity.ml.definitions.registry import io_shell as IO
    S = statics_of(statics)
    if len(ops) != 1:
        raise Declined(f"{FAMILY}: the row wires {len(ops)} args, the Definition takes 1")
    ids = np.asarray(ops[0]).reshape(-1)
    if ids.size != S["L"]:
        raise Declined(f"{FAMILY}: {ids.size} id words, L = {S['L']}")
    if ids.dtype.kind not in "iu":
        raise Declined(f"{FAMILY}: id words of {ids.dtype}, not integer words")
    if ids.dtype.itemsize > 4 and ids.size and (int(ids.min()) < 0 or int(ids.max()) >> 32):
        raise Declined(f"{FAMILY}: an id word past 32 bits, which no I32 output holds")
    words = (ids.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    out = evaluate_batch(bind(IO.StopRule, **S), words.reshape(1, -1), kernel="reference").reshape(-1)
    return {"out": out.astype(np.uint8)}


def check_targets(rng) -> list:
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import io_shell as IO
    return [bind(IO.StopRule, L=4, STOP=(3, 7), FINISH=f) for f in IO.FINISHES]


__all__ = ["EVALUATOR", "FAMILY", "check_targets", "statics_of", "stop_rule"]
