"""The length rule a request Program returns as one of its checks, which the verifier requires to be 1: `io_shell.StopRule_v1{L,
STOP, FINISH}` over the Program's L outputs, STOP the registered model's stop ids (`stop_ids`) and FINISH how the request's
outputs end in this Program (`spec`). Its config is the derive config's ``stop`` (the Build's `--stop`) and `build_program`'s
``stop`` alike, which the SRC's ``config_sha256`` holds, so a predicted Program and its trace carry the same check, the last
Call of the body and the last check. Imports without torch."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array, Value
from verity.ml.definitions.registry import io_shell as IO

#: the rule's Call's name in a request Program's body
NAME = "stop_rule"


def stop_ids(hf_config: Mapping[str, Any]) -> tuple[int, ...]:
    """The model's stop ids, its config's ``eos_token_id`` (an id, a list or none), increasing: outputs end at the first, kept."""
    e = hf_config.get("eos_token_id")
    return tuple(sorted({int(x) for x in ([] if e is None else [e] if isinstance(e, int) else e)}))


def spec(ids: Iterable[int], finish: str) -> dict:
    """The rule's config, ``{"ids": [...], "finish": ...}``: FINISH ``eos`` where this Program's last output ends the request
    on a stop id, ``length`` where none of its outputs is one; a ValueError where `StopRule_v1` has no rule for it."""
    s = {"ids": [int(x) for x in ids], "finish": finish}
    if why := IO.stop_refuse({"STOP": tuple(s["ids"]), "FINISH": finish}):
        raise ValueError(f"the stop rule's config {s}: {why}")
    return s


def what(config: Mapping[str, Any]) -> str:
    """The check's description, as the provenance's `checks` names it."""
    return (f"the length rule (StopRule_v1, finish {config['finish']}): no stop id among the outputs before the last, and the last "
            f"one exactly when the request finished on one")


def check(B: Any, ids: Coll, config: Mapping[str, Any]) -> tuple[Coll, str]:
    """The rule over ``ids``, the Program's outputs (an Array of i32), as the next Call of the Builder ``B``, and its
    description: (the bit, what it checks), the form `with_checks` and `derive.Ctx.check` take."""
    if not (isinstance(ids.type, Array) and ids.type.n >= 1 and ids.type.elem == Value(32)):
        raise ValueError(f"the stop rule reads a request's outputs, an Array of at least one i32, not {ids.type}")
    s = spec(config["ids"], config["finish"])
    fn = bind(IO.StopRule, L=ids.type.n, STOP=tuple(s["ids"]), FINISH=s["finish"])
    return B.call(fn, ids, name=NAME), what(s)
