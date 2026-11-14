"""The length rule a request Program returns as one of its checks, which the verifier requires to be 1: `stop_check.StopCheck_v1`
over the Program's outputs, a Call of `stop_rule.StopRule_v1` (vLLM's `check_stop`: the first output that is a stop id, else the
first that reaches max_tokens or the model's length) whose (n, finish) must be the Program's last output and the finish the
request had there (`of`). The stop ids are the registered model's (`stop_ids`), none under ignore_eos, and a body's own stop
fields are refused (`body_refusal`). A Program whose outputs follow earlier ones (a window carrying its request in) reads the rule
from its first output on: max_tokens less those and the prompt with them (`check`). Its config is the derive config's ``stop``
(the Build's `--stop`) and `build_program`'s ``stop`` alike, which the SRC's ``config_sha256`` holds, so a predicted Program and
its trace carry the same check, the last Call of the body and the last check. Imports without torch.

A class's Definition, which every request of its (n outputs) runs, holds of the rule only what they share: the stop ids and the
model's length (`CLASS_KEYS`). It computes the rule (`StopClass_v1`, `class_definition`) over its outputs and the request's
max_tokens and prompt length, values it reads (`request_values`), and gives (n, finish) where a request Program checks them
against its config's: where that check is 1, the class's Call gives (n, the config's finish), and only there."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array, Tuple, Value
from verity.definitions.registry import stop_check as SC
from verity.definitions.registry import stop_rule as X
from verity.definitions.registry.prims import I32

#: the rule's Call's name in a request Program's body
NAME = "stop_rule"
#: the rule's config's keys (`spec`)
KEYS = ("stops", "max_tokens", "prompt_len", "max_model_len", "outputs", "finish")
#: the config's keys a class's Definition holds as statics (`class_definition`); a request's max_tokens and prompt length are values
CLASS_KEYS = ("stops", "max_model_len")


def stop_ids(hf_config: Mapping[str, Any]) -> tuple[int, ...]:
    """The model's stop ids, its config's ``eos_token_id`` (an id, a list or none), increasing: outputs end at the first, kept."""
    e = hf_config.get("eos_token_id")
    return tuple(sorted({int(x) for x in ([] if e is None else [e] if isinstance(e, int) else e)}))


def spec(stops: Iterable[int], *, max_tokens: int, prompt_len: int, max_model_len: int, outputs: int, finish: int) -> dict:
    """The rule's config: the stop ids, the request's max_tokens, prompt length and the model's length, its outputs through the
    Program's last, and the finish code there (`stop_rule.FINISH`: 0 where it runs on); a ValueError where `StopCheck_v1` has
    no check for it."""
    ids = [int(x) for x in stops]
    s = {"stops": ids, "max_tokens": max_tokens, "prompt_len": prompt_len, "max_model_len": max_model_len,
         "outputs": outputs, "finish": finish}
    why = SC.refuse({"STOP": tuple(ids), "MAX_TOKENS": max_tokens, "PROMPT_LEN": prompt_len, "MAX_MODEL_LEN": max_model_len,
                     "FINISH": finish})
    if why or not (type(outputs) is int and outputs >= 1):
        raise ValueError(f"the stop rule's config {s}: {why or 'outputs must be a positive integer'}")
    return s


def of(finished: str, output_len: int, T: int, stops: Iterable[int], *, max_tokens: int, prompt_len: int,
       max_model_len: int) -> dict:
    """The rule's config for the Program whose last output is a request's (T + 1)-th, of `output_len`, that finished on
    `finished` (``eos``, ``length`` or ``abort``) under stop ids `stops` (the registered model's, none under ignore_eos): finish 1
    where it ends there on a stop id, 2 or 3 on its max_tokens or the model's length, 0 where it runs on (an abort, or a window
    that carries it out before its last output). A ValueError where it finished on a stop id with none."""
    stops = tuple(stops)
    if finished == "eos" and not stops:
        raise ValueError("finished on a stop id, and the registered model states none (its config's eos_token_id): register them")
    if finished not in ("eos", "length", "abort"):
        raise ValueError(f"finished on {finished!r}, which no length rule reads")
    code = (0 if output_len > T + 1 or finished == "abort" else 1 if finished == "eos" else 2 if output_len >= max_tokens else 3)
    return spec(stops, max_tokens=max_tokens, prompt_len=prompt_len, max_model_len=max_model_len, outputs=T + 1, finish=code)


def body_refusal(params: Mapping[str, Any]) -> str | None:
    """Why a request body's own stop fields leave its length unchecked, naming the fix; None where they don't. Under ignore_eos
    vLLM's `SamplingParams.update_from_generation_config` keeps only the body's `stop_token_ids` as stop ids, so with those refused
    such a request has none: its rule reads the length caps alone, and a finish on a stop id is refused (`of`)."""
    if params.get("stop") not in (None, []):
        return (f"stop strings {params['stop']!r}, which vLLM's detokenizer matches and no request Program checks, so its length "
                f"would be unchecked advice. The fix: a length rule over the stop strings' token sequences beside "
                f"`stop_rule.StopRule_v1`, which reads stop ids only")
    if params.get("stop_token_ids") not in (None, []):
        return (f"stop_token_ids {params['stop_token_ids']!r}, and the length rule's stop ids are the registered model's alone. The "
                f"fix: `StopRule_v1`'s stops the request's own ids beside the model's (its alone under ignore_eos), keyed in its Program")
    if params.get("ignore_eos") not in (None, False, True):
        return f"ignore_eos {params['ignore_eos']!r}, not a bool"
    return None


def what(config: Mapping[str, Any]) -> str:
    """The check's description, as the provenance's `checks` names it."""
    f = config["finish"]
    end = "run on past the last one" if f == 0 else f"stop at the last one, finish {f} ({X.FINISH[f]})"
    return (f"the length rule (StopCheck_v1 of StopRule_v1): the outputs {end}, under max_tokens {config['max_tokens']} and "
            f"max_model_len {config['max_model_len']}")


def check(B: Any, ids: Coll, config: Mapping[str, Any]) -> tuple[Coll, str]:
    """The rule over ``ids``, the Program's outputs (an Array of i32), as the next Call of the Builder ``B``, and its
    description: (the bit, what it checks), the form `with_checks` and `derive.Ctx.check` take."""
    if not (isinstance(ids.type, Array) and ids.type.n >= 1 and ids.type.elem == Value(32)):
        raise ValueError(f"the stop rule reads a request's outputs, an Array of at least one i32, not {ids.type}")
    fn, s = definition(ids.type.n, config)
    return B.call(fn, ids, name=NAME), what(s)


def request_values(n: int, config: Mapping[str, Any]) -> tuple[dict, int, int]:
    """Over a Program's n outputs, the rule as it reads from the first of them: (the config `spec` makes, max_tokens less the
    outputs before them, the prompt length with those)."""
    if sorted(config) != sorted(KEYS):
        raise ValueError(f"the stop rule's config has the keys {sorted(config)}, not {list(KEYS)}")
    s = spec(config["stops"], **{k: config[k] for k in KEYS[1:]})
    if (before := s["outputs"] - n) < 0:
        raise ValueError(f"the Program returns {n} outputs, and its request has {s['outputs']} through its last")
    return s, s["max_tokens"] - before, s["prompt_len"] + before


def definition(n: int, config: Mapping[str, Any]) -> tuple[Any, dict]:
    """The rule's Call over a Program's n outputs (`check`'s): (`StopCheck_v1` bound, the config `spec` makes)."""
    s, max_tokens, prompt_len = request_values(n, config)
    return bind(SC.StopCheck, T=n, STOP=tuple(s["stops"]), MAX_TOKENS=max_tokens, PROMPT_LEN=prompt_len,
                MAX_MODEL_LEN=s["max_model_len"], FINISH=s["finish"]), s


def _class_body(B: Any, S: Any, ids: Coll, max_tokens: Coll, prompt_len: Coll) -> Coll:
    o, stops = X.word_ops(B), S.STOP or (SC.NOT_AN_ID,)
    words = [o.word(s) for s in stops]
    return B.call(bind(X.StopRule, T=S.T, NSTOP=len(stops)), ids, max_tokens,
                  Coll(Array(len(stops), I32), concat([w.refs for w in words])), prompt_len, o.word(S.MAX_MODEL_LEN))


#: `StopClass_v1{T, STOP, MAX_MODEL_LEN}` (ids: Array<T, i32>, max_tokens: i32, prompt_len: i32) -> (n: i32, finish: i32):
#: `StopRule_v1` over a class's T outputs at its stop ids and model length, statics, and a request's max_tokens and prompt
#: length, values; with no stop ids it reads `stop_check.NOT_AN_ID`, as `StopCheck_v1` does
StopClass = CompositeDefinition(
    "StopClass", 1, ("T", "STOP", "MAX_MODEL_LEN"),
    lambda S: ((("ids", Array(S.T, I32)), ("max_tokens", I32), ("prompt_len", I32)), Tuple(I32, I32)), _class_body,
    register=False, refuse=SC.refuse, doc="StopRule_v1 over a class's outputs at its stop ids and model length",
    conformance="model-untested: StopRule_v1's, which holds no capture of vLLM's finish yet")


def class_definition(n: int, stops: Iterable[int], max_model_len: int) -> SpecializedDefinition:
    """The rule's Call in a class's Definition over its n outputs, bound from the class's statics alone (`CLASS_KEYS`):
    `StopClass_v1`, whose inputs are the outputs and a request's `request_values`, and whose (n, finish) a request Program's
    `definition` checks against its config's."""
    return bind(StopClass, T=n, STOP=tuple(int(x) for x in stops), MAX_MODEL_LEN=max_model_len)


def in_derive(ctx: Any, outs: Sequence[Coll], at: Any) -> None:
    """Under `derive`'s config ``stop``, `check` over the traced source's returned values, which must be one (the request's
    outputs), as the Program's last Call and check (`Ctx.check`), which no fx node lowers; a config it can't read refuses at
    fx node ``at``."""
    from verity_vllm.program.frontend.outcomes import refuse
    if (config := ctx.config.get("stop")) is None:
        return
    try:
        if len(outs) != 1:
            raise ValueError(f"it reads a request's outputs, one returned value, and the source returns {len(outs)}")
        ctx.check(*check(ctx.B, outs[0], config))
    except (TypeError, ValueError) as e:
        refuse("shape", at, None, f"the length rule (config `stop`): {e}")
    ctx.node_hints[len(ctx.B.nodes) - 1] = NAME


__all__ = ["NAME", "KEYS", "CLASS_KEYS", "StopClass", "stop_ids", "spec", "of", "body_refusal", "check", "request_values",
           "definition", "class_definition", "in_derive"]
