"""The Python SPOT's two-stage form (`VStar.Recursive.SampledTwo`, `verity/sampled-proofs/two-stage/v1`) is held to: its
parameters' parse in the Lean's order, through the interior's template circuits.

The form's own refusals (`split`'s form, the keys, `OUTPUTS_REFUSED`, a rule that is not two-stage, the challenger's
name, the declared items, `ONE_DEFINITION`, `ONE_KEY`) are mirrored word for word. The parts it hands to the one-stage
audit's decoders are the one-stage difftest's mirrors of them (`_rule` for `Select.Rule.ofJson`, `decode_params` for
`OneStage.paramsParsed`, `template_circuits` for `Run.templateCircuits`), which that difftest holds to the Lean by
verdict; `reason_mirrored` says which reasons are the form's own. Past the template circuits (the replay program, the
members' audit, the refusal of an interior audit on the boundary registration's window, `ONE_WINDOW`, the evidence, its
refusal of `2^64` statements or more, `STMTS_COUNT`, and the check, its refusals of a session's statement of `2^64`
bytes or more, `STMT_BYTES`, of a stage's bytes past one coin's, `STAGE_PAST`, of first-stage bytes past what the run read, `SOURCE_READ`, of a first session showing second-stage coins
on two windows, `INTERIOR_SHOWN`, or a coin tree's line there, `FIRST_TREE`, of a registration's record its canonical
text does not read back, `RECORD_READ`, of a boundary record so read without the commitment's digest, `RECORD_DIGEST`,
of an interior record, as given or so read, without the interior commitment's digest, `INTERIOR_DIGEST`, and of a second
session's log that does not open a coin tree's rounds stream by stream, `REP_STREAM_ORDER`, included; the form's
`#guard`s pin those) is outside the mirror: while `Spot.SecondStage.FORM` is false the form accepts nothing, which its
Lean proves at every input (`checkBlobs_refuses`)."""

from __future__ import annotations

from typing import Any

from reference import ONE_STAGE
from verity_difftest import Reject
from verity_difftest import protocol as P

PROTOCOL = "verity/sampled-proofs/two-stage/v1"
TWO_STAGE = "two-stage"
ONE_DEFINITION = ("the boundary program's proof blocks under its query are not one stratum: the two-stage form's replay "
                  "blocks are one Definition's, so a share of them by count is a share by weight")
ONE_KEY = "the boundary and the interior name one challenger key: both stages' coins are read under the run's one key"
OUTPUTS_REFUSED = "the two-stage form's check has no public outputs: the second stage's verdicts report none"
PARAM_KEYS = ["boundary", "challenger", "delta", "interior", "outputs", "public", "rule", "values"]
INTERIOR_KEYS = ["members", "params", "replay", "values"]


class Delegated(Reject):
    """A refusal of a decoder the form hands its input to, whose reason is that decoder's mirror's."""


def _keys(j: Any, keys: list[str], what: str) -> None:
    if not isinstance(j, dict) or sorted(j) != keys:
        raise Reject(f"the keys of {what} are not {', '.join(keys)}")


def _get(j: Any, k: str) -> Any:
    """`getD j k`: null for a missing key or a non-object."""
    return j.get(k) if isinstance(j, dict) else None


def split(j: Any, what: str) -> dict:
    """`SampledTwo.split`: an object naming the form ``two-stage``, without its ``form``."""
    if not isinstance(j, dict):
        raise Reject(f"the {what} is not an object")
    if not isinstance(j.get("form"), str):
        raise Reject(f"the {what} names no form")
    if j["form"] != TWO_STAGE:
        raise Reject(f"the {what} name the form {j['form']}, not \"{TWO_STAGE}\"")
    return {k: v for k, v in j.items() if k != "form"}


def _delegated(decode, *args) -> Any:
    try:
        return decode(*args)
    except Reject as e:
        raise Delegated(str(e)) from None


def params_at(j: Any) -> dict:
    """`SampledTwo.paramsAt`: `OneStage.paramsParsed` (`decode_params`)."""
    return _delegated(ONE_STAGE.decode_params, j)


def params_of_json(j: Any) -> dict:
    """`SampledTwo.paramsOfJson` through `Run.templateCircuits`; the rest is outside the mirror."""
    j = split(j, "parameters")
    _keys(j, PARAM_KEYS, "the parameters")
    if not isinstance(j["outputs"], list):
        raise Reject("outputs is not a list")
    if j["outputs"]:
        raise Reject(OUTPUTS_REFUSED)
    rule = _delegated(ONE_STAGE._rule, j["rule"])
    if not isinstance(rule, ONE_STAGE.service.TwoStage):
        raise Reject("the rule is not two-stage")
    if not isinstance(j["challenger"], str):
        raise Reject("challenger is not a string")
    if not isinstance(j["public"], list):
        raise Reject("public is not a list")
    if not all(isinstance(d, str) for d in j["public"]):
        raise Reject("a declared item is not a string")
    bp = params_at(j["boundary"])
    program = ONE_STAGE.PROGRAMS[ONE_STAGE.PO.program_sha512(bp["program"])]
    if len(ONE_STAGE.DR.strata_sizes(program, bp["query"])) != 1:
        raise Reject(ONE_DEFINITION)
    it = j["interior"]
    _keys(it, INTERIOR_KEYS, "the interior's parameters")
    ip = it["params"]
    key = _get(_get(j["boundary"], "challenger"), "key")
    if key is None or key != _get(_get(ip, "challenger"), "key"):
        raise Reject(ONE_KEY)
    interior = params_at({k: v for k, v in ip.items() if k != "proofs"} if isinstance(ip, dict) else {})
    try:
        ONE_STAGE.service.template_circuits(_get(ip, "proofs"), interior["query"])
    except ValueError as e:
        raise Delegated(str(e)) from None
    raise ValueError("outside the mirror: the interior's replay program and members")


def _outside(*_: Any) -> Any:
    raise ValueError("outside the mirror: the evidence and the check")


def answer(case: Any) -> dict:
    """`SampledTwo.protocol` on a `serve` case, on the cases the mirror decides: those its parameters refuse."""
    return P.answer(PROTOCOL, case, params=params_of_json, commitment=ONE_STAGE.decode_commitment,
                    evidence=_outside, digest=lambda c: c[1], check=_outside)


def refusal(params: Any) -> Exception:
    """What `params_of_json` raises on the parameters: the form's own refusal, a delegated decoder's (`Delegated`), or
    the `ValueError` at the mirror's end."""
    try:
        params_of_json(params)
    except (Reject, ValueError) as e:
        return e
    raise AssertionError("params_of_json returned: it raises on every input")


def reason_mirrored(params: Any) -> bool:
    """Whether the parameters' refusal is the form's own, whose reason the twin gives word for word, rather than a
    delegated decoder's, which is held to the Lean by its verdict."""
    return not isinstance(refusal(params), Delegated)
