"""SPOT's levels: how many levels of blocks (outer blocks, then inner blocks inside each sampled outer block) an audit
samples, each level a partition with its sampling design. One level is the audit over one partition and its sampling design; two
levels are the boundary docket's outer blocks under the two-level rule (its grain and count) and the interior's inner
blocks. More are refused by name until they are built.

The verdict's number is the auditor's: read off the shape of the case's parameters, fixed before any coin (`of_params`).
A run's record and a command's `ENV` are the prover's: they say what was served, and a record that differs from the
parameters is refused (`check`). Neither ever chooses what the verifier checks."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

#: the field a case's parameters and a run's record name the levels by, and the environment setting a command takes
KEY = "levels"
ENV = "SPOT_LEVELS"
BUILT = (1, 2)
#: `VStar.Recursive.Sampled.TWO_LEVELS_REFUSED`: two levels are built and refused by name at registration
NOT_PROVED = "levels-not-proved"
TWO_LEVELS_REFUSED = (f"{NOT_PROVED}: the parameters name 2 levels: the two-level audit is built and refused, since no "
                      "soundness theorem covers its accept; an audit is accepted at one level")
#: the parameters only two levels' have (`VStar.Recursive.SampledTwo`): the boundary docket, the interior and the rule
TWO_LEVEL_KEYS = ("boundary", "interior", "rule")
#: the boundary commitment's record format (`verity_vllm.records.boundary_commit.SCHEMA`), whose spans are outer blocks
BOUNDARY_RECORD = "verity-vllm/boundary-rows/v1"
#: a record with neither `KEY` nor a two-level format: written before the setting, of one level
ABSENT = 1


class Refused(ValueError):
    """A refusal of a level count, by its reason's name (``reason``)."""

    def __init__(self, reason: str, why: str):
        self.reason = reason
        super().__init__(f"{reason}: {why}")


def parse(value: Any, where: str) -> int:
    """``value`` as a level count, a JSON number (a string is refused, as the Lean's `levelsOf` refuses it); refused
    unless a whole number in `BUILT`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refused("levels-not-integer", f"{where} is {value!r}, not a whole number of levels")
    if value < 1:
        raise Refused("levels-not-positive", f"{where} is {value}: an audit samples at least one level")
    if value not in BUILT:
        raise Refused("levels-not-built", f"{where} is {value}: only {' or '.join(map(str, BUILT))} levels are built")
    return value


def shape(params: Mapping[str, Any]) -> int:
    """The level count the parameters' shape is: two levels' keys (`TWO_LEVEL_KEYS`) all or none."""
    held = [k for k in TWO_LEVEL_KEYS if k in params]
    if held and len(held) != len(TWO_LEVEL_KEYS):
        raise Refused("levels-shape", f"the parameters hold {held} of two levels' {list(TWO_LEVEL_KEYS)}: neither shape")
    return 2 if held else 1


def of_params(params: Mapping[str, Any]) -> int:
    """The auditor's level count (`VStar.Recursive.Sampled.split`): the parameters' `KEY`, one where absent, which must
    be their shape's (`shape`)."""
    if not isinstance(params, Mapping):
        raise Refused("levels-shape", f"the parameters are a {type(params).__name__}, not an object")
    n = shape(params)
    declared = parse(params[KEY], "the parameters' levels") if KEY in params else ABSENT
    if declared != n:
        raise Refused("levels-params-mismatch", f"the parameters name {declared} level(s), and their shape is {n}'s")
    return n


def served(record: Mapping[str, Any]) -> int:
    """The prover's level count, a run's record's: its format's (`BOUNDARY_RECORD`: two), else its `KEY`, else `ABSENT`."""
    if record.get("format") == BOUNDARY_RECORD:
        # Two levels by the record's format, not an absent `KEY` read as 2: absent means one level everywhere.
        if KEY in record and (n := parse(record[KEY], "the run's record's levels")) != 2:
            raise Refused("levels-format-mismatch", f"the run's record's levels is {n}, and its format "
                                                    f"{BOUNDARY_RECORD} is two levels'")
        return 2
    if KEY in record:
        return parse(record[KEY], "the run's record's levels")
    return ABSENT


def check(params: Mapping[str, Any], record: Mapping[str, Any]) -> int:
    """The auditor's level count (`of_params`), refused where the run's record (`served`) served another."""
    n, s = of_params(params), served(record)
    if s != n:
        raise Refused("levels-record-mismatch", f"the run's record served {s} level(s), and the parameters check {n}")
    return n


def command(raw: str | None, runs: tuple[int, ...], default: int) -> int:
    """A command's level count, from its `ENV` (``raw``, unset: ``default``); refused where the command can't run it,
    rather than running another."""
    if raw is None or raw == "":
        n = default
    else:
        n = parse(int(raw) if raw.strip().isdecimal() else raw, ENV)
    if n not in runs:
        raise Refused("levels-no-path", f"{ENV}={n}: this command runs {' or '.join(map(str, runs))} level(s), and no "
                                        f"other path here runs {n}")
    return n


def with_levels(case: Mapping[str, Any], levels: int, canonical: Callable[[dict], bytes]) -> dict[str, Any]:
    """``case`` as SPOT's dispatcher reads it (`VStar.Recursive.Sampled`): its params and its evidence (hex of
    ``canonical`` JSON) each naming ``levels``; refused where either names it already, or the params' shape is another
    level count."""
    import json

    evidence = json.loads(bytes.fromhex(case["evidence"]))
    params = case.get("params")
    if not isinstance(params, Mapping) or not isinstance(evidence, dict):
        raise Refused("levels-shape", "the case's params and evidence must be objects")
    if KEY in params or KEY in evidence:
        raise Refused("levels-named", "the case's params and evidence must name no levels")
    n = parse(levels, "the levels")
    out = {**case, "evidence": canonical(evidence | {KEY: n}).hex(), "params": {**params, KEY: n}}
    if (m := of_params(out["params"])) != n:
        raise Refused("levels-params-mismatch", f"the case's params are {m} levels' shape, not {n}'s")
    return out
