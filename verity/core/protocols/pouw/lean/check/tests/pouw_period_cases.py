"""The period protocol's difftest cases: each of `pouw_credit_cases.period_instance`'s periods (honest, or with one mutation
aimed at a refusal of `credit.period_credit`) as a `serve` case of ``verity/pouw/period/v1`` (`serve`: the verifier's
``{key, main, name}`` its parameters, the auditor's ledger its commitment, ``{pins, windows}`` its evidence), its windows
in the one form the period's guarantee covers (`COVERED`: J10-R's per-call unit over ``PouwH1Tile_v5``, at one K); one
mutation each of the protocol's own framing (`FRAMING`): a parameter dropped or added, the ledger not a list, the
evidence's keys other or the evidence a bare list; and one period in each form the guarantee doesn't cover (`FORMS`): a
Pearl-C pass among its windows, a circuit that is not an H1 window (a per-tile protocol's H1 tile), today's per-unit
noise, J10-R's v6 and units at two K."""
from __future__ import annotations

import random

import pouw_credit_cases as credit_cases
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import period_check as PC

FRAMING = ["params-missing", "params-extra", "ledger-object", "evidence-missing", "evidence-extra", "evidence-list"]
FORMS = ["form-pass", "form-not-h1", "form-per-unit", "form-v6", "form-strata"]
NAMES = credit_cases.PERIOD_NAMES + FRAMING + FORMS
COVERED = {"j10r": True, "nks": 1, "tile": 5}
#: a per-tile protocol's H1 tile (`core_check`'s ``PouwH1Tile_v1`` at K = 64, 4 × 8) as a window's program function
CORE_TILE = "PouwH1Tile_v1{EXACT=GemmAccumulatorS8_v2{K=64,DOT=BlackwellS8ImmaDot32_v2},R=4,C=8}"
FORM_OF = {"form-pass": COVERED, "form-not-h1": {**COVERED, "program_fn": CORE_TILE},
           "form-per-unit": {"nks": 1}, "form-v6": {**COVERED, "tile": 6}, "form-strata": {**COVERED, "nks": 2}}


def _hex(x: object) -> str:
    return CC._canon(x).encode().hex()


def serve(d: dict) -> dict:
    """`period_instance`'s period `d` as the protocol's `serve` case."""
    return {"commitment": _hex(d["ledger"]), "evidence": _hex({"pins": d["pins"], "windows": d["windows"]}),
            "params": {"key": d["key"], "main": d["main"], "name": d["name"]}}


def instance(rng: random.Random, name: str = "honest") -> dict:
    """One case: a period of `period_instance` in the covered form (`COVERED`, or `FORM_OF`'s), then mutation `name`."""
    d = credit_cases.period_instance(rng, "honest" if name in FRAMING or name in FORMS else name,
                                     **FORM_OF.get(name, COVERED))
    if name == "form-pass":
        i = rng.randrange(len(d["windows"]))
        d["windows"][i] = {"context": rng.randbytes(8).hex(), "descriptor": d["windows"][i]["descriptor"],
                           "manifest": rng.randbytes(8).hex(), "session": {}}
    case = serve(d)
    match name:
        case "params-missing":
            del case["params"][rng.choice(PC.PARAMS_KEYS)]
        case "params-extra":
            case["params"]["ledger"] = d["ledger"]
        case "ledger-object":
            case["commitment"] = _hex({"lines": d["ledger"]})
        case "evidence-missing":
            case["evidence"] = _hex({"windows": d["windows"]} if rng.random() < 0.5 else {"pins": d["pins"]})
        case "evidence-extra":
            case["evidence"] = _hex({"ledger": d["ledger"], "pins": d["pins"], "windows": d["windows"]})
        case "evidence-list":
            case["evidence"] = _hex(d["windows"])
    return case
