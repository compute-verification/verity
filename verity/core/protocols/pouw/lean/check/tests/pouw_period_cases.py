"""The period protocol's difftest cases: each of `pouw_credit_cases.period_instance`'s periods (honest, or with one mutation
aimed at a refusal of `credit.period_credit`) as a `serve` case of ``verity/pouw/period/v1`` (`serve`: the verifier's
``{key, main, name}`` its parameters, the auditor's ledger its commitment, ``{pins, windows}`` its evidence), and one
mutation each of the protocol's own framing (`FRAMING`): a parameter dropped or added, the ledger not a list, the
evidence's keys other or the evidence a bare list."""
from __future__ import annotations

import random

import pouw_credit_cases as credit_cases
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import period_check as PC

FRAMING = ["params-missing", "params-extra", "ledger-object", "evidence-missing", "evidence-extra", "evidence-list"]
NAMES = credit_cases.PERIOD_NAMES + FRAMING


def _hex(x: object) -> str:
    return CC._canon(x).encode().hex()


def serve(d: dict) -> dict:
    """`period_instance`'s period `d` as the protocol's `serve` case."""
    return {"commitment": _hex(d["ledger"]), "evidence": _hex({"pins": d["pins"], "windows": d["windows"]}),
            "params": {"key": d["key"], "main": d["main"], "name": d["name"]}}


def instance(rng: random.Random, name: str = "honest") -> dict:
    """One case: a period of `period_instance`, then mutation `name`."""
    d = credit_cases.period_instance(rng, "honest" if name in FRAMING else name)
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
