"""What the scan's period check reads of a run: the jitter bound it is charged at, a run's file, and an execution
phase's answer."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Sequence
from verity.protocols.proofs.spot.verifier.one_stage import audit as AU, registration as RG

#: the jitter bound D_j that K is charged at (`NetTiming.JitterBounded`, one bound over every step): a statement of k
#: steps, one a bucket, pays ⌈log2 D_j^k⌉ = k bits. Each bucket leaves at a time fixed before it is made
#: (`warden.Warden(paced=…)`, `active.Proxy(paced=…)`), yet the host's state after the server's writes for the next
#: bucket still moves a bucket's arrival offset, so J isn't met at D_j = 1
JITTER_BOUND = 2


def _found(run: Path, name: str) -> Path:
    """A network certifier run's file: in the run's directory (`warden_serve.py serve --out`), else its ``witness/``
    (the gate's `warden-rehearse --scan`)."""
    for d in (run, run / "witness"):
        if (d / name).is_file():
            return d / name
    raise SystemExit(f"--warden-run {run}: no {name} there or in its witness/ (warden_serve.py serve writes it)")


def window_answer(reg: dict, record: dict | None, delta: Fraction, reason: str, rests: Sequence[str] = ()) -> dict:
    """One-stage's answer for the group's registration ``reg`` as a period check reads an execution phase's scan answer
    (`NetworkCheck.Period`'s rows form, `pouw.core_check.answer_of`): from an accepted audit record (``record``,
    `audit`'s ``audit.json``), its sample and proved units with one-stage's profile and bound on incorrect proof
    blocks at ``delta`` (`audit.profile`, `audit.wrong_units`: every unit sampled, the bound 0), and the tied values
    its registration's R8 let through when there are any (``rests``, the registration verdict's
    ``rests_on_period_check``), on whose period check the acceptance rests; else a reject naming ``reason``."""
    commitment = RG.digest(reg)
    if record is None:
        return {"commitment": commitment, "profile": None, "protocol": RG.PROTOCOL, "public": None, "reason": reason,
                "verdict": "reject"}
    if record.get("accepted") is not True or RG.digest(record["registration"]) != commitment:
        raise SystemExit("an accepted scan's audit record is for another registration, or not accepted")
    draw = record["draw"]
    profile = AU.profile(reg, draw, delta).to_json()
    profile["delta"] = str(delta)
    wrong = AU.wrong_units(reg, draw, delta)
    return {"commitment": commitment, "profile": profile, "protocol": RG.PROTOCOL,
            "public": {"bound": wrong["bound"], "drawn": sorted(draw["units"]),
                       "per_template": wrong.get("per_template"), "proved": sorted(record["proved_units"]),
                       **({"rests_on_period_check": list(rests)} if rests else {})},
            "reason": None, "verdict": "accept"}
