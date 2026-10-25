"""The integrity leg: sampled proofs' one-stage audit of the period's window, from its records alone.

The audit is `benchmarks/prove_path/records_audit.py`'s, as `run.sh` with ``PP_TORCH_LOOP=1`` runs it: the registration
held to the circuit the verifier registered before the loop ran (``--circuit``) and to the run root the loop published,
the draw on live coins read after the receipt, C-Flock ``--zk`` behind the Lean firewall on the drawn members, and the
audit record of the verifier of record (`one_stage.verifier.audit_record`, the Lean ``sampled-proofs-verifier``).

`answer` turns what the audit wrote into sampled proofs' core answer (``verity/sampled-proofs/one-stage/v1``): the
commitment is the registration's digest, the verdict the audit record's, and on acceptance the public outputs and profile
are the ones `SampledProofs.OneStage.check` makes public (the drawn and proved units, the bound on wrong units at δ, each
template's under a stratified or work law; `audit.profile` as `Core.Integrity`, δ an exact ratio). A refusal before the
draw (the records, a check on the registration, the draw) rejects with its codes. An audit that ended with no summary is
the leg's error, not a verdict.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R

from .legs import Leg, Period

PROTOCOL = R.PROTOCOL
ROOT = Path(__file__).resolve().parents[3]
RECORDS_AUDIT = ROOT / "benchmarks" / "prove_path" / "records_audit.py"
#: the δ every period's audit is decided at (`records_audit.py --delta`'s default)
DELTA = 0.5


def _refusal(reason: str, commitment: str | None) -> dict[str, Any]:
    return {"commitment": commitment, "profile": None, "protocol": PROTOCOL, "public": None, "reason": reason,
            "verdict": "reject"}


def _codes(checks: Mapping[str, Any]) -> list[str]:
    return sorted(c for c, v in checks.items() if not (v.get("ok") if isinstance(v, Mapping) else v))


def answer(out: Path, registration: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Sampled proofs' answer from the audit `records_audit.py` wrote under ``out`` (its ``summary.json``, and the audit
    record ``window/audit.json`` when the draw ran), and the quantity its acceptance certifies: at most
    ``wrong_units_bound`` of the window's ``units`` are wrong, except with probability δ (``epsilon`` the share)."""
    commitment = R.digest(registration) if registration is not None else None
    try:
        summary = json.loads((out / "summary.json").read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"the records audit left no summary under {out}: {e}") from None
    if summary.get("refused"):
        return _refusal("refused: " + json.dumps(summary["refused"], sort_keys=True, default=str), commitment), None
    before = summary.get("checks_before_draw") or {}
    if before and not all(before.values()):
        return _refusal("before the draw: " + ", ".join(_codes(before)), commitment), None
    rec_file = out / "window" / "audit.json"
    if not rec_file.exists():
        audit = summary.get("audit") or {}
        if audit.get("accepted") is False:
            return _refusal("no audit record: " + json.dumps(audit.get("outcome"), sort_keys=True, default=str),
                            commitment), None
        raise RuntimeError(f"the records audit wrote no audit record and no refusal under {out}")
    rec = json.loads(rec_file.read_text())
    if rec.get("accepted") is not True:
        why = _codes(rec.get("checks") or {}) or sorted((summary.get("audit") or {}).get("why") or {}) or ["not accepted"]
        return _refusal("audit record: " + ", ".join(why), commitment), None
    fixed = summary.get("fixed_before_draw")
    if fixed is not None and not fixed.get("ok"):
        return _refusal("not fixed before the draw: " + ", ".join(fixed.get("codes") or []), commitment), None
    reg, draw, wrong = rec["registration"], rec["draw"], rec["wrong_units"]
    delta = Fraction(wrong["delta"])
    profile = A.profile(reg, draw, wrong["delta"]).to_json()
    profile["delta"] = str(delta)
    per_template = wrong.get("per_template")
    public = {"bound": wrong["bound"], "drawn": sorted(draw["units"]), "per_template": per_template,
              "proved": sorted(rec["proved_units"])}
    quantity = {"wrong_units_bound": wrong["bound"], "units": wrong["of"], "delta": str(delta),
                "epsilon": str(Fraction(wrong["bound"], wrong["of"])) if wrong["of"] else "0",
                **({"per_template": per_template} if per_template else {})}
    return ({"commitment": R.digest(reg), "profile": profile, "protocol": PROTOCOL, "public": public, "reason": None,
             "verdict": "accept"}, quantity)


def leg(period: Period) -> Leg:
    """The period's integrity audit (module docstring) into ``period.out``; its answer."""
    out = period.out / "audit"
    out.parent.mkdir(parents=True, exist_ok=True)
    t = period.tools
    argv = [sys.executable, str(RECORDS_AUDIT), "audit", "--window", str(period.records), "--law", period.law,
            "--circuit", period.circuit, "--run", period.run_root, "--delta", str(DELTA),
            "--m0-pythonpath", t["m0-pythonpath"], "--bin", t["flock-circuit"], "--lean", t["flock-verify"],
            "--firewall", t["flock-firewall"], "--out", str(out)]
    env = {**os.environ, **period.prover_env}
    with open(period.out / "audit.log", "w") as log, open(period.out / "audit.err", "w") as err:
        subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=err, check=False)
    try:
        registration = json.loads((period.records / "registration.json").read_text())
    except (OSError, ValueError):
        registration = None
    a, quantity = answer(out, registration)
    run = os.environ.get("RESEARCH_RUN_ID")
    return Leg(answer=a, quantity=quantity, runs=(run,) if run else ())


__all__ = ["DELTA", "PROTOCOL", "answer", "leg"]
