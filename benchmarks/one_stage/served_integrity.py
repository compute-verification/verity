"""Sampled proofs' answer on a served window, from the one-stage records audit
(`benchmarks/prove_path/records_audit.py`) that ran on it.

`answer` turns what the audit wrote into sampled proofs' core answer (``verity/sampled-proofs/one-stage/v1``): the
commitment is the registration's digest, the verdict the audit record's, and on acceptance the public outputs and profile
are the ones `SampledProofs.OneStage.check` makes public (the drawn and proved units, the bound on wrong units at δ, each
template's under a stratified or work law; `audit.profile` as `Core.Integrity`, δ an exact ratio). A refusal before the
sample (the records, a check on the registration, the sample) rejects with its codes; an audit record that doesn't accept,
with its failed checks and the first of Lean's refusals by member (a registered read that doesn't open its root names
its port). An audit that ended with no summary is an error (RuntimeError), not a verdict.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R

PROTOCOL = R.PROTOCOL


def _refusal(reason: str, commitment: str | None) -> dict[str, Any]:
    return {"commitment": commitment, "profile": None, "protocol": PROTOCOL, "public": None, "reason": reason,
            "verdict": "reject"}


def _codes(checks: Mapping[str, Any]) -> list[str]:
    return sorted(c for c, v in checks.items() if not (v.get("ok") if isinstance(v, Mapping) else v))


#: a refused member's verdicts, said in this order: its live session's verifier of record first, whose rejection leaves
#: no proof files, then the Lean replay that finds none, then the sample check
SAID = ("m0-verifier/", "lean-verify/", "lean-draw/")


def _said_first(kv: tuple[str, Any]) -> tuple[int, str]:
    k = kv[0]
    return next((i for i, p in enumerate(SAID) if k.startswith(p)), len(SAID)), k


def unproved(rec: Mapping[str, Any]) -> str:
    """The audit record's sampled proof blocks its statement doesn't prove (its ``proved`` check), by the sample's template
    strata: `` (drawn units not proved: T u0-u1, ...)``, or nothing."""
    units = (((rec.get("checks") or {}).get("proved") or {}).get("detail") or {}).get("units") or {}
    missing = sorted(set(units.get("drawn") or ()) - set(units.get("statement") or ()))
    if not missing:
        return ""
    named = []
    for s in (rec.get("draw") or {}).get("strata") or []:
        got = [u for u in missing if any(lo <= u < hi for lo, hi in s["units"])]
        if got:
            named.append(f"{s['template']} {got[0]}-{got[-1]}" if got == list(range(got[0], got[-1] + 1))
                         else f"{s['template']} {','.join(map(str, got))}")
    return f" (drawn units not proved: {'; '.join(named) or ','.join(map(str, missing))})"


def answer(out: Path, registration: Mapping[str, Any] | None) -> dict[str, Any]:
    """Sampled proofs' answer from the audit `records_audit.py` wrote under ``out`` (its ``summary.json``, and the audit
    record ``window/audit.json`` when the sample ran). On acceptance its public ``bound`` is the most of the profile's
    units that are wrong, except with probability δ."""
    commitment = R.digest(registration) if registration is not None else None
    try:
        summary = json.loads((out / "summary.json").read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"the records audit left no summary under {out}: {e}") from None
    if summary.get("refused"):
        return _refusal("refused: " + json.dumps(summary["refused"], sort_keys=True, default=str), commitment)
    before = summary.get("checks_before_draw") or {}
    if before and not all(before.values()):
        return _refusal("before the draw: " + ", ".join(_codes(before)), commitment)
    rec_file = out / "window" / "audit.json"
    if not rec_file.exists():
        audit = summary.get("audit") or {}
        if audit.get("accepted") is False:
            return _refusal("no audit record: " + json.dumps(audit.get("outcome"), sort_keys=True, default=str),
                            commitment)
        raise RuntimeError(f"the records audit wrote no audit record and no refusal under {out}")
    rec = json.loads(rec_file.read_text())
    if rec.get("accepted") is not True:
        lean = {k: v for k, v in sorted(((summary.get("audit") or {}).get("why") or {}).items(), key=_said_first) if v}
        why = _codes(rec.get("checks") or {}) or sorted(lean) or ["not accepted"]
        said = [f"{k}: {v}" for k, v in lean.items()]
        more = f"; and {len(said) - 3} more" if len(said) > 3 else ""
        return _refusal("audit record: " + ", ".join(why) + unproved(rec) + "".join(f"; {s}" for s in said[:3]) + more,
                        commitment)
    fixed = summary.get("fixed_before_draw")
    if fixed is not None and not fixed.get("ok"):
        return _refusal("not fixed before the draw: " + ", ".join(fixed.get("codes") or []), commitment)
    reg, draw, wrong = rec["registration"], rec["draw"], rec["wrong_units"]
    delta = Fraction(wrong["delta"]).limit_denominator(10**12)
    profile = A.profile(reg, draw, wrong["delta"]).to_json()
    profile["delta"] = str(delta)
    per_template = wrong.get("per_template")
    public = {"bound": wrong["bound"], "drawn": sorted(draw["units"]), "per_template": per_template,
              "proved": sorted(rec["proved_units"])}
    return {"commitment": R.digest(reg), "profile": profile, "protocol": PROTOCOL, "public": public, "reason": None,
            "verdict": "accept"}


__all__ = ["PROTOCOL", "answer"]
