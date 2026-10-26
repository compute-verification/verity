"""The integrity leg: sampled proofs' one-stage audit of the period's window, from its records alone
(`verity_all_protocols.table`'s interface: ``leg(period_dir, node, out) -> answer``).

The audit is `benchmarks/prove_path/records_audit.py`'s, as `run.sh` with ``PP_TORCH_LOOP=1`` runs it: the registration
held to the circuit the verifier registered before the loop ran (``--circuit``) and to the run root the loop published,
the draw on live coins read after the receipt, C-Flock ``--zk`` behind the Lean firewall on the drawn members, and the
audit record of the verifier of record (`one_stage.verifier.audit_record`, the Lean ``sampled-proofs-verifier``). A
period whose loop reads registered weights (``period.json``'s ``registered``, the runner's ``--registered``) is audited
against their pin (``records_audit.py --registered``).

`answer` turns what the audit wrote into sampled proofs' core answer (``verity/sampled-proofs/one-stage/v1``): the
commitment is the registration's digest, the verdict the audit record's, and on acceptance the public outputs and profile
are the ones `SampledProofs.OneStage.check` makes public (the drawn and proved units, the bound on wrong units at δ, each
template's under a stratified or work law; `audit.profile` as `Core.Integrity`, δ an exact ratio). A refusal before the
draw (the records, a check on the registration, the draw) rejects with its codes; an audit record that doesn't accept,
with its failed checks and the first of Lean's refusals by member (a registered read that doesn't open its root names
its port). An audit that ended with no summary is the leg's error, not a verdict.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R

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


def answer(out: Path, registration: Mapping[str, Any] | None) -> dict[str, Any]:
    """Sampled proofs' answer from the audit `records_audit.py` wrote under ``out`` (its ``summary.json``, and the audit
    record ``window/audit.json`` when the draw ran). On acceptance its public ``bound`` is the most of the profile's
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
        lean = {k: v for k, v in sorted(((summary.get("audit") or {}).get("why") or {}).items()) if v}
        why = _codes(rec.get("checks") or {}) or sorted(lean) or ["not accepted"]
        said = [f"{k}: {v}" for k, v in lean.items()]
        more = f"; and {len(said) - 3} more" if len(said) > 3 else ""
        return _refusal("audit record: " + ", ".join(why) + "".join(f"; {s}" for s in said[:3]) + more, commitment)
    fixed = summary.get("fixed_before_draw")
    if fixed is not None and not fixed.get("ok"):
        return _refusal("not fixed before the draw: " + ", ".join(fixed.get("codes") or []), commitment)
    reg, draw, wrong = rec["registration"], rec["draw"], rec["wrong_units"]
    delta = Fraction(wrong["delta"])
    profile = A.profile(reg, draw, wrong["delta"]).to_json()
    profile["delta"] = str(delta)
    per_template = wrong.get("per_template")
    public = {"bound": wrong["bound"], "drawn": sorted(draw["units"]), "per_template": per_template,
              "proved": sorted(rec["proved_units"])}
    return {"commitment": R.digest(reg), "profile": profile, "protocol": PROTOCOL, "public": public, "reason": None,
            "verdict": "accept"}


def keep(src: Path, dst: Path) -> None:
    """The audit's records from ``src`` into ``dst``: never a prover's file (its rows and salts) nor a circuit, whose pin
    the records hold (`benchmarks/prove_path/run.sh`'s ``keep``)."""
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        if not f.is_file() or "archive" in rel.parts[1:] or f.name == "circuit.txt" or f.name.endswith("-private.bin") \
                or f.name.startswith("inst-") or re.match(r"m[0-9]+-inst-", f.name):
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst / rel)


def leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """The period's integrity audit (module docstring) into ``out``; its answer."""
    p = json.loads((period_dir / "period.json").read_text())
    t, work = p["tools"], period_dir.parent / f"{period_dir.name}-sampled-proofs"
    argv = [sys.executable, str(RECORDS_AUDIT), "audit", "--window", str(period_dir), "--law", p["law"],
            "--circuit", p["circuit"], "--run", p["run_root"], "--delta", str(DELTA),
            "--m0-pythonpath", t["m0-pythonpath"], "--bin", t["flock-circuit"], "--lean", t["flock-verify"],
            "--firewall", t["flock-firewall"], "--out", str(work),
            *(["--registered", p["registered"]] if p.get("registered") else [])]
    env = {**os.environ, **p.get("prover_env", {})}
    with open(out / "audit.log", "w") as log, open(out / "audit.err", "w") as err:
        subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=err, check=False)
    keep(work, out / "audit")
    try:
        registration = json.loads((period_dir / "registration.json").read_text())
    except (OSError, ValueError):
        registration = None
    return answer(work, registration)


__all__ = ["DELTA", "PROTOCOL", "answer", "keep", "leg"]
