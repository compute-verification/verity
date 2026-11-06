"""Sampled proofs' answer on a served window, from the one-stage records audit
(`benchmarks/prove_path/records_audit.py`) that ran on it.

`answer` turns what the audit wrote into sampled proofs' core answer (``verity/sampled-proofs/one-stage/v1``): the
commitment is the registration's digest, the verdict the audit record's, and on acceptance the public outputs and profile
are the ones `SampledProofs.OneStage.check` makes public (the drawn and proved units, the bound on wrong units at δ, each
template's under a stratified or work law; `audit.profile` as `Core.Integrity`, δ an exact ratio). A refusal before the
sample (the records, a check on the registration, the sample) rejects with its codes, and so does one after it that
left no audit record (a drawn member whose ``--registered`` file the audit couldn't complete, by name); an audit record
that doesn't accept, with its failed checks and the first of Lean's refusals by member (a registered read that doesn't
open its root names its port). An audit that ended with no summary is an error (RuntimeError), not a verdict, and so is
one that opened the advice (`opened`), whatever its verdict.

`public_copy` writes what `answer` reads of an audit, what the onsite record credits its members by and the entries its
registered reads opened the window's own values at, and nothing else (`SUMMARY`, `RECORD`): the copy a verifier on
another machine reads, from which `answer` makes the same answer.

    python benchmarks/one_stage/served_integrity.py public-copy AUDIT DST
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R

PROTOCOL = R.PROTOCOL
#: what `answer` reads of the audit's ``summary.json``, by key (its value whole, or of those keys alone): whether it
#: opened the advice, its refusal, its checks before and after the draw (each one's outcome, ``checks`` where one
#: after the draw refused), and the audit's verdict, outcome and Lean's refusals; and what the onsite record holds the
#: scan's state to: the entries its registered reads opened the window's own values at (``registered.own``)
SUMMARY = {"compile": None, "refused": None, "checks_before_draw": None, "fixed_before_draw": ("ok", "codes"),
           "checks": None, "audit": ("accepted", "outcome", "why"), "registered": ("own",)}
#: what `answer` reads of the audit record ``window/audit.json``: its verdict, each check's outcome and the proved units
#: its statement covers, the draw, the registration, the bound on wrong units and the units proved; and what the onsite
#: record credits members by: each check after the draw's outcome, with the drawn members whose ``--registered`` files
#: the audit completed (``registered_complete``), and whether each drawn member's Lean verdict of record reported
#: exactly its file's reads (``registered_reads``)
RECORD = {"accepted": None, "checks": None, "draw": None, "registration": None, "wrong_units": None,
          "proved_units": None, "checks_after_draw": None, "registered_reads": None}


def _only(doc: Mapping[str, Any], keys: Mapping[str, Any]) -> dict[str, Any]:
    return {k: ({x: doc[k][x] for x in sub if x in doc[k]} if sub and isinstance(doc[k], Mapping) else doc[k])
            for k, sub in keys.items() if k in doc}


def _object(path: Path) -> Mapping[str, Any]:
    doc = json.loads(path.read_text())
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path}: not a JSON object")
    return doc


def _checks(checks: Any) -> Any:
    """Each check's outcome (``ok``, ``codes``), and the proved check's units (`unproved`), not its other detail."""
    if not isinstance(checks, Mapping):
        return checks
    out = {c: _only(v, {"ok": None, "codes": None}) if isinstance(v, Mapping) else v for c, v in checks.items()}
    proved = checks.get("proved")
    units = (proved.get("detail") or {}).get("units") if isinstance(proved, Mapping) else None
    if isinstance(units, Mapping):
        out["proved"]["detail"] = {"units": _only(units, {"drawn": None, "statement": None})}
    return out


def _after(checks: Any) -> Any:
    """`_checks` of the checks after the draw, with the drawn members ``registered_complete`` completed, by name."""
    out = _checks(checks)
    done = checks.get("registered_complete") if isinstance(checks, Mapping) else None
    if isinstance(done, Mapping) and isinstance(done.get("members"), Mapping):
        out["registered_complete"]["members"] = sorted(done["members"])
    return out


def _reads(reads: Any) -> Any:
    """``registered_reads``' outcome and each drawn member's, not the ports, values and roots it compared."""
    if not isinstance(reads, Mapping):
        return reads
    chunks = reads.get("chunks") or {}
    return {"ok": reads.get("ok"),
            "chunks": {k: {"ok": v.get("ok")} for k, v in chunks.items() if isinstance(v, Mapping)}}


def public_copy(out: Path, dst: Path) -> None:
    """What `answer` reads of the audit under ``out`` (`SUMMARY`, `RECORD`) written under ``dst``: its ``summary.json``
    and, where the sample ran, ``window/audit.json``. A missing summary or record is left missing, as `answer` reads it,
    and so is a ``compile`` that is None (`opened` reads the two alike); one that isn't a JSON object raises."""
    dst.mkdir(parents=True, exist_ok=True)
    if not (out / "summary.json").is_file():
        return
    summary = _only(_object(out / "summary.json"), SUMMARY)
    if summary.get("compile") is None:
        summary.pop("compile", None)
    if "checks" in summary:
        summary["checks"] = _after(summary["checks"])
    (dst / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    if not (out / "window" / "audit.json").is_file():
        return
    rec = _only(_object(out / "window" / "audit.json"), RECORD)
    for k, keep in (("checks", _checks), ("checks_after_draw", _after), ("registered_reads", _reads)):
        if k in rec:
            rec[k] = keep(rec[k])
    (dst / "window").mkdir(exist_ok=True)
    (dst / "window" / "audit.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")


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


def exact_delta(delta: float | str | Fraction) -> Fraction:
    """δ as the ratio it was given as, never a nearby one (`limit_denominator` takes any δ under about 5·10^-13 to 0,
    which one-stage refuses): a float's shortest decimal or its exact binary value, whichever has the smaller
    denominator, so 0.01 is 1/100 and 2.0 ** -64 is 1/2^64; a string or a ratio as it is."""
    if isinstance(delta, float):
        return min(Fraction(repr(delta)), Fraction(delta), key=lambda f: f.denominator)
    return Fraction(delta)


def opened(summary: Mapping[str, Any]) -> bool:
    """Whether the records audit whose ``summary.json`` is ``summary`` opened the advice (``records_audit.py audit
    --opening``, its ``compile`` check): a development check, which no answer of record reads, since no audit of record
    reads an advice: the record path's window is V3c's class window, whose circuit is the class's."""
    return summary.get("compile") is not None


def answer(out: Path, registration: Mapping[str, Any] | None) -> dict[str, Any]:
    """Sampled proofs' answer from the audit `records_audit.py` wrote under ``out`` (its ``summary.json``, and the audit
    record ``window/audit.json`` when the sample ran). On acceptance its public ``bound`` is the most of the profile's
    units that are wrong, except with probability δ."""
    commitment = R.digest(registration) if registration is not None else None
    try:
        summary = json.loads((out / "summary.json").read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"the records audit left no summary under {out}: {e}") from None
    if opened(summary):
        raise RuntimeError(f"the records audit under {out} opened the advice (records_audit.py audit --opening), a "
                           "development check: no answer of record reads it")
    if summary.get("refused"):
        return _refusal("refused: " + json.dumps(summary["refused"], sort_keys=True, default=str), commitment)
    before = summary.get("checks_before_draw") or {}
    if before and not all(before.values()):
        return _refusal("before the draw: " + ", ".join(_codes(before)), commitment)
    rec_file = out / "window" / "audit.json"
    if not rec_file.exists():
        checks = summary.get("checks") or {}
        if refused := [c for c in _codes(checks) if c not in before]:
            said = [c + (f" ({', '.join(map(str, codes))})" if (codes := (checks[c] or {}).get("codes")) else "")
                    for c in refused]
            return _refusal("after the draw: " + "; ".join(said), commitment)
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
    delta = exact_delta(wrong["delta"])
    profile = A.profile(reg, draw, wrong["delta"]).to_json()
    profile["delta"] = str(delta)
    per_template = wrong.get("per_template")
    public = {"bound": wrong["bound"], "drawn": sorted(draw["units"]), "per_template": per_template,
              "proved": sorted(rec["proved_units"])}
    return {"commitment": R.digest(reg), "profile": profile, "protocol": PROTOCOL, "public": public, "reason": None,
            "verdict": "accept"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="served_integrity.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("public-copy", help="what `answer` reads of the audit under AUDIT, written under DST")
    p.add_argument("audit", type=Path)
    p.add_argument("dst", type=Path)
    a = ap.parse_args(argv)
    public_copy(a.audit, a.dst)
    return 0


__all__ = ["PROTOCOL", "RECORD", "SUMMARY", "answer", "exact_delta", "opened", "public_copy"]


if __name__ == "__main__":
    sys.exit(main())
