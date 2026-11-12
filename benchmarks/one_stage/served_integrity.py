"""Sampled proofs' answer on a served window, from the one-stage records audit
(`benchmarks/prove_path/records_audit.py`) that ran on it.

`answer` turns what the audit wrote into sampled proofs' core answer (``verity/sampled-proofs/one-stage/v1``): the
commitment is the registration's digest, the verdict the audit record's, and on acceptance the profile is
`audit.profile` as `Core.Integrity` (δ an exact ratio) and the public outputs are the five `Spot.OneStage.check` makes
public (``drawn`` and ``proved``, the sampled and proved units; ``bound``, the bound on incorrect proof blocks at δ;
``per_template``, each template's under a stratified or work sampling design; ``complete``, the audit record's own:
whether every verdict ran, which `audit.answer_complete` reads) and two the Lean's answer doesn't have,
``coins`` and ``case_v2``. A refusal before the
sample (the records, a check on the registration, the sample) rejects with its codes; an audit record that doesn't accept,
with its failed checks and the first of Lean's refusals by member (a registered read that doesn't open its root names
its port). An audit that ended with no summary is an error (RuntimeError), not a verdict.

``public.coins`` says whose coins the sample is on: ``{key_sha512, session}``, the SHA-512 of the challenger key the
coins' evidence verified under, in lowercase hex, and the id of the one session the auditor's ledger opens on the
window's root under that key. Both are what a check verified, never what a flag or a printed line says: the audit
record's ``coins.verified`` (`a5.coins_outcome`), which the audit writes only where `coins.coins_reason` took the
evidence (the ledger's, the pins' and the segment's signatures under the key, and the ledger's one open on the window's
root), the stream's Lean run having read the same evidence at its select. They are not the parameters of the compiled
check's case: `Spot.OneStageZk` reads the challenger from its parameters, and a case that names it is where the two
values are to come from once an audit writes one. Until then a case that does name its challenger
(``window/case-v2.json``'s ``params.challenger``) is held to the audit record: the key's digest must be the recorded one.

``public.case_v2`` says which kind of accept this is. Every accept here is the Python audit's (`a5`'s): none is the
compiled check's, `Spot.OneStageZk`'s, which nothing here runs. The field says whether the audit wrote the case that
check reads, as the audit record says it (``case_v2``, `a5.case_written`, set by the audit when it wrote its cases and
not read from a case): ``{"written": true}``, or ``{"written": false, "why"}`` with the audit's own reason (sessions
that aren't `--zk`, a sample of several members), where no verifier of record can be run on the window at all.
``written`` true says a case file is there, not that the compiled check accepts it: `a5.write_case` writes its
parameters without a challenger, and `Spot.OneStageZk` refuses parameters that name none. A record or a statement
about the compiled check's accepts counts no answer here as one.

An accepted audit record is undecided, not an accept, where it names no such key and session or doesn't say whether
it wrote the case (an audit record written without them), or where the window's case names a challenger whose key is
another, or isn't read. A reader that holds the deployment's key record compares the digest with it; nothing here
holds a key record, so nothing here tells a development session's key from the auditor's.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from verity.protocols.spot.device.challenger import ChallengerKey
from verity.protocols.spot.verifier.one_stage import audit as A, registration as R

PROTOCOL = R.PROTOCOL


def _refusal(reason: str, commitment: str | None, verdict: str = "reject") -> dict[str, Any]:
    return {"commitment": commitment, "profile": None, "protocol": PROTOCOL, "public": None, "reason": reason,
            "verdict": verdict}


def whose_coins(rec: Mapping[str, Any]) -> dict[str, str] | None:
    """Whose coins the audit record ``rec``'s sample is on, as its coins' check verified it (``coins.verified``,
    `a5.coins_outcome`): ``{key_sha512, session}``, the challenger key's SHA-512 in lowercase hex and the session's id;
    None for a record that names neither in that form."""
    coins = rec.get("coins")
    held = coins.get("verified") if isinstance(coins, Mapping) else None
    if not isinstance(held, Mapping) or sorted(held) != ["key_sha512", "session"]:
        return None
    key, session = held["key_sha512"], held["session"]
    if type(key) is not str or len(key) != 128 or set(key) - set("0123456789abcdef") or type(session) is not str or not session:
        return None
    return {"key_sha512": key, "session": session}


def case_said(rec: Mapping[str, Any]) -> dict[str, Any] | None:
    """Whether the audit record ``rec``'s audit wrote the case for the compiled check, as the record says it
    (``case_v2``, `a5.case_written`): ``{"written": True}``, or ``{"written": False, "why"}`` with the audit's reason;
    None for a record that doesn't say it in that form."""
    said = rec.get("case_v2")
    if not isinstance(said, Mapping):
        return None
    if said.get("written") is True and sorted(said) == ["written"]:
        return {"written": True}
    why = said.get("why")
    if said.get("written") is False and sorted(said) == ["why", "written"] and type(why) is str and why:
        return {"written": False, "why": why}
    return None


def case_key(out: Path) -> str | None:
    """The SHA-512 of the challenger key the case for the compiled check under ``out`` names (``window/case-v2.json``'s
    ``params.challenger.key``, `Spot.OneStageZk`'s parameters), in lowercase hex; None where the audit left no such case
    or its parameters name no challenger. ValueError for a case that isn't read or a key in another form."""
    case = out / "window" / "case-v2.json"
    if not case.is_file():
        return None
    try:
        doc = json.loads(case.read_text())
        params = doc["params"] if isinstance(doc, Mapping) else None
        if not isinstance(params, Mapping):
            raise ValueError("it holds no parameters")
        named = params.get("challenger")
        if named is None:
            return None
        if not isinstance(named, Mapping):
            raise ValueError("its challenger is not an object")
        return hashlib.sha512(ChallengerKey.from_json(named.get("key")).public).hexdigest()
    except (OSError, KeyError, ValueError) as e:
        raise ValueError(f"{type(e).__name__}: {e}") from None


def _codes(checks: Mapping[str, Any]) -> list[str]:
    return sorted(c for c, v in checks.items() if not (v.get("ok") if isinstance(v, Mapping) else v))


#: a refused member's verdicts, said in this order: its live session's verifier first, whose rejection leaves
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
    units that are wrong, except with probability δ, and its public ``coins`` whose coins the sample is on
    (`whose_coins`) and its public ``case_v2`` whether the audit wrote the case for the compiled check (`case_said`). An
    accepted audit record that says neither, or beside a case for the compiled check that names another challenger key
    (`case_key`), is undecided (module docstring)."""
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
    coins = whose_coins(rec)
    if coins is None:
        return _refusal("the audit record names no challenger key and session a check verified its coins under "
                        "(coins.verified): its acceptance says nothing of whose coins the sample is on", R.digest(reg),
                        "undecided")
    kind = case_said(rec)
    if kind is None:
        return _refusal("the audit record doesn't say whether its audit wrote the case for the compiled check (case_v2): "
                        "its acceptance doesn't say which kind of accept it is", R.digest(reg), "undecided")
    try:
        named = case_key(out)
    except ValueError as e:
        return _refusal(f"the case for the compiled check (window/case-v2.json) isn't read for the challenger it names: {e}",
                        R.digest(reg), "undecided")
    if named is not None and named != coins["key_sha512"]:
        return _refusal(f"the case for the compiled check (window/case-v2.json) names a challenger key with SHA-512 {named}, "
                        f"and the audit's coins verified under the key with SHA-512 {coins['key_sha512']} (coins.verified)",
                        R.digest(reg), "undecided")
    delta = Fraction(wrong["delta"]).limit_denominator(10**12)
    profile = A.profile(reg, draw, wrong["delta"]).to_json()
    profile["delta"] = str(delta)
    per_template = wrong.get("per_template")
    public = {"bound": wrong["bound"], "case_v2": kind, "coins": coins, A.COMPLETE: rec.get("complete") is True,
              "drawn": sorted(draw["units"]), "per_template": per_template, "proved": sorted(rec["proved_units"])}
    return {"commitment": R.digest(reg), "profile": profile, "protocol": PROTOCOL, "public": public, "reason": None,
            "verdict": "accept"}


__all__ = ["PROTOCOL", "answer", "case_key", "case_said", "whose_coins"]
