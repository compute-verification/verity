"""The structured codes the sampled replay (C2 of record) stamps on its record, and the readers the Commit verdict decides from.

A reason the replay could not evaluate (`population.not_evaluable`, keyed by its text) carries a code in
`population.not_evaluable_codes`: {"class": a why class, "family": the Program family the reason counts, "domain": the family is in the
replay version's stated domain (`NOT_YET`), "tags": the rulings that name the gap}.  An unevaluated stratum carries `classes`, and the
sample carries `seed_form` (SEED_FORMS, stamped by the Commit that chose the seed).  The verdict reads those fields, never the texts, so
rewording a message cannot reclassify an outcome.

Records written before the codes carry only the texts.  `not_evaluable_codes`, `stratum_classes` and `seed_form` decode those from the
texts this package wrote, so a record of record grades as it always did.  Dependency-free like `commit_verdict`.
"""
from __future__ import annotations

#: the classes of a 'not evaluated' reason.  BY-NAME classes (a gap of this replay version, stated on the record) may be
#: carried on a PASS by the verdict's named-gap rule when form (B) covers the values; FAULT classes (the evaluator or the row->member mapping
#: failed at run time) never: a fault is a run that did not evaluate, not a gap on the record.
WHY_CLASSES_BY_NAME = ("no-evaluator", "state-linked", "plane-fed", "interior-composed-v2")
WHY_CLASSES_FAULT = ("evaluator-error", "mapping", "missing-input", "device", "other")

#: families the Program names that this replay version cannot replay yet -- counted, never skipped silently (their reasons are the
#: replay version's stated domain: `domain` true)
NOT_YET = {
    # RoPE_v1 has left this list: rows.EVALUATORS["RoPE_v1"].  Attention_v3 / Attention_v2 have too: rows.EVALUATORS -- the row
    # IS the unit under the query of record Q_module_v1.  Older
    # records carry the labels by name ("consumes the request's K/V history across steps (state-linked replay, v2)"; commit_verdict reads
    # them as the replay version of record's stated domain, why_class 'state-linked').  Gemm_v2 (the Hopper GEMM) is NOT here either: it
    # HAS a registered evaluator; before that its VUs were a form-(B)-required gap (commit_verdict._no_evaluator_gaps).
    # NOT_YET is a LABEL on the record, never evidence: a family here is an engineering gap of this replay version.
    "AttentionHead_v1": "consumes the request's K/V history across steps (state-linked replay, v2)",
}

#: how the sample seed was chosen: after commitment from the run root, from a verifier challenge bound to the run root, or by the prover
#: before the run (`--replay-seed`)
SEED_FORMS = ("run-root", "challenge", "prover-chosen")
ROOT_SEED_SOURCE = "run root (first 8 bytes, after commitment)"
PROVER_SEED_SOURCE = "--replay-seed (targeted)"
_CHALLENGE_SEED_PREFIX = "verifier challenge "
_NO_EVALUATOR_PREFIX = "no registered evaluator for "


def challenge_seed_source(challenge: str) -> str:
    """The `seed_source` text of the verifier-challenge form (`sample.challenge_seed`)."""
    return f"{_CHALLENGE_SEED_PREFIX}{challenge[:16]!r} bound to the run root (sha256(challenge|root)[:8], post-commitment)"


def why_class(why: str | None) -> str:
    """The class of a replay-time 'not evaluated' reason, read from the exception text the replay wrote."""
    w = str(why or "")
    if "plane slice mapping failed" in w:
        return "mapping"                                                         # a Program shape the plane mapping refuses: a fault
    if "moe_block_stream" in w:
        return "plane-fed"
    if "interior producer" in w or "composition not possible" in w or "composed evaluator" in w:
        return "interior-composed-v2"
    if "K/V history" in w or "state-linked" in w:
        return "state-linked"
    if "no registered evaluator" in w or "no derived_rows evaluator" in w or w.startswith("no evaluator"):
        return "no-evaluator"
    if "CUDA" in w or "AcceleratorError" in w or "hit the device" in w:
        return "device"
    if w.startswith("evaluator error") or "Error" in w.split(":")[0]:
        return "evaluator-error"
    if "not inside one committed member" in w or "exceeds the interior" in w or "spans fields" in w or "chain deeper" in w or "unmodelled" in w:
        return "mapping"
    if "not available" in w or "not committed" in w or "absent" in w or "missing" in w or "no committed" in w or "not retained" in w \
            or "ambiguous binding" in w:                 # bytes not retained, two bindings of one identity: faults, named
        return "missing-input"
    return "other"


def reason_code(cls: str, family: str | None = None, *, domain: bool = False, tags: tuple[str, ...] = ()) -> dict:
    return {"class": cls, "family": family, "domain": bool(domain), "tags": list(tags)}


def not_evaluable_codes(population: dict) -> dict[str, dict]:
    """Reason text -> code for every `population.not_evaluable` reason: as the replay stamped them, else decoded from the texts."""
    stamped = population.get("not_evaluable_codes")
    if isinstance(stamped, dict):
        return stamped
    domain = {str(v) for v in NOT_YET.values()}
    out = {}
    for why in map(str, population.get("not_evaluable") or {}):
        cls = why_class(why)
        fam = why[len(_NO_EVALUATOR_PREFIX):].strip() if cls == "no-evaluator" and why.startswith(_NO_EVALUATOR_PREFIX) else None
        out[why] = reason_code(cls, fam or None, domain=why in domain, tags=tuple(t for t in ("MAN-06f", "MAN-06g") if t in why))
    return out


def stratum_classes(entry: dict) -> list[str]:
    """The why classes of an unevaluated stratum (`unevaluated_strata[]`): as stamped, else decoded from its reason text."""
    return sorted(set(entry.get("classes") or [entry.get("why_class") or why_class(entry.get("why"))]))


def seed_form(sr: dict) -> str | None:
    """How the replay's sample seed was chosen (one of SEED_FORMS, or "unknown"): as stamped (`sample.seed_form`), else decoded from
    `seed_source`.  None when the record carries neither.  The summary shape (seed / seed_source at the top level) reads the same."""
    smp = sr.get("sample") if isinstance(sr.get("sample"), dict) else {}
    form = sr.get("seed_form") if sr.get("seed_form") is not None else smp.get("seed_form")
    if form is not None:
        return str(form)
    src = sr.get("seed_source") if sr.get("seed_source") is not None else smp.get("seed_source")
    if src is None:
        return None
    src = str(src)
    if src.startswith("--replay-seed"):
        return "prover-chosen"
    if src == ROOT_SEED_SOURCE:
        return "run-root"
    if src.startswith(_CHALLENGE_SEED_PREFIX):
        return "challenge"
    return "unknown"
