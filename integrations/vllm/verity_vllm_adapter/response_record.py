"""[A3 contract 2026-09-10] The prover's step-3 response record: the exemption request ``F`` and, per selected RU, the two fields the
contract's reporting requirements demand (docs/PROTOCOL.md "Reporting requirements added 2026-09-10 10:56 PT"; conformance R8, R17, R18).

Pure Python (no torch / vLLM): imported by ``selective_prover`` (writer) and ``e2e.verifier_v3`` (reader / checker) and unit-tested on CPU.

``exemption_request.json`` (schema ``veritor/exemption-request/v1``):

    {"schema", "instance_digest", "r_q", "J": [ru...], "F": [ru...], "retained": [ru...], "B_ex": int, "B_ex_unit": "replay_units",
     "budget_exceeded": bool, "per_ru": {"<ru>": {"request_id", "consistency": {...}, "response": {...}, "unsupported": {...}}}}

Field (a) -- **consistency of committed inputs and outputs** (R17a)::

    {"status": "established consistent" | "established inconsistent" | "unknown",
     "method": str, "evidence": {...}, "established_by": "prover" | "verifier" | null}

``consistency_from_unforced`` is the prover-side determination: during the teacher-forced replay the sampler's *unforced* choice
(the argmax of the recomputed logits at every output position, before the forced token is applied) is compared with the committed
output token at that position.  If every output position is covered and every unforced choice equals the committed token, then by
induction over positions the unforced evaluation of \\(f_R\\) on the committed inputs reproduces the committed output tuple exactly:
``established consistent`` (established by the PROVER's own evaluation under the replay configuration -- its equivalence to the
specified \\(C\\) is R19's, not established here).  One covered position whose unforced choice differs from the committed token is a
confirmed mismatch of \\(f_R(v_{in})\\) with \\(v_{out}\\) at that position: ``established inconsistent``.  Anything else (production not
greedy, positions not covered, hook did not fire) is ``unknown`` -- never inferred from "the forced trajectory was followed" (vacuous
under forcing), from RU roots agreeing, or from sampled checks passing.

Field (b) -- **ability to supply the response** (R17b)::

    {"step3_committed": bool, "exemption_requested": bool, "exemption_reason": str | null,
     "selected_checks_establishable": bool | null, "failure_stage": str | null, "failure_cause": str | null,
     "work": {"replay_s": float | null, "commit_s": float | null, "diagnostic_s": float | null}}

``selected_checks_establishable`` is filled after Q by the verifier (``e2e.verifier_v3.steps.ru_fields``); the prover leaves it null.

**Unsupported-check exposure** (R18, protocol consequence only; A5/A6 own the unsupported set)::

    {"unsupported_vus_n": int | null, "unsupported_kinds": [kind...], "source": str}

``n`` = the number of distinct VUs of the RU whose kind has no checker in the ACTUAL configuration (population kind counts x the
configured unsupported-kind set); the runner reports \\(1-(1-s)^n\\) per retained RU and \\(1-(1-s)^N\\) aggregate as *completion risk*.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import hashlib
import json
import math
from fractions import Fraction
from typing import Any

SCHEMA = "veritor/exemption-request/v1"
RU_FIELDS_SCHEMA = "veritor/ru-fields/v1"
CONSISTENT = "established consistent"
INCONSISTENT = "established inconsistent"
UNKNOWN = "unknown"
CONSISTENCY_STATUSES = (CONSISTENT, INCONSISTENT, UNKNOWN)
ESTABLISHED_BY = ("prover", "verifier")
FAILURE_STAGES = ("replay", "boundary", "capture", "commit", "consistency", "policy")

#: [integrator note 19:20Z, R18] The prover's exemption POLICY -- a PRE-Q rule (F is fixed at step 3, before r_s; Q is unknown):
#:   ``gamble``             exempt nothing for unsupported checks; step 5 rejects if Q draws an unsupported VU (completion risk 1-(1-s)^n);
#:   ``exempt-unsupported`` exempt every selected RU that contains >= 1 VU whose check the implementation cannot establish (n_R > 0, or n_R
#:                          unknown because the configuration was not supplied -- conservative), subject to |F| <= B_ex (never trimmed).
#: Both are selectable by configuration, named in the request and the record; neither may look at Q.
EXEMPTION_POLICIES = ("gamble", "exempt-unsupported")
POLICY_DECIDED_AT = "step 3: with the interior commitment, before r_s (Q unknown)"


def unknown(method: str, **evidence: Any) -> dict[str, Any]:
    return {"status": UNKNOWN, "method": method, "evidence": evidence, "established_by": None}


#: [W-A9-7] where the prover's unforced choice at an output position came from
UNFORCED_SOURCE_HOOK = "replay-hook"  # the sampler's trace-replay hook (the argmax tensor before the forced token is applied)
UNFORCED_SOURCE_LOGPROBS = "persisted-logprobs-rank1"  # the replay output's per-position logprobs: the rank-1 token of the recomputed logits


def unforced_from_logprobs(logprobs: Sequence[Mapping[int, Any] | None] | None, prompt_len: int) -> dict[int, int]:
    """[W-A9-7] The unforced argmax per absolute position from the replay output's PERSISTED per-position logprobs (vLLM ``--logprobs >= 1``:
    each output position carries a token -> (logprob, rank) mapping over the recomputed logits, the forced token included).  The token of
    rank 1 at output ``k`` is the recomputed argmax at position ``prompt_len - 1 + k`` -- the same quantity the trace-replay hook records,
    obtained from what the engine persists, so the determination no longer depends on the hook firing.  Positions without a rank-1 entry are
    left out (they count as not covered)."""

    out: dict[int, int] = {}
    for k, lp in enumerate(logprobs or []):
        if not lp:
            continue
        top = [tok for tok, e in lp.items() if int(getattr(e, "rank", 0) or (e.get("rank", 0) if isinstance(e, Mapping) else 0)) == 1]
        if len(top) == 1:
            out[prompt_len - 1 + k] = int(top[0])
    return out


def merge_unforced(hook: Mapping[int, int], persisted: Mapping[int, int]) -> tuple[dict[int, int], dict[str, Any]]:
    """[W-A9-7] One unforced-choice map from both sources, and the record of which source covered which positions (a disagreement between the
    two sources at a position is recorded and that position is NOT covered -- neither source is trusted over the other)."""

    merged: dict[int, int] = {}
    conflicts: list[dict[str, int]] = []
    for t in sorted(set(hook) | set(persisted)):
        if t in hook and t in persisted and hook[t] != persisted[t]:
            conflicts.append({"t": t, UNFORCED_SOURCE_HOOK: hook[t], UNFORCED_SOURCE_LOGPROBS: persisted[t]})
            continue
        merged[t] = hook[t] if t in hook else persisted[t]
    src = {"positions_from_hook": len(hook), "positions_from_persisted_logprobs": len(persisted), "source_conflicts": conflicts[:8], "source_conflict_count": len(conflicts),
           "sources": [s for s, m in ((UNFORCED_SOURCE_HOOK, hook), (UNFORCED_SOURCE_LOGPROBS, persisted)) if m]}
    return merged, src


def consistency_from_unforced(output_token_ids: list[int], unforced_by_position: dict[int, int], prompt_len: int, *,
                              greedy: bool, established_by: str = "prover", configuration: str = "", sources: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Field (a) from the prover's unforced sampler choices (see the module docstring).  ``unforced_by_position[t]`` is the argmax of the
    recomputed logits at absolute position ``t`` (``t = prompt_len - 1 + k`` produces output ``k``).  ``sources`` ([W-A9-7] :func:`merge_unforced`)
    records where the choices came from (replay hook / persisted logprobs) so the determination names its evidence."""

    n = len(output_token_ids)
    positions = [prompt_len - 1 + k for k in range(n)]
    covered = [t for t in positions if t in unforced_by_position]
    method = "prover replay (teacher-forced): unforced argmax of the recomputed logits at each output position vs the committed output token"
    if sources and sources.get("sources"):
        method += f" [unforced choice from: {', '.join(sources['sources'])}]"
    base = {"output_positions": n, "positions_covered": len(covered), "configuration": configuration, **({"unforced_sources": dict(sources)} if sources else {})}
    if not greedy:
        return unknown(method, **base, reason="production sampling was not greedy: the specified token-select is not the argmax; no determination")
    mismatches = [{"k": t - (prompt_len - 1), "committed": output_token_ids[t - (prompt_len - 1)], "recomputed": unforced_by_position[t]}
                  for t in covered if unforced_by_position[t] != output_token_ids[t - (prompt_len - 1)]]
    if mismatches:
        return {"status": INCONSISTENT, "method": method, "established_by": established_by,
                "evidence": {**base, "mismatches": mismatches[:16], "mismatch_count": len(mismatches),
                             "note": "a confirmed mismatch of f_R(v_in) with v_out at a covered position; it does not by itself identify the cause (kernel, capture, addressing, encoding)"}}
    if len(covered) == n and n > 0:
        return {"status": CONSISTENT, "method": method, "established_by": established_by,
                "evidence": {**base, "agree": n, "note": "every output position covered and equal: the unforced evaluation of f_R on the committed inputs reproduces the committed "
                                                         "output tuple (by induction over positions); established under the replay configuration (its equivalence to C is R19)"}}
    return unknown(method, **base, reason=("no output positions" if n == 0 else f"only {len(covered)} of {n} output positions have an unforced choice recorded"))


def response(*, step3_committed: bool, exemption_requested: bool, exemption_reason: str | None = None, failure_stage: str | None = None,
             failure_cause: str | None = None, replay_s: float | None = None, commit_s: float | None = None, diagnostic_s: float | None = None) -> dict[str, Any]:
    """Field (b).  ``selected_checks_establishable`` is null until the verifier fills it after Q."""

    if failure_stage is not None and failure_stage not in FAILURE_STAGES:
        raise ValueError(f"failure_stage {failure_stage!r} not in {FAILURE_STAGES}")
    return {"step3_committed": bool(step3_committed), "exemption_requested": bool(exemption_requested), "exemption_reason": exemption_reason,
            "selected_checks_establishable": None, "failure_stage": failure_stage, "failure_cause": failure_cause,
            "work": {"replay_s": replay_s, "commit_s": commit_s, "diagnostic_s": diagnostic_s}}


def unsupported(kind_counts: dict[str, int] | None, unsupported_kinds: list[str] | set[str] | None, *, source: str) -> dict[str, Any]:
    """R18: ``n`` distinct VUs of this RU whose kind has no checker in the actual configuration.  ``kind_counts`` = ``{"fn|form": count}``
    for ONE RU (population/v2 ``kind_counts()`` of the per-instance population when every RU is one program instance).  ``None`` when the
    configuration was not supplied: the field is present and null, never estimated."""

    if kind_counts is None or unsupported_kinds is None:
        return {"unsupported_vus_n": None, "unsupported_kinds": sorted(unsupported_kinds or []), "source": source + " (kind counts or unsupported set not supplied: n unknown)"}
    uk = set(unsupported_kinds)
    n = sum(int(c) for k, c in kind_counts.items() if k in uk)
    return {"unsupported_vus_n": n, "unsupported_kinds": sorted(k for k in uk if kind_counts.get(k, 0) > 0), "source": source}


def build(*, instance_digest: str, r_q: str, J: list[int], ru_request: dict[int, str], per_ru: dict[int, dict[str, Any]], B_ex: int,
          policy: str = "gamble") -> dict[str, Any]:
    """The exemption request.  ``F`` = the RUs of ``J`` whose response says ``exemption_requested`` (replay/boundary/capture/commit/consistency
    failures) plus, under ``policy == "exempt-unsupported"``, every RU with an unsupported VU (``per_ru[r]["unsupported"]["unsupported_vus_n"]``
    > 0 or unknown) -- decided from the configuration and the replay, NEVER from Q (unknown at step 3).  The prover never trims ``F`` to
    the budget (that would be a policy change to make the run pass): an over-budget request is recorded as such and the verifier refuses it."""

    Jl = sorted(int(r) for r in J)
    missing = [r for r in Jl if r not in per_ru]
    if missing:
        raise ValueError(f"per_ru lacks selected RU(s) {missing}: every RU of J gets both fields")
    extra = sorted(set(per_ru) - set(Jl))
    if extra:
        raise ValueError(f"per_ru names RU(s) {extra} outside J")
    if policy not in EXEMPTION_POLICIES:
        raise ValueError(f"exemption policy {policy!r} not in {EXEMPTION_POLICIES}")
    policy_exempted: list[int] = []
    if policy == "exempt-unsupported":
        for r in Jl:
            rsp, u = per_ru[r]["response"], per_ru[r].get("unsupported") or {}
            n = u.get("unsupported_vus_n")
            if not rsp["exemption_requested"] and (n is None or int(n) > 0):
                rsp["exemption_requested"] = True
                rsp["exemption_reason"] = (f"policy exempt-unsupported: the RU contains {n} VUs whose check the implementation cannot establish" if n is not None
                                           else "policy exempt-unsupported: the unsupported count is unknown (configuration not supplied) -- exempted conservatively")
                rsp["failure_stage"], rsp["failure_cause"] = "policy", rsp["exemption_reason"]
                policy_exempted.append(r)
    F = [r for r in Jl if per_ru[r]["response"]["exemption_requested"]]
    doc = {"schema": SCHEMA, "instance_digest": instance_digest, "r_q": r_q, "J": Jl, "F": F, "retained": [r for r in Jl if r not in set(F)],
           "B_ex": int(B_ex), "B_ex_unit": "replay_units", "budget_exceeded": len(F) > int(B_ex),
           "exemption_policy": {"name": policy, "decided_at": POLICY_DECIDED_AT, "policy_exempted": policy_exempted,
                                "rule": ("exempt nothing for unsupported checks; step 5 rejects when Q draws an unsupported VU" if policy == "gamble" else
                                         "exempt every selected RU with >= 1 unsupported VU (or unknown count); |F| <= B_ex is the verifier's check, F is never trimmed")},
           "per_ru": {str(r): {"request_id": ru_request[r], **per_ru[r]} for r in Jl},
           "note": "docs/PROTOCOL.md step 3: F ⊆ J submitted with the interior commitment, before r_s; per RU the two fields of the reporting requirements (R17) and the "
                   "unsupported-check count (R18)"}
    return doc


def digest(doc: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(doc: dict[str, Any], *, J: list[int], B_ex: int, instance_digest: str | None = None, r_q: str | None = None,
             policy: str | None = None) -> list[str]:
    """The verifier's structural checks on the request (step 4 runs these BEFORE r_s): schema, J agreement, F ⊆ J, |F| <= B_ex,
    both fields present and well-formed for every RU of J, the pre-Q exemption policy named (and == ``policy`` when the verifier configured one)."""

    p: list[str] = []
    if doc.get("schema") != SCHEMA:
        p.append(f"schema {doc.get('schema')!r} != {SCHEMA}")
        return p
    if instance_digest is not None and doc.get("instance_digest") != instance_digest:
        p.append("exemption request is for another instance")
    if r_q is not None and doc.get("r_q") != r_q:
        p.append("exemption request is bound to another r_q")
    Jl = sorted(int(r) for r in J)
    if list(doc.get("J", [])) != Jl:
        p.append(f"request J {doc.get('J')} != chronology J {Jl}")
    F = list(doc.get("F", []))
    if F != sorted(set(F)):
        p.append("F is not a sorted set")
    if not set(F) <= set(Jl):
        p.append(f"F names RU(s) {sorted(set(F) - set(Jl))} outside J")
    if len(F) > B_ex:
        p.append(f"|F| = {len(F)} > B_ex = {B_ex} (RU count)")
    if doc.get("B_ex") != B_ex:
        p.append(f"request states B_ex = {doc.get('B_ex')} but the instance declares {B_ex}")
    if doc.get("retained") != [r for r in Jl if r not in set(F)]:
        p.append("retained != J minus F")
    pol = doc.get("exemption_policy") or {}
    if pol.get("name") not in EXEMPTION_POLICIES:
        p.append(f"exemption_policy.name {pol.get('name')!r} not in {EXEMPTION_POLICIES} (the pre-Q policy must be named)")
    if policy is not None and pol.get("name") != policy:
        p.append(f"exemption_policy.name {pol.get('name')!r} != the configured policy {policy!r}")
    per = doc.get("per_ru") or {}
    if sorted(int(k) for k in per) != Jl:
        p.append("per_ru does not cover exactly J")
    for k, rec in per.items():
        c, rsp = rec.get("consistency") or {}, rec.get("response") or {}
        if c.get("status") not in CONSISTENCY_STATUSES:
            p.append(f"RU {k}: consistency.status {c.get('status')!r} not in {CONSISTENCY_STATUSES}")
        if c.get("status") != UNKNOWN and c.get("established_by") not in ESTABLISHED_BY:
            p.append(f"RU {k}: an established consistency must name who established it")
        if c.get("status") == UNKNOWN and c.get("established_by") is not None:
            p.append(f"RU {k}: unknown consistency cannot name an establisher")
        if not str(c.get("method", "")).strip():
            p.append(f"RU {k}: consistency.method missing")
        for f in ("step3_committed", "exemption_requested"):
            if not isinstance(rsp.get(f), bool):
                p.append(f"RU {k}: response.{f} must be a bool")
        if rsp.get("failure_stage") is not None and rsp.get("failure_stage") not in FAILURE_STAGES:
            p.append(f"RU {k}: response.failure_stage {rsp.get('failure_stage')!r} not in {FAILURE_STAGES}")
        if (int(k) in set(F)) != bool(rsp.get("exemption_requested")):
            p.append(f"RU {k}: F membership disagrees with response.exemption_requested")
        if int(k) not in set(F) and not rsp.get("step3_committed"):
            p.append(f"RU {k}: retained but step3_committed is false (a retained RU must have its step-3 values committed before r_s)")
        u = rec.get("unsupported")
        if not isinstance(u, dict) or "unsupported_vus_n" not in u:
            p.append(f"RU {k}: unsupported.unsupported_vus_n missing (R18: present, possibly null; never estimated)")
    return p


def p_at_least_one(s: Fraction, n: int) -> float:
    """\\(1-(1-s)^n\\) as a float: exact rational arithmetic for small n, ``-expm1(n * log1p(-s))`` otherwise (n reaches 1e11 for the
    legacy kind table; the exact Fraction power is infeasible there)."""

    if n <= 0:
        return 0.0
    if n <= 4096:
        return float(1 - (1 - s) ** n)
    return -math.expm1(n * math.log1p(-float(s)))


def completion_risk(s: Fraction | str, per_ru_n: dict[int, int | None]) -> dict[str, Any]:
    """R18: conditional on J, F: per retained RU \\(1-(1-s)^n\\); aggregate \\(1-(1-s)^N\\) with N = sum of distinct unsupported VUs over
    J minus F (RUs are disjoint, so the sum is without double counting).  Labelled completion risk, not rejection probability, not a
    security allowance, not an exemption.  Null when any n is unknown."""

    s = Fraction(s)
    per: dict[str, Any] = {}
    N = 0
    known = True
    for r, n in sorted(per_ru_n.items()):
        if n is None:
            per[str(r)] = {"n": None, "p_select_at_least_one": None}
            known = False
            continue
        per[str(r)] = {"n": int(n), "p_select_at_least_one": p_at_least_one(s, int(n))}
        N += int(n)
    agg = p_at_least_one(s, N) if known else None
    return {"label": "completion risk caused by unsupported checks (conditional on J, F); NOT the rejection probability, NOT a security error allowance, NOT an exemption",
            "s": str(s), "per_retained_ru": per, "N_distinct_unsupported": N if known else None, "aggregate_p_select_at_least_one": agg,
            "note": "counts from the actual configuration (population kind counts x configured unsupported-kind set); a fallback removes a kind only when implemented"}
