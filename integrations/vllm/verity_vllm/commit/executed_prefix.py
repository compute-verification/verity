"""The EXECUTED PREFIX OF RECORD of every request -- the execution extent check (`executed_prefix`),
the one fact the Commit's consumers count on.

A request has TWO lengths.  `served_output_len` a_r = the tokens the driver served (its EOS included);
`executed_prefix_len` = the engine rows that ran for it = the cap when the request generated its cap (length stop), a_r + LAG when it
stopped by EOS before the cap -- the LAG row(s) after the EOS are active internal computation (their VUs are required) whose sampled token
is not served.  LAG is the workload Program's DECLARED specialization parameter (Workload_v1{..., lag}); the served tokens are the driver's;
NOTHING is read from the trace.  Positions >= executed_prefix_len are programmed-but-unexecuted: not required of the record, and a committed
Call there is 'execution past the stop' (the caller checks the binding map against `request_steps`).

Fail closed, by name: an early stop (a_r < cap) with no declared stop rule, no declared LAG, or a last served token that is not a declared
EOS id is a problem -- the population then stays the cap's (nothing is accepted) and `ok` is False.  Never default LAG 0 or 1.

Example: on an OLMoE B2 i32/o32 mixed-stop workload r0 serves 11 / executes 12 (EOS 50279 at output index 10, LAG 1) and r1 32 / 32
(cap) -- G6 PASS with the lag exercised; a population placed over every request's cap would require r0's 20 unexecuted steps and fail
by name.

This holds in a box -- the ACTIVE-REFERENCE table only, on three conditions `check.executed_prefix.link_to_match_account` checks:
(a) n_exec is determined from the properly LINKED serving transcript: the Commit's own served tokens count only because they EQUAL the Match
record's served output (capture/tokens.json of the same attempt) and the Match account (global match G6: PASS, the same two lengths per request)
-- "the committer says EOS happened here" is never sufficient by itself; (b) the lag rule is attested: declared == the engine's read-back ==
the record's tokens.json; (c) any actual execution past the boundary FAILS (`rows_past_the_stop`).  NOT-EXECUTED-BY-RULE is an
active-reference implementation classification only.  In the lifted fixed Program the corresponding positions are REQUIRED logical values
assigned the canonical inactive representation tau(e) = bot; there is no not-required-because-not-executed category.

A config run has no Match phase.  Its linked serving transcript is the Commit's own token record (`check.commit_tokens`): the driver's
served tokens beside the commitment, each opened from the run root and verified equal to the committed sampled id it was served from
(`link_to_commit_account`).  The prefix then rests on committed values, not on the committer's count; with a Match record too, both hold
when the row committed its sampled ids, and the Match record alone when it committed none.

PROTOCOL SUMMARY (Commit phase: the execution extent check (`executed_prefix`) -- how far each request actually ran)
  Purpose: per request, derive the two lengths every Commit consumer counts on -- served_output_len (tokens served, EOS included) and
    executed_prefix_len (cap on a length stop; served + LAG on an EOS stop) -- from the served tokens + the Program's DECLARED stop
    rule (eos ids, lag), never from the trace; then accept it only when LINKED to an account of the same attempt: the Match account,
    or, with no Match phase, the Commit's token record bound to the commitment (`check.commit_tokens`).
  Inputs: served tokens per request, caps, `eos_of_record` + `lag` (global_program.stop_rule / Workload_v1.lag), the Match account
    (match/capture/tokens.json + global_match.json G6 via `check.executed_prefix.load_match_account`), the token record's link
    (binding.coverage.commit_account_link), the binding map (for `rows_past_the_stop`), the C2 boundary linkage's horizon.
  Outputs: dicts consumed by commit_delta / binding / sampled_replay: `executed_prefix_of_record` {requests{served_output_len,
    executed_prefix_len, cap, eos_stop}, ok, problems, any_eos_stop, lag}, `request_steps_of_record`, `rows_past_the_stop`,
    `check.executed_prefix.link_to_match_account` {ok, required, problems, tokens_equal, g6, match_dir}, `executed_prefix_facts` (the named facts
    sampled_replay takes, from whichever account is of record).
  Invariants: LAG is never defaulted to 0 or 1 -- an early stop with no declared EOS set / LAG, a non-EOS last token, or served > cap
    is a problem and the population stays the cap's; a prefix reduced from its cap on the committer's own count alone is never
    accepted (condition (a) above): it rests on the Match account or on served tokens opened from the run root, both when both are beside the
    Commit and the row committed its sampled ids; the lag must agree on every account (declared == tokens.json == G6, or declared == the
    token record's == the engine's read-back); the boundary linkage a token record leans on runs over the same executed prefix; a committed Call past the prefix fails by name.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

from verity_vllm.records.commit_binding import CommitBinding

ACTIVE_REFERENCE_ONLY = ("NOT-EXECUTED-BY-RULE is an active-reference implementation classification only. In the lifted fixed Program the "
                         "corresponding positions are REQUIRED logical values assigned the canonical inactive representation tau(e) = bot; "
                         "there is no not-required-because-not-executed category. (COORD R17-27)")

RULE = ("executed_prefix_len = cap when served_output_len == cap (length stop); served_output_len + LAG when the request stopped by EOS "
        "before its cap (the last served token must be a declared EOS id, F-pm-24, and it must be the FIRST EOS served -- tokens served after an "
        "EOS are served past the stop, F-r17-14); LAG = the workload Program's declared static (R17-1), never read from the trace; positions >= "
        "executed_prefix_len are programmed-but-unexecuted (not required; committed = execution past the stop)")


def executed_prefix_of_record(requests: Iterable[str], served_tokens: dict[str, list[int]] | None, *, caps: dict[str, int],
                              eos_of_record: Iterable[int] | None, lag: int | None) -> dict[str, Any]:
    """`requests`: the workload's request ids (r<i>); `served_tokens`: {r<i>: [served token ids]} from the driver (the Commit's own run; None =
    no driver tokens -> every request's prefix is its cap and `served_known` is False); `caps`: {r<i>: declared max_tokens};
    `eos_of_record`: the Program's declared EOS ids (None = undeclared); `lag`: the Program's declared LAG (None = undeclared / unattested).
    Returns {"ok", "rule", "lag", "eos_of_record", "served_known", "any_eos_stop", "requests": {r<i>: {...}}, "problems": [str]}."""
    eos_set = None if eos_of_record is None else {int(x) for x in eos_of_record}
    lag_v = None if lag is None else int(lag)
    out: dict[str, Any] = {}
    problems: list[str] = []
    any_eos = False
    for rid in requests:
        rid = str(rid)
        cap = int(caps[rid])
        toks = None if served_tokens is None else served_tokens.get(rid)
        rec: dict[str, Any] = {"cap": cap, "served_output_len": None, "executed_prefix_len": cap, "cap_exhausted": None, "eos_stop": None,
                               "last_served_token": None, "lag": lag_v, "why": None}
        if toks is None:
            rec["why"] = "no served tokens for the request from the driver: executed prefix = the cap (unverified)"
            if served_tokens is not None:
                problems.append(f"{rid}: {rec['why']}")
            out[rid] = rec
            continue
        a = len(toks)
        rec["served_output_len"] = a
        rec["last_served_token"] = int(toks[-1]) if a else None
        # the Program's rule stops at the FIRST declared EOS: a served list that carries an EOS at index k and more tokens
        # after it was served PAST the stop -- FAIL by name, whatever the last token is (served [EOS, x, EOS] is not an EOS stop at 3)
        first_eos = next((k for k, t in enumerate(toks) if eos_set is not None and int(t) in eos_set), None)
        rec["first_eos_index"] = first_eos
        if first_eos is not None and first_eos < a - 1:
            rec["why"] = (f"served past an EOS at index {first_eos}: served token {int(toks[first_eos])} is a declared EOS id but {a - 1 - first_eos} "
                          f"more token(s) were served after it (served {a}, cap {cap}); the Program's rule stops at the FIRST EOS -- served_output_len "
                          f"would be {first_eos + 1}")
            problems.append(f"{rid}: {rec['why']}")
        elif a > cap:
            rec["why"] = f"served {a} tokens over the declared cap {cap}"
            problems.append(f"{rid}: {rec['why']}")
        elif a == cap:
            rec["cap_exhausted"], rec["eos_stop"] = True, False
        elif a == 0:
            rec["cap_exhausted"], rec["eos_stop"] = False, None
            rec["why"] = "served no token at all: not a stop of the Program's rule"
            problems.append(f"{rid}: {rec['why']}")
        else:
            rec["cap_exhausted"], rec["eos_stop"] = False, True
            any_eos = True
            if eos_set is None:
                rec["why"] = (f"served {a} of its cap {cap} and the Program declares no stop rule (eos_token_ids): the early stop cannot be "
                              f"verified, fail closed")
                problems.append(f"{rid}: {rec['why']}")
            elif rec["last_served_token"] not in eos_set:
                rec["why"] = (f"served {a} of its cap {cap} but its last served token {rec['last_served_token']} is not a declared EOS id "
                              f"{sorted(eos_set)[:8]}")
                problems.append(f"{rid}: {rec['why']}")
            elif lag_v is None:
                rec["why"] = (f"stopped by EOS at served {a} of its cap {cap} but the Program declares no LAG (or the engine does not attest the "
                              f"declared scheduler): the executed prefix cannot be placed -- never default 0 or 1")
                problems.append(f"{rid}: {rec['why']}")
            elif lag_v not in (0, 1):
                rec["why"] = f"declared LAG {lag_v} is not in {{0, 1}}"
                problems.append(f"{rid}: {rec['why']}")
            else:
                rec["executed_prefix_len"] = min(cap, a + lag_v)
                rec["eos_step"] = a - 1                                        # request-relative: the row whose sampled token is the EOS
                rec["lag_steps"] = list(range(a, rec["executed_prefix_len"]))  # the placeholder row(s): VUs required, sampled token not served
        out[rid] = rec
    return {"ok": not problems, "rule": RULE, "lag": lag_v, "eos_of_record": (sorted(eos_set) if eos_set is not None else None),
            "served_known": served_tokens is not None, "any_eos_stop": any_eos, "requests": out, "problems": problems}


def request_steps_of_record(epr: dict[str, Any], arrivals: dict[str, int]) -> dict[str, list[int]]:
    """{r<i>: [engine steps the request executed]} = range(arrival, arrival + executed_prefix_len) from `executed_prefix_of_record`."""
    return {rid: list(range(int(arrivals.get(rid, 0)), int(arrivals.get(rid, 0)) + int(rec["executed_prefix_len"]))) for rid, rec in epr["requests"].items()}


def executed_prefix_facts(epr: dict[str, Any], binding: CommitBinding | Mapping[str, Any] | None, linkage: dict[str, Any] | None, arrivals: dict[str, int],
                           stop: dict[str, Any], served_tokens: dict[str, list[int]] | None = None) -> dict[str, dict[str, Any]]:
    """The per-request FACTS OF RECORD `sampled_replay.executed_prefix_of_record`
    normalises into the not-executed class -- every one a named record fact, never a literal: `a_r` READ from the Match record
    (binding.coverage.match_account_link: tokens.json served_by_request, else G6 lengths_by_request; `a_r_source` names the file + sha256;
    with NO Match record a_r is the Commit's own served length, labelled so, and `linked` is False -> the class faults by name);
    `served_len_commit` = this run's own served length; `linked` = the Match-account link held (Commit served == Match served per request,
    global match PASS, G6 agrees) AND the C2 boundary linkage passed; `lag_attested` = declared == tokens.json == G6 declared == G6 record_attests;
    `eos_declared` = the served token a_r-1 is a declared EOS id; `rows_past_stop` = coverage's execution-past-the-stop count.
    With `served_tokens` (this run's own served ids per request) the facts also carry `served` (the list)
    and `eos_ids` (the Program's declared set) so a consumer that derives the class from Continuation_v1's recurrence (r_t) does so from the
    record's own list -- ONE code path on Build, Match and Commit; the caller passes only the keys the consumer's signature accepts.
    With no Match record beside the Commit (no `match_dir`) the account is the Commit's token record (binding.coverage.commit_account_link,
    check.commit_tokens): `a_r` is its served length, `linked` = that link held, every served token of the request was opened from the
    run root equal to the committed sampled id, the C2 boundary linkage passed AND ran over exactly the request's executed prefix, and
    `lag_attested` is that link's, and `a_r_source` says so as a note, not a missing account file.  With both, both must hold wherever the token
    record places a shorter prefix; a token-record link not applied (a row that committed no sampled id) places none."""
    cov = None if binding is None else CommitBinding.of(binding, "the Commit's binding").coverage
    link = None if cov is None else cov.match_account_link
    tj = None if link is None else link.tokens_json
    g6 = None if link is None else link.global_match
    cl = None if cov is None else cov.commit_account_link
    by_commit = cl is not None and not (link is not None and link.match_dir)
    with_commit = by_commit or (cl is not None and cl.required)
    horizon = (linkage or {}).get("horizon") or {}
    # the account files that are NOT beside the Commit, named: a Commit whose match/capture/ moved away while it was queued has no
    # tokens.json (a_r per request, the lag attestation), the link falls to G6 for a_r, and the class would fault 'a_r is not linked
    # (linked=False)' + 'LAG rule not attested' -- a MISSING INPUT read as a verdict on the Program.  The fact names the missing file(s);
    # the consumer faults on it by name.
    account_missing = [] if by_commit else [leg for leg, present in (("tokens.json", tj is not None), ("global_match.json", g6 is not None)) if not present]
    past = {} if cov is None or cov.rows_past_the_stop is None else cov.rows_past_the_stop.steps_past_by_request
    eos_set = {int(x) for x in (stop.get("eos") or [])} if stop.get("eos") is not None else None
    decl_lag = stop.get("declared_lag")
    lk_ok = bool(linkage and linkage.get("result") is True)
    out: dict[str, dict[str, Any]] = {}
    for rid, rec in (epr.get("requests") or {}).items():
        served_commit = rec.get("served_output_len")
        m_served = None if tj is None else tj.served_by_request.get(rid)
        g_len = None if g6 is None or g6.served_by_request is None else g6.served_by_request.get(rid)
        tr_served = None if cl is None or cl.served_by_request is None else cl.served_by_request.get(rid)
        if m_served is not None:
            a_r, src = int(m_served), f"match record {tj.path} (sha256 {tj.sha256[:16]}): tokens.json generated_token_ids"
        elif g_len is not None:
            a_r, src = int(g_len), f"match record {g6.path} (sha256 {g6.sha256[:16]}): G6 lengths_by_request.served_output_len"
        elif by_commit and tr_served is not None:
            a_r, src = int(tr_served), (f"commit token record {cl.record_path} (sha256 {str(cl.record_sha256)[:16]}): generated_token_ids opened "
                                       "from the run root; no Match record beside the Commit (a config run has no Match stage): the prefix is "
                                       "linked to the Commit's token record")
        else:
            a_r, src = int(served_commit if served_commit is not None else rec["cap"]), "the Commit's OWN served tokens (no Match record beside the Commit: NOT linked)"
        if account_missing:
            src += f"; Match account leg(s) NOT beside the Commit: {', '.join(account_missing)}"
        linked = bool(link is not None and link.ok and tj is not None and tj.tokens_equal.get(rid) is True
                      and g6 is not None and g6.verdict == "PASS" and lk_ok)
        lag_attested = link is not None and link.lag_attested
        if with_commit:
            # the boundary linkage must have run over exactly this executed prefix: the committed tokens are the ones the next step consumed
            linked_c = bool(cl.ok is True and cl.tokens_committed is not None and cl.tokens_committed.get(rid) is True and lk_ok
                            and (horizon.get(rid) or {}).get("linked_steps") == rec.get("executed_prefix_len"))
            linked = linked_c if by_commit else (linked and linked_c)
            lag_attested = bool(cl.lag_attested) and (by_commit or lag_attested)
        out[rid] = {"a_r": a_r, "a_r_source": src, "lag": (int(decl_lag) if decl_lag is not None else -1), "cap": int(rec["cap"]), "arrive": int(arrivals.get(rid, 0)),
                    "eos_declared": (None if rec.get("last_served_token") is None or eos_set is None else (int(rec["last_served_token"]) in eos_set)),
                    "served_len_commit": served_commit, "linked": linked, "lag_attested": lag_attested,
                    "rows_past_stop": int(past.get(rid, 0)), "account_missing": list(account_missing)}
        if served_tokens is not None and served_tokens.get(rid) is not None:
            out[rid]["served"] = [int(t) for t in served_tokens[rid]]
            out[rid]["eos_ids"] = (sorted(eos_set) if eos_set is not None else None)
    return out


def facts_accepted_by(fn, facts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The facts a consumer's signature accepts (keyword-only parameters): a tree whose `executed_prefix_of_record` predates `served=` /
    `eos_ids=` still receives the arithmetic's facts; a newer one receives the served list too.  Never a TypeError."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return facts
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return facts
    keep = set(params)
    return {rid: {k: v for k, v in f.items() if k in keep} for rid, f in facts.items()}


#: the activity a PADDED record's binding map (`padded-binding-map/1`, padding_steps) stamps on every entry:
#: `bot-by-enablement` = the REQUIRED padding of a not-executed row (its value is tau(e) = bot by rule, NOT_EXECUTED_SCOPE verbatim), `active` =
#: an executed row's committed value, `registered-input` = a named input.  An UNPADDED record's entries carry no `activity` (= active).
ACTIVITY_PADDING = "bot-by-enablement"


def rows_past_the_stop(bmap: dict[str, Any], request_steps: dict[str, Iterable[int]], *, max_report: int = 8) -> dict[str, Any]:
    """Committed leaf entries of a declared request at an engine step OUTSIDE its executed prefix of record = 'execution past the stop'
    (a row the Program says was never executed carries committed values): named per request with the steps, FAIL by name.  Entries of
    other request ids (`*`, unmapped) are not this check's.

    On a PADDED record the rows past the stop are PRESENT by construction -- every not-executed row is a
    required value assigned bot -- and each such entry says so (`activity = bot-by-enablement`): those are the padding, counted
    (`padding_entries_n`), never execution past the stop.  An entry there that is `active` (or has no activity field: an unpadded record)
    IS execution past the stop, exactly as before.  The converse is named too: a `bot-by-enablement` entry INSIDE the executed prefix is a
    row the Program says was executed and the record says was not -- the prohibited inactive fallback, FAIL by name."""
    allowed = {str(k): {int(s) for s in v} for k, v in request_steps.items()}
    past: dict[str, set[int]] = {}
    inside: dict[str, set[int]] = {}
    n = n_pad = n_inside = 0
    for e in bmap.get("entries") or []:
        if e.get("kind") != "leaf":
            continue
        i = e.get("identity") or {}
        rid = str(i.get("request_id"))
        if rid not in allowed or i.get("step") is None:
            continue
        st = int(i["step"])
        padding = e.get("activity") == ACTIVITY_PADDING
        if st not in allowed[rid]:
            if padding:
                n_pad += 1
                continue
            past.setdefault(rid, set()).add(st)
            n += 1
        elif padding:
            inside.setdefault(rid, set()).add(st)
            n_inside += 1
    named = {rid: sorted(s)[:max_report] for rid, s in sorted(past.items())}
    counts = {rid: len(s) for rid, s in sorted(past.items())}
    named_in = {rid: sorted(s)[:max_report] for rid, s in sorted(inside.items())}
    errors = []
    if past:
        errors.append("execution past the stop: " + "; ".join(
            f"{rid} committed rows at engine steps {v}{'...' if len(past[rid]) > len(v) else ''} beyond its executed prefix "
            f"(steps {min(allowed[rid]) if allowed[rid] else '-'}..{max(allowed[rid]) if allowed[rid] else '-'})" for rid, v in named.items()))
    if inside:
        errors.append("padding inside the executed prefix (the prohibited inactive fallback): " + "; ".join(
            f"{rid} {ACTIVITY_PADDING} rows at engine steps {v}{'...' if len(inside[rid]) > len(v) else ''} the Program executed" for rid, v in named_in.items()))
    return {"ok": not past and not inside, "entries_n": n, "requests": named, "steps_past_by_request": counts,
            "padding_entries_n": n_pad, "padded_record": n_pad > 0 or n_inside > 0, "padding_inside_prefix": named_in,
            "error": None if not errors else " | ".join(errors)}


def restrict_engine_steps(engine_steps: list[dict[str, Any]], request_steps: dict[str, Iterable[int]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The manifest's declared per-engine-step composition (`engine_steps[e].members`: request x rows, the Program's cap shape) restricted
    to the executed prefix of record: a member whose request did not execute at e is dropped (its rows/n_seq recomputed); a step with no
    executing request is dropped.  Returns (restricted, {"members_dropped", "steps_dropped", "steps_changed"})."""
    allowed = {str(k): {int(s) for s in v} for k, v in request_steps.items()}
    out: list[dict[str, Any]] = []
    dropped_m = dropped_s = changed = 0
    for es in engine_steps or []:
        e = int(es["engine_step"])
        mem = [m for m in (es.get("members") or []) if str(m.get("request_id")) not in allowed or e in allowed[str(m.get("request_id"))]]
        d = len(es.get("members") or []) - len(mem)
        dropped_m += d
        if not mem:
            dropped_s += 1
            continue
        if d:
            changed += 1
            out.append(dict(es, members=mem, rows=sum(int(m["rows"]) for m in mem), n_seq=len(mem)))
        else:
            out.append(es)
    return out, {"members_dropped": dropped_m, "steps_dropped": dropped_s, "steps_changed": changed}


#: what a manifest family's identities are addressed by -- the population's not-executed split (sampled_replay.split_not_executed)
#: classifies Program ROWS by their `request_id`; a batch-level row (`*` + `requests`) and a named input (the prompt ids) are outside
#: its reach, so the Commit's count for those kinds is the only one on the record (reconcile_not_executed names them, never a FAIL there)
KIND_ROW, KIND_BATCH, KIND_INPUT = "row", "batch-level", "named-input"
NAMED_INPUT_FAMILIES = ("input_token_ids",)


def family_kind(x: dict[str, Any]) -> str:
    fam = str(x.get("family") or "")
    if fam in NAMED_INPUT_FAMILIES or str(x.get("op_path") or "") in ("prompt", "input_token_ids"):
        return KIND_INPUT
    if str(x.get("request_id")) == "*" and isinstance(x.get("requests"), list) and x["requests"]:
        return KIND_BATCH
    return KIND_ROW


def new_classification() -> dict[str, Any]:
    """The accumulator `restrict_manifest_identities(stats=)` / binding.load_required_manifest fill: the identities at request-relative steps
    >= n_exec, counted by family / request / kind -- the manifest of record itself is never edited (its digest and row count stay)."""
    return {"rule": RULE, "n": 0, "by_family": {}, "by_request": {}, "kind_by_family": {}, "scope": ACTIVE_REFERENCE_ONLY}


def _classify(stats: dict[str, Any] | None, x: dict[str, Any], rid: str, n: int = 1) -> None:
    if stats is None:
        return
    fam = str(x.get("family") or x.get("op_path") or "?")
    stats["n"] += n
    stats["by_family"][fam] = stats["by_family"].get(fam, 0) + n
    stats["by_request"][rid] = stats["by_request"].get(rid, 0) + n
    stats["kind_by_family"].setdefault(fam, family_kind(x))


def restrict_manifest_identities(identities: list[dict[str, Any]], request_steps: dict[str, Iterable[int]], *,
                                 engine_step_of: Any = None, stats: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], int]:
    """The flat manifest identities (`required_manifest["identities"]`, engine_step annotated) CLASSIFIED against the executed prefix of
    record for the layouts-only coverage check: a per-request row at an engine step its request did not execute is the named class
    not-executed (left out of the requirement, counted); a batch-level row (request `*` with `requests`) keeps only the requests
    executing at that step (the class when none does).  `engine_step_of(row) -> int` overrides the row's engine_step (the caller's
    rebasing by the observed schedule).  `stats` (new_classification()) receives the counts by family / request / kind.  Nothing here
    edits the manifest of record: the caller keeps its digest and identity count beside the classification.  Returns (required, n_class)."""
    allowed = {str(k): {int(s) for s in v} for k, v in request_steps.items()}
    out: list[dict[str, Any]] = []
    dropped = 0
    for x in identities:
        rid = str(x.get("request_id"))
        e = engine_step_of(x) if engine_step_of is not None else x.get("engine_step", x.get("step"))
        if e is None or x.get("family") == "weights":
            out.append(x)
            continue
        e = int(e)
        if rid == "*" and isinstance(x.get("requests"), list) and x["requests"]:
            keep = [q for q in x["requests"] if str(q) not in allowed or e in allowed[str(q)]]
            for q in x["requests"]:
                if str(q) in allowed and e not in allowed[str(q)]:
                    _classify(stats, x, str(q))
            if not keep:
                dropped += 1
                continue
            out.append(dict(x, requests=keep) if len(keep) != len(x["requests"]) else x)
            continue
        if rid in allowed and e not in allowed[rid]:
            dropped += 1
            _classify(stats, x, rid)
            continue
        out.append(x)
    return out, dropped


def expected_cells_from_population(manifest_identities: list[dict[str, Any]] | None, population: dict[str, Any]) -> dict[str, dict[str, int]]:
    """For the families the population's ROW split does not classify, the count the population's OWN facts imply:
    per batch-level row (a plane identity, `*` + requests) the (request, plane) CELLS of listed requests at engine_step - arrive(q) >= n_exec(q);
    per named-input identity (request r, step) one cell when its request-relative step >= n_exec(r).  n_exec / arrive are read from
    population.not_executed_by_rule.by_request -- never from the Commit's own prefix, so the arithmetic is a cross-check."""
    out: dict[str, dict[str, int]] = {}
    by_req = population.get("by_request") or {}
    nex = {str(k): int(v["n_exec"]) for k, v in by_req.items() if isinstance(v, dict) and v.get("n_exec") is not None}
    arr = {str(k): int(v.get("arrive", 0) or 0) for k, v in by_req.items() if isinstance(v, dict)}
    for x in manifest_identities or []:
        kind = family_kind(x)
        if kind == KIND_ROW:
            continue
        fam = str(x.get("family") or x.get("op_path") or "?")
        d = out.setdefault(fam, {"cells": 0, "rows": 0})
        d["rows"] += 1
        if kind == KIND_BATCH:
            e = x.get("engine_step", x.get("step"))
            if e is None:
                continue
            for q in x["requests"]:
                q = str(q)
                if q in nex and int(e) - arr.get(q, 0) >= nex[q]:
                    d["cells"] += 1
        else:
            rid = str(x.get("request_id"))
            if rid not in nex:
                continue
            e = x.get("engine_step")
            rel = (int(e) - arr.get(rid, 0)) if e is not None else (int(x["step"]) if x.get("step") is not None else None)
            if rel is not None and rel >= nex[rid]:
                d["cells"] += 1
    return out


def reconcile_not_executed(commit_class: dict[str, Any] | None, population: dict[str, Any] | None, epr: dict[str, Any] | None, *,
                           manifest_identities: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """ONE mechanism, checked on the record: the Commit's classification of the manifest
    identities (binding coverage, `commit_class` = load_required_manifest's header `not_executed_by_rule`) and the population's third bucket
    (`population["not_executed_by_rule"]`: identities at request-relative steps >= n_exec leave the pool before any draw) must
    name the SAME identities.  Per request: the population's n_exec == the Commit's executed_prefix_len (`epr` = executed_prefix_of_record).
    Per ROW family (kind `row`): the Commit's count == the population's N_excluded.  A NAMED-INPUT family (a per-request family
    by the executed-prefix rule; `kind_by_family[fam] = named-input`): the Commit's count == the population's N_excluded, exactly as
    for rows.  A BATCH-LEVEL family (the plane identity stays required while any listed request executes, so identity-level
    N_excluded 0 is right; the Commit expands the plane per request): the Commit's (request, plane) CELLS == the population's
    `totals.by_family[fam].plane_cells_not_executed.cells` (== `batch_level[fam].cells_not_executed`).
    When `manifest_identities` (engine-step annotated) are given, both are ALSO checked arithmetically against the population's own
    n_exec / arrive over the manifest's rows (expected_cells_from_population) -- a population statement that disagrees with its own facts
    FAILS.  A family the population does not reach (an older population tree: no kind, no cells statement) is NAMED under `outside_population_reach` with `population_reaches` False (pending, never read as equal) and, given the
    manifest, the Commit's count must still equal the arithmetic.  Not ok -> the caller fails coverage closed with `error`."""
    out: dict[str, Any] = {"schema": "verity-sweep/not-executed-reconciliation/v3", "ok": True, "problems": [], "n_exec": {}, "by_family": {},
                           "outside_population_reach": {}, "rule": RULE, "scope": ACTIVE_REFERENCE_ONLY}
    if not isinstance(population, dict) or not population:
        out["ok"] = False
        out["problems"].append("the population carries no not_executed_by_rule block (no one-mechanism classification): nothing to reconcile against")
        return out
    if not population.get("applied", False) and (epr or {}).get("any_eos_stop"):
        out["ok"] = False
        out["problems"].append("population.not_executed_by_rule not applied on a record with an EOS-stopped request")
    by_req = population.get("by_request") or {}
    for rid, r in sorted(((epr or {}).get("requests") or {}).items()):
        want = r.get("executed_prefix_len")
        got = (by_req.get(rid) or {}).get("n_exec") if isinstance(by_req.get(rid), dict) else None
        out["n_exec"][rid] = {"commit": want, "population": got, "equal": (want is not None and got is not None and int(want) == int(got))}
        if not out["n_exec"][rid]["equal"]:
            out["ok"] = False
            out["problems"].append(f"{rid}: executed prefix of record {want} != the population's n_exec {got} (two prefixes on one record)")
    totals = ((population.get("totals") or {}).get("by_family")) or {}
    cc = commit_class or {}
    expected = expected_cells_from_population(manifest_identities, population) if manifest_identities is not None else None
    pop_kind = population.get("kind_by_family") or {}                                      # the population names each family's kind
    pop_batch = population.get("batch_level") or {}                                        # batch_level[fam].cells_not_executed
    fams = set((cc.get("by_family") or {})) | set(totals)
    for fam in sorted(fams):
        mine = int((cc.get("by_family") or {}).get(fam, 0))
        theirs = totals.get(fam) or {}
        n_ex = int(theirs.get("N_excluded", 0)) if isinstance(theirs, dict) else 0
        kind = (cc.get("kind_by_family") or {}).get(fam, KIND_ROW)
        cells = int(expected[fam]["cells"]) if expected is not None and fam in expected else None
        if kind == KIND_ROW:
            out["by_family"][fam] = {"commit": mine, "population": n_ex, "equal": mine == n_ex}
            if mine != n_ex:
                out["ok"] = False
                out["problems"].append(f"{fam}: the Commit classified {mine} identities not-executed-by-rule, the population excluded {n_ex}")
            continue
        unit = "cells" if kind == KIND_BATCH else "identities"
        if kind == KIND_BATCH:
            # the population's statement of the (plane, request) cells past the stop -- the unit the Commit counts
            pc = theirs.get("plane_cells_not_executed") if isinstance(theirs, dict) else None
            pop_val = (int(pc["cells"]) if isinstance(pc, dict) and pc.get("cells") is not None
                       else int(pop_batch[fam]["cells_not_executed"]) if isinstance(pop_batch.get(fam), dict) and pop_batch[fam].get("cells_not_executed") is not None
                       else None)
            reaches = pop_val is not None
        else:
            # a per-request family by the executed-prefix rule -> N_excluded, exactly as for rows, once the population names it
            reaches = (pop_kind.get(fam) is not None and pop_kind.get(fam) != "row") or (cells is not None and n_ex == cells and n_ex > 0)
            pop_val = n_ex if reaches else None
        if reaches:
            ent = {"commit": mine, "population": pop_val, "equal": mine == pop_val, "kind": kind, "unit": unit,
                   **({"population_N_excluded": n_ex} if kind == KIND_BATCH else {})}
            if mine != pop_val:
                out["ok"] = False
                out["problems"].append(f"{fam} ({kind}): the Commit classified {mine} not-executed-by-rule {unit}, the population excluded {pop_val}")
            if cells is not None:
                ent["expected_from_population_facts"] = cells                  # the population's own n_exec / arrive over the manifest's rows
                if pop_val != cells:
                    out["ok"] = False
                    out["problems"].append(f"{fam} ({kind}): the population states {pop_val} not-executed {unit}, its own n_exec implies {cells}")
            out["by_family"][fam] = ent
            continue
        ent = {"kind": kind, "commit": mine, "population_N_excluded": n_ex, "population_reaches": False,
               "pending": f"this population tree does not reach {kind} families (no kind_by_family / cells statement)",
               "why": "the population split classifies Program rows by request_id; this family's identities are not rows of one request"}
        if cells is not None:
            ent["expected_from_population_facts"] = cells
            ent["equal"] = mine == cells
            if mine != cells:
                out["ok"] = False
                out["problems"].append(f"{fam} ({kind}): the Commit classified {mine} not-executed-by-rule {unit}, the population's n_exec implies {cells}")
        out["outside_population_reach"][fam] = ent
    out["total"] = (all(e.get("equal", False) for e in out["outside_population_reach"].values()) if out["outside_population_reach"] else True) and out["ok"]
    return out
