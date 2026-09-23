"""[R17-1 / F-moe-r17-03 (moe M-0283)] the EXECUTED PREFIX OF RECORD of every request -- the one fact the Commit's consumers count on.

BRIEF s4 / COORD R17-1: a request has TWO lengths.  `served_output_len` a_r = the tokens the driver served (its EOS included);
`executed_prefix_len` = the engine rows that ran for it = the cap when the request generated its cap (length stop), a_r + LAG when it
stopped by EOS before the cap -- the LAG row(s) after the EOS are active internal computation (their VUs are required) whose sampled token
is not served.  LAG is the workload Program's DECLARED specialization parameter (Workload_v1{..., lag}); the served tokens are the driver's;
NOTHING is read from the trace.  Positions >= executed_prefix_len are programmed-but-unexecuted: not required of the record, and a committed
row there is 'execution past the stop' (the caller checks the binding map against `request_steps`).

Fail closed, by name: an early stop (a_r < cap) with no declared stop rule, no declared LAG, or a last served token that is not a declared
EOS id (F-pm-24) is a problem -- the population then stays the cap's (nothing is accepted) and `ok` is False.  Never default LAG 0 or 1.

The first record: OLMoE B2 i32/o32 mixedstop (cp-moe 2026-09-20 18:45Z, lane e32550951): r0 served 11 / executed 12 (EOS 50279 at output
index 10, LAG 1), r1 32 / 32 (cap) -- G6 PASS with the lag exercised; the Commit's population, placed over every request's cap, required
r0's 20 unexecuted steps (bound 13808/19988) and failed by name.

COORD R17-27 (Daniel, 19:22Z) puts this in a box -- the ACTIVE-REFERENCE table only, on three conditions `link_to_match_account` checks:
(a) n_exec is determined from the properly LINKED serving transcript: the Commit's own served tokens count only because they EQUAL the Match
record's served output (capture/tokens.json of the same attempt) and the Match account (GM-01 G6: PASS, the same two lengths per request)
-- "the committer says EOS happened here" is never sufficient by itself; (b) the lag rule is attested: declared == the engine's read-back ==
the record's tokens.json; (c) any actual execution past the boundary FAILS (`rows_past_the_stop`).  NOT-EXECUTED-BY-RULE is an
active-reference implementation classification only.  In the lifted fixed Program the corresponding positions are REQUIRED logical values
assigned the canonical inactive representation tau(e) = bot; there is no not-required-because-not-executed category.

PROTOCOL SUMMARY (Commit stage: the executed prefix of record -- how far each request actually ran)
  Purpose: per request, derive the two lengths every Commit consumer counts on -- served_output_len (tokens served, EOS included) and
    executed_prefix_len (cap on a length stop; served + LAG on an EOS stop) -- from the served tokens + the Program's DECLARED stop
    rule (eos ids, lag), never from the trace; then accept it only when LINKED to the Match account of the same attempt.
  Inputs: served tokens per request, caps, `eos_of_record` + `lag` (global_program.stop_rule / Workload_v1.lag), the Match account
    (match/capture/tokens.json + global_match.json G6 via `load_match_account`), the binding map (for `rows_past_the_stop`).
  Outputs: dicts consumed by commit_delta / binding / sampled_replay: `executed_prefix_of_record` {requests{served_output_len,
    executed_prefix_len, cap, eos_stop}, ok, problems, any_eos_stop, lag}, `request_steps_of_record`, `rows_past_the_stop`,
    `link_to_match_account` {ok, required, problems, tokens_equal, g6}, `executed_prefix_facts` (the named facts sampled_replay takes).
  Invariants: LAG is never defaulted to 0 or 1 -- an early stop with no declared EOS set / LAG, a non-EOS last token, or served > cap
    is a problem and the population stays the cap's; a prefix reduced from its cap on the committer's own count alone is never
    accepted (R17-27 (a)); the lag must agree on every leg (declared == tokens.json == G6); a committed row past the prefix fails by name.
  Protocol spec: Notion -> "vLLM integration — protocol spec (Build · Match · Commit)".
"""
from __future__ import annotations

from typing import Any, Iterable

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
        # [rev F-r17-14] the Program's rule stops at the FIRST declared EOS: a served list that carries an EOS at index k and more tokens
        # after it was served PAST the stop -- FAIL by name, whatever the last token is (served [EOS, x, EOS] is not an EOS stop at 3)
        first_eos = next((k for k, t in enumerate(toks) if eos_set is not None and int(t) in eos_set), None)
        rec["first_eos_index"] = first_eos
        if first_eos is not None and first_eos < a - 1:
            rec["why"] = (f"served past an EOS at index {first_eos}: served token {int(toks[first_eos])} is a declared EOS id but {a - 1 - first_eos} "
                          f"more token(s) were served after it (served {a}, cap {cap}); the Program's rule stops at the FIRST EOS -- served_output_len "
                          f"would be {first_eos + 1} (F-r17-14)")
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
                              f"verified, fail closed (F-pm-24)")
                problems.append(f"{rid}: {rec['why']}")
            elif rec["last_served_token"] not in eos_set:
                rec["why"] = (f"served {a} of its cap {cap} but its last served token {rec['last_served_token']} is not a declared EOS id "
                              f"{sorted(eos_set)[:8]} (F-pm-24)")
                problems.append(f"{rid}: {rec['why']}")
            elif lag_v is None:
                rec["why"] = (f"stopped by EOS at served {a} of its cap {cap} but the Program declares no LAG (or the engine does not attest the "
                              f"declared scheduler): the executed prefix cannot be placed (R17-1) -- never default 0 or 1")
                problems.append(f"{rid}: {rec['why']}")
            elif lag_v not in (0, 1):
                rec["why"] = f"declared LAG {lag_v} is not in {{0, 1}} (R17-1)"
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


def load_match_account(match_dir: str | None) -> dict[str, Any]:
    """The Match attempt's serving transcript + account beside the Commit: `<match_dir>/capture/tokens.json` (the driver's served tokens,
    per-request two lengths, the document-level lag attestation -- bench.m1_capture) and `<match_dir>/../global_match.json` (GM-01's
    verdict, G6's lag block and lengths_by_request).  Missing files are reported, never invented."""
    import json
    import os
    out: dict[str, Any] = {"match_dir": match_dir, "tokens": None, "tokens_path": None, "global_match": None, "global_match_path": None, "notes": []}
    if not match_dir:
        out["notes"].append("no --match-dir: the Match record of this attempt is not beside the Commit")
        return out
    tp = os.path.join(match_dir, "capture", "tokens.json")
    gp = os.path.join(os.path.dirname(os.path.abspath(match_dir.rstrip("/"))), "global_match.json")
    for key, path in (("tokens", tp), ("global_match", gp)):
        out[f"{key}_path"] = path
        if not os.path.isfile(path):
            out["notes"].append(f"{os.path.relpath(path, match_dir)} not found")
            continue
        try:
            import hashlib
            raw = open(path, "rb").read()
            out[key] = json.loads(raw)
            out[f"{key}_sha256"] = hashlib.sha256(raw).hexdigest()
        except Exception as e:  # noqa: BLE001
            out["notes"].append(f"{os.path.relpath(path, match_dir)} unreadable: {e!r}"[:200])
    return out


def link_to_match_account(epr: dict[str, Any], account: dict[str, Any], *, order: dict[str, str], declared_lag: int | None,
                          served_tokens: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """[COORD R17-27 (a)+(b)] the executed prefix of record is ACCEPTED only when the Commit's own served tokens are linked to the served
    output and the Match account of the SAME attempt: for every request, the Match record's tokens.json generated ids == the Commit's
    served tokens (tokens_equal), its served_output_len / executed_prefix_len == this rule's; GM-01 (global_match.json, when present)
    PASSED and its G6 lengths_by_request agree; the lag rule is attested on every leg (Program declared == the Match record's tokens.json
    lag == G6's declared / record_attests; no variable lag).  `order`: {workload request_id: r<i>}.

    A request whose prefix was REDUCED from its cap (an EOS stop) with no linkable Match account is a problem: the committer's count alone
    is never sufficient (R17-27 (a)).  With no reduction anywhere (every request ran its cap) a missing account is recorded, not a
    problem -- nothing was placed on the committer's word.  `served_tokens` {r<i>: [the Commit's served ids]}: when given, tokens_equal
    is the FULL generated_token_ids list equality (rev S-r17-23 (c)); without it, length + last token.  Returns {"ok", "required",
    "problems", "tokens_equal", "g6", ...}."""
    problems: list[str] = []
    reduced = sorted(r for r, v in (epr.get("requests") or {}).items() if v.get("eos_stop"))
    required = bool(reduced)
    res: dict[str, Any] = {"required": required, "reduced_requests": reduced, "tokens_json": None, "global_match": None, "notes": list(account.get("notes") or []),
                           "rule": "R17-27 (a): n_exec from the LINKED serving transcript (Commit served == Match tokens.json == G6 account); "
                                   "(b) lag attested declared == tokens.json == G6; the committer's own count is never sufficient by itself"}
    tok = account.get("tokens")
    gm = account.get("global_match")
    if tok is None:
        if required:
            problems.append(f"executed prefix reduced from the cap for {reduced} on the committer's own served count alone: the Match record's "
                            f"capture/tokens.json is not beside the Commit ({'; '.join(res['notes']) or 'no account'}) -- never sufficient by itself (R17-27 (a))")
    else:
        by_rid: dict[str, dict[str, Any]] = {}
        for e in tok.get("requests") or []:
            r = order.get(str(e.get("request_id")))
            if r is not None:
                by_rid[r] = e
        tj_lag = (tok.get("lag") or {}).get("value") if isinstance(tok.get("lag"), dict) else None
        eq: dict[str, bool | None] = {}
        for rid, rec in (epr.get("requests") or {}).items():
            e = by_rid.get(rid)
            if e is None:
                eq[rid] = None
                problems.append(f"{rid}: not in the Match record's tokens.json (request ids {sorted(by_rid)[:8]}): the serving transcript does not cover it")
                continue
            m_served = [int(t) for t in (e.get("generated_token_ids") or [])]
            c_served = rec.get("served_output_len")
            c_toks = None if served_tokens is None else served_tokens.get(rid)
            if c_toks is not None:                                                  # rev S-r17-23 (c): the FULL list, not length + last
                same = [int(t) for t in c_toks] == m_served
                _first = next((i for i, (a, b) in enumerate(zip(c_toks, m_served)) if int(a) != int(b)), min(len(c_toks), len(m_served)))
            else:
                same = (c_served is not None) and (len(m_served) == int(c_served)) and (rec.get("last_served_token") is None or not m_served or int(m_served[-1]) == int(rec["last_served_token"]))
                _first = None
            eq[rid] = bool(same)
            if not same:
                problems.append(f"{rid}: the Commit served {c_served} tokens (last {rec.get('last_served_token')}) but the Match record's tokens.json served "
                                f"{len(m_served)} (last {m_served[-1] if m_served else None})"
                                + (f", first differing position {_first}" if _first is not None else "") + ": not one serving transcript (R17-27 (a))")
                continue
            m_exec = e.get("executed_prefix_len")
            m_srv = e.get("served_output_len")
            if m_srv is not None and int(m_srv) != int(c_served):
                problems.append(f"{rid}: tokens.json served_output_len {m_srv} != the Commit's {c_served} (R17-27 (a))")
            if m_exec is None and rec.get("eos_stop"):
                problems.append(f"{rid}: tokens.json executed_prefix_len is null (EOS stop under an unattested lag) while this rule places {rec['executed_prefix_len']} (R17-27 (b))")
            elif m_exec is not None and int(m_exec) != int(rec["executed_prefix_len"]):
                problems.append(f"{rid}: tokens.json executed_prefix_len {m_exec} != this rule's {rec['executed_prefix_len']} (served {c_served}, cap {rec['cap']}, LAG {epr.get('lag')}) (R17-27 (a))")
        res["tokens_json"] = {"path": account.get("tokens_path"), "sha256": account.get("tokens_sha256"), "lag": tj_lag,
                              "async_scheduling": (tok.get("lag") or {}).get("async_scheduling") if isinstance(tok.get("lag"), dict) else None,
                              "tokens_equal": eq, "tokens_equal_rule": ("full generated_token_ids list" if served_tokens is not None else "length + last token"),
                              "requests_n": len(by_rid),
                              # a_r AS THE MATCH RECORD SERVED IT, per request (the fact reqval's not-executed class reads; R17-27 (a))
                              "served_by_request": {r: len(e.get("generated_token_ids") or []) for r, e in by_rid.items()},
                              "executed_by_request": {r: e.get("executed_prefix_len") for r, e in by_rid.items()}}
        res["tokens_equal"] = all(v is True for v in eq.values()) if eq else None
        if declared_lag is not None and (tj_lag is None or int(tj_lag) != int(declared_lag)):
            problems.append(f"lag not attested on the record: the Program declares LAG {declared_lag}, the Match record's tokens.json attests {tj_lag!r} (R17-27 (b))")
    if gm is None:
        if required and tok is not None:
            res["notes"].append("global_match.json not beside the Match record: the G6 account leg is missing")
            problems.append(f"executed prefix reduced from the cap for {reduced}: GM-01's account (global_match.json) is not beside the Match record -- the "
                            f"Commit's count is linked to the served output only, not to the Match account (R17-27 (a))")
    else:
        g6 = next((c for c in (gm.get("checks") or []) if isinstance(c, dict) and c.get("id") == "G6"), None)
        lens = (g6 or {}).get("lengths_by_request") or {}
        gl = (g6 or {}).get("lag") or gm.get("lag") or {}
        res["global_match"] = {"path": account.get("global_match_path"), "sha256": account.get("global_match_sha256"), "verdict": gm.get("verdict"),
                               "failed_checks": gm.get("failed_checks"), "g6_status": (g6 or {}).get("status"),
                               "lag": {k: gl.get(k) for k in ("declared", "record_attests", "tokens_json_lag", "exercised", "variable_lag")}}
        if gm.get("verdict") != "PASS":
            problems.append(f"GM-01 verdict {gm.get('verdict')!r} (failed {gm.get('failed_checks')}): the Match account this prefix must be linked to did not pass (R17-27 (a))")
        if g6 is None:
            problems.append("global_match.json has no G6 check: no chronology account to link the executed prefix to (R17-27 (a))")
        else:
            if gl.get("variable_lag"):
                problems.append("G6 reports VARIABLE LAG on this record: no constant-lag placement is linkable (R17-1 / R17-27)")
            if declared_lag is not None:
                for leg in ("declared", "record_attests"):
                    if gl.get(leg) is None or int(gl[leg]) != int(declared_lag):
                        problems.append(f"G6 lag.{leg} {gl.get(leg)!r} != the Program's declared LAG {declared_lag} (R17-27 (b))")
            # recorder request id -> r<i> by the workload request id (G6's lengths are keyed by the recorder's id)
            g6_by_r: dict[str, dict[str, Any]] = {}
            for rec_rid, L in lens.items():
                e = next((x for x in (gm.get("requests") or []) if isinstance(x, dict) and str(x.get("request_id")) == str(rec_rid)), None)
                r = order.get(str((e or {}).get("workload_request_id"))) if e else None
                if r is None and e is not None and e.get("index") is not None:
                    r = f"r{int(e['index'])}"
                if r is not None:
                    g6_by_r[r] = L
            for rid, rec in (epr.get("requests") or {}).items():
                L = g6_by_r.get(rid)
                if L is None:
                    problems.append(f"{rid}: no G6 lengths_by_request entry (G6 covers {sorted(g6_by_r)[:8]}): not in the Match account (R17-27 (a))")
                    continue
                if rec.get("served_output_len") is not None and L.get("served_output_len") is not None and int(L["served_output_len"]) != int(rec["served_output_len"]):
                    problems.append(f"{rid}: G6 served_output_len {L['served_output_len']} != the Commit's {rec['served_output_len']} (R17-27 (a))")
                if L.get("executed_prefix_len") is not None and int(L["executed_prefix_len"]) != int(rec["executed_prefix_len"]):
                    problems.append(f"{rid}: G6 executed_prefix_len {L['executed_prefix_len']} != this rule's {rec['executed_prefix_len']} (R17-27 (a))")
            res["global_match"]["requests_linked"] = sorted(g6_by_r)
            res["global_match"]["lengths_by_request"] = {r: {k: L.get(k) for k in ("served_output_len", "executed_prefix_len", "cap", "cap_exhausted")} for r, L in g6_by_r.items()}
    # (b) on every leg, one boolean the C2 population's class reads: declared == tokens.json == G6.declared == G6.record_attests
    _legs = [declared_lag, (res.get("tokens_json") or {}).get("lag"),
             ((res.get("global_match") or {}).get("lag") or {}).get("declared"), ((res.get("global_match") or {}).get("lag") or {}).get("record_attests")]
    res["lag_attested"] = (declared_lag is not None and tok is not None and gm is not None and all(v is not None and int(v) == int(declared_lag) for v in _legs)
                           and not ((res.get("global_match") or {}).get("lag") or {}).get("variable_lag"))
    res["problems"] = problems
    res["ok"] = not problems
    res["classification"] = ACTIVE_REFERENCE_ONLY
    return res


def executed_prefix_facts(epr: dict[str, Any], binding: dict[str, Any], linkage: dict[str, Any] | None, arrivals: dict[str, int],
                           stop: dict[str, Any], served_tokens: dict[str, list[int]] | None = None) -> dict[str, dict[str, Any]]:
    """[COORD R17-26/R17-27; reqval M-0343, revb M-0349 n1/n2] the per-request FACTS OF RECORD reqval's `sampled_replay.executed_prefix_of_record`
    normalises into the not-executed(R17-1) class -- every one a named record fact, never a literal: `a_r` READ from the Match record
    (binding.coverage.match_account_link: tokens.json served_by_request, else G6 lengths_by_request; `a_r_source` names the file + sha256;
    with NO Match record a_r is the Commit's own served length, labelled so, and `linked` is False -> the class faults by name);
    `served_len_commit` = this run's own served length; `linked` = the Match-account link held (Commit served == Match served per request,
    GM-01 PASS, G6 agrees) AND the C2 boundary linkage passed; `lag_attested` = declared == tokens.json == G6 declared == G6 record_attests;
    `eos_declared` = the served token a_r-1 is a declared EOS id (F-pm-24); `rows_past_stop` = coverage's execution-past-the-stop count.
    [reqval M-0585 / revb S-r17b-20 R3] with `served_tokens` (this run's own served ids per request) the facts also carry `served` (the list)
    and `eos_ids` (the Program's declared set) so a consumer that derives the class from Continuation_v1's recurrence (r_t) does so from the
    record's own list -- ONE code path on Build, Match and Commit; the caller passes only the keys the consumer's signature accepts."""
    cov = (binding or {}).get("coverage") or {}
    link = cov.get("match_account_link") or {}
    tj = link.get("tokens_json") or {}
    g6 = link.get("global_match") or {}
    # [r19-int-21, F-r19-int-19] the account legs that are NOT beside the Commit, named: #39 (qwen25-15b i4096, l40s4) had its match/capture/
    # moved to /vault to free the disk (F-r19-int-14) while its Commit was queued; tokens.json (a_r per request, the lag attestation) went
    # with it, the link fell to the G6 leg for a_r and the class faulted 'a_r is not linked (linked=False)' + 'LAG rule not attested' after
    # 47 min -- a MISSING INPUT read as a verdict on the Program.  The fact names the missing leg(s); the consumer faults on it by name.
    account_missing = [leg for leg, present in (("tokens.json", bool(tj)), ("global_match.json", bool(g6))) if not present]
    past = (cov.get("rows_past_the_stop") or {}).get("steps_past_by_request") or {}
    eos_set = {int(x) for x in (stop.get("eos") or [])} if stop.get("eos") is not None else None
    decl_lag = stop.get("declared_lag")
    lk_ok = bool(linkage and linkage.get("result") is True)
    out: dict[str, dict[str, Any]] = {}
    for rid, rec in (epr.get("requests") or {}).items():
        served_commit = rec.get("served_output_len")
        m_served = (tj.get("served_by_request") or {}).get(rid)
        g_len = ((g6.get("lengths_by_request") or {}).get(rid) or {}).get("served_output_len")
        if m_served is not None:
            a_r, src = int(m_served), f"match record {tj.get('path')} (sha256 {str(tj.get('sha256'))[:16]}): tokens.json generated_token_ids"
        elif g_len is not None:
            a_r, src = int(g_len), f"match record {g6.get('path')} (sha256 {str(g6.get('sha256'))[:16]}): G6 lengths_by_request.served_output_len"
        else:
            a_r, src = int(served_commit if served_commit is not None else rec["cap"]), "the Commit's OWN served tokens (no Match record beside the Commit: NOT linked)"
        if account_missing:
            src += f"; Match account leg(s) NOT beside the Commit: {', '.join(account_missing)}"
        linked = bool(link.get("ok") is True and (tj.get("tokens_equal") or {}).get(rid) is True and g6.get("verdict") == "PASS" and lk_ok)
        out[rid] = {"a_r": a_r, "a_r_source": src, "lag": (int(decl_lag) if decl_lag is not None else -1), "cap": int(rec["cap"]), "arrive": int(arrivals.get(rid, 0)),
                    "eos_declared": (None if rec.get("last_served_token") is None or eos_set is None else (int(rec["last_served_token"]) in eos_set)),
                    "served_len_commit": served_commit, "linked": linked, "lag_attested": bool(link.get("lag_attested")),
                    "rows_past_stop": int(past.get(rid, 0)), "account_missing": list(account_missing)}
        if served_tokens is not None and served_tokens.get(rid) is not None:
            out[rid]["served"] = [int(t) for t in served_tokens[rid]]
            out[rid]["eos_ids"] = (sorted(eos_set) if eos_set is not None else None)
    return out


def facts_accepted_by(fn, facts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The facts a consumer's signature accepts (keyword-only parameters): a tree whose `executed_prefix_of_record` predates `served=` /
    `eos_ids=` (reqval a8f110f5f) still receives the arithmetic's facts; a newer one receives the served list too.  Never a TypeError."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return facts
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return facts
    keep = set(params)
    return {rid: {k: v for k, v in f.items() if k in keep} for rid, f in facts.items()}


#: [R17-42, reqval M-1318 (1)] the activity a PADDED record's binding map (`padded-binding-map/1`, pad's padding_steps) stamps on every entry:
#: `bot-by-enablement` = the REQUIRED padding of a not-executed row (its value is tau(e) = bot by rule, NOT_EXECUTED_SCOPE verbatim), `active` =
#: an executed row's committed value, `registered-input` = a named input.  An UNPADDED record's entries carry no `activity` (= active).
ACTIVITY_PADDING = "bot-by-enablement"


def rows_past_the_stop(bmap: dict[str, Any], request_steps: dict[str, Iterable[int]], *, max_report: int = 8) -> dict[str, Any]:
    """Committed leaf entries of a declared request at an engine step OUTSIDE its executed prefix of record = 'execution past the stop'
    (a row the Program says was never executed carries committed values): named per request with the steps, FAIL by name.  Entries of
    other request ids (`*`, unmapped) are not this check's.

    [R17-42 / reqval M-1318 (1)] On a PADDED record the rows past the stop are PRESENT by construction -- every not-executed row is a
    required value assigned bot -- and each such entry says so (`activity = bot-by-enablement`): those are the padding, counted
    (`padding_entries_n`), never execution past the stop.  An entry there that is `active` (or has no activity field: an unpadded record)
    IS execution past the stop, exactly as before.  The converse is named too: a `bot-by-enablement` entry INSIDE the executed prefix is a
    row the Program says was executed and the record says was not -- the prohibited inactive fallback (R17-42), FAIL by name."""
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
        errors.append("padding inside the executed prefix (inactive fallback, R17-42): " + "; ".join(
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


#: what a manifest family's identities are addressed by -- the population's not-executed split (reqval: sampled_replay.split_not_executed)
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
    not-executed(R17-1) (left out of the requirement, counted); a batch-level row (request `*` with `requests`) keeps only the requests
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
    """[revb M-0452 (i)(ii)] For the families the population's ROW split does not classify, the count the population's OWN facts imply:
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
    """[COORD R17-26/R17-27; rel M-0410 (a); revb M-0452] ONE mechanism, checked on the record: the Commit's classification of the manifest
    identities (binding coverage, `commit_class` = load_required_manifest's header `not_executed_by_rule`) and the population's third bucket
    (`population["not_executed_by_rule"]`, reqval: identities at request-relative steps >= n_exec leave the pool before any draw) must
    name the SAME identities.  Per request: the population's n_exec == the Commit's executed_prefix_len (`epr` = executed_prefix_of_record).
    Per ROW family (kind `row`): the Commit's count == the population's N_excluded.  A NAMED-INPUT family (revb (i): a per-request family
    by R17-1's rule; reqval 699bf6ee2 `kind_by_family[fam] = named-input`): the Commit's count == the population's N_excluded, exactly as
    for rows.  A BATCH-LEVEL family (revb (ii): the plane identity stays required while any listed request executes, so identity-level
    N_excluded 0 is right; the Commit expands the plane per request): the Commit's (request, plane) CELLS == the population's
    `totals.by_family[fam].plane_cells_not_executed.cells` (== `batch_level[fam].cells_not_executed`, reqval 8eeff13eb / d13ea5153).
    When `manifest_identities` (engine-step annotated) are given, both are ALSO checked arithmetically against the population's own
    n_exec / arrive over the manifest's rows (expected_cells_from_population) -- a population statement that disagrees with its own facts
    FAILS.  A family the population does not reach (an older population tree: no kind, no cells statement -- F-moe-r17-05 before
    699bf6ee2) is NAMED under `outside_population_reach` with `population_reaches` False (pending, never read as equal) and, given the
    manifest, the Commit's count must still equal the arithmetic.  Not ok -> the caller fails coverage closed with `error`."""
    out: dict[str, Any] = {"schema": "verity-sweep/not-executed-reconciliation/v3", "ok": True, "problems": [], "n_exec": {}, "by_family": {},
                           "outside_population_reach": {}, "rule": RULE, "scope": ACTIVE_REFERENCE_ONLY}
    if not isinstance(population, dict) or not population:
        out["ok"] = False
        out["problems"].append("the population carries no not_executed_by_rule block (reqval R17-26/R17-27 mechanism absent): nothing to reconcile against")
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
    pop_kind = population.get("kind_by_family") or {}                                      # reqval 699bf6ee2: the population names each family's kind
    pop_batch = population.get("batch_level") or {}                                        # reqval 8eeff13eb: batch_level[fam].cells_not_executed
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
                out["problems"].append(f"{fam}: the Commit classified {mine} identities not-executed(R17-1), the population excluded {n_ex}")
            continue
        unit = "cells" if kind == KIND_BATCH else "identities"
        if kind == KIND_BATCH:
            # revb (ii) / reqval d13ea5153: the population's statement of the (plane, request) cells past the stop -- the unit the Commit counts
            pc = theirs.get("plane_cells_not_executed") if isinstance(theirs, dict) else None
            pop_val = (int(pc["cells"]) if isinstance(pc, dict) and pc.get("cells") is not None
                       else int(pop_batch[fam]["cells_not_executed"]) if isinstance(pop_batch.get(fam), dict) and pop_batch[fam].get("cells_not_executed") is not None
                       else None)
            reaches = pop_val is not None
        else:
            # revb (i) / reqval 699bf6ee2: a per-request family by R17-1's rule -> N_excluded, exactly as for rows, once the population names it
            reaches = (pop_kind.get(fam) is not None and pop_kind.get(fam) != "row") or (cells is not None and n_ex == cells and n_ex > 0)
            pop_val = n_ex if reaches else None
        if reaches:
            ent = {"commit": mine, "population": pop_val, "equal": mine == pop_val, "kind": kind, "unit": unit,
                   **({"population_N_excluded": n_ex} if kind == KIND_BATCH else {})}
            if mine != pop_val:
                out["ok"] = False
                out["problems"].append(f"{fam} ({kind}): the Commit classified {mine} not-executed(R17-1) {unit}, the population excluded {pop_val}")
            if cells is not None:
                ent["expected_from_population_facts"] = cells                  # the population's own n_exec / arrive over the manifest's rows
                if pop_val != cells:
                    out["ok"] = False
                    out["problems"].append(f"{fam} ({kind}): the population states {pop_val} not-executed {unit}, its own n_exec implies {cells}")
            out["by_family"][fam] = ent
            continue
        ent = {"kind": kind, "commit": mine, "population_N_excluded": n_ex, "population_reaches": False,
               "pending": f"F-moe-r17-05: this population tree does not reach {kind} families (no kind_by_family / cells statement; reqval 699bf6ee2+)",
               "why": "the population split classifies Program rows by request_id; this family's identities are not rows of one request"}
        if cells is not None:
            ent["expected_from_population_facts"] = cells
            ent["equal"] = mine == cells
            if mine != cells:
                out["ok"] = False
                out["problems"].append(f"{fam} ({kind}): the Commit classified {mine} not-executed(R17-1) {unit}, the population's n_exec implies {cells}")
        out["outside_population_reach"][fam] = ent
    out["total"] = (all(e.get("equal", False) for e in out["outside_population_reach"].values()) if out["outside_population_reach"] else True) and out["ok"]
    return out
