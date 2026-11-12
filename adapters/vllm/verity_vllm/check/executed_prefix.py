"""The executed prefix's link to the Match account: conditions (a) and (b) of the ACTIVE-REFERENCE classification that
`commit.executed_prefix` states (the rule, the two lengths and the three conditions are its docstring).

`load_match_account` reads the Match attempt's serving transcript and GM-01's account beside the Commit; `link_to_match_account`
accepts a prefix reduced from its cap only when the Commit's served tokens equal the Match record's, G6 passed with the same two lengths
per request, and the lag agrees on every leg.  A config run has no Match stage: its link is `check.commit_tokens`.
"""
from __future__ import annotations

from typing import Any

from verity_vllm.commit.executed_prefix import ACTIVE_REFERENCE_ONLY
from verity_vllm.program.sampling_event import circuit_token_id
from verity_vllm.records.global_match import GlobalMatch


def load_match_account(match_dir: str | None) -> dict[str, Any]:
    """The Match attempt's serving transcript + account beside the Commit: `<match_dir>/capture/tokens.json` (the driver's served tokens,
    per-request two lengths, the document-level lag attestation -- bench.m1_capture) and `<match_dir>/../global_match.json` (GM-01's
    verdict, G6's lag block and lengths_by_request, as `records.global_match.GlobalMatch`).  Missing or unreadable files are reported,
    never invented."""
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
            out[key] = GlobalMatch.of(json.loads(raw), path) if key == "global_match" else json.loads(raw)
            out[f"{key}_sha256"] = hashlib.sha256(raw).hexdigest()
        except Exception as e:  # noqa: BLE001
            out["notes"].append(f"{os.path.relpath(path, match_dir)} unreadable: {e!r}"[:200])
    return out


def link_to_match_account(epr: dict[str, Any], account: dict[str, Any], *, order: dict[str, str], declared_lag: int | None,
                          served_tokens: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """Conditions (a) and (b) of the module docstring: the executed prefix of record is ACCEPTED only when the Commit's own served tokens are linked to the served
    output and the Match account of the SAME attempt: for every request, the Match record's tokens.json generated ids == the Commit's
    served tokens (tokens_equal), its served_output_len / executed_prefix_len == this rule's; GM-01 (global_match.json, when present)
    PASSED and its G6 lengths_by_request agree; the lag rule is attested on every leg (Program declared == the Match record's tokens.json
    lag == G6's declared / record_attests; no variable lag).  `order`: {workload request_id: r<i>}.

    A request whose prefix was REDUCED from its cap (an EOS stop) with no linkable Match account is a problem: the committer's count alone
    is never sufficient (condition (a)).  With no reduction anywhere (every request ran its cap) a missing account is recorded, not a
    problem -- nothing was placed on the committer's word.  `served_tokens` {r<i>: [the Commit's served ids]}: when given, tokens_equal
    is the FULL generated_token_ids list equality; without it, length + last token.  Returns {"ok", "required",
    "problems", "tokens_equal", "g6", ...}."""
    problems: list[str] = []
    reduced = sorted(r for r, v in (epr.get("requests") or {}).items() if v.get("eos_stop"))
    required = bool(reduced)
    res: dict[str, Any] = {"required": required, "reduced_requests": reduced, "tokens_json": None, "global_match": None, "notes": list(account.get("notes") or []),
                           "match_dir": account.get("match_dir"),
                           "rule": "(a) n_exec from the LINKED serving transcript (Commit served == Match tokens.json == G6 account); "
                                   "(b) lag attested declared == tokens.json == G6; the committer's own count is never sufficient by itself"}
    tok = account.get("tokens")
    gm = account.get("global_match")
    if tok is None:
        if required:
            problems.append(f"executed prefix reduced from the cap for {reduced} on the committer's own served count alone: the Match record's "
                            f"capture/tokens.json is not beside the Commit ({'; '.join(res['notes']) or 'no account'}) -- never sufficient by itself (executed-prefix condition (a))")
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
            try:
                m_served = [circuit_token_id(t) for t in (e.get("generated_token_ids") or [])]
            except ValueError as err:
                eq[rid] = False
                problems.append(f"{rid}: the Match record's tokens.json served an id that is no circuit's: {err}")
                continue
            c_served = rec.get("served_output_len")
            c_toks = None if served_tokens is None else served_tokens.get(rid)
            if c_toks is not None:                                                  # The FULL list, not length + last
                same = [int(t) for t in c_toks] == m_served
                _first = next((i for i, (a, b) in enumerate(zip(c_toks, m_served)) if int(a) != int(b)), min(len(c_toks), len(m_served)))
            else:
                same = (c_served is not None) and (len(m_served) == int(c_served)) and (rec.get("last_served_token") is None or not m_served or int(m_served[-1]) == int(rec["last_served_token"]))
                _first = None
            eq[rid] = bool(same)
            if not same:
                problems.append(f"{rid}: the Commit served {c_served} tokens (last {rec.get('last_served_token')}) but the Match record's tokens.json served "
                                f"{len(m_served)} (last {m_served[-1] if m_served else None})"
                                + (f", first differing position {_first}" if _first is not None else "") + ": not one serving transcript (executed-prefix condition (a))")
                continue
            m_exec = e.get("executed_prefix_len")
            m_srv = e.get("served_output_len")
            if m_srv is not None and int(m_srv) != int(c_served):
                problems.append(f"{rid}: tokens.json served_output_len {m_srv} != the Commit's {c_served} (executed-prefix condition (a))")
            if m_exec is None and rec.get("eos_stop"):
                problems.append(f"{rid}: tokens.json executed_prefix_len is null (EOS stop under an unattested lag) while this rule places {rec['executed_prefix_len']} (executed-prefix condition (b))")
            elif m_exec is not None and int(m_exec) != int(rec["executed_prefix_len"]):
                problems.append(f"{rid}: tokens.json executed_prefix_len {m_exec} != this rule's {rec['executed_prefix_len']} (served {c_served}, cap {rec['cap']}, LAG {epr.get('lag')}) (executed-prefix condition (a))")
        res["tokens_json"] = {"path": account.get("tokens_path"), "sha256": account.get("tokens_sha256"), "lag": tj_lag,
                              "async_scheduling": (tok.get("lag") or {}).get("async_scheduling") if isinstance(tok.get("lag"), dict) else None,
                              "tokens_equal": eq, "tokens_equal_rule": ("full generated_token_ids list" if served_tokens is not None else "length + last token"),
                              "requests_n": len(by_rid),
                              # a_r AS THE MATCH RECORD SERVED IT, per request (the fact the not-executed class reads)
                              "served_by_request": {r: len(e.get("generated_token_ids") or []) for r, e in by_rid.items()},
                              "executed_by_request": {r: e.get("executed_prefix_len") for r, e in by_rid.items()}}
        res["tokens_equal"] = all(v is True for v in eq.values()) if eq else None
        if declared_lag is not None and (tj_lag is None or int(tj_lag) != int(declared_lag)):
            problems.append(f"lag not attested on the record: the Program declares LAG {declared_lag}, the Match record's tokens.json attests {tj_lag!r} (executed-prefix condition (b))")
    if gm is None:
        if required and tok is not None:
            res["notes"].append("global_match.json not beside the Match record: the G6 account leg is missing")
            problems.append(f"executed prefix reduced from the cap for {reduced}: GM-01's account (global_match.json) is not beside the Match record -- the "
                            f"Commit's count is linked to the served output only, not to the Match account (executed-prefix condition (a))")
    else:
        g6 = gm.check("G6")
        lens = g6.lengths_by_request if g6 is not None and g6.lengths_by_request is not None else {}
        gl = (None if g6 is None else g6.lag) or gm.lag or {}
        failed = None if gm.failed_checks is None else list(gm.failed_checks)
        res["global_match"] = {"path": account.get("global_match_path"), "sha256": account.get("global_match_sha256"), "verdict": gm.verdict,
                               "failed_checks": failed, "g6_status": None if g6 is None else g6.status,
                               "lag": {k: gl.get(k) for k in ("declared", "record_attests", "tokens_json_lag", "exercised", "variable_lag")}}
        if gm.verdict != "PASS":
            problems.append(f"GM-01 verdict {gm.verdict!r} (failed {failed}): the Match account this prefix must be linked to did not pass (executed-prefix condition (a))")
        if g6 is None:
            problems.append("global_match.json has no G6 check: no chronology account to link the executed prefix to (executed-prefix condition (a))")
        else:
            if gl.get("variable_lag"):
                problems.append("G6 reports VARIABLE LAG on this record: no constant-lag placement is linkable (executed-prefix rule)")
            if declared_lag is not None:
                for leg in ("declared", "record_attests"):
                    if gl.get(leg) is None or int(gl[leg]) != int(declared_lag):
                        problems.append(f"G6 lag.{leg} {gl.get(leg)!r} != the Program's declared LAG {declared_lag} (executed-prefix condition (b))")
            # recorder request id -> r<i> by the workload request id (G6's lengths are keyed by the recorder's id)
            g6_by_r: dict[str, dict[str, Any]] = {}
            for rec_rid, L in lens.items():
                e = next((x for x in gm.requests if x.request_id == str(rec_rid)), None)
                r = order.get(str(e.workload_request_id)) if e else None
                if r is None and e is not None and e.index is not None:
                    r = f"r{e.index}"
                if r is not None:
                    g6_by_r[r] = L
            for rid, rec in (epr.get("requests") or {}).items():
                L = g6_by_r.get(rid)
                if L is None:
                    problems.append(f"{rid}: no G6 lengths_by_request entry (G6 covers {sorted(g6_by_r)[:8]}): not in the Match account (executed-prefix condition (a))")
                    continue
                if rec.get("served_output_len") is not None and L.get("served_output_len") is not None and int(L["served_output_len"]) != int(rec["served_output_len"]):
                    problems.append(f"{rid}: G6 served_output_len {L['served_output_len']} != the Commit's {rec['served_output_len']} (executed-prefix condition (a))")
                if L.get("executed_prefix_len") is not None and int(L["executed_prefix_len"]) != int(rec["executed_prefix_len"]):
                    problems.append(f"{rid}: G6 executed_prefix_len {L['executed_prefix_len']} != this rule's {rec['executed_prefix_len']} (executed-prefix condition (a))")
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
