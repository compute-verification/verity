"""G6 (chronology): arrivals, output events, served lengths under the declared EOS ids and LAG, tokens fed forward, achieved
concurrency and the sampler geometry of the scheduled workload state."""
from __future__ import annotations

import collections
import dataclasses
import os

from verity_vllm.check.match.declaration import eos_lag_rows
from verity_vllm.check.match.match_run import MatchRun, NamedCheck
from verity_vllm.check.match.sampler_geometry import sampler_geometry_check, scheduled_live_counts


@dataclasses.dataclass(frozen=True)
class LagRule:
    """The stop rule G6 applies: the declared EOS ids and LAG, what the record attests, and the LAG consumed (None: none declared)."""
    eos_ids: set[int] | None
    decl_lag: int | None
    lag_declared: bool
    rec_lag: int | None
    rec_lag_src: str
    eos_lag: int | None
    eos_lag_src: str


def chronology(run: MatchRun, facts: dict, present: dict[str, list[int]], engine_step_of: dict[int, int]) -> None:
    """G6: declared == realized == observed arrival; contiguous rows; output events = TokenSelects, the k-th = the k-th token, fed to
    the next row; bounded by the cap and the declared LAG."""
    res, gp, tokens, steps, samples, rec, B = run.res, run.gp, run.tokens, run.steps, run.samples, run.rec, run.fold
    tokens_optional, tokens_path, oracle_dir = run.tokens_optional, run.tokens_path, run.oracle_dir
    c = NamedCheck(res, "G6", "chronology: declared == realized == observed arrival; contiguous rows; output events = TokenSelects, k-th = k-th token, feed the next row; bounded")
    # [F-r17-01, rev M-0035] the driver's tokens.json is PART of the record: without it every output-association leg (driver-served token k
    # == the recorder's k-th TokenSelect, served length == the driver's, the R17-1 two-length claims, the tokens.json lag attestation) is
    # silently skipped.  Fail closed by name; `tokens_optional=True` (CLI --tokens-optional) accepts a record without it for mechanism /
    # fixture runs, LABELLED in checks[G6].tokens_json=false -- never a row of record (row_pod always produces the file).
    if tokens is None and not tokens_optional:
        c.fail(f"the record has no {os.path.relpath(tokens_path, oracle_dir)}: the output-association legs (driver-served tokens vs the recorder's "
               f"TokenSelects, served/executed length claims, tokens.json lag attestation) cannot run; a record without it is refused unless the caller "
               f"passes --tokens-optional (labelled, never a row of record) (F-r17-01)")
    tok_req = {r["request_id"]: r for r in (tokens or {}).get("requests", [])} if tokens else {}
    rule = _lag_rule(c, gp, facts, tokens)
    decl_lag, lag_declared, rec_lag, rec_lag_src = rule.decl_lag, rule.lag_declared, rule.rec_lag, rule.rec_lag_src
    _lag_obs: dict[str, int] = {}                                    # EOS-stopped request -> rows served after its EOS (for the variable-lag finding)
    idle_rule = 0
    for g in gp["requests"]:
        rid = g["request_id"]
        ent = next(e for e in res["requests"] if e["request_id"] == rid)
        _n_problems_before = len(c.problems)
        ps = present.get(rid)
        if not ps:
            continue
        tr, idle = _arrival(c, gp, present, tok_req, g, ent, ps)
        idle_rule += idle
        events, served = _output_events(c, steps, samples, g, ent, ps)
        _served_lengths(c, rule, tok_req, _lag_obs, steps, g, ent, ps, tr, events, served)
        _fed_forward(c, steps, rec, g, ent, ps, events, served)
        # [rev-pm M-0622 TIP (a)] eos_verified is a verdict on THIS request's G6 evidence: any G6 problem on the request (an output
        # association failure -- SampleEnd token != driver token -- a chronology mismatch, ...) makes it False, never True beside a FAIL
        if len(c.problems) > _n_problems_before and ent.get("eos_verified") is True:
            ent["eos_verified"] = False
            ent["eos_verified_note"] = f"{len(c.problems) - _n_problems_before} G6 problem(s) on this request: the EOS evidence is not accepted"
            ent.pop("eos_lag_rows_accepted", None)   # [R17-1] no lag row is 'accepted' on a request whose EOS evidence is not
        ent["g6_problems"] = len(c.problems) - _n_problems_before   # [R17 stoch] the scheduled workload state is built from established requests only
    conc = _achieved_concurrency(c, res, gp, steps, present)
    # [COORD M-0169 / M-0571 (3)] the top-p split count S is a WORKLOAD-level derived value of the chronology this check established:
    # S_t = SplitsFor_v1(|live(t)|, num_SMs) per engine step; every top-p TokenSelect of the fold at t must carry it (the launch's S,
    # host-computed by vLLM -- no kernel, no fold instance of its own). Nothing to check on a greedy fold; a top-p fold against a Program
    # without a declared num_SMs fails closed.
    # [R17 stoch, BRIEF §6 / COORD R17-1] |live(t)| is the SCHEDULED workload state -- declared arrivals, served lengths (the cap, or the
    # verified EOS position), the declared lag rows after an EOS stop (the post-EOS step is active computation: its row is in the sampler
    # batch) and the prefill chunks -- derived from the per-request facts established above; the record's logits rows and the SampleEnd
    # batch must equal it at every step. The observed row count is never the source of |live(t)|: an unscheduled sampler row would
    # otherwise inflate the batch and S would follow it. The live->S transitions and their causes go on the record.
    scheduled = scheduled_live_counts(res["requests"], present)
    geo = sampler_geometry_check(gp, B, engine_step_of, steps, rec.get("sampler_batch"), scheduled)
    for p in geo["problems"]:
        c.fail(p)
    # [R17-1] the lag finding of the record as a whole: exercised iff some request stopped on an EOS before its cap; VARIABLE lag (two
    # EOS-stopped requests of ONE record with different post-EOS row counts) is evidence the profile is not the constant-lag scheduler
    # the Program declares -- surfaced by name (a G6 FAIL + res.alerts entry; the lane posts ALERT + a DECISIONS-PENDING entry), never
    # absorbed into whichever value most requests show
    li = c.info["lag"]
    li["exercised"] = bool(_lag_obs)
    li["eos_stop_requests"] = sorted(_lag_obs)
    li["lag_rows_observed"] = dict(sorted(_lag_obs.items()))
    distinct = sorted(set(_lag_obs.values()))
    li["variable_lag"] = len(distinct) > 1
    if li["variable_lag"]:
        c.fail(f"VARIABLE LAG: the EOS-stopped requests of this record show different post-EOS row counts {li['lag_rows_observed']} under one "
               f"declared scheduler (Program LAG {decl_lag}, record attests {rec_lag_src}): not a constant-lag profile -- surface (ALERT + "
               f"DECISIONS-PENDING), never fold into a constant-lag model (R17-1)")
        res.setdefault("alerts", []).append({"kind": "variable_lag", "check": "G6", "lag_rows_observed": li["lag_rows_observed"], "declared": decl_lag,
                                             "record_attests": rec_lag, "action": "ALERT + DECISIONS-PENDING entry (R17-1)"})
    if not lag_declared and li["exercised"]:
        li["note"] = "pre-R17-1 Program (no lag declared) with EOS stops: every EOS-stopped request failed by name above; rebuild with the declared lag"
    elif not lag_declared:
        li["note"] = ("pre-R17-1 Program (no lag declared); no request stopped before its cap, so the lag parameter was not exercised on this record "
                      "-- the record is labelled, not accepted as a lag-N record")
    c.info.update({"arrivals_by_idle_rule": idle_rule, "step_links": len(rec["links"]), "tokens_json": bool(tokens), "achieved_concurrency": conc,
                   "generated_lengths": {e["request_id"]: e.get("generated_length") for e in res["requests"]}, "sampler_geometry": geo["info"],
                   "scheduled_live": {"max_live": scheduled["max_live"], "transitions": scheduled["transitions"],
                                      "unestablished_steps": len(scheduled["unestablished"]),
                                      "requests": scheduled["requests"]},
                   "lengths_by_request": {e["request_id"]: {"served_output_len": e.get("served_output_len"), "executed_prefix_len": e.get("executed_prefix_len"),
                                                            "cap": e.get("max_tokens"), "cap_exhausted": e.get("cap_exhausted")} for e in res["requests"]}})
    res["achieved_concurrency"] = conc
    res["sampler_geometry"] = geo["info"]
    res["scheduled_live"] = c.info["scheduled_live"]
    res["lag"] = dict(li)   # [R17-1] the finalizer's fields: declared / record_attests / exercised / variable_lag / lag_rows_observed / lengths
    c.done()


def _lag_rule(c: NamedCheck, gp: dict, facts: dict, tokens: dict | None) -> LagRule:
    """The declared stop rule and LAG G6 consumes, and their agreement with what the record attests."""
    _sr = gp.get("stop_rule") or {}
    eos_ids = (set(int(x) for x in _sr["eos_token_ids"]) if _sr.get("eos_token_ids")
               else (set() if _sr.get("ignore_eos") else None))   # ignore_eos: declared EMPTY set -> every early stop fails, named
    # [R17-1 / DP-pad-02 (b); supersedes the ruling-21 reader] LAG -- the rows the scheduler computes after an EOS stop -- is the PROGRAM's
    # declared specialization parameter (GP-01 `lag.value` == the Workload_v1 `lag` static == stop_rule.lag/eos_lag_rows; from_gprog verified
    # the block against the descriptor).  G6 consumes THAT value and nothing else; the record's own attestation (RunHeader.profile
    # .engine_kwargs.async_scheduling, the engine's RESOLVED read-back) must AGREE with it (G1 compares the engine fact; here the lag it
    # implies) -- a disagreement or a record that does not attest FAILS by name.  Nothing is ever inferred from the trace: a Program that
    # declares no lag (pre-R17-1 Build) cannot have an EOS stop checked (FAIL by name per request); when no request stops early the lag is
    # not exercised and the record is labelled so (lag.exercised false), never silently accepted as a lag-N record.
    rec_lag, rec_lag_src = eos_lag_rows(facts)                       # what the RECORD attests (for the agreement check + the label)
    _lag_block = gp.get("lag") if isinstance(gp.get("lag"), dict) else None
    decl_lag = (_lag_block or {}).get("value")
    if decl_lag is None:
        decl_lag = _sr.get("lag", _sr.get("eos_lag_rows"))
    decl_lag = int(decl_lag) if decl_lag is not None else None
    decl_src = (_lag_block or {}).get("source") or _sr.get("eos_lag_source") or "stop_rule.eos_lag_rows"
    lag_declared = decl_lag is not None
    if lag_declared and decl_lag not in (0, 1):
        c.fail(f"the Program declares lag {decl_lag!r}, not in {{0, 1}}: not a supported scheduler specialization (R17-1)")
        lag_declared = False
    if lag_declared and rec_lag is not None and decl_lag != rec_lag:
        c.fail(f"the Program is specialized for LAG {decl_lag} (async_scheduling={bool(decl_lag)}: {decl_src}) but the record attests "
               f"{rec_lag_src} -> LAG {rec_lag}: the record is not an execution of the declared scheduler (R17-1; the header's LAG-{rec_lag} "
               f"attestation on a LAG-{decl_lag} Program FAILS by name, never re-read from the trace)")
    if lag_declared and rec_lag is None:
        c.fail(f"the Program is specialized for LAG {decl_lag} (async_scheduling={bool(decl_lag)}: {decl_src}) but the record does not attest "
               f"async_scheduling ({rec_lag_src}; RunHeader.profile.engine_kwargs / RunHeader.engine_kwargs / capture/versions.json): the "
               f"declared scheduler cannot be checked against the record, refuse by name (R17-1)")
    # the lag G6 CONSUMES: the Program's declaration (None = pre-R17-1 Program: no rule to apply; every EOS stop then fails by name below)
    eos_lag = decl_lag if lag_declared else None
    eos_lag_src = (f"Program lag {decl_lag} ({decl_src}); record attests {rec_lag_src}" if lag_declared
                   else f"Program declares no lag (pre-R17-1 Build); record attests {rec_lag_src} (NOT used as the rule)")
    c.info["stop_rule"] = {"eos_token_ids": sorted(eos_ids) if eos_ids else None, "source": _sr.get("source"),
                           "rule": "an early stop must end in a declared EOS id; else FAIL (F-pm-24)",
                           "eos_lag_rows": eos_lag, "eos_lag_source": eos_lag_src, "eos_lag_rows_declared": decl_lag,
                           "lag_rule": "R17-1: after an EOS stop exactly LAG (the Program's declared static) more row(s) run, each fed the preceding "
                                       "sample, its sample fed nowhere and absent from generated_token_ids (executed_prefix_len = a_r + LAG, "
                                       "served_output_len = a_r); the record must attest the same async_scheduling; undeclared -> refuse by name",
                           "evidence": "the recorder's SampleEnd token, cross-checked against the driver's tokens.json (output association); "
                                       "the fold TokenSelect VALUE is Commit territory (C3 openings of the sampled ids), not Match's"}
    c.info["lag"] = {"declared": decl_lag, "declared_source": decl_src if lag_declared else None, "record_attests": rec_lag, "record_source": rec_lag_src,
                     "agree": (decl_lag == rec_lag) if (lag_declared and rec_lag is not None) else None,
                     "exercised": False, "eos_stop_requests": [], "lag_rows_observed": {}, "variable_lag": False,
                     "lengths": gp.get("lengths") or ((gp.get("chronology") or {}).get("lengths") if isinstance(gp.get("chronology"), dict) else None)}
    # [R17-1] the driver's tokens.json carries the same recorded fact (doc.lag.value from resolved_async_scheduling) and per request its
    # served_output_len / executed_prefix_len claims: one record, one fact -- a tokens.json whose lag disagrees with the header's attestation
    # is a record that does not attest one execution (named); the per-request claims are checked against the observed rows below
    _tok_lag = ((tokens or {}).get("lag") or {}).get("value") if isinstance((tokens or {}).get("lag"), dict) else None
    if _tok_lag is not None and rec_lag is not None and int(_tok_lag) != int(rec_lag):
        c.fail(f"the record's own sources disagree on the scheduler lag: tokens.json lag.value {_tok_lag} vs {rec_lag_src} -> LAG {rec_lag}: "
               f"not one attested execution (R17-1)")
    c.info["lag"]["tokens_json_lag"] = _tok_lag
    return LagRule(eos_ids=eos_ids, decl_lag=decl_lag, lag_declared=lag_declared, rec_lag=rec_lag, rec_lag_src=rec_lag_src,
                   eos_lag=eos_lag, eos_lag_src=eos_lag_src)


def _arrival(c: NamedCheck, gp: dict, present: dict[str, list[int]], tok_req: dict, g: dict, ent: dict, ps: list[int]) -> tuple[dict, int]:
    """One request's declared, realized and observed arrival and contiguous rows.  Returns its tokens.json entry and 1 when it was
    admitted early under the declared idle rule, else 0."""
    rid = g["request_id"]
    idle_rule = 0
    first, last = ps[0], ps[-1]
    ent.update({"first_step": first, "finish_step": last})
    if ps != list(range(first, last + 1)):
        c.fail(f"{rid}: rows not contiguous over engine steps {first}..{last} (preempted / resumed: not in the Program)")
    decl = int(g["arrive_step"])
    tr = tok_req.get(rid) or {}
    realized = tr.get("arrive_step_realized")
    if realized is None:
        realized = tr.get("first_step")
    ent["arrive_step_realized"] = realized
    if realized is not None and int(realized) != first:
        c.fail(f"{rid}: driver realized arrival {realized} != observed first engine step {first}")
    if first != decl:
        # the declared rule: admitted early only if every earlier-arriving request had finished (engine would idle)
        earlier = [q for q in gp["requests"] if (q["arrive_step"], q["index"]) < (decl, g["index"])]
        if first < decl and earlier and all(present.get(q["request_id"], [-1])[-1] < first for q in earlier):
            idle_rule += 1
            ent["arrival_note"] = f"admitted at {first} < declared {decl}: every earlier request had finished (declared idle rule)"
        else:
            c.fail(f"{rid}: declared arrival step {decl} != observed {first}")
    pst = g.get("programmed_steps")
    if pst and int(pst.get("prefill", decl)) != first and "arrival_note" not in ent:
        c.fail(f"{rid}: programmed prefill step {pst.get('prefill')} != observed {first}")
    if pst and pst.get("decode") and ent.get("prefill_chunks", 1) == 1:
        d0 = int(pst["decode"][0]) + (first - decl)
        if len(ps) > 1 and ps[1] != d0:
            c.fail(f"{rid}: programmed first decode step {d0} != observed {ps[1]}")
    if tr.get("first_step") is not None and int(tr["first_step"]) != first:
        c.fail(f"{rid}: driver first_step {tr['first_step']} != observed {first}")
    # (the driver's finish_step is compared in _served_lengths, once the request's accepted lag rows after an EOS stop are known)
    return tr, idle_rule


def _output_events(c: NamedCheck, steps: dict, samples: dict, g: dict, ent: dict, ps: list[int]) -> tuple[list, list]:
    """One request's output events (engine step, sampled token, logits position), and the served ones among them."""
    rid = g["request_id"]
    LP = g["prompt_len"]
    events = []  # (engine step, sampled token, logits position)
    for s in ps:
        lr = steps[s]["logits_rows"]
        js = [j for j, (r, _p) in enumerate(lr) if r == rid]
        if len(js) > 1:
            c.fail(f"{rid}: {len(js)} logits rows in engine step {s}")
        for j in js:
            pos = lr[j][1]
            samp = samples.get(s)
            if samp is None or j >= len(samp):
                c.fail(f"{rid}: no SampleEnd token for engine step {s} logits row {j}")
                continue
            events.append((s, samp[j], pos))
    served = [(s, t) for s, t, pos in events if pos >= LP - 1]      # a chunk's discarded sample has pos < LP-1
    ent["output_events"] = len(served)
    ent["output_event_steps"] = [s for s, _ in served][:6] + (["..."] if len(served) > 6 else [])
    ent["output_event_steps_last"] = served[-1][0] if served else None
    n_ts = ent.get("token_selects")
    if n_ts is not None and n_ts != len(events):
        c.fail(f"{rid}: {n_ts} TokenSelect instances in the fold but {len(events)} sampled logits rows in the record")
    return events, served


def _served_lengths(c: NamedCheck, rule: LagRule, tok_req: dict, lag_obs: dict[str, int], steps: dict, g: dict, ent: dict, ps: list[int],
                    tr: dict, events: list, served: list) -> None:
    """One request's answer and lag rows: its stop against the cap, the declared EOS ids and LAG, and the driver's claims."""
    eos_ids, decl_lag, lag_declared, rec_lag = rule.eos_ids, rule.decl_lag, rule.lag_declared, rule.rec_lag
    rec_lag_src, eos_lag, eos_lag_src = rule.rec_lag_src, rule.eos_lag, rule.eos_lag_src
    rid, LP, last = g["request_id"], g["prompt_len"], ps[-1]
    gen = tr.get("generated_token_ids")
    cap = int(g["max_tokens"])
    # [COORD M-0213 (1)(2) / ruling 21 / DP-pad-02] the request's served rows are its answer a_r = the events up to and including
    # the FIRST declared EOS (or all of them on a length stop) plus, after an EOS stop ONLY, exactly the placeholder rows the
    # record's attested scheduler computes (eos_lag_rows: async_scheduling=True -> 1, False -> 0, unattested -> refuse by name).
    # Every accepted lag row's input is the preceding sample (the feed-forward check below), its own sample is fed nowhere and
    # is absent from the driver's generated_token_ids; the fold holds a TokenSelect for it (token_selects == events). Anything
    # else -- more rows than attested, fewer (a deleted step), a lag row after a length stop -- FAILS by name.
    first_eos = next((j for j, (_s, t) in enumerate(served) if eos_ids and int(t) in eos_ids), None)
    # an EOS sampled AT the cap is a length stop (vLLM checks max_tokens before the EOS; the scheduler knows the length ahead: no
    # placeholder row); an EOS before the cap is an EOS stop and the lag rule applies
    eos_stop = first_eos is not None and first_eos + 1 < cap
    # a length stop's served output is AT MOST the cap: rows beyond it are execution past the cap (named below), never output
    a_r = first_eos + 1 if eos_stop else min(len(served), cap)
    lag_rows = len(served) - a_r
    ent["answer_events"] = a_r
    ent["lag_rows"] = lag_rows
    # [R17-1] the two lengths, kept distinct on the record: what was DESIGNATED as output vs what RAN
    ent["served_output_len"] = a_r
    ent["executed_prefix_len"] = len(served)
    ent["cap_exhausted"] = (a_r == cap)
    if eos_stop:
        lag_obs[rid] = lag_rows
        if lag_declared and (rec_lag is None or rec_lag != decl_lag):
            # the record-level disagreement (named once above) makes THIS request's EOS stop unacceptable too: its post-EOS rows cannot be
            # read as the declared lag when the record does not attest the Program's scheduler -- the request is not established (no
            # downstream state, stoch's scheduled live included, is built on it)
            c.fail(f"{rid}: EOS stop at engine step {served[first_eos][0]} on a record whose attested scheduler ({rec_lag_src}) is not the "
                   f"Program's declared LAG {decl_lag}: the {lag_rows} post-EOS row(s) are not accepted as the declared lag (R17-1)")
    # the driver's own claims for the two lengths (tokens.json, R17-1 schema: served = len(generated_token_ids), executed = served + the
    # record's lag on an EOS stop) must be what the record's rows show -- a claim beside the rows, named when it disagrees
    _tq = tok_req.get(rid) or {}
    for _k, _v in (("served_output_len", a_r), ("executed_prefix_len", len(served))):
        if _tq.get(_k) is not None and int(_tq[_k]) != int(_v):
            c.fail(f"{rid}: the driver's tokens.json claims {_k} {_tq[_k]} but the record's rows show {_v} (served_output_len {a_r}, "
                   f"executed_prefix_len {len(served)}, cap {cap}): the record's own claim disagrees with its rows (R17-1)")
    answer = [t for _, t in served[:a_r]]
    if gen is not None and answer != list(gen)[:a_r]:
        c.fail(f"{rid}: sampled tokens per engine step {answer[:8]} != driver's generated_token_ids {list(gen)[:8]} (output association)")
    if gen is not None and len(list(gen)) != a_r:
        c.fail(f"{rid}: the driver's generated_token_ids has {len(list(gen))} tokens but the request's answer is {a_r} event(s) "
               f"(up to its first declared EOS): {'the rows after the EOS are not output (a lag row sample reported)' if len(list(gen)) > a_r else 'output tokens the record never sampled'} (M-0213)")
    if len(served) > cap:
        # [R17-1, rev M-0027 (4)] REQUIRED from the trace, not assumed: a length stop runs no row after the cap under ANY lag (vLLM knows the
        # length ahead; the scheduler computes a post-EOS row only for an EOS stop) -- an executed row past the cap is execution past cap
        c.fail(f"{rid}: execution past cap -- {len(served)} output events executed but the declared max_tokens is {cap} (executed_prefix_len "
               f"{len(served)} > served_output_len {a_r} = cap; a length stop computes no post-cap row under any LAG, the Program's LAG "
               f"{eos_lag if eos_lag is not None else 'undeclared'} applies to EOS stops only) (output interface; R17-1)")
    if ps and a_r != cap:
        ent["finish"] = "eos" if a_r < cap else "?"
        if tr.get("finish_reason") not in (None, "stop"):
            c.fail(f"{rid}: {a_r} events < max_tokens but driver finish_reason {tr.get('finish_reason')!r}")
        # [rev-pm ALERT M-0537 / F-pm-24] the driver's finish_reason is the prover's word; the EOS itself is the evidence: the last
        # ANSWER token (tied to a SampleEnd + the driver's generated_token_ids by the output association above) must be one of the
        # Program's declared EOS ids. A request that stopped early on any other token (dropped, aborted, stop string, a prover that
        # skipped the tail) is NOT an EOS truncation and FAILS -- whatever finish_reason says. No declared set -> fail closed.
        last_tok = served[a_r - 1][1] if served and a_r else None
        if eos_ids is None:
            c.fail(f"{rid}: stopped after {a_r} < max_tokens {cap} events but the Program declares no EOS set "
                   f"(stop_rule.eos_token_ids): the truncation cannot be verified, fail closed (F-pm-24)")
            ent["eos_verified"] = False
        elif last_tok is None or int(last_tok) not in eos_ids:
            c.fail(f"{rid}: stopped after {a_r} < max_tokens {cap} events on token {last_tok}, not a declared EOS id "
                   f"{sorted(eos_ids)[:6]} (driver finish_reason {tr.get('finish_reason')!r}): an early stop without EOS is not modelled (F-pm-24)")
            ent["eos_verified"] = False
        else:
            ent["eos_verified"] = True
            ent["eos_token"] = int(last_tok)
    else:
        ent["finish"] = "length"
    # [LIFTING-SPEC v0.3 s7 [P1] (i), the other direction; R17-1] executed activity may not run PAST the answer by more than the
    # Program's DECLARED lag: executed_prefix_len == a_r + LAG exactly on an EOS stop (an EOS is among the served tokens), == a_r on a
    # length stop (vLLM knows the length ahead: no placeholder row). Under a declared ignore_eos (empty set) there is nothing to pass.
    finish_step_expected = last
    if eos_stop:
        eos_step = served[first_eos][0]
        finish_step_expected = eos_step
        ent["eos_step"] = eos_step
        if eos_lag is None:
            c.fail(f"{rid}: stopped on the declared EOS id {int(served[first_eos][1])} at engine step {eos_step} with {lag_rows} more "
                   f"row(s) executed after it, but the Program declares no LAG ({eos_lag_src}): the post-EOS rows cannot be checked against "
                   f"a declaration -- refuse by name, nothing is inferred from the trace or read from the header as the rule; rebuild the "
                   f"workload Program with the declared async-scheduling specialization (R17-1)")
            ent["eos_verified"] = False
        elif lag_rows > eos_lag:
            c.fail(f"{rid}: execution past EOS -- output event {first_eos} is the declared EOS id {int(served[first_eos][1])} but "
                   f"{lag_rows} more row(s) were executed where the Program's declared scheduler ({eos_lag_src}) computes LAG {eos_lag} after an EOS "
                   f"(executed_prefix_len {len(served)} > served_output_len {a_r} + LAG {eos_lag}): the Program calls the request finished there "
                   f"(LIFTING-SPEC s7 [P1]; R17-1)")
            ent["eos_verified"] = False
            ent["eos_passed_at_event"] = first_eos
        elif lag_rows < eos_lag:
            c.fail(f"{rid}: the Program's declared scheduler ({eos_lag_src}) computes LAG {eos_lag} post-EOS row(s) after an EOS stop but "
                   f"{lag_rows} were executed after the EOS at engine step {eos_step} (executed_prefix_len {len(served)} < served_output_len {a_r} + "
                   f"LAG {eos_lag}): a deleted step -- the record is not one execution of the declared engine (R17-1)")
            ent["eos_verified"] = False
        else:
            for k in range(a_r, len(served)):
                s_k, t_k = served[k]
                # the lag row's own sample is fed nowhere: no later row of this request at the position after it
                pos_k = [pos for s, _t, pos in events if s == s_k and pos >= LP - 1][0]
                fed_later = [s for s in ps if s > s_k and any(r == rid and p == pos_k + 1 for r, p in steps[s]["rows"])]
                if fed_later:
                    c.fail(f"{rid}: the lag row at engine step {s_k} (after the EOS at {eos_step}) sampled {int(t_k)} which is fed to the "
                           f"request's row at position {pos_k + 1} in engine step {fed_later[0]}: not a placeholder row (M-0213)")
                    ent["eos_verified"] = False
                if gen is not None and len(list(gen)) > k and int(list(gen)[k]) == int(t_k):
                    ent["eos_verified"] = False   # already failed above (gen longer than a_r): the leak is named there
    elif lag_rows:
        # a length stop with rows past the cap (a_r = cap < executed): already named 'execution past cap' above; the explicit rule for the record
        c.fail(f"{rid}: {lag_rows} row(s) executed after a length stop: the scheduler computes no placeholder row when the cap ends the request "
               f"(M-0213; R17-1: LAG applies to EOS stops only)")
    if ent.get("eos_verified") and eos_lag:
        ent["eos_lag_rows_accepted"] = lag_rows
    # the driver's finish_step names the step the request FINISHED at: the EOS step on an EOS stop (the lag row, if any, comes after
    # it -- pad's fixture M-0162: finish_step 10, EOS at 10, lag row at 11), the last row on a length stop
    if tr.get("finish_step") is not None and int(tr["finish_step"]) != finish_step_expected:
        c.fail(f"{rid}: driver finish_step {tr['finish_step']} != observed {finish_step_expected}"
               + (f" (the EOS step; the request's last served row is the lag row at {last})" if finish_step_expected != last else ""))


def _fed_forward(c: NamedCheck, steps: dict, rec: dict, g: dict, ent: dict, ps: list[int], events: list, served: list) -> None:
    """Each output event feeds the request's next input row; its StepLinks carry the sampled token."""
    rid, LP = g["request_id"], g["prompt_len"]
    fed = 0
    for (s, t, pos), s_next in zip(events, ps[1:] if ps else []):
        rows_next = steps[s_next]["rows"]
        toks_next = steps[s_next]["input_token_ids"]
        k = [i for i, (r, p) in enumerate(rows_next) if r == rid and p == pos + 1]
        if pos + 1 < LP:
            continue  # the next row is still prompt (chunked prefill): the chunk's sample is discarded
        if not k:
            c.fail(f"{rid}: no input row at position {pos + 1} in engine step {s_next} for the token sampled at step {s}")
        elif toks_next and toks_next[k[0]] != t:
            c.fail(f"{rid}: token sampled at step {s} ({t}) != input row token at step {s_next} ({toks_next[k[0]]})")
        else:
            fed += 1
    ent["events_fed_forward"] = fed
    for lk in rec["links"]:
        if lk["request_id"] == rid and lk.get("relation") == "eq" and lk.get("value") is not None:
            ev = [t for s, t, _ in events if s == lk["src_step"]]
            if ev and ev[0] != lk["value"]:
                c.fail(f"{rid}: StepLink value {lk['value']} at step {lk['step']} != sampled {ev[0]} at src_step {lk['src_step']}")
    if ent.get("T_observed") is not None and ent["T_observed"] != len(served) - 1:
        c.fail(f"{rid}: fold T_observed {ent['T_observed']} != output events - 1 ({len(served) - 1})")
    ent["generated_length"] = len(served)


def _achieved_concurrency(c: NamedCheck, res: dict, gp: dict, steps: dict, present: dict[str, list[int]]) -> dict:
    """Live requests per engine step, observed against programmed."""
    # achieved concurrency (BRIEF §11: a configured limit is not proof the run exercised that many simultaneous requests): the
    # observed number of distinct requests with rows per engine step, against the programmed one from the declaration alone
    # (members with a_r <= e <= a_r + cap_r - 1); observed < programmed is explained only by EOS-truncated requests
    obs_per_step = {s: len({r for r, _ in steps[s]["rows"]}) for s in steps}
    prog_per_step: dict[int, int] = collections.defaultdict(int)
    for g in gp["requests"]:
        for e in range(int(g["arrive_step"]), int(g["arrive_step"]) + int(g["max_tokens"])):
            prog_per_step[e] += 1
    obs_max = max(obs_per_step.values()) if obs_per_step else 0
    prog_max = max(prog_per_step.values()) if prog_per_step else 0
    conc = {"programmed_max": prog_max, "observed_max": obs_max, "declared_requests": len(gp["requests"]),
            "observed_steps_at_max": sum(1 for v in obs_per_step.values() if v == obs_max),
            "observed_histogram": dict(sorted(collections.Counter(obs_per_step.values()).items())),
            "programmed_histogram": dict(sorted(collections.Counter(prog_per_step.values()).items())),
            "engine_steps_observed": len(steps), "engine_steps_programmed": len(prog_per_step)}
    # [F-pm-24] the shortfall is explained only by EOS-VERIFIED truncation: an unverified early stop does not unlock it
    trunc = [e for e in res["requests"] if e.get("eos_truncated") or (e.get("finish") == "eos")]
    if obs_max != prog_max and not (trunc and all(e.get("eos_verified") is True for e in trunc)):
        c.fail(f"achieved concurrency {obs_max} != programmed {prog_max} with no EOS-verified truncated request "
               f"({len(trunc)} truncated, {sum(1 for e in trunc if e.get('eos_verified') is True)} EOS-verified): declared schedule not exercised")
    # [COORD M-0571 (1)] per ENGINE STEP: the waiver applies only to steps AFTER a verified EOS. A request that reached its cap or
    # stopped on a verified EOS occupies exactly the steps it was observed in; one that stopped early WITHOUT a verified EOS was
    # programmed to continue for (cap - served) more steps -- those steps are named here as "declared schedule not exercised"
    expected_per_step: dict[int, int] = collections.defaultdict(int)
    for e in res["requests"]:
        rid = e["request_id"]
        ps_r = present.get(rid) or []
        occ = set(ps_r)
        if e.get("finish") == "eos" and e.get("eos_verified") is not True and ps_r:
            short = int(e.get("max_tokens") or 0) - int(e.get("output_events") or 0)
            occ.update(range(ps_r[-1] + 1, ps_r[-1] + 1 + max(0, short)))
        for st in occ:
            expected_per_step[st] += 1
    bad_steps = [st for st in sorted(set(expected_per_step) | set(obs_per_step)) if obs_per_step.get(st, 0) != expected_per_step.get(st, 0)]
    if bad_steps:
        culprits = [e["request_id"] for e in res["requests"] if e.get("finish") == "eos" and e.get("eos_verified") is not True]
        c.fail(f"engine steps {bad_steps[:8]}{'...' if len(bad_steps) > 8 else ''}: live requests observed "
               f"{[obs_per_step.get(st, 0) for st in bad_steps[:8]]} != programmed {[expected_per_step.get(st, 0) for st in bad_steps[:8]]} -- the shortfall "
               f"is not after a verified EOS ({culprits} stopped early without one): declared schedule not exercised (M-0571 (1))")
    conc["steps_short_of_programmed"] = len(bad_steps)
    conc["verified_eos_steps"] = sorted({int(e["output_event_steps_last"]) for e in res["requests"] if e.get("eos_verified") is True and e.get("output_event_steps_last") is not None})
    return conc
