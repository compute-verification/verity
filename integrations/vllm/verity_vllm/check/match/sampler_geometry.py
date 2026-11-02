"""Sampler geometry: the top-p split count every TokenSelect of the fold carries, against the registered split count
(`topp_geometry.registered_splits`), and the fold's sampler steps against the scheduled live count."""
from __future__ import annotations

import collections
import re

_S_STATIC_RE = re.compile(r"[{,]S=(\d+)(?=[,}])")
_TOPP_FAMILY = "GumbelTopPTokenSelect_v1"
_SPLITS_OPERAND = 5   # GumbelTopPTokenSelect_v1{V}(logits, top_p, seed, pos, temperature, splits): the derived S operand


def top_p_split_count(spec: str, B=None, inst: dict | None = None) -> int | None:
    """The launch's top-p split count S carried by a fold TokenSelect instance.  The value of
    the select's `Const32[S]` splits operand (compare.RootView._bind_splits keeps it in `B.splits_values[select node]`; the constant itself
    is bound to the Program's request-owned `splits` input entry, or held equal to a single-request Program's own `Const32[S]` by
    `constant_splits_agree`), else the `S=<n>` static of the older form `GumbelTopPTokenSelect_v1{V,S}` (the fold's spec strings may carry
    a `prefix:`).  None when the instance carries no readable S."""
    if B is not None and inst is not None:
        v = getattr(B, "splits_values", {}).get(inst.get("k"))
        if v is not None:
            return int(v)
        # an unbound instance sequence (a RootView-shaped reader without compare's binding): the 6th operand must be ONE run
        # ("n", k, 0, 1) onto a `Const32[0x<S>]_v1` instance emitted for the select (fold._const, patterns._stage3 Const(32, S, "splits"))
        args = inst.get("args") or []
        if len(args) > _SPLITS_OPERAND and len(args[_SPLITS_OPERAND]) == 1:
            run = args[_SPLITS_OPERAND][0]
            if len(run) == 4 and run[0] == "n" and (run[2], run[3]) == (0, 1):
                j = (getattr(B, "k2i", None) or {}).get(run[1])
                if j is not None:
                    from verity_vllm.check.match.instance_sequence import const_value
                    cv = const_value(B.inst[j]["spec"])
                    if cv is not None and cv[0] == 32:
                        return int(cv[1])
    m = _S_STATIC_RE.search(spec.split(":", 1)[-1])
    return int(m.group(1)) if m else None


def constant_splits_agree(A, B) -> dict | None:
    """A Program carries the registered S = SplitsFor_v1(1, num_SMs) as its own `Const32[S]` per select,
    bound like the fold's and removed from both sequences by `compare.RootView._bind_splits`, so neither the canonical hash nor a DAG hash
    carries its value: the two sides agree only if the constants are equal at every event both have (an event on one side only is a
    step the structural comparison already reports, e.g. a component's steps past an EOS).  None when either side binds no constant
    (a derived-S Program reads its `splits` input instead).  Mismatches are [event, S of A, S of B]."""
    ea, eb = _splits_by_event(A), _splits_by_event(B)
    if not (ea and eb):
        return None
    events = sorted(set(ea) & set(eb))
    bad = [[e, ea[e], eb[e]] for e in events if ea[e] != eb[e]]
    return {"equal": not bad, "events": len(events), "mismatches": bad[:16]}


def _splits_by_event(P) -> dict[int, int]:
    """{event ordinal: S} of a compare.RootView's bound `Const32[S]` splits operands."""
    out = {}
    for k, S in getattr(P, "splits_values", {}).items():
        i = P.k2i.get(k)
        if i is not None and i in P.i2step:
            out[P.i2step[i][0]] = S
    return out


def scheduled_live_counts(requests: list[dict], present: dict[str, list[int]]) -> dict:
    """The SCHEDULED workload state per engine step, built from the per-request facts G6
    has just established -- never from the record's per-step row list.

    For every request r with rows: its sampler-active steps are the contiguous range
        E_r = [first_r, first_r + (prefill_chunks_r - 1) + a_r + lag_r)
    where first_r is its verified arrival (== the declared step, or admitted by the declared idle rule), prefill_chunks_r - 1 the chunk
    steps whose sample is discarded, a_r its answer length (the cap on a length stop; the position of the first declared EOS on a
    VERIFIED EOS stop) and lag_r the placeholder rows after an EOS stop = the attested AND declared scheduler lag (`eos_lag_rows`:
    LAG is a declared specialization parameter; the post-EOS step is active internal computation whose row IS in the sampler batch).
    |live(t)| = #{r : t in E_r}.  A request whose G6 evidence is not established (a problem on it, an unverified early stop) leaves the
    state UNESTABLISHED at every step it touches -- the caller fails closed there instead of falling back to the observed rows.

    Returns {"live": {step: n}, "unestablished": {step: [rid..]}, "transitions": [...], "requests": {rid: {...}}}; a transition names
    the step, the live count before/after and its causes (arrivals; departures with the stop kind and the lag rows that ended)."""
    live: dict[int, int] = collections.defaultdict(int)
    unest: dict[int, list[str]] = collections.defaultdict(list)
    arrivals: dict[int, list[str]] = collections.defaultdict(list)
    departures: dict[int, list[dict]] = collections.defaultdict(list)
    per_req: dict[str, dict] = {}
    for e in requests:
        rid = e["request_id"]
        ps = present.get(rid) or []
        if not ps:
            continue
        first = int(e.get("first_step", ps[0]))
        chunks = int(e.get("prefill_chunks") or 1)
        a_r = e.get("answer_events")
        lag = int(e.get("lag_rows") or 0)
        finish = e.get("finish")
        ok = (a_r is not None and finish in ("length", "eos")
              and (finish != "eos" or e.get("eos_verified") is True)
              and not e.get("g6_problems"))
        if not ok:
            for s in ps:
                unest[s].append(rid)
            per_req[rid] = {"established": False, "observed_steps": [ps[0], ps[-1]]}
            continue
        n = (chunks - 1) + int(a_r) + lag
        end = first + n   # exclusive
        exp = list(range(first, end))
        if exp != ps:
            for s in sorted(set(exp) | set(ps)):
                unest[s].append(rid)
            per_req[rid] = {"established": False, "scheduled_steps": [first, end - 1], "observed_steps": [ps[0], ps[-1]],
                            "note": "scheduled sampler steps != observed rows"}
            continue
        for s in exp:
            live[s] += 1
        arrivals[first].append(rid)
        departures[end].append({"request": rid, "finish": finish, "lag_rows": lag, "answer_events": int(a_r), "prefill_chunks": chunks})
        per_req[rid] = {"established": True, "first_step": first, "last_step": end - 1, "prefill_chunks": chunks, "answer_events": int(a_r),
                        "lag_rows": lag, "finish": finish}
    trans: list[dict] = []
    prev = 0
    for s in sorted(set(live) | set(arrivals) | set(departures)):
        cur = live.get(s, 0)
        if cur != prev or arrivals.get(s) or departures.get(s):
            trans.append({"step": s, "live_from": prev, "live_to": cur, "arrivals": sorted(arrivals.get(s, [])),
                          "departures": sorted(departures.get(s, []), key=lambda d: d["request"])})
        prev = cur
    return {"live": dict(live), "unestablished": {s: sorted(v) for s, v in sorted(unest.items())}, "transitions": trans, "requests": per_req,
            "max_live": max(live.values()) if live else 0}


def sampler_geometry_check(gp: dict, B, engine_step_of: dict[int, int], steps: dict, sampler_batch: dict[int, int] | None = None,
                           scheduled: dict | None = None) -> dict:
    """The top-p split geometry, checked per engine step.  The served engine runs every `_apply_topp_split` launch at the registered
    split count S = `topp_geometry.registered_splits(num_SMs)` whatever the sampler batch (`engine.topp_pin`), and the Program takes it
    as a constant; no kernel computes S, so the fold carries it only as the launch's S on every stochastic top-p TokenSelect.  num_SMs
    is the Program's declared target constant (`sampler_geometry.num_sms`, else `applicability.components.num_sms`).  Rule: every such
    instance carries S == registered_splits(num_SMs), and a declared `sampler_geometry.splits` is that S; the live row count never
    enters S.  A top-p fold against a Program that declares no num_SMs FAILS closed; an instance without a readable S FAILS by name.
    Greedy folds (no top-p TokenSelect) have nothing to check (`checked` = 0).

    With `scheduled` (scheduled_live_counts: the state G6 derived from arrivals, served lengths, verified EOS stops and the declared
    lag), the record's logits rows and the SampleEnd batch must both equal the scheduled live count (a sampler row the workload state
    does not schedule -- an undeclared request, a row past a stop, a missing lag row -- is named), a top-p select on a step the state
    schedules nobody at is named, a scheduled sampling step without a top-p select is named, and a step whose state is unestablished
    fails closed.  Without `scheduled` the SampleEnd batch is held to the observed logits rows."""
    from verity.ml.definitions.registry.sampling_vocab import SHARED_GREEDY
    from verity.ml.definitions.registry.topp_geometry import registered_splits
    from verity_vllm.program.sampling_event import family_of
    topp = (_TOPP_FAMILY, SHARED_GREEDY[_TOPP_FAMILY].id)   # either `sampler_construction`
    per_step: dict[int | None, set[int | None]] = collections.defaultdict(set)
    n_topp = 0
    for i, r in enumerate(B.inst):
        sp = r["spec"]
        if family_of(sp) not in topp:
            continue
        n_topp += 1
        per_step[engine_step_of.get(i)].add(top_p_split_count(sp, B, r))
    sched_live = (scheduled or {}).get("live") if isinstance(scheduled, dict) else None
    info: dict = {"relation": "S = registered_splits(num_SMs) = SplitsFor_v1(1, num_SMs) at every launch, whatever the live batch "
                              "(engine.topp_pin); num_SMs declared",
                  "top_p_token_selects": n_topp, "checked_steps": 0,
                  "live_source": ("scheduled workload state (G6: declared arrivals, served lengths = cap or verified EOS, declared lag rows, "
                                  "prefill chunks) -- the record's logits rows and SampleEnd batch must equal it" if sched_live is not None
                                  else "observed logits rows of the record (no scheduled state passed)")}
    problems: list[str] = []
    if scheduled:
        info["transitions"] = [dict(t) for t in scheduled.get("transitions", [])]
    if n_topp == 0:
        info["note"] = "no top-p TokenSelect in the fold: nothing to check"
        return {"info": info, "problems": problems}
    sg = gp.get("sampler_geometry") if isinstance(gp.get("sampler_geometry"), dict) else {}
    nsm = sg.get("num_sms")
    if nsm is None:
        nsm = ((gp.get("applicability") or {}).get("components") or {}).get("num_sms")
    info["num_sms"] = nsm
    info["declared_by"] = "sampler_geometry" if sg.get("num_sms") is not None else ("applicability.components.num_sms" if nsm is not None else None)
    if nsm is None:
        problems.append(f"the fold has {n_topp} top-p TokenSelect(s) (S = {sorted(x for v in per_step.values() for x in v if x is not None)[:6]}) "
                        f"but the Program declares no num_SMs (sampler_geometry.num_sms / applicability.components.num_sms): the registered "
                        f"split count cannot be evaluated, fail closed")
        return {"info": info, "problems": problems}
    exp = int(registered_splits(int(nsm)))
    info["registered_splits"] = exp
    declared = sg.get("splits")
    if declared is not None and declared != exp:
        problems.append(f"the Program declares sampler_geometry.splits = {declared!r}, not the registered split count {exp} of num_SMs={nsm}")
    mism: list[dict] = []
    hist: collections.Counter = collections.Counter()
    batch_dis: list[dict] = []
    sched_dis: list[dict] = []
    unest_steps: list[dict] = []
    unscheduled: list[dict] = []
    sched_unest = (scheduled or {}).get("unestablished") or {} if sched_live is not None else {}
    for s in sorted(per_step, key=lambda x: (x is None, x if x is not None else -1)):
        Ss = per_step[s]
        if s is None or s not in steps:
            problems.append(f"{len(Ss)} top-p TokenSelect S value(s) {sorted(x for x in Ss if x is not None)} on an engine step the record does not "
                            f"list (fold meta step {s!r}): the sampler batch is unknown, fail closed")
            continue
        live_obs = len(steps[s]["logits_rows"])
        sb = (sampler_batch or {}).get(s)
        if sched_live is not None:
            if sched_unest.get(s) or sched_unest.get(str(s)):
                unest_steps.append({"step": s, "requests": sched_unest.get(s) or sched_unest.get(str(s))})
                continue
            live = int(sched_live.get(s, sched_live.get(str(s), 0)))
            if live == 0:
                unscheduled.append({"step": s, "logits_rows": live_obs, "fold_S": sorted(x for x in Ss if x is not None)})
                continue
            if live != live_obs:
                sched_dis.append({"step": s, "scheduled_live": live, "logits_rows": live_obs})
            if sb is not None and int(sb) != live:
                batch_dis.append({"step": s, "scheduled_live": live, "logits_rows": live_obs, "sampler_batch": int(sb)})
        else:
            live = live_obs
            if sb is not None and int(sb) != live:
                batch_dis.append({"step": s, "logits_rows": live, "sampler_batch": int(sb)})
        for x in Ss:
            hist[f"live={live}:S={x}"] += 1
        if None in Ss:
            problems.append(f"engine step {s}: a top-p TokenSelect carries no readable split count S (neither a Const32[S] splits operand -- the 6th "
                            f"-- nor an S static)")
        bad = sorted(x for x in Ss if x is not None and x != exp)
        if bad:
            mism.append({"step": s, "live": live, "registered_S": exp, "fold_S": bad})
        info["checked_steps"] += 1
    if sched_live is not None:
        # every step the workload state schedules a sampler batch at must carry a top-p select of the fold (a stochastic row samples at
        # every engine step; a scheduled step without one is a missing launch, not a greedy step)
        missing = sorted(int(s) for s, n in sched_live.items() if int(n) > 0 and int(s) not in per_step and int(s) in steps)
        if missing:
            problems.append(f"{len(missing)} engine step(s) the scheduled workload state samples at (live > 0) carry no top-p TokenSelect in the fold: "
                            f"{missing[:8]}{'...' if len(missing) > 8 else ''} -- a sampler launch the record lacks, fail closed")
        if unest_steps:
            problems.append(f"{len(unest_steps)} engine step(s) whose scheduled workload state is NOT established (a request's arrival / served "
                            f"length / EOS / lag evidence failed in G6): {unest_steps[:6]} -- the record's own row count is not a substitute; "
                            f"fail closed")
        if unscheduled:
            problems.append(f"{len(unscheduled)} engine step(s) with a top-p TokenSelect where the scheduled workload state has NO live request: "
                            f"{unscheduled[:6]} -- a sampler launch outside the declared workload (undeclared request / rows past a stop)")
        if sched_dis:
            problems.append(f"{len(sched_dis)} engine step(s) whose logits rows != the scheduled live count {sched_dis[:6]}"
                            f"{'...' if len(sched_dis) > 6 else ''} -- the sampler batch is not the scheduled workload state (declared arrivals, "
                            f"served lengths to cap or verified EOS, declared lag rows)")
        info["scheduled_max_live"] = (scheduled or {}).get("max_live")
    if batch_dis:
        problems.append(f"{len(batch_dis)} engine step(s) whose SampleEnd logits_view batch != the "
                        f"{'scheduled live count' if sched_live is not None else 'record' + chr(39) + 's logits rows'}: {batch_dis[:6]} "
                        f"-- the sampler launch saw another batch than the chronology lists")
    if mism:
        problems.append(f"{len(mism)} engine step(s) whose top-p split count S is not the registered S = {exp} (num_SMs={nsm}): {mism[:6]}"
                        f"{'...' if len(mism) > 6 else ''} -- the launch ran at another split count than the one the Program pins")
    info.update({"steps_with_mismatch": len(mism), "live_to_S": dict(sorted(hist.items())),
                 "sampler_batch_recorded_steps": sum(1 for s in per_step if s is not None and (sampler_batch or {}).get(s) is not None)})
    return {"info": info, "problems": problems}
