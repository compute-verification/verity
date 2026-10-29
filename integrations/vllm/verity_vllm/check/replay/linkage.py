"""Boundary linkage (exhaustive, not sampled): every request's committed input ids linked to its declared prompt and to
its own previous-step sampled ids over the executed prefix (`boundary_linkage`), and the prescribed sampler inputs linked
to the declared workload (`prescribed_input_linkage`)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from verity_vllm.query.manifest import format as RM
from verity_vllm.commit.opened import UnverifiedUnlinked
from verity.ml.definitions.numerics.rows import SAMPLED_MEMBER
from verity_vllm.program.sampling_event import circuit_token_id
from verity_vllm.check.replay.index import IDS_MEMBER, SAMPLER_OP, SEED_PARAM
from verity_vllm.check.replay.opening import CommittedStore


LINKAGE_SCHEMA = "verity-sweep/boundary-linkage/v1"


def boundary_linkage(bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None], workload_requests: list[dict[str, Any]],
                     *, embed_op_path: str | None = None, splits_of_record: dict[str, Any] | None = None, store: Any = None,
                     eos_of_record: set[int] | list[int] | None = None, eos_lag_rows: int | None = None) -> dict[str, Any]:
    """Exhaustive ordered boundary linkage: for every declared request, committed `input_token_ids` at the arrival step == the declared
    prompt ids; at each later step == the request's committed `sampled_token_ids` of the previous step (i64 slot 0 of its block).
    `workload_requests`: [{"request_id": r<k>, "arrive_step", "prompt_token_ids", "steps": [engine steps], "seed": declared or None,
    "cached": C}]: a warm request's arrival step embeds prompt[C:] (its rows 0..C-1 are the producers', which the replay reads).
    Plus `prescribed_input_linkage` (seed / splits); `splits_of_record` = {request_id: {"derived": num_SMs} | other}.
    `store`: a CommittedStore-like reader (has/words) in place of (bmap, committed) -- tests.
    `eos_of_record`: the Program's declared stop rule (global_program.stop_rule eos_token_ids; None = undeclared): a request
    linked over FEWER steps than its declared cap must end on a declared EOS id, else unlinked by name.
    `eos_lag_rows`: the placeholder rows the ATTESTED scheduler runs after an EOS (RunHeader.profile.engine_kwargs
    .async_scheduling True -> 1, False -> 0; None = unattested -> every early stop is refused by name).  The EOS is verified at
    steps[-1 - lag]; the trailing `lag` step(s) are the placeholder(s) fed the EOS (their committed embed input == that EOS id, recorded).
    A read whose opening does not verify against the run root is an unlinked pair by name."""
    store = UnverifiedUnlinked(store if store is not None else CommittedStore(bmap, committed))
    # the embedding op_path: the (only) op with an `in` member in the map
    if embed_op_path is None:
        ops = sorted({k[2] for k in store.idx if k[3] == IDS_MEMBER})
        if len(ops) != 1:
            return {"schema": LINKAGE_SCHEMA, "result": None, "why": f"input_token_ids member bound under {len(ops)} op_paths: {ops[:3]}", "pairs": 0, "linked": 0, "unlinked": []}
        embed_op_path = ops[0]
    pairs, linked, unlinked, prompt_ok = 0, 0, [], {}
    for r in workload_requests:
        rid = str(r["request_id"])
        arrive = int(r.get("arrive_step") or 0)
        prompt = [int(t) for t in (r.get("prompt_token_ids") or [])][int(r.get("cached") or 0):]
        steps = [int(s) for s in (r.get("steps") or [])]
        if not steps:
            continue
        # arrival step: the prescribed prompt
        pairs += 1
        w, why = store.words(rid, arrive, embed_op_path, IDS_MEMBER, 0, len(prompt))
        if w is None:
            unlinked.append({"request_id": rid, "engine_step": arrive, "kind": "prompt", "why": why})
            prompt_ok[rid] = False
        else:
            got = [int(x) for x in np.asarray(w).astype(np.int64).tolist()]
            ok = got == prompt
            prompt_ok[rid] = ok
            if ok:
                linked += 1
            else:
                first = next((i for i, (a, b) in enumerate(zip(got, prompt)) if a != b), min(len(got), len(prompt)))
                unlinked.append({"request_id": rid, "engine_step": arrive, "kind": "prompt", "why": f"committed prompt ids != declared prompt at position {first}"})
        # each later step: the previous step's designated output
        for s in steps[1:]:
            pairs += 1
            w, why = store.words(rid, s, embed_op_path, IDS_MEMBER, 0, 1)
            sw, swhy = store.words(rid, s - 1, SAMPLER_OP, SAMPLED_MEMBER, 0, 1)
            if w is None or sw is None:
                unlinked.append({"request_id": rid, "engine_step": s, "kind": "chain", "why": why if w is None else swhy})
                continue
            tid, sid = int(np.asarray(w).astype(np.int64)[0]), int(np.asarray(sw).astype(np.int64)[0])
            try:
                circuit_token_id(sid)
            except ValueError as err:
                unlinked.append({"request_id": rid, "engine_step": s, "kind": "chain", "why": f"the committed sampled id of step {s - 1}: {err}"})
                continue
            if tid == sid:
                linked += 1
            else:
                unlinked.append({"request_id": rid, "engine_step": s, "kind": "chain", "why": f"committed input id {tid} != committed sampled id {sid} of step {s - 1}"})
    # the PRESCRIBED sampler inputs are linked to their approved
    # source here, independently of whether a replay sample exposes a substitution: the committed seed word == the DECLARED workload seed
    # of the request at every committed step; on a derived-S Program the committed splits word == SplitsFor_v1(committed live count, num_SMs
    # of record).  Their unlinked entries join `unlinked` (the same FAIL-by-name path); the boundary `pairs`/`linked` counts stay the boundary's.
    presc = prescribed_input_linkage(store, workload_requests, splits_of_record=splits_of_record)
    unlinked.extend(presc.pop("unlinked"))
    method = "ordered boundary linkage (declared prompt -> input_token_ids; sampled_token_ids[s-1] -> input_token_ids[s])"
    if presc["seed"].get("checked") or presc["splits"].get("checked"):
        method += " + prescribed-input linkage (" + "; ".join(k for k in ("seed", "splits") if presc[k].get("checked")) + ")"
    # the horizon each request was linked over, beside its declared cap and its Program of record's steps;
    # a horizon SHORTER than the declared cap is an early stop: the committed sampled id at its EOS step must be one
    # of the Program's declared EOS ids (stop_rule) -- Commit-side, not leaning on the Match chain's G6; undeclared -> fail closed.
    # the EOS step is steps[-1 - lag] where lag = the ATTESTED scheduler's placeholder rows after an EOS (async
    # scheduling runs ONE step after the EOS whose sample the driver discards: e.g. EOS 0 at step 10, 504 at step 11).
    # lag is the same header field GM-01 G6 (eos_lag_rows) and the lifted continuation's LAG read; unattested -> refuse by name, never default 0 or 1.
    horizon: dict[str, Any] = {}
    eos_set = None if eos_of_record is None else {int(x) for x in eos_of_record}
    lag = None if eos_lag_rows is None else int(eos_lag_rows)
    for r in workload_requests:
        if r.get("max_tokens") is None and r.get("program_steps") is None:
            continue
        rid = str(r["request_id"])
        steps = [int(s) for s in (r.get("steps") or [])]
        h: dict[str, Any] = {"linked_steps": len(steps), "declared_cap": r.get("max_tokens"), "program_steps": r.get("program_steps")}
        if r.get("max_tokens") is not None and steps and len(steps) < int(r["max_tokens"]):
            last = steps[-1]
            over = f"{rid} linked over {len(steps)} of its declared cap {r['max_tokens']} steps"
            e: dict[str, Any] = {"last_step": last, "lag": lag, "declared_eos": (sorted(eos_set)[:16] if eos_set is not None else None), "verified": False}
            w, why = store.words(rid, last, SAMPLER_OP, SAMPLED_MEMBER, 0, 1)
            e["sampled_id"] = None if w is None else int(np.asarray(w).astype(np.int64)[0])
            if lag is None:
                unlinked.append({"request_id": rid, "engine_step": last, "kind": "eos",
                                 "why": f"{over} and the record does not attest its scheduler's EOS lag (RunHeader.profile.engine_kwargs.async_scheduling unattested, "
                                        f"RULING 21): the EOS step cannot be named, refuse by name"})
            elif lag not in (0, 1):
                unlinked.append({"request_id": rid, "engine_step": last, "kind": "eos", "why": f"{over} but the attested EOS lag {lag} is not in {{0, 1}} (RULING 21): refuse by name"})
            elif len(steps) < lag + 1:
                unlinked.append({"request_id": rid, "engine_step": last, "kind": "eos", "why": f"{over}: fewer steps than the attested EOS lag {lag} + 1, no EOS step exists"})
            else:
                eos_step = steps[-1 - lag]
                e["eos_step"] = eos_step
                ew, ewhy = (w, why) if lag == 0 else store.words(rid, eos_step, SAMPLER_OP, SAMPLED_MEMBER, 0, 1)
                e["eos_id"] = None if ew is None else int(np.asarray(ew).astype(np.int64)[0])
                # the trailing lag step(s) are placeholders fed the EOS: their committed embed input is the EOS id (the chain above linked it)
                ph = steps[-lag:] if lag else []
                e["placeholder_steps"] = ph
                e["placeholder_input_ids"] = []
                for ps in ph:
                    pw, _pwhy = store.words(rid, ps, embed_op_path, IDS_MEMBER, 0, 1)
                    e["placeholder_input_ids"].append(None if pw is None else int(np.asarray(pw).astype(np.int64)[0]))
                if ew is None:
                    unlinked.append({"request_id": rid, "engine_step": eos_step, "kind": "eos", "why": f"{over} but the EOS step {eos_step}'s sampled id is not committed: {ewhy}"})
                elif eos_set is None:
                    unlinked.append({"request_id": rid, "engine_step": eos_step, "kind": "eos",
                                     "why": f"{over} and the Program declares no stop rule (eos_token_ids): the early stop cannot be verified, fail closed"})
                elif e["eos_id"] not in eos_set:
                    unlinked.append({"request_id": rid, "engine_step": eos_step, "kind": "eos",
                                     "why": f"{over} but its committed sampled id {e['eos_id']} at the EOS step {eos_step} (= last step {last} - lag {lag}) is not a declared EOS id {sorted(eos_set)[:8]}"})
                elif any(x != e["eos_id"] for x in e["placeholder_input_ids"]):
                    unlinked.append({"request_id": rid, "engine_step": ph[0], "kind": "eos",
                                     "why": f"{over}: the placeholder step(s) {ph} after the EOS at step {eos_step} are fed {e['placeholder_input_ids']}, not the EOS id {e['eos_id']}"})
                else:
                    e["verified"] = True
            h["eos_of_record"] = e
        horizon[rid] = h
    result = (pairs > 0 and not unlinked)
    return {"schema": LINKAGE_SCHEMA, "method": method,
            "result": result if pairs > 0 else None, "pairs": pairs, "linked": linked, "unlinked_n": len(unlinked), "unlinked": unlinked[:12],
            "prompt_linked": prompt_ok, "embed_op_path": embed_op_path, "prescribed": presc, **({"horizon": horizon} if horizon else {}),
            **({"prompt_cached": warm} if (warm := {str(r["request_id"]): int(r["cached"]) for r in workload_requests if r.get("cached")}) else {}),
            "why": None if result else (unlinked[0]["why"] if unlinked else "no (step, request) pairs to link")}


def arrivals_of_record(declared: dict[str, int], schedule: dict[str, Any] | None) -> tuple[dict[str, int], bool]:
    """The arrivals every C2 consumer (population placement, sampled replay, boundary linkage,
    CompiledKernelCheck) counts from: the ENGINE schedule of record's observed arrive steps when the population was placed on the observed
    committed-step axis (`schedule.use == "observed"`: every declared request observed, contiguous, exactly its cap), else the declaration.
    Returns (arrivals, moved) -- `moved` iff an arrival differs from the declaration.  A missing / failed schedule leaves the declaration."""
    out = {str(k): int(v) for k, v in (declared or {}).items()}
    if not isinstance(schedule, dict) or schedule.get("use") != "observed":
        return out, False
    obs = {str(k): int(v) for k, v in (schedule.get("observed_arrive_steps") or {}).items() if str(k) in out}
    moved = any(out[k] != v for k, v in obs.items())
    out.update(obs)
    return out, moved


def linkage_horizon(arrive: int, declared_cap: int, program_steps: int | None, executed: int | None = None) -> list[int]:
    """The engine steps a request is linked over: its Program OF RECORD's steps
    (T_r + 1: the served shape at B=1 eos-truncated / GP-01 composed at B>=2), capped by the declared cap; the cap alone when the Program's
    step count is unknown (older records).  A request whose committed steps are FEWER than its Program of record still fails by name
    (the chain rule inside the horizon is unchanged); the cap's never-executed steps are not 'unlinked' -- the Program never had them."""
    n = int(declared_cap) if program_steps is None else min(int(declared_cap), int(program_steps))
    # `executed`: the request's executed prefix of record (executed_prefix.executed_prefix_of_record: cap, or
    # served + LAG on an EOS stop under the DECLARED lag) -- at B>=2 GP-01 composes every request to its cap, so the Program's step count
    # is the cap's and the horizon is the executed prefix; boundary_linkage then verifies the EOS at steps[-1 - lag] and the placeholder
    # row(s) fed the EOS exactly as before.  None = the horizon `n` above.
    if executed is not None:
        n = min(n, max(int(executed), 0))
    return list(range(int(arrive), int(arrive) + max(n, 0)))


SEED_RULE = "committed runner.sampler/seed word == the DECLARED workload seed of the request (i64 exact) at every step with a committed sampled id (constant per request)"
SPLITS_RULE = "committed runner.sampler/splits word at step s == SplitsFor_v1(#requests with a committed sampled id at s, num_SMs of record) (REQ-03 derived S)"


def splits_of_record(loaded: dict) -> dict[str, Any]:
    """Each request Program's split count S of record (`{"derived": num_sms}` under derived S), from its Build result."""
    out = {}
    for k, v in loaded.items():
        smp = (v.get("result") or {}).get("sampling") or {}
        out[k] = {"derived": (smp.get("geometry") or {}).get("num_sms")} if smp.get("splits") == "derived" else smp.get("splits")
    return out


def _word_int(w: np.ndarray) -> int:
    a = np.asarray(w).reshape(-1)
    return int(a.view(np.int64)[0]) if a.dtype.itemsize == 8 else int(a.astype(np.int64)[0])


def prescribed_input_linkage(store: Any, workload_requests: list[dict[str, Any]], *, splits_of_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Commit-side linkage of the stochastic Program's PRESCRIBED inputs to their approved source.
    `workload_requests[i]["seed"]` = the request's DECLARED seed (the SamplingParams.seed the run was asked for; None on a greedy row);
    `splits_of_record` = {request_id: {"derived": num_SMs}} from the Build result's `sampling` (None / non-derived = no splits rule).
    SEED: for every (step, request) that committed a `runner.sampler/sampled_token_ids` word, the committed `runner.sampler/seed` word
    must exist and == the declared seed (i64 exact) -- the kernel reads seeds[req] + pos[req]; the seed is one word per request at every
    step (no per-step derivation).  SPLITS (derived S only): at every step with >= 1 committed sampled id, S* = SplitsFor_v1(live, num_SMs)
    with live = the number of requests with a committed sampled id at that step; every live request's committed splits word == S*.
    Missing word / unequal word / an unresolvable S -> unlinked BY NAME (kind 'seed' / 'splits').  A greedy row (no declared seed, no
    derived S) checks nothing and says so.  Returns {"seed": {...}, "splits": {...}, "unlinked": [...]}."""
    unlinked: list[dict[str, Any]] = []
    live_at: dict[int, list[str]] = {}
    seeded = 0
    seed_checked = seed_linked = 0
    for r in workload_requests:
        rid = str(r["request_id"])
        steps = [int(s) for s in (r.get("steps") or [])]
        live = [s for s in steps if store.has(rid, s, SAMPLER_OP, SAMPLED_MEMBER)]
        for s in live:
            live_at.setdefault(s, []).append(rid)
        declared = r.get("seed")
        if declared is None:
            continue
        seeded += 1
        declared = int(declared)
        if not live:
            unlinked.append({"request_id": rid, "engine_step": steps[0] if steps else None, "kind": "seed",
                             "why": f"{rid}: no committed sampled_token_ids at any of its {len(steps)} declared steps -- the declared seed {declared} is linked at no step"})
            continue
        for s in live:
            seed_checked += 1
            w, why = store.words(rid, s, SAMPLER_OP, SEED_PARAM, 0, 1)
            if w is None:
                unlinked.append({"request_id": rid, "engine_step": s, "kind": "seed", "why": f"prescribed seed of {rid} at step {s} not committed: {why}"})
                continue
            got = _word_int(w)
            if got == declared:
                seed_linked += 1
            else:
                unlinked.append({"request_id": rid, "engine_step": s, "kind": "seed",
                                 "why": f"committed runner.sampler/seed {got} != declared workload seed {declared} of {rid} at step {s} (prescribed input)"})
    seed_rec: dict[str, Any] = {"rule": SEED_RULE, "requests_declared": seeded, "checked": seed_checked, "linked": seed_linked}
    if seeded == 0:
        # 'not applicable' only when the record carries NO committed seed word: a stochastic run with no declared
        # seed (vLLM's own generator) that still bound runner.sampler/seed has a prescribed input with no approved source -> unlinked by name
        stray = [(rid, s) for s, rids in sorted(live_at.items()) for rid in rids if store.has(rid, s, SAMPLER_OP, SEED_PARAM)]
        if stray:
            rid0, s0 = stray[0]
            seed_rec["committed_without_source"] = len(stray)
            unlinked.append({"request_id": rid0, "engine_step": s0, "kind": "seed",
                             "why": f"committed runner.sampler/seed word at {len(stray)} (step, request) pair(s) (first {rid0} step {s0}) with NO declared workload seed: a prescribed input without an approved source"})
        else:
            seed_rec["why"] = "no declared seed on any request (greedy workload; no committed seed word): not applicable"
    # splits: derived S of record (one num_SMs across the requests' Programs; a disagreement is named, never averaged)
    nsms = set()
    derived_for: set[str] = set()
    for r in workload_requests:
        rid = str(r["request_id"])
        sp = (splits_of_record or {}).get(rid)
        if isinstance(sp, dict) and sp.get("derived") is not None:
            nsms.add(int(sp["derived"]))
            derived_for.add(rid)
    splits_rec: dict[str, Any] = {"rule": SPLITS_RULE, "requests_derived": len(derived_for), "checked": 0, "linked": 0, "steps": 0}
    if not derived_for:
        splits_rec["why"] = "no request Program declares derived S (constant S / greedy): not applicable"
    elif len(nsms) != 1:
        splits_rec["why"] = f"num_SMs of record disagree across the request Programs: {sorted(nsms)}"
        unlinked.append({"request_id": None, "engine_step": None, "kind": "splits", "why": "num_SMs of record disagree across the request Programs: " + str(sorted(nsms))})
    else:
        from verity.ml.definitions.registry.topp_geometry import splits_for_v1
        nsm = nsms.pop()
        splits_rec["num_sms"] = nsm
        sched: dict[int, int] = {}
        for s in sorted(live_at):
            rids = live_at[s]
            try:
                want = int(splits_for_v1(len(rids), nsm))
            except ValueError as e:
                unlinked.append({"request_id": None, "engine_step": s, "kind": "splits", "why": f"no S of record at step {s}: {e}"})
                continue
            sched[s] = want
            splits_rec["steps"] += 1
            for rid in rids:
                if rid not in derived_for:
                    unlinked.append({"request_id": rid, "engine_step": s, "kind": "splits", "why": f"{rid} is live at step {s} but its Program does not declare derived S"})
                    continue
                splits_rec["checked"] += 1
                w, why = store.words(rid, s, SAMPLER_OP, RM.SPLITS_PARAM, 0, 1)
                if w is None:
                    unlinked.append({"request_id": rid, "engine_step": s, "kind": "splits", "why": f"prescribed splits of {rid} at step {s} not committed: {why}"})
                    continue
                got = _word_int(w)
                if got == want:
                    splits_rec["linked"] += 1
                else:
                    unlinked.append({"request_id": rid, "engine_step": s, "kind": "splits",
                                     "why": f"committed runner.sampler/splits {got} != SplitsFor_v1(live {len(rids)}, num_SMs {nsm}) = {want} for {rid} at step {s} (prescribed input)"})
        splits_rec["schedule"] = {str(s): v for s, v in list(sched.items())[:64]}
    return {"seed": seed_rec, "splits": splits_rec, "unlinked": unlinked}
