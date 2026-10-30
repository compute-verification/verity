"""Value check of a stochastic (seeded temperature + top-p) row: recompute every snapshotted sampled row of the M1 capture
with the Program's own sampler Definition under the DECLARED inputs, and compare with the served token -- no tolerance.

For every snapshotted engine step with a `_gumbel_sample_kernel` launch (row_pod's Match snapshots steps `MATCH_SNAP_STEPS`, default 0,1):

    token', masked' = GumbelTopPTokenSelect_v1{V,S}(bf16 logits row, top_p, seed, pos, temperature)     # verity/ml/definitions/registry/sampling.py

with the bf16 logits row from the capture's snapshot of the sampler input (`SampleEnd.logits_view`), `seed` the request record's
DECLARED seed (`RunHeader.workload.requests[*].seed`, i.e. the workload manifest -- the Program's runtime input), `pos` the row's
StepStart position, temperature / top_p the declared policy, and S the split kernels' constexpr.  Compared, bit for bit:

  * `token' == SampleEnd.sampled_token_ids[row]` (the served token)                             -> rows[*].status exact-match / mismatch
  * `masked' == _gumbel_sample_kernel.logits_ptr[row]` (the fp32 row the kernel consumed)        -> rows[*].masked_row_equal_lanes == V
  * the kernel-consumed facts (`seeds_ptr[idx_mapping[row]]`, `temp_ptr[..]`, the mask kernel's `P[row]`, `pos_ptr[row]`) equal the
    declared ones; a fact that was NOT snapshotted at a checked row FAILS                      -> rows[*].device_facts_match
  * with `--build <row dir>`: the mask kernel's constexpr S equals the PROGRAM's S for the request's shape at this sampled event
    (Build result.json `sampling.splits`: 'derived' -> S_t = SplitsFor_v1(#logits rows at t, geometry.num_sms); a fixed int or
    per-event schedule otherwise)  -> rows[*].S_agree
  * every sampled row of the record is value-checked, else INCOMPLETE; the report's `structure_note` says what
    is DECLARED (fold Const/Param from the run header) vs OBSERVED (device-tied, here)

A wrong declared seed (manifest vs run) fails here by name: the kernel consumed another seed (device_facts_match False) AND the
recomputed token differs (mismatch, with probability ~1).  A mutated sampler chain (another kernel / another noise / another mask)
fails the masked-row or token comparison.  Fail closed: no snapshotted sampled row -> exit 2 (nothing was checked).

    PYTHONPATH=. verity-vllm stoch-recompute --run <match dir> [--out <json>]
    exit 0 PASS, 1 FAIL (named reasons in the report), 2 nothing checked

Implementation variant this checks against (recorded in the report): vLLM d9105ea80 V2 sampler with an explicit `SamplingParams.seed`
-> FlashInfer sampling disabled (`Sampler.sample`: `any_explicit_seed`), `torch.multinomial` never used; `_temperature_kernel` (div.full.f32)
-> `_apply_topp_split` (`_topp_sb_stats/step x5/mask`, S splits) -> `_gumbel_sample_kernel` (Philox `tl.rand(seed', pos*V+v)`, Gumbel
`-log(-log1p(-u))` via libdevice) + blockwise argmax.
"""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np
from dataclasses import dataclass

from verity_vllm.config import option, workload_arg, workload_file
from verity_vllm.records.workload import Workload

VARIANT = ("vllm d9105ea80 V2 sampler, explicit seed: FlashInfer disabled; _temperature_kernel -> _apply_topp_split(S) "
           "[_topp_sb_stats_kernel, _topp_sb_step_kernel x5, _topp_sb_mask_kernel] -> _gumbel_sample_kernel (Philox tl.rand + Gumbel) "
           "+ argmax/gather; torch.multinomial not used")
#: the prefix of a record entry naming an input file that is there but couldn't be read
UNREADABLE = "unreadable: "


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _request_facts(header) -> dict[str, dict]:
    wl = header.workload or {}
    s = dict(wl.get("sampling") or {})
    per = {p["request_id"]: p for p in s.get("per_request", []) or []}
    out: dict[str, dict] = {}
    for r in wl.get("requests", []) or []:
        p = per.get(r["request_id"], {})
        pick = lambda k, d: r.get(k) if r.get(k) is not None else (p.get(k) if p.get(k) is not None else s.get(k, d))  # noqa: E731
        out[r["request_id"]] = {"seed": None if r.get("seed") is None else int(r["seed"]), "temperature": float(pick("temperature", 0.0)),
                                "top_p": float(pick("top_p", 1.0)), "top_k": int(pick("top_k", 0) or 0)}
    for c in wl.get("children", []) or []:
        base = dict(out.get(c["parent"], {}))
        base["seed"] = None if c.get("seed") is None else int(c["seed"])
        out[c["request_id"]] = base
    return out


def _manifest_facts(path: str | Path) -> dict[str, dict]:
    """The DECLARED facts from the workload manifest itself (`workloads/<row>.json`: `variants[sampling.variant]` +
    per-request `seed`), independent of what the run's header recorded -- so a manifest that disagrees with the run is caught."""
    w = Workload.load(workload_file(path))
    p = w.variants.get(w.sampling_variant or "greedy") or {"temperature": 0.0, "top_p": 1.0, "top_k": 0}
    top = None if w.sampling is None else w.sampling.seed
    out = {}
    for r in w.requests:
        rp = w.variants.get(r.variant) if r.variant is not None else p
        s = r.seed if r.seed is not None else top
        out[r.request_id] = {"seed": s, "temperature": float(rp.get("temperature", 0.0)),
                             "top_p": float(rp.get("top_p", 1.0)), "top_k": int(rp.get("top_k", 0) or 0)}
    return out


def _eval_job(job: tuple) -> tuple[int, "np.ndarray", float]:
    """One sampled row through the Definition of record: GumbelTopPTokenSelect_v1{V,S} on (bf16 logits row, top_p, seed, pos, T) ->
    (token, the masked fp32 row it went through, seconds).  Module-level so a worker process can run it."""
    from verity_vllm.properties.admission import evaluate_spec
    from verity.ml.definitions.registry import sampling_topp_difftest as D
    V, S, words = job
    t1 = time.time()
    out = evaluate_spec(D.spec({"V": int(V), "S": int(S)}), words)
    tok = int(out[0])
    if tok & 0x80000000:
        tok -= 1 << 32
    return tok, np.array(out[1:1 + V], dtype=np.uint32), round(time.time() - t1, 3)


def _eval_job_rows(job: tuple) -> tuple[int, "np.ndarray", float]:
    """The same Definition over the whole row at once: `sampling_rows.gumbel_topp_token_select_row`
    (numpy twins of the registered gates, held word-equal to `evaluate_spec` in tests/program/test_sampling_rows.py) -> the same
    (token, masked fp32 row, seconds) triple.  The words layout is `_eval_job`'s: [bf16 row, top_p, seed, pos, temp] (uint64 arrays)."""
    from verity.ml.definitions.registry import sampling_rows as SR
    V, S, words = job
    t1 = time.time()
    bf, p_w, seed_w, pos_w, t_w = words
    out = SR.gumbel_topp_token_select_row(np.asarray(bf, dtype=np.uint64).astype(np.uint16), int(p_w[0]), int(seed_w[0]), int(pos_w[0]),
                                          int(t_w[0]), int(S))
    tok = int(out["token"])
    if tok & 0x80000000:
        tok -= 1 << 32
    return tok, np.asarray(out["masked"], dtype=np.uint32), round(time.time() - t1, 3)


EVALUATORS = {"lane": _eval_job, "rows": _eval_job_rows}
EVALUATOR_DOC = {"lane": "evaluate_spec: the registry interpreter walks GumbelTopPTokenSelect_v1{V,S} gate by gate over the V lanes",
                 "rows": "sampling_rows: numpy twins of the same registered gates over the whole row at once (verity/ml/definitions/registry/sampling_rows.py), "
                         "held word-equal to the lane walk by tests/program/test_sampling_rows.py and by the reference rows of this run"}


def _cpu_quota() -> int:
    """The CPUs this process may use: the cgroup quota when one is set (cpu.max), else os.cpu_count()."""
    import os
    try:
        q, per = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            return max(1, int(q) // int(per))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def _evaluate_all(jobs: list[tuple], n_jobs: int | None, evaluator: str = "lane") -> list[tuple[int, "np.ndarray", float]]:
    """Evaluate every pending row; `n_jobs` worker processes (fork; default min(32, the cgroup's CPU quota)) when there is more than one
    row.  `evaluator`: 'lane' (the interpreter, the path of record) or 'rows' (the vectorised twins; one row is milliseconds, so no pool)."""
    fn = EVALUATORS[evaluator]
    if evaluator == "rows":
        return [fn(j) for j in jobs]
    n = int(n_jobs) if n_jobs else min(32, _cpu_quota())
    if len(jobs) <= 1 or n <= 1:
        return [fn(j) for j in jobs]
    from verity_vllm.check.replay.fork import fork_pool
    with fork_pool(min(n, len(jobs))) as pool:
        return list(pool.imap(fn, jobs, chunksize=1))


def _program_splits(build_dir: str | Path | None, workload: str | Path | None) -> tuple[dict[str, list[int] | int | dict | None], list[str]]:
    """The PROGRAM's S per request: `sampling.splits` of the Build's `result.json` for the request's (LP, T) shape --
    `"derived"` (the Program takes S as its `splits` Input and declares the relation S_t = SplitsFor_v1(|live(t)|,
    num_SMs); returned as {"derived": num_sms} so the check derives S_t from the record's live rows at step t), else a fixed int / per-event
    schedule.  ({request_id: splits}, unreadable): a request whose shape has no Build result maps to None, and each Build result that
    could not be read is named with its error."""
    if build_dir is None or workload is None:
        return {}, []
    import glob
    by_shape: dict[tuple[int, int], list[int] | int | dict] = {}
    unreadable: list[str] = []
    for p in sorted(glob.glob(str(Path(build_dir) / "build_request*" / "result.json"))):
        try:
            r = json.loads(Path(p).read_text())
        except Exception as e:
            unreadable.append(f"{p}: {e!r}"[:300])
            continue
        if not r.get("ok", True) or r.get("wrapper") != "request":
            continue
        smp = r.get("sampling") or {}
        sp = smp.get("splits")
        if sp == "derived":
            nsm = (smp.get("geometry") or {}).get("num_sms")
            by_shape[(int(r["LP"]), int(r["T"]))] = {"derived": int(nsm)} if nsm is not None else {"derived": None}
        elif sp is not None:
            by_shape[(int(r["LP"]), int(r["T"]))] = sp
    out: dict[str, list[int] | int | None] = {}
    for q in Workload.load(workload_file(workload)).requests:
        out[q.request_id] = by_shape.get((q.prompt_length or 0, q.max_tokens - 1))
    return out, unreadable


def _stop_rule(build_dir: str | Path | None) -> dict:
    """The Program's declared stop rule (GP-01 `stop_rule`: eos_token_ids, eos_lag_rows) from the Build's workload_program.json, if given;
    {"source", "error"} when the file is there but can't be read."""
    if build_dir is None:
        return {}
    for p in (Path(build_dir) / "build_workload" / "workload_program.json", Path(build_dir) / "build_workload_program.json.gz"):
        if not p.exists():
            continue
        try:
            if p.suffix == ".gz":
                import gzip
                with gzip.open(p, "rt") as f:
                    gp = json.loads(f.read())
            else:
                gp = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            return {"source": str(p), "error": f"{type(e).__name__}: {e}"[:200]}
        return dict(gp.get("stop_rule") or {}) | {"source": str(p)}
    return {}


def record_lag_rows(header, snapshots_dir: Path | None = None) -> tuple[int | None, str]:
    """The scheduler's EOS lag rows read from the RECORD's own attestation of `async_scheduling` (True -> 1,
    False -> 0): RunHeader.profile.engine_kwargs, RunHeader.engine_kwargs, capture/versions.json engine_kwargs -- the same locations GM-01
    reads (global_match.record_engine_facts); None when the record does not attest it (the lag slot is then unjudgeable, said so), with
    an `unreadable: ` reason when versions.json is there but can't be read (the sample ledger's problem)."""
    cands: list[tuple[str, object]] = []
    prof = getattr(header, "profile", None) if not isinstance(header, dict) else header.get("profile")
    prof = prof if isinstance(prof, dict) else {}
    ek = prof.get("engine_kwargs") if isinstance(prof.get("engine_kwargs"), dict) else {}
    cands.append(("RunHeader.profile.engine_kwargs.async_scheduling", ek.get("async_scheduling")))
    ek_top = getattr(header, "engine_kwargs", None) if not isinstance(header, dict) else header.get("engine_kwargs")
    if isinstance(ek_top, dict):
        cands.append(("RunHeader.engine_kwargs.async_scheduling", ek_top.get("async_scheduling")))
    unread = None
    vp = Path(snapshots_dir) / "versions.json" if snapshots_dir is not None else None
    if vp is not None and vp.exists():
        try:
            v = json.loads(vp.read_text())
            ekv = v.get("engine_kwargs") if isinstance(v.get("engine_kwargs"), dict) else {}
            cands.append(("capture/versions.json engine_kwargs.async_scheduling", ekv.get("async_scheduling")))
        except (OSError, ValueError) as e:
            unread = f"{UNREADABLE}capture/versions.json ({type(e).__name__}: {e})"[:240] + " -- its engine_kwargs.async_scheduling is not read"
    for src, v in cands:
        if isinstance(v, bool):
            return (1 if v else 0), f"{src}={v!r}"
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            return (1 if v.strip().lower() == "true" else 0), f"{src}={v!r}"
    return None, unread or "not attested (async_scheduling absent from the record's engine_kwargs)"


def expected_executed_slots(served: list, cap: int, A: int | None, eos_ids: set | None) -> tuple[list[int] | None, dict]:
    """[R3's one mechanism] the executed set {t : r_t = 1} (t from 1) the Program's stop rule PREDICTS for one request from
    its SERVED tokens (draw k serves slot k + 1), its cap C and the scheduler profile A (0 = sync, 1 = async: one lag slot after an EOS):
    r_1 = 1; while no served token is an EOS, every slot up to C executes; the first EOS at slot t_eos (before C) is followed by exactly
    min(A, C - t_eos) executed lag slot(s) and nothing after -- the same rule as `verity_vllm.program.lifting.continuation.recurrence` /
    `executed_slots` (spec s7 three-state machine; n_executed = min(n + 1, C) under A = 1) and the executed-prefix rule (`commit.executed_prefix`).  Returns
    None (with the reason) when the rule cannot be evaluated: an undeclared EOS set with fewer draws than the cap, or an EOS before the cap
    with the profile unattested.  When the continuation module is importable the two computations are compared and a disagreement is named."""
    from verity_vllm.program.lifting import continuation as CT  # the Continuation_v1 recurrence: THE mechanism, not a twin of it

    info: dict = {"rule": "verity_vllm.program.lifting.continuation.recurrence / executed_slots (pad, spec s7): EOS at slot t < C -> min(A, C - t) lag slot(s), then inactive"}
    served = list(served)
    if eos_ids is None:
        if len(served) < cap:
            return None, info | {"unjudged": "the Program declares no stop_rule.eos_token_ids: an early stop cannot be read as an EOS stop"}
        eos_ids, t_eos = set(), None
    else:
        t_eos = next((k + 1 for k, y in enumerate(served[:cap]) if y is not None and int(y) in eos_ids), None)
    if t_eos is not None and t_eos < cap and A is None:
        return None, info | {"unjudged": f"EOS served at slot {t_eos} before the cap {cap} and the scheduler profile (eos_lag_rows / "
                                         f"async_scheduling) is unattested: the lag slot cannot be judged"}
    if A is not None and int(A) not in (0, 1):
        return None, info | {"unjudged": f"scheduler profile A = {A} is not an admitted profile (LAG in (0, 1)); the recurrence refuses it"}
    # served[:cap]: the recurrence runs over the C slots (`expected_draw_positions` does the same); draws past C are EXTRA positions for
    # the comparison to name, never a reason to stop judging.  A = 0 when no EOS is served before the cap: the lag term is
    # then never reached, so the profile does not matter.
    slots = CT.recurrence([None if y is None else int(y) for y in served[:cap]], cap=cap, A=int(A) if A is not None else 0, eos=eos_ids)
    exp = [int(t) for t in CT.executed_slots(slots)]
    info["t_eos"] = t_eos
    info["n_executed"] = len(exp)
    return exp, info


def draw_accounting(draws_seen: dict[str, list[dict]], workload: str | Path | None, build_dir: str | Path | None,
                    lag_attested: tuple[int | None, str] | None = None) -> dict:
    """The sample LEDGER of the record, per request, measured --
    not stated: 'RNG advances iff r_t' read LITERALLY as an ordered-set comparison of the DRAWN positions against the positions the
    Program's rule says are executed:

      * draw k of a request (its k-th sampled row, k from 0; slot t = k + 1) is the TopPTokenSelect at pos = prompt_len - 1 + k (the query
        position whose logits are sampled; the Philox counter is (seed, pos), no state);
      * the rule (`expected_executed_slots`, R3's one mechanism): every slot up to the cap C executes while no served token is an EOS; the
        first EOS at slot t < C is followed by exactly min(A, C - t) lag slot(s) (A = the record's attested async_scheduling, 0/1) and
        nothing after; a request that runs to C has no lag slot (nothing is scheduled past the length bound);
      * expected positions = {prompt_len - 1 + (t - 1) : r_t = 1}; every drawn position not in the set is 'the RNG advanced without r_t'
        (a sample past the cap without an EOS, a second draw after the lag slot, a sample for another row's counter), every expected position
        not sampled is 'an executed slot did not sample' (a deleted step); one sample per engine step, in order;
      * an undeclared stop rule with an early stop, or an EOS before the cap with the profile unattested, leaves the request UNJUDGED --
        named, never a pass.

    Returns the ledger + the list of PROBLEMS (each a FAIL reason by name) and whether the lag slot was EXERCISED on this record at all
    (when it was not, the record half of 'RNG advances iff r_t' stays 'stated, not measured on this record', said so)."""
    out: dict = {"convention": "draw k of a request = its k-th sampled row (k from 0) = slot t = k + 1 at pos = prompt_len - 1 + k, keyed Philox(seed, pos); "
                               "expected draws = {pos of slot t : r_t = 1} under the Program's stop rule (EOS at slot t < C -> min(A, C - t) lag slot(s))",
                 "requests": {}, "problems": [], "lag_slot_exercised": False, "requests_with_eos_before_cap": 0, "requests_at_cap": 0,
                 "requests_unjudged": 0, "stop_rule": {}, "measured": None}
    wl_req: dict[str, dict] = {}
    if workload is not None:
        try:
            for r in Workload.load(workload_file(workload)).requests:
                wl_req[r.request_id] = {"prompt_len": r.prompt_length or 0, "cap": r.max_tokens}
        except (OSError, ValueError) as e:  # noqa: BLE001
            out["problems"].append(f"draw ledger: the workload manifest could not be read ({e!r}): prompt_len / cap unknown, the ledger is unjudged")
    sr = _stop_rule(build_dir)
    if sr.get("error"):
        out["problems"].append(f"draw ledger: {UNREADABLE}the Program's stop rule {sr['source']} ({sr['error']}): its EOS ids and declared lag are unknown")
    eos_ids = set(int(x) for x in sr["eos_token_ids"]) if sr.get("eos_token_ids") else None
    # the lag: the RECORD's attestation of async_scheduling (the same fact GM-01 G6 reads), the Program's declared
    # stop_rule.eos_lag_rows beside it when it carries one; a disagreement is GM-01's to fail, named here
    lag_rec, lag_src = lag_attested if lag_attested is not None else (None, "not read")
    if lag_rec is None and str(lag_src).startswith(UNREADABLE):
        out["problems"].append(f"draw ledger: the record's lag attestation is {lag_src}")
    lag_decl = sr.get("eos_lag_rows")
    lag = lag_rec if lag_rec is not None else (int(lag_decl) if lag_decl is not None else None)
    out["stop_rule"] = {"eos_token_ids": sorted(eos_ids) if eos_ids else None, "eos_ids_source": sr.get("source"),
                        "eos_lag_rows": lag, "eos_lag_rows_attested": lag_rec, "eos_lag_source": lag_src, "eos_lag_rows_declared": lag_decl,
                        "attested": eos_ids is not None}
    if lag_rec is not None and lag_decl is not None and int(lag_decl) != lag_rec:
        out["problems"].append(f"draw ledger: the Program declares stop_rule.eos_lag_rows {lag_decl} but the record attests {lag_src} -> {lag_rec}")
    if lag is not None and int(lag) not in (0, 1):
        out["problems"].append(f"draw ledger: scheduler profile A = {lag} is not 0 or 1 (check_lag: one rule, no other profile)")
    probs = out["problems"]
    for rid, dr in sorted(draws_seen.items()):
        dr = sorted(dr, key=lambda x: (x["step"], x["row"]))
        w = wl_req.get(rid)
        ent: dict = {"draws": len(dr), "first_step": dr[0]["step"], "last_step": dr[-1]["step"], "first_pos": dr[0]["pos"], "last_pos": dr[-1]["pos"],
                     "prompt_len": w["prompt_len"] if w else None, "cap": w["cap"] if w else None}
        steps = [x["step"] for x in dr]
        ent["one_draw_per_step"] = len(set(steps)) == len(steps)
        if not ent["one_draw_per_step"]:
            probs.append(f"draw ledger {rid}: two sampled rows in one engine step ({[s for s in steps if steps.count(s) > 1][:3]}): a request draws once per step")
        if not w:
            out["requests"][rid] = ent
            continue
        L, T = w["prompt_len"], w["cap"]
        drawn = [int(x["pos"]) for x in dr]
        ent["positions_in_order"] = drawn == sorted(drawn) and len(set(drawn)) == len(drawn)
        if not ent["positions_in_order"]:
            probs.append(f"draw ledger {rid}: drawn positions are not strictly increasing with the engine step ({drawn[:6]}): a draw repeated or out of order")
        served = [x["served"] for x in dr]
        exp_slots, info = expected_executed_slots(served, T, lag, eos_ids)
        ent["rule"] = info
        if exp_slots is None:
            ent["finish"] = "unjudged"
            out["requests_unjudged"] += 1
            probs.append(f"draw ledger {rid}: {len(dr)} draws for cap {T} -- {info['unjudged']}")
            out["requests"][rid] = ent
            continue
        t_eos = info.get("t_eos")
        # the RULE's positions are those of `expected_draw_positions`: slot t at pos L - 1 + (t - 1), the query position whose
        # logits slot t samples = the record's own `pos` fact (device_facts_match on every checked row; qwen B8 L=1024: first draw 1023);
        # served[:T] and A = 0 without an EOS as in expected_executed_slots (the same recurrence; a disagreement between the two calls is named)
        from verity_vllm.program.lifting import continuation as CT
        if any(y is None for y in served[:T]):
            ent["finish"] = "unjudged"
            out["requests_unjudged"] += 1
            probs.append(f"draw ledger {rid}: a sampled row without its served token: the rule cannot be read for this request")
            out["requests"][rid] = ent
            continue
        pad_pos = CT.expected_draw_positions([int(y) for y in served[:T]], cap=T, A=int(lag) if lag is not None else 0,
                                             eos=eos_ids if eos_ids is not None else set(), prompt_len=L)
        expected = [int(p) for p in pad_pos["positions"]]
        if pad_pos["slots"] != exp_slots:
            probs.append(f"draw ledger {rid}: expected_draw_positions slots {pad_pos['slots'][:4]}.. != executed_slots {exp_slots[:4]}..: one recurrence, two answers")
        ent["lag_positions"] = pad_pos["lag_positions"]
        ent["first_forbidden_position"] = pad_pos["first_forbidden_position"]
        cmp = CT.compare_draw_positions(expected, drawn)          # the continuation's comparison: extra / missing / repeated / out-of-order BY NAME
        extra, missing = cmp["extra"], cmp["missing"]
        ent["expected_executed_slots"] = {"n": len(exp_slots), "first": exp_slots[0], "last": exp_slots[-1]}
        ent["expected_positions"] = {"n": len(expected), "first": expected[0], "last": expected[-1]}
        ent["drawn_not_expected"] = extra
        ent["expected_not_drawn"] = missing
        ent["repeated"] = cmp["repeated"]
        ent["ordered_set_equal"] = cmp["ok"]
        ent["compare"] = "verity_vllm.program.lifting.continuation.compare_draw_positions"
        if cmp["ok"] is False and not extra and not missing:      # repeated / out of order: the continuation's names, ours to carry
            probs.extend(f"draw ledger {rid}: {p}" for p in cmp["problems"])
        if t_eos is None or t_eos >= T:
            ent["finish"] = "length" if t_eos is None else "eos-at-cap"
            ent["lag_slot"] = ("not scheduled: the request ran to its cap (nothing is scheduled past the length bound)" if t_eos is None else
                               f"not scheduled: the EOS is served at the cap slot {t_eos} (no slot after it)")
            out["requests_at_cap"] += 1
        else:
            ent["finish"] = "eos"
            out["requests_with_eos_before_cap"] += 1
            out["lag_slot_exercised"] = True
            n_lag = min(int(lag), T - t_eos)
            k_eos = t_eos - 1
            ent["served_eos"] = {"draw_k": k_eos, "slot": t_eos, "step": dr[k_eos]["step"], "pos": dr[k_eos]["pos"], "token": dr[k_eos]["served"]}
            after = dr[k_eos + 1:]
            ent["draws_after_served_eos"] = len(after)
            ent["lag_slot"] = {"profile_A": int(lag), "lag_slots_expected": n_lag,
                               "bounded_by_cap": n_lag < int(lag),
                               "draws": [{"slot": t_eos + 1 + j, "pos_expected": L + k_eos + j, "pos": x["pos"], "step": x["step"],
                                          "pos_ok": x["pos"] == L + k_eos + j} for j, x in enumerate(after)],
                               "measured": drawn == expected}
        if extra:
            what = ("a draw past the length bound without an EOS" if t_eos is None else
                    ("a draw past the EOS's lag slot" if all(p > expected[-1] for p in extra) else "a draw at a position the rule does not execute"))
            probs.append(f"draw ledger {rid}: {len(extra)} drawn position(s) {extra[:4]} outside the executed set (slots 1..{exp_slots[-1]} of cap {T}"
                         + (f", EOS at slot {t_eos}, {min(int(lag), T - t_eos)} lag slot(s)" if t_eos is not None and t_eos < T else "")
                         + f"): {what} -- the RNG advanced without r_t")
        if missing:
            probs.append(f"draw ledger {rid}: {len(missing)} executed slot(s) did not draw (positions {missing[:4]} of the expected "
                         f"{expected[0]}..{expected[-1]}): a deleted step" + (" -- the lag slot did not draw" if t_eos is not None and t_eos < T and missing[-1] == expected[-1] else ""))
        out["requests"][rid] = ent
    n_ex = out["requests_with_eos_before_cap"]
    out["measured"] = (f"R5 MEASURED on this record: {n_ex} request(s) stopped at EOS before the cap; the drawn positions of every request equal "
                       f"the rule's executed set as an ordered set (EOS at slot t -> exactly min(A={lag}, C - t) lag draw(s), none after)"
                       if out["lag_slot_exercised"] and not probs else
                       ("R5 NOT MEASURED on this record: no request stopped at EOS before its cap (every request ran to its length bound: no lag "
                        "slot was scheduled) -- the ledger checks drawn positions == the executed set {prompt_len - 1 + k : k < cap} as an ordered "
                        "set, one draw per step; the lag-slot half stays stated by construction (rng.advances_iff_executed), not measured here"
                        if not out["lag_slot_exercised"] and not probs else
                        f"R5 FAILS on this record: {len(probs)} ledger problem(s) named in reasons"))
    return out


def recompute_run(log_path: Path, snapshots_dir: Path, max_rows: int | None = None, workload: str | Path | None = None,
                  jobs: int | None = None, build_dir: str | Path | None = None, evaluator: str = "lane",
                  reference_rows: int = 0) -> dict:
    from verity_vllm.program import sampling_rng as R
    from verity_vllm.check.fold_compare import SnapshotStore
    from verity_vllm.properties.admission import evaluate_spec
    from verity_vllm.observe.events import SampleEnd, StepStart, TritonLaunch, ViewDesc
    from verity_vllm.observe.log import read_log
    from verity.ml.definitions.registry import sampling_topp_difftest as D

    t0 = time.time()
    events = list(read_log(log_path))
    header = events[0]
    run_facts = _request_facts(header)                     # what the run was GIVEN (the capture header's request records)
    facts = dict(run_facts)
    header_vs_manifest: list[str] = []
    if workload is not None:
        # the manifest of record is the authority for the declared inputs; the header must agree with it (else FAIL, named)
        mf = _manifest_facts(workload)
        for rid, f in mf.items():
            h = run_facts.get(rid)
            if h is not None:
                for k in ("seed", "temperature", "top_p", "top_k"):
                    a, b = f.get(k), h.get(k)
                    same = (a == b) if k in ("seed", "top_k") else (a is not None and b is not None and _f32_bits(a) == _f32_bits(b))
                    if not same:
                        header_vs_manifest.append(f"{rid}: manifest declares {k}={a!r} but the run's header records {k}={b!r}")
        facts = {rid: dict(run_facts.get(rid, {}), **f) for rid, f in mf.items()} or facts
    store = SnapshotStore(snapshots_dir)
    per_step: dict[int, dict] = {}
    cur: StepStart | None = None
    for ev in events[1:]:
        if isinstance(ev, StepStart):
            cur = ev
            per_step[ev.step] = {"start": ev, "gumbel": None, "mask": None, "topp": [], "temp": None, "sample_end": None}
        elif isinstance(ev, TritonLaunch) and ev.phase == "sample" and cur is not None:
            k = ev.fn.rsplit(".", 1)[-1]
            d = per_step[ev.step]
            if k == "_gumbel_sample_kernel":
                d["gumbel"] = ev
            elif k == "_topp_sb_mask_kernel":
                d["mask"] = ev
                d["topp"].append(k)
            elif k in ("_topp_sb_stats_kernel", "_topp_sb_step_kernel"):
                d["topp"].append(k)
            elif k == "_temperature_kernel":
                d["temp"] = ev
        elif isinstance(ev, SampleEnd) and ev.step in per_step:
            per_step[ev.step]["sample_end"] = ev
    rows: list[dict] = []
    pending: list[tuple[int, tuple, np.ndarray | None]] = []      # (row index, (V, S, words), observed masked row)
    reasons: list[str] = list(header_vs_manifest)
    tally = {"exact-match": 0, "mismatch": 0, "no-snapshot": 0, "skipped": 0}
    rows_without_seed = 0             # skipped rows whose request record declares no seed: a FAIL, not an INCOMPLETE
    steps_seen = 0
    sampled_rows_total = 0            # every sampled row of the record counts; the verdict names k of n value-checked
    program_S, splits_unreadable = _program_splits(build_dir, workload)      # the Program's S per request (int or per-event schedule), if a Build is given
    reasons.extend(f"Build result unreadable ({u}): the Program's S of its request shape is not read" for u in splits_unreadable)
    events_of: dict[str, int] = {}                         # sampled events attributed to each request so far (the event index of the next row)
    s_disagree: list[str] = []
    chain_absent: list[str] = []      # engine steps whose sampler chain is incomplete in the record are NAMED, not dropped
    engine_steps = len(per_step)
    draws_seen: dict[str, list[dict]] = {}                 # every sampled row of every request, snapshot or not: the sample ledger
    for step, d in sorted(per_step.items()):
        g, se, ss = d["gumbel"], d["sample_end"], d["start"]
        _lrows = list(getattr(ss, "logits_rows", None) or ())
        _srv = list(getattr(se, "sampled_token_ids", None) or ()) if se is not None else []
        for _row, (_rid, _pos) in enumerate(_lrows):
            draws_seen.setdefault(str(_rid), []).append({"step": int(step), "row": int(_row), "pos": int(_pos),
                                                         "served": int(_srv[_row]) if _row < len(_srv) else None})
        if g is None or se is None or d["mask"] is None:
            # the denominator is the RECORD's sampled rows (StepStart.logits_rows at every engine step), not the tool's
            # view of which steps carry a complete chain: a step whose gumbel launch / top-p mask / SampleEnd is absent still counts
            # its rows as sampled-and-unchecked, and is named
            n_abs = len(getattr(ss, "logits_rows", None) or ()) or len(list(getattr(se, "sampled_token_ids", None) or ()))
            sampled_rows_total += n_abs
            missing = [nm for nm, present in (("_gumbel_sample_kernel launch", g is not None), ("_topp_sb_mask_kernel launch", d["mask"] is not None),
                                              ("SampleEnd", se is not None)) if not present]
            chain_absent.append(f"step {step}: sampler chain incomplete in the record ({', '.join(missing)} absent); {n_abs} sampled row(s) not value-checked")
            rows.append({"step": step, "status": "chain-absent", "missing": missing, "rows": n_abs})
            continue
        steps_seen += 1
        n_rows_step = len(getattr(ss, "logits_rows", None) or ()) or len(list(getattr(se, "sampled_token_ids", None) or ()))
        sampled_rows_total += n_rows_step
        bf = getattr(se, "logits_view", None)
        lv = g.args.get("logits_ptr")
        if not isinstance(bf, ViewDesc) or not isinstance(lv, ViewDesc):
            tally["skipped"] += 1
            rows.append({"step": step, "status": "skipped", "why": "no logits views recorded", "rows": n_rows_step})
            reasons.append(f"step {step}: {n_rows_step} sampled row(s) not value-checked (no logits views recorded)")
            continue
        bsnap = store.find(bf, before=g.id + 1)
        msnap = store.find(lv, at=g.id) or store.find(lv, before=g.id + 1)
        if bsnap is None:
            tally["no-snapshot"] += 1
            rows.append({"step": step, "status": "no-snapshot", "rows": n_rows_step})
            reasons.append(f"step {step}: {n_rows_step} sampled row(s) not value-checked (the logits row fed to the sampler was not snapshotted)")
            continue
        bf_words = np.asarray(store.extract(bsnap, bf)).reshape(bf.shape)
        masked_obs = R._as_f32(np.asarray(store.extract(msnap, lv)), lv.dtype).reshape(lv.shape) if msnap is not None else None
        R_, V = bf.shape
        cx = dict(g.constexprs or {})
        S = None
        if d["mask"] is not None:
            S = (d["mask"].constexprs or {}).get("S")
        seeds_snap = R._small_int_arg(store, g, "seeds_ptr")
        pos_snap = R._small_int_arg(store, g, "pos_ptr")
        temp_snap = R._small_arg(store, g, "temp_ptr", np.float32)
        map_snap = R._small_int_arg(store, g, "expanded_idx_mapping_ptr")
        p_snap = None
        if d["mask"] is not None:                                   # the split kernels' signature names the top-p buffer `P`
            p_snap = R._small_arg(store, d["mask"], "P", np.float32)
            if p_snap is None:
                p_snap = R._small_arg(store, d["mask"], "p_ptr", np.float32)
        served = list(se.sampled_token_ids)
        for row, (rid, pos_fact) in enumerate(ss.logits_rows):
            if max_rows is not None and len(rows) >= max_rows:
                break
            f = facts.get(rid)
            ev_i = events_of.get(rid, 0)                  # this row is request `rid`'s ev_i-th sampled event (prefill = event 0)
            events_of[rid] = ev_i + 1
            rec = {"step": step, "row": row, "request_id": rid, "event": ev_i, "pos": int(pos_fact),
                   "served": int(served[row]) if row < len(served) else None,
                   "V": int(V), "S": S, "chain": list(d["topp"]) + (["_temperature_kernel"] if d["temp"] is not None else []),
                   "gumbel_constexprs": {k: v for k, v in cx.items()}}
            if rid in program_S:      # record BOTH the kernel's S (mask constexpr) and the Program's S; name a disagreement
                sp = program_S[rid]
                if isinstance(sp, dict):       # derived: S_t = SplitsFor_v1(|live(t)|, num_SMs), |live(t)| = the record's logits rows at t
                    from verity.ml.definitions.registry.topp_geometry import splits_for as _sf
                    nsm = sp.get("derived")
                    pS = int(_sf(int(n_rows_step), int(nsm))) if nsm is not None and 1 <= int(n_rows_step) <= 64 else None
                    rec["program_S"] = pS
                    rec["S_source"] = {"kernel": "_topp_sb_mask_kernel constexpr S",
                                       "program": f"derived: SplitsFor_v1(live={int(n_rows_step)}, num_SMs={nsm}) (Build sampling.splits 'derived')"}
                    if pS is None:
                        s_disagree.append(f"step {step} row {row} ({rid}): the Program's S is derived but num_SMs is undeclared or live {n_rows_step} "
                                          f"is outside 1..64 -- S_t cannot be derived")
                    elif S is not None and int(S) != pS:
                        s_disagree.append(f"step {step} row {row} ({rid}) event {ev_i}: the mask kernel ran S={int(S)} but SplitsFor_v1(live={int(n_rows_step)}, "
                                          f"num_SMs={nsm}) = {pS} -- the record's split geometry is not the derived one")
                    rec["S_agree"] = (S is not None and pS is not None and int(S) == pS)
                    continue_check = False
                else:
                    continue_check = True
                if continue_check:
                    pS = None if sp is None else (int(sp[ev_i]) if isinstance(sp, list) and ev_i < len(sp) else (int(sp) if isinstance(sp, int) else None))
                    rec["program_S"] = pS
                    rec["S_source"] = {"kernel": "_topp_sb_mask_kernel constexpr S", "program": "Build result.json sampling.splits" +
                                       (f"[event {ev_i}]" if isinstance(sp, list) else "")}
                    if sp is None:
                        s_disagree.append(f"step {step} row {row} ({rid}): the Build has no request Program for this request's shape -- its S is undeclared")
                    elif pS is None:
                        s_disagree.append(f"step {step} row {row} ({rid}): sampled event {ev_i} beyond the Program's {len(sp)}-event schedule")
                    elif S is not None and int(S) != pS:
                        s_disagree.append(f"step {step} row {row} ({rid}) event {ev_i}: the mask kernel ran S={int(S)} but the Program declares "
                                          f"TopPMask S={pS} (Build sampling.splits) -- the record's split geometry is not the declared one")
                    rec["S_agree"] = (S is not None and pS is not None and int(S) == pS)
            if f is None or f["seed"] is None:
                rec.update(status="skipped", why="request record has no declared seed")
                tally["skipped"] += 1
                rows_without_seed += 1
                rows.append(rec)
                reasons.append(f"step {step} row {row}: no declared seed for {rid!r}")
                continue
            if S is None or any(bool(cx.get(k)) for k in ("IS_DRAFTING", "APPLY_TEMPERATURE", "PER_TOKEN_COL", "USE_FP64")) or f["top_k"] not in (0, -1):
                rec.update(status="skipped", why=f"not the seeded temperature + p-only split chain (S={S}, flags={cx}, top_k={f['top_k']})")
                tally["skipped"] += 1
                rows.append(rec)
                reasons.append(f"step {step} row {row}: {rec['why']}")
                continue
            T, p, seed = f["temperature"], f["top_p"], int(f["seed"])
            rec.update(declared={"seed": seed, "temperature": T, "top_p": p, "pos": int(pos_fact)})
            # what the kernels consumed (snapshotted small buffers), against the declared facts
            idx = int(map_snap[row]) if map_snap is not None and row < len(map_snap) else None
            dev = {"seed": int(seeds_snap[idx]) if (seeds_snap is not None and idx is not None and idx < len(seeds_snap)) else None,
                   "pos": int(pos_snap[row]) if (pos_snap is not None and row < len(pos_snap)) else None,
                   "temperature": float(temp_snap[idx]) if (temp_snap is not None and idx is not None and idx < len(temp_snap)) else None,
                   "top_p": float(p_snap[row]) if (p_snap is not None and row < len(p_snap)) else None}
            rec["device"] = dev
            mism = [k for k, v in dev.items() if v is not None and
                    ((k in ("seed", "pos") and int(v) != int(rec["declared"][k])) or
                     (k in ("temperature", "top_p") and _f32_bits(v) != _f32_bits(rec["declared"][k])))]
            unsnap = [k for k, v in dev.items() if v is None]
            rec["device_facts_match"] = not mism and not unsnap
            rec["device_facts_checked"] = [k for k, v in dev.items() if v is not None]
            rec["device_facts_not_snapshotted"] = unsnap
            if unsnap:      # the token alone is not evidence of the declared fact; the snapshotted device fact is
                reasons.append(f"step {step} row {row} ({rid}): the kernel-consumed {', '.join(unsnap)} not snapshotted -- the declared "
                               f"value cannot be tied to the execution at this step")
            if mism:
                got = ", ".join(f"{k}={dev[k]!r}" for k in mism)
                want = ", ".join(f"{k}={rec['declared'][k]!r}" for k in mism)
                reasons.append(f"step {step} row {row} ({rid}): the sampler kernels consumed {got} but the manifest declares {want}")
            # the Program's sampler Definition under the DECLARED inputs (evaluated below, `jobs` rows at a time: the registry evaluator
            # walks the V lanes in Python, ~16 s per Llama row; a B8 o128 row has ~500 sampled rows)
            words = [bf_words[row].astype(np.uint64).reshape(-1), np.array([_f32_bits(p)], dtype=np.uint64),
                     np.array([seed & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64), np.array([int(pos_fact) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64),
                     np.array([_f32_bits(T)], dtype=np.uint64)]
            pending.append((len(rows), (int(V), int(S), words), None if masked_obs is None else masked_obs[row].view(np.uint32).copy()))
            rows.append(rec)
    # evaluate the pending rows (the Definition of record, GumbelTopPTokenSelect_v1{V,S}) -- by the interpreter ('lane', the path of
    # record) or by the vectorised twins ('rows'); under 'rows', `reference_rows` of them ALSO go through the lane
    # walk and both results are compared word for word (token + masked row): a disagreement between the two evaluators is a FAIL by
    # name, never resolved in favour of either.  Both evaluators and the reference count are on the record.
    if evaluator not in EVALUATORS:
        raise ValueError(f"evaluator {evaluator!r}: one of {sorted(EVALUATORS)}")
    t_eval = time.time()
    results = _evaluate_all([j for _, j, _ in pending], jobs, evaluator)
    eval_seconds_total = round(time.time() - t_eval, 1)
    ref_check: dict = {"evaluator": evaluator, "reference_evaluator": None, "rows_requested": 0, "rows_checked": 0, "disagreements": [],
                       "seconds": 0.0}
    if evaluator != "lane" and reference_rows and pending:
        from verity_vllm.commit import challenge as CH
        pick = sorted(CH.stream(CH.reference_rows_seed(sampled_rows_total, len(pending))).sample(range(len(pending)), min(int(reference_rows), len(pending))))
        t_ref = time.time()
        ref_results = _evaluate_all([pending[k][1] for k in pick], jobs, "lane")
        ref_check.update(reference_evaluator="lane", rows_requested=int(reference_rows), rows_checked=len(pick), seconds=round(time.time() - t_ref, 1),
                         rows=[{"step": rows[pending[k][0]]["step"], "row": rows[pending[k][0]]["row"]} for k in pick],
                         selection="seeded sample over the value-checked rows (seed = sampled_rows_total x 7919 + rows pending)")
        for k, (rtok, rmasked, rsecs) in zip(pick, ref_results):
            tok, masked_rec, _ = results[k]
            rec = rows[pending[k][0]]
            same_tok, same_row = int(tok) == int(rtok), bool(masked_rec.shape == rmasked.shape and (masked_rec == rmasked).all())
            rec["reference_lane_walk"] = {"token": int(rtok), "token_equal": same_tok, "masked_row_equal": same_row, "eval_seconds": rsecs}
            if not (same_tok and same_row):
                ref_check["disagreements"].append({"step": rec["step"], "row": rec["row"], "request_id": rec["request_id"], "rows_token": int(tok),
                                                   "lane_token": int(rtok), "masked_row_equal": same_row})
                reasons.append(f"step {rec['step']} row {rec['row']} ({rec['request_id']}): the vectorised evaluator ({tok}) and the lane walk "
                               f"({rtok}; masked row {'equal' if same_row else 'DIFFERENT'}) disagree on the same Definition and inputs -- "
                               f"an evaluator defect, not a verdict on the record")
    for (i, (V, S, _), mobs), (tok, masked_rec, secs) in zip(pending, results):
        rec = rows[i]
        step, row, rid = rec["step"], rec["row"], rec["request_id"]
        seed, T, p, pos_fact = rec["declared"]["seed"], rec["declared"]["temperature"], rec["declared"]["top_p"], rec["declared"]["pos"]
        rec.update(token=tok, eval_seconds=secs, evaluator=evaluator)
        if mobs is not None:
            eq = int((masked_rec == mobs).sum())
            rec["masked_row_equal_lanes"] = eq
            rec["masked_row_equal"] = eq == V
            if eq != V:
                bad = np.nonzero(masked_rec != mobs)[0][:8].tolist()
                rec["masked_row_first_diffs"] = [{"v": int(v), "recomputed": f"0x{int(masked_rec[v]):08x}",
                                                  "observed": f"0x{int(mobs[v]):08x}"} for v in bad]
                reasons.append(f"step {step} row {row} ({rid}): the kernel-consumed masked fp32 row differs from TemperatureScale->TopPMask "
                               f"under the declared policy in {V - eq} lanes (first {bad[:4]})")
        st = "exact-match" if tok == rec["served"] else "mismatch"
        rec["status"] = st
        tally[st] += 1
        if st == "mismatch":
            reasons.append(f"step {step} row {row} ({rid}): served token {rec['served']} != GumbelTopPTokenSelect_v1{{V={V},S={S}}} "
                           f"under the declared seed {seed} / T={T} / p={p} / pos={int(pos_fact)} -> {tok}")
    checked = tally["exact-match"] + tally["mismatch"]
    device_ok = all(r.get("device_facts_match", True) for r in rows)
    masked_ok = all(r.get("masked_row_equal", True) for r in rows)
    reasons.extend(s_disagree)
    s_ok = not s_disagree
    s_checked = sum(1 for r in rows if r.get("S_agree") is True)
    reasons.extend(chain_absent)
    # the sample ledger: one sample per request per step at pos = prompt_len - 1 + k, draws == cap or
    # EOS + exactly eos_lag_rows lag draw(s) and none after; a ledger problem is a FAIL reason by name
    ledger = draw_accounting(draws_seen, workload, build_dir, lag_attested=record_lag_rows(header, snapshots_dir))
    reasons.extend(ledger["problems"])
    complete = checked == sampled_rows_total and not chain_absent
    if not complete:
        reasons.append(f"{checked} of {sampled_rows_total} sampled rows value-checked: the sampler chain is unverified at the others "
                       f"(no-snapshot {tally['no-snapshot']} step(s), skipped {tally['skipped']}) -- snapshot every sampled step (MATCH_SNAP_STEPS)")
    if checked == 0:
        verdict = "NOTHING-CHECKED"
    elif tally["mismatch"] or not device_ok or not masked_ok or not s_ok or header_vs_manifest or ref_check["disagreements"] or \
            ledger["problems"] or rows_without_seed:
        verdict = "FAIL"
    elif not complete:
        verdict = "INCOMPLETE"          # k of n value-checked is not a PASS (row_pod: anything but PASS fails the Match)
    else:
        verdict = "PASS" if not reasons else "FAIL"
    # what is declared vs what is observed, said in the verdict: the fold's sampler Const32/Const64 (temperature,
    # top_p, pos) and the seed Param are sourced from the run header's request records (structure, DECLARED); the execution tie is
    # THIS check -- the snapshotted small buffers the kernels consumed (seeds_ptr/pos_ptr/temp_ptr/P), at every checked row.
    n_dev = sum(1 for r in rows if r.get("device_facts_match") is True)
    structure_note = (f"fold sampler constants (Const32 temperature/top_p, Const64 pos) and the seed Param are DECLARED (run header request "
                      f"records, checked against the manifest of record: {'agree' if not header_vs_manifest else 'DISAGREE'}); OBSERVED "
                      f"(device-tied) at {n_dev} of {checked} checked rows via the kernel-consumed seeds_ptr/pos_ptr/temp_ptr/P snapshots"
                      + (f"; S: kernel constexpr == Program TopPMask S at {s_checked} of {checked} rows" if program_S else
                         "; S: kernel constexpr only (no --build given: the Program's S is checked by cb_a.compare spec equality, not here)"))
    return {"schema": "verity-r15/stoch-recompute/v2", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "log": str(log_path),
            "workload": None if workload is None else str(workload), "build": None if build_dir is None else str(build_dir),
            "declared_facts": facts, "run_header_facts": run_facts, "program_splits": program_S,
            "header_vs_manifest": header_vs_manifest, "structure_note": structure_note,
            "variant": VARIANT, "definition": "GumbelTopPTokenSelect_v1{V,S} = TemperatureScale_v1 -> TopPMask_v1 -> GumbelSelectF32_v1 (verity/ml/definitions/registry/sampling.py)",
            # WHICH evaluator produced the value check, and the lane-walk reference beside it: both named
            "evaluator": {"name": evaluator, "what": EVALUATOR_DOC[evaluator], "eval_seconds_total": eval_seconds_total,
                          "reference": ref_check, "jobs": jobs, "cpu_quota": _cpu_quota()},
            # the record states the RNG's statelessness so 'RNG advances iff r_t' is read off it: a lag slot
            # (executed, unserved) draws Philox(seed, L - 1 + n) with n served slots (slot t = n + 1 at pos L - 1 + (t - 1), the record's pos fact)
            # and nothing else moves; a wrong pos on a device row is a mismatch by name above
            "rng": {"kind": "counter (Philox4x32-10 word 0), key (seed_lo, seed_hi), counter (pos_lo, pos_hi, 0, 0) -- tl.randint(seed, pos)",
                    "state": "none: the draw of a row is a pure function of (seed, pos, lane); no draw is consumed or advanced by another row",
                    "advances_iff_executed": "a row's draw exists iff the row executed (r_t = 1) at its own pos; an EOS-stopped request's lag "
                                             "slot draws at pos prompt_len - 1 + n (n = its served slots; the record's pos of slot t is prompt_len - 1 + (t - 1)) and no later "
                                             "pos is drawn; the served rows' draws are unchanged",
                    "checked_here": "declared pos/seed (Program request facts) == device pos/seed (kernel-consumed pos_ptr/seeds_ptr snapshot) "
                                    "at every value-checked row (device_facts_checked); the draw LEDGER below (rng_advances_iff_run)",
                    # R5's record half, MEASURED per request: the ledger of every sampled row (snapshot or not)
                    "rng_advances_iff_run": ledger},
            "sampling": dict((header.workload or {}).get("sampling") or {}),
            "summary": {"engine_steps_in_record": engine_steps, "sampled_steps_in_log": steps_seen, "steps_chain_absent": len(chain_absent),
                        "sampled_rows_total": sampled_rows_total, "rows_checked": checked, "complete": complete,
                        "rows": tally, "device_facts_all_match": device_ok, "device_facts_rows": n_dev,
                        "masked_rows_all_equal": masked_ok, "program_S_all_agree": s_ok if program_S else None, "program_S_rows": s_checked,
                        "draw_ledger_ok": not ledger["problems"], "lag_slot_exercised": ledger["lag_slot_exercised"],
                        "requests_with_eos_before_cap": ledger["requests_with_eos_before_cap"], "requests_at_cap": ledger["requests_at_cap"],
                        "verdict": verdict, "seconds": round(time.time() - t0, 1)},
            "reasons": reasons, "rows": rows}


@dataclass(frozen=True, kw_only=True)
class Options:
    run: str = option("--run", help="Match dir (RUN_DIR/capture/{log.jsonl.gz,snapshots.json,values/})")
    out: str | None = option("--out", default=None)
    max_rows: int | None = option("--max-rows", type=int, default=None)
    workload: str | None = option("--workload", type=workload_arg, default=None,
                                  help="the workload manifest of record (declared seeds / policy); the run's header must agree with it")
    jobs: int | None = option("--jobs", type=int, default=None,
                              help="worker processes for the row evaluations (default min(32, cpus))")
    build: str | None = option("--build", default=None,
                               help="the row dir holding the Build's build_request*/result.json (the Program's S per request "
                               "shape, compared with the mask kernel's constexpr S at every checked row; needs --workload)")
    evaluator: str = option("--evaluator", env="STOCH_EVALUATOR", default="lane", choices=sorted(EVALUATORS),
                            help="'lane' = the registry interpreter over the V lanes (the path of record; default; env STOCH_EVALUATOR); 'rows' = the "
                            "vectorised twins of the same gates (verity/ml/definitions/registry/sampling_rows.py), milliseconds per row")
    reference_rows: int = option("--reference-rows", type=int, env="STOCH_REFERENCE_ROWS", default=8,
                                 help="under --evaluator rows: this many value-checked rows ALSO go through the lane walk; token + masked row must agree "
                                 "word for word or the verdict is FAIL by name (default 8; env STOCH_REFERENCE_ROWS)")


def main(a: Options) -> int:
    run = Path(a.run)
    rep = recompute_run(run / "capture" / "log.jsonl.gz", run / "capture", a.max_rows, workload=a.workload, jobs=a.jobs, build_dir=a.build,
                        evaluator=a.evaluator, reference_rows=a.reference_rows)
    out = Path(a.out) if a.out else run / "stoch_recompute.json"
    out.write_text(json.dumps(rep, indent=1, default=str))
    s = rep["summary"]
    ev = rep["evaluator"]
    print(f"stoch_recompute {run}: {s['verdict']} {s['rows_checked']} of {s['sampled_rows_total']} sampled rows value-checked "
          f"({s['engine_steps_in_record']} engine steps in the record, {s['steps_chain_absent']} with an incomplete sampler chain) rows={s['rows']} "
          f"device_facts={s['device_facts_all_match']} masked={s['masked_rows_all_equal']} program_S={s['program_S_all_agree']} "
          f"({s['seconds']}s; evaluator {ev['name']} {ev['eval_seconds_total']}s"
          + (f", lane-walk reference on {ev['reference']['rows_checked']} rows: {len(ev['reference']['disagreements'])} disagreements, "
             f"{ev['reference']['seconds']}s" if ev["reference"]["reference_evaluator"] else "") + f") -> {out}")
    print("  structure:", rep["structure_note"])
    for r in rep["reasons"][:12]:
        print("  FAIL:", r)
    return 0 if s["verdict"] == "PASS" else (2 if s["verdict"] == "NOTHING-CHECKED" else 1)
