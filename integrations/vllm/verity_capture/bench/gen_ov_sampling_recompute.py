"""[GEN ov-sampling] Bit-exact recomputation of every sampled row of a B1 sampling run from the trusted facts only --
no tolerance, no margin discipline (pod or laptop, CPU-only, needs RUN_DIR/capture/{log.jsonl.gz,snapshots.json,values/}).

For every snapshotted step with a `_gumbel_sample_kernel` launch, for every logits row:

    token' = argmax_v( logits_ptr[row][v] + g(seed, pos)[v] )      # fp32, first index wins; g = exact libdevice transcription
    g(seed, pos)[v] = -logf(-log1pf(-max(rand(randint(seed, pos), v), 2**-31)))   (gen_ov_sampling_libdevice.gumbel_noise_exact)

and compares `token'` with `SampleEnd.sampled_token_ids[row]` (the served token).  `seed` / `pos` / `temperature` are read
from the kernel's own argument snapshots (`seeds_ptr[expanded_idx_mapping_ptr[row]]`, `pos_ptr[row]`, `temp_ptr[...]`) and
cross-checked against the request record in the header (`workload.requests[*].seed`, `workload.children[*]`), so the
report separates the three questions the contract needs answered:

  1. is the served token a *pure function* of (fp32 logits row, seed, pos)?          -> rows[*].status exact-match/mismatch
  2. does the seed the kernel consumed equal the seed of the request record?          -> rows[*].seed_matches_request
  3. for unseeded requests: what did the kernel consume (a host draw, unexplained)?   -> rows[*].seed_source == "runner-only"

Top-k / top-p rows: the kernel's `logits_ptr` is the *masked* row (the -inf mask was applied by the preceding aten chain);
the recomputation of the mask itself from the bf16 logits (SampleEnd.logits_view) is reported separately in `masks`
(top-k: exact k-th-largest threshold; top-p: fp32 softmax + sequential cumsum -- see REPORT.md for why the cumsum is
NOT a trusted recomputation).

    python -m verity_capture.bench.gen_ov_sampling_recompute --run RUN_DIR [--out RUN_DIR/ov_recompute.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from verity_capture.bench import gen_sampling_rng as R
from verity_capture.bench.gen_ov_sampling_libdevice import gumbel_noise_exact


def _first_argmax_f32(y: np.ndarray) -> tuple[int, int, float]:
    """(winner, runner-up, fp32 margin) with the kernel's tie-break (lowest index among the maxima)."""
    best = int(np.argmax(y))
    rest = y.copy()
    rest[best] = -np.inf
    second = int(np.argmax(rest))
    return best, second, float(np.float32(y[best]) - np.float32(y[second]))


def recompute_run(log_path: Path, snapshots_dir: Path) -> dict:
    from verity_capture.compare import SnapshotStore
    from verity_capture.events import DispatchEnter, SampleEnd, StepStart, TritonLaunch, ViewDesc
    from verity_capture.log import read_log

    events = list(read_log(log_path))
    header = events[0]
    wl = header.workload
    samp = dict(wl.get("sampling") or {})
    req_seed: dict[str, int | None] = {r["request_id"]: r.get("seed") for r in wl.get("requests", [])}
    req_n: dict[str, int] = {r["request_id"]: int(r.get("n") or 1) for r in wl.get("requests", [])}
    child_of: dict[str, tuple[str, int]] = {}
    for c in wl.get("children", []) or []:
        req_seed[c["request_id"]] = c.get("seed")
        if c.get("parent") is not None:
            child_of[c["request_id"]] = (c["parent"], int(c.get("index", 0)))
    per_req_T: dict[str, float] = {p["request_id"]: float(p["temperature"]) for p in samp.get("per_request", []) or []}
    T_default = float(samp.get("temperature", 0.0))
    store = SnapshotStore(snapshots_dir)
    per_step: dict[int, dict] = {}
    cur: StepStart | None = None
    ops_sample: dict[int, list[str]] = {}
    for ev in events[1:]:
        if isinstance(ev, StepStart):
            cur = ev
            per_step[ev.step] = {"launch": None, "sample_end": None, "temp_launch": None, "aten": []}
        elif isinstance(ev, (TritonLaunch, DispatchEnter)) and ev.phase == "sample":
            ops_sample.setdefault(ev.step, []).append(ev.fn.split(".")[-1] if isinstance(ev, TritonLaunch) else ev.op)
            if isinstance(ev, TritonLaunch) and ev.fn.endswith("_gumbel_sample_kernel"):
                per_step[ev.step]["launch"] = (ev, cur)
            elif isinstance(ev, TritonLaunch) and ev.fn.endswith("_temperature_kernel"):
                per_step[ev.step]["temp_launch"] = ev
            elif isinstance(ev, DispatchEnter):
                per_step[ev.step]["aten"].append(ev)
        elif isinstance(ev, SampleEnd):
            per_step[ev.step]["sample_end"] = ev
    rows: list[dict] = []
    masks: list[dict] = []
    temps: list[dict] = []
    tally = {"exact-match": 0, "mismatch": 0, "greedy-match": 0, "greedy-mismatch": 0, "no-snapshot": 0}
    seed_tally = {"request": 0, "request-child": 0, "runner-only": 0, "request-mismatch": 0}
    for step, d in sorted(per_step.items()):
        if d["launch"] is None or d["sample_end"] is None:
            continue
        launch, ss = d["launch"]
        lv = launch.args.get("logits_ptr")
        if not isinstance(lv, ViewDesc):
            continue
        snap = store.find(lv, at=launch.id) or store.find(lv, before=launch.id + 1)
        if snap is None:
            tally["no-snapshot"] += 1
            rows.append({"step": step, "status": "no-snapshot"})
            continue
        logits = R._as_f32(store.extract(snap, lv), lv.dtype).reshape(lv.shape)
        cx = launch.constexprs or {}
        seeds_snap = R._small_int_arg(store, launch, "seeds_ptr")
        pos_snap = R._small_int_arg(store, launch, "pos_ptr")
        temp_snap = R._small_arg(store, launch, "temp_ptr", np.float32)
        map_snap = R._small_int_arg(store, launch, "expanded_idx_mapping_ptr")
        served = list(d["sample_end"].sampled_token_ids)
        # the bf16 sampler input (pre temperature / mask) for the mask recomputation
        bf = d["sample_end"].logits_view
        bf_logits = None
        if isinstance(bf, ViewDesc):
            bsnap = store.find(bf, before=launch.id + 1)
            if bsnap is not None:
                bf_logits = R._as_f32(store.extract(bsnap, bf), bf.dtype).reshape(bf.shape)
        for row, (rid, pos_fact) in enumerate(ss.logits_rows):
            idx = int(map_snap[row]) if map_snap is not None and row < len(map_snap) else row
            dev_seed = int(seeds_snap[idx]) if seeds_snap is not None else None
            pos = int(pos_snap[row]) if pos_snap is not None and row < len(pos_snap) else int(pos_fact)
            T = float(temp_snap[idx]) if temp_snap is not None else per_req_T.get(rid, T_default)
            rec_seed = req_seed.get(rid)
            parent = child_of.get(rid)
            if rec_seed is None and parent is not None and req_seed.get(parent[0]) is not None:
                rec_seed = int(req_seed[parent[0]]) + parent[1]
                src = "request-child"
            elif rec_seed is not None:
                src = "request"
            else:
                src = "runner-only"
            if rec_seed is not None and dev_seed is not None and int(rec_seed) != dev_seed:
                src = "request-mismatch"
            seed_tally[src] += 1
            rec = {"step": step, "row": row, "request_id": rid, "pos": pos, "pos_fact": int(pos_fact), "temperature": T,
                   "device_seed": dev_seed, "request_seed": rec_seed, "seed_source": src, "seed_matches_request": rec_seed == dev_seed,
                   "served": int(served[row]), "constexprs": {k: v for k, v in cx.items()}, "kernel_event": launch.id}
            x = logits[row].astype(np.float32)
            if T == 0.0:
                tok, second, margin = _first_argmax_f32(x)
                rec.update(mode="greedy", token=tok, runner_up=second, margin=margin)
                st = "greedy-match" if tok == rec["served"] else "greedy-mismatch"
            else:
                seed = dev_seed if dev_seed is not None else rec_seed
                if seed is None:
                    rec.update(mode="sampled", status="no-seed")
                    rows.append(rec)
                    continue
                g = gumbel_noise_exact(int(seed), pos, x.shape[0], is_drafting=bool(cx.get("IS_DRAFTING", False)))
                if bool(cx.get("APPLY_TEMPERATURE", False)):
                    x = (x / np.float32(T)).astype(np.float32)   # not bit-exact (div.full.f32); vLLM's default path pre-divides
                y = (x + g).astype(np.float32)
                tok, second, margin = _first_argmax_f32(y)
                rec.update(mode="sampled", token=tok, runner_up=second, margin=margin, gumbel_seed=R.gumbel_seed_of(int(seed), pos),
                           n_masked=int(np.isinf(x).sum()), winner_logit=float(x[tok]), winner_noise=float(g[tok]))
                st = "exact-match" if tok == rec["served"] else "mismatch"
            rec["status"] = st
            tally[st] += 1
            rows.append(rec)
            # temperature quotient: `_temperature_kernel` computes `f32(bf16 logits) / T` with Triton's fp32 `/` (div.full.f32).
            # Which rule reproduces the observed fp32 row bit-exactly (unmasked lanes only)?  A: the registry's div.full model
            # F32Mul(DivFullScaleA(x, T), DivFullRcp(T)); B: IEEE RN division; C: IEEE x * f32(1/T).
            if bf_logits is not None and T not in (0.0, 1.0) and d.get("temp_launch") is not None:
                xb = bf_logits[row].astype(np.float32)
                obs = logits[row].astype(np.float32)
                live = ~np.isneginf(obs)
                tq: dict = {"step": step, "row": row, "temperature": T, "lanes": int(live.sum())}
                with np.errstate(all="ignore"):
                    q_b = (xb / np.float32(T)).astype(np.float32)
                    q_c = (xb * np.float32(np.float32(1.0) / np.float32(T))).astype(np.float32)
                    tq["ieee_div_equal"] = int((q_b.view(np.uint32) == obs.view(np.uint32))[live].sum())
                    tq["ieee_mul_rcp_equal"] = int((q_c.view(np.uint32) == obs.view(np.uint32))[live].sum())
                    try:
                        from verity_vllm.registry import prims as P
                        tb = int(np.float32(T).view(np.uint32))
                        rcp = np.uint32(P.DivFullRcp.evaluate(tb)).view(np.float32)
                        xs = np.array([P.DivFullScaleA.evaluate(int(w), tb) for w in xb.view(np.uint32)[:1]], dtype=np.uint32)
                        scale_identity = int(xs[0]) == int(xb.view(np.uint32)[0])
                        q_a = (xb * rcp).astype(np.float32) if scale_identity else None
                        tq["div_full_model_equal"] = int((q_a.view(np.uint32) == obs.view(np.uint32))[live].sum()) if q_a is not None else None
                        tq["div_full_rcp_T"] = float(rcp)
                        tq["ieee_rcp_T"] = float(np.float32(1.0) / np.float32(T))
                    except Exception as e:  # noqa: BLE001 - the MUFU tables may be absent on this host
                        tq["div_full_model_equal"] = f"unavailable: {type(e).__name__}: {str(e)[:80]}"
                masks_t = tq
                temps.append(masks_t)
            # mask recomputation (top-k exact threshold; top-p sequential fp32 cumsum) against the kernel's -inf pattern
            k = int(samp.get("top_k", 0) or 0)
            p = float(samp.get("top_p", 1.0))
            if bf_logits is not None and (k > 0 or p < 1.0) and T != 0.0:
                base = bf_logits[row].astype(np.float32)
                if T not in (0.0, 1.0):
                    base = (base / np.float32(T)).astype(np.float32)
                observed_mask = np.isneginf(logits[row])
                m: dict = {"step": step, "row": row, "observed_masked": int(observed_mask.sum())}
                keep = np.ones_like(base, dtype=bool)
                if k > 0:
                    # vLLM `_topk_topp_kernel`: keep x > pivot plus the first `num_keep` boundary duplicates in vocab order,
                    # i.e. the k largest with ties broken by LOWEST index (order statistic on (-x, v)); exactly k survive.
                    order = np.argsort(-base, kind="stable")
                    keep_k = np.zeros_like(keep)
                    keep_k[order[:k]] = True
                    thr = base[order[k - 1]]
                    m["topk_threshold"] = float(thr)
                    m["topk_ties_at_threshold"] = int((base == thr).sum())
                    m["topk_kept"] = int(keep_k.sum())
                    m["topk_mask_equal_lowest_index_ties"] = bool(np.array_equal(~keep_k, observed_mask)) if p >= 1.0 else None
                    keep &= keep_k
                if p < 1.0:
                    # candidate rule: the exact nucleus (real arithmetic) of the row the kernel saw (after top-k when both are on):
                    # the smallest top set by probability with mass >= p; ties at the boundary value broken by LOWEST index (kept),
                    # the same order statistic as the top-k rule above.  `topp_boundary_gap` = mass(nucleus) - p (how far the
                    # boundary lies from p: the kernel's approximate pivot search can only differ from the exact nucleus inside its
                    # numerical error of that gap).
                    row_p = base.astype(np.float64)
                    if k > 0:
                        row_p = np.where(keep_k, row_p, -np.inf)
                    pr0 = np.exp(row_p - np.max(row_p))
                    pr0 /= pr0.sum()
                    order_p = np.argsort(-row_p, kind="stable")          # descending value, lowest index first among ties
                    cs0 = np.cumsum(pr0[order_p])
                    n_keep = int(np.searchsorted(cs0, p, side="left")) + 1
                    kept_p = np.zeros_like(keep)
                    kept_p[order_p[:n_keep]] = True
                    m["topp_kept_exact_nucleus"] = n_keep
                    m["topp_boundary_ties"] = int((row_p == row_p[order_p[n_keep - 1]]).sum())
                    m["topp_boundary_gap"] = float(cs0[n_keep - 1] - p)
                    m["topp_mask_equal_exact_nucleus_lowest_index_ties"] = bool(np.array_equal(~kept_p, observed_mask))
                    keep &= kept_p
                    # nucleus invariant of the OBSERVED kept set in exact (fp64) arithmetic over the row the kernel saw
                    # (after top-k when both are on): mass(kept) >= p and mass(kept) - min kept prob < p  <=>  the kept set is
                    # *a* nucleus; whether it is *the* nucleus is then only a question of the boundary value's rounding.
                    row_in = base.astype(np.float64)
                    if k > 0:
                        row_in = np.where(keep_k, row_in, -np.inf)
                    pr = np.exp(row_in - np.max(row_in))
                    pr /= pr.sum()
                    kept_obs = ~observed_mask
                    mass = float(pr[kept_obs].sum())
                    min_kept = float(pr[kept_obs].min()) if kept_obs.any() else 0.0
                    # the exact nucleus over the same row: smallest top set with mass >= p
                    desc = np.argsort(-pr, kind="stable")
                    cs = np.cumsum(pr[desc])
                    n_nuc = int(np.searchsorted(cs, p, side="left")) + 1
                    m["nucleus_exact_size"] = n_nuc
                    m["observed_kept"] = int(kept_obs.sum())
                    m["observed_mass"] = mass
                    m["observed_is_a_nucleus_fp64"] = bool(mass >= p and mass - min_kept < p)
                    m["observed_minus_exact_nucleus_tokens"] = int(kept_obs.sum()) - n_nuc
                m["recomputed_masked"] = int((~keep).sum())
                m["mask_equal"] = bool(np.array_equal(~keep, observed_mask))
                m["mask_symmetric_difference"] = int((keep == observed_mask).sum())
                masks.append(m)
    return {"schema": "verity-gen-ov-sampling/recompute/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "log": str(log_path), "sampling": samp, "sample_phase_ops_by_step": {str(k): v for k, v in sorted(ops_sample.items())},
            "summary": {"rows": tally, "seed_source": seed_tally, "masks_equal": sum(1 for m in masks if m["mask_equal"]), "masks": len(masks),
                        "temperature_rows": len(temps),
                        "temperature_div_full_model_exact_rows": sum(1 for t in temps if t.get("div_full_model_equal") == t["lanes"]),
                        "temperature_ieee_div_exact_rows": sum(1 for t in temps if t.get("ieee_div_equal") == t["lanes"]),
                        "temperature_ieee_mul_rcp_exact_rows": sum(1 for t in temps if t.get("ieee_mul_rcp_equal") == t["lanes"])},
            "rows": rows, "masks": masks, "temperature": temps}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    rep = recompute_run(run / "capture" / "log.jsonl.gz", run / "capture")
    out = Path(a.out) if a.out else run / "ov_recompute.json"
    out.write_text(json.dumps(rep, indent=1, default=str))
    s = rep["summary"]
    print(f"ov_recompute {run.name}: rows={s['rows']} seeds={s['seed_source']} masks_equal={s['masks_equal']}/{s['masks']} -> {out}")
    for r in rep["rows"]:
        if r.get("status") not in ("exact-match", "greedy-match"):
            print("  ", {k: r.get(k) for k in ("step", "row", "request_id", "status", "served", "token", "device_seed", "request_seed", "seed_source", "margin")})
    return 0 if s["rows"]["mismatch"] == 0 and s["rows"]["greedy-mismatch"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
