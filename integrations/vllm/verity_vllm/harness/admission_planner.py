"""ONE conservative resource planner for a Commit launch (COORD R17-29 = Daniel; fp8 owner, stoch with, exec wires, revb reviews).

The planner answers, for THIS collector implementation on the ALREADY-DEFINED workload (the declared schedule `requests[]`, the model's
history geometry, the collection layout and retention mode), the five R17-29 questions, each as a named term with a predicted byte count:

    gpu_staging    peak device staging          bounded: slots x window + scratch (the window ring, P0) | unbounded: the whole-step block
    host_staging   peak host / pinned staging   bounded: levels + kept (+ a 2nd pair's frozen pool)     | retain host: scheduled raw x 1.25
    hash_backlog   outstanding hash work        bounded: the learn budget (windows awaiting their re-hash) | unbounded: the largest step's
                                                pinned D2H copy -- the F-fp8-B8MEM allocation (learning-step re-hash source), whatever --retain
    opening_data   retained opening data        openings by replay: a fresh pool (levels + learn + picks x 256 B chunk) | else picks x window
    headroom       conservative headroom        engine RSS at admission + slack (engine growth after admission) + memory.current of co-tenants

It decides only whether the implementation can execute the workload under finite resources; it never determines semantics.  The HOST
bound is the sum of the four host terms; in bounded mode it is term-for-term `native_host.bounded_host_bound` (revb S-r17b-03 (iii),
the B1 try-5 / B8 admissions of record) plus the co-tenant term.  REQUIRED (R17-29): the DOMINANT CONTRIBUTOR is named -- the largest
host term and the largest single materialization of the stream (family, step, prompt tokens, bytes, and where this mode puts it) -- so a
refusal reads "hidden M1 stream (fa3, 36 layers): 66.9 GiB (prefill step 0, 3,537 prompt tokens) -- excluded from host, opened by
replay" and never merely "predicted OOM".

Step demand comes from `admission.scheduled_step_bytes` (stoch, F-dA-15: per engine step prefills / decodes / tokens / K / chunks + terms;
sound against 6 records' per-step n_bytes, 0 under-predicted steps).  The hidden M1 term of a MULTI-prompt prefill step is computed in
BOTH geometries and the planner takes the larger until a record of the backend calibrates it:  `per_sequence` (each prompt at its own
length -- what the FA2 records show, stoch M-0357) and `padded` (every sequence of the launch padded to the step's max_seqlen_q -- the
tap's buffer geometry, hidden_source.varlen_fwd HB = b x h, M = max_seqlen_q; the R16 FA3 B8 attempt 2 died at RSS 126 GB and climbing
inside the step-0 prefill under --retain none, above the per-sequence figure).  Both numbers are on the record; the B8 FP8 Commit's
footprint decides FA3.

CALIBRATION IS PART OF IT: `calibrate(plan, record_dir)` reads a Commit's own record -- verdict.host_peak (process VmHWM), footprint.jsonl
(pinned pool / RSS peaks, per-step staged bytes), commit/steps.jsonl (per-step soundness through `admission.compare_observed`) -- and
returns measured-vs-predicted per term where a term is measurable, plus one calibration ENTRY {mode, predicted, measured, ratio}.  The
decision uses the CALIBRATED bound = predicted x factor(mode), factor = max(1.0, the largest measured/predicted ratio among the mode's
EXACT entries) (>= 2 entries; else 1.0) -- calibration can only RAISE a derivation that under-predicted, never lower one (revb n-fp8-plan-1:
a ratio < 1 confirms the derivation as an upper bound and licenses nothing); board-quoted approximate figures (`approx: true`) never set
the factor, and one that over-ran its derivation keeps the decision at WARN.  REFUSE when the calibrated bound exceeds cap x (1 - margin);
WARN when the mode has no calibration or an approximate over-record; ADMIT otherwise.  The bound of record stays the derivation.  The bound is printed BEFORE the run it governs and the
record names it (`admission_of_record`), as revb M-0254 / S-r17b-03 (ii) require; a run whose VmHWM exceeds it is a finding against the
derivation, never a slack adjustment.

Torch-free.  CLI:  python -m verity_vllm.harness.admission_planner --workload <manifest.json> --sizing <arena_sizing_*.json>
                   [--bounded] [--retain host|none|sample] [--exclude fa3_hidden_m1] [--open-after-release] [--openings 64] [--pairs 3]
                   [--engine-rss-mb N] [--cap-bytes N] [--others-mb N] [--calibration <entries.jsonl>] [--record <commit dir>] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from verity_vllm.harness.admission_bound import compare_observed, host_cap, observed_steps, scheduled_step_bytes
from verity_vllm.acquire.hidden_source import CHUNK_WORDS, fa2_kblock_n, fa3_tile, mat_total_words

SCHEMA = "verity-sweep/admission-planner/v1"
MiB = 1 << 20
GiB = 1 << 30
HOST_TERMS = ("host_staging", "hash_backlog", "opening_data", "headroom")
TERMS = ("gpu_staging",) + HOST_TERMS
DEFAULT_MODE: dict[str, Any] = {"retain": "host", "bounded": True, "retain_exclude": ["fa3_hidden_m1"], "open_after_release": True,
                                "openings": 64, "pairs": 1, "window_bytes": 256 * MiB, "slots": 8, "scratch_bytes": None,
                                "learn_budget_bytes": 4 * GiB, "chunk_bytes": 256, "hidden_factor": 1.05, "other_factor": 2.0,
                                "slack_bytes": 6 * GiB, "levels_factor": 1.25}
# [R17-35] the SCOPE of every term -- whole-run retained data is not comparable with transient staging (Daniel s7); a measured figure is
# compared only with a predicted one of the same scope
TERM_SCOPE = {"gpu_staging": "transient device staging (window ring + scratch; recycled per window)",
              "host_staging": "whole-run retained host data (levels + kept chunks in the frozen pinned pool; freed at pool.reset)",
              "hash_backlog": "transient pinned host staging (learning windows awaiting their re-hash; capacity = learn budget)",
              "opening_data": "post-run host data (the replay pool for openings after release; allocated while the run's pool is still frozen)",
              "headroom": "process baseline (engine RSS at admission + growth) and other processes of the cgroup"}


def lifetime_overlaps(*, bounded: bool, retain: str, excl: set[str], open_after_release: bool, pairs: int) -> list[dict[str, Any]]:
    """[R17-35] the lifetimes the host bound ASSUMES to overlap at the peak (what its sum adds).  Each entry names the two lifetimes, whether the
    collector actually keeps them co-resident, and how the record tells (so `calibrate` can say 'assumed overlap that does not happen')."""
    out = [{"lifetimes": ["engine RSS", "collector pools"], "assumed": "co-resident for the whole run", "actual": "yes (the engine serves while the collector pins)",
            "measured_by": "footprint.jsonl rss_bytes - pinned_pool_bytes at the peak step vs headroom.engine_rss"}]
    if bounded:
        out.append({"lifetimes": ["levels (retained)", "kept chunks (retained)", "learn pool (transient windows)"],
                    "assumed": "co-resident: one frozen pool holds levels + kept, the learn budget is its window capacity",
                    "actual": "yes within one pool; the learn windows are RECYCLED (a window is reused after its re-hash completes: CUDA event + consumer done), so their peak is the budget, not the run's raw bytes",
                    "measured_by": "footprint.jsonl pinned_pool_bytes (frozen) vs host_staging + hash_backlog"})
        if open_after_release:
            out.append({"lifetimes": ["the run's frozen pool (levels + kept)", "the replay pool for openings after release"],
                        "assumed": "co-resident (the replay pool is a FRESH pool allocated while the run's pool is still frozen)",
                        "actual": "yes: release_staging frees the DEVICE staging (levels to host); the frozen view owns the run's pinned blocks (levels + kept) while the replay runs on a fresh pool",
                        "measured_by": "openings_after_release.host_rss_hwm_gb vs the RSS before replay (the replay pool's actual size vs opening_data)"})
        if pairs >= 2:
            out.append({"lifetimes": ["pair N's pool (levels + kept + learn)", "pair N+1's pool"],
                        "assumed": "one extra set co-resident (torch's pinned-host cache keeps the returned blocks; the next pair's pool takes fresh ones once)",
                        "actual": "measured B1 PAIRS=3: converged after the second pair (the cache reuses blocks from the third pair on) -- one set, not pairs-1 sets",
                        "measured_by": "rss timeline per pair (the step of RSS at each pair's first pool block)"})
        if {"fa3_hidden_m1", "fa2_hidden_m1"} & excl:
            out.append({"lifetimes": ["the hidden M1 stream's step plane", "the host pool"],
                        "assumed": "NOT co-resident: the plane is streamed through the device ring and never pinned (retain-exclude); its bytes are in no host term",
                        "actual": "yes by construction (bounded_staging.excluded_bytes on the record = the bytes kept off the host)",
                        "measured_by": "runs.jsonl commit.extra.bounded_staging.excluded_bytes vs excluded_hidden_bytes"})
    else:
        out.append({"lifetimes": ["the largest step's pinned D2H copy (learning re-hash source)", "the pool's transient window copies of every earlier step"],
                    "assumed": "co-resident (chunk-leaf-v1 returns `take` slices only at pool.reset: the pool grows with the run's raw bytes whatever --retain)",
                    "actual": "yes (FP8-STAGING s1 (2); the R16 B8 OOM inside step 0)",
                    "measured_by": "rss timeline at the peak step's end vs host_staging + hash_backlog"})
    return out


def _hidden_words(HB: int, M: int, NB: int, BN: int, D: int) -> int:
    return -(-mat_total_words(HB, M, NB, BN, D) // CHUNK_WORDS) * CHUNK_WORDS


def padded_launch_words(model: dict[str, Any], *, n_seq: int, max_seqlen_q: int, seqlen_k: int) -> int:
    """Words of ONE attention launch's M1 stream in the TAP'S BUFFER geometry: every sequence padded to max_seqlen_q (hidden_source.varlen_fwd:
    HB = b x h, M = max_seqlen_q; FA3 PackGQA slab (sequence, KV head), M = max_seqlen_q x g) -- native_host.schedule_stream_bound's model."""
    heads, kvh, hd = int(model["heads"]), int(model["kv_heads"]), int(model["head_dim"])
    g = max(1, heads // kvh)
    if n_seq <= 0 or not model.get("hidden_m1", True):
        return 0
    if str(model.get("attn_backend") or "fa2") == "fa3":
        if hd not in (64, 128):
            return 0
        _, bn = fa3_tile(hd, max_seqlen_q, g, 1)
        return _hidden_words(n_seq * kvh, max_seqlen_q * g, -(-seqlen_k // bn), bn, hd)
    bn = fa2_kblock_n(hd, 1) if hd in (64, 96, 128, 256) else 128
    if max_seqlen_q == 1 and heads > kvh:
        return _hidden_words(n_seq * kvh, g, -(-seqlen_k // bn), bn, hd)
    return _hidden_words(n_seq * heads, max_seqlen_q, -(-seqlen_k // bn), bn, hd)


def padded_hidden_per_step(sizing: dict[str, Any], requests: list[dict[str, Any]], sched: dict[str, Any], *, lag: int = 1) -> dict[int, int]:
    """The padded-geometry hidden bytes of every scheduled step (one launch per step: prefills + decode rows, M = the longest prompt)."""
    model = dict(sizing["model"])
    layers = int(model.get("layers") or 1)
    reqs = []
    for r in requests:
        p = int(r.get("prompt_len") or len(r.get("prompt_token_ids") or []))
        c, a = int(r.get("max_tokens") or 1), int(r.get("arrive_step") or 0)
        reqs.append((a, p, a + c + max(0, int(lag))))
    out: dict[int, int] = {}
    for st in sched["per_step"]:
        s = int(st["step"])
        pre = [p for a, p, _e in reqs if a == s]
        dec = [p for a, p, e in reqs if a < s < e]
        if not pre and not dec:
            continue
        m_q = max(pre) if pre else 1
        out[s] = layers * 4 * padded_launch_words(model, n_seq=len(pre) + len(dec), max_seqlen_q=m_q, seqlen_k=int(st["K"]))
    return out


def calibration_factor(entries: list[dict[str, Any]] | None, mode_key: str, *, min_entries: int = 2) -> dict[str, Any]:
    """factor = max(1.0, the LARGEST measured/predicted ratio among this mode's EXACT entries) -- calibration can only RAISE a derivation
    that under-predicted, never lower one (revb n-fp8-plan-1: a ratio < 1 confirms the derivation as an upper bound; it does not license
    shrinking it -- admitting on a shrunk bound would be admission by past under-utilization).  Board-quoted approximate RSS figures
    (`approx: true`, no VmHWM) are listed but never set the factor.  Fewer than `min_entries` records: factor 1.0, flagged uncalibrated."""
    rows = [e for e in (entries or []) if e.get("mode") == mode_key and e.get("ratio") is not None]
    exact = [e for e in rows if not e.get("approx")]
    if len(rows) < min_entries:
        return {"mode": mode_key, "entries": len(rows), "exact_entries": len(exact), "factor": 1.0, "calibrated": False,
                "why": f"{len(rows)} calibration record(s) for {mode_key} (< {min_entries}): the derivation stands uncalibrated"}
    worst = max(exact, key=lambda e: e["ratio"]) if exact else None
    worst_ratio = float(worst["ratio"]) if worst is not None else 1.0
    factor = max(1.0, worst_ratio)
    # an APPROXIMATE record that over-ran its derivation (ratio > factor) is a caution on the record -- the decision may not be a plain
    # ADMIT while one exists -- but never a number the bound is multiplied by
    approx_over = [{"row": e.get("row"), "ratio": round(float(e["ratio"]), 4)} for e in rows if e.get("approx") and float(e["ratio"]) > factor]
    return {"mode": mode_key, "entries": len(rows), "exact_entries": len(exact), "factor": factor, "calibrated": True,
            "worst_record": (worst.get("row") if worst is not None else None), "worst_ratio": worst_ratio,
            "rule": "factor = max(1.0, worst exact measured/predicted): calibration only raises the derivation; approx entries never set it",
            "ratios": sorted(round(float(e["ratio"]), 4) for e in rows),
            "approx_records": [e.get("row") for e in rows if e.get("approx")], "approx_over_records": approx_over}


def plan(sizing: dict[str, Any], requests: list[dict[str, Any]], *, mode: dict[str, Any] | None = None, engine_rss_bytes: int,
         host_cap_bytes: int | None = None, host_cap_source: str | None = None, others_current_bytes: int = 0, gpu_cap_bytes: int | None = None,
         max_num_batched_tokens: int | None = None, lag: int = 1, margin: float = 0.10, calibration: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    m = {**DEFAULT_MODE, **(mode or {})}
    retain, bounded = str(m["retain"]), bool(m["bounded"])
    excl = set(m.get("retain_exclude") or ()) if (bounded and retain == "host") else set()      # an exclusion only means something under retain host
    oar = bool(m.get("open_after_release"))
    pairs = int(m.get("pairs") or 1)
    win, slots, chunk = int(m["window_bytes"]), int(m["slots"]), int(m["chunk_bytes"])
    learn = int(m["learn_budget_bytes"])
    model = dict(sizing["model"])
    layers = int(model.get("layers") or 1)
    attn = str(model.get("attn_backend") or "fa2")

    sched = scheduled_step_bytes(sizing, requests, max_num_batched_tokens=max_num_batched_tokens, lag=lag)
    per_seq_hidden = {int(s["step"]): int(s["hidden_bytes"]) for s in sched["per_step"]}
    padded = padded_hidden_per_step(sizing, requests, sched, lag=lag)
    # the hidden term of every step in both geometries.  FA2: the per-sequence geometry IS the record's (stoch M-0357: 6 records, 0 under-
    # predicted steps, the exec llama B8 peak step at 0.971) -- the padded figure is carried, not used.  FA3: no multi-prompt record yet
    # (the R16 B8 died at RSS 126 GB and climbing inside step 0, above the per-sequence figure) -> the larger per step until the B8 FP8
    # Commit's footprint calibrates it.
    geometry_rule = "per_sequence (FA2: 6 calibrated records)" if attn != "fa3" else "max(per_sequence, padded) (FA3: no multi-prompt record yet; R16 B8 RSS evidence)"
    hidden_steps = ({s: max(per_seq_hidden.get(s, 0), padded.get(s, 0)) for s in per_seq_hidden} if attn == "fa3" else dict(per_seq_hidden))
    hidden_total_ps, hidden_total_pad = sum(per_seq_hidden.values()), sum(padded.values())
    hidden_total = sum(hidden_steps.values())
    other_total = int(sched["total_bytes"]) - hidden_total_ps                 # tokens + seqs + step padding (the non-hidden families)
    run_total = other_total + hidden_total
    steps_bytes = {int(s["step"]): int(s["bytes"]) - per_seq_hidden[int(s["step"])] + hidden_steps[int(s["step"])] for s in sched["per_step"]}
    peak_step = max(steps_bytes, key=steps_bytes.get)
    peak = next(s for s in sched["per_step"] if int(s["step"]) == peak_step)
    peak_bytes = steps_bytes[peak_step]

    terms: dict[str, dict[str, Any]] = {}
    if bounded:
        raw = int(hidden_total * float(m["hidden_factor"])) + int(other_total * float(m["other_factor"]))
        hidden_bound = int(hidden_total * float(m["hidden_factor"]))
        hidden_excluded = hidden_bound if ({"fa3_hidden_m1", "fa2_hidden_m1"} & excl) else 0
        kept = max(0, raw - hidden_excluded) if retain == "host" else 0
        levels = raw // 4
        picks = 3 * int(m["openings"]) + 64
        replay = (levels + learn + picks * chunk) if oar else 0
        pairs_extra = (levels + kept + learn) if pairs >= 2 else 0
        scratch = int(m["scratch_bytes"] if m.get("scratch_bytes") else win)
        terms["gpu_staging"] = {"bytes": slots * win + scratch, "where": "device", "how": f"{slots} slots x {win >> 20} MiB window ring + {scratch >> 20} MiB scratch (P0, bounded)",
                                "measured_by": "footprint.jsonl device staging / window_ring_<kind>.json"}
        terms["host_staging"] = {"bytes": levels + kept + pairs_extra, "where": "host", "parts": {"levels": levels, "kept": kept, "pairs_extra": pairs_extra},
                                 "how": (f"levels = raw/4 ({levels >> 20} MiB) + kept = raw - excluded hidden ({kept >> 20} MiB, retain {retain}"
                                         + (f", exclude {sorted(excl)}" if excl else "") + f") + pairs {pairs_extra >> 20} MiB ({pairs} pairs)"),
                                 "measured_by": "footprint.jsonl pinned_pool_bytes (frozen pool) / verdict.host_peak"}
        terms["hash_backlog"] = {"bytes": learn, "where": "host", "how": f"learn budget {learn >> 20} MiB: the learning steps' windows awaiting their re-hash (pool window capacity)",
                                 "measured_by": "footprint.jsonl pinned_pool_in_use max during learning steps"}
        terms["opening_data"] = {"bytes": replay, "where": "host", "parts": {"levels": levels if oar else 0, "learn": learn if oar else 0, "picks_x_chunk": picks * chunk if oar else 0},
                                 "how": (f"openings by replay on a fresh pool: levels + learn + {picks} picks x {chunk} B chunk" if oar else "openings from retained host bytes: 0 extra"),
                                 "measured_by": "openings_after_release.host_rss_hwm_gb - pre-replay RSS"}
        raw_rec = {"raw_bound_bytes": raw, "hidden_bound_bytes": hidden_bound, "excluded_hidden_bytes": hidden_excluded, "hidden_factor": m["hidden_factor"], "other_factor": m["other_factor"]}
    else:
        lf = float(m["levels_factor"])
        block = int(sizing.get("prefill_block_bytes") or sizing.get("block_bytes") or peak_bytes)
        terms["gpu_staging"] = {"bytes": block, "where": "device", "how": f"whole-step device block (per-step-block path) {block >> 20} MiB", "measured_by": "arena_sizing block_bytes"}
        # chunk-leaf-v1 unbounded: `_PinnedPool.take` slices (every window's transient copy) are returned only by pool.reset() at the END
        # of the run, so pinned host grows with the run's raw bytes WHATEVER --retain says (FP8-STAGING §1 (2): B1 25.1 GB raw in 31.9 GB
        # of pool under retain none); retain host adds the levels (x1.25, F-dA-15)
        terms["host_staging"] = {"bytes": int(run_total * (lf if retain == "host" else 1.0)), "where": "host",
                                 "how": (f"v1 pool grows with the run's raw bytes {run_total >> 20} MiB (transient window copies held to pool.reset)"
                                         + (f" x {lf} for levels (retain host, F-dA-15)" if retain == "host" else f" (retain {retain} does not return them)")),
                                 "measured_by": "verdict.host_peak / rss timeline"}
        # the learning step's re-hash source: the whole step resident at once beside the pool (F-fp8-B8MEM); counted additively -- conservative
        # by construction, and the unbounded path has no record with a VmHWM to calibrate it (R16 B8 attempt 2: killed at RSS 126 GB and
        # climbing inside step 0 under --retain none)
        terms["hash_backlog"] = {"bytes": peak_bytes, "where": "host", "how": (f"the largest step's pinned D2H copy for its learning re-hash (chunk-leaf-v1, whatever --retain): "
                                                                             f"step {peak_step} {peak_bytes >> 20} MiB -- the F-fp8-B8MEM allocation"),
                                 "measured_by": "footprint.jsonl end_step n_bytes of the peak step / RSS at its end"}
        terms["opening_data"] = {"bytes": 0, "where": "host", "how": "unbounded: openings from the retained bytes / re-execution, no extra pool",
                                 "measured_by": "openings record"}
        raw_rec = {"raw_bound_bytes": run_total, "levels_factor": lf}
    terms["headroom"] = {"bytes": int(engine_rss_bytes) + int(m["slack_bytes"]) + int(others_current_bytes or 0), "where": "host",
                         "parts": {"engine_rss": int(engine_rss_bytes), "slack": int(m["slack_bytes"]), "others_current": int(others_current_bytes or 0)},
                         "how": (f"engine RSS at admission {int(engine_rss_bytes) >> 20} MiB + slack {int(m['slack_bytes']) >> 20} MiB (engine growth after admission, "
                                 f"measured B1) + co-tenants' memory.current {int(others_current_bytes or 0) >> 20} MiB"),
                         "measured_by": "rss timeline before the first pool block / cgroup memory.current of other pids"}
    host_bound = sum(int(terms[t]["bytes"]) for t in HOST_TERMS)

    # DOMINANT CONTRIBUTOR (R17-29 REQUIRED): the largest host term, and the largest single materialization with where this mode puts it
    dom_term = max(HOST_TERMS, key=lambda t: int(terms[t]["bytes"]))
    hid_peak = hidden_steps.get(peak_step, 0)
    if bounded and ({"fa3_hidden_m1", "fa2_hidden_m1"} & excl):
        where = "excluded from host (retain-exclude) -- streamed through the device window ring, opened by replay"
    elif bounded and retain == "host":
        where = "kept on host in 256 B chunks (retain host, bounded)"
    elif bounded:
        where = f"not retained (retain {retain}); learning windows within the learn budget only"
    elif retain == "host":
        where = "the whole step's D2H pinned copy for its learning re-hash, then retained on host (unbounded)"
    else:
        where = "the whole step's D2H pinned copy for its learning re-hash (unbounded; --retain does not remove it)"
    kind = ((sched.get("dominant") or {}).get("peak_step") or {}).get("kind") if int((sched.get("dominant") or {}).get("peak_step", {}).get("step", -1)) == peak_step else \
        (f"prefill step {peak_step}: {peak['prefills']} prompt(s), {peak['tokens'] - peak['decodes']:,} prompt tokens, K {peak['K']}" if peak["prefills"] else f"decode step {peak_step}: {peak['decodes']} rows, K {peak['K']}")
    materialization = {"family": f"hidden M1 stream ({attn}, {layers} layers)", "step": peak_step, "kind": kind, "tokens": int(peak["tokens"]),
                       "hidden_bytes": hid_peak, "hidden_bytes_per_sequence": per_seq_hidden.get(peak_step, 0), "hidden_bytes_padded": padded.get(peak_step, 0),
                       "geometry_used": ("padded" if hid_peak == padded.get(peak_step, 0) and hid_peak != per_seq_hidden.get(peak_step, 0) else "per_sequence"),
                       "step_bytes": peak_bytes, "where": where}
    dominant = {"host_term": dom_term, "host_term_bytes": int(terms[dom_term]["bytes"]), "host_term_share": (int(terms[dom_term]["bytes"]) / host_bound) if host_bound else None,
                "largest_materialization": materialization,
                "text": (f"{dom_term} {int(terms[dom_term]['bytes']) >> 20} MiB of {host_bound >> 20} MiB host bound ({100 * int(terms[dom_term]['bytes']) / host_bound:.0f} %): "
                         f"{terms[dom_term]['how']}; largest materialization: {materialization['family']}: {hid_peak / GiB:.1f} GiB ({kind}) -- {where}")}

    # decision on the CALIBRATED bound; the effective cap is what the cgroup leaves for this run
    # the calibration class: staging mode, retention, exclusion, and the PAIRS class (1 vs >= 2: a multi-pair pool has its own growth --
    # a factor measured on 1-pair records is never applied to a 3-pair launch)
    mode_key = f"{'bounded' if bounded else 'unbounded'}:{retain}" + (":exclude" if (bounded and excl) else "") + (":p1" if pairs < 2 else ":pN")
    cal = calibration_factor(calibration, mode_key)
    calibrated = int(host_bound * cal["factor"])
    cap_eff = (int(host_cap_bytes) - int(others_current_bytes or 0)) if host_cap_bytes is not None else None
    thr = int(cap_eff * (1.0 - float(margin))) if cap_eff is not None else None
    if cap_eff is None:
        decision, why = "WARN", "no host memory cap readable: nothing to compare against; the bound is on the record"
    elif calibrated > thr:
        decision = "REFUSE"
        drv = (f" (factor {cal['factor']:.3f} from the worst {mode_key} record: {cal.get('worst_record')})" if cal["calibrated"] else " (uncalibrated: the derivation alone)")
        why = (f"calibrated host bound {calibrated >> 20} MiB (predicted {host_bound >> 20} x {cal['factor']:.3f}){drv} > {1 - margin:.2f} x effective cap {cap_eff >> 20} MiB "
               f"(cgroup {int(host_cap_bytes) >> 20} - co-tenants {int(others_current_bytes or 0) >> 20}) -- predicted OOM, not launched (BRIEF §6 / R17-7.3); "
               f"DOMINANT: {dominant['text']}")
    # (no branch admits when the derivation exceeds the threshold: calibrated >= host_bound by construction, so host_bound > thr implies
    #  calibrated > thr -> REFUSE above.  The former "uncalibrated bound exceeds but the calibrated fits" WARN admitted runs the derivation
    #  refused, on a factor < 1 -- removed, revb n-fp8-plan-1.)
    elif not cal["calibrated"]:
        decision = "WARN"
        why = f"bound {host_bound >> 20} MiB fits {thr >> 20} MiB but {cal['why']}; DOMINANT: {dominant['text']}"
    elif cal.get("approx_over_records"):
        decision = "WARN"
        ao = "; ".join(f"{r['row']} x{r['ratio']:.3f}" for r in cal["approx_over_records"])
        why = (f"calibrated bound {calibrated >> 20} MiB (x {cal['factor']:.3f}) fits {thr >> 20} MiB, but an APPROXIMATE {mode_key} record over-ran its "
               f"derivation ({ao}; board-quoted RSS, no VmHWM -- a caution on the record, not a factor): launch with the timeline watched; DOMINANT: {dominant['text']}")
    else:
        decision = "ADMIT"
        why = f"calibrated bound {calibrated >> 20} MiB <= {thr >> 20} MiB (margin {((cap_eff - calibrated) / cap_eff) * 100:.1f} %); DOMINANT: {dominant['text']}"
    gpu_note = None
    if gpu_cap_bytes is not None:
        gpu_note = {"gpu_cap_bytes": int(gpu_cap_bytes), "fits": int(terms["gpu_staging"]["bytes"]) <= int(gpu_cap_bytes)}
        if not gpu_note["fits"]:
            decision, why = "REFUSE", f"device staging {int(terms['gpu_staging']['bytes']) >> 20} MiB > device cap {int(gpu_cap_bytes) >> 20} MiB; " + why

    # [R17-35 = Daniel, CONSOLIDATION-NOTE s7] every number on the record says WHAT KIND it is and in WHAT SCOPE, and the bound says which
    # lifetimes it ASSUMES overlap -- so an overestimate is traced to a term (duplicated accounting / an overlap that does not happen / a
    # step-sized allocation the implementation now streams) by `calibrate`, never absorbed into a multiplier.
    for t in TERMS:
        terms[t]["scope"] = TERM_SCOPE[t]
        terms[t]["kind"] = "conservative_bound (capacity: the term at its cap)"
    overlaps = lifetime_overlaps(bounded=bounded, retain=retain, excl=excl, open_after_release=oar, pairs=pairs)
    gpu_pred = None
    if sizing.get("device_total_bytes") and sizing.get("device_free_bytes") is not None:
        eng_used = int(sizing["device_total_bytes"]) - int(sizing["device_free_bytes"])
        gpu_pred = {"scope": "whole device (torch mem_get_info): engine + collector staging; co-tenant GPU processes would be inside the observed figure",
                    "kind": "conservative_bound", "device_total_bytes": int(sizing["device_total_bytes"]),
                    "engine_used_at_admission_bytes": eng_used, "collector_staging_bytes": int(terms["gpu_staging"]["bytes"]),
                    "device_used_peak_bytes": eng_used + int(terms["gpu_staging"]["bytes"]),
                    "gpu_memory_utilization_max": sizing.get("gpu_memory_utilization_max"),
                    "how": (f"device used at admission {eng_used >> 20} MiB (engine weights + KV + activations reserve, measured) + collector staging "
                            f"{int(terms['gpu_staging']['bytes']) >> 20} MiB ({terms['gpu_staging']['how']})"),
                    "measured_by": "footprint.jsonl cuda.device_total - min(cuda.device_free) (device) / max(cuda.reserved) - first reserved (collector delta)"}

    return {"schema": SCHEMA, "mode": {k: (sorted(v) if isinstance(v, (set, frozenset)) else v) for k, v in m.items()}, "mode_key": mode_key,
            "terms": terms, "host_bound_bytes": host_bound, "host_bound_mib": host_bound >> 20, "calibration": cal, "calibrated_bound_bytes": calibrated,
            "kind": {"host_bound_bytes": "conservative_bound: sum of the four host terms at their caps, all assumed co-resident at the peak (lifetime_overlaps_assumed)",
                     "calibrated_bound_bytes": (f"empirical_forecast: host_bound x the worst measured/predicted of the {mode_key} class ({cal['entries']} records)"
                                                if cal["calibrated"] else "none: uncalibrated class -- the conservative bound is the only number"),
                     "scope": "process VmHWM of the Commit (engine + collector + pools) in the cgroup, plus co-tenants' anon bytes; compared with verdict.host_peak.process_vmhwm_bytes"},
            "lifetime_overlaps_assumed": overlaps, "gpu_predicted": gpu_pred,
            "host_cap_bytes": host_cap_bytes, "host_cap_source": host_cap_source, "others_current_bytes": int(others_current_bytes or 0),
            "effective_cap_bytes": cap_eff, "threshold_bytes": thr, "margin": float(margin),
            "margin_bytes": (cap_eff - calibrated) if cap_eff is not None else None, "margin_frac": ((cap_eff - calibrated) / cap_eff) if cap_eff else None,
            "decision": decision, "admitted": decision != "REFUSE", "why": why, "dominant": dominant, "gpu": gpu_note,
            "schedule": {"requests": sched["requests"], "steps": sched["steps"], "horizon": sched["horizon"], "prompt_tokens": sched["prompt_tokens"],
                         "total_bytes_per_sequence_hidden": int(sched["total_bytes"]), "hidden_total_per_sequence": hidden_total_ps, "hidden_total_padded": hidden_total_pad,
                         "hidden_total_used": hidden_total, "other_total": other_total, "run_total_used": run_total, "peak_step": peak_step, "peak_step_bytes": peak_bytes,
                         "per_step_hidden_geometry": geometry_rule, "attn_backend": attn, "lag": int(lag)},
            **raw_rec,
            "rule": ("R17-29: one planner; refusal only on the calibrated bound (factor = max(1.0, worst EXACT measured/predicted of the mode's records, >= 2 records): calibration only raises), "
                     "warning otherwise; the bound is printed before the run it governs; a VmHWM over it is a finding against the derivation"),
            "_sched": sched}


def calibrate(p: dict[str, Any], record_dir: str | Path, *, row: str | None = None) -> dict[str, Any]:
    """Measured vs predicted, per term where the record measures it, for one Commit's `commit/` directory; returns the calibration ENTRY too."""
    d = Path(record_dir)
    out: dict[str, Any] = {"record_dir": str(d), "row": row, "mode": p["mode_key"], "predicted_bytes": int(p["host_bound_bytes"]), "measured": {}, "ratios": {}}
    vj = d / "verdict.json"
    if vj.exists():
        v = json.loads(vj.read_text())
        hp = v.get("host_peak") or {}
        if hp.get("process_vmhwm_bytes") is not None:
            out["measured"]["host_total_vmhwm_bytes"] = int(hp["process_vmhwm_bytes"])
            out["measured"]["record_admission_bound_bytes"] = hp.get("host_peak_bound_bytes")
    fp = d / "footprint.jsonl"
    if not fp.exists() and (d.parent / "footprint.jsonl").exists():
        fp = d.parent / "footprint.jsonl"                      # the measure arms write it beside commit/ (run_r17.sh measure); the row's Commit inside
    if fp.exists():
        out["footprint_path"] = str(fp)
        rss_max = pool_max = in_use_max = 0
        rss_first = pool_first = None
        step_bytes: dict[int, int] = {}
        dev_total = dev_free_min = dev_free_first = res_max = res_first = None
        # [R17-35] SEGMENTS: the footprint alternates run segments (the committer's pool: levels + kept + learn windows) and replay segments
        # (after `release_staging`: the frozen view keeps the run's pinned blocks, the replay gets a FRESH pool -- pinned_pool_bytes then reports
        # only the live pool).  The decomposition at the peak needs, per segment: its last rss / pool and its pool max.
        segs: list[dict[str, Any]] = [{"kind": "run", "pool_max": 0, "rss_max": 0, "rss_end": 0, "pool_end": 0, "rss_first": None, "pool_first": None}]
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            _rss, _pool = int(r.get("rss_bytes") or 0), int(r.get("pinned_pool_bytes") or 0)
            if r.get("event") == "release_staging":
                segs.append({"kind": "replay", "pool_max": 0, "rss_max": 0, "rss_end": 0, "pool_end": 0, "rss_first": None, "pool_first": None})
            elif (r.get("event") == "end_step" and int(r.get("step", -1)) == 0 and _rss and segs[-1]["rss_first"] is not None
                  and segs[-1]["kind"] == "replay" and _pool != segs[-1]["pool_end"]):
                # the next PAIR's run: a step-0 end_step on a fresh pool after a replay (no event marks it; the PAIRS=3 B1 footprint shows the
                # replay pool 6,022 -> the new run's 9,563 at its step 0, while a run's second iteration keeps its pool byte for byte)
                segs.append({"kind": "run", "pool_max": 0, "rss_max": 0, "rss_end": 0, "pool_end": 0, "rss_first": None, "pool_first": None})
            if _rss:
                sg = segs[-1]
                if sg["rss_first"] is None:
                    sg["rss_first"], sg["pool_first"] = _rss, _pool
                sg["rss_max"], sg["pool_max"] = max(sg["rss_max"], _rss), max(sg["pool_max"], _pool)
                sg["rss_end"], sg["pool_end"] = _rss, _pool
                if rss_first is None:
                    rss_first, pool_first = _rss, _pool                           # the process at its first collected step: engine + collector baseline
                rss_max = max(rss_max, _rss)
            pool_max = max(pool_max, _pool)
            in_use_max = max(in_use_max, int(r.get("pinned_pool_in_use") or 0))
            cu = r.get("cuda") or {}
            if cu.get("device_total"):
                dev_total = int(cu["device_total"])
                dev_free_first = int(cu["device_free"]) if dev_free_min is None else dev_free_first
                dev_free_min = int(cu["device_free"]) if dev_free_min is None else min(dev_free_min, int(cu["device_free"]))
                res_max = max(res_max or 0, int(cu.get("reserved") or 0))
                res_first = int(cu.get("reserved") or 0) if res_first is None else res_first
            if r.get("event") == "end_step":
                sb = r.get("staged_bytes", r.get("n_bytes", r.get("used")))       # native_collect end_step: staged_bytes (acquired), used (arena cursor)
                if sb is not None:
                    step_bytes[int(r.get("step", -1))] = max(int(sb), step_bytes.get(int(r.get("step", -1)), 0))   # the largest step of that index across pairs
        segs = [s for s in segs if s["rss_first"] is not None]
        pk_i = max(range(len(segs)), key=lambda i: segs[i]["rss_max"]) if segs else None
        at_peak: dict[str, Any] = {}
        if pk_i is not None:
            sg = segs[pk_i]
            run_sg = sg if sg["kind"] == "run" else (segs[pk_i - 1] if pk_i >= 1 else None)
            first_run = next((s for s in segs if s["kind"] == "run"), None)
            at_peak = {"segment": pk_i, "segment_kind": sg["kind"], "segments": len(segs), "rss_peak_bytes": sg["rss_max"],
                       "pair_of_peak": sum(1 for s in segs[:pk_i + 1] if s["kind"] == "run") - 1,        # 0-based pair whose run/replay holds the peak
                       "pairs_seen": sum(1 for s in segs if s["kind"] == "run"),
                       # the run's frozen pool (levels + kept) at the end of ITS segment -- still resident in a replay segment (the frozen view owns it)
                       "frozen_pool_bytes": int(run_sg["pool_end"]) if run_sg else 0,
                       "run_pool_max_bytes": int(run_sg["pool_max"]) if run_sg else 0,          # incl. the learn windows before their recycling
                       "replay_pool_bytes": int(sg["pool_max"]) if sg["kind"] == "replay" else 0,
                       # engine + collector baseline: the process minus its pool at the END of the FIRST run (before any pool was released to
                       # the cache) -- the same figure at the peak's own run carries every earlier pair's released pool, which is the remainder's
                       "baseline_end_of_run_bytes": (int(first_run["rss_end"]) - int(first_run["pool_end"])) if first_run else None,
                       "baseline_end_of_peak_run_bytes": (int(run_sg["rss_end"]) - int(run_sg["pool_end"])) if run_sg else None,
                       "baseline_first_step_bytes": (int(rss_first) - int(pool_first or 0)) if rss_first else None}
            at_peak["remainder_bytes"] = int(sg["rss_max"]) - int(at_peak["baseline_end_of_run_bytes"] or 0) - at_peak["frozen_pool_bytes"] - at_peak["replay_pool_bytes"]
        out["measured"].update({"rss_peak_footprint_bytes": rss_max, "rss_first_footprint_bytes": rss_first, "pinned_pool_first_bytes": pool_first,
                                "pinned_pool_peak_bytes": pool_max, "pinned_pool_in_use_peak_bytes": in_use_max, "at_peak": at_peak,
                                "segments": [{k: v for k, v in s.items()} for s in segs]})
        if dev_total and dev_free_min is not None:
            # [R17-35] GPU predicted vs observed in the SAME scope: whole device (engine + collector) and the collector's own delta
            gp = p.get("gpu_predicted") or {}
            out["gpu"] = {"observed_device_used_peak_bytes": dev_total - dev_free_min, "observed_process_reserved_peak_bytes": res_max,
                          "observed_process_reserved_first_bytes": res_first,
                          # the collector's window ring is a native cudaMalloc (outside torch's allocator): torch reserved undercounts it -- the DEVICE
                          # delta since the first footprint event (engine growth + ring + scratch) is the figure comparable with gpu_staging
                          "observed_device_delta_bytes": (dev_free_first - dev_free_min) if dev_free_first is not None else None,
                          "observed_torch_reserved_delta_bytes": (res_max - res_first) if (res_max is not None and res_first is not None) else None,
                          "predicted_device_used_peak_bytes": gp.get("device_used_peak_bytes"), "predicted_collector_staging_bytes": gp.get("collector_staging_bytes"),
                          "scope": ("device: torch mem_get_info total - min free over the footprint (engine + collector; co-tenants inside); delta: first-event free - min free "
                                    "(the collector's ring + scratch + engine growth; the ring is native cudaMalloc, not in torch reserved)")}
            if gp.get("device_used_peak_bytes"):
                out["gpu"]["ratio_device"] = round((dev_total - dev_free_min) / int(gp["device_used_peak_bytes"]), 4)
            if gp.get("engine_used_at_admission_bytes") is not None:
                # the ring is allocated at arm time, BEFORE the first footprint event: the delta comparable with gpu_staging is measured from the
                # device used at ADMISSION (arena_sizing device_free_bytes), and it also carries the engine's own growth after admission
                out["gpu"]["observed_device_delta_since_admission_bytes"] = (dev_total - dev_free_min) - int(gp["engine_used_at_admission_bytes"])
                if gp.get("collector_staging_bytes"):
                    out["gpu"]["ratio_device_delta_over_staging"] = round(out["gpu"]["observed_device_delta_since_admission_bytes"] / int(gp["collector_staging_bytes"]), 4)
        pk = int(p["schedule"]["peak_step"])
        if pk in step_bytes:
            out["measured"]["peak_step_staged_bytes"] = step_bytes[pk]
            out["ratios"]["peak_step_staged_over_predicted"] = round(step_bytes[pk] / p["schedule"]["peak_step_bytes"], 4) if p["schedule"]["peak_step_bytes"] else None
    sj = d / "steps.jsonl"
    if sj.exists() and p.get("_sched"):                       # a plan re-read from its JSON has no schedule table: soundness needs the live plan
        obs = observed_steps(sj)
        if obs:
            out["soundness_per_step"] = compare_observed(p["_sched"], obs)
    rj = d / "runs.jsonl"
    if rj.exists():
        # the hidden family's MEASURED bytes over the run (bounded_staging.excluded_bytes) and the tap's launch geometries (HB, M, NB, BN, D):
        # the direct answer to the per-sequence vs padded question for this backend
        for line in rj.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("arm") == "control":
                continue
            ex = ((r.get("commit") or {}).get("extra") or {})
            bs = ex.get("bounded_staging") or {}
            geo = {k: v for k, v in (ex.get("native_collect") or {}).get("fa2_wrap", {}).items() if isinstance(v, dict) and "geometries" in v}
            if bs.get("excluded_bytes") or geo:
                sc = p["schedule"]
                hid_meas = int(bs.get("excluded_bytes") or 0)
                out["measured"]["hidden_family_bytes_run"] = hid_meas or None
                out["measured"]["hidden_launch_geometries"] = {k: {"launches": v.get("launches"), "bytes": v.get("bytes"), "geometries": v.get("geometries")} for k, v in geo.items()}
                if hid_meas:
                    ps, pd = int(sc["hidden_total_per_sequence"]), int(sc["hidden_total_padded"])
                    out["ratios"]["hidden_measured_over_per_sequence"] = round(hid_meas / ps, 4) if ps else None
                    out["ratios"]["hidden_measured_over_padded"] = round(hid_meas / pd, 4) if pd else None
                    out["hidden_geometry_measured"] = (("indistinguishable (single-sequence steps: the geometries coincide)" if ps == pd else
                                                        ("per_sequence" if abs(hid_meas - ps) <= abs(hid_meas - pd) else "padded")) if (ps and pd) else None)
                break
    t = p["terms"]
    meas_total = out["measured"].get("host_total_vmhwm_bytes") or out["measured"].get("rss_peak_footprint_bytes")
    if meas_total:
        out["measured_bytes"], out["measured_by"] = int(meas_total), ("process VmHWM" if out["measured"].get("host_total_vmhwm_bytes") else "footprint rss peak")
        out["ratio"] = round(int(meas_total) / int(p["host_bound_bytes"]), 4)
        out["ratios"]["host_total_over_bound"] = out["ratio"]
    if out["measured"].get("pinned_pool_peak_bytes"):
        pinned_pred = int(t["host_staging"]["bytes"]) + int(t["hash_backlog"]["bytes"]) + int(t["opening_data"]["bytes"])
        out["ratios"]["pinned_pool_over_predicted_pool_terms"] = round(out["measured"]["pinned_pool_peak_bytes"] / pinned_pred, 4) if pinned_pred else None
    out["gap_accounting"] = gap_accounting(p, out["measured"])
    out["bound_vs_forecast"] = {"conservative_bound_bytes": int(p["host_bound_bytes"]), "empirical_forecast_bytes": int(p.get("calibrated_bound_bytes") or p["host_bound_bytes"]),
                                "forecast_is_calibrated": bool((p.get("calibration") or {}).get("calibrated")), "observed_bytes": out.get("measured_bytes"),
                                "observed_by": out.get("measured_by"), "scope": (p.get("kind") or {}).get("scope")}
    out["entry"] = {k: out.get(k) for k in ("row", "mode", "predicted_bytes", "measured_bytes", "measured_by", "ratio")} | {"record_dir": str(d)}
    return out


def gap_accounting(p: dict[str, Any], measured: dict[str, Any]) -> list[dict[str, Any]]:
    """[R17-35 = Daniel s7] WHERE an over- (or under-) estimate comes from, term by term at THE PEAK MOMENT of the record in the same scope,
    with the reason class named: `capacity_not_filled` (a cap the run did not reach: bound factors, learn budget, replay pool), `baseline`
    (engine + collector outside the pools vs headroom), `unmodelled` (bytes at the peak no term predicts: pools released to torch's pinned-host
    cache rather than the OS -- the warm-up pool, earlier pairs' pools -- which the pairs term covers only for pairs >= 2), so the next bound
    is fixed by accounting, never by a multiplier.  The observed rows sum to the footprint's RSS peak; TOTAL compares the bound with the VmHWM."""
    t = p["terms"]
    rows: list[dict[str, Any]] = []
    vm = measured.get("host_total_vmhwm_bytes") or measured.get("rss_peak_footprint_bytes")
    ap = measured.get("at_peak") or {}
    if not ap or not vm:
        return rows
    parts = t["host_staging"].get("parts") or {}
    pairs_extra = int(parts.get("pairs_extra") or 0)
    pool_pred = int(t["host_staging"]["bytes"]) - pairs_extra + int(t["hash_backlog"]["bytes"])       # levels + kept + learn: ONE frozen pool's capacity
    hparts = t["headroom"].get("parts") or {}
    base_pred = int(hparts.get("engine_rss") or 0) + int(hparts.get("slack") or 0)
    base_obs = int(ap.get("baseline_end_of_run_bytes") or 0)
    rows.append({"term": "headroom (engine_rss + slack)", "scope": t["headroom"]["scope"], "predicted_bytes": base_pred, "observed_bytes": base_obs,
                 "gap_bytes": base_pred - base_obs,
                 "reason": (f"baseline: engine RSS at admission + slack vs the process minus its pool at the end of the run "
                            f"(first step {int(ap.get('baseline_first_step_bytes') or 0) >> 20} MiB -> {base_obs >> 20} MiB: collector + CUDA growth during the run)"),
                 "measured_by": "footprint.jsonl rss_bytes - pinned_pool_bytes at the run segment's last end_step"})
    fz = int(ap.get("frozen_pool_bytes") or 0)
    rows.append({"term": "host_staging + hash_backlog (one frozen pool)", "scope": t["host_staging"]["scope"], "predicted_bytes": pool_pred, "observed_bytes": fz,
                 "gap_bytes": pool_pred - fz,
                 "reason": ((f"capacity_not_filled: levels + kept are bounds (hidden x {p.get('hidden_factor')}, other x {p.get('other_factor')}) and the learn budget is a capacity "
                            f"(pool max during the run {int(ap.get('run_pool_max_bytes') or 0) >> 20} MiB incl. learn windows; frozen {fz >> 20} MiB)") if pool_pred >= fz else
                            "UNDER: the frozen pool grew past levels + kept + learn -- a finding against the derivation (compare levels/kept with pinned_pool_in_use)"),
                 "measured_by": "footprint.jsonl pinned_pool_bytes at the run segment's last end_step (levels + kept after the learn windows recycled)"})
    rp = int(ap.get("replay_pool_bytes") or 0)
    op = int(t["opening_data"]["bytes"])
    if op or rp:
        rows.append({"term": "opening_data (replay pool, beside the frozen pool)", "scope": t["opening_data"]["scope"], "predicted_bytes": op, "observed_bytes": rp,
                     "gap_bytes": op - rp,
                     "reason": ("capacity_not_filled: the replay pool below levels + learn + picks x chunk; the OVERLAP with the run's frozen pool IS realized (the frozen view keeps it)"
                                if ap.get("segment_kind") == "replay" else
                                "overlap_not_realized: the process peak is inside a RUN segment, not the replay (the replay pool did not raise the peak)"),
                     "measured_by": "footprint.jsonl pinned_pool_bytes max in the replay segment holding the RSS peak"})
    rem = int(ap.get("remainder_bytes") or 0)
    rows.append({"term": "released pools at the peak (torch's pinned cache: warm-up pool, earlier pairs' pools, host copies)",
                 "scope": "process bytes at the peak outside the first run's baseline and the live pools",
                 "predicted_bytes": pairs_extra, "observed_bytes": rem, "gap_bytes": pairs_extra - rem,
                 "reason": ((f"pairs term: one extra set (levels + kept + learn) predicted for {int(ap.get('pairs_seen') or 1)} pairs vs the released pools actually held "
                            f"(pair {ap.get('pair_of_peak')}'s peak; the cache REUSES blocks of equal size, so earlier pools are partly recycled, not stacked)")
                           if pairs_extra else
                           ("unmodelled: pool.reset returns pinned blocks to torch's CachingHostAllocator, not to the OS -- the learn-only warm-up pool stays resident under "
                            "the run's pool and no term predicts it at 1 pair")),
                 "measured_by": "rss peak - first run's baseline - frozen pool - replay pool"})
    obs_sum = sum(int(r["observed_bytes"]) for r in rows)
    rows.append({"term": "TOTAL", "scope": (p.get("kind") or {}).get("scope"), "predicted_bytes": int(p["host_bound_bytes"]), "observed_bytes": int(vm),
                 "gap_bytes": int(p["host_bound_bytes"]) - int(vm),
                 "reason": (f"sum (rows observed = footprint RSS peak {obs_sum >> 20} MiB; VmHWM {int(vm) >> 20} MiB; co-tenant term {int(hparts.get('others_current') or 0) >> 20} MiB "
                            "predicted is outside the process); the largest positive gap above names where the conservative bound is loose, a negative one what it misses"),
                 "measured_by": "process VmHWM" if measured.get("host_total_vmhwm_bytes") else "footprint rss peak"})
    return rows


def _load_workload_sizing(wl_path: str, sizing_path: str) -> tuple[dict, dict, int | None]:
    wl = json.loads(Path(wl_path).read_text())
    sizing = json.loads(Path(sizing_path).read_text())
    mnbt = (wl.get("engine_args_required") or {}).get("max_num_batched_tokens") or (sizing.get("envelope") or {}).get("max_num_batched_tokens")
    return wl, sizing, mnbt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workload", required=True); ap.add_argument("--sizing", required=True)
    ap.add_argument("--bounded", action="store_true"); ap.add_argument("--retain", default="host", choices=["host", "none", "sample"])
    ap.add_argument("--exclude", default="", help="comma-separated families excluded from host retention (bounded)")
    ap.add_argument("--open-after-release", action="store_true"); ap.add_argument("--openings", type=int, default=64); ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--window-mb", type=int, default=256); ap.add_argument("--slots", type=int, default=8); ap.add_argument("--learn-mb", type=int, default=4096)
    ap.add_argument("--engine-rss-mb", type=int, default=2200); ap.add_argument("--cap-bytes", type=int, default=None); ap.add_argument("--others-mb", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.10); ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--calibration", default=None, help="jsonl of calibration entries (one per Commit: mode, predicted_bytes, measured_bytes, ratio)")
    ap.add_argument("--record", default=None, help="a Commit's commit/ directory: report measured vs predicted per term + the calibration entry")
    ap.add_argument("--row", default=None); ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    wl, sizing, mnbt = _load_workload_sizing(a.workload, a.sizing)
    cap, src = (a.cap_bytes, "--cap-bytes") if a.cap_bytes is not None else host_cap()
    entries = [json.loads(l) for l in Path(a.calibration).read_text().splitlines() if l.strip()] if a.calibration else None
    mode = {"retain": a.retain, "bounded": a.bounded, "retain_exclude": [x for x in a.exclude.split(",") if x], "open_after_release": a.open_after_release,
            "openings": a.openings, "pairs": a.pairs, "window_bytes": a.window_mb * MiB, "slots": a.slots, "learn_budget_bytes": a.learn_mb * MiB}
    p = plan(sizing, wl["requests"], mode=mode, engine_rss_bytes=a.engine_rss_mb * MiB, host_cap_bytes=cap, host_cap_source=src,
             others_current_bytes=a.others_mb * MiB, max_num_batched_tokens=mnbt, lag=a.lag, margin=a.margin, calibration=entries)
    cal = calibrate(p, a.record, row=a.row or (wl.get("sweep") or {}).get("row_id")) if a.record else None
    p.pop("_sched", None)
    if a.json:
        print(json.dumps({"plan": p, "calibration_record": cal}, indent=1, default=str))
        return 0 if p["admitted"] else 3
    print(f"{(wl.get('sweep') or {}).get('row_id') or Path(a.workload).stem}: mode {p['mode_key']}, {p['schedule']['requests']} requests, {p['schedule']['steps']} steps, "
          f"peak step {p['schedule']['peak_step']} {p['schedule']['peak_step_bytes'] >> 20} MiB (hidden per-seq {p['dominant']['largest_materialization']['hidden_bytes_per_sequence'] >> 20} / "
          f"padded {p['dominant']['largest_materialization']['hidden_bytes_padded'] >> 20} MiB -> {p['dominant']['largest_materialization']['geometry_used']})")
    for t in TERMS:
        print(f"  {t:13s} {int(p['terms'][t]['bytes']) >> 20:>10,} MiB  [{p['terms'][t]['where']}]  {p['terms'][t]['how']}")
    print(f"  HOST BOUND {p['host_bound_bytes'] >> 20:,} MiB; calibrated x{p['calibration']['factor']:.3f} = {p['calibrated_bound_bytes'] >> 20:,} MiB; cap {((cap or 0) >> 20):,} MiB ({src}) "
          f"- others {a.others_mb:,} -> threshold {((p['threshold_bytes'] or 0) >> 20):,} MiB")
    print(f"  {p['decision']}: {p['why']}")
    if cal:
        print(f"  RECORD {a.record}: measured {cal.get('measured_bytes', 0) >> 20 if cal.get('measured_bytes') else None} MiB ({cal.get('measured_by')}) / predicted "
              f"{cal['predicted_bytes'] >> 20:,} MiB = {cal.get('ratio')}; ratios {cal['ratios']}; per-step soundness "
              f"{(cal.get('soundness_per_step') or {}).get('sound')} (under-predicted {(cal.get('soundness_per_step') or {}).get('steps_under_predicted')})")
        for g in cal.get("gap_accounting") or []:        # [R17-35] where the bound is loose, term by term, same scope
            print(f"    {g['term']:44s} predicted {int(g['predicted_bytes']) >> 20:>9,} MiB  observed {int(g['observed_bytes']) >> 20:>9,} MiB  gap {int(g['gap_bytes']) >> 20:>+9,} MiB  {g['reason']}")
        if cal.get("gpu"):
            gg = cal["gpu"]
            print(f"    GPU device used peak: predicted {((gg.get('predicted_device_used_peak_bytes') or 0) >> 20):,} MiB observed {gg['observed_device_used_peak_bytes'] >> 20:,} MiB "
                  f"(ratio {gg.get('ratio_device')}); collector staging predicted {((gg.get('predicted_collector_staging_bytes') or 0) >> 20):,} MiB vs device delta since admission "
                  f"{((gg.get('observed_device_delta_since_admission_bytes') or 0) >> 20):,} MiB (ratio {gg.get('ratio_device_delta_over_staging')}: ring + scratch + engine growth; "
                  f"since the first step {((gg.get('observed_device_delta_bytes') or 0) >> 20):,} MiB, torch reserved delta {((gg.get('observed_torch_reserved_delta_bytes') or 0) >> 20):,} MiB)")
        print(f"  ENTRY {json.dumps(cal['entry'])}")
    return 0 if p["admitted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
