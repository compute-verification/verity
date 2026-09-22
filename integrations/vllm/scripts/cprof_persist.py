#!/usr/bin/env python3
"""CPERF 2026-09-17 addendum: bytes committed / copied / retained / published for the pinned B0 store, the resident
state of the production ``MerkleTree`` vs the streaming builder (tracemalloc, synthetic trees of the real shapes),
and the replay-horizon curve from the measured per-RU timings.  Runs on the laptop from the JSON artifacts only.

    python scripts/cprof_persist.py --census docs/data/commitment-prof-2026-09-17/census-compact.json \
        --bench docs/data/commitment-prof-2026-09-17/bench-1.json --overlap docs/data/commitment-prof-2026-09-17/overlap-heavy.json \
        --out docs/data/commitment-prof-2026-09-17/persistence.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

from verity_vllm.commit import MerkleTree  # noqa: E402
from verity_vllm.commit_bench import fasttree as ft  # noqa: E402
from verity_vllm.commit_bench.stream import StreamingMerkle  # noqa: E402

PROFILE = "ec44fc6dd686"


def resident_state(n: int, tracked: bool = False) -> dict:
    ids = [f"p={PROFILE}/r=472ba20e-f4a0-5b64-90fe-ddb4971deff7/t=0/ph=decode/l={i % 30}/op=layers.{i % 30}.mlp.down_proj.gemm/port={i}" for i in range(n)]
    ids.sort()
    headers = [ft.leaf_header(c, "bfloat16", (576,)) for c in ids]
    value = bytes(1152)
    digests = [ft.leaf_digest(h, value) for h in headers]
    tracemalloc.start()
    t0 = tracemalloc.take_snapshot()
    tree = MerkleTree(dict(zip(ids, digests, strict=True)), "ru/x/000000")
    cur, peak_prod = tracemalloc.get_traced_memory()
    t1 = tracemalloc.take_snapshot()
    prod_bytes = sum(s.size_diff for s in t1.compare_to(t0, "filename"))
    tracemalloc.stop()
    del tree
    tracemalloc.start()
    b = StreamingMerkle(track=n // 2 if tracked else None)
    t0 = tracemalloc.take_snapshot()
    peak_digests = 0
    for d in digests:
        b.add_digest(d)
        peak_digests = max(peak_digests, b.resident_digests())
    b.finish()
    t1 = tracemalloc.take_snapshot()
    stream_bytes = sum(s.size_diff for s in t1.compare_to(t0, "filename"))
    tracemalloc.stop()
    return {"leaves": n, "production_MerkleTree_resident_bytes": prod_bytes, "production_peak_bytes": peak_prod,
            "production_holds": "leaves dict (id -> digest), parsed ids, all levels (2n digests), index map",
            "levels_digest_bytes": 32 * (2 * n - 1), "id_text_bytes": sum(len(c) for c in ids),
            "streaming_resident_bytes": stream_bytes, "streaming_peak_digests": peak_digests, "streaming_state_bytes_at_peak": 32 * peak_digests}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--overlap", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = json.loads(Path(a.census).read_text())
    b = json.loads(Path(a.bench).read_text())
    ov = json.loads(Path(a.overlap).read_text())
    t = c["totals"]
    ru, bd, w = t["ru"], t["boundary"], t["weights"]
    n_ru = ru["trees"]
    # ---- four-way distinction for the pinned store (RU trees = the q-scaled second stage; boundary = first stage) ----
    committed_logical = ru["raw_bytes"] + bd["raw_bytes"]
    committed_framed = ru["framed_preimage_bytes"] + bd["framed_preimage_bytes"]
    copied_current = {
        "d2h_pinned_arena": ru["raw_bytes"], "tobytes_in_emit": ru["raw_bytes"], "values_bin_write": t["ru_values_bin_bytes"],
        "memmap_slice_to_bytes_at_commit": ru["raw_bytes"], "total_host_bytes_touched": 4 * ru["raw_bytes"],
    }
    retained_current = {
        "values_bin": t["ru_values_bin_bytes"], "index_json": t["ru_index_json_bytes"], "commitments_json": t["commitments_json_bytes"],
        "leaf_hashes_in_index_json_hex": 64 * ru["leaves"], "tree_levels_if_kept_resident": 32 * (2 * ru["leaves"] - n_ru),
        "weights_values_bin": w["values_bin_bytes"], "weights_index_json": w["index_json_bytes"],
        "total_ru_store": t["ru_values_bin_bytes"] + t["ru_index_json_bytes"] + t["commitments_json_bytes"],
    }
    roots_only = {
        "ru_roots": 32 * n_ru, "ru_leaf_counts_u32": 4 * n_ru, "ru_ids_digests": 32 * n_ru,
        "boundary_roots": 32 * bd["trees"], "weights_root": 32, "per_ru_record_json_as_today": t["commitments_json_bytes"] / (n_ru + bd["trees"] + 1),
        "total_binary": 32 * n_ru + 4 * n_ru + 32 * n_ru + 32 * bd["trees"] + 32,
        "total_commitments_json_as_today": t["commitments_json_bytes"],
    }
    published = {"roots_in_transcript_binary": roots_only["total_binary"], "note": "what the verifier must see before the challenge: every RU root + boundary roots + weights root (+ ids digests today)"}
    # ---- resident state of the builders ----
    resident = {"decode_ru_364": resident_state(364), "prefill_ru_11586": resident_state(11586), "prefill_ru_11586_tracked": resident_state(11586, tracked=True)}
    # ---- replay horizon: measured per-RU costs ----
    per = ov["C_per_tree_compact"]
    dec = [v["done"] - v["seen"] for v in per.values() if v["leaves"] < 1000]
    pre = [v["done"] - v["seen"] for v in per.values() if v["leaves"] >= 1000]
    A = ov["A_replay_as_shipped"]["timing"]
    run_s = A["run_s"]
    fwd_s = A["capture"]["forward_s"]
    steps = 17  # engine steps of the workload (16 decode + 1 prefill batch); 64 RUs = 4 requests x 16 steps
    # per step-RU: lower bound = forward only (values regenerated on the GPU, hashed in stream); upper = the full
    # capture replay step as shipped (emit bookkeeping + store writes included)
    fwd_per_ru, run_per_ru = fwd_s / n_ru, run_s / n_ru
    kv_first_stage = c["plan_step_RU"]["a_unconditional"]["kv_rows"]["raw_bytes"] + c["plan_step_RU"]["a_unconditional"]["sampled_tokens"]["raw_bytes"]
    med_dec, max_pre = statistics.median(dec), max(pre)

    def rehash(prefills: int, decodes: int) -> float:
        return prefills * max_pre + decodes * med_dec

    horizon = [
        {"epoch": "root per step-RU + retained KV/sampled-token boundary (the (request,step) first stage)", "roots": n_ru,
         "root_metadata_bytes": n_ru * 68, "retained_boundary_bytes": kv_first_stage, "replay_window": "1 step-RU",
         "replay_s_C": [fwd_per_ru, run_per_ru], "rehash_s_C": rehash(0, 1), "rehash_prefill_s_C": rehash(1, 0)},
        {"epoch": "root per step-RU, nothing retained (KV prefix regenerated from the prompt)", "roots": n_ru, "root_metadata_bytes": n_ru * 68,
         "retained_boundary_bytes": 0, "replay_window": "prefill + up to 15 decode steps of the request",
         "replay_s_C": [16 * fwd_per_ru, 16 * run_per_ru], "rehash_s_C": rehash(0, 1)},
        {"epoch": "root per request (record_v5 RU), nothing retained", "roots": 4, "root_metadata_bytes": 4 * 68, "retained_boundary_bytes": 0,
         "replay_window": "the whole request", "replay_s_C": [16 * fwd_per_ru, 16 * run_per_ru], "rehash_s_C": rehash(1, 15)},
        {"epoch": "root per 4 steps, nothing retained", "roots": 16, "root_metadata_bytes": 16 * 68, "retained_boundary_bytes": 0,
         "replay_window": "prefill + up to 15 decode steps (KV prefix)", "replay_s_C": [16 * fwd_per_ru, 16 * run_per_ru], "rehash_s_C": rehash(1, 3)},
        {"epoch": "one root per run", "roots": 1, "root_metadata_bytes": 68, "retained_boundary_bytes": 0, "replay_window": "the whole run",
         "replay_s_C": [fwd_s, run_s], "rehash_s_C": rehash(4, 60)},
    ]
    doc = {
        "schema": "veritor/cperf/persistence/v1",
        "bytes": {"committed_logical_raw": committed_logical, "committed_framed_preimage": committed_framed, "copied_current_impl": copied_current,
                  "retained_current_impl": retained_current, "retained_root_only_design": roots_only, "published": published,
                  "ratio_retained_current_over_root_only": retained_current["total_ru_store"] / roots_only["total_binary"]},
        "resident_state": resident,
        "replay_horizon": {"source": "overlap-heavy.json A (H2 cp-cov-f heavy): run_s, C watcher per-tree build times [M]; combination [C]",
                           "replay_run_s_M": run_s, "forward_s_M": fwd_s, "engine_steps": steps, "forward_s_per_step_RU_C": fwd_per_ru, "run_s_per_step_RU_C": run_per_ru, "engine_build_s_M_H1": 7.419,
                           "rehash_decode_ru_median_s_M": statistics.median(dec), "rehash_prefill_ru_max_s_M": max(pre),
                           "curve": horizon},
        "bench_refs": {"packed_ru_trees_s": b["ablate"]["4_packed_identity_sha256"]["median_s"], "fold_only_s": b["ablate"]["5d_trusted_index_packed_fold"]["median_s"]},
    }
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
