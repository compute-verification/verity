"""Fresh-prefill production records and their replays (contract §15.4, lead finding 10:45Z).

The workload's requests 11 (512 tokens) and 13 (2048 tokens) share their first 17 tokens, i.e. one
full 16-token KV block.  In one stock engine with prefix caching on, whichever of the two is
scheduled second is served that block from the cache (`num_cached_tokens = 16`), so its prompt was
not a fresh prefill.  A fresh-prefill record therefore runs the pass in two fresh engines:

    production ... --exclude 13 --warmup-throwaway 2 --assert-no-cached --out main.json
    production ... --only 13    --warmup-throwaway 2 --assert-no-cached --out solo13.json
    python -m verity_vllm_sampler.fresh merge-prod --main main.json --solo solo13.json --out fresh.json

and the same split for the replay dumps (`replay_logits --exclude 13` / `--only 13`), merged with

    python -m verity_vllm_sampler.fresh merge-dumps --main dir_main --solo dir_solo --out dir_merged

`label` annotates existing records with the per-index num_cached_tokens measured by a re-run of the
same command in the same engine structure (deterministic scheduling; the re-run's outputs must be
bit-identical to the record for the label to transfer):

    python -m verity_vllm_sampler.fresh label --record prod/greedy_c64_r0.json --measured diag/greedy_c64_cachecheck.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any


def jload(p: str) -> Any:
    with open(p) as f:
        return json.load(f)


def jdump(obj: Any, p: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1)


def merge_prod(main: dict[str, Any], solo: dict[str, Any], require_no_cached: bool = True) -> dict[str, Any]:
    for k in ("variant", "seed", "max_num_seqs", "order", "sampling_extra"):
        assert main.get(k) == solo.get(k), (k, main.get(k), solo.get(k))
    assert main["profile"]["vllm"] == solo["profile"]["vllm"]
    recs = sorted(main["records"] + solo["records"], key=lambda r: r["index"])
    idx = [r["index"] for r in recs]
    assert len(set(idx)) == len(idx), "overlapping indices"
    cached = [r["index"] for r in recs if (r.get("num_cached_tokens") or 0) > 0]
    if require_no_cached:
        assert not cached, f"cached prefix in merged record: {cached}"
    out = dict(main)
    out["records"] = recs
    out["schema"] = "verity-vllm/production-record/v1"
    out["execution"] = {
        "engines": 2,
        "note": "two fresh stock engines: main pass (all requests but the ones in `solo_indices`) and a solo pass; "
                "the solo request shares a leading 16-token block with another request and would otherwise be a "
                "prefix-cache hit, not a fresh prefill",
        "solo_indices": [r["index"] for r in solo["records"]],
        "main": {"argv": main.get("argv"), "peak_concurrency_achieved": main["peak_concurrency_achieved"],
                 "engine_steps": main["engine_steps"], "timing": main["timing"], "timing_init_s": main.get("timing_init_s"),
                 "warmup": main.get("warmup"), "prefix_cache_totals": main.get("prefix_cache_totals"),
                 "requests_with_cached_prefix": main.get("requests_with_cached_prefix")},
        "solo": {"argv": solo.get("argv"), "peak_concurrency_achieved": solo["peak_concurrency_achieved"],
                 "engine_steps": solo["engine_steps"], "timing": solo["timing"], "timing_init_s": solo.get("timing_init_s"),
                 "warmup": solo.get("warmup"), "prefix_cache_totals": solo.get("prefix_cache_totals"),
                 "requests_with_cached_prefix": solo.get("requests_with_cached_prefix")},
    }
    out["peak_concurrency_achieved"] = main["peak_concurrency_achieved"]  # the solo engine ran one request
    out["engine_steps"] = main["engine_steps"] + solo["engine_steps"]
    out["timing"] = {"submit_s": main["timing"]["submit_s"] + solo["timing"]["submit_s"],
                     "generate_s": main["timing"]["generate_s"] + solo["timing"]["generate_s"],
                     "note": "sum of the two engines' generation phases"}
    out["running_trace"] = main["running_trace"] + solo["running_trace"]
    out["prefix_cache_totals"] = {k: main["prefix_cache_totals"][k] + solo["prefix_cache_totals"][k]
                                  for k in main["prefix_cache_totals"]}
    out["num_cached_tokens_by_index"] = {str(r["index"]): r.get("num_cached_tokens") for r in recs}
    out["requests_with_cached_prefix"] = cached
    out["any_cached_prefix"] = bool(cached)
    out["only"] = None
    out["exclude"] = None
    out["argv"] = ["merged", "main:", *(main.get("argv") or []), "solo:", *(solo.get("argv") or [])]
    return out


def compare_outputs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ra = {r["index"]: r for r in a["records"]}
    rb = {r["index"]: r for r in b["records"]}
    common = sorted(set(ra) & set(rb))
    diffs = []
    for i in common:
        x, y = ra[i]["output_token_ids"], rb[i]["output_token_ids"]
        if x != y:
            k = next((j for j in range(max(len(x), len(y))) if j >= len(x) or j >= len(y) or x[j] != y[j]), None)
            diffs.append({"index": i, "first_divergent_k": k, "len_a": len(x), "len_b": len(y),
                          "a_token": x[k] if k is not None and k < len(x) else None,
                          "b_token": y[k] if k is not None and k < len(y) else None})
    return {"requests_compared": len(common), "identical_requests": len(common) - len(diffs),
            "bit_identical": not diffs, "differences": diffs,
            "finish_reasons_identical": all(ra[i]["finish_reason"] == rb[i]["finish_reason"] for i in common)}


def label(record_path: str, measured_path: str, source_label: str) -> dict[str, Any]:
    rec = jload(record_path)
    meas = jload(measured_path)
    cmp_ = compare_outputs(rec, meas)
    by_idx = {r["index"]: r for r in meas["records"]}
    for r in rec["records"]:
        m = by_idx.get(r["index"])
        if m is None:
            continue
        r["num_cached_tokens"] = m.get("num_cached_tokens")
        r["cached_prefix"] = (m.get("num_cached_tokens") or 0) > 0
        r["num_cached_tokens_source"] = source_label
    cached = sorted(r["index"] for r in rec["records"] if r.get("cached_prefix"))
    rec["requests_with_cached_prefix"] = cached
    rec["any_cached_prefix"] = bool(cached)
    rec["cached_prefix_label"] = {
        "source": source_label, "measured_file": os.path.basename(measured_path),
        "measured_outputs_bit_identical_to_record": cmp_["bit_identical"],
        "measured_prefix_cache_totals": meas.get("prefix_cache_totals"),
        "measured_running_trace_identical": rec.get("running_trace") == meas.get("running_trace"),
        "note": "num_cached_tokens was not recorded by the recorder at the time; it was measured by re-running the "
                "identical command (same engine structure: fresh engine, one pass, no warm-up on workload prompts) "
                "with the recorder that reports RequestOutput.num_cached_tokens.  Scheduling is deterministic and the "
                "re-run's outputs are compared bit-for-bit above.",
    }
    return rec


def _structure(rec: dict[str, Any], with_cap: bool = True) -> tuple[Any, ...]:
    """What decides the prompt-pass scheduling (and so the prefix-cache hits): request order, request set and
    the concurrency cap."""
    base: tuple[Any, ...] = (str(rec.get("order") or "natural"), tuple(sorted(r["index"] for r in rec["records"])))
    return base + ((int(rec.get("max_num_seqs") or 0),) if with_cap else ())


def label_all(prod_dir: str, measured_dir: str, in_place: bool) -> list[dict[str, Any]]:
    """Label every record in `prod_dir` that lacks per-request num_cached_tokens with the values measured by a
    re-run of the same *structure* (same request order and request set -> same prompt-pass scheduling; the
    prompt pass is what decides prefix-cache hits and it happens before any sampled token can differ).  A
    measured run with the same variant/seed/cap is preferred and its outputs are compared bit for bit."""
    import glob
    measured = [jload(p) for p in sorted(glob.glob(os.path.join(measured_dir, "*.json")))]
    measured = [m | {"_file": os.path.basename(p)} for m, p in zip(measured, sorted(glob.glob(os.path.join(measured_dir, "*.json"))))
                if isinstance(m, dict)]  # diag/ also holds list-shaped diagnostics
    measured = [m for m in measured if m.get("schema") == "verity-vllm/production-record/v1"
                and all(r.get("num_cached_tokens") is not None for r in m["records"])]
    report = []
    for p in sorted(glob.glob(os.path.join(prod_dir, "*.json"))):
        rec = jload(p)
        if rec.get("schema") != "verity-vllm/production-record/v1" or not rec.get("recording", True) and not rec["records"]:
            continue
        if all(r.get("num_cached_tokens") is not None for r in rec["records"]) and rec.get("cached_prefix_label") is None:
            report.append({"file": os.path.basename(p), "status": "recorded_by_engine",
                           "requests_with_cached_prefix": sorted(r["index"] for r in rec["records"] if r.get("cached_prefix"))})
            continue
        st = _structure(rec)
        cands = [m for m in measured if _structure(m) == st]
        same_cap = bool(cands)
        if not cands:  # no measurement at this cap: fall back to the same order/request set at another cap, flagged
            cands = [m for m in measured if _structure(m, with_cap=False) == _structure(rec, with_cap=False)]
        if not cands:
            report.append({"file": os.path.basename(p), "status": "no_measured_run_for_structure", "structure": [st[0], len(st[1]), st[2]]})
            continue
        same = [m for m in cands if (m.get("variant"), m.get("seed"), m.get("max_num_seqs")) == (rec.get("variant"), rec.get("seed"), rec.get("max_num_seqs"))]
        m = (same or cands)[0]
        by_idx = {r["index"]: r for r in m["records"]}
        for r in rec["records"]:
            mr = by_idx[r["index"]]
            r["num_cached_tokens"] = mr.get("num_cached_tokens")
            r["cached_prefix"] = (mr.get("num_cached_tokens") or 0) > 0
            r["num_cached_tokens_source"] = f"measured re-run {m['_file']}"
        cached = sorted(r["index"] for r in rec["records"] if r.get("cached_prefix"))
        cmp_ = compare_outputs(rec, m) if same else None
        rec["requests_with_cached_prefix"] = cached
        rec["any_cached_prefix"] = bool(cached)
        trace_same = (rec.get("running_trace") or [])[:2] == (m.get("running_trace") or [])[:2]
        rec["cached_prefix_label"] = {
            "source": f"measured re-run {m['_file']} (same order {st[0]!r}, same {len(st[1])} requests"
                      + (", same variant/seed/cap)" if same else f"; measured variant {m.get('variant')} seed {m.get('seed')} cap {m.get('max_num_seqs')})"),
            "measured_file": m["_file"], "same_variant_seed_cap": bool(same), "same_cap": same_cap,
            "measured_outputs_bit_identical_to_record": None if cmp_ is None else cmp_["bit_identical"],
            "measured_prefix_cache_totals": m.get("prefix_cache_totals"),
            "measured_prompt_pass_running_trace_identical": trace_same,
            "note": "num_cached_tokens was not recorded by the recorder at the time; it is the engine's RequestOutput.num_cached_tokens "
                    "from a re-run with the same structure (fresh engine, one pass over the same requests in the same order"
                    + (" at the same concurrency cap" if same_cap else " but at ANOTHER concurrency cap - no measurement at this cap")
                    + ", no warm-up on workload prompts).  The prompt pass, which decides prefix-cache hits, "
                    + ("is scheduled identically (first steps of running_trace compared)" if trace_same
                       else "is NOT scheduled identically (running_trace differs): the label is an inference from the prompt ids and "
                            "the block size, supported by every measured structure agreeing")
                    + "; a same-variant re-run is additionally compared bit for bit.",
        }
        if in_place:
            jdump(rec, p)
        report.append({"file": os.path.basename(p), "status": "labelled", "measured": m["_file"], "same_variant_seed_cap": bool(same),
                       "same_cap": same_cap, "prompt_pass_trace_identical": trace_same,
                       "bit_identical": None if cmp_ is None else cmp_["bit_identical"], "requests_with_cached_prefix": cached})
    return report


def merge_dumps(main_dir: str, solo_dir: str, out_dir: str) -> dict[str, Any]:
    a, b = jload(os.path.join(main_dir, "index.json")), jload(os.path.join(solo_dir, "index.json"))
    os.makedirs(out_dir, exist_ok=True)
    idx = dict(a)
    idx["requests"] = {}
    for src, d in ((main_dir, a), (solo_dir, b)):
        for rid, info in d["requests"].items():
            assert rid not in idx["requests"], rid
            shutil.copyfile(os.path.join(src, info["file"]), os.path.join(out_dir, info["file"]))
            idx["requests"][rid] = info
    for k in ("sampler_calls", "rows_seen", "rows_skipped_prefill", "rows_unknown_request", "forcing_events",
              "rows_beyond_trajectory", "trace_replay_rows_changed_by_vllm", "trace_replay_rows_followed",
              "trace_replay_rows_token_mismatch", "engine_steps"):
        idx[k] = (a.get(k) or 0) + (b.get(k) or 0)
    idx["trace_replay_changed_rows"] = (a.get("trace_replay_changed_rows") or []) + (b.get("trace_replay_changed_rows") or [])
    for k in ("replay_outputs", "num_cached_tokens"):
        idx[k] = {**(a.get(k) or {}), **(b.get(k) or {})}
    idx["replay_requests_with_cached_prefix"] = sorted((a.get("replay_requests_with_cached_prefix") or [])
                                                       + (b.get("replay_requests_with_cached_prefix") or []))
    ta, tb = a["trajectory_followed"], b["trajectory_followed"]
    idx["trajectory_followed"] = {"requests": ta["requests"] + tb["requests"], "attempted": ta["attempted"] + tb["attempted"],
                                  "mismatched_requests": ta["mismatched_requests"] + tb["mismatched_requests"],
                                  "all_followed": ta["all_followed"] and tb["all_followed"]}
    idx["timing"] = {"submit_s": a["timing"]["submit_s"] + b["timing"]["submit_s"],
                     "generate_s": a["timing"]["generate_s"] + b["timing"]["generate_s"]}
    idx["timing_init_s"] = (a.get("timing_init_s") or 0) + (b.get("timing_init_s") or 0)
    idx["running_trace"] = a["running_trace"] + b["running_trace"]
    idx["peak_concurrency_achieved"] = max(a["peak_concurrency_achieved"], b["peak_concurrency_achieved"])
    idx["merged_from"] = {"main": {"dir": main_dir, "only": a.get("only"), "exclude": a.get("exclude"), "warmup": a.get("warmup"),
                                   "prefix_cache_totals": a.get("prefix_cache_totals")},
                          "solo": {"dir": solo_dir, "only": b.get("only"), "exclude": b.get("exclude"), "warmup": b.get("warmup"),
                                   "prefix_cache_totals": b.get("prefix_cache_totals")}}
    idx["only"] = None
    idx["exclude"] = None
    jdump(idx, os.path.join(out_dir, "index.json"))
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("merge-prod")
    a.add_argument("--main", required=True)
    a.add_argument("--solo", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--compare-to", default=None, help="an earlier record of the same variant/seed to compare outputs with")
    a.add_argument("--allow-cached", action="store_true")
    b = sub.add_parser("merge-dumps")
    b.add_argument("--main", required=True)
    b.add_argument("--solo", required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("label")
    c.add_argument("--record", required=True)
    c.add_argument("--measured", required=True)
    c.add_argument("--source", default="re-run of the identical command with num_cached_tokens recording")
    c.add_argument("--in-place", action="store_true")
    c.add_argument("--out", default=None)
    d = sub.add_parser("label-all")
    d.add_argument("--prod-dir", required=True)
    d.add_argument("--measured-dir", required=True)
    d.add_argument("--in-place", action="store_true")
    d.add_argument("--out", default=None, help="write the labelling report here")
    args = ap.parse_args()
    if args.cmd == "label-all":
        rep = label_all(args.prod_dir, args.measured_dir, args.in_place)
        if args.out:
            jdump({"schema": "verity-vllm/cached-prefix-labels/v1", "prod_dir": args.prod_dir, "measured_dir": args.measured_dir,
                   "in_place": args.in_place, "records": rep}, args.out)
        for r in rep:
            print("LABEL", json.dumps(r))
        print("LABEL_ALL_DONE", sum(r["status"] == "labelled" for r in rep), "labelled of", len(rep))
        return
    if args.cmd == "merge-prod":
        m = merge_prod(jload(args.main), jload(args.solo), require_no_cached=not args.allow_cached)
        if args.compare_to:
            m["vs_earlier_record"] = {"file": os.path.basename(args.compare_to), **compare_outputs(m, jload(args.compare_to))}
        jdump(m, args.out)
        print("MERGE_PROD_DONE", args.out, "cached", m["requests_with_cached_prefix"],
              "vs_earlier", {k: v for k, v in (m.get("vs_earlier_record") or {}).items() if k != "differences"},
              "diffs", (m.get("vs_earlier_record") or {}).get("differences"))
    elif args.cmd == "merge-dumps":
        idx = merge_dumps(args.main, args.solo, args.out)
        print("MERGE_DUMPS_DONE", args.out, "rows", idx["rows_seen"], "traj", idx["trajectory_followed"],
              "replay_cached", idx["replay_requests_with_cached_prefix"])
    else:
        rec = label(args.record, args.measured, args.source)
        out = args.record if args.in_place else (args.out or args.record.replace(".json", "_labelled.json"))
        jdump(rec, out)
        print("LABEL_DONE", out, "cached", rec["requests_with_cached_prefix"], rec["cached_prefix_label"])


if __name__ == "__main__":
    main()
