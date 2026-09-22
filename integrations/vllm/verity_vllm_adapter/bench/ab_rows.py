"""Rows + comparisons for a production-only compiled-vs-eager A/B (CONTRACT.md §15.5 step 1, §9). CPU only.

For ONE case x profile production record (the adapter's recorder output directory), writes into the
experiment directory:

* ``table1.jsonl``: one Table 1 row with every check count 0 and verdict ``NOT_RUN`` (production-only
  diagnostic: no replay, no commitments, no verify; the blocker says so). ``workload`` and
  ``workload_exercised`` come from the record (§15.8).
* ``table2.jsonl``: kernel-path rows, one per op family, ``runtime_impl`` = the CUDA kernel(s) that
  actually ran under this profile (from a torch.profiler ``kernels.json`` of the same model, flags
  and execution mode on this device), ``checker = none``, ``status = kernel_path_resolved``. These
  are identity fixtures (§16.8), not relation checks.
* ``token_comparison.json``: bit comparisons (no tolerance) of this record against every ``--other
  LABEL=production.jsonl``: per request identical tokens / first divergence; per position the chosen
  token and its chosen-token logprob (fp32 as serialised) equal; logprob equality restricted to the
  common prefix (same token, same context) is reported separately from the all-positions count.

    python -m verity_vllm_adapter.bench.ab_rows --case B1 --production <dir>/production --kernels <kernels.json> \
        --experiment-id B1-<pid>-<utc> --adapter-revision <sha> --out results/B1-<pid>-<utc> \
        --other h100-eager=<dir>/production.jsonl --other a100-compiled=results/B1-7fad.../production/production.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from verity_vllm import rows as R

OP_FAMILIES: list[tuple[str, str, list[str]]] = [
    # (op_id, description, kernel families from throughput.KERNEL_FAMILIES)
    ("layers.*.{self_attn.qkv_proj,self_attn.o_proj,mlp.gate_up_proj,mlp.down_proj}.gemm + lm_head.gemm", "GEMM", ["gemm.triton_batch_invariant", "gemm.cublas"]),
    ("layers.*.self_attn.attn.attn", "paged attention (prefill + decode)", ["attention.flash_attn", "attention.flashinfer", "attention.triton"]),
    ("layers.*.{input_layernorm,post_attention_layernorm}.rmsnorm + norm.rmsnorm", "RMSNorm (+ fused residual add)", ["rmsnorm.inductor_generated", "rmsnorm.cuda_custom_op", "rmsnorm.triton_batch_invariant"]),
    ("layers.*.mlp.act_fn.act_mul", "SiLU * up", ["activation.inductor_generated", "activation.cuda_custom_op"]),
    ("layers.*.self_attn.rotary_emb.rope", "rotary embedding", ["rope.cuda_custom_op", "inductor.generated"]),
    ("layers.*.self_attn.attn.kv_write", "KV-cache write", ["kv_cache.write"]),
    ("sampler.select", "greedy token selection", ["sampler", "reduce.aten"]),
]


def load_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def compare(a: list[dict[str, Any]], b: list[dict[str, Any]], la: str, lb: str) -> dict[str, Any]:
    bi = {r["request_id"]: r for r in b}
    per = []
    pos_cmp = pos_tok_eq = pos_lp_eq = prefix_pos = prefix_lp_eq = 0
    for ra in a:
        rb = bi.get(ra["request_id"])
        if rb is None:
            continue
        if list(ra["prompt_token_ids"]) != list(rb["prompt_token_ids"]):
            raise SystemExit(f"{ra['request_id']}: prompt ids differ between {la} and {lb}: not the same workload")
        ta, tb = list(ra["output_token_ids"]), list(rb["output_token_ids"])
        pa, pb = list(ra.get("chosen_logprobs") or []), list(rb.get("chosen_logprobs") or [])
        n = ra["prompt_len"]
        first = None
        for i, (x, y) in enumerate(itertools.zip_longest(ta, tb)):
            pos_cmp += 1
            same_tok = x == y
            pos_tok_eq += same_tok
            lp_eq = i < len(pa) and i < len(pb) and pa[i] == pb[i]
            pos_lp_eq += lp_eq
            if first is None:
                if same_tok:
                    prefix_pos += 1
                    prefix_lp_eq += lp_eq
                else:
                    first = i
        per.append({
            "request_id": ra["request_id"], "prompt_len": n, "max_tokens": len(ta), "identical": first is None,
            "first_differing_output_index": first, "first_differing_position": (n - 1 + first) if first is not None else None,
            "tokens_equal_positions": sum(x == y for x, y in zip(ta, tb)),
            "logprobs_equal_positions": sum(1 for i in range(min(len(pa), len(pb))) if pa[i] == pb[i]),
            la: {"first_divergent_token": ta[first] if first is not None and first < len(ta) else None,
                 "logprob": pa[first] if first is not None and first < len(pa) else None},
            lb: {"first_divergent_token": tb[first] if first is not None and first < len(tb) else None,
                 "logprob": pb[first] if first is not None and first < len(pb) else None},
        })
    return {
        "requests_compared": len(per),
        "requests_identical": sum(p["identical"] for p in per),
        "positions_compared": pos_cmp,
        "positions_token_equal": pos_tok_eq,
        "positions_logprob_bit_equal_all": pos_lp_eq,
        "positions_on_common_prefix": prefix_pos,
        "positions_logprob_bit_equal_on_common_prefix": prefix_lp_eq,
        "note": "bit comparisons only (tokens as ints, chosen-token logprobs as the fp32 values serialised by the recorder); "
                "positions after a request's first divergence follow different trajectories and are counted in *_all only",
        "per_request": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--production", required=True, help="the recorder's output dir (manifest.json, production.jsonl, ...)")
    ap.add_argument("--production-r2", default=None, help="a second fresh-process record of the same profile (same-device reproducibility)")
    ap.add_argument("--kernels", default=None, help="kernels.json (torch.profiler families) for this profile on this device")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--worker", default="W4d")
    ap.add_argument("--profile-label", required=True, help="e.g. h100-compiled")
    ap.add_argument("--other", action="append", default=[], help="LABEL=path/to/production.jsonl to compare against")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prod = Path(args.production)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((prod / "manifest.json").read_text())
    profile_id = json.loads((prod / "production.jsonl").read_text().splitlines()[0])["profile_id"]
    recs = load_jsonl(prod / "production.jsonl")
    wex = json.loads((prod / "workload_exercised.json").read_text())
    common = R.Common(args.experiment_id, args.case, profile_id, args.worker, args.adapter_revision)
    comp = manifest["compilation"]
    flags = [f"{k}={v}" for k, v in manifest["engine_flags"].items()] + (["enforce_eager=True"] if comp.get("enforce_eager") else [])

    # Table 1: NOT_RUN diagnostic row
    lens = sorted({r["prompt_len"] for r in recs})
    row1 = R.table1_row(
        common,
        workload={"requests": len(recs), "generated_positions": sum(len(r["output_token_ids"]) for r in recs),
                  "input_len_range": [lens[0], lens[-1]], "peak_concurrency_achieved": int(wex["peak_concurrency_achieved"])},
        production_changes="flags:[" + ",".join(flags) + "]",
        replay_restrictions=[],
        token_selection={"exact": 0, "attempted": 0},
        boundary_noninterference={"identical": False, "compared": 0, "mismatched_scalars": 0, "reference_method": "none (production-only)"},
        required_coverage={"bound": 0, "required": 0},
        authentication={"valid": 0, "attempted": 0},
        relation_checks={"same_runtime": {"exact": 0, "attempted": 0}, "independent": {"exact": 0, "attempted": 0}},
        structural={"result": "not_run", "failing_invariant": None},
        negative={"rejected": 0, "attempted": 0},
        blocker={"kind": "NOT_RUN", "reason": f"production-only compiled-vs-eager diagnostic on {manifest['gpu']['name']} ({args.profile_label}, §15.5 step 1): "
                                               "no replay, no commitments, no verify; token/logprob comparisons in token_comparison.json and README.md"},
        workload_exercised={k: wex[k] for k in ("recorded_pass", "enable_prefix_caching", "requests_with_cached_tokens", "in_workload_prefix_cache_hits",
                                                "cached_tokens_from_outside_the_recorded_pass", "prefix_cache_hits_at_admission", "peak_concurrency_achieved",
                                                "engine_steps", "preemptions", "effective_execution") if k in wex},
    )
    R.write_jsonl(out / "table1.jsonl", [row1])

    # Table 2: kernel-path rows
    rows2: list[dict[str, Any]] = []
    if args.kernels:
        kern = json.loads(Path(args.kernels).read_text())
        fams = kern.get("families", {})
        for op_id, desc, fam_keys in OP_FAMILIES:
            present = [(f, fams[f]) for f in fam_keys if f in fams]
            impl = "; ".join(f"{f}: " + " | ".join(k[:110] for k in v["kernels"][:3]) + f" (n={v['count']})" for f, v in present) or "unresolved"
            rows2.append(R.table2_row(
                common,
                op_id=op_id, phase="prefill+decode", shape="any (16 req x 128 in x 8 out profiled pass)", dtype="bfloat16",
                layout_or_quant=None, runtime_impl=impl, primitive_semantics="none (kernel identity fixture, §16.8)",
                decomposition="whole_operation",
                interface={"input_bytes": 0, "output_bits": 0, "range": None}, origin="captured",
                binding={"present": 0, "unresolved": 0}, checker="none",
                natural_checks={"exact": 0, "attempted": 0}, stress_checks={"exact": 0, "attempted": 0, "domain": None},
                mutations={"rejected": 0, "attempted": 0},
                checker_work={"seconds": 0.0, "flops_or_bytes": float(sum(v["device_time_us"] for _, v in present)), "includes_full_op": False},
                status="kernel_path_resolved" if present else "unresolved",
                limitation=f"{desc}: identity of the kernels that ran under profile {profile_id} ({args.profile_label}: mode={comp['mode']}, "
                           f"cudagraph={comp['cudagraph_mode']}, custom_ops={comp.get('custom_ops')}); flops_or_bytes = profiled device µs; no relation executed",
            ))
        R.write_jsonl(out / "table2.jsonl", rows2)

    # comparisons
    cmp: dict[str, Any] = {"this": {"label": args.profile_label, "profile_id": profile_id, "gpu": manifest["gpu"]["name"],
                                    "compilation": comp, "engine_flags": manifest["engine_flags"]}, "pairs": {}}
    if args.production_r2:
        r2 = load_jsonl(Path(args.production_r2) / "production.jsonl")
        cmp["pairs"][f"{args.profile_label}_vs_{args.profile_label}-r2"] = {"other_profile_id": r2[0]["profile_id"], **compare(recs, r2, args.profile_label, args.profile_label + "-r2")}
    for spec in args.other:
        label, _, path = spec.partition("=")
        other = load_jsonl(Path(path))
        cmp["pairs"][f"{args.profile_label}_vs_{label}"] = {"other_profile_id": other[0].get("profile_id"), "other_path": path, **compare(recs, other, args.profile_label, label)}
    (out / "token_comparison.json").write_text(json.dumps(cmp, indent=1) + "\n")
    for k, v in cmp["pairs"].items():
        print(f"{k}: identical {v['requests_identical']}/{v['requests_compared']}, tokens {v['positions_token_equal']}/{v['positions_compared']}, "
              f"logprobs bit-equal {v['positions_logprob_bit_equal_all']}/{v['positions_compared']} (common prefix {v['positions_logprob_bit_equal_on_common_prefix']}/{v['positions_on_common_prefix']})")
    print(f"wrote table1 (NOT_RUN), {len(rows2)} table2 rows, token_comparison.json -> {out}")


if __name__ == "__main__":
    main()
