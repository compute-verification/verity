# Step 6: what GLM-5.3's TP8 Commit on this tree established, per op family of the capture map. It is not core's IntegrityProfile,
# which is stated over a partition's proof units and the verifier's draw: with no Build there is no partition and nothing is drawn,
# so no unit is checked, nothing is bounded and there is no delta. Each family says whether it is committed, whether it has a
# Program, and what the H200 established about that Program (the runs of internal/glm53/h200-go-no-go.md, and the TP8 all-reduce
# and bf16 GEMM captures under $SWEEP_DIR, named by $AR_RUN and $BF16_RUN). Reads the Commit's records from $SWEEP_DIR/commit-probe.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
stamp profile | tee "$O/profile.log"
"$PY" - "$CMAP" "$SWEEP_DIR/commit-probe" "$SWEEP_DIR/ar-capture" "$SWEEP_DIR/bf16-gemm-capture" "$O" "${COMMIT_RUN:-}" "${AR_RUN:-}" \
  "${BF16_RUN:-}" <<'EOF' 2>&1 | tee -a "$O/profile.log"
import json, os, sys
from collections import defaultdict
from pathlib import Path

cmap_path, commit_dir, ar_dir, bf16_dir, out, commit_run, ar_run, bf16_run = sys.argv[1:9]
cmap = json.loads(Path(cmap_path).read_text())

H200 = {
    "fp8-block-linear": {"verdict": "GO", "runs": ["r20261012-220408-4943"], "art": "art:59b5ae25fa7c9b47e3db6bc5efc78926a411512731b61b2349827cff8c43c426",
                         "what": "GLM53-BLOCK-FP8 GO: 63 dense cases (9 linears, M 1-256), 0 mismatches, CUTLASS block FP8 (VLLM_USE_DEEP_GEMM=0)"},
    "fp8-moe-triton": {"verdict": "GO", "runs": ["r20261012-220408-4943"], "art": "art:59b5ae25fa7c9b47e3db6bc5efc78926a411512731b61b2349827cff8c43c426",
                       "what": "GLM53-BLOCK-FP8 GO: 6 MoE cases on the widened tuned configs this Commit serves (16x64x128 / 16x128x128 swap-AB wgmma)"},
    "sigmoid-router": {"verdict": "GO", "runs": ["r20261012-220523-6aa4", "r20261012-233150-e5f2"],
                       "what": "SIGMOID-ROUTER-CAPTURE OK mufu=True; SIGMOID-ROUTER-PROGRAM-CHECK OK rows=10240"},
    "dsa-indexer-q": {"verdict": "GO", "runs": ["r20261012-231007-6a23", "r20261013-001802-7e95"], "what": "q/served 12,945,408 words, 0 mismatches"},
    "dsa-indexer-k": {"verdict": "GO", "runs": ["r20261012-231007-6a23", "r20261013-001802-7e95"], "what": "k/served 403,512 words and k/probe, 0 mismatches"},
    "dsa-mqa-logits": {"verdict": "GO", "runs": ["r20261012-231007-6a23", "r20261013-001802-7e95"],
                       "what": "logits prefill 76,766 and decode 7,292 words, 0 mismatches, on DsaMqaLogit_v3 (v1 was a Program bug)"},
    "dsa-topk": {"verdict": "GO", "runs": ["r20261013-001802-7e95"], "what": "top-k short and long rows valid"},
    "mla-sparse-attention": {"verdict": "NO-GO at vLLM's default FA3 split heuristic (expected; pending Daniel's num_splits decision)",
                             "runs": ["r20261012-231007-6a23", "r20261013-001802-7e95"],
                             "what": "the served heuristic splits KV and combines: 3,264 and 3,240 of 57,344 words differ from SparseMlaHead_v1, which is "
                                     "FA3 at num_splits=1 (GLM53-DSA GO at one split, r20261013-001802-7e95). This tree serves the heuristic (the "
                                     "switch is not in it), so the unit would mismatch its Program: a pending serving decision, not a failure of the Commit"},
    "fused-rope-triton": {"verdict": "GO", "runs": ["r20261012-222325-ce92"], "what": "k_pe and q_pe settle on RoPEHeadGptJ_v2 (v1 differs on 48 and 384 words)"},
    "router-gemm": {"verdict": "available, no Program in this tree", "runs": ["r20261012-221745-18e0"],
                    "what": "cuteDSL ll_bf16_gemm AVAILABLE and active on decode (step 3); its Programs are on cursor/glm53-committed-ops-8c79, "
                            "not merged, and their H200 capture has not run"},
    "cublas-bf16-gemm": {"verdict": "no Program in this tree", "runs": [],
                         "what": "Programs on cursor/glm53-committed-ops-8c79 under a split-K-off cuBLAS switch this Commit does not serve"},
}
# The map still says "tonight" for these, but their Programs are registered in this tree.
IN_TREE = {"sigmoid-router": ["MoeRouterSigmoidBias_v1"],
           "fp8-moe-triton": ["MoeExpertFp8BlockGemm_v1", "MoeExpertFp8BlockGemmW_v1", "MoeExpertFp8BlockCoordinate_v1",
                              "MoeExpertFp8BlockCoordinateW_v1"]}

def capture(rec: Path, run: str, phases: tuple[str, ...] | None = None) -> dict | None:
    """A capture's record, or its cases of `phases` only ("<phase>/..." keys): GO when every one of them is."""
    if not rec.is_file():
        return None
    doc = json.loads(rec.read_text())
    cases = {k: c for k, c in (doc.get("cases") or {}).items() if phases is None or k.split("/")[0] in phases}
    ok = bool(doc.get("ok")) if phases is None else bool(cases) and all(c.get("ok") for c in cases.values())
    changed = (doc.get("served_default") or {}).get("changed")
    return {"run": run or None, "verdict": "GO" if ok else "NO-GO", "digest": doc.get("digest"), "record": str(rec), "cases": len(cases),
            "no_go_cases": sorted(k for k, c in cases.items() if not c.get("ok")), "errors": sorted(doc.get("errors") or []),
            # The cases whose result differs when the capture's switch is off, as this Commit's engine served them.
            "served_default_changed": sorted(k for k in changed if k in cases) if isinstance(changed, list) else None}


ar_rec = Path(ar_dir) / "glm53_tp8_allreduce_capture.json"
AR = {"tp-allreduce": (capture(ar_rec, ar_run, ("allreduce",)), "its all-reduce phase: GO where vLLM dispatched the custom all-reduce "
                       "(ca_comm), which this Commit's engine disables; NO-GO where it dispatched FlashInfer's (fi_ar_comm), as this "
                       "Commit's engine serves, consistent with its AllReduce_v2 cross-rank mismatches"),
      "fused-allreduce-rmsnorm": (capture(ar_rec, ar_run, ("fused",)), "its fused phase, FlashInfer's all-reduce + RMSNorm as the served "
                                  "decoder layers run it")}
bf16_rec = Path(bf16_dir) / "glm53_bf16_gemm_capture.json"
BF16 = {"router-gemm": (capture(bf16_rec, bf16_run, ("router",)), "GateLinear at M <= 16 on cuteDSL ll_bf16_gemm as served, past 16 "
                        "cuBLAS under the split-K-off switch"),
        "cublas-bf16-gemm": (capture(bf16_rec, bf16_run, ("linear", "bmm")), "lm_head shard, wk_weights_proj and the absorbs under the "
                             "split-K-off cuBLAS switch, which this Commit's engine does not set")}

taps = defaultdict(list)
for t in cmap["taps"]:
    taps[t.get("program")].append({"id": t["id"], "layers": t.get("layer") or t.get("layers"), "binder": (t.get("commit") or {}).get("binder")})

summary = json.loads((Path(commit_dir) / "summary.json").read_text()) if (Path(commit_dir) / "summary.json").is_file() else None
cov = (summary or {}).get("per_rank_coverage") or [{}]
commit = None if summary is None else {
    "run": commit_run or None, "dir": commit_dir, "of_record": False,
    "why_not_of_record": "no Build: tp-commit ran without --required-manifest, so no sampled replay, no weights pin, no Program digests",
    "tp_run_roots": summary.get("tp_run_roots"), "per_rank_roots": summary.get("per_rank_roots"),
    "per_rank_coverage": cov, "bytes_bound_all_ranks": sum(int(v.get("bytes_bound") or 0) for v in (cov[0] or {}).values()),
    "bytes_bound_per_rank": {r: int(v.get("bytes_bound") or 0) for r, v in (cov[0] or {}).items()},
    "commit_pass": summary.get("commit_pass"), "pass": summary.get("pass"), "tokens_equal": summary.get("tokens_equal"),
    "cross_rank_collectives": summary.get("cross_rank_collectives"), "weights_pin": summary.get("weights_pin"),
    "sampled_replay": summary.get("sampled_replay"), "required_classes": summary.get("required_classes")}

fams = []
for name, prog in cmap["programs"].items():
    ts = taps.get(name) or []
    if not ts:
        continue
    defs = prog.get("definitions") or IN_TREE.get(name) or []
    has_program = (prog.get("status") in ("exists", "port") and bool(defs)) or name in IN_TREE
    h = H200.get(name)
    if name in AR:
        c, what = AR[name]
        h = (c | {"runs": [c["run"]] if c["run"] else [], "what": "glm53_tp8_allreduce_capture_gpu.py at cursor/glm53-tp8-allreduce-3d06 "
                  f"against that branch's Programs, not merged into this tree: {what}"} if c else
             {"verdict": "not captured", "runs": [], "what": "Programs on cursor/glm53-tp8-allreduce-3d06, not merged"})
    if name in BF16 and BF16[name][0]:
        c, what = BF16[name]
        h = c | {"runs": [c["run"]] if c["run"] else [], "what": f"glm53_bf16_gemm_capture_gpu.py at cursor/glm53-tp8-allreduce-3d06 "
                 f"(cursor/glm53-committed-ops-8c79 merged) against that branch's Programs, not merged into this tree: {what}"}
    fams.append({"family": name, "map_status": prog.get("status"), "definitions": defs, "has_program_in_tree": has_program,
                 "committed": [t["id"] for t in ts if t["binder"]], "unbound_by_design": [t["id"] for t in ts if not t["binder"]],
                 "checked_in_commit": False, "h200": h or {"verdict": "no H200 capture in this campaign", "runs": []}})

doc = {"schema": "verity-vllm/glm53-h8-profile/v0", "row": os.environ.get("ROW"), "capture_map": cmap["id"],
       "tree": Path.cwd().parents[1].name,
       "integrity_profile": None,
       "why": "no Build of GLM-5.3 (r20261013-002315-8dd7: the derive crashes in DSA's fused_norm_rope; no DSA Build rules exist and the "
              "derive emits no opaque gates), so there is no partition of proof units, no sampled replay and nothing drawn: no unit is "
              "checked, no bound holds and there is no delta",
       "delta": None, "commit": commit, "families": fams,
       "served": {"switches": cmap["target"].get("switches"), "env": {k: os.environ.get(k) for k in (
           "VLLM_USE_DEEP_GEMM", "VLLM_BATCH_INVARIANT", "VLLM_TUNED_CONFIG_FOLDER", "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
           "VLLM_ENABLE_V1_MULTIPROCESSING", "NCCL_P2P_DISABLE")}, "engine_args": "gpu_memory_utilization 0.75, max_model_len 288, "
           "max_num_batched_tokens 2048, disable_custom_all_reduce true", "allreduce_backends": "FLASHINFER > SYMM_MEM > PYNCCL "
           "(r20261013-001654-3e77); its tp-capture header said flashinfer_allreduce false"}}
Path(out, "profile.json").write_text(json.dumps(doc, indent=1, default=str))

print(f"[profile] integrity profile: none ({doc['why']})")
if commit:
    print(f"[profile] commit (not of record): tp_run_roots={commit['tp_run_roots']} pass={commit['pass']} commit_pass={commit['commit_pass']} "
          f"bytes_bound={commit['bytes_bound_all_ranks']} over {len(commit['bytes_bound_per_rank'])} ranks")
    print(f"[profile] cross-rank collectives: {json.dumps((commit['cross_rank_collectives'] or {}).get('by_family'))}")
groups = defaultdict(list)
for f in fams:
    key = ("program + H200 GO" if f["has_program_in_tree"] and f["h200"]["verdict"] == "GO" else
           "program, H200 NO-GO" if f["has_program_in_tree"] and str(f["h200"]["verdict"]).startswith("NO-GO") else
           "program, no H200 capture" if f["has_program_in_tree"] else
           "copy" if f["map_status"] == "copy" else "committed-only (no Program in this tree)")
    groups[key].append(f["family"])
for k, v in groups.items():
    print(f"[profile] {k}: {', '.join(v)}")
for f in fams:
    print(f"[profile] {f['family']:24s} {f['map_status']:15s} committed={len(f['committed'])} unbound={f['unbound_by_design']} "
          f"h200={f['h200']['verdict']} {f['h200'].get('runs')}"
          + (f" served_default_changed={f['h200']['served_default_changed']}" if f["h200"].get("served_default_changed") else ""))
EOF
[ -f "$O/profile.json" ]
