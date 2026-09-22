"""CB-B builder driver: structural inputs (prompt length, max_tokens, greedy) + pinned vLLM model class + config
→ meta structural execution of every engine step → eager-fold lowering → Verity Program (flat, one request).

    PYTHONPATH=. /workspace/venv-cu129/bin/python -m verity_capture.experimental.cb_b.build_request \
        --model <snapshot dir> --prompt-len 155 --max-tokens 15 --out /workspace/cp/cbb/req155

Writes: program.json (encode_program), body.jsonl (one line per flat body node: spec id, argument runs, out type,
name hint), summary.json (fold report, memory, per-step counts, rules used, timings), opens.txt (audit hook).

Forbidden inputs are never read (audit-hooked): target program.json / instances.jsonl / capture log / aot/build.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

from .probe import OpenAudit, git_rev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--sampler", default="vllm", choices=["vllm", "argmax"])
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-program", action="store_true", help="fold only (no Program/digest); fast")
    ap.add_argument("--pad", action="store_true", help="also emit the Serve@3 padded rectangle (guards + row muxes) and compare")
    ap.add_argument("--pad-read-only", action="store_true", help="mux only rows actually read by later steps (skip the last step's dead rows)")
    ap.add_argument("--profile", default="vllm_d9105ea80_sm89_eager",
                    help="fold profile module under verity_capture.profiles (the pinned pattern→Definition bindings of the eager pipeline)")
    ap.add_argument("--model-repo", default="HuggingFaceTB/SmolLM2-135M", help="HF repo id recorded in the synthetic log header")
    ap.add_argument("--max-model-len", type=int, default=8192, help="engine max_model_len (serving parameter; must not exceed the model's)")
    ap.add_argument("--step-inputs", default="runner", choices=["runner", "runner-attn", "hand"],
                    help="per-step inputs: 'runner' = a real GPUModelRunner (no load_model) driven by SchedulerOutputs: prepare_inputs/prepare_attn/"
                         "build_attn_metadata (default); 'runner-attn' = only the attention side via attn_utils/BlockTables; 'hand' = hand-built")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    audit = OpenAudit(own_out=a.out)
    sys.addaudithook(audit)
    audit.active = True

    from . import fake_exec as fx
    from . import lower

    fx.apply_env()
    import torch

    import importlib

    prof = importlib.import_module(f"verity_capture.profiles.{a.profile}")

    summary: dict = {"rev": git_rev(), "argv": sys.argv, "torch": torch.__version__, "inputs": {"prompt_len": a.prompt_len,
                     "max_tokens": a.max_tokens, "sampling": "greedy", "model": a.model, "profile_id": prof.PROFILE.id}}
    t0 = time.perf_counter()
    cfg = fx.build_vllm_config(a.model, max_model_len=a.max_model_len)
    fx.init_distributed(cfg)
    model = fx.instantiate_model(cfg)
    summary["instantiate_s"] = time.perf_counter() - t0
    hf = cfg.model_config.hf_config
    summary["hf_config"] = {k: getattr(hf, k, None) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                            "num_key_value_heads", "intermediate_size", "vocab_size", "rms_norm_eps", "rope_theta", "tie_word_embeddings")}

    tr = fx.Trace()
    t1 = time.perf_counter()
    try:
        with fx.CudaGuard() as g:
            specs, outs = fx.run_request(tr, model, cfg, prompt_len=a.prompt_len, max_tokens=a.max_tokens, sampler=a.sampler,
                                         step_inputs=a.step_inputs)
        summary["exec"] = {"ok": True, "steps": len(specs), "events": len(tr.events), "cuda_kernel_events": getattr(g, "kernels", -1),
                           "cuda_kernel_events_by_name": getattr(g, "by_name", {}), "cuda_prep_index_kernels": getattr(g, "prep_kernels", 0),
                           "cuda_model_kernels": getattr(g, "model_kernels", -1), "step_inputs": a.step_inputs, "notes": tr.notes[-3:],
                           "wall_s": time.perf_counter() - t1, "rules_used": tr.rules_used, "storages": len(tr.storages.order)}
    except Exception as e:  # noqa: BLE001
        summary["exec"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:600]}", "traceback": traceback.format_exc()[-3000:],
                           "events": len(tr.events), "step": tr.step, "module": tr.module()}
        _write(a.out, summary, audit)
        print(json.dumps(summary["exec"], indent=1)[:3000])
        return 1

    t2 = time.perf_counter()
    try:
        from verity_capture.fold import summarize

        result, events = lower.fold_trace(tr, profile=prof.PROFILE, config=prof.C, prompt_len=a.prompt_len, max_tokens=a.max_tokens,
                                          model_repo=a.model_repo)
        fs = summarize(result)
        summary["fold"] = {"ok": True, "wall_s": time.perf_counter() - t2, "events": len(events), "digest_captured": result.digest,
                           "nodes": fs["nodes"], "instances": fs["instances"], "report": fs["report"], "memory": fs["memory"],
                           "steps": fs["steps"], "errors": fs["errors"], "error_classes": fs["error_classes"],
                           "error_examples": fs["error_examples"], "notes": fs["notes"], "timings": fs["timings"],
                           "params": fs["params"], "ret_type": fs["ret_type"]}
        # body: the flat instance sequence (what the comparator consumes)
        from verity.ir.refs import runs

        with open(os.path.join(a.out, "body.jsonl"), "w") as f:
            for k, n in enumerate(result.body.nodes):
                f.write(json.dumps({"k": k, "spec": n.fn.id, "args": [runs(c.refs) for c in n.args], "out": n.out.to_json(),
                                    "name": n.name}) + "\n")
        summary["fold"]["ret_runs"] = runs(result.body.ret)
        if not a.no_program and result.program is not None:
            from verity.ir.codec import encode_program

            desc = encode_program(result.program)
            with open(os.path.join(a.out, "program.json"), "w") as f:
                json.dump(desc, f)
            summary["fold"]["program_bytes"] = os.path.getsize(os.path.join(a.out, "program.json"))
    except Exception as e:  # noqa: BLE001
        summary["fold"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:800]}", "traceback": traceback.format_exc()[-4000:]}

    if a.pad and summary.get("fold", {}).get("ok"):
        t3 = time.perf_counter()
        try:
            from verity_capture.fold import workload_digest
            from verity.ir.codec import encode_program, program_digest
            from verity.ir.refs import runs
            from verity_vllm.registry import b1
            from verity_vllm.registry.serve3 import serve_v3

            from . import pad

            eos = int(getattr(hf, "eos_token_id", 0) or 0)      # model config (allowed input); B0: 0
            counts = [int(st["nodes"]) for st in fs["steps"]]
            padded, info = pad.pad_body(result.body, counts, eos, mux_last_step=not a.pad_read_only)
            peel = pad.peel_check(result.body, padded, info)
            D_MAX = len(counts) - 1
            prog, spec = pad.padded_program(padded, profile_id=prof.PROFILE.id, workload_digest=workload_digest(events[0].workload),
                                            eos=eos, D_MAX=D_MAX)
            pdesc = encode_program(prog)
            with open(os.path.join(a.out, "padded_program.json"), "w") as f:
                json.dump(pdesc, f)
            with open(os.path.join(a.out, "padded_body.jsonl"), "w") as f:
                for k, n in enumerate(padded.nodes):
                    f.write(json.dumps({"k": k, "spec": n.fn.id, "form": n.form, "n": n.n, "args": [runs(c.refs) for c in n.args],
                                        "out": n.out.to_json(), "name": n.name, "inserted": info["inserted_at"].get(k)}) + "\n")
            # the hand-written Serve_v3 at the same (LP, D_MAX): expanded gate count + top-level structure, comparison only
            C3 = {**prof.C, "EOS_ID": eos}
            sv3 = serve_v3(C3, a.prompt_len, D_MAX, gemm=b1.GEMM_AMPERE, attention=b1.attention_fa2(prof.C["D"]), norm=b1.norm_bundle(prof.C))
            sv3_gates = sv3.gates
            summary["pad"] = {"ok": True, "eos": eos, "wall_s": time.perf_counter() - t3, "stats": info["stats"], "peel": peel,
                              "padded_digest": program_digest(pdesc), "padded_ret": padded.ret_type.to_json(),
                              "padded_ret_runs": runs(padded.ret), "spec_id": spec.id,
                              "serve_v3": {"id": sv3.id, "gates": sv3_gates, "ret": sv3.body.ret_type.to_json(),
                                           "top_nodes": [(n.fn.id, n.form) for n in sv3.body.nodes][:8],
                                           "n_top_nodes": len(sv3.body.nodes)},
                              "gates_equal_serve_v3": sv3_gates == padded.gates}
            memo: dict = {}
            hp, hs = pad.gate_histogram(spec, memo), pad.gate_histogram(sv3, memo)
            summary["pad"]["gate_histogram_diff_vs_serve_v3"] = pad.histogram_diff(hp, hs)
            hf_flat = pad.gate_histogram(result.spec, memo)
            summary["pad"]["gate_histogram_flat_vs_serve_v3"] = pad.histogram_diff(hf_flat, hs)
            memo2: dict = {}
            np_, ns_ = pad.node_histogram(spec, memo2), pad.node_histogram(sv3, memo2)
            summary["pad"]["node_histogram_diff_vs_serve_v3"] = pad.histogram_diff(np_, ns_)
        except Exception as e:  # noqa: BLE001
            summary["pad"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:800]}", "traceback": traceback.format_exc()[-4000:]}
    summary["wall_total_s"] = time.perf_counter() - t0
    _write(a.out, summary, audit)
    print(json.dumps({k: v for k, v in summary.items() if k != "fold"}, indent=1, default=str)[:3000])
    fo = summary.get("fold", {})
    print("fold:", json.dumps({k: v for k, v in fo.items() if k not in ("steps", "report", "notes", "error_examples", "traceback")}, indent=1, default=str)[:3000])
    if fo.get("report"):
        print("report:", json.dumps(fo["report"], indent=1, default=str)[:4000])
    if fo.get("error_examples"):
        print("errors:", json.dumps(fo["error_examples"][:6], indent=1, default=str)[:4000])
    if fo.get("traceback"):
        print(fo["traceback"][-3000:])
    if "pad" in summary:
        print("pad:", json.dumps(summary["pad"], indent=1, default=str)[:6000])
    return 0 if fo.get("ok") else 1


def _write(out: str, summary: dict, audit: OpenAudit) -> None:
    import resource

    audit.active = False
    summary["opens_n"] = len(audit.opens)
    summary["opens_forbidden"] = audit.forbidden
    summary["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 3)   # Linux: KiB
    with open(os.path.join(out, "opens.txt"), "w") as f:
        f.write("\n".join(audit.opens))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    with open(os.path.join(out, "construction_manifest.json"), "w") as f:
        json.dump(_manifest(summary, audit), f, indent=1, default=str)


def _manifest(summary: dict, audit: OpenAudit) -> dict:
    """What the construction consumed and what it hand-authored — for an independent validator."""
    from . import fake_exec as fx

    model_dir = summary["inputs"]["model"]
    reads = sorted({l.split("\t")[0] for l in audit.opens if l.startswith(model_dir)})
    rules_used = summary.get("exec", {}).get("rules_used", {})
    fo, pa = summary.get("fold", {}), summary.get("pad", {})
    return {
        "schema": "cb-b/construction-manifest/v1",
        "revision": summary["rev"], "argv": summary["argv"], "torch": summary["torch"],
        "mechanism": "pinned vLLM LlamaForCausalLM instantiated by vLLM's initialize_model on the meta device; every engine step of the "
                     "request executed (forward → hidden[logits_indices] → compute_logits → gumbel_sample) under a TorchDispatchMode; "
                     "dispatcher calls / re-entered custom ops / Triton launches recorded (never executed); trace re-expressed as an "
                     "eager capture log over meta storage identities; verity_capture.fold (unmodified) resolves patterns and wires "
                     "dataflow; optional Serve@3 padding rewrite (pad.py)",
        "structural_inputs": {k: summary["inputs"][k] for k in ("prompt_len", "max_tokens", "sampling")},
        "model_config_files_read": reads,
        "hf_config": summary.get("hf_config"),
        "profile": summary["inputs"]["profile_id"],
        "registered_definitions_selected": (fo.get("report") or {}).get("by_spec"),
        "hand_authored": {
            "meta_rules_op_semantics": {op: getattr(r, "__name__", str(r)) for op, r in fx.META_RULES.items()},
            "reentered_custom_ops": sorted(fx.REENTER),
            "bi_cuda_overrides_reentered": "every `aten` override vLLM registers in batch-invariant mode (recorded from torch.library.Library.impl, "
                                           "nothing named) runs on meta as it would on the CUDA key; only its Triton launches are observed "
                                           f"(fired this build: {sorted(k for k in rules_used if k.startswith('bi:'))})",
            "triton_launch_hook": "JITFunction.run wrapper: binder → bound args/constexprs; backend.parse_options → num_warps/num_stages; no launch",
            "scaffolding": ([
                "EngineArgs → VllmConfig (enforce_eager, bf16, TP1, block_size 16, max_model_len) + set_current_vllm_config",
                "gloo TP1 distributed init (as the worker)",
                "set_default_torch_dtype(bf16) around initialize_model (as the worker; selects FA2)",
                "init_batch_invariance() (VLLM_BATCH_INVARIANT=1, as the worker)",
                "WorkerBase.__init__ parity: kernel_config.ir_op_priority.set_default(); vllm.ir.set_default_torch_wrap(compilation_config.ir_enable_torch_wrap)",
                "KV cache config: vLLM's get_kv_cache_spec → resolve_kv_cache_layout → get_kv_cache_configs (engine core)",
            ] + ({
                "runner": [
                    "per step: vLLM's GPUModelRunner (constructed on the CUDA device WITHOUT load_model) driven by a SchedulerOutput: add_requests/"
                    "update_requests → gather_batch_req_state → prepare_inputs (positions, query_start_loc, seq_lens, logits_indices) → prepare_attn "
                    "(block tables, slot mappings) → postprocess_num_computed_tokens; attn_utils.init_attn_backend + init_kv_cache (allocate + bind) + "
                    "build_attn_metadata (each backend's own metadata builder) — vLLM code executed, not hand-built",
                    "HAND: the scheduler's decisions as SchedulerOutput fields (prompt_len rows, one decode row per step, num_computed_tokens, block ids "
                    "g*request_blocks+i per KV-cache group = KV-manager stand-in), placeholder token values, load_model stand-ins (model_state/sampler/"
                    "prompt_logprobs_worker no-op bookkeeping stubs, decode_query_len=1), BatchExecutionDescriptor(cg_mode=NONE) (eager dispatch result), "
                    "copying the runner's int outputs to meta twins with known values",
                ],
                "runner-attn": [
                    "per step: vLLM's BlockTables + attn_utils.build_attn_metadata / build_slot_mappings_by_layer / init_kv_cache; HAND: positions, "
                    "query_start_loc, seq_lens, logits_indices, block ids",
                ],
                "hand": [
                    "per step: FlashAttentionMetadata + ForwardContext from (pos0, M, block_size); positions/slot_mapping/seq_lens/block_table/logits_indices "
                    "as known-value side tables; allocate_kv_cache + kv_cache binding by hand",
                ],
            }[summary.get("exec", {}).get("step_inputs", "runner")]) + [
                "execute_model body (not execute_model itself: needs cudagraph_manager/kv_connector/model_state from initialize_kv_cache/load_model): "
                "set_forward_context → model.forward → hidden[logits_indices] → compute_logits → gumbel_sample with runner-owned side tensors as known values",
            ]),
            "step_inputs": summary.get("exec", {}).get("step_inputs", "runner"),
            "model_family_specific_rules": [],
            "padding": "pad.py: PadGuard_v1{EOS} at every sampling event; batch(SelectBf16_v1, live_{s-1}, row, Const16[0]) on every "
                       "cross-step-read collection of every decode step (state found as cross-step reference edges); "
                       "last-step rows muxed by the learned state signature (Serve@3 §3 structural append)",
        },
        "rules_fired": rules_used,
        "outputs": {
            "program_digest_captured": fo.get("digest_captured"), "nodes": fo.get("nodes"), "ret_type": fo.get("ret_type"),
            "fold_errors": fo.get("errors"), "unsupported_roots": (fo.get("report") or {}).get("unsupported"),
            "padded_digest": pa.get("padded_digest"), "padded_nodes": (pa.get("stats") or {}).get("nodes_padded"),
            "padded_peel_ok": (pa.get("peel") or {}).get("ok"),
        },
        "cost": {"instantiate_s": summary.get("instantiate_s"), "exec_s": summary.get("exec", {}).get("wall_s"),
                 "fold_s": fo.get("wall_s"), "pad_s": pa.get("wall_s"), "wall_total_s": summary.get("wall_total_s"),
                 "max_rss_gb": summary.get("max_rss_gb"), "cuda_kernel_events": summary.get("exec", {}).get("cuda_kernel_events")},
        "forbidden_inputs": {"patterns": ["program.json", "instances.jsonl", "log.jsonl", "experimental/aot/", "/census/", "archive-20260917"],
                             "opens_matching": audit.forbidden, "opens_total": len(audit.opens)},
    }


if __name__ == "__main__":
    sys.exit(main())
