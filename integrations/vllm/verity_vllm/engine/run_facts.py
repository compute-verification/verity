"""What a run records about its engine and host: the profile manifest and profile id (built as the production recorder
builds them; fields that necessarily differ from the accepted record's manifest are listed in the notes), the model config subset,
and the versions and host documents.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity_vllm.engine.build import (Engine, compilation_facts, observed_flash_attn_version, resolved_async_scheduling,
                                      vllm_has_flashinfer)
from verity_vllm.engine.code_identity import generated_kernels_of_this_process


#: The accepted production record per checkpoint role (its tokens are the reference the runs are compared with).
ACCEPTED: dict[str, dict[str, str]] = {
    "B0": {"profile_id": "88e1b098d719", "record": "out/scale/e8/live/V5-N32x16-1req-bex0-inc-sc1"},
    "B1": {"profile_id": "40f0cebeb670 / c37c3f0b8363", "record": "fixtures/results/B1-40f0cebeb670-20260907T1242Z/production (and B1-c37c3f0b8363-20260907T1328Z; both RTX 4090 eager)"},
}
ACCEPTED_PROFILE_ID = ACCEPTED["B0"]["profile_id"]
ACCEPTED_RECORD = ACCEPTED["B0"]["record"]


def profile_manifest(engine: Engine, *, adapter_revision: str, wheel_sha256: str | None, vllm_commit: str | None,
                     sampling: dict) -> tuple[dict, str, dict]:
    """(manifest, profile_id, notes) via `profile.build_manifest` exactly like the production recorder; falls back
    to a sha256 of the engine kwargs + versions when build_manifest cannot run (and says so in the notes)."""
    from verity_vllm.engine import engine_profile as prof

    case = engine.case
    acc = ACCEPTED.get(case, {})
    notes: dict[str, Any] = {"case": case, "accepted_profile_id": acc.get("profile_id"), "accepted_record": acc.get("record")}
    # the manifest records the sampling actually requested (a non-greedy variant is another profile id by construction)
    sampling_desc = {"greedy": bool(sampling.get("greedy", True)), "temperature": float(sampling.get("temperature", 0.0)),
                     "top_p": float(sampling.get("top_p", 1.0)), "top_k": int(sampling.get("top_k", 0)),
                     "seeds": [] if sampling.get("seed") is None else [int(sampling["seed"])], "logprobs": sampling.get("logprobs")}
    if int(sampling.get("n") or 1) != 1:
        sampling_desc["n"] = int(sampling["n"])                     # greedy manifests stay byte-identical (same profile id as before)
    if sampling.get("per_request"):
        sampling_desc["per_request"] = list(sampling["per_request"])   # mixed batch: per-request variants
    try:
        manifest = prof.build_manifest(
            case=case, cp=engine.cp, llm=engine.llm, wheel_sha256=wheel_sha256, vllm_commit=vllm_commit,
            adapter_revision=adapter_revision,
            interception="none (production uninstrumented: StepTrace scheduler observer + engine outputs; Inductor/Triton caches read after the run)",
            # the header says what the engine was built with (a compiled engine used to be labelled enforce_eager True here)
            sampling=sampling_desc, env_flags=engine.env_flags, enforce_eager=bool(engine.kwargs.get("enforce_eager", False)),
            # the manifest's `parallelism.tensor_parallel_size` / `engine_kwargs.tensor_parallel_size` must say what the
            # engine was built with (the default 1 made a TP=2 rank's header claim TP=1; only the `backends.all_reduce` probe differed)
            tensor_parallel_size=int(engine.kwargs.get("tensor_parallel_size") or 1),
        )
        # the production recorder adds these after the run; eager production has no generated kernels
        manifest["compilation"]["generated_kernels"] = {"wrappers": 0, "distinct_kernel_sources": 0, "kernel_set_sha256": None,
                                                        "call_set_sha256": None, "ptx_code_set_sha256": None, "ptx_variants": 0}
        manifest["engine_kwargs"]["enable_prefix_caching"] = bool(engine.kwargs.get("enable_prefix_caching", False))
        manifest["engine_kwargs"]["max_num_seqs"] = engine.kwargs.get("max_num_seqs")
        if engine.kwargs.get("max_num_batched_tokens") is not None:
            manifest["engine_kwargs"]["max_num_batched_tokens"] = engine.kwargs["max_num_batched_tokens"]
        # the served dtype: `build_manifest` hard-codes "bfloat16"; when the harness passed an explicit
        # `dtype` to LLM(**kw) (a workload's `engine.args.dtype`, e.g. float16) the manifest must say so -- a served-fp16
        # record labelled bf16 is mislabelled evidence.  Runs without the kwarg are byte-identical (same profile id as before).
        if engine.kwargs.get("dtype") not in (None, "auto"):
            manifest["dtype"] = str(engine.kwargs["dtype"])
            manifest["engine_kwargs"]["dtype"] = str(engine.kwargs["dtype"])
        # the attention version the layers actually run (FA2 / FA3), and the declared target when the engine was built for one:
        # `TargetProfile.from_run_profile` reads these back on the CPU side (eager resolver) — the same object the builder applied
        manifest.setdefault("backends", {})["attention_flash_attn_version"] = observed_flash_attn_version(engine.model)
        manifest["backends"]["vllm_has_flashinfer"] = vllm_has_flashinfer()     # host fact vLLM keys kernel choices on (nvcc on PATH)
        # the block-scaled FP8 linear kernel the FP8 layers carry (None for a model without FP8 linears): the record attests
        # the kernel whose arithmetic the FP8 Definitions are (CutlassFp8BlockScaledMMKernel), beside the declared `target.fp8_block_gemm`
        from verity_vllm.program.frontend.target_profile import fp8_block_gemm_kernel_of
        fp8k = fp8_block_gemm_kernel_of(engine.model)
        if fp8k is not None:
            manifest["backends"]["fp8_block_gemm_kernel"] = fp8k
        # the fused-MoE experts class, attested only under a declared `target.moe_backend`: every MoE layer has one, so an unconditional
        # key would move the profile id of every earlier MoE run
        if engine.target is not None and engine.target.moe_backend is not None:
            from verity_vllm.program.frontend.target_profile import moe_experts_class_of
            manifest["backends"]["moe_experts_class"] = moe_experts_class_of(engine.model)
        if getattr(engine.target, "int8_exact", None) is not None:
            from verity_vllm.program.frontend.target_profile import int8_exact_of
            manifest["backends"]["int8_exact"] = int8_exact_of(engine.model)
            if "attention" in engine.target.int8_parts:
                from verity_vllm.program.frontend.kernel_readback import int8_k_scales_of
                manifest["backends"]["int8_k_scales"] = int8_k_scales_of(engine.model)
        # the MoE gates' cuteDSL ll_bf16_gemm tier (M <= 16), which leaves no torch op behind: written only when a gate took it, so
        # every record without such a gate keeps its profile id
        if moe_gate_ll_bf16_gemm_of(engine.model):
            manifest["backends"]["moe_gate_ll_bf16_gemm"] = True
        manifest["engine_kwargs"]["kernel_config"] = dict(engine.kwargs.get("kernel_config") or {})
        if engine.target is not None:
            manifest["target"] = engine.target.to_json()
            manifest["engine_kwargs"].update({k: v for k, v in engine.target.engine_args().items()})
        # the effective execution, read back from the built engine: engine_kwargs.enforce_eager = what LLM() got (eager runs
        # stay byte-identical: True as before); a compiled engine adds its compilation_config kwarg and the read-back facts (mode, cudagraph
        # mode, capture sizes, partition) to the header's `compilation` block -- the RunHeader attests the execution the global match compares against
        manifest["engine_kwargs"]["enforce_eager"] = bool(engine.kwargs.get("enforce_eager", False))
        if engine.kwargs.get("compilation_config") is not None:
            manifest["engine_kwargs"]["compilation_config"] = engine.kwargs["compilation_config"]
        # the record attests the async-scheduling mode the engine RESOLVED (scheduler_config after VllmConfig.__post_init__: True by
        # default on this stack), not the CLI default None the harness leaves in place -- async scheduling changes the step/observer
        # interleaving the global match and the Match resolver reason about, so a header without it is unattested execution.
        manifest["engine_kwargs"]["async_scheduling"] = resolved_async_scheduling(engine.llm)
        if "async_scheduling" in engine.kwargs:
            notes["async_scheduling_requested"] = engine.kwargs["async_scheduling"]
        facts = compilation_facts(engine.llm)
        if facts and facts.get("enforce_eager") is False:
            manifest.setdefault("compilation", {}).update({k: v for k, v in facts.items() if k != "enforce_eager"})
            manifest["compilation"]["enforce_eager"] = False
            # positive, record-level evidence of compilation an eager run cannot produce: the Inductor
            # output code of every compiled piece is on disk by now (vLLM compiles + captures during engine init), digested exactly as
            # commit_delta does (kernel_identity.collect: kernel_set_sha256 over {(kernel name, source sha256)}, call_set over the launch
            # programs).  The dir is vLLM's per-config inductor_cache (it exports TORCHINDUCTOR_CACHE_DIR at initialize_cache) or the explicit
            # per-process override; a null pin here on a compiled engine is an error of record (the global match's G1 fails closed on it).
            gk = generated_kernels_of_this_process()
            # the manifest (-> profile id) carries production.py's summary shape only; the per-piece inventory and the dir it was read from
            # go to the header notes (paths differ per process; the digests do not)
            manifest["compilation"]["generated_kernels"] = {k: gk.get(k) for k in ("wrappers", "distinct_kernel_sources", "kernel_set_sha256",
                                                                                   "call_set_sha256", "wrapper_norm_set_sha256")}
            manifest["compilation"]["generated_kernels"].update({"ptx_code_set_sha256": None, "ptx_variants": 0})
            if gk.get("error"):
                manifest["compilation"]["generated_kernels"]["error"] = gk["error"]
            notes["generated_kernels"] = {k: gk.get(k) for k in ("inductor_cache_dir", "per_process_cache_dir", "pieces", "kernels", "error")}
        pid = prof.profile_id_of(manifest)
        notes["profile_id_method"] = "profile.profile_id_of(profile.build_manifest(...)) as production.py; differs from the accepted " \
                                     "record's id by construction: gpu.pod, gpu.driver, adapter.revision, sampling.logprobs=None (D62), " \
                                     "engine_kwargs.enable_prefix_caching/max_num_seqs, generated_kernels placeholders"
    except Exception as e:  # noqa: BLE001
        import vllm
        import triton
        manifest = {"schema": "verity-capture/profile-fallback/v1", "case": case, "engine_kwargs": {k: v for k, v in engine.kwargs.items()
                    if k not in ("model", "tokenizer")}, "engine_flags": dict(engine.env_flags),
                    "python": {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__},
                    "vllm": {"version": vllm.__version__, "wheel_sha256": wheel_sha256, "commit": vllm_commit},
                    "model": {"repo": engine.cp["repo"], "revision": engine.cp["revision"], "weights_sha256": dict(engine.cp["safetensors_sha256"])},
                    "sampling": sampling_desc}
        pid = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
        notes["profile_id_method"] = f"FALLBACK sha256(engine kwargs + versions): build_manifest failed: {e!r}"[:400]
    return manifest, pid, notes


def moe_gate_ll_bf16_gemm_of(model) -> bool:
    """READ BACK whether any MoE gate (vLLM `GateLinear`) is eligible for its cuteDSL `ll_bf16_gemm` tier: its
    `allow_ll_bf16_gemm`, set at load from the device, the weight and out dtypes and cuteDSL's availability."""
    return any(getattr(m, "allow_ll_bf16_gemm", False) is True for m in model.modules())


def model_config_subset(engine: Engine) -> dict:
    hf = engine.llm.llm_engine.vllm_config.model_config.hf_config
    keys = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
            "vocab_size", "rms_norm_eps", "tie_word_embeddings", "rope_theta", "max_position_embeddings", "torch_dtype",
            "model_type", "architectures", "attention_bias", "mlp_bias", "rope_parameters", "rope_scaling",
            "qkv_bias", "attn_logit_softcapping", "final_logit_softcapping", "sliding_window", "hidden_act", "hidden_activation",
            # the LayerNorm / GPT-NeoX family's keys (absent on every other config, so their
            # records are unchanged): eps and rotary base under their names, the partial-rotary fraction, the parallel residual
            "layer_norm_eps", "layer_norm_epsilon", "rotary_emb_base", "rotary_pct", "partial_rotary_factor", "use_parallel_residual",
            # multi-head latent attention's and the MoE routing's keys, which an MLA configuration's
            # config root requires (absent on dense configurations; a MoE one gains the routing keys its profile's
            # header check compares when recorded)
            "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "first_k_dense_replace",
            "moe_intermediate_size", "num_experts", "num_local_experts", "n_routed_experts", "num_experts_per_tok", "norm_topk_prob",
            "n_group", "topk_group", "routed_scaling_factor", "n_shared_experts")
    out = {}
    for k in keys:
        v = getattr(hf, k, None)
        if v is not None:
            out[k] = v if isinstance(v, (int, float, str, bool, list)) else str(v)
    return out


def host_doc() -> dict:
    """CPU-side facts that bound the observer's Python cost and would otherwise confound overhead ladders across pods:
    the CPU model, the visible/quota cores, the kernel clocksource, and the measured cost of one `time.perf_counter()` read.
    On an `acpi_pm` host a clock read is a ~8 µs syscall (vs ~25 ns with `tsc`/`kvm-clock`), and the hook reads the clock
    four times per recorded call (M6 report, addendum)."""
    doc: dict[str, Any] = {"nproc": os.cpu_count()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                doc["cpu"] = line.split(":", 1)[1].strip()
                break
    except (OSError, ValueError):
        pass
    try:
        q = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        doc["cpu_quota"] = None if q[0] == "max" else float(q[0]) / float(q[1])
    except (OSError, IndexError, ValueError, ZeroDivisionError):
        pass
    try:
        doc["clocksource"] = Path("/sys/devices/system/clocksource/clocksource0/current_clocksource").read_text().strip()
    except (OSError, ValueError):
        pass
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        time.perf_counter()
    doc["clock_read_us"] = round((time.perf_counter() - t0) / n * 1e6, 3)
    return doc


def versions_doc(engine: Engine | None = None) -> dict:
    import importlib.metadata as md

    def ver(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    import triton
    import vllm

    doc = {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__, "vllm": vllm.__version__,
           "python": os.sys.version.split()[0], "numpy": np.__version__, "flashinfer": ver("flashinfer-python"),
           "transformers": ver("transformers")}
    try:
        props = torch.cuda.get_device_properties(0)
        doc["gpu"] = {"name": props.name, "cc": f"{props.major}.{props.minor}", "sms": int(props.multi_processor_count)}
    except (AssertionError, RuntimeError, torch.cuda.DeferredCudaCallError):
        pass
    doc["host"] = host_doc()
    if engine is not None:                    # the effective LLM(**kw) of this run (CLI overrides included)
        doc["engine_kwargs"] = {k: v for k, v in engine.kwargs.items() if k not in ("model", "tokenizer") and isinstance(v, (int, float, str, bool, type(None)))}
        # capture/versions.json is the global match's second engine-fact source (record_engine_facts: every location must
        # agree): the RESOLVED async_scheduling, the same read-back the RunHeader.profile carries -- never the requested kwarg (usually absent)
        doc["engine_kwargs"]["async_scheduling"] = resolved_async_scheduling(engine.llm)
        # the execution the workload DECLARED and the one the engine REALISES (vllm_config read-back), in every
        # run header that goes through this engine (Match capture/control, Commit control+treatment = the same engine object): the finalizer
        # compares `effective_execution.execution_label` with the row id and the Program's applicability.
        doc["declared_execution"] = getattr(engine, "execution", None)
        doc["effective_execution"] = getattr(engine, "effective", None)
    try:
        export = Path("EXPORT.json")
        if export.exists():
            doc["export"] = json.loads(export.read_text())
    except (OSError, ValueError):
        pass
    return doc
