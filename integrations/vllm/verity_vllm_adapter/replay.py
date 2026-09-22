"""Replay reference / replay+commit runner (contract §3 modes 2 and 3, §15.3/§15.4).

Same engine configuration as production plus the replay restrictions (``enable_trace_replay=True``,
``enforce_eager=True`` for the dispatch-mode rung), every production request re-issued with
``SamplingParams(trace_decode_token_ids=<recorded output ids>, logprobs=1, temperature=0)``.

``--mode reference``  : capture leaf hashes only (`ru/<rid>/<step>/leaves.json` sidecars) unless
                        ``--store-values``; used twice in fresh processes for noninterference.
``--mode commit``     : full capture in W2's format (values.bin + index.json per RU, boundaries,
                        weights, description.json, required.json) and ``commitments.json`` via
                        ``verity_vllm.commit``.

Written per run: ``replay.json`` (trajectory-followed check, unforced-sampler agreement, replay
restrictions, effective execution mode, capture summary and overhead, warm-up and buffer-reuse
tests), ``capture_summary.json``, ``kv_audit.json``, ``sampler_unforced.json``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from verity_vllm_adapter import profile as prof

ENV_FLAGS = prof.apply_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from verity_vllm import commit as vcommit  # noqa: E402
from verity_vllm import required as vrequired  # noqa: E402
from verity_vllm.capture import RunWriter  # noqa: E402
from verity_vllm.description import Description  # noqa: E402
from verity_vllm_adapter import kernel_identity, state_rows, weight_sources  # noqa: E402
from verity_vllm_adapter.capture import Capture, PinnedArena, install_kernel_wrappers  # noqa: E402
from verity_vllm_adapter.compiled import CompiledCapture  # noqa: E402
from verity_vllm_adapter.driver import run_requests  # noqa: E402
from verity_vllm_adapter.steptrace import StepTrace  # noqa: E402

PARTITION = {"partition_id": "row-default", "decomposition": {"default": "row"}}

TRACE_REPLAY_API = {
    "n": "must be 1",
    "max_tokens": "overridden to len(trace_decode_token_ids)",
    "stop/stop_token_ids/eos": "stopping disabled; finish_reason is 'length' at the trace end",
    "speculative_decoding": "rejected",
    "structured_outputs": "rejected",
    "prompt_logprobs": "rejected",
    "logprobs": "allowed; logprob of the forced token reflects the real distribution",
}

# §16.1 compiled-preserving replay: production's compilation, CUDA graphs off in replay only
REPLAY_RESTRICTIONS_COMPILED = {
    "enforce_eager": False,
    "cudagraph": "off in replay only (compilation_config.cudagraph_mode=NONE; production ran FULL_AND_PIECEWISE): the Python wrapper of every compiled piece must execute for the launch-level capture",
    "torch_compile": "on, production's Inductor configuration (mode 3, custom_ops none, piecewise at attention); generated kernel sources compared by content hash with production's (replay.json kernel_identity)",
    "enable_trace_replay": True,
    "enable_prefix_caching": "off (production keeps the stock default; every position of a replayed request is computed in its own steps; a prefix-cache hit reads another request's KV rows, the R3 row's identity rule)",
    "model_runner": "V2 (VLLM_USE_V2_MODEL_RUNNER=1; also declared in production)",
    "trace_replay_api": TRACE_REPLAY_API,
    "interception": "compiled-launch (CachingAutotuner.run wrapped inside the compiled pieces: output buffers snapshotted after each generated-kernel launch; TorchDispatchMode for the ops between/after the pieces: attention, kv-cache write, lm_head GEMM, sampler)",
}

REPLAY_RESTRICTIONS = {
    "enforce_eager": True,
    "cudagraph": "off (enforce_eager)",
    "torch_compile": "off (enforce_eager -> compilation mode NONE)",
    "enable_trace_replay": True,
    "enable_prefix_caching": "off (production keeps the stock default; every position of a replayed request is computed in its own steps; a prefix-cache hit reads another request's KV rows, the R3 row's identity rule)",
    "model_runner": "V2 (VLLM_USE_V2_MODEL_RUNNER=1; also declared in production)",
    "trace_replay_api": {
        "n": "must be 1",
        "max_tokens": "overridden to len(trace_decode_token_ids)",
        "stop/stop_token_ids/eos": "stopping disabled; finish_reason is 'length' at the trace end",
        "speculative_decoding": "rejected",
        "structured_outputs": "rejected",
        "prompt_logprobs": "rejected",
        "logprobs": "allowed; logprob of the forced token reflects the real distribution",
    },
    "interception": "dispatch-mode (TorchDispatchMode + custom-op wrappers around the batch-invariant Triton kernels)",
}


def load_production(prod_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((prod_dir / "manifest.json").read_text())
    rows = [json.loads(line) for line in (prod_dir / "production.jsonl").read_text().splitlines() if line.strip()]
    return manifest, rows


class Req:
    def __init__(self, row: dict[str, Any]) -> None:
        self.request_id = row["request_id"]
        self.prompt_token_ids = list(row["prompt_token_ids"])
        self.output_token_ids = list(row["output_token_ids"])
        self.sampling_params = dict(row["sampling_params"])
        self.finish_reason = row.get("finish_reason")
        self.chosen_logprobs = row.get("chosen_logprobs")


def replay_params(r: Req, logprobs: int | None) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=len(r.output_token_ids),
        logprobs=logprobs,
        trace_decode_token_ids=list(r.output_token_ids),
    )


def buffer_reuse_test(arena_bytes: int = 64 << 20) -> dict[str, Any]:
    """§15.4: snapshot a device tensor with the capture's mechanism, then mutate the storage and
    reuse the freed block *before* synchronising; the snapshot must hold the earlier bytes while a
    naive held reference shows the mutation."""
    arena = PinnedArena(arena_bytes)
    g = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.randn(4096, 576, device="cuda", dtype=torch.bfloat16, generator=g)
    expected = x.clone().cpu().view(torch.uint8).numpy().tobytes()
    host = arena.snapshot(x)  # async D2H on the current stream, no sync
    x.mul_(-3.0)  # in-place mutation queued behind the copy on the same stream
    held = x  # "holding a tensor is not holding its value"
    ptr = x.data_ptr()
    del x
    y = torch.zeros(4096, 576, device="cuda", dtype=torch.bfloat16)  # allocator may hand back the block
    y.add_(7.0)
    torch.cuda.synchronize()
    snap = host.view(torch.uint8).numpy().tobytes()
    return {
        "snapshot_holds_earlier_bytes": snap == expected,
        "naive_reference_changed": held.cpu().view(torch.uint8).numpy().tobytes() != expected,
        "block_reused_by_new_alloc": y.data_ptr() == ptr,
        "nbytes": len(expected),
        "mechanism": "PinnedArena.take: non_blocking copy_ into pinned host memory on the compute stream; "
                     "later in-place kernels and allocator reuse are stream-ordered behind it",
    }


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(prof.CASES))
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--production", required=True, help="production output dir (manifest.json, production.jsonl)")
    ap.add_argument("--out", required=True, help="run dir")
    ap.add_argument("--mode", choices=["reference", "commit"], required=True)
    ap.add_argument("--store-values", action="store_true", help="reference mode: also store values (W2 format)")
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--arena-gb", type=float, default=24.0)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="replay-only scheduler cap (tokens per engine step). The capture stages every recorded "
                         "output of a step in pinned host memory before it is split per request, ~2.4 MB per token for "
                         "Qwen2.5-1.5B; a pod whose host-memory cgroup cannot hold the production step size (8192 tokens) "
                         "chunks the prefill instead. A replay restriction recorded in replay.json (values are per token "
                         "under VLLM_BATCH_INVARIANT=1; the RU/step structure differs from production's)")
    ap.add_argument("--logprobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N requests (smoke tests)")
    ap.add_argument("--weights-values", choices=["store", "hash", "checkpoint"], default="store",
                help="store: bytes in the bundle; hash: sha256 only; checkpoint: bytes only for non-checkpoint buffers, "
                     "checkpoint tensors referenced by file sha256 + HF tensor names (§16.8; the verifier rebuilds and re-hashes them)")
    ap.add_argument("--skip-verify-weights", action="store_true")
    ap.add_argument("--replay-exec", choices=list(prof.REPLAY_EXECS), default="eager",
                    help="eager: enforce_eager + dispatch-mode capture (default); compiled-nograph: production's compilation with CUDA graphs off, launch-level capture (§16.1)")
    ap.add_argument("--no-captures", action="store_true", help="rung (a): run the replay and the token/logprob comparison only, no capture (kernel identity still collected)")
    ap.add_argument("--prefix-caching", action="store_true",
                    help="state rows (R3): replay WITH the prefix cache under the cross-request KV identity rule "
                         "(a hit request's KV reads bind to the source request's canonical ids; state_rows.py)")
    ap.add_argument("--engine-arg", action="append", default=[], metavar="KEY=VALUE",
                    help="replay-side LLM(...) override applied after production's declared overrides and recorded as a "
                         "replay restriction (W10 pod: max_num_batched_tokens=2048 keeps a step's captured bytes under the 31 GB container limit)")
    ap.add_argument("--rank-tag", type=int, default=None,
                    help="per-rank capture (lane TP, a port's adapter.capture_per_rank): every value id and weight id of this "
                         "process carries s=<rank> (the canonical id's shard field, capture-format.md), so the ranks' captures "
                         "can be written to separate run dirs and merged by id without collision; tensor_parallel_size is read "
                         "from the production manifest")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "replay.log"

    def log(*a: Any) -> None:
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    cp = prof.checkpoint_for(args.case, args.checkpoints)
    if not args.skip_verify_weights:
        prof.verify_local_weights(cp)
    prod_manifest, prod_rows = load_production(Path(args.production))
    profile_id = prof.profile_id_of(prod_manifest)
    reqs = [Req(r) for r in prod_rows]
    if args.limit:
        reqs = reqs[: args.limit]
    prompt_lens = {r.request_id: len(r.prompt_token_ids) for r in reqs}

    compiled = args.replay_exec == "compiled-nograph"
    prod_eager = bool(prod_manifest.get("engine_kwargs", {}).get("enforce_eager", False))
    if compiled and prod_eager:
        raise SystemExit("compiled-nograph replay requires a compiled production profile (this one declared enforce_eager)")
    # eager: the dispatcher-visible wrappers must exist before anything runs. compiled-nograph: they
    # must be installed only AFTER the engine compiled, so the traced pieces contain production's
    # original Triton GEMM (a custom op at trace time would change the generated code).
    patches = [] if compiled else install_kernel_wrappers()
    # max_num_batched_tokens: the production's declared scheduler flag (M2, varied batch composition;
    # replay must use it so the recorded schedule is reproducible) unless the W3c replay-only override
    # for host-memory-bound pods is given on the command line.
    prod_mnbt = prod_manifest.get("engine_kwargs", {}).get("max_num_batched_tokens")
    # tensor parallelism (lane TP): a declared production flag read from the manifest. With TP > 1 the
    # model lives in vLLM's worker processes (the driver holds no model, so in-process interception
    # observes nothing): the capture is installed in every worker (tp_worker.py, `worker_extension_cls`),
    # rank r's stream lands in <out>/rank<r>/ with `s=<r>` on every id, and tp_merge joins the ranks
    # into <out>/. CH's `--rank-tag` (one process, one rank) applies to the single-process capture only.
    tp = int((prod_manifest.get("parallelism") or {}).get("tensor_parallel_size", 1) or 1)
    prod_tp = tp  # [QZ, additive] replay_doc below names `prod_tp` (TP merge 1bd8af4 left it undefined: NameError at the end of every TP=1 replay)
    if tp > 1 and compiled:
        raise SystemExit("compiled-nograph replay under tensor parallelism is not implemented (eager TP replay only)")
    if tp > 1 and args.rank_tag is not None:
        raise SystemExit("--rank-tag names one rank of a single-process capture; a TP > 1 production is captured in every "
                         "worker process (tp_worker.py), each rank tagged by its own rank")
    if tp > 1:
        # the per-rank capture hashes and writes every leaf inside the worker's execute_model; a full-store
        # prefill step of the 16-request workload takes longer than the multiproc executor's default 300 s
        # RPC timeout (replay-side setting; production is untouched)
        os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "14400")
    # [QZ, additive] quantization (lane QZ, port b1-fp8): the production's declared scheme, read from the manifest, so
    # the replay loads (and online-quantises) the weights the same way and runs the same FP8 kernels under the capture
    prod_quant = prod_manifest.get("quantization")
    kw = prof.engine_kwargs(cp, replay=True, enforce_eager=prod_eager, replay_exec=args.replay_exec,
                            max_num_batched_tokens=prod_mnbt, tensor_parallel_size=tp, quantization=prod_quant)
    # state rows (W10): production's declared scheduler/cache overrides (manifest.engine_overrides:
    # chunk size, KV budget, ...) are replayed under the same values; the prefix cache stays off
    # unless --prefix-caching (R3). --engine-arg KEY=VALUE overrides come last and are recorded as
    # a replay restriction.
    replay_overrides = state_rows.replay_engine_overrides(prod_manifest, prefix_caching=args.prefix_caching)
    replay_args = state_rows.parse_engine_args(args.engine_arg)
    replay_overrides.update(replay_args)
    kw.update(replay_overrides)
    if args.max_num_batched_tokens is not None:  # W3c: host-memory-bound pods
        kw["max_num_batched_tokens"] = int(args.max_num_batched_tokens)
        log(f"replay restriction: max_num_batched_tokens={args.max_num_batched_tokens} "
            f"(host-memory bound pod; production ran {'with max_num_batched_tokens=' + str(prod_mnbt) if prod_mnbt is not None else 'stock'})")
    replay_restrictions = {
        **(REPLAY_RESTRICTIONS_COMPILED if compiled else REPLAY_RESTRICTIONS),
        **({"interception": "none (--no-captures: dispatch mode never entered, autotuner untouched; token/logprob agreement and kernel identity only)"} if args.no_captures else {}),
        **({"max_num_batched_tokens": f"{args.max_num_batched_tokens} (replay only; production {'declared ' + str(prod_mnbt) if prod_mnbt is not None else 'stock 8192'}): the pod's host-memory cgroup cannot stage a production-size step"}
           if args.max_num_batched_tokens is not None and int(args.max_num_batched_tokens) != prod_mnbt else {}),
    }
    if replay_args:
        replay_restrictions["engine_args"] = dict(replay_args)
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    if compiled:
        patches = install_kernel_wrappers()
    hf_config = llm.llm_engine.vllm_config.model_config.hf_config
    if tp > 1:
        runner = model = None  # in the worker processes
    else:
        runner = prof.get_model_runner(llm)
        model = runner.get_model()
    scheduler = prof.get_scheduler(llm)
    effective = prof.effective_compilation(llm)
    log(f"engine built in {build_s:.1f}s; effective compilation {effective}; tensor_parallel_size={tp}")
    if compiled and str(effective.get("cudagraph_mode")).split(".")[-1] not in ("NONE", "CUDAGraphMode.NONE"):
        raise SystemExit(f"compiled-nograph replay: effective cudagraph_mode is {effective.get('cudagraph_mode')}, expected NONE")

    # --- warm-up without capture (§15.4: captures are of actual execution) ---
    # Synthetic prompts, not workload ones: a workload prompt warmed here would be a prefix-cache hit
    # in the captured run and its prefill positions would never be captured.
    rng = np.random.default_rng(20260907)
    vocab = int(getattr(hf_config, "vocab_size", 32000))
    warm = [
        Req({"request_id": f"warmup-{i}", "prompt_token_ids": rng.integers(3, vocab - 1, size=64).tolist(),
             "output_token_ids": rng.integers(3, vocab - 1, size=8).tolist(), "sampling_params": {}, "finish_reason": None})
        for i in range(2)
    ]
    _, warm_timing = run_requests(llm, warm, lambda r: replay_params(r, args.logprobs))
    log(f"warm-up ({len(warm)} requests) {warm_timing['wall_s']:.2f}s")

    # --- capture ---
    store_values = (args.mode == "commit" or args.store_values) and not args.no_captures
    unforced: dict[str, dict[int, int]] = {}
    cap = None
    cc = None
    writer = None
    tp_install = None
    if tp > 1:
        # per-rank capture inside the worker processes (tp_worker.py); the driver holds no model
        tp_install = llm.collective_rpc("verity_tp_install", kwargs={
            "run_dir": str(out), "profile_id": profile_id, "manifest": prod_manifest, "store_values": store_values,
            "prompt_lens": prompt_lens, "arena_bytes": int(args.arena_gb * (1 << 30)), "no_record": bool(args.no_captures)})
        log(f"per-rank capture installed on {len(tp_install)} ranks: {tp_install}")
    else:
        writer = RunWriter(out, prod_manifest) if store_values else None
        if writer is not None and writer.profile_id != profile_id:
            raise SystemExit(f"profile_id mismatch: writer {writer.profile_id} vs production {profile_id}")
        cap = Capture(
            runner=runner, model=model, profile_id=profile_id, run_dir=out, writer=writer,
            hash_only=not store_values, prompt_lens=prompt_lens, arena_bytes=int(args.arena_gb * (1 << 30)), log=log,
            tokens={r.request_id: r.prompt_token_ids + r.output_token_ids for r in reqs}, cross_request_kv=args.prefix_caching,
            shard=args.rank_tag,
        )
        cap.no_record = bool(args.no_captures)
        if compiled:
            cc = CompiledCapture(cap, log=log)
            if not args.no_captures:
                cc.install()
        # unforced sampler result before trace replay overwrites it (§15.3, separate field)
        trs = getattr(getattr(runner, "sampler", None), "trace_replay_state", None)
        if trs is not None:
            orig_apply = trs.apply_trace

            def apply_trace(sampled, idx_mapping):
                cap.cur_unforced = sampled.detach().clone()
                return orig_apply(sampled, idx_mapping)

            trs.apply_trace = apply_trace
        orig_after_sample = cap.after_sample

        def after_sample(res, input_batch):
            orig_after_sample(res, input_batch)
            u = getattr(cap, "cur_unforced", None)
            if u is not None and cap.cur_logits_map is not None:
                vals = u.cpu().numpy().reshape(-1)
                for i, (rid, t) in enumerate(cap.cur_logits_map):
                    if i < len(vals) and (rid, t) in cap.cur_sampled_positions:
                        unforced.setdefault(rid, {})[t] = int(vals[i])
                cap.cur_unforced = None

        cap.after_sample = after_sample

    trace = StepTrace()
    trace.install(scheduler)
    torch.cuda.synchronize()
    t_run0 = time.perf_counter()
    outs, timing = run_requests(llm, reqs, lambda r: replay_params(r, args.logprobs))
    torch.cuda.synchronize()
    run_s = time.perf_counter() - t_run0
    trace.uninstall()
    tp_ranks: list[dict[str, Any]] = []
    if cap is not None:
        log(f"replay run {run_s:.2f}s: {cap.summary()['leaves']} leaves, {cap.ru_count} RUs, {cap.bytes_captured / 1e9:.2f} GB")
    else:
        tp_ranks = llm.collective_rpc("verity_tp_sync")
        log(f"replay run {run_s:.2f}s; per rank: {tp_ranks}")
    if cc is not None:
        cc.finalize_templates()
        cc.uninstall()

    # --- §16.1 generated-kernel identity: what Inductor generated in this process vs production ---
    kid = kernel_identity.collect(kernel_identity.inductor_cache_dir(), kernel_identity.triton_cache_dir(),
                                  vllm_cache_dir=str(getattr(llm.llm_engine.vllm_config.compilation_config, "cache_dir", "") or ""))
    (out / "kernels.json").write_text(json.dumps(kid, indent=1) + "\n")
    kernel_cmp = None
    prod_kernels = Path(args.production) / "kernels.json"
    if prod_kernels.exists():
        kernel_cmp = kernel_identity.compare(json.loads(prod_kernels.read_text()), kid)
        (out / "kernel_identity.json").write_text(json.dumps({"production": json.loads(prod_kernels.read_text())["summary"], "replay": kid["summary"], "compare": kernel_cmp}, indent=1) + "\n")
        log(f"kernel identity vs production: sources_identical={kernel_cmp['kernel_sources_identical']} wrappers_identical={kernel_cmp['wrappers_identical']} "
            f"call_programs_identical={kernel_cmp['call_programs_identical']} cubins_identical={kernel_cmp['cubins_identical']} only_a={kernel_cmp['kernel_sources']['only_a']} only_b={kernel_cmp['kernel_sources']['only_b']}")
    launched_sources = None
    if cc is not None and not args.no_captures:
        launched_sources = cc.write_sources(out)
        # every source actually launched in the capture must be among production's generated sources
        if prod_kernels.exists():
            prod_set = {k["source_sha256"] for w in json.loads(prod_kernels.read_text())["wrappers"] for k in w["kernels"]}
            not_in_prod = sorted(s for s in launched_sources if s not in prod_set)
            kernel_cmp["launched_sources_not_generated_by_production"] = not_in_prod
            log(f"launched sources {len(launched_sources)}, not among production's generated sources: {len(not_in_prod)}")
    # W1cF F8/F10 (audit-compiled-pass.md): production's autotune choice into the fused nodes, and
    # production's normalised PTX (production/ptx/) into the bundle next to the sources it compiles from.
    production_launch = None
    shipped_ptx = None
    if cc is not None and not args.no_captures:
        production_launch = cc.attach_production_launch_configs(json.loads(prod_kernels.read_text()) if prod_kernels.exists() else None)
        log(f"production launch configs: {production_launch['with_production_config']} nodes pinned to production's best_config, "
            f"{len(production_launch['without'])} without one, replay's choice differed on {len(production_launch['replay_differs'])}")
        shipped_ptx = kernel_identity.ship_ptx(Path(args.production) / "ptx", out / "kernels")
        log(f"shipped production PTX: {shipped_ptx}")

    # --- boundaries / weights / description / required (commit or stored reference) ---
    requests_info = {
        r.request_id: {
            "prompt_token_ids": r.prompt_token_ids, "output_token_ids": r.output_token_ids,
            "sampling_params": r.sampling_params, "seed": r.sampling_params.get("seed"),
            "finish_reason": r.finish_reason,  # the boundary is production's protected record
            # WB2 (audit-security-baseline finding 1): the chosen logprobs as production's API reported them
            # become one float32 `output_logprob` leaf per produced token (capture-format rev. 2026-09-07c);
            # a record without them yields a boundary without the leaves (the row then says `logprobs: unbound`).
            "chosen_logprobs": r.chosen_logprobs,
        }
        for r in reqs
    }
    symbols = {"H": int(hf_config.hidden_size), "V": int(getattr(hf_config, "vocab_size", 0))}
    boundary_problems: list[str] = []
    weights_info: dict[str, Any] = {}
    tp_finish: list[dict[str, Any]] = []
    tp_merge_stats: dict[str, Any] | None = None
    tp_required_error: str | None = None
    if cap is None:
        # TP: each rank writes boundaries / its weight shards / its description into <out>/rank<r>/;
        # the ranks are then joined into <out>/ (tp_merge). The unforced-sampler record is rank 0's
        # (every rank runs the same sampler on the same gathered logits; rank 1's is kept per rank).
        tp_finish = llm.collective_rpc("verity_tp_finish", kwargs={
            "requests_info": requests_info, "model_repo": cp["repo"], "weights_values": args.weights_values, "checkpoint": cp})
        for rk in tp_finish:
            boundary_problems.extend(f"rank{rk['rank']}: {p}" for p in rk["boundary_problems"])
        unforced = {rid: {int(t): tok for t, tok in d.items()} for rid, d in tp_finish[0]["unforced"].items()}
        weights_info = {"per_rank": [rk["weights"] for rk in tp_finish]}
        from verity_vllm_adapter import tp_merge

        if not args.no_captures:
            tp_merge_stats = tp_merge.merge_ranks(out, store_values=store_values, log=log)
        description = json.loads((out / "description.json").read_text()) if (out / "description.json").exists() else {"nodes": []}
        if not store_values or not (out / "manifest.json").exists():
            (out / "manifest.json").write_bytes(prof.canonical_json_bytes(prod_manifest))
        if store_values:
            try:
                desc_obj = Description(json.loads((out / "description.json").read_text()))
                required = vrequired.freeze_required(out / "required.json", desc_obj, PARTITION)
                log(f"description: {len(desc_obj.nodes)} nodes (joined over {len(tp_finish)} ranks), required interfaces: {len(required['interfaces'])}")
            except Exception as e:  # noqa: BLE001
                tp_required_error = repr(e)
                log(f"required.json not frozen for the TP run (description parser / required-set rule predates sharded nodes): {e!r}")
    else:
        description = cap.description(cp["repo"], symbols)

    # --- trajectory-followed check + logprobs (§15.3 fields kept separate) ---
    followed = 0
    per_request = []
    for r in reqs:
        o = outs[r.request_id]
        emitted = list(o.outputs[0].token_ids)
        ok = emitted == r.output_token_ids
        followed += ok
        lps = []
        if o.outputs[0].logprobs:
            for tok, lp in zip(emitted, o.outputs[0].logprobs, strict=False):
                e = lp.get(tok)
                lps.append(e.logprob if e is not None else None)
        unf = unforced.get(r.request_id, {})
        plen = len(r.prompt_token_ids)
        agree = sum(1 for k, tok in enumerate(r.output_token_ids) if unf.get(plen - 1 + k) == tok)
        per_request.append({
            "request_id": r.request_id,
            "trajectory_followed": ok,
            "emitted_len": len(emitted),
            "finish_reason": o.outputs[0].finish_reason,
            "production_finish_reason": r.finish_reason,
            "replay_chosen_logprobs": lps,
            "production_chosen_logprobs": r.chosen_logprobs,
            "logprobs_equal": (lps == r.chosen_logprobs) if r.chosen_logprobs is not None else None,
            "unforced_sampler_agrees": agree,
            "unforced_sampler_positions": len(unf),
            "output_len": len(r.output_token_ids),
        })
    lp_eq = [p["logprobs_equal"] for p in per_request if p["logprobs_equal"] is not None]

    if writer is not None:
        for b in cap.boundaries(requests_info):
            probs = b.validate()
            if probs:
                boundary_problems.append(f"{b.request_id}: {probs}")
            else:
                writer.write_boundary(b)
        t_w0 = time.perf_counter()
        ww = writer.weights(store_values=(args.weights_values in ("store", "checkpoint")), source_sha256=cp.get("safetensors_sha256"))
        n_params = 0
        n_referenced = 0
        for ptr, (name, p) in sorted(cap.params.items(), key=lambda kv: kv[1][0]):
            if name in cap.moe_expert_params and p.dim() == 3:
                # fused expert weights [E, ...]: one leaf per expert slot (<param>.e<slot>), so a
                # per-(token, slot) check opens only the expert it needs (W9)
                for e in range(int(p.shape[0])):
                    data = p[e].detach().contiguous().cpu().reshape(-1)
                    ww.add(f"{name}.e{e}", str(p.dtype).replace("torch.", ""), list(p.shape[1:]), data.view(torch.uint8).numpy().tobytes(),
                           shard=args.rank_tag)
                    n_params += 1
                continue
            data = p.detach().contiguous().cpu().reshape(-1)
            raw = data.view(torch.uint8).numpy().tobytes()
            src = weight_sources.source_for(name, cp) if args.weights_values == "checkpoint" else None
            ww.add(name, str(p.dtype).replace("torch.", ""), list(p.shape), raw, source=src, store=(None if src is None else False),
                   shard=args.rank_tag)
            n_params += 1
            n_referenced += src is not None
        ww.finish()
        weights_info = {"parameters": n_params, "seconds": time.perf_counter() - t_w0, "values_stored": args.weights_values == "store",
                        "mode": args.weights_values, "referenced_to_checkpoint": n_referenced, "stored": n_params - n_referenced if args.weights_values != "hash" else 0}
        writer.write_description(description)
        writer.finish()
        # validate with W2's parser before freezing the required set
        desc_obj = Description(json.loads((out / "description.json").read_text()))
        required = vrequired.freeze_required(out / "required.json", desc_obj, PARTITION)
        log(f"description: {len(desc_obj.nodes)} nodes, required interfaces: {len(required['interfaces'])}")
    elif cap is not None:
        (out / "description.json").write_text(json.dumps(description, indent=1, sort_keys=True))
        (out / "manifest.json").write_bytes(prof.canonical_json_bytes(prod_manifest))

    # --- buffer-reuse test (§15.4) ---
    reuse = buffer_reuse_test()
    log(f"buffer-reuse test: {reuse}")

    if cap is not None:
        summary = cap.summary()
        kv_audit = cap.kv_audit
    else:
        # joined view of the ranks' captures: counts summed, per-rank summaries kept
        per_rank = [rk["summary"] for rk in tp_finish]
        summary = {k: sum(int(s.get(k) or 0) for s in per_rank) for k in
                   ("leaves", "bytes_captured", "unbound_inputs", "module_outputs_not_recorded", "kv_reads_unresolved", "kv_reads_checked", "arena_overflow_allocs")}
        summary.update({"steps": per_rank[0].get("steps"), "replay_units": (tp_merge_stats or {}).get("replay_units", per_rank[0].get("replay_units")),
                        "templates": per_rank[0].get("templates"), "timing": {f"rank{rk['rank']}": rk["summary"].get("timing") for rk in tp_finish},
                        "errors": [e for rk in tp_finish for e in rk["summary"].get("errors", [])],
                        "per_rank": per_rank, "tensor_parallel": {"world_size": len(tp_finish), "merge": tp_merge_stats,
                                                                  "required_json_error": tp_required_error, "install": tp_install}})
        kv_audit = {f"rank{rk['rank']}": json.loads((Path(rk["out"]) / "kv_audit.json").read_text()) for rk in tp_finish}
    (out / "capture_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (out / "kv_audit.json").write_text(json.dumps(kv_audit, indent=1, default=str) + "\n")
    (out / "sampler_unforced.json").write_text(json.dumps(unforced, indent=0) + "\n")
    (out / "steptrace.json").write_text(json.dumps({"summary": trace.summary(), "steps": trace.steps}, indent=1) + "\n")

    replay_doc = {
        "case": args.case,
        "mode": args.mode,
        "replay_exec": args.replay_exec,
        "rank_tag": args.rank_tag,
        "tensor_parallel_size": tp,
        "no_captures": bool(args.no_captures),
        "store_values": store_values,
        "profile_id": profile_id,
        "adapter_revision": args.adapter_revision,
        "argv": sys.argv,
        "engine_kwargs": {k: v for k, v in kw.items()},
        "effective_execution": effective,
        "replay_restrictions": replay_restrictions,
        "patches": patches + (cc.summary()["patches"] if cc is not None else []),
        "kernel_identity": {"replay": kid["summary"], "vs_production": kernel_cmp, "launched_sources": launched_sources,
                            "cache_dirs": {"inductor": kid["inductor_cache_dir"], "triton": kid["triton_cache_dir"], "vllm": kid["vllm_compile_cache_dir"]},
                            "production_launch_configs": production_launch, "shipped_production_ptx": shipped_ptx},
        "compiled_capture": cc.summary() if cc is not None else None,
        "requests": len(reqs),
        "trajectory_followed": {"requests": len(reqs), "followed": followed},
        "unforced_sampler_agreement": {
            "positions": sum(p["output_len"] for p in per_request),
            "agree": sum(p["unforced_sampler_agrees"] for p in per_request),
            "recorded": sum(p["unforced_sampler_positions"] for p in per_request),
            "note": "argmax of the replay logits before trace replay overwrote the sampled token, vs the production token at that position",
        },
        "logprobs_vs_production": {"requests_compared": len(lp_eq), "equal": sum(lp_eq)},
        "per_request": per_request,
        "timing": {
            "engine_build_s": build_s, "warmup": warm_timing, "run_s": run_s, "driver": timing,
            "capture": summary["timing"], "weights": weights_info,
        },
        "capture": {k: summary.get(k) for k in ("steps", "replay_units", "leaves", "bytes_captured", "templates", "unbound_inputs",
                                                "module_outputs_not_recorded", "kv_reads_unresolved", "kv_reads_checked", "arena_overflow_allocs")},
        "tensor_parallel": summary.get("tensor_parallel"),
        "state_rows": state_rows.replay_state_summary(summary, trace.summary(), replay_overrides, prefix_caching=args.prefix_caching),
        "warmup_excluded": {"requests": len(warm), "note": "capture installed after the warm-up run; LLM() construction runs its own dummy/profile forwards before that"},
        "buffer_reuse_test": reuse,
        "boundary_problems": boundary_problems,
        "utc": prof.utc_stamp(),
    }
    (out / "replay.json").write_text(json.dumps(replay_doc, indent=1, default=str) + "\n")
    if boundary_problems:
        log(f"BOUNDARY_PROBLEMS {len(boundary_problems)} request(s) without a valid boundary; no commitments built: {boundary_problems[:3]}")
        raise SystemExit(3)

    if writer is not None or (cap is None and store_values):
        if cap is not None:
            # Hashing ~30 GB of committed values maps every values.bin; release the capture's pinned
            # arena (locked, unreclaimable host memory) first. A 31 GB container limit (W10 pod)
            # OOM-killed this step with the arena still allocated.
            cap.arena.buf = None
            gc.collect()
            host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
            if callable(host_empty_cache):
                host_empty_cache()
        t_c0 = time.perf_counter()
        commitments = vcommit.build_commitments(out, recompute_leaves=True, write=True)
        replay_doc["commitments"] = {"seconds": time.perf_counter() - t_c0, "trees": len(commitments.get("trees", {})),
                                     "sha256": sha256_file(out / "commitments.json")}
        (out / "replay.json").write_text(json.dumps(replay_doc, indent=1, default=str) + "\n")
        log(f"commitments: {replay_doc['commitments']}")
    log(f"DONE mode={args.mode} followed={followed}/{len(reqs)} leaves={summary['leaves']} unbound={summary['unbound_inputs']} "
        f"module_unrecorded={summary['module_outputs_not_recorded']} kv_unresolved={summary['kv_reads_unresolved']}"
        + (f" tp={len(tp_finish)} ranks" if tp_finish else ""))


if __name__ == "__main__":
    main()
