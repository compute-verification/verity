"""Compiled-preserving replay driver (W1d, CONTRACT §16.1 rungs (a)/(b)).

Modes
  inspect  build the compiled engine with the pass in dry-run + graph-dump mode, run one short request,
           write ``fx_ids.json`` + ``fx_graphs/`` (what the final post-grad graphs look like, which
           buffers the policy would record). No record ops are inserted.
  record   rung (a): record ops inserted, sink enabled per step (pinned snapshots + sha256 digests), no
           commitments. Writes ``replay_compiled.json`` (token/logprob/unforced-sampler agreement with
           production, record-op statistics), ``fx_ids.json``, ``record_digests.jsonl`` (per step,
           per id: shape, dtype, sha256 of the observed bytes), and, given ``--production-inductor-cache``,
           ``kernel_identity.json`` from :mod:`.kernels`.
  commit   rung (b): as ``record`` plus the capture-format bundle (RUs, boundaries, weights, description)
           written by :mod:`.capture` (CompiledCapture).

The engine is production's compiled configuration with ``cudagraph_mode=NONE`` (the only change; the
splitting ops, the Inductor config and ``custom_ops`` are production's) plus vLLM's trace replay and
prefix caching off (same replay restrictions as the eager replay, §15.x). Run with
``TORCHINDUCTOR_CACHE_DIR`` and ``VLLM_CACHE_ROOT`` pointing at fresh directories so the replay compile
is a real compile (the pass would be skipped on a vLLM compile-cache hit) and its kernel sources can be
compared with production's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity_vllm_adapter import profile as prof
from verity_vllm_adapter.capture import PinnedArena
from verity_vllm_adapter.compiled_fx import fx_pass, kernels, record_op
from verity_vllm_adapter.driver import external_id, run_requests
from verity_vllm_adapter.replay import Req, buffer_reuse_test, load_production, replay_params

ADAPTER_MODULE = "verity_vllm_adapter.compiled_fx"


REPLAY_RESTRICTIONS = {
    "enforce_eager": False,
    "torch_compile": "on: production's compiled graph (Inductor, piecewise split at the attention ops, custom_ops none); generated Triton kernel sources identical to production's compile (kernel_identity.json)",
    "cudagraph": "off in replay (cudagraph_mode=NONE; production FULL_AND_PIECEWISE): the record op's Python body copies buffers per call",
    "post_grad_pass": "verity RecordInsertionPass appended to vLLM's PostGradPassManager in the replay process only (verity::record side-effecting custom op after materialised buffers); production compiles without it",
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
    "interception": "compiled-fx-record-pass (verity::record inside the compiled graph, pinned-host copies) + eager-tail hooks (compute_logits / sample for lm_head and sampler rows)",
    "granularity": "compiled: one node per Inductor kernel launch (fused add+rmsnorm, silu*mul, rope, embedding+rmsnorm) and per user-Triton GEMM; attention outside the graph",
}


def restrictions_for(cudagraph: str, record_impl: str) -> dict[str, Any]:
    r = dict(REPLAY_RESTRICTIONS)
    if cudagraph == "production":
        r["cudagraph"] = ("on in replay (production's compilation_config unchanged, cudagraph_mode as production's effective FULL_AND_PIECEWISE): the record op is "
                          "captured inside the CUDA graphs; its device-side copy into a preallocated ring slot replays with the graph and the slots are drained after each step")
    if record_impl == "device":
        r["interception"] = ("compiled-fx-record-pass (verity::record inside the compiled graph, device ring-buffer copies drained per step) + "
                             "eager-tail hooks (compute_logits / sample for lm_head and sampler rows)")
    return r


TP_WORKER_MODULE = "verity_vllm_adapter.compiled_fx.tp_worker"
TP_WORKER_EXTENSION = TP_WORKER_MODULE + ".VerityCompiledFxTPWorkerExtension"
TP_WORKER_ENV_KEY = "VERITY_COMPILED_FX_TP"  # == tp_worker.ENV_KEY; the driver must not import tp_worker (see run_tp)


def compiled_engine_kwargs(cp: dict[str, Any], prod_manifest: dict[str, Any], cudagraph: str = "none", tensor_parallel_size: int = 1) -> dict[str, Any]:
    kw = prof.engine_kwargs(cp, replay=True, tensor_parallel_size=tensor_parallel_size)
    if tensor_parallel_size > 1:
        # [COMP, CX1] the compiled-fx capture's own worker extension (per-rank record pass + capture), not the eager one
        kw["worker_extension_cls"] = TP_WORKER_EXTENSION
    kw.pop("enforce_eager", None)  # compiled replay: the whole point
    prod_kw = prod_manifest.get("engine_kwargs", {})
    if prod_kw.get("enforce_eager"):
        raise SystemExit("production row is eager; compiled replay needs a compiled production profile")
    cc: dict[str, Any] = dict(prod_kw.get("compilation_config") or {})
    if cudagraph == "none":
        cc["cudagraph_mode"] = "NONE"
    elif cudagraph != "production":
        raise SystemExit(f"unknown --cudagraph {cudagraph}")
    kw["compilation_config"] = cc
    return kw


class StepHooks:
    """Per-step lifecycle around ``runner.execute_model`` / ``runner.sample``: enables the record sink
    for the forward, drains it after sampling (sync once, digest every observed buffer), keeps the
    row -> (request, position) maps, and the unforced sampler result."""

    def __init__(self, runner: Any, arena_bytes: int, prompt_lens: dict[str, int], hash_records: bool, on_step: Any = None) -> None:
        self.runner = runner
        self.arena = PinnedArena(arena_bytes)
        self.prompt_lens = prompt_lens
        self.hash_records = hash_records
        self.on_step = on_step  # callback(step_ctx) after drain (commit mode)
        self.step_index = 0
        self.steps: list[dict[str, Any]] = []
        self.unforced: dict[str, dict[int, int]] = {}
        self.cur_unforced: torch.Tensor | None = None
        self.digest_rows: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.timing = {"forward_s": 0.0, "drain_s": 0.0}
        self.ctx: dict[str, Any] = {}

    def install(self) -> None:
        runner = self.runner
        hooks = self
        orig_execute = runner.execute_model
        orig_sample = runner.sample

        def execute_model(scheduler_output, *a, **kw):
            dummy = kw.get("dummy_run", False) or (len(a) >= 2 and a[1])
            if dummy or scheduler_output.total_num_scheduled_tokens == 0:
                return orig_execute(scheduler_output, *a, **kw)
            hooks.begin_step(scheduler_output)
            t0 = time.perf_counter()
            try:
                out = orig_execute(scheduler_output, *a, **kw)
            finally:
                record_op.SINK.enabled = False
            hooks.timing["forward_s"] += time.perf_counter() - t0
            hooks.after_forward()
            return out

        def sample(hidden_states, input_batch, grammar_output):
            res = orig_sample(hidden_states, input_batch, grammar_output)
            hooks.after_sample(res, input_batch)
            return res

        runner.execute_model = execute_model
        runner.sample = sample
        trs = getattr(getattr(runner, "sampler", None), "trace_replay_state", None)
        if trs is not None:
            orig_apply = trs.apply_trace

            def apply_trace(sampled, idx_mapping):
                hooks.cur_unforced = sampled.detach().clone()
                return orig_apply(sampled, idx_mapping)

            trs.apply_trace = apply_trace

    def begin_step(self, scheduler_output) -> None:
        record_op.SINK.reset()
        self.arena.reset()
        record_op.SINK.snapshot = self.arena.snapshot
        record_op.SINK.enabled = True
        self.ctx = {"scheduler_output": scheduler_output}
        self.cur_unforced = None

    def after_forward(self) -> None:
        st = self.runner.execute_model_state
        ib = st.input_batch
        n = ib.num_reqs
        qsl = np.asarray(ib.query_start_loc_np[: n + 1])
        nct = np.asarray(ib.num_computed_tokens_np[:n])
        req_ids = [external_id(r) for r in ib.req_ids]
        rows_req = np.repeat(np.arange(n), np.diff(qsl))
        rows_pos = nct[rows_req] + (np.arange(qsl[-1]) - qsl[rows_req])
        self.ctx.update({
            "input_batch": ib, "state": st, "req_ids": req_ids, "qsl": qsl, "nct": nct, "rows_req": rows_req, "rows_pos": rows_pos,
            "num_tokens": int(ib.num_tokens), "num_tokens_after_padding": int(getattr(ib, "num_tokens_after_padding", ib.num_tokens)),
            "logits_map": [(req_ids[i], int(nct[i] + (qsl[i + 1] - qsl[i]) - 1)) for i in range(n)],
        })

    def is_sampled_position(self, rid: str, t: int) -> bool:
        plen = self.prompt_lens.get(rid)
        return plen is None or t >= plen - 1

    def after_sample(self, res, input_batch) -> None:
        t0 = time.perf_counter()
        sampler_output = res[0] if isinstance(res, tuple) else res
        req_ids = self.ctx.get("req_ids", [])
        try:
            sampled = sampler_output.sampled_token_ids.detach().to("cpu").numpy().reshape(len(req_ids), -1)[:, 0]
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "after_sample", "error": repr(e)})
            sampled = None
        u = self.cur_unforced
        if u is not None:
            vals = u.cpu().numpy().reshape(-1)
            for i, (rid, t) in enumerate(self.ctx["logits_map"]):
                if i < len(vals) and self.is_sampled_position(rid, t):
                    self.unforced.setdefault(rid, {})[t] = int(vals[i])
        ring = record_op.RING
        ring_drained = 0
        if ring is not None:
            m = int(self.ctx.get("num_tokens") or 0)
            if 0 < m <= ring.rows:
                # rung (c): the graphs (or the eager op bodies) copied into the ring slots; snapshot slot[:M] per key
                skip = {(r.runtime_piece, r.id) for r in record_op.SINK.records}
                drained = ring.drain(m, self.arena.snapshot, skip)
                ring_drained = len(drained)
                record_op.SINK.records.extend(drained)
            else:
                ring.written.clear()
        torch.cuda.synchronize()
        recs = list(record_op.SINK.records)
        ids = sorted({r.id for r in recs})
        step_doc = {
            "step": self.step_index, "num_reqs": len(req_ids), "num_tokens": self.ctx.get("num_tokens"), "num_tokens_after_padding": self.ctx.get("num_tokens_after_padding"),
            "record_calls": len(recs), "distinct_ids": len(ids), "duplicate_ids": len(recs) - len(ids), "arena_overflow_allocs": self.arena.overflow_allocs,
            "bytes": int(sum(r.host.numel() * r.host.element_size() for r in recs if r.host is not None)),
        }
        if ring is not None:
            step_doc["ring_drained"] = ring_drained
            step_doc["ring_host_fallback"] = len(recs) - ring_drained
        if self.hash_records:
            for r in recs:
                if r.host is None:
                    continue
                raw = r.host.contiguous().view(torch.uint8).numpy().tobytes()
                self.digest_rows.append({"step": self.step_index, "id": r.id, "piece": r.runtime_piece, "seq": r.seq, "shape": list(r.shape), "dtype": str(r.dtype).replace("torch.", ""),
                                         "sha256": hashlib.sha256(raw).hexdigest()})
        piece_templates: dict[int, set[int]] = {}
        for r in recs:
            piece_templates.setdefault(r.runtime_piece, set()).add(r.id // fx_pass.ID_STRIDE)
        step_doc["piece_templates"] = {str(k): sorted(v) for k, v in sorted(piece_templates.items())}
        step_doc["pieces_run"] = len(piece_templates)
        self.ctx.update({"records": recs, "sampled": sampled, "unforced": u, "step_index": self.step_index})
        if self.on_step is not None:
            try:
                self.on_step(self.ctx)
            except Exception as e:  # noqa: BLE001
                self.errors.append({"where": "on_step", "step": self.step_index, "error": repr(e)})
                raise
        self.steps.append(step_doc)
        self.timing["drain_s"] += time.perf_counter() - t0
        self.step_index += 1
        record_op.SINK.reset()


def compare_with_production(reqs: list[Req], outs: dict[str, Any], unforced: dict[str, dict[int, int]]) -> list[dict[str, Any]]:
    per_request = []
    for r in reqs:
        o = outs[r.request_id]
        emitted = list(o.outputs[0].token_ids)
        lps: list[float | None] = []
        if o.outputs[0].logprobs:
            for tok, lp in zip(emitted, o.outputs[0].logprobs, strict=False):
                e = lp.get(tok)
                lps.append(e.logprob if e is not None else None)
        unf = unforced.get(r.request_id, {})
        plen = len(r.prompt_token_ids)
        agree_pos = [k for k, tok in enumerate(r.output_token_ids) if unf.get(plen - 1 + k) == tok]
        disagree = [{"k": k, "t": plen - 1 + k, "production": tok, "unforced": unf.get(plen - 1 + k)} for k, tok in enumerate(r.output_token_ids) if unf.get(plen - 1 + k) != tok]
        lp_diff = None
        if r.chosen_logprobs is not None and lps:
            lp_diff = [i for i, (a, b) in enumerate(zip(lps, r.chosen_logprobs, strict=False)) if a != b]
        per_request.append({
            "request_id": r.request_id, "trajectory_followed": emitted == r.output_token_ids, "emitted_len": len(emitted), "output_len": len(r.output_token_ids),
            "finish_reason": o.outputs[0].finish_reason, "production_finish_reason": r.finish_reason,
            "logprobs_equal": (lps == r.chosen_logprobs) if r.chosen_logprobs is not None else None, "logprob_mismatch_positions": lp_diff,
            "replay_chosen_logprobs": lps, "production_chosen_logprobs": r.chosen_logprobs,
            "unforced_sampler_agrees": len(agree_pos), "unforced_sampler_positions": len(unf), "unforced_disagreements": disagree[:16],
        })
    return per_request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(prof.CASES))
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--production", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["inspect", "record", "reference", "commit"], required=True,
                    help="inspect: pass dry-run; record: rung (a) digests only; reference: hash-only leaf sidecars (noninterference reference); commit: full bundle")
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--arena-gb", type=float, default=8.0)
    ap.add_argument("--logprobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None, help="truncate every request's trajectory (small bundles)")
    ap.add_argument("--roles", default="extern_out,extern_in,graph_out,input")
    ap.add_argument("--placement", choices=["producer", "late"], default="late",
                    help="where the record call goes: right after the producer (B0 rows) or after the buffer's last in-graph reader (default; needed for Qwen2's fused bias-add+rope)")
    ap.add_argument("--lowering", choices=["implicit", "explicit"], default="explicit",
                    help="Inductor lowering of verity::record: implicit fallback (make_fallback: joins needs_realized_inputs and can realise pointwise "
                         "producers early; B0 rows) or an explicitly registered FallbackKernel lowering (default; invisible to other nodes' lowering)")
    ap.add_argument("--production-inductor-cache", default=None, help="production's TORCHINDUCTOR_CACHE_DIR copy for kernel identity")
    ap.add_argument("--weights-values", choices=["store", "hash"], default="store")
    ap.add_argument("--skip-verify-weights", action="store_true")
    ap.add_argument("--no-hash-records", action="store_true")
    ap.add_argument("--cudagraph", choices=["none", "production"], default="none",
                    help="none: cudagraph_mode=NONE in replay (rungs a/b); production: production's compilation_config unchanged, graphs captured with the record op inside (rung c)")
    ap.add_argument("--record-impl", choices=["host", "device"], default="host",
                    help="host: the op body copies to the pinned arena per call; device: preallocated device ring slots per (piece, id), drained per step (required with --cudagraph production)")
    ap.add_argument("--ring-rows", type=int, default=1024, help="rows per ring slot (>= the largest CUDA-graph capture size)")
    args = ap.parse_args()
    if args.cudagraph == "production" and args.record_impl != "device":
        raise SystemExit("--cudagraph production needs --record-impl device (the op's Python body does not run on graph replay)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "replay.log"

    def log(*a: Any) -> None:
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    env = {k: os.environ.get(k) for k in ("TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", "VLLM_BATCH_INVARIANT", "VLLM_ENABLE_V1_MULTIPROCESSING")}
    log(f"env {env}")
    if not env["TORCHINDUCTOR_CACHE_DIR"] or not env["VLLM_CACHE_ROOT"]:
        raise SystemExit("set TORCHINDUCTOR_CACHE_DIR and VLLM_CACHE_ROOT to fresh directories for the replay compile")
    # The pass runs at compile time only: a warm vLLM/Inductor cache would serve the compiled artefacts
    # (record ops included) without rebuilding the id table. Both directories are replay-private: wipe them.
    import shutil

    for k in ("TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT"):
        d = Path(env[k])
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    cp = prof.checkpoint_for(args.case, args.checkpoints)
    if not args.skip_verify_weights:
        prof.verify_local_weights(cp)
    prod_manifest, prod_rows = load_production(Path(args.production))
    profile_id = prof.profile_id_of(prod_manifest)
    reqs = [Req(r) for r in prod_rows]
    if args.limit:
        reqs = reqs[: args.limit]
    if args.max_tokens:
        for r in reqs:
            if len(r.output_token_ids) > args.max_tokens:
                r.output_token_ids = r.output_token_ids[: args.max_tokens]
                if r.chosen_logprobs is not None:
                    r.chosen_logprobs = r.chosen_logprobs[: args.max_tokens]
                r.finish_reason = "length"
                r.sampling_params = {**r.sampling_params, "max_tokens": args.max_tokens}  # debugging aid only: a truncated trajectory is not production's
    prompt_lens = {r.request_id: len(r.prompt_token_ids) for r in reqs}

    tp = int((prod_manifest.get("parallelism") or {}).get("tensor_parallel_size", 1) or 1)
    if tp > 1:
        run_tp(args, out=out, log=log, env=env, cp=cp, prod_manifest=prod_manifest, profile_id=profile_id, reqs=reqs, prompt_lens=prompt_lens, tp=tp)
        return

    # --- pass + hooks (replay process only) ---
    table = fx_pass.FxIdTable()
    roles = tuple(x for x in args.roles.split(",") if x)
    rpass = fx_pass.RecordInsertionPass(table, dry_run=(args.mode == "inspect"), dump_graphs=(args.mode == "inspect"), roles=roles, placement=args.placement)
    hooks_installed = fx_pass.install_hooks(rpass)

    from vllm import LLM  # noqa: E402  (after env + hooks)

    record_op.install()
    lowering_kind = record_op.install_inductor_lowering() if args.lowering == "explicit" else "implicit"
    log(f"verity::record Inductor lowering: {lowering_kind}")
    if args.lowering == "explicit" and lowering_kind != "explicit":
        raise SystemExit(f"explicit lowering requested but got {lowering_kind}")
    log(f"pass uuid {rpass.uuid()}; hooks {hooks_installed}")  # the uuid names the lowering kind: log it once that is settled (before the engine compiles)
    if args.record_impl == "device":
        record_op.RING = record_op.DeviceRing(args.ring_rows)
    kw = compiled_engine_kwargs(cp, prod_manifest, args.cudagraph)
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    runner = prof.get_model_runner(llm)
    model = runner.get_model()
    effective = prof.effective_compilation(llm)
    cc = llm.llm_engine.vllm_config.compilation_config
    effective.update({"splitting_ops": list(getattr(cc, "splitting_ops", []) or []), "use_inductor_graph_partition": bool(getattr(cc, "use_inductor_graph_partition", False)),
                      "compile_sizes": list(getattr(cc, "compile_sizes", []) or []), "backend": str(getattr(cc, "backend", "")), "pass_config": str(getattr(cc, "pass_config", "")),
                      "cudagraph_capture_sizes": list(getattr(cc, "cudagraph_capture_sizes", []) or []),
                      "max_cudagraph_capture_size": getattr(cc, "max_cudagraph_capture_size", None)})
    log(f"engine built in {build_s:.1f}s; effective {effective}; pass calls {rpass.calls}; pieces {len(table.pieces)}; ids {len(table.entries)}; "
        f"record lowering stats {record_op.LOWERING_STATS}")
    ring_doc = None
    if record_op.RING is not None:
        ring_doc = record_op.RING.stats()
        sizes = [int(s) for s in effective.get("cudagraph_capture_sizes") or []]
        ring_doc["cudagraph_capture_sizes_max"] = max(sizes) if sizes else None
        log(f"device ring after build: {ring_doc}")
        if args.cudagraph == "production":
            if sizes and max(sizes) > record_op.RING.rows:
                raise SystemExit(f"ring rows {record_op.RING.rows} < largest capture size {max(sizes)}; increase --ring-rows")
            if record_op.RING.refused_capture_alloc:
                raise SystemExit(f"record op asked to allocate a slot during CUDA-graph capture: {record_op.RING.refused_capture_alloc[:8]}")
            if record_op.RING.capture_calls == 0:
                raise SystemExit("no record op call ran while a stream was capturing: the graphs do not contain the record copies")
    if rpass.calls == 0 or not table.entries:
        raise SystemExit("the record pass did not run (compile-cache hit or hook not installed): no id table, refusing to continue")
    table.write(out / "fx_ids.json")
    if args.mode == "inspect":
        (out / "inspect.json").write_text(json.dumps({"effective_execution": effective, "engine_kwargs": kw, "pass_calls": rpass.calls, "pieces": len(table.pieces),
                                                      "ids": len(table.entries), "build_s": build_s, "env": env}, indent=1, default=str) + "\n")

    hooks = StepHooks(runner, int(args.arena_gb * (1 << 30)), prompt_lens, hash_records=not args.no_hash_records and args.mode != "inspect")
    cap = None
    writer = None
    if args.mode in ("commit", "reference"):
        from verity_vllm.capture import RunWriter
        from verity_vllm_adapter.compiled_fx.capture import CompiledCapture

        if args.mode == "commit":
            writer = RunWriter(out, prod_manifest)
            if writer.profile_id != profile_id:
                raise SystemExit(f"profile_id mismatch: writer {writer.profile_id} vs production {profile_id}")
        cap = CompiledCapture(runner=runner, model=model, profile_id=profile_id, run_dir=out, writer=writer, table=table, prompt_lens=prompt_lens, log=log,
                              cache_dir=Path(env["TORCHINDUCTOR_CACHE_DIR"]))
        hooks.on_step = cap.on_step

    # --- warm-up (record ops run but the sink is disabled: counts only) ---
    rng = np.random.default_rng(20260907)
    vocab = int(getattr(model.config, "vocab_size", 32000))
    warm = [Req({"request_id": f"warmup-{i}", "prompt_token_ids": rng.integers(3, vocab - 1, size=64).tolist(),
                 "output_token_ids": rng.integers(3, vocab - 1, size=8).tolist(), "sampling_params": {}, "finish_reason": None}) for i in range(2)]
    _, warm_timing = run_requests(llm, warm, lambda r: replay_params(r, args.logprobs))
    log(f"warm-up {warm_timing['wall_s']:.2f}s; record calls while disabled {record_op.SINK.calls_while_disabled}")
    calls_before = record_op.SINK.calls_total

    hooks.install()
    if cap is not None:
        cap.begin()
    torch.cuda.synchronize()
    t_run0 = time.perf_counter()
    outs, timing = run_requests(llm, reqs, lambda r: replay_params(r, args.logprobs))
    torch.cuda.synchronize()
    run_s = time.perf_counter() - t_run0
    per_request = compare_with_production(reqs, outs, hooks.unforced)
    followed = sum(1 for p in per_request if p["trajectory_followed"])
    lp_eq = [p["logprobs_equal"] for p in per_request if p["logprobs_equal"] is not None]
    unf_agree = sum(p["unforced_sampler_agrees"] for p in per_request)
    unf_total = sum(p["output_len"] for p in per_request)
    log(f"run {run_s:.1f}s: {len(hooks.steps)} steps, followed {followed}/{len(reqs)}, logprobs equal {sum(lp_eq)}/{len(lp_eq)}, unforced agree {unf_agree}/{unf_total}")

    if args.mode != "inspect":
        with (out / "record_digests.jsonl").open("w") as f:
            for row in hooks.digest_rows:
                f.write(json.dumps(row) + "\n")

    ids_seen: dict[int, int] = {}
    piece_to_template: dict[int, set[int]] = {}
    for row in hooks.digest_rows:
        ids_seen[row["id"]] = ids_seen.get(row["id"], 0) + 1
        piece_to_template.setdefault(row["piece"], set()).add(row["id"] // fx_pass.ID_STRIDE)
    calls_per_step = [s["record_calls"] for s in hooks.steps]
    record_stats = {
        "steps": len(hooks.steps), "record_calls_total": record_op.SINK.calls_total - calls_before, "calls_per_step_min": min(calls_per_step) if calls_per_step else 0,
        "calls_per_step_max": max(calls_per_step) if calls_per_step else 0, "table_ids": len(table.entries), "distinct_ids_seen": len(ids_seen),
        "ids_in_table_never_seen": sorted(set(table.entries) - set(ids_seen))[:50], "ids_seen_not_in_table": sorted(set(ids_seen) - set(table.entries))[:50],
        "duplicate_id_steps": sum(1 for s in hooks.steps if s["duplicate_ids"]), "arena_overflow_allocs": hooks.arena.overflow_allocs,
        "bytes_total": int(sum(s["bytes"] for s in hooks.steps)), "pieces_compiled_with_pass": len(table.pieces), "pass_calls": rpass.calls,
        "runtime_piece_to_template": {str(k): sorted(v) for k, v in sorted(piece_to_template.items())},
        "pieces_with_ambiguous_template": [k for k, v in piece_to_template.items() if len(v) != 1],
        "record_impl": args.record_impl, "cudagraph_arg": args.cudagraph,
        "ring_after_build": ring_doc, "ring_final": None if record_op.RING is None else record_op.RING.stats(),
        "ring_drained_per_step_min": min((s.get("ring_drained", 0) for s in hooks.steps), default=0),
        "ring_host_fallback_steps": sum(1 for s in hooks.steps if s.get("ring_host_fallback")),
    }

    kernel_identity = None
    if args.production_inductor_cache:
        kernel_identity = kernels.compare(Path(args.production_inductor_cache), Path(env["TORCHINDUCTOR_CACHE_DIR"]))
        kernel_identity["record_attribution"] = {str(k): v for k, v in sorted(kernels.record_attribution(Path(env["TORCHINDUCTOR_CACHE_DIR"])).items())}
        (out / "kernel_identity.json").write_text(json.dumps(kernel_identity, indent=1) + "\n")
        log(f"kernel identity: identical={kernel_identity['kernels_identical']} prod {kernel_identity['production']} replay {kernel_identity['replay']} "
            f"only_prod {len(kernel_identity['only_in_production'])} only_replay {len(kernel_identity['only_in_replay'])}")

    commit_doc: dict[str, Any] = {}
    if cap is not None:
        commit_doc = cap.finish(reqs, cp, args.weights_values, kernel_identity)
        if writer is not None:
            if commit_doc.get("boundary_problems"):
                log(f"BOUNDARY_PROBLEMS {commit_doc['boundary_problems'][:3]}; no commitments built")
                raise SystemExit(3)
            from verity_vllm import commit as vcommit

            t_c0 = time.perf_counter()
            commitments = vcommit.build_commitments(out, recompute_leaves=True, write=True)
            commit_doc["commitments"] = {"seconds": time.perf_counter() - t_c0, "trees": len(commitments.get("trees", {})),
                                         "sha256": hashlib.sha256((out / "commitments.json").read_bytes()).hexdigest()}
            log(f"commitments: {commit_doc['commitments']}")

    reuse = buffer_reuse_test()
    doc = {
        "case": args.case, "mode": args.mode, "profile_id": profile_id, "adapter_revision": args.adapter_revision, "adapter_module": ADAPTER_MODULE, "argv": sys.argv,
        "engine_kwargs": kw, "production_engine_kwargs": prod_manifest.get("engine_kwargs"), "effective_execution": effective, "env": env,
        "pass": {"uuid": rpass.uuid(), "roles": list(roles), "hooks": hooks_installed, "policy": rpass.policy, "placement": rpass.placement,
                 "record_lowering": lowering_kind, "record_lowering_stats": dict(record_op.LOWERING_STATS)},
        "build_s": build_s, "run_s": run_s, "timing": {**timing, **hooks.timing, "run_s": run_s}, "warmup": warm_timing,
        # Table 4 fields the verifier reads from replay.json (the eager adapter's names)
        "capture": {"bytes_captured": int(sum(s["bytes"] for s in hooks.steps)), "record_calls": int(sum(s.get("record_calls", 0) for s in hooks.steps))},
        "agreement": {"requests": len(reqs), "trajectory_followed": followed, "logprobs_equal": sum(lp_eq), "logprobs_compared": len(lp_eq),
                      "unforced_sampler_agrees": unf_agree, "unforced_sampler_positions": sum(p["unforced_sampler_positions"] for p in per_request), "output_tokens": unf_total},
        "per_request": per_request, "record_stats": record_stats, "steps": hooks.steps, "kernel_identity_summary": None if kernel_identity is None else
        {k: kernel_identity[k] for k in ("kernels_identical", "production", "replay", "common_distinct_kernels", "replay_record_ops_in_wrappers")} | {
            "only_in_production": [k["kernel_families"] for k in kernel_identity["only_in_production"]], "only_in_replay": [k["kernel_families"] for k in kernel_identity["only_in_replay"]]},
        "buffer_reuse_test": reuse, "errors": hooks.errors, "commit": commit_doc,
    }
    doc["replay_restrictions"] = restrictions_for(args.cudagraph, args.record_impl)
    impl_txt = "pinned-host copies per call" if args.record_impl == "host" else "device ring-buffer copies captured in the graphs, drained per step"
    doc["instrumentation_mode"] = (f"compiled-fx record pass (verity::record inside the Inductor graph, cudagraph_mode={effective.get('cudagraph_mode')}, "
                                   f"{impl_txt}) + eager-tail hooks for lm_head/sampler")
    (out / "replay_compiled.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    (out / "replay.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")  # the verifier reads replay.json (replay_restrictions)
    if writer is None:
        (out / "manifest.json").write_bytes(prof.canonical_json_bytes(prod_manifest))
    log("done")


def run_tp(args: argparse.Namespace, *, out: Path, log: Any, env: dict[str, Any], cp: dict[str, Any], prod_manifest: dict[str, Any], profile_id: str,
           reqs: list[Req], prompt_lens: dict[str, int], tp: int) -> None:
    """[COMP, CX1] compiled-fx replay of a tensor-parallel production: the model lives in vLLM's worker processes, so the
    record pass, the step hooks and the per-rank ``CompiledCapture`` run there (``compiled_fx/tp_worker.py``,
    ``worker_extension_cls``; the pass options travel in the environment so the extension installs them at import,
    before the piecewise compile). The driver only schedules, then joins the ranks (``tp_merge``) and commits."""
    from verity_vllm_adapter import tp_merge

    if args.mode == "inspect":
        raise SystemExit("inspect mode under TP: run with --mode record (fx_ids.json is written per rank)")
    # The extension's install runs at *import* in the worker process. vLLM forks the workers from this process, and a forked
    # child inherits sys.modules: if the driver had imported tp_worker, the workers would find it already imported and the
    # import-time hook would never run there (the first ref1 attempt failed exactly so: pass_calls 0 on both ranks).
    if TP_WORKER_MODULE in sys.modules:
        raise SystemExit(f"{TP_WORKER_MODULE} is imported in the driver; the worker-side import hook would not fire in forked workers")
    roles = tuple(x for x in args.roles.split(",") if x)
    os.environ[TP_WORKER_ENV_KEY] = json.dumps({"roles": list(roles), "placement": args.placement, "lowering": args.lowering, "dry_run": False, "dump_graphs": False,
                                                "record_impl": args.record_impl, "ring_rows": args.ring_rows, "cache_base": env["TORCHINDUCTOR_CACHE_DIR"]})
    os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "14400")  # the per-rank capture writes inside execute_model (as the eager TP replay)
    from vllm import LLM  # noqa: E402

    kw = compiled_engine_kwargs(cp, prod_manifest, args.cudagraph, tensor_parallel_size=tp)
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    effective = prof.effective_compilation(llm)
    cc = llm.llm_engine.vllm_config.compilation_config
    effective.update({"splitting_ops": list(getattr(cc, "splitting_ops", []) or []), "use_inductor_graph_partition": bool(getattr(cc, "use_inductor_graph_partition", False)),
                      "compile_sizes": list(getattr(cc, "compile_sizes", []) or []), "backend": str(getattr(cc, "backend", "")), "pass_config": str(getattr(cc, "pass_config", "")),
                      "cudagraph_capture_sizes": list(getattr(cc, "cudagraph_capture_sizes", []) or []),
                      "max_cudagraph_capture_size": getattr(cc, "max_cudagraph_capture_size", None)})
    states = llm.collective_rpc("verity_cfx_state")
    log(f"engine built in {build_s:.1f}s; effective {effective}; tensor_parallel_size={tp}; per-rank pass state {states}")
    bad = [s for s in states if not s["installed"] or s["pass_calls"] == 0 or s["ids"] == 0]
    if bad:
        raise SystemExit(f"the record pass did not run on every rank (compile-cache hit or hook not installed): {bad}")
    if len({s["inductor_cache_dir_effective"] for s in states}) != len(states):
        raise SystemExit(f"the ranks share an Inductor cache directory (cross-rank FX-graph cache hits possible): {[s['inductor_cache_dir_effective'] for s in states]}")
    hf_config = llm.llm_engine.vllm_config.model_config.hf_config

    # --- warm-up (record ops run but every rank's sink is disabled) ---
    rng = np.random.default_rng(20260907)
    vocab = int(getattr(hf_config, "vocab_size", 32000))
    warm = [Req({"request_id": f"warmup-{i}", "prompt_token_ids": rng.integers(3, vocab - 1, size=64).tolist(),
                 "output_token_ids": rng.integers(3, vocab - 1, size=8).tolist(), "sampling_params": {}, "finish_reason": None}) for i in range(2)]
    _, warm_timing = run_requests(llm, warm, lambda r: replay_params(r, args.logprobs))
    log(f"warm-up {warm_timing['wall_s']:.2f}s")

    install = llm.collective_rpc("verity_cfx_install", kwargs={
        "run_dir": str(out), "profile_id": profile_id, "manifest": prod_manifest, "mode": args.mode, "prompt_lens": prompt_lens,
        "arena_bytes": int(args.arena_gb * (1 << 30)), "hash_records": not args.no_hash_records})
    log(f"per-rank compiled-fx capture installed on {len(install)} ranks: {install}")
    armed = llm.collective_rpc("verity_cfx_arm")
    log(f"armed: {armed}")
    for a in armed:
        if a.get("errors"):
            log(f"rank {a['rank']} capture errors at arm: {a['errors']}")
    torch.cuda.synchronize()
    t_run0 = time.perf_counter()
    outs, timing = run_requests(llm, reqs, lambda r: replay_params(r, args.logprobs))
    torch.cuda.synchronize()
    run_s = time.perf_counter() - t_run0
    synced = llm.collective_rpc("verity_cfx_sync")
    log(f"run {run_s:.1f}s; per rank: {synced}")

    requests_info = {r.request_id: {"prompt_token_ids": r.prompt_token_ids, "output_token_ids": r.output_token_ids, "sampling_params": r.sampling_params,
                                    "finish_reason": r.finish_reason, "chosen_logprobs": r.chosen_logprobs} for r in reqs}
    fin = llm.collective_rpc("verity_cfx_finish", kwargs={"requests_info": requests_info, "checkpoint": cp, "weights_values": args.weights_values,
                                                          "production_inductor_cache": args.production_inductor_cache})
    fin = sorted(fin, key=lambda d: d["rank"])
    unforced = {rid: {int(t): tok for t, tok in d.items()} for rid, d in fin[0]["unforced"].items()}
    per_request = compare_with_production(reqs, outs, unforced)
    followed = sum(1 for p in per_request if p["trajectory_followed"])
    lp_eq = [p["logprobs_equal"] for p in per_request if p["logprobs_equal"] is not None]
    unf_agree = sum(p["unforced_sampler_agrees"] for p in per_request)
    unf_total = sum(p["output_len"] for p in per_request)
    log(f"followed {followed}/{len(reqs)}, logprobs equal {sum(lp_eq)}/{len(lp_eq)}, unforced agree {unf_agree}/{unf_total}")
    for rk in fin:
        if rk.get("boundary_problems"):
            log(f"BOUNDARY_PROBLEMS rank {rk['rank']}: {rk['boundary_problems'][:3]}; no commitments built")
            raise SystemExit(3)
    # every rank's unforced sampler record must agree (the same gathered logits on every rank)
    unforced_ranks_equal = all(rk["unforced"] == fin[0]["unforced"] for rk in fin)

    tp_merge_stats = None
    commit_doc: dict[str, Any] = {"per_rank": [rk["commit"] for rk in fin]}
    if args.mode in ("commit", "reference"):
        tp_merge_stats = tp_merge.merge_ranks(out, store_values=(args.mode == "commit"), log=log)
        if args.mode == "commit":
            from verity_vllm import commit as vcommit

            t_c0 = time.perf_counter()
            commitments = vcommit.build_commitments(out, recompute_leaves=True, write=True)
            commit_doc["commitments"] = {"seconds": time.perf_counter() - t_c0, "trees": len(commitments.get("trees", {})),
                                         "sha256": hashlib.sha256((out / "commitments.json").read_bytes()).hexdigest()}
            log(f"commitments: {commit_doc['commitments']}")
    if not (out / "manifest.json").exists():
        (out / "manifest.json").write_bytes(prof.canonical_json_bytes(prod_manifest))

    reuse = buffer_reuse_test()
    doc = {
        "case": args.case, "mode": args.mode, "profile_id": profile_id, "adapter_revision": args.adapter_revision, "adapter_module": ADAPTER_MODULE, "argv": sys.argv,
        "tensor_parallel_size": tp, "engine_kwargs": kw, "production_engine_kwargs": prod_manifest.get("engine_kwargs"), "effective_execution": effective, "env": env,
        "pass": {"per_rank": [rk["pass"] for rk in fin], "roles": list(roles), "policy": "materialised", "placement": args.placement, "record_lowering": args.lowering,
                 "installed_at": "worker import (compiled_fx.tp_worker, env VERITY_COMPILED_FX_TP)"},
        "build_s": build_s, "run_s": run_s, "timing": {**timing, "run_s": run_s, "per_rank": {str(rk["rank"]): rk["timing"] for rk in fin}}, "warmup": warm_timing,
        "capture": {"bytes_captured": int(sum(s["bytes"] for rk in fin for s in rk["steps"])), "record_calls": int(sum(s.get("record_calls", 0) for rk in fin for s in rk["steps"]))},
        "agreement": {"requests": len(reqs), "trajectory_followed": followed, "logprobs_equal": sum(lp_eq), "logprobs_compared": len(lp_eq),
                      "unforced_sampler_agrees": unf_agree, "unforced_sampler_positions": sum(p["unforced_sampler_positions"] for p in per_request), "output_tokens": unf_total,
                      "unforced_ranks_equal": unforced_ranks_equal},
        "per_request": per_request, "record_stats": {str(rk["rank"]): rk["record_stats"] for rk in fin}, "steps": {str(rk["rank"]): rk["steps"] for rk in fin},
        "kernel_identity_summary": {str(rk["rank"]): rk["kernel_identity_summary"] for rk in fin},
        "kernel_identity_error": {str(rk["rank"]): rk["kernel_identity_error"] for rk in fin if rk.get("kernel_identity_error")},
        "tensor_parallel": {"world_size": tp, "install": install, "armed": armed, "states": states, "merge": tp_merge_stats,
                            "inductor_cache_dirs": {str(rk["rank"]): rk["inductor_cache_dir"] for rk in fin},
                            "record_digests_sha256": {str(rk["rank"]): rk["record_digests_sha256"] for rk in fin}},
        "buffer_reuse_test": reuse, "errors": {str(rk["rank"]): rk["errors"] for rk in fin}, "commit": commit_doc,
    }
    doc["replay_restrictions"] = restrictions_for(args.cudagraph, args.record_impl)
    doc["replay_restrictions"]["tensor_parallel"] = (f"TP={tp}: the record pass and the compiled-fx capture run in every worker process (compiled_fx.tp_worker, "
                                                     f"worker_extension_cls); each rank writes rank<r>/ with s=<r> on every id; tp_merge joins the ranks")
    impl_txt = "pinned-host copies per call" if args.record_impl == "host" else "device ring-buffer copies captured in the graphs, drained per step"
    doc["instrumentation_mode"] = (f"compiled-fx record pass per rank (verity::record inside the Inductor graph, cudagraph_mode={effective.get('cudagraph_mode')}, "
                                   f"{impl_txt}) + eager-tail hooks for lm_head/sampler")
    (out / "replay_compiled.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    (out / "replay.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    (out / "sampler_unforced.json").write_text(json.dumps({rid: {str(t): tok for t, tok in d.items()} for rid, d in unforced.items()}, indent=0) + "\n")
    log("done")


if __name__ == "__main__":
    main()
