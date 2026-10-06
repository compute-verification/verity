"""M1: capture the raw observation log of the frozen workload under the observer (DESIGN §3, §10).

    HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONPATH=. VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 /workspace/venv-cu129/bin/python -m \
    verity_vllm.pipeline.m1_capture --workload workloads/workload_32x16_1req.json --out /vol/cp/l1/m1 \
    [--snapshot-steps 0,1,8] [--no-observer] [--max-tokens 16]

Writes DIR/log.jsonl.gz (the raw observation log), DIR/tokens.json (prompt ids, generated ids, per-step sampled
ids, the accepted record's ids for comparison), DIR/versions.json, DIR/stats.json, DIR/cos_sin_cache.npy (the rotary
module's GPU-computed bf16 table as uint16 words, the `cos_sin` weights field for compare.py) and, with --snapshot-steps,
DIR/values/<step>/*.bin + DIR/snapshots.json.  `--no-observer` runs the identical workload without installing the
observer (the non-interference control) and writes tokens.json / versions.json only.

[M6] `--case B1` (or a workload file carrying `"case"`) serves Qwen2.5-1.5B; the accepted tokens come from the workload
file's `accepted_token_ids` (see `vllm_adapter.load_workload`).

PROTOCOL SUMMARY (Match stage, GPU half: the capture entry -- one engine, one arm per process; writes the execution evidence)
  Purpose: serve the row's declared workload on a fresh vLLM engine and record what the engine did: with the observer installed
    (TorchDispatchMode + Triton launch + module hooks) the raw observation log -- execution evidence -- the fold turns into the Captured_v1
    Program (one Call per observed kernel); with
    --no-observer the same workload uninstrumented (the `control` arm, whose tokens must equal the capture's).
  Inputs: --workload wl.json, --checkpoints manifests/checkpoints.json, --case ROLE, --max-num-seqs B, --snapshot-steps,
    --snapshot-max-bytes[-op], --engine-arg k=v, --target (TargetProfile); env VERITY_TIMELINE, VLLM_BATCH_INVARIANT=1.
  Outputs (--out DIR = match/capture or match/control): log.jsonl.gz, tokens.json (per request: prompt / generated ids, finish_reason,
    served_output_len, executed_prefix_len, cap_exhausted; `lag` = the engine's resolved async_scheduling), spec_tokens.json
    (known BEFORE execution), values/<step>/*.bin + snapshots.json, stats.json, versions.json, cos_sin_cache.npy.
  Invariants: the two lengths stay distinct per request (executed = served + LAG only on an EOS stop before the cap) and are functions
    of the served tokens + the attested lag, never of the trace; LAG is None (unattested) when the engine does not report it -- no default;
    the warm-up runs on throwaway prompts and is never part of the record; nothing observed from the run goes into spec_tokens.json.
  Protocol spec: Notion -> "vLLM integration — protocol spec (Build · Match · Commit)".
"""

from __future__ import annotations

import time

_T_PROCESS0 = time.time()      # [R16 sys] process start (before the torch / vLLM imports)

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from verity_vllm.config import option
from verity_vllm.engine import engine_profile as prof  # noqa: F401  (apply_env before vllm import)

prof.apply_env()

from verity_vllm.engine import vllm_adapter as va  # noqa: E402
from verity_vllm.pipeline.timeline import Timeline  # noqa: E402  (torch-free; VERITY_TIMELINE -> JSONL, else no-op)

ACCEPTED_TOKENS = va.ACCEPTED_TOKENS   # the B0 request's accepted tokens (kept for the older importers); prefer workload["accepted"]


def dump_cos_sin(engine: "va.Engine", out: Path) -> Path | None:
    """`model.layers.0.self_attn.rotary_emb.cos_sin_cache` (bf16 [MAX_POS, D]) as uint16 words -> DIR/cos_sin_cache.npy.  The
    table is a derived weight computed on the GPU (vLLM `_compute_cos_sin_cache`, CUDA cosf/sinf); compare.py reads the
    `cos_sin` field from this file."""
    import numpy as np
    import torch

    from verity_vllm.check.root_policy import analytic_table_of_buffer   # the one by-name recognition of the two analytic-table buffers

    buf = None
    for name, b in engine.model.named_buffers():
        if analytic_table_of_buffer(name) == "cos_sin":
            buf = b
            break
    if buf is None:
        return None
    dtype = str(buf.dtype).removeprefix("torch.")
    if buf.dtype != torch.bfloat16:
        # [r8 E-cleanup, fp16 (d) `cos_sin_cache` root] a served-fp16 model's table is float16: it used to be silently NOT observed
        # (root_policy then fails "no captured cos_sin_cache.npy", which reads as a capture defect).  Observe it in its own dtype
        # under a dtype-qualified name -- never as `cos_sin_cache.npy`, whose readers (compare / replay / root_policy's bf16
        # analytic check) take bf16 words -- and label it; the bf16 path below is unchanged.  The frozen validator still has no
        # fp16 analytic table.
        if buf.dtype != torch.float16:
            return None
        words = buf.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        path = out / f"cos_sin_cache.{dtype}.npy"
        np.save(path, words)
        (out / "cos_sin_cache.json").write_text(json.dumps({"dtype": dtype, "shape": list(buf.shape), "file": path.name,
                                                             "words": "uint16 (raw IEEE binary16 words)", "buffer": name}, indent=1) + "\n")
        return path
    words = buf.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    path = out / "cos_sin_cache.npy"
    np.save(path, words)
    (out / "cos_sin_cache.json").write_text(json.dumps({"dtype": dtype, "shape": list(buf.shape), "file": path.name,
                                                         "words": "uint16 (raw bfloat16 words)", "buffer": name}, indent=1) + "\n")
    return path


def dump_normalizer(engine: "va.Engine", out: Path) -> Path | None:
    """[ADAPTER lane cov-dense 2026-09-17] Gemma-2's `model.normalizer` (0-d bf16 buffer `torch.tensor(hidden_size ** 0.5)`,
    no checkpoint tensor) as its one uint16 word -> DIR/normalizer.npy, the same observation `dump_cos_sin` makes of the other
    analytic-table root: root_policy (`analytic.check_normalizer`) compares the runtime's word with bf16(sqrt(H)).  Models
    without the buffer (Llama, Qwen, Phi-3, ...) write nothing."""
    import numpy as np
    import torch

    from verity_vllm.check.root_policy import analytic_table_of_buffer

    for name, b in engine.model.named_buffers():
        if analytic_table_of_buffer(name) == "normalizer":
            if b.dtype != torch.bfloat16 or b.dim() != 0:
                return None
            words = b.detach().reshape(1).contiguous().view(torch.int16).cpu().numpy().view(np.uint16).reshape(())
            path = out / "normalizer.npy"
            np.save(path, words)
            return path
    return None


def parse_steps(s: str | None) -> frozenset[int]:
    if not s:
        return frozenset()
    return frozenset(int(x) for x in s.split(",") if x.strip() != "")


@dataclass(frozen=True, kw_only=True)
class Options:
    workload: str = option("--workload", default="workloads/workload_32x16_1req.json")
    out: str = option("--out")
    checkpoints: str = option("--checkpoints", default="manifests/checkpoints.json")
    snapshot_steps: str | None = option("--snapshot-steps", default=None,
                                        help="comma-separated engine steps to snapshot, e.g. 0,1,8")
    snapshot_max_bytes: int = option("--snapshot-max-bytes", type=int, default=32 << 20)
    snapshot_max_bytes_op: int | None = option("--snapshot-max-bytes-op", type=int, default=None,
                                               help="lower per-tensor cap at dispatcher-call events (aten / custom ops); Triton launch events keep --snapshot-max-bytes "
                                               "(a large launch output is copied once at the launch, not at every view that touches it)")
    no_observer: bool = option("--no-observer", default=False, action="store_true",
                               help="non-interference control: same workload, no observer")
    no_reenter: bool = option("--no-reenter", default=False, action="store_true")
    no_triton: bool = option("--no-triton", default=False, action="store_true")
    no_module_hooks: bool = option("--no-module-hooks", default=False, action="store_true")
    max_tokens: int | None = option("--max-tokens", type=int, default=None)
    no_warmup: bool = option("--no-warmup", default=False, action="store_true",
                             help="skip the uninstrumented throwaway warm-up pass")
    adapter_revision: str | None = option("--adapter-revision", default=None)
    wheel_sha256: str = option("--wheel-sha256", default="7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3")
    vllm_commit: str = option("--vllm-commit", default="d9105ea80")
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    case: str | None = option("--case", default=None,
                              help="checkpoint role (B0 SmolLM2-135M, B1 Qwen2.5-1.5B); default: the workload file's, else B0")
    verify_weights: bool = option("--verify-weights", default=False, action="store_true",
                                  help="sha256-verify the local checkpoint files against the manifest before loading (root policy: weight identity)")
    engine_arg: list[str] = option("--engine-arg", action="append", default=[],
                                   help="LLM(**kw) override key=value (JSON value), repeatable; e.g. max_num_batched_tokens=64")
    target: str | None = option("--target", default=None,
                                help="[R13 hopper] TargetProfile JSON overrides / path the engine is built for (default: the workload's `target` block over this device)")
    acquisition_manifest: str | None = option("--acquisition-manifest", default=None,
                                              help="[v2 acquire, Phase 4a] the row's required-value manifest: the plan's hooks are installed beside "
                                              "the observer (occurrence counts only: DIR/acquisition_observed.json) and DIR/runtime_tree.json, acquisition_plan.json are "
                                              "written (default: $ACQUIRE_MANIFEST)")


def adapter_revision(args: Options) -> str:
    """`--adapter-revision`, else the exported tree's commit (EXPORT.json), else "unknown"."""
    if args.adapter_revision is not None:
        return args.adapter_revision
    try:
        return json.loads(Path("EXPORT.json").read_text()).get("commit", "unknown")
    except (OSError, ValueError, AttributeError):
        return "unknown"


def capture_protocols(wl: dict, args: Options, model: object) -> tuple[object | None, list[str]]:
    """The target's executors (PoUW) and their entries. Installed after the warm-up as in the Commit, they serve both arms, so the
    observed run executes what the Commit serves and records each executed call as its executor's own op (`protocol_options.at_capture`)."""
    from verity_vllm import protocol_options as PO

    target = va.target_of_workload(wl, args.target)
    protocols = PO.at_capture(target, model, PO.own_auditor(wl.get("protocols_config"), target))
    return protocols, list(protocols.entries) if protocols is not None else []


def main(args: Options) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(args.workload, args.max_tokens)
    revision = adapter_revision(args)

    case = va.case_of(wl, args.case)
    # [R16 sys, BRIEF §5/§10] stage telemetry of this process (m1_capture = one engine per arm: `capture` / `control` are separate processes)
    tl = Timeline.from_env("m1_capture", hot=False, arm=("control" if args.no_observer else "capture"))
    tl.event("startup.imports", "end", s=round(time.time() - _T_PROCESS0, 3))
    tl.start("engine.build", case=case, max_num_seqs=args.max_num_seqs)
    eng = va.build_engine(args.checkpoints, max_num_seqs=args.max_num_seqs, case=case, verify_weights=args.verify_weights,
                          engine_args=va.parse_engine_args(args.engine_arg), target=va.target_of_workload(wl, args.target),
                          execution=va.execution_of_workload(wl))     # [R15 EXEC-02] the row's execution decides enforce_eager, recorded in the header
    tl.end("engine.build", build_s=round(eng.build_s, 3), model=eng.cp["repo"], enforce_eager=eng.kwargs.get("enforce_eager"),
           engine_reused=False, restart_reason="fresh process (one m1_capture per Match arm)")
    print(f"[m1] engine built in {eng.build_s:.1f}s (case {case}, {eng.cp['repo']}); "
          f"kwargs={ {k: v for k, v in eng.kwargs.items() if k not in ('model', 'tokenizer')} }", flush=True)
    (out / "versions.json").write_text(json.dumps(va.versions_doc(eng), indent=1) + "\n")
    # [D88 artifact model] the specification's concrete workload as known BEFORE any engine step: the prompt token ids the workload
    # loader produced from the checkpoint's tokenizer, per-request output limits, arrival steps, sampling parameters, and the engine
    # kwargs the harness resolved.  Written before the warm-up; nothing observed from the run goes in here (tokens.json is the record).
    (out / "spec_tokens.json").write_text(json.dumps({
        "schema": "verity-gen/spec-tokens/v1", "known_before_execution": True,
        "tokenizer": {"repo": eng.cp["repo"], "revision": eng.cp.get("revision")},
        "requests": [{"request_id": r["request_id"], "prompt_token_ids": r["prompt_token_ids"], "prompt_len": len(r["prompt_token_ids"]),
                      "max_tokens": r["max_tokens"], "arrive_step": int(r.get("arrive_step") or 0)} for r in wl["requests"]],
        "sampling": wl["sampling"], "engine_kwargs": {k: v for k, v in eng.kwargs.items() if k not in ("model", "tokenizer")},
        # [R15 EXEC-02] the execution the row declares and the one the built engine realises (vllm_config read-back), beside the kwargs
        "row_id": wl.get("row_id"), "declared_execution": eng.execution, "effective_execution": eng.effective},
        indent=1, default=str) + "\n")
    cs = dump_cos_sin(eng, out)
    print(f"[m1] cos_sin_cache -> {cs}", flush=True)
    nz = dump_normalizer(eng, out)
    if nz is not None:
        print(f"[m1] normalizer -> {nz}", flush=True)
    # [v2 acquire, Phase 4a] the built engine's runtime tree (always) and, with a required-value manifest, the acquisition plan, written
    # before any run; the plan's hooks go in beside the observer below (counts / shapes only).
    import os

    from verity_vllm.acquire.install import IncompletePlan
    from verity_vllm.acquire.stage import MANIFEST_ENV, plan_at_stage

    man_path = args.acquisition_manifest or os.environ.get(MANIFEST_ENV) or None
    try:
        acq_docs = plan_at_stage(eng.model, getattr(eng, "runner", None), man_path, out, source="m1_capture")
    except IncompletePlan as e:
        print(f"[m1] REFUSE: acquisition plan: {e}", flush=True)
        sys.exit(3)
    for ln in acq_docs.log_lines("m1"):
        print(ln, flush=True)

    if not args.no_warmup:
        warm = va.throwaway_requests(eng, wl["requests"])
        with tl.span("prep.warmup_control"):
            _, tw = va.run_requests(eng, warm, wl["sampling"])
        print(f"[m1] warm-up (uninstrumented, {len(warm)} throwaway requests): {tw['wall_s']:.2f}s, {tw['engine_steps']} steps", flush=True)

    stats = None
    header = None
    protocols, protocol_entries = capture_protocols(wl, args, eng.model)
    if args.no_observer:
        t0 = time.perf_counter()
        with tl.span("work.control_run"):
            tokens, timing = va.run_requests(eng, wl["requests"], wl["sampling"])
        print(f"[m1] control run (no observer): {time.perf_counter() - t0:.2f}s", flush=True)
        per_step = None
    else:
        tl.start("work.capture_run")
        snap = parse_steps(args.snapshot_steps)
        header = va.make_header(eng, wl, adapter_revision=revision, wheel_sha256=args.wheel_sha256, vllm_commit=args.vllm_commit,
                                extra_notes={"argv": sys.argv, "snapshot_steps": sorted(snap), "weights_verified": bool(args.verify_weights),
                                             "protocols": protocol_entries})
        cfg = va.ObserverConfig(mode="record", reenter=not args.no_reenter, triton=not args.no_triton,
                                module_hooks=not args.no_module_hooks, write_log=True, snapshot_steps=snap,
                                snapshot_dir=out if snap else None, snapshot_max_bytes=args.snapshot_max_bytes,
                                snapshot_max_bytes_op=args.snapshot_max_bytes_op)
        cap = va.Capture(eng, header=header, log_path=out / "log.jsonl.gz", config=cfg)
        cap.install()
        print(f"[m1] observer installed: {len(cap.roots)} roots, profile_id={header.profile_id}, "
              f"reenter impls={header.notes.get('reenter_impls')}", flush=True)
        acq_inst, acq_sink = None, None
        if acq_docs.plan is not None:
            from verity_vllm.acquire.install import OccurrenceSink, install_from_plan

            acq_sink = OccurrenceSink()
            acq_inst = install_from_plan(acq_docs.plan, eng.model, getattr(eng, "runner", None), sink=acq_sink)
            print(f"[m1] acquisition plan hooks installed beside the observer: {len(acq_inst.installed)} sites "
                  f"({len(acq_inst.deferred)} deferred to their sources: taps / registered inputs / prescribed)", flush=True)
        t0 = time.perf_counter()
        tokens, timing = va.run_requests(eng, wl["requests"], wl["sampling"])
        run_s = time.perf_counter() - t0
        if acq_inst is not None:
            acq_inst.remove()                                  # before cap.close() restores the runner's sample(): the wraps unwind in order
            obs = acq_sink.to_json()
            obs["plan_digest"] = acq_docs.plan.digest
            obs["installed"] = [list(k) for k in acq_inst.keys()]
            (out / "acquisition_observed.json").write_text(json.dumps(obs, indent=1) + "\n")
            print(f"[m1] acquisition observed: {sum(s['occurrences'] for s in obs['sites'])} occurrences over {len(obs['sites'])} sites -> {out / 'acquisition_observed.json'}", flush=True)
        stats = cap.close()
        tl.end("work.capture_run", run_s=round(run_s, 3), events=stats.get("events_written"), steps=stats.get("steps"))
        per_step = {str(s): ids for s, ids in cap.sample_ids.items()}
        print(f"[m1] observed run {run_s:.2f}s: {stats['events_written']} events, {stats['steps']} steps, "
              f"recorder={stats['recorder']}, storage={stats['storage']}, snapshot_bytes={stats['snapshot_bytes']}, "
              f"errors={len(stats['errors'])}", flush=True)
        if stats["errors"]:
            print("[m1] ERRORS:", json.dumps(stats["errors"][:5], indent=1), flush=True)
        (out / "stats.json").write_text(json.dumps(stats, indent=1, default=str) + "\n")
        try:
            sz = (out / "log.jsonl.gz").stat().st_size
            print(f"[m1] log size {sz / 1e6:.1f} MB", flush=True)
        except OSError:
            pass

    if protocols is not None:
        protocols.close()
    doc = {"requests": [], "per_step_sampled": per_step, "timing": timing, "observer": not args.no_observer, "protocols": protocol_entries,
           "profile_id": header.profile_id if header is not None else None, "case": case, "model": eng.cp["repo"],
           # [coverage D87] the serving profile / context class the workload declares and the realized shape of this run
           "serving_profile": wl.get("serving_profile"), "context_class": wl.get("context_class"),
           "expected_engine_steps": wl.get("expected_engine_steps"), "engine_steps": timing.get("engine_steps")}
    all_equal = True
    per_req = (timing.get("requests") or {})
    # [R17-1 / DP-pad-02 (b)] the record's LAG is the engine's RESOLVED async_scheduling read back after the build -- the SAME fact the RunHeader
    # attests (profile.engine_kwargs.async_scheduling) and the Build's TargetProfile declares; True -> 1 post-EOS row, False -> 0, None ->
    # unattested (no default).  Per request the two lengths are kept distinct: served_output_len = the tokens served (generated_token_ids,
    # the answer up to and including the EOS or the cap); executed_prefix_len = served + LAG on an EOS stop before the cap (the post-EOS row
    # is active internal computation whose sampled token is NOT served).  Both are a function of the served tokens + the recorded fact,
    # never of the trace; GM-01 G6 verifies them against the rows the record actually holds.
    _asched = va.resolved_async_scheduling(eng.llm)
    _lag = None if _asched is None else (1 if _asched else 0)
    doc["lag"] = {"value": _lag, "async_scheduling": _asched,
                  "source": "vllm_adapter.resolved_async_scheduling(engine) after the build (= RunHeader.profile.engine_kwargs.async_scheduling)",
                  "rule": "served_output_len = len(generated_token_ids); executed_prefix_len = served_output_len + LAG iff finish_reason == 'stop' and "
                          "served_output_len < max_tokens, else served_output_len; cap_exhausted = served_output_len == max_tokens (R17-1)"}
    for r in wl["requests"]:
        rid = r["request_id"]
        acc = wl["accepted"].get(rid)
        gen = tokens.get(rid, [])
        eq = (acc is not None and gen == acc[: len(gen)] and len(gen) == len(acc)) if acc is not None else None
        all_equal = all_equal and bool(eq)
        steps_r = per_req.get(rid) or {}
        _served = len(gen)
        _cap = int(r["max_tokens"])
        _eos_stop = (str(steps_r.get("finish_reason")) == "stop") and _served < _cap
        _executed = (None if (_eos_stop and _lag is None) else (_served + _lag if _eos_stop else _served))
        doc["requests"].append({"request_id": rid, "prompt_token_ids": r["prompt_token_ids"], "max_tokens": r["max_tokens"],
                                "generated_token_ids": gen, "accepted_record_token_ids": acc, "equals_accepted": eq,
                                # realized lengths and schedule (D87): every run artifact records them
                                "prompt_len": len(r["prompt_token_ids"]), "n_generated": len(gen),
                                "arrive_step": int(r.get("arrive_step") or 0), "arrive_step_realized": steps_r.get("arrive_step_realized"),
                                "first_step": steps_r.get("first_step"), "finish_step": steps_r.get("finish_step"),
                                "finish_reason": steps_r.get("finish_reason"),
                                # [R17-1] the two lengths, distinct; cap exhaustion beside them; executed None = EOS stop under an unattested lag
                                "served_output_len": _served, "executed_prefix_len": _executed, "cap_exhausted": _served == _cap,
                                "eos_stop_before_cap": _eos_stop, "lag": _lag})
        print(f"[m1] {rid}: prompt_len={len(r['prompt_token_ids'])} arrive={int(r.get('arrive_step') or 0)} "
              f"steps={steps_r.get('first_step')}..{steps_r.get('finish_step')} ({steps_r.get('finish_reason')}) generated {gen}\n"
              f"[m1]   accepted  {acc}  equal={eq}", flush=True)
    doc["all_equal_accepted"] = all_equal
    # [GEN sampling] n > 1: the child sequences vLLM generated (ids f"{index}_{request_id}", the ids the log's StepStart rows carry)
    wl_ids = {r["request_id"] for r in wl["requests"]}
    children = {rid: toks for rid, toks in tokens.items() if rid not in wl_ids}
    if children:
        doc["children"] = [{"request_id": rid, "generated_token_ids": toks} for rid, toks in sorted(children.items())]
    (out / "tokens.json").write_text(json.dumps(doc, indent=1) + "\n")
    tl.event("process.end", "mark", rc=0, s=round(time.time() - _T_PROCESS0, 3))
    print(f"[m1] done -> {out}", flush=True)
