"""vLLM adapter (runtime side): build the engine under the frozen profile κ, and observe the in-process V2
`GPUModelRunner` step by step into the raw observation log.

Engine (DESIGN §1 / §10): `profile.apply_env()`, `cp = checkpoint_for(case, manifests/checkpoints.json)` (case B0 = SmolLM2-135M, B1 = Qwen2.5-1.5B [M6]),
`LLM(**engine_kwargs(cp, replay=False, enforce_eager=True), enable_prefix_caching=False, ...)`,
`runner = get_model_runner(llm)`.  The profile id is `profile_id_of(build_manifest(...))` exactly as the production
recorder builds it (`verity_vllm_adapter.production`); fields that necessarily differ from the accepted record's
manifest (pod id, driver, adapter revision, no generated-kernel identity) are listed in `RunHeader.notes`.

Step observation.  In the pinned wheel the worker calls `runner.execute_model(scheduler_output)` (input preparation
+ model forward; the result stays in `runner.execute_model_state`) and then `runner.sample_tokens(...)`, which
calls `runner.sample(hidden_states, input_batch, grammar_output)` (lm_head GEMM, sampler).  Both instance methods
are wrapped (pattern `capture.py:679-720`); the observer mode is entered separately in each, with phase "forward"
(the runner's own input-preparation dispatcher calls / Triton kernels before the model's root forward carry phase
"other" when module hooks are on) and phase "sample".  Dummy / profile runs and empty steps are passed through.

Log order requires `StepStart` (whose facts are known only after input preparation) before the step's dispatch
events, and `SampleEnd` needs the sampled ids: the step's events are **buffered in memory**; the tiny per-step
buffers the model actually read (`input_ids`, `positions`, the first layer's slot mapping, `logits_indices`, the
gathered block table) and the sampled ids are copied D2H *asynchronously* into pinned host memory right after the
forward / after `sample`, behind a CUDA event, and the step is written out at the start of the next step (or at
close), when the runner itself has long synchronised on its own output copy.  The observer therefore adds no
synchronisation of its own, except `torch.cuda.synchronize()` at the end of a step for which snapshots were
requested.

Roots: every `named_parameters()` / `named_buffers()` entry (`Root(role=weight|buffer)`, incl. the rotary
`cos_sin_cache`); the runner's KV caches and input buffers are registered as `StorageAlloc(origin="external")`
with a `Note` naming them (they are not roots: their contents are produced inside the observed computation).
Prompt rows become `Root(role=prompt_token, view=<one-element row view of input_ids>, values=[id])`; decode rows
become `StepLink(step, row, request_id, src_step, value)` where `src_step` is the step whose `SampleEnd` produced
the token (tracked per request from the sampler outputs).

Snapshots (evidence path, `--snapshot-steps`): after every recorded dispatcher call / Triton launch of a listed
step, every non-root tensor argument / output is copied D2H (async, pinned arena, stream-ordered right after the
call) as the raw bytes of the storage range the view spans; the files are content-addressed (sha256) under
`values/<step>/` with an index in `snapshots.json`.  Tensors larger than `snapshot_max_bytes` whose one dimension
equals the KV cache's block count are snapshotted per (leading index, block-in-use) sub-view; other oversize
tensors are listed in the index as skipped.  `snapshot_max_bytes_op` (optional, lower) is the cap applied at
dispatcher-call events only: a large buffer produced by a Triton launch is then copied once, at the launch (where the
Compute's output snapshot is looked up), and not again at every aten view / custom-op call that touches it (the
pinned copies of one step are the capture's peak host memory; OLMoE's fused-MoE intermediates at a 2048-token
prefill are 64 MiB x ~23 events x 16 layers).  Skipped entries record which cap applied.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .events import (Event, Note, Root, RunEnd, RunHeader, SampleEnd, StepEnd, StepLink, StepStart, StorageAlloc,
                     ViewDesc, to_json)
from .log import LogWriter
from .observer import EmptyMode, ModuleNamer, Observer, Recorder, encode_arg, reenter_impl_name
from .storage import StorageAdapter, dtype_name
from . import triton_adapter

from .arrivals import run_requests_arrivals  # [coverage D87] step-indexed arrivals (torch-free module, unit-tested)
from . import chunk_attribution as _chunks   # [chunked 2026-09-17] prefill-chunk / token-step attribution (torch-free, unit-tested)

SCHEMA = "verity-capture/log/v1"
CASE = "B0"                       # default checkpoint role (SmolLM2-135M); [M6] `build_engine(case="B1")` serves Qwen2.5-1.5B
#: The accepted production record per checkpoint role (its tokens are the reference the runs are compared with).
ACCEPTED: dict[str, dict[str, str]] = {
    "B0": {"profile_id": "88e1b098d719", "record": "out/scale/e8/live/V5-N32x16-1req-bex0-inc-sc1"},
    "B1": {"profile_id": "40f0cebeb670 / c37c3f0b8363", "record": "fixtures/results/B1-40f0cebeb670-20260907T1242Z/production (and B1-c37c3f0b8363-20260907T1328Z; both RTX 4090 eager)"},
}
ACCEPTED_PROFILE_ID = ACCEPTED["B0"]["profile_id"]
ACCEPTED_RECORD = ACCEPTED["B0"]["record"]


# ---------------------------------------------------------------------------------------------------------
# engine under the frozen profile
# ---------------------------------------------------------------------------------------------------------


def load_workload(path: str | Path, max_tokens: int | None = None) -> dict:
    """The workload file (`workloads/workload_32x16_1req.json`, `workload_qwen15_32x16_1req.json`) ->
    {"requests": [...], "sampling": {...}, "case": <checkpoint role or None>, "accepted": {request_id: accepted token ids}}.
    The accepted tokens come from the file's `accepted_token_ids` per request (M6) or `ACCEPTED_TOKENS` of the case."""
    doc = json.loads(Path(path).read_text())
    # [GEN sampling] an optional top-level `sampling` block selects the variant and the request seeds / fan-out:
    # {"variant": "greedy", "seed": null, "n": 1} (defaults = the greedy runs); a request may override `seed` / `n`.
    samp = dict(doc.get("sampling") or {})
    variant = str(samp.get("variant", "greedy"))
    params = (doc.get("variants") or {}).get(variant)
    if params is None:
        if variant != "greedy":
            raise KeyError(f"workload {path}: sampling variant {variant!r} is not in `variants`")
        params = {"temperature": 0.0, "top_p": 1.0, "top_k": 0}
    seed = None if samp.get("seed") is None else int(samp["seed"])
    n = int(samp.get("n", 1))
    reqs = []
    accepted: dict[str, list[int]] = {}
    per_request: list[dict] = []
    for r in doc["requests"]:
        rs = r.get("seed", seed)
        req = {"request_id": r["request_id"], "prompt_token_ids": [int(t) for t in r["prompt_token_ids"]],
               "max_tokens": int(max_tokens if max_tokens is not None else r["max_tokens"]),
               "seed": None if rs is None else int(rs), "n": int(r.get("n", n)),
               # [coverage D87] serving profile B8-mixed: the engine step at which the request is admitted (0 = with the first
               # step, the historical behaviour).  Step-indexed, not wall-clock, so the observer and control runs see the same
               # batch membership at every step.
               "arrive_step": int(r.get("arrive_step") or 0)}
        if r.get("variant") is not None:
            # [ov-sampling] a request may pin ANOTHER variant (mixed batch: one greedy row next to a sampled row); the
            # request record then carries its own temperature / top_p / top_k and the manifest lists them per request
            rp = (doc.get("variants") or {}).get(str(r["variant"]))
            if rp is None:
                raise KeyError(f"workload {path}: request {r['request_id']} variant {r['variant']!r} is not in `variants`")
            req.update({"variant": str(r["variant"]), "temperature": float(rp.get("temperature", 0.0)),
                        "top_p": float(rp.get("top_p", 1.0)), "top_k": int(rp.get("top_k", 0))})
            per_request.append({"request_id": req["request_id"], "variant": req["variant"], "temperature": req["temperature"],
                                "top_p": req["top_p"], "top_k": req["top_k"], "seed": req["seed"]})
        reqs.append(req)
        if r.get("accepted_token_ids") is not None:
            accepted[r["request_id"]] = [int(t) for t in r["accepted_token_ids"]]
    for rid, toks in ACCEPTED_TOKENS.items():
        accepted.setdefault(rid, list(toks))
    temperature = float(params.get("temperature", 0.0))
    sampling = {"temperature": temperature, "top_p": float(params.get("top_p", 1.0)), "top_k": int(params.get("top_k", 0)),
                "logprobs": None, "seed": seed, "n": n, "variant": variant,
                "greedy": temperature == 0.0 and all(p["temperature"] == 0.0 for p in per_request)}
    if per_request:
        sampling["per_request"] = per_request
    return {"requests": reqs, "sampling": sampling, "source": str(path), "model": doc.get("model"), "case": doc.get("case"),
            "accepted": accepted,
            # [coverage D87] declared scheduler knobs the workload cannot run without (e.g. max_num_batched_tokens for an
            # unchunked 8-request prefill); run_config passes them as --engine-arg, an explicit CLI value wins
            "engine_args_required": dict(doc.get("engine_args_required") or {}),
            "serving_profile": doc.get("serving_profile"), "context_class": doc.get("context_class"),
            "expected_engine_steps": doc.get("expected_engine_steps"),
            # [R15 EXEC-02] the sweep row this workload belongs to and the execution it declares (row-id suffix / explicit field): the
            # GPU stages hand `execution` to build_engine so enforce_eager is the row's, never the harness's literal (COORD M-0063 (2))
            "row_id": (doc.get("sweep") or {}).get("row_id") if isinstance(doc.get("sweep"), dict) else doc.get("row_id"),
            "execution": execution_of_workload(doc),
            # [R15 fp8] the row's declared TargetProfile overrides (`target` block, written by sweep/workload.py --target): the channel
            # `target_of_workload` documents ("else the workload manifest's target block") -- rc3 dropped it here, so every GPU stage
            # built its engine with target=None (engine ARGS still travelled through engine_args_required; environment knobs such as
            # VLLM_USE_DEEP_GEMM have no other channel, and the header carried no `profile.target`).  None when the row declares none.
            "target": dict(doc["target"]) if isinstance(doc.get("target"), dict) and doc.get("target") else None,
            "sha256": hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


#: The accepted record's generated tokens for the B0 workload's request (its file predates `accepted_token_ids`):
#: out/scale/e8/live/V5-N32x16-1req-bex0-inc-sc1/adapter/selective/ru/7cc7c3bb…/boundary.json
ACCEPTED_TOKENS: dict[str, list[int]] = {
    "7cc7c3bb-adf5-58cd-87fc-9f1b8e7c34d6": [359, 253, 1123, 1183, 282, 253, 1772, 338, 553, 7527, 288, 11492, 351, 260, 4340, 1357],
}


def case_of(workload: dict | None, case: str | None = None) -> str:
    """The checkpoint role of a run: an explicit `case`, else the workload file's `case`, else `CASE` (B0)."""
    return case or (workload or {}).get("case") or CASE


@dataclass
class Engine:
    llm: Any
    runner: Any
    model: Any
    cp: dict
    kwargs: dict
    env_flags: dict
    build_s: float
    case: str = CASE
    target: Any = None   # [R13 hopper] the declared `TargetProfile` the engine was built under (None = none declared: vLLM's own selection)
    execution: str | None = None   # [R15 EXEC-02] the row's declared execution label the engine was built FOR (bi-eager / bi-compiled / ...), None = undeclared
    effective: dict | None = None  # [R15 EXEC-02] the execution the built engine actually realises (effective_execution_doc), None before LLM()


# [R15 EXEC-02 / COORD M-0063 (2)] the row's execution label decides `enforce_eager`; it is never a literal of the harness.  Labels are the
# row-id vocabulary (`…__bi-eager`, `…__bi-eager-fa2`, `…__bi-compiled`); a label this table does not know fails closed (no guessing).
EXECUTION_ENFORCE_EAGER: dict[str, bool] = {"bi-eager": True, "bi-eager-fa2": True, "eager": True, "bi-compiled": False, "compiled": False}


def execution_of_workload(workload: dict | None) -> str | None:
    """[R15 EXEC-02] The execution label a workload manifest declares: an explicit top-level `execution` (or `sweep.execution`) when the
    schema carries one, else the last `__` token of `sweep.row_id` (`…__greedy__bi-eager` -> 'bi-eager'); None for a workload that is not
    a sweep row (no row id: the caller's default applies and the header records `declared_execution: null`)."""
    if not isinstance(workload, dict):
        return None
    sw = workload.get("sweep") if isinstance(workload.get("sweep"), dict) else {}
    for v in (workload.get("execution"), sw.get("execution")):
        if isinstance(v, str) and v:
            return v
    rid = sw.get("row_id") or workload.get("row_id")
    if isinstance(rid, str) and "__" in rid:
        return rid.rsplit("__", 1)[1]
    return None


def execution_label_of(enforce_eager: bool | None) -> str | None:
    """The row-id execution label the EFFECTIVE engine realises: eager -> 'bi-eager', compiled -> 'bi-compiled' (None when unknown)."""
    if enforce_eager is None:
        return None
    return "bi-eager" if bool(enforce_eager) else "bi-compiled"


def effective_execution_doc(llm: Any, model: Any = None) -> dict:
    """[R15 EXEC-02] What the built engine actually executes, read back from `llm.llm_engine.vllm_config` (not from the kwargs the harness
    asked for): eager/compiled (`enforce_eager`, compilation mode, cudagraph mode, custom ops, optimisation level), the KV/scheduler facts
    that condition a Program's applicability (prefix caching, chunked prefill, max_num_batched_tokens, max_num_seqs, max_model_len), dtype,
    and the attention implementation the layers carry.  `execution_label` is derived from the effective `enforce_eager` so the finalizer can
    compare it with the row id / the Program's declared applicability (COORD M-0063 (3)).  Every field is read; unreadable ones are None."""
    doc: dict[str, Any] = {}
    try:
        from verity_vllm_adapter import profile as prof
        doc.update(prof.effective_compilation(llm))
    except Exception as e:  # noqa: BLE001
        doc["compilation_error"] = repr(e)[:200]
    try:
        vc = llm.llm_engine.vllm_config
        cc, sc, mc = vc.cache_config, vc.scheduler_config, vc.model_config
        doc["enable_prefix_caching"] = bool(getattr(cc, "enable_prefix_caching", None)) if getattr(cc, "enable_prefix_caching", None) is not None else None
        doc["enable_chunked_prefill"] = bool(getattr(sc, "enable_chunked_prefill", None)) if getattr(sc, "enable_chunked_prefill", None) is not None else None
        doc["max_num_batched_tokens"] = int(getattr(sc, "max_num_batched_tokens")) if getattr(sc, "max_num_batched_tokens", None) is not None else None
        doc["max_num_seqs"] = int(getattr(sc, "max_num_seqs")) if getattr(sc, "max_num_seqs", None) is not None else None
        doc["max_model_len"] = int(getattr(mc, "max_model_len")) if getattr(mc, "max_model_len", None) is not None else None
        doc["dtype"] = str(getattr(mc, "dtype", None))
        doc["block_size"] = int(getattr(cc, "block_size")) if getattr(cc, "block_size", None) is not None else None
        doc["gpu_memory_utilization"] = float(getattr(cc, "gpu_memory_utilization")) if getattr(cc, "gpu_memory_utilization", None) is not None else None
        doc["async_scheduling"] = resolved_async_scheduling(llm)      # [R16 sys int-3] resolved, not the CLI default (M-0213)
    except Exception as e:  # noqa: BLE001
        doc["config_error"] = repr(e)[:200]
    if model is not None:
        try:
            doc["flash_attn_version"] = observed_flash_attn_version(model)
        except Exception:  # noqa: BLE001
            doc["flash_attn_version"] = None
    doc["execution_label"] = execution_label_of(doc.get("enforce_eager"))
    return doc


def parse_target_arg(s: str | None):
    """[R13 hopper] `--target` (JSON overrides / path, as `derive_step`'s) -> `TargetProfile`, or None when absent.  The runtime side of
    the ONE target source: `build_engine` applies `TargetProfile.engine_args()` (attention backend / `attention_config.flash_attn_version`)
    and the run header records the profile (`profile["target"]`) for the Match resolver and the collector."""
    if not s:
        return None
    from verity_vllm.frontend.target_profile import parse_target_arg as _parse
    return _parse(s)


def target_of_workload(workload: dict | None, cli: str | None = None):
    """[R13 hopper] The `TargetProfile` a GPU stage builds its engine for: the CLI `--target` (JSON overrides / path) when given, else the
    workload manifest's `target` block (the row's declared overrides, e.g. `{"flash_attn_version": 2}` for `…__bi-eager-fa2`) laid over
    THIS host's device facts (`TargetProfile.from_device()`: capability + SM count, as `derive_step` does without `--target`), else None
    (no declared target: vLLM's own selection, header records the device only)."""
    if cli:
        return parse_target_arg(cli)
    t = (workload or {}).get("target")
    if not isinstance(t, dict) or not t:
        return None
    from verity_vllm.frontend.target_profile import TargetProfile
    base = TargetProfile.from_device() if torch.cuda.is_available() else TargetProfile.default()
    d = base.to_json()
    unknown = set(t) - set(TargetProfile.__dataclass_fields__)   # [R15 fp8] the fields, not the serialised keys (None fields may be omitted)
    if unknown:
        raise ValueError(f"workload target: unknown TargetProfile fields {sorted(unknown)}")
    d.update(t)
    return TargetProfile.from_json(d)


def parse_engine_args(items: list[str] | None) -> dict[str, Any]:
    """[GEN batch SHARED-CODE] `--engine-arg key=value` (repeatable) -> `LLM(**kw)` overrides; the value is parsed as JSON when it is
    one (`64`, `true`, `null`, `"x"`), else kept as a string.  Only for scheduler/memory knobs the sweep varies (max_num_batched_tokens,
    enable_prefix_caching, ...); the pinned model / dtype / eager / attention profile stays `engine_kwargs`' business."""
    out: dict[str, Any] = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise ValueError(f"--engine-arg expects key=value, got {it!r}")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


_VERITY_INDUCTOR_DIR_MARK = "VERITY_INDUCTOR_DIR_PER_PROCESS"


def prepare_compiled_process_cache() -> dict[str, Any]:
    """[R15 compiled, rev-pm M-0266 (b)] Before a COMPILED engine is built: make this process's Inductor output code identifiable.  (1) A fresh,
    empty TORCHINDUCTOR_CACHE_DIR unless the caller already exported one it marks as per-process (compiled_commit.sh) -- torch's default
    /tmp/torchinductor_<user> is shared by every process on the pod, so a pin read from it would digest other rows' kernels; (2)
    VLLM_DISABLE_COMPILE_CACHE=1 -- a vLLM compile-cache hit replays another process's artifacts and emits no output code here (the pin
    would be null: a compiled record without positive evidence of compilation).  Costs one Inductor compilation per process (~70 s for
    Llama-3.2-1B), recorded in the header as compilation.compile_cache = disabled."""
    import tempfile

    out: dict[str, Any] = {}
    if os.environ.get(_VERITY_INDUCTOR_DIR_MARK) != "1":
        d = tempfile.mkdtemp(prefix="verity_inductor_")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = d
        os.environ[_VERITY_INDUCTOR_DIR_MARK] = "1"
        out["inductor_cache_dir"] = d
    else:
        out["inductor_cache_dir"] = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    out["compile_cache"] = "disabled"
    return out


def engine_kwargs_for(checkpoints: str | Path = "manifests/checkpoints.json", *, max_num_seqs: int = 8, case: str = CASE,
                      engine_args: dict[str, Any] | None = None, target: Any = None, execution: str | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """[R16 sys] The exact `LLM(**kw)` keyword set `build_engine` would build from these inputs (+ the env flags it applies and the
    checkpoint record), WITHOUT building: a hot worker compares it with its live engine's `kwargs` to decide reuse vs restart.  Applies the
    same environment knobs as `build_engine` (idempotent).  Refusals (unknown execution, contradicting enforce_eager) are the same."""
    from verity_vllm_adapter import profile as prof

    if target is not None:
        # [R15 fp8] the declared target's ENVIRONMENT knobs (`TargetProfile.env()`: VLLM_BATCH_INVARIANT, VLLM_USE_DEEP_GEMM for
        # `fp8_block_gemm`) are applied before vLLM reads them -- the same object the builder applied; the header records them
        # (`target_env`) and `target_mismatches` reads the selected FP8 kernel back
        os.environ.update(target.env())
    env_flags = prof.apply_env()
    if target is not None:
        env_flags = {**env_flags, **{k: os.environ[k] for k in target.env()}}
    cp = prof.checkpoint_for(case, checkpoints)
    if execution is not None and execution not in EXECUTION_ENFORCE_EAGER:
        raise RuntimeError(f"execution label {execution!r} is not modelled by the harness (known: {sorted(EXECUTION_ENFORCE_EAGER)}); refusing to build an engine")
    want_eager = EXECUTION_ENFORCE_EAGER[execution] if execution is not None else True
    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=want_eager)
    kw["enforce_eager"] = want_eager
    kw["enable_prefix_caching"] = False
    kw["max_num_seqs"] = int(max_num_seqs)
    if target is not None:
        kw.update(target.engine_args())
    kc = dict(kw.get("kernel_config") or {})
    kc.setdefault("enable_flashinfer_autotune", False)
    kw["kernel_config"] = kc
    kw.update(engine_args or {})
    if execution is not None and bool(kw.get("enforce_eager", True)) != want_eager:
        raise RuntimeError(f"execution mismatch: the row declares execution {execution!r} (enforce_eager={want_eager}) but the engine args "
                           f"request enforce_eager={kw.get('enforce_eager')!r}; refusing to build an engine for the wrong execution")
    return kw, env_flags, cp


def build_engine(checkpoints: str | Path = "manifests/checkpoints.json", *, max_num_seqs: int = 8,
                 verify_weights: bool = False, case: str = CASE, engine_args: dict[str, Any] | None = None, target: Any = None,
                 execution: str | None = None) -> Engine:
    from verity_vllm_adapter import profile as prof

    # [R15 EXEC-02 / COORD M-0063 (2)] `enforce_eager` comes from the row's declared execution (`execution_of_workload`), not from a literal:
    # an undeclared execution (no row id) keeps the historical eager default and is recorded as `declared_execution: null`; an unknown label
    # fails closed.  A CLI `--engine-arg enforce_eager=...` that contradicts the declared execution is refused BEFORE any engine is built,
    # so an eager engine can never serve a `__bi-compiled` row (and vice versa) under a green-looking header (rev-pm M-0040).
    # [R13 hopper] `kernel_config.enable_flashinfer_autotune=False` by construction (the autotune dummy run must not depend on the shell's PATH);
    # [GEN batch SHARED-CODE] explicit CLI engine args override last and are recorded in versions.json/run.json.  All of it in
    # `engine_kwargs_for` (R16 sys: the hot worker computes the same keyword set to decide reuse vs restart).
    kw, env_flags, cp = engine_kwargs_for(checkpoints, max_num_seqs=max_num_seqs, case=case, engine_args=engine_args, target=target, execution=execution)
    from vllm import LLM  # noqa: E402  (after apply_env)

    if verify_weights:
        prof.verify_local_weights(cp)
    want_eager = bool(kw["enforce_eager"])
    if kw.get("enforce_eager") is False:
        prepare_compiled_process_cache()
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    runner = prof.get_model_runner(llm)
    model = runner.get_model()
    eng = Engine(llm, runner, model, cp, kw, env_flags, build_s, case, target, execution, None)
    eng.effective = effective_execution_doc(llm, model)      # [R15 EXEC-02] read back from vllm_config: what this engine really executes
    if execution is not None and eng.effective.get("enforce_eager") is not None and bool(eng.effective["enforce_eager"]) != want_eager:
        raise RuntimeError(f"execution mismatch after build: the row declares {execution!r} but vllm_config.model_config.enforce_eager="
                           f"{eng.effective['enforce_eager']} (compilation {eng.effective.get('mode')}, cudagraph {eng.effective.get('cudagraph_mode')})")
    if target is not None:
        # fail closed: the engine's device and attention version must be the declared ones (the row is for THIS target, not the host's)
        bad = target_mismatches(eng)
        if bad:
            raise RuntimeError("engine does not realise the declared TargetProfile: " + "; ".join(bad))
    return eng


def vllm_has_flashinfer() -> bool | None:
    """[R13 hopper] vLLM's own `has_flashinfer()` verdict for this process (package present AND nvcc on PATH or cubins) -- the host fact
    that flips its FlashInfer autotune (cc >= 9.0) and vocab top-k kernel choices; None when the pinned vLLM has no such helper."""
    try:
        from vllm.utils.flashinfer import has_flashinfer
        return bool(has_flashinfer())
    except Exception:  # noqa: BLE001
        return None


def observed_flash_attn_version(model: Any) -> int | None:
    """[R13 hopper] The `vllm_flash_attn_version` the instantiated attention layers carry (one value over all layers), None if absent / mixed."""
    vs: set = set()
    for _, mod in model.named_modules():
        impl = getattr(mod, "impl", None)
        if impl is not None and hasattr(impl, "vllm_flash_attn_version"):
            vs.add(getattr(impl, "vllm_flash_attn_version"))
    return int(vs.pop()) if len(vs) == 1 and None not in vs else None


def target_mismatches(engine: Engine) -> list[str]:
    """[R13 hopper] Declared `TargetProfile` vs the built engine: device capability + SM count (`torch.cuda.get_device_properties`) and the
    attention layers' FA version (`effective_flash_attn_version`).  Empty = the engine executes the declared target."""
    tp = engine.target
    if tp is None:
        return []
    bad: list[str] = []
    p = torch.cuda.get_device_properties(0)
    if (int(p.major), int(p.minor)) != tuple(tp.compute_capability):
        bad.append(f"compute_capability declared {list(tp.compute_capability)}, device {[p.major, p.minor]}")
    if int(p.multi_processor_count) != int(tp.num_sms):
        bad.append(f"num_sms declared {tp.num_sms}, device {p.multi_processor_count}")
    fav = observed_flash_attn_version(engine.model)
    if fav != tp.effective_flash_attn_version():
        bad.append(f"flash_attn_version declared {tp.flash_attn_version!r} (effective {tp.effective_flash_attn_version()}), layers report {fav!r}")
    # [R15 compiled] the engine must realise the declared EXECUTION: eager profile -> enforce_eager True; `piecewise` -> torch.compile
    # (VLLM_COMPILE) + CUDAGraphMode PIECEWISE read back from the built vllm_config (an eager engine labelled compiled, or a compiled one
    # labelled eager, is refused here before anything is recorded)
    comp = compilation_facts(engine.llm)
    if comp:
        want = tp.COMPILATIONS.get(tp.compilation) if tp.compilation else None
        if want is None and comp.get("enforce_eager") is False:
            bad.append(f"profile declares eager execution, engine built with enforce_eager=False (compilation {comp.get('mode')!r}/{comp.get('cudagraph_mode')!r})")
        if want is not None:
            if comp.get("enforce_eager") is not False:
                bad.append(f"profile declares compilation {tp.compilation!r}, engine built with enforce_eager={comp.get('enforce_eager')!r}")
            if comp.get("cudagraph_mode") != want["cudagraph_mode"]:
                bad.append(f"profile declares cudagraph_mode {want['cudagraph_mode']!r}, engine runs {comp.get('cudagraph_mode')!r}")
            if comp.get("mode") not in (want["mode"], "3"):
                bad.append(f"profile declares CompilationMode {want['mode']!r}, engine runs {comp.get('mode')!r}")
            if bool(comp.get("emulate_precision_casts", False)) != bool(want.get("emulate_precision_casts", False)):
                bad.append(f"profile declares Inductor emulate_precision_casts={want.get('emulate_precision_casts')!r}, engine compiled with "
                           f"{comp.get('emulate_precision_casts')!r} (inductor_compile_config {comp.get('inductor_compile_config')!r})")
    if getattr(tp, "fp8_block_gemm", None) is not None:
        # [R15 fp8] the block-scaled FP8 GEMM kernel the FP8 linears carry must be the declared one (the builder read the same back)
        from verity_vllm.frontend.target_profile import fp8_block_gemm_kernel_of
        got = fp8_block_gemm_kernel_of(engine.model)
        if got != tp.fp8_block_gemm_kernel():
            bad.append(f"fp8_block_gemm declared {tp.fp8_block_gemm!r} ({tp.fp8_block_gemm_kernel()}), layers report {got!r}")
    return bad


def resolved_async_scheduling(llm: Any) -> bool | None:
    """[R16 sys int-3, coord M-0213 RULING 21] The async-scheduling mode the built engine actually runs.  vLLM's CLI default is None and is
    RESOLVED at `VllmConfig.__post_init__` (True unless pooling model / incompatible spec-decode / executor without support / ROCm DBO), so the
    value the harness passed -- usually nothing -- says nothing about execution.  Read back from `llm.llm_engine.vllm_config.scheduler_config`
    after the build; None means the engine did not expose a resolved bool (unattested), never "off"."""
    try:
        v = llm.llm_engine.vllm_config.scheduler_config.async_scheduling
    except Exception:  # noqa: BLE001
        return None
    return None if v is None else bool(v)


def compilation_facts(llm: Any) -> dict[str, Any]:
    """[R15 compiled] What the built engine actually compiles / graphs, read back from its vllm_config (never from the kwargs): CompilationMode
    (name; the pinned vLLM's IntEnum str() is the number, recorded too), CUDAGraphMode, capture sizes, inductor graph partition, enforce_eager,
    compile-cache switch.  Written into the run header's `compilation` block beside the profile's `effective_compilation` (GM-01 G1 compares a
    compiled Program's declared facts against it) and into commit_delta's versions.json (control and treatment must agree)."""
    out: dict[str, Any] = {}
    try:
        vc = llm.llm_engine.vllm_config
        cc = vc.compilation_config
        mode = getattr(cc, "mode", getattr(cc, "level", None))
        out["mode"] = getattr(mode, "name", None) or str(mode)
        out["mode_value"] = int(mode) if isinstance(mode, int) else str(mode)
        cgm = getattr(cc, "cudagraph_mode", None)
        out["cudagraph_mode"] = getattr(cgm, "name", None) or str(cgm)
        sizes = getattr(cc, "cudagraph_capture_sizes", None) or []
        out["capture_sizes"] = sorted(int(s) for s in list(sizes)[:256])
        out["max_cudagraph_capture_size"] = getattr(cc, "max_cudagraph_capture_size", None)
        out["use_inductor_graph_partition"] = bool(getattr(cc, "use_inductor_graph_partition", False))
        out["backend"] = str(getattr(cc, "backend", None) or ("inductor" if out["mode_value"] == 3 else None))
        out["enforce_eager"] = bool(vc.model_config.enforce_eager)
        out["compile_cache"] = "disabled" if os.environ.get("VLLM_DISABLE_COMPILE_CACHE") == "1" else "enabled"
        # [R15 compiled, M-0285 (b)] the Inductor config the pieces were compiled with: emulate_precision_casts makes the compiled kernels
        # perform the bf16 roundings the Program specifies at every module/op boundary (Inductor otherwise carries fp32 through fused
        # regions) -- the fact that decides whether a materialised boundary value can change the computation
        out["inductor_compile_config"] = dict(getattr(cc, "inductor_compile_config", None) or {})
        out["emulate_precision_casts"] = bool(out["inductor_compile_config"].get("emulate_precision_casts", False))
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:200]
    return out


# [R16 compiled, rev F-r16-09] ELF sections of a cubin that the DEVICE DOES NOT READ: debug/line info, embedded PTX text, and the host-side
# symbol/string tables.  The code identity is every OTHER section (deny-list, not a .text allow-list): .text.* (SASS), .nv.constant0.* (param
# bank + compiler constants), .nv.constant2/3 (constant-memory data), .nv.global(.init) (initialised device globals), .rodata, .nv.info*, ...
CUBIN_DENY_LIST: tuple[str, ...] = (".debug*", ".rel.debug*", ".rela.debug*", ".nv.debug*", ".nv_debug*", ".rel.nv_debug*", ".rela.nv_debug*", ".rel.nv.debug*",
                                    ".rela.nv.debug*", ".symtab", ".strtab", ".shstrtab")
_CUBIN_DENY_PREFIX = (".debug", ".rel.debug", ".rela.debug", ".nv.debug", ".nv_debug", ".rel.nv_debug", ".rela.nv_debug", ".rel.nv.debug", ".rela.nv.debug")
CUBIN_CODE_IDENTITY_RULE = ("sha256 over every ELF section of the executed cubin EXCEPT the named deny-list " + repr(CUBIN_DENY_LIST)
                            + " (debug/line info, embedded PTX text, symbol/string tables); NOT a .text allow-list")
_PATH_STRING_RE = None


def cubin_section_denied(name: str) -> bool:
    return name in (".symtab", ".strtab", ".shstrtab") or name.startswith(_CUBIN_DENY_PREFIX)


def _elf_section_bytes(b: bytes) -> dict[str, bytes]:
    """Pure-Python ELF32/64 section walk -> {section name: bytes} ({} if not an ELF)."""
    import struct

    if len(b) < 64 or b[:4] != b"\x7fELF":
        return {}
    is64, le = b[4] == 2, b[5] == 1
    E = "<" if le else ">"
    if is64:
        (e_shoff,) = struct.unpack_from(E + "Q", b, 0x28)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(E + "HHH", b, 0x3A)
    else:
        (e_shoff,) = struct.unpack_from(E + "I", b, 0x20)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(E + "HHH", b, 0x2E)
    hdrs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        if is64:
            sh_name, sh_type, _fl, _ad, sh_offset, sh_size = struct.unpack_from(E + "IIQQQQ", b, off)
        else:
            sh_name, sh_type, _fl, _ad, sh_offset, sh_size = struct.unpack_from(E + "IIIIII", b, off)
        hdrs.append((sh_name, sh_type, sh_offset, sh_size))
    if e_shstrndx >= len(hdrs):
        return {}
    _, _, stroff, strsz = hdrs[e_shstrndx]
    strtab = b[stroff:stroff + strsz]
    out: dict[str, bytes] = {}
    for sh_name, sh_type, sh_offset, sh_size in hdrs:
        end = strtab.find(b"\0", sh_name)
        name = strtab[sh_name:end if end >= 0 else None].decode(errors="replace") or "<null>"
        out[name] = b"" if sh_type == 8 else b[sh_offset:sh_offset + sh_size]          # SHT_NOBITS
    return out


def cubin_sections(b: bytes) -> dict[str, str]:
    """[R16 compiled] Per-section sha256 of a cubin (ELF), every section (deny-listed ones included, so that WHICH sections differ between
    two cubins is on record).  {} if not an ELF."""
    import hashlib

    return {n: hashlib.sha256(d).hexdigest() for n, d in _elf_section_bytes(b).items()}


def _split_path_strings(data: bytes) -> tuple[bytes, list[str]]:
    """Replace every printable run (>= 4 bytes) that is an ABSOLUTE PATH (starts with '/', no whitespace) by b'<PATH>' and return the
    normalised bytes + the removed strings.  Device data of Inductor/Triton kernels embeds the generated module's absolute source path
    (tl.device_assert / assert message strings in .nv.global.init) -- a per-process string, not computation."""
    import re

    global _PATH_STRING_RE
    if _PATH_STRING_RE is None:
        _PATH_STRING_RE = re.compile(rb"/[\x21-\x7e]{3,}")
    found: list[str] = []

    def _sub(m: "re.Match[bytes]") -> bytes:
        found.append(m.group().decode("latin1"))
        return b"<PATH>"

    return _PATH_STRING_RE.sub(_sub, data), found


def cubin_section_detail(b: bytes) -> dict[str, dict[str, Any]]:
    """[rev F-r16-09 (3)] For every NON-deny-listed section: sha256, size, sha256 of the bytes with absolute-path strings normalised
    (`residue_sha256`), and the path strings themselves -- so a cross-arm difference in device data can be DECODED from the record:
    residue equal + only path strings differ = per-process source path embedded in device data; anything else = not one specialization."""
    import hashlib

    out: dict[str, dict[str, Any]] = {}
    for n, d in _elf_section_bytes(b).items():
        if cubin_section_denied(n):
            continue
        norm, paths = _split_path_strings(d)
        ent: dict[str, Any] = {"sha256": hashlib.sha256(d).hexdigest(), "size": len(d), "residue_sha256": hashlib.sha256(norm).hexdigest()}
        if paths:
            ent["path_strings"] = paths[:16]
            ent["n_path_strings"] = len(paths)
        out[n] = ent
    return out


def cubin_code_sha256(sections: dict[str, str]) -> str | None:
    """sha256 over the sorted {(section name, sha256)} of every section NOT on CUBIN_DENY_LIST = everything the device loads (SASS, constant
    banks, initialised globals, rodata, kernel info), without debug/line-info/symbol sections.  None if the cubin has no .text section."""
    import hashlib

    code = sorted((k, v) for k, v in sections.items() if not cubin_section_denied(k))
    return hashlib.sha256(json.dumps(code).encode()).hexdigest() if any(k.startswith(".text") for k, _ in code) else None


def cubin_code_sha256_normalized(detail: dict[str, dict[str, Any]]) -> str | None:
    """As cubin_code_sha256 but over each non-deny-listed section's residue_sha256 (absolute-path strings normalised): equal across two
    processes iff every difference is a path string.  The strict digest is the identity; this one names what the difference is."""
    import hashlib

    code = sorted((k, v.get("residue_sha256")) for k, v in detail.items())
    return hashlib.sha256(json.dumps(code).encode()).hexdigest() if any(k.startswith(".text") for k, _ in code) else None


_TRITON_CACHE_INDEX: dict[str, dict[str, str]] = {}


def _triton_cache_files(h: str | None) -> dict[str, Any]:
    """Triton's compiled-kernel cache entry for hash `h` (TRITON_CACHE_DIR/<hash>/): the cubin bytes (when Inductor no longer holds asm) and
    the compile metadata json (num_warps / num_stages / target / options) -- what the executed binary was compiled WITH."""
    if not h:
        return {}
    import base64
    import glob

    root = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    cands = [os.path.join(root, str(h))]                          # Triton <= 3.3: <hex hash>/
    try:                                                           # Triton >= 3.4: base32(sha256 digest)/ (52 chars, no padding)
        cands.append(os.path.join(root, base64.b32encode(bytes.fromhex(str(h))).decode().rstrip("=")))
    except ValueError:
        pass
    d = next((c for c in cands if os.path.isdir(c)), None)
    if d is None:
        # last resort: the metadata json of every entry carries "hash": <hex> (indexed once per root)
        idx = _TRITON_CACHE_INDEX.get(root)
        if idx is None:
            idx = {}
            for mj in glob.glob(os.path.join(root, "*", "*.json")):
                if os.path.basename(mj).startswith("__grp__"):
                    continue
                try:
                    md = json.load(open(mj))
                    if isinstance(md, dict) and md.get("hash"):
                        idx[str(md["hash"])] = os.path.dirname(mj)
                except Exception:  # noqa: BLE001
                    continue
            _TRITON_CACHE_INDEX[root] = idx
        d = idx.get(str(h))
    if d is None:
        return {}
    out: dict[str, Any] = {"dir": d}
    cub = sorted(glob.glob(os.path.join(d, "*.cubin")))
    if cub:
        try:
            out["cubin"] = open(cub[0], "rb").read()
        except OSError:
            pass
    for mj in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(mj).startswith("__grp__"):
            continue
        try:
            md = json.load(open(mj))
            if isinstance(md, dict) and ("num_warps" in md or "target" in md):
                out["meta"] = {k: md.get(k) for k in ("name", "num_warps", "num_stages", "num_ctas", "cluster_dims", "enable_fp_fusion", "debug", "shared",
                                                      "ptx_version", "target", "maxnreg", "launch_cooperative_grid") if k in md}
                break
        except Exception:  # noqa: BLE001
            continue
    return out


def _binary_identity(k: Any, cfg: Any = None) -> dict[str, Any]:
    """One compiled Triton binary: triton hash (source x signature/specialization x options x env), cubin sha256 (bytes), per-section
    sha256 + code_sha256 (every section except CUBIN_DENY_LIST, rev F-r16-09) + code_sha256_normalized / section_detail (path strings in
    device data decoded), the compile metadata, and the autotune config it was compiled for."""
    import hashlib

    ent: dict[str, Any] = {"triton_hash": str(getattr(k, "hash", None)) if getattr(k, "hash", None) else None}
    asm = getattr(k, "asm", None)
    cub = asm.get("cubin") if isinstance(asm, dict) else None
    src = "asm" if cub is not None else None
    tc = _triton_cache_files(ent["triton_hash"])
    if cub is None and tc.get("cubin") is not None:
        cub, src = tc["cubin"], "triton_cache_dir"
    if tc.get("meta"):
        ent["compile_meta"] = tc["meta"]
    if cub is not None:
        secs = cubin_sections(cub)
        det = cubin_section_detail(cub)
        ent.update({"cubin_sha256": hashlib.sha256(cub).hexdigest(), "cubin_bytes": len(cub), "cubin_from": src,
                    "sections": secs, "code_sha256": cubin_code_sha256(secs), "code_sha256_normalized": cubin_code_sha256_normalized(det),
                    "section_detail": det, "code_identity_rule": CUBIN_CODE_IDENTITY_RULE,
                    "sections_denied": sorted(k for k in secs if cubin_section_denied(k))})
    md = getattr(k, "metadata", None)
    if md is not None and "compile_meta" not in ent:
        ent["compile_meta"] = {a: getattr(md, a, None) for a in ("num_warps", "num_stages", "num_ctas", "shared", "enable_fp_fusion", "debug") if hasattr(md, a)}
    if cfg is not None:
        ent["config"] = {"kwargs": {str(a): (int(b) if isinstance(b, int) else str(b)) for a, b in (getattr(cfg, "kwargs", None) or {}).items()},
                         "num_warps": getattr(cfg, "num_warps", None), "num_stages": getattr(cfg, "num_stages", None)}
    ent["id"] = ("code:" + ent["code_sha256"]) if ent.get("code_sha256") else (("triton:" + ent["triton_hash"]) if ent.get("triton_hash") else None)
    return ent


def loaded_code_objects() -> dict[str, Any]:
    """[R15 compiled, Daniel M-0776 (3)] Identity of the Inductor code objects LOADED IN THIS PROCESS -- read from the running interpreter,
    not from files: `torch._inductor.codecache.PyCodeCache` is the registry of every output-code module Inductor exec'd (torch >= 2.9:
    `modules` list + `modules_no_attr` by path; older: `cache` by content-hash key), and each `CachingAutotuner` bound in those modules is
    a compiled Triton kernel: its JIT source (`fn.src`), and the binaries the device runs (`compile_results[*].kernel` / `launchers[*].bin`:
    Triton's compiled-kernel `hash`, and the cubin bytes' sha256 while Inductor still holds them).  Digests: kernel_set_sha256 over the sorted
    {(kernel name, source sha256)}, binary_set_sha256 over the sorted binary identities (cubin sha256 or triton hash), module_keys_sha256.

    [R16 unit B] per compiled binary: triton hash + cubin sha256 + per-ELF-section sha256 + code_sha256 (every section except CUBIN_DENY_LIST, rev F-r16-09) + compile metadata +
    the autotune config it was compiled for (`compile_results[*]` = every candidate the process compiled; `launchers[*]` = the one(s) it
    EXECUTES after autotuning); `executed_set_sha256` over the sorted {(kernel, src sha, executed code id, executed config)} is the identity
    of the specialization the graphs replay.  Two processes running the same code objects differ in cubin BYTES when the debug/line-info
    sections carry the per-process module path (code_sha256 equal) or in EXECUTED config when autotuning picked another candidate."""
    import hashlib

    def _sha(b: Any) -> str:
        return hashlib.sha256(b if isinstance(b, (bytes, bytearray)) else str(b).encode()).hexdigest()

    try:
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        reg: list[tuple[str, Any]] = []
        seen: set[int] = set()
        for src_name in ("modules", "modules_no_attr", "cache"):
            coll = getattr(PyCodeCache, src_name, None)
            items = list(coll.items()) if isinstance(coll, dict) else [(getattr(m, "key", None) or getattr(m, "__file__", None) or str(id(m)), m) for m in (coll or [])]
            for k, m in items:
                if id(m) in seen:
                    continue
                seen.add(id(m)); reg.append((str(k), m))
        mods, kernels, bins_all = [], {}, set()
        for key, mod in reg:
            ks = 0
            for name, obj in list(vars(mod).items()):
                if not isinstance(obj, CachingAutotuner):
                    continue
                fn = getattr(obj, "fn", None)
                kname = getattr(fn, "__name__", None) or name
                src = getattr(fn, "src", None)
                bins: list[str] = []
                compiled: list[dict[str, Any]] = []          # every candidate binary this process compiled (autotune candidates included)
                for cr in list(getattr(obj, "compile_results", None) or []):
                    k = getattr(cr, "kernel", None)
                    b_ = _binary_identity(k, getattr(cr, "config", None))
                    compiled.append(b_)
                    asm = getattr(k, "asm", None)
                    cub = asm.get("cubin") if isinstance(asm, dict) else None
                    if cub is not None:
                        bins.append("cubin:" + _sha(cub))
                    elif getattr(k, "hash", None):
                        bins.append("triton:" + str(k.hash))
                # the EXECUTED specialization: the launcher(s) left after autotuning (Inductor keeps ONE launcher = the benchmark winner; the
                # graphs captured / the eager steps launch that binary) -- autotuning picks it per process by benchmark, so two processes
                # running the same code objects can execute differently configured binaries
                executed: list[dict[str, Any]] = []
                cfgs = []
                for l in list(getattr(obj, "launchers", None) or []):
                    c = getattr(l, "config", None)
                    b = getattr(l, "bin", None)
                    e_ = _binary_identity(b, c) if b is not None else ({"config": None, "id": None})
                    if b is None and c is not None:
                        # no `bin` on the launcher (older Inductor): the executed binary is the compile result whose config equals the launcher's
                        for b_ in compiled:
                            if b_.get("config") and c is not None and b_["config"] == {"kwargs": {str(a): (int(v) if isinstance(v, int) else str(v)) for a, v in (getattr(c, "kwargs", None) or {}).items()},
                                                                                        "num_warps": getattr(c, "num_warps", None), "num_stages": getattr(c, "num_stages", None)}:
                                e_ = dict(b_)
                                break
                    if c is not None:
                        cfgs.append({"kwargs": {str(a): (int(b2) if isinstance(b2, int) else str(b2)) for a, b2 in (getattr(c, "kwargs", None) or {}).items()},
                                     "num_warps": getattr(c, "num_warps", None), "num_stages": getattr(c, "num_stages", None)})
                        e_.setdefault("config", cfgs[-1])
                    executed.append(e_)
                    if b is not None:
                        asm = getattr(b, "asm", None)
                        if isinstance(asm, dict) and asm.get("cubin") is not None:
                            bins.append("cubin:" + _sha(asm["cubin"]))
                        elif getattr(b, "hash", None):
                            bins.append("triton:" + str(b.hash))
                if not executed and len(compiled) == 1:
                    executed = [dict(compiled[0])]
                bins_all.update(bins)
                candidates = [{"kwargs": {str(a): (int(b2) if isinstance(b2, int) else str(b2)) for a, b2 in (getattr(c, "kwargs", None) or {}).items()},
                               "num_warps": getattr(c, "num_warps", None), "num_stages": getattr(c, "num_stages", None)} for c in list(getattr(obj, "configs", None) or [])]
                ent = {"src_sha256": _sha(src) if src else None, "binaries": sorted(set(bins)), "n_launchers": len(getattr(obj, "launchers", None) or []),
                       "n_compile_results": len(compiled), "executed_configs": cfgs, "n_configs_candidate": len(candidates), "candidate_configs": candidates,
                       "compiled": compiled, "executed": executed, "module_key": key,
                       "autotuned": len(candidates) > 1 or len(compiled) > 1}
                if kname in kernels and kernels[kname]["src_sha256"] == ent["src_sha256"]:   # the same kernel bound in kernel module + wrapper namespace
                    kernels[kname]["binaries"] = sorted(set(kernels[kname]["binaries"]) | set(bins))
                    have = {json.dumps(x.get("id")) + json.dumps(x.get("config"), sort_keys=True) for x in kernels[kname]["compiled"]}
                    kernels[kname]["compiled"] += [x for x in compiled if json.dumps(x.get("id")) + json.dumps(x.get("config"), sort_keys=True) not in have]
                    have_e = {json.dumps(x.get("id")) + json.dumps(x.get("config"), sort_keys=True) for x in kernels[kname]["executed"]}
                    kernels[kname]["executed"] += [x for x in executed if json.dumps(x.get("id")) + json.dumps(x.get("config"), sort_keys=True) not in have_e]
                    kernels[kname]["executed_configs"] = kernels[kname]["executed_configs"] + [c for c in cfgs if c not in kernels[kname]["executed_configs"]]
                    kernels[kname]["n_compile_results"] = len(kernels[kname]["compiled"])
                    kernels[kname]["n_launchers"] += ent["n_launchers"]
                    kernels[kname]["autotuned"] = kernels[kname]["autotuned"] or ent["autotuned"]
                    kernels[kname].setdefault("module_keys", [kernels[kname]["module_key"]]).append(key)
                else:
                    kernels[kname] = ent
                ks += 1
            mods.append({"key": key[-64:], "path": getattr(mod, "__file__", None), "kernels": ks})
        kset = sorted((k, v["src_sha256"]) for k, v in kernels.items())
        # [R16 unit B] the EXECUTED specialization set: per kernel the code id(s) the launchers run (code_sha256 of the non-deny-listed sections when the
        # cubin is readable, else the triton hash) x the executed autotune config -- the identity of what the CUDA graphs replay
        exec_set = sorted((k, v["src_sha256"], json.dumps([(e.get("id"), json.dumps(e.get("config"), sort_keys=True)) for e in v["executed"]], sort_keys=True))
                          for k, v in kernels.items())
        code_set = sorted((k, e.get("code_sha256")) for k, v in kernels.items() for e in v["executed"] if e.get("code_sha256"))
        norm_set = sorted((k, e.get("code_sha256_normalized")) for k, v in kernels.items() for e in v["executed"] if e.get("code_sha256_normalized"))
        return {"modules": len(mods), "kernels": len(kernels), "kernels_with_binaries": sum(1 for v in kernels.values() if v["binaries"]),
                "kernel_set_sha256": _sha(json.dumps(kset)) if kset else None, "binary_set_sha256": _sha(json.dumps(sorted(bins_all))) if bins_all else None,
                "cubin_set_sha256": _sha(json.dumps(sorted(bins_all))) if bins_all else None,     # alias (compiled_merge reads it)
                "executed_set_sha256": _sha(json.dumps(exec_set)) if exec_set else None,
                "executed_code_set_sha256": _sha(json.dumps(code_set)) if code_set else None,
                "executed_code_set_sha256_normalized": _sha(json.dumps(norm_set)) if norm_set else None,
                "code_identity_rule": CUBIN_CODE_IDENTITY_RULE,
                "kernels_with_executed_code": len({k for k, _ in code_set}),
                "kernels_autotuned": sorted(k for k, v in kernels.items() if v.get("autotuned")),
                "binary_kinds": sorted({b.split(":", 1)[0] for b in bins_all}),
                "module_keys_sha256": _sha(json.dumps(sorted(m["key"] for m in mods))) if mods else None,
                "kernel_names": sorted(kernels), "per_kernel": kernels, "modules_list": mods,
                "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
                "read_from": "torch._inductor.codecache.PyCodeCache (loaded modules) x CachingAutotuner.fn.src / compile_results[*].kernel + launchers[*].bin "
                             "(triton hash, cubin sha256 from asm or TRITON_CACHE_DIR/<hash>/*.cubin, per-ELF-section sha256, code_sha256 = every section "
                             "except CUBIN_DENY_LIST, code_sha256_normalized + section_detail.path_strings (absolute-path strings in device data decoded), "
                             "compile metadata, autotune config) in this process"}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}


def loaded_code_objects_delta(start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    """[R16 unit B -- Daniel: repeated graph replay must be covered, not only compilation/warm-up] Compare two loaded_code_objects() snapshots of
    ONE process (after engine build = compile + Inductor autotune + CUDA-graph capture, and after the last measured run): the graphs replay
    exactly the launchers that existed at capture, so an equal executed set + no new modules/kernels/binaries means the artifact identity
    recorded at build is the identity of every replayed step; anything new (a recompile, a re-tune, a new specialization) is named."""
    if not isinstance(start, dict) or not isinstance(end, dict) or start.get("error") or end.get("error"):
        return {"ok": None, "why": f"snapshot unavailable: start {str((start or {}).get('error'))[:80]} end {str((end or {}).get('error'))[:80]}"}
    ps, pe = start.get("per_kernel") or {}, end.get("per_kernel") or {}
    new_k = sorted(set(pe) - set(ps))
    gone_k = sorted(set(ps) - set(pe))
    changed_exec = sorted(k for k in set(ps) & set(pe)
                          if json.dumps([(e.get("id"), e.get("config")) for e in ps[k].get("executed") or []], sort_keys=True)
                          != json.dumps([(e.get("id"), e.get("config")) for e in pe[k].get("executed") or []], sort_keys=True))
    changed_bins = sorted(k for k in set(ps) & set(pe) if sorted(ps[k].get("binaries") or []) != sorted(pe[k].get("binaries") or []))
    changed_src = sorted(k for k in set(ps) & set(pe) if ps[k].get("src_sha256") != pe[k].get("src_sha256"))
    out = {"modules_start": start.get("modules"), "modules_end": end.get("modules"), "new_modules": (end.get("modules") or 0) - (start.get("modules") or 0),
           "kernels_start": start.get("kernels"), "kernels_end": end.get("kernels"), "new_kernels": new_k[:40], "kernels_gone": gone_k[:40],
           "executed_changed": changed_exec[:40], "binaries_changed": changed_bins[:40], "sources_changed": changed_src[:40],
           "kernel_set_equal": start.get("kernel_set_sha256") == end.get("kernel_set_sha256"),
           "executed_set_equal": start.get("executed_set_sha256") == end.get("executed_set_sha256"),
           "executed_set_sha256": {"start": start.get("executed_set_sha256"), "end": end.get("executed_set_sha256")}}
    out["ok"] = bool(out["kernel_set_equal"] and out["executed_set_equal"] and not new_k and not changed_exec and not changed_bins and not changed_src
                     and out["new_modules"] == 0)
    out["why"] = None if out["ok"] else ("the code objects / executed specialization changed between build and end of run: " +
                                          ", ".join(f"{k}={v}" for k, v in out.items() if k in ("new_modules", "new_kernels", "executed_changed", "binaries_changed", "sources_changed") and v))
    return out


def generated_kernels_of_this_process(inductor_dir: str | None = None) -> dict[str, Any]:
    """[R15 compiled] Digest of the Inductor output code this process compiled (W1c's kernel identity: kernel_set_sha256 = sha256 of the sorted
    {(kernel name, source sha256)}; call_set_sha256 over the pieces' launch programs), plus where it was read from.  Shared by the record
    path (RunHeader.compilation.generated_kernels) and commit_delta's versions.json so the two pins are comparable byte for byte."""
    try:
        from verity_vllm_adapter import kernel_identity as KI

        ind = inductor_dir or os.environ.get("TORCHINDUCTOR_CACHE_DIR") or KI.inductor_cache_dir()
        doc = KI.collect(ind, KI.triton_cache_dir())
        gk = dict(doc["summary"])
        gk["inductor_cache_dir"] = ind
        gk["per_process_cache_dir"] = os.environ.get(_VERITY_INDUCTOR_DIR_MARK) == "1"      # set by prepare_compiled_process_cache / compiled_commit.sh
        gk["pieces"] = [{"path": w["path"], "norm_sha256": w["norm_sha256"], "call_sha256": w["call_sha256"], "kernels": len(w["kernels"]),
                         "launches": w["launches"], "extern_calls": w["extern_calls"]} for w in doc["wrappers"]]
        gk["kernels"] = doc["kernels"]
        gk["loaded"] = loaded_code_objects()                                       # [M-0776 (3)] the code objects this process actually loaded
        if gk["loaded"].get("kernel_names") is not None:
            gk["loaded"]["files_vs_loaded"] = {"in_files_not_loaded": sorted(set(doc["kernels"]) - set(gk["loaded"]["kernel_names"]))[:40],
                                               "loaded_not_in_files": sorted(set(gk["loaded"]["kernel_names"]) - set(doc["kernels"]))[:40]}
        if not gk.get("wrappers"):
            # kernel_identity digests the EMPTY set to a hex string; a pin over nothing is not evidence -> null it and say why
            gk["error"] = f"no Inductor output code under {ind!r} (compile cache hit from another process, or the dir is not this process's)"
        elif not gk["per_process_cache_dir"]:
            # torch's default /tmp/torchinductor_<user> holds every process's output code on the pod: a digest over it is not THIS engine's
            gk["error"] = f"Inductor output code read from a shared dir {ind!r} ({gk['wrappers']} pieces): not this process's identity"
        if gk.get("error"):
            for k in ("kernel_set_sha256", "call_set_sha256", "wrapper_set_sha256", "wrapper_norm_set_sha256"):
                gk[k] = None
        return gk
    except Exception as e:  # noqa: BLE001
        return {"kernel_set_sha256": None, "call_set_sha256": None, "wrappers": 0, "distinct_kernel_sources": 0, "error": repr(e)[:200]}


def profile_manifest(engine: Engine, *, adapter_revision: str, wheel_sha256: str | None, vllm_commit: str | None,
                     sampling: dict) -> tuple[dict, str, dict]:
    """(manifest, profile_id, notes) via `profile.build_manifest` exactly like the production recorder; falls back
    to a sha256 of the engine kwargs + versions when build_manifest cannot run (and says so in the notes)."""
    from verity_vllm_adapter import profile as prof

    case = engine.case
    acc = ACCEPTED.get(case, {})
    notes: dict[str, Any] = {"case": case, "accepted_profile_id": acc.get("profile_id"), "accepted_record": acc.get("record")}
    # [GEN sampling] the manifest records the sampling actually requested (a non-greedy variant is another profile id by construction)
    sampling_desc = {"greedy": bool(sampling.get("greedy", True)), "temperature": float(sampling.get("temperature", 0.0)),
                     "top_p": float(sampling.get("top_p", 1.0)), "top_k": int(sampling.get("top_k", 0)),
                     "seeds": [] if sampling.get("seed") is None else [int(sampling["seed"])], "logprobs": sampling.get("logprobs")}
    if int(sampling.get("n") or 1) != 1:
        sampling_desc["n"] = int(sampling["n"])                     # greedy manifests stay byte-identical (same profile id as before)
    if sampling.get("per_request"):
        sampling_desc["per_request"] = list(sampling["per_request"])   # [ov-sampling] mixed batch: per-request variants
    try:
        manifest = prof.build_manifest(
            case=case, cp=engine.cp, llm=engine.llm, wheel_sha256=wheel_sha256, vllm_commit=vllm_commit,
            adapter_revision=adapter_revision,
            interception="none (production uninstrumented: StepTrace scheduler observer + engine outputs; Inductor/Triton caches read after the run)",
            # [R15 compiled] the header says what the engine was built with (a compiled engine used to be labelled enforce_eager True here)
            sampling=sampling_desc, env_flags=engine.env_flags, enforce_eager=bool(engine.kwargs.get("enforce_eager", False)),
            # lane tp2: the manifest's `parallelism.tensor_parallel_size` / `engine_kwargs.tensor_parallel_size` must say what the
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
        # [r8 E-cleanup, fp16 M12] the served dtype: `build_manifest` hard-codes "bfloat16"; when the harness passed an explicit
        # `dtype` to LLM(**kw) (a workload's `engine_args_required.dtype`, e.g. float16) the manifest must say so -- a served-fp16
        # record labelled bf16 is mislabelled evidence.  Runs without the kwarg are byte-identical (same profile id as before).
        if engine.kwargs.get("dtype") not in (None, "auto"):
            manifest["dtype"] = str(engine.kwargs["dtype"])
            manifest["engine_kwargs"]["dtype"] = str(engine.kwargs["dtype"])
        # [R13 hopper] the attention version the layers actually run (FA2 / FA3), and the declared target when the engine was built for one:
        # `TargetProfile.from_run_profile` reads these back on the CPU side (eager resolver) — the same object the builder applied
        manifest.setdefault("backends", {})["attention_flash_attn_version"] = observed_flash_attn_version(engine.model)
        manifest["backends"]["vllm_has_flashinfer"] = vllm_has_flashinfer()     # host fact vLLM keys kernel choices on (nvcc on PATH)
        # [R15 fp8] the block-scaled FP8 linear kernel the FP8 layers carry (None for a model without FP8 linears): the record attests
        # the kernel whose arithmetic the FP8 Definitions are (CutlassFp8BlockScaledMMKernel), beside the declared `target.fp8_block_gemm`
        from verity_vllm.frontend.target_profile import fp8_block_gemm_kernel_of
        fp8k = fp8_block_gemm_kernel_of(engine.model)
        if fp8k is not None:
            manifest["backends"]["fp8_block_gemm_kernel"] = fp8k
        manifest["engine_kwargs"]["kernel_config"] = dict(engine.kwargs.get("kernel_config") or {})
        if engine.target is not None:
            manifest["target"] = engine.target.to_json()
            manifest["engine_kwargs"].update({k: v for k, v in engine.target.engine_args().items()})
        # [R15 compiled] the effective execution, read back from the built engine: engine_kwargs.enforce_eager = what LLM() got (eager runs
        # stay byte-identical: True as before); a compiled engine adds its compilation_config kwarg and the read-back facts (mode, cudagraph
        # mode, capture sizes, partition) to the header's `compilation` block -- the RunHeader attests the execution GM-01 compares against
        manifest["engine_kwargs"]["enforce_eager"] = bool(engine.kwargs.get("enforce_eager", False))
        if engine.kwargs.get("compilation_config") is not None:
            manifest["engine_kwargs"]["compilation_config"] = engine.kwargs["compilation_config"]
        # [R16 sys int-3, coord M-0213 RULING 21] the record attests the async-scheduling mode the engine RESOLVED (scheduler_config after
        # VllmConfig.__post_init__: True by default on this stack), not the CLI default None the harness leaves in place -- async scheduling
        # changes the step/observer interleaving GM-01 and the Match resolver reason about, so a header without it is unattested execution.
        # Every header now carries the key (profile id moves once, for all rows alike; leaf ids already tolerate that: p=<profile_id>/...).
        manifest["engine_kwargs"]["async_scheduling"] = resolved_async_scheduling(engine.llm)
        if "async_scheduling" in engine.kwargs:
            notes["async_scheduling_requested"] = engine.kwargs["async_scheduling"]
        facts = compilation_facts(engine.llm)
        if facts and facts.get("enforce_eager") is False:
            manifest.setdefault("compilation", {}).update({k: v for k, v in facts.items() if k != "enforce_eager"})
            manifest["compilation"]["enforce_eager"] = False
            # [R15 compiled, rev-pm M-0266/M-0268 (b)] positive, record-level evidence of compilation an eager run cannot produce: the Inductor
            # output code of every compiled piece is on disk by now (vLLM compiles + captures during engine init), digested exactly as
            # commit_delta does (kernel_identity.collect: kernel_set_sha256 over {(kernel name, source sha256)}, call_set over the launch
            # programs).  The dir is vLLM's per-config inductor_cache (it exports TORCHINDUCTOR_CACHE_DIR at initialize_cache) or the explicit
            # per-process override; a null pin here on a compiled engine is an error of record (GM-01 G1 fails closed on it after M-0266).
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


def model_config_subset(engine: Engine) -> dict:
    hf = engine.llm.llm_engine.vllm_config.model_config.hf_config
    keys = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
            "vocab_size", "rms_norm_eps", "tie_word_embeddings", "rope_theta", "max_position_embeddings", "torch_dtype",
            "model_type", "architectures", "attention_bias", "mlp_bias", "rope_parameters", "rope_scaling",
            "qkv_bias", "attn_logit_softcapping", "final_logit_softcapping", "sliding_window", "hidden_act", "hidden_activation",
            # [SHARED-CODE lane cov-ln, additive] the LayerNorm / GPT-NeoX family's keys (absent on every other config, so their
            # records are unchanged): eps and rotary base under their names, the partial-rotary fraction, the parallel residual
            "layer_norm_eps", "layer_norm_epsilon", "rotary_emb_base", "rotary_pct", "partial_rotary_factor", "use_parallel_residual")
    out = {}
    for k in keys:
        v = getattr(hf, k, None)
        if v is not None:
            out[k] = v if isinstance(v, (int, float, str, bool, list)) else str(v)
    return out


def sampling_params(req: dict, sampling: dict) -> Any:
    from vllm import SamplingParams

    seed = req.get("seed") if req.get("seed") is not None else sampling.get("seed")     # [GEN sampling] per-request seed / n
    # [ov-sampling] a request with its own variant (mixed batch) carries temperature / top_p / top_k; else the workload's
    return SamplingParams(temperature=float(req.get("temperature", sampling.get("temperature", 0.0))),
                          top_p=float(req.get("top_p", sampling.get("top_p", 1.0))),
                          top_k=int(req.get("top_k", sampling.get("top_k", 0))), max_tokens=int(req["max_tokens"]),
                          seed=None if seed is None else int(seed), n=int(req.get("n") or sampling.get("n") or 1),
                          logprobs=sampling.get("logprobs"))


def run_requests(engine: Engine, requests: list[dict], sampling: dict) -> tuple[dict[str, list[int]], dict]:
    """Add every request with *our* request ids and step the engine until all finish (driver.run_requests; with declared
    `arrive_step`s, `run_requests_arrivals`).  `timing["requests"]` carries the realized per-request steps (D87)."""
    class _R:
        def __init__(self, d: dict, order: int) -> None:
            self.request_id = d["request_id"]
            self.prompt_token_ids = list(d["prompt_token_ids"])
            self.max_tokens = int(d["max_tokens"])
            self.seed = d.get("seed")
            self.n = int(d.get("n") or 1)
            self.arrive_step = int(d.get("arrive_step") or 0)
            self._order = order
            # [ov-sampling] per-request variant (mixed batch): the request's own temperature / top_p / top_k, if pinned
            self.params = {k: d[k] for k in ("temperature", "top_p", "top_k") if k in d}

    # [GEN sampling] n > 1: vLLM forks child requests f"{index}_{request_id}" (v1/engine/parallel_sampling.py) — the ids the model
    # runner and the log see.  With the default CUMULATIVE output kind each step's RequestOutput carries only the children that produced
    # a token that step (the final one need not hold every child), so collect every CompletionOutput by (parent, index) across steps.
    by_child: dict[str, dict[int, list[int]]] = {}
    finish: dict[str, dict] = {}
    step_no = [0]
    # [chunked 2026-09-17, additive] prefill-chunk attribution: the in-process scheduler's per-request `num_computed_tokens` is read
    # after every step() (pure reads; nothing patched, nothing on the device), the returned tokens are counted per iteration; after
    # the run `chunk_attribution.attribute` adds prefill_steps / prefill_chunks / partial_prefill_steps / first_token_step /
    # token_steps to every `timing["requests"][rid]` and `timing["chunk_attribution"]` run-wide.  Every existing key is unchanged.
    chunks = _chunks.Collector(_chunks.scheduler_of(engine.llm))

    def _collect(outs) -> None:
        for o in outs:
            for c in o.outputs:
                by_child.setdefault(o.request_id, {})[int(c.index)] = [int(t) for t in c.token_ids]
            if o.finished and o.request_id not in finish:
                fr = [getattr(c, "finish_reason", None) for c in o.outputs]
                finish[o.request_id] = {"finish_step": step_no[0], "finish_reason": str(fr[0]) if len(fr) == 1 else [str(x) for x in fr]}
        step_no[0] += 1
        chunks.on_step(outs)
    rs = [_R(r, i) for i, r in enumerate(requests)]
    params_for = lambda r: sampling_params({"max_tokens": r.max_tokens, "seed": r.seed, "n": r.n, **r.params}, sampling)  # noqa: E731
    # [D87] one loop for every profile: with all arrive_steps 0 `run_requests_arrivals` is the driver's loop, and its `engine_steps`
    # counts forwards (the observer's / gates' unit) rather than step() iterations (vLLM V1 returns the previous forward's sample, so
    # iterations = forwards + 1); verity_vllm_adapter.driver.run_requests stays available for callers that want the raw iteration count
    outs, timing = run_requests_arrivals(engine.llm, rs, params_for, on_step=_collect)
    timing["arrivals"] = any(r.arrive_step > 0 for r in rs)
    try:
        # [rev F-r17-16 / R17-1] the attribution receives the RESOLVED async_scheduling as the lag of record (the same fact tokens.json's
        # `lag.value` and the RunHeader attest): a schedule-delta inference that contradicts it never rewrites the loop's step fields
        _asched = resolved_async_scheduling(engine.llm)
        _chunks.attribute(timing, chunks, {r.request_id: len(r.prompt_token_ids) for r in rs},
                          resolved_lag=None if _asched is None else (1 if _asched else 0))
    except Exception as e:  # noqa: BLE001  -- attribution is a record, never a reason to lose the run
        timing["chunk_attribution"] = {"source": chunks.source, "error": repr(e)[:300], "consistent": False}
    tokens = {}
    for rid, o in outs.items():
        kids = by_child.get(rid) or {0: [int(t) for t in o.outputs[0].token_ids]}
        tokens[rid] = kids[min(kids)]                    # the parent key keeps child 0 (n == 1: the request itself; unchanged shape)
        if len(kids) > 1:
            for i, toks in sorted(kids.items()):
                tokens[f"{i}_{rid}"] = toks
    return tokens, timing


def throwaway_requests(engine: Engine, like: list[dict], seed: int = 0x5EED_7A9B) -> list[dict]:
    """Warm-up prompts of the same lengths, random tokens (prefix caching is off; they share no block anyway)."""
    vocab = int(getattr(engine.llm.llm_engine.vllm_config.model_config.hf_config, "vocab_size", 32000))
    rng = np.random.default_rng(seed)
    out = []
    for i, r in enumerate(like):
        out.append({"request_id": f"warmup-{i}", "prompt_token_ids": rng.integers(3, vocab - 1, size=len(r["prompt_token_ids"])).tolist(),
                    "max_tokens": int(r["max_tokens"]), "arrive_step": int(r.get("arrive_step") or 0)})   # same schedule shape (D87)
    return out


# ---------------------------------------------------------------------------------------------------------
# the capture
# ---------------------------------------------------------------------------------------------------------


@dataclass
class ObserverConfig:
    mode: str = "record"                 # "none" (baseline timing only) | "empty" (EmptyMode) | "record"
    reenter: bool = True
    triton: bool = True
    module_hooks: bool = True
    write_log: bool = True               # False: build every event, discard the buffer at step end (M0 rungs c-f)
    snapshot_steps: frozenset = frozenset()
    snapshot_dir: Path | None = None
    snapshot_max_bytes: int = 32 << 20
    snapshot_max_bytes_op: int | None = None   # cap at dispatcher-call events (None: snapshot_max_bytes); Triton launches keep the full cap
    arena_bytes: int = 1 << 30


@dataclass
class _Step:
    step: int
    events: list[Event] = field(default_factory=list)
    forward_s: float = 0.0
    sample_s: float = 0.0
    t_wall0: float = 0.0
    wall_s: float = 0.0
    facts: dict = field(default_factory=dict)
    input_ids_view: ViewDesc | None = None
    positions_view: ViewDesc | None = None
    sampled_view: ViewDesc | None = None
    logits_view: ViewDesc | None = None
    sampled_shape: tuple[int, int] = (0, 0)
    event: Any = None
    n_snap: int = 0


class PinnedArena:
    """Bump allocator over one pinned host buffer (reset per step); overflow falls back to fresh pinned tensors."""

    def __init__(self, nbytes: int) -> None:
        self.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.off = 0
        self.overflow = 0

    def reset(self) -> None:
        self.off = 0

    def take(self, nbytes: int) -> torch.Tensor:
        aligned = (self.off + 255) // 256 * 256
        if aligned + nbytes > self.buf.numel():
            self.overflow += 1
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        v = self.buf[aligned:aligned + nbytes]
        self.off = aligned + nbytes
        return v


def spanned_bytes(t: torch.Tensor) -> tuple[int, int]:
    """(byte offset in the storage, byte length) of the storage range a view can touch."""
    es = t.element_size()
    if t.numel() == 0:
        return int(t.storage_offset()) * es, 0
    last = sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
    return int(t.storage_offset()) * es, (int(last) + 1) * es


def storage_bytes_view(t: torch.Tensor, byte_off: int, nbytes: int) -> torch.Tensor:
    """A contiguous uint8 view over `[byte_off, byte_off + nbytes)` of `t`'s storage (no device allocation)."""
    v = torch.empty(0, dtype=torch.uint8, device=t.device)
    v.set_(t.untyped_storage(), byte_off, (nbytes,), (1,))
    return v


class Capture:
    """Install with `install()`, run the workload, `close()`.  One instance per run."""

    def __init__(self, engine: Engine, *, header: RunHeader | None, log_path: str | Path | None,
                 config: ObserverConfig | None = None, log: Callable[..., None] = print) -> None:
        self.engine = engine
        self.runner = engine.runner
        self.model = engine.model
        self.cfg = config or ObserverConfig()
        self.log = log
        self.header = header
        self.storage = StorageAdapter()
        self.writer: LogWriter | None = LogWriter(log_path) if (self.cfg.write_log and log_path is not None) else None
        self.rec = Recorder(self.storage, self._sink)
        self.mode = Observer(self.rec, reenter=self.cfg.reenter)
        self.namer: ModuleNamer | None = None
        self.root_allocs: set[int] = set()
        self.roots: list[Root] = []
        self.step_index = 0
        self._cur: _Step | None = None
        self._pending: _Step | None = None
        self._pending_token: dict[str, tuple[int, int, int]] = {}   # request -> (src_step, token, position it feeds)
        self.step_times: list[dict] = []
        self.sample_ids: dict[int, list[int]] = {}                   # step -> sampled ids
        self.step_rows: dict[int, list[tuple[str, int]]] = {}
        self.errors: list[dict] = []
        self.events_written = 0
        self.timing = {"flush_s": 0.0, "snapshot_s": 0.0, "sync_s": 0.0}
        self._orig: dict[str, Any] = {}
        self._had_attr: dict[str, bool] = {}
        self._pin_cache: dict[str, torch.Tensor] = {}
        self._root_handles: list = []
        self._arena: PinnedArena | None = None
        self._snap_pending: list[dict] = []
        self._snap_index: list[dict] = []
        self._snap_bytes = 0
        self._snap_files: dict[str, str] = {}      # sha256 -> the file that holds these bytes (content-addressed across steps)
        self._kv_num_blocks: int | None = None
        self._kv_blocks_in_step: list[int] = []
        self.installed = False

    # ------------------------------------------------------------------ sinks

    def _sink(self, ev: Event) -> None:
        st = self._cur
        if st is not None:
            st.events.append(ev)
        elif self.writer is not None:
            self.writer.write(ev)
            self.events_written += 1

    def _write(self, ev: Event) -> None:
        if self.writer is not None:
            self.writer.write(ev)
            self.events_written += 1

    # ------------------------------------------------------------------ install / uninstall

    def install(self) -> None:
        if self.installed:
            return
        runner, model, cfg = self.runner, self.model, self.cfg
        if self.header is not None:
            self._write(self.header)
        self._install_roots()
        self._install_external_buffers()
        if cfg.mode == "record":
            if cfg.module_hooks:
                self.namer = ModuleNamer(self.rec, model)
                self._install_root_phase_hooks()
            if cfg.triton:
                triton_adapter.install(self.rec)
            if cfg.snapshot_steps:
                self.rec.after_call = self._after_call
                self.rec.after_launch = self._after_launch
        self._wrap_runner()
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        self._finalize_pending()
        for name in ("execute_model", "sample", "compute_logits"):
            if name not in self._orig:
                continue
            target = self.model if name == "compute_logits" else self.runner
            if self._had_attr.get(name):
                setattr(target, name, self._orig[name])
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
        if self.namer is not None:
            self.namer.remove()
            self.namer = None
        for h in getattr(self, "_root_handles", []):
            h.remove()
        self._root_handles = []
        if triton_adapter.is_installed():
            triton_adapter.set_recorder(None)
        self.rec.after_call = None
        self.rec.after_launch = None
        self.installed = False

    def close(self) -> dict:
        self.uninstall()
        stats = self.stats()
        if self.writer is not None:
            self._write(RunEnd(stats))
            self.writer.close()
        if self.cfg.snapshot_dir is not None and self.cfg.snapshot_steps:
            self._write_snapshot_index()
        return stats

    def stats(self) -> dict:
        return {"steps": self.step_index, "events_written": self.events_written, "recorder": dict(self.rec.counts),
                "encode_s": self.rec.encode_s, "storage": dict(self.storage.stats), "timing": dict(self.timing),
                "snapshot_bytes": self._snap_bytes, "snapshot_files": len(self._snap_files), "errors": self.errors[:20],
                "roots": len(self.roots), "step_times": self.step_times}

    def _install_roots(self) -> None:
        seen: set[int] = set()
        for role, items, origin in (("weight", list(self.model.named_parameters()), "param"),
                                    ("buffer", list(self.model.named_buffers()), "buffer")):
            for name, t in items:
                if not isinstance(t, torch.Tensor) or id(t) in seen:
                    continue
                seen.add(id(t))
                view = self.storage.view_of(t, origin=origin)
                self.root_allocs.add(view.alloc)
                for e in self.storage.events_pending():
                    self._write(e)
                r = Root(view, role, name)
                self.roots.append(r)
                self._write(r)

    def _install_external_buffers(self) -> None:
        named: dict[str, dict] = {}
        ib = getattr(self.runner, "input_buffers", None)
        for attr in ("input_ids", "positions"):
            t = getattr(ib, attr, None)
            if isinstance(t, torch.Tensor) and t.numel():
                v = self.storage.view_of(t, origin="external")
                named[f"input_buffers.{attr}"] = {"alloc": v.alloc, "shape": list(v.shape), "dtype": v.dtype}
        kvs = getattr(self.runner, "kv_caches", None) or []
        layer_of_alloc: dict[int, list[str]] = {}
        for name, mod in self.model.named_modules():
            kc = getattr(mod, "kv_cache", None)
            if isinstance(kc, torch.Tensor) and kc.numel():
                v = self.storage.view_of(kc, origin="external")
                layer_of_alloc.setdefault(v.alloc, []).append(name)
                named[f"kv_cache[{name}]"] = {"alloc": v.alloc, "shape": list(v.shape), "dtype": v.dtype, "offset": v.offset,
                                              "strides": list(v.strides)}
        for i, kc in enumerate(kvs):
            if isinstance(kc, torch.Tensor) and kc.numel():
                v = self.storage.view_of(kc, origin="external")
                named.setdefault(f"runner.kv_caches[{i}]", {"alloc": v.alloc, "shape": list(v.shape), "dtype": v.dtype})
        try:
            self._kv_num_blocks = int(self.runner.kv_cache_config.num_blocks)
        except Exception:  # noqa: BLE001
            self._kv_num_blocks = None
        for e in self.storage.events_pending():
            self._write(e)
        try:
            block_size = int(self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size)
        except Exception:  # noqa: BLE001
            block_size = None
        try:  # [r8 E-cleanup, hybrid KV cache] the KV-cache groups (layer names, block size, spec kind) for the record
            groups_note = [{"layers": [str(x) for x in g.layer_names], "block_size": int(g.kv_cache_spec.block_size),
                            "spec": type(g.kv_cache_spec).__name__} for g in self.runner.kv_cache_config.kv_cache_groups]
        except Exception:  # noqa: BLE001
            groups_note = None
        self._write(Note("external buffers (not roots): runner input buffers and KV caches",
                         {"buffers": named, "kv_num_blocks": self._kv_num_blocks, "block_size": block_size,
                          "kv_cache_groups": groups_note, "storage_weakref_frees": self.storage.use_weakrefs}))

    def _install_root_phase_hooks(self) -> None:
        rec = self.rec

        def pre(mod, args):
            if rec.active and rec.phase == "other":
                rec.phase = "forward"
                rec._root_forward = True

        def post(mod, args, out):
            if rec.active and getattr(rec, "_root_forward", False):
                rec.phase = "other"
                rec._root_forward = False

        self._root_handles = [self.model.register_forward_pre_hook(pre), self.model.register_forward_hook(post)]

    def _wrap_runner(self) -> None:
        runner, cap = self.runner, self
        for name in ("execute_model", "sample"):
            self._had_attr[name] = name in runner.__dict__
            self._orig[name] = getattr(runner, name)
        orig_execute, orig_sample = self._orig["execute_model"], self._orig["sample"]

        def execute_model(scheduler_output, *a, **kw):
            dummy = bool(kw.get("dummy_run", False)) or (len(a) >= 2 and bool(a[1]))
            if dummy or getattr(scheduler_output, "total_num_scheduled_tokens", 0) == 0:
                return orig_execute(scheduler_output, *a, **kw)
            cap._finalize_pending()
            st = cap._begin_step(scheduler_output)
            t0 = time.perf_counter()
            try:
                out = cap._run_observed(lambda: orig_execute(scheduler_output, *a, **kw), phase="forward")
            finally:
                st.forward_s = time.perf_counter() - t0
            if cap.cfg.mode == "record":
                cap._after_forward(st)
            return out

        def sample(hidden_states, input_batch, grammar_output, *a, **kw):
            st = cap._cur
            if st is None:
                return orig_sample(hidden_states, input_batch, grammar_output, *a, **kw)
            t0 = time.perf_counter()
            try:
                res = cap._run_observed(lambda: orig_sample(hidden_states, input_batch, grammar_output, *a, **kw), phase="sample")
            finally:
                st.sample_s = time.perf_counter() - t0
            if cap.cfg.mode == "record":
                cap._after_sample(st, res, input_batch)
            else:
                cap._end_timing_only(st)
            return res

        runner.execute_model = execute_model
        runner.sample = sample

        # the logits tensor is the return value of model.compute_logits (V2 runner: sample() -> compute_logits ->
        # sampler); record its ViewDesc only (no tensor reference is kept)
        model = self.model
        if hasattr(model, "compute_logits"):
            self._had_attr["compute_logits"] = "compute_logits" in model.__dict__
            self._orig["compute_logits"] = model.compute_logits
            orig_cl = self._orig["compute_logits"]

            def compute_logits(*a, **kw):
                out = orig_cl(*a, **kw)
                st = cap._cur
                if st is not None and cap.cfg.mode == "record" and isinstance(out, torch.Tensor):
                    try:
                        st.logits_view = cap.storage.view_of(out)
                    except Exception as e:  # noqa: BLE001
                        cap.errors.append({"where": "compute_logits", "step": st.step, "error": repr(e)[:200]})
                return out

            model.compute_logits = compute_logits

    def _run_observed(self, fn: Callable[[], Any], *, phase: str) -> Any:
        cfg, rec = self.cfg, self.rec
        if cfg.mode == "none":
            return fn()
        if cfg.mode == "empty":
            with EmptyMode():
                return fn()
        rec.active = True
        rec.phase = ("other" if (phase == "forward" and self.namer is not None) else phase)
        rec._root_forward = False
        rec.open.clear()
        rec.module_stack.clear()
        try:
            with self.mode:
                return fn()
        finally:
            rec.active = False
            rec.phase = "other"

    # ------------------------------------------------------------------ step lifecycle

    def _begin_step(self, scheduler_output: Any) -> _Step:
        st = _Step(self.step_index)
        st.t_wall0 = time.perf_counter()
        st.facts["sched_req_ids"] = list(getattr(scheduler_output, "num_scheduled_tokens", {}).keys())
        self.rec.step = st.step
        self._cur = st
        if st.step in self.cfg.snapshot_steps and self.cfg.mode == "record":
            if self._arena is None:
                self._arena = PinnedArena(self.cfg.arena_bytes)
            self._arena.reset()
            self._snap_pending = []
            self._kv_blocks_in_step = []
        return st

    def _pin(self, name: str, n: int, dtype: torch.dtype, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        """A (cached, grow-only) pinned host buffer for the async D2H copies of the step facts."""
        shape = shape or (n,)
        need = max(int(np.prod(shape)), 1)
        t = self._pin_cache.get(name)
        if t is None or t.numel() < need or t.dtype != dtype:
            t = torch.empty(need, dtype=dtype, pin_memory=True)
            self._pin_cache[name] = t
        return t[:need].view(*shape)

    def _after_forward(self, st: _Step) -> None:
        """Host-side facts + async D2H copies of the buffers the model read (stream-ordered after the forward)."""
        try:
            ems = self.runner.execute_model_state
            ib = ems.input_batch
            from verity_vllm_adapter.driver import external_id

            n = int(ib.num_reqs)
            qsl = np.asarray(ib.query_start_loc_np[: n + 1]).astype(np.int64).copy()
            nct = np.asarray(ib.num_computed_tokens_np[:n]).astype(np.int64).copy()
            idx = np.asarray(ib.idx_mapping_np[:n]).astype(np.int64).copy()
            num_tokens = int(ib.num_tokens)
            num_padded = int(ib.num_tokens_after_padding)
            req_ids = [external_id(r) for r in ib.req_ids[:n]]
            rs = self.runner.req_states
            prompt_len = {rid: int(rs.prompt_len.np[int(idx[i])]) for i, rid in enumerate(req_ids)}
            bt = self.runner.block_tables
            nblocks = [int(bt.num_blocks.np[0, int(idx[i])]) for i in range(n)] if bt.num_kv_cache_groups else [0] * n
            block_size = int(self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size)
            st.facts = {"n": n, "qsl": qsl, "nct": nct, "req_ids": req_ids, "num_tokens": num_tokens, "num_padded": num_padded,
                        "prompt_len": prompt_len, "nblocks": nblocks, "block_size": block_size,
                        "num_logits": int(ib.logits_indices.numel())}
            st.input_ids_view = self.storage.view_of(ib.input_ids)
            st.positions_view = self.storage.view_of(ib.positions)
            # async D2H of the tiny buffers the model actually read
            ids = self._pin("input_ids", num_tokens, ib.input_ids.dtype)
            ids.copy_(ib.input_ids[:num_tokens], non_blocking=True)
            pos = self._pin("positions", num_tokens, ib.positions.dtype)
            pos.copy_(ib.positions[:num_tokens], non_blocking=True)
            li = self._pin("logits_indices", int(ib.logits_indices.numel()), ib.logits_indices.dtype)
            li.copy_(ib.logits_indices, non_blocking=True)
            sm = ems.slot_mappings_by_layer
            slot_src = None
            if sm:
                first = next(iter(sm.values()))
                st.facts["slot_mapping_distinct"] = len({id(v) for v in sm.values()})
                st.facts["slot_mapping_layer"] = next(iter(sm.keys()))
                slot_src = first[:num_tokens]
                slots = self._pin("slots", num_tokens, first.dtype)
                slots.copy_(slot_src, non_blocking=True)
            if bt.num_kv_cache_groups and n:
                gathered = bt.input_block_tables[0]
                maxb = max(nblocks) if nblocks else 0
                if maxb:
                    bts = self._pin("block_tables", n * maxb, gathered.dtype, (n, maxb))
                    bts.copy_(gathered[:n, :maxb], non_blocking=True)
            # [r8 E-cleanup, hybrid KV cache] several KV-cache groups (vLLM hybrid KV-cache manager; Gemma-2: sliding-window
            # layers vs global layers): every group has its own block tables and slot mapping into the SAME physical KV
            # tensors, and the fields above are group 0's only.  Record every group (+ its layer names, so the fold can
            # address a layer's KV rows by ITS group).  Single-group runs record nothing extra (byte-identical logs).
            ngroups = int(bt.num_kv_cache_groups or 0)
            if ngroups > 1 and n:
                try:
                    groups_meta = []
                    kv_groups = self.runner.kv_cache_config.kv_cache_groups
                    for g in range(ngroups):
                        layers = [str(x) for x in getattr(kv_groups[g], "layer_names", [])]
                        nb_g = [int(bt.num_blocks.np[g, int(idx[i])]) for i in range(n)]
                        maxb_g = max(nb_g) if nb_g else 0
                        if maxb_g:
                            gathered_g = bt.input_block_tables[g]
                            bts_g = self._pin(f"block_tables_g{g}", n * maxb_g, gathered_g.dtype, (n, maxb_g))
                            bts_g.copy_(gathered_g[:n, :maxb_g], non_blocking=True)
                        sm_layer = next((ln for ln in layers if sm and ln in sm), None)
                        if sm_layer is not None:
                            slots_g = self._pin(f"slots_g{g}", num_tokens, sm[sm_layer].dtype)
                            slots_g.copy_(sm[sm_layer][:num_tokens], non_blocking=True)
                        groups_meta.append({"layers": layers, "block_size": int(kv_groups[g].kv_cache_spec.block_size),
                                            "nblocks": nb_g, "slot_layer": sm_layer})
                    st.facts["kv_groups"] = groups_meta
                except Exception as e:  # noqa: BLE001 - recorded, never silent: the StepStart then carries `ngroups` EMPTY
                    # groups, so every KV access of the step is Unsupported at resolve (REJECTED) instead of mis-addressed
                    self.errors.append({"where": "after_forward.kv_groups", "step": st.step, "error": repr(e)[:400]})
                    st.facts["kv_groups"] = [{"layers": [], "block_size": block_size, "nblocks": [0] * n, "slot_layer": None,
                                              "error": repr(e)[:400]} for _ in range(ngroups)]
            st.event = torch.cuda.Event()
            st.event.record()
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "after_forward", "step": st.step, "error": repr(e)[:400]})
            st.facts.setdefault("error", repr(e)[:400])

    def _after_sample(self, st: _Step, res: Any, input_batch: Any) -> None:
        try:
            so = res[0] if isinstance(res, tuple) else res
            sampled = so.sampled_token_ids
            st.sampled_view = self.storage.view_of(sampled)
            s2 = sampled.reshape(sampled.shape[0], -1) if sampled.dim() != 2 else sampled
            st.sampled_shape = (int(s2.shape[0]), int(s2.shape[1]))
            host = self._pin("sampled", int(s2.numel()), s2.dtype, st.sampled_shape)
            host.copy_(s2, non_blocking=True)
            if st.event is None:
                st.event = torch.cuda.Event()
            st.event.record()
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "after_sample", "step": st.step, "error": repr(e)[:400]})
        st.wall_s = time.perf_counter() - st.t_wall0
        if st.step in self.cfg.snapshot_steps and self.cfg.mode == "record":
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            self.timing["sync_s"] += time.perf_counter() - t0
            self._flush_snapshots(st)
        # the step is complete; it is written out at the next step start (or at close)
        self._cur = None
        self._pending = st
        self.step_index += 1
        # non-snapshot steps: the deferred write happens at the next execute_model; keep the buffer

    def _end_timing_only(self, st: _Step) -> None:
        """M0 rungs (a)/(b): no facts, no events; only the per-step timing."""
        st.wall_s = time.perf_counter() - st.t_wall0
        self.step_times.append({"step": st.step, "forward_s": st.forward_s, "sample_s": st.sample_s, "wall_s": st.wall_s,
                                "num_reqs": len(st.facts.get("sched_req_ids", [])), "events": 0, "snapshots": 0})
        self._cur = None
        self.step_index += 1

    def _finalize_pending(self) -> None:
        st = self._pending
        if st is None:
            return
        self._pending = None
        t0 = time.perf_counter()
        try:
            if st.event is not None:
                st.event.synchronize()
            self._emit_step(st)
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "finalize", "step": st.step, "error": repr(e)[:400]})
            # still write what we have so the log stays parseable
            self._write(Note("step finalisation failed", {"step": st.step, "error": repr(e)[:400]}))
            for ev in st.events:
                self._write(ev)
            self._write(StepEnd(st.step, st.forward_s, st.sample_s))
        st.events = []
        if self.writer is not None:
            self.writer.flush()
        self.timing["flush_s"] += time.perf_counter() - t0

    def _emit_step(self, st: _Step) -> None:
        f = st.facts
        if "n" not in f:
            raise RuntimeError(f"no step facts: {f.get('error')}")
        n, qsl, nct, req_ids = f["n"], f["qsl"], f["nct"], f["req_ids"]
        num_tokens = f["num_tokens"]
        ids = self._pin_cache["input_ids"][:num_tokens].tolist()
        pos_dev = self._pin_cache["positions"][:num_tokens].tolist()
        rows: list[tuple[str, int]] = []
        positions: list[int] = []
        for i in range(n):
            for k in range(int(qsl[i]), int(qsl[i + 1])):
                rows.append((req_ids[i], int(nct[i] + (k - qsl[i]))))
                positions.append(int(nct[i] + (k - qsl[i])))
        if pos_dev != positions:
            st.events.insert(0, Note("device positions differ from host-derived positions", {"step": st.step, "device": pos_dev[:8], "host": positions[:8]}))
            positions = [int(p) for p in pos_dev]
            rows = [(r, p) for (r, _), p in zip(rows, positions)]
        seq_lens = {req_ids[i]: int(nct[i] + (qsl[i + 1] - qsl[i])) for i in range(n)}
        num_computed = {req_ids[i]: int(nct[i]) for i in range(n)}
        nblocks = f["nblocks"]
        block_tables: dict[str, list[int]] = {}
        if max(nblocks, default=0) and "block_tables" in self._pin_cache:
            bt_host = self._pin_cache["block_tables"][: n * max(nblocks)].view(n, max(nblocks)).tolist()
            for i in range(n):
                block_tables[req_ids[i]] = [int(b) for b in bt_host[i][: nblocks[i]]]
        else:
            block_tables = {rid: [] for rid in req_ids}
        slot_mapping = self._pin_cache["slots"][:num_tokens].tolist() if "slots" in self._pin_cache and "slot_mapping_layer" in f else []
        nl = f["num_logits"]
        logits_indices = [int(x) for x in self._pin_cache["logits_indices"][:nl].tolist()]
        logits_rows = [(req_ids[i], int(nct[i] + (qsl[i + 1] - qsl[i]) - 1)) for i in range(n)]
        if len(logits_indices) != len(logits_rows):
            logits_rows = [rows[j] if 0 <= j < len(rows) else ("?", -1) for j in logits_indices]
        prompt_len = f["prompt_len"]
        phases = {"prefill" if p < prompt_len.get(r, 0) else "decode" for r, p in rows}
        phase_hint = phases.pop() if len(phases) == 1 else "mixed"
        kv_groups: list[dict] | None = None
        if f.get("kv_groups"):
            # [r8 E-cleanup, hybrid KV cache] one entry per KV-cache group (see `_after_forward`)
            kv_groups = []
            for g, meta in enumerate(f["kv_groups"]):
                nb_g = meta["nblocks"]
                tables: dict[str, list[int]] = {rid: [] for rid in req_ids}
                key = f"block_tables_g{g}"
                if max(nb_g, default=0) and key in self._pin_cache:
                    host = self._pin_cache[key][: n * max(nb_g)].view(n, max(nb_g)).tolist()
                    for i in range(n):
                        tables[req_ids[i]] = [int(b) for b in host[i][: nb_g[i]]]
                skey = f"slots_g{g}"
                sm_g = [int(s) for s in self._pin_cache[skey][:num_tokens].tolist()] if meta.get("slot_layer") and skey in self._pin_cache else []
                entry = {"layers": list(meta["layers"]), "block_size": int(meta["block_size"]), "block_tables": tables,
                         "slot_mapping": sm_g}
                if meta.get("error"):
                    entry["error"] = meta["error"]
                kv_groups.append(entry)
        start = StepStart(st.step, rows, [int(t) for t in ids], positions, seq_lens, num_computed, block_tables, f["block_size"],
                          [int(s) for s in slot_mapping], logits_rows, logits_indices, n, num_tokens, f["num_padded"],
                          st.input_ids_view, st.positions_view, phase_hint, kv_groups)
        # roots (prompt rows) and links (decode rows)
        roots: list[Root] = []
        links: list[StepLink] = []
        iv = st.input_ids_view
        for row, ((rid, p), tok) in enumerate(zip(rows, ids)):
            if p < prompt_len.get(rid, 0):
                if iv is not None:
                    rv = ViewDesc(iv.alloc, iv.offset + row * (iv.strides[0] if iv.strides else 1), (1,), (1,), iv.dtype, iv.device)
                    roots.append(Root(rv, "prompt_token", f"prompt[{rid}][{p}]", [int(tok)]))
            else:
                src = self._pending_token.get(rid)
                if src is not None and src[2] == p:
                    links.append(StepLink(st.step, row, rid, src[0], "eq", int(src[1])))
                    if int(src[1]) != int(tok):
                        st.events.insert(0, Note("StepLink value != input token", {"step": st.step, "row": row, "request_id": rid,
                                                                                  "link_value": int(src[1]), "input": int(tok)}))
                else:
                    st.events.insert(0, Note("decode row without a known source step", {"step": st.step, "row": row, "request_id": rid,
                                                                                         "position": p, "known": None if src is None else list(src)}))
        # sampled ids
        sampled_ids: list[int] = []
        if st.sampled_shape[0]:
            host = self._pin_cache["sampled"][: st.sampled_shape[0] * st.sampled_shape[1]].view(*st.sampled_shape)
            sampled_ids = [int(v) for v in host[:, 0].tolist()]
        sample_end = SampleEnd(st.step, st.logits_view, sampled_ids, st.sampled_view)
        for i, (rid, p) in enumerate(logits_rows):
            if i < len(sampled_ids):
                self._pending_token[rid] = (st.step, sampled_ids[i], p + 1)
        self.sample_ids[st.step] = sampled_ids
        self.step_rows[st.step] = rows
        # write: pending StorageAllocs (views referenced by StepStart / SampleEnd), StepStart, roots, links, the
        # buffered events, then the pending StorageFrees (deaths of this step's temporaries observed after the
        # step's events were buffered -- they must follow the StorageAllocs inside the buffer), SampleEnd, StepEnd
        pending = self.storage.events_pending()
        for e in pending:
            if isinstance(e, StorageAlloc):
                self._write(e)
        self._write(start)
        for r in roots:
            self._write(r)
        for l in links:
            self._write(l)
        for ev in st.events:
            self._write(ev)
        for e in pending:
            if not isinstance(e, StorageAlloc):
                self._write(e)
        self._write(sample_end)
        self._write(StepEnd(st.step, st.forward_s, st.sample_s))
        self.step_times.append({"step": st.step, "num_tokens": num_tokens, "num_reqs": n, "forward_s": st.forward_s,
                                "sample_s": st.sample_s, "wall_s": st.wall_s, "events": len(st.events), "phase": phase_hint,
                                "snapshots": st.n_snap})

    # ------------------------------------------------------------------ snapshots

    def _after_call(self, eid: int, op: str, args: tuple, kwargs: dict, out: Any) -> None:
        st = self._cur
        if st is None or st.step not in self.cfg.snapshot_steps:
            return
        try:
            from torch.utils._python_dispatch import _disable_current_modes

            cap = self.cfg.snapshot_max_bytes if self.cfg.snapshot_max_bytes_op is None else self.cfg.snapshot_max_bytes_op
            with _disable_current_modes():
                self._snap_struct(eid, "args", args, cap)
                for k, v in kwargs.items():
                    self._snap_struct(eid, f"kwargs.{k}", v, cap)
                self._snap_struct(eid, "outs", out, cap)
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "snapshot", "step": st.step, "id": eid, "error": repr(e)[:300]})

    def _after_launch(self, eid: int, fn: str, bound: dict) -> None:
        st = self._cur
        if st is None or st.step not in self.cfg.snapshot_steps:
            return
        try:
            from torch.utils._python_dispatch import _disable_current_modes

            with _disable_current_modes():
                for k, v in bound.items():
                    if isinstance(v, torch.Tensor):
                        self._snap_tensor(eid, f"args.{k}", v)
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "snapshot-triton", "step": st.step, "id": eid, "error": repr(e)[:300]})

    def _snap_struct(self, eid: int, path: str, x: Any, cap: int | None = None) -> None:
        if isinstance(x, torch.Tensor):
            self._snap_tensor(eid, path, x, cap=cap)
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                if isinstance(v, (torch.Tensor, list, tuple)):
                    self._snap_struct(eid, f"{path}[{i}]", v, cap)

    def _snap_tensor(self, eid: int, path: str, t: torch.Tensor, sub: dict | None = None, cap: int | None = None) -> None:
        if t.numel() == 0:
            return
        view = self.storage.view_of(t)
        if view.alloc in self.root_allocs:
            return
        full = self.cfg.snapshot_max_bytes
        if cap is None:
            cap = full
        off, nbytes = spanned_bytes(t)
        if nbytes > cap:
            nb = self._kv_num_blocks
            bdim = next((d for d, s in enumerate(t.shape) if nb and s == nb), None)
            if bdim is not None and sub is None:
                blocks = self._blocks_in_use()
                lead = [()] if bdim == 0 else [(k,) for k in range(int(t.shape[0]))] if bdim == 1 else None
                if lead is not None and blocks:
                    for l in lead:
                        for b in blocks:
                            v = t
                            for k in l:
                                v = v.select(0, k)
                            v = v.select(0, b)
                            self._snap_tensor(eid, path, v, sub={"dim": bdim, "index": list(l) + [b]}, cap=cap)
                    return
            why = f"spanned bytes {nbytes} > snapshot_max_bytes {cap}"
            if cap != full:
                why += f" (snapshot_max_bytes_op: dispatcher-call event; Triton launch events keep {full})"
            self._snap_index.append({"event": eid, "path": path, "view": view.to_json(), "dtype": view.dtype, "shape": list(view.shape),
                                     "skipped": why})
            return
        src = storage_bytes_view(t, off, nbytes)
        host = self._arena.take(nbytes)
        host.copy_(src, non_blocking=True)
        entry = {"event": eid, "path": path, "view": view.to_json(), "dtype": view.dtype, "shape": list(view.shape),
                 "strides": list(view.strides), "byte_offset": off, "nbytes": nbytes, "layout": "spanned-storage-bytes"}
        if sub is not None:
            entry["sub"] = sub
        self._snap_pending.append({"host": host, "entry": entry})
        if self._cur is not None:
            self._cur.n_snap += 1

    def _blocks_in_use(self) -> list[int]:
        """The physical KV blocks of every request scheduled in the current step (read once per snapshot step from the
        runner's block table, whose staged writes are applied at the start of execute_model; this read synchronises,
        which is allowed in snapshot steps only)."""
        if self._kv_blocks_in_step or self._cur is None:
            return self._kv_blocks_in_step
        try:
            bt = self.runner.block_tables
            rs = self.runner.req_states
            blocks: set[int] = set()
            # [r8 E-cleanup, hybrid KV cache] the blocks of EVERY KV-cache group: with several groups the physical tensors
            # are shared and a group-1 layer's rows live in group-1 blocks (group 0 alone left them un-snapshotted)
            for g in range(int(bt.num_kv_cache_groups or 1)):
                for rid in self._cur.facts.get("sched_req_ids", []):
                    ri = rs.req_id_to_index.get(rid)
                    if ri is None:
                        continue
                    count = int(bt.num_blocks.np[g, ri])
                    if count:
                        blocks.update(int(b) for b in bt.block_tables[g].gpu[ri, :count].cpu().tolist())
            self._kv_blocks_in_step = sorted(blocks)
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "blocks_in_use", "error": repr(e)[:300]})
            self._kv_blocks_in_step = []
        return self._kv_blocks_in_step

    def _flush_snapshots(self, st: _Step) -> None:
        t0 = time.perf_counter()
        d = Path(self.cfg.snapshot_dir) / "values" / str(st.step)
        d.mkdir(parents=True, exist_ok=True)
        for item in self._snap_pending:
            host: torch.Tensor = item["host"]
            entry = item["entry"]
            data = host.numpy().tobytes()
            sha = hashlib.sha256(data).hexdigest()
            # content-addressed across steps: identical bytes seen at an earlier step share that step's file.  (Before the
            # generalization sweep the entry pointed at values/<this step>/ even when the bytes had been written under an
            # earlier step: 4430 of 16334 index entries of a 16-step SmolLM2 capture named files that did not exist.)
            fname = self._snap_files.get(sha)
            if fname is None:
                fname = f"values/{st.step}/{sha[:16]}.bin"
                (Path(self.cfg.snapshot_dir) / fname).write_bytes(data)
                self._snap_files[sha] = fname
                self._snap_bytes += len(data)
            entry["sha256"] = sha
            entry["file"] = fname
            entry["step"] = st.step
            self._snap_index.append(entry)
        self._snap_pending = []
        self._arena.reset()
        self.timing["snapshot_s"] += time.perf_counter() - t0

    def _write_snapshot_index(self) -> None:
        p = Path(self.cfg.snapshot_dir) / "snapshots.json"
        p.write_text(json.dumps({"schema": "verity-capture/snapshots/v1", "layout": "spanned-storage-bytes",
                                 "note": "file = raw bytes of the storage range [byte_offset, byte_offset+nbytes) the view spans, "
                                         "copied D2H stream-ordered right after the call; content-addressed (sha256) so identical "
                                         "bytes share a file; 'sub' = sub-view (dim, index) of an oversize KV-cache tensor",
                                 "steps": sorted(self.cfg.snapshot_steps), "total_bytes": self._snap_bytes,
                                 "snapshot_max_bytes": self.cfg.snapshot_max_bytes, "snapshot_max_bytes_op": self.cfg.snapshot_max_bytes_op,
                                 "entries": self._snap_index}, separators=(",", ":")))


# ---------------------------------------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------------------------------------


def make_header(engine: Engine, workload: dict, *, adapter_revision: str, wheel_sha256: str | None = None,
                vllm_commit: str | None = None, extra_notes: dict | None = None) -> RunHeader:
    manifest, pid, notes = profile_manifest(engine, adapter_revision=adapter_revision, wheel_sha256=wheel_sha256,
                                            vllm_commit=vllm_commit, sampling=workload["sampling"])
    notes["reenter_impls"] = {op: reenter_impl_name(op) for op in ("vllm.unified_attention_with_output", "vllm.unified_kv_cache_update")}
    notes["phases"] = "forward = inside the model's root forward; other = the runner's input preparation / post-forward work inside " \
                      "execute_model; sample = runner.sample (lm_head GEMM + sampler)"
    notes["observer_sync"] = "no observer-issued synchronisation; step facts + sampled ids are copied D2H asynchronously behind a " \
                             "CUDA event and read at the next step start; torch.cuda.synchronize only at the end of snapshot steps"
    if extra_notes:
        notes.update(extra_notes)
    model = {"repo": engine.cp["repo"], "revision": engine.cp["revision"], "local_path": engine.cp.get("local_path"),
             "case": engine.case, "config": model_config_subset(engine)}
    wl = {"requests": workload["requests"], "sampling": workload["sampling"], "source": workload.get("source"), "sha256": workload.get("sha256")}
    # [GEN sampling] n > 1: the request ids the runner sees are vLLM's children (`parallel_sampling.ParentRequest.get_child_info`:
    # id f"{index}_{request_id}", seed + index when seeded).  Recorded as a fact of the fork; the request record itself is unchanged.
    children = [{"request_id": f"{i}_{r['request_id']}", "parent": r["request_id"], "index": i,
                 "seed": None if r.get("seed") is None else int(r["seed"]) + i}
                for r in workload["requests"] for i in range(int(r.get("n") or 1)) if int(r.get("n") or 1) > 1]
    if children:
        wl["children"] = children
    return RunHeader(SCHEMA, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pid, manifest, model, wl, notes)


def host_doc() -> dict:
    """[M6] CPU-side facts that bound the observer's Python cost and would otherwise confound overhead ladders across pods:
    the CPU model, the visible/quota cores, the kernel clocksource, and the measured cost of one `time.perf_counter()` read.
    On an `acpi_pm` host a clock read is a ~8 µs syscall (vs ~25 ns with `tsc`/`kvm-clock`), and the hook reads the clock
    four times per recorded call (M6 report, addendum)."""
    doc: dict[str, Any] = {"nproc": os.cpu_count()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                doc["cpu"] = line.split(":", 1)[1].strip()
                break
    except Exception:  # noqa: BLE001
        pass
    try:
        q = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        doc["cpu_quota"] = None if q[0] == "max" else float(q[0]) / float(q[1])
    except Exception:  # noqa: BLE001
        pass
    try:
        doc["clocksource"] = Path("/sys/devices/system/clocksource/clocksource0/current_clocksource").read_text().strip()
    except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            return None

    import triton
    import vllm

    doc = {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__, "vllm": vllm.__version__,
           "python": os.sys.version.split()[0], "numpy": np.__version__, "flashinfer": ver("flashinfer-python"),
           "transformers": ver("transformers")}
    try:
        props = torch.cuda.get_device_properties(0)
        doc["gpu"] = {"name": props.name, "cc": f"{props.major}.{props.minor}", "sms": int(props.multi_processor_count)}
    except Exception:  # noqa: BLE001
        pass
    doc["host"] = host_doc()
    if engine is not None:                    # [GEN batch SHARED-CODE] the effective LLM(**kw) of this run (CLI overrides included)
        doc["engine_kwargs"] = {k: v for k, v in engine.kwargs.items() if k not in ("model", "tokenizer") and isinstance(v, (int, float, str, bool, type(None)))}
        # [R16 sys SYS-04, coord M-0213] capture/versions.json is GM-01's second engine-fact source (record_engine_facts: every location must
        # agree): the RESOLVED async_scheduling, the same read-back the RunHeader.profile carries -- never the requested kwarg (usually absent)
        doc["engine_kwargs"]["async_scheduling"] = resolved_async_scheduling(engine.llm)
        # [R15 EXEC-02 / COORD M-0063 (2)] the execution the row DECLARED and the one the engine REALISES (vllm_config read-back), in every
        # run header that goes through this engine (Match capture/control, Commit control+treatment = the same engine object): the finalizer
        # compares `effective_execution.execution_label` with the row id and the Program's applicability (M-0063 (3)).
        doc["declared_execution"] = getattr(engine, "execution", None)
        doc["effective_execution"] = getattr(engine, "effective", None)
    try:
        export = Path("EXPORT.json")
        if export.exists():
            doc["export"] = json.loads(export.read_text())
    except Exception:  # noqa: BLE001
        pass
    return doc
