"""Build the vLLM engine under the frozen profile κ, and read back what the built engine executes.

`profile.apply_env()`, `cp = checkpoint_for(case, manifests/checkpoints.json)` (case B0 = SmolLM2-135M, B1 = Qwen2.5-1.5B),
`LLM(**engine_kwargs(cp, replay=False, enforce_eager=True), enable_prefix_caching=False, ...)`, `runner = get_model_runner(llm)`
(DESIGN §1 / §10).  The row's declared execution decides `enforce_eager`, and a declared `TargetProfile` adds its engine args and
environment knobs; after the build the effective execution, compilation, attention version and async-scheduling mode are read back
from the engine's `vllm_config` and checked against the declaration.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from verity_vllm.engine import env as engine_env


CASE = "B0"                       # default checkpoint role (SmolLM2-135M); [M6] `build_engine(case="B1")` serves Qwen2.5-1.5B


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
    environ: dict | None = None    # the process environment keys its build set or changed (`engine_env.set_since`), None when not built here


# [R15 EXEC-02 / COORD M-0063 (2)] the row's execution label decides `enforce_eager`; it is never a literal of the harness.  Labels are the
# row-id vocabulary (`…__bi-eager`, `…__bi-eager-fa2`, `…__bi-compiled`); a label this table does not know fails closed (no guessing).
EXECUTION_ENFORCE_EAGER: dict[str, bool] = {"bi-eager": True, "bi-eager-fa2": True, "eager": True, "bi-compiled": False, "compiled": False}


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
        from verity_vllm.engine import engine_profile as prof
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
    from verity_vllm.program.frontend.target_profile import parse_target_arg as _parse
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
    from verity_vllm.program.frontend.target_profile import TargetProfile
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


prepare_compiled_process_cache = engine_env.prepare_compiled_process_cache


def engine_kwargs_for(checkpoints: str | Path = "manifests/checkpoints.json", *, max_num_seqs: int = 8, case: str = CASE,
                      engine_args: dict[str, Any] | None = None, target: Any = None, execution: str | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """[R16 sys] The exact `LLM(**kw)` keyword set `build_engine` would build from these inputs (+ the env flags it applies and the
    checkpoint record), WITHOUT building: a hot worker compares it with its live engine's `kwargs` to decide reuse vs restart.  Applies the
    same environment knobs as `build_engine` (idempotent).  Refusals (unknown execution, contradicting enforce_eager) are the same."""
    from verity_vllm.engine import engine_profile as prof

    if target is not None:
        # [R15 fp8] the declared target's ENVIRONMENT knobs (`TargetProfile.env()`: VLLM_BATCH_INVARIANT, VLLM_USE_DEEP_GEMM for
        # `fp8_block_gemm`) are applied before vLLM reads them -- the same object the builder applied; the header records them
        # (`target_env`) and `target_mismatches` reads the selected FP8 kernel back
        target_env = engine_env.apply_target(target.env())
    env_flags = prof.apply_env()
    if target is not None:
        env_flags = {**env_flags, **target_env}
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
    from verity_vllm.engine import engine_profile as prof

    # [R15 EXEC-02 / COORD M-0063 (2)] `enforce_eager` comes from the row's declared execution (`execution_of_workload`), not from a literal:
    # an undeclared execution (no row id) keeps the historical eager default and is recorded as `declared_execution: null`; an unknown label
    # fails closed.  A CLI `--engine-arg enforce_eager=...` that contradicts the declared execution is refused BEFORE any engine is built,
    # so an eager engine can never serve a `__bi-compiled` row (and vice versa) under a green-looking header (rev-pm M-0040).
    # [R13 hopper] `kernel_config.enable_flashinfer_autotune=False` by construction (the autotune dummy run must not depend on the shell's PATH);
    # [GEN batch SHARED-CODE] explicit CLI engine args override last and are recorded in versions.json/run.json.  All of it in
    # `engine_kwargs_for` (R16 sys: the hot worker computes the same keyword set to decide reuse vs restart).
    env0 = engine_env.snapshot()
    kw, env_flags, cp = engine_kwargs_for(checkpoints, max_num_seqs=max_num_seqs, case=case, engine_args=engine_args, target=target, execution=execution)
    from vllm import LLM  # noqa: E402  (after apply_env)

    from verity_vllm.engine import sparse_mla
    sparse_mla.arm()

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
    eng = Engine(llm, runner, model, cp, kw, env_flags, build_s, case, target, execution, None, engine_env.set_since(env0))
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


def flash_attn_versions(model: Any) -> set:
    """The `vllm_flash_attn_version` values the instantiated attention impls carry; empty when none is a FlashAttention impl (vLLM's
    MLA impls, sparse ones included, carry none)."""
    vs: set = set()
    for _, mod in model.named_modules():
        impl = getattr(mod, "impl", None)
        if impl is not None and hasattr(impl, "vllm_flash_attn_version"):
            vs.add(getattr(impl, "vllm_flash_attn_version"))
    return vs


def observed_flash_attn_version(model: Any) -> int | None:
    """[R13 hopper] The `vllm_flash_attn_version` the instantiated attention layers carry (one value over all layers), None if absent / mixed."""
    vs = flash_attn_versions(model)
    return int(vs.pop()) if len(vs) == 1 and None not in vs else None


def target_mismatches(engine: Engine) -> list[str]:
    """[R13 hopper] Declared `TargetProfile` vs the built engine: device capability + SM count (`torch.cuda.get_device_properties`) and the
    attention layers' FA version (`effective_flash_attn_version`).  Empty = the engine executes the declared target.  An engine whose
    attention impls carry no FA version (GLM-5.3's sparse MLA) is no FA-version question, as for the Build's read-back
    (`pipeline.build`)."""
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
    if fav != tp.effective_flash_attn_version() and (fav is not None or flash_attn_versions(engine.model)):
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
        from verity_vllm.program.frontend.target_profile import fp8_block_gemm_kernel_of
        got = fp8_block_gemm_kernel_of(engine.model)
        if got != tp.fp8_block_gemm_kernel():
            bad.append(f"fp8_block_gemm declared {tp.fp8_block_gemm!r} ({tp.fp8_block_gemm_kernel()}), layers report {got!r}")
    if getattr(tp, "moe_backend", None) is not None:
        from verity_vllm.program.frontend.target_profile import moe_experts_class_of
        got = moe_experts_class_of(engine.model)
        if got != tp.moe_experts_class():
            bad.append(f"moe_backend declared {tp.moe_backend!r} ({tp.moe_experts_class()}), layers report {got!r}")
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
