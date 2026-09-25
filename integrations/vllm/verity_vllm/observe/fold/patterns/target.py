"""The record target's facts the patterns pin against: its `TargetProfile`, read from the run header, and the registered GEMM
specialization of that target."""

from __future__ import annotations

from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET, TargetProfile

from verity_vllm.observe.fold.resolver import StepContext

X01_SCRATCH_KEY = "x01_num_sms"


def _targets():
    from verity_vllm.program.registry import targets as T
    return T


def target_of(ctx: StepContext) -> tuple[TargetProfile, bool]:
    """[X-01] The record's `TargetProfile` — the ONE SM-count fact both builder bets and this resolver pin against.  For a
    captured log it is the capture host's own device query, recorded by the GPU phase in the run header
    (`RunHeader.profile["gpu"]`: `verity_vllm.engine.engine_profile.gpu_info` -> `multi_processor_count`, `compute_capability`) and read
    back through `TargetProfile.from_gpu_document`; a header without a GPU document (synthetic / hand-built logs) keeps
    `DEFAULT_TARGET`.  Returns (profile, recorded) where `recorded` says the header carried the fact."""
    try:
        # [R13 hopper] the DECLARED profile when the capture recorded one (`profile["target"]`: exact, incl. `flash_attn_version`), else the
        # device document + the layers' reported FA version (`TargetProfile.from_run_profile`)
        tp = TargetProfile.from_run_profile(ctx.header.profile or {})
    except (AttributeError, TypeError, ValueError):
        tp = None
    return (tp, True) if tp is not None else (DEFAULT_TARGET, False)


def header_gemm_arch_family(ctx: StepContext):
    """[R13 ampere] The tuned-GEMM arch family the CAPTURE host reported (`RunHeader.profile["gpu"]["tuned_gemm_arch_family"]`, written by
    `verity_vllm.engine.engine_profile.gpu_info` from vLLM's own `_get_tuned_matmul_arch_family`), or `...` when the header predates the field
    (the registry then derives it from the capability alone)."""
    try:
        gpu = (ctx.header.profile or {}).get("gpu") or {}
    except AttributeError:
        return ...
    return gpu["tuned_gemm_arch_family"] if isinstance(gpu, dict) and "tuned_gemm_arch_family" in gpu else ...


GEMM_TARGET_SCRATCH_KEY = "r13_gemm_target"


def gemm_target_of(ctx: StepContext):
    """[R13 ampere] The registered GEMM specialization of the record's target (`TargetProfile.gemm_target()` on `target_of(ctx)`, with
    the family the capture host reported): `(GemmTarget, None)` or `(None, refusal)` — the ONE source of the launch geometry every
    `matmul_kernel_persistent` launch must show (`GemmTarget.launch_config`), shared with Bet A (`vllm_bindings.gemm_pins_for`).  The
    selection is recorded once in `ctx.scratch[GEMM_TARGET_SCRATCH_KEY]` for the fold / resolution report."""
    from verity_vllm.program.registry.gemm_targets import try_gemm_target_for
    tp, recorded = target_of(ctx)
    fam = header_gemm_arch_family(ctx)
    gt, refusal = try_gemm_target_for(tp.compute_capability, fam)
    if ctx is None or not isinstance(getattr(ctx, "scratch", None), dict):
        return gt, refusal
    rec = ctx.scratch.setdefault(GEMM_TARGET_SCRATCH_KEY, {"target": gt.key if gt else None, "dot": gt.dot if gt else None, "refusal": refusal,
                                                            "compute_capability": list(tp.compute_capability), "target_recorded": recorded,
                                                            "header_tuned_gemm_arch_family": None if fam is ... else fam,
                                                            "tuned_table_sha256": gt.table_sha256() if gt else None, "launches": 0,
                                                            "basis": "R13 ampere: launch geometry = the registered GEMM target's (verity_vllm.program.registry.gemm_targets)"})
    rec["launches"] += 1
    return gt, refusal


def pinned_num_sms(ctx: StepContext, profile_pin: int) -> int:
    """[X-01] The `NUM_SMS` a persistent-GEMM launch must show: the record target's `num_sms` when the header recorded the device
    (exact — a launch for another GPU on this host is still a pin mismatch), else the profile's own pin (`DEFAULT_TARGET.num_sms`).
    `NUM_SMS` is the persistent grid size (how many CTAs walk the tile list): every output tile is one f32 accumulator advanced over
    the same ascending 64-wide k tiles whatever the grid, so it is a scheduling parameter of the launch, not a numerical one —
    which is why the value comes from the target and not from the numerics profile.  The choice is counted in
    `ctx.scratch[X01_SCRATCH_KEY]` so the fold / resolution report record which target fact was applied."""
    tp, recorded = target_of(ctx)
    if not recorded:
        return int(profile_pin)
    rec = ctx.scratch.setdefault(X01_SCRATCH_KEY, {"pinned": int(tp.num_sms), "source": "RunHeader.profile.gpu.multi_processor_count (TargetProfile.from_gpu_document)",
                                                   "profile_default": int(profile_pin), "compute_capability": list(tp.compute_capability), "launches": 0,
                                                   "basis": "X-01: NUM_SMS = the target's SM count (persistent grid size; scheduling only under BI)"})
    rec["launches"] += 1
    return int(tp.num_sms)
