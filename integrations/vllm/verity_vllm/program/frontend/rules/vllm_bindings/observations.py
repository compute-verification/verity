"""Fields observed on the construction host, and the one module state they live in (`_OBSERVED`)."""

from __future__ import annotations

from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET, TargetProfile


# -- fields observed on the construction host (derived, not declared) --------------------------------------------------------
#
# CB-C's validation found three Definition-selection fields both builder bets *declared*: the attention backend, the
# batch-invariant dispatch of `_C.fused_add_rms_norm` (RMSNormFusedCuda_v2 vs the superseded _v1, identical params), and the
# tensor-core instruction class of the GEMM (Gemm_v1 vs Gemm_v2{DOT}). The first two ARE observable at construction:
#   (a) the instantiated `Attention` layer carries the selector's output (`layer.impl` class, `layer.backend`, FA version,
#       sliding window / soft cap / alibi / sinks, kv-cache dtype, batch_invariant_enabled) — `observe_attention_impls`;
#   (b) at the pinned vLLM `RMSNorm.forward_cuda` reaches `_C.fused_add_rms_norm` ONLY through `rms_norm_batch_invariant`
#       (the non-BI branch is `forward_native`, plain aten), and the SAME switch installs the Triton overrides whose launches
#       the export graph shows (`_rms_norm_kernel`, `matmul_kernel_persistent`) — `observe_export_facts` derives the dispatch
#       from the graph + the installed override library, not from an env string.
# The tensor-core class stays DECLARED (compiler-chosen; nothing on a meta host observes it): fail-closed via the GEMM pins
# (kernel source sha + constexprs + geometry + NUM_SMS) and the profile's Gemm_v1 body.

ATTN_PINS = {"impl": "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl", "backend": "FLASH_ATTN", "vllm_flash_attn_version": 2,
             "attn_type": "decoder", "kv_cache_dtype": "auto", "alibi_slopes": None,
             "sinks": None, "batch_invariant_enabled": True, "dcp_world_size": 1, "pcp_world_size": 1}
"""What the FA2 attention kinds (`Attention_v3`, `AttentionSoftcap_v1`: causal GQA, single split under batch invariance) model;
every field is read from the instantiated layer. A layer whose selector picked another backend / FA3 / alibi / sinks / split-KV
combine is refused. Two further observed fields SELECT the kind rather than pin it (`attn_kind_for`): `logits_soft_cap` (0 → role
`attention`; C > 0 → role `attention_softcap` with CAP=C) and `sliding_window` (none → any T; a window W → only positions whose
whole history fits, T <= W, i.e. the window is inert — otherwise refused: no registered windowed kind)."""
ATTN_OBSERVED_SELECTORS = ("logits_soft_cap", "sliding_window")


def attn_kind_for(obs: dict) -> tuple[str, dict, int | None]:
    """(role, extra statics, window) selected by the layer's observed soft cap and sliding window."""
    cap = obs.get("logits_soft_cap") or 0
    win = obs.get("sliding_window")
    if isinstance(win, (list, tuple)):
        w = int(win[0]) if len(win) else -1
    elif win is None:
        w = -1
    else:
        w = int(win)
    window = None if w is None or w < 0 else w
    if cap and float(cap) > 0:
        return "attention_softcap", {"CAP": float(cap)}, window
    return "attention", {}, window

FUSED_NORM_DISPATCH_PIN = "batch_invariant"

_OBSERVED: dict = {"attention_impls": None, "export_facts": None, "target_profile": None, "launch_context": None,
                   "allreduce_settings": None}


def clear_observations() -> None:
    _OBSERVED["attention_impls"] = None
    _OBSERVED["export_facts"] = None
    _OBSERVED["target_profile"] = None
    _OBSERVED["launch_context"] = None
    _OBSERVED["allreduce_settings"] = None


def observe_allreduce_settings(settings: dict | None) -> None:
    """The TP row's served all-reduce switches, read from its workload before export."""
    _OBSERVED["allreduce_settings"] = dict(settings) if settings else None


def active_allreduce_settings() -> dict | None:
    return _OBSERVED.get("allreduce_settings")


def observe_launch_context(launch: dict | None) -> None:
    """[R15 hopper HOP-03] The LAUNCH CONTEXT the request Program is built under (`frontend.targets.parse_launch_context`: per request step
    the varlen launch's max_seqlen_q over ALL co-scheduled sequences).  The attention rule's FA3 tile selection reads it: a request's
    own rows are the launch's value only for a batch of one.  None = no declared context (own rows; B1 semantics, digests unchanged)."""
    _OBSERVED["launch_context"] = dict(launch) if launch else None


def active_launch_context() -> dict | None:
    return _OBSERVED.get("launch_context")


def observe_target_profile(profile: TargetProfile | None) -> None:
    """The TargetProfile the construction APPLIED (`export_compat.apply_target_profile`): the GEMM rule's `NUM_SMS` check reads it."""
    _OBSERVED["target_profile"] = profile


def active_target_profile() -> TargetProfile:
    return _OBSERVED.get("target_profile") or DEFAULT_TARGET


def observe_attention_impls(impls: dict) -> None:
    """`{layer_name: observation}` from the instantiated model (see `verity_vllm.program.frontend.vllm_meta.attention_impls`)."""
    _OBSERVED["attention_impls"] = dict(impls)


def observe_export_facts(facts: dict) -> None:
    """Graph-level facts of the export the rules select on (see `vllm_meta.export_facts`): `fused_add_rms_norm_dispatch`."""
    _OBSERVED["export_facts"] = dict(facts)


def attn_lookup_key(obs: dict) -> dict:
    return {k: obs.get(k) for k in (*ATTN_PINS, *ATTN_OBSERVED_SELECTORS)}


def _norm_attn(v):
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return v


def _attn_pins_ok(obs: dict | None, layer_name: str) -> str | None:
    if obs is None:
        return (f"attention implementation of {layer_name!r} was not observed at construction "
                "(observe_attention_impls(vllm_meta.attention_impls(model)) before derive) — refusing rather than assuming a backend")
    # [R13 hopper] the FA version pin is the ACTIVE target's effective one (`TargetProfile.effective_flash_attn_version`: the declared knob
    # or the selector's default for the capability), and only a version the target has a registered attention chain for is admitted
    # (`frontend.targets.check_flash_attn_registered`); every other field keeps the FA2 profile's pin.
    from verity_vllm.program.frontend import targets as T
    tp = active_target_profile()
    try:
        want_fav = T.check_flash_attn_registered(tp)
    except T.UnregisteredTarget as e:
        return f"numerics_unregistered: {e}"
    for k, want in ATTN_PINS.items():
        if k == "vllm_flash_attn_version":
            want = want_fav
        got = _norm_attn(obs.get(k, "<absent>"))
        if got != _norm_attn(want):
            return (f"numerics_unregistered: attention {k}={got!r} (observed on layer {layer_name!r}) differs from the target's {want!r}"
                    + (f" (TargetProfile flash_attn_version={tp.flash_attn_version!r}, cc {tp.capability_str})" if k == "vllm_flash_attn_version" else ""))
    return None


def _fused_norm_dispatch_ok() -> str | None:
    f = _OBSERVED["export_facts"]
    if f is None:
        return "fused_add_rms_norm dispatch was not observed at construction (observe_export_facts before derive) — refusing rather than assuming"
    d = f.get("fused_add_rms_norm_dispatch")
    if d != FUSED_NORM_DISPATCH_PIN:
        return (f"numerics_unregistered: `_C.fused_add_rms_norm` dispatch observed as {d!r} (evidence {f.get('evidence')!r}); the profile registers "
                f"only the {FUSED_NORM_DISPATCH_PIN} kernel (RMSNormFusedCuda_v2); the width-8 kernel (_v1) is superseded")
    return None


def declared_derived_accounting(profile: TargetProfile | None = None, applied: dict | None = None) -> dict:
    """The Definition-selection fields, split by how the construction learns them (for `construction_manifest.json`).
    The DECLARED block is generated from the TargetProfile (the same object the shims applied) and, when the driver passes the
    read-back record, carries `applied` next to `value` — no literal target fact lives here."""
    profile = profile or active_target_profile()
    applied = applied or {}
    return {
        "derived": [
            {"field": "attention backend / impl", "from": "instantiated `Attention.impl` class + layer attributes (selector output)",
             "key": list(ATTN_PINS), "pins": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ATTN_PINS.items()},
             "rule": "UnifiedAttentionWithOutput[vllm] v2"},
            {"field": "fused_add_rms_norm batch-invariant dispatch (RMSNormFusedCuda_v2 vs _v1)",
             "from": "export graph: `_C.fused_add_rms_norm` reached via `rms_norm_batch_invariant` + BI Triton launches present + override library installed",
             "pin": FUSED_NORM_DISPATCH_PIN, "rule": "FusedAddRmsNorm[vllm-bi] v1",
             "caveat": "the CUDA op's own kernel-width choice (`batch_invariant_launch` in layernorm_kernels.cu) reads the same env the Python switch reads; coupled, not observed"},
            {"field": "Triton kernel identity + tiling", "from": "launch node: kernel qualname, source sha256, constexprs, num_warps/num_stages, tensor meta",
             "rule": "TritonGemm[vllm-bi] / TritonRmsNorm[vllm-bi]"},
            {"field": "tuned-GEMM tiling per (N, K, M-bucket)", "from": "launch constexprs vs `gemm_pins_for` (vLLM's `_get_matmul_config` table for the declared capability)"},
        ],
        "declared": [
            {"field": "NUM_SMS", "value": profile.num_sms, "applied": applied.get("num_sms"), "source": "TargetProfile.num_sms",
             "why": "target GPU fact; the shim answers it to vLLM's num_compute_units on every host; the GEMM rule checks every launch against it"},
            {"field": "compute capability", "value": profile.capability_str, "applied": applied.get("compute_capability"), "source": "TargetProfile.compute_capability",
             "why": "target GPU fact; the shim answers it to vLLM's get_device_capability at instantiation AND export (FA version, BI GEMM family, tuned-GEMM table)"},
            {"field": "batch invariant", "value": profile.batch_invariant, "applied": applied.get("batch_invariant"), "source": "TargetProfile.batch_invariant -> env VLLM_BATCH_INVARIANT",
             "why": "the switch RMSNorm.forward_cuda / init_batch_invariance read; its consequence (dispatch) is derived from the export graph"},
            {"field": "attention backend knob", "value": profile.attention_backend, "applied": applied.get("attention_backend"), "source": "TargetProfile.attention_backend -> EngineArgs(attention_backend=)",
             "why": "None = vLLM's selector for the declared capability; the selector's RESULT (impl class, FA version) is observed per layer"},
            {"field": "tensor-core instruction class (Gemm_v1 vs Gemm_v2{DOT})", "value": _target_desc(profile), "source": "TargetProfile.compute_capability -> registry.targets",
             "why": "compiler-chosen (Triton -> PTX); not observable on a meta host; keyed by the declared capability's registered pipeline family; "
                    "fail-closed via kernel source sha + constexprs + geometry + NUM_SMS and the row's Match replay (bit-exact on captured operands)"},
            {"field": "flash_attn_version", "value": profile.flash_attn_version, "effective": profile.effective_flash_attn_version(),
             "applied": applied.get("flash_attn_version"), "source": "TargetProfile.flash_attn_version -> EngineArgs(attention_config={'flash_attn_version': v})",
             "why": "[R13 hopper] None = vLLM's selector for the declared capability (FA3 on sm_90); the instantiated layers' version is read back and pinned per layer"},
        ],
    }


def _target_desc(profile: TargetProfile) -> dict:
    from verity_vllm.program.frontend import targets as T
    try:
        d = T.describe(profile)
    except T.UnregisteredTarget as e:
        return {"unregistered": str(e)}
    return {k: d[k] for k in ("tensor_core_family", "dot", "gemm_kind", "attention_kind", "tuned_gemm_arch_family", "flash_attn_version_effective")}
