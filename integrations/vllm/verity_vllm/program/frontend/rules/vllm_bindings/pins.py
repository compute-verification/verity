"""The kernel pins of the admitted profile (`verity_vllm/engine/profiles/vllm_d9105ea80_sm89_eager.py`) and the GEMM pins of the
active target."""

from __future__ import annotations

from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET
from verity_vllm.program.frontend.rules.vllm_bindings.observations import _OBSERVED, active_target_profile


#: kernel pins of the admitted profile `vllm_d9105ea80_sm89_eager` (source: the profile module cited above).
#: `NUM_SMS` is a TARGET fact: its admitted value is the default `TargetProfile`'s (one source; the active profile overrides it
#: through `observe_target_profile`, and the GEMM rule then checks every launch against THAT value).
GEMM_PINS = {"kernel": "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent",
             # sha256 of the kernel's Python source at the pinned vLLM (d9105ea80); the eager capture records the same `src_sha`.
             # A renamed-but-identical kernel still binds; a same-named kernel with another body is refused.
             "src_sha256": "4a7e9290c5dbee5e98e04eced96a469438b7514665cdfc1659ed554e293a7287",
             "constexprs": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8,
                            "A_LARGE": False, "B_LARGE": False, "C_LARGE": False, "HAS_BIAS": False},
             "num_warps": 8, "num_stages": 3, "NUM_SMS": DEFAULT_TARGET.num_sms, "dtype": "torch.bfloat16"}
RMS_PINS = {"kernel": "vllm.model_executor.determinism.batch_invariant._rms_norm_kernel",
            "src_sha256": "aa15c9b0f34d3f78b63bc21b5373d1edca7a063b17ecc45ee1468e7b803f2524",
            "constexprs": {"BLOCK_SIZE": 1024, "HAS_WEIGHT": True}, "num_warps": 4, "dtype": "torch.bfloat16"}
#: [R15 dense] `aten::mean.dim` under batch invariance: `enable_batch_invariant_mode()` (batch_invariant.py:1119) registers
#: `_batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")`, and `mean_batch_invariant` -> `mean_dim` launches the
#: Triton `mean_kernel` (BLOCK_SIZE=1024, the JIT default 4 warps, the input viewed `(M, N, 1)`, one program per output element).  The
#: override is a CUDA dispatch-key registration the meta-device export never runs, so the export graph shows the plain `aten.mean.dim`
#: node; the rule binds `MeanTriton_v1` only when the export DERIVED batch invariance (BI Triton launches in the same graph — the Python
#: paths only `vllm_is_batch_invariant()` opens, the same predicate that installs the override library in the engine worker) AND the
#: installed vLLM's `mean_kernel` / `mean_dim` / `mean_batch_invariant` sources hash to the pinned ones (`vllm_meta.export_facts` reads
#: them in-process, the same `JITFunction.src` sha the eager capture records as `src_sha`).  The eager Match pattern
#: (`gen_dense_gemma2_patterns.MeanDimTriton`) checks the launch itself (grid, BLOCK_SIZE, num_warps, K=1, strides).
MEAN_PINS = {"kernel": "vllm.model_executor.determinism.batch_invariant.mean_kernel",
             "src_sha256": "94b6a25ba19f2471ae2dc06c49edbcc3a0937ccbe47d1f716cf393155caf711e",
             "mean_dim_src_sha256": "f0897d2b8e3d75a67c6c45c6ad69b6ecdd38d1ddcfac0281e0ffffb359e586fd",
             "mean_batch_invariant_src_sha256": "401f9643c811f94c3ed2e7ab5f3c03d04362f2f0fc0f8f9128c4e1cfaa813b93",
             "constexprs": {"BLOCK_SIZE": 1024}, "num_warps": 4, "dtype": "torch.float32", "override": "aten::mean.dim -> mean_batch_invariant (CUDA)"}


def active_gemm_target():
    """[R13 ampere] The registered GEMM specialization of the ACTIVE target (`TargetProfile.gemm_target()` of the profile the construction
    applied, else the default profile's): the one source of the launch geometry a GEMM node must show (`GemmTarget.launch_config`), of the
    Definition's k16 step and of the tuned-table identity the manifest records. Raises `gemm_targets.UnregisteredGemmTarget` for a target
    without one (Hopper/Blackwell/untested capabilities): the GEMM rule then refuses every launch (`numerics_unregistered`)."""
    return active_target_profile().gemm_target()


def _target_kinds():
    """(uses accepted kinds?, DOT) of the active target — the ONE selection the GEMM and attention emits consult (`frontend.targets`)."""
    from verity_vllm.program.frontend import targets as T
    tp = active_target_profile()
    return T.uses_accepted_kinds(tp), T.tensor_core_dot(tp)


def gemm_bias_role() -> tuple[str, dict] | None:
    """(vocabulary role, its extra statics) of a biased linear's one-Call Definition on the active target
    (`frontend.targets.gemm_bias_spec`), or None: the target keeps the launch and its bias add as two Calls."""
    from verity_vllm.program.frontend import targets as T
    tp = active_target_profile()
    if T.uses_accepted_kinds(tp):
        return "gemm_bias", {}
    dot = T.gemm_bias_dot(tp)
    return None if dot is None else ("gemm_bias_dot", {"DOT": dot})


def moe_expert_role(routed: bool) -> tuple[str, dict]:
    """(vocabulary role, its extra statics) of a `fused_moe_kernel` launch's per-pair Definition on the active target: the `_v1` kinds,
    or the `_dot` ones with the target's k16 step (`frontend.targets.moe_expert_dot`)."""
    from verity_vllm.program.frontend import targets as T
    dot = T.moe_expert_dot(active_target_profile())
    role = "moe_expert_gemm_w" if routed else "moe_expert_gemm"
    return (role, {}) if dot is None else (role + "_dot", {"DOT": dot})


def moe_expert_refusal(padded: bool = False) -> str | None:
    """None when the active target binds the fused-MoE block's expert GEMMs under the construction, else the refusal: a target without
    a registered step, or the padded construction on a target whose expert GEMMs take the `_v2{DOT}` Definitions."""
    from verity_vllm.program.frontend import targets as T
    try:
        dot = T.moe_expert_dot(active_target_profile())
    except T.UnregisteredTarget as e:
        return f"numerics_unregistered: {e}"
    if padded and dot is not None:
        return (f"numerics_unregistered: the active target's fused_moe_kernel runs the k16 step {dot.id}; MoeFfnFixedL_v1's expert bodies "
                "run the Ampere one and no padded Definition takes the step as a static")
    return None


def gemm_pins_for(M: int, N: int, K: int) -> tuple[dict, str]:
    """The pins a bf16 `[M,K] x [K,N]` launch must match on the active target: vLLM's `_get_matmul_config` picks a tabled (N, K) row's
    M-bucket tiling when the target's arch family has a tuned table, else the default — `GemmTarget.launch_config` of the registered
    target (`verity_vllm.program.frontend.gemm_targets`), the SAME record the eager resolver (`patterns.GemmLaunch`) pins. Same `Gemm_v1{K,N}`
    body either way (per-coordinate f32 accumulator over ascending k16 steps): that the target's geometry keeps it is the record's
    correspondence evidence, not this docstring. Returns (pins, source); raises `UnregisteredGemmTarget` for an unregistered target.

    [X-01] `NUM_SMS` is the ONE `TargetProfile` fact: the active profile's `num_sms` (what `export_compat.apply_target_profile` made
    vLLM's `num_compute_units()` answer, what the artifact identity carries, what the eager resolver pins from the record's device
    document) — `TargetProfile.from_device()` queries it once on a GPU host."""
    base = GEMM_PINS
    tp = _OBSERVED.get("target_profile")
    if tp is not None and tp.num_sms != GEMM_PINS["NUM_SMS"]:
        base = dict(GEMM_PINS, NUM_SMS=tp.num_sms)  # the active TargetProfile's SM count: what the shim applied, what the launch must show
    target = active_gemm_target()
    consts, warps, stages, source = target.launch_config(M, N, K)
    pins = dict(base)
    pins["constexprs"] = {**GEMM_PINS["constexprs"], **consts}
    pins["num_warps"], pins["num_stages"] = warps, stages
    return pins, source


def tuned_gemm_table_sha256() -> str | None:
    """Identity of the pinned tuned-GEMM table the active target's GEMM policy consults (an applicability condition); None for a
    table-less target (Ampere: default tiling for every shape) or an unregistered one."""
    try:
        return active_gemm_target().table_sha256()
    except Exception:  # noqa: BLE001
        return None


def lookup_key(meta: dict) -> dict:
    """The Definition-selection key of one opaque launch node — everything the rules test, in one record (REPORT / validator):
    kernel qualname + source sha256 + Triton version (implementation identity), constexprs and scalar launch arguments, effective
    launch geometry (num_warps/num_stages — explicit or Triton's defaults, flagged), grid, and per-tensor (shape, stride, dtype).
    The vLLM revision and GPU (NUM_SMS) are the profile's identity, pinned in GEMM_PINS/RMS_PINS."""
    return {"kernel": meta.get("kernel"), "src_sha256": meta.get("src_sha256"), "triton": meta.get("triton"),
            "constexprs": {k: meta.get("scalars", {}).get(k) for k in meta.get("constexprs", [])},
            "scalars": {k: v for k, v in meta.get("scalars", {}).items() if k not in set(meta.get("constexprs", []))},
            "geometry": meta.get("geometry"), "grid": meta.get("grid"),
            "tensors": dict(zip(meta.get("tensor_args", []), meta.get("tensor_meta", [])))}


def _pins_ok(meta: dict, pins: dict) -> str | None:
    """Every pinned field of the launch must match; an absent field is a mismatch (never assumed)."""
    scalars = meta.get("scalars", {})
    if "src_sha256" in pins and meta.get("src_sha256") != pins["src_sha256"]:
        return f"kernel source sha256 {meta.get('src_sha256')} differs from the profile pin {pins['src_sha256']} (same name, another kernel body)"
    for k, v in pins["constexprs"].items():
        if scalars.get(k) != v:
            return f"constexpr {k}={scalars.get(k)!r} differs from the profile pin {v!r}"
    geom = meta.get("geometry") or {}
    nw = geom.get("num_warps", scalars.get("num_warps"))
    if nw != pins["num_warps"]:
        return f"num_warps={nw} differs from the profile pin {pins['num_warps']}"
    if "num_stages" in pins:
        ns = geom.get("num_stages", scalars.get("num_stages"))
        if ns != pins["num_stages"]:
            return f"num_stages={ns} differs from the profile pin {pins['num_stages']}"
    if "NUM_SMS" in pins and scalars.get("NUM_SMS") != pins["NUM_SMS"]:
        return f"NUM_SMS={scalars.get('NUM_SMS')} differs from the profile pin {pins['NUM_SMS']} (another GPU is another profile)"
    return None
