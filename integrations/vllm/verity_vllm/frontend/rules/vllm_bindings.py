"""Per-op bindings for the graph `torch.export(strict=False)` produces from the ACTUAL pinned vLLM model classes under the
eager batch-invariant profile (CB-A, R9 2026-09-18). Every rule here names an *operator* (a vLLM custom op, a Triton kernel
launched directly by vLLM's batch-invariant code, an ATen cast); none names a model class, a layer, or an op order.

What the export of a real vLLM class contains that the vLLM-*shaped* examples (`derive_examples.py`) did not:

* `verity_cba::triton_launch` nodes (`triton_iface.py`): under `VLLM_BATCH_INVARIANT=1` the GEMMs
  (`aten::linear` -> `linear_batch_invariant` -> `matmul_kernel_persistent`, `batch_invariant.py:1039-1106`) and the
  un-fused RMSNorm (`rms_norm_batch_invariant` -> `_rms_norm_kernel`, `:977-1036`) are Python-level Triton launches, not
  torch ops. The launch node carries the kernel name and its constexprs, which the rules check against the profile's
  kernel pins (the same pins the eager pattern matchers use) before binding `Gemm{K,N}` / `RMSNormTriton{N,EPS}`.
* `vllm::unified_kv_cache_update(k, v, layer_name) -> Tensor` + `vllm::unified_attention_with_output(q, k, v, output!,
  layer_name, output_scale?, output_block_scale?, kv_cache_dummy_dep?) -> ()` (`layers/attention/attention.py:706-784`):
  the KV write and the attention read are two ops tied by a dummy data dependency. `KvCacheUpdate[vllm]` records the
  pending write; `UnifiedAttentionWithOutput[vllm]` owns the per-layer KV history (positions from the static `positions`
  argument's row count, exactly as `UnifiedAttention[vllm]`), checks the pending write is this call's own rows, emits one
  `Attention{T=P}` per row and rebinds `output`.
* `aten.to.dtype(int32 ids, int64)` (`VocabParallelEmbedding.forward`): the I32 token id widened for `aten.embedding`;
  value-preserving for ids in [0, V) -> a reinterpretation, no gate.

Profile pins (declared record, `verity_capture/profiles/vllm_d9105ea80_sm89_eager.py:60-80`; a launch that differs is
refused, never adapted to).
"""

from __future__ import annotations

import torch

from verity.ir.refs import Coll, array_of, concat
from verity.ir.types import Array, Value
from verity_vllm.frontend.derive import Alloc, StaticIndex, refuse
from verity_vllm.frontend.torch_frontend import _permute, _tensor_type, shape_of, width_of
from verity_vllm.frontend.rules.base import InPlaceRule, OpRule, RuleInfo, StateRule, ViewRule
from verity_vllm.frontend.rules.common import dtype_is, from_rows, rows_of, shape
from verity_vllm.frontend.rules.profile import FusedAddRmsNormRule as _FusedAddRmsNormRule
from verity_vllm.frontend.rules.triton_iface import launch_meta, launch_tensor, triton_launch
from verity_vllm.frontend.target_profile import DEFAULT_TARGET, TargetProfile

aten = torch.ops.aten
BF16 = torch.bfloat16

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
    """(uses accepted kinds?, DOT) of the active target — the ONE selection the GEMM and attention emits consult (`registry.targets`)."""
    from verity_vllm.registry import targets as T
    tp = active_target_profile()
    return T.uses_accepted_kinds(tp), T.tensor_core_dot(tp)


def gemm_pins_for(M: int, N: int, K: int) -> tuple[dict, str]:
    """The pins a bf16 `[M,K] x [K,N]` launch must match on the active target: vLLM's `_get_matmul_config` picks a tabled (N, K) row's
    M-bucket tiling when the target's arch family has a tuned table, else the default — `GemmTarget.launch_config` of the registered
    target (`verity_vllm.registry.gemm_targets`), the SAME record the eager resolver (`patterns.GemmLaunch`) pins. Same `Gemm_v1{K,N}`
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

_OBSERVED: dict = {"attention_impls": None, "export_facts": None, "target_profile": None, "launch_context": None}


def clear_observations() -> None:
    _OBSERVED["attention_impls"] = None
    _OBSERVED["export_facts"] = None
    _OBSERVED["target_profile"] = None
    _OBSERVED["launch_context"] = None


def observe_launch_context(launch: dict | None) -> None:
    """[R15 hopper HOP-03] The LAUNCH CONTEXT the request Program is built under (`registry.targets.parse_launch_context`: per request step
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
    """`{layer_name: observation}` from the instantiated model (see `verity_capture.experimental.cb_a.vllm_meta.attention_impls`)."""
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
    # (`registry.targets.check_flash_attn_registered`); every other field keeps the FA2 profile's pin.
    from verity_vllm.registry import targets as T
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
    from verity_vllm.registry import targets as T
    try:
        d = T.describe(profile)
    except T.UnregisteredTarget as e:
        return {"unregistered": str(e)}
    return {k: d[k] for k in ("tensor_core_family", "dot", "gemm_kind", "attention_kind", "tuned_gemm_arch_family", "flash_attn_version_effective")}


def _row_value(ctx, a, name: str, dtype=BF16) -> str | None:
    if not isinstance(a, torch.fx.Node):
        return f"{name} is not a graph value"
    v = ctx.val(a)
    if isinstance(v, Alloc):
        return None
    if not isinstance(v, Coll):
        return f"{name} is not a value collection ({type(v).__name__})"
    if not dtype_is(ctx, a, dtype):
        return f"numerics_unregistered: {name} has dtype {ctx.dtype_of(a)}; the profile's kinds are {dtype}"
    if not shape(ctx, a):
        return f"{name} is a scalar"
    return None


def _strides(n) -> tuple[int, ...] | None:
    """The exported operand's strides (fx `val` meta), or None."""
    v = n.meta.get("val") if isinstance(n, torch.fx.Node) else None
    return tuple(int(s) for s in v.stride()) if hasattr(v, "stride") else None


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


def _untransposed(ctx, b, K: int, N: int, hint: str) -> Coll:
    """`b` [K, N] is `W.t()` (strides checked by the caller); the [N, K] matrix it views. Taken from the transpose's operand when
    `b` is literally an `aten.t` / 2-D `transpose` node (the common case: a parameter), else by permuting the references back."""
    if isinstance(b, torch.fx.Node) and b.target in (aten.t.default, aten.transpose.int, aten.numpy_T.default) and shape(ctx, b.args[0]) == (N, K):
        return ctx.coll(b.args[0])
    from verity_vllm.frontend.torch_frontend import FrontendError
    try:
        return _permute(ctx.coll(b), [1, 0], hint)
    except FrontendError as e:
        refuse("unsupported_view", ctx.current, ctx.module_path(ctx.current), f"b is not a transposed view of a [N, K] matrix: {e.message}")


# -- direct Triton launches -----------------------------------------------------------------------------------------------


class TritonRmsNormRule(InPlaceRule):
    """`_rms_norm_kernel[grid](input_2d, weight, output, in_stride, out_stride, n_cols, eps, BLOCK_SIZE, HAS_WEIGHT)` launched by
    `rms_norm_batch_invariant` -> `RMSNormTriton{N,EPS}` per row; `output` (an allocation) is rebound."""
    modes = frozenset({"profile"})
    info = RuleInfo("TritonRmsNorm[vllm-bi]", 1, "in-place-binding",
                    applies_to="verity_cba::triton_launch of batch_invariant._rms_norm_kernel (kernel name + constexprs pinned)",
                    preconditions=("kernel qualified name is the profile's `_rms_norm_kernel`", "BLOCK_SIZE=1024, HAS_WEIGHT=True (profile pins)",
                                   "input is bf16 [R, N] with unit column stride (row stride = the launch's input_row_stride, >= N: a column slice of a wider "
                                   "row is allowed), weight bf16 [N], output an unwritten allocation [R, N] (row stride = output_row_stride)",
                                   "n_cols == N; eps is a static float; grid == (R,)"),
                    evidence_domain="structural; numerics: RMSNormTriton_v1 (the same Triton kernel, W11 model)")

    def targets(self):
        return {triton_launch.default}

    def applies(self, node, ctx):
        meta = launch_meta(node)
        if meta.get("kernel") != RMS_PINS["kernel"]:
            return f"kernel {meta.get('kernel')} is not {RMS_PINS['kernel']}"
        sc = meta.get("scalars", {})
        bad = _pins_ok(meta, RMS_PINS)
        if bad:
            return bad
        x, w, out = (launch_tensor(node, meta, n) for n in ("input_ptr", "weight_ptr", "output_ptr"))
        if x is None or w is None or out is None:
            return "tensor arguments input_ptr/weight_ptr/output_ptr not all present"
        r = _row_value(ctx, x, "input") or _row_value(ctx, w, "weight")
        if r:
            return r
        if not isinstance(ctx.val(out), (Alloc, Coll)):
            return "output is not an allocation/value"
        if str(ctx.dtype_of(out)) != RMS_PINS["dtype"]:
            return f"numerics_unregistered: output dtype {ctx.dtype_of(out)}; RMSNormTriton_v1 is the bf16 kernel"
        xs = shape(ctx, x)
        if len(xs) != 2 or shape(ctx, w) != (xs[1],) or shape(ctx, out) != xs:
            return f"shapes input {xs} weight {shape(ctx, w)} output {shape(ctx, out)}"
        if sc.get("n_cols") != xs[1]:
            return f"n_cols {sc.get('n_cols')} vs N={xs[1]}"
        # [R13 MOE-01, R12 F-moe-01 B-QNORM-STRIDE] the kernel reads row r at `input_ptr + r*input_row_stride` (columns contiguous) and writes
        # `output_ptr + r*output_row_stride`: the launch's strides must be the OPERANDS' row strides, which need not be N -- OLMoE's q_norm /
        # k_norm normalise column slices of the fused qkv GEMM output ([M, 2048] rows of stride 6144).  The operand's value collection is the
        # strided view's, so the emitted rows are exactly the rows the kernel reads; only the stride facts are checked here.
        for nm, t, st_key in (("input", x, "input_row_stride"), ("output", out, "output_row_stride")):
            st = _strides(t)
            if st is None or len(st) != 2 or st[1] != 1:
                return f"{nm} strides {st} are not (row_stride, 1)"
            if sc.get(st_key) != st[0] or st[0] < xs[1]:
                return f"{st_key}={sc.get(st_key)} vs the {nm} operand's row stride {st[0]} (N={xs[1]})"
        if not isinstance(sc.get("eps"), float):
            return "eps is not a static float"
        if meta.get("grid") not in ([xs[0]], "<callable>"):
            return f"grid {meta.get('grid')} is not (R,)"
        return True

    def emit(self, node, ctx):
        meta = launch_meta(node)
        sc = meta["scalars"]
        x, w, out = (launch_tensor(node, meta, n) for n in ("input_ptr", "weight_ptr", "output_ptr"))
        xc, wc = ctx.coll(x), ctx.coll(w)
        rows, lead = rows_of(xc)
        N, eps = rows[0].type.n, float(sc["eps"])
        fn = ctx.kind("rmsnorm", node, N=N, EPS=eps)
        outs = [ctx.call(fn, r, wc, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(out, from_rows(outs, lead), self, what="output")
        ctx.bind_weight_role(w, self, "rmsnorm.weight", N=N, EPS=eps)
        ctx.record(self, node, {"N": N, "EPS": eps, "rows": len(rows), "lookup_key": lookup_key(meta), "pins": RMS_PINS}, emitted=f"{fn.id} x{len(rows)}")


class TritonGemmRule(InPlaceRule):
    """`matmul_kernel_persistent[grid](a, b, c, bias, M, N, K, strides…, NUM_SMS, …constexprs)` launched by
    `matmul_persistent` from vLLM's `aten::linear` batch-invariant override -> `Gemm{K,N}` per row of `a`; `c` is rebound.
    `b` is `W.t()` (a view of the [N, K] parameter: stride_bk == 1, stride_bn == K); the kind takes W as [N, K]."""
    modes = frozenset({"profile"})
    info = RuleInfo("TritonGemm[vllm-bi]", 1, "in-place-binding",
                    applies_to="verity_cba::triton_launch of batch_invariant.matmul_kernel_persistent (kernel name + constexprs pinned)",
                    preconditions=("kernel qualified name is the profile's `matmul_kernel_persistent`; source sha256 pinned",
                                   "the active TargetProfile's capability has a REGISTERED GEMM specialization (verity_vllm.registry.gemm_targets: ada cc 8.9 with "
                                   "vLLM's tuned sm_89 table, ampere cc 8.0 default tiling only; anything else -> numerics_unregistered)",
                                   "tiling = that target's launch geometry for (M, N, K) (GemmTarget.launch_config: the default 128/128/64, 8 warps, 3 stages, or the "
                                   "M-bucket row of the target's pinned tuned table); GROUP_SIZE_M 8, *_LARGE False, HAS_BIAS False, "
                                   "NUM_SMS = the active TargetProfile's num_sms (default 128)",
                                   "a is bf16 [M, K] row-major (stride_am K, stride_ak 1); b is bf16 [K, N] with stride_bk 1, stride_bn K (= W[N,K].t())",
                                   "c is an unwritten allocation [M, N] row-major; bias_ptr is None; M, N, K equal the shapes"),
                    evidence_domain="structural; numerics: Gemm_v1{K,N} (the same Triton kernel; f32 accumulator over BLOCK_SIZE_K tiles)")

    def targets(self):
        return {triton_launch.default}

    def applies(self, node, ctx):
        meta = launch_meta(node)
        if meta.get("kernel") != GEMM_PINS["kernel"]:
            return f"kernel {meta.get('kernel')} is not {GEMM_PINS['kernel']}"
        sc = meta.get("scalars", {})
        from verity_vllm.registry.gemm_targets import UnregisteredGemmTarget
        try:
            pins, _src = gemm_pins_for(sc.get("M"), sc.get("N"), sc.get("K"))
        except UnregisteredGemmTarget as e:   # fail closed: no registered Gemm Definition for the target, whatever tiling the launch shows
            return str(e)
        bad = _pins_ok(meta, pins)
        if bad:
            return bad
        if sc.get("bias_ptr") is not None or launch_tensor(node, meta, "bias_ptr") is not None:
            return "bias_ptr is set (the fused bias epilogue is a different numeric than Gemm + BiasAdd)"
        a, b, c = (launch_tensor(node, meta, n) for n in ("a_ptr", "b_ptr", "c_ptr"))
        if a is None or b is None or c is None:
            return "tensor arguments a_ptr/b_ptr/c_ptr not all present"
        r = _row_value(ctx, a, "a") or _row_value(ctx, b, "b")
        if r:
            return r
        if not isinstance(ctx.val(c), (Alloc, Coll)):
            return "c is not an allocation/value"
        if str(ctx.dtype_of(c)) != GEMM_PINS["dtype"]:
            return f"numerics_unregistered: c dtype {ctx.dtype_of(c)}; Gemm_v1 is the bf16 kernel"
        as_, bs, cs = shape(ctx, a), shape(ctx, b), shape(ctx, c)
        if len(as_) != 2 or len(bs) != 2 or len(cs) != 2:
            return f"ranks a{as_} b{bs} c{cs}"
        M, K = as_
        K2, N = bs
        if K != K2 or cs != (M, N) or (sc.get("M"), sc.get("N"), sc.get("K")) != (M, N, K):
            return f"shapes a{as_} b{bs} c{cs} vs scalars M={sc.get('M')} N={sc.get('N')} K={sc.get('K')}"
        if (sc.get("stride_am"), sc.get("stride_ak")) != (K, 1):
            return f"a strides {sc.get('stride_am')},{sc.get('stride_ak')} are not row-major [M, K]"
        if (sc.get("stride_bk"), sc.get("stride_bn")) != (1, K):
            return f"b strides {sc.get('stride_bk')},{sc.get('stride_bn')} are not W[N, K].t()"
        if (sc.get("stride_cm"), sc.get("stride_cn")) != (N, 1):
            return f"c strides {sc.get('stride_cm')},{sc.get('stride_cn')} are not row-major [M, N]"
        if K % 16:
            return f"K={K} is not a multiple of the 16-wide k-step"
        return True

    def emit(self, node, ctx):
        meta = launch_meta(node)
        sc = meta["scalars"]
        a, b, c = (launch_tensor(node, meta, n) for n in ("a_ptr", "b_ptr", "c_ptr"))
        ac, bc = ctx.coll(a), ctx.coll(b)
        rows, lead = rows_of(ac)
        K, N = shape_of(bc.type)[0]
        w_nk = _untransposed(ctx, b, K, N, ctx.hint_name(node) + "/W")  # b = W.t(): the kind's W is [N, K] — a reference rearrangement
        accepted, dot = _target_kinds()   # [R13 hopper] the target's registered k16 step selects Gemm_v1 (accepted) or Gemm_v2{DOT}
        fn = ctx.kind("gemm", node, K=K, N=N) if accepted else ctx.kind("gemm_dot", node, K=K, N=N, DOT=dot)
        outs = [ctx.call(fn, r, w_nk, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(c, from_rows(outs, lead), self, what="c")
        ctx.bind_weight_role(b, self, "linear.weight", K=K, N=N)
        wl = ctx.weight_of(b)
        pins, src = gemm_pins_for(sc.get("M"), sc.get("N"), sc.get("K"))
        tgt = active_gemm_target()
        ctx.record(self, node, {"K": K, "N": N, "rows": len(rows), "lookup_key": lookup_key(meta), "pins": pins, "pin_source": src,
                                "gemm_target": tgt.key, "gemm_target_dot": tgt.dot, "W_is_parameter": wl.name if wl else None}, emitted=f"{fn.id} x{len(rows)}")


# -- KV cache write + attention read (two ops) -----------------------------------------------------------------------------


class _KvWrite:
    """The dummy result of `unified_kv_cache_update`: not a value; the pending (k, v) write of one layer."""

    def __init__(self, layer_name: str, k, v, node) -> None:
        self.layer_name, self.k, self.v, self.node = layer_name, k, v, node

    def __repr__(self) -> str:
        return f"_KvWrite({self.layer_name!r})"


def _vllm_op(name: str) -> set:
    try:
        return {getattr(torch.ops.vllm, name).default}
    except Exception:  # noqa: BLE001 — vLLM not importable (laptop): the rule has no targets
        return set()


class KvCacheUpdateRule(StateRule):
    """`vllm::unified_kv_cache_update(key[T,KVH,D], value[T,KVH,D], layer_name) -> Tensor[0]`: the rows this step appends to
    `layer_name`'s cache. Recorded as the layer's pending write; the attention rule that consumes the dummy checks it appends
    the same rows. No gates."""
    modes = frozenset({"profile"})
    info = RuleInfo("KvCacheUpdate[vllm]", 1, "state", applies_to="vllm.unified_kv_cache_update(key, value, layer_name: str) -> Tensor",
                    preconditions=("key, value are bf16 values [T, KVH, D] (or [T, KVH*D]) of equal shape", "layer_name is a static string",
                                   "the dummy result is consumed only by this layer's unified_attention_with_output"),
                    evidence_domain="structural: the KV write is made explicit by the attention rule's history")

    def targets(self):
        return _vllm_op("unified_kv_cache_update")

    def applies(self, node, ctx):
        k, v, layer_name = node.args[:3]
        r = _row_value(ctx, k, "key") or _row_value(ctx, v, "value")
        if r:
            return r
        if isinstance(ctx.val(k), Alloc) or isinstance(ctx.val(v), Alloc):
            return "key/value is an unwritten allocation"
        if not isinstance(layer_name, str):
            return "layer_name is not a static string"
        if shape(ctx, k) != shape(ctx, v):
            return f"key {shape(ctx, k)} vs value {shape(ctx, v)}"
        return True

    def emit(self, node, ctx):
        k, v, layer_name = node.args[:3]
        pending = ctx.state.setdefault("kv_pending", {})
        if layer_name in pending:
            refuse("side_input", node, ctx.module_path(node), f"two KV writes to {layer_name!r} without an attention read between them")
        pending[layer_name] = _KvWrite(layer_name, k, v, node)
        ctx.env[node] = pending[layer_name]
        ctx.record(self, node, {"layer_name": layer_name, "rows": shape(ctx, k)[0]}, emitted="(pending KV write; no gates)")


class UnifiedAttentionWithOutputRule(StateRule):
    """`vllm::unified_attention_with_output(q, k, v, output!, layer_name, output_scale?, output_block_scale?, kv_cache_dummy_dep?)`:
    the KV cache is implicit state keyed by `layer_name` (vLLM's forward context). Made explicit as in `UnifiedAttention[vllm]`:
    the rows k, v of this call extend the layer's history at positions len(history)..; the row at global position P reads
    `K[0:P+1], V[0:P+1]` as `Attention{T=P+1,NH,KVH,D}`; `output` (an allocation) is rebound to the attention rows.
    Causality is the op's contract (decoder attention), recorded as an assumed precondition."""
    modes = frozenset({"profile"})
    info = RuleInfo("UnifiedAttentionWithOutput[vllm]", 2, "state",
                    applies_to="vllm.unified_attention_with_output(query, key, value, Tensor! output, layer_name: str, output_scale=None, output_block_scale=None, kv_cache_dummy_dep=None)",
                    preconditions=("the layer's OBSERVED attention implementation (instantiated `Attention.impl` class, backend, FA version, window, soft cap, "
                                   "alibi, sinks, kv dtype, batch_invariant_enabled) equals ATTN_PINS — derived from the model instance, never assumed",
                                   "q [T, NH, D], k, v [T, KVH, D] bf16 (or flat rows with config D); NH % KVH == 0",
                                   "output is an unwritten allocation of q's shape", "layer_name is a static string owned by exactly one module",
                                   "output_scale and output_block_scale are None (no fp8 output quantisation)",
                                   "kv_cache_dummy_dep, if present, is this layer's pending write of the SAME k, v rows",
                                   "assumed (op contract): causal attention over the history + this call's rows"),
                    evidence_domain="structural; numerics: the vocabulary's attention kind per position (FA2 causal GQA)")

    def targets(self):
        return _vllm_op("unified_attention_with_output")

    def _args(self, node):
        a = list(node.args) + [None] * (8 - len(node.args))
        kw = node.kwargs
        return (a[0], a[1], a[2], a[3], a[4], kw.get("output_scale", a[5]), kw.get("output_block_scale", a[6]), kw.get("kv_cache_dummy_dep", a[7]))

    def applies(self, node, ctx):
        q, k, v, out, layer_name, oscale, obscale, dep = self._args(node)
        for nm, a in (("query", q), ("key", k), ("value", v)):
            r = _row_value(ctx, a, nm)
            if r:
                return r
            if isinstance(ctx.val(a), Alloc):
                return f"{nm} is an unwritten allocation"
        if not isinstance(ctx.val(out), (Alloc, Coll)):
            return "output is not an allocation/value"
        if not isinstance(layer_name, str):
            return "layer_name is not a static string"
        r = _attn_pins_ok((_OBSERVED["attention_impls"] or {}).get(layer_name) if _OBSERVED["attention_impls"] is not None else None, layer_name)
        if r:
            return r
        if oscale is not None or obscale is not None:
            return "numerics_unregistered: output_scale / output_block_scale set (fp8 output) — no registered kind"
        qs, ks, vs, os_ = shape(ctx, q), shape(ctx, k), shape(ctx, v), shape(ctx, out)
        if qs[0] != ks[0] or ks != vs:
            return f"row counts differ q{qs} k{ks} v{vs}"
        if os_ != qs:
            return f"output {os_} vs query {qs}"
        if dep is not None:
            d = ctx.val(dep)
            if not isinstance(d, _KvWrite):
                return f"kv_cache_dummy_dep is {type(d).__name__}, not a pending KV write"
            if d.layer_name != layer_name:
                return f"kv_cache_dummy_dep belongs to {d.layer_name!r}, not {layer_name!r}"
            if ctx.storage_base(d.k) is not ctx.storage_base(k) or ctx.storage_base(d.v) is not ctx.storage_base(v):
                return "the pending KV write's rows are not this call's key/value rows"
        return True

    def emit(self, node, ctx):
        q, k, v, out, layer_name, _, _, dep = self._args(node)
        qc, kc, vc = ctx.coll(q), ctx.coll(k), ctx.coll(v)
        D = ctx.config.get("D") or ctx.allowed_dims.get("D")
        qs, ks = shape_of(qc.type)[0], shape_of(kc.type)[0]
        T = qs[0]
        if len(qs) == 3:
            NH, D_ = qs[1], qs[2]
        else:
            if D is None:
                refuse("shape", node, ctx.module_path(node), "head dim D is needed to split flat q/k/v rows: pass config={'D': ...} or 3-D q/k/v")
            NH, D_ = qs[1] // D, D
        if D is not None and D_ != D:
            refuse("shape", node, ctx.module_path(node), f"head dim {D_} vs config D={D}")
        D = D_
        KVH = (ks[1] * (ks[2] if len(ks) > 2 else 1)) // D
        if NH % KVH:
            refuse("shape", node, ctx.module_path(node), f"NH={NH} not a multiple of KVH={KVH}")
        qrows = qc.reshape(Array(T, Array(NH, Array(D, Value(16)))))
        krows = kc.reshape(Array(T, Array(KVH, Array(D, Value(16)))))
        vrows = vc.reshape(Array(T, Array(KVH, Array(D, Value(16)))))
        hist = ctx.state.setdefault("kv_history", {})
        owner = ctx.state.setdefault("kv_owner", {})
        mp = ctx.module_path(node)
        if layer_name in owner and owner[layer_name] != mp:
            refuse("side_input", node, mp, f"KV history {layer_name!r} is written from two different modules ({owner[layer_name]} and {mp})")
        owner.setdefault(layer_name, mp)
        pending = ctx.state.setdefault("kv_pending", {})
        if dep is not None:
            pending.pop(layer_name, None)
        elif layer_name in pending:
            refuse("side_input", node, mp, f"a pending KV write to {layer_name!r} was not consumed by this attention call")
        krow_list, vrow_list = hist.setdefault(layer_name, ([], []))
        P0 = len(krow_list)
        for i in range(T):
            krow_list.append(krows[i])
            vrow_list.append(vrows[i])
        ctx.check_dim("T", T, node)
        ctx.check_dim("CONTEXT", P0 + T, node)
        outs = []
        obs_sel = (_OBSERVED["attention_impls"] or {}).get(layer_name, {})
        role, extra, window = attn_kind_for(obs_sel)
        # [R15 dense] both registered chains fold `softmax_scale = float(D ** -0.5)` into their constants (`b1._scale_log2_bits`,
        # `dense.softcap_mul_bits`); a layer whose selector scale differs (Gemma-2-9B/27B: `query_pre_attn_scalar != head_dim`) has no kind.
        obs_scale = obs_sel.get("scale")
        if obs_scale is not None and (not isinstance(obs_scale, (int, float)) or float(obs_scale) != float(D ** -0.5)):
            refuse("numerics_unregistered", node, mp, f"layer {layer_name!r} has an OBSERVED attention scale {obs_scale!r} != D^-0.5 = {D ** -0.5!r} "
                   f"(D={D}); the registered attention kinds fold softmax_scale = D^-0.5 and no kind takes another scale")
        # [R13 hopper] target-keyed chain: on the accepted target (Ampere step, FA2) the role stays `attention` / `attention_softcap`
        # (digests unchanged); otherwise `attention_dot` = Attention_v2{BN, DOT, INV} with the block size of the launch the target's FA
        # version issues (FA3's tile rule reads the launch's query rows: this call's T when it opens a history, 1 per request on decode).
        from verity_vllm.registry import targets as TG
        tp = active_target_profile()
        accepted = TG.uses_accepted_kinds(tp)
        geom = None
        # [R15 hopper HOP-03, M-0225] the tile rule reads the LAUNCH's max_seqlen_q, not this request's rows: under varlen every sequence of
        # the launch runs the tile of the longest one.  The request's step index: 0 when this call opens the layer's history (prefill,
        # own rows = T), else P0 - LP + 1 (decode, own rows = 1; LP = the rows the history was opened with).  The declared launch context
        # (`observe_launch_context`, derive_step --launch-max-seqlen-q, from the workload's co-scheduled prompts) raises it; absent, own rows.
        lp_of = ctx.state.setdefault("launch_LP", {})
        if P0 == 0:
            lp_of[layer_name] = T
        req_step = 0 if P0 == 0 else P0 - int(lp_of.get(layer_name, P0)) + 1
        own_rows = T if P0 == 0 else 1
        try:
            launch_sq, launch_src = TG.launch_seqlen_q(own_rows, req_step, active_launch_context())
        except TG.LaunchContextConflict as e:
            if tp.effective_flash_attn_version() == 3 and not accepted:
                refuse("numerics_unregistered", node, mp, f"layer {layer_name!r} request step {req_step}: FA3 tile (kBlockN) is not one value -- {e}",
                       launch_context="conflict", request_step=req_step)
            launch_sq, launch_src = own_rows, "own rows (context conflict ignored: the FA2 tile does not read seqlen_q)"
        except TG.LaunchContextInvalid as e:
            # [HOP-03b, rev-pm M-0326 T1/T2] a declared context that omits this step or contradicts the request's own rows is a wrong
            # structural input: refused on EVERY target (FA2's tile does not read it, but the Program would still carry a false declaration).
            refuse("numerics_unregistered", node, mp, f"layer {layer_name!r} request step {req_step}: invalid declared launch context -- {e}",
                   launch_context="invalid", request_step=req_step, own_rows=own_rows)
        if not accepted:
            if role != "attention":
                refuse("numerics_unregistered", node, mp, f"layer {layer_name!r}: kind {role!r} has no target-keyed variant on {tp.tensor_core_family} "
                       f"(FA{tp.effective_flash_attn_version()}); only the plain causal GQA chain is registered off the accepted target")
            geom = TG.attention_geometry(tp, D, launch_sq, NH, KVH)
        for i in range(T):
            P = P0 + i + 1
            if window is not None and P > window:
                refuse("numerics_unregistered", node, mp, f"layer {layer_name!r} has an OBSERVED sliding window of {window} keys and position {P - 1} "
                       f"has {P} keys in its history: the window is active and no windowed attention kind is registered "
                       f"(positions < {window} are exact under the unwindowed kind)")
            if geom is None:
                fn = ctx.kind(role, node, T=P, NH=NH, KVH=KVH, D=D, **extra)
            else:
                fn = ctx.kind("attention_dot", node, T=P, NH=NH, KVH=KVH, D=D, BN=geom["BN"], DOT=geom["DOT"], INV=geom["INV"])
            kcat = Coll(Array(P, Array(KVH, Array(D, Value(16)))), concat([r.refs for r in krow_list[:P]]))
            vcat = Coll(Array(P, Array(KVH, Array(D, Value(16)))), concat([r.refs for r in vrow_list[:P]]))
            outs.append(ctx.call(fn, qrows[i], kcat, vcat, node=node, rule=self, name=f"{ctx.hint_name(node)}[t={P - 1}]"))
        ctx.rebind(out, array_of(outs).reshape(_tensor_type(qs, 16)), self, what="output")
        obs = (_OBSERVED["attention_impls"] or {}).get(layer_name, {})
        ctx.record(self, node, {"layer_name": layer_name, "positions": [P0, P0 + T - 1], "NH": NH, "KVH": KVH, "D": D, "history_rows_after": P0 + T,
                                "kv_write_consumed": dep is not None, "assumed_causal": True,
                                "attention_impl_observed": attn_lookup_key(obs), "attention_pins": {**ATTN_PINS, "vllm_flash_attn_version": tp.effective_flash_attn_version()},
                                "attention_kind_selected": {"role": role if geom is None else "attention_dot", **extra, "inert_window": window,
                                                            **({} if geom is None else {"BN": geom["BN"], "DOT": geom["DOT"].name, "INV": geom["INV"].name,
                                                                                        "fa_version": geom["fa_version"], "tensor_core_family": tp.tensor_core_family,
                                                                                        # [R15 hopper HOP-03] what the tile rule read and where it came from
                                                                                        "launch_seqlen_q": launch_sq, "launch_seqlen_q_source": launch_src,
                                                                                        "request_step": req_step, "own_rows": own_rows})}},
                   emitted=f"{fn.id} x{T} (T={P0 + 1}..{P0 + T})")


# -- generic elementwise / activation bindings (dense2 vocabulary; none names a model) ----------------------------------------


def _C_op(name: str) -> set:
    from verity_vllm.frontend.rules.vllm_iface import real_C
    C = real_C()
    return {getattr(C, name).default} if C is not None and hasattr(C, name) else set()


class ScalarTensorMulRule(OpRule):
    """`aten.mul.Tensor(x bf16 [.., N], s bf16 0-d tensor)` -> `scale_tensor{N}` (Bf16MulScalarTensor_v1) per row, `s` shared.
    Keyed on the observed dtypes and the scalar's rank — the scalar is a graph value (a buffer), never a constant folded in."""
    modes = frozenset({"profile"})
    info = RuleInfo("ScalarTensorMul[aten]", 1, "op-binding", applies_to="aten.mul.Tensor(x bf16 [.., N], s bf16 [])",
                    preconditions=("x is a bf16 value collection of rank >= 1", "s is a bf16 value collection of rank 0 (or [1])"),
                    evidence_domain="structural; numerics: Bf16MulScalarTensor_v1 (bf16 x bf16 scalar tensor, bf16 out)")

    def targets(self):
        return {aten.mul.Tensor}

    @staticmethod
    def _operands(node, ctx):
        """(x, s) with `s` the scalar tensor, whichever order the graph wrote them (`aten.mul` is commutative and the kernel
        computes the same f32 product); None when neither operand is a 0-d / [1] tensor."""
        a, b = node.args
        for x, sc in ((a, b), (b, a)):
            if isinstance(ctx.val(sc), Coll) and shape(ctx, sc) in ((), (1,)) and shape(ctx, x):
                return x, sc
        return None

    def applies(self, node, ctx):
        if len(node.args) != 2 or not all(isinstance(a, torch.fx.Node) for a in node.args):
            return "operands are not both graph values"
        ops = self._operands(node, ctx)
        if ops is None:
            return f"neither operand is a scalar tensor (shapes {shape(ctx, node.args[0])} x {shape(ctx, node.args[1])})"
        x, sc = ops
        r = _row_value(ctx, x, "x")
        if r:
            return r
        if not dtype_is(ctx, sc, BF16):
            # bf16 x f32-tensor promotes (a 0-d tensor does not, but a [1] one does) / f32 x f32: other numerics, refused
            return f"numerics_unregistered: s has dtype {ctx.dtype_of(sc)}; Bf16MulScalarTensor_v1 is the bf16 x bf16-scalar-tensor kernel"
        return True

    def emit(self, node, ctx):
        x, sc = self._operands(node, ctx)
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        scol = ctx.coll(sc).reshape(Array(1, Value(16)))
        fn = ctx.kind("scale_tensor", node, N=N)
        outs = [ctx.call(fn, r, scol, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows), "scalar_shape": list(shape(ctx, sc))}, emitted=f"{fn.id} x{len(rows)}")


class GeluTanhAndMulRule(InPlaceRule):
    """`_C.gelu_tanh_and_mul(out!, x[.., 2I])` -> `gelu_tanh_mul{I}` (GeluTanhMul_v1) per row; `out` is rebound (mirrors SiluAndMul)."""
    modes = frozenset({"profile"})
    info = RuleInfo("GeluTanhAndMul[vllm]", 1, "in-place-binding", applies_to="_C.gelu_tanh_and_mul(Tensor! out, x[.., 2I])",
                    preconditions=("x is bf16 [.., 2I]", "out is an unwritten allocation (or value) [.., I]"),
                    evidence_domain="structural; numerics: GeluTanhMul_v1 (vLLM _C.gelu_tanh_and_mul)")

    def targets(self):
        return _C_op("gelu_tanh_and_mul")

    def applies(self, node, ctx):
        out, x = node.args[:2]
        r = _row_value(ctx, x, "x")
        if r:
            return r
        if not isinstance(ctx.val(out), (Alloc, Coll)):
            return "out is not an allocation/value"
        xs, os_ = shape(ctx, x), shape(ctx, out)
        if xs[-1] % 2 or os_ != xs[:-1] + (xs[-1] // 2,):
            return f"out {os_} vs x {xs}"
        return True

    def emit(self, node, ctx):
        out, x = node.args[:2]
        rows, lead = rows_of(ctx.coll(x))
        I = rows[0].type.n // 2
        fn = ctx.kind("gelu_tanh_mul", node, I=I)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(out, from_rows(outs, lead), self, what="out")
        ctx.record(self, node, {"I": I, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


class Bf16ScalarChainRule(OpRule):
    """bf16 row-wise ops with a static Python-float constant: `aten.div.Tensor(x, C)` -> `bf16_div_scalar{N,C}`,
    `aten.mul.Tensor(x, C)` -> `bf16_mul_scalar{N,C}`, `aten.tanh.default(x)` -> `bf16_tanh{N}`. The constant is part of the kind."""
    modes = frozenset({"profile"})
    info = RuleInfo("Bf16ScalarChain[aten]", 1, "op-binding", applies_to="aten.div.Tensor / aten.mul.Tensor (x bf16, C: float) ; aten.tanh.default(x bf16)",
                    preconditions=("x is a bf16 value collection of rank >= 1", "C is a static Python float (div/mul)"),
                    evidence_domain="structural; numerics: Bf16DivScalar_v1 / Bf16MulScalar_v1 / Bf16Tanh_v1")

    def targets(self):
        return {aten.div.Tensor, aten.mul.Tensor, aten.tanh.default}

    def applies(self, node, ctx):
        x = node.args[0]
        r = _row_value(ctx, x, "x")
        if r:
            return r
        if node.target is aten.tanh.default:
            return True
        if len(node.args) != 2 or not isinstance(node.args[1], (float, int)) or isinstance(node.args[1], bool):
            return "the second operand is not a static Python scalar"
        return True

    def emit(self, node, ctx):
        x = node.args[0]
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        if node.target is aten.tanh.default:
            fn, det = ctx.kind("bf16_tanh", node, N=N), {"N": N}
        elif node.target is aten.div.Tensor:
            fn, det = ctx.kind("bf16_div_scalar", node, N=N, C=float(node.args[1])), {"N": N, "C": float(node.args[1])}
        else:
            fn, det = ctx.kind("bf16_mul_scalar", node, N=N, C=float(node.args[1])), {"N": N, "C": float(node.args[1])}
        member = _softcap_member(node, ctx) if node.target is aten.mul.Tensor else None
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}/{member}[{i}]" if member else f"{ctx.hint_name(node)}[{i}]")
                for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {**det, "rows": len(rows), **({"member": member} if member else {})}, emitted=f"{fn.id} x{len(rows)}")


def _softcap_member(node, ctx) -> str | None:
    """[R15 dense / manifest] `aten.mul.Tensor(tanh(x / C), C)` — the last op of a logits soft cap `C * tanh(x / C)` — produces the value
    its module returns: the row names carry the member hint `out` (the manifest's `members_for` reads it; the div / tanh rows are the
    chain's interior).  Structural test only: the operand is a `aten.tanh` of a `aten.div.Tensor(_, C)` with the same C, in one module."""
    x = node.args[0]
    if not (isinstance(x, torch.fx.Node) and x.target is aten.tanh.default and isinstance(x.args[0], torch.fx.Node)):
        return None
    d = x.args[0]
    if d.target is not aten.div.Tensor or len(d.args) != 2 or not isinstance(d.args[1], (int, float)) or float(d.args[1]) != float(node.args[1]):
        return None
    if ctx.module_path(d) != ctx.module_path(node):
        return None
    return "out"


# -- the unfused norm chain (`ir.ops.rms_norm` / `ir.ops.fused_add_rms_norm` as ATen on an f32 copy) [R15 dense, F-dense-05] -----------
#
# A model whose norm module runs `forward_native` (vLLM `GemmaRMSNorm`; `RMSNorm` off the BI branch) exports the norm as a chain of ATen
# ops on an f32 copy of the bf16 activation:
#
#     w1  = w.to(f32) + 1.0                     `WidenBf16ToF32` (no gates)  ->  `AddScalar` -> AddScalarBf16{N, C=1.0}     (GemmaRMSNorm's (1 + w))
#     x32 = x.to(f32)                           `WidenBf16ToF32` (no gates: the f32 tensor keeps the bf16 WORDS)
#     [s32 = x32 + res.to(f32); res' = s32.to(bf16)]  `AddWidened` -> AddWidenedBf16{N};  `NarrowF32ToBf16` -> NarrowF32ToBf16{N} (member "1")
#     v   = pow(x32 | s32, 2)                   `Square`      -> SquareBf16{N} | SquareF32{N}
#     m   = mean(v, [-1], keepdim=True)         `MeanDimBI`   -> MeanTriton{N}                  (BI `mean_kernel`; dispatch DERIVED, sources pinned)
#     r   = rsqrt(m + eps)                      `AddScalar`   -> AddScalarF32{N=1, C=eps};  `Rsqrt` -> RsqrtF32{N=1}
#     y   = (x32 | s32) * r                     `MulF32`      -> ScaleRowBf16{N} | ScaleRowF32{N}
#     y1  = y.to(f32) * w1                      `ViewDtype` (same width: a view);  `MulF32` -> MulVecF32{N}
#     out = y1.to(bf16)                         `NarrowF32ToBf16` -> NarrowF32ToBf16{N} (member "out" | "0")
#
# The widening is EXACT, so no gate is emitted for it: the f32 tensor's value collection keeps the bf16 words (`Value<16>` leaves under an
# f32 fx tensor) and every Definition that reads it widens internally (`SquareBf16`, `ScaleRowBf16`, `AddScalarBf16`, `AddWidenedBf16` —
# the SAME specialization ids the eager Match fold resolves the chain to, `gen_dense_gemma2_patterns`).  A reader that has no `*Bf16`
# variant (rsqrt, the row-vector multiply, the narrowing, mean) refuses a widened operand explicitly; any OTHER rule that reached such a
# value would fail the Builder's type check (`Value<16>` where the Definition takes `Value<32>`) — fail-closed by construction.  Rows:
# one Definition call per trailing-axis row in row-major order, exactly as the eager kernels' `PerRow` operands.  Member hints (`/out`,
# `/0`, `/1` on the narrowing rows; nothing on the interior) name the values the norm module returns, for the required-value manifest.


def widened_bf16(ctx, a) -> bool | None:
    """True when graph value `a` is an f32 tensor whose bound collection carries bf16 words (the transparent widening of
    `WidenBf16ToF32`, or a view of one), False for a genuine f32 value, None when `a` is not an f32 value collection."""
    if not isinstance(a, torch.fx.Node) or not dtype_is(ctx, a, torch.float32):
        return None
    v = ctx.val(a)
    if not isinstance(v, Coll):
        return None
    return shape_of(v.type)[1] == 16


def _f32_rows(ctx, a, name: str) -> str | None:
    """A refusal string unless `a` is an f32 value collection of rank >= 1 (widened or genuine)."""
    if not isinstance(a, torch.fx.Node):
        return f"{name} is not a graph value"
    v = ctx.val(a)
    if not isinstance(v, Coll):
        return f"{name} is not a value collection ({type(v).__name__})"
    if not dtype_is(ctx, a, torch.float32):
        return f"numerics_unregistered: {name} has dtype {ctx.dtype_of(a)}; the f32 chain kinds take float32 tensors"
    if not shape(ctx, a):
        return f"{name} is a scalar"
    return None


def _static_scalar(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


_CAST_OPS = {aten.to.dtype, aten._to_copy.default, aten.to.dtype_layout, aten.to.device}


class WidenBf16ToF32Rule(ViewRule):
    """`aten.to.dtype(x bf16, float32)` (also `_to_copy` / `to.dtype_layout` / `to.device` with dtype=float32): the EXACT widening of a
    bf16 tensor. No gate: the f32 tensor denotes the same bf16 words, and each reader binds the Definition that widens internally
    (`SquareBf16`, `ScaleRowBf16`, `AddScalarBf16`, `AddWidenedBf16`); a reader without such a variant refuses (rules below) or fails
    the Builder's type check. This is the frontend counterpart of the eager fold's `widened_source` (the widened allocation gets no refs)."""
    modes = frozenset({"profile"})
    info = RuleInfo("WidenBf16ToF32[aten]", 1, "view", applies_to="aten.to.dtype / _to_copy / to.dtype_layout / to.device (x bfloat16 -> float32)",
                    preconditions=("operand is a bf16 value collection of rank >= 1 (16-bit leaves)", "result dtype is float32 (same shape)"),
                    evidence_domain="structural: an exact conversion carries no rounding; readers bind the *Bf16 kinds (Value<16> leaves under an f32 tensor)")

    def targets(self):
        return set(_CAST_OPS)

    def applies(self, node, ctx):
        x = node.args[0]
        if not isinstance(x, torch.fx.Node) or not isinstance(ctx.val(x), Coll):
            return f"operand is not a value collection ({type(ctx.val(x)).__name__ if isinstance(x, torch.fx.Node) else type(x).__name__})"
        if not (dtype_is(ctx, x, BF16) and dtype_is(ctx, node, torch.float32)):
            return f"{ctx.dtype_of(x)} -> {ctx.dtype_of(node)} is not the bfloat16 -> float32 widening"
        if shape_of(ctx.val(x).type)[1] != 16:
            return "operand collection does not carry 16-bit words"
        if not shape(ctx, x) or shape(ctx, node) != shape(ctx, x):
            return f"shape {shape(ctx, x)} -> {shape(ctx, node)} is not a same-shape conversion"
        return True

    def emit(self, node, ctx):
        x = node.args[0]
        ctx.define(node, ctx.coll(x), view_of=x)
        ctx.record(self, node, {"shape": list(shape(ctx, x)), "widened_words": 16}, emitted="transparent widening (no gates; readers bind *Bf16 kinds)")


class NarrowF32ToBf16Rule(OpRule):
    """`aten.to.dtype(y f32 [.., N], bfloat16)` -> `narrow_f32_bf16{N}` (NarrowF32ToBf16_v1: `cvt.rn.bf16.f32`, NaN -> 0x7FFF) per row.
    A genuine f32 operand only: narrowing a widened bf16 view is the identity on the words and is not a kernel vLLM runs in this chain
    (refused). Row names carry the member the norm module returns (`/1` for the residual `s32.to(bf16)`, `/0` for the normed output next
    to a residual, `/out` alone) — the required-value manifest's hint."""
    modes = frozenset({"profile"})
    info = RuleInfo("NarrowF32ToBf16[aten]", 1, "op-binding", applies_to="aten.to.dtype / _to_copy / to.dtype_layout / to.device (y float32 -> bfloat16)",
                    preconditions=("operand is a genuine f32 value collection of rank >= 1 (32-bit leaves; a widened bf16 view is refused)",
                                   "result dtype is bfloat16 (same shape)"),
                    evidence_domain="structural; numerics: NarrowF32ToBf16_v1 (F2fpBf16 per element)")

    def targets(self):
        return set(_CAST_OPS)

    def applies(self, node, ctx):
        x = node.args[0]
        if not isinstance(x, torch.fx.Node) or not isinstance(ctx.val(x), Coll):
            return f"operand is not a value collection ({type(ctx.val(x)).__name__ if isinstance(x, torch.fx.Node) else type(x).__name__})"
        if not (dtype_is(ctx, x, torch.float32) and dtype_is(ctx, node, BF16)):
            return f"{ctx.dtype_of(x)} -> {ctx.dtype_of(node)} is not the float32 -> bfloat16 narrowing"
        if widened_bf16(ctx, x):
            return "numerics_unregistered: narrowing a widened bf16 view (the identity on the words) is not a kernel of the modelled chain"
        if not shape(ctx, x) or shape(ctx, node) != shape(ctx, x):
            return f"shape {shape(ctx, x)} -> {shape(ctx, node)} is not a same-shape conversion"
        return True

    def emit(self, node, ctx):
        x = node.args[0]
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        fn = ctx.kind("narrow_f32_bf16", node, N=N)
        member = self._member(node, ctx)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}/{member}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows), "member": member}, emitted=f"{fn.id} x{len(rows)}")

    @staticmethod
    def _member(node, ctx) -> str:
        """`1` when the narrowed value is the residual sum (`AddWidened` fired on the operand), `0` when the module also produced one
        (fused form: (normed, residual)), else `out` (a module returning one tensor)."""
        adds = ctx.state.get("widened_adds", {})
        if node.args[0] in adds:
            return "1"
        mp = ctx.module_path(node)
        if mp is not None and any(ctx.module_path(a) == mp for a in adds):
            return "0"
        return "out"


class SquareRule(OpRule):
    """`aten.pow.Tensor_Scalar(x f32 [.., N], 2)` -> `square_bf16{N}` (x a widened bf16 view) | `square_f32{N}` per row: ATen's CUDA
    `pow_tensor_scalar_kernel` special-cases exponent 2 as `base * base` (one FMUL, RN). Any other exponent is refused."""
    modes = frozenset({"profile"})
    info = RuleInfo("Square[aten]", 1, "op-binding", applies_to="aten.pow.Tensor_Scalar(x float32 [.., N], 2)",
                    preconditions=("x is an f32 value collection of rank >= 1 (widened bf16 view -> SquareBf16, genuine f32 -> SquareF32)",
                                   "exponent is the static scalar 2"),
                    evidence_domain="structural; numerics: SquareBf16_v1 / SquareF32_v1 (one FMUL per element)")

    def targets(self):
        return {aten.pow.Tensor_Scalar}

    def applies(self, node, ctx):
        r = _f32_rows(ctx, node.args[0], "x")
        if r:
            return r
        e = _static_scalar(node.args[1]) if len(node.args) > 1 else None
        if e != 2.0:
            return f"numerics_unregistered: exponent {node.args[1]!r} is not 2 (only the squared form has a registered kind)"
        return True

    def emit(self, node, ctx):
        x = node.args[0]
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        w = widened_bf16(ctx, x)
        fn = ctx.kind("square_bf16" if w else "square_f32", node, N=N)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows), "operand_widened_bf16": bool(w)}, emitted=f"{fn.id} x{len(rows)}")


class MeanDimBIRule(OpRule):
    """`aten.mean.dim(x f32 [.., N], [-1], keepdim=True)` under batch invariance -> `mean_triton{N}` (MeanTriton_v1: the BI Triton
    `mean_kernel`, BLOCK_SIZE=1024, 4 warps) per row, output `[.., 1]`. The dispatch is DERIVED (see `MEAN_PINS`): the export must show
    BI Triton launches (the graph's own evidence that `vllm_is_batch_invariant()` holds in the construction process) and the installed
    vLLM's `mean_kernel` / `mean_dim` / `mean_batch_invariant` sources must hash to the pins; a genuine f32 operand only (no `MeanBf16`)."""
    modes = frozenset({"profile"})
    info = RuleInfo("MeanDimBI[aten]", 1, "op-binding", applies_to="aten.mean.dim(x float32 [.., N], dim=[-1], keepdim=True, dtype=None)",
                    preconditions=("x is a genuine f32 value collection of rank >= 1 (a widened bf16 view is refused)",
                                   "dim is the last axis only; keepdim=True; no dtype argument",
                                   "export facts show batch-invariant Triton launches in this graph (derived, never assumed)",
                                   "installed vLLM's mean_kernel src sha, mean_dim and mean_batch_invariant source shas == MEAN_PINS; "
                                   "enable_batch_invariant_mode registers aten::mean.dim"),
                    evidence_domain="structural; numerics: MeanTriton_v1 (BI mean_kernel reduction order, div.full.f32)")

    def targets(self):
        return {aten.mean.dim}

    @staticmethod
    def _dispatch_ok() -> str | None:
        f = _OBSERVED["export_facts"]
        if f is None:
            return "batch-invariant dispatch was not observed at construction (observe_export_facts before derive) — refusing rather than assuming"
        ev = f.get("evidence") or {}
        if not ev.get("bi_triton_kernels_launched"):
            return ("numerics_unregistered: the export shows no batch-invariant Triton launch, so `aten::mean.dim` is not known to dispatch to "
                    f"`mean_batch_invariant` (evidence {ev!r}); the eager ATen mean reduction has no registered kind")
        mo = ev.get("mean_dim_override") or {}
        if mo.get("error") or not mo:
            return f"batch_invariant.mean_kernel not observable in this process ({mo.get('error', 'no mean_dim_override facts')}) — refusing"
        for k in ("kernel", "src_sha256", "mean_dim_src_sha256", "mean_batch_invariant_src_sha256"):
            if mo.get(k) != MEAN_PINS[k]:
                return f"numerics_unregistered: installed {k}={mo.get(k)!r} differs from the pinned {MEAN_PINS[k]!r} (MeanTriton_v1 models the pinned kernel)"
        if mo.get("enable_batch_invariant_mode_registers_mean_dim") is not True:
            return "numerics_unregistered: the installed enable_batch_invariant_mode does not register aten::mean.dim"
        return None

    def applies(self, node, ctx):
        x = node.args[0]
        r = _f32_rows(ctx, x, "x")
        if r:
            return r
        if widened_bf16(ctx, x):
            return "numerics_unregistered: mean of a widened bf16 view has no Definition (MeanTriton_v1 reads f32)"
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keep = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        rank = len(shape(ctx, x))
        if not isinstance(dims, (list, tuple)) or len(dims) != 1 or int(dims[0]) not in (-1, rank - 1):
            return f"dim {dims!r} is not the last axis"
        if keep is not True:
            return "keepdim is not True"
        if node.kwargs.get("dtype") is not None or (len(node.args) > 3 and node.args[3] is not None):
            return "an explicit dtype argument is not the modelled call"
        if shape(ctx, node) != tuple(shape(ctx, x)[:-1]) + (1,):
            return f"output shape {shape(ctx, node)} is not [.., 1]"
        return self._dispatch_ok() or True

    def emit(self, node, ctx):
        x = node.args[0]
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        fn = ctx.kind("mean_triton", node, N=N)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows), "pins": MEAN_PINS,
                                "mean_dim_override_observed": (_OBSERVED["export_facts"].get("evidence") or {}).get("mean_dim_override")},
                   emitted=f"{fn.id} x{len(rows)}")


class RsqrtRule(OpRule):
    """`aten.rsqrt.default(x f32 [.., N])` -> `rsqrt_f32{N}` (RsqrtF32_v1: `rsqrt.approx.f32`, MUFU.RSQ + range scaling) per row; a
    genuine f32 operand only."""
    modes = frozenset({"profile"})
    info = RuleInfo("Rsqrt[aten]", 1, "op-binding", applies_to="aten.rsqrt.default(x float32 [.., N])",
                    preconditions=("x is a genuine f32 value collection of rank >= 1 (a widened bf16 view is refused)",),
                    evidence_domain="structural; numerics: RsqrtF32_v1 (RsqrtApprox_v1 per element)")

    def targets(self):
        return {aten.rsqrt.default}

    def applies(self, node, ctx):
        r = _f32_rows(ctx, node.args[0], "x")
        if r:
            return r
        if widened_bf16(ctx, node.args[0]):
            return "numerics_unregistered: rsqrt of a widened bf16 view has no Definition (RsqrtF32_v1 reads f32)"
        return True

    def emit(self, node, ctx):
        rows, lead = rows_of(ctx.coll(node.args[0]))
        N = rows[0].type.n
        fn = ctx.kind("rsqrt_f32", node, N=N)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


class AddScalarRule(OpRule):
    """`aten.add.Tensor(x f32 [.., N], C)` with a static Python scalar -> `add_scalar_bf16{N,C}` (x a widened bf16 view: GemmaRMSNorm's
    `weight.float() + 1.0`) | `add_scalar_f32{N,C}` (`variance + eps`) per row; ATen's CUDA `add_kernel` in opmath f32 with the scalar
    narrowed once (one FADD, RN). The constant is part of the kind. `alpha` must be 1."""
    modes = frozenset({"profile"})
    info = RuleInfo("AddScalar[aten]", 1, "op-binding", applies_to="aten.add.Tensor(x float32 [.., N], C: static scalar), alpha == 1",
                    preconditions=("x is an f32 value collection of rank >= 1 (widened -> AddScalarBf16, genuine -> AddScalarF32)",
                                   "the second operand is a static Python scalar", "alpha == 1"),
                    evidence_domain="structural; numerics: AddScalarBf16_v1 / AddScalarF32_v1 (one FADD per element)")

    def targets(self):
        return {aten.add.Tensor}

    def applies(self, node, ctx):
        if len(node.args) != 2:
            return "not a two-operand add"
        r = _f32_rows(ctx, node.args[0], "x")
        if r:
            return r
        if _static_scalar(node.args[1]) is None:
            return "the second operand is not a static Python scalar"
        if node.kwargs.get("alpha", 1) not in (1, 1.0):
            return "alpha != 1"
        return True

    def emit(self, node, ctx):
        x, c = node.args[0], _static_scalar(node.args[1])
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        w = widened_bf16(ctx, x)
        fn = ctx.kind("add_scalar_bf16" if w else "add_scalar_f32", node, N=N, C=c)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        if w and ctx.weight_of(x) is not None:
            ctx.bind_weight_role(x, self, "norm.weight_plus_one", N=N, C=c)
        ctx.record(self, node, {"N": N, "C": c, "rows": len(rows), "operand_widened_bf16": bool(w)}, emitted=f"{fn.id} x{len(rows)}")


class AddWidenedRule(OpRule):
    """`aten.add.Tensor(x f32 [.., N], r f32 [.., N])`, both the exact widening of bf16 tensors (`fused_add_rms_norm`'s eager fallback:
    `x.to(f32) + residual.to(f32)`) -> `add_widened_bf16{N}` (AddWidenedBf16_v1: one FADD, RN) per row. An f32 + f32 add whose operands
    are not both widened has no Definition and is refused; `alpha` must be 1."""
    modes = frozenset({"profile"})
    info = RuleInfo("AddWidened[aten]", 1, "op-binding", applies_to="aten.add.Tensor(x float32 [.., N], r float32 [.., N]) both widened bf16 views, alpha == 1",
                    preconditions=("both operands are f32 value collections of one shape whose leaves are bf16 words (WidenBf16ToF32 views)",
                                   "alpha == 1"),
                    evidence_domain="structural; numerics: AddWidenedBf16_v1 (one FADD per element)")

    def targets(self):
        return {aten.add.Tensor}

    def applies(self, node, ctx):
        if len(node.args) != 2:
            return "not a two-operand add"
        for nm, a in (("x", node.args[0]), ("r", node.args[1])):
            r = _f32_rows(ctx, a, nm)
            if r:
                return r
        if shape(ctx, node.args[0]) != shape(ctx, node.args[1]):
            return "operands differ in shape (broadcast is not a bound rule)"
        if node.kwargs.get("alpha", 1) not in (1, 1.0):
            return "alpha != 1"
        if not (widened_bf16(ctx, node.args[0]) and widened_bf16(ctx, node.args[1])):
            return "numerics_unregistered: f32 + f32 add whose operands are not both widened bf16 views has no Definition"
        return True

    def emit(self, node, ctx):
        ra, lead = rows_of(ctx.coll(node.args[0]))
        rb, _ = rows_of(ctx.coll(node.args[1]))
        N = ra[0].type.n
        fn = ctx.kind("add_widened_bf16", node, N=N)
        outs = [ctx.call(fn, x, y, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, (x, y) in enumerate(zip(ra, rb))]
        ctx.define(node, from_rows(outs, lead))
        ctx.state.setdefault("widened_adds", {})[node] = ctx.module_path(node)   # `NarrowF32ToBf16` names the residual member from this
        ctx.record(self, node, {"N": N, "rows": len(ra)}, emitted=f"{fn.id} x{len(ra)}")


class MulF32Rule(OpRule):
    """The f32 `aten.mul.Tensor` forms of the unfused norm: `x[.., N] * s[.., 1]` (a row scaled by its own f32 scalar) ->
    `scale_row_bf16{N}` (x a widened bf16 view) | `scale_row_f32{N}`; `x[.., N] * w[N]` (a shared f32 row vector, `[1,..,1,N]` too) ->
    `mul_vec_f32{N}` (a genuine f32 x only). One FMUL per element, RN. Any other broadcast is refused."""
    modes = frozenset({"profile"})
    info = RuleInfo("MulF32[aten]", 1, "op-binding", applies_to="aten.mul.Tensor(x float32 [.., N], s float32 [.., 1]) ; aten.mul.Tensor(x float32 [.., N], w float32 [N])",
                    preconditions=("x is an f32 value collection of rank >= 2 (row-scale: widened or genuine; row-vector: genuine only)",
                                   "the other operand is f32: [.., 1] with x's leading shape (row scale) or [N] / [1,..,1,N] (row vector)"),
                    evidence_domain="structural; numerics: ScaleRowBf16_v1 / ScaleRowF32_v1 / MulVecF32_v1 (one FMUL per element)")

    def targets(self):
        return {aten.mul.Tensor}

    def _form(self, node, ctx):
        """-> ("scale", x, s) | ("vec", x, w) | a refusal string."""
        if len(node.args) != 2:
            return "not a two-operand mul"
        x, y = node.args
        for nm, a in (("x", x), ("y", y)):
            r = _f32_rows(ctx, a, nm)
            if r:
                return r
        sx, sy = shape(ctx, x), shape(ctx, y)
        if len(sx) < 2:
            return f"x {sx} has no rows (rank >= 2 needed)"
        N = sx[-1]
        if sy == tuple(sx[:-1]) + (1,):
            if widened_bf16(ctx, y):
                return "numerics_unregistered: the per-row scale is a widened bf16 view (the chain's scale is the f32 rsqrt)"
            return "scale", x, y
        if sy == (N,) or (len(sy) >= 1 and sy[-1] == N and all(s == 1 for s in sy[:-1])):
            if widened_bf16(ctx, x):
                return "numerics_unregistered: row-vector mul of a widened bf16 view has no Definition (MulVecF32_v1 reads f32)"
            if widened_bf16(ctx, y):
                return "numerics_unregistered: the row vector is a widened bf16 view (the chain's vector is the f32 (1 + w))"
            return "vec", x, y
        return f"numerics_unregistered: f32 mul broadcast {sx} * {sy} is neither [.., N] * [.., 1] nor [.., N] * [N]"

    def applies(self, node, ctx):
        f = self._form(node, ctx)
        return f if isinstance(f, str) else True

    def emit(self, node, ctx):
        form, x, y = self._form(node, ctx)
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        if form == "scale":
            srows, _ = rows_of(ctx.coll(y))
            w = widened_bf16(ctx, x)
            fn = ctx.kind("scale_row_bf16" if w else "scale_row_f32", node, N=N)
            outs = [ctx.call(fn, r, s, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, (r, s) in enumerate(zip(rows, srows))]
            det = {"form": "row_scale", "operand_widened_bf16": bool(w)}
        else:
            wv = ctx.coll(y)
            if len(shape_of(wv.type)[0]) != 1:
                wv = wv.reshape(_tensor_type((N,), 32))
            fn = ctx.kind("mul_vec_f32", node, N=N)
            outs = [ctx.call(fn, r, wv, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
            det = {"form": "row_vector"}
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows), **det}, emitted=f"{fn.id} x{len(rows)}")


# -- [R15 fp8] the block-scaled FP8 linear: dynamic per-token-group activation quant + the blockwise CUTLASS scaled GEMM -----------------

FP8_E4M3 = torch.float8_e4m3fn
#: the quant op's pinned scalars (`per_token_group_quant_fp8(..., group_size=128, eps=1e-10, dtype=e4m3, use_ue8m0=False)` as vLLM's
#: `QuantFP8(group_shape=(1, 128), use_ue8m0=False)` calls it for `CutlassFp8BlockScaledMMKernel`); the Definition `Fp8GroupQuant_v1{K,G}`
#: hard-codes eps and the e4m3 range, so a launch with other values is refused (another kernel configuration, not this Definition).
FP8_QUANT_PINS = {"group_size": 128, "eps": 1e-10, "fp8_min": -448.0, "fp8_max": 448.0, "use_ue8m0": False}


def _f32_equal(a, b: float) -> bool:
    import numpy as _np
    return isinstance(a, (int, float)) and not isinstance(a, bool) and _np.float32(a) == _np.float32(b)


class PerTokenGroupFp8QuantRule(InPlaceRule):
    """`_C.per_token_group_fp8_quant(x bf16 [.., K], x_q! e4m3 [.., K], x_s! f32 [.., K/G], G, eps, fp8_min, fp8_max, use_ue8m0,
    column_major_scales, tma_aligned_scales)` -> `fp8_group_quant{K,G}` (Fp8GroupQuant_v1) per token row; BOTH allocations are rebound
    (`x_q` to port 0, `x_s` to port 1).  `x_s` is usually a permuted view of its allocation (column-major scales): the view node is what the
    GEMM reads, so the view node is rebound (AllocView v2).  The scale layout / TMA alignment flags are storage facts, not values."""
    modes = frozenset({"profile"})
    info = RuleInfo("PerTokenGroupFp8Quant[vllm]", 1, "in-place-binding",
                    applies_to="_C.per_token_group_fp8_quant(x, Tensor! x_q, Tensor! x_s, group_size, eps, fp8_min, fp8_max, use_ue8m0, column_major_scales, tma_aligned_scales)",
                    preconditions=("x is bf16 [.., K] with K % group_size == 0", "x_q is an unwritten allocation float8_e4m3fn [.., K]",
                                   "x_s is an unwritten allocation (or a view of one) f32 [.., K/group_size]",
                                   "group_size == 128, eps == 1e-10f, fp8_min/max == -448/448, use_ue8m0 == False (the pinned Definition)"),
                    evidence_domain="structural; numerics: Fp8GroupQuant_v1 (exact-model-tested on the H100 against _C.per_token_group_fp8_quant)")

    def targets(self):
        return _C_op("per_token_group_fp8_quant")

    @staticmethod
    def _scalars(node) -> dict:
        names = ("x", "x_q", "x_s", "group_size", "eps", "fp8_min", "fp8_max", "use_ue8m0", "column_major_scales", "tma_aligned_scales")
        d = dict(zip(names, node.args))
        d.update(node.kwargs)
        return d

    def applies(self, node, ctx):
        a = self._scalars(node)
        if not all(k in a for k in ("x", "x_q", "x_s", "group_size", "eps", "fp8_min", "fp8_max")):
            return "unexpected operator arity"
        r = _row_value(ctx, a["x"], "x")
        if r:
            return r
        xs_shape = shape(ctx, a["x"])
        G = a["group_size"]
        if not isinstance(G, int) or G != FP8_QUANT_PINS["group_size"]:
            return f"numerics_unregistered: group_size={G!r}; Fp8GroupQuant_v1 is pinned for group 128"
        if xs_shape[-1] % G:
            return f"K={xs_shape[-1]} is not a multiple of the group size {G}"
        for key in ("eps", "fp8_min", "fp8_max"):
            if not _f32_equal(a[key], FP8_QUANT_PINS[key]):
                return f"numerics_unregistered: {key}={a[key]!r} differs from the pinned {FP8_QUANT_PINS[key]!r} (another kernel configuration)"
        if bool(a.get("use_ue8m0", False)):
            return "numerics_unregistered: use_ue8m0=True (power-of-two scales, the DeepGEMM path) has no registered Definition"
        for name in ("x_q", "x_s"):
            if not isinstance(ctx.val(a[name]), Alloc):
                return f"{name} is not an unwritten allocation ({type(ctx.val(a[name])).__name__})"
        if not dtype_is(ctx, a["x_q"], FP8_E4M3):
            return f"numerics_unregistered: x_q has dtype {ctx.dtype_of(a['x_q'])}; the Definition quantises to float8_e4m3fn"
        if not dtype_is(ctx, a["x_s"], torch.float32):
            return f"x_s has dtype {ctx.dtype_of(a['x_s'])}, not float32"
        if shape(ctx, a["x_q"]) != xs_shape:
            return f"x_q {shape(ctx, a['x_q'])} vs x {xs_shape}"
        if shape(ctx, a["x_s"]) != xs_shape[:-1] + (xs_shape[-1] // G,):
            return f"x_s {shape(ctx, a['x_s'])} vs x {xs_shape} / {G}"
        return True

    def emit(self, node, ctx):
        a = self._scalars(node)
        rows, lead = rows_of(ctx.coll(a["x"]))
        K, G = rows[0].type.n, int(a["group_size"])
        fn = ctx.kind("fp8_group_quant", node, K=K, G=G)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(a["x_q"], from_rows([o[0] for o in outs], lead), self, what="x_q")
        ctx.rebind(a["x_s"], from_rows([o[1] for o in outs], lead), self, what="x_s")
        ctx.record(self, node, {"K": K, "G": G, "rows": len(rows), "eps": a["eps"], "fp8_min": a["fp8_min"], "fp8_max": a["fp8_max"],
                                "use_ue8m0": bool(a.get("use_ue8m0", False)), "column_major_scales": bool(a.get("column_major_scales", False)),
                                "tma_aligned_scales": bool(a.get("tma_aligned_scales", False))}, emitted=f"{fn.id} x{len(rows)}")


class CutlassScaledMmFp8BlockRule(InPlaceRule):
    """`_C.cutlass_scaled_mm(out! bf16 [.., N], a e4m3 [.., K], b = W.T ([K, N] view of e4m3 W [N, K]), a_scales f32 [.., K/G],
    b_scales = Ws.T ([K/G, N/G] view of f32 Ws [N/G, K/G]), bias None)` -> `scaled_mm_fp8_block{K,N,G}` (ScaledMmFp8Block_v1) per token row;
    `out` is rebound.  W and Ws are prescribed weight Inputs (the checkpoint's e4m3 words and f32 block scales); `a` / `a_scales` are the
    values `PerTokenGroupFp8Quant[vllm]` computed.  Binds the BLOCK-scaled shapes only: a per-tensor / per-token scaled_mm (scales [M,1] /
    [1,N] / 0-d) is another kernel (`ScaledMmFp8_v1` of registry.fp8, R14 QZ) and is refused here, named."""
    modes = frozenset({"profile"})
    info = RuleInfo("CutlassScaledMmFp8Block[vllm]", 1, "in-place-binding",
                    applies_to="_C.cutlass_scaled_mm(Tensor! out, a e4m3, b e4m3 [K, N], a_scales f32 [.., K/G], b_scales f32 [K/G, N/G], bias None)",
                    preconditions=("a is a float8_e4m3fn value [.., K]; out an unwritten bf16 allocation [.., N]",
                                   "b is a transposed view of an e4m3 [N, K] matrix; b_scales a transposed view of an f32 [N/G, K/G] matrix, G = 128",
                                   "a_scales is an f32 value [.., K/G] (the group scales the quant rule computed)", "bias is None",
                                   "K % G == 0 and N % G == 0"),
                    evidence_domain="structural; numerics: ScaledMmFp8Block_v1 (exact-model-tested on the H100 against the blockwise _C.cutlass_scaled_mm, both dispatch variants)")

    def targets(self):
        return _C_op("cutlass_scaled_mm")

    def applies(self, node, ctx):
        if len(node.args) < 5:
            return "unexpected operator arity"
        out, a, b, a_s, b_s = node.args[:5]
        bias = node.args[5] if len(node.args) > 5 else node.kwargs.get("bias")
        if bias is not None:
            return "numerics_unregistered: a fused bias in the blockwise FP8 epilogue has no registered Definition (vLLM adds the bias outside the kernel)"
        r = _row_value(ctx, a, "a", dtype=FP8_E4M3)
        if r:
            return r
        if not isinstance(ctx.val(out), Alloc):
            return f"out is not an unwritten allocation ({type(ctx.val(out)).__name__})"
        if not dtype_is(ctx, out, BF16):
            return f"numerics_unregistered: out has dtype {ctx.dtype_of(out)}; the Definition's epilogue is cvt.rn.bf16.f32"
        ash, bsh, ash_s, bsh_s, osh = shape(ctx, a), shape(ctx, b), shape(ctx, a_s), shape(ctx, b_s), shape(ctx, out)
        if bsh is None or len(bsh) != 2 or ash[-1] != bsh[0]:
            return f"b {bsh} is not [K, N] for a [.., K={ash[-1]}]"
        K, N = bsh
        G = FP8_QUANT_PINS["group_size"]
        if not dtype_is(ctx, b, FP8_E4M3):
            return f"numerics_unregistered: b has dtype {ctx.dtype_of(b)}; the Definition multiplies e4m3 by e4m3"
        if not (dtype_is(ctx, a_s, torch.float32) and dtype_is(ctx, b_s, torch.float32)):
            return f"scales are {ctx.dtype_of(a_s)} / {ctx.dtype_of(b_s)}, not float32"
        if ash_s is None or bsh_s is None or ash_s == ash[:-1] + (1,) or bsh_s in ((1, N), (N,), (), (1,)):
            return (f"numerics_unregistered: per-tensor / per-token scales a_scales {ash_s}, b_scales {bsh_s} -- the FastAccum per-tensor kernel "
                    f"(ScaledMmFp8_v1, registry.fp8) is not bound by this rule; this rule binds the blockwise kernel (a_scales [.., K/128], b_scales [K/128, N/128])")
        if K % G or N % G:
            return f"numerics_unregistered: K={K}, N={N} not multiples of the block {G} (vLLM pads such layers; not modelled)"
        if ash_s != ash[:-1] + (K // G,) or bsh_s != (K // G, N // G):
            return f"scale shapes a_scales {ash_s} (want [.., {K // G}]), b_scales {bsh_s} (want [{K // G}, {N // G}]) are not the 128x128 block layout"
        if osh != ash[:-1] + (N,):
            return f"out {osh} vs a {ash} x b {bsh}"
        if ctx.weight_of(b) is None or ctx.weight_of(b_s) is None:
            return "b / b_scales are not (views of) parameters; the Definition takes the checkpoint's e4m3 words and block scales as weight Inputs"
        return True

    def emit(self, node, ctx):
        out, a, b, a_s, b_s = node.args[:5]
        K, N = shape(ctx, b)
        G = FP8_QUANT_PINS["group_size"]
        w = _untransposed(ctx, b, K, N, ctx.hint_name(node) + "/w")
        ws = _untransposed(ctx, b_s, K // G, N // G, ctx.hint_name(node) + "/ws")
        ctx.bind_weight_role(b, self, "fp8_w", K=K, N=N)
        ctx.bind_weight_role(b_s, self, "fp8_ws", K=K, N=N, G=G)
        rows, lead = rows_of(ctx.coll(a))
        srows, _ = rows_of(ctx.coll(a_s))
        if len(rows) != len(srows):
            refuse("shape", node, ctx.module_path(node), f"a has {len(rows)} rows but a_scales {len(srows)}")
        fn = ctx.kind("scaled_mm_fp8_block", node, K=K, N=N, G=G)
        outs = [ctx.call(fn, r, s, w, ws, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, (r, s) in enumerate(zip(rows, srows))]
        ctx.rebind(out, from_rows(outs, lead), self, what="out")
        ctx.record(self, node, {"K": K, "N": N, "G": G, "rows": len(rows), "b_view": str(getattr(b, "target", None)),
                                "b_scales_view": str(getattr(b_s, "target", None))}, emitted=f"{fn.id} x{len(rows)}")


class SameDtypeCastRule(ViewRule):
    """`aten.to.dtype(x, dt)` / `_to_copy(x, dtype=dt)` with `dt == x.dtype`: the identity (ATen returns `self` for `to`; `_to_copy` is a
    copy of the same words).  vLLM's `BlockScaledMMLinearKernel.apply_weights` ends with `output.to(dtype=out_dtype)` on an already-bf16
    output.  Rank/shape unchanged; no gates."""
    info = RuleInfo("SameDtypeCast", 1, "view", applies_to="aten.to.dtype / _to_copy where the target dtype equals the operand's",
                    preconditions=("operand is a value collection", "target dtype == operand dtype"), evidence_domain="structural (no gates)")

    def targets(self):
        return {aten.to.dtype, aten._to_copy.default}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Coll):
            return f"operand is {type(v).__name__}"
        src, dst = ctx.dtype_of(node.args[0]), ctx.dtype_of(node)
        if src is None or dst is None or src != dst:
            return f"{src} -> {dst} is not a same-dtype cast"
        if shape(ctx, node) != shape(ctx, node.args[0]):
            return "shape changes"
        return True

    def emit(self, node, ctx):
        ctx.define(node, ctx.coll(node.args[0]), view_of=None if node.target is aten._to_copy.default else node.args[0])
        ctx.record(self, node, {"dtype": str(ctx.dtype_of(node))}, emitted="(no gates)")


# -- fused add + rms norm: the batch-invariant dispatch, derived ------------------------------------------------------------


class FusedAddRmsNormBIRule(_FusedAddRmsNormRule):
    """`FusedAddRmsNorm[vllm]` with the kernel dispatch DERIVED: binds `RMSNormFusedCuda_v2` (vLLM's generic fused_add_rms_norm_kernel,
    the one launched under batch invariance) only when the export shows the batch-invariant path (`observe_export_facts`); otherwise
    refuses — the width-8 kernel (`RMSNormFusedCuda_v1`) is superseded and not in the profile's vocabulary."""
    info = RuleInfo("FusedAddRmsNorm[vllm-bi]", 1, "in-place-binding", applies_to="_C.fused_add_rms_norm(Tensor! x, Tensor! residual, weight, eps)",
                    preconditions=("the export's fused_add_rms_norm dispatch was OBSERVED as batch-invariant (op reached via `rms_norm_batch_invariant`; "
                                   "BI Triton overrides installed and launched in the same graph) — derived, never assumed",)
                    + tuple(_FusedAddRmsNormRule.info.preconditions),
                    evidence_domain=_FusedAddRmsNormRule.info.evidence_domain)

    def applies(self, node, ctx):
        r = _fused_norm_dispatch_ok()
        if r:
            return r
        return super().applies(node, ctx)

    def emit(self, node, ctx):
        super().emit(node, ctx)
        # the parent recorded this application; add the derived dispatch to that record (one application per node)
        rec = ctx.applications[-1]
        if rec.get("node") == node.name:
            rec["rule"], rec["version"] = self.rule_id, self.info.version
            rec["preconditions_checked"] = {**rec.get("preconditions_checked", {}),
                                            "fused_add_rms_norm_dispatch": _OBSERVED["export_facts"].get("fused_add_rms_norm_dispatch"),
                                            "dispatch_evidence": _OBSERVED["export_facts"].get("evidence")}


# -- views of allocations ---------------------------------------------------------------------------------------------------


class AllocViewRule(ViewRule):
    """`aten.view/reshape(alloc, shape)` of an UNWRITTEN allocation (vLLM's `Attention.forward`: `output = torch.empty(...)` then
    `output.view(-1, NH, D)` before the kernel writes it): the view is the same uninitialised allocation with the viewed type;
    the in-place rule that writes the view rebinds it. Reading either before the write stays refused (Alloc semantics)."""
    info = RuleInfo("AllocView", 2, "view", applies_to="aten.view / reshape / flatten / unit-step slice / permute / t / transpose of an unwritten allocation",
                    preconditions=("operand is an unwritten Alloc", "view/reshape/flatten: leaf count is preserved; slice: step 1 (a sub-allocation "
                                   "of the same storage -- vLLM's MoE workspace carving `buf[:n].view(...)`); permute/t/transpose: the viewed "
                                   "type is the node's (an axis order over the same leaves -- vLLM's column-major FP8 group scales "
                                   "`torch.empty(K/G, M).permute(-1, -2)`)"), evidence_domain="structural (no gates)")

    def targets(self):
        # [R15 fp8] v2: + permute / t / transpose (the in-place rule that writes the view rebinds the VIEW node; the base stays an unread Alloc)
        return {aten.view.default, aten.reshape.default, aten._unsafe_view.default, aten.flatten.using_ints, aten.slice.Tensor,
                aten.permute.default, aten.t.default, aten.transpose.int}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Alloc):
            return f"operand is {type(v).__name__}, not an allocation"
        if node.target is aten.slice.Tensor and (len(node.args) > 4 and node.args[4] != 1):
            return f"slice step {node.args[4]} is not 1"
        return True

    def emit(self, node, ctx):
        from verity_vllm.frontend.torch_frontend import type_of_tensor
        ctx.define(node, Alloc(type_of_tensor(node.meta["val"]), node), view_of=node.args[0])


# -- token-id widening ------------------------------------------------------------------------------------------------------


class IntIdWidenRule(ViewRule):
    """`aten.to.dtype(ids: int32 value, int64)` / `_to_copy(..., dtype=int64)`: the I32 token id widened for `aten.embedding`
    (vLLM's `VocabParallelEmbedding.forward`). Value-preserving for ids in [0, 2^31): a reinterpretation, no gate; the value keeps
    its 32-bit leaves (the Embedding kind takes I32)."""
    info = RuleInfo("IntIdWiden", 1, "view", applies_to="aten.to.dtype / _to_copy from a 32-bit integer value to int64",
                    preconditions=("operand is a value collection with 32-bit integer leaves", "target dtype is int64"),
                    evidence_domain="structural: widening of a non-negative index is a reinterpretation")

    def targets(self):
        return {aten.to.dtype, aten._to_copy.default}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Coll):
            return f"operand is {type(v).__name__}"
        src, dst = ctx.dtype_of(node.args[0]), ctx.dtype_of(node)
        if src is None or dst is None or src.is_floating_point or dst.is_floating_point:
            return "not an integer -> integer cast"
        if dst != torch.int64 or width_of(src) != 32 or shape_of(v.type)[1] != 32:
            return f"{src} -> {dst} is not a 32-bit integer widened to int64"
        return True

    def emit(self, node, ctx):
        ctx.define(node, ctx.coll(node.args[0]), view_of=node.args[0])
        ctx.record(self, node, {"from": str(ctx.dtype_of(node.args[0])), "to": str(ctx.dtype_of(node)), "carried_as": "I32"}, emitted="(no gates)")


# -- the padded engine's step-boundary cache clear over implicit KV state ----------------------------------------------------


class CachePadRule(StateRule):
    """`vllm_iface.unified_cache_pad(live, zero, slot)`: for EVERY layer whose KV history this engine owns, row `slot` — the
    row the step just appended, which must be the last row of every history — now denotes `batch(SelectBf16_v1, live, row,
    zero-row)` against the step's one broadcast zero (serve3-spec §6 row muxes; the `Clear_OOB_MN` zero-row convention). The
    step's own attention has read the raw row (graph order); every later reader reads the muxed one. The caller must be an
    ancestor module of every history's owner (the engine owns its model's caches). A history with a different row count, a
    non-bf16/non-scalar zero, or a non-bit liveness is refused — nothing is synthesised for a history this call does not fit."""
    modes = frozenset({"profile"})
    info = RuleInfo("CachePad[vllm]", 1, "state", applies_to="vllm_iface.unified_cache_pad(live, zero, slot) -> ()",
                    preconditions=("live is a 1-element bool value collection (or a statically known bool)", "zero is a 1-element bf16 value collection",
                                   "every KV history has exactly slot+1 rows and is owned by a descendant of the caller's module",
                                   "at least one KV history exists"),
                    evidence_domain="structural; numerics: SelectBf16_v1 (exact-model-tested) per K/V word")

    def targets(self):
        from verity_vllm.frontend.rules.vllm_iface import targets_for
        return targets_for("unified_cache_pad")

    def applies(self, node, ctx):
        live, zero, slot = node.args[:3]
        from verity_vllm.frontend.rules.padding import _bit_value, _static_bit
        if _bit_value(ctx, live) is None and _static_bit(ctx, live) is None:
            return "live is neither a bool value collection nor a statically known bool tensor"
        if not isinstance(zero, torch.fx.Node) or not isinstance(ctx.val(zero), Coll) or not dtype_is(ctx, zero, BF16):
            return "zero is not a bf16 value collection"
        if int(torch.Size(shape(ctx, zero)).numel()) != 1:
            return f"zero has {shape(ctx, zero)} elements, not one"
        if not isinstance(slot, int):
            return "slot is not static"
        hist = ctx.state.get("kv_history", {})
        if not hist:
            return "no KV history exists yet (the cache clear precedes every attention)"
        return True

    def emit(self, node, ctx):
        from verity_vllm.frontend.rules.padding import _materialize_bits, _scalar_of
        live, zero, slot = node.args[:3]
        mp = ctx.module_path(node)
        hist = ctx.state["kv_history"]
        owner = ctx.state.get("kv_owner", {})
        c = _materialize_bits(ctx, live, self, node)
        cflat = c.reshape(Array(int(torch.Size(shape(ctx, live)).numel()), Value(1)))
        if cflat.type.n != 1:
            refuse("shape", node, mp, f"live has {cflat.type.n} elements; the step's cache clear is keyed on one bit")
        zc = ctx.coll(zero)
        zscalar = zc.reshape(Array(1, Value(16)))
        fn = ctx.kind("select_bf16", node)
        layers = []
        def _path(p) -> str:  # "a.b.c (Cls)" -> "a.b.c"; "<root> (Cls)" / None -> ""
            p = (p or "").split(" (", 1)[0]
            return "" if p in ("", "<root>") else p

        me = _path(mp)
        for layer_name in sorted(hist):
            own = _path(owner.get(layer_name))
            if not (me == "" or own == me or own.startswith(me + ".")):
                refuse("mutation", node, mp, f"KV history {layer_name!r} is owned by {owner.get(layer_name)!r}, not a descendant of {mp!r}")
            krow_list, vrow_list = hist[layer_name]
            if slot != len(krow_list) - 1:
                refuse("mutation", node, mp, f"unified_cache_pad rewrites row {slot} of {layer_name!r} whose history has {len(krow_list)} rows: "
                                             f"only the row this step appended (row {len(krow_list) - 1}) may be rewritten at the step boundary")
            for which, rows in (("k", krow_list), ("v", vrow_list)):
                row = rows[slot]
                n = row.type.leaves
                flat = row.reshape(Array(n, Value(16)))
                zrow = Coll(Array(n, Value(16)), concat([zscalar[0].refs] * n))
                out = ctx.B.batch(fn, _scalar_of(cflat), flat, zrow, axes=(None, 0, 0), name=f"{ctx.hint_name(node)}/{layer_name}/{which}[{slot}]")
                k = len(ctx.B.nodes) - 1
                ctx.node_hints[k] = f"{ctx.hint_name(node)}/{layer_name}/{which}[{slot}]"
                ctx.node_sources[k] = node.name
                rows[slot] = out.reshape(row.type)
            layers.append(layer_name)
            ctx.state.setdefault("kv_rewrites", []).append({"layer_name": layer_name, "slot": slot, "node": node.name, "pad": True})
        ctx.record(self, node, {"slot": slot, "layers": len(layers), "row_words": hist[layers[0]][0][slot].type.leaves},
                   emitted=f"batch(SelectBf16_v1, live, row, zero-row) x{2 * len(layers)} (one K and one V row per layer; state rows rebound)")


# -- [R13 lane tp, P4] tensor-parallel collectives: `verity_tp::*` ops emitted by the construction's CollectiveBus (cb_a/tp_ops.py) ----------


def _tp_op(name: str) -> set:
    try:
        from verity_capture.experimental.cb_a.tp_ops import ensure_ops
        ensure_ops()
        return {getattr(torch.ops.verity_tp, name).default}
    except Exception:  # noqa: BLE001 — construction scaffolding not importable: the rule has no targets
        return set()


def _hint_module(hint: str) -> str:
    """The module of a naming hint (`<module>/<node>[/<member>][i]` -> `<module>`; a bare node name -> ``)."""
    head = hint.split("/", 1)[0] if "/" in hint else ""
    return head.split("[", 1)[0]


def _collective_hint(node, ctx) -> str:
    """[R19 moe F-r19-moe-1, row #70] the naming hint of a TP collective row.  A collective whose rank-local operand is produced UNDER the
    collective's own module (a row-parallel linear's all-reduce of its Gemm, the MoE runner's all-reduce of its MoeSum, the logits
    processor's all-gather of its Gemm shard) is named under that module (`<module>/<node>`) as before: the Gemm / MoeSum row under the
    module IS the rank's partial (required_manifest TP-04 `members_for`).  A collective issued MID-module -- every rank-local operand row
    lives under ANOTHER module (OLMoE `OlmoeAttention._apply_qk_norm`: q and k all-gathered inside `self_attn`, each a slice of the
    `qkv_proj` Gemm's output) -- is named at its call SITE `<module>.<fx node>/<node>`, so the manifest (`collective_site`) requires it as
    the module's k-th collective identity WITH its own `shard_rank<r>` / `part_rank<r>` member, rather than folding both calls into one
    `<module>/out` of twice the width with no partial to bind (row #70: 6,944 unbound per rank).  The rank-local operand is the one that
    is NOT a Program input (the peer's partial arrives as a `peers_<k>` Input); when neither operand is an input (a TP1-style test
    Program) both are examined.  Its producing rows are read off the Program's own refs (`node_hints` of the rows' nodes: space "n")."""
    own_mod = ctx.stack_path(node) if node.meta.get("nn_module_stack") else ""
    base = ctx.hint_name(node)
    if not own_mod:
        return base
    producers: set[str] = set()
    for a in node.args[:2]:
        if not isinstance(a, torch.fx.Node) or ctx.is_input_storage(a) is not None:
            continue
        rows, _lead = rows_of(ctx.coll(a))
        for r in rows:
            ref = r.refs[0]
            if ref[0] == "n":
                producers.add(_hint_module(ctx.node_hints.get(int(ref[1]), "")))
            else:
                producers.add("")                                            # a parameter ref: no producing row (an input; not this module's)
    if not producers or own_mod in producers:
        return base
    return f"{own_mod}.{node.name}/{node.name}"


class AllReduce2Rule(OpRule):
    """`verity_tp::all_reduce2(part_rank0 [.., N] bf16, part_rank1 [.., N] bf16)` -> `all_reduce2{N}` (`AllReduce2_v1{N}`: one bf16 add per
    element = `ncclSum` bf16 of two terms) per row.  Operands are in RANK ORDER; one of them is this rank's partial (a value the rank
    computes: a K-sharded `Gemm_v1` output or an `EmbeddingShard` row), the other the PEER INPUT bound by `tp_links.json` to the other
    rank's committed partial — the received collective output is never a fresh trusted input.  Established on the pod (R13 tp step 1):
    the pynccl all-reduce result equals `bf16(f32(P_0) + f32(P_1))` bit-exactly on every recorded element."""
    modes = frozenset({"profile"})
    info = RuleInfo("AllReduce2[tp]", 1, "op-binding", applies_to="verity_tp::all_reduce2(a [.., N] bf16, b [.., N] bf16) (rank order)",
                    preconditions=("both operands are bf16 value collections of identical shape", "world size 2"),
                    evidence_domain="structural; numerics: AllReduce2_v1 (Bf16Add per element; exact-model-tested B1 TP2 + Llama TP2 step-1 probe)")

    def targets(self):
        return _tp_op("all_reduce2")

    def applies(self, node, ctx):
        for nm, a in (("a", node.args[0]), ("b", node.args[1])):
            r = _row_value(ctx, a, nm)
            if r:
                return r
        if shape(ctx, node.args[0]) != shape(ctx, node.args[1]):
            return f"partials differ in shape ({shape(ctx, node.args[0])} vs {shape(ctx, node.args[1])})"
        return True

    def emit(self, node, ctx):
        a, b = ctx.coll(node.args[0], "part_rank0"), ctx.coll(node.args[1], "part_rank1")
        ra, lead = rows_of(a)
        rb, _ = rows_of(b)
        N = ra[0].type.n
        fn = ctx.kind("all_reduce2", node, N=N)
        hint = _collective_hint(node, ctx)                                   # [R19 moe F-r19-moe-1] site-named when mid-module
        outs = [ctx.call(fn, array_of([x, y]), node=node, rule=self, name=f"{hint}[{i}]/all_reduce") for i, (x, y) in enumerate(zip(ra, rb))]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(ra), "world": 2}, emitted=f"{fn.id} x{len(ra)}")


class AllGather2Rule(OpRule):
    """`verity_tp::all_gather2(part_rank0 [.., N] bf16, part_rank1 [.., N] bf16)` -> `all_gather2{N}` (`AllGather2_v1{N}`: the [2N] row IS
    the two shards side by side — a copy, no arithmetic) per row; rank order = column order (vLLM's `all_gather` along the last dim)."""
    modes = frozenset({"profile"})
    info = RuleInfo("AllGather2[tp]", 1, "op-binding", applies_to="verity_tp::all_gather2(a [.., N] bf16, b [.., N] bf16) (rank order, trailing axis)",
                    preconditions=("both operands are bf16 value collections of identical shape", "world size 2"),
                    evidence_domain="structural; numerics: AllGather2_v1 (concatenation; exact-model-tested)")

    def targets(self):
        return _tp_op("all_gather2")

    def applies(self, node, ctx):
        for nm, a in (("a", node.args[0]), ("b", node.args[1])):
            r = _row_value(ctx, a, nm)
            if r:
                return r
        if shape(ctx, node.args[0]) != shape(ctx, node.args[1]):
            return f"shards differ in shape ({shape(ctx, node.args[0])} vs {shape(ctx, node.args[1])})"
        return True

    def emit(self, node, ctx):
        a, b = ctx.coll(node.args[0], "part_rank0"), ctx.coll(node.args[1], "part_rank1")
        ra, lead = rows_of(a)
        rb, _ = rows_of(b)
        N = ra[0].type.n
        fn = ctx.kind("all_gather2", node, N=N)
        hint = _collective_hint(node, ctx)                                   # [R19 moe F-r19-moe-1] site-named when mid-module
        outs = [ctx.call(fn, array_of([x, y]), node=node, rule=self, name=f"{hint}[{i}]/all_gather") for i, (x, y) in enumerate(zip(ra, rb))]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(ra), "world": 2}, emitted=f"{fn.id} x{len(ra)}")


class EmbeddingShardRule(OpRule):
    """`verity_tp::embedding_shard(ids, shard [VS, H] bf16, START, END)` -> `embedding_shard{VS,H,START}` (`EmbeddingShard_v1`: the row
    `shard[id - START]` if START <= id < END else zeros) per token — one rank of vLLM's `VocabParallelEmbedding` (`_C.vocab_parallel_embedding`);
    the shard is this rank's committed weight input (role `embedding.table` at [VS, H])."""
    modes = frozenset({"profile"})
    info = RuleInfo("EmbeddingShard[tp]", 1, "op-binding", applies_to="verity_tp::embedding_shard(ids I32 value, shard [VS,H] bf16, START, END)",
                    preconditions=("shard is a bf16 value collection [VS, H] with VS == END - START", "ids is a value collection of 32-bit token ids"),
                    evidence_domain="structural; numerics: EmbeddingShard_v1 (masked shard gather; exact-model-tested)")

    def targets(self):
        return _tp_op("embedding_shard")

    def applies(self, node, ctx):
        ids, table, start, end = node.args[:4]
        r = _row_value(ctx, table, "shard")
        if r:
            return r
        shp = shape(ctx, table)
        if shp is None or len(shp) != 2:
            return "shard is not 2-D"
        if not isinstance(start, int) or not isinstance(end, int) or end - start != shp[0]:
            return f"START/END ({start}, {end}) do not span the shard's VS={shp[0] if shp else None}"
        v = ctx.val(ids)
        if isinstance(v, StaticIndex):
            return "ids are static structure"
        if not isinstance(v, Coll):
            return "ids is not a value collection"
        if shape_of(v.type)[1] != 32:
            return f"numerics_unregistered: ids carry {shape_of(v.type)[1]}-bit leaves; EmbeddingShard_v1 takes I32 token ids"
        return True

    def emit(self, node, ctx):
        ids, table = ctx.coll(node.args[0], "ids"), ctx.coll(node.args[1], "shard")
        start = int(node.args[2])
        VS, H = shape_of(table.type)[0]
        shp = shape_of(ids.type)[0]
        n = 1
        for s in shp:
            n *= s
        flat = ids.reshape(Array(n, Value(32)))
        fn = ctx.kind("embedding_shard", node, VS=VS, H=H, START=start)
        outs = [ctx.call(fn, flat[i], table, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]/shard") for i in range(n)]
        ctx.define(node, array_of(outs).reshape(_tensor_type(tuple(shp) + (H,), 16)))
        ctx.bind_weight_role(node.args[1], self, "embedding.table", V=VS, H=H)
        ctx.record(self, node, {"VS": VS, "H": H, "START": start, "tokens": n}, emitted=f"{fn.id} x{n}")


VLLM_BINDING_RULES = [TritonRmsNormRule(), TritonGemmRule(), KvCacheUpdateRule(), UnifiedAttentionWithOutputRule(), AllocViewRule(), IntIdWidenRule(),
                      CachePadRule(), FusedAddRmsNormBIRule(), ScalarTensorMulRule(), GeluTanhAndMulRule(), Bf16ScalarChainRule(),
                      # [R15 dense] the unfused norm chain (forward_native norms as ATen on an f32 copy)
                      WidenBf16ToF32Rule(), NarrowF32ToBf16Rule(), SquareRule(), MeanDimBIRule(), RsqrtRule(), AddScalarRule(), AddWidenedRule(),
                      MulF32Rule(),
                      AllReduce2Rule(), AllGather2Rule(), EmbeddingShardRule(),
                      # [R15 fp8] the block-scaled FP8 linear (Fp8LinearMethod -> CutlassFp8BlockScaledMMKernel) + the same-dtype cast it ends with
                      PerTokenGroupFp8QuantRule(), CutlassScaledMmFp8BlockRule(), SameDtypeCastRule()]


def vllm_ruleset():
    """`DEFAULT_RULES` + the bindings above (the default rules already bind `_C.rotary_embedding`, `_C.silu_and_mul`, `aten.embedding`,
    `aten.argmax`, views/allocs). The default `FusedAddRmsNorm[vllm]` is REPLACED by `FusedAddRmsNorm[vllm-bi]`, which derives the
    kernel dispatch from the export instead of assuming it."""
    from verity_vllm.frontend.rules import DEFAULT_RULES
    from verity_vllm.frontend.rules.base import RuleSet
    base = [r for r in DEFAULT_RULES.rules if type(r) is not _FusedAddRmsNormRule]
    from verity_vllm.frontend.rules.vllm_moe import VLLM_MOE_RULES  # [R13 MOE-01] the fused-MoE block's leaf kernels
    from verity_vllm.frontend.rules.vllm_sampling import VLLM_SAMPLING_RULES  # [R15 stoch] the seeded temperature + top-p token select (GumbelTopPTokenSelect_v1{V,S})
    return RuleSet("default+vllm-bindings", base + VLLM_BINDING_RULES + VLLM_MOE_RULES + list(VLLM_SAMPLING_RULES), list(DEFAULT_RULES.families))


__all__ = ["VLLM_BINDING_RULES", "vllm_ruleset", "GEMM_PINS", "RMS_PINS", "ATTN_PINS", "FUSED_NORM_DISPATCH_PIN", "lookup_key", "gemm_pins_for",
           "tuned_gemm_table_sha256", "observe_target_profile", "active_target_profile",
           "observe_attention_impls", "observe_export_facts", "clear_observations", "attn_lookup_key", "attn_kind_for", "declared_derived_accounting", "FusedAddRmsNormBIRule",
           "ScalarTensorMulRule", "GeluTanhAndMulRule", "Bf16ScalarChainRule", "TritonRmsNormRule", "TritonGemmRule", "KvCacheUpdateRule",
           "UnifiedAttentionWithOutputRule", "AllocViewRule", "IntIdWidenRule", "CachePadRule",
           "MEAN_PINS", "widened_bf16", "WidenBf16ToF32Rule", "NarrowF32ToBf16Rule", "SquareRule", "MeanDimBIRule", "RsqrtRule", "AddScalarRule",
           "AddWidenedRule", "MulF32Rule",
"PerTokenGroupFp8QuantRule", "CutlassScaledMmFp8BlockRule", "SameDtypeCastRule", "FP8_QUANT_PINS"]
