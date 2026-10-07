"""Target-keyed registered numerical bindings — the Definition variant a GEMM / attention launch binds to on the declared target.

[R13 lane hopper, on lane ampere's mechanism.]  Two facts of the `TargetProfile` decide a launch's Definition, and the two registries
split along them:

* the GEMM *launch geometry* (tiling table / default tiling, the k16 step the Triton kernel lowers to) is `verity_vllm.program.frontend.gemm_targets`
  (lane ampere): `TargetProfile.gemm_target()` -> `GemmTarget{key, vllm_arch_family, tuned_table, dot, kind, evidence}`; `ada` / `ampere` /
  `hopper` / `blackwell_consumer` are registered there, anything else is `UnregisteredGemmTarget` (fail closed).  This module holds NO table copy: it resolves the
  record's `dot` NAME to the registered step primitive (`DOTS`) and binds the Definition;
* the *attention chain* is keyed by the effective FA version (`TargetProfile.effective_flash_attn_version()`: the declared knob, else vLLM's
  selector default for the capability — 3 on cc 9.x, 2 below) and by the target's step:
    FA2 -> `BN = fa2_kblock_n(D)` (128 at D <= 64, else 64; constant per head dim), `INV = Fa2InvSum_v1` (`sum == 0 ? 1 : 1/sum`);
    FA3 -> `BN = fa3_kblock_n(D, seqlen_q, NH // KVH)` (vllm-flash-attn 506341a1 `hopper/tile_size.h::tile_size_fwd_sm90` with
           `heuristics.h::use_one_mma_wg`: at D <= 64 the decode launch (`seqlen_q * (NH/KVH) <= 64`) runs the 64 x 192 tile, prefill
           192 x 128; at D = 128 both run kBlockN 128), `INV = Fa3InvSum_v1` (`sum == 0 || sum != sum ? 0 : 1/sum`).
  `TARGETS[family]["flash_attn_versions"]` says which versions have a registered chain on a step family; `check_flash_attn_registered` refuses
  the rest with the reason.

The Definition ids: on an Ampere-family step under FA2 the kinds are the accepted records' (`Gemm_v1{K,N}`, `Attention_v3{T,NH,KVH,D,BN}`:
their Program digests do not move).  Every other combination binds the statics-lifted chains `Gemm_v2{K,N,DOT}` /
`Attention_v2{T,NH,KVH,D,BN,DOT,INV}` (`b1.GemmV2`, `b1.AttentionV2`: the same bodies with the step and the epilogue as declared
statics).  Three consumers read this module for "which registered function does
this launch compute on the declared target": the Bet A builder (`verity_vllm.program.frontend.rules.vllm_bindings`), the eager Match resolver
(`verity_vllm.observe.fold.patterns`) and, through the run header's recorded target, the collector.  The per-target correspondence (the row's Match
replay on the captured operands; lane ampere's `gemm_target_correspondence` driver) is checked bit for bit, never a tolerance.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32, BlackwellNvf4OmmaDot64, HopperBF16WgmmaDot16, HopperE4m3QgmmaDot32
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import fa2_check_inf as F2
from verity_vllm.program.registry import attention_vocab
from verity_vllm.program.registry import value_inputs as VI
from verity_vllm.program.registry import fa3_check_inf as F3
from verity_vllm.program.registry import fp8, nvfp4
from verity_vllm.program.registry import fp8_vocab
from verity_vllm.program.registry import gemm_bias as GB
from verity_vllm.program.registry import prims as P
from verity_vllm.program.frontend.gemm_targets import UnregisteredGemmTarget
from verity_vllm.program.registry.hopper import Fa3InvSum


class UnregisteredTarget(ValueError):
    """The declared target has no registered numerical binding for the requested launch family."""


#: the registered k16 step primitives by the name a `GemmTarget.dot` record carries.
DOTS: dict[str, Any] = {"AmpereBF16TcDot16": P.AmpereBF16TcDot16, "HopperBF16WgmmaDot16": HopperBF16WgmmaDot16}
#: the registered e4m3 k32 step primitives by the name a `GemmTarget.fp8_dot` record carries.
FP8_DOTS: dict[str, Any] = {"HopperE4m3QgmmaDot32": HopperE4m3QgmmaDot32, "BlackwellE4m3QmmaDot32": BlackwellE4m3QmmaDot32}
#: the registered NVFP4 k64 block-scaled step primitives by the name a `GemmTarget.fp4_dot` record carries.
FP4_DOTS: dict[str, Any] = {"BlackwellNvf4OmmaDot64": BlackwellNvf4OmmaDot64}

#: attention-chain registration per tensor-core step family (`TargetProfile.tensor_core_family`).
TARGETS: dict[str, dict[str, Any]] = {
    "ampere": {"dot_id": "AmpereBF16TcDot16_v2",
               "instruction": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (Triton tl.dot on sm_80/sm_89; FA2 CUTLASS)",
               "pipeline": "veritor.core.silicon_total.AMPERE_BF16_DOT16_TOTAL (groups 8+8, width 25, floor -132; A100 + RTX 4090 conform)",
               "flash_attn_versions": (2,),
               "evidence": {"attention_fa2": "W11 (D=128, BN=64), E7 D54/D57 (D=64, BN=128); R12 canaries; R13 ampere A100 row Match replay"}},
    "hopper": {"dot_id": "HopperBF16WgmmaDot16_v1",
               "instruction": "wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 (Triton tl.dot on sm_90) / mma.sync.m16n8k16 (FA2 CUTLASS JIT to sm_90)",
               "pipeline": "veritor.core.silicon_total.HOPPER_BF16_WGMMA_K16 (one group of 16, width 26, floor -133; lane HP 2026-09-07) = "
                           "silicon.HOPPER_BF16_M16N8K16 (Hawkeye H100)",
               "flash_attn_versions": (2, 3),
               # a biased linear binds GemmBias_v2{K,N,DOT} with this family's step (`gemm_bias_dot`): serving has no source for the
               # pre-bias word of Gemm_v2 + BiasAdd_v1
               "gemm_bias_dot": True,
               "evidence": {"attention_fa2": "R13 hopper: forced-FA2 row `…__h100__…__bi-eager-fa2` Match replay (D=64, BN=128, DOT=Hopper, INV=Fa2InvSum)",
                            "attention_fa3": "lane HP CONF-b1-hopper (D=128, BN=128, 39 leaves exact, positions 0..526); R13 hopper: D=64 (BN 128 prefill / "
                                             "192 decode) — the native FA3 row's Match replay"}},
    "blackwell_consumer": {"dot_id": "HopperBF16WgmmaDot16_v1",
                           "instruction": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (cc 12.0 has no wgmma; under VLLM_BATCH_INVARIANT=1 "
                                          "vLLM's linear layers run the Triton matmul_kernel_persistent, then `output + bias` in bf16, at every M; "
                                          "other bf16 matmuls stay on cuBLASLt with split-K disabled, the same chain without a bias)",
                           "pipeline": "verity.ml.circuits.silicon.models.HOPPER_BF16_M16N8K16 (one group of 16, width 26, floor -133; the cc 12.0 mma.sync entry of "
                                       "verity.ml.circuits.silicon.instructions) = HOPPER_BF16_WGMMA_K16's parameters",
                           "flash_attn_versions": (2,),
                           "fa2_construction": "check-inf-per-iteration",
                           # the fused-MoE expert GEMMs bind MoeExpertGemm_v2 / MoeExpertGemmW_v2{DOT} with this family's step (`moe_expert_dot`)
                           "moe_expert_dot": True,
                           # a biased linear binds GemmBias_v2{K,N,DOT} with this family's step (`gemm_bias_dot`)
                           "gemm_bias_dot": True,
                           "evidence": {"attention_fa2": "the port's attention lane, RTX PRO 6000 (cc 12.0, 188 SMs, driver 595.91): property "
                                                         "fa2_target_capture ef4b2584 (r20260930-033258-f611, art:d342a748): vLLM selects FLASH_ATTN "
                                                         "FA2; _vllm_fa2_C carries cc 8.0 SASS + compute_80 PTX, so the kernel is the driver's JIT; "
                                                         "kBlockM 64, kBlockN fa2_kblock_n(D) at D 64/96/128/256; MUFU ex2/rcp = core's tables on all "
                                                         "2^32 inputs; 19,232 rows (122,228 heads) equal Attention_v2 with this DOT and Fa2InvSum on "
                                                         "every finite head (Attention_v3: 997 differ); fa_tap_exactness 9dee6b6f / 9403e9fe guarded "
                                                         "(r20260930-030755-9aab, art:592bc0ae)",
                                        "gemm_bias": "RTX PRO 6000 (driver 580.173.02), torch 2.13.0+cu129, vLLM d9105ea80 under "
                                                     "VLLM_BATCH_INVARIANT=1: linear_batch_invariant(x, W, b) on a biased qkv linear of the bf16 "
                                                     "sweep (K 896, N 1152) launches matmul_kernel_persistent, then a bf16 add, at M 1, 2, 7 and 256; all "
                                                     "306,432 words equal BiasAdd_v1(Gemm_v2{DOT=HopperBF16WgmmaDot16_v1}), and 80,572 of them "
                                                     "differ from the bias added before the rounding (F.linear's cuBLASLt epilogue), run "
                                                     "r20260930-140739-7974"}},
}


def target_family(profile) -> str:
    fam = profile.tensor_core_family
    if fam not in TARGETS:
        raise UnregisteredTarget(f"compute capability {profile.capability_str}: no registered tensor-core pipeline ({fam})")
    return fam


def gemm_target(profile, arch_family=...):
    """The registered GEMM specialization of the target (`gemm_targets.GemmTarget`), re-raised as `UnregisteredTarget` when there is none."""
    try:
        return profile.gemm_target(arch_family)
    except UnregisteredGemmTarget as e:
        raise UnregisteredTarget(str(e)) from e


def tensor_core_dot(profile):
    """The registered k16 step primitive of the declared target (the `GemmTarget.dot` record's primitive)."""
    name = gemm_target(profile).dot
    if name not in DOTS:
        raise UnregisteredTarget(f"GEMM target {profile.capability_str}: step {name!r} has no registered primitive")
    return DOTS[name]


def moe_expert_dot(profile):
    """The k16 step of the fused-MoE expert GEMMs' Definition on the declared target: None for the `_v1` Definitions, else the
    target's registered step for the `_v2{E,K,N,DOT}` ones.  A family binds them when its record says `moe_expert_dot` (its
    fused_moe_kernel correspondence ran on a step other than the Ampere/Ada one); every other family keeps the `_v1` records' binding
    (cc 9.0 too: its fused_moe_kernel lowering has no correspondence run yet).  An unregistered target raises `UnregisteredTarget`."""
    return tensor_core_dot(profile) if TARGETS[target_family(profile)].get("moe_expert_dot") else None


def gemm_bias_dot(profile):
    """The k16 step of a biased linear's one-Call Definition `GemmBias_v2{K,N,DOT}` on the declared target, or None: the accepted kinds
    bind `GemmBias_v1`, and a family whose record does not say `gemm_bias_dot` keeps `Gemm_v2` + `BiasAdd_v1`.  An unregistered target
    raises `UnregisteredTarget`."""
    if uses_accepted_kinds(profile):
        return None
    return tensor_core_dot(profile) if TARGETS[target_family(profile)].get("gemm_bias_dot") else None


def gemm_bias_spec(profile, K: int, N: int):
    """The registered one-Call Definition of a batch-invariant linear with a bias (the launch, then `output + bias` in bf16) on the
    declared target, or None when the target keeps the pair as two Calls."""
    if uses_accepted_kinds(profile):
        return bind(GB.GemmBias, K=int(K), N=int(N))
    dot = gemm_bias_dot(profile)
    return None if dot is None else bind(GB.GemmBiasV2, K=int(K), N=int(N), DOT=dot)


def moe_expert_gemm_spec(profile, E: int, K: int, N: int, *, routed: bool):
    """The registered Definition of one (token, slot) pair of a `fused_moe_kernel` launch on the declared target: the w13 launch
    (`routed` False) or the w2 launch with the routed weight."""
    from verity_vllm.program.registry import moe as M
    dot = moe_expert_dot(profile)
    if dot is None:
        return bind(M.MoeExpertGemmW if routed else M.MoeExpertGemm, E=int(E), K=int(K), N=int(N))
    return bind(M.MoeExpertGemmWV2 if routed else M.MoeExpertGemmV2, E=int(E), K=int(K), N=int(N), DOT=dot)


def moe_expert_fp8_dot(profile):
    """The e4m3 k32 step of the block-FP8 fused-MoE expert GEMMs (`TritonExperts`' `fused_moe_kernel`) on the declared target: the GEMM
    target's `fp8_moe_dot`, its own measurement rather than the per-tensor FP8 linear's (`fp8_dot`) or the blockwise one's
    (`fp8_block_dot`); refused by name where the target records none (cc 9.0: GLM-5.3's experts bind `_v1` through their capture map)."""
    gt = gemm_target(profile)
    dot = FP8_DOTS.get(gt.fp8_moe_dot) if gt.fp8_moe_dot else None
    if dot is None:
        raise UnregisteredTarget(f"block-FP8 fused-MoE experts on {gt.key} ({profile.capability_str}): no registered e4m3 step "
                                 f"(GemmTarget.fp8_moe_dot)")
    return dot


def moe_expert_fp8_gemm_spec(profile, E: int, K: int, N: int, G: int, KB: int, SWAP: bool, *, routed: bool):
    """The registered Definition of one (token, slot) pair of a block-FP8 `fused_moe_kernel` launch on the declared target:
    `MoeExpertFp8Gemm(W)_v1{E,K,N,G,KB,SWAP,DOT}`, the w13 launch (`routed` False) or the w2 launch with the routed weight."""
    from verity_vllm.program.registry import fp8_moe_experts as X
    return bind(X.MoeExpertFp8GemmW if routed else X.MoeExpertFp8Gemm, E=int(E), K=int(K), N=int(N), G=int(G), KB=int(KB),
                SWAP=bool(SWAP), DOT=moe_expert_fp8_dot(profile))


def tuned_gemm_table(profile) -> dict[tuple[int, int], dict[str, Any]]:
    """vLLM's tuned batch-invariant GEMM table of the declared target (`GemmTarget.tuned_table`; empty = default tiling for every shape)."""
    return dict(gemm_target(profile).tuned_table)


def check_flash_attn_registered(profile) -> int:
    """The effective FA version of the target, refused when the target has no registered attention chain for it."""
    fam = target_family(profile)
    v = profile.effective_flash_attn_version()
    if v not in TARGETS[fam]["flash_attn_versions"]:
        raise UnregisteredTarget(f"flash_attn_version {v} on {fam} ({profile.capability_str}): no registered attention chain")
    return v


def fa3_kblock_n(D: int, seqlen_q: int, qh_per_kvh: int) -> int:
    """kBlockN of the FA3 forward vLLM launches (vllm-flash-attn 506341a1 `hopper/tile_size.h::tile_size_fwd_sm90`, element_size 2,
    causal, not local, `paged_kv_non_TMA` = paged KV without TMA, `use_one_mma_wg = (D in {64, 128}) and seqlen_q * qh_per_kvh <= 64`
    (`hopper/heuristics.h`; `seqlen_q` = the launch's max_seqlen_q under varlen)):  D <= 64: one-MMA-warpgroup 64 x 192 else 192 x 128;
    D <= 96: 128 (paged non-TMA); D <= 128: 128 (causal).  Larger head dims are not registered."""
    D, seqlen_q, qh = int(D), int(seqlen_q), int(qh_per_kvh)
    one_wg = D in (64, 128) and seqlen_q * qh <= 64
    if D <= 64:
        return 192 if one_wg else 128
    if D <= 96:
        return 128
    if D <= 128:
        return 128
    raise UnregisteredTarget(f"FA3 head dim {D}: no registered kBlockN rule")


def fa3_tile_m(D: int, seqlen_q: int, qh_per_kvh: int) -> int:
    """kBlockM of the FA3 forward tile (the same `tile_size_fwd_sm90` / `use_one_mma_wg` rule as `fa3_kblock_n`): D <= 64: 64 (one MMA
    warpgroup) else 192; D <= 96: 192; D <= 128: 64 (one MMA warpgroup) else 128."""
    D, seqlen_q, qh = int(D), int(seqlen_q), int(qh_per_kvh)
    one_wg = D in (64, 128) and seqlen_q * qh <= 64
    if D <= 64:
        return 64 if one_wg else 192
    if D <= 96:
        return 192
    if D <= 128:
        return 64 if one_wg else 128
    raise UnregisteredTarget(f"FA3 head dim {D}: no registered kBlockM rule")


def fa3_check_inf(profile) -> bool:
    """True when the target builds FA3's attention rows with the per-iteration `Check_inf` chain (`Attention_v4`):
    an effective FA3 target whose effective `fa3_construction` is `"check-inf-per-iteration"` (the default)."""
    construction = getattr(profile, "effective_fa3_construction", None)
    return profile.effective_flash_attn_version() == 3 and construction is not None and construction() == "check-inf-per-iteration"


def fa3_masked_from(profile, T: int, NH: int, KVH: int, D: int, seqlen_q: int, row: int) -> int:
    """`MASKED_FROM` of an `Attention_v4` row: the FA3 tile's `n_block_min_causal_local_mask` for query row `row` of its sequence (its
    index among the sequence's query rows in the launch; `seqlen_q` = the launch's max_seqlen_q, which picks the tile)."""
    g = max(1, int(NH) // max(1, int(KVH)))
    return attention_vocab.fa3_masked_from(T, fa3_kblock_n(D, seqlen_q, g), row, fa3_tile_m(D, seqlen_q, g), g)


def fa2_check_inf(profile) -> bool:
    """True when the target builds FA2's attention rows with FA2's per-iteration `Check_inf` chain (`Attention_v5`): an effective FA2
    target whose family registers `fa2_construction = "check-inf-per-iteration"`.  The other FA2 families keep the chain their records
    were accepted under (every block guarded)."""
    return (profile.effective_flash_attn_version() == 2
            and TARGETS.get(profile.tensor_core_family, {}).get("fa2_construction") == "check-inf-per-iteration")


def fa2_masked_from(profile, T: int, D: int, seqlen_q: int, row: int, own_rows: int) -> int:
    """`MASKED_FROM` of an `Attention_v5` row: the lowest key block FA2's kernel guards for query row `row` of its sequence (its index among
    the sequence's `own_rows` query rows in the launch; the sequence's key count is the last row's, `T - 1 - row + own_rows`).  The launch's
    max_seqlen_q (`seqlen_q`) decides causality: `varlen_fwd` runs a launch of one query row per sequence non-causal."""
    return attention_vocab.fa2_masked_from(int(T) - 1 - int(row) + int(own_rows), own_rows, row, b1.fa2_kblock_n(D), int(seqlen_q) > 1)


class LaunchContextConflict(ValueError):
    """[R15 hopper HOP-03] the launch context of a request shape is not one value (requests of the same (LP, T) are co-scheduled with
    different batches) and the target's tile rule reads it: the component Program cannot be built for that shape as one Program."""


def parse_launch_context(spec: str | int | list | dict | None) -> dict[str, Any] | None:
    """[R15 hopper HOP-03] The LAUNCH CONTEXT a request Program is built under -- a structural input of the construction, like `sampling`.

    FA3's tile (`fa3_kblock_n`) reads the varlen launch's `max_seqlen_q` = the most query rows any ONE sequence of the launch has; the
    request wrapper sees only its own rows (its prompt on step 0, one row per decode step), which IS the launch's value for a batch of one
    (B1: unchanged) and is NOT when the request is co-scheduled with a longer prompt (measured M-0225: an 8-token prompt in a B8 prefill
    launch with a 1024-token prompt runs the 192 x 128 tile, BN 128, while its standalone Program said 64 x 192, BN 192).  `spec`:
        None                 -> None (no declared context: every step uses its own rows -- the B1 semantics, digests unchanged)
        int N                -> {0: N}                    the prefill step's launch has a sequence of N query rows (co-arrival, no later arrivals)
        list [n0, n1, ...]   -> {s: n_s}                  per request step s: the launch's max_seqlen_q at that engine step (1 = decode-only step)
        dict {"s": n}        -> same, string or int keys
        "conflict"           -> {"conflict": True}        requests of this shape have DIFFERENT contexts: refused where the tile rule reads it
    Returns {"max_seqlen_q_by_step": {int: int}, "conflict": bool, "declared": True}."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, str):
        s = spec.strip()
        if s.lower() == "conflict":
            return {"max_seqlen_q_by_step": {}, "conflict": True, "declared": True}
        if s.lstrip("-").isdigit():
            spec = int(s)
        else:
            import json
            spec = json.loads(s)
    if isinstance(spec, bool):
        raise ValueError(f"launch context: {spec!r}")
    if isinstance(spec, int):
        if spec < 1:
            raise ValueError(f"launch context: max_seqlen_q {spec} < 1")
        by = {0: int(spec)}
    elif isinstance(spec, list):
        by = {i: int(v) for i, v in enumerate(spec)}
    elif isinstance(spec, dict):
        if spec.get("conflict"):
            return {"max_seqlen_q_by_step": {}, "conflict": True, "declared": True}
        src = spec.get("max_seqlen_q_by_step", spec)
        by = {int(k): int(v) for k, v in src.items()}
    else:
        raise ValueError(f"launch context: unsupported spec {type(spec).__name__}")
    if any(v < 1 for v in by.values()) or any(k < 0 for k in by):
        raise ValueError(f"launch context: steps and max_seqlen_q must be >= 0 / >= 1: {by}")
    return {"max_seqlen_q_by_step": by, "conflict": False, "declared": True}


class LaunchContextInvalid(ValueError):
    """[R15 hopper HOP-03b, rev-pm M-0326 T1/T2] a DECLARED launch context that cannot be the context of this request: it does not name a
    step the request runs (a truncated declaration would silently fall back to own rows = the M-0225 bug through the CLI), or it names a
    max_seqlen_q below the request's own rows at that step (the launch contains the request; a smaller value is another shape's context).
    Refused on every target: it is a wrong structural input, not a tile question."""


def launch_seqlen_q(own_rows: int, step: int, launch: dict[str, Any] | None) -> tuple[int, str]:
    """[R15 hopper HOP-03] The `seqlen_q` the attention tile rule must read for a request's step: the launch's max over all sequences,
    never less than this request's own rows (the launch contains the request).  Returns (seqlen_q, source):
    ("own rows" when no context is declared; "declared launch context" when one is).  A declared context that does not name the step,
    or names a value below the request's own rows, is refused (`LaunchContextInvalid`) -- never completed or corrected (rev-pm M-0326 T1/T2:
    row_pod's `launch_context` always declares every step of the shape, so both are CLI-only misdeclarations)."""
    own = int(own_rows)
    if not launch:
        return own, "own rows"
    if launch.get("conflict"):
        raise LaunchContextConflict("the launch context of this request shape is not one value ('conflict'): requests of the same (LP, T) are "
                                    "co-scheduled with different batches; build one Program per (shape, arrival) or declare the contexts")
    by = launch.get("max_seqlen_q_by_step") or {}
    v = by.get(int(step))
    if v is None:
        raise LaunchContextInvalid(f"the declared launch context names steps {sorted(by)} but the request runs step {int(step)}: declare every "
                                   f"step of the shape (row_pod: verity_vllm.pipeline.launch_context) -- a step without a context would silently "
                                   f"take its own rows ({own}) for a launch that may hold a longer prefill")
    if int(v) < own:
        raise LaunchContextInvalid(f"the declared launch context says max_seqlen_q {int(v)} at step {int(step)} but this request alone has {own} "
                                   f"query rows there: the launch contains the request, so the declaration is another shape's context")
    return int(v), "declared launch context"


def attention_geometry(profile, D: int, seqlen_q: int, NH: int, KVH: int) -> dict[str, Any]:
    """`{"BN", "INV", "DOT", "fa_version"}` of the attention launch on the declared target for head dim D: the online-softmax block
    (keys per block), the epilogue reciprocal and the k16 step.  `seqlen_q` = the launch's query rows (per request; under varlen the
    batch max), consulted by FA3's tile rule only."""
    v = check_flash_attn_registered(profile)
    dot = tensor_core_dot(profile)
    if v == 2:
        return {"BN": b1.fa2_kblock_n(D), "INV": P.Fa2InvSum, "DOT": dot, "fa_version": 2}
    return {"BN": fa3_kblock_n(D, seqlen_q, max(1, int(NH) // max(1, int(KVH)))), "INV": Fa3InvSum, "DOT": dot, "fa_version": 3}


def uses_accepted_kinds(profile) -> bool:
    """True when the target binds the accepted records' kinds (`Gemm_v1`, `Attention_v3`): Ampere-family step under FA2."""
    return target_family(profile) == "ampere" and profile.effective_flash_attn_version() == 2


def gemm_spec(profile, K: int, N: int):
    """The registered GEMM Definition of a bf16 `[*,K] x W[N,K]^T` batch-invariant launch on the declared target."""
    if uses_accepted_kinds(profile):
        return bind(b1.Gemm, K=int(K), N=int(N))
    return bind(b1.GemmV2, K=int(K), N=int(N), DOT=tensor_core_dot(profile))


def scaled_mm_fp8_spec(profile, K: int, N: int, bias: bool):
    """The registered Definition of the per-tensor FP8 linear (`_C.cutlass_scaled_mm` over the quantised row) on the declared
    target: `ScaledMmFp8NoSwap_v1{K,N,BIAS,DOT}` where the GEMM target records the unswapped epilogue and its e4m3 step
    (`GemmTarget.fp8_scale_order == "no_swap"`, `fp8_dot`); refused by name elsewhere."""
    gt = gemm_target(profile)
    dot = FP8_DOTS.get(gt.fp8_dot) if gt.fp8_dot else None
    if gt.fp8_scale_order == "no_swap" and dot is not None:
        return bind(fp8.ScaledMmFp8NoSwap, K=int(K), N=int(N), BIAS=bool(bias), DOT=dot)
    raise UnregisteredTarget(f"per-tensor FP8 linear on {gt.key} ({profile.capability_str}): no registered binding")


def fp8_block_dot(profile):
    """The e4m3 k32 step the blockwise FP8 linear (`CutlassFp8BlockScaledMMKernel`) binds on the declared target, as the version 3
    Definitions' `DOT`; None on a target whose step is Hopper's (`ScaledMmFp8Block_v1` and its shared-scale form are that step's chain,
    and the recorded Programs bind them).  Refused by name where the GEMM target records no blockwise step (`GemmTarget.fp8_block_dot`):
    `_C.cutlass_scaled_mm` reaches another blockwise kernel there under the same op and kernel class."""
    gt = gemm_target(profile)
    dot = FP8_DOTS.get(gt.fp8_block_dot) if gt.fp8_block_dot else None
    if dot is None:
        raise UnregisteredTarget(f"blockwise FP8 linear on {gt.key} ({profile.capability_str}): this target's _C.cutlass_scaled_mm "
                                 f"blockwise kernel has no registered Definition")
    return None if dot is HopperE4m3QgmmaDot32 else dot


def scaled_mm_fp8_block_spec(profile, K: int, N: int, G: int, scale: str | None = None):
    """The registered Definition of the blockwise FP8 linear on one token row on the declared target, under the scale construction
    `scale` (default the target's `effective_fp8_scale_construction()`): `ScaledMmFp8BlockSharedScale_v1{K,N,G}` / `ScaledMmFp8Block_v1`
    on hopper, the version 3 forms `{K,N,G,DOT}` over the target's `fp8_block_dot` elsewhere; refused by name where there is none."""
    return fp8_vocab.block_gemm(K, N, G, scale or profile.effective_fp8_scale_construction(), fp8_block_dot(profile))


def _fp4_dot(profile):
    gt = gemm_target(profile)
    dot = FP4_DOTS.get(gt.fp4_dot) if gt.fp4_dot else None
    if dot is None:
        raise UnregisteredTarget(f"NVFP4 linear on {gt.key} ({profile.capability_str}): no registered binding")
    return dot


def nvfp4_quant_spec(profile, K: int):
    """The registered Definition of the NVFP4 activation quantizer (`cvt_fp16_to_fp4`) on one row: `Nvfp4ActQuant_v1{K}` where the
    GEMM target records the NVFP4 linear (`GemmTarget.fp4_dot`); refused by name elsewhere and for K not a multiple of 16."""
    _fp4_dot(profile)
    if int(K) % nvfp4.BLOCK:
        raise UnregisteredTarget(f"NVFP4 linear: K={K} is not a multiple of {nvfp4.BLOCK}")
    return bind(nvfp4.Nvfp4ActQuant, K=int(K))


def nvfp4_gemm_spec(profile, K: int, N: int):
    """The registered Definition of the NVFP4 GEMM (`cutlass_scaled_fp4_mm`) on one row: `ScaledMmNvfp4_v1{K,N,DOT}` with the
    target's block-scaled step; refused by name elsewhere and for K not a multiple of 64 (the step)."""
    dot = _fp4_dot(profile)
    if int(K) % nvfp4.STEP_K:
        raise UnregisteredTarget(f"NVFP4 linear: K={K} is not a multiple of {nvfp4.STEP_K}")
    return bind(nvfp4.ScaledMmNvfp4, K=int(K), N=int(N), DOT=dot)


def rmsnorm_fused_spec(profile, N: int, EPS: float | None = None, eps_input: bool = False):
    """The registered Definition of `_C.fused_add_rms_norm` on one row on the declared target: under `batch_invariant` the generic
    kernel (`RMSNormFusedCuda_v2{N,EPS}`; with `eps_input` its twin `RMSNormFusedCuda_v4{N}` that reads eps as an input); refused by
    name otherwise, where vLLM launches the width-8 kernel, whose block size off batch invariance has no registered Definition."""
    if not profile.batch_invariant:
        raise UnregisteredTarget("fused_add_rms_norm without batch invariance launches the width-8 fused_add_rms_norm_kernel<bf16, 8> "
                                 "(layernorm_kernels.cu, `!batch_invariant_launch`), whose block size off batch invariance is not "
                                 "pinned: RMSNormFusedCuda_v1 models that kernel at the batch-invariant block size, and v2 is the "
                                 "generic kernel launched only under batch invariance")
    if eps_input:
        return bind(VI.RMSNormFusedCudaV4, N=N)
    return bind(b1.RMSNormFusedCuda, N=N, EPS=EPS)


_SCALE_INPUT = {"Attention_v3": VI.AttentionV10, "Attention_v2": VI.AttentionV12, "Attention_v4": VI.AttentionV13,
                "Attention_v5": F2.AttentionV11, "AttentionSoftcap_v1": VI.AttentionSoftcapV4, "AttentionSoftcap_v2": VI.AttentionSoftcapV5}


def attention_spec(profile, T: int, NH: int, KVH: int, D: int, seqlen_q: int, row: int | None = None, own_rows: int | None = None,
                   cap: float | None = None, scale_input: bool = False):
    """The registered attention Definition of one query row with T visible keys on the declared target.  Under a per-iteration `Check_inf`
    construction the row's index among its sequence's query rows in the launch (`row`) selects its tile (FA3, `fa3_check_inf`); FA2
    (`fa2_check_inf`) also reads how many query rows the sequence has there (`own_rows`).  A logit soft cap `cap` selects the softcapped
    chain: `AttentionSoftcap_v1` on the accepted target, `AttentionSoftcap_v2` under FA2's per-iteration `Check_inf`, nothing elsewhere.
    `scale_input`: the constant chain's twin that reads its values as inputs (`_SCALE_INPUT`: the scale `scale_log2`; a softcapped
    chain's `softcap_mul` and `scale_log2`, with no `CAP`)."""
    if scale_input:
        spec = attention_spec(profile, T, NH, KVH, D, seqlen_q, row, own_rows, cap)
        return bind(_SCALE_INPUT[spec.base_id], **{k: v for k, v in spec.bindings.items() if k != "CAP"})
    if cap is not None:
        return softcapped(attention_spec(profile, T, NH, KVH, D, seqlen_q, row, own_rows), cap)
    g = attention_geometry(profile, D, seqlen_q, NH, KVH)
    if uses_accepted_kinds(profile):
        return bind(b1.AttentionV3, T=int(T), NH=int(NH), KVH=int(KVH), D=int(D), BN=g["BN"])
    if fa3_check_inf(profile):
        if row is None:
            raise UnregisteredTarget("fa3_construction check-inf-per-iteration: the row's index among its sequence's query rows is not known")
        return bind(F3.AttentionV4, T=int(T), NH=int(NH), KVH=int(KVH), D=int(D), BN=g["BN"], DOT=g["DOT"], INV=g["INV"],
                    MASKED_FROM=fa3_masked_from(profile, T, NH, KVH, D, seqlen_q, row))
    if fa2_check_inf(profile):
        if row is None or own_rows is None:
            raise UnregisteredTarget("fa2_construction check-inf-per-iteration: the row's index among its sequence's query rows, or their "
                                     "number, is not known")
        return bind(F2.AttentionV5, T=int(T), NH=int(NH), KVH=int(KVH), D=int(D), BN=g["BN"], DOT=g["DOT"], INV=g["INV"],
                    MASKED_FROM=fa2_masked_from(profile, T, D, seqlen_q, row, own_rows))
    return bind(b1.AttentionV2, T=int(T), NH=int(NH), KVH=int(KVH), D=int(D), BN=g["BN"], DOT=g["DOT"], INV=g["INV"])


def softcapped(spec, cap: float):
    """The softcapped chain of a plain attention row's Definition `spec`, with the same statics and `CAP = cap`: `AttentionSoftcap_v1`
    for `Attention_v3` (the accepted target), `AttentionSoftcap_v2` for `Attention_v5` (FA2's per-iteration `Check_inf`).  No other
    chain has a registered softcapped form (UnregisteredTarget)."""
    from verity_vllm.program.registry import dense as DN, fa2_softcap as F2S  # noqa: PLC0415
    body = {"Attention_v3": DN.AttentionSoftcap, "Attention_v5": F2S.AttentionSoftcapV2}.get(spec.base_id)
    if body is None:
        raise UnregisteredTarget(f"{spec.base_id} has no registered softcapped chain: a softcapped row is registered on the accepted target "
                                 f"and under FA2's per-iteration Check_inf")
    return bind(body, **spec.bindings, CAP=float(cap))


def describe(profile) -> dict[str, Any]:
    """The target's registered bindings as a record (applicability manifests, run headers, reports)."""
    fam = target_family(profile)
    t = TARGETS[fam]
    gt = gemm_target(profile)
    v = profile.effective_flash_attn_version()
    return {"tensor_core_family": fam, "dot": t["dot_id"], "instruction": t["instruction"], "pipeline": t["pipeline"],
            "flash_attn_version_effective": v, "flash_attn_registered": v in t["flash_attn_versions"],
            "gemm_target": gt.key, "tuned_gemm_arch_family": gt.vllm_arch_family, "tuned_gemm_keys": sorted(f"{n},{k}" for n, k in gt.tuned_table),
            "gemm_kind": "Gemm_v1{K,N}" if uses_accepted_kinds(profile) else f"Gemm_v2{{K,N,DOT={t['dot_id']}}}",
            "attention_kind": "Attention_v3{T,NH,KVH,D,BN=fa2_kblock_n(D)}" if uses_accepted_kinds(profile)
            else f"Attention_v4{{T,NH,KVH,D,BN=fa3_kblock_n(D,seqlen_q,NH/KVH),DOT={t['dot_id']},INV=Fa3InvSum_v1,MASKED_FROM}}"
            if v == 3 and fa3_check_inf(profile)
            else f"Attention_v5{{T,NH,KVH,D,BN=fa2_kblock_n(D),DOT={t['dot_id']},INV=Fa2InvSum_v1,MASKED_FROM}}" if fa2_check_inf(profile)
            else f"Attention_v2{{T,NH,KVH,D,BN={'fa2_kblock_n(D)' if v == 2 else 'fa3_kblock_n(D,seqlen_q,NH/KVH)'},DOT={t['dot_id']},"
                 f"INV={'Fa2InvSum_v1' if v == 2 else 'Fa3InvSum_v1'}}}",
            "evidence": {"gemm": list(gt.evidence), **t["evidence"]}}


__all__ = ["TARGETS", "DOTS", "FP8_DOTS", "FP4_DOTS", "UnregisteredTarget", "target_family", "gemm_target", "tensor_core_dot",
           "moe_expert_dot", "moe_expert_gemm_spec", "moe_expert_fp8_dot", "moe_expert_fp8_gemm_spec", "gemm_bias_dot", "gemm_bias_spec", "tuned_gemm_table",
           "check_flash_attn_registered", "fa3_kblock_n", "fa3_tile_m", "fa3_check_inf", "fa3_masked_from", "fa2_check_inf", "fa2_masked_from", "attention_geometry", "uses_accepted_kinds", "gemm_spec",
           "scaled_mm_fp8_spec", "fp8_block_dot", "scaled_mm_fp8_block_spec", "nvfp4_quant_spec", "nvfp4_gemm_spec", "rmsnorm_fused_spec", "attention_spec", "describe",
           "LaunchContextConflict", "parse_launch_context", "launch_seqlen_q"]
