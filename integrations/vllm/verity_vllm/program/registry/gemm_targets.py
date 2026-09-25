"""Registered GEMM *target specializations* of vLLM's batch-invariant `matmul_kernel_persistent` (pinned vLLM d9105ea80).

[R13 lane ampere — fix `B-AMPERE-GEMM-TILE` (F-hw-01/13, F-hw-10/11)]

One record per registered target arch family, read by every consumer through the `TargetProfile`
(`TargetProfile.gemm_target()`): Bet A's construction (`verity_vllm.program.frontend.rules.vllm_bindings.gemm_pins_for` — the launch a
GEMM node must show), the eager Match resolver (`verity_vllm.observe.fold.patterns.GemmLaunch` / `gen_llama_patterns.GemmLaunchTuned` —
the launch a captured GEMM must show), and the applicability manifest (`cb_a.derive_step.applicability_conditions` — the
target facts the artifact requires).  There is no second table anywhere: the profiles / rules hold no tiling literal of
their own any more.

What a record says
------------------
* `vllm_arch_family` — what vLLM's own `batch_invariant_configs._get_tuned_matmul_arch_family(cc)` answers for the
  target's capability.  With a family vLLM's `_get_matmul_config(M, N, K)` takes the M-bucket row of its tuned table for a
  tabled `(N, K)`; without one (Ampere: the function returns None) every shape launches the default tiling
  `128/128/64, 8 warps, 3 stages`.  The construction host reads the family back from vLLM under the target shim and refuses
  when it differs from the record (`export_compat.read_back_applied`); the capture host records it in the run header
  (`verity_vllm.engine.engine_profile.gpu_info`).
* `tuned_table` — the pinned copy of vLLM's tuned table for that family (identity = `table_sha256()`, an applicability
  condition), `{}` for a family without one.  `launch_config(M, N, K)` mirrors `_get_matmul_config` on that table.
* `dot` — the k16 tensor-core step the registered Definition binds: `Gemm_v1{K,N}` (`verity_vllm.program.registry.b1.GemmV2` with
  `DOT = AmpereBF16TcDot16`, `mma.sync.m16n8k16` bf16 -> f32) — per output coordinate one f32 accumulator advanced over the
  K/16 k16 steps in ascending k order from zero, RNE to bf16.  The tile constants and the pipeline depth decide which output
  coordinates a CTA owns and how the k tiles are staged, NOT the per-coordinate arithmetic — but that is a claim per
  target, established by `evidence`, never assumed from the kernel source being the same.
* `evidence` — the correspondence runs (`verity_vllm/program/kernels/gemm_target_correspondence.py`: the production
  launch on the model's GEMM shapes incl. M tails, bit-exact against `cpu_model.gemm_model` under the single ascending
  chain of the target's own instruction model) that admitted the record.

Registered: `ada` (cc 8.9, vLLM family "ada": the tuned sm_89 table — R10/R11 Tier A on the RTX 4090 / L40S captures),
`ampere` (cc 8.0, vLLM family None: default tiling for every shape — R13 correspondence on the A100) and `hopper` (cc 9.0, vLLM
family "hopper": the tuned sm_90 table; on sm_90 Triton lowers `tl.dot` to `wgmma`, a DIFFERENT k16 step (`HopperBF16WgmmaDot16`),
so the Definition is `Gemm_v2{K,N,DOT}` — R13 lane hopper).  Anything else (Blackwell; cc 8.6/8.7 with no run yet) is
`UnregisteredGemmTarget`: construction refuses (`numerics_unregistered`), the resolver answers Unsupported, and the
applicability check fails — a launch that happens to show a registered tiling on an unregistered target is still refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

#: vLLM's default tiling of `matmul_persistent` for bf16 (`batch_invariant.py::matmul_persistent`, `configs[torch.bfloat16]`).
DEFAULT_TILING: dict[str, int] = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3}
#: the constexprs a `Gemm_v1` launch fixes besides the tiling (no fused bias epilogue, 32-bit indexing).
FIXED_CONSTEXPRS: dict[str, Any] = {"GROUP_SIZE_M": 8, "A_LARGE": False, "B_LARGE": False, "C_LARGE": False, "HAS_BIAS": False}

#: `(BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps, num_stages)` per M-bucket upper bound, ascending; M above the last bucket takes
#: the last (the table's own rule).  VERBATIM `_BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["ada"]` of the pinned vLLM d9105ea80
#: (`batch_invariant_configs.py` sha256 72ea0d4df0ab5ae288abb7df5c203d2ad5f1709c7295d87ec72da775fddc1b7a), ALL FIVE keys.
#: Until R15 only the (2048, 2048) row (the one LLAMA32_1B exercised: M=32 -> the 32 bucket, M=1 -> the 1 bucket) was pinned; the
#: other four are the Qwen3-class dense shapes (hidden 2048, vocab 151936) and vLLM launches them TUNED on sm_89, so a Program
#: for such a model was refused at its first tabled launch (`constexpr BLOCK_SIZE_M=16 differs from the profile pin 128`:
#: Qwen3-30B-A3B lm_head at TP1, R15 lane moe GEMM-ADA-TABLE; the (2048, 6144) row has BLOCK_SIZE_K 128).  Read from vLLM on
#: cp-moe3 (L40S) 2026-09-19; `tests/program/test_gemm_targets.py` checks the copy against vLLM's table wherever vLLM imports.
#: (moved here from `verity_vllm.engine.profiles.gen_llama_patterns.ADA_TUNED_GEMM`, which re-exports it.)
#: APPLICABILITY IDENTITY (global M-0140 (3)): `REGISTERED["ada"].table_sha256()` is the `tuned_gemm_table_sha256` derive_step writes into
#: every L40S/4090 component's applicability, and GP-01 refuses a workload whose components disagree on it.  The one-key table of
#: r14-rc3 hashed to ADA_TABLE_SHA256_RC3 below (the value in every L40S artifact Built before the release that carries this file);
#: the five-key table hashes to a different value, so a pre-/post-change component pair is a NAMED Build refusal, never composed.
ADA_TABLE_SHA256_RC3 = "44775a8b9d5f734ec0c5527520e46fbf451ee634b7422a2b9e4781c0bd1ad323"
ADA_TUNED_GEMM: dict[tuple[int, int], dict[str, Any]] = {
    (2048, 2048): {"BLOCK_SIZE_K": 64, "m_buckets": ((1, (32, 128, 8, 5)), (4, (16, 256, 8, 3)), (8, (32, 128, 8, 3)), (16, (32, 128, 8, 3)),
                                                     (32, (16, 128, 8, 5)), (64, (16, 256, 4, 3)), (256, (64, 128, 8, 4)), (512, (64, 64, 4, 5)),
                                                     (1024, (64, 64, 4, 5)), (2048, (64, 64, 4, 5)))},
    (2048, 6144): {"BLOCK_SIZE_K": 128, "m_buckets": ((1, (16, 64, 4, 4)), (4, (16, 64, 8, 4)), (8, (16, 32, 8, 4)), (16, (16, 32, 8, 4)),
                                                      (32, (16, 64, 8, 3)), (64, (32, 64, 4, 4)), (256, (32, 64, 4, 4)), (512, (64, 64, 4, 3)),
                                                      (1024, (64, 64, 4, 3)), (2048, (64, 64, 4, 3)))},
    (4096, 2048): {"BLOCK_SIZE_K": 64, "m_buckets": ((1, (16, 256, 8, 3)), (4, (16, 64, 4, 3)), (8, (16, 128, 8, 2)), (16, (16, 32, 4, 5)),
                                                     (32, (16, 64, 4, 4)), (64, (64, 64, 8, 3)), (256, (64, 64, 4, 5)), (512, (64, 64, 4, 5)),
                                                     (1024, (128, 64, 4, 4)), (2048, (128, 64, 8, 4)))},
    (12288, 2048): {"BLOCK_SIZE_K": 64, "m_buckets": ((1, (16, 128, 8, 2)), (4, (16, 64, 8, 5)), (8, (16, 128, 8, 4)), (16, (16, 128, 4, 5)),
                                                      (32, (32, 128, 4, 5)), (64, (32, 256, 4, 3)), (256, (128, 128, 8, 3)), (512, (128, 64, 4, 4)),
                                                      (1024, (128, 128, 8, 3)), (2048, (128, 128, 4, 3)))},
    (151936, 2048): {"BLOCK_SIZE_K": 64, "m_buckets": ((1, (16, 32, 8, 5)), (4, (16, 32, 8, 5)), (8, (16, 32, 8, 5)), (16, (16, 32, 4, 4)),
                                                       (32, (16, 64, 4, 4)), (64, (32, 64, 4, 4)), (256, (128, 128, 4, 3)), (512, (128, 128, 4, 3)),
                                                       (1024, (128, 128, 4, 3)), (2048, (128, 128, 4, 3)))},
}

#: [R13 hopper] vLLM's sm_90 table, VERBATIM `_BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["hopper"]` of the pinned vLLM (`batch_invariant_configs.py`
#: sha256 72ea0d4df0ab5ae288abb7df5c203d2ad5f1709c7295d87ec72da775fddc1b7a), all five keys (every row `BLOCK_SIZE_K` 128, M<=1..16 buckets
#: `BLOCK_SIZE_M` 16).  Llama-3.2-1B hits (2048, 2048) (o_proj) only; Qwen3-class shapes (K 2048; V 151936) hit the rest.
HOPPER_TUNED_GEMM: dict[tuple[int, int], dict[str, Any]] = {
    (12288, 2048): {"BLOCK_SIZE_K": 128, "m_buckets": ((1, (16, 256, 4, 3)), (4, (16, 256, 4, 4)), (8, (16, 64, 4, 4)), (16, (16, 64, 4, 4)),
                                                      (32, (32, 64, 8, 5)), (64, (64, 64, 4, 4)), (256, (64, 128, 4, 3)), (512, (128, 128, 8, 3)),
                                                      (1024, (128, 256, 8, 2)), (2048, (128, 256, 8, 2)))},
    (2048, 6144): {"BLOCK_SIZE_K": 128, "m_buckets": ((1, (16, 32, 4, 4)), (4, (16, 64, 4, 4)), (8, (16, 64, 4, 4)), (16, (16, 32, 4, 3)),
                                                     (32, (32, 32, 8, 3)), (64, (64, 32, 4, 5)), (256, (64, 128, 4, 3)), (512, (128, 128, 4, 3)),
                                                     (1024, (64, 64, 4, 5)), (2048, (64, 128, 4, 4)))},
    (4096, 2048): {"BLOCK_SIZE_K": 128, "m_buckets": ((1, (16, 64, 8, 5)), (4, (16, 64, 4, 5)), (8, (16, 128, 4, 3)), (16, (16, 64, 4, 5)),
                                                     (32, (16, 64, 4, 4)), (64, (64, 32, 4, 3)), (256, (64, 256, 4, 2)), (512, (64, 64, 4, 3)),
                                                     (1024, (64, 128, 4, 3)), (2048, (64, 64, 4, 4)))},
    (151936, 2048): {"BLOCK_SIZE_K": 128, "m_buckets": ((1, (16, 256, 4, 4)), (4, (16, 256, 4, 4)), (8, (16, 256, 4, 3)), (16, (16, 256, 4, 3)),
                                                       (32, (32, 256, 8, 3)), (64, (64, 128, 4, 3)), (256, (64, 128, 4, 3)), (512, (128, 128, 8, 3)),
                                                       (1024, (128, 256, 8, 2)), (2048, (128, 256, 8, 2)))},
    (2048, 2048): {"BLOCK_SIZE_K": 128, "m_buckets": ((1, (16, 64, 4, 5)), (4, (16, 128, 8, 3)), (8, (16, 32, 4, 3)), (16, (32, 64, 8, 5)),
                                                     (32, (16, 128, 4, 5)), (64, (32, 64, 4, 2)), (256, (128, 64, 4, 4)), (512, (64, 256, 4, 2)),
                                                     (1024, (64, 64, 4, 3)), (2048, (64, 128, 4, 3)))},
}


class UnregisteredGemmTarget(LookupError):
    """The target has no registered GEMM specialization: fail closed (message starts with `numerics_unregistered:`)."""


@dataclass(frozen=True)
class GemmTarget:
    key: str
    compute_capabilities: tuple[tuple[int, int], ...]
    vllm_arch_family: str | None
    dot: str
    tuned_table: Mapping[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    kind: str = "Gemm_v1{K,N}"

    def launch_config(self, M: int, N: int, K: int) -> tuple[dict[str, Any], int, int, str]:
        """vLLM `_get_matmul_config(M, N, K, bf16, default)` on this target: `(constexprs, num_warps, num_stages, source)`;
        `source` names the table row + bucket or `default`."""
        row = self.tuned_table.get((N, K))
        if row is None:
            t = DEFAULT_TILING
            consts = {"BLOCK_SIZE_M": t["BLOCK_SIZE_M"], "BLOCK_SIZE_N": t["BLOCK_SIZE_N"], "BLOCK_SIZE_K": t["BLOCK_SIZE_K"], **FIXED_CONSTEXPRS}
            return consts, t["num_warps"], t["num_stages"], f"default[{self.key}]"
        bm, bn, warps, stages = row["m_buckets"][-1][1]
        bucket = row["m_buckets"][-1][0]
        for max_m, cfg in row["m_buckets"]:
            if max_m >= M:
                bm, bn, warps, stages = cfg
                bucket = max_m
                break
        consts = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": row["BLOCK_SIZE_K"], **FIXED_CONSTEXPRS}
        return consts, warps, stages, f"tuned_{self.key}[(N={N},K={K})] bucket M<={bucket}"

    def table_sha256(self) -> str | None:
        """Identity of the pinned tuned table this target consults (an applicability condition); None for a table-less target."""
        if not self.tuned_table:
            return None
        from verity_vllm.program.frontend.target_profile import canonical_bytes, sha256_hex
        return sha256_hex(canonical_bytes({f"{k[0]},{k[1]}": v for k, v in self.tuned_table.items()}))

    def describe(self) -> dict:
        return {"key": self.key, "compute_capabilities": [list(c) for c in self.compute_capabilities], "vllm_arch_family": self.vllm_arch_family,
                "kind": self.kind, "dot": self.dot, "tuned_table_keys": [f"N={n},K={k}" for (n, k) in self.tuned_table],
                "tuned_table_sha256": self.table_sha256(), "evidence": list(self.evidence)}


REGISTERED: dict[str, GemmTarget] = {
    "ada": GemmTarget(
        key="ada", compute_capabilities=((8, 9),), vllm_arch_family="ada", dot="AmpereBF16TcDot16", tuned_table=ADA_TUNED_GEMM,
        evidence=("R10/R11 Tier A: LLAMA32_1B / SMOLLM2 / QWEN05 eager captures on the RTX 4090 and L40S (sm_89), every GEMM launch "
                  "(default tiling and the (2048,2048) table buckets M<=1 / M<=32) bit-exact against Gemm_v1{K,N} — tests/observe/test_gen_llama.py, "
                  "docs/vllm-poc/numerics.md (mma.AMPERE_BF16: Ada's bf16 mma.sync is parameter-identical to Ampere's, docs/hardware-semantics.md)",
                  # [R15 moe GEMM-ADA-TABLE] the four Qwen3-class rows of the table were registered without their own launches until R15
                  "R15 lane moe: verity_vllm/program/kernels/gemm_target_correspondence.py --model ada-tuned-table on cp-moe3 (L40S, cc 8.9, 142 SMs, "
                  "vLLM d9105ea80 family 'ada'): ALL five tuned (N,K) rows x M in {1,2,4,5,8,9,16,17,32,33,64,65,256,257,512,513,1024,1033,2048,2049} "
                  "(+ stress families at M 1/17/257 on (2048,6144)/(4096,2048)) = 124 cases, 30 distinct launch configs (every tabled bm 16/32/64/128 "
                  "bucket incl. (151936,2048) M<=4 bm16 and the block_k-128 row (2048,6144)), 152,105,216 coordinates bit-exact against Gemm_v1{K,N}, "
                  "GPU re-run deterministic — out/gen/sweep/evidence/ada/gemm_target_l40s_ada_table.json",)),
    "ampere": GemmTarget(
        key="ampere", compute_capabilities=((8, 0),), vllm_arch_family=None, dot="AmpereBF16TcDot16", tuned_table={},
        evidence=("R13 lane ampere: verity_vllm/program/kernels/gemm_target_correspondence.py on cp-a100 (A100-SXM4-80GB, cc 8.0, 108 SMs): "
                  "all 5 Llama-3.2-1B GEMM shapes x M in {1,2,7,13,32,128,129,515,1024} (+ stress families at M 1/7/129) launched with the default "
                  "128/128/64 8w/3s tiling (vLLM family None), Triton PTX sm_80 mma.sync.m16n8k16 bf16 only, bit-exact against the single ascending "
                  "k16 chain of mma.AMPERE_BF16 — out/gen/sweep/evidence/ampere/gemm_target_a100.json",)),
    # [R13 hopper] sm_90: Triton lowers `tl.dot` to `wgmma.mma_async` — a different k16 step (`HopperBF16WgmmaDot16`: one group of 16 products,
    # 26-bit adder, floor -133; lane HP's wgmma re-measurement = Hawkeye's H100 `mma.sync.m16n8k16` model), so the Definition is the
    # statics-lifted `Gemm_v2{K,N,DOT}` (`verity_vllm.program.registry.targets.gemm_spec`), not `Gemm_v1`.  vLLM family "hopper": the tuned table above.
    "hopper": GemmTarget(
        key="hopper", compute_capabilities=((9, 0),), vllm_arch_family="hopper", dot="HopperBF16WgmmaDot16", tuned_table=HOPPER_TUNED_GEMM,
        kind="Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16_v1}",
        evidence=("lane HP CONF-b1-hopper (Qwen2.5-1.5B on an H100: GemmHopper = Gemm with DOT=HopperBF16WgmmaDot16 exact on every GEMM leaf); "
                  "R13 lane hopper: verity_vllm/program/kernels/gemm_target_correspondence.py on cp-h100a (cc 9.0, 132 SMs, family 'hopper': the "
                  "(2048,2048) tuned rows BLOCK_SIZE_M 16 / BLOCK_SIZE_K 128 at M<=16 and the default tiling elsewhere, Triton PTX sm_90 wgmma) — "
                  "out/gen/sweep/evidence/hopper/gemm_target_h100.json; row llama32-1b__bf16__h100__…__bi-eager-fa2 Match replay (o_proj M=1 tuned row + "
                  "the default-tiled GEMMs bit-exact against Gemm_v2{K,N,DOT=Hopper})",)),
}


def vllm_tuned_gemm_arch_family(compute_capability: tuple[int, int]) -> str | None:
    """vLLM's `_get_tuned_matmul_arch_family` for a capability: the pinned vLLM's own function when it is importable (construction and
    capture hosts), else its rule at the pinned revision d9105ea80 verbatim (`major 10 -> blackwell; 9 -> hopper; (8, 9) -> ada; else None`)
    for torch-free hosts (the fold).  `tests/program/test_gemm_targets.py` checks the two agree wherever vLLM is importable."""
    major, minor = int(compute_capability[0]), int(compute_capability[1])
    try:
        from vllm.model_executor.determinism.batch_invariant_configs import _get_tuned_matmul_arch_family  # noqa: PLC0415
        from vllm.platforms.interface import DeviceCapability  # noqa: PLC0415
        return _get_tuned_matmul_arch_family(DeviceCapability(major, minor))
    except Exception:  # noqa: BLE001  (vLLM absent or another revision's module layout: the pinned rule)
        pass
    if major == 10:
        return "blackwell"
    if major == 9:
        return "hopper"
    if major == 8 and minor == 9:
        return "ada"
    return None


def gemm_target_for(compute_capability, arch_family: Any = ...) -> GemmTarget:
    """The registered specialization for a target capability (and, when given, the arch family the host actually reported — it
    must be the record's; a differing host answer is a refusal, not a re-selection).  Raises `UnregisteredGemmTarget`."""
    cc = (int(compute_capability[0]), int(compute_capability[1]))
    fam = vllm_tuned_gemm_arch_family(cc) if arch_family is ... else arch_family
    for t in REGISTERED.values():
        if cc in t.compute_capabilities:
            if fam != t.vllm_arch_family:
                raise UnregisteredGemmTarget(f"numerics_unregistered: GEMM target cc {cc[0]}.{cc[1]} reports tuned_gemm_arch_family={fam!r}, the registered "
                                             f"'{t.key}' specialization requires {t.vllm_arch_family!r} (vLLM's table selection changed: not this Definition)")
            return t
    known = ", ".join(f"{t.key} (cc {'/'.join(f'{a}.{b}' for a, b in t.compute_capabilities)}, family {t.vllm_arch_family!r})" for t in REGISTERED.values())
    raise UnregisteredGemmTarget(f"numerics_unregistered: no registered GEMM specialization for target cc {cc[0]}.{cc[1]} "
                                 f"(tuned_gemm_arch_family={fam!r}); registered: {known}. A launch showing a registered tiling on this target is still "
                                 f"refused — register the target with its own correspondence evidence (verity_vllm/program/registry/gemm_targets.py)")


def try_gemm_target_for(compute_capability, arch_family=...) -> tuple[GemmTarget | None, str | None]:
    """`(target, None)` or `(None, refusal message)`."""
    try:
        return gemm_target_for(compute_capability, arch_family), None
    except UnregisteredGemmTarget as e:
        return None, str(e)
