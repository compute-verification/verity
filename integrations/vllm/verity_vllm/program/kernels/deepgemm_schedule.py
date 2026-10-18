"""DeepGEMM 8b1392b9's launch-configuration choice for the cc 9.0 block-FP8 1d2d GEMM, mirrored exactly, and the token counts
where it changes: the model behind the deepgemm capture's batch sweep (tests/properties/deepgemm_capture_gpu.py).

`fp8_gemm_nt` (GemmType::Normal) and `fp8_einsum`'s `sm90_fp8_bmm` (GemmType::Batched) pick one `Layout` per call in
`get_best_config<SM90ArchSpec>` (csrc/jit_kernels/heuristics/sm90.hpp, common.hpp at 8b1392b9): enumerate the candidates in a fixed
order (cluster_m 1..2, cluster_n 1..2 with cluster size at most 2, Batched without multicast; block_m 64, 128, then 16 when m <= 16,
32 when m <= 32, then 256 for bf16 output; block_n 16..192 step 16), drop a candidate whose 1d2d unroll, register or pipeline-stage
constraint fails, score each by its modelled L1/L2 cycles over the wave partition of `num_sms`, and keep the first strict minimum.
The chosen layout fixes the JIT'd kernel's template arguments (sm90_fp8_gemm_1d2d.hpp:34-68): block sizes, swizzles, stages,
threads, cluster and multicast.  None of it moves an output word: the kernel has no split-K, a tile's k-blocks accumulate in
ascending order whatever the partition, so the per-row Definitions (`registry.deepgemm`) hold at every M and the capture's job is
to demonstrate that at every configuration the sweep can reach.

The scoring divides int64 cycle counts by a float32 wave efficiency exactly as the C++ does, and a tie keeps the earlier
candidate, so `best_config` is bit-faithful to `compare`'s strict `<`.  A config depends on m only through `ceil(m / 16)`
(every candidate block_m is a multiple of 16, or 16 itself), so `boundaries` scans those steps and every boundary is of the
form 16t + 1.

The capture asserts the model: every DeepGEMM launch's predicted template arguments must match the JIT cache's generated
`kernel.cu`, and a sweep covers both sides of every boundary plus the endpoint (`sweep_ms`).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

#: SM90ArchSpec::smem_capacity, and the 1d2d kernel's fixed BLOCK_K for e4m3 operands
SMEM_CAPACITY = 232448
BLOCK_K = 128
#: kNumMaxStages, and its barrier bytes (kNumMaxStages * 8 * 2)
MAX_STAGES, BARRIER_BYTES = 16, 256
#: block_n candidates: 16..192 step 16 (lcm(16, block_n_multiple_of = 1); Kernel1D2D's end)
BLOCK_NS = tuple(range(16, 193, 16))
INT64_MAX = 2**63 - 1
#: the template's fixed strings (runtime_utils.hpp to_string; fp8_gemm_nt and fp8_einsum pass contiguous scales and bf16 out)
MAJOR_SFB, CD_DTYPE, EPILOGUE = "cute::UMMA::Major::K", "cutlass::bfloat16_t", "epilogue::transform::EpilogueIdentity"
NUM_TMA_THREADS = 128


class Config(NamedTuple):
    """One chosen layout: what `GemmConfig` carries into `sm90_fp8_gemm_1d2d_impl`'s template arguments."""

    block_m: int
    block_n: int
    cluster_m: int
    cluster_n: int
    num_stages: int
    swizzle_cd: int
    num_math_threads: int


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _align(x: int, a: int) -> int:
    return cdiv(x, a) * a


def _swizzle_mode(block_size: int, elem_size: int) -> int:
    """`get_swizzle_mode`: the largest of 128, 64, 32, 16 dividing block_size * elem_size."""
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    raise ValueError(f"no swizzle mode divides {block_size} * {elem_size}")


def num_stages(block_m: int, block_n: int, k: int) -> int:
    """`get_pipeline_config`'s stage count for the e4m3-in, bf16-out 1d2d kernel (f32 sfa per stage, k/128 extra sfb words)."""
    smem_cd = _align(block_m * block_n * 2, 1024)
    per_stage = block_m * BLOCK_K + block_n * BLOCK_K + _align(block_m * 4, 128)
    uniform_sfb = 1 if BLOCK_K % block_n == 0 else 2
    extra_sfb = _align(cdiv(k, BLOCK_K) * 4 * uniform_sfb, 8)
    return min((SMEM_CAPACITY - smem_cd - BARRIER_BYTES - extra_sfb) // per_stage, MAX_STAGES)


def _candidates(m: int, k: int, num_sms: int, batched: bool):
    """`get_layout_candidates` in its enumeration order, already filtered; yields (cluster_m, cluster_n, block_m, block_n, stages).

    Batched disables multicast; 256 is a candidate because the output is bf16; the unroll, register-pressure (block_m and block_n
    not both above 128) and stage-count constraints are sm90.hpp:84-108."""
    block_ms = [64, 128] + ([16] if m <= 16 else []) + ([32] if m <= 32 else []) + [256]
    clusters = ((1, 1),) if batched else ((1, 1), (1, 2), (2, 1))
    for cm, cn in clusters:
        if num_sms % (cm * cn):
            continue
        for bm in block_ms:
            for bn in BLOCK_NS:
                if bn > BLOCK_K and (bn % (bn - BLOCK_K)) and (BLOCK_K % (bn - BLOCK_K)):
                    continue
                if bm > 128 and bn > 128:
                    continue
                stages = num_stages(bm, bn, k)
                if stages < 3 or (bm * bn < 128 * 192 and stages < 4):
                    continue
                yield cm, cn, bm, bn, stages


def _cycles(m: int, n: int, k: int, groups: int, sms: int, cm: int, cn: int, bm: int, bn: int) -> int:
    """`get_layout_info`'s num_cycles: int64 L1/L2 cycle counts, float32 wave efficiency, truncated back to int64; a multicast
    layout with a single wave is INT64_MAX (sm90.hpp:201-238)."""
    blocks = cdiv(m, bm) * cdiv(n, bn) * groups
    waves = cdiv(blocks, sms)
    l2_bw = int(min(64.0 * sms, 8e6 / 1.3e3))
    l1_bw = 128 * sms
    bytes_l2_ab = k * (bm // cn + bn // cm)
    bytes_l1_ab = k * (bm + bn)
    bytes_l1_tc = k * (max(64, bm) + bn) + bm * bn * 2
    bytes_cd = bm * bn * 2
    l2_cycles = (bytes_l2_ab + bytes_cd) * blocks // l2_bw
    l1_cycles = (bytes_l1_ab + bytes_l1_tc + bytes_cd) * blocks // l1_bw
    if cm * cn > 1 and waves <= 1:
        return INT64_MAX
    efficiency = np.float32(blocks) / np.float32(waves * sms)
    return int(np.float32(max(l1_cycles, l2_cycles)) / efficiency)


def best_config(m: int, n: int, k: int, *, num_sms: int, num_groups: int = 1, batched: bool = False) -> Config:
    """The layout `get_best_config<SM90ArchSpec>` picks for an e4m3 x e4m3 -> bf16 1d2d GEMM (`fp8_gemm_nt` when not ``batched``,
    `sm90_fp8_bmm` when it is): the first candidate in enumeration order whose modelled cycles no later candidate strictly beats."""
    best, best_cycles = None, None
    for cm, cn, bm, bn, stages in _candidates(m, k, num_sms, batched):
        cycles = _cycles(m, n, k, num_groups, num_sms, cm, cn, bm, bn)
        if best_cycles is None or cycles < best_cycles:
            best, best_cycles = (cm, cn, bm, bn, stages), cycles
    if best is None:
        raise ValueError(f"no layout candidate at m={m}, n={n}, k={k}")
    cm, cn, bm, bn, stages = best
    return Config(block_m=bm, block_n=bn, cluster_m=cm, cluster_n=cn, num_stages=stages,
                  swizzle_cd=_swizzle_mode(bn, 2), num_math_threads=128 if bm <= 64 else 256)


def boundaries(n: int, k: int, *, num_sms: int, num_groups: int = 1, batched: bool = False,
               lo: int, hi: int) -> list[int]:
    """Every m in (lo, hi] where `best_config` differs from m - 1's.  A config depends on m only through cdiv(m, 16), so only
    m = 16t + 1 can be a boundary and the scan steps there."""
    out, prev = [], best_config(lo, n, k, num_sms=num_sms, num_groups=num_groups, batched=batched)
    for m in range(16 * cdiv(lo, 16) + 1, hi + 1, 16):
        cfg = best_config(m, n, k, num_sms=num_sms, num_groups=num_groups, batched=batched)
        if cfg != prev:
            out.append(m)
        prev = cfg
    return out


def sweep_ms(base: tuple[int, ...], n: int, k: int, *, num_sms: int, num_groups: int = 1, batched: bool = False,
             lo: int, hi: int) -> tuple[int, ...]:
    """The capture's batch sweep for one shape: ``base`` (its dense small-M points), both sides of every launch-configuration
    boundary in (lo, hi], and the endpoint hi, sorted without duplicates."""
    ms = {m for m in base if m <= hi} | {hi}
    for b in boundaries(n, k, num_sms=num_sms, num_groups=num_groups, batched=batched, lo=lo, hi=hi):
        ms |= {b - 1, b}
    return tuple(sorted(ms))


def template_args(n: int, k: int, *, num_groups: int, num_sms: int, gemm_type: str, config: Config) -> list[str]:
    """`sm90_fp8_gemm_1d2d_impl`'s 20 template arguments as `generate_impl` writes them (sm90_fp8_gemm_1d2d.hpp:34-68) for a
    launch at this shape and config: major_sfb, the compiled dims (m runtime, n and k compiled: the API's default "nk"),
    num_groups, block m/n/k, swizzles a/b/cd, stages, TMA and math threads, cluster size, multicast-on-B, SMs, GEMM type,
    output dtype and epilogue."""
    c = config
    swizzle_ab = _swizzle_mode(BLOCK_K, 1)
    return [MAJOR_SFB, "0", str(n), str(k), str(num_groups),
            str(c.block_m), str(c.block_n), str(BLOCK_K),
            str(swizzle_ab), str(swizzle_ab), str(c.swizzle_cd),
            str(c.num_stages), str(NUM_TMA_THREADS), str(c.num_math_threads),
            str(c.cluster_m * c.cluster_n), "true" if c.cluster_n > 1 else "false",
            str(num_sms), f"GemmType::{gemm_type}", CD_DTYPE, EPILOGUE]


__all__ = ["BLOCK_K", "Config", "best_config", "boundaries", "num_stages", "sweep_ms", "template_args"]
