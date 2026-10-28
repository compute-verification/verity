"""Resolved GEMM launch signatures on the A100 (sm_80) and the CPU model that reproduced each one
bit-exactly (raw evidence under
``fixtures/results/numerics-a100-*/raw/gemm_cublas.json`` and ``cublas_probe.json``).

The POC profile (``VLLM_BATCH_INVARIANT=1``) never reaches cuBLAS: every GEMM of the engine is
``matmul_kernel_persistent`` (a single ascending chain).  The cuBLAS entries below describe what
``torch.nn.functional.linear`` dispatches on the A100 *without* batch invariance for the
Qwen2.5-1.5B shapes, so that profile can be checked too.  A signature is the set of CUDA kernel
names one ``F.linear`` call launches (``torch.profiler``); the split count of a split-K kernel is
not visible in the names and was found by search, then confirmed on every natural coordinate of
the observed cases, so an entry is tied to the (K, M) ranges where it was observed.

``resolve(kernels, K, M, has_bias)`` returns a :class:`GemmModelSpec` or ``None`` (unresolved:
``gemvx`` at M=1, and any signature not in the table).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from verity.ml.definitions.numerics import cpu_model as CM
from verity.ml.definitions.numerics.mma import AMPERE_BF16

TRITON_KERNEL = "matmul_kernel_persistent"


def split_bounds(K: int, S: int, blk: int) -> tuple[int, ...]:
    """cuBLASLt contiguous split-K: ``S`` ranges of whole ``blk``-wide k blocks, ceil-sized, in
    K/16-tile units (the last range is shorter when ``ceil(nb / S) * S > nb``)."""
    nb = K // blk
    per = math.ceil(nb / S)
    return tuple(min(nb, s * per) * blk // 16 for s in range(S + 1))


def sliced_k_spec(K: int, slices: int, gran_tiles: int, bias_mode: int, reduce_mode: int = 0, first_slice: int = 0) -> CM.GemmModelSpec | None:
    """Slice-K as a CTA-internal warp split does it: slice ``s`` owns every ``gran_tiles``-tile
    group whose index is congruent to ``s`` mod ``slices`` (e.g. a 64x64 ``sliced1x2`` kernel with a
    64-wide K block: slice 0 takes k in [64i, 64i+32), slice 1 takes [64i+32, 64i+64)).  Each slice
    is one ascending chain; the FP32 partials are reduced at the end.  ``first_slice`` rotates
    which slice is partial 0 (the one a ``bias_mode=4`` epilogue adds the bias to first)."""
    ntiles = K // 16
    groups = [[t for t in range(ntiles) if (t // gran_tiles) % slices == s] for s in range(slices)]
    if any(len(g) == 0 for g in groups):
        return None
    groups = groups[first_slice:] + groups[:first_slice]
    order = tuple(t for g in groups for t in g)
    bounds = [0]
    for g in groups:
        bounds.append(bounds[-1] + len(g))
    rot = "" if first_slice == 0 else f", partial 0 = slice {first_slice}"
    return CM.GemmModelSpec(
        AMPERE_BF16.name,
        nsplit=slices,
        bounds=tuple(bounds),
        reduce_mode=reduce_mode,
        order=order,
        bias_mode=bias_mode,
        note=f"slice-K {slices} interleaved at {gran_tiles * 16}-wide k granularity{rot}",
    )


def sliced_1x2(K: int, has_bias: bool) -> CM.GemmModelSpec:
    """``*_sliced1x2_*`` 64x64 kernels: two warp slices own alternating 32-wide k blocks
    (slice 0: k in [64i, 64i+32), slice 1: [64i+32, 64i+64)), each an ascending chain; the bias
    is added in FP32 to slice 0's partial, then slice 1's partial is added, then RNE to BF16."""
    spec = sliced_k_spec(K, 2, 2, 4 if has_bias else 0)
    assert spec is not None
    return spec


@dataclass(frozen=True)
class ZooEntry:
    signature: tuple[str, ...]  # substrings that must all occur in the launch's kernel names
    forbid: tuple[str, ...]  # substrings that must not occur
    K: int | tuple[int, ...] | None  # the K value(s) where this entry was observed exact; None = any K
    M_range: tuple[int, int]  # inclusive range of M where this entry was observed
    model: str  # one of the builders below
    note: str

    def admits(self, K: int, M: int) -> bool:
        ks = self.K if isinstance(self.K, tuple) else (self.K,)
        return (self.K is None or K in ks) and self.M_range[0] <= M <= self.M_range[1]


_SINGLE_FORBID = ("splitKreduce", "Memset", "sliced")

# Every entry is one launch signature observed on the A100 with torch's default cuBLAS/cuBLASLt
# dispatch for the Qwen2.5-1.5B shapes (raw/gemm_cublas.json), at the K value(s) and M range it
# was observed at, and it was bit-exact on every natural coordinate there.  Nothing wider is
# claimed: a kernel of the same family at another tile config, K or M is unresolved.
ENTRIES: tuple[ZooEntry, ...] = (
    ZooEntry((TRITON_KERNEL,), (), None, (1, 1 << 30), "single", "vLLM batch-invariant matmul_kernel_persistent (POC profile); 199/199 cases exact on 0.28.0 and on 0.28.1rc1.dev472"),
    # 64x64 sliced-K (two warp slices over alternating 32-wide k blocks): 64x5_tn, 64x6_tn, relu_64x6_tn
    ZooEntry(("ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8", "f2f_stages_64x"), ("splitKreduce", "Memset"), 1536, (1, 100), "sliced1x2", "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8[_relu]_f2f_stages_64x{5,6}_tn; qkv/o/gate_up M=7..100, lm_head M=1..16"),
    # one chain, in-kernel FP32 bias epilogue (the _relu names are cuBLASLt's bias-epilogue kernels)
    ZooEntry(("ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn",), _SINGLE_FORBID, 1536, (100, 100), "single", "lm_head / gate_up_proj M=100"),
    ZooEntry(("ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn",), _SINGLE_FORBID, (1536, 8960), (512, 2048), "single", "o/gate_up/down/lm_head M=512 (lm_head, gate_up) and M=2048 (all)"),
    ZooEntry(("ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn",), _SINGLE_FORBID, 1536, (2048, 2048), "single", "qkv_proj M=2048 with bias"),
    ZooEntry(("ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn",), _SINGLE_FORBID, 1536, (64, 512), "single", "gate_up_proj M=64, o_proj M=512"),
    ZooEntry(("ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_64x3_tn",), _SINGLE_FORBID, 1536, (512, 512), "single", "qkv_proj M=512 with bias"),
    ZooEntry(("cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8",), _SINGLE_FORBID, 1536, (64, 64), "single", "lm_head M=64"),
    ZooEntry(("cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8", "splitKreduce_kernel<32, 16, int, float"), (), 8960, (7, 16), "splitk4_fp32", "down_proj M=7/16: split-K 4 over 32-wide k blocks, FP32 workspace partials summed ascending"),
    ZooEntry(("s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn", "splitKreduce_kernel<32, 16, int, __nv_bfloat16"), (), 8960, (64, 64), "splitk8_bf16", "down_proj M=64: split-K 8 over 64-wide k blocks, BF16 workspace partials (RNE) summed ascending in FP32"),
    ZooEntry(("s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn", "splitKreduce_kernel<32, 16, int, __nv_bfloat16"), (), 8960, (100, 100), "splitk8_bf16", "down_proj M=100: split-K 8 over 64-wide k blocks, BF16 workspace partials summed ascending in FP32"),
    ZooEntry(("s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn", "Memset"), ("splitKreduce",), 8960, (512, 512), "serial2", "down_proj M=512: serial split-K 2 through the BF16 output (s = bf16(p0); s = bf16(s + p1))"),
    ZooEntry(("s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn", "Memset"), ("splitKreduce",), 1536, (1, 1), "serial2", "gate_up_proj M=1: serial split-K 2 through the BF16 output"),
)

UNRESOLVED = {
    "gemvx": "cuBLAS internal::gemvx::kernel (M=1, K=1536, N<=2048: qkv/o/down at decode M=1): FMA-based GEMV with a warp-shuffle reduction, not the tensor-core MMA; no model attempted",
}


def build(model: str, K: int, has_bias: bool) -> CM.GemmModelSpec:
    bias_mode = 1 if has_bias else 0
    if model == "single":
        return CM.GemmModelSpec(AMPERE_BF16.name, bias_mode=bias_mode, note="single ascending chain")
    if model == "sliced1x2":
        return sliced_1x2(K, has_bias)
    if model == "splitk4_fp32":
        return CM.GemmModelSpec(AMPERE_BF16.name, nsplit=4, bounds=split_bounds(K, 4, 32), reduce_mode=0, bias_mode=bias_mode, note="split-K 4 in 32-wide k blocks, FP32 partials")
    if model == "splitk8_bf16":
        return CM.GemmModelSpec(AMPERE_BF16.name, nsplit=8, bounds=split_bounds(K, 8, 64), reduce_mode=3, bias_mode=bias_mode, note="split-K 8 in 64-wide k blocks, BF16-rounded partials")
    if model == "serial2":
        return CM.GemmModelSpec(AMPERE_BF16.name, nsplit=2, bounds=split_bounds(K, 2, 32), reduce_mode=7, bias_mode=bias_mode, note="serial split-K 2 through the BF16 output")
    raise KeyError(model)


def resolve(kernels: list[str] | tuple[str, ...] | str, K: int, M: int, has_bias: bool) -> CM.GemmModelSpec | None:
    names = [kernels] if isinstance(kernels, str) else list(kernels)
    blob = "\n".join(names)
    for entry in ENTRIES:
        if not entry.admits(K, M):
            continue
        if all(s in blob for s in entry.signature) and not any(f in blob for f in entry.forbid):
            return build(entry.model, K, has_bias)
    return None


def why_unresolved(kernels: list[str] | tuple[str, ...] | str) -> str | None:
    blob = "\n".join([kernels] if isinstance(kernels, str) else list(kernels))
    for key, reason in UNRESOLVED.items():
        if key in blob:
            return reason
    return "launch signature not in the resolved zoo" if blob else None
