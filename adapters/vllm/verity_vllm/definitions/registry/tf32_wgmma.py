"""The candidate sm_90 `wgmma.mma_async.sync.aligned.m64nNk8.f32.tf32.tf32` step on f32 words, each operand rounded RNE to TF32:
the k8 step cuBLAS's TF32 GEMM and DeepGEMM's TF32 kernels take."""
from __future__ import annotations

from verity.circuits.census import silicon_step
from verity.circuits.defs import primitive
from verity.circuits.silicon.models import HOPPER_BF16_WGMMA_K16, GroupSum
from verity.circuits.silicon.term import TF32, tf32_product
from verity.circuits.silicon.total import NAN, Inf, fp32_value, group_step_total, pack_value
from verity.circuits.types import Array

from verity_vllm.definitions.evidence import EVIDENCE
from verity_vllm.definitions.registry.prims import F32

M32 = 0xFFFFFFFF

_STEP_PROBE = EVIDENCE["hopper_tf32_k8_step_probe"]["run_record"]
_TF32_CONF = (f"exact-model-tested: a probe on a cc 9.0 GPU (132 SMs; {_STEP_PROBE}) fitted this k8 step alone (one group, "
              "26 bits, floor -133) on exact products after each f32 operand is rounded RNE to TF32")

#: The candidate k8 step's grouped sum: the sm_90 BF16 step's adder (`HOPPER_BF16_WGMMA_K16`) over the eight products
TF32_GROUPS, TF32_WIDTH, TF32_FLOOR = (8,), 26, -133
HOPPER_TF32_WGMMA_K8_CANDIDATE = GroupSum(
    name="hopper_tf32_wgmma_k8_candidate", arch=HOPPER_BF16_WGMMA_K16.arch, dtype="tf32", operand_bits=19, groups=TF32_GROUPS,
    width=TF32_WIDTH, zero_exponent=TF32_FLOOR, validated=f"{_STEP_PROBE}: the cc 9.0 probe (132 SMs) fitted this step alone")


def tf32_operand(word: int) -> int:
    """The 19-bit TF32 word an HGMMA TF32 operand reads: f32 rounded to nearest, ties to even.

    A NaN remains a NaN even when its payload lives entirely in the thirteen discarded bits.  The tensor-core step only
    observes its class, so one low TF32 payload bit is sufficient.
    """
    word &= M32
    high = word >> 13
    if word & 0x7F800000 == 0x7F800000:
        return high | int(bool(word & 0x7FFFFF))
    return ((word + 0xFFF + (high & 1)) & M32) >> 13


def tf32_product_total(a: int, b: int):
    """The product of two TF32 words as a `silicon.total` Value: NaN for a NaN operand or inf times zero, else a signed
    infinity, else exact (`term.tf32_product`)."""
    da, db = TF32.decode(a), TF32.decode(b)
    if da == "nan" or db == "nan":
        return NAN
    if isinstance(da, str) or isinstance(db, str):
        if (not isinstance(da, str) and da[2] == 0) or (not isinstance(db, str) and db[2] == 0):
            return NAN
        return Inf(bool(((a ^ b) >> 18) & 1))
    return tf32_product(a, b)


def tf32_dot_total(acc: int, a, b, groups=TF32_GROUPS, width: int = TF32_WIDTH, floor: int = TF32_FLOOR) -> int:
    """``acc + sum a[i] b[i]`` on f32 words at the given grouped-sum parameters (the candidate's by default)."""
    v, start = fp32_value(acc & M32), 0
    for size in groups:
        prods = [tf32_product_total(tf32_operand(a[i]), tf32_operand(b[i])) for i in range(start, start + size)]
        v = group_step_total(v, prods, width=width, zero_exponent=floor)
        start += size
    return pack_value(v)


@primitive("HopperTf32WgmmaDot8", 1, [("acc", F32), ("a", Array(8, F32)), ("b", Array(8, F32))], F32, conformance=_TF32_CONF)
def HopperTf32WgmmaDot8(acc: int, *ab: int) -> int:
    """Candidate: one accumulator coordinate of `wgmma.mma_async.sync.aligned.m64n{N}k8.f32.tf32.tf32` on sm_90 with f32 words
    as operands: each operand rounded RNE to a 19-bit TF32 word, the eight exact products and `acc` in one group through the Hopper BF16
    step's 26-bit truncating adder with exponent floor -133 (`HOPPER_TF32_WGMMA_K8_CANDIDATE`); total on every encoding as that
    step (NaN 0x7FFFFFFF, sticky infinities, a zero +0)."""
    return tf32_dot_total(acc, ab[:8], ab[8:16])


silicon_step(HopperTf32WgmmaDot8, HOPPER_TF32_WGMMA_K8_CANDIDATE)
