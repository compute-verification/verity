"""Per-instruction tensor-core models for Ampere ``mma.sync.m16n8k16`` (BF16 and FP16, FP32 accumulate).

BF16 is exactly :data:`veritor.core.silicon.AMPERE_BF16_M16N8K16` (Hawkeye's Ampere
parameters: two groups of eight products, 25-bit adder, exponent floor -132); this
module adds the FP16 operand product (``utils.cpp::multiply_float16_to_gfloat`` in
Hawkeye's simulator: 11x11-bit significands, product shifted by 3 to the 25-bit
grid, bias 15, subnormals unnormalised at exponent -14) and drives the same
``group_sum``.  Everything is integer arithmetic on raw encodings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.errors import InvalidArtifact
from verity.ml.circuits.silicon import silicon as S

__all__ = [
    "AMPERE_BF16",
    "AMPERE_F16",
    "MODELS",
    "TCModel",
    "f16_product",
    "model_for",
    "tc_dot",
    "tc_dot_chain",
]


def f16_product(a: int, b: int) -> S.Term:
    """Exact product of two IEEE binary16 words as a 25-bit-scaled :class:`~veritor.core.silicon.Term`."""

    ea, eb = (a >> 10) & 0x1F, (b >> 10) & 0x1F
    if ea == 0x1F or eb == 0x1F:
        raise InvalidArtifact("non-finite FP16 operand")
    sa = (a & 0x3FF) | (0x400 if ea else 0)
    sb = (b & 0x3FF) | (0x400 if eb else 0)
    magnitude = (sa * sb) << 3
    if magnitude == 0:
        return S.ZERO
    return S.Term(((a ^ b) >> 15) & 1 == 1, (ea or 1) + (eb or 1) - 30, magnitude)


@dataclass(frozen=True, slots=True)
class TCModel:
    """One ``mma`` shape's accumulation pipeline: ``groups`` in k order, adder ``width``, exponent floor."""

    name: str
    arch: str
    dtype: str  # "bf16" | "f16"
    instruction: str
    groups: tuple[int, ...]
    width: int
    zero_exponent: int
    version: str

    @property
    def k(self) -> int:
        return sum(self.groups)

    def product(self, a: int, b: int) -> S.Term:
        return S.bf16_product(a, b) if self.dtype == "bf16" else f16_product(a, b)

    @property
    def silicon_pipeline(self) -> S.Pipeline | None:
        """The ``veritor.core.silicon`` pipeline this model *is*, when it is one."""

        if self.dtype != "bf16":
            return None
        for p in S.PIPELINES.values():
            if (
                p.dtype == "bf16"
                and p.groups == self.groups
                and p.width == self.width
                and p.zero_exponent == self.zero_exponent
                and p.arch == self.arch
            ):
                return p
        return None


def tc_dot(model: TCModel, acc_bits: int, a: Sequence[int], b: Sequence[int]) -> int:
    """``D = acc + sum_k a[k] * b[k]`` as ``model``'s silicon computes it (FP32 words in and out)."""

    pipe = model.silicon_pipeline
    if pipe is not None:
        return S.tc_dot(pipe, acc_bits, a, b)
    if len(a) != model.k or len(b) != model.k:
        raise InvalidArtifact(f"{model.name} takes {model.k} operand pairs")
    acc = S.fp32_term(acc_bits)
    start = 0
    for size in model.groups:
        products = [model.product(a[i], b[i]) for i in range(start, start + size)]
        acc = S.group_sum(
            [acc, *products], width=model.width, zero_exponent=model.zero_exponent
        )
        if acc.magnitude and acc.exponent > S.FP32_MAX_EXPONENT:
            return S.pack_fp32(acc)
        start += size
    return S.pack_fp32(acc)


def tc_dot_chain(
    model: TCModel, acc_bits: int, a: Sequence[int], b: Sequence[int]
) -> int:
    """Fold :func:`tc_dot` over ``a``/``b`` in k-chunks of ``model.k`` (one GEMM output element, fixed k order)."""

    k = model.k
    if len(a) != len(b) or len(a) % k:
        raise InvalidArtifact(f"chain length must be a multiple of {k}")
    for start in range(0, len(a), k):
        acc_bits = tc_dot(model, acc_bits, a[start : start + k], b[start : start + k])
    return acc_bits


AMPERE_BF16 = TCModel(
    name="ampere_bf16_m16n8k16",
    arch="sm_80",
    dtype="bf16",
    instruction="mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
    groups=(8, 8),
    width=25,
    zero_exponent=-132,
    version="veritor.core.silicon.AMPERE_BF16_M16N8K16 (Hawkeye Ampere parameters)",
)

AMPERE_F16 = TCModel(
    name="ampere_f16_m16n8k16",
    arch="sm_80",
    dtype="f16",
    instruction="mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
    groups=(8, 8),
    width=25,
    zero_exponent=-132,
    version="verity.ml.definitions.numerics.mma.AMPERE_F16 v1 (Hawkeye Ampere FP16 product + BF16 group_sum)",
)

MODELS: dict[str, TCModel] = {m.name: m for m in (AMPERE_BF16, AMPERE_F16)}


def model_for(dtype: str) -> TCModel:
    return AMPERE_BF16 if dtype == "bf16" else AMPERE_F16
