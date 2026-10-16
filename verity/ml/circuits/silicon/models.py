"""Models: how a tensor-core instruction folds exact products into its FP32 accumulator.

A *model* is the executable, bit-exact semantics of one accumulator coordinate of one instruction class:
``step(c_word, a_words, b_words) -> d_word`` on raw encodings.  Two model shapes exist so far:

* :class:`GroupSum` -- Hawkeye's grouped, truncating accumulation (Ampere / Ada / Hopper, BF16 and FP8): the K
  exact products are summed together with the incoming accumulator in fixed-size *groups*; every term of a group
  is rescaled to the adder's ``width``, aligned to the group's maximum exponent by a truncating right shift,
  summed exactly, and the sum is normalised back to ``width`` bits by truncation (round toward zero).  The result
  of one group is the accumulator of the next.  ``Pipeline`` is this class's historical name and stays an alias.
* :class:`BlockScaledAlignAdd` -- Blackwell's block-scaled FP4 path (``mma.sync kind::mxf4nvf4`` on sm_120), as
  measured on the RTX 5090 by lanes fp4-5090 and fp4-re: four *exact* 16-term group dots over E2M1, per-block
  scales applied losslessly, then one five-way align-add of the four scaled terms and the carried accumulator whose
  alignment grid is set by the *scale exponents* of the leading live group (27 bits below the unit weight of its
  scaled dot) and by a 36-bit window riding the accumulator's exponent, truncating toward zero, normalised RZ, every
  zero result ``+0``.  This supersedes autoproof's *model5* (a 42-bit window anchored on the leading addend's msb
  plus a "hot group clip" at ``2^-19``): model5 is the special case of this rule in which the leading group's
  mantissa is 15 bits wide, and the hot clip is what the rule gives for scale exponent fields of 15 and 15.

The :class:`Model` protocol is what the rest of Verity may rely on; :data:`MODELS` lists every instance, and
``instructions.INSTRUCTIONS`` says which instruction each one is the model *of* and how well it is validated.

No IEEE rounding happens anywhere in either model; this is what NVIDIA's silicon does and it is *not* what a
sequence of FP32 FMAs would do.  GroupSum is the model recovered by Hawkeye (Badash, Boneh, Komargodski,
Srivastava, "Hawkeye: Reproducing GPU-Level Non-Determinism", MLSys 2026, arXiv 2603.20421); the body is a
bit-exact pure-Python port of https://github.com/badasherez/gpu-simulator (``src/Ampere_simulator.cpp``,
``src/Hopper_simulator.cpp``, ``src/Hopper_fp8_simulator.cpp``, ``utils/utils.cpp``, commit
``30703fcb309c943a6df5eee0277cb81815deb8f4``), generalised so that the group structure, internal width and
exponent floor are parameters, plus the saturation rule measured on Ada (a group leaving the FP32 range is a
sticky infinity).  BlockScaledAlignAdd started as a port of autoproof ``fp4-ref/src/native.rs`` (``transducer`` /
``native_dot_f32``) and ``spec/src/dyadic.rs`` (``trunc_rz``, ``to_f32``) as adopted in
``spec/SEMANTICS.lock.v1.json`` (replica: ``verity/ml/circuits/silicon/captures/autoproof-SEMANTICS.lock.v1.json``); its alignment rule was
re-fitted by lane fp4-re on the RTX 5090 captures of lane fp4-5090 (log entries D16, D17, D18) and confirmed by the
fp4-re runs cited in each instance's ``validated`` string.

The ``GroupSum`` code moved here from ``silicon.py`` unchanged (Phase 0 of the silicon campaign); ``silicon.py``
re-exports every name, so ``silicon.tc_dot`` is still the executable definition the frozen target names.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .errors import InvalidArtifact
from .integer import BLACKWELL_SM120_S8_M16N8K32, BLACKWELL_SM120_S8_M16N8K32_WRAP
from .term import (
    FP32_MAX_EXPONENT,
    FP32_MIN_NORMAL_EXPONENT,
    FP32_SIGNIFICAND_WIDTH,
    ZERO,
    Term,
    e2m1_product,
    fp32_term,
    pack_fp32,
    product,
)

__all__ = [
    "ADA_BF16_M16N8K16",
    "ADA_E4M3_M16N8K32",
    "ADA_E5M2_M16N8K32",
    "AMPERE_BF16_M16N8K16",
    "BLACKWELL_SM120_E2M1_M16N8K32",
    "BLACKWELL_SM120_E4M3_M16N8K32",
    "BLACKWELL_SM120_E5M2_M16N8K32",
    "BLACKWELL_SM120_MXF4",
    "BLACKWELL_SM120_MXF4_UNDERFLOW_SIGNED",
    "BLACKWELL_SM120_NVF4",
    "BlockScaledAlignAdd",
    "GroupSum",
    "HAWKEYE_AMPERE_GROUPSUM_E4M3_V0",
    "HOPPER_BF16_M16N8K16",
    "HOPPER_BF16_WGMMA_K16",
    "HOPPER_E4M3_K32",
    "HOPPER_E5M2_K32",
    "MODELS",
    "Model",
    "PIPELINES",
    "Pipeline",
    "group_sum",
    "pipeline_for",
    "tc_dot",
    "tc_dot_chain",
]


@runtime_checkable
class Model(Protocol):
    """One instruction class's accumulator step on raw words.

    ``step`` takes the incoming FP32 accumulator word and the operand words of one instruction (``a_words`` /
    ``b_words`` each ``words_per_step`` long: ``k`` element words, followed for block-scaled models by the
    ``k / block`` scale bytes) and returns the exact FP32 word the silicon writes.  Inputs outside the modelled
    domain (non-finite words, results the model does not define) raise :class:`InvalidArtifact` -- the model
    never guesses.  ``product`` is the exact product of one operand pair as a :class:`Term`.
    """

    name: str
    arch: str
    dtype: str
    operand_bits: int
    validated: str

    @property
    def k(self) -> int: ...

    @property
    def words_per_step(self) -> int: ...

    def step(self, c_word: int, a_words: Sequence[int], b_words: Sequence[int]) -> int: ...

    def product(self, a: int, b: int) -> Term: ...


# =====================================================================================================================
# GroupSum: Hawkeye's grouped truncating accumulation
# =====================================================================================================================

def group_sum(terms: Sequence[Term], *, width: int, zero_exponent: int) -> Term:
    """One grouped, truncating accumulation step (``*_simulator.cpp::group_sum``).

    ``width`` is the internal significand width of the adder (25 on Ampere,
    26 on Hopper, 14 on Hopper FP8) and ``zero_exponent`` the exponent floor
    (the maximum exponent used when every term is zero).  Every term is
    rescaled from the 24-bit FP32 grid to ``width`` bits, aligned to the
    largest exponent by a truncating right shift, summed exactly, and the
    result is normalised to ``width`` bits by truncation, denormalised if it
    is below the FP32 normal range, and truncated back to 24 bits.
    """

    rescale = width - FP32_SIGNIFICAND_WIDTH
    max_exponent = zero_exponent
    for term in terms:
        if term.magnitude and term.exponent > max_exponent:
            max_exponent = term.exponent
    total = 0
    for term in terms:
        if not term.magnitude:
            continue
        scaled = (
            term.magnitude << rescale if rescale >= 0 else term.magnitude >> -rescale
        )
        aligned = scaled >> (max_exponent - term.exponent)
        total += -aligned if term.negative else aligned
    if total == 0:
        return ZERO
    negative = total < 0
    magnitude = -total if negative else total
    bit_length = magnitude.bit_length()
    exponent = max_exponent + bit_length - width
    if bit_length > width:
        magnitude >>= bit_length - width
    else:
        magnitude <<= width - bit_length
    if exponent < FP32_MIN_NORMAL_EXPONENT:
        magnitude >>= FP32_MIN_NORMAL_EXPONENT - exponent
        exponent = FP32_MIN_NORMAL_EXPONENT
    magnitude = magnitude >> rescale if rescale >= 0 else magnitude << -rescale
    if magnitude == 0:
        return ZERO
    return Term(negative, exponent, magnitude)


@dataclass(frozen=True, slots=True)
class GroupSum:
    """The silicon parameters of one grouped-sum ``mma``/``wgmma`` shape on one architecture.

    ``groups`` lists, in k order, how many products each grouped sum takes;
    the incoming accumulator joins the first group and each group's result
    joins the next.  ``operand_bits`` is 16 for BF16 and 8 for E4M3/E5M2.
    ``dtype`` names the operand format (``term.FORMATS``); a ``width`` below
    24 means the incoming FP32 accumulator's significand is *truncated* to the
    adder before the grouped add (Hopper/Ada FP8: 14 bits), exactly as a
    product is.
    """

    name: str
    arch: str
    dtype: str
    operand_bits: int
    groups: tuple[int, ...]
    width: int
    zero_exponent: int
    validated: str

    @property
    def k(self) -> int:
        return sum(self.groups)

    @property
    def words_per_step(self) -> int:
        return self.k

    def product(self, a: int, b: int) -> Term:
        return product(self.dtype, a, b)

    def step(self, c_word: int, a_words: Sequence[int], b_words: Sequence[int]) -> int:
        """``D = acc + sum_k a[k] * b[k]`` exactly as this silicon computes it (FP32 words in and out)."""

        if len(a_words) != self.k or len(b_words) != self.k:
            raise InvalidArtifact(f"{self.name} takes {self.k} operand pairs")
        acc = fp32_term(c_word)
        start = 0
        for size in self.groups:
            products = [self.product(a_words[i], b_words[i]) for i in range(start, start + size)]
            acc = group_sum([acc, *products], width=self.width, zero_exponent=self.zero_exponent)
            if acc.magnitude and acc.exponent > FP32_MAX_EXPONENT:
                # A group whose result leaves the FP32 range saturates to an
                # infinity of its own sign, and that infinity is sticky through
                # the remaining groups (measured on Ada; not part of Hawkeye's
                # model, whose simulator never reaches this range).
                return pack_fp32(acc)
            start += size
        return pack_fp32(acc)


#: The historical name of :class:`GroupSum`; every existing import keeps working.
Pipeline = GroupSum


def tc_dot(pipeline: GroupSum, acc_bits: int, a: Sequence[int], b: Sequence[int]) -> int:
    """``D = acc + sum_k a[k] * b[k]`` exactly as ``pipeline``'s silicon computes it.

    ``acc_bits`` is an FP32 word, ``a`` and ``b`` are ``pipeline.k`` operand
    words each; the result is an FP32 word.  This is :meth:`GroupSum.step`.
    """

    return pipeline.step(acc_bits, a, b)


def tc_dot_chain(model: Model, acc_bits: int, a: Sequence[int], b: Sequence[int]) -> int:
    """Fold ``model.step`` over ``a``/``b`` in chunks of ``model.words_per_step`` (a GEMM output element)."""

    n = model.words_per_step
    if len(a) != len(b) or len(a) % n:
        raise InvalidArtifact(f"chain length must be a multiple of {n}")
    for start in range(0, len(a), n):
        acc_bits = model.step(acc_bits, a[start : start + n], b[start : start + n])
    return acc_bits


# --- Validated pipelines ----------------------------------------------------

AMPERE_BF16_M16N8K16 = GroupSum(
    name="ampere_bf16_m16n8k16",
    arch="sm_80",
    dtype="bf16",
    operand_bits=16,
    groups=(8, 8),
    width=25,
    zero_exponent=-132,
    validated="Hawkeye (A100), not re-measured here",
)

HOPPER_BF16_M16N8K16 = GroupSum(
    name="hopper_bf16_m16n8k16",
    arch="sm_90",
    dtype="bf16",
    operand_bits=16,
    groups=(16,),
    width=26,
    zero_exponent=-133,
    validated="Hawkeye (H100); re-measured on the mma.sync path by lane tc-probe-fp8 run r20260922-185339-e337 (25,698,304 elements, 0 mismatches) and on the RTX 5090 by lane fp4-5090 run r20260922-182911-1f10 (204,800/204,800)",
)

#: One accumulator coordinate of Hopper's ``wgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16`` (SASS ``HGMMA.64xNx16.F32.BF16``):
#: HOPPER_BF16_M16N8K16's parameters under the wgmma instruction's own name, so that the two instructions stay separate registry
#: entries.  The value does not depend on N, on the coordinate's position in the tile, on whether A comes from shared memory or
#: registers, or on ``scale-d`` (``0`` behaves as ``acc = +0``), so one k16 step is the primitive.
HOPPER_BF16_WGMMA_K16 = GroupSum(
    name="hopper_bf16_wgmma_k16",
    arch="sm_90",
    dtype="bf16",
    operand_bits=16,
    groups=(16,),
    width=26,
    zero_exponent=-133,
    validated=(
        "lane HP campaign, H100 80GB HBM3 (sm_90), 2026-09-07: 37,510 tiles x 8,192 outputs x 3 reps (wgmma m64n128k16 via Triton); "
        "lane wgmma-bf16, H100 80GB HBM3, 2026-09-26: wgmma m64n8k16 with A from shared memory (run r20260926-020157-242a) and from "
        "registers (r20260926-020210-aee8), 26,361,344 + 2,876,928 chained elements each and the lane HP records replayed, 0 mismatches"
    ),
)

HOPPER_E4M3_K32 = GroupSum(
    name="hopper_e4m3_wgmma_k32",
    arch="sm_90",
    dtype="e4m3",
    operand_bits=8,
    groups=(32,),
    width=14,
    zero_exponent=-139,
    validated="Hawkeye (H100, zero accumulator only), not re-measured here",
)

# The synthetic contract of the openvm-tc-bench spike: Ampere's BF16 GroupSum
# applied to E4M3 products, two groups of eight.  Kept so that the golden
# vectors of that spike remain checkable; it is not a hardware pipeline.
HAWKEYE_AMPERE_GROUPSUM_E4M3_V0 = GroupSum(
    name="hawkeye_ampere_groupsum_fp8e4m3_v0",
    arch="synthetic",
    dtype="e4m3",
    operand_bits=8,
    groups=(8, 8),
    width=25,
    zero_exponent=-132,
    validated="synthetic; the measured Ada E4M3 model is ADA_E4M3_M16N8K32",
)

# Ada Lovelace (RTX 4090, sm_89) as measured in this repository with one
# ``mma.sync`` per record; parameters recovered by
# gpu/tensor-core-semantics/characterize.py and validated by validate_tiles.py
# (results under gpu/tensor-core-semantics/results/).
ADA_BF16_M16N8K16 = GroupSum(
    name="ada_bf16_m16n8k16",
    arch="sm_89",
    dtype="bf16",
    operand_bits=16,
    groups=(8, 8),
    width=25,
    zero_exponent=-132,
    validated=(
        "RTX 4090 (sm_89, driver 580.159.03) mma.sync.m16n8k16.bf16: "
        "24,898,304/24,898,304 elements bit-exact over 194,518 tiles "
        "(random, subnormal, cancellation, mixed-magnitude, near-overflow, "
        "nonzero accumulators); identical parameters to Hawkeye's Ampere model"
    ),
)

ADA_E4M3_M16N8K32 = GroupSum(
    name="ada_e4m3_m16n8k32",
    arch="sm_89",
    dtype="e4m3",
    operand_bits=8,
    groups=(16, 16),
    width=14,
    zero_exponent=-139,
    validated=(
        "RTX 4090 (sm_89, driver 580.159.03) mma.sync.m16n8k32.e4m3: "
        "24,898,304/24,898,304 elements bit-exact over 194,518 tiles; "
        "Hopper's 14-bit FP8 adder but two groups of 16 instead of one of 32"
    ),
)

# --- E5M2 hypotheses (no silicon measured) -------------------------------------------------------------------------
# The FP8 adder does not know which FP8 format fed the products (the product of two E5M2 operands is a 6-bit
# integer at a wider exponent range); the hypothesis is therefore "the E4M3 pipeline with E5M2 products".  The
# probe lane must confirm groups, width and floor before either is called validated.

ADA_E5M2_M16N8K32 = GroupSum(
    name="ada_e5m2_m16n8k32",
    arch="sm_89",
    dtype="e5m2",
    operand_bits=8,
    groups=(16, 16),
    width=14,
    zero_exponent=-139,
    validated="RTX 4090 (sm_89), lane tc-probe-fp8 runs r20260922-175732-172a / r20260922-182641-e27d: 25,040,128 + 141,824 elements, 0 mismatches (ADA_E4M3_M16N8K32's pipeline with e5m2 products)",
)

HOPPER_E5M2_K32 = GroupSum(
    name="hopper_e5m2_wgmma_k32",
    arch="sm_90",
    dtype="e5m2",
    operand_bits=8,
    groups=(32,),
    width=14,
    zero_exponent=-139,
    validated="H100 (sm_90a), lane tc-probe-fp8 run r20260922-183837-940b: wgmma m64n8k32 e5m2, 25,554,432 elements, 0 mismatches (HOPPER_E4M3_K32's pipeline with e5m2 products)",
)

# Blackwell consumer (sm_120) FP8 mma.sync (SASS QMMA.16832): one group of all 32 products plus the accumulator in the 26-bit adder of
# its BF16 pipeline, so unlike Ada's and Hopper's 14-bit FP8 adders the accumulator is not truncated.  No FP8 product is below 2^-32,
# so the floor is unobservable anywhere in [-150, -124]; -133 is the BF16 pipeline's.
BLACKWELL_SM120_E4M3_M16N8K32 = GroupSum(
    name="blackwell_sm120_e4m3_m16n8k32",
    arch="sm_120",
    dtype="e4m3",
    operand_bits=8,
    groups=(32,),
    width=26,
    zero_exponent=-133,
    validated=(
        "RTX PRO 6000 Blackwell Server Edition (sm_120, driver 595.91.07, nvcc 12.8), lane vllm-sm120-tc-gemm: tc_probe run "
        "r20260930-030915-23f3 fitted by r20260930-031244-ecbc, 6,434,816 finite elements in 13 families, 0 mismatches; Ada's (16,16)/14, "
        "Hopper wgmma's (32,)/14 and the sm_90 two-HMMA lowering are refuted (4.91M / 4.81M / 0.76M mismatches); the fresh-seed capture "
        "r20260930-041757-0bde (25,040,128 elements, 14 families) evaluated under total semantics by r20260930-052000-e32d: 0 mismatches, "
        "the specials family included; the fresh-seed sweep r20260930-063457-5410 (driver 580.173.02, CUDA 13.0, SM clock locked at "
        "2,100 MHz, seed 20260931): 25,041,024 elements in 14 families, 0 mismatches (Ada's pipeline: 19,033,876)"
    ),
)

BLACKWELL_SM120_E5M2_M16N8K32 = GroupSum(
    name="blackwell_sm120_e5m2_m16n8k32",
    arch="sm_120",
    dtype="e5m2",
    operand_bits=8,
    groups=(32,),
    width=26,
    zero_exponent=-133,
    validated=(
        "RTX PRO 6000 Blackwell Server Edition (sm_120), lane vllm-sm120-tc-gemm: tc_probe and fit run r20260930-031447-a5ad, "
        "6,434,816 finite elements in 13 families, 0 mismatches (BLACKWELL_SM120_E4M3_M16N8K32's pipeline with e5m2 products); the "
        "fresh-seed sweep r20260930-063527-4d6b (driver 580.173.02, CUDA 13.0, SM clock locked at 2,100 MHz, seed 20260931): "
        "25,041,024 elements in 14 families, 0 mismatches (Ada's pipeline with e5m2 products: 15,747,373)"
    ),
)

PIPELINES: dict[str, GroupSum] = {
    pipeline.name: pipeline
    for pipeline in (
        AMPERE_BF16_M16N8K16,
        HOPPER_BF16_M16N8K16,
        HOPPER_E4M3_K32,
        HAWKEYE_AMPERE_GROUPSUM_E4M3_V0,
        ADA_BF16_M16N8K16,
        ADA_E4M3_M16N8K32,
    )
}


def pipeline_for(arch: str, dtype: str) -> GroupSum:
    """The unique validated pipeline of ``arch``/``dtype``."""

    matches = [p for p in PIPELINES.values() if p.arch == arch and p.dtype == dtype]
    if len(matches) != 1:
        raise InvalidArtifact(f"no unique pipeline for arch={arch!r} dtype={dtype!r}")
    return matches[0]


# =====================================================================================================================
# BlockScaledAlignAdd: Blackwell's block-scaled FP4 path as measured on the RTX 5090 (supersedes autoproof's model5)
# =====================================================================================================================

_E2M1_NUMERATOR = (0, 1, 2, 3, 4, 6, 8, 12)   # |value| * 2 for the eight magnitudes: {0, .5, 1, 1.5, 2, 3, 4, 6}
_FP32_SUBNORMAL_EXPONENT = -149              # weight of the LSB of an FP32 subnormal significand


def _e2m1_numerator(code: int) -> int:
    """The E2M1 nibble as an integer number of halves (``value = n / 2``); ``spec/formats.rs::numerator``."""

    if not 0 <= code < 16:
        raise InvalidArtifact("E2M1 operand outside 4 bits")
    n = _E2M1_NUMERATOR[code & 7]
    return -n if code & 8 else n


def _scale_dyadic(scale: str, code: int) -> tuple[int, int]:
    """One block-scale byte as an exact ``(mant, exp)``; invalid / NaN codes raise (``formats.rs::ScaleFormat``)."""

    if not 0 <= code < 256:
        raise InvalidArtifact("scale code outside 8 bits")
    if scale == "ue8m0":
        if code == 0xFF:
            raise InvalidArtifact("UE8M0 NaN scale")
        return 8, code - 130                        # 2^(code-127) written with a 4-bit mantissa like UE4M3's
    if scale == "ue4m3":
        # sm_120 reads a UE4M3 scale from its low 7 bits and ignores bit 7 (all 256 bytes, r20260930-093120-b2f2; the
        # 17 bit-7 vectors of the Lean scale-decode review replayed on the RTX PRO 6000, r20260930-161935-83a8), so
        # 0x80 is a zero scale like 0x00 and 0xFF is NaN like 0x7F
        code &= 0x7F
        e, m = code >> 3, code & 7
        if e == 15 and m == 7:
            raise InvalidArtifact("UE4M3 NaN scale")
        return (m, -9) if e == 0 else (8 + m, e - 10)
    raise InvalidArtifact(f"unknown scale format {scale!r}")


def _fp32_dyadic(word: int) -> tuple[int, int]:
    """A finite FP32 word as ``(mant, exp)`` (``dyadic.rs::from_f32``; both zeros are the zero dyadic)."""

    e = (word >> 23) & 0xFF
    if e == 0xFF:
        raise InvalidArtifact("non-finite FP32 accumulator")
    frac = word & 0x7F_FFFF
    m, ex = (frac, _FP32_SUBNORMAL_EXPONENT) if e == 0 else ((1 << 23) | frac, e - 150)
    return (-m if word >> 31 else m), ex


def _trunc_rz_to_grid(mant: int, exp: int, lsb: int) -> int:
    """``floor toward zero`` of ``mant * 2^exp`` onto the grid ``2^lsb``, returned as an integer multiple of ``2^lsb``."""

    if exp >= lsb:
        return mant << (exp - lsb)
    q = abs(mant) >> (lsb - exp)
    return -q if mant < 0 else q


@dataclass(frozen=True, slots=True)
class BlockScaledAlignAdd:
    """One ``mma.sync kind::mxf4nvf4`` firing on Blackwell sm_120 as measured on the RTX 5090.

    Per instruction (``k`` elements, ``k / group`` groups, one scale byte per ``block`` elements):

    1. every group's ``group`` E2M1 x E2M1 products are summed **exactly** as integers (quarters);
    2. the group's two block scales (UE4M3 for nvfp4, UE8M0 for mxfp4) multiply the group dot **losslessly**: the
       scaled dot is the integer ``gd * ma * mb`` at the *unit weight* ``2^X``, ``X = -2 + ea + eb``, where ``(ma, ea)``
       is the scale's 4-bit mantissa and exponent (UE4M3: ``8+m, e-10`` or ``m, -9`` for the subnormal field; UE8M0:
       ``8, code-130``);
    3. one five-way align-add of the four scaled dots and the carried FP32 accumulator, each truncated **toward
       zero** onto the common grid ``2^lsb``, summed exactly and normalised **RZ** to binary32, where::

           lsb = max( max{X_g : g participates} - frac_bits,      # the scale-exponent anchor
                      acc != 0 ? E_acc - (acc_window - 1) : -inf,   # the accumulator window
                      grid_floor )                                  # the adder's absolute floor

       ``E_acc`` is the accumulator's *exponent field* (``e - 127``; a subnormal accumulator counts as ``-126``, not
       its true msb).  A group *participates* in the anchor when at least one of its elementwise products is nonzero
       (both nibbles nonzero) **and** both its scale mantissas are nonzero -- whether or not its dot cancels to zero.
       With ``frac_bits = 27`` the adder keeps 27 bits below the unit weight of the leading participating group; with
       ``acc_window = 36`` the accumulator keeps 12 guard bits below its 24-bit significand; ``grid_floor = -174``
       is the deepest grid the adder realises (reachable with UE8M0 scales only: it binds when the leading unit weight
       is below ``2^-147``; NVFP4 unit weights are >= ``2^-20``).

    Every zero result is ``+0``: exact cancellation, every addend truncated away, all-zero products into a ``-0``
    accumulator (log entry D16).  A sum that leaves the binary32 range is *outside* the model and raises.
    ``underflow_zero`` fixes the sign of a **nonzero** sum lying entirely below the FP32 subnormal grid (reachable
    with UE8M0 scales only; log entry D13: the RTX 5090 gives ``+0``).

    Relation to autoproof's *model5* (``SEMANTICS.lock.v1`` ``gdfs_model``, log entry D12): model5 anchored the
    window on the msb of the largest *addend* (``lsb = E1 - 41``) and added a "hot group clip" at ``2^-19`` when a
    live group had both UE4M3 exponent fields at 15.  Both are what the rule above produces when the leading
    group's mantissa is 15 bits wide (the clip: ``X = 8`` for exponent fields 15/15, ``8 - 27 = -19``); when it is
    not, model5 mis-predicts opposite-sign accumulators 37..42 binades below the leading exponent by 1 ulp (D17,
    12,790 / 1.5M of lane fp4-5090's cases), the rule above does not (fp4-re: 0 / 1,736,704 nvfp4 and
    0 / 622,592 mxfp4 elements of the same captures).  The exponent-field accumulator anchor and ``grid_floor``
    were fitted by lane fp4-re on the subnormal-result regime of UE8M0 (run r20260922-215831-c3d8: 528 residual
    mismatches under the true-msb anchor without a floor -> 0; the floor's minimum is sharp, -173 and -175 each
    leave > 20; log entry D19).

    ``step`` takes ``a_words = a_codes[0..k] + a_scales[0..k/block]`` (likewise ``b``); :meth:`step_scaled`
    takes them apart.  ``dyadic.rs``-style helpers (``_trunc_rz_to_grid``, ``_to_f32_rz``) are autoproof's.
    """

    name: str
    arch: str
    dtype: str
    operand_bits: int
    scale: Literal["ue4m3", "ue8m0"] | None
    k_per_instruction: int
    group: int
    frac_bits: int
    acc_window: int
    grid_floor: int
    underflow_zero: Literal["signed", "positive"]
    validated: str

    def __post_init__(self) -> None:
        if self.dtype != "e2m1":
            raise InvalidArtifact("BlockScaledAlignAdd models E2M1 elements only")
        if self.k_per_instruction % self.group:
            raise InvalidArtifact("k must be a multiple of the group size")
        if self.scale is not None and self.k_per_instruction % self.block:
            raise InvalidArtifact("k must be a multiple of the scale block")
        if self.frac_bits < 0 or self.acc_window < 1:
            raise InvalidArtifact("frac_bits must be >= 0 and acc_window >= 1")
        if self.grid_floor > _FP32_SUBNORMAL_EXPONENT:
            raise InvalidArtifact("grid_floor above the FP32 subnormal grid would truncate the accumulator itself")

    @property
    def k(self) -> int:
        return self.k_per_instruction

    @property
    def block(self) -> int:
        """Elements per scale byte: 16 (UE4M3, nvfp4), 32 (UE8M0, mxfp4), the whole instruction when unscaled."""

        return {"ue4m3": 16, "ue8m0": 32}.get(self.scale, self.k_per_instruction)

    @property
    def scales_per_step(self) -> int:
        return self.k_per_instruction // self.block if self.scale else 0

    @property
    def words_per_step(self) -> int:
        return self.k_per_instruction + self.scales_per_step

    def product(self, a: int, b: int) -> Term:
        return e2m1_product(a, b)

    def step(self, c_word: int, a_words: Sequence[int], b_words: Sequence[int]) -> int:
        n = self.words_per_step
        if len(a_words) != n or len(b_words) != n:
            raise InvalidArtifact(f"{self.name} takes {self.k} element codes + {self.scales_per_step} scale bytes per operand")
        k = self.k
        return self.step_scaled(c_word, a_words[:k], b_words[:k], a_words[k:], b_words[k:])

    def group_terms(
        self,
        a_codes: Sequence[int],
        b_codes: Sequence[int],
        a_scales: Sequence[int] = (),
        b_scales: Sequence[int] = (),
    ) -> list[tuple[int, int, bool]]:
        """The scaled group dots of one firing as ``(mant, X, participates)``; see the class docstring."""

        k, ns = self.k, self.scales_per_step
        if len(a_codes) != k or len(b_codes) != k:
            raise InvalidArtifact(f"{self.name} takes {k} element codes per operand")
        if len(a_scales) != ns or len(b_scales) != ns:
            raise InvalidArtifact(f"{self.name} takes {ns} scale bytes per operand")
        terms: list[tuple[int, int, bool]] = []
        for g in range(0, k, self.group):
            gd = 0
            any_product = False
            for i in range(g, g + self.group):
                na, nb = _e2m1_numerator(a_codes[i]), _e2m1_numerator(b_codes[i])
                any_product |= na != 0 and nb != 0
                gd += na * nb
            mant, unit = gd, -2                     # halves x halves = quarters
            participates = any_product
            if self.scale:
                (ma, ea), (mb, eb) = (
                    _scale_dyadic(self.scale, a_scales[g // self.block]),
                    _scale_dyadic(self.scale, b_scales[g // self.block]),
                )
                mant, unit = mant * ma * mb, unit + ea + eb
                participates = participates and ma != 0 and mb != 0
            terms.append((mant, unit, participates))
        return terms

    def step_scaled(
        self,
        c_word: int,
        a_codes: Sequence[int],
        b_codes: Sequence[int],
        a_scales: Sequence[int] = (),
        b_scales: Sequence[int] = (),
    ) -> int:
        """One firing on element codes and scale bytes given separately; see the class docstring."""

        return self._transducer(c_word, self.group_terms(a_codes, b_codes, a_scales, b_scales))

    def align_add(self, c_word: int, terms: Sequence[tuple[int, int, bool]]) -> int:
        """One firing on its `group_terms`: ``step_scaled(c, a, b, sa, sb) == align_add(c, group_terms(a, b, sa, sb))``."""

        return self._transducer(c_word, terms)

    def grid(self, c_word: int, terms: Sequence[tuple[int, int, bool]]) -> int | None:
        """The exponent ``lsb`` of one firing's common grid ``2^lsb`` (the class docstring), or None when nothing
        participates and the accumulator is zero (the firing is then ``+0``)."""

        anchors = [unit - self.frac_bits for _, unit, participates in terms if participates]
        if _fp32_dyadic(c_word)[0]:
            e_field = (c_word >> 23) & 0xFF                 # the exponent field, not the true msb: subnormals anchor at -126
            anchors.append((e_field if e_field else 1) - 127 - (self.acc_window - 1))
        return max(*anchors, self.grid_floor) if anchors else None

    def _transducer(self, c_word: int, terms: Sequence[tuple[int, int, bool]]) -> int:
        """The five-way align-add on the common grid and the RZ normalisation."""

        acc_m, acc_e = _fp32_dyadic(c_word)
        lsb = self.grid(c_word, terms)
        if lsb is None:
            return 0                                        # nothing participates: +0 (a -0 accumulator included)
        total = sum(_trunc_rz_to_grid(m, unit, lsb) for m, unit, _ in terms if m)
        if acc_m:
            total += _trunc_rz_to_grid(acc_m, acc_e, lsb)
        if total == 0:
            return 0                                        # computed zero (cancellation included) is +0
        return self._to_f32_rz(total, lsb)

    def _to_f32_rz(self, n: int, lsb: int) -> int:
        """``dyadic.rs::to_f32(Round::Rz)`` of the nonzero ``n * 2^lsb``: normals, gradual underflow, no carry."""

        negative = n < 0
        mag = -n if negative else n
        e_val = mag.bit_length() - 1 + lsb              # floor(log2 |v|)
        grid = e_val - 23 if e_val >= FP32_MIN_NORMAL_EXPONENT else _FP32_SUBNORMAL_EXPONENT
        shift = grid - lsb
        q = mag >> shift if shift > 0 else mag << -shift
        if q == 0:
            # Entire magnitude below the subnormal grid (UE8M0 scales only); autoproof's code keeps the sign.
            return 0x8000_0000 if negative and self.underflow_zero == "signed" else 0
        e = q.bit_length() - 1 + grid
        if e > FP32_MAX_EXPONENT:
            raise InvalidArtifact("BlockScaledAlignAdd: sum leaves the binary32 range (not modelled; autoproof asserts)")
        bits = (((e + 127) << 23) | (q & 0x7F_FFFF)) if e >= FP32_MIN_NORMAL_EXPONENT else q
        return bits | (0x8000_0000 if negative else 0)

    def chain(
        self,
        c_word: int,
        a_codes: Sequence[int],
        b_codes: Sequence[int],
        a_scales: Sequence[int] = (),
        b_scales: Sequence[int] = (),
    ) -> int:
        """Ascending-K chain of firings over ``K`` codes (a multiple of ``k``) and their ``K / block`` scale bytes."""

        k, ns = self.k, self.scales_per_step
        if len(a_codes) != len(b_codes) or len(a_codes) % k:
            raise InvalidArtifact(f"chain length must be a multiple of {k}")
        steps = len(a_codes) // k
        if len(a_scales) != steps * ns or len(b_scales) != steps * ns:
            raise InvalidArtifact(f"{steps * ns} scale bytes expected per operand")
        for s in range(steps):
            c_word = self.step_scaled(
                c_word,
                a_codes[s * k : (s + 1) * k],
                b_codes[s * k : (s + 1) * k],
                a_scales[s * ns : (s + 1) * ns],
                b_scales[s * ns : (s + 1) * ns],
            )
        return c_word


_AUTOPROOF_PIN = (
    "autoproof SEMANTICS.lock.v1 (ADOPTED 2026-07-11, oracle-v3): RTX 5090 (sm_120, driver 570.195.03, GPU "
    "GPU-297048d1), kernel SASS 166a24a0..., mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3, ascending-K chain from a zero accumulator: the lockfile records 1,574,998 / "
    "1,574,998 dot statements bit-exact under model5 (K in {64, 128, 256}; random, extreme-scale-span, "
    "directed-cancellation, curated) and the v0 single-window W=26 hypothesis REFUTED (142,270 mismatches); the "
    "checked-in probes/results/probe3_1783788526.json scores the clip-less model2 at 1,574,997 / 1,574,998 and v0 at "
    "143,076 mismatches, and carries the 1,601 sample vectors verity replays (1,601 / 1,601) -- see log entry D12"
)

_FP4_RE_PIN = (
    "lane fp4-re 2026-09-22, RTX 5090 (sm_120, cc 12.0, driver 580.178.04, nvcc 12.8.93 -arch=sm_120a, GPU-506928f9), "
    "one mma.sync per output element, arbitrary finite FP32 accumulators (families: random operands and scales, uniform "
    "encodings, scale sweeps, hot-clip regime, large cancellation, accumulator window walks, tiny / subnormal / -0 "
    "accumulators, signed-zero matrix, accumulator-offset walk 24..56 binades, wide-mantissa top scales, anchor "
    "participation)"
)
_FP4_RE_NVF4 = (
    "SASS OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X; 0 mismatches over 8,339,456 elements: lane fp4-5090's captures "
    "r20260922-182011-88f8 (1,736,704; model5 29,118), fit run r20260922-215616-3c10 (2,252,800; model5 88,697) and the "
    "fresh-seed confirmation run r20260922-221716-65b9 (4,349,952; model5 114,148); autoproof probe3 vectors replayed "
    "1,601 / 1,601 in each run"
)
_FP4_RE_MXF4 = (
    "SASS OMMA.SF.16864.F32.E2M1.E2M1.E8, PTX mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue8m0; 0 mismatches over 7,389,184 elements: lane fp4-5090's captures "
    "r20260922-182548-e654 (622,592; model5+ue8m0-shift 19,709), fit run r20260922-215831-c3d8 (2,039,808; the "
    "true-msb accumulator anchor without a floor left 528, log entry D19), fresh-seed confirmation runs "
    "r20260922-221824-8d99 (3,940,352) and r20260922-222148-6342 (786,432; ue8m0 scale extremes 2^-127..2^127, "
    "sums below the subnormal grid, subnormal accumulators); the signed-underflow alternative misses 14,734 + 14,472 "
    "of the confirmation elements (D13)"
)
_FP4_RE_EDGES = (
    "outside the model (record-only family `specials`): a NaN accumulator of either sign and any payload returns "
    "0x7FFFFFFF, +-inf returns itself, a sum leaving binary32 returns +-inf of its sign (ue8m0 only; nvf4 cannot "
    "overflow); the same words observed on the B200 (sm_100 tcgen05, lane tcgen05: record-only family `specials` of every run)"
)

BLACKWELL_SM120_NVF4 = BlockScaledAlignAdd(
    name="blackwell_sm120_nvf4_m16n8k64",
    arch="sm_120",
    dtype="e2m1",
    operand_bits=4,
    scale="ue4m3",
    k_per_instruction=64,
    group=16,
    frac_bits=27,
    acc_window=36,
    grid_floor=-174,
    underflow_zero="positive",
    validated=(
        "RTX 5090 (sm_120) mma.sync kind::mxf4nvf4 ue4m3: " + _FP4_RE_PIN + "; " + _FP4_RE_NVF4 + "; " + _FP4_RE_EDGES
        + ". Zero-accumulator chains also pinned by " + _AUTOPROOF_PIN
    ),
)

# mxfp4 (UE8M0 scales over blocks of 32): the same adder, the same constants -- the UE8M0 scale is 2^(code-127)
# written as the 4-bit mantissa 8 at 2^(code-130), so the anchor follows the raw exponent fields with a 6-bit offset
# (lane fp4-5090 fitted 33 in raw-exponent units; that is 27 here).  Log entry D18.  ``underflow_zero`` (D13):
# the sign of a nonzero sum lying entirely below the FP32 subnormal grid.
BLACKWELL_SM120_MXF4 = BlockScaledAlignAdd(
    name="blackwell_sm120_mxf4_m16n8k64",
    arch="sm_120",
    dtype="e2m1",
    operand_bits=4,
    scale="ue8m0",
    k_per_instruction=64,
    group=16,
    frac_bits=27,
    acc_window=36,
    grid_floor=-174,
    underflow_zero="positive",
    validated="RTX 5090 (sm_120) mma.sync kind::mxf4nvf4 ue8m0: " + _FP4_RE_PIN + "; " + _FP4_RE_MXF4 + "; " + _FP4_RE_EDGES,
)

BLACKWELL_SM120_MXF4_UNDERFLOW_SIGNED = BlockScaledAlignAdd(
    name="blackwell_sm120_mxf4_m16n8k64_underflow_signed",
    arch="sm_120",
    dtype="e2m1",
    operand_bits=4,
    scale="ue8m0",
    k_per_instruction=64,
    group=16,
    frac_bits=27,
    acc_window=36,
    grid_floor=-174,
    underflow_zero="signed",
    validated=(
        "REFUTED on the RTX 5090 (lane fp4-re): as BLACKWELL_SM120_MXF4 but a nonzero sum below the subnormal grid keeps "
        "its sign, as autoproof dyadic.rs::to_f32 does; 17,470 / 2,039,808 (run r20260922-215831-c3d8), 14,734 / "
        "3,940,352 (r20260922-221824-8d99) and 14,472 / 786,432 (r20260922-222148-6342) mismatches, all of them -0 where "
        "the hardware gives +0 (log entry D13). Kept as the named alternative the probe scores alongside the pin."
    ),
)


# Blackwell datacenter (sm_100, B200): ``tcgen05.mma`` is a different instruction (tensor-memory accumulators, M = 128 minimum,
# issued by one thread, completion through an mbarrier) and lane fp4-re refused to assume it computes the sm_120 words.  Lane
# tcgen05 measured it: the sm_120 constants, unchanged, reproduce every word.  These are separate model objects so the registry
# names the architecture it was measured on; their parameters are BLACKWELL_SM120_*'s to the bit.
_TCGEN05_PIN = (
    "lane tcgen05 2026-09-23, NVIDIA B200 (sm_100, cc 10.0, driver 580.126.09, nvcc 12.9.86 -gencode compute_100a/sm_100a, "
    "GPU-d1d0c3f4), CUTLASS-free probe kernel benchmarks/tc_probe_fp4/tcgen05_fp4.cu: ONE tcgen05.mma.cta_group::1 (M128 N8 K64, "
    "SASS UTCOMMA) per tile with the 16 harness rows replicated into all 8 groups of 16 TMEM lanes (every replica agreed), "
    "arbitrary FP32 accumulators preloaded into TMEM with tcgen05.st and enable_input_d = 1 (all-+0 accumulators also run with "
    "enable_input_d = 0 and through the TMEM path: identical), K-major no-swizzle smem descriptors, scale factors in TMEM; the "
    "same 20 families as the sm_120 pin plus a K=128 chain (two MMAs on one TMEM accumulator)"
)
BLACKWELL_SM100_NVF4 = BlockScaledAlignAdd(
    name="blackwell_sm100_tcgen05_nvf4",
    arch="sm_100",
    dtype="e2m1",
    operand_bits=4,
    scale="ue4m3",
    k_per_instruction=64,
    group=16,
    frac_bits=27,
    acc_window=36,
    grid_floor=-174,
    underflow_zero="positive",
    validated=(
        "B200 (sm_100) tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X, SASS UTCOMMA.4X: " + _TCGEN05_PIN
        + "; 0 mismatches over 16,461,824 elements: r20260923-001528-f5c5 (303,104), r20260923-001759-d869 (6,463,488; autoproof "
        "probe3 vectors replayed 1,601 / 1,601 through the TMEM accumulator, K=128/256 cases as dependent MMAs), "
        "r20260923-001856-7d63 (1,134,592; K=128 chain), r20260923-002315-b793 (8,560,640); model5 (autoproof's control) "
        "misses 140,102 / 165,355 of the two big runs exactly as on the RTX 5090. The parameters are BLACKWELL_SM120_NVF4's."
    ),
)
BLACKWELL_SM100_MXF4 = BlockScaledAlignAdd(
    name="blackwell_sm100_tcgen05_mxf4",
    arch="sm_100",
    dtype="e2m1",
    operand_bits=4,
    scale="ue8m0",
    k_per_instruction=64,
    group=16,
    frac_bits=27,
    acc_window=36,
    grid_floor=-174,
    underflow_zero="positive",
    validated=(
        "B200 (sm_100) tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X (the mnemonic CUTLASS 3.9.2's "
        "SM100_MMA_MXF4_SS emits for UE8M0) and kind::mxf4nvf4.block_scale.scale_vec::2X, both SASS UTCOMMA: " + _TCGEN05_PIN
        + "; 0 mismatches over 12,898,304 elements: kind::mxf4 r20260923-001838-42a4 (3,956,736) and r20260923-002320-6fbf "
        "(3,956,736); kind::mxf4nvf4 2X r20260923-001832-17dc (3,956,736) and r20260923-001903-c420 (1,028,096; K=128 chain); "
        "the signed-underflow alternative misses 14,710 / 14,384 / 14,734 / 5,535 (D13 holds on sm_100 too). The parameters "
        "are BLACKWELL_SM120_MXF4's."
    ),
)

# The K=32 E2M1 row of Blackwell consumer (sm_120), ``kind::f8f6f4`` (SASS QMMA.16832): Hopper's BF16 adder (26 bits, floor
# 2^-133) over one group of all 32 exact E2M1 products, not the block-scaled adder of the K=64 rows (OMMA) -- the instruction
# fixes the arithmetic, not the element format.  autoproof's 1,400 RTX 5090 vectors of this instruction (zero accumulators)
# do not tell the two apart; the RTX PRO 6000 capture does (verity/ml/circuits/ops/tests/test_models.py).  Also measured with 0 mismatches, and
# outside this object because a GroupSum has one dtype and no scales: e2m1 x e4m3 and e4m3 x e2m1, and kind::mxf8f6f4
# .scale_vec::1X (QMMA.SF, UE8M0 entering as an exponent shift of each product).
BLACKWELL_SM120_E2M1_M16N8K32 = GroupSum(
    name="blackwell_sm120_e2m1_m16n8k32",
    arch="sm_120",
    dtype="e2m1",
    operand_bits=4,
    groups=(32,),
    width=26,
    zero_exponent=-133,
    validated=(
        "NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120a, cc 12.0, driver 580.173.02, nvcc 13.0.88) "
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32, one SASS QMMA.16832.F32.E2M1.E2M1 per tile, "
        "benchmarks/tc_probe_fp4 (k32.py) sweeps with arbitrary finite FP32 accumulators: run r20260930-071201-d356 "
        "(GPU-5f1149a4, seed 20260930) 1,671,168 gated elements, 0 mismatches, while two groups of 16 miss 88,660, the "
        "unscaled block-scaled adder (group 16 / 32) 193,027 / 192,566 and autoproof's model5 268,267; fresh-seed "
        "confirmation r20260930-073334-fdac (seed 20261003, gated on this model) 3,309,568 gated elements, 0 mismatches "
        "(the alternatives miss 176,662 / 385,736 / 384,745), both on GPU-5f1149a4; a 1024-step register chain (K = 32,768) "
        "on a second die, r20260930-073445-e2b6 (GPU-2b59d5fe, seed 20261004, gated on this model), 1,408 end words, "
        "0 mismatches. "
        "NaN/inf accumulators are outside the model (record-only). Capture sample: verity/ml/circuits/ops/tests/fixtures/tc-sm120-k32-e2m1-2026-09-30"
    ),
)

#: Every model instance by name -- the grouped-sum pipelines and the block-scaled hypotheses alike.
MODELS: dict[str, Model] = {
    m.name: m
    for m in (
        *PIPELINES.values(),
        HOPPER_BF16_WGMMA_K16,
        ADA_E5M2_M16N8K32,
        HOPPER_E5M2_K32,
        BLACKWELL_SM120_E4M3_M16N8K32,
        BLACKWELL_SM120_E5M2_M16N8K32,
        BLACKWELL_SM120_NVF4,
        BLACKWELL_SM120_MXF4,
        BLACKWELL_SM120_MXF4_UNDERFLOW_SIGNED,
        BLACKWELL_SM100_NVF4,
        BLACKWELL_SM100_MXF4,
        BLACKWELL_SM120_E2M1_M16N8K32,
        BLACKWELL_SM120_S8_M16N8K32,
        BLACKWELL_SM120_S8_M16N8K32_WRAP,
    )
}
