"""Models: how a tensor-core instruction folds exact products into its FP32 accumulator.

A *model* is the executable, bit-exact semantics of one accumulator coordinate of one instruction class:
``step(c_word, a_words, b_words) -> d_word`` on raw encodings.  Two model shapes exist so far:

* :class:`GroupSum` -- Hawkeye's grouped, truncating accumulation (Ampere / Ada / Hopper, BF16 and FP8): the K
  exact products are summed together with the incoming accumulator in fixed-size *groups*; every term of a group
  is rescaled to the adder's ``width``, aligned to the group's maximum exponent by a truncating right shift,
  summed exactly, and the sum is normalised back to ``width`` bits by truncation (round toward zero).  The result
  of one group is the accumulator of the next.  ``Pipeline`` is this class's historical name and stays an alias.
* :class:`BlockScaledAlignAdd` -- autoproof's *model5* of Blackwell's block-scaled FP4 path (``mma.sync
  kind::mxf4nvf4`` on sm_120): four *exact* 16-term group dots over E2M1, per-block scales applied losslessly,
  then one five-way align-add of the four scaled terms and the carried accumulator through a 42-bit window (a
  36-bit second window riding the accumulator's exponent, a "hot group clip" at ``2^-19``), normalised RZ.

The :class:`Model` protocol is what the rest of Verity may rely on; :data:`MODELS` lists every instance, and
``instructions.INSTRUCTIONS`` says which instruction each one is the model *of* and how well it is validated.

No IEEE rounding happens anywhere in either model; this is what NVIDIA's silicon does and it is *not* what a
sequence of FP32 FMAs would do.  GroupSum is the model recovered by Hawkeye (Badash, Boneh, Komargodski,
Srivastava, "Hawkeye: Reproducing GPU-Level Non-Determinism", MLSys 2026, arXiv 2603.20421); the body is a
bit-exact pure-Python port of https://github.com/badasherez/gpu-simulator (``src/Ampere_simulator.cpp``,
``src/Hopper_simulator.cpp``, ``src/Hopper_fp8_simulator.cpp``, ``utils/utils.cpp``, commit
``30703fcb309c943a6df5eee0277cb81815deb8f4``), generalised so that the group structure, internal width and
exponent floor are parameters, plus the saturation rule measured on Ada (a group leaving the FP32 range is a
sticky infinity).  BlockScaledAlignAdd is a port of autoproof ``fp4-ref/src/native.rs`` (``transducer`` /
``native_dot_f32``) and ``spec/src/dyadic.rs`` (``trunc_rz``, ``to_f32``) as adopted in
``spec/SEMANTICS.lock.v1.json`` (replica: ``fixtures/tc/autoproof-SEMANTICS.lock.v1.json``).

The ``GroupSum`` code moved here from ``silicon.py`` unchanged (Phase 0 of the silicon campaign); ``silicon.py``
re-exports every name, so ``silicon.tc_dot`` is still the executable definition the frozen target names.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .errors import InvalidArtifact
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
    "BLACKWELL_SM120_MXF4",
    "BLACKWELL_SM120_MXF4_UNDERFLOW_PLUS0",
    "BLACKWELL_SM120_NVF4",
    "BlockScaledAlignAdd",
    "GroupSum",
    "HAWKEYE_AMPERE_GROUPSUM_E4M3_V0",
    "HOPPER_BF16_M16N8K16",
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
    validated="Hawkeye (H100), not re-measured here",
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
    validated="synthetic; see docs/hardware-semantics.md for the Ada comparison",
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
    validated="HYPOTHESIS (unprobed): ADA_E4M3_M16N8K32's pipeline with e5m2 products",
)

HOPPER_E5M2_K32 = GroupSum(
    name="hopper_e5m2_wgmma_k32",
    arch="sm_90",
    dtype="e5m2",
    operand_bits=8,
    groups=(32,),
    width=14,
    zero_exponent=-139,
    validated="HYPOTHESIS (unprobed): HOPPER_E4M3_K32's pipeline with e5m2 products",
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
# BlockScaledAlignAdd: autoproof's model5 of Blackwell's block-scaled FP4 path
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
        return 1, code - 127
    if scale == "ue4m3":
        if code & 0x80:
            raise InvalidArtifact("UE4M3 scale with the padding bit set (PTX ISA: invalid)")
        e, m = (code >> 3) & 0xF, code & 7
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


def _ilog2(mant: int, exp: int) -> int:
    return abs(mant).bit_length() - 1 + exp


def _trunc_rz_to_grid(mant: int, exp: int, lsb: int) -> int:
    """``floor toward zero`` of ``mant * 2^exp`` onto the grid ``2^lsb``, returned as an integer multiple of ``2^lsb``."""

    if exp >= lsb:
        return mant << (exp - lsb)
    q = abs(mant) >> (lsb - exp)
    return -q if mant < 0 else q


@dataclass(frozen=True, slots=True)
class BlockScaledAlignAdd:
    """autoproof's *model5* (``SEMANTICS.lock.v1`` ``gdfs_model``) of one ``mma.sync kind::mxf4nvf4`` firing.

    Per instruction (``k`` elements, ``k / group`` groups, one scale byte per ``block`` elements):

    1. every group's ``group`` E2M1 x E2M1 products are summed **exactly** as integers (quarters);
    2. the group's two block scales (UE4M3 for nvfp4, UE8M0 for mxfp4) multiply the group dot **losslessly**;
    3. one five-way align-add: the (up to four) nonzero scaled group terms and the nonzero carried FP32 accumulator
       are each truncated **toward zero** at the common ``lsb = max(E1 - (w1 - 1), acc != 0 ? ilog2(acc) - (w2 - 1)
       : -inf, hot ? hot_clip_lsb : -inf)`` with ``E1`` the largest exponent among them, summed exactly and
       normalised **RZ** to binary32.  ``w1 = 42`` is the physical window below the largest addend; ``w2 = 36``
       rides the accumulator's exponent (12 guard bits below its 24-bit significand).
    4. the *hot group clip* (nvfp4 only -- its trigger reads the UE4M3 exponent field): if any group of the firing
       has a nonzero elementwise product AND both its scale codes are in ``0x78..0x7E`` (exponent field 15), every
       addend of the firing is additionally truncated toward zero at ``2^-19`` -- the ``2^-20`` bit, the deepest
       any NVFP4 term can carry, is dropped before the add.  Per firing; no leakage across the K chain.

    A firing whose addends are all zero leaves the accumulator word bit-identical (``-0`` included); a computed
    zero -- exact cancellation, or every addend truncated away -- is ``+0``.  A sum that leaves the binary32 range
    is *outside* the model (autoproof's oracle asserts; no silicon evidence) and raises.  ``underflow_zero`` names
    the one place autoproof's code and prose disagree (log entry D13): a nonzero sum entirely below the FP32
    subnormal grid becomes a zero of the sum's sign in ``dyadic.rs::to_f32`` (``"signed"``) although the lockfile
    says computed zeros are ``+0`` (``"positive"``); unreachable with UE4M3 scales, reachable with UE8M0.

    ``step`` takes ``a_words = a_codes[0..k] + a_scales[0..k/block]`` (likewise ``b``); :meth:`step_scaled`
    takes them apart.  Ported from ``fp4-ref/src/native.rs`` (``transducer``, ``native_dot_f32``) and
    ``spec/src/dyadic.rs``; cross-checked against the recorded RTX 5090 vectors in
    ``fixtures/tc/autoproof-probe-vectors.json`` (``tests/ml/test_models.py``).
    """

    name: str
    arch: str
    dtype: str
    operand_bits: int
    scale: Literal["ue4m3", "ue8m0"] | None
    k_per_instruction: int
    group: int
    w1: int
    w2: int
    hot_clip: bool
    hot_clip_lsb: int
    underflow_zero: Literal["signed", "positive"]
    validated: str

    def __post_init__(self) -> None:
        if self.dtype != "e2m1":
            raise InvalidArtifact("BlockScaledAlignAdd models E2M1 elements only")
        if self.k_per_instruction % self.group:
            raise InvalidArtifact("k must be a multiple of the group size")
        if self.scale is not None and self.k_per_instruction % self.block:
            raise InvalidArtifact("k must be a multiple of the scale block")
        if self.hot_clip and self.scale != "ue4m3":
            raise InvalidArtifact("the hot-group clip is defined by the UE4M3 exponent field")

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

    def step_scaled(
        self,
        c_word: int,
        a_codes: Sequence[int],
        b_codes: Sequence[int],
        a_scales: Sequence[int] = (),
        b_scales: Sequence[int] = (),
    ) -> int:
        """One firing on element codes and scale bytes given separately; see the class docstring."""

        k, ns = self.k, self.scales_per_step
        if len(a_codes) != k or len(b_codes) != k:
            raise InvalidArtifact(f"{self.name} takes {k} element codes per operand")
        if len(a_scales) != ns or len(b_scales) != ns:
            raise InvalidArtifact(f"{self.name} takes {ns} scale bytes per operand")
        terms: list[tuple[int, int]] = []
        hot = False
        for g in range(0, k, self.group):
            gd = 0
            any_product = False
            for i in range(g, g + self.group):
                na, nb = _e2m1_numerator(a_codes[i]), _e2m1_numerator(b_codes[i])
                any_product |= na != 0 and nb != 0
                gd += na * nb
            mant, exp = gd, -2                      # halves x halves = quarters
            if self.scale:
                sa_code, sb_code = a_scales[g // self.block], b_scales[g // self.block]
                (ma, ea), (mb, eb) = _scale_dyadic(self.scale, sa_code), _scale_dyadic(self.scale, sb_code)
                mant, exp = mant * ma * mb, exp + ea + eb
                if self.hot_clip and any_product and (sa_code & 0x78) == 0x78 and (sb_code & 0x78) == 0x78:
                    hot = True
            terms.append((mant, exp))
        return self._transducer(c_word, terms, hot)

    def _transducer(self, c_word: int, terms: Sequence[tuple[int, int]], hot: bool) -> int:
        """``native.rs::transducer``: the five-way align-add and the RZ normalisation."""

        acc_m, acc_e = _fp32_dyadic(c_word)
        addends = [t for t in terms if t[0] != 0]
        if acc_m:
            addends.append((acc_m, acc_e))
        if not addends:
            return c_word                                   # nothing to fold; -0 stays -0
        e1 = max(_ilog2(m, e) for m, e in addends)
        lsb = e1 - (self.w1 - 1)
        if acc_m:
            lsb = max(lsb, _ilog2(acc_m, acc_e) - (self.w2 - 1))
        if hot:
            lsb = max(lsb, self.hot_clip_lsb)
        total = sum(_trunc_rz_to_grid(m, e, lsb) for m, e in addends)
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
    ".row.col.f32.e2m1.e2m1.f32.ue4m3, ascending-K chain from a zero accumulator: 1,574,998 / 1,574,998 dot "
    "statements bit-exact (K in {64, 128, 256}; random, extreme-scale-span, directed-cancellation, curated); "
    "the v0 single-window W=26 hypothesis was REFUTED there (142,270 mismatches)"
)

BLACKWELL_SM120_NVF4 = BlockScaledAlignAdd(
    name="blackwell_sm120_nvf4_m16n8k64",
    arch="sm_120",
    dtype="e2m1",
    operand_bits=4,
    scale="ue4m3",
    k_per_instruction=64,
    group=16,
    w1=42,
    w2=36,
    hot_clip=True,
    hot_clip_lsb=-19,
    underflow_zero="signed",
    validated=(
        "autoproof: RTX 5090 (sm_120) mma.sync kind::mxf4nvf4, 1.57M dot statements pinned 2026-07-11 "
        "(SEMANTICS.lock.v1); NOT validated on sm_100 (B200 tcgen05). " + _AUTOPROOF_PIN
    ),
)

# mxfp4 (UE8M0 scales over blocks of 32) was NOT probed by autoproof; native.rs applies the same transducer with the
# hot clip off (its trigger is UE4M3-specific).  Two named hypotheses differ only where a nonzero sum lies entirely
# below the FP32 subnormal grid (reachable with UE8M0 scales down to 2^-127; unreachable in nvfp4): the sign of
# that zero.  Log entry D13.
BLACKWELL_SM120_MXF4 = BlockScaledAlignAdd(
    name="blackwell_sm120_mxf4_m16n8k64",
    arch="sm_120",
    dtype="e2m1",
    operand_bits=4,
    scale="ue8m0",
    k_per_instruction=64,
    group=16,
    w1=42,
    w2=36,
    hot_clip=False,
    hot_clip_lsb=-19,
    underflow_zero="signed",
    validated=(
        "HYPOTHESIS (unprobed): the nvfp4 five-way align-add (autoproof native.rs) with UE8M0 block-32 scales and "
        "no hot-group clip; underflow-to-zero keeps the sign as dyadic.rs::to_f32 does"
    ),
)

BLACKWELL_SM120_MXF4_UNDERFLOW_PLUS0 = BlockScaledAlignAdd(
    name="blackwell_sm120_mxf4_m16n8k64_underflow_plus0",
    arch="sm_120",
    dtype="e2m1",
    operand_bits=4,
    scale="ue8m0",
    k_per_instruction=64,
    group=16,
    w1=42,
    w2=36,
    hot_clip=False,
    hot_clip_lsb=-19,
    underflow_zero="positive",
    validated=(
        "HYPOTHESIS (unprobed): as BLACKWELL_SM120_MXF4 but a nonzero sum below the subnormal grid is +0, as the "
        "lockfile's zero_handling prose ('computed zero normalizes to +0') reads"
    ),
)


#: Every model instance by name -- the grouped-sum pipelines and the block-scaled hypotheses alike.
MODELS: dict[str, Model] = {
    m.name: m
    for m in (
        *PIPELINES.values(),
        ADA_E5M2_M16N8K32,
        HOPPER_E5M2_K32,
        BLACKWELL_SM120_NVF4,
        BLACKWELL_SM120_MXF4,
        BLACKWELL_SM120_MXF4_UNDERFLOW_PLUS0,
    )
}
