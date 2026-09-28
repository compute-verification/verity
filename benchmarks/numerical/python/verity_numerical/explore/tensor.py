"""Tensor-core cost model for the campaign's design space (costing lane, 2026-09-21).

Every earlier ``explore`` lane priced a field multiplication as one 64-bit SIMT multiply (``EXT_MULT_COST`` Goldilocks
equivalents).  ``archive/AGENTS.md`` ("Tensor cores are the target machine") makes the prover's first-class consumer
the GPU tensor core: UINT8 x UINT8 -> INT32 multiply-accumulate (MMA) with deferred modular reduction, TensorZKP's
formulation (Lu et al., USENIX Security 2026, ePrint 2026/1250).  This module re-prices the same relation (the WP2
checker contracts, ``security.census``; nothing is re-arithmetised) in **limb products** and converts to time on an
A100 / H100 at their INT8 dense peaks.  Everything is an ESTIMATE from the cost model below; the only measured numbers
are TensorZKP's (``TENSORZKP``) and the two candidates' recorded CPU runs, both used for calibration.

Price list (:func:`limbs`, :func:`mul_price`).  A field element of ``b`` bits is ``L = ceil(b / w)`` limbs of ``w``
bits; a product of two elements as a limb convolution is ``L_a L_b`` limb MACs (a matmul does the schoolbook
convolution, Karatsuba is not available to it).  Goldilocks (64 bits): 8 limbs, 64 MACs per base multiply; BabyBear /
Mersenne31 (31 bits): 4 limbs, 16 MACs.  A degree-``d`` extension element is ``dL`` limbs and an extension x extension
product is ``(dL)^2`` MACs (TensorZKP eq. 15-16: the ``kL x kL`` block); trace x challenge is ``L (dL)``.  Modular
reduction is deferred: one reduction of a ``(dL)^2`` INT32 block per inner product (or per output element of a
scalar-vector product), on SIMT cores, counted separately (:func:`reduction_words`).

INT32 accumulator (:func:`max_inner_length`).  With ``w``-bit unsigned limbs a single product is ``(2^w - 1)^2``; an
inner product of ``N`` such products must stay below ``2^31 - 1``: ``N < (2^31 - 1) / (2^w - 1)^2``.  ``w = 8``: 33,025
(TensorZKP eq. 23; they chunk at ``B = 32,768`` and add chunks in INT64).  ``w = 16``: 0 -- a single 16-bit x 16-bit
product overflows, so there is no 16-bit-limb option with INT32 accumulate; ``w = 12`` gives 128 and ``w = 11`` 512,
and A100/H100 tensor cores have no INT16 MMA anyway (the INT8 rate is the highest integer rate on both).  BF16 / FP16
MMA with FP32 accumulate are exact only while the sum stays below ``2^24``: with 8-bit limbs that is ``N < 256``, with
FP16's 11-bit significand ``N < 3`` at ``w = 11``; neither beats INT8 (:data:`LIMB_OPTIONS`).

Sumcheck as matmul (:func:`sumcheck_macs`, TensorZKP Algorithm 1 and 5.1 "higher degree").  A round over a table of
``t`` elements with ``m`` multilinear factors: the ``2^m`` even/odd combinations of the factors are inner products
(matmul) of the elementwise products of ``m - 1`` factors; the elementwise products themselves (Hadamards) are SIMT
work: TensorZKP forms ``g_b o h_c`` on SIMT and the eight triple inner products on tensor cores.  Folding each of the
``m`` tables is a scalar-vector product ``Z = V x C`` with ``C`` of shape ``dL x 4dL`` (eq. 14): ``4 (dL)^2`` MACs per
element, two of them (``r`` and ``1 - r``) per table.  Summing ``t`` over the rounds gives ``N`` elements, so a
sumcheck over ``N`` elements costs ``N (dL)^2 (2^m + 8 m)`` matmul MACs and ``N (2^m - 4)`` Hadamards: 20 (dL)^2 for
TensorZKP's degree-2 product (their 4.04 ms at 2^25 is 17% of the H100 INT8 peak: :func:`tensorzkp_calibration`).

Designs.  A-GKR (``archive/gkr/PROTOCOL.md``): data-parallel GKR over the census's layered checker (tables of
``units x carried width`` per layer, ``CHECKER_LAYER_WIDTHS_A``, degree 3, three factors ``eq, V_l, V_r``), logUp
fractional-sum GKR per table (levels of ``2^(n-k)`` fractions, three degree-3 terms over five tables), Ligero over the
committed matrix (encoding + row combination).  B-Ligero (``archive/direct/PROTOCOL.md``): Ligero over the whole
trace, helper columns, linear and quadratic row tests; the sparse constraint combination ``r^T A`` is per-unit sparse
(SIMT).  "Hint-free" (lookups on computed wires) in its two branches: commit the 140 query wires (+140 columns/unit,
depth 381 -- the PROTOCOL.md default) or do not (depth 630, ``commit_query_wires=False``).  Encoding options
(:data:`ENCODERS`): Reed-Solomon as a two-stage dense matrix NTT (``n = 128 x 128``: ``2 sqrt(n)`` base MACs per
symbol per stage on tensor cores, twiddles on SIMT -- keeps the protocols' Ligero parameters), or a Brakedown /
Spielman expander code (20 MACs per symbol, ``explore.pcs``'s assumption, gather-bound: TensorZKP's encoder reaches
6% of peak).  Merkle hashing runs on SIMT at a stated GB/s (:data:`HASH_RATES`).

VOLE (designated verifier, ``explore.vole``'s embedding): the online prover forms per multiplication check one
``F_{p^r} x F_{p^r}`` and two ``F_p x F_{p^r}`` products (elementwise -- Hadamards, SIMT) and two long inner products
with the verifier's ``chi`` (matmul); the sVOLE expansion is the LPN matrix-vector product (primal, ``d = 10`` local
code: 10 MACs per correlation, a gather; dense dual: ``k`` MACs per correlation, a matmul, and not viable).

Conversion.  ``t = MACs / (peak_MAC_per_s x utilisation)`` with utilisation 1.0 and TensorZKP's achieved 0.17;
SIMT field multiplications at a stated rate; hashing at a stated GB/s; the parts are summed (no overlap assumed).
``overhead.vs_native_peak`` = prover seconds per VU / ``contract.native_seconds_per_vu(1536)``.

Privacy knob (``zk``, zk-construction lane, 2026-09-22; ``backends/ZK_CONSTRUCTION.md``).  The accountant's rows
(``ligero_rows``) and committed elements already contain the honest-verifier ZK layer of the protocol documents: the
``k - l`` row randomizers (payload ``l = 3840`` of ``k = 4096`` symbols), the three blinding rows, the transcript pads
of A-GKR (``pad_elements`` x extension degree) and the HM96 leaf commitments.  ``zk="hvzk"`` (default) is
therefore what ``tensor.md`` reported; ``zk="none"`` removes them (payload ``k`` per row, no blinding rows, no pads,
plain Merkle leaves) and is the non-ZK diagnostic a kernel measurement corresponds to; ``zk="malicious"`` adds the
verifier-coin-commitment step 0 (HM96, ``rounds.protocol`` commitments verified by the prover; +1 depth).  Prover
randomness (randomizers, blinding rows, pads, HM96 strings) is priced as PRG output at :data:`PRG_BYTES_PER_S`.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from verity_numerical.bench import contract
from verity_numerical.explore import fields as fl
from verity_numerical.explore import vole as vl
from verity_numerical.security import accounting as acc
from verity_numerical.security import census

__all__ = [
    "DESIGNS",
    "ENCODERS",
    "FIELDS",
    "GPUS",
    "HASH_RATES",
    "LIMB_OPTIONS",
    "TENSORZKP",
    "ZK_LEVELS",
    "Design",
    "FieldSpec",
    "Gpu",
    "Priced",
    "design_row",
    "extension_degree",
    "field_price_list",
    "limbs",
    "main",
    "max_inner_length",
    "mul_price",
    "price_design",
    "price_vole",
    "reduction_words",
    "report",
    "sumcheck_macs",
    "tensorzkp_calibration",
    "unit_price",
]

K_TARGET = 1536
B_TARGET = 4096
UNITS_PER_VU = K_TARGET // 16
TARGETS = (-80, -128)

# ---------------------------------------------------------------------------------------------------------------------
# Hardware (datasheet peaks; dense, no sparsity)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gpu:
    name: str
    int8_tops: float               # dense INT8 tensor-core peak, tera-ops/s (1 MAC = 2 ops)
    bf16_tflops: float
    fp32_tflops: float             # SIMT peak, used to scale the SIMT field-multiplication rate
    hbm_bytes_per_s: float
    simt_field_mults_per_s: float  # F_t (64-bit) multiplications per second on SIMT cores (ASSUMPTION, see note)
    source: str

    @property
    def peak_macs_per_s(self) -> float:
        return self.int8_tops * 1e12 / 2


#: SIMT field-multiplication rate: ``explore.pcs`` used 1e11 Goldilocks mults/s as the indicative A100 figure (10% of
#: 19.5 TOPS INT32 / ~12 integer ops per 64-bit modular multiply); H100 scaled by its FP32 peak (67 / 19.5).  A 31-bit
#: multiply is taken at half the cost (``fields.FieldOption.mult_vs_goldilocks``).  Unmeasured.
GPUS: dict[str, Gpu] = {
    "A100": Gpu("NVIDIA A100 SXM4 80GB", 624, 312, 19.5, 2.039e12, 1.0e11,
                "NVIDIA A100 Tensor Core GPU datasheet: INT8 624 TOPS, BF16 312 TFLOPS, FP32 19.5 TFLOPS, HBM2e 2,039 GB/s"),
    "H100": Gpu("NVIDIA H100 SXM5 80GB", 1979, 989, 67, 3.35e12, 1.0e11 * 67 / 19.5,
                "NVIDIA H100 Tensor Core GPU datasheet (SXM): INT8 1,979 TOPS, BF16 989 TFLOPS, FP32 67 TFLOPS, HBM3 3.35 TB/s"),
}

#: The RTX 4090 (hash-gpu lane's probe device, not a campaign anchor): kept out of ``GPUS`` so the report's rows do not
#: change; ``reprice`` passes it to ``Priced.times`` explicitly.  Datasheet: INT8 dense 660.6 TOPS (1,321 sparse), BF16
#: 165.2 TFLOPS (FP32 accumulate), FP32 82.6 TFLOPS, GDDR6X 1,008 GB/s.
GPU_RTX4090 = Gpu("NVIDIA GeForce RTX 4090", 660.6, 165.2, 82.6, 1.008e12, 1.0e11 * 82.6 / 19.5,
                  "NVIDIA Ada GPU architecture whitepaper / RTX 4090 datasheet: INT8 660.6 TOPS dense, BF16 165.2 TFLOPS, FP32 82.6 TFLOPS, 1,008 GB/s")

#: Merkle-hash rates (bytes hashed per second) on SIMT cores.  Three stated values; none measured by this lane -- the
#: hash-gpu lane's measured 4090 rates are in ``reports/hash_gpu.json`` and applied by ``verity/ml/kernels/hash_gpu/reprice.py``
#: (or ``--hash-rate`` below).
HASH_RATES: dict[str, dict] = {
    "blake3_simt_conservative": {"bytes_per_s": 10e9, "basis": "explore.pcs indicative GPU hash rate (10 GB/s); conservative"},
    "blake3_simt": {"bytes_per_s": 50e9, "basis": "ASSUMPTION: Blake3 tree hashing on A100/H100 SIMT at 50 GB/s (long-message throughput; no measurement in hand)"},
    "tensorzkp_merkle_implied": {"bytes_per_s": None, "basis": "derived below from TensorZKP Table 1: 8.59 ms Merkle at 2^25 x 32 B x 1/rate; hash function adopted from BatchZK, unstated"},
    "poseidon2_goldilocks_simt": {"bytes_per_s": None, "basis": "Poseidon2-Goldilocks t=12 (explore.authentication): 118 x^7 S-boxes ~ 4 mults each + MDS ~ 500 F_t mults per 64 B absorbed -> ~8 mults/byte at the SIMT rate"},
}
POSEIDON2_MULTS_PER_BYTE = 500 / 64
#: AES-128 in counter mode on SIMT cores for the GGM puncturable PRF of the sVOLE expansion (bytes of PRG output per
#: second).  ASSUMPTION, unmeasured: GPU AES-CTR implementations report hundreds of GB/s; 100 GB/s is taken.
PRG_BYTES_PER_S = 100e9

# ---------------------------------------------------------------------------------------------------------------------
# Limbs and the price list
# ---------------------------------------------------------------------------------------------------------------------


def limbs(bits: int, w: int = 8) -> int:
    """Number of ``w``-bit limbs of a ``bits``-bit element."""
    if bits <= 0 or w <= 0:
        raise ValueError("bits and limb width must be positive")
    return -(-bits // w)


def mul_price(bits_a: int, bits_b: int, w: int = 8) -> int:
    """Limb MACs of one product of a ``bits_a``-bit by a ``bits_b``-bit element as a limb convolution (schoolbook)."""
    return limbs(bits_a, w) * limbs(bits_b, w)


def max_inner_length(w: int, acc_bits: int = 31, signed_products: bool = False) -> int:
    """Largest inner-product length ``N`` whose accumulated ``w``-bit limb products stay below ``2^acc_bits - 1``
    (TensorZKP eq. 23: ``N (2^w - 1)^2 < 2^31 - 1`` for UINT8 into SINT32).  0 means a single product overflows."""
    top = (2 ** (w - 1)) ** 2 if signed_products else (2**w - 1) ** 2
    return (2**acc_bits - 2) // top


def reduction_words(dl: int) -> int:
    """INT32 words to combine (with powers of 2^8 and one modular reduction per coefficient) after one inner-product
    block of ``dl``-limb elements: the ``dl x dl`` result block (TensorZKP eq. 17-18).  SIMT work, per inner product."""
    return dl * dl


LIMB_OPTIONS: dict[str, dict] = {
    "uint8_int32": {"limb_bits": 8, "accumulate": "INT32", "max_inner_length": max_inner_length(8),
                    "mma": "UINT8/INT8 MMA (A100, H100)", "viable": True,
                    "note": "TensorZKP's configuration; chunk at 32,768 and add chunks in INT64"},
    "int16_int32": {"limb_bits": 16, "accumulate": "INT32", "max_inner_length": max_inner_length(16),
                    "mma": "none: A100/H100 tensor cores have no INT16 MMA", "viable": False,
                    "note": "a single 16 x 16-bit product exceeds 2^31; 12-bit limbs give N <= 128, 11-bit N <= 512"},
    "int12_int32": {"limb_bits": 12, "accumulate": "INT32", "max_inner_length": max_inner_length(12),
                    "mma": "would need INT16 MMA (absent)", "viable": False,
                    "note": "N <= 128 defeats deferred reduction; no instruction to run it on"},
    "bf16_fp32": {"limb_bits": 8, "accumulate": "FP32 (24-bit significand)", "max_inner_length": (2**24 - 1) // (2**8 - 1) ** 2,
                  "mma": "BF16 MMA (A100 312, H100 989 TFLOPS)", "viable": False,
                  "note": "BF16 holds 8 significant bits so limbs are still 8-bit; exact sums only below 2^24 -> N < 256 at half the INT8 rate"},
    "fp16_fp32": {"limb_bits": 11, "accumulate": "FP32", "max_inner_length": (2**24 - 1) // (2**11 - 1) ** 2,
                  "mma": "FP16 MMA", "viable": False, "note": "11-bit limbs, N < 4 before the FP32 significand is exhausted"},
    "fp8_fp32": {"limb_bits": 4, "accumulate": "FP32", "max_inner_length": (2**24 - 1) // (2**4 - 1) ** 2,
                 "mma": "Hopper FP8 (E4M3: 4 significant bits)", "viable": False,
                 "note": "4-bit limbs quadruple the limb count (16x the MACs of 8-bit) for a 2x rate: never pays"},
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    bits: int
    option: fl.FieldOption
    simt_mult_vs_goldilocks: float

    @property
    def L(self) -> int:
        return limbs(self.bits)

    @property
    def element_bytes(self) -> float:
        return self.option.element_bytes


FIELDS: dict[str, FieldSpec] = {
    "goldilocks": FieldSpec("goldilocks", 64, fl.GOLDILOCKS, 1.0),
    "babybear": FieldSpec("babybear", 31, fl.BABYBEAR, 0.5),
    "m31": FieldSpec("m31", 31, fl.MERSENNE31, 0.5),
}


def extension_degree(fs: FieldSpec, target_log2: int, candidate: str = "A", B: int = B_TARGET) -> int:
    """The first challenge-extension degree that meets the operating point (fields lane / accounting, not re-derived):
    Goldilocks 2 / 3 at 2^-80 / 2^-128; the 31-bit primes 4 / 6 (T_range16's logUp identity term floors degree 5 at
    B = 4096)."""
    d, _t, _bud = fl.first_small_field_configuration(target_log2, fs.option, candidate, K_TARGET, B)
    return d


def field_price_list(target_log2: int = -128) -> dict:
    """Limb counts and MAC prices for every field at one operating point: base x base, trace x challenge, challenge x
    challenge, the extension degree, and the challenge field's limb count (the two families tie: 24 limbs at 2^-128)."""
    out = {}
    for name, fs in FIELDS.items():
        d = extension_degree(fs, target_log2)
        dl = d * fs.L
        out[name] = {
            "bits": fs.bits, "limbs": fs.L, "base_x_base_macs": fs.L**2, "extension_degree": d,
            "challenge_bits": fs.bits * d, "challenge_limbs": dl,
            "trace_x_challenge_macs": fs.L * dl, "challenge_x_challenge_macs": dl * dl,
            "reduction_words_per_inner_product": reduction_words(dl),
            "simt_mults_vs_goldilocks": fs.simt_mult_vs_goldilocks,
            "simt_ext_mult_goldilocks_equiv": acc.EXT_MULT_COST[d] * fs.simt_mult_vs_goldilocks,
        }
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Sumcheck and encoding atoms
# ---------------------------------------------------------------------------------------------------------------------


def sumcheck_macs(n_elements: int, m_factors: int, dl: int, terms: int = 1, tables: int | None = None) -> dict:
    """A sumcheck over ``n_elements`` (summed over its rounds) of ``terms`` products of ``m_factors`` multilinear
    tables each (``tables`` distinct tables folded; default ``m_factors``): matmul MACs ``n dl^2 (terms 2^m + 8 tables)``
    and SIMT Hadamards ``n terms (2^m - 4)`` (module docstring).  Degree 2, one term, two tables: 20 dl^2 (TensorZKP)."""
    tables = m_factors if tables is None else tables
    ip = terms * 2**m_factors
    return {"matmul_macs": n_elements * dl * dl * (ip + 8 * tables), "hadamards": n_elements * terms * max(0, 2**m_factors - 4),
            "inner_products_per_round": ip, "reductions": ip * math.ceil(math.log2(max(2, n_elements)))}


def packed_sumcheck_macs(n_elements: int, packed_rounds: int = 0, dl: int = 24, base_limbs: int = 4) -> dict:
    """Limb MACs of the eq-trick sumcheck of ``archive/gkr/packed`` (DESIGN.md 4) over ``n_elements`` base-field
    elements per table (two folded tables ``g, h``; ``eq`` is never a table), with the first ``packed_rounds = k`` rounds
    in the base field: packed round ``i`` does ``3 . 4^(i-1)`` base x base Hadamards per surviving pair and one
    ``base_limbs x dl`` weighted inner product each (``n / 2^i`` pairs); then ONE fold of both tables by the ``k``
    challenges (``2 . 2^k . base_limbs . 2 dl`` per output... = ``2 . 8 dl`` per input element); extension rounds
    ``i > k`` do 3 ``dl x dl`` inner products and 2 ``2 dl x 2 dl`` folds per pair.  ``k = 0`` is the eq-trick alone;
    the plain model (:func:`sumcheck_macs` with ``m = 3``, three folded tables) is returned for comparison.  The bench
    (``archive/gkr/packed/bench.py::mac_model``) counts the same MACs as issued."""
    n = n_elements
    packed = sum(3 * 4 ** (i - 1) * (n >> i) * base_limbs * dl for i in range(1, packed_rounds + 1))
    fold = 2 * 8 * dl * n if packed_rounds else 0
    ext = sum((n >> i) * (3 * dl * dl + 2 * 4 * dl * dl) for i in range(packed_rounds + 1, max(1, n.bit_length() - 1) + 1))
    plain = sumcheck_macs(n, 3, dl)["matmul_macs"]
    total = packed + fold + ext
    return {"packed_rounds": packed_rounds, "matmul_macs": total, "matmul_macs.packed_rounds": packed, "matmul_macs.fold_k": fold,
            "matmul_macs.ext_rounds": ext, "hadamards_base": sum(3 * 4 ** (i - 1) * (n >> i) for i in range(1, packed_rounds + 1)),
            "hadamards_ext": sum(3 * (n >> i) for i in range(packed_rounds + 1, max(1, n.bit_length() - 1) + 1)),
            "plain_model_matmul_macs": plain, "saving_vs_plain_model": plain / total if total else None}


ENCODERS: dict[str, dict] = {
    "rs_matrix_ntt": {"basis": "Reed-Solomon n=16384/k=4096 (the protocols' Ligero code) as two-stage dense matrix DFTs on tensor cores "
                               "(TensorFHE-style O(N sqrt N)): interpolate k = 64 x 64 (2 x 64 MACs per payload symbol), evaluate "
                               "n = 128 x 128 (2 x 128 per codeword symbol); twiddle multiplies (k + n per row) on SIMT",
                      "macs_per_row_base": 4096 * 128 + 16384 * 256, "simt_mults_per_row": 4096 + 16384,
                      "utilisation_anchor": "sumcheck-like dense matmul: 0.17"},
    "brakedown_expander": {"basis": "Brakedown/Spielman expander code, 20 base MACs per codeword symbol (explore.pcs assumption), "
                                    "SpMM gather (TensorZKP 5.2); changes the Ligero test to Brakedown's (rate 0.65, distance 0.07: "
                                    "more opened columns, explore.pcs.brakedown_row) -- not the protocols' code",
                           "macs_per_symbol_base": 20, "simt_mults_per_row": 0, "utilisation_anchor": "TensorZKP encoder: 0.06"},
    "rs_ntt_simt": {"basis": "the SIMT-priced baseline: butterflies as F_t mults on SIMT (accounting._ntt_ops)", "matmul": False},
}

# ---------------------------------------------------------------------------------------------------------------------
# TensorZKP calibration
# ---------------------------------------------------------------------------------------------------------------------

TENSORZKP: dict = {
    "paper": "Lu, Zhang, Guo, Liu, Qu, Wang, Chen, Zhang. TensorZKP: Repurposing GPU Tensor Cores for High-Performance "
             "Zero-Knowledge Proofs. USENIX Security 2026; ePrint 2026/1250",
    "hardware": "NVIDIA H100 80 GB, nvcc 12.6 (7.1)",
    "field": "BabyBear p = 2^31 - 2^27 + 1, extension degree k = 8, P(x) = x^8 - 11 (7.1); the exposition of 4.1 uses k = 4",
    "limbs": {"limb_bits": 8, "L_per_base": 4, "k": 8, "limbs_per_element": 32, "macs_per_ext_mult": 32 * 32,
              "scalar_matrix_shape": "kL x 4kL = 32 x 128"},
    "int32_bound": {"statement": "N (255)^2 < 2^31 - 1 => N < 33,025 (eq. 23)", "chunk": 32768, "chunk_sum": "SINT64"},
    "measured_ms_at_2^25": {"inner_product": 0.85, "scalar_vector": 0.91, "sumcheck_degree2": 4.04, "encoder": 11.58,
                            "end_to_end_2^25_mult_gates": 215.28,
                            "breakdown": {"merkle": 8.59, "encoder": 50.64, "open": 6.30, "perm_check": 134.00, "zero_check": 16.96}},
    "speedups": {"inner_product_vs_simt": 13.1, "scalar_vector_vs_simt": 10.0, "sumcheck_vs_icicle_gpu": 88, "encoder_vs_batchzk": 22,
                 "pcs_vs_simt": 16.9, "piop_vs_simt": 44.3, "end_to_end_vs_simt": 36.2, "end_to_end_vs_cpu": 955},
    "other": {"chunking_overhead_share": "< 2% (7.4)", "merkle_share": "< 5% (7.4)", "memory_gb_at_2^25": 52.9,
              "brakedown_rate_assumed_here": 0.65},
}


def tensorzkp_calibration() -> dict:
    """What TensorZKP's measured module times imply: limb MACs by the module's formula, the fraction of the H100 INT8
    peak achieved, the HBM floor of each kernel (32 B per element), and what its 215.28 ms means for our relation as a
    HyperPlonk circuit (gates = mul + lin + lookup per VU)."""
    h = GPUS["H100"]
    N = 2**25
    dl = 32
    m = TENSORZKP["measured_ms_at_2^25"]
    rows = {}

    def row(name, macs, bytes_moved, ms):
        t_peak = macs / h.peak_macs_per_s
        rows[name] = {"limb_macs": macs, "seconds_at_peak": t_peak, "measured_seconds": ms / 1e3,
                      "utilisation_of_int8_peak": t_peak / (ms / 1e3),
                      "hbm_floor_seconds": bytes_moved / h.hbm_bytes_per_s,
                      "hbm_fraction": (bytes_moved / h.hbm_bytes_per_s) / (ms / 1e3)}

    row("inner_product", N * dl * dl, 2 * N * dl, m["inner_product"])
    row("scalar_vector", N * dl * 4 * dl, 2 * N * dl, m["scalar_vector"])
    sc = sumcheck_macs(N, 2, dl)
    row("sumcheck_degree2", sc["matmul_macs"], 10 * N * dl, m["sumcheck_degree2"])       # ~5 reads + 2 writes of 2 tables per round
    enc_macs = N * ENCODERS["brakedown_expander"]["macs_per_symbol_base"] * dl * dl / TENSORZKP["other"]["brakedown_rate_assumed_here"]
    row("encoder", enc_macs, 2 * N * dl / TENSORZKP["other"]["brakedown_rate_assumed_here"], m["encoder"])
    rows["encoder"]["note"] = "MAC count rests on explore.pcs's 20-MACs-per-symbol Brakedown assumption; utilisation is only as good as that"
    merkle_bytes = N * dl / TENSORZKP["other"]["brakedown_rate_assumed_here"]
    merkle_rate = merkle_bytes / (m["breakdown"]["merkle"] / 1e3)
    HASH_RATES["tensorzkp_merkle_implied"]["bytes_per_s"] = merkle_rate
    vu = census.census_vu(K_TARGET)
    gates_per_vu = vu.mul + vu.lin_total + vu.lookup
    vus = N / gates_per_vu
    e2e = m["end_to_end_2^25_mult_gates"] / 1e3
    per_vu = e2e / vus
    return {
        "modules": rows,
        "realistic_utilisation": {"matmul_dense": round(rows["sumcheck_degree2"]["utilisation_of_int8_peak"], 3),
                                  "matmul_gather": round(rows["encoder"]["utilisation_of_int8_peak"], 3),
                                  "basis": "degree-2 sumcheck (dense, HBM-bound at ~80% of 3.35 TB/s) and the expander encoder (gather)"},
        "merkle_implied_bytes_per_s": merkle_rate,
        "our_relation_as_hyperplonk": {
            "gates_per_vu": gates_per_vu, "gates_basis": "mul + lin + lookup of census_vu(1536): one HyperPlonk gate per predicate",
            "vus_per_2^25_gate_proof": vus, "seconds_per_vu_end_to_end": per_vu,
            "overhead_end_to_end": contract.overhead(per_vu, K_TARGET),
            "overhead_zero_check_only": contract.overhead(m["breakdown"]["zero_check"] / 1e3 / vus, K_TARGET),
            "overhead_perm_check": contract.overhead(m["breakdown"]["perm_check"] / 1e3 / vus, K_TARGET),
            "overhead_pcs": contract.overhead((m["breakdown"]["merkle"] + m["breakdown"]["encoder"] + m["breakdown"]["open"]) / 1e3 / vus, K_TARGET),
            "note": "62% of TensorZKP's time is the permutation (wiring) product-check, which GKR (A) and a Ligero trace (B) do not have; "
                    "HyperPlonk has no lookup argument, so 29k range chunks per VU are counted as plain gates (undercount)",
        },
    }


# ---------------------------------------------------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    name: str
    candidate: str                 # "A" | "B"
    commit_query_wires: bool = True
    label: str = ""


DESIGNS: dict[str, Design] = {
    "A": Design("A", "A", True, "GKR + logUp-GKR + Ligero; 140 query wires committed (depth 381)"),
    "A-hintfree-depth630": Design("A-hintfree-depth630", "A", False, "GKR + logUp-GKR + Ligero; lookups on computed wires, nothing extra committed (depth 630)"),
    "B": Design("B", "B", True, "direct constraints in Ligero + logUp helper columns (depth 3)"),
}
#: The "+140 columns" branch of the hint-free variant is A-GKR's default layout (PROTOCOL.md 4.3).
HINT_FREE_BRANCHES = {"commit_query_wires": "A", "no_commit": "A-hintfree-depth630"}

#: Hint (witness) generation on SIMT: integer ops per unit.  ASSUMPTION (unmeasured on GPU): ~20 integer ops per checker
#: predicate (358 mul + 137 lin + 306 lookups + 225 hints ~ 1,000 predicates).  Anchor: B-Ligero's Rust generator
#: takes 4.7 s for 4,096 units single-threaded (archive/direct/README.md) = 1.15 ms per unit on one CPU core.
HINT_OPS_PER_UNIT = 20_000
HINT_CPU_SECONDS_PER_UNIT = 4.7 / 4096
SIMT_INT_OPS_PER_S = {"A100": 19.5e12 * 0.25, "H100": 67e12 * 0.25, "RTX4090": 82.6e12 * 0.25}      # 25% of the FP32/INT32 peak (ASSUMPTION)


@dataclass
class Priced:
    """Limb-priced work of one design x field x operating point for the B = 4096 batch."""

    design: str
    field: str
    target_log2: int
    extension_degree: int
    units: int
    committed_elements: int
    committed_bytes: float
    encoded_bytes: float
    hash_bytes: float
    matmul: dict = field(default_factory=dict)        # component -> limb MACs
    simt_field_mults: dict = field(default_factory=dict)   # component -> F_t-equivalent multiplications on SIMT
    simt_int_ops: dict = field(default_factory=dict)
    reductions_words: int = 0
    depth: int = 0
    notes: list = field(default_factory=list)
    zk: str = "hvzk"                  # none | hvzk | malicious (module docstring)
    prg_bytes: float = 0.0            # prover randomness: randomizers, blinding rows, pads, HM96 strings
    rows: int = 0
    communication_bytes: float = 0.0  # transcript bytes (accountant estimate) incl. step-0 commitments
    authentication: str = "none"      # explore.auth_integration.AUTH_LEVELS
    auth_delta: object = None         # the AuthDelta applied (None when authentication == "none")
    ligero_rows: int = 0                                    # rows of the committed Ligero matrix (what the encoder encodes)

    @property
    def matmul_total(self) -> float:
        return sum(self.matmul.values())

    @property
    def simt_total(self) -> float:
        return sum(self.simt_field_mults.values())

    def times(self, gpu: Gpu, gpu_key: str, utilisation: float, hash_rate: float, encoder: str = "rs_matrix_ntt",
              fs: FieldSpec | None = None, encoding_utilisation: float | None = None,
              encoder_seconds_per_row: float | None = None, measured: dict | None = None) -> dict:
        """Seconds per B=4096 batch by bucket and per VU, and the overhead.  ``encoding_utilisation`` (default: the
        same as ``utilisation``) prices the encoding component apart: the expander encoder is a gather.
        ``encoder_seconds_per_row`` replaces the modelled encoding bucket by a MEASURED encoder time per Ligero row
        (b-encode lane: the SIMT NTT on the RTX 4090), reported as ``t.encoding``; the encoding MACs then leave the
        matmul bucket (the encoder is not a tensor-core operation).

        ``measured`` (b-commit lane) replaces further modelled buckets by MEASURED seconds per batch, keyed
        ``"commitment"`` (fused encode + hash: replaces the encoding bucket, ``encoding_simt`` and the hash bucket),
        ``"encoding"`` / ``"hash"`` / ``"hint_gen"`` / ``"prg"`` (that bucket), ``"matmul:<component>"`` /
        ``"simt:<component>"`` (that component of :attr:`matmul` / :attr:`simt_field_mults`) and ``"extra:<name>"``
        (a bucket the model does not have, e.g. the column openings).  The result lists ``t.measured`` (name -> s),
        ``t.modelled`` (every component still priced by the model, name -> s) and ``measured_fraction``."""
        simt_scale = fs.simt_mult_vs_goldilocks if fs else 1.0
        u_enc = utilisation if encoding_utilisation is None else encoding_utilisation
        measured = dict(measured or {})
        if encoder_seconds_per_row is not None and "encoding" not in measured and "commitment" not in measured:
            measured["encoding"] = self.ligero_rows * encoder_seconds_per_row
        enc_measured = "encoding" in measured or "commitment" in measured
        mm_skip = {k[len("matmul:"):] for k in measured if k.startswith("matmul:")}
        simt_skip = {k[len("simt:"):] for k in measured if k.startswith("simt:")}
        if enc_measured:
            mm_skip.add("encoding")
            simt_skip.add("encoding_simt")
        modelled = {}
        for k, v in self.matmul.items():
            if k not in mm_skip:
                modelled[f"matmul:{k}"] = v / (gpu.peak_macs_per_s * (u_enc if k == "encoding" else utilisation))
        for k, v in self.simt_field_mults.items():
            if k not in simt_skip:
                modelled[f"simt:{k}"] = v * simt_scale / gpu.simt_field_mults_per_s
        if "hash" not in measured and "commitment" not in measured:
            modelled["hash"] = self.hash_bytes / hash_rate
        if "hint_gen" not in measured:
            modelled["hint_gen"] = self.units * HINT_OPS_PER_UNIT / SIMT_INT_OPS_PER_S[gpu_key]
        if "prg" not in measured:
            modelled["prg"] = self.prg_bytes / PRG_BYTES_PER_S
        t_mm = sum(v for k, v in modelled.items() if k.startswith("matmul:")) + sum(v for k, v in measured.items() if k.startswith("matmul:"))
        t_simt = sum(v for k, v in modelled.items() if k.startswith("simt:")) + sum(v for k, v in measured.items() if k.startswith("simt:"))
        t_enc = measured.get("encoding", modelled.get("matmul:encoding", 0.0))
        if "commitment" in measured:
            t_enc = 0.0
        t_hash = measured.get("hash", modelled.get("hash", 0.0)) if "commitment" not in measured else 0.0
        t_hint = measured.get("hint_gen", modelled.get("hint_gen", 0.0))
        t_prg = measured.get("prg", modelled.get("prg", 0.0))
        t_commit = measured.get("commitment", 0.0)
        t_extra = sum(v for k, v in measured.items() if k.startswith("extra:"))
        total = t_mm + (t_enc if enc_measured else 0.0) + t_simt + t_hash + t_hint + t_prg + t_commit + t_extra
        per_vu = total / B_TARGET
        t_measured_total = sum(measured.values())
        return {"gpu": gpu_key, "utilisation": utilisation, "zk": self.zk,
                "encoder": encoder if not enc_measured else (f"measured:{encoder_seconds_per_row:.3g}s/row" if encoder_seconds_per_row else "measured"),
                "t.matmul": t_mm, "t.encoding": t_enc, "encoding_measured": enc_measured,
                "t.simt_field": t_simt, "t.hash": t_hash, "t.hint_gen": t_hint, "t.prg": t_prg,
                "t.commitment": t_commit, "t.extra": t_extra,
                "t.total": total, "t.measured": measured, "t.modelled": modelled,
                "measured_fraction": t_measured_total / total if total else 0.0,
                "matmul_share_of_time": t_mm / total, "seconds_per_vu": per_vu,
                "overhead.vs_native_peak": contract.overhead(per_vu, K_TARGET),
                "hint_gen_cpu_anchor_seconds": self.units * HINT_CPU_SECONDS_PER_UNIT}


ZK_LEVELS = ("none", "hvzk", "malicious")
#: HM96 randomness per leaf / coin: x_j of 6k+4 bits at k = 160 (O(k) variant), rounded to bytes.
HM_RANDOM_BYTES = (6 * 160 + 4 + 7) // 8


def _zk_rows(committed: int, zk: str, payload_zk: int = fl.L_PAYLOAD, payload_none: int = fl.K_CODE) -> int:
    """Ligero rows for ``committed`` elements: payload ``l`` plus three blinding rows with ZK, payload ``k`` without."""
    if zk == "none":
        return -(-committed // payload_none)
    return -(-committed // payload_zk) + 3


def _small_field(fs: FieldSpec) -> fl.SmallFieldCensus:
    return fl.small_field_census(fs.option.trace)


def _ligero(target_log2: int, fs: FieldSpec, candidate: str, B: int) -> tuple[int, acc.LigeroParams]:
    d, t, _ = fl.first_small_field_configuration(target_log2, fs.option, candidate, K_TARGET, B)
    return d, acc.LigeroParams(n=fl.N_CODE, k=fl.K_CODE, l=fl.L_PAYLOAD, t=t)


def _encoding(rows: int, L: int, encoder: str) -> tuple[float, float]:
    """(matmul MACs, SIMT F_t mults) to encode ``rows`` Ligero rows of ``L``-limb symbols."""
    e = ENCODERS[encoder]
    if encoder == "rs_matrix_ntt":
        return rows * e["macs_per_row_base"] * L * L, rows * e["simt_mults_per_row"]
    if encoder == "brakedown_expander":
        return rows * fl.N_CODE * e["macs_per_symbol_base"] * L * L, 0.0
    if encoder == "rs_ntt_simt":
        return 0.0, acc._ntt_ops(rows, fl.N_CODE, fl.K_CODE)
    raise ValueError(encoder)


def price_design(design: Design, fs: FieldSpec, target_log2: int, B: int = B_TARGET, encoder: str = "rs_matrix_ntt",
                 zk: str = "hvzk", authentication: str = "none", auth_gadget: str | None = None, auth_sharing: str = "bench",
                 weights_public: bool = False) -> Priced:
    """Limb-price one design on one field at one operating point (module docstring for the model).  ``zk`` in
    :data:`ZK_LEVELS`: ``hvzk`` is the protocol documents' layer (and tensor.md's rows), ``none`` strips it, ``malicious``
    adds the step-0 verifier-coin commitment.  ``authentication`` (auth-integration lane, ``explore.auth_integration``):
    ``none`` is every number before that lane (``authentication = excluded`` on the ledger); ``in_circuit_sha256`` /
    ``leaf_v2_pcs_hash`` / ``algebraic`` / ``verifier_openings`` add that option's binding of x / w / y to the Verity
    roots (extra committed elements, GKR elements, quadratic products, row combinations, depth) so the result is the
    *complete* Verity proof's cost; ``weights_public`` routes W through verifier-side openings."""
    if zk not in ZK_LEVELS:
        raise ValueError(f"zk must be one of {ZK_LEVELS}, got {zk!r}")
    from verity_numerical.explore import auth_integration as ai

    if authentication not in ai.AUTH_LEVELS + ("verifier_openings",):
        raise ValueError(f"authentication must be one of {ai.AUTH_LEVELS}, got {authentication!r}")
    sfc = _small_field(fs)
    d, lig = _ligero(target_log2, fs, design.candidate, B)
    L, dl = fs.L, d * fs.L
    Fc = fs.option.trace.extension(d)
    tables = sfc.tables()
    units = B * UNITS_PER_VU
    p = Priced(design.name, fs.name, target_log2, d, units, 0, 0.0, 0.0, 0.0, zk=zk, authentication=authentication)
    ad = ai.delta(authentication, design.candidate, d, B, auth_sharing, weights_public, auth_gadget, lig, L)
    p.auth_delta = ad
    coin_bits = 160 if zk == "malicious" else None
    leaf_bits = None if zk == "none" else 160
    eb = fs.element_bytes
    if design.candidate == "A":
        cand = acc.CandidateA(K=K_TARGET, B=B, trace_field=fs.option.trace, challenge_field=Fc, ligero=lig,
                              hash_queries=fl.HASH_QUERIES, tables=tables, inputs_per_unit=sfc.columns_committed_a,
                              query_wires_per_unit=sfc.queries_on_computed_wires,
                              depth=acc.DepthOptions(commit_query_wires=design.commit_query_wires),
                              coin_commitment_bits=coin_bits, leaf_commitment_bits=leaf_bits)
        bud = acc.account_candidate_a(cand)
        prm = bud.parameters
        est = acc.census_estimate_a(cand)
        # checker GKR: per layer a degree-3 data-parallel sumcheck (eq, V_l, V_r) over units x carried width
        gkr_elements = units * sum(acc.CHECKER_LAYER_WIDTHS_A)
        sc = sumcheck_macs(gkr_elements, 3, dl)
        p.matmul["gkr_checker"] = sc["matmul_macs"]
        p.simt_field_mults["gkr_hadamards"] = sc["hadamards"] * acc.EXT_MULT_COST[d]
        # first round reads F_t wires (L x dl instead of dl x dl): credit the difference
        p.matmul["gkr_checker"] -= gkr_elements * (dl * dl - L * dl) * 8 // 2
        # logUp fractional GKR: leaves per table, levels of 2^(n-k) fractions; three degree-3 terms over five tables
        leaves = 0
        for tb in tables:
            shards, _per, n_dim = acc._shards(cand, tb)
            leaves += shards * 2**n_dim
        fr = sumcheck_macs(leaves, 3, dl, terms=3, tables=5)
        p.matmul["logup_fractional_gkr"] = fr["matmul_macs"] * cand.lookup_repetitions
        p.simt_field_mults["logup_hadamards"] = fr["hadamards"] * acc.EXT_MULT_COST[d]
        p.simt_field_mults["logup_tree_build"] = 3 * leaves * acc.EXT_MULT_COST[d]        # p0 q1 + p1 q0, q0 q1 per pair
        committed = prm["committed_trace_elements"]
        if zk == "none":
            committed -= prm["pad_elements"] * d          # the padded-transcript pads exist only for privacy
        # authentication: hash constraints as more checker-GKR elements / logUp leaves, hints as committed elements
        if ad.gkr_elements:
            asc = sumcheck_macs(int(ad.gkr_elements), 3, dl)
            p.matmul["auth_gkr"] = asc["matmul_macs"]
            p.simt_field_mults["auth_hadamards"] = asc["hadamards"] * acc.EXT_MULT_COST[d]
        if ad.lookup_leaves:
            afr = sumcheck_macs(int(ad.lookup_leaves), 3, dl, terms=3, tables=5)
            p.matmul["auth_logup"] = afr["matmul_macs"]
            p.simt_field_mults["auth_logup_hadamards"] = afr["hadamards"] * acc.EXT_MULT_COST[d] + 3 * ad.lookup_leaves * acc.EXT_MULT_COST[d]
        committed += int(ad.committed_elements)
        rows = _zk_rows(committed, zk)
        depth = prm["rounds.sequential_depth"] + ad.depth_delta
        # Ligero: encoding + one row combination (linear test only, no quadratic test: products are inside the GKR)
        enc_mm, enc_simt = _encoding(rows, L, encoder)
        p.matmul["encoding"] = enc_mm
        if enc_simt:
            p.simt_field_mults["encoding_simt"] = enc_simt
        p.matmul["ligero_row_combination"] = rows * fl.N_CODE * L * dl
        p.reductions_words = reduction_words(dl) * (sc["reductions"] + fr["reductions"]) + rows * fl.N_CODE * dl * 2
        trees = 1
    else:
        cand = acc.CandidateB(K=K_TARGET, B=B, trace_field=fs.option.trace, challenge_field=Fc, ligero=lig,
                              hash_queries=fl.HASH_QUERIES, tables=tables, witness_per_unit=sfc.columns_committed_b,
                              linear_per_unit=sfc.lin_total + 2 * sfc.mul, quadratic_per_unit=sfc.mul,
                              coin_commitment_bits=coin_bits, leaf_commitment_bits=leaf_bits)
        bud = acc.account_candidate_b(cand)
        prm = bud.parameters
        est = acc.census_estimate_b(cand)
        committed = prm["committed_trace_elements"] + int(ad.committed_elements)     # + authentication trace cells / helpers
        rows = _zk_rows(committed, zk)
        depth = prm["rounds.sequential_depth"] + ad.depth_delta
        helpers = sum(tb.queries_per_unit * units + tb.rows for tb in tables) + ad.lookup_leaves
        enc_mm, enc_simt = _encoding(rows, L, encoder)
        p.matmul["encoding"] = enc_mm
        if enc_simt:
            p.simt_field_mults["encoding_simt"] = enc_simt
        # linear test: one row combination over every row; quadratic test: combination over the product rows
        quad_rows = -(-(units * sfc.mul + int(ad.products_b)) // (fl.K_CODE if zk == "none" else fl.L_PAYLOAD))
        p.matmul["ligero_row_combination"] = (rows + quad_rows) * fl.N_CODE * L * dl
        # sparse per-unit constraint combination r^T A (4 terms per constraint, F_c x small constant): SIMT
        p.simt_field_mults["constraint_combination"] = 4 * prm["linear_constraints"] * d
        p.simt_field_mults["quadratic_hadamards"] = units * sfc.mul                          # x o y per product triple, F_t
        p.simt_field_mults["helper_inversions"] = 3 * helpers * acc.EXT_MULT_COST[d]         # Montgomery batch trick
        p.reductions_words = (rows + quad_rows) * fl.N_CODE * dl * 2
        trees = 2
    # hashing: the codeword matrix once, the Merkle tree(s); HM96 leaves (hvzk, malicious) hash the x_j strings too
    p.hash_bytes = rows * fl.N_CODE * eb + trees * fl.N_CODE * 2 * acc.DIGEST_BYTES
    if zk != "none":
        p.hash_bytes += trees * fl.N_CODE * acc.HM_LEAF_BYTES
        # prover randomness: k-l randomizers per row, 3 blinding rows of k symbols per tree, HM96 x_j per leaf, pads
        p.prg_bytes = (rows * (fl.K_CODE - fl.L_PAYLOAD) + trees * 3 * fl.K_CODE) * eb + trees * fl.N_CODE * HM_RANDOM_BYTES
        if design.candidate == "A":
            p.prg_bytes += prm["pad_elements"] * d * eb
    coins = est["rounds.protocol"] - (1 if zk == "malicious" else 0)
    if zk == "malicious":
        # step 0: the prover verifies R HM96 openings (hash of x_i each) and evaluates R universal hashes
        p.hash_bytes += coins * (acc.HM_LEAF_BYTES + acc.HM_DECOMMIT_BYTES)
        p.notes.append(f"step 0: {coins} verifier-coin commitments (HM96 k=160) verified by the prover; +1 depth")
    p.communication_bytes = est["transcript.bytes"]           # accountant estimate (8 B per trace element)
    if authentication != "none":
        # the option's own work outside the trace: row combinations over tensor rows (matmul), public linear
        # constraints (SIMT), native hashing by the prover, prover -> verifier bytes
        if ad.matmul_macs:
            p.matmul["auth_cross_tableau"] = ad.matmul_macs
        if ad.simt_field_mults:
            p.simt_field_mults["auth_linear_constraints"] = ad.simt_field_mults
        p.hash_bytes += ad.prover_hash_bytes
        p.prg_bytes += ad.prg_bytes
        p.communication_bytes += ad.proof_bytes
        p.notes.append(f"authentication={authentication}: " + "; ".join(ad.notes))
    p.rows = rows
    p.committed_elements = committed
    p.ligero_rows = rows
    p.committed_bytes = committed * eb
    p.encoded_bytes = rows * fl.N_CODE * eb
    p.depth = depth
    p.simt_int_ops["hint_generation"] = units * HINT_OPS_PER_UNIT
    if fs.name == "m31":
        p.notes.append("Mersenne31 has 2-adicity 1: the n=16384 Reed-Solomon code needs a circle-group FFT (Circle STARKs) or a "
                       "different evaluation set; the matrix-DFT price is taken as for BabyBear")
    return p


def unit_price(fs: FieldSpec, target_log2: int = -128) -> dict:
    """The transition unit (``census_unit`` / the field's small-field census) in limb MACs: (a) matmul-expressible work
    -- one degree-3 GKR sumcheck element per carried wire and one fractional-GKR leaf per lookup, as A-GKR runs
    them -- and (b) non-matmul work in F_t-equivalent SIMT multiplications, both per unit and per K=1536 VU."""
    sfc = _small_field(fs)
    d = extension_degree(fs, target_log2)
    dl = d * fs.L
    ec = acc.EXT_MULT_COST[d]
    carried = sum(acc.CHECKER_LAYER_WIDTHS_A) - (census.census_unit().mul - sfc.mul)
    sc = sumcheck_macs(carried, 3, dl)
    fr = sumcheck_macs(sfc.lookup, 3, dl, terms=3, tables=5)
    lookups_leaves = sfc.lookup
    row = {
        "field": fs.name, "target_log2": target_log2, "extension_degree": d, "challenge_limbs": dl,
        "census": {"mul": sfc.mul, "lin": sfc.lin_total, "lookup": sfc.lookup, "committed_A": sfc.columns_committed_a,
                   "committed_B_estimate": sfc.columns_committed_b, "carried_wires_gkr": carried},
        "matmul_limb_macs": {"gkr_checker": sc["matmul_macs"], "logup_fractional_gkr": fr["matmul_macs"],
                             "encoding_rs_matrix_ntt_per_committed_element": ENCODERS["rs_matrix_ntt"]["macs_per_row_base"] * fs.L**2 / fl.L_PAYLOAD,
                             "encoding_A_committed_elements": sfc.columns_committed_a + sfc.queries_on_computed_wires,
                             "encoding_B_committed_elements": sfc.columns_committed_b + sfc.lookup * d},
        "simt_field_mults_goldilocks_equiv": {"gkr_hadamards": sc["hadamards"] * ec * fs.simt_mult_vs_goldilocks,
                                              "logup_hadamards_and_tree": (fr["hadamards"] + 3 * lookups_leaves) * ec * fs.simt_mult_vs_goldilocks,
                                              "hint_generation_int_ops": HINT_OPS_PER_UNIT},
        "committed_bytes_A": (sfc.columns_committed_a + sfc.queries_on_computed_wires) * fs.element_bytes,
        "committed_bytes_B_estimate": (sfc.columns_committed_b + sfc.lookup * d) * fs.element_bytes,
    }
    enc_a = row["matmul_limb_macs"]["encoding_rs_matrix_ntt_per_committed_element"] * row["matmul_limb_macs"]["encoding_A_committed_elements"]
    row["matmul_limb_macs"]["encoding_A_total"] = enc_a
    row["matmul_limb_macs"]["total_A"] = sc["matmul_macs"] + fr["matmul_macs"] + enc_a
    row["per_vu"] = {k: v * UNITS_PER_VU for k, v in row["matmul_limb_macs"].items() if k.endswith("total_A") or k in ("gkr_checker", "logup_fractional_gkr")}
    row["per_vu"]["committed_bytes_A"] = row["committed_bytes_A"] * UNITS_PER_VU
    row["per_vu"]["committed_bytes_B_estimate"] = row["committed_bytes_B_estimate"] * UNITS_PER_VU
    return row


def design_row(design: Design, fs: FieldSpec, target_log2: int, utilisations: Sequence[float] = (1.0, 0.17),
               hash_rate_key: str = "blake3_simt", encoder: str = "rs_matrix_ntt",
               encoding_utilisation: float | None = None, zk: str = "hvzk", authentication: str = "none",
               weights_public: bool = False) -> dict:
    """One design x field x operating point: limb MACs per VU by component, committed / encoded / hashed bytes per VU,
    times and overheads per GPU x utilisation (``encoding_utilisation`` applies to the encoding component at the
    realistic point only), and the overhead's sensitivity to the Merkle hash rate on the H100 at the realistic point."""
    p = price_design(design, fs, target_log2, encoder=encoder, zk=zk, authentication=authentication, weights_public=weights_public)
    hr = HASH_RATES[hash_rate_key]["bytes_per_s"]
    times = {}
    for g, gpu in GPUS.items():
        for u in utilisations:
            eu = None if u == 1.0 else encoding_utilisation
            times[f"{g}@{u:g}"] = p.times(gpu, g, u, hr, encoder, fs, eu)
    u_real = min(utilisations)
    by_hash = {}
    for hk, hv in HASH_RATES.items():
        rate = hv["bytes_per_s"] if hv["bytes_per_s"] else (GPUS["H100"].simt_field_mults_per_s / POSEIDON2_MULTS_PER_BYTE if hk.startswith("poseidon2") else None)
        if rate:
            by_hash[hk] = p.times(GPUS["H100"], "H100", u_real, rate, encoder, fs, encoding_utilisation)["overhead.vs_native_peak"]
    mm_share = {k: v / p.matmul_total for k, v in p.matmul.items()}
    return {
        "design": design.name, "label": design.label, "field": fs.name, "target_log2": target_log2, "B": B_TARGET, "K": K_TARGET,
        "zk": p.zk, "authentication": p.authentication, "weights_public": weights_public,
        "extension_degree": p.extension_degree, "challenge_limbs": p.extension_degree * fs.L,
        "census.committed_elements": p.committed_elements, "committed_bytes_per_vu": p.committed_bytes / B_TARGET,
        "encoded_bytes_per_vu": p.encoded_bytes / B_TARGET, "hash_bytes_per_vu": p.hash_bytes / B_TARGET,
        "prg_bytes_per_vu": p.prg_bytes / B_TARGET, "communication_bytes_per_vu": p.communication_bytes / B_TARGET,
        "ligero_rows": p.rows,
        "limb_macs_per_vu": p.matmul_total / B_TARGET, "limb_macs_by_component_per_vu": {k: v / B_TARGET for k, v in p.matmul.items()},
        "matmul_share_of_components": mm_share,
        "simt_field_mults_per_vu": p.simt_total / B_TARGET, "simt_by_component_per_vu": {k: v / B_TARGET for k, v in p.simt_field_mults.items()},
        "reduction_int32_words_per_vu": p.reductions_words / B_TARGET,
        "rounds.sequential_depth": p.depth, "hash_rate": hash_rate_key, "encoder": encoder,
        "times": times, "overhead_H100_realistic_by_hash_rate": by_hash, "notes": p.notes,
    }


# ---------------------------------------------------------------------------------------------------------------------
# VOLE / IT-MAC (designated verifier)
# ---------------------------------------------------------------------------------------------------------------------

#: LPN parameters.  Primal LPN with a d-local linear code and regular noise, Wolverine (Weng-Yang-Katz-Wang, S&P 2021,
#: ePrint 2020/925, section 5 / Table): (n, k, t) = (10,805,248; 589,760; 1,319), d = 10, over F_{2^61-1}; also Ferret's
#: family (Yang-Weng-Lan-Katz-Wang, CCS 2020).  The dense *dual* LPN matrix-vector product (n x (n-k) public matrix) is
#: priced as the pure-matmul route for contrast; structured dual codes (quasi-cyclic, expand-accumulate) are O(1)-O(log n)
#: per output and are not priced here (null).
LPN = {"n": 10_805_248, "k": 589_760, "t": 1_319, "d": 10, "field": "F_{2^61-1}", "field_bits": 61,
       "source": "Wolverine ePrint 2020/925 (primal LPN, local linear code, regular noise); Ferret CCS 2020 (regular-noise sVOLE, GGM puncturable PRF)"}


def price_vole(target_log2: int = -128, B: int = B_TARGET, range_mode: str = "logup",
               utilisations: Sequence[float] = (1.0, 0.17)) -> dict:
    """QuickSilver-style online prover + sVOLE expansion in limb MACs (module docstring).  ``designated-verifier``."""
    vu = census.census_vu(K_TARGET)
    emb = vl.embed(vu, range_mode)
    r, bud = vl.first_extension(target_log2, vl._make(K_TARGET, B, range_mode, "quicksilver_arith_1t"))
    Lp = limbs(LPN["field_bits"])
    rl = r * Lp
    checks = emb.mul_checks * B
    corr = emb.vole * B + vl._batch_committed(vl.VoleParams(K=K_TARGET, B=B, mac_extension=r, range_mode=range_mode))
    # online: per check one F_{p^r} x F_{p^r} (M_x M_y) and two F_p x F_{p^r} (x M_y, y M_x): Hadamards on SIMT
    simt_online = checks * (acc.EXT_MULT_COST[r] + 2 * r)
    # batch check: <chi, A0>, <chi, A1>: two inner products of length checks (matmul), plus logUp helper inversions
    mm_batch = 2 * checks * rl * rl
    simt_helpers = 3 * emb.helpers * B
    # sVOLE expansion
    mm_lpn_primal = corr * LPN["d"] * Lp * Lp             # gather (SpMM over the batch); matmul-expressible only as SpMM
    mm_lpn_dual_dense = corr * LPN["k"] * Lp * Lp
    ggm_bytes = corr * 2 * 16                              # GGM tree: ~2 PRG (AES-128) evaluations of 16 B per output
    out = {"zk_mode": "designated-verifier", "proof_class_note": vl.PROOF_CLASS_NOTE, "range_mode": range_mode,
           "mac_extension": r, "mac_field_bits": LPN["field_bits"] * r, "mac_limbs": rl, "soundness_log2": round(acc.log2(bud.total()), 1),
           "per_vu": {"vole_correlations": emb.vole, "mul_checks": emb.mul_checks, "p2v_bytes_quicksilver": emb.p2v_bytes,
                      "p2v_bytes_antman_simd": 21_066_536 / B,
                      "matmul_limb_macs_batch_check": mm_batch / B, "simt_field_mults_online": simt_online / B,
                      "simt_field_mults_helpers": simt_helpers / B,
                      "lpn_primal_d10_limb_macs": mm_lpn_primal / B, "lpn_dual_dense_limb_macs": mm_lpn_dual_dense / B,
                      "ggm_prg_bytes": ggm_bytes / B},
           "per_correlation": {"lpn_primal_d10_macs_F_p": LPN["d"], "lpn_primal_d10_limb_macs": LPN["d"] * Lp * Lp,
                               "lpn_dual_dense_macs_F_p": LPN["k"], "lpn_dual_dense_limb_macs": LPN["k"] * Lp * Lp},
           "lpn": LPN, "times": {}}
    for g, gpu in GPUS.items():
        for u in utilisations:
            t_mm = (mm_batch + mm_lpn_primal) / (gpu.peak_macs_per_s * u)
            t_simt = (simt_online + simt_helpers) / gpu.simt_field_mults_per_s
            t_hash = ggm_bytes / PRG_BYTES_PER_S
            t_hint = B * UNITS_PER_VU * HINT_OPS_PER_UNIT / SIMT_INT_OPS_PER_S[g]
            total = t_mm + t_simt + t_hash + t_hint
            out["times"][f"{g}@{u:g}"] = {
                "t.matmul": t_mm, "t.simt_field": t_simt, "t.prg": t_hash, "t.hint_gen": t_hint, "t.total": total,
                "matmul_share_of_time": t_mm / total, "seconds_per_vu": total / B,
                "overhead.vs_native_peak": contract.overhead(total / B, K_TARGET),
                "overhead_if_dual_dense_lpn": contract.overhead((total + mm_lpn_dual_dense / (gpu.peak_macs_per_s * u)) / B, K_TARGET),
                "transfer_seconds_per_vu_1gbps_quicksilver": emb.p2v_bytes / 125e6,
            }
    out["notes"] = [
        "the online prover is SIMT-bound: M_x M_y and x M_y + y M_x are elementwise products (no matmul form with independent chi_i)",
        "primal d=10 LPN is a gather (SpMM over the batch); the dense dual LPN matmul is 5.9e5 MACs per correlation and not viable",
        "communication is the binding constraint (vole.py): 667 kB per VU QuickSilver, 5 kB per VU with AntMan SIMD (BGV assumption)",
        "field F_{2^61-1}: 8 limbs like Goldilocks; p > 2^60 keeps checker F1 (no small-field repairs)",
    ]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------------

UNCERTAINTIES: tuple[dict, ...] = (
    {"what": "tensor-core utilisation of the sumcheck/encoding kernels", "range": "0.05 - 0.5 (TensorZKP: 0.17 dense, 0.06 gather, HBM-bound)",
     "factor": 3, "moves": "every matmul time linearly; our limbs are 3-4x fewer per element than TensorZKP's 32, so the HBM floor is lower"},
    {"what": "the GKR table size (carried width x units, an upper bound: 3,338 carried wires per unit vs 495 gates)", "range": "1/6 - 1x",
     "factor": 6, "moves": "the largest matmul line of Candidate A; a non-layered (flattened) GKR or gate-indexed tables remove the copy wires"},
    {"what": "non-matmul work: hint generation (2e4 int ops/unit assumed; the CPU anchor is 1.15 ms/unit) and the SIMT field rate (1e11/s)",
     "range": "0.1 - 10x", "factor": 10, "moves": "the non-matmul share; at the CPU anchor hint generation alone is 1.9e6 overhead per VU on 16 cores"},
    {"what": "Merkle hash rate on SIMT (10 - 190 GB/s)", "range": "10 GB/s (pcs.py) to 190 GB/s (TensorZKP-implied)", "factor": 19,
     "moves": "the hash bucket: 1.6 GB (A) / 6.7 GB (B) hashed per batch at 2^-128 Goldilocks"},
)

FLIPS: dict[str, str] = {
    "Goldilocks stays (fields lane)": "FLIPS on the sumcheck side and the encoding side: the 2^-128 challenge fields are 24 limbs for both "
                                      "families (Goldilocks^3 = 192 bits, BabyBear^6 / M31^6 = 186 bits), so challenge x challenge costs 576 MACs "
                                      "either way; trace x challenge is 96 vs 192 and encoding 16 vs 64 MACs per base product, committed bytes "
                                      "halve. The SIMT penalty (EXT_MULT_COST 17 x 0.5 vs 6) that kept Goldilocks does not exist on tensor cores. "
                                      "Caveats: +9% lookups / +4% committed columns (F1/F3/F4 repairs, unverified by z3), degree-6 extensions "
                                      "unshipped in Plonky3/stwo, M31 needs a circle FFT.",
    "Ligero stays (pcs lane)": "holds, with the encoder changed: the RS code is kept but encoded as a two-stage dense matrix DFT on tensor "
                               "cores (O(N sqrt N) MACs, 287 base MACs per symbol); the SIMT NTT was the largest field_ops line and is "
                               "now a small fraction of the sumcheck work. Brakedown's expander code is 14x fewer MACs but gather-bound (9% of peak, TensorZKP encoder).",
    "VOLE out (hash-based only)": "the ruling changed (designated verifier in scope pending LPN); on tensor cores the online prover is "
                                  "SIMT-bound (Hadamards) and the batch check is a tiny matmul; the dense dual-LPN matmul is not viable, the "
                                  "d=10 primal code is a gather. Communication (667 kB/VU) stays the binding constraint.",
    "hashing is the bottleneck (SIMT lanes: commitment dominated every model)": "flips for A when the hash runs at >= 50 GB/s: the hash "
                                  "bucket is below the matmul bucket at 17% utilisation; for B (6.7 GB hashed per batch) it stays the largest bucket "
                                  "below 100 GB/s.",
}


def report() -> dict:
    cal = tensorzkp_calibration()
    util = cal["realistic_utilisation"]["matmul_dense"]
    rows = []
    for tgt in TARGETS:
        for design in DESIGNS.values():
            for fs in FIELDS.values():
                rows.append(design_row(design, fs, tgt, utilisations=(1.0, util)))
    for tgt in TARGETS:
        for dname in ("A", "B"):
            rows.append(design_row(DESIGNS[dname], FIELDS["goldilocks"], tgt, utilisations=(1.0, util), encoder="brakedown_expander",
                                   encoding_utilisation=cal["realistic_utilisation"]["matmul_gather"]))
    ranking = sorted(
        [r for r in rows if r["target_log2"] == -128 and r["encoder"] == "rs_matrix_ntt"],
        key=lambda r: r["times"][f"H100@{util:g}"]["overhead.vs_native_peak"])
    verdict = _field_verdict(rows, util)
    return {
        "meta": {"lane": "explore-tensor", "K": K_TARGET, "B": B_TARGET, "operating_points": TARGETS,
                 "native_seconds_per_vu": contract.native_seconds_per_vu(K_TARGET), "native_reference": contract.NATIVE_REFERENCE,
                 "realistic_utilisation": cal["realistic_utilisation"]},
        "gpus": {k: vars(g) for k, g in GPUS.items()},
        "limb_options": LIMB_OPTIONS,
        "price_list": {str(t): field_price_list(t) for t in TARGETS},
        "unit_price": {str(t): {f: unit_price(fs, t) for f, fs in FIELDS.items()} for t in TARGETS},
        "tensorzkp": TENSORZKP,
        "calibration": cal,
        "hash_rates": HASH_RATES,
        "encoders": ENCODERS,
        "hint_free_branches": HINT_FREE_BRANCHES,
        "rows": rows,
        "ranking_2^-128_H100_realistic": [
            {"design": r["design"], "field": r["field"], "overhead": r["times"][f"H100@{util:g}"]["overhead.vs_native_peak"],
             "overhead_A100_realistic": r["times"][f"A100@{util:g}"]["overhead.vs_native_peak"],
             "overhead_H100_peak": r["times"]["H100@1"]["overhead.vs_native_peak"],
             "limb_macs_per_vu": r["limb_macs_per_vu"], "committed_bytes_per_vu": r["committed_bytes_per_vu"],
             "matmul_share_of_time": r["times"][f"H100@{util:g}"]["matmul_share_of_time"], "depth": r["rounds.sequential_depth"]}
            for r in ranking],
        "field_verdict": verdict,
        "vole": {str(t): price_vole(t, utilisations=(1.0, util)) for t in TARGETS},
        "flips": FLIPS,
        "uncertainties": list(UNCERTAINTIES),
        "assumptions": ASSUMPTIONS,
    }


def _field_verdict(rows: list[dict], util: float) -> dict:
    out = {}
    for tgt in TARGETS:
        for dname in DESIGNS:
            sub = {r["field"]: r for r in rows if r["target_log2"] == tgt and r["design"] == dname and r["encoder"] == "rs_matrix_ntt"}
            g = sub["goldilocks"]["times"][f"H100@{util:g}"]["overhead.vs_native_peak"]
            best = min(sub, key=lambda f: sub[f]["times"][f"H100@{util:g}"]["overhead.vs_native_peak"])
            out[f"{dname}@2^{tgt}"] = {
                "winner": best, "goldilocks_overhead": g,
                "ratio_goldilocks_over_winner": g / sub[best]["times"][f"H100@{util:g}"]["overhead.vs_native_peak"],
                "limb_macs_ratio_goldilocks_over_31bit": sub["goldilocks"]["limb_macs_per_vu"] / sub["babybear"]["limb_macs_per_vu"],
                "committed_bytes_ratio_goldilocks_over_31bit": sub["goldilocks"]["committed_bytes_per_vu"] / sub["babybear"]["committed_bytes_per_vu"],
            }
    return out


ASSUMPTIONS: tuple[str, ...] = (
    ("Every figure is an ESTIMATE from the module docstring's cost model over the WP2 checker census and the fields lane's small-field "
    "deltas; the only measurements are TensorZKP's (H100) and the candidates' recorded CPU runs, used for calibration only."),
    ("A limb product is UINT8 x UINT8 -> INT32; an element of b bits is ceil(b/8) limbs; products are schoolbook convolutions (a matmul "
    "cannot Karatsuba); deferred reduction is one (dL)^2-word SIMT combine per inner product and is not converted to time (TensorZKP: < 2%)."),
    ("Sumcheck: N dl^2 (terms 2^m + 8 tables) matmul MACs and N terms (2^m - 4) SIMT Hadamards over all rounds (TensorZKP Algorithm 1 and "
    "the degree-3 recipe of 5.1); the GKR tables are units x carried width per layer (accounting's CHECKER_LAYER_WIDTHS_A, an upper "
    "bound that carries every lower-depth value); the first round's F_t operands are credited."),
    ("Encoding default: RS n=16384/k=4096 as two-stage dense matrix DFTs (287 base MACs per codeword symbol on tensor cores, k + n twiddle "
    "multiplies per row on SIMT); the Ligero soundness parameters are unchanged.  Brakedown rows use explore.pcs's 20 MACs per symbol."),
    ("Time = matmul MACs / (INT8 dense peak MAC/s x utilisation) + SIMT field mults / 1e11 per s (A100; H100 x 67/19.5; 31-bit x 0.5) + hash "
    "bytes / rate + hint generation (2e4 int ops per unit at 25% of the FP32 peak).  Buckets are summed, no overlap."),
    ("Utilisation 0.17 is TensorZKP's degree-2 sumcheck fraction of the H100 INT8 peak (their kernel is HBM-bound at ~80% of 3.35 TB/s "
    "with 32-limb elements); 1.0 is the datasheet peak.  Our elements are 8-24 limbs, so the HBM floor per element is lower."),
    ("Hint generation on GPU is unmeasured; the CPU anchor (4.7 s per 4,096 units, one thread) is reported beside the estimate."),
    ("VOLE: QuickSilver online costs per multiplication check (1 ext x ext + 2 base x ext Hadamards on SIMT; two length-checks inner products "
    "on tensor cores); LPN as Wolverine's primal d=10 local code (10 F_p MACs per correlation, gather) with the dense dual matmul "
    "(k = 589,760 MACs per correlation) shown for contrast; GGM PRG at 2 x 16 B per correlation."),
)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return f"{v:,}"


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if "--hash-rate" in args:            # override the default Merkle hash rate (GB/s), e.g. the hash-gpu lane's measured value
        i = args.index("--hash-rate")
        gbps = float(args[i + 1])
        del args[i:i + 2]
        HASH_RATES["blake3_simt"] = {"bytes_per_s": gbps * 1e9, "basis": f"--hash-rate {gbps} GB/s (CLI override)"}
        print(f"hash rate: {gbps} GB/s")
    enc_spr = None
    if "--encoder-seconds-per-row" in args:   # measured encoder (b-encode lane): replaces the modelled encoding bucket
        i = args.index("--encoder-seconds-per-row")
        enc_spr = float(args[i + 1])
        del args[i:i + 2]
        print(f"encoder: measured {enc_spr:.3g} s per Ligero row (encoding leaves the matmul bucket)")
        _orig_times = Priced.times

        def _times(self, *a, **kw):
            kw.setdefault("encoder_seconds_per_row", enc_spr)
            return _orig_times(self, *a, **kw)
        Priced.times = _times
    if args and args[0] == "--report":
        from pathlib import Path

        Path(args[1]).write_text(json.dumps(report(), indent=1, default=str) + "\n")
        return
    rep = report()
    util = rep["meta"]["realistic_utilisation"]["matmul_dense"]
    print(f"utilisation (TensorZKP sumcheck): {util}")
    print(json.dumps(rep["calibration"]["our_relation_as_hyperplonk"], indent=1))
    for tgt in TARGETS:
        print(f"\n== 2^{tgt}, K=1536, B=4096, encoder rs_matrix_ntt ==")
        print(f"{'design':22} {'field':10} {'d':>2} {'MACs/VU':>10} {'mm%':>5} {'commit B/VU':>12} {'ovh A100@1':>11} {'A100@u':>9} {'H100@1':>9} {'H100@u':>9} depth")
        for r in rep["rows"]:
            if r["target_log2"] != tgt or r["encoder"] != "rs_matrix_ntt":
                continue
            t = r["times"]
            print(f"{r['design']:22} {r['field']:10} {r['extension_degree']:>2} {_fmt(r['limb_macs_per_vu']):>10} "
                  f"{t[f'H100@{util:g}']['matmul_share_of_time']:5.0%} {_fmt(r['committed_bytes_per_vu']):>12} "
                  f"{_fmt(t['A100@1']['overhead.vs_native_peak']):>11} {_fmt(t[f'A100@{util:g}']['overhead.vs_native_peak']):>9} "
                  f"{_fmt(t['H100@1']['overhead.vs_native_peak']):>9} {_fmt(t[f'H100@{util:g}']['overhead.vs_native_peak']):>9} {r['rounds.sequential_depth']}")
    print("\nfield verdict:", json.dumps(rep["field_verdict"], indent=1))
    v = rep["vole"]["-128"]
    print(f"\nVOLE (designated-verifier) 2^-128 r={v['mac_extension']}:", json.dumps(v["times"], indent=1))


if __name__ == "__main__":       # pragma: no cover
    main()
