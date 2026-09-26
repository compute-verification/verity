"""Lattice scout: cost model for lattice (Module-SIS) proof systems on the frozen checker (lane/lattice-scout, 2026-09-22).

Question (``backends/LANES_OVERNIGHT.md`` section 1): for the exact bounded-integer checker (358 mul / 137 lin / 306
lookups per 16-product unit; 76,808 predicates per K=1536 VU; every honest witness value a non-negative integer below
2^40) and the B = 4096 batch, is there a credible path to lower *complete* prover cost than A-GKR, B-Ligero and the VOLE candidate?  Same
units as ``explore.tensor``: UINT8 x UINT8 -> INT32 limb MACs on tensor cores, SIMT field multiplications, bytes; time
on an H100 at the INT8 dense peak times a stated utilisation; ``overhead.vs_native_peak`` against 9.85 ps per VU.

What a lattice system proves natively.  An Ajtai commitment ``t = A s`` (``A`` uniform in ``R_q^{kappa x n}``,
``R_q = Z_q[X]/(X^d + 1)``, ``s`` short) is binding for short ``s`` under MSIS; it is Z_q-linear, so a commitment to
the batch is one dense GEMM ``A x [s_1 .. s_B]`` (:func:`ajtai_commit_macs`).  The relations proved on top are of two
kinds and the distinction is the whole scout:

* **LaBRADOR-family dot-product constraints** (Beullens-Seiler 2022/1341): ``sum a_ij <s_i, s_j> + sum <phi_i, s_i>
  + b = 0`` over ``R_q`` with a *global* l2-norm bound ``sum ||s_i||^2 <= beta^2`` (proved by a Johnson-Lindenstrauss
  projection with slack sqrt(128/30) ~ 2, Lemma 4.2 / Theorem 5.1).  The quadratic term is a ring-scalar times a full
  inner product; a *coordinate-wise* product ``x_k y_k = z_k`` (every one of the checker's 34,370 products per VU) is
  not of that form: ring products convolve coefficients.  The reduction from R1CS (their Section 6) reaches it by
  committing the operand vectors and using constant-term / automorphism handles, at one ring element per constrained
  coordinate.  The norm bound is *approximate* (slack 2): it certifies binding, never a checker range predicate.
* **Sumcheck-based lattice systems** (Hachi 2026/156, Akita 2026/1983, Neo/SuperNeo 2026/242, Symphony 2025/1905,
  LatticeFold+ 2025/247): the commitment is Ajtai, the constraint check is an ordinary sumcheck over an extension
  field ``E`` of ``Z_q`` (ring switching: Huang-Mao-Zhang 2025 as used by Hachi), and the norm/range part is a digit
  decomposition with a *vanishing-polynomial* (exact) range check proved by sumcheck (Akita 6.1: balanced base-``b``
  digits, degree ``b/2`` after the ``w -> w(w+1)`` symmetry, GKR-style product tree).  Coordinate-wise constraints are
  native (CCS / R1CS), and the checker's ``mod nothing`` equalities embed exactly as long as ``q`` exceeds every
  intermediate (:func:`exact_embedding`).  This is A-GKR's structure with the PCS swapped.

So the only complete lattice design that fits the checker is "A's sumchecks + a lattice PCS"; the LaBRADOR-direct
route (:func:`labrador_direct`) is priced for the record and is dominated by the ``r x N`` aggregation products of its
base protocol at one ring element per wire.  The reusable component is the lattice PCS (Ajtai commit + fold-opening +
digit range check) replacing RS-Ligero + Merkle in A (:func:`design_lattice_pcs` with ``candidate="A"``) or, for B,
replacing the Ligero row tests by a Spartan-style sumcheck over the trace (``candidate="B"``: B stops being B).

Anchors (measured, cited in :data:`LITERATURE`): Greyhound (2024/1293) commit 132 s / prove 41.2 s for N = 2^30 on one
Sapphire Rapids core; Akita (2026/1983, Table 8/9) commit 24.3 s / open 16.1 s one thread, 3.46 s / 2.71 s eight
threads for 2^30 32-bit coefficients on a Ryzen 9 9950X, proofs 61-72 KB, verifier 8-33 ms; LaBRADOR Theorem 5.1
knowledge error 2^-125 per level with ring dimension 64 and a > 2^128 challenge set; Ingonyama's ICICLE LaBRADOR
(GPU, BabyBear x KoalaBear composite q) whose ring choice zkSecurity (2026-04-30) showed collapses soundness to ~31
bits.  Everything else is an ESTIMATE with its assumption stated; ``None`` where no grounding exists.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from verity_numerical.bench import contract
from verity_numerical.explore import tensor as tc
from verity_numerical.security import census

__all__ = [
    "LITERATURE",
    "MSIS",
    "Msis",
    "WitnessBits",
    "ajtai_commit_macs",
    "design_lattice_pcs",
    "digit_range_check_macs",
    "exact_embedding",
    "fold_opening_macs",
    "labrador_direct",
    "main",
    "msis_output_bits",
    "report",
    "witness_bits",
]

K_TARGET = tc.K_TARGET
B_TARGET = tc.B_TARGET
UNITS_PER_VU = tc.UNITS_PER_VU
UTIL = 0.172                      # TensorZKP degree-2 sumcheck fraction of the H100 INT8 peak (tensor.py calibration)

# ---------------------------------------------------------------------------------------------------------------------
# Candidates (literature, with the numbers this lane could ground)
# ---------------------------------------------------------------------------------------------------------------------

LITERATURE: dict[str, dict] = {
    "labrador": {
        "paper": "Beullens, Seiler. LaBRADOR: Compact Proofs for R1CS from Module-SIS. CRYPTO 2023; ePrint 2022/1341",
        "statement": "dot-product constraints over R_q (ring-scalar x full inner products + linear terms) with one global l2 norm bound; R1CS mod 2^64+1 by reduction (Section 6)",
        "assumption": "MSIS over R_q = Z_q[X]/(X^64+1), q ~ 2^32 single precision (high inertia degree required: not NTT-friendly)",
        "setup": "transparent (uniform A, B, C, D)",
        "prover": "O(N) ring multiplications plus r x N aggregation products (h_ij = <phi_i, s_j>), multi-modular NTT (Greyhound library)",
        "proof_kb": (47, 58), "proof_kb_basis": "R1CS mod 2^64+1, 2^10 .. 2^20 constraints, 128-bit",
        "knowledge_error": "2^-125 per recursion level (Theorem 5.1), slack sqrt(128/30) ~ 2 on the extracted norm; ~7 levels for 50 KB",
        "verifier": "linear time", "zk": "not in the paper; LNP22 shim in the LaZer toolkit (Biasioli et al. 2026/1289)",
        "implementation": "lattice-dogs/labrador (C, AVX-512, single-thread; also Greyhound); LaZer; icicle-labrador (GPU, broken ring per zkSecurity 2026-04-30)",
        "prover_seconds_measured": None,
    },
    "greyhound": {
        "paper": "Nguyen, Seiler. Greyhound: Fast Polynomial Commitments from Lattices. CRYPTO 2024; ePrint 2024/1293",
        "statement": "univariate polynomial evaluation (sqrt(N) x sqrt(N) reshape, three-round core) composed with LaBRADOR",
        "assumption": "MSIS, d = 64, q ~ 2^32 multi-precision (q = 5 mod 8), challenges tau_1 = 32 (Section 5)",
        "setup": "transparent",
        "prover_seconds_measured": {"N=2^26": {"commit": 4.37, "prove": 2.03}, "N=2^28": {"commit": 21.2, "prove": 8.21},
                                    "N=2^30": {"commit": 132.0, "prove": 41.2},
                                    "hardware": "one Intel Xeon Sapphire Rapids core at 3.2 GHz, AVX-512 (Table 2)"},
        "proof_kb": (46, 53), "verifier_seconds": {"N=2^30": 2.80}, "zk": "sketched (LWE randomness on outer commitments, L=4 masks; +0.75 KB)",
        "implementation": "github.com/lattice-dogs/labrador (C, AVX-512)",
    },
    "hachi": {
        "paper": "Nguyen, O'Rourke, Zhang. Hachi: Efficient Lattice-Based Multilinear Polynomial Commitments over Extension Fields. ePrint 2026/156",
        "statement": "multilinear evaluation over F_{q^k} reduced to R_q statements; sumcheck over an extension field via ring switching (verifier does no R_q multiplication)",
        "assumption": "MSIS (l_inf), q ~ 2^32, d = 2^10 (5.2), 128-bit via the Lattice Estimator",
        "verifier_ms": {"l=30": 227}, "proof_kb": 55, "commit_vs_greyhound": "3-5x faster (same implementation, Figure 10)",
        "implementation": "georgeorourke/hachi-pcs (Rust prototype, streamed witness)", "prover_seconds_measured": None,
    },
    "akita": {
        "paper": "Dao, Bodaghi, Khajehpour, Vitto, Badakhshan, Georghiades, Liu, Zhang, Thaler. Akita: A High-Performance Lattice-Based PCS. ePrint 2026/1983",
        "statement": "multilinear PCS (K-ary fold to completion, setup offloading), batched openings, pay-per-bit commitment; integrated in Jolt (non-ZK)",
        "assumption": "MSIS at per-matrix ring dimensions (64 / 32 / 16 ...), moduli 2^32-99, 64-bit, 128-bit profiles (Table 5); exact Euclidean norm checks",
        "prover_seconds_measured": {"2^30 x 32-bit coefficients, 1 thread": {"commit": 24.3, "open": 16.1},
                                    "2^30 x 32-bit coefficients, 8 threads": {"commit": 3.46, "open": 2.71},
                                    "hardware": "AMD Ryzen 9 9950X, AVX-512 (Tables 8, 9)"},
        "proof_kb": (61, 72), "verifier_ms": (8.1, 33), "peak_ram_gib_2^35_bits": 4.7,
        "digit_range_check": "balanced base-b digits, w -> w(w+1) symmetry, GKR-style product tree: O(bL) prover (6.1)",
        "zk": "none ('akita and zk cannot be enabled together' in Jolt; listed as future work)",
        "implementation": "Rust (a16z/jolt crates/jolt-akita); GPU/Metal for Jolt, PCS kernels unstated",
        "vs_hash_pcs_8_threads_2^35_bits": {"akita_p32": {"commit": 3.46, "open": 2.71}, "plonky3_fri_koalabear": {"commit": 5.89, "open": 1.24},
                                            "binius64_basefold": {"commit": 5.01, "open": 1.41}, "whir_plonky3": {"commit": 5.53, "open": 8.38}},
    },
    "latticefold_plus": {
        "paper": "Boneh, Chen. LatticeFold+. CRYPTO 2025; ePrint 2025/247 (LatticeFold: 2024/257)",
        "statement": "folding of CCS/R1CS instances over a 64-bit field; algebraic (monomial) range proof; double commitments",
        "assumption": "MSIS", "prover": "5-10x faster than LatticeFold (paper estimate); no public benchmark of a full prover found",
        "proof_kb": "~110 (L=5, estimate: osdnk l2-folding note)", "implementation": "none public found (2026-09)", "prover_seconds_measured": None,
    },
    "neo_superneo": {
        "paper": "Nethermind (Setty et al.). Neo and SuperNeo: post-quantum folding with pay-per-bit costs over small fields. ePrint 2026/242 (Neo 2025/294)",
        "statement": "CCS over an extension field of a small prime, HyperNova-style single sumcheck, Ajtai commitments with pay-per-bit cost, multi-folding",
        "assumption": "MSIS", "prover_seconds_measured": None, "note": "Ajtai hash quoted 30x faster than Pedersen (Boneh's ZKProof7 slides citing Nethermind)",
    },
    "symphony": {
        "paper": "Chen. Symphony: Scalable SNARKs in the ROM from lattice-based high-arity folding. ePrint 2025/1905",
        "statement": "high-arity folding of NP statements, degree-3 sumcheck over an extension field; prover dominated by committing the input witnesses",
        "assumption": "MSIS", "prover_seconds_measured": None,
    },
    "lova": {
        "paper": "Fenzi, Knabenhans, Nguyen, Pham. Lova: lattice folding from unstructured lattices. ePrint 2024/1964",
        "statement": "Nova-style folding with an unstructured SIS commitment, power-of-two moduli, exact Euclidean norm proof",
        "assumption": "plain SIS", "implementation": "Rust", "prover_seconds_measured": None,
        "note": "unstructured A: the commitment is a plain dense Z_{2^k} matmul (ideal tensor-core shape) at the price of a lambda x larger matrix",
    },
    "hyperwolf": {
        "paper": "Zhang, Gao, Xiao. HyperWolf / HyperWolf*: lattice PCS with O(log N) proof and verifier. ePrint 2025/922 (Oct 2025 revision)",
        "statement": "k-dimensional tensor folding of the witness; exact Euclidean norm via <f,f> mod q = <f,f> over Z", "assumption": "MSIS over rings",
        "proof_kb": "~53 (N=2^30)", "prover_seconds_measured": None, "implementation": None,
    },
    "rok_family": {
        "paper": "Klooss, Lai, Nguyen, Osadnik et al.: RoK Paper SISsors (2025), RoK and Roll (2025), SALSAA (2025), RoKoKo (2025)",
        "statement": "verifier-succinct reductions of knowledge with exact norm bounds; RoKoKo measured in Akita Table 8 (8.27 s commit / 4.43 s open at 2^35 bits, 109-115 KB, 100-bit statistical soundness)",
        "assumption": "vanishing-SIS or related structured assumptions for the polylog-verifier results (not plain MSIS)", "prover_seconds_measured": None,
    },
    "zksecurity_labrador_bugs": {
        "source": "Raikos, Wong. Soundness failures in LaBRADOR implementations from NTT-friendly rings. zkSecurity, 2026-04-30",
        "finding": "icicle-labrador (GPU, composite BabyBear x KoalaBear q, fully splitting ring) ~31 bits of soundness; condor-rs ~6; Lazarus ~1; lattice-dogs and LaZer unaffected",
        "consequence": "the NTT-friendly-ring GPU shortcut is not available to LaBRADOR-style aggregation; sumcheck-based systems aggregate over E instead",
    },
}

# ---------------------------------------------------------------------------------------------------------------------
# MSIS parameters (heuristic; the Lattice Estimator replaces this)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Msis:
    """One Module-SIS commitment instance: modulus bits, ring dimension, digit alphabet, and the heuristic output size."""

    name: str
    q_bits: int
    d: int
    digit_bits: int          # balanced base 2^digit_bits digits committed (collision bound 2^digit_bits)
    log_delta: float = 0.005  # root-Hermite factor for ~128-bit classical BKZ (Core-SVP ~ beta 400+; ASSUMPTION)
    basis: str = ""

    @property
    def output_bits(self) -> float:
        return msis_output_bits(self.q_bits, self.digit_bits, self.log_delta)

    @property
    def kappa_d(self) -> int:
        """Output Z_q coordinates (module rank x ring dimension), rounded up to a multiple of d."""
        return math.ceil(self.output_bits / self.q_bits / self.d) * self.d


def msis_output_bits(q_bits: int, digit_bits: int, log_delta: float = 0.005) -> float:
    """HEURISTIC output size (bits of ``kappa d log q``) for an Ajtai commitment binding against collisions with
    ``||z||_inf <= 2^digit_bits``.  An attacker restricts to ``n''`` columns and runs BKZ: the shortest vector found has
    l2 norm ``~ delta^{n''} q^{m/n''}``, minimised at ``n'' = sqrt(m ln q / ln delta)`` where it is
    ``exp(2 sqrt(m ln q ln delta))``; its coordinates are ``~ l2 / sqrt(n'')``.  Binding needs that coordinate size above
    the digit bound, so ``2 sqrt(m ln q ln delta) - ln(sqrt(n'')) > digit_bits ln 2``.  Solved for ``m ln q`` (the output
    bits times ln 2).  Anchor: Akita Table 5 (q = 2^32 - 99, d = 64, rank 1 = 2048 output bits, binary digits over
    65,536 coordinates) is estimated at 147 quantum bits; the raw attack model above gives only ~574 bits for that case
    (it ignores the sub-exponential BKZ cost model and the module structure), so the result is scaled by the constant
    ``ANCHOR_SCALE`` that makes it reproduce the 2048-bit anchor exactly.  Not a security proof: the Lattice Estimator
    replaces it in a build lane; the scale factor (3.6x) is one of the three headline uncertainties."""
    if q_bits <= 0 or digit_bits < 0:
        raise ValueError("q_bits must be positive and digit_bits non-negative")
    ln2 = math.log(2)
    ln_delta = log_delta
    # iterate: n'' depends on m; two passes suffice
    m_lnq = ((digit_bits * ln2 + 2.0) / 2) ** 2 / ln_delta      # start with ln(sqrt(n'')) ~ 2
    for _ in range(4):
        n2 = math.sqrt(m_lnq / ln_delta)
        target = digit_bits * ln2 + 0.5 * math.log(n2)
        m_lnq = (target / 2) ** 2 / ln_delta
    return m_lnq / ln2 * ANCHOR_SCALE


def _raw_anchor() -> float:
    ln2 = math.log(2)
    m_lnq = (2.0 / 2) ** 2 / 0.005
    for _ in range(4):
        n2 = math.sqrt(m_lnq / 0.005)
        m_lnq = (0.5 * math.log(n2) / 2) ** 2 / 0.005
    return m_lnq / ln2


ANCHOR_SCALE: float = 2048.0 / _raw_anchor()  # Akita Table 5 rank-1 d=64 q~2^32 binary map: 2048 output bits (147 qbits)


MSIS: dict[str, Msis] = {
    "q64_chunk16": Msis("q64_chunk16", 64, 64, 16, basis="exact embedding needs q > 2^~52 (exact_embedding); Akita p64 profile; the source is committed in the "
                                                       "checker's own 16-bit range chunks (no source digit range check: the checker's lookups bound them, "
                                                       "binding holds for weak openings), collision bound 2^16"),
    "q64_hex": Msis("q64_hex", 64, 64, 4, basis="as q64_chunk16 but the source is re-decomposed into balanced base-16 digits (Hachi/Akita alphabet): 4x the digits, "
                                                "smaller MSIS output"),
    "q32_chunk16": Msis("q32_chunk16", 32, 64, 16, basis="LaBRADOR/Greyhound/Akita p32 modulus; the checker needs the small-field repairs (F1) and a q = 5 mod 8 prime "
                                                       "(not BabyBear); priced on tensor.py's BabyBear^6 A row"),
}
#: Fold arity of the opening (Akita: any fixed K >= 2; Greyhound: sqrt(N)); the response has N / K coefficients, each a sum of K
#: challenge-weighted source digits, so it carries ``digit_bits + log2(K) + log2(T)`` bits (T ~ 15: LaBRADOR's operator-norm
#: bound) and is re-decomposed into base-``2^RESPONSE_DIGIT_BITS`` digits for the exact range check.  ASSUMPTIONS.
FOLD_ARITY = 64
CHALLENGE_OPNORM_BITS = 4
RESPONSE_DIGIT_BITS = 4


# ---------------------------------------------------------------------------------------------------------------------
# Our relation in their language
# ---------------------------------------------------------------------------------------------------------------------

#: ASSUMPTIONS for column widths the census does not record: looked-up row columns (POW 512 rows x 3, ALIGN 512 x 4,
#: LEAD 31 x 3, NORM 32 x 4 -> outputs up to 32 bits; 16 taken as the mean), query wires on computed values (up to 2^40:
#: 32 taken), B-Ligero's product triples (operands copied: their own widths; products: sum of operand widths ~ 32).
LOOKUP_OUTPUT_BITS = 16
QUERY_WIRE_BITS = 32
PRODUCT_BITS = 32


@dataclass(frozen=True)
class WitnessBits:
    """Bits of bounded integers committed per VU, by origin, and their count in base-``2^digit_bits`` digits."""

    inputs: int
    hints: int
    lookup_outputs: int
    query_wires: int
    products_and_materialised: int

    @property
    def total_a(self) -> int:
        """A-GKR's committed set: inputs + hints + looked-up columns (+ query wires when committed)."""
        return self.inputs + self.hints + self.lookup_outputs + self.query_wires

    @property
    def total_b(self) -> int:
        return self.total_a + self.products_and_materialised

    def digits(self, digit_bits: int, which: str = "A") -> int:
        bits = self.total_a if which == "A" else self.total_b
        return -(-bits // digit_bits)


def witness_bits(vu: census.UnitCensus | None = None, commit_query_wires: bool = True) -> WitnessBits:
    """Bits per VU from the checker census: the exact hint widths (``hint_bits``), 16-bit operand words and one 32-bit
    accumulator word per unit as inputs, looked-up columns and query wires at the stated ASSUMPTION widths."""
    vu = vu or census.census_vu(K_TARGET)
    hints = sum(w * n for w, n in vu.hint_bits.items())
    inputs = 2 * K_TARGET * 16 + 16                       # a, b words and y16; the chained accumulator words are hints/products
    lookups = vu.columns_lookup_outputs * LOOKUP_OUTPUT_BITS
    qw = vu.queries_on_computed_wires * QUERY_WIRE_BITS if commit_query_wires else 0
    prods = (vu.columns_products + vu.columns_materialised) * PRODUCT_BITS
    return WitnessBits(inputs, hints, lookups, qw, prods)


def exact_embedding(vu: census.UnitCensus | None = None) -> dict:
    """How the ``mod nothing`` checker embeds mod q: every predicate is a polynomial identity over Z in bounded
    non-negative integers; it holds over Z iff it holds mod q when q exceeds the largest intermediate.  The widest
    products are the ``lt_pow`` realisations ``x * 2^k < 2^B`` (B <= 40: the 2^40 bound of the brief) and hint x hint
    products of <= 25-bit hints (50 bits); a linear identity sums a handful (<= 16, the group width) of such terms.
    So q > 2^(40 + 4) .. 2^(50 + 4): a 64-bit prime is enough with no range-check machinery for the wrap; a 32-bit q
    needs the fields lane's small-field realisations (checker F1) exactly as BabyBear does."""
    vu = vu or census.census_vu(K_TARGET)
    widest_hint = max(vu.hint_bits)
    return {
        "witness_bound_bits": 40, "widest_hint_bits": widest_hint, "widest_product_bits": max(40, 2 * widest_hint),
        "terms_per_identity_bound": 16, "q_bits_needed": max(40, 2 * widest_hint) + 4,
        "q_64_exact": True, "q_32_exact": False,
        "range_predicates_per_vu": vu.range_lookups, "fixed_lookups_per_vu": vu.fixed_lookups,
        "norm_bound_replaces_range_checks": False,
        "why": "a lattice norm bound is global (one l2 or l_inf bound, extracted with slack ~2 by JL or exact by <f,f> lifting) and certifies "
               "binding; the checker's 27k per-value range predicates per VU are exact per-coordinate statements and stay constraints "
               "(digit vanishing polynomials or the lookup argument).  Nothing in the checker is cheaper because the witness is short, "
               "except the commitment itself (pay-per-bit).",
    }


# ---------------------------------------------------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------------------------------------------------


def ajtai_commit_macs(digits: int, m: Msis, digit_bits: int | None = None) -> dict:
    """Ajtai commitment of ``digits`` base-``2^digit_bits`` digits as a dense Z_q GEMM: the negacyclic block matrix
    ``A`` (``kappa d x n d``, entries ``q_bits`` = ``ceil(q_bits/8)`` limbs) times the digit vector (``ceil(digit_bits/8)``
    8-bit limbs per digit).  Per digit: ``kappa d x limbs(q) x limbs(digit)`` limb MACs.  With the batch as the GEMM's
    N dimension (A shared by all 4096 VUs) this is the best-shaped matmul in the whole design space (K = n d up to 2^15
    per INT32 chunk, M = kappa d, N = 4096)."""
    lq = tc.limbs(m.q_bits)
    ld = tc.limbs(m.digit_bits if digit_bits is None else digit_bits)
    return {"limb_macs": digits * m.kappa_d * lq * ld, "per_digit": m.kappa_d * lq * ld, "kappa_d": m.kappa_d,
            "output_bytes": m.kappa_d * m.q_bits / 8, "shape": f"({m.kappa_d}) x ({digits}) x B, INT8 dense GEMM",
            "note": "NTT-domain evaluation would turn the short digits into full Z_q elements (lq x lq per product): the coefficient-domain GEMM keeps pay-per-bit"}


def fold_opening_macs(digits: int, dl: int, digit_limbs: int, degree: int = 2) -> dict:
    """Opening / folding proof (Hachi-Akita style K-ary fold with sumcheck, iterated to completion): total table size
    over the folds is a geometric series dominated by the first, so priced as ONE degree-``degree`` sumcheck over the
    ``digits`` (``tensor.sumcheck_macs``), with the first round's base-field operand credited as in tensor.py.
    Unpriced: Johnson-Lindenstrauss projections (additions), digit re-decomposition of the response (SIMT), the tail
    LaBRADOR levels (small)."""
    sc = tc.sumcheck_macs(digits, degree, dl)
    credit = digits * (dl * dl - digit_limbs * dl) * (2**degree) // 2
    return {"limb_macs": sc["matmul_macs"] - credit, "hadamards": sc["hadamards"], "rounds": math.ceil(math.log2(max(2, digits)))}


def response_digits(source_digits: int, source_digit_bits: int, fold_arity: int = FOLD_ARITY,
                    response_digit_bits: int = RESPONSE_DIGIT_BITS) -> dict:
    """The folded response ``z = sum_i c_i s_i`` has ``source_digits / fold_arity`` coefficients of
    ``source_digit_bits + log2(fold_arity) + CHALLENGE_OPNORM_BITS`` bits, re-decomposed into base-``2^response_digit_bits``
    digits: those are what the exact range check (and the response commitment) act on."""
    coeff_bits = source_digit_bits + math.ceil(math.log2(fold_arity)) + CHALLENGE_OPNORM_BITS
    per_coeff = -(-coeff_bits // response_digit_bits)
    coeffs = -(-source_digits // fold_arity)
    return {"coefficients": coeffs, "coefficient_bits": coeff_bits, "digits_per_coefficient": per_coeff, "digits": coeffs * per_coeff}


def digit_range_check_macs(digits: int, digit_bits: int, dl: int) -> dict:
    """Exact range check of ``digits`` response digits (Akita 6.1 / Hachi): the vanishing polynomial of the balanced
    alphabet has degree ``b = 2^digit_bits``; the ``w -> w(w+1)`` symmetry halves it to ``b/2`` factors per digit; a
    GKR-style product tree over those factors has ``~2 (b/2)`` nodes per digit in ``log2(b/2)`` layers of degree-2..3
    sumchecks (Akita's O(bL)).  Priced as ``b x digits`` elements of a degree-3 sumcheck (``tensor.sumcheck_macs(m=3)``),
    the leaf layer credited for base-field operands (digits are 1 limb).  This is the tensor-core price of running the
    tree as matmuls; Akita's small-value SIMT techniques are not priced."""
    b = 2**digit_bits
    layers = max(1, math.ceil(math.log2(b / 2)))
    elems = b * digits
    sc = tc.sumcheck_macs(elems, 3, dl)
    credit = (b // 2) * digits * (dl * dl - dl) * 8 // 2
    return {"limb_macs": sc["matmul_macs"] - credit, "hadamards": sc["hadamards"], "layers": layers, "vanishing_degree": b // 2, "elements": elems}


# ---------------------------------------------------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------------------------------------------------


def _a_row(field: str = "babybear", target: int = -128) -> dict:
    return tc.design_row(tc.DESIGNS["A"], tc.FIELDS[field], target, utilisations=(1.0, UTIL))


def design_lattice_pcs(msis_key: str = "q64_chunk16", candidate: str = "A", target_log2: int = -128,
                       commit_query_wires: bool = True, fold_arity: int = FOLD_ARITY) -> dict:
    """A-GKR (or B's trace) with the RS-Ligero + Merkle commitment replaced by a lattice PCS: Ajtai commitment
    of the source digits, K-ary fold opening (sumcheck), Ajtai commitment of the response, exact digit range check of
    the response.  Everything that is not the commitment layer is taken from ``explore.tensor``'s A row on the field
    the modulus dictates (the sumcheck field must contain Z_q): Goldilocks^3 for a 64-bit q, BabyBear^6 for a 32-bit q
    (with the caveat that the actual prime must be q = 5 mod 8, not BabyBear).  For ``candidate="B"`` the Ligero row
    tests are replaced by a Spartan-style degree-3 sumcheck over the committed trace (B's linear + quadratic
    constraints), depth ~ log2(rows): B becomes an explicit-trace A."""
    m = MSIS[msis_key]
    sumcheck_field = "goldilocks" if m.q_bits >= 64 else "babybear"
    a = _a_row(sumcheck_field, target_log2)
    fs = tc.FIELDS[sumcheck_field]
    dl = a["challenge_limbs"]
    vu = census.census_vu(K_TARGET)
    wb = witness_bits(vu, commit_query_wires)
    digits = wb.digits(m.digit_bits, candidate)
    commit = ajtai_commit_macs(digits, m)
    fold = fold_opening_macs(digits, dl, tc.limbs(m.digit_bits))
    resp = response_digits(digits, m.digit_bits, fold_arity)
    resp_commit = ajtai_commit_macs(resp["digits"], m, RESPONSE_DIGIT_BITS)
    rng = digit_range_check_macs(resp["digits"], RESPONSE_DIGIT_BITS, dl)
    macs = {"ajtai_commit": commit["limb_macs"], "ajtai_response_commit": resp_commit["limb_macs"],
            "fold_opening_sumcheck": fold["limb_macs"], "digit_range_check": rng["limb_macs"]}
    simt = {"fold_hadamards": fold["hadamards"] * tc.acc.EXT_MULT_COST[a["extension_degree"]],
            "range_hadamards": rng["hadamards"] * tc.acc.EXT_MULT_COST[a["extension_degree"]],
            "digit_decomposition_int_ops": digits + resp["digits"]}
    if candidate == "A":
        for k in ("gkr_checker", "logup_fractional_gkr"):
            macs[k] = a["limb_macs_by_component_per_vu"][k]
        for k in ("gkr_hadamards", "logup_hadamards", "logup_tree_build"):
            simt[k] = a["simt_by_component_per_vu"][k]
        depth = a["rounds.sequential_depth"] - 2 + fold["rounds"] + rng["layers"] * 2
    else:
        # Spartan-style: one degree-3 sumcheck over the constraint rows (eq x Az x Bz - eq x Cz: four tables) plus the sparse
        # second phase (nnz ~ 4 per constraint, F_c x small on SIMT); the lookups need the fractional-sum argument (A's logUp GKR)
        elems = vu.mul + vu.lin_total
        sc = tc.sumcheck_macs(elems, 3, dl, tables=4)
        macs["spartan_constraint_sumcheck"] = sc["matmul_macs"] - elems * (dl * dl - fs.L * dl) * 4
        macs["logup_fractional_gkr"] = a["limb_macs_by_component_per_vu"]["logup_fractional_gkr"]
        simt["spartan_hadamards"] = sc["hadamards"] * tc.acc.EXT_MULT_COST[a["extension_degree"]]
        simt["spartan_sparse_phase"] = 4 * elems * a["extension_degree"]
        for k in ("logup_hadamards", "logup_tree_build"):
            simt[k] = a["simt_by_component_per_vu"][k]
        depth = 2 * math.ceil(math.log2(max(2, elems * B_TARGET))) + fold["rounds"] + rng["layers"] * 2 + 40
    mm = sum(macs.values())
    t_mm = mm * B_TARGET / (tc.GPUS["H100"].peak_macs_per_s * UTIL)
    simt_field = sum(v for k, v in simt.items() if not k.endswith("int_ops"))
    t_simt = simt_field * fs.simt_mult_vs_goldilocks * B_TARGET / tc.GPUS["H100"].simt_field_mults_per_s
    t_hint = B_TARGET * UNITS_PER_VU * tc.HINT_OPS_PER_UNIT / tc.SIMT_INT_OPS_PER_S["H100"]
    total = t_mm + t_simt + t_hint
    ref = a["times"][f"H100@{UTIL:g}"]
    caveats = [
        ("the sumcheck field must contain Z_q: with q = 64-bit the checker sumcheck runs on 8-limb elements (priced on tensor.py's A/Goldilocks^3 row); "
        "with q = 32-bit the prime must be q = 5 mod 8 (LS18 invertible short challenges) or use RoKoKo-style probabilistic invertibility: "
        "BabyBear (fully splitting) is not the LaBRADOR/Greyhound choice; the small-field repairs (F1/F3/F4) are re-derived for the new prime "
        "(priced on the BabyBear^6 row)"),
        ("no range check on the source digits: they are the checker's own 16-bit range chunks (bounded by its lookups) and binding holds for weak "
        "openings; the exact range check acts on the folded response (Akita 6.1).  A one-shot PCS needs no norm proof on the source; a "
        "recursive/IVC use (LatticeFold) would"),
        ("knowledge error: LaBRADOR Theorem 5.1 gives 2^-125 per level at d = 64 (challenge set > 2^128 before operator-norm rejection); "
        "Akita reaches its 128-bit budget with per-fold accounting; a 2^-128 batch bound over the union of folds, range layers and the "
        "checker sumcheck is NOT established here: 2^-120 .. 2^-125 is what the papers deliver at d = 64"),
        ("ZK: Akita has none; Greyhound sketches LWE-randomised outer commitments; LaBRADOR needs the LaZer LNP22 shim (linear size). "
        "Malicious-verifier ZK adds MLWE to the assumption set and is unbuilt in every implementation found"),
        "utilisation 0.172 applied to the fold and range sumchecks (dense, as TensorZKP's); the Ajtai GEMM would run far above it",
    ]
    if candidate == "B":
        caveats.insert(0, "B under a lattice PCS has no Ligero row tests: the constraints are checked by a Spartan-style sumcheck, depth 3 -> ~log2(N) + folds; "
                          "this is an explicit-trace GKR-less A, not Candidate B")
    return {
        "design": f"{candidate}+lattice-PCS", "msis": m.name, "q_bits": m.q_bits, "d": m.d, "digit_bits": m.digit_bits,
        "kappa_d": m.kappa_d, "msis_output_bits_heuristic": round(m.output_bits), "sumcheck_field": sumcheck_field,
        "extension_degree": a["extension_degree"], "challenge_limbs": dl, "target_log2": target_log2, "B": B_TARGET, "K": K_TARGET,
        "fold_arity": fold_arity, "witness_bits_per_vu": vars(wb), "digits_per_vu": digits, "committed_bytes_per_vu": digits * m.digit_bits / 8,
        "response": resp, "range_check": {k: v for k, v in rng.items() if k != "limb_macs"},
        "commitment_bytes_per_commitment": commit["output_bytes"],
        "limb_macs_by_component_per_vu": macs, "limb_macs_per_vu": mm,
        "simt_by_component_per_vu": simt, "hash_bytes_per_vu": 0.0,
        "rounds.sequential_depth": depth,
        "proof_kb_per_batch_opening": "61-72 (Akita Table 8) per opening; plus the checker sumcheck transcript",
        "times_H100@0.172": {"t.matmul": t_mm, "t.simt_field": t_simt, "t.hash": 0.0, "t.hint_gen": t_hint, "t.total": total,
                             "matmul_share_of_time": t_mm / total, "seconds_per_vu": total / B_TARGET,
                             "overhead.vs_native_peak": contract.overhead(total / B_TARGET, K_TARGET)},
        "reference_row": {"design": a["design"], "field": a["field"], "overhead_H100@0.172": ref["overhead.vs_native_peak"],
                          "t.total": ref["t.total"], "t.hash": ref["t.hash"],
                          "encoding_plus_ligero_macs_per_vu": a["limb_macs_by_component_per_vu"]["encoding"] + a["limb_macs_by_component_per_vu"]["ligero_row_combination"],
                          "by_hash_rate": a["overhead_H100_realistic_by_hash_rate"]},
        "removed_from_reference": ["Reed-Solomon encoding (matrix DFT)", "Ligero row combination", "Merkle hashing (SIMT Blake3)"],
        "added": list(macs.keys())[:4] + ["digit decomposition (SIMT int ops)", "JL / norm lifting (unpriced, additions)"],
        "caveats": caveats,
    }


def labrador_direct(msis_key: str = "q32_chunk16", wires_per_poly: int = 1, r_exponent: float = 1 / 3) -> dict:
    """LaBRADOR base protocol on the checker written as dot-product constraints, one ring element per constrained wire
    (``wires_per_poly = 1``: coordinate-wise products need the coefficient isolated; ``d`` would pack d wires but then
    products convolve and the checker's 34k coordinate products per VU are no longer expressible without automorphism
    expansions, which multiply the constraint count by d).  N polys = columns_all x B; multiplicity r = N^r_exponent
    (the paper's amortisation); the aggregation products ``h_ij = <phi_i, s_j>`` cost ``r^2 n = r N`` ring products of
    a full Z_q polynomial by a short one; a ring product in the coefficient domain is d^2 scalar MACs (d log d + d via
    NTT, but q must NOT be NTT-friendly for the aggregation to be sound: zkSecurity 2026-04-30), each ``lq x 1`` limbs.
    Ajtai commitments: kappa x N ring products.  Garbage ``g_ij = <s_i, s_j>``: r^2 n / 2 short x short products."""
    m = MSIS[msis_key]
    vu = census.census_vu(K_TARGET)
    polys_per_vu = -(-vu.columns_all // wires_per_poly)
    N = polys_per_vu * B_TARGET
    r = max(2, round(N**r_exponent))
    n = -(-N // r)
    lq = tc.limbs(m.q_bits)
    ring_mul_scalar = m.d * m.d
    h = r * r * n * ring_mul_scalar * lq * 1
    g = r * r * n // 2 * ring_mul_scalar * 1 * 1
    commit = (m.kappa_d // m.d) * N * ring_mul_scalar * lq
    z = r * n * ring_mul_scalar * 1
    total = h + g + commit + z
    t = total / (tc.GPUS["H100"].peak_macs_per_s * UTIL)
    per_vu = t / B_TARGET
    return {
        "design": "LaBRADOR-direct (dot-product constraints, base protocol only)", "msis": m.name, "polys_per_vu": polys_per_vu, "N_polys": N,
        "r": r, "n": n, "limb_macs_per_vu": total / B_TARGET,
        "limb_macs_by_component_per_vu": {"h_ij_aggregation": h / B_TARGET, "g_ij_garbage": g / B_TARGET, "ajtai_commit": commit / B_TARGET, "z_response": z / B_TARGET},
        "overhead_H100@0.172": contract.overhead(per_vu, K_TARGET), "seconds_per_vu": per_vu,
        "greyhound_cpu_anchor": {"ns_per_coefficient_single_core": (132.0 + 41.2) / 2**30 * 1e9,
                                 "seconds_per_vu_if_one_wire_per_poly": polys_per_vu * m.d * (132.0 + 41.2) / 2**30,
                                 "overhead_cpu_single_core": contract.overhead(polys_per_vu * m.d * (132.0 + 41.2) / 2**30, K_TARGET),
                                 "basis": "Greyhound Table 2, N = 2^30 coefficients, commit + prove on one Sapphire Rapids core; a PCS opening, not a constraint proof"},
        "notes": [
            "the r x N term is the base protocol's amortisation cost; recursion levels are smaller and not priced",
            "one wire per ring element wastes d - 1 coefficients of every commitment and NTT; packing d wires per element breaks coordinate-wise products",
            "not a candidate: the sumcheck-based route (design_lattice_pcs) is the only lattice construction that fits the checker's constraint shape",
        ],
    }


# ---------------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------------

UNCERTAINTIES: tuple[dict, ...] = (
    {"what": "digit range check + fold opening as tensor-core sumchecks (priced as degree-3 / degree-2 sumchecks over 2x / 1x the digits at 0.172 utilisation)",
     "range": "0.3x - 3x", "factor": 3, "moves": "the lattice additions (~40% of the PCS-swap delta); Akita's O(bL) SIMT count and small-value tricks may be cheaper, a proper accounting of all folds costlier"},
    {"what": "Merkle hash rate on SIMT in the reference (what the lattice PCS removes)", "range": "10 - 190 GB/s (tensor.HASH_RATES)", "factor": 19,
     "moves": "the saving: at 10 GB/s A/Goldilocks is 2.5e7 and the swap wins 2.7x; at 190 GB/s A's hash bucket is 0.03 s per batch and the swap loses"},
    {"what": "MSIS output size and modulus (heuristic kappa d; q = 64-bit for exactness vs 32-bit with small-field repairs; d = 64 vs larger rings for the challenge set)",
     "range": "0.3x - 3x on the Ajtai GEMM (the raw BKZ model under-shoots the Akita anchor 3.6x and is scaled to it); +17% on the checker sumcheck if the field must be 64-bit", "factor": 4,
     "moves": "commitment MACs (7% of the row at kappa d = 896) and whether the checker sumcheck stays on 4-limb elements"},
)


def report() -> dict:
    vu = census.census_vu(K_TARGET)
    wb = witness_bits(vu, True)
    rows = [design_lattice_pcs(k, "A") for k in ("q64_chunk16", "q32_chunk16", "q64_hex")]
    rows.append(design_lattice_pcs("q64_chunk16", "A", commit_query_wires=False))
    rows.append(design_lattice_pcs("q32_chunk16", "A", fold_arity=16))
    rows.append(design_lattice_pcs("q32_chunk16", "A", fold_arity=256))
    rows.append(design_lattice_pcs("q64_chunk16", "B"))
    lab = labrador_direct()
    best = min(rows, key=lambda r: r["times_H100@0.172"]["overhead.vs_native_peak"])
    a_ref = _a_row("babybear", -128)["times"][f"H100@{UTIL:g}"]
    return {
        "meta": {"lane": "lattice-scout", "date": "2026-09-22", "K": K_TARGET, "B": B_TARGET, "target_log2": -128, "utilisation": UTIL,
                 "native_seconds_per_vu": contract.native_seconds_per_vu(K_TARGET)},
        "literature": LITERATURE,
        "msis": {k: {**{f: getattr(m, f) for f in ("name", "q_bits", "d", "digit_bits", "log_delta", "basis")},
                     "output_bits_heuristic": round(m.output_bits), "kappa_d": m.kappa_d} for k, m in MSIS.items()},
        "our_relation": {"census_vu": vu.summary(), "witness_bits_per_vu": vars(wb), "digits_per_vu_chunk16_A": wb.digits(16, "A"),
                         "digits_per_vu_chunk16_B": wb.digits(16, "B"), "exact_embedding": exact_embedding(vu),
                         "ajtai_commit_A_q64_chunk16": ajtai_commit_macs(wb.digits(16, "A"), MSIS["q64_chunk16"])},
        "rows": rows,
        "labrador_direct": lab,
        "best": {"design": best["design"], "msis": best["msis"], "fold_arity": best["fold_arity"],
                 "overhead_H100@0.172": best["times_H100@0.172"]["overhead.vs_native_peak"],
                 "reference_A_babybear_overhead": a_ref["overhead.vs_native_peak"],
                 "ratio_vs_A_babybear": best["times_H100@0.172"]["overhead.vs_native_peak"] / a_ref["overhead.vs_native_peak"],
                 "ratio_vs_own_field_reference": best["times_H100@0.172"]["overhead.vs_native_peak"] / best["reference_row"]["overhead_H100@0.172"],
                 "B_reference_overhead": 1.434e7, "C_reference_overhead": 3.1e6,
                 "soundness_reached": "2^-120 .. 2^-125 at d = 64 (LaBRADOR Thm 5.1 per level; Akita per-fold budget) -- not 2^-128 without a larger ring or repetition"},
        "verdict": VERDICT,
        "uncertainties": list(UNCERTAINTIES),
        "assumptions": ASSUMPTIONS,
    }


VERDICT: dict = {
    "complete_alternative_backend": "NO. Every lattice system that fits the checker's coordinate-wise quadratic constraints is 'a sumcheck over an "
                                    "extension field + an Ajtai-committed witness' (Hachi/Akita/Neo/Symphony/LatticeFold+); its prover is A's prover with "
                                    "the commitment layer swapped, so it cannot beat A by more than A's encoding + hashing buckets (20% of A at 50 GB/s) "
                                    "minus its own fold and digit range check.  The LaBRADOR dot-product route is 100x+ worse at one wire per ring element.  "
                                    "C (3.1e6, designated verifier) is untouched.",
    "reusable_component": "MAYBE, conditional: a lattice PCS (Ajtai GEMM on the checker's own 16-bit chunks + Akita-style K-ary fold + exact digit range "
                          "check on the response) replacing RS-Ligero + Merkle in A is hash-free; its commitment is the best-shaped INT8 GEMM in the "
                          "design space (kappa d x digits x 4096, 6.3e8 limb MACs/VU at the Akita-anchored kappa d = 896) and the model puts A+lattice-PCS at 1.01x A (32-bit q, BabyBear "
                          "row, 1.08e7) to 0.74x A (64-bit q, Goldilocks^3 row, 9.3e6 vs 1.25e7) at 50 GB/s Blake3 and fold arity 64 -- inside the hash-rate uncertainty "
                          "(A at 10 GB/s is 2.5e7: the swap wins 2.7x; at 190 GB/s A is ~9.5e6: the swap loses) and inside the fold-arity sensitivity "
                          "(K = 16: 1.46e7, K = 256: 9.6e6).  Costs: +~20 sequential rounds, MSIS (+MLWE for hiding) added to the assumption set, no ZK "
                          "in any implementation, knowledge error ~2^-120..-125 at d = 64, a 64-bit q for exactness or a q = 5 mod 8 32-bit prime with "
                          "the small-field repairs redone.  For B it is not a component: B loses its row tests and becomes an explicit-trace Spartan "
                          "design (the 7.9e6 of that row is the flattening of A's layered GKR, obtainable with Ligero too, not a lattice effect).",
    "follow_up_first_build": "Nothing before the GPU Blake3/Poseidon2 hash rate is measured (tensor.md uncertainty 2): the whole case for the swap is "
                             "the hash bucket.  If it lands below ~30 GB/s: (1) run jolt-akita (Rust, 32-bit profile) on A's committed object "
                             "(44k 16-bit chunks/VU x 4096) on vy-cpu2 for a measured commit+open per VU beside the Ligero CPU number; (2) write the "
                             "INT8 Ajtai GEMM kernel (A shared across the batch) and measure its utilisation on the L40S; (3) price the response range "
                             "check against logUp on the same digits with a real fold schedule (Akita's planner).  No lattice lane before that.",
}

ASSUMPTIONS: tuple[str, ...] = (
    ("Every figure is an ESTIMATE from this module's docstrings over the WP2 checker census; the only measured numbers are the cited papers' CPU "
    "benchmarks (Greyhound Table 2, Akita Tables 8-9) and tensor.py's TensorZKP calibration (utilisation 0.172)."),
    "Widths not in the census: looked-up columns 16 bits, query wires 32 bits, products/materialised 32 bits (LOOKUP_OUTPUT_BITS, QUERY_WIRE_BITS, PRODUCT_BITS).",
    ("MSIS output size from the heuristic in msis_output_bits (BKZ root-Hermite 1.005, column-restriction attack), rounded up to a multiple of d; the Lattice "
    "Estimator replaces it.  Scaled (ANCHOR_SCALE = 3.57) to reproduce Akita Table 5: rank-1 d=64 q=2^32 binary map = 2048 output bits at 147 quantum bits."),
    "Ajtai commitment as a coefficient-domain INT8 GEMM (A entries ceil(q_bits/8) limbs, digits 1 limb); response commitment at 2x the input commitment.",
    ("Fold opening = one degree-2 sumcheck over the digits (first-fold dominated); digit range check = degree-3 sumcheck over 2x the digits (product tree); "
    "both at 0.172 of the H100 INT8 peak; Hadamards on SIMT at tensor.py's rates; hint generation as tensor.py."),
    "The non-commitment buckets (GKR checker, logUp fractional GKR, Hadamards) are tensor.py's Candidate A BabyBear^6 row unchanged; the field caveat is stated per row.",
    "LaBRADOR-direct: r = N^(1/3), one wire per ring element, ring product d^2 scalar MACs (no NTT: q must not split), base protocol only.",
)


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    rep = report()
    if args and args[0] == "--report":
        from pathlib import Path

        Path(args[1]).write_text(json.dumps(rep, indent=1, default=str) + "\n")
        return
    print(json.dumps(rep["our_relation"]["witness_bits_per_vu"], indent=1))
    print(f"{'design':14} {'msis':12} {'K':>4} {'field':10} {'digits/VU':>9} {'MACs/VU':>9} {'mm%':>5} {'depth':>5} {'ovh H100@0.172':>15} {'ref (field)':>11}")
    for r in rep["rows"]:
        t = r["times_H100@0.172"]
        print(f"{r['design']:14} {r['msis']:12} {r['fold_arity']:>4} {r['sumcheck_field']:10} {r['digits_per_vu']:>9,} {r['limb_macs_per_vu']:>9.3g} "
              f"{t['matmul_share_of_time']:5.0%} {r['rounds.sequential_depth']:>5} {t['overhead.vs_native_peak']:>15.3g} "
              f"{r['reference_row']['overhead_H100@0.172']:>11.3g}")
        print(f"{'':14} {json.dumps({k: f'{v:.2g}' for k, v in r['limb_macs_by_component_per_vu'].items()})}")
    lab = rep["labrador_direct"]
    print(f"\nLaBRADOR-direct: r={lab['r']} N={lab['N_polys']:,} MACs/VU={lab['limb_macs_per_vu']:.3g} overhead={lab['overhead_H100@0.172']:.3g}; "
          f"Greyhound CPU anchor overhead={lab['greyhound_cpu_anchor']['overhead_cpu_single_core']:.3g}")
    print("\nbest:", json.dumps(rep["best"], indent=1))


if __name__ == "__main__":       # pragma: no cover
    main()
