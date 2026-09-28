"""Ligerito parameter calculator: proof size, soundness, interactive coins, prover / verifier op counts.

Ligerito (Novakovic-Angeris 2025, eprint 2025/1187) is recursive Ligero.  Level ``i`` (``i = 1..L``) commits a matrix
``M_i`` of ``tall_i`` rows x ``wide_i = 2^{k'_i}`` columns by encoding every COLUMN with a rate-``rho_i`` Reed-Solomon code
of length ``n_i`` (a Merkle tree over the ``n_i`` codeword ROWS, each ``wide_i`` symbols).  A partial sumcheck over the
``k'_i`` "wide" variables reduces the running inner-product claim to ``v_i = M_i r_i`` (length ``tall_i``), which is
reshaped into ``M_{i+1}`` (``tall_{i+1} x wide_{i+1}``, ``tall_{i+1} wide_{i+1} = tall_i``) and committed at the next
level; the verifier opens ``|S_i|`` codeword rows of level ``i`` and checks them against the *expected symbols*
``y_i = (G_i v_i)_{S_i}`` that the prover sends (an inner product with ``M_{i+1}``, batched into the next level's
sumcheck).  The last vector ``v_L`` is sent in the clear.  The paper's ``l`` is ``L + 1`` here ("``l = 2`` is one
commitment").

Sizes (paper (19)): level ``i`` costs ``|S_i| wide_i`` field elements (opened rows) + ``|S_i|`` Merkle multi-opening
paths + ``|S_i|`` expected symbols + ``3 k'_i`` sumcheck coefficients; the final vector costs ``tall_L`` elements.  The
level-1 matrix is over the BASE field (the witness: BabyBear, 4 B; the paper's F_2^32), everything after level 1 is
over the challenge field (BabyBear^6, 24 B; the paper's F_2^128, 16 B).

Soundness (paper (17), Reed-Solomon in the unique-decoding regime, ``e_i = (d_i - 1)/2 = (n_i - tall_i)/2``):

    per level i:  ((n_i - e_i)/n_i)^{|S_i|} = ((1 + rho_i)/2)^{|S_i|}      proximity (query) term
                + k'_i n_i / |F|                                             DP24 / DG24 log-randomness field term
                + 2 k'_i / |F|                                               partial sumcheck, degree 2, k'_i rounds
                + (#batched claims) / |F|                                    batching the y_{i-1} claims (i >= 2)

all over the challenge field ``|F| = p^6 ~ 2^186``; the query terms are what the parameters are sized against (the
field terms are 2^-150 and below).  The relation part (zero-check sumcheck etc., ``RelationShape``) adds its own terms;
``soundness()`` reports every term and the union.

Cells: our witness is ``rows_pad x cols`` with ``rows_pad = 4096`` (3769 real rows for fp8-ada: the pad rows are the
ZK mask rows) and ``cols = units per VU x VUs = 3 x 2^16`` (fp8-ada) or ``3 x 2^17`` (the BF16 relations): a factor 3
from K = 1536.  ``radix3=True`` keeps the 3 in the tall dimension (one ternary sumcheck variable; the RS code length is
a power of two anyway, so the actual rate is ``3/16`` at nominal 1/4), ``radix3=False`` pads the columns to ``2^18``.

Every estimate shows its formula in the returned dict; the throughput constants (``GPU``) are inputs, not results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

__all__ = ["BABYBEAR_P", "Fields", "OUR_FIELDS", "PAPER_FIELDS", "LigeritoParams", "RelationShape", "RELATIONS",
           "expected_merkle_siblings", "queries_for", "optimize", "paper_dims", "paper_table1", "our_params", "table2_cells",
           "ligero_today_ops"]

BABYBEAR_P = 15 * 2**27 + 1
KiB = 1024


# ---------------------------------------------------------------------------------------------------------------------
# Fields and hashes
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Fields:
    """Element sizes.  ``base_bytes``: the committed witness field (level 1 elements); ``ext_bytes``: the challenge field
    (levels >= 2, sumcheck messages, expected symbols, the final vector); ``ext_log2``: log2 |F| of the challenge field."""
    name: str
    base_bytes: int
    ext_bytes: int
    ext_log2: float
    hash_bytes: int = 32
    #: base mults per extension x extension multiplication (degree 6 schoolbook = 36 + reductions; Karatsuba ~ 18-24)
    ext_mul_base: int = 36
    #: base mults per extension x base multiplication
    ext_base_mul_base: int = 6


OUR_FIELDS = Fields("BabyBear / BabyBear^6", base_bytes=4, ext_bytes=24, ext_log2=6 * math.log2(BABYBEAR_P))
PAPER_FIELDS = Fields("F_2^32 / F_2^128 (paper)", base_bytes=4, ext_bytes=16, ext_log2=128.0)


# ---------------------------------------------------------------------------------------------------------------------
# Merkle multi-opening size (expectation over a uniform |S|-subset of the n leaves)
# ---------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def expected_merkle_siblings(n: int, t: int) -> float:
    """Expected number of sibling hashes in a batched Merkle opening of ``t`` distinct uniformly random leaves of a
    binary tree with ``n`` leaves (``n`` a power of two): at depth ``j`` (``2^j`` nodes) a sibling hash is sent for every
    occupied node whose sibling is unoccupied; ``P(node unoccupied) = q_j = C(n - n/2^j, t) / C(n, t)``,
    ``P(A occupied, sibling unoccupied) = q_j - q_{j-1}``, so ``E = sum_j 2^j (q_j - q_{j-1})``.  (The paper notes the
    batched opening size is not deterministic; this is its mean.  ``t = 1`` gives ``log2 n``.)"""
    if n & (n - 1) or n < 1:
        raise ValueError("n must be a power of two")
    if t <= 0:
        return 0.0
    if t >= n:
        return 0.0
    h = n.bit_length() - 1

    lg_n = math.lgamma(n + 1) - math.lgamma(n - t + 1)

    def q(j: int) -> float:
        # P(a fixed depth-j node has no queried leaf below it): C(n - n/2^j, t) / C(n, t)
        s = n >> j
        if n - s < t:
            return 0.0
        return math.exp(math.lgamma(n - s + 1) - math.lgamma(n - s - t + 1) - lg_n)

    total = 0.0
    prev = q(0)      # depth 0 = root: unoccupied w.p. 0
    for j in range(1, h + 1):
        cur = q(j)
        total += (2**j) * (cur - prev)
        prev = cur
    return total


def queries_for(target_log2: float, rate: float) -> int:
    """Smallest ``|S|`` with ``((1 + rho)/2)^{|S|} <= 2^target_log2`` (unique-decoding proximity term)."""
    per = math.log2((1 + rate) / 2)
    return max(1, math.ceil(target_log2 / per))


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


# ---------------------------------------------------------------------------------------------------------------------
# The relation side (what sits in front of the PCS): zero-check sumcheck + the (i, i') quadratic-form sumcheck
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RelationShape:
    """A compiled ``compile.System`` at batch scale: ``rows`` real rows per unit (padded to ``rows_pad``), ``units`` per
    VU, ``vus`` per batch, constraint counts (``quad`` quadratics incl. booleanity, ``nnz`` total coefficient count of
    all constraints = the sparse work per column evaluation), ``chain_rows`` linked rows (3: c.s, c.t, c.f)."""
    name: str
    rows: int
    units: int
    vus: int = 4096
    quad: int = 0
    lin: int = 0
    nnz: int = 0
    chain_rows: int = 3
    rows_pad: int = 4096
    #: today's B-Ligero cell for the bytes/s comparison: (prover seconds t.total, proof bytes, sub-batch l, sub-batches)
    today_seconds: float | None = None
    today_proof_bytes: int | None = None
    today_l: int = 16384
    today_device: str = ""

    @property
    def cols(self) -> int:
        return self.units * self.vus

    @property
    def cells(self) -> int:
        return self.rows_pad * self.cols

    @property
    def col_vars(self) -> int:
        """Binary column variables (the factor 3 of K = 1536 is a separate ternary variable)."""
        c = self.cols
        while c % 3 == 0:
            c //= 3
        return c.bit_length() - 1

    @property
    def has_radix3(self) -> bool:
        return self.cols % 3 == 0

    @property
    def row_vars(self) -> int:
        return self.rows_pad.bit_length() - 1


# Compiled with ``relation(name).compile(params, chain=True)`` at e0cf2cd (rows / linear / quadratic / nnz of all
# constraint expressions); today's cells from the coordinator's morning report (2026-09-23 12:15Z) section 1.
RELATIONS: dict[str, RelationShape] = {
    "fp8-ada": RelationShape("fp8-ada", rows=3769, units=48, quad=3532, lin=589, nnz=12671 + 6296,
                             today_seconds=0.252, today_proof_bytes=66_000_000, today_device="RTX 4090"),
    "fp8-hopper": RelationShape("fp8-hopper", rows=3396, units=48, quad=3164, lin=554, nnz=11459 + 5706,
                                today_seconds=0.356, today_proof_bytes=60_000_000, today_device="H100"),
    "bf16-hopper": RelationShape("bf16-hopper", rows=3292, units=96, quad=3164, lin=339, nnz=11244 + 5331,
                                 today_seconds=0.665, today_proof_bytes=118_000_000, today_device="H100"),
    "bf16-ampere": RelationShape("bf16-ampere", rows=3516, units=96, quad=3387, lin=347, nnz=11895 + 5921,
                                 today_seconds=0.781, today_proof_bytes=126_000_000, today_device="A100"),
    "fp4-nvf4": RelationShape("fp4-nvf4", rows=1583, units=24, quad=1389, lin=306, nnz=5080 + 1922, rows_pad=2048,
                              today_seconds=0.0714, today_proof_bytes=15_000_000, today_device="RTX 5090"),
}


# ---------------------------------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LigeritoParams:
    tall1: int                              # message length of the level-1 columns (rows of M_1); N = tall1 * 2^wide_log2[0]
    wide_log2: tuple[int, ...]              # k'_i, i = 1..L (L committed levels)
    rate_log2: tuple[int, ...]              # nominal rate 2^-rate_log2[i] per level: n_i = nextpow2(tall_i) << rate_log2[i]
    queries: tuple[int, ...]                # |S_i|
    fields: Fields = OUR_FIELDS
    target_log2: float = -128.0
    zk: bool = True                         # salted leaves (+hash_bytes per opened row) and the mask cells
    pcs_radix_log2: int = 1                 # sumcheck variables per interactive round in the PCS partial sumchecks
    send_expected_symbols: bool = False     # paper (19) counts |S_i| "expected symbols" y_i; ref.py / pcs.py let the verifier
                                            # compute them from the opened rows, so they are not sent (proto's measured proofs)
    relation: RelationShape | None = None
    zc_radix_log2: int = 1                  # variables per round in the zero-check sumcheck over the column variables
    ii_radix_log2: int = 4                  # variables per round in the sparse (i, i') quadratic-form sumcheck
    #: the number of first-level rows the ZK padding needs is |S_1|; the mask cells live in the pad rows (free)
    mask_cells: int = 256
    #: ZK per-column RS padding (ligerito-zk ``T_PAD``): message coefficients appended to every committed column at every
    #: level.  Same n_i; the code rate becomes (tall_i + t_pad) / n_i, and the proof carries t_pad ext of ``ybar`` per level.
    t_pad: int = 0

    # --- shape ------------------------------------------------------------------------------------------------------
    @property
    def L(self) -> int:
        return len(self.wide_log2)

    @property
    def cells(self) -> int:
        return self.tall1 << self.wide_log2[0]

    @property
    def log_cells(self) -> float:
        return math.log2(self.cells)

    def wide(self, i: int) -> int:
        return 1 << self.wide_log2[i]

    def tall(self, i: int) -> int:
        """Rows of M_i (i is 0-based): tall_1 / prod_{j=2..i} wide_j."""
        t = self.tall1
        for j in range(1, i + 1):
            if t % self.wide(j):
                raise ValueError(f"level {j + 1}: tall {t} not divisible by wide {self.wide(j)}")
            t //= self.wide(j)
        return t

    def n(self, i: int) -> int:
        return _next_pow2(self.tall(i)) << self.rate_log2[i]

    def rate(self, i: int) -> float:
        return (self.tall(i) + self.t_pad) / self.n(i)

    @property
    def final_len(self) -> int:
        return self.tall(self.L - 1)

    def elem_bytes(self, i: int) -> int:
        return self.fields.base_bytes if i == 0 else self.fields.ext_bytes

    def validate(self) -> None:
        if not (len(self.wide_log2) == len(self.rate_log2) == len(self.queries)):
            raise ValueError("wide_log2, rate_log2, queries must have one entry per level")
        for i in range(self.L):
            if self.queries[i] < 1 or self.queries[i] > self.n(i):
                raise ValueError(f"level {i + 1}: |S| = {self.queries[i]} not in [1, n = {self.n(i)}]")
            if self.n(i) <= self.tall(i) + self.t_pad:
                raise ValueError(f"level {i + 1}: rate >= 1")
            if self.n(i) > 2**27 and self.fields is OUR_FIELDS:
                raise ValueError(f"level {i + 1}: n = 2^{math.log2(self.n(i)):.0f} exceeds BabyBear's 2-adicity 27")
        self.final_len

    # --- rounds -----------------------------------------------------------------------------------------------------
    def pcs_rounds(self, i: int) -> int:
        return math.ceil(self.wide_log2[i] / self.pcs_radix_log2)

    def pcs_round_coeffs(self, i: int) -> int:
        """Coefficients sent for the level-i partial sumcheck: a degree-2 polynomial in each of c variables per round
        has 3^c coefficients (``c = 1``: the paper's 3 per round)."""
        k, c, total = self.wide_log2[i], self.pcs_radix_log2, 0
        while k > 0:
            total += 3 ** min(c, k)
            k -= c
        return total

    # --- proof bytes ------------------------------------------------------------------------------------------------
    def proof_bytes(self) -> dict:
        self.validate()
        f = self.fields
        out: dict = {"levels": []}
        rows = merkle = ysym = sc = 0.0
        for i in range(self.L):
            S, w, n = self.queries[i], self.wide(i), self.n(i)
            lv_rows = S * w * self.elem_bytes(i)
            sib = expected_merkle_siblings(n, S)
            lv_merkle = sib * f.hash_bytes + (S * f.hash_bytes if self.zk else 0)     # + salt per opened leaf under ZK
            lv_y = (S + 1) * f.ext_bytes if (self.send_expected_symbols and i < self.L - 1) else 0
            lv_sc = self.pcs_round_coeffs(i) * f.ext_bytes
            out["levels"].append({"tall": self.tall(i), "wide": w, "n": n, "rate": self.rate(i), "queries": S,
                                  "rows_bytes": lv_rows, "merkle_siblings": sib, "merkle_bytes": lv_merkle,
                                  "y_bytes": lv_y, "sumcheck_bytes": lv_sc, "rounds": self.pcs_rounds(i)})
            rows += lv_rows
            merkle += lv_merkle
            ysym += lv_y
            sc += lv_sc
        final = self.final_len * f.ext_bytes
        roots = self.L * f.hash_bytes
        ybar = self.L * self.t_pad * f.ext_bytes
        rel = self.relation_bytes()
        total = rows + merkle + ysym + sc + final + roots + ybar + rel["bytes"]
        out.update({"rows_bytes": rows, "merkle_bytes": merkle, "y_bytes": ysym, "pcs_sumcheck_bytes": sc, "final_bytes": final,
                    "roots_bytes": roots, "ybar_bytes": ybar, "relation_bytes": rel["bytes"], "relation": rel, "total_bytes": total,
                    "total_kib": total / KiB,
                    "formula": "sum_i (|S_i| wide_i elem_i + E[siblings](n_i, |S_i|) H [+ |S_i| H salt] [+ (|S_i|+1) ext if"
                               " expected symbols are sent] + coeffs_i ext) + tall_L ext + L H + relation messages"})
        return out

    def relation_bytes(self) -> dict:
        """Messages of the relation side: the zero-check sumcheck over the column variables (degree 3, plus one degree-7
        last round under ZK), the (i, i') quadratic-form sumcheck (degree 2 per variable), the three Z-masked R1CS
        claims and the Libra mask sums.  Zero when no relation is attached (pure PCS)."""
        r = self.relation
        if r is None:
            return {"bytes": 0, "zc_rounds": 0, "ii_rounds": 0, "coins": 0}
        e = self.fields.ext_bytes
        zc_vars = r.col_vars
        zc_rounds = math.ceil(zc_vars / self.zc_radix_log2) + (1 if r.has_radix3 else 0)
        # degree-3 per binary variable -> 4^c coefficients per radix-2^c round; the ternary variable: degree 6 -> 7
        zc_coeffs = sum(4 ** min(self.zc_radix_log2, zc_vars - j * self.zc_radix_log2)
                        for j in range(math.ceil(zc_vars / self.zc_radix_log2))) + (7 if r.has_radix3 else 0)
        zc_coeffs += 4      # the last round carries the Z(x)-masked claims: degree 7 instead of 3
        ii_vars = 2 * r.row_vars
        ii_rounds = math.ceil(ii_vars / self.ii_radix_log2)
        ii_coeffs = sum(3 ** min(self.ii_radix_log2, ii_vars - j * self.ii_radix_log2) for j in range(ii_rounds))
        claims = 3 + r.chain_rows + 3       # v_A, v_B, v_C, the shifted chain rows, Libra mask sums H_g
        return {"bytes": (zc_coeffs + ii_coeffs + claims) * e, "zc_rounds": zc_rounds, "ii_rounds": ii_rounds,
                "zc_coeffs": zc_coeffs, "ii_coeffs": ii_coeffs, "coins": zc_rounds + ii_rounds,
                "formula": "zero-check: ceil(col_vars / c) rounds x 4^c ext (+7 for the ternary variable, +4 for the masked"
                           " last round); (i,i'): ceil(2 row_vars / c') rounds x 3^c' ext; 9 claims"}

    # --- interactive coins ------------------------------------------------------------------------------------------
    def coins(self) -> dict:
        """Verifier messages on the prover's critical path.  PCS: k'_i / radix rounds per level, then per level >= 2
        the query set S_{i-1} (after root_i) and the batching coin (after y_{i-1}); the final query set S_L after v_L.
        Relation: one coin after root_1 (tau, rho, lambda), then every sumcheck round.  Step 0 (the verifier's coin
        commitments, PROTOCOL.md 8c) precedes root_1 and is not a wait."""
        rel = self.relation_bytes()
        pcs = sum(self.pcs_rounds(i) for i in range(self.L)) + 2 * (self.L - 1) + 1
        rel_coins = (1 + rel["coins"]) if self.relation is not None else 1
        return {"pcs": pcs, "relation": rel_coins, "total": pcs + rel_coins,
                "formula": "1 (tau,rho after root_1) + zero-check rounds + (i,i') rounds + sum_i ceil(k'_i / c) + 2 (L - 1) + 1"}

    def latency(self, rtt_s: float, prover_s: float, inflight_bytes: float, gpu_bytes: float) -> dict:
        """Coin latency per batch and what it takes to hide it: ``depth = ceil((coins x RTT + prover) / prover)`` batches in
        flight, each holding ``inflight_bytes`` (message + Merkle leaf hashes) -- against the GPU's memory."""
        c = self.coins()["total"]
        wait = c * rtt_s
        depth = math.ceil((wait + prover_s) / prover_s)
        fits = math.floor(gpu_bytes / inflight_bytes) if inflight_bytes > 0 else float("inf")
        eff_depth = max(1, min(depth, fits))
        throughput_s = max(prover_s, (wait + prover_s) / eff_depth)          # never faster than the compute-bound prover
        return {"coins": c, "wait_s": wait, "depth_needed": depth, "depth_fits": fits, "effective_depth": eff_depth,
                "seconds_per_batch": throughput_s, "slowdown_vs_prover": throughput_s / prover_s,
                "formula": "wait = coins x RTT; depth = ceil((wait + T_p) / T_p); s/batch = (wait + T_p) / min(depth, floor(mem / inflight))"}

    # --- soundness --------------------------------------------------------------------------------------------------
    def soundness(self, extra_terms_log2: dict[str, float] | None = None) -> dict:
        """Every error term (log2) and their union.  Paper (17) per level + the relation's terms (the zero-check's
        Schwartz-Zippel and sumcheck terms, the (i, i') sumcheck, the constraint batching) over the challenge field."""
        self.validate()
        F = self.fields.ext_log2
        terms: dict[str, float] = {}
        for i in range(self.L):
            S, k, n = self.queries[i], self.wide_log2[i], self.n(i)
            terms[f"L{i + 1}.proximity ((1+rho)/2)^|S|"] = S * math.log2((1 + self.rate(i)) / 2)
            terms[f"L{i + 1}.field k' n/|F|"] = math.log2(k * n) - F
            terms[f"L{i + 1}.sumcheck 2k'/|F|"] = math.log2(2 * k) - F
            if i >= 1:
                terms[f"L{i + 1}.batching (|S|+1)/|F|"] = math.log2(self.queries[i - 1] + 1) - F
        r = self.relation
        if r is not None:
            nvars = r.col_vars + (1 if r.has_radix3 else 0)
            terms["ZC.constraint batching (Q+Lin)/|F|"] = math.log2(r.quad + r.lin + 3 * r.chain_rows + 4) - F
            terms["ZC.Schwartz-Zippel eq(tau) col_vars/|F|"] = math.log2(nvars) - F
            terms["ZC.sumcheck (3 col_vars + 7 + 4)/|F|"] = math.log2(3 * nvars + 7 + 4) - F
            terms["II.sumcheck 2 x 2 row_vars/|F|"] = math.log2(4 * r.row_vars) - F
            terms["L1.claim batching (chain + masks)/|F|"] = math.log2(r.chain_rows + 8) - F
        if extra_terms_log2:
            terms.update(extra_terms_log2)
        total = _log2_sum(list(terms.values()))
        return {"terms_log2": terms, "total_log2": total, "target_log2": self.target_log2, "meets_target": total <= self.target_log2,
                "field_log2": F, "formula": "union bound of paper (17) per level + relation terms; e_i = (n_i - tall_i)/2"}

    # --- prover / verifier op counts ------------------------------------------------------------------------------
    def prover_ops(self, keep_codeword: bool = False) -> dict:
        """Base-field multiply-equivalents per bucket (formulas in the dict).  ``keep_codeword``: the level-1 codeword
        (``n_1 wide_1 base_bytes``) stays resident until the openings; otherwise the |S_1| opened rows are recomputed
        from the message (|S_1| N multiply-adds)."""
        f = self.fields
        N = self.cells
        enc = {}
        for i in range(self.L):
            n, w = self.n(i), self.wide(i)
            butterflies = w * (n // 2) * (n.bit_length() - 1)
            enc[f"L{i + 1}"] = butterflies * (1 if i == 0 else 6)     # an extension NTT = 6 base NTTs (base-field twiddles)
        hash_bytes = sum(self.n(i) * self.wide(i) * self.elem_bytes(i) for i in range(self.L))
        # PCS partial sumchecks: level 1 folds N base cells by extension challenges (base x ext), 2 extra evaluation
        # points per pair for the degree-2 round polynomial; later levels are ext x ext on N / wide_1 cells.
        pcs_sc = 3 * N * f.ext_base_mul_base + sum(3 * self.tall(i - 1) * f.ext_mul_base for i in range(1, self.L))
        open1 = 0 if keep_codeword else self.queries[0] * N
        # The opened-row claims of level i-1 are tensor claims with BASE-field row parts (powers of omega_s), so at level
        # i every one of the sum_{j<i} |S_j| carried terms costs u_t = M_i^T a_t = tall_i wide_i (ext x base) MACs: a
        # (T x tall_i) . (tall_i x wide_i) contraction per level (proto's "recursion_contract" bucket).
        claims = sum(sum(self.queries[:i]) * self.tall(i) * self.wide(i) * f.ext_base_mul_base for i in range(1, self.L))
        out = {"cells": N, "encode_butterflies_by_level": enc, "encode_base_ops": sum(enc.values()),
               "merkle_hash_bytes": hash_bytes, "pcs_sumcheck_base_ops": pcs_sc, "pcs_claims_contraction_base_ops": claims,
               "openings_recompute_base_ops": open1,
               "codeword_bytes_L1": self.n(0) * self.wide(0) * f.base_bytes,
               "message_bytes_L1": N * f.base_bytes,
               "leaf_hash_bytes_L1": self.n(0) * f.hash_bytes,
               "formulas": {"encode": "wide_i (n_i/2) log2 n_i butterflies (x6 for extension levels); coefficient-basis RS: no INTT",
                            "merkle": "sum_i n_i wide_i elem_i bytes hashed (BLAKE3)",
                            "pcs_sumcheck": "3 N (ext x base) + sum_{i>=2} 3 tall_{i-1} (ext x ext)",
                            "pcs_claims_contraction": "sum_{i>=2} (sum_{j<i} |S_j|) tall_i wide_i (ext x base): the carried RS-row"
                                                      " tensor claims contracted against M_i (a GEMM; fp64-limb GEMMs are 10x"
                                                      " slower than int32 Montgomery on a 4090)",
                            "openings": "|S_1| N multiply-adds (Horner at |S_1| points per column) unless the codeword is kept"}}
        r = self.relation
        if r is not None:
            m, cols = r.rows, r.cols
            F_base = r.quad + r.nnz                       # one full mult per quadratic + one constant mult per coefficient
            F_ext = r.quad * f.ext_mul_base + r.nnz * f.ext_base_mul_base
            round1 = (cols // 2) * 2 * F_base             # s(0) = 0 on the cube: evaluate F at X = 2, 3 only (base field)
            fold1 = (cols // 2) * m * f.ext_base_mul_base
            later = (cols // 2) * (3 * F_ext + m * f.ext_mul_base)    # sum_{r>=2} cols/2^r x (3 F-evals + fold) ~ cols/2 x (...)
            ii = 8 * r.nnz * (2 * r.row_vars) * f.ext_mul_base     # the sparse (i,i') sumcheck: nnz terms per round
            out.update({"zero_check_round1_base_ops": round1, "zero_check_fold1_base_ops": fold1,
                        "zero_check_later_rounds_base_ops": later, "ii_sumcheck_base_ops": ii,
                        "zero_check_ext_table_bytes_after_round1": (cols // 2) * m * f.ext_bytes,
                        "F_eval_cost": {"base_mults": F_base, "ext_base_equiv": F_ext}})
            out["formulas"].update({
                "zero_check_round1": "cols/2 pairs x 2 F-evals (X = 2, 3) x (quad + nnz) base mults",
                "zero_check_fold1": "cols/2 x m x (ext x base)",
                "zero_check_later": "sum_{r>=2} cols/2^r x (3 F-evals in the extension + m ext folds) ~ cols/2 x (3 F_ext + m ext_mul)",
                "ii_sumcheck": "nnz sparse terms x 2 row_vars rounds x ~8 evaluation points (ext)"})
        buckets = ["encode_base_ops", "pcs_sumcheck_base_ops", "pcs_claims_contraction_base_ops", "openings_recompute_base_ops",
                   "zero_check_round1_base_ops",
                   "zero_check_fold1_base_ops", "zero_check_later_rounds_base_ops", "ii_sumcheck_base_ops"]
        out["total_base_ops"] = sum(out.get(b, 0) for b in buckets)
        return out

    def verifier_ops(self) -> dict:
        """Extension multiplications and hashes.  Per level: |S_i| rows x wide_i (row . r) + the expected symbol
        (tensor evaluation, k_i mults) + the Merkle multi-opening (siblings + leaves hashed); the final check
        ``<t_L, v_L>``: (sum_{i<L} |S_i| + relation vectors) tensor rows of length tall_L; the relation: the sparse
        constraint MLEs (nnz) and the public-input MLEs."""
        f = self.fields
        ext_mults = hashes = 0.0
        for i in range(self.L):
            S, w, n = self.queries[i], self.wide(i), self.n(i)
            ext_mults += S * (w + math.log2(self.tall(i)) + 3 * self.pcs_round_coeffs(i))
            hashes += expected_merkle_siblings(n, S) + S * math.ceil(w * self.elem_bytes(i) / 64)
        tensor_rows = sum(self.queries[:-1]) + 1 + 3 * 20
        ext_mults += tensor_rows * self.final_len + sum(self.queries[:-1]) * sum(self.wide_log2)
        rel = self.relation
        pub = 0
        if rel is not None:
            ext_mults += rel.nnz * 2 + 8 * rel.lin + rel.rows           # M_rho(r_i, r_i'), c(r_i), pins
            pub = rel.vus * 17                                          # y16 + hash digests per VU (included-hash statement)
            ext_mults += pub
        return {"ext_mults": ext_mults, "hashes": hashes, "public_mle_terms": pub,
                "seconds_est": ext_mults / 25e6 + hashes / 1e6,
                "formula": "sum_i |S_i| (wide_i + log tall_i + 3 coeffs) + (sum|S_i| + 61) tall_L + nnz + rows + 17 vus; "
                           "time at 25 M ext-mults/s and 1 M hashes/s single-thread Rust"}


def _log2_sum(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log2(sum(2.0 ** (x - m) for x in xs))


# ---------------------------------------------------------------------------------------------------------------------
# The paper's optimizer (Appendix A) and rounding, then a local search on exact bytes
# ---------------------------------------------------------------------------------------------------------------------

def paper_dims(log_cells: int, L: int, queries: int, fields: Fields) -> tuple[int, ...]:
    """Appendix A, (20): minimize ``sum_i c_i x_i`` s.t. ``prod x_i = N`` with ``x_i = wide_i`` (``c_i = |S| log|F_i|``) for
    ``i <= L`` and ``x_{L+1} = tall_L`` (``c = log|F|``); ``x_i* = (prod c)^{1/(L+1)} N^{1/(L+1)} / c_i``, rounded to powers
    of two and repaired so the product is ``N``.  Returns ``(wide_log2_1..L, final_log2)``."""
    c = [queries * (fields.base_bytes if i == 0 else fields.ext_bytes) for i in range(L)] + [fields.ext_bytes]
    geo = math.exp(sum(math.log(v) for v in c) / (L + 1)) * 2 ** (log_cells / (L + 1))
    xs = [max(1, round(math.log2(geo / v))) for v in c]
    while sum(xs) != log_cells:
        if sum(xs) < log_cells:
            xs[xs.index(min(xs))] += 1
        else:
            xs[xs.index(max(xs))] -= 1
    return tuple(xs)


def _make(tall1: int, wide_log2: tuple[int, ...], rate_log2: tuple[int, ...], queries: tuple[int, ...], **kw) -> LigeritoParams:
    return LigeritoParams(tall1=tall1, wide_log2=tuple(wide_log2), rate_log2=tuple(rate_log2), queries=tuple(queries), **kw)


def allocate_queries(p: LigeritoParams, per_level_target_log2: float | None = None) -> LigeritoParams:
    """Set |S_i| so the union of the proximity terms meets ``target_log2`` at minimum bytes: start from the equal split
    (each level at ``target - log2 L``), then move queries from expensive levels (wide ext rows) to cheap ones while the
    union still holds.  (With fixed queries per level, as in the paper, pass ``per_level_target_log2``.)"""
    if per_level_target_log2 is not None:
        return replace(p, queries=tuple(queries_for(per_level_target_log2, p.rate(i)) for i in range(p.L)))
    L = p.L
    q = [queries_for(p.target_log2 - math.log2(L), p.rate(i)) for i in range(L)]
    p = replace(p, queries=tuple(q))
    per = [math.log2((1 + p.rate(i)) / 2) for i in range(L)]

    def union(qs):
        return _log2_sum([qs[i] * per[i] for i in range(L)])

    def cost(qs):
        return replace(p, queries=tuple(qs)).proof_bytes()["total_bytes"]

    best = cost(q)
    improved = True
    while improved:
        improved = False
        for i in range(L):              # take one query from level i ...
            for j in range(L):          # ... and give the cheapest repair to level j (possibly several)
                if i == j:
                    continue
                cand = list(q)
                if cand[i] <= 1:
                    continue
                cand[i] -= 1
                while union(cand) > p.target_log2 and cand[j] < 4 * q[j]:
                    cand[j] += 1
                if union(cand) > p.target_log2:
                    continue
                c = cost(cand)
                if c < best - 1e-9:
                    q, best, improved = cand, c, True
    return replace(p, queries=tuple(q))


def _split_cells(cells: int) -> tuple[int, int]:
    """``cells = odd x 2^e``."""
    e = 0
    while cells % 2 == 0:
        cells //= 2
        e += 1
    return cells, e


def optimize(cells: int, L: int, fields: Fields = OUR_FIELDS, target_log2: float = -128.0, rate_log2: int | tuple[int, ...] = 2,
             fixed_queries: int | None = None, zk: bool = True, relation: RelationShape | None = None, max_n_log2: int = 27,
             pcs_radix_log2: int = 1, min_wide_log2: int = 2) -> LigeritoParams:
    """Best parameters for ``cells = odd x 2^e`` (``odd`` = 1 or 3) with ``L`` committed levels: the paper's closed form +
    rounding as the start, then coordinate descent on exact bytes (moving one bit between the L wide dimensions and the
    final vector), queries allocated by ``allocate_queries`` (or ``fixed_queries`` per level, as in the paper).  The odd
    factor stays in the tall dimension (``tall_i = odd x 2^...``); the code length is capped at ``2^max_n_log2``."""
    odd, e = _split_cells(cells)
    rates = tuple(rate_log2 for _ in range(L)) if isinstance(rate_log2, int) else tuple(rate_log2)
    q0 = fixed_queries or queries_for(target_log2 - math.log2(L), 2.0 ** -rates[0])
    start = paper_dims(e, L, q0, fields)

    def build(dims) -> LigeritoParams | None:
        wide, final = dims[:-1], dims[-1]
        if min(wide) < min_wide_log2 or final < 0:
            return None
        tall1 = odd << (e - wide[0])
        p = _make(tall1, wide, rates, tuple(q0 for _ in range(L)), fields=fields, target_log2=target_log2, zk=zk,
                  relation=relation, pcs_radix_log2=pcs_radix_log2)
        try:
            p.validate()
        except ValueError:
            return None
        if any(p.n(i) > 2**max_n_log2 for i in range(L)):
            return None
        return p if fixed_queries else allocate_queries(p)

    def cost(dims) -> tuple[float, LigeritoParams | None]:
        p = build(dims)
        return (float("inf"), None) if p is None else (p.proof_bytes()["total_bytes"], p)

    dims = list(start)
    best, bp = cost(dims)
    if bp is None:      # the closed-form start may violate the NTT cap: shrink the tall dimension by widening level 1
        for w1 in range(start[0], e):
            dims = [w1] + list(start[1:-1]) + [e - w1 - sum(start[1:-1])]
            best, bp = cost(dims)
            if bp is not None:
                break
    if bp is None:
        raise ValueError(f"no feasible dims for 2^{e} x {odd} cells with L = {L} under n <= 2^{max_n_log2}")
    improved = True
    while improved:
        improved = False
        for i in range(L + 1):
            for j in range(L + 1):
                if i == j:
                    continue
                cand = list(dims)
                cand[i] -= 1
                cand[j] += 1
                c, p = cost(cand)
                if c < best - 1e-9:
                    dims, best, bp, improved = cand, c, p, True
    return bp


def best_over_levels(cells: int, levels=range(1, 7), **kw) -> LigeritoParams:
    best = None
    for L in levels:
        try:
            p = optimize(cells, L, **kw)
        except ValueError:
            continue
        if best is None or p.proof_bytes()["total_bytes"] < best.proof_bytes()["total_bytes"]:
            best = p
    return best


# ---------------------------------------------------------------------------------------------------------------------
# The paper's Table 1 and our batches
# ---------------------------------------------------------------------------------------------------------------------

PAPER_TABLE1_KIB = {20: 145, 24: 255, 28: 360, 30: 420}


def paper_table1(levels=range(1, 6)) -> dict[int, dict]:
    """Table 1 of the paper (F_2^32 witness, F_2^128 challenges, rate 1/4, |S| = 148 per level, SHA-256 paths, no ZK),
    with the paper's grid search over the number of rounds."""
    out = {}
    for lg, kib in PAPER_TABLE1_KIB.items():
        p = best_over_levels(1 << lg, levels, fields=PAPER_FIELDS, target_log2=-100.0, fixed_queries=148, zk=False, max_n_log2=40)
        pb = p.proof_bytes()
        out[lg] = {"params": p, "kib": pb["total_kib"], "paper_kib": kib, "ratio": pb["total_kib"] / kib,
                   "dims": p.wide_log2 + (int(math.log2(p.final_len)),), "L": p.L}
    return out


def our_params(rel: RelationShape, L: int | None = None, rate_log2=(2, 4, 4, 4, 4, 4), radix3: bool = True, **kw) -> LigeritoParams:
    """Parameters for one of our batches at 2^-128: the level-1 code at rate 1/4 (its length is the encode cost), the
    later, tiny levels at rate 1/16 (fewer queries for free)."""
    cells = rel.rows_pad * (rel.cols if radix3 else _next_pow2(rel.cols))
    rl = tuple(rate_log2)
    if L is None:
        best = None
        for L_ in range(2, 6):
            p = optimize(cells, L_, rate_log2=rl[:L_], relation=rel, **kw)
            if best is None or p.proof_bytes()["total_bytes"] < best.proof_bytes()["total_bytes"]:
                best = p
        return best
    return optimize(cells, L, rate_log2=rl[:L], relation=rel, **kw)


def ligero_today_ops(rel: RelationShape, l: int | None = None, D: int = 6, t: int = 197) -> dict:
    """Today's B-Ligero prover, same units (base multiply-equivalents; PROTOCOL.md 4 / 8): per sub-batch of ``l`` columns
    (``n = 4 l``): encode = m (INTT_l + NTT_n) butterflies; tests = D m n (proximity r.U) + D m n (linear alpha.U)
    + (quad + nnz) n (quadratic P0) + D 20 n (chain q); Merkle = 4 m n bytes; serialization = proof bytes."""
    l = l or rel.today_l
    n = 4 * l
    subs = math.ceil(rel.cols / l)
    m = rel.rows
    enc = m * ((l // 2) * (l.bit_length() - 1) + (n // 2) * (n.bit_length() - 1))
    tests = 2 * D * m * n + (rel.quad + rel.nnz) * n + D * 20 * n
    return {"sub_batches": subs, "encode_base_ops": enc * subs, "tests_base_ops": tests * subs, "merkle_hash_bytes": 4 * m * n * subs,
            "opened_bytes": t * m * 4 * subs, "proof_bytes": rel.today_proof_bytes, "total_base_ops": (enc + tests) * subs,
            "formula": "per sub-batch: m ((l/2) log l + (n/2) log n) + 2 D m n + (quad + nnz) n + 20 D n; x sub-batches"}


@dataclass(frozen=True)
class GPU:
    """Throughput inputs for the time estimates (NOT measurements): base-field multiply-equivalents per second in a
    Montgomery SIMT kernel, NTT butterflies per second (enc-hopper measured ~0.65 T/s barrier-bound on the H100 at
    n = 65536; a four-step 2^25 NTT is assumed no better), BLAKE3 bytes per second (verity/ml/kernels/hash_gpu), memory."""
    name: str
    base_mults_per_s: float
    butterflies_per_s: float
    hash_bytes_per_s: float
    mem_bytes: float


GPUS = {
    "RTX 4090": GPU("RTX 4090", 5e12, 0.5e12, 300e9, 24e9),
    "H100": GPU("H100", 8e12, 0.65e12, 500e9, 80e9),
    "A100": GPU("A100", 4e12, 0.4e12, 300e9, 80e9),
    "RTX 5090": GPU("RTX 5090", 7e12, 0.7e12, 400e9, 32e9),
}


def prover_seconds(p: LigeritoParams, gpu: GPU, keep_codeword: bool = False, witness_s: float = 0.0) -> dict:
    ops = p.prover_ops(keep_codeword=keep_codeword)
    enc = ops["encode_base_ops"] / gpu.butterflies_per_s
    hsh = ops["merkle_hash_bytes"] / gpu.hash_bytes_per_s
    arith = (ops["total_base_ops"] - ops["encode_base_ops"]) / gpu.base_mults_per_s
    return {"encode_s": enc, "merkle_s": hsh, "sumcheck_and_openings_s": arith, "witness_s": witness_s,
            "total_s": enc + hsh + arith + witness_s}


RATE_GRID = ((1, 2), (1, 4), (2, 2), (2, 4))     # (level-1 rate_log2, later-level rate_log2)


_RECOMMEND_CACHE: dict = {}


def recommend(rel: RelationShape, gpu: GPU, budget_bytes: int = 1 << 20, levels=range(3, 6), objective: str = "balanced") -> dict:
    """Scan ``RATE_GRID x levels``; return the min-bytes point, the min-estimated-prover-time point under ``budget_bytes``,
    the min-bytes point within 10 % of that time ("balanced") and the whole grid.  ``objective`` picks the one
    ``table2_cells`` reports.  Cached per (relation, gpu, budget, levels)."""
    key = (rel.name, gpu.name, budget_bytes, tuple(levels))
    if key in _RECOMMEND_CACHE:
        out = dict(_RECOMMEND_CACHE[key])
        out["pick"] = out[{"time": "fastest", "bytes": "smallest", "balanced": "balanced"}[objective]]
        return out
    grid = []
    for r1, r2 in RATE_GRID:
        for L in levels:
            try:
                p = our_params(rel, L=L, rate_log2=(r1,) + (r2,) * (L - 1))
            except ValueError:
                continue
            grid.append({"rates": (r1, r2), "L": L, "params": p, "bytes": p.proof_bytes()["total_bytes"],
                         "seconds": prover_seconds(p, gpu)["total_s"]})
    smallest = min(grid, key=lambda g: g["bytes"])
    fastest = min((g for g in grid if g["bytes"] <= budget_bytes), key=lambda g: g["seconds"])
    balanced = min((g for g in grid if g["seconds"] <= 1.1 * fastest["seconds"]), key=lambda g: g["bytes"])
    pick = {"time": fastest, "bytes": smallest, "balanced": balanced}[objective]
    _RECOMMEND_CACHE[key] = {"smallest": smallest, "fastest": fastest, "balanced": balanced, "grid": grid}
    return {"smallest": smallest, "fastest": fastest, "balanced": balanced, "pick": pick, "grid": grid}


TABLE2 = [("bf16-ampere", "A100"), ("bf16-hopper", "H100"), ("fp8-hopper", "H100"), ("fp8-ada", "RTX 4090"), ("fp4-nvf4", "RTX 5090")]


def table2_cells(rtt_s: float = 0.030, objective: str = "balanced") -> list[dict]:
    """Predicted proof bytes per batch and bytes/s for the five Table 2 cells at TODAY's prover times (the brief's D1
    checkpoint number), plus the op-count comparison and the coin latency."""
    out = []
    for name, dev in TABLE2:
        rel = RELATIONS[name]
        rec = recommend(rel, GPUS[dev], objective=objective)
        p = rec["pick"]["params"]
        pb = p.proof_bytes()
        gpu = GPUS[dev]
        ops = p.prover_ops()
        today = ligero_today_ops(rel)
        est = prover_seconds(p, gpu)
        inflight = ops["message_bytes_L1"] + ops["leaf_hash_bytes_L1"]
        lat = p.latency(rtt_s, est["total_s"] + 0.05, inflight, gpu.mem_bytes)
        out.append({"cell": name, "device": dev, "params": p, "proof_bytes": pb["total_bytes"], "proof_kib": pb["total_kib"],
                    "today_seconds": rel.today_seconds, "bytes_per_s_at_today": pb["total_bytes"] / rel.today_seconds,
                    "today_bytes_per_s": rel.today_proof_bytes / rel.today_seconds,
                    "soundness_log2": p.soundness()["total_log2"], "coins": p.coins()["total"], "latency": lat,
                    "ops_ratio_vs_today": ops["total_base_ops"] / today["total_base_ops"],
                    "encode_ratio_vs_today": ops["encode_base_ops"] / today["encode_base_ops"],
                    "prover_est": est, "verifier": p.verifier_ops(), "smallest": rec["smallest"], "rates": rec["pick"]["rates"]})
    return out


def _fmt_params(p: LigeritoParams) -> str:
    lv = ", ".join(f"L{i + 1}: {p.tall(i)}x{p.wide(i)} n={p.n(i)} rho={p.rate(i):.3f} |S|={p.queries[i]}" for i in range(p.L))
    return f"cells=2^{p.log_cells:.2f} {lv}; final {p.final_len}"


def main() -> None:     # pragma: no cover - report
    print("== paper Table 1 (F_2^32 / F_2^128, |S| = 148, rate 1/4) ==")
    for lg, r in paper_table1().items():
        print(f"2^{lg}: {r['kib']:.0f} KiB (paper {r['paper_kib']}, x{r['ratio']:.2f})  dims k'={r['dims']}  L={r['L']}")
    print("\n== our batches (BabyBear / BabyBear^6, 2^-128, ZK) ==")
    for c in table2_cells():
        p = c["params"]
        pb = p.proof_bytes()
        print(f"\n{c['cell']} on {c['device']} (rates 2^-{c['rates'][0]}, 2^-{c['rates'][1]}; smallest alternative "
              f"{c['smallest']['bytes'] / KiB:.0f} KiB at rates {c['smallest']['rates']} L={c['smallest']['L']}):\n  {_fmt_params(p)}")
        print(f"  proof {pb['total_kib']:.0f} KiB = rows {pb['rows_bytes'] / KiB:.0f} + merkle {pb['merkle_bytes'] / KiB:.0f} + y "
              f"{pb['y_bytes'] / KiB:.0f} + pcs-sc {pb['pcs_sumcheck_bytes'] / KiB:.1f} + final {pb['final_bytes'] / KiB:.0f} + relation "
              f"{pb['relation_bytes'] / KiB:.1f} KiB; soundness 2^{c['soundness_log2']:.2f}")
        print(f"  bytes/s at today's {c['today_seconds']} s: {c['bytes_per_s_at_today'] / 1e6:.2f} MB/s (today {c['today_bytes_per_s'] / 1e6:.0f} MB/s)")
        print(f"  coins {c['coins']}; at 30 ms RTT wait {c['latency']['wait_s']:.2f} s, depth needed {c['latency']['depth_needed']}, "
              f"fits {c['latency']['depth_fits']} -> {c['latency']['seconds_per_batch']:.2f} s/batch")
        print(f"  ops vs today: total x{c['ops_ratio_vs_today']:.2f}, encode x{c['encode_ratio_vs_today']:.2f}; est prover "
              f"{c['prover_est']['total_s']:.3f} s (enc {c['prover_est']['encode_s']:.3f}, merkle {c['prover_est']['merkle_s']:.3f}, "
              f"arith {c['prover_est']['sumcheck_and_openings_s']:.3f}); verifier ~{c['verifier']['seconds_est']:.3f} s")


if __name__ == "__main__":
    main()
