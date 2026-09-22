"""Soundness accounting for the two candidate numerical backends, as exact rationals.

Every bound below is a *statistical* soundness term of a public-coin interactive protocol with live verifier
challenges (no Fiat-Shamir), except the ``commitment_hash`` component, which is a *computational* term: the
generic collision bound of the hash used for the Merkle commitments.  The total is a union bound over the terms.

Sources (version read 2026-09-21):

* sumcheck: Lund-Fortnow-Karloff-Nisan; as stated in Haboeck ePrint 2022/1530 (rev. 2025-02-20) Theorem 2 and in
  Frigo-shelat ePrint 2024/2010 (rev. 2026-04-27) Theorem 1 (Thaler Prop. 4.1): ``sum_i deg_i / |F|``.
* data-parallel GKR layer: Thaler arXiv:1304.3812v4 section 7.3 (copy index as ``b`` extra sumcheck variables); the
  per-layer degree count is *our derivation* for the quad form of draft-google-cfrg-libzk-02 section 6.3.
* lookups: Haboeck 2022/1530 Theorem 4 (identity term ``((M+1)|H|-1)/(|F|-|H|)``); Papini-Haboeck ePrint 2023/1284
  (rev. 2025-02-20) Proposition 1 (fractional-sum GKR) and Proposition 2 (logUp-GKR).
* encoding/opening: Ames-Hazay-Ishai-Venkitasubramaniam ePrint 2022/1608 (2022-11-18) Theorem 4.7 and Appendix C.
* hash: generic birthday bound ``q^2 / 2^h``; a modelling choice, not a theorem about a particular hash.

The calculator never converts a field size into a security level: ``security_bits`` accepts a ``Budget`` only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

__all__ = [
    "BABYBEAR",
    "CHECKER_LAYERS_A",
    "CHECKER_TABLES",
    "CHECKER_UNIT",
    "GOLDILOCKS",
    "MERSENNE31",
    "MERSENNE61",
    "RELATION_LAYERS_A",
    "RELATION_TABLES",
    "RTTS_MS",
    "AccountingError",
    "Budget",
    "CandidateA",
    "CandidateB",
    "CheckerLayer",
    "DepthOptions",
    "Field",
    "LigeroParams",
    "LookupTable",
    "Term",
    "account_candidate_a",
    "account_candidate_b",
    "census_estimate_a",
    "census_estimate_b",
    "data_parallel_layer_error",
    "e2e_sweep",
    "first_configuration",
    "fractional_gkr_error",
    "hash_collision_bound",
    "hm_commitment_hiding",
    "ligero_error",
    "ligero_queries_for",
    "log2",
    "logup_gkr_error",
    "logup_identity_error",
    "per_proof_target",
    "random_combination_error",
    "redraw_terms",
    "rounds_and_depth_a",
    "rounds_and_depth_b",
    "schwartz_zippel_error",
    "security_bits",
    "sumcheck_error",
    "union_over_proofs",
]

GOLDILOCKS = 2**64 - 2**32 + 1
MERSENNE61 = 2**61 - 1
BABYBEAR = 15 * 2**27 + 1
MERSENNE31 = 2**31 - 1

COMPONENTS = ("batching", "sumcheck", "lookup", "encoding_opening", "commitment_hash")
NON_SOUNDNESS = ("zk", "completeness")   # recorded, never added to the soundness total


class AccountingError(ValueError):
    pass


def log2(x: Fraction | int) -> float:
    """log2 of a positive rational (exact big-int logs; -inf for zero)."""
    x = Fraction(x)
    if x < 0:
        raise AccountingError("log2 of a negative error")
    if x == 0:
        return float("-inf")
    return math.log2(x.numerator) - math.log2(x.denominator)


# ---------------------------------------------------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Field:
    """A finite field by its size.  Deliberately has no ``bits`` attribute: a field size is not a security level."""

    name: str
    size: int
    characteristic: int

    @staticmethod
    def prime(name: str, p: int) -> Field:
        return Field(name, p, p)

    def extension(self, degree: int) -> Field:
        return Field(f"{self.name}^{degree}", self.size**degree, self.characteristic)

    @property
    def log2_size(self) -> float:
        return math.log2(self.size)


# ---------------------------------------------------------------------------------------------------------------------
# Terms and budgets
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Term:
    component: str          # one of COMPONENTS, or one of NON_SOUNDNESS (not a soundness term)
    name: str
    error: Fraction
    kind: str = "statistical"   # statistical | computational | zk | completeness
    note: str = ""

    @property
    def log2(self) -> float:
        return log2(self.error)


@dataclass
class Budget:
    candidate: str
    parameters: dict = field(default_factory=dict)
    terms: list[Term] = field(default_factory=list)

    def add(self, component: str, name: str, error: Fraction | int, kind: str = "statistical", note: str = "") -> Term:
        if component not in COMPONENTS and component not in NON_SOUNDNESS:
            raise AccountingError(f"unknown component {component!r}")
        t = Term(component, name, Fraction(error), kind, note)
        self.terms.append(t)
        return t

    def soundness_terms(self) -> list[Term]:
        return [t for t in self.terms if t.kind in ("statistical", "computational")]

    def by_component(self) -> dict[str, Fraction]:
        out = {c: Fraction(0) for c in COMPONENTS}
        for t in self.soundness_terms():
            out[t.component] += t.error
        return out

    def statistical_total(self) -> Fraction:
        return sum((t.error for t in self.terms if t.kind == "statistical"), Fraction(0))

    def total(self) -> Fraction:
        """Union bound over statistical and computational soundness terms."""
        return sum((t.error for t in self.soundness_terms()), Fraction(0))

    def zk_total(self) -> Fraction:
        return sum((t.error for t in self.terms if t.kind == "zk"), Fraction(0))

    def completeness_total(self) -> Fraction:
        """Probability that an honest run has to redraw a challenge or is rejected; not a soundness term."""
        return sum((t.error for t in self.terms if t.kind == "completeness"), Fraction(0))

    def meets(self, target_log2: float) -> bool:
        """True iff the total soundness error is at most 2**target_log2 (target_log2 negative, e.g. -128); exact
        for integer targets."""
        if float(target_log2).is_integer():
            return self.total() <= Fraction(2) ** int(target_log2)
        return log2(self.total()) <= target_log2

    def table(self) -> str:
        """A compact text table: component, term, log2 error."""
        rows = [f"{'component':18} {'term':58} {'log2':>9}  kind"]
        for t in self.terms:
            rows.append(f"{t.component:18} {t.name[:58]:58} {t.log2:9.2f}  {t.kind}")
        for c, v in self.by_component().items():
            rows.append(f"{'= ' + c:18} {'':58} {log2(v):9.2f}")
        rows.append(f"{'= total':18} {'(union bound, soundness)':58} {log2(self.total()):9.2f}")
        rows.append(f"{'= zk':18} {'(statistical ZK loss, not soundness)':58} {log2(self.zk_total()):9.2f}")
        rows.append(f"{'= completeness':18} {'(honest redraw / rejection, not soundness)':58} {log2(self.completeness_total()):9.2f}")
        return "\n".join(rows)


def union_over_proofs(budget: Budget, n_proofs: int) -> Budget:
    """The soundness budget of ``n_proofs`` independent proofs (sub-batches, or the batches of one application run),
    each with ``budget``'s error: every soundness term is multiplied by ``n_proofs`` (the union bound), ZK and
    completeness terms are multiplied likewise, and the parameters record the count.  This is the only way a result
    that reports ``N_subbatches > 1`` may state its error: the per-proof budget is never quoted for a chopped batch."""

    if not isinstance(n_proofs, int) or n_proofs < 1:
        raise AccountingError(f"n_proofs must be a positive int, got {n_proofs!r}")
    out = Budget(budget.candidate, {**budget.parameters, "n_proofs": n_proofs})
    for t in budget.terms:
        out.terms.append(Term(t.component, t.name, t.error * n_proofs, t.kind, (t.note + f" x{n_proofs} proofs").strip()))
    return out


def per_proof_target(application_log2: float, n_proofs: int, other_errors: Fraction | int = 0) -> float:
    """The log2 error each of ``n_proofs`` proofs may have so that the union bound plus ``other_errors`` (the rest of
    the protocol's error, exact) stays within ``2**application_log2``.  Soundness is a parameter derived from the
    application budget, not a constant of the protocol."""

    if not isinstance(n_proofs, int) or n_proofs < 1:
        raise AccountingError(f"n_proofs must be a positive int, got {n_proofs!r}")
    total = Fraction(2) ** int(application_log2) if float(application_log2).is_integer() else Fraction(2.0 ** application_log2)
    remaining = total - Fraction(other_errors)
    if remaining <= 0:
        raise AccountingError("the other protocol errors alone exceed the application budget")
    return log2(remaining / n_proofs)


def security_bits(budget: object) -> float:
    """-log2 of a Budget's total soundness error.  Refuses anything else (in particular a Field or an integer)."""
    if isinstance(budget, (Field, int)) or not isinstance(budget, Budget):
        raise AccountingError(
            "bits of security are a property of a complete soundness budget, not of a field size; "
            "build a Budget (account_candidate_a / account_candidate_b) and pass it here")
    return -log2(budget.total())


# ---------------------------------------------------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------------------------------------------------

def sumcheck_error(degrees: Sequence[int], F: Field) -> Fraction:
    """LFKN92 sumcheck over ``len(degrees)`` variables with per-variable degrees ``degrees``: ``sum(deg)/|F|``."""
    if any(d < 0 for d in degrees):
        raise AccountingError("negative degree")
    return Fraction(sum(degrees), F.size)


def schwartz_zippel_error(total_degree: int, F: Field, excluded: int = 0) -> Fraction:
    """A nonzero polynomial of total degree ``d`` vanishes at a uniform point of ``F^m`` minus ``excluded`` values
    with probability at most ``d / (|F| - excluded)``."""
    if F.size - excluded <= 0:
        raise AccountingError("excluded set covers the field")
    return Fraction(total_degree, F.size - excluded)


def random_combination_error(F: Field, claims: int, mode: str = "independent") -> Fraction:
    """Combining ``claims`` claims into one.  ``independent``: one fresh scalar per claim (or one scalar for two
    claims, Longfellow's alpha): error ``1/|F|``.  ``powers``: scalars ``1, r, r^2, ...``: error ``(claims-1)/|F|``."""
    if claims < 1:
        raise AccountingError("no claims")
    if claims == 1:
        return Fraction(0)
    if mode == "independent":
        return Fraction(1, F.size)
    if mode == "powers":
        return Fraction(claims - 1, F.size)
    raise AccountingError(f"unknown mode {mode!r}")


def fractional_gkr_error(n: int, F: Field) -> Fraction:
    """Papini-Haboeck Proposition 1, exact form: ``(2(n-1)+1)/|F| + sum_{k=1}^{n-1} 3k/|F|`` for the fractional-sum
    GKR over a hypercube of dimension ``n`` (degree-3 sumchecks of ``k`` variables at layer ``k``).  The paper's
    simplified bound is ``n(3n+1)/(2|F|)``; the exact sum is never larger."""
    if n < 1:
        raise AccountingError("hypercube dimension must be >= 1")
    batching = 2 * (n - 1) + 1
    sumchecks = sum(3 * k for k in range(1, n))
    return Fraction(batching + sumchecks, F.size)


def logup_identity_error(num_lookups: int, table_rows: int, F: Field, challenge_space: str = "conditioned",
                         repetitions: int = 1, columns: int = 1, exclude_poles: bool | None = None) -> Fraction:
    """Haboeck Theorem 4, first term, in the ``N`` lookups / ``|T|`` rows form.  The cleared rational identity is a
    nonzero polynomial ``P(X)`` of degree at most ``d = N + |T| - 1`` whenever the multisets differ (Lemma 5; needs
    ``char(F) > N`` so multiplicities are nonzero field elements).

    ``challenge_space``:
      * ``"conditioned"`` (the protocols' public pole-redraw rule, PROTOCOL.md section 4.2 / 8): ``z`` is uniform over
        ``F`` minus the public table image ``{t_v(beta)}`` (at most ``|T|`` points).  ``P`` has at most ``d`` roots
        there, so the error is ``d / (|F| - |T|)``.  This is also the sampling set of Haboeck's Protocol 2 (his
        condition ``x not in -t(H)``, Remark 3), so Theorem 4's ``((M+1)|H|-1)/(|F|-|H|)`` is the same bound.
      * ``"whole_field"``: ``z`` uniform over all of ``F`` (Remark 3's overwhelmingly-complete variant): ``d / |F|``.
    The two differ by the factor ``|F|/(|F|-|T|) = 1 + |T|/(|F|-|T|)``; ``test_conditioned_vs_whole_field`` shows it.

    ``columns > 1`` (vector-valued lookup, Haboeck 3.4): rows are compressed with powers of a second challenge
    ``beta``; the cleared numerator has ``beta``-degree at most ``d (columns - 1)``.  *Derivation*: conditioning on
    ``beta`` and union-bounding adds ``d (columns - 1)/|F|``.  ``repetitions`` independent challenges raise the term
    to that power (the prover must satisfy every repetition)."""
    if exclude_poles is not None:                       # backward-compatible alias
        challenge_space = "conditioned" if exclude_poles else "whole_field"
    if num_lookups < 1 or table_rows < 1 or columns < 1:
        raise AccountingError("need at least one lookup, one table row and one column")
    if F.characteristic <= num_lookups:
        raise AccountingError(
            f"logUp needs char(F) > total lookups: char={F.characteristic}, lookups={num_lookups} (Haboeck Lemma 5)")
    if challenge_space == "conditioned":
        denom = F.size - table_rows
    elif challenge_space == "whole_field":
        denom = F.size
    else:
        raise AccountingError(f"unknown challenge_space {challenge_space!r}")
    if denom <= 0:
        raise AccountingError("table larger than the challenge field")
    d = num_lookups + table_rows - 1
    term = Fraction(d, denom) + Fraction(d * (columns - 1), F.size)
    return term ** repetitions


def redraw_terms(table_rows: int, F: Field, slots: int = 2) -> dict[str, Fraction]:
    """The public pole-redraw rule with ``slots`` pre-issued challenge values per table: ``redraw`` is the probability
    the first value lies in the table image (at most ``|T|/|F|``; a redraw costs no round trip because every slot is
    sent together); ``abort`` is the probability all ``slots`` values do (the session ends publicly).  Neither is a
    soundness term; neither depends on the witness."""
    if slots < 1:
        raise AccountingError("need at least one challenge slot")
    q = Fraction(table_rows, F.size)
    return {"redraw": q, "abort": q**slots}


def hm_commitment_hiding(n_commitments: int, k_bits: int) -> Fraction:
    """Halevi-Micali (CRYPTO 1996) Theorem 1: the commit strings of two messages are within statistical distance
    ``2^-k``; ``n_commitments`` independent commitments are within ``n 2^-k`` (union bound).  Used for the verifier's
    coin commitments (a *soundness* term: what a prover may learn about coins) and, when leaves are HM commitments to
    columns, for the hiding of unopened columns (a *zk* term)."""
    if n_commitments < 0 or k_bits < 1:
        raise AccountingError("bad commitment parameters")
    return Fraction(n_commitments, 2**k_bits)


def logup_gkr_error(num_lookups: int, table_rows: int, F: Field, repetitions: int = 1,
                    columns: int = 1) -> dict[str, Fraction]:
    """Papini-Haboeck Proposition 2 in the ``N`` / ``|T|`` form: identity term plus the fractional GKR over the
    smallest hypercube holding all ``N + |T|`` fractions.  Returns the two terms separately."""
    n = max(1, math.ceil(math.log2(num_lookups + table_rows)))
    return {
        "identity": logup_identity_error(num_lookups, table_rows, F, repetitions=repetitions, columns=columns),
        "fractional_gkr": repetitions * fractional_gkr_error(n, F),
        "hypercube_dim": Fraction(n),
    }


def data_parallel_layer_error(s_out: int, s_in: int, b: int, F: Field, *, alpha: bool = True,
                              assertions: bool = False) -> dict[str, Fraction | int]:
    """One layer of Thaler's data-parallel GKR in libzk quad form (our derivation, PROTOCOL.md A section 4).

    Sum over ``(g1, c, l1, r1)`` in ``{0,1}^{s_out + b + s_in + s_in}`` of
    ``[alpha*beta(z0,(g1,c)) + (1-alpha)*beta(z1,(g1,c))] * Q~(g1,l1,r1) * V~(l1,c) * V~(r1,c)``.
    Per-variable degrees: ``g1``: 2, ``c``: 3, ``l1``: 2, ``r1``: 2.  Rounds ``s_out + 2 s_in + b``.
    ``alpha``: the two-claims-to-one combination, ``1/|F|``.  ``assertions``: the ``beta`` scalar of the Z quad
    (libzk 6.3.3), ``1/|F|``."""
    if min(s_out, s_in, b) < 0:
        raise AccountingError("negative dimension")
    degrees = [2] * s_out + [3] * b + [2] * (2 * s_in)
    terms: dict[str, Fraction | int] = {
        "rounds": len(degrees),
        "sumcheck": sumcheck_error(degrees, F),
        "alpha": random_combination_error(F, 2) if alpha else Fraction(0),
        "beta": random_combination_error(F, 2) if assertions else Fraction(0),
        "pad_elements": sum(degrees) + 3,   # d evaluations per round (p(1) implied) + vl, vr, vl*vr
    }
    return terms


def hash_collision_bound(hash_bits: int, adversary_queries: int) -> Fraction:
    """Generic birthday bound ``q^2 / 2^h`` on finding a collision with ``q`` hash evaluations.  A modelling choice
    for the computational component; the actual statement is a reduction to collision finding."""
    if hash_bits <= 0 or adversary_queries < 1:
        raise AccountingError("bad hash parameters")
    return min(Fraction(1), Fraction(adversary_queries**2, 2**hash_bits))


# ---------------------------------------------------------------------------------------------------------------------
# Ligero (Ames-Hazay-Ishai-Venkitasubramaniam, ePrint 2022/1608)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LigeroParams:
    """Reed-Solomon ``[n, k, d = n-k+1]`` over the code field; ``l`` payload symbols per row; ``t`` opened columns;
    ``e`` proximity parameter (default: the largest with ``e < d/2``, i.e. ``floor((n-k)/2)``); ``sigma`` repetitions
    of the field-dependent tests."""

    n: int
    k: int
    l: int
    t: int
    e: int | None = None
    sigma: int = 1
    zk: bool = True

    @property
    def distance(self) -> int:
        return self.n - self.k + 1

    @property
    def proximity(self) -> int:
        return (self.n - self.k) // 2 if self.e is None else self.e

    @property
    def rate(self) -> Fraction:
        return Fraction(self.k, self.n)

    def validate(self, F_code: Field) -> None:
        if not (0 < self.l < self.k < self.n):
            raise AccountingError("need 0 < l < k < n")
        if F_code.size < self.n + self.l:
            raise AccountingError("Reed-Solomon needs |F_code| >= n + l distinct evaluation points (Thm 4.7)")
        if self.zk and not self.k > self.l + self.t:
            raise AccountingError(f"zero-knowledge needs k > l + t (Lemma 4.15): k={self.k}, l+t={self.l + self.t}")
        if not self.n > 2 * self.k:
            raise AccountingError("the quadratic test needs n > 2k, else (2k/n)^t >= 1 (Appendix C; Longfellow Thm 4)")
        if not 2 * self.proximity <= self.n - self.k:
            raise AccountingError("proximity parameter must satisfy e < d/2 (2e <= n-k)")
        if self.t < 1 or self.t > self.n:
            raise AccountingError("opened columns t must be in [1, n]")


def ligero_error(p: LigeroParams, F_code: Field, F_challenge: Field, analysis: str = "appendix_c",
                 combination: str = "uniform", n_linear: int | None = None, n_quadratic_rows: int | None = None,
                 ) -> dict[str, Fraction]:
    """Soundness terms of the Ligero ZKIPCP, decomposed.

    ``appendix_c`` (the analysis the authors use for parameters, Longfellow 2.2 'Improvement'): for ``e < d/2``
        ``(1 - e/n)^t + ((k+l)/n)^t + (2k/n)^t + (n+3)/|F|^sigma``
    split as IRS query ``(1-e/n)^t``, linear query ``((k+l)/n)^t``, quadratic query ``(2k/n)^t``, and field terms
    IRS ``(n+1)/|F|^sigma`` (BCIKS20 unique-decoding bound ``n/|F|`` plus ``1/|F|``), linear ``1/|F|^sigma``,
    quadratic ``1/|F|^sigma``.
    ``theorem_4_7``: for ``e < (n-k)/4``: ``(d+2)/|F|^sigma + (1-e/n)^t + 2((e+2k)/n)^t``.

    ``combination``: how the verifier's combination vectors for the *linear* and *quadratic* tests are drawn.
    ``"uniform"``: independent uniform coefficients (the paper).  ``"powers"``: ``1, rho, rho^2, ...`` of one uniform
    ``rho`` per test, so the challenge is one field element (needed when the constraint count is in the 10^8 range,
    Candidate B); *derivation*: a failing constraint makes the combined polynomial in ``rho`` nonzero of degree below
    the count, so the field terms become ``(n_linear - 1)/|F|^sigma`` and ``(n_quadratic_rows - 1)/|F|^sigma``
    (Schwartz-Zippel).  The IRS test keeps uniform row coefficients in both modes (no geometric-sequence proximity
    statement is relied on).

    Field terms use the *challenge* field (the field the random combinations are drawn from); the proximity-gap
    statement for a subfield code combined over an extension is an assumption recorded in PROTOCOL.md."""
    p.validate(F_code)
    n, k, l, t, e, s = p.n, p.k, p.l, p.t, p.proximity, p.sigma
    Fs = Fraction(F_challenge.size) ** s
    if combination == "uniform":
        lin_num, quad_num = 1, 1
    elif combination == "powers":
        if n_linear is None or n_quadratic_rows is None:
            raise AccountingError("combination='powers' needs n_linear and n_quadratic_rows")
        lin_num, quad_num = max(1, n_linear - 1), max(1, n_quadratic_rows - 1)
    else:
        raise AccountingError(f"unknown combination {combination!r}")
    if analysis == "appendix_c":
        return {
            "irs_query": (Fraction(n - e, n)) ** t,
            "linear_query": Fraction(k + l, n) ** t,
            "quadratic_query": Fraction(2 * k, n) ** t,
            "irs_field": Fraction(n + 1) / Fs,
            "linear_field": Fraction(lin_num) / Fs,
            "quadratic_field": Fraction(quad_num) / Fs,
        }
    if analysis == "theorem_4_7":
        if not 4 * e < n - k:
            raise AccountingError("Theorem 4.7 needs e < (n-k)/4")
        return {
            "irs_query": (Fraction(n - e, n)) ** t,
            "linear_query": Fraction(e + 2 * k, n) ** t,
            "quadratic_query": Fraction(e + 2 * k, n) ** t,
            "field": Fraction(p.distance + lin_num + quad_num) / Fs,
        }
    raise AccountingError(f"unknown analysis {analysis!r}")


def ligero_queries_for(target_log2: float, n: int, k: int, l: int, e: int | None = None) -> int:
    """Smallest ``t`` whose three Appendix-C query terms sum to at most ``2**target_log2``."""
    e = (n - k) // 2 if e is None else e
    if not n > 2 * k:
        raise AccountingError("n > 2k required")
    a, b, c = Fraction(n - e, n), Fraction(k + l, n), Fraction(2 * k, n)
    for t in range(1, n + 1):
        if log2(a**t + b**t + c**t) <= target_log2:
            return t
    raise AccountingError("no t <= n reaches the target; lower the rate")


# ---------------------------------------------------------------------------------------------------------------------
# Candidate descriptions
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LookupTable:
    """One public table: ``rows`` entries, ``queries_per_unit`` lookups per 16-product update, ``columns`` = key
    plus output columns (vector-valued lookup, Haboeck 3.4)."""

    name: str
    rows: int
    queries_per_unit: int
    columns: int = 1


@dataclass(frozen=True)
class CheckerLayer:
    """One layer of the per-unit checker: ``s_out`` = log2 of its output width, ``s_in`` = log2 of the width of the
    layer it reads (per unit).  ``assertions``: the layer carries Z-quad assertions (libzk 6.3.3)."""

    s_out: int
    s_in: int
    assertions: bool = True


# The census of ``verity.ml.tc.relation`` for AMPERE_BF16_M16N8K16, ALIGN_DIV: 146 mul, 104 lin, 172 lookups per
# 16-product update.  SUPERSEDED (2026-09-21): relation.py counts none of the red-team / WP2 repairs (RANGE(magW),
# direct remainder ranges, tight quotient bounds, the decode of an arbitrary incoming accumulator); kept only so the
# undercount can be shown (``security.census.diff_vs_relation``).
RELATION_TABLES: tuple[LookupTable, ...] = (
    LookupTable("T_bf16", 65536, 32, 4),
    LookupTable("T_pcls", 64, 16, 5),
    LookupTable("T_pow26", 512, 18, 3),
    LookupTable("T_exp25", 512, 2, 5),
    LookupTable("T_norm30", 31, 2, 3),
    LookupTable("T_iszero12", 4096, 4, 2),
    LookupTable("T_any9", 10, 6, 2),
    LookupTable("T_range16", 65536, 46, 1),
    LookupTable("T_range14", 16384, 2, 1),
    LookupTable("T_range13", 8192, 2, 1),
    LookupTable("T_range10", 1024, 36, 1),
    LookupTable("T_range8", 256, 2, 1),
    LookupTable("T_range5", 32, 2, 1),
    LookupTable("T_range3", 8, 2, 1),
)
RELATION_LAYERS_A: tuple[CheckerLayer, ...] = (
    CheckerLayer(8, 7), CheckerLayer(7, 6), CheckerLayer(6, 5), CheckerLayer(5, 8),
)


def _checker_census():
    """The WP2 checker census (``security.census``), computed once: the source of every per-unit default below."""
    from verity_numerical.security import census as _c
    unit = _c.census_unit()
    tables = tuple(
        LookupTable(name, _c.fixed_tables()[name][0], q, _c.fixed_tables()[name][1])
        for name, q in _c.lookups_by_table(unit).items()
    )
    shape = tuple(_c.layer_shape(unit))
    layers = tuple(CheckerLayer(L["s_out"], L["s_in"], assertions=(L["depth"] == "assert")) for L in shape)
    return unit, tables, layers, shape


CHECKER_UNIT, CHECKER_TABLES, CHECKER_LAYERS_A, CHECKER_SHAPE_A = _checker_census()
#: Per-unit values a GKR layer reads (carried widths, not rounded): the size Thaler's data-parallel prover works on.
CHECKER_LAYER_WIDTHS_A: tuple[int, ...] = tuple(L["width"] for L in CHECKER_SHAPE_A)
#: Alias kept for the protocol documents: the tables every candidate's lookups go through (16-bit range table
#: consolidated: a w-bit check is ``x 2^(16-w)`` in the one table) and the four fixed tables of the checker.
CENSUS_TABLES = CHECKER_TABLES
#: Candidate A's layered checker from the census (output layer first): the assertion layer, then one layer per
#: multiplicative depth with every lower-depth value carried (upper-bound widths; see ``census.layer_shape``).
DEFAULT_LAYERS_A = CHECKER_LAYERS_A


@dataclass(frozen=True)
class DepthOptions:
    """Round-trip reductions applied when counting the sequential challenge depth (PROTOCOL.md A section 5.4).

    Known / immediate (always applied): independent sumchecks run in parallel (all tables' fractional GKRs; the
    checker GKR beside them), all challenges a verifier can issue at once travel in one message, one prover message
    carries every prover response due at that point.

    Proposals (default off):
    * ``variables_per_round`` = c: bind ``c`` sumcheck variables per verifier message (the prover sends the c-variate
      round polynomial, ``(d+1)^c - 1`` values); round count ``ceil(v/c)``, Schwartz-Zippel error per message
      ``c d/|F|`` (unchanged total); prover work in those rounds grows ~``(d+1)^c/((d+1) c)``.
    * ``table_shard_bits`` = s: split each table's queries into ``2^s`` independent fractional GKRs with their own
      multiplicity vectors (all challenge-independent), hypercube dimension ``ceil(log2(N/2^s + |T|))``; costs
      ``(2^s - 1)|T|`` extra committed multiplicities per table; the identity error is charged per shard.
      A table is sharded only while ``N/2^s > |T|``.
    * ``commit_query_wires``: every lookup query that the census places on a computed wire (a product such as
      ``r * ipow``, or a linear expression such as ``t - nz``) is *also* committed as an input column bound to its
      expression, so every leaf claim of the lookup argument lands on the input layer and the lookups never wait for
      the checker (PROTOCOL.md A 4.3); costs ``CandidateA.query_wires_per_unit`` committed elements per unit.  When
      False the lookup claims merge into the checker layers that hold the query wires, and every such layer (all
      layers of depth <= ``census.max_query_depth``; for the WP2 checker that is every computed layer) follows the
      lookups on the critical path.
    """

    variables_per_round: int = 1
    table_shard_bits: int = 0
    commit_query_wires: bool = True

    def validate(self) -> None:
        if self.variables_per_round < 1 or self.table_shard_bits < 0:
            raise AccountingError("bad depth options")


@dataclass(frozen=True)
class CandidateA:
    """Data-parallel GKR + logUp-GKR + Longfellow-style deferred Ligero (PROTOCOL.md A)."""

    K: int
    B: int
    trace_field: Field
    challenge_field: Field
    ligero: LigeroParams
    layers: tuple[CheckerLayer, ...] = DEFAULT_LAYERS_A
    tables: tuple[LookupTable, ...] = CENSUS_TABLES
    # Committed per update (WP2 census): 34 public words (c, 32 operands, y) + 225 hints + 62 looked-up row columns.
    inputs_per_unit: int = CHECKER_UNIT.columns_committed_a
    # Lookup queries on computed wires (census ``queries_on_computed_wires``), committed when ``depth.commit_query_wires``.
    query_wires_per_unit: int = CHECKER_UNIT.queries_on_computed_wires
    query_layers: int = sum(1 for L in DEFAULT_LAYERS_A if not L.assertions)   # computed layers holding queries
    lookup_repetitions: int = 1
    hash_bits: int = 256
    hash_queries: int = 2**64
    challenge_field_elements_per_pad: int | None = None   # trace-field elements per pad; default = extension degree
    redraw_slots: int = 2               # pre-issued z_T values per table (public pole-redraw rule)
    leaf_commitment_bits: int | None = 160   # HM96 commitment per column (None: unsalted leaf hash, extra assumption)
    coin_commitment_bits: int | None = None  # step 0 (malicious-verifier ZK) with HM96 parameter k; None = HVZK only
    depth: DepthOptions = DepthOptions()

    @property
    def units(self) -> int:
        if self.K % 16:
            raise AccountingError("K must be a multiple of 16")
        return self.B * (self.K // 16)

    @property
    def b(self) -> int:
        return max(0, math.ceil(math.log2(self.units)))


@dataclass(frozen=True)
class CandidateB:
    """Direct local constraints in an interactive ZK Ligero argument, lookups by logUp (PROTOCOL.md B)."""

    K: int
    B: int
    trace_field: Field
    challenge_field: Field
    ligero: LigeroParams
    tables: tuple[LookupTable, ...] = CENSUS_TABLES
    # Committed witness elements per update before lookup helpers (WP2 census, layout ESTIMATE): 3 aligned slots per
    # field multiplication (Ligero's quadratic test needs x, y, z in the same column) + one slot per column that is
    # not a direct operand or product (``census.columns_committed_b``).
    witness_per_unit: int = CHECKER_UNIT.columns_committed_b
    # linear identities + range-chunk splits + two copy equalities per product (operand slots bound to their source)
    linear_per_unit: int = CHECKER_UNIT.lin_total + 2 * CHECKER_UNIT.mul
    quadratic_per_unit: int = CHECKER_UNIT.mul          # helper products are added per lookup
    lookup_mode: str = "helper_columns"  # helper_columns (second commitment round) | gkr (fractional sumcheck)
    lookup_repetitions: int = 1
    hash_bits: int = 256
    hash_queries: int = 2**64
    combination: str = "powers"          # Ligero linear/quadratic combination coefficients: powers of one element
    redraw_slots: int = 2
    leaf_commitment_bits: int | None = 160
    coin_commitment_bits: int | None = None
    depth: DepthOptions = DepthOptions()

    @property
    def units(self) -> int:
        if self.K % 16:
            raise AccountingError("K must be a multiple of 16")
        return self.B * (self.K // 16)


def _extension_degree(trace: Field, challenge: Field) -> int:
    d = round(math.log(challenge.size, trace.size))
    if trace.size**d != challenge.size:
        raise AccountingError("challenge field must be an extension of the trace field")
    return d


def _shards(p: CandidateA | CandidateB, tbl: LookupTable) -> tuple[int, int, int]:
    """(number of shards, lookups per shard, hypercube dimension per shard) for one table under the depth options."""
    N = tbl.queries_per_unit * p.units
    s = p.depth.table_shard_bits
    while s > 0 and N // 2**s <= tbl.rows:
        s -= 1
    shards = 2**s
    per = -(-N // shards)
    return shards, per, max(1, math.ceil(math.log2(per + tbl.rows)))


def _lookup_terms(B: Budget, p: CandidateA | CandidateB, tbl: LookupTable, gkr: bool) -> tuple[int, int, int]:
    """Adds the lookup terms of one table; returns (shards, hypercube dim per shard, fractional-GKR rounds per shard)."""
    Fc = p.challenge_field
    shards, per, n_dim = _shards(p, tbl)
    tag = f"{tbl.name}" + (f" x{shards} shards" if shards > 1 else "")
    ident = logup_identity_error(per, tbl.rows, Fc, repetitions=p.lookup_repetitions, columns=tbl.columns)
    B.add("lookup", f"{tag}: rational identity, N={per}, |T|={tbl.rows}, cols={tbl.columns}, conditioned on z not in image",
          shards * ident)
    rounds = 0
    if gkr:
        B.add("lookup", f"{tag}: fractional GKR over 2^{n_dim}",
              shards * p.lookup_repetitions * fractional_gkr_error(n_dim, Fc))
        rounds = p.lookup_repetitions * (n_dim * (n_dim - 1) // 2 + (n_dim - 1))
    rd = redraw_terms(tbl.rows, Fc, p.redraw_slots)
    B.add("completeness", f"{tbl.name}: z in public table image, first slot (redraw, no round trip)", rd["redraw"],
          kind="completeness")
    B.add("completeness", f"{tbl.name}: all {p.redraw_slots} slots in the image (public abort)", rd["abort"],
          kind="completeness")
    return shards, n_dim, rounds


def _commitment_terms(B: Budget, p: CandidateA | CandidateB, n_coins: int) -> None:
    B.add("commitment_hash", f"Merkle/commitment collisions, generic q^2/2^h with q=2^{int(math.log2(p.hash_queries))}, h={p.hash_bits}",
          hash_collision_bound(p.hash_bits, p.hash_queries), kind="computational",
          note="reduction to collision resistance; not a probability over verifier coins")
    if p.coin_commitment_bits is not None:
        B.add("commitment_hash", f"verifier coin commitments: HM96 hiding, {n_coins} coins x 2^-{p.coin_commitment_bits}",
              hm_commitment_hiding(n_coins, p.coin_commitment_bits), kind="statistical",
              note="what a prover may learn about committed coins; binding is a ZK-side (computational) requirement")
    if p.leaf_commitment_bits is not None:
        B.add("zk", f"column commitments: HM96 hiding, {p.ligero.n} leaves x 2^-{p.leaf_commitment_bits}",
              hm_commitment_hiding(p.ligero.n, p.leaf_commitment_bits), kind="zk")
    else:
        B.add("zk", "unsalted leaf hashes: hiding is an assumption on H beyond collision resistance (not quantified)",
              0, kind="zk")


def account_candidate_a(p: CandidateA) -> Budget:
    p.depth.validate()
    Fc, Ft = p.challenge_field, p.trace_field
    ext = _extension_degree(Ft, Fc)
    per_pad = p.challenge_field_elements_per_pad or ext
    B = Budget("A: data-parallel GKR + logUp-GKR + deferred Ligero (interactive)")
    b = p.b
    units = p.units

    # ---- batching / claim reductions ---------------------------------------------------------------------------------
    s0 = p.layers[0].s_out
    B.add("batching", f"output layer all-zero claim, random z in F^{s0 + b} (Schwartz-Zippel)",
          schwartz_zippel_error(s0 + b, Fc))
    rounds = 0
    pads = 0
    for j, layer in enumerate(p.layers):
        acc = data_parallel_layer_error(layer.s_out, layer.s_in, b, Fc, alpha=True, assertions=layer.assertions)
        rounds += int(acc["rounds"])
        pads += int(acc["pad_elements"])
        B.add("batching", f"layer {j}: alpha (two claims -> one)", acc["alpha"])
        if layer.assertions:
            B.add("batching", f"layer {j}: beta (Q + beta*Z assertions)", acc["beta"])
        B.add("sumcheck", f"layer {j}: {acc['rounds']} rounds, degrees 2/3/2/2 over (g1,c,l1,r1)", acc["sumcheck"])
    B.add("batching", "final input claims: gamma (two evaluation claims -> one Ligero linear constraint)",
          random_combination_error(Fc, 2))

    # ---- lookups: logUp-GKR per table (per shard), claims merged into the checker or the input constraints ---------
    extra_mult = 0
    for tbl in p.tables:
        shards, n_dim, r_sh = _lookup_terms(B, p, tbl, gkr=True)
        B.add("batching", f"{tbl.name}: merge table-side/witness-side claims into the layer / input claim",
              shards * random_combination_error(Fc, 2))
        rounds += shards * r_sh
        pads += shards * p.lookup_repetitions * (3 * (n_dim * (n_dim - 1) // 2) + 4 * n_dim)
        extra_mult += (shards - 1) * tbl.rows

    # ---- encoding / opening: Ligero over the committed inputs, multiplicities and pads --------------------------------
    per_unit = p.inputs_per_unit + (p.query_wires_per_unit if p.depth.commit_query_wires else 0)
    committed = units * per_unit + sum(t.rows for t in p.tables) + extra_mult + pads * per_pad
    lig = ligero_error(p.ligero, Ft, Fc)
    for name, val in lig.items():
        B.add("encoding_opening", f"Ligero {name} (n={p.ligero.n}, k={p.ligero.k}, l={p.ligero.l}, t={p.ligero.t}, e={p.ligero.proximity})", val)
    depth = rounds_and_depth_a(p)
    _commitment_terms(B, p, depth["coins"])

    rows = math.ceil(committed / p.ligero.l) + 3
    B.parameters.update({
        "K": p.K, "B": p.B, "units": units, "copy_bits_b": b,
        "trace_field": Ft.name, "challenge_field": Fc.name, "extension_degree": ext,
        "checker_layers": len(p.layers), "sumcheck_rounds_total": rounds, "pad_elements": pads,
        "committed_trace_elements": committed, "ligero_rows": rows,
        "opened_trace_elements": rows * p.ligero.t,
        "merkle_leaves": p.ligero.n,
        "zk_class": "COMPLETE_ZK_BACKEND (proposed step 0)" if p.coin_commitment_bits else "COMPLETE_HVZK_BACKEND",
        **{k: v for k, v in depth.items() if k.startswith("rounds.")},
    })
    return B


def account_candidate_b(p: CandidateB) -> Budget:
    p.depth.validate()
    Fc, Ft = p.challenge_field, p.trace_field
    ext = _extension_degree(Ft, Fc)
    B = Budget("B: direct constraints in interactive ZK Ligero + logUp (interactive)")
    units = p.units
    committed = units * p.witness_per_unit + sum(t.rows for t in p.tables)
    rounds = 0
    n_linear = units * p.linear_per_unit + sum(t.rows for t in p.tables)
    n_quadratic = units * p.quadratic_per_unit
    for tbl in p.tables:
        N = tbl.queries_per_unit * units
        if p.lookup_mode == "helper_columns":
            shards, _, _ = _lookup_terms(B, p, tbl, gkr=False)
            committed += p.lookup_repetitions * (N + shards * tbl.rows) * ext     # inverse columns h in F_c
            n_quadratic += p.lookup_repetitions * N
            n_linear += shards * tbl.rows
        elif p.lookup_mode == "gkr":
            shards, n_dim, r_sh = _lookup_terms(B, p, tbl, gkr=True)
            rounds += shards * r_sh
            committed += shards * p.lookup_repetitions * (3 * (n_dim * (n_dim - 1) // 2) + 4 * n_dim) * ext
            committed += (shards - 1) * tbl.rows
        else:
            raise AccountingError(f"unknown lookup_mode {p.lookup_mode!r}")
    lig = ligero_error(p.ligero, Ft, Fc, combination=p.combination, n_linear=n_linear,
                       n_quadratic_rows=-(-n_quadratic // p.ligero.l))
    for name, val in lig.items():
        B.add("encoding_opening", f"Ligero {name} (n={p.ligero.n}, k={p.ligero.k}, l={p.ligero.l}, t={p.ligero.t}, e={p.ligero.proximity}, {p.combination})", val)
    depth = rounds_and_depth_b(p)
    _commitment_terms(B, p, depth["coins"])
    rows = math.ceil(committed / p.ligero.l) + 3
    B.parameters.update({
        "K": p.K, "B": p.B, "units": units, "trace_field": Ft.name, "challenge_field": Fc.name,
        "extension_degree": ext, "lookup_mode": p.lookup_mode, "sumcheck_rounds_total": rounds,
        "committed_trace_elements": committed, "ligero_rows": rows,
        "opened_trace_elements": rows * p.ligero.t,
        "commitment_rounds": 2 if p.lookup_mode == "helper_columns" else 1,
        "merkle_leaves": p.ligero.n, "linear_constraints": n_linear, "quadratic_constraints": n_quadratic,
        "zk_class": "COMPLETE_ZK_BACKEND (proposed step 0)" if p.coin_commitment_bits else "COMPLETE_HVZK_BACKEND",
        **{k: v for k, v in depth.items() if k.startswith("rounds.")},
    })
    return B


# ---------------------------------------------------------------------------------------------------------------------
# Latency: protocol rounds vs sequential challenge depth (contract names rounds.protocol / rounds.sequential_depth)
# ---------------------------------------------------------------------------------------------------------------------

def _fractional_chain(n_dim: int, c: int) -> tuple[int, int]:
    """(protocol challenge slots, sequential depth) of one fractional GKR of dimension ``n_dim`` after its ``z``:
    layer 0 -> 1 needs one (mu, lambda) message; layer k = 1..n-1 needs ``k`` sumcheck challenges (``ceil(k/c)`` with
    ``c`` variables per message) followed by one (mu, lambda) message that must come *after* the four claims."""
    slots = 1 + sum(k + 1 for k in range(1, n_dim))
    depth = 1 + sum(-(-k // c) + 1 for k in range(1, n_dim))
    return slots, depth


def rounds_and_depth_a(p: CandidateA) -> dict[str, int]:
    """Protocol rounds and sequential challenge depth of Candidate A (PROTOCOL.md A 5.4).

    depth = [1 step-0 message] + 1 (all z_T, beta_T, z_0 in one message)
            + max(longest fractional-GKR chain, checker chain)  [+ the query-holding checker layers + gamma when
              the query wires are not committed] + 2 (Ligero r, Q).
    Each checker layer contributes ``ceil(rounds_i/c) + 1`` (the alpha/beta/merge message after v_l, v_r)."""
    p.depth.validate()
    c = p.depth.variables_per_round
    slots_lookup = depth_lookup = 0
    for tbl in p.tables:
        shards, _, n_dim = _shards(p, tbl)
        sl, dp = _fractional_chain(n_dim, c)
        slots_lookup += shards * (1 + sl) * p.lookup_repetitions      # + 1: the z_T slot itself
        depth_lookup = max(depth_lookup, dp)
    layer_rounds = [layer.s_out + p.b + 2 * layer.s_in for layer in p.layers]
    checker_slots = 1 + sum(r + 1 for r in layer_rounds) + 1          # z_0, per-layer challenges + alpha, gamma
    checker_depth = [-(-r // c) + 1 for r in layer_rounds]
    if p.depth.commit_query_wires:
        parallel = max(depth_lookup, sum(checker_depth) + 1)
        tail = 0
    else:
        # the ``query_layers`` deepest computed layers hold query wires: their sumchecks follow the lookups
        head = checker_depth[:len(checker_depth) - p.query_layers]
        parallel = max(depth_lookup, sum(head))
        tail = sum(checker_depth[len(checker_depth) - p.query_layers:]) + 1
    step0 = 1 if p.coin_commitment_bits is not None else 0
    depth = step0 + 1 + parallel + tail + 2
    slots = slots_lookup + checker_slots + 2
    return {"rounds.protocol": slots + step0, "rounds.sequential_depth": depth, "coins": slots,
            "depth.lookup_chain": depth_lookup, "depth.checker_chain": sum(checker_depth) + 1, "depth.tail": tail}


def rounds_and_depth_b(p: CandidateB) -> dict[str, int]:
    """Candidate B: ``helper_columns``: z (one message for all tables), r, Q -> depth 3; ``gkr``: 1 + longest
    fractional chain + 2.  Plus one for step 0."""
    p.depth.validate()
    step0 = 1 if p.coin_commitment_bits is not None else 0
    if p.lookup_mode == "helper_columns":
        slots = len(p.tables) + 2
        return {"rounds.protocol": slots + step0, "rounds.sequential_depth": 3 + step0, "coins": slots,
                "depth.lookup_chain": 0, "depth.checker_chain": 0, "depth.tail": 0}
    c = p.depth.variables_per_round
    slots = depth_lookup = 0
    for tbl in p.tables:
        shards, _, n_dim = _shards(p, tbl)
        sl, dp = _fractional_chain(n_dim, c)
        slots += shards * (1 + sl) * p.lookup_repetitions
        depth_lookup = max(depth_lookup, dp)
    return {"rounds.protocol": slots + 2 + step0, "rounds.sequential_depth": step0 + 1 + depth_lookup + 2,
            "coins": slots + 2, "depth.lookup_chain": depth_lookup, "depth.checker_chain": 0, "depth.tail": 0}


RTTS_MS: tuple[str, ...] = ("0", "0.1", "1", "10", "50")


def e2e_sweep(depth: int, prover_seconds: float = 0.0, verifier_seconds: float = 0.0,
              bytes_total: int = 0, bandwidth_bytes_per_s: float | None = None) -> dict[str, float]:
    """``t.e2e.rtt_<x>ms`` = prover + verifier compute (given) + depth x RTT [+ bytes / bandwidth].  The network
    floor is ``depth x RTT``; compute is an input, never estimated here."""
    out = {}
    for r in RTTS_MS:
        t = prover_seconds + verifier_seconds + depth * float(r) / 1000.0
        if bandwidth_bytes_per_s:
            t += bytes_total / bandwidth_bytes_per_s
        out[f"t.e2e.rtt_{r}ms"] = t
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Census estimates (contract names census.*; every figure here is an ESTIMATE from an explicit cost model)
# ---------------------------------------------------------------------------------------------------------------------

#: F_t-multiplication equivalents of one multiplication in the degree-d extension (Karatsuba-style): 1, 3, 6, 9.
EXT_MULT_COST = {1: 1, 2: 3, 3: 6, 4: 9, 5: 13, 6: 17}
HM_LEAF_BYTES = 144          # HM96 O(k)-bit commitment string, k = 128 (9k bits); decommit adds x: 6k+4 bits
HM_DECOMMIT_BYTES = 100
DIGEST_BYTES = 32


def _ntt_ops(rows: int, n: int, k: int) -> int:
    """Encoding one row: inverse NTT of size k (interpolate) + forward NTT of size n; butterflies as mult-equivalents."""
    return rows * ((k // 2) * max(1, int(math.log2(k))) + (n // 2) * max(1, int(math.log2(n))))


def census_estimate_a(p: CandidateA, prover_seconds: float = 0.0) -> dict[str, float]:
    """Cost model (ESTIMATES):
    * witness_elements: per unit inputs + every internal wire (all layer outputs) + all fractional-GKR nodes
      (2 F_c values per node, ~2 nodes per leaf) as F_t equivalents;
    * committed_elements / bytes_encoded / hash_bytes: from the budget: rows x n symbols x 8 B, hashed once, plus HM
      leaf strings and the Merkle tree;
    * field_ops (F_t mult-equivalents): NTT encoding; checker GKR 12 F_c mults per (wire, copy, layer); fractional
      GKRs 15 F_c mults per leaf-equivalent; Ligero tests 3 x rows x n x ext;
    * memory_passes / bytes_moved: 3 passes over the committed matrix (read witness, write codewords, hash) + 2
      passes per sumcheck table (checker layers and fractional trees);
    * communication: opened columns + decommitments + Merkle paths + padded transcript + Ligero responses +
      challenges (+ coin commitments/openings);
    * peak memory (retained policy): codeword matrix + witness + largest live sumcheck table."""
    bud = account_candidate_a(p)
    prm = bud.parameters
    ext = prm["extension_degree"]
    ec = EXT_MULT_COST[ext]
    units, n, k, t = p.units, p.ligero.n, p.ligero.k, p.ligero.t
    rows = prm["ligero_rows"]
    wires = CHECKER_UNIT.columns_products + CHECKER_UNIT.columns_materialised      # computed wires per unit (census)
    frac_leaves = sum(sum(2**_shards(p, tb)[2] for _ in range(_shards(p, tb)[0])) for tb in p.tables)
    witness_elements = units * (p.inputs_per_unit + wires) + 4 * frac_leaves * ext
    bytes_encoded = rows * n * 8
    hash_bytes = bytes_encoded + (n * HM_LEAF_BYTES if p.leaf_commitment_bits else 0) + 2 * n * DIGEST_BYTES
    widths = CHECKER_LAYER_WIDTHS_A if p.layers is DEFAULT_LAYERS_A else tuple(2**L.s_in for L in p.layers)
    gkr_ops = 12 * ec * units * sum(widths)
    frac_ops = 15 * ec * frac_leaves
    ligero_ops = 3 * rows * n * ext
    field_ops = _ntt_ops(rows, n, k) + gkr_ops + frac_ops + ligero_ops
    layer_tables = [units * w * ext * 8 for w in widths]
    bytes_moved = 3 * bytes_encoded + prm["committed_trace_elements"] * 8 + 2 * sum(layer_tables) + 2 * (2 * frac_leaves * ext * 8)
    memory_passes = 3 + 2 * len(p.layers) + 2 * len(p.tables)
    depth = rounds_and_depth_a(p)
    coins = depth["coins"]
    p2v = (rows * t * 8 + t * (HM_DECOMMIT_BYTES + int(math.log2(n)) * DIGEST_BYTES)
           + prm["pad_elements"] * ext * 8 + (n + (k + p.ligero.l) + 2 * k) * ext * 8 + 2 * DIGEST_BYTES)
    v2p = coins * ext * 8 + rows * ext * 8
    if p.coin_commitment_bits:
        v2p += coins * (HM_LEAF_BYTES + HM_DECOMMIT_BYTES)
    peak = bytes_encoded + prm["committed_trace_elements"] * 8 + max(layer_tables)
    return {
        "census.witness_elements": witness_elements,
        "census.committed_elements": prm["committed_trace_elements"],
        "census.bytes_encoded": bytes_encoded,
        "census.hash_bytes": hash_bytes,
        "census.field_ops": field_ops,
        "census.memory_passes": memory_passes,
        "census.bytes_moved": bytes_moved,
        "rounds.protocol": depth["rounds.protocol"],
        "rounds.sequential_depth": depth["rounds.sequential_depth"],
        "transcript.bytes": p2v + v2p,
        "estimate.transcript_p2v_bytes": p2v,
        "estimate.transcript_v2p_bytes": v2p,
        "estimate.peak_bytes": peak,
        **e2e_sweep(depth["rounds.sequential_depth"], prover_seconds),
    }


def census_estimate_b(p: CandidateB, prover_seconds: float = 0.0) -> dict[str, float]:
    """Cost model (ESTIMATES), same conventions as :func:`census_estimate_a`.  Helper columns: batch inversion
    3 F_c mults per helper (Haboeck section 5) plus 2 passes over the helper columns; no sumcheck tables in
    ``helper_columns`` mode."""
    bud = account_candidate_b(p)
    prm = bud.parameters
    ext = prm["extension_degree"]
    ec = EXT_MULT_COST[ext]
    units, n, k, t = p.units, p.ligero.n, p.ligero.k, p.ligero.t
    rows = prm["ligero_rows"]
    helpers = sum(tb.queries_per_unit * units + tb.rows for tb in p.tables)
    witness_elements = units * p.witness_per_unit + sum(tb.rows for tb in p.tables) + helpers * ext
    bytes_encoded = rows * n * 8
    hash_bytes = 2 * (n * HM_LEAF_BYTES if p.leaf_commitment_bits else 0) + bytes_encoded + 4 * n * DIGEST_BYTES
    helper_ops = 3 * ec * helpers
    ligero_ops = 3 * rows * n * ext + 4 * prm["linear_constraints"] * ext
    field_ops = _ntt_ops(rows, n, k) + helper_ops + ligero_ops
    bytes_moved = 3 * bytes_encoded + prm["committed_trace_elements"] * 8 + 2 * helpers * ext * 8
    memory_passes = 3 + 2
    depth = rounds_and_depth_b(p)
    coins = depth["coins"]
    p2v = (rows * t * 8 + 2 * t * (HM_DECOMMIT_BYTES + int(math.log2(n)) * DIGEST_BYTES)
           + (n + (k + p.ligero.l) + 2 * k) * ext * 8 + 4 * DIGEST_BYTES)
    v2p = coins * ext * 8 + rows * ext * 8
    if p.coin_commitment_bits:
        v2p += coins * (HM_LEAF_BYTES + HM_DECOMMIT_BYTES)
    peak = bytes_encoded + prm["committed_trace_elements"] * 8
    return {
        "census.witness_elements": witness_elements,
        "census.committed_elements": prm["committed_trace_elements"],
        "census.bytes_encoded": bytes_encoded,
        "census.hash_bytes": hash_bytes,
        "census.field_ops": field_ops,
        "census.memory_passes": memory_passes,
        "census.bytes_moved": bytes_moved,
        "rounds.protocol": depth["rounds.protocol"],
        "rounds.sequential_depth": depth["rounds.sequential_depth"],
        "transcript.bytes": p2v + v2p,
        "estimate.transcript_p2v_bytes": p2v,
        "estimate.transcript_v2p_bytes": v2p,
        "estimate.peak_bytes": peak,
        **e2e_sweep(depth["rounds.sequential_depth"], prover_seconds),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Operating points: the first configuration (extension degree, opened columns) that meets a target
# ---------------------------------------------------------------------------------------------------------------------

def first_configuration(target_log2: int, make, trace_field: Field, n: int, k: int, l: int,
                        extension_degrees: Sequence[int] = (2, 3, 4), hash_queries: int = 2**60,
                        **kwargs) -> tuple[int, int, Budget]:
    """Smallest extension degree, then smallest ``t``, such that ``make(trace_field, challenge_field, ligero,
    hash_queries=..., **kwargs)``'s budget meets ``2**target_log2``.  ``make`` builds a ``CandidateA``/``CandidateB``
    and returns its Budget.  Raises when no configuration in the sweep meets the target (the reason is in the
    message: which component floors above the target)."""
    reasons = []
    for d in extension_degrees:
        Fc = trace_field.extension(d)
        try:
            t0 = ligero_queries_for(target_log2 - 1, n, k, l)
        except AccountingError:
            continue
        probe = make(trace_field, Fc, LigeroParams(n=n, k=k, l=l, t=t0), hash_queries=hash_queries, **kwargs)
        others = probe.total() - probe.by_component()["encoding_opening"]
        if others >= Fraction(2) ** target_log2:
            comps = {c: round(log2(v), 1) for c, v in probe.by_component().items() if c != "encoding_opening" and v}
            reasons.append(f"degree {d}: non-encoding terms alone 2^{log2(others):.1f} {comps}")
            continue
        for t in range(t0, k - l):
            bud = make(trace_field, Fc, LigeroParams(n=n, k=k, l=l, t=t), hash_queries=hash_queries, **kwargs)
            if bud.meets(target_log2):
                return d, t, bud
        reasons.append(f"degree {d}: no t < k - l = {k - l} meets the target")
    raise AccountingError(f"no configuration meets 2^{target_log2}: " + "; ".join(reasons))
