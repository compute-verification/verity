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
    "BABYBEAR", "GOLDILOCKS", "MERSENNE31", "MERSENNE61",
    "AccountingError", "Budget", "CandidateA", "CandidateB", "CheckerLayer", "Field", "LigeroParams", "LookupTable",
    "Term",
    "account_candidate_a", "account_candidate_b", "data_parallel_layer_error", "fractional_gkr_error",
    "hash_collision_bound", "ligero_error", "ligero_queries_for", "log2", "logup_gkr_error", "logup_identity_error",
    "random_combination_error", "schwartz_zippel_error", "security_bits", "sumcheck_error",
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
    def prime(name: str, p: int) -> "Field":
        return Field(name, p, p)

    def extension(self, degree: int) -> "Field":
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


def logup_identity_error(num_lookups: int, table_rows: int, F: Field, exclude_poles: bool = True,
                         repetitions: int = 1, columns: int = 1) -> Fraction:
    """Haboeck Theorem 4, first term, in the ``N`` lookups / ``|T|`` rows form: the cleared rational identity is a
    nonzero polynomial of degree at most ``N + |T| - 1``; a uniform challenge outside the table's poles hits a
    root with probability at most ``(N + |T| - 1)/(|F| - |T|)``.  (With ``M`` columns of length ``|H|`` and a
    table of length ``|H|``: ``((M+1)|H| - 1)/(|F| - |H|)``, the paper's form.)  Requires ``char(F) > N`` so that
    multiplicities are nonzero field elements (Haboeck Lemma 5).  ``repetitions`` independent challenges raise the
    term to that power (the prover must satisfy every repetition).

    ``columns > 1`` (vector-valued lookup, Haboeck 3.4): rows are compressed with powers of a second challenge
    ``beta``; the cleared numerator has ``beta``-degree at most ``(N + |T| - 1)(columns - 1)``.  *Derivation*:
    conditioning on ``beta`` and union-bounding adds ``(N + |T| - 1)(columns - 1)/|F|`` to the term."""
    if num_lookups < 1 or table_rows < 1 or columns < 1:
        raise AccountingError("need at least one lookup, one table row and one column")
    if F.characteristic <= num_lookups:
        raise AccountingError(
            f"logUp needs char(F) > total lookups: char={F.characteristic}, lookups={num_lookups} (Haboeck Lemma 5)")
    denom = F.size - table_rows if exclude_poles else F.size
    if denom <= 0:
        raise AccountingError("table larger than the challenge field")
    d = num_lookups + table_rows - 1
    term = Fraction(d, denom) + Fraction(d * (columns - 1), F.size)
    return term ** repetitions


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


def ligero_error(p: LigeroParams, F_code: Field, F_challenge: Field, analysis: str = "appendix_c") -> dict[str, Fraction]:
    """Soundness terms of the Ligero ZKIPCP, decomposed.

    ``appendix_c`` (the analysis the authors use for parameters, Longfellow 2.2 'Improvement'): for ``e < d/2``
        ``(1 - e/n)^t + ((k+l)/n)^t + (2k/n)^t + (n+3)/|F|^sigma``
    split as IRS query ``(1-e/n)^t``, linear query ``((k+l)/n)^t``, quadratic query ``(2k/n)^t``, and field terms
    IRS ``(n+1)/|F|^sigma`` (BCIKS20 unique-decoding bound ``n/|F|`` plus ``1/|F|``), linear ``1/|F|^sigma``,
    quadratic ``1/|F|^sigma``.
    ``theorem_4_7``: for ``e < (n-k)/4``: ``(d+2)/|F|^sigma + (1-e/n)^t + 2((e+2k)/n)^t``.

    Field terms use the *challenge* field (the field the random combinations are drawn from); the proximity-gap
    statement for a subfield code combined over an extension is an assumption recorded in PROTOCOL.md."""
    p.validate(F_code)
    n, k, l, t, e, s = p.n, p.k, p.l, p.t, p.proximity, p.sigma
    Fs = Fraction(F_challenge.size) ** s
    if analysis == "appendix_c":
        return {
            "irs_query": (Fraction(n - e, n)) ** t,
            "linear_query": Fraction(k + l, n) ** t,
            "quadratic_query": Fraction(2 * k, n) ** t,
            "irs_field": Fraction(n + 1) / Fs,
            "linear_field": Fraction(1) / Fs,
            "quadratic_field": Fraction(1) / Fs,
        }
    if analysis == "theorem_4_7":
        if not 4 * e < n - k:
            raise AccountingError("Theorem 4.7 needs e < (n-k)/4")
        return {
            "irs_query": (Fraction(n - e, n)) ** t,
            "linear_query": Fraction(e + 2 * k, n) ** t,
            "quadratic_query": Fraction(e + 2 * k, n) ** t,
            "field": Fraction(p.distance + 2) / Fs,
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


# The census of ``verity.ml.tc.relation`` for AMPERE_BF16_M16N8K16, ALIGN_DIV (total relation; an upper bound for the
# finite-domain checker): 146 mul, 104 lin, 172 lookups per 16-product update.  Lookups by table:
CENSUS_TABLES: tuple[LookupTable, ...] = (
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

# Placeholder checker shape for Candidate A (compute variant): ASSUMPTION until the WP3 census lands.  Per unit:
# inputs 2^8 (operands, states, hints), three computed layers of widths 2^7, 2^6, 2^5, output layer of 2^8 assertions.
DEFAULT_LAYERS_A: tuple[CheckerLayer, ...] = (
    CheckerLayer(8, 7), CheckerLayer(7, 6), CheckerLayer(6, 5), CheckerLayer(5, 8),
)


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
    inputs_per_unit: int = 256          # committed private inputs + hints per update (ASSUMPTION, see PROTOCOL.md)
    lookup_repetitions: int = 1
    hash_bits: int = 256
    hash_queries: int = 2**64
    challenge_field_elements_per_pad: int | None = None   # trace-field elements per pad; default = extension degree

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
    # Committed witness elements per update before lookup helpers: 3 * 146 = 438 aligned quadratic slots (Ligero's
    # quadratic test needs x, y, z of each product in the same column) + <= 74 values that occur only in linear or
    # lookup predicates.  ASSUMPTION until the WP3 layout lands.
    witness_per_unit: int = 512
    lookup_mode: str = "helper_columns"  # helper_columns (second commitment round) | gkr (fractional sumcheck)
    lookup_repetitions: int = 1
    hash_bits: int = 256
    hash_queries: int = 2**64

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


def _pad_and_hash(budget: Budget, p: CandidateA | CandidateB) -> None:
    budget.add("commitment_hash", f"Merkle/commitment collisions, generic q^2/2^h with q=2^{int(math.log2(p.hash_queries))}, h={p.hash_bits}",
               hash_collision_bound(p.hash_bits, p.hash_queries), kind="computational",
               note="reduction to collision resistance; not a probability over verifier coins")


def account_candidate_a(p: CandidateA) -> Budget:
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

    # ---- lookups: logUp-GKR per table, claims merged into the checker layer holding the query block -----------------
    abort = Fraction(0)          # probability a uniform challenge lands in the table image
    for tbl in p.tables:
        N = tbl.queries_per_unit * units
        lk = logup_gkr_error(N, tbl.rows, Fc, repetitions=p.lookup_repetitions, columns=tbl.columns)
        B.add("lookup", f"{tbl.name}: rational identity, N={N}, |T|={tbl.rows}, cols={tbl.columns}, x{p.lookup_repetitions}", lk["identity"])
        B.add("lookup", f"{tbl.name}: fractional GKR over 2^{int(lk['hypercube_dim'])}", lk["fractional_gkr"])
        B.add("batching", f"{tbl.name}: merge table-side/witness-side claims into the layer claim",
              random_combination_error(Fc, 2))
        n_dim = int(lk["hypercube_dim"])
        rounds += p.lookup_repetitions * (n_dim * (n_dim - 1) // 2 + (n_dim - 1))
        pads += p.lookup_repetitions * (3 * (n_dim * (n_dim - 1) // 2) + 4 * n_dim)
        abort += Fraction(tbl.rows, Fc.size)
    B.add("completeness", "logUp challenge in a public table image: rejected publicly and redrawn (no witness pole for "
          "an honest prover, whose query values lie in the table)", abort, kind="completeness")

    # ---- encoding / opening: Ligero over the committed inputs, multiplicities and pads --------------------------------
    committed = units * p.inputs_per_unit + sum(t.rows for t in p.tables) + pads * per_pad
    lig = ligero_error(p.ligero, Ft, Fc)
    for name, val in lig.items():
        B.add("encoding_opening", f"Ligero {name} (n={p.ligero.n}, k={p.ligero.k}, l={p.ligero.l}, t={p.ligero.t}, e={p.ligero.proximity})", val)
    _pad_and_hash(B, p)

    rows = math.ceil(committed / p.ligero.l) + 3
    B.parameters.update({
        "K": p.K, "B": p.B, "units": units, "copy_bits_b": b,
        "trace_field": Ft.name, "challenge_field": Fc.name, "extension_degree": ext,
        "checker_layers": len(p.layers), "sumcheck_rounds_total": rounds, "pad_elements": pads,
        "committed_trace_elements": committed, "ligero_rows": rows,
        "opened_trace_elements": rows * p.ligero.t,
        "merkle_leaves": p.ligero.n,
    })
    return B


def account_candidate_b(p: CandidateB) -> Budget:
    Fc, Ft = p.challenge_field, p.trace_field
    ext = _extension_degree(Ft, Fc)
    B = Budget("B: direct constraints in interactive ZK Ligero + logUp (interactive)")
    units = p.units
    committed = units * p.witness_per_unit + sum(t.rows for t in p.tables)
    abort = Fraction(0)          # probability a uniform challenge lands in the table image
    rounds = 0
    for tbl in p.tables:
        N = tbl.queries_per_unit * units
        if p.lookup_mode == "helper_columns":
            B.add("lookup", f"{tbl.name}: rational identity, N={N}, |T|={tbl.rows}, cols={tbl.columns}, x{p.lookup_repetitions}",
                  logup_identity_error(N, tbl.rows, Fc, repetitions=p.lookup_repetitions, columns=tbl.columns))
            committed += p.lookup_repetitions * (N + tbl.rows) * ext     # inverse columns h = 1/(alpha - f) in F_c
        elif p.lookup_mode == "gkr":
            lk = logup_gkr_error(N, tbl.rows, Fc, repetitions=p.lookup_repetitions, columns=tbl.columns)
            B.add("lookup", f"{tbl.name}: rational identity, N={N}, |T|={tbl.rows}, cols={tbl.columns}, x{p.lookup_repetitions}", lk["identity"])
            B.add("lookup", f"{tbl.name}: fractional GKR over 2^{int(lk['hypercube_dim'])}", lk["fractional_gkr"])
            n_dim = int(lk["hypercube_dim"])
            rounds += p.lookup_repetitions * (n_dim * (n_dim - 1) // 2 + (n_dim - 1))
            committed += p.lookup_repetitions * (3 * (n_dim * (n_dim - 1) // 2) + 4 * n_dim) * ext
        else:
            raise AccountingError(f"unknown lookup_mode {p.lookup_mode!r}")
        abort += Fraction(tbl.rows, Fc.size)
    B.add("completeness", "logUp challenge in a public table image: rejected publicly and redrawn (no witness pole for "
          "an honest prover, whose query values lie in the table)", abort, kind="completeness")
    lig = ligero_error(p.ligero, Ft, Fc)
    for name, val in lig.items():
        B.add("encoding_opening", f"Ligero {name} (n={p.ligero.n}, k={p.ligero.k}, l={p.ligero.l}, t={p.ligero.t}, e={p.ligero.proximity})", val)
    _pad_and_hash(B, p)
    rows = math.ceil(committed / p.ligero.l) + 3
    B.parameters.update({
        "K": p.K, "B": p.B, "units": units, "trace_field": Ft.name, "challenge_field": Fc.name,
        "extension_degree": ext, "lookup_mode": p.lookup_mode, "sumcheck_rounds_total": rounds,
        "committed_trace_elements": committed, "ligero_rows": rows,
        "opened_trace_elements": rows * p.ligero.t,
        "commitment_rounds": 2 if p.lookup_mode == "helper_columns" else 1,
        "merkle_leaves": p.ligero.n,
    })
    return B
