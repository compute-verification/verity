"""Polynomial-commitment costing for the two committed objects of the campaign (PCS-comparison lane, wave 1).

The objects are taken from the soundness calculator, never copied: A-GKR commits its inputs, hints and query
wires (``account_candidate_a(...).parameters["committed_trace_elements"]``, 181M elements at ``K=1536, B=4096``) and
opens them at a few points of the cubic extension; B-Ligero commits its whole trace (780M elements) and tests it
by Ligero's linear / quadratic row tests.  Every PCS below is costed on both objects at one operating point
(``2^-128`` primary, ``2^-80`` diagnostic), hash-based, transparent and *interactive* (no Fiat-Shamir, so no
grinding), with the row keys of ``bench.contract`` (``census.*``, ``rounds.*``) plus ``proof_bytes``,
``verifier_hashes``, ``verifier_field_ops``.

Schemes and the soundness statements used (all statistical terms are per PCS; the candidate's own terms are the
calculator's):

* **Ligero** (Ames-Hazay-Ishai-Venkitasubramaniam 2022/1608, Appendix C): reproduced through
  :func:`accounting.first_configuration` -- the current PROTOCOL.md numbers.
* **Brakedown** (Golovnev-Lee-Setty-Thaler-Wahby 2021/1043, section 3): expander code, rate 0.65 and relative
  distance 0.07 (the 128-bit large-field instantiation of ``ark-poly-commit``; Haboeck 2023/769 gives tighter
  parameters); test error ``N/|F| + (1 - gamma/3)^t``.
* **FRI** (Ben-Sasson-Bentov-Horesh-Riabzev 2018) at rates 1/2, 1/4, 1/8, three soundness regimes:
  ``proven_udr`` -- unique decoding, BCIKS20 (Ben-Sasson-Carmon-Ishai-Kopparty-Saraf, proximity gaps, Theorem 1.2):
  proximity-gap error ``n/|F|`` per correlated-agreement application, query term ``((1+rho)/2)^s``;
  ``proven_ldr`` -- list decoding up to the Johnson bound, BCIKS20 Theorem 8.3 as stated in the ethSTARK documentation
  (ePrint 2021/582, section 5.10; the ``m`` parameter is optimised as in Winterfell's estimate): pre-query error
  ``(m+1/2)^7 n^2 / (2 rho^{3/2} |F|)`` per application, query term ``(sqrt(rho) (1 + 1/(2m)))^s``;
  ``conjectured`` -- ethSTARK Conjecture 1 (RS list-decodability up to capacity, DEEP-FRI Conjecture 2.3): error
  ``1/|F|`` per round and query term ``rho^s``.  The number of applications is one per batching step plus one per
  folding round (our union bound; a conservative reading).
* **BaseFold** (Zeilberger-Chen-Fisch 2023/1705): FRI folding interleaved with a sumcheck, the multilinear scheme
  A's openings need natively; list-decoding regime by Haboeck 2024/1571; costs = FRI + one sumcheck.
* **STIR** (Arnon-Chiesa-Fenzi-Yogev 2024/390) and **WHIR** (Arnon-Chiesa-Fenzi-Yogev 2024/1586): the rate improves
  by ``2^(k-1)`` per round (fold ``2^k``, domain halves), so later rounds need far fewer queries; the same three
  regimes are applied per round (WHIR's up-to-capacity statement needs its *mutual* correlated agreement
  conjecture, Conjecture 4.12 of 2024/1586).  Both are deeper than FRI with the same fold: STIR issues three
  challenges per round, WHIR ``k + 2`` (its sumcheck rounds are sequential).
* **Blaze** (Brehm-Chen-Fisch-Resch-Rothblum-Zeilberger 2024/1609) / Binius (Diamond-Posen 2023/1784):
  binary-field Ligero/RAA-style commitments.  Order-of-magnitude commitment cost only: the frozen checker is over
  Goldilocks, so these would change the trace field, which is a re-arithmetisation, not a PCS swap.

Cost conventions (ESTIMATES, same as ``census_estimate_*``): a field multiplication in ``F_{p^d}`` costs
``EXT_MULT_COST[d]`` trace-field mult-equivalents; an NTT butterfly is one; a hash call absorbs
:data:`HASH_RATE_BYTES`; digests are 32 B; ``rounds.sequential_depth`` is the PCS's own depth (its challenges that
cannot be batched), and ``total.sequential_depth`` adds the candidate's non-PCS depth from the calculator.  For
B-Ligero under a point-opening PCS, its linear/quadratic row tests have no meaning: the constraints must be
checked by a sumcheck first (Spartan-style, ``ceil(log2 N_B)`` sequential rounds; ``adapter.*`` columns), which is
the honest price of moving B off Ligero.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from verity_numerical.security import accounting as acc
from verity_numerical.security.accounting import (
    DIGEST_BYTES,
    EXT_MULT_COST,
    GOLDILOCKS,
    AccountingError,
    Field,
    LigeroParams,
    account_candidate_a,
    account_candidate_b,
    census_estimate_a,
    census_estimate_b,
    first_configuration,
    log2,
)

__all__ = [
    "COLUMNS",
    "CRATES",
    "HASH_RATE_BYTES",
    "REGIMES",
    "Committed",
    "PcsRow",
    "basefold_row",
    "blaze_row",
    "brakedown_row",
    "committed_objects",
    "fri_field_term",
    "fri_queries",
    "fri_row",
    "ligero_row",
    "main",
    "queries_per_round",
    "stir_row",
    "table",
    "whir_row",
]

GL = Field.prime("GL", GOLDILOCKS)
HASH_RATE_BYTES = 64          # one compression / permutation absorbs 64 B (SHA-256 block; Poseidon2 rate 8 x 8 B)
ELEM_BYTES = 8
REGIMES = ("proven_udr", "proven_ldr", "conjectured")
#: Throughput constants for the *indicative* GPU prover-seconds column (H100-class, order of magnitude; labelled).
GPU_HASH_BYTES_PER_S = 10e9
GPU_FIELD_OPS_PER_S = 1e11
GPU_HBM_BYTES_PER_S = 2e12

COLUMNS = (
    "pcs", "object", "regime", "params",
    "census.committed_elements", "census.bytes_encoded", "census.hash_bytes", "census.field_ops",
    "census.memory_passes", "census.bytes_moved",
    "prover_hashes", "proof_bytes", "verifier_hashes", "verifier_field_ops", "queries",
    "rounds.protocol", "rounds.sequential_depth", "adapter.sequential_depth", "total.sequential_depth",
    "soundness.log2", "zk", "est.peak_bytes", "est.prover_seconds_gpu", "code",
)


# ---------------------------------------------------------------------------------------------------------------------
# The two committed objects, from the calculator
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Committed:
    """One committed object.  ``elements`` in the trace field; ``points`` evaluation claims (A) or ``None`` for an
    object whose native opening is Ligero's row tests (B); ``other_depth`` is the candidate's sequential depth
    outside the PCS (calculator: A's 381 includes Ligero's two PCS challenges, B's 3 includes r and Q)."""

    name: str
    candidate: str
    elements: int
    points: int | None
    ext: int
    other_depth: int
    budget: acc.Budget
    ligero: LigeroParams

    @property
    def log2_elements(self) -> int:
        return math.ceil(math.log2(self.elements))


def _mk_a(Ft, Fc, lig, **kw):
    return account_candidate_a(acc.CandidateA(K=1536, B=kw.pop("B"), trace_field=Ft, challenge_field=Fc, ligero=lig, **kw))


def _mk_b(Ft, Fc, lig, **kw):
    return account_candidate_b(acc.CandidateB(K=1536, B=kw.pop("B"), trace_field=Ft, challenge_field=Fc, ligero=lig, **kw))


def committed_objects(target_log2: int = -128, B: int = 4096, K: int = 1536) -> dict[str, Committed]:
    """The first Ligero configuration of each candidate at ``target_log2`` (``first_configuration``: smallest
    extension degree, then smallest ``t``) and the objects it commits.  Nothing is copied from PROTOCOL.md."""
    if K != 1536:
        raise AccountingError("the campaign objects are costed at K = 1536 (the architecture decision's VU)")
    out = {}
    for name, mk in (("A", _mk_a), ("B", _mk_b)):
        d, t, bud = first_configuration(target_log2, mk, GL, 2**14, 2**12, 3840, B=B, hash_queries=2**60)
        out[name] = Committed(
            name=f"{name}:{'inputs+hints+query-wires' if name == 'A' else 'trace'}", candidate=name,
            elements=bud.parameters["committed_trace_elements"], points=2 if name == "A" else None, ext=d,
            other_depth=bud.parameters["rounds.sequential_depth"] - (2 if name == "A" else 3), budget=bud,
            ligero=LigeroParams(n=2**14, k=2**12, l=3840, t=t))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# FRI-family soundness
# ---------------------------------------------------------------------------------------------------------------------

def fri_field_term(regime: str, rate: Fraction, n: int, F: Field, m: int = 3) -> Fraction:
    """The non-query error of one correlated-agreement application (see the module docstring for sources)."""
    if regime == "proven_udr":
        return Fraction(n, F.size)
    if regime == "proven_ldr":
        return Fraction((2 * m + 1) ** 7 * n * n, 2**7 * 2 * F.size) / (rate ** Fraction(3, 2))
    if regime == "conjectured":
        return Fraction(1, F.size)
    raise AccountingError(f"unknown regime {regime!r}")


def _per_query(regime: str, rate: Fraction, m: int = 3) -> float:
    """``1 - theta``: the probability one query misses, as a float (the query terms are powers of it)."""
    if regime == "proven_udr":
        return float((1 + rate) / 2)
    if regime == "proven_ldr":
        return math.sqrt(float(rate)) * (1 + 1 / (2 * m))
    if regime == "conjectured":
        return float(rate)
    raise AccountingError(f"unknown regime {regime!r}")


def fri_queries(target_log2: float, rate: Fraction, regime: str, n: int, F: Field, applications: int,
                ) -> tuple[int, int, dict[str, Fraction]]:
    """Smallest query count ``s`` (and, for ``proven_ldr``, the ``m`` that first admits it) such that
    ``applications`` field terms plus the query term stay within ``2**target_log2``.  Raises when the field terms
    alone exceed the target (the field is too small for the regime at this ``n``)."""
    best = None
    ms = range(3, 200) if regime == "proven_ldr" else (3,)
    for m in ms:
        field_err = applications * fri_field_term(regime, rate, n, F, m)
        if field_err >= Fraction(2) ** target_log2:
            if regime == "proven_ldr":
                break                                         # m^7 only grows: stop
            raise AccountingError(f"{regime}: {applications} field terms alone 2^{log2(field_err):.1f} exceed 2^{target_log2}")
        rem = float(Fraction(2) ** target_log2 - field_err)
        s = math.ceil(math.log(rem) / math.log(_per_query(regime, rate, m)))
        if best is None or s < best[0]:
            best = (s, m, {"field": field_err, "query": Fraction(_per_query(regime, rate, m)) ** s})
    if best is None:
        raise AccountingError(f"{regime}: no m in [3, 200) keeps the field terms below 2^{target_log2} at n=2^{math.log2(n):.0f}")
    return best


def queries_per_round(target_log2: float, rates: Sequence[Fraction], sizes: Sequence[int], regime: str, F: Field,
                      ) -> tuple[list[int], Fraction]:
    """STIR/WHIR: each round gets an equal share of the target (``target - log2(rounds)``) at its own rate and
    domain size; returns the per-round query counts and the summed error."""
    share = target_log2 - math.log2(len(rates))
    qs, err = [], Fraction(0)
    for r, n in zip(rates, sizes):
        s, _m, terms = fri_queries(share, r, regime, n, F, applications=2)
        qs.append(s)
        err += terms["field"] + terms["query"]
    return qs, err


# ---------------------------------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class PcsRow:
    values: dict

    def __getitem__(self, k):
        return self.values[k]

    def as_dict(self) -> dict:
        return {k: self.values.get(k) for k in COLUMNS} | {k: v for k, v in self.values.items() if k not in COLUMNS}


def _finish(v: dict) -> PcsRow:
    v["prover_hashes"] = math.ceil(v["census.hash_bytes"] / HASH_RATE_BYTES)
    v["total.sequential_depth"] = v["rounds.sequential_depth"] + v.get("adapter.sequential_depth", 0) + v["other_depth"]
    v["est.prover_seconds_gpu"] = (v["census.hash_bytes"] / GPU_HASH_BYTES_PER_S + v["census.field_ops"] / GPU_FIELD_OPS_PER_S
                                   + v["census.bytes_moved"] / GPU_HBM_BYTES_PER_S)
    return PcsRow(v)


def _adapter(obj: Committed, ec: int) -> dict:
    """B-Ligero off Ligero: a Spartan-style sumcheck over its ``lin`` and ``mul`` constraints (two sumchecks in
    parallel, degree 3, ``ceil(log2 N_B)`` rounds; prover ~8 F_c mults per element; error ``3 log N / |F|``)."""
    if obj.points is not None:
        return {"adapter.sequential_depth": 0, "adapter.field_ops": 0, "adapter.proof_bytes": 0, "adapter.error": Fraction(0)}
    v = obj.log2_elements
    return {"adapter.sequential_depth": v + 1, "adapter.field_ops": 8 * ec * obj.elements,
            "adapter.proof_bytes": 2 * v * 4 * obj.ext * ELEM_BYTES,
            "adapter.error": Fraction(2 * 3 * v, GL.size ** obj.ext)}


def ligero_row(obj: Committed) -> PcsRow:
    """The current scheme, from the calculator's own estimates (PROTOCOL.md 9.4)."""
    Fc = GL.extension(obj.ext)
    if obj.candidate == "A":
        p = acc.CandidateA(K=1536, B=obj.budget.parameters["B"], trace_field=GL, challenge_field=Fc, ligero=obj.ligero, hash_queries=2**60)
        est = census_estimate_a(p)
        pcs_depth, protocol = 2, 2
    else:
        p = acc.CandidateB(K=1536, B=obj.budget.parameters["B"], trace_field=GL, challenge_field=Fc, ligero=obj.ligero, hash_queries=2**60)
        est = census_estimate_b(p)
        pcs_depth, protocol = 3, 3
    n, t = obj.ligero.n, obj.ligero.t
    rows = obj.budget.parameters["ligero_rows"]
    lig = obj.budget.by_component()["encoding_opening"]
    v = {
        "pcs": "Ligero RS (rate 1/4)", "object": obj.name, "regime": "proven (AHIV22 App. C, e < d/2)",
        "params": f"n=2^14 k=2^12 l=3840 t={t} F_c=GL^{obj.ext}",
        **{k: est[k] for k in est if k.startswith("census.")},
        "proof_bytes": est["estimate.transcript_p2v_bytes"],
        "verifier_hashes": t * (math.ceil(rows * ELEM_BYTES / HASH_RATE_BYTES) + int(math.log2(n)) + 2),
        "verifier_field_ops": 3 * t * rows * obj.ext + 3 * n * EXT_MULT_COST[obj.ext],
        "queries": t, "rounds.protocol": protocol, "rounds.sequential_depth": pcs_depth,
        "adapter.sequential_depth": 0, "other_depth": obj.other_depth,
        "soundness.log2": log2(lig),
        "zk": "native: 1 random row per test + k > l + t (AHIV22 Lemma 4.15), HM96 salted leaves; cost ~3 rows",
        "est.peak_bytes": est["estimate.peak_bytes"],
        "code": "libzk (Google, Apache-2.0, active); brakedown (Rust, ark-poly-commit LinearCodePCS); GPU: none published",
    }
    return _finish(v)


def _fri_layout(obj: Committed, rows_log2: int) -> tuple[int, int]:
    rows = 2**rows_log2
    return rows, math.ceil(obj.elements / rows)


def fri_row(obj: Committed, target_log2: int, rate: Fraction, regime: str, rows_log2: int = 20, arity_log2: int = 1,
            *, pcs_name: str = "FRI", ext: int | None = None) -> PcsRow:
    """Batched FRI over a ``cols x 2^rows_log2`` matrix (one Merkle tree, row leaves), one FRI on the random / eq
    combination of the columns; ``arity_log2`` folding.  For A the combination is fixed by the evaluation point (no
    challenge); with several claims one batching challenge is added."""
    ext = ext or obj.ext
    Fc = GL.extension(ext)
    ec = EXT_MULT_COST[ext]
    rows, cols = _fri_layout(obj, rows_log2)
    n = int(rows / rate)
    r = math.ceil(rows_log2 / arity_log2)
    claims = obj.points or 1
    applications = (1 if claims > 1 else 0) + r
    s, m, terms = fri_queries(target_log2, rate, regime, n, Fc, applications)
    sizes = [n // 2 ** (arity_log2 * i) for i in range(r)]
    enc = cols * ((rows // 2) * rows_log2 + (n // 2) * int(math.log2(n)))
    combine = n * cols * ext
    fold = 2 * ec * sum(sizes)
    bytes_encoded = n * cols * ELEM_BYTES
    hash_bytes = bytes_encoded + 2 * n * DIGEST_BYTES + sum(sz * ext * ELEM_BYTES + 2 * sz * DIGEST_BYTES for sz in sizes[1:])
    per_query = cols * ELEM_BYTES + int(math.log2(n)) * DIGEST_BYTES + sum(
        (2**arity_log2 - 1) * ext * ELEM_BYTES + int(math.log2(sz)) * DIGEST_BYTES for sz in sizes[1:])
    proof = (r + 1) * DIGEST_BYTES + s * per_query + ext * ELEM_BYTES
    vh = s * (math.ceil(cols * ELEM_BYTES / HASH_RATE_BYTES) + int(math.log2(n)) + sum(1 + int(math.log2(sz)) for sz in sizes[1:]))
    vf = s * (cols * ext + r * 2**arity_log2 * ec * 2)
    ad = _adapter(obj, ec)
    v = {
        "pcs": f"{pcs_name} rate {rate}", "object": obj.name, "regime": regime + (f" (m={m})" if regime == "proven_ldr" else ""),
        "params": f"rows=2^{rows_log2} cols={cols} n=2^{int(math.log2(n))} fold=2^{arity_log2} rounds={r} s={s} F_c=GL^{ext}",
        "census.committed_elements": obj.elements, "census.bytes_encoded": bytes_encoded, "census.hash_bytes": hash_bytes,
        "census.field_ops": enc + combine + fold + ad["adapter.field_ops"], "census.memory_passes": 5,
        "census.bytes_moved": 3 * bytes_encoded + obj.elements * ELEM_BYTES + 3 * n * ext * ELEM_BYTES,
        "proof_bytes": proof + ad["adapter.proof_bytes"], "verifier_hashes": vh, "verifier_field_ops": vf, "queries": s,
        "rounds.protocol": applications + 1, "rounds.sequential_depth": applications + 1,
        "adapter.sequential_depth": ad["adapter.sequential_depth"], "other_depth": obj.other_depth,
        "soundness.log2": log2(terms["field"] + terms["query"] + ad["adapter.error"]),
        "zk": f"add s={s} random rows per column + 1 blinding codeword on the batched polynomial (ethSTARK 5.5), salted leaves; "
              f"+{cols * s + n * ext:,} elements",
        "est.peak_bytes": bytes_encoded + obj.elements * ELEM_BYTES + n * ext * ELEM_BYTES,
        "code": "Plonky3 fri/ (MIT/Apache-2.0, active); winterfell fri (MIT); GPU: ICICLE (Ingonyama, Apache-2.0 since v3) NTT+Poseidon2, "
                "sppark (Supranational, Apache-2.0) NTT+Merkle, SP1/Plonky3 CUDA forks",
        "fri.terms": {k: log2(x) for k, x in terms.items()},
    }
    return _finish(v)


def basefold_row(obj: Committed, target_log2: int, rate: Fraction, regime: str, rows_log2: int = 20, *,
                 ext: int | None = None) -> PcsRow:
    """BaseFold(RS) = FRI (arity 2) + a sumcheck over the ``rows_log2`` folded variables, sharing its challenges."""
    ext = ext or obj.ext
    row = fri_row(obj, target_log2, rate, regime, rows_log2, 1, pcs_name="BaseFold(RS)", ext=ext)
    v = dict(row.values)
    ec = EXT_MULT_COST[ext]
    rows = 2**rows_log2
    v["census.field_ops"] += 4 * ec * rows
    v["proof_bytes"] += rows_log2 * 3 * ext * ELEM_BYTES
    v["verifier_field_ops"] += 3 * rows_log2 * ec
    v["soundness.log2"] = math.log2(2.0 ** v["soundness.log2"] + 2 * rows_log2 / float(GL.size ** ext))
    v["regime"] = v["regime"] + "; ZCF23 UDR / Haboeck 2024/1571 LDR"
    v["code"] = ("Basefold reference (Rust, hadasz/Zeilberger, MIT, research-grade); Plonky3 has no BaseFold; "
                 "scroll-tech/ceno mpcs basefold (Apache-2.0, active); GPU: none published")
    return _finish(v)


def _stir_like(obj: Committed, target_log2: int, rate: Fraction, regime: str, rows_log2: int, k: int, name: str,
               depth_per_round: int, sumcheck: bool, ext: int | None = None) -> PcsRow:
    ext = ext or obj.ext
    Fc = GL.extension(ext)
    ec = EXT_MULT_COST[ext]
    rows, cols = _fri_layout(obj, rows_log2)
    n0 = int(rows / rate)
    r = math.ceil(rows_log2 / k)
    degs = [2 ** max(0, rows_log2 - k * i) for i in range(r)]
    sizes = [n0 // 2**i for i in range(r)]
    rates = [Fraction(d, sz) for d, sz in zip(degs, sizes)]
    qs, err = queries_per_round(target_log2, rates, sizes, regime, Fc)
    enc0 = cols * ((rows // 2) * rows_log2 + (n0 // 2) * int(math.log2(n0)))
    re_enc = sum(ec * ((d // 2) * max(1, int(math.log2(d))) + (sz // 2) * int(math.log2(sz))) for d, sz in zip(degs[1:], sizes[1:]))
    combine = n0 * cols * ext
    sc = 3 * ec * sum(degs) if sumcheck else 0
    bytes_encoded = n0 * cols * ELEM_BYTES
    hash_bytes = bytes_encoded + 2 * n0 * DIGEST_BYTES + sum(sz * ext * ELEM_BYTES + 2 * sz * DIGEST_BYTES for sz in sizes[1:])
    proof = (r + 1) * DIGEST_BYTES
    vh = 0
    for i, q in enumerate(qs):
        leaf = cols * ELEM_BYTES if i == 0 else 2**k * ext * ELEM_BYTES
        path = int(math.log2(sizes[i])) - (0 if i == 0 else k)
        proof += q * (leaf + path * DIGEST_BYTES) + 2 * ext * ELEM_BYTES          # + out-of-domain answer
        vh += q * (math.ceil(leaf / HASH_RATE_BYTES) + path)
    if sumcheck:
        proof += rows_log2 * 3 * ext * ELEM_BYTES
    vf = sum(q * (cols * ext if i == 0 else 2**k * ec * 2) for i, q in enumerate(qs)) + r * 2**k * ec + (3 * rows_log2 * ec if sumcheck else 0)
    ad = _adapter(obj, ec)
    v = {
        "pcs": f"{name} rate {rate} fold 2^{k}", "object": obj.name, "regime": regime + (" + mutual CA conj." if regime == "conjectured" and sumcheck else ""),
        "params": f"rows=2^{rows_log2} cols={cols} n=2^{int(math.log2(n0))} rounds={r} queries/round={qs} F_c=GL^{ext}",
        "census.committed_elements": obj.elements, "census.bytes_encoded": bytes_encoded, "census.hash_bytes": hash_bytes,
        "census.field_ops": enc0 + re_enc + combine + sc + ad["adapter.field_ops"], "census.memory_passes": 5 + r,
        "census.bytes_moved": 3 * bytes_encoded + obj.elements * ELEM_BYTES + 3 * sum(sizes) * ext * ELEM_BYTES,
        "proof_bytes": proof + ad["adapter.proof_bytes"], "verifier_hashes": vh, "verifier_field_ops": vf, "queries": sum(qs),
        "rounds.protocol": depth_per_round * r + 1, "rounds.sequential_depth": depth_per_round * r + 1,
        "adapter.sequential_depth": ad["adapter.sequential_depth"], "other_depth": obj.other_depth,
        "soundness.log2": log2(err + ad["adapter.error"]),
        "zk": "no ZK variant in the papers; same masking as FRI applies to the first oracle only (the re-encodings are of the "
              "blinded polynomial); unreviewed",
        "est.peak_bytes": bytes_encoded + obj.elements * ELEM_BYTES + n0 * ext * ELEM_BYTES,
        "code": ("WorldcoinFoundation/whir (Rust, MIT/Apache-2.0, active 2025); Plonky3 has no WHIR/STIR; GPU: none published"
                 if sumcheck else "STIR reference in WizardOfMenlo/stir (Rust, MIT/Apache-2.0, research-grade); GPU: none"),
    }
    return _finish(v)


def stir_row(obj: Committed, target_log2: int, rate: Fraction, regime: str, rows_log2: int = 20, k: int = 4, *,
             ext: int | None = None) -> PcsRow:
    """STIR: fold ``2^k``, three challenges per round (fold, out-of-domain point, shift queries + combination)."""
    return _stir_like(obj, target_log2, rate, regime, rows_log2, k, "STIR", 3, sumcheck=False, ext=ext)


def whir_row(obj: Committed, target_log2: int, rate: Fraction, regime: str, rows_log2: int = 20, k: int = 4, *,
             ext: int | None = None) -> PcsRow:
    """WHIR: STIR's shape with ``k`` sequential sumcheck rounds per round (multilinear constraints natively)."""
    return _stir_like(obj, target_log2, rate, regime, rows_log2, k, "WHIR", k + 2, sumcheck=True, ext=ext)


def brakedown_row(obj: Committed, target_log2: int, rate: Fraction = Fraction(65, 100), gamma: Fraction = Fraction(7, 100),
                  encode_ops_per_symbol: int = 20) -> PcsRow:
    """Brakedown tensor commitment: ``rows x cols`` with ``rows = sqrt(ext N / t)`` balancing the combined row
    against the opened columns; test error ``M/|F| + (1 - gamma/3)^t`` (GLSTW21 section 3), evaluation phase
    ``(1 - 2 gamma/3)^t``; no NTT, ``encode_ops_per_symbol`` mult-equivalents per message symbol (Haboeck 2023/769:
    ~6-7 nonzeros per row in each of the two sparse products, twice recursively; a stated estimate)."""
    ext = obj.ext
    Fc = GL.extension(ext)
    ec = EXT_MULT_COST[ext]
    t = math.ceil((target_log2 - 1) / math.log2(1 - float(gamma) / 3))
    rows = max(2, math.ceil(math.sqrt(ext * obj.elements / t)))
    cols = math.ceil(obj.elements / rows)
    M = math.ceil(cols / rate)
    field_err = Fraction(M, Fc.size) + Fraction(1, Fc.size)
    err = field_err + Fraction(1 - gamma / 3) ** t + Fraction(1 - 2 * gamma / 3) ** t
    bytes_encoded = rows * M * ELEM_BYTES
    hash_bytes = bytes_encoded + 2 * M * DIGEST_BYTES
    ops = rows * cols * encode_ops_per_symbol + rows * cols * ext + M * encode_ops_per_symbol * ec
    proof = cols * ext * ELEM_BYTES * 2 + t * (rows * ELEM_BYTES + int(math.log2(M)) * DIGEST_BYTES)
    vh = t * (math.ceil(rows * ELEM_BYTES / HASH_RATE_BYTES) + math.ceil(math.log2(M)))
    vf = 2 * M * encode_ops_per_symbol * ec + 2 * t * rows * ext
    ad = _adapter(obj, ec)
    v = {
        "pcs": f"Brakedown expander (rate {float(rate):.2f}, dist {float(gamma):.2f})", "object": obj.name,
        "regime": "proven (GLSTW21 Thm 3.1 / Claim 3.1)",
        "params": f"rows={rows} cols={cols} M={M} t={t} F_c=GL^{ext}",
        "census.committed_elements": obj.elements, "census.bytes_encoded": bytes_encoded, "census.hash_bytes": hash_bytes,
        "census.field_ops": ops + ad["adapter.field_ops"], "census.memory_passes": 3,
        "census.bytes_moved": 3 * bytes_encoded + obj.elements * ELEM_BYTES,
        "proof_bytes": proof + ad["adapter.proof_bytes"], "verifier_hashes": vh, "verifier_field_ops": vf, "queries": t,
        "rounds.protocol": 2, "rounds.sequential_depth": 2,
        "adapter.sequential_depth": ad["adapter.sequential_depth"], "other_depth": obj.other_depth,
        "soundness.log2": log2(err + ad["adapter.error"]),
        "zk": "not in the paper; Ligero-style random row + masking row is a straightforward add (unreviewed)",
        "est.peak_bytes": bytes_encoded + obj.elements * ELEM_BYTES,
        "code": "ark-poly-commit linear_codes (arkworks, MIT/Apache-2.0, maintained); Brakedown/Lasso in a16z jolt (Rust, active) "
                "uses Hyrax/Zeromorph instead; GPU: none published (sparse mat-vec, irregular access)",
    }
    return _finish(v)


def blaze_row(obj: Committed, target_log2: int, rate: Fraction = Fraction(1, 4)) -> PcsRow:
    """Binary-field commitment (Blaze RAA / Binius Ligero-style): order-of-magnitude *commitment* cost with the
    Goldilocks elements written as 64 bits each.  Not a drop-in: the checker's arithmetic is Goldilocks."""
    bits = 64 * obj.elements
    enc_bits = int(bits / rate)
    t = math.ceil((target_log2 - 1) / math.log2(1 - float(Fraction(1, 8))))   # RAA distance ~1/4 at rate 1/4: query (1-d/2)^t
    v = {
        "pcs": f"Blaze/Binius binary (RAA, rate {rate})", "object": obj.name,
        "regime": "coarse: RAA relative distance 1/4 (BCFRRZ24 Table 1 order), queries (1 - d/2)^t",
        "params": f"bits=2^{math.log2(bits):.1f} t={t}; FIELD CHANGE REQUIRED (GF(2^k) tower, not Goldilocks)",
        "census.committed_elements": obj.elements, "census.bytes_encoded": enc_bits // 8,
        "census.hash_bytes": enc_bits // 8 + 2 * 2**20 * DIGEST_BYTES,
        "census.field_ops": 3 * enc_bits // 64, "census.memory_passes": 3,
        "census.bytes_moved": 3 * enc_bits // 8 + bits // 8,
        "proof_bytes": t * (math.ceil(math.sqrt(bits)) // 8 + 20 * DIGEST_BYTES), "verifier_hashes": t * 22,
        "verifier_field_ops": 3 * math.ceil(math.sqrt(bits)), "queries": t,
        "rounds.protocol": 2, "rounds.sequential_depth": 2, "adapter.sequential_depth": 0, "other_depth": obj.other_depth,
        "soundness.log2": None,
        "zk": "Blaze: none; Binius: none published",
        "est.peak_bytes": enc_bits // 8 + bits // 8,
        "code": "IrreducibleOSS/binius (Rust, Apache-2.0, active; GPU work announced, not open); Blaze: no public code",
    }
    return _finish(v)


# ---------------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------------

RATES = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 8))

CITATIONS = {
    "AHIV22": "Ames, Hazay, Ishai, Venkitasubramaniam. Ligero: lightweight sublinear arguments without a trusted setup. ePrint 2022/1608 (Thm 4.7, App. C, Lemma 4.15).",
    "GLSTW21": "Golovnev, Lee, Setty, Thaler, Wahby. Brakedown: linear-time and field-agnostic SNARKs for R1CS. ePrint 2021/1043 (section 3: test error N/|F| + (1 - gamma/3)^t).",
    "Hab23": "Haboeck. Brakedown's expander code. ePrint 2023/769 (tighter 128-bit parameters).",
    "BBHR18": "Ben-Sasson, Bentov, Horesh, Riabzev. Fast Reed-Solomon interactive oracle proofs of proximity. ICALP 2018.",
    "BCIKS20": "Ben-Sasson, Carmon, Ishai, Kopparty, Saraf. Proximity gaps for Reed-Solomon codes. FOCS 2020 / ePrint 2020/654 (Thm 1.2 unique decoding n/|F|; Thm 8.3 list decoding).",
    "ethSTARK": "Ben-Sasson (StarkWare). ethSTARK documentation. ePrint 2021/582 v1.2 (section 5.10: proven bound (m+1/2)^7 n^2/(2 rho^{3/2}|F|), query (sqrt(rho)(1+1/2m))^s; Conjecture 1: up to capacity, rho^s).",
    "BGKS19": "Ben-Sasson, Goldberg, Kopparty, Saraf. DEEP-FRI. ITCS 2020 / arXiv 1903.12243 (Conjecture 2.3: RS list-decodable up to capacity).",
    "Hab22": "Haboeck. A summary on the FRI low degree test. ePrint 2022/1216.",
    "BCHKS25": "Ben-Sasson, Carmon, Haboeck, Kopparty, Saraf. Proximity gaps for Reed-Solomon codes, revisited (sharper correlated agreement). 2025 (as summarised by LambdaClass, 'A sharper look at FRI'); not used in the numbers.",
    "BCKL23": "Block, Garreta, Katz, Thaler, Tiwari, Zajac. On the concrete security of non-interactive FRI (RBR soundness; Fiat-Shamir loss). ePrint 2024/1161 -- not applicable here: the protocol is interactive.",
    "ZCF23": "Zeilberger, Chen, Fisch. BaseFold: efficient field-agnostic polynomial commitment schemes from foldable codes. ePrint 2023/1705.",
    "Hab24": "Haboeck. Basefold in the list decoding regime. ePrint 2024/1571.",
    "ACFY24a": "Arnon, Chiesa, Fenzi, Yogev. STIR: Reed-Solomon proximity testing with fewer queries. CRYPTO 2024 / ePrint 2024/390.",
    "ACFY24b": "Arnon, Chiesa, Fenzi, Yogev. WHIR: Reed-Solomon proximity testing with super-fast verification. ePrint 2024/1586 (Conjecture 4.12: mutual correlated agreement up to capacity).",
    "BCFRRZ24": "Brehm, Chen, Fisch, Resch, Rothblum, Zeilberger. Blaze: fast SNARKs from interleaved RAA codes. ePrint 2024/1609.",
    "DP23": "Diamond, Posen. Succinct arguments over towers of binary fields (Binius). ePrint 2023/1784.",
    "Winterfell": "facebook/winterfell PR #151 (proven security estimate: m optimised per parameter set) -- the m-optimisation used here.",
}

ASSUMPTIONS = [
    "Objects come from accounting.first_configuration at each target (extension degree, then t): A commits inputs+hints+query wires (commit_query_wires=True), B its full trace in helper-column lookup mode; K=1536, B=4096, hash adversary 2^60 queries.",
    "Interactive protocol: no Fiat-Shamir loss, no grinding / proof-of-work bits; every FRI-family challenge is a live verifier message.",
    "FRI-family layout: the object is a cols x 2^20 matrix committed as one Merkle tree with row leaves; one FRI on the eq/random combination of the columns (the 'batch all columns into one FRI' cost); final polynomial degree 1; arity-2 folding for FRI/BaseFold, 2^4 for STIR/WHIR.",
    "Number of correlated-agreement applications in the union bound = one per batching step + one per folding round (conservative reading of BCIKS20 / ethSTARK); STIR/WHIR: two per round, target shared equally over rounds.",
    "proven_ldr uses only the dominant ethSTARK pre-query term (m+1/2)^7 n^2/(2 rho^{3/2}|F|) with m optimised in [3, 200); when GL^3 (192 bits) cannot host it at n = 2^23, the PCS challenge field is raised to GL^4 (marked in params) -- the candidate's own challenge field is unchanged.",
    "conjectured = ethSTARK Conjecture 1 (1/|F| per round, rho^s queries); WHIR additionally needs its mutual-correlated-agreement conjecture; neither is a theorem.",
    "Candidate B off Ligero: its linear/quadratic row tests are replaced by a Spartan-style sumcheck adapter over ceil(log2 N_B) = 30 variables (two parallel degree-3 sumchecks, 8 F_c mults per element, 2*3*30/|F| error); this adds 31 to B's sequential depth and is included in field_ops / proof_bytes.",
    "Field-operation unit: Goldilocks multiplication-equivalents; F_{p^d} mult = EXT_MULT_COST[d]; NTT butterfly = 1; a hash call absorbs 64 B; digest 32 B.",
    "Brakedown: rate 0.65 / relative distance 0.07 (ark-poly-commit 128-bit defaults), 20 mult-equivalents per message symbol to encode (estimate), rows chosen to balance combined row vs opened columns.",
    "Blaze/Binius: commitment order of magnitude only (Goldilocks elements as 64 bits, RAA rate 1/4, distance 1/4); a field change for the whole checker, not a PCS swap; soundness not computed.",
    "est.prover_seconds_gpu is indicative only: hash 10 GB/s + field ops 1e11/s + HBM 2 TB/s, summed; no measurement stands behind it.",
    "ZK columns state the masking mechanism and its element cost; none of the FRI-family ZK variants has been red-team reviewed for this protocol.",
]

CRATES = [
    {"name": "Plonky3 (p3-fri, p3-merkle-tree, p3-goldilocks)", "kind": "FRI, Goldilocks/BabyBear/M31, Poseidon2/Keccak", "license": "MIT OR Apache-2.0", "activity": "very active (Polygon Zero / community, 2025-26)", "gpu": "SP1/Succinct CUDA prover (closed), community CUDA forks"},
    {"name": "winterfell (fri, crypto)", "kind": "FRI over f64 (Goldilocks) / f128", "license": "MIT", "activity": "maintained (Facebook / Polygon Miden)", "gpu": "miden-gpu (Metal/CUDA, RPO/RPX hashes)"},
    {"name": "WorldcoinFoundation/whir", "kind": "WHIR multilinear PCS, Goldilocks/BabyBear/BN254 fields via arkworks", "license": "MIT OR Apache-2.0", "activity": "active 2025", "gpu": "none"},
    {"name": "WizardOfMenlo/stir", "kind": "STIR reference", "license": "MIT OR Apache-2.0", "activity": "research code, low activity", "gpu": "none"},
    {"name": "arkworks ark-poly-commit linear_codes (Brakedown, Ligero)", "kind": "linear-code multilinear PCS", "license": "MIT OR Apache-2.0", "activity": "maintained", "gpu": "none"},
    {"name": "google/longfellow-zk (libzk)", "kind": "Ligero + sumcheck/GKR, C++, the design Candidate A follows", "license": "Apache-2.0", "activity": "active (IETF draft-google-cfrg-libzk)", "gpu": "none"},
    {"name": "hadasz/basefold, scroll-tech/ceno (mpcs)", "kind": "BaseFold multilinear PCS", "license": "MIT / Apache-2.0", "activity": "research / active (Ceno)", "gpu": "none"},
    {"name": "IrreducibleOSS/binius", "kind": "binary-tower fields, Ligero/FRI-Binius PCS", "license": "Apache-2.0", "activity": "active", "gpu": "in-house, not open"},
    {"name": "ingonyama-zk/icicle", "kind": "CUDA NTT, MSM, Poseidon/Poseidon2, Keccak, Merkle; Goldilocks/BabyBear/M31 fields; Rust/C++/Go bindings", "license": "Apache-2.0 (v3+; earlier dual)", "activity": "very active", "gpu": "CUDA (Metal in progress)"},
    {"name": "supranational/sppark", "kind": "CUDA NTT/MSM/Merkle primitives (used by Filecoin, zkSync era)", "license": "Apache-2.0", "activity": "maintained", "gpu": "CUDA"},
    {"name": "0xPolygonZero/plonky2 cuda forks (e.g. plonky2-gpu), zkSync boojum-cuda", "kind": "Goldilocks NTT + Poseidon Merkle on GPU", "license": "MIT/Apache-2.0", "activity": "varies; boojum-cuda active (Matter Labs)", "gpu": "CUDA"},
]


def table(target_log2: int = -128, B: int = 4096, rows_log2: int = 20, objects: dict[str, Committed] | None = None) -> list[PcsRow]:
    """Every PCS x object at one operating point.  FRI rows for A are BaseFold rows (A's openings are multilinear);
    plain FRI is given for both objects as the univariate building block."""
    objs = objects or committed_objects(target_log2, B)
    rows: list[PcsRow] = []
    for obj in objs.values():
        rows.append(ligero_row(obj))
        rows.append(brakedown_row(obj, target_log2))
        for rate in RATES:
            for regime in REGIMES:
                for fn, label in ((fri_row, "FRI"), (basefold_row, "BaseFold(RS)"), (stir_row, "STIR"), (whir_row, "WHIR")):
                    rows.append(_first_extension(fn, label, obj, target_log2, rate, regime, rows_log2))
        rows.append(blaze_row(obj, target_log2))
    return rows


def _first_extension(fn, label: str, obj: Committed, target_log2: int, rate: Fraction, regime: str, rows_log2: int,
                     max_ext: int = 5) -> PcsRow:
    """The row at the object's challenge field, or at the first larger extension whose field terms admit the regime
    (the proven list-decoding bound has an ``n^2 / |F|`` term: ``GL^3`` is too small for it at ``n = 2^23``);
    an ``UNREACHABLE`` placeholder row when none does."""
    errors = []
    for ext in range(obj.ext, max_ext + 1):
        try:
            row = fn(obj, target_log2, rate, regime, rows_log2, ext=ext)
        except AccountingError as e:
            errors.append(f"GL^{ext}: {e}")
            continue
        if ext != obj.ext:
            row.values["params"] += f" (challenge field raised from GL^{obj.ext})"
            row.values["ext_raised"] = ext
        return row
    return PcsRow({"pcs": f"{label} rate {rate}", "object": obj.name, "regime": regime,
                   "params": "UNREACHABLE: " + "; ".join(errors), "rounds.sequential_depth": None,
                   "total.sequential_depth": None, "soundness.log2": None})


def killing_table(rows: Sequence[PcsRow]) -> list[dict]:
    """Per PCS configuration: A and B side by side (hash bytes, field ops, indicative GPU seconds, total depth) and
    the B/A ratios, plus B under that PCS against A under the current Ligero -- the question whether any PCS choice
    removes B's committed-size disadvantage."""
    by = {}
    for r in rows:
        if r.values.get("census.hash_bytes") is None:
            continue
        by.setdefault((r["pcs"], r["regime"]), {})[r["object"][0]] = r
    a_lig = next(r for r in rows if r["pcs"].startswith("Ligero") and r["object"].startswith("A"))
    out = []
    for (name, regime), d in by.items():
        if "A" not in d or "B" not in d:
            continue
        a, b = d["A"], d["B"]
        out.append({
            "pcs": name, "regime": regime,
            "A.hash_bytes": a["census.hash_bytes"], "B.hash_bytes": b["census.hash_bytes"],
            "B/A.hash_bytes": round(b["census.hash_bytes"] / a["census.hash_bytes"], 2),
            "A.field_ops": a["census.field_ops"], "B.field_ops": b["census.field_ops"],
            "B/A.field_ops": round(b["census.field_ops"] / a["census.field_ops"], 2),
            "A.est_gpu_s": round(a["est.prover_seconds_gpu"], 2), "B.est_gpu_s": round(b["est.prover_seconds_gpu"], 2),
            "B.hash_bytes / A.Ligero.hash_bytes": round(b["census.hash_bytes"] / a_lig["census.hash_bytes"], 2),
            "B.est_gpu_s / A.Ligero.est_gpu_s": round(b["est.prover_seconds_gpu"] / a_lig["est.prover_seconds_gpu"], 2),
            "A.total_depth": a["total.sequential_depth"], "B.total_depth": b["total.sequential_depth"],
            "A.proof_bytes": a["proof_bytes"], "B.proof_bytes": b["proof_bytes"],
        })
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.1f}" if abs(v) < 1e4 else f"{v:,.3g}"
    if isinstance(v, int):
        return f"{v:,}" if v < 10**6 else f"{v:.3g}"
    return str(v)


def render(rows: Sequence[PcsRow], keys: Sequence[str] = ("pcs", "object", "regime", "census.bytes_encoded", "census.hash_bytes",
                                                          "census.field_ops", "proof_bytes", "verifier_hashes", "queries",
                                                          "rounds.sequential_depth", "total.sequential_depth", "soundness.log2",
                                                          "est.prover_seconds_gpu")) -> str:
    head = " | ".join(keys)
    out = [head, "-" * len(head)]
    for r in rows:
        out.append(" | ".join(_fmt(r.values.get(k)) for k in keys))
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    json_path = None
    if "--json" in args:
        json_path = args[args.index("--json") + 1]
    report = {}
    for target in (-128, -80):
        objs = committed_objects(target)
        rows = table(target, objects=objs)
        print(f"\n== 2^{target}  (A: {objs['A'].elements:,} elements, F_c=GL^{objs['A'].ext}; "
              f"B: {objs['B'].elements:,} elements, F_c=GL^{objs['B'].ext}) ==")
        print(render(rows))
        kt = killing_table(rows)
        print("\n-- B vs A per PCS (hash bytes, indicative GPU s, total depth) --")
        for k in kt:
            print(f"{k['pcs']:44} {k['regime'][:22]:22} B/A hash {k['B/A.hash_bytes']:5.2f}  B/A ops {k['B/A.field_ops']:5.2f}  "
                  f"B vs A-Ligero hash {k['B.hash_bytes / A.Ligero.hash_bytes']:5.2f}  gpu {k['B.est_gpu_s / A.Ligero.est_gpu_s']:5.2f}  "
                  f"depth A {k['A.total_depth']} B {k['B.total_depth']}")
        report[str(target)] = {"objects": {k: {"elements": o.elements, "ext": o.ext, "ligero_t": o.ligero.t,
                                                "other_depth": o.other_depth} for k, o in objs.items()},
                               "rows": [r.as_dict() for r in rows], "killing_question": kt}
    if json_path:
        with open(json_path, "w") as f:
            json.dump({"generator": "verity_numerical.explore.pcs (every number is a cost-model ESTIMATE)",
                       "columns": COLUMNS, "assumptions": ASSUMPTIONS, "citations": CITATIONS, "crates": CRATES,
                       "tables": report}, f, indent=1, default=str)


if __name__ == "__main__":       # pragma: no cover
    main()
