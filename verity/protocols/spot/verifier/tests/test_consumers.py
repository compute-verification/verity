"""The applications' readings of the integrity profile: each is the profile's weighted bound under its own weights, equal
to the expression it replaces (core's `Stratified.harm_bound` over the sampling design, `worst_case` with a utility) on the same
fixtures."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb, log2
from pathlib import Path

import pytest

from verity.circuits.partition import Level
from verity.protocols.spot.verifier.profile import Bernoulli, IntegrityProfile, Stratified, Stratum, Subset
from verity.protocols.spot.verifier.consumers import (UNCERTIFIED_COMPUTE, compute_bound, entropy_bits,
                                                            exfiltration_bound, infiltration_bound, location_bits)
from verity.protocols.spot.verifier.service import REPLAY

#: A4 P6's layer 0 (one-stage-e2e): six templates, 6,771,765 proof blocks, in canonical order.
A4_SIZES = {"rms_triton": 287, "gemm2048": 6_171_648, "rope": 11_480, "rms_fused": 287, "silu_mul": 287,
            "gemm8192": 587_776}
#: Output bits per proof block (16-bit words).
A4_BITS = {"rms_triton": 2048 * 16, "gemm2048": 16, "rope": 64 * 16, "rms_fused": 2 * 2048 * 16, "silu_mul": 8192 * 16,
           "gemm8192": 16}
DELTA = 2.0 ** -20
FLOOR = {"rms_triton": 1, "gemm2048": 905, "rope": 2, "rms_fused": 1, "silu_mul": 1, "gemm8192": 86}
BY_BITS = {"rms_triton": 10, "gemm2048": 104, "rope": 12, "rms_fused": 20, "silu_mul": 39, "gemm8192": 10}
ROOT = Path(__file__).resolve().parents[5]
LEAN_VECTORS = Path(__file__).parent / "fixtures" / "exfiltration_lean.json"
LEAN_TWO_VECTORS = Path(__file__).parent / "fixtures" / "exfiltration_two_lean.json"


def a4_law(ks: dict) -> Stratified:
    return Stratified(tuple(Stratum(name, size, ks[name]) for name, size in A4_SIZES.items()))


def one_level(law) -> IntegrityProfile:
    n = law.population if isinstance(law, Stratified) else sum(A4_SIZES.values())
    return IntegrityProfile((Level("unit"),), (n,), {"unit": law}, DELTA)


def mib(bits: float) -> float:
    return bits / 8 / 2 ** 20


def test_infiltration_is_the_output_bit_weighted_bound_and_exfiltration_adds_the_location_term():
    """A4's sampling designs: the incorrect proof blocks' outputs, what the prover wrote in place of the circuit's values,
    total at most 9.40 MiB under the uniform subset:1024, 8.81 MiB with one proof block per template and 1.50 MiB sampling
    in proportion to output bits, each exactly core's `harm_bound` over the sampling design.  Which proof blocks are incorrect
    adds 0.08, 0.09 and 0.41 MiB (`location_bits`), so the exfiltration bounds are 9.48, 8.89 and 1.91 MiB."""
    segments = [(A4_BITS[name], size) for name, size in A4_SIZES.items()]
    p = one_level(Subset(1024))
    got = infiltration_bound(p, {"unit": segments})
    assert got == Stratified.uniform(sum(A4_SIZES.values()), 1024).harm_bound({"unit": segments}, DELTA)
    out = exfiltration_bound(p, {"unit": segments})
    assert 9.39 < mib(got) < 9.41 and out == got + location_bits(p) and 9.48 < mib(out) < 9.49
    for ks, lo, hi, out_lo, out_hi in ((FLOOR, 8.80, 8.82, 8.88, 8.90), (BY_BITS, 1.49, 1.51, 1.90, 1.92)):
        p = one_level(a4_law(ks))
        got, out = infiltration_bound(p, A4_BITS), exfiltration_bound(p, A4_BITS)
        assert got == a4_law(ks).harm_bound(A4_BITS, DELTA) and lo < mib(got) < hi
        assert out == got + location_bits(p) and out_lo < mib(out) < out_hi


def test_the_location_term_bounds_the_number_of_wrong_sets():
    """log2 of the sets of at most K = bound() of n proof blocks, sum_{j <= K} C(n, j), is at most `location_bits`, and within
    log2(n) + 1 bits of it; no incorrect proof block means no choice, and at least half the proof blocks incorrect means every subset."""
    from math import comb

    for n, k in ((512, 4), (512, 16), (4096, 38)):
        p = IntegrityProfile(("unit",), (n,), {"unit": Subset(k)}, 0.01)
        K = p.bound()
        exact = log2(sum(comb(n, j) for j in range(K + 1)))
        assert exact <= location_bits(p) <= exact + log2(n) + 1
    assert location_bits(IntegrityProfile(("unit",), (8,), {"unit": Subset(8)}, 0.01)) == 0.0
    assert location_bits(IntegrityProfile(("unit",), (8,), {"unit": Subset(1)}, 0.01)) == 8.0


@pytest.mark.parametrize("n, rate", [(512, Fraction(1, 2)), (10 ** 17, Fraction(1, 2)), (10 ** 18, Fraction(1, 2)),
                                     (10 ** 21, Fraction(1, 3)), (3348 * 10 ** 18, Fraction(1, 5))])
def test_the_location_term_bounds_the_wrong_sets_at_any_population(n, rate):
    """Past 2^53 proof blocks a handful of incorrect ones is a share of them no float tells from none: the term is
    still at least log2 of the number of sets, counted exactly, and within 1.5 bits a block of it."""
    p = IntegrityProfile(("unit",), (n,), {"unit": Bernoulli(rate)}, 0.01)
    K = p.bound()
    with localcontext() as context:
        context.prec = 50
        exact = Decimal(sum(comb(n, j) for j in range(K + 1))).ln() / Decimal(2).ln()
        assert 2 <= K <= 30 and exact <= Decimal(location_bits(p)) <= exact + Decimal("1.5") * K


@pytest.mark.parametrize("n, k", [(512, 4), (10 ** 9, 10 ** 4), (10 ** 12, 1000), (17_784 * 10 ** 12, 1000),
                                  (17_784 * 10 ** 12, 33 * 10 ** 6), (10 ** 17, 5), (10 ** 18, 10 ** 6),
                                  (10 ** 18, 10 ** 12), (3348 * 10 ** 18, 10 ** 5), (3348 * 10 ** 18, 2 * 10 ** 9),
                                  (3348 * 10 ** 18, 22 * 10 ** 14), (10 ** 21, 3)])
def test_the_entropy_holds_its_digits_at_any_population(n, k):
    """n H2(k / n) against its value in 50-digit decimals, to 2^-45 of it: far inside the margin `location_bits` raises
    it by, at populations where the float 1 - k / n is 1."""
    with localcontext() as context:
        context.prec = 50
        exact = (k * (Decimal(n) / k).ln() + (n - k) * (Decimal(n) / (n - k)).ln()) / Decimal(2).ln()
        assert abs(Decimal(entropy_bits(n, k)) - exact) <= exact * Decimal(2) ** -45


def test_the_bound_is_at_least_audit_exfiltration_evaluated_in_lean():
    """`fixtures/exfiltration_lean.json`: Lean evaluates `audit_exfiltration`'s bound (`lean/ExfiltrationVectors.lean`)
    on the uniform k-subset of n proof blocks at delta = 2^-e with b output bits per proof block: K, the least count meeting its
    `miss (K + 1) < delta` (the kernel checks miss (K + 1) < delta <= miss K, `grid_counts`), the harm bound H = b K,
    and admissible = sum_{j <= K} C(n, j), so the bound's numerator is 2^H admissible.  Core's count bound is K;
    `harm_bound` is at least H and below H + b; `location_bits` is at least log2 admissible and within log2(n + 1) of
    it, and n once K >= n / 2; so `exfiltration_bound` is at least log2 of the numerator."""
    data = json.loads(LEAN_VECTORS.read_text())
    assert data["statement"] == "FlockSoundness.Audit.Partition.audit_exfiltration" and len(data["vectors"]) == 24
    for v in data["vectors"]:
        n, K, H, b, admissible = v["n"], v["K"], v["H"], v["bits"], v["admissible"]
        p = IntegrityProfile(("unit",), (n,), {"unit": Subset(v["k"])}, 2.0 ** v["delta_log2"])
        assert p.bound() == K and H == b * K
        assert H <= p.harm_bound(b) < H + b
        loc = location_bits(p)
        if 2 * K >= n:
            assert loc == n and admissible <= 2 ** n
        else:
            assert log2(admissible) <= loc <= log2(admissible) + log2(n + 1)
        assert exfiltration_bound(p, b) >= H + log2(admissible)


def test_stratified_miss_is_the_lean_certified_greedy_fill():
    """The stratified vectors (`gridS_counts`): Lean fills each sampling design greedily in exact integers, and the kernel checks that
    tau separates each fill, in (0, 1], every factor taken at least tau and every factor left at most tau,
    cross-multiplied.  So the fill of K + 1 proof blocks attains miss (K + 1) < delta and the fill of K attains miss K >= delta
    (`stratified_miss_eq_greedy`), for every sampling design with those strata.  Core's `miss` is those fills' exact acceptance probabilities, at the
    grid's exact ties and near-ties too, its count bound is K, and its exfiltration reading bounds the numerator as for
    the uniform sampling design."""
    data = json.loads(LEAN_VECTORS.read_text())
    assert len(data["stratified"]) == 16
    for v in data["stratified"]:
        law = Stratified(tuple(Stratum(f"s{i}", n, k) for i, (n, k) in enumerate(v["strata"])))
        names = [s.name for s in law.strata]
        K, H, b, admissible, delta = v["K"], v["H"], v["bits"], v["admissible"], 2.0 ** v["delta_log2"]
        after, at = dict(zip(names, v["fill_next"])), dict(zip(names, v["fill"]))
        assert sum(after.values()) == K + 1 and sum(at.values()) == K
        assert law.miss(K + 1) == law.escape(after) < Fraction(delta) <= law.escape(at) == law.miss(K)
        n = law.population
        p = IntegrityProfile((Level("unit"),), (n,), {"unit": law}, delta)
        assert law.count_bound(delta) == p.bound() == K and H == b * K
        assert H <= p.harm_bound(b) < H + b
        loc = location_bits(p)
        if 2 * K >= n:
            assert loc == n and admissible <= 2 ** n
        else:
            assert log2(admissible) <= loc <= log2(admissible) + log2(n + 1)
        assert exfiltration_bound(p, b) >= H + log2(admissible)

def test_the_lean_vectors_were_generated_from_this_tree():
    """The vectors record the sha256 of the exporter and of the soundness package's files they read (`lean/generate.sh`):
    `audit_exfiltration` (`Audit/Influence.lean`), the uniform sampling design and `subset_miss` (`Audit/Law.lean`), the stratified
    sampling design and `stratified_miss_eq_greedy` (`Audit/Stratified.lean`, `Audit/StratifiedMiss.lean`), and the toolchain and
    Mathlib pins.  A change to any of them fails here until the vectors are regenerated."""
    recorded = json.loads(LEAN_VECTORS.read_text())["inputs"]
    soundness = "security/Proofs/"
    assert set(recorded) == {"verity/protocols/spot/verifier/tests/lean/ExfiltrationVectors.lean",
                             *(soundness + f for f in ("Flock/Soundness/Audit/Influence.lean", "Flock/Soundness/Audit/Law.lean",
                                                       "Flock/Soundness/Audit/Stratified.lean",
                                                       "Flock/Soundness/Audit/StratifiedMiss.lean",
                                                       "lake-manifest.json", "lean-toolchain"))}
    stale = [p for p, h in recorded.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    assert not stale, f"changed since the vectors were generated, rerun verity/protocols/spot/verifier/tests/lean/generate.sh: {stale}"


def test_the_two_stage_vectors_were_generated_from_this_tree():
    """The two-stage vectors record the sha256 of their exporter (`lean/ExfiltrationTwoVectors.lean`) and of the files they
    read: the two-stage design (`Claims/SamplingTwo.lean`), its count `card_influenceTwo_le_uniform`
    (`SamplingGeneric/TwoCount.lean`), its table (`SamplingGeneric/TwoTable.lean`), and the toolchain and Mathlib pins.  A
    change to any of them fails here until the vectors are regenerated."""
    recorded = json.loads(LEAN_TWO_VECTORS.read_text())["inputs"]
    soundness = "security/Proofs/"
    assert set(recorded) == {"verity/protocols/spot/verifier/tests/lean/ExfiltrationTwoVectors.lean",
                             "security/Definitions/Claims/SamplingTwo.lean",
                             *(soundness + f for f in ("Flock/SamplingGeneric/TwoCount.lean",
                                                       "Flock/SamplingGeneric/TwoTable.lean",
                                                       "lake-manifest.json", "lean-toolchain"))}
    stale = [p for p, h in recorded.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    assert not stale, f"changed since the vectors were generated, rerun verity/protocols/spot/verifier/tests/lean/generate.sh: {stale}"


def test_compute_is_the_work_weighted_bound():
    """A TOY profile over coarse proof blocks of 4 fine ones each: 64 fine proof blocks of skipped work, exactly `worst_case` with utility 4m; over two
    levels, `worst_case` too; under a stratified sampling design, the work-weighted `harm_bound`."""
    toy = IntegrityProfile(("coarse",), (512,), {"coarse": Bernoulli(Fraction(1, 4))}, 0.01)
    assert compute_bound(toy, 4) == toy.worst_case(lambda m: 4 * m) and round(compute_bound(toy, 4), 1) == 64.0
    two = IntegrityProfile(("group", "check"), (512, 4), {"group": Bernoulli(Fraction(1, 2)), "check": Subset(2)}, 0.01)
    assert compute_bound(two, 1) == two.worst_case(lambda m: m)
    work = {"gemm2048": 2048, "gemm8192": 8192, "rope": 128}
    assert compute_bound(one_level(a4_law(FLOOR)), work) == a4_law(FLOOR).harm_bound(work, DELTA)


def test_a_two_stage_profile_is_read_for_infiltration_and_exfiltration_and_refused_for_compute():
    """A two-stage design's profile, a level of replay blocks (`service.REPLAY`) above the proof blocks: infiltration reads
    it at the footprint (`worst_case` with utility b m), exfiltration adds which proof blocks are incorrect
    (`card_influenceTwo_le_uniform`), and compute refuses it, since its relation reads no interior and takes a trace whose
    interiors were never computed (`deferred_not_credited`).  A replay level that is the checked level is a one-stage
    design's, and compute reads it."""
    two = IntegrityProfile((REPLAY, "unit"), (512, 4), {REPLAY: Bernoulli(Fraction(1, 2)), "unit": Subset(2)}, 0.01)
    assert infiltration_bound(two, 16) == two.worst_case(lambda m: 16 * m)
    assert exfiltration_bound(two, 1) == two.worst_case(lambda m: m) + location_bits(two)
    with pytest.raises(ValueError, match=re.escape(UNCERTIFIED_COMPUTE)):
        compute_bound(two, 1)
    one = IntegrityProfile((REPLAY,), (512,), {REPLAY: Bernoulli(Fraction(1, 4))}, 0.01)
    assert compute_bound(one, 1) == one.worst_case(lambda m: m)


def test_the_two_stage_bound_is_at_least_the_footprint_count_evaluated_in_lean():
    """`fixtures/exfiltration_two_lean.json` (`lean/ExfiltrationTwoVectors.lean`, `gridTwo_influence`): R replay blocks of N
    proof blocks each, the first stage at rate num/den, k of each sampled replay block's proof blocks sampled, at
    delta = 2^-e with b bits per proof block.  Lean takes the largest
    product of the replay blocks' acceptance factors at each total count, in exact integers, and the kernel checks that
    every count past K escapes with probability at most delta, so the served outputs at any inputs number at most
    2^H admissible, H = b K and admissible = sum_{j <= K} C(R N, j).  Core's acceptance factor is Lean's; its count bound,
    the linear relaxation, is at least K and below K + N, and K itself when every replay block is replayed and the replay
    blocks are strata; so `exfiltration_bound` is at least log2 of the numerator."""
    data = json.loads(LEAN_TWO_VECTORS.read_text())
    assert data["statement"] == "Claims.Sampling.card_influenceTwo_le_uniform" and len(data["vectors"]) == 22
    for v in data["vectors"]:
        R, N, k, (num, den) = v["replay_blocks"], v["n_rho"], v["k"], v["rate"]
        K, H, b, admissible, delta = v["K"], v["H"], v["bits"], v["admissible"], 2.0 ** v["delta_log2"]
        p = IntegrityProfile((REPLAY, "unit"), (R, N), {REPLAY: Bernoulli(Fraction(num, den)), "unit": Subset(k)}, delta)
        assert [p.accept(m) for m in range(N + 1)] == [Fraction(a, v["accept_den"]) for a in v["accept"]]
        assert p.count("unit") == R * N and H == b * K
        assert K <= p.bound() < K + N and H <= p.harm_bound(b) < H + b * N
        if num == den:
            assert Stratified(tuple(Stratum(f"r{i}", N, k) for i in range(R))).count_bound(delta) == K
        loc = location_bits(p)
        if 2 * K >= R * N:
            assert loc == R * N and admissible <= 2 ** (R * N)
        else:
            assert log2(admissible) <= loc
        assert exfiltration_bound(p, b) >= H + log2(admissible)


def test_an_application_reads_nothing_the_profile_does_not_bound():
    """Weighting every proof block 1 reads the profile's own count (from above, within one proof block), for every sampling design;
    infiltration reads the same count, and exfiltration adds the location term to it."""
    for p in (one_level(Subset(1024)), one_level(a4_law(FLOOR)), one_level(a4_law(BY_BITS))):
        assert p.bound() <= compute_bound(p, 1) < p.bound() + 1 + 1e-6 * p.bound()
        assert infiltration_bound(p, 1) == compute_bound(p, 1)
        assert exfiltration_bound(p, 1) == compute_bound(p, 1) + location_bits(p)
