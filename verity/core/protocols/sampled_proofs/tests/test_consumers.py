"""The applications' readings of the integrity profile: each is the profile's weighted bound under its own weights, equal
to the expression it replaces (core's `Stratified.harm_bound` over the sampling design, `worst_case` with a utility) on the same
fixtures."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from math import log2
from pathlib import Path

from verity.ml.circuits.partition import Level
from verity.core.protocols.profile import Bernoulli, IntegrityProfile, Stratified, Stratum, Subset
from verity.core.protocols.sampled_proofs.consumers import compute_bound, exfiltration_bound, location_bits

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


def a4_law(ks: dict) -> Stratified:
    return Stratified(tuple(Stratum(name, size, ks[name]) for name, size in A4_SIZES.items()))


def one_level(law) -> IntegrityProfile:
    n = law.population if isinstance(law, Stratified) else sum(A4_SIZES.values())
    return IntegrityProfile((Level("unit"),), (n,), {"unit": law}, DELTA)


def mib(bits: float) -> float:
    return bits / 8 / 2 ** 20


def test_exfiltration_is_the_output_bit_weighted_bound_plus_the_location_term():
    """A4's sampling designs: the output bits in faulty proof blocks reach 9.40 MiB under the uniform subset:1024, 8.81 MiB with one proof block per
    template and 1.50 MiB drawing in proportion to output bits, each exactly core's `harm_bound` over the sampling design.  Which proof blocks
    are wrong adds 0.08, 0.09 and 0.41 MiB (`location_bits`), so the bounds are 9.48, 8.89 and 1.91 MiB."""
    segments = [(A4_BITS[name], size) for name, size in A4_SIZES.items()]
    uniform = Stratified.uniform(sum(A4_SIZES.values()), 1024)
    p = one_level(Subset(1024))
    got = exfiltration_bound(p, {"unit": segments})
    assert got == uniform.harm_bound({"unit": segments}, DELTA) + location_bits(p) and 9.48 < mib(got) < 9.49
    for ks, lo, hi in ((FLOOR, 8.88, 8.90), (BY_BITS, 1.90, 1.92)):
        p = one_level(a4_law(ks))
        got = exfiltration_bound(p, A4_BITS)
        assert got == a4_law(ks).harm_bound(A4_BITS, DELTA) + location_bits(p) and lo < mib(got) < hi


def test_the_location_term_bounds_the_number_of_wrong_sets():
    """log2 of the sets of at most K = bound() of n proof blocks, sum_{j <= K} C(n, j), is at most `location_bits`, and within
    log2(n) + 1 bits of it; no faulty proof block means no choice, and at least half the proof blocks wrong means every subset."""
    from math import comb

    for n, k in ((512, 4), (512, 16), (4096, 38)):
        p = IntegrityProfile(("unit",), (n,), {"unit": Subset(k)}, 0.01)
        K = p.bound()
        exact = log2(sum(comb(n, j) for j in range(K + 1)))
        assert exact <= location_bits(p) <= exact + log2(n) + 1
    assert location_bits(IntegrityProfile(("unit",), (8,), {"unit": Subset(8)}, 0.01)) == 0.0
    assert location_bits(IntegrityProfile(("unit",), (8,), {"unit": Subset(1)}, 0.01)) == 8.0


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
    soundness = "Security/Proofs/"
    assert set(recorded) == {"verity/core/protocols/sampled_proofs/tests/lean/ExfiltrationVectors.lean",
                             *(soundness + f for f in ("Flock/Soundness/Audit/Influence.lean", "Flock/Soundness/Audit/Law.lean",
                                                       "Flock/Soundness/Audit/Stratified.lean",
                                                       "Flock/Soundness/Audit/StratifiedMiss.lean",
                                                       "lake-manifest.json", "lean-toolchain"))}
    stale = [p for p, h in recorded.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    assert not stale, f"changed since the vectors were generated, rerun verity/core/protocols/sampled_proofs/tests/lean/generate.sh: {stale}"


def test_compute_is_the_work_weighted_bound():
    """A TOY profile over coarse proof blocks of 4 fine ones each: 64 fine proof blocks of skipped work, exactly `worst_case` with utility 4m; over two
    levels, `worst_case` too; under a stratified sampling design, the work-weighted `harm_bound`."""
    toy = IntegrityProfile(("replay",), (512,), {"replay": Bernoulli(Fraction(1, 4))}, 0.01)
    assert compute_bound(toy, 4) == toy.worst_case(lambda m: 4 * m) and round(compute_bound(toy, 4), 1) == 64.0
    two = IntegrityProfile(("replay", "check"), (512, 4), {"replay": Bernoulli(Fraction(1, 2)), "check": Subset(2)}, 0.01)
    assert compute_bound(two, 1) == two.worst_case(lambda m: m)
    work = {"gemm2048": 2048, "gemm8192": 8192, "rope": 128}
    assert compute_bound(one_level(a4_law(FLOOR)), work) == a4_law(FLOOR).harm_bound(work, DELTA)


def test_an_application_reads_nothing_the_profile_does_not_bound():
    """Weighting every proof block 1 reads the profile's own count (from above, within one proof block), for every sampling design; exfiltration
    adds the location term to it."""
    for p in (one_level(Subset(1024)), one_level(a4_law(FLOOR)), one_level(a4_law(BY_BITS))):
        assert p.bound() <= compute_bound(p, 1) < p.bound() + 1 + 1e-6 * p.bound()
        assert exfiltration_bound(p, 1) == compute_bound(p, 1) + location_bits(p)
