"""The applications' readings of the integrity profile: each is the profile's weighted bound under its own weights, equal
to the expression it replaces (core's `Stratified.harm_bound` over the law, `worst_case` with a utility) on the same
fixtures."""

from __future__ import annotations

from fractions import Fraction

from verity.ir.partition import Level
from verity.proofs.profile import Bernoulli, IntegrityProfile, Stratified, Stratum, Subset
from verity_one_stage.consumers import compute_bound, exfiltration_bound

#: A4 P6's layer 0 (one-stage-e2e): six templates, 6,771,765 units, in canonical order.
A4_SIZES = {"rms_triton": 287, "gemm2048": 6_171_648, "rope": 11_480, "rms_fused": 287, "silu_mul": 287,
            "gemm8192": 587_776}
#: Output bits per unit (16-bit words).
A4_BITS = {"rms_triton": 2048 * 16, "gemm2048": 16, "rope": 64 * 16, "rms_fused": 2 * 2048 * 16, "silu_mul": 8192 * 16,
           "gemm8192": 16}
DELTA = 2.0 ** -20
FLOOR = {"rms_triton": 1, "gemm2048": 905, "rope": 2, "rms_fused": 1, "silu_mul": 1, "gemm8192": 86}
BY_BITS = {"rms_triton": 10, "gemm2048": 104, "rope": 12, "rms_fused": 20, "silu_mul": 39, "gemm8192": 10}


def a4_law(ks: dict) -> Stratified:
    return Stratified(tuple(Stratum(name, size, ks[name]) for name, size in A4_SIZES.items()))


def one_level(law) -> IntegrityProfile:
    n = law.population if isinstance(law, Stratified) else sum(A4_SIZES.values())
    return IntegrityProfile((Level("unit"),), (n,), {"unit": law}, DELTA)


def mib(bits: float) -> float:
    return bits / 8 / 2 ** 20


def test_exfiltration_is_the_output_bit_weighted_bound():
    """A4's laws: the output bits in wrong units reach 9.40 MiB under the uniform subset:1024, 8.81 MiB with one unit per
    template and 1.50 MiB drawing in proportion to output bits, each exactly core's `harm_bound` over the law."""
    segments = [(A4_BITS[name], size) for name, size in A4_SIZES.items()]
    uniform = Stratified.uniform(sum(A4_SIZES.values()), 1024)
    got = exfiltration_bound(one_level(Subset(1024)), {"unit": segments})
    assert got == uniform.harm_bound({"unit": segments}, DELTA) and 9.39 < mib(got) < 9.41
    for ks, lo, hi in ((FLOOR, 8.80, 8.82), (BY_BITS, 1.49, 1.51)):
        got = exfiltration_bound(one_level(a4_law(ks)), A4_BITS)
        assert got == a4_law(ks).harm_bound(A4_BITS, DELTA) and lo < mib(got) < hi


def test_compute_is_the_work_weighted_bound():
    """Sampled proofs' TOY profile, over replay units of at most 4 VUs: 64 VUs of skipped work, exactly `worst_case` with
    utility 4m; under a stratified law, the work-weighted `harm_bound`."""
    toy = IntegrityProfile(("replay",), (512,), {"replay": Bernoulli(Fraction(1, 4))}, 0.01)
    assert compute_bound(toy, 4) == toy.worst_case(lambda m: 4 * m) and round(compute_bound(toy, 4), 1) == 64.0
    two = IntegrityProfile(("replay", "check"), (512, 4), {"replay": Bernoulli(Fraction(1, 2)), "check": Subset(2)}, 0.01)
    assert compute_bound(two, 1) == two.worst_case(lambda m: m)
    work = {"gemm2048": 2048, "gemm8192": 8192, "rope": 128}
    assert compute_bound(one_level(a4_law(FLOOR)), work) == a4_law(FLOOR).harm_bound(work, DELTA)


def test_an_application_reads_nothing_the_profile_does_not_bound():
    """Weighting every unit 1 reads the profile's own count (from above, within one unit), for every law."""
    for p in (one_level(Subset(1024)), one_level(a4_law(FLOOR)), one_level(a4_law(BY_BITS))):
        assert p.bound() <= compute_bound(p, 1) == exfiltration_bound(p, 1) < p.bound() + 1 + 1e-6 * p.bound()
