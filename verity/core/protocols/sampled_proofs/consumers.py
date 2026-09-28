"""What applications read off the audit's integrity profile.

The profile bounds wrong units and weights nothing.  An application weights a wrong unit by what it costs the
application and gets its own bound, at the profile's delta, from the profile's generic weighted bound
(`IntegrityProfile.harm_bound`).  Each reader here is only its weighting, and takes any `IntegrityProfile`, whichever
protocol produced it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import floor, inf, log2, nextafter

from verity.core.protocols.profile import MARGIN, IntegrityProfile


def location_bits(profile: IntegrityProfile) -> float:
    """Which units are wrong is the prover's choice too.  Except with probability delta at most K = `profile.bound()` of
    the profile's n checked units are wrong, so the wrong set is one of at most sum_{j <= K} C(n, j) sets, which is at
    most 2^(n H2(K / n)) for K <= n / 2 and 2^n past it.  This returns that exponent, rounded up: the location term
    (`card_admissible_le` in the soundness package's Lean)."""
    n = profile.count(profile.levels[-1].name)
    k = floor(profile.bound())
    if k <= 0 or n == 0:
        return 0.0
    if 2 * k >= n:
        return float(n)
    x = k / n
    bits = n * (-x * log2(x) - (1 - x) * log2(1 - x))
    return min(float(n), nextafter(bits * (1 + MARGIN), inf))


def exfiltration_bound(profile: IntegrityProfile,
                       output_bits: float | Mapping[str, float | Sequence[tuple[float, int]]]) -> float:
    """Exfiltration security, in bits: except with probability delta, what an accepted transcript delivers is one of at
    most 2^b values, so a receiver recovers a uniform l-bit secret with probability at most 2^(b - l), plus delta and
    the proof system's terms.  b is the output bits held by wrong units, which the prover chose (`harm_bound` weighted by
    `output_bits`), plus which units are wrong (`location_bits`).  The soundness package's Lean states this bound as
    `audit_exfiltration`.  `output_bits` is a unit's output bits: one number for every unit, or per stratum a number or
    (bits, units) pairs."""
    return profile.harm_bound(output_bits) + location_bits(profile)


def compute_bound(profile: IntegrityProfile, work: float | Mapping[str, float | Sequence[tuple[float, int]]]) -> float:
    """Compute security: except with probability delta, at most this much work lies in wrong units, the work the prover
    could have skipped.  `work` is a unit's work, given as `exfiltration_bound`'s `output_bits` is.  Under sampled
    proofs' profile, over replay units, a wrong unit's work is at most its class's largest RU: `n_v` verification units."""
    return profile.harm_bound(work)


__all__ = ["compute_bound", "exfiltration_bound", "location_bits"]
