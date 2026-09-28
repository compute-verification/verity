"""What applications read off the audit's integrity profile.

The profile bounds wrong units and weights nothing.  An application weights a wrong unit by what it costs the
application and gets its own bound, at the profile's delta, from the profile's generic weighted bound
(`IntegrityProfile.harm_bound`).  Each reader here is only its weighting, and takes any `IntegrityProfile`, whichever
protocol produced it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from verity.proofs.profile import IntegrityProfile


def exfiltration_bound(profile: IntegrityProfile,
                       output_bits: float | Mapping[str, float | Sequence[tuple[float, int]]]) -> float:
    """Exfiltration security: except with probability delta, at most this many output bits of the accepted transcript lie
    in wrong units, where the prover chose them.  `output_bits` is a unit's output bits: one number for every unit, or per
    stratum a number or (bits, units) pairs.  Which units are wrong is the prover's choice too, which adds at most
    n H2(b / n) bits for b = `profile.bound()` <= n / 2 of n units."""
    return profile.harm_bound(output_bits)


def compute_bound(profile: IntegrityProfile, work: float | Mapping[str, float | Sequence[tuple[float, int]]]) -> float:
    """Compute security: except with probability delta, at most this much work lies in wrong units, the work the prover
    could have skipped.  `work` is a unit's work, given as `exfiltration_bound`'s `output_bits` is.  Under sampled
    proofs' profile, over replay units, a wrong unit's work is at most its class's largest RU: `n_v` verification units."""
    return profile.harm_bound(work)


__all__ = ["compute_bound", "exfiltration_bound"]
