"""The slowdown a bound gives: the malicious developer's GPU-hours over the reference training algorithm's at its MFU,
`MFU₀`, at least 1 (Lean's `hardwareMultiple`).

`multiple_bound(MFU₀, s, x)` is `max(1, 1/(s + a x))`, with the unregistered leaves `s` training like the reference and
the registered share `x` converted into the reference's work at `a`: 1 under reference efficiency (`GLOSSARY.md`'s task
assumption: no run uses the hardware more efficiently than the reference, so free compute counts at `MFU₀`), and
`1/MFU₀` without it, free compute at peak, which is Lean's `remoteSlowdown` (`Slowdown.Guarantees.SlowdownRemote` states it
for the remote composition).
"""

from __future__ import annotations

import math


def multiple_bound(mfu: float, s: float, x: float, reference_efficiency: bool = True) -> float:
    """`max(1, 1/(s + a x))`: the GPU-hours a developer needs over an unconstrained one's at `MFU₀ = mfu`
    (`hardwareMultiple`), when the unregistered share `s` trains like it and the other terms `x`, shares of capacity,
    convert into its work at `a`. Under reference efficiency (`GLOSSARY.md`: no run uses the hardware more efficiently
    than the reference) free compute runs at `MFU₀` too, so `a = 1`; without it `x` counts at peak, `a = 1/MFU₀`, Lean's
    `Slowdown.Slowdown.remoteSlowdown mfu s x` (`Slowdown.Guarantees.SlowdownRemote`, `SlowdownRemoteMonitoringPeriod`). Below
    `x = (1 − s)/a` it is 1: those terms alone outrun the unconstrained developer."""
    rate = s + (x if reference_efficiency else x / mfu)
    return max(1.0, 1 / rate) if rate > 0 else math.inf
