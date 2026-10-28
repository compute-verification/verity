"""The rates corun.py reports, from the kernels' stamps and logs: pure Python, so the CPU tests hold them.

A stamp is one CTA's (t0, t1, c0, c1, sm, n, work): %globaltimer (ns) and clock64 at its start and end, its %smid, the
tiles (GEMM) or messages (hash) it finished and, for the hash, its compressions. The SM clock is clock64 over
%globaltimer. The peak is 4,096 s8 MACs an SM-clock (wgmma m64nNk32 s8 on the H200); θ is MACs a second over the H200's
989.4 T MAC/s (4,096 x 132 SMs x 1,830 MHz, the clock wgmma runs at). The bar is the public best SHA-512's 4.355 G
compressions a second at 1,830 MHz on 132 SMs: 0.01803 compressions an SM-clock.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

PEAK_MACS_PER_CLOCK = 4096
THETA_MACS_PER_S = 989.4e12
BAR_G_PER_S = 4.355
BAR_MHZ = 1830
SMS = 132
BAR_PER_CLOCK = BAR_G_PER_S * 1e9 / (SMS * BAR_MHZ * 1e6)
TARGET_THETA = 0.985
TARGET_BAR = 0.414


@dataclass(frozen=True)
class Stamp:
    t0: int
    t1: int
    c0: int
    c1: int
    sm: int
    n: int
    work: int = 0

    @property
    def ns(self) -> int:
        return self.t1 - self.t0

    @property
    def cycles(self) -> int:
        return self.c1 - self.c0


def window(stamps: list[Stamp]) -> tuple[int, int]:
    """The span every CTA ran in: the first start to the last end."""
    return min(s.t0 for s in stamps), max(s.t1 for s in stamps)


def mhz(stamps: list[Stamp]) -> float:
    """The SM clock over the CTAs' own spans: their cycles over their nanoseconds."""
    return sum(s.cycles for s in stamps) / sum(s.ns for s in stamps) * 1e3


def placement(stamps: list[Stamp], sms: int = SMS) -> tuple[int, int]:
    """The fewest and the most CTAs any SM held (an SM no CTA ran on counts 0)."""
    c = Counter(s.sm for s in stamps)
    per = [c.get(i, 0) for i in range(sms)]
    return min(per), max(per)


def gemm_rate(stamps: list[Stamp], macs_per_tile: int) -> dict:
    """The GEMM over its whole window: MACs an SM-clock against the peak (each CTA's MACs over its own cycles, summed over
    the SMs), θ (MACs over the window's seconds), the clock."""
    macs = sum(s.n for s in stamps) * macs_per_tile
    w0, w1 = window(stamps)
    per_clock = sum(s.n * macs_per_tile / s.cycles for s in stamps) / len(stamps)
    return {"macs": macs, "s": (w1 - w0) * 1e-9, "mhz": mhz(stamps), "macs_per_clock": per_clock,
            "peak_share": per_clock / PEAK_MACS_PER_CLOCK, "theta": macs / ((w1 - w0) * 1e-9) / THETA_MACS_PER_S}


def gemm_rate_in(logs: list[list[int]], log_every: int, macs_per_tile: int, w0: int, w1: int, mhz_in: float) -> dict:
    """The GEMM inside [w0, w1] (the hash's window), from each CTA's log of %globaltimer every `log_every` tiles: for each
    CTA the tiles between its first and last stamp inside the window over their time, summed; an SM-clock at `mhz_in`."""
    rate = 0.0
    for log in logs:
        inside = [t for t in log if w0 <= t <= w1]
        if len(inside) >= 2:
            rate += (len(inside) - 1) * log_every * macs_per_tile / ((inside[-1] - inside[0]) * 1e-9)
    per_clock = rate / (len(logs) * mhz_in * 1e6)
    return {"macs_per_s": rate, "macs_per_clock": per_clock, "peak_share": per_clock / PEAK_MACS_PER_CLOCK,
            "theta": rate / THETA_MACS_PER_S}


def hash_rate(stamps: list[Stamp], sms: int = SMS) -> dict:
    """The hash over its window: compressions an SM-clock (each SM's compressions over its CTAs' mean cycles, summed over
    the SMs, over the SMs) against the bar, and compressions a second."""
    by_sm: dict[int, list[Stamp]] = {}
    for s in stamps:
        by_sm.setdefault(s.sm, []).append(s)
    per_clock = sum(sum(s.work for s in v) / (sum(s.cycles for s in v) / len(v)) for v in by_sm.values()) / sms
    w0, w1 = window(stamps)
    work = sum(s.work for s in stamps)
    return {"compressions": work, "s": (w1 - w0) * 1e-9, "mhz": mhz(stamps), "per_clock": per_clock,
            "bar": per_clock / BAR_PER_CLOCK, "g_per_s": work / ((w1 - w0) * 1e-9) * 1e-9}


def covers(outer: list[Stamp], inner: list[Stamp], slack_ns: int = 1_000_000) -> bool:
    """Whether outer's window holds inner's, inner's end allowed `slack_ns` past outer's (the hash stops on a flag the
    GEMM's last CTA sets, and finishes the message it is on)."""
    (a0, a1), (b0, b1) = window(outer), window(inner)
    return a0 <= b0 and b1 <= a1 + slack_ns


def gap(theta_share: float, bar: float) -> dict:
    """The gap to the target: the peak share short of 0.985 and the hash short of 0.414 of the bar (negative: past it)."""
    return {"peak_short": TARGET_THETA - theta_share, "bar_short": TARGET_BAR - bar,
            "met": theta_share >= TARGET_THETA and bar >= TARGET_BAR}


def pj_per_mac(joules: float, macs: int) -> float:
    return joules / macs * 1e12
