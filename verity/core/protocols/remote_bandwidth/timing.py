"""The honest schedule of a remote session, the passes it certifies, and the verifier's bandwidth for openings.

`honest` plays the honest prover against the verifier's messages, in integer nanoseconds: `σ_0` leaves the verifier at
0 and each boundary message at `(t + 1)·T`, each reaches the prover `down_ns` later, and each root reaches the verifier
`up_ns` after its pass ends plus the chaining gap.  Passes run back to back, `pass_ns + gap_ns` apart (the gap is the
root's reduction, the next seed and its broadcast to every GPU); the pass in flight when a boundary message arrives is
its window's last.  The tests replay the schedule against `audit.Session` (`tests/test_remote_bandwidth.py`).

`python -m verity.core.protocols.remote_bandwidth.timing` prints the grid of the report: RTT 1–50 ms, T 0.1–10 s and
passes of 20–150 ms, for isolated windows (a session of one; the next starts at the deadline, since two unchained
windows open at once would share reads) and for sessions of n windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .codec import BLOCK_BYTES
from .receipts import DIGEST_BYTES, PER_LEAF

MS = 1_000_000
S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class Schedule:
    """The honest prover's session: each window's passes, every root's arrival at the verifier `(t, i, ns)`, each
    boundary's departure `b_t`, and the deadlines `D_t = b_t + RTT allowance + tail`."""

    passes: tuple[int, ...]
    arrivals: tuple[tuple[int, int, int], ...]
    boundaries: tuple[int, ...]
    deadlines: tuple[int, ...]
    pass_ns: int

    @property
    def span_ns(self) -> int:
        return self.deadlines[-1]

    @property
    def utilization(self) -> float:
        """The share of the session's span its certified passes fill at the node's capacity: `1 − μ_timing`."""
        return sum(self.passes) * self.pass_ns / self.span_ns


def honest(window_ns: int, windows: int, pass_ns: int, *, gap_ns: int = 0, down_ns: int, up_ns: int,
           rtt_allowance_ns: int | None = None, tail_ns: int | None = None) -> Schedule:
    """The honest prover's schedule.  The allowances default to the round trip and one pass with its gap, the least that
    never makes the honest prover late."""
    rtt_allowance_ns = down_ns + up_ns if rtt_allowance_ns is None else rtt_allowance_ns
    tail_ns = pass_ns + gap_ns if tail_ns is None else tail_ns
    boundaries = tuple((t + 1) * window_ns for t in range(windows))
    deadlines = tuple(b + rtt_allowance_ns + tail_ns for b in boundaries)
    passes, arrivals = [0] * windows, []
    t, start = 0, down_ns
    while t < windows:
        if start >= boundaries[t] + down_ns:
            t += 1
            continue
        end = start + pass_ns
        arrival = end + gap_ns + up_ns
        if arrival > deadlines[t]:
            raise ValueError(f"window {t}'s pass {passes[t]} arrives at {arrival} ns, after the deadline {deadlines[t]}")
        arrivals.append((t, passes[t], arrival))
        passes[t] += 1
        start = end + gap_ns
    return Schedule(tuple(passes), tuple(arrivals), boundaries, deadlines, pass_ns)


def unchained(window_ns: int, pass_ns: int, rtt_ns: int) -> float:
    """Utilization when a window's passes share one seed: one read of each block serves every pass's receipt, so a
    window forces one pass in `T + RTT`."""
    return pass_ns / (window_ns + rtt_ns)


def per_pass_challenge(pass_ns: int, rtt_ns: int) -> float:
    """Utilization when the verifier sends every pass's seed itself: a round trip a pass."""
    return pass_ns / (pass_ns + rtt_ns)


def opening_bytes(blocks: int, tile_bytes: int = 0) -> int:
    """One opening on the wire (`audit.Opening.size`): the block, its receipt leaf, both paths, the indices, and PoUW's
    tile evidence, which PoUW's own sampled check already pays for when the draws coincide."""
    receipt_depth = math.ceil(math.log2(max(1, math.ceil(blocks / PER_LEAF))))
    payload_depth = math.ceil(math.log2(max(1, blocks)))
    return 4 + 8 + BLOCK_BYTES + PER_LEAF * DIGEST_BYTES + DIGEST_BYTES * (receipt_depth + payload_depth) + tile_bytes + 12


def samples_for(epsilon: float, delta: float) -> int:
    """The fewest openings with `ε_s ≤ epsilon` at `delta`: `C(n − m, s)/C(n, s) ≤ (1 − m/n)^s`, so the bound on wrong
    receipts is at most `n·(1 − δ^(1/s)) ≤ n·ln(1/δ)/s`."""
    return math.ceil(math.log(1 / delta) / epsilon)


def wrong_bound(units: int, samples: int, delta: float) -> int:
    """An upper bound on `IntegrityProfile.bound` for `Subset(samples)` of `units`: `⌈units·(1 − δ^(1/s))⌉`."""
    return math.ceil(-units * math.expm1(math.log(delta) / samples))


def _grid() -> None:
    rtts = (1, 5, 10, 20, 50)
    windows_s = (0.1, 0.3, 1, 3, 10)
    passes_ms = (20, 50, 100, 150)
    gap_us = 100
    print(f"# chaining gap {gap_us} µs, RTT allowance = RTT, tail = one pass + gap; one-way delays RTT/2")
    print("| pass ms | T s | RTT ms | P/window | isolated | session n=10 | n=100 | unchained | per-pass σ |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in passes_ms:
        for w in windows_s:
            for r in rtts:
                if p * MS > w * S:
                    continue
                kw = dict(pass_ns=p * MS, gap_ns=gap_us * 1000, down_ns=r * MS // 2, up_ns=r * MS - r * MS // 2)
                iso = honest(int(w * S), 1, **kw)
                ten = honest(int(w * S), 10, **kw)
                hundred = honest(int(w * S), 100, **kw)
                print(f"| {p} | {w} | {r} | {iso.passes[0]} | {1 - iso.utilization:.2%} | {1 - ten.utilization:.2%} "
                      f"| {1 - hundred.utilization:.2%} | {1 - unchained(int(w * S), p * MS, r * MS):.1%} "
                      f"| {1 - per_pass_challenge(p * MS, r * MS):.1%} |")
    print()
    hidden_us = 146
    print(f"# the chaining gap: sessions of 100 windows of 1 s at RTT 20 ms; prefetching the next pass's first blocks "
          f"into on-chip storage hides {hidden_us} µs of it (S_chip / R = 263 MB / 1.8 TB/s, which μ_chip already "
          "charges)")
    print("| pass ms | gap µs | μ_timing | μ_timing with prefetch |")
    print("| --- | --- | --- | --- |")
    for p in passes_ms:
        for g in (0, 25, 100, 500):
            kw = dict(down_ns=10 * MS, up_ns=10 * MS)
            raw = honest(S, 100, p * MS, gap_ns=g * 1000, **kw)
            hid = honest(S, 100, p * MS, gap_ns=max(0, g - hidden_us) * 1000, **kw)
            print(f"| {p} | {g} | {1 - raw.utilization:.2%} | {1 - hid.utilization:.2%} |")
    print()
    print("# openings at ε_s = 0.1%, δ = 2^-40: bytes a window, and Gb/s per node")
    s = samples_for(1e-3, 2.0 ** -40)
    for name, blocks in (("one GPU, 94 GB", 94 * 10 ** 9 // BLOCK_BYTES), ("8 GPUs, 752 GB", 8 * 94 * 10 ** 9 // BLOCK_BYTES),
                         ("8 GPUs + 1.5 TB host", (8 * 94 + 1500) * 10 ** 9 // BLOCK_BYTES)):
        b = opening_bytes(blocks)
        print(f"{name}: N = {blocks}, s = {s}, {b} B an opening, {s * b / 1e6:.0f} MB a window; "
              + ", ".join(f"T = {w} s: {s * b * 8 / w / 1e9:.2f} Gb/s" for w in windows_s))


if __name__ == "__main__":
    _grid()
