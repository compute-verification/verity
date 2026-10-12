"""The honest prover's schedule on the challenger's coin stream, what the schedule lets a dishonest prover share between
passes, and the verifier's bandwidth for openings.

`honest` plays the honest prover against the coin stream in integer nanoseconds.  Coin k leaves the verifier at k·Δ
and reaches the prover `down_ns` plus a jitter later.  The prover starts pass k once the coin is there and pass k − 1's
reads are done, reads for `pass_ns`, then reduces the root in `gap_ns` (beside the next pass's reads) and sends it; the
root reaches the verifier `up_ns` plus a jitter later.  Each jitter is in [0, `jitter_ns`], a fixed pseudorandom
pattern.  The tests replay the schedule against `audit.Session` (`tests/test_remote_bandwidth.py`).

`deadline` is the least allowance A that never makes the honest prover late.  A dishonest prover's pass k may read
anywhere in [r_k + d↓_floor, D_k − d↑_floor], `A − RTT_floor` long, and coins leave Δ apart, so a read in the overlap of
two passes' windows may serve both (`audit.overlap_ns`).  `mu` is the steady-state μ_timing + μ_overlap a pass at coin
interval Δ.

`python -m verity.core.protocols.remote_bandwidth.timing` prints the grid of the report.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .audit import overlap_ns
from .codec import BLOCK_BYTES
from .receipts import DIGEST_BYTES, PER_LEAF, SALT_BYTES

MS = 1_000_000
US = 1_000
S = 1_000_000_000
JITTER_DOMAIN = b"verity/remote-bandwidth/timing-jitter/v0\0"


def deadline(pass_ns: int, *, gap_ns: int = 0, down_ns: int, up_ns: int, jitter_ns: int = 0) -> int:
    """The least A: the coin's trip down, the pass, the root's reduction and its trip up, each trip at its longest."""
    return down_ns + up_ns + 2 * jitter_ns + pass_ns + gap_ns


def _jitter(k: int, direction: bytes, jitter_ns: int) -> int:
    if not jitter_ns:
        return 0
    h = hashlib.sha512(JITTER_DOMAIN + direction + k.to_bytes(8, "big")).digest()
    return int.from_bytes(h[:8], "big") % (jitter_ns + 1)


@dataclass(frozen=True, slots=True)
class Stream:
    """The honest prover's session: coin k's release `released[k]`, its root's arrival `arrivals[k]` at the verifier,
    and the allowance `A`."""

    released: tuple[int, ...]
    arrivals: tuple[int, ...]
    deadline_ns: int
    pass_ns: int

    @property
    def deadlines(self) -> tuple[int, ...]:
        return tuple(r + self.deadline_ns for r in self.released)

    @property
    def span_ns(self) -> int:
        return self.deadlines[-1] - self.released[0]

    def mu(self, rtt_floor_ns: int = 0) -> float:
        """μ_timing + μ_overlap over the session: the span its passes don't fill, plus the time reads may be shared."""
        filled = len(self.released) * self.pass_ns
        return 1 - filled / self.span_ns + overlap_ns(self.released, self.deadlines, rtt_floor_ns) / self.span_ns


def honest(passes: int, interval_ns: int, pass_ns: int, *, gap_ns: int = 0, down_ns: int, up_ns: int, jitter_ns: int = 0,
           deadline_ns: int | None = None) -> Stream:
    """The honest prover's schedule over `passes` coins `interval_ns` apart; the allowance defaults to `deadline`'s."""
    if interval_ns < pass_ns:
        raise ValueError("coins come faster than passes end: the honest prover falls behind")
    a = deadline(pass_ns, gap_ns=gap_ns, down_ns=down_ns, up_ns=up_ns, jitter_ns=jitter_ns) if deadline_ns is None else deadline_ns
    released, arrivals, free = [], [], 0
    for k in range(passes):
        r = k * interval_ns
        start = max(r + down_ns + _jitter(k, b"down", jitter_ns), free)
        free = start + pass_ns
        arrival = free + gap_ns + up_ns + _jitter(k, b"up", jitter_ns)
        if arrival > r + a:
            raise ValueError(f"pass {k}'s root arrives at {arrival} ns, after its deadline {r + a}")
        released.append(r)
        arrivals.append(arrival)
    return Stream(tuple(released), tuple(arrivals), a, pass_ns)


def mu(interval_ns: int, pass_ns: int, *, gap_ns: int = 0, rtt_ns: int, jitter_ns: int = 0, rtt_floor_ns: int = 0) -> float:
    """The steady state a pass at the least deadline: `1 − p/Δ + max(0, A − RTT_floor − Δ)/Δ`.  With
    `U = RTT + 2J − RTT_floor`, that is `(g + U)/Δ` up to `Δ* = p + g + U` and `1 − p/Δ` beyond."""
    if interval_ns < pass_ns:
        raise ValueError("coins come faster than passes end")
    a = rtt_ns + 2 * jitter_ns + pass_ns + gap_ns
    return 1 - pass_ns / interval_ns + max(0, a - rtt_floor_ns - interval_ns) / interval_ns


def best_interval(pass_ns: int, *, gap_ns: int = 0, rtt_ns: int, jitter_ns: int = 0, rtt_floor_ns: int = 0) -> int:
    """`Δ* = A − RTT_floor`: consecutive passes' windows just don't overlap, and `mu` is `(g + U)/(p + g + U)`."""
    return max(pass_ns, rtt_ns + 2 * jitter_ns + pass_ns + gap_ns - rtt_floor_ns)


def chained(pass_ns: int, gap_ns: int) -> float:
    """The steady state if each seed also took the previous pass's root, for order alone (the coin still the only
    randomness): passes are then sequential however their windows overlap, and only the root's reduction and the
    seed's broadcast, g, go uncovered."""
    return gap_ns / (pass_ns + gap_ns)


def opening_bytes(blocks: int, tile_bytes: int = 0) -> int:
    """One opening on the wire (`audit.Opening.size`): the block, its receipt leaf and salt, both paths, the payload's
    salt, the indices, and PoUW's tile evidence, which PoUW's own sampled check already pays for when the draws
    coincide."""
    receipt_depth = math.ceil(math.log2(max(1, math.ceil(blocks / PER_LEAF))))
    payload_depth = math.ceil(math.log2(max(1, blocks)))
    return (8 + 8 + BLOCK_BYTES + PER_LEAF * DIGEST_BYTES + 2 * SALT_BYTES + DIGEST_BYTES * (receipt_depth + payload_depth)
            + tile_bytes + 12)


def samples_for(epsilon: float, delta: float) -> int:
    """The fewest openings with `ε_s ≤ epsilon` at `delta`: `C(n − m, s)/C(n, s) ≤ (1 − m/n)^s`, so the bound on wrong
    receipts is at most `n·(1 − δ^(1/s)) ≤ n·ln(1/δ)/s`."""
    return math.ceil(math.log(1 / delta) / epsilon)


def wrong_bound(units: int, samples: int, delta: float) -> int:
    """An upper bound on `IntegrityProfile.bound` for `Subset(samples)` of `units`: `⌈units·(1 − δ^(1/s))⌉`."""
    return math.ceil(-units * math.expm1(math.log(delta) / samples))


def _ns(ms: float) -> int:
    return round(ms * MS)


def _pct(x: float) -> str:
    return f"{x:.2%}" if x < 1 else "vacuous"


def _grid() -> None:
    gap = 50 * US
    jitter = _ns(0.05)
    print(f"# μ_timing + μ_overlap a pass, steady state, at the least deadline A = RTT + 2J + p + g; g = {gap // US} µs, "
          f"J = {jitter / MS} ms each way; floor = the trusted RTT floor as a fraction of RTT; U = RTT + 2J − floor")
    print("| pass ms | RTT ms | floor | U ms | A ms | Δ = p | Δ* ms | at Δ* | chained, for comparison |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in (20, 50):
        for rtt in (0.2, 1, 5, 20, 50):
            for f in (0, 0.5, 0.9):
                kw = dict(gap_ns=gap, rtt_ns=_ns(rtt), jitter_ns=jitter, rtt_floor_ns=_ns(f * rtt))
                star = best_interval(_ns(p), **kw)
                u = _ns(rtt) + 2 * jitter - _ns(f * rtt)
                print(f"| {p} | {rtt} | {f} | {u / MS:.2f} | {(_ns(rtt) + 2 * jitter + _ns(p) + gap) / MS:.2f} "
                      f"| {_pct(mu(_ns(p), _ns(p), **kw))} | {star / MS:.2f} | {_pct(mu(star, _ns(p), **kw))} "
                      f"| {chained(_ns(p), gap):.2%} |")
    print()
    print("# μ against the coin interval Δ: pass 20 ms, RTT 1 ms, J 0.05 ms, floor 0.5 RTT, g 50 µs")
    kw = dict(gap_ns=gap, rtt_ns=MS, jitter_ns=jitter, rtt_floor_ns=MS // 2)
    star = best_interval(20 * MS, **kw)
    deltas = (20 * MS, _ns(20.25), _ns(20.5), star, 21 * MS, 22 * MS, 25 * MS, 30 * MS)
    print("| " + " | ".join(f"Δ = {d / MS:g} ms" for d in deltas) + " |")
    print("| " + " | ".join("---" for _ in deltas) + " |")
    print("| " + " | ".join(f"{mu(d, 20 * MS, **kw):.2%}" for d in deltas) + " |")
    print()
    print("| jitter ms | " + " | ".join(f"J = {j}" for j in (0, 0.05, 0.2, 0.5, 1)) + " |")
    print("| --- | --- | --- | --- | --- | --- |")
    for p in (20, 50):
        row = []
        for j in (0, 0.05, 0.2, 0.5, 1):
            kj = dict(gap_ns=gap, rtt_ns=MS, jitter_ns=_ns(j), rtt_floor_ns=MS // 2)
            row.append(f"{mu(best_interval(_ns(p), **kj), _ns(p), **kj):.2%}")
        print(f"| at Δ*, pass {p} ms, RTT 1 ms, floor 0.5 | " + " | ".join(row) + " |")
    print()
    print("# sessions of W windows of T = 1 s, simulated (`honest`), at Δ*: the stream runs on across window boundaries, "
          "so a window's end costs nothing and only the session's two ends do")
    print("| pass ms | RTT ms | floor | Δ* ms | P a window | W = 1 | W = 10 | W = 100 | steady state |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p, rtt, f in ((20, 1, 0.5), (20, 0.2, 0.5), (50, 5, 0.5), (50, 1, 0.9)):
        kw = dict(gap_ns=gap, rtt_ns=_ns(rtt), jitter_ns=jitter, rtt_floor_ns=_ns(f * rtt))
        star = best_interval(_ns(p), **kw)
        per = S // star
        cells = []
        for w in (1, 10, 100):
            st = honest(w * per, star, _ns(p), gap_ns=gap, down_ns=_ns(rtt) // 2, up_ns=_ns(rtt) - _ns(rtt) // 2,
                        jitter_ns=jitter)
            cells.append(f"{st.mu(_ns(f * rtt)):.2%}")
        print(f"| {p} | {rtt} | {f} | {star / MS:.2f} | {per} | " + " | ".join(cells) + f" | {mu(star, _ns(p), **kw):.2%} |")
    print()
    print("# openings at ε_s = 0.1%, δ = 2^-40: bytes a window, and Gb/s per node")
    s = samples_for(1e-3, 2.0 ** -40)
    for name, blocks in (("one GPU, 94 GB", 94 * 10 ** 9 // BLOCK_BYTES), ("8 GPUs, 752 GB", 8 * 94 * 10 ** 9 // BLOCK_BYTES),
                         ("8 GPUs + 1.5 TB host", (8 * 94 + 1500) * 10 ** 9 // BLOCK_BYTES)):
        b = opening_bytes(blocks)
        print(f"{name}: N = {blocks}, s = {s}, {b} B an opening, {s * b / 1e6:.0f} MB a window; "
              + ", ".join(f"T = {w} s: {s * b * 8 / w / 1e9:.2f} Gb/s" for w in (0.1, 0.3, 1, 3, 10)))


if __name__ == "__main__":
    _grid()
