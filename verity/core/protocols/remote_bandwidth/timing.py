"""The honest schedule of a remote session, the passes it certifies, what the live-coin alternative would cost, and the
verifier's bandwidth for openings.

`honest` plays the honest prover against the verifier's messages in integer nanoseconds.  `σ_0` leaves the verifier at 0
and window t's boundary message at `(t + 1)·T`; each reaches the prover `down_ns` plus a jitter later.  Passes run back
to back, `pass_ns + gap_ns` apart: the gap is the root's reduction, the next seed and its broadcast to every GPU, which
the next pass waits on.  The pass in flight when a boundary message arrives is its window's last, and each root reaches
the verifier `up_ns` plus a jitter after its pass and its reduction end.  Each jitter is in [0, `jitter_ns`], a fixed
pseudorandom pattern.  The tests replay the schedule against `audit.Session` (`tests/test_remote_bandwidth.py`).

`deadline` is the least allowance A that never makes the honest prover late: the boundary message's trip down, the pass
in flight, its root's reduction and its trip up, each trip at its longest.

`live_mu` and `live_best_interval` are the steady state of the alternative `PROTOCOL.md`'s appendix weighs: a live coin a
pass and no chain, so that passes are ordered by the clock alone.

`python -m verity.core.protocols.remote_bandwidth.timing` prints the grid of the report.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .codec import BLOCK_BYTES
from .receipts import DEFAULT, DIGEST_BYTES, KINDS, PER_LEAF, SALT_BYTES

MS = 1_000_000
US = 1_000
S = 1_000_000_000
JITTER_DOMAIN = b"verity/remote-bandwidth/timing-jitter/v0\0"


def deadline(pass_ns: int, *, gap_ns: int = 0, down_ns: int, up_ns: int, jitter_ns: int = 0) -> int:
    """The least A: the boundary message's trip down, the pass in flight, its root's reduction and its trip up, each trip
    at its longest."""
    return down_ns + up_ns + 2 * jitter_ns + pass_ns + gap_ns


def _jitter(n: int, direction: bytes, jitter_ns: int) -> int:
    if not jitter_ns:
        return 0
    h = hashlib.sha512(JITTER_DOMAIN + direction + n.to_bytes(8, "big")).digest()
    return int.from_bytes(h[:8], "big") % (jitter_ns + 1)


@dataclass(frozen=True, slots=True)
class Schedule:
    """The honest prover's session: each window's passes, every root's arrival at the verifier `(t, i, ns)`, each
    boundary's departure `b_t`, and the deadlines `D_t = b_t + A`."""

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


def honest(window_ns: int, windows: int, pass_ns: int, *, gap_ns: int = 0, down_ns: int, up_ns: int, jitter_ns: int = 0,
           deadline_ns: int | None = None) -> Schedule:
    """The honest prover's schedule; the allowance defaults to `deadline`'s."""
    a = deadline(pass_ns, gap_ns=gap_ns, down_ns=down_ns, up_ns=up_ns, jitter_ns=jitter_ns) if deadline_ns is None else deadline_ns
    boundaries = tuple((t + 1) * window_ns for t in range(windows))
    deadlines = tuple(b + a for b in boundaries)
    heard = [down_ns + _jitter(t, b"down", jitter_ns) + (boundaries[t - 1] if t else 0) for t in range(windows + 1)]
    passes, arrivals = [0] * windows, []
    t, start = 0, heard[0]
    while t < windows:
        if start >= heard[t + 1]:
            t += 1
            continue
        end = start + pass_ns
        arrival = end + gap_ns + up_ns + _jitter(len(arrivals), b"up", jitter_ns)
        if arrival > deadlines[t]:
            raise ValueError(f"window {t}'s pass {passes[t]} arrives at {arrival} ns, after the deadline {deadlines[t]}")
        arrivals.append((t, passes[t], arrival))
        passes[t] += 1
        start = end + gap_ns
    return Schedule(tuple(passes), tuple(arrivals), boundaries, deadlines, pass_ns)


def chained(pass_ns: int, gap_ns: int) -> float:
    """The steady state a pass: each pass waits on the root before it, so only the root's reduction and the next seed's
    broadcast, g, go uncovered."""
    return gap_ns / (pass_ns + gap_ns)


def live_mu(interval_ns: int, pass_ns: int, *, gap_ns: int = 0, rtt_ns: int, jitter_ns: int = 0,
            rtt_floor_ns: int = 0) -> float:
    """With a live coin a pass and no chain, the steady state a pass at the least deadline `A = RTT + 2J + p + g`:
    `1 − p/Δ + max(0, A − RTT_floor − Δ)/Δ`, the second term the time one read may serve two passes.  With
    `U = RTT + 2J − RTT_floor`, that is `(g + U)/Δ` up to `Δ* = p + g + U` and `1 − p/Δ` beyond."""
    if interval_ns < pass_ns:
        raise ValueError("coins come faster than passes end")
    a = rtt_ns + 2 * jitter_ns + pass_ns + gap_ns
    return 1 - pass_ns / interval_ns + max(0, a - rtt_floor_ns - interval_ns) / interval_ns


def live_best_interval(pass_ns: int, *, gap_ns: int = 0, rtt_ns: int, jitter_ns: int = 0, rtt_floor_ns: int = 0) -> int:
    """`Δ* = A − RTT_floor`: consecutive passes' read windows just don't overlap, and `live_mu` is
    `(g + U)/(p + g + U)`."""
    return max(pass_ns, rtt_ns + 2 * jitter_ns + pass_ns + gap_ns - rtt_floor_ns)


def opening_bytes(blocks: int, tile_bytes: int = 0) -> int:
    """One opening on the wire (`audit.Opening.size`): the block, its receipt leaf and salt, both paths, the payload's
    salt, the indices, and PoUW's tile evidence, which PoUW's own sampled check already pays for when the draws
    coincide."""
    receipt_depth = math.ceil(math.log2(max(1, math.ceil(blocks / PER_LEAF))))
    payload_depth = math.ceil(math.log2(max(1, blocks)))
    return (4 + 8 + BLOCK_BYTES + PER_LEAF * DIGEST_BYTES + 2 * SALT_BYTES + DIGEST_BYTES * (receipt_depth + payload_depth)
            + tile_bytes + 12)


def samples_for(epsilon: float, delta: float) -> int:
    """The fewest openings with `ε_s ≤ epsilon` at `delta`: `C(n − m, s)/C(n, s) ≤ (1 − m/n)^s`, so the bound on wrong
    receipts is at most `n·(1 − δ^(1/s)) ≤ n·ln(1/δ)/s`.  A draw the prover can grind over `Q` transcripts needs
    `samples_for(epsilon, delta / Q)`."""
    return math.ceil(math.log(1 / delta) / epsilon)


def wrong_bound(units: int, samples: int, delta: float) -> int:
    """An upper bound on `IntegrityProfile.bound` for `Subset(samples)` of `units`: `⌈units·(1 − δ^(1/s))⌉`."""
    return math.ceil(-units * math.expm1(math.log(delta) / samples))


def _ns(ms: float) -> int:
    return round(ms * MS)


def _pct(x: float) -> str:
    return f"{x:.2%}" if x < 1 else "vacuous"


def _grid() -> None:
    gap, jitter = 50 * US, _ns(0.05)
    print(f"# receipts: {DEFAULT} by default, of the block alone; {', '.join(k for k in KINDS if k != DEFAULT)} kept for "
          "comparison.  The schedule and openings below are the same under either kind, with no tile evidence unless "
          "the registration binds z")
    print()
    print(f"# μ_timing of chained sessions: g = {gap // US} µs, J = {jitter / MS} ms each way, A = RTT + 2J + pass + g; "
          "one-way delays RTT/2; isolated is a session of one window")
    print("| pass ms | T s | RTT ms | passes a window | isolated | W = 10 | W = 100 | steady state g/(pass + g) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in (20, 50):
        for w in (0.3, 1, 10):
            for rtt in (1, 5, 20, 50):
                kw = dict(gap_ns=gap, down_ns=_ns(rtt) // 2, up_ns=_ns(rtt) - _ns(rtt) // 2, jitter_ns=jitter)
                runs = [honest(round(w * S), n, _ns(p), **kw) for n in (1, 10, 100)]
                print(f"| {p} | {w} | {rtt} | {runs[0].passes[0]} | "
                      + " | ".join(f"{1 - r.utilization:.2%}" for r in runs) + f" | {chained(_ns(p), gap):.2%} |")
    print()
    s = samples_for(1e-3, 2.0 ** -40)
    print(f"# openings at ε_s = 0.1%, δ = 2^-40: s = {s} from the window's coin; a Fiat–Shamir draw from the window's last "
          "root, ground over Q transcripts, needs s(δ/Q)")
    print("| Q | s | × the coin's |")
    print("| --- | --- | --- |")
    for q in (0, 20, 40, 64):
        sq = samples_for(1e-3, 2.0 ** -(40 + q))
        print(f"| 2^{q} | {sq} | {sq / s:.2f} |")
    print()
    print("# openings: bytes a window and Gb/s per node, with the coin's s")
    for name, blocks in (("one GPU, 94 GB", 94 * 10 ** 9 // BLOCK_BYTES), ("8 GPUs, 752 GB", 8 * 94 * 10 ** 9 // BLOCK_BYTES),
                         ("8 GPUs + 1.5 TB host", (8 * 94 + 1500) * 10 ** 9 // BLOCK_BYTES)):
        b = opening_bytes(blocks)
        print(f"{name}: N = {blocks}, {b} B an opening, {s * b / 1e6:.0f} MB a window; "
              + ", ".join(f"T = {w} s: {s * b * 8 / w / 1e9:.2f} Gb/s" for w in (0.3, 1, 10)))
    print()
    print(f"# appendix, live coins: a coin a pass, no chain; μ_timing + μ_overlap a pass in steady state at the least "
          f"deadline; floor = the trusted RTT floor as a fraction of RTT; U = RTT + 2J − floor")
    print("| pass ms | RTT ms | floor | U ms | Δ = pass | Δ* ms | at Δ* | chained |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in (20, 50):
        for rtt in (0.2, 1, 5, 20, 50):
            for f in (0, 0.5, 0.9):
                kw = dict(gap_ns=gap, rtt_ns=_ns(rtt), jitter_ns=jitter, rtt_floor_ns=_ns(f * rtt))
                star = live_best_interval(_ns(p), **kw)
                u = _ns(rtt) + 2 * jitter - _ns(f * rtt)
                print(f"| {p} | {rtt} | {f} | {u / MS:.2f} | {_pct(live_mu(_ns(p), _ns(p), **kw))} | {star / MS:.2f} "
                      f"| {_pct(live_mu(star, _ns(p), **kw))} | {chained(_ns(p), gap):.2%} |")
    print()
    print("| jitter, at Δ*: pass 20 ms, RTT 1 ms, floor 0.5 | " + " | ".join(f"J = {j} ms" for j in (0, 0.05, 0.2, 0.5, 1))
          + " |")
    print("| --- | --- | --- | --- | --- | --- |")
    row = []
    for j in (0, 0.05, 0.2, 0.5, 1):
        kj = dict(gap_ns=gap, rtt_ns=MS, jitter_ns=_ns(j), rtt_floor_ns=MS // 2)
        row.append(f"{live_mu(live_best_interval(20 * MS, **kj), 20 * MS, **kj):.2%}")
    print("| live coins | " + " | ".join(row) + " |")


if __name__ == "__main__":
    _grid()
