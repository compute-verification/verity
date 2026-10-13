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

`sketch_reading` is where sketch-sha512's bound is read (`PROTOCOL.md` §4): the rank r* that minimizes `sketch_rate`,
the failure each SHA-512 compression of the prover's buys, a guess's 2^-8r beside a fresh key's chance of signs of rank
below r on one of a block's 128 sectors (`sign_rank_at_most`, exact through F2: `gf2_rank_at_most`).  `sketch_failure`
is a session's, and `grind_compressions` what one grinding try costs.

`python -m verity.core.protocols.remote_bandwidth.timing` prints the grid of the report.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

from .codec import BLOCK_BYTES
from .receipts import DEFAULT, DIGEST_BYTES, KINDS, PER_LEAF, SALT_BYTES, SKETCH_ROWS

MS = 1_000_000
US = 1_000
S = 1_000_000_000
JITTER_DOMAIN = b"verity/remote-bandwidth/timing-jitter/v0\0"
SECTOR_BYTES = 32
SECTORS = BLOCK_BYTES // SECTOR_BYTES
SHA512_PER_S = 7.0e9
"""One RTX PRO 6000's SHA-512 compressions a second: block-sha512's 33 a block over 2,097,152 blocks in 9.87 ms
(#1916, art:dea3a309)."""


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

    @property
    def window_timing(self) -> tuple[float, ...]:
        """Each window's μ_timing over its own span `[b_(t−1), D_t]` (`audit.WindowLedger`): the spans of consecutive
        windows overlap by A, and a window's passes start when the pass in flight at its challenge ends, so each
        window is charged A and up to a pass, where the session is charged them once: at most
        `g/(p + g) + (A + J + p)/(T + A)`."""
        starts = (0, *self.boundaries[:-1])
        return tuple(1 - n * self.pass_ns / (d - b) for n, b, d in zip(self.passes, starts, self.deadlines))


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


def gf2_rank_at_most(rows: int, cols: int, r: int) -> Fraction:
    """The chance that a uniform `rows` × `cols` matrix over F2 has rank at most r, exactly: the matrices of rank s
    number `∏_{i<s} (2^rows − 2^i)(2^cols − 2^i)/(2^s − 2^i)`."""
    if r < 0:
        return Fraction(0)
    total = 0
    for s in range(min(r, rows, cols) + 1):
        num = den = 1
        for i in range(s):
            num *= (2 ** rows - 2 ** i) * (2 ** cols - 2 ** i)
            den *= 2 ** s - 2 ** i
        total += num // den
    return Fraction(total, 2 ** (rows * cols))


def sign_rank_at_most(rows: int, r: int, cols: int = SECTOR_BYTES) -> Fraction:
    """An upper bound on the chance that a uniform ±1 `rows` × `cols` matrix has rank at most r over Q, the field of
    the fiber bound.  Multiplying each column by its first row's sign, and replacing every other row v by
    `(first − v)/2`, keeps the rank over Q and leaves the all-ones row over a uniform 0/1 matrix B.  Its rank over Q is
    at least its rank over F2, which is `1 + rank(U)`, U being B with its first column added to the others and then
    dropped: a uniform `(rows − 1) × (cols − 1)` matrix over F2."""
    return gf2_rank_at_most(rows - 1, cols - 1, r - 1)


def sketch_rate(rows: int, r: int) -> Fraction:
    """sketch-sha512's failure a SHA-512 compression when its bound is read at rank r: a guess is right with chance at
    most `2^-8r` while every sector's block of signs has rank at least r, and a fresh key's signs give some sector of
    the block a rank below r with chance at most `SECTORS · sign_rank_at_most(rows, r − 1)`."""
    return Fraction(1, 2 ** (8 * r)) + SECTORS * sign_rank_at_most(rows, r - 1)


def sketch_reading(rows: int = SKETCH_ROWS) -> tuple[int, Fraction]:
    """The rank r* that minimizes `sketch_rate`, and the rate there."""
    return min(((r, sketch_rate(rows, r)) for r in range(1, min(rows, SECTOR_BYTES) + 1)), key=lambda x: x[1])


def sketch_failure(compressions: int, passes: int, rows: int = SKETCH_ROWS, r: int | None = None) -> Fraction:
    """A session's chance that some sketch-sha512 receipt is right without its block's sectors, read at rank r (r* by
    default), against a prover of `compressions` SHA-512 compressions over a session of `passes` passes.  Every key the
    prover sees is a seed it computed or one of the session's, so it sees at most `compressions + passes`, ground or
    not, and every guess is a compression: `C · 2^-8r + (C + P) · SECTORS · sign_rank_at_most(rows, r − 1)`."""
    if r is None:
        r = sketch_reading(rows)[0]
    return (compressions * Fraction(1, 2 ** (8 * r))
            + (compressions + passes) * SECTORS * sign_rank_at_most(rows, r - 1))


def grind_compressions(leaves: int) -> int:
    """SHA-512 compressions of one grinding try as the reference computes it, over a pass of `leaves` receipt leaves,
    reading no block: re-salting one leaf (9: its salt's digest, the key's and the tree leaf), that leaf's hash and its
    path to the root (2, and 3 a level), the next seed (2) and its signs' key (2).  A prover that keeps the constant
    prefixes' midstates spends fewer, and no fewer than 5: the root's last compression, and the seed's and the key's 2
    each, whose first blocks hold the root and the seed."""
    return 9 + 2 + 3 * math.ceil(math.log2(leaves)) + 2 + 2


def log2(x: Fraction) -> float:
    return math.log2(x.numerator) - math.log2(x.denominator)


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
          "one-way delays RTT/2; isolated is a session of one window; worst window is the largest of a W = 100 "
          "session's windows' own, each over [b_(t−1), D_t]")
    print("| pass ms | T s | RTT ms | passes a window | isolated | W = 10 | W = 100 | steady state g/(pass + g) "
          "| worst window |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in (20, 50):
        for w in (0.3, 1, 10):
            for rtt in (1, 5, 20, 50):
                kw = dict(gap_ns=gap, down_ns=_ns(rtt) // 2, up_ns=_ns(rtt) - _ns(rtt) // 2, jitter_ns=jitter)
                runs = [honest(round(w * S), n, _ns(p), **kw) for n in (1, 10, 100)]
                print(f"| {p} | {w} | {rtt} | {runs[0].passes[0]} | "
                      + " | ".join(f"{1 - r.utilization:.2%}" for r in runs) + f" | {chained(_ns(p), gap):.2%} "
                      f"| {max(runs[2].window_timing):.2%} |")
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
    print()
    print(f"# sketch-sha512's rank bound (§4): P(rank ≤ r) of a uniform ±1 rows × {SECTOR_BYTES} block of one sector's "
          f"signs over Q, bounded through F2; rate(r) = 2^-8r + {SECTORS}·P(rank ≤ r − 1) a SHA-512 compression; r* "
          "minimizes it")
    print("| rows | r | P(rank ≤ r), log2 | rate(r), log2 |")
    print("| --- | --- | --- | --- |")
    readings = {k: sketch_reading(k) for k in (SKETCH_ROWS, 64)}
    for k, (star, _) in readings.items():
        for r in range(min(k, SECTOR_BYTES) - 1, star - 2, -1):
            mark = " (r*)" if r == star else ""
            print(f"| {k} | {r}{mark} | {log2(sign_rank_at_most(k, r)):.2f} | {log2(sketch_rate(k, r)):.2f} |")
    print()
    node = 8 * SHA512_PER_S
    for k, (star, rate) in readings.items():
        print(f"{k} rows: r* = {star}, 2^{log2(rate):.2f} a compression; a session's failure stays below 2^-40 up to "
              f"C = 2^{-40 - log2(rate):.2f} compressions ({2 ** (-40 - log2(rate)) / node / 86400:.3g} days of 8 GPUs "
              f"at {SHA512_PER_S / 1e9:.1f}·10^9/s each), and below 2^-64 up to C = 2^{-64 - log2(rate):.2f} "
              f"({2 ** (-64 - log2(rate)) / node:.3g} s)")
    print()
    print("# a grinding try reads no block: compressions to make a key straight (at least 5 with the constant "
          f"midstates kept), at {SHA512_PER_S / 1e9:.1f}·10^9/s on one GPU; seeing a sector's 16 × 32 signs is 16 more")
    print("| pass | leaves | compressions | ns a try | tries a second, 8 GPUs, log2 |")
    print("| --- | --- | --- | --- | --- |")
    for name, blocks in (("one leaf", PER_LEAF), ("8 GiB", 2 ** 21), ("94 GB", 94 * 10 ** 9 // BLOCK_BYTES)):
        n = math.ceil(blocks / PER_LEAF)
        c = grind_compressions(n)
        print(f"| {name} | {n} | {c} | {c / SHA512_PER_S * 1e9:.2f} | {math.log2(node / c):.2f} |")
    print()
    print("| a session of 8 GPUs, every compression a guess or a key, 50 passes a second | C, log2 | failure at r*, 16 "
          "rows | 64 rows |")
    print("| --- | --- | --- | --- |")
    for name, secs in (("1 s", 1), ("10 min", 600), ("1 hour", 3600), ("1 day", 86400), ("30 days", 30 * 86400)):
        c, p = round(node * secs), secs * 50
        print(f"| {name} | {math.log2(c):.2f} | 2^{log2(sketch_failure(c, p)):.2f} | "
              f"2^{log2(sketch_failure(c, p, rows=64)):.2f} |")


if __name__ == "__main__":
    _grid()
