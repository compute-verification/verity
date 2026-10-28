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

`stated` is the claim of `PROTOCOL.md` §1 and §10 read through `audit.ledger`: the honest prover's session on one
node at the node's R and S_chip, under the default codec at the node's stated squaring ceiling, every window's whole
γ_t and the horizon's.  `spanning` is the store that spans every tier in proportion to its read rate, and
`store_floor` the store below which no T brings every window under 1%.  Each takes the node's figures, the Site's
(`verity.hardware.node`): this module holds the schedule and no hardware figure.

`sketch_reading` is where sketch-sha512's bound is read (`PROTOCOL.md` §4): the rank r* that minimizes `sketch_rate`,
the failure each SHA-512 compression of the prover's buys, a guess's 2^-8r beside a fresh key's chance of signs of rank
below r on one of a block's 128 sectors (`sign_rank_at_most`, exact through F2: `gf2_rank_at_most`).  `sketch_failure`
is a session's, and `grind_compressions` what one grinding try costs.

`python -m verity.core.protocols.remote_bandwidth.timing` prints the grid of the report on an RTX PRO 6000 node
(`verity.hardware.node.VY_NEBIUS`), ending with what would bring λ and μ_chip under 0.05% (wider receipts, a larger
store, a flush a pass).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

from verity.core.protocols.profile import IntegrityProfile, Subset
from verity.hardware import node as hardware

from . import audit, codec
from .codec import BLOCK_BYTES
from .receipts import DEFAULT, DIGEST_BYTES, KINDS, PER_LEAF, SALT_BYTES, SKETCH_ROWS

MS = 1_000_000
US = 1_000
S = 1_000_000_000
JITTER_DOMAIN = b"verity/remote-bandwidth/timing-jitter/v0\0"
SECTOR_BYTES = 32
SECTORS = BLOCK_BYTES // SECTOR_BYTES
SKETCH_OUTSIDE = 2 ** 20
"""F, the outside-compute factor: the SHA-512 compressions a prover makes over a session are at most F times the node's
own over its span, Q = F·C (`sketch_failure`'s `outside`).  Grinding a sketch-sha512 key reads no block, so it can run
on any hardware the prover reaches, not only the node's GPUs: 2^20 nodes' SHA-512, about 8.4M RTX PRO 6000s, past any
one operator's fleet.  At 16 rows it costs a session's line 0.4 bits of error (2^-29.95 to 2^-29.55 at 2118 passes),
since AuditRO's 2^-41 a pass dominates it; 64 rows would bring δs under 2^-50 at τ 1.226 a pass on the reader, the
option if that slack is ever cut.  Provisional, under the sketch's random-oracle assumption."""


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


def sketch_failure(compressions: int, passes: int, rows: int = SKETCH_ROWS, r: int | None = None,
                   blocks: int = 0, outside: int = SKETCH_OUTSIDE) -> Fraction:
    """A session's chance that some sketch-sha512 receipt is right without its block's sectors, read at rank r (r* by
    default), over a session of `passes` passes of `blocks` blocks, against a prover of `Q = outside · compressions`
    SHA-512 compressions: `compressions` the node's own over the span (C), and `outside` the outside-compute factor F
    (`SKETCH_OUTSIDE`).  Every key the prover sees is a seed it computed or one of the session's, so it sees at most
    `Q + passes`, ground or not, and every guess is a compression:
    `Q · 2^-8r + (Q + P) · (SECTORS · sign_rank_at_most(rows, r − 1) + N · 2^-512) + (Q + P)² · 2^-257`, the third
    term a key's 512-bit receipt of a block right without its hash, and the last the keys' distance from uniform
    (`SketchHashRandom`'s ε): two of the `Q + P` seeds whose 256-bit sign keys collide share their signs.  At 16 rows
    and r = 12 it is the δ at which `Accounting.Guarantees.Remote.SketchBoundDeployed` derives `SketchBound`."""
    if type(outside) is not int or outside < 1:
        raise ValueError("the outside-compute factor F is a positive integer")
    if r is None:
        r = sketch_reading(rows)[0]
    q = outside * compressions
    return (q * Fraction(1, 2 ** (8 * r))
            + (q + passes) * (SECTORS * sign_rank_at_most(rows, r - 1) + Fraction(blocks, 2 ** 512))
            + Fraction((q + passes) ** 2, 2 ** 257))


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


def spanning(node: hardware.NodeFigures, device_bytes: int) -> int:
    """The store that spans every tier of `node` in proportion to its read rate, `device_bytes` in each device's
    memory: host DRAM holds what it reads in the time the slowest device reads its share, so a pass of the whole store
    reads at R."""
    slowest = min(d.read_bytes_per_s for d in node.devices)
    return len(node.devices) * device_bytes + math.floor(device_bytes * node.host.read_bytes_per_s / slowest)


def stated(node: hardware.NodeFigures, store_bytes: int, window_ns: int, windows: int, *, rtt_ns: int, samples: int,
           delta: float = audit.DELTA, gap_ns: int = 50 * US, jitter_ns: int = 50 * US,
           read_bytes_per_s: float | None = None, capacity_bytes_per_s: float | None = None) -> audit.Ledger:
    """The ledger of the honest prover's session over a store of `store_bytes` on `node`, of `capacity_bytes_per_s`
    (R measured by default; R at spec, the headline): a pass is the store's read at `read_bytes_per_s` (R measured by
    default), `samples` openings a window at `delta` bound its wrong receipts (`wrong_bound`), the node's chips hold
    S_chip, and μ_rebuild is the default codec at the node's squaring ceiling.  A store that spans every tier
    (`spanning`) reads at R; one in device memory alone reads at the devices' rate, and μ_timing charges the idle host
    DRAM.  μ_chip is about S_chip/`store_bytes`, and a window's μ_timing about `(RTT + 2p)/T`, the pass in flight at its
    challenge and the one its boundary cuts short: a larger store trades the one for the other."""
    read_bytes_per_s = node.read_bytes_per_s if read_bytes_per_s is None else read_bytes_per_s
    capacity_bytes_per_s = node.read_bytes_per_s if capacity_bytes_per_s is None else capacity_bytes_per_s
    blocks = store_bytes // BLOCK_BYTES
    pass_ns = math.ceil(blocks * BLOCK_BYTES / read_bytes_per_s * S)
    st = honest(window_ns, windows, pass_ns, gap_ns=gap_ns, down_ns=rtt_ns // 2, up_ns=rtt_ns - rtt_ns // 2,
                jitter_ns=jitter_ns)
    starts = (0, *st.boundaries[:-1])
    record = audit.Record(bytes(64), bytes(64), blocks, BLOCK_BYTES, tuple(
        audit.WindowRecord(t, (blocks,) * n, b, d, d, IntegrityProfile(
            ("receipt",), (blocks * n,), {"receipt": Subset(samples)}, delta, drawn={"receipt": samples}))
        for t, (n, b, d) in enumerate(zip(st.passes, starts, st.deadlines))))
    return audit.ledger(record, capacity_bytes_per_s=capacity_bytes_per_s, chip_bytes=node.chip_bytes,
                        rebuild=audit.Rebuild(codec.DEFAULT, node.squarings_per_s[codec.DEFAULT.field.name]),
                        wrong=lambda pr: wrong_bound(pr.sizes[0], pr.drawn["receipt"], pr.delta))


def store_floor(node: hardware.NodeFigures, gap_ns: int = 0, target: float = 0.01) -> float:
    """The store on `node`, in bytes, spanning every tier, below which no T brings `stated`'s worst γ_t under
    `target`.  As T grows, a window's μ_sample and its ends vanish, and γ_t tends to
    `(g + S_chip/R + λ·p + μ_rebuild·(p + g))/(p + g)` with p the store's read at R: the chain's steady state, μ_chip,
    λ (`audit.named_bits` of the store's blocks, a
    block's share of them depending on the store's size, solved for by iteration) and μ_rebuild.  At `gap_ns` = 0 it
    is μ_chip, λ and μ_rebuild alone."""
    r, chip = node.read_bytes_per_s, node.chip_bytes
    rebuild = audit.Rebuild(codec.DEFAULT, node.squarings_per_s[codec.DEFAULT.field.name]).fraction(r)
    lam, g, blocks = audit.QUERIES_LOG2 / (8 * BLOCK_BYTES), gap_ns / S, 0
    while True:
        store = r * (g * (1 - target + rebuild) + chip / r) / (target - lam - rebuild)
        if store // BLOCK_BYTES == blocks:
            return store
        blocks = int(store // BLOCK_BYTES)
        lam = (audit.named_bits(blocks) + audit.SLACK_LOG2) / (8 * BLOCK_BYTES * blocks)


def _named_bits(k: int, queries_log2: int, binomial: bool) -> int:
    """The bits a pass of k receipts charges for AuditRO's C(q, k) at q = 2^queries_log2: `audit.named_bits`, the
    bound by q^k/k! that `ledger` charges, or k·log2 q, the looser bound by q^k."""
    return audit.named_bits(k, queries_log2) if binomial else k * queries_log2


def _sizing(at: hardware.NodeFigures, gap: int) -> None:
    gpus, read, chip = len(at.devices), at.read_bytes_per_s, at.chip_bytes
    gpu_store, hbm = 94 * 10 ** 9, gpus * 96 * 10 ** 9
    store = spanning(at, gpu_store)
    node = store // BLOCK_BYTES
    p, g_s = store / read, gap / S
    cap = read * (p + g_s)
    print(f"# λ under 0.05%: the node's store over every tier, 94 GB of each GPU's HBM and host DRAM's share "
          f"({store / 1e9:.0f} GB, {p * 1e3:.1f} ms passes), in the steady state at g = {gap // US} µs.  A receipt of m "
          f"blocks: its bits for C(q, k) at q = 2^{audit.QUERIES_LOG2} bounded by q^k, or by q^k/k! as ledger charges "
          "them, and the bits 0.05% allows; an opening on that store with a payload leaf a receipt or a block; τ a pass "
          "scaled from an 8 GiB pass (sketch 5.82 ms, hash 0.33 ms, tree 0.90 ms, over a 5.60 ms read); "
          "sketch-sha512's rate a compression at m·128 sectors, log2")
    print("| m | KiB | q^k bits | λ | q^k/k! bits | λ | bits at 0.05% | opening B, payload a receipt | a block | τ | rate |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in (1, 2, 3, 4, 8):
        k, units = store // (m * BLOCK_BYTES), node // m
        bits = [_named_bits(k, audit.QUERIES_LOG2, b) for b in (False, True)]
        base = (4 + 8 + m * BLOCK_BYTES + PER_LEAF * DIGEST_BYTES + SALT_BYTES + 12
                + DIGEST_BYTES * math.ceil(math.log2(math.ceil(units / PER_LEAF))))
        rate = min(Fraction(1, 2 ** (8 * r)) + m * SECTORS * sign_rank_at_most(SKETCH_ROWS, r - 1)
                   for r in range(1, SKETCH_ROWS + 1))
        print(f"| {m} | {m * 4} | {bits[0] / k:.2f} | {(bits[0] + audit.SLACK_LOG2) / 8 / cap:.4%} | {bits[1] / k:.2f} "
              f"| {(bits[1] + audit.SLACK_LOG2) / 8 / cap:.4%} | {0.0005 * 8 * cap / k:.2f} "
              f"| {base + SALT_BYTES + DIGEST_BYTES * math.ceil(math.log2(units))} "
              f"| {base + m * (SALT_BYTES + DIGEST_BYTES * math.ceil(math.log2(node)))} "
              f"| {(5.82 + (0.33 + 0.90) / m) / 5.60:.3f} | {log2(rate):.2f} |")
    k = store // BLOCK_BYTES
    q_log2 = 0.0005 * 8 * cap / k + math.lgamma(k + 1) / math.log(2) / k
    print(f"at 4 KiB, under 0.05% by q^k/k! needs q ≤ 2^{q_log2:.2f} queries to a pass's oracle: "
          f"{2 ** q_log2 / at.devices[0].sha512_per_s:.0f} s of one GPU's SHA-512, "
          f"{2 ** q_log2 / at.sha512_per_s:.0f} s of the node's")
    print()
    hbm_rate, host_read = at.device_read_bytes_per_s, at.host.read_bytes_per_s
    print(f"# μ_chip under 0.05%: S_chip/(R·(p + g)) by the store a pass reads at R, {chip / 0.0005 / 1e9:.0f} "
          f"GB at 0.05%, {chip / 0.0005 / hbm:.1f} times the node's {hbm // 10 ** 9} GB of HBM.  Host DRAM, at its "
          f"{host_read / 1e9:.1f} GB/s, holding more than its share lengthens a pass past the HBM's read only "
          "by leaving the HBM idle.  A flush of t_F a pass that leaves nothing of the store on chip adds t_F to the gap")
    print("| GB a GPU | store GB | pass ms | μ_chip |")
    print("| --- | --- | --- | --- |")
    for gb in (30, 62, 94, 263, 526):
        s_gb = spanning(at, gb * 10 ** 9)
        print(f"| {gb} | {s_gb / 1e9:.0f} | {s_gb / read * 1e3:.1f} | {chip / (read * (s_gb / read + g_s)):.4%} |")
    print(f"| pass ms, {gpus * gpu_store // 10 ** 9} GB of HBM and the rest host DRAM | store GB | μ_chip | idle |")
    print("| --- | --- | --- | --- |")
    for pass_s in (p, 0.1, 0.2, 1.0):
        reads = min(gpus * gpu_store, hbm_rate * pass_s) + host_read * pass_s
        print(f"| {pass_s * 1e3:.1f} | {reads / 1e9:.1f} | {chip / (read * pass_s):.3%} "
              f"| {1 - reads / (read * pass_s):.2%} |")
    print("| t_F µs | steady state | added |")
    print("| --- | --- | --- |")
    steady = g_s / (p + g_s)
    for tf in (0, 26, 52, 105):
        x = (g_s + tf * 1e-6) / (p + g_s + tf * 1e-6)
        print(f"| {tf} | {x:.4%} | {x - steady:.4%} |")
    print()
    print(f"# the horizon's whole γ and the worst window's γ_t at the stated T = {audit.WINDOW_NS // S} s, "
          f"{store / 1e9:.0f} GB over every tier, as stated and with what lowers λ and μ_chip (a flush: its added steady "
          "state on the timing, μ_chip at 0), and by q^k, the looser count")
    print("| RTT ms | γ | stated | by q^k | 16 KiB receipts | and a 26 µs flush | λ = μ_chip = 0 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    flush = (g_s + 26e-6) / (p + g_s + 26e-6) - steady
    for rtt in (1, 20):
        h = stated(at, store, audit.WINDOW_NS, 600 * S // audit.WINDOW_NS, rtt_ns=_ns(rtt), samples=audit.SAMPLES)
        for name, gamma, terms in (("horizon", h.gamma_horizon, h), ("worst γ_t", h.worst.gamma, h.worst)):
            rest = gamma - terms.counting
            lam = {(m, b): terms.counting * (_named_bits(store // (m * BLOCK_BYTES), audit.QUERIES_LOG2, b)
                                             + audit.SLACK_LOG2) / (audit.named_bits(k) + audit.SLACK_LOG2)
                   for m in (1, 4) for b in (False, True)}
            print(f"| {rtt} | {name} | {gamma:.3%} | {rest + lam[1, False]:.3%} | {rest + lam[4, True]:.3%} "
                  f"| {rest - terms.chip + lam[4, True] + flush:.3%} | {rest - terms.chip:.3%} |")


def _ns(ms: float) -> int:
    return round(ms * MS)


def _pct(x: float) -> str:
    return f"{x:.2%}" if x < 1 else "vacuous"


def _grid(node: hardware.NodeFigures) -> None:
    gap, jitter = 50 * US, _ns(0.05)
    gpus, one = len(node.devices), node.devices[0]
    read, spec, chip, sha512 = node.read_bytes_per_s, node.spec_bytes_per_s, node.chip_bytes, node.sha512_per_s
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
        for w in (1, 20, 60):
            for rtt in (1, 5, 20, 50):
                kw = dict(gap_ns=gap, down_ns=_ns(rtt) // 2, up_ns=_ns(rtt) - _ns(rtt) // 2, jitter_ns=jitter)
                runs = [honest(round(w * S), n, _ns(p), **kw) for n in (1, 10, 100)]
                print(f"| {p} | {w} | {rtt} | {runs[0].passes[0]} | "
                      + " | ".join(f"{1 - r.utilization:.2%}" for r in runs) + f" | {chained(_ns(p), gap):.2%} "
                      f"| {max(runs[2].window_timing):.2%} |")
    print()
    t_s, s = audit.WINDOW_NS // S, audit.SAMPLES
    quarter = samples_for(0.0025, audit.DELTA)
    openings = {1: samples_for(0.01, audit.DELTA), 20: quarter, t_s: s, 90: quarter}
    print(f"# openings at the stated T = {t_s} s, δ = 2^-40: s = {s} from the window's coin ({s / t_s:.0f} a second, "
          f"ε_s = ln(2^40)/s = {math.log(2.0 ** 40) / s:.4%}); the 20 s option's ε_s = 0.25% is {openings[20]} and the "
          f"1 s option's 1% is {openings[1]}.  A Fiat–Shamir draw from the window's last root, ground over Q "
          "transcripts, needs s(δ/Q) = ⌈s·(40 + log2 Q)/40⌉")
    print("| Q | s | × the coin's |")
    print("| --- | --- | --- |")
    for q in (0, 20, 40, 64):
        sq = -(-s * (40 + q) // 40)
        print(f"| 2^{q} | {sq} | {sq / s:.2f} |")
    print()
    print("# openings: bytes a window and Gb/s per node, at each window's openings: "
          + ", ".join(f"T = {w} s: {openings[w]}" for w in (1, 20, t_s)))
    for name, blocks in (("one GPU, 94 GB", 94 * 10 ** 9 // BLOCK_BYTES), ("8 GPUs, 752 GB", 8 * 94 * 10 ** 9 // BLOCK_BYTES),
                         ("8 GPUs + 1.5 TB host", (8 * 94 + 1500) * 10 ** 9 // BLOCK_BYTES)):
        b = opening_bytes(blocks)
        print(f"{name}: N = {blocks}, {b} B an opening; "
              + ", ".join(f"T = {w} s: {openings[w] * b / 1e6:.0f} MB a window, {openings[w] * b * 8 / w / 1e9:.3f} Gb/s"
                          for w in (1, 20, t_s)))
    print()
    hbm_rate = node.device_read_bytes_per_s
    print(f"# the stated claim (§1, §10), T = {t_s} s by default, 20 s and 1 s the options: the worst window's whole γ_t "
          "over its own span at R measured and at spec, and over a 10 min horizon of W windows the pooled μ_sample and "
          f"the whole γ, on one node of {gpus} {one.sku} cards and their host (R = {read / 1e12:.4f} TB/s "
          f"measured, {spec / 1e12:.4f} at spec, S_chip = {chip:,} B), the default codec at the node's "
          "stated ceiling, δ = 2^-40, g = 50 µs, J = 0.05 ms.  Over every tier, the store is GB a GPU of each GPU's HBM "
          "and host DRAM's share (`spanning`), and a pass its read at R; in HBM, the same GB a GPU and nothing on host "
          f"DRAM, a pass its read at the GPUs' {hbm_rate / 1e12:.3f} TB/s")
    print("| T s | openings | GB a GPU | tiers | store GB | pass ms | RTT ms | μ_timing | μ_sample | μ_chip | λ "
          "| μ_rebuild | worst γ_t | at spec | W | pooled μ_sample | horizon γ |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for w_s in (t_s, 20, 1, 90):
        for gb, tiers in ((30, "every"), (62, "every"), (94, "every"), (94, "HBM")):
            store, rate = (spanning(node, gb * 10 ** 9), read) if tiers == "every" else (
                gpus * gb * 10 ** 9, hbm_rate)
            for rtt in (1, 20):
                kw = dict(rtt_ns=_ns(rtt), samples=openings[w_s], read_bytes_per_s=rate)
                g = stated(node, store, w_s * S, 600 // w_s, **kw)
                at_spec = stated(node, store, w_s * S, 600 // w_s, capacity_bytes_per_s=spec, **kw).worst
                x = g.worst
                print(f"| {w_s} | {openings[w_s]} | {gb} | {tiers} | {store / 1e9:.0f} | {store / rate * 1e3:.1f} "
                      f"| {rtt} | {x.timing:.3%} | {x.sample:.3%} | {x.chip:.3%} | {x.counting:.3%} | {x.rebuild:.3%} "
                      f"| {x.gamma:.3%} | {at_spec.gamma:.3%} | {600 // w_s} | {g.horizon:.4%} | {g.gamma_horizon:.3%} |")
    print()
    share = gpus + node.host.read_bytes_per_s / min(d.read_bytes_per_s for d in node.devices)
    print(f"# the store floor (store_floor): below {store_floor(node) / 1e9:.1f} GB over every tier "
          f"({store_floor(node) / share / 1e9:.1f} GB a GPU) μ_chip, λ and μ_rebuild alone exceed 1% at any T; with the "
          f"chain's steady state at g = {gap // US} µs, no T meets 1% below {store_floor(node, gap) / 1e9:.1f} GB "
          f"({store_floor(node, gap) / share / 1e9:.1f} GB a GPU)")
    print()
    _sizing(node, gap)
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
    fl = SKETCH_OUTSIDE.bit_length() - 1
    for k, (star, rate) in readings.items():
        q40, q64 = -40 - log2(rate), -64 - log2(rate)
        print(f"{k} rows: r* = {star}, 2^{log2(rate):.2f} a compression; a session's failure stays below 2^-40 up to "
              f"Q = 2^{q40:.2f} compressions, the node's C = 2^{q40 - fl:.2f} at F = 2^{fl} "
              f"({2 ** (q40 - fl) / sha512:.3g} s of {gpus} GPUs at {one.sha512_per_s / 1e9:.1f}·10^9/s each), "
              f"and below 2^-64 up to Q = 2^{q64:.2f}, C = 2^{q64 - fl:.2f}")
    print()
    print("# a grinding try reads no block: compressions to make a key straight (at least 5 with the constant "
          f"midstates kept), at {one.sha512_per_s / 1e9:.1f}·10^9/s on one GPU; seeing a sector's 16 × 32 signs is 16 "
          "more")
    print(f"| pass | leaves | compressions | ns a try | tries a second, {gpus} GPUs, log2 |")
    print("| --- | --- | --- | --- | --- |")
    for name, blocks in (("one leaf", PER_LEAF), ("8 GiB", 2 ** 21), ("94 GB", 94 * 10 ** 9 // BLOCK_BYTES)):
        n = math.ceil(blocks / PER_LEAF)
        c = grind_compressions(n)
        print(f"| {name} | {n} | {c} | {c / one.sha512_per_s * 1e9:.2f} | {math.log2(sha512 / c):.2f} |")
    print()
    f = f"F = 2^{SKETCH_OUTSIDE.bit_length() - 1}"
    print(f"| a session of {gpus} GPUs, every compression a guess or a key, 50 passes a second | C, log2 | failure at "
          f"r*, 16 rows, F = 1 | {f} | 64 rows, {f} |")
    print("| --- | --- | --- | --- | --- |")
    for name, secs in (("1 s", 1), ("10 min", 600), ("1 hour", 3600), ("1 day", 86400), ("30 days", 30 * 86400)):
        c, p = round(sha512 * secs), secs * 50
        print(f"| {name} | {math.log2(c):.2f} | 2^{log2(sketch_failure(c, p, outside=1)):.2f} | "
              f"2^{log2(sketch_failure(c, p)):.2f} | 2^{log2(sketch_failure(c, p, rows=64)):.2f} |")


if __name__ == "__main__":
    _grid(hardware.node(*hardware.VY_NEBIUS))
