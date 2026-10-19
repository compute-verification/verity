"""RT-ALGO round 2: bit-exact checks of the credited-word algebra of three targets (note:20261019T0618Z-draft-rt-algo-r2).

- (c) The prefix barrier in the served layout. The served unit's receipt is 2 (KS + m) words over KS + m sets; the KS
  prefix sets are running partials, while T_(KS-1), O' and D_1 .. D_(m-1) are final-only (invariant under a reordering of
  the segments). The credit rule counts K = 32 KS MACs a cell, the prefix sets' worth, so the m digit sets carry none. A
  tile of all-zero A rows still does real noised work (X = b), which a dead-tile gate that reads A's bytes would zero.
- (a) Attention. The H1 scheme has no unit for attention's `DotBf16`. On a near-zero P row the residue identity collapses
  a full P̃·X contraction to 32 MACs, but a running-partial word adds one segment, one entry a residue, so it saves nothing.
- (b) FP-native. A one-coin entry with an even coefficient is a single square (FP-ADIC's ½ cap). A two-coin entry is a
  single square mod 2^m iff v₂(c) >= ⌈(m + 1)/2⌉ (17 at s32), so two coins reach c = 1 only below that; mod 8 two c = 4
  entries are one product. e4m3 reaches v₂(c) = 28, and those products overflow s32.

`collect()` returns (passed, failed); the CLI prints them.
"""
from __future__ import annotations

import itertools
import math
import random

from verity.core.protocols.pouw.circuit import compile as C, h1tile, segment as SG
from verity.core.protocols.pouw.circuit.h1tile import Layout, reference

M32 = (1 << 32) - 1


def _rand_i8(rng, n):
    return [rng.randint(-127, 127) for _ in range(n)]


def _unit(rng, r, c, K, zero_a=False):
    a = [[0] * K for _ in range(r)] if zero_a else [_rand_i8(rng, K) for _ in range(r)]
    b = [_rand_i8(rng, K) for _ in range(c)]
    salt = bytes(rng.getrandbits(8) for _ in range(64))
    roots = (bytes(rng.getrandbits(8) for _ in range(64)), bytes(rng.getrandbits(8) for _ in range(64)))
    return a, b, salt, rng.randint(0, 1 << 20), roots


# -- (c) the prefix barrier in the served layout -----------------------------------------------------------------------

def check_receipt_is_2_sets(rng):
    """The served unit's receipt is 2 (KS + m) words, one pair a set: KS prefix sets, O' and D_1 .. D_(m-1)."""
    for r, c, K in ((4, 8, 64), (4, 8, 128), (8, 16, 256)):
        lay = Layout(r, c, K)
        _z, receipt = reference(*_unit(rng, r, c, K))
        if lay.sets != lay.ks + lay.m or len(receipt) != 2 * lay.sets:
            return False, f"{r}x{c} K={K}: sets {lay.sets}, receipt {len(receipt)}, want 2 (KS + m) = {2 * (lay.ks + lay.m)}"
    return True, "receipt = 2 (KS + m) words at 4x8 K=64/128 and 8x16 K=256"


def check_final_words_are_order_invariant(rng):
    """Reordering the segments (X and W' moved as units) changes some T_s, s < KS - 1 (running partials), and leaves
    T_(KS-1), O' and every D_d unchanged (final-only: functions of the whole contraction and the residue sums)."""
    ks = 4
    m = SG.n_digits(ks)
    xr = [[_rand_i8(rng, 32) for _ in range(4)] for _ in range(ks)]
    bb = [[[rng.getrandbits(1) for _ in range(32)] for _ in range(4)] for _ in range(ks)]
    wr = [[_rand_i8(rng, 32) for _ in range(8)] for _ in range(ks)]
    cc = [[[rng.getrandbits(1) for _ in range(32)] for _ in range(8)] for _ in range(ks)]
    u = [[h1tile.u129(rng.getrandbits(8)) for _ in range(32)] for _ in range(8)]
    z = [[[rng.getrandbits(1) for _ in range(32)] for _ in range(4)] for _ in range(m - 1)]
    base = SG.call_words(xr, bb, wr, cc, u, z)
    perm = [2, 0, 3, 1]
    moved = SG.call_words(*[[seq[p] for p in perm] for seq in (xr, bb, wr, cc)], u, z)
    final = [ks - 1, *range(ks, ks + m)]
    if any(base[i] != moved[i] for i in final):
        return False, f"a final-only word moved: {[i for i in final if base[i] != moved[i]]}"
    if all(base[i] == moved[i] for i in range(ks - 1)):
        return False, "no running partial T_0 .. T_(KS-2) moved under a reordering"
    names = ", ".join([f"T_{ks - 1}", "O'", *(f"D_{d}" for d in range(1, m))])
    return True, f"KS={ks}, m={m}: {names} ({len(final)} of {ks + m} sets) order-invariant; T_0..T_{ks - 2} move"


def check_credit_is_prefix_worth():
    """The credit rule counts K = 32 KS MACs a cell (#2639's `credit.tally`, Cr = sum count N K), the prefix sets' worth,
    so the m digit sets are uncredited: at Llama-3.1-8B's K their share of the sets is at or above the headline 0.656%.
    This is the rule's formula; #2639 gives no useful-work credit until the same-GEMM gate."""
    rows = []
    for K, want_m in ((4096, 2), (14336, 3)):
        ks = K // 32
        m = SG.n_digits(ks)
        if 32 * ks != K or m != want_m:
            return False, f"K={K}: KS={ks}, m={m}, want m={want_m}"
        rows.append(f"K={K}: KS={ks}, m={m}, digit sets {m / (ks + m):.2%}")
    return True, "; ".join(rows)


def check_zero_rows_do_noised_work(rng):
    """A tile of all-zero A rows still carries noised work, X = b (main's ungated unit): its receipt has nonzero words. A
    dead-tile gate that zeroes the receipt where every A byte is zero (#2639's `PouwH1Tile_v2`) removes that work, while
    admission reads the queue, not the row."""
    _z, receipt = reference(*_unit(rng, 4, 8, 64, zero_a=True))
    if not any(w & M32 for w in receipt):
        return False, "an all-zero A tile's receipt is all zero without a gate"
    return True, f"all-zero A, 4x8 K=64: {sum(1 for w in receipt if w & M32)} of {len(receipt)} receipt words nonzero"


# -- (a) attention -----------------------------------------------------------------------------------------------------

def check_no_h1_unit_for_attention():
    """`eps1-sm120-h1fx1` has units for the s8 kinds only: attention's QKᵀ (`DotBf16`) has none and there is no `ANY`
    fallback, so compiling it raises CompileError("scheme"): no attention word is credited under H1 on main."""
    sch = C.SCHEMES[h1tile.SCHEME]
    if set(sch.units) != set(h1tile.KINDS) or C.ANY in sch.units or "DotBf16_v1" in h1tile.KINDS:
        return False, f"units {sorted(sch.units)}"
    try:
        sch.unit(C.Dot("DotBf16_v1", 64, None, None, None, None, None), C.Tile(4, 8, ()))
    except C.CompileError as e:
        if e.code != "scheme":
            return False, f"DotBf16 raised {e.code!r}, not 'scheme'"
        return True, f"units {sorted(sch.units)}; DotBf16 -> CompileError('scheme')"
    return False, "DotBf16 got a unit"


def check_near_zero_p_residue_trick(rng):
    """Adopted P·V at r = L = 32, a near-zero P row (P = -64 in Lean coordinates, so P̃ = -64 + U(κ mod 32)): the full
    contraction equals 32 MACs over the residue sums, a collapse of k MACs; but each 32-entry segment holds one entry a
    residue, so a running-partial word (one segment) still needs 32 MACs: the prefix barrier defeats it."""
    S, p0 = 4, -64
    k = 32 * S
    U = [rng.randint(-64, 64) for _ in range(32)]
    X = [rng.randint(-128, 127) + rng.getrandbits(1) for _ in range(k)]
    full = sum((p0 + U[t % 32]) * X[t] for t in range(k)) & M32
    Y = [sum(X[t] for t in range(r, k, 32)) for r in range(32)]
    via = sum((p0 + U[r]) * Y[r] for r in range(32)) & M32
    if full != via:
        return False, "the residue identity fails"
    if any(sorted(t % 32 for t in range(32 * s, 32 * s + 32)) != list(range(32)) for s in range(S)):
        return False, "a segment holds a residue twice"
    return True, f"k={k}: full contraction = 32 residue MACs ({k} direct); one entry a residue a segment"


# -- (b) FP-native -----------------------------------------------------------------------------------------------------

def _square_one(c, m):
    M = 1 << m
    return any((2 * lam * vb * vw - c) % M == 0 for lam in range(M) for vb in range(M) for vw in range(M))


def _square_two(c, m):
    M = 1 << m
    for lam, vB, vb in itertools.product(range(M), repeat=3):
        if (2 * lam * vB * vb - c) % M:
            continue
        if any((2 * lam * vB * w - c) % M == 0 and (2 * lam * vb * w) % M == 0 for w in range(M)):
            return True
    return False


def check_one_coin_even_is_square():
    """One live bit a side, c·β·b: a single square mod 2^m (½ a product, paired) iff c is even (FP-ADIC's ½ cap)."""
    for m in range(2, 6):
        for e in range(m + 1):
            c = (1 << e) % (1 << m)
            if _square_one(c, m) != (e >= 1):
                return False, f"m={m}, v2(c)={e}"
    return True, "m=2..5: single square iff v2(c) >= 1"


def check_two_coin_threshold():
    """Two coins on one side, c·β·(b + b′): a single square mod 2^m iff v₂(c) >= ⌈(m + 1)/2⌉. At s32 that is 17, so the
    two-coin design's c = 1 holds only for v₂(Δ_A·Δ_W) <= 16."""
    for m in range(2, 7):
        t = math.ceil((m + 1) / 2)
        for e in range(m + 1):
            c = (1 << e) % (1 << m)
            if _square_two(c, m) != (e >= t):
                return False, f"m={m}, v2(c)={e}: square={_square_two(c, m)}, threshold {t}"
    return True, f"m=2..6: single square iff v2(c) >= ceil((m+1)/2); at m=32 that is {math.ceil(33 / 2)}"


def check_two_coin_one_product_mod8():
    """Mod 8, two c = 4 two-coin entries are one product, (S1 − S2)(S1 + S2) with S_k = β_k + 2 b_k + 2 b′_k, plus an
    affine part, at every bit point: FP-NATIVE G1's "SAT mod 8, L = 2: 1 product UNSAT" holds only for v₂(c) <= 1."""
    M, c = 8, 4
    pts = list(itertools.product((0, 1), repeat=6))

    def resid(zz):
        B1, b1, p1, B2, b2, p2 = zz
        s1, s2 = B1 + 2 * b1 + 2 * p1, B2 + 2 * b2 + 2 * p2
        return ((s1 - s2) * (s1 + s2) - c * (B1 * b1 + B1 * p1 + B2 * b2 + B2 * p2)) % M

    a0 = resid((0,) * 6)
    a = [(resid(tuple(int(j == i) for j in range(6))) - a0) % M for i in range(6)]
    if not all((a0 + sum(x * y for x, y in zip(a, zz)) - resid(zz)) % M == 0 for zz in pts):
        return False, "the one-product program is not the target plus an affine part"
    return True, f"one product = target + affine (a0={a0}, a={a}) at all {len(pts)} bit points"


def check_e4m3_reaches_threshold():
    """On e4m3 (finest gap 1), a side's gap is 2^u, u = 0 .. 14, so v₂(c) = u_A + u_W reaches 28 >= 17; those
    high-binade products also overflow s32 (above 2^31)."""
    vals = sorted({(m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
                   for e in range(16) for m in range(8) if not (e == 15 and m == 7)} - {0.0})
    fine = min(q - p for p, q in zip(vals, vals[1:]))
    us = {round(math.log2((q - p) / fine)) for p, q in zip(vals, vals[1:])}
    top = (vals[-1] / fine) ** 2
    if max(us) != 14 or 2 * max(us) < 17 or top <= 2 ** 31:
        return False, f"u {min(us)}..{max(us)}, max product 2^{math.log2(top):.2f}"
    return True, f"u per side {min(us)}..{max(us)}, v2(c) up to {2 * max(us)}; max product 2^{math.log2(top):.2f} > 2^31"


def collect(seed: int = 20261019):
    rng = random.Random(seed)
    checks = [
        ("c1 receipt is 2 (KS + m)", lambda: check_receipt_is_2_sets(rng)),
        ("c2 final-only words order-invariant", lambda: check_final_words_are_order_invariant(rng)),
        ("c3 the credit rule counts the prefix sets' worth", check_credit_is_prefix_worth),
        ("c4 zero rows do noised work", lambda: check_zero_rows_do_noised_work(rng)),
        ("a1 no H1 unit for attention", check_no_h1_unit_for_attention),
        ("a2 near-zero P residue trick", lambda: check_near_zero_p_residue_trick(rng)),
        ("b1 one-coin even entry is a square", check_one_coin_even_is_square),
        ("b2 two-coin square threshold", check_two_coin_threshold),
        ("b3 two-coin L=2 one product mod 8", check_two_coin_one_product_mod8),
        ("b4 e4m3 reaches the threshold", check_e4m3_reaches_threshold),
    ]
    passed, failed = [], []
    for name, fn in checks:
        ok, detail = fn()
        (passed if ok else failed).append((name, detail))
    return passed, failed


def main():
    passed, failed = collect()
    for name, detail in passed:
        print(f"PASS  {name}: {detail}")
    for name, detail in failed:
        print(f"FAIL  {name}: {detail}")
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
