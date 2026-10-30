"""RT-ALGO round 2: bit-exact checks of the credited-word algebra of three targets (note:20261019T0618Z-draft-rt-algo-r2).

- (c) The prefix barrier in the compiled unit. The unit's receipt is 2 (KS + m) words over KS + m sets; T_0 .. T_(KS-2)
  are running partials, while T_(KS-1), O' and D_1 .. D_(m-1) are final-only (invariant under a reordering of the
  segments). The unit's gates, not only its reference, checkpoint every set, with T_s extending T_(s-1) on segment s.
  The credit rule counts K = 32 KS MACs a cell, the prefix sets' worth: T_(KS-1), 1/KS of it, is credited and
  final-only, and the m digit sets carry none. A tile of all-zero A rows still does real noised work (X = b), which a
  dead-tile gate that reads A's bytes zeroes.
- (a) Attention. The H1 scheme has no unit for attention's `DotBf16`. On a near-zero P row the residue identity collapses
  a full P̃·X contraction to 32 MACs, but a running-partial word adds one segment, one entry a residue, so it saves nothing.
- (b) FP-native. A one-coin entry with an even coefficient is a single square (FP-ADIC's ½ cap). A two-coin entry is a
  single square mod 2^m iff v₂(c) >= ⌈(m + 1)/2⌉, and mod 8 two c = 4 entries are one product: both inside what
  FP-NATIVE's `AdmWrap` (e + e′ < m) excludes. On design A's grids (e4m3i with its coin rule; E2M1×2 at integer block
  scales to 15) e + e′ <= 20 < 32, and e4m3i's coin at 448 leaves E4M3. A run that may shift a word certifies two of
  design A's entries with one product at any gap (Proposition Q's halving, generalized), cell-pure as Proposition Q is.

`collect()` returns (passed, failed); the CLI prints them.
"""
from __future__ import annotations

import itertools
import math
import random

from verity.ml.protocols.pouw.circuit import compile as C, h1tile, segment as SG
from verity.ml.protocols.pouw.circuit.h1tile import Layout, reference

M32 = (1 << 32) - 1


def _rand_i8(rng, n):
    return [rng.randint(-127, 127) for _ in range(n)]


def _unit(rng, r, c, K, zero_a=False):
    a = [[0] * K for _ in range(r)] if zero_a else [_rand_i8(rng, K) for _ in range(r)]
    b = [_rand_i8(rng, K) for _ in range(c)]
    salt = bytes(rng.getrandbits(8) for _ in range(64))
    roots = (bytes(rng.getrandbits(8) for _ in range(64)), bytes(rng.getrandbits(8) for _ in range(64)))
    return a, b, salt, rng.randint(0, 1 << 20), roots


# -- (c) the prefix barrier in the compiled unit -----------------------------------------------------------------------

def check_receipt_is_2_sets(rng):
    """The compiled unit's receipt is 2 (KS + m) words, one pair a set: KS prefix sets, O' and D_1 .. D_(m-1)."""
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
    """The credit rule counts K = 32 KS MACs a cell (#2639's `credit.tally`, Cr = sum count N K), the prefix sets' worth.
    Of those KS sets, T_(KS-1) is final-only (c2), so 1/KS of the credit rests on the fresh-coin barrier alone; the m digit
    sets are uncredited. Over Llama-3.1-8B's seven linears, weighted by MACs, the final-only share is near the headline
    0.656%. This is the rule's formula; #2639 gives no useful-work credit until the same-GEMM gate."""
    rows, last = [], {}
    for K, want_m in ((4096, 2), (14336, 3)):
        ks = K // 32
        m = SG.n_digits(ks)
        if 32 * ks != K or m != want_m:
            return False, f"K={K}: KS={ks}, m={m}, want m={want_m}"
        last[K] = 1 / ks
        rows.append(f"K={K}: KS={ks}, m={m}, final-only credited set 1/{ks} = {1 / ks:.2%}, digit sets {m / (ks + m):.2%}")
    linears = {"q": (4096, 4096), "k": (4096, 1024), "v": (4096, 1024), "o": (4096, 4096),
               "gate": (4096, 14336), "up": (4096, 14336), "down": (14336, 4096)}
    macs = {s: k * n for s, (k, n) in linears.items()}
    weighted = sum(macs[s] * last[linears[s][0]] for s in linears) / sum(macs.values())
    return True, "; ".join(rows) + f"; Llama-3.1-8B linears by MACs {weighted:.2%}"


def check_compiled_unit_is_cp_w():
    """On the compiled unit's own gates (`PouwH1Tile`'s body at one lane, 4 x 8, K = 128: KS = 4, m = 2), not its
    reference: the lane absorbs KS + m sets in one fx1 chain, each read by two `PouwXinFinal` halves (a checkpoint after
    every set); set s < KS is `PouwSegDot0` (s = 0) or `PouwSegDot` extending set s - 1's partials, reading byte segment s
    of its X row; O' extends T_(KS-1); every digit set is a fresh `PouwSegDot0`."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.refs import runs
    K, ks = 128, 4
    m = SG.n_digits(ks)
    grid = C.Tile(4, 8, tuple((i, j) for i in range(4) for j in range(8)))
    body = bind(h1tile.PouwH1Tile, EXACT=bind(BG.GemmAccumulatorS8, K=K, **BG.GEMM_SM120_S8), R=4, C=8,
                PAIRS=grid.pairs).body
    defn = [getattr(nd.fn, "defn", nd.fn) for nd in body.nodes]

    def srcs(k, arg):
        return [r[1] for r in runs(body.nodes[k].args[arg].refs)]

    finals = [k for k, d in enumerate(defn) if d is SG.PouwXinFinal]

    def cp_w(absorbs):
        if len(absorbs) != ks + m or any(srcs(b, 0) != [a] for a, b in zip(absorbs, absorbs[1:])):
            return f"{len(absorbs)} absorbs, not one chain of KS + m = {ks + m}"
        for s, a in enumerate(absorbs):
            halves = sorted(runs(body.nodes[k].args[0].refs)[0][2:] for k in finals if srcs(k, 0) == [a])
            if halves != [(0, 256), (256, 512)]:
                return f"set {s}: checkpoint halves {halves}"
        sets = [srcs(a, 1) for a in absorbs]
        for s, words in enumerate(sets):
            want = SG.PouwSegDot0 if s == 0 or s > ks else SG.PouwSegDot
            if len(words) != 32 or any(defn[w] is not want for w in words):
                return f"set {s}: not 32 {want.name}"
            if 0 < s <= ks and any(srcs(w, 0)[0] not in set(sets[s - 1]) for w in words):
                return f"set {s}: a word does not extend set {s - 1}"
            if s < ks:
                x = 0 if s == 0 else 1
                if any(runs(body.nodes[w].args[x].refs)[0][2:] != (256 * s, 256 * s + 256) for w in words):
                    return f"set {s}: a word reads X off segment {s}"
        return None

    absorbs = [k for k, d in enumerate(defn) if d is SG.PouwFx1Absorb]
    if why := cp_w(absorbs):
        return False, why
    if cp_w([absorbs[0], absorbs[2], absorbs[1], *absorbs[3:]]) is None:
        return False, "the check passes with sets 1 and 2 swapped"
    digits = ", ".join(f"D_{d}" for d in range(1, m))
    return True, (f"one lane, KS={ks}, m={m}: {ks + m} sets, a checkpoint after each; T_s extends T_(s-1) on segment s; "
                  f"O' extends T_{ks - 1}; {digits} fresh; refused with sets 1 and 2 swapped")


def check_zero_rows_do_noised_work(rng):
    """A tile of all-zero A rows still carries noised work, X = b (main's ungated unit): its receipt has nonzero words. A
    dead-tile gate that zeroes the receipt where every A byte is zero (#2639's `PouwH1Tile_v2`) removes that work, while
    admission reads the queue, not the row."""
    _z, receipt = reference(*_unit(rng, 4, 8, 64, zero_a=True))
    if not any(w & M32 for w in receipt):
        return False, "an all-zero A tile's receipt is all zero (a dead-tile gate zeroed its noised work)"
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
    """Two coins on one side, c·β·(b + b′): a single square mod 2^m iff v₂(c) >= ⌈(m + 1)/2⌉, i.e. e + e′ >= m + 1 with
    e = e′ = v₂(c). That is where one word falls; FP-NATIVE's SAT has a row of two cells fall already at e + e′ = m, so
    `AdmWrap`'s e + e′ < m is the line, and this threshold is inside what it excludes."""
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


def _v2(x):
    return (x & -x).bit_length() - 1


def _e4m3i():
    """E4M3's integer values (e4m3fn: no infinities, S.1111.111 is NaN), both signs."""
    pos = {int(v) for v in ((m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
                            for e in range(16) for m in range(8) if not (e == 15 and m == 7)) if v == int(v)}
    return sorted(pos | {-v for v in pos})


def check_e4m3i_design_a_admissible():
    """Design A on e4m3i (FP-NATIVE's FP8-A1): P̃ = ⌊P⌋ + Δ_P β with Δ_P the gap above ⌊P⌋, X̃ = RN(X) + Δ_X (b + b′ − 1)
    with Δ_X the coarser side's gap. Both coins' coefficient is Δ_P Δ_X, so e = e′ = max v₂(Δ_P Δ_X) = 10 and
    e + e′ = 20 < 32: `AdmWrap` holds at s32. At the top the coin leaves the grid: ⌊P⌋ = 448 and RN(X) = ±448 step to
    480, which E4M3 can't hold (a completeness cap, ±416)."""
    g = _e4m3i()
    on = set(g)
    up = {x: y - x for x, y in zip(g, g[1:])}
    down = {y: y - x for x, y in zip(g, g[1:])}
    dp = {x: up[x] for x in g if x in up}
    dx = {x: max(up.get(x, 0), down.get(x, 0)) for x in g}
    off_p = [x for x in g if x not in up]
    off_x = [x for x in g if x - dx[x] not in on or x + dx[x] not in on]
    e = max(_v2(dp[p]) + _v2(dx[x]) for p in dp for x in g if x not in off_x)
    if (max(g), e, sorted(off_p), sorted(off_x)) != (448, 10, [448], [-448, 448]) or 2 * e >= 32:
        return False, f"max {max(g)}, e={e}, P off-grid at {off_p}, X off-grid at {off_x}"
    return True, (f"{len(g)} levels to ±448; e = e' = {e}, e + e' = {2 * e} < 32; "
                  f"coin leaves E4M3 at P = 448 and X = ±448 (to 480)")


def check_fp4_design_a_admissible():
    """Design A on E2M1×2 at integer block scales (FP-NATIVE's FP4-A2, FPX-DESIGN's nvi): a coin's coefficient is
    Δ_P s_P Δ_X s_X. With scales to 15, e + e′ = 2 (2 + 3 + 2 + 3) = 20 < 32; `AdmWrap` at s32 needs
    v₂(s_P) + v₂(s_X) <= 11, which integer UE4M3 scales uncapped (64 = 2^6 .. 256 = 2^8) break."""
    pos = sorted({0, 1, 2, 3, 4, 6, 8, 12})
    gaps = [y - x for x, y in zip(pos, pos[1:])]
    worst = 2 * (max(_v2(d) for d in gaps) + max(_v2(s) for s in range(1, 16)))
    e_uncapped = 2 * (max(_v2(d) for d in gaps) + _v2(256))
    if (worst, 2 * worst) != (10, 20) or 2 * e_uncapped < 32:
        return False, f"e at scales to 15: {worst}; uncapped: {e_uncapped}"
    return True, f"scales to 15: e = e' = {worst}, e + e' = {2 * worst} < 32; a 256 scale both sides: e + e' = {2 * e_uncapped}"


def _prop_q_pair(c1, c2, M, v):
    """(N - affine) >> v mod 2^M for N = (s1 − s2)(s1 + s2), against design A's two entries' bilinear part, at all 64 bit
    points; also N's largest magnitude."""
    q1, q2 = c1 << (v - 1), c2 << (v - 1)
    top = 0
    for B1, b1, p1, B2, b2, p2 in itertools.product((0, 1), repeat=6):
        s1, s2 = B1 + q1 * (b1 + p1), B2 - q2 * (b2 + p2)
        n = (s1 - s2) * (s1 + s2)
        top = max(top, abs(n))
        aff = B1 + q1 * q1 * (b1 + p1) - B2 - q2 * q2 * (b2 + p2)
        if (n - aff) % (1 << v):
            return False, top
        if ((n - aff) >> v) % (1 << M) != (c1 * (B1 * b1 + B1 * p1) + c2 * (B2 * b2 + B2 * p2)) % (1 << M):
            return False, top
    return True, top


def check_design_a_prop_q():
    """Proposition Q's halving (note:20261016T0208Z-draft-fullfloor-s-r4 §1), generalized to design A: two entries with
    coin pairs (β_k, b_k), (β_k, b′_k) of coefficient c_k and no b b′ pair are one product N = (s1 − s2)(s1 + s2),
    s1 = β1 + q1 (b1 + b1′), s2 = β2 − q2 (b2 + b2′), q_k = c_k 2^(v−1): (N − affine) >> v is their bilinear part mod 2^M
    once v >= M + 1 − 2 v₂(c), the b b′ terms landing on 2^M. At e4m3i's c = 2^5 and M = 32, v = 23 and N fits s64; on a
    block's coin part (range [0, 2L c], lifted mod 2^7 at L = 32, c = 1), v = 8 and N fits FP32. One mutation: v one
    short fails. s mixes row i's β with column l's b, so the products are cell-pure, as Proposition Q's are."""
    out = []
    for c1, c2, M in ((32, 32, 32), (32, 128, 32), (1, 1, 7)):
        v = M + 1 - 2 * min(_v2(c1), _v2(c2))
        ok, top = _prop_q_pair(c1, c2, M, v)
        if not ok:
            return False, f"c=({c1},{c2}) M={M} v={v}: not the bilinear part"
        bad, _ = _prop_q_pair(c1, c2, M, v - 1)
        if bad:
            return False, f"c=({c1},{c2}) M={M}: v={v - 1} also certifies"
        out.append(f"c=({c1},{c2}) mod 2^{M}: v={v}, |N| < 2^{top.bit_length()}")
    return True, "; ".join(out)


def collect(seed: int = 20261019, gates: bool = True):
    """Every check; `gates` adds c5, which builds the unit's body (about 30 s)."""
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
        ("b4 e4m3i design A admissible at s32", check_e4m3i_design_a_admissible),
        ("b5 FP4 design A admissible at s32", check_fp4_design_a_admissible),
        ("b6 design A pairs under a shift", check_design_a_prop_q),
    ]
    if gates:
        checks.insert(4, ("c5 the compiled unit's gates are CP-W", check_compiled_unit_is_cp_w))
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
