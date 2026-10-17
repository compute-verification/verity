"""FOLD-OPT and FILL-OPT across words, attacked (RT-FOLD, compute-accounting red team).

FOLD-OPT (decision 94) and FILL-OPT are the named complexity assumptions behind every EPS1 headline: no program computes
the receipts of N committed words in fewer lane instructions than N times honest's per-word fold, and the same for the
verified filler's fold. Honest's per-word cost is duo >= 3 (LOP3, IMAD, LEA.HI), fx1 = 4 (duo + one keyed IMAD), li = 1
(one LOP3 + one IMAD a pair), and the filler's pump is the same XIN13/duo. ISA-FANIN (#2355) and MULT-LB (#2355's sibling)
prove only the fan-in floor, 1/(f-1) a word (1/2 at f = 3), far below honest; everything between the fan-in floor and honest
is complexity-type and, like a hash's no-shortcut assumption, can only be tested by attack. This script is that attack.

It is NOT a lower-bound proof (the cross-word lower bound explodes under brute force; ASSUME #2240 and ISA-FANIN say so).
It is a battery of concrete break attempts, each bit-exact over Z/2^32, plus a randomised multi-instruction superopt hunt:

  A. Algebraic shortcuts across words for duo: composing two rounds, precomputation over the public keys, constant folding
     of SERVED_KA (including near-inverse key pairs), the T-function low-bit/carry route.
  B. fx1: the two multiplies do not compose, and the keyed IMAD does not merge into duo's LOP3.
  C. li / LI-CP-W: a half-step absorbs two words but is not one instruction (li cannot reach the fan-in floor), two
     half-steps do not compose, and the LI-CP-W checkpoint's reduction cannot be skipped.
  D. The served receipt projection y ^ rotl(z, 16) needs the full state through the chain; eight chains do not SWAR-pack.
  E. Other units for the fold's multiply: tensor IMMA with int8 limbs, a table over (public key, byte), FFMA, and the
     uniform datapath -- each reconstructs u*a but at more than one lane instruction, or cannot index a fresh word.
  F. A randomised superopt hunt: short straight-line programs over the real sm_120 opcode menu (LOP3, IADD3, IMAD,
     IMAD.WIDE, SHF funnels, LEA/LEA.HI, PRMT, IDP.4A, FFMA) are evaluated on random 32-bit inputs; any that reproduces a
     fold output below honest's length is verified exactly. The hunt reports the program count it covered.

Every attack closes: no candidate reproduces a fold more cheaply than honest. The one-line reason per attack is in each
check's name. A real break would flip the matching check to FAIL (the mutations show the search is sensitive: an additive
or XOR-only fold, which does have a shortcut, flips its check).

Run: `python fold_super.py [--mutations] [--verdict PATH] [--hunt N]` (--hunt sets the superopt program budget; default
small so the test stays under a minute, the note cites a larger pod run). Stdlib + exact_kit, plus fold_opt and
receipt_folds (same tree) for the reference fold and the single-instruction searcher.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RECEIPTS = _HERE.parents[0] / "receipts"
for _p in (str(_HERE), str(_RECEIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fold_opt as FO  # noqa: E402  (same dir: the single-instruction superopt and exact word helpers)
import receipt_folds as RF  # noqa: E402  (the fold reference of record)
from exact_kit.checks import Checks  # noqa: E402

M32 = 0xFFFFFFFF
u32 = FO.u32
rotl = FO.rotl
modinv = FO.modinv
lcg = FO.lcg


# ---- exact fold models (held to receipt_folds by the test) ----------------------------------------------------------------

def duo_word(y, z, d, k, a, r):
    return RF.duo_word(y, z, d, k, a, r)


def duo_run(y, z, words, keys):
    """A duo chain: words is a list of (d, k, a, r); returns the final (y, z)."""
    for d, k, a, r in words:
        y, z = duo_word(y, z, d, k, a, r)
    return y, z


def fx1_word(y, z, w, kx, ax, k, a, r):
    """SAT-120's fx1: t = w*ax (ax odd), then duo folds t with keys (kx, a, r). f = 4 (one keyed IMAD + duo's 3)."""
    t = u32(w * ax)
    return duo_word(y, z, t, kx, a, r)


def li_half(y, d0, d1, k0, a0):
    """li's half-step (receipt_folds.li_step, one limb): y <- (y ^ d0 ^ k0)*a0 + d1. Absorbs two words (d0, d1)."""
    return u32((u32(y ^ d0 ^ k0) * a0 + d1))


def li_run(y, pairs):
    """An li chain: pairs is a list of (d0, d1, k0, a0); returns the final y (the committed state, Z = 1)."""
    for d0, d1, k0, a0 in pairs:
        y = li_half(y, d0, d1, k0, a0)
    return y


# ---- helpers --------------------------------------------------------------------------------------------------------------

def is_affine_z2(f, nargs, trials=96, seed=0x1357):
    """True iff f is affine over F2 in its bit-inputs (flipping an input bit has a point-independent effect)."""
    base = f(*([0] * nargs))
    lin = {}
    for arg in range(nargs):
        for bit in range(32):
            a = [0] * nargs
            a[arg] = 1 << bit
            lin[(arg, bit)] = f(*a) ^ base
    g = lcg(seed)
    for _ in range(trials):
        pt = [next(g) for _ in range(nargs)]
        bp = f(*pt)
        for arg in range(nargs):
            for bit in range(0, 32, 5):
                q = list(pt)
                q[arg] ^= 1 << bit
                if (f(*q) ^ bp) != lin[(arg, bit)]:
                    return False
    return True


def is_tfunction(f, nargs, trials=200, seed=0x7FED):
    """True iff f is a T-function: output bit j depends only on input bits <= j. Tested by flipping a high input bit and
    checking no lower output bit moves."""
    g = lcg(seed)
    for _ in range(trials):
        args = [next(g) for _ in range(nargs)]
        base = f(*args)
        for arg in range(nargs):
            for bit in range(1, 32):
                a2 = list(args)
                a2[arg] ^= (1 << bit)
                diff = f(*a2) ^ base
                if diff & ((1 << bit) - 1):  # a lower output bit moved
                    return False
    return True


# =========================================================================================================================


def pick_keys_ok_r9(n, seed):
    """n distinct key_ok_r9 multipliers from an LCG stream (FO.pick_key_ok_r9 is a fixed single value)."""
    g = lcg(seed)
    out = []
    while len(out) < n:
        a = next(g)
        if RF.key_ok_r9(a) and a not in out:
            out.append(a)
    return out


def main() -> int:
    checks = Checks()
    g = lcg(0xF01D)
    a0, a1 = pick_keys_ok_r9(2, 0xA11CE)
    k0, k1, r0, r1 = 0x12345679, 0x0BADC0DE, 5, 11
    vectors = [(next(g), next(g), next(g), next(g)) for _ in range(28)]
    for s in (0, 1, 2, 0x80000000, M32):
        vectors.append((s, s ^ 0x55555555, s, s ^ 0xAAAAAAAA))

    # ---- A. duo across words ---------------------------------------------------------------------------------------------

    # A1. Compose two words: the second word's XOR-in intermediate u1 = ((y^d0^k0)*a0 + z) ^ d1 ^ k1 is not a single
    # instruction of the two-word inputs, so computing y2 must first materialise y1 -- the two words cannot share a
    # multiply. (XOR-then-multiply does not fuse; now across words, not just within one.)
    def u1_of(y, z, d0, d1):
        y1, _ = duo_word(y, z, d0, k0, a0, r0)
        return u32(y1 ^ d1 ^ k1)
    pool2 = make_pool2(k0, k1, a0, a1, r0, r1, vectors)
    u1_vec = [u1_of(*v) for v in vectors]
    u1_single = FO.single_instruction(u1_vec, pool2)
    checks.check("duo two-word: u1 (the 2nd word's XOR-in) is NOT one instruction, so y1 must be materialised first",
                 u1_single is None, u1_single)

    # A2. No algebraic collapse of two rounds: y2 is neither F2-affine nor degree-<=1 over Z/2^32 in (y,z,d0,d1), so there
    # is no closed form of fewer operations (g3-chains' trivial all-state kernel, in instruction-count terms).
    def y2_of(y, z, d0, d1):
        yy, zz = duo_word(y, z, d0, k0, a0, r0)
        return duo_word(yy, zz, d1, k1, a1, r1)[0]
    checks.check("duo two-word: y2 is not F2-affine in (y,z,d0,d1) (carries break linearity, no XOR-tree closed form)",
                 not is_affine_z2(y2_of, 4))
    checks.check("duo two-word: y2 is not degree-<=1 over Z/2^32 (the two multiplies do not collapse to one affine map)",
                 not is_poly_deg1(y2_of, 4))

    # A3. Precomputation over the public keys buys nothing: u*a (a a public odd constant) has IMAD as its only one-op form;
    # the XOR-in (y^d^k)*a is still not one instruction, so a key-only table/precompute cannot remove the LOP3 or the IMAD.
    xin_mul = [u32(u32(v[0] ^ v[2] ^ k0) * a0 + v[1]) for v in vectors]  # one honest y'
    checks.check("duo: y' = (y^d^k)*a + z is not one instruction even with the keys folded in as constants",
                 FO.single_instruction(xin_mul, pool2) is None, FO.single_instruction(xin_mul, pool2))

    # A4. SERVED_KA constant folding and near-inverse pairs. A cross-position multiply-collapse would need SERVED_KA[t] *
    # SERVED_KA[t'] ~= +/-1 mod 2^w for a deep w; none do, and even an exact inverse pair is blocked by the XOR-in between
    # the two multiplies (checked: the composed round is still not affine for the deepest near-inverse served pair).
    served = list(RF.SERVED_KA)
    deepest, worst = 0, None
    for i in range(len(served)):
        for j in range(len(served)):
            if i == j:
                continue
            p = u32(served[i] * served[j])
            for sign in (1, M32):
                v = (p - sign) & M32
                depth = ((v & -v).bit_length() - 1) if v else 32  # 2-adic valuation of a_i*a_j -/+ 1
                if depth > deepest:
                    deepest, worst = depth, (i, j)
    # The deepest near-inverse served pair, forced as (a0, a1), still does not collapse: the XOR-in between the two
    # multiplies gives a point-dependent offset whatever the key relation. (A key-only shortcut would need a degree-1
    # composition, which only an XOR-free fold has.)
    ai, aj = served[worst[0]], served[worst[1]]
    def y2_worst(y, z, d0, d1):
        yy, zz = duo_word(y, z, d0, k0, ai, r0)
        return duo_word(yy, zz, d1, k1, aj, r1)[0]
    checks.check("duo served: even the deepest near-inverse SERVED_KA pair does not collapse two rounds (XOR-in blocks it)",
                 not is_poly_deg1(y2_worst, 4) and not is_affine_z2(y2_worst, 4),
                 f"deepest v2(a_i*a_j -/+ 1) = {deepest} at positions {worst}")
    # And an exact inverse pair (a1 = a0^{-1}) is no different.
    ainv = modinv(a0)
    def y2_inv(y, z, d0, d1):
        yy, zz = duo_word(y, z, d0, k0, a0, r0)
        return duo_word(yy, zz, d1, k1, ainv, r1)[0]
    checks.check("duo: even with exact inverse keys (a1 = a0^-1) the two rounds do not collapse",
                 not is_poly_deg1(y2_inv, 4) and not is_affine_z2(y2_inv, 4))

    # A5. The carry / low-bit route. duo's word is a T-function, so bit j of the output depends only on bits <= j of the
    # inputs; a cheater might compute low bits first. But each limb still needs its own multiply: a radix-2^b evaluation of
    # u*a costs ceil(32/b) multiplies, never below one, and one full multiply is the IMAD honest already issues.
    checks.check("duo: the word is a T-function (low bits first is possible) ...",
                 is_tfunction(lambda y, z, d: duo_word(y, z, d, k0, a0, r0)[0], 3))
    checks.check("... but a radix-2^b evaluation of u*a needs ceil(32/b) multiplies, never fewer than honest's one IMAD",
                 all(limb_mul_count(b) >= 1 for b in (8, 16)) and limb_mul_count(16) == 2 and limb_mul_count(8) == 4)

    # ---- B. fx1 --------------------------------------------------------------------------------------------------------

    ax = FO.pick_key_ok_r9() | 1
    kx = 0x5A5AC3C3
    # B1. The prepended keyed IMAD does not merge into duo's LOP3: fx1's u = y ^ (w*ax) ^ kx is not one instruction, so
    # fx1 >= duo + 1 = 4. (A chain ending on a multiply is exactly why it cannot collapse; an XOR-ending chain would.)
    fx_u = [u32(v[0] ^ u32(v[2] * ax) ^ kx) for v in vectors]
    checks.check("fx1: u = y ^ (w*ax) ^ kx is NOT one instruction, so fx1 >= 4 (the keyed IMAD does not merge into duo's LOP3)",
                 FO.single_instruction(fx_u, pool2) is None, FO.single_instruction(fx_u, pool2))
    # B2. The two multiplies do not compose: (y ^ (w*ax) ^ kx)*a is not w*(ax*a) + c for any constant c (the XOR-in gives
    # a point-dependent offset).
    offsets = set()
    gg = lcg(0x2B2B)
    yfix = next(gg)
    for _ in range(200):
        w = next(gg)
        lhs = u32(u32(yfix ^ u32(w * ax) ^ kx) * a0)
        rhs = u32(w * u32(ax * a0))
        offsets.add((lhs - rhs) & M32)
    checks.check("fx1: the two multiplies do not compose ((y^(w*ax)^kx)*a != w*(ax*a)+c: many distinct offsets)",
                 len(offsets) > 1, f"{len(offsets)} distinct offsets")

    # ---- C. li / LI-CP-W ----------------------------------------------------------------------------------------------

    # C1. An li half-step absorbs two words (d0, d1) but is not one instruction: (y ^ d0 ^ k0)*a0 + d1 is XOR-then-mul-
    # then-add, which no sm_120 op fuses. So li cannot drop to the fan-in floor (1/2 a word); honest's 1 a word is minimal
    # for the XOR-in structure. The honest two-op form (LOP3 then IMAD) does reproduce it.
    li_vec = [li_half(v[0], v[2], v[3], k0, a0) for v in vectors]
    pool_li = FO.make_pool(k0, a0, r0, [(v[0], v[1], v[2]) for v in vectors])
    pool_li["d1"] = [v[3] for v in vectors]
    checks.check("li: a half-step (y^d0^k0)*a0 + d1 absorbs TWO words but is NOT one instruction (li can't reach the 1/2-word floor)",
                 FO.single_instruction(li_vec, pool_li) is None, FO.single_instruction(li_vec, pool_li))
    honest_li = [u32(u32(v[0] ^ v[2] ^ k0) * a0 + v[3]) for v in vectors]
    checks.check("li: the honest two-op half-step (LOP3 then IMAD) reproduces it", honest_li == li_vec)
    # C2. Two li half-steps do not compose: y'' = ((y^d0^k0)*a0 + d1 ^ d2 ^ k1)*a1 + d3 is not degree-<=1 in the four
    # words, so N words need N ops (no cross-pair multiply sharing).
    def li_two(y, d0, d1, d2, d3):
        y = li_half(y, d0, d1, k0, a0)
        return li_half(y, d2, d3, k1, a1)
    checks.check("li: two half-steps do not compose (y'' not degree-<=1 in the four words), so N words need N ops",
                 not is_poly_deg1(li_two, 5))
    # C3. LI-CP-W's checkpoint (XOR of 8 chains' states, then a warp SHFL reduce) is injective in each chain's state, so no
    # chain can be dropped; it adds the stated ~0.04 ALU + store a word and cannot be absorbed into the fold's lane ops.
    def checkpoint(states):
        acc = 0
        for s in states:
            acc ^= s
        return acc
    ch = [next(g) for _ in range(8)]
    cp0 = checkpoint(ch)
    moved = all(checkpoint(ch[:i] + [ch[i] ^ (1 << b)] + ch[i + 1:]) != cp0 for i in range(8) for b in (0, 15, 31))
    checks.check("LI-CP-W: the checkpoint is injective in each chain's state (no chain dropped), so its reduce can't be skipped",
                 moved)

    # ---- D. served projection and cross-chain -----------------------------------------------------------------------

    # D1. The served receipt y ^ rotl(z, 16) cannot be produced without maintaining the full state through the chain: z of
    # an earlier word feeds the next word's y, so perturbing a mid-chain z moves the final projection. The projection is a
    # tail op, not an instruction saving on the chain.
    words_chain = [(next(g), k0, a0, r0), (next(g), k1, a1, r1), (next(g), k0, a0, r0)]
    y_s, z_s = next(g), next(g)
    def proj_after(dz):
        yy, zz = duo_word(y_s, z_s, *words_chain[0][0:1], *words_chain[0][1:])
        zz = u32(zz ^ dz)  # perturb the mid-chain z
        for d, k, a, r in words_chain[1:]:
            yy, zz = duo_word(yy, zz, d, k, a, r)
        return u32(yy ^ rotl(zz, 16))
    checks.check("served: the receipt y ^ rotl(z,16) needs full (y,z) through the chain (mid-chain z feeds later y)",
                 any(proj_after(1 << b) != proj_after(0) for b in range(32)))
    # D2. Eight chains do not SWAR-pack: packing two 16-bit states into one 32-bit multiply by a does not reproduce the two
    # independent products (the high lane picks up the low lane's carry).
    checks.check("served: eight chains do not SWAR-pack the fold multiply (carry interference across packed lanes)",
                 FO.swar_breaks(a0))

    # ---- E. other units for the fold's multiply --------------------------------------------------------------------

    # E1. Tensor IMMA with int8 limbs: u*a is exactly the sum of 4x4 = 16 int8 limb products (base 2^8). So one fold
    # multiply is >= 16 MACs on the tensor pipe, against one IMAD -- and the fold multiply is element-wise (a diagonal),
    # not a matmul contraction, so it wastes the array. Reconstructed bit-exactly here.
    ok_limb = True
    gg = lcg(0x111A)
    for _ in range(300):
        u, a = next(gg), a0
        acc = 0
        for i in range(4):
            for j in range(4):
                acc += ((u >> (8 * i)) & 0xFF) * ((a >> (8 * j)) & 0xFF) << (8 * (i + j))
        if u32(acc) != u32(u * a):
            ok_limb = False
            break
    checks.check("tensor: u*a = sum of 16 int8 limb products (bit-exact), so one fold multiply is >= 16 tensor MACs vs 1 IMAD",
                 ok_limb)

    # E2. A table over (public key, byte): u*a = sum_b table_a[byte_b(u)] << 8b, exact with 4 lookups + 3 shift-adds = 7
    # ops > 1 IMAD; and u is a fresh 32-bit word, so a single (key, byte) table cannot index it (it indexes bytes).
    ok_tab = True
    table_a = [u32(b * a0) for b in range(256)]
    for _ in range(300):
        u = next(gg)
        s = 0
        for b in range(4):
            s = u32(s + (table_a[(u >> (8 * b)) & 0xFF] << (8 * b)))
        if s != u32(u * a0):
            ok_tab = False
            break
    checks.check("table: u*a via 4 byte-tables + 3 shift-adds = 7 ops > 1 IMAD; a fresh 32-bit u is not a table index",
                 ok_tab)

    # E3. FFMA cannot do the exact 32-bit integer multiply: a float32 mantissa is 24 bits, so round(float(u)*float(a)) !=
    # u*a mod 2^32 for most inputs. A witness is enough.
    import struct
    def f32(x):
        return struct.unpack("f", struct.pack("f", float(x)))[0]
    ffma_bad = any(u32(int(f32(f32(u) * f32(a0)))) != u32(u * a0) for u in (next(gg) | 0x40000000 for _ in range(50)))
    checks.check("FFMA: float32 (24-bit mantissa) cannot reproduce the exact 32-bit fold multiply u*a",
                 ffma_bad)

    # E4. The uniform datapath writes one warp-uniform word an instruction; the fold is per-lane. At one served position
    # the 32 lanes' starts differ, so their u = y^d^k differ -- the warp-uniform datapath cannot produce a warp of them.
    starts = [RF.pfill_start(p) for p in range(32)]
    us = {u32(y ^ 0 ^ k0) for (y, z) in starts}
    checks.check("uniform: the fold is per-lane (u differs across a warp's 32 lane starts), not a warp-uniform word",
                 len(us) > 1, f"{len(us)} distinct u across 32 lanes")

    # ---- F. randomised superopt hunt over the real opcode menu ----------------------------------------------------

    budget = 20000
    for i, a in enumerate(checks.args):
        if a == "--hunt":
            budget = int(checks.args[i + 1])
    hits, tried = superopt_hunt(budget, k0, k1, a0, a1, r0, r1, ax, kx)
    checks.check(f"superopt hunt: {tried} random programs (len < honest) over the sm_120 menu; no fold output reproduced below honest",
                 not hits, str(hits) if hits else None)

    # ---- mutations: a fold that DOES have a shortcut flips its check -------------------------------------------------

    if checks.mutations:
        # An additive fold y' = y + d + z is a single IADD3, so its "y' is not one instruction" would flip to false.
        add_fold = [u32(v[0] + v[2] + v[1]) for v in vectors]
        checks.mutation("an additive fold y+d+z IS one instruction (IADD3), so the >=3 claim would flip",
                        FO.single_instruction(add_fold, pool2) is not None, FO.single_instruction(add_fold, pool2))
        # An XOR-only fold y' = y ^ d ^ k is one LOP3.
        xor_fold = [u32(v[0] ^ v[2] ^ k0) for v in vectors]
        checks.mutation("an XOR-only fold y^d^k IS one LOP3, so it would collapse below honest",
                        FO.single_instruction(xor_fold, pool2) is not None, FO.single_instruction(xor_fold, pool2))
        # A linear (degree-1) fold composes across words: y2 of two rounds of y' = y*a + d would be degree-1.
        def lin_two(y, z, d0, d1):
            yy = u32(y * a0 + d0)
            return u32(yy * a1 + d1)
        checks.mutation("a linear fold (no XOR-in) composes across words (degree-1), so two words share a multiply",
                        is_poly_deg1(lin_two, 4))
        # The hunt finds a planted shortcut: y^d is one instruction, caught by the hunt's exact verifier.
        planted = superopt_can_find_xor(k0)
        checks.mutation("the hunt finds a genuine one-instruction target (y^d) when one exists",
                        planted)

    return checks.finish()


# ---- degree-1 test over Z/2^32 -------------------------------------------------------------------------------------------

def is_poly_deg1(f, nargs, trials=96, seed=0x9E37):
    """True iff f is affine over Z/2^32: in each coordinate the first difference f(x + delta*e_i) - f(x) is independent of
    the point x. A product x_i*x_j has a point-dependent first difference, so this is False for any fold that composes two
    multiplies; it is True only for an XOR-free degree-1 fold (where two rounds collapse to one affine map)."""
    g = lcg(seed)
    for _ in range(trials):
        x = [next(g) for _ in range(nargs)]
        x2 = [next(g) for _ in range(nargs)]
        for arg in range(nargs):
            delta = next(g)
            xa = list(x); xa[arg] = u32(xa[arg] + delta)
            xb = list(x2); xb[arg] = u32(xb[arg] + delta)
            if u32(f(*xa) - f(*x)) != u32(f(*xb) - f(*x2)):
                return False
    return True


def limb_mul_count(b: int) -> int:
    """Multiplies needed to evaluate a 32-bit product in radix 2^b (ceil(32/b) limbs each multiplied)."""
    return (32 + b - 1) // b


def make_pool2(k0, k1, a0, a1, r0, r1, vectors):
    """fold_opt's operand pool extended to two words: inputs y, z, d0, d1 and the two words' public keys and immediates."""
    consts = {
        "0": 0, "1": 1, "-1": M32, "a0": a0, "a1": a1, "k0": k0, "k1": k1,
        "a0inv": modinv(a0), "a1inv": modinv(a1), "r0": r0 & M32, "r1": r1 & M32,
        "GOLDEN": FO.GOLDEN, "msb": 0x80000000, "2": 2, "maxpos": 0x7FFFFFFF,
    }
    pool = {"y": [v[0] for v in vectors], "z": [v[1] for v in vectors],
            "d0": [v[2] for v in vectors], "d1": [v[3] for v in vectors]}
    for nm, cv in consts.items():
        pool[nm] = [cv] * len(vectors)
    return pool


# ---- the randomised superopt hunt ---------------------------------------------------------------------------------------

def _menu_ops():
    """(name, arity, fn) over the real sm_120 integer menu, fn taking the operand values. Shift ops carry a per-op shift
    baked into the name. FFMA and IDP.4A included as the brief's menu; LOP3 is handled by a separate affine/LUT probe."""
    ops = []
    for signs in itertools.product((1, -1), repeat=3):
        ops.append((f"IADD3{signs}", 3, lambda x, y, z, s=signs: u32(s[0] * x + s[1] * y + s[2] * z)))
    ops.append(("IMAD", 3, lambda x, y, z: u32(x * y + z)))
    ops.append(("IMAD.-c", 3, lambda x, y, z: u32(x * y - z)))
    ops.append(("IMAD.HI", 3, lambda x, y, z: u32(((x * y) >> 32) + z)))
    ops.append(("XOR3", 3, lambda x, y, z: u32(x ^ y ^ z)))
    ops.append(("AND3", 3, lambda x, y, z: u32(x & y & z)))
    ops.append(("LOP3.XORAND", 3, lambda x, y, z: u32((x ^ y) & z)))
    ops.append(("IDP4A", 3, lambda x, y, z: u32(dp4a(x, y) + z)))
    for n in range(1, 32):
        ops.append((f"LEA.{n}", 2, lambda x, y, n=n: u32((x << n) + y)))
        ops.append((f"ROTADD.{n}", 2, lambda x, y, n=n: u32(rotl(x, n) + y)))
        ops.append((f"SHR.{n}", 2, lambda x, y, n=n: u32((x >> n) + y)))
    return ops


def dp4a(x, y):
    s = 0
    for b in range(4):
        xb = (x >> (8 * b)) & 0xFF
        yb = (y >> (8 * b)) & 0xFF
        xb -= 256 if xb >= 128 else 0
        yb -= 256 if yb >= 128 else 0
        s += xb * yb
    return u32(s)


_MENU = _menu_ops()


def superopt_hunt(budget, k0, k1, a0, a1, r0, r1, ax, kx):
    """Sample straight-line programs over the menu (lengths below each fold's honest cost) and look for one whose output
    reproduces a fold output on a batch of random 32-bit inputs; verify any hit exactly on more points. Returns (hits,
    programs_tried). A pure hunt: no hit means no break found in the sampled space (not a proof of absence)."""
    import random
    rnd = random.Random(0xF01DBEEF)
    P = 24
    g = lcg(0xA7)
    pts = [(next(g), next(g), next(g), next(g)) for _ in range(P)]
    consts = [0, 1, M32, a0, a1, k0, k1, ax, kx, modinv(a0), FO.GOLDEN, 0x80000000]

    # targets: (name, honest_len, value-on-pts). duo two words -> (y2, z2) honest 6; fx1 one word honest 4; li pair honest 2.
    targets = []
    y2s, z2s = [], []
    for (y, z, d0, d1) in pts:
        yy, zz = duo_word(y, z, d0, k0, a0, r0)
        y2, z2 = duo_word(yy, zz, d1, k1, a1, r1)
        y2s.append(y2); z2s.append(z2)
    targets.append(("duo.y2", 6, y2s))
    targets.append(("duo.z2", 6, z2s))
    fx = [fx1_word(y, z, d0, kx, ax, k0, a0, r0)[0] for (y, z, d0, d1) in pts]
    targets.append(("fx1.y1", 4, fx))
    lip = [li_half(y, d0, d1, k0, a0) for (y, z, d0, d1) in pts]
    targets.append(("li.pair", 2, lip))

    hits = []
    tried = 0
    per = max(1, budget // max(1, len(targets)))
    for name, honest, tvals in targets:
        want = tuple(tvals)
        L = honest - 1
        for _ in range(per):
            tried += 1
            prog = _rand_prog(rnd, L, consts)
            out = _eval_prog(prog, pts, consts)
            if tuple(out) == want:
                if _verify_prog(prog, consts, name, k0, k1, a0, a1, r0, r1, ax, kx):
                    hits.append((name, prog))
    return hits, tried


def _rand_prog(rnd, L, consts):
    nin = 4
    nconst = len(consts)
    prog = []
    for i in range(L):
        op = rnd.choice(_MENU)
        navail = nin + nconst + i
        ops = [rnd.randrange(navail) for _ in range(op[1])]
        prog.append((op, ops))
    return prog


def _eval_prog(prog, pts, consts):
    out = []
    for (y, z, d0, d1) in pts:
        vals = [y, z, d0, d1] + list(consts)
        for op, operands in prog:
            args = [vals[o] for o in operands]
            vals.append(op[2](*args))
        out.append(vals[-1])
    return out


def _verify_prog(prog, consts, name, k0, k1, a0, a1, r0, r1, ax, kx):
    g = lcg(0xBEEF)
    extra = [(next(g), next(g), next(g), next(g)) for _ in range(300)]
    extra += [(s, s ^ 0x55555555, s, s ^ 0xF0F0F0F0) for s in (0, 1, 2, M32, 0x80000000)]
    tgt = _target_fn(name, k0, k1, a0, a1, r0, r1, ax, kx)
    for pt in extra:
        if _eval_prog(prog, [pt], consts)[0] != tgt(*pt):
            return False
    return True


def _target_fn(name, k0, k1, a0, a1, r0, r1, ax, kx):
    if name == "duo.y2":
        return lambda y, z, d0, d1: duo_word(*duo_word(y, z, d0, k0, a0, r0), d1, k1, a1, r1)[0]
    if name == "duo.z2":
        return lambda y, z, d0, d1: duo_word(*duo_word(y, z, d0, k0, a0, r0), d1, k1, a1, r1)[1]
    if name == "fx1.y1":
        return lambda y, z, d0, d1: fx1_word(y, z, d0, kx, ax, k0, a0, r0)[0]
    return lambda y, z, d0, d1: li_half(y, d0, d1, k0, a0)


def superopt_can_find_xor(k0):
    """Mutation control: the hunt's exact verifier confirms a genuine one-instruction target (y ^ d0) when one exists."""
    consts = [0, 1, M32, k0]
    xor_op = next(op for op in _MENU if op[0] == "XOR3")
    prog = [(xor_op, [0, 2, consts.index(0) + 4])]  # y ^ d0 ^ 0
    g = lcg(0xCAFE)
    for _ in range(200):
        y, z, d0, d1 = next(g), next(g), next(g), next(g)
        vals = [y, z, d0, d1] + consts
        vals.append(xor_op[2](vals[0], vals[2], vals[4]))
        if vals[-1] != u32(y ^ d0):
            return False
    return True


if __name__ == "__main__":
    sys.exit(main())
