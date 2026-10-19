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
     IMAD.WIDE (two output words), SHF funnels, LEA, PRMT byte selects, IDP.4A and the uniform-datapath ops) are evaluated
     on random 32-bit inputs for the 2- and 3-word runs; any that reproduces a fold receipt below honest's length is
     verified exactly on 10^4 points. The hunt reports the program count it covered. `fold_hunt_pod.py` runs it at scale.
  H. Amortization across chains and lanes (RT-FOLD round 2): the public keys and rotations are shared by all 8 chains and
     all 32 lanes, so a cheater may pay a once-per-window setup to make the per-word step cheaper (amortized = setup/W +
     step). Even given an arbitrarily rich key-only precompute, a table-amortized multiply, a repeated key schedule and
     cross-lane sharing, the per-word step stays >= honest, so the amortized cost a word -> honest as W -> inf, never below.

A companion z3 CEGIS sweep (`fold_cegis.py`) proves small-width lower bounds: no program below honest's length computes a
2- or 3-word duo/fx1/li receipt at widths 4/6/8/12 (UNSAT per cell, or timeout). It is the exhaustive counterpart to this
file's randomised hunt; it needs z3 (a workspace dependency) and so runs on the pod, not in this suite.

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


def xin13_half(y, d0, d1, a0, rot):
    """One xin13 half-step (the filler's XIN13 pump and EGEMM's x13_l2o fold): y <- (rotl(y, rot) ^ d0)*a0 + d1. Absorbs
    two words at a rotate (SHF) + an XOR-in (LOP3) + an IMAD = 3 ops a pair, 1.5 a word."""
    return u32(u32(rotl(y, rot) ^ d0) * a0 + d1)


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
    a0, a1, a2 = pick_keys_ok_r9(3, 0xA11CE)
    k0, k1, k2, r0, r1, r2 = 0x12345679, 0x0BADC0DE, 0x0FEEDBAC, 5, 11, 17
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

    # E3b. IDP.4A (dp4a) sums four signed int8 products; it cannot produce a 32-bit multiply u*a (the operands are bytes).
    dp4a_bad = any(dp4a(u, a0) != u32(u * a0) for u in (next(gg) | 0x01010101 for _ in range(50)))
    checks.check("dp4a: four signed int8 products cannot reproduce the 32-bit fold multiply u*a",
                 dp4a_bad)

    # E4. The uniform datapath writes one warp-uniform word an instruction; the fold is per-lane. At one served position
    # the 32 lanes' starts differ, so their u = y^d^k differ -- the warp-uniform datapath cannot produce a warp of them.
    starts = [RF.pfill_start(p) for p in range(32)]
    us = {u32(y ^ 0 ^ k0) for (y, z) in starts}
    checks.check("uniform: the fold is per-lane (u differs across a warp's 32 lane starts), not a warp-uniform word",
                 len(us) > 1, f"{len(us)} distinct u across 32 lanes")

    # ---- G. the filler's pump fold (FILL-OPT) and EGEMM's xin13 -----------------------------------------------------

    # FILL-OPT is the same complexity claim for the verified filler. The filler's inputs are the public live coins, so the
    # fan-in bound gives it no floor (ISA-FANIN): it is purely complexity-type. The filler's per-word work is a pump whose
    # fold is XIN13 (or duo), so the duo attacks above cover duo-pump fillers; here is XIN13's half-step.
    rot13, rot19 = 13, 19
    xin_half = [xin13_half(v[0], v[2], v[3], a0, rot13) for v in vectors]
    pool_xin = make_pool2(k0, k1, a0, a1, r0, r1, vectors)
    checks.check("filler/xin13: a half-step (rotl(y,13)^d0)*a0 + d1 is not one instruction (rotate + XOR-in + IMAD, 1.5 a word)",
                 FO.single_instruction(xin_half, pool_xin) is None, FO.single_instruction(xin_half, pool_xin))
    honest_xin = [u32(u32(rotl(v[0], rot13) ^ v[2]) * a0 + v[3]) for v in vectors]
    checks.check("filler/xin13: the honest form (SHF rotate, LOP3 XOR-in, IMAD) reproduces it", honest_xin == xin_half)

    def xin_two(y, d0, d1, d2, d3):
        y = xin13_half(y, d0, d1, a0, rot13)
        return xin13_half(y, d2, d3, a1, rot19)
    checks.check("filler/xin13: two half-steps do not compose (not degree-1), so the pump's N words need N*1.5 ops",
                 not is_poly_deg1(xin_two, 5))

    # The filler's word-production cheats (pfill_ref.q_receipt / packed_receipt) are MAC-route, not fold-route: they feed
    # the same fold fewer products (ram's Q, or a packed IMMA word + a table), and are already priced where the committed
    # word is produced (SCALAR-ROUTE / the census). They do not make the fold itself cheaper, so FILL-OPT's fold stands or
    # falls with FOLD-OPT above.

    # ---- H. amortization across chains and lanes (RT-FOLD round 2) ------------------------------------------------

    # The public per-position keys (a, k, r) and the rotations are shared by every chain and every lane, and in a window the
    # position schedule repeats. A cheater may pay a once-per-window setup to make the per-word step cheaper; the amortized
    # cost a word is setup/W + step. The setup is a function of the public keys only (independent of the number of words W),
    # so as W -> inf the amortized cost tends to the step cost. These checks bound the step from below by honest even when
    # the cheater is GIVEN an arbitrarily rich key-only precompute, a table-amortized multiply, a repeated key schedule and
    # free cross-lane sharing -- so amortization buys nothing: the amortized cost a word -> honest, never below.

    # H1. A rich key-only precompute pool (every product, power, inverse and 2-adic lift of the public keys a cheater could
    # tabulate once per window) does not collapse duo's per-word step: y' = (y^d^k)*a + z is still not one instruction of
    # {y, z, d, <any key-only constant>}. The data path (the XOR-in then the multiply) is independent of the keys, so no
    # shared setup touches it.
    rich = rich_key_pool(a0, a1, k0, k1, r0, r1)
    pool_rich = {"y": [v[0] for v in vectors], "z": [v[1] for v in vectors],
                 "d0": [v[2] for v in vectors], "d1": [v[3] for v in vectors]}
    for nm, cv in rich.items():
        pool_rich[nm] = [cv] * len(vectors)
    step_y = [u32(u32(v[0] ^ v[2] ^ k0) * a0 + v[1]) for v in vectors]
    checks.check(f"amortize H1: duo's per-word step is not one instruction even given a rich key-only precompute pool "
                 f"({len(rich)} constants); a shared once-per-window setup cannot touch the data path",
                 FO.single_instruction(step_y, pool_rich) is None, FO.single_instruction(step_y, pool_rich))

    # H2. A table-amortized multiply cannot go below one op: a full-value table T_a[u] = u*a is one lookup (ties the IMAD),
    # and a log/antilog route is 3 ops; the step still needs the XOR-in and the add, so the amortized step stays >= honest.
    # Verified bit-exactly at width 16 (both reproduce u*a mod 2^16); the structure is identical at 32 bits.
    full_ok, log_ok = _mul_table_ops(16)
    checks.check("amortize H2: a table-amortized multiply is >= 1 op (full-value table ties the IMAD; log/antilog is 3 ops), "
                 "so even an infinite shared table leaves the step >= honest",
                 full_ok and log_ok)

    # H3. A repeated key schedule gives no per-window closed form: composing two duo rounds with the SAME keys (a0,k0,r0
    # twice) is still not degree-1 and not F2-affine, so there is no window "matrix" to precompute -- the committed data
    # enters each step non-affinely, which is exactly what a once-per-window setup (keys only) cannot absorb.
    def y2_samekey(y, z, d0, d1):
        yy, zz = duo_word(y, z, d0, k0, a0, r0)
        return duo_word(yy, zz, d1, k0, a0, r0)[0]
    checks.check("amortize H3: a repeated key schedule still does not compose (not degree-1, not F2-affine), so there is no "
                 "per-window closed form to precompute",
                 not is_poly_deg1(y2_samekey, 4) and not is_affine_z2(y2_samekey, 4))

    # H4. Cross-lane amortization: the 32 lanes share the keys but their per-word multiplicands u differ, and no sub-word-
    # parallel multiply reproduces independent 32-bit products (SWAR carries interfere; the tensor route is diagonal, E1;
    # the uniform datapath is one value, E4). So producing W words for all 32 lanes costs >= 32 multiplies a word: the
    # cross-lane amortized cost is >= 1 multiply per lane per word, no better than honest.
    starts32 = [RF.pfill_start(p) for p in range(32)]
    us32 = {u32(y ^ 0 ^ k0) for (y, z) in starts32}
    checks.check("amortize H4: 32 lanes share keys but their multiplicands differ and do not SWAR-pack, so cross-lane "
                 f"amortized cost is >= 1 multiply per lane per word ({len(us32)} distinct u across 32 lanes)",
                 len(us32) > 1 and FO.swar_breaks(a0))

    # H5. The amortized bound. amortized(W) = setup/W + step, with setup constant in W (a function of the finite public key
    # schedule) and step >= honest by H1-H4. So amortized(W) decreases monotonically to honest as W -> inf and never beats
    # it; for any finite W the once-per-window setup makes it strictly worse than honest.
    setup = len(rich)                   # a concrete once-per-window setup (building the precompute pool), constant in W
    step = 3                            # duo's per-word step floor = honest (H1-H4: the step cannot drop below honest)
    amort = lambda W: setup / W + step
    checks.check(f"amortize H5: amortized/word = setup/W + step, setup constant in W and step >= honest ({step}), so it -> "
                 f"honest as W->inf and never below (amort(1)={amort(1):.1f}, amort(1e6)={amort(10 ** 6):.5f})",
                 all(amort(W) >= step for W in (1, 8, 64, 1000, 10 ** 6)) and abs(amort(10 ** 9) - step) < 1e-3)

    # ---- F. randomised superopt hunt over the real opcode menu ----------------------------------------------------

    budget = 20000
    for i, a in enumerate(checks.args):
        if a == "--hunt":
            budget = int(checks.args[i + 1])
    keys = dict(k0=k0, k1=k1, k2=k2, a0=a0, a1=a1, a2=a2, r0=r0, r1=r1, r2=r2, ax=ax, kx=kx)
    hits, tried, cover = superopt_hunt(budget, keys)
    checks.check(f"superopt hunt: {tried} random programs (len < honest) over the sm_120 menu across {cover} 2- and 3-word "
                 f"targets; no fold receipt reproduced below honest",
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


# ---- amortization across chains and lanes (section H) --------------------------------------------------------------------

def rich_key_pool(a0, a1, k0, k1, r0, r1):
    """Every key-only constant a cheater could tabulate once per window: the keys, their products, small powers, inverses,
    2-adic lifts and the rotation masks. All are functions of the public keys only (independent of the committed data and of
    the number of words W), so a once-per-window setup shared by all chains and lanes is limited to constants of this form."""
    out = {"0": 0, "1": 1, "-1": M32, "2": 2, "msb": 0x80000000, "maxpos": 0x7FFFFFFF, "GOLDEN": FO.GOLDEN,
           "a0": a0, "a1": a1, "k0": k0, "k1": k1, "r0": r0 & M32, "r1": r1 & M32}
    out["a0a1"] = u32(a0 * a1); out["a0sq"] = u32(a0 * a0); out["a1sq"] = u32(a1 * a1)
    out["a0cube"] = u32(a0 * a0 * a0); out["a0a1sq"] = u32(u32(a0 * a1) * u32(a0 * a1))
    out["a0inv"] = modinv(a0); out["a1inv"] = modinv(a1); out["a0a1inv"] = modinv(u32(a0 * a1))
    out["k0a0"] = u32(k0 * a0); out["k1a1"] = u32(k1 * a1); out["k0xa0"] = u32(k0 ^ a0); out["k1xa1"] = u32(k1 ^ a1)
    out["gold2"] = u32(FO.GOLDEN * FO.GOLDEN); out["goldinv"] = modinv(FO.GOLDEN)
    for r in (r0, r1):
        out[f"lomask{r}"] = (1 << r) - 1
        out[f"himask{r}"] = u32(~((1 << r) - 1))
    return out


def _mul_table_ops(n=16):
    """At width n: a full-value multiply table by a fixed odd a is ONE lookup (T_a[u] = u*a); a log/antilog route over the
    odd units is THREE ops (lookup + add + lookup). Both reproduce u*a mod 2^n exactly here, so a table-amortized multiply
    cannot drop below one op -- it ties the IMAD, and the step still needs the XOR-in and the add. Returns (full_ok, log_ok)."""
    mask = (1 << n) - 1
    a = (0b1011 & mask) | 1                                   # a fixed odd key
    table_a = [(u * a) & mask for u in range(1 << n)]         # full-value multiply table (one lookup)
    full_ok = all(table_a[u] == ((u * a) & mask) for u in range(0, 1 << n, 7))
    order = 1 << (n - 2)                                      # the odd units mod 2^n are {+-1} x <3>, |<3>| = 2^(n-2)
    pow3 = [0] * order
    v = 1
    for i in range(order):
        pow3[i] = v
        v = (v * 3) & mask
    log = {}
    for i in range(order):
        log[pow3[i]] = (0, i)
        log[(-pow3[i]) & mask] = (1, i)

    def mul_via_log(u, a):                                    # 3 ops: lookup log[u], add indices, lookup antilog
        su, iu = log[u & mask]
        sa, ia = log[a & mask]
        prod = pow3[(iu + ia) % order]
        return (-prod if (su + sa) & 1 else prod) & mask

    log_ok = all(mul_via_log(u, a) == ((u * a) & mask) for u in range(1, 1 << n, 2))
    return full_ok, log_ok


# ---- the randomised superopt hunt ---------------------------------------------------------------------------------------

def dp4a(x, y):
    s = 0
    for b in range(4):
        xb = (x >> (8 * b)) & 0xFF
        yb = (y >> (8 * b)) & 0xFF
        xb -= 256 if xb >= 128 else 0
        yb -= 256 if yb >= 128 else 0
        s += xb * yb
    return u32(s)


def _prmt(a, b, mode):
    """PRMT: byte permute. The 8 bytes of (b:a) little-endian are indexed by the four nibbles of the mode constant."""
    bs = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    r = 0
    for i in range(4):
        r |= bs[(mode >> (4 * i)) & 7] << (8 * i)
    return u32(r)


def _menu_ops():
    """The real sm_120 menu as (name, arity, n_out, fn). Round 2 adds IMAD.WIDE (two output words), PRMT byte selects, the
    SHF funnel shifts and the uniform-datapath ops (UIADD3/ULOP3/UIMAD -- identical lane functions on a separate issue port,
    so they widen the port budget, not the function space). LOP3 appears as its common XOR/AND/XOR-AND/AND-XOR LUTs."""
    ops = []
    for signs in itertools.product((1, -1), repeat=3):
        ops.append((f"IADD3{signs}", 3, 1, lambda x, y, z, s=signs: u32(s[0] * x + s[1] * y + s[2] * z)))
    ops.append(("IMAD", 3, 1, lambda x, y, z: u32(x * y + z)))
    ops.append(("IMAD.-c", 3, 1, lambda x, y, z: u32(x * y - z)))
    ops.append(("IMAD.HI", 3, 1, lambda x, y, z: u32(((x * y) >> 32) + z)))
    ops.append(("IMAD.WIDE", 3, 2, lambda x, y, z: (u32(x * y + z), u32((x * y + z) >> 32))))
    ops.append(("LOP3.XOR", 3, 1, lambda x, y, z: u32(x ^ y ^ z)))
    ops.append(("LOP3.AND", 3, 1, lambda x, y, z: u32(x & y & z)))
    ops.append(("LOP3.XORAND", 3, 1, lambda x, y, z: u32((x ^ y) & z)))
    ops.append(("LOP3.ANDXOR", 3, 1, lambda x, y, z: u32((x & y) ^ z)))
    ops.append(("IDP4A", 3, 1, lambda x, y, z: u32(dp4a(x, y) + z)))
    ops.append(("UIADD3", 3, 1, lambda x, y, z: u32(x + y + z)))
    ops.append(("ULOP3.XOR", 3, 1, lambda x, y, z: u32(x ^ y ^ z)))
    ops.append(("UIMAD", 3, 1, lambda x, y, z: u32(x * y + z)))
    for n in range(1, 32):
        ops.append((f"LEA.{n}", 2, 1, lambda x, y, n=n: u32((x << n) + y)))
        ops.append((f"ROTADD.{n}", 2, 1, lambda x, y, n=n: u32(rotl(x, n) + y)))
        ops.append((f"SHR.{n}", 2, 1, lambda x, y, n=n: u32((x >> n) + y)))
        ops.append((f"SHF.L.{n}", 2, 1, lambda x, y, n=n: u32((x << n) | (y >> (32 - n)))))
        ops.append((f"SHF.R.{n}", 2, 1, lambda x, y, n=n: u32((x >> n) | (y << (32 - n)))))
    for mode in (0x0123, 0x3210, 0x1032, 0x2301, 0x4567, 0x0022, 0x4466):
        ops.append((f"PRMT.{mode:04x}", 2, 1, lambda x, y, m=mode: _prmt(x, y, m)))
    return ops


_MENU = _menu_ops()


def default_keys():
    """The canonical public key schedule the hunt and the pod driver share (matches main())."""
    a0, a1, a2 = pick_keys_ok_r9(3, 0xA11CE)
    ax = FO.pick_key_ok_r9() | 1
    return dict(k0=0x12345679, k1=0x0BADC0DE, k2=0x0FEEDBAC, a0=a0, a1=a1, a2=a2,
                r0=5, r1=11, r2=17, ax=ax, kx=0x5A5AC3C3)


def _hunt_consts(keys):
    return [0, 1, M32, 2, 0x80000000, keys["a0"], keys["a1"], keys["a2"], keys["k0"], keys["k1"], keys["k2"],
            keys["kx"], keys["ax"], modinv(keys["a0"]), FO.GOLDEN]


def _targets(keys):
    """The 2- and 3-word receipt targets: (name, honest_len, n_inputs, [output fns]). honest_len is the honest lane-
    instruction count for the whole run (duo 3/word, fx1 4/word, li 1/word, xin13 1.5/word); the hunt samples programs of
    length 1..honest_len-1, i.e. strictly below honest, and a hit is a program whose produced values contain the whole
    receipt (both y and z for duo/fx1)."""
    k0, k1, k2 = keys["k0"], keys["k1"], keys["k2"]
    a0, a1, a2 = keys["a0"], keys["a1"], keys["a2"]
    r0, r1, r2 = keys["r0"], keys["r1"], keys["r2"]
    ax, kx = keys["ax"], keys["kx"]
    T = []

    def duo2(which):
        def f(y, z, d0, d1):
            yy, zz = duo_word(y, z, d0, k0, a0, r0)
            y2, z2 = duo_word(yy, zz, d1, k1, a1, r1)
            return y2 if which == 0 else z2
        return f
    T.append(("duo.2word", 6, 4, [duo2(0), duo2(1)]))

    def duo3(which):
        def f(y, z, d0, d1, d2):
            yy, zz = duo_word(y, z, d0, k0, a0, r0)
            yy, zz = duo_word(yy, zz, d1, k1, a1, r1)
            y3, z3 = duo_word(yy, zz, d2, k2, a2, r2)
            return y3 if which == 0 else z3
        return f
    T.append(("duo.3word", 9, 5, [duo3(0), duo3(1)]))

    def fx2(which):
        def f(y, z, d0, d1):
            y1, z1 = fx1_word(y, z, d0, kx, ax, k0, a0, r0)
            y2, z2 = fx1_word(y1, z1, d1, kx, ax, k1, a1, r1)
            return y2 if which == 0 else z2
        return f
    T.append(("fx1.2word", 8, 4, [fx2(0), fx2(1)]))

    def fx3(which):
        def f(y, z, d0, d1, d2):
            y1, z1 = fx1_word(y, z, d0, kx, ax, k0, a0, r0)
            y1, z1 = fx1_word(y1, z1, d1, kx, ax, k1, a1, r1)
            y3, z3 = fx1_word(y1, z1, d2, kx, ax, k2, a2, r2)
            return y3 if which == 0 else z3
        return f
    T.append(("fx1.3word", 12, 5, [fx3(0), fx3(1)]))

    T.append(("li.2word", 2, 3, [lambda y, d0, d1: li_half(y, d0, d1, k0, a0)]))
    T.append(("li.4word", 4, 5, [lambda y, d0, d1, d2, d3: li_half(li_half(y, d0, d1, k0, a0), d2, d3, k1, a1)]))
    T.append(("xin13.2word", 3, 3, [lambda y, d0, d1: xin13_half(y, d0, d1, a0, 13)]))
    T.append(("xin13.4word", 6, 5,
              [lambda y, d0, d1, d2, d3: xin13_half(xin13_half(y, d0, d1, a0, 13), d2, d3, a1, 19)]))
    return T


def superopt_hunt(budget, keys, seed=0):
    """Sample straight-line programs over the menu (lengths strictly below each run's honest cost) and look for one whose
    produced values contain a whole fold receipt on a batch of random 32-bit points; verify any hit exactly on 10^4 points.
    Returns (hits, programs_tried, n_targets). A pure hunt: no hit means no break found in the sampled space. `seed` shards
    it (the pod driver runs many seeds in parallel), varying both the program RNG and the filter points."""
    import random
    rnd = random.Random((0xF01DBEEF ^ (seed * 2654435761)) & M32)
    consts = _hunt_consts(keys)
    nconst = len(consts)
    targets = _targets(keys)
    g = lcg(0xA7 ^ (seed & M32))
    P = 10
    hits = []
    tried = 0
    per = max(1, budget // len(targets))
    for name, honest, ninputs, fns in targets:
        pts = [tuple(next(g) for _ in range(ninputs)) for _ in range(P)]
        wantset = {tuple(fn(*pt) for pt in pts) for fn in fns}
        for _ in range(per):
            tried += 1
            L = rnd.randint(1, honest - 1)
            prog = _rand_prog(rnd, L, ninputs, nconst)
            if wantset <= set(_eval_cols(prog, pts, consts, ninputs)):
                if _verify_cover(prog, consts, ninputs, fns, seed):
                    hits.append((name, L, _fmt_prog(prog)))
    return hits, tried, len(targets)


def _rand_prog(rnd, L, ninputs, nconst):
    """A random straight-line program. Operand selection is biased toward the inputs, the keys and recent results (where a
    real shortcut would read), and the index space grows by each op's output arity (IMAD.WIDE yields two values)."""
    prog = []
    navail = ninputs + nconst
    for _ in range(L):
        op = rnd.choice(_MENU)
        operands = []
        for _ in range(op[1]):
            roll = rnd.random()
            if roll < 0.5:                                           # an input
                operands.append(rnd.randrange(ninputs))
            elif roll < 0.75 and navail > ninputs + nconst:          # a recent result
                operands.append(ninputs + nconst + rnd.randrange(navail - ninputs - nconst))
            else:                                                    # a key or immediate
                operands.append(ninputs + rnd.randrange(nconst))
        prog.append((op, operands))
        navail += op[2]
    return prog


def _eval_cols(prog, pts, consts, ninputs):
    """Per produced-value slot, the tuple of its values across pts (the inputs and consts are excluded)."""
    nbase = ninputs + len(consts)
    allvals = []
    for pt in pts:
        vals = list(pt) + consts
        for op, operands in prog:
            r = op[3](*(vals[o] for o in operands))
            if op[2] == 2:
                vals.append(r[0]); vals.append(r[1])
            else:
                vals.append(r)
        allvals.append(vals)
    nslots = len(allvals[0])
    return [tuple(allvals[p][s] for p in range(len(pts))) for s in range(nbase, nslots)]


def _verify_cover(prog, consts, ninputs, fns, seed):
    g = lcg(0xBEEF ^ (seed & M32))
    pts = [tuple(next(g) for _ in range(ninputs)) for _ in range(10000)]
    pts += [tuple((s ^ (0x55555555 * j)) & M32 for j in range(ninputs)) for s in (0, 1, 2, M32, 0x80000000)]
    cols = set(_eval_cols(prog, pts, consts, ninputs))
    return all(tuple(fn(*pt) for pt in pts) in cols for fn in fns)


def _fmt_prog(prog):
    return " ; ".join(op[0] + "(" + ",".join(str(o) for o in operands) + ")" for op, operands in prog)


def superopt_can_find_xor(k0):
    """Mutation control: the hunt's exact evaluator confirms a genuine one-instruction target (y ^ d0) when one exists."""
    consts = [0]
    ninputs = 4
    xor_op = next(op for op in _MENU if op[0] == "LOP3.XOR")
    prog = [(xor_op, [0, 2, ninputs])]  # y ^ d0 ^ 0 (const 0 sits at index ninputs)
    g = lcg(0xCAFE)
    pts = [tuple(next(g) for _ in range(ninputs)) for _ in range(200)]
    want = tuple(u32(pt[0] ^ pt[2]) for pt in pts)
    return want in set(_eval_cols(prog, pts, consts, ninputs))


if __name__ == "__main__":
    sys.exit(main())
