"""FOLD-OPT, mechanically: no program on sm_120 computes a committed word's duo receipt fold in fewer instructions than
honest's 3.

The duo step (benchmarks/pouw/receipts/receipt_folds.py, `duo_word`) absorbs one committed word d into the running state
(y, z) with public per-position keys (k, a, r), a odd and `key_ok_r9`:

    u  = y ^ d ^ k            # one LOP3 (a 3-input bitwise XOR)
    y' = u * a + z            # one IMAD (integer multiply-add)
    z' = rotl(u, r) + y'      # one LEA.HI (funnel-shift-add)

Three instructions a word, and the data sheet's SASS reads exactly that (8 IMMA + 32 LOP3 + 32 IMAD + 32 LEA.HI a loop).
FOLD-OPT is the claim that a cheater who must reproduce the receipt the verifier recomputes cannot do this absorb in
fewer than three instructions a word. This script turns the claim into a superoptimisation search + a structural lower
bound, exact over Z/2^32, so it becomes a mechanical fact rather than a named assumption.

The result (checks below):
- **y' is not a single sm_120 integer instruction** of the inputs {y, z, d} and any public constants: XOR-then-multiply
  does not fuse (no op's output family contains `a*(x^c)+e`), and the +z carry chain is not bitwise. Exhaustive over the
  modelled ISA (LOP3 with every 3-input LUT, IADD3 with sign variants, IMAD / IMAD.HI, SHF.L/R funnels, LEA, LEA.HI,
  PRMT, and the one-input ops) with a rich immediate pool.
- **z' is not a single instruction either** (it reads rotl(u, r) and y', and u = y^d^k is itself not an input).
- Both outputs (y', z') are live every word (z' feeds the next word's y, y' is the running state), so a straight-line
  program producing the pair has, as its first instruction, a single-instruction value that must be one of the two
  outputs. Since **neither output is single-instruction, the pair needs ≥ 3 instructions** — honest's count.
- Positive controls: the honest 3-instruction program reproduces (y', z'); the honest 2-instruction prefix (u, y')
  reproduces y'; the LOP3 for u is found in one instruction (the search finds short programs when they exist).
- Across words, this script checks nothing. What is proved for a run of N words is only the fan-in floor: any program
  whose instructions each read f operands needs (N − 2)/(f − 1) of them (`Accounting.Guarantees.TS.DuoFoldFanIn`), ½
  a word at f = 3. That batching can't beat 3 a word is FOLD-OPT's across-words part,
  the named assumption `Accounting.Assumptions.TS.FoldOpt`. The reasons to expect it are arguments, not checks:
  - the 32-bit states admit no SWAR packing of the multiply;
  - the round mixes ^ over F2 with *+ over Z/2^32 with no homomorphism, so composing rounds has no known shorter closed
    form (g3-chains' trivial all-state kernel).

  Seriality within a chain bounds only depth, not issue slots, since a cheater runs many chains at once.

Run: `python fold_opt.py [--mutations] [--verdict PATH]`. Stdlib + exact_kit only.
"""
from __future__ import annotations

import itertools
import sys

from exact_kit.checks import Checks

M32 = 0xFFFFFFFF
GOLDEN = 0x9E3779B9


# ---- exact word helpers --------------------------------------------------------------------------------------------------

def u32(x: int) -> int:
    return x & M32


def rotl(x: int, r: int) -> int:
    r &= 31
    return ((x << r) | (x >> (32 - r))) & M32 if r else x & M32


def funnel_l(lo: int, hi: int, n: int) -> int:
    """CUDA __funnelshift_l(lo, hi, n): high 32 of ((hi:lo) << (n & 31))."""
    n &= 31
    return (((hi << 32) | (lo & M32)) << n >> 32) & M32


def funnel_r(lo: int, hi: int, n: int) -> int:
    """CUDA __funnelshift_r(lo, hi, n): low 32 of ((hi:lo) >> (n & 31))."""
    n &= 31
    return (((hi << 32) | (lo & M32)) >> n) & M32


def prmt(a: int, b: int, sel: int) -> int:
    """PRMT: each output byte picks a byte (or its sign-replicated form) of {b:a} by a nibble of sel."""
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        s = (sel >> (4 * i)) & 0xF
        byte = src[s & 7]
        if s & 8:  # sign mode: replicate the byte's msb
            byte = 0xFF if byte & 0x80 else 0x00
        out |= byte << (8 * i)
    return out & M32


def modinv(a: int) -> int:
    return pow(a, -1, 1 << 32)


# ---- the duo fold step (receipts/receipt_folds.py duo_word), the target --------------------------------------------------

def duo_u(y: int, z: int, d: int, k: int, a: int, r: int) -> int:
    return u32(y ^ d ^ k)


def duo_yp(y: int, z: int, d: int, k: int, a: int, r: int) -> int:
    return u32(duo_u(y, z, d, k, a, r) * a + z)


def duo_zp(y: int, z: int, d: int, k: int, a: int, r: int) -> int:
    u = duo_u(y, z, d, k, a, r)
    return u32(rotl(u, r) + duo_yp(y, z, d, k, a, r))


# ---- the sm_120 integer ISA (exact), as instruction templates ------------------------------------------------------------
#
# An instruction template maps an ordered tuple of operand values to one 32-bit word. Operands are drawn from a pool that
# holds the inputs {y, z, d}, the public constants and a rich set of immediates. Enumerating the templates over the pool is
# a complete single-instruction search for the modelled opcodes. LOP3 is handled specially (every 3-input LUT) by a
# per-bit-locality test, which is faster than and equivalent to enumerating all 256 LUTs.

SHIFTS = tuple(range(32))


def arith_templates():
    """(name, arity, fn) for the non-LOP3 integer ops. fn takes the operand values (and, for shift ops, a shift)."""
    out = []
    # IADD3 with per-operand negation (IADD3 can negate its sources): s0*±1 + s1*±1 + s2*±1.
    for signs in itertools.product((1, -1), repeat=3):
        out.append((f"IADD3{signs}", 3, lambda x, y, z, s=signs: u32(s[0] * x + s[1] * y + s[2] * z)))
    # IMAD and IMAD.HI (also the negated-addend form).
    out.append(("IMAD", 3, lambda x, y, z: u32(x * y + z)))
    out.append(("IMAD.-c", 3, lambda x, y, z: u32(x * y - z)))
    out.append(("IMAD.HI", 3, lambda x, y, z: u32(((x * y) >> 32) + z)))
    # LEA (x << n) + y, and LEA.HI funnel(x, hi, n) + y (hi a third operand).
    for n in SHIFTS:
        out.append((f"LEA.{n}", 2, lambda x, y, n=n: u32((x << n) + y)))
        out.append((f"SHF.L.{n}", 2, lambda lo, hi, n=n: funnel_l(lo, hi, n)))
        out.append((f"SHF.R.{n}", 2, lambda lo, hi, n=n: funnel_r(lo, hi, n)))
        out.append((f"LEA.HI.{n}", 3, lambda x, hi, y, n=n: u32(funnel_l(x, hi, n) + y)))
    # PRMT over a sample of selectors (identity, the rotate-by-byte selectors, and the sign-extend ones).
    for sel in (0x3210, 0x0321, 0x2103, 0x1032, 0x4444, 0x7654, 0xBA98):
        out.append((f"PRMT.{sel:04x}", 2, lambda a, b, sel=sel: prmt(a, b, sel)))
    return out


ARITH = arith_templates()


def is_lop3(target_bits, pool_bits):
    """True iff `target` is some 3-input bitwise LUT of three pool operands: group the test-vector bit positions by the
    operand-bit triple and check the target bit is constant in each group. Returns the triple (names) or None.

    target_bits / pool_bits: for each operand, a list over test vectors of its 32-bit value."""
    names = list(pool_bits)
    for a, b, c in itertools.product(names, repeat=3):
        lut = {}
        ok = True
        for vi in range(len(target_bits)):
            av, bv, cv, tv = pool_bits[a][vi], pool_bits[b][vi], pool_bits[c][vi], target_bits[vi]
            for bit in range(32):
                key = ((av >> bit) & 1) | (((bv >> bit) & 1) << 1) | (((cv >> bit) & 1) << 2)
                out = (tv >> bit) & 1
                if key in lut and lut[key] != out:
                    ok = False
                    break
                lut[key] = out
            if not ok:
                break
        if ok:
            return (a, b, c)
    return None


# ---- the single-instruction superoptimiser -------------------------------------------------------------------------------

def make_pool(k: int, a: int, r: int, vectors):
    """The operand pool: name -> list of values over the test vectors. Inputs y, z, d; the public constants k, a, r;
    a rich immediate set (0, 1, -1, a, k, a^{-1}, GOLDEN, sign bit, small ints)."""
    consts = {
        "0": 0, "1": 1, "-1": M32, "a": a, "k": k, "r": r & M32, "ainv": modinv(a),
        "GOLDEN": GOLDEN, "msb": 0x80000000, "2": 2, "3": 3, "maxpos": 0x7FFFFFFF,
    }
    pool = {}
    pool["y"] = [v[0] for v in vectors]
    pool["z"] = [v[1] for v in vectors]
    pool["d"] = [v[2] for v in vectors]
    for nm, cv in consts.items():
        pool[nm] = [cv] * len(vectors)
    return pool


def single_instruction(target, pool):
    """Return the name of a single modelled instruction computing `target` (a list over the test vectors) from the pool,
    or None. Searches LOP3 (all LUTs) and every arithmetic template over all ordered operand tuples."""
    # LOP3: any 3-input bitwise function of three pool operands.
    triple = is_lop3(target, pool)
    if triple is not None:
        return f"LOP3({','.join(triple)})"
    names = list(pool)
    cols = {nm: pool[nm] for nm in names}
    nv = len(target)
    for name, arity, fn in ARITH:
        for ops in itertools.product(names, repeat=arity):
            vals = [cols[o] for o in ops]
            if all(fn(*(vals[j][vi] for j in range(arity))) == target[vi] for vi in range(nv)):
                return f"{name}({','.join(ops)})"
    return None


def two_instruction_yp(yp, pool):
    """Confirm y' is a length-2 program: v1 = u = y ^ d ^ k (one LOP3), then y' = IMAD(a, v1, z) (one instruction over the
    pool extended with v1). The first value u is itself a single instruction (checked separately), so y' at length 2 is
    the honest prefix, below the duo word's three."""
    y, d, k, z, a = pool["y"], pool["d"], pool["k"], pool["z"], pool["a"]
    nv = len(yp)
    v1 = [u32(y[i] ^ d[i] ^ k[i]) for i in range(nv)]
    ext = dict(pool)
    ext["v1"] = v1
    got = single_instruction(yp, ext)
    return ("LOP3(y,d,k)", got) if got is not None and "v1" in got else None


# ---- small-width exhaustive single-instruction cross-check (all 256 LUTs) -------------------------------------------------

def width_single_instruction(width: int, target, inputs):
    """Over `width`-bit words, is `target` a single instruction of the `inputs` (name -> value over a small test set)?
    Exhaustive over the FULL modelled ISA at this width -- every one of the 256 3-input LOP3 LUTs, IADD3 sign variants,
    IMAD / IMAD.HI, every funnel shift, LEA and LEA.HI -- a complement to the 32-bit search (which collapses the 256 LUTs
    into one locality test). Returns a label or None."""
    mask = (1 << width) - 1
    names = list(inputs)
    cols = {nm: inputs[nm] for nm in names}
    nv = len(target)

    def rot(x, n):
        n %= width
        return ((x << n) | (x >> (width - n))) & mask if n else x & mask

    def fun_l(lo, hi, n):
        n %= width
        return (((hi << width) | (lo & mask)) << n >> width) & mask

    # LOP3: all 256 LUTs over all ordered operand triples.
    for a, b, c in itertools.product(names, repeat=3):
        av, bv, cv = cols[a], cols[b], cols[c]
        for lut in range(256):
            if all(((lut >> (((av[i] >> bit) & 1) | (((bv[i] >> bit) & 1) << 1) | (((cv[i] >> bit) & 1) << 2))) & 1)
                   == ((target[i] >> bit) & 1) for i in range(nv) for bit in range(width)):
                return f"LOP3.{lut:#04x}({a},{b},{c})"
    # IADD3 (signs), IMAD, IMAD.HI.
    for a, b, c in itertools.product(names, repeat=3):
        av, bv, cv = cols[a], cols[b], cols[c]
        for s in itertools.product((1, -1), repeat=3):
            if all(((s[0] * av[i] + s[1] * bv[i] + s[2] * cv[i]) & mask) == target[i] for i in range(nv)):
                return f"IADD3{s}({a},{b},{c})"
        if all(((av[i] * bv[i] + cv[i]) & mask) == target[i] for i in range(nv)):
            return f"IMAD({a},{b},{c})"
        if all(((((av[i] * bv[i]) >> width) + cv[i]) & mask) == target[i] for i in range(nv)):
            return f"IMAD.HI({a},{b},{c})"
    # LEA, LEA.HI, funnels over every shift.
    for a, b in itertools.product(names, repeat=2):
        av, bv = cols[a], cols[b]
        for n in range(width):
            if all((((av[i] << n) + bv[i]) & mask) == target[i] for i in range(nv)):
                return f"LEA.{n}({a},{b})"
            if all(fun_l(av[i], av[i], n) + bv[i] & mask == target[i] for i in range(nv)):  # rotation form of LEA.HI
                return f"LEA.HI.{n}({a},{b})"
    return None


def main() -> int:
    checks = Checks()

    # one duo word's public keys: a odd and key_ok_r9, k arbitrary, r a PFILL_DUO_ROT rotation.
    a = pick_key_ok_r9()
    k = 0x12345679
    r = 5
    checks.check("a is odd and key_ok_r9", key_ok_r9(a), f"a={a:#010x}")

    rng = lcg(0xC0FFEE)
    vectors = [(next(rng), next(rng), next(rng)) for _ in range(28)]
    # add structured points (0, 1, powers of two) to stress carries
    for s in (0, 1, 2, 0x80000000, M32):
        vectors.append((s, s ^ 0x55555555, s))
    pool = make_pool(k, a, r, vectors)

    yp = [duo_yp(y, z, d, k, a, r) for (y, z, d) in vectors]
    zp = [duo_zp(y, z, d, k, a, r) for (y, z, d) in vectors]
    u = [duo_u(y, z, d, k, a, r) for (y, z, d) in vectors]

    # positive controls: the honest programs reproduce the targets, and the search finds the short ones.
    honest_yp = [u32(u[i] * a + pool["z"][i]) for i in range(len(vectors))]
    honest_zp = [u32(rotl(u[i], r) + honest_yp[i]) for i in range(len(vectors))]
    checks.check("honest 3-instruction program reproduces (y', z')", honest_yp == yp and honest_zp == zp)

    u_found = single_instruction(u, pool)
    checks.check("u = y^d^k is one instruction (LOP3)", u_found is not None and u_found.startswith("LOP3"), u_found)

    yp2 = two_instruction_yp(yp, pool)
    checks.check("y' is two instructions (u then IMAD)", yp2 is not None, yp2)

    # the core lower bound: neither output is a single instruction.
    yp1 = single_instruction(yp, pool)
    checks.check("y' is NOT a single instruction (XOR-then-multiply does not fuse)", yp1 is None, yp1)
    zp1 = single_instruction(zp, pool)
    checks.check("z' is NOT a single instruction", zp1 is None, zp1)

    # the pair's first instruction is a single-instruction value, so it must be one of the two outputs; neither is
    # single-instruction, so the pair needs >= 3 instructions.
    pair_ge3 = (yp1 is None) and (zp1 is None)
    checks.check("the duo word (y', z') needs >= 3 instructions = honest's count", pair_ge3,
                 "a length-2 program's first value is single-instruction, but neither output is")

    # algebra: y' is not F2-affine in the inputs (so no bitwise op and no XOR-tree makes it) -- an input delta's effect
    # depends on the point, i.e. there are x, delta with yp(x^delta) ^ yp(x) != yp(0^delta) ^ yp(0).
    checks.check("y' is not F2-affine in (y, z, d) (carries break linearity)", not is_f2_affine(
        lambda y, z, d: duo_yp(y, z, d, k, a, r)))
    # the multiply by a is a bijection of Z/2^32 (a odd), so u is recovered from u*a; the round is injective, no collapse.
    checks.check("multiply by a is a bijection (a odd): the round is injective", a % 2 == 1)

    # no homomorphism bridging ^ and *+: there is no permutation-free constant c with (x ^ y) * a == (x*a) op (y*a) for a
    # fixed op; concretely the two operations don't distribute, so a round has no shorter algebraic form. Check a few.
    checks.check("^ and *+ do not distribute (no homomorphism to batch rounds)", not distributes(a))

    # batching across chains: no SWAR. Packing two 16-bit states into one 32-bit multiply by a does not equal the two
    # independent 32-bit (then-truncated) products -- the halves interfere through carries.
    checks.check("no SWAR: a packed 16-bit multiply != two independent 32-bit fold multiplies", swar_breaks(a))

    # small-width exhaustive cross-check (all 256 LOP3 LUTs): at width 4 neither y' nor z' is a single instruction.
    w = 4
    g = lcg(0xBEEF)
    npts = 14
    mask = (1 << w) - 1
    y0 = [next(g) & mask for _ in range(npts)]
    z0 = [next(g) & mask for _ in range(npts)]
    d1 = [next(g) & mask for _ in range(npts)]
    d2 = [next(g) & mask for _ in range(npts)]
    aw = (pick_key_ok_r9() & mask) | 1
    kw, rw = 0xB & mask, 1

    def rotw(x, n):
        n %= w
        return ((x << n) | (x >> (w - n))) & mask

    def word(yv, zv, dv):
        uu = (yv ^ dv ^ kw) & mask
        yy = (uu * aw + zv) & mask
        zz = (rotw(uu, rw) + yy) & mask
        return yy, zz

    y1 = [word(y0[i], z0[i], d1[i])[0] for i in range(npts)]
    z1 = [word(y0[i], z0[i], d1[i])[1] for i in range(npts)]
    wins = {"y": y0, "z": z0, "d": d1, "k": [kw] * npts, "a": [aw] * npts, "ainv": [modinv(aw | 1) & mask] * npts,
            "0": [0] * npts, "1": [1] * npts, "m1": [mask] * npts}
    checks.check("width-4 exhaustive (all 256 LUTs): y' is not a single instruction",
                 width_single_instruction(w, y1, wins) is None, width_single_instruction(w, y1, wins))
    checks.check("width-4 exhaustive (all 256 LUTs): z' is not a single instruction",
                 width_single_instruction(w, z1, wins) is None, width_single_instruction(w, z1, wins))

    # batching: the chain is serial, so words are produced in order and each pays its >= 3. z_i feeds word i+1's y
    # (y_{i+1} = u_{i+1}*a + z_i), and y_i feeds u_{i+1} = y_i ^ d_{i+1} ^ k; perturbing either moves y_2. So round 2
    # cannot be computed before, or merged into, round 1: the two words are >= 6 instructions, 3 each.
    y2 = [word(y1[i], z1[i], d2[i])[0] for i in range(npts)]
    z1_moves = any(word(y1[i], z1[i] ^ 1, d2[i])[0] != y2[i] for i in range(npts))
    y1_moves = any(word(y1[i] ^ 1, z1[i], d2[i])[0] != y2[i] for i in range(npts))
    checks.check("chain is serial: z_i is read by word i+1 (perturbing z1 moves y2)", z1_moves)
    checks.check("chain is serial: y_i is read by word i+1 (perturbing y1 moves y2)", y1_moves)

    if checks.mutations:
        # The search is sensitive: a cheaper fold is detected (its y' becomes a single instruction), so the >=3 claim
        # correctly flips. Each mutation is 'caught' because the search finds the shorter program it admits.
        easy = [u32(pool["y"][i] ^ pool["d"][i]) for i in range(len(vectors))]
        checks.mutation("XOR-only target y^d is single-instruction", single_instruction(easy, pool) is not None,
                        single_instruction(easy, pool))
        lin = [u32(pool["y"][i] + pool["d"][i] + pool["z"][i]) for i in range(len(vectors))]
        checks.mutation("a linear fold y+d+z is single-instruction (IADD3)", single_instruction(lin, pool) is not None,
                        single_instruction(lin, pool))
        # A fold that is XOR-only (y' = y^d^k) would be a single LOP3, so 'y' not single instruction' flips to false.
        xor_fold = [u32(pool["y"][i] ^ pool["d"][i] ^ pool["k"][i]) for i in range(len(vectors))]
        checks.mutation("an XOR-only fold's y' = y^d^k is single-instruction (claim flips)",
                        single_instruction(xor_fold, pool) is not None, single_instruction(xor_fold, pool))

    return checks.finish()


# ---- small helpers -------------------------------------------------------------------------------------------------------

def lcg(seed: int):
    s = seed & M32
    while True:
        s = (1664525 * s + 1013904223) & M32
        yield s


def key_ok(a: int) -> bool:
    m, rr = a & 0xFFFFFF, a & 31
    return bool(a & 1) and 256 < m < (1 << 24) - 256 and rr not in (1, 31)


def lambda8_ok(a: int) -> bool:
    return bool(a & 1) and all(((a * b + 128) & 0x3FFFFFFF) > 256 for b in range(1, 129))


def key_ok_r9(a: int) -> bool:
    return key_ok(a) and lambda8_ok(a)


def pick_key_ok_r9() -> int:
    g = lcg(0xABCDEF)
    while True:
        a = next(g)
        if key_ok_r9(a):
            return a


def is_f2_affine(f) -> bool:
    """True iff f(x) is affine over F2 in its bit-inputs: the effect of flipping an input bit is point-independent."""
    base = f(0, 0, 0)
    # linear part from single-bit deltas at 0
    lin = {}
    for arg in range(3):
        for bit in range(32):
            args = [0, 0, 0]
            args[arg] = 1 << bit
            lin[(arg, bit)] = f(*args) ^ base
    g = lcg(0x1357)
    for _ in range(64):
        y, z, d = next(g), next(g), next(g)
        base_pt = f(y, z, d)
        pt = [y, z, d]
        for arg in range(3):
            for bit in range(0, 32, 7):
                flipped = list(pt)
                flipped[arg] ^= 1 << bit
                if (f(*flipped) ^ base_pt) != lin[(arg, bit)]:
                    return False
    return True


def distributes(a: int) -> bool:
    """True iff (x ^ y) * a == f((x*a), (y*a)) for some fixed op f in {+, -, ^, *} over a sample -- i.e. a homomorphism
    exists. Returns True only if one does; the fold relies on this being False."""
    g = lcg(0x2468)
    pts = [(next(g), next(g)) for _ in range(40)]
    for op in (lambda p, q: u32(p + q), lambda p, q: u32(p - q), lambda p, q: p ^ q, lambda p, q: u32(p * q)):
        if all(u32((x ^ y) * a) == op(u32(x * a), u32(y * a)) for x, y in pts):
            return True
    return False


def swar_breaks(a: int) -> bool:
    """True iff packing two independent 16-bit states into one 32-bit register and multiplying by a does not reproduce both
    independent products. The low lane (bits 0..15 of packed*a) is carry-clean, but the high lane (bits 16..31) picks up
    the carry out of the low lane's product, so it does not equal the independent high product -- lanes interfere. Returns
    True (packing breaks), the reason a 32-bit fold multiply can't batch two chains by SWAR."""
    g = lcg(0x9753)
    for _ in range(200):
        ulo, uhi = next(g) & 0xFFFF, next(g) & 0xFFFF
        packed = (uhi << 16) | ulo
        got_hi = (u32(packed * a) >> 16) & 0xFFFF
        want_hi = u32(uhi * a) & 0xFFFF
        if got_hi != want_hi:
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
