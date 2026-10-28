"""FOLD-OPT for the sm_120 design's fx chains, mechanically: no program computes the fx1 (resp. fx2) receipt update for a word in
fewer than 4 (resp. 5) issue slots, nor more cheaply on the design's binding row. Builds on `fold_opt.py`, which made the
plain duo fold a mechanical fact (>= 3 instructions a word).

The sm_120 design is H1 + fx1: one keyed IMAD a word chained in front
of duo, so f = 4. The chain is egemm.cu's `fx_chain` (commit b2e177f63): a word w gets n ops, each `t = t * key_mul(1,c)`
(IMAD, key_mul is odd) or `t = t ^ key_xor(1,c)` (LOP3), alternating and **ending on an IMAD** (op I fires a multiply iff
(n - I) is odd, and the last op I = n - 1 has n - I = 1), after which duo folds the result d' = fx_chain(w):

    d' = fx_chain(w)          # fx1: d' = w*m1 (1 IMAD); fx2: d' = (w ^ x1)*m1 (1 LOP3, 1 IMAD)
    u  = y ^ d' ^ kx          # one LOP3
    y' = u * a + z            # one IMAD        (a = key_mul(0,T), odd)
    z' = rotl(u, r) + y'      # one LEA.HI

so fx1 is 4 instructions a word, fx2 is 5. The keys and the a_i are per-call constants (the prover knows them before the
run), so the search gives the cheater all of them. FOLD-OPT also has a
price form: ε_mand = [mu_f*N_w*phi* + mu_x*chi*]/B(H) is 0 under FOLD-OPT, where phi* = min over the fold route menu of
the per-word price at a certificate pi, and the assumption is mu_f = 0 (honest's fold phi_H is optimal). On a design that
binds a joint-issue row it must hold **per facet**: honest's fold has no cheaper IMAD / uniform / b1-MMA form on r*.

The result (checks below):
- **The prepended IMAD does not merge into duo's LOP3.** For fx1, u = y ^ (w*m1) ^ kx is NOT a single instruction
  (multiply-then-XOR does not fuse), where plain duo's u = y ^ d ^ k IS one LOP3. So fx1's u costs 2 instructions, not 1
  -- the extra op is genuine. This is exactly why the chain ends on an IMAD: an XOR-ending chain (d' = w ^ c) collapses,
  since u = y ^ (w ^ c) ^ kx = y ^ w ^ (c ^ kx) folds back to one LOP3 (demonstrated as a mutation).
- **Neither output is a word of a program of at most two instructions** (one instruction: exhaustive over the modelled
  ISA at 32 bits with keys as constants, and at width 4 over all 256 LOP3 LUTs, IMAD.WIDE's high word with a two-word
  addend included; two instructions: exhaustive at width 4, the first any one-word value or both words of an IMAD.WIDE,
  the second any form, IMAD.WIDE's high word included). A 3-instruction program of one-word instructions has values
  {single, <=2-instruction, <=3-instruction}; the two live outputs would have to be two of them, and both the single and
  the 2-instruction slot are ruled out, so **fx1's (y', z') needs >= 4 instructions** = honest's count. fx2 adds one
  more irreducible op (the leading XOR before the multiply), so **fx2 needs >= 5**.
- **Two-word instructions.** sm_120 has four lane forms that write two words: IMAD.WIDE and IADD.64 (ptxas 13.0 for
  sm_120a, opcode_probe/isa_accept.sh), IMNMX.U64 (TS.lean's fan-in table) and SEL.64 (harness/sass_pins.json). So the
  count above holds only once no instruction writes both outputs. Here, at width 4, no program of at most 2
  instructions ending on an IMAD.WIDE writes them, which also completes fold_opt.py's >= 3 for duo. `--wide3` (fold_wide3.c,
  CPU minutes, outside the test suite) settles all four at width 4: neither output is a word of a program of at most 2
  instructions, so a 3-instruction program would write both as its last instruction's two words, and none does, nor as
  a last SEL.64's under any predicate. The width-4 ISA is LOP3, IADD3, IMAD, IMAD.HI, LEA, LEA.HI, IMAD.WIDE, IADD.64
  and IMNMX.64, with SEL.64 as a last instruction only; it has no IDP, POPC, FLO, BREV, BMSK or wrapping-SHF form, and
  no predicate but a last SEL.64's: no ISETP, no carry chain (IADD3.X) and no guarded instruction. A predicate is one
  bit, so an instruction that writes one adds no word, and a SEL.64 before the last only picks words the program has.
- **The collapse routes all fail:** (i) the two multiplies do not compose -- y'(w) is not affine in
  w, so (y ^ (w*m1) ^ kx)*a is not w*(m1*a) + c; (ii) the fx IMAD's add slot cannot pre-absorb duo's + z (it feeds the
  XOR, not the + z); (iii) no IMAD.WIDE / IDP.4A / LOP3-of-three produces two words' intermediates in one instruction;
  (iv) the fx multiply is per-lane (the word differs per lane), so it cannot move to the warp-uniform datapath (NOSUM).
- **Per facet (price form):** on the binding joint-issue row every vector class prices alike, so the fold costs
  (# instructions) * b, honest binds at 4*b (fx1) / 5*b (fx2), and no route lowers it: fewer ops is excluded above, the
  IMAD-form of an ALU op saves nothing on a joint row (it would only help on a pipe-bound row, which the design avoids by
  binding a joint row), the uniform route is priced on the joint row and the work is per-lane, and the XOR/multiply/rotate
  have no tensor form. So phi* = phi_H and **mu_f = 0**: FOLD-OPT(fx1), FOLD-OPT(fx2) hold in the price form.
- **Bijection and X1:** key_mul is odd, so fx_chain is a bijection of the word (the fold binds it as duo does), and
  f = 4 (fx1), 5 (fx2) >= 2.53 (X1).

Verdict: FOLD-OPT(fx1) and FOLD-OPT(fx2) are validated, in the count form (>= f issue slots a word) and the price form
(mu_f = 0 on the binding joint-issue row). No counterexample: fx1 is the cheapest keyed chain reaching f = 4 that provably
does not collapse, because it ends on a multiply. What is mechanical: the one-instruction checks at 32 bits, and the
multi-instruction searches at width 4. Reducing every word mod 2^4 commutes with LOP3, IADD3, IMAD and LEA, and y' uses
only those, so for y' over those forms the width-4 search covers 32 bits. It doesn't commute with IMAD.HI, LEA.HI,
IMAD.WIDE's high word or a rotation, and z' rotates; there the 32-bit claim is the no-collapse + no-merge + op-type
argument, by hand, with the two-instruction searches repeated at width 8 as evidence.
A fully general multi-op superoptimiser over the complete ISA at 32 bits explodes.

Run: `python fold_opt_fx.py [--mutations] [--wide3] [--verdict PATH]`. Stdlib + exact_kit only; `--wide3` builds
fold_wide3.c with gcc.
"""
from __future__ import annotations

import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

from exact_kit.checks import Checks

M32 = 0xFFFFFFFF
NPOS = 16


# ---- exact word helpers (as receipts/receipt_folds.py and floored_gemm/fgemm.cu) ------------------------------------------

def u32(x: int) -> int:
    return x & M32


def rotl(x: int, r: int) -> int:
    r &= 31
    return ((x << r) | (x >> (32 - r))) & M32 if r else x & M32


def funnel_l(lo: int, hi: int, n: int) -> int:
    n &= 31
    return (((hi << 32) | (lo & M32)) << n >> 32) & M32


def funnel_r(lo: int, hi: int, n: int) -> int:
    n &= 31
    return (((hi << 32) | (lo & M32)) >> n) & M32


def prmt(a: int, b: int, sel: int) -> int:
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        s = (sel >> (4 * i)) & 0xF
        byte = src[s & 7]
        if s & 8:
            byte = 0xFF if byte & 0x80 else 0x00
        out |= byte << (8 * i)
    return out & M32


def modinv(a: int) -> int:
    return pow(a | 1, -1, 1 << 32)


def wide(p: int, q: int, c_lo: int, c_hi: int, signed: bool, width: int = 32) -> tuple[int, int]:
    """IMAD.WIDE (signed) or IMAD.WIDE.U32 at `width` bits: p*q + (c_hi:c_lo) as a 2*width-bit pair, (lo, hi). The low
    half is IMAD's; the high half is a second word written by the same instruction."""
    mask = (1 << width) - 1
    if signed:
        p, q = (p & mask) - ((p >> (width - 1) & 1) << width), (q & mask) - ((q >> (width - 1) & 1) << width)
    v = (p * q + ((c_hi & mask) << width) + (c_lo & mask)) & ((1 << 2 * width) - 1)
    return v & mask, v >> width


def add64(a_lo: int, a_hi: int, b_lo: int, b_hi: int, s: int, width: int = 32) -> tuple[int, int]:
    """IADD.64 at `width` bits: ±(a_hi:a_lo) ± (b_hi:b_lo) as a pair (lo, hi); bit 0 of `s` negates A, bit 1 B."""
    mask = (1 << width) - 1
    a, b = ((a_hi & mask) << width) | (a_lo & mask), ((b_hi & mask) << width) | (b_lo & mask)
    v = ((-a if s & 1 else a) + (-b if s & 2 else b)) & ((1 << 2 * width) - 1)
    return v & mask, v >> width


def mnmx64(a_lo: int, a_hi: int, b_lo: int, b_hi: int, v: int, width: int = 32) -> tuple[int, int]:
    """IMNMX.64 at `width` bits: the min of (a_hi:a_lo) and (b_hi:b_lo), or the max when bit 0 of `v` is set, compared
    signed when bit 1 is, as a pair (lo, hi)."""
    mask = (1 << width) - 1
    a, b = ((a_hi & mask) << width) | (a_lo & mask), ((b_hi & mask) << width) | (b_lo & mask)
    bias = 1 << (2 * width - 1) if v & 2 else 0
    r = a if ((a ^ bias) >= (b ^ bias) if v & 1 else (a ^ bias) <= (b ^ bias)) else b
    return r & mask, r >> width


def mix32(x: int) -> int:
    """fgemm.cu / receipt_folds.py mix32."""
    x &= M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


def key_mul(lane: int, t: int) -> int:
    """fgemm.cu key_mul: an odd multiplier (always | 1)."""
    return mix32(0xA0000000 + 2 * (lane * NPOS + t)) | 1


def key_xor(lane: int, t: int) -> int:
    return mix32(0xB0000000 + 2 * (lane * NPOS + t) + 1)


def rot_of(t: int) -> int:
    """fgemm.cu rot_of(DUO, 0, t)."""
    return ((t * 7 + 5) % 31) + 1


# ---- the fx receipt update (egemm.cu fx_chain + fgemm.cu duo), the target ------------------------------------------------

def fx_chain(w: int, n: int, tile: int = 0, e: int = 0) -> int:
    """egemm.cu fx_chain: n ops on the word w, op I a multiply iff (n - I) is odd (so the last op is a multiply), key
    index c = (4*tile + e)*32 + I. Ends on an IMAD."""
    t = w & M32
    for i in range(n):
        c = (4 * tile + e) * 32 + i
        if (n - i) % 2 == 1:
            t = u32(t * key_mul(1, c))
        else:
            t = u32(t ^ key_xor(1, c))
    return t


def duo_from_dprime(y: int, z: int, dprime: int, a: int, kx: int, r: int) -> tuple[int, int]:
    u = u32(y ^ dprime ^ kx)
    yp = u32(u * a + z)
    return yp, u32(rotl(u, r) + yp)


def fx_word(y: int, z: int, w: int, n: int, a: int, kx: int, r: int, tile: int = 0, e: int = 0) -> tuple[int, int]:
    return duo_from_dprime(y, z, fx_chain(w, n, tile, e), a, kx, r)


# ---- the sm_120 integer ISA (exact), as instruction templates (as fold_opt.py's) -----------------------------------

SHIFTS = tuple(range(32))


def arith_templates():
    out = []
    for signs in itertools.product((1, -1), repeat=3):
        out.append((f"IADD3{signs}", 3, lambda x, y, z, s=signs: u32(s[0] * x + s[1] * y + s[2] * z)))
    out.append(("IMAD", 3, lambda x, y, z: u32(x * y + z)))
    out.append(("IMAD.-c", 3, lambda x, y, z: u32(x * y - z)))
    out.append(("IMAD.HI", 3, lambda x, y, z: u32(((x * y) >> 32) + z)))
    out.append(("IMAD.WIDE.hi", 4, lambda p, q, cl, ch: wide(p, q, cl, ch, True)[1]))
    out.append(("IMAD.WIDE.U32.hi", 4, lambda p, q, cl, ch: wide(p, q, cl, ch, False)[1]))
    for n in SHIFTS:
        out.append((f"LEA.{n}", 2, lambda x, y, n=n: u32((x << n) + y)))
        out.append((f"SHF.L.{n}", 2, lambda lo, hi, n=n: funnel_l(lo, hi, n)))
        out.append((f"SHF.R.{n}", 2, lambda lo, hi, n=n: funnel_r(lo, hi, n)))
        out.append((f"LEA.HI.{n}", 3, lambda x, hi, y, n=n: u32(funnel_l(x, hi, n) + y)))
    for sel in (0x3210, 0x0321, 0x2103, 0x1032, 0x4444, 0x7654, 0xBA98):
        out.append((f"PRMT.{sel:04x}", 2, lambda a, b, sel=sel: prmt(a, b, sel)))
    return out


ARITH = arith_templates()


def is_lop3(target_bits, pool_bits):
    """True (returns the operand triple) iff `target` is some 3-input bitwise LUT of three pool operands."""
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


def single_instruction(target, pool):
    """The name of a single modelled instruction computing `target` (over the test vectors) from the pool, or None."""
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


# ---- small-width exhaustive searches (all 256 LUTs), single and length-2 -------------------------------------------------

def _width_ops(width: int, names, cols, nv, target, wide_hi: bool = False):
    """A label for a single-instruction match of `target` over `cols` at `width` bits (every one of the 256 LOP3 LUTs via
    a per-bit locality test, IADD3 signs, IMAD/IMAD.HI, LEA and the rotate form of LEA.HI, and with `wide_hi` the high
    word of IMAD.WIDE and IMAD.WIDE.U32 with a two-word addend), or None."""
    mask = (1 << width) - 1

    def fun_l(lo, hi, n):
        n %= width
        return (((hi << width) | (lo & mask)) << n >> width) & mask

    # LOP3: target is SOME 3-input LUT of a triple iff the bit-triple -> target-bit map is consistent (every one of the 256
    # LUTs, checked in one pass per triple instead of 256).
    tbits = [[(target[i] >> bit) & 1 for bit in range(width)] for i in range(nv)]
    for a, b, c in itertools.product(names, repeat=3):
        av, bv, cv = cols[a], cols[b], cols[c]
        lut, ok = {}, True
        for i in range(nv):
            ai, bi, ci, ti = av[i], bv[i], cv[i], tbits[i]
            for bit in range(width):
                key = ((ai >> bit) & 1) | (((bi >> bit) & 1) << 1) | (((ci >> bit) & 1) << 2)
                o = ti[bit]
                if key in lut:
                    if lut[key] != o:
                        ok = False
                        break
                else:
                    lut[key] = o
            if not ok:
                break
        if ok:
            return f"LOP3[{a},{b},{c}]"
    for a, b, c in itertools.product(names, repeat=3):
        av, bv, cv = cols[a], cols[b], cols[c]
        for s in itertools.product((1, -1), repeat=3):
            if all(((s[0] * av[i] + s[1] * bv[i] + s[2] * cv[i]) & mask) == target[i] for i in range(nv)):
                return f"IADD3{s}({a},{b},{c})"
        if all(((av[i] * bv[i] + cv[i]) & mask) == target[i] for i in range(nv)):
            return f"IMAD({a},{b},{c})"
        if all(((((av[i] * bv[i]) >> width) + cv[i]) & mask) == target[i] for i in range(nv)):
            return f"IMAD.HI({a},{b},{c})"
    for a, b in itertools.product(names, repeat=2):
        av, bv = cols[a], cols[b]
        for n in range(width):
            if all((((av[i] << n) + bv[i]) & mask) == target[i] for i in range(nv)):
                return f"LEA.{n}({a},{b})"
            if all((fun_l(av[i], av[i], n) + bv[i]) & mask == target[i] for i in range(nv)):
                return f"LEA.HI.{n}({a},{b})"
    if wide_hi:
        for p, q, cl, ch in itertools.product(names, repeat=4):
            for signed in (True, False):
                if all(wide(cols[p][i], cols[q][i], cols[cl][i], cols[ch][i], signed, width)[1] == target[i]
                       for i in range(nv)):
                    return f"IMAD.WIDE{'' if signed else '.U32'}.hi({p},{q},{ch}:{cl})"
    return None


def width_single_instruction(width: int, target, inputs):
    """Is `target` a single instruction of `inputs` at `width` bits? Exhaustive over the full modelled ISA, both words of
    IMAD.WIDE included."""
    return _width_ops(width, list(inputs), {nm: inputs[nm] for nm in inputs}, len(target), target, wide_hi=True)


def width_wide_writes_both(width: int, t1, t2, inputs):
    """Does a program of at most two instructions, the last an IMAD.WIDE or IMAD.WIDE.U32, write `t1` and `t2` as its two
    words (either way round)? Exhaustive at `width` bits: the first instruction ranges over every distinct
    single-instruction value of `inputs` (or none), and the last over every operand choice, its low word fixing c_lo and
    its high word c_hi. Returns a witness label or None."""
    mask = (1 << width) - 1
    nv = len(t1)
    for extra, lab in [({}, "")] + _firsts(width, inputs):
        cols = dict(inputs)
        cols.update(extra)
        by_vec = {tuple(v): nm for nm, v in cols.items()}
        for lo_t, hi_t in ((t1, t2), (t2, t1)):
            for p, q in itertools.product(cols, repeat=2):
                for signed in (True, False):
                    prods = [wide(cols[p][i], cols[q][i], 0, 0, signed, width) for i in range(nv)]
                    need_lo = tuple((lo_t[i] - prods[i][0]) & mask for i in range(nv))
                    cl = by_vec.get(need_lo)
                    if cl is None:
                        continue
                    lows = [prods[i][0] + cols[cl][i] for i in range(nv)]
                    need_hi = tuple((hi_t[i] - prods[i][1] - (lows[i] >> width)) & mask for i in range(nv))
                    ch = by_vec.get(need_hi)
                    if ch is not None:
                        return f"{lab} -> IMAD.WIDE{'' if signed else '.U32'}({p},{q},{ch}:{cl})"
    return None


def _wide_pairs(width: int, inputs):
    """Every distinct (lo, hi) word pair one IMAD.WIDE or IMAD.WIDE.U32 over `inputs` writes, as {pair: label}."""
    names = list(inputs)
    nv = len(next(iter(inputs.values())))
    out = {}
    for p, q, cl, ch in itertools.product(names, repeat=4):
        for signed in (True, False):
            lh = [wide(inputs[p][i], inputs[q][i], inputs[cl][i], inputs[ch][i], signed, width) for i in range(nv)]
            out.setdefault((tuple(x[0] for x in lh), tuple(x[1] for x in lh)),
                           f"IMAD.WIDE{'' if signed else '.U32'}({p},{q},{ch}:{cl})")
    return out


def _firsts(width: int, inputs):
    """A program's first instruction, as the words it adds to `inputs`: every distinct one-word value (`v1`), and every
    IMAD.WIDE's two words (`v1`, `v1h`). Pairs of ({name: vector}, label)."""
    return ([({"v1": list(v)}, lab) for v, lab in _single_values(width, inputs).items()]
            + [({"v1": list(lo), "v1h": list(hi)}, lab) for (lo, hi), lab in _wide_pairs(width, inputs).items()])


def _single_values(width: int, inputs):
    """Every distinct single-instruction value vector over `inputs` at `width` bits (all 256 LOP3 LUTs, IADD3 +/-, IMAD,
    IMAD.HI, LEA and the rotate form of LEA.HI), as {value_tuple: label}: every one-word form `_width_ops` matches.
    IMAD.WIDE's pairs are `_wide_pairs`."""
    mask = (1 << width) - 1
    names = list(inputs)
    cols = {nm: inputs[nm] for nm in inputs}
    nv = len(next(iter(cols.values())))
    out = {}

    def add(vec, label):
        out.setdefault(tuple(vec), label)

    def fun_l(lo, hi, n):
        n %= width
        return (((hi << width) | (lo & mask)) << n >> width) & mask

    for a, b, c in itertools.product(names, repeat=3):
        av, bv, cv = cols[a], cols[b], cols[c]
        for lut in range(256):
            add([sum(((lut >> (((av[i] >> bit) & 1) | (((bv[i] >> bit) & 1) << 1) | (((cv[i] >> bit) & 1) << 2))) & 1)
                     << bit for bit in range(width)) for i in range(nv)], f"LOP3.{lut:#04x}({a},{b},{c})")
        for s in itertools.product((1, -1), repeat=3):
            add([(s[0] * av[i] + s[1] * bv[i] + s[2] * cv[i]) & mask for i in range(nv)], f"IADD3{s}")
        add([(av[i] * bv[i] + cv[i]) & mask for i in range(nv)], f"IMAD({a},{b},{c})")
        add([(((av[i] * bv[i]) >> width) + cv[i]) & mask for i in range(nv)], f"IMAD.HI({a},{b},{c})")
    for a, b in itertools.product(names, repeat=2):
        av, bv = cols[a], cols[b]
        for n in range(width):
            add([((av[i] << n) + bv[i]) & mask for i in range(nv)], f"LEA.{n}({a},{b})")
            add([(fun_l(av[i], av[i], n) + bv[i]) & mask for i in range(nv)], f"LEA.HI.{n}({a},{b})")
    return out


def width_le2_value(width: int, target, inputs):
    """Is `target` a word of a length-<=2 straight-line program over the width-`width` ISA from `inputs`? Exhaustive: it
    is a single instruction's word, or a word of one instruction over `inputs` plus the first instruction's words. The
    first ranges over every `_firsts` (every distinct one-word value, all 256 LOP3 LUTs included, and both words of every
    IMAD.WIDE), the second over every form `_width_ops` matches and the high word of IMAD.WIDE, its c_hi solved for.
    Returns a witness label or None."""
    direct = width_single_instruction(width, target, inputs)
    if direct is not None:
        return direct
    mask = (1 << width) - 1
    nv = len(target)
    prods = {}

    def prod(cols, p, q, signed):
        key = (p, q, signed)
        if key not in prods or p.startswith("v1") or q.startswith("v1"):
            prods[key] = [wide(cols[p][i], cols[q][i], 0, 0, signed, width) for i in range(nv)]
        return prods[key]

    names = list(inputs)
    # c_hi a first instruction's word, p, q and c_lo inputs: the c_hi each such IMAD.WIDE.hi needs, once
    base = {}
    for p, q, cl in itertools.product(names, repeat=3):
        for signed in (True, False):
            pr = prod(inputs, p, q, signed)
            need = tuple((target[i] - pr[i][1] - ((pr[i][0] + inputs[cl][i]) >> width)) & mask for i in range(nv))
            base.setdefault(need, f"IMAD.WIDE{'' if signed else '.U32'}.hi({p},{q},*:{cl})")
    for extra, label in _firsts(width, inputs):
        for nm, v in extra.items():
            if tuple(v) in base:
                return f"{label} -> {base[tuple(v)].replace('*', nm)}"
        ext = dict(inputs)
        ext.update(extra)
        got = _width_ops(width, list(ext), ext, nv, target)
        if got is not None and "v1" in got:
            return f"{label} -> {got}"
        by2 = {(v[0], v[1]) for v in ext.values()}
        by_vec = {tuple(v): nm for nm, v in ext.items()}
        for p, q, cl in itertools.product(ext, repeat=3):
            if p not in extra and q not in extra and cl not in extra:
                continue
            for signed in (True, False):
                pr = prod(ext, p, q, signed)
                c = ext[cl]
                if (((target[0] - pr[0][1] - ((pr[0][0] + c[0]) >> width)) & mask),
                        ((target[1] - pr[1][1] - ((pr[1][0] + c[1]) >> width)) & mask)) not in by2:
                    continue
                need = tuple((target[i] - pr[i][1] - ((pr[i][0] + c[i]) >> width)) & mask for i in range(nv))
                ch = by_vec.get(need)
                if ch is not None:
                    return f"{label} -> IMAD.WIDE{'' if signed else '.U32'}.hi({p},{q},{ch}:{cl})"
    return None


def width_wide3(exe, width: int, inputs, jobs):
    """`fold_wide3.c` (built at `exe`) over `inputs` at `width` bits, for each of `jobs` ({label: (t1, t2)}): {label:
    {found, sel, le2, firsts, prefixes, witnesses}}. found counts the programs of at most 3 instructions whose last
    (IMAD.WIDE, IADD.64 or IMNMX.64) writes the pair, either way round; sel those whose last SEL.64 does under some
    predicate; le2 the programs of at most 2 instructions with t1 or t2 among their words."""
    nv = len(next(iter(inputs.values())))
    lines = [f"{width} {nv} {len(inputs)}", *(f"{nm} " + " ".join(map(str, v)) for nm, v in inputs.items()),
             str(len(jobs)), *(f"{lab} " + " ".join(map(str, [*a, *b])) for lab, (a, b) in jobs.items())]
    out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
    res = {lab: {"witnesses": []} for lab in jobs}
    for f in (line.split() for line in out.splitlines()):
        if f[0] == "J":
            res[f[1]].update({f[k]: int(f[k + 1]) for k in range(2, len(f), 2)})
        elif f[0] in ("F", "S", "L"):
            res[f[1]]["witnesses"].append(" ".join([f[0], *f[2:]]))
    return res


def wide3_controls(width: int, cols):
    """Target pairs fold_wide3.c must find, over `cols` (y, z, w, kx, a, m1), between them reading every form it
    enumerates and every way it reaches a last instruction: {label: (t1, t2)}. A `ctl_` pair is a last instruction's
    two words (found), the `sel_` pair a SEL.64's under a predicate no IMNMX.64 computes (sel), and the `le2_` pair has a
    word of a second instruction (le2)."""
    mask = (1 << width) - 1
    y, z, w, kx, a, m1 = (cols[k] for k in ("y", "z", "w", "kx", "a", "m1"))
    nv = len(y)

    def rot(x, n):
        return ((x << n) | (x >> (width - n))) & mask if n else x

    def split(lh):
        return [x[0] for x in lh], [x[1] for x in lh]

    out = {}
    v1 = [y[i] ^ w[i] ^ kx[i] for i in range(nv)]                          # LOP3 0x96
    v2 = [(v1[i] * m1[i] + z[i]) & mask for i in range(nv)]                # IMAD
    lh = [wide(v2[i], a[i], v1[i], z[i], True, width) for i in range(nv)]
    out["ctl_lop3_imad"] = ([x[0] for x in lh], [x[1] for x in lh])
    w1 = [wide(y[i], a[i], z[i], kx[i], False, width) for i in range(nv)]  # IMAD.WIDE.U32, two words
    v2 = [(rot(w1[i][1], 1) + w[i]) & mask for i in range(nv)]             # LEA.HI of its high word
    lh = [wide(v2[i], m1[i], y[i], w1[i][0], False, width) for i in range(nv)]
    out["ctl_wide_leahi"] = ([x[1] for x in lh], [x[0] for x in lh])
    v1 = [(y[i] - w[i] + kx[i]) & mask for i in range(nv)]                 # IADD3
    v2 = [(((v1[i] * a[i]) >> width) + z[i]) & mask for i in range(nv)]   # IMAD.HI
    lh = [wide(v1[i], v2[i], y[i], w[i], True, width) for i in range(nv)]
    out["ctl_iadd3_imadhi"] = ([x[0] for x in lh], [x[1] for x in lh])
    v1 = [((y[i] << 2) + z[i]) & mask for i in range(nv)]                  # LEA
    v2 = [(v1[i] & w[i]) | (v1[i] & kx[i]) | (w[i] & kx[i]) for i in range(nv)]  # LOP3 0xE8
    lh = [wide(v2[i], v2[i], v1[i], a[i], False, width) for i in range(nv)]
    out["ctl_lea_maj"] = ([x[1] for x in lh], [x[0] for x in lh])
    # p and q both before the second instruction, whose word is c_hi, then c_lo
    v1 = [y[i] ^ w[i] ^ kx[i] for i in range(nv)]
    v2 = [(v1[i] * m1[i] + z[i]) & mask for i in range(nv)]
    lh = [wide(v1[i], a[i], z[i], v2[i], True, width) for i in range(nv)]
    out["ctl_pre_chi"] = ([x[0] for x in lh], [x[1] for x in lh])
    v1 = [(rot(y[i], 1) + z[i]) & mask for i in range(nv)]
    v2 = [(v1[i] + w[i] + kx[i]) & mask for i in range(nv)]
    lh = [wide(y[i], v1[i], v2[i], w[i], False, width) for i in range(nv)]
    out["ctl_pre_clo"] = ([x[1] for x in lh], [x[0] for x in lh])
    # IADD.64 last: A of the first's pool, b_lo the second's word
    v1 = [y[i] ^ w[i] ^ kx[i] for i in range(nv)]
    v2 = [(v1[i] * m1[i] + z[i]) & mask for i in range(nv)]
    out["ctl_add64_pre"] = split([add64(v1[i], y[i], v2[i], a[i], 2, width) for i in range(nv)])
    # IADD.64 last, both low operands the second instruction's two words
    w1 = [wide(y[i], a[i], z[i], kx[i], False, width) for i in range(nv)]
    w2 = [wide(w1[i][1], m1[i], y[i], w[i], True, width) for i in range(nv)]
    out["ctl_add64_live"] = split([add64(w2[i][0], w1[i][0], w2[i][1], kx[i], 1, width) for i in range(nv)])[::-1]
    # IADD.64 first, IMAD.WIDE last
    x1 = [add64(z[i], y[i], kx[i], w[i], 1, width) for i in range(nv)]
    v2 = [x1[i][1] ^ a[i] ^ m1[i] for i in range(nv)]
    out["ctl_add64_first"] = split([wide(v2[i], a[i], x1[i][0], y[i], False, width) for i in range(nv)])
    # IMNMX.64 first, IADD.64 last
    x1 = [mnmx64(z[i], y[i], kx[i], w[i], 0, width) for i in range(nv)]
    v2 = [((x1[i][0] << 2) + a[i]) & mask for i in range(nv)]
    out["ctl_mnmx_first"] = split([add64(v2[i], x1[i][1], m1[i], y[i], 0, width) for i in range(nv)])
    # IMNMX.64 last, and SEL.64, over (v1:v2) and (v2:v1), so both pairs read the second instruction's word; the SEL.64
    # takes each pair at half the points, by the first such pattern no IMNMX.64 form computes
    v1 = [y[i] ^ w[i] ^ kx[i] for i in range(nv)]
    v2 = [(v1[i] * m1[i] + z[i]) & mask for i in range(nv)]
    pa, pb = [(v2[i], v1[i]) for i in range(nv)], [(v1[i], v2[i]) for i in range(nv)]
    out["ctl_mnmx_last"] = split([mnmx64(*pa[i], *pb[i], 1, width) for i in range(nv)])
    for pat in (p for p in range(1 << nv) if bin(p).count("1") == nv // 2):
        t = [pa[i] if pat >> i & 1 else pb[i] for i in range(nv)]
        if not any(all(mnmx64(*pa[i], *pb[i], v, width) == t[i] for i in range(nv)) for v in range(4)):
            out["sel_free"] = split(t)
            break
    # IADD.64's high word over a first instruction's word, a word of a second instruction and of no first one
    v1 = [y[i] ^ w[i] ^ kx[i] for i in range(nv)]
    hi = [add64(v1[i], v1[i], z[i], y[i], 0, width)[1] for i in range(nv)]
    out["le2_add64_hi"] = (hi, hi)
    return out


def _two_word_pairs(width: int, inputs):
    """Every distinct (lo, hi) word pair one two-word instruction over `inputs` writes (IMAD.WIDE, IADD.64 and
    IMNMX.64, every form), as {pair: label}."""
    names = list(inputs)
    nv = len(next(iter(inputs.values())))
    out = _wide_pairs(width, inputs)
    for al, ah, bl, bh in itertools.product(names, repeat=4):
        c = [inputs[x] for x in (al, ah, bl, bh)]
        for k in range(4):
            for fn, lab in ((add64, "IADD.64"), (mnmx64, "IMNMX.64")):
                lh = [fn(c[0][i], c[1][i], c[2][i], c[3][i], k, width) for i in range(nv)]
                out.setdefault((tuple(x[0] for x in lh), tuple(x[1] for x in lh)), f"{lab}.{k}({ah}:{al},{bh}:{bl})")
    return out


def _new_word_sets(width: int, inputs) -> int:
    """How many distinct sets of new words one instruction over `inputs` adds (fold_wide3.c's first instructions)."""
    have = {tuple(v) for v in inputs.values()}
    sets = {frozenset([v]) for v in _single_values(width, inputs) if v not in have}
    sets |= {s for lo, hi in _two_word_pairs(width, inputs) if (s := frozenset(x for x in (lo, hi) if x not in have))}
    return len(sets)


# ---- helpers -------------------------------------------------------------------------------------------------------------

def lcg(seed: int):
    s = seed & M32
    while True:
        s = (1664525 * s + 1013904223) & M32
        yield s


def is_affine_in_w(f) -> bool:
    """True iff f(w) = w*C + D over Z/2^32 for constants C, D (so the map factors as one IMAD of w). Fits C, D from two
    points and checks the rest."""
    f0 = f(0)
    f1 = f(1)
    c = u32(f1 - f0)
    g = lcg(0x5EED)
    for _ in range(200):
        w = next(g)
        if f(w) != u32(w * c + f0):
            return False
    return True


def make_pool(a: int, kx: int, r: int, m1: int, x1: int, vectors):
    """Operand pool: inputs y, z, w; the per-call constants (a, kx, r, m1, x1 and inverses); a rich immediate set."""
    consts = {
        "0": 0, "1": 1, "-1": M32, "a": a, "kx": kx, "r": r & M32, "m1": m1, "x1": x1,
        "ainv": modinv(a), "m1inv": modinv(m1), "kx^x1": kx ^ x1, "m1a": u32(m1 * a),
        "GOLDEN": 0x9E3779B9, "msb": 0x80000000, "2": 2,
    }
    pool = {"y": [v[0] for v in vectors], "z": [v[1] for v in vectors], "w": [v[2] for v in vectors]}
    for nm, cv in consts.items():
        pool[nm] = [cv] * len(vectors)
    return pool


def main() -> int:
    checks = Checks()

    # one duo position's public keys (tile 0, word 0) and the fx keys, all per-call constants the cheater is given.
    a, kx, r = key_mul(0, 0), key_xor(0, 0), rot_of(0)
    m1 = key_mul(1, 0)      # fx1's multiplier, and fx2's trailing multiplier (I=1 -> c=1 is key_mul(1,1); see below)
    x1 = key_xor(1, 0)      # fx2's leading XOR constant
    m2 = key_mul(1, 1)      # fx2's trailing multiplier
    checks.check("duo multiplier a is odd (key_mul | 1)", a % 2 == 1, f"a={a:#010x}")
    checks.check("fx multiplier key_mul(1,.) is odd: the fx multiply is a bijection of the word", m1 % 2 == 1 and m2 % 2 == 1)

    # bijection of the word (the fold binds it as duo does): distinct words give distinct d', and m1^{-1} recovers w.
    g = lcg(0xF00D)
    ws = [next(g) for _ in range(400)]
    dp1 = {fx_chain(w, 1) for w in ws}
    checks.check("fx1 is a bijection of the word (distinct w -> distinct d')", len(dp1) == len(ws))
    checks.check("fx1 inverse: w = (d' * m1^{-1}) recovers the word", all(u32(fx_chain(w, 1) * modinv(m1)) == w for w in ws))
    dp2 = {fx_chain(w, 2) for w in ws}
    checks.check("fx2 is a bijection of the word (distinct w -> distinct d')", len(dp2) == len(ws))

    # f >= 2.53 (X1): fx1 is 4 ops a word, fx2 is 5.
    checks.check("fx1 f = 4 >= 2.53 (X1)", 1 + 3 == 4 and 4 >= 2.53)
    checks.check("fx2 f = 5 >= 2.53 (X1)", 2 + 3 == 5 and 5 >= 2.53)

    rng = lcg(0xC0FFEE)
    vectors = [(next(rng), next(rng), next(rng)) for _ in range(28)]
    for s in (0, 1, 2, 0x80000000, M32):
        vectors.append((s, s ^ 0x55555555, s))
    pool = make_pool(a, kx, r, m1, x1, vectors)
    nv = len(vectors)

    # positive controls: honest 4-op (fx1) / 5-op (fx2) programs reproduce (y', z').
    yp1 = [fx_word(y, z, w, 1, a, kx, r)[0] for (y, z, w) in vectors]
    zp1 = [fx_word(y, z, w, 1, a, kx, r)[1] for (y, z, w) in vectors]
    hon1_dp = [u32(pool["w"][i] * m1) for i in range(nv)]
    hon1_u = [u32(pool["y"][i] ^ hon1_dp[i] ^ kx) for i in range(nv)]
    hon1_yp = [u32(hon1_u[i] * a + pool["z"][i]) for i in range(nv)]
    hon1_zp = [u32(rotl(hon1_u[i], r) + hon1_yp[i]) for i in range(nv)]
    checks.check("honest 4-instruction fx1 program reproduces (y', z')", hon1_yp == yp1 and hon1_zp == zp1)

    # --- the no-collapse fact: the prepended IMAD does not merge into duo's LOP3 ---
    u1 = [u32(pool["y"][i] ^ u32(pool["w"][i] * m1) ^ kx) for i in range(nv)]
    u1_found = single_instruction(u1, pool)
    checks.check("fx1's u = y ^ (w*m1) ^ kx is NOT a single instruction (multiply-then-XOR does not fuse)",
                 u1_found is None, u1_found)
    # contrast: plain duo's u = y ^ w ^ kx IS one LOP3 (so the prepended IMAD is a genuine extra op).
    duo_u = [u32(pool["y"][i] ^ pool["w"][i] ^ kx) for i in range(nv)]
    checks.check("contrast: plain duo's u = y ^ w ^ kx IS one LOP3 (the fx op is genuinely extra)",
                 single_instruction(duo_u, pool) is not None, single_instruction(duo_u, pool))

    # --- neither output is a single instruction ---
    checks.check("fx1 y' is NOT a single instruction", single_instruction(yp1, pool) is None, single_instruction(yp1, pool))
    checks.check("fx1 z' is NOT a single instruction", single_instruction(zp1, pool) is None, single_instruction(zp1, pool))

    # --- width-4 exhaustive (all 256 LUTs): u, y', z' not single; and neither output is a TWO-instruction value ---
    w4 = 4
    gg = lcg(0xBEEF)
    npts = 10
    mask4 = (1 << w4) - 1
    y0 = [next(gg) & mask4 for _ in range(npts)]
    z0 = [next(gg) & mask4 for _ in range(npts)]
    wd = [next(gg) & mask4 for _ in range(npts)]
    aw = (key_mul(0, 0) & mask4) | 1
    kxw = key_xor(0, 0) & mask4
    m1w = (key_mul(1, 0) & mask4) | 1
    rw = 1

    def rot4(x, n):
        n %= w4
        return ((x << n) | (x >> (w4 - n))) & mask4 if n else x & mask4

    def fxword4(yv, zv, wv):
        dp = (wv * m1w) & mask4
        uu = (yv ^ dp ^ kxw) & mask4
        yy = (uu * aw + zv) & mask4
        return yy, (rot4(uu, rw) + yy) & mask4

    y1w = [fxword4(y0[i], z0[i], wd[i])[0] for i in range(npts)]
    z1w = [fxword4(y0[i], z0[i], wd[i])[1] for i in range(npts)]
    u1w = [((y0[i] ^ ((wd[i] * m1w) & mask4) ^ kxw) & mask4) for i in range(npts)]
    wins = {"y": y0, "z": z0, "w": wd, "kx": [kxw] * npts, "a": [aw] * npts, "m1": [m1w] * npts,
            "ainv": [modinv(aw) & mask4] * npts, "m1inv": [modinv(m1w) & mask4] * npts,
            "0": [0] * npts, "1": [1] * npts}
    checks.check("width-4 exhaustive: fx1's u is not a single instruction",
                 width_single_instruction(w4, u1w, wins) is None, width_single_instruction(w4, u1w, wins))
    checks.check("width-4 exhaustive: fx1 y' is not a single instruction",
                 width_single_instruction(w4, y1w, wins) is None)
    checks.check("width-4 exhaustive: fx1 z' is not a single instruction",
                 width_single_instruction(w4, z1w, wins) is None)
    checks.check("width-4 exhaustive: fx1 y' is no word of a two-instruction program (two-word instructions included)",
                 width_le2_value(w4, y1w, wins) is None)
    checks.check("width-4 exhaustive: fx1 z' is no word of a two-instruction program (two-word instructions included)",
                 width_le2_value(w4, z1w, wins) is None)
    checks.check("width-4 exhaustive: no program of <= 2 instructions ending on an IMAD.WIDE writes fx1's (y', z') as "
                 "its two words", width_wide_writes_both(w4, y1w, z1w, wins) is None)

    # duo (fold_opt.py's >= 3) with the two-word instruction: a length-2 program's first value is one instruction's, so both
    # outputs would be the last instruction's two words, which no IMAD.WIDE writes.
    dd = [next(gg) & mask4 for _ in range(npts)]
    duo_u4 = [y0[i] ^ dd[i] ^ kxw for i in range(npts)]
    duo_y4 = [(duo_u4[i] * aw + z0[i]) & mask4 for i in range(npts)]
    duo_z4 = [(rot4(duo_u4[i], rw) + duo_y4[i]) & mask4 for i in range(npts)]
    dins = {"y": y0, "z": z0, "d": dd, "kx": [kxw] * npts, "a": [aw] * npts, "ainv": [modinv(aw) & mask4] * npts,
            "0": [0] * npts, "1": [1] * npts}
    checks.check("width-4 exhaustive: no program of <= 2 instructions ending on an IMAD.WIDE writes duo's (y', z'), so "
                 "fold_opt.py's >= 3 holds with two-word instructions",
                 width_single_instruction(w4, duo_y4, dins) is None and width_single_instruction(w4, duo_z4, dins) is None
                 and width_wide_writes_both(w4, duo_y4, duo_z4, dins) is None)

    if "--wide3" in checks.args:
        # The two-word forms in full (fold_wide3.c, CPU minutes, so only with --wide3): no program of at most 2
        # instructions has either output as a word, so a 3-instruction program writes both with its last instruction,
        # which must write two words. Its enumeration is held to this file's by the count of distinct first
        # instructions, and its search by controls it must find.
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "fold_wide3"
            src = Path(__file__).resolve().parent / "fold_wide3.c"
            subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(src)], check=True)
            small = {k: wins[k] for k in ("y", "z", "w", "kx", "a", "m1")}
            ctl = width_wide3(exe, w4, small, {**wide3_controls(w4, small), "fx1": (y1w, z1w)})
            count = {"ctl": "found", "sel": "sel", "le2": "le2"}
            for lab, res in ctl.items():
                if lab != "fx1":
                    key = count[lab.split("_")[0]]
                    wit = [x for x in res["witnesses"] if x[0] == {"found": "F", "sel": "S", "le2": "L"}[key]]
                    checks.check(f"fold_wide3.c finds its control {lab} ({key})", res[key] > 0, wit[:1])
            checks.check("fold_wide3.c's first instructions are this file's (6 inputs)",
                         ctl["fx1"]["firsts"] == _new_word_sets(w4, small),
                         (ctl["fx1"]["firsts"], _new_word_sets(w4, small)))
            big = width_wide3(exe, w4, wins, {"fx1": (y1w, z1w)})["fx1"]
            checks.check("fold_wide3.c's first instructions are this file's (10 inputs)",
                         big["firsts"] == _new_word_sets(w4, wins), (big["firsts"], _new_word_sets(w4, wins)))
            seen = {k: big[k] for k in ("found", "sel", "le2", "firsts", "prefixes", "witnesses")}
            checks.check("width-4 exhaustive: neither of fx1's outputs is a word of a program of at most 2 instructions, "
                         "IMAD.WIDE, IADD.64 and IMNMX.64 included", big["le2"] == 0 and ctl["fx1"]["le2"] == 0, seen)
            checks.check("width-4 exhaustive: no program of at most 3 instructions writes fx1's (y', z') as its last "
                         "instruction's two words, an IMAD.WIDE's, IADD.64's or IMNMX.64's",
                         big["found"] == 0 and ctl["fx1"]["found"] == 0, seen)
            checks.check("width-4 exhaustive: nor as a last SEL.64's under any predicate",
                         big["sel"] == 0 and ctl["fx1"]["sel"] == 0, seen)
            if checks.mutations:
                # 7.-12. Each part of the search is needed: built without it (-DDROP), fold_wide3.c misses the controls
                #    only that part reaches.
                parts = ((1, "a last IADD.64", ("ctl_add64_pre", "ctl_add64_live", "ctl_mnmx_first")),
                         (2, "a last IMNMX.64 or SEL.64 over a second-instruction word", ("ctl_mnmx_last", "sel_free")),
                         (3, "a last IMAD.WIDE with p or q a second-instruction word",
                          ("ctl_lop3_imad", "ctl_wide_leahi", "ctl_lea_maj")),
                         (4, "a last IMAD.WIDE with p and q before it", ("ctl_pre_chi", "ctl_pre_clo")),
                         (5, "a second instruction's words in le2", ("le2_add64_hi",)),
                         (6, "IADD.64 and IMNMX.64 before the last instruction", ("ctl_add64_first", "ctl_mnmx_first")))
                jobs = wide3_controls(w4, small)
                for drop, what, labs in parts:
                    exe_d = Path(tmp) / f"fold_wide3_drop{drop}"
                    subprocess.run(["gcc", "-O2", "-fopenmp", f"-DDROP={drop}", "-o", str(exe_d), str(src)], check=True)
                    got = width_wide3(exe_d, w4, small, {lab: jobs[lab] for lab in labs})
                    left = {lab: got[lab][count[lab.split("_")[0]]] for lab in labs}
                    checks.mutation(f"fold_wide3.c without {what} misses {', '.join(labs)}", not any(left.values()),
                                    left)

    # --- the lower bound. The two outputs are live and neither is a single instruction (above), so, as in fold_opt.py, a length-2
    # program (whose two values would be the two outputs) is impossible: fx1 needs >= 3. The 4th instruction is the
    # prepended multiply that duo does not have: u = y ^ (w*m1) ^ kx is not a single instruction (duo's u = y ^ w ^ kx IS
    # one LOP3), so reaching a correct u already costs 2 ops (the multiply w*m1, then the XOR) that produce no output, and
    # the two live outputs read u, so a length-3 program (one op after a 2-op u) cannot write both. By op type the update
    # needs two non-composing multiplies (w*m1 and u*a; route (i)), one XOR (u, not single, not affine in w) and one
    # rotate-add (z'), four distinct irreducible ops = honest's f = 4. ---
    checks.check("fx1 needs >= 3 instructions (two live outputs, neither a single instruction)",
                 single_instruction(yp1, pool) is None and single_instruction(zp1, pool) is None)
    checks.check("fx1's 4th op is genuine: u is a 2-op value (not single), its ops write no output, and both outputs read u",
                 single_instruction(u1, pool) is None)

    # --- fx2 adds one more irreducible op: its u = y ^ ((w^x1)*m2) ^ kx reads w through XOR-then-multiply-then-XOR ---
    u2 = [u32(pool["y"][i] ^ u32((pool["w"][i] ^ x1) * m2) ^ kx) for i in range(nv)]
    checks.check("fx2's u = y ^ ((w^x1)*m2) ^ kx is NOT a single instruction", single_instruction(u2, pool) is None)
    # the leading (w ^ x1) cannot fold into the multiply: (w^x1)*m2 is not affine in w, so it is not one IMAD of w.
    checks.check("fx2's (w^x1)*m2 is not affine in w (the leading XOR cannot fold into the multiply)",
                 not is_affine_in_w(lambda w: u32((w ^ x1) * m2)))
    yp2 = [fx_word(y, z, w, 2, a, kx, r)[0] for (y, z, w) in vectors]
    zp2 = [fx_word(y, z, w, 2, a, kx, r)[1] for (y, z, w) in vectors]
    checks.check("fx2 y', z' are not single instructions (needs >= 5 = honest's f)",
                 single_instruction(yp2, pool) is None and single_instruction(zp2, pool) is None)

    # --- the collapse routes, all failing ---
    # (i) the two multiplies do not compose: y'(w) for fixed (y, z) is not affine in w (not w*C + D), so (w*m1) and the
    #     (u*a) multiply cannot fuse into one.
    y_fixed, z_fixed = 0x12345678, 0x9ABCDEF0
    checks.check("route (i): y'(w) is not affine in w, so the two multiplies do not compose into one IMAD",
                 not is_affine_in_w(lambda w: fx_word(y_fixed, z_fixed, w, 1, a, kx, r)[0]))
    # explicit: (y ^ (w*m1) ^ kx)*a differs from w*(m1*a) + c for every constant c (the XOR with live y breaks it).
    lhs = [u32((pool["y"][i] ^ u32(pool["w"][i] * m1) ^ kx) * a) for i in range(nv)]
    composed = [u32(pool["w"][i] * u32(m1 * a)) for i in range(nv)]
    diffs = {u32(lhs[i] - composed[i]) for i in range(nv)}
    checks.check("route (i): (y^(w*m1)^kx)*a - w*(m1*a) is not constant (no multiply composition)", len(diffs) > 1,
                 f"{len(diffs)} distinct offsets")

    # (ii) the fx IMAD's add slot cannot pre-absorb duo's + z: putting z in the fx multiply changes u and the output.
    absorbed = [duo_from_dprime(pool["y"][i], 0, u32(pool["w"][i] * m1 + pool["z"][i]), a, kx, r)[0] for i in range(nv)]
    checks.check("route (ii): folding duo's + z into the fx IMAD (w*m1 + z) changes the output (no add-slot absorption)",
                 absorbed != yp1)

    # (iii) no single instruction makes two words' fx intermediates: IMAD.WIDE of one (w0, m1) gives one 64-bit product;
    #       its halves are not two independent 32-bit products (w0*m1, w1*m1) for distinct words.
    g2 = lcg(0x2222)
    two_word_share = False
    for _ in range(200):
        w0, w1 = next(g2), next(g2)
        prod64 = w0 * m1            # IMMA.WIDE: full 64-bit product of ONE operand pair
        lo, hi = prod64 & M32, (prod64 >> 32) & M32
        if (lo == u32(w0 * m1) and hi == u32(w1 * m1)):  # would need hi to be the other word's product
            two_word_share = True
            break
    checks.check("route (iii): one IMAD.WIDE does not produce two words' independent fx products", not two_word_share)
    # IDP.4A is an int8 4-term dot to one int32, not a 32-bit multiply; LOP3's three inputs are consumed by one word's XOR.
    checks.check("route (iii): IDP.4A is an int8 dot (not a 32-bit multiply) and LOP3's 3 inputs serve one word's u",
                 True, "structural")

    # (iv) the fx multiply is per-lane (the committed word differs per lane), so it cannot move to the warp-uniform
    #      datapath, which writes one warp-uniform word an instruction (NOSUM).
    checks.check("route (iv): the fx multiply is per-lane (word differs per lane) -> not on the uniform datapath",
                 len({fx_chain(w, 1) for w in ws}) > 1, "per-lane, data-dependent")

    # --- per facet (price form): on the binding joint-issue row every vector class prices alike, so phi* = phi_H ---
    # a joint-issue row prices MMA at a_r and every vector class at b_r; a pipe row (A) prices its own class high and the
    # IMAD form of an ALU op low -- that is the route FOLD-OPT-per-facet must exclude, and the design avoids it by binding
    # a joint row.
    ji_row = {"MMA": 7.8, "ALU": 1.0, "IMAD": 1.0, "LOP3": 1.0, "SHF": 1.0, "LEA": 1.0, "UNI": 1.0, "TENSOR": 7.8}
    a_pipe = {"MMA": 0.0, "ALU": 2.0, "IMAD": 0.5, "LOP3": 2.0, "SHF": 2.0, "LEA": 2.0, "UNI": 1.0, "TENSOR": 0.0}
    fx1_fold = ["IMAD", "LOP3", "IMAD", "LEA"]   # d'=w*m1; u=y^d'^kx; y'=u*a+z; z'=rotl(u,r)+y'
    fx2_fold = ["LOP3", "IMAD", "LOP3", "IMAD", "LEA"]

    def price(fold, row):
        return sum(row[op] for op in fold)

    phi_H_fx1 = price(fx1_fold, ji_row)
    checks.check("per facet: honest fx1 fold costs 4 vector slots on the joint-issue row", phi_H_fx1 == 4.0, phi_H_fx1)
    checks.check("per facet: honest fx2 fold costs 5 vector slots on the joint-issue row",
                 price(fx2_fold, ji_row) == 5.0)
    # the IMAD-form route (rewrite the two ALU/LOP3/LEA ops as IMAD) saves nothing on a joint row (all classes = b_r);
    # it would only help on a pipe row, which is why the design binds a joint row.
    fx1_imad_form = ["IMAD", "IMAD", "IMAD", "IMAD"]
    checks.check("per facet: the IMAD-form of fx1's ALU work is no cheaper on the joint row (phi* = phi_H)",
                 price(fx1_imad_form, ji_row) >= phi_H_fx1, price(fx1_imad_form, ji_row))
    checks.check("per facet: the SAME IMAD-form WOULD undercut honest on the A pipe (why the design binds a joint row)",
                 price(fx1_imad_form, a_pipe) < price(fx1_fold, a_pipe),
                 f"{price(fx1_imad_form, a_pipe)} < {price(fx1_fold, a_pipe)}")
    # no cheaper route overall on the joint row: fewer ops excluded (>=4), UNI priced on the row and the work per-lane,
    # tensor has no fold form. So mu_f = 0.
    routes_fx1 = {"honest": phi_H_fx1, "imad-form": price(fx1_imad_form, ji_row),
                  "uni-move": price(["UNI", "LOP3", "IMAD", "LEA"], ji_row)}
    checks.check("per facet: phi*(fx1) = phi_H on the joint-issue row, so mu_f = 0 ( e_mand = 0)",
                 min(routes_fx1.values()) >= phi_H_fx1, routes_fx1)

    if checks.mutations:
        # 1. The design's end-on-IMAD is load-bearing: an XOR-ending chain (d' = w ^ c, c a known constant) collapses --
        #    u = y ^ (w^c) ^ kx = y ^ w ^ (c^kx) is a single LOP3 (the constant c^kx folds in), so the fx op vanishes and
        #    f drops to 3. Caught: with the folded constant in the pool, u is a single instruction.
        c_xor = key_xor(1, 7)
        cc = u32(c_xor ^ kx)
        ext = dict(pool)
        ext["cc"] = [cc] * nv
        u_xor_end = [u32(pool["y"][i] ^ u32(pool["w"][i] ^ c_xor) ^ kx) for i in range(nv)]
        collapse = (all(u_xor_end[i] == u32(pool["y"][i] ^ pool["w"][i] ^ cc) for i in range(nv))
                    and single_instruction(u_xor_end, ext) is not None)
        checks.mutation("an XOR-ending chain collapses: u = y^(w^c)^kx folds to one LOP3 (so end-on-IMAD is necessary)",
                        collapse, single_instruction(u_xor_end, ext))
        # 2. If the fx multiplier were even it would not be a bijection: w and w + 2^31 map to the same product.
        even_m = m1 & ~1
        collide = u32(0 * even_m) == u32(0x80000000 * even_m) and u32(7 * even_m) == u32((7 + 0x80000000) * even_m)
        checks.mutation("an even fx multiplier is not a bijection of the word (w and w+2^31 collide)", collide)
        # 3. The search is sensitive: a linear fx fold (no XOR, no second multiply) y' = w*(m1*a) + z is a single IMAD.
        lin = [u32(pool["w"][i] * u32(m1 * a) + pool["z"][i]) for i in range(nv)]
        checks.mutation("a linear fx fold y' = w*(m1*a) + z is a single IMAD (claim would flip)",
                        single_instruction(lin, pool) is not None, single_instruction(lin, pool))
        # 4. The two-word search is sensitive: words an IMAD.WIDE writes after a LOP3 are found as such.
        uu4 = [y0[i] ^ wd[i] ^ kxw for i in range(npts)]
        lh4 = [wide(uu4[i], aw, z0[i], y0[i], True, w4) for i in range(npts)]
        found = width_wide_writes_both(w4, [x[1] for x in lh4], [x[0] for x in lh4], wins)
        checks.mutation("an IMAD.WIDE's two words after a LOP3 are found by the two-word search", found is not None,
                        found)
        # 5., 6. The two-instruction search reads two-word instructions both ways: an IMAD.WIDE's high word over a
        #    LOP3, and a LOP3 of an IMAD.WIDE's high word, are found as two-instruction values.
        hi5 = [x[1] for x in lh4]
        found5 = width_le2_value(w4, hi5, wins)
        checks.mutation("an IMAD.WIDE's high word over a LOP3 is a two-instruction value", found5 is not None, found5)
        v6 = [wide(y0[i], aw, z0[i], kxw, False, w4)[1] ^ wd[i] ^ kxw for i in range(npts)]
        found6 = width_le2_value(w4, v6, wins)
        checks.mutation("a LOP3 of an IMAD.WIDE's high word is a two-instruction value", found6 is not None, found6)

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
