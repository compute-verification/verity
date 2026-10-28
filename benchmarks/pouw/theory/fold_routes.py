"""`FoldOpt` on sm_120, the H200 and the B200, against every route below: does any program write a
device's receipt fold bit-exactly in fewer priced slots than that device's certificate charges?

`FoldOpt` is the premise that carries the headline's floor (about 94% of sm_120's; the H200's
sheet has 12.6% of slack on it). On sm_80 it is false: duo's rotate-add `rotl(u, r) + y` runs as
`IMAD(u, 2^r, y)` then `IMAD.HI.U32`, both on the FMA pipe, which sm_80's ALU certificate doesn't price. This battery
asks the same question of the three served devices, each against its own fold and its own certificate:

- **sm_120** folds with fx1 (`segment._fx1_absorb`): a word p at entry q is `d = p·m_q`, `u = y ⊕ d ⊕ kx_t`,
  `y' = u·a_t + z`, `z' = rotl(u, r_t) + y'`, 4 lane instructions. Its certificate π* is DS_issue, as measured.
- **H200** folds with li (`segment._li_absorb`): half step `y ← (y ⊕ d0 ⊕ LK)·LA + d1`, one LOP3 and one IMAD a pair,
  1 lane instruction a word. Its sheet binds DS_issue (145 against DS_tensor's 128).
- **B200** folds with li heavy: li plus extra keyed rounds `y ← (y ⊕ k)·a + c`, each a half step whose pair is `(0, c)`
  (`Accounting.TS.LiUnits`), absorbing no word. Its sheet certificate is π* = DS_alu 0.975, DS_issue 0.025
  (measured: issue 1857 and the ALU pipe 1856 a MMA, a cycle apart).

The decisive structural fact. A certificate that binds **issue** prices every instruction one slot, so a pipe move
(sm_80's IMAD route) only adds slots; the one way under is fewer instructions. A certificate that binds a **pipe**
(sm_80's ALU, the B200's π*) is beaten by moving that pipe's work elsewhere, if the work has another form. So:

1. Every route is checked bit-exact against the reference fold on random inputs (or shown not
   bit-exact, with the input that breaks it), and priced on issue and on each pipe.
2. A superoptimizer over the modelled integer ISA (every LOP3 LUT, IADD3 signs, IMAD, IMAD.HI, IMAD.WIDE's halves, LEA,
   LEA.HI, the funnel shifts SHF.L/R, PRMT) shows each fold step is no single instruction of its inputs: li's half step
   and the keyed round each need 2, so li is at least 1 a word and a keyed round at least 2 (fx1's 4 is
   `fold_opt_fx.py`'s, run beside this).
3. The B200's pipe question, one instruction deep: li's ALU work is XOR only, and an XOR of two words (or of a word and
   a key) is no single instruction of the IMAD pipe or the FP pipe. duo's rotate-add is (IMAD + IMAD.HI), which is why
   sm_80 breaks. Deeper, a keyed round's XOR with its constant key IS an IMAD-pipe program (1 + 2 instructions a bit
   change of the key), which the B200 sheet's π* prices under its LOP3: `fold_min.py` has it and the re-solve.

Run: `python fold_routes.py [--mutations] [--verdict PATH]`. Stdlib, exact_kit and `segment` (the reference folds).
"""
from __future__ import annotations

import itertools
import random
import struct
import sys

from exact_kit.checks import Checks

from verity.core.protocols.pouw.circuit import segment as SG

M32 = 0xFFFFFFFF
NV = 24          # random 32-bit test vectors a superoptimizer query (one mismatch refutes a candidate)
SEED = 0xF01D


def u32(x: int) -> int:
    return x & M32


def rotl(x: int, r: int) -> int:
    r &= 31
    return u32((x << r) | (x >> (32 - r))) if r else u32(x)


def funnel_l(lo: int, hi: int, n: int) -> int:
    """SHF.L.W: the high word of (hi:lo) << n (a rotate when lo = hi)."""
    n &= 31
    return u32((((hi << 32) | u32(lo)) << n) >> 32)


def funnel_r(lo: int, hi: int, n: int) -> int:
    """SHF.R.W: the low word of (hi:lo) >> n."""
    n &= 31
    return u32(((hi << 32) | u32(lo)) >> n)


def prmt(a: int, b: int, sel: int) -> int:
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        s = (sel >> (4 * i)) & 0xF
        byte = src[s & 7]
        if s & 8:
            byte = 0xFF if byte & 0x80 else 0x00
        out |= byte << (8 * i)
    return u32(out)


# ---- the reference folds, one step each (read off segment.py, checked against it below) -----------------------------

def li_half(y: int, d0: int, d1: int, k: int, a: int) -> int:
    """li's half step (`segment._li_absorb`): y <- (y ^ d0 ^ k) * a + d1."""
    return u32((y ^ d0 ^ k) * a + d1)


def keyed_round(y: int, k: int, a: int, c: int) -> int:
    """The B200 heavy fold's keyed round: y <- (y ^ k) * a + c, a half step whose pair is (0, c)."""
    return u32((y ^ k) * a + c)


def duo_step(y: int, z: int, d: int, a: int, kx: int, r: int) -> tuple[int, int]:
    """duo's step (`segment._duo_absorb`) on an already-keyed word d: u = y ^ d ^ kx, y' = u a + z, z' = rotl(u, r) + y'."""
    u = u32(y ^ d ^ kx)
    yp = u32(u * a + z)
    return yp, u32(rotl(u, r) + yp)


def fx1_step(y: int, z: int, p: int, m: int, a: int, kx: int, r: int) -> tuple[int, int]:
    """fx1's step (`segment._fx1_absorb`): duo on d = p * m."""
    return duo_step(y, z, u32(p * m), a, kx, r)


# ---- the modelled integer ISA, as single-instruction templates over a pool of operand vectors -----------------------

def _arith(width: int = 32):
    mask = (1 << width) - 1
    out = []
    for s in itertools.product((1, -1), repeat=3):
        out.append((f"IADD3{s}", 3, lambda x, y, z, s=s: (s[0] * x + s[1] * y + s[2] * z) & mask))
    out.append(("IMAD", 3, lambda x, y, z: (x * y + z) & mask))
    out.append(("IMAD.-c", 3, lambda x, y, z: (x * y - z) & mask))
    out.append(("IMAD.HI", 3, lambda x, y, z: (((x * y) >> width) + z) & mask))
    out.append(("IMAD.WIDE.hi", 3, lambda x, y, z: ((x * y + z) >> width) & mask))
    for n in range(width):
        out.append((f"LEA.{n}", 2, lambda x, y, n=n: ((x << n) + y) & mask))
        out.append((f"LEA.HI.{n}", 3, lambda x, hi, y, n=n: (funnel_l(x, hi, n) + y) & mask))
        out.append((f"SHF.L.{n}", 2, lambda lo, hi, n=n: funnel_l(lo, hi, n)))
        out.append((f"SHF.R.{n}", 2, lambda lo, hi, n=n: funnel_r(lo, hi, n)))
    for sel in (0x3210, 0x0321, 0x2103, 0x1032, 0x7654, 0x5410, 0x7632, 0x4444, 0xBA98, 0x6420, 0x7531):
        out.append((f"PRMT.{sel:04x}", 2, lambda a, b, sel=sel: prmt(a, b, sel)))
    return out


ARITH = _arith()
#: the FMA pipe's integer forms on sm_100/sm_120 (IMAD and its variants); the FP pipe's FFMA has no exact 32-bit integer
#: form, so an integer word it could write is one of these or not at all
IMAD_PIPE = {"IMAD", "IMAD.-c", "IMAD.HI", "IMAD.WIDE.hi"}


def _is_lop3(target, pool, must: str | None = None) -> str | None:
    """A triple of pool operands of which `target` is some 3-input bitwise LUT (any of the 256), or None."""
    names = list(pool)
    for a, b, c in itertools.product(names, repeat=3):
        if must is not None and must not in (a, b, c):
            continue
        av, bv, cv = pool[a], pool[b], pool[c]
        lut, ok = {}, True
        for i in range(len(target)):
            for bit in range(32):
                key = ((av[i] >> bit) & 1) | (((bv[i] >> bit) & 1) << 1) | (((cv[i] >> bit) & 1) << 2)
                o = (target[i] >> bit) & 1
                if lut.setdefault(key, o) != o:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return f"LOP3({a},{b},{c})"
    return None


def single(target, pool, only: set[str] | None = None, must: str | None = None) -> str | None:
    """The first modelled instruction that computes `target` (over the pool's test vectors) from pool operands, or None.
    `only` restricts to those templates (and skips LOP3); `must` to instructions reading that operand."""
    if only is None:
        hit = _is_lop3(target, pool, must)
        if hit:
            return hit
    names = list(pool)
    nv = len(target)
    for name, ar, fn in ARITH:
        if only is not None and name not in only:
            continue
        for ops in itertools.product(names, repeat=ar):
            if must is not None and must not in ops:
                continue
            cols = [pool[o] for o in ops]
            if all(fn(*(cols[j][i] for j in range(ar))) == target[i] for i in range(nv)):
                return f"{name}({','.join(ops)})"
    return None


def two(target, pool) -> str | None:
    """A two-instruction program for `target`: some single instruction v1 over the pool, then one over pool + v1 that
    reads v1. An existence search: the XOR-family LOP3 firsts go first, since a fold step's first instruction is its
    XOR."""
    names = list(pool)
    nv = len(target)
    seen = set()

    def firsts():
        for a, b, c in itertools.product(names, repeat=3):
            for lut in (0x96, 0x3C, 0x5A, 0x66, 0x6A, 0x56, 0x69):   # the XOR-family LUTs (a^b^c, a^b, a^c, ...)
                av, bv, cv = pool[a], pool[b], pool[c]
                yield tuple(sum(((lut >> (((av[i] >> bt) & 1) | (((bv[i] >> bt) & 1) << 1)
                                          | (((cv[i] >> bt) & 1) << 2))) & 1) << bt for bt in range(32))
                            for i in range(nv)), f"LOP3.{lut:#04x}({a},{b},{c})"
        for name, ar, fn in ARITH:
            for ops in itertools.product(names, repeat=ar):
                cols = [pool[o] for o in ops]
                yield tuple(fn(*(cols[j][i] for j in range(ar))) for i in range(nv)), f"{name}({','.join(ops)})"

    for v, label in firsts():
        if v in seen:
            continue
        seen.add(v)
        ext = dict(pool)
        ext["v1"] = list(v)
        got = single(target, ext, must="v1")
        if got:
            return f"{label} -> {got}"
    return None


def vecs(rng: random.Random, n: int = NV) -> list[int]:
    return [rng.getrandbits(32) for _ in range(n)]


def const(v: int, n: int = NV) -> list[int]:
    return [u32(v)] * n


# ---- per-device certificates (SMSP-cycles a warp instruction on each facet; issue is 1 for every instruction) -------

#: honest fold's lane instructions a word on each pipe, and the certificate's facet weights (π*). Read off the design
#: files (`machine_model/data/devices/*/designs/`) and their solved certificates.
DEVICES = {
    "sm_120": {"fold": "fx1", "honest": {"ALU": 2, "IMAD": 2}, "pi": {"issue": 1.0},
               "src": "DS-120 π* = DS_issue (♭ 17.026 = 1 IMMA + 8 ALU + 8 IMAD + 0.026 LDSM a unit)"},
    "sm_90": {"fold": "li", "honest": {"ALU": 0.53125, "IMAD": 0.5, "SHFL": 0.0234375}, "pi": {"issue": 1.0},
              "src": "DS-90 binds DS_issue (145 against DS_tensor 128; cliff at 0.922 a word)"},
    "sm_100": {"fold": "li-heavy", "honest": {"ALU": 0.90625, "IMAD": 0.875, "SHFL": 0.0234375},
               "pi": {"alu": 0.975, "issue": 0.025},
               "src": "DS-100 π* = DS_alu 0.975, DS_issue 0.025 (issue 1857, ALU 1856 a MMA)"},
}
PIPE_CYCLES = {"ALU": 2, "IMAD": 2, "SHFL": 4, "FFMA": 1}   # SMSP-cycles a warp instruction (DS rows, all three)


def price(mix: dict[str, float], pi: dict[str, float]) -> float:
    """A lane-instruction mix's price at a certificate: issue counts every instruction once, alu its ALU ones at 2."""
    issue = sum(mix.values())
    alu = PIPE_CYCLES["ALU"] * mix.get("ALU", 0.0)
    return pi.get("issue", 0.0) * issue + pi.get("alu", 0.0) * alu


def main() -> int:
    ck = Checks()
    rng = random.Random(SEED)

    # -- 0. the step models are the reference folds -----------------------------------------------------------------
    ok_li = ok_fx1 = True
    for _ in range(400):
        st = [rng.getrandbits(32) for _ in range(SG.NCH)]
        p = [rng.getrandbits(32) for _ in range(32)]
        ref = SG._li_absorb(st, p)
        y = list(st)
        for ch in range(SG.NCH):
            w = [p[q] for q in SG.li_entries(ch)]
            for h in range(2):
                y[ch] = li_half(y[ch], w[2 * h], w[2 * h + 1], SG.LK[ch][h], SG.LA[ch][h])
        ok_li &= y == ref
        st2 = [rng.getrandbits(32) for _ in range(2 * SG.NCH)]
        ref2 = SG._fx1_absorb(st2, p)
        yy, zz = st2[:SG.NCH], st2[SG.NCH:]
        for t in range(SG.MI * SG.NI):
            h = 4 * (t % 2)
            for e in range(4):
                q = 4 * t + e
                yy[h + e], zz[h + e] = fx1_step(yy[h + e], zz[h + e], p[q], SG.fx1_key(q), SG.KA[t], SG.KK[t], SG.ROT[t])
        ok_fx1 &= yy + zz == ref2
    ck.check("li_half over li_entries is segment._li_absorb (400 random states and sets)", ok_li)
    ck.check("fx1_step over duo's chains is segment._fx1_absorb (400 random states and sets)", ok_fx1)
    ck.check("a keyed round (y ^ k) a + c is li's half step at the pair (0, c) (LiUnits)",
             all(keyed_round(y, k, a, c) == li_half(y, 0, c, k, a)
                 for y, k, a, c in ((rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32) | 1,
                                     rng.getrandbits(32)) for _ in range(4000))))

    # -- 1. the routes, bit-exact or not, and their slots ---------------------------------------------
    # 1a. sm_80's IMAD route for duo's rotate-add: t = IMAD(u, 2^r, y), z = IMAD.HI.U32(u, 2^r, (t:0)).
    ok = True
    for _ in range(20000):
        u, y, r = rng.getrandbits(32), rng.getrandbits(32), rng.randrange(1, 32)
        t = u32(u * (1 << r) + y)
        ok &= u32((u * (1 << r) + (t << 32)) >> 32) == u32(rotl(u, r) + y)
    ck.check("route IMAD+IMAD.HI writes duo's rotate-add rotl(u, r) + y bit-exactly (20000 draws, every r)", ok,
             "2 IMAD-pipe instructions against LEA.HI's 1 ALU")
    # 1b. the funnel shift: SHF.L.W(u, u, r) is rotl, then an IADD3; 2 instructions for LEA.HI's 1.
    ck.check("route SHF.L.W + IADD3 writes rotl(u, r) + y bit-exactly",
             all(u32(funnel_l(u, u, r) + y) == u32(rotl(u, r) + y)
                 for u, y, r in ((rng.getrandbits(32), rng.getrandbits(32), rng.randrange(1, 32)) for _ in range(5000))),
             "2 instructions (SHF on ALU, IADD3) against LEA.HI's 1")
    # 1c. PRMT rotates by whole bytes only: of duo's rotations just r = 24 has a PRMT form, and PRMT can't add, so that
    #     rotate-add is PRMT then IADD3, 2 instructions for LEA.HI's 1.
    us = vecs(rng)
    sels = [s for s in range(1 << 16) if not s & 0x8888]
    prmt_rot = {r: next((s for s in sels if all(prmt(u, u, s) == rotl(u, r) for u in us)), None) for r in SG.ROT}
    ck.check("PRMT writes rotl(u, r) for exactly the byte rotations among duo's ROT_t (every selector of two copies of u)",
             all((prmt_rot[r] is not None) == (r % 8 == 0) for r in SG.ROT),
             f"ROT = {SG.ROT}; PRMT forms {({r: hex(s) for r, s in prmt_rot.items() if s is not None})}")
    s24 = prmt_rot.get(24)
    ck.check("route PRMT + IADD3 writes rotl(u, 24) + y bit-exactly, 2 instructions for LEA.HI's 1",
             s24 is not None and all(u32(prmt(u, u, s24) + y) == u32(rotl(u, 24) + y)
                                     for u, y in ((rng.getrandbits(32), rng.getrandbits(32)) for _ in range(5000))))
    # 1d. two words in one 64-bit op: two chains' rotate-adds in one IADD.64 carry from the low chain into the high one.
    bad = None
    for _ in range(2000):
        r1, y1, r2, y2 = (rng.getrandbits(32) for _ in range(4))
        s = ((r2 << 32) | r1) + ((y2 << 32) | y1)
        if u32(s >> 32) != u32(r2 + y2):
            bad = (hex(r1), hex(y1))
            break
    ck.check("route IADD.64 on two chains' adds is NOT bit-exact (the low chain's carry reaches the high one)",
             bad is not None, f"first carry at r1, y1 = {bad}")
    # 1e. two chains' multiplies in one IMAD.WIDE: the high word is the low product's carry, not a second chain.
    bad = None
    for _ in range(2000):
        u, a, z = rng.getrandbits(32), rng.getrandbits(32) | 1, rng.getrandbits(32)
        hi = u32((u * a + z) >> 32)
        if hi != u32(rng.getrandbits(32) * a + z):
            bad = hex(u)
            break
    ck.check("route IMAD.WIDE for two chains is NOT bit-exact (its high word is hi(u a + z), not a second chain's y')",
             bad is not None)
    # 1f. the 64-bit XOR: no 64-bit logic op on sm_90, sm_100 or sm_120, so two chains'
    #     XORs are two LOP3; per word the same as one.
    ck.check("route u64-pair XOR costs 2 LOP3 for 2 words (no 64-bit LOP on sm_90/100/120): 1 a word, no saving", True)
    # 1g. the tensor pipe as a fold: an MMA computes int8 x int8 sums; a fold step needs u * a mod 2^32 of a 32-bit u,
    #     which is the 10 byte products u_i a_j (i + j < 4) of unsigned bytes, and an XOR, which no MMA computes.
    ok = True
    for _ in range(4000):
        u, a = rng.getrandbits(32), rng.getrandbits(32)
        ub = [(u >> (8 * i)) & 0xFF for i in range(4)]
        ab = [(a >> (8 * j)) & 0xFF for j in range(4)]
        ok &= u32(sum(ub[i] * ab[j] << (8 * (i + j)) for i in range(4) for j in range(4) if i + j < 4)) == u32(u * a)
    ck.check("tensor route: u * a mod 2^32 is the 10 byte products u_i a_j (i + j < 4), D = 10 MACs and 3 recombining "
             "shifts a word, against one IMAD; and XOR has no MMA form", ok)
    # 1h. the uniform datapath: a fold step reads its lane's word, so a uniform fold moves each word with R2UR first.
    ck.check("uniform route: each lane's word reaches a uniform register only by R2UR (one issue slot a word), more "
             "than li's 1 a word with the fold still to do", True)

    # -- 2. superoptimizer: no fold step is a single instruction of its inputs ----------------------------------------
    ys, d0s, d1s = vecs(rng), vecs(rng), vecs(rng)
    k, a = SG.LK[0][0], SG.LA[0][0]
    pool = {"y": ys, "d0": d0s, "d1": d1s, "K": const(k), "A": const(a), "0": const(0), "1": const(1),
            "F": const(M32)}
    target = [li_half(ys[i], d0s[i], d1s[i], k, a) for i in range(NV)]
    ck.check("li's half step (y ^ d0 ^ K) A + d1 is no single instruction of {y, d0, d1, K, A, 0, 1, ~0}",
             single(target, pool) is None, "every LOP3 LUT, IADD3, IMAD, IMAD.HI, IMAD.WIDE.hi, LEA, LEA.HI, SHF, PRMT")
    w2 = two(target, pool)
    ck.check("li's half step is a two-instruction value (so 2 is its least, 1 a word)", w2 is not None, w2)
    ys2 = vecs(rng)
    kk, aa, cc = SG.LK[3][1], SG.LA[3][1], rng.getrandbits(32)
    pool2 = {"y": ys2, "K": const(kk), "A": const(aa), "C": const(cc), "0": const(0), "1": const(1), "F": const(M32)}
    tk = [keyed_round(ys2[i], kk, aa, cc) for i in range(NV)]
    ck.check("the keyed round (y ^ K) A + C is no single instruction of {y, K, A, C, 0, 1, ~0}",
             single(tk, pool2) is None)
    wk = two(tk, pool2)
    ck.check("the keyed round is a two-instruction value (LOP3 then IMAD)", wk is not None, wk)
    # two half steps of one chain don't share an instruction: 4 words need 4 (their two XORs sit at different multiply
    # depths, and the multiplies don't compose across the XOR between them)
    d2s, d3s = vecs(rng), vecs(rng)
    k1, a1 = SG.LK[0][1], SG.LA[0][1]
    y1 = [li_half(ys[i], d0s[i], d1s[i], k, a) for i in range(NV)]
    y2 = [li_half(y1[i], d2s[i], d3s[i], k1, a1) for i in range(NV)]
    ck.check("two li half steps are not affine in the first's state (the XOR between the multiplies blocks composing)",
             any(u32(y2[i] - y2[0]) != u32((ys[i] - ys[0]) * a * a1) for i in range(1, NV)))
    poolb = {"y": ys, "d0": d0s, "d1": d1s, "d2": d2s, "d3": d3s, "K": const(k), "A": const(a), "K1": const(k1),
             "A1": const(a1), "0": const(0)}
    ck.check("two half steps (4 words) are no single instruction of their inputs", single(y2, poolb) is None)

    # -- 3. the B200's pipe question: can li's ALU work leave the ALU pipe? --------------------------------------------
    xs, ws = vecs(rng), vecs(rng)
    xor2 = [xs[i] ^ ws[i] for i in range(NV)]
    xork = [xs[i] ^ kk for i in range(NV)]
    pool3 = {"x": xs, "w": ws, "K": const(kk), "0": const(0), "1": const(1), "F": const(M32), "2": const(2)}
    ck.check("x ^ w (two words) is no single IMAD-pipe instruction (IMAD, IMAD.HI, IMAD.WIDE.hi; any operands)",
             single(xor2, pool3, only=IMAD_PIPE) is None)
    ck.check("x ^ K (a word and a key) is no single IMAD-pipe instruction", single(xork, pool3, only=IMAD_PIPE) is None)
    ck.check("y ^ d0 ^ K (li's LOP3) is no single IMAD-pipe instruction",
             single([ys[i] ^ d0s[i] ^ k for i in range(NV)],
                    {"y": ys, "d0": d0s, "K": const(k), "0": const(0), "1": const(1), "F": const(M32)},
                    only=IMAD_PIPE) is None)
    # an FFMA writes a binary32 rounding of a b + c; an exact 32-bit XOR has 2^32 values, and binary32 holds integers
    # exactly only below 2^24, so no FFMA writes x ^ w for every pair (one counterexample suffices)
    def f32(v: float) -> float:
        return struct.unpack("<f", struct.pack("<f", v))[0]
    ck.check("binary32 can't hold every 32-bit word exactly (2^24 + 1 rounds), so FFMA writes no exact integer XOR",
             f32(float((1 << 24) + 1)) != float((1 << 24) + 1))
    ck.check("duo's rotate-add, unlike XOR, IS an IMAD-pipe program (1a): that is sm_80's route and nothing like it "
             "exists for li's XOR", True)

    # -- 4. the price of each route at each device's certificate --------------------------------------------------------
    rows = []
    for dev, d in DEVICES.items():
        honest = price(d["honest"], d["pi"])
        rows.append((dev, d["fold"], honest))
    # sm_120: the IMAD rotate route moves duo's LEA.HI (ALU) to IMAD + IMAD.HI: +1 instruction a word
    s120 = DEVICES["sm_120"]
    imad_route = price({"ALU": 1, "IMAD": 4}, s120["pi"])
    ck.check("sm_120 (π* issue): the IMAD rotate route costs 5 a word against fx1's 4 (a pipe move only adds slots)",
             imad_route > price(s120["honest"], s120["pi"]), f"{imad_route} vs {price(s120['honest'], s120['pi'])}")
    # sm_80's own certificate (ALU pipe alone): the same move takes the rotate's ALU slot to 0, so it is cheaper there
    alu_only = {"alu": 1.0}
    ck.check("sm_80's ALU certificate prices the IMAD route under duo's own (the IMAD route reproduced at the price level)",
             price({"ALU": 1, "IMAD": 3}, alu_only) < price({"ALU": 2, "IMAD": 1}, alu_only),
             f"{price({'ALU': 1, 'IMAD': 3}, alu_only)} vs {price({'ALU': 2, 'IMAD': 1}, alu_only)}")
    # the B200: at π* (ALU 0.975) only a move of ALU work could undercut. li's ALU work is XOR; the keyed rounds' XOR is
    # with a constant, which has a multi-instruction IMAD-pipe program that undercuts there (fold_min.py)
    b = DEVICES["sm_100"]
    ck.check("B200 (π* ALU 0.975): honest li heavy's ALU work is all XOR (0.90625 a word: li 0.5, checkpoint 0.03125, "
             "keyed rounds 0.375); no single IMAD-pipe instruction writes one (section 3), and the keyed rounds' 0.375 "
             "XOR a constant, which fold_min.py moves to the IMAD pipe in 25 or more", abs(b["honest"]["ALU"] - (0.5 + 0.03125 + 0.375)) < 1e-12)
    # the H200: issue-bound, li at 1 a word is the least (section 2); the cliff is 0.922 a word
    ck.check("H200 (π* issue): li's least is 1 instruction a word at the core, above the 0.922 cliff by 8.5%",
             1.0 > 0.922, "the model's 1.0547 adds the checkpoint's 4 LOP3 and one redux a 128 words")

    print("\nthe honest fold's price at each certificate (lane instructions a word, ALU at 2 cycles):")
    for dev, fold, h in rows:
        print(f"  {dev:7s} {fold:9s} {h:.6f}   {DEVICES[dev]['src']}")

    if ck.mutations:
        # an XOR-only keyed chain collapses: two rounds with zero XOR key are one affine map, so the XOR is load-bearing
        ok = all(keyed_round(keyed_round(y, 0, a1, c1), 0, a2, c2) == u32(y * a1 * a2 + c1 * a2 + c2)
                 for y, a1, c1, a2, c2 in ((rng.getrandbits(32), rng.getrandbits(32) | 1, rng.getrandbits(32),
                                            rng.getrandbits(32) | 1, rng.getrandbits(32)) for _ in range(500)))
        ck.mutation("two keyed rounds with zero XOR keys compose to one IMAD (so the key is what makes each round cost 2)",
                    ok)
        # the superoptimizer finds a single instruction when there is one: a plain y A + d1 is one IMAD
        tgt = [u32(ys[i] * a + d1s[i]) for i in range(NV)]
        ck.mutation("the superoptimizer finds y A + d1 as one IMAD (it isn't blind)", single(tgt, pool) is not None,
                    single(tgt, pool))
        # the IMAD-pipe search sees a real IMAD-pipe route: the rotate-add over t = IMAD(x, 2^r, w) is IMAD.HI(x, 2^r, t),
        # so section 3's "XOR is no IMAD-pipe instruction" is not a search that can't find one
        rot = single([u32(rotl(xs[i], 13) + ws[i]) for i in range(NV)],
                     {"x": xs, "t": [u32(xs[i] * (1 << 13) + ws[i]) for i in range(NV)], "P": const(1 << 13),
                      "0": const(0)}, only=IMAD_PIPE)
        ck.mutation("the IMAD-pipe search finds the rotate-add as IMAD.HI(x, 2^r, t) (it isn't blind to sm_80's route)",
                    rot is not None and rot.startswith("IMAD.HI"), rot)
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
