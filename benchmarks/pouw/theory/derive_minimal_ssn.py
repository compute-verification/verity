#!/usr/bin/env python3
"""OneInstructionMinimal for ssn-1l: no single sm_90 integer lane instruction places coin bit p ≠ 24 of a raw XOF word
at bit 24 of a data register, so each later step's funnel-shift rotate (`derive_extra`) is a distinct coin-dependent write.

ssn-1l's later k32 steps each serve one noise lane, lane 3 of a register whose lanes 0-2 are data: X = A | (rotr(h_w,
(p - 24) mod 32) & 0x01000000), A < 2^24, p the lane's bit of the row's coin word h_w (`noise_sweep.py`'s
`ssn_1l_layout`: every p in 0..31 occurs). Honest spends a SHF.L.W rotate and a LOP3 a step, and the LOP3 alone at
p = 24. The target is T_p(A, h) = A | bit_p(h)·2^24 on A < 2^24 and every h. A family is the functions one instruction
computes at every modifier, operand assignment (A, h or a constant in each source; a further register is a constant,
since F must match at every value of it) and constant; each certificate holds for every constant:

- one source (MOV, BREV, IABS, POPC, FLO, I2I, SGXT, BMSK and SHF of one register, SHFL of a register): T reads A and h.
- `bitwise` (LOP3 and PLOP3 at every LUT, ATOM and RED's AND, OR and XOR): output bit 24 reads only bit 24 of its sources.
- `routing` (PRMT at every selector and mode, SHF at every direction, width, signedness, half and shift, constant or a
  register's, SGXT at every width): enumerated, each refuted at a sampled point. PRMT's modes move whole bytes, a subset
  of the default mode's selectors, which are checked byte by byte (an output byte reads its selector nibble alone); a
  selector in h or A is refuted by byte 0.
- `linear` (IADD3 and IADD3.X at every sign and repetition, VIADD, IMAD and IMAD.WIDE's low word with A or a constant
  as the multiplier, IMAD.SHL, LEA, LEA.X, LEA.HI with h as the high word, ATOM and RED's ADD): at a fixed A,
  F = φ + β·h, so F's weight on h's bit i is β·2^i.
- `shift-add` (LEA.HI and LEA.HI.SX32 with h as the low word, SHF.R into an add): F = φ + (h >> k), signed or not, whose
  weights are 2^(i - k) from bit k (the top bit's negated when signed).
- `dot` (IDP.4A and IDP.2A.LO/.HI at every signedness, A or a constant the other operand, the accumulator h or not):
  F = c·h + Σ_f K_f·field_f(h) + const is linear in h's bits too; each field's K is forced by its lowest bit's weight.
- `monotone` (IMAD.HI and IMAD.WIDE's high word with A or a constant as the multiplier, I2I and I2IP's saturating
  conversions, VIMNMX with a constant): each output field is monotone in h. At p ≠ 31 T ignores bit 31 of h, and a
  monotone field invariant under h ↦ h ⊕ 2^31 is constant. At p = 31 T is constant on h < 2^31 and on h ≥ 2^31 and
  jumps by 2^24: a member constant on h < 2^31 is IMAD.HI's with |multiplier| ≤ 2, or a min/max constant everywhere
  (a saturating conversion of h never is), so it moves by at most 3.
- `select` (SEL, VIMNMX and VIMNMX3 at 32 bits and 16x2, RELU included): each output field is a source's field or one of
  two constants. Certificate: a field with three points whose targets are pairwise distinct and none a source's.
- `range` (POPC, FLO, BMSK, ISETP and PLOP3's predicates): outputs in a small set.
- `absdiff` (VABSDIFF4.U8 per byte or summed, into an add that may be h): enumerated over K on the byte holding bit p.

The positive control p = 24: `bitwise` must fail to refute (one LOP3 forms X) and that LOP3 must match T.

Outside the check (printed with the verdict): the floating-point lane instructions, DPX VIADDMNMX, products of h with
itself (IMAD(h, h, ·), IDP(h, h)), and warp instructions reading another lane's register (MATCH, REDUX, VOTE). It
assumes `lane-semantics` (each instruction computes its documented function), and like `derive_minimal.py` it shows
only that X takes two instructions: that the first one's register can't serve two targets is the open sharing lemma.

  uv run --no-project --with-editable . --with-editable benchmarks/pouw/exact_kit python -B
      benchmarks/pouw/theory/derive_minimal_ssn.py [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import functools
import random
import sys

from exact_kit.checks import Checks

M32 = (1 << 32) - 1
LANE = 24
SAMPLES = 1 << 12


def target(p: int, a: int, h: int) -> int:
    return a | ((h >> p) & 1) << LANE


def rotl(x: int, r: int) -> int:
    r %= 32
    return ((x << r) | (x >> (32 - r))) & M32 if r else x


def weights(t, a0) -> list[int]:
    return [(t(a0, 1 << i) - t(a0, 0)) & M32 for i in range(32)]


def is_linear(t, a0, rng) -> bool:
    w = weights(t, a0)
    return all((t(a0, h) - t(a0, 0)) & M32 == sum(w[i] for i in range(32) if h >> i & 1) & M32
               for h in (rng.getrandbits(32) for _ in range(SAMPLES)))


def points(rng, n=8):
    return [(rng.getrandbits(24), rng.getrandbits(32)) for _ in range(n)] + [(0, 0), (0xFFFFFF, M32)]


def one_source(t, a0) -> tuple[bool, str]:
    hs = [t(a0, 1 << i) for i in range(32)] + [t(a0, 0)]
    as_ = {t(a, 0) for a in (0, 1, 0xFFFFFF)}
    return len(set(hs)) > 1 and len(as_) > 1, "T moves with h at a fixed A and with A at a fixed h"


def bitwise(t, a0) -> tuple[bool, str]:
    hs = [0] + [1 << i for i in range(32)]
    for h in hs:
        for h2 in hs:
            diff = t(a0, h) ^ t(a0, h2)
            for i in range(32):
                if not ((h ^ h2) >> i) & 1 and (diff >> i) & 1:
                    return True, f"h {h:#x} and {h2:#x} agree at bit {i}; the target differs there"
    return False, "no witness"


def prmt(x: int, y: int, sel: int) -> int:
    src = x | y << 32
    out = 0
    for k in range(4):
        n = (sel >> 4 * k) & 0xF
        b = (src >> 8 * (n & 7)) & 0xFF
        out |= ((0xFF if b & 0x80 else 0) if n & 8 else b) << 8 * k
    return out


def shf(lo: int, hi: int, s: int, left: bool, signed: bool, high: bool) -> int:
    v = hi << 32 | lo
    if signed and hi >> 31:
        v -= 1 << 64
    r = (v << s) if left else (v >> s)
    return (r >> 32 if high else r) & M32


def sgxt(x: int, w: int, signed: bool) -> int:
    if w >= 32:
        return x
    v = x & ((1 << w) - 1)
    return (v - (1 << w) if signed and w and v >> (w - 1) & 1 else v) & M32


def routing(t, rng) -> tuple[bool, str]:
    pts = points(rng)

    def refuted(f) -> bool:
        return any(f(a, h) != t(a, h) for a, h in pts)

    # a register selector: with h, h's bits 0-3 pick output byte 0 among eight bytes that include A's bytes 0 and 3, 1
    # and 0 at A = 1, while T's byte 0 ignores h; with A, at a fixed h each output byte is one of h's bytes, a constant's
    # four or a sign (0x00, 0xFF), while T's byte 0 takes 256 values over A
    if any(t(1, h) & 0xFF != t(1, 0) & 0xFF for _, h in pts) or len({t(a, 0) & 0xFF for a in range(256)}) <= 7:
        return False, "PRMT with a register selector isn't refuted"
    for order in ((lambda a, h: (a, h)), (lambda a, h: (h, a))):
        live = [[n for n in range(16) if all((prmt(*order(a, h), n << 4 * k) >> 8 * k & 0xFF)
                                             == (t(a, h) >> 8 * k & 0xFF) for a, h in pts)] for k in range(4)]
        if all(live):
            sel = sum(live[k][0] << 4 * k for k in range(4))
            return False, f"PRMT selector {sel:#06x} matches at every point"
    regs = {"A": lambda a, h: a, "h": lambda a, h: h, "0": lambda a, h: 0, "-1": lambda a, h: M32}
    shifts = [(str(s), (lambda s: lambda a, h: s)(s)) for s in range(65)]
    shifts += [(f"{r}&{m}", (lambda r, m: lambda a, h: regs[r](a, h) & m)(r, m)) for r in "Ah" for m in (31, 63)]
    n = 0
    for lo_n, lo in regs.items():
        for hi_n, hi in regs.items():
            for s_n, s in shifts:
                for left in (True, False):
                    for signed in (False, True):
                        for high in (False, True):
                            n += 1
                            f = (lambda lo, hi, s, left, signed, high: lambda a, h: shf(
                                lo(a, h), hi(a, h), s(a, h), left, signed, high))(lo, hi, s, left, signed, high)
                            if not refuted(f):
                                return False, f"SHF lo {lo_n} hi {hi_n} by {s_n} ({left=}, {signed=}, {high=})"
    for x_n in "Ah":
        for w_n, w in shifts[:33] + [(f"{r}&31", (lambda r: lambda a, h: regs[r](a, h) & 31)(r)) for r in "Ah"]:
            for signed in (False, True):
                n += 1
                f = (lambda x, w, signed: lambda a, h: sgxt(x(a, h), w(a, h), signed))(regs[x_n], w, signed)
                if not refuted(f):
                    return False, f"SGXT of {x_n} at width {w_n}"
    return True, f"PRMT byte by byte at both operand orders, {n} SHF and SGXT forms, each refuted at {len(pts)} points"


def linear(w) -> tuple[bool, str]:
    beta = w[0]
    bad = next((i for i in range(32) if (beta * (1 << i)) & M32 != w[i]), None)
    if bad is None:
        return False, f"β {beta:#x} matches"
    return True, f"β is forced to F's bit-0 weight {beta:#x}, then bit {bad}'s weight is {w[bad]:#x}"


def shift_add(w) -> tuple[bool, str]:
    for k in range(32):
        for signed in (False, True):
            u = [0] * k + [1 << (i - k) for i in range(k, 32)]
            if signed:
                u[31] = (-(1 << (31 - k))) & M32
            if u == w:
                return False, f"h >> {k} ({signed=}) matches"
    return True, "no shift k from 0 to 31, signed or not, has T's weights"


#: a dot's fields of h: (lowest bit, width) and the range of the K multiplying each
DOTS = {"IDP.4A, h's bytes": ([(8 * b, 8) for b in range(4)], range(-128, 256)),
        "IDP.2A, h's halves": ([(0, 16), (16, 16)], range(-128, 256)),
        "IDP.2A.LO, h's low bytes": ([(0, 8), (8, 8)], range(-32768, 65536)),
        "IDP.2A.HI, h's high bytes": ([(16, 8), (24, 8)], range(-32768, 65536))}


def dot(w, fields, krange) -> tuple[bool, str]:
    """Some c in {0, 1} and signedness where every field's K, forced by its lowest bit, is out of range or misses."""
    for c in (0, 1):
        for signed in (False, True):
            ok, rest = True, set(range(32))
            for lo, width in fields:
                k = (w[lo] - c * (1 << lo)) & M32
                k = k if k <= krange[-1] else k - (1 << 32)
                bits = range(lo, lo + width)
                rest -= set(bits)
                unit = [(1 << (i - lo)) * (-1 if signed and i == lo + width - 1 else 1) for i in bits]
                if not (krange[0] <= k <= krange[-1]
                        and all((c * (1 << i) + k * u) & M32 == w[i] for i, u in zip(bits, unit))):
                    ok = False
                    break
            if ok and all((c * (1 << i)) & M32 == w[i] for i in rest):
                return False, f"c {c}, signed {signed}: every field has a K"
    return True, "no accumulator and signedness gives every field a K in range"


def monotone(t, a0, p, rng) -> tuple[bool, str]:
    hs = [rng.getrandbits(32) for _ in range(SAMPLES)]
    if p != 31:
        inv = all(t(a0, h) == t(a0, h ^ (1 << 31)) for h in hs)
        return inv and len({t(a0, h) for h in hs}) > 1, "T(h) = T(h ⊕ 2^31) and T isn't constant"
    lo = {t(a0, h & ~(1 << 31)) for h in hs}
    hi = {t(a0, h | (1 << 31)) for h in hs}
    jump = (next(iter(hi)) - next(iter(lo))) & M32 if len(lo) == len(hi) == 1 else None
    return jump is not None and 3 < jump < (1 << 32) - 3, (f"T is constant on each half of h's range and jumps by "
                                                          f"{jump if jump is None else hex(jump)}")


def select(t, rng) -> tuple[bool, str]:
    for width in (32, 16):
        for f0 in range(0, 32, width):
            mask = (1 << width) - 1

            def fld(x):
                return (x >> f0) & mask
            got = {}
            for a, h in [(j << f0 & 0xFFFFFF, 1 << q) for j in range(1, 8) for q in range(32)] + points(rng, 64):
                v = fld(t(a, h))
                if v not in (fld(a), fld(h)):
                    got.setdefault(v, (a, h))
            if len(got) >= 3:
                return True, f"{width}-bit field at bit {f0}: {len(got)} targets none a source's ({sorted(got)[:3]})"
    return False, "no field with three"


def small_range(t, a0) -> tuple[bool, str]:
    masks = {((((1 << w) - 1) << p) | (((1 << w) - 1) >> (32 - p) if p else 0)) & M32 for w in range(33)
             for p in range(32)}
    v = t(0x123456, 0)
    return v not in set(range(33)) | {M32} | masks, f"T(0x123456, 0) = {v:#x}, outside POPC/FLO/BMSK/predicates"


def absdiff(t, p) -> tuple[bool, str]:
    b = p // 8

    def tstep(x):
        return (t(0, x << 8 * b) - t(0, 0)) & M32
    want = [tstep(x) for x in range(256)]
    for per_byte in (False, True):
        wt = 1 << 8 * b if per_byte else 1
        for c in (0, 1):
            for k in [None, *range(256)]:
                if all((c * (x << 8 * b) + (0 if k is None else wt * (abs(x - k) - abs(k)))) & M32 == want[x]
                       for x in range(256)):
                    return False, f"per byte {per_byte}, c {c}, K {k}"
    return True, f"no K, accumulator or layout moves F as T moves over byte {b} of h"


def families(p, rng) -> dict[str, tuple[bool, str]]:
    t = functools.partial(target, p)
    a0 = rng.getrandbits(24)
    w = weights(t, a0)
    fam = {"one-source": one_source(t, a0), "bitwise": bitwise(t, a0), "routing": routing(t, rng),
           "linear": linear(w), "shift-add": shift_add(w), "monotone": monotone(t, a0, p, rng),
           "select": select(t, rng), "range": small_range(t, a0), "absdiff": absdiff(t, p)}
    fam |= {f"dot: {name}": dot(w, fields, kr) for name, (fields, kr) in DOTS.items()}
    return fam


def main() -> int:
    checks = Checks()
    rng = random.Random(20261029)
    pts = points(rng, 4096)
    checks.check("T is the served extraction: A | (rotr(h, (p - 24) mod 32) & 0x01000000) at every p",
                 all(target(p, a, h) == a | (rotl(h, (LANE - p) % 32) & 0x01000000) for p in range(32) for a, h in pts))
    checks.check("honest's two instructions at p ≠ 24 (SHF.L.W h, h by (24 - p) mod 32, then LOP3 A | (R & 2^24)), "
                 "and one LOP3 at p = 24",
                 all(target(p, a, h) == a | (rotl(h, (LANE - p) % 32) & (1 << LANE)) for p in range(32)
                     for a, h in pts) and all(target(LANE, a, h) == a | (h & 1 << LANE) for a, h in pts))
    checks.check("T is linear in h's bits at a fixed A (the solver families' premise)",
                 all(is_linear(functools.partial(target, p), rng.getrandbits(24), rng) for p in range(32)))
    for p in range(32):
        fam = families(p, rng)
        for name, (ok, detail) in fam.items():
            if p == LANE and name == "bitwise":
                checks.check(f"p {p}: the positive control, one LOP3 forms X, so bitwise can't refute it",
                             not ok, detail)
            else:
                checks.check(f"p {p}: no {name} instruction forms X", ok, detail)
    if checks.mutations:
        def w_of(f):
            return weights(f, 0x123456)
        own = {
            "one-source": lambda: one_source(lambda a, h: rotl(h, 7), 0)[0],
            "bitwise": lambda: bitwise(lambda a, h: a ^ h ^ 0x5A5A5A5A, 0)[0],
            "routing (PRMT)": lambda: routing(lambda a, h: prmt(a, h, 0x4210), rng)[0],
            "routing (SHF)": lambda: routing(lambda a, h: shf(a, h, 8, True, False, True), rng)[0],
            "linear": lambda: linear(w_of(lambda a, h: (a + 3 * h) & M32))[0],
            "shift-add": lambda: shift_add(w_of(lambda a, h: (a + (h >> 5)) & M32))[0],
            "dot (IDP.4A)": lambda: dot(w_of(lambda a, h: (a + sum(k * ((h >> 8 * b) & 0xFF) for b, k in
                                                                  enumerate((1, -2, 3, 5)))) & M32),
                                        *DOTS["IDP.4A, h's bytes"])[0],
            "monotone (p ≠ 31)": lambda: monotone(lambda a, h: (a + (h * 5 >> 32)) & M32, 0, 0, rng)[0],
            "monotone (p = 31)": lambda: monotone(lambda a, h: (a + (h >> 31)) & M32, 0, 31, rng)[0],
            "select": lambda: select(lambda a, h: h, rng)[0],
            "range": lambda: small_range(lambda a, h: bin(h).count("1"), 0)[0],
            "absdiff": lambda: absdiff(lambda a, h: (a + abs((h & 0xFF) - 17)) & M32, 0)[0],
        }
        for name, refutes in own.items():
            checks.mutation(f"the {name} family's own function: its certificate refuses to rule it out", not refutes())
    print("outside the check: floating-point lane instructions, DPX VIADDMNMX, products of h with itself, warp "
          "instructions reading another lane's register; assumes lane-semantics; the sharing lemma stays open")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
