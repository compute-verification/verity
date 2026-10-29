#!/usr/bin/env python3
"""OneInstructionMinimal: no single sm_90 integer lane instruction forms a packed X register from A and a raw XOF word.

NCP forms X's first block byte-wise: entry b of a register is `PouwFormX(a_b, PouwNoiseByte(h, σ(b)))`, the int7
activation plus (byte σ(b) of the XOF word h, & 63) − 31, plus 128, as u8 (`verity/core/protocols/pouw/circuit/words.py`).
Honest spends a LOP3 masking h and an add a register. If no one instruction maps (A, h) to the register, the masked noise
is a noise-dependent register the adversary writes too, and the noise's derivation term of `premise_buys.py` is forced.
The check runs at A as committed (int7 two's-complement bytes) and pre-biased (u8 a + 128), at σ the identity and the
byte reversal.

A family is a set of functions F(A, h) one instruction computes, at every modifier, operand assignment (A, h, an XOF
word or a constant in each source; a second register read is a constant here, since F must match at every value of it)
and constant. Each certificate holds for every constant, so nothing sweeps 2^32:

- `bitwise` (LOP3 at every LUT): output bit i reads only bit i of its sources. Certificate: two inputs equal at bit i,
  the target differing there.
- `routing` (PRMT at every selector and mode, SHF and funnel shifts, BREV, moves, byte and bit extracts): an output bit
  is a function of at most one source bit. Certificate: an output bit that two different source bits each flip.
- `affine` (IADD3 and IADD3.X at every sign and repetition, VIADD, LEA, LEA.X and IMAD.SHL at every shift, SHF.L by a
  constant, IMAD and IMAD.WIDE's low word, A·h included, since at a fixed A a product is affine in h): at a fixed A,
  F = φ + β·h mod 2^32. Certificate: the target ignores bit 6 of h, so 64β ≡ 0 and 2^26 | β, and then no β among those
  64 moves the target's low 26 bits as h's bit 0 does.
- `monotone` (IMAD.HI with a constant multiplier or with A at a fixed A, LEA.HI and SHF.R into an add, I2I and I2IP's
  saturating conversions, field by field): each output field is monotone in h, signed or unsigned. Certificate: the
  target ignores bit 31 of h (bit 7 of a byte), and a monotone field invariant under h ↦ h ⊕ 2^31 is constant over
  [h, h + 2^31] for every h, so constant; the target is not.
- `dot` (IDP4A and IDP2A at .LO and .HI, s8/u8 and s16/u16, with A or a constant as the other operand, plus an
  accumulator that may be h): F = c·h + Σ K_j·field_j(h) + const. Certificate: for c ∈ {0, 1}, some byte where no
  coefficient both cancels bit 6's flip and moves the target as bit 0 does.
- `select` (SEL, IMNMX, VIMNMX, VIMNMX3, VIBMNMX, FSEL): the output is a source or one of two constants. Certificate:
  three targets, pairwise distinct, none equal to a source.
- `range` (POPC, FLO, BMSK, ISETP's predicate): outputs in a small set. Certificate: a target outside it.
- `absdiff` (VABSDIFF4 per byte or summed, into an add that may be h): certificate by bytes, as `dot`; `IABS` (|h|):
  the target ignores bit 31, |h| doesn't.

Outside the check (printed with the verdict): the floating-point lane instructions (FADD, FMUL, FFMA, HADD2, HMUL2,
HFMA2, DADD, DMUL, DFMA, MUFU and conversions to or from float), the DPX add-min/max family (VIADDMNMX), 32-bit
VABSDIFF, and every opcode the census lists but this file's families don't name.

  uv run --no-project --with-editable . --with-editable benchmarks/pouw/exact_kit python -B
      benchmarks/pouw/theory/derive_minimal.py [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import functools
import random
import sys

from exact_kit.checks import Checks
from verity.core.protocols.pouw.circuit.words import PouwFormX, PouwNoiseByte

M32 = (1 << 32) - 1
SIGMAS = {"identity": (0, 1, 2, 3), "reversal": (3, 2, 1, 0)}
FORMATS = ("committed int7", "biased u8")
SAMPLES = 1 << 14


def entry(fmt: str, a_byte: int) -> int:
    """The int7 value a register byte holds."""
    v = a_byte - 128 if fmt == "biased u8" else (a_byte & 0x7F) - (a_byte & 0x40) * 2
    return v & 0x7F


def target(fmt: str, sigma, a: int, h: int) -> int:
    return sum(PouwFormX.evaluate(entry(fmt, (a >> 8 * b) & 0xFF), PouwNoiseByte.evaluate(h, sigma[b])) << 8 * b
               for b in range(4))


def zero_a(fmt: str) -> int:
    return 0x80808080 if fmt == "biased u8" else 0


def valid_a(fmt: str, rng: random.Random) -> int:
    vals = [rng.randrange(-64, 64) for _ in range(4)]
    return sum(((v + 128) if fmt == "biased u8" else (v & 0xFF)) << 8 * i for i, v in enumerate(vals))


def bitwise(t, a0) -> tuple[bool, str]:
    for h in range(64):
        for h2 in range(64):
            diff = t(a0, h) ^ t(a0, h2)
            for i in range(32):
                if not ((h ^ h2) >> i) & 1 and (diff >> i) & 1:
                    return True, f"h {h:#x} and {h2:#x} agree at bit {i}; the target differs there"
    return False, "no witness"


def routing(t, a0) -> tuple[bool, str]:
    for j in range(32):
        srcs = {}
        for base in range(64):
            for p in range(32):
                if ((t(a0, base) ^ t(a0, base ^ (1 << p))) >> j) & 1:
                    srcs.setdefault(p, base)
        if len(srcs) >= 2:
            p, q = sorted(srcs)[:2]
            return True, f"output bit {j} flips with h bit {p} (at h {srcs[p]:#x}) and h bit {q} (at h {srcs[q]:#x})"
    return False, "no witness"


def affine(t, a0) -> tuple[bool, str]:
    s1, s64 = (t(a0, 1) - t(a0, 0)) & M32, (t(a0, 64) - t(a0, 0)) & M32
    return (64 * s1 - s64) & M32 != 0, (f"an affine F has F(64) − F(0) = 64·(F(1) − F(0)); the target's steps are "
                                       f"{s1:#x} and {s64:#x}")


def monotone(t, a0, rng: random.Random) -> tuple[bool, str]:
    inv = all(t(a0, h) == t(a0, h ^ (1 << 31)) for h in (rng.getrandbits(32) for _ in range(SAMPLES)))
    return inv and t(a0, 0) != t(a0, 1), (f"t(h) = t(h ⊕ 2^31) at {SAMPLES} random h (and by `PouwNoiseByte`, which "
                                          "reads bits 0–5 of a byte); t(0) ≠ t(1)")


def bytes_cert(t, a0, delta) -> tuple[bool, str]:
    """Some byte b, for each c ∈ {0, 1}, where no coefficient K moves F as the target moves when byte b's bit 0, and
    its bit 6 at every h_b < 64, goes 0 → 1; `delta(c, b, K, bit, hb)` is F's change from h_b = hb."""
    out = []
    for c in (0, 1):
        found = None
        for b in range(4):
            def tstep(bit, hb):
                return (t(a0, (hb | 1 << bit) << 8 * b) - t(a0, hb << 8 * b)) & M32
            want = [tstep(6, hb) for hb in range(64)]
            ks = [k for k in delta.range(b) if delta(c, b, k, 0, 0) & M32 == tstep(0, 0)
                  and all(delta(c, b, k, 6, hb) & M32 == w for hb, w in enumerate(want))]
            if not ks:
                found = b
                break
        if found is None:
            return False, f"c {c}: every byte has a coefficient"
        out.append(f"c {c}: byte {found}")
    return True, "no coefficient at " + ", ".join(out)


class DotLayout:
    """F = c·h + Σ_j K_j·field_j(h): `fields` maps h's byte b to (the bit weight of its bit 0 within its field) or None
    when the dot doesn't read it; K ranges over `krange`."""

    def __init__(self, name: str, weights, krange):
        self.name, self.weights, self.krange = name, weights, krange

    def range(self, b):
        return self.krange if self.weights[b] is not None else (0,)

    def __call__(self, c, b, k, bit, hb):
        w = self.weights[b]
        return c * (1 << (8 * b + bit)) + (k * (w << bit) if w is not None else 0)


DOTS = (DotLayout("IDP4A, h's bytes against s8/u8 K", (1, 1, 1, 1), range(-128, 256)),
        DotLayout("IDP2A, h's halves against bytes", (1, 1 << 8, 1, 1 << 8), range(-128, 256)),
        DotLayout("IDP2A.LO, h's low bytes against halves", (1, 1, None, None), range(-32768, 65536)),
        DotLayout("IDP2A.HI, h's high bytes against halves", (None, None, 1, 1), range(-32768, 65536)))


class AbsLayout:
    """F = c·h + Σ_b |h_b − K_b|·w_b: VABSDIFF4 summed (w_b = 1) or per byte (w_b = 2^8b)."""

    def __init__(self, per_byte: bool):
        self.per_byte = per_byte

    def range(self, b):
        return range(256)

    def __call__(self, c, b, k, bit, hb):
        w = 1 << 8 * b if self.per_byte else 1
        return c * (1 << (8 * b + bit)) + w * (abs((hb | 1 << bit) - k) - abs(hb - k))


def select(t, a0) -> tuple[bool, str]:
    hs = [h for h in range(256) if t(a0, h) not in (a0, h)]
    vals = sorted({t(a0, h) for h in hs})
    return len(vals) >= 3, f"{len(vals)} distinct targets none equal to a source (A {a0:#x}, h < 256)"


def small_range(t, a0) -> tuple[bool, str]:
    masks = {((((1 << w) - 1) << p) | (((1 << w) - 1) >> (32 - p) if p else 0)) & M32 for w in range(33)
             for p in range(32)}
    rng_set = set(range(33)) | {M32} | masks
    v = t(a0, 0)
    return v not in rng_set, f"target {v:#x} outside POPC/FLO's [0, 32] ∪ {{-1}} and BMSK's {len(masks)} masks"


def iabs(t, a0) -> tuple[bool, str]:
    def abs32(x):
        return (-x if x & (1 << 31) else x) & M32
    return (t(a0, 0) == t(a0, 1 << 31) and abs32(0) != abs32(1 << 31),
            "IABS(h) at h 0 and 2^31 differs (0, 2^31); the target doesn't")


def main() -> int:
    checks = Checks()
    rng = random.Random(20261029)
    checks.check("the target is words.py's: PouwFormX(-64, PouwNoiseByte(0x3f, 0)) = 96",
                 PouwFormX.evaluate(0x40, PouwNoiseByte.evaluate(0x3F, 0)) == 96)
    a_c = valid_a("committed int7", rng)
    a_b = a_c ^ 0x80808080
    checks.check("the biased format is the committed one with each byte's sign bit flipped (a + 128)",
                 all(target("committed int7", SIGMAS["identity"], a_c, h) == target("biased u8", SIGMAS["identity"],
                                                                                      a_b, h)
                     for h in (rng.getrandbits(32) for _ in range(256))))
    honest = all(target(f, s, a, h) == ((((a ^ 0x80808080) if f == "committed int7" else a)
                                         + (h_perm & 0x3F3F3F3F) - 0x1F1F1F1F) & M32)
                 for f in FORMATS for s in SIGMAS.values()
                 for a, h in ((valid_a(f, rng), rng.getrandbits(32)) for _ in range(512))
                 for h_perm in (sum(((h >> 8 * s[b]) & 0xFF) << 8 * b for b in range(4)),))
    checks.check("honest's two instructions: X = A' + (h & 0x3F3F3F3F) − 0x1F1F1F1F with no carry between bytes "
                 "(A' = A biased; committed A needs A ^ 0x80808080 first, a salt-independent LOP3)", honest)
    verdicts = {}
    for f in FORMATS:
        for sname, s in SIGMAS.items():
            t = functools.partial(target, f, s)
            a0 = zero_a(f)
            fam = {"bitwise": bitwise(t, a0), "routing": routing(t, a0), "affine": affine(t, a0),
                   "monotone": monotone(t, a0, rng), "select": select(t, a0), "range": small_range(t, a0),
                   "absdiff summed": bytes_cert(t, a0, AbsLayout(False)),
                   "absdiff per byte": bytes_cert(t, a0, AbsLayout(True)), "IABS": iabs(t, a0)}
            fam |= {f"dot: {d.name}": bytes_cert(t, a0, d) for d in DOTS}
            verdicts[(f, sname)] = fam
            for name, (ok, detail) in fam.items():
                checks.check(f"{f}, σ {sname}: no {name} instruction forms the register", ok, detail)
    if checks.mutations:
        def ub(h, b):
            return (h >> 8 * b) & 0xFF
        own = {"bitwise": (bitwise, lambda a, h: a ^ h ^ 0x5A5A5A5A),
               "routing": (routing, lambda a, h: ((h << 5) | (h >> 27)) & M32),
               "affine": (affine, lambda a, h: (a + h - 0x1F1F1F1F) & M32),
               "monotone": (lambda t, a0: monotone(t, a0, rng), lambda a, h: (a + (h >> 3)) & M32),
               "dot (IDP4A)": (lambda t, a0: bytes_cert(t, a0, DOTS[0]),
                               lambda a, h: (a + sum(k * ub(h, b) for b, k in enumerate((1, -2, 3, 5)))) & M32),
               "select": (select, lambda a, h: h),
               "range": (small_range, lambda a, h: bin(h).count("1")),
               "absdiff summed": (lambda t, a0: bytes_cert(t, a0, AbsLayout(False)),
                                  lambda a, h: (a + sum(abs(ub(h, b) - 17) for b in range(4))) & M32),
               "IABS": (iabs, lambda a, h: (-h if h & (1 << 31) else h) & M32)}
        for name, (cert, member) in own.items():
            checks.mutation(f"the {name} family's own function: its certificate refuses to rule it out",
                            not cert(member, 0)[0])
    print("outside the check: floating-point lane instructions (FADD, FMUL, FFMA, HADD2, HMUL2, HFMA2, DADD, DMUL, "
          "DFMA, MUFU, float conversions), DPX VIADDMNMX, 32-bit VABSDIFF, and any census opcode the families don't "
          "name")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
