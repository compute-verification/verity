"""Conformance vectors for `Definitions.Pouw.Fp8Atom`: `HOPPER_E4M3_K32` steps computed by `verity.ml.circuits.silicon`, as Lean data.

    PYTHONPATH=<verity> python scripts/fp8atom_vectors.py <verity checkout> [--out DIR]

Every expected word is `verity.ml.circuits.silicon.total_fp8.tc_dot_total_e4m3(HOPPER_E4M3_K32, acc, a, b)`; on finite inputs it is
also checked against `HOPPER_E4M3_K32.step`. The chained steps of `verity/ml/circuits/silicon/captures/vectors/fp8-hopper-wgmma-k{32,1536}.json`
(model replays, not captures) are added as they stand. The kernel pins a subset; the Lean audit runs the full sets as a
compiled test (`Proofs/Pouw/Bulk.lean`, under `runs` in `lean-audit.json`), which is not a proof.

Kernel-pinned, as Lean data (their checks, one theorem per part, in `Proofs/Pouw/Fp8Atom/Checks/`):
* `Definitions/Pouw/Fp8Atom/Vectors/C<nn>.lean` and `Definitions/Pouw/Fp8Atom/Vectors.lean`: the steps `fp8atom_cover.json`
  selects (two greedy branch covers of the 10,549 generated steps) and every reference step;
* `Definitions/Pouw/Fp8Atom/Vectors/Add.lean`: the FP32 adds `fp8atom_cover.json` selects (a greedy cover of the 5,000);
* `Definitions/Pouw/Fp8Atom/Vectors/Ada.lean`: the RTX 4090 captures `golden/ada_e4m3_m16n8k32.json` (real silicon, the
  same adder with two groups of 16), for `adaStep`.

The bulk, as gzipped text with the SHA-256 of each uncompressed file in `conformance/SHA256SUMS` (`runs` in
`lean-audit.json` pins them; `verity/core/protocols/pouw/tests/test_lean_package.py` checks that this script still writes them):
* `conformance/steps.txt.gz`: the 10,549 steps, then red-team R1's 166,000 differential steps (`R1_FAMILIES`);
* `conformance/adds.txt.gz`: the 5,000 FP32 adds.

`Definitions/Pouw/Fp8Atom/Vectors/coverage.json` records the families, what they reach, and what is pinned.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from verity.ml.circuits.silicon import models, term
from verity.ml.circuits.silicon.total_fp8 import tc_dot_total_e4m3

P = models.HOPPER_E4M3_K32
ADA = models.ADA_E4M3_M16N8K32
SEED = 2026092819
CHUNK = 500
PART = 100
NAN_CODES = (0x7F, 0xFF)
FINITE = [c for c in range(256) if c & 0x7F != 0x7F]
SUBNORMAL = [c for c in range(256) if (c >> 3) & 0xF == 0 and c & 7]
BIG = [c for c in FINITE if (c >> 3) & 0xF >= 13]


def fp32(sign: int, field: int, man: int) -> int:
    return (sign << 31) | (field << 23) | man


def rand_word(rng: random.Random, lo: int = 1, hi: int = 254) -> int:
    return fp32(rng.getrandbits(1), rng.randint(lo, hi), rng.getrandbits(23))


def nan_word(rng: random.Random) -> int:
    return fp32(rng.getrandbits(1), 255, rng.randint(1, (1 << 23) - 1))


def ops(rng: random.Random, pool, n: int = 32) -> list[int]:
    return [rng.choice(pool) for _ in range(n)]


def mixed(rng: random.Random) -> int:
    u = rng.random()
    if u < 0.10:
        return rng.choice((0x00, 0x80))
    if u < 0.25:
        return rng.choice(SUBNORMAL)
    if u < 0.30:
        return rng.choice((0x7E, 0xFE))
    return rng.choice(FINITE)


def family_random(rng):
    return 0, ops(rng, FINITE), ops(rng, FINITE)


def family_mixed(rng):
    return 0, [mixed(rng) for _ in range(32)], [mixed(rng) for _ in range(32)]


def family_acc(rng):
    """Nonzero finite accumulators, with exponents around the products' (so the accumulator is truncated too)."""
    a, b = ops(rng, FINITE), ops(rng, FINITE)
    u = rng.random()
    if u < 0.6:
        acc = rand_word(rng, 127 - 20, 127 + 20)
    elif u < 0.8:
        acc = rand_word(rng)
    else:
        acc = rand_word(rng, 127 - 12, 127 + 12)
        k = rng.randrange(32)
        a[k:] = [0] * (32 - k)
    return acc, a, b


def family_special(rng):
    """NaN operands, NaN and infinite accumulators, signed zeros, all-448 lanes."""
    a, b = ops(rng, FINITE), ops(rng, FINITE)
    kind = rng.randrange(10)
    if kind == 0:
        a[rng.randrange(32)] = rng.choice(NAN_CODES)
        acc = rand_word(rng) if rng.random() < 0.5 else 0
    elif kind == 1:
        b[rng.randrange(32)] = rng.choice(NAN_CODES)
        acc = fp32(rng.getrandbits(1), 255, 0)
    elif kind == 2:
        acc = nan_word(rng)
    elif kind == 3:
        acc = fp32(rng.getrandbits(1), 255, 0)
    elif kind == 4:
        acc = fp32(1, 0, 0)
        a = [rng.choice((0x00, 0x80)) for _ in range(32)]
    elif kind == 5:
        acc = rng.choice((0, 1 << 31))
        a = [rng.choice((0x7E, 0xFE)) for _ in range(32)]
        b = [0x7E] * 32
    elif kind == 6:
        acc = fp32(rng.getrandbits(1), 254, (1 << 23) - 1)
    elif kind == 7:
        for i in rng.sample(range(32), rng.randint(1, 4)):
            (a if rng.random() < 0.5 else b)[i] = rng.choice(NAN_CODES)
        acc = rng.choice((nan_word(rng), fp32(rng.getrandbits(1), 255, 0), rand_word(rng)))
    elif kind == 8:
        acc = rng.choice((1, 1 << 31 | 1, fp32(0, 1, 0), fp32(1, 1, 0)))
        a = [rng.choice((0x00, 0x80)) for _ in range(32)]
    else:
        acc = rng.choice((0, 1 << 31))
        a = [0x7E] * 32
        b = [0x7E] * 32
        if rng.random() < 0.5:
            b[rng.randrange(32)] = 0xFE
    return acc, a, b


def family_subnormal(rng):
    """Subnormal operands and FP32 subnormal or tiny accumulators (the denormalising shift and the floor)."""
    kind = rng.randrange(4)
    a = ops(rng, SUBNORMAL + [0x00, 0x80])
    b = ops(rng, SUBNORMAL + [0x00, 0x80, 0x08, 0x88])
    if kind == 0:
        acc = fp32(rng.getrandbits(1), 0, rng.randint(1, (1 << 23) - 1))
        a = [rng.choice((0x00, 0x80)) for _ in range(32)]
    elif kind == 1:
        acc = fp32(rng.getrandbits(1), rng.randint(0, 12), rng.getrandbits(23))
    elif kind == 2:
        acc = fp32(rng.getrandbits(1), rng.randint(127 - 30, 127 - 5), rng.getrandbits(23))
    else:
        acc = 0
    return acc, a, b


def family_saturation(rng):
    """Accumulators near the top of the FP32 range with large products: overflow to a sticky infinity, or not."""
    a, b = ops(rng, BIG), ops(rng, BIG)
    sign = rng.getrandbits(1)
    if rng.random() < 0.7:
        a = [x & 0x7F | (sign << 7) for x in a]
        b = [x & 0x7F for x in b]
    acc = fp32(sign if rng.random() < 0.8 else 1 - sign, rng.randint(250, 254), rng.getrandbits(23))
    if rng.random() < 0.3:
        a = [rng.choice((0x00, 0x80)) for _ in range(32)]
        acc = fp32(rng.getrandbits(1), 254, rng.randint((1 << 23) - 64, (1 << 23) - 1))
    return acc, a, b


def family_cancel(rng):
    """Products in opposite-sign pairs, with a small residue or none; accumulators that cancel the products."""
    a, b = [0] * 32, [0] * 32
    pairs = rng.randint(8, 16)
    idx = rng.sample(range(32), 2 * pairs)
    for j in range(pairs):
        x, y = rng.choice(FINITE), rng.choice(FINITE)
        i1, i2 = idx[2 * j], idx[2 * j + 1]
        a[i1], b[i1] = x, y
        a[i2], b[i2] = x, y ^ 0x80
    rest = [i for i in range(32) if i not in idx]
    if rng.random() < 0.5:
        for i in rest:
            a[i], b[i] = rng.choice(SUBNORMAL + [0]), rng.choice(FINITE)
    acc = 0
    u = rng.random()
    if u < 0.3:
        total = sum(_value(a[i]) * _value(b[i]) for i in range(32))
        acc = _fp32_of(-total)
    elif u < 0.6:
        acc = rand_word(rng, 127 - 10, 127 + 10)
    return acc, a, b


def family_wide(rng):
    """Products spanning the whole exponent range: small terms truncated away by the alignment."""
    a = [rng.choice(BIG) if rng.random() < 0.2 else rng.choice(SUBNORMAL) for _ in range(32)]
    b = [rng.choice(BIG) if rng.random() < 0.2 else rng.choice(SUBNORMAL + FINITE) for _ in range(32)]
    acc = rand_word(rng, 127 - 40, 127 + 30) if rng.random() < 0.5 else 0
    return acc, a, b


def _value(code: int) -> float:
    d = term.decode_e4m3(code)
    s, e, m = d
    return (-1) ** s * m * 2.0 ** e


def _fp32_of(x: float) -> int:
    import struct

    return struct.unpack("<I", struct.pack("<f", x))[0]


FAMILIES = [
    ("random", family_random, 1500),
    ("mixed", family_mixed, 1000),
    ("accumulator", family_acc, 2000),
    ("special", family_special, 1500),
    ("subnormal", family_subnormal, 1500),
    ("saturation", family_saturation, 1000),
    ("cancellation", family_cancel, 1500),
    ("wide", family_wide, 500),
]


# Red-team R1's differential families (`internal/pouw-fp8/lean-atom-r1-diff-gen.py`, 29 Sep 00:15Z), in the bulk only.
# Each draws from its own generator in the order R1's script does, so the steps are R1's.
R1_SEED = 20260929


def _max_exp(a, b):
    es = [t.exponent for t in (term.e4m3_product(x, y) for x, y in zip(a, b)) if t.magnitude]
    return max(es) if es else None


def r1_acc_edge(rng):
    a = [rng.choice(FINITE) for _ in range(32)]
    b = [rng.choice(FINITE) for _ in range(32)]
    E = _max_exp(a, b)
    E = 0 if E is None else E
    e = max(1, min(254, E + 127 + rng.randint(-16, 3)))
    low = rng.choice((0, 0x3FF, 0x200, 0x1FF, 0x400, rng.getrandbits(10)))
    m = (rng.getrandbits(13) << 10) | low
    return fp32(rng.getrandbits(1), e, m & 0x7FFFFF), a, b


def r1_signmix(rng):
    s = rng.getrandbits(1)
    a = [rng.choice(FINITE) & 0x7F | (s << 7) for _ in range(32)]
    b = [rng.choice(FINITE) & 0x7F for _ in range(32)]
    E = _max_exp(a, b) or 0
    return fp32(1 - s, max(1, min(254, E + 127 + rng.randint(-2, 2))), rng.getrandbits(23)), a, b


def r1_carry(rng):
    c = rng.choice([x for x in FINITE if (x >> 3) & 0xF >= 8 and x & 0x80 == 0])
    b = [rng.choice((0x38, 0x3F, 0x3E, 0x39)) for _ in range(32)]
    acc = 0 if rng.random() < 0.5 else fp32(0, 127 + rng.randint(0, 12), rng.getrandbits(23))
    return acc, [c] * 32, b


def r1_acc_only(rng):
    a = [rng.choice((0x00, 0x80)) for _ in range(32)]
    b = [rng.choice(FINITE) for _ in range(32)]
    k = rng.randrange(4)
    if k == 0:
        acc = fp32(rng.getrandbits(1), 0, rng.getrandbits(23))
    elif k == 1:
        acc = fp32(rng.getrandbits(1), rng.randint(1, 3), rng.getrandbits(23))
    elif k == 2:
        acc = rng.choice((0, 1 << 31, 0x3FF, 0x800003FF, 0x400, 0x80000400, 0x00800000, 0x807FFFFF))
    else:
        acc = fp32(rng.getrandbits(1), rng.randint(1, 254), rng.getrandbits(23))
    return acc, a, b


def r1_h1like(rng):
    base = rng.randint(1, 12)
    a = [rng.choice([x for x in FINITE if abs(((x >> 3) & 0xF) - base) <= 2]) for _ in range(32)]
    b = [rng.choice(FINITE) for _ in range(32)]
    t = rng.randrange(32)
    a[t] = rng.choice((0x7E, 0xFE, 0x76, 0xF6))
    return 0, a, b


def r1_small(rng):
    a = [rng.choice(SUBNORMAL + [0x08, 0x88, 0x09, 0x89]) for _ in range(32)]
    b = [rng.choice(SUBNORMAL + [0x08, 0x88]) for _ in range(32)]
    acc = 0 if rng.random() < 0.5 else fp32(rng.getrandbits(1), rng.randint(0, 110), rng.getrandbits(23))
    return acc, a, b


def r1_huge_acc(rng):
    a = [rng.choice(FINITE) for _ in range(32)]
    b = [rng.choice(FINITE) for _ in range(32)]
    return fp32(rng.getrandbits(1), rng.randint(140, 254), rng.getrandbits(23)), a, b


def r1_nan_payload(rng):
    a = [rng.choice(FINITE) for _ in range(32)]
    b = [rng.choice(FINITE) for _ in range(32)]
    return fp32(rng.getrandbits(1), 255, rng.choice((1, 0x400000, 0x7FFFFF, rng.getrandbits(23) | 1))), a, b


R1_FAMILIES = [
    ("r1_acc_edge", r1_acc_edge, 40000),
    ("r1_signmix", r1_signmix, 30000),
    ("r1_carry", r1_carry, 10000),
    ("r1_acc_only", r1_acc_only, 20000),
    ("r1_h1like", r1_h1like, 40000),
    ("r1_small", r1_small, 20000),
    ("r1_huge_acc", r1_huge_acc, 5000),
    ("r1_nan_payload", r1_nan_payload, 1000),
]


def expected(acc: int, a, b) -> int:
    d = tc_dot_total_e4m3(P, acc, a, b)
    finite = all(x & 0x7F != 0x7F for x in (*a, *b)) and (acc >> 23) & 0xFF != 0xFF
    if finite:
        assert d == P.step(acc, a, b), (acc, a, b)
    return d


def out_class(w: int) -> str:
    e, m = (w >> 23) & 0xFF, w & 0x7FFFFF
    if w == 0x7FFFFFFF:
        return "nan"
    if e == 0xFF:
        return "-inf" if w >> 31 else "+inf"
    if w == 0:
        return "+0"
    if w == 1 << 31:
        return "-0"
    return "subnormal" if e == 0 else "normal"


def reference_steps(root: Path) -> list[tuple[str, int, list[int], list[int], int]]:
    out = []
    for name in ("fp8-hopper-wgmma-k32", "fp8-hopper-wgmma-k1536"):
        d = json.loads((root / "verity/ml/circuits/silicon/captures/vectors" / f"{name}.json").read_text())
        a, b = [int(x, 16) for x in d["a"]], [int(x, 16) for x in d["b"]]
        for s, step in enumerate(d["steps"]):
            acc, got = int(step["c_in"], 16), int(step["c_out"], 16)
            sa, sb = a[32 * s : 32 * s + 32], b[32 * s : 32 * s + 32]
            assert expected(acc, sa, sb) == got, (name, s)
            out.append((f"ref:{name}", acc, sa, sb, got))
    return out


def pack(codes) -> int:
    return sum(c << (8 * i) for i, c in enumerate(codes))


def lean_vec(acc, a, b, d) -> str:
    return f"⟨0x{acc:08X}, 0x{pack(a):064X}, 0x{pack(b):064X}, 0x{d:08X}⟩"


def header(module: str) -> str:
    return (f"import Definitions.Pouw.Fp8Atom.{module}\n\n"
            "/-! Generated by `scripts/fp8atom_vectors.py`; do not edit. -/\n\n"
            "namespace Pouw.Fp8Atom.Vectors\n\n")


def write_chunk(path: Path, name: str, doc: str, vecs, typ: str = "Vec", fmt=lean_vec, imp: str = "Atom") -> list[str]:
    """One module: parts of `PART` vectors (`<name>p<i>`), each checked by its own kernel declaration, and `<name>`."""
    parts = []
    out = header(imp)
    for i in range(0, len(vecs), PART):
        part = f"{name}p{i // PART}"
        parts.append(part)
        body = ",\n  ".join(fmt(*v) for v in vecs[i : i + PART])
        out += f"/-- {doc}, part {i // PART}. -/\ndef {part} : List {typ} := [\n  {body}]\n\n"
    out += f"/-- {doc} -/\ndef {name} : List {typ} := " + " ++ ".join(parts) + "\n\nend Pouw.Fp8Atom.Vectors\n"
    path.write_text(out)
    return parts


def write_check(path: Path, module: str, name: str, parts: list[str], pred: str, what: str) -> None:
    out = (f"import Definitions.Pouw.Fp8Atom.Vectors.{module}\n\n"
           "/-! Generated by `scripts/fp8atom_vectors.py`; do not edit. -/\n\n"
           f"namespace Pouw.Fp8Atom.Checks\n\n")
    for part in parts:
        out += (f"/-- {what}, part {part}, evaluated by the kernel. -/\n"
                f"theorem {part} : Vectors.{part}.all ({pred}) = true := by decide +kernel\n\n")
    rules = ", ".join(parts) if len(parts) == 1 else "List.all_append, " + ", ".join(parts) + ", Bool.and_self"
    out += (f"/-- {what}. -/\ntheorem {name} : Vectors.{name}.all ({pred}) = true := by\n"
            f"  simp only [Vectors.{name}, {rules}]\n\n"
            "end Pouw.Fp8Atom.Checks\n")
    path.write_text(out)


# The FP32 add: IEEE binary32 addition (numpy, round to nearest even, subnormals kept) on finite words whose exact sum
# is zero or normal, cross-checked against an exact rounding to 24 significant bits.

def f32(w: int) -> np.float32:
    return np.array([w], dtype=np.uint32).view(np.float32)[0]


def word(x: np.float32) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def rn24(x: Fraction) -> Fraction:
    if x == 0:
        return x
    k = abs(x).numerator.bit_length() - abs(x).denominator.bit_length()
    k = k - 1 if abs(x) < Fraction(2) ** k else k
    u = Fraction(2) ** (k - 23)
    m = x / u
    q = math.floor(m)
    r = m - q
    return (q + (1 if r > Fraction(1, 2) or (r == Fraction(1, 2) and q % 2) else 0)) * u


def normal_word(rng, lo: int = 1, hi: int = 254, sign=None) -> int:
    s = rng.randrange(2) if sign is None else sign
    return fp32(s, rng.randint(lo, hi), rng.randrange(1 << 23))


def add_pair(rng, fam: str) -> tuple[int, int]:
    if fam == "add_random":
        return normal_word(rng), normal_word(rng)
    if fam == "add_close":
        a = normal_word(rng, 2, 250)
        e = (a >> 23) & 0xFF
        return a, normal_word(rng, max(1, e - 3), min(254, e + 3))
    if fam == "add_cancel":
        a = normal_word(rng, 2, 254)
        return a, (a ^ (1 << 31)) + rng.randint(-40, 40) if rng.random() < 0.8 else a ^ (1 << 31)
    if fam == "add_tie":
        a = normal_word(rng, 30, 250)
        e = (a >> 23) & 0xFF
        j = rng.randrange(1 << rng.randint(0, 20))
        b = word(np.float32((2 * j + 1) * 2.0 ** (e - 127 - 24 - rng.randint(0, 2))))
        return a, b ^ (rng.randrange(2) << 31)
    if fam == "add_absorb":
        a = normal_word(rng, 40, 250)
        e = (a >> 23) & 0xFF
        return a, normal_word(rng, max(1, e - 26 - rng.randint(0, 20)), e - 24)
    if fam == "add_zero":
        z = rng.choice((0, 1 << 31))
        x = normal_word(rng) if rng.random() < 0.8 else fp32(rng.randrange(2), 0, rng.randrange(1, 1 << 23))
        return (z, x) if rng.random() < 0.5 else (x, z)
    if fam == "add_subnormal_in":
        return fp32(rng.randrange(2), 0, rng.randrange(1, 1 << 23)), normal_word(rng, 1, 30)
    if fam == "add_big":
        return normal_word(rng, 240, 254), normal_word(rng, 200, 254)
    if fam == "add_binade":
        k = rng.randint(30, 250)
        a = fp32(rng.randrange(2), k, rng.choice((0, 0, 1, (1 << 23) - 1)))
        return a, normal_word(rng, k - 26, k - 22)
    raise ValueError(fam)


ADD_FAMILIES = [("add_random", 1000), ("add_close", 800), ("add_cancel", 600), ("add_tie", 600), ("add_absorb", 600),
                ("add_zero", 300), ("add_subnormal_in", 300), ("add_big", 300), ("add_binade", 500)]
MIN_NORMAL = Fraction(2) ** -126


def add_vectors(rng) -> list[tuple[str, int, int, int]]:
    rows = []
    for fam, count in ADD_FAMILIES:
        n = 0
        while n < count:
            a, b = add_pair(rng, fam)
            x, y = f32(a), f32(b)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            with np.errstate(over="ignore"):
                z = np.float32(x + y)
            s = Fraction(float(x)) + Fraction(float(y))
            if not np.isfinite(z) or (s != 0 and abs(s) < MIN_NORMAL):
                continue
            assert Fraction(float(z)) == rn24(s), (fam, hex(a), hex(b))
            rows.append((fam, a, b, word(z)))
            n += 1
    return rows


def lean_add(a, b, c) -> str:
    return f"⟨0x{a:08X}, 0x{b:08X}, 0x{c:08X}⟩"


def add_class(a: int, b: int, c: int) -> str:
    x, y, z = (Fraction(float(f32(w))) for w in (a, b, c))
    if z == 0:
        return "zero"
    if z == x + y:
        return "exact"
    if z in (x, y):
        return "absorbed"
    lo = abs(x + y)
    k = lo.numerator.bit_length() - lo.denominator.bit_length()
    k = k - 1 if lo < Fraction(2) ** k else k
    return "tie" if ((x + y) / Fraction(2) ** (k - 24)).denominator == 1 else "rounded"


def write_bulk(path: Path, lines: list[str]) -> str:
    """Gzip `lines` reproducibly (no name, no time stamp); return the SHA-256 of the uncompressed text."""
    text = "".join(line + "\n" for line in lines).encode()
    with open(path, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as g:
        g.write(text)
    return hashlib.sha256(text).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[6] / "Security")
    args = ap.parse_args()
    rng = random.Random(SEED)
    rows = []
    for fam, gen, count in FAMILIES:
        for _ in range(count):
            acc, a, b = gen(rng)
            rows.append((fam, acc, a, b, expected(acc, a, b)))
    rows += reference_steps(args.verity)
    adds = add_vectors(rng)
    r1rng = random.Random(R1_SEED)
    r1rows = []
    for fam, gen, count in R1_FAMILIES:
        for _ in range(count):
            acc, a, b = gen(r1rng)
            r1rows.append((fam, acc, a, b, expected(acc, a, b)))

    cover = json.loads((Path(__file__).resolve().parent / "fp8atom_cover.json").read_text())
    keep = {i for c in cover["steps"].values() for i in c["indices"]}
    pinned = [r for i, r in enumerate(rows) if i in keep or r[0].startswith("ref:")]
    refs = sum(1 for r in rows if r[0].startswith("ref:"))
    pinned_adds = [adds[i] for i in sorted(cover["adds"]["indices"])]

    vdir = args.out / "Definitions/Pouw/Fp8Atom/Vectors"
    cdir = args.out / "Proofs/Pouw/Fp8Atom/Checks"
    vdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    for old in (*vdir.glob("C*.lean"), *cdir.glob("C*.lean")):
        old.unlink()
    names = []
    for c in range(0, len(pinned), CHUNK):
        name = f"c{c // CHUNK:02d}"
        names.append(name)
        span = f"Vectors {c} to {min(c + CHUNK, len(pinned)) - 1}"
        parts = write_chunk(vdir / f"C{c // CHUNK:02d}.lean", name, span, [r[1:] for r in pinned[c : c + CHUNK]])
        write_check(cdir / f"C{c // CHUNK:02d}.lean", f"C{c // CHUNK:02d}", name, parts, "Vec.holds hopperStep",
                    span + " hold for `hopperStep`")
    imports = "".join(f"import Definitions.Pouw.Fp8Atom.Vectors.C{i:02d}\n" for i in range(len(names)))
    (args.out / "Definitions/Pouw/Fp8Atom/Vectors.lean").write_text(
        imports + "\n/-! Generated by `scripts/fp8atom_vectors.py`; do not edit. -/\n\nnamespace Pouw.Fp8Atom\n\n"
        f"/-- The {len(pinned)} kernel-pinned conformance vectors: `verity.ml.circuits.silicon`'s `HOPPER_E4M3_K32` words on the "
        f"branch covers of `scripts/fp8atom_cover.json` and the {refs} reference steps (`scripts/fp8atom_vectors.py`, "
        f"seed {SEED}). -/\ndef vectors : List Vec :=\n  " + " ++\n  ".join(f"Vectors.{n}" for n in names)
        + "\n\nend Pouw.Fp8Atom\n")

    golden = json.loads((args.verity / "verity/ml/circuits/ops/tests/fixtures/golden/ada_e4m3_m16n8k32.json").read_text())
    ada = []
    for rec in golden["records"]:
        a = [int(rec["a"][i : i + 2], 16) for i in range(0, 64, 2)]
        b = [int(rec["b"][i : i + 2], 16) for i in range(0, 64, 2)]
        acc, d = int(rec["acc"], 16), int(rec["d"], 16)
        assert ADA.step(acc, a, b) == d
        ada.append((acc, a, b, d))
    parts = write_chunk(vdir / "Ada.lean", "ada",
                        f"The {len(ada)} RTX 4090 captures of `mma.sync.m16n8k32.e4m3` (`golden/ada_e4m3_m16n8k32.json`)", ada)
    write_check(cdir / "Ada.lean", "Ada", "ada", parts, "Vec.holds adaStep", f"The {len(ada)} RTX 4090 captures hold for `adaStep`")
    parts = write_chunk(vdir / "Add.lean", "add",
                        f"The {len(pinned_adds)} kernel-pinned FP32 add vectors (IEEE binary32, seed {SEED}; the cover "
                        "of `scripts/fp8atom_cover.json`)", [r[1:] for r in pinned_adds], "AddVec", lean_add, "Fp32")
    write_check(cdir / "Add.lean", "Add", "add", parts, "AddVec.holds",
                f"The {len(pinned_adds)} FP32 add vectors hold for `rnAdd`")

    bdir = args.out / "conformance"
    bdir.mkdir(exist_ok=True)
    sums = {
        "steps.txt": write_bulk(bdir / "steps.txt.gz", [f"{acc:08x} {pack(a):064x} {pack(b):064x} {d:08x} {fam}"
                                                        for fam, acc, a, b, d in rows + r1rows]),
        "adds.txt": write_bulk(bdir / "adds.txt.gz", [f"{a:08x} {b:08x} {c:08x} {fam}" for fam, a, b, c in adds]),
    }
    (bdir / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    (args.out / "Proofs/Pouw/Fp8Atom/Checks.lean").write_text(
        "".join(f"import Proofs.Pouw.Fp8Atom.Checks.C{i:02d}\n" for i in range(len(names)))
        + "import Proofs.Pouw.Fp8Atom.Checks.Ada\nimport Proofs.Pouw.Fp8Atom.Checks.Add\n\n"
        "/-! Generated by `scripts/fp8atom_vectors.py`; do not edit. -/\n")

    cov = {"seed": SEED, "total": len(rows), "chunks": len(names), "ada_captures": len(ada), "families": {}}
    fams = Counter(r[0] for r in rows)
    for fam in fams:
        sel = [r for r in rows if r[0] == fam]
        cov["families"][fam] = {
            "steps": len(sel),
            "outputs": dict(Counter(out_class(r[4]) for r in sel)),
            "nonzero_acc": sum(1 for r in sel if r[1] & 0x7FFFFFFF),
            "nan_operand": sum(1 for r in sel if any(x & 0x7F == 0x7F for x in (*r[2], *r[3]))),
        }
    cov["outputs"] = dict(Counter(out_class(r[4]) for r in rows))
    cov["add"] = {"total": len(adds), "families": {f: dict(Counter(add_class(*r[1:]) for r in adds if r[0] == f))
                                                   for f, _ in ADD_FAMILIES},
                  "outputs": dict(Counter(add_class(*r[1:]) for r in adds))}
    cov["pinned"] = {"steps": len(pinned), "reference_steps": refs,
                     "covers": {k: len(c["indices"]) for k, c in cover["steps"].items()},
                     "families": dict(Counter(r[0] for r in pinned)),
                     "outputs": dict(Counter(out_class(r[4]) for r in pinned)),
                     "adds": len(pinned_adds), "add_outputs": dict(Counter(add_class(*r[1:]) for r in pinned_adds))}
    cov["bulk"] = {"steps": len(rows) + len(r1rows), "adds": len(adds),
                   "r1_families": {f: {"steps": n,
                                       "outputs": dict(Counter(out_class(r[4]) for r in r1rows if r[0] == f))}
                                   for f, _, n in R1_FAMILIES},
                   "sha256": sums}
    (vdir / "coverage.json").write_text(json.dumps(cov, indent=1) + "\n")
    json.dump(cov, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
