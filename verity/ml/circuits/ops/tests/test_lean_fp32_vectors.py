"""The Lean FP32 arithmetic (`Security/Definitions/Core/Fp32.lean`) is held to `verity.ml.circuits.ops.fp32` by vectors:
`Security/Proofs/Core/Fp32/Vectors.lean` states each one as a theorem the kernel checks against the Lean, and this test
checks the same vectors against the Python, and that they reach every case the Python distinguishes. A change to either
side's semantics then fails a build.

Run as a script, this file writes the vectors from the Python: hand-picked words (NaNs of every kind, the infinities,
signed zeros, subnormals, the extremes), then words sampled from a seeded generator, kept when they reach a rounding case
(`_rounding`) the vectors don't have twice yet, then uniform words."""
from __future__ import annotations

import random
import re
from fractions import Fraction
from pathlib import Path

from verity.ml.circuits.ops import fp32 as F

LEAN = Path(__file__).resolve().parents[5] / "Security"
VECTORS = LEAN / "Proofs" / "Core" / "Fp32" / "Vectors.lean"
THEOREM = re.compile(r"theorem \w+ : (\w+)((?: 0x[0-9a-f]+)+) = (0x[0-9a-f]+) :=")

#: each Lean definition: the primitive it states, the exact operation it rounds, and the fewest vectors it is held to
PRIMS = {
    "f32AddV2": (F.F32AddV2, "add", 100),
    "f32MulV2": (F.F32MulV2, "mul", 100),
    "f32DivV2": (F.F32DivV2, "div", 100),
    "f32FmaV2": (F.F32FmaV2, "fma", 100),
    "f32AddFtzV2": (F.F32AddFtzV2, "add", 60),
    "f32SubFtzV2": (F.F32SubFtzV2, "sub", 60),
    "f32MulFtzV2": (F.F32MulFtzV2, "mul", 60),
    "f32FmaFtzV2": (F.F32FmaFtzV2, "fma", 60),
    "f32FmaSubFtzV2": (F.F32FmaSubFtzV2, "fmasub", 60),
    "f32Fma": (F.F32Fma, "fma", 30),
    "f32FmaFtz": (F.F32FmaFtz, "fma", 30),
    "f32FmaSubFtz": (F.F32FmaSubFtz, "fmasub", 30),
    "f32Max": (F.F32Max, None, 50),
    "guardNegInfZero": (F.GuardNegInfZero, None, 8),
    "bf16Add": (F.Bf16Add, "add", 40),
    "bf16AddF2fp": (F.Bf16AddF2fp, "add", 40),
    "bf16MulF32": (F.Bf16MulF32, "mul", 40),
    "bf16MulBf16": (F.Bf16MulBf16, "mul", 40),
}
FTZ = {"f32AddFtzV2", "f32SubFtzV2", "f32MulFtzV2", "f32FmaFtzV2", "f32FmaSubFtzV2", "f32FmaFtz", "f32FmaSubFtz",
       "f32Max"}
#: the operands and results that are BF16 words (`bf16MulF32`'s scale is an FP32 word)
BF16_ARGS = {"bf16Add": (0, 1), "bf16AddF2fp": (0, 1), "bf16MulF32": (0,), "bf16MulBf16": (0, 1)}

_TIES = ("exact", "below half", "half, even", "half, odd", "above half")
_ROUNDING = {f"{tie}, normal" for tie in _TIES} | {"carry", "overflow"}
_SPECIAL = {"NaN operand", "signalling NaN operand", "negative NaN operand", "infinite operand", "-0 operand",
            "subnormal operand", "NaN result", "+inf result", "-inf result", "+0 result", "-0 result", "normal result"}
_SUBNORMAL = {f"{tie}, subnormal" for tie in _TIES} | {"underflow", "carry to normal"}
_FLUSH = {"flushed operand", "flushed -operand", "flushed result", "carry to normal"}
_BF16 = {"bf16 exact", "bf16 below half", "bf16 half, even", "bf16 half, odd", "bf16 above half", "bf16 overflow"}
#: what each definition's vectors reach. A finite sum is a multiple of the least subnormal, so a subnormal sum is exact;
#: a quotient of 24-bit significands is never a tie in the normal range, nor within half a unit below a power of two.
_ADD = _SPECIAL | _ROUNDING | {"exact, subnormal", "subnormal result", "exact zero"}
_FMA = _SPECIAL | _ROUNDING | _SUBNORMAL | {"subnormal result", "exact zero"}
_FMA_V1 = _SPECIAL | {"half, even, normal", "half, odd, normal", "exact zero", "overflow"}
_BF16_OPS = _BF16 | _SPECIAL | {"overflow", "exact zero"}
REQUIRED = {
    "f32AddV2": _ADD,
    "f32MulV2": _FMA,
    "f32DivV2": _FMA - {"half, even, normal", "half, odd, normal", "carry"},
    "f32FmaV2": _FMA,
    "f32AddFtzV2": _ADD - {"subnormal result"} | _FLUSH - {"carry to normal"},
    "f32SubFtzV2": _ADD - {"subnormal result"} | _FLUSH - {"carry to normal"},
    "f32MulFtzV2": _FMA - {"subnormal result"} | _FLUSH,
    "f32FmaFtzV2": _FMA - {"subnormal result"} | _FLUSH,
    "f32FmaSubFtzV2": _FMA - {"subnormal result"} | _FLUSH,
    "f32Fma": _FMA_V1 | {"half, odd, subnormal", "subnormal result"},
    "f32FmaFtz": _FMA_V1 | _FLUSH,
    "f32FmaSubFtz": _FMA_V1 | _FLUSH,
    "f32Max": {"NaN operand", "signalling NaN operand", "negative NaN operand", "infinite operand", "-0 operand",
               "subnormal operand", "flushed operand", "flushed -operand", "NaN result", "+inf result", "-inf result",
               "+0 result", "-0 result", "normal result"},
    "guardNegInfZero": {"NaN operand", "infinite operand", "-0 operand", "subnormal operand", "NaN result",
                        "+inf result", "+0 result", "-0 result", "subnormal result", "normal result"},
    "bf16Add": _BF16_OPS,
    "bf16AddF2fp": _BF16_OPS,
    "bf16MulF32": _BF16_OPS,
    "bf16MulBf16": _BF16_OPS,
}


def _f32(w: int, bf16: bool) -> int:
    return w << 16 if bf16 else w


def _class(w: int) -> str:
    e, m = (w >> 23) & 0xFF, w & 0x7FFFFF
    if e == 0xFF:
        return "NaN" if m else "+inf" if w >> 31 == 0 else "-inf"
    if e == 0:
        return ("+0" if w >> 31 == 0 else "-0") if m == 0 else "subnormal"
    return "normal"


def _rounding(x: Fraction) -> tuple[str, str]:
    """Which branches `fp32._round_f32` takes on the nonzero `x`: how the significand rounds, and where the word
    lands."""
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, -126)
    q = a / Fraction(2) ** (e - 23)
    n = q.numerator // q.denominator
    r = q - n
    half = Fraction(1, 2)
    tie = ("exact" if r == 0 else "below half" if r < half else "above half" if r > half
           else f"half, {'odd' if n & 1 else 'even'}")
    up = n + (r > half or (r == half and n & 1))
    w = F.f32_round(x) & 0x7FFFFFFF
    zone = ("underflow" if w == 0 else "overflow" if w == 0x7F800000 else "carry" if up == 1 << 24
            else "carry to normal" if up == 1 << 23 and n < up else "subnormal" if w < 0x800000 else "normal")
    return tie, zone


def _exact(op: str, words: list[int]) -> Fraction | None:
    decoded = [F.f32_decode(w) for w in words]
    if any(c != 0 for c, _ in decoded):
        return None
    a, b, *c = (v for _, v in decoded)
    if op == "div":
        return a / b if b else None
    return {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "fma": lambda: a * b + c[0],
            "fmasub": lambda: a * b - c[0]}[op]()


def kinds(name: str, args: tuple[int, ...]) -> set[str]:
    """The cases of `verity.ml.circuits.ops.fp32` that a vector reaches: its operands' and result's classes, and how its exact
    result rounds."""
    prim, op, _ = PRIMS[name]
    bf16 = BF16_ARGS.get(name, ())
    words = [_f32(w, i in bf16) for i, w in enumerate(args)]
    out = _f32(prim.evaluate(*args), bool(bf16))
    found = {f"{_class(out)} result"}
    for i, w in enumerate(words):
        c = _class(w)
        found.add({"+inf": "infinite", "-inf": "infinite", "+0": "zero"}.get(c, c) + " operand")
        if c == "NaN":
            found |= {"signalling NaN operand"} if not w & 0x400000 else set()
            found |= {"negative NaN operand"} if w >> 31 else set()
        if c == "subnormal" and name in FTZ:
            found.add("flushed operand" if w >> 31 == 0 else "flushed -operand")
    if op is None:
        return found
    rounded = [F.ftz_bits(w) for w in words] if name in FTZ else words
    x = _exact(op, rounded)
    if x == 0:
        found.add("exact zero")
    elif x is not None:
        tie, zone = _rounding(x)
        found |= {f"{tie}, {zone}" if zone in ("normal", "subnormal") else zone}
        if name in FTZ and zone == "subnormal":
            found.add("flushed result")
    if bf16 and x is not None:
        u = F.f32_round(x)
        low = u & 0xFFFF
        found.add("bf16 exact" if low == 0 else "bf16 below half" if low < 0x8000 else "bf16 above half" if low > 0x8000
                  else f"bf16 half, {'odd' if u >> 16 & 1 else 'even'}")
        if out & 0x7FFFFFFF == 0x7F800000 and u & 0x7FFFFFFF != 0x7F800000:
            found.add("bf16 overflow")
    return found


def _vectors() -> list[tuple[str, tuple[int, ...], int]]:
    return [(name, tuple(int(w, 16) for w in args.split()), int(out, 16))
            for name, args, out in THEOREM.findall(VECTORS.read_text())]


def test_lean_fp32_vectors_are_the_python() -> None:
    vectors = _vectors()
    for name, args, out in vectors:
        assert PRIMS[name][0].evaluate(*args) == out, (name, [hex(w) for w in args])
    for name, (_, _, least) in PRIMS.items():
        assert sum(v[0] == name for v in vectors) >= least, name


def test_lean_fp32_vectors_reach_every_case() -> None:
    reached: dict[str, set[str]] = {name: set() for name in PRIMS}
    for name, args, _ in _vectors():
        reached[name] |= kinds(name, args)
    for name, required in REQUIRED.items():
        assert required <= reached[name], (name, sorted(required - reached[name]))


# ---------------------------------------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------------------------------------

ZERO, NZERO, ONE, NONE, INF, NINF = 0, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000
MAX, MIN_SUB, NMAX_SUB = 0x7F7FFFFF, 0x00000001, 0x807FFFFF
#: quiet and signalling NaNs of both signs, with and without a payload
NANS = (0x7FC00000, 0xFFC00000, 0x7FC12345, 0x7F800001, 0xFFBFFFFF, 0x7FFFFFFF)
GRID = (ZERO, NZERO, INF, NINF, ONE, NONE, NMAX_SUB, MAX)
FTZ_GRID = (NZERO, INF, NINF, ONE, MIN_SUB, NMAX_SUB)
MAX_GRID = (ZERO, NZERO, NINF, ONE, MIN_SUB, NMAX_SUB)
#: the operand pairs of a product: finite, signed zeros, an infinity, an infinity times zero
PRODUCTS = ((ZERO, ONE), (NZERO, ONE), (ZERO, NONE), (INF, ONE), (NINF, ONE), (INF, ZERO), (ONE, NINF), (ONE, ONE),
            (MAX, MAX), (MIN_SUB, MIN_SUB), (NONE, NMAX_SUB))
ADDENDS = (ZERO, NZERO, INF, NINF, NONE, MIN_SUB)
BF16_WORDS = (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFF81, 0x7F81, 0x3F80, 0xBF80, 0x7F7F, 0x0001, 0x8040)
#: rare roundings: significands 10610063 * 13264529 = 2^47 - 1 round up to 2^24, and (2^24 - 1) * 2^-150 rounds up
#: from the subnormals to the least normal
DIRECTED = {"mul": [(0x3FA1E58F, 0x3FCA6691), (0xBFA1E58F, 0x3FCA6691)], "div": [(0x00FFFFFF, 0x40000000),
                                                                             (0x80FFFFFF, 0x40000000)]}


def _word(rng: random.Random, field: int | None = None) -> int:
    """A word whose significand often ends in zeros, so that sums and products of it land on halves."""
    if field is None:
        field = rng.choice((rng.randrange(1, 255), rng.randrange(0, 8), rng.randrange(246, 255)))
    mantissa = rng.getrandbits(23)
    if rng.random() < 0.7:
        cut = rng.randrange(24)
        mantissa = (mantissa >> cut) << cut
    if rng.random() < 0.1:
        mantissa = 0x7FFFFF
    return rng.getrandbits(1) << 31 | min(max(field, 0), 254) << 23 | mantissa


def _field(w: int) -> int:
    return max((w >> 23) & 0xFF, 1)


def _candidate(op: str, rng: random.Random) -> tuple[int, ...]:
    a = _word(rng)
    # the exponent field the result should land near, from subnormal underflow to overflow
    target = rng.choice((rng.randrange(1, 255), rng.randrange(-26, 2), rng.randrange(250, 258)))
    if op in ("add", "sub"):
        b = _word(rng, _field(a) - rng.randrange(0, 27))
        low = rng.randrange(0, 26)
        if low:
            b = (b >> low << low) | rng.choice((1 << (low - 1), (1 << (low - 1)) - 1, (1 << (low - 1)) + 1, 0))
        return a, b
    if op == "mul":
        return a, _word(rng, target - _field(a) + 127)
    if op == "div":
        return a, _word(rng, _field(a) - target + 127)
    b = _word(rng, target - _field(a) + 127)
    product = F.F32Mul.evaluate(a, b)
    c = rng.choice((_word(rng, _field(product) - rng.randrange(-2, 27)), product ^ 0x80000000,
                    (product ^ 0x80000000) + rng.randrange(-2, 3) & 0xFFFFFFFF))
    return (a, b, c) if not F.f32_is_nan(c) else (a, b, ZERO)


CORE = ("f32AddV2", "f32MulV2", "f32DivV2", "f32FmaV2")
_SEARCHED = _ROUNDING | _SUBNORMAL | _FLUSH | {f"{tie}, {zone}" for tie in _TIES for zone in ("normal", "subnormal")}


def _cases(name: str, rng: random.Random) -> list[tuple[int, ...]]:
    _, op, _ = PRIMS[name]
    if name == "guardNegInfZero":
        return [(w,) for w in (NINF, INF, ZERO, NZERO, 0x7FC00000, 0xFF800001, ONE, NONE, MIN_SUB)]
    if name.startswith("bf16"):
        second = BF16_WORDS if name != "bf16MulF32" else (ZERO, NZERO, INF, NINF, 0x7FC00000, ONE, MAX, NONE, MIN_SUB)
        cases = [(a, b) for a in BF16_WORDS[:6] for b in second[:4]] + [(BF16_WORDS[9], second[-3])]
        cases += _search(name, rng, lambda: _bf16_candidate(name, op, rng), 2, lambda k: k & _BF16)
        return cases + [(rng.getrandbits(16), rng.getrandbits(16 if name != "bf16MulF32" else 32)) for _ in range(4)]
    arity = 3 if op in ("fma", "fmasub") else 2
    if arity == 2:
        partner = (ONE, INF, ZERO)
        cases = [(n, partner[i % 3]) for i, n in enumerate(NANS)] + [(partner[i % 3], n) for i, n in enumerate(NANS)]
        cases += [(0x7FC12345, 0xFFBFFFFF)]
        grid = GRID if name in CORE else MAX_GRID if name == "f32Max" else FTZ_GRID
        cases += [(a, b) for a in grid for b in grid]
    else:
        cases = [(n, ONE, ONE) for n in NANS[:3]] + [(ONE, n, ZERO) for n in NANS[3:]] + [(ONE, ONE, n) for n in NANS]
        products = PRODUCTS if name.endswith("V2") else PRODUCTS[::2]
        cases += [(a, b, c) for a, b in products for c in (ADDENDS if name.endswith("V2") else ADDENDS[1::2])]
    if name == "f32Max":
        return cases + [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(8)]
    if name in FTZ:
        cases += _flush_cases(op)
    if name.endswith("V2"):
        cases += DIRECTED.get(op, [])
        each = 2 if name in CORE else 1
        cases += _search(name, rng, lambda: _candidate(op, rng), each, lambda k: k & _SEARCHED)
    else:
        cases += _search(name, rng, lambda: _candidate(op, rng), 1,
                         lambda k: k & {"half, even, normal", "half, odd, normal", "half, odd, subnormal"})
    uniform = 16 if name in CORE else 4
    return cases + [tuple(rng.getrandbits(32) for _ in range(arity)) for _ in range(uniform)]


def _flush_cases(op: str) -> list[tuple[int, ...]]:
    """Subnormal operands of both signs, and results the flush takes to zero or that round up to the least normal."""
    if op in ("add", "sub"):
        return [(0x00400000, 0x00400000), (0x80000003, ONE), (ONE, 0x807FFFFF), (0x00800001, 0x80800000),
                (0x00800003, 0x80800001), (0x80800000, 0x00800001)]
    if op == "mul":
        return [(0x00400000, ONE), (ONE, 0x80000001), (0x00800000, 0x3F000000), (0x80800000, 0x3F000000),
                (0x00FFFFFF, 0x3F000000), (0x00FFFFFE, 0x3F7FFFFF)]
    return [(0x00400000, ONE, ONE), (ONE, 0x80000001, ZERO), (ONE, ONE, 0x807FFFFF), (0x00800000, 0x3F000000, ZERO),
            (0x00800000, 0x3F000000, 0x80000000), (0x00FFFFFF, 0x3F000000, ZERO), (0x00800001, ONE, 0x80800000)]


def _bf16_candidate(name: str, op: str, rng: random.Random) -> tuple[int, int]:
    a = _word(rng) >> 16
    if name == "bf16MulF32":
        return a, _word(rng, rng.choice((rng.randrange(100, 160), 254 - _field(a << 16) + 127)))
    if op == "mul":
        return a, _word(rng, rng.choice((rng.randrange(100, 160), 254 - _field(a << 16) + 127))) >> 16
    return a, _word(rng, _field(a << 16) - rng.randrange(0, 12)) >> 16


def _search(name: str, rng: random.Random, candidate, each: int, select, tries: int = 40000) -> list[tuple[int, ...]]:
    """Candidates kept while they reach a selected case that fewer than `each` kept candidates reach."""
    kept, count = [], {}
    for _ in range(tries):
        args = candidate()
        new = [k for k in select(kinds(name, args)) if count.get(k, 0) < each]
        if new:
            kept.append(args)
            for k in new:
                count[k] = count.get(k, 0) + 1
    return kept


_OPERANDS = ("NaN operand", "signalling NaN operand", "negative NaN operand", "infinite operand", "zero operand",
             "-0 operand", "subnormal operand", "flushed operand", "flushed -operand")


def _describe(name: str, args: tuple[int, ...]) -> str:
    found = kinds(name, args)
    operands = [k for k in _OPERANDS if k in found]
    rounding = sorted(k for k in found - set(_OPERANDS) if not k.endswith(("result", "operand")))
    result = [k for k in sorted(found) if k.endswith("result")]
    return "; ".join(", ".join(part) for part in (operands, rounding, result) if part)


def generate() -> list[tuple[str, tuple[int, ...]]]:
    rng = random.Random(20261003)
    vectors = []
    for name in PRIMS:
        seen = set()
        for args in _cases(name, rng):
            if args not in seen:
                seen.add(args)
                vectors.append((name, args))
    return vectors


def render(vectors: list[tuple[str, tuple[int, ...]]]) -> str:
    lines = [
        "import Definitions.Core.Fp32",
        "",
        "/-!",
        "# Vectors: `verity.ml.circuits.ops.fp32`'s primitives",
        "",
        "Each theorem is one input of a primitive of `verity.ml.circuits.ops.fp32` and the word its Python evaluator returns,",
        "checked by the kernel against the definition of the same name in `Verity.Fp32`.",
        "`verity/ml/circuits/ops/tests/test_lean_fp32_vectors.py` checks the same vectors against the Python, and that",
        "they reach every case the Python distinguishes, so a change on either side fails a build: the definitions",
        "cannot drift silently. That test, run as a script, writes this file.",
        "-/",
        "",
        "namespace Verity.Fp32.Vectors",
        "",
        "open Verity.Fp32",
    ]
    index: dict[str, int] = {}
    for name, args in vectors:
        i = index[name] = index.get(name, -1) + 1
        bf16 = BF16_ARGS.get(name, ())
        words = " ".join(f"0x{w:04x}" if j in bf16 else f"0x{w:08x}" for j, w in enumerate(args))
        out = PRIMS[name][0].evaluate(*args)
        out = f"0x{out:04x}" if bf16 else f"0x{out:08x}"
        theorem = f"theorem {name}_{i:03d} : {name} {words} = {out} := by decide +kernel"
        lines += ["", f"/-- {_describe(name, args)} -/", theorem]
    return "\n".join(lines + ["", "end Verity.Fp32.Vectors", ""])


if __name__ == "__main__":
    VECTORS.parent.mkdir(parents=True, exist_ok=True)
    VECTORS.write_text(render(generate()))
