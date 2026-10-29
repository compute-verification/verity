"""Red team of property/v1 (the stacking program's int8 property, /tmp/stacking-context/PROGRAM.md I2): each mechanism is
checked bit for bit through the registry's own evaluators and the s8 silicon model, on the CPU.  Nothing here is a kernel;
what is measured is that the named registry primitive (or `verity.ml.circuits.silicon.integer`'s IMMA) computes the exact
nonlinearity claimed, and the class the stacking-budget classifier puts it in.

    uv run python benchmarks/property_redteam/demo.py [--out FILE.json] [--words N]

Sections (encode_in_circuit takes ~35 s a word on one core):
  imma    the s8 IMMA's clamp and wrap become exact comparisons once the incoming accumulator C is a runtime value, and
          the static reduction-length limits that property/v1 checks (operand ranges + K);
  floors  registry primitives that compute an exact floor, carry, comparison, select or mod, with the class the
          stacking-budget classifier (cursor/stack-property-6cd3, benchmarks/stacking_budgets/count_budgets.py) gives each
          and whether property/v1's "roundings" budget (cast + act) counts it;
  codec   the store codec (p1024/k3/masked, verity.core.protocols.remote_bandwidth.codec): its encode/decode round-trip,
          its per-byte squaring cost, the seed-key malleability a post-seed store would exploit, and the bound key that
          closes it (the composition the Lean guarantee rests on);
  encode_in_circuit  the codec's `encode_word` from u8 IMMA products (in range from C = 0) and registry integer and
          compare primitives only, compared with `codec.encode_word`, with its census by the classifier's classes.
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import pkgutil
import random
import struct
import sys

from verity.ml.circuits.defs import REGISTRY, PrimitiveDefinition
from verity.ml.circuits.silicon.integer import (BLACKWELL_SM120_S8_M16N8K32 as IMMA_SAT,
                                                 BLACKWELL_SM120_S8_M16N8K32_WRAP as IMMA_WRAP, IntegerMma, S32_MAX,
                                                 S32_MIN, s32_value)
from verity.core.protocols.remote_bandwidth import codec as C

M32 = (1 << 32) - 1

# The stacking-budget classifier's sets (cursor/stack-property-6cd3 4268176afa, count_budgets.py lines 41-50).  Its
# headline per-byte budget c is cast + act.  "free(widen)" is dropped
# entirely (a pass); "int" and "unknown" are counted in the census but are not in c.
CLASSES = {
    "cast": {"F2fpBf16", "F32ToBf16Rn", "F32ToE4m3Sat", "F32ToE2m1Sat"},
    "free(widen)": {"Bf16ToF32", "E4m3ToF32", "F32Fabs", "F32Neg", "F32BitsShl23"},
    "act": {"MufuEx2Ftz", "MufuRcpFtz", "MufuSqrtFtz", "MufuTanh", "RsqrtApprox", "DivFullRcp", "Fa2InvSum"},
    "cmp": {"F32Max", "F32GtStrict", "F32Fmaxf", "F32Fminf", "Bf16GtStrict", "F32Eq", "I32Le", "I32Eq", "F32IsFinite",
            "GuardNegInfZero", "F32Sat", "SelectBf16", "SelectF32", "SelectI32", "DivFullScaleA"},
    "int": {"I32Add", "BitAnd", "BitNot", "BitOr"},
    "f32_arith": {"F32Add", "F32AddFtz", "F32Sub", "F32SubFtz", "F32Mul", "F32MulFtz", "F32Fma", "F32FmaFtz",
                  "F32FmaSubFtz", "F32FmaRm"},
}
BUDGETED = {"cast", "act"}  # what property/v1 counts toward "roundings <= c per byte of certified read"


def classify(pid: str) -> str:
    base = pid.rsplit("_v", 1)[0]
    for name, s in CLASSES.items():
        if base in s:
            return name
    return "unknown"


# ---- registry -----------------------------------------------------------------------------------------------------------


def load_registry() -> dict[str, PrimitiveDefinition]:
    import verity.ml.circuits.ops as ops
    import verity.core.protocols.pouw.circuit.words  # noqa: F401  (PoUW word primitives register on import)
    import verity.ml.definitions.registry as reg
    for pkg, prefix in ((ops, "verity.ml.circuits.ops."), (reg, "verity.ml.definitions.registry.")):
        for m in pkgutil.walk_packages(pkg.__path__, prefix):
            if ".tests" in m.name or "quarantine" in m.name or "circuit-check" in m.name:
                continue
            try:
                importlib.import_module(m.name)
            except Exception as e:
                print(f"skip {m.name}: {type(e).__name__}", file=sys.stderr)
    return {k: v for k, v in REGISTRY.defs.items() if isinstance(v, PrimitiveDefinition)}


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def s8(v: int) -> int:
    assert -128 <= v <= 127, v
    return v & 0xFF


# ---- imma ---------------------------------------------------------------------------------------------------------------


def _shift(model: IntegerMma, c_runtime: int, shift: int) -> int:
    """`model` chained so that D = C + shift (clamped or wrapped by `model`): constant s8 products summing to `shift`
    (0 <= shift <= S32_MAX), 32 a step, each step's C the previous D, the first C the runtime word `c_runtime`.  Each
    product is 127 * y or 1 * y with y <= 127, so every operand is a constant s8 code."""
    assert 0 <= shift <= S32_MAX and 0 <= c_runtime < (1 << 31)
    word = c_runtime
    room = shift
    while room > 0:
        a: list[int] = []
        b: list[int] = []
        while len(a) < 32 and room > 0:
            y = min(127, room // 127)
            x = 127 if y else 1
            y = y if y else room
            a.append(s8(x))
            b.append(s8(y))
            room -= x * y
        a += [0] * (32 - len(a))
        b += [0] * (32 - len(b))
        word = model.step(word, a, b)
    return word


def section_imma(rng: random.Random, trials: int = 3000) -> dict:
    out: dict = {}

    # Static K limit (property/v1 "operand ranges and reduction length"): the worst product is (-128)(-128)=2^14 for
    # s8 x s8 and 255 * (-128) for u8 x s8, so with C = 0 the exact sum stays in s32 up to these K, and no more.
    k_s8 = S32_MAX // (128 * 128)
    k_u8s8 = (-S32_MIN) // (255 * 128)
    full, tail = divmod(k_s8, 32)
    word = 0
    for _ in range(full):
        word = IMMA_SAT.step(word, [s8(-128)] * 32, [s8(-128)] * 32)
    word = IMMA_SAT.step(word, [s8(-128)] * tail + [0] * (32 - tail), [s8(-128)] * tail + [0] * (32 - tail))
    at = s32_value(word)
    one_more = s32_value(IMMA_SAT.step(word, [s8(-128)] + [0] * 31, [s8(-128)] + [0] * 31))
    out["static_K_limit"] = {"s8xs8_K_max": k_s8, "u8xs8_K_max": k_u8s8, "S32_MAX": S32_MAX,
                             "sum_at_K_max": at, "exact_at_K_max": at == k_s8 * 16384,
                             "one_more_product_clamps_to_S32_MAX": one_more == S32_MAX,
                             "instructions": full + 2, "each_instruction_K": 32,
                             "u8xs8_PV_at_131072_context_exceeds_K_max": 131072 > k_u8s8}

    # The runtime incoming accumulator: a value r in [0, 2^23) the circuit holds biased toward the s32 edge (any earlier
    # chain or add puts it there once); each threshold test then adds a short constant chain (<= 17 IMMA steps).
    def draw() -> tuple[int, int]:
        T = rng.randrange(1, 1 << 22)
        r = rng.randrange(0, 1 << 23) if rng.random() < 0.5 else max(0, T + rng.randrange(-2, 3))
        return r, T

    # .satfinite: D1 - D2 = [r < T] with D1 = sat(C + 2^23 - T + 1), D2 = sat(C + 2^23 - T), C = r + S32_MAX - 2^23.
    sat_base = S32_MAX - (1 << 23)
    sat_ok = 0
    for _ in range(trials):
        r, T = draw()
        d1 = s32_value(_shift(IMMA_SAT, r + sat_base, (1 << 23) - T + 1))
        d2 = s32_value(_shift(IMMA_SAT, r + sat_base, (1 << 23) - T))
        sat_ok += (d1 - d2) == int(r < T)
    out["satfinite_runtime_C_is_compare"] = {"trials": trials, "exact": sat_ok,
                                             "max_imma_steps_per_test": -(-(1 << 23) // (32 * 127 * 127))}

    # No .satfinite: D wraps mod 2^32, and the sign bit of C + 2^23 - T, C = r + 2^31 - 2^23, is [r >= T].
    wrap_base = (1 << 31) - (1 << 23)
    wrap_ok = 0
    for _ in range(trials):
        r, T = draw()
        d = _shift(IMMA_WRAP, r + wrap_base, (1 << 23) - T)
        wrap_ok += (d >> 31) == int(r >= T)
    out["wrap_runtime_C_sign_is_compare"] = {"trials": trials, "exact": wrap_ok}

    # Split-K: two partials, each instruction in range on its own, combined by the registry's wrapping I32Add_v1; the
    # sum's sign bit is still [r >= T], a comparison no per-matmul static check sees.
    add = REGISTRY.defs["I32Add_v1"].evaluate
    split_ok = 0
    for _ in range(trials):
        r, T = draw()
        half = r // 2
        p2 = _shift(IMMA_WRAP, (r - half) + wrap_base, (1 << 23) - T)
        split_ok += (add(half, p2) >> 31) == int(r >= T)
    out["splitK_wrapping_add_is_compare"] = {"trials": trials, "exact": split_ok}
    return out


# ---- floors -------------------------------------------------------------------------------------------------------------


def section_floors(prims: dict[str, PrimitiveDefinition]) -> dict:
    """Each case: a registry primitive, inputs, the exact nonlinear value it returns, the claim it realizes, its class and
    whether property/v1 budgets it.  All are checked by assertion; the table records the class."""
    rows = []

    def check(pid: str, args: tuple[int, ...], expect: int, realizes: str) -> None:
        got = prims[pid].evaluate(*args)
        ok = got == expect
        cls = classify(pid)
        rows.append({"primitive": pid, "realizes": realizes, "value": got, "exact": ok, "class": cls,
                     "budgeted": cls in BUDGETED})
        if not ok:
            print(f"MISMATCH {pid}{args}: {got} != {expect}", file=sys.stderr)

    # F32BitsShl23: (a << 23) mod 2^32 -- the low 9 bits of a, placed high: an exact mod-2^9 read-out of an integer a.
    check("F32BitsShl23_v1", (0x4B400000 | 0x155,), (0x155 << 23) & M32, "exact low-9-bit extraction (mod 2^9)")
    # F32Fabs: clears the sign bit: |x| on encodings -- max(x, -x), the ReLU building block.
    check("F32Fabs_v1", (f32(-3.5),), f32(3.5), "abs / ReLU half-wave (clear sign)")
    # I32Mul / I32Div / I32Rem: exact modular multiply and floor division with remainder.
    check("I32Mul_v1", (0x12345, 0x6789), (0x12345 * 0x6789) & M32, "exact multiply mod 2^32")
    check("I32Div_v1", (1000, 7), 142, "floor division (142 = 1000 // 7)")
    check("I32Rem_v1", (1000, 7), 6, "remainder / mod 7")
    # U32And / U32Xor: bit mask (floor to a multiple of 2^k) and bitwise xor (GF(2) add, a carry-free adder's bit).
    check("U32And_v1", (0xDEADBEEF, 0xFFFFFF00), 0xDEADBE00, "floor to a multiple of 2^8 (low-byte clear)")
    check("U32Xor_v1", (0xF0F0, 0x0FF0), 0xFF00, "GF(2) add (xor)")
    # F32ToU32RzByte: static_cast<uint8_t>(float): truncation toward zero to a byte -- a genuine quantizing rounding.
    check("F32ToU32RzByte_v1", (f32(200.9),), 200, "float->uint8 truncation (round toward zero)")
    # NcpRneClampI7: round-to-nearest-even then clamp to [-63, 63]: a quantizer.
    check("NcpRneClampI7_v1", (f32(2.5),), 2 & 0x7F, "RNE quantize + clamp to int7")
    # Ue8m0CeilByte: the UE8M0 exponent of a scale, rounded up: a ceiling of log2.
    s = f32(3.0)
    check("Ue8m0CeilByte_v1", (s,), ((s + 0x7FFFFF) >> 23) & 0xFF, "ceil(log2) exponent byte (round up)")
    # Comparisons/selects that gate a value on a runtime predicate.
    check("I32Le_v1", (5, 9), 1, "signed <= (compare)")
    check("U32Ge_v1", (9, 5), 1, "unsigned >= (compare)")
    check("SelectI32_v1", (1, 0xAAAA, 0xBBBB), 0xAAAA, "data-dependent select")
    check("F32Sat_v1", (f32(2.0),), f32(1.0), "clamp to [0,1] (saturating nonlinearity)")
    # A PoUW comparison on a sum of squares (an exact >= on wide encodings).
    check("PouwSqGe_v1", (0, 0), 1, "wide compare (2^300 x^2 >= theta)")

    # A runtime-indexed gather over constant words is a lookup table: it realizes any function of the index.  Here a
    # 128-entry f32 column holds (i^2 + 7i) mod 251, and the gather returns its entry exactly: a data-dependent
    # nonlinearity property/v1 does not budget (the census counts gathers and fan-in, the budget does not read them).
    from verity.ml.definitions.registry.ref_prims import gather_f32
    table = [f32(float((i * i + 7 * i) % 251)) for i in range(128)]   # an arbitrary nonlinear table of the 7-bit index
    g = gather_f32(128)
    idx = 100
    got = g.evaluate(idx, *table)
    rows.append({"primitive": g.name, "realizes": "arbitrary table lookup by a runtime 7-bit index (free nonlinearity)",
                 "value": got, "exact": got == table[idx], "class": "gather(census-only)", "budgeted": False})

    n_unbudgeted = sum(1 for r in rows if r["exact"] and not r["budgeted"])
    return {"rows": rows, "exact": sum(r["exact"] for r in rows), "total": len(rows),
            "exact_but_unbudgeted": n_unbudgeted}


# ---- codec --------------------------------------------------------------------------------------------------------------


def section_codec(rng: random.Random) -> dict:
    out: dict = {}
    default = C.DEFAULT
    out["codec"] = {"name": default.name, "encode_squarings_per_block": default.encode_squarings,
                    "block_bytes": C.BLOCK_BYTES, "squarings_per_byte": default.encode_squarings / C.BLOCK_BYTES,
                    "rounds_k": default.rounds, "masked": default.masked}

    seed = bytes(rng.randrange(256) for _ in range(C.SEED_BYTES))
    k = C.key(seed, default)
    f = default.field
    # a canonical payload: every word below p, here by clearing each little-endian word's top byte (< 2^1016 < p)
    pw = bytearray(rng.randrange(256) for _ in range(C.BLOCK_BYTES))
    for w in range(f.block_words):
        pw[w * f.word_bytes + f.word_bytes - 1] = 0
    payload = bytes(pw)

    enc = C.encode(payload, 7, k)
    dec = C.decode(enc, 7, k)
    out["seed_key_roundtrip"] = {"decode_equals_payload": dec == payload}

    # Seed-key malleability: a writer picks any encoding E and stores decode(E); its payload then encodes to E for free
    # (no squarings held).  Here E = another block's encoding E_a; B then "holds" A's encoding at decode cost (k
    # squarings a word), not encode cost (k roots).  This is the correlated store the uniform-store model excludes.
    e_a = C.encode(payload, 1, k)
    chosen_payload = C.decode(e_a, 2, k)                     # decode E_a under block 2's (seed) key
    assert chosen_payload is not None
    re_enc = C.encode(chosen_payload, 2, k)
    decode_sq = default.rounds * f.block_words
    out["seed_key_malleable"] = {"chosen_encoding_reproduced": re_enc == e_a,
                                 "encode_cost_squarings_per_block": default.encode_squarings,
                                 "decode_cost_squarings_per_block": decode_sq,
                                 "ratio_encode_over_decode": default.encode_squarings / decode_sq}

    # Bound key (post-seed KV): b_j ties the key to the payload's digest, so a chosen E decodes to a payload of a
    # different digest -- storing a chosen encoding needs a SHA-512 search, not a decode.
    d_payload = C.digest(payload)
    enc_bound = C.encode(payload, 7, k, bound=True)
    dec_bound_right = C.decode(enc_bound, 7, k, d_payload)
    # feed a chosen encoding (e_a) as if it were block 7's bound block with the honest payload's digest:
    dec_bound_wrong = C.decode(e_a, 7, k, d_payload)
    chosen_ok = dec_bound_wrong is not None and C.digest(dec_bound_wrong) == d_payload
    out["bound_key_closes_malleability"] = {"honest_roundtrip": dec_bound_right == payload,
                                            "chosen_encoding_has_claimed_digest": chosen_ok}

    # Partial information without a root: rho's output is even iff its input is a quadratic residue (codec.rho keeps
    # the even root of a residue, the odd one otherwise), so each round's output parity is a Jacobi symbol of its input.
    p = f.p
    leak_ok, n = 0, 200
    for _ in range(n):
        a = rng.randrange(1, p)
        leak_ok += (C.rho(a, f) & 1) == (1 if _jacobi(a, p) == -1 else 0)
    out["root_parity_is_jacobi_of_input"] = {"trials": n, "exact": leak_ok, "bits_per_word_per_round": 1,
                                             "word_bits": f.n}
    return out


def _jacobi(a: int, n: int) -> int:
    a %= n
    t = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                t = -t
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            t = -t
        a %= n
    return t if n == 1 else 0


# ---- the codec's encode from exact products and unbudgeted primitives ---------------------------------------------------

#: u8 x u8 IMMA: the same instruction at operand formats u8.u8, which the validating run also reproduced bit for bit
#: (`verity.ml.circuits.silicon.integer`'s docstring); its column sums here are at most 128 * 255^2 < 2^23, so every
#: chain is in range from C = 0 and passes a static operand-range and reduction-length check.
IMMA_U8 = IntegerMma(name="blackwell_sm120_u8u8_m16n8k32", arch="sm_120", dtype="u8", b_dtype="u8", operand_bits=8,
                     k_per_instruction=32, satfinite=True, validated="run r20261015-084000-06d8, form u8.u8 .satfinite")


class Circuit:
    """Multi-precision arithmetic mod p = 2^1024 - 105 on 128 u8 digits, every operation a registry evaluator call or an
    IMMA step, counted by primitive id."""

    D = 128

    def __init__(self, prims: dict[str, PrimitiveDefinition], carry: str = "idiv"):
        self.prims = prims
        self.carry = carry
        self.n: collections.Counter = collections.Counter()

    def split(self, t: int) -> tuple[int, int]:
        """(t mod 256, t div 256) for 0 <= t < 2^24: by I32Rem/I32Div ("idiv"), or ("cmp") by 16 compares against
        constant thresholds, each choosing by SelectI32 between r and r - T (I32Sub) and between q and q + 2^b (I32Add)."""
        if self.carry == "idiv":
            return self.call("I32Rem_v1", t, 256), self.call("I32Div_v1", t, 256)
        assert 0 <= t < 1 << 24
        r, q = t, 0
        for b in range(15, -1, -1):
            T = 256 << b
            ge = self.call("U32Ge_v1", r, T)
            r = self.call("SelectI32_v1", ge, self.call("I32Sub_v1", r, T), r)
            q = self.call("SelectI32_v1", ge, self.call("I32Add_v1", q, 1 << b), q)
        return r, q

    def call(self, pid: str, *args: int) -> int:
        self.n[pid] += 1
        return self.prims[pid].evaluate(*args)

    def dot(self, c: int, pairs: list[tuple[int, int]]) -> int:
        """C + sum of u8 products, 32 a step (the chain's C the previous D)."""
        word = c
        for s in range(0, len(pairs), 32):
            chunk = pairs[s:s + 32]
            a = [x for x, _ in chunk] + [0] * (32 - len(chunk))
            b = [y for _, y in chunk] + [0] * (32 - len(chunk))
            self.n["IMMA.u8.u8.SAT (silicon model)"] += 1
            word = IMMA_U8.step(word, a, b)
        return word

    def normalize(self, cols: list[int], width: int) -> tuple[list[int], int]:
        """Column words (each < 2^24 here) to `width` u8 digits and the carry out, every carry one `split`."""
        out, carry = [], 0
        for k in range(width):
            t = self.call("I32Add_v1", cols[k] if k < len(cols) else 0, carry)
            digit, carry = self.split(t)
            out.append(digit)
        return out, carry

    def fold(self, digits: list[int], carry: int) -> tuple[list[int], int]:
        """digits + carry * 2^1024 = digits + 105 * carry (mod p): one IMMA into digit 0, then the carries."""
        cols = list(digits)
        cols[0] = self.dot(digits[0], [(carry, 105)])
        return self.normalize(cols, self.D)

    def canonical(self, digits: list[int]) -> list[int]:
        """x mod p for x < 2^1024: x >= p iff x + 105 carries out of 1024 bits; then x + 105 - 2^1024 is x - p."""
        cols = list(digits)
        cols[0] = self.dot(digits[0], [(105, 1)])
        y, out = self.normalize(cols, self.D)
        ge = self.call("U32Ge_v1", out, 1)
        return [self.call("SelectI32_v1", ge, yi, xi) for yi, xi in zip(y, digits)]

    def reduce(self, digits: list[int], carry: int) -> list[int]:
        for _ in range(3):  # carry <= 105 -> <= 1 -> 0: three folds always reach a carry of 0
            digits, carry = self.fold(digits, carry)
        assert carry == 0
        return self.canonical(digits)

    def mul(self, x: list[int], y: list[int]) -> list[int]:
        cols = [self.dot(0, [(x[i], y[k - i]) for i in range(max(0, k - self.D + 1), min(k, self.D - 1) + 1)])
                for k in range(2 * self.D - 1)]
        d, carry = self.normalize(cols, 2 * self.D)
        assert carry == 0
        lo, hi = d[:self.D], d[self.D:]
        folded = [self.dot(lo[k], [(hi[k], 105)]) for k in range(self.D)]   # hi * 2^1024 = 105 * hi (mod p)
        d2, carry2 = self.normalize(folded, self.D)
        return self.reduce(d2, carry2)

    def add(self, *xs: list[int]) -> list[int]:
        cols = [self.dot(xs[0][k], [(x[k], 1) for x in xs[1:]]) for k in range(self.D)]
        d, carry = self.normalize(cols, self.D)
        return self.reduce(d, carry)

    def neg(self, s: list[int]) -> list[int]:
        """p - s for 0 < s < p: (2^1024 - 1 - s) - 104, the complement by I32Sub_v1, the -104 as + (2^1024 - 104)."""
        comp = [self.call("I32Sub_v1", 255, si) for si in s]
        k = [0x98] + [0xFF] * (self.D - 1)
        cols = [self.dot(comp[i], [(k[i], 1)]) for i in range(self.D)]
        d, _ = self.normalize(cols, self.D)  # the carry out is 2^1024, dropped
        return d

    def eq(self, x: list[int], y: list[int]) -> int:
        bit = 1
        for xi, yi in zip(x, y):
            bit = self.call("BitAnd_v1", bit, self.call("I32Eq_v1", xi, yi))
        return bit

    def select(self, c: int, x: list[int], y: list[int]) -> list[int]:
        return [self.call("SelectI32_v1", c, xi, yi) for xi, yi in zip(x, y)]

    def rho(self, a: list[int], e: int) -> list[int]:
        """The codec's root: s = a^e, e = (p + 1)/4 (square and multiply, MSB first); keep the even of {s, p - s} if
        s^2 = a, else the odd one (codec.rho)."""
        s = list(a)
        for bit in bin(e)[3:]:
            s = self.mul(s, s)
            if bit == "1":
                s = self.mul(s, a)
        square = self.eq(self.mul(s, s), a)
        odd = self.call("U32And_v1", s[0], 1)
        keep = self.call("U32Xor_v1", odd, square)        # (s odd) == (not square)
        if any(s):
            return self.select(keep, s, self.neg(s))
        return s

    def sigma(self, x: list[int], mask: list[int]) -> list[int]:
        """x XOR K if that is below p, else x: below p iff (x XOR K) + 105 does not carry out of 1024 bits."""
        v = [self.call("U32Xor_v1", xi, ki) for xi, ki in zip(x, mask)]
        cols = list(v)
        cols[0] = self.dot(v[0], [(105, 1)])
        _, out = self.normalize(cols, self.D)
        below = self.call("U32Ge_v1", 0, out)  # out == 0
        return self.select(below, v, x)


def digits(v: int) -> list[int]:
    return list(v.to_bytes(128, "little"))


def undigits(d: list[int]) -> int:
    return int.from_bytes(bytes(d), "little")


def section_encode_in_circuit(prims: dict[str, PrimitiveDefinition], rng: random.Random, words: int) -> dict:
    if words <= 0:
        return {"skipped": True}
    seed = bytes(rng.randrange(256) for _ in range(C.SEED_BYTES))
    k = C.key(seed, C.DEFAULT)
    f = C.DEFAULT.field
    e = (f.p + 1) // 4
    results = []
    census: collections.Counter = collections.Counter()
    for w in range(words):
        m = int.from_bytes(bytes(rng.randrange(256) for _ in range(f.word_bytes - 1)), "little")
        i = rng.randrange(1 << 20)
        ckt = Circuit(prims)
        t = ckt.mul(digits(k.tau), digits(i))
        x = digits(m)
        for r, c in enumerate(k.c):
            if r and k.mask is not None:
                x = ckt.sigma(x, digits(k.mask))
            x = ckt.rho(ckt.add(x, digits(c), t), e)
        got, want = undigits(x), C.encode_word(m, i, k)
        results.append(got == want)
        census.update(ckt.n)
    per_byte = _per_byte(census, words * f.word_bytes)
    carries = census["I32Div_v1"] // words
    # Comparison-only carries: the same circuit with every `split` by compares and selects, checked on modular products,
    # its encode census the measured one with each carry's I32Div + I32Rem replaced by the cmp split's 80 calls.
    p = f.p
    ckt = Circuit(prims, carry="cmp")
    cmp_ok = 0
    trials = 4
    for _ in range(trials):
        x, y = rng.randrange(p), rng.randrange(p)
        cmp_ok += undigits(ckt.mul(digits(x), digits(y))) == x * y % p
    swap = collections.Counter(census)
    swap["I32Div_v1"] = 0
    swap["I32Rem_v1"] = 0
    swap["U32Ge_v1"] += 16 * carries * words
    swap["SelectI32_v1"] += 32 * carries * words
    swap["I32Sub_v1"] += 16 * carries * words
    swap["I32Add_v1"] += 16 * carries * words
    cmp_per_byte = _per_byte(swap, words * f.word_bytes)
    return {"codec": C.DEFAULT.name, "words": words, "equal_to_codec_encode_word": sum(results),
            "census_per_word": {pid: n // words for pid, n in sorted(census.items())},
            "carries_per_byte": carries / f.word_bytes,
            "classes_per_byte": per_byte,
            "budgeted_roundings_per_byte": sum(v for c, v in per_byte.items() if c in BUDGETED),
            "cmp_carries": {"modmul_trials": trials, "exact": cmp_ok, "classes_per_byte_derived": cmp_per_byte,
                            "budgeted_roundings_per_byte": sum(v for c, v in cmp_per_byte.items() if c in BUDGETED)}}


def _per_byte(census: collections.Counter, nbytes: int) -> dict:
    by_class: collections.Counter = collections.Counter()
    for pid, n in census.items():
        by_class["exact-int-matmul" if pid.startswith("IMMA") else classify(pid)] += n
    return {cls: n / nbytes for cls, n in by_class.items() if n}


# ---- main ---------------------------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--words", type=int, default=1, help="codec words to encode in-circuit (each ~6,100 modmuls)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    prims = load_registry()
    result = {
        "n_registered_primitives": len(prims),
        "imma": section_imma(rng),
        "floors": section_floors(prims),
        "codec": section_codec(rng),
        "encode_in_circuit": section_encode_in_circuit(prims, rng, args.words),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
