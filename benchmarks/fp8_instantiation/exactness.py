"""When is sm_120's FP8 mma exact?  A closed form of one QMMA.16832 step, checked against the repo's model, and the chains.

`BLACKWELL_SM120_E4M3_M16N8K32` is `GroupSum(groups=(32,), width=26, zero_exponent=-133)`.  Reading `models.group_sum`:
every term t (the FP32 accumulator and the 32 exact products) is a `Term` (sign, exponent e_t = weight of bit 23,
magnitude); M = max(-133, e_t over nonzero terms); each magnitude is scaled by 2^(26-24) and shifted right by M - e_t
(truncation of |t| onto the grid g = 2^(M-25)); the signed sum is exact; it is normalised to 26 bits by a right shift
(truncation), denormalised below 2^-126 by a further right shift, and shifted right by 2 to 24 bits.  The three
truncations compose (floor(floor(x/2^a)/2^b) = floor(x/2^(a+b))), so

    step(C, a, b) = RZ_f32( sum_t sign(t) * g * floor(|t| / g) ),   g = 2^(M - 25),               (closed form)

with RZ_f32 truncation toward zero to the FP32 grid (24 significant bits, 2^-149 below 2^-126), +0 for a zero sum, and
+-inf when the result's exponent passes 127.  `closed_form_step` is that formula on exact integers; `check_closed_form`
compares it with `GroupSum.step` on random and adversarial words.

    python exactness.py model            the closed form against the model, and the integer-grid tests on the model
    python exactness.py ops OUTDIR       operand files for chain_probe (Pearl-C's operands, non-integer grids)
    python exactness.py compare OUTDIR DEVDIR [--json FILE]   the device's words for those files against model and exact
"""
from __future__ import annotations

import json
import math
import random
import struct
import sys
import time
from pathlib import Path

from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32 as QMMA
from verity.ml.circuits.silicon.term import e4m3_product, fp32_term

SCALE = 160                     # every value here is an integer multiple of 2^-160 (FP32 subnormals: 2^-149)
WIDTH, FLOOR = QMMA.width, QMMA.zero_exponent
FINITE_E4M3 = [c for c in range(256) if c & 0x7F != 0x7F]


def term_int(term) -> int:
    """A Term's exact value as an integer multiple of 2^-SCALE."""
    v = term.magnitude << (term.exponent - 23 + SCALE)
    return -v if term.negative else v


def rz_f32_word(n: int) -> int:
    """RZ_f32 of n * 2^-SCALE as FP32 bits; +0 for 0; +-inf past 2^128 (the model's saturation)."""
    if n == 0:
        return 0
    sign, m = (0x80000000 if n < 0 else 0), abs(n)
    e = m.bit_length() - 1 - SCALE
    grid = e - 23 if e >= -126 else -149
    q = m >> (grid + SCALE)
    if e > 127:
        return sign | 0x7F800000
    return sign | ((((e + 127) << 23) | (q & 0x7FFFFF)) if e >= -126 else q)


def step_terms(c_word: int, a: list[int], b: list[int]):
    return [fp32_term(c_word)] + [e4m3_product(x, y) for x, y in zip(a, b)]


def grid_exponent(terms) -> int:
    """log2 g = M - 25: the weight of the adder's last bit (26 bits below and including bit 23 of the leading term)."""
    m = FLOOR
    for t in terms:
        if t.magnitude and t.exponent > m:
            m = t.exponent
    return m - (WIDTH - 1)


def closed_form_step(c_word: int, a: list[int], b: list[int]) -> int:
    terms = step_terms(c_word, a, b)
    lsb = grid_exponent(terms) + SCALE
    total = 0
    for t in terms:
        v = abs(term_int(t)) >> lsb << lsb
        total += -v if t.negative else v
    return rz_f32_word(total)


def exact_sum(c_word: int, a: list[int], b: list[int]) -> int:
    return sum(term_int(t) for t in step_terms(c_word, a, b))


def word_int(word: int) -> int | None:
    """An FP32 word's value as an integer multiple of 2^-SCALE (None for inf/NaN)."""
    if (word >> 23) & 0xFF == 0xFF:
        return None
    return term_int(fp32_term(word))


def explain(c_word: int, a: list[int], b: list[int]) -> tuple[bool, bool]:
    """(every term on the grid g, exact sum representable in FP32): the two conditions of the exactness theorem."""
    terms = step_terms(c_word, a, b)
    lsb = grid_exponent(terms) + SCALE
    aligned = all(abs(term_int(t)) % (1 << lsb) == 0 for t in terms if t.magnitude)
    v = sum(term_int(t) for t in terms)
    return aligned, word_int(rz_f32_word(v)) == v


# ---- 1. the closed form against the model ---------------------------------------------------------------------------

def random_f32(rng: random.Random) -> int:
    kind = rng.random()
    if kind < 0.05:
        return rng.choice([0, 0x80000000])
    if kind < 0.10:
        return rng.randrange(1, 1 << 23) | (rng.getrandbits(1) << 31)          # subnormal
    if kind < 0.60:
        e = rng.randrange(100, 160)                                             # near the products' range
    else:
        e = rng.randrange(1, 255)
    return (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)


def random_codes(rng: random.Random, style: str) -> list[int]:
    if style == "any":
        return [rng.choice(FINITE_E4M3) for _ in range(32)]
    if style == "sparse":
        return [rng.choice(FINITE_E4M3) if rng.random() < 0.2 else 0 for _ in range(32)]
    if style == "spread":                                                      # a few huge, the rest tiny
        return [rng.choice([0x7E, 0xFE, 0x77]) if rng.random() < 0.1 else rng.choice(list(range(1, 16)) + list(range(0x81, 0x90)))
                for _ in range(32)]
    raise ValueError(style)


def check_closed_form(n: int, seed: int = 1) -> dict:
    rng = random.Random(seed)
    counts = {"cases": 0, "mismatches": 0, "inexact": 0, "inexact_aligned": 0, "inexact_unaligned": 0,
              "exact_but_condition_false": 0, "condition_true_but_inexact": 0}
    styles = ["any", "sparse", "spread"]
    t0 = time.time()
    for i in range(n):
        c = random_f32(rng)
        a, b = random_codes(rng, styles[i % 3]), random_codes(rng, styles[(i // 3) % 3])
        model = QMMA.step(c, a, b)
        cf = closed_form_step(c, a, b)
        counts["cases"] += 1
        if model != cf:
            counts["mismatches"] += 1
            if counts["mismatches"] <= 3:
                counts.setdefault("examples", []).append([hex(c), a, b, hex(model), hex(cf)])
            continue
        v = exact_sum(c, a, b)
        is_exact = (model & 0x7FFFFFFF) < 0x7F800000 and word_int(model) == v
        aligned, representable = explain(c, a, b)
        if not is_exact:
            counts["inexact"] += 1
            counts["inexact_aligned" if aligned else "inexact_unaligned"] += 1
        if is_exact and not (aligned and representable):
            counts["exact_but_condition_false"] += 1
        if aligned and representable and not is_exact:
            counts["condition_true_but_inexact"] += 1
    counts["seconds"] = round(time.time() - t0, 1)
    return counts


# ---- 2. the integer grid on the model ----------------------------------------------------------------------------------

def e4m3_of_int(v: int) -> int:
    if v == 0:
        return 0
    m = abs(v)
    e = m.bit_length() - 1
    frac = (m << 3 >> e) - 8
    assert ((frac + 8) << e) >> 3 == m and m <= 16, v
    return (0x80 if v < 0 else 0) | ((e + 7) << 3) | frac


def f32_word_of_int(v: int) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def check_integer_single(seed: int = 2) -> dict:
    """Every grid pair (x, y), |x|, |y| <= 16, as the one nonzero product of a step, against critical accumulators C (exact
    FP32 integers up to 2^26), plus full random atoms: the step is the exact sum whenever that sum is an FP32 value."""
    rng = random.Random(seed)
    cs = {0}
    for e in range(0, 27):
        for d in (-3, -1, 0, 1, 3):
            for s in (1, -1):
                v = s * ((1 << e) + d)
                if abs(v) <= 1 << 26 and word_int(f32_word_of_int(v)) == v << SCALE:
                    cs.add(v)
    cs |= {s * ((1 << 24) - j) for j in (1, 31, 255, 256, 257) for s in (1, -1)}
    grid = range(-16, 17)
    out = {"accumulators": len(cs), "cases": 0, "exact_expected": 0, "exact_mismatch": 0, "inexact_expected": 0,
           "inexact_mismatch_vs_rz": 0}
    zeros = [0] * 31
    for c in sorted(cs):
        cw = f32_word_of_int(c)
        for x in grid:
            for y in grid:
                word = QMMA.step(cw, [e4m3_of_int(x)] + zeros, [e4m3_of_int(y)] + zeros)
                v = c + x * y
                out["cases"] += 1
                if word_int(f32_word_of_int(v)) == v << SCALE:
                    out["exact_expected"] += 1
                    out["exact_mismatch"] += word != (0 if v == 0 else f32_word_of_int(v))
                else:
                    out["inexact_expected"] += 1
                    out["inexact_mismatch_vs_rz"] += word != rz_f32_word(v << SCALE)
    atoms = 0
    atom_mismatch = 0
    for _ in range(20000):
        xs = [rng.randint(-16, 16) for _ in range(32)]
        ys = [rng.randint(-16, 16) for _ in range(32)]
        s = sum(x * y for x, y in zip(xs, ys))
        c = rng.randint(-(1 << 24) + 8192, (1 << 24) - 8192)
        word = QMMA.step(f32_word_of_int(c), [e4m3_of_int(x) for x in xs], [e4m3_of_int(y) for y in ys])
        atoms += 1
        atom_mismatch += word_int(word) != (c + s) << SCALE
    out["random_atoms"], out["random_atom_mismatch"] = atoms, atom_mismatch
    return out


def model_chain(xs: list[int], ys: list[int], g: int | None) -> int:
    """The repo's chain (pearl_c.chain's rule) on grid integers: per window of g atoms a fresh +0 accumulator, one RNE add."""
    from verity.core.protocols.pouw.schemes import pearl_c
    from verity.core.protocols.pouw.schemes.pearl_c_device import SM120, SM120_UNPROMOTED

    dev = SM120 if g else SM120_UNPROMOTED
    return pearl_c.chain([e4m3_of_int(x) for x in xs], [e4m3_of_int(y) for y in ys], dev)


def check_integer_chains() -> list[dict]:
    """The model's own chains (no closed form) at K = 65536 on the extreme integer families."""
    K = 65536
    cases = {
        "all +16 x +16 (2^24)": ([16] * K, [16] * K),
        "all +16 x -16 (-2^24)": ([16] * K, [-16] * K),
        "225 then 256s (odd prefixes up to 2^24 - 31)": ([15] + [16] * (K - 1), [15] + [16] * (K - 1)),
        "climb to 2^23, cancel to 0": ([16] * K, [16] * (K // 2) + [-16] * (K // 2)),
        "2^24 then +3 (K = 65568, control)": ([16] * K + [1] * 32, [16] * K + [1, 1, 1] + [0] * 29),
    }
    rng = random.Random(3)
    cases["uniform random"] = ([rng.randint(-16, 16) for _ in range(K)], [rng.randint(-16, 16) for _ in range(K)])
    out = []
    for name, (xs, ys) in cases.items():
        exact = sum(x * y for x, y in zip(xs, ys))
        row = {"case": name, "K": len(xs), "exact": exact}
        for tag, g in (("unpromoted", None), ("promoted_g4", 4)):
            w = model_chain(xs, ys, g)
            v = struct.unpack("<f", struct.pack("<I", w))[0]
            row[tag] = v if v != int(v) else int(v)
            row[tag + "_exact"] = v == exact
        out.append(row)
    return out


# ---- 3. operand files for the device: Pearl-C's operands and non-integer grids ----------------------------------------

def write_ops(path: Path, a_rows: list[list[int]], b_rows: list[list[int]]) -> None:
    k = len(a_rows[0])
    with open(path, "wb") as f:
        f.write(struct.pack("<3I", len(a_rows), len(b_rows), k))
        for r in a_rows + b_rows:
            assert len(r) == k
            f.write(bytes(r))


def read_ops(path: Path) -> tuple[list[list[int]], list[list[int]]]:
    raw = path.read_bytes()
    m, n, k = struct.unpack("<3I", raw[:12])
    rows = [list(raw[12 + i * k: 12 + (i + 1) * k]) for i in range(m + n)]
    return rows[:m], rows[m:]


def pearlc_rows(rng: random.Random, n_rows: int, k: int, side: int, flavor: str):
    """Pearl-C v1 forming on sm120 (`pearl_c.form_v1`, `pearl_c_device.SM120`) of activation-like (side 0) or weight-like
    (side 1) FP32 rows; returns the E4M3 codes and the per-row (alpha, rho, beta) and admissibility."""
    from verity.core.protocols.pouw.schemes import pearl_c as PC
    from verity.core.protocols.pouw.schemes.pearl_c_device import SM120

    seed_b = PC.seed_b(bytes(rng.getrandbits(8) for _ in range(32)), b"weights-root".ljust(64, b"\0"), "h3")
    seed_e = PC.unit_seed_a(b"salt".ljust(32, b"\0"), b"a-root".ljust(64, b"\0"), b"weights-root".ljust(64, b"\0"), 0, "h3") \
        if side == 0 else seed_b
    f = PC._basis(seed_b, side, k, PC.LINE_NORM_V1)
    rows = []
    for _ in range(n_rows):
        if flavor == "gauss":
            x = [rng.gauss(0, 1) for _ in range(k)]
        elif flavor == "outliers":                    # 1% of channels 30x (within the liveness rule's k/64)
            x = [rng.gauss(0, 1) * (30 if c % 100 == 7 else 1) for c in range(k)]
        elif flavor == "relu":
            x = [max(0.0, rng.gauss(0, 1)) for _ in range(k)]
        elif flavor == "weight":
            x = [rng.gauss(0, 0.02) for _ in range(k)]
        else:
            raise ValueError(flavor)
        rows.append([PC.P.f32_bits(v) for v in x])
    e = [PC.sample_line(seed_e, side, 0, i) for i in range(n_rows)]
    formed = PC.form_v1(rows, e, f, SM120)
    ok = [PC.P.f32_of_bits(r) * PC.P.f32_of_bits(al) >= PC.P.SIGMA_MIN and PC.live_row(row)
          for r, al, row in zip(formed.rho, formed.alpha, formed.x)]
    return formed.codes, ok


def build_ops(outdir: Path, seed: int = 4) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    k = 8192
    meta = {}
    a_codes, a_ok = [], []
    for flavor in ("gauss", "outliers", "relu"):
        c, ok = pearlc_rows(rng, 16, k, 0, flavor)
        a_codes += c
        a_ok += ok
    b_codes, b_ok = pearlc_rows(rng, 32, k, 1, "weight")
    write_ops(outdir / "ops0_pearlc.bin", a_codes, b_codes)
    meta["ops0_pearlc"] = {"M": len(a_codes), "N": len(b_codes), "K": k, "a_flavors": "16 gauss, 16 outliers, 16 relu",
                           "b_flavor": "weight N(0, 0.02^2)", "a_admissible": sum(a_ok), "b_admissible": sum(b_ok)}
    # a wide grid (x in {+-16, +-1/8}): products from 2^-6 to 2^8; exact while every prefix stays below 2^20 (g <= 2^-6)
    wide = [0x58, 0xD8, 0x20, 0xA0]                     # +16, -16, +1/8, -1/8
    a_w = [[rng.choice(wide) for _ in range(k)] for _ in range(16)]
    b_w = [[rng.choice(wide) for _ in range(k)] for _ in range(8)]
    # a half-integer grid (|x| <= 8 in steps of 1/2): products on 2^-2 up to 64, chains below 2^22: exact
    halves = [c for c in FINITE_E4M3 if (v := code_value(c)) == round(v * 2) / 2 and abs(v) <= 8]
    a_h = [[rng.choice(halves) for _ in range(k)] for _ in range(16)]
    b_h = [[rng.choice(halves) for _ in range(k)] for _ in range(8)]
    write_ops(outdir / "ops1_grids.bin", a_w + a_h, b_w + b_h)
    meta["ops1_grids"] = {"M": 32, "N": 16, "K": k, "rows": "A 0-15 x B 0-7: wide grid {+-16, +-1/8}; A 16-31 x B 8-15: "
                          "half grid |x| <= 8 step 1/2 (other pairs mixed)"}
    (outdir / "ops_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def code_value(c: int) -> float:
    s = -1.0 if c & 0x80 else 1.0
    e, m = (c >> 3) & 0xF, c & 7
    return s * (m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


def code_int(c: int) -> int:
    """An E4M3 value as an integer multiple of 2^-9."""
    v = code_value(c) * 512
    assert v == int(v)
    return int(v)


def theorem_chain(a: list[int], b: list[int], g: int | None) -> bool:
    """The theorem applied step by step: every atom step (each window starts from +0) meets both conditions, and every
    promotion add's exact sum is an FP32 value.  True means every intermediate is the exact prefix, so the word is exact."""
    atoms = len(a) // 32
    win = g or atoms
    tot = 0
    for w0 in range(0, atoms, win):
        acc_word = 0
        acc = 0
        for j in range(w0, min(w0 + win, atoms)):
            aa, bb = a[32 * j: 32 * j + 32], b[32 * j: 32 * j + 32]
            aligned, representable = explain(acc_word, aa, bb)
            if not (aligned and representable):
                return False
            acc = exact_sum(acc_word, aa, bb)
            acc_word = rz_f32_word(acc)
        tot += acc
        if word_int(rz_f32_word(tot)) != tot:
            return False
    return True


def compare(opsdir: Path, devdir: Path) -> dict:
    """Device words of every ops file against the model (`pearl_c.chain`, both devices) and the exact rational sums."""
    from verity.core.protocols.pouw.schemes import pearl_c
    from verity.core.protocols.pouw.schemes.pearl_c_device import SM120, SM120_UNPROMOTED

    out = {}
    for idx, name in enumerate(sorted(p.stem for p in opsdir.glob("ops*_*.bin"))):
        a_rows, b_rows = read_ops(opsdir / f"{name}.bin")
        ai = [[code_int(c) for c in r] for r in a_rows]
        bi = [[code_int(c) for c in r] for r in b_rows]
        m, n = len(a_rows), len(b_rows)
        for tag, dev in (("u", SM120_UNPROMOTED), ("p", SM120)):
            words = struct.unpack(f"<{m * n}I", (devdir / f"ops{idx}.{tag}.bin").read_bytes())
            stats = {"elements": m * n, "device_vs_model_mismatch": 0, "exact": 0, "correctly_rounded": 0,
                     "theorem_exact": 0, "theorem_exact_but_inexact": 0, "exact_without_theorem": 0,
                     "error_ulps": [], "error_sign": {"toward_zero": 0, "away": 0}, "rel_err_max": 0.0}
            blocks = {}
            for i in range(m):
                for j in range(n):
                    w = words[i * n + j]
                    model = pearl_c.chain(a_rows[i], b_rows[j], dev)
                    stats["device_vs_model_mismatch"] += w != model
                    exact = sum(x * y for x, y in zip(ai[i], bi[j]))          # at 2^-18
                    got = word_int(w)
                    ex = exact << (SCALE - 18)
                    key = f"A{i // 16}xB{j // 8}" if name.endswith("grids") else f"A{i // 16}"
                    blk = blocks.setdefault(key, {"elements": 0, "exact": 0})
                    blk["elements"] += 1
                    certified = theorem_chain(a_rows[i], b_rows[j], dev.g)
                    stats["theorem_exact"] += certified
                    stats["theorem_exact_but_inexact"] += certified and got != ex
                    stats["exact_without_theorem"] += (not certified) and got == ex
                    if got == ex:
                        stats["exact"] += 1
                        blk["exact"] += 1
                        continue
                    rne = struct.unpack("<I", struct.pack("<f", exact / 2.0 ** 18))[0]
                    stats["correctly_rounded"] += w == rne
                    ulp = 2.0 ** (math.floor(math.log2(abs(exact / 2.0 ** 18))) - 23) if exact else 2.0 ** -149
                    err = (got - ex) / 2.0 ** SCALE
                    stats["error_ulps"].append(abs(err) / ulp)
                    stats["error_sign"]["toward_zero" if abs(got) < abs(ex) else "away"] += 1
                    if exact:
                        stats["rel_err_max"] = max(stats["rel_err_max"], abs(err) / abs(exact / 2.0 ** 18))
            e = sorted(stats.pop("error_ulps"))
            stats["error_ulps_median_max"] = [e[len(e) // 2], e[-1]] if e else [0, 0]
            stats["by_block"] = blocks
            out[f"{name}.{tag}"] = stats
    return out


def main(argv: list[str]) -> int:
    if argv[:1] == ["model"]:
        res = {"closed_form_vs_model": check_closed_form(int(argv[1]) if len(argv) > 1 else 30000),
               "integer_single_steps": check_integer_single(), "integer_chains_on_model": check_integer_chains()}
        print(json.dumps(res, indent=1))
        return 0 if res["closed_form_vs_model"]["mismatches"] == 0 else 1
    if argv[:1] == ["ops"]:
        print(json.dumps(build_ops(Path(argv[1])), indent=1))
        return 0
    if argv[:1] == ["compare"]:
        res = compare(Path(argv[1]), Path(argv[2]))
        text = json.dumps(res, indent=1)
        if "--json" in argv:
            Path(argv[argv.index("--json") + 1]).write_text(text)
        print(text)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
