"""The routes outside design A's ring-program scope, priced on the FP machine against design A's word.

FP-NATIVE's scope card ("Card: the ring-program scope premise") quotes no ε for
design A until the machine model prices the shift class (Proposition Q's class) on sm_120 and B200 at no less
than the honest word, since an FP32 accumulator's rounding makes its shift free. This prices it, with the two other
routes outside the scope (target 2), against design A's honest row (fpx_design's
`fpx-a1-e4m3i-fp`) and its credit row (`-x0`), on the data sheets (DS-120, DS-100) and the measured vectors
(VF3r-16, DC-100-ri):

- `shift`: N = (s1 - s2)(s1 + s2) and (N - affine) >> v, two entries a product, operands per cell;
- `winograd`: (x1 + y2)(x2 + y1) less one-sided products, two entries a product, shift-free and exact, but its one-sided
  products are shared across rows (row-impure), and its operands are per cell;
- `packing`: a rounded MMA shared across the tile, A = b1 + 2^s b2 and B = 2^s t1 + t2, two entries a k-slot, its
  junk cleared by magnitude.

A tensor route is priced at two operand layouts: `shared` (every accumulator of an instruction useful, the most a tile
can do) and `per-cell` (an operand that depends on its cell serves one accumulator a k-slot in its row and one in its
column, so at most min(M, N) useful products a k-slot: Proposition Q's layout fact). Everything else is in the cheater's
favour: operands free, the shift free, any one-sided product free, products on lanes one instruction each in the
cheapest class that holds them, and the fold design A's own.

Run from the repository root: `python benchmarks/pouw/rt2_fp/shift_price.py [--json OUT]`.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import tomllib
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MM = ROOT / "benchmarks" / "pouw" / "machine_model"
sys.path.insert(0, str(MM))
from verity_machine_model.eps import machine  # noqa: E402

L = 32
LANES = 32

# fpx_design's design-A rows (benchmarks/pouw/fpx_design/fpx_eps.py at d7d292f583): the base design and the
# extraction it adds to the base's fold, lane instructions a word. The credit row (`-x0`) adds none.
DEVICES = {
    "sm_120": dict(data=MM / "data", base="dither-c-peel-fx1", fold_add={"ALU": 1.0, "IMAD": 1.0},
                   vectors={"DS-120": 0.5, "VF3r-16": 1.22}),
    "sm_100": dict(data=MM / "data" / "devices" / "sm_100", base="dither-2s-licpw-redux-heavy-l32",
                   fold_add={"ALU": 1.5, "IMAD": 1.5}, vectors={"DS-100": 0.5, "DC-100-ri": 1.22}),
}
NOISE = {"linears": "two-sided", "lm_head": "two-sided", "qk": "two-sided", "pv": "peel"}

# Each tensor form's shape (m, n, k) and whether it is 2:4 sparse: data/forms.toml on sm_120, the census kinds on sm_100.
SM100_FORMS = {"IMMA16816_S8": (16, 8, 16), "IMMA16832_S8": (16, 8, 32), "UTCHMMA_F16_M128_N256": (128, 256, 16),
               **{f"UTCIMMA_M{m}_N{n}": (m, n, 32) for m, n in ((128, 32), (128, 64), (128, 128), (128, 256),
                                                                (64, 64), (64, 128), (64, 256))},
               **{f"UTCQMMA_E4M3_M128_N{n}": (128, n, 32) for n in (128, 256)}}


def operand_formats(kind: str) -> tuple[str, ...]:
    """The operand formats a form reads (f8f6f4's every 8-, 6- and 4-bit format for the FP8 forms)."""
    if kind.startswith(("IMMA", "UTCIMMA")):
        return ("s8", "u8")
    if kind.startswith(("QMMA", "UTCQMMA")):
        return ("e4m3", "e5m2", "e3m2", "e2m3", "e2m1")
    if kind.startswith("OMMA"):
        return ("e2m1",)
    if kind in ("HMMA1688T", "HMMA1684T"):
        return ("tf32",)
    if kind.startswith(("HMMA", "UTCHMMA")):
        return ("f16", "bf16")
    raise ValueError(kind)


# Significand bits (with the implicit one) and the largest finite value.
FLOATS = {"e2m1": (2, 6), "e2m3": (4, 7.5), "e3m2": (3, 28), "e4m3": (4, 448), "e5m2": (3, 57344),
          "f16": (11, 65504), "bf16": (8, 3.39e38), "tf32": (11, 3.4e38)}


def holds(fmt: str, v: int) -> bool:
    """Whether the integer v is exactly a value of the format (every nonzero integer here is normal)."""
    if fmt == "s8":
        return -128 <= v <= 127
    if fmt == "u8":
        return 0 <= v <= 255
    p, top = FLOATS[fmt]
    a = abs(v)
    if a == 0:
        return True
    if a > top:
        return False
    return a % (1 << max(a.bit_length() - p, 0)) == 0


def accumulator_top(kind: str) -> int:
    """The largest integer the form's accumulator holds with every smaller one: s32, FP32 (p = 24, FpMma's sm_120 and
    B200 premise) or f16 (HMMA16816F16)."""
    if kind.startswith(("IMMA", "UTCIMMA")):
        return 2 ** 31 - 1
    return 2 ** 11 if kind == "HMMA16816F16" else 2 ** 24


def fits(kind: str, A: set[int], B: set[int], acc: int) -> bool:
    fm = operand_formats(kind)
    return (acc <= accumulator_top(kind) and any(all(holds(f, x) for x in A) for f in fm)
            and any(all(holds(f, x) for x in B) for f in fm))


# --- the routes' operands -------------------------------------------------------------------------------------------

def shift_operands(c_set: tuple[int, ...], M: int, v: int, data: tuple[int, ...] = (0,)) -> tuple[set, set]:
    """The shift class on two entries: s1 = d1 + β1 + q1 B1, s2 = d2 + β2 − q2 B2 with q = c·2^(v−1), β a
    bit, B = b + b′, d a data offset; the operands are a = s1 − s2 and b = s1 + s2. Checks, on the bits, that
    ((N − affine) >> v) mod 2^M is the two entries' bilinear part c1 β1 B1 + c2 β2 B2 when d = 0."""
    A, Bs = set(), set()
    for c1, c2 in product(c_set, repeat=2):
        q1, q2 = c1 << (v - 1), c2 << (v - 1)
        for b1, b2, x1, x1p, x2, x2p in product((0, 1), repeat=6):
            B1, B2 = x1 + x1p, x2 + x2p
            for d1, d2 in product(data, repeat=2):
                s1, s2 = d1 + b1 + q1 * B1, d2 + b2 - q2 * B2
                A.add(s1 - s2)
                Bs.add(s1 + s2)
            if data == (0,):
                s1, s2 = b1 + q1 * B1, b2 - q2 * B2
                N = (s1 - s2) * (s1 + s2)
                aff = b1 - b2 + (c1 * c1 << (2 * v - 2)) * B1 - (c2 * c2 << (2 * v - 2)) * B2
                assert (N - aff) % (1 << v) == 0
                got = ((N - aff) >> v) % (1 << M)
                assert got == (c1 * b1 * B1 + c2 * b2 * B2) % (1 << M), (c1, c2, v, M)
    return A, Bs


def e4m3i_grid(top: int = 448) -> list[int]:
    """Every integer E4M3 holds up to `top`, both signs."""
    pos = [x for x in range(top + 1) if holds("e4m3", x)]
    return sorted({s * x for x in pos for s in (1, -1)})


def winograd_operands(xs: list[int], ys: list[int]) -> set[int]:
    return {x + y for x in xs for y in ys}


@lru_cache(maxsize=None)
def packing_triples(n: int) -> frozenset[tuple[int, int, int]]:
    """Reachable (U, J_low, J_high) over n packed k-slots on zero data at gap 1: P̃ = β ∈ {0, 1}, X̃ = b + b′ − 1."""
    if n == 0:
        return frozenset({(0, 0, 0)})
    slot = {(b1 * t1 + b2 * t2, b1 * t2, b2 * t1) for b1, b2 in product((0, 1), repeat=2)
            for t1, t2 in product((-1, 0, 1), repeat=2)}
    return frozenset((u + a, jl + b, jh + c) for u, jl, jh in packing_triples(n - 1) for a, b, c in slot)


@lru_cache(maxsize=None)
def packing_decodes(n: int, s: int) -> bool:
    """Whether 2^s U + J_low + 2^(2s) J_high determines U over n slots."""
    seen: dict[int, int] = {}
    for u, jl, jh in packing_triples(n):
        t = (u << s) + jl + (jh << (2 * s))
        if seen.setdefault(t, u) != u:
            return False
    return True


@lru_cache(maxsize=None)
def packing_max_s(kind: str) -> int:
    """The largest s with A = β1 + 2^s β2 and B = 2^s t1 + t2 both in the form's operand formats."""
    best = -1
    for s in range(0, 24):
        A = {b1 + (b2 << s) for b1, b2 in product((0, 1), repeat=2)}
        B = {(t1 << s) + t2 for t1, t2 in product((-1, 0, 1), repeat=2)}
        if fits(kind, A, B, (L // 2) * max(abs(a * b) for a in A for b in B)):
            best = s
    return best


# --- the machine ----------------------------------------------------------------------------------------------------

def honest_rows(dev: str, tmp: Path) -> dict:
    """Design A's honest and credit rows on each vector, through the machine model's CLI, as fpx_design writes them."""
    d = DEVICES[dev]
    data = tmp / dev
    shutil.copytree(d["data"], data, ignore=shutil.ignore_patterns("devices") if dev == "sm_120" else None)

    def resolved(name: str) -> dict:
        t = tomllib.loads((data / "designs" / f"{name}.toml").read_text())
        under = resolved(t["base"]) if "base" in t else tomllib.loads((data / "designs.toml").read_text())
        return {**{k: v for k, v in under.items() if k not in ("name", "title", "summary", "aliases", "base")}, **t}

    base = resolved(d["base"])
    fold = dict(base["fold"]) if "fold" in base else dict(
        tomllib.loads((data / "honest" / f"{base['honest']}.toml").read_text())["fold"])
    fname = fold.pop("name")
    out = {}
    for row, add in (("honest", d["fold_add"]), ("credit", {})):
        f = dict(fold)
        for k, x in add.items():
            f[k] = f.get(k, 0.0) + x
        name = "rt2fp-a1" if add else "rt2fp-a1-x0"
        lines = [f'base = "{d["base"]}"', f'name = "{name}"', 'title = "design A (FPX-DESIGN fpx-a1-e4m3i-fp)"',
                 'summary = "RT2-FP"', "plain = 0.0", "", "[fold]",
                 f'name = "{fname}{"+x" if add else ""}"', *[f"{k} = {x!r}" for k, x in f.items()], "", "[noise]",
                 *[f'{k} = "{x}"' for k, x in NOISE.items()]]
        (data / "designs" / f"{name}.toml").write_text("\n".join(lines) + "\n")
        for vec, q in d["vectors"].items():
            r = subprocess.run([sys.executable, "-m", "verity_machine_model.cli", "--data", str(data), "eps", name,
                                "--credit", "floor", "--operands", "free", "--q", str(q), "--vector", vec, "--json"],
                               capture_output=True, text=True, cwd=MM, env={"PYTHONPATH": str(MM), "PATH": "/usr/bin:/bin"},
                               check=True)
            j = json.loads(r.stdout)
            fold_mix = {k: v for k, v in j["registry"]["fold"].items() if k != "name"}
            out[(row, vec)] = dict(flat=j["baseline"]["cycles_per_mma"], cheat=j["cheat"]["cycles_per_mma"],
                                   form=j["honest_form"], fold=fold_mix, binding=j["baseline"]["binding"],
                                   core=j["core"], q=q)
    return out, data


def forms_of(dev: str, data: Path) -> dict[str, tuple[int, int, int]]:
    """Each tensor form's (m, n, k_dense): k halved on a 2:4 sparse form, as data/forms.toml counts it."""
    if dev == "sm_100":
        return dict(SM100_FORMS)
    t = tomllib.loads((data / "forms.toml").read_text())
    return {k: (v["shape"][0], v["shape"][1], v["shape"][2] // (2 if v.get("sparse") else 1))
            for k, v in t.items() if isinstance(v, dict) and "shape" in v}


def per_word(pi, n: dict[str, float]) -> float:
    return max(pi.load(n).values())


def price_routes(dev: str, vec: str, data: Path, rows: dict, ops: dict) -> list[dict]:
    pi = machine(vec, data)
    forms = {k: s for k, s in forms_of(dev, data).items() if k in pi.c}
    out = []
    for row in ("honest", "credit"):
        h = rows[(row, vec)]
        m, n, _ = forms_of(dev, data)[h["form"]]
        words = m * n
        fold = {k: v / LANES for k, v in h["fold"].items()}
        ref = dict(flat=h["flat"] / words, cheat=h["cheat"] / words)
        for cls, (A, B, lane_classes, note) in ops.items():
            best = {}
            for kind, (fm, fn, fk) in forms.items():
                if cls == "packing":
                    s = packing_max_s(kind)
                    if s < 0:
                        continue
                    slots = max((k for k in range(1, min(fk, L // 2) + 1) if packing_decodes(k, s)), default=0)
                    if slots == 0:
                        continue
                    parts = math.ceil(L / (2 * slots))
                    layouts = {"shared": fm * fn / parts}
                    extra = {"ALU": (2 * parts - 1) / LANES}
                else:
                    if not fits(kind, A, B, (L // 2) * max(abs(a * b) for a in A for b in B)):
                        continue
                    layouts = {"shared": fm * fn * min(1.0, 2 * fk / L), "per-cell": min(fm * fn, min(fm, fn) * 2 * fk / L)}
                    extra = {}
                for lay, w in layouts.items():
                    nn = dict(fold)
                    for k, v in extra.items():
                        nn[k] = nn.get(k, 0.0) + v
                    nn[kind] = nn.get(kind, 0.0) + 1 / w
                    t = per_word(pi, nn)
                    if lay not in best or t < best[lay][0]:
                        best[lay] = (t, kind)
            if lane_classes:
                lane = []
                for k in lane_classes:
                    if k in pi.c:
                        nn = dict(fold)
                        nn[k] = nn.get(k, 0.0) + (L / 2) / LANES
                        lane.append((per_word(pi, nn), k))
                if lane:
                    best["lanes"] = min(lane)
            for lay, (t, kind) in best.items():
                out.append(dict(device=dev, vector=vec, row=row, route=cls, layout=lay, via=kind, per_word=t,
                                vs_flat=t / ref["flat"], vs_cheat=t / ref["cheat"], note=note))
        q = h["q"]
        out.append(dict(device=dev, vector=vec, row=row, route="model", layout="cheat", via=h["form"],
                        per_word=ref["cheat"], vs_flat=ref["cheat"] / ref["flat"], vs_cheat=1.0,
                        note=f"the model's own cheater at q* {q}, binding {','.join(h['binding'])}, core {h['core']:.5f}"))
    return out


# --- R2's step 3 ----------------------------------------------------------------------------------------------------

def v2(x: Fraction) -> float:
    if x == 0:
        return math.inf
    n, d, v = x.numerator, x.denominator, 0
    while n % 2 == 0:
        n //= 2
        v += 1
    while d % 2 == 0:
        d //= 2
        v -= 1
    return v


def step3_trial(rng: random.Random, minimal: bool = True) -> tuple[bool, float]:
    """One run of R2's step 3 on random integral weights w_s[x]: eliminate the b-slices, each pivot (term, b-slice) of
    least 2-adic valuation (or of most, the mutant). Returns whether every weight stayed integral and the least
    valuation of the accumulated coefficients ξ on the other slices, against −(the largest pivot valuation)."""
    terms, nb, no = rng.randint(4, 9), rng.randint(1, 3), rng.randint(1, 4)
    nb = min(nb, terms)
    W = [[Fraction(rng.choice((1, -1)) * rng.randint(1, 7) << rng.randint(0, 4)) if rng.random() < 0.8 else Fraction(0)
          for _ in range(nb + no)] for _ in range(terms)]
    bs, others = list(range(nb)), list(range(nb, nb + no))
    xi = {x: {y: Fraction(int(x == y)) for y in range(nb + no)} for x in range(nb + no)}
    used, ok, emax = set(), True, 0.0
    for _ in range(nb):
        cand = [(v2(W[s][y]), s, y) for s in range(terms) if s not in used for y in bs if W[s][y] != 0]
        if not cand:
            return ok, 0.0
        v, s, y = (min if minimal else max)(cand)
        emax = max(emax, v)
        for x in bs + others:
            if x == y:
                continue
            z = W[s][x] / W[s][y]
            for t in range(terms):
                if t not in used:
                    W[t][x] -= z * W[t][y]
            for k in xi[x]:
                xi[x][k] -= z * xi[y][k]
        used.add(s)
        bs.remove(y)
        ok &= all(v2(W[t][x]) >= 0 for t in range(terms) for x in bs + others)
    least = min((v2(c) for x in others for c in xi[x].values() if c != 0), default=0.0)
    return ok, least - (-emax)


def step3(trials: int = 3000, seed: int = 20261019) -> dict:
    rng = random.Random(seed)
    res = {"minimal": [0, 0, 0], "mutant": [0, 0, 0]}
    for _ in range(trials):
        for mode, minimal in (("minimal", True), ("mutant", False)):
            ok, slack = step3_trial(rng, minimal)
            res[mode][0] += 1
            res[mode][1] += (not ok)
            res[mode][2] += (slack < 0)
    return {m: dict(trials=a, weight_not_integral=b, xi_below_minus_e=c) for m, (a, b, c) in res.items()}


# --- main -----------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    grid = e4m3i_grid()
    zero_P, zero_X = [0, 1], [-1, 0, 1]
    coin7 = shift_operands((1,), M=7, v=8)
    coin32 = shift_operands((32,), M=32, v=23)
    gaps = tuple(1 << u for u in range(11))
    coin_word = shift_operands(gaps, M=17, v=18)
    for bad in ((1,), (32,)):
        M, v = (7, 7) if bad == (1,) else (32, 22)
        try:
            shift_operands(bad, M=M, v=v)
            raise SystemExit(f"the shift witness holds at v − 1 = {v}: RT-ALGO-3's bound is wrong")
        except AssertionError:
            pass
    wz = winograd_operands(zero_P, zero_X)
    wg = winograd_operands(grid, grid)
    ops = {
        "shift-gap1": (*coin7, ("FFMA",), "one k32 word's coin part at gap 1 (zero data), c = 1 mod 2^7, v = 8: "
                       "RT-ALGO-3's FP32-exact witness"),
        "shift-gaps": (*coin_word, ("FFMA", "IMAD"), "one k32 word's coin part at e4m3i's gaps (c to 2^10), mod 2^17, "
                       "v = 18"),
        "shift-s64": (*coin32, ("IMAD",), "the coin part mod 2^32 at c = 2^5, v = 23: RT-ALGO-3's s64 witness"),
        "winograd-zero": (wz, wz, ("FFMA",), "Winograd on zero data at gap 1: operands x + y in {-1, 0, 1, 2}"),
        "winograd-e4m3i": (wg, wg, ("FFMA",), "Winograd on e4m3i to 448: operands x + y to 896, FP32-exact over 16 "
                           "products (16·896² < 2^24)"),
        "packing": (set(), set(), (), "a rounded MMA shared across the tile on zero data at gap 1: two entries a "
                    "k-slot at the largest s the form's formats hold, its slots cut to what decodes, one lane "
                    "instruction a word to clear the high junk"),
    }
    tmp = Path(tempfile.mkdtemp(prefix="rt2fp-"))
    try:
        result = {"operands": {}, "packing": {}, "routes": [], "step3": step3()}
        for name, (A, B, _, note) in ops.items():
            if A:
                result["operands"][name] = dict(a=[min(A), max(A)], b=[min(B), max(B)],
                                                bits=max(abs(x) for x in A | B).bit_length(), note=note)
        need = min(s for s in range(1, 24) if packing_decodes(L // 2, s))
        result["packing"] = {"s_needed_16_slots": need,
                             "slots_at_s": {s: max((k for k in range(1, L // 2 + 1) if packing_decodes(k, s)), default=0)
                                            for s in range(1, need + 1)},
                             "full_word_s_needed": (2 * 16 * 448 * 448).bit_length() + 1}
        for dev in DEVICES:
            rows, data = honest_rows(dev, tmp)
            for vec in DEVICES[dev]["vectors"]:
                result["routes"] += price_routes(dev, vec, data, rows, ops)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("operands (a and b ranges, bits):")
    for k, o in result["operands"].items():
        print(f"  {k:15s} a {o['a']} b {o['b']} {o['bits']} bits — {o['note']}")
    p = result["packing"]
    print(f"packing on zero data: s ≥ {p['s_needed_16_slots']} decodes 16 slots; slots that decode at s: "
          f"{p['slots_at_s']}; full e4m3i words need s ≥ {p['full_word_s_needed']}")
    print(f"R2 step 3 (random integral weights): {result['step3']}")
    print(f"{'device':7s} {'vector':10s} {'row':7s} {'route':15s} {'layout':9s} {'via':24s} {'cyc/word':>9s} "
          f"{'/flat':>7s} {'/cheat':>7s}")
    for r in result["routes"]:
        print(f"{r['device']:7s} {r['vector']:10s} {r['row']:7s} {r['route']:15s} {r['layout']:9s} {r['via']:24s} "
              f"{r['per_word']:9.5f} {r['vs_flat']:7.3f} {r['vs_cheat']:7.3f}")
    worst = min((r for r in result["routes"] if r["route"] != "model"), key=lambda r: r["vs_cheat"])
    print(f"cheapest route outside the scope against the model's cheater: {worst['route']} {worst['layout']} via "
          f"{worst['via']} on {worst['device']} {worst['vector']} ({worst['row']}): {worst['vs_cheat']:.3f}×")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
