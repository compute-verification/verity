"""x₁, the decode cost a byte and γ at 2^768 − 825 and 2^1024 − 105 beside 2^512 − 569, from this directory's squaring
rates, priced on ra-codec-mix's table v5 (art:b839fbd0979f4b2a) by its step model (`benchmarks/codec_mix/table.py` on
cursor/ra-codec-mix-ea59).

x₁ at b bits is codec-mix's `x1_of`: 175 at p512, scaled by kv.cu's root bytes a second (338.6 at 768, 600 at 1,024).
Its reference squarer at b, R_ref, runs 5.13e10 times kv.cu's root squarings at b over its own at 512. An adversary's
factor over it is s·(1 + h), as in `gamma.py`: s is its GPU's squaring rate over R_ref (both at 2,092 MHz), h the host's
128 cores over 8 reference GPUs at the max clock. Then x₁,adv = x₁/(s·(1 + h)), and the factor is what codec-mix's
ADVERSARIES takes, now one a width. The honest encode squares with the same forms, so x₁ honest = x₁/s at the best one.

A ceiling model is cycles a warp-squaring a sub-partition: the probe's 4.63 a column product with its carry, 4.02 an
IMAD.WIDE alone, 0.6 a doubling shift, and about 1 cycle a Karatsuba combining add (75 at 16 limbs and one level,
scaled by the limbs, and by 1.5 a level deeper: three halves of half the size).

The decode is k squarings a word. kv.cu's decode time a byte places each codec on p512's k axis (κ). With the codec's
own decode and encode at the best measured form, its time a byte drops by kv.cu's squaring rate over the best (g), so
κ' = 2 + (g·t − t₂)/(t₃ − t₂) on p512's measured curves. d' is codec-mix's d with W at κ' and E at x₁ honest, and
attention held as measured: at 768 bits it is 3.1× its κ projection, the cost of a layout without tensor cores, which
a faster squaring doesn't remove.
"""

from __future__ import annotations

from gamma import GPUS, MHZ, MU, REF, REF_MHZ, X1, rate_of_cycles

C = {512: 569, 768: 825, 1024: 105}
CORES = 128
MAX_CLOCK = 1.16
#: kv.cu as built, at 2,092 MHz (ra-codec-mix's kv-codec.json, r20261012-183859-2055, art:c1625f3cebececb0): roots a
#: second (the fastest build of the width), squarings a second, and decode words a second by (lib, k).
KV_ROOTS = {512: 1.0592e8, 768: 3.6495e7, 1024: 1.5446e7}
KV_SQ = {512: 5.6012e10, 768: 2.8977e10, 1024: 1.6003e10}
KV_DECODE = {("p512", 2): 2.2524e10, ("p512", 3): 1.5422e10, ("p512-m", 2): 2.2021e10, ("p512-m", 3): 1.4998e10,
             ("p768-m", 2): 1.2156e10, ("p768-m", 3): 8.2402e9, ("p1024-m", 2): 7.2214e9, ("p1024-m", 3): 4.8782e9}
#: sq.sh's medians at 4 waves and MHZ, by FOLD: 512 r20261012-175857-517b, -183030-b044, -185914-35f2; 768
#: r20261012-194528-b277 (13, 14: -201257-9201); 1,024 r20261012-194557-c72f (13, 14: -201349-106e).
SQ = {512: {1: 5.71e10, 5: 6.531e10, 6: 6.40e10, 7: 6.35e10},
      768: {1: 2.9284e10, 5: 3.1857e10, 6: 3.3577e10, 7: 3.1328e10, 11: 2.8857e10, 12: 2.3019e10,
            13: 3.2987e10, 14: 3.1951e10},
      1024: {1: 1.7425e10, 5: 1.8612e10, 6: 1.9698e10, 7: 1.8252e10, 11: 1.8867e10, 12: 1.5812e10,
             13: 1.9545e10, 14: 1.8787e10}}
#: The best form less its MOV and SEL, cycles at the free fit's 1.16 an instruction (`cycle_fit.py --price`).
LESS_MOV_SEL = {512: 707, 768: 1360, 1024: 2365}
#: The host's IFMA, squarings a second a physical core, both threads, G = 1 (ifma.sh: r20261012-183544-2d44,
#: r20261012-195420-b71d, r20261012-195829-3929).
HOST_CORE = {512: 2.84e8, 768: 1.445e8, 1024: 7.0e7}

#: ra-codec-tensor's step at p512, as codec-mix's STEP: d, W, A, E at k = 1, 2, 3.
STEP = {
    "8B": {"d": (1.207, 1.360, 1.875), "W": (0.086, 0.114, 0.137), "A": (0.022, 0.127, 0.594),
           "E": (0.026, 0.052, 0.077)},
    "70B": {"d": (1.506, 1.813, 2.191), "W": (0.390, 0.640, 0.881), "A": (0.011, 0.065, 0.197),
            "E": (0.007, 0.015, 0.022)},
}
#: Table v5's masked rows: (b, k, lib, κ, attention measured over projected, d at 8B and 70B).
ROWS = {
    "p512 k=2 masked": (512, 2, "p512-m", 2.0498, 0.19250 / 0.18443, (1.391, 1.834)),
    "p512 k=3 masked": (512, 3, "p512-m", 3.0915, 0.70894 / 0.69373, (1.934, 2.230)),
    "p768 k=2 masked": (768, 2, "p768-m", 2.5106, 1.27102 / 0.41025, (2.426, 2.293)),
    "p768 k=3 masked": (768, 3, "p768-m", 3.7858, 2.04455 / 1.03450, (3.272, 2.798)),
    "p1024 k=2 masked": (1024, 2, "p1024-m", 3.2154, 0.98934 / 0.75510, (2.297, 2.369)),
    "p1024 k=3 masked": (1024, 3, "p1024-m", 4.8418, 1.82928 / 1.55166, (3.228, 3.009)),
}


def limbs(b: int) -> int:
    return b // 32


def col_cycles(n: int) -> float:
    return n * (n + 1) / 2 * 4.63 + n * 4.02 + (2 * n - 1) * 0.6


def kara_cycles(n: int, levels: int) -> float:
    h = n >> levels
    return 3**levels * h * (h + 1) / 2 * 4.63 + n * 4.02 + 75 * n / 16 * sum(1.5**j for j in range(levels))


def pipe_cycles(n: int, levels: int) -> float:
    h = n >> levels
    return (3**levels * h * (h + 1) / 2 + n) * 4.02


def r_ref(b: int) -> float:
    return REF * KV_ROOTS[b] * (b - 2) / (KV_ROOTS[512] * 510)


def x1_ref(b: int) -> float:
    return X1 * KV_ROOTS[512] * 64 / (KV_ROOTS[b] * b / 8)


def s_of(b: int, rate: float) -> float:
    return rate * REF_MHZ / MHZ / r_ref(b)


def h_of(b: int, core_rate: float) -> float:
    return CORES * core_rate / (GPUS * r_ref(b) * MAX_CLOCK)


def ifma_bound(b: int) -> float:
    n = -(-b // 52)
    return 8 * 2 * 3.9e9 / (n * n + 3 * n)


def best(b: int) -> tuple[int, float]:
    f = max(SQ[b], key=SQ[b].get)
    return f, SQ[b][f]


def ceilings(b: int) -> dict[str, float]:
    """s of each model at b bits."""
    n = limbs(b)
    out = {"column form at the probe's mix": s_of(b, rate_of_cycles(col_cycles(n))),
           "best form less its MOV and SEL": s_of(b, rate_of_cycles(LESS_MOV_SEL[b]))}
    for lv in (1, 2, 3):
        out[f"Karatsuba x{lv} at the probe's mix"] = s_of(b, rate_of_cycles(kara_cycles(n, lv)))
    out["column form, multiply pipe only"] = s_of(b, rate_of_cycles(pipe_cycles(n, 0)))
    for lv in (1, 2, 3):
        out[f"Karatsuba x{lv}, multiply pipe only"] = s_of(b, rate_of_cycles(pipe_cycles(n, lv)))
    return out


def adversaries(b: int) -> dict[str, float]:
    """codec-mix's factor at b bits: s·(1 + h)."""
    h_meas, h_bound = h_of(b, HOST_CORE[b]), h_of(b, ifma_bound(b))
    c = ceilings(b)
    kara = max(v for k, v in c.items() if k.startswith("Karatsuba") and "probe" in k)
    return {"measured best, host measured": s_of(b, best(b)[1]) * (1 + h_meas),
            "best less its MOV and SEL, host at the IFMA bound": c["best form less its MOV and SEL"] * (1 + h_bound),
            "Karatsuba at the probe's mix (best depth), host at the bound": kara * (1 + h_bound)}


def interp(ys: tuple[float, float, float], kappa: float) -> float:
    if kappa <= 2:
        return ys[0] + (ys[1] - ys[0]) * (kappa - 1)
    return ys[1] + (ys[2] - ys[1]) * (kappa - 2)


def step_d(point: str, kappa: float, k: int, x1: float, scale: float, kappa_a: float | None = None) -> float:
    """codec-mix's d (`row`, pad 0): the plain step less W, A and E at k = 2, plus W at κ, A at κ_a (κ unless given)
    and E at k·x₁."""
    s = STEP[point]
    base = s["d"][1] - s["W"][1] - s["A"][1] - s["E"][1]
    a = scale * interp(s["A"], kappa if kappa_a is None else kappa_a)
    return base + interp(s["W"], kappa) + a + s["E"][0] * k * x1 / X1


def ns_a_kb(lib: str, k: int, b: int) -> float:
    return 1e12 / (KV_DECODE[(lib, k)] * b / 8)


def g_of(b: int) -> float:
    return KV_SQ[b] / (best(b)[1] * REF_MHZ / MHZ)


def kappa_of(lib: str, k: int, b: int, g: float = 1.0) -> float:
    t2, t3 = ns_a_kb("p512", 2, 512), ns_a_kb("p512", 3, 512)
    return 2 + (g * ns_a_kb(lib, k, b) - t2) / (t3 - t2)


def decode_a_byte(b: int, k: int, rates: dict[int, float]) -> float:
    """k squarings over b/8 bytes, against p512's two over 64."""
    return (k / (rates[b] * b / 8)) / (2 / (rates[512] * 64))


def gamma(d: float, k: int, x1: float) -> float:
    return MU + 100 * d / (k * x1)


def main() -> None:
    for b in C:
        f, r = best(b)
        h_meas, h_bound = h_of(b, HOST_CORE[b]), h_of(b, ifma_bound(b))
        s = s_of(b, r)
        print(f"p{b} = 2^{b} - {C[b]}: R_ref {r_ref(b):.4e}/s, x₁ {x1_ref(b):.1f}; best FOLD {f} {r:.4e}/s "
              f"({rate_of_cycles(1) / r:.0f} cycles), {r * REF_MHZ / MHZ / KV_SQ[b]:.3f}x kv.cu, s {s:.3f}; "
              f"h {100 * h_meas:.1f}% measured, {100 * h_bound:.1f}% at the IFMA bound; x₁ honest {x1_ref(b) / s:.1f}")
        for name, v in ceilings(b).items():
            print(f"    {name:42} s {v:.3f}")
        for name, v in adversaries(b).items():
            print(f"  adversary {name:58} factor {v:.3f}  x₁,adv {x1_ref(b) / v:6.1f}")
    best_rates = {b: best(b)[1] for b in C}
    print("decode a byte against p512 k=2: kv.cu's squaring, the best forms', kv.cu's measured decode")
    for b in (768, 1024):
        for k in (2, 3):
            lib = f"p{b}-m"
            print(f"  p{b} k={k}: {decode_a_byte(b, k, KV_SQ):.3f}  {decode_a_byte(b, k, best_rates):.3f}  "
                  f"{ns_a_kb(lib, k, b) / ns_a_kb('p512-m', 2, 512):.3f}")
    print("γ = 0.38 + 100·d/(k·x₁,adv) (masked rows, no risk), 8B / 70B; d is table v5's, d' the best form's")
    for name, (b, k, lib, kappa, scale, d) in ROWS.items():
        x_h = x1_ref(b) / s_of(b, best(b)[1])
        kp = kappa_of(lib, k, b, g_of(b))
        dp = tuple(step_d(p, kp, k, x_h, scale, kappa_a=kappa) for p in STEP)
        print(f"  {name}: κ {kappa:.2f} → κ' {kp:.2f}; d {d[0]:.3f} / {d[1]:.3f}, d' {dp[0]:.3f} / {dp[1]:.3f}")
        for adv, fac in adversaries(b).items():
            x = x1_ref(b) / fac
            print(f"    {adv:58} x₁,adv {x:6.1f}  γ {gamma(d[0], k, x):.2f}% / {gamma(d[1], k, x):.2f}%  "
                  f"(d': {gamma(dp[0], k, x):.2f}% / {gamma(dp[1], k, x):.2f}%)")


if __name__ == "__main__":
    main()
