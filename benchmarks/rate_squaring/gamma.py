"""x₁ and γ for p512 at k = 2 from this directory's squaring rates, by ra-rate-redteam's formula (`node_gamma.py`,
branch cursor/ra-rate-redteam-3cf5 at 6fdc2a021): γ = μ_chip + μ_sample + d/(k·x₁,adv), x₁,adv = x₁/(s·(1 + h)), with
x₁ = 175 at the lanes' 5.13e10 squarings/s (2,092 MHz, ÷ 1.16 for the max clock), s a GPU's squaring rate over 5.13e10
at the same clock, and h the host CPUs' squaring rate over the eight GPUs' 8 × 5.13e10 × 1.16.

The honest encode squares with the same kernels, so x₁ honest is x₁/s at the best measured form. A rate in cycles is a
warp-squaring a sub-partition: 4 sub-partitions × 188 SMs × 32 lanes a clock. Prints one table a row.
"""

from __future__ import annotations

MU = 0.28 + 0.10  # μ_chip + μ_sample, the tracker's
K = 2
X1 = 175.0
REF, REF_MHZ = 5.13e10, 2092.0
MHZ = 2100.0  # the SM clock of this directory's runs (nvidia-smi before and after)
GPUS = 8
GPU_MAX = REF * 1.16
D = (1.27, 1.81)
LANES_A_CLOCK = 4 * 188 * 32


def rate_of_cycles(cycles: float) -> float:
    return LANES_A_CLOCK * MHZ * 1e6 / cycles


def s_of(rate: float) -> float:
    return rate * REF_MHZ / MHZ / REF


# Measured: sq.sh's median at 4 waves (r20261012-182613-5df1, r20261012-183030-b044). Modelled: cycles a warp-squaring.
GPU = {
    "FOLD 0, the red team's form (measured)": s_of(5.079e10),
    "FOLD 5, column-chained (measured best)": s_of(6.531e10),
    "column form at the probe's mix (713 cycles)": s_of(rate_of_cycles(713)),
    "Karatsuba at the probe's mix (640 cycles)": s_of(rate_of_cycles(640)),
    "column form, multiply pipe only (152 x 4.02)": s_of(rate_of_cycles(152 * 4.02)),
    "Karatsuba, multiply pipe only (124 x 4.02)": s_of(rate_of_cycles(124 * 4.02)),
}
# Squarings/s a physical core (ifma.sh, r20261012-183544-2d44: G = 1, both threads of 24 cores), and the IFMA-only port
# bound (130 IFMA per 8 words on two ports at 3.9 GHz); the node's 2 × Xeon 6776P have 128 cores, its VM shows 96.
HOST = {"none": 0.0, "IFMA measured, 128 cores": 128 * 2.84e8, "IFMA-only bound, 128 cores": 128 * 8 * 2 * 3.9e9 / 130}


def gamma(x: float, k: int) -> tuple[float, ...]:
    return tuple(MU + 100 * d / (k * x) for d in D)


def main() -> None:
    print(f"x₁ honest at FOLD 5: {X1 / GPU['FOLD 5, column-chained (measured best)']:.1f} (x₁ = {X1:.0f} at s = 1)")
    print("γ = μ_chip + μ_sample + d/(k·x₁,adv), d = 1.27 / 1.81, at k = 2 and k = 3; d* is where γ reaches 1% at k = 2")
    for gname, s in GPU.items():
        for hname, host in HOST.items():
            h = host / (GPUS * GPU_MAX)
            x = X1 / (s * (1 + h))
            g2, g3 = gamma(x, K), gamma(x, 3)
            d_star = (1 - MU) * K * x / 100
            print(f"  {gname:46} s {s:4.2f}  host {hname:26} h {100 * h:5.2f}%  x₁,adv {x:6.1f}  "
                  f"k2 γ {g2[0]:.2f}% / {g2[1]:.2f}%  d* {d_star:.2f}  k3 γ {g3[0]:.2f}% / {g3[1]:.2f}%")


if __name__ == "__main__":
    main()
