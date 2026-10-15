"""γ for p512 at k = 2 under Daniel's 16:13Z rulings (decode-free, HBM capacity, node-level), from measured inputs.

γ = μ_chip + μ_sample + μ_rebuild + μ_host, with μ_rebuild = d/(k·x₁,adv) (memory-accounting's mainline model) and
μ_host = d·B_spare/ΣB_HBM: the words a host process can sketch (or DMA into a GPU) from host DRAM in one step, as a
share of the node's HBM, when host DRAM is inside the device but outside the audit.

x₁,adv = x₁/(s·(1 + h)): x₁ = 175 is ra-codec-tensor's p512 encode at the max clock (r20261012-095329-3aac, ÷ 1.16),
s the adversary's squaring speed-up over the best measured form (sqfloor at 512 bits: 1.0 measured, 1.2 the multiply
pipe at 100%, 1.45 the issue bound) and h the host CPUs' share of the GPUs' squaring rate (host_sq: GMP measured; an
AVX-512 IFMA kernel at its port bound: ten 52-bit limbs, about 130 IFMA and 60 other vector operations per 8 words, on
two vector ports at 3 GHz). Prints one table a row.
"""

from __future__ import annotations

MU_CHIP = 0.28
MU_SAMPLE = 0.10
K = 2
X1 = 175.0
GPUS = 8
GPU_SQ_PER_S = 5.13e10 * 1.16  # rates art at 2,092 MHz, times the clock to 2,430 MHz
B_HBM = 1308.5  # GB/s, the rates art's read
HOST_CORES = 128  # 2 × Xeon 6776P, physical
HOST_SQ = {"none": 0.0, "GMP (measured 1.12e7/core)": 1.12e7, "IFMA (port bound 2.5e8/core)": 2.5e8}
S = {"measured": 1.0, "pipe 100%": 1.2, "issue bound": 1.45}
B_SPARE = {"isolated (0)": 0.0, "measured, 48 threads of 1 socket (186)": 186.0,
           "2 sockets at that rate (372)": 372.0, "DDR5-6400 x 8 ch x 2 (819, spec)": 819.0,
           "virtio disk, if 5 GB/s (unmeasured)": 5.0, "NIC 400 Gb/s, whole words, no certifier": 50.0}
D = (1.27, 1.81)


def main() -> None:
    gpu_phi = GPUS * GPU_SQ_PER_S
    print("rebuild term, device + host CPUs (μ_chip + μ_sample + d/(k·x₁,adv)), d = 1.27 / 1.81")
    for sname, s in S.items():
        for hname, per_core in HOST_SQ.items():
            h = HOST_CORES * per_core / gpu_phi
            x = X1 / (s * (1 + h))
            g = [MU_CHIP + MU_SAMPLE + 100 * d / (K * x) for d in D]
            print(f"  s {sname:12} host {hname:28} h {100 * h:5.2f}%  x₁,adv {x:6.1f}  γ {g[0]:.2f}% / {g[1]:.2f}%")
    print("μ_host = d·B_spare/ΣB_HBM for a store outside the audit, ΣB_HBM = 8 × 1,308.5 GB/s; d = 1.27 / 1.81")
    for bname, b in B_SPARE.items():
        m = [100 * d * b / (GPUS * B_HBM) for d in D]
        print(f"  {bname:42} μ_host {m[0]:6.2f}% / {m[1]:6.2f}%")
    for budget in (0.10, 0.26):
        lo, hi = (budget / 100 * GPUS * B_HBM / d for d in D)
        print(f"  spare host read bandwidth for μ_host ≤ {budget:.2f}%: ≤ {hi:.1f} GB/s (d 1.81) to {lo:.1f} GB/s (d 1.27)")


if __name__ == "__main__":
    main()
