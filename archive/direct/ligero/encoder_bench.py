"""Encoder micro-benchmark on the committed shape (m rows x l -> n): the int64 radix-2 NTT (what the prover uses)
against the dense matmul DFT in INT8 limbs / INT32 accumulate on the tensor cores (``torch._int_mm``), one-stage
(l x n matrix) and two-stage (sqrt decomposition of the length-l transform; the n-point evaluation as 4 shifted
length-l transforms).  Correctness of the limb matmul is checked against the NTT on a slice.

    python -m archive.direct.ligero.encoder_bench --m 3372 --l 4096 --reps 5
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from .field import GEN, P, encode, root_of_unity


def _sync():
    torch.cuda.synchronize()


def timeit(fn, reps):
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def dft_matrix(l: int, n: int, device) -> torch.Tensor:
    """E (l x n): message evaluations on H (w_l^j) -> codeword on g<w_n>: E[j, c] = (1/l) sum_k w_l^(-jk) (g w_n^c)^k."""
    wl, wn = root_of_unity(l), root_of_unity(n)
    j = torch.arange(l, device=device, dtype=torch.int64)
    c = torch.arange(n, device=device, dtype=torch.int64)
    # Lagrange form: E[j, c] = (1/l) * sum_k (eta_c w_l^-j)^k = (1/l) * (eta_c^l - 1) / (eta_c w_l^-j - 1)
    eta = torch.tensor([GEN * pow(wn, int(ci), P) % P for ci in c.tolist()], device=device)
    eta_l = torch.tensor([pow(int(e), l, P) for e in eta.tolist()], device=device)
    winv = torch.tensor([pow(wl, (-int(ji)) % l, P) for ji in j.tolist()], device=device)
    linv = pow(l, P - 2, P)
    num = ((eta_l - 1) % P * linv) % P                                    # (n,)
    den = (winv[:, None] * eta[None, :] - 1) % P                          # (l, n), nonzero (coset)
    # batch inverse via Fermat on int64 is slow; use Python pow per element would be worse -- do it in chunks
    from .field import batch_modinv
    deninv = batch_modinv(den.reshape(-1)).reshape(l, n)
    return (num[None, :] * deninv) % P


def limbs(x: torch.Tensor, k: int) -> torch.Tensor:
    """(R, C) in [0, 2^(8k)) -> (k, R, C) int8-representable limbs (as int8 tensors: values 0..255 need uint8; we
    use signed limbs in [-128, 128) by centring, which _int_mm accepts)."""
    out = []
    for i in range(k):
        out.append(((x >> (8 * i)) & 255) - 128)
    return torch.stack(out).to(torch.int8)


def matmul_encode(W: torch.Tensor, E: torch.Tensor, n: int) -> torch.Tensor:
    """W (m, l) @ E (l, n) mod p via 4x4 INT8 limb products with INT32 accumulation (l <= 2^16 terms of < 2^16)."""
    m, l = W.shape
    Wl = limbs(W, 4)                       # (4, m, l)
    El = limbs(E, 4)                       # (4, l, n)
    acc = torch.zeros((m, n), dtype=torch.int64, device=W.device)
    # centring correction: (a-128)(b-128) = ab - 128 a - 128 b + 128^2; fold with row/col sums
    rowsum = [(Wl[i].to(torch.int64) + 128).sum(1) for i in range(4)]      # (m,)
    colsum = [(El[j].to(torch.int64) + 128).sum(0) for j in range(4)]      # (n,)
    for i in range(4):
        for j in range(4):
            prod = torch._int_mm(Wl[i], El[j]).to(torch.int64)             # (m, n) int32 exact
            prod = prod + 128 * rowsum[i][:, None] + 128 * colsum[j][None, :] - 128 * 128 * l
            acc = (acc + (prod % P) * pow(2, 8 * (i + j), P)) % P
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=3372)
    ap.add_argument("--l", type=int, default=4096)
    ap.add_argument("--rate-log2", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args(argv)
    dev = "cuda"
    m, l, n = args.m, args.l, args.l << args.rate_log2
    W = torch.randint(0, P, (m, l), dtype=torch.int64, device=dev)
    t_ntt = timeit(lambda: encode(W, n), args.reps)
    print(f"int64 radix-2 NTT encode ({m} x {l} -> {n}): {t_ntt*1e3:.1f} ms  "
          f"({m * n * 4 / t_ntt / 1e9:.1f} GB/s of codeword)")
    t0 = time.perf_counter()
    E = dft_matrix(l, n, dev)
    print(f"DFT matrix ({l} x {n}) built in {time.perf_counter() - t0:.1f}s")
    ref = encode(W[:64], n)
    got = matmul_encode(W[:64], E, n)
    print("limb matmul == NTT on 64 rows:", torch.equal(ref, got))
    # one-stage dense: 16 int8 matmuls of (m x l) @ (l x n)
    Wl = limbs(W, 4)
    El = limbs(E, 4)

    def one_stage_raw():
        for i in range(4):
            for j in range(4):
                torch._int_mm(Wl[i], El[j])
    t_raw = timeit(one_stage_raw, args.reps)
    macs = 16 * m * l * n
    print(f"one-stage dense INT8 matmul, 16 limb products only: {t_raw*1e3:.1f} ms = {2*macs/t_raw/1e12:.0f} TOPS")
    t_full = timeit(lambda: matmul_encode(W, E, n), args.reps)
    print(f"one-stage dense INT8 matmul + limb recombination mod p: {t_full*1e3:.1f} ms")
    # two-stage estimate: l = s^2; each length-l transform = 2 x (m*s x s) @ (s x s) matmuls + twiddle; n-point = 4 of them
    s = 1 << ((l.bit_length() - 1) // 2)
    s2 = l // s
    A = torch.randint(-128, 127, (m * s2, s), dtype=torch.int8, device=dev)
    B = torch.randint(-128, 127, (s, s), dtype=torch.int8, device=dev)
    A2 = torch.randint(-128, 127, (m * s, s2), dtype=torch.int8, device=dev)
    B2 = torch.randint(-128, 127, (s2, s2), dtype=torch.int8, device=dev)

    def two_stage_raw():
        for _ in range(4):                 # 4 shifted length-l transforms make the n = 4 l evaluations
            for _ in range(16):
                torch._int_mm(A, B)
                torch._int_mm(A2, B2)
    t_two = timeit(two_stage_raw, args.reps)
    print(f"two-stage ({s} x {s2}) INT8 matmuls only (x4 for n = 4l, no twiddles/recombination): {t_two*1e3:.1f} ms")


if __name__ == "__main__":
    main()
