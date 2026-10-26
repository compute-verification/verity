"""The Triton kernels of the Pearl-C device path's liveness screen (`pouw_pearl_c_screen`, which imports this module at
first use: Triton is needed only there): ``screen``, `pouw_pearl_c_screen.screen` as one program a row; and ``scatter``, a
CUDA graph's last step, which screens the call's rows into the pass and copies its leaves and A root there, in one launch.
Both run ``screen_row``, one row's screen.

A row's screen is an FP64 sum of the squares at positions ≡ 0 (mod 8), the band's bounds rounded to FP32 as `screen` rounds
them, the dead and in-band counts, and the noise floor from the row's α and ρ; ``scatter`` with ``GATHER`` screens program
p's row at ``at[p]`` of x, α and ρ into flag pair p.  Its sum's order is not torch's, so a row can
come out unsure in one and decided in the other; both leave only rows within the band undecided, and `excluded` decides those
exactly, so the records are the same.
"""
import triton
import triton.language as tl


@triton.jit
def screen_row(x, alpha, aux, out, row, k, k8, sigma, band, BLOCK: tl.constexpr):
    base = x + row.to(tl.int64) * k
    acc = tl.zeros([BLOCK], dtype=tl.float64)
    for s in range(0, k, BLOCK):
        j = s + tl.arange(0, BLOCK)
        v = tl.load(base + j, mask=j < k, other=0.0).to(tl.float32)
        acc += tl.where(j % 8 == 0, (v * v).to(tl.float64), 0.0)
    thr = 64.0 * tl.sum(acc, axis=0) / k8
    hi = (thr * (1.0 + band)).to(tl.float32)
    lo = (thr * (1.0 - band)).to(tl.float32)
    dead = tl.zeros([BLOCK], dtype=tl.int32)
    near = tl.zeros([BLOCK], dtype=tl.int32)
    for s in range(0, k, BLOCK):
        j = s + tl.arange(0, BLOCK)
        live = j < k
        v = tl.load(base + j, mask=live, other=0.0).to(tl.float32)
        sq = v * v
        dead += tl.where(live & (sq > hi), 1, 0)
        near += tl.where(live & (sq >= lo) & (sq <= hi), 1, 0)
    d = tl.sum(dead, axis=0)
    b = tl.sum(near, axis=0)
    floor = tl.where(tl.load(aux + 2 * row + 1).to(tl.float64) * tl.load(alpha + row).to(tl.float64) < sigma, 1, 0)
    sure_fail = tl.where(64 * d > k, 1, 0)
    sure_pass = tl.where(64 * (d + b) <= k, 1, 0)
    tl.store(out, tl.maximum(floor, sure_fail).to(tl.uint8))
    tl.store(out + 1, ((1 - floor) * (1 - sure_fail) * (1 - sure_pass)).to(tl.uint8))

@triton.jit
def screen(x, alpha, aux, out, k, k8, sigma, band, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    screen_row(x, alpha, aux, out + 2 * row, row, k, k8, sigma, band, BLOCK)

@triton.jit
def scatter(x, alpha, aux, rows, k, k8, sigma, band, flags_to, leaves, leaves_to, root, roots_to, offs, n_leaves, n_root, at,
            BLOCK: tl.constexpr, GATHER: tl.constexpr):
    p, b0 = tl.program_id(0), tl.cdiv(n_leaves, BLOCK)
    if p < rows:
        if GATHER:
            row = tl.load(at + p)
        else:
            row = p
        screen_row(x, alpha, aux, flags_to + tl.load(offs + 1) + 2 * p, row, k, k8, sigma, band, BLOCK)
    elif p < rows + b0:
        i = (p - rows) * BLOCK + tl.arange(0, BLOCK)
        tl.store(leaves_to + tl.load(offs) + i, tl.load(leaves + i, mask=i < n_leaves), mask=i < n_leaves)
    else:
        i = tl.arange(0, BLOCK)
        tl.store(roots_to + tl.load(offs + 2) + i, tl.load(root + i, mask=i < n_root), mask=i < n_root)
