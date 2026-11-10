"""The bulk-bandwidth audit's digest: a keyed universal hash over a random set of 64 KiB labels of the PoUS store `C`.

The verifier names a random set of `R` label indices of `C` and a fresh key; the server must return a digest that depends
on every byte of those labels within the deadline. An honest GPU reads `R * 64 KiB` from HBM and folds it; a server that
keeps a fraction `f` of the labels in host DRAM must move `f * R * 64 KiB` over PCIe first, which a deadline tuned to HBM
speed does not allow.

The digest is a two-level polynomial (Toeplitz / Carter--Wegman) hash over `GF(p)` with the Mersenne prime `p = 2^31 - 1`:

- each 64 KiB label is 16,384 little-endian `uint32` words `w[0..W-1]`; its label hash is `P_g(block) = sum_j w[j] g^{W-1-j}`
  evaluated at a verifier point `g` (Horner), so changing any word changes the value except with probability `W/p ~ 2^-17`;
- the label hashes are folded with a second verifier point `z`: `D = sum_r h[r] z^r mod p`, so changing, dropping or
  reordering any label changes `D` except with probability `R/p`.

One lane gives per-label almost-universality `~2^-17`; a deployment runs `HL` independent lanes with fresh `(g, z)` for
`~2^-17*HL` (the prototype measures `HL = 1` and `2` to show the digest stays HBM-bound either way). The field fits every
intermediate product in `int64` (`acc, g < 2^31`, so `acc*g < 2^62`), which is why the GPU kernel needs no 128-bit
arithmetic and runs at memory speed. The constants here are the audit's wire format, mirrored by the Triton kernel in
`device_digest` and the numpy reference in `reference_block_hash`; `tests/test_digest_reference.py` pins them.
"""

from __future__ import annotations

import numpy as np

BLOCK_BYTES = 65536
WORDS = BLOCK_BYTES // 4  # 16384 uint32 words per 64 KiB label
MOD = (1 << 31) - 1  # the Mersenne prime 2^31 - 1
INTERLEAVE = 1024  # the Triton kernel's SIMD width; a label is N_STEPS interleaved Horner passes
N_STEPS = WORDS // INTERLEAVE  # 16
assert INTERLEAVE * N_STEPS == WORDS


def lane_points(key: int, lane: int) -> tuple[int, int]:
    """The evaluation points `(g, z)` of hash lane `lane` under a verifier key, both in `[2, p-1]` (never 0 or 1)."""
    g = (((key ^ (0x9E3779B97F4A7C15 * (lane + 1))) & 0x7FFFFFFF) % (MOD - 2)) + 2
    z = (((key ^ (0xC2B2AE3D27D4EB4F * (lane + 7))) & 0x7FFFFFFF) % (MOD - 2)) + 2
    return int(g), int(z)


def reference_block_hash(words: np.ndarray, g: int) -> int:
    """`P_g(block) = sum_j w[j] g^{W-1-j} mod p` by Horner, the oracle the kernel must match. `words` is `uint32`."""
    acc = 0
    for w in words.astype(np.uint64):
        acc = (acc * g + (int(w) % MOD)) % MOD
    return acc


def reference_block_hash_interleaved(words: np.ndarray, g: int) -> int:
    """The same value computed as the kernel computes it: `INTERLEAVE` interleaved Horner lanes in `g^INTERLEAVE`, then a
    combine by `g^{INTERLEAVE-1-l}`. Pinned equal to `reference_block_hash` by the test so the kernel's layout is checked
    against the plain polynomial."""
    gp = pow(g, INTERLEAVE, MOD)
    acc = [0] * INTERLEAVE
    w = words.astype(np.uint64)
    for step in range(N_STEPS):
        base = step * INTERLEAVE
        for lane in range(INTERLEAVE):
            acc[lane] = (acc[lane] * gp + (int(w[base + lane]) % MOD)) % MOD
    total = 0
    for lane in range(INTERLEAVE):
        total = (total + acc[lane] * pow(g, INTERLEAVE - 1 - lane, MOD)) % MOD
    return total


def reference_digest(blocks: np.ndarray, idx: np.ndarray, key: int, lane: int = 0) -> int:
    """The full audit digest `D = sum_r P_g(C[idx[r]]) z^r mod p` for one lane. `blocks` is `[N, WORDS] uint32`."""
    g, z = lane_points(key, lane)
    acc = 0
    zp = 1
    for r in range(len(idx)):
        h = reference_block_hash(blocks[int(idx[r])], g)
        acc = (acc + h * zp) % MOD
        zp = (zp * z) % MOD
    return acc


def _z_powers(z: int, r: int) -> np.ndarray:
    out = np.empty(r, dtype=np.int64)
    cur = 1
    for i in range(r):
        out[i] = cur
        cur = (cur * z) % MOD
    return out


_KERNEL = None
_KERNEL2 = None


def _build():
    """Compile the Triton kernels once. The modular arithmetic uses Mersenne reduction for `p = 2^31 - 1`: a 62-bit
    product is folded by `(x & p) + (x >> 31)` twice and one conditional subtract, with no 64-bit division, so the kernel
    is memory-bound (an integer `% p` on the device is a slow remainder that would make the digest compute-bound and blur
    the HBM-vs-PCIe separation). `_mm(a, g, w, p) = (a*g + w) mod p` and `_combine` returns `sum_l acc[l]*c[l] mod p`; both
    give the same value as the plain `% p` reference, which `cmd_validate` checks bit-exactly."""
    global _KERNEL, _KERNEL2
    if _KERNEL is not None:
        return _KERNEL, _KERNEL2
    import triton
    import triton.language as tl

    @triton.jit
    def _red(x, p):
        # x in [0, 2^62): two folds then a conditional subtract land it in [0, p).
        x = (x & p) + (x >> 31)
        x = (x & p) + (x >> 31)
        return tl.where(x >= p, x - p, x)

    @triton.jit
    def _mm(a, g, w, p):
        return _red(a * g + w, p)

    @triton.jit
    def _combine(acc, c, p):
        # sum_l acc[l]*c[l] mod p; each product < 2^62 is reduced before the sum so the sum stays < INTERLEAVE*p < 2^62.
        return _red(tl.sum(_red(acc * c, p)), p)

    @triton.jit
    def block_hash_kernel(
        c_ptr,  # int32 [N, WORDS], a label per row
        idx_ptr,  # int64 [R], the challenged rows
        combine_ptr,  # int64 [INTERLEAVE], g^{INTERLEAVE-1-l}
        out_ptr,  # int64 [R], the per-label hashes
        gp,  # int64 scalar, g^INTERLEAVE mod p
        p,  # int64 scalar, the modulus
        INTERLEAVE: tl.constexpr,
        N_STEPS: tl.constexpr,
    ):
        b = tl.program_id(0)
        row = tl.load(idx_ptr + b)
        base = row * (INTERLEAVE * N_STEPS)
        off = tl.arange(0, INTERLEAVE)
        acc = tl.zeros([INTERLEAVE], dtype=tl.int64)
        for step in range(N_STEPS):
            w = tl.load(c_ptr + base + step * INTERLEAVE + off).to(tl.int64) & 0xFFFFFFFF
            w = (w & p) + (w >> 31)  # a 32-bit word -> [0, p] in one fold
            acc = _mm(acc, gp, w, p)
        tl.store(out_ptr + b, _combine(acc, tl.load(combine_ptr + off), p))

    @triton.jit
    def block_hash2_kernel(
        c_ptr, idx_ptr, combine0_ptr, combine1_ptr, out0_ptr, out1_ptr, gp0, gp1, p,
        INTERLEAVE: tl.constexpr, N_STEPS: tl.constexpr,
    ):
        b = tl.program_id(0)
        row = tl.load(idx_ptr + b)
        base = row * (INTERLEAVE * N_STEPS)
        off = tl.arange(0, INTERLEAVE)
        acc0 = tl.zeros([INTERLEAVE], dtype=tl.int64)
        acc1 = tl.zeros([INTERLEAVE], dtype=tl.int64)
        for step in range(N_STEPS):
            w = tl.load(c_ptr + base + step * INTERLEAVE + off).to(tl.int64) & 0xFFFFFFFF
            w = (w & p) + (w >> 31)
            acc0 = _mm(acc0, gp0, w, p)
            acc1 = _mm(acc1, gp1, w, p)
        tl.store(out0_ptr + b, _combine(acc0, tl.load(combine0_ptr + off), p))
        tl.store(out1_ptr + b, _combine(acc1, tl.load(combine1_ptr + off), p))

    _KERNEL, _KERNEL2 = block_hash_kernel, block_hash2_kernel
    return _KERNEL, _KERNEL2


def _kernel():
    """The one-lane kernel that hashes labels (built lazily so the module imports without torch/triton for the CPU
    reference)."""
    return _build()[0]


def _kernel2():
    """Two independent hash lanes in one read of each label, to show a stronger digest stays HBM-bound."""
    return _build()[1]


def block_hashes2(C, idx, g0: int, g1: int):
    """Two lanes' per-label hashes from one read of each challenged label. Outputs live on `idx`'s device (the CUDA
    device), so `C` may be a pinned host tensor read over PCIe (UVA zero-copy)."""
    import torch

    kernel = _kernel2()
    r = idx.numel()
    out0 = torch.empty(r, dtype=torch.int64, device=idx.device)
    out1 = torch.empty(r, dtype=torch.int64, device=idx.device)

    def combine(g):
        return torch.tensor([pow(int(g), INTERLEAVE - 1 - lane, MOD) for lane in range(INTERLEAVE)],
                            dtype=torch.int64, device=idx.device)

    kernel[(r,)](C, idx, combine(g0), combine(g1), out0, out1,
                 pow(int(g0), INTERLEAVE, MOD), pow(int(g1), INTERLEAVE, MOD), MOD, INTERLEAVE, N_STEPS)
    return out0, out1


def block_hashes(C, idx, g: int, out=None):
    """Run the kernel: per-label hashes `P_g(C[idx[r]])` as an `int64 [R]` device tensor. `C` is `int32 [N, WORDS]`; it
    may be a pinned host tensor (UVA zero-copy) since the outputs are keyed off `idx`'s (CUDA) device."""
    import torch

    kernel = _kernel()
    r = idx.numel()
    if out is None:
        out = torch.empty(r, dtype=torch.int64, device=idx.device)
    gp = pow(int(g), INTERLEAVE, MOD)
    combine = torch.tensor(
        [pow(int(g), INTERLEAVE - 1 - lane, MOD) for lane in range(INTERLEAVE)],
        dtype=torch.int64,
        device=idx.device,
    )
    kernel[(r,)](C, idx, combine, out, gp, MOD, INTERLEAVE, N_STEPS)
    return out


def fold(hashes, z: int) -> int:
    """Fold per-label hashes into the digest `sum_r h[r] z^r mod p` (tiny; reads only the `[R]` hash array)."""
    import torch

    r = hashes.numel()
    zp = torch.tensor(_z_powers(int(z), r), device=hashes.device)
    return int(((hashes % MOD) * zp % MOD).sum().item() % MOD)


# --- precomputed-table variants: the key-dependent tables (combine, z-powers) are built once, off the timed path, so a
# --- measurement times only the device read + reduce, the way a native responder (which expands the key in C/CUDA, not a
# --- Python loop) would. `driver.py` times these; `block_hashes`/`device_digest`/`fold` above stay for the bit-exact
# --- `validate` check, which is not timed.


def combine_table(g: int, device):
    import torch

    return torch.tensor([pow(int(g), INTERLEAVE - 1 - l, MOD) for l in range(INTERLEAVE)],
                        dtype=torch.int64, device=device)


def z_powers_dev(z: int, r: int, device):
    import torch

    return torch.tensor(_z_powers(int(z), r), dtype=torch.int64, device=device)


def block_hashes_pre(C, idx, combine, gp: int, out):
    """One lane's per-label hashes with a precomputed `combine` table and `gp = g^INTERLEAVE mod p` into a reused `out`."""
    _kernel()[(idx.numel(),)](C, idx, combine, out, gp, MOD, INTERLEAVE, N_STEPS)
    return out


def block_hashes2_pre(C, idx, c0, c1, gp0: int, gp1: int, out0, out1):
    _kernel2()[(idx.numel(),)](C, idx, c0, c1, out0, out1, gp0, gp1, MOD, INTERLEAVE, N_STEPS)
    return out0, out1


def fold_pre(hashes, zp) -> int:
    """Fold with precomputed device z-powers `zp` (a GPU reduction + one scalar read-back)."""
    return int(((hashes % MOD) * zp % MOD).sum().item() % MOD)


def device_digest(C, idx, key: int, lane: int = 0) -> int:
    """The full digest on the device for one lane, as `reference_digest` computes it on the CPU."""
    g, z = lane_points(key, lane)
    return fold(block_hashes(C, idx, g), z)
