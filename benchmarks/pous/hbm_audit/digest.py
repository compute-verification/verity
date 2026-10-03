"""The bulk-bandwidth audit's digest: a keyed universal hash over a random set of 64 KiB blocks of the PoUS store `C`.

The verifier names a random set of `R` block indices of `C` and a fresh key; the server must return a digest that depends
on every byte of those blocks within the deadline. An honest GPU reads `R * 64 KiB` from HBM and folds it; a server that
keeps a fraction `f` of the blocks in host DRAM must move `f * R * 64 KiB` over PCIe first, which a deadline tuned to HBM
speed does not allow.

The digest is a two-level polynomial (Toeplitz / Carter--Wegman) hash over `GF(p)` with the Mersenne prime `p = 2^31 - 1`:

- each 64 KiB block is 16,384 little-endian `uint32` words `w[0..W-1]`; its block hash is `P_g(block) = sum_j w[j] g^{W-1-j}`
  evaluated at a verifier point `g` (Horner), so changing any word changes the value except with probability `W/p ~ 2^-17`;
- the block hashes are folded with a second verifier point `z`: `D = sum_r h[r] z^r mod p`, so changing, dropping or
  reordering any block changes `D` except with probability `R/p`.

One lane gives per-block almost-universality `~2^-17`; a deployment runs `HL` independent lanes with fresh `(g, z)` for
`~2^-17*HL` (the prototype measures `HL = 1` and `2` to show the digest stays HBM-bound either way). The field fits every
intermediate product in `int64` (`acc, g < 2^31`, so `acc*g < 2^62`), which is why the GPU kernel needs no 128-bit
arithmetic and runs at memory speed. The constants here are the audit's wire format, mirrored by the Triton kernel in
`device_digest` and the numpy reference in `reference_block_hash`; `tests/test_pous_hbm_audit.py` pins them.
"""

from __future__ import annotations

import numpy as np

BLOCK_BYTES = 65536
WORDS = BLOCK_BYTES // 4  # 16384 uint32 words per 64 KiB block
MOD = (1 << 31) - 1  # the Mersenne prime 2^31 - 1
INTERLEAVE = 1024  # the Triton kernel's SIMD width; a block is N_STEPS interleaved Horner passes
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


def _kernel2():
    """Two independent hash lanes in one read of each block, to show a stronger digest stays HBM-bound."""
    global _KERNEL2
    if _KERNEL2 is not None:
        return _KERNEL2
    import triton
    import triton.language as tl

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
            w = tl.load(c_ptr + base + step * INTERLEAVE + off)
            w = (w.to(tl.int64) & 0xFFFFFFFF) % p
            acc0 = (acc0 * gp0 + w) % p
            acc1 = (acc1 * gp1 + w) % p
        c0 = tl.load(combine0_ptr + off)
        c1 = tl.load(combine1_ptr + off)
        tl.store(out0_ptr + b, tl.sum((acc0 * c0) % p) % p)
        tl.store(out1_ptr + b, tl.sum((acc1 * c1) % p) % p)

    _KERNEL2 = block_hash2_kernel
    return _KERNEL2


def block_hashes2(C, idx, g0: int, g1: int):
    """Two lanes' per-block hashes from one read of each challenged block."""
    import torch

    kernel = _kernel2()
    r = idx.numel()
    out0 = torch.empty(r, dtype=torch.int64, device=C.device)
    out1 = torch.empty(r, dtype=torch.int64, device=C.device)

    def combine(g):
        return torch.tensor([pow(int(g), INTERLEAVE - 1 - lane, MOD) for lane in range(INTERLEAVE)],
                            dtype=torch.int64, device=C.device)

    kernel[(r,)](C, idx, combine(g0), combine(g1), out0, out1,
                 pow(int(g0), INTERLEAVE, MOD), pow(int(g1), INTERLEAVE, MOD), MOD, INTERLEAVE, N_STEPS)
    return out0, out1


def _kernel():
    """The Triton block-hash kernel (imported lazily so the module loads without torch/triton for the CPU reference test)."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def block_hash_kernel(
        c_ptr,  # int32 [N, WORDS], a block per row
        idx_ptr,  # int64 [R], the challenged rows
        combine_ptr,  # int64 [INTERLEAVE], g^{INTERLEAVE-1-l}
        out_ptr,  # int64 [R], the per-block hashes
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
            w = tl.load(c_ptr + base + step * INTERLEAVE + off)
            w = w.to(tl.int64) & 0xFFFFFFFF
            acc = (acc * gp + (w % p)) % p
        combine = tl.load(combine_ptr + off)
        h = tl.sum((acc * combine) % p)
        tl.store(out_ptr + b, h % p)

    _KERNEL = block_hash_kernel
    return _KERNEL


def block_hashes(C, idx, g: int, out=None):
    """Run the kernel: per-block hashes `P_g(C[idx[r]])` as an `int64 [R]` device tensor. `C` is `int32 [N, WORDS]`."""
    import torch

    kernel = _kernel()
    r = idx.numel()
    if out is None:
        out = torch.empty(r, dtype=torch.int64, device=C.device)
    gp = pow(int(g), INTERLEAVE, MOD)
    combine = torch.tensor(
        [pow(int(g), INTERLEAVE - 1 - lane, MOD) for lane in range(INTERLEAVE)],
        dtype=torch.int64,
        device=C.device,
    )
    kernel[(r,)](C, idx, combine, out, gp, MOD, INTERLEAVE, N_STEPS)
    return out


def fold(hashes, z: int) -> int:
    """Fold per-block hashes into the digest `sum_r h[r] z^r mod p` (tiny; reads only the `[R]` hash array)."""
    import torch

    r = hashes.numel()
    zp = torch.tensor(_z_powers(int(z), r), device=hashes.device)
    return int(((hashes % MOD) * zp % MOD).sum().item() % MOD)


def device_digest(C, idx, key: int, lane: int = 0) -> int:
    """The full digest on the device for one lane, as `reference_digest` computes it on the CPU."""
    g, z = lane_points(key, lane)
    return fold(block_hashes(C, idx, g), z)
