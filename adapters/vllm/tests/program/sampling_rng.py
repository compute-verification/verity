"""Torch-free recomputation of the uniforms behind vLLM d9105ea80's Gumbel-max sampler noise.

vLLM's sampler (`vllm/v1/worker/gpu/sample/gumbel.py::_gumbel_sample_kernel`) samples no random tensor from a torch
generator: for every logits row it evaluates, inside the kernel, a *pure function* of three integers --

    gumbel_seed = tl.randint(seed, pos)                    # Philox4x32-10, key = request seed (int64), counter = position
    u          = tl.rand(gumbel_seed, vocab_index)         # Philox4x32-10, key = gumbel_seed (uint32), counter = vocab index
    u          = max(u, 4.6566127342e-10)                  # fp32
    g          = -log(-log1p(-u))                          # fp32 (libdevice logf / log1pf on the GPU)
    token      = argmax_v(logits[v] / T + g[v])            # fp32, first index on ties

-- where `seed` is the request's `SamplingParams.seed` (or a host-side `np.random.randint` sample when None:
`sample/states.py`), and `pos` is the absolute position of the sampled token.  This module recomputes the first two lines with
numpy integers and one IEEE fp32 multiply (Triton's `python/triton/language/random.py`, v3.7.1, transcribed line by line):
Philox, the int32 -> fp32 conversion and the `* 2**-31` scaling are exact.  The clamp, the logs and the argmax are the sampler
Definition's (`verity/definitions/registry/sampling.py`), which the value check of a stochastic row evaluates
(`check.replay.stoch_recompute`).
"""

from __future__ import annotations

import numpy as np

# Triton `random.py` (32-bit path) and vLLM `gumbel.py` constants
PHILOX_KEY_A = np.uint32(0x9E3779B9)
PHILOX_KEY_B = np.uint32(0xBB67AE85)
PHILOX_ROUND_A = np.uint32(0xD2511F53)
PHILOX_ROUND_B = np.uint32(0xCD9E8D57)
TL_RAND_SCALE_F32 = np.float32(4.6566127342e-10)     # `uint_to_uniform_float`: (2**23 - 1) / 2**23 * 2**-31, rounded to fp32
TL_RAND_MIN_F32 = np.float32(4.6566127342e-10)       # vLLM `_TL_RAND_MIN` (clamp before the flipped Gumbel transform)
DRAFT_NOISE_SALT = 1 << 30
MASK32 = 0xFFFFFFFF


def philox4x32_10(c0, c1, c2, c3, k0, k1, rounds: int = 10):
    """Triton `philox_impl` for uint32 state: `rounds` Philox rounds of counter (c0..c3) under key (k0, k1).  Vectorised over
    numpy uint32 arrays (broadcast); identical to Random123's Philox4x32 (same multipliers, Weyl constants and lane order)."""
    c0, c1, c2, c3 = (np.asarray(x, dtype=np.uint32) for x in (c0, c1, c2, c3))
    k0, k1 = np.uint32(k0), np.uint32(k1)
    with np.errstate(over="ignore"):
        for _ in range(rounds):
            _c0, _c2 = c0, c2
            hi_b = ((_c2.astype(np.uint64) * np.uint64(PHILOX_ROUND_B)) >> np.uint64(32)).astype(np.uint32)
            hi_a = ((_c0.astype(np.uint64) * np.uint64(PHILOX_ROUND_A)) >> np.uint64(32)).astype(np.uint32)
            c0 = hi_b ^ c1 ^ k0
            c2 = hi_a ^ c3 ^ k1
            c1 = _c2 * PHILOX_ROUND_B          # low 32 bits (uint32 wraparound), Triton `tl.mul(..., sanitize_overflow=False)`
            c3 = _c0 * PHILOX_ROUND_A
            k0 = np.uint32((int(k0) + int(PHILOX_KEY_A)) & MASK32)
            k1 = np.uint32((int(k1) + int(PHILOX_KEY_B)) & MASK32)
    return c0, c1, c2, c3


def tl_randint4x(seed: int, offsets, *, seed_bits: int = 64):
    """Triton `randint4x(seed, offset)` -> four uint32 blocks.  `seed` is reinterpreted as uint64 (int64 two's complement when
    `seed_bits == 64`; a uint32 scalar such as `gumbel_seed` zero-extends, so its key high word is 0) and split into the Philox
    key (lo, hi); the offsets become the counter (lo word, hi word for >32-bit offsets, 0, 0)."""
    offsets = np.asarray(offsets)
    seed_u = int(seed) & (MASK32 if seed_bits == 32 else 0xFFFFFFFFFFFFFFFF)
    k0, k1 = seed_u & MASK32, (seed_u >> 32) & MASK32
    if offsets.dtype.itemsize > 4:
        off = offsets.astype(np.int64)
        lo = (off & MASK32).astype(np.uint32)
        hi = ((off >> 32) & MASK32).astype(np.uint32)      # arithmetic shift of the int64, then truncation to uint32
    else:
        lo = offsets.astype(np.int64).__and__(MASK32).astype(np.uint32)
        hi = np.zeros_like(lo)
    zero = np.zeros_like(lo)
    return philox4x32_10(lo, hi, zero, zero, k0, k1)


def tl_randint(seed: int, offsets, *, seed_bits: int = 64) -> np.ndarray:
    """Triton `randint`: the first Philox output word (uint32)."""
    return tl_randint4x(seed, offsets, seed_bits=seed_bits)[0]


def uint_to_uniform_float(x: np.ndarray) -> np.ndarray:
    """Triton `uint_to_uniform_float` (32-bit): bitcast to int32, fold negatives (`-x - 1`), scale by 2**-31-ish in fp32."""
    xi = np.asarray(x, dtype=np.uint32).view(np.int32)
    with np.errstate(over="ignore"):
        xi = np.where(xi < 0, -xi - np.int32(1), xi).astype(np.int32)
    return (xi.astype(np.float32) * TL_RAND_SCALE_F32).astype(np.float32)


def tl_rand(seed: int, offsets, *, seed_bits: int = 64) -> np.ndarray:
    """Triton `rand(seed, offset)`: fp32 uniforms in [0, 1)."""
    return uint_to_uniform_float(tl_randint(seed, offsets, seed_bits=seed_bits))


def gumbel_seed_of(seed: int, pos: int, *, is_drafting: bool = False) -> int:
    """`gumbel_seed = tl.randint(seed, pos)` (vLLM `gumbel_noised_argmax`): the per-(request seed, position) stream key."""
    p = int(pos) + (DRAFT_NOISE_SALT if is_drafting else 0)
    return int(tl_randint(int(seed), np.asarray([p], dtype=np.int64))[0])
