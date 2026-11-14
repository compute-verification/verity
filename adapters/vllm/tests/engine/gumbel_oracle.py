"""vLLM's Gumbel noise row, bit-exact and torch-free: Philox4x32-10 and `uint_to_uniform_float` from `program.sampling_rng`,
`__nv_logf` and `__nv_log1pf` from the libdevice transcription (`verity.definitions.numerics.libdevice`)."""
from __future__ import annotations

import numpy as np
from verity.definitions.numerics.libdevice import F32, nv_log1pf, nv_logf

from tests.program import sampling_rng as R


def gumbel_noise_exact(seed: int, pos: int, vocab_size: int, *, is_drafting: bool = False) -> np.ndarray:
    """vLLM `gumbel_noised_argmax`'s fp32 noise row, bit-exact: u = max(tl.rand(randint(seed, pos), v), 2^-31ish);
    g = -logf(-log1pf(-u))."""
    gs = R.gumbel_seed_of(seed, pos, is_drafting=is_drafting)
    keys: np.ndarray = np.arange(int(vocab_size), dtype=np.int32)
    u = np.maximum(R.tl_rand(gs, keys, seed_bits=32), R.TL_RAND_MIN_F32).astype(F32)
    return (-nv_logf((-nv_log1pf((-u).astype(F32))).astype(F32))).astype(F32)
