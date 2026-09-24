"""The two declared conversions, exhaustively over the 16-bit source domain (deliverable 3 mechanics)."""
from __future__ import annotations

import numpy as np

from verity_vllm.commit.fa2_prototype.derived import conv_bf16_to_f32, conv_f2fp_bf16

ALL_BF16 = np.arange(1 << 16, dtype=np.uint16)


def test_bf16_to_f32_is_total_injective_and_the_c10_lift() -> None:
    lifted = np.frombuffer(b"".join(conv_bf16_to_f32(w.tobytes()) for w in ALL_BF16), dtype=np.uint32)
    assert lifted.shape == (65536,)
    assert len(set(lifted.tolist())) == 65536                                  # injective
    assert np.array_equal(lifted, ALL_BF16.astype(np.uint32) << 16)            # c10 `f32_from_bits(x << 16)`
    finite = ~np.isnan(lifted.view(np.float32))
    # value semantics on the finite/inf domain: the fp32 value equals the bf16 value (numpy has no bf16; compare via truncation)
    back = (lifted[finite] >> 16).astype(np.uint16)
    assert np.array_equal(back, ALL_BF16[finite])


def test_f2fp_bf16_inverts_the_lift_on_non_nan_and_canonicalises_nan() -> None:
    for w in ALL_BF16:
        f32 = conv_bf16_to_f32(w.tobytes())
        got = np.frombuffer(conv_f2fp_bf16(f32), dtype=np.uint16)[0]
        if np.isnan(np.frombuffer(f32, dtype=np.float32)[0]):
            assert got == 0x7FFF                                              # F2FP.BF16 NaN word (attention-model.md §3 B5)
        else:
            assert got == w                                                   # RNE of an exactly representable value is the identity
