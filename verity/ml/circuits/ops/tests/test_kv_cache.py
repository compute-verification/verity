"""vLLM's BF16-to-E4M3 KV-cache append conversion."""
from __future__ import annotations

import random

import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.kv_cache import KvCacheAppendE4m3, kv_cache_append_e4m3_word


@pytest.mark.parametrize(
    "x,scale,want",
    [
        (0x3F80, 0x3F800000, 0x38),  # 1 / 1 = 1
        (0x3F80, 0x40000000, 0x30),  # 1 / 2 = 0.5
        (0x3F80, 0x3F000000, 0x40),  # 1 / 0.5 = 2
        (0xBF80, 0x00000000, 0xFE),  # -1 / +0 = -inf -> -448
        (0x0000, 0x00000000, 0x7F),  # 0 / 0 = NaN
        (0x7F80, 0x3F800000, 0x7E),  # +inf saturates
        (0x7FC1, 0x3F800000, 0x7F),  # BF16 NaN becomes E4M3 NaN
        (0x8000, 0x3F800000, 0x80),  # signed zero is preserved
    ],
)
def test_known_vllm_kv_cache_words(x: int, scale: int, want: int) -> None:
    assert kv_cache_append_e4m3_word(x, scale) == want
    assert evaluate(bind(KvCacheAppendE4m3), x, scale) == (want,)


def test_definition_matches_the_cpu_reference_on_every_bf16_class() -> None:
    rng = random.Random(20261014)
    xs = [0, 0x8000, 1, 0x8001, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF]
    scales = [0, 0x80000000, 1, 0x3F000000, 0x3F800000, 0x40000000, 0x7F800000, 0xFF800000, 0x7FC00000]
    xs += [rng.getrandbits(16) for _ in range(200)]
    scales += [rng.getrandbits(32) for _ in range(200)]
    definition = bind(KvCacheAppendE4m3)
    for x, scale in zip(xs, scales):
        assert evaluate(definition, x, scale) == (kv_cache_append_e4m3_word(x, scale),)
