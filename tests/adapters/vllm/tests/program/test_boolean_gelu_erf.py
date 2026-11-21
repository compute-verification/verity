"""`verity_vllm.definitions.registry.boolean_gelu_erf`'s `GeluErfBf16_v2`, `act`'s measured `GeluErfBf16_v1` on bits: it is
Boolean, names its word primitive and is its Boolean version, its gates are pinned, and it gives the word on every one of the
65,536 bf16 words."""
from __future__ import annotations

import numpy as np
from verity.circuits import boolean as BL
from verity.circuits.boolean.trace import word_of
from verity.circuits.evaluation import bits as BS
from verity.circuits.ops.boolean.versions import boolean_version

from verity_vllm.definitions.registry import act as A
from verity_vllm.definitions.registry import boolean_gelu_erf as BG


def test_it_is_boolean_names_its_word_and_is_its_boolean_version() -> None:
    d = BG.GeluErfBf16
    assert word_of(d) is A.GeluErfBf16 and d.id == "GeluErfBf16_v2"
    assert BL.is_boolean(d.bind()) and boolean_version(A.GeluErfBf16).id == d.bind().id
    assert BS.compile(d.bind()).counts == {"and": 545, "xor": 5213, "not": 62}


def test_it_is_the_measured_word_on_every_bf16_word() -> None:
    x = np.arange(1 << 16, dtype=np.uint64)
    got = BS.evaluate_words(BG.GeluErfBf16.bind(), x.reshape(-1, 1)).reshape(-1).tolist()
    bad = [u for u in range(1 << 16) if got[u] != A.gelu_erf_bf16_bits(u)]
    assert not bad, [(hex(u), hex(got[u]), hex(A.gelu_erf_bf16_bits(u))) for u in bad[:5]]
