"""circuit-check's bindings of `verity.core.protocols.certifier.compile_slice`'s Definition, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any

#: corpus-smallkv's engine (`benchmarks/network_e2e/encodings/tests/fixtures/corpus-smallkv`)
SMALLKV = {"max_num_seqs": 256, "max_num_batched_tokens": 256, "max_model_len": 4096, "num_gpu_blocks": 100,
           "block_size": 16, "long_prefill_token_threshold": 0, "enable_chunked_prefill": True,
           "enable_prefix_caching": True, "watermark": 0.0, "scheduler_reserve_full_isl": True}


def _network_compile_slice_roots() -> list[Any]:
    """The compile slice at its corpus slice's class: corpus-smallkv's first seven steps at (R 2, W 1, A 2), three
    arrivals, seven decisions, a 32-bit codeword (`tests/fixtures/compile_slice_smallkv.json`)."""
    from verity.core.protocols.certifier.compile_slice import statics, unit

    return [unit(**statics(2, 1, 2, SMALLKV, 6, Q=3, D=7, P=7, K=3, SEG=32))]
