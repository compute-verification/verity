"""circuit-check's bindings of `verity.core.protocols.certifier.compile_slice`'s Definition, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any

#: the synthetic engine the slice's tests trace (`tests/test_network_certifier_compile_slice.py`'s ENGINE)
ENGINE = {"max_num_seqs": 4, "max_num_batched_tokens": 16, "max_model_len": 64, "num_gpu_blocks": 20, "block_size": 4,
          "enable_prefix_caching": True, "enable_chunked_prefill": True, "scheduler_reserve_full_isl": True}


def _network_compile_slice_roots() -> list[Any]:
    """The compile slice at its tests' synthetic class: four steps at (R 2, W 1, A 2) of 4-token blocks, three
    arrivals, five decisions, a 32-bit codeword, a 6-entry prefix cache of 4-bit tokens and 4 units a request. The
    corpus slice's class (`tests/fixtures/compile_slice_smallkv.json`) is about 7.6M ANDs, past what circuit-check
    traces; the ZK run holds it to its reference."""
    from verity.core.protocols.certifier.compile_slice import statics, unit

    return [unit(**statics(2, 1, 2, ENGINE, 3, Q=3, D=5, P=4, K=3, SEG=32, CN=6, CTW=4, CU=4))]
