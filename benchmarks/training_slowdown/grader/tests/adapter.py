"""What the grader's tests take from the vLLM adapter (`verity_vllm`) that its public interface doesn't offer yet.

The adapter's public interface is the modules ``tests/test_vllm_public_api.py`` lists in ``PUBLIC`` and, from each, the names
in its ``__all__``. In the grader's tests this module is the one file that imports past it, each name once, and every other
file takes the name from here (``test_adapter.py``). A name gains a public home when the adapter lists it; then its users
import it from there and its line here goes.

What is left is the training routes the tiny step stands in on: the tables a test adds its own shapes to for the test's
length, and cuBLAS's split-K bounds the grader's k16 count reads.

| Here | Non-public home | Used by |
| --- | --- | --- |
| `TRAIN_CC`, `TRAIN_LINEAR_ROLES`, `TRAIN_ROUTES`, `aligned_bounds`, `residue_first` | `verity_vllm.definitions.registry.cublas_bf16` | conftest.py |
| `TRAIN_SDPA` | `verity_vllm.definitions.registry.fa2_train` | conftest.py |
| `TRAIN_SDPA_BWD` | `verity_vllm.definitions.registry.fa2_train_bwd` | conftest.py |
"""
from __future__ import annotations

from verity_vllm.definitions.registry.cublas_bf16 import (TRAIN_CC, TRAIN_LINEAR_ROLES, TRAIN_ROUTES, aligned_bounds,
                                                          residue_first)
from verity_vllm.definitions.registry.fa2_train import TRAIN_SDPA
from verity_vllm.definitions.registry.fa2_train_bwd import TRAIN_SDPA_BWD

__all__ = ["TRAIN_CC", "TRAIN_LINEAR_ROLES", "TRAIN_ROUTES", "TRAIN_SDPA", "TRAIN_SDPA_BWD", "aligned_bounds",
           "residue_first"]
