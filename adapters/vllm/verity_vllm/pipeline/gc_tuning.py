"""`verity_vllm.gc_tuning.tune` under the module's older path, for the two Match checkers (`check.match.global_match`,
`check.match.batch_decomp`) that import it from here until the layer table places `verity_vllm.gc_tuning` in core, where
`check` may read it."""
from __future__ import annotations

from verity_vllm.gc_tuning import tune

__all__ = ["tune"]
