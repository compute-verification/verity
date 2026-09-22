"""A :class:`~verity_vllm.commit.TreeCache` that keeps the weight openings it has built (RC).

The reference harness re-reads a weight tensor from the bundle and rebuilds its ``RangeOpening``
(format v2: one leaf per row block) for every VU that opens it -- about 10 s for the 467 MB tied
embedding / lm_head table, 7 s for the 131072-row RoPE cache.  A verifier authenticates a weight
tensor once and reuses it; this cache does the same.  Nothing about authentication changes: the
opening object returned is the one the plain cache would have built.
"""

from __future__ import annotations

from verity_vllm.capture import CaptureReader
from verity_vllm.commit import Opening, RangeOpening, TreeCache


class MemoCache(TreeCache):
    def __init__(self, reader: CaptureReader, *, recompute_leaves: bool = True) -> None:
        super().__init__(reader, recompute_leaves=recompute_leaves)
        self._weight_openings: dict[str, Opening | RangeOpening] = {}
        self.weight_opens = 0
        self.weight_reuses = 0

    def open(self, canonical_id: str) -> Opening | RangeOpening:
        if canonical_id.startswith("w="):
            o = self._weight_openings.get(canonical_id)
            if o is None:
                self.weight_opens += 1
                o = self._weight_openings[canonical_id] = super().open(canonical_id)
            else:
                self.weight_reuses += 1
            return o
        return super().open(canonical_id)
