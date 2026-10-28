"""The filler a pass reads: a `filler_plan` for `verity_vllm.protocols.remote_bandwidth.NodeMemory` (`NodeMemory(filler_plan=...)`).

Each pass reads up to `budget(j)` bytes of idle registered blocks after the weights and the live KV: the KV regions'
`store.idle_blocks(region)` (free pages, and allocated pages' blocks no write has reached) and every filler region
whole (a sized pool's static filler in HBM, the host tier's). The blocks are taken in one rotation over all of them,
carried from pass to pass, so every idle block is read once a cycle and a pass's filler is never the same few blocks
(what keeps μ_chip at S_chip / |store| rather than at the size of one pass's filler). Torch-free; it reads only
`store.layout().regions` (`name`, `kind`, `blocks`, `block_bytes`) and `store.idle_blocks`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def slack(R_GBps: float, step_s: float, registered_bytes: float) -> int:
    """The bytes a step leaves of R·T: what a pass's filler would have to read for the step to reach R."""
    return max(0, int(R_GBps * 1e9 * step_s - registered_bytes))


class RotatingFiller:
    def __init__(self, budget: Callable[[int], int]) -> None:
        self.budget = budget
        self.cursor = 0

    def candidates(self, store) -> list[tuple[str, int, np.ndarray]]:
        out = []
        for info in store.layout().regions:
            if info.kind == "kv":
                out.append((info.name, info.block_bytes, np.asarray(store.idle_blocks(info.name), np.int64)))
            elif info.kind == "filler":
                out.append((info.name, info.block_bytes, np.arange(info.blocks, dtype=np.int64)))
        return out

    def __call__(self, store, j: int) -> Sequence[tuple[str, np.ndarray]]:
        cands = [c for c in self.candidates(store) if len(c[2])]
        total = sum(len(b) for _, _, b in cands)
        want = self.budget(j)
        if not total or want <= 0:
            return []
        starts = np.cumsum([0] + [len(b) for _, _, b in cands])
        pos, taken, out = self.cursor % total, 0, []
        while want > 0 and taken < total:
            k = int(np.searchsorted(starts, pos, side="right")) - 1
            name, bb, blocks = cands[k]
            lo = pos - int(starts[k])
            n = min(len(blocks) - lo, -(-want // bb), total - taken)
            out.append((name, blocks[lo:lo + n]))
            want -= n * bb
            taken += n
            pos = (pos + n) % total
        self.cursor = pos
        return out
