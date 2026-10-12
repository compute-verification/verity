"""numpy checks and encoding of a remote bandwidth pass's plan from an (R, 2) integer array of runs: the batch form of
`verity.core.protocols.remote_bandwidth.receipts.Plan`, whose `from_array` calls it.  ml imports no core, so that
package's tests hold the two to the same digests and the same refusals.

    checked(runs, blocks)   None if `runs` is not an (R, 2) integer array; else (encoding, firsts, ends), the plan's
                            u32be R ‖ (u64be first ‖ u64be count) × R and its runs' firsts and running ends as native
                            u64 bytes, or ValueError with the reference's message for the first of its rules broken

The rules are the reference's, in its order, each over every run: 1 to 2^32 − 1 runs; each run (first, count) with
count ≥ 1 inside a store of `blocks` (checked as first ≤ blocks − count, which no u64 overflows); no run starting where
the one before it ends; and disjoint runs.  The last is checked on the firsts and the ends sorted apart, the k-th end
at most the (k+1)-th first for every k: runs of count ≥ 1 are disjoint exactly then.  (Sorted by first, disjoint runs
are sorted by end too.  Conversely, the k runs ending by the (k+1)-th first start before it, so no two runs share a
first and those k are the first k by first: the k-th by first ends by the next one's first.)
`blocks` is the caller's, an int from 1 to 2^64 − 1.
"""

from __future__ import annotations

import struct

import numpy as np

RUNS = "a plan is 1 to 2^32 − 1 runs"
RUN = "a run is (first, count) with count ≥ 1, inside the store"
CONTINUED = "a run that starts where the one before it ends is part of it"
DISJOINT = "a plan's runs are disjoint: no block is read twice in a pass"


def checked(runs: object, blocks: int) -> tuple[bytes, bytes, bytes] | None:
    if (not isinstance(runs, np.ndarray) or runs.ndim != 2 or runs.shape[1] != 2
            or runs.dtype.kind not in "iu"):
        return None
    if not 1 <= len(runs) < 1 << 32:
        raise ValueError(RUNS)
    if runs.dtype.kind == "i" and (runs < 0).any():
        raise ValueError(RUN)
    r = runs.astype(np.uint64)
    first, count = r[:, 0], r[:, 1]
    n = np.uint64(blocks)
    if not ((count >= 1) & (count <= n) & (first <= n - np.minimum(count, n))).all():
        raise ValueError(RUN)
    end = first + count
    if (end[:-1] == first[1:]).any():
        raise ValueError(CONTINUED)
    if (np.sort(end)[:-1] > np.sort(first)[1:]).any():
        raise ValueError(DISJOINT)
    return struct.pack(">I", len(r)) + r.astype(">u8").tobytes(), first.tobytes(), np.cumsum(count).tobytes()
