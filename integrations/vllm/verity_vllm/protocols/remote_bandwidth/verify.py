"""`verify_opening`: the verifier's side of one opening of a `NodeMemory`'s pass, the reference the store's openings are
tested against."""
from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence

from verity.core.protocols.remote_bandwidth import receipts as rb
from verity_vllm.protocols.remote_bandwidth.interface import Opening, PassCommitment, Receipt, RegionInfo
from verity_vllm.protocols.remote_bandwidth.registry import canonical, plan_of, runs_of


def verify_opening(info: RegionInfo, pc: PassCommitment, plan: Sequence[int], o: Opening, receipt: Receipt,
                   decode: Callable[[bytes, int], bytes], payload: bytes | None, z: bytes | None = None) -> bool:
    """The verifier's side of one opening (the reference): the plan's block at the position, its digest, the receipt
    recomputed from the block and found at the position in its leaf (`rb.leaf_receipt`), the leaf under the opened
    salt and its two paths together to the pass's root (`rb.verify_leaf`), and, with ``payload`` (what the block's
    registration source names), the block decoding to it."""
    seg = pc.segments[bisect.bisect_right([s.position for s in pc.segments], o.position) - 1]
    if int(plan[o.position]) != o.block or seg.region != o.region or not info.first <= o.block < info.first + info.blocks:
        return False
    runs = canonical(runs_of(plan))
    try:
        p = plan_of(runs, int((runs[:, 0] + runs[:, 1]).max()))
    except ValueError:
        return False
    if p.digest != pc.plan_digest or sum(s.count for s in pc.segments) != len(plan):
        return False
    leaf = b"".join(o.leaf_receipts)
    if not receipt.keyed(pc.seed)(o.block, o.data, z) == o.receipt == rb.leaf_receipt(leaf, o.position, len(plan)):
        return False
    if not rb.verify_leaf(pc.seed, p, pc.root, o.position, leaf, o.salt, o.leaf_path + o.segment_path):
        return False
    return payload is None or decode(o.data, info.word_index(o.block - info.first)) == payload
