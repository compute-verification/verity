"""`verify_opening`: the verifier's side of one opening of a `Store`'s pass, the reference the store's openings are
tested against."""
from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from typing import Any

from verity.core.protocols.remote_bandwidth import receipts as rb
from verity.core.protocols.remote_bandwidth.receipts import PER_LEAF
from verity_vllm.protocols.remote_bandwidth.interface import Opening, PassCommitment, Receipt, RegionInfo
from verity_vllm.protocols.remote_bandwidth.registry import canonical, leaf_digest, node_of, plan_of, runs_of


def _climb(dom: Any, digest: bytes, c: int, level: int, path: Sequence[bytes]) -> bytes:
    for k, sibling in enumerate(path, level):
        at = c >> k
        digest = dom.node(k, at >> 1, *((digest, sibling) if at % 2 == 0 else (sibling, digest)))
    return digest


def verify_opening(info: RegionInfo, pc: PassCommitment, plan: Sequence[int], o: Opening, receipt: Receipt,
                   decode: Callable[[bytes, int], bytes], payload: bytes | None, z: bytes | None = None) -> bool:
    """The verifier's side of one opening (the reference): the plan's block at the position, its digest, the receipt
    recomputed from the block, its leaf (under the opened salt) through its segment's node, and the two paths together
    to the pass's root (the driver's `rb.verify_leaf`), and, with ``payload`` (what the block's registration source
    names), the block decoding to it."""
    k = bisect.bisect_right([s.position for s in pc.segments], o.position) - 1
    seg, c, positions = pc.segments[k], o.position // PER_LEAF, len(plan)
    if int(plan[o.position]) != o.block or seg.region != o.region or not info.first <= o.block < info.first + info.blocks:
        return False
    runs = canonical(runs_of(plan))
    try:
        p = plan_of(runs, int((runs[:, 0] + runs[:, 1]).max()))
    except ValueError:
        return False
    if p.digest != pc.plan_digest or sum(s.count for s in pc.segments) != positions:
        return False
    if receipt.keyed(pc.seed)(o.block, o.data, z) != o.receipt:
        return False
    if len(o.leaf_receipts) != min(PER_LEAF, positions - c * PER_LEAF) or o.leaf_receipts[o.position % PER_LEAF] != o.receipt:
        return False
    leaf = b"".join(o.leaf_receipts)
    dom, (level, _) = rb.domain(pc.seed, p), node_of(seg)
    if len(o.leaf_path) != level:
        return False
    if pc.segment_roots and _climb(dom, leaf_digest(dom, c, leaf, o.salt), c, 0, o.leaf_path) != pc.segment_roots[k]:
        return False
    if not rb.verify_leaf(pc.seed, p, pc.root, o.position, leaf, o.salt, o.leaf_path + o.segment_path):
        return False
    return payload is None or decode(o.data, info.word_index(o.block - info.first)) == payload
