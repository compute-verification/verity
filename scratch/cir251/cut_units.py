"""The long row's units as `dsa_topk_reads.cut` names them at the size test's two rows (N = WHOLE_MAX and the history
table's capacity), today and with the cut's five `_v1` unit Definitions replaced by their `_v2`s (the switch's part (a)).
Binds Definitions only; builds no circuit.  One JSON object on stdout."""
import json
import sys

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.commitments.gates.embedding import read
from verity.commitments.history import CAPACITY
from verity.definitions.registry import dsa_topk_mask_cut as X
from verity.definitions.registry import dsa_topk_reads as R

TOPK, LATENT = 2048, 576
SWAP = {(R, "DsaMaskPositions"): R.DsaMaskPositionsV2, (R, "DsaPositionsMerge"): R.DsaPositionsMergeV2,
        (X, "DsaTopKMaskWhole"): X.DsaTopKMaskWholeV2, (X, "DsaTopKDigitPick"): X.DsaTopKDigitPickV2,
        (X, "DsaTopKMaskBlock"): X.DsaTopKMaskBlockV2}


def units() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in (X.WHOLE_MAX, CAPACITY):
        row = bind(R.DsaTopKReadRows, N=n, TOPK=TOPK, BLOCK=R.BLOCK, READ=read(CAPACITY, LATENT))
        for u, c in R.cut(row):
            out.setdefault(descriptor_id(u), []).append(f"N={n} x{c}")
    return out


before = units()
for (mod, name), v2 in SWAP.items():
    setattr(mod, name, v2)
X.plan.cache_clear()
after = units()
json.dump({"capacity": CAPACITY, "before": before, "after": after,
           "gone": sorted(set(before) - set(after)), "new": sorted(set(after) - set(before)),
           "kept": sorted(set(before) & set(after))}, sys.stdout, indent=1)
