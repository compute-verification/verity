"""What lane 0 of `_rejection_kernel` holds after the warp's halves split, under three readings of what a stopped lane gives
the next `shfl.sync`, for the hand vector `halves-split` (lane 0 accepts position 0 and lanes 16-31 refuse it).  CPU only.

After the split lanes 16-31 branch around the block that loads a tile's maximum and sum and runs the shuffles, so at each
later position lanes 0-15 read from them, in the first stage of the maximum (`%r45`) and of the sum (`%r133`, the product
`ex2 * sumexp`), words the PTX does not assign.  The readings:

    stale    the words the lane held at the split position: its tile's maximum and product of position 0
    zero     the word 0 for both
    padding  -inf for the maximum and 0 for the product, what a lane past the last tile loads

For each later position: lane 0's maximum, sum and log-sum-exp word under the reading, and whether it would accept.  The
word to compare with is the capture's kept `target_rejected_logsumexp` of this request, given as --card.

    python scratch/cir264/halves_stale.py [--card 0xc1af360a]
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "adapters" / "vllm", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SV = importlib.import_module("tests.program.sampled_accept_vectors")
A = importlib.import_module("verity.definitions.registry.sampled_accept")
SA = importlib.import_module("verity.definitions.registry.sampling")
fma32 = importlib.import_module("verity.definitions.registry.topp_split").fma32

F = np.float32
V, K, TILE, P = 154880, 5, A.STATS_TILE, A.PADDED_TILES


def _pad(m, s) -> tuple[np.ndarray, np.ndarray]:
    mm, ss = np.full(P, -np.inf, F), np.zeros(P, F)
    mm[:len(m)], ss[:len(s)] = np.asarray(m, F), np.asarray(s, F)
    return mm, ss


def lane0(m, s, upper_max, upper_product) -> tuple[np.float32, np.float32, int]:
    """Lane 0's maximum, sum and log-sum-exp word of a row when lanes 16-31 give the first stage ``upper_max`` and
    ``upper_product`` (16 words each) in place of their tiles'."""
    low = np.arange(16)
    top = np.maximum(m[:16], upper_max).max()
    with np.errstate(invalid="ignore", over="ignore"):
        a = fma32(A.tl_exp(m[:16] - top), s[:16], upper_product).astype(F)
        for k in (8, 4, 2, 1):
            a = (a + a[low ^ k]).astype(F)
        log = np.uint32(SA.nv_logf_gpu(int(a[:1].view(np.uint32)[0]), ftz=False)).view(F)
        return top, a[0], int(np.asarray(top + log, F).reshape(1).view(np.uint32)[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--card", type=lambda x: int(x, 0), default=None, help="the card's kept log-sum-exp word of the request")
    a = ap.parse_args()
    v = {x["name"]: x for x in SV.vectors(K, V, TILE)}["halves-split"]
    ref, logits, draft = v["ref"], v["logits"], v["draft"]
    assert ref["halves_split"] == 0
    m0, s0 = _pad(ref["max"][0], ref["sumexp"][0])
    readings = {"stale": (m0[16:], (A.tl_exp(m0 - m0.max()) * s0).astype(F)[16:]),
                "zero": (np.zeros(16, F), np.zeros(16, F)),
                "padding": (np.full(16, -np.inf, F), np.zeros(16, F))}
    for i in range(1, K):
        tiles = sorted({int(t) for t in np.flatnonzero(np.isfinite(logits[i])) // TILE})
        print(f"position {i}: tiles holding a finite logit {tiles}, the draft token's tile {draft[i] // TILE}, lane 0's "
              f"log-sum-exp with all 32 lanes {ref['lse'][i]:#010x}")
    for name, (um, up) in readings.items():
        print(name)
        live, last = True, None
        for i in range(1, K):
            top, total, lse = lane0(*_pad(ref["max"][i], ref["sumexp"][i]), um, up)
            lp = F(logits[i, draft[i]] - np.uint32(lse).view(F))
            test = bool(lp > np.uint32(ref["log_u"][i]).view(F))
            print(f"  position {i}: maximum {top!s:>10} sum {total!s:>14} log-sum-exp {lse:#010x} accepts {test}"
                  + ("" if live else " (not reached)"))
            if live:
                last = lse
            live = live and test
        note = "" if a.card is None else f"; the card's {a.card:#010x}: {'equal' if last == a.card else 'differs'}"
        print(f"  the last position reached keeps {last:#010x}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
