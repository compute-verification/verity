"""The decoder shapes the skeleton runs: ALLP-IFACE's `verity_torch_loop.weights.Shape` (CMP-95 item 1), whose tensors and
weights stream the store holds. `FROZEN` is the H200 demo's frozen shape; the others are small instances of the same MQA
architecture (one KV head, query heads covering the width) for CPU runs and tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "toy"))
from verity_torch_loop.weights import H200, TINY as IFACE_TINY, Shape  # noqa: E402

#: RMSNorm's epsilon, 1e-5 as the nearest f32 (circuits' plan, `EPS_1E5`)
EPS = 0x3727C5AC
#: PV's K: the probabilities and the value column padded with +0 to K (`GemmCoordinatePad_v1{T, K=32}`)
PV_K = 32

FROZEN = H200
#: width 32, 1 layer, 2 query heads of 16, MLP 64, context 8
TINY = IFACE_TINY
#: Phase 1's widths with the frozen shape's head size: width 128, 2 layers, 2 heads of 64, MLP 512
SMALL = Shape("e2e-mqa-w128-l2", 128, 2, 2, 64, 1, 512, 256, 32, 128)
#: a GPU slice: width 2,048, 4 layers, 32 heads of 64, MLP 8,192
SLICE = Shape("e2e-mqa-w2048-l4", 2048, 4, 32, 64, 1, 8192, 256, 32, 256)
SHAPES = {s.name: s for s in (FROZEN, SLICE, SMALL, TINY)} | {"h200": FROZEN, "slice": SLICE, "small": SMALL, "tiny": TINY}


def check(s: Shape) -> Shape:
    if s.kv_heads != 1 or s.heads * s.head_dim != s.width or s.context > PV_K:
        raise ValueError(f"{s.name}: the skeleton runs MQA (one KV head), heads x head_dim = width, context <= {PV_K}")
    return s
