"""The knobs of verity/remote/v0's design (the golden-demo brief) and their grids.

A design is a dict of knob -> value.  `grid(name)` is the product of the knobs' values in `GRIDS[name]`, in a fixed order,
so shard i of n is every n-th design from i (`shard`), and a design's index names it across runs.

Knobs (the brief's list):
- block size and fusion: `tile` (r x c output cells of a linear a proof block), `kchunk` (K cut into chunks, each
  committing its running sums), `fuse` (what's committed: every op's output; the int8 path once; gate_up, SiLU and
  quantize one Call);
- bits a committed word: `bits` (PoUW's prefix-sum words), `agg` (cells a word);
- noise: `noise` (Dither's permuted periodic noise, periodic without permutation, fresh along K), `lp` (the period L'),
  `t` (noise periods a segment, L = t L');
- commitment: `row` (the Commit's hm96 row bytes), `drow` (PoUW's value-root row bytes);
- sampling: `sampling` (q proportional to a block's gates, Achievability's; or one flat rate), `joint` (PoUW's check
  inside the values block of the same cells, or Dither's own proof blocks), `stmt` (a statement a template, or the
  universal unit's one a size class), `credit` (the K 2,048 calls, or all four linears);
- engineering: `seal` (b921's per-step tree, or roots only), `host` (today's host remainder, or none), `kernels` (the
  served hash kernels with the gather, or at the bar in place), `overlap` (none, the no-cost co-run share, 4 warps).
"""
from __future__ import annotations

from itertools import product

KNOBS = ("tile", "kchunk", "fuse", "bits", "agg", "noise", "lp", "t", "row", "drow", "sampling", "joint", "stmt",
         "credit", "seal", "host", "kernels", "overlap")

GRIDS: dict[str, dict[str, tuple]] = {
    "coarse": {
        "tile": ((1, 1), (1, 8), (4, 4), (8, 8), (16, 16), (32, 32)),
        "kchunk": (1, 2, 4),
        "fuse": ("all", "dedupe", "fuse"),
        "bits": (8, 12, 16),
        "agg": (1,),
        "noise": ("perm", "periodic", "fresh"),
        "lp": (32, 64, 128, 256, 512),
        "t": (1, 2, 3, 4, 8),
        "row": (1024, 4096, 16384),
        "drow": (1024, 4096, 16384),
        "sampling": ("work", "uniform"),
        "joint": ("separate", "joint"),
        "stmt": ("template", "universal"),
        "credit": ("k2048", "all4"),
        "seal": ("tree", "roots"),
        "host": ("today", "fixed"),
        "kernels": ("served", "bar"),
        "overlap": ("none", "corun", "corun4"),
    },
}
GRIDS["smoke"] = {k: v[:2] for k, v in GRIDS["coarse"].items()}
GRIDS["fine"] = {**GRIDS["coarse"],
                 "tile": ((1, 1), (1, 2), (2, 2), (1, 4), (2, 4), (4, 4), (1, 8), (8, 8), (16, 16), (32, 32)),
                 "kchunk": (1, 2, 4, 8),
                 "bits": (6, 8, 10, 12, 16),
                 "lp": (32, 64, 96, 128, 192, 256, 384, 512),
                 "t": (1, 2, 3, 4, 6, 8),
                 "row": (512, 1024, 2048, 4096, 8192, 16384),
                 "drow": (512, 1024, 2048, 4096, 8192, 16384)}


def size(name: str) -> int:
    n = 1
    for k in KNOBS:
        n *= len(GRIDS[name][k])
    return n


def grid(name: str):
    g = GRIDS[name]
    for i, values in enumerate(product(*(g[k] for k in KNOBS))):
        yield i, dict(zip(KNOBS, values))


def shard(name: str, i: int, n: int):
    for j, d in grid(name):
        if j % n == i:
            yield j, d


def key(d: dict) -> str:
    """A design's short name."""
    tile = "x".join(map(str, d["tile"]))
    return (f"tile{tile}/kc{d['kchunk']}/{d['fuse']}/b{d['bits']}/a{d['agg']}/{d['noise']}/L'{d['lp']}/t{d['t']}/"
            f"row{d['row']}/drow{d['drow']}/{d['sampling']}/{d['joint']}/{d['stmt']}/{d['credit']}/{d['seal']}/"
            f"{d['host']}/{d['kernels']}/{d['overlap']}")
