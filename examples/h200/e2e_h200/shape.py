"""The decoder shapes the skeleton runs: `served_weights.Shape`, whose tensors and
weights stream the store holds. `FROZEN` is the H200 demo's frozen shape (the toy); the others are small instances of the same
MQA architecture (one KV head, query heads covering the width) for CPU runs and tests.

`Forms` are the row forms of circuits' FAMILIES map (`plan`) at a shape: the frozen shape's are circuits', a smaller shape takes the same forms capped at its widths, and `MINI`'s are small ones
that exercise every level of the map (several chunk rows, RMSNorm groups, quantizer chunks and amax parts) on the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

from e2e_h200.served_weights import H200, TINY as IFACE_TINY, Shape

#: RMSNorm's epsilon, 1e-5 as the nearest f32 (circuits' plan, `EPS_1E5`)
EPS = 0x3727C5AC
#: attention's keys, padded: `QkSoftmaxRowPad_v1{T=32}` and `PvRow_v1{T=32, K=32}`, absent keys reading a zero V row
PV_K = 32

FROZEN = H200
#: width 32, 1 layer, 2 query heads of 16, MLP 64, context 8
TINY = IFACE_TINY
#: width 128, 1 layer, 8 query heads of 16, MLP 256, context 8: `FORMS` gives it every level of the map
MINI = Shape("e2e-mqa-w128-l1", 128, 1, 8, 16, 1, 256, 256, 8, 16)
#: MINI with its linears' registered rows a quantizer chunk (32 codes), as the frozen shape's are (8,192): each ``wrow{r}`` a
#: proof block reads is one registered row (`weights_commit.reads`)
MINI_ROWS = Shape("e2e-mqa-w128-l1-r32", 128, 1, 8, 16, 1, 256, 256, 8, 32)
#: Phase 1's widths with the frozen shape's head size: width 128, 2 layers, 2 heads of 64, MLP 512
SMALL = Shape("e2e-mqa-w128-l2", 128, 2, 2, 64, 1, 512, 256, 32, 128)
#: a GPU slice: width 2,048, 4 layers, 32 heads of 64, MLP 8,192
SLICE = Shape("e2e-mqa-w2048-l4", 2048, 4, 32, 64, 1, 8192, 256, 32, 256)
SHAPES = {s.name: s for s in (FROZEN, SLICE, SMALL, MINI, MINI_ROWS, TINY)} | {
    "h200": FROZEN, "slice": SLICE, "small": SMALL, "mini": MINI, "mini-rows": MINI_ROWS, "tiny": TINY}


@dataclass(frozen=True)
class Forms:
    """``chunk``: the words of a chunk row, the residual stream's and every row op's input and output rows (128, 256 B);
    ``sums_row``: RMSNorm's chunk sums a committed row (32, 128 B: `RmsSumSqChunks_v1{C = 32 chunk, K = chunk}`);
    ``group_chunks``: the chunk sums an RMSNorm group sums (64, so G = 2 at N = 16,384); ``quant_chunk``: a quantizer chunk's
    codes (`QuantChunkS8_v1{8192}`, and the s8 coordinates' R = K / 8192 rows); ``amax_part``: the widest `RowAbsMax_v1` (16,384;
    a wider row is parts and a combine over the last part and the partials); under the ``norm`` fusion, ``norm_part``: an
    amax part (`RmsNormAbsMax_v1{2048}`) and ``norm_codes``: a code block's words (`RmsNormQuantCodes_v1{512}`), the widest
    Flock's stager lowers in under 20 GB."""
    chunk: int = 128
    sums_row: int = 32
    group_chunks: int = 64
    quant_chunk: int = 8192
    amax_part: int = 16384
    norm_part: int = 2048
    norm_codes: int = 512

    def rows(self, n: int) -> int:
        return n // self.chunk

    def sums_c(self, n: int) -> int:
        """`RmsSumSqChunks`'s C at a row of n."""
        return min(n, self.sums_row * self.chunk)

    def groups(self, n: int) -> int:
        """`RmsNormOutSums`'s G at a row of n."""
        s = self.rows(n)
        return s // self.group_chunks if s >= self.group_chunks else 1

    def qchunk(self, n: int) -> int:
        return min(self.quant_chunk, n)

    def parts(self, n: int) -> tuple[int, int]:
        """(parts, part width) of `RowAbsMax` over a row of n: one part while n fits."""
        return (n // self.amax_part, self.amax_part) if n > self.amax_part else (1, n)

    def norm_parts(self, n: int) -> tuple[int, int]:
        """(parts, part width) of the fused amax over an RMSNorm row of n: one part while n fits."""
        return (n // self.norm_part, self.norm_part) if n > self.norm_part else (1, n)

    def ncodes(self, n: int) -> int:
        """A fused code block's words over an RMSNorm row of n."""
        return min(self.norm_codes, self.qchunk(n))


#: MINI: 4 chunk rows of 32 a row of the width, 2 sums rows, 2 RMSNorm groups, quantizer chunks of 32 (R = K / 32), amax
#: parts of 64 (2 a row of the width, 4 of the MLP's), heads gathered in pairs as at the frozen shape; fused, 2 amax parts of 64
#: and 2 code blocks of 16 a chunk
FORMS = {s.name: Forms(chunk=32, sums_row=2, group_chunks=2, quant_chunk=32, amax_part=64, norm_part=64, norm_codes=16)
         for s in (MINI, MINI_ROWS)}


def forms(s: Shape) -> Forms:
    f = FORMS.get(s.name) or Forms(chunk=min(128, s.width))
    for what, n in (("width", s.width), ("mlp", s.mlp), ("vocab", s.vocab)):
        if n % f.chunk or n % f.qchunk(n) or n % f.parts(n)[1] or f.rows(n) % f.groups(n):
            raise ValueError(f"{s.name}: the {what} {n} isn't whole chunk rows of {f.chunk}, quantizer chunks and amax parts")
    if s.width % f.norm_parts(s.width)[1] or f.qchunk(s.width) % f.ncodes(s.width):
        raise ValueError(f"{s.name}: the width {s.width} isn't whole fused amax parts and code blocks")
    if f.chunk % s.head_dim or s.heads % (f.chunk // s.head_dim) or f.rows(s.width) % (f.sums_c(s.width) // f.chunk):
        raise ValueError(f"{s.name}: chunk rows of {f.chunk} don't gather whole heads of {s.head_dim} or whole sums rows")
    return f


def check(s: Shape) -> Shape:
    if s.kv_heads != 1 or s.heads * s.head_dim != s.width or s.context > PV_K:
        raise ValueError(f"{s.name}: the skeleton runs MQA (one KV head), heads x head_dim = width, context <= {PV_K}")
    forms(s)
    return s
