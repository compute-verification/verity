"""The native reference `fp8.linear` must evaluate to: #295's scheme itself.

root_A is `audit.commit_rows` over A's rows (the protocol's own commitment); the forming, the checked words and the useful
output are `schemes.fp8_is.FP8IS` over `D3s(sums=...)`, run tile by tile through its `encode`, `checked` and `useful` on
16 × 16 tiles (a test instance: the scheme's production tiles are 64 × 64, and a cell's words do not depend on the tiling).
Only the dequantization after Z is the circuit's own: y = bf16(s·t·Z) + bias, by the core primitives the circuit calls.
"""
from __future__ import annotations

from dataclasses import dataclass

from verity.ir.defs import REGISTRY

from ..audit import activation_binding, commit_rows
from ..protocol import Context, Matmul
from ..protocol import Shape as PShape
from ..protocol import grid_tiles
from ..schemes import fp8_is as S8
from .fp8 import D, Lengths, Shape, upper_nodes

M32 = (1 << 32) - 1


@dataclass(frozen=True)
class Call:
    """One linear call's inputs: A's rows and B's rows as E4M3 codes, the dequantization's scales and bias as FP32 and BF16
    words, and the declaration (salt, index, weight id)."""

    shape: Shape
    a: tuple[tuple[int, ...], ...]
    s: tuple[int, ...]
    b: tuple[tuple[int, ...], ...]
    t: tuple[int, ...]
    bias: tuple[int, ...]
    salt: bytes
    index: int
    weight: int
    sums: str = "f16"

    @property
    def lengths(self) -> Lengths:
        return Lengths.of(self.index, self.weight)

    def args(self) -> list:
        """The flat input leaves of `fp8.linear`, in parameter order."""
        m, n = self.shape.m, self.shape.n
        out = [[x for r in self.a for x in r], list(self.s), [x for r in self.b for x in r], list(self.t), list(self.bias),
               list(self.salt), [self.index >> 32, self.index & M32], [self.weight >> 32, self.weight & M32],
               list(range(m)), list(range(m)), list(range(n))]
        if self.shape.width > m:
            out.append(list(range(m, self.shape.width)))
        if upper_nodes(self.shape):
            out.append([v for p in upper_nodes(self.shape) for v in p])
        return out


@dataclass(frozen=True)
class Result:
    root: bytes
    x: list[list[int]]                   # X̃'s rows, 3k codes each
    y: list[list[int]]                   # Ỹ's columns
    words: list[list[int]]               # per tile (row-major), per cell (row-major): I_1 … I_T, R_2 … R_(T−1)
    z: list[list[int]]                   # R_T per output, as FP32 words
    out: list[list[int]]                 # y, BF16


def scheme(sums: str = "f16") -> S8.FP8IS:
    """#295's scheme on its target forming, as a test instance on 16 × 16 tiles."""
    return S8.FP8IS(S8.D3s(sums=sums), mnk_min=D, tile=(D, D))


def root_a(call: Call) -> bytes:
    return commit_rows(activation_binding(call.salt, call.index), call.index, [bytes(r) for r in call.a]).commitment.root


def dequant(z: int, s: int, t: int, bias: int) -> int:
    mul, cast, add = REGISTRY["F32Mul_v2"].evaluate, REGISTRY["F32ToBf16Rn_v1"].evaluate, REGISTRY["Bf16Add_v1"].evaluate
    return add(cast(mul(mul(s, t), z)), bias)


def evaluate(call: Call) -> Result:
    sh = call.shape
    sc = scheme(call.sums)
    u = Matmul(call.index, PShape(sh.m, sh.k, sh.n), call.weight)
    root = root_a(call)
    ctx = Context(call.salt, u, root, bytes(32))                 # root_B: the D-3 forming does not read it
    noise = sc.noise(ctx)
    x: list[list[int]] = [[] for _ in range(sh.m)]
    y: list[list[int]] = [[] for _ in range(sh.n)]
    z = [[0] * sh.n for _ in range(sh.m)]
    words = []
    for tile in grid_tiles(u.shape, D, D):
        ops = sc.encode(ctx, noise, [bytes(call.a[i]) for i in tile.rows], [bytes(call.b[j]) for j in tile.cols], tile)
        chk = sc.checked(ctx, ops)
        useful = sc.useful(ctx, noise, ops, chk)
        tw = []
        for bi, i in enumerate(tile.rows):
            x[i] = ops.x[bi]
            for bj, j in enumerate(tile.cols):
                y[j] = ops.y[bj]
                atoms, running = chk.cells[bi][bj]
                tw += atoms + running[1:-1]
                z[i][j] = running[-1]
                assert S8.f32_float(z[i][j]) == useful[bi][bj] or useful[bi][bj] != useful[bi][bj]
        words.append(tw)
    out = [[dequant(z[i][j], call.s[i], call.t[j], call.bias[j]) for j in range(sh.n)] for i in range(sh.m)]
    return Result(root, x, y, words, z, out)


__all__ = ["Call", "Result", "dequant", "evaluate", "root_a", "scheme"]
