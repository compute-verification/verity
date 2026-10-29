"""Circuits' FAMILIES map at a shape: the step's tables, one op's outputs each, and for each table the units whose output rows are its bytes.

A table is one op's output at one step, one row a sequence slot (`Table`; a quantizer's also has the f32 scale as the row's
word). Its `Units` are a family of one template: unit i's output is one committed row, ``nbytes`` at ``offset + i * stride``
of the table's row (with ``scale``, a second row: the word). The commit layout (`commit.layout`) commits every unit's rows as
one commit unit, so a drawn unit opens exactly its outputs; a gather's rows are the bytes its parts' units wrote, committed
again as the rows its readers read. With `shape.Forms`' chunk of C words (128 at the frozen shape):

* the residual stream, every RMSNorm's output and the rows every row op reads are chunk rows of C words (256 B);
* RMSNorm: ``RmsSumSqChunks_v1{C = 32 C, K = C}`` writes 32 chunk sums a row (128 B), then ``RmsNormOutSums_v1{C, S, G, N,
  EPS}`` writes each output chunk row, reading its x and w chunks and the sums rows;
* the quantizer: ``RowAbsMax_v1{N, RELU}`` (a row past 16,384: parts, then ``RowAbsMax_v1{16384 + parts - 1}`` over the last
  part and the partials), then ``QuantChunkS8_v1{8192, RELU}`` a chunk, each writing its codes row and the row's scale;
* the W8A8 linears: one ``ScaledMmS8CoordinateRows_v1{K, R = K / 8192, DOT}`` a coordinate, reading the quantizer's R code rows,
  its 16-bit output one row; ``AllGather_v1{WORLD, N = 1}`` gathers them into the rows their readers read: a head's q, the K
  row and the V row (WORLD = head_dim, 128 B), and the O, up, down and head outputs (WORLD = C, chunk rows);
* attention padded to T = 32 keys: ``QkSoftmaxRowPad_v1{T, D}`` a head (q's row and the slot's key rows, n = position + 1 a
  position-kind input; p_j = +0 past n) and ``PvRow_v1{T, K = 32, D, DOT}`` a head (absent keys read a registered all-zero
  V row), then ``AllGather_v1{WORLD = C / D, N = D}`` pairs the heads' rows into chunk rows for O's quantizer;
* the residual add ``Bf16AddRow_v2{C}`` a chunk row; the embedding one ``EmbeddingCoordinate_v1{V}`` a coordinate, gathered
  by ``AllGather_v1{C, 1}``; the token ``Argmax_v2{V}`` over the logits' chunk rows.

A key's DOT is the step `replay` binds: the s8 coordinates' ``GEMM_SM90_S8`` and PV's ``GEMM_HOPPER``. At the frozen shape a
slot commits 6,102,102 units in 6,102,832 rows, 5,970,688 of them s8 coordinates' 16-bit rows.

``fused`` names the map's fusions (M1's map has none), each a proof block whose intermediate is never committed:

* ``residual``: each residual add in its projection's W8A8 epilogue. ``wo``'s and ``wdown``'s coordinates are
  ``ScaledMmS8ResidualCoordinateRows_v1{K, R, DOT}``, coordinate i adding the projection's word to the residual stream's
  word i, and write the residual's table (``attn_residual``, ``mlp_residual``); the projections' own tables and the
  ``Bf16AddRow_v2`` families go.
* ``norm``: each RMSNorm into its quantizer (``attn``, ``mlp`` into ``up``, ``final`` into ``head``). The sums stay; the
  quantizer's amax is ``RmsNormAbsMax_v1{C, S, G, N, EPS}`` a part of `shape.Forms`' ``norm_part`` words (2,048) and
  ``RowAbsMax_v1{parts, false}`` over the partials, and its codes ``RmsNormQuantCodes_v1{C, S, G, N, EPS}`` a block of
  ``norm_codes`` words (512), the amax parts and code blocks reading the stream, the gain and the sums; each chunk row's
  scale is ``QuantScaleS8_v1`` of the row's amax. The rows are a chunk's codes and its scale as before, so the coordinates
  read what they read unfused. The norm outputs' tables (``attn_norm``, ``mlp_norm``, ``final_norm``) and those quantizers'
  ``RmsNormOutSums_v1`` and ``QuantChunkS8_v1`` families go.
* ``pv``: each head's softmax into its PV row. ``pv``'s blocks are ``QkSoftmaxPvPad_v1{T, K = 32, D, DOT}`` a head, reading
  q's row, the slot's key and value rows and n, and the ``softmax`` table and its ``QkSoftmaxRowPad_v1`` family go. Its key
  names the Boolean Definition and its step, ``GEMM_HOPPER``'s, since no word Definition is its view.
"""
from __future__ import annotations

from dataclasses import dataclass

from verity.ml.circuits.ops.boolean import gemm as BG

from ._replay import RP
from .shape import PV_K, Shape, forms

DEFINITION, PENDING = "definition", "pending-definition"
#: the map's Definitions on main (some in `verity_vllm`'s registry); the rest are pending
ON_MAIN = frozenset({"ScaledMmS8CoordinateRows_v1", "ScaledMmS8ResidualCoordinateRows_v1", "AllGather_v1",
                     "EmbeddingCoordinate_v1", "RmsNormAbsMax_v1", "RmsNormQuantCodes_v1", "QuantScaleS8_v1",
                     "QkSoftmaxPvPad_v1"})
EPS_KEY = 1e-05
FUSIONS = frozenset({"norm", "pv", "residual"})


@dataclass(frozen=True)
class Table:
    op: str
    nbytes: int      # a slot's bytes of the op's output
    quant: bool = False     # the row's word (the f32 scale) is committed after the codes


@dataclass(frozen=True)
class Units:
    name: str
    key: str
    table: str
    count: int
    nbytes: int
    stride: int
    offset: int = 0
    scale: bool = False

    @property
    def template(self) -> str:
        return self.key.split("{")[0]

    @property
    def status(self) -> str:
        return DEFINITION if self.template in ON_MAIN else PENDING

    @property
    def bits(self) -> int:
        """A unit's output bits."""
        return 8 * self.nbytes + 32 * self.scale

    @property
    def scale_only(self) -> bool:
        """Whether its output is its quantizer row's scale alone (`QuantScaleS8_v1`): unit i writes the scale of the chunk row
        of codes ``i * stride ..``."""
        return self.template == "QuantScaleS8_v1"


def key(name: str, **statics: object) -> str:
    return RP.key_of(name, statics)


Entry = tuple[Table, tuple[Units, ...]]


def _units(name: str, k: str, table: str, count: int, nbytes: int, offset: int = 0, scale: bool = False) -> Units:
    return Units(name, k, table, count, nbytes, nbytes, offset, scale)


class Map:
    """The map at ``shape`` with the fusions ``fused`` (`FUSIONS`): `groups` in the order a step writes them (the embedding,
    each layer, the head)."""

    def __init__(self, shape: Shape, fused: frozenset[str] = frozenset()) -> None:
        if not fused <= FUSIONS:
            raise ValueError(f"the map's fusions are {sorted(FUSIONS)}, not {sorted(fused - FUSIONS)}")
        self.shape, self.f, self.fused = shape, forms(shape), frozenset(fused)

    # -- the ops

    def norm(self, op: str) -> list[Entry]:
        """RMSNorm ``op`` (``l0.attn``, ``final``): its sums table ``op_sums`` and its output ``op_norm``."""
        f, n = self.f, self.shape.width
        c, sc, s = f.chunk, f.sums_c(n), f.rows(n)
        sums = _units(op + "_sums", key("RmsSumSqChunks_v1", C=sc, K=c), op + "_sums", n // sc, 4 * (sc // c))
        out = _units(op + "_norm", key("RmsNormOutSums_v1", C=c, **self._norm_statics(n)), op + "_norm", s, 2 * c)
        return [(Table(op + "_sums", 4 * s), (sums,)), (Table(op + "_norm", 2 * n), (out,))]

    def _norm_statics(self, n: int) -> dict[str, object]:
        return {"S": self.f.rows(n), "G": self.f.groups(n), "N": n, "EPS": EPS_KEY}

    def normed(self, norm: str, op: str) -> list[Entry]:
        """RMSNorm ``norm`` into the quantizer ``op`` (no ReLU). Fused under ``norm``: no ``norm_norm`` table; the amax an
        ``RmsNormAbsMax_v1`` a part and ``RowAbsMax_v1`` over the partials (``op_amax``: the partials, then the row's), the
        codes an ``RmsNormQuantCodes_v1`` a block, each part and block reading the stream, the gain and ``norm_sums``, and
        each chunk row's scale a ``QuantScaleS8_v1``."""
        n = self.shape.width
        if "norm" not in self.fused:
            return self.norm(norm) + self.quant(op, n, False)
        f, st = self.f, self._norm_statics(n)
        (parts, pw), q, cw = f.norm_parts(n), f.qchunk(n), f.ncodes(n)
        if parts > 1:
            amax = (_units(op + "_amax.part", key("RmsNormAbsMax_v1", C=pw, **st), op + "_amax", parts, 2),
                    _units(op + "_amax", key("RowAbsMax_v1", N=parts, RELU=False), op + "_amax", 1, 2, offset=2 * parts))
        else:
            amax = (_units(op + "_amax", key("RmsNormAbsMax_v1", C=n, **st), op + "_amax", 1, 2),)
        codes = _units(op + "_quant", key("RmsNormQuantCodes_v1", C=cw, **st), op + "_quant", n // cw, cw)
        scale = Units(op + "_quant.scale", key("QuantScaleS8_v1"), op + "_quant", n // q, 4, q)
        return self.norm(norm)[:1] + [(Table(op + "_amax", 2 * (parts + (parts > 1))), amax),
                                      (Table(op + "_quant", n, True), (codes, scale))]

    def quant(self, op: str, n: int, relu: bool) -> list[Entry]:
        """The quantizer ``op`` over a row of n: its amax table ``op_amax`` (the partials, then the row's) and ``op_quant``."""
        f = self.f
        parts, w = f.parts(n)
        amax = ([_units(op + "_amax.part", key("RowAbsMax_v1", N=w, RELU=relu), op + "_amax", parts - 1, 2)] if parts > 1 else [])
        amax.append(_units(op + "_amax", key("RowAbsMax_v1", N=w + parts - 1 if parts > 1 else n, RELU=relu), op + "_amax", 1, 2,
                           offset=2 * (parts - 1)))
        q = f.qchunk(n)
        codes = _units(op + "_quant", key("QuantChunkS8_v1", N=q, RELU=relu), op + "_quant", n // q, q, scale=True)
        return [(Table(op + "_amax", 2 * parts), tuple(amax)), (Table(op + "_quant", n, True), (codes,))]

    def linear(self, op: str, n: int, k: int, world: int, residual: str | None = None) -> list[Entry]:
        """The W8A8 linear ``op`` (n coordinates over K = k), gathered ``world`` coordinates a row. With ``residual``, the
        residual add ``residual`` in its epilogue: its coordinates write that table, and ``op`` has none."""
        table, name = (residual, "ScaledMmS8ResidualCoordinateRows_v1") if residual else (op, "ScaledMmS8CoordinateRows_v1")
        coord = _units(op, key(name, K=k, R=k // self.f.qchunk(k), DOT=RP.G.GEMM_SM90_S8["DOT"]), table, n, 2)
        gather = _units(op + ".gather", key("AllGather_v1", WORLD=world, N=1), table, n // world, 2 * world)
        return [(Table(table, 2 * n), (coord, gather))]

    def residual(self, op: str) -> list[Entry]:
        c, n = self.f.chunk, self.shape.width
        return [(Table(op, 2 * n), (_units(op, key("Bf16AddRow_v2", N=c), op, n // c, 2 * c),))]

    def projection(self, op: str, k: int, residual: str) -> list[Entry]:
        """The linear ``op`` back to the width (K = k), then the residual add ``residual``: fused under ``residual``."""
        n, c = self.shape.width, self.f.chunk
        if "residual" in self.fused:
            return self.linear(op, n, k, c, residual)
        return self.linear(op, n, k, c) + self.residual(residual)

    # -- the groups

    def embed(self) -> list[Entry]:
        sh, c = self.shape, self.f.chunk
        return [(Table("embed", 2 * sh.width), (_units("embed", key("EmbeddingCoordinate_v1", V=sh.vocab), "embed", sh.width, 2),
                                               _units("embed.gather", key("AllGather_v1", WORLD=c, N=1), "embed", sh.width // c,
                                                      2 * c)))]

    def attention(self, p: str) -> list[Entry]:
        """Layer ``p``'s attention, a head a block: ``QkSoftmaxRowPad_v1`` writes ``softmax`` and ``PvRow_v1`` reads it and
        writes ``pv``. Fused under ``pv``: ``QkSoftmaxPvPad_v1`` writes ``pv`` and there is no ``softmax``."""
        sh, c = self.shape, self.f.chunk
        d, h = sh.head_dim, sh.heads
        gather = _units(p + "pv.gather", key("AllGather_v1", WORLD=c // d, N=d), p + "pv", h * d // c, 2 * c)
        if "pv" in self.fused:
            head = _units(p + "pv", key("QkSoftmaxPvPad_v1", T=PV_K, K=PV_K, D=d, DOT=BG.GEMM_HOPPER["DOT"]), p + "pv", h, 2 * d)
            return [(Table(p + "pv", 2 * h * d), (head, gather))]
        soft = _units(p + "softmax", key("QkSoftmaxRowPad_v1", T=PV_K, D=d), p + "softmax", h, 2 * PV_K)
        pv = _units(p + "pv", key("PvRow_v1", T=PV_K, K=PV_K, D=d, DOT=RP.G.GEMM_HOPPER["DOT"]), p + "pv", h, 2 * d)
        return [(Table(p + "softmax", 2 * h * PV_K), (soft,)), (Table(p + "pv", 2 * h * d), (pv, gather))]

    def layer(self, i: int) -> list[Entry]:
        sh, c, p = self.shape, self.f.chunk, f"l{i}."
        w, d, h = sh.width, sh.head_dim, sh.heads
        out = self.normed(p + "attn", p + "attn")
        out += self.linear(p + "wq", h * d, w, d) + self.linear(p + "wk", d, w, d) + self.linear(p + "wv", d, w, d)
        out += self.attention(p)
        out += self.quant(p + "o", w, False) + self.projection(p + "wo", h * d, p + "attn_residual")
        out += self.normed(p + "mlp", p + "up") + self.linear(p + "wup", sh.mlp, w, c)
        out += self.quant(p + "down", sh.mlp, True) + self.projection(p + "wdown", sh.mlp, p + "mlp_residual")
        return out

    def head(self) -> list[Entry]:
        sh = self.shape
        out = self.normed("final", "head") + self.linear("head", sh.vocab, sh.width, self.f.chunk)
        return out + [(Table("argmax", 2), (_units("argmax", key("Argmax_v2", V=sh.vocab), "argmax", 1, 2),))]

    def groups(self) -> list[list[Entry]]:
        return [self.embed()] + [self.layer(i) for i in range(self.shape.layers)] + [self.head()]

    def units(self) -> dict[str, Units]:
        return {u.name: u for g in self.groups() for _, us in g for u in us}

    def counts(self, group: list[Entry]) -> dict[str, int]:
        """Units a slot of each template key in ``group``."""
        out: dict[str, int] = {}
        for _, us in group:
            for u in us:
                out[u.key] = out.get(u.key, 0) + u.count
        return out

