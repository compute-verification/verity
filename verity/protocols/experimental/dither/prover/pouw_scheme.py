"""Dither v3.2 (`verity/protocols/experimental/dither/PROTOCOL.md`) as a
`verity.protocols.resource_accounting.compute.pouw.verifier.protocol.PoUWScheme`, on its CPU reference (`dither`): the
tile-level entry points of the resource scheme ``dither`` (`verity.protocols.experimental.dither.scheme`).

A tile is one of the value root's 128 × 128 output tiles at full K (`dither.Shape.tile_of`, row-major). Its operands
are each row's K int8 codes (`dither.operand_row`), and its noise is the call's seeds (`dither.seeds`, or `fill_seeds`
for a filler). Its checked values are each output cell's T_1 .. T_S and O (`dither.cell_values`), which a proof block's
Definition holds the committed words to (`dither.block_inputs`, `check_block`, `check_last`). Its useful output is O,
(A + d_A)(B + d_B)ᵀ as a signed 32-bit integer. No credit assumes a bound over operands the prover chose, so every
well-formed tile is admissible.

The parameters are an (L, L′) pair of `PARAMETERS`, keyed by the card's short name (its card file's ``short``, under
``calibration/hardware/cards``) and whether registration pins B: the pairs the protocol settles for the RTX PRO 6000
("The segment and the noise period"), L = 1,024 with L′ = 128 when registration pins B, and L = 768 with L′ = 256 for
weights the prover chooses. No pair is settled for the H200. PoUW's kernel schemes reach it as the resource scheme
``dither`` on a card (`verity.protocols.resource_accounting.compute.pouw.prover.schemes.of`).

Three methods ask for what Dither has no code for, and raise `NotImplementedError` naming it:

* `noise`: a call's seeds come from the challenger's coin opened after the call's operand roots and the digest of the
  network attester's receipt of them (`dither.seeds`). A `Context` carries neither, only the execution phase's salt
  and the roots.
* `wref`: a call's cost is its honest cost at μ° on the card's tables (W_b, W_p), which no code computes. Dither has no
  W1 cost.
* `digest`: a tile's values are `Shape.lpt` rows of the call's value root, each a hiding leaf salted under the prover's
  private key (`dither.commit`, `leaf_h`), and a sampled proof block is proved, not replayed. There is no transcript
  leaf per tile.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import (
    Certificate, Context, DomainError, Matmul, Shape, Tile, grid_tiles)

from . import dither as D

NAME = "dither"
#: (card's short name, whether registration pins B) -> (L, L′)
PARAMETERS: dict[tuple[str, bool], tuple[int, int]] = {("rtx-pro-6000", True): (1024, 128),
                                                       ("rtx-pro-6000", False): (768, 256)}

Cells = list[list[tuple[list[int], int]]]


class Dither:
    """Dither v3.2 at one (L, L′) pair of `PARAMETERS`."""

    def __init__(self, card: str, pinned: bool) -> None:
        if (card, pinned) not in PARAMETERS:
            raise ValueError(f"{NAME}: no parameters on {card!r} with B {'pinned' if pinned else 'chosen'}; they are "
                             + ", ".join(f"{c} with B {'pinned' if p else 'chosen'}" for c, p in PARAMETERS))
        self.name, self.card, self.pinned = NAME, card, pinned
        self.L, self.lp = PARAMETERS[card, pinned]
        self.certificate = Certificate(
            conjecture="SAKMatmul",
            gap0=None,
            omega=Fraction(1),
            claims=("cr/sha-512", "live-verifier"),
            hypotheses=(
                "Pouw.Assumptions.SAKMatmul over the noised words, open",
                "Pouw.Assumptions.SAKSha on the call's commitments, open",
                ("both are stated for the no-fold round's 32-bit prefix sums, not for v3.2's 16-bit T_s and 32-bit O: "
                 "no guarantee states v3.2's proof block"),
            ),
            allowances=("segment S is checked through O and earns no credit: an output cell credits K − L MACs",),
            parameters=(("L", self.L), ("lp", self.lp)),
        )

    def shape(self, shape: Shape) -> D.Shape:
        """The reference's call of `shape` at these parameters; `ValueError` when it has none."""
        return D.Shape(shape.m, shape.n, shape.k, lp=self.lp, L=self.L)

    def check(self, matmul: Matmul) -> None:
        try:
            self.shape(matmul.shape)
        except ValueError as e:
            raise DomainError(f"{self.name} at L = {self.L}, L′ = {self.lp}: {e}") from None

    def tiles(self, shape: Shape) -> list[Tile]:
        return grid_tiles(shape, D.TILE, D.TILE)

    def wref(self, shape: Shape) -> int:
        raise NotImplementedError(f"{self.name}: a call's cost is its honest cost at μ° on the card's tables (W_b, W_p), "
                                  "which no code computes; Dither has no W1 cost")

    def noise(self, ctx: Context) -> Mapping[str, bytes]:
        raise NotImplementedError(f"{self.name}: a call's seeds come from the challenger's coin opened after its operand "
                                  "roots and the network attester's receipt of them (dither.seeds); a Context carries "
                                  "neither")

    def encode(self, ctx: Context, noise: Mapping[str, bytes], a_rows: Sequence[bytes], b_rows: Sequence[bytes],
               tile: Tile) -> D._Cells | None:
        sh = self.shape(ctx.matmul.shape)
        if any(len(r) != sh.k for r in (*a_rows, *b_rows)):
            return None
        a = {i: D.operand_row(r) for i, r in zip(tile.rows, a_rows, strict=True)}
        b = {j: D.operand_row(r) for j, r in zip(tile.cols, b_rows, strict=True)}
        return D._cells(sh, noise, a, b, tile.rows, tile.cols)

    def admissible(self, ctx: Context, ops: D._Cells) -> bool:
        return True

    def checked(self, ctx: Context, ops: D._Cells) -> Cells:
        """``checked[i][j]``: output cell (i, j)'s T_1 .. T_S and O."""
        sh = self.shape(ctx.matmul.shape)
        return [[(D._words(sh, x, w, sh.S), D._out(ad, bd)) for w, bd in zip(ops.W, ops.Bd)] for x, ad in zip(ops.X, ops.Ad)]

    def digest(self, ctx: Context, noise: Mapping[str, bytes], checked: Cells) -> bytes:
        raise NotImplementedError(f"{self.name}: a tile's values are Shape.lpt hiding rows of the call's value root "
                                  "(dither.commit), and a sampled proof block is proved, not replayed; there is no "
                                  "transcript leaf per tile")

    def useful(self, ctx: Context, noise: Mapping[str, bytes], ops: D._Cells, checked: Cells) -> list[list[int]]:
        return [[o - (1 << 32) if o >> 31 else o for _, o in row] for row in checked]


__all__ = ["NAME", "PARAMETERS", "Dither"]
