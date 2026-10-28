import Proofs.Pouw.H1Zy.ChainVectors
import Proofs.Pouw.H1Zy.Combine
import Proofs.Pouw.H1Zy.CombineBits
import Proofs.Pouw.H1Zy.DequantVectors
import Proofs.Pouw.H1Zy.Lane
import Proofs.Pouw.H1Zy.LaneBits
import Proofs.Pouw.H1Zy.Tile

/-!
# H1's drawn-leaf statements: their Definitions on words mean what `LeafChecks` reads

`Tile`, a tile unit's one statement: at `32 ∣ K` and `K < 2^17`, `PouwH1TileZy_v1{TILE}`'s `ok = true` gives every
cell's record z as the exact dot of its rows' codes and its y as the dequant of it, over `PouwH1Tile_v1` or
`PouwH1Tile_v3` at any noise (`Pouw.H1Zy.tileOk_spec`). `Proofs.Accounting.TS.Session.Tile.Computes` reads it as
`TileChecks`.

`Combine`: `PouwH1Combine_v1`'s `ok` is `true` exactly when the lanes' partials XOR, in lane order, to the receipt
(`Pouw.H1Zy.combineOk_iff`). `Proofs.Accounting.TS.Session.Lanes.Combine` reads it as `LeafChecks`' combine field.

`ChainVectors`: `Pouw.H1Zy.h1Chain`'s pieces, seeds and noise stream, and its chain on a lane's noise, on values the
Python computes, checked by the kernel.

`CombineBits`: the combine's body on bits computes it on words (`Pouw.H1Zy.combineBits_bitsOf`), each piece's lowering
its word view.

`DequantVectors`: `Pouw.H1Zy.zyDequant` and its three primitives on words the Python evaluators compute, checked by
the kernel.

`Lane`: at `32 ∣ K` and `K < 2^17` (the bind's range), `PouwH1LaneZy_v1`'s `ok = true` gives the low index bits only,
every lane cell's record z as the exact dot of its rows' codes and its y as the dequant of it, and every partial word as
the masked chain word (`Pouw.H1Zy.laneOk_spec`). `Proofs.Accounting.TS.Session.Lanes.Lane` reads it as `LeafChecks`'
lane field.

`LaneBits`: the lane's body on bits computes it on words (`Pouw.H1Zy.laneBits_bv`), each restated piece's lowering its
word view (the multiply's, `Pouw.H1Zy.immaS8Bits_bv`, from Baugh–Wooley and the clamp).

`Stream`: at the same `K`, a lane's `ok` over `PouwH1Tile_v3` is its `ok` over `PouwH1Tile_v1` at every seed
(`Pouw.H1Zy.laneOkZ_eq_laneOk`, `Pouw.H1Zy.laneZ3_eq`).

`LaneChunk`: a lane's `n` chunk statements, each `ok = true`, chunk `c + 1`'s carry in chunk `c`'s carry out, give the
whole lane's `ok = true` at the chain machine's words on the rows they read joined (`Pouw.H1Zy.laneOk_of_chunks`), and so
`laneOk_spec`'s exact z, dequant and partials over the whole `K` (`Pouw.H1Zy.chunks_spec`). No guarantee reads it yet.
-/
