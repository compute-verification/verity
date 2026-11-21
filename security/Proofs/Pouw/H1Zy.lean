module

public import Proofs.Pouw.H1Zy.Lane
public import Definitions.Pouw.H1Zy.LaneBits
public import Definitions.Pouw.H1Zy.Stream
public import Proofs.Pouw.H1Zy.LaneChunk
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Mathlib.Tactic.NormNum.Pow

/-!
# H1's drawn-leaf statements: their Definitions on words mean what `LeafChecks` reads

`Combine`: `PouwH1Combine_v1`'s `ok` is `true` exactly when the lanes' partials XOR, in lane order, to the receipt
(`Pouw.H1Zy.combineOk_iff`).

`ChainVectors`: `Pouw.H1Zy.h1Chain`'s pieces, seeds and noise stream on values the Python computes, checked by the
kernel.

`Lane`: at `32 ∣ K` and `K < 2^17` (the bind's range), `PouwH1LaneZy_v1`'s `ok = true` gives the low index bits only,
every lane cell's record z as the exact dot of its rows' codes and its y as the dequant of it, and every partial word as
the masked chain word (`Pouw.H1Zy.laneOk_spec`).

`Stream` (`Definitions.Pouw.H1Zy.Stream`): a lane's `ok` over `PouwH1Tile_v3` (`Pouw.H1Zy.laneOkZ` at `laneZ3`). No
theorem relates it to its `ok` over `PouwH1Tile_v1`.

`LaneChunk`: a lane's `n` chunk statements, each `ok = true`, chunk `c + 1`'s carry in chunk `c`'s carry out, give the
whole lane's `ok = true` at the chain machine's words on the rows they read joined (`Pouw.H1Zy.laneOk_of_chunks`), and so
`laneOk_spec`'s exact z, dequant and partials over the whole `K` (`Pouw.H1Zy.chunks_spec`). No guarantee reads it yet.
-/

@[expose] public section
