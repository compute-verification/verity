import Proofs.Pouw.H1Zy.Combine
import Proofs.Pouw.H1Zy.Lane
import Proofs.Pouw.H1Zy.Stream
import Proofs.Pouw.H1Zy.Vectors

/-!
# H1's drawn-leaf statements: their Definitions on words mean what `LeafChecks` reads

`Combine`: `PouwH1Combine_v1`'s `ok` is `true` exactly when the lanes' partials XOR, in lane order, to the receipt
(`Pouw.H1Zy.combineOk_iff`). `Proofs.Accounting.TS.Session.Lanes.Combine` reads it as `LeafChecks`' combine field.

`Lane`: at `32 ∣ K` and `K < 2^17` (the bind's range), `PouwH1LaneZy_v1`'s `ok = true` gives the low index bits only,
every lane cell's record z as the exact dot of its rows' codes and its y as the dequant of it, and every partial word as
the masked chain word (`Pouw.H1Zy.laneOk_spec`). `Proofs.Accounting.TS.Session.Lanes.Lane` reads it as `LeafChecks`'
lane field.

`Stream`: at the same `K`, `PouwH1Tile_v3`'s z of a cell is `GemmAccumulatorS8_v2`'s word for every dither, digit
coin, coin bit and noise (`Pouw.H1Zy.streamZ_eq_gemmAccS8`), so a lane's `ok` over v3 is its `ok` over v1
(`Pouw.H1Zy.laneOkZ_eq_laneOk`). `Proofs.Accounting.TS.Session.Lanes.Stream` reads it as `LeafChecks` at v3's lanes.

`Vectors`: the lane's chain words (`Pouw.H1Zy.lanePartial`) and the dequant (`Pouw.H1Zy.dequant`) against
`h1zy.lane_partial` and `h1zy.dequant`, evaluated by the kernel.
-/
