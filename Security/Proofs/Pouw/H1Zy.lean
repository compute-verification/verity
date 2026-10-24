import Proofs.Pouw.H1Zy.Combine
import Proofs.Pouw.H1Zy.Lane
import Proofs.Pouw.H1Zy.Tile

/-!
# H1's drawn-leaf statements: their Definitions on words mean what `LeafChecks` reads

`Combine`: `PouwH1Combine_v1`'s `ok` is `true` exactly when the lanes' partials XOR, in lane order, to the receipt
(`Pouw.H1Zy.combineOk_iff`). `Proofs.Accounting.TS.Session.Lanes.Combine` reads it as `LeafChecks`' combine field.

`Lane`: at `32 ∣ K` and `K < 2^17` (the bind's range), `PouwH1LaneZy_v1`'s `ok = true` gives the low index bits only,
every lane cell's record z as the exact dot of its rows' codes and its y as the dequant of it, and every partial word as
the masked chain word (`Pouw.H1Zy.laneOk_spec`). `Proofs.Accounting.TS.Session.Lanes.Lane` reads it as `LeafChecks`'
lane field.

`Tile`: at the same `K`, `PouwH1TileZy_v1{TILE}`'s `ok = true` gives every cell's record z as the exact dot of its rows'
codes and its y as the dequant of it, over `PouwH1Tile_v1` or `PouwH1Tile_v3` at any noise (`Pouw.H1Zy.tileOk_spec`).
`Proofs.Accounting.TS.Session.Tile.Computes` reads it as `TileChecks`.
-/
