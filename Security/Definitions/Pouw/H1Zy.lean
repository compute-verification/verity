import Definitions.Pouw.H1Zy.Chain
import Definitions.Pouw.H1Zy.Combine
import Definitions.Pouw.H1Zy.CombineBits
import Definitions.Pouw.H1Zy.Dequant
import Definitions.Pouw.H1Zy.Lane
import Definitions.Pouw.H1Zy.LaneBits
import Definitions.Pouw.H1Zy.Stream
import Definitions.Pouw.H1Zy.Tile
import Definitions.Pouw.H1Zy.LaneChunk

/-!
# H1's drawn-leaf statements on words (`verity/core/protocols/pouw/circuit/h1zy.py`)

`Tile`, a tile unit's one statement: `PouwH1TileZy_v1{TILE}`'s `ok` over `PouwH1Tile_v1` (EXACT's z) or
`PouwH1Tile_v3` (the one-stream's z at its noise), on its primitives' word views, with the dequant's words as a
parameter. Proved in `Proofs.Pouw.H1Zy` to check the exact z and y.

`Combine`: `PouwH1Combine_v1{TILE}`'s body on its primitives' word views, the receipt check over a leaf's lanes'
partials. Proved to be the XOR check in `Proofs.Pouw.H1Zy`.

`Chain`: a lane's chain words (`h1tile._chain`), the fx1 chain over its rows' and columns' noise from seed_j, keyed from
σ_R, as `LaneChunk`'s step machine (`h1Chain K`): the seeds and noise stream (SHA-512), the layout, and duo, on `Tile`'s
segment and digit pieces. `Accounting.Guarantees.TS.LeafLanesSound` reads its words as the lanes' (`laneChain`);
`Proofs.Pouw.H1Zy.ChainVectors` holds it to the Python.

`CombineBits`: the same body on bits, its pieces (`PouwXor32_v2`, `PouwEq32_v2`, `PouwAndBit_v2`) at their lowerings.
Proved in `Proofs.Pouw.H1Zy` to compute `Combine`'s.

`Dequant`: `PouwZyDequant_v1`, the served dequant RN_bf16(RN32(RN32(RN32(z) · s_a) · s_w)), on its primitives' word
views by integer arithmetic on the encodings (`zyDequant`). `Accounting.Guarantees.TS.LeafLanesSound` reads it as the
lane's dequant; `Proofs.Pouw.H1Zy.DequantVectors` holds it to the Python evaluators.

`Lane`: `PouwH1LaneZy_v1{TILE}`'s body at a 32 × 32 leaf over `PouwH1Tile_v1`, on its primitives' word views, with the
dequant's and the chain's words as parameters. Proved in `Proofs.Pouw.H1Zy` to check the exact z, y and partials.

`LaneBits`: the same body on bits, its pieces (the s8 multiply's steps, `PouwZyCell_v2`, `PouwMuxBit_v2`, `PouwMask32_v2`,
`PouwEq32_v2`) at their lowerings, the dequant and the chain's words at their word views. Proved in `Proofs.Pouw.H1Zy`
to compute `Lane`'s.

`Stream`: `PouwH1LaneZy_v1{PouwH1Tile_v3}`'s `ok`, `Lane`'s with each cell's z `Tile`'s `streamZ` at the lane's noise
(`laneOkZ` at `laneZ3`). Proved in `Proofs.Pouw.H1Zy` to be `Lane`'s `ok`.

`LaneChunk`: `PouwH1LaneZyChunk_v1{TILE, CHUNKS, CHUNK}`'s body, one of a lane's chunk statements along `K` past
`K = 4096`, on its primitives' word views: the carry (`h1zy.CARRY`) as a record, the chain as a step machine
(`ChainSteps`, a parameter, whose words over the whole `K` are `chainOf`), the dequant a parameter. Proved in
`Proofs.Pouw.H1Zy` to check the whole lane at accepted chunks linked by their carries.
-/
