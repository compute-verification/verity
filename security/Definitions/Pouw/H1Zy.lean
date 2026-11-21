module

public import Definitions.Pouw.H1Zy.Combine
public import Definitions.Pouw.H1Zy.Lane
public import Definitions.Pouw.H1Zy.LaneBits
public import Definitions.Pouw.H1Zy.Stream
public import Definitions.Pouw.H1Zy.Tile
public import Definitions.Pouw.H1Zy.LaneChunk

/-!
# H1's drawn-leaf statements on words (`verity/protocols/accounting/pouw/prover/circuit/h1zy.py`)

`Tile`: `PouwH1Tile_v3`'s residue-sum and digit pieces on their word views.

`Combine`: `PouwH1Combine_v1{TILE}`'s body on its primitives' word views, the receipt check over a leaf's lanes'
partials.

`Lane`: `PouwH1LaneZy_v1{TILE}`'s body at a 32 × 32 leaf over `PouwH1Tile_v1`, on its primitives' word views, with the
dequant's and the chain's words as parameters.

`LaneBits`: the s8 multiply's steps on bits, at their lowering.

`Stream`: `PouwH1LaneZy_v1{TILE}`'s `ok`, `Lane`'s with each cell's z a parameter (`laneOkZ`).

`LaneChunk`: `PouwH1LaneZyChunk_v1{TILE, CHUNKS, CHUNK}`'s body, one of a lane's chunk statements along `K` past
`K = 4096`, on its primitives' word views: the carry (`h1zy.CARRY`) as a record, the chain as a step machine
(`ChainSteps`, a parameter, whose words over the whole `K` are `chainOf`), the dequant a parameter.
-/

@[expose] public section
