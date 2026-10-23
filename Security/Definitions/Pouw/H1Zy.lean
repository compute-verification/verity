import Definitions.Pouw.H1Zy.Combine
import Definitions.Pouw.H1Zy.Lane

/-!
# H1's drawn-leaf statements on words (`verity/core/protocols/pouw/circuit/h1zy.py`)

`Combine`: `PouwH1Combine_v1{TILE}`'s body on its primitives' word views, the receipt check over a leaf's lanes'
partials. Proved to be the XOR check in `Proofs.Pouw.H1Zy`.

`Lane`: `PouwH1LaneZy_v1{TILE}`'s body at a 32 × 32 leaf over `PouwH1Tile_v1`, on its primitives' word views, with the
dequant's and the chain's words as parameters. Proved in `Proofs.Pouw.H1Zy` to check the exact z, y and partials.
-/
