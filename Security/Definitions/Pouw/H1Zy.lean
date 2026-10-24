import Definitions.Pouw.H1Zy.Chain
import Definitions.Pouw.H1Zy.Combine
import Definitions.Pouw.H1Zy.Dequant
import Definitions.Pouw.H1Zy.Lane
import Definitions.Pouw.H1Zy.Stream

/-!
# H1's drawn-leaf statements on words (`verity/core/protocols/pouw/circuit/h1zy.py`)

`Combine`: `PouwH1Combine_v1{TILE}`'s body on its primitives' word views, the receipt check over a leaf's lanes'
partials. Proved to be the XOR check in `Proofs.Pouw.H1Zy`.

`Lane`: `PouwH1LaneZy_v1{TILE}`'s body at a 32 × 32 leaf over `PouwH1Tile_v1`, on its primitives' word views, with the
dequant's and the chain's words as parameters. Proved in `Proofs.Pouw.H1Zy` to check the exact z, y and partials.

`Dequant`: `PouwZyDequant_v1`'s word view, `h1zy.dequant` (`dequantW`), on `Verity.Fp32`'s IEEE f32 arithmetic.

`Chain`: a lane's chain words (`chainWords`), `h1zy.lane_partial`'s fold from the seeds, the noise and the fold keys,
with the spec's SHA-512. `Proofs.Pouw.H1Zy.Vectors` holds both to the Python.

`Stream`: `PouwH1Tile_v3`'s z of a cell (`streamZ`, `h1tile.stream_cell`: its lane's O' and D words recombined, its
wrap correction and its dither peel) and of a lane's cell from the seed's noise (`laneZ3`), and the lane's `ok` at any
tile's z (`laneOkZ`). Proved in `Proofs.Pouw.H1Zy` to be EXACT's z at every noise.
-/
