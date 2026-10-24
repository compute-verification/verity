import Definitions.Pouw.H1Zy.Combine
import Definitions.Pouw.H1Zy.CombineBits
import Definitions.Pouw.H1Zy.Dequant
import Definitions.Pouw.H1Zy.Lane
import Definitions.Pouw.H1Zy.LaneBits

/-!
# H1's drawn-leaf statements on words (`verity/core/protocols/pouw/circuit/h1zy.py`)

`Combine`: `PouwH1Combine_v1{TILE}`'s body on its primitives' word views, the receipt check over a leaf's lanes'
partials. Proved to be the XOR check in `Proofs.Pouw.H1Zy`.

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
-/
