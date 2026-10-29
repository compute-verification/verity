import Proofs.Pouw.Lead.Network.Rot
import Proofs.Pouw.Lead.Network.Defs
import Proofs.Pouw.Lead.Network.Cells
import Proofs.Pouw.Lead.Network.Term

/-!
# The lead's network term (`Pouw.SecurityProofs.Lead.Network`)

The network term at the egress caps, under `OutsourceWithinEgress`. Its proofs go in `Lead/Network/`, its declarations
in `Pouw.SecurityProofs.Lead.Network`, and each file is imported here.

* `Rot`: what one exchanged bit carries under `-rot`'s weights at full 64-row tiles (`n_max ≤ 256 / 3`), and the term
  `n_max W_mac E` over the stock MACs at the lead's MACs a token and caps of 32 and 320 bits.
* `Defs`: the definitions the network piece's statements read, each to move unchanged into `Definitions/Pouw/Lead/`:
  the layout with the coin lanes on every step (20 data lanes at step 0), Llama-3.2-1B's forward, and `networkGap`.
* `Cells`: every cell of the lead, step 0 included, is a `-rot` cell (`lead_cells_rot`).
* `Term`: the network term at the three points at caps 32 and 320, at Llama-3.2-1B's own calls, and summed over a
  forward (`llama_forward_network`).
-/
