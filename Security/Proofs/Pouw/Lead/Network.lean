import Proofs.Pouw.Lead.Network.Rot
import Proofs.Pouw.Lead.Network.Cells
import Proofs.Pouw.Lead.Network.Term

/-!
# The lead's network term (`Pouw.SecurityProofs.Lead.Network`)

* `Rot`: what one exchanged bit carries under `-rot`'s weights at full 64-row tiles (`n_max ≤ 256 / 3`), and the term
  `n_max W_mac E` over the stock MACs at the lead's MACs a token and caps of 32 and 320 bits.
* `Cells`: every cell of the lead, step 0 included, is a `-rot` cell of the words it is computed from, so it does at
  most `256 / 3` MACs a bit of its message (`lead_cells_macs_le`).
* `Term`: what `Rot`'s theorems prove of the statement's definitions: the lead's cells carry the three coin lanes at
  every step, a forward's network term from its bits (`networkFwd`), a forward's calls' bits (`forward_sum`) and
  Llama-3.2-1B's forward (`networkLlama`).

Its proofs go in `Lead/Network/`, its declarations in `Pouw.SecurityProofs.Lead.Network`, and each file is imported
here.
-/
