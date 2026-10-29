import Proofs.Pouw.Lead.Network.Rot
import Proofs.Pouw.Lead.Network.Term

/-!
# The lead's network term (`Pouw.SecurityProofs.Lead.Network`)

* `Rot`: what one exchanged bit carries under `-rot`'s weights at full 64-row tiles (`n_max ≤ 256 / 3`), and the term
  `n_max W_mac E` over the stock MACs at the lead's MACs a token and caps of 32 and 320 bits.
* `Term`: what `Rot`'s theorems prove of the statement's definitions: the lead's cells carry the three coin lanes at
  every step, and a call's network term at caps 32 and 320 (`networkGap_le`, `networkGap_le320`).

Its proofs go in `Lead/Network/`, its declarations in `Pouw.SecurityProofs.Lead.Network`, and each file is imported
here.
-/
