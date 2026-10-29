import Proofs.Pouw.Lead.Network.Rot

/-!
# The lead's network term

The network term at the egress caps, under `OutsourceWithinEgress`.

Its proofs go in `Lead/Network/`, its declarations in `Pouw.SecurityProofs.Lead.Network`, and each file is imported
here.

* `Rot`: what one exchanged bit carries under `-rot`'s weights at full 64-row tiles (`n_max ≤ 256 / 3`), and the term
  `n_max W_mac E` over the stock MACs at the lead's MACs a token and caps of 32 and 320 bits.
-/
