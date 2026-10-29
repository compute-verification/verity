import Proofs.Pouw.Lead.Tags.SsnProofs
import Proofs.Pouw.Lead.Tags.TagProofs

/-!
# The lead's tags: distinctness and shift separation

Its proofs are in `Lead/Tags/`, its new declarations in `Pouw.SecurityProofs.Lead.Tags`.

* `SsnProofs`: served sparse noise's committed words on the full coin grid, for the 4- and 5-lane layouts. They are
  distinct, and no two differ by a constant (carrier separation).
* `TagProofs`: row tags against drawn weights. Two rows' tiles agree for at most a `q^−|J|` share of the draws, and
  the success set over a family fixed before the draw is bounded.
-/
