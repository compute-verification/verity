import Proofs.Pouw.Lead.Tags.Fold
import Proofs.Pouw.Lead.Tags.Shift
import Proofs.Pouw.Lead.Tags.SsnProofs
import Proofs.Pouw.Lead.Tags.TagProofs

/-!
# The lead's tags, coin lanes and epilogue fold

Its proofs are in `Lead/Tags/`, its new declarations in `Pouw.SecurityProofs.Lead.Tags`.

* `Fold`: the epilogue's IADD3 and FFMA give stock's word while `|A·Bᵀ| ≤ 2^22`, whatever the row and column sources.
* `Shift`: the lead's committed words (`Pouw.Lead.word`) are shift-separated on its coins: no two differ by a constant,
  and none is a function of one coin plus a constant, for at most 28,560 rows and columns.
* `SsnProofs`: served sparse noise's committed words on the full coin grid, for the 4- and 5-lane layouts. They are
  distinct, and no two differ by a constant (carrier separation).
* `TagProofs`: row tags against drawn weights. Two rows' tiles agree for at most a `q^−|J|` share of the draws, and
  the success set over a family fixed before the draw is bounded.
-/
