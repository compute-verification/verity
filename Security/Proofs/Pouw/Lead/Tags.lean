import Proofs.Pouw.Lead.Tags.Exact
import Proofs.Pouw.Lead.Tags.Fold
import Proofs.Pouw.Lead.Tags.Shift
import Proofs.Pouw.Lead.Tags.SsnProofs
import Proofs.Pouw.Lead.Tags.TagProofs
import Proofs.Pouw.Lead.Tags.Tile

/-!
# The lead's tags, coin lanes and epilogue fold

Its proofs are in `Lead/Tags/`, its new declarations in `Pouw.SecurityProofs.Lead.Tags`.

* `Exact`: with data in range, every committed word fits a 32-bit register for at most 4,588 steps; at the lead, for an
  inner dimension of at most 128,460.
* `Fold`: the epilogue's IADD3 and FFMA give stock's word while `|A·Bᵀ| ≤ 2^22`, whatever the row and column sources.
* `Shift`: the lead's committed words (`Pouw.Lead.word`) are shift-separated on its coins: no two differ by a constant,
  and none is a function of one coin plus a constant, for at most 28,560 rows and columns.
* `SsnProofs`: served sparse noise's committed words on the full coin grid, for the 4- and 5-lane layouts. They are
  distinct, and no two differ by a constant (carrier separation).
* `Tile`: two of the lead's MMA tiles of committed words (`Pouw.Lead.tileOf`) at different places, or one and a tile
  fixed before the coins, are equal on at most a `7^−8` share of the coins: by the row tag when the rows or columns
  differ, and by `-rot`'s coin lanes when only the step does.
* `TagProofs`: row tags against drawn weights. Two rows' tiles agree for at most a `q^−|J|` share of the draws, and
  the success set over a family fixed before the draw is bounded.
-/
