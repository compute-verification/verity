import Proofs.Pouw.Lead.Tags.Exact
import Proofs.Pouw.Lead.Tags.Distinct
import Proofs.Pouw.Lead.Tags.Fold
import Proofs.Pouw.Lead.Tags.Repeats
import Proofs.Pouw.Lead.Tags.Shift
import Proofs.Pouw.Lead.Tags.SsnProofs
import Proofs.Pouw.Lead.Tags.TagProofs
import Proofs.Pouw.Lead.Tags.Tile

/-!
# The lead's tags, coin lanes and epilogue fold

Its proofs are in `Lead/Tags/`, its new declarations in `Pouw.SecurityProofs.Lead.Tags`.

* `Distinct`: the lead's `repeats_le`, `repeats_le_of` at a call the verifier admits (`Layout.Admits`), and
  `word_exact`, `Exact`'s bound at `K ≤ 65,536`; `not_repeats_le`: without the cap the bound fails at `K = 0`.
* `Repeats`: `repeats_le_of`, `repeats_le`'s bound for an inner dimension of at least 1 and at most `2^40` tiles a
  call (`repeats_le_caps`: at most `2^15` rows, `2^20` columns and an inner dimension of `2^16`), by the call's pairs
  of tiles: each agrees on at most a `2^−56` share of the coins through a tile's own coin lane at its step, or `7^−8`
  for two tiles at one step, column block and coin tile, through their first rows' tags.
* `Exact`: with data in range, every committed word fits a 32-bit register for at most 4,587 steps; at the lead, for an
  inner dimension of at most 128,429.
* `Fold`: the epilogue's IADD3 and FFMA give stock's word while `|p| ≤ 2^22`, `p` the accumulator less its row and
  column sources (`(A·Bᵀ)_ij`, and served noisy `(A·Bᵀ)_ij` plus the noise), whatever those sources are.
* `Shift`: the lead's committed words (`Pouw.Lead.word`) are shift-separated on its coins: no two differ by a constant,
  and none is a function of one coin plus a constant, for at most 28,560 rows and any number of columns.
* `SsnProofs`: served sparse noise's committed words on the full coin grid, for the 4- and 5-lane layouts. They are
  distinct, and no two differ by a constant (carrier separation).
* `Tile`: two of the lead's MMA tiles of committed words (`Pouw.Lead.tileOf`) at different places, or one and a tile
  fixed before the coins, are equal on at most a `7^−8` share of the coins: by the row tag when the rows or columns
  differ, and by `-rot`'s coin lanes when only the step does.
* `TagProofs`: row tags against drawn weights. Two rows' tiles agree for at most a `q^−|J|` share of the draws, and
  the success set over a family fixed before the draw is bounded.
-/
