import Proofs.Pouw.Lead.CoinBranch.Cells
import Proofs.Pouw.Lead.CoinBranch.Kept
import Proofs.Pouw.Lead.CoinBranch.Registers
import Proofs.Pouw.Lead.CoinBranch.Revisit
import Proofs.Pouw.Lead.CoinBranch.Writes

/-!
# The lead's coin-branch bound (proofs)

A program that branches on the coins skips only the words it can predict equal on its path.

* `Cells`: `MH100w`'s floor applied cell by cell to a program chosen from the coins, and the served sparse-noise
  layout's one-bit merges.
* `Registers`: the chunk model against such a program, outcome by outcome: per-entry credit at word writes and under
  whole-register writes (model W), the attained side, and priced selection over uniform branches.
* `Kept`: what one run can skip under whole-tile writes (equal pairs and initial-memory hits), the exchange of sums
  over the coins, and two rows' fiber of a column's step-0 tag weights.
* `Revisit`: a tile returning to a value it held, bounded by the returning step's fresh coins alone; on `-rot` a
  16 × 8 tile does so on at most a `2^−163` share of them, whatever the data and the other coins.
* `Writes`: a run holds a value at some point exactly when it writes it or starts with it (`holds_iff`), and the
  copy argument for the lead (`distinct_le_writes`): its distinct committed tiles cost whole-tile writes, less the
  table hits.
* `GrindProofs` (imported by `Cells`): target words lost on a success set of the coins.
* `PointwiseProofs` (imported by `Registers`): the chunk model's bounds for every run.
-/
