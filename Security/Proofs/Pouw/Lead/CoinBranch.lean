import Proofs.Pouw.Lead.CoinBranch.Cells
import Proofs.Pouw.Lead.CoinBranch.Registers

/-!
# The lead's coin-branch bound (proofs)

A program that branches on the coins skips only the words it can predict equal on its path.

* `Cells`: `MH100w`'s floor applied cell by cell to a program chosen from the coins, and the served sparse-noise
  layout's one-bit merges.
* `Registers`: the chunk model against such a program, outcome by outcome: per-entry credit at word writes and under
  whole-register writes (model W), the attained side, and priced selection over uniform branches.
-/
