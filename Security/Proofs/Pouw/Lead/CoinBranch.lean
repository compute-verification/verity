import Proofs.Pouw.Lead.CoinBranch.AdaptiveProofs
import Proofs.Pouw.Lead.CoinBranch.GrindProofs
import Proofs.Pouw.Lead.CoinBranch.PointwiseAdaptiveProofs
import Proofs.Pouw.Lead.CoinBranch.PointwiseProofs

/-!
# The lead's coin branch: a program that reads the coins

Its proofs are in `Lead/CoinBranch/`, its new declarations in `Pouw.SecurityProofs.Lead.CoinBranch`.

* `GrindProofs`: target words lost on a success set of the coins (grinding).
* `AdaptiveProofs`: a program chosen from the coins, in `MH100w`, paid cell by cell, and the step merge on the served
  4-lane layout.
* `PointwiseProofs`: the chunk model's bounds for every run.
* `PointwiseAdaptiveProofs`: an adaptive program pays an entry's distinct values, per entry, lane and register.
-/
