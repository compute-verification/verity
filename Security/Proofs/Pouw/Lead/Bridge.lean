import Proofs.Pouw.Lead.Bridge.Game
import Proofs.Pouw.Lead.Bridge.Witness

/-!
# The lead's bridge to the PoUW game

* `Game`: the game's `Gγ` for lead calls from `G_tail` (`GammaFromTail`), and `EndToEnd` through it
  (`EndToEndFromTail`).
* `Witness`: their honest and nontrivial witnesses at T 512 × 64, and `FixedGapBelow` at the three deployed points
  from `NontrivialTail` (`fixedGapBelow`).

Its proofs go in `Lead/Bridge/`, its declarations in `Pouw.SecurityProofs.Lead.Bridge`, and each file is imported here.
-/
