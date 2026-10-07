import Definitions.Pouw.Game.Deadline
import Definitions.Pouw.PearlC.Game

/-!
# The deadline in Pearl-C's game (trusted)

`Definitions.Pouw.Game.Deadline`'s timed run and verifier's rule are the game's; Pearl-C's adversaries are its own type, so
their budget by the deadline is restated here. The named assumption is `Pouw.PearlC.Assumptions.HonestClock`.
-/

namespace Pouw.PearlC

variable {Q R S : Type}

/-- The run spends at most `T` after the salt's release and by the deadline `td`, and preprocessing and the online phase
together make at most `q` oracle calls, for every oracle and salt (surely). -/
def Adversary.BoundedBy {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (run : TimedRun Q R S) (td T : ℝ)
    (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), run.spentIn H s run.ts td ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

end Pouw.PearlC
