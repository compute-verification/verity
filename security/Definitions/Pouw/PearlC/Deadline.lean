module

public import Definitions.Pouw.Game.Deadline
public import Definitions.Pouw.PearlC.Game

/-!
# The deadline in Pearl-C's game (trusted)

`Definitions.Pouw.Game.Deadline`'s timed run, the device's entries and the verifier's rule are the game's; Pearl-C's adversaries are its own type, so
their budget by the deadline and what the deadline reads of them are restated here. The named assumption is
`Pouw.Assumptions.HonestClock`, the one both games read, at `CostModel.adversaries`.
-/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

/-- The run spends at most `T` in the real window of length `d` from the salt's entry, and preprocessing and the
online phase together make at most `q` oracle calls, for every oracle and salt (surely). -/
def Adversary.BoundedBy {CM : CostModel Q R S} {k : String} (A : Adversary CM) (U : Workload) (st : Stamps Q R S k)
    (run : TimedRun Q R S) (d T : ℝ) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), run.spentIn H s (st.ts s) (st.ts s + d) ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- The adversaries of Pearl-C's cost model `CM` on the workload `U`, as the deadline reads them. -/
def CostModel.adversaries (CM : CostModel Q R S) (U : Workload) : Adversaries Q R S (Adversary CM) Trace where
  pre A := A.pre
  trace A := A.trace U
  cost A H s := CM.cost A.online H U A.act (A.pre.eval H) s
  calls A H s := CM.calls A.online H U A.act (A.pre.eval H) s

end Pouw.PearlC
