module

public import Definitions.Pouw.Game.Defs

/-! The witness cost model `outputPriced`, which forbids nothing and only prices commitments, used only by the satisfiability witnesses. -/

@[expose] public section

namespace Pouw

open Classical in
/-- The output-priced model: programs are oracle programs of the workload, activation input, state and salt; cost
is the GEMM size of every committed unit. -/
noncomputable def outputPriced (Q R S : Type) : CostModel Q R S where
  Prog := Workload → (ℕ → Mat) → State → S → QTree Q R Trace
  run prog H U act σ s := (prog U act σ s).eval H
  cost prog H U act σ s :=
    ∑ u ∈ (Finset.range U.layout.N).filter (fun u => ((prog U act σ s).eval H u).isSome),
      ((U.layout.shape u).Wmm : ℝ)
  calls prog H U act σ s := (prog U act σ s).calls H

/-- The free model: the same programs, all at cost 0. Used only to show that the game can be lost. -/
def freeModel (Q R S : Type) : CostModel Q R S where
  Prog := Workload → (ℕ → Mat) → State → S → QTree Q R Trace
  run prog H U act σ s := (prog U act σ s).eval H
  cost _ _ _ _ _ _ := 0
  calls prog H U act σ s := (prog U act σ s).calls H

end Pouw
