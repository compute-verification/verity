import Definitions.Pouw.Game.Defs

/-!
# The witness cost model (trusted definitions; used only by the satisfiability witnesses)

`outputPriced`: a program is an adaptive oracle program (`QTree`: a tree of queries to `H`, with any computation
between queries) that may read the workload, the activation input, the state and the salt; it reaches `H` only
through queries, and `calls` counts the queries on the path it takes. Its cost is `∑ m_u·k_u·n_u` over the units it
commits to (`some`): committing a unit's transcript costs that unit's GEMM, and nothing else costs anything.

Every function of the oracle answers is a program here, so the model forbids nothing; it only prices commitments.
It is not M_4090, where committing (hashing) is free. What this witness shows and does not show: `NOTES.md`.
-/

namespace Pouw

open Classical in
/-- The output-priced model: programs are oracle programs of the workload, activation input, state and salt; cost
is the GEMM size of every committed unit. -/
noncomputable def outputPriced (Q R S : Type) : CostModel Q R S where
  Prog := Workload → (ℕ → Mat) → State → S → QTree Q R CommitLog
  run prog H U act σ s := (prog U act σ s).eval H
  cost prog H U act σ s :=
    ∑ u ∈ (Finset.range U.layout.N).filter (fun u => ((prog U act σ s).eval H u).isSome),
      ((U.layout.shape u).Wmm : ℝ)
  calls prog H U act σ s := (prog U act σ s).calls H

/-- The free model: the same programs, all at cost 0. Used only to show that the game can be lost. -/
def freeModel (Q R S : Type) : CostModel Q R S where
  Prog := Workload → (ℕ → Mat) → State → S → QTree Q R CommitLog
  run prog H U act σ s := (prog U act σ s).eval H
  cost _ _ _ _ _ _ := 0
  calls prog H U act σ s := (prog U act σ s).calls H

end Pouw
