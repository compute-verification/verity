import Definitions.Pouw.Game.Defs

/-!
# NCP's witness cost model (trusted; used only by `TT_NCP`'s satisfiability witness)

`outputPriced3`: `outputPriced` (`Definitions/Pouw/Witness/Model.lean`) with every committed unit priced at `3·m·k·n`,
the work `TT_NCP` credits. It forbids nothing and only prices commitments; it is not M_4090.
-/

namespace Pouw.NCP

open Classical in
/-- The output-priced model at `3·m·k·n` per committed unit. -/
noncomputable def outputPriced3 (Q R S : Type) : CostModel Q R S where
  Prog := Workload → (ℕ → Mat) → State → S → QTree Q R CommitLog
  run prog H U act σ s := (prog U act σ s).eval H
  cost prog H U act σ s :=
    ∑ u ∈ (Finset.range U.layout.N).filter (fun u => ((prog U act σ s).eval H u).isSome),
      3 * ((U.layout.shape u).Wmm : ℝ)
  calls prog H U act σ s := (prog U act σ s).calls H

end Pouw.NCP
