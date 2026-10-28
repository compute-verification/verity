import Definitions.Pouw.Assumptions.Dimension.OpenPipe
import Definitions.Pouw.Assumptions.Dimension.H100Assumptions
import Definitions.Pouw.Assumptions.Dimension.H100LoopFree
import Definitions.Pouw.Assumptions.Dimension.H100Measured

/-!
# The dimension argument's guarantee (trusted)

Each guarantee is a `def G : Prop`, stated over `Definitions.Pouw`'s definitions with the named assumptions it rests on as
hypotheses (`Pouw.Assumptions`). Its proof is the theorem
`Pouw.SecurityProofs.Dimension.G : Pouw.Guarantees.Dimension.G`. A2 on the checked words, whichever pipe class
(`Pouw.Assumptions.Dimension.OpenPipe`). Each keeps the text, the docstring and the `open`s of the statement it was in
its source module.
-/

namespace Pouw.Guarantees.Dimension

open Pouw.Dimension Pouw.Dimension.OpenPipe Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions
  Pouw.Dimension.Forest

/-- **On the checked words, A2 does not depend on the pipe class**: under checkpoint independence and MinRank ≥ 5,
for every machine and `γ₂ ≤ 1`, A2 into `QuadPipe 8` and A2 into `OpenQuadPipe 16 8`, both for the non-final
checked words, are each equivalent to the per-word cost `(1 − γ₂)·16`. One way, a matching program pays 16 per word
(`Theorem1Open`); the other way, the honest program on the words `q` outputs, one priced 16 instruction per word, is
in both classes. -/
def A2WordsIff : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (M : WordCostModel (Noise N m n k)) (γ₂ : ℝ), CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → γ₂ ≤ 1 →
    (A2_algebraicSufficiency γ₂ M (QuadPipe 8) (wordSet (word k m n N R A B) U) ↔
        PerWordCost γ₂ M (word k m n N R A B) U) ∧
      (A2_algebraicSufficiency γ₂ M (OpenQuadPipe 16 8) (wordSet (word k m n N R A B) U) ↔
        PerWordCost γ₂ M (word k m n N R A B) U)

end Pouw.Guarantees.Dimension
