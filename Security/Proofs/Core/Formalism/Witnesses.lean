module

public import Definitions.Core.Guarantees.Formalism.Witnesses
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Tactic.NormNum
public import Mathlib.Tactic.Ring

/-!
# The size of a `(c, k)`-partitioned circuit and Theorem 3: proofs of the honest and nontrivial lemmas

The proofs of `Security/Definitions/Core/Guarantees/Formalism/Witnesses.lean`, at the instances of
`Security/Definitions/Core/Formalism/Instances.lean`, each by evaluation: the weights are natural numbers, so `decide`
computes each counted weight and `exact_mod_cast` carries it to `ℝ`.
-/

@[expose] public section

namespace Verity.SecurityProofs

theorem CKSize_honest : Guarantees.CKSize_honest := ⟨⟨by decide, by decide⟩, by decide, by decide⟩

theorem CKSize_nontrivial : Guarantees.CKSize_nontrivial :=
  ⟨fun h => absurd (h.card 0) (by decide), by decide⟩

theorem Counting_honest : Guarantees.Counting_honest := by
  have cap : ∀ u, ∑ v ∈ Formalism.counted ∩ Formalism.Honest.gates u, Formalism.weight v ≤ 2 := by
    decide +kernel
  have total : ∑ v ∈ Formalism.counted, Formalism.weight v = 4 := by decide +kernel
  have attained : (2 : ℝ) / 2 * ∑ v ∈ Formalism.counted, (Formalism.weight v : ℝ) = (2 * 2 : ℕ) := by
    rw [show ∑ v ∈ Formalism.counted, (Formalism.weight v : ℝ) = 4 by exact_mod_cast total]
    norm_num
  refine ⟨by decide, by norm_num, fun u => by exact_mod_cast cap u, attained, fun T₀ _ => ?_⟩
  rw [slowdown, show (2 : ℝ) / (T₀ * 2) * ∑ v ∈ Formalism.counted, (Formalism.weight v : ℝ) =
    (2 / 2 * ∑ v ∈ Formalism.counted, (Formalism.weight v : ℝ)) / T₀ by ring, attained]

theorem Counting_nontrivial : Guarantees.Counting_nontrivial := by
  have over : ∑ v ∈ Formalism.counted ∩ Formalism.Honest.gates 1, Formalism.heavy v = 3 := by decide +kernel
  have total : ∑ v ∈ Formalism.counted, Formalism.heavy v = 5 := by decide +kernel
  refine ⟨by decide, by norm_num, fun h => ?_, ?_⟩
  · have h1 := h 1
    rw [show ∑ v ∈ Formalism.counted ∩ Formalism.Honest.gates 1, (Formalism.heavy v : ℝ) = 3 by
      exact_mod_cast over] at h1
    norm_num at h1
  · rw [show ∑ v ∈ Formalism.counted, (Formalism.heavy v : ℝ) = 5 by exact_mod_cast total]
    norm_num

end Verity.SecurityProofs
