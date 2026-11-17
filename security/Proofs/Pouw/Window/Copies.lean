module

public import Definitions.Pouw.Window
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! The service's units read the tile, not the session, so its incorrect sets are copy-closed and PoUW's family on them
is `harmSets` alone. -/

@[expose] public section

namespace Pouw.Window

open scoped ENNReal

theorem incorrectCopyClosed_copies {Tr : Type} {N : ℕ} (V : UnitStmts Tr fun _ : Fin 1 => N) (S : ℕ) :
    IncorrectCopyClosed (V.copies S) := by
  intro X σ σ' t h
  simp only [UnitStmts.incorrect, Finset.mem_filter, Finset.mem_univ, true_and] at h ⊢
  exact h

theorem harmSets_copyClosed_iff {Tr : Type} {S N : ℕ} {U : UnitStmts Tr fun _ : Fin S => N}
    (hU : IncorrectCopyClosed U) (Wref : WUnit (fun _ : Fin S => N) → ℕ) (ε : ℝ≥0∞) (X : Tr) :
    (harmSets Wref ε (U.incorrect X) ∧ CopyClosed (U.incorrect X)) ↔ harmSets Wref ε (U.incorrect X) :=
  ⟨And.left, fun h => ⟨h, hU X⟩⟩

theorem copies_harm_copyClosed {Tr : Type} {N : ℕ} (V : UnitStmts Tr fun _ : Fin 1 => N) (S : ℕ)
    (Wref : WUnit (fun _ : Fin S => N) → ℕ) (ε : ℝ≥0∞) (X : Tr) :
    (harmSets Wref ε ((V.copies S).incorrect X) ∧ CopyClosed ((V.copies S).incorrect X)) ↔
      harmSets Wref ε ((V.copies S).incorrect X) :=
  harmSets_copyClosed_iff (incorrectCopyClosed_copies V S) Wref ε X

end Pouw.Window
