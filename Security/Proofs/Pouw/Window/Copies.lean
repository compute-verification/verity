import Definitions.Pouw.Window

/-!
# The service's units read the tile, not the session: its wrong sets are copy-closed

The deployed window's units are the `N` tiles in each of its sessions (`serviceLaw`), and their statements are the
tiles' statements read in every session (`UnitStmts.copies`), since an instance's public inputs are its tile `t` and the
window's common words, never the session. So at every transcript a tile is wrong in every session or in none
(`wrongCopyClosed_copies`), and on such statements PoUW's family `harmSets W_ref ε ∧ CopyClosed`, at which it reads
`WindowEps` and `ServiceEscapes` gives `η = 0`, holds of a wrong set exactly when `harmSets` does
(`harmSets_copyClosed_iff`).
-/

namespace Pouw.Window

open scoped ENNReal

/-- **The service's wrong sets are copy-closed**: unit `⟨σ, t⟩` of `V.copies S` is wrong in `X` exactly when tile `t`
is wrong in `X` under `V`, whatever the session `σ`. -/
theorem wrongCopyClosed_copies {Tr : Type} {N : ℕ} (V : UnitStmts Tr fun _ : Fin 1 => N) (S : ℕ) :
    WrongCopyClosed (V.copies S) := by
  intro X σ σ' t h
  simp only [UnitStmts.wrong, Finset.mem_filter, Finset.mem_univ, true_and] at h ⊢
  exact h

/-- **On copy-closed wrong sets, PoUW's family is the harm family**: the `CopyClosed` conjunct of
`harmSets W_ref ε ∧ CopyClosed` holds of every wrong set, so the deployed window's harm event is `harmSets` alone. -/
theorem harmSets_copyClosed_iff {Tr : Type} {S N : ℕ} {U : UnitStmts Tr fun _ : Fin S => N}
    (hU : WrongCopyClosed U) (Wref : WUnit (fun _ : Fin S => N) → ℕ) (ε : ℝ≥0∞) (X : Tr) :
    (harmSets Wref ε (U.wrong X) ∧ CopyClosed (U.wrong X)) ↔ harmSets Wref ε (U.wrong X) :=
  ⟨And.left, fun h => ⟨h, hU X⟩⟩

/-- The deployed statements' wrong sets are in PoUW's family whenever they hold the share `ε` of the work. -/
theorem copies_harm_copyClosed {Tr : Type} {N : ℕ} (V : UnitStmts Tr fun _ : Fin 1 => N) (S : ℕ)
    (Wref : WUnit (fun _ : Fin S => N) → ℕ) (ε : ℝ≥0∞) (X : Tr) :
    (harmSets Wref ε ((V.copies S).wrong X) ∧ CopyClosed ((V.copies S).wrong X)) ↔
      harmSets Wref ε ((V.copies S).wrong X) :=
  harmSets_copyClosed_iff (wrongCopyClosed_copies V S) Wref ε X

end Pouw.Window
