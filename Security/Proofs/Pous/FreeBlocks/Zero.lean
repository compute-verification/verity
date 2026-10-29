import Definitions.Pous.Model.FreeBlocks

/-!
# The free-blocks audit with no free blocks is the sequential audit

`seqAcceptsFree` with `F = 0` is `seqAccepts`, so `AuditSecureFree sch 0` is `AuditSecure sch .sequential`: the game
`P2SlackFamilyFreeBlocks` reads changes only what a free block lets through.
-/

namespace Pous.FreeBlocks

variable {M : IdealModel}

theorem seqAnswers_length {ℓ S B : ℕ} (f : M.Qry → M.Ans) (step : ℕ → Bits S → Fin B → OracleAlgorithm M (Bits ℓ × Bits S)) :
    ∀ (j : ℕ) (is : List (Fin B)) (σ : Bits S), (seqAnswers f step j is σ).length = is.length
  | _, [], _ => rfl
  | j, i :: is, σ => by simp [seqAnswers, seqAnswers_length f step (j + 1) is]

theorem seqAcceptsFree_zero (sch : Scheme M) {S D Q : ℕ} (k : ℕ) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (Fr : FreeSet sch 0) (A₂ : SeqResponder sch S D Q) (x : Outcome sch k) :
    seqAcceptsFree sch k W A₁ Fr A₂ x ↔ seqAccepts sch k W A₁ A₂ x := by
  simp only [seqAcceptsFree, seqAccepts]
  have hF : ∀ W pp C ω, Fr.free W pp C ω = ∅ := fun W pp C ω =>
    Finset.card_eq_zero.mp (Nat.le_zero.mp (Fr.card_le W pp C ω))
  simp only [hF, Finset.notMem_empty, not_false_eq_true, forall_const]
  constructor
  · intro h
    apply List.ext_getElem?
    intro n
    by_cases hn : n < k
    · rw [h ⟨n, hn⟩]; simp [hn]
    · have hl := seqAnswers_length (M.answer x.1) (A₂.step W (sch.enc W x.1 x.2.1).1) 0 (List.ofFn x.2.2)
        (A₁ W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1)
      simp only [List.length_ofFn] at hl
      rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by simp; omega)]
  · intro h j
    rw [h]; simp

/-- With no free blocks, the free-blocks target is the sequential audit's. -/
theorem auditSecureFree_zero (sch : Scheme M) (S D Q k : ℕ) (δ ε : ENNReal) :
    AuditSecureFree sch 0 S D Q k δ ε ↔ AuditSecure sch .sequential S D Q k δ ε := by
  have hpr : ∀ W (A₁ : Preprocessor sch S) (Fr : FreeSet sch 0) (A₂ : SeqResponder sch S D Q),
      pr (seqAcceptsFree sch k W A₁ Fr A₂) = pr (seqAccepts sch k W A₁ A₂) := by
    intro W A₁ Fr A₂
    have h : seqAcceptsFree sch k W A₁ Fr A₂ = seqAccepts sch k W A₁ A₂ :=
      funext fun x => propext (seqAcceptsFree_zero sch k W A₁ Fr A₂ x)
    simp only [h]
  constructor
  · intro h W A₁ A₂
    rw [← hpr W A₁ ⟨fun _ _ _ _ => ∅, fun _ _ _ _ => by simp⟩ A₂]
    exact h W A₁ _ A₂
  · intro h W A₁ Fr A₂
    rw [hpr]
    exact h W A₁ A₂

end Pous.FreeBlocks
