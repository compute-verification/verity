import Definitions.Pous.Model.Audit
import Proofs.Pous.Sanity
import Proofs.Pous.Thm1.Theorem1iTimed

/-!
# One more round never helps: the sequential audit's acceptance is monotone in `k`

`pr (seqAccepts sch (k + 1) W A₁ A₂) ≤ pr (seqAccepts sch k W A₁ A₂)` (`pr_seqAccepts_succ_le`): the first `k` answers
of a `(k + 1)`-round run are those of the `k`-round run on the first `k` indices, so an accepted `(k + 1)`-round run
restricts to an accepted `k`-round one, and the extra index is uniform and independent. `pr_seqAccepts_le` iterates it.
`Pous.SecurityProofs.P2RingAuditSecureOfReduces` uses it to carry the `m1p` row at `k = 104` to `k = 105`.
-/

namespace PousP2Bridge

open Pous
open scoped ENNReal

variable {M : IdealModel}

theorem seqAnswers_take {ℓ S B : ℕ} (f : M.Qry → M.Ans) (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S)) :
    ∀ (j : ℕ) (is is' : List (Fin B)) (σ : Bits S),
      (seqAnswers f step j (is ++ is') σ).take is.length = seqAnswers f step j is σ
  | _, [], _, _ => by simp [seqAnswers]
  | j, i :: is, is', σ => by
      simp only [List.cons_append, seqAnswers, List.length_cons, List.take_succ_cons]
      rw [seqAnswers_take f step (j + 1) is is']

/-- The outcome of `k + 1` rounds is that of `k` rounds and one more index. -/
def snocEquiv (sch : Scheme M) (k : ℕ) : Outcome sch k × Fin sch.B ≃ Outcome sch (k + 1) where
  toFun y := (y.1.1, y.1.2.1, Fin.snoc y.1.2.2 y.2)
  invFun x := ((x.1, x.2.1, Fin.init x.2.2), x.2.2 (Fin.last k))
  left_inv y := by simp
  right_inv x := by simp

theorem seqAccepts_of_succ (sch : Scheme M) {S D Q k : ℕ} (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : SeqResponder sch S D Q) (y : Outcome sch k × Fin sch.B)
    (h : seqAccepts sch (k + 1) W A₁ A₂ (snocEquiv sch k y)) : seqAccepts sch k W A₁ A₂ y.1 := by
  obtain ⟨⟨ω, r, I⟩, i⟩ := y
  simp only [seqAccepts, snocEquiv, Equiv.coe_fn_mk] at h ⊢
  rw [List.ofFn_succ', List.ofFn_succ'] at h
  simp only [Fin.snoc_castSucc, Fin.snoc_last, List.concat_eq_append] at h
  have ht := congrArg (List.take (List.ofFn I).length) h
  rw [seqAnswers_take, List.take_left' (by simp)] at ht
  exact ht

theorem pr_seqAccepts_succ_le (sch : Scheme M) {S D Q : ℕ} (k : ℕ) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : SeqResponder sch S D Q) :
    pr (seqAccepts sch (k + 1) W A₁ A₂) ≤ pr (seqAccepts sch k W A₁ A₂) := by
  calc pr (seqAccepts sch (k + 1) W A₁ A₂)
      = pr (fun y => seqAccepts sch (k + 1) W A₁ A₂ (snocEquiv sch k y)) :=
        (PousThm1.pr_equiv (snocEquiv sch k) _).symm
    _ ≤ pr (fun y : Outcome sch k × Fin sch.B => seqAccepts sch k W A₁ A₂ y.1) :=
        pr_mono fun y => seqAccepts_of_succ sch W A₁ A₂ y
    _ = pr (seqAccepts sch k W A₁ A₂) := PousThm1.pr_fst _

theorem pr_seqAccepts_le (sch : Scheme M) {S D Q k k' : ℕ} (hk : k ≤ k') (W : Bits sch.n)
    (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q) :
    pr (seqAccepts sch k' W A₁ A₂) ≤ pr (seqAccepts sch k W A₁ A₂) := by
  induction hk with
  | refl => exact le_rfl
  | step _ ih => exact (pr_seqAccepts_succ_le sch _ W A₁ A₂).trans ih

end PousP2Bridge
