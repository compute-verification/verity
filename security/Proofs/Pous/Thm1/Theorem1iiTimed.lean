module

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Data.Fintype.Powerset
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.GCongr
public import Mathlib.Tactic.Linarith
public import Definitions.Pous
public import Definitions.Pous.Assumptions
public import Proofs.Pous.Statements
public import Proofs.Pous.Sanity
public import Proofs.Pous.Accounting.Numbers
public import Proofs.Pous.Thm1.Theorem1ii

/-! Report Theorem 1(ii), timed: a proof of `Pous.Guarantees.Theorem1iiTimed` with the per-block expander of
`TimedINC`. -/

@[expose] public section

namespace PousThm1ii

open Pous
open scoped ENNReal

end PousThm1ii

open Pous PousThm1ii in
theorem PousProofs.theorem1_ii_timed : Pous.Guarantees.Theorem1iiTimed := by
  intro M sch D Q β S k π hinc W A₁ A₂
  obtain ⟨c, d, hcd⟩ := exists_code (fits S sch.B sch.ℓ β) β (card_fits_le S sch.B sch.ℓ β)
  let good : M.Ω × sch.Coins → Fin sch.B → Prop := fun a i =>
    (A₂.prog W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
      (M.answer a.1) = (sch.enc W a.1 a.2).2 i
  have hre : pr (timedProdAccepts sch k W A₁ A₂) =
      pr (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) => ∀ j, good x.1 (x.2 j)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm
      (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) => ∀ j, good x.1 (x.2 j))
  rw [hre]
  refine (pr_all_good_le sch.B sch.ℓ β S k sch.B_pos good).trans (add_le_add ?_ le_rfl)
  let cmp : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits β :=
    fun W pp C ω => c (A₁ W pp C ω, patch (fun i => (A₂.prog W pp (A₁ W pp C ω) i).run (M.answer ω)) C)
  let exp : Bits sch.n → Bits sch.t → Bits β → Fin sch.B → OracleAlgorithm M (Bits sch.ℓ) :=
    fun W pp w i => ((d w).2 i).elim (A₂.prog W pp (d w).1 i) OracleAlgorithm.ret
  have hexp : ∀ W pp w i, (exp W pp w i).Bounded D Q := by
    intro W pp w i
    cases h : (d w).2 i with
    | none => simp only [exp, h, Option.elim]; exact A₂.bounded _ _ _ _
    | some b => simp only [exp, h, Option.elim]; exact ret_bounded _ _ _
  refine le_trans (pr_mono fun a ha => ?_) (hinc W cmp exp hexp)
  have hfit := mem_fits_patch (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)
    (fun i => (A₂.prog W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
      (M.answer a.1))
    (sch.enc W a.1 a.2).2 ha
  show ∀ i, (exp W _ (cmp W _ _ a.1) i).run (M.answer a.1) = _
  intro i
  simp only [exp, cmp, hcd _ hfit]
  rw [← patch_getD (fun i => (A₂.prog W (sch.enc W a.1 a.2).1
    (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run (M.answer a.1)) (sch.enc W a.1 a.2).2 i]
  cases patch _ _ i <;> rfl
