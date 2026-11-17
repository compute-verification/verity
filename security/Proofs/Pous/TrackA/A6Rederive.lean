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
public import Proofs.Pous.Thm1.Theorem1iTimed

/-! A6, cheap re-derivation: `PousProofs.a6_rederive_untimed` and `a6_rederive_timed`, as drafted. -/

@[expose] public section

namespace PousThm1

open Pous
open scoped ENNReal

section progMap

variable {M : IdealModel} {α β : Type} (g : α → β) (f : M.Qry → M.Ans)

end progMap

end PousThm1

namespace PousProofs

open Pous PousThm1
open scoped ENNReal

theorem a6_rederive_untimed {M : IdealModel} (sch : Scheme M) (κ : ℕ) (π : ℝ≥0∞)
    (z : Bits sch.n → M.Ω → sch.Coins → Bits κ) (R : Bits sch.n → Bits sch.t → Bits κ → M.Ω → sch.Code)
    (hR : ∀ W ω r, R W (sch.enc W ω r).1 (z W ω r) ω = (sch.enc W ω r).2) (h : INC sch κ π) : 1 ≤ π := by
  classical
  let cmp : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits κ := fun W pp C ω =>
    if hr : ∃ r, sch.enc W ω r = (pp, C) then z W ω hr.choose else fun _ => false
  have hW := h (fun _ => false) cmp R
  rwa [pr_eq_one fun x => ?_] at hW
  show R _ _ (cmp _ _ _ x.1) x.1 = _
  have hex : ∃ r', sch.enc (fun _ => false) x.1 r' =
      ((sch.enc (fun _ => false) x.1 x.2).1, (sch.enc (fun _ => false) x.1 x.2).2) := ⟨x.2, rfl⟩
  simp only [cmp, hex, dite_true]
  have := hR (fun _ => false) x.1 hex.choose
  rw [hex.choose_spec] at this
  exact this

theorem a6_rederive_timed {M : IdealModel} (sch : Scheme M) (κ D Q k : ℕ)
    (z : Bits sch.n → M.Ω → sch.Coins → Bits κ)
    (R : Bits sch.n → Bits sch.t → Bits κ → Fin sch.B → OracleAlgorithm M (Bits sch.ℓ))
    (hRb : ∀ W pp s i, (R W pp s i).Bounded D Q)
    (hR : ∀ W ω r i, (R W (sch.enc W ω r).1 (z W ω r) i).run (M.answer ω) = (sch.enc W ω r).2 i) :
    ∃ (A₁ : Preprocessor sch κ) (A₂ : SeqResponder sch κ D Q), ∀ W, pr (seqAccepts sch k W A₁ A₂) = 1 := by
  classical
  refine ⟨fun W pp C ω => if hr : ∃ r, sch.enc W ω r = (pp, C) then z W ω hr.choose else fun _ => false,
    ⟨fun W pp _ σ i => progMap (fun b => (b, σ)) (R W pp σ i), fun W pp _ σ i => bounded_progMap _ (hRb W pp σ i)⟩,
    fun W => pr_eq_one fun x => ?_⟩
  show seqAnswers _ (fun _ σ i => progMap (fun b => (b, σ)) (R W _ σ i)) 0 _ _ = _
  rw [seqAnswers_keep, List.map_ofFn]
  refine congrArg List.ofFn (funext fun j => ?_)
  have hex : ∃ r', sch.enc W x.1 r' = ((sch.enc W x.1 x.2.1).1, (sch.enc W x.1 x.2.1).2) := ⟨x.2.1, rfl⟩
  simp only [Function.comp_apply, hex, dite_true]
  have := hR W x.1 hex.choose (x.2.2 j)
  rw [hex.choose_spec] at this
  exact this

end PousProofs
