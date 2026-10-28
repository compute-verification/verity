import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Powerset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers
import Proofs.Pous.Thm1.Theorem1iTimed

/-!
# A6, cheap re-derivation: `PousProofs.a6_rederive_untimed` and `a6_rederive_timed`, as drafted

The advice `z W ω r` depends on the coins, which the compressor (or preprocessor) never sees. It uses instead
the advice of any coins `r'` consistent with what it does see, `enc W ω r' = (pp, C)`: the true coins are
consistent, and `R` rebuilds `(enc W ω r').2 = C` from them.

* Untimed: the expander is `R`, so it succeeds at every setup, and `INC(κ, π)` forces `π ≥ 1`.
* Timed: `A₁` stores that advice, and each sequential round runs `R` on the challenged index and keeps the state
  (`progMap`, `seqAnswers_keep`), so every challenge sequence is answered correctly.
-/

namespace PousThm1

open Pous
open scoped ENNReal

/-! ### Programs -/

section progMap

variable {M : IdealModel} {α β : Type} (g : α → β) (f : M.Qry → M.Ans)

end progMap

/-! ### Sequential responders -/

end PousThm1

namespace PousProofs

open Pous PousThm1
open scoped ENNReal

/-- A6, untimed. If `C = R(W, pp, z)` for some `κ`-bit advice `z` (any function of the setup), `INC(κ, π)` fails
for every `π < 1`. -/
theorem a6_rederive_untimed {M : IdealModel} (sch : Scheme M) (κ : ℕ) (π : ℝ≥0∞)
    (z : Bits sch.n → M.Ω → sch.Coins → Bits κ) (R : Bits sch.n → Bits sch.t → Bits κ → M.Ω → sch.Code)
    (hR : ∀ W ω r, R W (sch.enc W ω r).1 (z W ω r) ω = (sch.enc W ω r).2) (h : INC sch κ π) : 1 ≤ π := by
  classical
  -- the compressor stores the advice of any coins consistent with `(pp, C)`
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

/-- A6, timed. If every block is re-derived from `κ`-bit advice by a `(D, Q)`-bounded program, a `κ`-bit adversary
passes every sequential audit with probability 1 (the timed form of `redesign.md` Proposition 1). -/
theorem a6_rederive_timed {M : IdealModel} (sch : Scheme M) (κ D Q k : ℕ)
    (z : Bits sch.n → M.Ω → sch.Coins → Bits κ)
    (R : Bits sch.n → Bits sch.t → Bits κ → Fin sch.B → OracleProg M (Bits sch.ℓ))
    (hRb : ∀ W pp s i, (R W pp s i).Bounded D Q)
    (hR : ∀ W ω r i, (R W (sch.enc W ω r).1 (z W ω r) i).run (M.answer ω) = (sch.enc W ω r).2 i) :
    ∃ (A₁ : Preprocessor sch κ) (A₂ : SeqResponder sch κ D Q), ∀ W, pr (seqAccepts sch k W A₁ A₂) = 1 := by
  classical
  -- `A₁` stores the advice of any coins consistent with `(pp, C)`; each round re-derives the block and keeps it
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
