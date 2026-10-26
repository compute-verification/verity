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
import Proofs.Pous.Thm1.Theorem1ii

/-!
# Report Theorem 1(ii), timed: a proof of `Pous.Guarantees.Theorem1iiTimed`

The basic compression argument (report §3, proof of Theorem 1(ii)), with the per-block expander of `TimedINC`. Fix
`W`, `A₁` and the `(D, Q)`-bounded product-form responder `A₂`. For a setup `a = (ω, r)` with `σ = A₁(W, pp, C, ω)`,
block `i` is *good* if `A₂.prog(W, pp, σ, i)` run against `ω` returns `C i`, and `E a` says the wrong blocks fit in
`β` bits: `S + B + #wrong·ℓ ≤ β`.

1. Averaging (`pr_all_good_le`). Off `E`, `#good ≤ p₀·B`, so `k` uniform challenges all land on good blocks with
   probability `(#good/B)^k ≤ p₀^k`. Hence `Pr[accept] ≤ Pr[E] + p₀^k`.
2. Compression. The message `(σ, patch)`, where `patch i = some (C i)` on wrong blocks and `none` on good ones, takes
   at most `2^S·2^B·2^(β−S−B) = 2^β` values on `E` (`card_fits_le`), so it has a `β`-bit code (`exists_code`). The
   message does not depend on `ω`, so the expander for block `i` decodes it without the primitive and is either
   `ret (C i)` or `A₂.prog(W, pp, σ, i)`: `(D, Q)`-bounded. On `E` every block is rebuilt, so `Pr[E] ≤ π` by
   `TimedINC D Q β π`.
-/

namespace PousThm1ii

open Pous
open scoped ENNReal

/-! ### Uniform probability -/

/-! ### The threshold `p₀` -/

/-! ### Compression -/

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
  let exp : Bits sch.n → Bits sch.t → Bits β → Fin sch.B → Prog M (Bits sch.ℓ) :=
    fun W pp w i => ((d w).2 i).elim (A₂.prog W pp (d w).1 i) Prog.ret
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
