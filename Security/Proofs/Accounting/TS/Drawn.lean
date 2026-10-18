import Proofs.Accounting.TS.Fork

/-!
# (I8) for a sampled window

Stated after RT-I8 (note:20261018T0728Z-draft-rt-i8, decision 114); the proofs are `sorry` stubs until E1(c) per unit
(`UnitFixed`) has its card.

`DrawnRightOff`: the event is "in some round, binding fails (`MovedIn`) or some drawn unit's proof is unsound", which
the union bound puts at `S·(δ_bind + δ_ks)`. Off it, an accepted proof opens its unit's committed values (not
`MovedIn`), which satisfy its relation (not `Unsound`), so they are right (`Pins`).

`ProofsSoundOfDrawn`: `DrawnRightOff`'s event; off it, on an accepted path every drawn unit's proof is accepted (`Of`),
so the ideal game accepts (`Of`), and the good units' words over `S` (`UnitFixed`) are at least
`(S·N_w − d·N_w)/S` by `Covers` and `Σ wt = S·N_w`.

`DrawnBindingOfFork`: a round's binding failure moves one of its positions (`Moved`); per position, the moved mass
squared is at most the forked mass at its cut, by the set-valued form of `BindingOfFork`'s count: with `A`
continuations opening the position, `E_v` of them to `v` alone and `c` the plurality, the moved are `M = A − E_c` and
the forked pairs `A² − Σ_v E_v² ≥ A² − E_c·A = A·M ≥ M²`. The union over `L` and Cauchy–Schwarz give `|L|²`.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- A continuation's continuation from the same coin is a continuation of the first path. -/
theorem spliceAt_left {C B : Type} {m : ℕ} : ∀ {n : ℕ} (w k : ℕ) (x z z' : Paths (Paths C m × B) n),
    spliceAt w k (spliceAt w k x z) z' = spliceAt w k x z'
  | 0, _, _, _, _, _ => rfl
  | _ + 1, 0, k, x, z, z' => by simp only [spliceAt, splice_left]
  | _ + 1, w + 1, k, x, z, z' => by simp only [spliceAt, spliceAt_left w k x.2 z.2 z'.2]

/-- Swapping two paths' coins from round `w`'s coin `k` on, twice, gives the first back. -/
theorem spliceAt_swap {C B : Type} {m : ℕ} : ∀ {n : ℕ} (w k : ℕ) (x z : Paths (Paths C m × B) n),
    spliceAt w k (spliceAt w k x z) (spliceAt w k z x) = x
  | 0, _, _, _, _ => rfl
  | _ + 1, 0, k, x, z => by simp only [spliceAt, splice_swap]
  | _ + 1, w + 1, k, x, z => by simp only [spliceAt, spliceAt_swap w k x.2 z.2]

namespace Plurality

variable {X : Type} [Fintype X] {sp : X → X → X}

/-- Summing over every path and every continuation of it is summing over paths, each `|X|` times. -/
theorem sum_sp (hs : ∀ x z, sp (sp x z) (sp z x) = x) (q : X → ℕ) :
    ∑ x, ∑ z, q (sp x z) = Fintype.card X * ∑ y, q y := by
  let Φ : X × X → X × X := fun p => (sp p.1 p.2, sp p.2 p.1)
  have hΦ : Function.Involutive Φ := fun p => by simp only [Φ, hs]
  have h := Equiv.sum_comp (hΦ.toPerm Φ) (fun p => q p.1)
  simp only [Function.Involutive.coe_toPerm, Φ] at h
  rw [← Fintype.sum_prod_type' (fun x z => q (sp x z)), h, Fintype.sum_prod_type (fun p => q p.1),
    Finset.mul_sum]
  simp [Finset.sum_const, Finset.card_univ, mul_comm]

end Plurality

/-- A sampled window's drawn units are right at their committed values, outside binding and joint soundness failures. -/
theorem DrawnRightOff : Accounting.Guarantees.TS.DrawnRightOff := by
  sorry

/-- (I8) for a sampled window, from `DrawnRightOff`'s pieces, the share's cover and E1(c) per unit. -/
theorem ProofsSoundOfDrawn : Accounting.Guarantees.TS.ProofsSoundOfDrawn := by
  sorry

/-- Binding at every position's cut from the fork form. -/
theorem DrawnBindingOfFork : Accounting.Guarantees.TS.DrawnBindingOfFork := by
  sorry

end Accounting.SecurityProofs.TS
