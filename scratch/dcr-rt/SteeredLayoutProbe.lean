import Models.Accounting.Assumptions.Stacking

/-! DCR-RT scratch, not for landing (red team of `Accounting.Guarantees.DeclaredCircuitRequirement`).

`SteeredLayout layout steer` constrains no prover by itself: every layout that reads a finite, nonempty `Y` is steered
at `a = |Y|` bits (a one-hot code of `y`). So G's soundness never rests on `SteeredLayout`; what a deployment rests on is
the size of the `a` it instantiates G at, which is `ScheduleDeterministic`'s to justify. -/

open Accounting

theorem DCRRT.steeredLayout_exists {Pfx Y : Type} {N T M : ℕ} [Fintype Y] [Nonempty Y]
    (layout : Pfx → Y → Fin N ↪ Fin T × Fin M) :
    ∃ (a : ℕ) (steer : Pfx → Y → Pous.Bits a), Assumptions.SteeredLayout layout steer := by
  classical
  let e := Fintype.equivFin Y
  let code : Y → Pous.Bits (Fintype.card Y) := fun y i => decide (i = e y)
  have hcode : Function.Injective code := by
    intro y₁ y₂ h
    have h1 : decide (e y₁ = e y₁) = decide (e y₁ = e y₂) := congrFun h (e y₁)
    exact e.injective (of_decide_eq_true (h1.symm.trans (decide_eq_true rfl)))
  refine ⟨Fintype.card Y, fun _ y => code y, ?_⟩
  show ∃ slot : Pfx → Pous.Bits (Fintype.card Y) → Fin N ↪ Fin T × Fin M, ∀ c y, layout c y = slot c (code y)
  exact ⟨fun c s => layout c (Function.invFun code s), fun c y => by
    show layout c y = layout c (Function.invFun code (code y))
    rw [Function.leftInverse_invFun hcode y]⟩

#print axioms DCRRT.steeredLayout_exists
