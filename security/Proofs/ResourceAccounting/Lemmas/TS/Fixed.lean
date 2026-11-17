module

public import Definitions.Pouw.Assumptions.TS
public import Definitions.Pouw.Assumptions.TS.Fixed

/-! Steps of proofs of `UnitFixed` from option A's fixed word (`Definitions.Pouw.TS.Fixed`), pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity

def UnitFixedOfWords : Prop :=
  ∀ {K T L U : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] {C B : Type} [Fintype C] [Fintype B]
    {S m s : ℕ} {J Q Pos V N W : Type} [Fintype J] [Nonempty V] [AddCommGroup W]
    (P : Play (Paths C m × B) S K T L U) (pf : WordLayer C B S m s J Q Pos V N W)
    (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ),
    (∀ w j, 0 ≤ wt w j) → (∀ w j, wt w j ≠ 0 → pf.Fixed wt (w, j)) → ModesRun pf mode → CountsFixed P pf wt folded →
      UnitFixed P pf.toSampledLayer mode wt folded

/-- Its additive operands, and so its words, may differ on the continuations from its coin. -/
def AdditiveFixed : Prop :=
  ∀ {C B : Type} [Fintype C] [Fintype B] {S m s : ℕ} {J Q Pos V N W : Type} [Nonempty V] [AddCommGroup W]
    (pf : WordLayer C B S m s J Q Pos V N W) (u : Fin S × J), (∀ k, pf.role u k = .mul → pf.Bound u k) →
    ∀ x, pf.Computed x u →
      pf.work x u = pf.prod u (pf.mulView x u) (pf.noise x u) ∧
        ∀ z, pf.mulView (spliceAt u.1 (pf.coin u.1 u.2) x z) u = pf.mulView x u

end ResourceAccounting.Guarantees.TS
