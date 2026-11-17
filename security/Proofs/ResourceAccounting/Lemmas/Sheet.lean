module

public import Definitions.Registration.Guarantees.Sheet

/-! Steps of `Definitions.Registration.Guarantees.Sheet`'s proofs, each a `def G : Prop` proved by the
`SecurityProofs` theorem of the same name and pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity

def SheetFacetCeilings : Prop :=
  ∀ {K P : Type} [Fintype K] [DecidableEq P] (S : Sheet K P) (Q : Run K),
    (∀ k, 0 ≤ Q.n k) → SheetClock S Q → SheetDispatch Q → SheetTensor S Q → SheetPipe S Q → SheetSmem S Q →
    FacetCeilings S.machine Q

def SheetScalarRouteHalf : Prop :=
  ∀ {K R T L U F X : Type} [Fintype T] [Fintype L] [Fintype U] (Z : Model K R T L U F X) (A : Anatomy K T L U),
    A.Split → SheetWords Z A → Z.q ≤ 1 / 2 → Z.qT ≤ 1 / 2 → Z.bT ≤ 0 → ScalarRoute Z A

end ResourceAccounting.Guarantees.TS
