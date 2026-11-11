module

public import Definitions.Accounting.Guarantees.Sheet

/-!
# What the data-sheet axioms give: lemmas

The statements of `Definitions.Accounting.Guarantees.Sheet` that are steps of its guarantees' proofs rather than
guarantees. Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same
name, and audited with the proofs: no record pins it.

* `SheetFacetCeilings`: (I1) on the data-sheet machine, from A1–A5.
* `SheetScalarRouteHalf`: (I3) at the ½ floor (`q, q_T ≤ ½`, `b_T ≤ 0`), from A6 and the labelling's split.
-/

@[expose] public section

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **(I1) from the data sheet.** On a run with counts `≥ 0`, A1–A5 make every facet of the data-sheet machine a
ceiling. -/
def SheetFacetCeilings : Prop :=
  ∀ {K P : Type} [Fintype K] [DecidableEq P] (S : Sheet K P) (Q : Run K),
    (∀ k, 0 ≤ Q.n k) → SheetClock S Q → SheetDispatch Q → SheetTensor S Q → SheetPipe S Q → SheetSmem S Q →
    FacetCeilings S.machine Q

/-- **(I3) at the ½ floor.** On a labelled run the labelling splits, A6 gives SCALAR-ROUTE at any floors
`q, q_T ≤ ½` and `b_T ≤ 0`. -/
def SheetScalarRouteHalf : Prop :=
  ∀ {K R T L U F X : Type} [Fintype T] [Fintype L] [Fintype U] (Z : Model K R T L U F X) (A : Anatomy K T L U),
    A.Split → SheetWords Z A → Z.q ≤ 1 / 2 → Z.qT ≤ 1 / 2 → Z.bT ≤ 0 → ScalarRoute Z A

end Accounting.Guarantees.TS
