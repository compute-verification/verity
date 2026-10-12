import Definitions.Accounting.Actual

/-!
# Ingress by source (trusted)

`ingress X I` counts every gate placed outside `X` that `A[X, I]` reads, wherever it was placed. The bandwidth audit's
guarantee on the actual circuit (B3 in `note:console/20261012T0645Z-draft-actual-circuit-model`) is about one source:
the bits the cores read *from HBM* in the window, beside what they read over the side links. So it needs ingress from a
unit `Y`, and capacities per source rather than per unit:

* `ingressFrom X Y I`: `in_X(I; Y)`, the width of the gates placed in `Y` and outside `X` that `A[X, I]` reads;
* `LinksFrom 𝒮 β`: for each pair `(X, Y)` of `𝒮`, at most `β X Y` bits a tick cross from `Y` into `X`, the model's
  `β_e` summed over the links from `Y` to `X`.

Ingress is the sum over a partition of the outside into sources, and at most the sum over a cover
(`Accounting.SecurityProofs.Actual.ingress_le_ingressFrom_add`).
-/

namespace Accounting

open Verity Finset

namespace Placed

variable {L : Type*} (A : Placed L)

open Classical in
/-- **Ingress from `Y`** `in_X(I; Y)`: the width of the gates placed in `Y` and outside `X` that `A[X, I]` reads. -/
noncomputable def ingressFrom (X Y : Finset L) (I : Set ℕ) : ℕ :=
  A.width ((A.into (A.region X I)).filter fun a => A.x a ∉ X ∧ A.x a ∈ Y)

/-- **Links by source**: for each pair `(X, Y)` of `𝒮`, at most `β X Y` bits a tick cross from `Y` into `X`. -/
def LinksFrom (𝒮 : Set (Finset L × Finset L)) (β : Finset L → Finset L → ℕ) : Prop :=
  ∀ p ∈ 𝒮, ∀ a b, A.ingressFrom p.1 p.2 (Set.Ico a b) ≤ β p.1 p.2 * (b - a)

end Placed

end Accounting
