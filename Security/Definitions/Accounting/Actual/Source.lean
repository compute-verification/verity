import Definitions.Accounting.Physical

/-!
# Ingress by source (trusted)

`ingress X I` counts every gate placed outside `X` that `A[X, I]` reads, wherever it was placed. The bandwidth audit's
guarantee on the actual circuit (B3 in `note:console/20261012T0645Z-draft-actual-circuit-model`) is about one source:
the bits the cores read *from HBM* in the window, beside what they read over the side links, which count against the
side budget `β_side (T + RTT)`. So it needs ingress from a location `Y`, and the capacity of the links from `Y`:

* `ingressFrom X Y I`: `in_X(I; Y)`, the width of the gates placed in `Y` and outside `X` that `A[X, I]` reads;
* `linkCapacityFrom ends β X Y`: the total capacity of the links across `X`'s boundary whose end outside `X` is in `Y`.

Ingress is the sum over a partition of the outside into sources, and at most the sum over a cover, and under `Links`
ingress from `Y` is at most `linkCapacityFrom` a tick (`Accounting.SecurityProofs.Actual.ingressFrom_le`).
-/

namespace Accounting

open Verity Finset

namespace Placed

variable {L : Type*} (A : Placed L)

open Classical in
/-- **Ingress from `Y`** `in_X(I; Y)`: the width of the gates placed in `Y` and outside `X` that `A[X, I]` reads. -/
noncomputable def ingressFrom (X Y : Finset L) (I : Set ℕ) : ℕ :=
  A.width ((A.into (A.region X I)).filter fun a => A.x a ∉ X ∧ A.x a ∈ Y)

end Placed

section Links

variable {L E : Type*} [Fintype E]

open Classical in
/-- The total capacity of the links across `X`'s boundary whose end outside `X` is in `Y`. -/
noncomputable def linkCapacityFrom (ends : E → L × L) (β : E → ℕ) (X Y : Finset L) : ℕ :=
  ∑ e ∈ (boundary ends X).filter (fun e => ((ends e).1 ∉ X ∧ (ends e).1 ∈ Y) ∨ ((ends e).2 ∉ X ∧ (ends e).2 ∈ Y)), β e

end Links

end Accounting
