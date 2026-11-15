module

public import Definitions.Pouw.PearlC.Mend

/-!
# rev1's forced zeros: a product no salt makes count is one cut at both ends of the support

`DeviceRev1`'s debit charges a forced zero (`Mend.ForcedZero`), a product cut at both ends of the noise support, for
the property it stands for, a product cut at every choice of support pairs for its two rows' elements
(`Mend.CutForEverySalt`). The two ends are such choices, the same flat pair at every element of both rows, so
`forcedZero_of_cutForEverySalt` gives the debit's predicate from the property: whatever no salt makes count, the debit
charges.

* `endPair_mem_support`: each end's pair, `(flatLine 0, flatLineF s)` with `s` clear or set, is in `SaltDead.support`.
* `saltRow_endPair`: a row's codes with every element at an end's pair are `endRow`'s.
-/

@[expose] public section

namespace Pouw.PearlC.Mend

open Pouw.Fp8Atom

/-- The pair every element takes at the support's plus end (`minus` clear) or minus end (`minus` set). -/
def endPair (minus : Bool) : (Fin 32 → Code) × (Fin 32 → Code) :=
  (SaltDead.flatLine fun _ => false, SaltDead.flatLineF fun _ => minus)

theorem endPair_mem_support (minus : Bool) : endPair minus ∈ SaltDead.support :=
  ⟨fun _ => false, fun _ => minus, rfl⟩

theorem saltRow_endPair (p : Pipeline) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x : Fin k → ℚ)
    (minus : Bool) : saltRow p alpha beta k x (fun _ => endPair minus) = endRow p alpha beta k x minus :=
  rfl

/-- **A product cut at every salt is a forced zero**: the support's two ends are salts of every element of both rows, so
a product cut at every choice of support pairs is cut at both ends. -/
theorem forcedZero_of_cutForEverySalt {p : Pipeline} {G : ℕ} {alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ} {k : ℕ}
    {x w : Fin k → ℚ} {l : Fin k} (h : CutForEverySalt p G alpha beta k x w l) : ForcedZero p G alpha beta k x w l :=
  ⟨h (fun _ => endPair false) (fun _ => endPair false) (fun _ => endPair_mem_support _) (fun _ => endPair_mem_support _),
    h (fun _ => endPair true) (fun _ => endPair true) (fun _ => endPair_mem_support _) (fun _ => endPair_mem_support _)⟩

end Pouw.PearlC.Mend
