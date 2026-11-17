module

public import Definitions.Pouw.PearlC.Mend

/-! rev1's forced zeros: a product no salt makes count is one cut at both ends of the support. -/

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

theorem forcedZero_of_cutForEverySalt {p : Pipeline} {G : ℕ} {alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ} {k : ℕ}
    {x w : Fin k → ℚ} {l : Fin k} (h : CutForEverySalt p G alpha beta k x w l) : ForcedZero p G alpha beta k x w l :=
  ⟨h (fun _ => endPair false) (fun _ => endPair false) (fun _ => endPair_mem_support _) (fun _ => endPair_mem_support _),
    h (fun _ => endPair true) (fun _ => endPair true) (fun _ => endPair_mem_support _) (fun _ => endPair_mem_support _)⟩

end Pouw.PearlC.Mend
