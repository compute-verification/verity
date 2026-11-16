module

public import Definitions.Slowdown.Covering

@[expose] public section

namespace Slowdown.SecurityProofs.Covering

open Verity Finset Slowdown.Covering

variable {C : Circuit}

theorem FeedsWithin.mem_right {R : Finset (Fin C.N)} {g t : Fin C.N} (h : FeedsWithin C R g t) : t ∈ R := by
  induction h with
  | refl ht => exact ht
  | step _ _ _ ih => exact ih

end Slowdown.SecurityProofs.Covering
