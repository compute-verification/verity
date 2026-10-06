import Definitions.Flock.Audit.OneStage
import Definitions.Flock.Audit.Harm

/-! The definitions of `Proofs.Flock.Soundness.Audit.Influence` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Circuit

variable (C : Circuit)

/-- `g` feeds `t` along a path none of whose gates is in `K`, both ends included. -/
inductive FeedsAvoiding (K : Finset (Fin C.N)) : Fin C.N → Fin C.N → Prop
  | refl {t : Fin C.N} : t ∉ K → FeedsAvoiding K t t
  | step {g h t : Fin C.N} : g ∉ K → g ∈ (C.op h).args → FeedsAvoiding K h t → FeedsAvoiding K g t

end Circuit

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- **A separator of the units `B` from the delivered wires `D`** (a downstream cut): every path from a gate of a unit
of `B` to a wire of `D` meets `K`. -/
def Separates (K : Finset (Fin C.N)) (B : Finset (Fin n)) (D : Finset (Fin C.N)) : Prop :=
  ∀ g t u, P.unit g = some u → u ∈ B → t ∈ D → ¬ C.FeedsAvoiding K g t

/-- What a transcript delivers on the wires `D`. -/
def out (D : Finset (Fin C.N)) (X : Fin C.N → Bool) : D → Bool := fun t => X t

end Partition

end FlockSoundness.Audit
