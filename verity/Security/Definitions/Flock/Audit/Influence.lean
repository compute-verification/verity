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

variable {P}

variable (P)

open Classical in
/-- The committed wires of the units of `B`: what they export. -/
noncomputable def exits (B : Finset (Fin n)) : Finset (Fin C.N) :=
  P.committed.filter fun g => ∃ u ∈ B, P.unit g = some u

/-- What a transcript delivers on the wires `D`. -/
def out (D : Finset (Fin C.N)) (X : Fin C.N → Bool) : D → Bool := fun t => X t

open Classical in
/-- What the transcripts with the anchors' inputs whose wrong units lie in `B` deliver on `D`. -/
noncomputable def delivered (D : Finset (Fin C.N)) (a : Fin C.N → Bool) (B : Finset (Fin n)) : Finset (D → Bool) :=
  univ.filter fun y => ∃ X, C.InputsAgree X a ∧ P.wrong X ⊆ B ∧ out D X = y

open Classical in
/-- **The influence set at `δ`**: what is delivered on `D` with the wrong units of some set whose escape `e B` is at
least `δ`. -/
noncomputable def influenceSet (D : Finset (Fin C.N)) (e : Finset (Fin n) → ℝ≥0∞) (a : Fin C.N → Bool)
    (δ : ℝ≥0∞) : Finset (D → Bool) :=
  (univ.filter fun B => δ ≤ e B).biUnion fun B => P.delivered D a B

section Harm

variable {L : Law n} {a : Fin C.N → Bool} {D : Finset (Fin C.N)}

end Harm

section OneStage

variable {L : Law n} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}
  {X : (R : Reg) → Cont L Reg session R → (Fin C.N → Bool)}
  {ext : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Option (Fin C.N → Bool)}
  {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞} {D : Finset (Fin C.N)}

end OneStage

section TwoStage

variable {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞}
  {D : Finset (Fin C.N)}

end TwoStage

end Partition

end FlockSoundness.Audit
