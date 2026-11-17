module

public import Flock.AnnotExpand

/-! The description-level annotation check's property `CheckAgrees`, which nothing proves: `Flock.Annot.check` is code no
soundness theorem reaches and no verifier calls. -/

@[expose] public section

namespace Properties.Annot

open Flock.Annot
open Flock.Program (Prog)

def holds (e : Array (Array Nat)) (l : Nat) : List Nat := (List.range e.size).filter fun g => e[g]!.contains l

def Holds (Φ : Meaning) (P : Prog) (n : Nat) (e : Array (Array Nat)) : Prop :=
  e.size = (P.defs[P.root]!).gates ∧
  (∀ l, ∀ g ∈ holds e l, l < n) ∧
  (Φ.tiling = some .partition → ∀ g < e.size,
    (isInputGate P g = .ok true → e[g]! = #[]) ∧ (isInputGate P g = .ok false → e[g]!.size = 1)) ∧
  (Φ.tiling = some .cover → ∀ g < e.size, isInputGate P g = .ok false → 0 < e[g]!.size) ∧
  (∀ m, Φ.maxCount = some m → n ≤ m) ∧
  (∀ c, Φ.maxGates = some c → ∀ l < n, (holds e l).length ≤ c)

/-- The converse is not claimed: the check refuses some annotations whose expansion satisfies `Φ`. -/
def CheckAgrees (Φ : Meaning) : Prop :=
  ∀ (P : Prog) (D : Desc) (v : CheckResult), check Φ P D = .ok v → v.accept = true →
    ∃ e, expand P D = .ok e ∧ Holds Φ P v.count e

end Properties.Annot
