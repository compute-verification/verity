module

public import Definitions.Pouw.NCP.Defs
import Mathlib.Tactic.Bound.Init

/-! Noise that cancels inside one checked product: the lead's algebra only, with no game instance, as the lead is not reviewed. -/

@[expose] public section

namespace Pouw.NCP.Pinned

open Pouw.NCP

def Wrap : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ), InInt32 (mm k A B i j) →
    mmW (3 * k) (stackX k A E1 R) (stackY k B R F1) i j = mm k A B i j

def WrapInt7 : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ), k ≤ 2 ^ 19 - 1 →
    (∀ l < k, InInt7 (A i l)) → (∀ l < k, InInt7 (B l j)) →
    mmW (3 * k) (stackX k A E1 R) (stackY k B R F1) i j = mm k A B i j

/-- The witness: the identity, for every `k`. -/
def SignedPermWitness : Prop :=
  ∀ k : ℕ, SignedPerm k (fun i j => if i = j then 1 else 0)

end Pouw.NCP.Pinned
