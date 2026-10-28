import Definitions.Pouw.NCP.Defs

/-!
# Noise that cancels inside one checked product: pinned statements (trusted)

Algebra of the lead only; no game instance (the lead is not reviewed).
-/

namespace Pouw.NCP.Pinned

open Pouw.NCP

/-- **int32 exactness with wraparound**: the int32 accumulator over the `3k`-deep stacked product returns `A·B`
exactly whenever `A·B` is in int32, whatever the intermediate sums do. -/
def Wrap : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ), InInt32 (mm k A B i j) →
    mmW (3 * k) (stackX k A E1 R) (stackY k B R F1) i j = mm k A B i j

/-- **int32 exactness for int7 operands**: for int7 `A`, `B` and `k ≤ 2^19 − 1`, with any noise. -/
def WrapInt7 : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ), k ≤ 2 ^ 19 - 1 →
    (∀ l < k, InInt7 (A i l)) → (∀ l < k, InInt7 (B l j)) →
    mmW (3 * k) (stackX k A E1 R) (stackY k B R F1) i j = mm k A B i j

/-- **The hypotheses of `Range` are satisfiable**: for every `k`, the identity is a signed permutation. -/
def SignedPermWitness : Prop :=
  ∀ k : ℕ, SignedPerm k (fun i j => if i = j then 1 else 0)

end Pouw.NCP.Pinned
