import Definitions.Core.PrivatePartition

/-! # What a private partition's public predicates establish -/

namespace Verity.Guarantees

/-- **The per-Call check suffices**: when every unit's outputs inside its Call meet the width rule, every unit's
outputs in the whole circuit do. -/
def PrivateFitsOfLocal : Prop :=
  ∀ (C : Circuit) (m n : ℕ) (K : Calls C m) (P : Partition C n) (X : ℕ),
    P.FitsLocally K X → P.Fits X

/-- **The leakage weight**: under the width rule, any set of units, such as the wrong ones, has at most `X` output bits
per unit. -/
def PrivateOutBitsLe : Prop :=
  ∀ (C : Circuit) (n : ℕ) (P : Partition C n) (X : ℕ), P.Fits X →
    ∀ T : Finset (Fin n), ∑ u ∈ T, P.outBits u ≤ T.card * X

/-- **A committed partition fits the public count, or the two commitments hold a collision**: when `r₀` commits to
`root` under `Q`, any commitment `r` with the same root, of any partition under any salts, has at most `Q.units`
units, each within the width rule, so any set of them has at most `Q.X` output bits per unit; or `H` has a collision
between a string that `r₀` hashes on its way to the root and a different one that `r` hashes on its way there
(`Advice.Committed.hashInputs`).

It holds for every hash `H`; the deployment's is SHA-512. The collision ranges over the finitely many strings the two
commitments hash, so for an `H` injective on them there is none, and then `r`'s partition fits. The premises are
format facts, not properties of `H`: the frame-v3 messages are separated (`Advice.Framing.Separated`), and the rows
decode to their partition (`Advice.Rows.Decodes`). -/
def CommittedMeets : Prop :=
  ∀ (C : Circuit) (H : List UInt8 → Digest) (F : Advice.Framing), F.Separated →
    ∀ (A : Advice.Rows C), A.Decodes →
      ∀ (Q : Predicates) (root : Digest) (r₀ : Advice.Committed C), Commits H F A Q r₀ root →
        ∀ r : Advice.Committed C, r.root H F A = root →
          (r.n ≤ Q.units ∧ r.P.Fits Q.X ∧ ∀ T : Finset (Fin r.n), ∑ u ∈ T, r.P.outBits u ≤ T.card * Q.X) ∨
            ∃ x ∈ r₀.hashInputs H F A, ∃ y ∈ r.hashInputs H F A, x ≠ y ∧ H x = H y  -- a collision of `H`

end Verity.Guarantees
