import Definitions.Core.Expand
import Definitions.Nci

/-!
# NCI's policy as a `(c, k)`-partition

`Complies Γ α` asks for IUs of at most `Γ` gates, each reading at most `α` novel bits per gate. On a Boolean
declaration a novel bit is an unregistered input gate, so an IU reads at most `α |I| ≤ α Γ` of them, and padding each IU
to `Γ` gates makes the division a `(Γ, ⌊αΓ⌋)`-partition of a declaration that contains it (`CompliesCK`), on which the
working draft's Lemma 1 and Theorem 3 apply. A multi-bit declaration first takes its bit-level expansion
(`ExpandComplies`): an IU of gates at most `W` bits wide becomes one of at most `Γ W` one-bit gates, reading the same
novel bits.
-/

namespace Nci.Guarantees

open Verity Finset

/-- **NCI's policy is a padded `(c, k)`-partition.** A Boolean declaration that complies with the NCI policy at `(Γ, α)`
has a division into IUs that pads to a `(Γ, ⌊αΓ⌋)`-partition. -/
def CompliesCK : Prop :=
  ∀ (Γ : ℕ) (α : ℝ) (d : Registered), d.D.Boolean → Complies Γ α d →
    ∃ (B : ℕ) (P : Partition d.D B) (d' : Registered) (P' : Partition d'.D B),
      Nonempty (Pad d P d' P') ∧ P'.IsCK d' Γ ⌊α * Γ⌋₊

/-- **NCI's policy through the bit-level expansion.** If every computed gate is at least one bit wide and every gate at
most `W`, a declaration that complies at `(Γ, α)` has an expansion that complies at `(Γ W, α)`. -/
def ExpandComplies : Prop :=
  ∀ (Γ W : ℕ) (α : ℝ) (d : Registered), 0 ≤ α → (∀ g, d.D.isInput g = false → 1 ≤ d.D.w g) →
    (∀ g, d.D.w g ≤ W) → Complies Γ α d → Complies (Γ * W) α d.expand

end Nci.Guarantees
