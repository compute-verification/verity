import Definitions.Core.Expand
import Definitions.InferenceOnly

/-!
# The policy at a rate as a `(c, k)`-partition

`CompliesRate c α` asks for IUs of at most `c` gates, each reading at most `α` novel bits per gate. On a Boolean
declaration a novel bit is an unregistered input gate, so an IU reads at most `α |I| ≤ α c` of them, and padding each IU
to `c` gates makes the division a `(c, ⌊αc⌋)`-partition of a declaration that contains it (`CompliesCK`), which
complies at `(c, ⌊αc⌋)` and on which the working draft's Lemma 1 and Theorem 3 apply. A multi-bit declaration first
takes its bit-level expansion (`ExpandComplies`): an IU of gates at most `W` bits wide becomes one of at most `c W`
one-bit gates, reading the same novel bits.
-/

namespace InferenceOnly.Guarantees

open Verity Finset

/-- **The policy at a rate is a padded `(c, k)`-partition.** A Boolean declaration that complies at `c` gates per IU
and rate `α` has a division into IUs that pads to a `(c, ⌊αc⌋)`-partition. -/
def CompliesCK : Prop :=
  ∀ (c : ℕ) (α : ℝ) (d : Declaration), d.D.Boolean → CompliesRate c α d →
    ∃ (B : ℕ) (P : Partition d.D B) (d' : Declaration) (P' : Partition d'.D B),
      Nonempty (Pad d P d' P') ∧ P'.IsCK d' c ⌊α * c⌋₊

/-- **The policy at a rate through the bit-level expansion.** If every computed gate is at least one bit wide and every
gate at most `W`, a declaration that complies at `c` gates per IU and rate `α` has an expansion that complies at
`c W` gates per IU and the same rate. -/
def ExpandComplies : Prop :=
  ∀ (c W : ℕ) (α : ℝ) (d : Declaration), 0 ≤ α → (∀ g, d.D.isInput g = false → 1 ≤ d.D.w g) →
    (∀ g, d.D.w g ≤ W) → CompliesRate c α d → CompliesRate (c * W) α d.expand

end InferenceOnly.Guarantees
