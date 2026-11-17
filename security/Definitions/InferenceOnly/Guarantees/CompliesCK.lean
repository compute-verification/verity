module

public import Definitions.Core.Expand
public import Definitions.InferenceOnly

/-! The policy at a rate as a `(c, k)`-partition (`CompliesCK`), a multi-bit declaration through its bit-level
expansion. -/

@[expose] public section

namespace InferenceOnly.Guarantees

open Verity Finset

/-- **The policy at a rate is a `(c, k)`-partition.** A Boolean declaration that complies at `c` gates per
inference block and rate `α` has a division into inference blocks that is a `(c, ⌊αc⌋)`-partition. -/
def CompliesCK : Prop :=
  ∀ (c : ℕ) (α : ℝ) (d : RegisteredCircuit), d.D.Boolean → CompliesRate c α d →
    ∃ (B : ℕ) (P : Partition d.D B), P.IsCK d c ⌊α * c⌋₊

/-- **The policy at a rate through the bit-level expansion.** If every computed gate is at least one bit wide and every
gate at most `W`, a declaration that complies at `c` gates per inference block and rate `α` has an expansion that
complies at `c W` gates per inference block and the same rate. -/
def ExpandComplies : Prop :=
  ∀ (c W : ℕ) (α : ℝ) (d : RegisteredCircuit), 0 ≤ α → (∀ g, d.D.isInput g = false → 1 ≤ d.D.w g) →
    (∀ g, d.D.w g ≤ W) → CompliesRate c α d → CompliesRate (c * W) α d.expand

end InferenceOnly.Guarantees
