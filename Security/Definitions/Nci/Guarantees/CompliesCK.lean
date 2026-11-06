module

public import Definitions.Core.Expand
public import Definitions.Nci

/-!
# The policy at a rate as a `(c, k)`-partition

`CompliesRate c α` asks for inference blocks of at most `c` gates, each reading at most `α` fresh input bits per gate.
On a Boolean declaration a fresh input bit is one input gate outside `θ`, so an inference block reads at most
`α |I| ≤ α c` of them, and the division is a `(c, ⌊αc⌋)`-partition (`CompliesCK`), which complies at `(c, ⌊αc⌋)` and
on which the working draft's Lemma 1 and Theorem 3 apply. A multi-bit declaration first takes its bit-level expansion
(`ExpandComplies`): an inference block of gates at most `W` bits wide becomes one of at most `c W` one-bit gates,
reading the same fresh input bits.
-/

@[expose] public section

namespace Nci.Guarantees

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

end Nci.Guarantees
