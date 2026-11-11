module

public import Definitions.Nci

/-!
# The rate rule across a lowering: a statement

The record's check judges the rate rule on a circuit whose gates are as wide as the values they hold. A Boolean
circuit lowers such a circuit when each of its gates belongs to one gate of the other (`CircuitLowering`): the bits of
an input gate are input gates, a computed gate has at least one bit, at most a gate's width of its bits are output
bits, and a bit reads only bits of its own gate and output bits of the gates its gate reads.

`LoweredComplies` states what a lowering carries of `Nci.CompliesRate`: a block of at most `c` gates becomes one of at
most `κ * c` bits, where `κ` bounds a gate's bits; a block's fresh input bits don't grow, since an output bit a block
reads from outside belongs to a gate it read from outside, and a gate has at most its width of them; and the rate
`α` is kept, since a block has at least as many bits as gates. `Nci.Guarantees.ExpandComplies` is the case of the
bit-level expansion, where a gate's bits are its width and all of them are output bits.

The statement is not proved here, and nothing here says that a circuit file's circuit lowers a record's.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Verity

/-- **The Boolean circuit `B` lowers the circuit `D`**: gate `x` of `B` is a bit of gate `word x` of `D`, and `out`
holds the output bits, the ones a bit of another gate may read. -/
structure CircuitLowering (D B : Verity.Circuit) where
  /-- the gate of `D` that a gate of `B` is a bit of -/
  word : Fin B.N → Fin D.N
  /-- the output bits -/
  out : Finset (Fin B.N)
  /-- every gate of `B` holds one bit -/
  bit : ∀ x, B.w x = 1
  /-- the bits of an input gate are input gates, and those of a computed gate are computed -/
  input : ∀ x, B.isInput x = D.isInput (word x)
  /-- a computed gate has a bit -/
  computed : ∀ g, D.isInput g = false → ∃ x, word x = g
  /-- a gate has at most its width of output bits -/
  width : ∀ g, (out.filter fun x => word x = g).card ≤ D.w g
  /-- a bit reads bits of its own gate, and output bits of the gates its gate reads -/
  reads : ∀ x, ∀ a ∈ B.reads x, word a = word x ∨ (a ∈ out ∧ word a ∈ D.reads (word x))

/-- **The rate rule across a lowering.** Where `b`'s circuit lowers `d`'s, with θ the bits of θ's gates and at most
`κ` bits a gate, a `d` that complies at `c` gates and rate `α` gives a `b` that complies at `κ * c` gates and the same
rate. -/
def LoweredComplies : Prop :=
  ∀ (c κ : ℕ) (α : ℝ) (d b : Verity.RegisteredCircuit) (lw : CircuitLowering d.D b.D), 0 ≤ α →
    (∀ x, x ∈ b.θ ↔ lw.word x ∈ d.θ) →
    (∀ g, (Finset.univ.filter fun x => lw.word x = g).card ≤ κ) →
    Nci.CompliesRate c α d → Nci.CompliesRate (κ * c) α b

end NciCheck.Properties.Record.SecurityProofs
