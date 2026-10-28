import Definitions.Core.Expand

/-!
# The bit-level expansion's guarantees

A multi-bit circuit (NCI's check, the sampled-proof templates, any `w > 1` declaration) reaches the working draft's
Boolean model through `Circuit.expand`: the expansion computes the circuit bit for bit on every input, is Boolean when
every computed gate of positive width reads a gate of positive width, has as many gates as the circuit has bits, and a
partition's proof blocks keep their sizes in bits and read at most the bits they read before.
-/

namespace Verity.Guarantees

open Finset

/-- **The expansion computes the circuit, bit for bit**, on every input: the expansion's evaluation is the circuit's
evaluation on the inputs assembled gate by gate, read bit by bit. -/
def ExpandEval : Prop :=
  ∀ (C : Circuit) (a : C.expand.Transcript), C.expand.eval a = C.expandT (C.eval (C.collapse a))

/-- **A bundle read in order is the expansion's bits**: the bits of the gates `l` in `X` are the expansion's gates for
those bits, in order, in `X` read bit by bit. -/
def ExpandBits : Prop :=
  ∀ (C : Circuit) (X : C.Transcript) (l : List (Fin C.N)),
    C.expand.bits (C.expandT X) (l.flatMap fun g => List.ofFn (C.bit g)) = C.bits X l

/-- **The expansion is Boolean** when every computed gate of positive width reads a gate of positive width. -/
def ExpandBoolean : Prop :=
  ∀ C : Circuit, C.BitFaninPos → C.expand.Boolean

/-- **Sizes in bits.** The expansion's size is the width of the circuit's computed gates; a proof block of an expanded
partition has as many gates as its proof block has bits, and reads at most as many unregistered bits as its proof block
did. -/
def ExpandSize : Prop :=
  ∀ (d : RegisteredCircuit) (B : ℕ) (P : Partition d.D B),
    d.D.expand.size d.D.expand.computed = d.D.width d.D.computed ∧
      ∀ u, d.D.expand.size (P.expand.gates u) = d.D.width (P.gates u) ∧
        (d.expand.unregistered (P.expand.gates u)).card ≤ d.D.width (d.unregistered (P.gates u))

/-- **AND counts in bits.** The expansion has at most as many AND gates as the circuit's computed gates have bits, and
an expanded proof block at most as many as its proof block has bits. -/
def ExpandAndSize : Prop :=
  ∀ (d : RegisteredCircuit) (B : ℕ) (P : Partition d.D B),
    d.D.expand.andSize d.D.expand.computed ≤ d.D.width d.D.computed ∧
      ∀ u, d.D.expand.andSize (P.expand.gates u) ≤ d.D.width (P.gates u)

end Verity.Guarantees
