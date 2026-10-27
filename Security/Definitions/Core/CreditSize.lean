import Mathlib.Data.ZMod.Defs
import Definitions.Core.Declaration

/-!
# Credit in AND gates

Credit and every reported count are AND counts (Daniel, 26 Oct, tentative; LEAN-63). The all-gate count
`Circuit.size` stays for partitions and counting.

* An **AND gate** (`Circuit.IsAnd`) is an operation gate whose function of the gates it reads is not affine over
  GF(2) (`Circuit.Affine`): on a Boolean circuit, a gate outside the XOR/NOT basis C-Flock proves for free.
* The **AND count** `C.andSize S` (`Circuit.andSize`) is the number of AND gates among `S`.
* **Credit** (`creditSize lower d`) is the AND count of the registered declared circuit under one public lowering,
  pinned at registration.

These are `Definitions.Core.Formalism`'s definitions on lean's ONEFORM branch, name for name and word for word, so
that file replaces this one.
-/

namespace Verity

open Finset

namespace Circuit

/-- A value's least significant bit, in `ZMod 2`. -/
def bitZ {n : ℕ} (x : BitVec n) : ZMod 2 := if x.getLsbD 0 then 1 else 0

/-- Gate `g` computes an **affine** function over GF(2) of the gates it reads: its bit is a constant plus the XOR of the
bits of some of them. On a Boolean circuit these are the gates of an XOR/NOT basis, which C-Flock proves for free. -/
def Affine (C : Circuit) (g : Fin C.N) : Prop :=
  ∃ (S : Finset (Fin C.N)) (c : ZMod 2), S ⊆ C.reads g ∧ ∀ v, bitZ (C.op g v) = c + ∑ a ∈ S, bitZ (v a)

/-- **An AND gate** of a Boolean circuit: an operation gate whose function of the gates it reads is not affine over
GF(2). `Verity.Circuit`'s gate functions carry no tag, so AND is this predicate on the function. C-Flock's `and a b`
(`a ≠ b`) and its rows `(⊕ a) ∧ (⊕ b)` whose two XORs differ are AND gates, and its XOR, NOT and input gates are not
(`Verity.Guarantees.ToCircuitAnd`). -/
def IsAnd (C : Circuit) (g : Fin C.N) : Prop := C.isInput g = false ∧ ¬ C.Affine g

open Classical in
/-- **The AND count** of the gates `S`, the number of AND gates among them: on `C.computed`, the circuit's
multiplicative complexity, the gates C-Flock pays for. Credit (`creditSize`) and every reported count are stated in
`andSize`. Exclusion results are not converted to ANDs: they are reported as `N` counted operations (FP8 MACs, say),
each labelled with that operation's canonical AND count, since converting them to ANDs of hidden work would need a
ceiling on ANDs per resource unit over all programs, which is vacuous. All-gate counts (the `(c, k)`-partition, the
draft's `Φ`, `c_S`, `size`) stay in `size`, linked to `andSize` by `andSize_le_size`. -/
noncomputable def andSize (C : Circuit) (S : Finset (Fin C.N)) : ℕ := (S.filter fun g => C.IsAnd g).card

end Circuit

/-- **Credit**: the AND count of the registered declared circuit `d` under the one public lowering `lower`, pinned at
registration, never the prover's own lowering. Each credit record carries `lower`'s digest. -/
noncomputable def creditSize (lower : Declaration → Circuit) (d : Declaration) : ℕ :=
  (lower d).andSize Finset.univ

end Verity
