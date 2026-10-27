import Mathlib.Data.ZMod.Defs
import Definitions.Core.Declaration

/-!
# Credit in AND gates

Credit and every reported amount of computation are sizes in AND gates (Daniel, 26 Oct, tentative; the Glossary's
Size and Credit). The all-gate count `Circuit.size` stays for partitions and counting.

* An **AND gate** (`Circuit.IsAnd`) is an operation gate whose function of the gates it reads is not affine over
  GF(2) (`Circuit.Affine`): on a Boolean circuit, a gate outside the XOR/NOT basis C-Flock proves for free.
* The **AND count** `C.andSize S` (`Circuit.andSize`) is the number of AND gates among `S`.
* A **lowering** (`Lowering`) is the Boolean circuit a declaration is credited as: one public function of the
  registered declaration, pinned in the registration, never the prover's own lowering.
* The **credit size** `creditSize lower D` is the AND count of `lower D`, the registered declaration `D` under the
  lowering `lower`. It is the one definition of what credit counts, so a change of ruling changes it alone.
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
(`a ≠ b`) and its rows `(⊕ a) ∧ (⊕ b)` whose two XORs differ are AND gates, and its XOR, NOT and input gates are
not. -/
def IsAnd (C : Circuit) (g : Fin C.N) : Prop := C.isInput g = false ∧ ¬ C.Affine g

open Classical in
/-- **The AND count** of the gates `S`, the number of AND gates among them: on `C.computed`, the circuit's
multiplicative complexity, the gates C-Flock pays for. Credit and every reported count are stated in `andSize`. -/
noncomputable def andSize (C : Circuit) (S : Finset (Fin C.N)) : ℕ := (S.filter fun g => C.IsAnd g).card

end Circuit

/-- **A lowering**: the Boolean circuit a declaration is credited as. It is one public function of the registered
declaration, pinned in the registration, never the prover's own lowering. -/
abbrev Lowering : Type := Declaration → Circuit

/-- **The credit size** of gates of `lower D`: their AND count in the registered declaration `D` under the lowering
`lower` pinned at registration. Credit counts it. -/
noncomputable def creditSize (lower : Lowering) (D : Declaration) : Finset (Fin (lower D).N) → ℕ := (lower D).andSize

end Verity
