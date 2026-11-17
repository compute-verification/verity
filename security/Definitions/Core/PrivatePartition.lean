module

public import Definitions.Core.Advice
public import Definitions.Core.RegisteredCircuit

/-! Private partitions with public predicates (`verity/partition/v2`), which the verifier checks against the advice
root without seeing the partition. -/

@[expose] public section

namespace Verity

open Finset

/-- The public predicates of a private partition: what the verifier holds besides the root. -/
structure Predicates where
  /-- The bound on a unit's output bits: the predicates' `width`. -/
  X : ℕ
  /-- The padded unit count, a level of a public grid. -/
  units : ℕ
  maxGates : ℕ
  /-- The most bits a unit may read from outside its gates. -/
  maxInBits : ℕ

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- Unit `u`'s outputs: out(R) for its gates `R`. -/
def outs (u : Fin n) : Finset (Fin C.N) := C.exits (P.gates u)

/-- `ob(u)`: the total width of unit `u`'s outputs, in bits. -/
def outBits (u : Fin n) : ℕ := C.width (P.outs u)

def outCount (u : Fin n) : ℕ := (P.outs u).card

/-- The width rule for a set of outputs `S`: at most `X` bits. -/
def WidthRule (X : ℕ) (S : Finset (Fin C.N)) : Prop := C.width S ≤ X

/-- Every unit's outputs meet the width rule (G4). -/
def Fits (X : ℕ) : Prop := ∀ u, WidthRule (C := C) X (P.outs u)

/-- `P` meets the public predicates `Q`: no unit is empty, every unit fits, the unit count is at most the padded one,
and every unit is within the caps. -/
def Meets (Q : Predicates) : Prop :=
  (∀ u, (P.gates u).Nonempty) ∧ P.Fits Q.X ∧ n ≤ Q.units ∧
    ∀ u, (P.gates u).card ≤ Q.maxGates ∧ C.width (C.into (P.gates u)) ≤ Q.maxInBits

end Partition

/-- `r` commits to `root` under `Q`: the proof blocks the registered compiler `B` outputs on the registered values `θ`
and `r`'s advice meet `Q`, and the advice tree over the advice's rows (`A`) under `r`'s salts, framed by `F` and hashed
by `H`, has root `root`. The commitment's statement is that some `r` does; the verifier holds `B`, `θ`, `Q` and `root`,
never `r`. -/
def Commits (B : Compiler) (θ : List Registration.Theta) (H : List UInt8 → Digest) (F : Advice.Framing)
    (A : Advice.Rows) (Q : Predicates) (r : Advice.Committed) (root : Digest) : Prop :=
  (B.run θ r.a).P.Meets Q ∧ r.root H F A = root

/-- The circuit's computed gates grouped into its root Calls `0, …, m − 1`, with the gates the Calls return. A value read
across Calls, and every output of the circuit, is an input or a returned gate. -/
structure Calls (C : Circuit) (m : ℕ) extends Partition C m where
  returned : Finset (Fin C.N)
  cross : ∀ g h, g ∈ C.reads h → unit h ≠ unit g → C.isInput g = true ∨ g ∈ returned
  out : ∀ g ∈ C.outputs, C.isInput g = true ∨ g ∈ returned

namespace Partition

variable {C : Circuit} {n m : ℕ} (P : Partition C n)

/-- Unit `u`'s outputs inside its Call: its gates that the Calls return, or that a gate of another unit of the same Call
reads. These depend only on the Call's own graph and its cut. -/
def localOuts (K : Calls C m) (u : Fin n) : Finset (Fin C.N) :=
  (P.gates u).filter fun g =>
    g ∈ K.returned ∨ ∃ h, g ∈ C.reads h ∧ K.unit h = K.unit g ∧ P.unit h ≠ P.unit g

/-- Every unit's outputs inside its Call meet the width rule: the check made once per Definition. -/
def FitsLocally (K : Calls C m) (X : ℕ) : Prop := ∀ u, WidthRule (C := C) X (P.localOuts K u)

end Partition

end Verity
