import Definitions.Core.Advice

/-!
# Private partitions with public predicates

A private partition is the prover's advice, registered under a hidden root before any coin that depends on it: its rows,
under salts of the prover's choosing, in the advice tree (`Advice`). The verifier never sees it: it holds the public
predicates `Q` and the root, and the registration proves that the circuit has a partition meeting `Q` whose advice has
that root (`Registered`). This is `verity/partition/v2`; `verity/partition/v1`'s public partition is unchanged.

`Partition`'s type already gives covering and disjointness: every computed gate is in exactly one unit, and inputs are in
none. The predicates add the rest of the draft's global ones:
* no unit is empty (G2);
* the width rule (G4, `Fits`): unit `u`'s outputs, out(R) for its gates `R` (`outs`), total at most `X` bits, with no
  single-value exception (`from_v1` maps Q_word{X, W} to `X`);
* the count (G6): `n` units, at most the padded count `Q.units`;
* the caps (G7): each unit has at most `Q.maxGates` gates and reads at most `Q.maxInBits` bits from outside.

The width rule holds for every unit, wrong or not, which is what bounds the bits that wrong units can leak.

The registration checks the width rule once per Definition, not per unit. Its advice is a Call table, the circuit's
computed gates grouped into root Calls with each Call's returned gates (`Calls`), and per Definition a cut, an owner per
gate of the Call. Unit `u`'s outputs inside its Call (`localOuts`) are its gates that the Call returns, or that a gate of
another unit of the same Call reads. They depend only on the Call's own graph, so all Calls of one Definition share them,
and every global output of `u` is a local one.
-/

namespace Verity

open Finset

/-- The public predicates of a private partition: what the verifier holds besides the root. -/
structure Predicates where
  /-- The bound on a unit's output bits: the predicates' `width`. -/
  X : ℕ
  /-- The padded unit count, a level of a public grid. -/
  units : ℕ
  /-- The most gates a unit may have. -/
  maxGates : ℕ
  /-- The most bits a unit may read from outside its gates. -/
  maxInBits : ℕ

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- Unit `u`'s outputs: out(R) for its gates `R`. -/
def outs (u : Fin n) : Finset (Fin C.N) := C.exits (P.gates u)

/-- `ob(u)`: the total width of unit `u`'s outputs, in bits. -/
def outBits (u : Fin n) : ℕ := C.width (P.outs u)

/-- `og(u)`: the number of unit `u`'s outputs. -/
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

/-- `r` registers `root` under `Q`: its partition meets `Q`, and the advice tree over the partition's rows (`A`) under
`r`'s salts, framed by `F` and hashed by `H`, has root `root`. The registration's statement is that some `r` does; the
verifier holds `Q` and `root`, never `r`. -/
def Registered {C : Circuit} (H : List UInt8 → Digest) (F : Advice.Framing) (A : Advice.Rows C) (Q : Predicates)
    (r : Advice.Registration C) (root : Digest) : Prop :=
  r.P.Meets Q ∧ r.root H F A = root

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
