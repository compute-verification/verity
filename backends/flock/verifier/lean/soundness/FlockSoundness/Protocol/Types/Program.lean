import FlockSoundness.Protocol.E2E
import Flock.Derive
import Flock.DeriveCheck

/-!
# A program of units over circuit types: its rows circuit, its Boolean circuit, and `RowsL1` between them (phase 1, S4c)

A program (`UProg`) is `k` inputs, input `one` the constant, then units in order. A unit (`UnitSpec`) is the verifier's
derivation for its layout, `deriveChecked` on a type `t`. Each input bit of a unit reads a
source (`Src`): a program input other than the constant, or an earlier unit's output. Several bits of a unit may read one
source. The statement's zero, which its forced-zero rows carry, is a program input like `one`, a constant of the
statement and no unit's gate: a zero leaf's bits, and a leaf cut's bits past its leaf, read it. From it:
- `toRows`, its rows circuit `C` (a `Lowering.Prog`): each unit's composed rows, its input columns on their sources' output
  rows, its constant on `one`;
- `CB`, its Boolean circuit: each unit's unit gates (`Types.Flatten.unitGates`), its inputs on their sources' copies, its
  `one` on `one`;
- `mapG`, reading each input, `one` and copy of `CB` off `C`: an input off the same input, a copy off its output's row.

Every unit output is an output of both circuits. `pairs` gives one `UnitPair` per unit, so `rowsL1`: at values with 1 on
`one`, every unit wrong on the Boolean circuit is wrong on its rows. It carries `E2E.lean`'s theorems, which are about the
rows circuit, to the Boolean circuit (`UProg.flock_e2e_count`, `Types/ProgramE2E.lean`).
-/

open Flock Flock.DeriveAll

namespace FlockSoundness

open Audit Types.Flatten Compose Audit.Partition

/-! ## `flatten`'s input gates are its first `inBits + 1` -/

namespace Types.Flatten

variable (types : String → Option CircuitType) (words : String → Option (Array ℕ)) (loOf : ℕ → ℕ) (F : ℕ)
  (t : CircuitType)

end Types.Flatten

/-! ## A unit, and its output columns -/

theorem getD_of_lt {a : Array ℕ} {i : ℕ} (h : i < a.size) : a.getD i 0 = a[i] := by
  simp [Array.getD, h]

namespace UnitSpec

variable (s : UnitSpec)

end UnitSpec

/-! ## The program -/

/-- Where a unit's input bit reads: program input `a`, or output `o` of unit `v`. -/
inductive Src
  | input (a : ℕ)
  | out (v o : ℕ)

namespace UProg

variable {k one : ℕ}

/-! ### The rows circuit -/

/-! ### The Boolean circuit -/

/-! ### Reading the Boolean circuit off the rows circuit -/

/-! ### Gates and units, old and new -/

/-! ### Each unit's instances -/

/-! ### Committed gates -/

/-! ### One pair per unit -/

end UProg

end FlockSoundness
