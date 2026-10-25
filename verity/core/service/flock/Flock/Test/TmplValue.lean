/-
`Tmpl.evalDef` against the unrolled rows (`Flock.TmplValue.rows_eval` on one unit): `Flock.Test.Tmpl`'s unit, its column
values computed in order from the rows `Tmpl.rows` unrolls, its three input bits each way and the constant 1, satisfies
every row, and at each copy of a pair (at 128 and 131) and of an AND inside it, those values are the copy's Definition's
`evalDef` on its bound inputs. And `evalDef` on its own: the AND, a pair, and an XOR of two inputs.
-/
import Flock.Test.Tmpl
import Flock.TmplValue

namespace Flock.Test.TmplValue

open Flock Flock.Test.Tmpl

def tmpl : _root_.Flock.Tmpl.Tmpl :=
  match _root_.Flock.Tmpl.parse tmplLines with
  | .ok t => t
  | .error _ => default

def rows : _root_.Flock.Tmpl.Acc :=
  match tmpl.rows with
  | .ok acc => acc
  | .error _ => _root_.Flock.Tmpl.Acc.empty 0

/-- The unit's column values: input bit `x` from `ins`, the constant 1, every other column its row's value on the ones
before it. -/
def honest (ins : Nat → Bool) : Array Bool :=
  (List.range 400).foldl (fun arr x =>
    let z := fun y => if y = 399 then true else arr.getD y false
    arr.push (if x = 399 then true else if x < 3 then ins x
      else _root_.Flock.Tmpl.rowVal z (x, rows.ra.getD x #[], rows.rb.getD x #[]))) #[]

def val (ins : Nat → Bool) (y : Nat) : Bool := if y = 399 then true else (honest ins).getD y false

def inputs : List (Nat → Bool) :=
  (List.range 8).map fun m i => decide (m / 2 ^ i % 2 = 1)

/-- Every unrolled row holds at `val ins`. -/
def holds (ins : Nat → Bool) : Bool :=
  (List.range 400).all fun x =>
    val ins x == _root_.Flock.Tmpl.rowVal (val ins) (x, rows.ra.getD x #[], rows.rb.getD x #[])

/-- At each pair, copy `j` at `128 + 3j` bound to input bits `(j, j + 1)`, and at each AND inside it, the values are
`evalDef`'s. -/
def composes (ins : Nat → Bool) : Bool :=
  (List.range 2).all fun j =>
    let inp := [val ins j, val ins (j + 1)]
    ((List.range 3).all fun y => val ins (128 + 3 * j + y) == tmpl.evalDef 1 inp true y) &&
      (List.range 2).all fun i => val ins (128 + 3 * j + i) == tmpl.evalDef 0 inp true 0

#guard (_root_.Flock.Tmpl.parse tmplLines).toOption.isSome
#guard tmpl.rows.toOption.isSome
#guard inputs.all holds
#guard inputs.all composes
-- the AND: own column 0 is `in 1 ∧ in 0`
#guard [[false, false], [false, true], [true, false], [true, true]].map (tmpl.evalDef 0 · true 0) ==
  [false, false, false, true]
-- a pair: both ANDs, and their sum (zero) times the constant
#guard (List.range 3).map (tmpl.evalDef 1 [true, true] true) == [true, true, false]
#guard (List.range 3).map (tmpl.evalDef 1 [true, false] true) == [false, false, false]
-- an XOR: own column 0 is `(in 0 ⊕ in 1) ∧ const`, so the constant's value enters
#guard [[false, false], [false, true], [true, false], [true, true]].map
    (_root_.Flock.Tmpl.eval #[{ cols := 1, ins := 2, rows := #[(0, #[1, 2], #[3])], calls := #[] }] true 1 0 · 0) ==
  [false, true, true, false]
#guard _root_.Flock.Tmpl.eval #[{ cols := 1, ins := 2, rows := #[(0, #[1, 2], #[3])], calls := #[] }] false 1 0
  [true, false] 0 == false

end Flock.Test.TmplValue
