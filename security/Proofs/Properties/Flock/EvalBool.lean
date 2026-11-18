module

public import Properties.Flock.EvalBool
public import Flock.Test.Lower
meta import Properties.Flock.EvalBool
meta import Flock.Test.Lower

/-!
# `Prog.evalBool` against the vectors

`Prog.evalBool` (`Properties.Flock.EvalBool`) on `Flock.Test.Lower`'s two Boolean programs: PROTOCOL.md §9's
`ExampleAdd2_v1` (a batch over an axis, two calls, a constant, a `Not_v1`, an `Xor3_v1`) gives the vectors' word
`x + y + 8` on each of its 16 inputs, and the ripple adder `ExampleScanAdd3_v1` (a scan) gives `x + y` on each of its 64.
On every one of those inputs it agrees with the output rows of the template `Lower.lower` writes, unrolled and solved.
-/

@[expose] public section

namespace FlockSoundness.EvalBool

open Flock Flock.Test.Lower

/-- Definition `id`'s value on the bits of `x` and `y`, `n` bits each, as a word; `none` if `P` has no `id`. -/
def evalWord (P : Program.Prog) (id : String) (n x y : Nat) : Option Nat :=
  P.index[id]?.map fun d => word (P.evalBool d ((bits x n ++ bits y n).toList)).toArray

#guard match prog with
  | .ok P => WORDS.all fun (x, y, o) => evalWord P "ExampleAdd2_v1" 2 x y == some o
  | .error _ => false

#guard match Program.decode SCAN_DESC.toUTF8 with
  | .ok P => (List.range 8).all fun x => (List.range 8).all fun y => evalWord P "ExampleScanAdd3_v1" 3 x y == some (x + y)
  | .error _ => false

-- the evaluator and the lowered template's output rows agree, bit for bit
#guard match prog, tmpl with
  | .ok P, .ok t => (List.range 4).all fun x => (List.range 4).all fun y =>
      match outputs t 4 (bits x 2 ++ bits y 2), P.index["ExampleAdd2_v1"]? with
      | .ok bs, some d => P.evalBool d (bits x 2 ++ bits y 2).toList == bs.toList
      | _, _ => false
  | _, _ => false

#guard match Program.decode SCAN_DESC.toUTF8, scanTmpl with
  | .ok P, .ok t => (List.range 8).all fun x => (List.range 8).all fun y =>
      match outputs t 4 (bits x 3 ++ bits y 3), P.index["ExampleScanAdd3_v1"]? with
      | .ok bs, some d => P.evalBool d (bits x 3 ++ bits y 3).toList == bs.toList
      | _, _ => false
  | _, _ => false

end FlockSoundness.EvalBool
