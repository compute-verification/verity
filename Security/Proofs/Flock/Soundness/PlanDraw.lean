import Randomness

/-!
# The PoUW window draw, as a Lean spec

`protocols/pouw/verity_pouw/circuit/plan.py`'s `draw` (#364), which `Traced.draw` calls for each call of a window under
the window's one key, and the window around it:

* `context call name`: `plan.draw`'s context for a stratum, `("call", *call, "stratum", name)`, or `("stratum", name)`
  without a call index.
* `drawn key call strata L`: the units `plan.draw` draws for one call, before sorting: each stratum's
  `Key.subset(k, len(units), context)`, mapped through its units, each swap's stream read to `L` bytes.
* `window key calls L`: every call admitted, no call index framed twice (X-SPC-107's admission inside the draw), then each
  call's `drawn` under the one key, its units in the window's numbering.
-/

namespace FlockSoundness.PlanDraw

open Randomness

/-- A layout stratum as `plan.draw` reads it: its template name, its units in order, and the size its law gives it. -/
structure Stratum where
  name : String
  units : List Nat
  k : Nat

/-- `plan.draw`'s context for a stratum. -/
def context (call : List Part) (name : String) : List Part :=
  (if call.isEmpty then [] else .str "call" :: call) ++ [.str "stratum", .str name]

/-- The units `plan.draw` draws for one call, before sorting. -/
def drawn (key : ByteArray) (call : List Part) : List Stratum → Nat → Option (List Nat)
  | [], _ => some []
  | s :: rest, L => do
    let loc ← subset key s.k s.units.length (context call s.name) L
    let r ← drawn key call rest L
    pure (loc.toList.map (fun i => s.units.getD i 0) ++ r)

/-- A window call: its index, `(forward, position)`, and its layout's strata, units in the window's numbering. -/
structure Call where
  index : List Part
  strata : List Stratum

/-- The calls' draws under the one key, in order. -/
def drawAll (key : ByteArray) : List Call → Nat → Option (List Nat)
  | [], _ => some []
  | c :: cs, L => do
    let d ← drawn key c.index c.strata L
    let r ← drawAll key cs L
    pure (d ++ r)

/-- **The window draw**: refused if a call index repeats, else every call's draw under the one key. -/
def window (key : ByteArray) (calls : List Call) (L : Nat) : Option (List Nat) :=
  if (calls.map fun c => frames c.index).Nodup then drawAll key calls L else none

end FlockSoundness.PlanDraw
