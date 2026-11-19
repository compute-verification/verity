module

public import Flock.Lower
public import Flock.Test.Lower
meta import Flock.Lower
meta import Flock.Test.Lower

/-!
`Lower.lower` on a one-gate Definition built as a `Prog`, not decoded: refused where the program is one `Program.decode`
refuses (a return shorter than its type, a reference past its source's leaves), and on the honest Definition the
template computes it (`¬a`).
-/

@[expose] public section

namespace Flock.Test.LowerBad

open Flock Flock.Program Flock.Test.Lower

def notDef : Defn :=
  { key := "Not_v1", id := "Not_v1", prim := true, params := #[("a", .value 1)], ret := .value 1, gates := 1, paramLeaves := 1, outLeaves := 1 }

def node (r : Part) : Node := { form := .prim, fn := 0, args := #[{ ty := .value 1, seq := Seq.ofParts #[r] }], out := .value 1 }

def bad (outLeaves : Nat) (r : Part) (ret : Part) : Defn :=
  { key := "Bad_v1", id := "Bad_v1", prim := false, params := #[("a", .value 1)], ret := .array outLeaves (.value 1),
    body := { params := #[("a", .value 1)], retType := .array outLeaves (.value 1), nodes := #[node r],
              ret := Seq.ofParts #[ret], offsets := #[] }, gates := 1, paramLeaves := 1, outLeaves }

def prog (b : Defn) : Prog := { defs := #[notDef, b], index := {}, root := 1, programFn := "", sha512 := ByteArray.empty }

-- a return of 1 leaf under a type of 2
#guard (Lower.lower (prog (bad 2 (.affine true 0 0 1) (.affine false 0 0 1))) 1 #[16] #[16]).toOption.isNone
-- a parameter reference past the parameters' leaves
#guard match Lower.lower (prog (bad 1 (.affine true 0 3 1) (.affine false 0 0 1))) 1 #[16] #[16] with
  | .error e => e.startsWith Lower.PAST
  | .ok _ => false
-- a node reference past its leaves
#guard match Lower.lower (prog (bad 1 (.affine true 0 0 1) (.affine false 0 1 1))) 1 #[16] #[16] with
  | .error e => e.startsWith Lower.PAST
  | .ok _ => false
-- the same Definition, honest: `¬a`
#guard match Lower.lower (prog (bad 1 (.affine true 0 0 1) (.affine false 0 0 1))) 1 #[16] #[16] with
  | .ok t => (List.range 2).all fun x => (outputs t 1 (bits x 1)).toOption == some #[x == 0]
  | .error _ => false

/-! ## Two scans `Work.scan` refuses

`SCAN_DESC`'s adder with one Definition changed after decoding, so these reach `Lower.lower` only built by hand:
`Program.decode` refuses both first. -/

/-- `SCAN_DESC`'s `ExampleScanAdd3_v1` lowered, Definition `id` changed by `f`. -/
def scanWith (id : String) (f : Defn → Defn) : V Tmpl.Tmpl := do
  let P ← Program.decode SCAN_DESC.toUTF8
  let some i := P.index[id]? | throw s!"no {id}"
  let some d := P.index["ExampleScanAdd3_v1"]? | throw "no ExampleScanAdd3_v1"
  Lower.lower { P with defs := P.defs.modify i f } d #[16] #[16]

/-- The adder's scan (its node 1) with no arguments: no init for its carry. -/
def noInit (d : Defn) : Defn :=
  { d with body := { d.body with nodes := d.body.nodes.modify 1 fun x => { x with args := #[] } } }

/-- The step's carry three leaves wide; it still returns two. -/
def wideCarry (d : Defn) : Defn :=
  let wide := fun (ps : Array (String × Ty)) => ps.modify 0 fun p => (p.1, .array 3 (.value 1))
  { d with params := wide d.params, body := { d.body with params := wide d.body.params }, paramLeaves := d.paramLeaves + 2 }

#guard match scanWith "ExampleScanAdd3_v1" noInit with
  | .error e => e.endsWith "ExampleAddStep_v1: a scan without its carry's init"
  | .ok _ => false

#guard match scanWith "ExampleAddStep_v1" wideCarry with
  | .error e => e.endsWith "ExampleAddStep_v1: a scan whose callee returns fewer leaves than its carry"
  | .ok _ => false

end Flock.Test.LowerBad
