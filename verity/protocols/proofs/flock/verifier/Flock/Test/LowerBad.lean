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

end Flock.Test.LowerBad
