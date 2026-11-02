/-
`Registered.checkComplete`, which `flock-verify` runs whenever `--registered` is given: a lane statement whose index row
`pub` its public file states, committed afresh (no registered read), passes `Registered.check` on its other ports and is
refused by `checkComplete`; with `pub` among the verifier's ports, it passes.
-/
import Flock.Test.RegFormats

namespace Flock.Test.RegComplete

open Lean (Json)
open Flock.Test.RegFormats (dom bcs root v2 v2Port)

def pubPort : Port := { name := "pub", words := 1, chunks := 1, nb := 0, bits := 16, v2 := true }

/-- The lane's circuit: the registered port `w` and its index row `pub`. -/
def lane : Circuit := { (default : Circuit) with ports := #[v2Port, pubPort] }

/-- Two instances reading `w`'s table rows 0 and 1 and `pub`'s one row, committed afresh: the header registers `w` only. -/
def pub : Public :=
  let leaves := bcs.mapIdx fun i bc => Registered.leafAs v2 dom i (Hm96.Default512.treeLeaf bc)
  { header := Json.mkObj [("units", Json.mkObj [("indices", Lean.toJson [0, 1])]),
      ("registered", Json.mkObj [("w", Json.mkObj [("value", Json.str "weights"), ("positions", Lean.toJson [0, 1]),
        ("paths", Json.arr #[Json.str (hexOf leaves[1]!), Json.str (hexOf leaves[0]!)])])])],
    n := 2, digests := bcs.push (ByteArray.mk (Array.replicate 32 4)), outputs := #[], sha := .empty,
    tables := some { counts := #[2, 1], starts := #[0, 2], refs := #[0, 0, 1, 0] } }

/-- `pub` for the circuit of `w` alone: its table's refs, one per instance (a ref is `refs[i * ports + p]`). -/
def pubW : Public := { pub with digests := bcs, tables := some { counts := #[2], starts := #[0], refs := #[0, 1] } }

def wEntry : Json := Json.mkObj [("value", Json.str "weights"), ("root", Json.str (hexOf (root v2))),
  ("leaves", Lean.toJson (2 : Nat)), ("domain", Json.str (hexOf dom)), ("reads", Lean.toJson [some 0, some 1]),
  ("words", Lean.toJson (32 : Nat)), ("row_schema", Json.str v2)]

def file : Json := Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [("w", wEntry)])]

/-- `flock-verify`'s registered reads of `f`: `Registered.check`, then `checkComplete`. -/
def run (c : Circuit) (f : Json) (p : Public := pub) : Except String Unit := do
  let own ← Registered.ofJson f
  Registered.check c p (some own) true
  Registered.checkComplete c own

def why (c : Circuit) (f : Json) : String :=
  match run c f with
  | .error e => e
  | .ok () => ""

-- the verifier's reads hold `w` and say nothing of `pub`: `check` passes the index row stated by the public file
#guard (Registered.ofJson file >>= fun own => Registered.check lane pub (some own) true).toBool

-- and `checkComplete` refuses it by name
#guard !(run lane file).toBool
#guard why lane file ==
  "registered reads (complete): the statement's port pub is no port of the verifier's registered reads, which name every port the statement's outputs read: its rows would be committed afresh, stated by the public file"

-- a circuit whose every port is the verifier's passes
#guard (Registered.checkComplete lane #[{ (default : Registered.Port) with name := "w" },
  { (default : Registered.Port) with name := "pub" }]).toBool
#guard (run { lane with ports := #[v2Port] } file pubW).toBool
#guard Registered.complete { lane with ports := #[v2Port] } #[{ (default : Registered.Port) with name := "w" }]
#guard !Registered.complete lane #[{ (default : Registered.Port) with name := "w" }]

-- a file of this format names no `complete`: the check is not the file's to switch off
#guard !(Registered.ofJson (Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [("w", wEntry)]),
  ("complete", Json.bool false)])).toBool

end Flock.Test.RegComplete
