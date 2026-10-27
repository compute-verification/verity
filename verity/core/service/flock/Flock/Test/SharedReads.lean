/-
`HmIn.checkReads` (META `shared_reads`, PROTOCOL.md §16.10): a shared read passes only when instance `i` reads table row
`⌊i/G⌋` of a table of one row per group, the instances are whole groups, and its position is the instance index mod `G`
(seed `instance`, derivation `mod:G`): one writer a window. Four instances, an input row `x` and an output row `out`.
-/
import Flock.HmRow

namespace Flock.Test.SharedReads

open Lean (Json toJson)

def circuit (reads : Json) : Circuit :=
  { (default : Circuit) with
    ports := #[{ name := "x", words := 64, chunks := 1, nb := 1 }, { name := "out", words := 64, chunks := 1, nb := 1 }],
    outRowPorts := #[0], metaJ := Json.mkObj [("shared_reads", reads)] }

def ports (g : Nat) (seed : String := "instance") : HmIn.Ports :=
  { json := Json.arr #[Json.mkObj [("port", Json.str "j"), ("words", toJson 1), ("seed", Json.str seed),
      ("derivation", Json.str s!"mod:{g}")]], ports := #[("j", 1)] }

def read (port : String) (g : Nat) : Json :=
  Json.mkObj [("port", Json.str port), ("group", toJson g), ("position", Json.str "j")]

/-- `x`'s table of `rx` rows and each instance's ref into it; `out`'s refs are the instances' own. -/
def tables (rx : Nat) (xs : Array Nat) : Tables :=
  { counts := #[rx, 4], starts := #[0, rx], refs := (Array.range 4).foldl (fun a i => (a.push (xs.getD i 0)).push i) #[] }

def why (reads : Array Json) (g : Nat) (t : Option Tables) (pi : HmIn.Ports := ports g) : String :=
  match HmIn.checkReads (circuit (Json.arr reads)) pi 4 t with
  | .error e => e
  | .ok () => ""

#guard why #[read "x" 2] 2 (some (tables 2 #[0, 0, 1, 1])) == ""
#guard why #[read "x" 1] 1 (some (tables 4 #[0, 1, 2, 3])) == ""
#guard why #[read "x" 4] 4 (some (tables 1 #[0, 0, 0, 0])) == ""
-- without public input ports there is no read to check (`HmIn.parse` refuses the key there)
#guard (HmIn.checkReads (circuit (Json.arr #[read "x" 2])) {} 4 none).toBool
#guard (HmIn.checkReads { circuit Json.null with metaJ := Json.mkObj [] } (ports 2) 4 none).toBool

#guard why #[read "x" 2] 2 (some (tables 2 #[0, 1, 0, 1])) == "shared read x: instance 1 reads table row 1, not its group's 0"
#guard why #[read "x" 2] 2 (some (tables 2 #[0, 0, 1, 0])) == "shared read x: instance 3 reads table row 0, not its group's 1"
#guard why #[read "x" 2] 2 (some (tables 4 #[0, 0, 1, 1])) ==
  "shared read x: its table is not one row per group of 2 (2 rows)"
#guard why #[read "x" 3] 3 (some (tables 2 #[0, 0, 0, 1])) == "shared read x: 4 instances are not whole groups of 3"
#guard why #[read "x" 2] 4 (some (tables 2 #[0, 0, 1, 1])) ==
  "shared read x: its position j is not the instance index mod 2 (seed instance, derivation mod:2)"
#guard why #[read "x" 2] 2 (some (tables 2 #[0, 0, 1, 1])) (ports 2 "rope/positions") ==
  "shared read x: its position j is not the instance index mod 2 (seed instance, derivation mod:2)"
#guard why #[read "out" 1] 1 (some (tables 4 #[0, 1, 2, 3])) == "shared read out: not an input row"
#guard why #[read "x" 1, read "x" 1] 1 (some (tables 4 #[0, 1, 2, 3])) == "shared read x: named twice"
#guard why #[read "x" 0] 1 (some (tables 4 #[0, 1, 2, 3])) ==
  "META: a shared read names no port, position, or group (a positive count of instances)"
#guard why #[read "x" 1] 1 none == "shared reads: the public file is not a shared-row file (header shared_rows)"
#guard why #[] 1 none == "META: shared_reads names no port"

end Flock.Test.SharedReads
