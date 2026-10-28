/-
`KsScope`: the `--zk` statement passes `tagsWhy`, and a typed statement or one of public outputs is refused. A unit of one
input group that returns the outputs passes `filesWhy`; a unit reading two input groups, a tail stage returning the
outputs (both in `scopeOkI`, which `ProvedScope.check` admits under hidden outputs) and a shared-row public file are
refused. At one table any draw passes `drawWhy`; at more, a subset draw whose `k` the tables divide passes, and a draw of
`k` they don't divide, a draw that does not parse and no draw are refused.
-/
import Flock.KsScope
import Flock.Test.Wires

namespace Flock.Test.KsScope

open Lean (Json)
open Flock Flock.KsScope

#guard tagsWhy Tags.circuit == none
#guard (tagsWhy Tags.circuitTypes).isSome
#guard (tagsWhy Tags.circuitEb90718f).isSome
#guard (tagsWhy { Tags.circuit with retainRounds := false }).isSome
#guard (tagsWhy { Tags.circuit with merkleLeaf := "flock-leaf/sha512-unsalted" }).isSome

/-- A unit of one input group, returning the outputs. -/
def one : Circuit := Wires.circ #[Wires.net #[⟨0, 1, #[1]⟩] 1 #[1] #[(#[0], #[0])]] #[1] #[]

#guard ProvedScope.scopeOkH one
#guard filesWhy one default == none
#guard ProvedScope.scopeOkI (Wires.rms 0) && filesWhy (Wires.rms 0) default == some SCOPE_WHY
#guard ProvedScope.scopeOkI Wires.tail && filesWhy Wires.tail default == some SCOPE_WHY
#guard filesWhy one { (default : Public) with tables := some default } == some TABLES_WHY

def n (x : Nat) : Json := Lean.toJson x

def subset (k : Nat) : Json := Json.mkObj [("law", "subset"), ("population", n 8), ("k", n k),
  ("units", Json.arr ((Array.range k).map n))]

#guard drawWhy 1 none == none && drawWhy 1 (some (subset 3)) == none
#guard drawWhy 2 (some (subset 4)) == none && drawWhy 4 (some (subset 4)) == none
#guard (drawWhy 2 (some (subset 3))).isSome && (drawWhy 4 (some (subset 6))).isSome
#guard (drawWhy 2 (some (Json.mkObj [("law", "subset")]))).isSome
#guard (drawWhy 2 none).isSome

/-- `checkOf`'s refusal of `c` under `tags`, at an empty public file, which no statement loads. -/
def why (tags : Tags) (c : Circuit) (count : Nat) (draw : Option Json) : Option String :=
  match checkOf (.ok c) tags ByteArray.empty count draw with
  | .ok () => none
  | .error e => some e

#guard why Tags.circuitTypes one 1 none == some (ProvedScope.refusal (tagsWhy Tags.circuitTypes).get!)
#guard (why Tags.circuit one 1 none).isSome

end Flock.Test.KsScope
