module

public import Spot.OneStageZkStatement

/-!
# One-stage v2: the members named once, and a wire into a member

Between `Spot/OneStageZkStatement.lean` and `Spot/OneStageZk.lean`, in the namespace `Spot.OneStageZk`: the members
naming each name once (`namesOnce`, at `portTwice`, which `Member.portsOnce` reads too), and whether a wire of C's
wiring enters a member (`wireReaches`, at each range's base among its template's units, `localBases`), which
`fieldRoots` reads.
-/

@[expose] public section

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

/-- The first port of `ps` a later one names again, the two compared as strings; none where each is named once. -/
def portTwice : List String → Option String
  | [] => none
  | p :: ps => if p ∈ ps then some p else portTwice ps

/-- The refusal of members of which two are named `name` (`membersStructure`): the members' roots, slices and
statements are each keyed by a member's name. -/
def NAME_TWICE_WHY (name : String) : String :=
  s!"proofs: two members are named {name}: a member's roots, slice and statement are each read by its name"

/-- **The members `ms` name each name once**, compared as strings (`portTwice`); two of one name refuse by it
(`NAME_TWICE_WHY`). -/
def namesOnce (ms : Array Member) : V Unit :=
  match portTwice (ms.toList.map (·.name)) with
  | some n => throw (NAME_TWICE_WHY n)
  | none => pure ()

/-- Each range of `rs` (template, first unit, count) at its base among its template's own units: the counts of its
template's earlier ranges. -/
def localBases (rs : Array (String × Nat × Nat)) : Array Nat :=
  (rs.foldl (fun (acc : Array Nat × Std.HashMap String Nat) (t, _, c) =>
    (acc.1.push (acc.2.getD t 0), acc.2.insert t (acc.2.getD t 0 + c))) (#[], {})).1

/-- **The wire `w` enters the member `m`** over the ranges `rs` at their bases `lbase` (`localBases`): a range of
`w`'s reader holds `m`'s units, and `w`'s unit window, in its template's own units, meets `m`'s there. -/
def wireReaches (rs : Array (String × Nat × Nat)) (lbase : Array Nat) (m : Member) (w : Spot.Wires.Wire) : Bool :=
  (List.range rs.size).any fun j =>
    let (t, a, c) := rs[j]!
    let lo := lbase[j]! + (m.first - a)
    t == w.reader && decide (a ≤ m.first) && decide (m.first + m.count ≤ a + c) &&
      decide (lo < w.start + w.units / max w.cuts 1) && decide (w.start < lo + m.count)

/-- The reader row a wire `w` of `W` covers, as the public file names it (`interface.row_name`): its port where `W`
leaves the port uncut, else `<port>#<j>`, `j` the wire's first cut row (`Spot.Edges.partWires` gives each wire one). -/
def wireRow (W : Spot.Wires.Wiring) (w : Spot.Wires.Wire) : String :=
  if W.cutsOf w.reader w.port ≤ 1 then w.port else s!"{w.port}#{w.cut0}"

/-- **C's wiring under the partition**: the templates of the partition's ranges `rs`, read off C
(`Spot.RecordText.tmplAt`), in(S)'s row cuts (`ioCuts`), and the wires C's root states (`Spot.Edges.edgesOf`). -/
def derivedWiring (P : Flock.Program.Prog) (rs : Array Flock.TemplateQuery.TRange) : V Spot.Wires.Wiring := do
  let ts ← (Spot.RecordText.unitsOf rs).mapM (Spot.RecordText.tmplAt P rs)
  let W : Spot.Wires.Wiring := { templates := ts.map fun t => (t.tid, t.wired), cuts := ← ioCuts P rs, wires := [] }
  return { W with wires := ← Spot.Edges.edgesOf P rs W }

/-- **The reader rows C wires into the member `m`**, each `(row, <writer>/o<j>)`: for each wire of `W` that enters `m`
over the ranges `rs` (`wireReaches`), its reader row (`wireRow`) and its writer's output field's tree, the pairs
`portTreeHeld` admits a registered port at. -/
def wiredRows (W : Spot.Wires.Wiring) (rs : Array (String × Nat × Nat)) (m : Member) : Array (String × String) :=
  let lbase := localBases rs
  (W.wires.filter (wireReaches rs lbase m)).toArray.map fun w => (wireRow W w, s!"{w.writer}/{w.out}")

/-- `wiredRows` for each member at C's wiring under the verifier's partition (`derivedWiring`), C and the partition
the parameters' own; none for every member where C states no edge, or where C, the partition or the wiring is
refused, which `membersRootsWired` refuses by name later. -/
def wiredRowsOf (p : Params) (ms : Array Member) : Member → Array (String × String) :=
  let got : V (Option (Spot.Wires.Wiring × Array (String × Nat × Nat))) := do
    if ms.all (·.own.registered.isNone) then return none
    let P ← Program.parse p.own.program
    if !hasEdge P.prog then return none
    let rs ← Partition.evaluate P (getD p.own.partition "query")
    return some (← derivedWiring P.prog rs, rs.map fun r => (r.template, r.first, r.count))
  match got with
  | .ok (some (W, rs)) => wiredRows W rs
  | _ => fun _ => #[]

end Spot.OneStageZk
