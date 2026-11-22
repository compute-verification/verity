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

end Spot.OneStageZk
