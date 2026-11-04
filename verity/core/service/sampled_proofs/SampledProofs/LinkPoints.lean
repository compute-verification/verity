/-
A member's link points. Each session of a member's statement opens its link at `PT_LOCAL + nbl g c` points, for its
`c` proved units and its circuit's `g` units a proof block (META `vus_per_block`), as `HmRow.setup` and `HmOut`'s
setup compute it. A draw over several members draws a random count of each member's units, so the guarantee for the
per-member form (`JointAuditV5.hcls_of_bounds`) reads one count of link points per member: at every draw the member's
count lies between its fewest and most drawn units (`bounds`), its statement's whole public file (exactly its units,
`PublicUnits.exactly`) has the most, and the registration refuses a member whose fewest and most give different counts
(`refused`, `Registration.check`'s `link-points`).

Every draw the deployed draw can give counts (`Bridge.execSel`, `Draw.drawUnits`): one that runs out of coins draws
every unit, so a member's most is all its units; one that ends draws, under `subset k`, `k` distinct units of the
population, under `bernoulli a b` each unit when `a ≥ b > 0` and any number of them otherwise (every unit when `b = 0`,
which never ends), and under `stratified` and `work` `k_s` distinct units of each stratum `s` (a stratum with `k_s`
above its size never ends). The fewest is then what is left after the draw's units fill the rest of the population, or
of each stratum, first. O(members × strata × ranges), the descriptor's size, never the units'.
-/
import Flock.Net
import SampledProofs.Draw

namespace SampledProofs.LinkPoints

open Lean (Json)
open Flock (V)

/-- The link points beyond `PT_LOCAL` of a session of `c` proved units at `g` units a proof block: `max 3 ⌈log₂ ⌈c/g⌉⌉`,
`HmRow.setup`'s `nbl`. -/
def nbl (g c : Nat) : Nat := max 3 (Flock.log2ceil (max ((c + g - 1) / g) 1))

/-- The units of the `[lo, hi)` ranges `rs`. -/
def size (rs : Array (Nat × Nat)) : Nat := rs.foldl (fun acc (lo, hi) => acc + (hi - lo)) 0

/-- The units in both `a` and `b`, each `[lo, hi)` ranges disjoint among themselves. -/
def overlap (a b : Array (Nat × Nat)) : Nat :=
  a.foldl (fun acc (lo, hi) => b.foldl (fun acc (lo', hi') => acc + (min hi hi' - max lo lo')) acc) 0

/-- The fewest of the `n` units `us` a stratified draw over `strata` holds when it ends: each stratum's `k_s` less the
stratum's units outside `us`; `n` when a stratum draws more than it has, so the draw never ends. -/
def strataFewest (strata : Array Flock.Draw.Stratum) (us : Array (Nat × Nat)) (n : Nat) : Nat :=
  if strata.any (fun s => s.size < s.k) then n
  else strata.foldl (fun acc s => acc + (s.k - (s.size - overlap us s.units))) 0

/-- The fewest of the units `us` of `[0, population)` a draw under `law` holds when it ends (module docstring). -/
def fewest (law : Flock.Draw.Law) (population : Nat) (us : Array (Nat × Nat)) : Nat :=
  let n := size us
  match law with
  | .subset k => if k ≤ population then k - (population - n) else n
  | .bernoulli a b => if a != 0 && b ≤ a then n else 0
  | .stratified _ strata => strataFewest strata us n
  | .work _ strata => strataFewest (strata.map (·.toStratum)) us n

/-- Each member's fewest and most drawn units under `law` over `[0, population)`: a member is its units, `[lo, hi)`
ranges, and its circuit's units a proof block. A member with no unit, a range of none, ranges that meet, a unit outside
the population, or no unit a proof block is refused. -/
def bounds (law : Flock.Draw.Law) (population : Nat) (ms : Array (Array (Nat × Nat) × Nat)) : V (Array (Nat × Nat)) :=
  ms.mapIdxM fun i (us, g) => do
    if g == 0 then throw s!"link points: member {i}: its vus_per_block is 0: a proof block holds at least one unit"
    let rs := us.qsort fun a b => a.1 < b.1
    if rs.isEmpty || rs.any (fun (lo, hi) => hi ≤ lo || population < hi)
        || (List.range (rs.size - 1)).any (fun j => rs[j + 1]!.1 < rs[j]!.2) then
      throw s!"link points: member {i}: its units are not nonempty, disjoint [lo, hi) ranges of the population's {population}"
    pure (fewest law population us, size us)

/-- The members whose fewest and most drawn units give different link points (`bounds`), by index. -/
def refused (law : Flock.Draw.Law) (population : Nat) (ms : Array (Array (Nat × Nat) × Nat)) : V (Array Nat) := do
  let bs ← bounds law population ms
  return (List.range ms.size).toArray.filter fun i =>
    let g := ms[i]!.2
    let (lo, hi) := bs[i]!
    nbl g lo != nbl g hi

/-- The law the draw runs on (`OneStage.drawnFrom`): the verifier's own expanded over its program and query
(`Draw.deriveLaw`), else its subset or Bernoulli law as C-Flock reads it (`Draw.flockLaw`); with the population. -/
def lawOf (p : Program) (query law : Json) (table : Option Json) : V (Flock.Draw.Law × Nat) := do
  let population ← Partition.units p query
  match ← Draw.deriveLaw p query law table with
  | some l => pure (l, population)
  | none => pure (← Draw.flockLaw law population, population)

/-- `{"law", "population", "members"}` (the law as `Draw.flockLaw` reads it, each member `{"units", "vus_per_block"}`) to
`{"bounds": [[fewest, most], ...], "refused": [index, ...]}`; a malformed law or member is refused. -/
def piece (j : Json) : V Json := do
  let some population := nat? (getD j "population") | throw "the population is not a natural number"
  let law ← Draw.flockLaw (getD j "law") population
  let .arr ms := getD j "members" | throw "the members are not a list"
  let ms ← ms.mapM fun m => do
    let some g := nat? (getD m "vus_per_block") | throw "a member's vus_per_block is not a natural number"
    let some us := (Draw.ranges? (getD m "units")).bind fun rs =>
        rs.mapM fun (lo, hi) => if 0 ≤ lo then some (lo.toNat, hi.toNat) else none
      | throw "a member's units are not a nonempty list of [lo, hi] with 0 <= lo < hi"
    pure (us, g)
  let bs ← bounds law population ms
  let pair := fun ((a, b) : Nat × Nat) => Json.arr #[Lean.toJson a, Lean.toJson b]
  return Json.mkObj [("bounds", Json.arr (bs.map pair)),
    ("refused", Json.arr ((← refused law population ms).map Lean.toJson))]

#guard nbl 1 0 == 3 && nbl 1 8 == 3 && nbl 1 9 == 4 && nbl 2 16 == 3 && nbl 2 17 == 4
#guard fewest (.subset 2) 10 #[(0, 10)] == 2 && fewest (.subset 2) 10 #[(0, 3)] == 0
#guard fewest (.subset 11) 10 #[(0, 3)] == 3
#guard fewest (.bernoulli 1 2) 10 #[(0, 10)] == 0 && fewest (.bernoulli 1 1) 10 #[(0, 10)] == 10
#guard fewest (.stratified 2 #[{ template := "a", units := #[(0, 4)], k := 2 }, { template := "b", units := #[(4, 10)], k := 3 }])
  10 #[(2, 8)] == 0 + 1
#guard (refused (.subset 2) 10 #[(#[(0, 10)], 1), (#[(0, 10)], 2)]).toOption == some #[0]
#guard (refused (.subset 2) 10 #[(#[(0, 11)], 1)]).toOption == none

end SampledProofs.LinkPoints
