/-
A member's link points. Each session of a member's statement opens its link at `PT_LOCAL + nbl g c` points, for its
`c` proved units and its circuit's `g` units a proof block (META `vus_per_block`), as `HmRow.setup` and `HmOut`'s
setup compute it. A sample over several members samples a random count of each member's units, so the guarantee for the
per-member form (`JointAuditV5.hcls_of_bounds`) reads one count of link points per member. A sample that runs out of the
verifier's public coins fails the window (`OneStageZk.RAN_OUT`), so only a sample that ends gives counts: between the
member's fewest and most units (`ended`, `bounds`), and the registration refuses a member whose fewest and most give
different counts, and a sampling design no sample of which ends (`refused`, `Registration.check`'s `link-points`).

A sample that ends (`Draw.drawUnits`) samples, under `subset k`, `k` distinct units of the population (`k` above the
population never ends); under `bernoulli a b`, none when `a = 0`, each unit when `a ≥ b > 0`, and any number of them
otherwise (`b = 0 < a` never ends); and under `stratified` and `work`, `k_s` distinct units of each stratum `s` (a
stratum with `k_s` above its size never ends). The fewest is what is left after the sample's units fill the rest of the
population, or of each stratum, first; the most is `k`, or each stratum's `k_s`, where the member holds that many. A
sampling design no sample of which ends gives no counts, and the registration refuses it (`ends`, `NEVER_ENDS`), so one it passes
has a sample that ends. O(members × strata × ranges), the descriptor's size, never the units'.
-/
module

public import Flock.Net
public import Spot.Draw
meta import Flock.Net
meta import Spot.Draw

@[expose] public section

namespace Spot.LinkPoints

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

/-- The fewest and most of the units `us` a stratified sample over `strata` holds when it ends: summed over the strata,
`k_s` less the stratum's units outside `us`, and `k_s` or the stratum's units in `us`, whichever is fewer; `none` when a
stratum samples more than it has, so the sample never ends. -/
def strataEnded (strata : Array Flock.Draw.Stratum) (us : Array (Nat × Nat)) : Option (Nat × Nat) :=
  if strata.any (fun s => s.size < s.k) then none
  else some (strata.foldl (fun (lo, hi) s =>
    let o := overlap us s.units
    (lo + (s.k - (s.size - o)), hi + min s.k o)) (0, 0))

/-- The fewest and most of the units `us` of `[0, population)` a sample under `law` holds when it ends (module
docstring); `none` when no sample under `law` ends. -/
def ended (law : Flock.Draw.Law) (population : Nat) (us : Array (Nat × Nat)) : Option (Nat × Nat) :=
  let n := size us
  match law with
  | .subset k => if k ≤ population then some (k - (population - n), min k n) else none
  | .bernoulli a b => if a == 0 then some (0, 0) else if b == 0 then none else if b ≤ a then some (n, n) else some (0, n)
  | .stratified _ strata => strataEnded strata us
  | .work _ strata => strataEnded (strata.map (·.toStratum)) us

/-- Each member's fewest and most sampled proof blocks under `law` over `[0, population)` when the sample ends (`ended`): a member
is its units, `[lo, hi)` ranges, and its circuit's units a proof block. A member with no unit, a range of none, ranges
that meet, a unit outside the population, or no unit a proof block is refused. -/
def bounds (law : Flock.Draw.Law) (population : Nat) (ms : Array (Array (Nat × Nat) × Nat)) :
    V (Array (Option (Nat × Nat))) :=
  ms.mapIdxM fun i (us, g) => do
    if g == 0 then throw s!"link points: member {i}: its vus_per_block is 0: a proof block holds at least one unit"
    let rs := us.qsort fun a b => a.1 < b.1
    if rs.isEmpty || rs.any (fun (lo, hi) => hi ≤ lo || population < hi)
        || (List.range (rs.size - 1)).any (fun j => rs[j + 1]!.1 < rs[j]!.2) then
      throw s!"link points: member {i}: its units are not nonempty, disjoint [lo, hi) ranges of the population's {population}"
    pure (ended law population us)

/-- Whether some sample under `law` over `[0, population)` ends: whether `ended` gives counts, which reads only the sampling design and
the population, never the member's units. -/
def ends (law : Flock.Draw.Law) (population : Nat) : Bool := (ended law population #[]).isSome

/-- The refusal of a sampling design no sample of which ends (`ends`). -/
def NEVER_ENDS : String := "link points: no draw under the law ends, so each of its windows fails"

/-- The members whose fewest and most sampled proof blocks, over the samples that end, give different link points (`bounds`), by
index; refused (`NEVER_ENDS`) when no sample under `law` ends. -/
def refused (law : Flock.Draw.Law) (population : Nat) (ms : Array (Array (Nat × Nat) × Nat)) : V (Array Nat) := do
  let bs ← bounds law population ms
  if !ends law population then throw NEVER_ENDS
  return (List.range ms.size).toArray.filter fun i =>
    let g := ms[i]!.2
    match bs[i]! with
    | some (lo, hi) => nbl g lo != nbl g hi
    | none => false

/-- The sampling design the sample runs on (`OneStage.drawnFrom`): the verifier's own expanded over its program and query
(`Draw.deriveLaw`), else its subset or Bernoulli law as Flock reads it (`Draw.flockLaw`); with the population. -/
def lawOf (p : Program) (query law : Json) (table : Option Json) : V (Flock.Draw.Law × Nat) := do
  let population ← Partition.units p query
  match ← Draw.deriveLaw p query law table with
  | some l => pure (l, population)
  | none => pure (← Draw.flockLaw law population, population)

/-- `{"law", "population", "members"}` (the sampling design as `Draw.flockLaw` reads it, each member `{"units", "vus_per_block"}`) to
`{"bounds": [[fewest, most], ...], "refused": [index, ...]}`; a malformed law or member, or a sampling design no sample of which ends
(`NEVER_ENDS`), is refused. -/
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
  let pair : Option (Nat × Nat) → Json
    | some (a, b) => Json.arr #[Lean.toJson a, Lean.toJson b]
    | none => Json.null
  return Json.mkObj [("bounds", Json.arr (bs.map pair)),
    ("refused", Json.arr ((← refused law population ms).map Lean.toJson))]

#guard nbl 1 0 == 3 && nbl 1 8 == 3 && nbl 1 9 == 4 && nbl 2 16 == 3 && nbl 2 17 == 4
#guard ended (.subset 2) 10 #[(0, 10)] == some (2, 2) && ended (.subset 2) 10 #[(0, 3)] == some (0, 2)
#guard ended (.subset 11) 10 #[(0, 3)] == none
#guard ended (.bernoulli 1 2) 10 #[(0, 10)] == some (0, 10) && ended (.bernoulli 1 1) 10 #[(0, 10)] == some (10, 10)
#guard ended (.bernoulli 0 3) 10 #[(0, 10)] == some (0, 0) && ended (.bernoulli 1 0) 10 #[(0, 10)] == none
#guard ended (.stratified 2 #[{ template := "a", units := #[(0, 4)], k := 2 }, { template := "b", units := #[(4, 10)], k := 3 }])
  10 #[(2, 8)] == some (0 + 1, 2 + 3)
#guard (refused (.subset 20) 40 #[(#[(0, 30)], 1), (#[(0, 30)], 4)]).toOption == some #[0]
#guard (refused (.subset 2) 10 #[(#[(0, 10)], 1)]).toOption == some #[]
#guard ends (.subset 10) 10 && !ends (.subset 11) 10 && ends (.bernoulli 0 0) 10 && !ends (.bernoulli 1 0) 10
#guard !ends (.stratified 3 #[{ template := "a", units := #[(0, 2)], k := 3 }]) 10
#guard [Flock.Draw.Law.subset 11, .bernoulli 1 0, .stratified 3 #[{ template := "a", units := #[(0, 2)], k := 3 }]].all
  fun law => match refused law 10 #[(#[(0, 10)], 1)] with | .error e => e == NEVER_ENDS | .ok _ => false
#guard (refused (.subset 2) 10 #[(#[(0, 11)], 1)]).toOption == none

end Spot.LinkPoints
