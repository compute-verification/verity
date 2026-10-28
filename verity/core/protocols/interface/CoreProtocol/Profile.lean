/-
The integrity profile: what an accepted check established, as `verity.core.protocols.profile.IntegrityProfile`
records it. These are the facts checked exhaustively, the verifier's sampling design over named levels of a partition
hierarchy (coarse to fine; the finest level is the checked one), the members drawn at each level, and δ. The JSON is
`IntegrityProfile.to_json`'s, except that δ is an exact ratio string (`Core.Ratio`) and not a float.

A protocol whose acceptance is not a sampling law over proof units (a timed storage audit, a certificate check) declares
its own profile type; this one is shared by the protocols that sample. `Integrity.ofJson` reads it back, for a protocol
that consumes another's answer.
-/
import Lean.Data.Json
import Flock.Canon
import Flock.Draw
import CoreProtocol.Ratio

namespace Core

open Lean (Json)
open Flock (V)

structure Stratum where
  name : String
  size : Nat
  k : Nat
  deriving BEq, DecidableEq, Repr

/-- A stratum of the work sampling design: its name, size and `k`, the work of each of its members (an integer in any
fixed unit, `0` for members that do none) and its floor, the fewest members it draws (`profile.WorkStratum`). -/
structure WorkStratum extends Stratum where
  work : Nat
  floor : Nat
  deriving BEq, DecidableEq, Repr

/-- How a level is drawn, its sampling design: each member independently with probability `p`, exactly `k` members
uniformly, uniform draws inside disjoint strata of the checked level, or the work design `work:k`, uniform draws inside
strata whose `k_s` its rule sizes by work above each stratum's floor (`Flock.Draw.workK`) (`profile.Bernoulli`,
`Subset`, `Stratified`, `Work`). The sample of a work design is a stratified one at its `k_s`. -/
inductive Draw where
  | bernoulli (p : Ratio)
  | subset (k : Nat)
  | stratified (strata : Array Stratum)
  | work (k : Nat) (strata : Array WorkStratum)
  deriving BEq, DecidableEq, Repr

/-- A work design's population work, `Σ_s work_s · size_s`. -/
def WorkStratum.total (strata : Array WorkStratum) : Nat := strata.foldl (fun acc s => acc + s.work * s.size) 0

/-- Why `strata` are not the work design `work:k`'s, else none (`profile.Work`'s refusals): `1 ≤ k ≤ population`,
distinct names, a floor of at least one in every stratum, a population that does some work, and each stratum's `k` the
rule's, `Flock.Draw.workK` above its floor. -/
def WorkStratum.problem (k : Nat) (strata : Array WorkStratum) : Option String := Id.run do
  let population := strata.foldl (fun acc s => acc + s.size) 0
  if strata.isEmpty || k < 1 || k > population then return some "a work design needs strata and 1 <= k <= population"
  let total := WorkStratum.total strata
  if total == 0 then return some "a work design's population does no work"
  let mut seen : Array String := #[]
  for s in strata do
    if seen.contains s.name then return some s!"stratum '{s.name}' twice"
    seen := seen.push s.name
    if s.floor < 1 then return some s!"stratum '{s.name}': a floor is at least one"
    if s.k != Flock.Draw.workK k s.work s.size total s.floor then
      return some s!"stratum '{s.name}': k is not the work rule's"
  return none

/-- The sampling design of C-Flock's law (`Flock.Draw.Law`) at its sizes: each stratum named by its template, its size
its ranges' total. `none` for a Bernoulli law of denominator 0. -/
def Draw.ofFlock : Flock.Draw.Law → Option Draw
  | .subset k => some (.subset k)
  | .bernoulli a b => (Ratio.mk' a b).map .bernoulli
  | .stratified _ strata => some (.stratified (strata.map fun s => { name := s.template, size := s.size, k := s.k }))
  | .work k strata => some (.work k (strata.map fun s =>
      { name := s.template, size := s.size, k := s.k, work := s.work, floor := s.floor }))

structure Level where
  name : String
  /-- The id of the query whose family is this level's members, or `none` for a level bound to no Program. -/
  query : Option String
  size : Nat
  draw : Option Draw
  deriving BEq, DecidableEq, Repr

structure Integrity where
  /-- Digests of the evaluated Programs, in commit order. -/
  programs : Array String
  /-- Families of `Input` values checked against external sources. -/
  prescribed : Array String
  /-- Every other `Input` equals a committed output of an earlier Program in the transcript. -/
  linked : Bool
  /-- Gate families proved on every audit, outside the draws. -/
  covered : Array String
  levels : Array Level
  delta : Ratio
  /-- Members actually drawn at each level, all passed, in level order. -/
  drawn : Array (String × Nat)
  deriving BEq, DecidableEq, Repr

private def sorted (xs : Array String) : Json := Json.arr ((xs.qsort (· < ·)).map Json.str)

def Draw.toJson : Draw → Json
  | .bernoulli p => Json.mkObj [("bernoulli", Json.str p.toString)]
  | .subset k => Json.mkObj [("subset", Json.num k)]
  | .stratified strata => Json.mkObj [("law", Json.str "stratified"),
      ("strata", Json.arr (strata.map fun s =>
        Json.mkObj [("name", Json.str s.name), ("size", Json.num s.size), ("k", Json.num s.k)]))]
  | .work k strata => Json.mkObj [("law", Json.str "work"), ("k", Json.num k),
      ("strata", Json.arr (strata.map fun s =>
        Json.mkObj [("name", Json.str s.name), ("size", Json.num s.size), ("k", Json.num s.k),
          ("work", Json.num s.work), ("floor", Json.num s.floor)]))]

def Level.toJson (l : Level) : Json :=
  Json.mkObj [("name", Json.str l.name), ("query", match l.query with | some q => Json.str q | none => Json.null),
    ("size", Json.num l.size), ("draw", match l.draw with | some d => d.toJson | none => Json.null)]

def Integrity.toJson (p : Integrity) : Json :=
  Json.mkObj [("programs", Json.arr (p.programs.map Json.str)), ("prescribed", sorted p.prescribed),
    ("linked", Json.bool p.linked), ("covered", sorted p.covered), ("levels", Json.arr (p.levels.map Level.toJson)),
    ("delta", Json.str p.delta.toString),
    ("drawn", Json.mkObj (p.drawn.toList.map fun (n, k) => (n, Json.num k)))]

/-- `a`, when its JSON is `j` written back: the decoders below accept only what `toJson` writes (keys, sorted lists,
lowest-terms ratios), so every value has one JSON spelling. -/
private def exactly {α} (toJson : α → Json) (j : Json) (ctx : String) (a : α) : V α := do
  if (← Flock.canon (toJson a)) != (← Flock.canon j) then throw s!"{ctx}: not in the form toJson writes"
  pure a

private def strs (j : Json) (k ctx : String) : V (Array String) := do
  (← Flock.J.arr j k ctx).mapM (Flock.J.asStr · s!"{ctx}.{k}")

private def nullOr {α} (f : Json → String → V α) (j : Json) (k ctx : String) : V (Option α) := do
  match ← Flock.J.get j k ctx with
  | Json.null => pure none
  | v => some <$> f v s!"{ctx}.{k}"

def Draw.ofJson (j : Json) (ctx : String) : V Draw := do
  let d ← if Flock.J.has j "bernoulli" then do pure (.bernoulli (← Ratio.ofString (← Flock.J.str j "bernoulli" ctx) s!"{ctx}.bernoulli"))
    else if Flock.J.has j "subset" then do pure (.subset (← Flock.J.nat j "subset" ctx))
    else if Flock.J.has j "strata" && j.getObjVal? "law" matches .ok (.str "work") then do
      let k ← Flock.J.nat j "k" ctx
      let strata ← (← Flock.J.arr j "strata" ctx).mapM fun s => do
        let c := s!"{ctx}.strata"
        pure ({ name := ← Flock.J.str s "name" c, size := ← Flock.J.nat s "size" c, k := ← Flock.J.nat s "k" c,
                work := ← Flock.J.nat s "work" c, floor := ← Flock.J.nat s "floor" c } : WorkStratum)
      if let some why := WorkStratum.problem k strata then throw s!"{ctx}: {why}"
      pure (.work k strata)
    else if Flock.J.has j "strata" then do
      let strata ← (← Flock.J.arr j "strata" ctx).mapM fun s => do
        let c := s!"{ctx}.strata"
        pure ({ name := ← Flock.J.str s "name" c, size := ← Flock.J.nat s "size" c, k := ← Flock.J.nat s "k" c } : Stratum)
      pure (.stratified strata)
    else throw s!"{ctx}: not a bernoulli, subset, stratified or work draw"
  exactly Draw.toJson j ctx d

def Level.ofJson (j : Json) (ctx : String) : V Level := do
  let l : Level := {
    name := ← Flock.J.str j "name" ctx, query := ← nullOr Flock.J.asStr j "query" ctx, size := ← Flock.J.nat j "size" ctx,
    draw := ← nullOr Draw.ofJson j "draw" ctx }
  exactly Level.toJson j ctx l

/-- The inverse of `toJson`. `drawn` is read in level order; a count for a level the profile doesn't name is refused. -/
def Integrity.ofJson (j : Json) (ctx : String := "profile") : V Integrity := do
  let mut levels := #[]
  for l in ← Flock.J.arr j "levels" ctx do
    levels := levels.push (← Level.ofJson l s!"{ctx}.levels[{levels.size}]")
  let counts ← Flock.J.get j "drawn" ctx
  let mut drawn := #[]
  for l in levels do
    if Flock.J.has counts l.name then drawn := drawn.push (l.name, ← Flock.J.nat counts l.name s!"{ctx}.drawn")
  let p : Integrity := {
    programs := ← strs j "programs" ctx, prescribed := ← strs j "prescribed" ctx, linked := ← Flock.J.bool j "linked" ctx,
    covered := ← strs j "covered" ctx, levels, delta := ← Ratio.ofString (← Flock.J.str j "delta" ctx) s!"{ctx}.delta",
    drawn }
  exactly Integrity.toJson j ctx p

private def workExample : Draw :=
  .work 3 #[{ name := "a", size := 4, k := 2, work := 3, floor := 1 }, { name := "b", size := 6, k := 1, work := 1, floor := 1 }]

#guard (Draw.ofJson workExample.toJson "draw").toOption == some workExample
#guard (Draw.ofJson (Draw.stratified #[{ name := "a", size := 4, k := 2 }]).toJson "draw").toOption ==
  some (.stratified #[{ name := "a", size := 4, k := 2 }])
#guard (Draw.ofJson (Draw.work 3 #[{ name := "a", size := 4, k := 1, work := 3, floor := 1 }]).toJson "draw").isOk == false
#guard (Draw.ofJson (Draw.work 3 #[{ name := "a", size := 4, k := 3, work := 0, floor := 3 }]).toJson "draw").isOk == false

end Core
