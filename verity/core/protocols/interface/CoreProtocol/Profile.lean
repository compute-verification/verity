/-
The integrity profile: what an accepted check established, as `verity.core.protocols.profile.IntegrityProfile`
records it. These are the facts checked exhaustively, the verifier's law over named levels of a partition hierarchy
(coarse to fine; the finest level is the checked one), the members drawn at each level, and δ. The JSON is
`IntegrityProfile.to_json`'s, except that δ is an exact ratio string (`Core.Ratio`) and not a float.

A protocol whose acceptance is not a sampling law over proof units (a timed storage audit, a certificate check) declares
its own profile type; this one is shared by the protocols that sample. `Integrity.ofJson` reads it back, for a protocol
that consumes another's answer.
-/
import Lean.Data.Json
import Flock.Canon
import CoreProtocol.Ratio

namespace Core

open Lean (Json)
open Flock (V)

structure Stratum where
  name : String
  size : Nat
  k : Nat
  deriving BEq, DecidableEq, Repr

/-- How a level is drawn: each member independently with probability `p`, exactly `k` members uniformly, or uniform
draws inside disjoint strata of the checked level (`profile.Bernoulli`, `Subset`, `Stratified`). -/
inductive Draw where
  | bernoulli (p : Ratio)
  | subset (k : Nat)
  | stratified (strata : Array Stratum)
  deriving BEq, DecidableEq, Repr

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
    else if Flock.J.has j "strata" then do
      let strata ← (← Flock.J.arr j "strata" ctx).mapM fun s => do
        let c := s!"{ctx}.strata"
        pure ({ name := ← Flock.J.str s "name" c, size := ← Flock.J.nat s "size" c, k := ← Flock.J.nat s "k" c } : Stratum)
      pure (.stratified strata)
    else throw s!"{ctx}: not a bernoulli, subset or stratified draw"
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

end Core
