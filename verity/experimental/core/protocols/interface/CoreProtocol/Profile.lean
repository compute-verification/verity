/-
The integrity profile: what an accepted check established, as `verity.experimental.protocols.profile.IntegrityProfile`
records it. These are the facts checked exhaustively, the verifier's law over named levels of a partition hierarchy
(coarse to fine; the finest level is the checked one), the members drawn at each level, and δ. The JSON is
`IntegrityProfile.to_json`'s, except that δ is an exact ratio string (`Core.Ratio`) and not a float.

A protocol whose acceptance is not a sampling law over proof units (a timed storage audit, a certificate check) declares
its own profile type; this one is shared by the protocols that sample.
-/
import Lean.Data.Json
import CoreProtocol.Ratio

namespace Core

open Lean (Json)

structure Stratum where
  name : String
  size : Nat
  k : Nat
  deriving BEq, Repr

/-- How a level is drawn: each member independently with probability `p`, exactly `k` members uniformly, or uniform
draws inside disjoint strata of the checked level (`profile.Bernoulli`, `Subset`, `Stratified`). -/
inductive Draw where
  | bernoulli (p : Ratio)
  | subset (k : Nat)
  | stratified (strata : Array Stratum)
  deriving BEq, Repr

structure Level where
  name : String
  /-- The id of the query whose family is this level's members, or `none` for a level bound to no Program. -/
  query : Option String
  size : Nat
  draw : Option Draw
  deriving BEq, Repr

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
  deriving BEq, Repr

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

end Core
