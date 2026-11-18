/-
The challenger's pin (`Spot.ChallengerPin`) on SPOT's executable (`sampled`) and on a composite: a case whose SPOT
parameters name, at any level, no challenger, a retired key or another key than the pin's is refused by name (`reason`),
and a composite also over its PoUW period member's `key`, the challenger's that signs the draw coin
(`Spot.ChallengerPin.periodReason`). Under the pin, parameters the check can't read the keys from are refused too
(`Spot.ChallengerPin.UNREAD`).

The window's root needs no second comparison: SPOT draws its coins (`Challenger.sampleCoins`) only from the ledger's open
whose root is the parameters' window root, in a ledger checked under the case's key (`auditedOpen`), withheld on that
root (`withheldOn`). Under the pinned key a root the challenger never opened is refused there.
-/
module

public import Spot.ChallengerPin
public import VStarRecursive.Sampled
meta import VStarRecursive.Sampled

@[expose] public section

namespace VStar.Recursive.ChallengerPin

open Lean (Json)
open Flock (V)
open Spot
open Spot.ChallengerPin (Pin keyReason periodReason parse unpinned UNREAD)

/-- The challenger key each level of `p` names: one level's, or a two-level case's boundary and interior. -/
def levels : Sampled.Params → List (Option Challenger.ChallengerKey)
  | .inl p => [p.base.challenger.map (·.1)]
  | .inr p => [p.boundary.challenger.map (·.1), p.interior.challenger.map (·.1)]

/-- Why `p` is refused under `pin`: the first level naming no challenger, a retired key or another key. -/
def reason (pin : Pin) (p : Sampled.Params) : Option String :=
  (levels p).findSome? (keyReason pin)

/-- `reason` on SPOT's parameters `j`, `unread` where they don't parse. -/
def jsonReason (pin : Pin) (j : Json) (unread : Option String := none) : Option String :=
  match Sampled.paramsOfJson j with
  | .ok p => reason pin p
  | .error _ => unread

/-- `jsonReason` on each SPOT component, and `periodReason` on each PoUW period component, of a composite's
parameters `{"components":[{"id","params"},...]}`. -/
def compositeReason (pin : Pin) (j : Json) : Option String :=
  match getD j "components" with
  | .arr xs => xs.toList.findSome? fun x =>
    if getD x "id" == Json.str Sampled.PROTOCOL then jsonReason pin (getD x "params") (some UNREAD)
    else if getD x "id" == Json.str Spot.ChallengerPin.PERIOD then periodReason pin (getD x "params")
    else none
  | _ => some UNREAD

/-- `P`, refusing parameters `why` names before its own parse reads them. -/
def guarded (P : Core.BlobProtocol) (why : Json → Option String) : Core.BlobProtocol :=
  { P with params := fun j => do
      if let some r := why j then throw r
      P.params j }

/-- SPOT's protocol under `pin`; parameters that don't parse are left to its own refusal. -/
def sampled (pin : Pin) : Core.BlobProtocol := guarded Sampled.protocol (jsonReason pin)

/-- A composite under `pin`, over its SPOT and PoUW period components. -/
def composite (pin : Pin) (P : Core.BlobProtocol) : Core.BlobProtocol := guarded P (compositeReason pin)

/-- The command line of a composite `P` (`Core.BlobProtocol.main`), under the pin `parse` reads ahead of it. -/
def compositeMain (P : Core.BlobProtocol) (args : List String) : IO UInt32 := do
  match ← parse args with
  | (some pin, rest) => (composite pin P).main rest
  | (none, rest) => unpinned P.main rest

end VStar.Recursive.ChallengerPin
