/-
The composite a configuration's subprotocols make (`verity run CONFIG`): onsite (`Core.Onsite.protocol`,
`verity/onsite/v0`) when they are onsite's, remote (`Core.Remote.protocol`, `verity/remote/v0`) when they are remote's,
listed in any order. These two are the only composites a verifier runs: protocol soundness covers only these two, so
`ofIds` refuses any other set by name (an id neither composes, one listed twice, none, or a set that is neither one's).

    product IDS id | check PARAMS COMMITMENT EVIDENCE | serve        IDS: the subprotocols' ids, comma-separated

The command after `IDS` is the core CLI's (`CoreProtocol.Cli`), of the composite `ofIds IDS`.
-/
import CoreProtocol.Cli
import CoreOnsite
import CoreRemote

namespace Core.Composites

open Composition

/-- The composites, each with its subprotocols' ids. -/
def composites : List (BlobProtocol × List String) :=
  [(Core.Onsite.protocol, Core.Onsite.components.map (·.id)),
    (Core.Remote.protocol, Core.Remote.components.map (·.id))]

/-- Every subprotocol of some composite, by id. -/
def known : List String := (composites.flatMap (·.2)).eraseDups

/-- The composite over exactly the subprotocols `ids` names, in any order, or why the list is refused. -/
def ofIds (ids : List String) : Except String BlobProtocol :=
  match ids.find? (fun i => !known.contains i) with
  | some i => .error s!"{i} is no composite's subprotocol ({", ".intercalate known})"
  | none =>
    match ids.find? (fun i => ids.count i > 1) with
    | some i => .error s!"{i} is listed twice"
    | none =>
      match composites.find? (fun (_, xs) => xs.length == ids.length && xs.all ids.contains) with
      | some (P, _) => .ok P
      | none =>
        let named := composites.map fun (P, xs) => s!"{P.id}: {", ".intercalate xs}"
        .error (s!"[{", ".intercalate ids}] is no composite's subprotocols ({"; ".intercalate named})" ++
          ": protocol soundness covers only those composites")

/-- `product IDS ...`: the core CLI of `ofIds IDS`; 64 when `IDS` is refused or missing. -/
def main (args : List String) : IO UInt32 := do
  match args with
  | ids :: rest =>
    match ofIds ((ids.splitOn ",").filter (· ≠ "")) with
    | .ok P => P.main rest
    | .error e => IO.eprintln s!"product: {e}"; return 64
  | [] => IO.eprintln "usage: product IDS id | check PARAMS COMMITMENT EVIDENCE | serve"; return 64

private def idOf (ids : List String) : Option String := (ofIds ids).toOption.map (·.id)

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

private def onsiteIds : List String := Core.Onsite.components.map (·.id)
private def remoteIds : List String := Core.Remote.components.map (·.id)

#guard composites.map (·.1.id) == ["verity/onsite/v0", "verity/remote/v0"]
#guard onsiteIds.all known.contains && remoteIds.all known.contains
#guard known.length == (onsiteIds ++ remoteIds).eraseDups.length

-- the composites by their subprotocols, in any order, and no other set
#guard idOf onsiteIds == some "verity/onsite/v0"
#guard idOf onsiteIds.reverse == some "verity/onsite/v0"
#guard idOf remoteIds == some "verity/remote/v0"
#guard idOf remoteIds.reverse == some "verity/remote/v0"
#guard idOf [] == none
#guard idOf ["verity/sampled-proofs/one-stage/v1"] == none
#guard idOf ["verity/sampled-proofs/one-stage/v2"] == none
#guard idOf ["verity/sampled-proofs/one-stage/v2", "verity/certifier/v0"] == none
#guard idOf (onsiteIds.drop 1) == none
#guard idOf ["verity/wipe/v0"] == none
#guard idOf ["verity/pouw/v1", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"] == none
#guard idOf (remoteIds ++ ["verity/pouw/period/v1"]) == none
#guard idOf (onsiteIds ++ ["verity/pouw/v1"]) == none
#guard contains (match ofIds ["verity/sampled-proofs/one-stage/v2"] with | .error e => e | .ok _ => "")
  "protocol soundness covers only those composites"
#guard contains (match ofIds ["verity/sampled-proofs/one-stage/v1"] with | .error e => e | .ok _ => "")
  "is no composite's subprotocol"

end Core.Composites
