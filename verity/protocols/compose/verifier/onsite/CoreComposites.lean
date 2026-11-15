/-
The composite a configuration's subprotocols make (`verity run CONFIG`): onsite (`Core.Onsite.protocol`,
`verity/onsite/v0`) when they are onsite's, remote (`Core.Remote.V1.protocol`, `verity/remote/v1`) when they are
remote's, listed in any order. These two are the only composites a verifier runs: protocol soundness covers only these
two (`Verity.verifier`), so `ofIds` refuses any other set by name (an id no composite composes, one listed twice, none,
or a set that is no composite's). Remote's development composite, H1's `Core.Remote.protocol` (`verity/remote/v0`), is
no composite here: `ofIds` refuses its subprotocols, and it runs only as its own development executable (`remote`).

    product IDS id | check PARAMS COMMITMENT EVIDENCE | serve        IDS: the subprotocols' ids, comma-separated

The command after `IDS` is the core CLI's (`CoreProtocol.Cli`), of the composite `ofIds IDS`.
-/
module

public import CoreOnsite
public import CoreProtocol.Cli
public import CoreRemote
meta import CoreOnsite
meta import CoreProtocol.Cli
meta import CoreRemote

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Core.Composites

open Composition

/-- The composites, each with its subprotocols' ids. -/
def composites : List (BlobProtocol × List String) :=
  [(Core.Onsite.protocol, Core.Onsite.components.map (·.id)),
    (Core.Remote.V1.protocol, Core.Remote.V1.components.map (·.id))]

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
private def remoteV1Ids : List String := Core.Remote.V1.components.map (·.id)

#guard composites.map (·.1.id) == ["verity/onsite/v0", "verity/remote/v1"]
#guard onsiteIds.all known.contains && remoteV1Ids.all known.contains
#guard known.length == (onsiteIds ++ remoteV1Ids).eraseDups.length

-- the composites by their subprotocols, in any order, and no other set
#guard idOf onsiteIds == some "verity/onsite/v0"
#guard idOf onsiteIds.reverse == some "verity/onsite/v0"
-- H1's development composite is none: its subprotocols are refused, by name
#guard idOf remoteIds == none
#guard idOf remoteIds.reverse == none
#guard contains (match ofIds remoteIds with | .error e => e | .ok _ => "")
  "is no composite's subprotocols (verity/onsite/v0: "
#guard idOf remoteV1Ids == some "verity/remote/v1"
#guard idOf remoteV1Ids.reverse == some "verity/remote/v1"
#guard idOf (remoteV1Ids ++ ["verity/pouw/period/v1"]) == none
#guard idOf [] == none
#guard idOf ["verity/sampled-proofs/one-stage/v1"] == none
#guard idOf ["verity/sampled-proofs/one-stage/v2"] == none
#guard idOf ["verity/sampled-proofs/v1", "verity/certifier/v0"] == none
#guard idOf (onsiteIds.drop 1) == none
#guard idOf ["verity/wipe/v0"] == none
#guard idOf ["verity/pouw/v1", "verity/pouw/v1", "verity/sampled-proofs/v1"] == none
#guard idOf (remoteIds ++ ["verity/pouw/period/v1"]) == none
#guard idOf (onsiteIds ++ ["verity/pouw/v1"]) == none
#guard contains (match ofIds ["verity/sampled-proofs/v1"] with | .error e => e | .ok _ => "")
  "protocol soundness covers only those composites"
#guard contains (match ofIds ["verity/sampled-proofs/one-stage/v1"] with | .error e => e | .ok _ => "")
  "is no composite's subprotocol"
#guard contains (match ofIds ["verity/sampled-proofs/one-stage/v2"] with | .error e => e | .ok _ => "")
  "is no composite's subprotocol"

end Core.Composites
