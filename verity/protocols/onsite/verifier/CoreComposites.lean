/-
The composite a configuration's subprotocols make (`verity run CONFIG`): onsite (`Core.Onsite.protocol`,
`verity/onsite/v0`) when they are onsite's, remote (`Core.Remote.protocol`, `verity/remote/v0`) when they are remote's,
and the composite over one of two smaller onsite lists (`network`, `compute`) when they are that list's, listed in any
order. A composite is its list of ids in the order of the ids, so one set of subprotocols has one composite and one id.
These four are the only composites a verifier runs, so `ofIds` refuses any other set by name (an id none composes, one
listed twice, none, or a set that is none of the four).

A site's log is registered for exactly these lists (`Core.Site.Registration.Composite.all`, which the guards below hold
the table to), and registration's two clauses read the list: a list with the network certifier's check holds the Site's
ways out to a certifier, and one with a check a memory challenger takes part in holds its memory to a challenger's link
(`Core.Site.Entry.Covered`, `Core.Site.Entry.Reached`). A list `ofIds` comes to accept beyond these is added to both in
the order of its ids.

    product IDS id | check PARAMS COMMITMENT EVIDENCE | serve        IDS: the subprotocols' ids, comma-separated

The command after `IDS` is the core CLI's (`CoreProtocol.Cli`), of the composite `ofIds IDS`.
-/
module

public import CoreOnsite
public import CoreProtocol.Cli
public import CoreProtocol.Site
public import CoreRemote
meta import CoreOnsite
meta import CoreProtocol.Cli
meta import CoreProtocol.Site
meta import CoreRemote

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Core.Composites

open Composition

/-- The ties between subprotocols: every window of PoUW's period on the policy's record, and PoUS and the remote
bandwidth audit on one store. A composite has each tie whose subprotocols it lists. -/
def ties : List Tie := [Core.Remote.windows, Core.Onsite.store]

/-- The id of the composite over the list `k` that has no id of its own: its ids, comma-separated, after a prefix no
protocol's id has. -/
def idOver (k : List String) : String := "verity/composite/v0:" ++ ",".intercalate k

/-- **The composite over the list `k`**: of onsite's subprotocols, those `k` lists, in the order of their ids, under the
one rule onsite and remote compose by (`Core.Composition.tiedSameB`), with the subprotocols that commit to one record
and the ties, each where `k` lists its subprotocols. -/
def over (k : List String) : BlobProtocol :=
  tiedSameB (idOver k) (Core.Onsite.tied.filter (· ∈ k)) (ties.filter fun t => t.names.all fun n => n.1 ∈ k)
    (Core.Onsite.components.filter (·.id ∈ k))

/-- Network certification, sampled proofs and the wipe. -/
def network : List String := Core.Site.Registration.Composite.network

/-- `network`'s, the inference-only policy and PoUW over the period. -/
def compute : List String := Core.Site.Registration.Composite.compute

/-- The composites, each with its subprotocols' ids. -/
def composites : List (BlobProtocol × List String) :=
  [(Core.Onsite.protocol, Core.Onsite.components.map (·.id)),
    (Core.Remote.protocol, Core.Remote.components.map (·.id)),
    (over network, network), (over compute, compute)]

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

#guard composites.map (·.1.id) == ["verity/onsite/v0", "verity/remote/v0",
  "verity/composite/v0:verity/certifier/v0,verity/sampled-proofs/v1,verity/wipe/v0",
  "verity/composite/v0:verity/certifier/v0,verity/inference-only/v0,verity/pouw/period/v1,verity/sampled-proofs/v1," ++
    "verity/wipe/v0"]
-- each composite's subprotocols are the ones its list names, in the list's order, which is the order of the ids
#guard composites.all fun (_, xs) => xs.all known.contains && xs.eraseDups == xs
#guard (over network).id == idOver network && (over compute).id == idOver compute
#guard (Core.Onsite.components.filter (·.id ∈ network)).map (·.id) == network
#guard (Core.Onsite.components.filter (·.id ∈ compute)).map (·.id) == compute
#guard orderedB (Core.Onsite.components.filter (·.id ∈ network))
#guard orderedB (Core.Onsite.components.filter (·.id ∈ compute))
-- the smaller lists' ties: neither lists PoUS, so neither has the store's; `compute` lists the policy and PoUW's
-- period, so it has the windows'; `network` lists neither
#guard (ties.filter fun t => t.names.all fun n => n.1 ∈ network).map (·.what) == []
#guard (ties.filter fun t => t.names.all fun n => n.1 ∈ compute).map (·.what) == [Core.Remote.windows.what]
#guard Core.Onsite.tied.filter (· ∈ network) == ["verity/sampled-proofs/v1"]
#guard Core.Onsite.tied.filter (· ∈ compute) == Core.Onsite.tied
-- registration's clauses read these lists: both list the certifier's check and a check a memory challenger takes
-- part in, and remote's lists neither
#guard Core.Site.Registration.CERTIFIER == NetworkCheck.Period.protocol.id
#guard Core.Site.Registration.CHALLENGED ==
  [PousCheck.RunProtocol.protocol.id, MemoryChallenger.WipeProtocol.protocol.id]
#guard [network, compute, onsiteIds].all fun k =>
  k.contains Core.Site.Registration.CERTIFIER && k.any Core.Site.Registration.CHALLENGED.contains
#guard !remoteIds.contains Core.Site.Registration.CERTIFIER &&
  !remoteIds.any Core.Site.Registration.CHALLENGED.contains
#guard onsiteIds == Core.Site.Registration.Composite.onsite && remoteIds == Core.Site.Registration.Composite.remote
-- a log is registered for exactly the composites the table builds, each named by its own ids
#guard composites.map (·.2) == Core.Site.Registration.Composite.all
#guard Core.Site.Registration.Composite.all.map Core.Site.Registration.Composite.name ==
  ["onsite", "remote", ",".intercalate network, ",".intercalate compute]
#guard onsiteIds.all known.contains && remoteIds.all known.contains
#guard known.length == (onsiteIds ++ remoteIds).eraseDups.length

-- the composites by their subprotocols, in any order, and no other set
#guard idOf onsiteIds == some "verity/onsite/v0"
#guard idOf onsiteIds.reverse == some "verity/onsite/v0"
#guard idOf remoteIds == some "verity/remote/v0"
#guard idOf remoteIds.reverse == some "verity/remote/v0"
#guard idOf network == some (idOver network) && idOf network.reverse == some (idOver network)
#guard idOf compute == some (idOver compute) && idOf compute.reverse == some (idOver compute)
#guard idOf (network ++ ["verity/pouw/period/v1"]) == none
#guard idOf (compute.drop 1) == none
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
