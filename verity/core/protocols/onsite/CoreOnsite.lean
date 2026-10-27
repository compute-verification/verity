/-
The two composites, onsite and remote Verity, built by one constructor (`restrict`): `Core.Composition.product` over
onsite's subprotocols restricted to a subset, under the composite's id.

* Onsite (`verity/onsite/v0`, `protocol`): remote Verity's accounting with the verifier's hardware at the site, so it
  also accounts for space, bandwidth and communication. Its subprotocols are sampled proofs' one-stage audit
  (`SampledProofs.OneStage.protocol`), proof of useful work (`PouwCheck.protocol`), PoUS's check of a run
  (`PousCheck.RunProtocol.protocol`), the remote bandwidth audit's session (`RemoteAudit.Protocol.protocol`), and the
  network certifier over the period (`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the period's link-windows
  certified and proved, and the bound `K` on what left the site).
* Remote (`verity/remote/v0`, `remote`): what the verifier establishes with no hardware of its own at the site, onsite
  restricted to sampled proofs and PoUW.

Every subprotocol of a composite accepts or the composite doesn't, and any subprotocol's rejection rejects it whatever
the others answer. These two are the only composites a verifier runs: `ofIds` refuses any other set of subprotocols,
since protocol soundness covers only these two.

    product IDS id | check PARAMS COMMITMENT EVIDENCE | serve        IDS: the subprotocols' ids, comma-separated

The command after `IDS` is the core CLI's (`CoreProtocol.Cli`), of the composite `ofIds IDS`.
-/
import CoreComposition.Cases
import NetworkCheck.Period
import PousCheck.RunProtocol
import PouwCheck.Protocol
import RemoteAudit.Protocol
import SampledProofs.OneStage

namespace Core.Onsite

open Composition

/-- Onsite's subprotocols, in the order of their ids. -/
def subprotocols : List Protocol :=
  [NetworkCheck.Period.protocol, PousCheck.RunProtocol.protocol, PouwCheck.protocol, RemoteAudit.Protocol.protocol,
    SampledProofs.OneStage.protocol]

/-- The one composite constructor: the composite `id` over those of onsite's subprotocols whose ids `ids` lists, in the
order of their ids. -/
def restrict (id : String) (ids : List String) : Protocol := product id (subprotocols.filter (ids.contains ·.id))

def protocol : Protocol := restrict "verity/onsite/v0" (subprotocols.map (·.id))

/-- Remote's subprotocols, by id: sampled proofs and PoUW. -/
def remoteIds : List String := [PouwCheck.protocol.id, SampledProofs.OneStage.protocol.id]

def remote : Protocol := restrict "verity/remote/v0" remoteIds

/-- The composites, each with the ids of the subprotocols it restricts onsite to. -/
def composites : List (Protocol × List String) := [(protocol, subprotocols.map (·.id)), (remote, remoteIds)]

/-- The composite over exactly the subprotocols `ids` names, in any order, or why the list is refused: a subprotocol
onsite doesn't have, one named twice, or a set that is neither onsite's nor remote's. -/
def ofIds (ids : List String) : Except String Protocol :=
  let known := subprotocols.map (·.id)
  match ids.find? (fun i => !known.contains i) with
  | some i => .error s!"{i} is not one of onsite's subprotocols ({", ".intercalate known})"
  | none =>
    match ids.find? (fun i => ids.count i > 1) with
    | some i => .error s!"{i} is listed twice"
    | none =>
      match composites.find? (fun (_, xs) => xs.length == ids.length && xs.all ids.contains) with
      | some (P, _) => .ok P
      | none => .error (s!"[{", ".intercalate ids}] is neither remote's subprotocols ({", ".intercalate remoteIds})" ++
          s!" nor onsite's ({", ".intercalate known}): protocol soundness covers only those two composites")

/-- `product IDS ...`: the core CLI of `ofIds IDS`; 64 when `IDS` is refused or missing. -/
def productMain (args : List String) : IO UInt32 := do
  match args with
  | ids :: rest =>
    match ofIds ((ids.splitOn ",").filter (· ≠ "")) with
    | .ok P => P.main rest
    | .error e => IO.eprintln s!"product: {e}"; return 64
  | [] => IO.eprintln "usage: product IDS id | check PARAMS COMMITMENT EVIDENCE | serve"; return 64

/-- A case its subprotocol's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : Protocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (C : Protocol) (xs : List Case) : Answer :=
  Cases.answerCases C xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

private def idOf (ids : List String) : Option String := (ofIds ids).toOption.map (·.id)

#guard ordered subprotocols
#guard subprotocols.map (·.id) == ["verity/certifier/v0", "verity/pous/p2/v4/run", "verity/pouw/v1",
  "verity/remote-bandwidth/v0", "verity/sampled-proofs/one-stage/v1"]
#guard ordered (subprotocols.filter (remoteIds.contains ·.id))
#guard remoteIds.all (subprotocols.map (·.id)).contains

-- the composites by their subprotocols, in any order, and no other set
#guard idOf (subprotocols.map (·.id)) == some "verity/onsite/v0"
#guard idOf (subprotocols.map (·.id)).reverse == some "verity/onsite/v0"
#guard idOf remoteIds == some "verity/remote/v0"
#guard idOf remoteIds.reverse == some "verity/remote/v0"
#guard idOf [] == none
#guard idOf ["verity/sampled-proofs/one-stage/v1"] == none
#guard idOf ["verity/sampled-proofs/one-stage/v1", "verity/certifier/v0"] == none
#guard idOf ((subprotocols.map (·.id)).drop 1) == none
#guard idOf ["verity/wipe/v0"] == none
#guard idOf ["verity/pouw/v1", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v1"] == none
#guard contains (match ofIds ["verity/sampled-proofs/one-stage/v1"] with | .error e => e | .ok _ => "")
  "protocol soundness covers only those two composites"

-- every subprotocol refuses its case: onsite rejects on each, and makes nothing public
private def allRefused : Answer := answer protocol (subprotocols.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard subprotocols.all fun P => (allRefused.reason.map (contains · s!"{P.id} reject: params: ")).getD false
-- remote's subprotocols alone are not onsite's
#guard (answer protocol [refused PouwCheck.protocol, refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/certifier/v0 is missing"

-- both of remote's subprotocols refuse their cases: remote rejects on both, and makes nothing public
private def bothRefused : Answer := answer remote [refused PouwCheck.protocol, refused SampledProofs.OneStage.protocol]
#guard bothRefused.verdict == .reject
#guard bothRefused.published.isNone && bothRefused.profile.isNone
#guard bothRefused.reason.map (·.startsWith "verity/pouw/v1 reject: params: ") == some true
#guard bothRefused.reason.map (contains · "; verity/sampled-proofs/one-stage/v1 reject: params: ") == some true
#guard (answer remote [refused PouwCheck.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is missing"
-- onsite's other subprotocols are not remote's
#guard (answer remote [refused NetworkCheck.Period.protocol, refused PouwCheck.protocol,
    refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/certifier/v0 is not a component of this composition"

end Core.Onsite
