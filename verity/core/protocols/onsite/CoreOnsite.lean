/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space, bandwidth and communication. Its subprotocols are integrity by SPOT
(`VStar.Recursive.Sampled.protocol`, `verity/sampled-proofs/v1`: one-stage v2's audit on the proofs themselves, on the
blobs the caller hands in; one-stage v1, whose verdicts are the evidence's, and the recursive audit, which SPOT's
soundness doesn't cover, are refused), the
inference-only policy (`NciCheck.Protocol.protocol`,
`verity/inference-only/v0`: the declared circuit divides into inference blocks within their gate and fresh-input
budgets), work by proof of useful work over the period (`PouwCheck.Period.protocol`, `verity/pouw/period/v1`: the
period's windows credited all or nothing), space by PoUS's check of a run (`PousCheck.RunProtocol.protocol`), bandwidth
by the remote bandwidth audit's P2 session on PoUS's store (`RemoteAudit.P2Protocol.protocol`,
`verity/remote-bandwidth/p2/v0`), communication by the network certifier over the period
(`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the period's link-windows certified and proved, and the bound
`K` on what left the site), and the wipe by the memory challenger's streamed-label wipe of the period
(`MemoryChallenger.WipeProtocol.protocol`, `verity/wipe/v0`). `Core.Composition.tiedSameB` composes them on the caller's
blobs: every subprotocol accepts or onsite doesn't, and any subprotocol's rejection rejects onsite whatever the others
answer. The policy's commitment and sampled proofs' are the run's registration record (in either form, the one-stage
one), and both must commit to the same one; every window of the period must name that record as its Commit record,
remote's tie (`Core.Remote.windows`). Two records reject onsite whatever the subprotocols answer. PoUS and the remote
bandwidth audit must also be on one store (`store`): two stores reject onsite unless it rejects already.

A stated exception: the policy's v0 publishes the declared division and each inference block's profile (its gates and
fresh-input bits) in the clear. Onsite composes it for one run of record only, as an exception to
everything-zero-knowledge, and v1's private form (the division committed under hiding leaves, compliance proved, a
padded profile) replaces it.
-/
module

public import CoreComposition.Blobs
public import CoreComposition.Cases
public import CoreComposition.Tie
public import CoreRemote
public import MemoryChallenger.WipeProtocol
public import NciCheck.Protocol
public import NetworkCheck.Period
public import PousCheck.RunProtocol
public import PouwCheck.Period
public import RemoteAudit.P2Protocol
public import Spot.OneStage
public import Spot.OneStageZk
public import VStarRecursive.Sampled
meta import CoreComposition.Blobs
meta import CoreComposition.Cases
meta import CoreComposition.Tie
meta import CoreRemote
meta import MemoryChallenger.WipeProtocol
meta import NciCheck.Protocol
meta import NetworkCheck.Period
meta import PousCheck.RunProtocol
meta import PouwCheck.Period
meta import RemoteAudit.P2Protocol
meta import Spot.OneStage
meta import Spot.OneStageZk
meta import VStarRecursive.Sampled

@[expose] public section

namespace Core.Onsite

open Composition

/-- Onsite's subprotocols, in the order of their ids. -/
def components : List BlobProtocol :=
  [NetworkCheck.Period.protocol, NciCheck.Protocol.protocol, PousCheck.RunProtocol.protocol, PouwCheck.Period.protocol,
    RemoteAudit.P2Protocol.protocol, VStar.Recursive.Sampled.protocol, MemoryChallenger.WipeProtocol.protocol]

/-- The subprotocols that must commit to one record: the policy and sampled proofs. -/
def tied : List String := [NciCheck.Protocol.protocolId, VStar.Recursive.Sampled.protocol.id]

/-- PoUS and the remote bandwidth audit on one store: the root of `C*` PoUS's setup published (its commitment's `root`)
and the store the P2 session audits, the root its registration names (its commitment's `reg.root`, the case's
`store_root`), each read by its own decoder. -/
def store : Tie :=
  ⟨"store",
    [(PousCheck.RunProtocol.protocol.id,
        fun k => (PousCheck.RunProtocol.decodeCommitment k.commitment).toOption.map ([·.root])),
      (RemoteAudit.P2Protocol.protocol.id,
        fun k => (RemoteAudit.P2Protocol.decodeCommitment k.commitment).toOption.map ([·.held.reg.root]))]⟩

/-- Onsite's ties: every window of PoUW's period on the policy's record (remote's, `Core.Remote.windows`), and PoUS and
the remote bandwidth audit on one store. -/
def ties : List Tie := [Core.Remote.windows, store]

def protocol : BlobProtocol := tiedSameB "verity/onsite/v0" tied ties components

/-- A case its subprotocol's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : BlobProtocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (xs : List Case) (blobs : Array ByteArray := #[]) : Answer :=
  Cases.answerCases (protocol.on blobs) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) == ["verity/certifier/v0", "verity/inference-only/v0", "verity/pous/p2/v4/run",
  "verity/pouw/period/v1", "verity/remote-bandwidth/p2/v0", "verity/sampled-proofs/v1", "verity/wipe/v0"]
#guard tied.all fun s => (components.map (·.id)).contains s
#guard ties.all fun t => t.names.all fun n => (components.map (·.id)).contains n.1

-- every subprotocol refuses its case: onsite rejects on each, with or without blobs, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard components.all fun P => (allRefused.reason.map (contains · s!"{P.id} reject: params: ")).getD false
#guard (answer (components.map refused) #["not a circuit file".toUTF8]).reason == allRefused.reason
-- a part of onsite's subprotocols is not onsite's, onsite has no policy-less or wipe-less form, and neither v1's nor
-- v2's one-stage audit alone is onsite's integrity subprotocol
#guard (answer [refused PouwCheck.Period.protocol, refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/certifier/v0 is missing"
#guard (answer ((components.filter (·.id != NciCheck.Protocol.protocolId)).map refused)).reason ==
  some "params: verity/inference-only/v0 is missing"
#guard (answer ((components.take 6).map refused)).reason == some "params: verity/wipe/v0 is missing"
#guard (answer ((components.take 5).map refused ++
    [refused Spot.OneStage.protocol, refused MemoryChallenger.WipeProtocol.protocol])).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"
#guard (answer ((components.take 5).map refused ++
    [refused Spot.OneStageZk.protocol, refused MemoryChallenger.WipeProtocol.protocol])).reason ==
  some "params: verity/sampled-proofs/one-stage/v2 is not a component of this composition"
-- one-stage v1, whose verdicts are the evidence's, is refused by name: a one-stage statement naming no proofs is v1's,
-- and onsite rejects on it with sampled proofs' refusal (`VStar.Recursive.Sampled.V1_REFUSED`)
private def v1 : Answer := answer (components.map fun P =>
  if P.id == VStar.Recursive.Sampled.protocol.id then { refused P with params := "{\"form\":\"one-stage\",\"law\":{}}" }
  else refused P)
#guard v1.verdict == .reject && v1.published.isNone
#guard (v1.reason.map (contains · (s!"verity/sampled-proofs/v1 reject: params: the parameters name no proofs: " ++
  VStar.Recursive.Sampled.V1_REFUSED))).getD false

-- the policy and sampled proofs committing to one record leave each subprotocol to answer as itself; two records
-- reject onsite, naming both, whatever the subprotocols answer
private def record (p : String) : ByteArray :=
  ("{\"format\":\"verity/registration/v1\",\"law\":{},\"partition\":\"p\",\"population\":1,\"program\":\"" ++ p ++
    "\",\"query\":\"q\",\"roots\":[],\"scheme\":\"s\",\"window\":{\"context\":\"c\",\"kind\":\"k\",\"public\":{}}}").toUTF8

private def committed (policy audit : ByteArray) : Answer :=
  answer (components.map fun P =>
    if P.id == NciCheck.Protocol.protocolId then { refused P with commitment := policy }
    else if P.id == VStar.Recursive.Sampled.protocol.id then { refused P with commitment := audit }
    else refused P)

#guard (committed (record "x") (record "x")).reason == allRefused.reason
#guard (committed (record "y") (record "x")).verdict == .reject
#guard (committed (record "y") (record "x")).published.isNone
#guard (committed (record "y") (record "x")).reason.map
  (·.startsWith "record: verity/inference-only/v0 committed ") == some true
#guard (committed (record "y") (record "x")).reason.map
  (contains · ", verity/sampled-proofs/v1 committed ") == some true
#guard !((committed (record "y") (record "x")).reason.map (contains · "reject: params")).getD true

-- every window of PoUW's period on the policy's record: one record leaves the verdict to the subprotocols, and a window
-- on another record rejects onsite, naming both, whatever the subprotocols answer but a rejection, which stands
/-- A registration record every record reader decodes, a window's Commit record among them. -/
private def registration (n : Nat) : String := s!"\{\"format\":\"verity/registration/v2\",\"n\":{n},\"registered\":[]}"

private def windowOn (r : String) : String := s!"\{\"commit\":\"{Flock.hexOf r.toUTF8}\"}"

/-- Onsite's cases, each refused, with the policy's committing to `policy` and the period's windows on the records
`rs`. -/
private def windowsOn (policy : String) (rs : List String) : List Case :=
  components.map fun P =>
    if P.id == NciCheck.Protocol.protocolId then { refused P with commitment := policy.toUTF8 }
    else if P.id == PouwCheck.Period.protocolId then
      { refused P with evidence := ("{\"pins\":[],\"windows\":[" ++ ",".intercalate (rs.map windowOn) ++ "]}").toUTF8 }
    else refused P

/-- Why onsite's ties reject the outcome `o` its subprotocols reached on `windowsOn policy rs`; `none` when they hold
it. -/
private def stray (policy : String) (rs : List String) (o : Outcome Keyed Keyed) : Option String :=
  match holdTies ties (windowsOn policy rs) o with
  | .reject r => some r
  | _ => none

private def answered : List (Outcome Keyed Keyed) := [.accept [] [], .undecided "u"]

#guard answered.all fun o => stray (registration 0) [registration 0, registration 0] o == none
-- a window on another record
#guard answered.all fun o => (stray (registration 0) [registration 0, registration 1] o).map
  (·.startsWith "record: verity/inference-only/v0 committed ") == some true
#guard answered.all fun o => (stray (registration 0) [registration 0, registration 1] o).map
  (contains · ", verity/pouw/period/v1 committed ") == some true
-- every window on a record that isn't the policy's
#guard answered.all fun o => (stray (registration 1) [registration 0] o).isSome
-- a window naming no Commit record leaves the period to refuse it
#guard answered.all fun o => stray (registration 0) [registration 0, "{}"] o == none
-- a subprotocol's rejection stands, and onsite's answer on those cases is its subprotocols' refusals
#guard stray (registration 0) [registration 1] (.reject "r") == some "r"
#guard (answer (windowsOn (registration 0) [registration 1])).reason.map
  (·.startsWith "verity/certifier/v0 reject: params: ") == some true

end Core.Onsite
