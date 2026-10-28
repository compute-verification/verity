/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space, bandwidth and communication. Its subprotocols are integrity by sampled proofs' one-stage audit on the
proofs themselves (`SampledProofs.OneStageZk.protocol`, `verity/sampled-proofs/one-stage/v2`: flock-verify `--zk` run on
each session's files, which the caller hands in as blobs), the inference-only policy (`NciCheck.Protocol.protocol`,
`verity/inference-only/v0`: the declared circuit divides into inference blocks within their gate and fresh-input
budgets), work by proof of useful work over the period (`PouwCheck.Period.protocol`, `verity/pouw/period/v1`: the
period's windows credited all or nothing), space by PoUS's check of a run (`PousCheck.RunProtocol.protocol`), bandwidth
by the remote bandwidth audit's session (`RemoteAudit.Protocol.protocol`), communication by the network certifier over
the period (`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the period's link-windows certified and proved, and
the bound `K` on what left the site), and the wipe by the memory challenger's streamed-label wipe of the period
(`MemoryChallenger.WipeProtocol.protocol`, `verity/wipe/v0`). `Core.Composition.sameB` composes them on the caller's
blobs: every subprotocol accepts or onsite doesn't, and any subprotocol's rejection rejects onsite whatever the others
answer. The policy's commitment and the one-stage audit's are the run's registration record, and both must commit to
the same one: two records' digests reject onsite whatever the subprotocols answer.

A stated exception: the policy's v0 publishes the declared division and each inference block's profile (its gates and
fresh-input bits) in the clear. Onsite composes it for the run of record of 28 Oct only, as an exception to
everything-zero-knowledge, by Daniel's ruling: "Let's run it in the clear tonight as a stated exception." v1's
private form (the division committed under hiding leaves, compliance proved, a padded profile) replaces it.
-/
import CoreComposition.Blobs
import CoreComposition.Cases
import MemoryChallenger.WipeProtocol
import NciCheck.Protocol
import NetworkCheck.Period
import PousCheck.RunProtocol
import PouwCheck.Period
import RemoteAudit.Protocol
import SampledProofs.OneStageZk

namespace Core.Onsite

open Composition

/-- Onsite's components, in the order of their ids. -/
def components : List BlobProtocol :=
  [NetworkCheck.Period.protocol, NciCheck.Protocol.protocol, PousCheck.RunProtocol.protocol, PouwCheck.Period.protocol,
    RemoteAudit.Protocol.protocol, SampledProofs.OneStageZk.protocol, MemoryChallenger.WipeProtocol.protocol]

/-- The components that must commit to one record: the policy and the one-stage audit. -/
def tied : List String := [NciCheck.Protocol.protocolId, SampledProofs.OneStageZk.protocol.id]

def protocol : BlobProtocol := sameB "verity/onsite/v0" tied components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : BlobProtocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (xs : List Case) (blobs : Array ByteArray := #[]) : Answer :=
  Cases.answerCases (protocol.on blobs) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) == ["verity/certifier/v0", "verity/inference-only/v0", "verity/pous/p2/v4/run",
  "verity/pouw/period/v1", "verity/remote-bandwidth/v0", "verity/sampled-proofs/one-stage/v2", "verity/wipe/v0"]
#guard tied.all fun s => (components.map (·.id)).contains s

-- every component refuses its case: onsite rejects on each, with or without blobs, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard components.all fun P => (allRefused.reason.map (contains · s!"{P.id} reject: params: ")).getD false
#guard (answer (components.map refused) #["not a circuit file".toUTF8]).reason == allRefused.reason
-- a part of onsite's components is not onsite's, onsite has no policy-less or wipe-less form, and v1's audit is not
-- onsite's integrity component
#guard (answer [refused PouwCheck.Period.protocol, refused SampledProofs.OneStageZk.protocol]).reason ==
  some "params: verity/certifier/v0 is missing"
#guard (answer ((components.filter (·.id != NciCheck.Protocol.protocolId)).map refused)).reason ==
  some "params: verity/inference-only/v0 is missing"
#guard (answer ((components.take 6).map refused)).reason == some "params: verity/wipe/v0 is missing"
#guard (answer ((components.take 5).map refused ++
    [refused SampledProofs.OneStage.protocol, refused MemoryChallenger.WipeProtocol.protocol])).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"

-- the policy and the one-stage audit committing to one record leave each component to answer as itself; two records
-- reject onsite, naming both, whatever the components answer
private def record (p : String) : ByteArray :=
  ("{\"format\":\"verity/registration/v1\",\"law\":{},\"partition\":\"p\",\"population\":1,\"program\":\"" ++ p ++
    "\",\"query\":\"q\",\"roots\":[],\"scheme\":\"s\",\"window\":{\"context\":\"c\",\"kind\":\"k\",\"public\":{}}}").toUTF8

private def committed (policy audit : ByteArray) : Answer :=
  answer (components.map fun P =>
    if P.id == NciCheck.Protocol.protocolId then { refused P with commitment := policy }
    else if P.id == SampledProofs.OneStageZk.protocol.id then { refused P with commitment := audit }
    else refused P)

#guard (committed (record "x") (record "x")).reason == allRefused.reason
#guard (committed (record "y") (record "x")).verdict == .reject
#guard (committed (record "y") (record "x")).published.isNone
#guard (committed (record "y") (record "x")).reason.map
  (·.startsWith "record: verity/inference-only/v0 committed ") == some true
#guard (committed (record "y") (record "x")).reason.map
  (contains · ", verity/sampled-proofs/one-stage/v2 committed ") == some true
#guard !((committed (record "y") (record "x")).reason.map (contains · "reject: params")).getD true

end Core.Onsite
