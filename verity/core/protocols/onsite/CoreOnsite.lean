/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for PoUS, the bandwidth audit, the network certifier and the wipe. Its components are sampled proofs' one-stage
audit on the proofs themselves (`SampledProofs.OneStageZk.protocol`, `verity/sampled-proofs/one-stage/v2`: flock-verify
`--zk` run on each session's files, which the caller hands in as blobs), PoUW over the period
(`PouwCheck.Period.protocol`, `verity/pouw/period/v1`: the period's windows credited all or nothing), PoUS's check of a
run (`PousCheck.RunProtocol.protocol`), the remote bandwidth audit's session (`RemoteAudit.Protocol.protocol`), the
network certifier over the period (`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the period's link-windows
certified and proved, and the bound `K` on what left the site), and the memory challenger's streamed-label wipe of the
period (`MemoryChallenger.WipeProtocol.protocol`, `verity/wipe/v0`). Composed on the caller's blobs, every component accepts
or onsite doesn't, and any component's rejection rejects onsite whatever the others answer. Before composing, in its
commit phase, onsite checks the registration by NCI's check of record (`NciCheck.Protocol.protocol`, `verity/nci/v1`, a
registration check and not a component): a registration it refuses refuses onsite, and so does one other than the
registration sampled proofs' one-stage audit commits to (`Core.Composition.registeredB`).
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
  [NetworkCheck.Period.protocol, PousCheck.RunProtocol.protocol, PouwCheck.Period.protocol, RemoteAudit.Protocol.protocol,
    SampledProofs.OneStageZk.protocol, MemoryChallenger.WipeProtocol.protocol]

def protocol : BlobProtocol :=
  registeredB "verity/onsite/v0" NciCheck.Protocol.protocol SampledProofs.OneStageZk.protocol.id components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : BlobProtocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

/-- The answer of what onsite composes once NCI accepts the registration. -/
private def answer (xs : List Case) (blobs : Array ByteArray := #[]) : Answer :=
  Cases.answerCases ((productB protocol.id components).on blobs) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) == ["verity/certifier/v0", "verity/pous/p2/v4/run", "verity/pouw/period/v1",
  "verity/remote-bandwidth/v0", "verity/sampled-proofs/one-stage/v2", "verity/wipe/v0"]

-- every component refuses its case: onsite rejects on each, with or without blobs, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard components.all fun P => (allRefused.reason.map (contains · s!"{P.id} reject: params: ")).getD false
#guard (answer (components.map refused) #["not a circuit file".toUTF8]).reason == allRefused.reason
-- a refused wipe refuses onsite
#guard (allRefused.reason.map (contains · "verity/wipe/v0 reject: params: ")).getD false
-- a part of onsite's components is not onsite's, onsite has no wipe-less form, and v1's audit is not one of onsite's
-- components
#guard (answer [refused PouwCheck.Period.protocol, refused SampledProofs.OneStageZk.protocol]).reason ==
  some "params: verity/certifier/v0 is missing"
#guard (answer ((components.take 5).map refused)).reason == some "params: verity/wipe/v0 is missing"
#guard (answer ((components.take 4).map refused ++
    [refused SampledProofs.OneStage.protocol, refused MemoryChallenger.WipeProtocol.protocol])).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"

-- a registration NCI refuses refuses onsite before any component is run, and onsite's params carry the registration
private def unregistered (blobs : Array ByteArray := #[]) : Answer :=
  Cases.answerRegistered (protocol.on blobs) (refused NciCheck.Protocol.protocol) (components.map refused)
    ⟨(List.replicate 128 0).toArray⟩
#guard (unregistered).verdict == .reject && (unregistered).published.isNone
#guard (unregistered).reason.map (·.startsWith "registration: verity/nci/v1 reject: params: ") == some true
#guard !components.any fun P => ((unregistered).reason.map (contains · s!"{P.id} reject")).getD false
#guard (unregistered #["not a circuit file".toUTF8]).reason == (unregistered).reason
#guard (Cases.answerCases (protocol.on #[]) (components.map refused) ⟨(List.replicate 128 0).toArray⟩).reason ==
  some "params: the composition's params: the keys must be [components, registration], not [components]"

end Core.Onsite
