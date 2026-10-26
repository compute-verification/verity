/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space, bandwidth and communication. Its components are integrity by sampled proofs' one-stage audit on the
proofs themselves (`SampledProofs.OneStageZk.protocol`, `verity/sampled-proofs/one-stage/v2`: flock-verify `--zk` run on
each session's files, which the caller hands in as blobs), work by proof of useful work (`PouwCheck.protocol`), space by
PoUS's check of a run (`PousCheck.RunProtocol.protocol`), bandwidth by the remote bandwidth audit's session
(`RemoteAudit.Protocol.protocol`), communication by the network certifier over the period
(`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the period's link-windows certified, and proved under one-stage
v2 on the same blobs, and the bound `K` on what left the site), and the wipe by the memory challenger's streamed-label wipe of the period
(`MemoryChallenger.WipeProtocol.protocol`, `verity/wipe/v0`). `Core.Composition.productB` composes them on the caller's
blobs: every component accepts or onsite doesn't, and any component's rejection rejects onsite whatever the others
answer.
-/
import CoreComposition.Blobs
import CoreComposition.Cases
import MemoryChallenger.WipeProtocol
import NetworkCheck.Period
import PousCheck.RunProtocol
import PouwCheck.Protocol
import RemoteAudit.Protocol
import SampledProofs.OneStageZk

namespace Core.Onsite

open Composition

/-- Onsite's components, in the order of their ids. -/
def components : List BlobProtocol :=
  [NetworkCheck.Period.protocol, PousCheck.RunProtocol.protocol, PouwCheck.protocol, RemoteAudit.Protocol.protocol,
    SampledProofs.OneStageZk.protocol, MemoryChallenger.WipeProtocol.protocol]

def protocol : BlobProtocol := productB "verity/onsite/v0" components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : BlobProtocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (xs : List Case) (blobs : Array ByteArray := #[]) : Answer :=
  Cases.answerCases (protocol.on blobs) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) == ["verity/certifier/v0", "verity/pous/p2/v4/run", "verity/pouw/v1",
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
-- remote's components alone are not onsite's, onsite has no wipe-less form, and v1's audit is not onsite's integrity
-- component
#guard (answer [refused PouwCheck.protocol, refused SampledProofs.OneStageZk.protocol]).reason ==
  some "params: verity/certifier/v0 is missing"
#guard (answer ((components.take 5).map refused)).reason == some "params: verity/wipe/v0 is missing"
#guard (answer ((components.take 4).map refused ++
    [refused SampledProofs.OneStage.protocol, refused MemoryChallenger.WipeProtocol.protocol])).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"

end Core.Onsite
