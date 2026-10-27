/-
The toy deployment (`verity/toy/v0`): the verifier of record of `integrations/torch_loop`, served under sampled proofs
plus the network certifier. Its components are integrity by sampled proofs' one-stage audit on the proofs themselves
(`SampledProofs.OneStageZk.protocol`, `verity/sampled-proofs/one-stage/v2`) and communication by the network certifier
over the period (`NetworkCheck.Period.protocol`, `verity/certifier/v0`). Its commit phase is onsite's: NCI's check of
record (`NciCheck.Protocol.protocol`, `verity/nci/v1`) on the registration sampled proofs' one-stage audit commits to
(`Core.Composition.registeredB`). Composed on the caller's blobs, both components accept or the toy doesn't, and either
one's rejection rejects the toy whatever the other answers.
-/
import CoreComposition.Blobs
import CoreComposition.Cases
import NciCheck.Protocol
import NetworkCheck.Period
import SampledProofs.OneStageZk

namespace Core.Toy

open Composition

/-- The toy's components, in the order of their ids. -/
def components : List BlobProtocol := [NetworkCheck.Period.protocol, SampledProofs.OneStageZk.protocol]

def protocol : BlobProtocol :=
  registeredB "verity/toy/v0" NciCheck.Protocol.protocol SampledProofs.OneStageZk.protocol.id components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : BlobProtocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

/-- The answer of what the toy composes once NCI accepts the registration. -/
private def answer (xs : List Case) (blobs : Array ByteArray := #[]) : Answer :=
  Cases.answerCases ((productB protocol.id components).on blobs) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) == ["verity/certifier/v0", "verity/sampled-proofs/one-stage/v2"]
#guard protocol.id == "verity/toy/v0"

-- both components refuse their cases: the toy rejects on each, with or without blobs, and makes nothing public
private def bothRefused : Answer := answer (components.map refused)
#guard bothRefused.verdict == .reject
#guard bothRefused.published.isNone && bothRefused.profile.isNone
#guard bothRefused.reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard components.all fun P => (bothRefused.reason.map (contains · s!"{P.id} reject: params: ")).getD false
#guard (answer (components.map refused) #["not a circuit file".toUTF8]).reason == bothRefused.reason
-- sampled proofs alone is not the toy, and v1's audit is not the toy's integrity component
#guard (answer [refused SampledProofs.OneStageZk.protocol]).reason == some "params: verity/certifier/v0 is missing"
#guard (answer [refused NetworkCheck.Period.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v2 is missing"
#guard (answer [refused NetworkCheck.Period.protocol, refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"

-- a registration NCI refuses refuses the toy before any component is run, and the toy's params carry the registration
private def unregistered (blobs : Array ByteArray := #[]) : Answer :=
  Cases.answerRegistered (protocol.on blobs) (refused NciCheck.Protocol.protocol) (components.map refused)
    ⟨(List.replicate 128 0).toArray⟩
#guard (unregistered).verdict == .reject && (unregistered).published.isNone
#guard (unregistered).reason.map (·.startsWith "registration: verity/nci/v1 reject: params: ") == some true
#guard !components.any fun P => ((unregistered).reason.map (contains · s!"{P.id} reject")).getD false
#guard (unregistered #["not a circuit file".toUTF8]).reason == (unregistered).reason
#guard (Cases.answerCases (protocol.on #[]) (components.map refused) ⟨(List.replicate 128 0).toArray⟩).reason ==
  some "params: the composition's params: the keys must be [components, registration], not [components]"

end Core.Toy
