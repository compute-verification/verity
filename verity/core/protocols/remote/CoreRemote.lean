/-
Remote Verity (`verity/remote/v0`): what the verifier establishes with no hardware of its own at the site. Its
components are integrity by sampled proofs' one-stage audit on the proofs themselves
(`SampledProofs.OneStageZk.protocol`, `verity/sampled-proofs/one-stage/v2`: flock-verify `--zk` run on each session's
files, which the caller hands in as blobs) and work by proof of useful work (`PouwCheck.protocol`).
`Core.Composition.productB` composes them on the caller's blobs: both accept or remote doesn't, and either one's
rejection rejects remote whatever the other answers.
-/
import CoreComposition.Blobs
import CoreComposition.Cases
import PouwCheck.Protocol
import SampledProofs.OneStageZk

namespace Core.Remote

open Composition

/-- Remote's components, in the order of their ids. -/
def components : List BlobProtocol := [PouwCheck.protocol, SampledProofs.OneStageZk.protocol]

def protocol : BlobProtocol := productB "verity/remote/v0" components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : BlobProtocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (xs : List Case) : Answer := Cases.answerCases (protocol.on #[]) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) == ["verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"]

-- both refuse their cases: remote rejects on both, and makes nothing public
private def bothRefused : Answer := answer [refused PouwCheck.protocol, refused SampledProofs.OneStageZk.protocol]
#guard bothRefused.verdict == .reject
#guard bothRefused.published.isNone && bothRefused.profile.isNone
#guard bothRefused.reason.map (·.startsWith "verity/pouw/v1 reject: params: ") == some true
#guard bothRefused.reason.map (contains · "; verity/sampled-proofs/one-stage/v2 reject: params: ") == some true
#guard (answer [refused PouwCheck.protocol]).reason == some "params: verity/sampled-proofs/one-stage/v2 is missing"
-- v1's audit is not remote's integrity component
#guard (answer [refused PouwCheck.protocol, refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"

end Core.Remote
