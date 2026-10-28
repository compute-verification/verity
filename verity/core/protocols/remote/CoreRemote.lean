/-
Remote Verity (`verity/remote/v0`): what the verifier establishes with no hardware of its own at the site. Its
components are integrity by sampled proofs' one-stage audit (`SampledProofs.OneStage.protocol`) and work by proof of
useful work (`PouwCheck.protocol`). `Core.Composition.product` composes them: both accept or remote doesn't, and either
one's rejection rejects remote whatever the other answers.
-/
import CoreComposition.Cases
import PouwCheck.Protocol
import SampledProofs.OneStage

namespace Core.Remote

open Composition

/-- Remote's components, in the order of their ids. -/
def components : List Protocol := [PouwCheck.protocol, SampledProofs.OneStage.protocol]

def protocol : Protocol := product "verity/remote/v0" components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : Protocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (xs : List Case) : Answer := Cases.answerCases protocol xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard ordered components
#guard components.map (·.id) == ["verity/pouw/v1", "verity/sampled-proofs/one-stage/v1"]

-- both refuse their cases: remote rejects on both, and makes nothing public
private def bothRefused : Answer := answer [refused PouwCheck.protocol, refused SampledProofs.OneStage.protocol]
#guard bothRefused.verdict == .reject
#guard bothRefused.published.isNone && bothRefused.profile.isNone
#guard bothRefused.reason.map (·.startsWith "verity/pouw/v1 reject: params: ") == some true
#guard bothRefused.reason.map (contains · "; verity/sampled-proofs/one-stage/v1 reject: params: ") == some true
#guard (answer [refused PouwCheck.protocol]).reason == some "params: verity/sampled-proofs/one-stage/v1 is missing"

end Core.Remote
