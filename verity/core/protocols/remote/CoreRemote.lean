/-
Remote Verity (`verity/remote/v0`): what the verifier establishes with no hardware of its own at the site. Its
subprotocols are integrity by sampled proofs' one-stage audit (`SampledProofs.OneStage.protocol`), work by proof of
useful work (`PouwCheck.protocol`), and the inference-only policy (`NciCheck.Protocol.protocol`,
`verity/inference-only/v0`: the declared circuit divides into inference blocks within their gate and fresh-input
budgets), which reads only the declared circuit and the registered information and so needs no hardware at the site.
`Core.Composition.product` composes them: every subprotocol accepts or remote doesn't, and any subprotocol's rejection
rejects remote whatever the others answer.
-/
import CoreComposition.Cases
import NciCheck.Protocol
import PouwCheck.Protocol
import SampledProofs.OneStage

namespace Core.Remote

open Composition

/-- Remote's components, in the order of their ids. -/
def components : List Protocol := [NciCheck.Protocol.protocol, PouwCheck.protocol, SampledProofs.OneStage.protocol]

def protocol : Protocol := product "verity/remote/v0" components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : Protocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (xs : List Case) : Answer := Cases.answerCases protocol xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard ordered components
#guard components.map (·.id) == ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v1"]

-- every subprotocol refuses its case: remote rejects on each, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/inference-only/v0 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/pouw/v1 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/sampled-proofs/one-stage/v1 reject: params: ") == some true
#guard (answer [refused PouwCheck.protocol, refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/inference-only/v0 is missing"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is missing"

end Core.Remote
