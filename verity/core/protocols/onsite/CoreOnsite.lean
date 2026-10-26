/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space, bandwidth and communication. Its components are integrity by sampled proofs' one-stage audit
(`SampledProofs.OneStage.protocol`), work by proof of useful work (`PouwCheck.protocol`), space by proof of useful space,
bandwidth by proof of useful bandwidth, and communication twice: by the network certificate check
(`NetworkCheck.Check.protocol`), and by the NCI policy's check of record (`NciCheck.Protocol.protocol`), that the
declaration divides into isolation units each reading at most `α` bits of novel information per gate; space and
bandwidth are stubs (`Core.Composition.Stub`) until their ports land. `Core.Composition.product` composes them: every
component accepts or onsite doesn't, and any component's rejection rejects onsite whatever the others answer.
-/
import CoreComposition.Cases
import CoreComposition.Stubs
import NciCheck.Protocol
import NetworkCheck.Check
import PouwCheck.Protocol
import SampledProofs.OneStage

namespace Core.Onsite

open Composition

/-- Onsite's components, in the order of their ids. -/
def components : List Protocol :=
  [NciCheck.Protocol.protocol, NetworkCheck.Check.protocol, PouwCheck.protocol, SampledProofs.OneStage.protocol,
    Stub.bandwidth, Stub.pous]

def protocol : Protocol := product "verity/onsite/v0" components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : Protocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (xs : List Case) : Answer := Cases.answerCases protocol xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard ordered components
#guard components.map (·.id) == ["verity/nci/v1", "verity/network-certificate/v4", "verity/pouw/v1",
  "verity/sampled-proofs/one-stage/v1", "verity/stub/bandwidth/v0", "verity/stub/pous/v0"]

-- NCI, communication, work and integrity refuse their cases while space is undecided: onsite rejects on all four, and
-- makes nothing public
private def refusedBeside : Answer :=
  answer [refused NciCheck.Protocol.protocol, refused NetworkCheck.Check.protocol, refused PouwCheck.protocol,
    refused SampledProofs.OneStage.protocol, (Cases.accepts Stub.bandwidth.id).case, (Cases.undecided Stub.pous.id).case]
#guard refusedBeside.verdict == .reject
#guard refusedBeside.published.isNone && refusedBeside.profile.isNone
#guard refusedBeside.reason.map (·.startsWith "verity/nci/v1 reject: params: ") == some true
#guard refusedBeside.reason.map (contains · "; verity/network-certificate/v4 reject: params: ") == some true
#guard refusedBeside.reason.map (contains · "; verity/pouw/v1 reject: params: ") == some true
#guard refusedBeside.reason.map (contains · "; verity/sampled-proofs/one-stage/v1 reject: params: ") == some true
#guard !(refusedBeside.reason.map (contains · "verity/stub/pous/v0")).getD true
-- remote's components alone are not onsite's
#guard (answer [refused PouwCheck.protocol, refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/nci/v1 is missing"

end Core.Onsite
