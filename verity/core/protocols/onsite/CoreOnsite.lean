/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space, bandwidth and communication. Its components are integrity by sampled proofs' one-stage audit
(`SampledProofs.OneStage.protocol`), work by proof of useful work over the period (`PouwCheck.Period.protocol`,
`verity/pouw/period/v1`: the period's windows credited all or nothing), space by PoUS's check of a run
(`PousCheck.RunProtocol.protocol`), bandwidth by the remote bandwidth audit's session (`RemoteAudit.Protocol.protocol`),
and communication by the network certifier over the period (`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the
period's link-windows certified and proved, and the bound `K` on what left the site). `Core.Composition.product` composes
them: every component accepts or onsite doesn't, and any component's rejection rejects onsite whatever the others answer.
-/
import CoreComposition.Cases
import NetworkCheck.Period
import PousCheck.RunProtocol
import PouwCheck.Period
import RemoteAudit.Protocol
import SampledProofs.OneStage

namespace Core.Onsite

open Composition

/-- Onsite's components, in the order of their ids. -/
def components : List Protocol :=
  [NetworkCheck.Period.protocol, PousCheck.RunProtocol.protocol, PouwCheck.Period.protocol, RemoteAudit.Protocol.protocol,
    SampledProofs.OneStage.protocol]

def protocol : Protocol := product "verity/onsite/v0" components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : Protocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (xs : List Case) : Answer := Cases.answerCases protocol xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard ordered components
#guard components.map (·.id) == ["verity/certifier/v0", "verity/pous/p2/v4/run", "verity/pouw/period/v1",
  "verity/remote-bandwidth/v0", "verity/sampled-proofs/one-stage/v1"]

-- every component refuses its case: onsite rejects on each, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard components.all fun P => (allRefused.reason.map (contains · s!"{P.id} reject: params: ")).getD false
-- a part of onsite's components is not onsite's
#guard (answer [refused PouwCheck.Period.protocol, refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/certifier/v0 is missing"

end Core.Onsite
