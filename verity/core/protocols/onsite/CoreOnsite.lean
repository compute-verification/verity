/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space, bandwidth and communication. Its components are integrity by sampled proofs' one-stage audit
(`SampledProofs.OneStage.protocol`), work by proof of useful work over the period (`PouwCheck.Period.protocol`,
`verity/pouw/period/v1`: the period's windows credited all or nothing), space by PoUS's check of a run
(`PousCheck.RunProtocol.protocol`), bandwidth by the remote bandwidth audit's session (`RemoteAudit.Protocol.protocol`),
and communication by the network certifier over the period (`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the
period's link-windows certified and proved, and the bound `K` on what left the site). `Core.Composition.product` composes
them: every component accepts or onsite doesn't, and any component's rejection rejects onsite whatever the others answer.
Before composing, in its commit phase, onsite checks the registration by NCI's check of record
(`NciCheck.Protocol.protocol`, `verity/inference-only/v0`, a registration check and not a component): a registration
it refuses refuses onsite (`Core.Composition.registered`), and so does one other than the registration record sampled
proofs' one-stage audit commits to: one registration digest in both.
-/
import CoreComposition.Cases
import NciCheck.Protocol
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

def protocol : Protocol :=
  registered "verity/onsite/v0" NciCheck.Protocol.protocol SampledProofs.OneStage.protocol.id components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : Protocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

/-- The answer of what onsite composes once NCI accepts the registration. -/
private def answer (xs : List Case) : Answer :=
  Cases.answerCases (product protocol.id components) xs ⟨(List.replicate 64 0).toArray⟩

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

-- a registration NCI refuses refuses onsite before any component is run, and onsite's params carry the registration
private def unregistered : Answer :=
  Cases.answerRegistered protocol (refused NciCheck.Protocol.protocol) (components.map refused)
    ⟨(List.replicate 128 0).toArray⟩
#guard unregistered.verdict == .reject && unregistered.published.isNone
#guard unregistered.reason.map (·.startsWith "registration: verity/inference-only/v0 reject: params: ") == some true
#guard !components.any fun P => (unregistered.reason.map (contains · s!"{P.id} reject")).getD false
#guard (Cases.answerCases protocol (components.map refused) ⟨(List.replicate 128 0).toArray⟩).reason ==
  some "params: the composition's params: the keys must be [components, registration], not [components]"
-- the registration NCI checks is the record sampled proofs' one-stage audit commits to, by its digest
#guard decide (committedBy SampledProofs.OneStage.protocol.id components
  (components.map fun P => { refused P with commitment := "{}".toUTF8 }) =
  (SampledProofs.Registration.digest (Lean.Json.mkObj [])).toOption)

end Core.Onsite
