/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space, bandwidth and communication. Its components are sampled proofs on the proofs themselves
(`VStar.Recursive.Sampled.protocol`, `verity/sampled-proofs/v1`: the form the params name, one-stage v2's
audit, flock-verify `--zk` run on each session's files, or the recursive audit, V*'s compiled check run on the recursive
session's parts, each on the files the caller hands in as blobs), work by proof of useful work (`PouwCheck.protocol`), space by
PoUS's check of a run (`PousCheck.RunProtocol.protocol`), bandwidth by the remote bandwidth audit's session
(`RemoteAudit.Protocol.protocol`), and communication by the network certifier over the period
(`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the period's link-windows certified and proved, and the bound `K`
on what left the site). It composes them on the caller's blobs as `Core.Composition.productB` does: every component
accepts or onsite doesn't, and any component's rejection rejects onsite whatever the others answer. Before composing, in
its commit phase, onsite checks the registration by NCI's check of record (`NciCheck.Protocol.protocol`, `verity/nci/v1`,
a registration check and not a component): a registration it refuses refuses onsite (`Core.Composition.registered`), and
so does one other than the registration record sampled proofs' component commits to: one registration digest in both.
-/
import CoreComposition.Blobs
import CoreComposition.Cases
import NciCheck.Protocol
import NetworkCheck.Period
import PousCheck.RunProtocol
import PouwCheck.Protocol
import RemoteAudit.Protocol
import SampledProofs.OneStageZk
import VStarRecursive.Sampled

namespace Core.Onsite

open Composition

/-- Onsite's components, in the order of their ids. -/
def components : List BlobProtocol :=
  [NetworkCheck.Period.protocol, PousCheck.RunProtocol.protocol, PouwCheck.protocol, RemoteAudit.Protocol.protocol,
    VStar.Recursive.Sampled.protocol]

/-- The component whose commitment is the registration record NCI checks. -/
def committer : String := VStar.Recursive.Sampled.protocol.id

/-- `registered` of the components after NCI's registration check, whose check on blobs `bs` is `checkRegistered` of the
components on `bs`. -/
def protocol : BlobProtocol :=
  { registered "verity/onsite/v0" NciCheck.Protocol.protocol committer (components.map (·.toProtocol)) with
    checkBlobs := fun ps c ev bs =>
      checkRegistered "verity/onsite/v0" NciCheck.Protocol.protocol committer (components.map (·.onBlobs bs)) ps c ev }

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : BlobProtocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

/-- The answer of what onsite composes once NCI accepts the registration. -/
private def answer (xs : List Case) (blobs : Array ByteArray := #[]) : Answer :=
  Cases.answerCases ((productB protocol.id components).on blobs) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) == ["verity/certifier/v0", "verity/pous/p2/v4/run", "verity/pouw/v1",
  "verity/remote-bandwidth/v0", "verity/sampled-proofs/v1"]

-- every component refuses its case: onsite rejects on each, with or without blobs, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard components.all fun P => (allRefused.reason.map (contains · s!"{P.id} reject: params: ")).getD false
#guard (answer (components.map refused) #["not a circuit file".toUTF8]).reason == allRefused.reason
-- remote's components alone are not onsite's, and neither v1's nor v2's audit alone is onsite's sampled proofs component
#guard (answer [refused PouwCheck.protocol, refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/certifier/v0 is missing"
#guard (answer ((components.take 4).map refused ++ [refused SampledProofs.OneStage.protocol])).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"
#guard (answer ((components.take 4).map refused ++ [refused SampledProofs.OneStageZk.protocol])).reason ==
  some "params: verity/sampled-proofs/one-stage/v2 is not a component of this composition"

-- a registration NCI refuses refuses onsite before any component is run, with or without blobs, and onsite's params
-- carry the registration
private def unregisteredOn (blobs : Array ByteArray) : Answer :=
  Cases.answerRegistered (protocol.on blobs) (refused NciCheck.Protocol.protocol) (components.map refused)
    ⟨(List.replicate 128 0).toArray⟩
private def unregistered : Answer := unregisteredOn #[]
#guard unregistered.verdict == .reject && unregistered.published.isNone
#guard unregistered.reason.map (·.startsWith "registration: verity/nci/v1 reject: params: ") == some true
#guard !components.any fun P => (unregistered.reason.map (contains · s!"{P.id} reject")).getD false
#guard (unregisteredOn #["not a circuit file".toUTF8]).reason == unregistered.reason
#guard (Cases.answerCases (protocol.on #[]) (components.map refused) ⟨(List.replicate 128 0).toArray⟩).reason ==
  some "params: the composition's params: the keys must be [components, registration], not [components]"
-- the registration NCI checks is the record sampled proofs' component commits to, by its digest
#guard decide (committedBy committer (components.map (·.toProtocol))
  (components.map fun P => { refused P with commitment := "{}".toUTF8 }) =
  (SampledProofs.Registration.digest (Lean.Json.mkObj [])).toOption)

end Core.Onsite
