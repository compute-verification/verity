/-
Remote with the network certifier (`verity/remote-network/v0`): remote's accounting (`Core.Remote.protocol`) and
communication through the auditor's network certifiers at the site's boundary, with no memory challenger. Its
subprotocols are remote's three (`Core.Remote.components`: the inference-only policy, PoUW's period credit and
sampled proofs) and the network certifier over the period (`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the
period's link-windows certified and proved, and the bound `K` on what left the site). `Core.Composition.tiedSameB`
composes them on the caller's blobs under remote's record subprotocols (`Core.Remote.tied`) and remote's one tie
(`Core.Remote.windows`), with no tie of its own, so its verdict is the AND of remote's and the network certifier's: it
accepts when both accept, and either's rejection rejects it whatever the other answers.

Its registration is onsite's boundary condition without onsite's memory condition: every boundary link split by one of
the auditor's network certifiers (`uncertified`), and no memory challenger's reach (`challengerReach`).

The network certifier's period reads its audits' answers as its evidence carries them; nothing here holds them to
sampled proofs' answer (`NetworkCheck.Period`'s "What it doesn't check"). The policy's v0 publishes the declared
division and each inference block's profile in the clear, remote's stated exception (`Core.Remote`).
-/
import CoreComposition.Cases
import CoreComposition.Stubs
import CoreRemote
import NetworkCheck.Period

namespace Core.RemoteNetwork

open Composition

def protocolId : String := "verity/remote-network/v0"

/-- The subprotocols, in the order of their ids: the network certifier's period, then remote's. -/
def components : List BlobProtocol := NetworkCheck.Period.protocol :: Core.Remote.components

def protocol : BlobProtocol := tiedSameB protocolId Core.Remote.tied [Core.Remote.windows] components

/-- A case of the subprotocol `id` that its own `answer` refuses: empty parameters, commitment and evidence. -/
private def refusedAs (id : String) : Case :=
  { id := id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def refused (P : BlobProtocol) : Case := refusedAs P.id

private def answer (xs : List Case) : Answer := Cases.answerCases (protocol.on #[]) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) ==
  ["verity/certifier/v0", "verity/inference-only/v0", "verity/pouw/period/v1", "verity/sampled-proofs/v1"]
#guard (components.drop 1).map (·.id) == Core.Remote.components.map (·.id)
#guard Core.Remote.tied.all fun s => (components.map (·.id)).contains s
#guard Core.Remote.windows.names.all fun n => (components.map (·.id)).contains n.1

-- every subprotocol refuses its case: the composite rejects on each, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard components.all fun P => (allRefused.reason.map (contains · s!"{P.id} reject: params: ")).getD false
-- remote's subprotocols alone are not this composite's, nor is the network certifier's period alone
#guard (answer (Core.Remote.components.map refused)).reason == some "params: verity/certifier/v0 is missing"
#guard (answer [refused NetworkCheck.Period.protocol]).reason == some "params: verity/inference-only/v0 is missing"
#guard (answer [refused NetworkCheck.Period.protocol, refused NciCheck.Protocol.protocol,
    refused PouwCheck.Period.protocol]).reason == some "params: verity/sampled-proofs/v1 is missing"
#guard (answer [refused NetworkCheck.Period.protocol, refused NciCheck.Protocol.protocol,
    refused PouwCheck.Period.protocol, refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"

-- remote's record tie: the policy and sampled proofs on two records reject, whatever the subprotocols answer
private def record (n : Nat) : String := s!"\{\"format\":\"verity/registration/v2\",\"n\":{n},\"registered\":[]}"

private def recordsOf (policy sampled : String) : Answer :=
  answer [refused NetworkCheck.Period.protocol, { refused NciCheck.Protocol.protocol with commitment := policy.toUTF8 },
    refused PouwCheck.Period.protocol, { refused VStar.Recursive.Sampled.protocol with commitment := sampled.toUTF8 }]

#guard (recordsOf (record 0) (record 0)).reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard (recordsOf (record 1) (record 0)).verdict == .reject
#guard (recordsOf (record 1) (record 0)).published.isNone
#guard (recordsOf (record 1) (record 0)).reason.map
  (·.startsWith "record: verity/inference-only/v0 committed ") == some true
#guard !((recordsOf (record 1) (record 0)).reason.map (contains · "reject: params")).getD true

/-! ### The verdict is the AND of remote's and the network certifier's

Over every verdict of the four subprotocols, the composite of stand-ins under its ids, record subprotocols and tie
(`Stub.stub`: each accepts, rejects or is undecided as its case says) answers what remote's composite of the same
stand-ins and the network certifier's stand-in answer together: accept when both accept, reject when either rejects,
and undecided otherwise. -/

private def standIns (ids : List String) : List Protocol := ids.map Stub.stub

private def composite : Protocol :=
  tiedSame protocolId Core.Remote.tied [Core.Remote.windows] (standIns (components.map (·.id)))

private def remote : Protocol :=
  tiedSame "verity/remote/v0" Core.Remote.tied [Core.Remote.windows] (standIns (Core.Remote.components.map (·.id)))

private def both : Verdict → Verdict → Verdict
  | .reject, _ | _, .reject => .reject
  | .accept, .accept => .accept
  | _, _ => .undecided

private def caseOf (id : String) : Verdict → Cases.Stubbed
  | .accept => Cases.accepts id
  | .reject => Cases.rejects id
  | .undecided => Cases.undecided id

private def verdicts : List Verdict := [.accept, .reject, .undecided]

private def andHolds (v : Verdict) (ws : List Verdict) : Bool :=
  let certifier := caseOf NetworkCheck.Period.protocolId v
  let rest := (Core.Remote.components.map (·.id)).zip ws |>.map fun (i, w) => caseOf i w
  let own := (Stub.stub certifier.id).answer (Cases.stubParams certifier).toUTF8 certifier.root
    (Cases.stubEvidence certifier).toUTF8
  (Cases.answer composite (certifier :: rest)).verdict == both (Cases.answer remote rest).verdict own.verdict

#guard verdicts.all fun v => verdicts.all fun a => verdicts.all fun b => verdicts.all fun c => andHolds v [a, b, c]
#guard (Cases.answer composite ((components.map (·.id)).map Cases.accepts)).verdict == .accept

end Core.RemoteNetwork
