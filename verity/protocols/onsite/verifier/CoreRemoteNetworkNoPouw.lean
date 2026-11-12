/-
Remote with the network certifier, without PoUW (`verity/remote-network-no-pouw/v0`), **development**: the subprotocols
of remote with the network certifier's development composite (`Core.RemoteNetwork`) but PoUW's period credit. Its
subprotocols are the network certifier over the period (`NetworkCheck.Period.protocol`, `verity/certifier/v0`), the
inference-only policy (`NciCheck.Protocol.protocol`, `verity/inference-only/v0`) and sampled proofs
(`VStar.Recursive.Sampled.protocol`, `verity/sampled-proofs/v1`), each as remote with the network certifier reads it.
`Core.Composition.tiedSameB` composes them on the caller's blobs under remote's record subprotocols (`Core.Remote.tied`)
and remote with the network certifier's egress tie (`Core.RemoteNetwork.egress`), so its verdict is the AND of its
three subprotocols' over one registration: it accepts when all three accept and every compute audit of the network
certifier's period is on the policy's record, and any one's rejection rejects it whatever the others answer.

It credits no verified compute: PoUW's period is none of its subprotocols, so a case of it is refused by name, and
remote's window tie (`Core.Remote.windows`), which reads the period's windows, is none of its ties. It is not remote
with the network certifier with a subprotocol switched off: it has its own id.

It is no verifier: `Core.Composites` doesn't list it, so no configuration names it, the Site has no registration
composite for it, and no soundness theorem covers it; it runs only as its development executable
(`remote-network-no-pouw`). What it doesn't check is remote with the network certifier's: the network certifier's
period reads its audits' answers as its evidence carries them, and the policy's v0 publishes the declared division and
each inference block's profile in the clear (`Core.Remote`).
-/
module

public import CoreComposition.Cases
public import CoreComposition.Stubs
public import CoreRemoteNetwork
meta import CoreComposition.Cases
meta import CoreComposition.Stubs
meta import CoreRemoteNetwork

@[expose] public section

namespace Core.RemoteNetworkNoPouw

open Composition

def protocolId : String := "verity/remote-network-no-pouw/v0"

/-- The subprotocols, in the order of their ids: the network certifier's period, the policy and sampled proofs. -/
def components : List BlobProtocol :=
  [NetworkCheck.Period.protocol, NciCheck.Protocol.protocol, VStar.Recursive.Sampled.protocol]

/-- Every compute audit of the network certifier's period on the policy's record; no window tie, since no
subprotocol reads a PoUW period. -/
def ties : List Tie := [Core.RemoteNetwork.egress]

def protocol : BlobProtocol := tiedSameB protocolId Core.Remote.tied ties components

/-- A case of the subprotocol `id` that its own `answer` refuses: empty parameters, commitment and evidence. -/
private def refusedAs (id : String) : Case :=
  { id := id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def refused (P : BlobProtocol) : Case := refusedAs P.id

private def answer (xs : List Case) : Answer := Cases.answerCases (protocol.on #[]) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

private def ids : List String := components.map (·.id)

#guard orderedB components
#guard components.map (·.id) == ["verity/certifier/v0", "verity/inference-only/v0", "verity/sampled-proofs/v1"]
-- remote with the network certifier's subprotocols but PoUW's period, which no tie names either
#guard ids == (Core.RemoteNetwork.components.map (·.id)).filter (· != PouwCheck.Period.protocolId)
#guard Core.Remote.tied.all ids.contains
#guard ties.all fun t => t.names.all fun n => ids.contains n.1
#guard ties.all fun t => t.names.all fun n => n.1 != PouwCheck.Period.protocolId
#guard ties.map (fun t => (t.what, t.names.map (·.1))) == [("record", [NciCheck.Protocol.protocolId, NetworkCheck.Period.protocolId])]
#guard protocolId != Core.RemoteNetwork.protocolId

-- every subprotocol refuses its case: the composite rejects on each, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard components.all fun P => (allRefused.reason.map (contains · s!"{P.id} reject: params: ")).getD false
-- PoUW's period is not a subprotocol of it: its case is refused by name, beside the three, and remote with the network
-- certifier's four cases are not this composite's
#guard (answer [refused NetworkCheck.Period.protocol, refused NciCheck.Protocol.protocol, refused PouwCheck.Period.protocol,
    refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/pouw/period/v1 is not a component of this composition"
#guard (answer (Core.RemoteNetwork.components.map refused)).reason ==
  some "params: verity/pouw/period/v1 is not a component of this composition"
-- each of the three is needed
#guard (answer [refused NciCheck.Protocol.protocol, refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/certifier/v0 is missing"
#guard (answer [refused NetworkCheck.Period.protocol, refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/inference-only/v0 is missing"
#guard (answer [refused NetworkCheck.Period.protocol, refused NciCheck.Protocol.protocol]).reason ==
  some "params: verity/sampled-proofs/v1 is missing"
#guard (answer [refused NetworkCheck.Period.protocol, refused NciCheck.Protocol.protocol,
    refused Spot.OneStage.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"

-- remote's record tie: the policy and sampled proofs on two records reject, whatever the subprotocols answer
private def record (n : Nat) : String := s!"\{\"format\":\"verity/registration/v2\",\"n\":{n},\"registered\":[]}"

private def recordsOf (policy sampled : String) : Answer :=
  answer [refused NetworkCheck.Period.protocol, { refused NciCheck.Protocol.protocol with commitment := policy.toUTF8 },
    { refused VStar.Recursive.Sampled.protocol with commitment := sampled.toUTF8 }]

#guard (recordsOf (record 0) (record 0)).reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard (recordsOf (record 1) (record 0)).verdict == .reject
#guard (recordsOf (record 1) (record 0)).published.isNone
#guard (recordsOf (record 1) (record 0)).reason.map
  (·.startsWith "record: verity/inference-only/v0 committed ") == some true
#guard !((recordsOf (record 1) (record 0)).reason.map (contains · "reject: params")).getD true

/-! ### The verdict is the AND of its three subprotocols'

Over every verdict of the three subprotocols, the composite of stand-ins under its ids, record subprotocols and tie
(`Stub.stub`: each accepts, rejects or is undecided as its case says) accepts when all three accept, rejects when one
rejects, and is undecided otherwise. -/

private def composite : Protocol := tiedSame protocolId Core.Remote.tied ties (ids.map Stub.stub)

private def caseOf (id : String) : Verdict → Cases.Stubbed
  | .accept => Cases.accepts id
  | .reject => Cases.rejects id
  | .undecided => Cases.undecided id

private def verdicts : List Verdict := [.accept, .reject, .undecided]

private def own (x : Cases.Stubbed) : Verdict :=
  ((Stub.stub x.id).answer (Cases.stubParams x).toUTF8 x.root (Cases.stubEvidence x).toUTF8).verdict

private def andOf (vs : List Verdict) : Verdict :=
  if vs.contains .reject then .reject else if vs.all (· == .accept) then .accept else .undecided

private def andHolds (vs : List Verdict) : Bool :=
  let xs := ids.zip vs |>.map fun (i, v) => caseOf i v
  (Cases.answer composite xs).verdict == andOf (xs.map own) && xs.map own == vs

#guard verdicts.all fun a => verdicts.all fun b => verdicts.all fun c => andHolds [a, b, c]
#guard (Cases.answer composite (ids.map Cases.accepts)).verdict == .accept

end Core.RemoteNetworkNoPouw
