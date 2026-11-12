/-
Remote with the network certifier (`verity/remote-network/v0`), **development**: the accounting of remote's H1 composite
(`Core.Remote.protocol`, `verity/remote/v0`, whose work is PoUW's period credit, not Pearl+'s run verdict) and
communication through the auditor's network certifiers at the site's boundary, with no memory challenger. Its
subprotocols are that composite's three (`Core.Remote.components`: the inference-only policy, PoUW's period credit and
sampled proofs) and the network certifier over the period (`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the
period's execution phases certified and proved, and the bound `K` on what left the site).
`Core.Composition.tiedSameB` composes them on the caller's blobs under remote's record subprotocols
(`Core.Remote.tied`), remote's tie (`Core.Remote.windows`) and its own (`egress`), so its verdict is the AND of
remote's and the network certifier's over one registration: it accepts when both accept and the network certifier's
period is on the policy's record, and either's rejection rejects it whatever the other answers.

The egress tie holds the network certifier's period to the run's registration: every compute audit's registration in
the period's commitment must be the record the policy commits to, by one-stage's digest, and a period with no compute
audit is refused. In the rows form each execution phase's scan reads its `prog` from a compute audit's outputs
(`progTied`), so there the egress the period bounds reads the run's registered values. A tie only refuses: it adds no
step that accepts.

It is no verifier, as the H1 composite it extends is none: `Core.Composites` doesn't list it, so no configuration names
it, the Site has no registration composite for it, and no soundness theorem covers it; it runs only as its development
executable (`remote-network`). The same holds of the composite without PoUW (`Core.RemoteNetworkNoPouw`), which reads
this module's egress tie: no composite of record is over the network certifier's period but onsite's.

The network certifier's period reads its audits' answers as its evidence carries them; nothing here holds them to
sampled proofs' answer (`NetworkCheck.Period`'s "What it doesn't check"). The policy's v0 publishes the declared
division and each inference block's profile in the clear, remote's stated exception (`Core.Remote`).
-/
module

public import CoreComposition.Cases
public import CoreComposition.Stubs
public import CoreRemote
public import NetworkCheck.Period
meta import CoreComposition.Cases
meta import CoreComposition.Stubs
meta import CoreRemote
meta import NetworkCheck.Period

@[expose] public section

namespace Core.RemoteNetwork

open Composition

def protocolId : String := "verity/remote-network/v0"

/-- The subprotocols, in the order of their ids: the network certifier's period, then remote's. -/
def components : List BlobProtocol := NetworkCheck.Period.protocol :: Core.Remote.components

/-- The digests of the registrations the network certifier's period reads its compute audits' answers against (its
commitment's `audits`, as `NetworkCheck.Period.protocol` decodes it), in order. A period with no compute audit names
the empty string, which no record's digest is, so that the egress tie refuses it rather than hold with nothing named. -/
def auditRecords (k : Case) : Option (List ByteArray) := do
  let c ← (NetworkCheck.Period.commitmentOf k.commitment).toOption
  let ds ← c.audits.toList.mapM fun r => (Flock.WindowRecord.digest r).toOption
  pure (if ds.isEmpty then [ByteArray.empty] else ds)

/-- Every compute audit of the network certifier's period on the policy's record. -/
def egress : Tie :=
  ⟨"record", [(NciCheck.Protocol.protocolId, Core.Remote.policyRecord), (NetworkCheck.Period.protocolId, auditRecords)]⟩

def ties : List Tie := [Core.Remote.windows, egress]

def protocol : BlobProtocol := tiedSameB protocolId Core.Remote.tied ties components

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
#guard ties.all fun t => t.names.all fun n => (components.map (·.id)).contains n.1

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
    refused PouwCheck.Period.protocol, refused Spot.OneStage.protocol]).reason ==
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

-- the egress tie: every compute audit of the network certifier's period on the policy's record holds; one on another
-- record, or a period with no compute audit, names the policy's record and the period's stray (the empty string when
-- it has none); a period that doesn't decode names nothing, and its own rejection stands
private def periodOn (audits : List String) : Case :=
  { refused NetworkCheck.Period.protocol with
    commitment := s!"\{\"audits\":[{",".intercalate audits}],\"windows\":[]}".toUTF8 }

private def policyOn (r : String) : Case := { refused NciCheck.Protocol.protocol with commitment := r.toUTF8 }

private def digestOf (r : String) : String :=
  ((Lean.Json.parse r).toOption.bind fun j => (Flock.WindowRecord.digestHex j).toOption).getD ""

#guard digestOf (record 0) != "" && digestOf (record 0) != digestOf (record 1)
#guard (auditRecords (periodOn [record 0, record 0])).map (·.map Flock.hexOf) ==
  some [digestOf (record 0), digestOf (record 0)]
#guard (Core.Remote.policyRecord (policyOn (record 0))).map (·.map Flock.hexOf) == some [digestOf (record 0)]
#guard egress.strays [periodOn [record 0], policyOn (record 0)] == none
#guard egress.strays [periodOn [record 0, record 0], policyOn (record 0)] == none
#guard egress.strays [periodOn [record 1], policyOn (record 0)] ==
  some s!"record: verity/inference-only/v0 committed {digestOf (record 0)}, verity/certifier/v0 committed \
    {digestOf (record 1)}"
#guard egress.strays [periodOn [record 0, record 1], policyOn (record 0)] ==
  some s!"record: verity/inference-only/v0 committed {digestOf (record 0)}, verity/certifier/v0 committed \
    {digestOf (record 1)}"
#guard egress.strays [periodOn [], policyOn (record 0)] ==
  some s!"record: verity/inference-only/v0 committed {digestOf (record 0)}, verity/certifier/v0 committed "
#guard egress.strays [refused NetworkCheck.Period.protocol, policyOn (record 0)] == none
#guard egress.strays [periodOn [record 1]] == none

/-! ### The verdict is the AND of remote's and the network certifier's

Over every verdict of the four subprotocols, the composite of stand-ins under its ids, record subprotocols and tie
(`Stub.stub`: each accepts, rejects or is undecided as its case says) answers what remote's composite of the same
stand-ins and the network certifier's stand-in answer together: accept when both accept, reject when either rejects,
and undecided otherwise. -/

private def standIns (ids : List String) : List Protocol := ids.map Stub.stub

private def composite : Protocol :=
  tiedSame protocolId Core.Remote.tied ties (standIns (components.map (·.id)))

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
