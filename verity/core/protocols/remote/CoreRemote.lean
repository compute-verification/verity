/-
Remote Verity: what the verifier establishes with no hardware of its own at the site. Two composites, which differ in
their work subprotocol:

* `Core.Remote.V1.protocol` (`verity/remote/v1`), **the composite of record**: the inference-only policy, Pearl+'s run
  verdict (`PouwCheck.PearlPlus.Protocol.protocol`, `verity/pouw/pearl-plus/v1`: a served run's retained passes, the
  work-weighted sample over their tiles at δ = 1/100, and the run's credit at Pearl's assumption) and sampled proofs.
  `Verity.Registration.Composite.remote` reads it.
* `Core.Remote.protocol` (`verity/remote/v0`), **development**: H1's period credit in Pearl+'s place, with the
  period's windows tied to the run's record (`windows`). It is no verifier: `Core.Composites` doesn't list it, so no
  configuration names it, and no soundness theorem covers it; it runs only as its development executable (`remote`).

The id bumps with the work subprotocol, because an accepted answer of each means something different: v0's work is a
period's credit, every window credited all or nothing and named on the run's record, and v1's is one run's credit at
Pearl's assumption at δ, published as Pearl+'s `Credit`; a caller that read a v0 answer as v1's would read an H1 period
credit as a Pearl+ run's. The cases each takes are different too (v1's work case is `verity/pouw/pearl-plus/v1`'s), so
neither accepts the other's input.

v1 ties the run Pearl+ credits to the run's record (`V1.run`): Pearl+'s commitment names the digest of the registration
record its run is on, which must be the record the policy and sampled proofs commit to, so a run naming another record
rejects v1 whatever the subprotocols answer. Pearl+'s check holds the passes to its window record W (the record
`served_passes.py retain --window-record` writes, which registers each call's A rows and y and each weight's B rows,
``pearl-b/<w>``, at the root of PoUW's tree of them), and v1 ties W to the run's record (`V1.window`): the policy's
record must be chained right after W (its `previous` is W's digest), as the served pipeline writes it, so the values the
record takes from W are the ones Pearl+ credited. W states its version (its header's ``version``). At version 2 the
weights Pearl+'s tiles open B against are W's ``pearl-b/`` values, each registered once in the run's record's log, and
a version-2 W that registers no weight is refused, so PoUW's credited work reads the weights that log registers. A W
that states no version is version 1, written before W registered the weights, and Pearl+'s verdict on it names its
weights untied (`PouwCheck.PearlPlus.WEIGHTS_UNTIED`): there the tie is by the record alone, nothing holds PoUW's
committed weights to the weights the record registers, and remote doesn't establish that PoUW's credited work and the
audited inference read the same weights. The audited circuit, cut at its linears, reads no B row.

v0's subprotocols are the inference-only policy (`NciCheck.Protocol.protocol`, `verity/inference-only/v0`: the declared
circuit divides into inference blocks within their gate and fresh-input budgets, which reads only the declared circuit
and the registered information and so needs no hardware at the site), work by proof of useful work over the period
(`PouwCheck.Period.protocol`, `verity/pouw/period/v1`: the period's windows credited all or nothing), onsite's work
subprotocol, and integrity by sampled proofs (`VStar.Recursive.Sampled.protocol`, `verity/sampled-proofs/v1`, in the
form its params name: one-stage v2's audit on the proofs themselves, or the recursive audit, V*'s compiled check run on
the recursive session's parts, each on the blobs the caller hands in; one-stage v1, whose verdicts are the evidence's, is
refused). `Core.Composition.tiedSameB` composes them on the caller's blobs: every
subprotocol accepts or remote doesn't, and any one's rejection rejects remote whatever the others answer. The policy's
commitment and sampled proofs' are the run's record, and both must commit to the same one; every window of the period
must name that record as its Commit record (`windows`). Two records reject remote whatever the subprotocols answer.

A stated exception: the policy's v0 publishes the declared division and each inference block's profile (its gates and
<<<<<<< ours
fresh-input bits) in the clear. Remote composes it as an exception to everything-zero-knowledge, for the demos' runs,
until v1's private form (the division committed under hiding leaves, compliance proved, a padded profile) replaces it.
||||||| base
fresh-input bits) in the clear. Remote composes it for the run of record of 28 Oct only, as an exception to
everything-zero-knowledge, by Daniel's ruling: "Let's run it in the clear tonight as a stated exception." v1's
private form (the division committed under hiding leaves, compliance proved, a padded profile) replaces it after
15:00Z on 28 Oct.
=======
fresh-input bits) in the clear. Remote composes it for one run of record only, as an exception to
everything-zero-knowledge, and v1's private form (the division committed under hiding leaves, compliance proved, a
padded profile) replaces it.
>>>>>>> theirs
-/
module

public import CoreComposition.Blobs
public import CoreComposition.Cases
public import CoreComposition.Tie
public import NciCheck.Protocol
public import PouwCheck.Period
public import PouwPearlPlus.Protocol
public import SampledProofs.OneStage
public import SampledProofs.OneStageZk
public import VStarRecursive.Sampled
meta import CoreComposition.Blobs
meta import CoreComposition.Cases
meta import CoreComposition.Tie
meta import NciCheck.Protocol
meta import PouwCheck.Period
meta import PouwPearlPlus.Protocol
meta import SampledProofs.OneStage
meta import SampledProofs.OneStageZk
meta import VStarRecursive.Sampled

@[expose] public section

namespace Core.Remote

open Composition

/-- Remote's subprotocols, in the order of their ids. -/
def components : List BlobProtocol :=
  [NciCheck.Protocol.protocol, PouwCheck.Period.protocol, VStar.Recursive.Sampled.protocol]

/-- The subprotocols that must commit to one record: the policy and sampled proofs. -/
def tied : List String := [NciCheck.Protocol.protocolId, VStar.Recursive.Sampled.protocol.id]

/-- The record the policy's case commits to: its commitment's digest, as the policy decodes it. -/
def policyRecord (k : Case) : Option (List ByteArray) :=
  (NciCheck.Protocol.protocol.commitment k.commitment).toOption.map fun c => [NciCheck.Protocol.protocol.digest c]

/-- The digests of the window records a period's case names, in its close's order: each window's Commit record as the
period's credit reads it (`PouwCheck.Credit.commitOf?`, under the auditor's ledger); `none` when the evidence doesn't
decode or a window names no Commit record, which the period refuses. -/
def windowRecords (k : Case) : Option (List ByteArray) := do
  let e ← (PouwCheck.Period.evidenceOf k.evidence).toOption
  let .arr ws := e.windows | none
  ws.toList.mapM fun w => (PouwCheck.Credit.commitOf? (PouwCheck.get w "commit")).map (·.1)

/-- Every window of the period on the policy's record. -/
def windows : Tie :=
  ⟨"record", [(NciCheck.Protocol.protocolId, policyRecord), (PouwCheck.Period.protocolId, windowRecords)]⟩

def protocol : BlobProtocol := tiedSameB "verity/remote/v0" tied [windows] components

/-- A case of the subprotocol `id` that its own `answer` refuses: empty parameters, commitment and evidence. -/
private def refusedAs (id : String) : Case :=
  { id := id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def refused (P : BlobProtocol) : Case := refusedAs P.id

private def answer (xs : List Case) : Answer := Cases.answerCases (protocol.on #[]) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) ==
  ["verity/inference-only/v0", "verity/pouw/period/v1", "verity/sampled-proofs/v1"]
#guard tied.all fun s => (components.map (·.id)).contains s
#guard windows.names.all fun n => (components.map (·.id)).contains n.1

-- every subprotocol refuses its case: remote rejects on each, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/inference-only/v0 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/pouw/period/v1 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/sampled-proofs/v1 reject: params: ") == some true
#guard (answer [refused PouwCheck.Period.protocol, refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/inference-only/v0 is missing"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.Period.protocol]).reason ==
  some "params: verity/sampled-proofs/v1 is missing"
-- PoUW's window check is not remote's work subprotocol, and neither v1's nor v2's one-stage audit alone is its
-- integrity subprotocol
#guard (answer [refused NciCheck.Protocol.protocol, refusedAs "verity/pouw/v1",
    refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/pouw/v1 is not a component of this composition"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.Period.protocol,
    refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.Period.protocol,
    refused SampledProofs.OneStageZk.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v2 is not a component of this composition"

/-- A registration record every record reader decodes, a window's Commit record among them. -/
private def record (n : Nat) : String := s!"\{\"format\":\"verity/registration/v2\",\"n\":{n},\"registered\":[]}"

private def committing (P : BlobProtocol) (r : String) : Case := { refused P with commitment := r.toUTF8 }

private def recordsOf (policy period sampled : String) : Answer :=
  answer [committing NciCheck.Protocol.protocol policy, committing PouwCheck.Period.protocol period,
    committing VStar.Recursive.Sampled.protocol sampled]

-- the policy and sampled proofs on one record: the tie holds, and the verdict is the subprotocols' (each refuses its
-- params)
#guard (recordsOf (record 0) "[]" (record 0)).reason.map
  (·.startsWith "verity/inference-only/v0 reject: params: ") == some true
-- the policy's record is not sampled proofs': remote rejects on the tie, whatever the subprotocols answer
#guard (recordsOf (record 1) "[]" (record 0)).verdict == .reject
#guard (recordsOf (record 1) "[]" (record 0)).published.isNone
#guard (recordsOf (record 1) "[]" (record 0)).reason.map
  (·.startsWith "record: verity/inference-only/v0 committed ") == some true
#guard (recordsOf (record 1) "[]" (record 0)).reason.map
  (contains · ", verity/sampled-proofs/v1 committed ") == some true
#guard !((recordsOf (record 1) "[]" (record 0)).reason.map (contains · "reject: params")).getD true

-- a period's windows name records by their Commit records' digests, which are the policy's and sampled proofs' digests
-- of one record; a window on another record is a stray
private def windowOn (r : String) : String := s!"\{\"commit\":\"{Flock.hexOf r.toUTF8}\"}"
private def periodOn (rs : List String) : Case :=
  { refused PouwCheck.Period.protocol with
    evidence := ("{\"pins\":[],\"windows\":[" ++ ",".intercalate (rs.map windowOn) ++ "]}").toUTF8 }
private def strayOf (policy : String) (rs : List String) : Option String :=
  windows.strays [committing NciCheck.Protocol.protocol policy, periodOn rs]
#guard (windowRecords (periodOn [record 0, record 0])).map (·.length) == some 2
#guard windowRecords (periodOn [record 0]) == policyRecord (committing NciCheck.Protocol.protocol (record 0))
#guard policyRecord (committing NciCheck.Protocol.protocol (record 0)) ==
  (VStar.Recursive.Sampled.protocol.commitment (record 0).toUTF8).toOption.map
    fun c => [VStar.Recursive.Sampled.protocol.digest c]
#guard strayOf (record 0) [record 0, record 0] == none
-- a window on another record
#guard (strayOf (record 0) [record 0, record 1]).map (·.startsWith
  "record: verity/inference-only/v0 committed ") == some true
#guard (strayOf (record 0) [record 0, record 1]).map (contains · ", verity/pouw/period/v1 committed ") == some true
-- every window on a record that isn't the policy's
#guard (strayOf (record 1) [record 0]).isSome
-- a window naming no Commit record leaves the period to refuse it
#guard strayOf (record 0) [record 0, "{}"] == none

/-! ## `verity/remote/v1`, the composite of record -/

namespace V1

/-- v1's subprotocols, in the order of their ids: Pearl+'s run verdict is the work component. -/
def components : List BlobProtocol :=
  [NciCheck.Protocol.protocol, PouwCheck.PearlPlus.Protocol.protocol, VStar.Recursive.Sampled.protocol]

/-- The subprotocols that must commit to one record: the policy and sampled proofs, as in v0. -/
def tied : List String := [NciCheck.Protocol.protocolId, VStar.Recursive.Sampled.protocol.id]

/-- The record Pearl+'s case names: its commitment's `record`, as Pearl+ decodes it. -/
def runRecord (k : Case) : Option (List ByteArray) :=
  (PouwCheck.PearlPlus.Protocol.commitmentOf k.commitment).toOption.map fun c => [c.record]

/-- Pearl+'s run on the policy's record. -/
def run : Tie :=
  ⟨"record", [(NciCheck.Protocol.protocolId, policyRecord), (PouwCheck.PearlPlus.Protocol.protocolId, runRecord)]⟩

/-- The window record W Pearl+'s check holds the run's passes to: its digest, as Pearl+ decodes its commitment. -/
def runWindow (k : Case) : Option (List ByteArray) :=
  (PouwCheck.PearlPlus.Protocol.commitmentOf k.commitment).toOption.map fun c => [c.windowDigest]

/-- The record the policy's record is chained right after (its `previous`), as the policy decodes its record; the empty
string of bytes, no record's digest, when it names none, so that a record chained after nothing is a stray. -/
def policyPrevious (k : Case) : Option (List ByteArray) :=
  (NciCheck.Protocol.commitmentOf k.commitment).toOption.map fun c =>
    [((c.record.getObjValAs? String "previous").toOption.bind fun s => (Flock.unhex s).toOption).getD ByteArray.empty]

/-- Pearl+'s passes on the window record the policy's record is chained right after: the values the run's record
takes from W are the ones Pearl+ credited. -/
def window : Tie :=
  ⟨"window", [(NciCheck.Protocol.protocolId, policyPrevious), (PouwCheck.PearlPlus.Protocol.protocolId, runWindow)]⟩

def ties : List Tie := [run, window]

def protocol : BlobProtocol := tiedSameB "verity/remote/v1" tied ties components

private def answerV1 (xs : List Case) : Answer := Cases.answerCases (protocol.on #[]) xs ⟨(List.replicate 64 0).toArray⟩

#guard orderedB components
#guard components.map (·.id) ==
  ["verity/inference-only/v0", "verity/pouw/pearl-plus/v1", "verity/sampled-proofs/v1"]
#guard tied.all fun s => (components.map (·.id)).contains s

-- every subprotocol refuses its case: v1 rejects on each, and makes nothing public
private def allRefused : Answer := answerV1 (components.map refused)
#guard allRefused.protocol == "verity/remote/v1"
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/inference-only/v0 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/pouw/pearl-plus/v1 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/sampled-proofs/v1 reject: params: ") == some true
#guard (answerV1 [refused NciCheck.Protocol.protocol, refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/pouw/pearl-plus/v1 is missing"
-- H1's period is not v1's work subprotocol: a v0 case set is refused
#guard (answerV1 [refused NciCheck.Protocol.protocol, refused PouwCheck.Period.protocol,
    refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/pouw/period/v1 is not a component of this composition"
-- and Pearl+ is not v0's
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.PearlPlus.Protocol.protocol,
    refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/pouw/pearl-plus/v1 is not a component of this composition"

-- the policy's record is not sampled proofs': v1 rejects on the tie, whatever the subprotocols answer
private def recordsOfV1 (policy sampled : String) : Answer :=
  answerV1 [committing NciCheck.Protocol.protocol policy, refused PouwCheck.PearlPlus.Protocol.protocol,
    committing VStar.Recursive.Sampled.protocol sampled]
#guard (recordsOfV1 (record 0) (record 0)).reason.map
  (·.startsWith "verity/inference-only/v0 reject: params: ") == some true
#guard (recordsOfV1 (record 1) (record 0)).verdict == .reject
#guard (recordsOfV1 (record 1) (record 0)).reason.map
  (·.startsWith "record: verity/inference-only/v0 committed ") == some true

-- Pearl+'s run names a record by its digest, the policy's and sampled proofs' digest of it; a run naming another
-- record is a stray
#guard run.names.all fun n => (components.map (·.id)).contains n.1
private def digestOf (r : String) : ByteArray :=
  ((policyRecord (committing NciCheck.Protocol.protocol r)).bind (·.head?)).getD ByteArray.empty
private def pearlIn (r w : String) : Case :=
  { refused PouwCheck.PearlPlus.Protocol.protocol with
    commitment := ("{\"manifests\":[{}],\"record\":\"" ++ Flock.hexOf (digestOf r) ++
      "\",\"window\":{\"dropped\":[],\"record\":" ++ w ++ ",\"rounds\":null}}").toUTF8 }
private def pearlOn (r : String) : Case := pearlIn r (record 9)
private def runStray (policy run : String) : Option String :=
  V1.run.strays [committing NciCheck.Protocol.protocol policy, pearlOn run]
#guard (digestOf (record 0)).size == 64
#guard runRecord (pearlOn (record 0)) == policyRecord (committing NciCheck.Protocol.protocol (record 0))
#guard runStray (record 0) (record 0) == none
#guard (runStray (record 1) (record 0)).map (·.startsWith "record: verity/inference-only/v0 committed ") == some true
#guard (runStray (record 1) (record 0)).map (contains · ", verity/pouw/pearl-plus/v1 committed ") == some true
-- a run whose commitment doesn't decode leaves Pearl+ to refuse it
#guard V1.run.strays [committing NciCheck.Protocol.protocol (record 0), refused PouwCheck.PearlPlus.Protocol.protocol] ==
  none

-- Pearl+'s passes are tied to its window record W, and the policy's record must be chained right after W: one chained
-- after another record, or after none, is a stray
private def chainedAfter (n : Nat) (w : String) : String :=
  s!"\{\"format\":\"verity/registration/v2\",\"n\":{n},\"previous\":\"{Flock.hexOf (digestOf w)}\",\"registered\":[]}"
private def windowStray (policy : String) (w : String) : Option String :=
  V1.window.strays [committing NciCheck.Protocol.protocol policy, pearlIn policy w]
#guard window.names.all fun n => (components.map (·.id)).contains n.1
#guard runWindow (pearlIn (record 0) (record 9)) == some [digestOf (record 9)]
#guard policyPrevious (committing NciCheck.Protocol.protocol (chainedAfter 0 (record 9))) == some [digestOf (record 9)]
#guard windowStray (chainedAfter 0 (record 9)) (record 9) == none
#guard (windowStray (chainedAfter 0 (record 8)) (record 9)).map
  (·.startsWith "window: verity/inference-only/v0 committed ") == some true
#guard (windowStray (chainedAfter 0 (record 8)) (record 9)).map (contains · ", verity/pouw/pearl-plus/v1 committed ") ==
  some true
#guard (windowStray (record 0) (record 9)).isSome
-- both ties hold on a run on the policy's record whose W that record is chained right after
#guard (ties.findSome? (·.strays [committing NciCheck.Protocol.protocol (chainedAfter 0 (record 9)),
  pearlIn (chainedAfter 0 (record 9)) (record 9)])) == none

end V1

end Core.Remote
