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
circuit divides into inference blocks within their gate and fresh-input budgets, which reads only the registered circuit
and the registered information and so needs no hardware at the site), work by proof of useful work over the period
(`PouwCheck.Period.protocol`, `verity/pouw/period/v1`: the period's windows credited all or nothing), onsite's work
subprotocol, and integrity by SPOT (`VStar.Recursive.Sampled.protocol`, `verity/sampled-proofs/v1`: one-stage
v2's audit on the proofs themselves, on the blobs the caller hands in; one-stage v1, whose verdicts are the evidence's,
and the recursive audit, which SPOT's soundness doesn't cover, are refused). `Core.Composition.tiedSameB`
composes them on the caller's blobs: every
subprotocol accepts or remote doesn't, and any one's rejection rejects remote whatever the others answer. The policy's
commitment and sampled proofs' are the run's record, and both must commit to the same one; every window of the period
must name that record as its Commit record (`windows`). Two records reject remote whatever the subprotocols answer.
The tie is by the record alone: nothing here holds PoUW's committed weights to the weights the record registers, so
remote doesn't establish that PoUW's credited work and the audited inference read the same weights.

A stated exception: the policy's v0 publishes the declared division and each inference block's profile (its gates and
fresh-input bits) in the clear. Remote composes it for one run of record only, as an exception to
everything-zero-knowledge, and v1's private form (the division committed under hiding leaves, compliance proved, a
padded profile) replaces it.
-/
module

public import CoreComposition.Blobs
public import CoreComposition.Cases
public import CoreComposition.Tie
public import NciCheck.Protocol
public import PouwCheck.Period
public import PouwPearlPlus.Protocol
public import Spot.OneStage
public import Spot.OneStageZk
public import VStarRecursive.Sampled
meta import CoreComposition.Blobs
meta import CoreComposition.Cases
meta import CoreComposition.Tie
meta import NciCheck.Protocol
meta import PouwCheck.Period
meta import PouwPearlPlus.Protocol
meta import Spot.OneStage
meta import Spot.OneStageZk
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
    refused Spot.OneStage.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.Period.protocol,
    refused Spot.OneStageZk.protocol]).reason ==
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

/-! ### One window's registration in the log, and the run's record reusing it

`W` is the serving process's registration of a window's tables, the log's first record, in the form a served window
writes it. `R`, the run's registration record, is chained after it (`previous`) and registers those tables under the
names PoUW's Commit gives them (`PouwCheck.Credit.commitRows`, `commitIds`) as output values that reuse `W`'s entries
(`outputs`). The policy's case, sampled proofs' (one-stage v2's, `Spot.OneStageZk`) and the period's one window
all name `R`. `R` registers one window's tables, so a period credits at most one window on it: every window must name
`R` (`windows`), and PoUW's ROOTS1 holds each window's tables to `R`'s (`Internal.arow`). -/

open Lean (Json)

private def rep (b : String) : String := String.join (List.replicate 64 b)
private def jsonOf (s : String) : Json := (Json.parse s).toOption.getD .null
private def textOf (j : Json) : String := (Flock.canon j).toOption.getD ""
private def digestOf (j : Json) : String := (Spot.Registration.digestHex j).toOption.getD ""

/-- A table of `leaves` rows of one 16-bit word. -/
private def tableEntry (value root domain : String) (leaves : Nat) : Json :=
  Json.mkObj [("domain", domain), ("leaves", leaves), ("root", root), ("value", value), ("words", 1)]

/-- The window's round-0 A table at K = 64, one call of M = 4 rows of one chunk, and its round-0 input-ids table. -/
private def windowA : Json := tableEntry "pouw/h1/a/64@0" (rep "ab") (rep "a1") 4
private def windowIds : Json := tableEntry "pouw/h1/ids@0" (rep "ba") (rep "a2") 8

private def recW : Json := Json.mkObj [("format", "verity/registration/v2"), ("previous", Json.null),
  ("program", rep "11"), ("protocol", Spot.Registration.PROTOCOL),
  ("registered", Json.arr #[windowA, windowIds]),
  ("window", Json.mkObj [("entry", 0), ("kind", "served"), ("open_root", rep "0e"), ("refused", Json.arr #[])])]

/-- The audited circuit: one `CoreSpotPair` instance over `x`, beside the window's two tables as registered
parameters. -/
private def descriptor : String :=
  r#"{"annotations":{},"callees":["CoreSpotXor8_v1","Input8_v1","RegisteredInput16_v1","CoreSpotPair_v0"],"definitions":{"CoreSpotPair_v0":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,1]},"type":0},{"refs":{"affine":["p",0,1,1]},"type":0}],"fn":0,"form":"primitive","out":0}],"params":[["x",["Array",2,["Value",8]]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",8]},"form":"composite","id":"CoreSpotPair_v0"},"CoreSpotXor8_v1":{"form":"primitive","id":"CoreSpotXor8_v1","params":[["a",["Value",8]],["b",["Value",8]]],"ret":["Value",8]},"Input8_v1":{"form":"primitive","id":"Input8_v1","params":[],"ret":["Value",8]},"RegisteredInput16_v1":{"form":"primitive","id":"RegisteredInput16_v1","params":[],"ret":["Value",16]},"Root[CoreRemoteReuse_v0{N=1}]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":1,"form":"batch","n":2,"out":1},{"args":[],"axes":[],"fn":2,"form":"batch","n":8,"out":2},{"args":[],"axes":[],"fn":2,"form":"batch","n":4,"out":3},{"args":[{"refs":{"affine":["n",0,0,2]},"type":4}],"axes":[0],"fn":3,"form":"batch","n":1,"out":5}],"params":[],"ret":{"affine":["n",3,0,1]},"ret_type":["Array",1,["Value",8]]},"form":"composite","id":"Root[CoreRemoteReuse_v0{N=1}]_v1"}},"program_fn":"CoreRemoteReuse_v0{N=1}","root":"Root[CoreRemoteReuse_v0{N=1}]_v1","schema":"verity-ir/descriptor/v1","types":[["Value",8],["Array",2,["Value",8]],["Array",8,["Value",16]],["Array",4,["Value",16]],["Array",1,["Array",2,["Value",8]]],["Array",1,["Value",8]]]}"#

private def signature : Json := jsonOf
  r#"[["x",["Array",1,["Array",2,["Value",8]]]],["input-ids/0",["Array",8,["Value",16]]],["row-table/0/k64",["Array",4,["Value",16]]]]"#

private def circuit : String :=
  match Flock.Program.decode descriptor.toUTF8 with
  | .ok P => Flock.hexOf P.sha512
  | .error _ => ""

private def query : Json := jsonOf r#"{"name":"Q_template_instance","params":{"template":"CoreSpotPair_v0"},"version":0}"#
private def partition : Json := Json.mkObj [("program", circuit), ("query", query)]
private def law : Json := jsonOf r#"{"k":1,"law":"subset"}"#
private def ports : Json := Json.mkObj [("out", Json.mkObj [("domain", rep "d1"), ("leaves", 2)]),
  ("x", Json.mkObj [("domain", rep "d2"), ("leaves", 1)])]

/-- The run's record after `previous`, registering `registered`. -/
private def runRecord (previous : Json) (registered : Array Json) : Json := Json.mkObj [("anchors", Json.arr #[]),
  ("format", "verity/registration/v2"), ("law", law), ("leaf_layer", rep "ee"),
  ("partition", (Spot.Partition.digest partition).getD ""), ("population", 1), ("previous", previous),
  ("program", circuit), ("protocol", Spot.Registration.PROTOCOL),
  ("query", (Spot.Registration.queryDigest query).toOption.getD ""), ("registered", Json.arr registered),
  ("roots", Json.arr #[Json.mkObj [("domain", rep "d1"), ("leaves", 2), ("port", "out"), ("root", rep "cd")],
    Json.mkObj [("domain", rep "d2"), ("leaves", 1), ("port", "x"), ("root", rep "ab")]]),
  ("scheme", "frame-v3-sha512/hm96-sha512"), ("window", Json.mkObj [("kind", "served"), ("run", "r-1")])]

/-- `W`'s tables under the Commit's names: its entries, root and domain included. -/
private def reused : Array Json := #[windowA.setObjVal! "value" (PouwCheck.Credit.commitRows 0 64),
  windowIds.setObjVal! "value" (PouwCheck.Credit.commitIds 0)]

private def recR : Json := runRecord (digestOf recW) reused

/-- Each of the Commit's names an output value reusing `record`'s entry of the window's table. -/
private def outputsOf (record : String) : Json := Json.mkObj [
  (PouwCheck.Credit.commitIds 0, Json.mkObj [("record", record), ("value", "pouw/h1/ids@0")]),
  (PouwCheck.Credit.commitRows 0 64, Json.mkObj [("record", record), ("value", "pouw/h1/a/64@0")])]

/-- The verifier's own registered file (`flock-verify --registered`'s): `R`'s tables under the Commit's names, which no
proof block of the audited circuit reads. -/
private def readsFile : Json := Json.mkObj [("format", Flock.Registered.FORMAT), ("ports", Json.mkObj (reused.toList.map
  fun x => ((x.getObjValD "value").getStr?.toOption.getD "", Json.mkObj [("domain", x.getObjValD "domain"),
    ("leaves", x.getObjValD "leaves"), ("reads", Json.arr #[Json.null]), ("root", x.getObjValD "root"),
    ("value", x.getObjValD "value"), ("words", x.getObjValD "words")])))]

/-- The auditor's challenger, whose signed release the sample's coins come in: an LMS key over SHA-512 at `Lms.deployed`'s
parameters, the only ones v2 reads a challenger's signatures under (`OneStageZk.KEY_PARAMS`). -/
private def challenger : Json := Json.mkObj [("key", Json.mkObj [("H", 20), ("W", 1),
  ("public", String.join (List.replicate 80 "c1"))]), ("label", "draw")]

/-- One-stage v2's params: v1's over the log `log`, with `outputs`, the challenger and `W`'s open root as the window's,
and `proofs`, the circuit and public files' SHA-512s and the verifier's registered file. -/
private def sampledParams (log : Array Json) (outputs : Option Json) : Json := Json.mkObj ([("anchors", Json.arr #[]),
  ("challenger", challenger), ("delta", "1/10"), ("form", "one-stage"), ("law", law), ("leaf_layer", rep "ee"),
  ("log", Json.arr log), ("parameters", signature), ("partition", partition), ("ports", ports),
  ("program", jsonOf descriptor),
  ("proofs", Json.mkObj [("circuit", rep "c0"), ("public", rep "c2"), ("registered", readsFile)]),
  ("query", query), ("registered", Json.mkObj [(PouwCheck.Credit.commitIds 0, 1), (PouwCheck.Credit.commitRows 0 64, 1)]),
  ("scheme", "frame-v3-sha512/hm96-sha512"), ("window_root", rep "0e"), ("work_table", Json.null)] ++
    (outputs.map fun o => [("outputs", o)]).getD [])

/-- One-stage v2's evidence of an audit that drew the one unit and proved it in one session: the session's record's
SHA-512 (`proofs`) in place of verdicts, and coins no challenger signed. -/
private def sampledEvidence (rec : Json) : Json :=
  Json.mkObj [("coins", Json.mkObj []), ("draw", jsonOf r#"{"k":1,"law":"subset","population":1,"units":[0]}"#),
    ("form", "one-stage"), ("instances", 1),
    ("proofs", Json.mkObj [("sessions", Json.arr #[Json.mkObj [("name", rep "ab"), ("record", rep "c3"),
      ("reps", Json.mkObj [])]])]),
    ("proved_units", jsonOf "[0]"), ("receipt", Json.mkObj [("registration", digestOf rec)]),
    ("served", Json.null), ("session_roots", Json.null), ("sessions", Json.arr #[rep "ab"]),
    ("statement", jsonOf r#"{"instances":1}"#)]

/-- The policy's case on `rec` after `log`, declaring the audited circuit as one inference block of its 15 gates. -/
private def policyOn (log : Array Json) (rec : Json) : Case :=
  { id := NciCheck.Protocol.protocolId,
    params := textOf (Json.mkObj [("alpha", jsonOf "[961,1]"), ("gamma", 15), ("log", Json.arr log),
      ("parameters", signature), ("program", descriptor)]),
    commitment := (textOf rec).toUTF8,
    evidence := r#"{"owner":["input","input","input",0],"provenance":[],"theta":[]}"#.toUTF8 }

private def sampledOn (log : Array Json) (outputs : Option Json) (rec : Json) : Case :=
  { id := VStar.Recursive.Sampled.protocol.id, params := textOf (sampledParams log outputs),
    commitment := (textOf rec).toUTF8, evidence := (textOf (sampledEvidence rec)).toUTF8 }

/-- A period of one window per record of `recs`, each naming it as its Commit record, under an empty ledger. -/
private def periodNaming (recs : List Json) : Case :=
  { id := PouwCheck.Period.protocolId, params := r#"{"key":null,"main":[],"name":"p"}"#, commitment := "[]".toUTF8,
    evidence := (textOf (Json.mkObj [("pins", Json.arr #[]), ("windows", Json.arr (recs.map fun r =>
      Json.mkObj [("commit", Flock.hexOf (textOf r).toUTF8)]).toArray)])).toUTF8 }

private def answerOf (P : Core.Protocol) (k : Case) : Answer := P.answer k.params.toUTF8 k.commitment k.evidence

private def noBlobs (P : BlobProtocol) : Core.Protocol := P.onBlobs (Core.Blobs.ofArray #[])

/-- The binding of the components' digests on the cases `xs`, as `cs` decode their commitments. -/
private def boundOn (cs : List Core.Protocol) (xs : List Case) : ByteArray :=
  (binding "verity/remote/v0" (answersOf cs (xs.map fun x => ⟨x.id, x.params⟩) xs)).getD ByteArray.empty

-- every reader of remote reads one record, `R`: the policy's, sampled proofs' and the window's Commit record
#guard (policyRecord (policyOn #[recW] recR)).map (·.map Flock.hexOf) == some [digestOf recR]
#guard committedBy VStar.Recursive.Sampled.protocol.id (components.map noBlobs)
  [policyOn #[recW] recR, periodNaming [recR], sampledOn #[recW] (some (outputsOf (digestOf recW))) recR] ==
    (policyRecord (policyOn #[recW] recR)).bind List.head?
#guard windowRecords (periodNaming [recR]) == policyRecord (policyOn #[recW] recR)
#guard (PouwCheck.Credit.commitOf? (.str (Flock.hexOf (textOf recR).toUTF8))).map (Flock.hexOf ·.1) ==
  some (digestOf recR)

private def policy (log : Array Json) : Answer := answerOf NciCheck.Protocol.protocol (policyOn log recR)
private def sampled (log : Array Json) (outputs : Option Json) : Answer :=
  answerOf (noBlobs VStar.Recursive.Sampled.protocol) (sampledOn log outputs recR)

/-- The verdicts `flock-verify verify --zk` writes for `e`'s sessions (`OneStageZk.verdictJson`'s names), each of record
and accepting, reporting the roots it opened `rec`'s registered values against. -/
private def grantedOf (rec : Json) (e : Spot.OneStageZk.Evidence) : Array (String × Json) :=
  let reads := match rec.getObjValD "registered" with
    | .arr xs => xs.toList.flatMap fun x => [0, 1].map fun (i : Nat) =>
      (s!"{(x.getObjValD "value").getStr?.toOption.getD ""}.{i}",
        Json.mkObj [("root", x.getObjValD "root"), ("value", x.getObjValD "value")])
    | _ => []
  e.sessions.map fun s => (s!"flock-verify/{s.name}", Json.mkObj [("accepted", true), ("of_record", true),
    ("registered", Json.mkObj reads), ("session", s.name)])

/-- Sampled proofs with its sample's coins and its sessions' proofs granted: `VStar.Recursive.Sampled.protocol`'s decoders
and digest, and in the one-stage form the check `OneStageZk.checkBlobsAt` ends in, v1's `check` on the verdicts the
sessions' run gives (`grantedOf`); the recursive form refused. -/
private def proved : Core.Protocol where
  id := VStar.Recursive.Sampled.PROTOCOL
  Params := VStar.Recursive.Sampled.Params
  Commitment := Spot.OneStage.Commitment
  Evidence := VStar.Recursive.Sampled.Evidence
  Public := Spot.OneStage.Public
  Profile := Core.Integrity
  params := VStar.Recursive.Sampled.paramsOfJson
  commitment := Spot.OneStage.commitmentOf
  evidence := VStar.Recursive.Sampled.evidenceOf
  digest c := c.digest
  check p c e := match p, e with
    | .inl p, .inl e => Spot.OneStage.check p.base c { e.base with verdicts := grantedOf c.record e }
    | _, _ => .reject "the one-stage form only"
  publicJson := Spot.OneStage.publicJson
  profileJson := Core.Integrity.toJson

private def provedOn (log : Array Json) (outputs : Option Json) : Answer := answerOf proved (sampledOn log outputs recR)

-- the policy accepts `R` after `W`, the log's first record, whose digest `R`'s previous names; with no `W` before it, the
-- log's chain refuses `R` (`trusted`)
#guard (policy #[recW]).verdict == .accept
#guard (policy #[]).verdict == .reject
#guard (policy #[]).reason.map (·.startsWith "trusted: ") == some true

-- sampled proofs' one-stage form is v2's: with no blob and coins no challenger signed, it refuses `R` before any session is
-- verified; v1's form of the audit, its verdicts the evidence's, is refused by name
#guard (sampled #[recW] (some (outputsOf (digestOf recW)))).verdict == .reject
#guard (sampled #[recW] (some (outputsOf (digestOf recW)))).reason.map (·.startsWith "coins: ") == some true
#guard (answerOf (noBlobs VStar.Recursive.Sampled.protocol) { sampledOn #[recW] (some (outputsOf (digestOf recW))) recR with
    evidence := (textOf (Json.mkObj [("form", "one-stage"), ("verdicts", Json.mkObj [])])).toUTF8 }).reason ==
  some s!"evidence: the evidence names no proofs: {VStar.Recursive.Sampled.V1_REFUSED}"

-- with the coins and the session's proofs granted, v2 accepts `R` after `W`, its tables output values that reuse `W`'s
-- entries; it refuses `R` without the outputs (`R7-domain`: R7 holds the tables to the audited circuit's own domains,
-- which `W`'s are not), with outputs naming a record that isn't the log's (`R7-output`), and with no `W` in the log
-- (`R6-previous` too)
#guard (provedOn #[recW] (some (outputsOf (digestOf recW)))).verdict == .accept
#guard (provedOn #[recW] none).verdict == .reject
#guard (provedOn #[recW] none).reason == some "registration: R7-domain"
#guard (provedOn #[recW] (some (outputsOf (rep "77")))).verdict == .reject
#guard (provedOn #[recW] (some (outputsOf (rep "77")))).reason == some "registration: R7-output, R7-domain"
#guard (provedOn #[] (some (outputsOf (digestOf recW)))).verdict == .reject
#guard (provedOn #[] (some (outputsOf (digestOf recW)))).reason == some "registration: R6-previous, R7-output, R7-domain"

-- PoUW's ROOTS1 (`Internal.arow`) holds the window's tables to `R`'s, which are `W`'s entries; `W` registers them under
-- the window's names, not the Commit's
private def rounds : Array Json := #[Json.mkObj [("request_table", rep "cd"), ("requests", Flock.hexOf
  (PouwCheck.Credit.requestsRoot ((Flock.unhex (rep "cd")).toOption.getD .empty) ((Flock.unhex (rep "ba")).toOption.getD .empty)))]]
private def layout : Json := jsonOf r#"{"calls":[["q",0]],"linears":[["q",64,64]],"m":4,"tile":[4,64]}"#
private def arowOn (rec : Json) : Option (Except (String × String) Unit) :=
  (PouwCheck.Credit.commitOf? (.str (Flock.hexOf (textOf rec).toUTF8))).map fun (_, reg) =>
    PouwCheck.Credit.Internal.arow rounds (Json.arr #[windowA, windowIds]) reg layout "pouw/h1/"
#guard (arowOn recR).map (·.toOption.isSome) == some true
#guard (arowOn recW).map (fun r => match r with | .error (c, _) => c | .ok _ => "ok") == some "arow"

-- the tie: the period's one window on `R` holds it, a window on `W` is a stray
#guard windows.strays [policyOn #[recW] recR, periodNaming [recR]] == none
#guard windows.strays [policyOn #[recW] recR, periodNaming [recW]] ==
  some s!"record: {NciCheck.Protocol.protocolId} committed {digestOf recR}, {PouwCheck.Period.protocolId} committed {digestOf recW}"

-- remote on the three cases: the policy accepts `R` and the window names it, so remote's refusals are the period's own,
-- whose credit reads an auditor's ledger, here empty, and sampled proofs', whose coins no challenger signed
private def remoteOn (window : Json) : Answer :=
  let xs := [policyOn #[recW] recR, periodNaming [window], sampledOn #[recW] (some (outputsOf (digestOf recW))) recR]
  Cases.answerCases (protocol.on #[]) xs (boundOn (components.map noBlobs) xs)
#guard (remoteOn recR).verdict == .reject
#guard (remoteOn recR).reason.map (·.startsWith s!"{PouwCheck.Period.protocolId} reject: period: ") == some true
#guard (remoteOn recR).reason.map (contains · s!"; {VStar.Recursive.Sampled.protocol.id} reject: coins: ") == some true
#guard (remoteOn recR).reason.map (contains · s!"{NciCheck.Protocol.protocolId} reject") == some false

/-- The period with its credit granted: `PouwCheck.Period.protocol`'s decoders and digest, its check crediting one window
whatever the ledger and the window say. -/
private def credited : Core.Protocol where
  id := PouwCheck.Period.protocolId
  Params := PouwCheck.Period.Params
  Commitment := PouwCheck.Period.Commitment
  Evidence := PouwCheck.Period.Evidence
  Public := PouwCheck.Credit.PeriodCredit
  Profile := PouwCheck.Period.Profile
  params := PouwCheck.Period.paramsOf
  commitment := PouwCheck.Period.commitmentOf
  evidence := PouwCheck.Period.evidenceOf
  digest := PouwCheck.Period.Commitment.digest
  check p _ _ := .accept ⟨[[0]], [0], 1, PouwCheck.Credit.DELTA, [none]⟩ ⟨p.name, 1⟩
  publicJson := PouwCheck.Credit.PeriodCredit.toJson
  profileJson := PouwCheck.Period.profileJson

/-- Remote's composition (`tiedSame` over its records and its tie) with the period's credit and sampled proofs' coins and
proofs granted. -/
private def crediting : Core.Protocol :=
  tiedSame "verity/remote/v0" tied [windows] [NciCheck.Protocol.protocol, credited, proved]

private def creditingOn (window : Json) : Answer :=
  let xs := [policyOn #[recW] recR, periodNaming [window], sampledOn #[recW] (some (outputsOf (digestOf recW))) recR]
  let cs := [NciCheck.Protocol.protocol, credited, proved]
  Cases.answerCases crediting xs (boundOn cs xs)

-- with those granted, remote accepts the one window on `R`, and refuses a window on `W` by its tie
#guard (creditingOn recR).verdict == .accept
#guard (creditingOn recW).verdict == .reject
#guard (creditingOn recW).reason ==
  some s!"record: {NciCheck.Protocol.protocolId} committed {digestOf recR}, {PouwCheck.Period.protocolId} committed {digestOf recW}"

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
