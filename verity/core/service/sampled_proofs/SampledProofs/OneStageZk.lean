/-
Sampled proofs' one-stage audit on the proofs themselves (`verity/sampled-proofs/one-stage/v2`), a `Core.BlobProtocol`: the
v1 audit (`OneStage`) with the verdicts computed here, from the statement's files, rather than read from the evidence.

* **params**: v1's, plus `proofs`, exactly `{"circuit", "registered"}`: the SHA-512 the auditor committed for the circuit
  file (lowercase hex of 64 bytes) and the verifier's own registered file (`flock-verify --registered`'s object, or null
  for a program that reads no registered value). The partition must not be null: the sessions are verified against the
  verifier's own partition object; and v1's `challenger`, with its `window_root`, must be named (`NO_CHALLENGER`);
* **commitment**: v1's (the registration record);
* **evidence**: v1's without `verdicts`, plus `proofs`, exactly `{"public", "sessions"}`: the public file's SHA-512 and,
  for each of v1's `sessions` in order, `{"name", "record", "reps"}`: its name, its `session.json`'s SHA-512 and
  `{file: SHA-512}` naming at least each rep file the statement's session reads (`<stream with / as .>.bin`);
* **blobs**: the files those digests name (`Core.Blobs`), and the MUFU tables any lookup reads, found under their pinned
  SHA-512 as `flock-verify --archive` finds them;
* **coins** (`coinsOf`): v1's (`OneStage.coinsOf`), what `Challenger.drawCoins` reads from the evidence's `coins` under
  the challenger's key, on the verifier's window; none read refuses. The check below runs at them (`checkBlobsAt`): v1's
  `checkAt` refuses a draw other than the one they give before v1's `check`;
* **check**: `flock-verify verify --zk --statement verity/flock-circuit` with `--circuit`, `--public`, `--partition`,
  `--program` and `--registered` the verifier's own and one `--session` a session, as `one_stage.proofs.verify` runs it
  (`verdicts`, the C-Flock library calls `verifyCmd` makes on that path, its last the files' ports and roots,
  `Flock.PortRoots.checkOf`, and the form `Flock.Guarantees.SampledKSLiveV4` covers, `Flock.KsScope.checkOf`); then v1's
  `check` on those verdicts, one of
  record a session (`flock-verify/<name>`). A session accepted under another program than the partition object's, proved
  units other than the evidence's `proved_units`, or a circuit, public or record file no blob holds, rejects; a rep
  file no blob holds is that session's reject, as a file `flock-verify` can't read is.

**The group form** proves a population of several templates, one statement a template (`scan_zk.members_of`'s
`first`, `ingress-w` and `egress-w`): the parameters' `proofs` are exactly `{"members"}`, a non-empty list of
`{"name", "circuit", "registered"}`, distinct names, each member's committed circuit digest and its own registered file
(the auditor's: the evidence names no circuit or registered file); the evidence's are exactly `{"members"}`, a list of
`{"name", "public", "sessions"}`, the parameters' members in order, each its public file and its sessions as above
(none for a member with no unit drawn), v1's `sessions` the members' sessions in order. The check (`checkGroupAt`): the
registration's `leaf_layer` is the SHA-512 of the canonical `{name: public file's SHA-512}` (`leafLayer`), so each
member's public file is the one registered before the draw; each member's sessions are verified as above on the
group's program and partition object with the member's circuit, public and registered file, on one run, so every
session's coin-tree key is fresh among the whole group's (`runGroup`); the units the sessions proved, once all
accepted, are each proved once and together the evidence's `proved_units`; then v1's `checkAt`. A member's unit reads
another's output (the scan's state, a warden's rows) only as a registered read: the group holds unit t's output to be
unit t + 1's input only through the hm96 binding of the registered root both read and each session's knowledge
soundness at its registered ports, never by comparing values in the clear.

Without blobs (`toProtocol`) every check rejects: no blob holds the circuit file.
-/
import CoreProtocol.Blobs
import SampledProofs.OneStage
import Flock

namespace SampledProofs.OneStageZk

open Lean (Json)
open Flock (V)

def PROTOCOL : String := "verity/sampled-proofs/one-stage/v2"
/-- The statement every drawn unit's session proves (`one_stage.proofs.STATEMENT`). -/
def STATEMENT : String := "verity/flock-circuit"
/-- The refusal of parameters naming no challenger: the draw's coins are never the prover's to choose. -/
def NO_CHALLENGER : String :=
  "the parameters name no challenger: the draw's coins are the auditor's challenger's signed release (challenger, " ++
    "with window_root), never the prover's record"

/-- The verifier's own inputs to `flock-verify verify --zk`: the program's descriptor (as JSON and as the bytes
`--program` reads), the partition object, the circuit file's committed SHA-512 and the registered file. -/
structure Own where
  programJ : Json
  program : ByteArray
  partition : Json
  circuit : String
  registered : Option Json

/-- The parameters: v1's, the verifier's own inputs and, in the group form, its members, each by name with its own
inputs (the group's program and partition object, the member's circuit digest and registered file). The single form
has no member. -/
structure Params where
  base : OneStage.Params
  own : Own
  members : Array (String × Own) := #[]

structure RunFiles where
  name : String
  record : String
  reps : Array (String × String)

/-- A member's proofs in the group form's evidence: its name, its public file's SHA-512 and its sessions. -/
structure Member where
  name : String
  pub : String
  sessions : Array RunFiles

/-- The evidence: v1's, and the single form's public file and sessions, or the group form's members. -/
structure Evidence where
  base : OneStage.Evidence
  pub : String
  sessions : Array RunFiles
  members : Array Member := #[]

private def exactly (j : Json) (ks : Array String) (what : String) : V Unit :=
  if keys? j != some (sortStrings ks) then throw s!"the keys of {what} are not {", ".intercalate (sortStrings ks).toList}"
  else pure ()

private def digest? (s : String) : Bool := s.length == 128 && isHex s

private def without (j : Json) (k : String) : List (String × Json) :=
  match j with
  | .obj kvs => (kvs.toArray.filter (·.1 != k)).toList
  | _ => []

/-- `one_stage.proofs.reads_registered`'s name: `RegisteredInput<w>_v1`, `w` a positive decimal without a leading zero. -/
def registeredInputName (k : String) : Bool :=
  let cs := k.toList
  let p := "RegisteredInput".toList
  let s := "_v1".toList
  let mid := (cs.drop p.length).take (cs.length - p.length - s.length)
  cs.length > p.length + s.length && cs.take p.length == p && cs.drop (cs.length - s.length) == s &&
    mid.all Char.isDigit && mid.head? != some '0'

/-- `one_stage.proofs.reads_registered`: the descriptor has an input bound at registration, or no `definitions` object
to tell. -/
def readsRegistered (program : Json) : Bool :=
  match getD program "definitions" with
  | .obj kvs => kvs.toArray.any fun (k, _) => registeredInputName k
  | _ => true

private def circuitOf (pr : Json) : V (String × Option Json) := do
  let .str circuit := getD pr "circuit" | throw "proofs: circuit is not a SHA-512 digest (128 lowercase hex)"
  if !digest? circuit then throw "proofs: circuit is not a SHA-512 digest (128 lowercase hex)"
  let registered ← match getD pr "registered" with
    | .null => pure none
    | r@(.obj _) => pure (some r)
    | _ => throw "proofs: registered is not the verifier's registered file (an object) or null"
  pure (circuit, registered)

/-- A group form's member in the parameters: exactly `{"circuit", "name", "registered"}`. -/
private def memberOwnOf (programJ : Json) (program : ByteArray) (partition : Json) (m : Json) : V (String × Own) := do
  exactly m #["circuit", "name", "registered"] "a member of proofs"
  let .str name := getD m "name" | throw "proofs: a member's name is not a string"
  let (circuit, registered) ← circuitOf m
  pure (name, { programJ, program, partition, circuit, registered })

def paramsOfJson (j : Json) : V Params := do
  let some pr := field? j "proofs"
    | throw "the parameters name no proofs (the auditor's circuit digest and the verifier's registered reads)"
  let group := (field? pr "members").isSome
  exactly pr (if group then #["members"] else #["circuit", "registered"]) "proofs"
  let rest := Json.mkObj (without j "proofs")
  let base ← OneStage.paramsOfJson rest
  if base.challenger.isNone then throw NO_CHALLENGER
  let some partition := base.verifier.partition
    | throw "partition is null: the sessions are verified against the verifier's own partition object"
  let programJ := getD rest "program"
  let program := (← Flock.canon programJ).toUTF8
  if group then
    let .arr ms := getD pr "members" | throw "proofs: members is not a list"
    if ms.isEmpty then throw "proofs: the group names no member"
    let members ← ms.mapM (memberOwnOf programJ program partition)
    if !decide (members.toList.map (·.1)).Nodup then throw "proofs: two members have one name"
    pure { base, own := { programJ, program, partition, circuit := "", registered := none }, members }
  else
    let (circuit, registered) ← circuitOf pr
    pure { base, own := { programJ, program, partition, circuit, registered } }

/-- The `verdicts` command's own inputs: exactly `{"circuit", "partition", "program", "registered"}`, the program and
partition objects. -/
def ownOfJson (j : Json) : V Own := do
  exactly j #["circuit", "partition", "program", "registered"] "the verifier's own inputs"
  let (circuit, registered) ← circuitOf j
  let (programJ@(.obj _), partition@(.obj _)) := (getD j "program", getD j "partition")
    | throw "the program and the partition are not objects"
  pure { programJ, program := (← Flock.canon programJ).toUTF8, partition, circuit, registered }

private def sessionOf (j : Json) : V RunFiles := do
  exactly j #["name", "record", "reps"] "a session of the proofs"
  let (.str name, .str record, .obj _) := (getD j "name", getD j "record", getD j "reps")
    | throw "a session of the proofs is not a name, its record's digest and its rep files' digests"
  let reps ← (entries (getD j "reps")).mapM fun (f, d) => match d with
    | .str d => pure (f, d)
    | _ => throw s!"session {name}: rep file {f}'s digest is not a string"
  pure { name, record, reps }

/-- The evidence's `proofs`: the public file's digest and the sessions. -/
def proofsOf (pr : Json) : V (String × Array RunFiles) := do
  exactly pr #["public", "sessions"] "the evidence's proofs"
  let .str pub := getD pr "public" | throw "proofs: public is not a SHA-512 digest"
  let .arr ss := getD pr "sessions" | throw "proofs: sessions is not a list"
  pure (pub, ← ss.mapM sessionOf)

/-- A group form's member in the evidence: exactly `{"name", "public", "sessions"}`. -/
private def memberOf (j : Json) : V Member := do
  exactly j #["name", "public", "sessions"] "a member of the evidence's proofs"
  let .str name := getD j "name" | throw "proofs: a member's name is not a string"
  let (pub, sessions) ← proofsOf (Json.mkObj (without j "name"))
  pure { name, pub, sessions }

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  if !(j matches .obj _) then throw "the evidence is not an object"
  if (field? j "verdicts").isSome then
    throw "the evidence names verdicts: this audit's verdicts are its own run of the verifier on the proofs"
  let some pr := field? j "proofs" | throw "the evidence names no proofs (the public file and the sessions' files)"
  let base ← OneStage.evidenceOf (← Flock.canon (Json.mkObj (without j "proofs" ++ [("verdicts", Json.mkObj [])]))).toUTF8
  if (field? pr "members").isSome then
    exactly pr #["members"] "the evidence's proofs"
    let .arr ms := getD pr "members" | throw "proofs: members is not a list"
    let members ← ms.mapM memberOf
    if (members.flatMap (·.sessions)).map (·.name) != base.sessions then
      throw "proofs: the members' sessions are not the evidence's sessions, in order"
    pure { base, pub := "", sessions := #[], members }
  else
    let (pub, sessions) ← proofsOf pr
    if sessions.map (·.name) != base.sessions then throw "proofs: the sessions are not the evidence's sessions, in order"
    pure { base, pub, sessions }

/-- `flock-verify`'s `checkIdentifier` without the verifier's own identifier: a public header that names one refuses. -/
def checkIdentifier (st : Flock.Stmt) : V Unit :=
  match st.pub.header.getObjVal? "identifier" with
  | .error _ => pure ()
  | .ok j => do
    let .obj _ := j | throw "identifier: the public header's identifier is not a JSON object"
    if let some e := Flock.Program.domainError j "identifier" then
      throw s!"identifier: the public header's identifier is outside canonical JSON's RFC 8785 domain: {e}"
    throw "identifier: the public header names an identifier; pass the verifier's own (--identifier F)"

/-- `flock-verify`'s `buildSessionOf` on an hm96-sha512 row statement with none of `--work`, `--stratified`,
`--closure`, `--public-inputs` or `--identifier`. -/
def buildSessionOf (parsed : V Flock.Circuit) (tags : Flock.Tags) (circuitB pubB : ByteArray) (coins : Flock.Coins)
    (tables : List (String × ByteArray)) (count : Nat) (draw partition : Option Json) (program : Option ByteArray)
    (registered : Option (Array Flock.Registered.Port)) : V (Flock.SessionSpec × Array Flock.Stmt) := do
  (if !tags.hm96Rows then throw s!"{tags.name}: not an hm96-sha512 row statement" else pure () : V Unit)
  let r ← (if tags.hiddenOutputs then
      Flock.Stmt.setupTablesOf parsed tags circuitB pubB coins count tables draw partition program
    else Flock.Stmt.setupTables tags circuitB pubB coins count tables draw partition program :
      V (Flock.SessionSpec × Array Flock.Stmt))
  for st in r.2 do
    Flock.checkRegistered st registered
    checkIdentifier st
  Flock.Registered.checkFull registered draw
  return r

/-- `flock-verify`'s `programDigest`: SHA-512 of the canonical descriptor without `annotations`. -/
def programDigest (b : ByteArray) : Option String := do
  let j ← (Json.parse (← String.fromUTF8? b)).toOption
  let bare := match j with
    | .obj kvs => Json.mkObj (kvs.toArray.filter (·.1 != "annotations")).toList
    | x => x
  return Flock.hexOf (Flock.Sha512.hash (← (Flock.canon bare).toOption).toUTF8)

private def headerLine (b : ByteArray) : ByteArray := Id.run do
  for i in [0:b.size] do
    if b[i]! == 10 then return b.extract 0 i
  return b

/-- `one_stage.proofs.proved_units`: the public file's unit indices (`units.indices`) of the instances the session's
record drew (`unit_draw.units`), ascending. -/
def provedUnits (pubB : ByteArray) (record : String) : V (Array Nat) := do
  let some line := String.fromUTF8? (headerLine pubB) | throw "the public file's header is not UTF-8"
  let h ← Json.parse line
  let .arr is := getD (getD h "units") "indices" | throw "the public file's header names no units.indices"
  let .arr ds := getD (getD (← Json.parse record) "unit_draw") "units" | throw "the record's unit_draw names no units"
  let units ← ds.mapM fun d => match nat? d >>= fun i => is[i]? >>= nat? with
    | some u => pure u
    | none => throw "the record draws an instance the public file's units.indices does not name"
  return units.qsort (· < ·)

/-- The pinned MUFU lookup tables the blobs hold, each found under its pinned SHA-512 as `flock-verify --archive` finds
it. -/
def tablesOf (bs : Core.Blobs) : List (String × ByteArray) :=
  Flock.MUFU_TABLES.filterMap fun t =>
    (bs.entries.find? fun (d, _) => Flock.hexOf d == t.sha512).map fun (_, b) => (t.name, b)

/-- What `flock-verify verify --zk` reads besides a session's directory, as `verdicts` runs it: the statement's tags
`tags`, the circuit and public files, one table, the lookup tables the blobs hold, and the verifier's own partition
object and program. -/
def inputsOf (tags : Flock.Tags) (p : Own) (circuitB pubB : ByteArray) (bs : Core.Blobs) : Flock.Witness.Inputs :=
  { tags, circuitFile := circuitB, publicFile := pubB, count := 1, tables := tablesOf bs, partition := some p.partition,
    program := some p.program }

/-- The verifier's registered reads, as `verifyCmd` derives them from its registered file: its own ports, the reads it
knows, and the registered parameters. -/
structure Reads where
  registered : Option (Array Flock.Registered.Port)
  known : Array (String × Array (Array Nat × ByteArray))
  params : Array String

def readsOf : Option Json → V Reads
  | none => pure ⟨none, #[], #[]⟩
  | some j => do
    let own ← Flock.Registered.ofJson j
    let known ← Flock.Registered.knownOf j
    Flock.Registered.checkKnownAll own known
    pure ⟨some own, known ++ Flock.KnownRead.fullRows own (← Flock.KnownRead.readOf j), ← Flock.Registered.paramsOf j⟩

/-- The checks `verifyCmd` makes on a session's tables once they are set up, before `Zk.shapeOk`. -/
def setupChecks (I : Flock.Witness.Inputs) (rd : Reads) (sts : Array Flock.Stmt) : V Unit := do
  if let some own := rd.registered then sts.forM fun (st : Flock.Stmt) => Flock.Registered.checkComplete st.c own
  sts.forM fun (st : Flock.Stmt) => Flock.Registered.checkCover st.c I.program I.partition rd.registered rd.params
  sts.forM fun (st : Flock.Stmt) => Flock.KnownRead.check st.c rd.known
  Flock.RecSetup.checks none none I.tags I.tables sts I.circuitFile I.partition I.program rd.registered none none none
  Flock.RecSetup.oneFull none sts

/-- The checks `verifyCmd` makes on a session's tables after `Zk.shapeOk`. -/
def recChecks (rd : Reads) (sts : Array Flock.Stmt) : V Unit := do
  sts.forM (Flock.RecCoins.check · rd.registered (Flock.RecCoins.Source.atC3 none none none) (Flock.RecSetup.c3 none none))
  sts.forM (Flock.RecVStar.checkPubRows · rd.registered none none)
  sts.forM (Flock.RecVStar.checkTopRows · rd.registered none)
  sts.forM (Flock.RecCoins.checkLink · rd.registered (Flock.RecCoins.Source.atC3 none none none) none)

/-- The session `verifyCmd` sets up at `draw` on `I` (`buildSessionOf` under `--zk`'s tags and coins), with the checks it
makes there. -/
def sessionAt (I : Flock.Witness.Inputs) (rd : Reads) (draw : Option Json) : V (Flock.SessionSpec × Array Flock.Stmt) := do
  let tags := Flock.Witness.zkTags I
  Flock.Registered.checkProgram I.program rd.registered.isSome
  Flock.RecSetup.checkInner none none (pure (none, none))
  let r ← buildSessionOf (Flock.ProvedScope.circuitOf tags I.circuitFile I.tables) tags I.circuitFile I.publicFile
    (Flock.Witness.zkCoins I) I.tables I.count draw I.partition I.program rd.registered
  setupChecks I rd r.2
  r.2.forM Flock.Zk.shapeOk
  recChecks rd r.2
  pure r

/-- `sessionAt`'s session as `verify` reads it: the tables' verifier-side setups, the partition finding and the
registered openings, once the draw's law holds. -/
def setupAt (I : Flock.Witness.Inputs) (rd : Reads) (draw : Option Json) :
    V (Flock.SessionSpec × Array Flock.Setup × Json × Json) := do
  let (spec, sts) ← (sessionAt I rd draw).mapError (s!"setup: {·}")
  (if draw.isSome then Flock.ProvedScope.drawLaw true (Flock.Witness.zkTags I) false draw else pure () : V Unit)
  pure (spec, sts.map Flock.Setup.ofCircuit, (sts[0]?.map (·.partition)).getD Json.null,
    Flock.registeredOpenings sts rd.registered)

/-- What `verdicts` carries from one session to the next: the coin-tree keys the earlier sessions used, the verdicts so
far, the units the accepted ones proved, and whether every one accepted. -/
structure Run where
  coinKeys : Array String
  out : Array (String × Json)
  proved : Array Nat
  all : Bool

/-- A session's set-up statement, as `setupAt` returns it. -/
abbrev SetUp := V (Flock.SessionSpec × Array Flock.Setup × Json × Json)

/-- The session's proof files: per rep stream of its setup, the blob its `reps` name for it. -/
def filesOf (bs : Core.Blobs) (s : RunFiles) (st : SetUp) : V (Array ByteArray) := do
  let (spec, _, _) ← st
  spec.repStreams.mapM fun (_, _, w) => do
    let n := s!"{w.replace "/" "."}.bin"
    let some (_, d) := s.reps.find? (·.1 == n) | throw s!"session files: the session names no rep file {n}"
    (bs.get d s!"session files: {n}")

/-- The record's coin-tree key, when the session's setup draws its coins from a coin tree. -/
def coinKeyOf (record : String) (st : SetUp) : Option String :=
  match st with
  | .ok (spec, _, _) =>
    if spec.coinTree.isNone then none
    else (Json.parse record).toOption.bind fun j =>
      (j.getObjVal? "coin_tree").toOption.bind fun ct => (ct.getObjValAs? String "key").toOption
  | .error _ => none

/-- `verify --zk`'s verdict on one session set up as `st` at `draw`: its proof files, its coin-tree key fresh among
`coinKeys`, `Zk.verify` and the checks `verifyCmd` makes after it, its last the files' ports and roots
(`ksFiles`, `Flock.PortRoots.checkOf`) and the form `Flock.Guarantees.SampledKSLiveV4` covers at `draw`
(`Flock.KsScope.checkOf`). -/
def verdictOf (I : Flock.Witness.Inputs) (scope ksFiles : V Unit) (coinKeys : Array String) (record : String)
    (draw : Option Json) (st : SetUp) (files : V (Array ByteArray)) (coinKey : Option String) : V Unit := do
  let proofs ← files
  (match coinKey with | some k => Flock.CoinTree.checkFresh coinKeys k | none => pure () : V Unit)
  let (spec, tabs, _) ← st
  Flock.Zk.verify spec tabs record proofs
  scope
  Flock.ProvedScope.sizesOk true (tabs.map (·.m))
  (if draw.isNone then Flock.ProvedScope.drawLaw true (Flock.Witness.zkTags I) false draw else pure () : V Unit)
  Flock.RecSetup.fullDraw none false draw record
  Flock.RecSetup.innerDrawn none
  Flock.ProvedScope.programHeld I.program
  Flock.RecordText.check record
  ksFiles
  let tags := Flock.Witness.zkTags I
  Flock.KsScope.checkOf (Flock.ProvedScope.circuitOf tags I.circuitFile I.tables) tags I.publicFile I.count draw

/-- A session's verdict as `verify` writes it (`flock-verify/<name>`, of record). -/
def verdictJson (rd : Reads) (programSha : Option String) (s : RunFiles) (st : SetUp) (ok : Bool) (why : String)
    (units : Array Nat) : String × Json :=
  let (part, opened) := match st with | .ok (_, _, p, o) => (p, o) | .error _ => (Json.null, Json.null)
  (s!"flock-verify/{s.name}", Json.mkObj ([("accepted", Json.bool ok),
    ("of_record", Json.bool true), ("session", Json.str s.name), ("why", Json.str why), ("units", natsJson units)] ++
    (if part.isNull then [] else [("partition", part)]) ++
    (match programSha with | some h => [("program_sha512", Json.str h)] | none => []) ++
    (match rd.registered with | some r => [("registered", Flock.Registered.finding r)] | none => []) ++
    (if opened.isNull then [] else [("openings", opened)])))

/-- `r` with one more session's verdict `verdict`: a session that did not accept fails the run; one that did proves the
units its record drew, once it names the partition object's program (`programSha` against `want`, else a refusal). -/
def push (I : Flock.Witness.Inputs) (rd : Reads) (programSha : Option String) (want : Json) (r : Run) (s : RunFiles)
    (record : String) (st : SetUp) (coinKey : Option String) (verdict : V Unit) : V Run := do
  let coinKeys := match coinKey with | some k => r.coinKeys.push k | none => r.coinKeys
  match verdict with
  | .error why =>
    pure { coinKeys, all := false, proved := r.proved, out := r.out.push (verdictJson rd programSha s st false why #[]) }
  | .ok () =>
    if (programSha.map Json.str) != some want then
      throw s!"session {s.name}'s verdict names program {programSha.getD "none"}, not the partition object's {want.compress}"
    else do
      let units ← provedUnits I.publicFile record
      pure { coinKeys, all := r.all, proved := r.proved ++ units,
             out := r.out.push (verdictJson rd programSha s st true "" units) }

/-- One session's `flock-verify verify --zk` on the blobs, after the sessions `r` holds: set up at its record's own draw
(`setupAt`), its rep files the blobs its `reps` name, its coin-tree key fresh, `Zk.verify` and the checks `verifyCmd`
makes after it (`verdictOf`), its verdict pushed (`push`). A refusal when its record is in no blob or not UTF-8, or it
accepted naming another program than the partition object's. -/
def step (I : Flock.Witness.Inputs) (rd : Reads) (bs : Core.Blobs) (scope ksFiles : V Unit) (programSha : Option String)
    (want : Json) (r : Run) (s : RunFiles) : V Run := do
  let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
    | throw s!"session {s.name}'s record is not UTF-8"
  let draw := Flock.Witness.recordDraw record
  let st := setupAt I rd draw
  let coinKey := coinKeyOf record st
  push I rd programSha want r s record st coinKey
    (verdictOf I scope ksFiles r.coinKeys record draw st (filesOf bs s st) coinKey)

/-- One `flock-verify verify --zk` over the evidence's sessions on the blobs, as `one_stage.proofs.verify` runs it, each
session set up at its own record's draw: every session's verdict, the units the accepted ones proved and whether all
accepted. A refusal for what `proofs.verify` refuses: registered values read without the verifier's registered file, a
file no blob holds, an accepted session naming another program than the partition object's, or proved units other than
the evidence's. `runFrom` folds from the run `r0` of the sessions before (a group's earlier members'), `run` from none. -/
def runFrom (r0 : Run) (p : Own) (pub : String) (sessions : Array RunFiles) (provedUnits? : Option (Array Nat))
    (bs : Core.Blobs) : V Run := do
  let some tags0 := Flock.Tags.find? STATEMENT | throw s!"the statement {STATEMENT} is not known"
  (if tags0.plainLeaves then throw s!"{tags0.name}: a statement with plain SHA-512 leaves: --zk's hiding needs hm96 leaves"
    else pure () : V Unit)
  (if p.registered.isNone && readsRegistered p.programJ then
    throw ("registered: the program reads registered values (a RegisteredInput among its definitions) and the " ++
      "verifier holds no registered reads: verify with its own registered file")
    else pure () : V Unit)
  let circuitB ← bs.get p.circuit "circuit"
  let pubB ← bs.get pub "public file"
  let I := inputsOf tags0 p circuitB pubB bs
  let rd ← readsOf p.registered
  let tags := Flock.Witness.zkTags I
  let scope := Flock.ProvedScope.checkOf (Flock.ProvedScope.circuitOf tags I.circuitFile I.tables) tags true pubB I.count
    false rd.registered.isSome
  let ksFiles := Flock.PortRoots.checkOf (Flock.ProvedScope.circuitOf tags I.circuitFile I.tables) tags pubB
  let r ← sessions.foldlM (step I rd bs scope ksFiles (programDigest p.program) (getD p.partition "program")) r0
  match provedUnits? with
  | some want =>
    if r.all && r.proved.qsort (· < ·) != want then throw "the units the sessions proved are not the evidence's proved_units"
    else pure r
  | none => pure r

def run (p : Own) (pub : String) (sessions : Array RunFiles) (provedUnits? : Option (Array Nat)) (bs : Core.Blobs) :
    V Run :=
  runFrom ⟨#[], #[], #[], true⟩ p pub sessions provedUnits? bs

/-- The sessions' verdicts in order (`run`'s). -/
def verdicts (p : Own) (pub : String) (sessions : Array RunFiles) (provedUnits? : Option (Array Nat)) (bs : Core.Blobs) :
    V (Array (String × Json)) :=
  (·.out) <$> run p pub sessions provedUnits? bs

/-- The coins the draw read (module docstring, **coins**): v1's, which only the parameters' challenger gives; a refusal
when none read. -/
def coinsOf (p : Params) (c : OneStage.Commitment) (e : Evidence) : V ByteArray := do
  match ← OneStage.coinsOf p.base c e.base with
  | some ξ => pure ξ
  | none => throw s!"coins: {NO_CHALLENGER}"

/-- The single form at the coins `ξ` (module docstring, **check**): the sessions' verdicts on the blobs, then v1's
`checkAt` at `ξ` on them. v1 accepts only verdicts that all accepted (`VerdictsAccept`); the run's own `all` says it
again, so an acceptance reads the run rather than the verdicts' JSON. -/
def checkOneAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match run p.own e.pub e.sessions (some e.base.provedUnits) bs with
  | .error err => .reject s!"proofs: {err}"
  | .ok r =>
    match OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
    | .accept pub prof => if r.all then .accept pub prof else .reject "proofs: a session's verifier did not accept"
    | o => o

/-- The members' leaf layer (`scan_zk.audit`'s `leaf_layer`): the SHA-512, as lowercase hex, of the canonical object
naming each member's public file's SHA-512 by the member's name. -/
def leafLayer (xs : Array Member) : V String := do
  let t ← Flock.canon (Json.mkObj (xs.map fun x => (x.name, Json.str x.pub)).toList)
  pure (Flock.hexOf (Flock.Sha512.hash t.toUTF8))

/-- Whether the units are strictly increasing: each once, in order. -/
def increasing : List Nat → Bool
  | a :: b :: l => a < b && increasing (b :: l)
  | _ => true

/-- One member's sessions on the run `r` of the members before it: the parameters' member `y.1` (its name and own
inputs) and the evidence's `y.2` (its public file and sessions). -/
def memberStep (bs : Core.Blobs) (r : Run) (y : (String × Own) × Member) : V Run :=
  (runFrom r y.1.2 y.2.pub y.2.sessions none bs).mapError (s!"member {y.1.1}: {·}")

/-- The group form's sessions: the evidence's members, the parameters' by name in order, each member's sessions run
(`runFrom`) on its own inputs and its public file, on one run, so a session's coin-tree key is fresh among every
earlier member's sessions' too. -/
def runGroup (ms : Array (String × Own)) (xs : Array Member) (bs : Core.Blobs) : V Run := do
  (if xs.map (·.name) != ms.map (·.1) then throw "the members are not the parameters' members, in order"
    else pure () : V Unit)
  (ms.zip xs).foldlM (memberStep bs) ⟨#[], #[], #[], true⟩

/-- The group form at the coins `ξ`: the registration's `leaf_layer` is the members' public files' (`leafLayer`),
committed before the draw; the members' sessions run (`runGroup`); the units they proved, once all accepted, each once
and together the evidence's `proved_units`; then v1's `checkAt` at `ξ` on the computed verdicts, as `checkOneAt`. -/
def checkGroupAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match leafLayer e.members with
  | .error err => .reject s!"proofs: the members' leaf layer: {err}"
  | .ok layer =>
    if getD c.record "leaf_layer" != .str layer then
      .reject "proofs: the registration's leaf_layer is not the members' public files' leaf layer"
    else
    match runGroup p.members e.members bs with
    | .error err => .reject s!"proofs: {err}"
    | .ok r =>
      if r.all && (r.proved.qsort (· < ·) != e.base.provedUnits || !increasing e.base.provedUnits.toList) then
        .reject "proofs: the units the members' sessions proved are not the evidence's proved_units, each once"
      else
      match OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
      | .accept pub prof => if r.all then .accept pub prof else .reject "proofs: a session's verifier did not accept"
      | o => o

/-- The check at the coins `ξ`: the single form with no member in the parameters or the evidence (`checkOneAt`), else
the group form (`checkGroupAt`). -/
def checkBlobsAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  if p.members.isEmpty && e.members.isEmpty then checkOneAt p c ξ e bs else checkGroupAt p c ξ e bs

/-- The audit's check on the blobs: at the coins `coinsOf` reads (`checkBlobsAt`), refused with its reason when none
read. -/
def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  Core.atCoins (coinsOf p c e) (fun ξ => checkBlobsAt p c ξ e bs)

def protocol : Core.BlobProtocol where
  id := PROTOCOL
  Params := Params
  Commitment := OneStage.Commitment
  Evidence := Evidence
  Public := OneStage.Public
  Profile := Core.Integrity
  params := paramsOfJson
  commitment := OneStage.commitmentOf
  evidence := evidenceOf
  digest c := c.digest
  check p c e := checkBlobs p c e Core.Blobs.empty
  publicJson := OneStage.publicJson
  profileJson := Core.Integrity.toJson
  checkBlobs := checkBlobs

/-- The audit on the blobs `bs` reads its draw's coins (`coinsOf`) and decides at them (`checkBlobsAt`). -/
def live (bs : Array ByteArray) : Core.Live (protocol.on bs) where
  Coins := ByteArray
  coins := coinsOf
  checkAt p c ξ e := checkBlobsAt p c ξ e (Core.Blobs.ofArray bs)
  factors _ _ _ := rfl

/-- `protocol.main`'s commands, and `verdicts OWN PROOFS [BLOB ...]`: the sessions' verdicts alone (`verdicts`, with no
proved units to bind), from the verifier's own inputs (`ownOfJson`) and the evidence's `proofs` (`proofsOf`), each file
canonical JSON; one line, `{"verdicts": [verdict, ...]}` in session order or `{"refused": why}`, exit 0 or 1. The
difftest holds it to `flock-verify verify --zk` (`one_stage.proofs.verify`). -/
def main (args : List String) : IO UInt32 := do
  match args with
  | "verdicts" :: own :: proofs :: blobs =>
    let bs ← blobs.toArray.mapM fun (f : String) => IO.FS.readBinFile f
    let ownB ← IO.FS.readBinFile own
    let proofsB ← IO.FS.readBinFile proofs
    let r : V (Array (String × Json)) := do
      let (pub, sessions) ← proofsOf (← Core.canonicalJson proofsB)
      verdicts (← ownOfJson (← Core.canonicalJson ownB)) pub sessions none (Core.Blobs.ofArray bs)
    let line := match r with
      | .ok vs => Json.mkObj [("verdicts", Json.arr (vs.map (·.2)))]
      | .error e => Json.mkObj [("refused", Json.str e)]
    IO.println ((Flock.canon line).toOption.getD line.compress)
    return if r.toBool then 0 else 1
  | _ => protocol.main args

#guard registeredInputName "RegisteredInput96_v1"
#guard registeredInputName "RegisteredInput1_v1"
#guard !registeredInputName "RegisteredInput0_v1"
#guard !registeredInputName "RegisteredInput01_v1"
#guard !registeredInputName "RegisteredInput_v1"
#guard !registeredInputName "RegisteredInput96_v2"
#guard !registeredInputName "XRegisteredInput96_v1"
#guard readsRegistered (Json.mkObj [])
#guard !readsRegistered (Json.mkObj [("definitions", Json.mkObj [("Rope_v1", Json.null)])])
#guard readsRegistered (Json.mkObj [("definitions", Json.mkObj [("RegisteredInput96_v1", Json.null)])])
#guard (provedUnits "{\"units\":{\"indices\":[7,3,9]}}\nrest".toUTF8 "{\"unit_draw\":{\"units\":[2,0]}}").toOption
  == some #[7, 9]
#guard (provedUnits "{\"units\":{\"indices\":[7]}}".toUTF8 "{\"unit_draw\":{\"units\":[1]}}").toOption.isNone
#guard increasing [] && increasing [4] && increasing [1, 2, 5]
#guard !increasing [1, 1] && !increasing [2, 1] && !increasing [1, 3, 2]
#guard (leafLayer #[{ name := "b", pub := "11", sessions := #[] }, { name := "a", pub := "22", sessions := #[] }]).toOption
  == some ("8e82810f38502b9e2b3fb50e258fa77b09e2146451cadf7f921d9ca9ec10277b" ++
    "171a0e19fdf168302d22fb608ea48a13a8d43478ddda665146e2b0aa46dc9066")

end SampledProofs.OneStageZk
