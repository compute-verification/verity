/-
Sampled proofs' one-stage audit on the proofs themselves (`verity/sampled-proofs/one-stage/v2`), a `Core.BlobProtocol`: the
v1 audit (`OneStage`) with the verdicts computed here, from the statement's files, rather than read from the evidence.

* **params**: v1's, plus `proofs`, exactly `{"circuit", "registered"}`: the SHA-512 the auditor committed for the circuit
  file (lowercase hex of 64 bytes) and the verifier's own registered file (`flock-verify --registered`'s object, or null
  for a program that reads no registered value). The partition must not be null: the sessions are verified against the
  verifier's own partition object;
* **commitment**: v1's (the registration record);
* **evidence**: v1's without `verdicts`, plus `proofs`, exactly `{"public", "sessions"}`: the public file's SHA-512 and,
  for each of v1's `sessions` in order, `{"name", "record", "reps"}`: its name, its `session.json`'s SHA-512 and
  `{file: SHA-512}` naming at least each rep file the statement's session reads (`<stream with / as .>.bin`);
* **blobs**: the files those digests name (`Core.Blobs`), and the MUFU tables any lookup reads, found under their pinned
  SHA-512 as `flock-verify --archive` finds them;
* **check**: `flock-verify verify --zk --statement verity/flock-circuit` with `--circuit`, `--public`, `--partition`,
  `--program` and `--registered` the verifier's own and one `--session` a session, as `one_stage.proofs.verify` runs it
  (`verdicts`, the C-Flock library calls `verifyCmd` makes on that path); then v1's `check` on those verdicts, one of
  record a session (`flock-verify/<name>`). A session accepted under another program than the partition object's, proved
  units other than the evidence's `proved_units`, or a circuit, public or record file no blob holds, rejects; a rep
  file no blob holds is that session's reject, as a file `flock-verify` can't read is.

Without blobs (`toProtocol`) every check rejects: no blob holds the circuit file.
-/
import CoreProtocol.Blobs
import SampledProofs.OneStage
import SampledProofs.ChallengerRounds
import SampledProofs.Hello
import Flock

namespace SampledProofs.OneStageZk

open Lean (Json)
open Flock (V)

def PROTOCOL : String := "verity/sampled-proofs/one-stage/v2"
/-- The statement every drawn unit's session proves (`one_stage.proofs.STATEMENT`). -/
def STATEMENT : String := "verity/flock-circuit"

/-- The verifier's own inputs to `flock-verify verify --zk`: the program's descriptor (as JSON and as the bytes
`--program` reads), the partition object, the circuit file's committed SHA-512 and the registered file. -/
structure Own where
  programJ : Json
  program : ByteArray
  partition : Json
  circuit : String
  registered : Option Json

structure Params where
  base : OneStage.Params
  own : Own

structure Session where
  name : String
  record : String
  reps : Array (String × String)
  /-- the segments of the session's uses of the challenger's coins (`sessionUses`), in order -/
  coins : Array Json := #[]

structure Evidence where
  base : OneStage.Evidence
  pub : String
  sessions : Array Session

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

def paramsOfJson (j : Json) : V Params := do
  let some pr := field? j "proofs"
    | throw "the parameters name no proofs (the auditor's circuit digest and the verifier's registered reads)"
  exactly pr #["circuit", "registered"] "proofs"
  let (circuit, registered) ← circuitOf pr
  let rest := Json.mkObj (without j "proofs")
  let base ← OneStage.paramsOfJson rest
  let some partition := base.verifier.partition
    | throw "partition is null: the sessions are verified against the verifier's own partition object"
  let programJ := getD rest "program"
  pure { base, own := { programJ, program := (← Flock.canon programJ).toUTF8, partition, circuit, registered } }

/-- The `verdicts` command's own inputs: exactly `{"circuit", "partition", "program", "registered"}`, the program and
partition objects. -/
def ownOfJson (j : Json) : V Own := do
  exactly j #["circuit", "partition", "program", "registered"] "the verifier's own inputs"
  let (circuit, registered) ← circuitOf j
  let (programJ@(.obj _), partition@(.obj _)) := (getD j "program", getD j "partition")
    | throw "the program and the partition are not objects"
  pure { programJ, program := (← Flock.canon programJ).toUTF8, partition, circuit, registered }

private def sessionOf (j : Json) : V Session := do
  let withCoins := (field? j "coins").isSome
  exactly j (if withCoins then #["coins", "name", "record", "reps"] else #["name", "record", "reps"])
    "a session of the proofs"
  let (.str name, .str record, .obj _) := (getD j "name", getD j "record", getD j "reps")
    | throw "a session of the proofs is not a name, its record's digest and its rep files' digests"
  let reps ← (entries (getD j "reps")).mapM fun (f, d) => match d with
    | .str d => pure (f, d)
    | _ => throw s!"session {name}: rep file {f}'s digest is not a string"
  let coins ← match getD j "coins" with
    | .arr ss => pure ss
    | .null => pure #[]
    | _ => throw s!"session {name}: coins is not a list of segments"
  pure { name, record, reps, coins }

/-- The evidence's `proofs`: the public file's digest and the sessions. -/
def proofsOf (pr : Json) : V (String × Array Session) := do
  exactly pr #["public", "sessions"] "the evidence's proofs"
  let .str pub := getD pr "public" | throw "proofs: public is not a SHA-512 digest"
  let .arr ss := getD pr "sessions" | throw "proofs: sessions is not a list"
  pure (pub, ← ss.mapM sessionOf)

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  if !(j matches .obj _) then throw "the evidence is not an object"
  if (field? j "verdicts").isSome then
    throw "the evidence names verdicts: this audit's verdicts are its own run of the verifier on the proofs"
  let some pr := field? j "proofs" | throw "the evidence names no proofs (the public file and the sessions' files)"
  let base ← OneStage.evidenceOf (← Flock.canon (Json.mkObj (without j "proofs" ++ [("verdicts", Json.mkObj [])]))).toUTF8
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

/-- A session's set-up statement, as `setupAt` returns it. -/
abbrev SetUp := V (Flock.SessionSpec × Array Flock.Setup × Json × Json)

/-- What `verdicts` carries from one session to the next: the coin-tree keys the earlier sessions used, the verdicts so
far, the units the accepted ones proved, whether every one accepted, and the setups so far by their draw's canonical
JSON, so sessions at one draw share one setup. -/
structure Run where
  coinKeys : Array String
  out : Array (String × Json)
  proved : Array Nat
  all : Bool
  setups : Array (Option String × SetUp)

/-- The session's proof files: per rep stream of its setup, the blob its `reps` name for it. -/
def filesOf (bs : Core.Blobs) (s : Session) (st : SetUp) : V (Array ByteArray) := do
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
`coinKeys`, `Zk.verify` and the checks `verifyCmd` makes after it. -/
def verdictOf (I : Flock.Witness.Inputs) (scope : V Unit) (coinKeys : Array String) (record : String)
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

/-- A session's verdict as `verify` writes it (`flock-verify/<name>`, of record). -/
def verdictJson (rd : Reads) (programSha : Option String) (s : Session) (st : SetUp) (ok : Bool) (why : String)
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
def push (I : Flock.Witness.Inputs) (rd : Reads) (programSha : Option String) (want : Json) (r : Run) (s : Session)
    (record : String) (st : SetUp) (coinKey : Option String) (verdict : V Unit) : V Run := do
  let coinKeys := match coinKey with | some k => r.coinKeys.push k | none => r.coinKeys
  match verdict with
  | .error why =>
    pure { r with coinKeys, all := false, out := r.out.push (verdictJson rd programSha s st false why #[]) }
  | .ok () =>
    if (programSha.map Json.str) != some want then
      throw s!"session {s.name}'s verdict names program {programSha.getD "none"}, not the partition object's {want.compress}"
    else do
      let units ← provedUnits I.publicFile record
      let out := r.out.push (verdictJson rd programSha s st true "" units)
      pure { r with coinKeys, proved := r.proved ++ units, out }

/-- One session's `flock-verify verify --zk` on the blobs, after the sessions `r` holds: set up at its record's own draw
(`setupAt`, once per draw: a draw `r` has set up reuses its setup), its rep files the blobs its `reps` name, its
coin-tree key fresh, `Zk.verify` and the checks `verifyCmd` makes after it (`verdictOf`), its verdict pushed (`push`). A
refusal when its record is in no blob or not UTF-8, or it accepted naming another program than the partition object's. -/
def step (I : Flock.Witness.Inputs) (rd : Reads) (bs : Core.Blobs) (scope : V Unit) (programSha : Option String)
    (want : Json) (r : Run) (s : Session) : V Run := do
  let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
    | throw s!"session {s.name}'s record is not UTF-8"
  let draw := Flock.Witness.recordDraw record
  let key := draw.map fun d => (Flock.canon d).toOption.getD ""
  let (st, r) := match r.setups.find? (·.1 == key) with
    | some (_, x) => (x, r)
    | none => let x := setupAt I rd draw; (x, { r with setups := r.setups.push (key, x) })
  let coinKey := coinKeyOf record st
  push I rd programSha want r s record st coinKey (verdictOf I scope r.coinKeys record draw st (filesOf bs s st) coinKey)

/-- One `flock-verify verify --zk` over the evidence's sessions on the blobs, as `one_stage.proofs.verify` runs it, each
session set up at its own record's draw: every session's verdict, the units the accepted ones proved and whether all
accepted. A refusal for what `proofs.verify` refuses: registered values read without the verifier's registered file, a
file no blob holds, an accepted session naming another program than the partition object's, or proved units other than
the evidence's. -/
def run (p : Own) (pub : String) (sessions : Array Session) (provedUnits? : Option (Array Nat)) (bs : Core.Blobs) :
    V Run := do
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
  let r ← sessions.foldlM (step I rd bs scope (programDigest p.program) (getD p.partition "program"))
    ⟨#[], #[], #[], true, #[]⟩
  match provedUnits? with
  | some want =>
    if r.all && r.proved.qsort (· < ·) != want then throw "the units the sessions proved are not the evidence's proved_units"
    else pure r
  | none => pure r

/-- The sessions' verdicts in order (`run`'s). -/
def verdicts (p : Own) (pub : String) (sessions : Array Session) (provedUnits? : Option (Array Nat)) (bs : Core.Blobs) :
    V (Array (String × Json)) :=
  (·.out) <$> run p pub sessions provedUnits? bs

/-- The session stream its link points are drawn on (`Firewall.Coins.SESSION`). -/
def POINTS : String := "session"

private def hexes (j : Json) (what : String) : V (Array String) := do
  let .arr xs := j | throw s!"{what} is not a list of hex strings"
  xs.mapM fun | .str s => pure s | _ => throw s!"{what} is not a list of hex strings"

/-- What a session's link points are released after: the SHA-512 of its tables' root_B digests
(`link.root_b_sha512`, 64 bytes each), in the record's key order. -/
def pointsAfter (link : Json) : V ByteArray := do
  let .obj _ := getD link "root_b_sha512" | throw "record: link.root_b_sha512 is not an object"
  let ds ← (entries (getD link "root_b_sha512")).mapM fun (t, d) => match d with
    | .str h => if h.length == 128 then Flock.unhex h else throw s!"record: link.root_b_sha512.{t} is not 64 bytes of hex"
    | _ => throw s!"record: link.root_b_sha512.{t} is not 64 bytes of hex"
  return Flock.Sha512.hash (ds.foldl (· ++ ·) .empty)

/-- Session `slot`'s uses of the challenger's coins, each with the coins (as the record's F128 hex) its bytes must
begin with: its link points, of family `slot/<slot>/session` after `pointsAfter`, then each stream's rounds in the
record's order, of family `slot/<slot>/<stream>` after the round's `msg_sha512` (`Round.sha`); `⌈n/4⌉` coins of 64
bytes for `n` F128s. -/
def sessionUses (slot : Nat) (record : String) : V (List (Challenger.Use × Array String)) := do
  let j ← (Json.parse record).mapError (s!"record: not JSON: {·}")
  let link := getD j "link"
  let .arr ps := getD link "points" | throw "record: link.points is not a list"
  let pts ← ps.foldlM (fun acc p => return acc ++ (← hexes p "a link point")) #[]
  let mut out := [({ label := Hello.slotStream slot POINTS, after := ← pointsAfter link, coins := (pts.size + 3) / 4 },
    pts)]
  let .arr ss := getD j "streams" | throw "record: streams is not a list"
  for s in ss do
    let .str name := getD s "stream" | throw "record: a stream's name is not a string"
    if name == POINTS then throw s!"record: a stream is named {POINTS}, the link points' family"
    let .arr rs := getD s "rounds" | throw s!"record: {name}'s rounds are not a list"
    for r in rs do
      let .str sha := getD r "msg_sha512" | throw s!"record: a round of {name} has no msg_sha512"
      if sha.length != 128 then throw s!"record: a round of {name}'s msg_sha512 is not 64 bytes of hex"
      let cs ← hexes (getD r "coins") s!"the coins of a round of {name}"
      out := out ++ [({ label := Hello.slotStream slot name, after := ← Flock.unhex sha, coins := (cs.size + 3) / 4 }, cs)]
  return out

/-- Why the coins `ξ` don't begin with the F128s `want` (record hex, `{hi:016x}{lo:016x}`, 16 bytes each); none when
they do. -/
def coinsMismatch (want : Array String) (ξ : ByteArray) : Option String :=
  (want.toList.zipIdx.find? fun (h, i) => Flock.hexOf (ξ.extract (16 * i) (16 * i + 16)) != h).map fun (_, i) =>
    s!"coin {i} is not the challenger's"

def ROUNDS_NEED_CHALLENGER : String :=
  "v2 reads every session's link points and round coins from the auditor's challenger, and the parameters name " ++
  "none (challenger and window_root)"

/-- Why the window's coins are not all the challenger's (`Challenger.roundCoins`), one release a use; none when they
are. The uses are the draw's (its family, after the registration's digest, as many coins as its segment opens) and
each session's (`sessionUses`, slot `i` the session's index), each with its segment: the draw's `coins.segment`, then
each session's `coins`. The log must release the draw before any of a session's families, and a session's link
points before its rounds (`order`), and each use of a session's coins must be one run among that session's releases
(`groups`, one a session). Every coin a session's record holds must be the challenger's (`coinsMismatch`). -/
def roundsWhy (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) : Option String :=
  match p.base.challenger, p.base.verifier.windowRoot with
  | some (key, label), some w =>
    let r : V Unit := do
      let draw := getD e.base.coins "segment"
      let .arr opened := getD draw "openings" | throw Challenger.SEGMENT_FORM
      let mut uses : Array Challenger.Use := #[{ label, after := c.digest, coins := opened.size }]
      let mut wants : Array (Option (String × Array String)) := #[none]
      let mut segments : Array Json := #[draw]
      let mut order : Array (String × String) := #[]
      let mut groups : Array (List String) := #[]
      for (s, i) in e.sessions.toList.zipIdx do
        let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
          | throw s!"session {s.name}'s record is not UTF-8"
        let us ← (sessionUses i record).mapError (s!"session {s.name}: {·}")
        unless s.coins.size == us.length do
          throw s!"session {s.name}: {s.coins.size} segments for its {us.length} uses of the challenger's coins"
        let points := Hello.slotStream i POINTS
        let fams := (us.map (·.1.label)).eraseDups
        order := order ++ (fams.map (label, ·)).toArray ++ ((fams.filter (· != points)).map (points, ·)).toArray
        groups := groups.push fams
        uses := uses ++ us.toArray.map (·.1)
        wants := wants ++ us.toArray.map fun (u, hs) => some (s!"{s.name}'s {u.label}", hs)
        segments := segments ++ s.coins
      let coins := Json.mkObj [("ledger", getD e.base.coins "ledger"), ("main", getD e.base.coins "main"),
        ("pins", getD e.base.coins "pins"), ("segments", Json.arr segments), ("session", getD e.base.coins "session")]
      let ξs ← Challenger.roundCoins key w uses.toList order.toList groups.toList coins
      for (ξ, want) in ξs.zip wants.toList do
        if let some (what, hs) := want then
          if let some why := coinsMismatch hs ξ then throw s!"session {what}: {why}"
    match r with
    | .ok () => none
    | .error why => some why
  | _, _ => some ROUNDS_NEED_CHALLENGER

/-- v2's check: the sessions' verdicts (`run`), then v1's check on them at the draw's coins (`OneStage.checkLive`:
`OneStage.coinsOf`, `OneStage.checkAt`), an accept only when every session's verifier accepted (the run's own `all`, so an
acceptance reads the run rather than the verdicts' JSON), and only when every coin the window read is the challenger's,
one release a use (`roundsWhy`). -/
def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match run p.own e.pub e.sessions (some e.base.provedUnits) bs with
  | .error err => .reject s!"proofs: {err}"
  | .ok r =>
    match OneStage.checkLive p.base c { e.base with verdicts := r.out } with
    | .accept pub prof =>
      if r.all then
        match roundsWhy p c e bs with
        | some why => .reject s!"proofs: round coins: {why}"
        | none => .accept pub prof
      else .reject "proofs: a session's verifier did not accept"
    | o => o

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

/-- `protocol.main`'s commands, and `verdicts OWN PROOFS [BLOB ...]`: the sessions' verdicts alone (`verdicts`, with no
proved units to bind), from the verifier's own inputs (`ownOfJson`) and the evidence's `proofs` (`proofsOf`), each file
canonical JSON; one line, `{"verdicts": [verdict, ...]}` in session order or `{"refused": why}`, exit 0 or 1. The
difftest holds it to `flock-verify verify --zk` (`one_stage.proofs.verify`). `uses SLOT RECORD`: the session record's
uses of the challenger's coins at slot `SLOT` (`sessionUses`), one line `{"uses": [{after, coins, label, want}, ...]}`
or `{"refused": why}`, which `one_stage.proofs.session_uses` is held to. -/
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
  | ["uses", slot, record] =>
    let text ← IO.FS.readFile record
    let r : V (List (Challenger.Use × Array String)) := do
      let some i := slot.toNat? | throw "the slot is not a natural number"
      sessionUses i text
    let line := match r with
      | .ok us => Json.mkObj [("uses", Json.arr (us.toArray.map fun (u, hs) => Json.mkObj [("after",
          Json.str (Flock.hexOf u.after)), ("coins", Lean.toJson u.coins), ("label", Json.str u.label),
          ("want", Json.arr (hs.map Json.str))]))]
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
#guard ((sessionUses 1 ("{\"link\":{\"points\":[[\"" ++ "aa".pushn 'a' 30 ++ "\"]],\"root_b_sha512\":{\"t\":\"" ++
    "".pushn '0' 128 ++ "\"}},\"streams\":[{\"stream\":\"t/rep0\",\"rounds\":[{\"msg_sha512\":\"" ++ "".pushn '1' 128 ++
    "\",\"coins\":[\"a\",\"b\",\"c\",\"d\",\"e\"]}]}]}")).toOption.map
    (·.map fun (u, hs) => (u.label, u.coins, hs.size))) == some [("slot/1/session", 1, 1), ("slot/1/t/rep0", 2, 5)]
#guard coinsMismatch #["00".pushn '0' 30] (ByteArray.mk (Array.replicate 64 0)) == none
#guard (coinsMismatch #["01".pushn '0' 30] (ByteArray.mk (Array.replicate 64 0))).isSome

end SampledProofs.OneStageZk
