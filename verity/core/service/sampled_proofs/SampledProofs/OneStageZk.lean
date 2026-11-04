/-
Sampled proofs' one-stage audit on the proofs themselves (`verity/sampled-proofs/one-stage/v2`), a `Core.BlobProtocol`: the
v1 audit (`OneStage`) with the verdicts computed here, from the statement's files, rather than read from the evidence.

* **params**: v1's, plus `proofs`, in one of two forms (`Form`). The one-statement form is exactly `{"circuit",
  "registered"}`: the SHA-512 the auditor committed for the circuit file (lowercase hex of 64 bytes) and the verifier's
  own registered file (`flock-verify --registered`'s object, or null for a program that reads no registered value). The
  per-member form is exactly `{"members"}`, an object from each member's name (a statement's units of the population) to
  exactly `{"units", "circuit", "registered"}`: its units as `[lo, hi)` ranges, no unit in two members' ranges, and its
  statement's circuit digest and registered file, as the one-statement form's. The partition must not be null: the
  sessions are verified against the verifier's own partition object; and v1's `challenger`, with its `window_root`, must
  be named (`NO_CHALLENGER`);
* **commitment**: v1's (the registration record);
* **evidence**: v1's without `verdicts`, plus `proofs`: in the one-statement form exactly `{"public", "sessions"}`, the
  public file's SHA-512 and, per session, `{"name", "record", "reps"}`: its name, its `session.json`'s SHA-512 and
  `{file: SHA-512}` naming at least each rep file the statement's session reads (`<stream with / as .>.bin`); in the
  per-member form exactly `{"statements"}`, a list of `{"member", "public", "sessions"}`, a statement a drawn member.
  Either way the sessions, statement by statement, are v1's `sessions` in order;
* **blobs**: the files those digests name (`Core.Blobs`), and the MUFU tables any lookup reads, found under their pinned
  SHA-512 as `flock-verify --archive` finds them;
* **coins** (`coinsOf`): v1's (`OneStage.coinsOf`), what `Challenger.drawCoins` reads from the evidence's `coins` under
  the challenger's key, on the verifier's window; none read refuses. The check below runs at them (`checkBlobsAt`): v1's
  `checkAt` refuses a draw other than the one they give before v1's `check`;
* **trees** (`treesOf`, `treesHeld`): the coin trees that same session's signed release log shows the challenger sampled
  and opened (`Challenger.trustedTreesOf`). Each session's coins must be one tree's, a tree no other session's, its
  record's `coin_tree`, link points and every round's coins as the challenger logged them, and every tree logged must be
  a session's (`Challenger.sessionsTreeReason`); else the check refuses, before any session is verified;
* **check**: `flock-verify verify --zk --statement verity/flock-circuit` with `--circuit`, `--public`, `--partition`,
  `--program` and `--registered` the verifier's own and one `--session` a session, as `one_stage.proofs.verify` runs it
  (`verdicts`, the C-Flock library calls `verifyCmd` makes on that path, its last the files' ports and roots,
  `Flock.PortRoots.checkOf`, and the form `Flock.Guarantees.SampledKSLiveV4` covers, `Flock.KsScope.checkOf`); then v1's
  `check` on those verdicts, one of
  record a session (`flock-verify/<name>`). A session accepted under another program than the partition object's, proved
  units other than the evidence's `proved_units`, or a circuit, public or record file no blob holds, rejects; a rep
  file no blob holds is that session's reject, as a file `flock-verify` can't read is;
* **per member** (`runMembers`): before any session runs, the statements are held as public structure
  (`membersHeld`): a statement naming no member of the parameters or one an earlier statement named, statements out of
  the members' order (by first unit, the registration's: one order, one statement), or a statement of more than one
  session (one session on live coins, never retried), rejects. Then each statement is verified as the one statement is,
  on its member's circuit and registered file, after the statements before it, so no coin tree serves two statements.
  Its proofs in the other form than the parameters', or a session proving a unit outside its member's units, rejects;
  the evidence's `proved_units` are held to the union of the statements'. A stratum may span members, so a member's
  drawn count may be zero and it has no statement. Only a stratum's `k` is not read here: a statement is one table
  (`inputsOf`'s `count`), so no stratum's draw is split over a statement's tables. Every acceptance in this form is
  then refused (`MULTI_REFUSED`) until the guarantee for a sample over several members' statements is in the lock.

Without blobs (`toProtocol`) every check rejects: no blob holds the circuit file.
-/
import CoreProtocol.Blobs
import SampledProofs.OneStage
import SampledProofs.ChallengerTrees
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
/-- The refusal of the per-member form's acceptance: no guarantee in the lock yet covers a sample over several members'
statements under one draw. -/
def MULTI_REFUSED : String :=
  "the per-member form (proofs.members) is refused until the guarantee for a sample over several members' statements " ++
    "under one draw is in the lock: verify one statement (proofs {circuit, registered})"

/-- The verifier's own inputs to `flock-verify verify --zk`: the program's descriptor (as JSON and as the bytes
`--program` reads), the partition object, the circuit file's committed SHA-512 and the registered file. -/
structure Own where
  programJ : Json
  program : ByteArray
  partition : Json
  circuit : String
  registered : Option Json

/-- A member of the per-member form: its name, its units as `[lo, hi)` ranges, and its statement's own inputs. -/
structure Member where
  name : String
  units : Array (Nat × Nat)
  own : Own

/-- The parameters' `proofs` (module docstring, **params**): one statement on the verifier's own inputs, or a statement
a drawn member, each on its member's. -/
inductive Form where
  | one (own : Own)
  | members (ms : Array Member)

def Form.perMember : Form → Bool
  | .one _ => false
  | .members _ => true

structure Params where
  base : OneStage.Params
  form : Form

structure RunFiles where
  name : String
  record : String
  reps : Array (String × String)

/-- One statement of the evidence's `proofs`: the member it proves (none in the one-statement form), its public file's
SHA-512 and its sessions. -/
structure StatementFiles where
  member : Option String
  pub : String
  sessions : Array RunFiles

structure Evidence where
  base : OneStage.Evidence
  statements : Array StatementFiles

/-- The evidence's sessions, statement by statement. -/
def Evidence.sessions (e : Evidence) : Array RunFiles := e.statements.foldl (· ++ ·.sessions) #[]

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

/-- The per-member form's `members` (module docstring, **params**): each a nonempty name, exactly its keys, its units
`Draw.ranges?` of naturals, no unit in two members' ranges, and its circuit digest and registered file (`circuitOf`). -/
private def membersOf (j : Json) : V (Array (String × Array (Nat × Nat) × String × Option Json)) := do
  let .obj _ := j | throw "proofs: members is not an object from a member's name to its units, circuit and registered"
  let ms ← (entries j).mapM fun (name, m) => do
    if name.isEmpty then throw "proofs: a member's name is empty"
    exactly m #["circuit", "registered", "units"] s!"member {name}"
    let (circuit, registered) ← circuitOf m
    let some units := (Draw.ranges? (getD m "units")).bind fun rs =>
        rs.mapM fun (lo, hi) => if 0 ≤ lo then some (lo.toNat, hi.toNat) else none
      | throw s!"proofs: member {name}'s units are not a nonempty list of [lo, hi] with 0 <= lo < hi"
    pure (name, units, circuit, registered)
  if ms.isEmpty then throw "proofs: members names no member"
  let rs := (ms.foldl (fun acc (_, us, _) => acc ++ us) #[]).qsort (·.1 < ·.1)
  for i in [1:rs.size] do
    if rs[i]!.1 < rs[i - 1]!.2 then
      throw s!"proofs: unit {rs[i]!.1} is in two members' units: a unit is one member's"
  pure ms

def paramsOfJson (j : Json) : V Params := do
  let some pr := field? j "proofs"
    | throw "the parameters name no proofs (the auditor's circuit digest and the verifier's registered reads)"
  let mine ← if keys? pr == some #["members"] then Sum.inr <$> membersOf (getD pr "members") else do
    exactly pr #["circuit", "registered"] "proofs"
    (Sum.inl <$> circuitOf pr : V (Sum (String × Option Json) _))
  let rest := Json.mkObj (without j "proofs")
  let base ← OneStage.paramsOfJson rest
  if base.challenger.isNone then throw NO_CHALLENGER
  let some partition := base.verifier.partition
    | throw "partition is null: the sessions are verified against the verifier's own partition object"
  let programJ := getD rest "program"
  let program := (← Flock.canon programJ).toUTF8
  let own := fun ((circuit, registered) : String × Option Json) => ({ programJ, program, partition, circuit, registered } : Own)
  pure { base, form := match mine with
    | .inl cr => .one (own cr)
    | .inr ms => .members (ms.map fun (name, units, cr) => { name, units, own := own cr }) }

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

/-- The evidence's `proofs` in either form (module docstring, **evidence**): one statement, naming no member, or the
per-member form's statements, each `proofsOf` with its member's nonempty name. -/
def statementsOf (pr : Json) : V (Array StatementFiles) := do
  if keys? pr != some #["statements"] then
    let (pub, sessions) ← proofsOf pr
    return #[{ member := none, pub, sessions }]
  let .arr ss := getD pr "statements" | throw "proofs: statements is not a list"
  ss.mapM fun s => do
    exactly s #["member", "public", "sessions"] "a statement of the proofs"
    let .str member := getD s "member" | throw "proofs: a statement's member is not a member's name"
    if member.isEmpty then throw "proofs: a statement's member is not a member's name"
    let (pub, sessions) ← proofsOf (Json.mkObj (without s "member"))
    pure { member := some member, pub, sessions }

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  if !(j matches .obj _) then throw "the evidence is not an object"
  if (field? j "verdicts").isSome then
    throw "the evidence names verdicts: this audit's verdicts are its own run of the verifier on the proofs"
  let some pr := field? j "proofs" | throw "the evidence names no proofs (the public file and the sessions' files)"
  let base ← OneStage.evidenceOf (← Flock.canon (Json.mkObj (without j "proofs" ++ [("verdicts", Json.mkObj [])]))).toUTF8
  let e : Evidence := { base, statements := ← statementsOf pr }
  if e.sessions.map (·.name) != base.sessions then throw "proofs: the sessions are not the evidence's sessions, in order"
  pure e

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

/-- One `flock-verify verify --zk` over a statement's sessions on the blobs, as `one_stage.proofs.verify` runs it, after
the sessions `r` holds, each session set up at its own record's draw: every session's verdict, the units the accepted
ones proved and whether all accepted. A refusal for what `proofs.verify` refuses: registered values read without the
verifier's registered file, a file no blob holds, or an accepted session naming another program than the partition
object's. -/
def runFrom (p : Own) (pub : String) (sessions : Array RunFiles) (bs : Core.Blobs) (r : Run) : V Run := do
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
  sessions.foldlM (step I rd bs scope ksFiles (programDigest p.program) (getD p.partition "program")) r

/-- `r` once every session accepted, held to the evidence's proved units `want`: a refusal when the units they proved
are others. -/
def provedHeld (want : Option (Array Nat)) (r : Run) : V Run :=
  match want with
  | some want =>
    if r.all && r.proved.qsort (· < ·) != want then throw "the units the sessions proved are not the evidence's proved_units"
    else pure r
  | none => pure r

/-- One statement's run (`runFrom`, after no session), its proved units held to `provedUnits?` (`provedHeld`). -/
def run (p : Own) (pub : String) (sessions : Array RunFiles) (provedUnits? : Option (Array Nat)) (bs : Core.Blobs) :
    V Run := do
  provedHeld provedUnits? (← runFrom p pub sessions bs ⟨#[], #[], #[], true⟩)

/-- Whether unit `u` is in one of the ranges `rs`. -/
def within (rs : Array (Nat × Nat)) (u : Nat) : Bool := rs.any fun (lo, hi) => lo ≤ u && u < hi

/-- A member's first unit, its lowest. Members' units are disjoint, so it orders them, and in the registration's order:
member `k`'s units start at its base, which grows with `k`. -/
def Member.first (m : Member) : Nat := m.units.foldl (fun a (lo, _) => min a lo) ((m.units[0]?.map (·.1)).getD 0)

/-- The per-member form's statements as public structure (module docstring, **per member**), before any session runs:
each names a member of the parameters that no earlier statement named, in the canonical order (each member's first unit,
`Member.first`, after the previous one's: one order, so one statement), with at most one session (one pass on live
coins, never retried). Their members, statement by statement. -/
def membersHeld (ms : Array Member) (sts : Array StatementFiles) : V (Array Member) := do
  let mut held : Array Member := #[]
  for st in sts do
    let some name := st.member
      | throw "the evidence's proofs are one statement's (public, sessions) and the parameters' per member (members)"
    let some m := ms.find? (·.name == name) | throw s!"member {name}: the parameters name no such member"
    if held.any (·.name == name) then throw s!"member {name}: a second statement: a member's units are one statement's"
    if let some prev := held.back? then
      if m.first < prev.first then
        throw (s!"member {name}: its statement follows member {prev.name}'s, whose units start after its own: the " ++
          "statements are in the members' order (by first unit, the registration's), one order for one statement")
    if st.sessions.size > 1 then
      throw (s!"member {name}: its statement has {st.sessions.size} sessions: a statement is one session on live " ++
        "coins, never retried")
    held := held.push m
  pure held

/-- The per-member form's run (module docstring, **per member**): the statements' structure held (`membersHeld`), then
each statement on its member's own inputs (`runFrom`) after the statements before it, so `step`'s coin-tree keys are
fresh across statements; a refusal when a statement proves a unit outside its member's units, and for what `runFrom`
refuses; then the union of the statements' proved units held to the evidence's (`provedHeld`). -/
def runMembers (ms : Array Member) (sts : Array StatementFiles) (want : Array Nat) (bs : Core.Blobs) : V Run := do
  let held ← membersHeld ms sts
  let mut r : Run := ⟨#[], #[], #[], true⟩
  for (st, m) in sts.zip held do
    let name := m.name
    let r' ← (runFrom m.own st.pub st.sessions bs r).mapError (s!"member {name}: {·}")
    if let some u := (r'.proved.extract r.proved.size r'.proved.size).find? (!within m.units ·) then
      throw s!"member {name}: its statement proved unit {u}, outside the member's units"
    r := r'
  provedHeld (some want) r

/-- The evidence's statements run in the parameters' form: the one statement's (`run`), or each member's
(`runMembers`); a refusal when the evidence's proofs are in the other form. -/
def runAll (p : Params) (e : Evidence) (bs : Core.Blobs) : V Run :=
  match p.form, e.statements.toList with
  | .one own, [{ member := none, pub, sessions }] => run own pub sessions (some e.base.provedUnits) bs
  | .one _, _ => throw "the evidence's proofs are per member (statements) and the parameters' one statement's (circuit, registered)"
  | .members ms, _ => runMembers ms e.statements e.base.provedUnits bs

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

/-- The coin trees the window's challenger session shows it sampled and opened (module docstring, **trees**): read from
the evidence's `coins` under the parameters' challenger's key, on the verifier's window. -/
def treesOf (p : Params) (e : Evidence) : V (Array Challenger.LoggedTree) :=
  match p.base.challenger, p.base.verifier.windowRoot with
  | some (key, _), some w => (Challenger.trustedTreesOf key w e.base.coins).mapError (s!"coins: {·}")
  | _, _ => throw s!"coins: {NO_CHALLENGER}"

/-- The evidence's sessions held to the challenger's trees `ts` (module docstring, **trees**): each session's record
on the blobs, refused as `verdicts` refuses it when no blob holds it or it isn't UTF-8, then
`Challenger.sessionsTreeReason`. -/
def treesHeld (ts : Array Challenger.LoggedTree) (e : Evidence) (bs : Core.Blobs) : V Unit := do
  let mut named : Array (String × Json) := #[]
  for s in e.sessions do
    let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
      | throw s!"session {s.name}'s record is not UTF-8"
    named := named.push (s.name, (Json.parse record).toOption.getD Json.null)
  if let some why := Challenger.sessionsTreeReason ts named then throw why

/-- The check at the coins `ξ` (module docstring, **check**): each session's coins held to the trees the challenger
logged (`treesOf`, `treesHeld`), then the sessions' verdicts on the blobs (`runAll`), then v1's `checkAt` at `ξ` on
them. v1 accepts only verdicts that all accepted (`VerdictsAccept`); the run's own `all` says it again, so an acceptance
reads the run rather than the verdicts' JSON. In the per-member form what would accept is refused (`MULTI_REFUSED`). -/
def checkBlobsAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match treesOf p e >>= (treesHeld · e bs) >>= fun () => runAll p e bs with
  | .error err => .reject s!"proofs: {err}"
  | .ok r =>
    match OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
    | .accept pub prof =>
      if r.all then (if p.form.perMember then .reject s!"proofs: {MULTI_REFUSED}" else .accept pub prof)
      else .reject "proofs: a session's verifier did not accept"
    | o => o

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
#guard within #[(0, 4), (8, 9)] 8 && !within #[(0, 4), (8, 9)] 4 && !within #[] 0

private def memberJ (units : List (Nat × Nat)) : Json :=
  Json.mkObj [("circuit", Json.str ("".pushn 'c' 128)), ("registered", Json.null),
    ("units", Json.arr (units.toArray.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi]))]

#guard ((membersOf (Json.mkObj [("a", memberJ [(0, 4)]), ("b", memberJ [(4, 6), (8, 9)])])).toOption.map
  (·.map fun (n, us, _) => (n, us))) == some #[("a", #[(0, 4)]), ("b", #[(4, 6), (8, 9)])]
#guard (membersOf (Json.mkObj [("a", memberJ [(0, 4)]), ("b", memberJ [(3, 6)])])).toOption.isNone
#guard (membersOf (Json.mkObj [("a", memberJ [(0, 4), (2, 3)])])).toOption.isNone
#guard (membersOf (Json.mkObj [("a", memberJ [])])).toOption.isNone
#guard (membersOf (Json.mkObj [])).toOption.isNone
#guard (statementsOf (Json.mkObj [("statements", Json.arr #[Json.mkObj [("member", "a"), ("public", "p"),
  ("sessions", Json.arr #[])]])])).toOption.map (·.map (·.member)) == some #[some "a"]
#guard (statementsOf (Json.mkObj [("public", "p"), ("sessions", Json.arr #[])])).toOption.map (·.map (·.member)) ==
  some #[none]

private def memberT (name : String) (units : Array (Nat × Nat)) : Member :=
  { name, units, own := { programJ := Json.null, program := .empty, partition := Json.null, circuit := "", registered := none } }

private def statementT (member : String) (sessions : Nat := 1) : StatementFiles :=
  { member := some member, pub := "p", sessions := (List.replicate sessions { name := "s", record := "r", reps := #[] }).toArray }

private def membersT : Array Member := #[memberT "m2" #[(2, 3)], memberT "m10" #[(10, 11)], memberT "m0" #[(5, 6), (0, 1)]]

private def whyT (r : V (Array Member)) : String := match r with
  | .error e => e
  | .ok _ => ""

#guard (memberT "m0" #[(5, 6), (0, 1)]).first == 0
#guard ((membersHeld membersT #[statementT "m0", statementT "m2", statementT "m10"]).toOption.map (·.map (·.name))) ==
  some #["m0", "m2", "m10"]
#guard ((membersHeld membersT #[statementT "m2" 0, statementT "m10"]).toOption.map (·.map (·.name))) == some #["m2", "m10"]
#guard ((membersHeld membersT #[]).toOption.map (·.size)) == some 0
#guard (whyT (membersHeld membersT #[statementT "m2", statementT "m0"])).startsWith
  "member m0: its statement follows member m2's, whose units start after its own"
#guard (whyT (membersHeld membersT #[statementT "m0", statementT "m10", statementT "m2"])).startsWith
  "member m2: its statement follows member m10's"
#guard whyT (membersHeld membersT #[statementT "m0", statementT "m0"]) ==
  "member m0: a second statement: a member's units are one statement's"
#guard (whyT (membersHeld membersT #[statementT "m0" 2])).startsWith "member m0: its statement has 2 sessions"
#guard whyT (membersHeld membersT #[statementT "m3"]) == "member m3: the parameters name no such member"

end SampledProofs.OneStageZk
