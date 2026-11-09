/-
Sampled proofs' one-stage audit on the proofs themselves (`verity/sampled-proofs/one-stage/v2`), a `Core.BlobProtocol`: the
v1 audit (`OneStage`) with the verdicts computed here, from the statement's files, rather than read from the evidence.

* **params**: v1's, plus `proofs`, exactly `{"circuit", "public", "registered"}`: the SHA-512s the auditor committed for
  the circuit and public files (lowercase hex of 64 bytes) and the verifier's own registered file (`flock-verify
  --registered`'s object, or null for a program that reads no registered value). The partition must not be null: the
  sessions are verified against the verifier's own partition object; and v1's `challenger`, with its `window_root`, must
  be named (`NO_CHALLENGER`), its key at the deployed LMS parameters, `Lms.deployed` (`KEY_PARAMS`);
* **commitment**: v1's (the registration record);
* **evidence**: v1's without `verdicts`, its `coins` within the challenger evidence's limits (`Challenger.withinAt`
  at `Challenger.LIMITS`), plus `proofs`, exactly `{"sessions"}`: for each of v1's `sessions` in order,
  `{"name", "record", "reps"}`: its name, its `session.json`'s SHA-512 and `{file: SHA-512}` naming at least each rep
  file the statement's session reads (`<stream with / as .>.bin`). The sampled proof blocks are proved in one session, whose
  statement is theirs: the check refuses evidence naming none or several (`oneSession`) before it reads a tree;
* **blobs**: the files those digests name (`Core.Blobs`), and the MUFU tables any lookup reads, found under their pinned
  SHA-512 as `flock-verify --archive` finds them;
* **coins** (`coinsOf`): v1's (`OneStage.coinsOf`), what `Challenger.drawCoins` reads from the evidence's `coins` under
  the challenger's key, on the verifier's window; none read refuses. The check below runs at them (`checkBlobsAt`): v1's
  `checkAt` refuses a sample other than the one they give before v1's `check`;
* **trees** (`treesOf`, `orderHeld`, `treesHeld`): the coin trees that same session's signed release log shows the
  challenger sampled and opened (`Challenger.trustedTreesOf`). The log must release the sample, then the link, then the
  other rounds: no coin of the parameters' challenger family after the registration's digest released after a round is
  opened, and the first round opened a tree's `session` stream (`Challenger.sampleLinkFirst`, `SAMPLE_LINK_FIRST`). Each
  session's coins must be one tree's, a tree no other session's, its record's `coin_tree`, link points and every
  round's coins as the challenger logged them, its link's `root_b` naming exactly the session's tables
  (`sessionTables`), the roots the link points are opened after, each entry the length its table's statement fixes,
  `2^cap₀` digests (`rootBTables`, `Challenger.LINK_ROOT_LEN`), refused before the link's message is hashed, which is
  what makes that message parse one way, and every tree logged must be a session's (`Challenger.sessionsTreeReason`);
  else the check refuses, before any session is verified. A session whose statement sets up no table has no length to
  hold, and its own verdict rejects;
* **population** (`popHeld`): the registration's population held to the verifier's program: under its query, the
  program's proof blocks read stratum by stratum must be the record's `population` units in order (`Slots.popOk`), as
  the service's run refuses at registration (`Run.run`, `population-order`), else the check refuses, before any session
  is verified. v1's R2 holds only their count, and the session's knowledge soundness reads slot `u` as proof block `u`;
* **public** (`publicHeld`): the public file's header held to the registration (`Record.matchesPublic`): its scheme,
  roots and their domains and its instance count the record's, and its `units.indices` the population's units in
  order (`indicesHeld`, `indicesHeld_of_publicHeld`), so no header written after the sample relabels an instance as
  another unit; else the check refuses, before any session is verified;
* **check**: `flock-verify verify --zk --statement verity/flock-circuit` with `--circuit`, `--public`, `--partition`,
  `--program` and `--registered` the verifier's own and one `--session` a session, as `one_stage.proofs.verify` runs it
  (`verdicts`, the Flock library calls `verifyCmd` makes on that path, its last the files' ports, domains and roots,
  `Flock.PortRoots.checkOf`, and the form `Flock.Guarantees.SampledKSLiveV3` covers, or at a shared-row public file
  `SampledKSLiveV5`'s, `Flock.KsScope.checkOf`), and a session's verdict holds only where the registration's population
  is the circuit file's units in order (`FilePart.check`: its strata over the verifier's program and query, read stratum
  by stratum, are its units in order, and the public file holds exactly those, so slot `u` is the file's unit `u`, as
  `Flock.Bridge.partOf` reads it); then v1's `check` on those verdicts, one of
  record a session (`flock-verify/<name>`). A session accepted under another program than the partition object's, proved
  units other than the evidence's `proved_units`, or a circuit, public or record file no blob holds, rejects; a rep
  file no blob holds is that session's reject, as a file `flock-verify` can't read is. So does a length a session's
  statement can't hold, since it writes each in an 8-byte word: the circuit or public file, the lookup tables, the
  program, the partition object, the registered reads or a session's draw at `2^64` bytes or more (`fitsWhy`,
  `drawFitsWhy`, the JSON ones as `Flock.RecordText.printT` prints them).

Without blobs (`toProtocol`) every check rejects: no blob holds the circuit file.
-/
module

public import CoreProtocol.Blobs
public import Spot.OneStage
public import Spot.ChallengerTrees
public import Spot.Slots
public import Spot.FilePart
public import Flock
public import Flock.JsonParse
meta import CoreProtocol.Blobs
meta import Spot.OneStage
meta import Spot.ChallengerTrees
meta import Spot.Slots
meta import Spot.FilePart
meta import Flock
meta import Flock.JsonParse

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

def PROTOCOL : String := "verity/sampled-proofs/one-stage/v2"
/-- The statement every sampled proof block's session proves (`one_stage.proofs.STATEMENT`). -/
def STATEMENT : String := "verity/flock-circuit"
/-- The refusal of parameters naming no challenger: the sample's coins are never the prover's to choose. -/
def NO_CHALLENGER : String :=
  "the parameters name no challenger: the draw's coins are the auditor's challenger's signed release (challenger, " ++
    "with window_root), never the prover's record"
/-- The refusal of a challenger key at other LMS parameters than the deployed ones (`Lms.deployed`), the only ones the
game's signature verifier (`Verity.lmsVerify`) reads. -/
def KEY_PARAMS : String :=
  s!"the challenger's key is not at the deployed LMS parameters (W = {Lms.deployed.W}, H = {Lms.deployed.H}): v2 " ++
    "reads the challenger's signatures only under those"

/-- The key is at `Lms.deployed`, field by field (`Lms.Params` derives only `BEq`). -/
def deployedKey (key : Challenger.ChallengerKey) : Bool :=
  key.params.N == Lms.deployed.N && key.params.W == Lms.deployed.W && key.params.H == Lms.deployed.H

/-- The refusal of a template instance of no unit: each proof block would hold no gate. Every unit holds a computed row,
its own constant, so `units_per_vu ≥ 1` is all a proof block's gate needs. -/
def NO_ROW_WHY : String := "the circuit's units_per_vu is 0: a template instance of no unit, a proof block of no gate"

/-- The widest public size class of a proof block's outputs, in bits (`Claims.Sampling.unitClassMax`). -/
def UNIT_CLASS_MAX : Nat := 262144

/-- The refusal of a template instance that returns more bits than the widest size class holds. -/
def WIDE_WHY : String :=
  "a template instance returns more than 262144 bits (16 per output leaf of its units_per_vu units): past the " ++
    "widest size class"

/-- **The bits a template instance returns**: 16 per output leaf of each of its `units_per_vu` units. -/
def returnedBits (c : Flock.Circuit) : Nat := 16 * ((List.range c.upv).map fun u => (c.leavesOut.getD u #[]).size).sum

/-- **The unit template's refusals** on the parsed circuit: a template instance of no unit (`NO_ROW_WHY`), and one
returning more than `UNIT_CLASS_MAX` bits (`WIDE_WHY`). -/
def templateOk (c : Flock.Circuit) : V Unit := do
  if c.upv = 0 then throw NO_ROW_WHY
  if returnedBits c > UNIT_CLASS_MAX then throw WIDE_WHY

/-- The verifier's own inputs to `flock-verify verify --zk`: the program's descriptor (as JSON and as the bytes
`--program` reads), the partition object, the circuit and public files' committed SHA-512s and the registered file. -/
structure Own where
  programJ : Json
  program : ByteArray
  partition : Json
  circuit : String
  pub : String
  registered : Option Json

structure Params where
  base : OneStage.Params
  own : Own

structure RunFiles where
  name : String
  record : String
  reps : Array (String × String)

structure Evidence where
  base : OneStage.Evidence
  sessions : Array RunFiles

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

private def digestOf (pr : Json) (k : String) : V String := do
  let .str d := getD pr k | throw s!"proofs: {k} is not a SHA-512 digest (128 lowercase hex)"
  if !digest? d then throw s!"proofs: {k} is not a SHA-512 digest (128 lowercase hex)"
  pure d

/-- The verifier's own files in `pr`: the circuit and public files' digests and the registered file. -/
private def committedOf (pr : Json) : V (String × String × Option Json) := do
  let circuit ← digestOf pr "circuit"
  let pub ← digestOf pr "public"
  let registered ← match getD pr "registered" with
    | .null => pure none
    | r@(.obj _) => pure (some r)
    | _ => throw "proofs: registered is not the verifier's registered file (an object) or null"
  pure (circuit, pub, registered)

def paramsOfJson (j : Json) : V Params := do
  let some pr := field? j "proofs"
    | throw ("the parameters name no proofs (the auditor's circuit and public files' digests and the verifier's " ++
      "registered reads)")
  exactly pr #["circuit", "public", "registered"] "proofs"
  let (circuit, pub, registered) ← committedOf pr
  let rest := Json.mkObj (without j "proofs")
  let base ← OneStage.paramsOfJson rest
  let some (key, _) := base.challenger | throw NO_CHALLENGER
  if !deployedKey key then throw KEY_PARAMS
  let some partition := base.verifier.partition
    | throw "partition is null: the sessions are verified against the verifier's own partition object"
  let programJ := getD rest "program"
  pure { base, own := { programJ, program := (← Flock.canon programJ).toUTF8, partition, circuit, pub, registered } }

/-- The `verdicts` command's own inputs: exactly `{"circuit", "partition", "program", "public", "registered"}`, the
program and partition objects. -/
def ownOfJson (j : Json) : V Own := do
  exactly j #["circuit", "partition", "program", "public", "registered"] "the verifier's own inputs"
  let (circuit, pub, registered) ← committedOf j
  let (programJ@(.obj _), partition@(.obj _)) := (getD j "program", getD j "partition")
    | throw "the program and the partition are not objects"
  pure { programJ, program := (← Flock.canon programJ).toUTF8, partition, circuit, pub, registered }

private def sessionOf (j : Json) : V RunFiles := do
  exactly j #["name", "record", "reps"] "a session of the proofs"
  let (.str name, .str record, .obj _) := (getD j "name", getD j "record", getD j "reps")
    | throw "a session of the proofs is not a name, its record's digest and its rep files' digests"
  let reps ← (entries (getD j "reps")).mapM fun (f, d) => match d with
    | .str d => pure (f, d)
    | _ => throw s!"session {name}: rep file {f}'s digest is not a string"
  pure { name, record, reps }

/-- The evidence's `proofs`: the sessions. The public file is the parameters', committed before any coin. -/
def proofsOf (pr : Json) : V (Array RunFiles) := do
  exactly pr #["sessions"] "the evidence's proofs"
  let .arr ss := getD pr "sessions" | throw "proofs: sessions is not a list"
  ss.mapM sessionOf

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  if !(j matches .obj _) then throw "the evidence is not an object"
  if (field? j "verdicts").isSome then
    throw "the evidence names verdicts: this audit's verdicts are its own run of the verifier on the proofs"
  let some pr := field? j "proofs" | throw "the evidence names no proofs (the sessions' files)"
  let base ← OneStage.evidenceOf (← Flock.canon (Json.mkObj (without j "proofs" ++ [("verdicts", Json.mkObj [])]))).toUTF8
  unless Challenger.withinAt Challenger.LIMITS base.coins do throw Challenger.LIMITS.form
  let sessions ← proofsOf pr
  if sessions.map (·.name) != base.sessions then throw "proofs: the sessions are not the evidence's sessions, in order"
  pure { base, sessions }

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
  let j ← (Flock.JsonParse.parse (← String.fromUTF8? b)).toOption
  let bare := match j with
    | .obj kvs => Json.mkObj (kvs.toArray.filter (·.1 != "annotations")).toList
    | x => x
  return Flock.hexOf (Flock.Sha512.hash (← (Flock.canon bare).toOption).toUTF8)

/-- The refusal of a public file whose instances are not the registration's population units in order (`indicesHeld`). -/
def INDICES_WHY : String :=
  "public file: its units.indices are not the registration's population units in order (instance i is unit i)"

/-- The header `h`'s `units.indices` are the registration `rec`'s population units in order, `[0, population)`: the unit
v1 credits for an instance a session proves (`provedUnits`) is that instance's slot, which the session's knowledge
soundness reads (`popHeld`), fixed by the registration before any coin rather than by a header written after it. -/
def indicesHeld (rec h : Json) : Bool :=
  match nat? (getD rec "population"), getD (getD h "units") "indices" with
  | some n, .arr is => is.size == n && (List.range n).all fun i => (is[i]? >>= nat?) == some i
  | _, _ => false

/-- The public file held to the registration `rec` (module docstring, **public**; `registration.matches_public`, as the
one-stage drivers hold theirs before the sample): its header's scheme, `frame_v3` roots and their `domain_ids`, and
`instances` against the record's scheme, roots, domains and population, a refusal naming the codes that fail; then its
`units.indices` the population's units in order (`indicesHeld`). -/
def publicHeld (rec : Json) (pubB : ByteArray) : V Unit := do
  let h ← FilePart.header pubB
  let fv := getD h "frame_v3"
  let c := Record.matchesPublic rec (getD h "scheme") (getD fv "roots") (getD fv "domain_ids") (getD h "instances")
  if !c.codes.isEmpty then throw s!"public file: not the registration's: {", ".intercalate c.codes.toList}"
  if !indicesHeld rec h then throw INDICES_WHY

/-- `publicHeld` passes only on a public file whose header's `units.indices` are the registration's population units in
order (`indicesHeld`): the unit `provedUnits` credits for instance `i` is `i`, fixed by the registration before any coin. -/
theorem indicesHeld_of_publicHeld {rec : Json} {pubB : ByteArray} (h : publicHeld rec pubB = .ok ()) :
    ∃ hd, FilePart.header pubB = .ok hd ∧ indicesHeld rec hd = true := by
  unfold publicHeld at h
  cases hh : FilePart.header pubB with
  | error e => simp [hh, bind, Except.bind] at h
  | ok hd =>
    refine ⟨hd, rfl, ?_⟩
    cases hi : indicesHeld rec hd
    · simp only [hh, hi, bind, Except.bind, Bool.not_false, ↓reduceIte, throw, throwThe, MonadExceptOf.throw] at h
      split at h <;> simp_all
    · rfl

/-- `one_stage.proofs.proved_units`: the public file's unit indices (`units.indices`, as the population check reads
them: `FilePart.indicesOf` of `FilePart.header`) of the instances the session's record drew (`unit_draw.units`),
ascending. -/
def provedUnits (pubB : ByteArray) (record : String) : V (Array Nat) := do
  let is ← FilePart.indicesOf (← FilePart.header pubB)
  let .arr ds := getD (getD (← Flock.JsonParse.parse record) "unit_draw") "units"
    | throw "the record's unit_draw names no units"
  let units ← ds.mapM fun d => match nat? d >>= (is[·]?) with
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

/-- The bound on each length a session's statement holds: `flock-verify`'s statement writes every length in an 8-byte
word. -/
def WORD : Nat := 2 ^ 64

/-- A JSON value's length as a session's statement holds it: its text as `Flock.RecordText.printT` prints it, in
UTF-8. -/
def printedSize (j : Json) : Nat := (Flock.RecordText.printT j).utf8ByteSize

/-- What a session's statement would hold at `limit` bytes or more, the first of: the circuit and public files, the
number of lookup tables, a lookup table's name (in UTF-8) or bytes, the program, the partition object and the
registered reads (`printedSize`); `none` when every length is under `limit`. -/
def fitsWhy (limit : Nat) (circuit pub : ByteArray) (tables : List (String × ByteArray)) (program : Option ByteArray)
    (partition registered : Option Json) : Option String :=
  if limit ≤ circuit.size then some "the circuit file"
  else if limit ≤ pub.size then some "the public file"
  else if limit ≤ tables.length then some "the number of lookup tables"
  else if tables.any (fun t => limit ≤ t.1.utf8ByteSize || limit ≤ t.2.size) then some "a lookup table"
  else if program.any (limit ≤ ·.size) then some "the program"
  else if partition.any (limit ≤ printedSize ·) then some "the partition object"
  else if registered.any (limit ≤ printedSize ·) then some "the registered reads"
  else none

/-- `some "the draw"` when a session's statement would hold its record's draw `draw` at `limit` bytes or more
(`printedSize`). -/
def drawFitsWhy (limit : Nat) (draw : Option Json) : Option String :=
  if draw.any (limit ≤ printedSize ·) then some "the draw" else none

/-- The refusal of a length `fitsWhy` or `drawFitsWhy` names at `WORD`. -/
def FITS : String := "2^64 bytes or more, past the 8-byte word a session's statement holds each length in"

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
registered openings, once the sample's law holds. -/
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

/-- The record's coin-tree key, when the session's setup flips its coins from a coin tree. -/
def coinKeyOf (record : String) (st : SetUp) : Option String :=
  match st with
  | .ok (spec, _, _) =>
    if spec.coinTree.isNone then none
    else (Flock.JsonParse.parse record).toOption.bind fun j =>
      (j.getObjVal? "coin_tree").toOption.bind fun ct => (ct.getObjValAs? String "key").toOption
  | .error _ => none

/-- `verify --zk`'s verdict on one session set up as `st` at `draw`: its proof files, its coin-tree key fresh among
`coinKeys`, `Zk.verify` and the checks `verifyCmd` makes after it, its last the files' ports and roots
(`ksFiles`, `Flock.PortRoots.checkOf`) and the form `Flock.Guarantees.SampledKSLiveV4` covers at `draw`
(`Flock.KsScope.checkOf`); then the registration's population the public file's units in order (`population`,
`FilePart.check`). -/
def verdictOf (I : Flock.Witness.Inputs) (scope ksFiles population : V Unit) (coinKeys : Array String)
    (record : String) (draw : Option Json) (st : SetUp) (files : V (Array ByteArray)) (coinKey : Option String) :
    V Unit := do
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
  population

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

/-- One session's `flock-verify verify --zk` on the blobs, after the sessions `r` holds: set up at its record's own sample
(`setupAt`), its rep files the blobs its `reps` name, its coin-tree key fresh, `Zk.verify` and the checks `verifyCmd`
makes after it, the population check last (`verdictOf`), its verdict pushed (`push`). A refusal when its record is in
no blob or not UTF-8, its draw doesn't fit the session's statement (`drawFitsWhy`), or it accepted naming another
program than the partition object's. -/
def step (I : Flock.Witness.Inputs) (rd : Reads) (bs : Core.Blobs) (scope ksFiles population : V Unit)
    (programSha : Option String) (want : Json) (r : Run) (s : RunFiles) : V Run := do
  let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
    | throw s!"session {s.name}'s record is not UTF-8"
  let draw := Flock.Witness.recordDraw record
  (match drawFitsWhy WORD draw with
    | some what => throw s!"session {s.name}: {what}: {FITS}"
    | none => pure () : V Unit)
  let st := setupAt I rd draw
  let coinKey := coinKeyOf record st
  push I rd programSha want r s record st coinKey
    (verdictOf I scope ksFiles population r.coinKeys record draw st (filesOf bs s st) coinKey)

/-- The units the accepted sessions proved (`proved`, in session order) held to the evidence's `proved_units` (`want`):
sorted, they must be it. `want` is strictly ascending (`Draw.checkDrawFull`), so a unit two sessions prove (or one
session twice) refuses, and so does a sampled proof block no session proves. -/
def provedHeld (proved want : Array Nat) : V Unit :=
  if proved.qsort (· < ·) != want then throw "the units the sessions proved are not the evidence's proved_units"
  else pure ()

/-- One `flock-verify verify --zk` over the evidence's sessions on the blobs, as `one_stage.proofs.verify` runs it, each
session set up at its own record's sample: every session's verdict, the units the accepted ones proved and whether all
accepted. A session accepts only where the registration's population is the public file's units in order
(`FilePart.check`, last). A refusal for what `proofs.verify` refuses: registered values read without the verifier's registered file, a
file no blob holds, an accepted session naming another program than the partition object's, or proved units other than
the evidence's (`provedHeld`); and for a length a session's statement can't hold (`fitsWhy`, `drawFitsWhy`, at
`WORD`). -/
def run (p : Own) (sessions : Array RunFiles) (provedUnits? : Option (Array Nat)) (bs : Core.Blobs) : V Run := do
  let some tags0 := Flock.Tags.find? STATEMENT | throw s!"the statement {STATEMENT} is not known"
  (if tags0.plainLeaves then throw s!"{tags0.name}: a statement with plain SHA-512 leaves: --zk's hiding needs hm96 leaves"
    else pure () : V Unit)
  (if p.registered.isNone && readsRegistered p.programJ then
    throw ("registered: the program reads registered values (a RegisteredInput among its definitions) and the " ++
      "verifier holds no registered reads: verify with its own registered file")
    else pure () : V Unit)
  let circuitB ← bs.get p.circuit "circuit"
  let pubB ← bs.get p.pub "public file"
  let I := inputsOf tags0 p circuitB pubB bs
  (match fitsWhy WORD I.circuitFile I.publicFile I.tables I.program I.partition p.registered with
    | some what => throw s!"{what}: {FITS}"
    | none => pure () : V Unit)
  let rd ← readsOf p.registered
  let tags := Flock.Witness.zkTags I
  let scope : V Unit := do
    Flock.ProvedScope.checkOf (Flock.ProvedScope.circuitOf tags I.circuitFile I.tables) tags true pubB I.count false
      rd.registered.isSome
    templateOk (← Flock.ProvedScope.circuitOf tags I.circuitFile I.tables)
  let ksFiles := Flock.PortRoots.checkOf (Flock.ProvedScope.circuitOf tags I.circuitFile I.tables) tags pubB
  let population := FilePart.check p.program (getD p.partition "query") pubB
  let r ← sessions.foldlM
    (step I rd bs scope ksFiles population (programDigest p.program) (getD p.partition "program"))
    ⟨#[], #[], #[], true⟩
  if let some want := provedUnits? then
    if r.all then provedHeld r.proved want
  pure r

/-- The sessions' verdicts in order (`run`'s). -/
def verdicts (p : Own) (sessions : Array RunFiles) (provedUnits? : Option (Array Nat)) (bs : Core.Blobs) :
    V (Array (String × Json)) :=
  (·.out) <$> run p sessions provedUnits? bs

/-- The coins the sample read (module docstring, **coins**): v1's, which only the parameters' challenger gives; a refusal
when none read. -/
def coinsOf (p : Params) (c : OneStage.Commitment) (e : Evidence) : V ByteArray := do
  match ← OneStage.coinsOf p.base c e.base with
  | some ξ => pure ξ
  | none => throw s!"coins: {NO_CHALLENGER}"

/-- The coin trees the window's challenger session shows it sampled and opened (module docstring, **trees**): read from
the evidence's `coins` under the parameters' challenger's key, on the verifier's window: v1's parameters and evidence,
which the recursive audit's trees are read from too (`VStar.Recursive.innerCoinsHeld`). -/
def treesOf (p : OneStage.Params) (e : OneStage.Evidence) : V (Array Challenger.LoggedTree) :=
  match p.challenger, p.verifier.windowRoot with
  | some (key, _), some w => (Challenger.trustedTreesOf key w e.coins).mapError (s!"coins: {·}")
  | _, _ => throw s!"coins: {NO_CHALLENGER}"

/-- The tables of the session `run` sets up: the statement's tags' at `inputsOf`'s count, one table. None when the
statement is unknown, which `run` refuses. -/
def sessionTables : Array String := ((Flock.Tags.find? STATEMENT).map (·.tableNames 1)).getD #[]

/-- The byte length of table `st`'s `root_B`: the level-0 cap a `--zk` proof's reader reads (`Flock.ZkProof.reader`),
`2^cap₀` digests of the table's Merkle scheme, `cap₀` the first stratum of `Flock.Zk.schedule` at the table's `m`.
`Zk.verify` holds `root_B` to that cap (S16/R1), so the statement fixes it. -/
def rootBLen (st : Flock.Setup) : V Nat := do
  let (sch, _) ← Flock.Zk.schedule st.m
  pure (2 ^ (sch.levels.getD 0 default).cap * st.merkle.digestLen)

/-- The session's tables (`sessionTables`), each with its `root_B`'s byte length (`rootBLen`) at the session `record`
sets up: `setupAt` at the record's sample on the verifier's own files `p`, as `run` sets it up. None where it sets up no
table, and then `run` sets up none either, so the session's own verdict rejects. -/
def rootBTables (p : Own) (bs : Core.Blobs) (record : String) : Array (String × Option Nat) :=
  let tabs : Option (Array Flock.Setup) := (do
    let some tags0 := Flock.Tags.find? STATEMENT | throw s!"the statement {STATEMENT} is not known"
    let I := inputsOf tags0 p (← bs.get p.circuit "circuit") (← bs.get p.pub "public file") bs
    let (_, tabs, _) ← setupAt I (← readsOf p.registered) (Flock.Witness.recordDraw record)
    pure tabs : V (Array Flock.Setup)).toOption
  sessionTables.mapIdx fun i t => (t, tabs.bind fun ts => ts[i]?.bind fun st => (rootBLen st).toOption)

/-- The evidence's sessions held to the challenger's trees `ts` (module docstring, **trees**): each session's record
on the blobs, refused as `verdicts` refuses it when no blob holds it or it isn't UTF-8, read as `Flock.Record.decode`
reads it (`Flock.JsonParse.parse`), then `Challenger.sessionsTreeReason`, each session at its tables with the `root_B`
lengths its statement fixes on the verifier's own files `p` (`rootBTables`). -/
def treesHeld (p : Own) (ts : Array Challenger.LoggedTree) (e : Evidence) (bs : Core.Blobs) : V Unit := do
  let mut named : Array (String × Array (String × Option Nat) × Json) := #[]
  for s in e.sessions do
    let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
      | throw s!"session {s.name}'s record is not UTF-8"
    named := named.push (s.name, rootBTables p bs record, (Flock.JsonParse.parse record).toOption.getD Json.null)
  if let some why := Challenger.sessionsTreeReason ts named then throw why

/-- The evidence's one session (module docstring, **evidence**): evidence naming none or several refuses. -/
def oneSession (sessions : Array RunFiles) : V Unit :=
  if sessions.size == 1 then pure ()
  else throw s!"the evidence names {sessions.size} sessions: the drawn units are proved in one session"

/-- The refusal of a registration whose population is not the program's proof blocks in order (`popHeld`). -/
def POPULATION_ORDER : String :=
  "population-order: the program's proof blocks under the query are not the registration's population units in order"

/-- The registration's population held to the verifier's program (module docstring, **population**): the record names
a population `n`, and under the verifier's query the program's proof blocks are its `n` units in order
(`Slots.popOk`). -/
def popHeld (v : Registration.Verifier) (record : Json) : V Unit :=
  match nat? (getD record "population") with
  | none => throw "the record names no population"
  | some n => if Slots.popOk v.program v.query n then pure () else throw POPULATION_ORDER

/-- The challenger session's release log in the evidence's `coins`, a line each, as `Challenger.trustedTreesOf` reads
it. -/
def linesOf (e : Evidence) : V (List String) := do
  let .arr ls := getD (getD e.base.coins "session") "log" | throw "the session's log is not a list of lines"
  ls.toList.mapM fun | .str l => pure l | _ => throw "a line of the session's log is not a string"

/-- The order the window's challenger session released in (module docstring, **trees**): the sample, then the link,
then the other rounds (`Challenger.sampleLinkFirst`) over its release log's lines after the head, at the parameters'
challenger family and the registration's digest. -/
def orderHeld (p : Params) (c : OneStage.Commitment) (e : Evidence) : V Unit := do
  let some (_, label) := p.base.challenger | throw s!"coins: {NO_CHALLENGER}"
  let lines ← linesOf e
  unless Challenger.sampleLinkFirst label c.digest lines.tail do throw s!"coins: {Challenger.SAMPLE_LINK_FIRST}"

/-- The check at the coins `ξ` (module docstring, **check**): the evidence's one session (`oneSession`), its coins held
to the trees the challenger logged in the order it must release them (`treesOf`, `orderHeld`, `treesHeld`), the
registration's population held to the program (`popHeld`), the parameters' public file held to the registration
(`publicHeld`), then the session's verdict on the blobs, then v1's `checkAt` at `ξ` on it. v1 accepts only verdicts
that all accepted (`VerdictsAccept`); the run's own `all` says it again, so an acceptance reads the run rather than the
verdicts' JSON. -/
def checkBlobsAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  let r : V Run := do
    oneSession e.sessions
    let ts ← treesOf p.base e.base
    orderHeld p c e
    treesHeld p.own ts e bs
    popHeld p.base.verifier c.record
    let pubB ← bs.get p.own.pub "public file"
    publicHeld c.record pubB
    run p.own e.sessions (some e.base.provedUnits) bs
  match r with
  | .error err => .reject s!"proofs: {err}"
  | .ok r =>
    match OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
    | .accept pub prof => if r.all then .accept pub prof else .reject "proofs: a session's verifier did not accept"
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

/-- The audit on the blobs `bs` reads its sample's coins (`coinsOf`) and decides at them (`checkBlobsAt`). -/
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
      let sessions ← proofsOf (← Core.canonicalJson proofsB)
      verdicts (← ownOfJson (← Core.canonicalJson ownB)) sessions none (Core.Blobs.ofArray bs)
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
#guard (provedHeld (#[9, 3] ++ #[7]) #[3, 7, 9]).toBool
#guard !(provedHeld (#[3, 7] ++ #[7, 9]) #[3, 7, 9]).toBool
#guard !(provedHeld (#[3] ++ #[9]) #[3, 7, 9]).toBool
#guard !((provedUnits "{\"units\":{\"indices\":[7,3,9]}}".toUTF8 "{\"unit_draw\":{\"units\":[2,2]}}") >>=
  (provedHeld · #[9])).toBool
#guard (oneSession #[⟨"s", "", #[]⟩]).toBool
#guard !(oneSession #[]).toBool
#guard !(oneSession #[⟨"s", "", #[]⟩, ⟨"t", "", #[]⟩]).toBool

private def heldRecord : Json :=
  (Flock.JsonParse.parse
    "{\"population\":2,\"roots\":[{\"domain\":\"dd\",\"port\":\"x\",\"root\":\"aa\"}],\"scheme\":\"s\"}").toOption.getD
    .null

private def refusal : V Unit → String
  | .error e => e
  | .ok () => ""

#guard refusal (publicHeld heldRecord
  ("{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"," ++
    "\"units\":{\"indices\":[0,1]}}\nrest").toUTF8) == ""
#guard [("[1,0]" : String), "[0,2]", "[0]", "[0,1,2]", "[\"0\",1]"].all fun is => refusal (publicHeld heldRecord
  ("{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"," ++
    "\"units\":{\"indices\":" ++ is ++ "}}").toUTF8) == INDICES_WHY
#guard refusal (publicHeld heldRecord
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"}".toUTF8) ==
  INDICES_WHY
#guard refusal (publicHeld heldRecord
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"ab\"}},\"instances\":2,\"scheme\":\"s\"}".toUTF8) ==
  "public file: not the registration's: roots"
#guard refusal (publicHeld heldRecord
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"de\"},\"roots\":{\"x\":\"aa\",\"y\":\"bb\"}},\"instances\":3,\"scheme\":\"s\"}".toUTF8) ==
  "public file: not the registration's: roots, domains, population"
#guard (refusal (publicHeld heldRecord "not a public file".toUTF8)).startsWith "the public file's header is not JSON: "

#guard (rootBLen { (default : Flock.Setup) with m := 25, merkle := Flock.MerkleScheme.hm96Sha512 }).toOption == some 16384
#guard (rootBLen { (default : Flock.Setup) with m := 26, merkle := Flock.MerkleScheme.hm96Sha512 }).toOption == some 8192

end Spot.OneStageZk
