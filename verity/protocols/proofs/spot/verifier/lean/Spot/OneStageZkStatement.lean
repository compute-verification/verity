module

public import CoreProtocol.Blobs
public import Spot.OneStage
public import Spot.ChallengerTrees
public import Spot.Slots
public import Spot.PopOrder
public import Spot.FilePart
public import Spot.CircuitKept
public import Flock
public import Flock.JsonParse
public import Std.Data.HashMap
import all Init.Util

/-!
# One-stage v2: the parameters, one statement's check, and a member's statement and files

The first half of the one-stage audit on the proofs themselves; `Spot/OneStageZk.lean`, which imports this module, has the
form's description and the second half: the run over the members, the evidence's parts, the checks and the commands.

Here, in the namespace `Spot.OneStageZk`:

* the parameters and the evidence in both forms (`paramsOfJson`, `membersParamsOfJson`, `membersOf`, `evidenceOf`,
  `membersEvidenceOf`);
* the registration held to the verifier's program (`popHeld`) and one statement's check on its files (`statementOf`);
* a member, its slice of the registration, its population and its statement (`Member`, `Member.statement`);
* the files' pass over the members (`membersFiles`), and the same pass with each circuit file parsed once and the program's
  strata evaluated once (`ProgramKept`, `keptFiles`, `membersFilesKept`), which is compiled in its place
  (`membersFiles_eq_kept`).
-/

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

/-- A member of the per-member form: its name, its units as `[lo, hi)` ranges, ascending, its circuit's units a proof
block (META `vus_per_block`) and its statement's own inputs. -/
structure Member where
  name : String
  units : Array (Nat × Nat)
  g : Nat
  own : Own

/-- The parameters: v1's, the verifier's own inputs, and in the per-member form (module docstring, **per member**)
its members, `own` then holding the program and the partition every member's statement reads and no file. -/
structure Params where
  base : OneStage.Params
  own : Own
  members : Option (Array Member)

structure RunFiles where
  name : String
  record : String
  reps : Array (String × String)

/-- One statement of the per-member form's evidence: the member it proves and its sessions. -/
structure StatementFiles where
  member : Option String
  sessions : Array RunFiles

/-- The evidence: v1's and the sessions' files, in v1's `sessions`' order, and in the per-member form the statements
those sessions are of, a sampled member each. -/
structure Evidence where
  base : OneStage.Evidence
  sessions : Array RunFiles
  statements : Option (Array StatementFiles)

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

/-- The refusal of a sampling design of kind `kind` that is no `subset` one, at the one statement's form
(`paramsOfJson`): its soundness statement covers a uniform sample of a fixed count alone. -/
def SUBSET_ONLY (kind : String) : String :=
  s!"the sampling design is {kind}, which the one-statement audit's soundness statement doesn't cover: it covers a \
    subset sampling design"

/-- The one statement's parameters as parsed, before `paramsOfJson`'s refusal of a sampling design that is no `subset`
one. The per-member form's parameters read them (`membersParamsOfJson`), under any sampling design v1 parses. -/
def paramsParsed (j : Json) : V Params := do
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
  pure { base, own := { programJ, program := (← Flock.canon programJ).toUTF8, partition, circuit, pub, registered },
         members := none }

/-- **The one statement's parameters** (`paramsParsed`), with every sampling design that is no `subset` one refused by
its kind (`SUBSET_ONLY`), after every refusal of `paramsParsed`. The kind is the parsed sampling design's, the one the
sample is sampled under (`OneStage.drawnFrom`, `Draw.deriveLaw`): v1 has already resolved a rate stated as text over
the population (`OneStage.design`), so `rate:NUM/DEN[:T]` is a `subset` one here and passes, and `stratified-rate:` and
`work-rate:` are refused as `stratified` and `work`. -/
def paramsOfJson (j : Json) : V Params := do
  let p ← paramsParsed j
  let kind := match getD p.base.verifier.law "law" with | .str s => s | _ => ""
  if kind != "subset" then throw (SUBSET_ONLY kind)
  pure p

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
  pure { base, sessions, statements := none }

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

/-- The pinned MUFU tables the circuit file `circuitB` looks up, in `Flock.MUFU_TABLES`' order: the names under its META's
`lookups` (`Flock.HmOut.metaOf`, the file's second line), none where it has none. -/
def lookupsOf (circuitB : ByteArray) : V (List String) := do
  match (← Flock.HmOut.metaOf circuitB).getObjVal? "lookups" with
  | .ok (.obj kvs) =>
    let names := kvs.toArray.map (·.1)
    pure ((Flock.MUFU_TABLES.filter fun t => names.contains t.name).map (·.name))
  | .ok .null | .error _ => pure []
  | .ok _ => throw "META: lookups is not an object"

/-- The refusal of blobs whose pinned MUFU tables are not the registered circuit's lookups (`tablesHeld`). -/
def NO_TABLES : String :=
  "tables: the blobs' pinned MUFU tables are not the ones the registered circuit looks up (its META's lookups)"

/-- The blobs' lookup tables held to the registration's (module docstring, **blobs**): the pinned MUFU tables the blobs
hold (`tablesOf`) must be the ones the registered circuit looks up (`lookupsOf`), else they refuse, naming both. -/
def tablesHeld (p : Own) (bs : Core.Blobs) : V Unit := do
  let want ← lookupsOf (← bs.get p.circuit "circuit")
  let got := (tablesOf bs).map (·.1)
  let names (ns : List String) := if ns.isEmpty then "none" else ", ".intercalate ns
  if got != want then throw s!"{NO_TABLES}: the circuit looks up {names want}, the blobs hold {names got}"

/-- `tablesHeld` past its reads: the lookup tables a statement is handed, `tables`, held to the ones its circuit looks
up, `want`. Other names refuse: a table the circuit does not look up, as one it looks up that is not handed. -/
def tablesMatch (tables : List (String × ByteArray)) (want : List String) : V Unit := do
  let got := tables.map (·.1)
  let names (ns : List String) := if ns.isEmpty then "none" else ", ".intercalate ns
  if got != want then throw s!"{NO_TABLES}: the circuit looks up {names want}, the blobs hold {names got}"

/-- `tablesHeld` is its reads of the circuit file, then `tablesMatch` at the blobs' lookup tables. -/
theorem tablesHeld_match (p : Own) (bs : Core.Blobs) :
    tablesHeld p bs =
      bs.get p.circuit "circuit" >>= fun circuitB => lookupsOf circuitB >>= tablesMatch (tablesOf bs) :=
  rfl

/-- The blobs' pinned MUFU tables among the names `names`, in `Flock.MUFU_TABLES`' order. -/
def tablesAmong (names : List String) (bs : Core.Blobs) : List (String × ByteArray) :=
  (tablesOf bs).filter fun t => names.contains t.1

/-- **A member's own lookup tables** (module docstring, **per member**): the blobs' pinned MUFU tables its registered
circuit looks up (`lookupsOf`), and no other the blobs hold, so a table another member's circuit looks up is none of
its. A refusal where no blob holds the circuit file or its META's `lookups` does not read. -/
def ownTables (p : Own) (bs : Core.Blobs) : V (List (String × ByteArray)) :=
  bs.get p.circuit "circuit" >>= fun circuitB => lookupsOf circuitB >>= fun names => pure (tablesAmong names bs)

/-- The case of the `tables_held` and `own_tables` pieces, `{"lookups": ..., "digests": [hex, ...]}`: the verifier's
own files naming a circuit file whose META is `{"lookups": lookups}` (`{}` where `j` has no `lookups`), and blobs
holding it and, under the digests `digests` (lowercase hex), whatever their bytes. -/
def tablesCase (j : Json) : V (Own × Core.Blobs) := do
  let .arr ds := getD j "digests" | throw "digests is not a list"
  let entries ← ds.mapM fun
    | .str s => return (← Flock.unhex s, ByteArray.empty)
    | _ => throw "digests is not a list of hex strings"
  let m := match j.getObjVal? "lookups" with
    | .ok l => Json.mkObj [("lookups", l)]
    | .error _ => Json.mkObj []
  let c := s!"circuit\nMETA {m.compress}\n".toUTF8
  let d := Flock.Sha512.hash c
  let own : Own :=
    { programJ := Json.null, program := .empty, partition := Json.null, circuit := Flock.hexOf d, pub := "",
      registered := none }
  return (own, ⟨#[(d, c)] ++ entries⟩)

/-- `tablesHeld` on the case `j` names (`tablesCase`): the `tables_held` piece, to `{"why": refusal or null}`. -/
def tablesPiece (j : Json) : V Json := do
  let (own, bs) ← tablesCase j
  return Json.mkObj [("why", match tablesHeld own bs with
    | .ok () => Json.null
    | .error w => Json.str w)]

/-- A member's own lookup tables on the case `j` names (`tablesCase`), held to its circuit's lookups as its statement
holds them (`ownTables`, then `tablesMatch`): the `own_tables` piece, to `{"tables": [name, ...], "why": refusal or
null}`, the tables by name, null where they do not read. -/
def ownTablesPiece (j : Json) : V Json := do
  let (own, bs) ← tablesCase j
  return match ownTables own bs with
    | .error w => Json.mkObj [("tables", Json.null), ("why", Json.str w)]
    | .ok tables =>
      Json.mkObj [("tables", Lean.toJson (tables.map (·.1))),
        ("why", match bs.get own.circuit "circuit" >>= lookupsOf >>= tablesMatch tables with
          | .ok () => Json.null
          | .error w => Json.str w)]

/-- What `flock-verify verify --zk` reads besides a session's directory, as `verdicts` runs it: the statement's tags
`tags`, the circuit and public files, one table, the lookup tables the blobs hold, and the verifier's own partition
object and program. -/
def inputsOf (tags : Flock.Tags) (p : Own) (circuitB pubB : ByteArray) (bs : Core.Blobs) : Flock.Witness.Inputs :=
  { tags, circuitFile := circuitB, publicFile := pubB, count := 1, tables := tablesOf bs, partition := some p.partition,
    program := some p.program }

/-- `inputsOf` at the lookup tables `tables` a statement is handed, rather than every pinned table the blobs hold. -/
def inputsAt (tags : Flock.Tags) (p : Own) (circuitB pubB : ByteArray) (tables : List (String × ByteArray)) :
    Flock.Witness.Inputs :=
  { tags, circuitFile := circuitB, publicFile := pubB, count := 1, tables, partition := some p.partition,
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

/-- `some "the draw"` when a session's statement would hold its record's sample `draw` at `limit` bytes or more
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
registered openings, once the sampling design holds. -/
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
no blob or not UTF-8, its sample doesn't fit the session's statement (`drawFitsWhy`), or it accepted naming another
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
(`FilePart.check`, last). A refusal, first, of blobs holding other lookup tables than the registered circuit's (`tablesHeld`),
then for what `proofs.verify` refuses: registered values read without the verifier's registered file, a
file no blob holds, an accepted session naming another program than the partition object's, or proved units other than
the evidence's (`provedHeld`); and for a length a session's statement can't hold (`fitsWhy`, `drawFitsWhy`, at
`WORD`). -/
def run (p : Own) (sessions : Array RunFiles) (provedUnits? : Option (Array Nat)) (bs : Core.Blobs) : V Run := do
  tablesHeld p bs
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
  let (sch, _) ← Flock.Zk.schedule st.wideLeaf st.m
  pure (2 ^ (sch.levels.getD 0 default).cap * st.merkle.digestLen)

/-- The session's tables (`sessionTables`), each with its `root_B`'s byte length (`rootBLen`) at the session `record`
sets up: `setupAt` at the record's sample on what `inputs` makes of the statement's tags and the circuit and public
files of the verifier's own files `p`. None where it sets up no table. -/
def rootBTablesAt (inputs : Flock.Tags → ByteArray → ByteArray → V Flock.Witness.Inputs) (p : Own) (bs : Core.Blobs)
    (record : String) : Array (String × Option Nat) :=
  let tabs : Option (Array Flock.Setup) := (do
    let some tags0 := Flock.Tags.find? STATEMENT | throw s!"the statement {STATEMENT} is not known"
    let I ← inputs tags0 (← bs.get p.circuit "circuit") (← bs.get p.pub "public file")
    let (_, tabs, _) ← setupAt I (← readsOf p.registered) (Flock.Witness.recordDraw record)
    pure tabs : V (Array Flock.Setup)).toOption
  sessionTables.mapIdx fun i t => (t, tabs.bind fun ts => ts[i]?.bind fun st => (rootBLen st).toOption)

/-- `rootBTablesAt` on `inputsOf`, every pinned table the blobs hold, as `run` sets the session up. None where it sets
up no table, and then `run` sets up none either, so the session's own verdict rejects. -/
def rootBTables (p : Own) (bs : Core.Blobs) (record : String) : Array (String × Option Nat) :=
  rootBTablesAt (fun tags circuitB pubB => pure (inputsOf tags p circuitB pubB bs)) p bs record

/-- `rootBTablesAt` for a member's session: on the member's own files `p` at its own lookup tables (`ownTables`), the
inputs its statement holds (`Member.statement`), whatever other pinned table the blobs hold. -/
def ownRootBTables (p : Own) (bs : Core.Blobs) (record : String) : Array (String × Option Nat) :=
  rootBTablesAt (fun tags circuitB pubB => do pure (inputsAt tags p circuitB pubB (← ownTables p bs))) p bs record

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
(`Slots.popOk`). This module imports `Spot.PopOrder`, so the call runs from the query's ranges
(`Slots.popOk_eq_popOkRanges`) and builds no proof block: without the import it would run `Slots.layout`, a proof block
for every input gate of the root. -/
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

/-- The one statement's check at the coins `ξ` (module docstring, **check**): the evidence's one session
(`oneSession`), its coins held to the trees the challenger logged in the order it must release them (`treesOf`,
`orderHeld`, `treesHeld`), the registration's population held to the program (`popHeld`), the parameters' public file
held to the registration (`publicHeld`), then the session's verdict on the blobs, then v1's `checkAt` at `ξ` on it. v1
accepts only verdicts that all accepted (`VerdictsAccept`); the run's own `all` says it again, so an acceptance reads
the run rather than the verdicts' JSON. -/
def checkOneAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
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

/-! ## The per-member form (module docstring, **per member**) -/

/-- The ranges ascend from `at_`: each `[lo, hi)` holds a unit and starts at or after the end of the one before. -/
def ascending : Nat → List (Nat × Nat) → Bool
  | _, [] => true
  | at_, (lo, hi) :: rs => decide (at_ ≤ lo) && decide (lo < hi) && ascending hi rs

/-- The per-member form's `members`, on the program and the partition `shared` every member's statement reads: each a
nonempty name with exactly its keys, its circuit and public files' digests and registered file as the one statement's
(`committedOf`), its units ranges `[lo, hi)` of naturals, ascending (`ascending`), its circuit's units a proof block
positive, and no unit in two members' ranges. -/
def membersOf (shared : Own) (j : Json) : V (Array Member) := do
  let .obj _ := j
    | throw "proofs: members is not an object from a member's name to its files, its units and its vus_per_block"
  let ms ← (entries j).mapM fun (name, m) => do
    if name.isEmpty then throw "proofs: a member's name is empty"
    exactly m #["circuit", "public", "registered", "units", "vus_per_block"] s!"member {name}"
    let (circuit, pub, registered) ← committedOf m
    let some units := (Draw.ranges? (getD m "units")).bind fun rs =>
        rs.mapM fun (lo, hi) => if 0 ≤ lo then some (lo.toNat, hi.toNat) else none
      | throw s!"proofs: member {name}'s units are not a nonempty list of [lo, hi] with 0 <= lo < hi"
    if !ascending 0 units.toList then
      throw s!"proofs: member {name}'s units are not ascending, each range after the one before"
    let some g := (nat? (getD m "vus_per_block")).filter (0 < ·)
      | throw s!"proofs: member {name}'s vus_per_block is not a positive natural number"
    pure ({ name, units, g, own := { shared with circuit := circuit, pub := pub, registered := registered } } : Member)
  if ms.isEmpty then throw "proofs: members names no member"
  let rs := (ms.foldl (fun acc m => acc ++ m.units) #[]).qsort (·.1 < ·.1)
  for i in [1:rs.size] do
    if rs[i]!.1 < rs[i - 1]!.2 then
      throw s!"proofs: unit {rs[i]!.1} is in two members' units: a unit is one member's"
  pure ms

/-- The per-member form's evidence `proofs`: exactly `{"statements"}`, a list of exactly `{"member", "sessions"}`, a
member's nonempty name and its sessions (`sessionOf`). -/
def statementsOf (pr : Json) : V (Array StatementFiles) := do
  exactly pr #["statements"] "the evidence's proofs"
  let .arr ss := getD pr "statements" | throw "proofs: statements is not a list"
  ss.mapM fun s => do
    exactly s #["member", "sessions"] "a statement of the proofs"
    let .str member := getD s "member" | throw "proofs: a statement's member is not a member's name"
    if member.isEmpty then throw "proofs: a statement's member is not a member's name"
    let .arr rs := getD s "sessions" | throw "proofs: a statement's sessions is not a list"
    pure { member := some member, sessions := ← rs.mapM sessionOf }

/-- A digest in the place of a file's the per-member form's parameters name none of (`membersParamsOfJson`). -/
private def NO_FILE : String := "".pushn '0' 128

/-- **The per-member form's parameters**: `proofs` exactly `{"members"}` (`membersOf`, on the program and the
partition every member's statement reads), and everything else as the one statement's parameters hold it
(`paramsParsed`, read at `proofs` of no file, so under any sampling design v1 parses and not the one statement's `subset` alone):
`own` is that program and partition, with no file. The registration's check reads the members, each its units and its
circuit's units a proof block (`Registration.Verifier.members`), so v1's check refuses a registration under which a
sample that ends misses a member, or a member's sampled proof blocks give two counts of link points (`link-points`,
`Registration.linkRefused`). -/
def membersParamsOfJson (j : Json) : V Params := do
  let some pr := field? j "proofs" | throw "the parameters name no proofs (the members and their statements' files)"
  exactly pr #["members"] "proofs"
  let one := Json.mkObj [("circuit", Json.str NO_FILE), ("public", Json.str NO_FILE), ("registered", Json.null)]
  let p ← paramsParsed (Json.mkObj (without j "proofs" ++ [("proofs", one)]))
  let own := { p.own with circuit := "", pub := "" }
  let ms ← membersOf own (getD pr "members")
  pure { p with base := { p.base with verifier := { p.base.verifier with members := ms.map fun m => (m.units, m.g) } },
                own, members := some ms }

/-- **The parameters in either form**: the per-member form's where `proofs` is exactly `{"members"}`, else the one
statement's. `protocol` reads its parameters by it. -/
def paramsEitherOfJson (j : Json) : V Params :=
  if (field? j "proofs").bind keys? == some #["members"] then membersParamsOfJson j else paramsOfJson j

/-- **The per-member form's evidence**: `proofs` exactly `{"statements"}` (`statementsOf`), and everything else as the
one statement's evidence holds it (`evidenceOf`, read at `proofs` of the statements' sessions in order); its
sessions are the statements', as `statementsOf` read them, in v1's `sessions`' order (`membersEvidenceOf_sessions`). -/
def membersEvidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  let some pr := field? j "proofs" | throw "the evidence names no proofs (the statements' sessions' files)"
  let statements ← statementsOf pr
  let sessions := match getD pr "statements" with
    | .arr ss => ss.flatMap fun s => match getD s "sessions" with | .arr rs => rs | _ => #[]
    | _ => #[]
  let one := Json.mkObj (without j "proofs" ++ [("proofs", Json.mkObj [("sessions", Json.arr sessions)])])
  let e ← evidenceOf (← Flock.canon one).toUTF8
  pure { e with sessions := statements.flatMap (·.sessions), statements := some statements }

/-- **The evidence in either form**: the per-member form's where `proofs` is exactly `{"statements"}`, else the one
statement's. `protocol` reads its evidence by it. -/
def evidenceEitherOf (b : ByteArray) : V Evidence :=
  match Core.canonicalJson b with
  | .ok j => if (field? j "proofs").bind keys? == some #["statements"] then membersEvidenceOf b else evidenceOf b
  | .error why => .error why

/-- A member's first unit, its lowest. Members' units are disjoint, so it orders them, in the registration's order. -/
def Member.first (m : Member) : Nat := m.units.foldl (fun a x => min a x.1) ((m.units[0]?.map (·.1)).getD 0)

/-- A member's units, each range's in order. -/
def Member.indices (m : Member) : Array Nat := m.units.flatMap fun x => (Array.range (x.2 - x.1)).map (x.1 + ·)

/-- Whether the member `m`'s statement may follow the members `held`'s: its first unit is not before the last one's. -/
def follows (held : Array Member) (m : Member) : Bool :=
  match held.back? with
  | some prev => !decide (m.first < prev.first)
  | none => true

/-- One statement held, after the members `held` of the statements before it: it names a member of the parameters
`ms` that no earlier statement named, in the members' order (`follows`: by first unit, the registration's, one order
so one statement), and has exactly one session (one pass on live coins, never retried; none would prove nothing, and
an unsampled member has no statement). The members so far, with its own. -/
def heldStep (ms : Array Member) (held : Array Member) (st : StatementFiles) : V (Array Member) :=
  match st.member with
  | none => throw "a statement of the proofs names no member, and the parameters' proofs are per member"
  | some name =>
    match ms.find? (·.name == name) with
    | none => throw s!"member {name}: the parameters name no such member"
    | some m =>
      match held.any (·.name == name) with
      | true => throw s!"member {name}: a second statement: a member's units are one statement's"
      | false =>
        match follows held m with
        | false =>
          throw (s!"member {name}: its statement follows a member's whose units start after its own: the " ++
            "statements are in the members' order (by first unit, the registration's)")
        | true =>
          match st.sessions.size != 1 with
          | true =>
            throw (s!"member {name}: its statement has {st.sessions.size} sessions: a statement is exactly one " ++
              "session on live coins, never retried, and an undrawn member has no statement")
          | false => pure (held.push m)

/-- **The per-member form's statements as public structure**, before any session is read (`heldStep` on each, in
order): their members, statement by statement. -/
def membersHeld (ms : Array Member) (sts : Array StatementFiles) : V (Array Member) :=
  sts.foldlM (heldStep ms) #[]

/-- The header `h`'s `units.indices` are exactly the units `us`, in order: `indicesHeld` at a member's units in place
of the population's. -/
def unitsHeld (us : Array Nat) (h : Json) : Bool :=
  match getD (getD h "units") "indices" with
  | .arr is => is.size == us.size && (List.range us.size).all fun i => (is[i]? >>= nat?) == us[i]?
  | _ => false

/-- The refusal of a member's public file of other units than the member's (`Member.population`). -/
def MEMBER_UNITS : String :=
  "units: the public file's units.indices are not exactly the member's units, in order: slot i of its table is the " ++
    "member's i-th unit"

/-- The units of the ranges `us` in `[a, b)`. -/
def unitsIn (us : Array (Nat × Nat)) (a b : Nat) : Nat := us.foldl (fun n x => n + (min x.2 b - max x.1 a)) 0

/-- **A member's strata's sizes**: the registration's `strata` (`Partition.strata`), each restricted to the member's
units, one size a stratum in the strata's order, a stratum holding none of the member's units at 0. The strata read
stratum by stratum are the registration's units in order (`FilePart.sizes`) and the member's ranges ascend, so the
member's units read stratum by stratum are its units in order (`Member.indices`): slots `[Σ_{k' < k} n_k', Σ_{k' ≤ k}
n_k')` of its table are its units in stratum `k`. None where the strata are not the units in order or a unit of the
member's is past them. A function of the parameters alone: nothing the prover sends moves a unit between strata. -/
def Member.sizes (m : Member) (strata : Array (String × Array (Nat × Nat))) : Option (Array Nat) :=
  (FilePart.sizes strata).bind fun ns =>
    let z := ns.foldl (fun (z : Nat × Array Nat) n => (z.1 + n, z.2.push (unitsIn m.units z.1 (z.1 + n)))) (0, #[])
    if m.units.all (·.2 ≤ z.1) then some z.2 else none

/-- The refusal of a member whose strata the registration's do not give (`Member.sizes`). -/
def MEMBER_STRATA : String :=
  "partition: the registration's strata, read stratum by stratum, are not its units in order, or a unit of the " ++
    "member's is past them: a member's strata are the registration's, each restricted to the member's units"

/-- A member's population check at the registration's `strata` and its public file's header `h`: the member's strata
are the registration's restricted to its units (`Member.sizes`), and the header's units are exactly the member's, in
order (`unitsHeld`). -/
def Member.heldAt (m : Member) (strata : Array (String × Array (Nat × Nat))) (h : Json) : V Unit :=
  match m.sizes strata with
  | none => throw MEMBER_STRATA
  | some _ => if unitsHeld m.indices h then pure () else throw MEMBER_UNITS

/-- The port a registration names a window's row tables' root at (`verity.commitments.windows.rows.TABLE_PORT`). -/
def TABLE_PORT : String := "row-table"

/-- The refusal of a registration that names a row tables' root (`rootsOf`). -/
def ROW_TABLES : String :=
  "roots: the registration names a row tables' root (row-table), the window's and no member's: the per-member " ++
    "form is refused on a window with row tables until its guarantee covers a root every member reads"

/-- Every way a root's port reads as `<name>/<port name>`: the name and the port name at each `/` of the characters
given, the first `/` first, after the characters `pre` holds reversed. A member's name holds `/`s, so each `/` of a
port is a place a name can end. -/
def splitsFrom : List Char → List Char → List (String × String)
  | _, [] => []
  | pre, c :: cs =>
    if c = '/' then (String.ofList pre.reverse, String.ofList cs) :: splitsFrom (c :: pre) cs
    else splitsFrom (c :: pre) cs

/-- Every way the port `port` reads as `<name>/<port name>` (`splitsFrom`). -/
def splitsOf (port : String) : List (String × String) := splitsFrom [] port.toList

/-- One root `r` of the registration given to its member, among the members `acc` holds a name of (`rootsOf`): the
member whose name, then `/`, the root's port begins with, the root pushed on that member's with the name and the `/`
taken off its port. A refusal where the root is a row tables' (`ROW_TABLES`), no member's, or two members'. -/
def rootStep (acc : Std.HashMap String (Array Json)) (r : Json) : V (Std.HashMap String (Array Json)) := do
  let .str port := getD r "port" | throw "roots: a root of the registration names no port"
  if port == TABLE_PORT then throw ROW_TABLES
  match (splitsOf port).filter fun x => acc.contains x.1 with
  | [(name, bare)] =>
    pure (acc.modify name (·.push (Json.mkObj [("domain", getD r "domain"), ("leaves", getD r "leaves"),
      ("port", .str bare), ("root", getD r "root")])))
  | [] => throw s!"roots: the registration's root {port} is no member's: a member's roots are named <member>/<port>"
  | _ =>
    throw (s!"roots: the registration's root {port} is two members': a member's roots are named " ++
      "<member>/<port>, and a root is one member's")

/-- **The registration's roots, each its member's**: the roots of the record `rec` by the name of the member each is
of, in the record's order, each as the record holds it with the member's name and the `/` after it taken off its
port (`rootStep`). A root is the member's whose name, then `/`, its port begins with, the whole name: the record's
writer names a member's ports `<name>/<port name>`, and that rule is the prover's, so it is held here. A root that is
no member's refuses, and one that is two members' (a name holds `/`s, so one port can read as two members'), and a row
tables' root (`ROW_TABLES`). So every root of a record this reads is exactly one member's, by a function of the
parameters' members and the record alone, and a member's statement reads no other (`Member.slice`). One pass over the
record's roots. -/
def rootsOf (ms : Array Member) (rec : Json) : V (Std.HashMap String (Array Json)) :=
  match getD rec "roots" with
  | .arr rs => rs.foldlM rootStep (ms.foldl (fun acc m => acc.insert m.name #[]) {})
  | _ => throw "roots: the registration's roots are not a list"

/-- How many units a member holds. -/
def Member.count (m : Member) : Nat := m.units.foldl (fun n x => n + (x.2 - x.1)) 0

/-- **A member's slice of the registration `rec`**, at the roots `rootsOf` gives each member: the window's scheme,
the member's roots, their ports bare as its public file names them, and the member's units' count as its population.
What the one statement's public file is held to of the record (`publicHeld`), restricted to the member; the verifier
computes it from the parameters and the record, and nothing the prover sends after them chooses it. -/
def Member.slice (m : Member) (rec : Json) (roots : Std.HashMap String (Array Json)) : Json :=
  Json.mkObj [("population", Lean.toJson m.count), ("roots", .arr (roots.getD m.name #[])),
    ("scheme", getD rec "scheme")]

/-- A public file's header `h` held to the record `rec`, as `publicHeld` holds the one statement's
(`Record.matchesPublic`): its scheme, its `frame_v3` roots and their `domain_ids`, and its `instances` against the
record's scheme, roots, domains and population, a refusal naming the codes that fail. -/
def headerHeld (rec h : Json) : V Unit :=
  let fv := getD h "frame_v3"
  let c := Record.matchesPublic rec (getD h "scheme") (getD fv "roots") (getD fv "domain_ids") (getD h "instances")
  if c.codes.isEmpty then pure () else throw s!"public file: not the registration's: {", ".intercalate c.codes.toList}"

/-- **A member's population check**, in the place of the one statement's `FilePart.check` and `publicHeld`
(`Member.heldAt` at the strata of the member's query over its program, as `FilePart.check` reads them, then
`headerHeld` at the member's `slice` of the registration): its strata are the registration's restricted to its units,
of the sizes `Member.sizes` gives, its public file `pubB` holds exactly the member's units, in order, so slot `i` of
its session's table is the member's `i`-th unit and the units a session proves (`provedUnits`, read off this file)
are the member's, and the file's scheme, roots, domains and instances are its slice's. -/
def Member.population (m : Member) (slice : Json) (pubB : ByteArray) : V Unit := do
  let strata ← Partition.strata (← Program.parse m.own.program) (getD m.own.partition "query")
  let h ← FilePart.header pubB
  m.heldAt strata h
  headerHeld slice h

/-- A member's circuit held to the parameters' units a proof block, at the units a proof block `g` its circuit file
parses with: a circuit file that parses with another `vus_per_block` refuses. One that does not parse (`none`) passes
here, and the scope check of the member's files refuses it (`StepAt.filesHeld`). -/
def Member.blockHeld (m : Member) (g : Option Nat) : V Unit :=
  match g with
  | some g =>
    if g != m.g then
      throw s!"vus_per_block: the member's circuit has {g} units a proof block, not the {m.g} the parameters name"
    else pure ()
  | none => pure ()

/-- The scope check a session's verdict runs at the inputs `I` and the registered reads `rd`, as `run` builds it, on
the circuit as `parsed` read it. -/
def scopeAt (parsed : V Flock.Circuit) (I : Flock.Witness.Inputs) (rd : Reads) : V Unit := do
  Flock.ProvedScope.checkOf parsed (Flock.Witness.zkTags I) true I.publicFile I.count false rd.registered.isSome
  templateOk (← parsed)

/-- **What `step` reads of one statement**: its inputs, the verifier's registered reads, the scope, files and
population checks its sessions' verdicts run, and its program's digest against the partition object's program; and,
of its circuit file alone, the form the sampled audit's knowledge soundness covers without a session's sample
(`ksForm`) and the units a proof block it parses with (`g`). Nothing of it reads a session's record, coins or sample,
and it holds each check's result, never the parsed circuit. -/
structure StepAt where
  I : Flock.Witness.Inputs
  rd : Reads
  scope : V Unit
  ksFiles : V Unit
  ksForm : V Unit
  g : Option Nat
  population : V Unit
  programSha : Option String
  want : Json

/-- One session's `step` at the statement `a`, on the blobs `bs`, after the run `r`. -/
def StepAt.step (a : StepAt) (bs : Core.Blobs) (r : Run) (s : RunFiles) : V Run :=
  OneStageZk.step a.I a.rd bs a.scope a.ksFiles a.population a.programSha a.want r s

/-- The statement at the verifier's own inputs `p`, the tags `tags0`, the circuit and public files, the lookup tables
`tables` and the registered reads `rd`, as `run` builds its own at the blobs' tables, with `population` its population
check. The circuit file is parsed once, and every check that reads the parse is made on it here. -/
def StepAt.ofOwn (p : Own) (tags0 : Flock.Tags) (circuitB pubB : ByteArray) (tables : List (String × ByteArray))
    (rd : Reads) (population : V Unit) : StepAt :=
  let I := inputsAt tags0 p circuitB pubB tables
  let parsed := Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables
  { I, rd, scope := scopeAt parsed I rd
    ksFiles := Flock.PortRoots.checkOf parsed (Flock.Witness.zkTags I) pubB
    ksForm := Flock.KsScope.filesOf parsed (Flock.Witness.zkTags I) I.publicFile I.count
    g := parsed.toOption.map (·.g)
    population
    programSha := programDigest p.program
    want := getD p.partition "program" }

/-- **The statement the sessions at the inputs `p` are held to on the lookup tables `tables` it is handed, read off
the blobs before any session**: `run`'s own checks in `run`'s order (the tables handed the circuit's, by name, as `run`
holds the blobs' to them, `tablesMatch`; the statement's tags with hiding leaves, registered values read only with the
verifier's registered file, the circuit and public files, each length within a statement's word, the registered
reads), then `StepAt.ofOwn` at those tables with the population check `population` on the public file. A refusal
where `run` refuses on blobs whose pinned tables are `tables`. -/
def statementOf (p : Own) (population : ByteArray → V Unit) (tables : List (String × ByteArray)) (bs : Core.Blobs) :
    V StepAt :=
  (bs.get p.circuit "circuit" >>= fun circuitB => lookupsOf circuitB >>= tablesMatch tables) >>= fun _ =>
  match Flock.Tags.find? STATEMENT with
  | none => throw s!"the statement {STATEMENT} is not known"
  | some tags0 =>
    match tags0.plainLeaves with
    | true => throw s!"{tags0.name}: a statement with plain SHA-512 leaves: --zk's hiding needs hm96 leaves"
    | false =>
      match p.registered.isNone && readsRegistered p.programJ with
      | true =>
        throw ("registered: the program reads registered values (a RegisteredInput among its definitions) and the " ++
          "verifier holds no registered reads: verify with its own registered file")
      | false =>
        bs.get p.circuit "circuit" >>= fun circuitB =>
        bs.get p.pub "public file" >>= fun pubB =>
        match fitsWhy WORD circuitB pubB tables (some p.program) (some p.partition) p.registered with
        | some what => throw s!"{what}: {FITS}"
        | none =>
          readsOf p.registered >>= fun rd => pure (StepAt.ofOwn p tags0 circuitB pubB tables rd (population pubB))

/-- **The statement a member's session is held to**: `statementOf` at the member's own inputs and its own lookup tables
(`ownTables`: the blobs' pinned tables its circuit looks up, so a table of the blobs that only another member's circuit
looks up does not refuse it, and one its circuit looks up that the blobs do not hold does, `NO_TABLES`), its
population check the member's at its `slice` of the registration (`Member.population`), and its circuit's units a
proof block the parameters' (`Member.blockHeld`). -/
def Member.statement (m : Member) (slice : Json) (bs : Core.Blobs) : V StepAt :=
  (ownTables m.own bs >>= fun tables => statementOf m.own (m.population slice) tables bs) >>= fun a =>
    m.blockHeld a.g >>= fun _ => pure a

/-- **A statement's files held to what its sessions' verdicts check of the files alone**, with no session: the scope
(`scope`), the files' ports and roots (`ksFiles`), the form the sampled audit's knowledge soundness covers, without a
session's sample (`ksForm`, `Flock.KsScope.filesOf`), and the population check (`population`), in `verdictOf`'s
order. It reads the statement's results and parses nothing. -/
def StepAt.filesHeld (a : StepAt) : V Unit :=
  a.scope >>= fun _ => a.ksFiles >>= fun _ => a.ksForm >>= fun _ => a.population

/-- **A member's statement with its files held** (`Member.statement`, then `StepAt.filesHeld`), a refusal naming the
member. It reads the member's own files and no session, so it is the same for a member the sample holds and one it
missed. -/
def Member.files (m : Member) (slice : Json) (bs : Core.Blobs) : V StepAt :=
  (m.statement slice bs >>= fun a => a.filesHeld >>= fun _ => pure a).mapError (s!"member {m.name}: {·}")

/-- **Every member's files, sampled or not**: each member's statement with its files held (`Member.files`), at its
`slice` of the registration, in the members' order, under its name. Compiled as `membersFilesKept`, which reads the
blobs' lookup tables once, a circuit file once for the members that name it, and a program and the strata of its
partition's query once for the members in a row that read it, and returns the same (`membersFiles_eq_kept`). -/
def membersFiles (slice : Member → Json) (bs : Core.Blobs) : List Member → V (List (String × StepAt))
  | [] => pure []
  | m :: ms => m.files (slice m) bs >>= fun a => membersFiles slice bs ms >>= fun as => pure ((m.name, a) :: as)

/-- The tags a statement's sessions run under (`Flock.Witness.zkTags`): of the statement's inputs they read only its
own tags `tags0` and its circuit file. -/
def tagsAt (tags0 : Flock.Tags) (circuitB : ByteArray) : Flock.Tags :=
  Flock.Zk.tags tags0 (Flock.Zk.maskLogOf circuitB) 1

/-- **What the files' pass keeps of a circuit file**, each a function of the file, the statement's tags and the
blobs' lookup tables, so the members that name one file share one: the blob under its digest (`circuit`), the pinned
tables its META looks up (`lookups`), those of them the blobs hold, the lookup tables of a member that names the file
(`tables`, as `ownTables` reads them), the tags its statement's sessions run under (`tags`), and what the statement's
checks read of its parse at those tables (`kept`: the circuit's own verdicts and the circuit without its rows,
`Spot.CircuitKept.Kept`). It does not hold the parse. -/
structure FileKept where
  circuit : V ByteArray
  lookups : V (List String)
  tables : V (List (String × ByteArray))
  tags : Flock.Tags
  kept : V CircuitKept.Kept

/-- What the files' pass keeps of the circuit file under the digest `d`, at the statement's tags `tags0` and the
blobs' lookup tables `pinned`: the file is found, its META read and its rows parsed, at the tables of `pinned` it
looks up, here and nowhere else in the pass. -/
def fileOf (tags0 : Flock.Tags) (pinned : List (String × ByteArray)) (bs : Core.Blobs) (d : String) : FileKept :=
  match bs.get d "circuit" with
  | .error e => { circuit := .error e, lookups := .error e, tables := .error e, tags := tags0, kept := .error e }
  | .ok circuitB =>
    match lookupsOf circuitB with
    | .error e =>
      { circuit := .ok circuitB, lookups := .error e, tables := .error e, tags := tagsAt tags0 circuitB
        kept := .error e }
    | .ok names =>
      { circuit := .ok circuitB, lookups := .ok names, tables := .ok (pinned.filter fun t => names.contains t.1)
        tags := tagsAt tags0 circuitB
        kept := (Flock.ProvedScope.circuitOf (tagsAt tags0 circuitB) circuitB
            (pinned.filter fun t => names.contains t.1)).map
          (CircuitKept.keptOf (tagsAt tags0 circuitB) true 1 templateOk) }

/-- What `fileOf` keeps under a digest whose blob is `circuitB`, a circuit file that looks up `names`. -/
theorem fileOf_ok {tags0 : Flock.Tags} {pinned : List (String × ByteArray)} {bs : Core.Blobs} {d : String}
    {circuitB : ByteArray} {names : List String} (hc : bs.get d "circuit" = .ok circuitB)
    (hl : lookupsOf circuitB = .ok names) :
    fileOf tags0 pinned bs d =
      { circuit := .ok circuitB, lookups := .ok names, tables := .ok (pinned.filter fun t => names.contains t.1)
        tags := tagsAt tags0 circuitB
        kept := (Flock.ProvedScope.circuitOf (tagsAt tags0 circuitB) circuitB
            (pinned.filter fun t => names.contains t.1)).map
          (CircuitKept.keptOf (tagsAt tags0 circuitB) true 1 templateOk) } := by
  unfold fileOf
  rw [hc]
  dsimp only
  rw [hl]

/-- What `fileOf` keeps under a digest whose blob's META's `lookups` does not read: the refusal. -/
theorem fileOf_unread {tags0 : Flock.Tags} {pinned : List (String × ByteArray)} {bs : Core.Blobs} {d e : String}
    {circuitB : ByteArray} (hc : bs.get d "circuit" = .ok circuitB) (hl : lookupsOf circuitB = .error e) :
    fileOf tags0 pinned bs d =
      { circuit := .ok circuitB, lookups := .error e, tables := .error e, tags := tagsAt tags0 circuitB
        kept := .error e } := by
  unfold fileOf
  rw [hc]
  dsimp only
  rw [hl]

/-- What `fileOf` keeps under a digest no blob has: the refusal. -/
theorem fileOf_error {tags0 : Flock.Tags} {pinned : List (String × ByteArray)} {bs : Core.Blobs} {d e : String}
    (hc : bs.get d "circuit" = .error e) :
    fileOf tags0 pinned bs d =
      { circuit := .error e, lookups := .error e, tables := .error e, tags := tags0, kept := .error e } := by
  unfold fileOf
  rw [hc]

/-- A bind on a value returned is its continuation at the value. -/
theorem ok_bind {α β : Type} (a : α) (f : α → V β) : (Except.ok a : V α) >>= f = f a :=
  rfl

/-- A bind on a refusal is the refusal. -/
theorem error_bind {α β : Type} (e : String) (f : α → V β) : (Except.error e : V α) >>= f = Except.error e :=
  rfl

/-- The strata of the partition `partition`'s query over a program as parsed, `parsed`: what a member's population
check reads of its program and its partition (`Member.population`). -/
def strataOn (parsed : V Program) (partition : Json) : V (Array (String × Array (Nat × Nat))) :=
  parsed >>= fun p => Partition.strata p (getD partition "query")

/-- Whether two readings of a query's strata are one: the same strata, or the same refusal. -/
def sameStrata : V (Array (String × Array (Nat × Nat))) → V (Array (String × Array (Nat × Nat))) → Bool
  | .ok a, .ok b => decide (a = b)
  | .error a, .error b => decide (a = b)
  | _, _ => false

/-- Two readings `sameStrata` takes as one are equal. -/
theorem sameStrata_eq : ∀ {a b : V (Array (String × Array (Nat × Nat)))}, sameStrata a b = true → a = b
  | .ok _, .ok _, h => congrArg Except.ok (of_decide_eq_true h)
  | .error _, .error _, h => congrArg Except.error (of_decide_eq_true h)
  | .ok _, .error _, h => nomatch h
  | .error _, .ok _, h => nomatch h

/-- `sameStrata` takes a reading as itself. -/
theorem sameStrata_refl : ∀ (a : V (Array (String × Array (Nat × Nat)))), sameStrata a a = true
  | .ok _ => decide_eq_true rfl
  | .error _ => decide_eq_true rfl

/-- **What the files' pass keeps of a program**, for the members that read it: the bytes, their parse
(`Program.parse`) and the digest a session's record must name (`programDigest`), each a function of the bytes; and
the partition the member before read with it, with the strata of that partition's query over the parse (`strata`,
which `held` says they are), so the members in a row that read one program under one partition evaluate its query
once. -/
structure ProgramKept where
  bytes : ByteArray
  parsed : V Program
  sha : Option String
  partition : Json
  strata : V (Array (String × Array (Nat × Nat)))
  held : strata = strataOn parsed partition

/-- What is kept of the program `bytes`, parsed as `parsed` and of the digest `sha`, under the partition
`partition`: the strata of its query evaluated here. -/
def programRead (bytes : ByteArray) (parsed : V Program) (sha : Option String) (partition : Json) : ProgramKept :=
  { bytes, parsed, sha, partition, strata := strataOn parsed partition, held := rfl }

/-- What the files' pass keeps of the program `b` under the partition `partition`: `b` parsed and hashed here, and
the partition's query evaluated on the parse. -/
def programOf (b : ByteArray) (partition : Json) : ProgramKept :=
  programRead b (Program.parse b) (programDigest b) partition

/-- A program kept with any strata that are its partition's is the program kept with them evaluated. -/
theorem programRead_of {bytes : ByteArray} {parsed : V Program} {sha : Option String} {partition : Json}
    {strata : V (Array (String × Array (Nat × Nat)))} (h : strata = strataOn parsed partition) :
    ({ bytes, parsed, sha, partition, strata, held := h } : ProgramKept) = programRead bytes parsed sha partition := by
  subst h
  rfl

/-- The strata kept are those of the kept partition's query, so `sameStrata` takes the two as one under a partition
equal to the kept one. -/
theorem ProgramKept.same (pk : ProgramKept) {partition : Json} (e : partition = pk.partition) :
    sameStrata (strataOn pk.parsed partition) pk.strata = true := by
  subst e
  rw [← pk.held]
  exact sameStrata_refl pk.strata

/-- What `withPtrEq` answers is what its continuation answers. -/
theorem withPtrEq_answer {α : Type} (a b : α) (k : Unit → Bool) (h : a = b → k () = true) :
    withPtrEq a b k h = k () := by
  simp only [withPtrEq]

/-- **What is kept of `pk`'s program for a member whose partition is `partition`**: `pk`'s strata where `partition`
is the partition `pk` evaluated them under, else the strata of `partition`'s query on `pk`'s parse. Compiled, the
partition the members share is one object, which `withPtrEq` sees without reading it, so no query is evaluated; a
partition that is another object has its query evaluated and the strata compared (`sameStrata`), and evaluated once
more where they differ. Either way the result is `programRead` at `partition` (`ProgramKept.under_eq`). -/
def ProgramKept.under (pk : ProgramKept) (partition : Json) : ProgramKept :=
  if h : withPtrEq partition pk.partition (fun _ => sameStrata (strataOn pk.parsed partition) pk.strata)
      (fun e => pk.same e) = true then
    { bytes := pk.bytes, parsed := pk.parsed, sha := pk.sha, partition, strata := pk.strata
      held := (sameStrata_eq (a := strataOn pk.parsed partition) (b := pk.strata)
        ((withPtrEq_answer _ _ _ _).symm.trans h)).symm }
  else programRead pk.bytes pk.parsed pk.sha partition

/-- What `ProgramKept.under` keeps is `pk`'s program with the strata of `partition`'s query on its parse. -/
theorem ProgramKept.under_eq (pk : ProgramKept) (partition : Json) :
    pk.under partition = programRead pk.bytes pk.parsed pk.sha partition := by
  unfold ProgramKept.under
  split
  · exact programRead_of _
  · rfl

/-- What is kept of the program `b` under the partition `partition` after the member before read `prog`: `prog`
under `partition` where its bytes are `b` (`ProgramKept.under`), else `b` parsed and hashed anew, so members in a
row that read one program parse and hash it once, and evaluate their partition's query on it once. -/
def ProgramKept.next (prog : Option ProgramKept) (b : ByteArray) (partition : Json) : ProgramKept :=
  match prog with
  | some pk => if pk.bytes = b then pk.under partition else programOf b partition
  | none => programOf b partition

/-- What is kept of a program after a member is what `programOf` reads of its bytes under the member's partition. -/
theorem ProgramKept.next_eq {prog : Option ProgramKept}
    (h : ∀ pk, prog = some pk → pk = programOf pk.bytes pk.partition) (b : ByteArray) (partition : Json) :
    ProgramKept.next prog b partition = programOf b partition := by
  unfold ProgramKept.next
  cases prog with
  | none => rfl
  | some pk =>
    dsimp only
    split
    · rename_i hb
      have hp : pk.parsed = Program.parse pk.bytes := congrArg ProgramKept.parsed (h pk rfl)
      have hs : pk.sha = programDigest pk.bytes := congrArg ProgramKept.sha (h pk rfl)
      subst hb
      rw [ProgramKept.under_eq, hp, hs] <;> rfl
    · rfl

/-- `Member.population` on the strata of its partition's query over its program, `strata`, read before it. -/
def Member.populationOn (m : Member) (strata : V (Array (String × Array (Nat × Nat)))) (slice : Json)
    (pubB : ByteArray) : V Unit := do
  let strata ← strata
  let h ← FilePart.header pubB
  m.heldAt strata h
  headerHeld slice h

/-- `Member.population` is `Member.populationOn` at the strata of the member's partition's query over its program's
parse. -/
theorem Member.population_on (m : Member) (slice : Json) :
    m.population slice = m.populationOn (strataOn (Program.parse m.own.program) m.own.partition) slice := by
  funext pubB
  exact (bind_assoc (Program.parse m.own.program)
    (fun p => Partition.strata p (getD m.own.partition "query")) _).symm

/-- `StepAt.ofOwn` on what is kept of the circuit file and the program: the lookup tables `tables` it is handed, the
tags its sessions run under `tags`, what its checks read of the parse `k`, and the program's digest `sha`. Every
check that read the parse reads `k`, and the three that load the public file read one load of it
(`Spot.CircuitKept.loadOf`). -/
def StepAt.ofKept (p : Own) (tags0 : Flock.Tags) (circuitB pubB : ByteArray) (tables : List (String × ByteArray))
    (rd : Reads) (population : V Unit) (tags : Flock.Tags) (k : V CircuitKept.Kept) (sha : Option String) : StepAt :=
  let pub := CircuitKept.loadOf k tags pubB
  { I := inputsAt tags0 p circuitB pubB tables, rd
    scope := CircuitKept.checkOf k pub tags true pubB 1 false rd.registered.isSome >>= fun _ => k >>= fun k => k.own
    ksFiles := CircuitKept.portsOf k pub
    ksForm := CircuitKept.filesOf k pub tags pubB
    g := k.toOption.map (·.c.g)
    population
    programSha := sha
    want := getD p.partition "program" }

/-- The scope check on the parse is the scope check on what is kept of it and the public file's one load, then the
kept verdict of `templateOk`. -/
theorem scopeAt_kept (parsed : V Flock.Circuit) (I : Flock.Witness.Inputs) (rd : Reads) :
    scopeAt parsed I rd =
      (CircuitKept.checkOf (parsed.map (CircuitKept.keptOf (Flock.Witness.zkTags I) true I.count templateOk))
          (CircuitKept.loadOf (parsed.map (CircuitKept.keptOf (Flock.Witness.zkTags I) true I.count templateOk))
            (Flock.Witness.zkTags I) I.publicFile)
          (Flock.Witness.zkTags I) true I.publicFile I.count false rd.registered.isSome >>= fun _ =>
        parsed.map (CircuitKept.keptOf (Flock.Witness.zkTags I) true I.count templateOk) >>= fun k => k.own) := by
  unfold scopeAt
  rw [CircuitKept.checkOf_kept parsed _ true _ _ false _ templateOk]
  cases parsed <;> rfl

/-- **`StepAt.ofOwn` is `StepAt.ofKept` at what `fileOf` keeps of its circuit file**: the statement on the parse is
the statement on the circuit's own verdicts and the circuit without its rows. -/
theorem StepAt.ofOwn_kept (p : Own) (tags0 : Flock.Tags) (circuitB pubB : ByteArray)
    (tables : List (String × ByteArray)) (rd : Reads) (population : V Unit) :
    StepAt.ofOwn p tags0 circuitB pubB tables rd population =
      StepAt.ofKept p tags0 circuitB pubB tables rd population (tagsAt tags0 circuitB)
        ((Flock.ProvedScope.circuitOf (tagsAt tags0 circuitB) circuitB tables).map
          (CircuitKept.keptOf (tagsAt tags0 circuitB) true 1 templateOk)) (programDigest p.program) := by
  show StepAt.mk _ _ _ _ _ _ _ _ _ = StepAt.mk _ _ _ _ _ _ _ _ _
  congr 1
  · exact scopeAt_kept _ _ _
  · exact CircuitKept.portsOf_kept _ _ _ true 1 templateOk
  · exact CircuitKept.filesOf_kept _ _ _ true 1 templateOk
  · exact CircuitKept.keptOf_g _ _ true 1 templateOk

/-- `statementOf` at a member's own lookup tables, on what the files' pass keeps of its circuit file (`fk`) and its
program's digest (`sha`), at the statement's tags `tags0`: the tables kept of the file, then `statementOf`'s checks
in its order, each read of the circuit file taken from `fk`. -/
def statementOn (tags0 : Flock.Tags) (fk : FileKept) (sha : Option String) (p : Own)
    (population : ByteArray → V Unit) (bs : Core.Blobs) : V StepAt :=
  fk.tables >>= fun tables =>
  (fk.lookups >>= tablesMatch tables) >>= fun _ =>
  match tags0.plainLeaves with
  | true => throw s!"{tags0.name}: a statement with plain SHA-512 leaves: --zk's hiding needs hm96 leaves"
  | false =>
    match p.registered.isNone && readsRegistered p.programJ with
    | true =>
      throw ("registered: the program reads registered values (a RegisteredInput among its definitions) and the " ++
        "verifier holds no registered reads: verify with its own registered file")
    | false =>
      fk.circuit >>= fun circuitB =>
      bs.get p.pub "public file" >>= fun pubB =>
      match fitsWhy WORD circuitB pubB tables (some p.program) (some p.partition) p.registered with
      | some what => throw s!"{what}: {FITS}"
      | none =>
        readsOf p.registered >>= fun rd =>
          pure (StepAt.ofKept p tags0 circuitB pubB tables rd (population pubB) fk.tags fk.kept sha)

/-- **`statementOf` at the inputs' own lookup tables (`ownTables`) is `statementOn` at what `fileOf` keeps of its
circuit file, read at the blobs' lookup tables, and its program's digest**, where the statement's tags are known. -/
theorem statementOf_on {tags0 : Flock.Tags} (ht : Flock.Tags.find? STATEMENT = some tags0) (p : Own)
    (population : ByteArray → V Unit) (bs : Core.Blobs) :
    (ownTables p bs >>= fun tables => statementOf p population tables bs) =
      statementOn tags0 (fileOf tags0 (tablesOf bs) bs p.circuit) (programDigest p.program) p population bs := by
  unfold ownTables statementOf statementOn
  simp only [StepAt.ofOwn_kept, ht]
  cases hc : bs.get p.circuit "circuit" with
  | error e =>
    rw [fileOf_error hc]
    simp only [error_bind] <;> rfl
  | ok circuitB =>
    simp only [ok_bind]
    cases hl : lookupsOf circuitB with
    | error e =>
      rw [fileOf_unread hc hl]
      simp only [error_bind] <;> rfl
    | ok names =>
      rw [fileOf_ok hc hl]
      simp only [ok_bind, pure_bind] <;> rfl

/-- `Member.files` on what the files' pass keeps of the member's circuit file (`fk`) and of its program (`pk`):
`statementOn` in place of `statementOf`, and the population check on the strata kept of the program. -/
def Member.filesOn (m : Member) (tags0 : Flock.Tags) (fk : FileKept) (pk : ProgramKept) (slice : Json)
    (bs : Core.Blobs) : V StepAt :=
  ((statementOn tags0 fk pk.sha m.own (m.populationOn pk.strata slice) bs >>= fun a =>
      m.blockHeld a.g >>= fun _ => pure a) >>=
    fun a => a.filesHeld >>= fun _ => pure a).mapError (s!"member {m.name}: {·}")

/-- `Member.files` is `Member.filesOn` at what `fileOf` keeps of its circuit file, read at the blobs' lookup tables,
and what `programOf` keeps of its program under its partition. -/
theorem Member.files_on {tags0 : Flock.Tags} (ht : Flock.Tags.find? STATEMENT = some tags0) (m : Member)
    (slice : Json) (bs : Core.Blobs) :
    m.files slice bs =
      m.filesOn tags0 (fileOf tags0 (tablesOf bs) bs m.own.circuit) (programOf m.own.program m.own.partition)
        slice bs := by
  unfold Member.files Member.statement Member.filesOn
  rw [statementOf_on ht, Member.population_on] <;> rfl

/-- **The circuit files kept for the members still to come**, each under its digest as the members name it. -/
abbrev KeptFiles := List (String × FileKept)

/-- What is kept under the digest `d`, the first there. -/
def KeptFiles.find (d : String) : KeptFiles → Option FileKept
  | [] => none
  | e :: rest => if e.1 = d then some e.2 else KeptFiles.find d rest

/-- Every file kept is what `fileOf` reads under its digest. -/
def KeptFiles.Held (tags0 : Flock.Tags) (tables : List (String × ByteArray)) (bs : Core.Blobs) (kept : KeptFiles) :
    Prop :=
  ∀ e ∈ kept, e.2 = fileOf tags0 tables bs e.1

/-- What `KeptFiles.find` returns under a digest is what `fileOf` reads under it. -/
theorem KeptFiles.find_held {tags0 : Flock.Tags} {tables : List (String × ByteArray)} {bs : Core.Blobs} {d : String}
    {fk : FileKept} : ∀ {kept : KeptFiles}, kept.Held tags0 tables bs → kept.find d = some fk →
      fk = fileOf tags0 tables bs d
  | [], _, h => by cases h
  | e :: rest, hk, h => by
    unfold KeptFiles.find at h
    split at h
    · rename_i he
      cases h
      rw [← he]
      exact hk e List.mem_cons_self
    · exact KeptFiles.find_held (fun e' he' => hk e' (List.mem_cons_of_mem _ he')) h

/-- `membersFiles` on the circuit files `kept` and the program `prog` the member before read: each member's files
held on what is kept under its circuit's digest, which is read under it (`fileOf`) for the first member that names
it and kept for the members still to come, and on what is kept of its program under its partition
(`ProgramKept.next`). A circuit file is found, read and parsed once in the pass, at the tables of the blobs' lookup
tables `tables` it looks up, which are read once before the pass, and a program is parsed and hashed, and its
partition's query evaluated on it, once for the members in a row that read it under the partition the parameters give
them all. -/
def keptFiles (slice : Member → Json) (bs : Core.Blobs) (tags0 : Flock.Tags) (tables : List (String × ByteArray)) :
    KeptFiles → Option ProgramKept → List Member → V (List (String × StepAt))
  | _, _, [] => pure []
  | kept, prog, m :: ms =>
    let pk := ProgramKept.next prog m.own.program m.own.partition
    match kept.find m.own.circuit with
    | some fk =>
      m.filesOn tags0 fk pk (slice m) bs >>= fun a =>
        keptFiles slice bs tags0 tables kept (some pk) ms >>= fun as => pure ((m.name, a) :: as)
    | none =>
      let fk := fileOf tags0 tables bs m.own.circuit
      m.filesOn tags0 fk pk (slice m) bs >>= fun a =>
        keptFiles slice bs tags0 tables ((m.own.circuit, fk) :: kept) (some pk) ms >>= fun as =>
          pure ((m.name, a) :: as)

/-- `keptFiles` on files each of which is what `fileOf` reads under its digest, and a program that is what
`programOf` reads of its bytes under its partition, is `membersFiles`. -/
theorem keptFiles_eq {tags0 : Flock.Tags} (ht : Flock.Tags.find? STATEMENT = some tags0) (slice : Member → Json)
    (bs : Core.Blobs) : ∀ (ms : List Member) (kept : KeptFiles) (prog : Option ProgramKept),
      kept.Held tags0 (tablesOf bs) bs → (∀ pk, prog = some pk → pk = programOf pk.bytes pk.partition) →
      keptFiles slice bs tags0 (tablesOf bs) kept prog ms = membersFiles slice bs ms
  | [], _, _, _, _ => rfl
  | m :: ms, kept, prog, hk, hp => by
    have hnext : ∀ pk, some (programOf m.own.program m.own.partition) = some pk →
        pk = programOf pk.bytes pk.partition := by
      intro pk h
      cases h
      rfl
    rw [keptFiles, membersFiles, Member.files_on ht, ProgramKept.next_eq hp]
    cases hf : kept.find m.own.circuit with
    | some fk =>
      have hfk := KeptFiles.find_held hk hf
      subst hfk
      dsimp only
      rw [keptFiles_eq ht slice bs ms kept _ hk hnext]
    | none =>
      dsimp only
      rw [keptFiles_eq ht slice bs ms _ _ (fun e he => by
        rcases List.mem_cons.1 he with rfl | he
        · rfl
        · exact hk e he) hnext]

/-- **Every member's files, a circuit file read and parsed once for the members that name it**, and the program's
strata evaluated once for the members that read it: `keptFiles` from no file and no program kept, at the statement's
tags and the blobs' lookup tables, each read once. It returns what `membersFiles` returns, a refusal included
(`membersFiles_eq_kept`), and is compiled in its place. -/
def membersFilesKept (slice : Member → Json) (bs : Core.Blobs) (ms : List Member) : V (List (String × StepAt)) :=
  match Flock.Tags.find? STATEMENT with
  | some tags0 => keptFiles slice bs tags0 (tablesOf bs) [] none ms
  | none => membersFiles slice bs ms

/-- **The files' pass with each circuit file read and parsed once is the files' pass**: `membersFilesKept` returns
exactly what `membersFiles` returns, a statement for each member or the first refusal, so every statement about
`membersFiles` is one about the compiled pass. -/
@[csimp] theorem membersFiles_eq_kept : @membersFiles = @membersFilesKept := by
  funext slice bs ms
  unfold membersFilesKept
  cases ht : Flock.Tags.find? STATEMENT with
  | none => rfl
  | some tags0 => exact (keptFiles_eq ht slice bs ms [] none (fun _ h => nomatch h) (fun _ h => nomatch h)).symm

end Spot.OneStageZk
