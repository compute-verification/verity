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
* **blobs**: the files those digests name (`Core.Blobs`). The registered circuit names the lookup tables its session
  reads (its META's `lookups`, `lookupsOf`), so blobs whose pinned MUFU tables (`tablesOf`, each under its SHA-512 as
  `flock-verify --archive` finds one) are other than those refuse (`tablesHeld`, `NO_TABLES`) before the session is set
  up: the tables the session `run` sets up reads are the registered circuit's;
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
  program, the partition object, the registered reads or a session's sample at `2^64` bytes or more (`fitsWhy`,
  `drawFitsWhy`, the JSON ones as `Flock.RecordText.printT` prints them);
* **per member** (`checkMembersAt`): the form for a population of several members, a statement a sampled member. It
  accepts nothing (`MULTI_REFUSED`, `checkMembersAt_refuses`) until the guarantee for one sample over several
  members' statements is in the lock, and `protocol` does not read it: `paramsOfJson` and `evidenceOf` refuse
  `proofs` of other keys than the one statement's, so their parameters name no members (`paramsOfJson_members`),
  their evidence no statements (`evidenceOf_statements`), and the check at the coins (`checkBlobsAt`, which `live`
  runs) is the one statement's (`checkOneAt`: `checkBlobsAt_parsed`, `live_checkAt_parsed`). The form's own parsers
  read it (`membersParamsOfJson`, `membersEvidenceOf`; `paramsEitherOfJson` and `evidenceEitherOf` read either
  form), and on parameters that name members with evidence that names statements `checkBlobsAt` is
  `checkMembersAt`; parameters of one form with evidence of the other reject. Its parameters' `proofs` is exactly
  `{"members"}` (`membersOf`), an object from each member's name to exactly
  `{"circuit", "public", "registered", "units", "vus_per_block"}`: its
  statement's files, as the one statement's, its units as `[lo, hi)` ranges, ascending, no unit in two members', and
  its circuit's units a proof block. Its evidence's `proofs` is exactly `{"statements"}` (`statementsOf`), a list of
  exactly `{"member", "sessions"}`, the statements' sessions in order being v1's `sessions`. The statements are held as
  public structure before any session is read (`membersHeld`): each names a member of the parameters that no earlier
  statement named, in the members' order (by first unit), with exactly one session, on live coins and never retried;
  an undrawn member has no statement. Each session's coins are one tree's, as the one statement's are, and its tree's
  last opening is its closing round (`Challenger.memberTreeReason`): one more `session` round, of no coins, opened after
  the SHA-512 of the session's evidence (`sessionEvidence`: its record's bytes, then each rep file's name and bytes,
  each under an 8-byte length, `evidenceBytes`, which `evidenceParts` decodes), so the files the verdict reads are
  fixed before the next member's coins; and the log opens the trees one after another in the sessions' order, no round
  of a tree before the closing round of the one before it or after its own (`Challenger.membersOrder`). Then each
  statement's session is verified as the one statement's is, on its member's own files (`Member.statement`,
  `runMembers`), after the statements before it, so no coin tree serves two statements: its circuit's units a proof
  block must be the ones the parameters name, and its verdict holds only where its strata are the registration's, each
  restricted to its units (`Member.sizes`: the parameters fix each stratum's size in a member, and nothing the prover
  sends moves a unit between strata), and its public file's units are exactly its member's, in order
  (`Member.population`), so every unit its session proves is its member's; the units the statements proved must be the
  evidence's `proved_units`. The registration is held to the members before any statement's session: the members'
  units are its population, each member one range of it (`laidHeld`), and each of its roots is exactly one member's,
  the member whose name, then `/`, the root's port begins with (`rootsOf`; a root that is no member's or two members'
  refuses, and so does a window's row tables' root, `ROW_TABLES`). A member's public file is held to its slice of the
  registration (`Member.slice`, `headerHeld`): the window's scheme, the member's roots and their domains under their
  bare ports, and the member's count of units as its instances. A member may read registered values, with its own
  registered file as the one statement's, and a value is one tree for every member (`classHeld`): two members' ports of
  one value that differ in its root, leaves, row width, domain or leaf schema refuse, whether or not either is sampled.

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
public import Std.Data.HashMap
meta import CoreProtocol.Blobs
meta import Spot.OneStage
meta import Spot.ChallengerTrees
meta import Spot.Slots
meta import Spot.FilePart
meta import Flock
meta import Flock.JsonParse
meta import Std.Data.HashMap

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
  pure { base, own := { programJ, program := (← Flock.canon programJ).toUTF8, partition, circuit, pub, registered },
         members := none }

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

/-- `tablesHeld` on a circuit file whose META is `{"lookups": lookups}` (`{}` where `j` has no `lookups`) and blobs
under the digests `digests` (lowercase hex), whatever their bytes: the `tables_held` piece,
`{"lookups": ..., "digests": [hex, ...]}` to `{"why": refusal or null}`. -/
def tablesPiece (j : Json) : V Json := do
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
  return Json.mkObj [("why", match tablesHeld own ⟨#[(d, c)] ++ entries⟩ with
    | .ok () => Json.null
    | .error w => Json.str w)]

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

/-- The refusal of the per-member form's acceptance: no guarantee in the lock yet covers one sample over several
members' statements. -/
def MULTI_REFUSED : String :=
  "the per-member form (proofs.members) is refused until the guarantee for a sample over several members' " ++
    "statements under one draw is in the lock: verify one statement (proofs {circuit, public, registered})"

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
(`paramsOfJson`, read at `proofs` of no file): `own` is that program and partition, with no file. -/
def membersParamsOfJson (j : Json) : V Params := do
  let some pr := field? j "proofs" | throw "the parameters name no proofs (the members and their statements' files)"
  exactly pr #["members"] "proofs"
  let one := Json.mkObj [("circuit", Json.str NO_FILE), ("public", Json.str NO_FILE), ("registered", Json.null)]
  let p ← paramsOfJson (Json.mkObj (without j "proofs" ++ [("proofs", one)]))
  let own := { p.own with circuit := "", pub := "" }
  pure { p with own, members := some (← membersOf own (getD pr "members")) }

/-- **The parameters in either form**: the per-member form's where `proofs` is exactly `{"members"}`, else the one
statement's. `protocol` reads `paramsOfJson`, which refuses the per-member form's. -/
def paramsEitherOfJson (j : Json) : V Params :=
  if (field? j "proofs").bind keys? == some #["members"] then membersParamsOfJson j else paramsOfJson j

/-- **The per-member form's evidence**: `proofs` exactly `{"statements"}` (`statementsOf`), and everything else as the
one statement's evidence holds it (`evidenceOf`, read at `proofs` of the statements' sessions in order), so its
sessions are the statements', in v1's `sessions`' order. -/
def membersEvidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  let some pr := field? j "proofs" | throw "the evidence names no proofs (the statements' sessions' files)"
  let statements ← statementsOf pr
  let sessions := match getD pr "statements" with
    | .arr ss => ss.flatMap fun s => match getD s "sessions" with | .arr rs => rs | _ => #[]
    | _ => #[]
  let one := Json.mkObj (without j "proofs" ++ [("proofs", Json.mkObj [("sessions", Json.arr sessions)])])
  let e ← evidenceOf (← Flock.canon one).toUTF8
  pure { e with statements := some statements }

/-- **The evidence in either form**: the per-member form's where `proofs` is exactly `{"statements"}`, else the one
statement's. `protocol` reads `evidenceOf`, which refuses the per-member form's. -/
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
an undrawn member has no statement). The members so far, with its own. -/
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

/-- The refusal of members that are not the registration's population, a range each (`laidHeld`). -/
def MEMBERS_LAID : String :=
  "members: the members' units are not the registration's population, each member's one range of it: a member is " ++
    "the units [base, base + n), and the members' counts sum to the population"

/-- **The members laid over the registration's population**: each member's units are one range, within the record's
population, and the members' counts sum to it. The parameters' members share no unit (`membersOf`), so every unit of
the population is exactly one member's, and the members in the order of their first units lie one after another from
unit 0: a member's units are `[base, base + n)`, `base` the count of the members before it. -/
def laidHeld (ms : Array Member) (rec : Json) : V Unit :=
  match nat? (getD rec "population") with
  | none => throw "the record names no population"
  | some n =>
    if ms.all (fun m => m.units.size == 1 && m.units.all (·.2 ≤ n)) && ms.foldl (fun a m => a + m.count) 0 == n then
      pure ()
    else throw MEMBERS_LAID

/-- The refusal of two members' ports of one registered value that are not one tree's (`classHeld`): `diff` is the
first field they differ in (`Flock.RecVStar.portDiff?`). -/
def classTwo (first second value diff : String) : String :=
  s!"registered: members {first} and {second} read the registered value {value} as two ({diff}): a value the " ++
    "window registers is one tree, read by every member at one root"

/-- The member `name`'s registered port `q` after the ports before it: `acc` holds, for each value read so far, the
first port of it and its member's name. A port of a value `acc` holds must be that port in the value's root, leaves,
row width, domain and leaf schema (`Flock.RecVStar.portDiff?`, without the units' reads). -/
def classPort (name : String) (acc : Std.HashMap String (String × Flock.Registered.Port))
    (q : Flock.Registered.Port) : V (Std.HashMap String (String × Flock.Registered.Port)) :=
  match acc.get? q.value with
  | none => pure (acc.insert q.value (name, q))
  | some (first, q0) =>
    match Flock.RecVStar.portDiff? q0 q false with
    | none => pure acc
    | some diff => throw (classTwo first name q.value diff)

/-- One member's registered ports after the members before it (`classPort` on each); a member whose registered file
is no registered reads refuses, and one with none reads nothing. -/
def classStep (acc : Std.HashMap String (String × Flock.Registered.Port)) (m : Member) :
    V (Std.HashMap String (String × Flock.Registered.Port)) :=
  match m.own.registered with
  | none => pure acc
  | some j =>
    (Flock.Registered.ofJson j).mapError (s!"member {m.name}: {·}") >>= fun own => own.foldlM (classPort m.name) acc

/-- **A registered value is one tree for every member**: over the parameters' members, sampled or not, every port of
one value names one root, one count of leaves, one row width, one domain and one leaf schema (`classStep`), so the
values the members read are a function of the parameters, each with one root. It is one pass over the parameters'
members, each member's registered file read once, and it reads no blob. -/
def classHeld (ms : Array Member) : V Unit :=
  ms.foldlM classStep {} >>= fun _ => pure ()

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

/-- A member's circuit held to the parameters' units a proof block: a circuit file that parses with another
`vus_per_block` refuses. One that does not parse is its session's reject. -/
def Member.blockHeld (m : Member) (parsed : V Flock.Circuit) : V Unit :=
  match parsed with
  | .ok c =>
    if c.g != m.g then
      throw s!"vus_per_block: the member's circuit has {c.g} units a proof block, not the {m.g} the parameters name"
    else pure ()
  | .error _ => pure ()

/-- The scope check a session's verdict runs at the inputs `I` and the registered reads `rd`, as `run` builds it. -/
def scopeOf (I : Flock.Witness.Inputs) (rd : Reads) : V Unit := do
  let tags := Flock.Witness.zkTags I
  Flock.ProvedScope.checkOf (Flock.ProvedScope.circuitOf tags I.circuitFile I.tables) tags true I.publicFile I.count
    false rd.registered.isSome
  templateOk (← Flock.ProvedScope.circuitOf tags I.circuitFile I.tables)

/-- **What `step` reads of one statement**: its inputs, the verifier's registered reads, the scope, files and
population checks its sessions' verdicts run, and its program's digest against the partition object's program. Nothing
of it reads a session's record, coins or sample. -/
structure StepAt where
  I : Flock.Witness.Inputs
  rd : Reads
  scope : V Unit
  ksFiles : V Unit
  population : V Unit
  programSha : Option String
  want : Json

/-- One session's `step` at the statement `a`, on the blobs `bs`, after the run `r`. -/
def StepAt.step (a : StepAt) (bs : Core.Blobs) (r : Run) (s : RunFiles) : V Run :=
  OneStageZk.step a.I a.rd bs a.scope a.ksFiles a.population a.programSha a.want r s

/-- The statement at the verifier's own inputs `p`, the tags `tags0`, the circuit and public files, the blobs' lookup
tables and the registered reads `rd`, as `run` builds its own, with `population` its population check. -/
def StepAt.ofOwn (p : Own) (tags0 : Flock.Tags) (circuitB pubB : ByteArray) (bs : Core.Blobs) (rd : Reads)
    (population : V Unit) : StepAt :=
  let I := inputsOf tags0 p circuitB pubB bs
  { I, rd, scope := scopeOf I rd
    ksFiles := Flock.PortRoots.checkOf (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables)
      (Flock.Witness.zkTags I) pubB
    population
    programSha := programDigest p.program
    want := getD p.partition "program" }

/-- **The statement the sessions at the inputs `p` are held to, read off the blobs before any session**: `run`'s own
checks in `run`'s order (the blobs' lookup tables the circuit's, the statement's tags with hiding leaves, registered
values read only with the verifier's registered file, the circuit and public files, each length within a statement's
word, the registered reads), then `StepAt.ofOwn` with the population check `population` on the public file. A
refusal where `run` refuses. -/
def statementOf (p : Own) (population : ByteArray → V Unit) (bs : Core.Blobs) : V StepAt :=
  tablesHeld p bs >>= fun _ =>
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
        match fitsWhy WORD circuitB pubB (tablesOf bs) (some p.program) (some p.partition) p.registered with
        | some what => throw s!"{what}: {FITS}"
        | none =>
          readsOf p.registered >>= fun rd => pure (StepAt.ofOwn p tags0 circuitB pubB bs rd (population pubB))

/-- **The statement a member's session is held to**: `statementOf` at the member's own inputs, its population check the
member's at its `slice` of the registration (`Member.population`), and its circuit's units a proof block the
parameters' (`Member.blockHeld`). -/
def Member.statement (m : Member) (slice : Json) (bs : Core.Blobs) : V StepAt :=
  statementOf m.own (m.population slice) bs >>= fun a =>
  m.blockHeld (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags a.I) a.I.circuitFile a.I.tables) >>= fun _ => pure a

/-- **One statement's sessions, after the sessions the run `r` holds**: the statement `st`, read before any session,
then `step` on each session in order. -/
def runFrom (st : V StepAt) (sessions : Array RunFiles) (bs : Core.Blobs) (r : Run) : V Run :=
  st >>= fun a => sessions.foldlM (fun r s => a.step bs r s) r

/-- One statement's run after the run `r`: its sessions at its member's statement, at the member's `slice` of the
registration (`runFrom`), a refusal naming the member. -/
def stepMember (slice : Member → Json) (bs : Core.Blobs) (r : Run) (z : StatementFiles × Member) : V Run :=
  (runFrom (z.2.statement (slice z.2) bs) z.1.sessions bs r).mapError (s!"member {z.2.name}: {·}")

/-- **The per-member form's run**, at each member's `slice` of the registration: the statements' structure held
(`membersHeld`), then each statement at its member's statement after the statements before it (`stepMember`), so
`step`'s coin-tree keys are fresh across statements; then, where every session accepted, the union of the statements'
proved units held to the evidence's `want` (`provedHeld`). Each sampled member's files are read once; a member no
statement names is never read. -/
def runMembers (slice : Member → Json) (ms : Array Member) (sts : Array StatementFiles) (want : Array Nat)
    (bs : Core.Blobs) : V Run :=
  membersHeld ms sts >>= fun held =>
  (sts.zip held).foldlM (stepMember slice bs) ⟨#[], #[], #[], true⟩ >>= fun r =>
  (match r.all with
    | true => provedHeld r.proved want
    | false => pure ()) >>= fun _ => pure r

/-- **The bytes a closing round's digest is of**: the number of `parts` in an 8-byte big-endian word, then each
part's length in such a word and its bytes, as a frame holds its parts (`Flock.FV3.input`). `evidenceParts` reads the
parts back, so the digest fixes each part's bytes, not a digest of them. -/
def evidenceBytes (parts : Array ByteArray) : ByteArray :=
  parts.foldl (fun acc p => acc ++ Flock.be64 p.size.toUInt64 ++ p) (Flock.be64 parts.size.toUInt64)

/-- `n` parts of `b` from offset `o` to its end, each an 8-byte length and its bytes; none where `b` is not that. -/
def partsFrom (b : ByteArray) : Nat → Nat → Option (List ByteArray)
  | 0, o => if o == b.size then some [] else none
  | n + 1, o =>
    if o + 8 ≤ b.size then
      let stop := o + 8 + Flock.Draw.beNat b o 8
      if stop ≤ b.size then (partsFrom b n stop).map (b.extract (o + 8) stop :: ·) else none
    else none

/-- **The decoder of `evidenceBytes`**: the parts `b` holds, none where `b` is not some parts' bytes. -/
def evidenceParts (b : ByteArray) : Option (Array ByteArray) :=
  if 8 ≤ b.size then (partsFrom b (Flock.Draw.beNat b 0 8) 8).map (·.toArray) else none

/-- The refusal of a session's evidence `evidenceBytes` can't hold. -/
def EVIDENCE_FITS : String :=
  "its evidence holds a file of 2^64 bytes or more, or as many files, past the 8-byte word its closing round's " ++
    "digest holds each length in"

/-- **A session's evidence, the bytes its closing round is opened after the SHA-512 of** (`evidenceBytes`): its
record's bytes, then, for each rep file it names, in the order its `reps` holds them (their names', sorted), the file's
name in UTF-8 and its bytes, each the blob the verdict reads it from (`step`, `filesOf`). A refusal where no blob holds
one, as `step` refuses a record, or where a length is past a word (`EVIDENCE_FITS`). -/
def sessionEvidence (bs : Core.Blobs) (s : RunFiles) : V ByteArray := do
  let record ← bs.get s.record s!"session {s.name}'s record"
  let reps ← s.reps.mapM fun (f, d) => do
    let b ← bs.get d s!"session {s.name}'s files: {f}"
    pure #[f.toUTF8, b]
  let parts := #[record] ++ reps.flatten
  if decide (WORD ≤ parts.size) || parts.any (fun part => decide (WORD ≤ part.size)) then
    throw s!"session {s.name}: {EVIDENCE_FITS}"
  pure (evidenceBytes parts)

/-- The statements' sessions held to the challenger's trees `ts` and its release log's lines `log`, each statement
with its member: each session's record on the blobs, refused as `treesHeld` refuses it, at its tables with the
`root_B` lengths its statement fixes on its member's own files (`rootBTables`) and the SHA-512 of its evidence
(`sessionEvidence`), then `Challenger.membersTreeReason`: each session's coins one tree's, the tree closed after that
digest, and the trees opened one after another in the sessions' order. -/
def membersTreesHeld (ts : Array Challenger.LoggedTree) (log : List String) (sts : Array (StatementFiles × Member))
    (bs : Core.Blobs) : V Unit := do
  let mut named : Array (String × Array (String × Option Nat) × Json × String) := #[]
  for (st, m) in sts do
    for s in st.sessions do
      let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
        | throw s!"session {s.name}'s record is not UTF-8"
      let evidence ← sessionEvidence bs s
      named := named.push (s.name, rootBTables m.own bs record,
        (Flock.JsonParse.parse record).toOption.getD Json.null, sha512Hex evidence)
  if let some why := Challenger.membersTreeReason ts log named then throw why

/-- **The per-member form's sessions' verdicts** on the parameters `p`, its members `ms`, the registration `c`, the
evidence `e` and its statements `sts`: the statements' structure (`membersHeld`), their sessions v1's in order, the
sessions' coins held to the trees the challenger logged in the order it must release them, each tree closed after its
session's evidence before the next is opened (`treesOf`, `orderHeld`, `membersTreesHeld`), the registration's
population held to the program (`popHeld`) and the members laid over it (`laidHeld`), each registered value the
members read one tree (`classHeld`), the registration's roots each one member's (`rootsOf`), then each statement's
session on the blobs, at its member's slice of the registration (`Member.slice`, `runMembers`). -/
def membersRun (p : Params) (ms : Array Member) (c : OneStage.Commitment) (e : Evidence) (sts : Array StatementFiles)
    (bs : Core.Blobs) : V Run := do
  let held ← membersHeld ms sts
  if (sts.flatMap (·.sessions)).map (·.name) != e.base.sessions then
    throw "the statements' sessions are not the evidence's sessions, in order"
  let ts ← treesOf p.base e.base
  orderHeld p c e
  membersTreesHeld ts (← linesOf e) (sts.zip held) bs
  popHeld p.base.verifier c.record
  laidHeld ms c.record
  classHeld ms
  let roots ← rootsOf ms c.record
  runMembers (fun m => m.slice c.record roots) ms sts e.base.provedUnits bs

/-- **The per-member form's check at the coins `ξ`** (module docstring, **per member**): the statements' sessions'
verdicts (`membersRun`), then v1's `checkAt` at `ξ` on them, as the one statement's check; and what would accept is
refused (`MULTI_REFUSED`). -/
def checkMembersAt (p : Params) (ms : Array Member) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence)
    (sts : Array StatementFiles) (bs : Core.Blobs) : Core.Outcome OneStage.Public Core.Integrity :=
  match membersRun p ms c e sts bs with
  | .error err => .reject s!"proofs: {err}"
  | .ok r =>
    match OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
    | .accept _ _ =>
      .reject (if r.all then s!"proofs: {MULTI_REFUSED}" else "proofs: a session's verifier did not accept")
    | .reject why => .reject why
    | .undecided why => .undecided why

/-- **The per-member form accepts nothing**: whatever the parameters, the registration, the coins, the evidence and
the blobs. -/
theorem checkMembersAt_refuses (p : Params) (ms : Array Member) (c : OneStage.Commitment) (ξ : ByteArray)
    (e : Evidence) (sts : Array StatementFiles) (bs : Core.Blobs) (pub : OneStage.Public) (prof : Core.Integrity) :
    checkMembersAt p ms c ξ e sts bs ≠ .accept pub prof := by
  intro h
  unfold checkMembersAt at h
  split at h
  · cases h
  · split at h <;> cases h

/-- The refusal of parameters and evidence of two forms (`checkBlobsAt`). -/
def FORMS : String :=
  "proofs: the parameters' proofs and the evidence's are not of one form: one statement's files and its sessions, " ++
    "or the members and their statements"

/-- **The check at the coins `ξ`**, by the form of the parameters' `proofs` and the evidence's: the one statement's
(`checkOneAt`), or the per-member form's (`checkMembersAt`, which accepts nothing); parameters of one form with
evidence of the other reject (`FORMS`). -/
def checkBlobsAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match p.members, e.statements with
  | none, none => checkOneAt p c ξ e bs
  | some ms, some sts => checkMembersAt p ms c ξ e sts bs
  | _, _ => .reject FORMS

/-- **What `checkBlobsAt` accepts, the one statement's check accepts**, on parameters and evidence of the one
statement's form: the per-member form accepts nothing (`checkMembersAt_refuses`). -/
theorem checkBlobsAt_accept {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : checkBlobsAt p c ξ e bs = .accept pub prof) :
    checkOneAt p c ξ e bs = .accept pub prof ∧ p.members = none ∧ e.statements = none := by
  unfold checkBlobsAt at h
  split at h
  · rename_i hm hs
    exact ⟨h, hm, hs⟩
  · exact absurd h (checkMembersAt_refuses _ _ _ _ _ _ _ _ _)
  · cases h

/-- On parameters that name no members and evidence that names no statements, the check is the one statement's. -/
theorem checkBlobsAt_one {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    (hm : p.members = none) (hs : e.statements = none) : checkBlobsAt p c ξ e bs = checkOneAt p c ξ e bs := by
  unfold checkBlobsAt
  simp only [hm, hs]

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

/-- **Parsed parameters name no members**: whatever `paramsOfJson` returns is of the one statement's form. -/
theorem paramsOfJson_members {j : Json} {p : Params} (h : paramsOfJson j = .ok p) : p.members = none := by
  unfold paramsOfJson at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  all_goals try (rename_i ht _ _ _ _; simp [throw, throwThe, MonadExceptOf.throw] at ht; done)
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rfl

/-- **Parsed evidence names no statements**: whatever `evidenceOf` returns is of the one statement's form. -/
theorem evidenceOf_statements {b : ByteArray} {e : Evidence} (h : evidenceOf b = .ok e) : e.statements = none := by
  unfold evidenceOf at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rfl

/-- **On what `protocol` parses, the check at the coins is the one statement's**: `protocol` reads its parameters by
`paramsOfJson` and its evidence by `evidenceOf`, which return the one statement's form alone (`paramsOfJson_members`,
`evidenceOf_statements`), where `checkBlobsAt` is `checkOneAt` (`checkBlobsAt_one`). So `checkMembersAt` runs on no
input `protocol` parses. -/
theorem checkBlobsAt_parsed {j : Json} {b : ByteArray} {p : Params} {e : Evidence} (hp : paramsOfJson j = .ok p)
    (he : evidenceOf b = .ok e) (c : OneStage.Commitment) (ξ : ByteArray) (bs : Core.Blobs) :
    checkBlobsAt p c ξ e bs = checkOneAt p c ξ e bs :=
  checkBlobsAt_one (paramsOfJson_members hp) (evidenceOf_statements he)

/-- **`live`'s check at the coins is the one statement's on what `protocol` parses** (`checkBlobsAt_parsed`), on any
blobs. -/
theorem live_checkAt_parsed (bs : Array ByteArray) {j : Json} {b : ByteArray} {p : Params} {e : Evidence}
    (hp : paramsOfJson j = .ok p) (he : evidenceOf b = .ok e) (c : OneStage.Commitment) (ξ : ByteArray) :
    (live bs).checkAt p c ξ e = checkOneAt p c ξ e (Core.Blobs.ofArray bs) :=
  checkBlobsAt_parsed hp he c ξ (Core.Blobs.ofArray bs)

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

/-- A circuit file whose META looks up the MUFU tables `names`. -/
private def lookingUp (names : List String) : ByteArray :=
  s!"circuit\nMETA {(Json.mkObj [("lookups", Json.mkObj (names.map (·, Json.mkObj [])))]).compress}\nrest".toUTF8

/-- The verifier's own files naming the circuit file `c` and no public file. -/
private def ownOf (c : ByteArray) : Own :=
  { programJ := Json.mkObj [("definitions", Json.mkObj [])], program := .empty, partition := Json.null,
    circuit := Flock.hexOf (Flock.Sha512.hash c), pub := "", registered := none }

/-- Blobs holding the circuit file `c` and, under their pinned SHA-512s whatever their bytes, the MUFU tables `names`. -/
private def withTables (c : ByteArray) (names : List String) : Core.Blobs :=
  ⟨#[(Flock.Sha512.hash c, c)] ++ (Flock.MUFU_TABLES.filter fun t =>
    names.contains t.name).toArray.map fun t => ((Flock.unhex t.sha512).toOption.getD .empty, .empty)⟩

private def held (lookups tables : List String) : String :=
  refusal (tablesHeld (ownOf (lookingUp lookups)) (withTables (lookingUp lookups) tables))

#guard held [] [] == ""
#guard held ["ex2", "rcp"] ["rcp", "ex2"] == ""
#guard held [] ["rcp"] == s!"{NO_TABLES}: the circuit looks up none, the blobs hold rcp"
#guard held ["ex2"] ["ex2", "rsq", "sqrt", "rcp"] == s!"{NO_TABLES}: the circuit looks up ex2, the blobs hold rsq, sqrt, rcp, ex2"
#guard held ["ex2", "rcp"] ["ex2"] == s!"{NO_TABLES}: the circuit looks up rcp, ex2, the blobs hold ex2"
#guard (lookupsOf "circuit\nMETA {\"lookups\":[]}\n".toUTF8).toOption == none
#guard (lookupsOf "circuit\nMETA {\"lookups\":null}\n".toUTF8).toOption == some []
#guard (lookupsOf "circuit\nMETA {\"lookups\":{\"foo\":{},\"ex2\":{}}}\n".toUTF8).toOption == some ["ex2"]
#guard (refusal (tablesHeld (ownOf (lookingUp [])) ⟨#[]⟩)).startsWith "circuit: "
#guard refusal ((fun _ => ()) <$> run (ownOf (lookingUp [])) #[] none (withTables (lookingUp []) ["rcp"])) ==
  s!"{NO_TABLES}: the circuit looks up none, the blobs hold rcp"
#guard (refusal ((fun _ => ()) <$> run (ownOf (lookingUp ["rcp"])) #[] none (withTables (lookingUp ["rcp"]) ["rcp"]))).startsWith
  "public file: "
#guard (tablesPiece (Json.mkObj [("lookups", Json.mkObj [("rcp", Json.mkObj [])]), ("digests", Json.arr #[Json.str
  "8a317adbcc078bf70ae01959b679d49f97cd9c8a93d6f39b7325c93d0fd550fa4396a4c3a0568d2387c2ff2f33eb76453b33a7b8890d97752d86cc1d971aeacc",
  Json.str "00"])])).toOption == some (Json.mkObj [("why", Json.null)])
#guard (tablesPiece (Json.mkObj [("digests", Json.arr #[Json.str
  "8a317adbcc078bf70ae01959b679d49f97cd9c8a93d6f39b7325c93d0fd550fa4396a4c3a0568d2387c2ff2f33eb76453b33a7b8890d97752d86cc1d971aeacc"])])).toOption ==
  some (Json.mkObj [("why", Json.str s!"{NO_TABLES}: the circuit looks up none, the blobs hold rcp")])
#guard !(tablesPiece (Json.mkObj [("digests", Json.arr #[Json.str "0"])])).toBool

#guard (rootBLen { (default : Flock.Setup) with m := 25, merkle := Flock.MerkleScheme.hm96Sha512 }).toOption == some 16384
#guard (rootBLen { (default : Flock.Setup) with m := 26, merkle := Flock.MerkleScheme.hm96Sha512 }).toOption == some 8192

/-- The program and the partition the members' statements share, naming no file. -/
private def sharedT : Own :=
  { programJ := Json.null, program := .empty, partition := Json.null, circuit := "", pub := "", registered := none }

private def memberJ (units : List (Nat × Nat)) : Json :=
  Json.mkObj [("circuit", Json.str ("".pushn 'c' 128)), ("public", Json.str ("".pushn 'd' 128)),
    ("registered", Json.null),
    ("units", Json.arr (units.toArray.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])),
    ("vus_per_block", Lean.toJson (1 : Nat))]

#guard ((membersOf sharedT (Json.mkObj [("a", memberJ [(0, 4)]), ("b", memberJ [(4, 6), (8, 9)])])).toOption.map
  (·.map fun m => (m.name, m.units, m.g, m.own.circuit.length, m.own.pub.length))) ==
  some #[("a", #[(0, 4)], 1, 128, 128), ("b", #[(4, 6), (8, 9)], 1, 128, 128)]
#guard (membersOf sharedT (Json.mkObj [("a", memberJ [(0, 4)]), ("b", memberJ [(3, 6)])])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("a", memberJ [(0, 4), (2, 3)])])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("a", memberJ [(5, 6), (0, 1)])])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("a", memberJ [])])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("", memberJ [(0, 4)])])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("a", (memberJ [(0, 4)]).setObjVal! "vus_per_block"
  (Lean.toJson (0 : Nat)))])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("a", (memberJ [(0, 4)]).setObjVal! "interface" Json.null)])).toOption.isNone
#guard (statementsOf (Json.mkObj [("statements", Json.arr #[Json.mkObj [("member", Json.str "a"),
  ("sessions", Json.arr #[])]])])).toOption.map (·.map (·.member)) == some #[some "a"]
#guard (statementsOf (Json.mkObj [("statements", Json.arr #[Json.mkObj [("member", Json.str ""),
  ("sessions", Json.arr #[])]])])).toOption.isNone
#guard (statementsOf (Json.mkObj [("sessions", Json.arr #[])])).toOption.isNone

private def paramsWhy (r : V Params) : String :=
  match r with
  | .error e => e
  | .ok _ => ""

private def evidenceWhy (r : V Evidence) : String :=
  match r with
  | .error e => e
  | .ok _ => ""

/-- Parameters of nothing but `proofs`: each parser refuses them, the one statement's past its `proofs`. -/
private def proofsOnly (pr : Json) : Json := Json.mkObj [("proofs", pr)]

private def oneProofs : Json :=
  Json.mkObj [("circuit", Json.str NO_FILE), ("public", Json.str NO_FILE), ("registered", Json.null)]

#guard paramsWhy (paramsOfJson (proofsOnly (Json.mkObj [("members", Json.mkObj [])]))) ==
  "the keys of proofs are not circuit, public, registered"
#guard paramsWhy (membersParamsOfJson (proofsOnly oneProofs)) == "the keys of proofs are not members"
#guard paramsWhy (paramsOfJson (proofsOnly oneProofs)) != ""
#guard paramsWhy (paramsOfJson (proofsOnly oneProofs)) != "the keys of proofs are not circuit, public, registered"
#guard paramsWhy (membersParamsOfJson (proofsOnly (Json.mkObj [("members", Json.mkObj [])]))) ==
  paramsWhy (paramsOfJson (proofsOnly oneProofs))
#guard paramsWhy (paramsEitherOfJson (proofsOnly (Json.mkObj [("members", Json.mkObj [])]))) ==
  paramsWhy (paramsOfJson (proofsOnly oneProofs))
#guard paramsWhy (paramsEitherOfJson (proofsOnly oneProofs)) == paramsWhy (paramsOfJson (proofsOnly oneProofs))
#guard paramsWhy (paramsEitherOfJson (proofsOnly (Json.mkObj [("statements", Json.arr #[])]))) ==
  "the keys of proofs are not circuit, public, registered"
#guard evidenceWhy (membersEvidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8) ==
  "the keys of the evidence's proofs are not statements"
#guard evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8) != ""
#guard evidenceWhy (evidenceOf "{\"proofs\":{\"statements\":[]}}".toUTF8) != ""
#guard evidenceWhy (membersEvidenceOf "{\"proofs\":{\"statements\":[]}}".toUTF8) ==
  evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8)
#guard evidenceWhy (evidenceEitherOf "{\"proofs\":{\"statements\":[]}}".toUTF8) ==
  evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8)
#guard evidenceWhy (evidenceEitherOf "{\"proofs\":{\"sessions\":[]}}".toUTF8) ==
  evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8)
#guard evidenceWhy (evidenceEitherOf "not JSON".toUTF8) == evidenceWhy (evidenceOf "not JSON".toUTF8)

private def memberT (name : String) (units : Array (Nat × Nat)) : Member := { name, units, g := 1, own := sharedT }

private def statementT (member : String) (sessions : Nat := 1) : StatementFiles :=
  { member := some member, sessions := (List.replicate sessions { name := "s", record := "r", reps := #[] }).toArray }

private def membersT : Array Member :=
  #[memberT "m2" #[(2, 3)], memberT "m10" #[(10, 11)], memberT "m0" #[(0, 1), (5, 6)]]

private def whyT {α : Type} : V α → String
  | .error e => e
  | .ok _ => ""

#guard (memberT "m0" #[(0, 1), (5, 6)]).first == 0
#guard (memberT "m" #[(2, 4), (7, 8)]).indices == #[2, 3, 7]
#guard ascending 0 [(0, 1), (1, 3), (7, 8)]
#guard !ascending 0 [(0, 2), (1, 3)]
#guard !ascending 0 [(2, 2)]
#guard ((membersHeld membersT #[statementT "m0", statementT "m2", statementT "m10"]).toOption.map (·.map (·.name))) ==
  some #["m0", "m2", "m10"]
#guard ((membersHeld membersT #[statementT "m2", statementT "m10"]).toOption.map (·.map (·.name))) ==
  some #["m2", "m10"]
#guard ((membersHeld membersT #[]).toOption.map (·.size)) == some 0
#guard (whyT (membersHeld membersT #[statementT "m2" 0, statementT "m10"])).startsWith
  "member m2: its statement has 0 sessions"
#guard (whyT (membersHeld membersT #[statementT "m2", statementT "m0"])).startsWith
  "member m0: its statement follows a member's whose units start after its own"
#guard (whyT (membersHeld membersT #[statementT "m0", statementT "m10", statementT "m2"])).startsWith
  "member m2: its statement follows a member's"
#guard whyT (membersHeld membersT #[statementT "m0", statementT "m0"]) ==
  "member m0: a second statement: a member's units are one statement's"
#guard (whyT (membersHeld membersT #[statementT "m0" 2])).startsWith "member m0: its statement has 2 sessions"
#guard whyT (membersHeld membersT #[statementT "m3"]) == "member m3: the parameters name no such member"
#guard (whyT (membersHeld membersT #[{ member := none, sessions := #[] }])).startsWith "a statement of the proofs names"

private def unitsJ (us : Array Nat) : Json := Json.mkObj [("units", Json.mkObj [("indices", natsJson us)])]

#guard unitsHeld #[2, 3, 7] (unitsJ #[2, 3, 7])
#guard !unitsHeld #[2, 3, 7] (unitsJ #[2, 3])
#guard !unitsHeld #[2, 3] (unitsJ #[2, 3, 7])
#guard !unitsHeld #[2, 3] (unitsJ #[3, 2])
#guard !unitsHeld #[] Json.null
#guard unitsIn #[(2, 4), (7, 8)] 3 8 == 2
#guard unitsIn #[(2, 4), (7, 8)] 4 7 == 0
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 3)]), ("b", #[(3, 8)])] == some #[1, 2]
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 2), (2, 5)]), ("b", #[(5, 9)])] == some #[2, 1]
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 2)]), ("b", #[(2, 8)])] == some #[0, 3]
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 3)]), ("b", #[(3, 7)])] == none
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 1), (2, 8)]), ("b", #[(1, 2)])] == none
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt #[("a", #[(0, 3)]), ("b", #[(3, 8)])] (unitsJ #[2, 3, 7])) == ""
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt #[("a", #[(0, 3)]), ("b", #[(3, 8)])] (unitsJ #[0, 1, 2])) ==
  MEMBER_UNITS
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt #[("a", #[(0, 3)]), ("b", #[(3, 7)])] (unitsJ #[2, 3, 7])) ==
  MEMBER_STRATA
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).population Json.null
  "{\"units\":{\"indices\":[2,3,7]}}\nrest".toUTF8) != ""
#guard (runMembers (fun _ => Json.null) membersT #[] #[] ⟨#[]⟩).toOption.map (·.all) == some true
#guard whyT (runMembers (fun _ => Json.null) membersT #[] #[1] ⟨#[]⟩) ==
  "the units the sessions proved are not the evidence's proved_units"
#guard (whyT (runMembers (fun _ => Json.null) membersT #[statementT "m2"] #[] ⟨#[]⟩)).startsWith
  "member m2: circuit: "

private def rootJ (port root : String) : Json :=
  Json.mkObj [("domain", Json.str ("d" ++ root)), ("leaves", Lean.toJson (2 : Nat)), ("port", Json.str port),
    ("root", Json.str root)]

/-- A registration of the population `n` and the roots given, each a port and a root. -/
private def recordT (n : Nat) (roots : List (String × String)) : Json :=
  Json.mkObj [("population", Lean.toJson n), ("roots", Json.arr (roots.toArray.map fun (p, r) => rootJ p r)),
    ("scheme", Json.str "s")]

/-- Two members whose names hold a `/`. -/
private def keyedT : Array Member := #[memberT "t/c0" #[(0, 2)], memberT "t/c1" #[(2, 5)]]

private def strT : Json → String
  | .str s => s
  | _ => ""

/-- The roots `rootsOf` gives the member `name` of `keyedT`, each its bare port and its root. -/
private def rootsT (roots : List (String × String)) (name : String) : Option (Array (String × String)) :=
  (rootsOf keyedT (recordT 5 roots)).toOption.map fun rs =>
    (rs.getD name #[]).map fun r => (strT (getD r "port"), strT (getD r "root"))

#guard splitsOf "t/c0/x" == [("t", "c0/x"), ("t/c0", "x")]
#guard splitsOf "x" == []
#guard splitsOf "/x/" == [("", "x/"), ("/x", "")]
#guard rootsT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c0" == some #[("x", "aa"), ("y", "cc")]
#guard rootsT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c1" == some #[("y", "bb")]
#guard rootsT [] "t/c0" == some #[]
#guard (whyT (rootsOf #[memberT "t/c0" #[(0, 2)], memberT "t/c0/u" #[(2, 5)]]
  (recordT 5 [("t/c0/u/y", "bb")]))).startsWith "roots: the registration's root t/c0/u/y is two members'"
#guard (whyT (rootsOf keyedT (recordT 5 [("t/c2/x", "aa")]))).startsWith
  "roots: the registration's root t/c2/x is no member's"
#guard (whyT (rootsOf keyedT (recordT 5 [("t/c0", "aa")]))).startsWith
  "roots: the registration's root t/c0 is no member's"
#guard whyT (rootsOf keyedT (recordT 5 [("t/c0/x", "aa"), ("row-table", "bb")])) == ROW_TABLES
#guard whyT (rootsOf keyedT (Json.mkObj [("roots", Json.arr #[Json.mkObj [("root", Json.str "aa")]])])) ==
  "roots: a root of the registration names no port"
#guard whyT (rootsOf keyedT Json.null) == "roots: the registration's roots are not a list"
#guard (memberT "m" #[(2, 4), (7, 8)]).count == 3
#guard refusal (laidHeld keyedT (recordT 5 [])) == ""
#guard refusal (laidHeld keyedT (recordT 6 [])) == MEMBERS_LAID
#guard refusal (laidHeld keyedT (recordT 4 [])) == MEMBERS_LAID
#guard refusal (laidHeld #[memberT "a" #[(0, 2), (3, 5)], memberT "b" #[(2, 3)]] (recordT 5 [])) == MEMBERS_LAID
#guard refusal (laidHeld #[memberT "a" #[(0, 2)], memberT "b" #[(3, 6)]] (recordT 5 [])) == MEMBERS_LAID
#guard refusal (laidHeld keyedT Json.null) == "the record names no population"

/-- A registered file of the ports given, each its name, its value, the hex digit its root repeats and its leaves. -/
private def regT (ports : List (String × String × Char × Nat)) : Json :=
  Json.mkObj [("format", Json.str Flock.Registered.FORMAT),
    ("ports", Json.mkObj (ports.map fun (name, value, c, leaves) => (name, Json.mkObj [
      ("domain", Json.str ("".pushn 'd' 128)), ("leaves", Lean.toJson leaves), ("reads", Json.arr #[]),
      ("root", Json.str ("".pushn c 128)), ("value", Json.str value)])))]

/-- The member `name` with the registered file `reg`. -/
private def readerT (name : String) (reg : Json) : Member :=
  { memberT name #[(0, 1)] with own := { sharedT with registered := some reg } }

#guard refusal (classHeld keyedT) == ""
#guard refusal (classHeld #[readerT "a" (regT [("p", "w", 'a', 2)]), memberT "b" #[(1, 2)],
  readerT "c" (regT [("q", "w", 'a', 2), ("r", "v", 'b', 2)])]) == ""
#guard whyT (classHeld #[readerT "a" (regT [("p", "w", 'a', 2)]), memberT "b" #[(1, 2)],
  readerT "c" (regT [("q", "w", 'b', 2)])]) ==
  classTwo "a" "c" "w" ("root " ++ "".pushn 'a' 128 ++ ", against " ++ "".pushn 'b' 128)
#guard whyT (classHeld #[readerT "a" (regT [("p", "w", 'a', 2)]), readerT "c" (regT [("q", "w", 'a', 3)])]) ==
  classTwo "a" "c" "w" "leaves 2, against 3"
#guard whyT (classHeld #[readerT "a" (regT [("p", "w", 'a', 2), ("q", "w", 'b', 2)])]) ==
  classTwo "a" "a" "w" ("root " ++ "".pushn 'a' 128 ++ ", against " ++ "".pushn 'b' 128)
#guard (whyT (classHeld #[memberT "a" #[(0, 1)], readerT "c" Json.null])).startsWith "member c: "

/-- The slice `rootsOf` and `Member.slice` give the member `name` of `keyedT` on the roots given. -/
private def sliceT (roots : List (String × String)) (name : String) : Json :=
  match rootsOf keyedT (recordT 5 roots), keyedT.find? (·.name == name) with
  | .ok rs, some m => m.slice (recordT 5 roots) rs
  | _, _ => Json.null

/-- A public file's header of the scheme `s`, `n` instances and the roots given, each a bare port and a root. -/
private def headerT (n : Nat) (roots : List (String × String)) : Json :=
  Json.mkObj [("frame_v3", Json.mkObj [("domain_ids", Json.mkObj (roots.map fun (p, r) => (p, Json.str ("d" ++ r)))),
    ("roots", Json.mkObj (roots.map fun (p, r) => (p, Json.str r)))]), ("instances", Lean.toJson n),
    ("scheme", Json.str "s")]

#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c0")
  (headerT 2 [("x", "aa"), ("y", "cc")])) == ""
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c1")
  (headerT 3 [("y", "bb")])) == ""
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c1")
  (headerT 3 [("y", "cc")])) == "public file: not the registration's: roots, domains"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c0")
  (headerT 2 [("x", "aa")])) == "public file: not the registration's: roots, domains"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb")] "t/c0")
  (headerT 2 [("x", "aa"), ("y", "cc")])) == "public file: not the registration's: roots"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb")] "t/c0") (headerT 5 [("x", "aa")])) ==
  "public file: not the registration's: population"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa")] "t/c1") (headerT 3 [])) == ""

#guard Flock.hexOf (evidenceBytes #["ab".toUTF8]) == "0000000000000001" ++ "0000000000000002" ++ "6162"
#guard (evidenceParts (evidenceBytes #["ab".toUTF8, .empty, "c".toUTF8])).map (·.map Flock.hexOf) ==
  some #["6162", "", "63"]
#guard (evidenceParts (evidenceBytes #[])).map (·.size) == some 0
#guard Flock.hexOf (evidenceBytes #["a".toUTF8, "b".toUTF8]) != Flock.hexOf (evidenceBytes #["ab".toUTF8])
#guard (evidenceParts ((evidenceBytes #["ab".toUTF8]).push 0)).isNone
#guard (evidenceParts ((evidenceBytes #["ab".toUTF8]).extract 0 17)).isNone
#guard (evidenceParts .empty).isNone
#guard (whyT (sessionEvidence ⟨#[]⟩ { name := "s", record := "".pushn 'a' 128, reps := #[] })).startsWith
  "session s's record: "

end Spot.OneStageZk
