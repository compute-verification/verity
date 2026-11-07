/-
Sampled proofs' one-stage audit on the proofs themselves (`verity/sampled-proofs/one-stage/v2`), a `Core.BlobProtocol`: the
v1 audit (`OneStage`) with the verdicts computed here, from the statement's files, rather than read from the evidence.

* **params**: v1's, plus `proofs`, exactly `{"circuit", "public", "registered"}` and optionally `"members"`: the
  SHA-512s the auditor committed for the circuit and public files (lowercase hex of 64 bytes), the verifier's own
  registered file (`flock-verify --registered`'s object, or null for a program that reads no registered value), and
  the verifier's own layout of a window (`layoutOf`): `[[key, n], ...]`, its members in unit order, member `key`'s ports
  registered as `key/<port>`, its units the `n` after those of the members before it. Without `members` the whole
  registration is one member, its ports bare. The partition must not be null: the sessions are verified against the
  verifier's own partition object; and v1's `challenger`, with its `window_root`, must be named (`NO_CHALLENGER`);
* **commitment**: v1's (the registration record);
* **evidence**: v1's without `verdicts`, plus `proofs`, exactly `{"sessions"}`: for each of v1's `sessions` in order,
  `{"name", "record", "reps"}`: its name, its `session.json`'s SHA-512 and `{file: SHA-512}` naming at least each rep
  file the statement's session reads (`<stream with / as .>.bin`). The drawn units are proved in one session, whose
  statement is theirs: the check refuses evidence naming none or several (`oneSession`) before it reads a tree;
* **blobs**: the files those digests name (`Core.Blobs`), and the MUFU tables any lookup reads, found under their pinned
  SHA-512 as `flock-verify --archive` finds them;
* **coins** (`coinsOf`): v1's (`OneStage.coinsOf`), what `Challenger.drawCoins` reads from the evidence's `coins` under
  the challenger's key, on the verifier's window; none read refuses. The check below runs at them (`checkBlobsAt`): v1's
  `checkAt` refuses a draw other than the one they give before v1's `check`;
* **trees** (`treesOf`, `treesHeld`): the coin trees that same session's signed release log shows the challenger sampled
  and opened (`Challenger.trustedTreesOf`). Each session's coins must be one tree's, a tree no other session's, its
  record's `coin_tree`, link points and every round's coins as the challenger logged them, and every tree logged must be
  a session's (`Challenger.sessionsTreeReason`); else the check refuses, before any session is verified;
* **population** (`popHeld`): the registration's population held to the verifier's program: under its query, the
  program's proof blocks read stratum by stratum must be the record's `population` units in order (`Slots.popOk`), as
  the service's run refuses at registration (`Run.run`, `population-order`), else the check refuses, before any session
  is verified. v1's R2 holds only their count, and the session's knowledge soundness reads slot `u` as proof block `u`;
* **member** (`memberOf`): the member the session's statement is, found from the draw: the layout's member whose units
  `[base, base + n)` hold every unit the draw proves (`Draw.proved`, which v1's `checkAt` holds to the challenger's
  coins), never a member the case names. The layout's counts must sum to the record's population, and a draw that
  proves units of two members, or of none, refuses, before any session is verified;
* **public** (`publicHeld`): the public file's header held to that member's slice of the commitment's registration
  (`sliceOf`, `memberCodes`, `Record.matchesPublic`'s codes on it): its scheme the record's, its ports exactly the
  registered ports under `key/`, matched by whole path segment and named without that prefix, with their roots and
  domains, none missing and none extra, its instance count the member's `n`, and its `units.indices` the member's units
  in order, `base + i` for instance `i` (`indicesHeld`, `indicesHeld_of_publicHeld`), so no header written after the
  draw relabels an instance as another unit, and the instances the session proves are the drawn units less `base`;
  else the check refuses, before any session is verified;
* **check**: `flock-verify verify --zk --statement verity/flock-circuit` with `--circuit`, `--public`, `--partition`,
  `--program` and `--registered` the verifier's own and one `--session` a session, as `one_stage.proofs.verify` runs it
  (`verdicts`, the C-Flock library calls `verifyCmd` makes on that path, its last the files' ports and roots,
  `Flock.PortRoots.checkOf`, and the form `Flock.Guarantees.SampledKSLiveV4` covers, `Flock.KsScope.checkOf`); then v1's
  `check` on those verdicts, one of
  record a session (`flock-verify/<name>`). A session accepted under another program than the partition object's, proved
  units other than the evidence's `proved_units`, or a circuit, public or record file no blob holds, rejects; a rep
  file no blob holds is that session's reject, as a file `flock-verify` can't read is.

Without blobs (`toProtocol`) every check rejects: no blob holds the circuit file.
-/
import CoreProtocol.Blobs
import SampledProofs.OneStage
import SampledProofs.ChallengerTrees
import SampledProofs.Slots
import SampledProofs.PopOrder
import Flock
import Std.Data.HashMap
import Std.Data.HashSet

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
`--program` reads), the partition object, the circuit and public files' committed SHA-512s and the registered file. -/
structure Own where
  programJ : Json
  program : ByteArray
  partition : Json
  circuit : String
  pub : String
  registered : Option Json

/-- A member of the registered window as the verifier lays it out (module docstring, **params**, **member**): the key
its ports are registered under, its first unit and its unit count. -/
structure Member where
  key : String
  base : Nat
  n : Nat
  deriving BEq, Repr

structure Params where
  base : OneStage.Params
  own : Own
  /-- The verifier's layout of the window (`layoutOf`); none when the whole registration is one member. -/
  members : Option (Array Member) := none

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

/-- The refusal of a `proofs.members` that is not a layout (`layoutOf`). -/
def MEMBERS_WHY : String :=
  "proofs: members is not the verifier's layout of the window ([[key, n], ...] in unit order: each key non-empty, " ++
    "distinct and not another's whole-segment prefix, each n positive)"

/-- `key`'s prefixes that end before one of its `/`s: a member keyed so would hold `key`'s ports (`sliceOf`). -/
private def segmentPrefixes (key : String) : List String :=
  let cs := key.toList
  (List.range cs.length).filterMap fun i => if cs[i]? == some '/' then some (String.ofList (cs.take i)) else none

/-- The verifier's own layout of the window (module docstring, **params**): `[[key, n], ...]` in unit order, each
member's base the sum of the counts before it; a refusal (`MEMBERS_WHY`) unless every key is non-empty, distinct and no
other's whole-segment prefix, so no port is two members', and every `n` is positive. -/
def layoutOf (j : Json) : V (Array Member) := do
  let .arr ms := j | throw MEMBERS_WHY
  let mut out : Array Member := #[]
  let mut base := 0
  for x in ms do
    let some (key, n) := (match x with
      | .arr a => if a.size != 2 then none else match a[0]!, nat? a[1]! with
        | .str k, some n => some (k, n)
        | _, _ => none
      | _ => none) | throw MEMBERS_WHY
    if key.isEmpty || n == 0 then throw MEMBERS_WHY
    out := out.push { key, base, n }
    base := base + n
  let keys : Std.HashSet String := out.foldl (fun s m => s.insert m.key) {}
  if keys.size != out.size || out.any (fun m => (segmentPrefixes m.key).any keys.contains) then throw MEMBERS_WHY
  pure out

def paramsOfJson (j : Json) : V Params := do
  let some pr := field? j "proofs"
    | throw ("the parameters name no proofs (the auditor's circuit and public files' digests and the verifier's " ++
      "registered reads)")
  let layout := field? pr "members"
  exactly pr (#["circuit", "public", "registered"] ++ if layout.isSome then #["members"] else #[]) "proofs"
  let (circuit, pub, registered) ← committedOf pr
  let members ← layout.mapM layoutOf
  let rest := Json.mkObj (without j "proofs")
  let base ← OneStage.paramsOfJson rest
  if base.challenger.isNone then throw NO_CHALLENGER
  let some partition := base.verifier.partition
    | throw "partition is null: the sessions are verified against the verifier's own partition object"
  let programJ := getD rest "program"
  pure { base, members,
         own := { programJ, program := (← Flock.canon programJ).toUTF8, partition, circuit, pub, registered } }

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
  let j ← (Json.parse (← String.fromUTF8? b)).toOption
  let bare := match j with
    | .obj kvs => Json.mkObj (kvs.toArray.filter (·.1 != "annotations")).toList
    | x => x
  return Flock.hexOf (Flock.Sha512.hash (← (Flock.canon bare).toOption).toUTF8)

private def headerLine (b : ByteArray) : ByteArray := Id.run do
  for i in [0:b.size] do
    if b[i]! == 10 then return b.extract 0 i
  return b

/-- The public file's header as `HmRow.loadPublic` reads it: parsed, then `Flock.norm`. -/
private def headerOf (pubB : ByteArray) : V Json := do
  let some line := String.fromUTF8? (headerLine pubB) | throw "the public file's header is not UTF-8"
  Flock.norm <$> (Json.parse line).mapError (s!"the public file's header is not JSON: {·}")

/-- The refusal of a draw that proves units of more than one member (`memberOf`): the case verifies one member's session. -/
def ONE_MEMBER : String :=
  "the draw proves units outside one member of the layout: the case verifies one member's session"

/-- The member the draw lands in (module docstring, **member**): the layout `members`' member whose units
`[base, base + n)` hold every unit the draw proves (`Draw.proved`), which v1's `checkAt` holds to the challenger's
coins; the record's whole population as one unkeyed member when the parameters lay out none. A refusal when the
record names no population, the layout's counts are not it, or the draw proves no unit or units outside one member. -/
def memberOf (members : Option (Array Member)) (record draw : Json) : V Member := do
  let some pop := nat? (getD record "population") | throw "the record names no population"
  match members with
  | none => pure { key := "", base := 0, n := pop }
  | some ms =>
    let total := ms.foldl (fun t m => t + m.n) 0
    if total != pop then
      throw s!"the layout's members count {total} units, not the record's population {pop}"
    let us := Draw.proved draw
    let inM := fun (m : Member) (v : Int) => decide ((m.base : Int) ≤ v ∧ v < m.base + m.n)
    let some u := us[0]? | throw "the draw proves no unit: no member to verify"
    let some m := ms.find? (inM · u) | throw s!"{ONE_MEMBER}: it proves unit {u}, in no member"
    match us.find? (!inM m ·) with
    | some v =>
      let other := ((ms.find? (inM · v)).map (s!"member {·.key}")).getD "no member"
      throw s!"{ONE_MEMBER}: it proves unit {u} of member {m.key} and unit {v} of {other}"
    | none => pure m

/-- The registration `rec`'s roots under member `key` (module docstring, **public**): each port registered as `key/p`,
named `p`, with its last root (`Record.byPort`'s); every port, as named, for the unkeyed member `""`. The prefix ends at
its `/`, so a slice takes whole path segments: member `T/c24` holds none of `T/c2436`'s ports. One pass over the roots. -/
def sliceOf (rec : Json) (key : String) : Std.HashMap String Json :=
  let pre := if key.isEmpty then [] else key.toList ++ ['/']
  (match getD rec "roots" with | .arr rs => rs | _ => #[]).foldl (init := {}) fun acc r =>
    match getD r "port" with
    | .str p =>
      let cs := p.toList
      if cs.take pre.length == pre then acc.insert (String.ofList (cs.drop pre.length)) r else acc
    | _ => acc

/-- `Record.matchesPublic`'s codes on member `m`'s slice of the registration `rec` (`sliceOf`), in its order: the
header's `scheme` not the record's; its `frame_v3` `roots` not exactly the slice's ports with their roots (one missing,
one extra, or one's root another); its `domain_ids` not exactly those ports with their domains; its `instances` not
the member's count `n`. -/
def memberCodes (rec : Json) (m : Member) (scheme roots domains instances : Json) : Array String := Id.run do
  let s := sliceOf rec m.key
  let rs := entries roots
  let mut out := #[]
  if !jeq scheme (getD rec "scheme") then out := out.push "scheme"
  if !(rs.size == s.size && rs.all fun (p, v) => (s.get? p).any fun r => jeq (getD r "root") v) then
    out := out.push "roots"
  if !((entries domains).size == s.size && s.toArray.all fun (p, r) => jeq (getD domains p) (getD r "domain")) then
    out := out.push "domains"
  if nat? instances != some m.n then out := out.push "population"
  return out

/-- The refusal of a public file whose instances are not its member's units in order (`indicesHeld`). -/
def INDICES_WHY : String :=
  "public file: its units.indices are not its member's units in order (instance i is unit base + i)"

/-- The header `h`'s `units.indices` are member `m`'s units in order, `[base, base + n)`: the unit v1 credits for an
instance a session proves (`provedUnits`) is that instance's slot in the window, which the session's knowledge
soundness reads (`popHeld`), fixed by the layout and the registration before any coin rather than by a header written
after it. -/
def indicesHeld (m : Member) (h : Json) : Bool :=
  match getD (getD h "units") "indices" with
  | .arr is => is.size == m.n && (List.range m.n).all fun i => (is[i]? >>= nat?) == some (m.base + i)
  | _ => false

/-- The public file held to member `m` of the registration `rec` (module docstring, **public**;
`registration.matches_public` on the member's slice): its header's scheme, `frame_v3` roots and their `domain_ids`,
and `instances` against the record's scheme and the slice's roots and domains and the member's count (`memberCodes`), a
refusal naming the codes that fail; then its `units.indices` the member's units in order (`indicesHeld`). -/
def publicHeld (rec : Json) (m : Member) (pubB : ByteArray) : V Unit := do
  let h ← headerOf pubB
  let fv := getD h "frame_v3"
  let codes := memberCodes rec m (getD h "scheme") (getD fv "roots") (getD fv "domain_ids") (getD h "instances")
  let whose := if m.key.isEmpty then "" else s!" for member {m.key}"
  if !codes.isEmpty then throw s!"public file: not the registration's{whose}: {", ".intercalate codes.toList}"
  if !indicesHeld m h then throw INDICES_WHY

/-- `publicHeld` passes only on a public file whose header's `units.indices` are member `m`'s units in order
(`indicesHeld`): the unit `provedUnits` credits for instance `i` is `base + i`, fixed by the layout and the registration
before any coin. -/
theorem indicesHeld_of_publicHeld {rec : Json} {m : Member} {pubB : ByteArray} (h : publicHeld rec m pubB = .ok ()) :
    ∃ hd, headerOf pubB = .ok hd ∧ indicesHeld m hd = true := by
  unfold publicHeld at h
  cases hh : headerOf pubB with
  | error e => simp [hh, bind, Except.bind] at h
  | ok hd =>
    refine ⟨hd, rfl, ?_⟩
    cases hi : indicesHeld m hd
    · simp only [hh, hi, bind, Except.bind, Bool.not_false, ↓reduceIte, throw, throwThe, MonadExceptOf.throw] at h
      split at h <;> simp_all
    · rfl

/-- `one_stage.proofs.proved_units`: the public file's unit indices (`units.indices`) of the instances the session's
record drew (`unit_draw.units`), ascending. -/
def provedUnits (pubB : ByteArray) (record : String) : V (Array Nat) := do
  let h ← headerOf pubB
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

/-- The units the accepted sessions proved (`proved`, in session order) held to the evidence's `proved_units` (`want`):
sorted, they must be it. `want` is strictly ascending (`Draw.checkDrawFull`), so a unit two sessions prove (or one
session twice) refuses, and so does a drawn unit no session proves. -/
def provedHeld (proved want : Array Nat) : V Unit :=
  if proved.qsort (· < ·) != want then throw "the units the sessions proved are not the evidence's proved_units"
  else pure ()

/-- One `flock-verify verify --zk` over the evidence's sessions on the blobs, as `one_stage.proofs.verify` runs it, each
session set up at its own record's draw: every session's verdict, the units the accepted ones proved and whether all
accepted. A refusal for what `proofs.verify` refuses: registered values read without the verifier's registered file, a
file no blob holds, an accepted session naming another program than the partition object's, or proved units other than
the evidence's (`provedHeld`). -/
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
  let rd ← readsOf p.registered
  let tags := Flock.Witness.zkTags I
  let scope := Flock.ProvedScope.checkOf (Flock.ProvedScope.circuitOf tags I.circuitFile I.tables) tags true pubB I.count
    false rd.registered.isSome
  let ksFiles := Flock.PortRoots.checkOf (Flock.ProvedScope.circuitOf tags I.circuitFile I.tables) tags pubB
  let r ← sessions.foldlM (step I rd bs scope ksFiles (programDigest p.program) (getD p.partition "program"))
    ⟨#[], #[], #[], true⟩
  if let some want := provedUnits? then
    if r.all then provedHeld r.proved want
  pure r

/-- The sessions' verdicts in order (`run`'s). -/
def verdicts (p : Own) (sessions : Array RunFiles) (provedUnits? : Option (Array Nat)) (bs : Core.Blobs) :
    V (Array (String × Json)) :=
  (·.out) <$> run p sessions provedUnits? bs

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

/-- The check at the coins `ξ` (module docstring, **check**): the evidence's one session (`oneSession`), its coins held
to the trees the challenger logged (`treesOf`, `treesHeld`), the registration's population held to the program
(`popHeld`), the member the draw lands in (`memberOf`), the parameters' public file held to that member's slice of the
registration (`publicHeld`), then the session's verdict on the blobs, then v1's `checkAt` at `ξ` on it. v1 accepts only verdicts that all accepted (`VerdictsAccept`); the run's own
`all` says it again, so an acceptance reads the run rather than the verdicts' JSON. -/
def checkBlobsAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  let r : V Run := do
    oneSession e.sessions
    treesHeld (← treesOf p e) e bs
    popHeld p.base.verifier c.record
    publicHeld c.record (← memberOf p.members c.record e.base.draw) (← bs.get p.own.pub "public file")
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
  (Json.parse "{\"population\":2,\"roots\":[{\"domain\":\"dd\",\"port\":\"x\",\"root\":\"aa\"}],\"scheme\":\"s\"}").toOption.getD
    .null

private def refusal : V Unit → String
  | .error e => e
  | .ok () => ""

private def whole : Member := { key := "", base := 0, n := 2 }

#guard (memberOf none heldRecord .null).toOption == some whole
#guard refusal (publicHeld heldRecord whole
  ("{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"," ++
    "\"units\":{\"indices\":[0,1]}}\nrest").toUTF8) == ""
#guard [("[1,0]" : String), "[0,2]", "[0]", "[0,1,2]", "[\"0\",1]"].all fun is => refusal (publicHeld heldRecord whole
  ("{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"," ++
    "\"units\":{\"indices\":" ++ is ++ "}}").toUTF8) == INDICES_WHY
#guard refusal (publicHeld heldRecord whole
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"}".toUTF8) ==
  INDICES_WHY
#guard refusal (publicHeld heldRecord whole
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"ab\"}},\"instances\":2,\"scheme\":\"s\"}".toUTF8) ==
  "public file: not the registration's: roots"
#guard refusal (publicHeld heldRecord whole
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"de\"},\"roots\":{\"x\":\"aa\",\"y\":\"bb\"}},\"instances\":3,\"scheme\":\"s\"}".toUTF8) ==
  "public file: not the registration's: roots, domains, population"
#guard (refusal (publicHeld heldRecord whole "not a public file".toUTF8)).startsWith "the public file's header is not JSON: "

private def js (s : String) : Json := (Json.parse s).toOption.getD .null

/-- A window of three members, `T/c24`'s units `[0, 2)`, `T/c2436`'s `[2, 5)` and `U/c0`'s `[5, 6)`, beside a row table's
port no member holds. -/
private def windowRecord : Json := js (r#"{"population":6,"scheme":"s","roots":[
  {"domain":"d1","port":"T/c24/x","root":"a1"},{"domain":"d1","port":"T/c24/y","root":"a2"},
  {"domain":"d2","port":"T/c2436/x","root":"b1"},{"domain":"d2","port":"T/c2436/y","root":"b2"},
  {"domain":"d3","port":"U/c0/x","root":"c1"},{"domain":"d4","port":"row-table/0/w","root":"e1"}]}"#)

private def windowLayout : Array Member :=
  ((layoutOf (js r#"[["T/c24",2],["T/c2436",3],["U/c0",1]]"#)).toOption).getD #[]

private def c24 : Member := { key := "T/c24", base := 0, n := 2 }
private def c2436 : Member := { key := "T/c2436", base := 2, n := 3 }

private def memberRefusal (draw : String) : String :=
  match memberOf (some windowLayout) windowRecord (js draw) with
  | .error e => e
  | .ok m => m.key

private def header (roots domains : String) (instances : Nat) (indices : String) : ByteArray :=
  (s!"\{\"frame_v3\":\{\"domain_ids\":{domains},\"roots\":{roots}},\"instances\":{instances},\"scheme\":\"s\"," ++
    s!"\"units\":\{\"indices\":{indices}}}\nrest").toUTF8

#guard windowLayout == #[c24, c2436, { key := "U/c0", base := 5, n := 1 }]
#guard (layoutOf (js r#"[["T/c2",1],["T/c24",1]]"#)).toBool
#guard [r#"[["T",1],["T/c1",1]]"#, r#"[["T/c1",1],["T/c1",2]]"#, r#"[["",1]]"#, r#"[["T/c1",0]]"#, r#"[["T/c1",-1]]"#,
    r#"[["T/c1"]]"#, r#"[[1,1]]"#, r#"{"T/c1":1}"#].all fun l => refusal ((fun _ => ()) <$> layoutOf (js l)) == MEMBERS_WHY

-- The member comes from the draw: the one whose units hold every unit it proves, its closure's too.
#guard memberRefusal r#"{"units":[3]}"# == "T/c2436"
#guard memberRefusal r#"{"units":[2,4]}"# == "T/c2436"
#guard memberRefusal r#"{"units":[0],"closure":[1]}"# == "T/c24"
-- A drawn member the case doesn't verify: the draw proves units of two members, or of none.
#guard memberRefusal r#"{"units":[1,3]}"# ==
  s!"{ONE_MEMBER}: it proves unit 1 of member T/c24 and unit 3 of member T/c2436"
#guard memberRefusal r#"{"units":[2],"closure":[5]}"# ==
  s!"{ONE_MEMBER}: it proves unit 2 of member T/c2436 and unit 5 of member U/c0"
#guard memberRefusal r#"{"units":[6]}"# == s!"{ONE_MEMBER}: it proves unit 6, in no member"
#guard memberRefusal r#"{"units":[]}"# == "the draw proves no unit: no member to verify"
#guard (match memberOf (some (windowLayout.extract 0 2)) windowRecord (js r#"{"units":[3]}"#) with
  | .error e => e | .ok _ => "") == "the layout's members count 5 units, not the record's population 6"

-- A slice is the ports under its key by whole path segment, its prefix stripped: `T/c24` holds none of `T/c2436`'s.
#guard (sliceOf windowRecord "T/c24").size == 2 && ((sliceOf windowRecord "T/c24").get? "y").isSome
#guard ((sliceOf windowRecord "T/c24").get? "x").any fun r => jeq (getD r "root") (.str "a1")
#guard (sliceOf windowRecord "T/c2").size == 0
#guard (sliceOf windowRecord "").size == 6

#guard refusal (publicHeld windowRecord c2436
  (header r#"{"x":"b1","y":"b2"}"# r#"{"x":"d2","y":"d2"}"# 3 "[2,3,4]")) == ""
-- The wrong member: `T/c24`'s public file where the draw landed in `T/c2436`.
#guard refusal (publicHeld windowRecord c2436 (header r#"{"x":"a1","y":"a2"}"# r#"{"x":"d1","y":"d1"}"# 2 "[0,1]")) ==
  "public file: not the registration's for member T/c2436: roots, domains, population"
-- A port outside the member's key, one missing, or one under its registered name rather than the member's.
#guard refusal (publicHeld windowRecord c24
  (header r#"{"x":"a1","y":"a2","36/x":"b1"}"# r#"{"x":"d1","y":"d1"}"# 2 "[0,1]")) ==
  "public file: not the registration's for member T/c24: roots"
#guard refusal (publicHeld windowRecord c2436 (header r#"{"x":"b1"}"# r#"{"x":"d2","y":"d2"}"# 3 "[2,3,4]")) ==
  "public file: not the registration's for member T/c2436: roots"
#guard refusal (publicHeld windowRecord c2436
  (header r#"{"T/c2436/x":"b1","T/c2436/y":"b2"}"# r#"{"x":"d2","y":"d2"}"# 3 "[2,3,4]")) ==
  "public file: not the registration's for member T/c2436: roots"
#guard refusal (publicHeld windowRecord c2436
  (header r#"{"x":"b1","y":"b2"}"# r#"{"x":"d2","y":"d2","z":"d2"}"# 3 "[2,3,4]")) ==
  "public file: not the registration's for member T/c2436: domains"
-- Its instances are the member's units in order, never the window's from 0.
#guard ["[0,1,2]", "[2,3]", "[3,4,5]", "[2,4,3]"].all fun is => refusal (publicHeld windowRecord c2436
  (header r#"{"x":"b1","y":"b2"}"# r#"{"x":"d2","y":"d2"}"# 3 is)) == INDICES_WHY

end SampledProofs.OneStageZk
