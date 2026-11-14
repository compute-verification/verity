/-
The audit's check reading its blobs one at a time (`Core.Stream`), so that the per-member form's blobs, larger together
than memory, are checked holding one circuit file, one member's public file and one session's files at a time, and what
the files' pass keeps of the rest.

* `planOf`: the per-member form's order, read off the parameters and the evidence before any blob: each circuit file in
  the order the members first name it, then each member that names it, in the members' order: its public file, and for a
  member a statement names, its session's record and rep files. A refusal where a digest is named twice or is a pinned
  table's, so each blob fed has its own SHA-512;
* `State`, `step`: the pinned tables first, kept whole; then, a blob at a time, what the files' pass keeps of a circuit
  file (`fileOf`), each member's files' verdict (`Member.filesOn`) and, for a member a statement names, its session's
  entry of the trees' check and its verdict's pieces (`SessionKept`), each on the blobs held when they are computed;
* `finish`: `checkBlobs`' order on what the step kept: the coins, the form's structure with the trees' check on the kept
  entries, the roots, the statements' members, every member's files, each statement's session after the ones before it,
  the proved units, then v1's check;
* `keepAll`: every other form's stream, which keeps every blob and is `checkBlobs` on them;
* `protocol`: `OneStageZk.protocol` with these streams, and `StreamSound`, the statement that its streams decide what
  `checkBlobs` decides on every feed in their order (`Core.StreamProtocol.Sound`), which `run_accept` reads.
-/
module

public import CoreProtocol.Stream
public import Spot.OneStageZk

/-!
# The per-member check as a stream

`planOf`, the order the blobs are read in; `State`, `step` and `finish`, the check on what each step keeps; `keepAll`,
every other form's stream; and `protocol` with `StreamSound`, the statement that its streams decide what `checkBlobs`
decides.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk.Streamed

open Lean (Json)
open Flock (V)

/-- What a planned blob is to the check. -/
inductive Role where
  /-- a circuit file, under its digest as the members name it; the members that name it come next -/
  | circuit (d : String)
  /-- member `m`'s public file (its place among the parameters' members), and the statement that names it, if one does -/
  | pub (m : Nat) (st : Option Nat)
  /-- statement `st`'s session's record -/
  | record (st : Nat)
  /-- that session's `j`-th rep file, in its `reps`' order -/
  | rep (st j : Nat)
  deriving Inhabited, BEq

/-- The SHA-512 the lowercase hex `d` names, as `Core.Blobs.get` reads it. -/
def digestBytes (d what : String) : V ByteArray := do
  let b ← Flock.unhex d
  if b.size != 64 || Flock.hexOf b != d then throw s!"{what}: {d} is not a SHA-512 digest (128 lowercase hex)"
  pure b

/-- The pinned MUFU tables' SHA-512, in `Flock.MUFU_TABLES`' order. -/
def pinned : List ByteArray := Flock.MUFU_TABLES.filterMap fun t => (Flock.unhex t.sha512).toOption

/-- The circuit files' digests in the order the members `ms` first name them, and each one's members, in order. -/
def groupsOf (ms : Array Member) : Array (String × Array Nat) :=
  let (order, at_) := ms.zipIdx.foldl
    (fun (acc : Array String × Std.HashMap String (Array Nat)) (m, i) =>
      match acc.2[m.own.circuit]? with
      | some is => (acc.1, acc.2.insert m.own.circuit (is.push i))
      | none => (acc.1.push m.own.circuit, acc.2.insert m.own.circuit #[i]))
    (#[], {})
  order.map fun d => (d, at_.getD d #[])

/-- A planned blob: its SHA-512, read off the digest `d` (`digestBytes`), and its role. -/
def blob (d what : String) (r : Role) : V (ByteArray × Role) := do
  pure (← digestBytes d what, r)

/-- Member `i`'s blobs: its public file, and where statement `k` names it (`stOf`), that statement's session's record
and rep files. -/
def memberPlan (ms : Array Member) (sts : Array StatementFiles) (stOf : Std.HashMap String Nat) (i : Nat) :
    V (Array (ByteArray × Role)) := do
  let some m := ms[i]? | throw "plan: a member out of range"
  let st := stOf[m.name]?
  let pub ← blob m.own.pub s!"member {m.name}'s public file" (.pub i st)
  match st with
  | none => pure #[pub]
  | some k =>
    let some s := (sts[k]?.bind (·.sessions[0]?)) | throw s!"member {m.name}: its statement names no session"
    let record ← blob s.record s!"session {s.name}'s record" (.record k)
    let reps ← s.reps.zipIdx.mapM fun ((f, rd), j) => blob rd s!"session {s.name}'s files: {f}" (.rep k j)
    pure (#[pub, record] ++ reps)

/-- One circuit file's blobs: the file, then each member that names it, in the members' order. -/
def groupPlan (ms : Array Member) (sts : Array StatementFiles) (stOf : Std.HashMap String Nat)
    (g : String × Array Nat) : V (Array (ByteArray × Role)) := do
  let c ← blob g.1 "circuit" (.circuit g.1)
  let members ← g.2.mapM (memberPlan ms sts stOf)
  pure (#[c] ++ members.flatten)

/-- The planned blobs' SHA-512 are distinct and none one of the tables' `tabs`, else a refusal naming the first that
is. -/
def distinctFrom (tabs : List ByteArray) : Std.HashSet String → List ByteArray → V Unit
  | _, [] => pure ()
  | seen, b :: bs =>
    let h := Flock.hexOf b
    if tabs.any (fun t => decide (t = b)) then throw s!"plan: {h} is a pinned table's SHA-512, and the tables come first"
    else if seen.contains h then throw s!"plan: {h} is named twice, and the streamed check reads each blob once"
    else distinctFrom tabs (seen.insert h) bs

/-- **The per-member form's order** (module docstring): its statements' structure held first (`membersHeld`), each
blob under its SHA-512 with its role, group by group (`groupPlan`), the SHA-512 distinct and none a pinned table's
(`distinctFrom`). -/
def planOf (ms : Array Member) (sts : Array StatementFiles) : V (Array (ByteArray × Role)) := do
  let held ← membersHeld ms sts
  let stOf : Std.HashMap String Nat := held.zipIdx.foldl (fun acc (m, i) => acc.insert m.name i) {}
  let out := (← (groupsOf ms).mapM (groupPlan ms sts stOf)).flatten
  distinctFrom pinned {} (out.toList.map (·.1))
  pure out

/-- `verdictOf` past the coin-tree key's freshness, on the session's proof files `proofs`. -/
def restOf (I : Flock.Witness.Inputs) (scope ksFiles population : V Unit) (record : String) (sample : Option Json)
    (st : SetUp) (proofs : Array ByteArray) : V Unit := do
  let (spec, tabs, _) ← st
  Flock.Zk.verify spec tabs record proofs
  scope
  Flock.ProvedScope.sizesOk true (tabs.map (·.m))
  (if sample.isNone then Flock.ProvedScope.drawLaw true (Flock.Witness.zkTags I) false sample else pure () : V Unit)
  Flock.RecSetup.fullDraw none false sample record
  Flock.RecSetup.innerDrawn none
  Flock.ProvedScope.programHeld I.program
  Flock.RecordText.check record
  ksFiles
  let tags := Flock.Witness.zkTags I
  Flock.KsScope.checkOf (Flock.ProvedScope.circuitOf tags I.circuitFile I.tables) tags I.publicFile I.count sample
  population

/-- What `verdictJson` reads of a session's setup: its partition finding and openings. -/
def foundOf (st : SetUp) : Json × Json :=
  match st with | .ok (_, _, p, o) => (p, o) | .error _ => (Json.null, Json.null)

/-- `verdictJson` on what it reads of the setup (`foundOf`). -/
def verdictJsonOn (rd : Reads) (programSha : Option String) (s : RunFiles) (found : Json × Json) (ok : Bool)
    (why : String) (units : Array Nat) : String × Json :=
  let (part, opened) := found
  (s!"flock-verify/{s.name}", Json.mkObj ([("accepted", Json.bool ok),
    ("of_record", Json.bool true), ("session", Json.str s.name), ("why", Json.str why), ("units", natsJson units)] ++
    (if part.isNull then [] else [("partition", part)]) ++
    (match programSha with | some h => [("program_sha512", Json.str h)] | none => []) ++
    (match rd.registered with | some r => [("registered", Flock.Registered.finding r)] | none => []) ++
    (if opened.isNull then [] else [("openings", opened)])))

/-- **What the check keeps of one session's `step`**, made while its statement's files are held: the refusals before
its verdict (`opening`), its coin-tree key, its proof files found (`files`), its verdict past the key's freshness
(`rest`), what its verdict's JSON reads of its setup (`found`) and the units its record samples (`units`). Only the key's
freshness reads the sessions before it, so `StepKept.step` makes the verdict on any run after the fact. -/
structure StepKept where
  opening : V Unit
  coinKey : Option String
  files : V Unit
  rest : V Unit
  found : Json × Json
  units : V (Array Nat)
  rd : Reads
  programSha : Option String
  want : Json
  s : RunFiles

/-- What is kept of `a.step bs r s` for any run `r`, on the blobs `bs`. -/
def StepKept.of (a : StepAt) (bs : Core.Blobs) (s : RunFiles) : StepKept :=
  let base : StepKept :=
    { opening := pure (), coinKey := none, files := pure (), rest := pure (), found := (Json.null, Json.null)
      units := pure #[], rd := a.rd, programSha := a.programSha, want := a.want, s }
  match bs.get s.record s!"session {s.name}'s record" with
  | .error e => { base with opening := .error e }
  | .ok rb =>
    match String.fromUTF8? rb with
    | none => { base with opening := .error s!"session {s.name}'s record is not UTF-8" }
    | some record =>
      let sample := Flock.Witness.recordDraw record
      match drawFitsWhy WORD sample with
      | some what => { base with opening := .error s!"session {s.name}: {what}: {FITS}" }
      | none =>
        let st := setupAt a.I a.rd sample
        let files := filesOf bs s st
        { base with
          coinKey := coinKeyOf record st
          files := files >>= fun _ => pure ()
          rest := files >>= restOf a.I a.scope a.ksFiles a.population record sample st
          found := foundOf st
          units := provedUnits a.I.publicFile record }

/-- `step` on what was kept of it, after the run `r`: `push` on the verdict made of the kept pieces, the coin-tree
key's freshness read against `r`. -/
def StepKept.step (k : StepKept) (r : Run) : V Run := do
  k.opening
  let coinKeys := match k.coinKey with | some key => r.coinKeys.push key | none => r.coinKeys
  let fresh : V Unit := match k.coinKey with
    | some key => Flock.CoinTree.checkFresh r.coinKeys key
    | none => pure ()
  match k.files >>= fun _ => fresh >>= fun _ => k.rest with
  | .error why =>
    pure { coinKeys, all := false, proved := r.proved,
           out := r.out.push (verdictJsonOn k.rd k.programSha k.s k.found false why #[]) }
  | .ok () =>
    if (k.programSha.map Json.str) != some k.want then
      throw s!"session {k.s.name}'s verdict names program {k.programSha.getD "none"}, not the partition object's {k.want.compress}"
    else do
      let units ← k.units
      pure { coinKeys, all := r.all, proved := r.proved ++ units,
             out := r.out.push (verdictJsonOn k.rd k.programSha k.s k.found true "" units) }

/-- What the check keeps of a statement's session: its entry of the trees' check (`membersTreesHeld`'s, or that
check's refusal), and, where its member's statement read, what is kept of its `step`. -/
structure SessionKept where
  trees : V (String × Array (String × Option Nat) × Json × String)
  step : Option StepKept

/-- The session `s` of a statement whose member is `m`, on the blobs `bs` and that member's statement `a`. -/
def SessionKept.of (m : Member) (a : V StepAt) (bs : Core.Blobs) (s : RunFiles) : SessionKept :=
  { trees := do
      let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
        | throw s!"session {s.name}'s record is not UTF-8"
      let evidence ← sessionEvidence bs s
      pure (s.name, rootBTables m.own bs record, (Flock.JsonParse.parse record).toOption.getD Json.null,
        sha512Hex evidence)
    step := a.toOption.map fun a => StepKept.of a bs s }

/-- What the check reads besides the blobs: the parameters, the registration, the evidence, the members, the
statements, the statement's tags and each member's slice of the registration. -/
structure Ctx where
  p : Params
  c : OneStage.Commitment
  e : Evidence
  ms : Array Member
  sts : Array StatementFiles
  tags0 : Option Flock.Tags
  slice : Member → Json

def Ctx.of (p : Params) (c : OneStage.Commitment) (e : Evidence) (ms : Array Member) (sts : Array StatementFiles) :
    Ctx :=
  let roots := (rootsOf ms c.record).toOption.getD {}
  { p, c, e, ms, sts, tags0 := Flock.Tags.find? STATEMENT, slice := fun m => m.slice c.record roots }

/-- **What the stream holds**: the plan's roles and how many planned blobs came; the pinned tables, whole; the circuit
file open and what the files' pass keeps of it; the program kept; the public file of the member being read, and its
statement while its session's files come; that session's record and rep files; each member's files' verdict, by its
place; each statement's session kept, by its place. -/
structure State where
  roles : Array Role
  i : Nat
  tables : Array (ByteArray × ByteArray)
  file : Option (ByteArray × ByteArray × Option FileKept)
  prog : Option ProgramKept
  pub : Option (ByteArray × ByteArray)
  cur : Option (V StepAt)
  record : Option (ByteArray × ByteArray)
  reps : Array (ByteArray × ByteArray)
  files : Array (V Unit)
  sessions : Array (Option SessionKept)

def State.init (C : Ctx) (roles : Array Role) : State :=
  { roles, i := 0, tables := #[], file := none, prog := none, pub := none, cur := none, record := none, reps := #[]
    files := Array.replicate C.ms.size (.error "the member's public file came in no blob")
    sessions := Array.replicate C.sts.size none }

/-- The blobs the stream holds, each under its SHA-512. -/
def State.blobs (S : State) : Core.Blobs :=
  ⟨S.tables ++ (S.file.map fun (d, b, _) => (d, b)).toArray ++ S.pub.toArray ++ S.record.toArray ++ S.reps⟩

/-- A member's files' verdict on the blobs held, read as the files' pass reads it (`Member.filesOn`): on what is kept
of its program under its partition (`ProgramKept.next`). -/
def memberFiles (C : Ctx) (S : State) (m : Member) : V StepAt × Option ProgramKept :=
  match C.tags0 with
  | none => (m.files (C.slice m) S.blobs, S.prog)
  | some tags0 =>
    let tables := tablesOf S.blobs
    let fk := match S.file with | some (_, _, some fk) => fk | _ => fileOf tags0 tables S.blobs m.own.circuit
    let pk := ProgramKept.next S.prog m.own.program m.own.partition
    (m.filesOn tags0 tables fk pk (C.slice m) S.blobs, some pk)

/-- Statement `st`'s session, once its last file came: kept, and its record and rep files dropped. -/
def closeSession (C : Ctx) (st : Nat) (S : State) : State :=
  let next := match S.roles[S.i]? with | some (.rep k _) => k == st | _ => false
  if next then S else
    match C.sts[st]?.bind (·.sessions[0]?), (C.sts[st]?.bind (·.member)).bind fun n => C.ms.find? (·.name == n) with
    | some s, some m =>
      { S with sessions := S.sessions.set! st (some (SessionKept.of m (S.cur.getD (.error "")) S.blobs s)),
               record := none, reps := #[] }
    | _, _ => { S with record := none, reps := #[] }

/-- **One blob, by its role**: a pinned table, before any planned blob, is kept whole; a circuit file opens its
group, dropping the one before; a public file has its member's files' verdict kept; a session's record and rep files
are held until its last, when the session is kept (`closeSession`). -/
def step (C : Ctx) (S : State) (h b : ByteArray) : State :=
  if S.i = 0 ∧ pinned.any (fun t => decide (t = h)) then { S with tables := S.tables.push (h, b) } else
  let role := S.roles[S.i]?
  let S := { S with i := S.i + 1 }
  match role with
  | some (.circuit d) =>
    let held : Core.Blobs := ⟨S.tables.push (h, b)⟩
    { S with file := some (h, b, C.tags0.map fun tags0 => fileOf tags0 (tablesOf held) held d), pub := none,
             cur := none, record := none, reps := #[] }
  | some (.pub mi st) =>
    match C.ms[mi]? with
    | none => S
    | some m =>
      let S := { S with pub := some (h, b), cur := none, record := none, reps := #[] }
      let (a, prog) := memberFiles C S m
      { S with prog, files := S.files.set! mi (a >>= fun _ => pure ()), cur := st.map fun _ => a }
  | some (.record st) => closeSession C st { S with record := some (h, b), reps := #[] }
  | some (.rep st _) => closeSession C st { S with reps := S.reps.push (h, b) }
  | none => S

/-- `membersTreesHeld` on the entries kept, statement by statement. -/
def treesKept (ts : Array Challenger.LoggedTree) (log : List String) (n : Nat) (S : State) : V Unit := do
  let mut named : Array (String × Array (String × Option Nat) × Json × String) := #[]
  for k in [0:n] do
    let some (some sk) := S.sessions[k]? | throw "a statement's session came in no blob"
    named := named.push (← sk.trees)
  if let some why := Challenger.membersTreeReason ts log named then throw why

/-- `membersStructure` with the trees' check on the entries kept. -/
def structureKept (C : Ctx) (S : State) : V Unit := do
  let held ← membersHeld C.ms C.sts
  if (C.sts.flatMap (·.sessions)).map (·.name) != C.e.base.sessions then
    throw "the statements' sessions are not the evidence's sessions, in order"
  let ts ← treesOf C.p.base C.e.base
  orderHeld C.p C.c C.e
  treesKept ts (← linesOf C.e) (C.sts.zip held).size S
  popHeld C.p.base.verifier C.c.record
  laidHeld C.ms C.c.record
  classHeld C.ms

/-- One statement's session on what was kept of it, after the run `r`, a refusal naming its member. -/
def stepKept (S : State) (r : Run) (z : (StatementFiles × Member) × Nat) : V Run :=
  (match S.sessions[z.2]? with
    | some (some sk) =>
      match sk.step with
      | some k => k.step r
      | none => throw "the statement's member's files did not read"
    | _ => throw "a statement's session came in no blob").mapError (s!"member {z.1.2.name}: {·}")

/-- `membersRun` on what the step kept. -/
def runKept (C : Ctx) (S : State) : V Run :=
  structureKept C S >>= fun _ =>
  rootsOf C.ms C.c.record >>= fun _ =>
  membersHeld C.ms C.sts >>= fun held =>
  sampledHeld held C.e.base.provedUnits >>= fun _ =>
  S.files.toList.forM id >>= fun _ =>
  (C.sts.zip held).zipIdx.foldlM (stepKept S) ⟨#[], #[], #[], true⟩ >>= fun r =>
  (match r.all with
    | true => provedHeld r.proved C.e.base.provedUnits
    | false => pure ()) >>= fun _ => pure r

/-- **`checkBlobs` in the per-member form on what the step kept**: `checkMembersAt` at the coins, `membersRun` read
as `runKept`. -/
def finish (C : Ctx) (S : State) : Core.Outcome OneStage.Public Core.Integrity :=
  Core.atCoins (coinsOf C.p C.c C.e) fun ξ =>
    match runKept C S with
    | .error err => .reject s!"proofs: {err}"
    | .ok r =>
      match OneStage.checkAt C.p.base C.c (some ξ) { C.e.base with verdicts := r.out } with
      | .accept pub prof => if r.all then .accept pub prof else .reject "proofs: a session's verifier did not accept"
      | .reject why => .reject why
      | .undecided why => .undecided why

/-- The per-member form's stream on the members `ms` and the statements `sts`. -/
def membersStream (p : Params) (c : OneStage.Commitment) (e : Evidence) (ms : Array Member)
    (sts : Array StatementFiles) : Core.Stream (Core.Outcome OneStage.Public Core.Integrity) :=
  let C := Ctx.of p c e ms sts
  let plan := planOf ms sts
  { State := State
    tables := pinned
    plan := plan.map (·.map (·.1))
    init := State.init C ((plan.toOption.getD #[]).map (·.2))
    step := step C
    finish := finish C }

/-- The digests the one statement's form names, each once: its circuit and public files, then each session's record
and rep files. -/
def oneNames (p : Params) (e : Evidence) : V (Array ByteArray) := do
  let names := #[p.own.circuit, p.own.pub] ++ e.sessions.flatMap fun s => #[s.record] ++ s.reps.map (·.2)
  let mut out : Array ByteArray := #[]
  for d in names do
    let b ← digestBytes d "plan"
    if !out.contains b then out := out.push b
  pure out

/-- **Every other form's stream**: it keeps every blob and is `checkBlobs` on them (`Core.Stream.whole`). -/
def keepAll (p : Params) (c : OneStage.Commitment) (e : Evidence) (plan : V (Array ByteArray)) :
    Core.Stream (Core.Outcome OneStage.Public Core.Integrity) :=
  Core.Stream.whole pinned plan (checkBlobs p c e)

def stream (p : Params) (c : OneStage.Commitment) (e : Evidence) :
    Core.Stream (Core.Outcome OneStage.Public Core.Integrity) :=
  match p.members, e.statements with
  | some ms, some sts => membersStream p c e ms sts
  | none, none => keepAll p c e (oneNames p e)
  | _, _ => keepAll p c e (pure #[])

def protocol : Core.StreamProtocol :=
  { OneStageZk.protocol with stream }

/-- **The streamed check decides what `checkBlobs` decides on every feed in its order**: the statement
`Core.StreamProtocol.run_accept` reads to make the streamed accept `checkBlobs`' accept. `keepAll_sound` proves it
for every form but the per-member one, and `Spot.OneStageZkStreamKept` for that one and so for all (`streamSound`). -/
def StreamSound : Prop := protocol.Sound

/-- **Every form but the per-member one's stream is `checkBlobs` on the blobs fed**, in any order. -/
theorem keepAll_sound (p : Params) (c : OneStage.Commitment) (e : Evidence) (plan : V (Array ByteArray))
    (bs : List ByteArray) :
    (keepAll p c e plan).finish ((keepAll p c e plan).steps bs) = checkBlobs p c e (Core.Blobs.ofArray bs.toArray) :=
  Core.Stream.whole_finish _ _ _ bs

end Spot.OneStageZk.Streamed
