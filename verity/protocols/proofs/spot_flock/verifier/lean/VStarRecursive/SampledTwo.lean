/-
SPOT's two-stage form (`verity/sampled-proofs/v1`, params and evidence `{"form": "two-stage", ...}`): a boundary
registration sampled once on the challenger's coins (the first stage, its replay blocks), the interior registration
over the sampled replay blocks' proof blocks sampled once more on a challenger session of its own (the second stage),
the second stage's sessions verified as one-stage v2's per-member form verifies its members' sessions, and the stream's
run of record over both stages (`Spot.Run.run`), whose second stage holds the sessions' statements to the verdicts. It
refuses last, behind one switch (`FORM`, which is `Spot.SecondStage.FORM`): every check runs, then a two-stage audit is
refused while the two-stage form is outside the soundness theorem (`checkBlobsAt_refuses`). Nothing reads this module: `VStar.Recursive.Sampled` refuses the form by name.

* **params**: exactly `{"boundary", "challenger", "delta", "interior", "outputs", "public", "rule", "values"}`, the
  auditor's, fixed before any coin:
  * `boundary`: the boundary registration's parameters, as the stream's run reads them (`OneStage.paramsParsed`); its
    program's proof blocks under its query are one stratum, the replay block's one Definition (`ONE_DEFINITION`), so a
    share of replay blocks by count is their share by weight; it names the challenger at the interior's key
    (`ONE_KEY`), so both stages' coins are read under one key, and a coin family other than the interior's
    (`SAME_FAMILY`), so each stage's coins are a family of their own;
  * `rule`: the selection, two-stage only (`Select.Rule.twoStage`);
  * `values` and `public`: the Spec's boundary values (`{"name", "registered"}`) and the items both registrations
    show; `challenger`: the challenger's name its receipts carry; `delta`: the audit records' δ;
  * `outputs`: the check's public outputs, empty (`OUTPUTS_REFUSED`): no verdict this form computes reports any;
  * `interior`: exactly `{"members", "params", "replay", "values"}`: the interior registration's parameters, their
    `proofs` the stream's (`{"templates"}`, each template's circuit, reference header and registered reads,
    `Run.templateCircuits`); the replay block's program object; the Spec's interior values; and the interior audit's
    members (one-stage v2's `proofs.members`, `OneStageZk.membersOf`), each member's circuit one a template names
    (`MEMBER_CIRCUIT`). The interior audit's parameters are the interior's with `proofs` its members
    (`OneStageZk.membersParamsOfJson`).
* **commitment**: the boundary registration's record (`OneStage.commitmentOf`).
* **evidence**: exactly `{"coins", "interior", "received", "record", "source", "statements"}`: the first stage's coins'
  evidence, its receipt's place in the log and the bytes it read (hex); the interior audit's evidence in v2's
  per-member form (`OneStageZk.membersEvidenceOf`), whose sessions, statement, proved units, instances, served units,
  session roots and coins are the second stage's; the interior registration's record; and the second-stage sessions'
  statements the second-stage statement check reads. Each stage's coin evidence names no key
  (`Challenger.DRAW_FORM`): both are read under the parameters' challenger key, each stage's as coins of the family
  its registration's parameters name. The copy of main's first-parent history they are read under is the first
  stage's evidence's own, as one-stage v2 reads its evidence's, and the second stage's must be the same copy
  (`Challenger.COINS_MAIN`); the verifier's own copy, as a parameter, is the open fix. The first stage's coin
  evidence is never `null`, on one window as on two (`FIRST_COINS_NULL`): a two-stage audit's first stage reads coins.
* **check** (`checkBlobsAt`), in order:
  1. the interior audit's commitment (`OneStage.commitmentOf` of the interior record) and its coins
     (`OneStageZk.coinsOf`);
  2. the second stage's sessions, as v2's per-member form runs them (`OneStageZk.membersRun`: the statements held as
     public structure, each session's coins one tree of the interior's challenger session, closed after its
     evidence, the trees opened in order, the members laid over the registration, each session verified on its
     member's files); every session must accept;
  3. each verdict stamped with what the second-stage statement check reads of it, as `verity-verify-proof verify` writes it
     and the stream's driver hands it on (`stamped`): the delivered sessions, the session's public header's and
     registered reads' digests, its sample and one table;
     neither stage's bytes run past one coin's (`stagesFit`, `STAGE_PAST`): the first stage's bytes are read as one
     coin's first bytes and the second stage's as its coins' in order, so bytes past a coin's are none of theirs;
  4. the stream's run of record over both stages (`Run.run` of `inputOf`): the boundary audit over the replay
     sample at the boundary's session roots, and the interior audit over the second sample with the second-stage
     statement check, at the rest; it reads each registration's record as the verifier's own parser reads the
     record's canonical text, which must print back to that text (`readBack`, `RECORD_READ`);
     the evidence's first-stage bytes are exactly the ones the run's first stage read
     (`sourceRead`, `SOURCE_READ`), since the run holds only those to the challenger's coins; and on two windows the
     first session's log shows no coin of the second stage's family after the interior's digest (`interiorUnshown`,
     `INTERIOR_SHOWN`); and on two windows the stages' sessions are in order on one ledger (`stagesOrdered`): the
     first stage's copy of the auditor's ledger is a prefix of the second stage's (`LEDGER_PREFIX`), and on the
     second's, read under the parameters' challenger key, the first stage's session's `end` line has a lower `seq`
     than the second stage's session's `reserve` line (`STAGES_ORDER`); and on two windows no line of the first
     session's log is a coin tree's (`firstTreeless`, `FIRST_TREE`): the first stage's session only samples its
     sample, so every coin tree of the play is on the second stage's session, where the interior audit reads them
     in order from its log's first;
  5. the interior audit's verdict at its coins under the second stage's sampling design (`secondCheck`), which gives
     the public output and the integrity profile;
  6. last, the switch (`finish`): the run must accept, and the form is refused while `FORM` is off.

**The last step** (`secondCheck`, step 5) is the form's own variant of `OneStage.checkAt`, since the interior
registration is sampled under two sampling designs: its record's, a subset of all its `m` proof blocks
(`subsetDesign`), which R1 holds it to as the run of record does, and the second stage's, which its sample is sampled
under. It reads the sample the coins `ξ₂` give under the second stage's design, every coin opened read
(`OneStage.coinsReason`), then R1–R8 at the subset design, the evidence's sample held to the second stage's design
(`Draw.checkDraw`), the record's checks over its proved units (`OneStage.recordChecks`), the verdicts
(`OneStage.verdictOf`), and the bound under the second stage's design. That design is computed, never read from the
evidence (`secondDesignOf`): `Select.secondLaw` of the first stage's sample (the proved units of the sample the run of
record decided) over the replay block's cells under the interior's query (`Select.replayCells`) at the rule's counts,
the function and the arguments the run of record's second stage samples by. Its refusals carry `OneStage.check`'s
step prefixes, so a reason names the same step in both forms.
-/
module

public import VStarRecursive.Protocol
public import Spot.Run
meta import VStarRecursive.Protocol
meta import Spot.Run

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Flock (V)
open Spot

def TWO_STAGE : String := "two-stage"

/-- **The switch**: the two-stage form accepts only where `Spot.SecondStage.FORM` lets the second-stage statement check
accept, so one change flips both. -/
def FORM : Bool := SecondStage.FORM

theorem form_off : FORM = false := rfl

/-- The refusal of a boundary population whose replay blocks are not one Definition's. -/
def ONE_DEFINITION : String :=
  "the boundary program's proof blocks under its query are not one stratum: the two-stage form's replay blocks are " ++
    "one Definition's, so a share of them by count is a share by weight"

/-- The refusal of a boundary and an interior whose parameters name two challenger keys, or none. -/
def ONE_KEY : String :=
  "the boundary and the interior name one challenger key: both stages' coins are read under the run's one key"

/-- The refusal of parameters whose two stages name one coin family (`familiesApart`). -/
def SAME_FAMILY : String :=
  "the boundary and the interior name one coin family: each stage's coins are a family of their own, released " ++
    "after their own registration's digest"

/-- The stages' coin families as the check reads them (`inputOf`), the boundary's `b` and the interior audit's `i`:
apart when both are named and differ. -/
def familiesApart (b i : Option String) : Bool := b.isSome && b != i

#guard familiesApart (some "draw/0") (some "draw/second/0")
#guard !familiesApart (some "draw/0") (some "draw/0")
#guard !familiesApart none none

/-- The refusal of public outputs: no verdict this form computes reports any. -/
def OUTPUTS_REFUSED : String :=
  "the two-stage form's check has no public outputs: the second stage's verdicts report none"

/-- The refusal of a member whose circuit no template names (the driver's `pins`). -/
def MEMBER_CIRCUIT (m : String) : String :=
  s!"member {m}'s circuit is not a circuit the interior's proofs name for a template"

structure Params where
  boundary : OneStage.Params
  rule : Select.Rule
  values : Array Service.Value
  declared : Array String
  challenger : String
  delta : Json
  interior : OneStage.Params
  circuits : Array (String × String × Json × Json)
  replay : Spot.Program
  interiorValues : Array Service.Value
  audit : OneStageZk.Params
  members : Array OneStageZk.Member

structure Evidence where
  received : String
  source : ByteArray
  coins : Json
  interior : OneStageZk.Evidence
  statements : Array OneStageZk.StatementFiles
  record : Json
  stmts : Array Json

private def keysAre (j : Json) (ks : Array String) (what : String) : V Unit :=
  if keys? j != some (sortStrings ks) then throw s!"the keys of {what} are not {", ".intercalate (sortStrings ks).toList}"
  else pure ()

private def without (j : Json) (k : String) : List (String × Json) :=
  ((entries j).filter (·.1 != k)).toList

/-- The Spec's values, `{"name", "registered"}` each. -/
def valuesOf (j : Json) (what : String) : V (Array Service.Value) := do
  let .arr vs := j | throw s!"{what} is not a list"
  vs.mapM fun v => match getD v "name", getD v "registered" with
    | .str name, .bool registered => pure ({ name, registered } : Service.Value)
    | _, _ => throw s!"a value of {what} is not a string name with a boolean"

/-- Parameters as the stream's run reads them (`OneStage.paramsParsed`). -/
def paramsAt (j : Json) : V OneStage.Params := do
  OneStage.paramsParsed (← Core.canonicalJson (← Flock.canon j).toUTF8)

/-- The form's rest: the object less its `form`, which must be `two-stage`. -/
def split (j : Json) (what : String) : V Json := do
  let .obj _ := j | throw s!"the {what} is not an object"
  let .str form := getD j "form" | throw s!"the {what} names no form"
  if form != TWO_STAGE then throw s!"the {what} name the form {form}, not \"{TWO_STAGE}\""
  pure (Json.mkObj (without j "form"))

/-- The parameters (module docstring, **params**). -/
def paramsOfJson (j : Json) : V Params := do
  let j ← split j "parameters"
  keysAre j #["boundary", "challenger", "delta", "interior", "outputs", "public", "rule", "values"] "the parameters"
  let .arr os := getD j "outputs" | throw "outputs is not a list"
  if !os.isEmpty then throw OUTPUTS_REFUSED
  let rule ← Select.Rule.ofJson (getD j "rule")
  let .twoStage _ _ _ := rule | throw "the rule is not two-stage"
  let .str challenger := getD j "challenger" | throw "challenger is not a string"
  let .arr ds := getD j "public" | throw "public is not a list"
  let declared ← ds.mapM fun | .str s => pure s | _ => throw "a declared item is not a string"
  let bp ← paramsAt (getD j "boundary")
  if (Select.strataSizes bp.verifier.program bp.verifier.query).length != 1 then throw ONE_DEFINITION
  let it := getD j "interior"
  keysAre it #["members", "params", "replay", "values"] "the interior's parameters"
  let ip := getD it "params"
  let key := getD (getD (getD j "boundary") "challenger") "key"
  if key matches .null then throw ONE_KEY
  if !jeq key (getD (getD ip "challenger") "key") then throw ONE_KEY
  let interior ← paramsAt (Json.mkObj (without ip "proofs"))
  let circuits ← Spot.Run.templateCircuits (getD ip "proofs") interior.verifier.query
  let .obj _ := getD it "replay" | throw "the interior's replay is not a program object"
  let replay ← Spot.Program.parse (← Flock.canon (getD it "replay")).toUTF8
  let audit ← OneStageZk.membersParamsOfJson
    (Json.mkObj (without ip "proofs" ++ [("proofs", Json.mkObj [("members", getD it "members")])]))
  if !familiesApart (bp.challenger.map (·.2)) (audit.base.challenger.map (·.2)) then throw SAME_FAMILY
  let some members := audit.members | throw "the interior audit names no members"
  for m in members do
    if !circuits.any (·.2.1 == m.own.circuit) then throw (MEMBER_CIRCUIT m.name)
  pure { boundary := bp, rule, values := ← valuesOf (getD j "values") "values", declared, challenger,
         delta := getD j "delta", interior, circuits, replay,
         interiorValues := ← valuesOf (getD it "values") "the interior's values", audit, members }

def FIRST_COINS_NULL : String :=
  "the first stage's coin evidence is null: a two-stage audit's first stage reads the challenger's coins, on one " ++
    "window as on two"

/-- The evidence (module docstring, **evidence**). -/
def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← split (← Core.canonicalJson b) "evidence"
  keysAre j #["coins", "interior", "received", "record", "source", "statements"] "the evidence"
  if getD j "coins" matches .null then throw FIRST_COINS_NULL
  let .str received := getD j "received" | throw "received is not a string"
  let .str source := getD j "source" | throw "source is not hex"
  let source ← Flock.unhex source
  let .obj _ := getD j "record" | throw "the interior registration's record is not an object"
  let .arr stmts := getD j "statements" | throw "statements is not a list"
  let interior ← OneStageZk.membersEvidenceOf (← Flock.canon (getD j "interior")).toUTF8
  let some statements := interior.statements | throw "the interior audit's evidence names no statements"
  pure { received, source, coins := getD j "coins", interior, statements, record := getD j "record",
         stmts }

/-- The ports of the registration `record`'s roots. -/
def portsOf (record : Json) : Array String :=
  match getD record "roots" with
  | .arr rs => rs.filterMap fun r => match getD r "port" with | .str p => some p | _ => none
  | _ => #[]

/-- The session roots split at the boundary registration's ports, as the stream's two-stage outcome splits them: the
boundary's (null for none) and the rest (null where the evidence names none). -/
def rootsSplit (boundary : Json) (roots : Option Json) : Json × Json :=
  match roots with
  | none => (.null, .null)
  | some r =>
    let ports := portsOf boundary
    let (mine, rest) := (entries r).partition fun (p, _) => ports.contains p
    (if mine.isEmpty then .null else Json.mkObj mine.toList, Json.mkObj rest.toList)

/-- **Each verdict as the second-stage statement check reads it** (check, 3): the verdicts in session order with the
delivered sessions, and what `verity-verify-proof verify` names of the session it checked: its public file's header's digest,
its registered reads' (where its member reads any), its sample and one table. A refusal where a member's public file, a
session's record or a header is not on the blobs or not readable. -/
def stamped (p : Params) (e : Evidence) (bs : Core.Blobs) (out : Array (String × Json)) :
    V (Array (String × Json)) := do
  let named := e.statements.flatMap fun st => st.sessions.map fun s => (st.member.getD "", s)
  if named.size != out.size then throw "the verdicts are not one a session"
  let sessions := Json.arr (e.interior.base.sessions.map Json.str)
  (named.zip out).mapM fun ((member, s), (k, v)) => do
    let some m := p.members.find? (·.name == member) | throw s!"no member {member}"
    let header ← FilePart.header (← bs.get m.own.pub s!"member {member}'s public file")
    let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
      | throw s!"session {s.name}'s record is not UTF-8"
    let reads ← match m.own.registered with
      | some r => pure [("registered_sha512", Json.str (← canonSha512 r))]
      | none => pure []
    let sample := match Flock.Witness.recordDraw record with
      | some d => [("unit_draw", d)]
      | none => []
    let fields := [("sessions", sessions), ("public_header_sha512", Json.str (← canonSha512 header)),
      ("session_tables", Lean.toJson (1 : Nat))] ++ reads ++ sample
    pure (k, fields.foldl (fun acc (f, x) => acc.setObjVal! f x) v)

/-- **The stream's run of record over both stages** (check, 4), the input `Run.run` reads: the boundary's parameters,
record, rule, values, declared items, challenger, receipt `received₁`'s place and the bytes `ξ` it read, no closure, the
second stage's sessions and the verdicts `vs`, no public outputs, the interior audit's statement, δ, the boundary's
session roots and its coins; and the interior's parameters, replay block, values, record, receipt's place `received₂`,
the bytes `ξ₂` it read, its coins, proved units, instances, served units, session roots, the statements and the
templates' circuits. As the stream's two-stage outcome hands its run of record the same (`_audit_two`). The run reads
both stages' coins under the interior audit's challenger key (`key`, the boundary's by `ONE_KEY`), the first stage's
as coins of the family the boundary's parameters name and the second stage's as coins of the interior's (`family`),
on the first stage's evidence's copy of main's first-parent history (`main`; module docstring, **evidence**). -/
def inputOf (p : Params) (c : OneStage.Commitment) (ξ ξ₂ : ByteArray) (e : Evidence) (received₂ : String)
    (vs : Array (String × Json)) : Spot.Run.Input :=
  let b := e.interior.base
  let (bRoots, iRoots) := rootsSplit c.record b.sessionRoots
  { params := p.boundary, record := c.record, rule := p.rule, values := p.values, declared := p.declared,
    challenger := p.challenger, received := e.received, source := ξ, closure := none,
    sessions := b.sessions.map Json.str, verdicts := vs, outputs := #[], statement := b.statement, delta := p.delta,
    provedUnits := none, instances := none, served := Json.null, sessionRoots := bRoots, coins := e.coins,
    family := p.boundary.challenger.map (·.2), key := p.audit.base.challenger.map (·.1),
    main := getD e.coins "main",
    interior := some
      { params := p.interior, replay := p.replay, values := p.interiorValues, record := e.record,
        received := received₂, source := ξ₂, coins := b.coins, family := p.audit.base.challenger.map (·.2),
        provedUnits := some (b.provedUnits.map Int.ofNat), instances := some (Int.ofNat b.instances),
        served := Lean.toJson b.served, sessionRoots := iRoots, statements := some e.stmts,
        circuits := p.circuits } }

def RECORD_READ : String :=
  "a registration's record is not the one the verifier's parser reads from its canonical text: the run of record " ++
    "reads each record as that parser reads it"

/-- **A registration's record as the verifier's own parser reads it** (check, 4): the record's canonical text
(`Flock.canon`) as `Flock.JsonParse.parse` reads it, which must print back to that text (`RECORD_READ`). The run of
record reads this value, so an accepted audit's records are values of a parser the kernel unfolds, each with the
canonical text it was read from. `Flock.JsonParse.parse` returns the value `Lean.Json.parse` does, so it is the
record the form parsed. -/
def readBack (record : Json) : V Json := do
  let s ← Flock.canon record
  let j ← Flock.JsonParse.parse s
  let s' ← Flock.canon j
  if s' == s then pure j else throw RECORD_READ

/-- Check steps 1 to 3: the interior audit's commitment and the bytes its sample read, its sessions' run, every
session accepting, and the run of record's input over the stamped verdicts, each registration's record as the
verifier's parser reads it (`readBack`). -/
def inputAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    V (OneStage.Commitment × ByteArray × OneStageZk.Run × Spot.Run.Input) := do
  let c₂ ← OneStage.commitmentOf (← Flock.canon e.record).toUTF8
  let ξ₂ ← OneStageZk.coinsOf p.audit c₂ e.interior
  let r ← OneStageZk.membersRun p.audit p.members c₂ e.interior e.statements bs
  if !r.all then throw "proofs: a session's verifier did not accept"
  let vs ← stamped p e bs r.out
  let .str received₂ := getD e.interior.base.receipt "received" | throw "the interior's receipt names no place"
  let j₁ ← readBack c.record
  let j₂ ← readBack e.record
  pure (c₂, ξ₂, r, inputOf p { c with record := j₁ } ξ ξ₂ { e with record := j₂ } received₂ vs)

def SOURCE_READ : String :=
  "the evidence's first-stage bytes run past what the first stage read: the run holds only the bytes it read to the " ++
    "challenger's coins, so the evidence carries exactly those"

/-- **The first stage's bytes are the ones it read** (check, 4): a decided run reports how many bytes of `ξ` its first
stage read (`read`), and `ξ` has none past them (`SOURCE_READ`). A run that reports none is refused by its own symbol
(check, 6). -/
def sourceRead (ξ : ByteArray) (run : Json) : V Unit :=
  match Spot.nat? (getD run "read") with
  | some read => if read < ξ.size then .error SOURCE_READ else .ok ()
  | none => .ok ()

/-- One coin's bytes: 65,536 elements of 16 bytes. -/
def STAGE_BYTES : Nat := 65536 * 16

def STAGE_PAST (stage : String) : String :=
  s!"the {stage} stage's bytes run past one coin's {STAGE_BYTES} bytes: a stage's sample reads one coin's bytes"

/-- **Neither stage's bytes run past one coin's** (check, 3): the first stage's bytes `ξ` and the second stage's `ξ₂`
are each at most `STAGE_BYTES` long (`STAGE_PAST`). A sampler that runs out of bytes samples fresh ones, and the coins'
check takes a read over any number of whole coins, so nothing else bounds them. -/
def stagesFit (ξ ξ₂ : ByteArray) : V Unit :=
  if STAGE_BYTES < ξ.size then .error (STAGE_PAST "first")
  else if STAGE_BYTES < ξ₂.size then .error (STAGE_PAST "second")
  else .ok ()

def INTERIOR_SHOWN : String :=
  "on two windows the first session's log shows coins of the interior's family released after the interior " ++
    "registration's digest: the second stage's coins are its own session's alone"

/-- Whether the roots a session's log shows, by family (`Challenger.Roots`), show a coin of family `label` released
after `after`. -/
def interiorShown (roots : Challenger.Roots) (label : String) (after : ByteArray) : Bool :=
  (((roots.find? (·.1 == label)).map (·.2)).getD []).contains after

/-- **On two windows, the first session releases none of the second stage's coins** (check, 4): where the interior
audit's window is not the boundary registration's, the first session's log (the session `e.coins` names on the
boundary's window under the interior audit's challenger key, `Challenger.trustedChallengePhaseOf`) shows no coin of the
second stage's family, the one the interior audit's parameters name, released after the interior registration's
digest (`INTERIOR_SHOWN`), so the coins of that family after that digest are the second session's alone. The
interior audit's parameters always name a challenger (`OneStageZk.NO_CHALLENGER`); where the boundary names no window
the run of record refuses a first stage that read any byte (`Run.coinsCode`), and this adds nothing. -/
def interiorUnshown (p : Params) (e : Evidence) : V Unit :=
  match p.audit.base.challenger, p.boundary.verifier.windowRoot with
  | some (k, label), some w =>
    if some w == p.audit.base.verifier.windowRoot then .ok () else do
      let after ← Registration.digest e.record
      let (_, roots) ← Challenger.trustedChallengePhaseOf (getD e.coins "ledger") (Json.mkObj [("root", Json.str w)])
        k (getD e.coins "pins") (getD e.coins "main") (getD e.coins "session")
      if interiorShown roots label after then .error INTERIOR_SHOWN else .ok ()
  | _, _ => .ok ()

def LEDGER_PREFIX : String :=
  "the first stage's copy of the auditor's ledger is not a prefix of the second stage's: both stages' sessions are " ++
    "lines of one ledger"

def STAGES_ORDER : String :=
  "the first stage's session's end line is not before the second stage's session's reserve line on the auditor's " ++
    "ledger: the second stage's session is reserved after the first stage's has ended"

/-- Why the first stage's session, ended at `ended`, and the second stage's, reserved at `reserve`, are not in order
on one ledger (`STAGES_ORDER`): the first has no end line, the second no reserve line, or the end's `seq` is not
below the reserve's. -/
def orderReason (ended reserve : Option Challenger.LedgerEntry) : Option String :=
  match ended, reserve with
  | some e, some r => if e.seq < r.seq then none else some STAGES_ORDER
  | _, _ => some STAGES_ORDER

/-- **On two windows, the stages' sessions are in order on one ledger** (check, 4): the first stage's copy of the
auditor's ledger is a prefix of the second stage's (`Challenger.extendsHead`, `LEDGER_PREFIX`); on the second's, read
under the interior audit's challenger key (`Challenger.checked`), the session opened for the boundary's window has an
`end` line, the session opened for the interior audit's window a `reserve` line, and the end's `seq` is below the
reserve's (`orderReason`). On one window both stages read one session, and this adds nothing. -/
def stagesOrdered (p : Params) (e : Evidence) : V Unit :=
  match p.audit.base.challenger, p.boundary.verifier.windowRoot, p.audit.base.verifier.windowRoot with
  | some (k, _), some w₁, some w₂ =>
    if w₁ == w₂ then .ok () else do
      let (.arr first, .arr second) := (getD e.coins "ledger", getD e.interior.base.coins "ledger")
        | throw LEDGER_PREFIX
      if !Challenger.extendsHead first second then throw LEDGER_PREFIX
      let (wins, _) ← Challenger.checked (.arr second) k
      let ended := (wins.find? (·.root == w₁)).bind (·.ended)
      let reserve := (wins.find? (·.root == w₂)).bind fun w =>
        (second.filterMap Challenger.ledgerEntry?).find? fun l => l.kind == "reserve" && l.session == w.opened.session
      match orderReason ended reserve with
      | some why => throw why
      | none => .ok ()
  | _, _, _ => .ok ()

def FIRST_TREE : String :=
  "on two windows the first session's log holds a coin tree's line: the first stage's session only samples its " ++
    "sample, and the play's coin trees are the second stage's session's"

/-- Whether the release log lines `ls` hold a coin tree's line (`Challenger.treeLine`). -/
def treeShown (ls : List String) : Bool := ls.any Challenger.treeLine

/-- The lines of the first stage's session's release log, as its coin evidence carries them. -/
def firstLines (e : Evidence) : V (List String) := do
  let .arr ls := getD (getD e.coins "session") "log" | throw "the session's log is not a list of lines"
  ls.toList.mapM fun | .str l => pure l | _ => throw "a line of the session's log is not a string"

/-- **On two windows, the first session's log holds no coin tree's line** (check, 4): where the interior audit's
window is not the boundary registration's, no line of the first stage's session's release log (`firstLines`) is a
coin tree's, one the challenger wrote when it sampled a tree or opened a round of one (`FIRST_TREE`). The interior
audit reads its sessions' coin trees in order from its own session's log's first, and the first stage's coin check
drops such lines unread (`Challenger.releasedRootsOf`), so nothing else refuses a first session that sampled one. A
first stage only samples its sample, so no honest play has such a line. On one window both stages read one session,
whose log holds the second stage's trees, and this adds nothing. -/
def firstTreeless (p : Params) (e : Evidence) : V Unit :=
  match p.boundary.verifier.windowRoot with
  | some w =>
    if some w == p.audit.base.verifier.windowRoot then .ok () else do
      let ls ← firstLines e
      if treeShown ls then .error FIRST_TREE else .ok ()
  | none => .ok ()

/-- **The second stage's sampling design** (check, 5), computed and never read: `Select.secondLaw` of the first stage's
sample, the proved units of the sample the run of record `run` decided, over the cells the replay block's program has
under the interior's query (`Select.replayCells`) at the rule's counts (`Select.Counts.over`), as the run of record's
second stage computes it. -/
def secondDesignOf (p : Params) (run : Json) : V Json := do
  let .twoStage _ _ k := p.rule | throw "the rule is not two-stage"
  let q := p.interior.verifier.query
  let ns ← Select.replayCells p.replay q
  let ks ← k.over (← Select.templateIds q) ns
  pure (Select.secondLaw ((Draw.proved (getD run "draw")).map Int.toNat) ns ks)

/-- The interior registration's sampling design as the run of record reads it: a subset of all its `m` proof blocks. -/
def subsetDesign (m : Nat) : Json := Json.mkObj [("law", "subset"), ("k", Lean.toJson m)]

/-- `secondCheck` past the coins: R1–R8 at the verifier `v`, the sample held to the second stage's sampling design
`design`, the record's checks, the verdicts, and the bound under `design` at δ `delta`, as `OneStage.check` has them. -/
def secondChecked (v : Registration.Verifier) (delta : Core.Ratio) (c : OneStage.Commitment) (m : Nat) (design : Json)
    (e : OneStage.Evidence) : Core.Outcome OneStage.Public Core.Integrity :=
  let rec_ := c.record
  let codes := Registration.check rec_ v
  if !codes.isEmpty then .reject s!"registration: {", ".intercalate codes.toList}" else
  let sampleCodes := Draw.checkDraw e.draw (getD rec_ "population") design
  if !sampleCodes.isEmpty then .reject s!"draw: {", ".intercalate sampleCodes.toList}" else
  let failed := OneStage.recordChecks rec_ (Flock.hexOf c.digest) e
  if !failed.isEmpty then .reject s!"the audit's checks failed: {", ".intercalate failed.toList}" else
  match OneStage.verdictOf e.verdicts with
  | .reject r => .reject r
  | .undecided r => .undecided r
  | .accept () () =>
  match Record.profileDraw design with
  | some prof =>
    let units := match getD e.draw "units" with | .arr us => us.filterMap nat? | _ => #[]
    let str := fun (k : String) => match getD rec_ k with | .str s => s | _ => ""
    .accept
      { drawn := units, proved := (Draw.proved e.draw).filterMap fun (i : Int) => if 0 ≤ i then some i.toNat else none,
        bound := Bound.boundOf m prof delta,
        perTemplate := (Bound.strataOf m prof).map (·.map fun s => (s.name, Bound.stratumBound s delta)) }
      { programs := #[str "program"], prescribed := #[], linked := false, covered := #[],
        levels := #[{ name := Record.LEVEL, query := some (str "partition"), size := m, draw := some prof }],
        delta, drawn := #[(Record.LEVEL, units.size)] }
  | none => .reject "the second stage's sampling design has no profile"

/-- **The interior audit's check at the second stage's sampling design** (check, 5; module docstring, **the last
step**): the interior registration's population `m`; the sample the coins `ξ₂` give under `design`
(`OneStage.coinsReason` at the verifier whose sampling design is `subsetDesign m`, so that the sample is `design`'s,
`Draw.flockLaw`), every coin opened read; then `secondChecked` at that verifier. -/
def secondCheck (v : Registration.Verifier) (delta : Core.Ratio) (c : OneStage.Commitment) (design : Json)
    (ξ₂ : ByteArray) (e : OneStage.Evidence) : Core.Outcome OneStage.Public Core.Integrity :=
  match nat? (getD c.record "population") with
  | none => .reject "registration: the interior record names no population"
  | some m =>
    match OneStage.coinsReason { v with law := subsetDesign m } design c.record e.draw ξ₂ with
    | some why => .reject s!"draw: coins: {why}"
    | none => secondChecked { v with law := subsetDesign m } delta c m design e

/-- Check steps 1 to 5: the interior audit's verdict at its coins, and the run of record's output, its second-stage
statement check at the switch `form`, neither stage's bytes past one coin's, its first stage having read every byte of
`ξ`, the first session releasing none of the second stage's coins, the stages' sessions in order on one ledger, the
first session's log holding no coin tree's line. -/
def prepare (form : Bool) (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    V (Core.Outcome OneStage.Public Core.Integrity × Json) := do
  let (c₂, ξ₂, r, i) ← inputAt p c ξ e bs
  stagesFit ξ ξ₂
  let run ← Spot.Run.run i form
  sourceRead ξ run
  interiorUnshown p e
  stagesOrdered p e
  firstTreeless p e
  let design ← secondDesignOf p run
  pure (secondCheck p.audit.base.verifier p.audit.base.delta c₂ design ξ₂ { e.interior.base with verdicts := r.out }, run)

/-- **The last step** (check, 6) at the switch `form`: the interior audit's verdict where it doesn't accept; where it
does, the run of record must accept, and then the form is refused while `form` is off. -/
def finishWith (form : Bool) (two : Core.Outcome OneStage.Public Core.Integrity) (run : Json) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match two with
  | .reject why => .reject s!"two-stage: the interior's audit: {why}"
  | .undecided why => .undecided s!"two-stage: the interior's audit: {why}"
  | .accept pub prof =>
    if !jeq (getD run "symbol") (Json.str "accept") then .reject s!"two-stage: the run of record: {run.compress}"
    else if form then .accept pub prof
    else .reject s!"two-stage: {SecondStage.FORM_WHY}"

/-- The last step at the switch (`FORM`). -/
def finish : Core.Outcome OneStage.Public Core.Integrity → Json → Core.Outcome OneStage.Public Core.Integrity :=
  finishWith FORM

/-- **The check at the first stage's bytes `ξ`, at the switch `form`** (module docstring, **check**). The proofs about
an accepted two-stage audit read it at `form = true`, so that nothing they prove rests on the switch being off. -/
def checkBlobsAtWith (form : Bool) (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence)
    (bs : Core.Blobs) : Core.Outcome OneStage.Public Core.Integrity :=
  match prepare form p c ξ e bs with
  | .error err => .reject s!"two-stage: {err}"
  | .ok (two, run) => finishWith form two run

/-- **The check at the first stage's bytes `ξ`**, at the switch (`FORM`). -/
def checkBlobsAt : Params → OneStage.Commitment → ByteArray → Evidence → Core.Blobs →
    Core.Outcome OneStage.Public Core.Integrity :=
  checkBlobsAtWith FORM

/-- The check at the bytes the evidence says the first stage read, which the run of record holds to the
challenger's coins (`Run.coinsHold`). -/
def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  checkBlobsAt p c e.source e bs

theorem finish_refuses (hF : FORM = false) (two : Core.Outcome OneStage.Public Core.Integrity) (run : Json)
    (pub : OneStage.Public) (prof : Core.Integrity) : finish two run ≠ .accept pub prof := by
  unfold finish finishWith
  split
  · simp
  · simp
  · split
    · simp
    · simp [hF]

/-- **The two-stage form accepts nothing while the switch is off**: whatever the parameters, the registration, the
bytes, the evidence and the blobs. -/
theorem checkBlobsAt_refuses (hF : FORM = false) (p : Params) (c : OneStage.Commitment) (ξ : ByteArray)
    (e : Evidence) (bs : Core.Blobs) (pub : OneStage.Public) (prof : Core.Integrity) :
    checkBlobsAt p c ξ e bs ≠ .accept pub prof := by
  unfold checkBlobsAt checkBlobsAtWith
  split
  · simp
  · exact finish_refuses hF _ _ _ _

theorem checkBlobs_refuses (hF : FORM = false) (p : Params) (c : OneStage.Commitment) (e : Evidence)
    (bs : Core.Blobs) (pub : OneStage.Public) (prof : Core.Integrity) : checkBlobs p c e bs ≠ .accept pub prof :=
  checkBlobsAt_refuses hF p c e.source e bs pub prof

def PROTOCOL : String := "verity/sampled-proofs/two-stage/v1"

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

#guard (paramsOfJson (Json.mkObj [("form", "one-stage")])).toOption.isNone
#guard match paramsOfJson (Json.mkObj [("form", TWO_STAGE), ("outputs", Json.arr #["y"])]) with
  | .error e => e.startsWith "the keys of the parameters"
  | .ok _ => false
#guard match evidenceOf "{\"form\":\"two-stage\"}".toUTF8 with
  | .error e => e.startsWith "the keys of the evidence"
  | .ok _ => false
#guard match evidenceOf ("{\"coins\":null,\"form\":\"two-stage\",\"interior\":{},\"received\":\"r\",\"record\":{}," ++
    "\"source\":\"\",\"statements\":[]}").toUTF8 with
  | .error e => e == FIRST_COINS_NULL
  | .ok _ => false
#guard orderReason (some { (default : Challenger.LedgerEntry) with seq := 2 })
  (some { (default : Challenger.LedgerEntry) with seq := 3 }) == none
#guard orderReason (some { (default : Challenger.LedgerEntry) with seq := 3 })
  (some { (default : Challenger.LedgerEntry) with seq := 3 }) == some STAGES_ORDER
#guard orderReason none (some { (default : Challenger.LedgerEntry) with seq := 3 }) == some STAGES_ORDER
#guard orderReason (some { (default : Challenger.LedgerEntry) with seq := 2 }) none == some STAGES_ORDER
#guard rootsSplit (Json.mkObj [("roots", Json.arr #[Json.mkObj [("port", "b.ingress")]])])
    (some (Json.mkObj [("b.ingress", "r1"), ("t/x", "r2")])) ==
  (Json.mkObj [("b.ingress", "r1")], Json.mkObj [("t/x", "r2")])
#guard rootsSplit (Json.mkObj [("roots", Json.arr #[])]) (some (Json.mkObj [("t/x", "r2")])) ==
  (Json.null, Json.mkObj [("t/x", "r2")])
#guard match finish (.reject "r") (Json.mkObj [("symbol", "accept")]) with
  | .reject why => why == "two-stage: the interior's audit: r"
  | _ => false
#guard match sourceRead (ByteArray.mk #[1, 2, 3]) (Json.mkObj [("read", Lean.toJson (2 : Nat))]) with
  | .error e => e == SOURCE_READ
  | .ok _ => false
#guard (sourceRead (ByteArray.mk #[1, 2]) (Json.mkObj [("read", Lean.toJson (2 : Nat))])).toOption.isSome
#guard (sourceRead (ByteArray.mk #[1, 2]) (Json.mkObj [("step", "select")])).toOption.isSome
#guard (stagesFit (ByteArray.mk (Array.replicate STAGE_BYTES 0))
  (ByteArray.mk (Array.replicate STAGE_BYTES 0))).toOption.isSome
#guard match stagesFit (ByteArray.mk (Array.replicate (STAGE_BYTES + 1) 0)) ByteArray.empty with
  | .error why => why == STAGE_PAST "first"
  | .ok _ => false
#guard match stagesFit ByteArray.empty (ByteArray.mk (Array.replicate (STAGE_BYTES + 1) 0)) with
  | .error why => why == STAGE_PAST "second"
  | .ok _ => false
#guard treeShown ["{\"event\":\"root\"}", "{\"event\":\"coin-tree\"}"]
#guard treeShown ["{\"event\":\"tree-open\"}"]
#guard !treeShown ["{\"event\":\"root\"}", "{\"event\":\"release\"}", "coin-tree"]
#guard !treeShown []
#guard (readBack (Json.mkObj [("b", Json.arr #[Json.str "x"]), ("a", Json.null)])).toOption ==
  some (Json.mkObj [("a", Json.null), ("b", Json.arr #[Json.str "x"])])
#guard (readBack (Json.num ⟨1, 1⟩)).toOption.isNone
-- an object whose tree holds one key twice, which no parser returns: its text reads back as one key
#guard match readBack (Json.obj ⟨⟨.inner 2 "a" (Json.num 1) (.inner 1 "a" (Json.num 2) .leaf .leaf) .leaf⟩⟩) with
  | .error why => why == RECORD_READ
  | .ok _ => false
#guard interiorShown [("boundary", [ByteArray.mk #[1]]), ("interior", [ByteArray.mk #[1], ByteArray.mk #[7]])]
  "interior" (ByteArray.mk #[7])
#guard !interiorShown [("interior", [ByteArray.mk #[1]])] "interior" (ByteArray.mk #[7])
#guard !interiorShown [("boundary", [ByteArray.mk #[7]])] "interior" (ByteArray.mk #[7])

end VStar.Recursive.SampledTwo
