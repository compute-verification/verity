/-
SPOT's two-stage form (`verity/sampled-proofs/v1`, params and evidence `{"form": "two-stage", ...}`): a boundary
registration sampled once on the challenger's coins (the first stage, its replay blocks), the interior registration
over the sampled replay blocks' proof blocks sampled once more on a challenger session of its own (the second stage),
the second stage's sessions verified as one-stage v2's per-member form verifies its members' sessions, and the stream's
run of record over both stages (`Spot.Run.run`), whose second stage holds the sessions' statements to the verdicts. It
refuses last, behind one switch (`FORM`, which is `Spot.SecondStage.FORM`): every check runs, then a two-stage audit is
refused while the two-stage form is outside the soundness theorem (`checkBlobsAt_refuses`). `VStar.Recursive.Sampled` reads the form and runs this check.

* **params**: exactly `{"boundary", "challenger", "delta", "interior", "outputs", "public", "rule", "values"}`, the
  auditor's, fixed before any coin:
  * `boundary`: the boundary registration's parameters, as the stream's run reads them (`OneStage.paramsParsed`); its
    program's proof blocks under its query are one stratum, the replay block's one Definition (`ONE_DEFINITION`), so a
    share of replay blocks by count is their share by weight; it names the challenger at the interior's key
    (`ONE_KEY`), so both stages' coins are read under one key;
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
  statements the second-stage statement check reads.
* **check** (`checkBlobsAt`), in order:
  1. the interior audit's commitment (`OneStage.commitmentOf` of the interior record) and its coins
     (`OneStageZk.coinsOf`);
  2. the second stage's sessions, as v2's per-member form runs them (`OneStageZk.membersRun`: the statements held as
     public structure, each session's coins one tree of the interior's challenger session, closed after its
     evidence, the trees opened in order, the members laid over the registration, each session verified on its
     member's files); every session must accept;
  3. each verdict stamped with what the second-stage statement check reads of it, as `flock-verify verify` writes it
     and the stream's driver hands it on (`stamped`): the delivered sessions, the session's public header's and
     registered reads' digests, its sample and one table;
     neither stage's bytes run past one coin's (`stagesFit`, `STAGE_PAST`): the first stage's bytes are read as one
     coin's first bytes and the second stage's as its coins' in order, so bytes past a coin's are none of theirs;
  4. the stream's run of record over both stages (`Run.run` of `inputOf`): the boundary audit over the replay
     sample at the boundary's session roots, and the interior audit over the second sample with the second-stage
     statement check, at the rest; the evidence's first-stage bytes are exactly the ones the run's first stage read
     (`sourceRead`, `SOURCE_READ`), since the run holds only those to the challenger's coins; and on two windows the
     first session's log shows no coin of the second stage's family after the interior's digest (`interiorUnshown`,
     `INTERIOR_SHOWN`);
  5. the interior audit's verdict at its coins (`OneStage.checkAt`, as v2's per-member form decides it), which gives
     the public output and the integrity profile;
  6. last, the switch (`finish`): the run must accept, and the form is refused while `FORM` is off.
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

/-- The refusal of public outputs: no verdict this form computes reports any. -/
def ONE_KEY : String :=
  "the boundary and the interior name one challenger key: both stages' coins are read under the run's one key"

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
  let some members := audit.members | throw "the interior audit names no members"
  for m in members do
    if !circuits.any (·.2.1 == m.own.circuit) then throw (MEMBER_CIRCUIT m.name)
  pure { boundary := bp, rule, values := ← valuesOf (getD j "values") "values", declared, challenger,
         delta := getD j "delta", interior, circuits, replay,
         interiorValues := ← valuesOf (getD it "values") "the interior's values", audit, members }

/-- The evidence (module docstring, **evidence**). -/
def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← split (← Core.canonicalJson b) "evidence"
  keysAre j #["coins", "interior", "received", "record", "source", "statements"] "the evidence"
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
delivered sessions, and what `flock-verify verify` names of the session it checked: its public file's header's digest,
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
    let draw := match Flock.Witness.recordDraw record with
      | some d => [("unit_draw", d)]
      | none => []
    let fields := [("sessions", sessions), ("public_header_sha512", Json.str (← canonSha512 header)),
      ("session_tables", Lean.toJson (1 : Nat))] ++ reads ++ draw
    pure (k, fields.foldl (fun acc (f, x) => acc.setObjVal! f x) v)

/-- **The stream's run of record over both stages** (check, 4), the input `Run.run` reads: the boundary's parameters,
record, rule, values, declared items, challenger, receipt `received₁`'s place and the bytes `ξ` it read, no closure, the
second stage's sessions and the verdicts `vs`, no public outputs, the interior audit's statement, δ, the boundary's
session roots and its coins; and the interior's parameters, replay block, values, record, receipt's place `received₂`,
the bytes `ξ₂` it read, its coins, proved units, instances, served units, session roots, the statements and the
templates' circuits. As the stream's two-stage outcome hands its run of record the same (`_audit_two`). -/
def inputOf (p : Params) (c : OneStage.Commitment) (ξ ξ₂ : ByteArray) (e : Evidence) (received₂ : String)
    (vs : Array (String × Json)) : Spot.Run.Input :=
  let b := e.interior.base
  let (bRoots, iRoots) := rootsSplit c.record b.sessionRoots
  { params := p.boundary, record := c.record, rule := p.rule, values := p.values, declared := p.declared,
    challenger := p.challenger, received := e.received, source := ξ, closure := none,
    sessions := b.sessions.map Json.str, verdicts := vs, outputs := #[], statement := b.statement, delta := p.delta,
    provedUnits := none, instances := none, served := Json.null, sessionRoots := bRoots, coins := e.coins,
    interior := some
      { params := p.interior, replay := p.replay, values := p.interiorValues, record := e.record,
        received := received₂, source := ξ₂, coins := b.coins,
        provedUnits := some (b.provedUnits.map Int.ofNat), instances := some (Int.ofNat b.instances),
        served := Lean.toJson b.served, sessionRoots := iRoots, statements := some e.stmts,
        circuits := p.circuits } }

/-- Check steps 1 to 3: the interior audit's commitment and the bytes its sample read, its sessions' run, every
session accepting, and the run of record's input over the stamped verdicts. -/
def inputAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    V (OneStage.Commitment × ByteArray × OneStageZk.Run × Spot.Run.Input) := do
  let c₂ ← OneStage.commitmentOf (← Flock.canon e.record).toUTF8
  let ξ₂ ← OneStageZk.coinsOf p.audit c₂ e.interior
  let r ← OneStageZk.membersRun p.audit p.members c₂ e.interior e.statements bs
  if !r.all then throw "proofs: a session's verifier did not accept"
  let vs ← stamped p e bs r.out
  let .str received₂ := getD e.interior.base.receipt "received" | throw "the interior's receipt names no place"
  pure (c₂, ξ₂, r, inputOf p c ξ ξ₂ e received₂ vs)

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
registration's window is not the boundary registration's, the first session's log (the session `e.coins` names on the
boundary's window, `Challenger.trustedChallengePhaseOf`) shows no coin of the second stage's family released after the
interior registration's digest (`INTERIOR_SHOWN`), so the coins of that family after that digest are the second
session's alone. Where a piece is missing or unreadable the run has already refused, and this adds nothing. -/
def interiorUnshown (p : Params) (e : Evidence) : V Unit :=
  match p.boundary.verifier.windowRoot, getD (getD e.interior.base.coins "segment") "label" with
  | some w, .str label =>
    if some w == p.interior.verifier.windowRoot then .ok () else do
      let after ← Registration.digest e.record
      let k ← Challenger.ChallengerKey.ofJson (getD e.coins "key")
      let (_, roots) ← Challenger.trustedChallengePhaseOf (getD e.coins "ledger") (Json.mkObj [("root", Json.str w)]) k
        (getD e.coins "pins") (getD e.coins "main") (getD e.coins "session")
      if interiorShown roots label after then .error INTERIOR_SHOWN else .ok ()
  | _, _ => .ok ()

/-- Check steps 1 to 5: the interior audit's verdict at its coins, and the run of record's output, its second-stage
statement check at the switch `form`, neither stage's bytes past one coin's, its first stage having read every byte of
`ξ`, the first session releasing none of the second stage's coins. -/
def prepare (form : Bool) (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    V (Core.Outcome OneStage.Public Core.Integrity × Json) := do
  let (c₂, ξ₂, r, i) ← inputAt p c ξ e bs
  stagesFit ξ ξ₂
  let run ← Spot.Run.run i form
  sourceRead ξ run
  interiorUnshown p e
  pure (OneStage.checkAt p.audit.base c₂ (some ξ₂) { e.interior.base with verdicts := r.out }, run)

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
#guard interiorShown [("boundary", [ByteArray.mk #[1]]), ("interior", [ByteArray.mk #[1], ByteArray.mk #[7]])]
  "interior" (ByteArray.mk #[7])
#guard !interiorShown [("interior", [ByteArray.mk #[1]])] "interior" (ByteArray.mk #[7])
#guard !interiorShown [("boundary", [ByteArray.mk #[7]])] "interior" (ByteArray.mk #[7])

end VStar.Recursive.SampledTwo
