/-
Sampled proofs' period, the audits' half, and the period's check (`verity/sampled-proofs/period/v1`, `protocol`): did
every root a lane committed at the auditor's challenger during a monitoring period pass sampled proofs' audit, on the
one sample the challenger gave it?

The period's check audits every window the ledger's half lists (`Spot.Period.windowsOf`) with the one-stage audit on
the proofs themselves (`OneStageZk`). The verifier holds one set of parameters for the whole monitoring period
(`AuditParams`): the ledger's half's, its copy of main's first-parent history (`main`), and `audit`, one-stage v2's
parameters less what a window sets. A case supplies nothing of the parameters that the check does not hold to the
ledger or to the registration.

**The commitment** is the auditor's ledger, its signed lines. **The evidence** (`Evidence`) is the host's pins log
and, for each window in the ledger's order, its challenger session's files and one case for each registration that
session sampled: the registration's record, its public file's digest, and one-stage v2's evidence for it, whose
`coins` name only the sample's segment. The ledger, main's history, the pins log and the session's files that
one-stage v2 reads a sample's coins from are the period's own (`evidenceAt`), never a case's.

**The check** (`checkBlobs`), in order, the first refusal the answer, each under its code (`Code`):

1. the ledger's half (`windowsOf`): `coin`, `early`, `missing`, `outside`, `period`;
2. the evidence holds as many windows' sessions as the ledger lists windows (`evidence`);
3. then each window, in the ledger's order (`windowAt`): its session's files are the ones the ledger sealed, run from
   a commit the pins log records on main's first-parent line, and its open withheld its coins (`coin`);
4. its session's signed log shows coins of the family released after exactly one registration, the coin after the
   window's own root aside (`sampledOn`): none is a root committed and never sampled (`missing`), and several are
   refused (`members`): only a window of one registration is covered, so a second registration on a session, another
   rank's or the same one registered again, is never passed over;
5. the evidence holds one case, of that registration (`evidence`);
6. the parameters at the window are one-stage v2's, and the case's public file shares no rows (`names`);
7. one-stage v2 accepts the case at those parameters (`audit`). A reject refuses the period; an undecided leaves the
   period undecided unless a later window refuses it.

On acceptance it publishes the lane and the monitoring period, each window with the registration audited and what
one-stage v2 published for it, and the lane's other sessions (`Public`); its profile is each audit's, in order.

**The parameters at a window** (`paramsAt`), key by key of one-stage v2's:

* `window_root`: the window's root, which the auditor's ledger opens the window's session on. Set by the check;
  `audit` naming one is refused.
* `challenger`: the period's own key and family (`AuditParams.ledger`'s `key` and `label`). Set by the check; `audit`
  naming one is refused.
* `log`: `audit`'s own log, then the records of the audits before this one in the ledger's order, each a registration
  the check audits. Set by the check from `audit`'s and its own earlier records, never the case's.
* `proofs.public`: the case's, the digest of the registration's public file. One-stage v2 holds the file to the
  registration's record (`OneStageZk.publicHeld`) and its rows to the header's roots (the session's verifier); `audit`
  naming one is refused. A public file whose header names `shared_rows` is refused (`SHARED_ROWS`): the check does
  not hold a file's references into the rows its members share to the registration.
* `seed_salt`, `day`, `declarations`: one window's own, read from its session or from the ledger, and not derived
  here: `audit` naming one is refused.
* every other key, `proofs.circuit` and `proofs.registered` among them: `audit`'s, the same at every window. A
  monitoring period whose windows differ in one of them is not covered: its windows are refused by one-stage v2 under
  the parameters they do not fit.
-/
module

public import Spot.Period
public import Spot.OneStageZk

@[expose] public section

namespace Spot.Period

open Lean (Json)
open Flock (V)

/-- The verifier's own parameters of a period's check (module docstring): the ledger's half's (`ledger`), the
challenger key's JSON as the verifier gave it (`keyJ`), the verifier's copy of main's first-parent history (`main`)
and one-stage v2's parameters less what a window sets (`audit`). -/
structure AuditParams where
  ledger : Params
  keyJ : Json
  main : Json
  audit : Json

/-- The keys of one-stage v2's parameters that the check sets at a window. -/
def WINDOW_KEYS : Array String := #["challenger", "window_root"]

/-- The keys of one-stage v2's parameters that are one window's own and that the check does not derive. -/
def UNDERIVED_KEYS : Array String := #["day", "declarations", "seed_salt"]

def AUDIT_FORM : String :=
  "the audit's parameters are one-stage v2's less what a window sets: an object with a log, proofs of exactly " ++
  "circuit and registered, and no challenger, window_root, day, declarations or seed_salt"

/-- One-stage v2's parameters at the window whose root is `root`, for the audit of the registration whose public
file's digest is `pub`, after the audits whose records are `before` (module docstring). -/
def paramsAt (p : AuditParams) (root pub : String) (before : Array Json) : Json :=
  let pr := getD p.audit "proofs"
  let log := match getD p.audit "log" with
    | .arr l => l
    | _ => #[]
  Json.mkObj (((entries p.audit).filter fun kv => kv.1 != "log" && kv.1 != "proofs").toList ++
    [("challenger", Json.mkObj [("key", p.keyJ), ("label", Json.str p.ledger.label)]),
     ("log", Json.arr (log ++ before)),
     ("proofs", Json.mkObj [("circuit", getD pr "circuit"), ("public", Json.str pub),
                            ("registered", getD pr "registered")]),
     ("window_root", Json.str root)])

/-- A root and a digest of the right form, at which `AuditParams.of` reads `audit` once. -/
def ZEROS : String := "".pushn '0' 128

/-- The parameters from their JSON, exactly `{"audit", "key", "label", "lane", "main", "start", "stop"}`: the ledger's
half's (`Params.of`), and `audit` of `AUDIT_FORM` that one-stage v2 reads at a window (`paramsAt` at `ZEROS`, read by
`OneStageZk.paramsOfJson`), so parameters one-stage v2 would refuse at every window are refused here. -/
def AuditParams.of (j : Json) : V AuditParams := do
  unless keys? j == some #["audit", "key", "label", "lane", "main", "start", "stop"] do
    throw "a period's parameters' keys are audit, key, label, lane, main, start, stop"
  let ledger ← Params.of (getD j "key") (getD j "label") (getD j "lane") (getD j "start") (getD j "stop")
  let audit := getD j "audit"
  let formed := (audit matches .obj _) && (getD audit "log" matches .arr _) &&
    keys? (getD audit "proofs") == some #["circuit", "registered"] &&
    (WINDOW_KEYS ++ UNDERIVED_KEYS).all fun k => (field? audit k).isNone
  unless formed do throw AUDIT_FORM
  let p : AuditParams := { ledger, keyJ := getD j "key", main := getD j "main", audit }
  let _ ← (OneStageZk.paramsOfJson (paramsAt p ZEROS ZEROS #[])).mapError (s!"audit: {·}")
  return p

/-! ## The commitment and the evidence -/

/-- The commitment: the auditor's ledger, its signed lines, with the SHA-512 of the bytes it came in. -/
structure Commitment where
  ledger : Json
  digest : ByteArray

def Commitment.of (b : ByteArray) : V Commitment := do
  let j ← Core.canonicalJson b
  unless j matches .arr _ do throw "the commitment is not the auditor's ledger, a list of its lines"
  return { ledger := j, digest := Flock.Sha512.hash b }

/-- One audit of a window's evidence: the registration's record, the digest of its public file, and one-stage v2's
evidence for it, whose `coins` name the sample's segment alone (`evidenceAt`). -/
structure Case where
  registration : Json
  pub : String
  evidence : Json

/-- A window's evidence: its challenger session's files `{close, hellos, log, record}` and its cases. -/
structure WindowEvidence where
  session : Json
  audits : Array Case

/-- The evidence: the host's pins log and each window's, in the ledger's order. -/
structure Evidence where
  pins : Json
  windows : Array WindowEvidence

def EVIDENCE_FORM : String :=
  "a period's evidence is {pins, windows}: the host's pins log and, for each window in the ledger's order, " ++
  "{audits, session}: its challenger session's files and, for each registration the session sampled, " ++
  "{evidence, public, registration}"

def Evidence.of (j : Json) : V Evidence := do
  unless keys? j == some #["pins", "windows"] do throw EVIDENCE_FORM
  let .arr ws := getD j "windows" | throw EVIDENCE_FORM
  let windows ← ws.mapM fun w => do
    unless keys? w == some #["audits", "session"] do throw EVIDENCE_FORM
    let .arr cs := getD w "audits" | throw EVIDENCE_FORM
    let audits ← cs.mapM fun a => do
      unless keys? a == some #["evidence", "public", "registration"] do throw EVIDENCE_FORM
      let .str pub := getD a "public" | throw EVIDENCE_FORM
      pure ({ registration := getD a "registration", pub, evidence := getD a "evidence" } : Case)
    pure ({ session := getD w "session", audits } : WindowEvidence)
  return { pins := getD j "pins", windows }

def CASE_COINS : String :=
  "a case's evidence names its coins as {segment} alone: the ledger, main's history, the pins log and the " ++
  "session's files are the period's own"

/-- One-stage v2's evidence for a case: the case's own, with its coins' ledger, main's history, pins log and session
files the period's (the commitment, the verifier's parameter and the period's evidence), and only the segment the
case's. -/
def evidenceAt (p : AuditParams) (ledger pins session e : Json) : V Json := do
  unless keys? (getD e "coins") == some #["segment"] do throw CASE_COINS
  let coins := Json.mkObj [("ledger", ledger), ("main", p.main), ("pins", pins),
    ("segment", getD (getD e "coins") "segment"), ("session", session)]
  return Json.mkObj (((entries e).filter fun kv => kv.1 != "coins").toList ++ [("coins", coins)])

/-! ## The check -/

/-- The registrations a session's roots show a coin of family `label` released after, as lowercase hex in the order
the session received them, each once: the window's own root `root` aside, which a coin is released after before any
registration. -/
def sampledOn (label root : String) (roots : Challenger.Roots) : Array String :=
  (((roots.find? (·.1 == label)).map (·.2)).getD []).foldl
    (fun acc r => let h := Flock.hexOf r; if h == root || acc.contains h then acc else acc.push h) #[]

/-- An accepted audit: the registration's digest and what one-stage v2 published and profiled for it. -/
structure Audit where
  registration : String
  published : OneStage.Public
  profile : Core.Integrity

/-- A window with its accepted audits. -/
structure Passed where
  window : Window
  audits : Array Audit

/-- What an accepted period publishes: the lane and the monitoring period, each window with its audits, and the
lane's other sessions, of which nothing is held. -/
structure Public where
  lane : String
  start : String
  stop : String
  windows : Array Passed
  others : Array Session

/-- The check's state along the windows: the records of the registrations audited so far, which the next audit's
`log` holds; the windows passed; and the first audit one-stage v2 left undecided. -/
structure Progress where
  before : Array Json := #[]
  passed : Array Passed := #[]
  undecided : Option String := none

/-- A refusal of the challenger's records, under `coin`. -/
def coinOf {α : Type} (x : V α) : Except (Code × String) α :=
  match x with
  | .ok a => .ok a
  | .error why => .error (Code.coin, why)

def ONE_MEMBER : String :=
  "only a window of one registration is covered, and a session that sampled several (several ranks', or one " ++
  "registered again) is refused"

def SHARED_ROWS : String :=
  "the public file's header names shared_rows: a registration whose members share row tables is not covered"

/-- One case at a window: the registration's record and what one-stage v2 answers on it. The case's registration is
the one the session sampled (`want`, `evidence`); the parameters at the window are one-stage v2's and its public file
shares no rows (`names`); its evidence is one-stage v2's (`evidence`). -/
def caseAt (p : AuditParams) (ledger pins session : Json) (w : Window) (want : String) (before : Array Json)
    (k : Case) (bs : Core.Blobs) : Except (Code × String) (Json × Core.Outcome OneStage.Public Core.Integrity) := do
  let c ← match Flock.canon k.registration >>= fun s => OneStage.commitmentOf s.toUTF8 with
    | .ok c => pure c
    | .error why => throw (Code.evidence, s!"session {w.session}: registration: {why}")
  let got := Flock.hexOf c.digest
  unless got == want do
    throw (Code.evidence, s!"session {w.session}: the case is of registration {got.take 16}, and the session's " ++
      s!"log shows its {p.ledger.label} coins released after {want.take 16}")
  let params ← match OneStageZk.paramsOfJson (paramsAt p w.root k.pub before) with
    | .ok q => pure q
    | .error why => throw (Code.names, s!"session {w.session}: {why}")
  let header ← match bs.get k.pub "public file" >>= FilePart.header with
    | .ok h => pure h
    | .error why => throw (Code.names, s!"session {w.session}: {why}")
  if (field? header "shared_rows").isSome then throw (Code.names, s!"session {w.session}: {SHARED_ROWS}")
  let e ← match evidenceAt p ledger pins session k.evidence >>= Flock.canon >>=
      fun s => OneStageZk.evidenceOf s.toUTF8 with
    | .ok e => pure e
    | .error why => throw (Code.evidence, s!"session {w.session}: {why}")
  return (c.record, OneStageZk.checkBlobs params c e bs)

/-- One window (module docstring, **The check**): its session is the one the ledger sealed, run from a commit the pins
log records on main's first-parent line, and withheld (`coin`); it sampled one registration (`missing`, `members`);
the evidence holds one case (`evidence`); and one-stage v2 answers the case (`caseAt`): an accept adds the window, a
reject refuses the period (`audit`), and an undecided is kept as the period's, the first one, while later windows are
still checked. -/
def windowAt (p : AuditParams) (r : Read) (ledger pins : Json) (bs : Core.Blobs) (st : Progress) (w : Window)
    (ev : WindowEvidence) : Except (Code × String) Progress := do
  let (a, opened) ← coinOf (Challenger.auditedIn r.wins r.periods (Json.mkObj [("root", Json.str w.root)]))
  if let some why := Challenger.pinsReason a.session opened pins then throw (Code.coin, why)
  if let some why := Challenger.firstParentReason a.session a.pinned p.main then throw (Code.coin, why)
  let (_, roots) ← coinOf (Challenger.sessionNamed a p.ledger.key ev.session)
  unless Challenger.withheldOn ledger w.root do throw (Code.coin, Challenger.WITHHELD)
  let sampled := sampledOn p.ledger.label w.root roots
  let want ← match sampled with
    | #[] => throw (Code.missing, s!"session {w.session} released no {p.ledger.label} coin after a registration: " ++
        "its root was committed and never sampled")
    | #[d] => pure d
    | _ => throw (Code.members, s!"session {w.session} sampled {sampled.size} registrations: {ONE_MEMBER}")
  let #[k] := ev.audits
    | throw (Code.evidence, s!"session {w.session}: {ev.audits.size} cases for the one registration its log " ++
        "shows sampled")
  let (record, o) ← caseAt p ledger pins ev.session w want st.before k bs
  let at_ := s!"session {w.session}, registration {want.take 16}"
  match o with
  | .accept published profile =>
    let audit : Audit := { registration := want, published, profile }
    return { st with before := st.before.push record, passed := st.passed.push { window := w, audits := #[audit] } }
  | .reject why => throw (Code.audit, s!"{at_}: {why}")
  | .undecided why =>
    return { st with before := st.before.push record, undecided := st.undecided <|> some s!"{at_}: {why}" }

/-- The ledger's half, then every window in the ledger's order against its evidence (`windowAt`): the windows and
the lane's other sessions with the check's state at the end, or the first refusal. -/
def run (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) :
    Except (Code × String) (Windows × Progress) := do
  let r ← read p.ledger c.ledger
  let ws ← windowsIn p.ledger r
  unless e.windows.size == ws.windows.size do
    throw (Code.evidence, s!"the evidence holds {e.windows.size} windows' sessions, and the ledger lists " ++
      s!"{ws.windows.size} windows in the monitoring period")
  let st ← (ws.windows.zip e.windows).foldlM (init := ({} : Progress))
    fun st (w, ev) => windowAt p r c.ledger e.pins bs st w ev
  return (ws, st)

/-- **The period's check** on the blobs: a refusal of `run` rejects under its code; an audit one-stage v2 left
undecided leaves the period undecided; otherwise every window of the lane in the monitoring period has one accepted
audit of the one registration its session sampled, and the period is accepted, publishing them with the lane's other
sessions, its profile each audit's in order. -/
def checkBlobs (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome Public (Array Core.Integrity) :=
  match run p c e bs with
  | .error (code, why) => .reject s!"{code.name}: {why}"
  | .ok (ws, st) =>
    match st.undecided with
    | some why => .undecided s!"audit: {why}"
    | none =>
      .accept { lane := p.ledger.lane, start := p.ledger.start, stop := p.ledger.stop, windows := st.passed,
                others := ws.others }
        (st.passed.flatMap fun x => x.audits.map (·.profile))

def Audit.toJson (a : Audit) : Json :=
  Json.mkObj [("published", OneStage.publicJson a.published), ("registration", Json.str a.registration)]

def Passed.toJson (x : Passed) : Json :=
  Json.mkObj [("audits", Json.arr (x.audits.map Audit.toJson)), ("period", Json.str x.window.period),
    ("phase", Lean.toJson x.window.phase), ("root", Json.str x.window.root), ("session", Json.str x.window.session)]

def Public.toJson (x : Public) : Json :=
  Json.mkObj [("lane", Json.str x.lane), ("others", Json.arr (x.others.map Session.toJson)),
    ("start", Json.str x.start), ("stop", Json.str x.stop), ("windows", Json.arr (x.windows.map Passed.toJson))]

def PROTOCOL : String := "verity/sampled-proofs/period/v1"

def protocol : Core.BlobProtocol where
  id := PROTOCOL
  Params := AuditParams
  Commitment := Commitment
  Evidence := Evidence
  Public := Public
  Profile := Array Core.Integrity
  params := AuditParams.of
  commitment := Commitment.of
  evidence b := do Evidence.of (← Core.canonicalJson b)
  digest c := c.digest
  check p c e := checkBlobs p c e Core.Blobs.empty
  publicJson := Public.toJson
  profileJson ps := Json.arr (ps.map Core.Integrity.toJson)
  checkBlobs := checkBlobs

/-! The difftest's pieces. -/

/-- `{"before", "case", "ledger", "params", "pins", "root", "session"}` to what one-stage v2 is asked at a case
(`paramsAt`, `evidenceAt`): `{evidence, params}`, or `{refused}` where the parameters or the case's evidence are not
of their form. A request in another form is the request's fault. -/
def askedPiece (j : Json) : V Json := do
  unless keys? j == some #["before", "case", "ledger", "params", "pins", "root", "session"] do
    throw "a request's keys are before, case, ledger, params, pins, root, session"
  let (.arr before, .str root) := (getD j "before", getD j "root") | throw "before is a list and root a string"
  let k := getD j "case"
  let .str pub := getD k "public" | throw "the case's public is not a string"
  return match AuditParams.of (getD j "params") with
    | .error why => Json.mkObj [("refused", Json.str s!"params: {why}")]
    | .ok p =>
      match evidenceAt p (getD j "ledger") (getD j "pins") (getD j "session") (getD k "evidence") with
      | .error why => Json.mkObj [("refused", Json.str s!"evidence: {why}")]
      | .ok e => Json.mkObj [("evidence", e), ("params", paramsAt p root pub before)]

/-- `{"blobs", "evidence", "ledger", "params"}` to the period's check's answer (`checkBlobs`): `{verdict: accept,
published, profile}`, or `{verdict, reason}` for a reject or an undecided; `{refused}` where the parameters or the
evidence are not of their form. A request in another form is the request's fault. -/
def checkPiece (j : Json) : V Json := do
  unless keys? j == some #["blobs", "evidence", "ledger", "params"] do
    throw "a request's keys are blobs, evidence, ledger, params"
  let (.arr hs, .arr _) := (getD j "blobs", getD j "ledger") | throw "blobs and ledger are lists"
  let blobs ← hs.mapM fun h =>
    match h with
    | .str s => Flock.unhex s
    | _ => throw "a blob is not a hex string"
  return match AuditParams.of (getD j "params") with
    | .error why => Json.mkObj [("refused", Json.str s!"params: {why}")]
    | .ok p =>
      match Evidence.of (getD j "evidence") with
      | .error why => Json.mkObj [("refused", Json.str s!"evidence: {why}")]
      | .ok e =>
        match checkBlobs p { ledger := getD j "ledger", digest := ByteArray.empty } e (Core.Blobs.ofArray blobs) with
        | .accept pub prof => Json.mkObj [("profile", Json.arr (prof.map Core.Integrity.toJson)),
            ("published", pub.toJson), ("verdict", Json.str "accept")]
        | .reject why => Json.mkObj [("reason", Json.str why), ("verdict", Json.str "reject")]
        | .undecided why => Json.mkObj [("reason", Json.str why), ("verdict", Json.str "undecided")]

end Spot.Period
