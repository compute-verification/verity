module

public import PouwCheck.Credit
public import Spot.ChallengerSession

/-!
# A period's credit with one replay of the auditor's ledger (`Credit.periodCredit`, compiled)

`Credit.periodCredit` reads the auditor's ledger through `Spot.Challenger.checked` (each line's parse, the hash chain and
each entry's LMS signature) once in `Credit.period` and five more times a window: `creditedChallengePhase` for the
window and for its sample session, `drawSession`, `roundsCheck` and `periodWeights`, each through `auditedOpen`. Each
reads it at the same ledger and the key `Credit.periodCredit` reads, so one replay does.

Each function on that path has a twin here, its body as written with `checked ledger` a parameter `replay`
(`…With`), equal to it at `replay := checked ledger` by `rfl`. `periodCreditOnce` replays the ledger once at the
verifier's key and hands every twin that replay (`replayAt`), and `periodCredit_once`, proved, makes the compiler run
it wherever `Credit.periodCredit` is called. Nothing a guarantee reads changes: `Credit.periodCredit` and every
function it calls are as written; only their compiled code is this.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Challenger.Once

open Lean (Json)
open Flock (V)

/-- What `checked ledger` returns for a key: a ledger's replay. -/
abbrev Replay := ChallengerKey → V (Array Win × Array MonitoringPeriod)

/-! `auditedOpen`'s private helpers, as `Spot.ChallengerSession` writes them: its private part stays its own. -/

private def strOf (j : Json) : String := match j with | .str s => s | _ => ""

private def jsonTrue (j : Json) : Bool := match j with | .bool true => true | _ => false

private def isDigest (j : Json) (n : Nat := 128) : Bool :=
  match j with
  | .str s => s.length == n && (Flock.unhex s).toOption.isSome
  | _ => false

/-- `auditedOpen` with `replay` its ledger's replay. -/
def auditedOpenWith (replay : Replay) (window : Json) (key : ChallengerKey) : V (Audited × Json) := do
  unless isRun window do throw s!"the window's identity is malformed: {WINDOW_FORM}"
  let (wins, periods) ← replay key
  let root := strOf (getD window "root")
  let some w := wins.find? (·.root == root) | throw "the auditor's ledger opens no challenger session for this window"
  let o := w.opened
  let s := o.session
  unless o.schema == LEDGER_SCHEMA do
    throw s!"the auditor's ledger opens this window's session {s} under {o.schema}, which names no prover and seals no refused hellos"
  let some e := w.ended
    | throw s!"the auditor's ledger has no end for session {s}: the session never ended, or the ledger is older than its end"
  let b := e.body
  unless jsonTrue (getD b "closed") do
    throw s!"the auditor's ledger ends session {s} unclosed: its client didn't close it"
  let files := getD b "files"
  let malformed := s!"the auditor's ledger's end of session {s} is malformed: {END_FORM}"
  let some recordLeaf := nat? (getD b "record_leaf") | throw malformed
  let some refusedHellos := nat? (getD b "refused_hellos") | throw malformed
  unless isDigest (getD b "commitment") && SEALED_FILES.all (fun f => (field? files f).isSome) do throw malformed
  let p := periods.find? (·.windows.contains s)
  if let some p := p then
    if p.cap == Json.null || p.circuit == Json.null then
      throw s!"the auditor's ledger's period {p.name}, which holds session {s}, commits no cap or circuit"
  return ({ session := s, host := strOf (getD o.body "host"), prover := laneOf o.body, files, recordLeaf,
            commitment := strOf (getD b "commitment"), refusedHellos, pinned := strOf (getD o.body "pinned"),
            period := p.map (·.name), cap := (p.map (·.cap)).getD Json.null,
            circuit := (p.map (·.circuit)).getD Json.null }, o.body)

theorem auditedOpen_eq (ledger window : Json) (key : ChallengerKey) :
    auditedOpen ledger window key = auditedOpenWith (checked ledger) window key := rfl

/-- `auditedChallengePhase` with `replay` its ledger's replay. -/
def auditedChallengePhaseWith (replay : Replay) (window : Json) (key : ChallengerKey) : V Audited :=
  return (← auditedOpenWith replay window key).1

theorem auditedChallengePhase_eq (ledger window : Json) (key : ChallengerKey) :
    auditedChallengePhase ledger window key = auditedChallengePhaseWith (checked ledger) window key := rfl

/-- `pinnedChallengePhase` with `replay` its ledger's replay. -/
def pinnedChallengePhaseWith (replay : Replay) (window : Json) (key : ChallengerKey) (pins main : Json) : V Audited := do
  let (a, opened) ← auditedOpenWith replay window key
  if let some why := pinsReason a.session opened pins then throw why
  if let some why := firstParentReason a.session (strOf (getD opened "pinned")) main then throw why
  return a

theorem pinnedChallengePhase_eq (ledger window : Json) (key : ChallengerKey) (pins main : Json) :
    pinnedChallengePhase ledger window key pins main = pinnedChallengePhaseWith (checked ledger) window key pins main :=
  rfl

/-- `trustedChallengePhaseOf` with `replay` its ledger's replay. -/
def trustedChallengePhaseOfWith (replay : Replay) (window : Json) (key : ChallengerKey) (pins main session : Json)
    : V (LiveCommitment × Roots) := do
  challengePhaseNamed (← pinnedChallengePhaseWith replay window key pins main) key session

theorem trustedChallengePhaseOf_eq (ledger window : Json) (key : ChallengerKey) (pins main session : Json) :
    trustedChallengePhaseOf ledger window key pins main session =
      trustedChallengePhaseOfWith (checked ledger) window key pins main session := rfl

/-- `creditedChallengePhase` with `replay` its ledger's replay. -/
def creditedChallengePhaseWith (replay : Replay) (window : Json) (key : ChallengerKey) (pins main : Json)
    (descriptor : ByteArray) : V Audited := do
  let a ← pinnedChallengePhaseWith replay window key pins main
  if let some why := circuitReason a.period a.circuit descriptor then throw why
  return a

theorem creditedChallengePhase_eq (ledger window : Json) (key : ChallengerKey) (pins main : Json)
    (descriptor : ByteArray) :
    creditedChallengePhase ledger window key pins main descriptor =
      creditedChallengePhaseWith (checked ledger) window key pins main descriptor := rfl

/-- Whether two keys are one: their parameters and public keys equal, decided field by field. -/
def sameKey (a b : ChallengerKey) : Prop :=
  a.params.N = b.params.N ∧ a.params.W = b.params.W ∧ a.params.H = b.params.H ∧ a.pub.data = b.pub.data

instance instDecidableSameKey (a b : ChallengerKey) : Decidable (sameKey a b) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

theorem eq_of_sameKey {a b : ChallengerKey} (h : sameKey a b) : a = b := by
  obtain ⟨⟨N, W, H⟩, ⟨d⟩⟩ := a
  obtain ⟨⟨N', W', H'⟩, ⟨d'⟩⟩ := b
  obtain ⟨h1, h2, h3, h4⟩ := h
  cases h1; cases h2; cases h3; cases h4; rfl

/-- `checked ledger`, its value at `k0` held as `c0`: a replay at `k0` made once, wherever it is read. -/
def replayAt (ledger : Json) (k0 : ChallengerKey) (c0 : V (Array Win × Array MonitoringPeriod)) : Replay :=
  fun k => if sameKey k k0 then c0 else checked ledger k

theorem replayAt_checked (ledger : Json) (k0 : ChallengerKey) : replayAt ledger k0 (checked ledger k0) = checked ledger := by
  funext k
  unfold replayAt
  split
  · next h => rw [eq_of_sameKey h]
  · rfl

end Spot.Challenger.Once

namespace PouwCheck.Once

open Lean (Json)
open Spot.Challenger (ChallengerKey LiveOpening)
open Spot.Challenger.Once (Replay trustedChallengePhaseOfWith)

/-- `roundsCheck` with `replay` its ledger's replay. -/
def roundsCheckWith (replay : Replay) (key : Json) (pins main window session roots openings : Json) : R Unit := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let (c, received) ← (trustedChallengePhaseOfWith replay window k pins main session).mapError fun e =>
    ("coin", s!"the challenger's session is not trusted: {e}")
  let some rs := (match roots with
      | .arr rs => rs.toList.mapM fun r => if (str? r).any isHexDigest then hex? r else none
      | _ => none) | throw ("rounds", "the round roots are not a list of SHA-512 digests")
  let .arr os := openings | throw ("rounds", "the round coins are not a list")
  let os ← os.toList.zipIdx.mapM fun (o, i) => (LiveOpening.ofJson o).mapError fun e => ("coin", s!"round coin {i}: {e}")
  if let some why := (← port (PouwWindow.roundsReason c rs os)) then coin why
  unless (received.lookup PouwWindow.ROUND).getD [] == rs do
    throw ("rounds", "the round roots are not the roots the challenger's log shows it received, in order")

theorem roundsCheck_eq (key ledger pins main window session roots openings : Json) :
    roundsCheck key ledger pins main window session roots openings =
      roundsCheckWith (Spot.Challenger.checked ledger) key pins main window session roots openings := rfl

end PouwCheck.Once

namespace PouwCheck.Credit.Once

open Lean (Json)
open PouwCommit (Part tagged Domain)
open Spot.Challenger (ChallengerKey LiveOpening liveReason)
open Spot.Challenger.Once (Replay trustedChallengePhaseOfWith auditedChallengePhaseWith creditedChallengePhaseWith
  replayAt replayAt_checked)
open PouwCheck.Once (roundsCheckWith)

/-- `drawSession` with `replay` its ledger's replay. -/
def drawSessionWith (replay : Replay) (ledger : Json) (k : ChallengerKey) (pins main session : Json) (root : String) :
    R (Spot.Challenger.LiveCommitment × Spot.Challenger.Roots) := do
  let got ← (trustedChallengePhaseOfWith replay (Json.mkObj [("root", Json.str root)]) k pins main session).mapError
    fun e => ("coin", s!"the draw's session is not trusted: {e}")
  unless Spot.Challenger.withheldOn ledger root do throw ("coin", Spot.Challenger.WITHHELD)
  pure got

theorem drawSession_eq (ledger : Json) (k : ChallengerKey) (pins main session : Json) (root : String) :
    drawSession ledger k pins main session root =
      drawSessionWith (Spot.Challenger.checked ledger) ledger k pins main session root := rfl

/-- `Internal.creditedDraw` with `replay` its ledger's replay. -/
def creditedDrawWith (replay : Replay) (ledger : Json) (k : ChallengerKey) (pins main draw : Json) (root : ByteArray) :
    R Unit := do
  unless (str? (get draw "root")) == some (Flock.hexOf root) do throw ("draw", DRAW_ROOT)
  let (c, received) ← drawSessionWith replay ledger k pins main (get draw "session") (Flock.hexOf root)
  let got := (received.lookup PouwWindow.DRAW).getD []
  if got.isEmpty then throw ("draw", NO_DRAW)
  unless got == [root] do throw ("draw", CHANGED)
  let why := match LiveOpening.ofJson (get draw "coin") with
    | .error e => some e
    | .ok o => match liveReason c o PouwWindow.DRAW 0 root with
      | .error e => some e
      | .ok w => w
  if let some w := why then throw ("coin", s!"the draw coin: {w}")

theorem creditedDraw_eq (ledger : Json) (k : ChallengerKey) (pins main draw : Json) (root : ByteArray) :
    Internal.creditedDraw ledger k pins main draw root =
      creditedDrawWith (Spot.Challenger.checked ledger) ledger k pins main draw root := rfl

/-- `Internal.credited` with `replay` its ledger's replay. -/
def creditedWith (replay : Replay) (key ledger pins main salt calls served commit rounds draw : Json) : R (List Nat) := do
  let (k, out, root) ← Internal.creditedRoot key salt calls served commit rounds draw
  creditedDrawWith replay ledger k pins main draw root
  pure out

theorem credited_eq (key ledger pins main salt calls served commit rounds draw : Json) :
    Internal.credited key ledger pins main salt calls served commit rounds draw =
      creditedWith (Spot.Challenger.checked ledger) key ledger pins main salt calls served commit rounds draw := rfl

/-- `periodWeights` with `replay` its ledger's replay. -/
def periodWeightsWith (replay : Replay) (key ledger window rounds layout entries : Json) (program : String)
    (prefix_ : Json) : R (String × String) := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let a ← (auditedChallengePhaseWith replay window k).mapError fun e =>
    ("coin", s!"the challenger's session is not trusted: {e}")
  let some name := a.period
    | weightsRefuse ("the window is in no period of the auditor's ledger, so no period-open signed the weights' " ++
        s!"root it opens ({WEIGHTS_FIELD}): reserve its session inside a period whose open signs it")
  let signed ← committedWeights ledger name
  let requests := match rounds with
    | .arr xs => (xs[0]?.map (get · "requests")).getD .null
    | _ => .null
  let w ← Internal.weights window layout entries requests (.str a.session) (.str program) prefix_
  let got := Flock.hexOf w
  unless got == signed do
    weightsRefuse (s!"the window's registered weights' root {got.take 16}... is not {signed.take 16}..., the one " ++
      s!"period {name}'s open signed: a B row's codes or s_w differ from the ones the period registered before its " ++
      "first window's salt. Open every window of a period on its registered B rows")
  pure (a.session, name)

theorem periodWeights_eq (key ledger window rounds layout entries : Json) (program : String) (prefix_ : Json) :
    periodWeights key ledger window rounds layout entries program prefix_ =
      periodWeightsWith (Spot.Challenger.checked ledger) key ledger window rounds layout entries program prefix_ := rfl

/-- `windowLayout` with `replay` its ledger's replay. -/
def windowLayoutWith (replay : Replay)
    (key ledger pins main window session salt calls served commit rounds draw descriptor entries proofs live k rows :
      Json) :
    R (List Nat × Json × List Nat) := do
  let got ← creditedWith replay key ledger pins main salt calls served commit rounds draw
  let log := match rounds with
    | .arr xs => xs
    | _ => #[]
  roundsCheckWith replay key pins main window session (.arr (log.map (get · "root"))) (.arr (log.map (get · "coin")))
  let some d := (match descriptor with
      | .str s => if isHexBytes s then hex? (.str s) else none
      | _ => none) | layoutRefuse DESCRIPTOR_HEX
  let P ← programOf d
  let layout ← layoutOf P
  let some t := (nat? rows).filter (1 ≤ ·) | layoutRefuse ROWS_FORM
  let r := (match get layout "tile" with
    | .arr a => (nat? (a.getD 0 .null)).getD 0
    | _ => 0)
  let m := (nat? (get layout "m")).getD 0
  unless slotOf t r == m do layoutRefuse (otherSlot t r m)
  let program := Flock.hexOf (Flock.Sha512.hash d)
  let (.arr rows, .arr lcalls) := (calls, get layout "calls") | layoutRefuse LAYOUT_FORM
  unless rows.size == lcalls.size do
    throw ("manifest", s!"the manifest's {rows.size} rows are not the circuit's {lcalls.size} calls: each call of " ++
      "the window's Program is one manifest row, in order")
  Internal.aLink rounds entries rows.size (tableRows layout log.size ((str? (prefixOf P)).getD ""))
  let (sid, name) ← periodWeightsWith replay key ledger window rounds layout entries program (prefixOf P)
  let ss ← strata layout
  Internal.arow log entries (((commitOf? commit).map (·.2)).getD #[]) layout ((str? (prefixOf P)).getD "")
  let rows ← Internal.servedRows calls layout log.size t
  let maxTokens ← committedLength ledger name
  Internal.length maxTokens rows layout log.size
  proved (← windowDrawn window draw k ss) proofs
  Internal.registration ((str? (get window "root")).getD "") program ((str? (prefixOf P)).getD "") entries proofs ss
    layout log.size sid
  Internal.live live t log.size ((str? (prefixOf P)).getD "") program ((str? (get window "root")).getD "") entries rows
    maxTokens
  pure (got.filter fun i => 0 < rows.getD i 0, layout, rows)

theorem windowLayout_eq
    (key ledger pins main window session salt calls served commit rounds draw descriptor entries proofs live k rows :
      Json) :
    windowLayout key ledger pins main window session salt calls served commit rounds draw descriptor entries proofs live
        k rows =
      windowLayoutWith (Spot.Challenger.checked ledger) key ledger pins main window session salt calls served commit
        rounds draw descriptor entries proofs live k rows := rfl

/-- `period` with `replay` its ledger's replay. -/
def periodWith (replay : Replay) (key ledger name roots : Json) : R Unit := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  discard <| (replay k).mapError fun e =>
    ("coin", s!"the auditor's ledger is not trusted: {e}")
  let .str name := name | throw ("period", "the period is not named")
  let .arr lines := ledger | throw ("period", "the period is not named")
  let es := lines.filterMap Spot.Challenger.ledgerEntry?
  let named := fun (e : Spot.Challenger.LedgerEntry) => get e.body "period" == Json.str name
  unless es.any (fun e => e.kind == "period-open" && named e) do
    throw ("period", s!"the auditor's ledger opens no period {name}: credit counts only a period the ledger opens " ++
      "and closes")
  let some close := (es.find? fun e => e.kind == "period-close" && named e).map (·.body)
    | throw ("period", s!"period {name} has no close in the auditor's ledger, so it earns nothing: close it " ++
        "(period-close) and present the ledger after its close")
  let ws := match get close "windows" with
    | .arr a => a
    | _ => #[]
  unless get close "complete" == Json.bool true do
    match ws.findIdx? (fun w => get w "complete" != Json.bool true) with
    | none => throw ("period", s!"period {name} closed with no window, so it earns nothing")
    | some i => throw ("period", s!"period {name} closed incomplete: its window {i} (session " ++
        s!"{(str? (get ws[i]! "session")).getD ""}) never opened or didn't end complete before the close, so the " ++
        "period earns nothing")
  let some rs := (match roots with
      | .arr a => a.toList.mapM fun r => if (digest? r).isSome then str? r else none
      | _ => none) | throw ("period", "the period's sessions are not a list of open roots (SHA-512 digests)")
  let want := ws.toList.map fun w => (str? (get w "root")).getD ""
  let wanted : Std.HashMap String Unit := want.foldl (fun m r => m.insert r ()) {}
  let presented : Std.HashMap String Unit := rs.foldl (fun m r => m.insert r ()) {}
  for i in [0:max want.length rs.length] do
    if i ≥ want.length then
      throw ("period", s!"the sessions presented run past the {want.length} that period {name}'s close lists: only " ++
        "its close's sessions count, each once")
    let w := want[i]!
    if i ≥ rs.length || (rs[i]! != w && !presented.contains w) then
      throw ("missing", s!"session {i} of period {name} ({w.take 16}...) is missing: every session between the " ++
        "period's open and its close counts, so present them all, in the close's order")
    let r := rs[i]!
    if r != w then
      if !wanted.contains r then
        throw ("period", s!"session {r.take 16}... is not one of period {name}'s sessions: only the sessions its " ++
          "close lists count")
      throw ("order", s!"session {i} of period {name} is {r.take 16}..., not {w.take 16}..., the close's session {i}: " ++
        "present the period's sessions in the order its close lists them")

theorem period_eq (key ledger name roots : Json) :
    period key ledger name roots = periodWith (Spot.Challenger.checked ledger) key ledger name roots := rfl

/-- `periodCredit` with `replay` its ledger's replay. -/
def periodCreditWith (replay : Replay) (key ledger pins main name windows : Json) : R PeriodCredit := do
  let some ws := (match windows with
      | .arr a => if a.all (fun w => keys w == some PERIOD_KEYS) then some a.toList else none
      | _ => none) | throw ("period", WINDOWS_FORM)
  let rootOf := fun (j : Json) => match j with
    | .obj _ => get j "root"
    | _ => .null
  let roots := Json.arr (ws.toArray.flatMap fun w => #[rootOf (get w "window"), rootOf (get w "draw")])
  periodWith replay key ledger name roots
  let nm := (str? name).getD ""
  let bigK ← committedK ledger nm
  let t ← committedRows ledger nm
  discard <| committedWeights ledger nm
  let periodPin := committedPin ledger nm
  let floor := committedFloor ledger nm
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let mut out : List (List Nat) := []
  let mut work : List Nat := []
  let mut eps : List (Option Rat) := []
  for (w, i) in ws.zipIdx do
    let f := get w
    let some d := (match f "descriptor" with
        | .str s => if isHexBytes s then hex? (.str s) else none
        | _ => none) | throw ("circuit", s!"window {i} of period {nm}: {DESCRIPTOR_HEX}")
    let a ← (creditedChallengePhaseWith replay (f "window") k pins main d).mapError fun e =>
      ("circuit", s!"window {i} of period {nm} doesn't count toward it, so the period earns nothing: {e}")
    if let some why := pinHygiene nm i main a.pinned periodPin floor then throw why
    let dr ← (creditedChallengePhaseWith replay (Json.mkObj [("root", rootOf (f "draw"))]) k pins main
      d).mapError fun e =>
      ("circuit", s!"window {i}'s draw session of period {nm} doesn't count toward it, so the period earns nothing: {e}")
    if let some why := pinHygiene nm i main dr.pinned periodPin floor DRAW_UNIT then throw why
    let (got, layout, rows) ← (windowLayoutWith replay key ledger pins main (f "window") (f "session") (f "salt")
      (f "calls") (f "served") (f "commit") (f "rounds") (f "draw") (f "descriptor") (f "entries") (f "proofs")
      (f "live") (Lean.toJson bigK) (Lean.toJson t)).mapError
      fun (c, why) => ("window", s!"window {i} of period {nm} is refused, so the period earns nothing: {c}: {why}")
    out := out ++ [got]
    work := work ++ [creditedWork layout got rows]
    eps := eps ++ [established layout got rows bigK]
  pure ⟨out, work, bigK, DELTA, eps⟩

theorem periodCredit_eq (key ledger pins main name windows : Json) :
    periodCredit key ledger pins main name windows =
      periodCreditWith (Spot.Challenger.checked ledger) key ledger pins main name windows := rfl

/-- `periodCreditWith` on the replay at `k0` made once, `c0`: kept out of line so that `c0`, an argument, is computed
once, before the call. -/
@[noinline] def periodCreditAt (ledger : Json) (k0 : ChallengerKey)
    (c0 : Flock.V (Array Spot.Challenger.Win × Array Spot.Challenger.MonitoringPeriod))
    (key pins main name windows : Json) : R PeriodCredit :=
  periodCreditWith (replayAt ledger k0 c0) key ledger pins main name windows

/-- `periodCredit` with one replay of the ledger: at the verifier's key, read by every step that reads the ledger's
replay. A key that doesn't parse is refused before any replay, as `periodCredit` refuses it. -/
def periodCreditOnce (key ledger pins main name windows : Json) : R PeriodCredit :=
  match ChallengerKey.ofJson key with
  | .ok k0 => periodCreditAt ledger k0 (Spot.Challenger.checked ledger k0) key pins main name windows
  | .error _ => periodCreditWith (Spot.Challenger.checked ledger) key ledger pins main name windows

/-- The compiler runs `periodCreditOnce` wherever `periodCredit` is called. -/
@[csimp] theorem periodCredit_once : @periodCredit = @periodCreditOnce := by
  funext key ledger pins main name windows
  rw [periodCredit_eq]
  unfold periodCreditOnce periodCreditAt
  split
  · rw [replayAt_checked]
  · rfl

end PouwCheck.Credit.Once
