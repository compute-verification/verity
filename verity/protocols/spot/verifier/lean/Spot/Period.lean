/-
Sampled proofs' period, the ledger's half: which sessions of a lane the auditor's ledger opened during a monitoring
period, and which of them are the period's windows (`tests/test_period_difftest.py` holds this to a Python reference
that lives in the test alone: there is no Python verifier).

The verifier holds the auditor's challenger key, the prover's lane on the gate, the family a sample's coins are of
(`label`) and the monitoring period's two ends on the auditor's clock, `start` included and `stop` not (`Params`). The
ledger is the auditor's signed lines, as `Challenger.checked` reads them under that key.

**The monitoring period in the ledger.** The challenger's host stamps each entry with its clock (`stampOf`: a
reserve's and an open's `started_utc`, an end's `ended_utc`, a period's `opened_utc` and `closed_utc`, a declaration's
`declared_utc`), and reads the clock before the entry takes the ledger's lock, so two entries written together may
carry their stamps out of order. The monitoring period is therefore a run of the ledger's entries and not a set of
stamps: from the first entry stamped at or after `start` up to, and not including, the first stamped at or after
`stop` (`reaches`). Monitoring periods that follow one another, one's `stop` the next's `start`, then split a ledger's
entries with none in two and none between, whatever its stamps say, and a ledger read again with entries appended
gives each earlier monitoring period the same run. A ledger with no entry stamped at or after `stop` is refused
(`early`): a ledger cut before the monitoring period's end could hide a later session.

**Sessions** (`sessionsOf`): every session of the lane that a v2 open of that run opens on a root, in the ledger's
order and whatever its layout, each with its root, the families its open carries, whether its open says it withholds
its coin commitment, its place among the lane's reserves (`reservesOf`: a session counted where a period counts it,
at its reserve, or at its open when it was opened in one step) and the period that lists it, if one does.

**Windows** (`windowsOf`): the sessions whose layout holds the family `label` (`holds`). In order, and the first
refusal is the answer: there is at least one (`missing`); each is in a period of the lane (`outside`); each such
period is closed, and every session it lists that was opened is a window (`period`); no session it lists was left
unopened, and each window has ended (`missing`). The lane's other sessions are returned beside the windows (`others`)
with nothing held of them: a reader sees every root the lane committed in the monitoring period, a window or not.

The difftest's piece: `period_windows`.
-/
module

public import Spot.ChallengerSession

@[expose] public section

namespace Spot.Period

open Lean (Json)
open Flock (V)

/-- Why the ledger's half refuses: the ledger is not the auditor's (`coin`), it does not reach the monitoring period's
end (`early`), a window is in no period (`outside`), a window's period is open or holds a session that is no window
(`period`), or there is no window, a reserve never opened or a window not ended (`missing`). -/
inductive Code where
  | coin
  | early
  | outside
  | period
  | missing
  deriving DecidableEq, Repr, Inhabited

def Code.name : Code → String
  | .coin => "coin"
  | .early => "early"
  | .outside => "outside"
  | .period => "period"
  | .missing => "missing"

/-- The verifier's own parameters for the ledger's half (module docstring). -/
structure Params where
  key : Challenger.ChallengerKey
  label : String
  lane : String
  start : String
  stop : String

def PARAMS_FORM : String :=
  "a period's parameters name a family and a lane, neither empty, and a monitoring period's two ends as " ++
  "YYYY-MM-DDTHH:MM:SSZ, its start before its stop"

/-- The parameters from their JSON: the challenger's key as `ChallengerKey.ofJson` reads it, `label` and `lane`
nonempty strings, `start` and `stop` times of `Challenger.later`'s form with `start` before `stop`. -/
def Params.of (key label lane start stop : Json) : V Params := do
  let (.str label, .str lane, .str start, .str stop) := (label, lane, start, stop) | throw PARAMS_FORM
  unless !label.isEmpty && !lane.isEmpty && Challenger.later start 0 == some start &&
      Challenger.later stop 0 == some stop && decide (start < stop) do
    throw PARAMS_FORM
  return { key := ← Challenger.ChallengerKey.ofJson key, label, lane, start, stop }

/-- The key of the time the challenger's host stamps an entry of kind `kind` with. -/
def stampKey (kind : String) : Option String :=
  match kind with
  | "reserve" | "open" => some "started_utc"
  | "end" => some "ended_utc"
  | "period-open" => some "opened_utc"
  | "period-close" => some "closed_utc"
  | "declaration" => some "declared_utc"
  | _ => none

/-- An entry's stamp: the time its body holds under its kind's key (`stampKey`), when it is one of
`Challenger.later`'s form; `none` otherwise. -/
def stampOf (e : Challenger.LedgerEntry) : Option String :=
  match stampKey e.kind with
  | some k =>
    match getD e.body k with
    | .str t => if Challenger.later t 0 == some t then some t else none
    | _ => none
  | none => none

/-- The number of the first of the entries `es` stamped at or after `t`; `none` when none is. -/
def reaches (es : Array Challenger.LedgerEntry) (t : String) : Option Nat :=
  es.findIdx? fun e =>
    match stampOf e with
    | some s => !decide (s < t)
    | none => false

/-- The ledger's entries, each line as `Challenger.ledgerEntry?` reads it: every line, once `Challenger.checked`
passes. -/
def entriesOf (ledger : Json) : Array Challenger.LedgerEntry :=
  match ledger with
  | .arr lines => lines.filterMap Challenger.ledgerEntry?
  | _ => #[]

/-- Lane `lane`'s reserves among the entries `es`, in order: the session of each v2 reserve of the lane and of each
v2 open of it that binds no reserve, which is where a period counts a session. -/
def reservesOf (es : Array Challenger.LedgerEntry) (lane : String) : Array String :=
  es.filterMap fun e =>
    if e.schema == Challenger.LEDGER_SCHEMA && Challenger.laneOf e.body == lane &&
        (e.kind == "reserve" || e.kind == "open" && (field? e.body "reserve").isNone) then some e.session
    else none

/-- A session of the lane that the ledger opens on a root in the monitoring period (module docstring, **Sessions**). -/
structure Session where
  session : String
  root : String
  families : Json
  withheld : Bool
  place : Nat
  period : Option String

/-- What the ledger's half reads of a checked ledger: its windows and periods as `Challenger.checked` returns them,
and the lane's sessions of the monitoring period. -/
structure Read where
  wins : Array Challenger.Win
  periods : Array Challenger.MonitoringPeriod
  sessions : Array Session

def early (stop : String) : String :=
  s!"the auditor's ledger holds no entry stamped at or after {stop}: a ledger that does not reach the monitoring " ++
  "period's end could hide a later session"

/-- The ledger checked once under the parameters' key (`coin`), reaching the monitoring period's end (`early`), with
the lane's sessions of the monitoring period (module docstring, **Sessions**). -/
def read (p : Params) (ledger : Json) : Except (Code × String) Read := do
  let (wins, periods) ← match Challenger.checked ledger p.key with
    | .ok r => pure r
    | .error why => throw (Code.coin, why)
  let es := entriesOf ledger
  let some stop := reaches es p.stop | throw (Code.early, early p.stop)
  let start := (reaches es p.start).getD stop
  let reserves := reservesOf es p.lane
  let sessions := wins.filterMap fun w =>
    let e := w.opened
    if e.schema == Challenger.LEDGER_SCHEMA && Challenger.laneOf e.body == p.lane && start ≤ e.seq && e.seq < stop then
      some { session := e.session, root := w.root, families := getD e.body "families"
             withheld := (match getD e.body "withheld" with | .bool true => true | _ => false)
             place := (reserves.findIdx? (· == e.session)).getD reserves.size
             period := (periods.find? (·.windows.contains e.session)).map (·.name) }
    else none
  return { wins, periods, sessions }

/-- Every session of the lane the ledger opens on a root in the monitoring period (module docstring, **Sessions**). -/
def sessionsOf (p : Params) (ledger : Json) : Except (Code × String) (Array Session) :=
  (·.sessions) <$> read p ledger

/-- Whether the layout `families`, a list of `[label, coins]`, holds the family `label`: one of its entries is a list
whose first element is that string. -/
def holds (label : String) (families : Json) : Bool :=
  match families with
  | .arr fs => fs.any fun f =>
    match f with
    | .arr xs =>
      (match xs[0]? with
      | some (.str l) => l == label
      | _ => false)
    | _ => false
  | _ => false

/-- A window of the period's check: the ledger's period that lists it, its place among the check's windows in the
ledger's order, its session and its root. -/
structure Window where
  period : String
  phase : Nat
  session : String
  root : String

/-- The monitoring period's windows, and the lane's other sessions in it. -/
structure Windows where
  windows : Array Window
  others : Array Session

/-- The windows among what `read` gave, and the lane's other sessions (module docstring, **Windows**), or the first
refusal. -/
def windowsIn (p : Params) (r : Read) : Except (Code × String) Windows := do
  let (sampled, others) := r.sessions.partition fun s => holds p.label s.families
  if sampled.isEmpty then
    throw (Code.missing, s!"lane {p.lane} opens no session holding a {p.label} coin in the monitoring period")
  let windows ← sampled.zipIdx.mapM fun (s, i) =>
    match s.period with
    | some name => pure ({ period := name, phase := i, session := s.session, root := s.root } : Window)
    | none => throw (Code.outside, s!"session {s.session} is in no period of lane {p.lane}")
  let held := r.periods.filter fun mp => sampled.any (·.period == some mp.name)
  let opened := fun (ses : String) => r.wins.any (·.opened.session == ses)
  if let some mp := held.find? (!·.closed) then
    throw (Code.period, s!"period {mp.name} is not closed")
  if let some (name, ses) := held.findSome? fun mp =>
      (mp.windows.find? fun ses => opened ses && !sampled.any (·.session == ses)).map (mp.name, ·) then
    throw (Code.period, s!"period {name}'s session {ses} is not a window: it was opened outside the monitoring " ++
      s!"period, or it holds no {p.label} coin")
  if let some (name, ses) := held.findSome? fun mp =>
      (mp.windows.find? fun ses => !opened ses).map (mp.name, ·) then
    throw (Code.missing, s!"period {name}'s session {ses} was reserved and never opened")
  if let some s := sampled.find? fun s => !r.wins.any fun w => w.opened.session == s.session && w.ended.isSome then
    throw (Code.missing, s!"session {s.session} has not ended")
  return { windows, others }

/-- The monitoring period's windows and the lane's other sessions in it (module docstring, **Windows**), or the first
refusal: the ledger read once (`read`), then its windows (`windowsIn`). -/
def windowsOf (p : Params) (ledger : Json) : Except (Code × String) Windows := do
  windowsIn p (← read p ledger)

/-! The difftest's piece. -/

def Session.toJson (s : Session) : Json :=
  Json.mkObj [("families", s.families), ("period", s.period.elim Json.null Json.str), ("place", Lean.toJson s.place),
    ("root", Json.str s.root), ("session", Json.str s.session), ("withheld", Json.bool s.withheld)]

def Window.toJson (w : Window) : Json :=
  Json.mkObj [("period", Json.str w.period), ("phase", Lean.toJson w.phase), ("root", Json.str w.root),
    ("session", Json.str w.session)]

private def refused (code : Code) (why : String) : List (String × Json) :=
  [("code", Json.str code.name), ("refused", Json.str why)]

/-- `{"key", "label", "lane", "ledger", "start", "stop"}` to `{others, sessions, windows}`: `read`'s sessions and
`windowsIn`'s windows and others; `{code, refused, sessions}` where `windowsIn` refuses and `{code, refused}` where
`read` does. Malformed parameters are the request's fault. -/
def windowsPiece (j : Json) : V Json := do
  unless keys? j == some #["key", "label", "lane", "ledger", "start", "stop"] do
    throw "a request's keys are key, label, lane, ledger, start, stop"
  let p ← Params.of (getD j "key") (getD j "label") (getD j "lane") (getD j "start") (getD j "stop")
  return match read p (getD j "ledger") with
    | .error (code, why) => Json.mkObj (refused code why)
    | .ok r =>
      let sessions := ("sessions", Json.arr (r.sessions.map Session.toJson))
      match windowsIn p r with
      | .ok ws => Json.mkObj [("others", Json.arr (ws.others.map Session.toJson)), sessions,
          ("windows", Json.arr (ws.windows.map Window.toJson))]
      | .error (code, why) => Json.mkObj (refused code why ++ [sessions])

end Spot.Period
