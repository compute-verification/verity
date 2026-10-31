/-
# The proof system's stream: its own rules (`service._Control`)

What `service.Stream` decides of each op on its own, around the checks whose verifiers of record are their own pieces
(`Service.checkPublic`, `Registration.check`, `Select.select`, `Draw.checkDraw`, `Service.checkVerdicts`, `Record.record`,
`Run.run`): the failure budget, the registrations open at once, the epoch's salt and registration deadline, and each
registration's once-only orders (drawn, session, delivered, final) and proof deadline. `run` decides one op on
what the rules hold of the stream (a `State`) and gives what they hold after it, as `service._step` does. The stream holds
its state of record from these outputs alone, from the empty state.

A state holds the stream's counts and only the entries of the registrations the op reads: the ones it names, and the first
round of a session that covers one of them. So an op is decided on a state of a bounded size whatever the stream's length,
and an op that reads an entry the state doesn't hold is an error. As JSON:

* `size`, the registrations received; `opened`, those without a final outcome, in the order received; `failures`;
* `salted_at`, the salt's release or null; `ended`, null or the late registration's readings `{received_at, salted_at}`;
* `halted`, null or the cause of the halt of the stream's operator (`Fleet`, R5-all);
* `entries`, in order, each `{order, registration, draw, proved, drawn_at, delivered, session, final}`: `draw`
  null until drawn, then with `proved` and `drawn_at`, and `session` null or the covered draws of the session over it.

An op is an object with `op` and the fields it reads:

* `ready`; `salt` `{at}`; `salt_released` `{released_at, now}`;
* `register` `{received_at, checks, registration}`;
* `select` `{order, registration, checks, draw, proved, at}`;
* `session` `{draws}`, each draw `[order, registration, draw]`;
* `deliver` `{order, registration, draw, sessions, now}`, and `deliver_session` `{stream, draws, sessions, now}`;
* `outcome` `{order, registration, draw, now, verdicts, accepted}`;
* `halt` `{cause}`, the fleet's halt of the stream's operator (`Fleet.failures`).

`registration` and `draw` are the challenger's tokens for them (the SHA-512 of a `Registration`'s fields and of a
`Draw`'s), so two are the same exactly when Python's objects are equal; `order` is the registration's, or null where it
isn't an integer. Readings are the challenger's clock: a JSON number, kept as written (so the state gives it back as Python
wrote it) and read as the double nearest it (`PyJson.nearest`, so a Python float reads back exactly), with Python's float
arithmetic. `checks` are the checks' codes, or null where they pass; `verdicts` the verdicts' hold's codes or null;
`accepted` the audit records' and the run's verdict; `proved` the draw's proved units' count; `sessions` the delivered
session records' SHA-512.

A decision is `{"refused": codes}`, `{"pending": true}`, `{"ok": value}`, or `{"unchecked": field}` where the op passes
the stream's rules up to a field it doesn't carry yet (a reading, the checks, the verdicts): the challenger reads its clock
or runs the checks only then, and asks again. A refusal changes nothing but a late registration's, which ends the epoch.
An error is an op or a state this port doesn't read.

A halt is final. It ends every registration without a final outcome at once, as `halted`, with no failure of its own:
nothing accepts it after that, and outcomes already final stay. A halted stream refuses every op (`halted`) and changes
nothing, so it receives no registration again.
-/
import SampledProofs.Run

namespace SampledProofs.Stream

open Lean (Json)
open Flock (V)

/-- The most rounds one proof session covers (`service.ROUNDS`). -/
def ROUNDS : Nat := 4

def ACCEPT : String := "accept"
def REJECT : String := "reject"
def LATE : String := "late"

/-- A JSON number as the double nearest it: Python's float of it, exactly. -/
def float? : Json → Option Float
  | .num ⟨m, e⟩ => (PyJson.nearest m.natAbs (10 ^ e)).map fun (mm, ee) =>
      let f := mm.toUInt64.toFloat.scaleB ee
      if m < 0 then -f else f
  | _ => none

/-- A reading as written, where it is one. -/
def reading? (j : Json) : Option Json := (float? j).map fun _ => j

/-- The stream's public parameters: its failure budget, its rounds, its check's deadlines (seconds on the
challenger's clock) and its digest. -/
structure Params where
  budget : Nat
  rounds : Nat
  registrationDeadline : Option Float
  proofDeadline : Float
  stream : String

def Params.ofJson (j : Json) : V Params := do
  if keys? j != some #["budget", "proof_deadline", "registration_deadline", "rounds", "stream"] then
    throw "params are exactly {budget, rounds, registration_deadline, proof_deadline, stream}"
  let some budget := nat? (getD j "budget") | throw "a failure budget is a positive integer"
  if budget == 0 then throw "a failure budget is a positive integer"
  let some rounds := nat? (getD j "rounds") | throw s!"a session covers 1 to {ROUNDS} rounds"
  if rounds == 0 || rounds > ROUNDS then throw s!"a session covers 1 to {ROUNDS} rounds"
  let some pd := float? (getD j "proof_deadline") | throw "a proof deadline is a positive number of seconds"
  if !(pd > 0) then throw "a proof deadline is a positive number of seconds"
  let rd ← match getD j "registration_deadline" with
    | .null => pure none
    | x => match float? x with
      | some d => if d > 0 then pure (some d) else throw "a registration deadline is a positive number of seconds"
      | none => throw "a registration deadline is a positive number of seconds"
  let .str stream := getD j "stream" | throw "stream is the stream's digest"
  return { budget, rounds, registrationDeadline := rd, proofDeadline := pd, stream }

/-- A covered draw: its registration's order (none where it isn't a natural number) and token, and the draw's token. -/
abbrev Covered := Option Nat × String × String

def covered? : Json → Option Covered
  | .arr #[o, .str r, .str d] => some (nat? o, r, d)
  | _ => none

def Covered.toJson (c : Covered) : Json :=
  Json.arr #[(c.1.map fun o => Lean.toJson o).getD .null, .str c.2.1, .str c.2.2]

/-- What the stream's rules hold of one registration. -/
structure Entry where
  order : Nat
  registration : String
  draw : Option String := none
  proved : Nat := 0
  drawnAt : Option Json := none
  delivered : Bool := false
  session : Option (Array Covered) := none
  final : Bool := false
  deriving Inhabited

structure State where
  size : Nat := 0
  /-- the registrations without a final outcome, in the order they were received -/
  opened : Array Nat := #[]
  failures : Nat := 0
  saltedAt : Option Json := none
  ended : Option Json := none
  /-- the cause of the operator's halt, once its fleet halted it -/
  halted : Option Json := none
  /-- the entries the op reads, in order -/
  entries : Array Entry := #[]

private def str? : Json → Option String
  | .str s => some s
  | _ => none

private def bool? : Json → Option Bool
  | .bool b => some b
  | _ => none

/-- null as none, else `f`'s reading of it or the error `msg`. -/
private def nullOr {α} (j : Json) (f : Json → Option α) (msg : String) : V (Option α) :=
  match j with
  | .null => pure none
  | x => match f x with
    | some a => pure (some a)
    | none => throw msg

private def ascending : List Nat → Bool
  | a :: b :: rest => a < b && ascending (b :: rest)
  | _ => true

def Entry.ofJson (j : Json) : V Entry := do
  if keys? j != some #["delivered", "draw", "drawn_at", "final", "order", "proved", "registration", "session"]
  then throw "an entry is exactly {order, registration, draw, proved, drawn_at, delivered, session, final}"
  let some order := nat? (getD j "order") | throw "an entry's order is a natural number"
  let some registration := str? (getD j "registration") | throw "an entry's registration is a token"
  let draw ← nullOr (getD j "draw") str? "an entry's draw is null or a token"
  let some proved := nat? (getD j "proved") | throw "an entry's proved is a count"
  let drawnAt ← nullOr (getD j "drawn_at") reading? "an entry's drawn_at is null or a reading"
  let some delivered := bool? (getD j "delivered") | throw "an entry's delivered is a boolean"
  let session ← nullOr (getD j "session") (fun | .arr cs => cs.mapM covered? | _ => none)
    "an entry's session is null or its covered draws"
  let some final := bool? (getD j "final") | throw "an entry's final is a boolean"
  return { order, registration, draw, proved, drawnAt, delivered, session, final }

def State.ofJson (j : Json) : V State := do
  if keys? j != some #["ended", "entries", "failures", "halted", "opened", "salted_at", "size"] then
    throw "a state is exactly {size, opened, failures, salted_at, ended, halted, entries}"
  let some size := nat? (getD j "size") | throw "size is a count"
  let .arr os := getD j "opened" | throw "opened is a list of orders"
  let some opened := os.mapM nat? | throw "opened is a list of orders"
  let some failures := nat? (getD j "failures") | throw "failures is a count"
  let saltedAt ← nullOr (getD j "salted_at") reading? "salted_at is null or a reading"
  let ended ← nullOr (getD j "ended") (fun | .obj kvs => some (Json.obj kvs) | _ => none)
    "ended is null or the late registration's readings"
  let halted ← nullOr (getD j "halted") (fun | .obj kvs => some (Json.obj kvs) | _ => none)
    "halted is null or the halt's cause"
  let .arr es := getD j "entries" | throw "entries is a list"
  let entries ← es.mapM Entry.ofJson
  if !opened.all (· < size) then throw "an opened order is a registration's"
  let orders := (entries.map (·.order)).toList
  if !(orders.all (· < size) && ascending orders) then throw "entries are registrations', in order, once each"
  return { size, opened, failures, saltedAt, ended, halted, entries }

def Entry.toJson (e : Entry) : Json := Json.mkObj [
  ("order", Lean.toJson e.order), ("registration", .str e.registration), ("draw", (e.draw.map .str).getD .null),
  ("proved", Lean.toJson e.proved), ("drawn_at", e.drawnAt.getD .null),
  ("delivered", .bool e.delivered), ("session", (e.session.map fun cs => Json.arr (cs.map Covered.toJson)).getD .null),
  ("final", .bool e.final)]

def State.toJson (s : State) : Json := Json.mkObj [
  ("size", Lean.toJson s.size), ("opened", natsJson s.opened), ("failures", Lean.toJson s.failures),
  ("salted_at", s.saltedAt.getD .null), ("ended", s.ended.getD .null), ("halted", s.halted.getD .null),
  ("entries", Json.arr (s.entries.map Entry.toJson))]

inductive Decision where
  | refused (codes : Array String)
  | pending
  | unchecked (field : String)
  | ok (value : Json)

def Decision.toJson : Decision → Json
  | .refused cs => Json.mkObj [("refused", Json.arr (cs.map Json.str))]
  | .pending => Json.mkObj [("pending", Json.bool true)]
  | .unchecked f => Json.mkObj [("unchecked", Json.str f)]
  | .ok v => Json.mkObj [("ok", v)]

/-- A step: it ends early with a decision that changes nothing (`stop`), or errs on an op it doesn't read (`bad`). -/
abbrev M := StateT State (ExceptT Decision V)

def stop {α} (d : Decision) : M α := throwThe Decision d
def no {α} (code : String) : M α := stop (.refused #[code])
def bad {α} (msg : String) : M α := throwThe String msg

/-- The op's field `k`, or `unchecked` where it doesn't carry it yet. -/
def need (op : Json) (k : String) : M Json :=
  match field? op k with
  | some x => pure x
  | none => stop (.unchecked k)

/-- The reading `k`, as written and as a double. -/
def reading (op : Json) (k : String) : M (Json × Float) := do
  let j ← need op k
  let some f := float? j | bad s!"{k} is a reading of the clock"
  pure (j, f)

def token (j : Json) (k : String) : M String :=
  match j with
  | .str s => pure s
  | _ => bad s!"{k} is a token"

/-- `checks` (or `verdicts`): null where they passed, else their codes. -/
def codes (op : Json) (k : String) : M (Option (Array String)) := do
  match ← need op k with
  | .null => pure none
  | .arr xs => some <$> xs.mapM fun | .str c => pure c | _ => bad s!"{k} are codes"
  | _ => bad s!"{k} are null or codes"

def ready (p : Params) : M Unit := do
  let s ← get
  if s.failures ≥ p.budget then no "stopped"
  if s.ended.isSome then no "epoch-ended"
  if s.opened.size ≥ min p.rounds (p.budget - s.failures) then no "open"

/-- The entry of order `o`, which the state holds. -/
def held (o : Nat) : M Entry := do
  let some e := (← get).entries.find? (·.order == o) | bad s!"the state holds no entry {o}"
  pure e

/-- `Stream._entry`: the registration of that order, if it is that one. -/
def entryAt (o : Option Nat) (reg : String) : M Entry := do
  let some o := o | no "unknown"
  if o ≥ (← get).size then no "unknown"
  let e ← held o
  if e.registration != reg then no "unknown"
  pure e

def entryOf (op : Json) : M Entry := do
  entryAt (nat? (getD op "order")) (← token (getD op "registration") "registration")

def drawn (op : Json) (e : Entry) : M Unit := do
  if e.draw != some (← token (getD op "draw") "draw") then no "not-drawn"

def put (e : Entry) : M Unit :=
  modify fun s => { s with entries := s.entries.map fun x => if x.order == e.order then e else x }

/-- `Stream._finish`: the final outcome `sym`, one failure unless it accepts. -/
def finish (e : Entry) (sym : String) : M Unit := do
  put { e with final := true }
  modify fun s => { s with failures := s.failures + (if sym == ACCEPT then 0 else 1), opened := s.opened.erase e.order }

/-- `Stream._late`: past the proof deadline from the draw of the session's first round (its own, alone). -/
def late (p : Params) (e : Entry) (now : Float) : M Bool := do
  let first ← match e.session.bind (·[0]?) with
    | some (some o, _, _) => held o
    | _ => pure e
  let some t := first.drawnAt.bind float? | bad "a drawn entry has its draw's reading"
  pure (now - t > p.proofDeadline)

def register (p : Params) (op : Json) : M Decision := do
  ready p
  if let some d := p.registrationDeadline then
    let (tj, t) ← reading op "received_at"
    let s ← get
    let onTime := match s.saltedAt.bind float? with
      | some at_ => t - at_ ≤ d
      | none => false
    if !onTime then
      set { s with ended := some (Json.mkObj [("received_at", tj), ("salted_at", s.saltedAt.getD .null)]) }
      return .refused #["late-registration"]
  if let some cs ← codes op "checks" then stop (.refused cs)
  let reg ← token (← need op "registration") "registration"
  let s ← get
  let order := s.size
  set { s with size := order + 1, entries := s.entries.push { order, registration := reg },
               opened := s.opened.push order }
  return .ok (Lean.toJson order)

def select (op : Json) : M Decision := do
  let e ← entryOf op
  if e.draw.isSome then no "drawn"
  if let some cs ← codes op "checks" then stop (.refused cs)
  let draw ← token (← need op "draw") "draw"
  let some proved := nat? (← need op "proved") | bad "proved is a count"
  let (at_, _) ← reading op "at"
  put { e with draw := some draw, proved := proved, drawnAt := some at_ }
  return .ok Json.null

def draws (op : Json) : M (Array Covered) := do
  let .arr ds := getD op "draws" | bad "draws are a list"
  ds.mapM fun d => match covered? d with
    | some c => pure c
    | none => bad "a covered draw is [order, registration, draw]"

def session (p : Params) (op : Json) : M Decision := do
  let ds ← draws op
  if ds.size == 0 || ds.size > p.rounds then no "rounds"
  for (o, r, d) in ds do
    let e ← entryAt o r
    if e.draw != some d then no "not-drawn"
    if e.final then no "final"
    if e.delivered then no "delivered"
    if e.session.isSome then no "covered"
  let orders := ds.filterMap (·.1)
  if orders != (Array.range ds.size).map (orders[0]! + ·) then no "order"
  modify fun s => { s with entries := s.entries.map fun e => if orders.contains e.order then { e with session := some ds } else e }
  return .ok (natsJson orders)

def sessionsOk (op : Json) : Bool :=
  match getD op "sessions" with
  | .arr ss => !ss.isEmpty && ss.all Run.digest?
  | _ => false

def deliver (p : Params) (op : Json) : M Decision := do
  let e ← entryOf op
  drawn op e
  if e.final then no "final"
  if e.delivered then no "delivered"
  if e.session.isSome then no "covered"
  if !sessionsOk op then no "sessions"
  let isLate ← late p e (← reading op "now").2
  let e := { e with delivered := true }
  put e
  if isLate then
    finish e LATE
    return .ok (Json.str LATE)
  return .ok Json.null

def deliverSession (p : Params) (op : Json) : M Decision := do
  let stream ← token (getD op "stream") "stream"
  let ds ← draws op
  if stream != p.stream || ds.isEmpty then no "not-session"
  let mut es := #[]
  for (o, r, _) in ds do
    let e ← entryAt o r
    if e.session != some ds then no "not-session"
    if e.final then no "final"
    if e.delivered then no "delivered"
    es := es.push e
  if !sessionsOk op then no "sessions"
  let isLate ← late p es[0]! (← reading op "now").2
  for e in es do
    put { e with delivered := true }
  if !isLate then return .ok Json.null
  for e in es do
    finish { e with delivered := true } LATE
  return .ok (Json.arr (es.map fun _ => Json.str LATE))

def outcome (p : Params) (op : Json) : M Decision := do
  let e ← entryOf op
  drawn op e
  if e.final then no "final"
  if !e.delivered then
    if ← late p e (← reading op "now").2 then
      finish e LATE
      return .ok (Json.str LATE)
    stop .pending
  if let some cs ← codes op "verdicts" then
    stop (if cs == #["pending"] then .pending else .refused cs)
  let .bool accepted ← need op "accepted" | bad "accepted is a boolean"
  let sym := if accepted then ACCEPT else REJECT
  finish e sym
  return .ok (Json.str sym)

def unsalted : M Unit := do
  let s ← get
  if s.saltedAt.isSome then no "salted"
  if s.ended.isSome then no "epoch-ended"

/-- An entry once a halt ended the registrations `opened`: final if it is one of them, else as it was. -/
def haltEntry (opened : Array Nat) (e : Entry) : Entry := if opened.contains e.order then { e with final := true } else e

/-- `Stream._halt`: the fleet halted the stream's operator, for `cause`. Every registration without a final outcome ends
now, final, with no failure of its own; the decision is their orders. -/
def halt (op : Json) : M Decision := do
  let s ← get
  let cause := getD op "cause"
  if !(cause matches .obj _) then bad "a halt's cause is an object"
  else if !s.opened.all (fun o => s.entries.any (·.order == o)) then bad "the state holds every open entry"
  else do
    set { s with halted := some cause, opened := #[], entries := s.entries.map (haltEntry s.opened) }
    return .ok (natsJson s.opened)

def OPS : Array String :=
  #["ready", "salt", "salt_released", "register", "select", "session", "deliver", "deliver_session", "outcome", "halt"]

def opsMsg : String :=
  "an op is one of ready, salt, salt_released, register, select, session, deliver, deliver_session, outcome, halt"

/-- The op `name`'s own rule. -/
def act (p : Params) (op : Json) : String → M Decision
  | "ready" => do ready p; return .ok Json.null
  | "salt" => do
    unsalted
    let (at_, _) ← reading op "at"
    modify fun s => { s with saltedAt := some at_ }
    return .ok Json.null
  | "salt_released" => do
    unsalted
    let (rj, r) ← reading op "released_at"
    if r > (← reading op "now").2 then no "released-later"
    modify fun s => { s with saltedAt := some rj }
    return .ok Json.null
  | "register" => register p op
  | "select" => select op
  | "session" => session p op
  | "deliver" => deliver p op
  | "deliver_session" => deliverSession p op
  | "outcome" => outcome p op
  | "halt" => halt op
  | _ => bad opsMsg

/-- The op's rule, unless the stream halted: then every op is refused, `halted`. -/
def step (p : Params) (op : Json) : M Decision := do
  let s ← get
  match getD op "op" with
  | .str name => if !OPS.contains name then bad opsMsg else if s.halted.isSome then no "halted" else act p op name
  | _ => bad opsMsg

/-- The op's decision on `s`, and what the rules hold after it: `s` itself where the op ends early. -/
def run (p : Params) (s : State) (op : Json) : V (Decision × State) := do
  match ← ((step p op).run s).run with
  | .ok (d, s') => pure (d, s')
  | .error d => pure (d, s)

end SampledProofs.Stream
