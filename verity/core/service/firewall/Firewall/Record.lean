/-
# The firewall's record (`firewall/v0`, live PROTOCOL.md §10.5)

`record.jsonl` in the firewall's state directory: its events in order, one line each, as Python's `json.dumps(ev,
sort_keys=True)` writes it, each with its `seq`. Rust's `flock_live::release` and Python's `verity_flock.firewall` keep it
byte for byte (`live/src/firewall_record_vectors.json`); this is the same record in Lean, and the same rules over it:

1. A recursive session's V* statements and their order are fixed when it opens (`parts`), and it ends at its first stop.
   `admit` serves only the next statement not yet released, once: a request for any other is a stop at that next
   statement, never opened. After a stop, and after the last statement's release, every outer session of it is refused.
2. One outer slot (`slot`): a holder that ended before its session's outcome is recorded as that session's stop by the
   next one to take the slot.
3. A recursive session opened with a release schedule (`start_ms`, one window a statement) releases statement `j` at
   `start_ms + windows_ms[0] + ... + windows_ms[j]` (`releaseMs`); a missed time is a stop (`Firewall.Release`).
4. A stream opened with a failure `budget` b stops at its b-th failure (`failures`): a recursive session's stop, or a
   refusal of a recursive session that had neither stopped nor released its last statement, on a stream not yet stopped.

The functions here are pure: each takes the events read so far and returns its result and the events after it, which only
ever extend them. `Record.read` reads the record whole and fails closed: every line must be exactly a line the firewall
writes, with its `seq` its position, so a torn line (a writer killed inside its append), and any line the firewall would
not have written, refuses every call. Rust and Python read any JSON line and default what they can't find; Lean refuses it
instead. Their other liberties Lean doesn't take either: a `\r` before a line's end, a budget or release time past 2^64
(Rust's u64), a statement name that isn't a string.
-/
import Lean.Data.Json
import Flock.Canon
import Firewall.PyJson

namespace Core.Firewall

open Lean (Json)
open Flock (V)

def KIND : String := "firewall/v0"

/-- The inner session's part name, never a statement's. -/
def INNER : String := "inner"

/-- A release schedule (rule 3): `start_ms`, and one window a statement. -/
structure Schedule where
  start : Nat
  windows : Array Nat
  deriving BEq, Inhabited, Repr

/-- A recursive session's `open` event: its stream, its statements in their order, the stream's budget, its schedule. -/
structure Open where
  session : String
  stream : String
  parts : Array String
  budget : Option Nat
  release : Option Schedule
  deriving Inhabited, Repr

inductive Event where
  | opened (o : Open)
  /-- the recursive session's inner session finished (`part` is `inner`) -/
  | finish (session : String)
  | stop (session part why : String)
  | release (session part : String)
  | refused (session part : String)
  | slot (session part : String)
  | wait (session part : String)
  deriving Inhabited, Repr

abbrev Events := Array Event

def Event.session : Event → String
  | .opened o => o.session
  | .finish s | .stop s _ _ | .release s _ | .refused s _ | .slot s _ | .wait s _ => s

/-! ## The record's lines -/

private def kv (k v : String) : String := pyStr k ++ ": " ++ v

private def obj (items : List String) : String := "{" ++ ", ".intercalate items ++ "}"

def Schedule.py (r : Schedule) : String :=
  obj [kv "start_ms" (toString r.start), kv "windows_ms" ("[" ++ ", ".intercalate (r.windows.toList.map toString) ++ "]")]

/-- The event's line, without its newline, at position `seq`. -/
def Event.line (seq : Nat) : Event → String
  | .opened o => obj <|
      (o.budget.toList.map fun b => kv "budget" (toString b)) ++
      [kv "event" (pyStr "open"), kv "parts" (pyStrs o.parts)] ++
      (o.release.toList.map fun r => kv "release" r.py) ++
      [kv "seq" (toString seq), kv "session" (pyStr o.session), kv "stream" (pyStr o.stream)]
  | .finish s => tail "finish" s INNER []
  | .stop s p why => tail "stop" s p [kv "why" (pyStr why)]
  | .release s p => tail "release" s p []
  | .refused s p => tail "refused" s p []
  | .slot s p => tail "slot" s p []
  | .wait s p => tail "wait" s p []
where
  tail (event s p : String) (rest : List String) : String :=
    obj ([kv "event" (pyStr event), kv "part" (pyStr p), kv "seq" (toString seq), kv "session" (pyStr s)] ++ rest)

/-- The lines of `evs` from position `start` on, each with its newline: what appending them writes. -/
def linesFrom (evs : Events) (start : Nat) : String :=
  String.join ((List.range (evs.size - start)).map fun k => (evs[start + k]!).line (start + k) ++ "\n")

/-! ## What an open must be -/

/-- A recursive session's statements (rule 1): at least one, distinct, each a name other than `inner`. -/
def partsOk (parts : Array String) : V Unit :=
  let distinct := (List.range parts.size).all fun i => (List.range parts.size).all fun j => i == j || parts[i]! != parts[j]!
  if parts.isEmpty || !distinct || parts.any (fun p => p.isEmpty || p == INNER) then
    throw s!"a recursive session's statements are distinct names, at least one, none of them {INNER}, not {pyStrs parts}"
  else pure ()

private def natOf : Json → Option Nat
  | Json.num n => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
  | _ => none

/-- A release schedule (rule 3) for `n` statements: exactly `{"start_ms": S, "windows_ms": [one positive width a
statement]}`. -/
def scheduleOf (r : Json) (n : Nat) : V Schedule :=
  let bad : V Schedule := throw (s!"a release schedule is \{\"start_ms\": S, \"windows_ms\": [one width > 0 for each " ++
    s!"of the {n} statements]}, not {pyJson r}")
  match r with
  | Json.obj kvs =>
    match kvs.toArray.qsort (fun a b => a.1 < b.1) with
    | #[("start_ms", s), ("windows_ms", Json.arr ws)] =>
      match natOf s, ws.mapM natOf with
      | some s, some ws => if ws.size == n && ws.all (· > 0) then pure ⟨s, ws⟩ else bad
      | _, _ => bad
    | _ => bad
  | _ => bad

def budgetOk : Option Nat → V Unit
  | some 0 => throw "a failure budget is a positive integer, not 0"
  | _ => pure ()

/-! ## Reading the record -/

private def eventOf (j : Json) : V (Event × Nat) := do
  let ctx := "event"
  let kind ← Flock.J.str j "event" ctx
  let seq ← Flock.J.nat j "seq" ctx
  let session ← Flock.J.str j "session" ctx
  let part := Flock.J.str j "part" ctx
  let ev ← match kind with
    | "open" =>
      let parts ← (← Flock.J.arr j "parts" ctx).mapM (Flock.J.asStr · "event.parts")
      partsOk parts
      let budget ← if Flock.J.has j "budget" then some <$> Flock.J.nat j "budget" ctx else pure none
      budgetOk budget
      let release ← if Flock.J.has j "release" then some <$> scheduleOf (← Flock.J.get j "release" ctx) parts.size else pure none
      pure (.opened { session, stream := ← Flock.J.str j "stream" ctx, parts, budget, release })
    | "finish" => pure (.finish session)
    | "stop" => pure (.stop session (← part) (← Flock.J.str j "why" ctx))
    | "release" => pure (.release session (← part))
    | "refused" => pure (.refused session (← part))
    | "slot" => pure (.slot session (← part))
    | "wait" => pure (.wait session (← part))
    | k => throw s!"no event {k} in {KIND}"
  pure (ev, seq)

/-- The firewall's state: the record's text and its events. -/
structure Record where
  text : String
  events : Events
  deriving Inhabited

/-- The record whole, or why the firewall serves nothing (a line it can't read, or one it didn't write). Blank lines are
skipped, as Rust's and Python's readers skip them, and the last line needs no newline. -/
def Record.read (text : String) : V Record := do
  let lines := (text.splitOn "\n").filter (!·.isEmpty)
  let mut evs : Events := #[]
  for line in lines do
    let k := evs.size
    let ev ← match Json.parse line with
      | .error e => throw s!"the record's line {k + 1} is not JSON ({e}): a torn write, so the firewall serves nothing"
      | .ok j =>
        match eventOf j with
        | .error e => throw s!"the record's line {k + 1} is not an event of {KIND}: {e}"
        | .ok (ev, _) => pure ev
    if ev.line k != line then
      throw s!"the record's line {k + 1} is not the firewall's own (event {k} as the firewall writes it is {ev.line k})"
    evs := evs.push ev
  return { text, events := evs }

/-- The record after `evs`, which extend its events: its text with their lines appended. -/
def Record.extend (r : Record) (evs : Events) : Record := { text := r.text ++ linesFrom evs r.events.size, events := evs }

/-- One op as the firewall runs it on its file: read the record whole, act on its events, append the events the op adds.
Rust and Python read the file again at every op, under its lock, so a record one op left unreadable (its line appended to
a last line without a newline) is the next op's error. -/
def Record.op {α : Type} (r : Record) (f : Events → α × Events) : V (α × Record) := do
  let r ← Record.read r.text
  let (a, evs) := f r.events
  pure (a, r.extend evs)

/-! ## The rules over the events -/

/-- The recursive session's `open`. -/
def openOf (evs : Events) (s : String) : Option Open :=
  evs.findSome? fun | .opened o => if o.session == s then some o else none | _ => none

/-- Where the recursive session stopped: its first stop's part. -/
def stopOf (evs : Events) (s : String) : Option String :=
  evs.findSome? fun | .stop s' p _ => if s' == s then some p else none | _ => none

/-- Whether the recursive session's inner session finished. -/
def finished (evs : Events) (s : String) : Bool :=
  evs.any fun | .finish s' => s' == s | _ => false

/-- The stream's budget: its first open's. -/
def budgetOf (evs : Events) (stream : String) : Option Nat :=
  (evs.findSome? fun | .opened o => if o.stream == stream then some o else none | _ => none).bind (·.budget)

/-- The position of the recursive session's next statement, the first of its `parts` not released; none once all are. -/
def nextOf (evs : Events) (o : Open) : Option Nat :=
  let released := evs.filterMap fun | .release s p => if s == o.session then some p else none | _ => none
  o.parts.findIdx? fun p => !released.contains p

structure Tally where
  /-- the stream's recursive sessions and their statements, the latest open first -/
  mine : List (String × Array String) := []
  released : List (String × String) := []
  stopped : List String := []
  n : Nat := 0

/-- One event of rule 4's count on a stream with budget `b`. -/
def Tally.step (stream : String) (b : Option Nat) (t : Tally) : Event → Tally
  | .opened o => if o.stream == stream then { t with mine := (o.session, o.parts) :: t.mine } else t
  | .release s p => if (t.mine.lookup s).isSome then { t with released := (s, p) :: t.released } else t
  | .stop s _ _ => if (t.mine.lookup s).isSome then { t with stopped := s :: t.stopped, n := t.n + 1 } else t
  | .refused s _ =>
    match t.mine.lookup s with
    | some ps =>
      if !(ps.all fun p => t.released.contains (s, p)) && !t.stopped.contains s && b.all (t.n < ·) then
        { t with n := t.n + 1 }
      else t
    | none => t
  | _ => t

/-- The failures on `stream` (rule 4). -/
def failures (evs : Events) (stream : String) : Nat :=
  (evs.foldl (Tally.step stream (budgetOf evs stream)) {}).n

/-- The last `slot` event (its position, session and part), if no outcome of its session at its part (`release`, `stop`
or `refused`) follows it. -/
def orphan (evs : Events) : Option (Nat × String × String) :=
  match (List.range evs.size).reverse.findSome? fun k => match evs[k]! with
      | .slot s p => some (k, s, p)
      | _ => none with
  | none => none
  | some (k, s, p) =>
    let done := (evs.extract (k + 1) evs.size).any fun
      | .release s' p' | .refused s' p' => s' == s && p' == p
      | .stop s' _ _ => s' == s
      | _ => false
    if done then none else some (k, s, p)

/-- When statement `j` is released under `release` (rule 3): the end of its window. -/
def releaseMs (release : Option Schedule) (j : Nat) : Option Nat :=
  release.map fun r => r.start + (r.windows.toList.take (j + 1)).foldl (· + ·) 0

private def showBudget : Option Nat → String
  | some b => toString b
  | none => "None"

/-- `rec_live` opens a recursive session: why it may not (its statements, budget or schedule), its id, or why the
firewall refuses it (the stream stopped at its budget). The first check needs no record: `openArgs`. -/
inductive Opened where
  | ok (session : String)
  | refused (why : String)
  | error (why : String)

/-- What an open is given, checked before the record is read. -/
def openArgs (parts : Array String) (budget : Option Nat) (release : Option Json) : V (Option Schedule) := do
  partsOk parts
  budgetOk budget
  match release with
  | some r => some <$> scheduleOf r parts.size
  | none => pure none

private def sid (n : Nat) : String :=
  let d := toString n
  "r" ++ String.ofList (List.replicate (4 - d.length) '0') ++ d

def openSession (evs : Events) (stream : String) (parts : Array String) (budget : Option Nat) (release : Option Schedule) :
    Opened × Events :=
  let first := evs.findSome? fun | .opened o => if o.stream == stream then some o else none | _ => none
  match first with
  | some f =>
    if f.budget != budget then (.error s!"stream {stream}'s budget is {showBudget f.budget}, not {showBudget budget}", evs)
    else go
  | none => go
where
  go : Opened × Events :=
    let n := failures evs stream
    match budget with
    | some b =>
      if n ≥ b then (.refused s!"stream {stream} stopped at its failure budget ({n} of {b})", evs) else made
    | none => made
  made : Opened × Events :=
    let s := sid (evs.filter (fun | .opened _ => true | _ => false)).size
    (.ok s, evs.push (.opened { session := s, stream, parts, budget, release }))

def finish (evs : Events) (s : String) : Events := evs.push (.finish s)

/-- The recursive session stops at `part`: its one failure, recorded once (false if it had already stopped). -/
def stop (evs : Events) (s part why : String) : Bool × Events :=
  if (stopOf evs s).isSome then (false, evs) else (true, evs.push (.stop s part why))

/-- An outer proof of the recursive session was released (the challenger's verdict accepted it). -/
def release (evs : Events) (s part : String) : Events := evs.push (.release s part)

/-- The firewall's one outer slot, taken by a session that found it free: a `slot` whose session has no outcome after it
is that session's stop, then `slot`. -/
def slot (evs : Events) (s part : String) : Events :=
  let evs := match orphan evs with
    | some (seq, s', p') =>
      if (stopOf evs s').isNone then evs.push (.stop s' p' s!"the slot's holder ended before its outcome (slot {seq})")
      else evs
    | none => evs
  evs.push (.slot s part)

/-- An admitted outer session: its recursive session's stream, its statement's position, and when its proofs are released
(rule 3), if scheduled. -/
structure Admitted where
  stream : String
  index : Nat
  releaseMs : Option Nat
  deriving BEq, Repr

/-- Whether an outer session of `s` at `part` may be served now: its recursive session is open, has not stopped, has a
statement not yet released, its stream has not stopped at its budget, its inner session finished, and `part` is its next
statement. A request for any other statement is the recursive session's stop at the next one (never opened). A refusal is
recorded (`refused`). -/
def admit (evs : Events) (s part : String) : Except String Admitted × Events :=
  match openOf evs s with
  | none => refuse s!"recursive session {s} was never opened" evs
  | some o =>
    match stopOf evs s with
    | some p => refuse s!"recursive session {s} stopped at {p}: the firewall serves nothing more of it" evs
    | none =>
      match nextOf evs o with
      | none => refuse s!"recursive session {s} released all its statements: the firewall serves nothing more of it" evs
      | some j =>
        let n := failures evs o.stream
        match budgetOf evs o.stream with
        | some b =>
          if n ≥ b then refuse s!"stream {o.stream} stopped at its failure budget ({n} of {b})" evs
          else served o j
        | none => served o j
where
  refuse (why : String) (evs : Events) : Except String Admitted × Events := (.error why, evs.push (.refused s part))
  served (o : Open) (j : Nat) : Except String Admitted × Events :=
    if !finished evs s then refuse s!"recursive session {s}: its inner session has not finished" evs
    else
      let nxt := o.parts[j]!
      if nxt != part then
        refuse s!"recursive session {s} stopped at {nxt}: {part} is not its next statement"
          (evs.push (.stop s nxt s!"never opened: {part} was asked in its place"))
      else (.ok { stream := o.stream, index := j, releaseMs := releaseMs o.release j }, evs)

end Core.Firewall
