/-
# One outer session, held to the record's rules (Rust's `release::outer_session`)

Take the slot, then admit the session (`Firewall.Record`); `prove` runs it up to the barrier where its proofs would leave,
`ready` is the send check's verdict on what it proved, then at its release time if it has one (a missed time is a stop) the
proofs are sent and the challenger's verdict decides release or stop. `decide` is everything before the send: the
firewall's own decision, which `Firewall.Protocol` checks with the send check as `Flock.Firewall.Contract.run`. The prover,
the send check and the send are given here as what they returned, a fault (Rust's panic) included: a fault anywhere is a
stop, never the firewall's end. Each of the session's ops on the record is one `Record.op`, which reads the record again,
as Rust's does: an error is a record the firewall can't read, and the session ends there with nothing sent.
-/
import Firewall.Record

namespace Core.Firewall

open Flock (V)

/-- What a step outside the firewall (the prover, the send check) returned. -/
inductive Run where
  | ok
  | err (why : String)
  /-- a fault, Rust's panic, with its message -/
  | fault (msg : String)
  deriving BEq, Inhabited, Repr

def faultWhy (msg : String) : String := s!"the firewall's fault (a panic: {String.ofList (msg.toList.take 200)})"

/-- The firewall's decision on an admitted or refused outer session, before anything is sent. -/
inductive Decision where
  /-- refused before the prover ran: nothing reaches the challenger -/
  | refused (why : String)
  /-- admitted, then stopped before its proofs left -/
  | stop (why : String)
  /-- the proofs may leave, at `a.releaseMs` if scheduled -/
  | go (a : Admitted)
  deriving Inhabited

/-- What the prover, the send check and the clock (`now`, ms) decide of an admitted session. -/
def Decision.after (a : Admitted) (now : Nat) (prove ready : Run) : Decision :=
  match prove with
  | .err why => .stop why
  | .fault m => .stop (faultWhy m)
  | .ok =>
    match ready with
    | .err why => .stop s!"the send check refused: {why}"
    | .fault m => .stop (faultWhy m)
    | .ok =>
      match a.releaseMs with
      | some t => if now > t then .stop s!"missed its release time ({now} ms, after {t} ms)" else .go a
      | none => .go a

/-- Everything before the send, at time `now` (ms): the decision and the record after it. A stop's event is not yet
recorded (`outerSession` records it). -/
def decide (r : Record) (s part : String) (now : Nat) (prove ready : Run) : V (Decision × Record) := do
  let ((), r) ← r.op fun evs => ((), slot evs s part)
  match ← r.op (admit · s part) with
  | (.error why, r) => pure (.refused why, r)
  | (.ok a, r) => pure (.after a now prove ready, r)

inductive Kind where
  | release
  | stop
  | refused
  deriving BEq, Inhabited, Repr

def Kind.name : Kind → String
  | .release => "release"
  | .stop => "stop"
  | .refused => "refused"

/-- An outer session's outcome: `release`, `stop` or `refused`, why when not released, and the release time it waited
for before sending (Rust's clock's `sleep_until_ms`), if scheduled and sent. -/
structure Served where
  kind : Kind
  why : Option String
  sentAt : Option Nat
  deriving Inhabited, Repr

/-- One outer session: `send` is the challenger's verdict on the proofs sent (true: accepted), or a fault while sending. -/
def outerSession (r : Record) (s part : String) (now : Nat) (prove ready : Run) (send : Except String Bool) :
    V (Served × Record) := do
  match ← decide r s part now prove ready with
  | (.refused why, r) => pure ({ kind := .refused, why := some why, sentAt := none }, r)
  | (.stop why, r) => stopped why none r
  | (.go a, r) =>
    match send with
    | .ok true =>
      let ((), r) ← r.op fun evs => ((), release evs s part)
      pure ({ kind := .release, why := none, sentAt := a.releaseMs }, r)
    | .ok false => stopped "the challenger's verdict rejects it" a.releaseMs r
    | .error m => stopped (faultWhy m) a.releaseMs r
where
  stopped (why : String) (sentAt : Option Nat) (r : Record) : V (Served × Record) := do
    let (_, r) ← r.op (stop · s part why)
    pure ({ kind := .stop, why := some why, sentAt }, r)

end Core.Firewall
