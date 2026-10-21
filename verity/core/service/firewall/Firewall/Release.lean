/-
# One outer session's stages, held to the record's rules

Take the slot, then admit the session (`seat`, `Firewall.Record`); `prove` runs it up to the barrier where its proofs
would leave, `ready` is the send check's verdict on what it proved, then at its release time if it has one (a missed time
is a stop) the proofs are sent (`Decision.after`), and the challenger's verdict decides release or stop (`settle`).
`decide` is everything before the send: the firewall's own decision, which `Firewall.Protocol` checks with the send check
as `Flock.Firewall.Contract.run`. The prover and the send check are given as what they returned, a fault included: a fault
anywhere is a stop, never the firewall's end. Each of the session's ops on the record is one `Record.op`, which reads the
record again: an error is a record the firewall can't read, and the session ends there with nothing sent.
`Firewall.Loop` runs these stages on the record's file around the session loop (`core-firewall session`), and the
difftest's `outer` (`Firewall.Json`) runs them in turn on a record and a clock it is given.
-/
import Firewall.Record

namespace Core.Firewall

open Flock (V)

/-- What a step outside the firewall (the prover, the send check) returned. -/
inductive Run where
  | ok
  | err (why : String)
  /-- a fault, with its message -/
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

/-- The session's ops once it holds the slot: the slot's events, then its admission, and the record after them. -/
def seat (r : Record) (s part : String) : V (Except String Admitted × Record) := do
  let ((), r) ← r.op fun evs => ((), slot evs s part)
  r.op (admit · s part)

/-- Everything before the send, at time `now` (ms): the decision and the record after it. A stop's event is not yet
recorded (`stopAt` records it). -/
def decide (r : Record) (s part : String) (now : Nat) (prove ready : Run) : V (Decision × Record) := do
  match ← seat r s part with
  | (.error why, r) => pure (.refused why, r)
  | (.ok a, r) => pure (.after a now prove ready, r)

/-- The stop of `s` at `part` for `why`, recorded. -/
def stopAt (r : Record) (s part why : String) : V (Unit × Record) := do
  let (_, r) ← r.op (stop · s part why)
  pure ((), r)

/-- What the challenger's verdict on the proofs sent does: accepted (`.ok true`) is the release; rejected, or an error
while sending (its why), is the stop, recorded. The stop's why, if it stopped. -/
def settle (r : Record) (s part : String) (verdict : Except String Bool) : V (Option String × Record) := do
  let why := match verdict with
    | .ok true => none
    | .ok false => some "the challenger's verdict rejects it"
    | .error why => some why
  match why with
  | none =>
    let ((), r) ← r.op fun evs => ((), release evs s part)
    pure (none, r)
  | some w =>
    let ((), r) ← stopAt r s part w
    pure (some w, r)

end Core.Firewall
