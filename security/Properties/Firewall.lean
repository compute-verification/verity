module

public import Firewall.Release

/-!
# The ZK server's record: what its Lean code (`Firewall.Record`, `Firewall.Release`) does

The rules a recursive session's stop leak rests on (live PROTOCOL.md §10.5), stated against the release side's functions
as they run: `Core.Firewall.admit`, which admits an outer session on the record's events, `Core.Firewall.stop`, which
records a recursive session's stop, and `Core.Firewall.Decision.after`, which decides whether an admitted session's
proofs leave.

* `AdmitNext`: `admit` admits only the recursive session's next statement not yet released, in the order fixed when it
  opened, once its inner session finished and before any stop, released at its schedule's time, and admitting records
  nothing.
* `StopEnds`: once the recursive session stopped, `admit` refuses every part, recording only the refusal, and `stop`
  records no second stop.
* `OutOfTurnStops`: a part asked out of its turn is the recursive session's stop at its next statement, never opened.
* `AfterOnTime`: `Decision.after` lets the proofs of only the admitted session leave, only when the prover and the send
  check returned `ok`, and, where the session is scheduled, only at or before its release time.

`Proofs.Properties.Firewall` proves them, with `AdmitNext`'s and `AfterOnTime`'s honest and nontrivial lemmas at the
instances below.
-/

@[expose] public section

namespace Properties.Firewall

open Core.Firewall

/-- `admit evs s part` admitted `a` with the record after it `evs'`: the recursive session `s` was opened as `o`, has not
stopped, and its inner session finished; `a` is its next statement not yet released, which is `part`, on `o`'s stream,
released at `o`'s schedule's time for it; and admitting recorded nothing. -/
def AdmittedNext (evs : Events) (s part : String) (a : Admitted) (evs' : Events) : Prop :=
  evs' = evs ∧ ∃ o, openOf evs s = some o ∧ stopOf evs s = none ∧ finished evs s = true ∧
    nextOf evs o = some a.index ∧ o.parts[a.index]? = some part ∧ a.stream = o.stream ∧
    a.releaseMs = releaseMs o.release a.index

/-- **`admit` serves only the next statement, and only before any stop.** -/
def AdmitNext : Prop :=
  ∀ (evs : Events) (s part : String) (a : Admitted) (evs' : Events),
    admit evs s part = (.ok a, evs') → AdmittedNext evs s part a evs'

/-- The recursive session `s` ended on the record `evs`: once it stopped, `admit` refuses every part, recording only the
refusal, and `stop` records no second stop. -/
def Ended (evs : Events) (s : String) : Prop :=
  (stopOf evs s).isSome →
    (∀ part, ∃ why, admit evs s part = (.error why, evs.push (.refused s part))) ∧
      ∀ part why, stop evs s part why = (false, evs)

/-- **The first stop ends the recursive session.** -/
def StopEnds : Prop := ∀ (evs : Events) (s : String), Ended evs s

/-- `part` asked out of its turn on the record `evs`: if the recursive session `s` was opened as `o`, has not stopped,
its inner session finished, its stream is within its failure budget, and its next statement `j` is not `part`, then
`admit` refuses `part` and records the recursive session's stop at statement `j`, never opened. -/
def OutOfTurn (evs : Events) (s part : String) : Prop :=
  ∀ (o : Open) (j : Nat), openOf evs s = some o → stopOf evs s = none → finished evs s = true →
    (∀ b, budgetOf evs o.stream = some b → failures evs o.stream < b) → nextOf evs o = some j →
    o.parts[j]? ≠ some part → ∃ why evs', admit evs s part = (.error why, evs') ∧ stopOf evs' s = o.parts[j]?

/-- **A part asked out of its turn is the recursive session's stop where it stands.** -/
def OutOfTurnStops : Prop := ∀ (evs : Events) (s part : String), OutOfTurn evs s part

/-- `Decision.after a now prove ready` let `a'`'s proofs leave: `a'` is `a`, the prover and the send check returned
`ok`, and where `a` is scheduled, `now` is at or before its release time. -/
def SentOnTime (a : Admitted) (now : Nat) (prove ready : Run) (a' : Admitted) : Prop :=
  a' = a ∧ prove = .ok ∧ ready = .ok ∧ ∀ t, a.releaseMs = some t → now ≤ t

/-- **The proofs leave only on time.** -/
def AfterOnTime : Prop :=
  ∀ (a : Admitted) (now : Nat) (prove ready : Run) (a' : Admitted),
    Decision.after a now prove ready = .go a' → SentOnTime a now prove ready a'

/-! ## The instances the honest and nontrivial lemmas read

A recursive session as `openSession` opens the first one, with one statement `p` on stream `st`, no failure budget and no
schedule, whose inner session finished, and the outer session of `p` that `admit` admits on that record; and an admitted
session scheduled for 10 ms. -/

/-- The recursive session `r0000`'s open: one statement, `p`, on stream `st`, with no failure budget and no schedule. -/
def oneStatement : Open := { session := "r0000", stream := "st", parts := #["p"], budget := none, release := none }

/-- The record once `r0000` opened and its inner session finished. -/
def finishedRecord : Events := #[.opened oneStatement, .finish "r0000"]

/-- `r0000`'s statement `p` admitted: on stream `st`, its first statement, with no release time. -/
def firstAdmitted : Admitted := { stream := "st", index := 0, releaseMs := none }

/-- An admitted session whose proofs are released at 10 ms. -/
def scheduledAdmitted : Admitted := { stream := "st", index := 0, releaseMs := some 10 }

end Properties.Firewall
