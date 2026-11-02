import Mathlib.Algebra.BigOperators.Fin
import Firewall.Release
import Proofs.Flock.Recursive.StopLeak
import Proofs.Flock.Claims.ZK.Server

/-!
# A recursive session's stops, counted from the firewall's rules

`StopLeak`'s `prCoin_cut_le` bounds what a view released at a stop leaks by the number of stops, and `RecStops M` counts
a recursive session's, `1 + Σ_p (M p + 1)`. This file states that the firewall's rules (`Firewall.Record`,
`Firewall.Release`, run as `Firewall.Loop` and `rec_live` run them) release a recursive session's view cut at one of
`RecStops M`, and a stream's views at one of `Claims.ZKServer.Pattern`'s, which `stream_leak` counts.

**The run** (`StopCount.run`) is a list of steps, each one op on the record's events by the code's own function, in any
order: each op holds `record.lock` while it reads the record and appends to it, and one outer session at a time holds
`slot.lock` (`holder`). `rec_live` opens a recursive session (`openSession`, its statements checked by `partsOk`), runs
its inner session, statement `0`, and records its `finish` once its `M s 0` messages reached the auditor, or its stop at
`inner`. An outer session of `s` at `part` that finds the slot taken records `wait`; one that takes it records its slot
and is admitted or refused (`seat`: `slot`, then `admit`). Nothing of a refused session reaches the auditor; an admitted
one's messages do, one `relay` at a time, at most statement `j`'s `M s (j + 1)`, before its `Finish` and, once
`Decision.after` lets its proofs go, after it, from its release time on (`left`). The auditor's verdict comes after its
last message, and `settle`'s rule makes it the release or the stop; a fault while sending is the stop. A session cut
before its `Finish` was decided is its stop (`stopAt`); one that ends with nothing recorded (killed, or an op that could
not read the record) leaves the slot, and the next `slot` records its stop.

The model reads the record as its events: every op reads back the events the firewall wrote (`Record.read` refuses any
line that is not `Event.line` of the event it parses to).

* `RecStopsOfRun`: at every run and every recursive session it opened, what reached the auditor of each statement is
  `cutAt r` for one `r : RecStops M`, nothing of any other statement, and each statement's proofs left at its release
  time where the session is scheduled.
* `StreamStopsOfRun`: at every run, `n` recursive sessions with the same statements' message counts `m` reached the
  auditor as one `Pattern n (Σ_p (m p + 1)) B` decodes them, `B` the sessions with a statement not released
  (`unreleased`).
-/

namespace Flock.StopCount

open Core.Firewall

/-- The outer slot's holder: an outer session of `session` at `part`, admitted as `admitted`, `going` once
`Decision.after` let its proofs leave. -/
structure Holder where
  session : String
  part : String
  admitted : Admitted
  going : Bool

/-- The firewall's state in a run: the record's events, the outer slot's holder, how many of statement `p`'s messages
of recursive session `s` reached the auditor (`sent s p`, statement `0` the inner session, `j + 1` V*'s statement `j`),
and when statement `p`'s proofs left (`left s p`). -/
structure St where
  evs : Events
  holder : Option Holder
  sent : String → ℕ → ℕ
  left : String → ℕ → Option ℕ

def St.init : St := ⟨#[], none, fun _ _ => 0, fun _ _ => none⟩

/-- `f` with one more message at `s`, `p`. -/
def bump (f : String → ℕ → ℕ) (s : String) (p : ℕ) : String → ℕ → ℕ :=
  fun s' p' => if s' = s ∧ p' = p then f s p + 1 else f s' p'

/-- `f` with `a` at `s`, `p`. -/
def put {α : Type} (f : String → ℕ → α) (s : String) (p : ℕ) (a : α) : String → ℕ → α :=
  fun s' p' => if s' = s ∧ p' = p then a else f s' p'

/-- One op of a run. -/
inductive Step where
  /-- `rec_live` opens a recursive session on `stream` with its statements `parts` -/
  | openRec (stream : String) (parts : Array String) (budget : Option ℕ) (release : Option Core.Firewall.Schedule)
  /-- the next message of `s`'s inner session reaches the auditor -/
  | innerSend (s : String)
  /-- `s`'s inner session finished -/
  | innerFinish (s : String)
  /-- `s`'s inner session stopped -/
  | innerStop (s why : String)
  /-- an outer session of `s` at `part` finds the slot taken -/
  | wait (s part : String)
  /-- an outer session of `s` at `part` takes the free slot and is admitted or refused -/
  | seat (s part : String)
  /-- the holder's next message reaches the auditor -/
  | relay
  /-- the holder's `Finish` at `now` (ms), the prover and the send check having returned `prove` and `ready` -/
  | barrier (now : ℕ) (prove ready : Core.Firewall.Run)
  /-- the auditor's verdict on the holder's proofs, or the fault that cut them while they left -/
  | verdict (v : Except String Bool)
  /-- the holder ended before its `Finish` was decided -/
  | cut (why : String)
  /-- the holder ended with nothing recorded and left the slot -/
  | die

/-- The holder's statement: `j + 1` for V*'s statement `j`. -/
def Holder.stmt (h : Holder) : ℕ := h.admitted.index + 1

/-- One op on the state, by the code's rules, given each statement's message count `M s p`. A step its guard refuses
leaves the state as it was. -/
def step (M : String → ℕ → ℕ) (σ : St) : Step → St
  | .openRec stream parts budget release =>
    match partsOk parts, budgetOk budget with
    | .ok (), .ok () => { σ with evs := (openSession σ.evs stream parts budget release).2 }
    | _, _ => σ
  | .innerSend s =>
    if (openOf σ.evs s).isSome && !finished σ.evs s && decide (σ.sent s 0 < M s 0) then
      { σ with sent := bump σ.sent s 0 }
    else σ
  | .innerFinish s =>
    if (openOf σ.evs s).isSome && !finished σ.evs s && decide (σ.sent s 0 = M s 0) then
      { σ with evs := finish σ.evs s }
    else σ
  | .innerStop s why =>
    if (openOf σ.evs s).isSome && !finished σ.evs s then { σ with evs := (stop σ.evs s INNER why).2 } else σ
  | .wait s part => { σ with evs := σ.evs.push (.wait s part) }
  | .seat s part =>
    match σ.holder with
    | some _ => σ
    | none =>
      match admit (slot σ.evs s part) s part with
      | (.ok a, evs) => { σ with evs, holder := some ⟨s, part, a, false⟩ }
      | (.error _, evs) => { σ with evs }
  | .relay =>
    match σ.holder with
    | some h =>
      if σ.sent h.session h.stmt < M h.session h.stmt then { σ with sent := bump σ.sent h.session h.stmt } else σ
    | none => σ
  | .barrier now prove ready =>
    match σ.holder with
    | some h =>
      if h.going then σ
      else
        match Decision.after h.admitted now prove ready with
        | .go a =>
          { σ with holder := some { h with going := true },
                   left := put σ.left h.session h.stmt (some (a.releaseMs.getD now)) }
        | .stop why | .refused why => { σ with evs := (stop σ.evs h.session h.part why).2, holder := none }
    | none => σ
  | .verdict v =>
    match σ.holder with
    | some h =>
      let after := match v with
        | .ok _ => decide (σ.sent h.session h.stmt = M h.session h.stmt)
        | .error _ => true
      if h.going && after then
        let evs := match v with
          | .ok true => release σ.evs h.session h.part
          | .ok false => (stop σ.evs h.session h.part "the challenger's verdict rejects it").2
          | .error why => (stop σ.evs h.session h.part why).2
        { σ with evs, holder := none }
      else σ
    | none => σ
  | .cut why =>
    match σ.holder with
    | some h => if h.going then σ else { σ with evs := (stop σ.evs h.session h.part why).2, holder := none }
    | none => σ
  | .die => { σ with holder := none }

/-- The state after `steps`, from the empty record. -/
def run (M : String → ℕ → ℕ) (steps : List Step) : St := steps.foldl (step M) St.init

/-- **What reached the auditor of statement `p` at the stop `r`**: at none, every message of every statement; at the
first statement to stop, `q`, after its `k`-th message, every message of each statement before `q`, `k` of `q`'s, and
none after; nothing of a statement past the session's `n`. -/
def cutAt {n : ℕ} (M : Fin n → ℕ) (r : Flock.SecurityProofs.RecStops M) (p : ℕ) : ℕ :=
  if h : p < n then
    match r with
    | none => M ⟨p, h⟩
    | some ⟨q, k⟩ => if p < q.val then M ⟨p, h⟩ else if p = q.val then k.val else 0
  else 0

/-- The sessions among `os` with a statement not released. -/
def unreleased {n : ℕ} (evs : Events) (os : Fin n → Open) : ℕ :=
  (Finset.univ.filter fun i => (nextOf evs (os i)).isSome).card

/-- Session `i`'s stop in the pattern `π`: none where it runs through, else its form, `RecStops m`'s stops numbered by
`finSigmaFinEquiv`. -/
def patternStop {n k B : ℕ} (m : Fin k → ℕ) (π : Claims.ZKServer.Pattern n (∑ p, (m p + 1)) B) (i : Fin n) :
    Flock.SecurityProofs.RecStops m :=
  (π.1 i).map finSigmaFinEquiv.symm

end Flock.StopCount

namespace Flock.Guarantees

/-- **A recursive session's view is cut at one of its stops.** At every run (`StopCount.run`: the firewall's ops in any
order, at each statement's message count `M s p`) and every recursive session `s` it opened as `o`, there is a stop
`r : RecStops`, at `o`'s statements' counts, such that what reached the auditor of each statement of `s` is `cutAt r`'s,
nothing of any other, and each statement `j` whose proofs left did so at its release time (`releaseMs`) where `o` is
scheduled. -/
def RecStopsOfRun : Prop :=
  ∀ (M : String → ℕ → ℕ) (steps : List StopCount.Step) (s : String) (o : Core.Firewall.Open),
    Core.Firewall.openOf (StopCount.run M steps).evs s = some o →
    ∃ r : Flock.SecurityProofs.RecStops (fun p : Fin (o.parts.size + 1) => M s p),
      (∀ p, (StopCount.run M steps).sent s p = StopCount.cutAt _ r p) ∧
      ∀ j t, (StopCount.run M steps).left s (j + 1) = some t → o.release.isSome →
        Core.Firewall.releaseMs o.release j = some t

/-- **A stream's views are cut at one stop pattern.** At every run and `n` recursive sessions `ss` it opened as `os`,
each with `k` statements, the inner session's included, of message counts `m`, there is a pattern of at most
`unreleased` stops, the sessions with a statement not released, at which what reached the auditor of each session is
`cutAt` its stop's. -/
def StreamStopsOfRun : Prop :=
  ∀ (M : String → ℕ → ℕ) (steps : List StopCount.Step) {n k : ℕ} (m : Fin k → ℕ) (ss : Fin n → String)
    (os : Fin n → Core.Firewall.Open),
    (∀ i, Core.Firewall.openOf (StopCount.run M steps).evs (ss i) = some (os i)) → (∀ i, (os i).parts.size + 1 = k) →
    (∀ i (p : Fin k), M (ss i) p = m p) →
    ∃ π : Claims.ZKServer.Pattern n (∑ p, (m p + 1)) (StopCount.unreleased (StopCount.run M steps).evs os),
      ∀ i p, (StopCount.run M steps).sent (ss i) p = StopCount.cutAt m (StopCount.patternStop m π i) p

end Flock.Guarantees
