module

public import Firewall.Release
public import Proofs.Flock.Recursive.StopLeak

/-!
# A recursive session's stops, counted from the ZK server's rules

`StopLeak`'s `prCoin_cut_le` bounds what a view released at a stop leaks by the number of stops, and `RecStops M` counts
a recursive session's, `1 + Σ_p (M p + 1)`. This file states that the ZK server's rules (`Firewall.Record`,
`Firewall.Release`, run as `Firewall.Loop` and `rec_live` run them) release a recursive session's view cut at one of
`RecStops M`: the stop `recStop` that the run reached, read off what reached the auditor.

**The run** (`StopCount.run`) is a list of steps, each one op on the record's events by the code's own function, in any
order: each op holds `record.lock` while it reads the record and appends to it, and one outer session at a time holds
`slot.lock` (`holder`). `rec_live` opens a recursive session (`openSession`, its statements and budget checked by
`partsOk` and `budgetArg`), runs its inner session, statement `0`, and records its `finish` once its `M s 0` messages
reached the auditor, or its stop at `inner`. An outer session of `s` at `part` that finds the slot taken records `wait`; one that takes it records its slot
and is admitted or refused (`seat`: `slot`, then `admit`). Nothing of a refused session reaches the auditor; an admitted
one's messages do, one `relay` at a time, at most statement `j`'s `M s (j + 1)`, before its `Finish` and, once
`Decision.after` lets its proofs go, after it, from its release time on (`left`). The auditor's verdict comes after its
last message, and `settle`'s rule makes it the release or the stop; a fault while sending is the stop. A session cut
before its `Finish` was decided is its stop (`stopAt`); one that ends with nothing recorded (killed, or an op that could
not read the record) leaves the slot, and the next `slot` records its stop.

The model reads the record as its events: every op reads back the events the ZK server wrote (`Record.read` refuses any
line that is not `Event.line` of the event it parses to).

* `RecStopsOfRun`: at every run and every recursive session `s`, what reached the auditor of each statement of `s` is
  `cutAt (recStop …)`'s, nothing of any other statement, and each statement's from the one outer session of it
  admitted, if any (`adm`: the slot records a holder that ended without its outcome as its stop, so no statement is
  admitted twice); `s` runs through (`recStop … = none`) unless a statement of it was not released (`unreleasedOf`);
  and each statement's proofs left at its release time where `s` is scheduled.
* `recCut`, `recReleased`: the whole view, one part a statement, cut at a stop by a cut of one statement's part
  (`cutP`, at how many of its messages are kept), and the view the run released, each statement's part cut at what
  reached the auditor of it. `recReleased … = recCut … (recStop …) …` is `RecStopsOfRun`'s first conjunct read at the
  view.
* `recStopAt`, `sent_cutAt_at`, `recReleased_eq_at`: the stop read at a fixed number of statements `n`, so that it ranges
  over one `RecStops` at every run, as `prCoin_cut_le` counts it, and the view cut at it where `s` has `n` statements.
-/

@[expose] public section

namespace Flock.StopCount

open Core.Firewall

/-- The outer slot's holder: an outer session of `session` at `part`, admitted as `admitted`, `going` once
`Decision.after` let its proofs leave. -/
structure Holder where
  session : String
  part : String
  admitted : Admitted
  going : Bool

/-- The ZK server's state in a run: the record's events, the outer slot's holder, how many of statement `p`'s messages
of recursive session `s` reached the auditor (`sent s p`, statement `0` the inner session, `j + 1` V*'s statement `j`),
how many outer sessions of it were admitted (`adm s p`), and when its proofs left (`left s p`). -/
structure St where
  evs : Events
  holder : Option Holder
  sent : String → ℕ → ℕ
  adm : String → ℕ → ℕ
  left : String → ℕ → Option ℕ

def St.init : St := ⟨#[], none, fun _ _ => 0, fun _ _ => 0, fun _ _ => none⟩

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
    match partsOk parts, budgetArg budget with
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
      | (.ok a, evs) => { σ with evs, holder := some ⟨s, part, a, false⟩, adm := bump σ.adm s (a.index + 1) }
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

/-- How many statements the recursive session `s` has in `σ`, its inner session's included: none if never opened. -/
def nOf (σ : St) (s : String) : ℕ := ((openOf σ.evs s).map fun o => o.parts.size + 1).getD 0

/-- The release schedule `s` was opened with, if any. -/
def schedOf (σ : St) (s : String) : Option Core.Firewall.Schedule := (openOf σ.evs s).bind (·.release)

/-- Whether `s` was opened and a statement of it is not released. -/
def unreleasedOf (σ : St) (s : String) : Bool :=
  match openOf σ.evs s with
  | some o => (nextOf σ.evs o).isSome
  | none => false

/-- **Where `s` stopped in `σ`**, read off what reached the auditor: its first statement `q` of which fewer than its
`M s q` messages did, after the `k` that did; none if every message of every statement did. -/
def recStop (M : String → ℕ → ℕ) (σ : St) (s : String) :
    Flock.SecurityProofs.RecStops (fun p : Fin (nOf σ s) => M s p) :=
  match (List.finRange (nOf σ s)).find? (fun q : Fin (nOf σ s) => decide (σ.sent s q < M s q)) with
  | none => none
  | some q => some ⟨q, ⟨min (σ.sent s q) (M s q), Nat.lt_succ_of_le (Nat.min_le_right _ _)⟩⟩

/-- **The whole view cut at the stop `r`**: statement `p`'s part `v p` cut by `cutP p`, at how many of its messages are
kept, `cutAt r p`. -/
def recCut {n : ℕ} (M : Fin n → ℕ) {V : Fin n → Type} (cutP : (p : Fin n) → ℕ → V p → V p)
    (r : Flock.SecurityProofs.RecStops M) (v : (p : Fin n) → V p) : (p : Fin n) → V p :=
  fun p => cutP p (cutAt M r p) (v p)

/-- **The view the run `σ` released of `s`**: statement `p`'s part of the whole view `v` cut by `cutP p` at how many of
its messages reached the auditor. -/
def recReleased (σ : St) (s : String) {V : Fin (nOf σ s) → Type} (cutP : (p : Fin (nOf σ s)) → ℕ → V p → V p)
    (v : (p : Fin (nOf σ s)) → V p) : (p : Fin (nOf σ s)) → V p :=
  fun p => cutP p (σ.sent s p) (v p)

end Flock.StopCount

namespace Flock.Guarantees

/-- **A recursive session's view is cut at its stop.** At every run (`StopCount.run`: the ZK server's ops in any order,
at each statement's message count `M s p`) and every recursive session `s`: what reached the auditor of each statement
of `s` is what `cutAt` keeps at the stop the run reached (`recStop`), nothing of any statement past `s`'s, and each
statement's from one outer session, admitted at most once; `s` runs through unless a statement of it was not released;
and where `s` is scheduled, each statement `j` whose proofs left did so at its release time (`releaseMs`).

It is about `StopCount.run`, a model of the loop's ops, not about `Firewall.Loop` or `rec_live` themselves. Its ops are
the record's own functions, and four of its guards are the model's: only the outer slot's holder relays, and at most
`M s p` messages of each statement (`relay`); the inner session sends at most `M s 0` messages and finishes only after
its last (`innerSend`, `innerFinish`); the auditor's verdict releases a statement only after its last message
(`verdict`); and every op reads back the events the record holds as written (`St.evs`). That `Firewall.Loop` and
`rec_live` keep these guards is open, among it that nothing of an outer session reaches the auditor after its stop. -/
def RecStopsOfRun : Prop :=
  ∀ (M : String → ℕ → ℕ) (steps : List StopCount.Step) (s : String),
    (∀ p, (StopCount.run M steps).sent s p =
      StopCount.cutAt _ (StopCount.recStop M (StopCount.run M steps) s) p) ∧
    (∀ p, (StopCount.run M steps).adm s p ≤ 1) ∧
    (StopCount.recStop M (StopCount.run M steps) s = none ∨ StopCount.unreleasedOf (StopCount.run M steps) s = true) ∧
    ∀ j, StopCount.schedOf (StopCount.run M steps) s = none ∨ (StopCount.run M steps).left s (j + 1) = none ∨
      (StopCount.run M steps).left s (j + 1) = Core.Firewall.releaseMs (StopCount.schedOf (StopCount.run M steps) s) j

end Flock.Guarantees

namespace Flock.StopCount

/-- **The released view is the whole view cut at the run's stop**, at every cut of one statement's part. -/
theorem recReleased_eq (h : Flock.Guarantees.RecStopsOfRun) (M : String → ℕ → ℕ) (steps : List Step) (s : String)
    {V : Fin (nOf (run M steps) s) → Type} (cutP : (p : Fin (nOf (run M steps) s)) → ℕ → V p → V p)
    (v : (p : Fin (nOf (run M steps) s)) → V p) :
    recReleased (run M steps) s cutP v = recCut (fun p => M s p) cutP (recStop M (run M steps) s) v :=
  funext fun p => by
    unfold recReleased recCut
    rw [(h M steps s).1 p]

/-- **The stop at `n` statements**: `recStop` read at a fixed `n`, so that its type is one at every run (a stop that
`StopLeak`'s `prCoin_cut_le` counts ranges over one `RecStops`): where `s` has `n` statements, `recStop`; else none. -/
def recStopAt (n : ℕ) (M : String → ℕ → ℕ) (σ : St) (s : String) : Flock.SecurityProofs.RecStops (fun p : Fin n => M s p) :=
  if h : nOf σ s = n then h ▸ recStop M σ s else none

/-- Where `s` has `n` statements, what reached the auditor of each is `cutAt`'s at `recStopAt n`. -/
theorem sent_cutAt_at (h : Flock.Guarantees.RecStopsOfRun) (M : String → ℕ → ℕ) (steps : List Step) (s : String)
    {n : ℕ} (hn : nOf (run M steps) s = n) (p : ℕ) :
    (run M steps).sent s p = cutAt (fun p : Fin n => M s p) (recStopAt n M (run M steps) s) p := by
  subst hn
  unfold recStopAt
  split
  · exact (h M steps s).1 p
  · rename_i hne
    exact absurd rfl hne

/-- **The released view is the whole view cut at `recStopAt n`**, where `s` has `n` statements: the form
`prCoin_cut_le` reads, its stop in `RecStops (fun p : Fin n => M s p)` at every run. -/
theorem recReleased_eq_at (h : Flock.Guarantees.RecStopsOfRun) (M : String → ℕ → ℕ) (steps : List Step) (s : String)
    {n : ℕ} (hn : nOf (run M steps) s = n) {V : Fin n → Type} (cutP : (p : Fin n) → ℕ → V p → V p)
    (v : (p : Fin n) → V p) :
    (fun p => cutP p ((run M steps).sent s p) (v p)) = recCut (fun p => M s p) cutP (recStopAt n M (run M steps) s) v :=
  funext fun p => by
    unfold recCut
    rw [sent_cutAt_at h M steps s hn p]

end Flock.StopCount
