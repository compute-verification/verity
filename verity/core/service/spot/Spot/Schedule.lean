/-
**The step schedule** (the Python's `schedule`): the replay blocks of a two-stage window as runs of positions of its
requests, which the boundary registration's record commits before the first coin (`window.schedule`), each run
`{"request", "first_span", "spans", "first_position", "length"}`: replay blocks `first_span …` up to `first_span + spans`
are each `length` positions of request `request`, the first from position `first_position`. The schedule is public, as
the replay blocks' boundaries are, and it is the prover's choice, fixed before the first coin.

* `ofJson?`: the runs, each an object of exactly those keys, a string and four naturals;
* `why`: why runs are no schedule of `population` replay blocks, checked at the size of the runs: a run of no block or of
  no position, a run that doesn't start at the block after the runs before it (so the runs cover blocks `0 …` in order),
  a run that doesn't start at its request's position after that request's runs before it (so each request's runs cover
  its positions `0 …` in order), or runs that don't end at the population;
* `at?`: replay block `s`'s request and its positions `[a, a + length)`, `a = first_position + (s − first_span)·length`,
  from the run that holds it;
* `failure`: why a record's schedule is refused, none for a record that commits none.
-/
module

public import Spot.Json
meta import Spot.Json

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Schedule

open Lean (Json)

/-- A run of replay blocks: `spans` blocks from `firstSpan`, each `length` positions of request `request`, the first
from `firstPosition`. -/
structure Run where
  request : String
  firstSpan : Nat
  spans : Nat
  firstPosition : Nat
  length : Nat
  deriving Repr, BEq, Inhabited

/-- A run's keys, sorted. -/
def KEYS : Array String := #["first_position", "first_span", "length", "request", "spans"]

/-- A run from its JSON: exactly `KEYS`, the request a string and the rest naturals. -/
def Run.ofJson? (j : Json) : Option Run :=
  if keys? j != some KEYS then none else
  match getD j "request", nat? (getD j "first_span"), nat? (getD j "spans"), nat? (getD j "first_position"),
      nat? (getD j "length") with
  | .str r, some f, some n, some p, some l => some ⟨r, f, n, p, l⟩
  | _, _, _, _, _ => none

/-- The runs of a schedule's JSON, a list of runs. -/
def ofJson? : Json → Option (Array Run)
  | .arr rs => rs.mapM Run.ofJson?
  | _ => none

/-- The position after request `r`'s runs among `next`, each request's next position: `0` before its first run. -/
def nextOf (next : Array (String × Nat)) (r : String) : Nat := ((next.find? (·.1 == r)).map (·.2)).getD 0

/-- `next` with request `r`'s next position `p`. -/
def setNext (next : Array (String × Nat)) (r : String) (p : Nat) : Array (String × Nat) :=
  if next.any (·.1 == r) then next.map fun x => if x.1 == r then (r, p) else x else next.push (r, p)

/-- One run of `why`'s pass: the refusal so far, the block after the runs before, each request's next position. -/
def step : Option String × Nat × Array (String × Nat) → Run → Option String × Nat × Array (String × Nat)
  | (some w, b, next), _ => (some w, b, next)
  | (none, b, next), r =>
    if r.spans == 0 || r.length == 0 then (some s!"a run from block {r.firstSpan} of no block or no position", b, next)
    else if r.firstSpan != b then (some s!"a run from block {r.firstSpan}, not from block {b}", b, next)
    else if r.firstPosition != nextOf next r.request then
      let why := s!"a run of {r.request} from position {r.firstPosition}, not from position {nextOf next r.request}"
      (some why, b, next)
    else (none, b + r.spans, setNext next r.request (r.firstPosition + r.spans * r.length))

/-- Why runs `rs` are no schedule of `population` replay blocks (module docstring); none if they are one. -/
def why (population : Nat) (rs : Array Run) : Option String :=
  match rs.foldl step (none, 0, #[]) with
  | (some w, _, _) => some w
  | (none, b, _) => if b != population then some s!"runs of {b} blocks, not the population's {population}" else none

/-- Replay block `s`'s request and its positions `[a, b)` under runs `rs`; none for a block no run holds. -/
def at? (rs : Array Run) (s : Nat) : Option (String × Nat × Nat) := do
  let r ← rs.find? fun r => r.firstSpan ≤ s && s < r.firstSpan + r.spans
  let a := r.firstPosition + (s - r.firstSpan) * r.length
  return (r.request, a, a + r.length)

/-- Why record `rec`'s schedule (`window.schedule`) is refused: one that isn't a list of runs, a population that isn't a
natural, or `why`'s refusal; none for a record that commits no schedule. -/
def failure (rec : Json) : Option String :=
  match getD (getD rec "window") "schedule" with
  | .null => none
  | j => match ofJson? j, nat? (getD rec "population") with
    | some rs, some m => why m rs
    | none, _ => some "the schedule is not a list of runs"
    | some _, none => some "the record's population is not a natural number"

private def exRun (r : String) (f n p l : Nat) : Json :=
  Json.mkObj [("request", .str r), ("first_span", Lean.toJson f), ("spans", Lean.toJson n),
    ("first_position", Lean.toJson p), ("length", Lean.toJson l)]

private def exRec (population : Nat) (schedule : Json) : Json :=
  Json.mkObj [("population", Lean.toJson population), ("window", Json.mkObj [("schedule", schedule)])]

private def exTwo : Json := .arr #[exRun "a" 0 3 0 1, exRun "b" 3 2 0 4, exRun "a" 5 1 3 2]

#guard failure (Json.mkObj [("population", Lean.toJson (4 : Nat))]) == none
#guard failure (exRec 6 exTwo) == none
#guard failure (exRec 7 exTwo) == some "runs of 6 blocks, not the population's 7"
#guard failure (exRec 6 (.arr #[exRun "a" 0 3 0 1, exRun "b" 4 2 0 4, exRun "a" 6 1 3 2])) ==
  some "a run from block 4, not from block 3"
#guard failure (exRec 6 (.arr #[exRun "a" 0 3 0 1, exRun "b" 3 2 0 4, exRun "a" 5 1 4 2])) ==
  some "a run of a from position 4, not from position 3"
#guard failure (exRec 6 (.arr #[exRun "a" 0 3 1 1, exRun "b" 3 3 0 4])) ==
  some "a run of a from position 1, not from position 0"
#guard failure (exRec 3 (.arr #[exRun "a" 0 3 0 0])) == some "a run from block 0 of no block or no position"
#guard failure (exRec 3 (.arr #[(exRun "a" 0 3 0 1).setObjVal! "extra" .null])) ==
  some "the schedule is not a list of runs"
#guard failure (exRec 3 (Json.mkObj [])) == some "the schedule is not a list of runs"
#guard failure ((exRec 3 (.arr #[exRun "a" 0 3 0 1])).setObjVal! "population" (.str "3")) ==
  some "the record's population is not a natural number"
#guard (ofJson? exTwo).bind (at? · 4) == some ("b", 4, 8)
#guard (ofJson? exTwo).bind (at? · 5) == some ("a", 3, 5)
#guard (ofJson? exTwo).bind (at? · 2) == some ("a", 2, 3)
#guard (ofJson? exTwo).bind (at? · 6) == none

end Spot.Schedule
