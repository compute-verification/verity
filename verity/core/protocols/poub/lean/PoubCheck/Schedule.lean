/-!
# The schedule check: each pass's plan is a fixed function of the published schedule

`schedule.py`'s check, of record here (`ScheduleDeterministic` in `Security/Definitions/Accounting/Schedule.lean` is
what it is for). The prover publishes the session's schedule: when each block it holds beside the registered store is
written and freed, which registered blocks it frees and when, and the step of each pass. The check holds each pass's
plan to the schedule's plan at the pass's step (`Schedule.plan`, the blocks resident then, in store order) and the
session's appends to the schedule's (`Schedule.appendRuns`), refuses a schedule over the cap on its bits the prover
committed before the session's first coin (`audit.Params.schedule_cap`), and returns the schedule's size in bits
(`Schedule.bits`). So a credited pass reads the layout the verifier's own code computes from the schedule: the prover
steers it only through the schedule, whose bits are at most the cap whatever the coins, and a plan that follows
anything else (a coin, an encoding) is refused. `Properties.PoubCheck.ScheduleChecked` states it.

* A block of an append is `none`, never resident (withdrawn, or its encode never landed), or `(w, f)`: written at step
  `w`, resident from step `w + 1` through step `f` (`f > w`), or to the session's end if `f` is `none`
  (memory-map/v1: a step's record takes effect after its reads).
* A registered run `(first, count, f)` is freed at step `f`: resident through `f`. Every other registered block is
  resident throughout. The runs are in store order and disjoint.
* `steps` are the passes' steps, strictly increasing, below 2^64.

The schedule's encoding (`encode`) is LEB128 integers, prefix-free: the registered runs, each append's blocks (a block a
tag, 0 never resident, 1 never freed, else `f − w + 1`, then `w`) and the steps (the first, then each one's rise), each
list after its length. Fewer than 2^(a+1) schedules encode in at most a bits, so the cap `a`, fixed before the coins,
bounds the layouts a prover can steer to over every coin, where the bits of the one schedule it publishes after them
would not: its schedule's length can follow the coins. Code: Lean core only.
-/

namespace PoubCheck.Schedule

/-- Steps are below 2^64. -/
def STEP_LIMIT : Nat := 2 ^ 64

/-- An appended block: `none` never resident, or its write's step and its free's, if it is freed. -/
abbrev Block := Option (Nat × Option Nat)

structure Schedule where
  registered : Array (Nat × Nat × Nat)
  appends : Array (Array Block)
  steps : Array Nat
deriving Inhabited

/-- Whether block `b` is resident at step `s`: written before `s`, and not freed before it. -/
def residentAt (b : Block) (s : Nat) : Bool :=
  match b with
  | none => false
  | some (w, none) => decide (w < s)
  | some (w, some f) => decide (w < s) && decide (s ≤ f)

/-- The half-open spans of the blocks resident at step `s` in a store of `blocks` registered blocks, in store order:
the registered ones not freed before `s`, then each appended one at its index, `blocks` on, in append order. -/
def spans (sc : Schedule) (blocks s : Nat) : Array (Nat × Nat) := Id.run do
  let mut out : Array (Nat × Nat) := #[]
  let mut cur := 0
  for (f, c, e) in sc.registered do
    if e < s then
      if cur < f then out := out.push (cur, f)
      cur := f + c
  if cur < blocks then out := out.push (cur, blocks)
  let mut j := blocks
  for a in sc.appends do
    for b in a do
      if residentAt b s then out := out.push (j, j + 1)
      j := j + 1
  return out

/-- Spans in store order as a plan's runs `(first, count)`, a span that starts where the one before it ends joined to
it. -/
def runsOf (spans : Array (Nat × Nat)) : Array (Nat × Nat) :=
  (spans.foldl (fun (out : Array (Nat × Nat)) (x : Nat × Nat) =>
    match out.back? with
    | some (a, b) => if b == x.1 then out.pop.push (a, x.2) else out.push x
    | none => out.push x) #[]).map fun (a, b) => (a, b - a)

/-- The schedule's plan at step `s`: the blocks resident then, as runs in store order. -/
def plan (sc : Schedule) (blocks s : Nat) : Array (Nat × Nat) := runsOf (spans sc blocks s)

/-- The schedule's appends as the verifier numbers them: `(first, count)`, the first at `blocks`, each the next fresh
index. -/
def appendRuns (sc : Schedule) (blocks : Nat) : Array (Nat × Nat) :=
  (sc.appends.foldl (fun (acc : Array (Nat × Nat) × Nat) a => (acc.1.push (acc.2, a.size), acc.2 + a.size))
    (#[], blocks)).1

/-- LEB128's bytes of `n`: seven bits a byte, low first, the high bit set on every byte but the last. -/
def varint (n : Nat) : List UInt8 :=
  if n < 128 then [n.toUInt8] else (n % 128 + 128).toUInt8 :: varint (n / 128)
termination_by n
decreasing_by omega

/-- A block's integers: its tag (0 never resident, 1 never freed, else `f − w + 1`), then its write's step. -/
def blockCode : Block → List Nat
  | none => [0]
  | some (w, none) => [1, w]
  | some (w, some f) => [f - w + 1, w]

/-- The first, then each one's rise over the one before. -/
def rises : List Nat → Nat → List Nat
  | [], _ => []
  | x :: rest, prev => (x - prev) :: rises rest x

/-- The schedule's integers, each list after its length. -/
def numbers (sc : Schedule) : List Nat :=
  [sc.registered.size] ++ sc.registered.toList.flatMap (fun (f, c, e) => [f, c, e]) ++
    [sc.appends.size] ++ sc.appends.toList.flatMap (fun a => a.size :: a.toList.flatMap blockCode) ++
    [sc.steps.size] ++ rises sc.steps.toList 0

/-- The schedule's encoding. -/
def encode (sc : Schedule) : List UInt8 := (numbers sc).flatMap varint

/-- The schedule's size in bits. -/
def bits (sc : Schedule) : Nat := 8 * (encode sc).length

/-- Why a schedule is not one, in a store of `blocks` registered blocks: the first rule it breaks. -/
def malformed (sc : Schedule) (blocks : Nat) : Option String :=
  let rs := sc.registered
  if (List.range rs.size).any fun k =>
      let (f, c, e) := rs[k]!
      c = 0 || f + c > blocks || e ≥ STEP_LIMIT || (k > 0 && rs[k - 1]!.1 + rs[k - 1]!.2.1 > f) then
    some "the registered frees are runs of the registered store, in store order and disjoint, each at a step below 2^64"
  else
    let bad := (List.range sc.appends.size).findSome? fun i =>
      let a := sc.appends[i]!
      if a.size = 0 then some s!"append {i} has no block"
      else (List.range a.size).findSome? fun k =>
        match a[k]! with
        | none => none
        | some (w, none) => if w ≥ STEP_LIMIT then some s!"append {i}'s block {k} is written at a step past 2^64" else none
        | some (w, some f) =>
          if f ≥ STEP_LIMIT then some s!"append {i}'s block {k} is freed at a step past 2^64"
          else if f ≤ w then some s!"append {i}'s block {k} is freed at step {f}, not after its write at step {w}"
          else none
    match bad with
    | some why => some why
    | none =>
      if (List.range sc.steps.size).any fun k => sc.steps[k]! ≥ STEP_LIMIT || (k > 0 && sc.steps[k - 1]! ≥ sc.steps[k]!) then
        some "the passes' steps strictly increase below 2^64"
      else none

/-- The first of the session's appends `(first, count)` that is not the schedule's, as the reason. -/
def appendsDiffer (sc : Schedule) (blocks : Nat) (appends : Array (Nat × Nat)) : Option String :=
  let want := appendRuns sc blocks
  if appends.size ≠ want.size then some s!"the session made {appends.size} appends, the schedule {want.size}"
  else (List.range appends.size).findSome? fun i =>
    if appends[i]! = want[i]! then none
    else some s!"append {i} is ({appends[i]!.1}, {appends[i]!.2}), the schedule's ({want[i]!.1}, {want[i]!.2})"

/-- The first pass `(step, runs)` whose step is not the schedule's or whose plan is not the schedule's at its step, as
the reason. -/
def passesDiffer (sc : Schedule) (blocks : Nat) (passes : Array (Nat × Array (Nat × Nat))) : Option String :=
  if passes.size ≠ sc.steps.size then some s!"the session has {passes.size} passes, the schedule {sc.steps.size}"
  else (List.range passes.size).findSome? fun k =>
    if passes[k]!.1 ≠ sc.steps[k]! then some s!"pass {k} is at step {passes[k]!.1}, the schedule's at {sc.steps[k]!}"
    else if passes[k]!.2 ≠ plan sc blocks passes[k]!.1 then
      some s!"pass {k}'s plan is not the schedule's at step {passes[k]!.1}"
    else none

/-- The schedule check of a session over `blocks` registered blocks, at the cap `cap` on its schedule's bits that the
prover committed with the session's parameters before its first coin: its passes `(step, runs)` in order, each plan's
runs as the verifier received them, and its appends `(first, count)` in order. The schedule's bits, at most `cap`,
when every pass's plan is the schedule's at its step and the appends are the schedule's; otherwise why not. -/
def check (sc : Schedule) (blocks cap : Nat) (passes : Array (Nat × Array (Nat × Nat))) (appends : Array (Nat × Nat)) :
    Except String Nat := do
  if let some why := malformed sc blocks then throw why
  if cap < bits sc then throw s!"the schedule is {bits sc} bits, over the cap of {cap} committed before the coins"
  if let some why := appendsDiffer sc blocks appends then throw why
  if let some why := passesDiffer sc blocks passes then throw why
  pure (bits sc)

end PoubCheck.Schedule
