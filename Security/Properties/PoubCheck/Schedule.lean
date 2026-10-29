import Definitions.Accounting.Schedule
import PoubCheck.Schedule

/-!
# The schedule check fixes each credited pass's plan (`PoubCheck.Schedule`)

`PoubCheck.Schedule.check` (`verity/core/protocols/poub/lean/PoubCheck/Schedule.lean`, which
`schedule.py` is held to by the difftest, and without whose acceptance `audit.ledger` credits nothing after the seed)
holds a session's passes and appends to the schedule the prover publishes, at the cap on the schedule's bits the
prover committed before the session's first coin, and returns the schedule's bits.

* `ScheduleChecked`: whenever the check accepts, the schedule is well formed, the passes are at its steps, each pass's
  plan is the schedule's plan at the pass's step (`Schedule.plan`, computed by the verifier), the appends are the
  schedule's (`Schedule.appendRuns`), and the bits returned are its encoding's (`Schedule.bits`), at most the cap.
* `ScheduleFixesLayout`: so the schedule fixes them: two sessions the check accepts on one schedule have the same
  passes and the same appends. A layout that follows a coin at a schedule the verifier counted is refused.

They take no premise. The schedule's bits (`scheduleBits`) are one-to-one on well-formed schedules
(`ScheduleCode.scheduleBits_injective` in `Proofs.Properties.PoubCheck.Schedule`: LEB128 is prefix-free, and each
list comes after its length). What is left between the code and the model's layout (`Accounting.ScheduleDeterministic`,
`Accounting.Assumptions.SteeredLayout`) is one `Prop`, `ScheduleLayoutGap`: that the model's layout of a pass is the
verifier's position map of the session's appends and the plan the pass sent. Under it the check's acceptance at the
cap `a` at every coin gives `ScheduleDeterministic` at `a` and `SteeredLayout` at `a + 1`
(`scheduleDeterministic_of_checked`, `steered_of_checked`). The cap is fixed before the coins, so it is one `a` at
every coin, where the bits of the schedule published after them can follow the coins.
-/

open PoubCheck

namespace Properties.PoubCheck

/-- Whenever the schedule check accepts schedule `sc` over `blocks` registered blocks at cap `cap` for passes
`(step, runs)` and appends `(first, count)`, returning `b`: `sc` is well formed, the passes are at its steps in order,
each pass's runs are the schedule's plan at its step, the appends are the schedule's, and `b` is the schedule's bits,
at most `cap`. -/
def ScheduleChecked : Prop :=
  ∀ (sc : Schedule.Schedule) (blocks cap : Nat) (passes : Array (Nat × Array (Nat × Nat)))
    (appends : Array (Nat × Nat)) (b : Nat),
    Schedule.check sc blocks cap passes appends = .ok b →
      Schedule.malformed sc blocks = none ∧ passes.map Prod.fst = sc.steps ∧
        (∀ p ∈ passes, p.2 = Schedule.plan sc blocks p.1) ∧ appends = Schedule.appendRuns sc blocks ∧
        b = Schedule.bits sc ∧ b ≤ cap

/-- Whenever the schedule check accepts two sessions on one schedule, at any caps, they have the same passes, plans
included, and the same appends. -/
def ScheduleFixesLayout : Prop :=
  ∀ (sc : Schedule.Schedule) (blocks cap cap' : Nat) (passes passes' : Array (Nat × Array (Nat × Nat)))
    (appends appends' : Array (Nat × Nat)) (b b' : Nat),
    Schedule.check sc blocks cap passes appends = .ok b → Schedule.check sc blocks cap' passes' appends' = .ok b' →
      passes = passes' ∧ appends = appends'

/-- The schedule's encoding as bits, each byte low bit first: `Schedule.bits sc` of them (`scheduleBits_length`). -/
def scheduleBits (sc : Schedule.Schedule) : List Bool :=
  (Schedule.encode sc).flatMap fun b => List.ofFn fun i : Fin 8 => b.toNat.testBit i

/-- **What lies between the schedule check and the model's layout.** At each prefix `c` of the session's coins and
each codec and append coins `y`, the model's layout of the pass is `pos c` of the session's appends `appended c y` and
the runs `sent c y` the pass's plan sent, `pos c` being the verifier's map from the store's block indices to the
model's positions. -/
def ScheduleLayoutGap {Pfx Y : Type} {N T M : ℕ} (pos : Pfx → Array (ℕ × ℕ) → Array (ℕ × ℕ) → Fin N ↪ Fin T × Fin M)
    (appended sent : Pfx → Y → Array (ℕ × ℕ)) (layout : Pfx → Y → Fin N ↪ Fin T × Fin M) : Prop :=
  ∀ c y, layout c y = pos c (appended c y) (sent c y)

end Properties.PoubCheck
