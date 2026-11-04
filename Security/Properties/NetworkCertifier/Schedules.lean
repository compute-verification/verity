import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Set.Card
import CertifierDevice.Relay
import Definitions.Core.Assumptions
import Definitions.NetworkCertifier.PeriodConditions
import Definitions.NetworkCertifier.RecordRows

/-!
# The schedules the network certifier runs, and its record format holds them

The properties of the relay's schedule (`CertifierDevice.Relay`: `warden.py`'s refusal, admission, egress limit and
cut), stated against its code at the parameters the auditor reads (`NetworkCheck.Period.Schedule`), over a monitoring
period of `n` windows of `T` buckets on the arrivals `ready` of its connections (each the first tick that finds it
queued, in arrival order):

* `SchedulesHeldAt`: every schedule the relay can run there (`relaySchedules`: one for each `END`-or-`FAIL` index of
  each connection) gives each connection at most its cap (`Schedule.caps`) and fits the rows format at `S = lanes·r`
  slots (`RowsFormat.Fits`), so it is `NetTiming.SchedulesHeld` for any record format whose `fits` is the rows
  format's;
* `SchedulesCoveredAt`: where the network certifier runs the relay's code (`trusted-host`) and no secret moves the
  arrivals, the relay admits no more connections than the caps count, and the schedule it runs for each secret is one
  of `relaySchedules` (`NetTiming.SchedulesCovered`): the count's `schedOf` is the network certifier's run;
* `SchedulesCountedAt`: those schedules number at most `∏ c, (cap c + 1)`, the count the verifier's `schedules` term
  charges (`NetworkCheck.Period.capsTerm` of the caps).

`relaySchedule` reads the relay's lanes as a `NetTiming.WardenSchedule` over the period's connections in admission
order, the ordinals' order: connection `c` is the `c`-th lane admitted, and one no lane holds sends no cell and is
admitted past the run, `D` buckets after the one before it, so it holds no bucket of the period and shares none with
another.

**What they leave out.** The arrivals are fixed: a secret that moves when a client connects moves the schedules too,
and `relaySchedules` is then a union over the arrivals it can give. The period starts at the run's first tick. A
connection the relay fails for its client's sake is not one these schedules give (`CertifierDevice.Relay`).
-/

namespace Properties.NetworkCertifier

open NetworkCheck.Period (Schedule)
open CertifierDevice.Relay (Conn Lane Decision lanes)

/-- The connections a monitoring period of `n` windows can carry, one cap each (`Schedule.caps`): at most `lanes` are
live in a bucket, and each is live in the bucket that admits it. -/
def periodConns (s : Schedule) (n : ℕ) : ℕ := s.lanes * (n * s.window)

/-- Connection `c`'s cap over `n` windows: its entry of `Schedule.caps`. -/
def capOf (s : Schedule) (n : ℕ) (c : Fin (periodConns s n)) : ℕ := ((s.caps n)[c.val]?).getD 0

/-- **The relay's lanes as a schedule** over the period's connections, in admission order: connection `c` is the
`c`-th lane of `ls`, with its admission tick and its cells; one past them sends no cell and is admitted at
`n·T + D·(c + 1)`, past the run. -/
def relaySchedule (s : Schedule) (n : ℕ) (ls : List Lane) : NetTiming.WardenSchedule (Fin (periodConns s n)) where
  r := s.slots
  D := s.delay
  adm c := (ls[c.val]?.map Lane.adm).getD (n * s.window + s.delay * (c.val + 1))
  cells c := (ls[c.val]?.map Lane.cells).getD 0

/-- **The schedules the relay can run** over `n` windows on the arrivals `ready`: the one it runs for each list of
connections arriving at `ready`, whatever their `END`-or-`FAIL` indices. -/
def relaySchedules (s : Schedule) (n : ℕ) (ready : List ℕ) :
    Set (NetTiming.WardenSchedule (Fin (periodConns s n))) :=
  {σ | ∃ conns : List Conn, conns.map Conn.ready = ready ∧
    σ = relaySchedule s n (lanes (CertifierDevice.Relay.run s (n * s.window) conns))}

/-- **The record format holds the relay's schedules**: at a schedule of `r ≥ 1` cells a bucket, for a record format `f`
whose `fits` is the rows format `F`'s over `n` windows at `S = lanes·r` slots (`F`'s connections the period's, in
order, and its window `T` buckets), every schedule the relay can run on any arrivals gives each connection at most its
cap, and `f` holds it (`NetTiming.SchedulesHeld`). -/
def SchedulesHeldAt : Prop :=
  ∀ (s : Schedule) (n : ℕ) (ready : List ℕ) {Cell Ob : Type} {L K : ℕ}
    (F : NetTiming.RowsFormat (Fin (periodConns s n)) Cell L K)
    (f : NetTiming.TapRecord (Fin (periodConns s n)) Cell Ob),
    0 < s.slots → F.conns = List.finRange (periodConns s n) → F.T = s.window →
    (∀ σ, f.fits σ ↔ F.Fits s.allocation n σ) →
    NetTiming.SchedulesHeld f (capOf s n) (relaySchedules s n ready)

/-- **The network certifier runs one of the relay's schedules for each secret**: at a schedule of `r ≥ 1` cells a
bucket, where the device `dev` runs the relay's code over `n` windows (`trusted-host`: `relay dev` is
`CertifierDevice.Relay.run`) and each secret `z` of `Zs` gives connections `conns z` arriving at `ready`, the relay
admits no more connections than the caps count, and the schedule it runs for `z` is one of `relaySchedules`
(`NetTiming.SchedulesCovered`). -/
def SchedulesCoveredAt : Prop :=
  ∀ {Dev Z : Type} (relay : Dev → List Conn → List Decision) (dev : Dev) (s : Schedule) (n : ℕ) (ready : List ℕ)
    (Zs : Set Z) (conns : Z → List Conn),
    0 < s.slots → Verity.Assumptions.TrustedHost dev relay (CertifierDevice.Relay.run s (n * s.window)) →
    (∀ z ∈ Zs, (conns z).map Conn.ready = ready) →
    (∀ z ∈ Zs, (lanes (relay dev (conns z))).length ≤ periodConns s n) ∧
      NetTiming.SchedulesCovered (relaySchedules s n ready) Zs fun z => relaySchedule s n (lanes (relay dev (conns z)))

/-- **The relay's schedules are counted by the caps**: at a schedule of `r ≥ 1` cells a bucket, the schedules the relay
can run over `n` windows on the arrivals `ready` number at most `∏ c, (cap c + 1)`: on fixed arrivals, each lane's
admission is a function of the lanes before it, so a schedule is its connections' cells. -/
def SchedulesCountedAt : Prop :=
  ∀ (s : Schedule) (n : ℕ) (ready : List ℕ), 0 < s.slots →
    (relaySchedules s n ready).encard ≤ ((∏ c, (capOf s n c + 1) : ℕ) : ℕ∞)

end Properties.NetworkCertifier
