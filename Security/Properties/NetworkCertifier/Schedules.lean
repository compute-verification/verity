import Mathlib.Data.Set.Card
import NetworkCertifier.Rows
import Definitions.NetworkCertifier.PeriodConditions
import Definitions.NetworkCertifier.RecordRows

/-!
# The schedules the rows audit accepts, and its record format holds them

The properties of the schedule a run's records show, stated against the certifier program's audit of the active network
certifier's records as rows (`NetworkCertifier.Rows.audit`, which `network_check`'s `network-rows` runs over a run's
signed records) at its parameters (`NetworkCertifier.Rows.Params`), over a monitoring period of `n` windows of `T`
buckets. Nothing assumes what code the network certifier runs: `trusted-host` covers only that its signed records are
the traffic that crossed, at the times it crossed, and the guarantee reading these takes it there. The schedules `Ss`
are by definition the ones the audit accepts (`auditSchedules`), on any records, so on any arrivals:

* `SchedulesCoveredAt`: if the audit accepts a run's records, they show at most the connections the caps count, and the
  schedule they show is one of `auditSchedules`, so it is `NetTiming.SchedulesCovered` at `schedOf` the accepted
  records' schedule;
* `SchedulesHeldAt`: every schedule of `auditSchedules` gives each connection at most `egress_cells` cells and fits the
  rows format at `S = lanes·r` slots (`RowsFormat.Fits`), so it is `NetTiming.SchedulesHeld` for any record format
  whose `fits` is the rows format's;
* `SchedulesCountedAt`: those schedules number at most `(lanes + 1)^(n·T) · (egress_cells + 1)^(lanes·n·T)`: a
  schedule is its admissions, the ordinals admitted in ordinal order and at most `lanes` in a bucket, and each
  connection's cells.

`wardenOf` reads the audit's schedule as a `NetTiming.WardenSchedule` over the period's connections: connection `c` is
ordinal `c + 1`, and one past the run's sends no cell and is admitted past the run, `D` buckets after the one before
it, so it holds no bucket of the period and shares none with another.

**What they leave out.** The count is over every arrival list: a client may time its arrivals to what it has observed,
and the admissions a secret moves that way are the count's first factor, `(lanes + 1)^(n·T)`. The rows form's
`schedules` term (`NetworkCheck.Period.rowsTerms`) charges `capsTerm` of the caps, the second factor, and not the
first.
-/

open NetworkCertifier.Rows (Params Record Schedule Summary audit)

namespace Properties.NetworkCertifier

/-- The connections a monitoring period of `n` windows can carry, one cap each (`NetworkCheck.Period.Schedule.caps`):
at most `lanes` are live in a bucket, and each is live in the bucket that admits it. -/
def periodConns (p : Params) (n : ℕ) : ℕ := p.lanes * (n * p.window)

/-- **The audit's schedule as a `NetTiming.WardenSchedule`** over the period's connections: connection `c` is ordinal
`c + 1`, with its admission bucket and its lane's cells; one past the run's sends no cell and is admitted at
`n·T + D·(c + 1)`, past the run. -/
def wardenOf (p : Params) (n : ℕ) (s : Schedule) : NetTiming.WardenSchedule (Fin (periodConns p n)) where
  r := p.slots
  D := p.delay
  adm c := s.adm.getD c.val (n * p.window + p.delay * (c.val + 1))
  cells c := s.cells.getD c.val 0

/-- **The schedules the audit accepts** over `n` windows: the one shown by each run's records that `audit` accepts,
whatever arrivals and data they carry. -/
def auditSchedules (p : Params) (n : ℕ) : Set (NetTiming.WardenSchedule (Fin (periodConns p n))) :=
  {σ | ∃ (ingress egress fails : Array Record) (sum : Summary), egress.size = n ∧
    audit p ingress egress fails = .ok sum ∧ σ = wardenOf p n sum.schedule}

/-- **The schedule accepted records show is counted**: if `audit` accepts a run's records over its `W` windows, they
show at most the `lanes·W·T` connections the caps count, and the schedule they show is one of `auditSchedules`. -/
def SchedulesCoveredAt : Prop :=
  ∀ (p : Params) (ingress egress fails : Array Record) (sum : Summary),
    audit p ingress egress fails = .ok sum →
      sum.schedule.adm.size ≤ periodConns p egress.size ∧
        wardenOf p egress.size sum.schedule ∈ auditSchedules p egress.size

/-- **The record format holds the audit's schedules**: for a record format `f` whose `fits` is the rows format `F`'s
over `n` windows at `S = lanes·r` slots (`F`'s connections the period's, in order, and its window `T` buckets), every
schedule the audit accepts gives each connection at most `egress_cells` cells, and `f` holds it
(`NetTiming.SchedulesHeld`). -/
def SchedulesHeldAt : Prop :=
  ∀ (p : Params) (n : ℕ) {Cell Ob : Type} {L K : ℕ} (F : NetTiming.RowsFormat (Fin (periodConns p n)) Cell L K)
    (f : NetTiming.TapRecord (Fin (periodConns p n)) Cell Ob),
    F.conns = List.finRange (periodConns p n) → F.T = p.window →
    (∀ σ, f.fits σ ↔ F.Fits (p.lanes * p.slots) n σ) →
    NetTiming.SchedulesHeld f (fun _ => p.egressCells) (auditSchedules p n)

/-- **The audit's schedules are counted** over `n` windows, on every arrival list: they number at most
`(lanes + 1)^(n·T)` admissions, as the ordinals are admitted in ordinal order and at most `lanes` in a bucket, times
`egress_cells + 1` cells for each of the period's connections. -/
def SchedulesCountedAt : Prop :=
  ∀ (p : Params) (n : ℕ),
    (auditSchedules p n).encard ≤ (((p.lanes + 1) ^ (n * p.window) * (p.egressCells + 1) ^ periodConns p n : ℕ) : ℕ∞)

end Properties.NetworkCertifier
