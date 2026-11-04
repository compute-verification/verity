import Mathlib.Data.Set.Card
import NetworkCertifier.Rows
import Definitions.NetworkCertifier.RecordRows

/-!
# The schedule the rows audit accepts fits the rows format

The property of the schedule a run's records show, stated against the certifier program's audit of the active network
certifier's records as rows (`NetworkCertifier.Rows.audit`, which `network_check`'s `network-rows` runs over a run's
signed records) at its parameters (`NetworkCertifier.Rows.Params`). Nothing assumes what code the network certifier
runs: `trusted-host` covers only that its signed records are the traffic that crossed, at the times it crossed, and the
guarantee reading this takes it there.

`SchedulesCoveredAt` is `NetTiming.SchedulesCovered` into the schedules whose `(r, D)` is registered and which fit the
rows format (`RowsFormat.Fits`), at the registered set `Θ = {(r, D)}` of the audit's parameters: if the audit accepts a
run's records over its `W` windows, they show at most the `lanes·W·T` connections the period's caps count, and the
schedule they show has the parameters' `r` and `D` and fits a well-formed rows format of `T`-bucket windows at
`S = lanes·r` slots. It reads no arrivals: it holds of every run's records, so of every run a client's strategy makes
against the node's egress.

`wardenOf` reads the audit's schedule as a `NetTiming.WardenSchedule` over the period's connections: connection `c` is
ordinal `c + 1`, and one past the run's sends no cell and is admitted past the run, `D` buckets after the one before
it, so it holds no bucket of the period and shares none with another.
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

/-- **The schedule accepted records show is registered and fits**: if `audit` accepts a run's records over its `W`
windows, they show at most the `lanes·W·T` connections the caps count, and the schedule they show is in
`{s | (s.r, s.D) ∈ Θ ∧ F.Fits S W s}` at `Θ = {(r, D)}` and `S = lanes·r`, for every well-formed rows format `F` of
the period's connections whose windows are `T` buckets. -/
def SchedulesCoveredAt : Prop :=
  ∀ (p : Params) {Cell : Type} {L K : ℕ} (ingress egress fails : Array Record) (sum : Summary)
    (F : NetTiming.RowsFormat (Fin (periodConns p egress.size)) Cell L K),
    F.Wf → F.T = p.window → audit p ingress egress fails = .ok sum →
      sum.schedule.adm.size ≤ periodConns p egress.size ∧
        wardenOf p egress.size sum.schedule ∈
          {s | (s.r, s.D) ∈ ({(p.slots, p.delay)} : Set (ℕ × ℕ)) ∧ F.Fits (p.lanes * p.slots) egress.size s}

end Properties.NetworkCertifier
