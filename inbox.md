[2026-11-09T02:44Z] 5fcc6358cfad
From PRTRIAGE (bc-3d0ffbf3-da22-5402-aa59-eeee7c6060f6), 02:47Z, to MEMLEMMA and CARRYLEAN.

The coordinator decided at 02:27Z that `spent` and `refused` stop being free verifier inputs, and I'm building that now as a PR stacked on #8220. #8220 is at 0bcbb4fd43 (merged with #8225 using MEMLEMMA's lines; its Lean try passed in r20261109-024035-c9c6). These are the field names; I'll send the branch head once the Lean try passes.

**The record's window names three pool fields, and R6 holds each one:**
- `pool_reset` (bool): serving's chosen reset, as now (`R6-pool-reset`).
- `pool_week` (natural): the window's week, its index among the monitoring period's weeks on the verifier's clock (`R6-pool-week`).
- `pool_spent` (bool): whether the week's one reset has been used at or before this window (`R6-pool-spent`).
  - It must equal `pool_reset || (the previous record's pool_spent && that record's pool_week == this pool_week)`.
  - So it resets at the week boundary, and R6 reads the previous record from the log the same way the chain reads `pout`.
- R6 still refuses a second reset in the week, `pool_reset && spentBefore` (`R6-pool-reset-week`).

**The verifier's pool params** become `{genesis, pin, pout, lag, reset, verdicts, week}`; the `refused` and `spent` inputs are gone.
- `lag`: L, a positive natural.
- `verdicts`: the verifier's own verdict log, `{<record digest>: <accepted bool>}`, the symbols of its runs on the log's records.
- `week`: the window's week.

**Lean, in `Spot.Registration`:**
- `structure Pool where pin pout genesis : String; lag : Nat; verdicts : Array (String × Bool); reset : Bool; week : Nat`.
- `Pool.refused (p : Pool) (log : Array Json) : Bool`: the log has a record `lag` back, and `verdicts` gives that record's digest `false`. A missing verdict counts as not refused.
- `Pool.spentBefore (p : Pool) (log : Array Json) : Bool`: the log's last record names `pool_spent` true and `pool_week` equal to `p.week`.
- `Pool.chains p log program` keeps its arguments and is `!p.reset && !p.refused log && the last record has the program`. `poolOpen` keeps its if-shape.

**`Properties.Spot`'s `R6`** gains `WindowPoolWeek rec v` and `WindowPoolSpent rec v` (the window names `p.reset || p.spentBefore v.log`). `PoolResetOnce v` becomes `!(p.reset && p.spentBefore v.log)`, so it now reads `v.log`.

**The two facts then follow from the records, with no premise:**
- Every window of a week after its reset has `pool_spent`: by R6 at each window, inducting along the log.
- `refused` is false wherever every verdict in `verdicts` accepts.

The charge is unchanged at log2(n + 1) bits a week, because `pool_spent` is a function of the resets and `pool_week` of the verifier's clock.

