import Definitions.Accounting.Additive

/-!
# Compute additivity's capacity from the machine model 𝓜 (trusted)

Compute additivity (W: `Accounting.Assumptions.RateComputeAdditive` for a joint run,
`Accounting.Remote.Assumptions.ComputeAdditive` for the remote audit) says the works the subprotocols certify on one
node add under its capacity. 𝓜 (`Verity.Accel.Machine`) gives the capacity part, counted once per resource:

* **Rates add over runs back to back** (`AccelRate`). A device's period is runs of 𝓜 back to back
  (`Accounting.AccelRun`), and its instructions are one stream. Runs that each meet a rate at a backlog of `b` ticks
  meet it together at `n·b`, for `n` runs. 𝓜's rate (G2, `Machine.Rate`) is that rate at `b = slack`, run by run, so a
  period of 𝓜's runs meets it at a backlog of `n·slack` ticks.
* **Disjoint instructions share the capacity** (`AccelAdditive`). Instructions that different subprotocols run are
  disjoint sets of the stream, so over a window of `T` ticks of a period that meets the rate at a backlog of `β` ticks,
  they use at most `ρ_r (T + β)` of each resource `r` together.

Neither takes an assumption. 𝓜 bounds a run's backlog, not a period's: each datum a device receives starts a run, so the
period's backlog `n·slack` grows with the data it receives.

The proofs are in `Security/Proofs/Accounting/Additive.lean`.
-/

open scoped NNRat

namespace Accounting.Guarantees

open Finset Verity.Accel

/-- **AccelRate**: runs of 𝓜 whose instructions each use, over every window, at most `ρ_r` units of each resource `r` a
tick plus `b` ticks' worth meet that rate back to back at `n·b` ticks' worth, for `n` runs. At `b = slack` each run's
hypothesis is 𝓜's rate (G2, `Machine.Rate`). -/
def AccelRate : Prop :=
  ∀ {ι : Type} (m : Machine ι) (es : List (AccelRun m)) (b : ℕ),
    (∀ e ∈ es, ∀ (w : Window) (r : m.Res),
      ∑ t ∈ w.ticks, Machine.usage e.instrs t r ≤ m.ρ r * ((w.len + b : ℕ) : ℚ≥0)) →
    ∀ (w : Window) (r : m.Res),
      ∑ t ∈ w.ticks, Machine.usage (stream es) t r ≤ m.ρ r * ((w.len + es.length * b : ℕ) : ℚ≥0)

/-- **AccelAdditive**: over any window `w` of a period whose instructions `iss` use, over every window, at most `ρ_r`
units of each resource `r` a tick plus `β` ticks' worth (`β = n·slack` for `n` runs of 𝓜, by `AccelRate`), instructions
that different subprotocols issue (disjoint sets `S j` of the instructions issued during `w`) use at most `ρ_r (T + β)`
units of each resource `r` together. -/
def AccelAdditive : Prop :=
  ∀ {ι : Type} (m : Machine ι) (iss : List (List m.Instr)) (β : ℕ),
    (∀ (w : Window) (r : m.Res), ∑ t ∈ w.ticks, Machine.usage iss t r ≤ m.ρ r * ((w.len + β : ℕ) : ℚ≥0)) →
    ∀ (w : Window) {k : ℕ} (S : Fin k → Finset (ℕ × ℕ)), (∀ j, S j ⊆ issued iss w) →
      (∀ i j, i ≠ j → Disjoint (S i) (S j)) → ∀ r : m.Res,
        ∑ j, ∑ p ∈ S j, occUse iss r p ≤ m.ρ r * ((w.len + β : ℕ) : ℚ≥0)

end Accounting.Guarantees
