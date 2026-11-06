import Definitions.Accounting.Additive

/-!
# Compute additivity's capacity from the machine model 𝓜 (trusted)

Compute additivity (W: `Accounting.Assumptions.RateComputeAdditive` for a joint run,
`Accounting.Remote.Assumptions.ComputeAdditive` for the remote audit) says the works the subprotocols certify on one
node add under its capacity. 𝓜 (`Verity.Accel.Machine`) gives the capacity part, counted once per resource:

* **Rates add over runs back to back** (`AccelRate`). A device's period is runs of 𝓜 back to back
  (`Accounting.AccelRun`), and its instructions are one stream. Runs that each use at most `ρ_r` units of each resource
  `r` a second over their own seconds use at most that over the period's seconds, their sum. 𝓜's rate (G2,
  `Machine.Fits`) is that bound, run by run.
* **Disjoint instructions share the capacity** (`AccelAdditive`). Instructions that different subprotocols run are
  disjoint sets of the stream, so over a period of `T` seconds that meets the rate, they use at most `ρ_r T` of each
  resource `r` together.

Neither takes an assumption. 𝓜 has no time inside a run, so the windows these bound are blocks of a period's runs, each
a period itself: a block of the runs `es` meets `AccelRate`'s hypothesis when `es` does.

The proofs are in `Security/Proofs/Accounting/Additive.lean`.
-/

open scoped NNRat

namespace Accounting.Guarantees

open Finset Verity.Accel

/-- **AccelRate**: runs of 𝓜 whose instructions each use at most `ρ_r T` units of each resource `r`, at the run's
seconds `T`, use at most `ρ_r` times their seconds summed back to back. Each run's hypothesis follows from 𝓜's rate
(G2, `Machine.Fits` at the run's seconds, which also counts the code's fetch). -/
def AccelRate : Prop :=
  ∀ (m : Machine) (es : List (AccelRun m)),
    (∀ e ∈ es, ∀ r : Res, Machine.units e.instrs r ≤ m.card.ρ r * e.T) →
    ∀ r : Res, Machine.units (stream es) r ≤ m.card.ρ r * periodSeconds es

/-- **AccelAdditive**: over a period whose instructions `iss` use at most `ρ_r T` units of each resource `r` in its `T`
seconds (`T` its runs' seconds summed, by `AccelRate`), instructions that different subprotocols issue (disjoint sets
`S j` of its instructions) use at most `ρ_r T` units of each resource `r` together. -/
def AccelAdditive : Prop :=
  ∀ (m : Machine) (iss : List m.Instr) (T : ℚ≥0),
    (∀ r : Res, Machine.units iss r ≤ m.card.ρ r * T) →
    ∀ {k : ℕ} (S : Fin k → Finset ℕ), (∀ j, S j ⊆ issued iss) →
      (∀ i j, i ≠ j → Disjoint (S i) (S j)) → ∀ r : Res,
        ∑ j, ∑ p ∈ S j, occUse iss r p ≤ m.card.ρ r * T

end Accounting.Guarantees
