module

public import Definitions.Compose.Additive

/-! Compute additivity's capacity from the machine model 𝓜, with no assumption. -/

@[expose] public section

open scoped NNRat

namespace ResourceAccounting.Guarantees

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

end ResourceAccounting.Guarantees
