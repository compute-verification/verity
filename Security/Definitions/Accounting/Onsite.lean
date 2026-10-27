import Definitions.Accounting.JointLive
import Definitions.Accounting.Physical

/-!
# The onsite class, and the links half of it on every run (trusted)

The end-to-end overhead guarantee's Step 3 (note:training-20261027T0134Z-draft-e2e-overhead-proof): Layer 2's claims
(`Accounting.Guarantees.JointPeriodLive`, `JointHorizonBlocksLive`) imply Layer 3's hypotheses
(`Accounting.Guarantees.Actual.DampenerWindowsCap`) on the run's actual circuit.

* `OnsiteClass`: the six hypotheses of `DampenerWindowsCap` that read the actual circuit `A`, verbatim and in its
  order: causality, the compute rate at one per gate, the links as declared, each location's boundary capacity at most
  `αF`, the state at each window's start at most `R`, and the developer's declaration over `[0, t)` containing the
  reference circuit `c`. The rest of its hypotheses read the parameters (`0 < ω`, the locations' disjointness,
  `Φ (X i) ω ≤ Γ`, the prices) or the reference circuit (`T.Threaded`, `WeightsIncompressible`), not `A`.
* `CertifierLinks`: on every run of a map `S` from a game's runs to actual circuits, `OnsiteClass`'s links half,
  `Placed.Links` and the boundary capacity.

A period's view of the run (`Accounting.PeriodLive`) carries no actual circuit: its claims read the commitment's
openings, the device model's schedule and fields of the run. Each bridge takes the run's actual circuit as `S o`, the
run `o` of the game, and says what it meets on the runs where a claim holds; the bridges that no result gives are named
premises (`Accounting.Assumptions.WipeState`, `IntegrityContains`).
-/

namespace Accounting

open Verity Finset

/-- **The onsite class** `𝒜'_onsite`: the actual circuit `A` meets every hypothesis of
`Accounting.Guarantees.Actual.DampenerWindowsCap` that reads it, as it states them. `A` is causal at the latencies `ℓ`,
each location `X i` computes at most `Φ (X i)` gates a tick, every wire between two leaves runs on a declared link
within its capacity (`link`, `ends`, `β`), each location's boundary links carry at most `α F` bits a tick
(`linkCapacity`), each location's state at the start of each window `[kω, (k + 1)ω)` before `t` is at most `R` bits
(`Placed.state`), and the developer's declaration over `[0, t)` at the locations, with the baseline `θ`, contains the
reference circuit `c` (`Placed.declaration`). -/
def OnsiteClass {L E : Type} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t ω : ℕ) (R F α : ℝ) (c : Declaration) : Prop :=
  A.Causal ℓ ∧ A.ComputeRate (Set.range X) (fun _ => 1) Φ ∧ A.Links ends β link ∧
    (∀ i, (linkCapacity ends β (X i) : ℝ) ≤ α * F) ∧
    (∀ i (k : ℕ), k * ω < t → (A.state (X i) (k * ω) : ℝ) ≤ R) ∧
    c.Contains (A.declaration (univ.biUnion X) t θ hθ)

/-- **The links half of the onsite class on every run**: on each run `o`, the actual circuit `(S o).A` runs every wire
between two leaves on a declared link within its capacity (`Placed.Links` at `link o`), and each location `X i`'s
boundary links carry at most `α F` bits a tick. -/
def CertifierLinks {O L E : Type} [Fintype E] (S : O → Actual.Run L) (ends : E → L × L) (β : E → ℕ)
    (link : ∀ o, Fin (S o).A.N → Fin (S o).A.N → E) (n : ℕ) (X : Fin n → Finset L) (α F : ℝ) : Prop :=
  ∀ o, (S o).A.Links ends β (link o) ∧ ∀ i, (linkCapacity ends β (X i) : ℝ) ≤ α * F

end Accounting
