import Definitions.Accounting.Guarantees.Actual.Threaded
import Definitions.Nci.Dampener

/-!
# The compute dampener's windows from the memory cap over threads (trusted)

`Nci.Guarantees.DampenerWindows` takes the monitored servers' windows as one hypothesis over real numbers: `n` servers
over a horizon of `H` ticks cut into `⌈H/L⌉` windows of `L` ticks, each starting from at most `R` bits of state and taking
in at most `αF` bits a tick, hold `b_a (M_r + M_d) ≤ n ⌈H/L⌉ K Λ(K)` counted multiplications of the residual and the
declared computation, `K = R + αF L`. `DampenerWindowsCap` derives that hypothesis on the actual circuit from the memory
cap over threads (`ThreadedRuntime`): at a whole-tick window `ω = L` over the horizon `t = H`, with each location's link
capacity at most `αF` (`r = αF`), when one threading `T` of a reference circuit that the developer's declaration over
`[0, t)` contains counts the residual and the declared computation together (`M_r + M_d ≤ |T.M|`), with `Λ` at `T`'s
reuse (`Nci.Threading.bound`, which is `Nci.Dampener.Lambda`). `ThreadedRuntime` counts the windows as
`n ((t + ω − 1)/ω)`, which is `n ⌈t/ω⌉₊`.
-/

namespace Accounting.Guarantees.Actual

open Verity Nci Finset Nci.Dampener

/-- **The compute dampener's windows from the memory cap over threads.** In `ThreadedRuntime`'s setting, with each
location's link capacity at most `α F` bits a tick, a whole-tick window `ω > 0`, the horizon `t` and `K = R + α F ω`:
when the residual's `M_r` and the declared computation's `M_d` counted multiplications number together at most those of
one threading `T` of the reference circuit `c` (`M_r + M_d ≤ |T.M|`), they meet `Nci.Guarantees.DampenerWindows`' bound
on the servers' windows at `H = t`, `L = ω` and `T`'s reuse: `b_a (M_r + M_d) ≤ n ⌈t/ω⌉₊ K Λ(K)`.

Reads are counted by width, as in `ThreadedRuntime`. -/
def DampenerWindowsCap : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t ω Γ : ℕ) (R F α b_w b_a : ℝ)
    (c : Registered) (T : Threading c) (d e Mr Md : ℕ),
    0 < ω → (∀ i j, i ≠ j → Disjoint (X i) (X j)) →
    A.Causal ℓ → A.ComputeRate (Set.range X) (fun _ => 1) Φ → A.Links ends β link →
    0 < b_w → 0 < b_a → 0 < d →
    (∀ i, Φ (X i) * ω ≤ Γ) → (∀ i, (linkCapacity ends β (X i) : ℝ) ≤ α * F) →
    (∀ i (k : ℕ), k * ω < t → (A.state (X i) (k * ω) : ℝ) ≤ R) →
    c.Contains (A.declaration (univ.biUnion X) t θ hθ) →
    T.Threaded Γ (R + α * F * ω) d e → Nci.Assumptions.WeightsIncompressible c T Γ (R + α * F * ω) b_w b_a →
    Mr + Md ≤ T.M.card →
      b_a * ((Mr : ℝ) + Md) ≤ n * ⌈(t : ℝ) / ω⌉₊ * ((R + α * F * ω) * Lambda T.reuse e d b_w (R + α * F * ω))

/-- **The compute dampener's windows from the memory cap over threads, per parameter slot.** `DampenerWindowsCap` from
`ThreadedRuntimeSlots`, with the reference circuit's weights incompressible per parameter slot of a slot map `σ`
(`Nci.Assumptions.WeightsIncompressibleSlots`): `b_a (M_r + M_d) ≤ n ⌈t/ω⌉₊ K Λ(K)`.

Reads are counted by width, as in `ThreadedRuntime`. -/
def DampenerWindowsCapSlots : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t ω Γ : ℕ) (R F α b_w b_a : ℝ)
    (c : Registered) (T : Threading c) (σ : T.Slots) (d e Mr Md : ℕ),
    0 < ω → (∀ i j, i ≠ j → Disjoint (X i) (X j)) →
    A.Causal ℓ → A.ComputeRate (Set.range X) (fun _ => 1) Φ → A.Links ends β link →
    0 < b_w → 0 < b_a → 0 < d →
    (∀ i, Φ (X i) * ω ≤ Γ) → (∀ i, (linkCapacity ends β (X i) : ℝ) ≤ α * F) →
    (∀ i (k : ℕ), k * ω < t → (A.state (X i) (k * ω) : ℝ) ≤ R) →
    c.Contains (A.declaration (univ.biUnion X) t θ hθ) →
    T.Threaded Γ (R + α * F * ω) d e → Nci.Assumptions.WeightsIncompressibleSlots c T σ Γ (R + α * F * ω) b_w b_a →
    Mr + Md ≤ T.M.card →
      b_a * ((Mr : ℝ) + Md) ≤ n * ⌈(t : ℝ) / ω⌉₊ * ((R + α * F * ω) * Lambda T.reuse e d b_w (R + α * F * ω))

end Accounting.Guarantees.Actual
