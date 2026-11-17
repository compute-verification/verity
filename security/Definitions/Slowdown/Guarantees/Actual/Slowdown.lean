module

public import Definitions.Slowdown.Guarantees.Actual.Threaded
public import Definitions.Slowdown.Slowdown

/-! The slowdown's windows, derived on the actual circuit from the memory cap over threads. -/

@[expose] public section

open Slowdown

namespace ResourceAccounting.Guarantees.Actual

open Verity InferenceOnly Finset Slowdown.Slowdown

/-- **The slowdown's windows from the memory cap over threads.** In `ThreadedRuntime`'s setting, with each
location's link capacity at most `α F` bits a tick, a whole-tick window `ω > 0`, the horizon `t` and `K = R + α F ω`:
when the residual's `M_r` and the declared computation's `M_d` counted multiplications number together at most those of
one threading `T` of the reference training algorithm `c` (`M_r + M_d ≤ |T.M|`), they meet `Slowdown.Guarantees.SlowdownExecutionPhases`' bound
on the servers' windows at `H = t`, `L = ω` and `T`'s reuse: `b_a (M_r + M_d) ≤ n ⌈t/ω⌉₊ K Λ(K)`.

Reads are counted by width, as in `ThreadedRuntime`. -/
def SlowdownServerBlocksCap : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t ω Γ : ℕ) (R F α b_w b_a : ℝ)
    (c : RegisteredCircuit) (T : Threading c) (d e Mr Md : ℕ),
    0 < ω → (∀ i j, i ≠ j → Disjoint (X i) (X j)) →
    A.Causal ℓ → A.ComputeRate (Set.range X) (fun _ => 1) Φ → A.Links ends β link →
    0 < b_w → 0 < b_a → 0 < d →
    (∀ i, Φ (X i) * ω ≤ Γ) → (∀ i, (linkCapacity ends β (X i) : ℝ) ≤ α * F) →
    (∀ i (k : ℕ), k * ω < t → (A.state (X i) (k * ω) : ℝ) ≤ R) →
    c.Contains (A.declaration (univ.biUnion X) t θ hθ) →
    T.Threaded Γ (R + α * F * ω) d e → Slowdown.Assumptions.WeightsIncompressible c T Γ (R + α * F * ω) b_w b_a →
    Mr + Md ≤ T.M.card →
      b_a * ((Mr : ℝ) + Md) ≤ n * ⌈(t : ℝ) / ω⌉₊ * ((R + α * F * ω) * Lambda T.reuse e d b_w (R + α * F * ω))

/-- **The slowdown's windows from the memory cap over threads, per parameter slot.** `SlowdownServerBlocksCap` from
`ThreadedRuntimeSlots`, with the reference training algorithm's weights incompressible per parameter slot of a slot map `σ`
(`Slowdown.Assumptions.WeightsIncompressibleSlots`): `b_a (M_r + M_d) ≤ n ⌈t/ω⌉₊ K Λ(K)`.

Reads are counted by width, as in `ThreadedRuntime`. -/
def SlowdownServerBlocksCapSlots : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t ω Γ : ℕ) (R F α b_w b_a : ℝ)
    (c : RegisteredCircuit) (T : Threading c) (σ : T.Slots) (d e Mr Md : ℕ),
    0 < ω → (∀ i j, i ≠ j → Disjoint (X i) (X j)) →
    A.Causal ℓ → A.ComputeRate (Set.range X) (fun _ => 1) Φ → A.Links ends β link →
    0 < b_w → 0 < b_a → 0 < d →
    (∀ i, Φ (X i) * ω ≤ Γ) → (∀ i, (linkCapacity ends β (X i) : ℝ) ≤ α * F) →
    (∀ i (k : ℕ), k * ω < t → (A.state (X i) (k * ω) : ℝ) ≤ R) →
    c.Contains (A.declaration (univ.biUnion X) t θ hθ) →
    T.Threaded Γ (R + α * F * ω) d e → Slowdown.Assumptions.WeightsIncompressibleSlots c T σ Γ (R + α * F * ω) b_w b_a →
    Mr + Md ≤ T.M.card →
      b_a * ((Mr : ℝ) + Md) ≤ n * ⌈(t : ℝ) / ω⌉₊ * ((R + α * F * ω) * Lambda T.reuse e d b_w (R + α * F * ω))

end ResourceAccounting.Guarantees.Actual
