import Models.Accounting.Guarantees.Actual
import Models.Nci.Assumptions.Threads

/-!
# The memory cap over threads (trusted)

Tier 3's bound on the monitored servers (`benchmarks/nci/tiers.py`, "Memory cap on the residual"): a server's window
reads its starting state plus what enters during it, `K = R + r ω`, and holds at most `K Λ(K) / b_a` counted
multiplications of a threaded reference circuit `c`, where `Λ = Threading.bound` is NCI's Theorem 1 over threads
(`Nci.Guarantees.ThreadedReads`). Every gate placed on the servers counts, declared or not: the windows divide the actual
circuit's region, and `c` need only be contained in it.

The windows are `Windows`' `A[X i, [kω, (k + 1)ω) ∩ [0, t)]`, with its memory constraint replaced by a bound `R` on the
state at each window's start, so that what a wipe or exhaustion leaves plugs in; `r` bounds each location's link
capacity `β_X`, and `Γ ≥ Φ_X ω` its gates a window.

* `ThreadedRuntime`: `b_a |M| ≤ n ⌈t/ω⌉ K Λ(K)`, so `t ≥ ω (b_a |M| / (n K Λ(K)) − 1)`.
* `ThreadedRate`: at the window `ω = R/r` in the quadratic regime, with a wipe period dividing the window, at most
  `reuse · n R r / (d b_w b_a)` counted multiplications a tick.

Both count reads by width: `Placed.state` and `novelReads` are widths of the gates read. A wipe bounds information, what
the state can determine, not the width of the gates a window reads from before its start, so composing these with a wipe
needs the read bound in its determination form.
-/

namespace Accounting.Guarantees.Actual

open Verity Nci Finset

/-- **The memory cap over threads** (tier 3's server-window bound). For an actual circuit `A` meeting causality,
unit-priced compute rate and links at the developer's pairwise disjoint locations `X i`, a window `ω > 0`, a horizon `t`,
and a reference circuit `c` that the developer's declaration over `[0, t)` contains, threaded by `T` with
`T.Threaded Γ K d e` and `WeightsIncompressible` at `(Γ, K)`, `K = R + r ω`: when each location does at most `Γ` gates
a window (`Φ_{X i} ω ≤ Γ`), its link capacity is at most `r`, and its state at the start of each window
`[kω, (k + 1)ω)` that begins before `t` is at most `R`, the counted multiplications number at most `K Λ(K) / b_a` per
window, `Λ = T.bound`:
`b_a |T.M| ≤ n ⌈t/ω⌉ K Λ(K)`, so `t ≥ ω (b_a |T.M| / (n K Λ(K)) − 1)`.

Reads are counted by width (`Placed.state`, `novelReads`); a wipe bounds information, so composing this with a wipe
needs the determination form of the read bound. -/
def ThreadedRuntime : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t ω Γ : ℕ) (R r b_w b_a : ℝ)
    (c : Declaration) (T : Threading c) (d e : ℕ),
    0 < ω → (∀ i j, i ≠ j → Disjoint (X i) (X j)) →
    A.Causal ℓ → A.ComputeRate (Set.range X) (fun _ => 1) Φ → A.Links ends β link →
    0 < b_w → 0 < b_a → 0 < d →
    (∀ i, Φ (X i) * ω ≤ Γ) → (∀ i, (linkCapacity ends β (X i) : ℝ) ≤ r) →
    (∀ i (k : ℕ), k * ω < t → (A.state (X i) (k * ω) : ℝ) ≤ R) →
    c.Contains (A.declaration (univ.biUnion X) t θ hθ) →
    T.Threaded Γ (R + r * ω) d e → Nci.Assumptions.WeightsIncompressible c T Γ (R + r * ω) b_w b_a →
      b_a * T.M.card ≤ (n * ((t + ω - 1) / ω) : ℕ) * ((R + r * ω) * T.bound (R + r * ω) b_w d e) ∧
      (ω : ℝ) * (b_a * T.M.card / (n * ((R + r * ω) * T.bound (R + r * ω) b_w d e)) - 1) ≤ t

/-- **The memory cap's rate at the optimal window** (tier 3, the quadratic regime). In `ThreadedRuntime`'s setting, when
the state at each wipe (every multiple of the period `p` before `t`) is at most `R`, the period divides the window and the
window divides `t`, the window is the optimal `ω = R/r` (`r ω = R`, which minimizes `(R + r ω)² / ω`), so `K = 2R`, and
`Λ(2R)` is in its quadratic regime (`e ≤ reuse · 2R / (4 d b_w)`): `b_a b_w d |T.M| ≤ reuse · n R r t`, at most
`reuse · n R r / (d b_w b_a)` counted multiplications a tick, `2 R r / (d b_w b_a)` a server at reuse 2. When counted
multiplications are a fraction `κ` of `c`'s gates, 2 gates each, `κ b_a b_w d size(c) ≤ 2 reuse · n R r t`: against the
servers' `n Φ t` gates, an efficiency of at most `2 reuse · R r / (κ b_a b_w d Φ)`, `4 R r / (κ b_a b_w d Φ)` at reuse 2.

Reads are counted by width, as in `ThreadedRuntime`. -/
def ThreadedRate : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t ω p Γ : ℕ) (R r b_w b_a : ℝ)
    (c : Declaration) (T : Threading c) (d e : ℕ),
    0 < ω → (∀ i j, i ≠ j → Disjoint (X i) (X j)) →
    A.Causal ℓ → A.ComputeRate (Set.range X) (fun _ => 1) Φ → A.Links ends β link →
    0 < b_w → 0 < b_a → 0 < d →
    (∀ i, Φ (X i) * ω ≤ Γ) → (∀ i, (linkCapacity ends β (X i) : ℝ) ≤ r) →
    (∀ i (j : ℕ), j * p < t → (A.state (X i) (j * p) : ℝ) ≤ R) → p ∣ ω → ω ∣ t → r * ω = R →
    (e : ℝ) ≤ T.reuse * (2 * R) / (4 * d * b_w) →
    c.Contains (A.declaration (univ.biUnion X) t θ hθ) →
    T.Threaded Γ (2 * R) d e → Nci.Assumptions.WeightsIncompressible c T Γ (2 * R) b_w b_a →
      b_a * b_w * d * T.M.card ≤ T.reuse * n * R * r * t ∧
      ∀ κ : ℝ, κ * c.D.size c.D.computed ≤ 2 * T.M.card →
        κ * b_a * b_w * d * c.D.size c.D.computed ≤ 2 * T.reuse * n * R * r * t

end Accounting.Guarantees.Actual
