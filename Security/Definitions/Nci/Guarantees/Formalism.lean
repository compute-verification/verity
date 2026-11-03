import Definitions.Nci.Assumptions.Formalism

/-!
# The training formalism v1: its theorems

The three theorems of Daniel's training formalism v1 (`Definitions.Nci.Formalism`), Fit, Slowdown and Security, proved in
`Nci.Formalism.SecurityProofs`. Its two lemmas, reconstruction (§1) and the three kinds of a block's inputs (§2), are
`Nci.Formalism.reconstruction` and `Nci.Formalism.reads_split` in `Security/Proofs/Nci/Formalism/Reconstruction.lean`.

* **Fit** (§5): full blocks that read within `k` give `ρ = 1`. Serving (`FitServing`): `Δ + s τ ≤ k`, and with `Δ = 0`
  honest serving is free, one block per session (`FitHonestServing`). Training reduced by a tree (`FitTraining`):
  `Δ + 2 Δ_grad + s τ ≤ k` for each tree block; Daniel's `2 Δ_grad + s τ ≤ k` is its case `Δ = 0`, a step whose tokens
  run on the registered weights, and a tree block that runs its tokens on the updated model reads its description too.
  Fit counts tokens `s` as a block reads them; `s = c / w` holds when the update's decoder takes no gates of the block,
  and a decoder of `d` gates leaves room for `(c - d) / w`;
* **Slowdown** (§5): under A1, `ρ ≥ φ (1 - x₀ - ε) s a / M`, for any `M ≥ max_x x (k - u(x))`, any `x₀` at least the
  bottom share and tokens linked up to a share `ε`; the constant is `1`, and `ε = 0` is Daniel's bound. Evenly spread,
  `u(x) = x u(1)` with `u(1) ≥ k` (`SlowdownEven`): `ρ ≥ 4 φ s a (u(1) - k - ε u(1)) / k²`, Daniel's `4 φ s a u(1) / k²`
  with the factor `1 - k / u(1) - ε`. A frozen tail of share `t` (`SlowdownFrozenTail`): `ρ ≥ φ (1 - t - ε) s a / (k t)`,
  Daniel's `φ s a / (k t)` with the factor `1 - t - ε`, which is near `1` for a short tail;
* **Security** (§5): under A1 and A2, a run of the methods considered that reaches `ℓ*` in `N` periods of `n` blocks of
  `c` gates has `N n c ≥ ρ̲ p H(ℓ*)`, `ρ̲` Slowdown's bound for it and `p` A2's floor.

Slowdown, its two cases and Security each have an honest and a nontrivial lemma, their premises holding together and
each bound failing without A1 (`Definitions.Nci.Guarantees.Formalism.Witnesses`).
-/

namespace Nci.Formalism.Guarantees

open Finset
open Verity (Declaration Partition)
open Nci.Formalism Nci.Formalism.Assumptions
open scoped NNReal ENNReal

/-- **Fit, serving**: session blocks that read within `k`, `Δ + s τ ≤ k`, give `ρ = 1`. -/
def FitServing : Prop :=
  ∀ (d : Declaration) (B : ℕ) (P : Partition d.D B) (c k Δ s τ : ℕ), SessionBlocks d P c Δ s τ → 0 < B → 0 < c →
    ServingFits Δ s τ k → slowdown d c k = 1

/-- **Fit, honest serving**: with `Δ = 0`, session blocks that read `s τ ≤ k` give `ρ = 1`, one block per session. -/
def FitHonestServing : Prop :=
  ∀ (d : Declaration) (B : ℕ) (P : Partition d.D B) (c k s τ : ℕ), SessionBlocks d P c 0 s τ → 0 < B → 0 < c →
    ServingFits 0 s τ k → slowdown d c k = 1

/-- **Fit, training**: tree blocks that read within `k`, `Δ + 2 Δ_grad + s τ ≤ k`, give `ρ = 1`. -/
def FitTraining : Prop :=
  ∀ (d : Declaration) (B : ℕ) (P : Partition d.D B) (c k Δ Δgrad s τ : ℕ), TreeBlocks d P c Δ Δgrad s τ → 0 < B →
    0 < c → TreeFits Δ Δgrad s τ k → slowdown d c k = 1

/-- **Slowdown**: under A1, `ρ ≥ φ (1 - x₀ - ε) s a / M`. -/
def Slowdown : Prop :=
  ∀ (d : Declaration) (L : Type) [Fintype L] (A : Annotation d L) (c k : ℕ) (u : ℝ≥0 → ℝ) (a x₀ ε M : ℝ),
    A.Valid → CutCost d A c u a → A.TokensLinked ε → PeakBound u k M → BottomShare u k x₀ → 0 ≤ a →
    ENNReal.ofReal (A.depShare * (1 - x₀ - ε) * A.tokensPerBlock c * a / M) ≤ slowdown d c k

/-- **Slowdown, evenly spread update** `u(x) = x u(1)` with `u(1) ≥ k > 0`: `ρ ≥ 4 φ s a (u(1) - k - ε u(1)) / k²`. -/
def SlowdownEven : Prop :=
  ∀ (d : Declaration) (L : Type) [Fintype L] (A : Annotation d L) (c k : ℕ) (u₁ a ε : ℝ),
    A.Valid → CutCost d A c (fun x => (x : ℝ) * u₁) a → A.TokensLinked ε → 0 < k → (k : ℝ) ≤ u₁ → 0 ≤ a →
    ENNReal.ofReal (4 * A.depShare * A.tokensPerBlock c * a * (u₁ - k - ε * u₁) / (k : ℝ) ^ 2) ≤ slowdown d c k

/-- **Slowdown, a single updated layer with a frozen tail** of share `t`: `ρ ≥ φ (1 - t - ε) s a / (t k)`. -/
def SlowdownFrozenTail : Prop :=
  ∀ (d : Declaration) (L : Type) [Fintype L] (A : Annotation d L) (c k : ℕ) (u : ℝ≥0 → ℝ) (a t ε : ℝ),
    A.Valid → CutCost d A c u a → A.TokensLinked ε → FrozenTail u k t → 0 ≤ t → 0 ≤ a →
    ENNReal.ofReal (A.depShare * (1 - t - ε) * A.tokensPerBlock c * a / (t * k)) ≤ slowdown d c k

/-- **Security**: under A1 and A2, a run of `methods` whose update stays within `D` bits and that reaches `ℓ*` through a
`(c, k)`-partition whose blocks fit in `N` periods of `n` blocks has `N n c ≥ ρ̲ p H(ℓ*)`, `ρ̲` Slowdown's bound. -/
def Security : Prop :=
  ∀ (methods : Set Run) (dec : Verity.Decoder (List Bool)) (ℓ : List Bool → ℝ) (ℓstar : ℝ) (D : ℕ∞) (p : ℝ) (r : Run)
    (c k n N B : ℕ) (P : Partition r.d.D B) (u : ℝ≥0 → ℝ) (a x₀ ε M : ℝ),
    EfficiencyFloor methods dec c ℓ ℓstar D p → r.Among methods → r.A.Valid → r.Within dec c D → r.Reaches ℓ ℓstar →
    CutCost r.d r.A c u a → r.A.TokensLinked ε → PeakBound u k M → BottomShare u k x₀ → 0 ≤ a → 0 ≤ p →
    IsCKPartition r.d P c k → InPeriods B N n →
    ENNReal.ofReal (r.A.depShare * (1 - x₀ - ε) * r.A.tokensPerBlock c * a / M * p) * honestCost methods ℓ ℓstar ≤
      ((N * n * c : ℕ) : ℝ≥0∞)

end Nci.Formalism.Guarantees
