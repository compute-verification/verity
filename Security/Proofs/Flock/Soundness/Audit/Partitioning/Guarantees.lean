import Proofs.Flock.Soundness.Audit.Partitioning.Model

/-!
# Optimal partitioning: the guarantees

What the verifier must spend, over every draw law, to bound each adversary's gain, and which partition minimizes it. The
definitions they read are in `Partitioning.Model`; `Λ` below is `ln(1/δ)`.

**Free compute.** A wrong set `B` frees the work `∑_{u ∈ B} w u`, and the target is `MeetsTarget L w C δ`.

* `ConverseExact`, `ConverseAsymptotic`: a law under which every set of `K` units escapes with probability at most `δ`
  draws at least `(n − K + 1)(1 − δ^{1/K}) ≥ (n − K + 1)/K · Λ · (1 − Λ/(2K))` units in expectation.
* `Achievability`: independent draws with `q_u = min(1, Λ/C · w u)` meet the target, up to any `ε` per unit.
* `GranularityOptimum` (headline): with `n` units of work `s` each, the cheapest law costs
  `c(s) · (ns/C) · Λ` up to the factor `1 − s/C − C/(ns) − sΛ/(2C)`, with `c(s) = f + o·s`. The total work `ns` and so
  `(ns/C) · Λ` don't depend on `s`. `GranularityChoice`: so the best granularity minimizes `c(s)`, up to that factor.
* `HeavyUnit`: a unit doing work at least `C` is drawn with probability at least `1 − δ`.
* `UnequalDraws`, `UnequalCost`: with unequal units of work in `[a, A]` and `A · Λ ≤ C`, every law meeting the target
  draws at least `1/ρ` of `Achievability`'s `F/C · Λ` (`F` the total work), and pays at least `1/ρ²` of its cost, less
  one term per class `[aρ^j, aρ^{j+1})` of units. `UnequalCost` also assumes `a ≤ A`, which matters only with no units.

**Exfiltration**, as `audit_exfiltration` bounds it: the target is `CertifiesLeak L b δ ℓ`. These compare upper bounds
on the leak, not leaks an adversary achieves.

* `ExfiltrationOptimum`: with every unit `b` bits wide and `K` the most wrong units the target `ℓ` allows
  (`b·K + locBits n K ≤ ℓ < b·(K+1) + locBits n (K+1)`), a certifying law draws at least `(n − K)(1 − δ^{1/(K+1)})`
  units, and one draws at most `n · Λ/(K+1)`, up to any `ε`.
* `RefinementRule` (headline): splitting each unit into `r` of width `b'` costs at least what the unsplit partition
  achieves, up to the factor `(1 − K/n)(1 − Λ/(2(K+1)))`, unless `b' + log₂ r < b + log₂(K+2)/(K+1)`: refining pays
  only when it cuts each unit's width by more than about `log₂ r` bits.

**Two-stage**, with `ψ = detectionRisk N k p`:

* `RiskConcave`: for `k ≥ 2` and `p ≤ 1/2`, the increments of `ψ` do not increase.
* `Concentration`: so for wrong proof units spread over replay units with `m_i` each, `∑_i ψ(m_i)` is smallest when
  they fill whole replay units, with at most one partial: the prover's least detectable placement concentrates.
* `NestedEscape`: `ψ` is the nested law's detection risk. Under `nested`, a wrong set with `m_i` wrong proof units in
  replay unit `i` escapes with probability `∏_i (1 − p + p · C(N − m_i, k)/C(N, k))`, which for `p < 1` is
  `exp(−∑_i ψ(m_i))`.
-/

namespace FlockSoundness.Audit.Partitioning.Guarantees

open Finset

/-- **The converse, exactly.** A law under which every set of at least `K` of its `n` units escapes with probability at
most `δ` draws at least `(n − K + 1)(1 − δ^{1/K})` units in expectation. -/
def ConverseExact : Prop :=
  ∀ (n K : ℕ) (L : Law n) (δ : ℝ), 0 < K → K ≤ n → (L.miss K).toReal ≤ δ →
    ((n : ℝ) - K + 1) * (1 - δ ^ ((K : ℝ)⁻¹)) ≤ meanDraws L

/-- **The converse, asymptotically**: `(n − K + 1)(1 − δ^{1/K}) ≥ (n − K + 1)/K · Λ · (1 − Λ/(2K))`, so a law meeting
`ConverseExact`'s hypothesis draws about `(n/K) · ln(1/δ)` units when `K ≫ ln(1/δ)`. -/
def ConverseAsymptotic : Prop :=
  ∀ (n K : ℕ) (L : Law n) (δ : ℝ), 0 < K → K ≤ n → 0 < δ → δ ≤ 1 → (L.miss K).toReal ≤ δ →
    ((n : ℝ) - K + 1) / K * Real.log δ⁻¹ * (1 - Real.log δ⁻¹ / (2 * K)) ≤ meanDraws L

/-- **Achievability**: for any `ε > 0`, some law meets the free-compute target while drawing unit `u` with probability
at most `min(1, Λ/C · w u) + ε`. -/
def Achievability : Prop :=
  ∀ (n : ℕ) (w : Fin n → ℝ) (C δ ε : ℝ), (∀ u, 0 ≤ w u) → 0 < C → 0 < δ → δ ≤ 1 → 0 < ε →
    ∃ L : Law n, MeetsTarget L w C δ ∧ ∀ u, (L.incl u).toReal ≤ min 1 (Real.log δ⁻¹ / C * w u) + ε

/-- **The granularity optimum.** With `n` units of work `s` each, every law meeting the free-compute target costs at
least `c(s) · (ns/C) · Λ · (1 − s/C − C/(ns) − sΛ/(2C))`, and some law costs at most `c(s) · (ns/C) · Λ + ε`. -/
def GranularityOptimum : Prop :=
  ∀ (n : ℕ) (s C f o δ : ℝ), 0 < s → 0 < C → 0 ≤ f → 0 ≤ o → 0 < δ → δ ≤ 1 →
    (∀ L : Law n, MeetsTarget L (fun _ => s) C δ →
      unitCost f o s * (n * s / C * Real.log δ⁻¹) * (1 - s / C - C / (n * s) - s * Real.log δ⁻¹ / (2 * C)) ≤
        cost L fun _ => unitCost f o s) ∧
    ∀ ε > 0, ∃ L : Law n, MeetsTarget L (fun _ => s) C δ ∧
      cost L (fun _ => unitCost f o s) ≤ unitCost f o s * (n * s / C * Real.log δ⁻¹) + ε

/-- **The best granularity minimizes `c(s)`.** Divide the same work into `n` units of size `s` or `n'` of size `s'`.
For every law meeting the target at `s'`, some law meets it at `s` and costs at most `c(s)/c(s')` times as much, up to
the factor `1 − s'/C − C/(n's') − s'Λ/(2C)` and any `ε`. Stated cross-multiplied. -/
def GranularityChoice : Prop :=
  ∀ (n n' : ℕ) (s s' C f o δ : ℝ), 0 < s → 0 < s' → (n : ℝ) * s = n' * s' → 0 < C → 0 ≤ f → 0 ≤ o → 0 < δ →
    δ ≤ 1 → ∀ L' : Law n', MeetsTarget L' (fun _ => s') C δ → ∀ ε > 0, ∃ L : Law n, MeetsTarget L (fun _ => s) C δ ∧
      unitCost f o s' * ((1 - s' / C - C / (n' * s') - s' * Real.log δ⁻¹ / (2 * C)) *
          cost L fun _ => unitCost f o s) ≤
        unitCost f o s * cost L' (fun _ => unitCost f o s') + ε

/-- **A heavy unit is drawn almost surely**: under a law meeting the free-compute target, a unit doing work at least
`C` is drawn with probability at least `1 − δ`. -/
def HeavyUnit : Prop :=
  ∀ (n : ℕ) (w : Fin n → ℝ) (C δ : ℝ) (L : Law n), MeetsTarget L w C δ → ∀ u, C ≤ w u → 1 - δ ≤ (L.incl u).toReal

/-- **The converse for unequal units, in draws.** With every unit's work in `[a, A]`, `A · Λ ≤ C` and `ρ > 1`, every law
meeting the free-compute target draws at least `Λ (1 − AΛ/(2C)) (F/(ρC) · (1 − A/C) − J)` units in expectation, where
`F` is the total work and `J = ⌊log_ρ (A/a)⌋ + 1` the number of classes `[aρ^j, aρ^{j+1})`. -/
def UnequalDraws : Prop :=
  ∀ (n : ℕ) (w : Fin n → ℝ) (ρ a A C δ : ℝ), 1 < ρ → 0 < a → (∀ u, a ≤ w u ∧ w u ≤ A) → 0 < C → 0 < δ → δ ≤ 1 →
    A * Real.log δ⁻¹ ≤ C → ∀ L : Law n, MeetsTarget L w C δ →
      Real.log δ⁻¹ * (1 - A * Real.log δ⁻¹ / (2 * C)) *
          ((∑ u, w u) / (ρ * C) * (1 - A / C) - (⌊Real.logb ρ (A / a)⌋₊ + 1)) ≤ meanDraws L

/-- **The converse for unequal units, in cost**, with unit `u` costing `c(w u) = f + o · w u`: every law meeting the
target pays at least `Λ (1 − AΛ/(2C)) (∑_u w u · c(w u)/(ρ²C) · (1 − A/C) − J · c(A))`, against `Achievability`'s
`Λ/C · ∑_u w u · c(w u)`. Any unit implies `a ≤ A`; with no units the bounds on `w` say nothing, and without `a ≤ A` a
negative `A` makes `c(A)` negative and the bound positive while every law costs `0`. -/
def UnequalCost : Prop :=
  ∀ (n : ℕ) (w : Fin n → ℝ) (ρ a A C δ f o : ℝ), 1 < ρ → 0 < a → a ≤ A → (∀ u, a ≤ w u ∧ w u ≤ A) → 0 < C →
    0 < δ → δ ≤ 1 → 0 ≤ f → 0 ≤ o → A * Real.log δ⁻¹ ≤ C → ∀ L : Law n, MeetsTarget L w C δ →
      Real.log δ⁻¹ * (1 - A * Real.log δ⁻¹ / (2 * C)) *
          ((∑ u, w u * unitCost f o (w u)) / (ρ ^ 2 * C) * (1 - A / C) -
            (⌊Real.logb ρ (A / a)⌋₊ + 1) * unitCost f o A) ≤ cost L fun u => unitCost f o (w u)

/-- **The exfiltration optimum, for the bound.** With every unit `b` bits wide, let `K` be the most wrong units the
target `ℓ` allows: `b·K + locBits n K ≤ ℓ < b·(K+1) + locBits n (K+1)`. Every law whose bound certifies `ℓ` draws at
least `(n − K)(1 − δ^{1/(K+1)})` units in expectation, and some law certifying `ℓ` draws at most `n · Λ/(K+1) + ε`. -/
def ExfiltrationOptimum : Prop :=
  ∀ (n b K : ℕ) (δ ℓ : ℝ), 0 < δ → δ ≤ 1 → (b : ℝ) * K + locBits n K ≤ ℓ →
    ℓ < (b : ℝ) * (K + 1) + locBits n (K + 1) →
    (∀ L : Law n, CertifiesLeak L (fun _ => b) δ ℓ → ((n : ℝ) - K) * (1 - δ ^ ((K + 1 : ℝ)⁻¹)) ≤ meanDraws L) ∧
    ∀ ε > 0, ∃ L : Law n, CertifiesLeak L (fun _ => b) δ ℓ ∧ meanDraws L ≤ n * Real.log δ⁻¹ / (K + 1) + ε

/-- **The refinement rule, for the bound.** Split each of `n` units of work `s` and width `b` into `r` units of work
`s/r` and width `b'`, with `K` as in `ExfiltrationOptimum` and `2(K + 1) ≤ n`. If `b + log₂(K+2)/(K+1) ≤ b' + log₂ r`,
the unsplit partition certifies `ℓ` at cost `c(s) · n · Λ/(K+1) + ε`, and every law on the split partition certifying
`ℓ` costs at least `(1 − K/n)(1 − Λ/(2(K+1)))` times that: splitting pays only when it cuts the width by more than
`log₂ r − log₂(K+2)/(K+1)` bits. -/
def RefinementRule : Prop :=
  ∀ (n r b b' K : ℕ) (s f o δ ℓ : ℝ), 1 ≤ r → 0 < s → 0 ≤ f → 0 ≤ o → 0 < δ → δ ≤ 1 → 2 * (K + 1) ≤ n →
    (b : ℝ) * K + locBits n K ≤ ℓ → ℓ < (b : ℝ) * (K + 1) + locBits n (K + 1) →
    (b : ℝ) + Real.logb 2 (K + 2) / (K + 1) ≤ b' + Real.logb 2 r →
    (∀ ε > 0, ∃ L : Law n, CertifiesLeak L (fun _ => b) δ ℓ ∧
      cost L (fun _ => unitCost f o s) ≤ unitCost f o s * (n * Real.log δ⁻¹ / (K + 1)) + ε) ∧
    ∀ L' : Law (r * n), CertifiesLeak L' (fun _ => b') δ ℓ →
      (1 - K / n) * (1 - Real.log δ⁻¹ / (2 * (K + 1))) * (unitCost f o s * (n * Real.log δ⁻¹ / (K + 1))) ≤
        cost L' fun _ => unitCost f o (s / r)

/-- **The detection risk is concave** in the number of wrong proof units, for `k ≥ 2` and `p ≤ 1/2`. -/
def RiskConcave : Prop :=
  ∀ (N k m : ℕ) (p : ℝ), 2 ≤ k → 0 ≤ p → p ≤ 1 / 2 →
    detectionRisk N k p (m + 2) - detectionRisk N k p (m + 1) ≤ detectionRisk N k p (m + 1) - detectionRisk N k p m

/-- **Concentration.** For `k ≥ 2` and `p ≤ 1/2`, wrong proof units spread over `R` replay units of `N`, `m i` in
replay unit `i`, have a total detection risk at least that of the same number `M = ∑_i m i` packed into `⌊M/N⌋` full
replay units and one holding `M mod N`. -/
def Concentration : Prop :=
  ∀ (N k R : ℕ) (p : ℝ), 2 ≤ k → k ≤ N → 0 ≤ p → p ≤ 1 / 2 → ∀ m : Fin R → ℕ, (∀ i, m i ≤ N) →
    (((∑ i, m i) / N : ℕ) : ℝ) * detectionRisk N k p N + detectionRisk N k p ((∑ i, m i) % N) ≤
      ∑ i, detectionRisk N k p (m i)

/-- **The nested law's escape.** Under `nested R N k a b`, with `p = a/b ≤ 1`, a set `B` of proof units with
`m_i = wrongIn B i` of them in replay unit `i` escapes with probability `∏_i (1 − p + p · C(N − m_i, k)/C(N, k))`: each
replay unit escapes its two stages independently of the others. For `p < 1` that is `exp(−∑_i ψ(m_i))`, with
`ψ = detectionRisk N k p`.

This is variant (a) of `Audit.TwoStage`: every commitment, the replay units' roots and every proof unit's, comes before
the one nested draw, so `nested` is a one-stage law over the proof units. Under variant (b), where the prover commits a
drawn replay unit's interior after the first stage, the escape is `effEscape` over the replay units, not `ψ`. -/
def NestedEscape : Prop :=
  ∀ (R N k a b : ℕ) [NeZero b] (hk : k ≤ N), a ≤ b → ∀ B : Finset (Fin (R * N)),
    ((nested R N k a b hk).escape B).toReal =
        ∏ i, (1 - (a / b : ℝ) + (a / b : ℝ) * ((N - wrongIn B i).choose k : ℝ) / (N.choose k : ℝ)) ∧
      (a < b → ((nested R N k a b hk).escape B).toReal =
        Real.exp (-∑ i, detectionRisk N k (a / b) (wrongIn B i)))

end FlockSoundness.Audit.Partitioning.Guarantees
