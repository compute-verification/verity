module

public import Proofs.Flock.Soundness.Audit.Partitioning.Model

/-! Optimal partitioning: the guarantees, `Λ` being `ln(1/δ)`. -/

@[expose] public section

namespace FlockSoundness.Audit.Partitioning.Guarantees

open Finset

def ConverseExact : Prop :=
  ∀ (n K : ℕ) (L : SamplingDesign n) (δ : ℝ), 0 < K → K ≤ n → (L.miss K).toReal ≤ δ →
    ((n : ℝ) - K + 1) * (1 - δ ^ ((K : ℝ)⁻¹)) ≤ meanSamples L

def ConverseAsymptotic : Prop :=
  ∀ (n K : ℕ) (L : SamplingDesign n) (δ : ℝ), 0 < K → K ≤ n → 0 < δ → δ ≤ 1 → (L.miss K).toReal ≤ δ →
    ((n : ℝ) - K + 1) / K * Real.log δ⁻¹ * (1 - Real.log δ⁻¹ / (2 * K)) ≤ meanSamples L

def Achievability : Prop :=
  ∀ (n : ℕ) (w : Fin n → ℝ) (C δ ε : ℝ), (∀ u, 0 ≤ w u) → 0 < C → 0 < δ → δ ≤ 1 → 0 < ε →
    ∃ L : SamplingDesign n, MeetsTarget L w C δ ∧ ∀ u, (L.incl u).toReal ≤ min 1 (Real.log δ⁻¹ / C * w u) + ε

def GranularityOptimum : Prop :=
  ∀ (n : ℕ) (s C f o δ : ℝ), 0 < s → 0 < C → 0 ≤ f → 0 ≤ o → 0 < δ → δ ≤ 1 →
    (∀ L : SamplingDesign n, MeetsTarget L (fun _ => s) C δ →
      unitCost f o s * (n * s / C * Real.log δ⁻¹) * (1 - s / C - C / (n * s) - s * Real.log δ⁻¹ / (2 * C)) ≤
        cost L fun _ => unitCost f o s) ∧
    ∀ ε > 0, ∃ L : SamplingDesign n, MeetsTarget L (fun _ => s) C δ ∧
      cost L (fun _ => unitCost f o s) ≤ unitCost f o s * (n * s / C * Real.log δ⁻¹) + ε

/-- Stated cross-multiplied. -/
def GranularityChoice : Prop :=
  ∀ (n n' : ℕ) (s s' C f o δ : ℝ), 0 < s → 0 < s' → (n : ℝ) * s = n' * s' → 0 < C → 0 ≤ f → 0 ≤ o → 0 < δ →
    δ ≤ 1 → ∀ L' : SamplingDesign n', MeetsTarget L' (fun _ => s') C δ → ∀ ε > 0, ∃ L : SamplingDesign n, MeetsTarget L (fun _ => s) C δ ∧
      unitCost f o s' * ((1 - s' / C - C / (n' * s') - s' * Real.log δ⁻¹ / (2 * C)) *
          cost L fun _ => unitCost f o s) ≤
        unitCost f o s * cost L' (fun _ => unitCost f o s') + ε

def ServedGranularityOptimum : Prop :=
  ∀ (n : ℕ) (s C f o δ w β h : ℝ), 0 < s → 0 < C → 0 ≤ f → 0 ≤ o → 0 < δ → δ ≤ 1 →
    (∀ L : SamplingDesign n, MeetsTarget L (fun _ => s) C δ →
      unitCost f o s * (n * s / C * Real.log δ⁻¹) * (1 - s / C - C / (n * s) - s * Real.log δ⁻¹ / (2 * C)) +
          w * (n * β * h) ≤
        cost L (fun _ => unitCost f o s) + w * (n * β * h)) ∧
    ∀ ε > 0, ∃ L : SamplingDesign n, MeetsTarget L (fun _ => s) C δ ∧
      cost L (fun _ => unitCost f o s) + w * (n * β * h) ≤
        unitCost f o s * (n * s / C * Real.log δ⁻¹) + w * (n * β * h) + ε

def ServedGranularityChoice : Prop :=
  ∀ (F C f o δ w β h : ℝ), 0 < F → 0 < C → 0 < o → 0 < δ → δ < 1 → 0 ≤ w → 0 ≤ β → 0 ≤ h → ∀ s, 0 < s →
    servedCost F C f o δ w β h (Real.sqrt (w * β * h * C / (o * Real.log δ⁻¹))) ≤ servedCost F C f o δ w β h s

def HeavyUnit : Prop :=
  ∀ (n : ℕ) (w : Fin n → ℝ) (C δ : ℝ) (L : SamplingDesign n), MeetsTarget L w C δ → ∀ u, C ≤ w u → 1 - δ ≤ (L.incl u).toReal

def UnequalSamples : Prop :=
  ∀ (n : ℕ) (w : Fin n → ℝ) (ρ a A C δ : ℝ), 1 < ρ → 0 < a → (∀ u, a ≤ w u ∧ w u ≤ A) → 0 < C → 0 < δ → δ ≤ 1 →
    A * Real.log δ⁻¹ ≤ C → ∀ L : SamplingDesign n, MeetsTarget L w C δ →
      Real.log δ⁻¹ * (1 - A * Real.log δ⁻¹ / (2 * C)) *
          ((∑ u, w u) / (ρ * C) * (1 - A / C) - (⌊Real.logb ρ (A / a)⌋₊ + 1)) ≤ meanSamples L

/-- It assumes `a ≤ A`, which matters only with no units: there a negative `A` makes `c(A)` negative and the
bound positive while every sampling design costs `0`. -/
def UnequalCost : Prop :=
  ∀ (n : ℕ) (w : Fin n → ℝ) (ρ a A C δ f o : ℝ), 1 < ρ → 0 < a → a ≤ A → (∀ u, a ≤ w u ∧ w u ≤ A) → 0 < C →
    0 < δ → δ ≤ 1 → 0 ≤ f → 0 ≤ o → A * Real.log δ⁻¹ ≤ C → ∀ L : SamplingDesign n, MeetsTarget L w C δ →
      Real.log δ⁻¹ * (1 - A * Real.log δ⁻¹ / (2 * C)) *
          ((∑ u, w u * unitCost f o (w u)) / (ρ ^ 2 * C) * (1 - A / C) -
            (⌊Real.logb ρ (A / a)⌋₊ + 1) * unitCost f o A) ≤ cost L fun u => unitCost f o (w u)

/-- It compares upper bounds on the leak, not leaks an adversary achieves. -/
def ExfiltrationOptimum : Prop :=
  ∀ (n b K : ℕ) (δ ℓ : ℝ), 0 < δ → δ ≤ 1 → (b : ℝ) * K + locBits n K ≤ ℓ →
    ℓ < (b : ℝ) * (K + 1) + locBits n (K + 1) →
    (∀ L : SamplingDesign n, CertifiesLeak L (fun _ => b) δ ℓ → ((n : ℝ) - K) * (1 - δ ^ ((K + 1 : ℝ)⁻¹)) ≤ meanSamples L) ∧
    ∀ ε > 0, ∃ L : SamplingDesign n, CertifiesLeak L (fun _ => b) δ ℓ ∧ meanSamples L ≤ n * Real.log δ⁻¹ / (K + 1) + ε

/-- It compares upper bounds on the leak, not leaks an adversary achieves. -/
def RefinementRule : Prop :=
  ∀ (n r b b' K : ℕ) (s f o δ ℓ : ℝ), 1 ≤ r → 0 < s → 0 ≤ f → 0 ≤ o → 0 < δ → δ ≤ 1 → 2 * (K + 1) ≤ n →
    (b : ℝ) * K + locBits n K ≤ ℓ → ℓ < (b : ℝ) * (K + 1) + locBits n (K + 1) →
    (b : ℝ) + Real.logb 2 (K + 2) / (K + 1) ≤ b' + Real.logb 2 r →
    (∀ ε > 0, ∃ L : SamplingDesign n, CertifiesLeak L (fun _ => b) δ ℓ ∧
      cost L (fun _ => unitCost f o s) ≤ unitCost f o s * (n * Real.log δ⁻¹ / (K + 1)) + ε) ∧
    ∀ L' : SamplingDesign (r * n), CertifiesLeak L' (fun _ => b') δ ℓ →
      (1 - K / n) * (1 - Real.log δ⁻¹ / (2 * (K + 1))) * (unitCost f o s * (n * Real.log δ⁻¹ / (K + 1))) ≤
        cost L' fun _ => unitCost f o (s / r)

def RiskConcave : Prop :=
  ∀ (N k m : ℕ) (p : ℝ), 2 ≤ k → 0 ≤ p → p ≤ 1 / 2 →
    detectionRisk N k p (m + 2) - detectionRisk N k p (m + 1) ≤ detectionRisk N k p (m + 1) - detectionRisk N k p m

def Concentration : Prop :=
  ∀ (N k R : ℕ) (p : ℝ), 2 ≤ k → k ≤ N → 0 ≤ p → p ≤ 1 / 2 → ∀ m : Fin R → ℕ, (∀ i, m i ≤ N) →
    (((∑ i, m i) / N : ℕ) : ℝ) * detectionRisk N k p N + detectionRisk N k p ((∑ i, m i) % N) ≤
      ∑ i, detectionRisk N k p (m i)

/-- `nested` commits everything before its one sample, so it is a one-stage sampling design; the two-stage
design's acceptance probability is this product at the footprints only under a Bernoulli(`p`) first stage. -/
def NestedEscape : Prop :=
  ∀ (R N k a b : ℕ) [NeZero b] (hk : k ≤ N), a ≤ b → ∀ B : Finset (Fin (R * N)),
    ((nested R N k a b hk).missProb B).toReal =
        ∏ i, (1 - (a / b : ℝ) + (a / b : ℝ) * ((N - incorrectIn B i).choose k : ℝ) / (N.choose k : ℝ)) ∧
      (a < b → ((nested R N k a b hk).missProb B).toReal =
        Real.exp (-∑ i, detectionRisk N k (a / b) (incorrectIn B i)))

end FlockSoundness.Audit.Partitioning.Guarantees
