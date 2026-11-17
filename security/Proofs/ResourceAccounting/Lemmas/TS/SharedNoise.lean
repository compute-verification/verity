module

public import Definitions.Pouw.TS.SharedNoise

/-! Steps of proofs for H1 + fx1's noise derived from one seed a call (`Definitions.Pouw.TS.SharedNoise`), pinned by
no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS Pouw.RelativeSecurity

def CondReceiptLuckMarginal : Prop :=
  ∀ {Ω Z O : Type} [Fintype Ω] [Fintype Z] [Fintype O] (e : Ω ≃ Z × O) (pr : ℝ) {n : ℕ} (G : ReceiptGame Ω n),
    G.CondReceiptLuck e pr → G.ReceiptLuck pr

open Classical in
/-- The order of the columns is any fixed one: relabel them. -/
def GridLuck : Prop :=
  ∀ {Rw Cl : Type} [Fintype Rw] [Fintype Cl] {r n : ℕ} (w : Fin r → Fin n → ℝ) (dead : Fin r → Fin n → Rw → Prop)
    (lucky : Fin r → Fin n → (Fin r → Rw) → (Fin n → Cl) → Prop) (p q : Fin r → Fin n → ℝ) (θ κ : ℝ),
    0 < n → 0 < r → 0 ≤ θ → 0 ≤ κ → (∀ i j, 0 ≤ w i j) → (∀ i j, 0 ≤ p i j ∧ p i j ≤ 1) →
    (∀ i j ρ γ γ', (∀ k, k ≤ j → γ k = γ' k) → (lucky i j ρ γ ↔ lucky i j ρ γ')) →
    (∀ i j ρ γ, ¬dead i j (ρ i) → prob (fun c => lucky i j ρ (Function.update γ j c)) ≤ p i j) →
    (∀ i j, prob (dead i j) ≤ q i j) →
    (∀ i j, q i j * Real.exp (n * θ * w i j) +
        (1 - q i j) * (p i j * Real.exp (r * θ * w i j) + (1 - p i j) * Real.exp (-(κ * (r * θ) * w i j))) ^
          ((n : ℝ) / r) ≤ 1) →
    ∑ ω : (Fin r → Rw) × (Fin n → Cl),
        Real.exp (θ * ∑ i, ∑ j, w i j * ((1 + κ) * (if lucky i j ω.1 ω.2 then 1 else 0) - κ)) ≤
      Fintype.card ((Fin r → Rw) × (Fin n → Cl))

def GridTemplateCondition : Prop :=
  ∀ e : ℝ, 0 < e → ∀ w : ℕ, w ≤ 32 →
    (1 / 2 : ℝ) ^ w * Real.exp (e * (1 / (4 * max 64 e)) * w) +
        (1 - (1 / 2 : ℝ) ^ w) *
          (1 / 2 * Real.exp (1 / (4 * max 64 e) * w) + (1 - 1 / 2) * Real.exp (-(4 * (1 / (4 * max 64 e)) * w))) ^ e ≤
      1

open Classical in
def GridTemplateMoment : Prop :=
  ∀ {Rw Cl : Type} [Fintype Rw] [Fintype Cl] {r n : ℕ} (w : Fin r → Fin n → ℕ) (dead : Fin r → Fin n → Rw → Prop)
    (lucky : Fin r → Fin n → (Fin r → Rw) → (Fin n → Cl) → Prop), 0 < r → 0 < n → (∀ i j, w i j ≤ 32) →
    (∀ i j ρ γ γ', (∀ k, k ≤ j → γ k = γ' k) → (lucky i j ρ γ ↔ lucky i j ρ γ')) →
    (∀ i j ρ γ, ¬dead i j (ρ i) → prob (fun c => lucky i j ρ (Function.update γ j c)) ≤ 1 / 2) →
    (∀ i j, prob (dead i j) ≤ (1 / 2) ^ w i j) →
    ∑ ω : (Fin r → Rw) × (Fin n → Cl),
        Real.exp (1 / (4 * max (64 * (r : ℝ)) n) *
          ∑ i, ∑ j, (w i j : ℝ) * ((1 + 4) * (if lucky i j ω.1 ω.2 then 1 else 0) - 4)) ≤
      Fintype.card ((Fin r → Rw) × (Fin n → Cl))

end ResourceAccounting.Guarantees.TS
