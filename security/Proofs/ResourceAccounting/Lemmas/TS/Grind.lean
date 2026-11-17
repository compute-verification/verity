module

public import Definitions.Pouw.RelativeSecurity

/-! Steps of proofs that price the grinding premium under hashed seeds, pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open Pouw.RelativeSecurity

def ExchangeTail : Prop :=
  ∀ {B : Type} [Fintype B] (f : B → ℝ) (θ M π : ℝ), 0 ≤ θ → 0 ≤ M →
    ∑ b, Real.exp (θ * f b) ≤ M * Fintype.card B → prob (fun b => π ≤ f b) ≤ M * Real.exp (-(θ * π))

def GrindTail : Prop :=
  ∀ {Ω₀ B : Type} [Fintype Ω₀] [Fintype B] {T : ℕ} (Bad : Fin T → Ω₀ → (Fin T → B) → Prop) (τ : ℝ), 0 ≤ τ →
    (∀ q ω₀ a, prob (fun b : B => Bad q ω₀ (Function.update a q b)) ≤ τ) →
    prob (fun ω : Ω₀ × (Fin T → B) => ∃ q, Bad q ω.1 ω.2) ≤ T * τ

def GrindPremium : Prop :=
  ∀ {Ω₀ B : Type} [Fintype Ω₀] [Fintype B] {T : ℕ} (g R : Fin T → Ω₀ → (Fin T → B) → ℝ)
    (Break : Ω₀ × (Fin T → B) → Prop) (θ M π δb : ℝ), 0 ≤ θ → 0 ≤ M →
    (∀ q ω₀ a, ∑ b : B, Real.exp (θ * g q ω₀ (Function.update a q b)) ≤ M * Fintype.card B) →
    prob Break ≤ δb → (∀ ω q, ¬Break ω → R q ω.1 ω.2 = g q ω.1 ω.2) →
    prob (fun ω : Ω₀ × (Fin T → B) => ∃ q, π ≤ R q ω.1 ω.2) ≤ T * M * Real.exp (-(θ * π)) + δb

def GrindPremiumAt : Prop :=
  ∀ {Ω₀ B : Type} [Fintype Ω₀] [Fintype B] {T : ℕ} (g R : Fin T → Ω₀ → (Fin T → B) → ℝ)
    (Break : Ω₀ × (Fin T → B) → Prop) (θ M δ δb : ℝ), 0 < θ → 0 < M → 0 < δ →
    (∀ q ω₀ a, ∑ b : B, Real.exp (θ * g q ω₀ (Function.update a q b)) ≤ M * Fintype.card B) →
    prob Break ≤ δb → (∀ ω q, ¬Break ω → R q ω.1 ω.2 = g q ω.1 ω.2) →
    prob (fun ω : Ω₀ × (Fin T → B) => ∃ q, Real.log (T * M / δ) / θ ≤ R q ω.1 ω.2) ≤ δ + δb

def GrindMass : Prop :=
  ∀ {Ω₀ B : Type} [Fintype Ω₀] [Fintype B] {T : ℕ} (g : Fin T → Ω₀ → (Fin T → B) → ℝ) (θ M : ℝ),
    (∀ q ω₀ a, ∑ b : B, Real.exp (θ * g q ω₀ (Function.update a q b)) ≤ M * Fintype.card B) →
    ∑ ω : Ω₀ × (Fin T → B), ∑ q, Real.exp (θ * g q ω.1 ω.2) ≤ T * M * Fintype.card (Ω₀ × (Fin T → B))

open Classical in
def LuckMoment : Prop :=
  ∀ {Ω : Type} [Fintype Ω] (lucky : Ω → Prop) (p w θ κ : ℝ), 0 ≤ w → 0 ≤ θ → 0 ≤ κ → prob lucky ≤ p →
    ∑ ω, Real.exp (θ * (w * ((1 + κ) * (if lucky ω then 1 else 0) - κ))) ≤
      (p * Real.exp (θ * w) + (1 - p) * Real.exp (-(κ * θ * w))) * Fintype.card Ω

def PrivateMoment : Prop :=
  ∀ {R ι : Type} {E : ι → Type} [Fintype R] [Fintype ι] [DecidableEq ι] [∀ i, Fintype (E i)]
    (g : ∀ i, R → E i → ℝ) (θ : ℝ) (M : ι → ℝ),
    (∀ i r, ∑ x, Real.exp (θ * g i r x) ≤ M i * Fintype.card (E i)) →
    ∑ ω : R × (∀ i, E i), Real.exp (θ * ∑ i, g i ω.1 (ω.2 i)) ≤ (∏ i, M i) * Fintype.card (R × (∀ i, E i))

open Classical in
def TileLuck : Prop :=
  ∀ {Ω₀ E : Type} [Fintype Ω₀] [Fintype E] {n r : ℕ} (w : Fin n → Fin r → ℝ) (dead : Fin n → Fin r → Ω₀ → Prop)
    (lucky : Fin n → Fin r → Ω₀ → E → Prop) (p q : Fin n → Fin r → ℝ) (θ κ M : ℝ),
    0 < n → 0 < r → 0 ≤ θ → 0 ≤ κ → (∀ k i, 0 ≤ w k i) → (∀ k i, 0 ≤ p k i ∧ p k i ≤ 1) →
    (∀ k i ω₀, ¬dead k i ω₀ → prob (lucky k i ω₀) ≤ p k i) → (∀ k i, prob (dead k i) ≤ q k i) →
    (∀ k i, q k i * Real.exp (n * (r * θ * w k i)) +
        (1 - q k i) * (p k i * Real.exp (r * θ * w k i) + (1 - p k i) * Real.exp (-(κ * (r * θ) * w k i))) ^ n ≤ M) →
    ∑ ω : Ω₀ × (Fin n → E),
        Real.exp (θ * ∑ k, ∑ i, w k i * ((1 + κ) * (if lucky k i ω.1 (ω.2 k) then 1 else 0) - κ)) ≤
      M * Fintype.card (Ω₀ × (Fin n → E))

def TemplateCondition : Prop :=
  ∀ w : ℕ, w ≤ 32 →
    (1 / 2 : ℝ) ^ w * Real.exp (32 * (1 / 256 * w)) +
      (1 - (1 / 2 : ℝ) ^ w) * (1 / 2 * Real.exp (1 / 256 * w) + (1 - 1 / 2) * Real.exp (-(4 * (1 / 256) * w))) ^ 32
        ≤ 1

open Classical in
def TemplateMoment : Prop :=
  ∀ {Ω₀ E : Type} [Fintype Ω₀] [Fintype E] {r : ℕ} (w : Fin 32 → Fin r → ℕ) (dead : Fin 32 → Fin r → Ω₀ → Prop)
    (lucky : Fin 32 → Fin r → Ω₀ → E → Prop), 0 < r → (∀ k i, w k i ≤ 32) →
    (∀ k i ω₀, ¬dead k i ω₀ → prob (lucky k i ω₀) ≤ 1 / 2) → (∀ k i, prob (dead k i) ≤ (1 / 2) ^ w k i) →
    ∑ ω : Ω₀ × (Fin 32 → E),
        Real.exp (1 / (256 * r) * ∑ k, ∑ i, (w k i : ℝ) * ((1 + 4) * (if lucky k i ω.1 (ω.2 k) then 1 else 0) - 4)) ≤
      Fintype.card (Ω₀ × (Fin 32 → E))

end ResourceAccounting.Guarantees.TS
