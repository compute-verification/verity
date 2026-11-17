module

public import Definitions.Pouw.Assumptions.TS
public import Definitions.Pouw.TS.Steer

/-! Steps of proofs on steering under one σ, pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Assumptions.TS

open ResourceAccounting.TS

/-- LIN-STEER, which is false (`LinSteerFalse`): zeroing a fraction `φ = |Z|/|act|` of the row's noise-active
coordinates saves at most `φ` of the consumer's work. -/
def LinSteer {m : ℕ} (c : Consumer m) : Prop :=
  ∀ Z ⊆ c.act, c.cost ∅ - c.cost Z ≤ (Z.card : ℝ) / c.act.card * c.cost ∅

end ResourceAccounting.Assumptions.TS

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity

def LinSteerSegments : Prop :=
  ∀ {m G : ℕ} (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ), 0 < c →
    (LinSteer (segConsumer seg act c) ↔
      ∀ g g', 0 < segCount seg act g → 0 < segCount seg act g' → segCount seg act g = segCount seg act g')

def LinSteerFalse : Prop :=
  ∃ (m G : ℕ) (seg : Fin m → Fin G) (act : Finset (Fin m)), ¬ LinSteer (segConsumer seg act 1)

def SegmentSteer : Prop :=
  ∀ {m G : ℕ} (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ) (a : ℕ), 0 ≤ c → 0 < a →
    (∀ g, 0 < segCount seg act g → a ≤ segCount seg act g) →
    ∀ Z ⊆ act, (segConsumer seg act c).cost ∅ - (segConsumer seg act c).cost Z ≤ (Z.card : ℝ) / a * c

def AdaptSteerFail : Prop :=
  ∀ {m G : ℕ} (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ), 0 < c →
    (segActive seg act ∅).card < act.card → ¬ LinSteer (adaptConsumer seg act c)

def SteerUnion : Prop :=
  ∀ {Ω₀ B N K : Type} [Fintype Ω₀] [Fintype B] [Fintype K] {G : ℕ} (η : Fin G → B → N)
    (v : K → Ω₀ → Fin G → N) (μ : ℝ) (n : ℕ), 0 ≤ μ → (∀ g t, prob (fun b : B => η g b = t) ≤ μ) →
    prob (fun ω : Ω₀ × (Fin G → B) => ∃ κ, ∃ S : Finset (Fin G), S.card = n ∧ ∀ g ∈ S, η g (ω.2 g) = v κ ω.1 g) ≤
      Fintype.card K * G.choose n * μ ^ n

def GrindUnion : Prop :=
  ∀ {Ω₀ B N : Type} [Fintype Ω₀] [Fintype B] {T G : ℕ} (η : Fin G → B → N)
    (v : Fin T → Ω₀ → (Fin T → Fin G → B) → Fin G → N) (μ : ℝ) (n : ℕ), 0 ≤ μ →
    (∀ g t, prob (fun b : B => η g b = t) ≤ μ) →
    (∀ q ω₀ a a', (∀ q', q' ≠ q → a q' = a' q') → v q ω₀ a = v q ω₀ a') →
    prob (fun ω : Ω₀ × (Fin T → Fin G → B) =>
        ∃ q, ∃ S : Finset (Fin G), S.card = n ∧ ∀ g ∈ S, η g (ω.2 q g) = v q ω.1 ω.2 g) ≤
      T * G.choose n * μ ^ n

end ResourceAccounting.Guarantees.TS
