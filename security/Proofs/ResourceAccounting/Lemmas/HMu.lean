module

public import Definitions.Pouw.Guarantees.HMu

/-! Steps of `Definitions.Pouw.Guarantees.HMu`'s proofs, each a `def G : Prop` proved by the `SecurityProofs` theorem
of the same name and pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.HMu

open ResourceAccounting.HMu Pouw.RelativeSecurity

def LemmaB : Prop :=
  ∀ (Ω O α β : Type) [Fintype Ω] (R : SkipRound Ω O α β) (μ : ℝ) (Z : ℕ), R.HMuPlus μ Z →
    prob R.lucky ≤ Z * μ * prob (fun ω => 0 < R.d (R.view ω))

def PersistentOneWord : Prop :=
  ∀ (S W : Type) [AddCommGroup W] (step : ℕ → S → W → S) (s₀ : S) (w : ℕ → W) (P : Finset ℕ) (i c : ℕ),
    (∀ n x, Function.Injective (fun s => step n s x)) → (∀ n s, Function.Injective (step n s)) →
    i ∈ P → i < c → (∀ n ∈ P, n ≠ i → c ≤ n) →
    Function.Injective (fun X : W => chain step s₀ (persist w P X) c)

end ResourceAccounting.Guarantees.HMu
