module

public import Definitions.Pouw.Confidence

/-! The actual-circuit model's localization over strategies, which takes causality from every tick as its one assumption. -/

@[expose] public section

namespace ResourceAccounting.Guarantees

/-- **Localization.** For a strategy `S` causal from every tick, a location `X`, the rest `Y` of the prover apart
from it and `W = [t₀, t₁)`: if what the rest can tell at `t₀` and what it reads during `W` from outside `X` and itself
don't depend on the coins, then for every `u ≤ t₁` what the rest does before `u` is a function `T` of what it read
from `X` during `[t₀, u)`, and at most `2^e` values of what it reads from `X` during `W` give at most `2^e` of what it
does before `t₁`. -/
def Localization : Prop :=
  ∀ {L Ω : Type*} [LinearOrder L] (S : Ω → Actual.Run L) (X Y : Finset L) (t₀ t₁ : ℕ), Disjoint X Y → CausalFrom S →
    (∀ r r', (S r).history Y t₀ = (S r').history Y t₀) →
    (∀ r r', (S r).crossing (↑X ∪ ↑Y)ᶜ ↑Y (Set.Ico t₀ t₁) = (S r').crossing (↑X ∪ ↑Y)ᶜ ↑Y (Set.Ico t₀ t₁)) →
    (∀ u ≤ t₁, ∃ T : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) →
        Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ × ℕ),
      ∀ r, (S r).acts Y (Set.Iio u) = T ((S r).crossing ↑X ↑Y (Set.Ico t₀ u))) ∧
    ∀ e : ℕ, (Set.range fun r => (S r).crossing ↑X ↑Y (Set.Ico t₀ t₁)).encard ≤ ((2 ^ e : ℕ) : ℕ∞) →
      (Set.range fun r => (S r).acts Y (Set.Iio t₁)).encard ≤ ((2 ^ e : ℕ) : ℕ∞)

/-- **Recall** (B2's count): in a strategy `S` causal from every tick, if what crosses into a location `X` during
`[t, u)` doesn't depend on the coins, then what `X` does during `[t, u)` is a function `T` of its state's content at
`t`, a string of at most `mem X t` bits. -/
def Recall : Prop :=
  ∀ {L Ω : Type*} [LinearOrder L] (S : Ω → Actual.Run L) (X : Finset L) (t u : ℕ), CausalFrom S →
    (∀ r r', (S r).crossing (↑X)ᶜ ↑X (Set.Ico t u) = (S r').crossing (↑X)ᶜ ↑X (Set.Ico t u)) →
    ∃ T : List Bool → Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ × ℕ),
      ∀ r, (S r).acts X (Set.Ico t u) = T ((S r).content X t) ∧ ((S r).content X t).length ≤ (S r).mem X t

end ResourceAccounting.Guarantees
