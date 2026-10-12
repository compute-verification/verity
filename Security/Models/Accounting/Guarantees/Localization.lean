import Definitions.Accounting.Confidence

/-!
# Localization (trusted)

Section 6 of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.3), over strategies: with every value that
doesn't depend on the challenge held fixed, a strategy `S` maps the challenge's values `Ω` to runs. When what the rest
`Y` of the prover can tell at the window's start (`history`) and what it hears during `W = [t₀, t₁)` from outside `X`
and itself (the verifier's messages) don't depend on the challenge, what it does before each tick of `W` (its gates,
what it holds, and its sends to `X`, those still in flight at `t₀` too) is a prefix-causal table keyed by what reached
it from `X` during `W`, values and ticks; and when that takes at most `2^e` values, so does what the rest does. `X`
with that table, fixed before the challenge, is the prover of the protocol's model: what memory accounting's B2 and B3
use. It takes causality from every tick (`CausalFrom`), the one assumption it uses. The proof is in
`Security/Proofs/Accounting/Actual/Localization.lean`.
-/

namespace Accounting.Guarantees

/-- **Localization.** For a strategy `S` causal from every tick, a location `X`, the rest `Y` of the prover apart
from it and `W = [t₀, t₁)`: if what the rest can tell at `t₀` and what it reads during `W` from outside `X` and itself
don't depend on the coins, then for every `u ≤ t₁` what the rest does before `u` is a function `T` of what it read
from `X` during `[t₀, u)`, and at most `2^e` values of what it reads from `X` during `W` give at most `2^e` of what it
does before `t₁`. -/
def Localization : Prop :=
  ∀ {L Ω : Type*} (S : Ω → Actual.Run L) (X Y : Finset L) (t₀ t₁ : ℕ), Disjoint X Y → CausalFrom S →
    (∀ r r', (S r).history Y t₀ = (S r').history Y t₀) →
    (∀ r r', (S r).crossing (↑X ∪ ↑Y)ᶜ ↑Y (Set.Ico t₀ t₁) = (S r').crossing (↑X ∪ ↑Y)ᶜ ↑Y (Set.Ico t₀ t₁)) →
    (∀ u ≤ t₁, ∃ T : Set (ℕ × L × ℕ × L × ℕ × ℕ) →
        Set (ℕ × L × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ),
      ∀ r, (S r).acts ↑Y (Set.Iio u) = T ((S r).crossing ↑X ↑Y (Set.Ico t₀ u))) ∧
    ∀ e : ℕ, (Set.range fun r => (S r).crossing ↑X ↑Y (Set.Ico t₀ t₁)).encard ≤ ((2 ^ e : ℕ) : ℕ∞) →
      (Set.range fun r => (S r).acts ↑Y (Set.Iio t₁)).encard ≤ ((2 ^ e : ℕ) : ℕ∞)

end Accounting.Guarantees
