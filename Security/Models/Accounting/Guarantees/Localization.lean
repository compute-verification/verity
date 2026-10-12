import Definitions.Accounting.Confidence

/-!
# Localization (trusted)

Section 6 of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.2), over strategies: with every value that
doesn't depend on the challenge held fixed, a strategy `S` maps the challenge's values `Ω` to runs. When the rest `Y` of
the prover starts the window `W = [t₀, t₁)` in the same state and hears the same from outside `X` and itself (the
verifier's messages) whatever the challenge, what it does during `W`, its sends to `X` among them, is a prefix-causal
table keyed by what `X` sent it; and when `X`'s sends take at most `2^e` values, so do the rest's doings. `X` with that
table, fixed before the challenge, is the prover of the protocol's model: what memory accounting's B2 and B3 use. It
takes causality from every tick (`CausalFrom`), the one assumption it uses. The proof is in
`Security/Proofs/Accounting/Actual/Localization.lean`.
-/

namespace Accounting.Guarantees

/-- **Localization.** For a strategy `S`, a location `X`, the rest `Y` of the prover apart from it and `W = [t₀, t₁)`:
under causality from every tick, if the rest's state over continuations at `t₀` and what it reads during `W` from
outside `X` and itself don't depend on the coins, then for every `u ≤ t₁` what the rest does during `[t₀, u)` is a
function `T` of what `X` sent it during `[t₀, u)`, and at most `2^e` values of what `X` sends it during `W` give at most
`2^e` of what it does. -/
def Localization : Prop :=
  ∀ {L Ω : Type*} (S : Ω → Actual.Run L) (X Y : Finset L) (t₀ t₁ : ℕ), Disjoint X Y → CausalFrom S →
    (∀ r r', (S r).seen '' ↑(stateOver S Y t₀ r) = (S r').seen '' ↑(stateOver S Y t₀ r')) →
    (∀ r r', (S r).crossing (↑X ∪ ↑Y)ᶜ ↑Y (Set.Ico t₀ t₁) = (S r').crossing (↑X ∪ ↑Y)ᶜ ↑Y (Set.Ico t₀ t₁)) →
    (∀ u ≤ t₁, ∃ T : Set (ℕ × L × ℕ × L × ℕ × ℕ) → Set (ℕ × L × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ),
      ∀ r, (S r).acts ↑Y (Set.Ico t₀ u) = T ((S r).crossing ↑X ↑Y (Set.Ico t₀ u))) ∧
    ∀ e : ℕ, (Set.range fun r => (S r).crossing ↑X ↑Y (Set.Ico t₀ t₁)).encard ≤ ((2 ^ e : ℕ) : ℕ∞) →
      (Set.range fun r => (S r).acts ↑Y (Set.Ico t₀ t₁)).encard ≤ ((2 ^ e : ℕ) : ℕ∞)

end Accounting.Guarantees
