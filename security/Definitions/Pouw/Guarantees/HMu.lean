module

public import Definitions.Pouw.HMu

/-! (H-μ⁺) from the coins, Lemma B, the point mass of fresh noise and Lemma B′; the view's independence from the fresh bits stays a premise. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.HMu

open ResourceAccounting.HMu Pouw.RelativeSecurity

/-- **(H-μ) from fresh coins (Erdős's Littlewood–Offord bound).** On coins `Ω₀ × (J → Bool)`, uniform, a skipped
value `a ω₀ + Σ_j [e_j]·v_j ω₀` whose `u = |J|` weights are nonzero integers takes any value `t` with probability at
most `C(u, ⌊u/2⌋)/2^u` given any view `V` that is a function of `Ω₀`. -/
def CoinPointMass : Prop :=
  ∀ (Ω₀ O J : Type) [Fintype Ω₀] [Fintype J] [DecidableEq J] (V : Ω₀ → O) (a : Ω₀ → ℤ) (v : Ω₀ → J → ℤ) (o : O)
    (t : ℤ),
    prob (fun ω : Ω₀ × (J → Bool) => V ω.1 = o ∧ (∀ j, v ω.1 j ≠ 0) ∧ a ω.1 + coinSum (v ω.1) ω.2 = t) ≤
      ((Fintype.card J).choose (Fintype.card J / 2) : ℝ) / 2 ^ Fintype.card J *
        prob (fun ω : Ω₀ × (J → Bool) => V ω.1 = o)

/-- **(H-μ) from one fresh coin, in any additive group.** On coins `Ω₀ × (J → Bool)`, uniform, a skipped value
`a ω₀ + Σ_j [e_j]·v_j ω₀` with some nonzero weight takes any value `t` with probability at most `1/2` given any view
`V` that is a function of `Ω₀`: flipping a bit of nonzero weight moves it. -/
def CoinPointMassHalf : Prop :=
  ∀ (Ω₀ O J G : Type) [Fintype Ω₀] [Fintype J] [DecidableEq J] [AddCommGroup G] (V : Ω₀ → O) (a : Ω₀ → G)
    (v : Ω₀ → J → G) (o : O) (t : G),
    prob (fun ω : Ω₀ × (J → Bool) => V ω.1 = o ∧ (∃ j, v ω.1 j ≠ 0) ∧ a ω.1 + coinSum (v ω.1) ω.2 = t) ≤
      1 / 2 * prob (fun ω : Ω₀ × (J → Bool) => V ω.1 = o)

/-- **Receipt luck from the coins.** A `CoinRound` with integer skipped values is lucky, where its honest receipt has
fibre at most `Z` and its `u = |J|` weights are nonzero, with probability at most `Z·C(u, ⌊u/2⌋)/2^u` times its
probability of being deficient. -/
def ReceiptLuckOfCoins : Prop :=
  ∀ (Ω₀ J β : Type) [Fintype Ω₀] [Fintype J] [DecidableEq J] (R : CoinRound Ω₀ J ℤ β) (Z : ℕ),
    prob (fun ω : Ω₀ × (J → Bool) => R.Inj Z ω.1 ∧ (∀ j, R.weight ω.1 j ≠ 0) ∧ R.lucky ω) ≤
      Z * (((Fintype.card J).choose (Fintype.card J / 2) : ℝ) / 2 ^ Fintype.card J) *
        prob (fun ω : Ω₀ × (J → Bool) => 0 < R.d ω.1)

/-- **Receipt luck from one coin, in any additive group.** A `CoinRound` is lucky, where its honest receipt has fibre
at most `Z` and some weight is nonzero, with probability at most `Z/2` times its probability of being deficient. -/
def ReceiptLuckOfCoin : Prop :=
  ∀ (Ω₀ J G β : Type) [Fintype Ω₀] [Fintype J] [DecidableEq J] [AddCommGroup G] (R : CoinRound Ω₀ J G β)
    (Z : ℕ),
    prob (fun ω : Ω₀ × (J → Bool) => R.Inj Z ω.1 ∧ (∃ j, R.weight ω.1 j ≠ 0) ∧ R.lucky ω) ≤
      Z * (1 / 2) * prob (fun ω : Ω₀ × (J → Bool) => 0 < R.d ω.1)

end ResourceAccounting.Guarantees.HMu
