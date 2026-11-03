import Definitions.Accounting.HMu

/-!
# (H-μ⁺) from the coins: Lemma B, the point mass of fresh noise, and Lemma B′ for persistent patterns (trusted)

The statements, over `Definitions/Accounting/HMu.lean`. The proofs are in `Security/Proofs/Accounting/HMu/`.

* `LemmaB`: under (H-μ⁺) at `(μ, Z)` (`SkipRound.HMuPlus`), a round is lucky with probability at most `Z·μ` times its
  probability of being deficient. This is (L)'s per-round form (`Accounting.TS.RoundGame.LuckAt`) at `p = Z·μ`, for a
  check that passes only when the committed receipt is the honest one.
* `CoinPointMass`: a skipped value `a + Σ_j [e_j]·v_j` with `u` fresh uniform bits `e` and nonzero integer weights has
  probability at most `C(u, ⌊u/2⌋)/2^u` at any value, given any view that doesn't read `e` (Erdős's Littlewood–Offord
  bound): `1/2` at `u = 1`, `0.140` at `u = 32`.
* `CoinPointMassHalf`: in any additive group (`Z/2³²` included), one nonzero weight gives at most `1/2`.
* `ReceiptLuckOfCoins` and `ReceiptLuckOfCoin`: the two together with Lemma B, on a `CoinRound`: its luck is at most
  `Z·C(u, ⌊u/2⌋)/2^u`, or `Z/2`, times its probability of being deficient.
* `PersistentOneWord`: Lemma B′ for a persistent pattern, one word a unit. For a fold whose step is injective in the
  state and in the word, if the pattern's first word is its only word before a committed checkpoint, the checkpoint's
  state is injective in the skipped value: (Inj) with `Z = 1`, and no differential term.

What the coins can't give, and so what stays a premise:

* The view's independence from the fresh bits is in `CoinRound`'s types. Under the one-σ order the prover holds σ
  before it evaluates. Every coin of a skipped product's noise is also read by the honest computation of each other
  word that shares it: a dither bit of a row's coordinate enters that row segment's word in every column, and a column's
  `U` every row's. So a cheater that computes those words has a view that determines the skipped value, whose point
  mass is then `1`. There (H-μ) is computational: the cheater hasn't computed the product. No distribution of coins
  supplies that.
* The skip decision must be in the view, with no leak: a cheater that learns one bit about the fresh coins can skip
  exactly when that bit pins the skipped value, and is then lucky on every deficient round. So a leak of `ℓ` bits costs
  a factor `2^ℓ` on the absolute luck, but no bound relative to the deficiency survives.
* The fold's keys are drawn with σ, so a bound may not average over the start state. For a pattern with two or more
  words in the first unit it reaches (two copies of one entry in a unit, a rank-1 skip across a unit's words, the served
  `y ⊕ rotl(z, 16)` projection), Lemma B′ needs a fibre bound at the start state the cheater knows. (Inj) alone gives
  none, and an all-state no-kernel statement isn't one.
-/

namespace Accounting.Guarantees.HMu

open Accounting.HMu Pouw.RelativeSecurity

/-- **Lemma B (blind luck).** Under (H-μ⁺) at `(μ, Z)`, a round is lucky (deficient, with the honest receipt
committed) with probability at most `Z·μ` times its probability of being deficient. -/
def LemmaB : Prop :=
  ∀ (Ω O α β : Type) [Fintype Ω] (R : SkipRound Ω O α β) (μ : ℝ) (Z : ℕ), R.HMuPlus μ Z →
    prob R.lucky ≤ Z * μ * prob (fun ω => 0 < R.d (R.view ω))

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

/-- **Lemma B′ for a persistent pattern, one word a unit.** For a fold whose step is injective in the state for every
word and in the word for every state, a pattern `P` whose first position `i` is its only position before the checkpoint
`c > i`: the state at `c` of the words with `X` added at every position of `P` determines `X`. -/
def PersistentOneWord : Prop :=
  ∀ (S W : Type) [AddCommGroup W] (step : ℕ → S → W → S) (s₀ : S) (w : ℕ → W) (P : Finset ℕ) (i c : ℕ),
    (∀ n x, Function.Injective (fun s => step n s x)) → (∀ n s, Function.Injective (step n s)) →
    i ∈ P → i < c → (∀ n ∈ P, n ≠ i → c ≤ n) →
    Function.Injective (fun X : W => chain step s₀ (persist w P X) c)

end Accounting.Guarantees.HMu
