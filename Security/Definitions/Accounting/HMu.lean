import Definitions.Pouw.RelativeSecurity

/-!
# (H-μ⁺): the receipt-luck premise, and the rounds whose coins give it (trusted)

The vocabulary of `Definitions.Accounting.Guarantees.HMu`, over PoUW's uniform probability (`Pouw.RelativeSecurity.prob`).
(H-μ⁺): each skipped value's point mass is at most `μ` given the cheater's view, with Lemma B′ for correlated skips and
persistent patterns. This is that premise in the form Lemma B consumes it, and the rounds on which the noise's coins
give it.

* `SkipRound`: a round at its receipt, as Lemma B reads it.
  * `view` is what the conditioning fixes: everything the cheater's committed receipt and its skip decision are
    functions of, with the other words' skipped values. So the honest receipt is a function `honest` of the view and of
    one word's skipped value `skipped`: `X_t`, the bilinear part its committed value lacks, or, for a correlated skip
    `X = M·v`, Lemma B′'s free operands `v`.
  * The share `d` is decided from the view.
  * `lucky`: the round is deficient and its committed receipt is the honest one. With every absorbed word pinned in the
    proof, it passes only then.
* `SkipRound.PointMass μ` is (H-μ): on every deficient view, each value of the skipped value has probability at most
  `μ` given the view. `SkipRound.Inj Z` is Lemma B's (Inj) with fibre `Z`: on every deficient view, at most `Z` of the skipped
  values the view leaves possible give one honest receipt. `SkipRound.HMuPlus μ Z` is both: (H-μ⁺) as Lemma B consumes it.
* `coinSum v e = Σ_j [e_j]·v_j`: a skipped value's part on fresh coin bits (`Accounting.Tag.unreadSum`'s sum, over any
  finite index and any additive group). Every noise in play puts a skipped value in the form `base + coinSum weight e`,
  with `base` and `weight` fixed functions of committed data and the round's other coins:
  * Dither's `{0,1}` bit per operand coordinate (the weights are the other operand's entries);
  * `±1` noise (`s = 2e − 1`: weights `2v`, base shifted by `−Σ v`);
  * H1's `U` on `±16` (five planes of weights 1, 2, 4, 8 and 17, times the operand entry);
  * the cohalf Q/K dither (eight planes).
* `CoinRound`: a round on coins `Ω₀ × (J → Bool)` whose skipped value is `base + coinSum weight e`. Its cheater's
  committed receipt and share are functions of `Ω₀` alone: that it doesn't read the fresh bits `e` is in the types.
  `CoinRound.Inj Z ω₀`: at `ω₀`, at most `Z` of the skipped values the fresh bits can give have one honest receipt.
* `chain`: a fold's state after `n` words, from a start state and a step at each position. `persist w P X`: the words
  `w` with the skipped value `X` added at every position of `P`, as one skipped product shifts every later prefix word
  of its entry (Lemma B′'s persistent pattern).
-/

namespace Accounting.HMu

open Pouw.RelativeSecurity

/-- **A round at its receipt**, as Lemma B reads it. `view` is what the conditioning fixes: everything the cheater's
committed receipt and skip decision are functions of, with the other words' skipped values. `skipped` is one word's
skipped value (`X_t`, or Lemma B′'s free operands `v` for a correlated skip `X = M·v`), `d` the round's deficient
share, `committed` the cheater's receipt and `honest` the honest receipt, from the view and the skipped value. -/
structure SkipRound (Ω O α β : Type) where
  view : Ω → O
  skipped : Ω → α
  d : O → ℝ
  committed : O → β
  honest : O → α → β

namespace SkipRound

variable {Ω O α β : Type}

/-- **The round is lucky**: deficient, and its committed receipt is the honest one. With every absorbed word pinned in
the proof, a round passes only when its committed receipt is the honest one. -/
def lucky (R : SkipRound Ω O α β) (ω : Ω) : Prop :=
  R.committed (R.view ω) = R.honest (R.view ω) (R.skipped ω) ∧ 0 < R.d (R.view ω)

/-- **(H-μ) at `μ`**: on every deficient view, each value of the skipped value has probability at most `μ` given the
view. -/
def PointMass [Fintype Ω] (μ : ℝ) (R : SkipRound Ω O α β) : Prop :=
  ∀ o, 0 < R.d o → ∀ t, prob (fun ω => R.view ω = o ∧ R.skipped ω = t) ≤ μ * prob (fun ω => R.view ω = o)

/-- **(Inj) with fibre `Z`**: on every deficient view, at most `Z` of the skipped values the view leaves possible give
one honest receipt. `Z = 1` when the honest receipt is injective in the skipped value, as a fold that commits its whole
state is in each word. -/
def Inj (Z : ℕ) (R : SkipRound Ω O α β) : Prop :=
  ∀ o, 0 < R.d o → ∀ b, ∃ s : Finset α, s.card ≤ Z ∧
    ∀ ω, R.view ω = o → R.honest o (R.skipped ω) = b → R.skipped ω ∈ s

/-- **(H-μ⁺) at `(μ, Z)`**, as Lemma B consumes it: (H-μ) at `μ` and (Inj) with fibre `Z`. -/
def HMuPlus [Fintype Ω] (μ : ℝ) (Z : ℕ) (R : SkipRound Ω O α β) : Prop := R.PointMass μ ∧ R.Inj Z

end SkipRound

/-- **A skipped value's part on fresh coin bits**: `Σ_j [e_j]·v_j`. -/
def coinSum {J G : Type} [Fintype J] [AddCommMonoid G] (v : J → G) (e : J → Bool) : G :=
  ∑ j, if e j then v j else 0

/-- **A round whose skipped value is a fixed function of committed data plus fresh coins**, on coins
`Ω₀ × (J → Bool)`: the skipped value is `base ω₀ + coinSum (weight ω₀) e`. The cheater's committed receipt and its
share are functions of `Ω₀` alone, so what it commits doesn't read the fresh bits `e`. -/
structure CoinRound (Ω₀ J G β : Type) where
  base : Ω₀ → G
  weight : Ω₀ → J → G
  d : Ω₀ → ℝ
  committed : Ω₀ → β
  honest : Ω₀ → G → β

namespace CoinRound

variable {Ω₀ J G β : Type} [Fintype J] [AddCommMonoid G]

/-- The skipped value on the round's coins. -/
def skipped (R : CoinRound Ω₀ J G β) (ω : Ω₀ × (J → Bool)) : G := R.base ω.1 + coinSum (R.weight ω.1) ω.2

/-- **The round is lucky**: deficient, and its committed receipt is the honest one. -/
def lucky (R : CoinRound Ω₀ J G β) (ω : Ω₀ × (J → Bool)) : Prop :=
  R.committed ω.1 = R.honest ω.1 (R.skipped ω) ∧ 0 < R.d ω.1

/-- **(Inj) with fibre `Z` at `ω₀`**: of the skipped values the fresh bits can give at `ω₀`, at most `Z` have one
honest receipt. -/
def Inj (Z : ℕ) (R : CoinRound Ω₀ J G β) (ω₀ : Ω₀) : Prop :=
  ∀ b, ∃ s : Finset G, s.card ≤ Z ∧ ∀ e, R.honest ω₀ (R.skipped (ω₀, e)) = b → R.skipped (ω₀, e) ∈ s

end CoinRound

/-- **A fold's state after `n` words**, from the start state `s₀`, with `step n` the step at position `n`. -/
def chain {S W : Type} (step : ℕ → S → W → S) (s₀ : S) (w : ℕ → W) : ℕ → S
  | 0 => s₀
  | n + 1 => step n (chain step s₀ w n) (w n)

/-- **A persistent pattern**: the words `w` with `X` added at every position of `P`. -/
def persist {W : Type} [Add W] (w : ℕ → W) (P : Finset ℕ) (X : W) : ℕ → W := fun n => if n ∈ P then w n + X else w n

end Accounting.HMu
