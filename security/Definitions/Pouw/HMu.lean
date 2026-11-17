module

public import Definitions.Pouw.RelativeSecurity
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! (H-μ⁺), the receipt-luck premise in the form Lemma B consumes it, and the rounds whose coins give it. -/

@[expose] public section

namespace ResourceAccounting.HMu

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

/-- With every absorbed word pinned in the proof, a round passes only when its committed receipt is the honest one. -/
def lucky (R : SkipRound Ω O α β) (ω : Ω) : Prop :=
  R.committed (R.view ω) = R.honest (R.view ω) (R.skipped ω) ∧ 0 < R.d (R.view ω)

/-- **(H-μ) at `μ`**: on every deficient view, each value of the skipped value has probability at most `μ` given the
view. -/
def PointMass [Fintype Ω] (μ : ℝ) (R : SkipRound Ω O α β) : Prop :=
  ∀ o, 0 < R.d o → ∀ t, prob (fun ω => R.view ω = o ∧ R.skipped ω = t) ≤ μ * prob (fun ω => R.view ω = o)

/-- `Z = 1` when the honest receipt is injective in the skipped value, as a fold that commits its whole state is in each
word. -/
def Inj (Z : ℕ) (R : SkipRound Ω O α β) : Prop :=
  ∀ o, 0 < R.d o → ∀ b, ∃ s : Finset α, s.card ≤ Z ∧
    ∀ ω, R.view ω = o → R.honest o (R.skipped ω) = b → R.skipped ω ∈ s

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

def chain {S W : Type} (step : ℕ → S → W → S) (s₀ : S) (w : ℕ → W) : ℕ → S
  | 0 => s₀
  | n + 1 => step n (chain step s₀ w n) (w n)

def persist {W : Type} [Add W] (w : ℕ → W) (P : Finset ℕ) (X : W) : ℕ → W := fun n => if n ∈ P then w n + X else w n

end ResourceAccounting.HMu
