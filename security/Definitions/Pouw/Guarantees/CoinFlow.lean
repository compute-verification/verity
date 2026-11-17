module

public import Definitions.Pouw.CoinFlow
public import Definitions.Pouw.HMu
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

/-! The coin rules give (H-μ)'s view independence for every Program the Build admits, outside each noise injection's forward cone. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.CoinFlow

open ResourceAccounting.CoinFlow ResourceAccounting.HMu Pouw.RelativeSecurity

/-- **A noise injection's coins reach only its forward cone (rule (b)).** In a Program the Build admits, two coin samples
that agree off a noise injection's fresh coins give every node outside its forward cone the same value. -/
def ViewIndependent : Prop :=
  ∀ (m : ℕ) (C V : Type) [DecidableEq C] [Inhabited V] (P : FlowGraph m C V), P.admits = true →
    ∀ (k : Fin m) (σ σ' : C → Bool), (∀ c, c ∉ P.noiseCoins k → σ c = σ' c) →
      ∀ g, ¬P.Desc k g → P.eval σ g = P.eval σ' g

/-- **A value that isn't a sink reads the coins only through noise injections (rule (a)).** In a Program the Build
admits, two coin samples that agree on the fresh coins of every noise injection a non-fold node reads, through any chain,
give it the same value. -/
def CoinsThroughNoise : Prop :=
  ∀ (m : ℕ) (C V : Type) [DecidableEq C] [Inhabited V] (P : FlowGraph m C V), P.admits = true →
    ∀ (g : Fin m) (σ σ' : C → Bool), P.fold g = false →
      (∀ k, P.Desc k g → ∀ c ∈ P.noiseCoins k, σ c = σ' c) → P.eval σ g = P.eval σ' g

/-- **A noise injection's forward cone is its one reader's (rule (b)).** In a Program the Build admits, a node that
reads a noise injection's value through some chain is the injection or reads the injection's one reader. -/
def OneReader : Prop :=
  ∀ (m : ℕ) (C V : Type) [DecidableEq C] (P : FlowGraph m C V), P.admits = true →
    ∀ (k r g : Fin m), P.noise k = true → k ∈ P.reads r → P.Desc k g → g = k ∨ P.Desc r g

/-- **(H-μ) from a noise injection's coins, outright.** In a Program the Build admits, over uniform coins, a skipped
value `a + Σ_j [e_j]·v_j` on the `u` fresh coins of node `k`, whose base, nonzero integer weights and view are functions
of `View k`, takes any value `t` with probability at most `C(u, ⌊u/2⌋)/2^u` given the view. -/
def PointMass : Prop :=
  ∀ (m : ℕ) (C V : Type) [Fintype C] [DecidableEq C] [Inhabited V] (P : FlowGraph m C V), P.admits = true →
    ∀ (k : Fin m) (O : Type) (F : P.View k → O) (a : P.View k → ℤ) (v : P.View k → {c // c ∈ P.noiseCoins k} → ℤ)
      (o : O) (t : ℤ),
      prob (fun σ : C → Bool => F (P.view k σ) = o ∧ (∀ j, v (P.view k σ) j ≠ 0) ∧
          a (P.view k σ) + coinSum (v (P.view k σ)) (fun j => σ j.1) = t) ≤
        (((P.noiseCoins k).card).choose ((P.noiseCoins k).card / 2) : ℝ) / 2 ^ (P.noiseCoins k).card *
          prob (fun σ : C → Bool => F (P.view k σ) = o)

end ResourceAccounting.Guarantees.CoinFlow
