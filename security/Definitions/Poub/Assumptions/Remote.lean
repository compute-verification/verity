module

public import Definitions.Poub.Remote
public import Definitions.Poub.Remote.Run
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

/-! PoUB's named assumptions on a run: `Identity`, `ComputeAdditive`, `OutsideWork` and `ZCost`. -/

@[expose] public section

namespace ResourceAccounting.Remote.Assumptions

open ResourceAccounting.Remote

variable {τ : Type} {N : Node τ} {a : Audit} {Ω : Type}

/-- **Identity** (statement C: no undeclared device of any kind): on every outcome, no location outside the node reads a
tier, holds state or rebuilds a word's bits for the receipts. -/
def Identity (r : Run N a Ω) : Prop := ∀ ω, r.readOut ω = 0 ∧ r.stateOut ω = 0 ∧ r.rebuiltOut ω = 0

/-- **W at the rebuild pricing**: on every outcome, the node's work at π_r outside PoUW's account and PoUW's account's
add within `Φ_r · T`. -/
def ComputeAdditive (r : Run N a Ω) : Prop := ∀ ω, r.rebuildWork ω + r.pouwWork ω ≤ N.rebuild * a.T

/-- **Outside work, at `Φ_out`**: on every outcome, the outside compute's work at PoUW's pricing is at most
`Φ_out · T`. -/
def OutsideWork (r : Run N a Ω) (Φout : ℝ) : Prop := ∀ ω, (r.workOut ω : ℝ) ≤ Φout * a.T

/-- **PoUW's conjecture for the bound `z`**: outside the node, each `z_b` bits of bound `z` cost `W_tile` at PoUW's
pricing, except with probability `δ` when the audit accepts. -/
def ZCost [Fintype Ω] (r : Run N a Ω) (Wtile δ : ℝ) : Prop :=
  r.Holds δ fun ω => (r.zOut ω : ℝ) * Wtile ≤ a.zb * r.workOut ω

end ResourceAccounting.Remote.Assumptions
