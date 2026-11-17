module

public import Proofs.Flock.Claims.Bridge.Hm96Tree
public import Proofs.Flock.Claims.Complete.Zk.Coins
public import Proofs.Flock.Claims.KOf
public import Proofs.Flock.Claims.KS.PlainAcc
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init

/-! # What the plain KS proofs read of `Claims.Recursion.BoundDeploy`, without its recursion imports. -/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)

open FlockDeploy (Coin)

/-- A rep's schedule at a table of the session (`rec_live.rep_rounds`, `Flock.RecVStar.repRounds`): at the table's `m`
and `k_log` and the session's claims, `2 + points · regions` as `FlockDeploy.repRounds` counts them; empty off
`fast100`. -/
def repSchedAt (spec : _root_.Flock.SessionSpec) (st : _root_.Flock.Stmt) : List _root_.Flock.RecVStar.SchedRound :=
  ((_root_.Flock.RecVStar.repRounds st.m st.c.kLog (2 + spec.points * st.regions.size)).map Array.toList).getD []

/-- A table's level-0 top's nodes, `2^{cap₀}` at `fast100`'s first level (`2^7 = 128` at its 218 queries). -/
def level0Nodes (st : _root_.Flock.Stmt) : ℕ :=
  2 ^ ((_root_.Flock.fast100 st.m).map fun s => (s.levels.getD 0 default).cap).getD 0

end RecDeploy
