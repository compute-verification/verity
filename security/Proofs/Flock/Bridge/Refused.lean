module

public import Proofs.Flock.Bridge.Canon
public import Flock.PortRoots
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
import Proofs.Flock.Soundness.Game

/-!
# The run's population check

The SPOT run refuses a registration whose population isn't the game's `n` units in order (`Spot.Run.run`,
`population-order`, `Spot.Slots.popOk`).

* `Bridge.RunPopOk desc qr n`: the descriptor parses and its population under the query is `n` units in order
  (`Slots.popOk`), which the run checks at registration;
* `popMatches_of_popOk`: that check gives `Bridge.PopMatches`.
-/

@[expose] public section

namespace Flock.Bridge

open Properties.Certifier.Bytes (bs bs_inj)

/-- **The run's population check is `PopMatches`.** -/
theorem popMatches_of_popOk {desc : Hash.Msg} {qr : Lean.Json} {n : ℕ} {p : Spot.Program}
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (h : Spot.Slots.popOk p qr n = true) :
    PopMatches desc qr n := by
  show popCheck desc qr n = true
  unfold popCheck
  rw [hp]
  exact h

/-- **The SPOT run passes the registration's population** (`Spot.Run.run`, `population-order`):
the descriptor `desc` parses and its population under the query `qr` is `n` units in order (`Slots.popOk`). -/
def RunPopOk (desc : Hash.Msg) (qr : Lean.Json) (n : ℕ) : Prop :=
  ∃ p, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧ Spot.Slots.popOk p qr n = true

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

end Flock.Bridge
