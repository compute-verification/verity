module

public import Proofs.Pouw.TS.Session.Lanes.Layer
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

/-! `LeafChecks`, the premise that each of a leaf's 33 circuits computes its Definition at its port rows, read through
the decodes `decL` and `decC`. -/

@[expose] public section

namespace ResourceAccounting.Assumptions.TS

open ResourceAccounting.TS ResourceAccounting.TS.Checked ResourceAccounting.TS.Lanes Flock.SoundAtBridge PouwWindow.Lanes
open Pouw.RelativeSecurity (Paths)

/-- `accounting-ts-leaf-checks`, backed by circuit-check and `h1zy`'s tests, not proved: the Lean has no lowering of a
Definition to a circuit, and the circuit file's input layout (`PouwWindow.Lanes.wireOf?`) is part of the premise. It
is asked only at a `K` the tile binds at (`32 ∣ K`, `K < 2^17`) because from `2^17` the s8 accumulator can clamp. -/
def LeafChecks {K nw : ℕ} {Pub : Type} (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (sess : Option (Fin 32) → RowsSession)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw) : Prop :=
  (32 ∣ K → K < 2 ^ 17 → ∀ l (vals : Fin (sess (some l)).Circ.N → Bool),
      (∀ u, ∀ g ∈ (sess (some l)).P.io u, (sess (some l)).Circ.eval vals g = vals g) →
      (decL (sess (some l)) (portRead (sess (some l)) vals)).ok = true →
      LaneOk dq part (decL (sess (some l)) (portRead (sess (some l)) vals))) ∧
    ∀ vals : Fin (sess none).Circ.N → Bool,
      (∀ u, ∀ g ∈ (sess none).P.io u, (sess none).Circ.eval vals g = vals g) →
      (decC (sess none) (portRead (sess none) vals)).ok = true →
      CombineOk (decC (sess none) (portRead (sess none) vals))

end ResourceAccounting.Assumptions.TS
