module

public import Proofs.Pouw.TS.Session.Tile.Layer
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

/-! `TileChecks`, the premise that a tile session's key program computes `PouwH1TileZy_v1`'s Definition at its decoded
registered ports. -/

@[expose] public section

namespace ResourceAccounting.Assumptions.TS

open ResourceAccounting.TS ResourceAccounting.TS.Checked ResourceAccounting.TS.Held ResourceAccounting.TS.Tile Flock.SoundAtBridge

/-- `accounting-ts-tile-checks`, a fact of `PouwH1TileZy_v1`'s circuit, not proved. It is asked only at a `K` the tile
binds at (`32 ∣ K`, `K < 2^17`) because from `2^17` the s8 accumulator can clamp, and holds only where every input
`ok` depends on is registered, which the complete check in `tileLayer`'s acceptance gives. -/
def TileChecks {R C K : ℕ} (dq : List ℤ → ℤ) (s : RowsSession) (ok : String)
    (dec : (ℕ → ℕ → Option Bool) → ℕ → TileView R C K) : Prop :=
  32 ∣ K → K < 2 ^ 17 → ∀ (vals : Fin s.Circ.N → Bool) (u : Fin s.n), (∀ g ∈ s.P.io u, s.Circ.eval vals g = vals g) →
    (∀ g, OkWire s ok u g → ∃ w, OkVal s (s.vb.pos g) w ∧ vals g = s.vb.bit w g) →
    TileOk dq (dec (rowsReg s vals) u.val)

end ResourceAccounting.Assumptions.TS
