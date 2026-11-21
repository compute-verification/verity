module

public import Proofs.Pouw.TS.Session.Lanes.Decode
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

/-!
# Rows that hold a leaf's encodings decode to it

`decLane` and `decComb` read each field's bits at its port and first bit in the layout (`Lanes.Decode`). Where each
field's bits hold the encoding of the leaf's word there, and the words are in range, the decodes give the leaf back
(`Lanes.Range`):

* `LeafInRange`: the committed leaf's codes are s8, its scales' and y's encodings u32 and u16, its z s32;
* `LaneHolds`: lane `l`'s ports' rows hold its A rows (`laneA`) and B rows (`laneB`), the leaf's record, the public
  inputs' admit bits, salt, position and roots and its index `l`, each field at its place, and `ok = 1`; `CombHolds`:
  the combine's hold the receipt and `ok = 1`;
* `decLane_eq`, `decComb_eq`: then the decodes are `laneView` and the combine's view, at the partials rows they read
  (lane `l`'s and the combine's `p<l>` are one row, `Lanes.Partials`).
-/

@[expose] public section

namespace ResourceAccounting.TS.Lanes

open PouwWindow.Lanes Flock.SoundAtBridge

/-- **An A or B row is in range**: s8 codes, and its scale's encoding a u32. -/
structure RowInRange {K : ℕ} (r : Row K) : Prop where
  codes : ∀ k, -2 ^ 7 ≤ r.codes k ∧ r.codes k < 2 ^ 7
  scale : 0 ≤ r.scale ∧ r.scale < 2 ^ 32

end ResourceAccounting.TS.Lanes
