module

public import Proofs.Pouw.TS.Session.Lanes.Assumptions
public import Definitions.Pouw.H1Zy.CombineBits
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
import Proofs.Pouw.H1Zy.Combine
import Proofs.Pouw.H1Zy.CombineBits

/-! The combine's part of `LeafChecks`: its Definition's `ok = 1` means `CombineOk` at every input, which leaves
`CombineComputes`, equivalent to `CombineLowers`. -/

@[expose] public section

namespace ResourceAccounting.TS.Lanes

open ResourceAccounting.TS ResourceAccounting.TS.Checked Flock.SoundAtBridge PouwWindow.Lanes

theorem combineOk_iff_xorAll {nw : ℕ} (p : Fin 32 → Fin nw → BitVec 32) (r : Fin nw → BitVec 32) :
    Pouw.H1Zy.combineOk p r = true ↔ xorAll p = r := by
  rw [Pouw.H1Zy.combineOk_iff]
  exact ⟨fun h => funext h, fun h w => congrFun h w⟩

theorem combineOk_of_def {nw : ℕ} (v : CombineView nw) (hdef : v.ok = Pouw.H1Zy.combineOk v.parts v.receipt)
    (hok : v.ok = true) : CombineOk v :=
  (combineOk_iff_xorAll v.parts v.receipt).1 (hdef ▸ hok)

/-- Claims 1 and 2 of `LeafChecks`' combine field: the input wires sit where `decC` reads them, and the circuit computes
`PouwH1Combine_v1`. -/
def CombineComputes {nw : ℕ} (sess : Option (Fin 32) → RowsSession)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw) : Prop :=
  ∀ vals : Fin (sess none).Circ.N → Bool,
    (∀ u, ∀ g ∈ (sess none).P.io u, (sess none).Circ.eval vals g = vals g) →
    (decC (sess none) (portRead (sess none) vals)).ok =
      Pouw.H1Zy.combineOk (decC (sess none) (portRead (sess none) vals)).parts
        (decC (sess none) (portRead (sess none) vals)).receipt

/-- **The combine's circuit computes its pieces' composition at its port rows**: at the combine's session `sess none`, on
every wire assignment the key program's evaluation agrees with on every unit's wires, the `ok` word `decC` reads from the
port rows is `PouwH1Combine_v1`'s body on bits (`Pouw.H1Zy.combineBits`: `PouwXor32_v2`, `PouwEq32_v2` and
`PouwAndBit_v2` at their lowerings) on the bits of the partials and receipt `decC` reads. What it rests on is
circuit-check's lowering of `PouwH1Combine_v1`, which composes those three pieces. -/
def CombineLowers {nw : ℕ} (sess : Option (Fin 32) → RowsSession)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw) : Prop :=
  ∀ vals : Fin (sess none).Circ.N → Bool,
    (∀ u, ∀ g ∈ (sess none).P.io u, (sess none).Circ.eval vals g = vals g) →
    (decC (sess none) (portRead (sess none) vals)).ok =
      Pouw.H1Zy.combineBits (fun l w => Pouw.H1Zy.bitsOf ((decC (sess none) (portRead (sess none) vals)).parts l w))
        (fun w => Pouw.H1Zy.bitsOf ((decC (sess none) (portRead (sess none) vals)).receipt w))

theorem combineLowers_iff {nw : ℕ} (sess : Option (Fin 32) → RowsSession)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw) :
    CombineLowers sess decC ↔ CombineComputes sess decC := by
  unfold CombineLowers CombineComputes
  simp only [Pouw.H1Zy.combineBits_bitsOf]

theorem combineComputes_of_lowers {nw : ℕ} {sess : Option (Fin 32) → RowsSession}
    {decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw} (h : CombineLowers sess decC) :
    CombineComputes sess decC :=
  (combineLowers_iff sess decC).1 h

theorem leafChecks_of_combineComputes {K nw : ℕ} {Pub : Type} (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (sess : Option (Fin 32) → RowsSession)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw)
    (hL : 32 ∣ K → K < 2 ^ 17 → ∀ l (vals : Fin (sess (some l)).Circ.N → Bool),
      (∀ u, ∀ g ∈ (sess (some l)).P.io u, (sess (some l)).Circ.eval vals g = vals g) →
      (decL (sess (some l)) (portRead (sess (some l)) vals)).ok = true →
      LaneOk dq part (decL (sess (some l)) (portRead (sess (some l)) vals)))
    (hC : CombineComputes sess decC) : ResourceAccounting.Assumptions.TS.LeafChecks dq part sess decL decC :=
  ⟨hL, fun vals hv hok => combineOk_of_def _ (hC vals hv) hok⟩

end ResourceAccounting.TS.Lanes
