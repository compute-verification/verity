module

public import Definitions.Pouw.TS.Receipt
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

/-! One noise seed a call, the call's coin and a tile's own part of it, at which the three luck premises per tile are read with their statements unchanged. -/

@[expose] public section

namespace ResourceAccounting.TS

open Pouw.RelativeSecurity

abbrev CallCoin (m n : ℕ) (Rw Cl Ky : Type) : Type := (Fin m → Rw) × (Fin n → Cl) × (Fin m × Fin n → Ky)

abbrev CallRest (m n : ℕ) (Rw Cl Ky : Type) (t : Fin m × Fin n) : Type :=
  ((Fin m → Rw) × (Fin n → Cl)) × ({s : Fin m × Fin n // s ≠ t} → Ky)

def CallCoin.keySplit {m n : ℕ} {Rw Cl Ky : Type} (t : Fin m × Fin n) :
    CallCoin m n Rw Cl Ky ≃ CallRest m n Rw Cl Ky t × Ky :=
  ((Equiv.prodAssoc _ _ _).symm.trans
      (Equiv.prodCongr (Equiv.refl _) ((Equiv.piSplitAt t fun _ => Ky).trans (Equiv.prodComm _ _)))).trans
    (Equiv.prodAssoc _ _ _).symm

namespace ReceiptGame

variable {Ω : Type}

/-- At `e = CallCoin.keySplit t` the own part is tile `t`'s fold keys, with every block's noise and every other tile's keys fixed. -/
def CondReceiptLuck {Z O : Type} [Fintype O] (e : Ω ≃ Z × O) (pr : ℝ) : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right same next =>
      (∀ z, prob (fun o => same (e.symm (z, o)) ∧ ¬right (e.symm (z, o)) ∧ 0 < d (e.symm (z, o))) ≤
          pr * prob (fun o => ¬right (e.symm (z, o)) ∧ 0 < d (e.symm (z, o)))) ∧
        ∀ ω, CondReceiptLuck e pr (next ω)

end ReceiptGame

end ResourceAccounting.TS
