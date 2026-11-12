module

public import Definitions.Hash.Hm96
public import Mathlib.Tactic.Basic
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
# What the hash area's witnesses share

A list of messages whose digests are pairwise different holds no collision among any of its messages
(`not_collisionIn`), and the hm96 mask of an all-zero salt is zero (`mask_zero`), so a witness's commit string is its
digest beside the salt's hash, which the kernel computes without the mask's 1536 terms a bit.
-/

@[expose] public section

namespace Hash.SecurityProofs

theorem not_collisionIn {H : Msg → Digest} {xs ys : List Msg} (hs : xs ⊆ ys)
    (hn : (ys.map fun m => (H m).bytes).Nodup) : ¬ CollisionIn H xs := by
  rintro ⟨x, hx, y, hy, hxy, he⟩
  exact hxy (List.inj_on_of_nodup_map hn (hs hx) (hs hy) he)

theorem mask_zero (key y : Msg) (hy : ∀ b ∈ y, b = 0) : Hm96.mask key y = List.replicate 64 0 := by
  have hb : ∀ t, Hm96.bit y t = false := fun t => by
    have h0 : y.getD (t / 8) 0 = 0 := by
      rw [List.getD_eq_getElem?_getD]
      cases h : y[t / 8]? with
      | none => rfl
      | some b => exact hy b (List.mem_of_getElem? h)
    unfold Hm96.bit
    rw [h0]
    simp
  have hb' : Hm96.bit y = fun _ => false := funext hb
  simp only [Hm96.mask, Hm96.maskBit, hb', List.filter_false, List.foldl_nil, Bool.false_eq_true, ite_false]
  decide +kernel

end Hash.SecurityProofs
