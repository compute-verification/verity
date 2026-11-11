module

public import Definitions.Hash.Tree
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
# The tree over computable leaves opens at every rank

`Tree.rootOf` over a leaf function accepts each rank below the domain's count with that rank's leaf and its own path
(`Tree.sibsAt`): the climb from a node with its siblings reaches its ancestor (`climb_nodeAt`), and the domain's depth
covers its count (`le_two_pow_depth`), so the climb from a leaf ends at the root.
-/

@[expose] public section

namespace Hash.SecurityProofs.TreeRoot

open Hash Hash.Tree

theorem le_two_pow_depth (n : ℕ) : n ≤ 2 ^ depth n := by
  unfold depth
  split
  · simp; omega
  · have := @Nat.lt_log2_self (n - 1)
    omega

/-- One step of the climb: the node message over `cur` and its sibling, in the order `cur`'s parity gives, hashes to
their parent. -/
theorem parent (H : Msg → Digest) (D : Domain) (leaf : ℕ → Digest) (lv cur : ℕ) :
    H (if cur % 2 = 0 then nodeMsg D lv (cur / 2) (nodeAt H D leaf lv cur) (nodeAt H D leaf lv (sib cur))
      else nodeMsg D lv (cur / 2) (nodeAt H D leaf lv (sib cur)) (nodeAt H D leaf lv cur)) =
      nodeAt H D leaf (lv + 1) (cur / 2) := by
  show _ = H (nodeMsg D lv (cur / 2) (nodeAt H D leaf lv (2 * (cur / 2))) (nodeAt H D leaf lv (2 * (cur / 2) + 1)))
  unfold sib
  split_ifs with h
  · rw [show 2 * (cur / 2) = cur by omega]
  · rw [show 2 * (cur / 2) + 1 = cur by omega, show 2 * (cur / 2) = cur - 1 by omega]

/-- The climb from the node at level `lv` and index `cur`, with its `k` siblings from the tree, ends at its ancestor `k`
levels up. -/
theorem climb_nodeAt (H : Msg → Digest) (D : Domain) (leaf : ℕ → Digest) :
    ∀ (k lv cur : ℕ), (climb H D lv cur (nodeAt H D leaf lv cur)
      (List.ofFn fun j : Fin k => nodeAt H D leaf (lv + j) (sib (cur / 2 ^ j.val)))).2 =
        nodeAt H D leaf (lv + k) (cur / 2 ^ k)
  | 0, lv, cur => by simp [climb]
  | k + 1, lv, cur => by
    rw [List.ofFn_succ]
    simp only [climb, Fin.val_zero, Nat.add_zero, pow_zero, Nat.div_one, Fin.val_succ]
    rw [parent H D leaf lv cur]
    have e : (List.ofFn fun j : Fin k => nodeAt H D leaf (lv + (j.val + 1)) (sib (cur / 2 ^ (j.val + 1)))) =
        List.ofFn fun j : Fin k => nodeAt H D leaf (lv + 1 + j.val) (sib (cur / 2 / 2 ^ j.val)) := by
      congr 1
      funext j
      rw [Nat.div_div_eq_div_mul, ← pow_succ', show lv + 1 + j.val = lv + (j.val + 1) by omega]
    rw [e, climb_nodeAt H D leaf k (lv + 1) (cur / 2), Nat.div_div_eq_div_mul, ← pow_succ',
      show lv + 1 + k = lv + (k + 1) by omega]

/-- **The tree's own path accepts**: at a rank below the domain's count, the tree over `leaf` accepts that rank's leaf
with its path (`Tree.sibsAt`) against its root (`Tree.rootOf`). -/
theorem accepts_rootOf (H : Msg → Digest) (D : Domain) (leaf : ℕ → Digest) (rank : ℕ) (h : rank < D.count) :
    Tree.accepts H D (rootOf H D leaf) rank (leaf rank) (sibsAt H D leaf rank) := by
  unfold Tree.accepts
  refine ⟨h, by simp [sibsAt], ?_⟩
  have hc := climb_nodeAt H D leaf (depth D.count) 0 rank
  simp only [Nat.zero_add] at hc
  have h0 : rank / 2 ^ depth D.count = 0 :=
    Nat.div_eq_of_lt (lt_of_lt_of_le h (le_two_pow_depth D.count))
  rw [h0] at hc
  exact hc

end Hash.SecurityProofs.TreeRoot
