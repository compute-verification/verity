module

public import Flock.Zk
import all Init.Data.Nat.Power2.Basic
public import Mathlib.Algebra.Group.Nat.Defs
public import Mathlib.Tactic.ToDual
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Mathlib.Tactic.NormNum.Pow

/-! The pads' domain, the length of the `--zk` inner code, pinned to the code. -/

@[expose] public section

namespace Claims.Flock.Numbers

open _root_.Flock.Zk

/-- **The pads' domain**: the inner code's length at a pads layout, `verifyInner`'s `n`. -/
def padDomain (lay : PadLayout) : ℕ := (lay.len + K_PAD).nextPowerOfTwo * 2 ^ INNER_LOG_INV_RATE

theorem padMsg_eq (m kLog k0 : ℕ) :
    (PadLayout.new m kLog k0).len + K_PAD = 396 + 2 * (m - 6) + 2 * (kLog - 6) + 4 * k0 := by
  simp only [PadLayout.len, PadLayout.tw, PadLayout.hab, PadLayout.zp, PadLayout.lc, PadLayout.fc, PadLayout.fa,
    PadLayout.new, K_PAD]
  omega

theorem nextPowerOfTwo_eq_of {n k : ℕ} (h1 : 2 ^ k < n) (h2 : n ≤ 2 ^ (k + 1)) :
    n.nextPowerOfTwo = 2 ^ (k + 1) := by
  refine Nat.le_antisymm (Nat.nextPowerOfTwo_le h2 ⟨k + 1, rfl⟩) ?_
  obtain ⟨j, hj⟩ := Nat.isPowerOfTwo_nextPowerOfTwo n
  have hle := Nat.le_nextPowerOfTwo n
  rw [hj] at hle ⊢
  refine Nat.pow_le_pow_right (by norm_num) ?_
  by_contra hc
  have : 2 ^ j ≤ 2 ^ k := Nat.pow_le_pow_right (by norm_num) (by omega)
  omega

theorem padDomain_eq {m kLog k0 : ℕ} (h : 2 * (m - 6) + 2 * (kLog - 6) + 4 * k0 ≤ 628) :
    padDomain (PadLayout.new m kLog k0) = if 2 * (m - 6) + 2 * (kLog - 6) + 4 * k0 ≤ 116 then 2 ^ 12 else 2 ^ 13 := by
  unfold padDomain
  rw [padMsg_eq, INNER_LOG_INV_RATE]
  split_ifs with h'
  · rw [nextPowerOfTwo_eq_of (k := 8) (by norm_num; omega) (by norm_num; omega)]; norm_num
  · rw [nextPowerOfTwo_eq_of (k := 9) (by norm_num; omega) (by norm_num; omega)]; norm_num

theorem padDomain_le {m kLog k0 : ℕ} (hm : m ≤ 35) (hk : kLog ≤ 27) (h0 : k0 ≤ 10) :
    padDomain (PadLayout.new m kLog k0) ≤ 2 ^ 13 := by
  rw [padDomain_eq (by omega)]
  split_ifs <;> norm_num

theorem padDomain_ge {m kLog k0 : ℕ} (hm : m ≤ 35) (hk : kLog ≤ 27) (h0 : k0 ≤ 10) :
    2 ^ 12 ≤ padDomain (PadLayout.new m kLog k0) := by
  rw [padDomain_eq (by omega)]
  split_ifs <;> norm_num

end Claims.Flock.Numbers
