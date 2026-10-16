import Flock.Zk
import Mathlib.Tactic.NormNum

/-!
# The pads' domain: the length of the `--zk` inner code, pinned to the code

Under `--zk` a rep's pads are committed as one codeword of the inner code, and the inner proof opens `Q = 168` of its
columns (`Flock.Zk.verifyInner`). Its length `N` (`padDomain`, `verifyInner`'s `n`, the proofs' `ZkExec.innerN`) is the
pads message, `PadLayout.len` pads and `K_PAD = 192` padding symbols, rounded up to a power of two, at rate
`1/2^INNER_LOG_INV_RATE = 1/8`:

* the pads message at `PadLayout.new m kLog k₀` is `396 + 2(m − 6) + 2(kLog − 6) + 4k₀` symbols (`padMsg_eq`), so
  `372 + 2m + 2kLog + 4k₀` from `m, kLog ≥ 6`;
* so `N = 2^12` when `2(m − 6) + 2(kLog − 6) + 4k₀ ≤ 116` and `N = 2^13` otherwise, while that sum is at most `628`
  (`padDomain_eq`): at every statement in `--zk` scope (`m ≤ 35`, `kLog ≤ 27`, `k₀ ≤ 6`) `N` is `2^12` or `2^13`, and
  which one depends on `kLog` as well as `m` (`padDomain_le`, `padDomain_ge`).
-/

namespace Claims.Flock.Numbers

open _root_.Flock.Zk

/-- **The pads' domain**: the inner code's length at a pads layout, `verifyInner`'s `n`. -/
def padDomain (lay : PadLayout) : ℕ := (lay.len + K_PAD).nextPowerOfTwo * 2 ^ INNER_LOG_INV_RATE

/-- The pads message at `PadLayout.new m kLog k₀`: `396 + 2(m − 6) + 2(kLog − 6) + 4k₀` symbols. -/
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

/-- **The pads' domain, pinned**: `2^12` when `2(m − 6) + 2(kLog − 6) + 4k₀ ≤ 116`, else `2^13`, up to `628`. -/
theorem padDomain_eq {m kLog k0 : ℕ} (h : 2 * (m - 6) + 2 * (kLog - 6) + 4 * k0 ≤ 628) :
    padDomain (PadLayout.new m kLog k0) = if 2 * (m - 6) + 2 * (kLog - 6) + 4 * k0 ≤ 116 then 2 ^ 12 else 2 ^ 13 := by
  unfold padDomain
  rw [padMsg_eq, INNER_LOG_INV_RATE]
  split_ifs with h'
  · rw [nextPowerOfTwo_eq_of (k := 8) (by norm_num; omega) (by norm_num; omega)]; norm_num
  · rw [nextPowerOfTwo_eq_of (k := 9) (by norm_num; omega) (by norm_num; omega)]; norm_num

/-- In `--zk` scope the pads' domain is at most `2^13`. -/
theorem padDomain_le {m kLog k0 : ℕ} (hm : m ≤ 35) (hk : kLog ≤ 27) (h0 : k0 ≤ 6) :
    padDomain (PadLayout.new m kLog k0) ≤ 2 ^ 13 := by
  rw [padDomain_eq (by omega)]
  split_ifs <;> norm_num

/-- The pads' domain is at least `2^12`. -/
theorem padDomain_ge {m kLog k0 : ℕ} (hm : m ≤ 35) (hk : kLog ≤ 27) (h0 : k0 ≤ 6) :
    2 ^ 12 ≤ padDomain (PadLayout.new m kLog k0) := by
  rw [padDomain_eq (by omega)]
  split_ifs <;> norm_num

end Claims.Flock.Numbers
