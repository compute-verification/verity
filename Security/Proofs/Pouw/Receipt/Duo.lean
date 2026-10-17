import Definitions.Pouw.Receipt.Duo
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic.LinearCombination

/-!
# duo's words are bound: one word, and two words at every state

One word: `d ↦ u = y ⊕ d ⊕ k` is a bijection and `u ↦ u·a + z` is injective for odd `a`.

Two words: write the pairs' XOR differences `c₁ = d₁ ⊕ e₁`, `c₂ = d₂ ⊕ e₂`. Equal `z` after word 2 forces equal `u₂`
(`z − y = rotl(u)` and rotation is injective), so on every start state `(u ⊕ c₁)·a + z = (u·a + z) ⊕ c₂` and
`rotl(u) + (u·a + z) = rotl(u ⊕ c₁) + ((u ⊕ c₁)·a + z)`. At `z = −u·a` the first gives `(u ⊕ c₁)·a = u·a + c₂`, at
`u = 0` it gives `c₂ + z = z ⊕ c₂` for every `z`, so `c₂ + c₂ = 0` and `c₁·a = c₂`; then `u ⊕ c₁ = u + c₁` for every `u`,
so `c₁ + c₁ = 0`. Both are `0` or the top bit `t`, `c₁·a = c₂` makes them both `0` or both `t`, and the second equation
at `u = z = 0` gives `rotl(c₁) = −c₂`, which `rotl(t) ≠ t` rules out for `0 < r < n`.
-/

namespace Pouw.ReceiptProofs

open Pouw.Receipt

theorem mul_odd_cancel {n : ℕ} {a u v : BitVec n} (ha : a.toNat % 2 = 1) (h : u * a = v * a) : u = v := by
  apply BitVec.eq_of_toNat_eq
  have h1 := congrArg BitVec.toNat h
  simp only [BitVec.toNat_mul] at h1
  have hc : Nat.gcd (2 ^ n) a.toNat = 1 :=
    Nat.Coprime.pow_left n (Nat.coprime_two_left.mpr (Nat.odd_iff.mpr ha))
  have h2 := Nat.ModEq.cancel_right_of_coprime hc h1
  unfold Nat.ModEq at h2
  rwa [Nat.mod_eq_of_lt u.isLt, Nat.mod_eq_of_lt v.isLt] at h2

theorem rotateLeft_inj {n r : ℕ} {x y : BitVec n} (h : x.rotateLeft r = y.rotateLeft r) : x = y := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hr : r % n < n := Nat.mod_lt _ (by omega)
  by_cases hlt : i + r % n < n
  · have h1 := congrArg (fun v => v.getLsbD (i + r % n)) h
    have c1 : ¬ (i + r % n < r % n) := by omega
    simp only [BitVec.getLsbD_rotateLeft, c1, ↓reduceIte, Nat.add_sub_cancel] at h1
    simpa [hlt] using h1
  · have h1 := congrArg (fun v => v.getLsbD (i + r % n - n)) h
    have c1 : i + r % n - n < r % n := by omega
    simp only [BitVec.getLsbD_rotateLeft, c1, ↓reduceIte] at h1
    have e : n - r % n + (i + r % n - n) = i := by omega
    rwa [e] at h1

theorem zero_rotateLeft {n r : ℕ} : (0#n).rotateLeft r = 0#n := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_rotateLeft, BitVec.getLsbD_zero, Bool.and_false]
  split <;> rfl

theorem self_add_self {n : ℕ} (hn : 0 < n) {c : BitVec n} (h : c + c = 0#n) :
    c = 0#n ∨ c = BitVec.twoPow n (n - 1) := by
  have h1 := congrArg BitVec.toNat h
  simp only [BitVec.toNat_add, BitVec.toNat_ofNat, Nat.zero_mod] at h1
  have hlt := c.isLt
  have hp : 2 ^ n = 2 * 2 ^ (n - 1) := by
    rw [← Nat.pow_succ']
    congr 1
    omega
  obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero h1
  have hk2 : k < 2 := by
    by_contra hcon
    have : 2 ^ n * 2 ≤ 2 ^ n * k := Nat.mul_le_mul_left _ (by omega)
    omega
  obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
  · left
    apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_ofNat, Nat.zero_mod]
    omega
  · right
    apply BitVec.eq_of_toNat_eq
    rw [BitVec.toNat_twoPow, Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (by omega))]
    omega

theorem twoPow_ne_zero {n : ℕ} (hn : 0 < n) : BitVec.twoPow n (n - 1) ≠ 0#n := by
  intro h
  have h1 := congrArg BitVec.toNat h
  rw [BitVec.toNat_twoPow, Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (by omega))] at h1
  simp at h1

theorem rotateLeft_twoPow_ne {n r : ℕ} (h0 : 0 < r) (h1 : r < n) :
    (BitVec.twoPow n (n - 1)).rotateLeft r ≠ BitVec.twoPow n (n - 1) := by
  intro h
  have h2 := congrArg (fun v => v.getLsbD (n - 1)) h
  simp only [BitVec.getLsbD_rotateLeft, BitVec.getLsbD_twoPow, Nat.mod_eq_of_lt h1] at h2
  have c1 : ¬ (n - 1 < r) := by omega
  simp [c1] at h2
  omega

theorem xor_xor_xor_cancel {n : ℕ} (u d k : BitVec n) : u ^^^ d ^^^ k ^^^ d ^^^ k = u := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases u.getLsbD i <;> cases d.getLsbD i <;> cases k.getLsbD i <;> rfl

theorem xor_xor_xor_swap {n : ℕ} (u d e k : BitVec n) : u ^^^ d ^^^ k ^^^ e ^^^ k = u ^^^ (d ^^^ e) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases u.getLsbD i <;> cases d.getLsbD i <;> cases e.getLsbD i <;> cases k.getLsbD i <;> rfl

theorem xor_shift {n : ℕ} {y y' d e k : BitVec n} (h : y ^^^ d ^^^ k = y' ^^^ e ^^^ k) : y' = y ^^^ (d ^^^ e) := by
  have h1 := congrArg (fun v => v ^^^ e ^^^ k) h
  rw [xor_xor_xor_cancel, xor_xor_xor_swap] at h1
  exact h1.symm

/-- One word, at every state. -/
theorem duoWord_injective {n : ℕ} (k : BitVec n) (a : OddKey n) (r : ℕ) (s : BitVec n × BitVec n) :
    Function.Injective (duoWord k a.1 r s) := by
  intro d e h
  have h1 := congrArg Prod.fst h
  simp only [duoWord] at h1
  have h2 := mul_odd_cancel a.2 (add_right_cancel h1)
  exact (BitVec.xor_right_inj s.1).mp ((BitVec.xor_left_inj k).mp h2)

/-- Two words: the final state on every start state determines both words. -/
theorem duoPair_injective {n : ℕ} (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (k₂ a₂ : BitVec n)
    (r₂ : ℕ) : Function.Injective (duoPair k₁ a₁.1 r₁.1 k₂ a₂ r₂) := by
  obtain ⟨a, ha⟩ := a₁
  obtain ⟨r, hr0, hrn⟩ := r₁
  have hn : 0 < n := by omega
  rintro ⟨d₁, d₂⟩ ⟨e₁, e₂⟩ h
  -- every start state, written through u = y ⊕ d₁ ⊕ k₁ and z: the second word's XOR difference c₂ makes up the
  -- first's, c₁, in both coordinates
  have H : ∀ u z : BitVec n,
      (u ^^^ (d₁ ^^^ e₁)) * a + z = (u * a + z) ^^^ (d₂ ^^^ e₂) ∧
      u.rotateLeft r + (u * a + z) = (u ^^^ (d₁ ^^^ e₁)).rotateLeft r + ((u ^^^ (d₁ ^^^ e₁)) * a + z) := by
    intro u z
    have hs := congrFun h (u ^^^ d₁ ^^^ k₁, z)
    simp only [duoPair, duoWord, xor_xor_xor_cancel, xor_xor_xor_swap, Prod.mk.injEq] at hs
    obtain ⟨hy, hz⟩ := hs
    rw [hy] at hz
    have hU := rotateLeft_inj (add_right_cancel hz)
    rw [hU] at hy
    exact ⟨xor_shift hU, add_left_cancel hy⟩
  set c₁ := d₁ ^^^ e₁ with hc₁
  set c₂ := d₂ ^^^ e₂ with hc₂
  -- (u ⊕ c₁)·a = u·a + c₂, from z = −u·a
  have Q1 : ∀ u : BitVec n, (u ^^^ c₁) * a = u * a + c₂ := by
    intro u
    have h1 := (H u (-(u * a))).1
    rw [add_neg_cancel, show (0 : BitVec n) = 0#n from rfl, BitVec.zero_xor] at h1
    linear_combination h1
  have Q2 : ∀ z : BitVec n, c₂ + z = z ^^^ c₂ := by
    intro z
    have h1 := (H 0 z).1
    rw [Q1 0] at h1
    simpa using h1
  have hc₂2 : c₂ + c₂ = 0#n := by rw [Q2 c₂, BitVec.xor_self]
  have hca : c₁ * a = c₂ := by simpa using Q1 0
  have hc₁x : ∀ u : BitVec n, u ^^^ c₁ = u + c₁ := by
    intro u
    apply mul_odd_cancel ha
    rw [Q1 u, add_mul, hca]
  have hc₁2 : c₁ + c₁ = 0#n := by rw [← hc₁x c₁, BitVec.xor_self]
  have hrot : 0#n = c₁.rotateLeft r + c₂ := by
    have h1 := (H 0 0).2
    simpa [zero_rotateLeft, hca] using h1
  have key : c₁ = 0#n ∧ c₂ = 0#n := by
    rcases self_add_self hn hc₁2 with h0 | ht
    · refine ⟨h0, ?_⟩
      rw [← hca, h0, BitVec.zero_mul]
    · exfalso
      rcases self_add_self hn hc₂2 with h0 | ht2
      · apply twoPow_ne_zero hn
        apply mul_odd_cancel ha
        rw [← ht, hca, h0, BitVec.zero_mul]
      · apply rotateLeft_twoPow_ne hr0 hrn
        have : c₁.rotateLeft r = -c₂ := eq_neg_of_add_eq_zero_left hrot.symm
        rw [ht] at this
        rw [this, ← ht2]
        exact (eq_neg_of_add_eq_zero_left hc₂2).symm
  obtain ⟨h1, h2⟩ := key
  simp only [Prod.mk.injEq]
  exact ⟨BitVec.xor_eq_zero_iff.mp h1, BitVec.xor_eq_zero_iff.mp h2⟩

end Pouw.ReceiptProofs
