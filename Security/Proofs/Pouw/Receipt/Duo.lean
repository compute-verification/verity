import Definitions.Pouw.Receipt.Duo
import Mathlib.Basic.Finite.Prod
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic.LinearCombination

/-!
# duo's words are bound: one word, and runs of two and three words at every state

One word: `d ↦ u = y ⊕ d ⊕ k` is a bijection and `u ↦ u·a + z` is injective for odd `a`.

Two words: write the pairs' XOR differences `c₁ = d₁ ⊕ e₁`, `c₂ = d₂ ⊕ e₂`. Equal `z` after word 2 forces equal `u₂`
(`z − y = rotl(u)` and rotation is injective), so on every start state `(u ⊕ c₁)·a + z = (u·a + z) ⊕ c₂` and
`rotl(u) + (u·a + z) = rotl(u ⊕ c₁) + ((u ⊕ c₁)·a + z)`. At `z = −u·a` the first gives `(u ⊕ c₁)·a = u·a + c₂`, at
`u = 0` it gives `c₂ + z = z ⊕ c₂` for every `z`, so `c₂ + c₂ = 0` and `c₁·a = c₂`; then `u ⊕ c₁ = u + c₁` for every `u`,
so `c₁ + c₁ = 0`. Both are `0` or the top bit `t`, `c₁·a = c₂` makes them both `0` or both `t`, and the second equation
at `u = z = 0` gives `rotl(c₁) = −c₂`, which `rotl(t) ≠ t` rules out for `0 < r < n`.

Three words: equal final states force equal `z₂`. Moving the start state's `z` by `t` moves both runs' `(y₁, z₁)` by
`(t, t)`, so `u₂` and `u₂′` both gain `t`, and each `z₂` moves by `t` plus `±2^(r₂−1)`, the sign `u₂`'s top bit; so
`u₂` and `u₂′` have equal top bits on every start state. At `u₁ = 0`, `y₁′ = y₁ + δ` with `δ = c₁·a₁` and `y₁` free,
so the top bits of `y + δ` and `y` differ by the same bit for every `y`, which forces `δ ∈ {0, t}` (try `y = 0` and
`y = t − 1`), so `c₁ ∈ {0, t}`. With `c₁ = t` the start states
`(u₁, z) = (0, 0)` and `(t, t)` give the same `y₁` and `y₁′` and `z₂ − z₂′` apart by `2·2^(r₁−1) ≠ 0`. So `c₁ = 0`, the
first word is shared, one word's map on states is a bijection, and the two-word theorem binds the rest.
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

theorem xor_twoPow_of_false {n j : ℕ} {x : BitVec n} (h : x.getLsbD j = false) :
    x ^^^ BitVec.twoPow n j = x + BitVec.twoPow n j := by
  have hand : x &&& BitVec.twoPow n j = 0#n := by
    apply BitVec.eq_of_getLsbD_eq
    intro i _
    simp only [BitVec.getLsbD_and, BitVec.getLsbD_twoPow, BitVec.getLsbD_zero]
    by_cases hij : j = i
    · subst hij; simp [h]
    · simp [hij]
  rw [BitVec.add_eq_or_of_and_eq_zero _ _ hand]
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor, BitVec.getLsbD_or, BitVec.getLsbD_twoPow]
  by_cases hij : j = i
  · subst hij; simp [h]
  · simp [hij]

theorem xor_twoPow_of_true {n j : ℕ} (hj : j < n) {x : BitVec n} (h : x.getLsbD j = true) :
    x ^^^ BitVec.twoPow n j = x - BitVec.twoPow n j := by
  have hy : (x ^^^ BitVec.twoPow n j).getLsbD j = false := by
    simp [h, hj]
  have h1 := xor_twoPow_of_false hy
  have h2 : x ^^^ BitVec.twoPow n j ^^^ BitVec.twoPow n j = x := by
    rw [BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero]
  rw [h2] at h1
  linear_combination -h1

theorem twoPow_add_twoPow_ne {n j : ℕ} (hj : j + 1 < n) : BitVec.twoPow n j + BitVec.twoPow n j ≠ 0 := by
  intro h
  have h1 := congrArg BitVec.toNat h
  simp only [BitVec.toNat_add, BitVec.toNat_twoPow, BitVec.ofNat_eq_ofNat, BitVec.toNat_ofNat, Nat.zero_mod] at h1
  have hlt : 2 ^ j < 2 ^ n := Nat.pow_lt_pow_right (by norm_num) (by omega)
  have hlt2 : 2 ^ (j + 1) < 2 ^ n := Nat.pow_lt_pow_right (by norm_num) (by omega)
  rw [Nat.mod_eq_of_lt hlt, pow_succ] at *
  rw [Nat.mod_eq_of_lt (by omega)] at h1
  have : 0 < 2 ^ j := Nat.two_pow_pos j
  omega

theorem getLsbD_of_xor_twoPow_sub {n j : ℕ} (hj : j + 1 < n) {x y : BitVec n}
    (h : (x ^^^ BitVec.twoPow n j) - x = (y ^^^ BitVec.twoPow n j) - y) : x.getLsbD j = y.getLsbD j := by
  have hj' : j < n := by omega
  cases hx : x.getLsbD j <;> cases hy : y.getLsbD j
  · rfl
  · rw [xor_twoPow_of_false hx, xor_twoPow_of_true hj' hy] at h
    exact absurd (by linear_combination h) (twoPow_add_twoPow_ne hj)
  · rw [xor_twoPow_of_true hj' hx, xor_twoPow_of_false hy] at h
    exact absurd (by linear_combination -h) (twoPow_add_twoPow_ne hj)
  · rfl

theorem top_add_top {n : ℕ} (hn : 0 < n) : BitVec.twoPow n (n - 1) + BitVec.twoPow n (n - 1) = 0 := by
  apply BitVec.eq_of_toNat_eq
  have hp : 2 ^ n = 2 * 2 ^ (n - 1) := by
    rw [← Nat.pow_succ']
    congr 1
    omega
  simp only [BitVec.toNat_add, BitVec.toNat_twoPow, BitVec.ofNat_eq_ofNat, BitVec.toNat_ofNat, Nat.zero_mod]
  rw [Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (by omega)), hp]
  simp [← two_mul]

theorem add_top {n : ℕ} (hn : 0 < n) (x : BitVec n) :
    x + BitVec.twoPow n (n - 1) = x ^^^ BitVec.twoPow n (n - 1) := by
  cases hx : x.getLsbD (n - 1)
  · exact (xor_twoPow_of_false hx).symm
  · rw [xor_twoPow_of_true (by omega) hx]
    linear_combination top_add_top hn

theorem top_mul_odd {n : ℕ} (hn : 0 < n) {a : BitVec n} (ha : a.toNat % 2 = 1) :
    BitVec.twoPow n (n - 1) * a = BitVec.twoPow n (n - 1) := by
  apply BitVec.eq_of_toNat_eq
  have hp : 2 ^ n = 2 * 2 ^ (n - 1) := by
    rw [← Nat.pow_succ']
    congr 1
    omega
  have hlt : 2 ^ (n - 1) < 2 ^ n := Nat.pow_lt_pow_right (by norm_num) (by omega)
  simp only [BitVec.toNat_mul, BitVec.toNat_twoPow, Nat.mod_eq_of_lt hlt]
  obtain ⟨b, hb⟩ : ∃ b, a.toNat = 2 * b + 1 := ⟨a.toNat / 2, by omega⟩
  have e : 2 ^ (n - 1) * (2 * b + 1) = 2 ^ (n - 1) + 2 ^ n * b := by
    rw [hp, Nat.mul_add, Nat.mul_one]
    ac_rfl
  rw [hb, e, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt]

theorem rotateLeft_xor {n r : ℕ} (x y : BitVec n) :
    (x ^^^ y).rotateLeft r = x.rotateLeft r ^^^ y.rotateLeft r := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_rotateLeft, BitVec.getLsbD_xor]
  split <;> simp [hi]

theorem rotateLeft_top {n r : ℕ} (h0 : 0 < r) (h1 : r < n) :
    (BitVec.twoPow n (n - 1)).rotateLeft r = BitVec.twoPow n (r - 1) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_rotateLeft, BitVec.getLsbD_twoPow, Nat.mod_eq_of_lt h1]
  split
  · have e : (n - 1 = n - r + i) ↔ (r - 1 = i) := by omega
    simp [e, show n - 1 < n by omega, show r - 1 < n by omega]
  · have e1 : ¬ (n - 1 = i - r) := by omega
    have e2 : ¬ (r - 1 = i) := by omega
    simp [e1, e2]

theorem getLsbD_rotateLeft_pred {n r : ℕ} (h0 : 0 < r) (h1 : r < n) (x : BitVec n) :
    (x.rotateLeft r).getLsbD (r - 1) = x.getLsbD (n - 1) := by
  simp only [BitVec.getLsbD_rotateLeft, Nat.mod_eq_of_lt h1, show r - 1 < r by omega, ↓reduceIte]
  congr 1
  omega

theorem duoWord_eq_snd {n : ℕ} {k a : BitVec n} {r : ℕ} {s s' : BitVec n × BitVec n} {d d' : BitVec n}
    (h : duoWord k a r s d = duoWord k a r s' d') : s.2 = s'.2 := by
  simp only [duoWord, Prod.mk.injEq] at h
  obtain ⟨hy, hz⟩ := h
  rw [hy] at hz
  have hU := rotateLeft_inj (add_right_cancel hz)
  rw [hU] at hy
  exact add_left_cancel hy

theorem duoWord_state_injective {n : ℕ} (k a : BitVec n) (r : ℕ) (d : BitVec n) :
    Function.Injective (fun s => duoWord k a r s d) := by
  rintro ⟨y, z⟩ ⟨y', z'⟩ h
  simp only [duoWord, Prod.mk.injEq] at h
  obtain ⟨hy, hz⟩ := h
  rw [hy] at hz
  have hU := rotateLeft_inj (add_right_cancel hz)
  rw [hU] at hy
  have hz' := add_left_cancel hy
  have hy' : y = y' := (BitVec.xor_left_inj d).mp ((BitVec.xor_left_inj k).mp hU)
  simp [hy', hz']

theorem duoWord_state_surjective {n : ℕ} (k a : BitVec n) (r : ℕ) (d : BitVec n) :
    Function.Surjective (fun s => duoWord k a r s d) :=
  haveI : Finite (BitVec n) := Finite.of_equiv _ BitVec.equivFin.symm.toEquiv
  Finite.injective_iff_surjective.mp (duoWord_state_injective k a r d)

theorem xor_xor_xor_top {n : ℕ} (x m d k : BitVec n) : (x ^^^ m) ^^^ d ^^^ k = (x ^^^ d ^^^ k) ^^^ m := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases x.getLsbD i <;> cases m.getLsbD i <;> cases d.getLsbD i <;> cases k.getLsbD i <;> rfl

theorem rotateLeft_mul_top {n q : ℕ} (hn : 0 < n) {b : BitVec n} (hb : b.toNat % 2 = 1) (x w : BitVec n) :
    (x ^^^ BitVec.twoPow n (n - 1)).rotateLeft q + ((x ^^^ BitVec.twoPow n (n - 1)) * b + w) =
      x.rotateLeft q + (x * b + w) + BitVec.twoPow n (n - 1) +
        ((x.rotateLeft q ^^^ (BitVec.twoPow n (n - 1)).rotateLeft q) - x.rotateLeft q) := by
  rw [rotateLeft_xor, ← add_top hn x, add_mul, top_mul_odd hn hb]
  linear_combination

theorem msb_add_const {n : ℕ} (hn : 0 < n) (δ : BitVec n)
    (h : (δ + BitVec.ofNat n (2 ^ (n - 1) - 1)).getLsbD (n - 1) = δ.getLsbD (n - 1)) :
    δ = 0 ∨ δ = BitVec.twoPow n (n - 1) := by
  have hp : 2 ^ n = 2 * 2 ^ (n - 1) := by
    rw [← Nat.pow_succ']
    congr 1
    omega
  have hN : 0 < 2 ^ (n - 1) := Nat.two_pow_pos _
  rw [← BitVec.msb_eq_getLsbD_last, ← BitVec.msb_eq_getLsbD_last, BitVec.msb_eq_decide,
    BitVec.msb_eq_decide, BitVec.toNat_add, BitVec.toNat_ofNat] at h
  have hD := δ.isLt
  rw [hp] at hD h
  rw [Nat.mod_eq_of_lt (show 2 ^ (n - 1) - 1 < 2 * 2 ^ (n - 1) by omega)] at h
  have key : δ.toNat = 0 ∨ δ.toNat = 2 ^ (n - 1) := by
    by_cases hlt : δ.toNat + (2 ^ (n - 1) - 1) < 2 * 2 ^ (n - 1)
    · rw [Nat.mod_eq_of_lt hlt] at h
      by_cases h1 : 2 ^ (n - 1) ≤ δ.toNat
      · simp only [h1, decide_true, decide_eq_true_eq] at h
        omega
      · simp only [h1, decide_false, decide_eq_false_iff_not] at h
        omega
    · rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)] at h
      have h1 : 2 ^ (n - 1) ≤ δ.toNat := by omega
      simp only [h1, decide_true, decide_eq_true_eq] at h
      omega
  rcases key with h0 | ht
  · left
    apply BitVec.eq_of_toNat_eq
    simp [h0]
  · right
    apply BitVec.eq_of_toNat_eq
    rw [ht, BitVec.toNat_twoPow, Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (by omega))]

/-- Three words: the final state on every start state determines all three words. -/
theorem duoTriple_injective {n : ℕ} (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (k₂ : BitVec n)
    (a₂ : OddKey n) (r₂ : Rotation n) (k₃ a₃ : BitVec n) (r₃ : ℕ) :
    Function.Injective (duoTriple k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 k₃ a₃ r₃) := by
  obtain ⟨a, ha⟩ := a₁
  obtain ⟨r, hr0, hrn⟩ := r₁
  obtain ⟨b, hb⟩ := a₂
  obtain ⟨q, hq0, hqn⟩ := r₂
  have hn : 0 < n := by omega
  rintro ⟨d₁, d₂, d₃⟩ ⟨e₁, e₂, e₃⟩ h
  -- z after the second word agrees on every start state, written through u = y ⊕ d₁ ⊕ k₁ and z
  have Z : ∀ u z : BitVec n,
      ((u * a + z) ^^^ d₂ ^^^ k₂).rotateLeft q +
          (((u * a + z) ^^^ d₂ ^^^ k₂) * b + (u.rotateLeft r + (u * a + z))) =
        (((u ^^^ (d₁ ^^^ e₁)) * a + z) ^^^ e₂ ^^^ k₂).rotateLeft q +
          ((((u ^^^ (d₁ ^^^ e₁)) * a + z) ^^^ e₂ ^^^ k₂) * b +
            ((u ^^^ (d₁ ^^^ e₁)).rotateLeft r + ((u ^^^ (d₁ ^^^ e₁)) * a + z))) := by
    intro u z
    have hs := duoWord_eq_snd (congrFun h (u ^^^ d₁ ^^^ k₁, z))
    simpa only [duoWord, xor_xor_xor_cancel, xor_xor_xor_swap] using hs
  -- moving z by the top bit moves (y₁, z₁) by (t, t) in both runs; z₂ then fixes the top bit of u₂
  have M : ∀ u z : BitVec n, ((u * a + z) ^^^ d₂ ^^^ k₂).getLsbD (n - 1) =
      (((u ^^^ (d₁ ^^^ e₁)) * a + z) ^^^ e₂ ^^^ k₂).getLsbD (n - 1) := by
    intro u z
    have e0 := Z u z
    have e1 := Z u (z + BitVec.twoPow n (n - 1))
    have s1 : (u * a + (z + BitVec.twoPow n (n - 1))) ^^^ d₂ ^^^ k₂ =
        ((u * a + z) ^^^ d₂ ^^^ k₂) ^^^ BitVec.twoPow n (n - 1) := by
      rw [← add_assoc, add_top hn, xor_xor_xor_top]
    have s2 : ((u ^^^ (d₁ ^^^ e₁)) * a + (z + BitVec.twoPow n (n - 1))) ^^^ e₂ ^^^ k₂ =
        (((u ^^^ (d₁ ^^^ e₁)) * a + z) ^^^ e₂ ^^^ k₂) ^^^ BitVec.twoPow n (n - 1) := by
      rw [← add_assoc, add_top hn, xor_xor_xor_top]
    rw [s1, s2, rotateLeft_mul_top hn hb, rotateLeft_mul_top hn hb] at e1
    have hd : (((u * a + z) ^^^ d₂ ^^^ k₂).rotateLeft q ^^^ (BitVec.twoPow n (n - 1)).rotateLeft q) -
          ((u * a + z) ^^^ d₂ ^^^ k₂).rotateLeft q =
        ((((u ^^^ (d₁ ^^^ e₁)) * a + z) ^^^ e₂ ^^^ k₂).rotateLeft q ^^^
          (BitVec.twoPow n (n - 1)).rotateLeft q) - (((u ^^^ (d₁ ^^^ e₁)) * a + z) ^^^ e₂ ^^^ k₂).rotateLeft q := by
      linear_combination e1 - e0
    rw [rotateLeft_top hq0 hqn] at hd
    have hbit := getLsbD_of_xor_twoPow_sub (by omega) hd
    rwa [getLsbD_rotateLeft_pred hq0 hqn, getLsbD_rotateLeft_pred hq0 hqn] at hbit
  -- so y₁′ − y₁ = c₁·a at u = 0 is 0 or t, and c₁ is 0 or t
  have hc : d₁ ^^^ e₁ = 0 ∨ d₁ ^^^ e₁ = BitVec.twoPow n (n - 1) := by
    have m0 := M 0 0
    have m1 := M 0 (BitVec.ofNat n (2 ^ (n - 1) - 1))
    have hb0 : (BitVec.ofNat n (2 ^ (n - 1) - 1)).getLsbD (n - 1) = false := by
      simp [BitVec.getLsbD_ofNat, Nat.testBit_two_pow_sub_one]
    simp only [BitVec.ofNat_eq_ofNat, BitVec.zero_mul, BitVec.zero_add, BitVec.add_zero, BitVec.zero_xor,
      BitVec.getLsbD_xor, hb0, Bool.false_xor] at m0 m1
    have hm := m0.symm.trans m1
    simp only [Bool.xor_left_inj] at hm
    rcases msb_add_const hn _ hm.symm with h0 | ht
    · left
      apply mul_odd_cancel ha
      rw [h0, zero_mul]
    · right
      apply mul_odd_cancel ha
      rw [ht, top_mul_odd hn ha]
  have hct : d₁ ^^^ e₁ ≠ BitVec.twoPow n (n - 1) := by
    intro hct
    -- u = 0, z = 0 and u = t, z = t give the same y₁ = 0 and y₁′ = t, and z₂ − z₂′ differs by 2·2^(r−1)
    have f0 := Z 0 0
    have f1 := Z (BitVec.twoPow n (n - 1)) (BitVec.twoPow n (n - 1))
    rw [hct] at f0 f1
    simp only [BitVec.ofNat_eq_ofNat, BitVec.zero_mul, BitVec.zero_add, BitVec.add_zero, BitVec.zero_xor,
      BitVec.xor_self, top_mul_odd hn ha, top_add_top hn, zero_rotateLeft] at f0 f1
    have h2 : (BitVec.twoPow n (n - 1)).rotateLeft r + (BitVec.twoPow n (n - 1)).rotateLeft r = 0 := by
      linear_combination f1 - f0
    rw [rotateLeft_top hr0 hrn] at h2
    exact twoPow_add_twoPow_ne (by omega) h2
  have hd₁ : d₁ = e₁ := by
    rcases hc with h0 | ht
    · exact BitVec.xor_eq_zero_iff.mp h0
    · exact absurd ht hct
  subst hd₁
  have hpair : duoPair k₂ b q k₃ a₃ r₃ (d₂, d₃) = duoPair k₂ b q k₃ a₃ r₃ (e₂, e₃) := by
    funext s
    obtain ⟨s₀, rfl⟩ := duoWord_state_surjective k₁ a r d₁ s
    exact congrFun h s₀
  have h23 := duoPair_injective k₂ ⟨b, hb⟩ ⟨q, hq0, hqn⟩ k₃ a₃ r₃ hpair
  simp only [Prod.mk.injEq] at h23 ⊢
  exact ⟨trivial, h23⟩

end Pouw.ReceiptProofs
