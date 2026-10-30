import Proofs.Pouw.Receipt.Duo

/-!
# duo's served receipt `y ⊕ rotl(z, h)` binds one word up to no XOR constant

Two words `d ≠ e` whose receipts differ by one XOR constant `κ` on every start state would cancel in two chains of a
parity group, which absorb with the same keys. Write `c = d ⊕ e`, `δ = c·a` and `ε = rotl(c, r) + c·a`. From the start
states with `u = y ⊕ d ⊕ k = 0` and `z = x`, the runs end at `(x, x)` and `(x + δ, x + ε)`, so for every `x`

    (x ⊕ (x + δ)) ⊕ rotl(x ⊕ (x + ε), h) = κ.

Bit `n − 1` says the carry into bit `n − 1` of `x + δ` equals the carry into bit `n − 1 − h` of `x + ε` for every `x`;
the second depends only on `x mod 2^(n−1−h)`, so the first can't step from `0` to `1` as `x` rises, and
`δ ≡ 0 (mod 2^(n−1))` (`carry_top_const`). Bit `h − 1` gives `ε ≡ 0 (mod 2^(n−1))` the same way. So `δ ∈ {0, t}`,
`t = 2^(n−1)`; `δ = 0` is `c = 0`, and `δ = t` is `c = t`, where `ε = 2^(r−1) + t`, which `0 < r < n` keeps off
`{0, t}`.
-/

namespace Pouw.ReceiptProofs

open Pouw.Receipt

/-- Equal spreads are receipts that differ by one XOR constant on every start state. -/
theorem duoWordSpread_eq_iff {n : ℕ} (k a : BitVec n) (r h : ℕ) (d e : BitVec n) :
    duoWordSpread k a r h d = duoWordSpread k a r h e ↔
      ∃ κ, ∀ s, duoReceipt h (duoWord k a r s d) ^^^ duoReceipt h (duoWord k a r s e) = κ := by
  constructor
  · intro H
    refine ⟨duoReceipt h (duoWord k a r (0, 0) d) ^^^ duoReceipt h (duoWord k a r (0, 0) e), fun s => ?_⟩
    have e1 := congrFun (congrFun H s) (0, 0)
    simp only [duoWordSpread] at e1
    apply BitVec.eq_of_getLsbD_eq
    intro i _
    have e2 := congrArg (fun v => v.getLsbD i) e1
    simp only [BitVec.getLsbD_xor] at e2 ⊢
    revert e2
    cases (duoReceipt h (duoWord k a r s d)).getLsbD i <;> cases (duoReceipt h (duoWord k a r s e)).getLsbD i <;>
      cases (duoReceipt h (duoWord k a r (0, 0) d)).getLsbD i <;>
      cases (duoReceipt h (duoWord k a r (0, 0) e)).getLsbD i <;> decide
  · rintro ⟨κ, H⟩
    funext s s'
    have e1 := H s
    have e2 := H s'
    simp only [duoWordSpread]
    apply BitVec.eq_of_getLsbD_eq
    intro i _
    have f1 := congrArg (fun v => v.getLsbD i) e1
    have f2 := congrArg (fun v => v.getLsbD i) e2
    simp only [BitVec.getLsbD_xor] at f1 f2 ⊢
    revert f1 f2
    cases (duoReceipt h (duoWord k a r s d)).getLsbD i <;> cases (duoReceipt h (duoWord k a r s e)).getLsbD i <;>
      cases (duoReceipt h (duoWord k a r s' d)).getLsbD i <;>
      cases (duoReceipt h (duoWord k a r s' e)).getLsbD i <;> cases κ.getLsbD i <;> decide

/-- The carry threshold: if the carry into bit `n − 1` of `X + δ` equals the carry into a lower bit `i` of `X + ε` for
every `X`, then `δ ≡ 0 (mod 2^(n−1))`. Otherwise `X = 2^(n−1) − (δ mod 2^(n−1)) − 2^i` (or `0`) and `X + 2^i` agree
modulo `2^i` and straddle the threshold. -/
theorem carry_top_const {n i : ℕ} (hi : i + 1 < n) (δ ε : BitVec n)
    (h : ∀ X : BitVec n, BitVec.carry (n - 1) X δ false = BitVec.carry i X ε false) :
    δ.toNat % 2 ^ (n - 1) = 0 := by
  by_contra hD
  have h2n : 2 ^ n = 2 * 2 ^ (n - 1) := by rw [← Nat.pow_succ']; congr 1; omega
  have hI : 2 ^ i * 2 ≤ 2 ^ (n - 1) := by
    rw [← Nat.pow_succ]; exact Nat.pow_le_pow_right (by norm_num) (by omega)
  have hIpos : 0 < 2 ^ i := Nat.two_pow_pos i
  have hDlt : δ.toNat % 2 ^ (n - 1) < 2 ^ (n - 1) := Nat.mod_lt _ (Nat.two_pow_pos _)
  have e0 := h (BitVec.ofNat n (2 ^ (n - 1) - δ.toNat % 2 ^ (n - 1) - 2 ^ i))
  have e1 := h (BitVec.ofNat n (2 ^ (n - 1) - δ.toNat % 2 ^ (n - 1) - 2 ^ i + 2 ^ i))
  simp only [BitVec.carry, BitVec.toNat_ofNat, Bool.toNat_false, Nat.add_zero] at e0 e1
  rw [Nat.mod_eq_of_lt (show 2 ^ (n - 1) - δ.toNat % 2 ^ (n - 1) - 2 ^ i < 2 ^ n by omega),
    Nat.mod_eq_of_lt (show 2 ^ (n - 1) - δ.toNat % 2 ^ (n - 1) - 2 ^ i < 2 ^ (n - 1) by omega)] at e0
  rw [Nat.mod_eq_of_lt (show 2 ^ (n - 1) - δ.toNat % 2 ^ (n - 1) - 2 ^ i + 2 ^ i < 2 ^ n by omega),
    Nat.mod_eq_of_lt (show 2 ^ (n - 1) - δ.toNat % 2 ^ (n - 1) - 2 ^ i + 2 ^ i < 2 ^ (n - 1) by omega),
    Nat.add_mod_right] at e1
  have e2 := e0.trans e1.symm
  simp only [decide_eq_decide] at e2
  omega

theorem mod_top_eq_zero {n : ℕ} (hn : 0 < n) {v : BitVec n} (h : v.toNat % 2 ^ (n - 1) = 0) :
    v = 0 ∨ v = BitVec.twoPow n (n - 1) := by
  have h2n : 2 ^ n = 2 * 2 ^ (n - 1) := by rw [← Nat.pow_succ']; congr 1; omega
  have hlt := v.isLt
  obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero h
  have hq2 : q < 2 := by
    by_contra hcon
    have : 2 ^ (n - 1) * 2 ≤ 2 ^ (n - 1) * q := Nat.mul_le_mul_left _ (by omega)
    omega
  obtain rfl | rfl : q = 0 ∨ q = 1 := by omega
  · left
    apply BitVec.eq_of_toNat_eq
    simp [hq]
  · right
    apply BitVec.eq_of_toNat_eq
    rw [hq, BitVec.toNat_twoPow, Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (by omega))]
    simp

theorem getLsbD_rotateLeft_top {n h : ℕ} (h0 : 0 < h) (h1 : h < n) (x : BitVec n) :
    (x.rotateLeft h).getLsbD (n - 1) = x.getLsbD (n - 1 - h) := by
  simp only [BitVec.getLsbD_rotateLeft, Nat.mod_eq_of_lt h1, show ¬ (n - 1 < h) by omega, ↓reduceIte,
    show n - 1 < n by omega, decide_true, Bool.true_and]

theorem bool_carry {p q x y c₁ c₂ : Bool}
    (h : (p ^^ q) = (((p ^^ (x ^^ c₁)) ^^ (q ^^ (y ^^ c₂))) ^^ (x ^^ y))) : c₁ = c₂ := by
  revert h; cases p <;> cases q <;> cases x <;> cases y <;> cases c₁ <;> cases c₂ <;> decide

/-- One word: two words whose served receipts differ by one XOR constant on every start state are equal, for an odd
multiplier, a rotation `0 < r < n` and a receipt rotation `0 < h < n`. -/
theorem duoWordProj_injective {n : ℕ} (k : BitVec n) (a : OddKey n) (r h : Rotation n) :
    Function.Injective (duoWordSpread k a.1 r.1 h.1) := by
  obtain ⟨a, ha⟩ := a
  obtain ⟨r, hr0, hrn⟩ := r
  obtain ⟨h, hh0, hhn⟩ := h
  have hn : 0 < n := by omega
  intro d e hde
  obtain ⟨c, hc⟩ : ∃ c, c = d ^^^ e := ⟨_, rfl⟩
  obtain ⟨δ, hδ⟩ : ∃ δ, δ = c * a := ⟨_, rfl⟩
  obtain ⟨ε, hε⟩ : ∃ ε, ε = c.rotateLeft r + c * a := ⟨_, rfl⟩
  -- from u = 0 and z = x, the runs end at (x, x) and (x + δ, x + ε)
  have E : ∀ x : BitVec n,
      x ^^^ x.rotateLeft h = ((x + δ) ^^^ (x + ε).rotateLeft h) ^^^ (δ ^^^ ε.rotateLeft h) := by
    intro x
    have e1 := congrFun (congrFun hde (0 ^^^ d ^^^ k, x)) (0 ^^^ d ^^^ k, 0)
    simp only [duoWordSpread, duoReceipt, duoWord, xor_xor_xor_cancel, xor_xor_xor_swap, ← hc] at e1
    simp only [BitVec.ofNat_eq_ofNat, BitVec.zero_xor, BitVec.zero_mul, zero_rotateLeft, BitVec.zero_add,
      BitVec.add_zero, BitVec.xor_zero] at e1
    rw [show c * a + x = x + δ by rw [hδ]; exact add_comm _ _,
      show c.rotateLeft r + (x + δ) = x + ε by rw [hε, hδ]; ac_rfl,
      show c.rotateLeft r + c * a = ε from hε.symm, ← hδ] at e1
    exact e1
  have C1 : ∀ X : BitVec n, BitVec.carry (n - 1) X δ false = BitVec.carry (n - 1 - h) X ε false := by
    intro X
    have e := congrArg (fun v => v.getLsbD (n - 1)) (E X)
    simp only [BitVec.getLsbD_xor, getLsbD_rotateLeft_top hh0 hhn, BitVec.getLsbD_add (by omega : n - 1 < n),
      BitVec.getLsbD_add (by omega : n - 1 - h < n)] at e
    exact bool_carry e
  have C2 : ∀ X : BitVec n, BitVec.carry (n - 1) X ε false = BitVec.carry (h - 1) X δ false := by
    intro X
    have e := congrArg (fun v => v.getLsbD (h - 1)) (E X)
    simp only [BitVec.getLsbD_xor, getLsbD_rotateLeft_pred hh0 hhn, BitVec.getLsbD_add (by omega : n - 1 < n),
      BitVec.getLsbD_add (by omega : h - 1 < n)] at e
    exact (bool_carry e).symm
  have mδ := carry_top_const (by omega) δ ε C1
  have mε := carry_top_const (by omega) ε δ C2
  rcases mod_top_eq_zero hn mδ with h0 | ht
  · have : c = 0 := by
      apply mul_odd_cancel ha
      rw [← hδ, h0, zero_mul]
    rw [hc] at this
    exact BitVec.xor_eq_zero_iff.mp this
  · exfalso
    have hct : c = BitVec.twoPow n (n - 1) := by
      apply mul_odd_cancel ha
      rw [← hδ, ht, top_mul_odd hn ha]
    rw [hε, hct, top_mul_odd hn ha, rotateLeft_top hr0 hrn, BitVec.toNat_add, BitVec.toNat_twoPow,
      BitVec.toNat_twoPow, Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (by omega) : 2 ^ (r - 1) < 2 ^ n),
      Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (by omega) : 2 ^ (n - 1) < 2 ^ n)] at mε
    have h2n : 2 ^ n = 2 * 2 ^ (n - 1) := by rw [← Nat.pow_succ']; congr 1; omega
    have hr : 2 ^ (r - 1) < 2 ^ (n - 1) := Nat.pow_lt_pow_right (by norm_num) (by omega)
    have hr1 : 0 < 2 ^ (r - 1) := Nat.two_pow_pos _
    rw [Nat.mod_eq_of_lt (show 2 ^ (r - 1) + 2 ^ (n - 1) < 2 ^ n by omega), Nat.add_mod_right,
      Nat.mod_eq_of_lt hr] at mε
    omega

end Pouw.ReceiptProofs
