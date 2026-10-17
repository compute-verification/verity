import Definitions.Pouw.Receipt.Li
import Proofs.Pouw.Receipt.Duo
import Mathlib.Tactic.Ring

/-!
# li's words are bound in a checkpoint, at every width and length

li's half step `y ↦ (y ⊕ d₀ ⊕ k)·a + d₁` is a T-function: bit `i` of its output reads only bits `≤ i` of its inputs.
Say `q` is `p` moved at bit `v` (`LowDiff v p q`) when they agree below bit `v` and differ at it: modulo `2^(v+1)`,
`q = p + 2^v`, or equally `q = p ⊕ 2^v`, since adding the top bit of `v + 1` bits carries out of them. A common XOR, a
common addend and an odd multiplier each keep it: `(p + 2^v)·a ≡ p·a + 2^v` for odd `a`. So a change of `d₀` (odd `a`)
or of `d₁` (any `a`) whose lowest set bit is `v` moves the half step's output at bit `v`; every later half step with an
odd multiplier keeps the move; and XOR with the other chains' final states keeps it. A nonzero change has a lowest set
bit, so the checkpoint is injective in each word.
-/

namespace Pouw.ReceiptProofs

open Pouw.Receipt

/-- `q` is `p` moved at bit `v` and at no bit below it. -/
def LowDiff {n : ℕ} (v : ℕ) (p q : BitVec n) : Prop :=
  q.setWidth (v + 1) = p.setWidth (v + 1) + BitVec.twoPow (v + 1) v

theorem twoPow_top_toNat (v : ℕ) : (BitVec.twoPow (v + 1) v).toNat = 2 ^ v := by
  rw [BitVec.toNat_twoPow, Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (by omega))]

theorem add_twoPow_top {v : ℕ} (x : BitVec (v + 1)) :
    x + BitVec.twoPow (v + 1) v = x ^^^ BitVec.twoPow (v + 1) v := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [BitVec.getLsbD_add hi, BitVec.getLsbD_xor]
  have hc : BitVec.carry i x (BitVec.twoPow (v + 1) v) false = false := by
    have h2 : 2 ^ v % 2 ^ i = 0 := Nat.mod_eq_zero_of_dvd (Nat.pow_dvd_pow 2 (by omega))
    have h3 : x.toNat % 2 ^ i < 2 ^ i := Nat.mod_lt _ (Nat.two_pow_pos i)
    simp only [BitVec.carry, twoPow_top_toNat, h2, Bool.toNat_false, Nat.add_zero, decide_eq_false_iff_not, not_le]
    exact h3
  rw [hc, Bool.xor_false]

theorem xor_xor_self_left {n : ℕ} (p q : BitVec n) : p ^^^ (p ^^^ q) = q := by
  rw [← BitVec.xor_assoc, BitVec.xor_self, BitVec.zero_xor]

theorem lowDiff_iff {n v : ℕ} {p q : BitVec n} :
    LowDiff v p q ↔ (p ^^^ q).setWidth (v + 1) = BitVec.twoPow (v + 1) v := by
  unfold LowDiff
  rw [BitVec.setWidth_xor, add_twoPow_top]
  constructor
  · intro h
    rw [h, xor_xor_self_left]
  · intro h
    rw [← h, xor_xor_self_left]

theorem lt_of_lowBit {n v : ℕ} {x : BitVec n} (h : x.setWidth (v + 1) = BitVec.twoPow (v + 1) v) : v < n := by
  by_contra hn
  have h1 := congrArg (fun y => y.getLsbD v) h
  simp only [BitVec.getLsbD_setWidth, BitVec.getLsbD_twoPow, BitVec.getLsbD_of_ge x v (by omega)] at h1
  simp at h1

theorem xor_xor_cancel_right {n : ℕ} (p q c : BitVec n) : p ^^^ c ^^^ (q ^^^ c) = p ^^^ q := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases p.getLsbD i <;> cases q.getLsbD i <;> cases c.getLsbD i <;> rfl

theorem xor_xor_cancel_left {n : ℕ} (c p q : BitVec n) : c ^^^ p ^^^ (c ^^^ q) = p ^^^ q := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases c.getLsbD i <;> cases p.getLsbD i <;> cases q.getLsbD i <;> rfl

theorem lowDiff_xor {n v : ℕ} {p q : BitVec n} (c : BitVec n) (h : LowDiff v p q) :
    LowDiff v (p ^^^ c) (q ^^^ c) := by
  rw [lowDiff_iff] at h ⊢
  rwa [xor_xor_cancel_right]

theorem lowDiff_add {n v : ℕ} (hv : v + 1 ≤ n) {p q : BitVec n} (c : BitVec n) (h : LowDiff v p q) :
    LowDiff v (p + c) (q + c) := by
  unfold LowDiff at h ⊢
  rw [BitVec.setWidth_add _ _ hv, BitVec.setWidth_add _ _ hv, h]
  ring

theorem twoPow_mul_odd {v : ℕ} {b : BitVec (v + 1)} (hb : b.toNat % 2 = 1) :
    BitVec.twoPow (v + 1) v * b = BitVec.twoPow (v + 1) v := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_mul, twoPow_top_toNat]
  obtain ⟨j, hj⟩ : ∃ j, b.toNat = 2 * j + 1 := ⟨b.toNat / 2, by omega⟩
  have e : 2 ^ v * (2 * j + 1) = 2 ^ v + 2 ^ (v + 1) * j := by ring
  have hp : 2 ^ v < 2 ^ (v + 1) := Nat.pow_lt_pow_right (by norm_num) (by omega)
  rw [hj, e, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hp]

theorem lowDiff_mul {n v : ℕ} (hv : v + 1 ≤ n) {p q a : BitVec n} (ha : a.toNat % 2 = 1) (h : LowDiff v p q) :
    LowDiff v (p * a) (q * a) := by
  unfold LowDiff at h ⊢
  have hb : (a.setWidth (v + 1)).toNat % 2 = 1 := by
    rw [BitVec.toNat_setWidth, Nat.mod_mod_of_dvd _ (dvd_pow_self 2 (by omega : v + 1 ≠ 0))]
    exact ha
  rw [BitVec.setWidth_mul _ _ hv, BitVec.setWidth_mul _ _ hv, h, add_mul, twoPow_mul_odd hb]

/-- A change of `d₀` whose lowest set bit is `v` moves a half step with an odd multiplier at bit `v`. -/
theorem liHalf_word₀ {n v : ℕ} (hv : v + 1 ≤ n) (k : BitVec n) {a : BitVec n} (ha : a.toNat % 2 = 1)
    (y d₀ d₀' d₁ : BitVec n) (h : (d₀ ^^^ d₀').setWidth (v + 1) = BitVec.twoPow (v + 1) v) :
    LowDiff v (liHalf k a y (d₀, d₁)) (liHalf k a y (d₀', d₁)) := by
  simp only [liHalf]
  refine lowDiff_add hv _ (lowDiff_mul hv ha (lowDiff_xor _ ?_))
  rwa [lowDiff_iff, xor_xor_cancel_left]

/-- A change of `d₁` whose lowest set bit is `v` moves a half step at bit `v`, whatever its multiplier. -/
theorem liHalf_word₁ {n v : ℕ} (hv : v + 1 ≤ n) (k a y d₀ d₁ d₁' : BitVec n)
    (h : (d₁ ^^^ d₁').setWidth (v + 1) = BitVec.twoPow (v + 1) v) :
    LowDiff v (liHalf k a y (d₀, d₁)) (liHalf k a y (d₀, d₁')) := by
  simp only [liHalf]
  rw [add_comm _ d₁, add_comm _ d₁']
  exact lowDiff_add hv _ (lowDiff_iff.mpr h)

/-- A half step with an odd multiplier keeps a move of the state. -/
theorem liHalf_state {n v : ℕ} (hv : v + 1 ≤ n) (k : BitVec n) {a : BitVec n} (ha : a.toNat % 2 = 1)
    (d : BitVec n × BitVec n) {y y' : BitVec n} (h : LowDiff v y y') :
    LowDiff v (liHalf k a y d) (liHalf k a y' d) :=
  lowDiff_add hv _ (lowDiff_mul hv ha (lowDiff_xor _ (lowDiff_xor _ h)))

/-- Half steps with odd multipliers keep a move of the state. -/
theorem liChain_state {n v : ℕ} (hv : v + 1 ≤ n) (post : List (LiOddHalfStep n)) {y y' : BitVec n}
    (h : LowDiff v y y') :
    LowDiff v (liChain (post.map LiOddHalfStep.val) y) (liChain (post.map LiOddHalfStep.val) y') := by
  induction post generalizing y y' with
  | nil => exact h
  | cons hd tl ih =>
    simp only [List.map_cons, liChain, List.foldl_cons, LiOddHalfStep.val] at ih ⊢
    exact ih (liHalf_state hv _ hd.2.1.2 _ h)

theorem liChain_append {n : ℕ} (pre post : List (LiHalfStep n)) (h : LiHalfStep n) (y : BitVec n) :
    liChain (pre ++ h :: post) y = liChain post (liHalf h.1 h.2.1 (liChain pre y) h.2.2) := by
  simp only [liChain, List.foldl_append, List.foldl_cons]

theorem foldl_xor {α : Type} {n : ℕ} (f : α → BitVec n) (l : List α) (z : BitVec n) :
    l.foldl (fun x c => x ^^^ f c) z = z ^^^ l.foldl (fun x c => x ^^^ f c) 0 := by
  induction l generalizing z with
  | nil => simp
  | cons hd tl ih =>
    simp only [List.foldl_cons]
    rw [ih, ih (0 ^^^ f hd)]
    apply BitVec.eq_of_getLsbD_eq
    intro i _
    simp only [BitVec.getLsbD_xor, Bool.xor_assoc]
    simp

theorem liCheckpoint_split {n : ℕ} (before after : List (List (LiHalfStep n) × BitVec n))
    (c : List (LiHalfStep n) × BitVec n) :
    liCheckpoint (before ++ c :: after) = liCheckpoint before ^^^ liChain c.1 c.2 ^^^ liCheckpoint after := by
  unfold liCheckpoint
  rw [List.foldl_append, List.foldl_cons, foldl_xor _ after]

theorem xor_mid_cancel {n : ℕ} (b p q c : BitVec n) : b ^^^ p ^^^ c ^^^ (b ^^^ q ^^^ c) = p ^^^ q := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases b.getLsbD i <;> cases p.getLsbD i <;> cases q.getLsbD i <;> cases c.getLsbD i <;> rfl

/-- A change of the middle half step's words that moves it at bit `v` from every state moves the checkpoint at bit
`v` and at no bit below it. -/
theorem liCheckpoint_lowDiff {n v : ℕ} (hv : v + 1 ≤ n) (before after : List (List (LiHalfStep n) × BitVec n))
    (pre : List (LiHalfStep n)) (k a : BitVec n) (d d' : BitVec n × BitVec n) (post : List (LiOddHalfStep n))
    (y : BitVec n) (hstep : ∀ s, LowDiff v (liHalf k a s d) (liHalf k a s d')) :
    (liCheckpoint (before ++ (pre ++ (k, a, d') :: post.map LiOddHalfStep.val, y) :: after) ^^^
        liCheckpoint (before ++ (pre ++ (k, a, d) :: post.map LiOddHalfStep.val, y) :: after)).setWidth (v + 1) =
      BitVec.twoPow (v + 1) v := by
  rw [liCheckpoint_split, liCheckpoint_split, xor_mid_cancel, liChain_append, liChain_append, BitVec.xor_comm]
  exact lowDiff_iff.mp (liChain_state hv post (hstep _))

theorem exists_lowBit {n : ℕ} {x : BitVec n} (hx : x ≠ 0) :
    ∃ v, x.setWidth (v + 1) = BitVec.twoPow (v + 1) v := by
  have hex : ∃ i, x.getLsbD i = true := by
    by_contra h
    push Not at h
    apply hx
    apply BitVec.eq_of_getLsbD_eq
    intro i _
    simpa using h i
  classical
  refine ⟨Nat.find hex, BitVec.eq_of_getLsbD_eq fun i hi => ?_⟩
  rw [BitVec.getLsbD_setWidth, BitVec.getLsbD_twoPow]
  rcases Nat.lt_or_ge i (Nat.find hex) with h | h
  · have h1 := Nat.find_min hex h
    simp only [Bool.not_eq_true] at h1
    simp only [h1, Bool.and_false]
    have : Nat.find hex ≠ i := by omega
    simp [this]
  · have e : i = Nat.find hex := by omega
    subst e
    simp [Nat.find_spec hex]

theorem ne_of_lowBit {n v : ℕ} {X X' : BitVec n}
    (h : (X' ^^^ X).setWidth (v + 1) = BitVec.twoPow (v + 1) v) : X' ≠ X := by
  rintro rfl
  have h1 := congrArg BitVec.toNat h
  rw [BitVec.xor_self, BitVec.toNat_setWidth, twoPow_top_toNat] at h1
  simp only [BitVec.toNat_zero, Nat.zero_mod] at h1
  exact (Nat.two_pow_pos v).ne h1

/-- One half step is a bijection of the state, for an odd multiplier. -/
theorem liHalf_bijective {n : ℕ} (k : BitVec n) (a : OddKey n) (d : BitVec n × BitVec n) :
    Function.Bijective (fun y => liHalf k a.1 y d) := by
  have hinj : Function.Injective (fun y => liHalf k a.1 y d) := by
    intro y y' h
    simp only [liHalf] at h
    have h2 := mul_odd_cancel a.2 (add_right_cancel h)
    exact (BitVec.xor_left_inj d.1).mp ((BitVec.xor_left_inj k).mp h2)
  have : Finite (BitVec n) := Finite.of_equiv _ BitVec.equivFin.symm.toEquiv
  exact hinj.bijective_of_finite

/-- Either word's change by `x`, whose lowest set bit is `v`, moves the checkpoint at bit `v` and below it nowhere. -/
theorem liCheckpoint_lowBit {n v : ℕ} (x : LowBit n v) (before after : List (List (LiHalfStep n) × BitVec n))
    (pre : List (LiHalfStep n)) (k : BitVec n) (a : OddKey n) (d : BitVec n × BitVec n) (post : List (LiOddHalfStep n))
    (y : BitVec n) :
    (liCheckpoint (before ++ (pre ++ (k, a.1, (d.1 ^^^ x.1, d.2)) :: post.map LiOddHalfStep.val, y) :: after) ^^^
        liCheckpoint (before ++ (pre ++ (k, a.1, d) :: post.map LiOddHalfStep.val, y) :: after)).setWidth (v + 1) =
      BitVec.twoPow (v + 1) v ∧
    (liCheckpoint (before ++ (pre ++ (k, a.1, (d.1, d.2 ^^^ x.1)) :: post.map LiOddHalfStep.val, y) :: after) ^^^
        liCheckpoint (before ++ (pre ++ (k, a.1, d) :: post.map LiOddHalfStep.val, y) :: after)).setWidth (v + 1) =
      BitVec.twoPow (v + 1) v := by
  have hv : v + 1 ≤ n := lt_of_lowBit x.2
  constructor
  · exact liCheckpoint_lowDiff hv before after pre k a.1 d _ post y fun s =>
      liHalf_word₀ hv k a.2 s d.1 _ d.2 (by rw [xor_xor_self_left]; exact x.2)
  · exact liCheckpoint_lowDiff hv before after pre k a.1 d _ post y fun s =>
      liHalf_word₁ hv k a.1 s d.1 d.2 _ (by rw [xor_xor_self_left]; exact x.2)

/-- The checkpoint is injective in either word of every half step it absorbed. -/
theorem liCheckpoint_word_injective {n : ℕ} (before after : List (List (LiHalfStep n) × BitVec n))
    (pre : List (LiHalfStep n)) (k : BitVec n) (a : OddKey n) (d : BitVec n × BitVec n) (post : List (LiOddHalfStep n))
    (y : BitVec n) :
    Function.Injective (fun d₀ =>
      liCheckpoint (before ++ (pre ++ (k, a.1, (d₀, d.2)) :: post.map LiOddHalfStep.val, y) :: after)) ∧
    Function.Injective (fun d₁ =>
      liCheckpoint (before ++ (pre ++ (k, a.1, (d.1, d₁)) :: post.map LiOddHalfStep.val, y) :: after)) := by
  constructor
  · intro e e' h
    by_contra hne
    have hx : e ^^^ e' ≠ 0 := fun h0 => hne (by rw [← BitVec.xor_eq_zero_iff]; exact h0)
    obtain ⟨v, hlow⟩ := exists_lowBit hx
    have hv : v + 1 ≤ n := lt_of_lowBit hlow
    exact ne_of_lowBit (liCheckpoint_lowDiff hv before after pre k a.1 (e, d.2) (e', d.2) post y fun s =>
      liHalf_word₀ hv k a.2 s e e' d.2 hlow) h.symm
  · intro e e' h
    by_contra hne
    have hx : e ^^^ e' ≠ 0 := fun h0 => hne (by rw [← BitVec.xor_eq_zero_iff]; exact h0)
    obtain ⟨v, hlow⟩ := exists_lowBit hx
    have hv : v + 1 ≤ n := lt_of_lowBit hlow
    exact ne_of_lowBit (liCheckpoint_lowDiff hv before after pre k a.1 (d.1, e) (d.1, e') post y fun s =>
      liHalf_word₁ hv k a.1 s d.1 e e' hlow) h.symm

end Pouw.ReceiptProofs
