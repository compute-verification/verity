import Proofs.Pouw.Receipt.DuoReceipt
import Proofs.Pouw.Receipt.Fx1

/-!
# Two duo words have no all-state XOR kernel

Two word pairs `d` and `e` whose final states differ by one XOR constant `κ = (κ_y, κ_z)` on every start state are
equal. Write `c₁ = d₁ ⊕ e₁`, `t = 2^(n−1)`, the start state through `u = y ⊕ d₁ ⊕ k₁` and `z`, and the second word's
`v = y₁ ⊕ d₂ ⊕ k₂` (`v′` in the run of `e`).

* `c₁ ∈ {0, t}`. Moving `z` by `t` moves both runs' `(y₁, z₁)` by `(t, t)`, which leaves `y₂` and moves `z₂` by
  `±2^(r₂−1)`, the sign `v`'s top bit. Both moves keep the XOR of the two `z₂` at `κ_z`, so the carries out of bit
  `r₂ − 1` agree (`getLsbD_xor_of_shift`) and `v`, `v′` have top bits that differ by bit `r₂ − 1` of `κ_z` on every
  start state; at `u = 0` that forces `c₁·a₁ ∈ {0, t}` (`xor_eq_zero_or_top_of_const`).
* `c₁ ≠ t`. Then `y₁′ = y₁ + t` whatever `u`, so `v′` is fixed by `v`, and subtracting the kernel's `y` coordinate from
  its `z` coordinate gives `rotl(v′, r₂) + (W ⊕ κ_y) = (rotl(v, r₂) + W) ⊕ κ_z` for every `y₁` and `W = y₂`. At
  `v = 0` this makes `C = rotl(v′, r₂)` satisfy `C + C = 0` and `C = κ_y ⊕ κ_z`, and at `W = 0` it makes
  `κ_z + κ_z = 0`, so `κ_y ∈ {0, t}`. The start states `(u, z) = (0, 0)` and `(t, t)` reach the same `y₁` and `y₁′`
  with `z₁′ − z₁` apart by `2·2^(r₁−1) ≠ 0`, so `y₂′ − y₂` can't be the one constant `κ_y`.
* `c₁ = 0`: the shared first word permutes the states, so the second word alone has a full-state XOR constant, and
  then its served receipt (`h = 1`) has one, which `duoWordProj_injective` rules out.

fx1 is duo on the words times odd multipliers (`fx1Pair`), so the same holds for it.
-/

namespace Pouw.ReceiptProofs

open Pouw.Receipt

theorem xor_cross {n : ℕ} {a b c d : BitVec n} (h : a ^^^ b = c ^^^ d) : a ^^^ c = b ^^^ d := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  have e := congrArg (fun v => v.getLsbD i) h
  simp only [BitVec.getLsbD_xor] at e ⊢
  revert e
  cases a.getLsbD i <;> cases b.getLsbD i <;> cases c.getLsbD i <;> cases d.getLsbD i <;> decide

theorem eq_xor_of_xor_eq {n : ℕ} {a b κ : BitVec n} (h : a ^^^ b = κ) : b = a ^^^ κ := by
  rw [← h, ← BitVec.xor_assoc, BitVec.xor_self, BitVec.zero_xor]

/-- Equal spreads: the two runs' final states differ by the same XOR on every start state. -/
theorem stateSpread_const {n : ℕ} {F G : BitVec n × BitVec n → BitVec n × BitVec n}
    (h : stateSpread F = stateSpread G) (s s₀ : BitVec n × BitVec n) :
    (F s).1 ^^^ (G s).1 = (F s₀).1 ^^^ (G s₀).1 ∧ (F s).2 ^^^ (G s).2 = (F s₀).2 ^^^ (G s₀).2 := by
  have e := congrFun (congrFun h s) s₀
  simp only [stateSpread, Prod.mk.injEq] at e
  exact ⟨xor_cross e.1, xor_cross e.2⟩

theorem duoReceipt_xor {n : ℕ} (h : ℕ) (S S' : BitVec n × BitVec n) :
    duoReceipt h S ^^^ duoReceipt h S' = (S.1 ^^^ S'.1) ^^^ (S.2 ^^^ S'.2).rotateLeft h := by
  simp only [duoReceipt, rotateLeft_xor]
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases S.1.getLsbD i <;> cases S'.1.getLsbD i <;> cases (S.2.rotateLeft h).getLsbD i <;>
    cases (S'.2.rotateLeft h).getLsbD i <;> rfl

theorem xor_xor_xor_shift {n : ℕ} (u d e k : BitVec n) : u ^^^ (d ^^^ e) ^^^ e ^^^ k = u ^^^ d ^^^ k := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases u.getLsbD i <;> cases d.getLsbD i <;> cases e.getLsbD i <;> cases k.getLsbD i <;> rfl

theorem xor_xor_self_zero {n : ℕ} (d k : BitVec n) : (d ^^^ k) ^^^ d ^^^ k = 0#n := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor, BitVec.getLsbD_zero]
  cases d.getLsbD i <;> cases k.getLsbD i <;> rfl

theorem xor_eq_add_of_zero_or_top {n : ℕ} (hn : 0 < n) {κ : BitVec n}
    (h : κ = 0#n ∨ κ = BitVec.twoPow n (n - 1)) (x : BitVec n) : x ^^^ κ = x + κ := by
  rcases h with rfl | rfl
  · rw [BitVec.xor_zero, BitVec.add_zero]
  · exact (add_top hn x).symm

theorem carry_twoPow_self {n j : ℕ} (hj : j < n) (x : BitVec n) :
    BitVec.carry j x (BitVec.twoPow n j) false = false := by
  have h1 : (BitVec.twoPow n j).toNat % 2 ^ j = 0 := by
    rw [BitVec.toNat_twoPow, Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) hj), Nat.mod_self]
  have h2 := Nat.mod_lt x.toNat (Nat.two_pow_pos j)
  unfold BitVec.carry
  rw [h1]
  apply decide_eq_false
  simp only [Bool.toNat_false, Nat.add_zero, ge_iff_le, not_le]
  exact h2

/-- Adding `2^j` flips bit `j`. -/
theorem getLsbD_add_twoPow_self {n j : ℕ} (hj : j < n) (x : BitVec n) :
    (x + BitVec.twoPow n j).getLsbD j = !x.getLsbD j := by
  rw [BitVec.getLsbD_add hj, carry_twoPow_self hj]
  simp [hj]

/-- Adding `2^j` carries bit `j` into bit `j + 1`. -/
theorem getLsbD_add_twoPow_succ {n j : ℕ} (hj : j + 1 < n) (x : BitVec n) :
    (x + BitVec.twoPow n j).getLsbD (j + 1) = (x.getLsbD (j + 1) ^^ x.getLsbD j) := by
  rw [BitVec.getLsbD_add hj, BitVec.carry_succ, carry_twoPow_self (by omega)]
  simp [BitVec.getLsbD_twoPow, show j < n by omega]

/-- Subtracting `2^j` borrows into bit `j + 1` exactly when bit `j` is clear. -/
theorem getLsbD_sub_twoPow_succ {n j : ℕ} (hj : j + 1 < n) (x : BitVec n) :
    (x - BitVec.twoPow n j).getLsbD (j + 1) = (x.getLsbD (j + 1) ^^ !x.getLsbD j) := by
  obtain ⟨w, rfl⟩ : ∃ w, x = w + BitVec.twoPow n j := ⟨x - BitVec.twoPow n j, (sub_add_cancel _ _).symm⟩
  rw [show w + BitVec.twoPow n j - BitVec.twoPow n j = w by linear_combination, getLsbD_add_twoPow_succ hj,
    getLsbD_add_twoPow_self (by omega)]
  cases w.getLsbD (j + 1) <;> cases w.getLsbD j <;> rfl

/-- Moving `W` and `W′` by `±2^j`, each sign set by bit `j` of its own `R` (`(R ⊕ 2^j) − R`), keeps `W ⊕ W′` only if
bit `j` of `W ⊕ W′` is the two signs' XOR: bit `j + 1` of each moved sum is its carry or borrow out of bit `j`. -/
theorem getLsbD_xor_of_shift {n j : ℕ} (hj : j + 1 < n) (W W' R R' : BitVec n)
    (h : W ^^^ W' = (W + ((R ^^^ BitVec.twoPow n j) - R)) ^^^ (W' + ((R' ^^^ BitVec.twoPow n j) - R'))) :
    (W ^^^ W').getLsbD j = (R.getLsbD j ^^ R'.getLsbD j) := by
  have shift : ∀ X Q : BitVec n, (X + ((Q ^^^ BitVec.twoPow n j) - Q)).getLsbD (j + 1) =
      (X.getLsbD (j + 1) ^^ (X.getLsbD j ^^ Q.getLsbD j)) := by
    intro X Q
    cases hQ : Q.getLsbD j
    · rw [xor_twoPow_of_false hQ, show Q + BitVec.twoPow n j - Q = BitVec.twoPow n j by linear_combination,
        getLsbD_add_twoPow_succ hj, Bool.xor_false]
    · rw [xor_twoPow_of_true (by omega) hQ,
        show X + (Q - BitVec.twoPow n j - Q) = X - BitVec.twoPow n j by linear_combination, getLsbD_sub_twoPow_succ hj]
      cases X.getLsbD (j + 1) <;> cases X.getLsbD j <;> rfl
  have hb := congrArg (fun v => v.getLsbD (j + 1)) h
  simp only [BitVec.getLsbD_xor] at hb ⊢
  rw [shift, shift] at hb
  revert hb
  cases W.getLsbD (j + 1) <;> cases W'.getLsbD (j + 1) <;> cases W.getLsbD j <;> cases W'.getLsbD j <;>
    cases R.getLsbD j <;> cases R'.getLsbD j <;> decide

/-- `xor_eq_zero_or_top` when the two `u`'s top bits differ by one constant `β` rather than agree. -/
theorem xor_eq_zero_or_top_of_const {n : ℕ} (hn : 0 < n) {a : BitVec n} (ha : a.toNat % 2 = 1)
    (c d₂ e₂ k₂ : BitVec n) (β : Bool)
    (M : ∀ u z : BitVec n, ((u * a + z) ^^^ d₂ ^^^ k₂).getLsbD (n - 1) =
      ((((u ^^^ c) * a + z) ^^^ e₂ ^^^ k₂).getLsbD (n - 1) ^^ β)) :
    c = 0 ∨ c = BitVec.twoPow n (n - 1) := by
  have htop : (BitVec.twoPow n (n - 1)).getLsbD (n - 1) = true := by
    simp [show n - 1 < n by omega]
  cases β
  · exact xor_eq_zero_or_top hn ha c d₂ e₂ k₂ (fun u z => by rw [M u z, Bool.xor_false])
  · refine xor_eq_zero_or_top hn ha c d₂ (e₂ ^^^ BitVec.twoPow n (n - 1)) k₂ (fun u z => ?_)
    rw [M u z]
    simp only [BitVec.getLsbD_xor, htop]
    cases ((u ^^^ c) * a + z).getLsbD (n - 1) <;> cases e₂.getLsbD (n - 1) <;> cases k₂.getLsbD (n - 1) <;> rfl

/-- The pair's final state from the start state `(u ⊕ d₁ ⊕ k₁, z)`, through `u` and `z`. -/
theorem duoPair_at {n : ℕ} (k₁ a : BitVec n) (r : ℕ) (k₂ b : BitVec n) (q : ℕ) (d₁ d₂ u z : BitVec n) :
    duoPair k₁ a r k₂ b q (d₁, d₂) (u ^^^ d₁ ^^^ k₁, z) =
      (((u * a + z) ^^^ d₂ ^^^ k₂) * b + (u.rotateLeft r + (u * a + z)),
        ((u * a + z) ^^^ d₂ ^^^ k₂).rotateLeft q +
          (((u * a + z) ^^^ d₂ ^^^ k₂) * b + (u.rotateLeft r + (u * a + z)))) := by
  simp only [duoPair, duoWord, xor_xor_xor_cancel]

/-- The other pair from the same start state: its first word's `u` is `u ⊕ (d₁ ⊕ e₁)`. -/
theorem duoPair_at_other {n : ℕ} (k₁ a : BitVec n) (r : ℕ) (k₂ b : BitVec n) (q : ℕ) (d₁ e₁ e₂ u z : BitVec n) :
    duoPair k₁ a r k₂ b q (e₁, e₂) (u ^^^ d₁ ^^^ k₁, z) =
      duoPair k₁ a r k₂ b q (e₁, e₂) ((u ^^^ (d₁ ^^^ e₁)) ^^^ e₁ ^^^ k₁, z) := by
  rw [xor_xor_xor_shift]

/-- Two words: word pairs whose final states differ by one XOR constant on every start state are equal, for odd
multipliers and rotations `0 < r < n`. -/
theorem duoPairXor_injective {n : ℕ} (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (k₂ : BitVec n)
    (a₂ : OddKey n) (r₂ : Rotation n) :
    Function.Injective (fun d => stateSpread (duoPair k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 d)) := by
  obtain ⟨a, ha⟩ := a₁
  obtain ⟨r, hr0, hrn⟩ := r₁
  obtain ⟨b, hb⟩ := a₂
  obtain ⟨q, hq0, hqn⟩ := r₂
  have hn : 0 < n := by omega
  rintro ⟨d₁, d₂⟩ ⟨e₁, e₂⟩ h
  -- the two runs' final states differ by κ = (κy, κz) on every start state
  obtain ⟨κy, κz, HY, HZ⟩ : ∃ κy κz : BitVec n,
      (∀ u z : BitVec n, (duoPair k₁ a r k₂ b q (d₁, d₂) (u ^^^ d₁ ^^^ k₁, z)).1 ^^^
        (duoPair k₁ a r k₂ b q (e₁, e₂) (u ^^^ d₁ ^^^ k₁, z)).1 = κy) ∧
      (∀ u z : BitVec n, (duoPair k₁ a r k₂ b q (d₁, d₂) (u ^^^ d₁ ^^^ k₁, z)).2 ^^^
        (duoPair k₁ a r k₂ b q (e₁, e₂) (u ^^^ d₁ ^^^ k₁, z)).2 = κz) :=
    ⟨_, _, fun u z => (stateSpread_const h (u ^^^ d₁ ^^^ k₁, z) (0, 0)).1,
      fun u z => (stateSpread_const h (u ^^^ d₁ ^^^ k₁, z) (0, 0)).2⟩
  -- moving z by t moves (y₁, z₁) by (t, t), so z₂ moves by ±2^(q−1), the sign v's top bit
  have shiftZ : ∀ c w u z : BitVec n,
      (duoPair k₁ a r k₂ b q (c, w) (u ^^^ c ^^^ k₁, z + BitVec.twoPow n (n - 1))).2 =
        (duoPair k₁ a r k₂ b q (c, w) (u ^^^ c ^^^ k₁, z)).2 +
          ((((u * a + z) ^^^ w ^^^ k₂).rotateLeft q ^^^ BitVec.twoPow n (q - 1)) -
            ((u * a + z) ^^^ w ^^^ k₂).rotateLeft q) := by
    intro c w u z
    rw [duoPair_at, duoPair_at]
    dsimp only
    have s1 : (u * a + (z + BitVec.twoPow n (n - 1))) ^^^ w ^^^ k₂ =
        ((u * a + z) ^^^ w ^^^ k₂) ^^^ BitVec.twoPow n (n - 1) := by
      rw [← add_assoc, add_top hn, xor_xor_xor_top]
    rw [s1, rotateLeft_mul_top hn hb, rotateLeft_top hq0 hqn]
    linear_combination top_add_top hn
  -- so v and v′ have top bits that differ by bit q − 1 of κz on every start state
  have M : ∀ u z : BitVec n, ((u * a + z) ^^^ d₂ ^^^ k₂).getLsbD (n - 1) =
      ((((u ^^^ (d₁ ^^^ e₁)) * a + z) ^^^ e₂ ^^^ k₂).getLsbD (n - 1) ^^ κz.getLsbD (q - 1)) := by
    intro u z
    have e0 := HZ u z
    have e1 := HZ u (z + BitVec.twoPow n (n - 1))
    rw [duoPair_at_other k₁ a r k₂ b q d₁ e₁ e₂] at e0 e1
    rw [shiftZ, shiftZ] at e1
    have hbit := getLsbD_xor_of_shift (by omega : q - 1 + 1 < n) _ _ _ _ (e0.trans e1.symm)
    rw [e0, getLsbD_rotateLeft_pred hq0 hqn, getLsbD_rotateLeft_pred hq0 hqn] at hbit
    rw [hbit]
    have key : ∀ x y : Bool, x = (y ^^ (x ^^ y)) := by decide
    exact key _ _
  have hc := xor_eq_zero_or_top_of_const hn ha (d₁ ^^^ e₁) d₂ e₂ k₂ _ M
  have hct : d₁ ^^^ e₁ ≠ BitVec.twoPow n (n - 1) := by
    intro hct
    -- y₁′ = y₁ + t whatever u; subtracting the y coordinate from the z coordinate leaves v and y₂ alone
    have E1 : ∀ u z : BitVec n,
        (((u * a + z) + BitVec.twoPow n (n - 1)) ^^^ e₂ ^^^ k₂).rotateLeft q +
            ((((u * a + z) ^^^ d₂ ^^^ k₂) * b + (u.rotateLeft r + (u * a + z))) ^^^ κy) =
          (((u * a + z) ^^^ d₂ ^^^ k₂).rotateLeft q +
            (((u * a + z) ^^^ d₂ ^^^ k₂) * b + (u.rotateLeft r + (u * a + z)))) ^^^ κz := by
      intro u z
      have ey := HY u z
      have ez := HZ u z
      rw [duoPair_at_other k₁ a r k₂ b q d₁ e₁ e₂, duoPair_at, duoPair_at, hct] at ey ez
      dsimp only at ey ez
      have hY : (u ^^^ BitVec.twoPow n (n - 1)) * a + z = (u * a + z) + BitVec.twoPow n (n - 1) := by
        rw [← add_top hn u, add_mul, top_mul_odd hn ha]
        linear_combination
      rw [hY] at ey ez
      rw [eq_xor_of_xor_eq ey] at ez
      exact eq_xor_of_xor_eq ez
    have E1' : ∀ Y W : BitVec n,
        ((Y + BitVec.twoPow n (n - 1)) ^^^ e₂ ^^^ k₂).rotateLeft q + (W ^^^ κy) =
          ((Y ^^^ d₂ ^^^ k₂).rotateLeft q + W) ^^^ κz := by
      intro Y W
      obtain ⟨u, hu⟩ := rotateLeft_surjective (n := n) r (W - (Y ^^^ d₂ ^^^ k₂) * b - Y)
      simp only at hu
      have e := E1 u (Y - u * a)
      rw [show u * a + (Y - u * a) = Y by linear_combination, hu,
        show (Y ^^^ d₂ ^^^ k₂) * b + (W - (Y ^^^ d₂ ^^^ k₂) * b - Y + Y) = W by linear_combination] at e
      exact e
    have hv' : ∀ Y : BitVec n, ((Y + BitVec.twoPow n (n - 1)) ^^^ e₂ ^^^ k₂).rotateLeft q =
        (Y ^^^ d₂ ^^^ k₂).rotateLeft q ^^^
          (((d₂ ^^^ k₂) + BitVec.twoPow n (n - 1)) ^^^ e₂ ^^^ k₂).rotateLeft q := by
      intro Y
      rw [← rotateLeft_xor, add_top hn, add_top hn]
      congr 1
      apply BitVec.eq_of_getLsbD_eq
      intro i _
      simp only [BitVec.getLsbD_xor]
      cases Y.getLsbD i <;> cases d₂.getLsbD i <;> cases k₂.getLsbD i <;> cases e₂.getLsbD i <;>
        cases (BitVec.twoPow n (n - 1)).getLsbD i <;> rfl
    -- at v = 0: C = rotl(v′, q) has C + C = 0 and C = κy ⊕ κz
    have F : ∀ W : BitVec n,
        (((d₂ ^^^ k₂) + BitVec.twoPow n (n - 1)) ^^^ e₂ ^^^ k₂).rotateLeft q + (W ^^^ κy) = W ^^^ κz := by
      intro W
      have e := E1' (d₂ ^^^ k₂) W
      rwa [xor_xor_self_zero, zero_rotateLeft, BitVec.zero_add] at e
    generalize (((d₂ ^^^ k₂) + BitVec.twoPow n (n - 1)) ^^^ e₂ ^^^ k₂).rotateLeft q = C at F hv'
    have F0 := F 0#n
    have F1 := F κy
    have F2 := F κz
    simp only [BitVec.zero_xor, BitVec.xor_self, BitVec.add_zero] at F0 F1 F2
    rw [BitVec.xor_comm κz κy, ← F1] at F2
    have hC0 := self_add_self hn F2
    -- at W = 0 and rotl(v, q) = κz: κz + κz = 0
    obtain ⟨w, hw⟩ := rotateLeft_surjective (n := n) q κz
    simp only at hw
    have G := E1' (w ^^^ d₂ ^^^ k₂) 0#n
    rw [hv', xor_xor_xor_cancel, hw, xor_eq_add_of_zero_or_top hn hC0, BitVec.zero_xor, BitVec.add_zero,
      BitVec.xor_self] at G
    have hzz : κz + κz = 0#n := by linear_combination G - F0
    have hκz := self_add_self hn hzz
    have hκy : κy = 0#n ∨ κy = BitVec.twoPow n (n - 1) := by
      have e : κy = C ^^^ κz := by rw [F1, BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero]
      rw [e]
      rcases hC0 with h0 | h0 <;> rcases hκz with h1 | h1 <;> simp [h0, h1]
    -- (u, z) = (0, 0) and (t, t): the same y₁ and y₁′, z₁′ − z₁ apart by 2·2^(r−1)
    have g0 := HY 0 0
    have g1 := HY (BitVec.twoPow n (n - 1)) (BitVec.twoPow n (n - 1))
    rw [duoPair_at_other k₁ a r k₂ b q d₁ e₁ e₂, duoPair_at, duoPair_at, hct] at g0 g1
    dsimp only at g0 g1
    simp only [BitVec.ofNat_eq_ofNat, BitVec.zero_mul, BitVec.zero_add, BitVec.add_zero, BitVec.zero_xor,
      BitVec.xor_self, top_mul_odd hn ha, top_add_top hn, zero_rotateLeft] at g0 g1
    have f0 := eq_xor_of_xor_eq g0
    have f1 := eq_xor_of_xor_eq g1
    rw [xor_eq_add_of_zero_or_top hn hκy] at f0 f1
    have h2 : (BitVec.twoPow n (n - 1)).rotateLeft r + (BitVec.twoPow n (n - 1)).rotateLeft r = 0 := by
      linear_combination f0 - f1
    rw [rotateLeft_top hr0 hrn] at h2
    exact twoPow_add_twoPow_ne (by omega) h2
  have hd₁ : d₁ = e₁ := by
    rcases hc with h0 | ht
    · exact BitVec.xor_eq_zero_iff.mp h0
    · exact absurd ht hct
  subst hd₁
  -- the shared first word permutes the states: the second word's receipt (h = 1) has one XOR constant
  have hspread : duoWordSpread k₂ b q 1 d₂ = duoWordSpread k₂ b q 1 e₂ := by
    funext s s'
    obtain ⟨s₀, rfl⟩ := duoWord_state_surjective k₁ a r d₁ s
    obtain ⟨s₀', rfl⟩ := duoWord_state_surjective k₁ a r d₁ s'
    have e := congrFun (congrFun h s₀) s₀'
    simp only [stateSpread, duoPair, Prod.mk.injEq] at e
    simp only [duoWordSpread]
    rw [duoReceipt_xor, duoReceipt_xor, e.1, e.2]
  exact Prod.ext rfl (duoWordProj_injective k₂ ⟨b, hb⟩ ⟨q, hq0, hqn⟩ ⟨1, by omega, by omega⟩ hspread)

/-- fx1's two words: `duoPairXor_injective` on the words times odd multipliers. -/
theorem fx1PairXor_injective {n : ℕ} (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (m₁ : OddKey n)
    (k₂ : BitVec n) (a₂ : OddKey n) (r₂ : Rotation n) (m₂ : OddKey n) :
    Function.Injective (fun d => stateSpread (fx1Pair k₁ a₁.1 r₁.1 m₁.1 k₂ a₂.1 r₂.1 m₂.1 d)) := by
  intro d e h
  have h' := duoPairXor_injective k₁ a₁ r₁ k₂ a₂ r₂
    (show stateSpread (duoPair k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 (d.1 * m₁.1, d.2 * m₂.1)) =
      stateSpread (duoPair k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 (e.1 * m₁.1, e.2 * m₂.1)) from h)
  simp only [Prod.mk.injEq] at h'
  exact Prod.ext (mul_odd_injective m₁ h'.1) (mul_odd_injective m₂ h'.2)

end Pouw.ReceiptProofs
