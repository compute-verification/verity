import Properties.Proofs.Pouw.Grid

/-!
# Proofs of the draw's properties (`Properties.Pouw`, "The draw")

`KPrimeLeast`, on Lean core: the fixed-point bounds `fixedPow` computes are sound (`fixedPow_sound`), so whatever they
decide is the exact answer and `missAtMost` is the integer comparison `a^k·dd ≤ dn·b^k` (`missAtMost_eq`), which is
`q^k ≤ δ` (`exact_iff`) and holds from some `k` on (`exact_mono`); the bisection finds the least such `k`
(`leastAbove_spec`).
-/

namespace Properties.Pouw.Window

open PouwWindow

/-! ### Bisection -/

theorem leastAbove_spec (pred : Nat → Bool) (mono : ∀ a b, a ≤ b → pred a = true → pred b = true) (lo hi : Nat)
    (hlt : lo < hi) (hhi : pred hi = true) :
    lo < leastAbove pred lo hi ∧ leastAbove pred lo hi ≤ hi ∧ pred (leastAbove pred lo hi) = true ∧
      ∀ j, lo < j → j < leastAbove pred lo hi → pred j = false := by
  fun_induction leastAbove pred lo hi with
  | case1 lo hi h mid hm ih =>
    obtain ⟨h1, h2, h3, h4⟩ := ih (by omega) hm
    exact ⟨h1, by omega, h3, h4⟩
  | case2 lo hi h mid hm ih =>
    obtain ⟨h1, h2, h3, h4⟩ := ih (by omega) hhi
    refine ⟨by omega, h2, h3, fun j hj hjk => ?_⟩
    by_cases hjm : j ≤ mid
    · cases hp : pred j
      · rfl
      · exact absurd (mono j mid hjm hp) (by simp [hm])
    · exact h4 j (by omega) hjk
  | case3 lo hi h =>
    exact ⟨hlt, Nat.le_refl _, hhi, fun j h1 h2 => by omega⟩

theorem leastUpTo_spec (pred : Nat → Bool) (mono : ∀ a b, a ≤ b → pred a = true → pred b = true) (n : Nat)
    (hn : 1 ≤ n) :
    1 ≤ leastUpTo pred n ∧ leastUpTo pred n ≤ n ∧ (leastUpTo pred n < n → pred (leastUpTo pred n) = true) ∧
      ∀ j, 1 ≤ j → j < leastUpTo pred n → pred j = false := by
  unfold leastUpTo
  split
  · rename_i h
    obtain ⟨h1, h2, h3, h4⟩ := leastAbove_spec pred mono 0 n (by omega) h
    exact ⟨h1, h2, fun _ => h3, fun j hj => h4 j hj⟩
  · rename_i h
    refine ⟨hn, Nat.le_refl _, fun h' => absurd h' (Nat.lt_irrefl _), fun j _ hj => ?_⟩
    cases hp : pred j
    · rfl
    · exact absurd (mono j n (Nat.le_of_lt hj) hp) h

/-! ### The fixed-point bounds -/

theorem ceilDiv_mul_ge {x d : Nat} (hd : 0 < d) : x ≤ ceilDiv x d * d := by
  unfold ceilDiv
  have h1 := Nat.div_add_mod (x + d - 1) d
  have h2 := Nat.mod_lt (x + d - 1) hd
  rw [Nat.mul_comm]; omega

theorem lower_mul {x y A1 A2 B1 B2 one : Nat} (hone : 0 < one) (hx : x * B1 ≤ A1 * one) (hy : y * B2 ≤ A2 * one) :
    x * y / one * (B1 * B2) ≤ A1 * A2 * one := by
  refine Nat.le_of_mul_le_mul_right ?_ hone
  calc x * y / one * (B1 * B2) * one = x * y / one * one * (B1 * B2) := Nat.mul_right_comm ..
    _ ≤ x * y * (B1 * B2) := Nat.mul_le_mul_right _ (Nat.div_mul_le_self _ _)
    _ = (x * B1) * (y * B2) := by grind
    _ ≤ (A1 * one) * (A2 * one) := Nat.mul_le_mul hx hy
    _ = A1 * A2 * one * one := by grind

theorem upper_mul {x y A1 A2 B1 B2 one : Nat} (hone : 0 < one) (hx : A1 * one ≤ x * B1) (hy : A2 * one ≤ y * B2) :
    A1 * A2 * one ≤ ceilDiv (x * y) one * (B1 * B2) := by
  refine Nat.le_of_mul_le_mul_right ?_ hone
  calc A1 * A2 * one * one = (A1 * one) * (A2 * one) := by grind
    _ ≤ (x * B1) * (y * B2) := Nat.mul_le_mul hx hy
    _ = x * y * (B1 * B2) := by grind
    _ ≤ ceilDiv (x * y) one * one * (B1 * B2) := Nat.mul_le_mul_right _ (ceilDiv_mul_ge hone)
    _ = ceilDiv (x * y) one * (B1 * B2) * one := Nat.mul_right_comm ..

/-- `fixedPowGo` keeps `lo ≤ (a/b)^j·one ≤ hi` for the exponent `j` it has multiplied in, with the base at `(a/b)^s`. -/
theorem go_sound (a b one : Nat) (hone : 0 < one) : ∀ fuel k lo hi bl bh j s, k < 2 ^ fuel →
    lo * b ^ j ≤ a ^ j * one → a ^ j * one ≤ hi * b ^ j → bl * b ^ s ≤ a ^ s * one → a ^ s * one ≤ bh * b ^ s →
    (fixedPowGo one fuel k lo hi bl bh).1 * b ^ (j + k * s) ≤ a ^ (j + k * s) * one ∧
      a ^ (j + k * s) * one ≤ (fixedPowGo one fuel k lo hi bl bh).2 * b ^ (j + k * s) := by
  intro fuel
  induction fuel with
  | zero =>
    intro k lo hi bl bh j s hk h1 h2 _ _
    have : k = 0 := by simp at hk; omega
    subst this
    simpa [fixedPowGo] using ⟨h1, h2⟩
  | succ fuel ih =>
    intro k lo hi bl bh j s hk h1 h2 h3 h4
    have hk' : k / 2 < 2 ^ fuel := by rw [Nat.pow_succ] at hk; omega
    have s3 : (bl * bl / one) * b ^ (s + s) ≤ a ^ (s + s) * one := by
      rw [Nat.pow_add, Nat.pow_add]; exact lower_mul hone h3 h3
    have s4 : a ^ (s + s) * one ≤ ceilDiv (bh * bh) one * b ^ (s + s) := by
      rw [Nat.pow_add, Nat.pow_add]; exact upper_mul hone h4 h4
    simp only [fixedPowGo]
    by_cases hodd : k % 2 = 1
    · simp only [hodd, ↓reduceIte]
      have he : j + k * s = (j + s) + k / 2 * (s + s) := by
        have : k = 2 * (k / 2) + 1 := by omega
        rw [this]; grind
      rw [he]
      refine ih (k / 2) _ _ _ _ (j + s) (s + s) hk' ?_ ?_ s3 s4
      · rw [Nat.pow_add, Nat.pow_add]; exact lower_mul hone h1 h3
      · rw [Nat.pow_add, Nat.pow_add]; exact upper_mul hone h2 h4
    · simp only [hodd, ↓reduceIte]
      have he : j + k * s = j + k / 2 * (s + s) := by
        have : k = 2 * (k / 2) := by omega
        rw [this]; grind
      rw [he]
      exact ih (k / 2) _ _ _ _ j (s + s) hk' h1 h2 s3 s4

theorem lt_two_pow_bitLength (k : Nat) : k < 2 ^ bitLength k := by
  rcases Nat.eq_zero_or_pos k with rfl | h
  · simp [bitLength]
  · exact (bitLength_spec h).2

theorem fixedPow_sound (a b k prec : Nat) (hb : 0 < b) :
    (fixedPow a b k prec).1 * b ^ k ≤ a ^ k * 2 ^ prec ∧ a ^ k * 2 ^ prec ≤ (fixedPow a b k prec).2 * b ^ k := by
  have := go_sound a b (2 ^ prec) (pow_pos' _) (bitLength k) k (2 ^ prec) (2 ^ prec) (a * 2 ^ prec / b)
    (ceilDiv (a * 2 ^ prec) b) 0 1 (lt_two_pow_bitLength k) (by simp) (by simp)
    (by simpa using Nat.div_mul_le_self (a * 2 ^ prec) b) (by simpa using ceilDiv_mul_ge (x := a * 2 ^ prec) hb)
  simpa [fixedPow] using this

/-- Whatever the bounds decide is the exact answer, so `missAtMostGo` is the exact comparison. -/
theorem missAtMostGo_eq (a b k dn dd : Nat) (hb : 0 < b) :
    ∀ fuel prec, missAtMostGo a b k dn dd fuel prec = missAtMostExact a b k dn dd := by
  intro fuel
  induction fuel with
  | zero => intro prec; rfl
  | succ fuel ih =>
    intro prec
    have hbk : 0 < b ^ k := Nat.pow_pos hb
    have hp := pow_pos' prec
    obtain ⟨s1, s2⟩ := fixedPow_sound a b k prec hb
    simp only [missAtMostGo]
    generalize fixedPow a b k prec = r at s1 s2
    obtain ⟨lo, hi⟩ := r
    simp only at s1 s2 ⊢
    split
    · rename_i h
      symm; unfold missAtMostExact; rw [decide_eq_true_iff]
      refine Nat.le_of_mul_le_mul_right ?_ hp
      calc a ^ k * dd * 2 ^ prec = a ^ k * 2 ^ prec * dd := Nat.mul_right_comm ..
        _ ≤ hi * b ^ k * dd := Nat.mul_le_mul_right _ s2
        _ = hi * dd * b ^ k := Nat.mul_right_comm ..
        _ ≤ dn * 2 ^ prec * b ^ k := Nat.mul_le_mul_right _ h
        _ = dn * b ^ k * 2 ^ prec := Nat.mul_right_comm ..
    · split
      · rename_i _ h
        symm; unfold missAtMostExact; rw [decide_eq_false_iff_not, Nat.not_le]
        refine Nat.lt_of_mul_lt_mul_right (a := 2 ^ prec) ?_
        calc dn * b ^ k * 2 ^ prec = dn * 2 ^ prec * b ^ k := Nat.mul_right_comm ..
          _ < lo * dd * b ^ k := Nat.mul_lt_mul_of_pos_right h hbk
          _ = lo * b ^ k * dd := Nat.mul_right_comm ..
          _ ≤ a ^ k * 2 ^ prec * dd := Nat.mul_le_mul_right _ s1
          _ = a ^ k * dd * 2 ^ prec := Nat.mul_right_comm ..
      · exact ih _

theorem missAtMost_eq (q δ : Rat) (k : Nat) :
    missAtMost q k δ = missAtMostExact q.num.toNat q.den k δ.num.toNat δ.den := by
  unfold missAtMost
  dsimp only
  split
  · rfl
  · exact missAtMostGo_eq _ _ _ _ _ q.den_pos _ _

theorem exact_iff {q δ : Rat} (hq : 0 ≤ q) (hδ : 0 ≤ δ) (k : Nat) :
    missAtMostExact q.num.toNat q.den k δ.num.toNat δ.den = true ↔ q ^ k ≤ δ := by
  rw [Rat.le_iff, Rat.num_pow, Rat.den_pow]
  unfold missAtMostExact
  rw [decide_eq_true_iff]
  have ha : (q.num.toNat : Int) = q.num := Int.toNat_of_nonneg (Rat.num_nonneg.2 hq)
  have hd : (δ.num.toNat : Int) = δ.num := Int.toNat_of_nonneg (Rat.num_nonneg.2 hδ)
  rw [← ha, ← hd]
  norm_cast

theorem exact_mono {a b dn dd : Nat} (hab : a ≤ b) {i j : Nat} (hij : i ≤ j)
    (h : missAtMostExact a b i dn dd = true) : missAtMostExact a b j dn dd = true := by
  induction j with
  | zero => have : i = 0 := by omega
            subst this; exact h
  | succ j ih =>
    rcases Nat.lt_or_eq_of_le hij with hij | rfl
    · have := ih (by omega)
      unfold missAtMostExact at this ⊢
      rw [decide_eq_true_iff] at this ⊢
      calc a ^ (j + 1) * dd = a * (a ^ j * dd) := by grind
        _ ≤ a * (dn * b ^ j) := Nat.mul_le_mul_left _ this
        _ ≤ b * (dn * b ^ j) := Nat.mul_le_mul_right _ hab
        _ = dn * b ^ (j + 1) := by grind
    · exact h

/-! ### K′ -/

theorem kPrimeLeast : KPrimeLeast := by
  intro eps δ credited n w hd
  have hd' := hd
  obtain ⟨he0, he1, hδ0, hδ1, hn, hw, hc0, hc1⟩ := hd
  dsimp only
  have hcR : (0 : Rat) < credited := by exact_mod_cast hc0
  have hmR : (0 : Rat) < (n : Rat) * w := Rat.mul_pos (by exact_mod_cast (by omega : (0 : Int) < n))
    (by exact_mod_cast (by omega : (0 : Int) < w))
  have hcm : (credited : Rat) ≤ (n : Rat) * w := by exact_mod_cast hc1
  have hp0 : 0 < missShare eps credited n w := by
    unfold missShare; rw [Rat.div_def]
    exact Rat.mul_pos (Rat.mul_pos he0 hcR) (Rat.inv_pos.2 hmR)
  have hp1 : missShare eps credited n w ≤ 1 := by
    rw [← Rat.not_lt]; intro h
    unfold missShare at h; rw [Rat.lt_div_iff hmR] at h
    have := Rat.mul_le_mul_of_nonneg_right he1 (Rat.le_of_lt hcR)
    grind
  have hnN : 1 ≤ n.toNat := by omega
  simp only [kPrime, hd', ↓reduceIte]
  by_cases hp : missShare eps credited n w = 1
  · simp only [hp, ↓reduceIte]
    refine ⟨1, rfl, Nat.le_refl _, hnN, fun _ => ?_, fun j h1 h2 => absurd h2 (by omega)⟩
    grind
  · simp only [hp, ↓reduceIte]
    generalize hq : 1 - missShare eps credited n w = q
    have hq0 : 0 ≤ q := by grind
    have hq1 : q ≤ 1 := by grind
    have hab : q.num.toNat ≤ q.den := by
      have h1 := (Rat.le_iff q 1).1 hq1
      have := Rat.num_nonneg.2 hq0
      rw [show (1 : Rat).num = 1 from rfl, show (1 : Rat).den = 1 from rfl] at h1
      omega
    have hpred : ∀ k, missAtMost q k δ = true ↔ q ^ k ≤ δ := fun k => by
      rw [missAtMost_eq]; exact exact_iff hq0 (Rat.le_of_lt hδ0) k
    have mono : ∀ i j, i ≤ j → missAtMost q i δ = true → missAtMost q j δ = true := by
      intro i j hij h; rw [missAtMost_eq] at h ⊢; exact exact_mono hab hij h
    obtain ⟨h1, h2, h3, h4⟩ := leastUpTo_spec (fun k => missAtMost q k δ) mono n.toNat hnN
    refine ⟨_, rfl, h1, h2, fun h => (hpred _).1 (h3 h), fun j hj hjk => ?_⟩
    have := h4 j hj hjk
    rw [← Rat.not_le]; intro h'; rw [← hpred] at h'; simp_all

end Properties.Pouw.Window
