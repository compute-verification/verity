import Properties.Pouw

/-!
# Proofs of the grid's properties (`Properties.Pouw`, "The grid")

On Lean core: `Nat.log2`'s bounds give a number's bit length, and so its shift; a point is a positive multiple of
`2 ^ shift` (`isPoint_iff`); `point` steps by `2 ^ gap ℓ` (`point_succ`), so it is strictly increasing, and `level`
inverts it. The floor point, the pad, the bucket and the levels follow.
-/

namespace Properties.Pouw.Window

open PouwWindow

/-! ### Bit lengths -/

theorem bitLength_spec {n : Nat} (h : 0 < n) : 2 ^ (bitLength n - 1) ≤ n ∧ n < 2 ^ bitLength n := by
  have h0 : n ≠ 0 := by omega
  simp only [bitLength, h0, ↓reduceIte, Nat.add_sub_cancel]
  exact ⟨Nat.log2_self_le h0, Nat.lt_log2_self⟩

theorem bitLength_le {n k : Nat} (h : n < 2 ^ k) : bitLength n ≤ k := by
  unfold bitLength
  split
  · omega
  · have := (Nat.log2_lt (by assumption)).2 h; omega

theorem lt_bitLength {n k : Nat} (h : 2 ^ k ≤ n) : k < bitLength n := by
  have h0 : n ≠ 0 := by have := Nat.two_pow_pos k; omega
  simp only [bitLength, h0, ↓reduceIte]
  have := (Nat.le_log2 h0).2 h; omega

theorem bitLength_eq {n k : Nat} (h1 : 2 ^ k ≤ n) (h2 : n < 2 ^ (k + 1)) : bitLength n = k + 1 := by
  have := bitLength_le h2; have := lt_bitLength h1; omega

theorem pow_pos' (k : Nat) : 0 < 2 ^ k := Nat.two_pow_pos k

/-- Below `2 ^ (shift n + bits + 1)`, and from `2 ^ (shift n + bits)` when the shift is positive. -/
theorem shift_spec {bits n : Nat} (h : 0 < n) :
    n < 2 ^ (shift bits n + bits + 1) ∧ (0 < shift bits n → 2 ^ (shift bits n + bits) ≤ n) := by
  obtain ⟨h1, h2⟩ := bitLength_spec h
  unfold shift
  refine ⟨Nat.lt_of_lt_of_le h2 (Nat.pow_le_pow_right (by omega) (by omega)), fun hs => ?_⟩
  exact Nat.le_trans (Nat.pow_le_pow_right (by omega) (by omega)) h1

theorem shift_le_of_lt {bits n e : Nat} (h : n < 2 ^ (bits + 1 + e)) : shift bits n ≤ e := by
  have := bitLength_le h; unfold shift; omega

/-! ### Floor points -/

theorem fp_le (bits n : Nat) : floorPoint bits n ≤ n := Nat.div_mul_le_self _ _

theorem dvd_fp (bits n : Nat) : 2 ^ shift bits n ∣ floorPoint bits n := Nat.dvd_mul_left _ _

theorem lt_fp_add (bits n : Nat) : n < floorPoint bits n + 2 ^ shift bits n := by
  have := Nat.div_add_mod n (2 ^ shift bits n)
  have := Nat.mod_lt n (pow_pos' (shift bits n))
  unfold floorPoint; rw [Nat.mul_comm]; omega

/-- A multiple of `2 ^ shift n` at most `n` is at most `floorPoint n`. -/
theorem le_fp {bits n a : Nat} (hd : 2 ^ shift bits n ∣ a) (h : a ≤ n) : a ≤ floorPoint bits n := by
  obtain ⟨c, rfl⟩ := hd
  unfold floorPoint
  rw [Nat.mul_comm (2 ^ _) c]
  refine Nat.mul_le_mul_right _ ?_
  exact (Nat.le_div_iff_mul_le (pow_pos' _)).2 (by rw [Nat.mul_comm]; exact h)

theorem fp_high {bits n : Nat} (h : 0 < n) : 2 ^ (bitLength n - 1) ≤ floorPoint bits n := by
  obtain ⟨h1, -⟩ := bitLength_spec h
  refine le_fp (Nat.pow_dvd_pow 2 ?_) h1
  have := bitLength_le (n := n) (k := bitLength n) (bitLength_spec h).2
  unfold shift; omega

theorem bitLength_fp {bits n : Nat} (h : 0 < n) : bitLength (floorPoint bits n) = bitLength n := by
  have h1 := fp_high (bits := bits) h
  have h2 := Nat.lt_of_le_of_lt (fp_le bits n) (bitLength_spec h).2
  have := bitLength_le h2
  have := lt_bitLength h1
  omega

theorem shift_fp {bits n : Nat} (h : 0 < n) : shift bits (floorPoint bits n) = shift bits n := by
  unfold shift; rw [bitLength_fp h]

theorem fp_pos {bits n : Nat} (h : 0 < n) : 0 < floorPoint bits n :=
  Nat.lt_of_lt_of_le (pow_pos' _) (fp_high h)

/-! ### Points -/

theorem isPoint_iff (bits n : Nat) : IsPoint bits n ↔ 0 < n ∧ 2 ^ shift bits n ∣ n := by
  constructor
  · rintro ⟨m, e, rfl, hm, hm'⟩
    refine ⟨Nat.mul_pos hm (pow_pos' e), Nat.dvd_trans (Nat.pow_dvd_pow 2 ?_) (Nat.dvd_mul_left _ _)⟩
    apply shift_le_of_lt
    rw [Nat.pow_add]
    exact Nat.mul_lt_mul_of_pos_right hm' (pow_pos' e)
  · rintro ⟨h, c, hc⟩
    have hs := (shift_spec (bits := bits) h).1
    generalize shift bits n = e at hc hs ⊢
    subst hc
    refine ⟨c, e, Nat.mul_comm _ _, Nat.pos_of_ne_zero (by rintro rfl; simp at h), ?_⟩
    rw [Nat.add_assoc, Nat.pow_add] at hs
    exact Nat.lt_of_mul_lt_mul_left hs

theorem onGridMeans : OnGridMeans := by
  intro bits n
  rw [isPoint_iff, onGrid, Bool.and_eq_true, decide_eq_true_iff, beq_iff_eq, Nat.dvd_iff_mod_eq_zero]

theorem isPoint_fp {bits n : Nat} (h : 0 < n) : IsPoint bits (floorPoint bits n) := by
  rw [isPoint_iff, shift_fp h]; exact ⟨fp_pos h, dvd_fp bits n⟩

theorem le_fp_of_isPoint {bits n g : Nat} (hg : IsPoint bits g) (h : g ≤ n) : g ≤ floorPoint bits n := by
  have hn : 0 < n := Nat.lt_of_lt_of_le ((isPoint_iff bits g).1 hg).1 h
  by_cases hl : g < 2 ^ (bitLength n - 1)
  · exact Nat.le_of_lt (Nat.lt_of_lt_of_le hl (fp_high hn))
  · have hb : bitLength g = bitLength n := by
      have := lt_bitLength (Nat.le_of_not_lt hl)
      have := bitLength_le (Nat.lt_of_le_of_lt h (bitLength_spec hn).2)
      omega
    have hs : shift bits g = shift bits n := by unfold shift; rw [hb]
    exact le_fp (hs ▸ ((isPoint_iff bits g).1 hg).2) h

/-- `floorPoint n` is the largest point at most `n ≥ 1`. -/
theorem floorPoint_greatest (bits n : Nat) (h : 1 ≤ n) :
    IsPoint bits (floorPoint bits n) ∧ floorPoint bits n ≤ n ∧ ∀ g, IsPoint bits g → g ≤ n → g ≤ floorPoint bits n :=
  ⟨isPoint_fp h, fp_le _ _, fun _ hg hle => le_fp_of_isPoint hg hle⟩

theorem fp_of_isPoint {bits g : Nat} (hg : IsPoint bits g) : floorPoint bits g = g :=
  Nat.le_antisymm (fp_le _ _) (le_fp_of_isPoint hg (Nat.le_refl _))


/-! ### `point` and `level` -/

/-- The step from `point ℓ` to `point (ℓ + 1)` is `2 ^ gap ℓ`. -/
def gap (bits ℓ : Nat) : Nat :=
  if ℓ < 2 ^ (bits + 1) - 1 then 0 else (ℓ - (2 ^ (bits + 1) - 1)) / 2 ^ bits + 1

theorem two_le_pow_succ (bits : Nat) : 2 ≤ 2 ^ (bits + 1) := by
  rw [Nat.pow_succ]; have := pow_pos' bits; omega

/-- A mantissa of `bits + 1` bits, its top bit set, shifted by `e`: its shift is `e`. -/
theorem mant_shift {bits m e : Nat} (h1 : 2 ^ bits ≤ m) (h2 : m < 2 ^ (bits + 1)) :
    shift bits (m * 2 ^ e) = e ∧ m * 2 ^ e / 2 ^ e = m := by
  refine ⟨?_, Nat.mul_div_cancel m (pow_pos' e)⟩
  have hb : bitLength (m * 2 ^ e) = bits + e + 1 := by
    apply bitLength_eq
    · rw [Nat.pow_add]; exact Nat.mul_le_mul_right _ h1
    · rw [Nat.add_right_comm, Nat.pow_add]; exact Nat.mul_lt_mul_of_pos_right h2 (pow_pos' e)
  unfold shift; omega

theorem divmod {X q t : Nat} (h : t < X) : (X * q + t) / X = q ∧ (X * q + t) % X = t := by
  rw [Nat.add_comm, Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt h, Nat.add_mul_mod_self_left,
    Nat.mod_eq_of_lt h]
  simp

theorem point_succ (bits ℓ : Nat) : point bits (ℓ + 1) = point bits ℓ + 2 ^ gap bits ℓ := by
  have hX := pow_pos' bits
  have hB : 2 ^ (bits + 1) = 2 ^ bits * 2 := Nat.pow_succ ..
  unfold point gap
  by_cases h1 : ℓ + 1 < 2 ^ (bits + 1) - 1
  · have h0 : ℓ < 2 ^ (bits + 1) - 1 := by omega
    simp [h1, h0]
  · by_cases h0 : ℓ < 2 ^ (bits + 1) - 1
    · have hr : ℓ + 1 - (2 ^ (bits + 1) - 1) = 0 := by omega
      simp only [h1, h0, ↓reduceIte, hr, Nat.zero_mod, Nat.zero_div, Nat.add_zero, Nat.pow_zero, Nat.zero_add,
        Nat.pow_one]
      omega
    · simp only [h1, h0, ↓reduceIte]
      generalize hr : ℓ - (2 ^ (bits + 1) - 1) = r
      have hr' : ℓ + 1 - (2 ^ (bits + 1) - 1) = r + 1 := by omega
      rw [hr']
      have hd := Nat.div_add_mod r (2 ^ bits)
      have ht := Nat.mod_lt r hX
      generalize r / 2 ^ bits = q at hd
      generalize r % 2 ^ bits = t at hd ht
      subst hd
      by_cases hc : t + 1 < 2 ^ bits
      · obtain ⟨d1, m1⟩ := divmod (q := q) hc
        rw [Nat.add_assoc]; simp only [d1, m1]
        rw [show 2 ^ bits + (t + 1) = (2 ^ bits + t) + 1 by omega, Nat.add_mul,
          Nat.one_mul]
      · obtain ⟨d1, m1⟩ := divmod (q := q + 1) hX
        have e1 : 2 ^ bits * q + t + 1 = 2 ^ bits * (q + 1) + 0 := by rw [Nat.mul_succ]; omega
        have ht' : 2 ^ bits + t + 1 = 2 ^ bits * 2 := by omega
        rw [e1]; simp only [d1, m1]
        rw [Nat.add_zero, Nat.pow_succ 2 (q + 1), ← Nat.mul_assoc]
        calc 2 ^ bits * 2 ^ (q + 1) * 2 = (2 ^ bits * 2) * 2 ^ (q + 1) := Nat.mul_right_comm ..
          _ = (2 ^ bits + t + 1) * 2 ^ (q + 1) := by rw [ht']
          _ = (2 ^ bits + t) * 2 ^ (q + 1) + 2 ^ (q + 1) := Nat.succ_mul ..

theorem point_lt_succ (bits ℓ : Nat) : point bits ℓ < point bits (ℓ + 1) := by
  rw [point_succ]; have := pow_pos' (gap bits ℓ); omega

theorem point_strictMono {bits a b : Nat} (h : a < b) : point bits a < point bits b := by
  induction b with
  | zero => omega
  | succ b ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | h
    · exact Nat.lt_trans (ih h) (point_lt_succ bits b)
    · subst h; exact point_lt_succ bits a

theorem point_mono {bits a b : Nat} (h : a ≤ b) : point bits a ≤ point bits b := by
  rcases Nat.lt_or_eq_of_le h with h | rfl
  · exact Nat.le_of_lt (point_strictMono h)
  · exact Nat.le_refl _

theorem point_zero (bits : Nat) : point bits 0 = 1 := by
  have := two_le_pow_succ bits
  simp [point, show 0 < 2 ^ (bits + 1) - 1 by omega]

theorem isPoint_point (bits ℓ : Nat) : IsPoint bits (point bits ℓ) := by
  have hX := pow_pos' bits
  unfold point
  split
  · exact ⟨ℓ + 1, 0, by simp, by omega, by omega⟩
  · refine ⟨_, _, rfl, by omega, ?_⟩
    have := Nat.mod_lt (ℓ - (2 ^ (bits + 1) - 1)) hX; rw [Nat.pow_succ]; omega

/-- `point ℓ`'s shift is its step. -/
theorem shift_point (bits ℓ : Nat) : shift bits (point bits ℓ) = gap bits ℓ := by
  have hX := pow_pos' bits
  have hB : 2 ^ (bits + 1) = 2 ^ bits * 2 := Nat.pow_succ ..
  unfold gap
  by_cases h : ℓ < 2 ^ (bits + 1) - 1
  · have hp : point bits ℓ = ℓ + 1 := by simp [point, h]
    rw [hp]; simp only [h, ↓reduceIte]
    have := bitLength_le (n := ℓ + 1) (k := bits + 1) (by omega)
    unfold shift; omega
  · simp only [point, h, ↓reduceIte]
    have := Nat.mod_lt (ℓ - (2 ^ (bits + 1) - 1)) hX
    exact (mant_shift (by omega) (by omega)).1

theorem level_point (bits ℓ : Nat) : level bits (point bits ℓ) = ℓ := by
  have hX := pow_pos' bits
  have hB : 2 ^ (bits + 1) = 2 ^ bits * 2 := Nat.pow_succ ..
  by_cases h : ℓ < 2 ^ (bits + 1) - 1
  · have hp : point bits ℓ = ℓ + 1 := by simp [point, h]
    rw [hp]; simp [level, show ℓ + 1 < 2 ^ (bits + 1) by omega]
  · have hp : point bits ℓ = (2 ^ bits + (ℓ - (2 ^ (bits + 1) - 1)) % 2 ^ bits) *
        2 ^ ((ℓ - (2 ^ (bits + 1) - 1)) / 2 ^ bits + 1) := by simp [point, h]
    generalize hr : ℓ - (2 ^ (bits + 1) - 1) = r at hp
    have ht := Nat.mod_lt r hX
    have hd := Nat.div_add_mod r (2 ^ bits)
    obtain ⟨hs, hm⟩ := mant_shift (bits := bits) (e := r / 2 ^ bits + 1) (m := 2 ^ bits + r % 2 ^ bits)
      (by omega) (by omega)
    rw [← hp] at hs hm
    have hge : ¬ point bits ℓ < 2 ^ (bits + 1) := by
      rw [hp, hB]
      refine Nat.not_lt.2 (Nat.mul_le_mul (by omega) ?_)
      exact Nat.le_trans (by simp) (Nat.pow_le_pow_right (by omega) (Nat.succ_le_succ (Nat.zero_le _)))
    simp only [level, hge, ↓reduceIte, hs, hm, Nat.add_sub_cancel]
    rw [Nat.mul_comm]; omega

theorem point_level {bits g : Nat} (hg : IsPoint bits g) : point bits (level bits g) = g := by
  obtain ⟨hpos, hd⟩ := (isPoint_iff bits g).1 hg
  have hX := pow_pos' bits
  have hB : 2 ^ (bits + 1) = 2 ^ bits * 2 := Nat.pow_succ ..
  by_cases h : g < 2 ^ (bits + 1)
  · simp [level, point, h, show g - 1 < 2 ^ (bits + 1) - 1 by omega]; omega
  · have hs : 0 < shift bits g := by
      have := lt_bitLength (Nat.le_of_not_lt h); unfold shift; omega
    obtain ⟨h1, h2⟩ := shift_spec (bits := bits) hpos
    have h2 := h2 hs
    generalize he : shift bits g = e at hs h1 h2 hd
    obtain ⟨m, rfl⟩ := hd
    have hm1 : 2 ^ bits ≤ m := by
      rw [Nat.pow_add] at h2; exact Nat.le_of_mul_le_mul_left h2 (pow_pos' e)
    have hm2 : m < 2 ^ (bits + 1) := by
      rw [Nat.add_assoc, Nat.pow_add] at h1; exact Nat.lt_of_mul_lt_mul_left h1
    have hdiv : 2 ^ e * m / 2 ^ e = m := Nat.mul_div_cancel_left m (pow_pos' e)
    simp only [level, h, ↓reduceIte, he, hdiv]
    have hl : ¬ (2 ^ (bits + 1) - 1 + (e - 1) * 2 ^ bits + m - 2 ^ bits < 2 ^ (bits + 1) - 1) := by omega
    simp only [point, hl, ↓reduceIte]
    have hr : 2 ^ (bits + 1) - 1 + (e - 1) * 2 ^ bits + m - 2 ^ bits - (2 ^ (bits + 1) - 1) =
        (m - 2 ^ bits) + 2 ^ bits * (e - 1) := by rw [Nat.mul_comm]; omega
    rw [hr, Nat.add_mul_mod_self_left, Nat.add_mul_div_left _ _ hX, Nat.mod_eq_of_lt (by omega),
      Nat.div_eq_of_lt (by omega), show 2 ^ bits + (m - 2 ^ bits) = m by omega,
      show 0 + (e - 1) + 1 = e by omega, Nat.mul_comm]

theorem pointEnumerates : PointEnumerates := fun bits =>
  ⟨point_zero bits, point_lt_succ bits, isPoint_point bits, level_point bits, fun _ hg => point_level hg⟩

/-- The point after a point `g` is at most every point above `g`. -/
theorem next_le {bits g h : Nat} (hg : IsPoint bits g) (hh : IsPoint bits h) (hlt : g < h) :
    point bits (level bits g + 1) ≤ h := by
  rw [← point_level hh]
  apply point_mono
  refine Nat.lt_of_not_le fun hle => ?_
  have := point_mono (bits := bits) hle
  rw [point_level hg, point_level hh] at this; omega

theorem lt_next {bits g : Nat} (hg : IsPoint bits g) : g < point bits (level bits g + 1) := by
  have := point_lt_succ bits (level bits g); rwa [point_level hg] at this

/-- `pad n` is the least point at least `n ≥ 1`. -/
theorem pad_least (bits n : Nat) (hn : 1 ≤ n) :
    IsPoint bits (pad bits n) ∧ n ≤ pad bits n ∧ ∀ g, IsPoint bits g → n ≤ g → pad bits n ≤ g := by
  have hfp := isPoint_fp (bits := bits) hn
  have hle := fp_le bits n
  unfold pad
  dsimp only
  split
  · rename_i heq
    rw [heq] at hfp ⊢
    exact ⟨hfp, Nat.le_refl _, fun _ _ h => h⟩
  · rename_i hne
    have hp := isPoint_point bits (level bits (floorPoint bits n) + 1)
    have hgt := lt_next hfp
    refine ⟨hp, ?_, fun g hg hng => next_le hfp hg (by omega)⟩
    exact Nat.le_of_not_lt fun h => Nat.lt_irrefl _ (Nat.lt_of_lt_of_le hgt (le_fp_of_isPoint hp (Nat.le_of_lt h)))

/-- A window of `real ≥ 1` real tiles is credited the largest point at most `real` and pads to the least point above it. -/
theorem bucket_credits (bits real : Nat) (hr : 1 ≤ real) :
    let w := bucket bits real
    w = Work.at bits w.level ∧ w.credited = floorPoint bits real ∧ w.credited ≤ real ∧ real < w.n ∧
      w.n = pad bits (real + 1) := by
  have hfp := isPoint_fp (bits := bits) hr
  have hp := isPoint_point bits (level bits (floorPoint bits real) + 1)
  have hgt := lt_next hfp
  have hn : real < point bits (level bits (floorPoint bits real) + 1) :=
    Nat.lt_of_not_le fun h => Nat.lt_irrefl _ (Nat.lt_of_lt_of_le hgt (le_fp_of_isPoint hp h))
  obtain ⟨hpad, hpad1, hpad2⟩ := pad_least bits (real + 1) (by omega)
  dsimp only [bucket, Work.at]
  refine ⟨rfl, point_level hfp, by rw [point_level hfp]; exact fp_le _ _, hn, Nat.le_antisymm ?_ (hpad2 _ hp hn)⟩
  exact next_le hfp hpad (by have := fp_le bits real; omega)

theorem cost_arith {R F Y X : Nat} (hX : 0 < X) (hR : R < F + Y) (hF : Y * X ≤ F) (hF0 : 0 < F) :
    R * (X - 1) < F * X ∧ (F + Y) * X ≤ F * (X + 1) := by
  constructor
  · obtain ⟨c, rfl⟩ : ∃ c, X = c + 1 := ⟨X - 1, by omega⟩
    rw [Nat.add_sub_cancel]
    rcases Nat.eq_zero_or_pos c with rfl | hc
    · simpa using hF0
    · have h1 : R * c < (F + Y) * c := Nat.mul_lt_mul_of_pos_right hR hc
      have h2 : Y * c ≤ Y * (c + 1) := Nat.mul_le_mul_left _ (by omega)
      rw [Nat.add_mul] at h1; rw [Nat.mul_succ]; omega
  · rw [Nat.add_mul, Nat.mul_succ]; omega

theorem bucket_cost (bits real : Nat) : CostBounded bits real (bucket bits real) := by
  intro hr
  have hX := pow_pos' bits
  have h0 : 0 < real := by omega
  have hfp := isPoint_fp (bits := bits) h0
  have hc : (bucket bits real).credited = floorPoint bits real := point_level hfp
  have hn : (bucket bits real).n = floorPoint bits real + 2 ^ shift bits real := by
    show point bits (level bits (floorPoint bits real) + 1) = _
    rw [point_succ, point_level hfp, ← shift_point, point_level hfp, shift_fp h0]
  have hF : 2 ^ shift bits real * 2 ^ bits ≤ floorPoint bits real := by
    by_cases hs : shift bits real = 0
    · rw [hs, Nat.pow_zero, Nat.one_mul]
      have : floorPoint bits real = real := by simp [floorPoint, hs]
      omega
    · have := fp_high (bits := bits) h0
      rw [← Nat.pow_add]
      exact Nat.le_trans (Nat.pow_le_pow_right (by omega) (by unfold shift at hs ⊢; omega)) this
  rw [hc, hn]
  exact cost_arith hX (lt_fp_add bits real) hF (fp_pos h0)

theorem bucket_cost3 (real : Nat) : CostBounded3 real (bucket gridBits real) := by
  intro hr
  have := bucket_cost gridBits real (by simpa [gridBits] using hr)
  simp only [gridBits, Nat.reducePow, Nat.reduceSub, Nat.reduceAdd] at this ⊢
  omega

theorem fp_mono {bits a b : Nat} (ha : 0 < a) (h : a ≤ b) : floorPoint bits a ≤ floorPoint bits b :=
  le_fp_of_isPoint (isPoint_fp ha) (Nat.le_trans (fp_le _ _) h)

theorem level_mono {bits g h : Nat} (hg : IsPoint bits g) (hh : IsPoint bits h) (hle : g ≤ h) :
    level bits g ≤ level bits h :=
  Nat.le_of_not_lt fun hlt => by
    have := point_strictMono (bits := bits) hlt
    rw [point_level hg, point_level hh] at this; omega

theorem levels_count (bits lo hi : Nat) (hlo : 1 ≤ lo) : LevelsCounted bits lo hi (levels bits lo hi) := by
  intro hle
  have hhi : 0 < hi := by omega
  have hm := level_mono (isPoint_fp (bits := bits) hlo) (isPoint_fp hhi) (fp_mono hlo hle)
  simp only [levels, bucket, Work.at]
  refine ⟨by omega, fun ℓ => ⟨?_, ?_⟩⟩
  · rintro ⟨r, h1, h2, rfl⟩
    have := level_mono (isPoint_fp (bits := bits) hlo) (isPoint_fp (by omega)) (fp_mono hlo h1)
    have := level_mono (isPoint_fp (bits := bits) (by omega : 0 < r)) (isPoint_fp hhi) (fp_mono (by omega) h2)
    omega
  · rintro ⟨h1, h2⟩
    have h2' : ℓ ≤ level bits (floorPoint bits hi) := by omega
    by_cases hp : point bits ℓ ≤ lo
    · refine ⟨lo, Nat.le_refl _, hle, ?_⟩
      have := level_mono (isPoint_point bits ℓ) (isPoint_fp hlo) (le_fp_of_isPoint (isPoint_point bits ℓ) hp)
      rw [level_point] at this; omega
    · refine ⟨point bits ℓ, by omega, ?_, ?_⟩
      · have := point_mono (bits := bits) h2'
        rw [point_level (isPoint_fp hhi)] at this
        exact Nat.le_trans this (fp_le _ _)
      · rw [fp_of_isPoint (isPoint_point bits ℓ), level_point]

end Properties.Pouw.Window
