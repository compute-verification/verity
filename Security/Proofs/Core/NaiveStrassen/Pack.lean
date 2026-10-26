/-!
# Packed rows (NaiveStrassen's kernel tables)

A row of a cost table is one natural number whose 64-bit fields are its entries. The kernel evaluates `>>>`, `&&&`, `+`
and `-` on such numbers as single GMP operations, so a comparison of two whole rows (`leRow`, one guard bit a field) is
a handful of kernel steps rather than one a field.

`getW w X j` is field `j` of `X` at width `w`, and `packW w f n` packs `f 0, …, f (n - 1)`. The kernel-facing forms
(`fld`, `row`, `packF`, `packRows`) are shifts and masks, each proved equal to its `getW` or `packW` form; the lemmas
here say what a field of a sum, a multiple, a mask or a packing is, and that `leRow` compares every field.
-/

namespace Verity.Accel.NaiveStrassen.Packed

/-- Field `j` of `X`, `w` bits wide. -/
def getW (w X j : Nat) : Nat := X / 2 ^ (w * j) % 2 ^ w

/-- `f 0, …, f (n - 1)` packed in `w`-bit fields. -/
def packW (w : Nat) (f : Nat → Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => packW w f n + 2 ^ (w * n) * f n

/-! ## Kernel-facing forms -/

/-- Field `j` of a row (64 bits). -/
def fld (X j : Nat) : Nat := (X >>> (64 * j)) &&& 18446744073709551615

/-- `f 0, …, f (n - 1)` in 64-bit fields, by `Nat.rec` (the kernel's fastest loop). -/
def packF (f : Nat → Nat) (n : Nat) : Nat :=
  Nat.rec (motive := fun _ => Nat) 0 (fun j acc => acc + (f j <<< (64 * j))) n

/-- `c` rows of `n` fields each. -/
def packRows (f : Nat → Nat) (n c : Nat) : Nat :=
  Nat.rec (motive := fun _ => Nat) 0 (fun i acc => acc + (f i <<< (64 * n * i))) c

/-- The mask of a row of `n` fields. -/
def rowMask (n : Nat) : Nat := 2 ^ (64 * n) - 1

/-- Row `i` of a table of `n`-field rows. -/
def row (T n i : Nat) : Nat := (T >>> (64 * n * i)) &&& rowMask n

/-- The guard bit (bit 63) of each of `n` fields. -/
def guards (n : Nat) : Nat := packF (fun _ => 9223372036854775808) n

/-- Bits 56–63 of each of `n` fields. -/
def highs (n : Nat) : Nat := packF (fun _ => 18374686479671623680) n

/-- Every field of `X` is at most the same field of `Y`, among the first `n` (when both are below `2^63`). -/
def leRow (n X Y : Nat) : Bool := ((Y + guards n) - X) &&& guards n == guards n

/-- Every one of `X`'s `n` fields is below `2^56`, and `X` has no other bits. -/
def small (n X : Nat) : Bool := X < 2 ^ (64 * n) && X &&& highs n == 0

/-! ## `getW` -/

theorem two_pow_mul_succ (w j : Nat) : 2 ^ (w * (j + 1)) = 2 ^ w * 2 ^ (w * j) := by
  rw [Nat.mul_succ, Nat.pow_add, Nat.mul_comm]

theorem getW_zero (w X : Nat) : getW w X 0 = X % 2 ^ w := by simp [getW]

theorem getW_succ (w X j : Nat) : getW w X (j + 1) = getW w (X / 2 ^ w) j := by
  simp only [getW, two_pow_mul_succ, Nat.div_div_eq_div_mul]

theorem getW_lt (w X j : Nat) : getW w X j < 2 ^ w := Nat.mod_lt _ (Nat.two_pow_pos w)

theorem getW_eq_zero {w X k : Nat} (h : X < 2 ^ (w * k)) {j : Nat} (hj : k ≤ j) : getW w X j = 0 := by
  have : X < 2 ^ (w * j) := Nat.lt_of_lt_of_le h (Nat.pow_le_pow_right (by decide) (Nat.mul_le_mul_left w hj))
  simp [getW, Nat.div_eq_of_lt this]

theorem getW_add_mul {w P k : Nat} (hP : P < 2 ^ (w * k)) (c j : Nat) :
    getW w (P + 2 ^ (w * k) * c) j = if j < k then getW w P j else getW w c (j - k) := by
  induction k generalizing P j with
  | zero =>
    have : P = 0 := by simpa using hP
    subst this
    simp
  | succ k ih =>
    have hw := Nat.two_pow_pos w
    have hsplit : P + 2 ^ (w * (k + 1)) * c = P % 2 ^ w + 2 ^ w * (P / 2 ^ w + 2 ^ (w * k) * c) := by
      rw [two_pow_mul_succ, Nat.mul_add, Nat.mul_assoc]
      have := Nat.mod_add_div P (2 ^ w)
      omega
    have hP' : P / 2 ^ w < 2 ^ (w * k) := by
      rw [Nat.div_lt_iff_lt_mul hw, Nat.mul_comm, ← two_pow_mul_succ]; exact hP
    cases j with
    | zero =>
      simp only [getW_zero, hsplit, Nat.add_mul_mod_self_left, Nat.mod_mod, Nat.zero_lt_succ, ite_true]
    | succ j =>
      rw [getW_succ, hsplit, Nat.add_mul_div_left _ _ hw, Nat.div_eq_of_lt (Nat.mod_lt _ hw), Nat.zero_add,
        ih hP', getW_succ]
      by_cases hj : j < k
      · simp [hj, Nat.succ_lt_succ hj]
      · simp [hj, Nat.succ_lt_succ_iff, Nat.succ_sub_succ]

theorem getW_mod {w X n j : Nat} : getW w (X % 2 ^ (w * n)) j = if j < n then getW w X j else 0 := by
  have hm : X % 2 ^ (w * n) < 2 ^ (w * n) := Nat.mod_lt _ (Nat.two_pow_pos _)
  by_cases hj : j < n
  · have := getW_add_mul hm (X / 2 ^ (w * n)) j
    rw [Nat.mod_add_div] at this
    simp [this, hj]
  · simp [hj, getW_eq_zero hm (Nat.le_of_not_lt hj)]

theorem getW_add {w : Nat} : ∀ {j X Y : Nat}, (∀ k ≤ j, getW w X k + getW w Y k < 2 ^ w) →
    getW w (X + Y) j = getW w X j + getW w Y j := by
  intro j
  induction j with
  | zero =>
    intro X Y h
    have := h 0 (Nat.le_refl 0)
    simp only [getW_zero] at this ⊢
    rw [Nat.add_mod, Nat.mod_eq_of_lt this]
  | succ j ih =>
    intro X Y h
    have hw := Nat.two_pow_pos w
    have h0 := h 0 (Nat.zero_le _)
    simp only [getW_zero] at h0
    have hdiv : (X + Y) / 2 ^ w = X / 2 ^ w + Y / 2 ^ w := by
      have hx := Nat.mod_add_div X (2 ^ w)
      have hy := Nat.mod_add_div Y (2 ^ w)
      have : X + Y = (X % 2 ^ w + Y % 2 ^ w) + 2 ^ w * (X / 2 ^ w + Y / 2 ^ w) := by
        rw [Nat.mul_add]; omega
      rw [this, Nat.add_mul_div_left _ _ hw, Nat.div_eq_of_lt h0, Nat.zero_add]
    rw [getW_succ, hdiv, ih, ← getW_succ, ← getW_succ]
    intro k hk
    have := h (k + 1) (Nat.succ_le_succ hk)
    rwa [getW_succ, getW_succ] at this

theorem getW_mul {w c : Nat} : ∀ {j X : Nat}, (∀ k ≤ j, c * getW w X k < 2 ^ w) →
    getW w (c * X) j = c * getW w X j := by
  intro j
  induction j with
  | zero =>
    intro X h
    have := h 0 (Nat.le_refl 0)
    simp only [getW_zero] at this ⊢
    have e : c * X = c * (X % 2 ^ w) + 2 ^ w * (c * (X / 2 ^ w)) := by
      conv => lhs; rw [← Nat.mod_add_div X (2 ^ w)]
      rw [Nat.mul_add, Nat.mul_left_comm]
    rw [e, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt this]
  | succ j ih =>
    intro X h
    have hw := Nat.two_pow_pos w
    have h0 := h 0 (Nat.zero_le _)
    simp only [getW_zero] at h0
    have hdiv : c * X / 2 ^ w = c * (X / 2 ^ w) := by
      have hx := Nat.mod_add_div X (2 ^ w)
      have : c * X = c * (X % 2 ^ w) + 2 ^ w * (c * (X / 2 ^ w)) := by
        conv => lhs; rw [← hx]
        rw [Nat.mul_add, Nat.mul_left_comm]
      rw [this, Nat.add_mul_div_left _ _ hw, Nat.div_eq_of_lt h0, Nat.zero_add]
    rw [getW_succ, hdiv, ih, ← getW_succ]
    intro k hk
    have := h (k + 1) (Nat.succ_le_succ hk)
    rwa [getW_succ] at this

theorem getW_and (w X M j : Nat) : getW w (X &&& M) j = getW w X j &&& getW w M j := by
  simp only [getW, Nat.and_div_two_pow, Nat.and_mod_two_pow]

/-! ## `packW` -/

theorem packW_lt {w : Nat} {f : Nat → Nat} : ∀ {n : Nat}, (∀ j < n, f j < 2 ^ w) → packW w f n < 2 ^ (w * n)
  | 0, _ => by simp [packW]
  | n + 1, h => by
    have ih := packW_lt (n := n) (fun j hj => h j (Nat.lt_succ_of_lt hj))
    have hf : f n + 1 ≤ 2 ^ w := h n (Nat.lt_succ_self n)
    have : 2 ^ (w * n) * (f n + 1) ≤ 2 ^ (w * n) * 2 ^ w := Nat.mul_le_mul_left _ hf
    rw [packW, two_pow_mul_succ, Nat.mul_comm (2 ^ w)]
    rw [Nat.mul_succ] at this
    omega

theorem getW_packW {w : Nat} {f : Nat → Nat} : ∀ {n : Nat}, (∀ j < n, f j < 2 ^ w) → ∀ j : Nat,
    getW w (packW w f n) j = if j < n then f j else 0
  | 0, _, j => by simp [packW, getW]
  | n + 1, h, j => by
    have h' : ∀ j < n, f j < 2 ^ w := fun j hj => h j (Nat.lt_succ_of_lt hj)
    rw [packW, getW_add_mul (packW_lt h')]
    by_cases hj : j < n
    · rw [ite_eq_left hj, getW_packW h', ite_eq_left hj, ite_eq_left (Nat.lt_succ_of_lt hj)]
    · rw [ite_eq_right hj]
      have hfn : f n < 2 ^ (w * 1) := by rw [Nat.mul_one]; exact h n (Nat.lt_succ_self n)
      by_cases hjn : j = n
      · subst hjn
        simp [getW_zero, Nat.mod_eq_of_lt (h j (Nat.lt_succ_self j))]
      · have : 1 ≤ j - n := by omega
        rw [getW_eq_zero hfn this, ite_eq_right (by omega)]

theorem packW_succ' (w : Nat) (f : Nat → Nat) : ∀ n : Nat,
    packW w f (n + 1) = f 0 + 2 ^ w * packW w (fun j => f (j + 1)) n
  | 0 => by simp [packW]
  | n + 1 => by
    rw [packW, packW_succ' w f n, packW, two_pow_mul_succ, Nat.mul_add, Nat.mul_assoc, Nat.add_assoc]

/-- A number whose `n` fields are each at most `c` is at most `c` packed `n` times. -/
theorem le_packW_const {w c : Nat} : ∀ {n X : Nat}, X < 2 ^ (w * n) → (∀ j < n, getW w X j ≤ c) →
    X ≤ packW w (fun _ => c) n
  | 0, X, hX, _ => by simp at hX; simp [hX, packW]
  | n + 1, X, hX, h => by
    have hw := Nat.two_pow_pos w
    rw [packW_succ']
    have h0 := h 0 (Nat.zero_lt_succ n)
    rw [getW_zero] at h0
    have hX' : X / 2 ^ w < 2 ^ (w * n) := by
      rw [Nat.div_lt_iff_lt_mul hw, Nat.mul_comm, ← two_pow_mul_succ]; exact hX
    have ih := le_packW_const (n := n) hX' (fun j hj => by
      have := h (j + 1) (Nat.succ_lt_succ hj); rwa [getW_succ] at this)
    have := Nat.mod_add_div X (2 ^ w)
    have := Nat.mul_le_mul_left (2 ^ w) ih
    omega

/-! ## The kernel forms are the `getW` and `packW` forms -/

theorem fld_eq (X j : Nat) : fld X j = getW 64 X j := by
  rw [fld, Nat.shiftRight_eq_div_pow, show (18446744073709551615 : Nat) = 2 ^ 64 - 1 by decide,
    Nat.and_two_pow_sub_one_eq_mod]
  rfl

theorem row_eq (T n i : Nat) : row T n i = getW (64 * n) T i := by
  rw [row, rowMask, Nat.shiftRight_eq_div_pow, Nat.and_two_pow_sub_one_eq_mod]
  rfl

theorem packF_eq (f : Nat → Nat) : ∀ n, packF f n = packW 64 f n
  | 0 => rfl
  | n + 1 => by
    show packF f n + (f n <<< (64 * n)) = _
    rw [packF_eq f n, Nat.shiftLeft_eq, Nat.mul_comm (f n)]
    rfl

theorem packRows_eq (f : Nat → Nat) (n : Nat) : ∀ c, packRows f n c = packW (64 * n) f c
  | 0 => rfl
  | c + 1 => by
    show packRows f n c + (f c <<< (64 * n * c)) = _
    rw [packRows_eq f n c, Nat.shiftLeft_eq, Nat.mul_comm (f c)]
    rfl

theorem and_rowMask (X n : Nat) : X &&& rowMask n = X % 2 ^ (64 * n) := by
  rw [rowMask, Nat.and_two_pow_sub_one_eq_mod]

/-! ## Fields of rows -/

theorem fld_lt (X j : Nat) : fld X j < 2 ^ 64 := by rw [fld_eq]; exact getW_lt _ _ _

theorem row_lt (T n i : Nat) : row T n i < 2 ^ (64 * n) := by rw [row_eq]; exact getW_lt _ _ _

theorem fld_row_ge {T n i j : Nat} (hj : n ≤ j) : fld (row T n i) j = 0 := by
  rw [fld_eq]; exact getW_eq_zero (k := n) (row_lt T n i) hj

theorem fld_packF {f : Nat → Nat} {n : Nat} (hf : ∀ j < n, f j < 2 ^ 64) (j : Nat) :
    fld (packF f n) j = if j < n then f j else 0 := by
  rw [fld_eq, packF_eq, getW_packW hf]

theorem row_packRows {f : Nat → Nat} {n c : Nat} (hf : ∀ i < c, f i < 2 ^ (64 * n)) (i : Nat) :
    row (packRows f n c) n i = if i < c then f i else 0 := by
  rw [row_eq, packRows_eq, getW_packW hf]

theorem fld_mask (X n j : Nat) : fld (X &&& rowMask n) j = if j < n then fld X j else 0 := by
  rw [and_rowMask, fld_eq, fld_eq, show 64 * n = 64 * n from rfl, getW_mod]

theorem fld_add {X Y n : Nat} (h : ∀ k < n, fld X k + fld Y k < 2 ^ 64) {j : Nat} (hj : j < n) :
    fld (X + Y) j = fld X j + fld Y j := by
  simp only [fld_eq] at *
  exact getW_add fun k hk => h k (Nat.lt_of_le_of_lt hk hj)

theorem fld_mul {c X n : Nat} (h : ∀ k < n, c * fld X k < 2 ^ 64) {j : Nat} (hj : j < n) :
    fld (c * X) j = c * fld X j := by
  simp only [fld_eq] at *
  exact getW_mul fun k hk => h k (Nat.lt_of_le_of_lt hk hj)

theorem fld_guards {n j : Nat} (hj : j < n) : fld (guards n) j = 9223372036854775808 := by
  rw [guards, fld_packF (fun _ _ => by decide), ite_eq_left hj]

theorem fld_highs {n j : Nat} (hj : j < n) : fld (highs n) j = 18374686479671623680 := by
  rw [highs, fld_packF (fun _ _ => by decide), ite_eq_left hj]

/-! ## `small` and `leRow` -/

theorem small_fld {n X : Nat} (h : small n X = true) : X < 2 ^ (64 * n) ∧ ∀ j < n, fld X j < 2 ^ 56 := by
  simp only [small, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  refine ⟨h.1, fun j hj => ?_⟩
  have h0 : fld X j &&& 18374686479671623680 = 0 := by
    have := congrArg (fun Z => getW 64 Z j) h.2
    simp only [getW_and, ← fld_eq, fld_highs hj] at this
    rw [this, fld_eq, getW, Nat.zero_div, Nat.zero_mod]
  have hx := fld_lt X j
  have hd : (fld X j &&& 18374686479671623680) / 2 ^ 56 = fld X j / 2 ^ 56 &&& 255 := by
    rw [Nat.and_div_two_pow]
  rw [h0, show (255 : Nat) = 2 ^ 8 - 1 by decide, Nat.and_two_pow_sub_one_eq_mod] at hd
  have : fld X j / 2 ^ 56 < 2 ^ 8 := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos 56)]; exact hx
  rw [Nat.mod_eq_of_lt this] at hd
  have : fld X j / 2 ^ 56 = 0 := by simpa using hd.symm
  exact (Nat.div_eq_zero_iff_lt (Nat.two_pow_pos 56)).mp this

theorem leRow_fld_aux : ∀ {n X Y Z : Nat}, X < 2 ^ (64 * n) → (∀ j < n, getW 64 X j < 2 ^ 63) →
    (∀ j < n, getW 64 Y j < 2 ^ 63) → Z + X = Y + guards n → Z &&& guards n = guards n →
    ∀ j < n, getW 64 X j ≤ getW 64 Y j
  | 0, _, _, _, _, _, _, _, _, j, hj => absurd hj (Nat.not_lt_zero j)
  | n + 1, X, Y, Z, hX, hXf, hYf, hsum, hand, j, hj => by
    have hB := Nat.two_pow_pos 64
    have hg : guards (n + 1) = 9223372036854775808 + 2 ^ 64 * guards n := by
      simp only [guards, packF_eq, packW_succ']
    -- low fields
    have hX0 := hXf 0 (Nat.zero_lt_succ n)
    have hY0 := hYf 0 (Nat.zero_lt_succ n)
    simp only [getW_zero] at hX0 hY0
    have hZ0 : 2 ^ 63 ≤ Z % 2 ^ 64 := by
      have := congrArg (fun W => getW 64 W 0) hand
      simp only [getW_and, ← fld_eq, fld_guards (Nat.zero_lt_succ n)] at this
      rw [fld_eq, getW_zero] at this
      have hle := Nat.and_le_left (n := Z % 2 ^ 64) (m := 9223372036854775808)
      rw [this] at hle
      exact hle
    have hZ := Nat.mod_add_div Z (2 ^ 64)
    have hXd := Nat.mod_add_div X (2 ^ 64)
    have hYd := Nat.mod_add_div Y (2 ^ 64)
    have hZlt := Nat.mod_lt Z hB
    -- the low sum carries nothing
    have hs : Z % 2 ^ 64 + X % 2 ^ 64 = Y % 2 ^ 64 + 2 ^ 63 ∧
        Z / 2 ^ 64 + X / 2 ^ 64 = Y / 2 ^ 64 + guards n := by
      rw [hg] at hsum
      have e1 : Z + X = (Z % 2 ^ 64 + X % 2 ^ 64) + 2 ^ 64 * (Z / 2 ^ 64 + X / 2 ^ 64) := by
        rw [Nat.mul_add]; omega
      have e2 : Y + (9223372036854775808 + 2 ^ 64 * guards n) =
          (Y % 2 ^ 64 + 2 ^ 63) + 2 ^ 64 * (Y / 2 ^ 64 + guards n) := by
        rw [Nat.mul_add]; omega
      rw [e1, e2] at hsum
      have m1 := congrArg (· % 2 ^ 64) hsum
      have d1 := congrArg (· / 2 ^ 64) hsum
      simp only [Nat.add_mul_mod_self_left, Nat.add_mul_div_left _ _ hB] at m1 d1
      have hlt : Y % 2 ^ 64 + 2 ^ 63 < 2 ^ 64 := by
        have : (2 : Nat) ^ 64 = 2 ^ 63 + 2 ^ 63 := by decide
        omega
      simp only [Nat.mod_eq_of_lt hlt, Nat.div_eq_of_lt hlt] at m1 d1
      have hlo := Nat.mod_add_div (Z % 2 ^ 64 + X % 2 ^ 64) (2 ^ 64)
      have hq : (Z % 2 ^ 64 + X % 2 ^ 64) / 2 ^ 64 = 0 := by
        have hle : (Z % 2 ^ 64 + X % 2 ^ 64) / 2 ^ 64 ≤ 1 := by
          rw [Nat.div_le_iff_le_mul_add_pred hB]
          have : (2 : Nat) ^ 64 = 2 ^ 63 + 2 ^ 63 := by decide
          omega
        have : (2 : Nat) ^ 64 = 2 ^ 63 + 2 ^ 63 := by decide
        rcases Nat.le_one_iff_eq_zero_or_eq_one.mp hle with h | h
        · exact h
        · rw [h] at hlo; omega
      rw [hq] at hlo d1
      constructor <;> omega
    cases j with
    | zero =>
      simp only [getW_zero]
      omega
    | succ j =>
      rw [getW_succ, getW_succ]
      refine leRow_fld_aux (Z := Z / 2 ^ 64) ?_ ?_ ?_ hs.2 ?_ j (Nat.lt_of_succ_lt_succ hj)
      · rw [Nat.div_lt_iff_lt_mul hB, Nat.mul_comm, ← two_pow_mul_succ]; exact hX
      · intro k hk; have := hXf (k + 1) (Nat.succ_lt_succ hk); rwa [getW_succ] at this
      · intro k hk; have := hYf (k + 1) (Nat.succ_lt_succ hk); rwa [getW_succ] at this
      · have := congrArg (· / 2 ^ 64) hand
        simp only [Nat.and_div_two_pow] at this
        rw [hg, Nat.add_mul_div_left _ _ hB,
          Nat.div_eq_of_lt (show (9223372036854775808 : Nat) < 2 ^ 64 by decide), Nat.zero_add] at this
        exact this

/-- `leRow` compares each of the first `n` fields, when every field compared is below `2^63` and `X` has no others. -/
theorem leRow_fld {n X Y : Nat} (h : leRow n X Y = true) (hX : X < 2 ^ (64 * n))
    (hXf : ∀ j < n, fld X j < 2 ^ 63) (hYf : ∀ j < n, fld Y j < 2 ^ 63) : ∀ j < n, fld X j ≤ fld Y j := by
  simp only [fld_eq] at *
  simp only [leRow, beq_iff_eq] at h
  have hG : X ≤ guards n := by
    rw [guards, packF_eq]
    exact le_packW_const hX fun j hj => Nat.le_of_lt (hXf j hj)
  exact leRow_fld_aux hX hXf hYf (by omega) h

end Verity.Accel.NaiveStrassen.Packed
