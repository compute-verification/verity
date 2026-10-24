/-
PROTOCOL.md §2.1, §2.2, §2.4: the fields.
-/
namespace Flock

/-- GF(2^128) = F2[x]/(x^128 + x^7 + x^2 + x + 1). Bit `i` of `hi·2^64 + lo` is the coefficient of `x^i`. -/
structure F128 where
  lo : UInt64
  hi : UInt64
  deriving BEq, Inhabited, Repr, DecidableEq, Hashable

namespace F128

def zero : F128 := ⟨0, 0⟩
def one : F128 := ⟨1, 0⟩

@[inline] def add (a b : F128) : F128 := ⟨a.lo ^^^ b.lo, a.hi ^^^ b.hi⟩

instance instAdd : Add F128 := ⟨add⟩
instance instOfNatOfNatNat : OfNat F128 0 := ⟨zero⟩
instance instOfNatOfNatNat_1 : OfNat F128 1 := ⟨one⟩

@[inline] def isZero (a : F128) : Bool := a.lo == 0 && a.hi == 0

/-- Bits `i, i + 1, …` of the carry-less product: for each set bit of `bb` (bit `i` of the multiplier) the 128-bit value
`a · x^i` is added into `(lo, hi)`. -/
def clmulGo (a : UInt64) : Nat → Nat → UInt64 → UInt64 → UInt64 → UInt64 × UInt64
  | 0, _, _, lo, hi => (lo, hi)
  | fuel + 1, i, bb, lo, hi =>
    if bb &&& 1 = 1 then
      clmulGo a fuel (i + 1) (bb >>> 1) (lo ^^^ (a <<< i.toUInt64)) (if i = 0 then hi else hi ^^^ (a >>> (64 - i).toUInt64))
    else clmulGo a fuel (i + 1) (bb >>> 1) lo hi

/-- Carry-less product of two 64-bit words, as (low word, high word). -/
def clmul (a b : UInt64) : UInt64 × UInt64 := clmulGo a 64 0 b 0 0

/-- The carry-less product of two naturals: `a · x^i` added for each set bit `i` of `b`. -/
def clmulNat (a b : Nat) : Nat :=
  if b = 0 then 0 else (if b % 2 = 1 then a else 0) ^^^ (clmulNat a (b / 2) <<< 1)
termination_by b
decreasing_by omega

theorem clmulNat_step (a b : Nat) : clmulNat a b = (if b % 2 = 1 then a else 0) ^^^ (clmulNat a (b / 2) <<< 1) := by
  rw [clmulNat]
  split
  · rename_i h; subst h; rw [show (0 : Nat) / 2 = 0 from rfl, clmulNat]; simp
  · rfl

theorem clmulNat_zero (a : Nat) : clmulNat a 0 = 0 := by rw [clmulNat]; simp

theorem clmulNat_one (a : Nat) : clmulNat a 1 = a := by rw [clmulNat_step, show 1 / 2 = 0 from rfl, clmulNat_zero]; simp

theorem clmulNat_xor_left (a a' b : Nat) : clmulNat (a ^^^ a') b = clmulNat a b ^^^ clmulNat a' b := by
  rcases Nat.eq_zero_or_pos b with rfl | hb
  · simp [clmulNat_zero]
  · rw [clmulNat_step, clmulNat_step a, clmulNat_step a', clmulNat_xor_left a a' (b / 2), Nat.shiftLeft_xor_distrib]
    by_cases h : b % 2 = 1 <;> simp only [h, ite_true, ite_false, Nat.zero_xor] <;> ac_rfl
termination_by b
decreasing_by omega

theorem clmul_ite_xor_mod (a b b' : Nat) :
    (if (b ^^^ b') % 2 = 1 then a else 0) = (if b % 2 = 1 then a else 0) ^^^ (if b' % 2 = 1 then a else 0) := by
  have h := Nat.testBit_xor b b' 0
  simp only [Nat.testBit_zero] at h
  rcases Nat.mod_two_eq_zero_or_one b with hb | hb <;> rcases Nat.mod_two_eq_zero_or_one b' with hb' | hb' <;>
    simp_all

theorem clmulNat_xor_right (a b b' : Nat) : clmulNat a (b ^^^ b') = clmulNat a b ^^^ clmulNat a b' := by
  rcases Nat.eq_zero_or_pos (b + b') with h | h
  · have hb : b = 0 := by omega
    have hb' : b' = 0 := by omega
    subst hb hb'; simp [clmulNat_zero]
  · rw [clmulNat_step, clmulNat_step a b, clmulNat_step a b', Nat.xor_div_two, clmulNat_xor_right a (b / 2) (b' / 2),
      Nat.shiftLeft_xor_distrib, clmul_ite_xor_mod]
    ac_rfl
termination_by b + b'
decreasing_by omega

theorem clmulNat_shiftLeft_left (a b k : Nat) : clmulNat (a <<< k) b = clmulNat a b <<< k := by
  rcases Nat.eq_zero_or_pos b with rfl | hb
  · simp [clmulNat_zero]
  · rw [clmulNat_step, clmulNat_step a, clmulNat_shiftLeft_left a (b / 2) k, Nat.shiftLeft_xor_distrib,
      ← Nat.shiftLeft_add, ← Nat.shiftLeft_add, Nat.add_comm k 1]
    by_cases h : b % 2 = 1 <;> simp [h]
termination_by b
decreasing_by omega

theorem clmulNat_shiftLeft_right (a b : Nat) : ∀ k, clmulNat a (b <<< k) = clmulNat a b <<< k
  | 0 => by simp
  | k + 1 => by
    rw [Nat.shiftLeft_add, Nat.shiftLeft_add, ← clmulNat_shiftLeft_right a b k, clmulNat_step a]
    have h2 : (b <<< k <<< 1) % 2 = 0 := by rw [Nat.shiftLeft_eq]; omega
    have h3 : (b <<< k <<< 1) / 2 = b <<< k := by rw [Nat.shiftLeft_eq (b <<< k)]; omega
    simp [h2, h3]

theorem clmulNat_lt (a b ka : Nat) (ha : a < 2 ^ ka) : ∀ kb, b < 2 ^ kb → clmulNat a b < 2 ^ (ka + kb)
  | 0, hb => by
    have : b = 0 := by simpa using hb
    subst this; rw [clmulNat_zero]; exact Nat.two_pow_pos _
  | kb + 1, hb => by
    rw [clmulNat_step]
    have ih := clmulNat_lt a (b / 2) ka ha kb (by rw [Nat.pow_succ] at hb; omega)
    apply Nat.xor_lt_two_pow
    · split
      · exact Nat.lt_of_lt_of_le ha (Nat.pow_le_pow_right (by decide) (by omega))
      · exact Nat.two_pow_pos _
    · rw [Nat.shiftLeft_eq, Nat.pow_one, ← Nat.add_assoc, Nat.pow_succ]; omega

/-- Base-16 digit `m`: bits `4m` to `4m + 3`. -/
def clmulNib (v m : Nat) : Nat := (v >>> (4 * m)) % 16

theorem testBit_clmulNib (v p : Nat) : v.testBit p = (clmulNib v (p / 4)).testBit (p % 4) := by
  rw [clmulNib, show (16 : Nat) = 2 ^ 4 from rfl, Nat.testBit_mod_two_pow, Nat.testBit_shiftRight]
  simp only [Nat.mod_lt _ (by decide : 4 > 0), decide_true, Bool.true_and]
  congr 1; omega

theorem clmulNib_zero (v : Nat) : clmulNib v 0 = v % 16 := by simp [clmulNib]

theorem clmulNib_succ (v m : Nat) : clmulNib v (m + 1) = clmulNib (v / 16) m := by
  simp only [clmulNib, Nat.shiftRight_eq_div_pow, Nat.div_div_eq_div_mul]
  congr 2
  rw [show 4 * (m + 1) = 4 * m + 4 by omega, Nat.pow_add, Nat.mul_comm]

theorem clmulNib_add : ∀ (m u v : Nat), (∀ j, clmulNib u j + clmulNib v j < 16) →
    clmulNib (u + v) m = clmulNib u m + clmulNib v m
  | 0, u, v, h => by have := h 0; simp only [clmulNib_zero] at *; omega
  | m + 1, u, v, h => by
    have h0 := h 0
    simp only [clmulNib_zero] at h0
    rw [clmulNib_succ, clmulNib_succ, clmulNib_succ, show (u + v) / 16 = u / 16 + v / 16 by omega]
    exact clmulNib_add m _ _ fun j => by rw [← clmulNib_succ, ← clmulNib_succ]; exact h (j + 1)

theorem clmulNib_mul16_zero (w : Nat) : clmulNib (16 * w) 0 = 0 := by rw [clmulNib_zero]; omega

theorem clmulNib_mul16_succ (w m : Nat) : clmulNib (16 * w) (m + 1) = clmulNib w m := by
  rw [clmulNib_succ, show 16 * w / 16 = w by omega]

theorem clmulNib_lt (v m : Nat) : clmulNib v m < 16 := Nat.mod_lt _ (by decide)

/-- Every set bit at a multiple of 4. -/
def ClmulSpaced (x : Nat) : Prop := ∀ p, p % 4 ≠ 0 → x.testBit p = false

theorem clmulNib_le_one_of_bits :
    ∀ d, d < 16 → d.testBit 1 = false → d.testBit 2 = false → d.testBit 3 = false → d ≤ 1 := by
  decide

theorem clmulNib_le_one {x : Nat} (hx : ClmulSpaced x) (m : Nat) : clmulNib x m ≤ 1 := by
  have hb : ∀ k, k % 4 ≠ 0 → k < 4 → (clmulNib x m).testBit k = false := fun k hk h4 => by
    have := testBit_clmulNib x (4 * m + k)
    rw [show (4 * m + k) / 4 = m by omega, show (4 * m + k) % 4 = k by omega, hx _ (by omega)] at this
    exact this.symm
  exact clmulNib_le_one_of_bits _ (clmulNib_lt x m) (hb 1 (by decide) (by decide)) (hb 2 (by decide) (by decide))
    (hb 3 (by decide) (by decide))

theorem clmul_testBit_add_zero (x y : Nat) : (x + y).testBit 0 = (x.testBit 0 ^^ y.testBit 0) := by
  simp only [Nat.testBit_zero]
  rcases Nat.mod_two_eq_zero_or_one x with hx | hx <;> rcases Nat.mod_two_eq_zero_or_one y with hy | hy <;>
    simp [Nat.add_mod, hx, hy]

/-- On bit-spaced operands with fewer than 16 digits in `y`, no 4-bit field of the integer product overflows, so its
bits at multiples of 4 are the carry-less product's, which has no other bits. -/
theorem clmulNat_spaced (x : Nat) (hx : ClmulSpaced x) : ∀ n, n ≤ 15 → ∀ y, ClmulSpaced y → y < 16 ^ n →
    (∀ m, clmulNib (x * y) m ≤ n) ∧ (∀ m, (x * y).testBit (4 * m) = (clmulNat x y).testBit (4 * m)) ∧
      (∀ p, p % 4 ≠ 0 → (clmulNat x y).testBit p = false)
  | 0, _, y, _, hy => by
    have : y = 0 := by simpa using hy
    subst this
    refine ⟨fun m => ?_, fun m => ?_, fun p _ => ?_⟩ <;> simp [clmulNib, clmulNat_zero]
  | n + 1, hn, y, hsy, hy => by
    have hy0 : y % 16 ≤ 1 := by rw [← clmulNib_zero]; exact clmulNib_le_one hsy 0
    have hsy' : ClmulSpaced (y / 16) := fun p hp => by
      rw [show y / 16 = y >>> 4 by rw [Nat.shiftRight_eq_div_pow], Nat.testBit_shiftRight]
      exact hsy _ (by omega)
    obtain ⟨ihd, ihb, iho⟩ := clmulNat_spaced x hx n (by omega) (y / 16) hsy' (by rw [Nat.pow_succ] at hy; omega)
    let A := if y % 16 = 1 then x else 0
    have hyA : x * y = A + 16 * (x * (y / 16)) := by
      have e : y = y % 16 + 16 * (y / 16) := (Nat.mod_add_div y 16).symm
      rcases (by omega : y % 16 = 0 ∨ y % 16 = 1) with h | h
      · simp only [A, h]; conv => lhs; rw [e]
        rw [h, Nat.zero_add, Nat.mul_left_comm]; simp
      · simp only [A, h]; conv => lhs; rw [e]
        rw [h, Nat.mul_add, Nat.mul_one, Nat.mul_left_comm]; simp
    have hyx : y = y % 16 ^^^ (y / 16) <<< 4 := by
      apply Nat.eq_of_testBit_eq; intro p
      rw [Nat.testBit_xor, Nat.testBit_shiftLeft, show (16 : Nat) = 2 ^ 4 from rfl, Nat.testBit_mod_two_pow,
        Nat.testBit_div_two_pow]
      by_cases hp : p < 4
      · simp [hp, show ¬ p ≥ 4 by omega]
      · simp [hp, show p ≥ 4 by omega, show p - 4 + 4 = p by omega]
    have hcl : clmulNat x y = A ^^^ (clmulNat x (y / 16) <<< 4) := by
      conv => lhs; rw [hyx]
      rw [clmulNat_xor_right, clmulNat_shiftLeft_right]
      rcases (by omega : y % 16 = 0 ∨ y % 16 = 1) with h | h <;> simp [A, h, clmulNat_zero, clmulNat_one]
    have hA1 : ∀ m, clmulNib A m ≤ 1 := fun m => by
      by_cases h : y % 16 = 1
      · simp only [A, h, ite_true]; exact clmulNib_le_one hx m
      · simp only [A, h]; simp [clmulNib]
    have hAo : ∀ p, p % 4 ≠ 0 → A.testBit p = false := fun p hp => by
      by_cases h : y % 16 = 1
      · simp only [A, h, ite_true]; exact hx p hp
      · simp [A, h]
    have h16 : ∀ m, clmulNib (16 * (x * (y / 16))) m ≤ n := fun m => by
      cases m with
      | zero => rw [clmulNib_mul16_zero]; omega
      | succ m => rw [clmulNib_mul16_succ]; exact ihd m
    have hsum : ∀ j, clmulNib A j + clmulNib (16 * (x * (y / 16))) j < 16 := fun j => by
      have := hA1 j; have := h16 j; omega
    refine ⟨fun m => ?_, fun m => ?_, fun p hp => ?_⟩
    · rw [hyA, clmulNib_add m _ _ hsum]; have := hA1 m; have := h16 m; omega
    · rw [hyA, hcl, testBit_clmulNib, show 4 * m / 4 = m by omega, show 4 * m % 4 = 0 by omega,
        clmulNib_add m _ _ hsum, clmul_testBit_add_zero, Nat.testBit_xor, Nat.testBit_shiftLeft]
      congr 1
      · rw [testBit_clmulNib A (4 * m), show 4 * m / 4 = m by omega, show 4 * m % 4 = 0 by omega]
      · cases m with
        | zero => rw [clmulNib_mul16_zero]; simp
        | succ m =>
          rw [clmulNib_mul16_succ, show 4 * (m + 1) - 4 = 4 * m by omega, ← ihb m,
            testBit_clmulNib (x * (y / 16)) (4 * m),
            show 4 * m / 4 = m by omega, show 4 * m % 4 = 0 by omega]
          simp only [show 4 * (m + 1) ≥ 4 by omega, decide_true, Bool.true_and]
    · rw [hcl, Nat.testBit_xor, hAo p hp, Nat.testBit_shiftLeft]
      by_cases h4 : p ≥ 4
      · simp [h4, iho (p - 4) (by omega)]
      · simp [h4]

theorem clmulNat_mul_spaced (a b x y : Nat) (ha : a < 4) (hb : b < 4) (hx : ∀ p, p % 4 ≠ a → x.testBit p = false)
    (hy : ∀ p, p % 4 ≠ b → y.testBit p = false) (hy32 : y < 2 ^ 32) (p : Nat) :
    ((x * y).testBit p && (p % 4 == (a + b) % 4)) = (clmulNat x y).testBit p := by
  have shl : ∀ (v r : Nat), r < 4 → (∀ p, p % 4 ≠ r → v.testBit p = false) → v = (v >>> r) <<< r :=
    fun v r hr hv => by
    apply Nat.eq_of_testBit_eq; intro q
    rw [Nat.testBit_shiftLeft, Nat.testBit_shiftRight]
    by_cases hq : q ≥ r
    · simp [hq, show r + (q - r) = q by omega]
    · simp [hq, hv q (by omega)]
  have sp : ∀ (v r : Nat), (∀ p, p % 4 ≠ r → v.testBit p = false) → ClmulSpaced (v >>> r) :=
    fun v r hv q hq => by
    rw [Nat.testBit_shiftRight]; exact hv _ (by omega)
  obtain ⟨-, hbit, hoff⟩ := clmulNat_spaced (x >>> a) (sp x a hx) 8 (by decide) (y >>> b) (sp y b hy)
    (Nat.lt_of_le_of_lt (Nat.shiftRight_le _ _) hy32)
  have hm : x * y = ((x >>> a) * (y >>> b)) <<< (a + b) := by
    conv => lhs; rw [shl x a ha hx, shl y b hb hy]
    simp only [Nat.shiftLeft_eq, Nat.pow_add]
    rw [Nat.mul_mul_mul_comm]
  have hc : clmulNat x y = clmulNat (x >>> a) (y >>> b) <<< (a + b) := by
    conv => lhs; rw [shl x a ha hx, shl y b hb hy]
    rw [clmulNat_shiftLeft_left, clmulNat_shiftLeft_right, ← Nat.shiftLeft_add, Nat.add_comm b a]
  rw [hm, hc, Nat.testBit_shiftLeft, Nat.testBit_shiftLeft]
  by_cases hp : p ≥ a + b
  · by_cases hq : (p - (a + b)) % 4 = 0
    · have e := hbit ((p - (a + b)) / 4)
      rw [show 4 * ((p - (a + b)) / 4) = p - (a + b) by omega] at e
      simp [hp, e, show p % 4 = (a + b) % 4 by omega]
    · simp [hp, hoff _ hq, show ¬ p % 4 = (a + b) % 4 by omega]
  · simp [hp]

/-- The carry-less product of the low 32 bits of `x` and `y`, from sixteen integer products of `x`, `y` masked to every
4th bit: each 4-bit field of such a product sums at most 8 one-bit terms, so it never carries
(`clmulNat_mul_spaced`). -/
@[inline] def bmul32 (x y : UInt64) : UInt64 :=
  let x0 := x &&& 0x11111111
  let x1 := x &&& 0x22222222
  let x2 := x &&& 0x44444444
  let x3 := x &&& 0x88888888
  let y0 := y &&& 0x11111111
  let y1 := y &&& 0x22222222
  let y2 := y &&& 0x44444444
  let y3 := y &&& 0x88888888
  ((x0 * y0 ^^^ x1 * y3 ^^^ x2 * y2 ^^^ x3 * y1) &&& 0x1111111111111111) |||
    ((x0 * y1 ^^^ x1 * y0 ^^^ x2 * y3 ^^^ x3 * y2) &&& 0x2222222222222222) |||
    ((x0 * y2 ^^^ x1 * y1 ^^^ x2 * y0 ^^^ x3 * y3) &&& 0x4444444444444444) |||
    ((x0 * y3 ^^^ x1 * y2 ^^^ x2 * y1 ^^^ x3 * y0) &&& 0x8888888888888888)

theorem clmulMask64 (c : Nat) (hc : c < 2 ^ 64) (r : Nat) (h : ∀ p, p < 64 → c.testBit p = (p % 4 == r)) (p : Nat) :
    c.testBit p = (decide (p < 64) && p % 4 == r) := by
  by_cases hp : p < 64
  · simp [hp, h p hp]
  · simp only [hp, decide_false, Bool.false_and]
    exact Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hc (Nat.pow_le_pow_right (by decide) (by omega)))

theorem clmulMask32 (c : Nat) (hc : c < 2 ^ 32) (r : Nat) (h : ∀ p, p < 32 → c.testBit p = (p % 4 == r)) (p : Nat) :
    c.testBit p = (decide (p < 32) && p % 4 == r) := by
  by_cases hp : p < 32
  · simp [hp, h p hp]
  · simp only [hp, decide_false, Bool.false_and]
    exact Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hc (Nat.pow_le_pow_right (by decide) (by omega)))

theorem bmul32_eq (x y : UInt64) : (bmul32 x y).toNat = clmulNat (x.toNat % 2 ^ 32) (y.toNat % 2 ^ 32) := by
  have m1 := clmulMask32 0x11111111 (by decide) 0 (by decide)
  have m2 := clmulMask32 0x22222222 (by decide) 1 (by decide)
  have m4 := clmulMask32 0x44444444 (by decide) 2 (by decide)
  have m8 := clmulMask32 0x88888888 (by decide) 3 (by decide)
  have n1 := clmulMask64 0x1111111111111111 (by decide) 0 (by decide)
  have n2 := clmulMask64 0x2222222222222222 (by decide) 1 (by decide)
  have n4 := clmulMask64 0x4444444444444444 (by decide) 2 (by decide)
  have n8 := clmulMask64 0x8888888888888888 (by decide) 3 (by decide)
  have lt32 : ∀ (v : UInt64) (c : Nat), c < 2 ^ 32 → v.toNat &&& c < 2 ^ 32 := fun v c hc => Nat.and_lt_two_pow _ hc
  have mulN : ∀ (u v : UInt64), u.toNat < 2 ^ 32 → v.toNat < 2 ^ 32 → (u * v).toNat = u.toNat * v.toNat :=
    fun u v hu hv => by
      rw [UInt64.toNat_mul, Nat.mod_eq_of_lt]
      exact Nat.lt_of_lt_of_le (Nat.mul_lt_mul_of_lt_of_lt hu hv) (by decide)
  have off : ∀ (v : UInt64) (c r : Nat), (∀ p, c.testBit p = (decide (p < 32) && p % 4 == r)) →
      ∀ p, p % 4 ≠ r → (v.toNat &&& c).testBit p = false := fun v c r hc p hp => by
    rw [Nat.testBit_and, hc]; simp [hp]
  let X := fun c => x.toNat &&& c
  let Y := fun c => y.toNat &&& c
  have Q := fun (a b ca cb : Nat) (ha : a < 4) (hb : b < 4) (hca : ca < 2 ^ 32) (hcb : cb < 2 ^ 32)
    (mca : ∀ p, ca.testBit p = (decide (p < 32) && p % 4 == a))
    (mcb : ∀ p, cb.testBit p = (decide (p < 32) && p % 4 == b)) =>
    clmulNat_mul_spaced a b (X ca) (Y cb) ha hb (off x ca a mca) (off y cb b mcb) (lt32 y cb hcb)
  have hxs : x.toNat % 2 ^ 32 = X 0x11111111 ^^^ X 0x22222222 ^^^ X 0x44444444 ^^^ X 0x88888888 := by
    apply Nat.eq_of_testBit_eq; intro p
    simp only [X, Nat.testBit_xor, Nat.testBit_and, Nat.testBit_mod_two_pow, m1, m2, m4, m8]
    by_cases hp : p < 32 <;> simp [hp] <;> rcases (by omega : p % 4 = 0 ∨ p % 4 = 1 ∨ p % 4 = 2 ∨ p % 4 = 3)
      with h | h | h | h <;> simp [h]
  have hys : y.toNat % 2 ^ 32 = Y 0x11111111 ^^^ Y 0x22222222 ^^^ Y 0x44444444 ^^^ Y 0x88888888 := by
    apply Nat.eq_of_testBit_eq; intro p
    simp only [Y, Nat.testBit_xor, Nat.testBit_and, Nat.testBit_mod_two_pow, m1, m2, m4, m8]
    by_cases hp : p < 32 <;> simp [hp] <;> rcases (by omega : p % 4 = 0 ∨ p % 4 = 1 ∨ p % 4 = 2 ∨ p % 4 = 3)
      with h | h | h | h <;> simp [h]
  rw [hxs, hys]
  simp only [clmulNat_xor_left, clmulNat_xor_right]
  have P : ∀ (ca cb : UInt64), ca.toNat < 2 ^ 32 → cb.toNat < 2 ^ 32 →
      ((x &&& ca) * (y &&& cb)).toNat = X ca.toNat * Y cb.toNat := fun ca cb hca hcb => by
    rw [mulN _ _ (by rw [UInt64.toNat_and]; exact lt32 x _ hca) (by rw [UInt64.toNat_and]; exact lt32 y _ hcb),
      UInt64.toNat_and, UInt64.toNat_and]
  apply Nat.eq_of_testBit_eq; intro p
  simp only [bmul32, UInt64.toNat_or, UInt64.toNat_and, UInt64.toNat_xor, Nat.testBit_or, Nat.testBit_and,
    Nat.testBit_xor]
  rw [← Q 0 0 0x11111111 0x11111111 (by decide) (by decide) (by decide) (by decide) m1 m1,
    ← Q 0 1 0x11111111 0x22222222 (by decide) (by decide) (by decide) (by decide) m1 m2,
    ← Q 0 2 0x11111111 0x44444444 (by decide) (by decide) (by decide) (by decide) m1 m4,
    ← Q 0 3 0x11111111 0x88888888 (by decide) (by decide) (by decide) (by decide) m1 m8,
    ← Q 1 0 0x22222222 0x11111111 (by decide) (by decide) (by decide) (by decide) m2 m1,
    ← Q 1 1 0x22222222 0x22222222 (by decide) (by decide) (by decide) (by decide) m2 m2,
    ← Q 1 2 0x22222222 0x44444444 (by decide) (by decide) (by decide) (by decide) m2 m4,
    ← Q 1 3 0x22222222 0x88888888 (by decide) (by decide) (by decide) (by decide) m2 m8,
    ← Q 2 0 0x44444444 0x11111111 (by decide) (by decide) (by decide) (by decide) m4 m1,
    ← Q 2 1 0x44444444 0x22222222 (by decide) (by decide) (by decide) (by decide) m4 m2,
    ← Q 2 2 0x44444444 0x44444444 (by decide) (by decide) (by decide) (by decide) m4 m4,
    ← Q 2 3 0x44444444 0x88888888 (by decide) (by decide) (by decide) (by decide) m4 m8,
    ← Q 3 0 0x88888888 0x11111111 (by decide) (by decide) (by decide) (by decide) m8 m1,
    ← Q 3 1 0x88888888 0x22222222 (by decide) (by decide) (by decide) (by decide) m8 m2,
    ← Q 3 2 0x88888888 0x44444444 (by decide) (by decide) (by decide) (by decide) m8 m4,
    ← Q 3 3 0x88888888 0x88888888 (by decide) (by decide) (by decide) (by decide) m8 m8]
  repeat rw [P _ _ (by decide) (by decide)]
  simp only [X, Y, UInt64.reduceToNat, n1, n2, n4, n8]
  have : Std.Commutative (α := Bool) (· != ·) := ⟨by decide⟩
  by_cases hp : p < 64
  · simp only [hp, decide_true, Bool.true_and]
    rcases (by omega : p % 4 = 0 ∨ p % 4 = 1 ∨ p % 4 = 2 ∨ p % 4 = 3) with h | h | h | h <;> simp [h] <;> ac_rfl
  · have pb : ∀ ca cb : Nat, ca < 2 ^ 32 → cb < 2 ^ 32 →
        ((x.toNat &&& ca) * (y.toNat &&& cb)).testBit p = false :=
      fun ca cb hca hcb => Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le
        (Nat.mul_lt_mul_of_lt_of_lt (lt32 x ca hca) (lt32 y cb hcb))
        (show 2 ^ 32 * 2 ^ 32 ≤ 2 ^ p by rw [← Nat.pow_add]; exact Nat.pow_le_pow_right (by decide) (by omega)))
    simp [hp, pb]

/-- The 128-bit value of a (low word, high word) pair. -/
def clmulPair (r : UInt64 × UInt64) : Nat := r.1.toNat ^^^ (r.2.toNat <<< 64)

theorem testBit_clmulPair (r : UInt64 × UInt64) (p : Nat) :
    (clmulPair r).testBit p = if p < 64 then r.1.toNat.testBit p else r.2.toNat.testBit (p - 64) := by
  rw [clmulPair, Nat.testBit_xor, Nat.testBit_shiftLeft]
  by_cases hp : p < 64
  · simp [hp, show ¬ p ≥ 64 by omega]
  · simp [hp, show p ≥ 64 by omega, Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le r.1.toNat_lt
      (Nat.pow_le_pow_right (by decide) (by omega) : 2 ^ 64 ≤ 2 ^ p))]

theorem clmulPair_inj {r s : UInt64 × UInt64} (h : clmulPair r = clmulPair s) : r = s := by
  have hb := fun p => congrArg (Nat.testBit · p) h
  simp only [testBit_clmulPair] at hb
  obtain ⟨r1, r2⟩ := r
  obtain ⟨s1, s2⟩ := s
  have e1 : r1.toNat = s1.toNat := Nat.eq_of_testBit_eq fun p => by
    by_cases hp : p < 64
    · simpa [hp] using hb p
    · rw [Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le r1.toNat_lt (Nat.pow_le_pow_right (by decide) (by omega))),
        Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le s1.toNat_lt (Nat.pow_le_pow_right (by decide) (by omega)))]
  have e2 : r2.toNat = s2.toNat := Nat.eq_of_testBit_eq fun p => by
    simpa [show ¬ p + 64 < 64 by omega] using hb (p + 64)
  rw [UInt64.toNat_inj.mp e1, UInt64.toNat_inj.mp e2]

theorem clmulGo_clmulNat (a : UInt64) (n : Nat) : ∀ (i : Nat) (bb lo hi : UInt64), i + n ≤ 64 →
    clmulPair (clmulGo a n i bb lo hi) = clmulPair (lo, hi) ^^^ (clmulNat a.toNat (bb.toNat % 2 ^ n) <<< i) := by
  induction n with
  | zero => intro i bb lo hi _; simp [clmulGo, Nat.mod_one, clmulNat_zero]
  | succ n ih =>
    intro i bb lo hi h
    have hstep : clmulNat a.toNat (bb.toNat % 2 ^ (n + 1)) =
        (if bb.toNat % 2 = 1 then a.toNat else 0) ^^^ (clmulNat a.toNat ((bb >>> 1).toNat % 2 ^ n) <<< 1) := by
      rw [clmulNat_step, Nat.pow_succ, Nat.mul_comm, Nat.mod_mul_right_mod, Nat.mod_mul_right_div_self,
        UInt64.toNat_shiftRight, show (1 : UInt64).toNat % 64 = 1 from rfl, Nat.shiftRight_eq_div_pow, Nat.pow_one]
    have hb : (bb &&& 1 = 1) ↔ bb.toNat % 2 = 1 := by
      rw [← UInt64.toNat_inj, UInt64.toNat_and, show (1 : UInt64).toNat = 1 from rfl, Nat.and_one_is_mod]
    have hiu : i.toUInt64.toNat = i := by simp [Nat.toUInt64]; omega
    rw [hstep, Nat.shiftLeft_xor_distrib, ← Nat.shiftLeft_add, Nat.add_comm 1 i]
    by_cases hbb : bb &&& 1 = 1
    · rw [clmulGo]
      simp only [hbb, hb.mp hbb, ↓reduceIte]
      have key : clmulPair ((lo ^^^ (a <<< i.toUInt64)), (if i = 0 then hi else hi ^^^ (a >>> (64 - i).toUInt64))) =
          clmulPair (lo, hi) ^^^ (a.toNat <<< i) := by
        apply Nat.eq_of_testBit_eq; intro p
        simp only [testBit_clmulPair, Nat.testBit_xor, Nat.testBit_shiftLeft, UInt64.toNat_xor, UInt64.toNat_shiftLeft,
          Nat.testBit_mod_two_pow, hiu, Nat.mod_eq_of_lt (show i < 64 by omega)]
        by_cases hi0 : i = 0
        · subst hi0
          by_cases hp : p < 64
          · simp [hp]
          · simp [hp, Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le a.toNat_lt
              (Nat.pow_le_pow_right (by decide) (by omega) : 2 ^ 64 ≤ 2 ^ p))]
        · have h64 : (64 - i).toUInt64.toNat % 64 = 64 - i := by simp [Nat.toUInt64]; omega
          simp only [hi0, ↓reduceIte, UInt64.toNat_xor, UInt64.toNat_shiftRight, Nat.testBit_xor,
            Nat.testBit_shiftRight, h64]
          by_cases hp : p < 64
          · simp [hp]
          · simp [hp, show p ≥ i by omega, show 64 - i + (p - 64) = p - i by omega]
      rw [ih (i + 1) _ _ _ (by omega), key, Nat.xor_assoc]
    · rw [clmulGo]
      simp only [hbb, show ¬ bb.toNat % 2 = 1 from fun h' => hbb (hb.mpr h'), ↓reduceIte]
      rw [ih (i + 1) _ _ _ (by omega)]
      simp

theorem clmul_clmulNat (a b : UInt64) : clmulPair (clmul a b) = clmulNat a.toNat b.toNat := by
  rw [clmul, clmulGo_clmulNat a 64 0 b 0 0 (by decide), Nat.mod_eq_of_lt b.toNat_lt]
  simp [clmulPair]

/-- `clmul` from three `bmul32` (Karatsuba on 32-bit halves). -/
@[inline] def clmulK (a b : UInt64) : UInt64 × UInt64 :=
  let p0 := bmul32 a b
  let p2 := bmul32 (a >>> 32) (b >>> 32)
  let p1 := bmul32 (a ^^^ (a >>> 32)) (b ^^^ (b >>> 32)) ^^^ p0 ^^^ p2
  (p0 ^^^ (p1 <<< 32), p2 ^^^ (p1 >>> 32))

theorem clmulK_eq_clmul (a b : UInt64) : clmulK a b = clmul a b := by
  apply clmulPair_inj
  rw [clmul_clmulNat]
  have split : ∀ v : UInt64, v.toNat = v.toNat % 2 ^ 32 ^^^ ((v >>> 32).toNat % 2 ^ 32) <<< 32 := fun v => by
    apply Nat.eq_of_testBit_eq; intro p
    simp only [Nat.testBit_xor, Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft, UInt64.toNat_shiftRight,
      show (32 : UInt64).toNat % 64 = 32 from rfl, Nat.testBit_shiftRight]
    by_cases hp : p < 32
    · simp [hp, show ¬ p ≥ 32 by omega]
    · by_cases hp64 : p < 64
      · simp [hp, show p ≥ 32 by omega, show p - 32 < 32 by omega, show 32 + (p - 32) = p by omega]
      · simp [hp, show p ≥ 32 by omega, show ¬ p - 32 < 32 by omega, Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le
          v.toNat_lt (Nat.pow_le_pow_right (by decide) (by omega) : 2 ^ 64 ≤ 2 ^ p))]
  have hm : ∀ v : UInt64, (v ^^^ (v >>> 32)).toNat % 2 ^ 32 = v.toNat % 2 ^ 32 ^^^ (v >>> 32).toNat % 2 ^ 32 :=
    fun v => by rw [UInt64.toNat_xor, Nat.xor_mod_two_pow]
  have hlt : ∀ u v : UInt64, clmulNat (u.toNat % 2 ^ 32) (v.toNat % 2 ^ 32) < 2 ^ 64 := fun u v =>
    clmulNat_lt _ _ 32 (Nat.mod_lt _ (by decide)) 32 (Nat.mod_lt _ (by decide))
  have e0 := bmul32_eq a b
  have e2 := bmul32_eq (a >>> 32) (b >>> 32)
  have em := bmul32_eq (a ^^^ (a >>> 32)) (b ^^^ (b >>> 32))
  rw [hm, hm] at em
  have hbit : ∀ (u v : UInt64) (q : Nat), q ≥ 64 →
      (clmulNat (u.toNat % 2 ^ 32) (v.toNat % 2 ^ 32)).testBit q = false :=
    fun u v q hq => Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le (hlt u v) (Nat.pow_le_pow_right (by decide) hq))
  have b00 := hbit a b
  have b01 := hbit a (b >>> 32)
  have b10 := hbit (a >>> 32) b
  have b11 := hbit (a >>> 32) (b >>> 32)
  conv => rhs; rw [split a, split b]
  simp only [clmulNat_xor_left, clmulNat_xor_right, clmulNat_shiftLeft_left, clmulNat_shiftLeft_right, clmulK] at em ⊢
  generalize a.toNat % 2 ^ 32 = A0 at *
  generalize (a >>> 32).toNat % 2 ^ 32 = A1 at *
  generalize b.toNat % 2 ^ 32 = B0 at *
  generalize (b >>> 32).toNat % 2 ^ 32 = B1 at *
  have e1 : (bmul32 (a ^^^ (a >>> 32)) (b ^^^ (b >>> 32)) ^^^ bmul32 a b ^^^ bmul32 (a >>> 32) (b >>> 32)).toNat =
      clmulNat A0 B1 ^^^ clmulNat A1 B0 := by
    rw [UInt64.toNat_xor, UInt64.toNat_xor, em, e0, e2]
    apply Nat.eq_of_testBit_eq; intro q
    simp only [Nat.testBit_xor]
    cases (clmulNat A0 B0).testBit q <;> cases (clmulNat A0 B1).testBit q <;> cases (clmulNat A1 B0).testBit q <;>
      cases (clmulNat A1 B1).testBit q <;> rfl
  generalize bmul32 (a ^^^ (a >>> 32)) (b ^^^ (b >>> 32)) ^^^ bmul32 a b ^^^ bmul32 (a >>> 32) (b >>> 32) = P1 at e1
  apply Nat.eq_of_testBit_eq; intro p
  rw [testBit_clmulPair]
  simp only [UInt64.toNat_xor, UInt64.toNat_shiftLeft, UInt64.toNat_shiftRight, e0, e1, e2,
    show (32 : UInt64).toNat % 64 = 32 from rfl, Nat.testBit_xor, Nat.testBit_shiftLeft, Nat.testBit_shiftRight,
    Nat.testBit_mod_two_pow]
  by_cases h1 : p < 32
  · simp [show p < 64 by omega, show ¬ p ≥ 32 by omega]
  · by_cases h2 : p < 64
    · simp only [h2, show p ≥ 32 by omega, show ¬ p - 32 ≥ 32 by omega, ↓reduceIte, decide_true, decide_false,
        Bool.true_and, Bool.false_and, Bool.xor_false]
      cases (clmulNat A0 B1).testBit (p - 32) <;> cases (clmulNat A1 B0).testBit (p - 32) <;> simp
    · by_cases h3 : p < 96
      · simp only [h2, show p ≥ 32 by omega, show p - 32 ≥ 32 by omega, b00 p (by omega), ↓reduceIte,
          decide_true, Bool.true_and, show 32 + (p - 64) = p - 32 by omega, show p - 32 - 32 = p - 64 by omega,
          Bool.false_xor]
        cases (clmulNat A0 B1).testBit (p - 32) <;> cases (clmulNat A1 B0).testBit (p - 32) <;>
          cases (clmulNat A1 B1).testBit (p - 64) <;> rfl
      · simp only [h2, show p ≥ 32 by omega, show p - 32 ≥ 32 by omega, b00 p (by omega), b01 (p - 32) (by omega),
          b10 (p - 32) (by omega), ↓reduceIte, decide_true, Bool.true_and, show 32 + (p - 64) = p - 32 by omega,
          show p - 32 - 32 = p - 64 by omega, Bool.false_xor, Bool.xor_false]

/-- Compiled code runs `clmulK`; the kernel and every proof see `clmul`. `@[csimp]` rewrites only code compiled after
it, so it stays above `mul`. -/
@[csimp] theorem clmul_eq_clmulK : @clmul = @clmulK :=
  funext fun a => funext fun b => (clmulK_eq_clmul a b).symm

/-- Reduce the 256-bit carry-less product `r3·2^192 + r2·2^128 + r1·2^64 + r0` modulo the GHASH polynomial. -/
@[inline] def reduce (r0 r1 r2 r3 : UInt64) : F128 :=
  let tl := r2 ^^^ (r2 <<< 1) ^^^ (r2 <<< 2) ^^^ (r2 <<< 7)
  let th := r3 ^^^ ((r3 <<< 1) ||| (r2 >>> 63)) ^^^ ((r3 <<< 2) ||| (r2 >>> 62)) ^^^ ((r3 <<< 7) ||| (r2 >>> 57))
  let o := (r3 >>> 63) ^^^ (r3 >>> 62) ^^^ (r3 >>> 57)
  ⟨r0 ^^^ tl ^^^ o ^^^ (o <<< 1) ^^^ (o <<< 2) ^^^ (o <<< 7), r1 ^^^ th⟩

def mul (a b : F128) : F128 :=
  let (p0l, p0h) := clmul a.lo b.lo
  let (p2l, p2h) := clmul a.hi b.hi
  let (ml, mh) := clmul (a.lo ^^^ a.hi) (b.lo ^^^ b.hi)
  let m1l := ml ^^^ p0l ^^^ p2l
  let m1h := mh ^^^ p0h ^^^ p2h
  reduce p0l (p0h ^^^ m1l) (p2l ^^^ m1h) p2h

/-- `mul` on its arguments' words. Taking words, not cells, it can't reuse an argument's cell, so compiled code borrows
both factors: a factor that tasks share (a challenge, a scalar) is never counted, which parallel tasks would contend
on. -/
@[noinline] def mulW (al ah bl bh : UInt64) : F128 :=
  let (p0l, p0h) := clmul al bl
  let (p2l, p2h) := clmul ah bh
  let (ml, mh) := clmul (al ^^^ ah) (bl ^^^ bh)
  let m1l := ml ^^^ p0l ^^^ p2l
  let m1h := mh ^^^ p0h ^^^ p2h
  reduce p0l (p0h ^^^ m1l) (p2l ^^^ m1h) p2h

@[inline] def mulK (a b : F128) : F128 := mulW a.lo a.hi b.lo b.hi

/-- Compiled code runs `mulK`; the kernel and every proof see `mul`. Above `instMul`, so every product compiles to it. -/
@[csimp] theorem mul_eq_mulK : @mul = @mulK := rfl

instance instMul : Mul F128 := ⟨mul⟩

def square (a : F128) : F128 := mul a a

/-- `k` square-and-multiply steps by `a` from `acc`: `acc^(2^k) · a^(2^k − 1)`. -/
def invGo (a : F128) : Nat → F128 → F128
  | 0, acc => acc
  | k + 1, acc => invGo a k (mul (square acc) a)

/-- `a^(2^128 − 2)`: the inverse, with `inv 0 = 0` (§2.1). 2^128 − 2 = 0b111…10: 127 ones, then one zero. -/
def inv (a : F128) : F128 := square (invGo a 127 one)

/-- The 16-byte encoding `le64(lo) ‖ le64(hi)`. -/
def toBytes (a : F128) : ByteArray :=
  ⟨Array.ofFn (n := 16) fun i =>
    if i.val < 8 then (a.lo >>> (8 * i.val).toUInt64).toUInt8 else (a.hi >>> (8 * (i.val - 8)).toUInt64).toUInt8⟩

def pushBytes (out : ByteArray) (a : F128) : ByteArray := out ++ a.toBytes

@[inline] def bit (a : F128) (i : Nat) : Bool :=
  if i < 64 then (a.lo >>> i.toUInt64) &&& 1 == 1 else (a.hi >>> (i - 64).toUInt64) &&& 1 == 1

/-- Byte `j < 16`: bits `8j` to `8j + 7`, as a number below `256` (`testBit_byte`). -/
@[inline] def byte (a : F128) (j : Nat) : Nat :=
  if j < 8 then ((a.lo >>> (8 * j).toUInt64) &&& 255).toNat else ((a.hi >>> (8 * (j - 8)).toUInt64) &&& 255).toNat

protected theorem add_comm (a b : F128) : a + b = b + a := by
  show F128.mk (a.lo ^^^ b.lo) (a.hi ^^^ b.hi) = F128.mk (b.lo ^^^ a.lo) (b.hi ^^^ a.hi)
  rw [UInt64.xor_comm a.lo, UInt64.xor_comm a.hi]

protected theorem add_assoc (a b c : F128) : a + b + c = a + (b + c) := by
  show F128.mk (a.lo ^^^ b.lo ^^^ c.lo) (a.hi ^^^ b.hi ^^^ c.hi) =
    F128.mk (a.lo ^^^ (b.lo ^^^ c.lo)) (a.hi ^^^ (b.hi ^^^ c.hi))
  rw [UInt64.xor_assoc, UInt64.xor_assoc]

protected theorem add_zero (a : F128) : a + 0 = a := by
  show F128.mk (a.lo ^^^ 0) (a.hi ^^^ 0) = a
  rw [UInt64.xor_zero, UInt64.xor_zero]

protected theorem zero_add (a : F128) : 0 + a = a := by
  rw [F128.add_comm, F128.add_zero]

theorem and_one_xor (b c : UInt64) : ((b ^^^ c) &&& 1 = 1) ↔ ((b &&& 1 = 1) ↔ ¬(c &&& 1 = 1)) := by
  have h : ∀ x : UInt64, (x &&& 1 = 1) ↔ x.toNat.testBit 0 = true := fun x => by
    rw [Nat.testBit_zero, decide_eq_true_iff, ← Nat.and_one_is_mod, ← show (1 : UInt64).toNat = 1 from rfl,
      ← UInt64.toNat_and]
    exact ⟨fun e => by rw [e], fun e => UInt64.toNat_inj.mp e⟩
  rw [h, h, h, UInt64.toNat_xor, Nat.testBit_xor]
  cases b.toNat.testBit 0 <;> cases c.toNat.testBit 0 <;> simp

theorem xor_cancel (x y t : UInt64) : x ^^^ y = (x ^^^ t) ^^^ (y ^^^ t) := by
  rw [show (x ^^^ t) ^^^ (y ^^^ t) = (x ^^^ y) ^^^ (t ^^^ t) by ac_rfl, UInt64.xor_self, UInt64.xor_zero]

/-- `clmulGo` is additive in the multiplier and the accumulator together. -/
theorem clmulGo_xor (a : UInt64) : ∀ (n i : Nat) (b c l1 h1 l2 h2 : UInt64),
    clmulGo a n i (b ^^^ c) (l1 ^^^ l2) (h1 ^^^ h2) =
      ((clmulGo a n i b l1 h1).1 ^^^ (clmulGo a n i c l2 h2).1, (clmulGo a n i b l1 h1).2 ^^^ (clmulGo a n i c l2 h2).2)
  | 0, _, _, _, _, _, _, _ => rfl
  | n + 1, i, b, c, l1, h1, l2, h2 => by
    simp only [clmulGo, UInt64.shiftRight_xor]
    have hx := and_one_xor b c
    by_cases hb : b &&& 1 = 1 <;> by_cases hc : c &&& 1 = 1
    · rw [if_neg (fun h => (hx.mp h).mp hb hc), if_pos hb, if_pos hc, ← clmulGo_xor a n (i + 1)]
      congr 1
      · exact xor_cancel _ _ _
      · split
        · rfl
        · exact xor_cancel _ _ _
    · rw [if_pos (hx.mpr ⟨fun _ => hc, fun _ => hb⟩), if_pos hb, if_neg hc, ← clmulGo_xor a n (i + 1)]
      congr 1
      · ac_rfl
      · split
        · rfl
        · ac_rfl
    · rw [if_pos (hx.mpr ⟨fun h => absurd h hb, fun h => absurd hc h⟩), if_neg hb, if_pos hc,
        ← clmulGo_xor a n (i + 1)]
      congr 1
      · ac_rfl
      · split
        · rfl
        · ac_rfl
    · rw [if_neg (fun h => hb ((hx.mp h).mpr hc)), if_neg hb, if_neg hc, ← clmulGo_xor a n (i + 1)]

theorem clmul_xor (a b c : UInt64) :
    clmul a (b ^^^ c) = ((clmul a b).1 ^^^ (clmul a c).1, (clmul a b).2 ^^^ (clmul a c).2) := by
  unfold clmul
  rw [← clmulGo_xor, show (0 : UInt64) ^^^ 0 = 0 from rfl]

theorem clmulGo_zero (a : UInt64) : ∀ (n i : Nat) (lo hi : UInt64), clmulGo a n i 0 lo hi = (lo, hi)
  | 0, _, _, _ => rfl
  | n + 1, i, lo, hi => by
    rw [clmulGo, show (0 : UInt64) >>> 1 = 0 from rfl]
    simpa using clmulGo_zero a n (i + 1) lo hi

theorem clmul_one (x : UInt64) : clmul x 1 = (x, 0) := by
  unfold clmul
  rw [clmulGo, if_pos (by decide : (1 : UInt64) &&& 1 = 1), if_pos (rfl : (0 : Nat) = 0),
    show (1 : UInt64) >>> 1 = 0 from rfl, clmulGo_zero, show (0 : Nat).toUInt64 = 0 from rfl, UInt64.shiftLeft_zero,
    UInt64.zero_xor]

theorem clmul_zero (x : UInt64) : clmul x 0 = (0, 0) := clmulGo_zero x 64 0 0 0

/-- The two shifted words `reduce` joins with `|||` share no bit, so `|||` there is `^^^`. -/
theorem shift_or_eq_xor (x y : UInt64) (k : Nat) (hk : 0 < k ∧ k < 64) :
    (x <<< k.toUInt64) ||| (y >>> (64 - k).toUInt64) = (x <<< k.toUInt64) ^^^ (y >>> (64 - k).toUInt64) := by
  apply UInt64.toBitVec_inj.mp
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [UInt64.toBitVec_or, UInt64.toBitVec_xor, BitVec.getLsbD_or, BitVec.getLsbD_xor]
  simp [BitVec.getLsbD_shiftLeft, BitVec.getLsbD_ushiftRight, Nat.toUInt64]
  rw [Nat.mod_eq_of_lt hk.2, Nat.mod_eq_of_lt (show 64 - k < 64 by omega)]
  by_cases hik : i < k
  · simp [hik]
  · rw [BitVec.getLsbD_of_ge y.toBitVec (64 - k + i) (by omega)]
    simp

theorem reduce_xor (r0 r1 r2 r3 s0 s1 s2 s3 : UInt64) :
    reduce (r0 ^^^ s0) (r1 ^^^ s1) (r2 ^^^ s2) (r3 ^^^ s3) = reduce r0 r1 r2 r3 + reduce s0 s1 s2 s3 := by
  have h1 : ∀ x y : UInt64, (x <<< 1 ||| y >>> 63) = x <<< 1 ^^^ y >>> 63 := fun x y =>
    shift_or_eq_xor x y 1 (by omega)
  have h2 : ∀ x y : UInt64, (x <<< 2 ||| y >>> 62) = x <<< 2 ^^^ y >>> 62 := fun x y =>
    shift_or_eq_xor x y 2 (by omega)
  have h7 : ∀ x y : UInt64, (x <<< 7 ||| y >>> 57) = x <<< 7 ^^^ y >>> 57 := fun x y =>
    shift_or_eq_xor x y 7 (by omega)
  simp only [reduce, HAdd.hAdd, Add.add, add, h1, h2, h7]
  simp only [UInt64.shiftLeft_xor, UInt64.shiftRight_xor]
  congr 1 <;> ac_rfl

protected theorem mul_add (a b c : F128) : a * (b + c) = a * b + a * c := by
  show mul a ⟨b.lo ^^^ c.lo, b.hi ^^^ c.hi⟩ = mul a b + mul a c
  have hm : clmul (a.lo ^^^ a.hi) (b.lo ^^^ c.lo ^^^ (b.hi ^^^ c.hi)) =
      ((clmul (a.lo ^^^ a.hi) (b.lo ^^^ b.hi)).1 ^^^ (clmul (a.lo ^^^ a.hi) (c.lo ^^^ c.hi)).1,
        (clmul (a.lo ^^^ a.hi) (b.lo ^^^ b.hi)).2 ^^^ (clmul (a.lo ^^^ a.hi) (c.lo ^^^ c.hi)).2) := by
    rw [← clmul_xor, show b.lo ^^^ c.lo ^^^ (b.hi ^^^ c.hi) = b.lo ^^^ b.hi ^^^ (c.lo ^^^ c.hi) by ac_rfl]
  simp only [mul, clmul_xor a.lo b.lo c.lo, clmul_xor a.hi b.hi c.hi, hm]
  generalize clmul a.lo b.lo = p0 at *
  generalize clmul a.lo c.lo = q0 at *
  generalize clmul a.hi b.hi = p2 at *
  generalize clmul a.hi c.hi = q2 at *
  generalize clmul (a.lo ^^^ a.hi) (b.lo ^^^ b.hi) = pm at *
  generalize clmul (a.lo ^^^ a.hi) (c.lo ^^^ c.hi) = qm at *
  obtain ⟨_, _⟩ := p0
  obtain ⟨_, _⟩ := q0
  obtain ⟨_, _⟩ := p2
  obtain ⟨_, _⟩ := q2
  obtain ⟨_, _⟩ := pm
  obtain ⟨_, _⟩ := qm
  simp only
  rw [← reduce_xor]
  congr 1 <;> ac_rfl

protected theorem mul_one (a : F128) : a * 1 = a := by
  show mul a ⟨1, 0⟩ = a
  simp only [mul, UInt64.xor_zero, clmul_one, clmul_zero]
  show F128.mk _ _ = a
  rw [show a.lo ^^^ a.hi ^^^ a.lo = a.hi by
    rw [show a.lo ^^^ a.hi ^^^ a.lo = a.hi ^^^ (a.lo ^^^ a.lo) by ac_rfl, UInt64.xor_self, UInt64.xor_zero]]
  simp only [UInt64.zero_xor, UInt64.xor_zero, UInt64.zero_shiftLeft, UInt64.zero_shiftRight, UInt64.zero_or]

protected theorem mul_zero (a : F128) : a * 0 = 0 := by
  show mul a ⟨0, 0⟩ = ⟨0, 0⟩
  simp only [mul, UInt64.xor_zero, clmul_zero]
  rfl

/-- `x · (r + 1) = x · r + x`: one product where `eqStep` reads two. -/
theorem mul_add_one (x r : F128) : x * (r + 1) = x * r + x := by
  rw [F128.mul_add, F128.mul_one]

/-- `clmulGo` is additive in the multiplicand and the accumulator together. -/
theorem clmulGo_xor_left (a a' : UInt64) : ∀ (n i : Nat) (b l1 h1 l2 h2 : UInt64),
    clmulGo (a ^^^ a') n i b (l1 ^^^ l2) (h1 ^^^ h2) =
      ((clmulGo a n i b l1 h1).1 ^^^ (clmulGo a' n i b l2 h2).1, (clmulGo a n i b l1 h1).2 ^^^ (clmulGo a' n i b l2 h2).2)
  | 0, _, _, _, _, _, _ => rfl
  | n + 1, i, b, l1, h1, l2, h2 => by
    simp only [clmulGo]
    split
    · rw [← clmulGo_xor_left a a' n (i + 1)]
      congr 1
      · rw [UInt64.shiftLeft_xor]; ac_rfl
      · split
        · rfl
        · rw [UInt64.shiftRight_xor]; ac_rfl
    · rw [← clmulGo_xor_left a a' n (i + 1)]

theorem clmul_xor_left (a a' b : UInt64) :
    clmul (a ^^^ a') b = ((clmul a b).1 ^^^ (clmul a' b).1, (clmul a b).2 ^^^ (clmul a' b).2) := by
  unfold clmul
  rw [← clmulGo_xor_left, show (0 : UInt64) ^^^ 0 = 0 from rfl]

protected theorem add_mul (a b c : F128) : (a + b) * c = a * c + b * c := by
  show mul ⟨a.lo ^^^ b.lo, a.hi ^^^ b.hi⟩ c = mul a c + mul b c
  have hm : clmul (a.lo ^^^ b.lo ^^^ (a.hi ^^^ b.hi)) (c.lo ^^^ c.hi) =
      ((clmul (a.lo ^^^ a.hi) (c.lo ^^^ c.hi)).1 ^^^ (clmul (b.lo ^^^ b.hi) (c.lo ^^^ c.hi)).1,
        (clmul (a.lo ^^^ a.hi) (c.lo ^^^ c.hi)).2 ^^^ (clmul (b.lo ^^^ b.hi) (c.lo ^^^ c.hi)).2) := by
    rw [← clmul_xor_left, show a.lo ^^^ b.lo ^^^ (a.hi ^^^ b.hi) = a.lo ^^^ a.hi ^^^ (b.lo ^^^ b.hi) by ac_rfl]
  simp only [mul, clmul_xor_left a.lo b.lo c.lo, clmul_xor_left a.hi b.hi c.hi, hm]
  generalize clmul a.lo c.lo = p0 at *
  generalize clmul b.lo c.lo = q0 at *
  generalize clmul a.hi c.hi = p2 at *
  generalize clmul b.hi c.hi = q2 at *
  generalize clmul (a.lo ^^^ a.hi) (c.lo ^^^ c.hi) = pm at *
  generalize clmul (b.lo ^^^ b.hi) (c.lo ^^^ c.hi) = qm at *
  obtain ⟨_, _⟩ := p0
  obtain ⟨_, _⟩ := q0
  obtain ⟨_, _⟩ := p2
  obtain ⟨_, _⟩ := q2
  obtain ⟨_, _⟩ := pm
  obtain ⟨_, _⟩ := qm
  simp only
  rw [← reduce_xor]
  congr 1 <;> ac_rfl

theorem bitU64 (x : UInt64) (s : Nat) (hs : s < 64) : ((x >>> s.toUInt64) &&& 1 == 1) = x.toNat.testBit s := by
  have hn : ((x >>> s.toUInt64) &&& 1).toNat = x.toNat / 2 ^ s % 2 := by
    rw [UInt64.toNat_and, UInt64.toNat_shiftRight, Nat.toUInt64, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega),
      Nat.mod_eq_of_lt (by omega), show (1 : UInt64).toNat = 1 from rfl, Nat.and_one_is_mod,
      Nat.shiftRight_eq_div_pow]
  rw [Nat.testBit_eq_decide_div_mod_eq]
  have h1 : ∀ c : UInt64, (c == 1) = decide (c.toNat = 1) := fun c => by
    by_cases h : c = 1
    · subst h; rfl
    · have : c.toNat ≠ 1 := fun e => h (UInt64.toNat_inj.mp (by rw [e]; rfl))
      rw [show (c == 1) = false from beq_eq_false_iff_ne.mpr h]
      simp [this]
  rw [h1, hn]

theorem byteU64 (x : UInt64) (j i : Nat) (hj : j < 8) (hi : i < 8) :
    ((x >>> (8 * j).toUInt64) &&& 255).toNat.testBit i = x.toNat.testBit (8 * j + i) := by
  rw [UInt64.toNat_and, UInt64.toNat_shiftRight, Nat.toUInt64, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega),
    Nat.mod_eq_of_lt (by omega), Nat.testBit_and, Nat.testBit_shiftRight,
    show (255 : UInt64).toNat = 2 ^ 8 - 1 from rfl, Nat.testBit_two_pow_sub_one]
  simp [hi]

theorem byte_lt (a : F128) (j : Nat) : a.byte j < 256 := by
  unfold byte
  split <;> (rw [UInt64.toNat_and]; exact Nat.lt_of_le_of_lt Nat.and_le_right (by decide))

/-- Bit `i` of byte `j` is bit `8j + i`. -/
theorem testBit_byte (a : F128) {j i : Nat} (hj : j < 16) (hi : i < 8) : (a.byte j).testBit i = a.bit (8 * j + i) := by
  unfold byte bit
  by_cases h : j < 8
  · rw [if_pos h, if_pos (show 8 * j + i < 64 by omega), byteU64 _ _ _ h hi, bitU64 _ _ (by omega)]
  · rw [if_neg h, if_neg (show ¬ 8 * j + i < 64 by omega), byteU64 _ _ _ (by omega) hi, bitU64 _ _ (by omega),
      show 8 * j + i - 64 = 8 * (j - 8) + i by omega]

/-- Byte `j < 8` of `x`, in place. -/
@[inline] def bslot (x : UInt64) (j : Nat) : UInt64 := ((x >>> (8 * j).toUInt64) &&& 255) <<< (8 * j).toUInt64

theorem getLsbD_bslot (x : UInt64) (j i : Nat) (hj : j < 8) (hi : i < 64) :
    (bslot x j).toBitVec.getLsbD i = (decide (8 * j ≤ i ∧ i < 8 * j + 8) && x.toBitVec.getLsbD i) := by
  simp only [bslot, UInt64.toBitVec_shiftLeft, UInt64.toBitVec_and, UInt64.toBitVec_shiftRight, Nat.toUInt64]
  simp [Nat.mod_eq_of_lt (show 8 * j < 64 by omega)]
  rw [show (255#64) = BitVec.ofNat 64 (2 ^ 8 - 1) from rfl, BitVec.getLsbD_ofNat, Nat.testBit_two_pow_sub_one]
  by_cases h1 : i < 8 * j
  · simp [h1, show ¬ 8 * j ≤ i by omega]
  · rw [show 8 * j + (i - 8 * j) = i by omega]
    by_cases h2 : i < 8 * j + 8
    · simp [h1, h2, hi, show 8 * j ≤ i by omega, show i - 8 * j < 8 by omega, show i - 8 * j < 64 by omega]
    · simp [h1, h2, show ¬ i - 8 * j < 8 by omega]

/-- A word is the sum of its eight bytes in place. -/
theorem u64_bytes (x : UInt64) :
    x = bslot x 0 ^^^ bslot x 1 ^^^ bslot x 2 ^^^ bslot x 3 ^^^ bslot x 4 ^^^ bslot x 5 ^^^ bslot x 6 ^^^ bslot x 7 := by
  apply UInt64.toBitVec_inj.mp
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [UInt64.toBitVec_xor, BitVec.getLsbD_xor]
  rw [getLsbD_bslot x 0 i (by omega) hi, getLsbD_bslot x 1 i (by omega) hi, getLsbD_bslot x 2 i (by omega) hi,
    getLsbD_bslot x 3 i (by omega) hi, getLsbD_bslot x 4 i (by omega) hi, getLsbD_bslot x 5 i (by omega) hi,
    getLsbD_bslot x 6 i (by omega) hi, getLsbD_bslot x 7 i (by omega) hi]
  rcases (by omega : i < 8 ∨ (8 ≤ i ∧ i < 16) ∨ (16 ≤ i ∧ i < 24) ∨ (24 ≤ i ∧ i < 32) ∨ (32 ≤ i ∧ i < 40) ∨
    (40 ≤ i ∧ i < 48) ∨ (48 ≤ i ∧ i < 56) ∨ (56 ≤ i ∧ i < 64)) with h | h | h | h | h | h | h | h <;>
  simp (disch := omega) only [decide_eq_true, decide_eq_false, Bool.true_and, Bool.false_and, Bool.xor_false,
    Bool.false_xor]

/-- Byte `b < 256` at byte `j < 16`. -/
@[inline] def byteMono (j : Nat) (b : UInt64) : F128 :=
  if j < 8 then ⟨b <<< (8 * j).toUInt64, 0⟩ else ⟨0, b <<< (8 * (j - 8)).toUInt64⟩

/-- Byte `j` of `v` as a word. -/
@[inline] def byteW (v : F128) (j : Nat) : UInt64 :=
  if j < 8 then (v.lo >>> (8 * j).toUInt64) &&& 255 else (v.hi >>> (8 * (j - 8)).toUInt64) &&& 255

theorem byte_eq (v : F128) (j : Nat) : v.byte j = (byteW v j).toNat := by
  unfold byte byteW; split <;> rfl

theorem byteMono_byteW (v : F128) (j : Nat) :
    byteMono j (byteW v j) = if j < 8 then ⟨bslot v.lo j, 0⟩ else ⟨0, bslot v.hi (j - 8)⟩ := by
  unfold byteMono byteW bslot; split <;> rfl

/-- `v` is the sum of its sixteen bytes in place. -/
theorem bytes_sum (v : F128) : v =
    byteMono 0 (byteW v 0) + byteMono 1 (byteW v 1) + byteMono 2 (byteW v 2) + byteMono 3 (byteW v 3) +
    byteMono 4 (byteW v 4) + byteMono 5 (byteW v 5) + byteMono 6 (byteW v 6) + byteMono 7 (byteW v 7) +
    byteMono 8 (byteW v 8) + byteMono 9 (byteW v 9) + byteMono 10 (byteW v 10) + byteMono 11 (byteW v 11) +
    byteMono 12 (byteW v 12) + byteMono 13 (byteW v 13) + byteMono 14 (byteW v 14) + byteMono 15 (byteW v 15) := by
  simp only [byteMono_byteW, Nat.reduceLT, ↓reduceIte, Nat.reduceSub, HAdd.hAdd, Add.add, add]
  simp only [UInt64.xor_zero, UInt64.zero_xor]
  rw [← u64_bytes, ← u64_bytes]

/-- `byteMono (k / 256) (k mod 256) · r` at `k < 4096`: `mulBy`'s table of products by `r`. -/
def mulTable (r : F128) : Array F128 :=
  Array.ofFn (n := 4096) fun k => byteMono (k.val / 256) (k.val % 256).toUInt64 * r

/-- `v · r` from `r`'s table (`mulBy_mulTable`): one entry a byte of `v`, added. -/
@[inline] def mulBy (t : Array F128) (v : F128) : F128 :=
  t[v.byte 0]! + t[256 + v.byte 1]! + t[512 + v.byte 2]! + t[768 + v.byte 3]! + t[1024 + v.byte 4]! +
    t[1280 + v.byte 5]! + t[1536 + v.byte 6]! + t[1792 + v.byte 7]! + t[2048 + v.byte 8]! + t[2304 + v.byte 9]! +
    t[2560 + v.byte 10]! + t[2816 + v.byte 11]! + t[3072 + v.byte 12]! + t[3328 + v.byte 13]! +
    t[3584 + v.byte 14]! + t[3840 + v.byte 15]!

theorem mulTable_get (r v : F128) (j : Nat) (hj : j < 16) :
    (mulTable r)[256 * j + v.byte j]! = byteMono j (byteW v j) * r := by
  have hb := v.byte_lt j
  rw [getElem!_pos (mulTable r) (256 * j + v.byte j) (by simp [mulTable]; omega)]
  simp only [mulTable, Array.getElem_ofFn]
  rw [show (256 * j + v.byte j) / 256 = j by omega, show (256 * j + v.byte j) % 256 = v.byte j by omega, byte_eq,
    show Nat.toUInt64 (byteW v j).toNat = byteW v j from UInt64.toNat_inj.mp (by
      rw [Nat.toUInt64, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (UInt64.toNat_lt _)])]

theorem mulBy_mulTable (r v : F128) : mulBy (mulTable r) v = v * r := by
  have h := fun j hj => mulTable_get r v j hj
  rw [mulBy, show v.byte 0 = 256 * 0 + v.byte 0 by omega, show 256 + v.byte 1 = 256 * 1 + v.byte 1 by omega,
    show 512 + v.byte 2 = 256 * 2 + v.byte 2 by omega, show 768 + v.byte 3 = 256 * 3 + v.byte 3 by omega,
    show 1024 + v.byte 4 = 256 * 4 + v.byte 4 by omega, show 1280 + v.byte 5 = 256 * 5 + v.byte 5 by omega,
    show 1536 + v.byte 6 = 256 * 6 + v.byte 6 by omega, show 1792 + v.byte 7 = 256 * 7 + v.byte 7 by omega,
    show 2048 + v.byte 8 = 256 * 8 + v.byte 8 by omega, show 2304 + v.byte 9 = 256 * 9 + v.byte 9 by omega,
    show 2560 + v.byte 10 = 256 * 10 + v.byte 10 by omega, show 2816 + v.byte 11 = 256 * 11 + v.byte 11 by omega,
    show 3072 + v.byte 12 = 256 * 12 + v.byte 12 by omega, show 3328 + v.byte 13 = 256 * 13 + v.byte 13 by omega,
    show 3584 + v.byte 14 = 256 * 14 + v.byte 14 by omega, show 3840 + v.byte 15 = 256 * 15 + v.byte 15 by omega]
  rw [h 0 (by omega), h 1 (by omega), h 2 (by omega), h 3 (by omega), h 4 (by omega), h 5 (by omega), h 6 (by omega),
    h 7 (by omega), h 8 (by omega), h 9 (by omega), h 10 (by omega), h 11 (by omega), h 12 (by omega),
    h 13 (by omega), h 14 (by omega), h 15 (by omega)]
  conv => rhs; rw [bytes_sum v]
  simp only [F128.add_mul]

/-- `r · byteMono (k / 256) (k mod 256)` at `k < 4096`: `mulBy`'s table of products by `r` on the left. -/
def mulTableL (r : F128) : Array F128 :=
  Array.ofFn (n := 4096) fun k => r * byteMono (k.val / 256) (k.val % 256).toUInt64

theorem mulTableL_get (r v : F128) (j : Nat) (hj : j < 16) :
    (mulTableL r)[256 * j + v.byte j]! = r * byteMono j (byteW v j) := by
  have hb := v.byte_lt j
  rw [getElem!_pos (mulTableL r) (256 * j + v.byte j) (by simp [mulTableL]; omega)]
  simp only [mulTableL, Array.getElem_ofFn]
  rw [show (256 * j + v.byte j) / 256 = j by omega, show (256 * j + v.byte j) % 256 = v.byte j by omega, byte_eq,
    show Nat.toUInt64 (byteW v j).toNat = byteW v j from UInt64.toNat_inj.mp (by
      rw [Nat.toUInt64, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (UInt64.toNat_lt _)])]

theorem mulBy_mulTableL (r v : F128) : mulBy (mulTableL r) v = r * v := by
  have h := fun j hj => mulTableL_get r v j hj
  rw [mulBy, show v.byte 0 = 256 * 0 + v.byte 0 by omega, show 256 + v.byte 1 = 256 * 1 + v.byte 1 by omega,
    show 512 + v.byte 2 = 256 * 2 + v.byte 2 by omega, show 768 + v.byte 3 = 256 * 3 + v.byte 3 by omega,
    show 1024 + v.byte 4 = 256 * 4 + v.byte 4 by omega, show 1280 + v.byte 5 = 256 * 5 + v.byte 5 by omega,
    show 1536 + v.byte 6 = 256 * 6 + v.byte 6 by omega, show 1792 + v.byte 7 = 256 * 7 + v.byte 7 by omega,
    show 2048 + v.byte 8 = 256 * 8 + v.byte 8 by omega, show 2304 + v.byte 9 = 256 * 9 + v.byte 9 by omega,
    show 2560 + v.byte 10 = 256 * 10 + v.byte 10 by omega, show 2816 + v.byte 11 = 256 * 11 + v.byte 11 by omega,
    show 3072 + v.byte 12 = 256 * 12 + v.byte 12 by omega, show 3328 + v.byte 13 = 256 * 13 + v.byte 13 by omega,
    show 3584 + v.byte 14 = 256 * 14 + v.byte 14 by omega, show 3840 + v.byte 15 = 256 * 15 + v.byte 15 by omega]
  rw [h 0 (by omega), h 1 (by omega), h 2 (by omega), h 3 (by omega), h 4 (by omega), h 5 (by omega), h 6 (by omega),
    h 7 (by omega), h 8 (by omega), h 9 (by omega), h 10 (by omega), h 11 (by omega), h 12 (by omega),
    h 13 (by omega), h 14 (by omega), h 15 (by omega)]
  conv => rhs; rw [bytes_sum v]
  simp only [F128.mul_add]

theorem byteMono_xor (j : Nat) (x y : UInt64) : byteMono j (x ^^^ y) = byteMono j x + byteMono j y := by
  unfold byteMono
  split <;> simp only [HAdd.hAdd, Add.add, add, UInt64.shiftLeft_xor, UInt64.xor_zero]

theorem nib_split (k : Nat) : (k % 256).toUInt64 = (k % 16).toUInt64 ^^^ (k / 16 % 16 * 16).toUInt64 := by
  apply UInt64.toBitVec_inj.mp
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [UInt64.toBitVec_xor, BitVec.getLsbD_xor, Nat.toUInt64, UInt64.toBitVec_ofNat', BitVec.getLsbD_ofNat,
    show (256 : Nat) = 2 ^ 8 from rfl, show (16 : Nat) = 2 ^ 4 from rfl, Nat.testBit_mod_two_pow,
    Nat.testBit_div_two_pow, Nat.testBit_mul_two_pow]
  by_cases h4 : i < 4
  · simp (disch := omega) only [decide_eq_true, decide_eq_false, Bool.true_and, Bool.false_and, Bool.xor_false]
  · by_cases h8 : i < 8
    · simp (disch := omega) only [decide_eq_true, decide_eq_false, Bool.true_and, Bool.false_and, Bool.false_xor]
      rw [Nat.sub_add_cancel (by omega)]
    · simp (disch := omega) only [decide_eq_true, decide_eq_false, Bool.true_and, Bool.false_and, Bool.false_xor]

/-- `byteMono (k / 32) c · r` at `k < 512`, for `c` the nibble `k mod 16` of byte `k / 32`, low (`k mod 32 < 16`) or
high. -/
def nibTable (r : F128) : Array F128 :=
  Array.ofFn (n := 512) fun k =>
    byteMono (k.val / 32) (if k.val % 32 < 16 then (k.val % 16).toUInt64 else (k.val % 16 * 16).toUInt64) * r

/-- `mulTable` from 512 products: an entry is its byte's two nibbles' entries of `nibTable`, added. -/
def mulTableFast (r : F128) : Array F128 :=
  let n := nibTable r
  Array.ofFn (n := 4096) fun k => n[32 * (k.val / 256) + k.val % 16]! + n[32 * (k.val / 256) + 16 + k.val / 16 % 16]!

theorem nib_get (f : Nat → UInt64 → F128) (k : Nat) (hk : k < 4096) :
    (Array.ofFn (n := 512) fun i =>
        f (i.val / 32) (if i.val % 32 < 16 then (i.val % 16).toUInt64 else (i.val % 16 * 16).toUInt64))[32 * (k / 256) + k % 16]! +
      (Array.ofFn (n := 512) fun i =>
        f (i.val / 32) (if i.val % 32 < 16 then (i.val % 16).toUInt64 else (i.val % 16 * 16).toUInt64))[32 * (k / 256) + 16 + k / 16 % 16]! =
      f (k / 256) (k % 16).toUInt64 + f (k / 256) (k / 16 % 16 * 16).toUInt64 := by
  rw [getElem!_pos _ _ (by rw [Array.size_ofFn]; omega), getElem!_pos _ _ (by rw [Array.size_ofFn]; omega)]
  simp only [Array.getElem_ofFn]
  rw [show (32 * (k / 256) + k % 16) / 32 = k / 256 by omega, show (32 * (k / 256) + k % 16) % 32 = k % 16 by omega,
    show (32 * (k / 256) + 16 + k / 16 % 16) / 32 = k / 256 by omega,
    show (32 * (k / 256) + 16 + k / 16 % 16) % 32 = 16 + k / 16 % 16 by omega,
    if_pos (show k % 16 < 16 by omega), if_neg (show ¬ 16 + k / 16 % 16 < 16 by omega),
    show (32 * (k / 256) + k % 16) % 16 = k % 16 by omega,
    show (32 * (k / 256) + 16 + k / 16 % 16) % 16 = k / 16 % 16 by omega]

@[csimp] theorem mulTable_eq_fast : @mulTable = @mulTableFast := by
  funext r
  apply Array.ext (by simp [mulTable, mulTableFast])
  intro k h1 _
  have hk : k < 4096 := by simpa [mulTable] using h1
  simp only [mulTable, mulTableFast, Array.getElem_ofFn]
  have h := nib_get (fun j c => byteMono j c * r) k hk
  simp only [nibTable]
  rw [h, nib_split, byteMono_xor, F128.add_mul]

/-- `nibTable` with `r` on the left. -/
def nibTableL (r : F128) : Array F128 :=
  Array.ofFn (n := 512) fun k =>
    r * byteMono (k.val / 32) (if k.val % 32 < 16 then (k.val % 16).toUInt64 else (k.val % 16 * 16).toUInt64)

/-- `mulTableL` from 512 products, as `mulTableFast`. -/
def mulTableLFast (r : F128) : Array F128 :=
  let n := nibTableL r
  Array.ofFn (n := 4096) fun k => n[32 * (k.val / 256) + k.val % 16]! + n[32 * (k.val / 256) + 16 + k.val / 16 % 16]!

@[csimp] theorem mulTableL_eq_fast : @mulTableL = @mulTableLFast := by
  funext r
  apply Array.ext (by simp [mulTableL, mulTableLFast])
  intro k h1 _
  have hk : k < 4096 := by simpa [mulTableL] using h1
  simp only [mulTableL, mulTableLFast, Array.getElem_ofFn]
  have h := nib_get (fun j c => r * byteMono j c) k hk
  simp only [nibTableL]
  rw [h, nib_split, byteMono_xor, F128.mul_add]

/-- `mulTableFast`'s additions over a `nibTable` already built: tasks that each need their own `mulTable r` share
one `nibTable r` and pay no product. -/
def expandNib (n : Array F128) : Array F128 :=
  (Array.range 4096).map fun k => n[32 * (k / 256) + k % 16]! + n[32 * (k / 256) + 16 + k / 16 % 16]!

theorem expandNib_nibTable (r : F128) : expandNib (nibTable r) = mulTable r := by
  rw [mulTable_eq_fast]
  apply Array.ext (by simp [expandNib, mulTableFast])
  intro k h1 _
  simp only [expandNib, mulTableFast, Array.getElem_map, Array.getElem_range, Array.getElem_ofFn]

/-- `x^i` for `i < 128`. -/
def monomial (i : Nat) : F128 :=
  if i < 64 then ⟨(1 : UInt64) <<< i.toUInt64, 0⟩ else ⟨0, (1 : UInt64) <<< (i - 64).toUInt64⟩

private def hexDigit (n : UInt64) : Char :=
  let d := n.toNat
  if d < 10 then Char.ofNat (48 + d) else Char.ofNat (87 + d)

/-- Record hex: `{hi:016x}{lo:016x}` (§5.1). -/
def toHex (a : F128) : String := Id.run do
  let mut s := ""
  for i in [0:16] do s := s.push (hexDigit ((a.hi >>> (4 * (15 - i)).toUInt64) &&& 0xf))
  for i in [0:16] do s := s.push (hexDigit ((a.lo >>> (4 * (15 - i)).toUInt64) &&& 0xf))
  return s

end F128

/-- GF(2^256) = F[u]/(u^2 + u + x^{-1}) (§2.4); an element is `c0 + c1·u`. -/
structure F256 where
  c0 : F128
  c1 : F128
  deriving BEq, Inhabited, Repr, DecidableEq

namespace F256

/-- `x^{-1}` in F. -/
def xinv : F128 := ⟨0x43, 0x8000000000000000⟩

def zero : F256 := ⟨0, 0⟩
def one : F256 := ⟨1, 0⟩
/-- `u`. -/
def u : F256 := ⟨0, 1⟩
def ofF (a : F128) : F256 := ⟨a, 0⟩

@[inline] def add (a b : F256) : F256 := ⟨a.c0 + b.c0, a.c1 + b.c1⟩

def mul (a b : F256) : F256 :=
  let p0 := a.c0 * b.c0
  let p1 := a.c1 * b.c1
  let p2 := (a.c0 + a.c1) * (b.c0 + b.c1)
  ⟨p0 + xinv * p1, p2 + p0⟩

/-- `K · F`, coordinate-wise. -/
@[inline] def smul (a : F256) (s : F128) : F256 := ⟨a.c0 * s, a.c1 * s⟩

instance instAdd : Add F256 := ⟨add⟩
instance instMul : Mul F256 := ⟨mul⟩
instance instOfNatOfNatNat : OfNat F256 0 := ⟨zero⟩
instance instOfNatOfNatNat_1 : OfNat F256 1 := ⟨one⟩

end F256

/-- `φ8(α) = g`, a root of `X^8 + X^4 + X^3 + X + 1` in F (§2.2). -/
def phi8Gen : F128 := ⟨0x6b8330483c2e9849, 0x0dcb364640a222fe⟩

/-- `g^i`. -/
def phi8Pow : Nat → F128
  | 0 => F128.one
  | i + 1 => phi8Pow i * phi8Gen

/-- `Σ_{i < n} bit_i(v) · g^i`. -/
def phi8Sum (v : Nat) : Nat → F128
  | 0 => F128.zero
  | i + 1 => if v.testBit i then phi8Sum v i + phi8Pow i else phi8Sum v i

/-- `φ8(v) = Σ bit_i(v) · g^i` for bytes `v` (§2.2), tabulated. -/
def phi8Table : Array F128 := Array.ofFn (n := 256) fun v => phi8Sum v.val 8

@[inline] def phi8 (v : Nat) : F128 := phi8Table.getD (v % 256) 0

end Flock
