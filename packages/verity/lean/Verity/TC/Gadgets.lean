import Verity.TC.Relation

/-!
# The range and is-zero gadgets, chunk by chunk

`Relation.RANGE` and `Relation.ISZERO` are the predicates the gadgets of `relation.Census` enforce. Here the gadgets
themselves, with their chunk columns as a circuit holds them:

* `Census.RANGE x bits`: `x` split into little-endian chunks of 16 bits (the last one `bits % 16` wide), one linear
  split and one range-table lookup per chunk (`rangeWidths`, `Chunks`). `range_gadget`: it holds exactly when
  `0 ≤ x < 2^bits`.
* `Census.ISZERO x bits z`: `x` split into chunks of 12 bits, one is-zero lookup per chunk, the flags multiplied into
  `z` (`isZeroWidths`, `ZeroChunks`). `iszero_gadget`: it holds exactly when `x` is in range and `z = [x = 0]`.
-/

namespace Verity.TC.Gadgets

open Verity.TC.Relation

/-- `x` is the little-endian sum of chunks of the given widths, each in its range table. -/
def Chunks : List Nat → Int → Prop
  | [], x => x = 0
  | w :: ws, x => ∃ c y, 0 ≤ c ∧ c < 2 ^ w ∧ Chunks ws y ∧ x = c + 2 ^ w * y

/-- The chunk widths `Census.RANGE` uses: 16-bit chunks, the last `bits % 16` wide. -/
def rangeWidths (bits : Nat) : List Nat :=
  List.replicate (bits / 16) 16 ++ (if bits % 16 ≠ 0 then [bits % 16] else [])

/-- The chunk widths `Census.ISZERO` uses: 12-bit chunks, the last `bits % 12` wide. -/
def isZeroWidths (bits : Nat) : List Nat :=
  List.replicate (bits / 12) 12 ++ (if bits % 12 ≠ 0 then [bits % 12] else [])

/-- `x` is the little-endian sum of in-range chunks, and `z` is the product of their is-zero flags. -/
def ZeroChunks : List Nat → Int → Int → Prop
  | [], x, z => x = 0 ∧ z = 1
  | w :: ws, x, z => ∃ c y z', 0 ≤ c ∧ c < 2 ^ w ∧ ZeroChunks ws y z' ∧ x = c + 2 ^ w * y ∧
      z = (if c = 0 then 1 else 0) * z'

theorem pow_pos' (w : Nat) : (0 : Int) < 2 ^ w := by
  have := Nat.two_pow_pos w
  exact_mod_cast this

theorem chunks_iff : ∀ (ws : List Nat) (x : Int), Chunks ws x ↔ 0 ≤ x ∧ x < 2 ^ ws.sum
  | [], x => by simp [Chunks]; omega
  | w :: ws, x => by
    simp only [Chunks, List.sum_cons]
    have hp := pow_pos' w
    have hp' := pow_pos' ws.sum
    have hpow : (2 : Int) ^ (w + ws.sum) = 2 ^ w * 2 ^ ws.sum := Int.pow_add _ _ _
    constructor
    · rintro ⟨c, y, hc0, hc1, hy, rfl⟩
      obtain ⟨hy0, hy1⟩ := (chunks_iff ws y).mp hy
      refine ⟨by have := Int.mul_nonneg (Int.le_of_lt hp) hy0; omega, ?_⟩
      rw [hpow]
      have : 2 ^ w * (y + 1) ≤ 2 ^ w * 2 ^ ws.sum := Int.mul_le_mul_of_nonneg_left (by omega) (Int.le_of_lt hp)
      rw [Int.mul_add, Int.mul_one] at this
      omega
    · rintro ⟨hx0, hx1⟩
      refine ⟨x % 2 ^ w, x / 2 ^ w, Int.emod_nonneg _ (Int.ne_of_gt hp), Int.emod_lt_of_pos _ hp,
        (chunks_iff ws _).mpr ⟨Int.ediv_nonneg hx0 (Int.le_of_lt hp), ?_⟩, ?_⟩
      · rw [hpow] at hx1
        exact Int.ediv_lt_of_lt_mul hp (by rw [Int.mul_comm]; exact hx1)
      · have := Int.mul_ediv_add_emod x (2 ^ w); omega

theorem sum_replicate (n w : Nat) : (List.replicate n w).sum = n * w := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.replicate_succ, ih, Nat.succ_mul]; omega

theorem rangeWidths_sum (bits : Nat) : (rangeWidths bits).sum = bits := by
  unfold rangeWidths
  rw [List.sum_append, sum_replicate]
  split <;> simp <;> omega

theorem isZeroWidths_sum (bits : Nat) : (isZeroWidths bits).sum = bits := by
  unfold isZeroWidths
  rw [List.sum_append, sum_replicate]
  split <;> simp <;> omega

/-- `Census.RANGE`'s chunks hold exactly when `x ∈ [0, 2^bits)`: `RANGE`. -/
theorem range_gadget (x : Int) (bits : Nat) : Chunks (rangeWidths bits) x ↔ RANGE x bits := by
  rw [chunks_iff, rangeWidths_sum]; rfl

theorem zeroChunks_iff : ∀ (ws : List Nat) (x z : Int),
    ZeroChunks ws x z ↔ (0 ≤ x ∧ x < 2 ^ ws.sum) ∧ z = if x = 0 then 1 else 0
  | [], x, z => by
    simp only [ZeroChunks, List.sum_nil]
    constructor
    · rintro ⟨rfl, rfl⟩; simp
    · rintro ⟨⟨h1, h2⟩, h3⟩
      have : x = 0 := by simp at h2; omega
      subst this; simp at h3; exact ⟨rfl, h3⟩
  | w :: ws, x, z => by
    simp only [ZeroChunks, List.sum_cons]
    have hp := pow_pos' w
    have hp' := pow_pos' ws.sum
    have hpow : (2 : Int) ^ (w + ws.sum) = 2 ^ w * 2 ^ ws.sum := Int.pow_add _ _ _
    constructor
    · rintro ⟨c, y, z', hc0, hc1, hy, rfl, rfl⟩
      obtain ⟨⟨hy0, hy1⟩, rfl⟩ := (zeroChunks_iff ws y z').mp hy
      have hr := (chunks_iff (w :: ws) (c + 2 ^ w * y)).mp ⟨c, y, hc0, hc1, (chunks_iff ws y).mpr ⟨hy0, hy1⟩, rfl⟩
      simp only [List.sum_cons] at hr
      refine ⟨hr, ?_⟩
      have hm := Int.mul_nonneg (Int.le_of_lt hp) hy0
      by_cases hc : c = 0
      · subst hc
        by_cases hy : y = 0
        · subst hy; simp
        · have : 2 ^ w * y ≠ 0 := Int.mul_ne_zero (Int.ne_of_gt hp) hy
          simp [hy, this]
      · have : c + 2 ^ w * y ≠ 0 := by omega
        simp [hc, this]
    · rintro ⟨⟨hx0, hx1⟩, rfl⟩
      obtain ⟨c, y, hc0, hc1, hy, hxy⟩ := (chunks_iff (w :: ws) x).mpr (by simp only [List.sum_cons]; exact ⟨hx0, hx1⟩)
      obtain ⟨hy0, hy1⟩ := (chunks_iff ws y).mp hy
      refine ⟨c, y, if y = 0 then 1 else 0, hc0, hc1, (zeroChunks_iff ws y _).mpr ⟨⟨hy0, hy1⟩, rfl⟩, hxy, ?_⟩
      have hm := Int.mul_nonneg (Int.le_of_lt hp) hy0
      by_cases hc : c = 0
      · subst hc
        by_cases hyz : y = 0
        · subst hyz; simp [hxy]
        · have : 2 ^ w * y ≠ 0 := Int.mul_ne_zero (Int.ne_of_gt hp) hyz
          simp [hxy, hyz, this]
      · have : x ≠ 0 := by omega
        simp [hc, this]

/-- `Census.ISZERO`'s chunks and flags hold exactly when `x ∈ [0, 2^bits)` and `z = [x = 0]`: `ISZERO`. -/
theorem iszero_gadget (x z : Int) (bits : Nat) : ZeroChunks (isZeroWidths bits) x z ↔ ISZERO x bits z := by
  rw [zeroChunks_iff, isZeroWidths_sum]; rfl

end Verity.TC.Gadgets
