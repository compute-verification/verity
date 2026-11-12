module

public import Spot.Wires

/-! The span `Wire.span` computes from a wire's digits bounds the writer of every reader row it covers
(`Wire.writerAt_mem_span`): the first of the lemmas holding `check` to `checkRef`. The box `Wire.box` bounds the writers
of the reader indices between two (`Wire.writerAt_mem_box`). -/

@[expose] public section

namespace Spot.Wires

/-- One digit's term lies between `min 0 (s (r − 1))` and `max 0 (s (r − 1))`. -/
theorem digit_bounds (d r : Nat) (s : Int) (hd : d < r) :
    min 0 (s * ((r : Int) - 1)) ≤ (d : Int) * s ∧ (d : Int) * s ≤ max 0 (s * ((r : Int) - 1)) := by
  have hd' : (d : Int) ≤ (r : Int) - 1 := by omega
  have h0 : (0 : Int) ≤ d := by omega
  rw [Int.mul_comm s]
  rcases Int.le_total 0 s with hs | hs
  · have a := Int.mul_le_mul_of_nonneg_right hd' hs
    have b := Int.mul_nonneg h0 hs
    constructor
    · exact Int.le_trans (Int.min_le_left _ _) b
    · exact Int.le_trans a (Int.le_max_right _ _)
  · have a := Int.mul_le_mul_of_nonpos_right hd' hs
    have b := Int.mul_nonpos_of_nonneg_of_nonpos h0 hs
    constructor
    · exact Int.le_trans (Int.min_le_right _ _) a
    · exact Int.le_trans b (Int.le_max_left _ _)

theorem accum_bounds : ∀ (ps : List (Nat × Int)) (i : Nat), i < prodR ps →
    lowOf ps ≤ accum ps i ∧ accum ps i ≤ highOf ps
  | [], _, _ => by simp [accum, lowOf, highOf]
  | (r, s) :: ps, i, hi => by
    have hi' : i < r * prodR ps := hi
    have h1 : 1 ≤ r := Nat.pos_of_ne_zero fun h => by simp [h] at hi'
    have hq : i / r < prodR ps :=
      (Nat.div_lt_iff_lt_mul (by omega)).2 (by rw [Nat.mul_comm]; exact hi')
    have ih := accum_bounds ps (i / r) hq
    have hb := digit_bounds (i % r) r s (Nat.mod_lt _ (by omega))
    simp only [accum, lowOf, highOf]
    omega

/-- **Every writer a wire reads lies in its span**: `check`'s bounds in closed form bound the writer of every reader row
the wire covers. -/
theorem Wire.writerAt_mem_span (w : Wire) (i : Nat) (hi : i < w.units) :
    w.span.1 ≤ w.writerAt i ∧ w.writerAt i ≤ w.span.2 := by
  have := accum_bounds w.pairs i hi
  simp only [Wire.span, Wire.writerAt]
  omega

/-- A term `d · s` with `d` between `a` and `b` lies between theirs. -/
theorem term_between {a d b : Nat} (s : Int) (h1 : a ≤ d) (h2 : d ≤ b) :
    min ((a : Int) * s) ((b : Int) * s) ≤ (d : Int) * s ∧ (d : Int) * s ≤ max ((a : Int) * s) ((b : Int) * s) := by
  have h1' : (a : Int) ≤ d := by omega
  have h2' : (d : Int) ≤ b := by omega
  rcases Int.le_total 0 s with hs | hs
  · exact ⟨Int.le_trans (Int.min_le_left _ _) (Int.mul_le_mul_of_nonneg_right h1' hs),
      Int.le_trans (Int.mul_le_mul_of_nonneg_right h2' hs) (Int.le_max_right _ _)⟩
  · exact ⟨Int.le_trans (Int.min_le_right _ _) (Int.mul_le_mul_of_nonpos_right h2' hs),
      Int.le_trans (Int.mul_le_mul_of_nonpos_right h1' hs) (Int.le_max_left _ _)⟩

theorem accum_box : ∀ (ps : List (Nat × Int)) {lo hi i : Nat}, lo ≤ i → i ≤ hi →
    boxLow ps lo hi ≤ accum ps i ∧ accum ps i ≤ boxHigh ps lo hi
  | [], _, _, _, _, _ => by simp [accum, boxLow, boxHigh]
  | (r, s) :: ps, lo, hi, i, h1, h2 => by
    have q1 : lo / r ≤ i / r := Nat.div_le_div_right h1
    have q2 : i / r ≤ hi / r := Nat.div_le_div_right h2
    simp only [accum, boxLow, boxHigh]
    split
    · rename_i heq
      have hq : i / r = lo / r := by omega
      have dl := Nat.div_add_mod lo r
      have di := Nat.div_add_mod i r
      have dh := Nat.div_add_mod hi r
      rw [hq] at di
      rw [← heq] at dh
      generalize r * (lo / r) = m at dl di dh
      have := term_between s (show lo % r ≤ i % r by omega) (show i % r ≤ hi % r by omega)
      rw [hq]
      omega
    · rename_i hne
      have hr : r ≠ 0 := fun h => hne (by simp [h])
      have hb := digit_bounds (i % r) r s (Nat.mod_lt _ (Nat.pos_of_ne_zero hr))
      have ih := accum_box ps q1 q2
      omega

/-- **The writers of reader indices `lo … hi` lie in the wire's box there.** -/
theorem Wire.writerAt_mem_box (w : Wire) {lo hi i : Nat} (h1 : lo ≤ i) (h2 : i ≤ hi) :
    (w.box lo hi).1 ≤ w.writerAt i ∧ w.writerAt i ≤ (w.box lo hi).2 := by
  have := accum_box w.pairs h1 h2
  simp only [Wire.box, Wire.writerAt]
  omega

end Spot.Wires
