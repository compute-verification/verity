import SampledProofs.Wires

/-! The span `Wire.span` computes from a wire's digits bounds the writer of every reader row it covers
(`Wire.writerAt_mem_span`): the first of the lemmas holding `check` to `checkRef`. -/

namespace SampledProofs.Wires

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

end SampledProofs.Wires
