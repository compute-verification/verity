import Proofs.Flock.Soundness.Discharge.Hm.Gadget
import Proofs.Flock.Level3.GF128

/-!
# V*'s gadgets: Karatsuba multiplication in GF(2^128)

A GF(2^128) product of two hidden elements as rows of the verifier's builder (`Flock.HmNets`), each element 128 forms,
bit `k` being bit `k` of `2^64·hi + lo` (`Flock.F128`):

* `kara n`: Karatsuba on two polynomials of `2^n` coefficients (`gf2k.karatsuba`). One row for `n = 0`; otherwise three
  half-size products (the low halves, the high halves, the halves' sums, in that order) recombined by XOR, which costs no
  row. `kara 7` is `3^7 = 2,187` rows.
* `reduceG`: the reduction modulo `G = X^128 + X^7 + X^2 + X + 1` (`gf2k.reduce_forms`), linear, no row: `lo + X^128·hi`
  becomes `lo + (X^7 + X^2 + X + 1)·hi`, twice (`foldG`).

`sound_mul128`: whenever every row holds and the input forms carry the bits of `x` and `y`, the output forms carry the
bits of `x * y`, the executable's product, which is level 3's GF(2^128) product (`F128.poly_mul`).
-/

namespace FlockVBridge

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm Polynomial
open FlockLevel3 (P b2 b2_true b2_false b2_xor b2_and w128 w128_lt coeff_natPoly G_monic G_degree two_eq_zero_P)

/-! ### Forms as polynomials -/

/-- Coefficient-wise XOR of two lists of forms, the shorter padded with zero. -/
def addL : List Lin → List Lin → List Lin
  | [], g => g
  | f :: fs, [] => f :: fs
  | f :: fs, g :: gs => (f ^^^ g) :: addL fs gs

/-- Multiplication by `X^k`. -/
def shiftL (k : ℕ) (f : List Lin) : List Lin := List.replicate k Lin.zero ++ f

/-- The polynomial over GF(2) whose coefficient `i` is form `i`'s value under `z`. -/
noncomputable def polyOf (z : Var → Bool) : List Lin → P
  | [] => 0
  | f :: fs => C (b2 (ev z f)) + X * polyOf z fs

theorem coeff_polyOf (z : Var → Bool) :
    ∀ (f : List Lin) (i : ℕ), (polyOf z f).coeff i = b2 (ev z (f.getD i Lin.zero))
  | [], i => by simp [polyOf]
  | f :: fs, 0 => by simp [polyOf]
  | f :: fs, i + 1 => by simp [polyOf, coeff_polyOf z fs i]

theorem polyOf_addL (z : Var → Bool) : ∀ f g : List Lin, polyOf z (addL f g) = polyOf z f + polyOf z g
  | [], g => by simp [addL, polyOf]
  | f :: fs, [] => by simp [addL, polyOf]
  | f :: fs, g :: gs => by
      simp only [addL, polyOf, ev_xor, b2_xor, map_add, polyOf_addL z fs gs]
      ring

theorem polyOf_shiftL (z : Var → Bool) (k : ℕ) (f : List Lin) : polyOf z (shiftL k f) = X ^ k * polyOf z f := by
  induction k with
  | zero => simp [shiftL]
  | succ k ih =>
      simp only [shiftL, List.replicate_succ, List.cons_append, polyOf, ev_zero, b2_false, map_zero, zero_add] at ih ⊢
      rw [ih]
      ring

theorem polyOf_take_drop (z : Var → Bool) :
    ∀ (h : ℕ) (f : List Lin), polyOf z f = polyOf z (f.take h) + X ^ h * polyOf z (f.drop h)
  | 0, f => by simp [polyOf]
  | h + 1, [] => by simp [polyOf]
  | h + 1, f :: fs => by
      simp only [List.take_succ_cons, List.drop_succ_cons, polyOf]
      rw [polyOf_take_drop z h fs]
      ring

theorem polyOf_le_one (z : Var → Bool) : ∀ a : List Lin, a.length ≤ 1 → polyOf z a = C (b2 (ev z (a.headD Lin.zero)))
  | [], _ => by simp [polyOf]
  | [f], _ => by simp [polyOf]
  | _ :: _ :: _, h => by simp at h

theorem length_addL : ∀ f g : List Lin, (addL f g).length = max f.length g.length
  | [], g => by simp [addL]
  | f :: fs, [] => by simp [addL]
  | f :: fs, g :: gs => by
      simp only [addL, List.length_cons, length_addL fs gs]
      omega

theorem coeff_polyOf_ge (z : Var → Bool) (f : List Lin) (i : ℕ) (hi : f.length ≤ i) : (polyOf z f).coeff i = 0 := by
  rw [coeff_polyOf, List.getD_eq_default _ _ hi, ev_zero, b2_false]

theorem degree_polyOf_lt (z : Var → Bool) (f : List Lin) {n : ℕ} (h : f.length ≤ n) : (polyOf z f).degree < n := by
  rw [degree_lt_iff_coeff_zero]
  exact fun m hm => coeff_polyOf_ge z f m (h.trans hm)

/-! ### Karatsuba -/

/-- **Karatsuba** on two polynomials of at most `2^n` coefficients each: the product's coefficients. One row for
`n = 0`; otherwise the products of the low halves, of the high halves and of the halves' sums, in that order, and XORs. -/
def kara : ℕ → List Lin → List Lin → M (List Lin)
  | 0, a, b => do
      let c ← fresh (a.headD Lin.zero) (b.headD Lin.zero)
      pure [c]
  | n + 1, a, b => do
      let p0 ← kara n (a.take (2 ^ n)) (b.take (2 ^ n))
      let p2 ← kara n (a.drop (2 ^ n)) (b.drop (2 ^ n))
      let p1 ← kara n (addL (a.take (2 ^ n)) (a.drop (2 ^ n))) (addL (b.take (2 ^ n)) (b.drop (2 ^ n)))
      pure (addL (addL p0 (shiftL (2 ^ n) (addL (addL p1 p0) p2))) (shiftL (2 ^ n * 2) p2))

/-- **Karatsuba is the product.** Whenever every row holds, the result has at most `2·2^n − 1` coefficients and is the
product of the inputs' polynomials. -/
theorem sound_kara : ∀ (n : ℕ) (a b : List Lin), a.length ≤ 2 ^ n → b.length ≤ 2 ^ n →
    Sound (kara n a b) fun z c => c.length ≤ 2 * 2 ^ n - 1 ∧ polyOf z c = polyOf z a * polyOf z b
  | 0, a, b, ha, hb => by
      unfold kara
      refine Sound.bind' (sound_fresh _ _) fun c => Sound.pure' fun z hc => ⟨by simp, ?_⟩
      rw [polyOf_le_one z a (by simpa using ha), polyOf_le_one z b (by simpa using hb)]
      simp [polyOf, hc, b2_and]
  | n + 1, a, b, ha, hb => by
      have hpos := Nat.two_pow_pos n
      have hlt : ∀ f : List Lin, f.length ≤ 2 ^ (n + 1) → (f.take (2 ^ n)).length ≤ 2 ^ n ∧
          (f.drop (2 ^ n)).length ≤ 2 ^ n ∧ (addL (f.take (2 ^ n)) (f.drop (2 ^ n))).length ≤ 2 ^ n := by
        intro f hf
        simp only [List.length_take, List.length_drop, length_addL]
        rw [pow_succ] at hf
        generalize 2 ^ n = h at *
        omega
      obtain ⟨ha0, ha1, ha2⟩ := hlt a ha
      obtain ⟨hb0, hb1, hb2⟩ := hlt b hb
      unfold kara
      refine Sound.bind' (sound_kara n _ _ ha0 hb0) fun p0 => ?_
      refine Sound.bind' (sound_kara n _ _ ha1 hb1) fun p2 => ?_
      refine Sound.bind' (sound_kara n _ _ ha2 hb2) fun p1 => ?_
      refine Sound.pure' fun z h1 h2 h0 => ⟨?_, ?_⟩
      · simp only [length_addL, shiftL, List.length_append, List.length_replicate]
        have l0 := h0.1; have l1 := h1.1; have l2 := h2.1
        rw [pow_succ]
        generalize 2 ^ n = h at *
        omega
      · simp only [polyOf_addL, polyOf_shiftL] at h1 ⊢
        rw [polyOf_take_drop z (2 ^ n) a, polyOf_take_drop z (2 ^ n) b]
        generalize 2 ^ n = h at *
        linear_combination (1 + X ^ h) * h0.2 + X ^ h * h1.2 + (X ^ h + X ^ (h * 2)) * h2.2 +
          (X ^ h * (polyOf z (a.take h) * polyOf z (b.take h) + polyOf z (a.drop h) * polyOf z (b.drop h))) *
            two_eq_zero_P

/-! ### The reduction modulo `G` -/

/-- Multiplication by `S = X^7 + X^2 + X + 1`. -/
def mulS (g : List Lin) : List Lin := addL (addL (addL g (shiftL 1 g)) (shiftL 2 g)) (shiftL 7 g)

/-- One fold modulo `G`: `lo + X^128·hi` becomes `lo + S·hi`, since `X^128 = S` modulo `G` over GF(2). -/
def foldG (f : List Lin) : List Lin := addL (f.take 128) (mulS (f.drop 128))

/-- The reduction of a product of two elements (at most 255 coefficients): two folds. -/
def reduceG (f : List Lin) : List Lin := foldG (foldG f)

theorem polyOf_mulS (z : Var → Bool) (g : List Lin) : polyOf z (mulS g) = FlockLevel3.S * polyOf z g := by
  simp only [mulS, polyOf_addL, polyOf_shiftL, FlockLevel3.S]
  ring

theorem polyOf_foldG (z : Var → Bool) (f : List Lin) :
    polyOf z (foldG f) = polyOf z f + FlockLevel3.G * polyOf z (f.drop 128) := by
  rw [foldG, polyOf_addL, polyOf_mulS, polyOf_take_drop z 128 f]
  unfold FlockLevel3.G
  linear_combination (-(X ^ 128 * polyOf z (f.drop 128))) * two_eq_zero_P

theorem length_mulS (g : List Lin) : (mulS g).length = g.length + 7 := by
  simp only [mulS, length_addL, shiftL, List.length_append, List.length_replicate]
  omega

theorem length_foldG (f : List Lin) : (foldG f).length = max (min 128 f.length) (f.length - 128 + 7) := by
  simp only [foldG, length_addL, length_mulS, List.length_take, List.length_drop]

theorem length_reduceG (f : List Lin) (h : f.length ≤ 255) : (reduceG f).length ≤ 128 := by
  simp only [reduceG, length_foldG]
  omega

/-- **The reduction is `%ₘ G`.** -/
theorem polyOf_reduceG (z : Var → Bool) (f : List Lin) (h : f.length ≤ 255) :
    polyOf z (reduceG f) = polyOf z f %ₘ FlockLevel3.G := by
  have hdeg := degree_polyOf_lt z _ (length_reduceG f h)
  have hr : polyOf z (reduceG f) =
      polyOf z f + FlockLevel3.G * (polyOf z (f.drop 128) + polyOf z ((foldG f).drop 128)) := by
    rw [reduceG, polyOf_foldG, polyOf_foldG]
    ring
  calc polyOf z (reduceG f) = polyOf z (reduceG f) %ₘ FlockLevel3.G :=
        ((modByMonic_eq_self_iff G_monic).2 (by rw [G_degree]; exact_mod_cast hdeg)).symm
    _ = polyOf z f %ₘ FlockLevel3.G := modByMonic_eq_of_dvd_sub G_monic
      ⟨polyOf z (f.drop 128) + polyOf z ((foldG f).drop 128), by rw [hr]; ring⟩

/-! ### GF(2^128) multiplication -/

/-- **GF(2^128) multiplication**: Karatsuba, then the reduction. -/
def mul128 (a b : List Lin) : M (List Lin) := do
  let p ← kara 7 a b
  pure (reduceG p)

/-- Bit `k` of an element of `F128`: bit `k` of `2^64·hi + lo`. -/
def bitOf (x : Flock.F128) (k : ℕ) : Bool := (w128 x.lo x.hi).testBit k

theorem b2_inj {x y : Bool} (h : b2 x = b2 y) : x = y := by
  cases x <;> cases y <;> first | rfl | exact absurd h (by decide)

theorem polyOf_eq_poly (z : Var → Bool) (a : List Lin) (ha : a.length = 128) (x : Flock.F128)
    (hx : ∀ k < 128, ev z (a.getD k Lin.zero) = bitOf x k) : polyOf z a = FlockLevel3.F128.poly x := by
  ext k
  rw [coeff_polyOf, FlockLevel3.F128.poly, coeff_natPoly]
  by_cases hk : k < 128
  · rw [hx k hk, bitOf]
  · rw [List.getD_eq_default _ _ (by omega), ev_zero,
      Nat.testBit_lt_two_pow (lt_of_lt_of_le (w128_lt _ _) (Nat.pow_le_pow_right (by norm_num) (by omega)))]

/-- **The Karatsuba circuit is GF(2^128) multiplication.** Whenever every row holds and the input forms carry the bits of
`x` and `y`, the output forms carry the bits of `x * y` (`Flock.F128.mul`, level 3's product by `F128.poly_mul`). -/
theorem sound_mul128 (a b : List Lin) (ha : a.length = 128) (hb : b.length = 128) :
    Sound (mul128 a b) fun z c => ∀ x y : Flock.F128, (∀ k < 128, ev z (a.getD k Lin.zero) = bitOf x k) →
      (∀ k < 128, ev z (b.getD k Lin.zero) = bitOf y k) → ∀ k < 128, ev z (c.getD k Lin.zero) = bitOf (x * y) k := by
  unfold mul128
  refine Sound.bind' (sound_kara 7 a b (by rw [ha]; norm_num) (by rw [hb]; norm_num)) fun p => Sound.pure' fun z hp x y hx hy k _ => ?_
  have hc : polyOf z (reduceG p) = FlockLevel3.F128.poly (x * y) := by
    rw [polyOf_reduceG z p (by simpa using hp.1), hp.2, polyOf_eq_poly z a ha x hx, polyOf_eq_poly z b hb y hy,
      FlockLevel3.F128.poly_mul]
  have := congrArg (fun q => q.coeff k) hc
  simp only [coeff_polyOf, FlockLevel3.F128.poly, coeff_natPoly] at this
  exact b2_inj this

end FlockVBridge
