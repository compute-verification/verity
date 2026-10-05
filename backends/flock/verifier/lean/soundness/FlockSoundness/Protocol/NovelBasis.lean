import FlockSoundness.LigeritoDecode

/-!
# The novel basis: `X̂_c` has degree at most `c`

`Arith.Correct.xhat_poly` gives, at every `L ≤ 40`, the `2^L` monomials `X̂_c` as polynomials of
degree below `2^L`, linearly independent. The padded level-0 code needs more: its codewords are
the polynomials of degree below `2^cols + t`, spanned by the first `2^cols + t` monomials at
`cols + 1` bits. This file derives the degree structure from `xhat_poly` alone
(`xP_natDegree_le`), by induction on `L`:

* a polynomial function of degree below `|F|` has one polynomial (`poly_eq_of_eval`), so level
  `L + 1`'s polynomial for `c < 2^L` is level `L`'s, and `X̂_{2^L + c} = X̂_{2^L}·X̂_c`;
* level `L`'s polynomials span the polynomials of degree below `2^L`, so the top one, `X̂_{2^L − 1}`,
  has degree `2^L − 1`;
* `X̂_{2^{L+1} − 1} = X̂_{2^L}·X̂_{2^L − 1}` has degree below `2^{L+1}`, so `X̂_{2^L}` has degree at
  most `2^L`.

Then the first `k ≤ 2^L` monomials span the polynomials of degree below `k` (`xP_span_lt`).
-/

namespace FlockSoundness

open Model Polynomial

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem Xhat_succ_low (A : Arith F K) (L c : ℕ) (hc : c < 2 ^ L) (ω : F) :
    A.Xhat c (L + 1) ω = A.Xhat c L ω := by
  unfold Arith.Xhat
  rw [Fin.prod_univ_castSucc]
  simp [Nat.testBit_eq_false_of_lt hc]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem Xhat_succ_high (A : Arith F K) (L c : ℕ) (hc : c < 2 ^ L) (ω : F) :
    A.Xhat (2 ^ L + c) (L + 1) ω = A.What L ω * A.Xhat c L ω := by
  unfold Arith.Xhat
  rw [Fin.prod_univ_castSucc]
  have h1 : ∀ i : Fin L, (2 ^ L + c).testBit i = c.testBit i := fun i =>
    Nat.testBit_two_pow_add_gt i.isLt c
  have h2 : (2 ^ L + c).testBit L = true := by
    rw [Nat.testBit_two_pow_add_eq, Nat.testBit_eq_false_of_lt hc]; rfl
  simp only [Fin.coe_castSucc, Fin.val_last, h1, h2, if_true]
  ring

variable (A : Arith F K) (hA : A.Correct)

end FlockSoundness
