import Proofs.Flock.Soundness.LigeritoDecode

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

omit [DecidableEq F] in
/-- A polynomial function of degree below `|F|` has one polynomial. -/
theorem poly_eq_of_eval {p p' : F[X]} (hp : p.natDegree < Fintype.card F)
    (hp' : p'.natDegree < Fintype.card F) (h : ∀ ω, p.eval ω = p'.eval ω) : p = p' := by
  have := Polynomial.eq_zero_of_natDegree_lt_card_of_eval_eq_zero (p - p') (f := id)
    Function.injective_id (fun ω => by simp [h ω]) ((natDegree_sub_le _ _).trans_lt (max_lt hp hp'))
  exact sub_eq_zero.1 this

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

theorem two_pow_succ_eq (L : ℕ) : 2 ^ (L + 1) = 2 * 2 ^ L := by ring

theorem lt_two_pow_succ {c L : ℕ} (h : c < 2 ^ L) : c < 2 ^ (L + 1) := by
  have := two_pow_succ_eq L; omega

theorem add_lt_two_pow_succ {c L : ℕ} (h : c < 2 ^ L) : 2 ^ L + c < 2 ^ (L + 1) := by
  have := two_pow_succ_eq L; omega

variable (A : Arith F K) (hA : A.Correct)

omit [DecidableEq F] [DecidableEq K] in
include hA in
theorem lt_card_F {N : ℕ} (hN : N ≤ 2 ^ 42) : N < Fintype.card F := by
  rw [hA.card_F]
  calc N ≤ 2 ^ 42 := hN
    _ < 2 ^ 128 := Nat.pow_lt_pow_right (by norm_num) (by norm_num)

theorem xP_ne_zero (L : ℕ) (hL : L ≤ 40) (c : Fin (2 ^ L)) : xP A hA L hL c ≠ 0 :=
  (xP_linearIndependent A hA L hL).ne_zero c

/-- Level `L + 1`'s polynomial for `c < 2^L` is level `L`'s. -/
theorem xP_succ_low (L : ℕ) (hL : L + 1 ≤ 40) (c : ℕ) (hc : c < 2 ^ L) :
    xP A hA (L + 1) hL ⟨c, lt_two_pow_succ hc⟩ = xP A hA L (by omega) ⟨c, hc⟩ := by
  refine poly_eq_of_eval
    (lt_card_F A hA ((xP_natDegree A hA _ hL _).le.trans (Nat.pow_le_pow_right (by norm_num)
      (by omega))))
    (lt_card_F A hA ((xP_natDegree A hA L _ _).le.trans (Nat.pow_le_pow_right (by norm_num)
      (by omega)))) fun ω => ?_
  rw [← xP_eval, ← xP_eval]
  exact Xhat_succ_low A L c hc ω

/-- `X̂_{2^L + c} = X̂_{2^L}·X̂_c` at level `L + 1`. -/
theorem xP_succ_high (L : ℕ) (hL : L + 1 ≤ 40) (c : ℕ) (hc : c < 2 ^ L) :
    xP A hA (L + 1) hL ⟨2 ^ L + c, add_lt_two_pow_succ hc⟩ =
      xP A hA (L + 1) hL ⟨2 ^ L, by simpa using add_lt_two_pow_succ (Nat.two_pow_pos L)⟩ *
        xP A hA L (by omega) ⟨c, hc⟩ := by
  have hdeg1 := xP_natDegree A hA (L + 1) hL ⟨2 ^ L + c, add_lt_two_pow_succ hc⟩
  have hdeg2 := xP_natDegree A hA (L + 1) hL
    ⟨2 ^ L, by simpa using add_lt_two_pow_succ (Nat.two_pow_pos L)⟩
  have hdeg3 := xP_natDegree A hA L (by omega) ⟨c, hc⟩
  have h40 : 2 ^ (L + 1) ≤ 2 ^ 40 := Nat.pow_le_pow_right (by norm_num) hL
  refine poly_eq_of_eval (lt_card_F A hA (by
      calc _ ≤ 2 ^ (L + 1) := hdeg1.le
        _ ≤ 2 ^ 40 := h40
        _ ≤ 2 ^ 42 := by norm_num))
    (lt_card_F A hA ((natDegree_mul_le).trans (by
      have : 2 ^ L ≤ 2 ^ (L + 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
      calc _ ≤ 2 ^ (L + 1) + 2 ^ (L + 1) := by omega
        _ ≤ 2 ^ 40 + 2 ^ 40 := by omega
        _ ≤ 2 ^ 42 := by norm_num))) fun ω => ?_
  rw [eval_mul, ← xP_eval, ← xP_eval, ← xP_eval]
  simp only
  rw [Xhat_succ_high A L c hc ω, show 2 ^ L = 2 ^ L + 0 by rfl, Xhat_succ_high A L 0
    (Nat.two_pow_pos L) ω]
  simp [Arith.Xhat]

/-- **The degree structure.** At every `L ≤ 40`, `X̂_c` is a polynomial of degree at most `c`. -/
theorem xP_natDegree_le : ∀ (L : ℕ) (hL : L ≤ 40) (c : Fin (2 ^ L)),
    (xP A hA L hL c).natDegree ≤ c.val
  | 0, hL, c => by
      have := xP_natDegree A hA 0 hL c
      simp only [pow_zero, Nat.lt_one_iff] at this
      omega
  | L + 1, hL, c => by
      have hL' : L ≤ 40 := by omega
      have ih := xP_natDegree_le L hL'
      have hpos := Nat.two_pow_pos L
      -- the top monomial of level `L` has degree `2^L − 1`
      have htop : 2 ^ L - 1 ≤ (xP A hA L hL' ⟨2 ^ L - 1, by omega⟩).natDegree := by
        by_contra hlt
        push Not at hlt
        obtain ⟨m, hm⟩ := xP_span A hA L hL' ((Polynomial.X : F[X]) ^ (2 ^ L - 1))
          (by rw [Polynomial.natDegree_X_pow]; omega)
        have hle : (∑ c, C (m c) * xP A hA L hL' c).natDegree ≤ 2 ^ L - 2 := by
          refine natDegree_sum_le_of_forall_le _ _ fun c' _ => (natDegree_C_mul_le _ _).trans ?_
          by_cases hc' : c'.val = 2 ^ L - 1
          · have : c' = ⟨2 ^ L - 1, by omega⟩ := Fin.ext hc'
            rw [this]; omega
          · have := ih c'; have := c'.isLt; omega
        rw [hm, Polynomial.natDegree_X_pow] at hle
        have : 1 ≤ L := by
          by_contra h0
          have : L = 0 := by omega
          subst this
          simp at hlt
        have : 2 ≤ 2 ^ L := by
          calc 2 = 2 ^ 1 := by norm_num
            _ ≤ 2 ^ L := Nat.pow_le_pow_right (by norm_num) this
        omega
      -- so `X̂_{2^L}` has degree at most `2^L`
      have hW : (xP A hA (L + 1) hL
          ⟨2 ^ L, by simpa using add_lt_two_pow_succ (Nat.two_pow_pos L)⟩).natDegree ≤ 2 ^ L := by
        have hprod := xP_succ_high A hA L hL (2 ^ L - 1) (by omega)
        have hdeg := xP_natDegree A hA (L + 1) hL ⟨2 ^ L + (2 ^ L - 1), add_lt_two_pow_succ (by omega)⟩
        rw [hprod, natDegree_mul (xP_ne_zero A hA _ _ _) (xP_ne_zero A hA _ _ _)] at hdeg
        have := two_pow_succ_eq L
        omega
      by_cases hc : c.val < 2 ^ L
      · have := xP_succ_low A hA L hL c.val hc
        rw [show c = ⟨c.val, lt_two_pow_succ hc⟩ from Fin.ext rfl, this]
        exact ih ⟨c.val, hc⟩
      · have hc' : c.val - 2 ^ L < 2 ^ L := by
          have := c.isLt; have := two_pow_succ_eq L; omega
        have hprod := xP_succ_high A hA L hL (c.val - 2 ^ L) hc'
        rw [show c = ⟨2 ^ L + (c.val - 2 ^ L), add_lt_two_pow_succ hc'⟩ from Fin.ext (by
          simp only; omega), hprod]
        refine natDegree_mul_le.trans ?_
        have := ih ⟨c.val - 2 ^ L, hc'⟩
        simp only at this ⊢
        omega

/-- **The first `k` monomials span the polynomials of degree below `k`** (at `L` bits, `k ≤ 2^L`). -/
theorem xP_span_lt (L : ℕ) (hL : L ≤ 40) (k : ℕ) (hk : k ≤ 2 ^ L) (P : F[X]) (hP : P.natDegree < k) :
    ∃ m : Fin k → F, ∑ c, Polynomial.C (m c) * xP A hA L hL (Fin.castLE hk c) = P := by
  classical
  have hk0 : 0 < k := by omega
  have : Nonempty (Fin k) := ⟨⟨0, hk0⟩⟩
  have hdeg : ∀ p : F[X], p.natDegree < k → p ∈ Polynomial.degreeLT F k := fun p hp =>
    Polynomial.mem_degreeLT.2 (lt_of_le_of_lt Polynomial.degree_le_natDegree
      (WithBot.coe_lt_coe.2 hp))
  let b : Fin k → Polynomial.degreeLT F k := fun c => ⟨xP A hA L hL (Fin.castLE hk c),
    hdeg _ (lt_of_le_of_lt (xP_natDegree_le A hA L hL _) c.isLt)⟩
  have hli : LinearIndependent F b :=
    LinearIndependent.of_comp (Polynomial.degreeLT F k).subtype
      ((xP_linearIndependent A hA L hL).comp _ (Fin.castLE_injective hk))
  have hcard : Fintype.card (Fin k) = Module.finrank F (Polynomial.degreeLT F k) := by
    rw [Fintype.card_fin, (Polynomial.degreeLTEquiv F k).finrank_eq, Module.finrank_fin_fun]
  have hspan := hli.span_eq_top_of_card_eq_finrank hcard
  have hmem : (⟨P, hdeg P hP⟩ : Polynomial.degreeLT F k) ∈ Submodule.span F (Set.range b) := by
    rw [hspan]; exact Submodule.mem_top
  obtain ⟨m, hm⟩ := (Submodule.mem_span_range_iff_exists_fun F).1 hmem
  refine ⟨m, ?_⟩
  have := congrArg Subtype.val hm
  simpa [b, Polynomial.smul_eq_C_mul] using this

end FlockSoundness
