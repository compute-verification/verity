import FlockSoundness.Defs
import FlockSoundness.Game.Poly
import FlockSoundness.Game.Union

/-!
# The opening phase: ring switching and batching (PROTOCOL.md §12)

From a doomed start (for every codeword near the committed table, some claim handed to the
opening is false), the opening ends doomed (no codeword near the table has the batched value)
except with probability `L · (7·n + 1)/|F|`, for `n` claims and `L` codewords (`opening_phase_of_card`).

The argument, per codeword `M` with witness `z` (paper Appendix B):

* **The honest message.** For a claim with point `x = (x₀, x')`, the honest ring-switching
  message is `ŝ[v] = Σ_y eq(x', y)·z[128y + v]`, and the claim's value is
  `Σ_v w[v]·ŝ[v]` (`claim_eval_eq`). So the prover passes the check on a false claim only with a
  message `s ≠ ŝ`.
* **Ring switching.** The verifier's recombination of the honest message is the inner product
  of the packed table with `Φ_{r''}(eq(x', ·))` (`recombine_shat`), the quantity the batched
  claim `Σ_y f(y)·b(y) = T` checks.
* **One claim.** For `s ≠ ŝ`, `recombine(s, r'') − recombine(ŝ, r'')` is a nonzero multilinear
  polynomial in the seven coins `r''` (`recombine_poly`), so it vanishes with probability at most
  `7/|F|` (`prCoin_recombine_le`).
* **The batch.** A nonzero linear form in the batching coins vanishes with probability at most
  `1/|F|` (`prCoin_batch_le`).
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Equality weights at Boolean points -/

/-- In characteristic 2, the equality weight at a Boolean point is the indicator of that point. -/
theorem eqAt_bool {R : Type} [CommRing R] [CharP R 2] {n : ℕ} (u u₀ : ℕ) (hu : u < 2 ^ n)
    (hu₀ : u₀ < 2 ^ n) :
    eqAt (fun j : Fin n => if u₀.testBit j then (1 : R) else 0) u = if u = u₀ then 1 else 0 := by
  have h11 : (1 : R) + 1 = 0 := by
    have := CharP.cast_eq_zero R 2
    rw [Nat.cast_ofNat] at this
    rw [← this]; norm_num [one_add_one_eq_two]
  unfold eqAt
  by_cases heq : u = u₀
  · subst heq
    rw [if_pos rfl]
    refine Finset.prod_eq_one fun j _ => ?_
    by_cases hb : u.testBit j <;> simp [hb]
  · rw [if_neg heq]
    -- some bit below `n` differs
    have : ∃ j : Fin n, u.testBit j ≠ u₀.testBit j := by
      by_contra hall
      push Not at hall
      apply heq
      refine Nat.eq_of_testBit_eq fun i => ?_
      by_cases hi : i < n
      · exact hall ⟨i, hi⟩
      · have h1 : u.testBit i = false := Nat.testBit_eq_false_of_lt
          (lt_of_lt_of_le hu (Nat.pow_le_pow_right (by norm_num) (by omega)))
        have h2 : u₀.testBit i = false := Nat.testBit_eq_false_of_lt
          (lt_of_lt_of_le hu₀ (Nat.pow_le_pow_right (by norm_num) (by omega)))
        rw [h1, h2]
    obtain ⟨j, hj⟩ := this
    refine Finset.prod_eq_zero (Finset.mem_univ j) ?_
    cases h : u.testBit j <;> cases h' : u₀.testBit j <;> simp_all

/-! ## Multilinear polynomials in the equality basis -/

/-- The multilinear polynomial `Σ_u d[u]·eq(X, u)` over `n` variables. -/
noncomputable def eqPoly {n : ℕ} (d : Fin (2 ^ n) → F) : MvPolynomial (Fin n) F :=
  ∑ u : Fin (2 ^ n), MvPolynomial.C (d u) *
    ∏ j : Fin n, (if u.val.testBit j then MvPolynomial.X j else 1 + MvPolynomial.X j)

omit [Fintype F] [DecidableEq F] in
theorem eval_eqPoly {n : ℕ} (d : Fin (2 ^ n) → F) (r : Fin n → F) :
    MvPolynomial.eval r (eqPoly d) = ∑ u : Fin (2 ^ n), d u * eqAt r u.val := by
  unfold eqPoly eqAt
  simp only [map_sum, map_mul, MvPolynomial.eval_C, map_prod]
  refine Finset.sum_congr rfl fun u _ => ?_
  congr 1
  refine Finset.prod_congr rfl fun j _ => ?_
  split_ifs <;> simp

omit [Fintype F] [DecidableEq F] in
theorem totalDegree_eqPoly_le {n : ℕ} (d : Fin (2 ^ n) → F) : (eqPoly d).totalDegree ≤ n := by
  unfold eqPoly
  refine (MvPolynomial.totalDegree_finsetSum _ _).trans (Finset.sup_le fun u _ => ?_)
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  rw [MvPolynomial.totalDegree_C, zero_add]
  refine (MvPolynomial.totalDegree_finsetProd _ _).trans ?_
  calc ∑ j : Fin n, (if u.val.testBit j then MvPolynomial.X j else 1 + MvPolynomial.X j :
        MvPolynomial (Fin n) F).totalDegree
      ≤ ∑ _j : Fin n, 1 := Finset.sum_le_sum fun j _ => by
        split_ifs
        · rw [MvPolynomial.totalDegree_X]
        · refine (MvPolynomial.totalDegree_add _ _).trans (max_le ?_ ?_)
          · rw [MvPolynomial.totalDegree_one]; omega
          · rw [MvPolynomial.totalDegree_X]
    _ = n := by simp

theorem eqPoly_ne_zero [CharP F 2] {n : ℕ} (d : Fin (2 ^ n) → F) (u₀ : Fin (2 ^ n))
    (hd : d u₀ ≠ 0) : eqPoly d ≠ 0 := by
  intro h0
  have := congrArg (MvPolynomial.eval (fun j : Fin n => if u₀.val.testBit j then (1 : F) else 0)) h0
  rw [eval_eqPoly, map_zero] at this
  have hsum : ∑ u : Fin (2 ^ n), d u * eqAt (fun j : Fin n => if u₀.val.testBit j then (1 : F) else 0)
      u.val = d u₀ := by
    rw [Finset.sum_eq_single u₀]
    · rw [eqAt_bool u₀.val u₀.val u₀.isLt u₀.isLt, if_pos rfl, mul_one]
    · intro u _ hu
      rw [eqAt_bool u.val u₀.val u.isLt u₀.isLt, if_neg (fun h => hu (Fin.ext h)), mul_zero]
    · simp
  exact hd (hsum ▸ this)

/-- **A nonzero multilinear function vanishes rarely.** If some coefficient in the equality basis
is nonzero, `Σ_u d[u]·eq(r, u)` vanishes at a uniform `r` with probability at most `n/|F|`. -/
theorem prCoin_eqSum_le [CharP F 2] {n : ℕ} (d : Fin (2 ^ n) → F) (u₀ : Fin (2 ^ n))
    (hd : d u₀ ≠ 0) :
    prCoin (fun r : Fin n → F => ∑ u : Fin (2 ^ n), d u * eqAt r u.val = 0) ≤
      (n : ℝ≥0∞) / Fintype.card F := by
  calc prCoin (fun r : Fin n → F => ∑ u : Fin (2 ^ n), d u * eqAt r u.val = 0)
      = prCoin (fun r : Fin n → F => MvPolynomial.eval r (eqPoly d) = 0) := by
        congr 1; funext r; rw [eval_eqPoly]
    _ ≤ ((eqPoly d).totalDegree : ℝ≥0∞) / Fintype.card F :=
        prCoin_mvEval_eq_zero_le _ (eqPoly_ne_zero d u₀ hd)
    _ ≤ (n : ℝ≥0∞) / Fintype.card F := by
        gcongr; exact_mod_cast totalDegree_eqPoly_le d

/-- **The batch.** A nonzero linear form in uniform coins vanishes with probability at most
`1/|F|`. -/
theorem prCoin_batch_le {n : ℕ} (d : Fin n → F) (k₀ : Fin n) (hd : d k₀ ≠ 0) :
    prCoin (fun γ : Fin n → F => ∑ k, γ k * d k = 0) ≤ 1 / Fintype.card F := by
  classical
  let Q : MvPolynomial (Fin n) F := ∑ k, MvPolynomial.C (d k) * MvPolynomial.X k
  have hevalQ : ∀ γ, MvPolynomial.eval γ Q = ∑ k, γ k * d k := fun γ => by
    simp only [Q, map_sum, map_mul, MvPolynomial.eval_C, MvPolynomial.eval_X, mul_comm]
  have hQ : Q ≠ 0 := by
    intro h0
    have := congrArg (MvPolynomial.eval (fun k => if k = k₀ then (1 : F) else 0)) h0
    rw [hevalQ, map_zero, Finset.sum_eq_single k₀ (fun k _ hk => by simp [hk])
      (by simp)] at this
    simp at this
    exact hd this
  have hdeg : Q.totalDegree ≤ 1 := by
    refine (MvPolynomial.totalDegree_finsetSum _ _).trans (Finset.sup_le fun k _ => ?_)
    refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_C, MvPolynomial.totalDegree_X]
  calc prCoin (fun γ : Fin n → F => ∑ k, γ k * d k = 0)
      = prCoin (fun γ : Fin n → F => MvPolynomial.eval γ Q = 0) := by
        congr 1; funext γ; rw [hevalQ]
    _ ≤ (Q.totalDegree : ℝ≥0∞) / Fintype.card F := prCoin_mvEval_eq_zero_le _ hQ
    _ ≤ 1 / Fintype.card F := by gcongr; exact_mod_cast hdeg


/-! ## The honest ring-switching message -/

/-- The point beyond the packed seven address bits: `x[1 … m−7]`. -/
def Model.Claim.tail {m : ℕ} (c : Claim F m) (hm : 7 ≤ m) : Fin (m - 7) → F :=
  fun j => c.x ⟨j.val + 1, by omega⟩

theorem two_pow_eq_mul {m : ℕ} (hm : 7 ≤ m) : 2 ^ m = 2 ^ (m - 7) * 128 := by
  rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add, Nat.sub_add_cancel hm]

/-- The address `128·y + v` of packed bit `v` of packed element `y`. -/
def addr128 {m : ℕ} (hm : 7 ≤ m) (y : Fin (2 ^ (m - 7))) (v : Fin 128) : Fin (2 ^ m) :=
  ⟨128 * y.val + v.val, by
    rw [two_pow_eq_mul hm]; have := y.isLt; have := v.isLt; omega⟩

/-- The honest ring-switching message for a claim on a witness (§12.2):
`ŝ[v] = Σ_y eq(x', y)·z[128y + v]`. -/
def shat {m : ℕ} (hm : 7 ≤ m) (c : Claim F m) (z : Witness m) : Fin 128 → F :=
  fun v => ∑ y : Fin (2 ^ (m - 7)), eqAt (c.tail hm) y.val * ((z (addr128 hm y v)).val : F)

omit [Fintype F] [DecidableEq F] in
/-- Splitting off the lowest coordinate of an equality weight. -/
theorem eqAt_split {m : ℕ} (hm : 7 ≤ m) (x : Fin (m - 6) → F) (y b : ℕ) (hb : b < 2) :
    eqAt x (2 * y + b) = (if b = 1 then x ⟨0, by omega⟩ else 1 + x ⟨0, by omega⟩) *
      eqAt (fun j : Fin (m - 7) => x ⟨j.val + 1, by omega⟩) y := by
  have h : m - 7 + 1 = m - 6 := by omega
  unfold eqAt
  rw [← (finCongr h).prod_comp, Fin.prod_univ_succ]
  have hc0 : (finCongr h) 0 = ⟨0, by omega⟩ := Fin.ext rfl
  have hcs : ∀ j : Fin (m - 7), (finCongr h) j.succ = ⟨j.val + 1, by omega⟩ := fun j => Fin.ext rfl
  simp only [hc0, hcs]
  congr 1
  · have hb' : (2 * y + b).testBit 0 = decide (b = 1) := by
      rw [Nat.testBit_zero]; congr 1; apply propext; omega
    rw [hb']
    by_cases hb1 : b = 1 <;> simp [hb1]
  · refine Finset.prod_congr rfl fun j _ => ?_
    have htb : (2 * y + b).testBit (j.val + 1) = y.testBit j.val := by
      rw [Nat.testBit_succ]; congr 1; omega
    rw [htb]

/-- **The claim's value from the honest message** (§12.2): `Σ_v w[v]·ŝ[v]`. -/
theorem claim_eval_eq {m : ℕ} (hm : 7 ≤ m) (c : Claim F m) (z : Witness m) :
    c.eval z = ∑ v : Fin 128, packedWeight c hm v * shat hm c z v := by
  classical
  let e : Fin (2 ^ (m - 7)) × Fin 128 ≃ Fin (2 ^ m) :=
    finProdFinEquiv.trans (finCongr (two_pow_eq_mul hm).symm)
  have he : ∀ p, e p = addr128 hm p.1 p.2 := fun p => by
    ext; simp [e, addr128, finProdFinEquiv]; ring
  unfold Claim.eval
  rw [← e.sum_comp, Fintype.sum_prod_type, Finset.sum_comm]
  refine Finset.sum_congr rfl fun v _ => ?_
  unfold shat
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun y _ => ?_
  simp only [he]
  have hmod : (addr128 hm y v).val % 64 = v.val % 64 := by
    simp only [addr128]; omega
  have hdiv : (addr128 hm y v).val / 64 = 2 * y.val + v.val / 64 := by
    simp only [addr128]; omega
  have hskip : c.skip ⟨(addr128 hm y v).val % 64, Nat.mod_lt _ (by norm_num)⟩ =
      c.skip ⟨v.val % 64, Nat.mod_lt _ (by norm_num)⟩ := congrArg c.skip (Fin.ext hmod)
  rw [hskip, hdiv, eqAt_split hm c.x y.val (v.val / 64) (by omega)]
  have ht : c.tail hm = fun j : Fin (m - 7) => c.x ⟨j.val + 1, by omega⟩ := rfl
  simp only [packedWeight, ht]
  ring

end FlockSoundness
