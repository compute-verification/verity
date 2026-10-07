import Proofs.Flock.Soundness.Defs
import Proofs.Flock.Soundness.Game.Poly
import Proofs.Flock.Soundness.Game.Union

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


/-! ## Packing is a bijection of additive groups -/

section Packing

variable (A : Arith F K) (hA : A.Correct)
include hA

theorem unpack_zero : A.unpack 0 = 0 := by
  have h := hA.unpack_add 0 0
  rw [add_zero] at h
  exact (add_eq_left.mp h.symm)

theorem pack_add (w w' : Fin 128 → ZMod 2) : A.pack (w + w') = A.pack w + A.pack w' := by
  have h := hA.unpack_add (A.pack w) (A.pack w')
  rw [hA.unpack_pack, hA.unpack_pack] at h
  rw [← h, hA.pack_unpack]

theorem pack_zero : A.pack 0 = 0 := by
  rw [← unpack_zero A hA, hA.pack_unpack]

/-- `unpack` as an additive map. -/
def unpackHom : F →+ (Fin 128 → ZMod 2) := AddMonoidHom.mk' A.unpack hA.unpack_add

/-- `pack` as an additive map. -/
def packHom : (Fin 128 → ZMod 2) →+ F := AddMonoidHom.mk' A.pack (pack_add A hA)

theorem unpackHom_apply (a : F) : unpackHom A hA a = A.unpack a := rfl

theorem packHom_apply (w : Fin 128 → ZMod 2) : packHom A hA w = A.pack w := rfl

/-- Multiplying by a bit: `unpack (a·b) = b • unpack a`. -/
theorem unpack_mul_bit (a : F) (b : ZMod 2) : A.unpack (a * (b.val : F)) = b • A.unpack a := by
  fin_cases b
  · show A.unpack (a * ((0 : ZMod 2).val : F)) = (0 : ZMod 2) • A.unpack a
    rw [show (0 : ZMod 2).val = 0 from rfl, Nat.cast_zero, mul_zero, zero_smul, unpack_zero A hA]
  · show A.unpack (a * ((1 : ZMod 2).val : F)) = (1 : ZMod 2) • A.unpack a
    rw [show (1 : ZMod 2).val = 1 from rfl, Nat.cast_one, mul_one, one_smul]

theorem pack_smul_bit (b : ZMod 2) (w : Fin 128 → ZMod 2) :
    A.pack (b • w) = (b.val : F) * A.pack w := by
  fin_cases b
  · show A.pack ((0 : ZMod 2) • w) = ((0 : ZMod 2).val : F) * A.pack w
    rw [show (0 : ZMod 2).val = 0 from rfl, Nat.cast_zero, zero_mul, zero_smul, pack_zero A hA]
  · show A.pack ((1 : ZMod 2) • w) = ((1 : ZMod 2).val : F) * A.pack w
    rw [show (1 : ZMod 2).val = 1 from rfl, Nat.cast_one, one_mul, one_smul]

end Packing

/-! ## Ring switching -/

/-- **The ring-switching identity** (§12.3–§12.4, paper Appendix B). If the witness's packed
element `y` is `f y` (bit `v` of `f y` is `z[128y + v]`), the verifier's recombination of the
honest message is the inner product of `f` with `Φ_{r''}(eq(x', ·))`. -/
theorem recombine_shat (A : Arith F K) (hA : A.Correct) {m : ℕ} (hm : 7 ≤ m) (c : Claim F m)
    (z : Witness m) (f : Fin (2 ^ (m - 7)) → F)
    (hz : ∀ y v, z (addr128 hm y v) = A.unpack (f y) v) (r : Fin 7 → F) :
    recombine A (shat hm c z) r =
      ∑ y : Fin (2 ^ (m - 7)), f y * phi A r (eqAt (c.tail hm) y.val) := by
  classical
  -- the transposed bits of the honest message
  have hbits : ∀ u : Fin 128, (fun v => A.unpack (shat hm c z v) u) =
      ∑ y : Fin (2 ^ (m - 7)), A.unpack (eqAt (c.tail hm) y.val) u • A.unpack (f y) := by
    intro u
    funext v
    simp only [shat]
    rw [← unpackHom_apply A hA, map_sum]
    simp only [unpackHom_apply, unpack_mul_bit A hA, hz, Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    exact Finset.sum_congr rfl fun y _ => mul_comm _ _
  have hpack : ∀ u : Fin 128, A.pack (fun v => A.unpack (shat hm c z v) u) =
      ∑ y : Fin (2 ^ (m - 7)), ((A.unpack (eqAt (c.tail hm) y.val) u).val : F) * f y := by
    intro u
    rw [hbits u, ← packHom_apply A hA, map_sum]
    simp only [packHom_apply, pack_smul_bit A hA, hA.pack_unpack]
  unfold recombine phi
  simp only [hpack, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun y _ => Finset.sum_congr rfl fun u _ => by ring

/-- **One claim.** A message other than the honest one survives the seven coins `r''` (the
recombinations agree) with probability at most `7/|F|`. -/
theorem prCoin_recombine_le (A : Arith F K) (hA : A.Correct) (s s' : Fin 128 → F) (hs : s ≠ s') :
    prCoin (fun r : Fin 7 → F => recombine A s r = recombine A s' r) ≤ 7 / Fintype.card F := by
  classical
  have := hA.charTwo
  let d : Fin 128 → F := fun u =>
    A.pack (fun v => A.unpack (s v) u) - A.pack (fun v => A.unpack (s' v) u)
  -- some transposed bit differs
  obtain ⟨v, hv⟩ : ∃ v, s v ≠ s' v := by
    by_contra h; push Not at h; exact hs (funext h)
  have hunp : A.unpack (s v) ≠ A.unpack (s' v) := fun h => hv (by
    rw [← hA.pack_unpack (s v), h, hA.pack_unpack])
  obtain ⟨u₀, hu₀⟩ : ∃ u, A.unpack (s v) u ≠ A.unpack (s' v) u := by
    by_contra h; push Not at h; exact hunp (funext h)
  have hd : d u₀ ≠ 0 := by
    simp only [d, sub_ne_zero]
    intro h
    have := congrArg A.unpack h
    rw [hA.unpack_pack, hA.unpack_pack] at this
    exact hu₀ (congrFun this v)
  have hdiff : ∀ r : Fin 7 → F,
      recombine A s r - recombine A s' r = ∑ u : Fin 128, d u * eqAt r u.val := fun r => by
    simp only [recombine, d, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun u _ => by ring
  calc prCoin (fun r : Fin 7 → F => recombine A s r = recombine A s' r)
      = prCoin (fun r : Fin 7 → F => ∑ u : Fin 128, d u * eqAt r u.val = 0) := by
        congr 1; funext r
        rw [← hdiff r, sub_eq_zero]
    _ ≤ ((7 : ℕ) : ℝ≥0∞) / Fintype.card F := prCoin_eqSum_le (n := 7) d u₀ hd
    _ = 7 / Fintype.card F := by norm_num


/-! ## The opening as a loop over claims, then a batch -/

/-- One claim's ring switch (O3, O5): the prover's message, then seven coins. -/
def opStep {m : ℕ} (c : Claim F m) : Game (Claim F m × (Fin 128 → F) × (Fin 7 → F)) := do
  let s ← recv (Fin 128 → F)
  let r'' ← draw (Fin 7 → F)
  pure (c, s, r'')

/-- The opening's verdict from the switched claims and the batching coins (O6). -/
def opResult (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (claims : List (Claim F m))
    (switched : List (Claim F m × (Fin 128 → F) × (Fin 7 → F))) (γ : Fin claims.length → F) :
    Option (OpenOut F) :=
  let ok := switched.all fun (c, s, _) => decide ((∑ v, packedWeight c hm v * s v) = c.value)
  let T := ∑ k : Fin claims.length, γ k *
    recombine A (switched.getD k.val (claims.getD k.val ⟨0, 0, 0⟩, 0, 0)).2.1
      (switched.getD k.val (claims.getD k.val ⟨0, 0, 0⟩, 0, 0)).2.2
  let b : ℕ → F := fun y => ∑ k : Fin claims.length, γ k *
    let e := switched.getD k.val (claims.getD k.val ⟨0, 0, 0⟩, 0, 0)
    phi A e.2.2 (eqAt (fun j : Fin (m - 7) => e.1.x ⟨j.val + 1, by omega⟩) y)
  if ok then some ⟨T, b⟩ else none

theorem opening_eq (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (claims : List (Claim F m)) :
    opening A hm claims = (claims.mapM opStep).bind fun sw =>
      (draw (Fin claims.length → F)).bind fun γ => .ret (opResult A hm claims sw γ) := rfl

/-! ## One codeword -/

/-- The witness a message packs, at address `128y + v`, is bit `v` of the packed element `y`. -/
theorem wit_addr128 (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule) (h7 : 7 ≤ S.m)
    (M : Message F sch.k0 sch.level₀.logCols) (y : Fin (2 ^ (S.m - 7))) (v : Fin 128) :
    wit A S sch M (addr128 h7 y v) = A.unpack (packedAt sch M y.val) v := by
  have hdiv : (addr128 h7 y v).val / 128 = y.val := by simp only [addr128]; omega
  have hmod : (addr128 h7 y v).val % 128 = v.val := by simp only [addr128]; omega
  simp only [wit, witnessOf, packedAt, hdiv, hmod]
  split_ifs with h
  · rfl
  · rw [unpack_zero A hA]; rfl

/-- The event that the opening ends outside the doomed set for the codeword `M`. -/
def EM (S : Statement) (sch : Schedule) (M : Message F sch.k0 sch.level₀.logCols)
    (o : Option (OpenOut F)) : Prop :=
  ∃ op, o = some op ∧ (∑ y ∈ Finset.range (2 ^ (S.m - 7)), packedAt sch M y * op.b y) = op.T

/-- **The opening, one codeword.** If some claim is false for `M`'s witness, the opening ends
with `M` undoomed with probability at most `(7·n + 1)/|F|`. -/
theorem value_EM_le (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule) (h7 : 7 ≤ S.m)
    (claims : List (Claim F S.m)) (M : Message F sch.k0 sch.level₀.logCols)
    (hM : ∃ c ∈ claims, ¬ c.Holds (wit A S sch M)) :
    value (EM S sch M) (opening A h7 claims) ≤
      ((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
  classical
  set z := wit A S sch M with hz_def
  have hz : ∀ y v, z (addr128 h7 y v) = A.unpack (packedAt sch M y.val) v :=
    fun y v => wit_addr128 A hA S sch h7 M y v
  -- a switched claim is bad when its message is dishonest yet its recombination is right
  let B : Claim F S.m × (Fin 128 → F) × (Fin 7 → F) → Prop := fun x =>
    x.2.1 ≠ shat h7 x.1 z ∧ recombine A x.2.1 x.2.2 = recombine A (shat h7 x.1 z) x.2.2
  let D : List (Claim F S.m × (Fin 128 → F) × (Fin 7 → F)) → Prop := fun sw =>
    sw.map Prod.fst = claims ∧ ¬ ∃ x ∈ sw, B x
  rw [opening_eq]
  refine (value_bind_le _ _ _ D (1 / Fintype.card F) fun sw hD => ?_).trans ?_
  · -- the batch: one nonzero linear form in the coins `γ`
    show value (EM S sch M) (Game.coin _ fun γ => Game.ret (opResult A h7 claims sw γ)) ≤ _
    rw [value_coin_ret]
    let e : Fin claims.length → Claim F S.m × (Fin 128 → F) × (Fin 7 → F) := fun k =>
      sw.getD k.val (claims.getD k.val ⟨0, 0, 0⟩, 0, 0)
    let dd : Fin claims.length → F := fun k =>
      recombine A (e k).2.1 (e k).2.2 - recombine A (shat h7 (e k).1 z) (e k).2.2
    by_cases hok : (sw.all fun (c, s, _) =>
        decide ((∑ v, packedWeight c h7 v * s v) = c.value)) = true
    · -- the verdict is accepting only if the batched difference vanishes
      have hsum : ∀ γ, EM S sch M (opResult A h7 claims sw γ) → ∑ k, γ k * dd k = 0 := by
        intro γ ⟨op, hop, hT⟩
        simp only [opResult, hok, if_true, Option.some.injEq] at hop
        subst hop
        simp only at hT
        have hrs : ∀ k : Fin claims.length,
            ∑ y ∈ Finset.range (2 ^ (S.m - 7)),
              packedAt sch M y * phi A (e k).2.2 (eqAt ((e k).1.tail h7) y) =
            recombine A (shat h7 (e k).1 z) (e k).2.2 := fun k => by
          rw [recombine_shat A hA h7 (e k).1 z (fun y => packedAt sch M y.val) hz,
            ← Fin.sum_univ_eq_sum_range (fun y => packedAt sch M y * phi A (e k).2.2
              (eqAt ((e k).1.tail h7) y))]
        have hL : ∑ y ∈ Finset.range (2 ^ (S.m - 7)), packedAt sch M y *
            (∑ k : Fin claims.length, γ k * phi A (e k).2.2 (eqAt ((e k).1.tail h7) y)) =
            ∑ k : Fin claims.length, γ k * recombine A (shat h7 (e k).1 z) (e k).2.2 := by
          simp only [Finset.mul_sum, ← hrs, Finset.mul_sum]
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun y _ => by ring
        have hT' := hT
        erw [hL] at hT'
        simp only [dd, mul_sub, Finset.sum_sub_distrib, sub_eq_zero]
        exact hT'.symm
      -- and the doomed claim forces a nonzero difference
      obtain ⟨c, hc, hfalse⟩ := hM
      obtain ⟨hmap, hnoB⟩ := hD
      have hlen : sw.length = claims.length := by rw [← hmap, List.length_map]
      obtain ⟨i, hi, hci⟩ := List.mem_iff_getElem.1 hc
      have hi' : i < sw.length := hlen ▸ hi
      have he : e ⟨i, hi⟩ = sw[i] := by
        simp only [e]; exact List.getD_eq_getElem _ _ hi'
      have hfst : (sw[i]).1 = c := by
        have h1 := congrArg (fun l : List (Claim F S.m) => l[i]?) hmap
        simp only [List.getElem?_map, List.getElem?_eq_getElem hi', List.getElem?_eq_getElem hi,
          Option.map_some, Option.some.injEq] at h1
        rw [← hci]; exact h1
      have hcheck := List.all_eq_true.1 hok (sw[i]) (List.getElem_mem hi')
      simp only [decide_eq_true_eq] at hcheck
      have hne : dd ⟨i, hi⟩ ≠ 0 := by
        intro h0
        simp only [dd, he, sub_eq_zero] at h0
        refine hnoB ⟨sw[i], List.getElem_mem hi', ?_, h0⟩
        intro hs
        apply hfalse
        unfold Claim.Holds
        rw [claim_eval_eq h7 c z, ← hfst]
        rw [hs] at hcheck
        exact hcheck
      calc prCoin (fun γ => EM S sch M (opResult A h7 claims sw γ))
          ≤ prCoin (fun γ : Fin claims.length → F => ∑ k, γ k * dd k = 0) :=
            prCoin_mono fun γ h => hsum γ h
        _ ≤ 1 / Fintype.card F := prCoin_batch_le dd ⟨i, hi⟩ hne
    · -- a failed check rejects
      calc prCoin (fun γ => EM S sch M (opResult A h7 claims sw γ))
          ≤ prCoin (fun _ : Fin claims.length → F => False) := prCoin_mono fun γ ⟨op, hop, _⟩ => by
            simp only [opResult, hok, if_false, Bool.false_eq_true, reduceCtorEq] at hop
        _ = 0 := by simp [prCoin, avg]
        _ ≤ _ := zero_le
  · -- the loop over claims
    have hstep : ∀ c : Claim F S.m, value B (opStep c) ≤ 7 / Fintype.card F := fun c => by
      show value B (Game.send _ fun s => Game.coin _ fun r => Game.ret (c, s, r)) ≤ _
      refine value_round_le _ _ _ _ _ fun s => ?_
      by_cases hs : s = shat h7 c z
      · calc prCoin (fun r => B (c, s, r)) ≤ prCoin (fun _ : Fin 7 → F => False) :=
              prCoin_mono fun r h => h.1 hs
          _ = 0 := by simp [prCoin, avg]
          _ ≤ _ := zero_le
      · exact (prCoin_mono fun r h => h.2).trans (prCoin_recombine_le A hA s _ hs)
    have hinv : ∀ c : Claim F S.m, value (fun x => x.1 ≠ c) (opStep c) = 0 := fun c => by
      show value _ (Game.send _ fun s => Game.coin _ fun r => Game.ret (c, s, r)) = 0
      simp [value_send, value_coin, value_ret]
    calc value (fun sw => ¬ D sw) (claims.mapM opStep) + 1 / Fintype.card F
        ≤ (value (fun sw : List _ => sw.map Prod.fst ≠ claims) (claims.mapM opStep)
            + value (fun sw => ∃ x ∈ sw, B x) (claims.mapM opStep)) + 1 / Fintype.card F := by
          refine add_le_add ((value_mono (fun sw h => ?_) _).trans (value_or_le _ _ _)) le_rfl
          by_cases hm : sw.map Prod.fst = claims
          · exact Or.inr (by simpa [D, hm] using h)
          · exact Or.inl hm
      _ ≤ (0 + claims.length * (7 / Fintype.card F)) + 1 / Fintype.card F := by
          refine add_le_add (add_le_add (value_mapM_map_ne opStep Prod.fst hinv claims).le
            (value_mapM_exists_le opStep B _ hstep claims)) le_rfl
      _ = ((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
          simp only [zero_add, div_eq_mul_inv]
          push_cast
          ring

/-! ## The phase -/

open Classical in
/-- **The opening phase** (§12), for a list `W` of at most `L` messages: from a doomed start (every
message of `W` falsifies some claim), the opening ends undoomed with probability at most
`L · (7·n + 1)/|F|`. -/
theorem opening_phase_of_list (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (h7 : 7 ≤ S.m) (W : Set (Message F sch.k0 sch.level₀.logCols)) (claims : List (Claim F S.m))
    (hdoom : ∀ M ∈ W, ∃ c ∈ claims, ¬ c.Holds (wit A S sch M)) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ W).card ≤ L) :
    value (fun o => ¬ DOpW S sch W o) (opening A h7 claims) ≤
      L * (((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
  classical
  set s := Finset.univ.filter fun M => M ∈ W
  calc value (fun o => ¬ DOpW S sch W o) (opening A h7 claims)
      ≤ value (fun o => ∃ M ∈ s, EM S sch M o) (opening A h7 claims) := by
        refine value_mono (fun o h => ?_) _
        cases o with
        | none => exact absurd trivial h
        | some op =>
          simp only [DOpW, not_forall, not_not] at h
          obtain ⟨M, hM, hsum⟩ := h
          exact ⟨M, by simp [s, hM], op, rfl, hsum⟩
    _ ≤ ∑ M ∈ s, value (EM S sch M) (opening A h7 claims) := value_exists_finset_le _ _ _
    _ ≤ ∑ _M ∈ s, ((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F :=
        Finset.sum_le_sum fun M hM => value_EM_le A hA S sch h7 claims M
          (hdoom M (by simpa [s] using hM))
    _ = s.card * (((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ L * (((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
        gcongr

open Classical in
/-- **The opening phase** for the codewords near the committed table: `opening_phase_of_list` at
`closeMsgs`. -/
theorem opening_phase_of_card (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (h7 : 7 ≤ S.m) (C : Oracle F) (claims : List (Claim F S.m))
    (hdoom : ∀ M ∈ closeMsgs A sch C, ∃ c ∈ claims, ¬ c.Holds (wit A S sch M)) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ closeMsgs A sch C).card ≤ L) :
    value (fun o => ¬ DOp A S sch C o) (opening A h7 claims) ≤
      L * (((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
  simp only [dOp_eq]
  exact opening_phase_of_list A hA S sch h7 _ claims hdoom L hL

end FlockSoundness
