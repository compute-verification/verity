import Proofs.Flock.Soundness.Rounds
import Proofs.Flock.Soundness.Defs
import Proofs.Flock.Soundness.Game.Union
import Proofs.Flock.Soundness.Game.Interleave
import Mathlib.LinearAlgebra.Lagrange

/-!
# The lincheck phase (PROTOCOL.md §10)

For one codeword near the committed table, whose witness `z` makes the `c` claim and the region
claims true: the doomed zerocheck says the `a` claim, the `b` claim or the pin fails. Then the
lincheck ends with a true `ab` claim for `z` with probability at most `(2 + 2(k_log − 6) + 63)/|F|`
(`value_lincheck_le`):

* **α, β (L1, L2).** The lincheck starts from `α·v_a + v_b + β`; the truth is
  `α·a(z) + b(z) + β·pin(z)`, which is `Σ_c comb[c]·h[c]` for the witness's block-folded columns
  `h` (`combSum_eq`). The difference is a nonzero polynomial of degree 1 in `(α, β)`: one coin each.
* **The rounds (L3).** Each keeps a false claim false except with probability `2/|F|`
  (`Rounds.lincheckRound_sound`), about the vectors folded so far (`value_lincheckRounds_le`).
* **The final check and `r'` (L4, L5).** A passing check forces `z_partial` off the folded columns;
  the `ab` claim's value `Σ_s L_s(r')·z_partial[s]` then misses the truth `Σ_s L_s(r')·h'[s]`
  (`abEval_eq`) except with probability `63/|F|`, a nonzero polynomial of degree below 64 through
  the skip nodes (`prCoin_lagS_le`).

`lincheck_phase_of_card` takes the union over a list of at most `L` codewords.
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Equality weights and sums over split indices -/

/-- Splitting an equality weight at coordinate `a`: low bits `w < 2^a`, high bits `o`. -/
theorem eqAt_split_at {R : Type} [CommRing R] {N : ℕ} (a b : ℕ) (hN : a + b = N) (r : Fin N → R)
    (w o : ℕ) (hw : w < 2 ^ a) :
    eqAt r (w + 2 ^ a * o) = eqAt (fun j : Fin a => r ⟨j.val, by omega⟩) w *
      eqAt (fun j : Fin b => r ⟨a + j.val, by omega⟩) o := by
  subst hN
  unfold eqAt
  rw [Fin.prod_univ_add]
  congr 1
  · refine Finset.prod_congr rfl fun j _ => ?_
    rw [add_comm w, Nat.testBit_two_pow_mul_add o hw]
    simp only [Fin.val_castAdd, j.isLt, if_true]
    rfl
  · refine Finset.prod_congr rfl fun j _ => ?_
    rw [add_comm w, Nat.testBit_two_pow_mul_add o hw]
    simp only [Fin.val_natAdd, show ¬ a + j.val < a by omega, if_false,
      show a + j.val - a = j.val by omega]
    rfl

/-- Sums over `Fin (m·n)` as double sums, the index being `j + n·i`. -/
theorem sum_fin_mul {M : Type} [AddCommMonoid M] {m n : ℕ} (f : Fin (m * n) → M) :
    ∑ x, f x = ∑ i : Fin m, ∑ j : Fin n, f (finProdFinEquiv (i, j)) := by
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]

/-- Addresses split into blocks: `a = o·2^k_log + i`. -/
theorem sum_addr {M : Type} [AddCommMonoid M] (S : Statement) (f : Fin (2 ^ S.m) → M) :
    ∑ a, f a = ∑ o : Fin (2 ^ S.nbl), ∑ i : Fin (2 ^ S.kLog), f (S.addr o i) := by
  have h : 2 ^ S.nbl * 2 ^ S.kLog = 2 ^ S.m := by rw [← pow_add, Nat.sub_add_cancel S.hk]
  rw [← (finCongr h).sum_comp, sum_fin_mul]
  refine Finset.sum_congr rfl fun o _ => Finset.sum_congr rfl fun i _ => ?_
  congr 1
  ext
  simp only [finCongr_apply, Fin.val_cast, finProdFinEquiv_apply_val, Statement.addr]
  ring

/-- In-block indices split at the skip: `i = s + 64·w`. -/
theorem sum_block {M : Type} [AddCommMonoid M] (k : ℕ) (hk : 6 ≤ k) (g : Fin (2 ^ k) → M) :
    ∑ i, g i = ∑ w : Fin (2 ^ (k - 6)), ∑ s : Fin 64,
      g ⟨s.val + 64 * w.val, by
        have h : 2 ^ k = 2 ^ (k - 6) * 64 := by
          rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel hk]
        have := s.isLt; have := w.isLt; rw [h]; nlinarith⟩ := by
  have h : 2 ^ (k - 6) * 64 = 2 ^ k := by
    rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel hk]
  rw [← (finCongr h).sum_comp, sum_fin_mul]
  refine Finset.sum_congr rfl fun w _ => Finset.sum_congr rfl fun s _ => ?_
  congr 1

/-- In characteristic 2, reading bits into `F` is a ring homomorphism. -/
theorem val_cast_eq [CharP F 2] (x : ZMod 2) :
    ((x.val : ℕ) : F) = ZMod.castHom (dvd_refl 2) F x := by
  rw [ZMod.castHom_apply, ZMod.cast_eq_val]

/-! ## Two coin bounds -/

/-- A nonzero linear form in one coin vanishes with probability at most `1/|F|`. -/
theorem prCoin_linear_le (a b : F) (ha : a ≠ 0) :
    prCoin (fun x : F => a * x + b = 0) ≤ 1 / Fintype.card F := by
  have hP : (C a * Polynomial.X + C b : F[X]) ≠ 0 := by
    intro h
    have := congrArg (fun p => p.coeff 1) h
    simp only [coeff_add, coeff_C_mul_X, coeff_C, if_true, one_ne_zero, if_false, add_zero,
      coeff_zero] at this
    exact ha this
  have hdeg : (C a * Polynomial.X + C b : F[X]).natDegree ≤ 1 := natDegree_linear_le
  calc prCoin (fun x : F => a * x + b = 0)
      = prCoin (fun x : F => (C a * Polynomial.X + C b : F[X]).eval x = 0) := by simp
    _ ≤ ((C a * Polynomial.X + C b : F[X]).natDegree : ℝ≥0∞) / Fintype.card F := prCoin_eval_eq_zero_le _ hP
    _ ≤ 1 / Fintype.card F := by gcongr; exact_mod_cast hdeg

/-- A nonzero vector on the 64 skip nodes stays visible through the Lagrange weights at a live
coin, except with probability `63/|F|`: it is a polynomial of degree below 64 (§9). -/
theorem prCoin_lagS_le (A : Arith F K) (hA : A.Correct) (d : Fin 64 → F) (hd : d ≠ 0) :
    prCoin (fun ζ : F => ∑ s, A.lagS ζ s * d s = 0) ≤ 63 / Fintype.card F := by
  obtain ⟨node, hinj, hinterp⟩ := hA.lagS_interp
  have hvs : Set.InjOn node (Finset.univ : Finset (Fin 64)) := hinj.injOn
  set p := Lagrange.interpolate Finset.univ node d with hp_def
  have hnode : ∀ s, p.eval (node s) = d s := fun s =>
    Lagrange.eval_interpolate_at_node d hvs (Finset.mem_univ s)
  have hp : p ≠ 0 := by
    intro h0
    apply hd
    funext s
    have := hnode s
    rw [h0, eval_zero] at this
    exact this.symm
  have hdeg := Lagrange.degree_interpolate_lt (r := d) hvs
  rw [Finset.card_univ, Fintype.card_fin] at hdeg
  have hnat : p.natDegree < 64 := (natDegree_lt_iff_degree_lt hp).2 hdeg
  have hsum : ∀ ζ, ∑ s, A.lagS ζ s * d s = p.eval ζ := fun ζ => by
    rw [← hinterp p hnat ζ]
    simp only [hnode]
  simp_rw [hsum]
  refine (prCoin_eval_eq_zero_le p hp).trans ?_
  gcongr
  exact_mod_cast (by omega : p.natDegree ≤ 63)

/-! ## The rounds -/

/-- The true round polynomial of `Σ_i u_i·v_i` at half-length `h`:
`T(X) = Σ_{i<h} (u_i + X·(u_{i+h} + u_i))·(v_i + X·(v_{i+h} + v_i))`. -/
noncomputable def roundPoly (h : ℕ) (u v : ℕ → F) : F[X] :=
  ∑ i ∈ Finset.range h,
    (C (u i) + C (u (i + h) + u i) * Polynomial.X) *
      (C (v i) + C (v (i + h) + v i) * Polynomial.X)

omit [Fintype F] [DecidableEq F] in
theorem roundPoly_natDegree_le (h : ℕ) (u v : ℕ → F) : (roundPoly h u v).natDegree ≤ 2 := by
  unfold roundPoly
  refine natDegree_sum_le_of_forall_le _ _ fun i _ => ?_
  have hlin : ∀ a b : F, (C a + C b * Polynomial.X : F[X]).natDegree ≤ 1 := fun a b =>
    (natDegree_add_le _ _).trans (max_le (by simp) ((natDegree_C_mul_le _ _).trans (by simp)))
  have h1 := hlin (u i) (u (i + h) + u i)
  have h2 := hlin (v i) (v (i + h) + v i)
  exact natDegree_mul_le.trans (by omega)

omit [Fintype F] [DecidableEq F] in
theorem roundPoly_eval (h : ℕ) (u v : ℕ → F) (t : F) :
    (roundPoly h u v).eval t = ∑ i ∈ Finset.range h, foldTop t h u i * foldTop t h v i := by
  simp only [roundPoly, eval_finset_sum, eval_mul, eval_add, eval_C, eval_X, foldTop]
  exact Finset.sum_congr rfl fun i _ => by ring

omit [Fintype F] [DecidableEq F] in
theorem roundPoly_sum [CharP F 2] (h : ℕ) (u v : ℕ → F) :
    (roundPoly h u v).eval 0 + (roundPoly h u v).eval 1 =
      ∑ i ∈ Finset.range (2 * h), u i * v i := by
  rw [roundPoly_eval, roundPoly_eval, two_mul, Finset.sum_range_add]
  have hfold : ∀ w : ℕ → F, ∀ i, foldTop 1 h w i = w (h + i) := fun w i => by
    simp only [foldTop, one_mul]
    rw [add_comm h i, add_comm (w (i + h)), ← add_assoc, Rounds.add_self, zero_add]
  congr 1
  · exact Finset.sum_congr rfl fun i _ => by simp [foldTop]
  · exact Finset.sum_congr rfl fun i _ => by rw [hfold, hfold]

theorem lincheckRounds_succ (n : ℕ) (claim : F) :
    lincheckRounds (n + 1) claim = Game.send (F × F) fun q => Game.coin F fun t =>
      (lincheckRounds n (q.2 * t ^ 2 + (claim + q.1 + q.1 + q.2) * t + (claim + q.1))).bind
        fun r => Game.ret (Fin.cons t r.1, r.2) := by
  rw [lincheckRounds]
  rfl

/-- **The lincheck rounds** (L3). From a false claim about `Σ_{i < 2^n·64} u_i·v_i`, the rounds
end with a true claim about the folded vectors' inner product with probability at most
`2n/|F|`, whatever the prover sends. -/
theorem value_lincheckRounds_le [CharP F 2] : ∀ (n : ℕ) (u v : ℕ → F) (claim : F),
    claim ≠ ∑ i ∈ Finset.range (2 ^ n * 64), u i * v i →
    value (fun r : (Fin n → F) × F => r.2 = ∑ s ∈ Finset.range 64,
        foldAll (List.ofFn r.1) (2 ^ n * 64) u s * foldAll (List.ofFn r.1) (2 ^ n * 64) v s)
      (lincheckRounds n claim) ≤ ((2 * n : ℕ) : ℝ≥0∞) / Fintype.card F
  | 0, u, v, claim, hc => by
      show value _ (Game.ret (Fin.elim0, claim)) ≤ _
      rw [value_ret, if_neg]
      · exact zero_le
      · simpa [foldAll] using hc
  | n + 1, u, v, claim, hc => by
      rw [lincheckRounds_succ, value_send]
      refine iSup_le fun q => ?_
      have hH : 2 ^ (n + 1) * 64 / 2 = 2 ^ n * 64 := by rw [pow_succ]; omega
      have h2H : 2 * (2 ^ n * 64) = 2 ^ (n + 1) * 64 := by rw [pow_succ]; ring
      refine (value_coin_le _ F _ (fun t => q.2 * t ^ 2 + (claim + q.1 + q.1 + q.2) * t +
          (claim + q.1) = (roundPoly (2 ^ n * 64) u v).eval t) (2 / Fintype.card F)
          (((2 * n : ℕ) : ℝ≥0∞) / Fintype.card F) ?_ fun t ht => ?_).trans ?_
      · refine Rounds.lincheckRound_sound claim _ (roundPoly_natDegree_le _ u v) ?_ q.1 q.2
        rwa [roundPoly_sum, h2H]
      · rw [value_bind_ret]
        refine (value_mono (fun r hr => ?_) _).trans
          (value_lincheckRounds_le n (foldTop t (2 ^ n * 64) u) (foldTop t (2 ^ n * 64) v) _ ?_)
        · simp only [List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, foldAll, hH] at hr
          exact hr
        · rwa [roundPoly_eval] at ht
      · rw [ENNReal.div_add_div_same]
        gcongr
        push_cast
        ring_nf
        exact le_rfl

/-! ## The witness through the lincheck's eyes -/

/-- The witness's block-folded columns: `h[c] = Σ_o eq(ρ_out, o)·ω[o·2^k_log + c]`. -/
def hcol (S : Statement) (ρOut : Fin S.nbl → F) (ω : Witness S.m) : ℕ → F := fun c =>
  if hc : c < 2 ^ S.kLog then
    ∑ o : Fin (2 ^ S.nbl), eqAt ρOut o.val * ((ω (S.addr o ⟨c, hc⟩)).val : F)
  else 0

theorem two_pow_kLog (S : Statement) : 2 ^ S.kLog = 2 ^ (S.kLog - 6) * 64 := by
  rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel S.hk6]

/-- A claim at a point `p ‖ ρ_out` (inner bits, then the block bits) evaluates through the
block-folded columns. -/
theorem claim_eval_split (S : Statement) (c : Claim F S.m) (p : Fin (S.kLog - 6) → F)
    (ρOut : Fin S.nbl → F)
    (hp : ∀ j : Fin (S.kLog - 6), c.x ⟨j.val, by have := S.hk; omega⟩ = p j)
    (hout : ∀ j : Fin S.nbl,
      c.x ⟨S.kLog - 6 + j.val, by
        have := S.hk; have := S.hk6; have hj : j.val < S.m - S.kLog := j.isLt; omega⟩ = ρOut j)
    (ω : Witness S.m) :
    c.eval ω = ∑ i : Fin (2 ^ S.kLog), c.skip ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ *
      eqAt p (i.val / 64) * hcol S ρOut ω i.val := by
  have hk := S.hk
  have hk6 := S.hk6
  have hnbl : S.nbl = S.m - S.kLog := rfl
  have hpow := two_pow_kLog S
  unfold Claim.eval
  rw [sum_addr S, Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [hcol, dif_pos i.isLt, Finset.mul_sum]
  refine Finset.sum_congr rfl fun o _ => ?_
  have hval : (S.addr o i).val = 64 * (2 ^ (S.kLog - 6) * o.val) + i.val := by
    simp only [Statement.addr, hpow]; ring
  have hmod : (S.addr o i).val % 64 = i.val % 64 := by
    rw [hval, Nat.mul_add_mod]
  have hdiv : (S.addr o i).val / 64 = i.val / 64 + 2 ^ (S.kLog - 6) * o.val := by
    rw [hval, Nat.mul_add_div (by norm_num)]; ring
  have hi : i.val / 64 < 2 ^ (S.kLog - 6) := by
    have := lt_of_lt_of_eq i.isLt hpow; omega
  have hskip : c.skip ⟨(S.addr o i).val % 64, Nat.mod_lt _ (by norm_num)⟩ =
      c.skip ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ := congrArg c.skip (Fin.ext hmod)
  rw [hskip, hdiv, eqAt_split_at (S.kLog - 6) S.nbl (by omega) c.x _ _ hi]
  simp only [hp, hout]
  ring

theorem addr_div (S : Statement) (o : Fin (2 ^ S.nbl)) (i : Fin (2 ^ S.kLog)) :
    (S.addr o i).val / 2 ^ S.kLog = o.val := by
  simp only [Statement.addr]
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (Nat.two_pow_pos _), Nat.div_eq_of_lt i.isLt,
    zero_add]

theorem addr_mod (S : Statement) (o : Fin (2 ^ S.nbl)) (i : Fin (2 ^ S.kLog)) :
    (S.addr o i).val % 2 ^ S.kLog = i.val := by
  simp only [Statement.addr]
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt i.isLt]

/-- The A-side bit at in-block index `i` of block `o` is `(A₀·z_o)_i`. -/
theorem aVec_addr (S : Statement) (z : Witness S.m) (o : Fin (2 ^ S.nbl)) (i : Fin (2 ^ S.kLog)) :
    aVec S z (S.addr o i) = (S.A₀.mulVec (S.block z o)) i := by
  simp only [aVec]
  congr 2
  all_goals first | exact Fin.ext (addr_div S o i) | exact addr_mod S o i

/-- The B-side bit at in-block index `i` of block `o` is `(B₀·z_o)_i`. -/
theorem bVec_addr (S : Statement) (z : Witness S.m) (o : Fin (2 ^ S.nbl)) (i : Fin (2 ^ S.kLog)) :
    bVec S z (S.addr o i) = (S.B₀.mulVec (S.block z o)) i := by
  simp only [bVec]
  congr 2
  all_goals first | exact Fin.ext (addr_div S o i) | exact addr_mod S o i

/-- The A-side columns are `A₀` applied to the columns. -/
theorem hcol_aVec [CharP F 2] (S : Statement) (ρOut : Fin S.nbl → F) (z : Witness S.m)
    (i : Fin (2 ^ S.kLog)) :
    hcol S ρOut (aVec S z) i.val =
      ∑ c : Fin (2 ^ S.kLog), ((S.A₀ i c).val : F) * hcol S ρOut z c.val := by
  simp only [hcol, dif_pos i.isLt, dif_pos (Fin.isLt _)]
  simp only [aVec_addr, Matrix.mulVec, dotProduct, val_cast_eq, map_sum, map_mul]
  simp only [← val_cast_eq, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun o _ => ?_
  simp only [Statement.block]
  ring

/-- The B-side columns are `B₀` applied to the columns. -/
theorem hcol_bVec [CharP F 2] (S : Statement) (ρOut : Fin S.nbl → F) (z : Witness S.m)
    (i : Fin (2 ^ S.kLog)) :
    hcol S ρOut (bVec S z) i.val =
      ∑ c : Fin (2 ^ S.kLog), ((S.B₀ i c).val : F) * hcol S ρOut z c.val := by
  simp only [hcol, dif_pos i.isLt, dif_pos (Fin.isLt _)]
  simp only [bVec_addr, Matrix.mulVec, dotProduct, val_cast_eq, map_sum, map_mul]
  simp only [← val_cast_eq, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun o _ => ?_
  simp only [Statement.block]
  ring

/-- `ρ_in`: the zerocheck's challenges for the inner bits (as the lincheck reads them). -/
def ρIn (S : Statement) (zc : ZcOut F S.m) : Fin (S.kLog - 6) → F :=
  fun j => zc.ρ ⟨j.val, by have := S.hk; omega⟩

/-- The zerocheck's `a`, `b` claims at `(z, ρ)`: the point is `ρ_in ‖ ρ_out`. -/
theorem zcClaim_eval (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) (v : F)
    (ω : Witness S.m) :
    (⟨v, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).eval ω =
      ∑ i : Fin (2 ^ S.kLog), A.lagS zc.z ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ *
        eqAt (ρIn S zc) (i.val / 64) * hcol S (zc.ρOut S) ω i.val := by
  refine claim_eval_split S _ _ _ (fun j => rfl) (fun j => ?_) ω
  simp only [ZcOut.ρOut]
  rw [dif_pos]

/-- **The lincheck's sum** (§10): `Σ_c comb[c]·h[c] = α·a(z) + b(z) + β·pin(z)` for the true
`a`, `b` claim values and the pin's extension at `ρ_out`. -/
theorem combSum_eq [CharP F 2] (A : Arith F K) (S : Statement) (zc : ZcOut F S.m)
    (z : Witness S.m) (α β : F) :
    ∑ c ∈ Finset.range (2 ^ S.kLog),
        comb A S zc.z α β (ρIn S zc) c * hcol S (zc.ρOut S) z c =
      α * (⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).eval (aVec S z)
        + (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).eval (bVec S z)
        + β * pinMLE S z (zc.ρOut S) := by
  rw [zcClaim_eval, zcClaim_eval]
  simp only [hcol_aVec, hcol_bVec]
  rw [← Fin.sum_univ_eq_sum_range]
  have hpin : pinMLE S z (zc.ρOut S) = hcol S (zc.ρOut S) z S.pin.val := by
    simp only [pinMLE, hcol, dif_pos S.pin.isLt, Statement.block]
  rw [hpin]
  set hc : ℕ → F := hcol S (zc.ρOut S) z
  set e : Fin (2 ^ S.kLog) → F := fun i =>
    A.lagS zc.z ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ * eqAt (ρIn S zc) (i.val / 64) with he
  have hite : ∑ c : Fin (2 ^ S.kLog), (if c.val = S.pin.val then β else 0) * hc c.val =
      β * hc S.pin.val := by
    rw [Finset.sum_eq_single S.pin]
    · simp
    · intro c _ hcp
      rw [if_neg (fun h => hcp (Fin.ext h)), zero_mul]
    · intro h; exact absurd (Finset.mem_univ _) h
  have hL : ∑ c : Fin (2 ^ S.kLog), comb A S zc.z α β (ρIn S zc) c.val * hc c.val =
      ∑ i : Fin (2 ^ S.kLog), ∑ c : Fin (2 ^ S.kLog),
        e i * (α * ((S.A₀ i c).val : F) + ((S.B₀ i c).val : F)) * hc c.val + β * hc S.pin.val := by
    rw [← hite, Finset.sum_comm, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun c _ => ?_
    simp only [comb, dif_pos c.isLt, add_mul, Finset.sum_mul, he]
  rw [hL]
  simp only [he, Finset.mul_sum, ← Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun c _ => ?_
  ring

/-- **Folding by the rounds' coins** (top bit first) evaluates the columns' multilinear
extension over the inner bits at the reversed coins. -/
theorem foldAll_eq_sum : ∀ (n : ℕ) (ts : Fin n → F) (v : ℕ → F) (s : ℕ),
    foldAll (List.ofFn ts) (2 ^ n * 64) v s =
      ∑ w : Fin (2 ^ n), eqAt (fun j : Fin n => ts ⟨n - 1 - j.val, by omega⟩) w.val *
        v (s + 64 * w.val)
  | 0, ts, v, s => by simp [foldAll, eqAt]
  | n + 1, ts, v, s => by
      have hH : 2 ^ (n + 1) * 64 / 2 = 2 ^ n * 64 := by rw [pow_succ]; omega
      rw [List.ofFn_succ, foldAll, hH, foldAll_eq_sum n]
      have h2 : 2 * 2 ^ n = 2 ^ (n + 1) := by rw [pow_succ]; ring
      rw [← (finCongr h2).sum_comp, sum_fin_mul, Fin.sum_univ_two, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun w _ => ?_
      have hsplit : ∀ b : Fin 2, eqAt (fun j : Fin (n + 1) => ts ⟨n + 1 - 1 - j.val, by omega⟩)
          ((finCongr h2) (finProdFinEquiv (b, w))).val =
          eqAt (fun j : Fin n => ts ⟨n - 1 - j.val + 1, by omega⟩) w.val *
            (if b.val = 1 then ts 0 else 1 + ts 0) := by
        intro b
        simp only [finCongr_apply, Fin.val_cast, finProdFinEquiv_apply_val]
        rw [eqAt_split_at n 1 rfl _ _ _ w.isLt]
        congr 1
        · congr 1
          funext j
          congr 1
          ext
          simp only
          omega
        · simp only [eqAt, Finset.univ_unique, Fin.default_eq_zero, Fin.val_zero,
            Finset.prod_singleton, Nat.testBit_zero]
          have : ts ⟨n + 0 + 1 - 1 - n, by omega⟩ = ts 0 := congrArg ts (Fin.ext (by simp))
          fin_cases b <;> simp [this]
      rw [hsplit 0, hsplit 1]
      simp only [finCongr_apply, Fin.val_cast, finProdFinEquiv_apply_val, Fin.val_zero,
        Fin.val_one, Fin.succ_mk, foldTop]
      rw [if_neg (by norm_num : (0 : ℕ) ≠ 1), if_pos trivial]
      have hv : s + 64 * w.val + 2 ^ n * 64 = s + 64 * (w.val + 2 ^ n * 1) := by ring
      have hv0 : s + 64 * (w.val + 2 ^ n * 0) = s + 64 * w.val := by ring
      rw [hv, hv0]
      ring

/-- The `ab` claim's point (L5): the rounds' coins reversed on the inner bits, then `ρ_out`. -/
def abPoint (S : Statement) (zc : ZcOut F S.m) (ts : Fin (S.kLog - 6) → F) : Fin (S.m - 6) → F :=
  fun i => if h : i.val < S.kLog - 6 then ts ⟨S.kLog - 6 - 1 - i.val, by omega⟩ else zc.ρ i

/-- **The `ab` claim's true value**: `Σ_s L_s(r')·h'[s]`, `h'` the columns folded by the rounds'
coins. -/
theorem abEval_eq (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) (ts : Fin (S.kLog - 6) → F)
    (w r' : F) (z : Witness S.m) :
    (⟨w, A.lagS r', abPoint S zc ts⟩ : Claim F S.m).eval z =
      ∑ s : Fin 64, A.lagS r' s *
        foldAll (List.ofFn ts) (2 ^ S.kLog) (hcol S (zc.ρOut S) z) s.val := by
  have hk6 := S.hk6
  rw [claim_eval_split S _ (fun j => ts ⟨S.kLog - 6 - 1 - j.val, by omega⟩) (zc.ρOut S)
    (fun j => by simp only [abPoint, dif_pos j.isLt]) (fun j => ?_) z]
  · have hlen : foldAll (List.ofFn ts) (2 ^ S.kLog) =
        foldAll (List.ofFn ts) (2 ^ (S.kLog - 6) * 64) := by rw [two_pow_kLog S]
    rw [hlen, sum_block S.kLog hk6, Finset.sum_comm]
    refine Finset.sum_congr rfl fun sidx _ => ?_
    rw [foldAll_eq_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun widx _ => ?_
    have hmod : (sidx.val + 64 * widx.val) % 64 = sidx.val := by omega
    have hdiv : (sidx.val + 64 * widx.val) / 64 = widx.val := by omega
    have hskip : A.lagS r' ⟨(sidx.val + 64 * widx.val) % 64, Nat.mod_lt _ (by norm_num)⟩ =
        A.lagS r' sidx := congrArg (A.lagS r') (Fin.ext hmod)
    simp only [hskip, hdiv]
    ring
  · simp only [abPoint, ZcOut.ρOut, show ¬ (S.kLog - 6 + j.val < S.kLog - 6) by omega,
      dif_neg, not_false_eq_true]
    rw [dif_pos]

/-- The lincheck's verdict (L4, L5). -/
def lcResult (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) (α β : F)
    (ts : Fin (S.kLog - 6) → F) (claim : F) (zP : Fin 64 → F) (r' : F) : Option (Claim F S.m) :=
  if (∑ s : Fin 64, foldAll (List.ofFn ts) (2 ^ S.kLog) (comb A S zc.z α β (ρIn S zc)) s * zP s)
      = claim
  then some ⟨∑ s : Fin 64, A.lagS r' s * zP s, A.lagS r', abPoint S zc ts⟩ else none

theorem lincheck_eq (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) :
    lincheck A S zc = Game.coin F fun α => Game.coin F fun β =>
      (lincheckRounds (S.kLog - 6) (α * zc.va + zc.vb + β)).bind fun r =>
        Game.send (Fin 64 → F) fun zP => Game.coin F fun r' =>
          Game.ret (lcResult A S zc α β r.1 r.2 zP r') := rfl

/-- **The lincheck, one codeword** (§10). If the `a` claim, the `b` claim or the pin fails for the
witness `z`, the lincheck ends with a true `ab` claim for `z` with probability at most
`(2 + 2(k_log − 6) + 63)/|F|`. -/
theorem value_lincheck_le (A : Arith F K) (hA : A.Correct) (S : Statement) (zc : ZcOut F S.m)
    (z : Witness S.m)
    (hbad : ¬ ((⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (aVec S z)
      ∧ (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (bVec S z)
      ∧ pinMLE S z (zc.ρOut S) = 1)) :
    value (fun o => ∃ ab, o = some ab ∧ ab.Holds z) (lincheck A S zc) ≤
      ((2 + 2 * (S.kLog - 6) + 63 : ℕ) : ℝ≥0∞) / Fintype.card F := by
  classical
  have := hA.charTwo
  set n := S.kLog - 6 with hn
  set a := (⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).eval (aVec S z) with ha_def
  set b := (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).eval (bVec S z) with hb_def
  set pn := pinMLE S z (zc.ρOut S) with hpn_def
  set hz := hcol S (zc.ρOut S) z with hz_def
  have hlen : 2 ^ S.kLog = 2 ^ n * 64 := two_pow_kLog S
  have hzero : prCoin (fun _ : F => False) = 0 := by simp [prCoin, avg]
  rw [lincheck_eq]
  refine (value_coin_le _ F _ (fun α => pn = 1 ∧ α * (zc.va - a) + (zc.vb - b) = 0)
    (1 / Fintype.card F) (1 / Fintype.card F + (((2 * n : ℕ) : ℝ≥0∞) / Fintype.card F +
      63 / Fintype.card F)) ?_ fun α hα => ?_).trans ?_
  · -- α: the `a` or `b` claim is off, unless the pin is
    by_cases hp1 : pn = 1
    · by_cases hva : zc.va - a = 0
      · have hvb : zc.vb - b ≠ 0 := by
          intro hvb
          refine hbad ⟨?_, ?_, hp1⟩
          · show a = zc.va; linear_combination -hva
          · show b = zc.vb; linear_combination -hvb
        calc prCoin (fun α : F => pn = 1 ∧ α * (zc.va - a) + (zc.vb - b) = 0)
            ≤ prCoin (fun _ : F => False) := prCoin_mono fun α h => by
              rw [hva, mul_zero, zero_add] at h; exact hvb h.2
          _ = 0 := hzero
          _ ≤ _ := zero_le
      · exact (prCoin_mono fun α h => by rw [mul_comm] at h; exact h.2).trans
          (prCoin_linear_le _ _ hva)
    · calc prCoin (fun α : F => pn = 1 ∧ α * (zc.va - a) + (zc.vb - b) = 0)
          ≤ prCoin (fun _ : F => False) := prCoin_mono fun α h => hp1 h.1
        _ = 0 := hzero
        _ ≤ _ := zero_le
  · refine value_coin_le _ F _ (fun β => α * (zc.va - a) + (zc.vb - b) + β * (1 - pn) = 0)
      (1 / Fintype.card F) _ ?_ fun β hβ => ?_
    · -- β: the pin is off, or the α-combination already is
      by_cases hp1 : pn = 1
      · calc prCoin (fun β : F => α * (zc.va - a) + (zc.vb - b) + β * (1 - pn) = 0)
            ≤ prCoin (fun _ : F => False) := prCoin_mono fun β h => by
              rw [hp1, sub_self, mul_zero, add_zero] at h; exact hα ⟨hp1, h⟩
          _ = 0 := hzero
          _ ≤ _ := zero_le
      · have hne : (1 : F) - pn ≠ 0 := sub_ne_zero.2 (Ne.symm hp1)
        exact (prCoin_mono fun β h => by
          rw [show α * (zc.va - a) + (zc.vb - b) + β * (1 - pn) =
            (1 - pn) * β + (α * (zc.va - a) + (zc.vb - b)) by ring] at h; exact h).trans
          (prCoin_linear_le _ _ hne)
    · -- the start claim is false; the rounds, then the final check and `r'`
      have hstart : α * zc.va + zc.vb + β ≠
          ∑ i ∈ Finset.range (2 ^ n * 64), comb A S zc.z α β (ρIn S zc) i * hz i := by
        rw [← hlen, combSum_eq]
        intro h
        apply hβ
        linear_combination h
      refine (value_bind_le _ _ _ (fun r : (Fin n → F) × F => r.2 ≠ ∑ s ∈ Finset.range 64,
          foldAll (List.ofFn r.1) (2 ^ n * 64) (comb A S zc.z α β (ρIn S zc)) s *
            foldAll (List.ofFn r.1) (2 ^ n * 64) hz s) (63 / Fintype.card F)
          (fun r hr => ?_)).trans ?_
      · refine value_round_le _ _ _ _ _ fun zP => ?_
        by_cases hchk : (∑ s : Fin 64, foldAll (List.ofFn r.1) (2 ^ S.kLog)
            (comb A S zc.z α β (ρIn S zc)) s * zP s) = r.2
        · -- a passing check puts `z_partial` off the folded columns
          set V : Fin 64 → F := fun s => foldAll (List.ofFn r.1) (2 ^ S.kLog) hz s.val with hV
          have hd : (fun s => zP s - V s) ≠ 0 := by
            intro h0
            apply hr
            have hzP : ∀ s, zP s = V s := fun s => sub_eq_zero.1 (congrFun h0 s)
            rw [← hchk, ← hlen, ← Fin.sum_univ_eq_sum_range
              (fun s => foldAll (List.ofFn r.1) (2 ^ S.kLog) (comb A S zc.z α β (ρIn S zc)) s *
                foldAll (List.ofFn r.1) (2 ^ S.kLog) hz s)]
            exact Finset.sum_congr rfl fun s _ => by rw [hzP s]
          refine (prCoin_mono fun r' h => ?_).trans (prCoin_lagS_le A hA _ hd)
          obtain ⟨ab, hab, hholds⟩ := h
          simp only [lcResult, hchk, if_true, Option.some.injEq] at hab
          subst hab
          unfold Claim.Holds at hholds
          rw [abEval_eq] at hholds
          simp only [mul_sub, Finset.sum_sub_distrib, sub_eq_zero]
          exact hholds.symm
        · calc prCoin (fun r' : F => ∃ ab, lcResult A S zc α β r.1 r.2 zP r' = some ab ∧
                ab.Holds z)
              ≤ prCoin (fun _ : F => False) := prCoin_mono fun r' ⟨ab, hab, _⟩ => by
                simp only [lcResult, hchk, if_false, reduceCtorEq] at hab
            _ = 0 := hzero
            _ ≤ _ := zero_le
      · gcongr
        refine (value_mono (fun r h => not_not.1 h) _).trans ?_
        exact value_lincheckRounds_le n _ _ _ hstart
  · rw [ENNReal.div_add_div_same, ENNReal.div_add_div_same, ENNReal.div_add_div_same]
    gcongr
    push_cast
    ring_nf
    exact le_rfl

/-! ## The phase -/

open Classical in
/-- **The lincheck phase** (§10), for a list `W` of at most `L` messages: from a doomed zerocheck, the
lincheck ends undoomed with probability at most `L·(2 + 2(k_log − 6) + 63)/|F|`. -/
theorem lincheck_phase_of_list (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (W : Set (Message F sch.k0 sch.level₀.logCols)) (extra : List (Claim F S.m)) (zc : ZcOut F S.m)
    (hzc : DZcW A S sch W extra (some zc)) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ W).card ≤ L) :
    value (fun o => ¬ DLcW A S sch W extra zc o) (lincheck A S zc) ≤
      L * (((2 + 2 * (S.kLog - 6) + 63 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
  classical
  set s := Finset.univ.filter fun M => M ∈ W
  let E : Message F sch.k0 sch.level₀.logCols → Option (Claim F S.m) → Prop := fun M o =>
    ∃ ab, o = some ab ∧ ab.Holds (wit A S sch M) ∧ (cClaim A zc).Holds (wit A S sch M)
      ∧ ∀ e ∈ extra, e.Holds (wit A S sch M)
  calc value (fun o => ¬ DLcW A S sch W extra zc o) (lincheck A S zc)
      ≤ value (fun o => ∃ M ∈ s, E M o) (lincheck A S zc) := by
        refine value_mono (fun o h => ?_) _
        cases o with
        | none => exact absurd trivial h
        | some ab =>
          simp only [DLcW, not_forall, not_not] at h
          obtain ⟨M, hM, hab⟩ := h
          exact ⟨M, by simp [s, hM], ab, rfl, hab⟩
    _ ≤ ∑ M ∈ s, value (E M) (lincheck A S zc) := value_exists_finset_le _ _ _
    _ ≤ ∑ _M ∈ s, ((2 + 2 * (S.kLog - 6) + 63 : ℕ) : ℝ≥0∞) / Fintype.card F := by
        refine Finset.sum_le_sum fun M hM => ?_
        have hMc : M ∈ W := by simpa [s] using hM
        by_cases hce : (cClaim A zc).Holds (wit A S sch M) ∧ ∀ e ∈ extra, e.Holds (wit A S sch M)
        · refine (value_mono (fun o ⟨ab, hab, hh, _⟩ => ⟨ab, hab, hh⟩) _).trans
            (value_lincheck_le A hA S zc _ fun ⟨ha, hb, hp⟩ => hzc M hMc ⟨ha, hb, hce.1, hp, hce.2⟩)
        · calc value (E M) (lincheck A S zc) ≤ value (fun _ => False) (lincheck A S zc) :=
                value_mono (fun o ⟨_, _, _, hc, he⟩ => hce ⟨hc, he⟩) _
            _ = 0 := value_false _
            _ ≤ _ := zero_le
    _ = s.card * (((2 + 2 * (S.kLog - 6) + 63 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ L * (((2 + 2 * (S.kLog - 6) + 63 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
        gcongr

open Classical in
/-- **The lincheck phase** for the codewords near the committed table: `lincheck_phase_of_list` at
`closeMsgs`. -/
theorem lincheck_phase_of_card (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (C : Oracle F) (extra : List (Claim F S.m)) (zc : ZcOut F S.m)
    (hzc : DZc A S sch C extra (some zc)) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ closeMsgs A sch C).card ≤ L) :
    value (fun o => ¬ DLc A S sch C extra zc o) (lincheck A S zc) ≤
      L * (((2 + 2 * (S.kLog - 6) + 63 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
  simp only [dLc_eq]
  exact lincheck_phase_of_list A hA S sch _ extra zc (by rw [← dZc_eq]; exact hzc) L hL

end FlockSoundness
