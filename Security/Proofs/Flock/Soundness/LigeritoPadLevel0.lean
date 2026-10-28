import Proofs.Flock.Soundness.PadCode
import Proofs.Flock.Soundness.LigeritoPhase
import Definitions.Flock.LigeritoPad

/-!
# The padded level 0: lane rounds, extra lanes, and decoding

The pieces of `ligeritoPad`'s level 0 that differ from `ligerito`'s:

* **The lane rounds** send no message after the last coin (`foldRoundsNL`). For a post-condition
  that does not read that message, they are the lane rounds (`value_foldRoundsNL_le`).
* **The extra lanes' coins** (`drawKs`). Each coin `ρ_i` is an affine line `w + ρ_i·X_i` for mutual
  correlated agreement (`ExtrasGood`, `value_drawKs_extras_le`), and a coin on the restarted
  claim, an affine function of the coins with a nonzero constant for every list element
  (`value_drawKs_affine_le`).
* **Rep `r`'s list** (`padList`): the row families, witness lanes then the rep's extra lanes, within
  the padded radius of the rep's rows. A family's next-level message is its witness lanes' messages
  folded by the lane challenges plus its extra lanes' weighted by `ρ` (`padFold`).
* **Decoding** (`decode_pad`): if no fold round and no extra coin was an MCA event, and the padded
  column agrees with the encoding of `G` on `(1 − γ')·2^d` positions, then `G` is `padFold` of a
  family in the list. The padding correction `ybar` is then the family's folded padding.
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## The lane rounds without their last message -/

theorem value_drawK_bind_mono (A : Arith F K) {α β : Type} {p : α → Prop} {p' : β → Prop}
    {f : K → Game α} {g : K → Game β} (h : ∀ r, value p (f r) ≤ value p' (g r)) :
    value p ((drawK A).bind f) ≤ value p' ((drawK A).bind g) := by
  show value p (Game.coin (Fin 2 → F) fun w => f (A.ofLimbs (w 0) (w 1))) ≤
    value p' (Game.coin (Fin 2 → F) fun w => g (A.ofLimbs (w 0) (w 1)))
  rw [value_coin, value_coin]
  exact Finset.sum_le_sum fun w _ => mul_le_mul_right (h _) _

/-- **The lane rounds without their last message** accept no more often than with it. -/
theorem value_foldRoundsNL_le (A : Arith F K) :
    ∀ (k : ℕ) (p : List K × K → Prop) (s : K) (q : Quad K), 1 ≤ k →
      value p (foldRoundsNL A k s q) ≤
        value (fun out : List K × K × Quad K => p (out.1, out.2.1)) (foldRounds A k s q)
  | 0, _, _, _, h => absurd h (by norm_num)
  | 1, p, s, q, _ => by
      rw [foldRounds_succ]
      refine value_drawK_bind_mono A fun r => ?_
      rw [value_send]
      refine le_iSup_of_le ((0 : K), (0 : K)) ?_
      show value p (Game.ret ([r], q.eval r)) ≤ value _ (Game.ret (r :: [], q.eval r, _))
      classical
      simp only [value_ret]
      exact le_rfl
  | k + 2, p, s, q, _ => by
      rw [foldRounds_succ]
      refine value_drawK_bind_mono A fun r => ?_
      show value p (Game.send (K × K) fun msg =>
          (foldRoundsNL A (k + 1) (q.eval r) (fromMsg msg (q.eval r))).bind fun out =>
            Game.ret (r :: out.1, out.2)) ≤ _
      rw [value_send, value_send]
      refine iSup_mono fun msg => ?_
      rw [value_bind_ret, value_bind_ret]
      exact value_foldRoundsNL_le A (k + 1) (fun out => p (r :: out.1, out.2)) _ _ (by omega)

/-! ## The extra lanes' coins -/

theorem drawKs_succ (A : Arith F K) (e : ℕ) :
    drawKs A (e + 1) = (drawK A).bind fun r => (drawKs A e).bind fun rs => Game.ret (Fin.cons r rs) :=
  rfl

omit [DecidableEq K] in
/-- An affine equation with a nonzero constant has at most one root. -/
theorem prCoin_affine_le (a c : K) (ha : a ≠ 0) :
    prCoin (fun r : K => a + r * c = 0) ≤ 1 / Fintype.card K := by
  classical
  by_cases hc : c = 0
  · refine (prCoin_mono (q := fun _ => False) fun r h => ha (by simpa [hc] using h)).trans ?_
    simp [prCoin, avg]
  · have hP : Polynomial.C a + Polynomial.C c * Polynomial.X ≠ 0 := by
      intro h
      have := congrArg (Polynomial.coeff · 1) h
      simp at this
      exact hc this
    have hdeg : (Polynomial.C a + Polynomial.C c * Polynomial.X).natDegree ≤ 1 := by
      refine (Polynomial.natDegree_add_le _ _).trans (max_le (by simp) ?_)
      exact (Polynomial.natDegree_C_mul_le _ _).trans (by simp)
    refine (prCoin_mono fun r h => ?_).trans ((prCoin_eval_eq_zero_le _ hP).trans ?_)
    · simp only [Polynomial.eval_add, Polynomial.eval_C, Polynomial.eval_mul, Polynomial.eval_X]
      rw [mul_comm]; exact h
    · gcongr
      exact_mod_cast hdeg

/-- **The restarted claim.** If `a_M ≠ 0` for every list element, the `e` coins make some
`a_M + Σ_i ρ_i·c_{M,i}` vanish with probability at most `e·|S|/|K|`. -/
theorem value_drawKs_affine_le (A : Arith F K) (hA : A.Correct) {ι : Type} (S : Finset ι) :
    ∀ (e : ℕ) (a : ι → K) (c : ι → Fin e → K), (∀ M ∈ S, a M ≠ 0) →
      value (fun ρ : Fin e → K => ∃ M ∈ S, a M + ∑ i, ρ i * c M i = 0) (drawKs A e) ≤
        ((e * S.card : ℕ) : ℝ≥0∞) / Fintype.card K
  | 0, a, c, ha => by
      show value _ (Game.ret Fin.elim0) ≤ _
      classical
      rw [value_ret, if_neg]
      · exact zero_le
      · rintro ⟨M, hM, h⟩
        exact ha M hM (by simpa using h)
  | e + 1, a, c, ha => by
      classical
      rw [drawKs_succ]
      refine (value_drawK_le A hA _ _ (fun r => ∃ M ∈ S, a M + r * c M 0 = 0)
        ((S.card : ℝ≥0∞) / Fintype.card K) (((e * S.card : ℕ) : ℝ≥0∞) / Fintype.card K) ?_
        fun r hr => ?_).trans ?_
      · calc prCoin (fun r => ∃ M ∈ S, a M + r * c M 0 = 0)
            ≤ ∑ M ∈ S, prCoin (fun r => a M + r * c M 0 = 0) := prCoin_exists_le _ _
          _ ≤ ∑ _M ∈ S, (1 : ℝ≥0∞) / Fintype.card K :=
              Finset.sum_le_sum fun M hM => prCoin_affine_le _ _ (ha M hM)
          _ = (S.card : ℝ≥0∞) / Fintype.card K := by
              rw [Finset.sum_const, nsmul_eq_mul, mul_one_div]
      · rw [value_bind_ret]
        push Not at hr
        refine (value_mono (fun rs h => ?_) _).trans (value_drawKs_affine_le A hA S e
          (fun M => a M + r * c M 0) (fun M i => c M i.succ) fun M hM => hr M hM)
        obtain ⟨M, hM, h⟩ := h
        refine ⟨M, hM, ?_⟩
        rw [Fin.sum_univ_succ] at h
        simp only [Fin.cons_zero, Fin.cons_succ] at h
        linear_combination h
      · rw [ENNReal.div_add_div_same]
        gcongr
        push_cast
        ring_nf
        exact le_rfl

/-- No extra lane's coin is an MCA event: the lanes are added one at a time, `w ↦ w + ρ_i·X_i`. -/
def ExtrasGood {ι : Type} [Fintype ι] (code : Submodule K (ι → K)) (δ : ℝ) :
    (e : ℕ) → (Fin e → K) → (Fin e → ι → K) → (ι → K) → Prop
  | 0, _, _, _ => True
  | e + 1, ρ, Xs, w => ¬ IsMCA (AffineLineGenerator K) code (ρ 0) ![w, Xs 0] δ ∧
      ExtrasGood code δ e (fun i => ρ i.succ) (fun i => Xs i.succ) (fun q => w q + ρ 0 * Xs 0 q)

omit [DecidableEq K] in
/-- **The extra lanes, deterministically.** If no extra coin is an MCA event and `w + Σ_i ρ_i·X_i`
agrees with a codeword on a set `T` of at least `(1 − δ)n` positions, so do `w` and every `X_i`. -/
theorem agree_of_extras {ι : Type} [Fintype ι] (code : Submodule K (ι → K)) (δ : ℝ) :
    ∀ (e : ℕ) (ρ : Fin e → K) (Xs : Fin e → ι → K) (w : ι → K), ExtrasGood code δ e ρ Xs w →
      ∀ T : Finset ι, (T.card : ℝ) ≥ Fintype.card ι * (1 - δ) →
        (∃ c ∈ code, ∀ q ∈ T, c q = w q + ∑ i, ρ i * Xs i q) →
          (∃ c ∈ code, ∀ q ∈ T, c q = w q) ∧ ∀ i, ∃ c ∈ code, ∀ q ∈ T, c q = Xs i q
  | 0, _, _, w, _, T, _, ⟨c, hc, hcT⟩ =>
      ⟨⟨c, hc, fun q hq => by rw [hcT q hq]; simp⟩, fun i => i.elim0⟩
  | e + 1, ρ, Xs, w, hx, T, hT, ⟨c, hc, hcT⟩ => by
      obtain ⟨hmca, hgood⟩ := hx
      obtain ⟨⟨c', hc', hc'T⟩, hrest⟩ := agree_of_extras code δ e _ _ _ hgood T hT
        ⟨c, hc, fun q hq => by rw [hcT q hq, Fin.sum_univ_succ]; ring⟩
      obtain ⟨p, hp⟩ := (not_isMCA_iff_forall_exists_codewords (AffineLineGenerator K) code (ρ 0)
        ![w, Xs 0] δ).1 hmca T hT (mem_projected_of_agree code T _ c' hc' fun q hq => by
          rw [hc'T q hq]
          simp [Fin.sum_univ_two])
      refine ⟨⟨(p 0).val, (p 0).property, fun q hq => by rw [hp 0 q hq]; rfl⟩, fun i => ?_⟩
      refine Fin.cases ⟨(p 1).val, (p 1).property, fun q hq => by rw [hp 1 q hq]; rfl⟩
        (fun j => hrest j) i

/-- **The extra lanes' MCA events.** Over the `e` coins, on fixed rows, some coin is an MCA event
with probability at most `e·ε`. -/
theorem value_drawKs_extras_le (A : Arith F K) (hA : A.Correct) {ι : Type} [Fintype ι]
    (code : Submodule K (ι → K)) (δ : ℝ) (ε : ℝ≥0∞)
    (hε : ∀ U : Fin 2 → ι → K, prCoin (fun γ : K => IsMCA (AffineLineGenerator K) code γ U δ) ≤ ε) :
    ∀ (e : ℕ) (Xs : Fin e → ι → K) (w : ι → K),
      value (fun ρ => ¬ ExtrasGood code δ e ρ Xs w) (drawKs A e) ≤ (e : ℝ≥0∞) * ε
  | 0, Xs, w => by
      show value _ (Game.ret Fin.elim0) ≤ _
      simp [ExtrasGood]
  | e + 1, Xs, w => by
      rw [drawKs_succ]
      refine (value_drawK_le A hA _ _ (fun r => IsMCA (AffineLineGenerator K) code r ![w, Xs 0] δ)
        ε ((e : ℝ≥0∞) * ε) (hε _) fun r hr => ?_).trans ?_
      · rw [value_bind_ret]
        refine (value_mono (fun rs h => ?_) _).trans (value_drawKs_extras_le A hA code δ ε hε e
          (fun i => Xs i.succ) (fun q => w q + r * Xs 0 q))
        simp only [ExtrasGood, Fin.cons_zero, Fin.cons_succ, not_and] at h
        exact h hr
      · rw [Nat.cast_succ, add_mul, one_mul, add_comm]

/-! ## Rep `r`'s list -/

/-- Rep `r`'s lanes: the witness lanes, then its extra lanes. -/
def padLane (k0 e r : ℕ) : Fin (2 ^ k0 + e) → ℕ :=
  Fin.addCases (fun ℓ : Fin (2 ^ k0) => ℓ.val) (extraLane k0 e r)

/-- Rep `r`'s rows of the committed table. -/
def padRows (k0 e r d : ℕ) (C : Oracle F) : Fin (2 ^ k0 + e) → Fin (2 ^ d) → F :=
  fun ℓ q => C (padLane k0 e r ℓ) q.val

open Classical in
/-- **Rep `r`'s list**: the row families within the padded radius of the rep's rows. -/
noncomputable def padList (A : Arith F K) (l : Level) (t k0 e r : ℕ) (C : Oracle F) :
    Finset (Fin (2 ^ k0 + e) → Fin (2 ^ l.logCols + t) → F) :=
  Finset.univ.filter fun R => CloseRows A l.logLen (2 ^ l.logCols + t) (l.padRadius t)
    (padRows k0 e r l.logLen C) R

/-- A padded lane's message: its first `2^cols` coefficients less `c₀` times its padding. -/
def padMsgAt {cols t : ℕ} (c₀ : F) (a : Fin (2 ^ cols + t) → F) (x : Fin (2 ^ cols)) : F :=
  a (Fin.castAdd t x) - if h : x.val < t then c₀ * a ⟨2 ^ cols + x.val, by omega⟩ else 0

/-- A row family's witness lanes' messages. -/
def padWit {k0 e cols t : ℕ} (c₀ : F) (R : Fin (2 ^ k0 + e) → Fin (2 ^ cols + t) → F) :
    Message F k0 cols :=
  fun ℓ => padMsgAt c₀ (R (Fin.castAdd e ℓ))

/-- **A row family's next-level message**: its witness lanes' messages folded by `ch`, plus its
extra lanes' weighted by `ρ`. -/
def padFold (A : Arith F K) {k0 e cols t : ℕ} (c₀ : F) (ch : List K) (ρ : Fin e → K)
    (R : Fin (2 ^ k0 + e) → Fin (2 ^ cols + t) → F) (x : ℕ) : K :=
  foldMsg A ch (padWit c₀ R) x + ∑ i, ρ i *
    (if h : x < 2 ^ cols then A.embed (padMsgAt c₀ (R (Fin.natAdd (2 ^ k0) i)) ⟨x, h⟩) else 0)

/-- The padded code over `K`. -/
noncomputable def codePad (A : Arith F K) (hA : A.Correct) (l : Level) (t : ℕ)
    (hd : l.logLen ≤ 64) : Submodule K (Fin (2 ^ l.logLen) → K) :=
  ReedSolomon.code (domainK A hA l.logLen hd) (2 ^ l.logCols + t)

/-- Rep `r`'s extra rows over `K`. -/
def xrowsK (A : Arith F K) (k0 e r d : ℕ) (C : Oracle F) : Fin e → Fin (2 ^ d) → K :=
  fun i q => A.embed (C (extraLane k0 e r i) q.val)

/-! ## Decoding the padded column -/

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem Xhat_lift (A : Arith F K) (c cols : ℕ) (hc : c < 2 ^ cols) :
    ∀ j ω, A.Xhat c (cols + j) ω = A.Xhat c cols ω
  | 0, _ => rfl
  | j + 1, ω => by
      rw [← add_assoc, Xhat_succ_low A (cols + j) c
        (lt_of_lt_of_le hc (Nat.pow_le_pow_right (by norm_num) (by omega))), Xhat_lift A c cols hc j ω]

/-- A `K` combination of the first `k` monomials, with `ℕ`-indexed coefficients. -/
def encKN (A : Arith F K) (d k : ℕ) (b : ℕ → K) (q : ℕ) : K :=
  ∑ c ∈ Finset.range k, b c * A.embed (A.Xhat c d (A.omega q))

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem encKN_eq_encKAt (A : Arith F K) (d k : ℕ) (b : ℕ → K) (q : ℕ) :
    encKN A d k b q = encKAt A d k (fun c => b c.val) q := by
  unfold encKN encKAt
  rw [← Fin.sum_univ_eq_sum_range (fun c => b c * A.embed (A.Xhat c d (A.omega q))) k]

/-- **A `K` combination of `k` monomials vanishing on `k` positions is zero**, `ℕ`-indexed. -/
theorem encKN_eq_zero (A : Arith F K) (hA : A.Correct) (d k : ℕ) (hd : d ≤ 40) (hd64 : d ≤ 64)
    (hk : k ≤ 2 ^ d) (b : ℕ → K) (T : Finset (Fin (2 ^ d))) (hT : k ≤ T.card)
    (h : ∀ q ∈ T, encKN A d k b q.val = 0) : ∀ c < k, b c = 0 := by
  intro c hc
  have := encKAt_eq_zero A hA d k hd hd64 hk (fun c => b c.val) T hT fun q hq => by
    rw [← encKN_eq_encKAt]; exact h q hq
  exact congrFun this ⟨c, hc⟩

/-- The next level's encoding on the padded code's monomials. -/
theorem encK_eq_encKN (A : Arith F K) (l : Level) (t : ℕ) (hcols : l.logCols ≤ l.logLen)
    (G : ℕ → K) (q : ℕ) :
    encK A l.logCols G q = encKN A l.logLen (2 ^ l.logCols + t)
      (fun c => if c < 2 ^ l.logCols then G c else 0) q := by
  unfold encK encKN
  rw [Finset.sum_range_add]
  have h0 : ∀ j, ¬ (2 ^ l.logCols + j < 2 ^ l.logCols) := fun j => by omega
  simp only [h0, if_false, zero_mul, Finset.sum_const_zero, add_zero]
  refine Finset.sum_congr rfl fun x hx => ?_
  have hx' := Finset.mem_range.1 hx
  have := Xhat_lift A x l.logCols hx' (l.logLen - l.logCols) (A.omega q)
  rw [show l.logCols + (l.logLen - l.logCols) = l.logLen by omega] at this
  rw [if_pos hx', this]

/-- `ybar`, `ℕ`-indexed. -/
def ybarN {t : ℕ} (ybar : Fin t → K) (j : ℕ) : K := if h : j < t then ybar ⟨j, h⟩ else 0

/-- The padding correction's coefficients on the padded code's monomials. -/
def padCorrN (A : Arith F K) (cols t : ℕ) (c₀ : F) (ybar : Fin t → K) (c : ℕ) : K :=
  (if c < t then ybarN ybar c * A.embed c₀ else 0) +
    (if 2 ^ cols ≤ c then ybarN ybar (c - 2 ^ cols) else 0)

/-- The padding correction on the padded code's monomials. -/
theorem padCorr_eq_encKN (A : Arith F K) (l : Level) (t : ℕ) (ht : t ≤ 2 ^ l.logCols) (c₀ : F)
    (ybar : Fin t → K) (q : ℕ) :
    padCorr A l.logLen l.logCols t c₀ ybar q =
      encKN A l.logLen (2 ^ l.logCols + t) (padCorrN A l.logCols t c₀ ybar) q := by
  set E : ℕ → K := fun c => A.embed (A.Xhat c l.logLen (A.omega q)) with hE
  have h1 : ∑ x ∈ Finset.range (2 ^ l.logCols), padCorrN A l.logCols t c₀ ybar x * E x =
      ∑ j : Fin t, ybar j * A.embed c₀ * E j.val := by
    rw [Finset.sum_congr rfl fun x hx => by
      rw [padCorrN, if_neg (show ¬ 2 ^ l.logCols ≤ x by have := Finset.mem_range.1 hx; omega),
        add_zero]]
    rw [← Finset.sum_subset (Finset.range_mono ht) fun x _ hx => by
      rw [if_neg (by simpa using hx), zero_mul]]
    rw [← Fin.sum_univ_eq_sum_range (fun x => (if x < t then ybarN ybar x * A.embed c₀ else 0) *
      E x) t]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp [ybarN, j.isLt]
  have h2 : ∑ j ∈ Finset.range t, padCorrN A l.logCols t c₀ ybar (2 ^ l.logCols + j) *
      E (2 ^ l.logCols + j) = ∑ j : Fin t, ybar j * E (2 ^ l.logCols + j.val) := by
    rw [← Fin.sum_univ_eq_sum_range (fun j => padCorrN A l.logCols t c₀ ybar (2 ^ l.logCols + j) *
      E (2 ^ l.logCols + j)) t]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hj : ¬ (2 ^ l.logCols + j.val < t) := by omega
    simp [padCorrN, hj, ybarN, j.isLt]
  calc padCorr A l.logLen l.logCols t c₀ ybar q
      = ∑ j : Fin t, (ybar j * A.embed c₀ * E j.val + ybar j * E (2 ^ l.logCols + j.val)) := by
        unfold padCorr
        refine Finset.sum_congr rfl fun j _ => ?_
        simp only [hE, map_add, map_mul]
        ring
    _ = ∑ x ∈ Finset.range (2 ^ l.logCols), padCorrN A l.logCols t c₀ ybar x * E x +
        ∑ j ∈ Finset.range t, padCorrN A l.logCols t c₀ ybar (2 ^ l.logCols + j) *
          E (2 ^ l.logCols + j) := by
        rw [h1, h2, Finset.sum_add_distrib]
    _ = encKN A l.logLen (2 ^ l.logCols + t) (padCorrN A l.logCols t c₀ ybar) q := by
        unfold encKN
        rw [Finset.sum_range_add]

end FlockSoundness
