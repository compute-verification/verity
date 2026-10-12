import Proofs.Flock.Soundness.LigeritoPadDecode

/-!
# The padded level 0's queries, and on to level 1

`phaseA` for the padded level 0 (`phaseA0Pad`): from `PostPad0`, the level-1 commitment, its two
out-of-domain samples, and level 0's queries on the padded column leave `PreFold` on level 1,
except with probability `midAPad`. The out-of-domain samples are `phaseA`'s (`value_twoOod_good`).
The queries are `phaseA`'s on a column function `v` (`value_queryAndBatchV_le`,
`exists_zero_of_not_preFoldV`). A claimed fold that satisfies the claim is far from the padded
column, since otherwise it decodes to a family of the list (`decode_pad`), whose claim is false
(`query_bound_pad`). Its mismatch survives the queries with probability at most `(√ρ' + η)^Q`.
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **After the padded level 0's folds and extra lanes** (level 0's `PostFold`): the running claim
is false for every family of rep `r`'s list, against the next stage's weight, and no fold round and
no extra coin was an MCA event. -/
structure PostPad0 (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F) (l : Level)
    (hd64 : l.logLen ≤ 64) (t : ℕ) (c₀ : F) (k0 e r : ℕ) (C : Oracle F) (ch : List K)
    (ρ : Fin e → K) (st : LigState F K) : Prop where
  sum : st.q.eval 0 + st.q.eval 1 = st.s
  len : ch.length = k0
  ok : PostOK ch st.levels st.oods st.cons l.logCols
  mca : MCAGood (codePad A hA l t hd64) (l.padRadius t) k0 ch (rowsK A k0 l.logLen C)
  xgood : ExtrasGood (codePad A hA l t hd64) (l.padRadius t) e ρ (xrowsK A k0 e r l.logLen C)
    (foldAllRows k0 ch (rowsK A k0 l.logLen C))
  doom : ∀ M ∈ padList A l t k0 e r C, st.s ≠ ∑ x ∈ Finset.range (2 ^ l.logCols),
    padFold A c₀ ch ρ M x * resAt A n b ch st.levels st.oods st.cons l.logCols x

/-! ## The queries on a column function -/

theorem queryAndBatchV_eq (A : Arith F K) (lv : Level) (v : ℕ → K) (start : ℕ)
    (st : LigState F K) :
    queryAndBatchV A lv v start st =
      Game.coin (Fin lv.queries → F) fun w => Game.coin (Fin (ceilLog2 lv.queries) → F) fun α =>
        Game.send (K × K) fun msg => Game.coin F fun β =>
          Game.ret { st with
            s := st.s + A.embed β * consValueV A (List.ofFn α) v (positions A lv.queries lv.logLen w)
            q := st.q.addScaled (fromMsg msg
              (consValueV A (List.ofFn α) v (positions A lv.queries lv.logLen w))) (A.embed β)
            cons := st.cons ++
              [⟨lv.logCols, positions A lv.queries lv.logLen w, List.ofFn α, β, start⟩] } := rfl

/-- The consistency claim's error against a claimed fold `G`: `Σ_j eq(α)_j·(v[p_j] − Enc(G)[p_j])`. -/
def consGapV (A : Arith F K) (cols : ℕ) (v : ℕ → K) (G : ℕ → K) (α : List F) (pos : List ℕ) : K :=
  ((pos.zipIdx).map fun pj => A.embed (eqList α pj.2) * (v pj.1 - encK A cols G pj.1)).sum

open Classical in
/-- The positions where `v` differs from the encoding of `G`. -/
noncomputable def mismatchV (A : Arith F K) (cols d : ℕ) (v : ℕ → K) (G : ℕ → K) : Finset ℕ :=
  (Finset.range (2 ^ d)).filter fun p => v p ≠ encK A cols G p

open Classical in
/-- **The queries and the consistency coins, on a column function.** As `value_queryAndBatch_le`. -/
theorem value_queryAndBatchV_le (A : Arith F K) (hA : A.Correct) (l : Level) (hd : l.logLen ≤ 64)
    (hQ : l.queries ≤ 2 ^ 64) (v : ℕ → K) (start : ℕ) (st : LigState F K)
    {ι : Type} (S : Finset ι) (D : ι → K) (G : ι → ℕ → K) (qe : ℝ)
    (huniq : ∀ M ∈ S, D M = 0 → ∀ M' ∈ S, D M' = 0 → M' = M)
    (hfar : ∀ M ∈ S, D M = 0 →
      (1 - ((mismatchV A l.logCols l.logLen v (G M)).card : ℝ) / 2 ^ l.logLen) ^ l.queries ≤ qe)
    (P : LigState F K → Prop)
    (hP : ∀ (w : Fin l.queries → F) (α : Fin (ceilLog2 l.queries) → F) (β : F) (st' : LigState F K),
      st'.s = st.s + A.embed β * consValueV A (List.ofFn α) v (positions A l.queries l.logLen w) →
      st'.cons = st.cons ++ [⟨l.logCols, positions A l.queries l.logLen w, List.ofFn α, β, start⟩] →
      st'.levels = st.levels → st'.oods = st.oods → st'.prev = st.prev →
      st'.q.eval 0 + st'.q.eval 1 = st.q.eval 0 + st.q.eval 1 +
        A.embed β * consValueV A (List.ofFn α) v (positions A l.queries l.logLen w) →
      P st' → ∃ M ∈ S, D M + A.embed β *
        consGapV A l.logCols v (G M) (List.ofFn α) (positions A l.queries l.logLen w) = 0) :
    value P (queryAndBatchV A l v start st) ≤
      ENNReal.ofReal qe +
        ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F + S.card / Fintype.card F) := by
  haveI : CharP K 2 := hA.charTwoK
  set Q := l.queries
  set pos := fun w : Fin Q → F => positions A Q l.logLen w
  set gap := fun (M : ι) (w : Fin Q → F) (α : Fin (ceilLog2 Q) → F) =>
    consGapV A l.logCols v (G M) (List.ofFn α) (pos w)
  let Bq : (Fin Q → F) → Prop := fun w =>
    ∃ M ∈ S, D M = 0 ∧ ∀ p ∈ pos w, p ∉ mismatchV A l.logCols l.logLen v (G M)
  let Bα : (Fin Q → F) → (Fin (ceilLog2 Q) → F) → Prop := fun w α =>
    ∃ M ∈ S, D M = 0 ∧ gap M w α = 0
  let Bβ : (Fin Q → F) → (Fin (ceilLog2 Q) → F) → F → Prop := fun w α β =>
    ∃ M ∈ S, D M + A.embed β * gap M w α = 0 ∧ ¬ (D M = 0 ∧ gap M w α = 0)
  have hBq : prCoin Bq ≤ ENNReal.ofReal qe := by
    by_cases hex : ∃ M ∈ S, D M = 0
    · obtain ⟨M, hM, hD⟩ := hex
      refine (prCoin_mono (q := fun w => ∀ p ∈ positions A Q l.logLen w,
          p ∉ mismatchV A l.logCols l.logLen v (G M)) fun w ⟨M', hM', hD', h⟩ => by
        rw [← huniq M hM hD M' hM' hD']; exact h).trans ?_
      refine (prCoin_positions_miss_le A hA Q l.logLen hd _ (Finset.filter_subset _ _)).trans ?_
      exact ENNReal.ofReal_le_ofReal (hfar M hM hD)
    · refine (prCoin_mono (q := fun _ => False) fun w ⟨M, hM, hD, _⟩ => hex ⟨M, hM, hD⟩).trans ?_
      simp [prCoin, avg]
  have hBα : ∀ w, ¬ Bq w → prCoin (Bα w) ≤ (ceilLog2 Q : ℝ≥0∞) / Fintype.card F := by
    intro w hw
    by_cases hex : ∃ M ∈ S, D M = 0
    · obtain ⟨M, hM, hD⟩ := hex
      obtain ⟨p, hp, hpm⟩ : ∃ p ∈ pos w, p ∈ mismatchV A l.logCols l.logLen v (G M) := by
        by_contra hno
        push Not at hno
        exact hw ⟨M, hM, hD, hno⟩
      obtain ⟨j₀, hj₀, hpj⟩ := List.getElem_of_mem hp
      have hlen : (pos w).length = Q := positions_length A Q l.logLen w
      set vv : ℕ → K := fun j => v ((pos w).getD j 0) - encK A l.logCols (G M) ((pos w).getD j 0)
      have hgap : ∀ α, gap M w α = ∑ j ∈ Finset.range Q, A.embed (eqList (List.ofFn α) j) * vv j := by
        intro α
        simp only [gap, consGapV]
        rw [sum_map_zipIdx (fun pj => A.embed (eqList (List.ofFn α) pj.2) *
          (v pj.1 - encK A l.logCols (G M) pj.1)) (pos w) 0, hlen]
        simp only [zero_add, vv]
      have hv : vv j₀ ≠ 0 := by
        simp only [vv, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj₀, Option.getD_some,
          hpj]
        simp only [mismatchV, Finset.mem_filter] at hpm
        exact sub_ne_zero.2 hpm.2
      refine (prCoin_mono (q := fun α => ∑ j ∈ Finset.range Q,
          A.embed (eqList (List.ofFn α) j) * vv j = 0) fun α ⟨M', hM', hD', h⟩ => by
        rw [← hgap, huniq M hM hD M' hM' hD'] at *; exact h).trans ?_
      exact prCoin_embed_eqList_le A hA (le_two_pow_ceilLog2 Q hQ) vv j₀ (by omega) hv
    · refine (prCoin_mono (q := fun _ => False) fun α ⟨M, hM, hD, _⟩ => hex ⟨M, hM, hD⟩).trans ?_
      simp [prCoin, avg]
  have hBβ : ∀ w α, prCoin (Bβ w α) ≤ S.card / Fintype.card F := fun w α =>
    prCoin_embed_affine_union_le A S D fun M => gap M w α
  have hleaf : ∀ w α β st', ¬ Bα w α → ¬ Bβ w α β →
      st'.s = st.s + A.embed β * consValueV A (List.ofFn α) v (pos w) →
      st'.cons = st.cons ++ [⟨l.logCols, pos w, List.ofFn α, β, start⟩] →
      st'.levels = st.levels → st'.oods = st.oods → st'.prev = st.prev →
      st'.q.eval 0 + st'.q.eval 1 = st.q.eval 0 + st.q.eval 1 +
        A.embed β * consValueV A (List.ofFn α) v (pos w) → ¬ P st' := by
    intro w α β st' hα hβ h1 h2 h3 h4 h5 h6 hPst
    obtain ⟨M, hM, h0⟩ := hP w α β st' h1 h2 h3 h4 h5 h6 hPst
    have : D M = 0 ∧ gap M w α = 0 := by
      by_contra hn; exact hβ ⟨M, hM, h0, hn⟩
    exact hα ⟨M, hM, this.1, this.2⟩
  rw [queryAndBatchV_eq]
  refine (value_coin_le P _ _ Bq _ _ hBq fun w hw => ?_)
  refine (value_coin_le P _ _ (Bα w) _ _ (hBα w hw) fun α hα => ?_)
  rw [value_send]
  refine iSup_le fun msg => ?_
  refine (value_coin_le P _ _ (Bβ w α) _ 0 (hBβ w α) fun β hβ => ?_).trans (by rw [add_zero])
  rw [value_ret]
  split_ifs with hPst
  · exact absurd hPst (hleaf w α β _ hα hβ (by rfl) (by rfl) (by rfl) (by rfl) (by rfl)
      (Quad.addScaled_sum _ _ _ _))
  · exact le_rfl

theorem card_agree_add_mismatchV (A : Arith F K) (cols d : ℕ) (v : ℕ → K) (G : ℕ → K) :
    (Finset.univ.filter fun q : Fin (2 ^ d) => v q.val = encK A cols G q.val).card +
      (mismatchV A cols d v G).card = 2 ^ d := by
  classical
  rw [card_filter_fin (2 ^ d) fun p => v p = encK A cols G p]
  have h := Finset.card_filter_add_card_filter_not (s := Finset.range (2 ^ d))
    (fun p => v p = encK A cols G p)
  rw [Finset.card_range] at h
  convert h using 3
  unfold mismatchV
  exact Finset.filter_congr_decidable _ _ _

/-- **The far fold, padded.** A claimed fold `G` whose value is the claim is far from the padded
column: its mismatch survives the queries with probability at most `(√ρ' + η)^Q`. -/
theorem query_bound_pad (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F) (l : Level) (t : ℕ)
    (hd : l.logLen ≤ 40) (hd64 : l.logLen ≤ 64) (hcols : l.logCols ≤ l.logLen)
    (ht : t ≤ 2 ^ l.logCols) (hk : 2 ^ l.logCols + t ≤ 2 ^ l.logLen) (c₀ : F) (k0 e r : ℕ)
    (C : Oracle F) (ch : List K) (ρ : Fin e → K) (ybar : Fin t → K) (st : LigState F K)
    (hpost : PostPad0 A hA n b l hd64 t c₀ k0 e r C ch ρ st) (G : ℕ → K)
    (hs : st.s = ∑ x ∈ Finset.range (2 ^ l.logCols),
      G x * resAt A n b ch st.levels st.oods st.cons l.logCols x) :
    (1 - ((mismatchV A l.logCols l.logLen
        (padColumn A l.logLen l.logCols t c₀ ch ρ (extraLane k0 e r) ybar C) G).card : ℝ) /
      2 ^ l.logLen) ^ l.queries ≤ l.padQueryError t := by
  classical
  set v := padColumn A l.logLen l.logCols t c₀ ch ρ (extraLane k0 e r) ybar C
  have hmis : (mismatchV A l.logCols l.logLen v G).card ≤ 2 ^ l.logLen := by
    have := card_agree_add_mismatchV A l.logCols l.logLen v G
    omega
  have h2 : (0 : ℝ) < 2 ^ l.logLen := by positivity
  have hle : 1 - ((mismatchV A l.logCols l.logLen v G).card : ℝ) / 2 ^ l.logLen ≤
      Real.sqrt (l.padRate t) + Accounting.eta := by
    by_contra hgt
    push Not at hgt
    set T := Finset.univ.filter fun q : Fin (2 ^ l.logLen) => v q.val = encK A l.logCols G q.val
    have hT := card_agree_add_mismatchV A l.logCols l.logLen v G
    have hTr : (T.card : ℝ) = 2 ^ l.logLen - (mismatchV A l.logCols l.logLen v G).card := by
      rw [eq_sub_iff_add_eq]; exact_mod_cast hT
    have hTc : (T.card : ℝ) ≥ Fintype.card (Fin (2 ^ l.logLen)) * (1 - l.padRadius t) := by
      rw [Fintype.card_fin]
      have hrad : 1 - l.padRadius t = Real.sqrt (l.padRate t) + Accounting.eta := by
        unfold Level.padRadius; ring
      have hm := mul_lt_mul_of_pos_left hgt h2
      rw [mul_sub, mul_one, mul_div_cancel₀ _ h2.ne'] at hm
      push_cast
      rw [hrad, hTr]
      linarith
    obtain ⟨M₀, hM₀, hG⟩ := decode_pad A hA l t hd hd64 hcols ht hk c₀ k0 e r C ch hpost.len ρ
      ybar hpost.mca hpost.xgood G T hTc fun q hq => (Finset.mem_filter.1 hq).2
    refine hpost.doom M₀ hM₀ ?_
    rw [hs]
    exact Finset.sum_congr rfl fun x hx => by rw [hG x (Finset.mem_range.1 hx)]
  have hnn : 0 ≤ 1 - ((mismatchV A l.logCols l.logLen v G).card : ℝ) / 2 ^ l.logLen := by
    rw [sub_nonneg, div_le_one h2]; exact_mod_cast hmis
  exact pow_le_pow_left₀ hnn hle _

/-- **A failed `PreFold` is a zero batched difference**, on a column function: as
`exists_zero_of_not_preFold`. -/
theorem exists_zero_of_not_preFoldV (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F)
    (initial : List K) (l l' : Level) (v : ℕ → K) (st : LigState F K)
    (hok : PostOK initial st.levels st.oods st.cons l.logCols)
    (hdim : l.logCols + 1 = l'.lanes + l'.logCols) (C' : Oracle F)
    (s₂ : K) (q₂ : Quad K) (oods₂ : List (OodCtx F)) (hsum₂ : q₂.eval 0 + q₂.eval 1 = s₂)
    (hshape : ∀ o ∈ oods₂, o.split = some st.levels.length ∧ o.z.length = l'.logCols + l'.lanes)
    (pos : List ℕ) (α : List F) (β : F) (st' : LigState F K)
    (h1 : st'.s = s₂ + A.embed β * consValueV A α v pos)
    (h2 : st'.cons = st.cons ++ [⟨l.logCols, pos, α, β, st.levels.length⟩])
    (h3 : st'.levels = st.levels) (h4 : st'.oods = st.oods ++ oods₂)
    (h6 : st'.q.eval 0 + st'.q.eval 1 = q₂.eval 0 + q₂.eval 1 + A.embed β * consValueV A α v pos)
    (hno : ¬ PreFold A n b initial l' C' l.logCols st') :
    ∃ M ∈ closeL A l'.radius l'.lanes l'.logCols l'.logLen C',
      oodDiff A (l'.logCols + l'.lanes) (flat M)
          (∑ x ∈ Finset.range (2 ^ l.logCols), unsplit A (flat M) x *
            resAt A n b initial st.levels st.oods st.cons l.logCols x) (s₂, q₂, oods₂) +
        A.embed β * consGapV A l.logCols v (unsplit A (flat M)) α pos = 0 := by
  have hdim' : l'.logCols + l'.lanes = l.logCols + 1 := by omega
  have hq' : st'.q.eval 0 + st'.q.eval 1 = st'.s := by rw [h6, h1, hsum₂]
  have hok' : PreOK initial st'.levels st'.oods st'.cons l.logCols := by
    rw [h2, h3, h4]
    exact preOK_of_postOK initial st.levels st.oods st.cons l.logCols hok oods₂
      (fun o ho => ⟨(hshape o ho).1, by rw [(hshape o ho).2, hdim']⟩) _ rfl
  obtain ⟨M, hM, hM'⟩ : ∃ M ∈ closeL A l'.radius l'.lanes l'.logCols l'.logLen C',
      st'.s = ∑ y ∈ Finset.range (2 ^ (l.logCols + 1)),
        A.embed (flat M y) * preW A n b initial st'.levels st'.oods st'.cons l.logCols y := by
    by_contra hall
    push Not at hall
    exact hno ⟨hq', hok', hdim, hall⟩
  refine ⟨M, hM, ?_⟩
  rw [h2, h3, h4, sum_preW_new A n b initial st.levels st.oods oods₂ st.cons l.logCols _
    (fun o ho => (hok.ood o ho).1) (fun o ho => (hshape o ho).1) rfl, h1] at hM'
  have hgap : consGapV A l.logCols v (unsplit A (flat M)) α pos =
      consValueV A α v pos - ((pos.zipIdx).map fun pj =>
        A.embed (eqList α pj.2) * encK A l.logCols (unsplit A (flat M)) pj.1).sum := by
    unfold consGapV consValueV
    rw [← list_sum_map_sub]
    exact congrArg List.sum (List.map_congr_left fun pj _ => by ring)
  rw [hgap, oodDiff, hdim']
  dsimp only
  linear_combination hM'

/-! ## From the padded level 0 to level 1 -/

/-- The next commitment, its OOD samples and level 0's queries on the column function `v`, then the
continuation (as `midRound`). -/
def midRoundV (A : Arith F K) (lv : Level) (dim : ℕ) (v : ℕ → K) (st : LigState F K)
    (k : Oracle F → LigState F K → Game Bool) : Game Bool := do
  let C' ← recv (Oracle F)
  let (s, q, oods) ← twoOod A dim st.levels.length st.s st.q
  let st := { st with s := s, q := q, oods := st.oods ++ oods }
  let st ← queryAndBatchV A lv v st.levels.length st
  k C' st

/-- Everything of `padMidLevelError` after level 0's folds and extra lanes. -/
noncomputable def midAPad (l l' : Level) (t : ℕ) : ℝ :=
  pairs l'.listBound * (((l'.logCols + l'.lanes : ℕ) : ℝ) / qF) ^ 2 + 2 * l'.listBound / qF
    + l.padQueryError t + l'.listBound * ceilLog2 l.queries / qF + l'.listBound / qF

theorem padMidLevelError_eq (l l' : Level) (t e : ℕ) :
    padMidLevelError l l' t e = l.padFoldError t e (l.padListBound t) + midAPad l l' t := by
  unfold padMidLevelError midAPad; ring

theorem padQueryError_nonneg (l : Level) (t : ℕ) : 0 ≤ l.padQueryError t := by
  unfold Level.padQueryError eta
  positivity

theorem midAPad_nonneg (l l' : Level) (t : ℕ) : 0 ≤ midAPad l l' t := by
  have hq : (0 : ℝ) < qF := by unfold qF; positivity
  have h1 : 0 ≤ pairs l'.listBound * (((l'.logCols + l'.lanes : ℕ) : ℝ) / qF) ^ 2 :=
    mul_nonneg (pairs_nonneg _) (sq_nonneg _)
  have h2 := padQueryError_nonneg l t
  unfold midAPad
  have : (0 : ℝ) ≤ 2 * l'.listBound / qF := by positivity
  have : (0 : ℝ) ≤ l'.listBound * ceilLog2 l.queries / qF := by positivity
  have : (0 : ℝ) ≤ l'.listBound / qF := by positivity
  linarith

theorem midAPad_bound (l l' : Level) (t L : ℕ) (hL : L ≤ l'.listBound)
    (hF : Fintype.card F = 2 ^ 128) :
    (L.choose 2 : ℝ≥0∞) * ((((l'.logCols + l'.lanes : ℕ) : ℕ) : ℝ≥0∞) / Fintype.card F) ^ 2 +
        2 * ((L : ℝ≥0∞) / Fintype.card F) +
      (ENNReal.ofReal (l.padQueryError t) + ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F +
        (L : ℝ≥0∞) / Fintype.card F)) ≤ ENNReal.ofReal (midAPad l l' t) := by
  have hmid := midA_bound (F := F) l' l' L hL hF
  have hA : midA l' l' = pairs l'.listBound * (((l'.logCols + l'.lanes : ℕ) : ℝ) / qF) ^ 2 +
      2 * l'.listBound / qF + l'.queryError + l'.listBound * ceilLog2 l'.queries / qF +
      l'.listBound / qF := rfl
  have hL1 : 1 ≤ l'.listBound := by
    unfold Level.listBound; have := Nat.one_le_two_pow (n := l'.logInvRate); omega
  have hqE := padQueryError_nonneg l t
  have hqF : (0 : ℝ) < qF := by unfold qF; positivity
  have hp := pairs_nonneg l'.listBound
  set t1 := pairs l'.listBound * (((l'.logCols + l'.lanes : ℕ) : ℝ) / qF) ^ 2
  set t2 := 2 * (l'.listBound : ℝ) / qF
  set t4 := (l'.listBound : ℝ) * ceilLog2 l.queries / qF
  set t5 := (l'.listBound : ℝ) / qF
  have h1n : 0 ≤ t1 := mul_nonneg hp (sq_nonneg _)
  have h2n : 0 ≤ t2 := by positivity
  have h4n : 0 ≤ t4 := by positivity
  have h5n : 0 ≤ t5 := by positivity
  have hsplit : ENNReal.ofReal (midAPad l l' t) = ENNReal.ofReal t1 + ENNReal.ofReal t2 +
      ENNReal.ofReal (l.padQueryError t) + ENNReal.ofReal t4 + ENNReal.ofReal t5 := by
    unfold midAPad
    rw [ENNReal.ofReal_add (by positivity) h5n, ENNReal.ofReal_add (by positivity) h4n,
      ENNReal.ofReal_add (by positivity) hqE, ENNReal.ofReal_add h1n h2n]
  rw [hsplit, hF, Nat.cast_pow, Nat.cast_ofNat]
  have e1 : (L.choose 2 : ℝ≥0∞) * ((((l'.logCols + l'.lanes : ℕ) : ℕ) : ℝ≥0∞) / 2 ^ 128) ^ 2 ≤
      ENNReal.ofReal t1 := by
    simp only [t1]
    rw [ENNReal.ofReal_mul hp, ENNReal.ofReal_pow (by positivity), ofReal_nat_div_qF']
    gcongr
    rw [pairs, ← Nat.cast_choose_two, ENNReal.ofReal_natCast]
    exact_mod_cast Nat.choose_le_choose 2 hL
  have e2 : 2 * ((L : ℝ≥0∞) / 2 ^ 128) ≤ ENNReal.ofReal t2 := by
    simp only [t2]
    rw [show (2 : ℝ) * l'.listBound = ((2 * l'.listBound : ℕ) : ℝ) by push_cast; ring,
      ofReal_nat_div_qF', ← mul_div_assoc]
    gcongr
    exact_mod_cast Nat.mul_le_mul_left 2 hL
  have e4 : (ceilLog2 l.queries : ℝ≥0∞) / 2 ^ 128 ≤ ENNReal.ofReal t4 := by
    simp only [t4]
    rw [show (l'.listBound : ℝ) * ceilLog2 l.queries =
      ((l'.listBound * ceilLog2 l.queries : ℕ) : ℝ) by push_cast; ring, ofReal_nat_div_qF']
    gcongr
    exact_mod_cast Nat.le_mul_of_pos_left _ hL1
  have e5 : (L : ℝ≥0∞) / 2 ^ 128 ≤ ENNReal.ofReal t5 := by
    simp only [t5]
    rw [ofReal_nat_div_qF']
    gcongr
  calc _ ≤ ENNReal.ofReal t1 + ENNReal.ofReal t2 +
          (ENNReal.ofReal (l.padQueryError t) + (ENNReal.ofReal t4 + ENNReal.ofReal t5)) :=
        add_le_add (add_le_add e1 e2) (add_le_add le_rfl (add_le_add e4 e5))
    _ = _ := by ring

open Classical in
/-- **From the padded level 0 to level 1.** From `PostPad0`, the commitment of level 1, its two
OOD samples and level 0's queries on the padded column leave `PreFold` on level 1 except with
probability `midAPad`. -/
theorem phaseA0Pad (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F) (l l' : Level) (t : ℕ)
    (hd : l.logLen ≤ 40) (hd64 : l.logLen ≤ 64) (hcols : l.logCols ≤ l.logLen)
    (ht : t ≤ 2 ^ l.logCols) (hk : 2 ^ l.logCols + t ≤ 2 ^ l.logLen) (hf' : l'.Fits)
    (hQ : l.queries ≤ 2 ^ 64) (c₀ : F) (k0 e r : ℕ) (ch : List K) (ρ : Fin e → K)
    (ybar : Fin t → K) (st : LigState F K)
    (hpost : PostPad0 A hA n b l hd64 t c₀ k0 e r st.prev ch ρ st)
    (hdim : l.logCols + 1 = l'.lanes + l'.logCols)
    (k : Oracle F → LigState F K → Game Bool) (εk : ℝ≥0∞)
    (hk' : ∀ C' st', PreFold A n b ch l' C' l.logCols st' →
      value (fun r => r = true) (k C' st') ≤ εk) :
    value (fun r => r = true) (midRoundV A l (l'.logCols + l'.lanes)
      (padColumn A l.logLen l.logCols t c₀ ch ρ (extraLane k0 e r) ybar st.prev) st k) ≤
      ENNReal.ofReal (midAPad l l' t) + εk := by
  haveI : CharP K 2 := hA.charTwoK
  have hdim' : l'.logCols + l'.lanes = l.logCols + 1 := by omega
  set v := padColumn A l.logLen l.logCols t c₀ ch ρ (extraLane k0 e r) ybar st.prev
  show value _ (Game.send (Oracle F) fun C' => _) ≤ _
  rw [value_send]
  refine iSup_le fun C' => ?_
  set S' := closeL A l'.radius l'.lanes l'.logCols l'.logLen C'
  have hS' : S'.card ≤ l'.listBound :=
    card_close_le A hA l' hf' l'.lanes (tbl l'.lanes l'.logLen C')
  let base : Message F l'.lanes l'.logCols → K := fun M =>
    ∑ x ∈ Finset.range (2 ^ l.logCols), unsplit A (flat M) x *
      resAt A n b ch st.levels st.oods st.cons l.logCols x
  refine (value_bind_le _ _ _ (OodGood A S' (fun M => flat M) base (l'.logCols + l'.lanes)
      st.levels.length st.s) (ENNReal.ofReal (l.padQueryError t) + ((ceilLog2 l.queries : ℝ≥0∞) /
        Fintype.card F + S'.card / Fintype.card F) + εk) fun y hy => ?_).trans ?_
  · obtain ⟨s₂, q₂, oods₂⟩ := y
    obtain ⟨hsum₂, hshape, hgood⟩ := hy
    show value _ ((queryAndBatchV A l v st.levels.length
      { st with s := s₂, q := q₂, oods := st.oods ++ oods₂ }).bind (k C')) ≤ _
    refine (value_bind_le _ _ _ (PreFold A n b ch l' C' l.logCols) εk (hk' C')).trans
      (add_le_add ?_ le_rfl)
    refine value_queryAndBatchV_le A hA l hd64 hQ v st.levels.length
      { st with s := s₂, q := q₂, oods := st.oods ++ oods₂ } S'
      (fun M => oodDiff A (l'.logCols + l'.lanes) (flat M) (base M) (s₂, q₂, oods₂))
      (fun M => unsplit A (flat M)) (l.padQueryError t)
      (fun M hM hD M' hM' hD' => (hgood M hM hD).2 M' hM' hD')
      (fun M hM hD => query_bound_pad A hA n b l t hd hd64 hcols ht hk c₀ k0 e r st.prev ch ρ ybar
        st hpost _ (hgood M hM hD).1) _ ?_
    intro w α β st' h1 h2 h3 h4 h5 h6 hno
    exact exists_zero_of_not_preFoldV A hA n b ch l l' v st hpost.ok hdim C'
      s₂ q₂ oods₂ hsum₂ hshape _ _ β st' h1 h2 h3 h4 h6 hno
  · calc value (fun y => ¬ OodGood A S' (fun M => flat M) base (l'.logCols + l'.lanes)
            st.levels.length st.s y) (twoOod A (l'.logCols + l'.lanes) st.levels.length st.s st.q) +
          (ENNReal.ofReal (l.padQueryError t) + ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F +
            S'.card / Fintype.card F) + εk)
        ≤ ((S'.card.choose 2 : ℝ≥0∞) * (((l'.logCols + l'.lanes : ℕ) : ℝ≥0∞) / Fintype.card F) ^ 2 +
            2 * (S'.card / Fintype.card F)) + (ENNReal.ofReal (l.padQueryError t) +
              ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F + S'.card / Fintype.card F) + εk) :=
          add_le_add (value_twoOod_good A hA S' (fun M => flat M) (l'.logCols + l'.lanes)
            (fun M _ M' _ h => flat_inj M M' (fun y hy => h y (by rw [hdim', hdim]; exact hy)))
            base st.levels.length st.s st.q hpost.sum) le_rfl
      _ = ((S'.card.choose 2 : ℝ≥0∞) * (((l'.logCols + l'.lanes : ℕ) : ℝ≥0∞) /
            Fintype.card F) ^ 2 + 2 * (S'.card / Fintype.card F) +
              (ENNReal.ofReal (l.padQueryError t) +
              ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F + S'.card / Fintype.card F))) + εk := by
          ring
      _ ≤ ENNReal.ofReal (midAPad l l' t) + εk :=
          add_le_add (midAPad_bound l l' t S'.card hS' hA.card_F) le_rfl

end FlockSoundness
