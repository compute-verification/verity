import Proofs.Flock.Recursive.HoloExec

/-!
# Piece (a), R's opening count: the registered opening is sound at unique decoding

After both reps' sparse phases, `Model.regPhase` opens their two registered claims together (`opening`, the ring
switches and `γ`) and runs Ligerito at R's registered schedule on the registered word `w` (`regOpening`). Level 0 of
`fast100`'s schedule at `mReg` holds R's stacked bits exactly at rate at most `1/2` (`fast100_uniqueDecoding`), so a
message within `(1 − ρ)/2 − η` of `w` is the only one there. `regOpen_sound`: where no such message packs a table on
which both claims hold, the opening accepts with probability at most `regOpenError`, with no list (the reps' level 0 is
at the list-decoding radius; R's is not).

Ligerito's level 0 is `ligerito_sound`'s at the unique-decoding radius `udRadius`: its doom is over the (at most one)
message within `udRadius` of `w` (`card_closeL_ud_le`), the folds' MCA events stay at the Johnson radius, and the level's
queries catch a fold `udRadius`-far from it (`udQuery_bound`, `udQueryError`). Every later level is `ligerito_sound`'s.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge.Recursion
open Game Model Accounting
open scoped ENNReal

/-- **`fast100`'s schedule at `M` holds `M` bits at level 0 at rate `1/2`**: level 0 has `k₀` lanes of `2^logCols`
packed elements of 128 bits, `k₀ + logCols + 7 = M`, for every `M` it covers. -/
theorem fast100_uniqueDecoding (M : ℕ) (s : Accounting.Schedule) (hs : Accounting.fast100 M = some s) :
    s.levels ≠ [] ∧ s.m = M ∧ s.level₀.lanes = s.k0 ∧ s.k0 + s.level₀.logCols + 7 = M ∧
      1 ≤ s.level₀.logInvRate := by
  by_cases hm : M < 40
  · have key : ∀ M : Fin 40, ∀ s ∈ Accounting.fast100 M.val, s.levels ≠ [] ∧ s.m = M.val ∧
        s.level₀.lanes = s.k0 ∧ s.k0 + s.level₀.logCols + 7 = M.val ∧ 1 ≤ s.level₀.logInvRate := by
      decide +kernel
    exact key ⟨M, hm⟩ s hs
  · rw [Refine.model_fast100_none M (by omega)] at hs; cases hs

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **R's opening**, as `regPhase` runs it after both sparse phases: the two registered claims opened together, then
Ligerito at R's schedule `rsch` on the registered word `w`. -/
def regOpening (A : Arith F K) (sh : SparseShape) (hreg : 7 ≤ sh.mReg) (rsch : Schedule) (c₁ c₂ : RegClaim F sh)
    (w : Oracle F) : Game Bool := do
  match ← opening A hreg [c₁.toClaim, c₂.toClaim] with
  | none => pure false
  | some op => ligerito A rsch op.T op.b w

/-! ## The unique-decoding radius -/

/-- Level `l`'s unique-decoding radius `(1 − ρ)/2 − η`. -/
noncomputable def udRadius (l : Level) : ℝ := (1 - l.rate) / 2 - eta

theorem udRadius_le (l : Level) : udRadius l ≤ l.radius := by
  have hr : 0 ≤ l.rate := by unfold Level.rate; positivity
  have hs := Real.sq_sqrt hr
  unfold udRadius Level.radius
  nlinarith [sq_nonneg (1 - Real.sqrt l.rate)]

theorem udQueryError_eq (l : Level) : udQueryError l = (1 - udRadius l) ^ l.queries := by
  unfold udQueryError udRadius
  congr 1
  ring

/-- **At most one message is within `udRadius` of a table**: two such agree on more than `2^logCols` positions. -/
theorem card_closeL_ud_le (A : Arith F K) (hA : A.Correct) (l : Level) (hf : l.Fits) (lanes : ℕ)
    (C : Oracle F) : (closeL A (udRadius l) lanes l.logCols l.logLen C).card ≤ 1 := by
  classical
  have hagree : ∀ M ∈ closeL A (udRadius l) lanes l.logCols l.logLen C,
      ∃ T : Finset (Fin (2 ^ l.logLen)), (T.card : ℝ) ≥ (1 - udRadius l) * 2 ^ l.logLen ∧
        ∀ q ∈ T, ∀ ℓ, tbl lanes l.logLen C ℓ q = Model.encode A l.logLen M ℓ q := by
    intro M hM
    simp only [closeL, Finset.mem_filter, Finset.mem_univ, true_and] at hM
    unfold Model.Close at hM
    exact ⟨_, hM, fun q hq => (Finset.mem_filter.1 hq).2⟩
  refine Finset.card_le_one.2 fun M hM M' hM' => ?_
  obtain ⟨T, hT, hTM⟩ := hagree M hM
  obtain ⟨T', hT', hTM'⟩ := hagree M' hM'
  have hcap : 2 ^ l.logCols < (T ∩ T').card := by
    have hu : ((T ∪ T').card : ℝ) + (T ∩ T').card = T.card + T'.card := by
      exact_mod_cast Finset.card_union_add_card_inter T T'
    have hle : ((T ∪ T').card : ℝ) ≤ 2 ^ l.logLen := by
      have := Finset.card_le_univ (T ∪ T')
      rw [Fintype.card_fin] at this
      exact_mod_cast this
    have hlen : (2 : ℝ) ^ l.logLen * l.rate = 2 ^ l.logCols := by
      unfold Level.rate Level.logLen
      rw [pow_add, inv_pow, mul_assoc, mul_inv_cancel₀ (pow_ne_zero _ two_ne_zero), mul_one]
    have h2 : (0 : ℝ) < 2 ^ l.logLen := by positivity
    have heta : Accounting.eta = 1 / 200 := rfl
    unfold udRadius at hT hT'
    rw [heta] at hT hT'
    have : (2 : ℝ) ^ l.logCols < (T ∩ T').card := by nlinarith
    exact_mod_cast this
  funext ℓ c
  have hp : rowPoly A hA hf.1 (M ℓ) - rowPoly A hA hf.1 (M' ℓ) = 0 := by
    refine Polynomial.eq_zero_of_natDegree_lt_card_of_eval_eq_zero' _
      ((T ∩ T').image fun q => A.omega q.val) (fun ω hω => ?_) ?_
    · obtain ⟨q, hq, rfl⟩ := Finset.mem_image.1 hω
      have e := (hTM q (Finset.mem_inter.1 hq).1 ℓ).symm.trans (hTM' q (Finset.mem_inter.1 hq).2 ℓ)
      rw [Polynomial.eval_sub, rowPoly_eval A hA hf.1, rowPoly_eval A hA hf.1]
      exact sub_eq_zero.2 e
    · rw [Finset.card_image_of_injOn fun q _ q' _ hqq' =>
          Fin.ext (hA.omega_injective l.logLen hf.2 q.isLt q'.isLt hqq')]
      exact lt_of_le_of_lt (Polynomial.natDegree_sub_le _ _)
        ((max_lt (rowPoly_natDegree A hA hf.1 _) (rowPoly_natDegree A hA hf.1 _)).trans hcap)
  have hsum : ∑ c', (M ℓ c' - M' ℓ c') • xP A hA l.logCols hf.1 c' = 0 := by
    rw [← hp]
    simp only [rowPoly, ← Finset.sum_sub_distrib, sub_smul, Polynomial.smul_eq_C_mul, ← sub_mul,
      Polynomial.C_sub]
  have := Fintype.linearIndependent_iff.1 (xP_linearIndependent A hA l.logCols hf.1) _ hsum c
  exact sub_eq_zero.1 this

/-- **Decoding a fold at a radius `δ` within the Johnson radius** (`decode` at `δ`): the folds no MCA event and the
fold agreeing with the encoding of `G` on `(1 − δ)·2^d` positions, `G` is the fold of a message `δ`-close to the
table. -/
theorem udDecode (A : Arith F K) (hA : A.Correct) (l : Level) (hf : l.Fits) (lanes : ℕ)
    (C : Oracle F) (ch : List K) (hch : ch.length = lanes)
    (hgood : MCAGood (codeK A hA l hf) l.radius lanes ch (rowsK A lanes l.logLen C))
    (δ : ℝ) (hδ : δ ≤ l.radius) (G : ℕ → K) (T : Finset (Fin (2 ^ l.logLen)))
    (hT : (T.card : ℝ) ≥ Fintype.card (Fin (2 ^ l.logLen)) * (1 - δ))
    (hag : ∀ q ∈ T, foldColumn A ch (fun ℓ => C ℓ q.val) = encK A l.logCols G q.val) :
    ∃ M ∈ closeL A δ lanes l.logCols l.logLen C, ∀ x < 2 ^ l.logCols, G x = foldMsg A ch M x := by
  classical
  haveI : CharP K 2 := hA.charTwoK
  have hTJ : (T.card : ℝ) ≥ Fintype.card (Fin (2 ^ l.logLen)) * (1 - l.radius) :=
    le_trans (mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg _)) hT
  have hfold : ∀ q : Fin (2 ^ l.logLen), foldAllRows lanes ch (rowsK A lanes l.logLen C) q =
      foldColumn A ch (fun ℓ => C ℓ q.val) := by
    intro q
    rw [foldAllRows_eq_sum lanes ch hch, foldColumn, hch,
      ← Fin.sum_univ_eq_sum_range (fun ℓ => eqList ch ℓ * A.embed (C ℓ q.val))]
    rfl
  have hrows := rowsAgreeOn_of_foldAll (codeK A hA l hf) l.radius lanes ch _ hgood T hTJ
    ⟨_, encK_mem A hA l hf G, fun q hq => by rw [hfold q, hag q hq]⟩
  have hmsg : ∀ ℓ : Fin (2 ^ lanes), ∃ m : Fin (2 ^ l.logCols) → F,
      ∀ q ∈ T, A.encodeAt l.logCols m q.val = C ℓ q.val := by
    intro ℓ
    obtain ⟨c, hc, hcT⟩ := hrows ℓ
    haveI : NeZero (2 ^ l.logCols) := ⟨(Nat.two_pow_pos _).ne'⟩
    obtain ⟨P, hPdeg, hPc⟩ := ReedSolomon.mem_code_iff_eval_of_ne_zero.1 hc
    obtain ⟨P₀, P₁, hP₀, hPl⟩ := poly_limbs A hA P
    obtain ⟨m, hm⟩ := xP_span A hA l.logCols hf.1 P₀ (lt_of_le_of_lt hP₀ hPdeg)
    refine ⟨m, fun q hq => ?_⟩
    have h1 := hcT q hq
    rw [← hPc q] at h1
    simp only [domainK, Function.Embedding.trans_apply, Function.Embedding.coeFn_mk,
      domainEmb_apply] at h1
    rw [hPl, rowsK, ← ofLimbs_embed] at h1
    rw [← rowPoly_eval A hA hf.1, rowPoly, hm]
    exact (ofLimbs_inj A hA h1).1
  choose m hm using hmsg
  let M : Message F lanes l.logCols := fun ℓ x => m ℓ x
  have hclose : M ∈ closeL A δ lanes l.logCols l.logLen C := by
    rw [closeL, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    unfold Model.Close
    refine le_trans ?_ ((Nat.cast_le (α := ℝ)).2 (Finset.card_le_card (s := T) fun q hq => ?_))
    · rw [Fintype.card_fin] at hT; push_cast at hT; linarith
    · simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      intro ℓ
      simp only [tbl, Model.encode]
      exact (hm ℓ q hq).symm
  refine ⟨M, hclose, ?_⟩
  have hdiff : ∀ q ∈ T, encK A l.logCols (fun x => G x - foldMsg A ch M x) q.val = 0 := by
    intro q hq
    have h1 : encK A l.logCols (foldMsg A ch M) q.val = foldColumn A ch (fun ℓ => C ℓ q.val) := by
      unfold encK
      rw [← Fin.sum_univ_eq_sum_range (fun x => foldMsg A ch M x *
        A.embed (A.Xhat x l.logCols (A.omega q.val)))]
      simp only [foldMsg_eq A ch hch M, Finset.sum_mul]
      rw [Finset.sum_comm, foldColumn, hch,
        ← Fin.sum_univ_eq_sum_range (fun ℓ => eqList ch ℓ * A.embed (C ℓ q.val))]
      refine Finset.sum_congr rfl fun ℓ _ => ?_
      rw [← hm ℓ q hq, Arith.encodeAt, map_sum, Finset.mul_sum]
      exact Finset.sum_congr rfl fun x _ => by simp only [map_mul, M]; ring
    unfold encK at h1 ⊢
    simp only [sub_mul, Finset.sum_sub_distrib]
    rw [h1, hag q hq]
    unfold encK
    exact sub_self _
  intro x hx
  exact sub_eq_zero.1 (encK_eq_zero A hA l hf _ T (card_gt_of_agree l T hTJ).le hdiff x hx)

/-! ## Level 0 at the unique-decoding radius -/

/-- `PostFold` with the doom over the messages within `udRadius` of the level's table (the folds' MCA events still at
its radius). -/
structure PostFoldUD (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F) (initial : List K)
    (l : Level) (hf : l.Fits) (lanes : ℕ) (C : Oracle F) (ch : List K) (st : LigState F K) :
    Prop where
  sum : st.q.eval 0 + st.q.eval 1 = st.s
  len : ch.length = lanes
  ok : PostOK initial st.levels st.oods st.cons l.logCols
  mca : MCAGood (codeK A hA l hf) l.radius lanes ch (rowsK A lanes l.logLen C)
  doom : ∀ M ∈ closeL A (udRadius l) lanes l.logCols l.logLen C,
    st.s ≠ ∑ x ∈ Finset.range (2 ^ l.logCols),
      foldMsg A ch M x * resAt A n b initial st.levels st.oods st.cons l.logCols x

/-- **The far fold at `δ`** (`query_bound_of_doom` at `δ`): a claimed fold whose value is a claim false for every fold
of a message `δ`-close to the table survives the queries with probability at most `(1 − δ)^Q`. -/
theorem udQuery_bound (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F)
    (initial : List K) (l : Level) (hf : l.Fits) (lanes : ℕ) (C : Oracle F) (ch : List K)
    (st : LigState F K) (δ : ℝ) (hδ : δ ≤ l.radius) (hlen : ch.length = lanes)
    (hmca : MCAGood (codeK A hA l hf) l.radius lanes ch (rowsK A lanes l.logLen C))
    (hdoom : ∀ M ∈ closeL A δ lanes l.logCols l.logLen C,
      st.s ≠ ∑ x ∈ Finset.range (2 ^ l.logCols),
        foldMsg A ch M x * resAt A n b initial st.levels st.oods st.cons l.logCols x)
    (G : ℕ → K) (hs : st.s = ∑ x ∈ Finset.range (2 ^ l.logCols),
      G x * resAt A n b initial st.levels st.oods st.cons l.logCols x) :
    (1 - ((mismatch A l.logCols l.logLen C ch G).card : ℝ) / 2 ^ l.logLen) ^ l.queries ≤
      (1 - δ) ^ l.queries := by
  classical
  have hmis : (mismatch A l.logCols l.logLen C ch G).card ≤ 2 ^ l.logLen := by
    have := card_agree_add_mismatch A l.logCols l.logLen C ch G
    omega
  have h2 : (0 : ℝ) < 2 ^ l.logLen := by positivity
  have hle : 1 - ((mismatch A l.logCols l.logLen C ch G).card : ℝ) / 2 ^ l.logLen ≤ 1 - δ := by
    by_contra hgt
    push Not at hgt
    set T := Finset.univ.filter fun q : Fin (2 ^ l.logLen) =>
      foldColumn A ch (fun ℓ => C ℓ q.val) = encK A l.logCols G q.val
    have hT := card_agree_add_mismatch A l.logCols l.logLen C ch G
    have hTr : (T.card : ℝ) = 2 ^ l.logLen - (mismatch A l.logCols l.logLen C ch G).card := by
      rw [eq_sub_iff_add_eq]; exact_mod_cast hT
    have hTc : (T.card : ℝ) ≥ Fintype.card (Fin (2 ^ l.logLen)) * (1 - δ) := by
      have hlt : ((mismatch A l.logCols l.logLen C ch G).card : ℝ) / 2 ^ l.logLen < δ := by linarith
      rw [div_lt_iff₀ h2] at hlt
      rw [Fintype.card_fin, hTr]
      push_cast
      linarith
    obtain ⟨M₀, hM₀, hG⟩ := udDecode A hA l hf lanes C ch hlen hmca δ hδ G T hTc
      fun q hq => (Finset.mem_filter.1 hq).2
    refine hdoom M₀ hM₀ ?_
    rw [hs]
    exact Finset.sum_congr rfl fun x hx => by rw [hG x (Finset.mem_range.1 hx)]
  have hnn : 0 ≤ 1 - ((mismatch A l.logCols l.logLen C ch G).card : ℝ) / 2 ^ l.logLen := by
    rw [sub_nonneg, div_le_one h2]; exact_mod_cast hmis
  exact pow_le_pow_left₀ hnn hle _

open Classical in
/-- `value_queryAndBatch_le` (with the `K` message sent) for any bound `ε` on the queries missing the mismatch. -/
theorem udQueryAndBatch_le (A : Arith F K) (hA : A.Correct) (l : Level) (hd : l.logLen ≤ 64)
    (hQ : l.queries ≤ 2 ^ 64) (C : Oracle F) (ch : List K) (start : ℕ) (st : LigState F K)
    {ι : Type} (S : Finset ι) (D : ι → K) (G : ι → ℕ → K) (ε : ℝ)
    (huniq : ∀ M ∈ S, D M = 0 → ∀ M' ∈ S, D M' = 0 → M' = M)
    (hfar : ∀ M ∈ S, D M = 0 →
      (1 - ((mismatch A l.logCols l.logLen C ch (G M)).card : ℝ) / 2 ^ l.logLen) ^ l.queries ≤ ε)
    (P : LigState F K → Prop)
    (hP : ∀ (w : Fin l.queries → F) (α : Fin (ceilLog2 l.queries) → F) (β : F) (st' : LigState F K),
      st'.s = st.s + A.embed β * consValue A (List.ofFn α) ch C (positions A l.queries l.logLen w) →
      st'.cons = st.cons ++ [⟨l.logCols, positions A l.queries l.logLen w, List.ofFn α, β, start⟩] →
      st'.levels = st.levels → st'.oods = st.oods → st'.prev = st.prev →
      st'.q.eval 0 + st'.q.eval 1 = st.q.eval 0 + st.q.eval 1 +
        A.embed β * consValue A (List.ofFn α) ch C (positions A l.queries l.logLen w) →
      P st' → ∃ M ∈ S, D M + A.embed β *
        consGap A l.logCols C ch (G M) (List.ofFn α) (positions A l.queries l.logLen w) = 0) :
    value P (queryAndBatch A l C ch start st true) ≤
      ENNReal.ofReal ε + ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F + S.card / Fintype.card F) := by
  haveI : CharP K 2 := hA.charTwoK
  set Q := l.queries
  set pos := fun w : Fin Q → F => positions A Q l.logLen w
  set gap := fun (M : ι) (w : Fin Q → F) (α : Fin (ceilLog2 Q) → F) =>
    consGap A l.logCols C ch (G M) (List.ofFn α) (pos w)
  let Bq : (Fin Q → F) → Prop := fun w =>
    ∃ M ∈ S, D M = 0 ∧ ∀ p ∈ pos w, p ∉ mismatch A l.logCols l.logLen C ch (G M)
  let Bα : (Fin Q → F) → (Fin (ceilLog2 Q) → F) → Prop := fun w α =>
    ∃ M ∈ S, D M = 0 ∧ gap M w α = 0
  let Bβ : (Fin Q → F) → (Fin (ceilLog2 Q) → F) → F → Prop := fun w α β =>
    ∃ M ∈ S, D M + A.embed β * gap M w α = 0 ∧ ¬ (D M = 0 ∧ gap M w α = 0)
  have hBq : prCoin Bq ≤ ENNReal.ofReal ε := by
    by_cases hex : ∃ M ∈ S, D M = 0
    · obtain ⟨M, hM, hD⟩ := hex
      refine (prCoin_mono (q := fun w => ∀ p ∈ positions A Q l.logLen w,
          p ∉ mismatch A l.logCols l.logLen C ch (G M)) fun w ⟨M', hM', hD', h⟩ => by
        rw [← huniq M hM hD M' hM' hD']; exact h).trans ?_
      refine (prCoin_positions_miss_le A hA Q l.logLen hd _ (Finset.filter_subset _ _)).trans ?_
      exact ENNReal.ofReal_le_ofReal (hfar M hM hD)
    · refine (prCoin_mono (q := fun _ => False) fun w ⟨M, hM, hD, _⟩ => hex ⟨M, hM, hD⟩).trans ?_
      simp [prCoin, avg]
  have hBα : ∀ w, ¬ Bq w → prCoin (Bα w) ≤ (ceilLog2 Q : ℝ≥0∞) / Fintype.card F := by
    intro w hw
    by_cases hex : ∃ M ∈ S, D M = 0
    · obtain ⟨M, hM, hD⟩ := hex
      obtain ⟨p, hp, hpm⟩ : ∃ p ∈ pos w, p ∈ mismatch A l.logCols l.logLen C ch (G M) := by
        by_contra hno
        push Not at hno
        exact hw ⟨M, hM, hD, hno⟩
      obtain ⟨j₀, hj₀, hpj⟩ := List.getElem_of_mem hp
      have hlen : (pos w).length = Q := positions_length A Q l.logLen w
      set v : ℕ → K := fun j => foldColumn A ch (fun ℓ => C ℓ ((pos w).getD j 0)) -
        encK A l.logCols (G M) ((pos w).getD j 0)
      have hgap : ∀ α, gap M w α = ∑ j ∈ Finset.range Q, A.embed (eqList (List.ofFn α) j) * v j := by
        intro α
        simp only [gap, consGap, List.zipIdx]
        rw [sum_map_zipIdx (fun pj => A.embed (eqList (List.ofFn α) pj.2) *
          (foldColumn A ch (fun ℓ => C ℓ pj.1) - encK A l.logCols (G M) pj.1)) (pos w) 0, hlen]
        simp only [zero_add, v]
      have hv : v j₀ ≠ 0 := by
        simp only [v, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj₀, Option.getD_some,
          hpj]
        simp only [mismatch, Finset.mem_filter] at hpm
        exact sub_ne_zero.2 hpm.2
      refine (prCoin_mono (q := fun α => ∑ j ∈ Finset.range Q,
          A.embed (eqList (List.ofFn α) j) * v j = 0) fun α ⟨M', hM', hD', h⟩ => by
        rw [← hgap, huniq M hM hD M' hM' hD'] at *; exact h).trans ?_
      exact prCoin_embed_eqList_le A hA (le_two_pow_ceilLog2 Q hQ) v j₀ (by omega) hv
    · refine (prCoin_mono (q := fun _ => False) fun α ⟨M, hM, hD, _⟩ => hex ⟨M, hM, hD⟩).trans ?_
      simp [prCoin, avg]
  have hBβ : ∀ w α, prCoin (Bβ w α) ≤ S.card / Fintype.card F := fun w α =>
    prCoin_embed_affine_union_le A S D fun M => gap M w α
  have hleaf : ∀ w α β st', ¬ Bα w α → ¬ Bβ w α β →
      st'.s = st.s + A.embed β * consValue A (List.ofFn α) ch C (pos w) →
      st'.cons = st.cons ++ [⟨l.logCols, pos w, List.ofFn α, β, start⟩] →
      st'.levels = st.levels → st'.oods = st.oods → st'.prev = st.prev →
      st'.q.eval 0 + st'.q.eval 1 = st.q.eval 0 + st.q.eval 1 +
        A.embed β * consValue A (List.ofFn α) ch C (pos w) → ¬ P st' := by
    intro w α β st' hα hβ h1 h2 h3 h4 h5 h6 hPst
    obtain ⟨M, hM, h0⟩ := hP w α β st' h1 h2 h3 h4 h5 h6 hPst
    have : D M = 0 ∧ gap M w α = 0 := by
      by_contra hn; exact hβ ⟨M, hM, h0, hn⟩
    exact hα ⟨M, hM, this.1, this.2⟩
  rw [queryAndBatch_true]
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

/-- `exists_zero_of_not_preFold` from the lists' shape alone. -/
theorem udExistsZero (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K) (l l' : Level)
    (C : Oracle F) (ch : List K) (st : LigState F K)
    (hok : PostOK initial st.levels st.oods st.cons l.logCols)
    (hdim : l.logCols + 1 = l'.lanes + l'.logCols) (C' : Oracle F)
    (s₂ : K) (q₂ : Quad K) (oods₂ : List (OodCtx F)) (hsum₂ : q₂.eval 0 + q₂.eval 1 = s₂)
    (hshape : ∀ o ∈ oods₂, o.split = some st.levels.length ∧ o.z.length = l'.logCols + l'.lanes)
    (pos : List ℕ) (α : List F) (β : F) (st' : LigState F K)
    (h1 : st'.s = s₂ + A.embed β * consValue A α ch C pos)
    (h2 : st'.cons = st.cons ++ [⟨l.logCols, pos, α, β, st.levels.length⟩])
    (h3 : st'.levels = st.levels) (h4 : st'.oods = st.oods ++ oods₂)
    (h6 : st'.q.eval 0 + st'.q.eval 1 = q₂.eval 0 + q₂.eval 1 + A.embed β * consValue A α ch C pos)
    (hno : ¬ PreFold A n b initial l' C' l.logCols st') :
    ∃ M ∈ closeL A l'.radius l'.lanes l'.logCols l'.logLen C',
      oodDiff A (l'.logCols + l'.lanes) (flat M)
          (∑ x ∈ Finset.range (2 ^ l.logCols), unsplit A (flat M) x *
            resAt A n b initial st.levels st.oods st.cons l.logCols x) (s₂, q₂, oods₂) +
        A.embed β * consGap A l.logCols C ch (unsplit A (flat M)) α pos = 0 := by
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
  have hcv : consValue A α ch C pos = ((pos.zipIdx).map fun pj =>
      A.embed (eqList α pj.2) * foldColumn A ch (fun ℓ => C ℓ pj.1)).sum := rfl
  have hgap : consGap A l.logCols C ch (unsplit A (flat M)) α pos =
      consValue A α ch C pos - ((pos.zipIdx).map fun pj =>
        A.embed (eqList α pj.2) * encK A l.logCols (unsplit A (flat M)) pj.1).sum := by
    rw [hcv, consGap, ← list_sum_map_sub]
    exact congrArg List.sum (List.map_congr_left fun pj _ => by ring)
  rw [hgap, oodDiff, hdim']
  dsimp only
  linear_combination hM'

theorem ennreal_reshuffle (a c d u e : ℝ≥0∞) : a + (u + c + e) ≤ a + (d + c) + u + e :=
  calc a + (u + c + e) = a + c + u + e := by ring
    _ ≤ a + (d + c) + u + e := add_le_add (add_le_add (add_le_add le_rfl le_add_self) le_rfl) le_rfl

open Classical in
/-- **From level 0 to level 1 at unique decoding** (`phaseA` from `PostFoldUD`): level 0's queries cost `udQueryError`
on top of `midA`. -/
theorem udPhaseA (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F) (initial : List K)
    (l l' : Level) (hf : l.Fits) (hf' : l'.Fits) (hQ : l.queries ≤ 2 ^ 64) (lanes : ℕ)
    (ch : List K) (st : LigState F K) (hpost : PostFoldUD A hA n b initial l hf lanes st.prev ch st)
    (hdim : l.logCols + 1 = l'.lanes + l'.logCols) (dim : ℕ) (hd : dim = l'.logCols + l'.lanes)
    (k : Oracle F → LigState F K → Game Bool) (εk : ℝ≥0∞)
    (hk : ∀ C' st', PreFold A n b initial l' C' l.logCols st' →
      value (fun r => r = true) (k C' st') ≤ εk) :
    value (fun r => r = true) (midRound A l dim ch st k) ≤
      ENNReal.ofReal (midA l l') + ENNReal.ofReal (udQueryError l) + εk := by
  subst hd
  haveI : CharP K 2 := hA.charTwoK
  have hdim' : l'.logCols + l'.lanes = l.logCols + 1 := by omega
  show value _ (Game.send (Oracle F) fun C' => _) ≤ _
  rw [value_send]
  refine iSup_le fun C' => ?_
  set S' := closeL A l'.radius l'.lanes l'.logCols l'.logLen C'
  have hS' : S'.card ≤ l'.listBound :=
    card_close_le A hA l' hf' l'.lanes (tbl l'.lanes l'.logLen C')
  let base : Message F l'.lanes l'.logCols → K := fun M =>
    ∑ x ∈ Finset.range (2 ^ l.logCols), unsplit A (flat M) x *
      resAt A n b initial st.levels st.oods st.cons l.logCols x
  refine (value_bind_le _ _ _ (OodGood A S' (fun M => flat M) base (l'.logCols + l'.lanes)
      st.levels.length st.s) (ENNReal.ofReal (udQueryError l) + ((ceilLog2 l.queries : ℝ≥0∞) /
        Fintype.card F + S'.card / Fintype.card F) + εk) fun y hy => ?_).trans ?_
  · obtain ⟨s₂, q₂, oods₂⟩ := y
    obtain ⟨hsum₂, hshape, hgood⟩ := hy
    show value _ ((queryAndBatch A l st.prev ch st.levels.length
      { st with s := s₂, q := q₂, oods := st.oods ++ oods₂ } true).bind (k C')) ≤ _
    refine (value_bind_le _ _ _ (PreFold A n b initial l' C' l.logCols) εk (hk C')).trans
      (add_le_add ?_ le_rfl)
    refine udQueryAndBatch_le A hA l hf.2 hQ st.prev ch st.levels.length
      { st with s := s₂, q := q₂, oods := st.oods ++ oods₂ } S'
      (fun M => oodDiff A (l'.logCols + l'.lanes) (flat M) (base M) (s₂, q₂, oods₂))
      (fun M => unsplit A (flat M)) (udQueryError l)
      (fun M hM hD M' hM' hD' => (hgood M hM hD).2 M' hM' hD')
      (fun M hM hD => (udQuery_bound A hA n b initial l hf lanes st.prev ch st (udRadius l)
        (udRadius_le l) hpost.len hpost.mca hpost.doom _ (hgood M hM hD).1).trans_eq
        (udQueryError_eq l).symm) _ ?_
    intro w α β st' h1 h2 h3 h4 h5 h6 hno
    exact udExistsZero A n b initial l l' st.prev ch st hpost.ok hdim C'
      s₂ q₂ oods₂ hsum₂ hshape _ _ β st' h1 h2 h3 h4 h6 hno
  · calc value (fun y => ¬ OodGood A S' (fun M => flat M) base (l'.logCols + l'.lanes)
            st.levels.length st.s y) (twoOod A (l'.logCols + l'.lanes) st.levels.length st.s st.q) +
          (ENNReal.ofReal (udQueryError l) + ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F +
            S'.card / Fintype.card F) + εk)
        ≤ ((S'.card.choose 2 : ℝ≥0∞) * (((l'.logCols + l'.lanes : ℕ) : ℝ≥0∞) / Fintype.card F) ^ 2 +
            2 * (S'.card / Fintype.card F)) + (ENNReal.ofReal (udQueryError l) +
              ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F + S'.card / Fintype.card F) + εk) :=
          add_le_add (value_twoOod_good A hA S' (fun M => flat M) (l'.logCols + l'.lanes)
            (fun M _ M' _ h => flat_inj M M' (fun y hy => h y (by rw [hdim', hdim]; exact hy)))
            base st.levels.length st.s st.q hpost.sum) le_rfl
      _ ≤ ((S'.card.choose 2 : ℝ≥0∞) * (((l'.logCols + l'.lanes : ℕ) : ℝ≥0∞) / Fintype.card F) ^ 2 +
            2 * (S'.card / Fintype.card F) + (ENNReal.ofReal l.queryError +
              ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F + S'.card / Fintype.card F))) +
            ENNReal.ofReal (udQueryError l) + εk := ennreal_reshuffle _ _ _ _ _
      _ ≤ ENNReal.ofReal (midA l l') + ENNReal.ofReal (udQueryError l) + εk :=
          add_le_add (add_le_add (midA_bound l l' S'.card hS' hA.card_F) le_rfl) le_rfl

/-- **The level-0 folds at unique decoding** (`phaseB0` from a claim false for every message within `udRadius`). -/
theorem udPhaseB0 (A : Arith F K) (hA : A.Correct) (n : ℕ)
    (b : ℕ → F) (l : Level) (hl : l.Ok) (hf : l.Fits) (k0 : ℕ) (hk0 : l.lanes = k0)
    (hn : n = k0 + l.logCols) (C : Oracle F) (z₀ : List F) (hz : z₀.length = n) (β₀ : F) (s₀ : K)
    (q₀ : Quad K) (hq₀ : q₀.eval 0 + q₀.eval 1 = s₀) (dim : ℕ)
    (hdoom : ∀ M ∈ closeL A (udRadius l) k0 l.logCols l.logLen C,
      s₀ ≠ ∑ y ∈ Finset.range (2 ^ n), A.embed (flat M y) * W0 A b z₀ β₀ y) :
    value (fun out : List K × K × Quad K => ¬ PostFoldUD A hA n b out.1 l hf k0 C out.1
        ⟨out.2.1, out.2.2, dim, [], [⟨z₀, β₀, none⟩], [], C, l⟩) (foldRounds A k0 s₀ q₀) ≤
      ENNReal.ofReal (l.foldError l.listBound) := by
  classical
  haveI : CharP K 2 := hA.charTwoK
  set S := closeL A (udRadius l) k0 l.logCols l.logLen C
  have hS : S.card ≤ l.listBound := (card_closeL_ud_le A hA l hf k0 C).trans (by
    unfold Level.listBound; have := Nat.one_le_two_pow (n := l.logInvRate); omega)
  set u : Message F k0 l.logCols → ℕ → K := fun M y => A.embed (flat M y)
  set v := W0 A b z₀ β₀
  have himp : ∀ out : List K × K × Quad K, ¬ PostFoldUD A hA n b out.1 l hf k0 C out.1
      ⟨out.2.1, out.2.2, dim, [], [⟨z₀, β₀, none⟩], [], C, l⟩ →
      (out.1.length ≠ k0 ∨ out.2.2.eval 0 + out.2.2.eval 1 ≠ out.2.1) ∨
        (¬ MCAGood (codeK A hA l hf) l.radius k0 out.1 (rowsK A k0 l.logLen C) ∨
          ∃ M ∈ S, out.2.1 = ∑ y ∈ Finset.range (2 ^ l.logCols),
            foldLoL out.1 (u M) y * foldLoL out.1 v y) := by
    intro out hno
    by_contra hall
    simp only [not_or, not_not, not_exists, not_and] at hall
    obtain ⟨⟨hlen, hsum⟩, hmca, hdoom'⟩ := hall
    refine hno ⟨hsum, hlen, ⟨fun o ho => ?_, fun c hc => by simp at hc⟩, hmca,
      fun M hM heq => hdoom' M hM ?_⟩
    · simp only [List.mem_singleton] at ho
      subst ho
      exact ⟨Or.inl rfl, by simp [oodFixed, origOf, hz, hlen, hn]⟩
    · have heq' : out.2.1 = _ := heq
      rw [heq']
      refine Finset.sum_congr rfl fun y hy => ?_
      rw [foldLoL_W0 A n b z₀ hz β₀ out.1 l.logCols (by omega) y (Finset.mem_range.1 hy)]
      rfl
  have hsumε := value_foldRounds_sum_le A hA S u v k0 l.logCols s₀ q₀ hq₀
    (fun M hM => by rw [← hn]; exact hdoom M hM)
  calc value _ (foldRounds A k0 s₀ q₀)
      ≤ value (fun out : List K × K × Quad K =>
          (out.1.length ≠ k0 ∨ out.2.2.eval 0 + out.2.2.eval 1 ≠ out.2.1) ∨
            (¬ MCAGood (codeK A hA l hf) l.radius k0 out.1 (rowsK A k0 l.logLen C) ∨
              ∃ M ∈ S, out.2.1 = ∑ y ∈ Finset.range (2 ^ l.logCols),
                foldLoL out.1 (u M) y * foldLoL out.1 v y)) (foldRounds A k0 s₀ q₀) :=
        value_mono himp _
    _ ≤ (value (fun out : List K × K × Quad K => out.1.length ≠ k0) _ +
          value (fun out : List K × K × Quad K => out.2.2.eval 0 + out.2.2.eval 1 ≠ out.2.1) _) +
        (value (fun out : List K × K × Quad K =>
            ¬ MCAGood (codeK A hA l hf) l.radius k0 out.1 (rowsK A k0 l.logLen C)) _ +
          value (fun out : List K × K × Quad K => ∃ M ∈ S, out.2.1 =
            ∑ y ∈ Finset.range (2 ^ l.logCols), foldLoL out.1 (u M) y * foldLoL out.1 v y) _) :=
        (value_or_le _ _ _).trans (add_le_add (value_or_le _ _ _) (value_or_le _ _ _))
    _ ≤ (0 + 0) + (((2 ^ k0 - 1 : ℕ) : ℝ≥0∞) * ENNReal.ofReal (l.mcaA / qK) +
          ((2 * k0 * S.card : ℕ) : ℝ≥0∞) / Fintype.card K) :=
        add_le_add (add_le_add (value_foldRounds_length A hA _ _ _).le
          (value_foldRounds_sum_invariant A hA _ _ _ hq₀).le)
          (add_le_add (value_foldRounds_mca_le A hA (codeK A hA l hf) l.radius _
            (prCoin_level_mca_le A hA l hl hf) k0 _ _ _) hsumε)
    _ ≤ ENNReal.ofReal (l.foldError l.listBound) := by
        rw [zero_add, zero_add, add_comm]
        rcases foldError_bound (K := K) l l.listBound (mcaA_nonneg l hl) with h | h
        · rw [hk0] at h
          refine le_trans (add_le_add ?_ le_rfl) h
          gcongr
        · exact absurd hA.card_K h

/-- **Ligerito is sound at unique decoding** (`ligerito_sound` with level 0 at `udRadius`): if the claim is false for
every message within `udRadius` of the committed level-0 table, Ligerito accepts with probability at most
`1/|F| + udQueryError + ligeritoError`. -/
theorem ligerito_sound_ud (A : Arith F K) (hA : A.Correct) (sch : Schedule)
    (hshape : 2 ≤ sch.levels.length ∧ (sch.levels.headD default).lanes = sch.k0 ∧
      sch.m - 7 = sch.k0 + (sch.levels.headD default).logCols ∧
        levelsChained sch.levels sch.ext = true)
    (T : F) (b : ℕ → F) (C : Oracle F)
    (hdoom : ∀ M ∈ closeL A (udRadius sch.level₀) sch.k0 sch.level₀.logCols sch.level₀.logLen C,
      (∑ y ∈ Finset.range (2 ^ (sch.m - 7)), flat M y * b y) ≠ T) :
    value (fun r => r = true) (ligerito A sch T b C) ≤
      ENNReal.ofReal (1 / qF + udQueryError sch.level₀ + ligeritoError sch.levels) := by
  classical
  haveI : CharP K 2 := hA.charTwoK
  obtain ⟨m, k0, levels, ext⟩ := sch
  obtain ⟨hlen, hk0, hn, hchain⟩ := hshape
  rcases levels with _ | ⟨l₀, _ | ⟨l₁, rest⟩⟩
  · simp at hlen
  · simp at hlen
  dsimp only [List.headD, Schedule.level₀] at hk0 hn hdoom ⊢
  have h' := hchain
  simp only [levelsChained, Bool.and_eq_true, decide_eq_true_eq] at h'
  obtain ⟨⟨hgood₀, hdim₀⟩, hrest⟩ := h'
  obtain ⟨hl₀, hf₀, _, hQ₀⟩ := levelGood_spec hgood₀
  obtain ⟨_, hf₁, _, _⟩ := levelGood_spec (levelsChained_head_good hrest)
  set S₀ := closeL A (udRadius l₀) k0 l₀.logCols l₀.logLen C
  have hS₀ : S₀.card ≤ 1 := card_closeL_ud_le A hA l₀ hf₀ k0 C
  set ε₂ : ℝ≥0∞ := ENNReal.ofReal (l₀.foldError l₀.listBound) +
    (ENNReal.ofReal (midA l₀ l₁) + ENNReal.ofReal (udQueryError l₀) +
      ENNReal.ofReal (ligeritoError (l₁ :: rest)))
  rw [ligerito_eq]
  refine (value_coin_le_of_forall _ _ _ (S₀.card / Fintype.card F + ε₂) fun z₀ => ?_).trans ?_
  · rw [value_send]
    refine iSup_le fun y₀ => ?_
    let s₀ : F → K := fun β₀ => A.embed T + A.embed β₀ * A.embed y₀
    refine value_coin_le _ F _ (fun β₀ => ∃ M ∈ S₀, s₀ β₀ = ∑ y ∈ Finset.range (2 ^ (m - 7)),
        A.embed (flat M y) * W0 A b (List.ofFn z₀) β₀ y) _ _ ?_ fun β₀ hβ₀ => ?_
    · refine le_trans (prCoin_mono fun β₀ hβ => ?_) (prCoin_embed_affine_union_le A S₀
        (fun M => A.embed T - A.embed (∑ y ∈ Finset.range (2 ^ (m - 7)), flat M y * b y))
        (fun M => A.embed y₀ - mleK A (m - 7) (flat M) (List.ofFn z₀)))
      obtain ⟨M, hM, heq⟩ := hβ
      have htruth : ∑ y ∈ Finset.range (2 ^ (m - 7)), A.embed (flat M y) *
          W0 A b (List.ofFn z₀) β₀ y =
          A.embed (∑ y ∈ Finset.range (2 ^ (m - 7)), flat M y * b y) +
            A.embed β₀ * mleK A (m - 7) (flat M) (List.ofFn z₀) := by
        simp only [W0, mleK, mul_add, Finset.sum_add_distrib, Finset.mul_sum, map_sum, map_mul]
        congr 1
        exact Finset.sum_congr rfl fun y _ => by ring
      refine ⟨M, hM, ?_, fun ⟨h0, _⟩ => hdoom M hM ?_⟩
      · have heq' : A.embed T + A.embed β₀ * A.embed y₀ = _ := heq
        linear_combination heq' + htruth
      · exact (A.embed.injective (sub_eq_zero.1 h0)).symm
    · rw [value_send]
      refine iSup_le fun msg => ?_
      have hq₀ := fromMsg_sum msg (s₀ β₀)
      refine (value_bind_le (foldRounds A k0 (s₀ β₀) (fromMsg msg (s₀ β₀))) _ _
        (fun out : List K × K × Quad K => PostFoldUD A hA (m - 7) b out.1 l₀ hf₀ k0 C out.1
          ⟨out.2.1, out.2.2, m - 7 - k0 + 1, [], [⟨List.ofFn z₀, β₀, none⟩], [], C, l₀⟩)
        (ENNReal.ofReal (midA l₀ l₁) + ENNReal.ofReal (udQueryError l₀) +
          ENNReal.ofReal (ligeritoError (l₁ :: rest)))
        fun out hpost => ?_).trans ?_
      · obtain ⟨initial, s, q⟩ := out
        exact udPhaseA A hA (m - 7) b initial l₀ l₁ hf₀ hf₁ hQ₀ k0 initial
          ⟨s, q, m - 7 - k0 + 1, [], [⟨List.ofFn z₀, β₀, none⟩], [], C, l₀⟩ hpost hdim₀
          (m - 7 - k0 + 1) (by omega) _ _ fun C' st' hpre' =>
            value_recursiveRounds_le A hA (m - 7) b initial ext (l₁ :: rest)
              { st' with prev := C', prevLevel := l₁ } l₀.logCols hrest
              ⟨hpre'.sum, hpre'.ok, hpre'.dim, hpre'.doom⟩
      · exact add_le_add (udPhaseB0 A hA (m - 7) b l₀ hl₀ hf₀ k0 hk0 hn C (List.ofFn z₀)
          (by simp) β₀ (s₀ β₀) _ hq₀ (m - 7 - k0 + 1) fun M hM h => hβ₀ ⟨M, hM, h⟩) le_rfl
  · have hok := levelsChained_ok hchain
    have hq : (0 : ℝ) < qF := by unfold qF; positivity
    have hfe := foldError_nonneg l₀ (max l₀.listBound l₁.listBound) hl₀
    have hm := midA_nonneg l₀ l₁
    have hr := ligeritoError_nonneg (l₁ :: rest) (fun l hl => hok l (List.mem_cons_of_mem _ hl))
    have h1 : (0 : ℝ) ≤ 1 / qF := by positivity
    have hu : (0 : ℝ) ≤ udQueryError l₀ := by
      unfold udQueryError Level.rate Accounting.eta; positivity
    calc (S₀.card : ℝ≥0∞) / Fintype.card F + ε₂
        ≤ ENNReal.ofReal (1 / qF) +
            (ENNReal.ofReal (l₀.foldError (max l₀.listBound l₁.listBound)) +
              (ENNReal.ofReal (midA l₀ l₁) + ENNReal.ofReal (udQueryError l₀) +
                ENNReal.ofReal (ligeritoError (l₁ :: rest)))) := by
          refine add_le_add ?_ (add_le_add (ENNReal.ofReal_le_ofReal
            (foldError_mono l₀ (le_max_left _ _))) le_rfl)
          calc (S₀.card : ℝ≥0∞) / Fintype.card F ≤ ((1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
                gcongr <;> exact_mod_cast hS₀
            _ = ENNReal.ofReal (1 / qF) := by
                rw [show (1 : ℝ) / qF = ((1 : ℕ) : ℝ) / qF by simp, ofReal_nat_div_qF', hA.card_F]
                norm_num
      _ = ENNReal.ofReal (1 / qF + udQueryError l₀ + ligeritoError (l₀ :: l₁ :: rest)) := by
          rw [ligeritoError, midLevelError_eq,
            ENNReal.ofReal_add (add_nonneg h1 hu) (add_nonneg (add_nonneg hfe hm) hr),
            ENNReal.ofReal_add h1 hu, ENNReal.ofReal_add (add_nonneg hfe hm) hr,
            ENNReal.ofReal_add hfe hm]
          ring

/-! ## R's opening -/

/-- **R's opening is sound at unique decoding**: at `fast100`'s schedule for the stacked table, if no message within
level 0's unique-decoding radius of `w` packs a table on which both claims hold, the opening accepts with probability at
most `regOpenError`. -/
theorem regOpen_sound (A : Arith F K) (hA : A.Correct) (sh : SparseShape) (rsch : Schedule)
    (hf : Accounting.fast100 sh.mReg = some rsch) (hreg : 7 ≤ sh.mReg) (c₁ c₂ : RegClaim F sh) (w : Oracle F)
    (hdoom : ∀ M : Message F rsch.k0 rsch.level₀.logCols,
      Model.Close ((1 - rsch.level₀.rate) / 2 - eta) (asTable rsch w) (Model.encode A rsch.level₀.logLen M) →
        ¬ (c₁.toClaim.Holds (witnessOf A sh.mReg rsch.k0 rsch.level₀.logCols M) ∧
          c₂.toClaim.Holds (witnessOf A sh.mReg rsch.k0 rsch.level₀.logCols M))) :
    value (fun b => b = true) (regOpening A sh hreg rsch c₁ c₂ w) ≤ ENNReal.ofReal (regOpenError rsch) := by
  classical
  obtain ⟨-, hm, -, -, -⟩ := fast100_uniqueDecoding sh.mReg rsch hf
  have hshape := fast100_shape sh.mReg rsch hf
  have hfit : rsch.level₀.Fits := (levelGood_spec (levelsChained_head_good hshape.2.2.2)).2.1
  set S₀ := closeL A (udRadius rsch.level₀) rsch.k0 rsch.level₀.logCols rsch.level₀.logLen w with hS₀def
  have hS₀ : S₀.card ≤ 1 := card_closeL_ud_le A hA rsch.level₀ hfit rsch.k0 w
  let S : Statement := ⟨sh.mReg, 6, by omega, le_rfl, ⟨0, Nat.two_pow_pos _⟩, 0, 0, []⟩
  let D : Option (OpenOut F) → Prop := fun o => ¬ ∃ M ∈ S₀, ∃ op : OpenOut F, o = some op ∧
    (∑ y ∈ Finset.range (2 ^ (rsch.m - 7)), flat M y * op.b y) = op.T
  have hopen : value (fun o => ¬ D o) (opening A hreg [c₁.toClaim, c₂.toClaim]) ≤
      ((7 * [c₁.toClaim, c₂.toClaim].length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
    refine le_trans (value_mono (q := fun o => ∃ M ∈ S₀, EM S rsch M o) (fun o ho => ?_) _)
      ((value_exists_finset_le _ _ _).trans ?_)
    · have ho' : ∃ M ∈ S₀, ∃ op : OpenOut F, o = some op ∧
          (∑ y ∈ Finset.range (2 ^ (rsch.m - 7)), flat M y * op.b y) = op.T :=
        Classical.byContradiction ho
      obtain ⟨M, hM, op, hop, h⟩ := ho'
      refine ⟨M, hM, op, hop, ?_⟩
      rw [← h]
      show ∑ y ∈ Finset.range (2 ^ (sh.mReg - 7)), packedAt rsch M y * op.b y = _
      rw [hm]
      rfl
    · calc ∑ M ∈ S₀, value (EM S rsch M) (opening A hreg [c₁.toClaim, c₂.toClaim])
          ≤ ∑ _M ∈ S₀, ((7 * [c₁.toClaim, c₂.toClaim].length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F :=
            Finset.sum_le_sum fun M hM => value_EM_le A hA S rsch hreg [c₁.toClaim, c₂.toClaim] M (by
              have hc : Model.Close ((1 - rsch.level₀.rate) / 2 - eta) (asTable rsch w)
                  (Model.encode A rsch.level₀.logLen M) := by
                rw [hS₀def] at hM
                simp only [closeL, Finset.mem_filter, Finset.mem_univ, true_and] at hM
                exact hM
              have hn := hdoom M hc
              by_contra hall
              push Not at hall
              exact hn ⟨hall _ (by simp), hall _ (by simp)⟩)
        _ ≤ 1 * (((7 * [c₁.toClaim, c₂.toClaim].length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
            rw [Finset.sum_const, nsmul_eq_mul]
            gcongr <;> exact_mod_cast hS₀
        _ = _ := one_mul _
  unfold regOpening
  refine (value_bind_le (opening A hreg [c₁.toClaim, c₂.toClaim]) _ _ D
    (ENNReal.ofReal (1 / qF + udQueryError rsch.level₀ + ligeritoError rsch.levels))
    fun o hD => ?_).trans ?_
  · rcases o with _ | op
    · show value (fun b => b = true) (Game.ret false) ≤ _
      simp
    · exact ligerito_sound_ud A hA rsch hshape op.T op.b w fun M hM h => hD ⟨M, hM, op, rfl, h⟩
  · have hq : (0 : ℝ) < qF := by unfold qF; positivity
    have hu : (0 : ℝ) ≤ udQueryError rsch.level₀ := by
      unfold udQueryError Level.rate Accounting.eta; positivity
    have hr : 0 ≤ ligeritoError rsch.levels :=
      ligeritoError_nonneg _ (levelsChained_ok hshape.2.2.2)
    have hsplit : regOpenError rsch = ((15 : ℕ) : ℝ) / qF +
        (1 / qF + udQueryError rsch.level₀ + ligeritoError rsch.levels) := by
      unfold regOpenError
      rw [show 7 * Flock.RecVStar.STACKED_CLAIMS + 2 = 16 from rfl]
      push_cast
      ring
    have h15 : ((7 * [c₁.toClaim, c₂.toClaim].length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F =
        ENNReal.ofReal (((15 : ℕ) : ℝ) / qF) := by
      rw [ofReal_nat_div_qF', hA.card_F]
      norm_num
    have h1 : (0 : ℝ) ≤ 1 / qF := by positivity
    have h15' : (0 : ℝ) ≤ ((15 : ℕ) : ℝ) / qF := by positivity
    rw [hsplit, ENNReal.ofReal_add h15' (add_nonneg (add_nonneg h1 hu) hr), ← h15]
    exact add_le_add hopen le_rfl

end FlockC3.Fork
