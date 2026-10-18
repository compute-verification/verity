import Proofs.Flock.Claims.Complete.Replay.LigeritoCons

/-!
# The honest prover's weight is the verifier's residual weight

The honest prover keeps a weight `w` (`LigState.w`); the verifier's final check reads the residual weights `res`
(`resE`), which are the model's `resAt` on the verifier's lists (`gOf_resE`). At every stage the prover's weight is the
model's weight on the lists so far:

* **level 0** (`gv_w0`): the batched basis plus the first out-of-domain claim, `resAt` with no challenges;
* **level 0's challenges** (`fold_init`): folding `resAt` with no levels at challenges `ch` is `resAt` with `ch` as the
  original challenges;
* **a split** (`gv_liftU_preW`): lifting the weight by `u` is `preW`, the model's weight on a split table;
* **a claim on the split table**: an out-of-domain sample adds its term to `preW` (`gv_addOod`), and a level's
  consistency claim, `β·u^b·V` with `V = consW`, adds its `consTerm` (`gv_addCons`): `V(c)` is the model's
  `Σ_j eq(α)_j·X̂_c(ω_{pos_j})` (`embed_consW`, from `fOf_xhats`);
* **the end** (`res_honest`): the residual weights are the prover's last weight plus the last consistency claim.
-/

namespace FlockComplete.Replay

open Flock FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Lg

/-! ## The subspace products -/

/-- `xhats` after `i` doublings: entry `c` is the product of `wh[k]` over the bits `k` of `c`. -/
theorem xhats_go (wh : Array F128) : ∀ i : ℕ,
    ((List.range i).foldl (fun t k => t ++ t.map (· * wh[k]!)) #[1]).size = 2 ^ i ∧
    ∀ c < 2 ^ i, fOf ((List.range i).foldl (fun t k => t ++ t.map (· * wh[k]!)) #[1])[c]! =
      ∏ k : Fin i, if c.testBit k then fOf wh[(k : ℕ)]! else 1
  | 0 => by
      refine ⟨rfl, fun c hc => ?_⟩
      obtain rfl : c = 0 := by simpa using hc
      simp [fOf_one]
  | i + 1 => by
      obtain ⟨hs, hc⟩ := xhats_go wh i
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      set T := (List.range i).foldl (fun t k => t ++ t.map (· * wh[k]!)) #[1]
      refine ⟨by simp [hs, pow_succ]; ring, fun c hc' => ?_⟩
      rw [Fin.prod_univ_castSucc]
      by_cases hlt : c < 2 ^ i
      · rw [getElem!_pos _ c (by simp; omega), Array.getElem_append_left (by omega), ← getElem!_pos T c (by omega),
          hc c hlt]
        simp [Nat.testBit_eq_false_of_lt hlt]
      · obtain ⟨c', rfl⟩ : ∃ c', c = 2 ^ i + c' := ⟨c - 2 ^ i, by omega⟩
        have hc'' : c' < 2 ^ i := by rw [pow_succ] at hc'; omega
        rw [getElem!_pos _ _ (by simp; rw [pow_succ] at hc'; omega), Array.getElem_append_right (by omega)]
        simp only [Array.getElem_map, show 2 ^ i + c' - T.size = c' by omega]
        rw [← getElem!_pos T c' (by omega), fOf_mul, hc c' hc'']
        have h1 : ∀ k : Fin i, (2 ^ i + c').testBit k = c'.testBit k := fun k =>
          Nat.testBit_two_pow_add_gt k.isLt c'
        have h2 : (2 ^ i + c').testBit i = true := by
          rw [Nat.testBit_two_pow_add_eq, Nat.testBit_eq_false_of_lt hc'']; rfl
        simp [h1, h2]

/-- **The prover's subspace products are the model's** `X̂_c(ω_q)`. -/
theorem fOf_xhats (q cols c : ℕ) (hc : c < 2 ^ cols) :
    fOf (xhats (svTable (cols + 1)) q cols)[c]! = execArith.Xhat c cols (execArith.omega q) := by
  unfold xhats Model.Arith.Xhat
  rw [(xhats_go _ cols).2 c hc]
  refine Finset.prod_congr rfl fun k _ => ?_
  rw [whats_get (cols + 1) q cols k (by omega) k.isLt]
  rfl

/-- **The consistency weight** at column `c` is the model's `Σ_j eq(α)_j·X̂_c(ω_{pos_j})`. -/
theorem embed_consW (cols : ℕ) (pos : Array ℕ) (alpha : Array F128) (hpa : pos.size ≤ 2 ^ alpha.size) {c : ℕ}
    (hc : c < 2 ^ cols) :
    FlockLevel3.embed (fOf (consW cols pos alpha)[c]!) =
      ((pos.toList.zipIdx).map fun pj => execArith.embed (Model.eqList (alpha.toList.map fOf) pj.2) *
        execArith.embed (execArith.Xhat c cols (execArith.omega pj.1))).sum := by
  unfold consW
  rw [getElem!_range_map _ hc]
  show FlockLevel3.embed (FlockLevel3.GF128.of _) = _
  rw [Zc.of_sumF, map_sum, zipIdx_map_eq pos.toList 0, Array.length_toList, FlockLevel3.list_sum_range]
  refine Finset.sum_congr rfl fun j hj => ?_
  have hj' := Finset.mem_range.1 hj
  show FlockLevel3.embed (fOf (_ * _)) = _
  rw [fOf_mul, map_mul, eqTable_eqList _ _ (by omega), getElem!_pos (pos.map _) j (by simpa using hj'),
    Array.getElem_map, ← getElem!_pos pos j hj', fOf_xhats _ _ _ hc, List.getD_eq_getElem _ _ (by simpa using hj'),
    Array.getElem_toList, ← getElem!_pos pos j hj']
  rfl

/-- A consistency claim made at the current stage reads the consistency weight times its coin. -/
theorem consTerm_consOf (levels : List (List 𝕂)) (ext : ℕ) (pos : Array ℕ) (alpha : Array F128) (b : F128)
    (hpa : pos.size ≤ 2 ^ alpha.size) {y : ℕ} (hy : y < 2 ^ ext) :
    FlockSoundness.consTerm execArith levels ext (consOf ⟨ext, pos, alpha, b, levels.length⟩) y =
      FlockLevel3.embed (fOf (consW ext pos alpha)[y]!) * FlockLevel3.embed (fOf b) := by
  rw [FlockSoundness.consTerm_new execArith levels ext _ rfl y, embed_consW ext pos alpha hpa hy, mul_comm]
  rfl

/-! ## Level 0 -/

section Fold

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

theorem mleAt_bits [CharP K 2] (N : ℕ) (c : ℕ → K) {y : ℕ} (hy : y < 2 ^ N) :
    mleAt N c (Model.bitsK N y) = c y := by
  unfold mleAt
  rw [Finset.sum_eq_single_of_mem y (Finset.mem_range.2 hy)]
  · rw [eqList_bitsK _ y y hy hy]; simp
  · intro w hw hwy
    rw [eqList_bitsK _ y w hy (Finset.mem_range.1 hw)]; simp [hwy]

/-- **Folding level 0's challenges**: with no levels and no consistency claims, folding `resAt` at `ch` puts `ch`
among the original challenges. -/
theorem fold_init [CharP K 2] (A : Model.Arith F K) (n : ℕ) (b : ℕ → F) (ch : List K) (oods : List (Model.OodCtx F))
    (m : ℕ) (hm : ch.length ≤ m) (hood : ∀ o ∈ oods, o.split = none ∧ o.z.length = m) (y : ℕ) :
    foldLoL ch (resAt A n b [] [] oods [] m) y = resAt A n b ch [] oods [] (m - ch.length) y := by
  unfold resAt
  simp only [List.map_nil, List.sum_nil, add_zero]
  rw [foldLoL_add, foldLoL_list_sum]
  have hc0 : Model.cffFrom A ([] : List (List K)) 0 = 1 := by simp [Model.cffFrom]
  simp only [hc0, mul_one]
  congr 1
  · have h := foldLoL_mle_bits n (fun w => A.embed (b w)) [] ch m hm y
    simp only [mleAt] at h
    simp only [Model.bHat, origOf, List.map_nil, List.flatten_nil, List.append_nil, List.nil_append] at h ⊢
    exact h
  · refine congrArg List.sum (List.map_congr_left fun o ho => ?_)
    obtain ⟨hs, hz⟩ := hood o ho
    have hfx : ∀ init : List K, FlockSoundness.oodFixed init [] o = init := fun init => by
      simp [FlockSoundness.oodFixed, hs, origOf]
    have hsc : oodScale A ([] : List (List K)) o = 1 := by simp [oodScale, hs, Model.cffFrom]
    rw [show oodTerm A [] [] m o = fun y => A.embed o.β * oodScale A [] o *
        mleAt o.z.length (Model.eqList (o.z.map A.embed)) (FlockSoundness.oodFixed [] [] o ++ Model.bitsK m y) from
        funext fun y => oodTerm_eq_mle A [] [] m o y (by rw [hfx]; simp [hz]),
      oodTerm_eq_mle A ch [] (m - ch.length) o y (by rw [hfx, hz]; omega), foldLoL_mul_left, hfx, hfx,
      foldLoL_mle_bits _ _ [] ch m hm, List.nil_append]

/-- Level 0's weight with no challenges yet: the batched basis plus the first out-of-domain claim. -/
theorem resAt_w0 [CharP K 2] (A : Model.Arith F K) (n : ℕ) (b : ℕ → F) (o : Model.OodCtx F) (ho : o.split = none)
    (hz : o.z.length = n) {y : ℕ} (hy : y < 2 ^ n) :
    resAt A n b [] [] [o] [] n y = A.embed (b y) + A.embed o.β * Model.eqList (o.z.map A.embed) y := by
  unfold resAt
  have hc0 : Model.cffFrom A ([] : List (List K)) 0 = 1 := by simp [Model.cffFrom]
  have hfx : FlockSoundness.oodFixed ([] : List K) [] o = [] := by simp [FlockSoundness.oodFixed, ho, origOf]
  have hsc : oodScale A ([] : List (List K)) o = 1 := by simp [oodScale, ho, Model.cffFrom]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, hc0, mul_one]
  have hm : mleAt o.z.length (Model.eqList (o.z.map A.embed)) (Model.bitsK n y) =
      Model.eqList (o.z.map A.embed) y := by
    rw [show n = o.z.length from hz.symm]
    exact mleAt_bits _ _ (by rw [hz]; exact hy)
  rw [oodTerm_eq_mle A [] [] n o y (by rw [hfx]; simp [hz]), hfx, hsc, List.nil_append, hm, mul_one]
  congr 1
  have h := mleAt_bits n (fun w => A.embed (b w)) hy
  simp only [mleAt] at h
  simp only [Model.bHat, origOf, List.map_nil, List.flatten_nil, List.append_nil, List.nil_append]
  exact h

end Fold

/-- The batched basis's entries as model elements. -/
noncomputable def bOf (b : Array F128) : ℕ → 𝔽 := fun w => fOf b[w]!

/-- **Level 0's weight**: the batched basis plus the first out-of-domain claim's `β·eq(z, ·)`. -/
theorem gv_w0 (b z : Array F128) (b0 : F128) (n : ℕ) (hb : b.size = 2 ^ n) (hz : z.size = n) {y : ℕ}
    (hy : y < 2 ^ n) :
    gv (addScaled (embedF b) (embedF (eqTable z)) b0) y =
      resAt execArith n (bOf b) [] [] [oodOf ⟨z, b0, none⟩] [] n y := by
  rw [resAt_w0 execArith n (bOf b) _ rfl (by simp [oodOf, hz]) hy,
    gv_addScaled _ _ _ (by rw [embedF_size, hb]; exact hy),
    gv_embedF _ (by rw [hb]; exact hy), gv_embedF _ (by rw [FlockLevel3.eqTable_size, hz]; exact hy),
    eqTable_eqList _ _ (by rw [hz]; exact hy)]
  simp only [oodOf]
  rw [← map_eqList execArith, mul_comm]
  rfl

/-! ## A split table's weight -/

/-- **A split** lifts the weight by `u`: on a table whose weight is `resAt`, before its out-of-domain claims, the
lifted weight is `preW`. -/
theorem gv_liftU_preW (n : ℕ) (b : ℕ → 𝔽) (initial : List 𝕂) (levels : List (List 𝕂))
    (oods : List (Model.OodCtx 𝔽)) (cons : List (Model.ConsCtx 𝔽)) (ext : ℕ) (w : Array F256)
    (hw : w.size = 2 ^ ext) (hold : ∀ o ∈ oods, o.split ≠ some levels.length)
    (hres : ∀ x < 2 ^ ext, gv w x = resAt execArith n b initial levels oods cons ext x) {y : ℕ}
    (hy : y < 2 ^ (ext + 1)) :
    gv (liftU w) y = preW execArith n b initial levels oods cons ext y := by
  have hy2 : y / 2 < 2 ^ ext := by rw [pow_succ] at hy; omega
  rw [gv_liftU w (by rw [hw]; rw [pow_succ] at hy; omega), hres _ hy2]
  unfold preW
  rw [List.filter_eq_self.2 fun o ho => by simpa using hold o ho,
    List.filter_eq_nil_iff.2 fun o ho => by simpa using hold o ho]
  simp only [List.map_nil, List.sum_nil, add_zero]
  rfl

/-- **An out-of-domain claim** on the split table adds `β·eq(z, ·)` to `preW`. -/
theorem gv_addOod (n : ℕ) (b : ℕ → 𝔽) (initial : List 𝕂) (levels : List (List 𝕂))
    (oods : List (Model.OodCtx 𝔽)) (cons : List (Model.ConsCtx 𝔽)) (ext : ℕ) (w : Array F256) (z : Array F128)
    (b0 : F128) (hw : w.size = 2 ^ (ext + 1)) (hz : z.size = ext + 1)
    (hpre : ∀ y < 2 ^ (ext + 1), gv w y = preW execArith n b initial levels oods cons ext y) {y : ℕ}
    (hy : y < 2 ^ (ext + 1)) :
    gv (addScaled w (embedF (eqTable z)) b0) y =
      preW execArith n b initial levels (oods ++ [oodOf ⟨z, b0, some levels.length⟩]) cons ext y := by
  rw [gv_addScaled _ _ _ (by rw [hw]; exact hy), hpre y hy,
    gv_embedF _ (by rw [FlockLevel3.eqTable_size, hz]; exact hy), eqTable_eqList _ _ (by rw [hz]; exact hy)]
  unfold preW
  rw [List.filter_append, List.filter_append, List.filter_singleton, List.filter_singleton]
  simp only [oodOf, ne_eq, not_true_eq_false, decide_false, Bool.cond_false, List.append_nil, decide_true,
    Bool.cond_true, List.map_append, List.map_cons, List.map_nil, List.sum_append, List.sum_cons, List.sum_nil,
    add_zero]
  rw [← map_eqList execArith, mul_comm (FlockLevel3.embed _), add_assoc]
  rfl

/-- **A level's consistency claim** on the split table adds `β·u^b·V` to `preW`: its `consTerm`, read at the split
coordinate's other side. -/
theorem gv_addCons (n : ℕ) (b : ℕ → 𝔽) (initial : List 𝕂) (levels : List (List 𝕂))
    (oods : List (Model.OodCtx 𝔽)) (cons : List (Model.ConsCtx 𝔽)) (ext : ℕ) (w : Array F256) (pos : Array ℕ)
    (alpha : Array F128) (bq : F128) (hw : w.size = 2 ^ (ext + 1)) (hpa : pos.size ≤ 2 ^ alpha.size)
    (hpre : ∀ y < 2 ^ (ext + 1), gv w y = preW execArith n b initial levels oods cons ext y) {y : ℕ}
    (hy : y < 2 ^ (ext + 1)) :
    gv (addScaled w (liftU (embedF (consW ext pos alpha))) bq) y =
      preW execArith n b initial levels oods (cons ++ [consOf ⟨ext, pos, alpha, bq, levels.length⟩]) ext y := by
  have hy2 : y / 2 < 2 ^ ext := by rw [pow_succ] at hy; omega
  rw [gv_addScaled _ _ _ (by rw [hw]; exact hy), hpre y hy,
    gv_liftU _ (by rw [embedF_size, consW_size]; rw [pow_succ] at hy; omega),
    gv_embedF _ (by rw [consW_size]; exact hy2)]
  unfold preW
  rw [resAt_cons_append]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  rw [consTerm_consOf levels ext pos alpha bq hpa hy2]
  have hu : execArith.u = FlockLevel3.u := rfl
  rw [hu]
  ring

/-! ## The residual weights -/

theorem gOf_map_mul (a : Array F256) (c : F256) (y : ℕ) : gOf (a.map (· * c))[y]! = gOf a[y]! * gOf c := by
  by_cases hy : y < a.size
  · rw [getElem!_pos _ y (by simpa using hy), Array.getElem_map, gOf_mul, ← getElem!_pos a y hy]
  · rw [getElem!_neg _ y (by simpa using hy), getElem!_neg a y hy, show (default : F256) = 0 from rfl, gOf_zero,
      zero_mul]

/-- **The residual weights are the model's** `resAt` on the verifier's lists, when the basis at the original point is
the model's `bHat`, the levels have four challenges, and every context has its dimensions (as `finalProg` checks). -/
theorem gOf_resE (basisAt : Array F256 → ℕ → Array F256) (n : ℕ) (initCh : Array F256)
    (levels : Array (Array F256)) (ext : ℕ) (st : LState) (b : ℕ → 𝔽) (hl : ∀ l ∈ levels.toList, l.size = 4)
    (horig : (origE initCh levels).size + ext = n)
    (hpos : ∀ cs ∈ st.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size)
    (hood : ∀ o ∈ st.oods.toList, o.z.size = (Refine.oodFixed levels (origE initCh levels) o).1.size + ext)
    (hcons : ∀ cs ∈ st.conss.toList, (Refine.consFixed levels cs.start).size + ext = cs.cols)
    (hbasis : ∀ y < 2 ^ ext, gOf (basisAt (origE initCh levels) ext)[y]! =
      Model.bHat execArith n b ((origE initCh levels).toList.map gOf ++ Model.bitsK ext y)) {y : ℕ}
    (hy : y < 2 ^ ext) :
    gOf (resE basisAt n initCh levels ext st)[y]! = resAt execArith n b (initCh.toList.map gOf) (levelsOf levels)
      (st.oods.toList.map oodOf) (st.conss.toList.map consOf) ext y := by
  unfold resE resAt
  rw [consFold_get _ _ _ hy, gOf_foldl_add, oodFold_get _ _ hy, gOf_foldl_add, gOf_map_mul, hbasis y hy,
    prodFromE_eq levels hl, origE_toList]
  simp only [FlockSoundness.origOf]
  refine congrArg₂ (· + ·) (congrArg₂ (· + ·) rfl ?_) ?_
  · rw [List.map_map]
    refine congrArg List.sum (List.map_congr_left fun o ho => ?_)
    have hz := hood o ho
    cases hsp : o.split with
    | none =>
        have hfs : Refine.oodFixed levels (origE initCh levels) o = (origE initCh levels, prodFromE levels 0) := by
          simp [Refine.oodFixed, hsp]
        rw [hfs] at hz
        simp only [Function.comp, hfs]
        rw [gOf_oodTerm ext _ o y hz, prodFromE_eq levels hl]
        simp only [FlockSoundness.oodTerm, FlockSoundness.oodScale, FlockSoundness.oodFixed, FlockSoundness.origOf,
          oodOf, hsp, origE_toList]
    | some a =>
        have hfs : Refine.oodFixed levels (origE initCh levels) o = ((levels.extract a levels.size).foldl
            (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size) #[], prodFromE levels (a + 1)) := by
          simp [Refine.oodFixed, hsp]
        rw [hfs] at hz
        simp only [Function.comp, hfs]
        rw [gOf_oodTerm ext _ o y hz, prodFromE_eq levels hl, oodFixed_some levels hl a]
        simp only [FlockSoundness.oodTerm, FlockSoundness.oodScale, FlockSoundness.oodFixed, oodOf, hsp]
  · rw [List.map_map]
    refine congrArg List.sum (List.map_congr_left fun cs hcs' => ?_)
    have hc := hcons cs hcs'
    have hle := consFixed_size_le initCh levels cs.start
    simp only [Function.comp]
    rw [gOf_consTerm (n + 1) ext levels hl cs y hy hc (by omega) (hpos cs hcs')]
    simp only [FlockSoundness.consTerm, FlockSoundness.consFixed, consOf]

/-- **The final residual weights** are the prover's last weight plus the last consistency claim's `β·V`, made at the
current stage. -/
theorem res_honest (basisAt : Array F256 → ℕ → Array F256) (n : ℕ) (initCh : Array F256)
    (levels : Array (Array F256)) (ext : ℕ) (st : LState) (b : ℕ → 𝔽) (hl : ∀ l ∈ levels.toList, l.size = 4)
    (horig : (origE initCh levels).size + ext = n)
    (hpos : ∀ cs ∈ st.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size)
    (hood : ∀ o ∈ st.oods.toList, o.z.size = (Refine.oodFixed levels (origE initCh levels) o).1.size + ext)
    (hcons : ∀ cs ∈ st.conss.toList, (Refine.consFixed levels cs.start).size + ext = cs.cols)
    (hbasis : ∀ y < 2 ^ ext, gOf (basisAt (origE initCh levels) ext)[y]! =
      Model.bHat execArith n b ((origE initCh levels).toList.map gOf ++ Model.bitsK ext y))
    (cons : Array ConsCtx) (pos : Array ℕ) (alpha : Array F128) (bq : F128)
    (hst : st.conss = cons.push ⟨ext, pos, alpha, bq, levels.size⟩) (hpa : pos.size ≤ 2 ^ alpha.size)
    (w : Array F256) (hw : ∀ y < 2 ^ ext, gv w y = resAt execArith n b (initCh.toList.map gOf) (levelsOf levels)
      (st.oods.toList.map oodOf) (cons.toList.map consOf) ext y) {y : ℕ} (hy : y < 2 ^ ext) :
    gv (resE basisAt n initCh levels ext st) y =
      gv w y + gv (embedF (consW ext pos alpha)) y * FlockLevel3.embed (fOf bq) := by
  rw [gv, gOf_resE basisAt n initCh levels ext st b hl horig hpos hood hcons hbasis hy, hst, Array.toList_push,
    List.map_append, resAt_cons_append, hw y hy, gv_embedF _ (by rw [consW_size]; exact hy)]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  have hlen : levels.size = (levelsOf levels).length := by simp [levelsOf]
  rw [hlen, consTerm_consOf (levelsOf levels) ext pos alpha bq hpa hy]

end Lg

end FlockComplete.Replay
