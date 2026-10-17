import Proofs.Flock.Claims.Complete.Replay.LigeritoWeight
import Proofs.Flock.Soundness.LigeritoRes

/-!
# The honest Ligerito's weight is the verifier's residual weight, stage by stage

The honest prover's weight `w` (`LigState.w`) and the verifier's final-check weights (`resE`) are one function read at two
times: the model's `resAt` at the challenges and claims so far (`FlockSoundness.resAt`), or its `preW` on a split table
before the level's lanes are folded. `WAt w … ext` says the prover's `2^ext` entries are `resAt`'s, `WPre` that its
`2^(ext+1)` entries are `preW`'s. Every step of the prover keeps one of them:

* **the start** (`WAt.start`): the basis `b` is `bHat` on the cube;
* **level 0's out-of-domain sample** (`WAt.oodHead`): `eq(z, ·)` is its `oodTerm` (`gv_eqTable`);
* **a level-0 challenge** (`WAt.chal0`): folding `resAt` appends the challenge to the original point;
* **a split** (`WAt.lift`): the lifted weight is `preW`'s first part (`gv_liftU`);
* **an out-of-domain sample on the split table** (`WPre.ood`): `preW`'s second part gains its `eq(z, ·)`;
* **a level's consistency claim** (`WPre.cons`): the lifted `consW` is the new context's `consTerm` (`consTerm_new`,
  from `fOf_xhats`: the prover's `X̂_c(ω_q)` are the model's);
* **a level's four challenges** (`WPre.fold`): folding `preW` is the next stage's `resAt` (`foldLoL_preW`);
* **the end** (`WAt.final`, with `gOf_resE`): the verifier's `resE` is `resAt` with the last consistency context, so
  it is the prover's weight plus that context's `consW`, scaled by its coin: the residual hypothesis of `lhsE_honest`,
  which with it gives the final identity at the verifier's own weights (`lhsE_resE`).
-/

namespace FlockComplete.Replay

open Flock FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Lg

/-! ## The executable's tables as the model's -/

/-- `x ↦ f x` mapped through a ring hom is `eq` at the mapped point. -/
theorem eqList_map {R S : Type} [CommRing R] [CommRing S] (f : R →+* S) (l : List R) (j : ℕ) :
    Model.eqList (l.map f) j = f (Model.eqList l j) := by
  rw [eqList_eq, eqList_eq, map_prod, List.length_map]
  refine Finset.prod_congr rfl fun i hi => ?_
  have hi' := Finset.mem_range.1 hi
  rw [List.getD_eq_getElem _ _ (by simpa using hi'), List.getD_eq_getElem _ _ hi', List.getElem_map]
  split_ifs <;> simp

/-- `embedF (eqTable z)` reads as the model's `eq(z, ·)`. -/
theorem gv_eqTable (z : Array F128) {y : ℕ} (hy : y < 2 ^ z.size) :
    gv (embedF (eqTable z)) y = Model.eqList ((z.toList.map fOf).map execArith.embed) y := by
  rw [gv_embedF _ (by rw [FlockLevel3.eqTable_size]; exact hy), eqTable_eqList z y hy, eqList_map]
  rfl

theorem xhats_size (sv : Array F128) (q cols : ℕ) : (xhats sv q cols).size = 2 ^ cols :=
  (xhats_go _ cols).1

/-- **The consistency weight** `consW` is the model's `Σ_j eq(α)[j]·X̂_c(ω_{pos_j})`. -/
theorem consW_get (cols : ℕ) (pos : Array ℕ) (alpha : Array F128) (hpos : pos.size ≤ 2 ^ alpha.size) {c : ℕ}
    (hc : c < 2 ^ cols) :
    fOf (consW cols pos alpha)[c]! = ((pos.toList.zipIdx).map fun pj =>
      Model.eqList (alpha.toList.map fOf) pj.2 * execArith.Xhat c cols (execArith.omega pj.1)).sum := by
  unfold consW
  rw [getElem!_range_map _ hc]
  show FlockLevel3.GF128.of _ = _
  rw [Zc.of_sumF, zipIdx_map_eq _ 0, Array.length_toList, FlockLevel3.list_sum_range]
  refine Finset.sum_congr rfl fun j hj => ?_
  have hj' := Finset.mem_range.1 hj
  show fOf _ = _
  rw [fOf_mul, eqTable_eqList _ _ (by omega), getElem!_pos (pos.map _) j (by simpa using hj'), Array.getElem_map,
    fOf_xhats _ _ _ hc, List.getD_eq_getElem _ _ (by simpa using hj'), Array.getElem_toList]

/-! ## The invariants -/

variable (bF : ℕ → 𝔽) (n : ℕ)

/-- **The prover's weight is the verifier's residual weight** at a stage: `2^ext` entries, each the model's `resAt` at
the challenges `initial` and `levels` and the claims `oods` and `cons` so far. -/
structure WAt (w : Array F256) (initial : List 𝕂) (levels : List (List 𝕂)) (oods : List (Model.OodCtx 𝔽))
    (cons : List (Model.ConsCtx 𝔽)) (ext : ℕ) : Prop where
  size : w.size = 2 ^ ext
  get : ∀ y < 2 ^ ext, gv w y = resAt execArith n bF initial levels oods cons ext y

/-- **The prover's weight on a split table** before the level's lanes are folded: `2^(ext+1)` entries, each the model's
`preW`. -/
structure WPre (w : Array F256) (initial : List 𝕂) (levels : List (List 𝕂)) (oods : List (Model.OodCtx 𝔽))
    (cons : List (Model.ConsCtx 𝔽)) (ext : ℕ) : Prop where
  size : w.size = 2 ^ (ext + 1)
  get : ∀ y < 2 ^ (ext + 1), gv w y = preW execArith n bF initial levels oods cons ext y

variable {bF n}

/-! ## Level 0 -/

theorem bHat_eq (b : ℕ → 𝔽) (p : List 𝕂) :
    Model.bHat execArith n b p = mleAt n (fun w => execArith.embed (b w)) p := rfl

/-- **The start**: the batched basis `b` is `bHat` on the cube, with no challenge and no claim. -/
theorem WAt.start (b : Array F128) (hb : b.size = 2 ^ n) :
    WAt (fun w => fOf b[w]!) n (embedF b) [] [] [] [] n := by
  refine ⟨by rw [embedF_size, hb], fun y hy => ?_⟩
  rw [gv_embedF _ (by omega)]
  simp only [resAt, origOf, List.map_nil, List.flatten_nil, List.append_nil, List.nil_append, List.sum_nil,
    add_zero, Model.cffFrom, List.drop_nil, List.prod_nil, mul_one]
  rw [bHat_eq, mleAt_bits n _ hy]
  rfl

/-- Before level 0's lanes split, every claim is level 0's out-of-domain sample, on the original point. -/
def Orig (initial : List 𝕂) (ext : ℕ) (oods : List (Model.OodCtx 𝔽)) : Prop :=
  ∀ o ∈ oods, o.split = none ∧ o.z.length = initial.length + ext

theorem resAt_orig {initial : List 𝕂} {oods : List (Model.OodCtx 𝔽)} {ext : ℕ} (h : Orig initial ext oods) (y : ℕ) :
    resAt execArith n bF initial [] oods [] ext y =
      mleAt n (fun w => execArith.embed (bF w)) (initial ++ Model.bitsK ext y) +
        (oods.map fun o => execArith.embed o.β *
          mleAt o.z.length (Model.eqList (o.z.map execArith.embed)) (initial ++ Model.bitsK ext y)).sum := by
  unfold resAt
  have hc : Model.cffFrom execArith ([] : List (List 𝕂)) 0 = 1 := by simp [Model.cffFrom]
  have ho : origOf initial ([] : List (List 𝕂)) = initial := by simp [origOf]
  rw [hc, mul_one, ho, List.map_nil, List.sum_nil, add_zero]
  refine congrArg₂ (· + ·) (bHat_eq bF _) ?_
  refine congrArg List.sum (List.map_congr_left fun o hmem => ?_)
  obtain ⟨hs, hz⟩ := h o hmem
  have hf : FlockSoundness.oodFixed initial ([] : List (List 𝕂)) o = initial := by
    simp [FlockSoundness.oodFixed, hs, ho]
  rw [oodTerm_eq_mle execArith initial [] ext o y (by rw [hf, hz]), hf]
  simp [oodScale, hs, Model.cffFrom]

/-- **Level 0's out-of-domain sample** `z` on all `ext` original coordinates, with coin `β`. -/
theorem WAt.oodHead {w : Array F256} {initial : List 𝕂} {oods : List (Model.OodCtx 𝔽)} {ext : ℕ}
    (h : WAt bF n w initial [] oods [] ext) (ho : Orig initial ext oods) (hi : initial = []) (z : Array F128)
    (hz : z.size = ext) (β : F128) :
    WAt bF n (addScaled w (embedF (eqTable z)) β) initial [] (oods ++ [⟨z.toList.map fOf, fOf β, none⟩]) [] ext ∧
      Orig initial ext (oods ++ [⟨z.toList.map fOf, fOf β, none⟩]) := by
  subst hi
  have ho' : Orig [] ext (oods ++ [⟨z.toList.map fOf, fOf β, none⟩]) := by
    intro o hmem
    rcases List.mem_append.1 hmem with hm | hm
    · exact ho o hm
    · rw [List.mem_singleton.1 hm]
      simp [hz]
  refine ⟨⟨by rw [addScaled_size, h.size], fun y hy => ?_⟩, ho'⟩
  rw [gv_addScaled _ _ _ (by rw [h.size]; exact hy), h.get y hy, gv_eqTable _ (by rw [hz]; exact hy),
    resAt_orig ho, resAt_orig ho', List.map_append, List.sum_append, List.map_singleton, List.sum_singleton,
    List.length_map, Array.length_toList, hz, List.nil_append, mleAt_bits ext _ hy]
  rw [show execArith.embed = FlockLevel3.embed from rfl]
  ring

/-- **A level-0 challenge** `r`: folding appends it to the original point. -/
theorem WAt.chal0 {w : Array F256} {initial : List 𝕂} {oods : List (Model.OodCtx 𝔽)} {ext : ℕ}
    (h : WAt bF n w initial [] oods [] (ext + 1)) (ho : Orig initial (ext + 1) oods) (r : F256) :
    WAt bF n (foldK r w) (initial ++ [gOf r]) [] oods [] ext ∧ Orig (initial ++ [gOf r]) ext oods := by
  have ho' : Orig (initial ++ [gOf r]) ext oods := fun o hmem => by
    obtain ⟨hs, hz⟩ := ho o hmem
    exact ⟨hs, by rw [hz, List.length_append, List.length_singleton]; ring⟩
  refine ⟨⟨by rw [foldK_size, h.size, pow_succ]; omega, fun y hy => ?_⟩, ho'⟩
  have hy2 : 2 * y + 1 < 2 ^ (ext + 1) := by rw [pow_succ]; omega
  rw [gv_foldK _ _ (by rw [h.size, pow_succ]; omega)]
  have e : foldLo (gOf r) (gv w) y =
      foldLoL [gOf r] (fun x => resAt execArith n bF initial [] oods [] (ext + 1) x) y := by
    simp only [foldLoL, foldLo]
    rw [h.get _ (by omega), h.get _ hy2]
  rw [e, resAt_orig ho', foldLoL_congr _ _ (fun x => mleAt n (fun w => execArith.embed (bF w))
      (initial ++ Model.bitsK (ext + 1) x) + (oods.map fun o => execArith.embed o.β *
        mleAt o.z.length (Model.eqList (o.z.map execArith.embed)) (initial ++ Model.bitsK (ext + 1) x)).sum) y
      fun x _ => resAt_orig ho _]
  rw [foldLoL_add, foldLoL_mle_bits _ _ _ _ _ (by simp), foldLoL_list_sum]
  simp only [foldLoL_mul_left, foldLoL_mle_bits _ _ _ [gOf r] (ext + 1) (by simp), List.length_singleton,
    Nat.add_sub_cancel]

/-! ## A recursive level -/

/-- **A split**: the weight lifted by `u` onto the split table is `preW`, with no claim on that table yet. -/
theorem WAt.lift {w : Array F256} {initial : List 𝕂} {levels : List (List 𝕂)} {oods : List (Model.OodCtx 𝔽)}
    {cons : List (Model.ConsCtx 𝔽)} {ext : ℕ} (h : WAt bF n w initial levels oods cons ext)
    (ho : ∀ o ∈ oods, o.split ≠ some levels.length) :
    WPre bF n (liftU w) initial levels oods cons ext := by
  refine ⟨by rw [(liftU_spec w).1, h.size, pow_succ]; ring, fun y hy => ?_⟩
  have hy' : y / 2 < 2 ^ ext := by rw [pow_succ] at hy; omega
  rw [gv_liftU _ (by rw [h.size]; omega), h.get _ hy', preW, List.filter_eq_self.2 (fun o hm => by simpa using ho o hm),
    List.filter_eq_nil_iff.2 (fun o hm => by simpa using ho o hm), List.map_nil, List.sum_nil, add_zero]
  rfl

/-- **An out-of-domain sample on the split table** `g` at `z` (`ext + 1` coordinates), with coin `β`. -/
theorem WPre.ood {w : Array F256} {initial : List 𝕂} {levels : List (List 𝕂)} {oods : List (Model.OodCtx 𝔽)}
    {cons : List (Model.ConsCtx 𝔽)} {ext : ℕ} (h : WPre bF n w initial levels oods cons ext) (z : Array F128)
    (hz : z.size = ext + 1) (β : F128) :
    WPre bF n (addScaled w (embedF (eqTable z)) β) initial levels
      (oods ++ [⟨z.toList.map fOf, fOf β, some levels.length⟩]) cons ext := by
  refine ⟨by rw [addScaled_size, h.size], fun y hy => ?_⟩
  rw [gv_addScaled _ _ _ (by rw [h.size]; exact hy), h.get y hy, gv_eqTable _ (by rw [hz]; exact hy), preW, preW,
    List.filter_append, List.filter_append]
  simp only [ne_eq, decide_not, List.filter_cons, List.filter_nil, decide_true, Bool.not_true, Bool.false_eq_true,
    ↓reduceIte, List.append_nil, List.map_append, List.sum_append, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, add_zero]
  rw [show execArith.embed = FlockLevel3.embed from rfl]
  ring

/-- **A level's consistency claim** at positions `pos` batched by `alpha`, with coin `β`: the context recorded at this
level (`start = levels.length`) on its `ext` columns, its weight `consW` lifted onto the split table. -/
theorem WPre.cons {w : Array F256} {initial : List 𝕂} {levels : List (List 𝕂)} {oods : List (Model.OodCtx 𝔽)}
    {cons : List (Model.ConsCtx 𝔽)} {ext : ℕ} (h : WPre bF n w initial levels oods cons ext) (pos : Array ℕ)
    (alpha : Array F128) (hpos : pos.size ≤ 2 ^ alpha.size) (β : F128) {start : ℕ} (hs : start = levels.length) :
    WPre bF n (addScaled w (liftU (embedF (consW ext pos alpha))) β) initial levels oods
      (cons ++ [consOf ⟨ext, pos, alpha, β, start⟩]) ext := by
  refine ⟨by rw [addScaled_size, h.size], fun y hy => ?_⟩
  have hy' : y / 2 < 2 ^ ext := by rw [pow_succ] at hy; omega
  rw [gv_addScaled _ _ _ (by rw [h.size]; exact hy), h.get y hy,
    gv_liftU _ (by rw [embedF_size, consW_size]; omega), gv_embedF _ (by rw [consW_size]; exact hy'),
    consW_get _ _ _ hpos hy', preW, preW, resAt_cons_append, List.map_singleton, List.sum_singleton,
    consTerm_new execArith levels ext _ hs]
  simp only [consOf, map_list_sum, List.map_map, Function.comp_def, map_mul]
  have hu : execArith.u = FlockLevel3.u := rfl
  have he : execArith.embed = FlockLevel3.embed := rfl
  rw [hu, he]
  ring

/-- **A level's four challenges** `ch`: folding the split table's weight by them is the next stage's `resAt`. -/
theorem WPre.fold {w : Array F256} {initial : List 𝕂} {levels : List (List 𝕂)} {oods : List (Model.OodCtx 𝔽)}
    {cons : List (Model.ConsCtx 𝔽)} {ext : ℕ} (h : WPre bF n w initial levels oods cons (ext + 3))
    (hok : PreOK initial levels oods cons (ext + 3)) (ch : Array F256) (hch : ch.size = 4) :
    WAt bF n (foldKs ch.toList w) initial (levels ++ [ch.toList.map gOf]) oods cons ext := by
  have hd : w.size / 2 ^ ch.toList.length = 2 ^ ext := by
    rw [h.size, Array.length_toList, hch, show ext + 3 + 1 = 4 + ext by ring, pow_add,
      Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]
  refine ⟨by rw [foldKs_size, hd], fun y hy => ?_⟩
  rw [gv_foldKs _ _ _ (by rw [hd]; exact hy)]
  obtain ⟨c, cs, hcs, hlen⟩ : ∃ c cs, ch.toList = c :: cs ∧ cs.length = 3 := by
    rcases hl : ch.toList with _ | ⟨c, cs⟩
    · have := congrArg List.length hl
      simp [hch] at this
    · have := congrArg List.length hl
      simp only [Array.length_toList, hch, List.length_cons] at this
      exact ⟨c, cs, rfl, by omega⟩
  rw [hcs, List.map_cons, foldLoL_congr _ _ (preW execArith n bF initial levels oods cons (ext + 3)) y
      fun x hx => h.get _ ?_]
  · exact foldLoL_preW execArith n bF initial levels oods cons (ext + 3) _ _ ext (by simp [hlen]; omega) hok y hy
  · simp only [List.length_cons, List.length_map, hlen] at hx ⊢
    have h4 : 2 ^ (ext + 3 + 1) = 2 ^ (3 + 1) * 2 ^ ext := by rw [← pow_add]; ring_nf
    rw [h4]
    calc x + 2 ^ (3 + 1) * y < 2 ^ (3 + 1) + 2 ^ (3 + 1) * y := Nat.add_lt_add_right hx _
      _ = 2 ^ (3 + 1) * (y + 1) := by ring
      _ ≤ 2 ^ (3 + 1) * 2 ^ ext := Nat.mul_le_mul_left _ hy

/-! ## The end -/

/-- **The residual weights at the end**: when the verifier's weights are `resAt` with a last consistency context
recorded at the last level, they are the prover's weight plus that context's `consW`, scaled by its coin. -/
theorem WAt.final {w : Array F256} {initial : List 𝕂} {levels : List (List 𝕂)} {oods : List (Model.OodCtx 𝔽)}
    {cons : List (Model.ConsCtx 𝔽)} {ext : ℕ} (h : WAt bF n w initial levels oods cons ext) (pos : Array ℕ)
    (alpha : Array F128) (hpos : pos.size ≤ 2 ^ alpha.size) (β : F128) {start : ℕ} (hs : start = levels.length)
    {res : Array F256} (hres : ∀ y < 2 ^ ext, gOf res[y]! =
      resAt execArith n bF initial levels oods (cons ++ [consOf ⟨ext, pos, alpha, β, start⟩]) ext y) :
    ∀ y < 2 ^ ext, gv res y = gv w y + gv (embedF (consW ext pos alpha)) y * FlockLevel3.embed (fOf β) := by
  intro y hy
  rw [gv, hres y hy, resAt_cons_append, List.map_singleton, List.sum_singleton, consTerm_new execArith levels ext _ hs,
    h.get y hy, gv_embedF _ (by rw [consW_size]; exact hy), consW_get _ _ _ hpos hy]
  simp only [consOf, map_list_sum, List.map_map, Function.comp_def, map_mul]
  have he : execArith.embed = FlockLevel3.embed := rfl
  rw [he]
  ring

/-- **The final identity at the verifier's own weights**: on the last folded table `f` with the claim `s = Σ f·w`, the
prover's weight `w` the model's `resAt` (`WAt`), and the last consistency claim `val = Σ f·consW` with coin `β`
recorded last at the last level, the final check's sum over `splitK f` against the verifier's `resE` is `s + β·val`. -/
theorem lhsE_resE {basisAt : Array F256 → ℕ → Array F256} {initCh : Array F256} {levels : Array (Array F256)}
    {ext : ℕ} {st : LState} {oods : List (Model.OodCtx 𝔽)} {cons : List (Model.ConsCtx 𝔽)} {f w : Array F256}
    (hl : ∀ l ∈ levels.toList, l.size = 4)
    (horig : (origE initCh levels).size + ext = n)
    (hpos : ∀ cs ∈ st.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size)
    (hbasis : ∀ y < 2 ^ ext, gOf (basisAt (origE initCh levels) ext)[y]! =
      Model.bHat execArith n bF ((origE initCh levels).toList.map gOf ++ Model.bitsK ext y))
    (hood : ∀ o ∈ st.oods.toList, o.z.size = (Refine.oodFixed levels (origE initCh levels) o).1.size + ext)
    (hcons : ∀ cs ∈ st.conss.toList, (Refine.consFixed levels cs.start).size + ext = cs.cols)
    (hW : WAt bF n w (initCh.toList.map gOf) (levelsOf levels) oods cons ext)
    (hoods : st.oods.toList.map oodOf = oods) {cs₀ : List Flock.ConsCtx} {cc : Flock.ConsCtx} (hcs : st.conss.toList = cs₀ ++ [cc])
    (hcons₀ : cs₀.map consOf = cons) (hcols : cc.cols = ext) (hstart : cc.start = levels.size)
    (hf : f.size = 2 ^ ext) {s val : F256}
    (hsum : gOf s = ∑ x ∈ Finset.range (2 ^ ext), gv f x * gv w x)
    (hval : gOf val = ∑ x ∈ Finset.range (2 ^ ext), gv f x * gv (embedF (consW ext cc.pos cc.alpha)) x) :
    lhsE (splitK f) ext (resE basisAt n initCh levels ext st) = s + val.smul cc.beta := by
  have hpc : cc.pos.size ≤ 2 ^ cc.alpha.size := hpos cc (by rw [hcs]; simp)
  refine lhsE_honest f w _ _ ext s val cc.beta hf hsum hval ?_
  refine hW.final cc.pos cc.alpha hpc cc.beta (start := cc.start) (by rw [hstart]; simp [levelsOf]) fun y hy => ?_
  rw [gOf_resE basisAt n initCh levels ext st bF hl horig hpos hood hcons hbasis hy, hoods, hcs, List.map_append,
    hcons₀, List.map_singleton]
  obtain ⟨c1, c2, c3, c4, c5⟩ := cc
  simp only at hcols
  subst hcols
  rfl

end Lg

end FlockComplete.Replay
