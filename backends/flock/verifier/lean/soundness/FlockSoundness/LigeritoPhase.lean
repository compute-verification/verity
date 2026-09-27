import FlockSoundness.LigeritoMid

/-!
# Ligerito is sound (PROTOCOL.md §13)

From a claim `Σ_y f(y)·b(y) = T` false for every message near the committed level-0 table,
Ligerito accepts with probability at most `εLig`:
* level 0: the OOD claim's batching coin `β₀` (`L₀/|F|`), then the level-0 folds (`phaseB0`);
* every next level: `phaseA` from `PostFold` to `PreFold`, then its folds (`phaseB`);
* the last level: its queries against the clear residual and the final check (`finalStage`).
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## The schedules' shape -/

/-- A level the proof applies to. -/
def levelGood (l : Level) : Bool :=
  decide l.Ok && decide l.Fits && decide (1 ≤ l.lanes) && decide (l.queries ≤ 2 ^ 64)

/-- The levels chain: every level good, each level's columns plus the split coordinate are the
next level's variables, and the last level's columns are the clear residual's. -/
def levelsChained : List Level → ℕ → Bool
  | [], _ => false
  | [l], ext => levelGood l && decide (l.logCols = ext)
  | l :: l' :: rest, ext =>
      levelGood l && decide (l.logCols + 1 = l'.lanes + l'.logCols) && levelsChained (l' :: rest) ext

theorem levelGood_spec {l : Level} (h : levelGood l = true) :
    l.Ok ∧ l.Fits ∧ 1 ≤ l.lanes ∧ l.queries ≤ 2 ^ 64 := by
  simp only [levelGood, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

/-- The shape of every `fast100` schedule. -/
theorem fast100_shape : ∀ m s, fast100 m = some s →
    2 ≤ s.levels.length ∧ (s.levels.headD default).lanes = s.k0 ∧
      s.m - 7 = s.k0 + (s.levels.headD default).logCols ∧ levelsChained s.levels s.ext = true := by
  intro m s h
  have hm : ∀ m', fast100 m' ≠ none → 22 ≤ m' ∧ m' ≤ 35 := by
    intro m' hm'
    by_contra hc
    apply hm'
    unfold fast100
    split <;> simp_all
  obtain ⟨h1, h2⟩ := hm m (by simp [h])
  have key : ∀ m', 22 ≤ m' → m' ≤ 35 → ∀ s', fast100 m' = some s' →
      (decide (2 ≤ s'.levels.length) && decide ((s'.levels.headD default).lanes = s'.k0) &&
        decide (s'.m - 7 = s'.k0 + (s'.levels.headD default).logCols) &&
        levelsChained s'.levels s'.ext) = true := by
    intro m' h1' h2' s' hs'
    interval_cases m' <;> (simp only [fast100, Option.some.injEq] at hs'; subst hs'; decide +kernel)
  have := key m h1 h2 s h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at this
  exact ⟨this.1.1.1, this.1.1.2, this.1.2, this.2⟩

/-! ## Level 0 -/

theorem mleAt_bits_self [CharP K 2] (N : ℕ) (c : ℕ → K) (y : ℕ) (hy : y < 2 ^ N) :
    mleAt N c (bitsK N y) = c y := by
  unfold mleAt
  rw [Finset.sum_eq_single_of_mem y (Finset.mem_range.2 hy)]
  · rw [eqList_bitsK _ y y hy hy, if_pos rfl, mul_one]
  · intro w hw hwy
    rw [eqList_bitsK _ y w hy (Finset.mem_range.1 hw), if_neg hwy, mul_zero]

/-- The level-0 weight: the claim `Σ_y f(y)·b(y) = T` and the level-0 OOD claim `f̂(z₀) = y₀`,
batched by `β₀`. -/
def W0 (A : Arith F K) (b : ℕ → F) (z₀ : List F) (β₀ : F) (y : ℕ) : K :=
  A.embed (b y) + A.embed β₀ * eqList (z₀.map A.embed) y

/-- **The level-0 fold identity.** -/
theorem foldLoL_W0 [CharP K 2] (A : Arith F K) (n : ℕ) (b : ℕ → F) (z₀ : List F)
    (hz : z₀.length = n) (β₀ : F) (ch : List K) (e : ℕ) (hn : n = ch.length + e) (x : ℕ)
    (hx : x < 2 ^ e) :
    foldLoL ch (W0 A b z₀ β₀) x = resAt A n b ch [] [⟨z₀, β₀, none⟩] [] e x := by
  have hself : ∀ y < 2 ^ n, W0 A b z₀ β₀ y =
      mleAt n (fun w => A.embed (b w)) ([] ++ bitsK n y) +
        A.embed β₀ * mleAt z₀.length (eqList (z₀.map A.embed)) ([] ++ bitsK n y) := by
    intro y hy
    rw [List.nil_append, mleAt_bits_self n _ y hy, hz, mleAt_bits_self n _ y hy]
    rfl
  rw [foldLoL_congr ch _ (fun y => mleAt n (fun w => A.embed (b w)) ([] ++ bitsK n y) +
      A.embed β₀ * mleAt z₀.length (eqList (z₀.map A.embed)) ([] ++ bitsK n y)) x fun y hy => hself _ (by
        rw [hn, pow_add]
        calc y + 2 ^ ch.length * x < 2 ^ ch.length + 2 ^ ch.length * x := by omega
          _ = 2 ^ ch.length * (x + 1) := by ring
          _ ≤ 2 ^ ch.length * 2 ^ e := Nat.mul_le_mul_left _ hx),
    foldLoL_add, foldLoL_mul_left, foldLoL_mle_bits _ _ [] ch n (by omega),
    foldLoL_mle_bits _ _ [] ch n (by omega), List.nil_append, show n - ch.length = e by omega]
  have hz' : (⟨z₀, β₀, none⟩ : OodCtx F).z.length =
      (oodFixed ch [] (⟨z₀, β₀, none⟩ : OodCtx F)).length + e := by
    simp [oodFixed, origOf, hz, hn]
  unfold resAt
  rw [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, List.map_nil,
    List.sum_nil, add_zero, oodTerm_eq_mle A ch [] e _ x hz']
  simp only [oodFixed, oodScale, origOf, cffFrom, List.map_nil, List.flatten_nil, List.append_nil,
    List.drop_nil, List.prod_nil, mul_one, bHat, mleAt]

/-- **The level-0 folds.** From a level-0 claim false for every message of the list against
`W0`, after the folds by `initial` the claim is false for every fold, against `resAt` with the
level-0 OOD claim, and no round was an MCA event, except with probability `foldError`. -/
theorem phaseB0 (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K) (hA : A.Correct) (n : ℕ)
    (b : ℕ → F) (l : Level) (hl : l.Ok) (hf : l.Fits) (k0 : ℕ) (hk0 : l.lanes = k0)
    (hn : n = k0 + l.logCols) (C : Oracle F) (z₀ : List F) (hz : z₀.length = n) (β₀ : F) (s₀ : K)
    (q₀ : Quad K) (hq₀ : q₀.eval 0 + q₀.eval 1 = s₀) (dim : ℕ)
    (hdoom : ∀ M ∈ closeL A l.radius k0 l.logCols l.logLen C,
      s₀ ≠ ∑ y ∈ Finset.range (2 ^ n), A.embed (flat M y) * W0 A b z₀ β₀ y) :
    value (fun out : List K × K × Quad K => ¬ PostFold A hA n b out.1 l hf k0 C out.1
        ⟨out.2.1, out.2.2, dim, [], [⟨z₀, β₀, none⟩], [], C, l⟩) (foldRounds A k0 s₀ q₀) ≤
      ENNReal.ofReal (l.foldError l.listBound) := by
  classical
  haveI : CharP K 2 := hA.charTwoK
  set S := closeL A l.radius k0 l.logCols l.logLen C
  have hS : S.card ≤ l.listBound := card_close_le A hA l hf k0 (tbl k0 l.logLen C)
  set u : Message F k0 l.logCols → ℕ → K := fun M y => A.embed (flat M y)
  set v := W0 A b z₀ β₀
  have himp : ∀ out : List K × K × Quad K, ¬ PostFold A hA n b out.1 l hf k0 C out.1
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
  have hmcaε : ∀ U : Fin 2 → Fin (2 ^ l.logLen) → K,
      prCoin (fun γ : K => IsMCA (AffineLineGenerator K) (codeK A hA l hf) γ U l.radius) ≤
        ENNReal.ofReal (l.mcaA / qK) := prCoin_level_mca_le hMCA A hA l hl hf
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
          (add_le_add (value_foldRounds_mca_le A hA (codeK A hA l hf) l.radius _ hmcaε k0 _ _ _)
            hsumε)
    _ ≤ ENNReal.ofReal (l.foldError l.listBound) := by
        rw [zero_add, zero_add, add_comm]
        rcases foldError_bound (K := K) l l.listBound (mcaA_nonneg l hl) with h | h
        · rw [hk0] at h
          refine le_trans (add_le_add ?_ le_rfl) h
          gcongr
        · exact absurd hA.card_K h

/-! ## The last level -/

theorem ofReal_lastLevel (A : Arith F K) (hA : A.Correct) (l : Level) (hl : l.Ok) :
    ENNReal.ofReal (l.foldError l.listBound) + (ENNReal.ofReal l.queryError +
        ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F + ((({()} : Finset Unit).card : ℕ) : ℝ≥0∞) /
          Fintype.card F)) ≤ ENNReal.ofReal (lastLevelError l) := by
  have hq : (0 : ℝ) < qF := by unfold qF; positivity
  have h1 := foldError_nonneg l l.listBound hl
  have h2 := queryError_nonneg l
  have h3 : (0 : ℝ) ≤ (ceilLog2 l.queries : ℝ) / qF := by positivity
  have h4 : (0 : ℝ) ≤ 1 / qF := by positivity
  unfold lastLevelError
  rw [ENNReal.ofReal_add (add_nonneg (add_nonneg h1 h2) h3) h4,
    ENNReal.ofReal_add (add_nonneg h1 h2) h3, ENNReal.ofReal_add h1 h2]
  rw [Finset.card_singleton, hA.card_F, show (1 : ℝ) / qF = ((1 : ℕ) : ℝ) / qF by simp,
    ofReal_nat_div_qF', ofReal_nat_div_qF', Nat.cast_pow, Nat.cast_ofNat]
  apply le_of_eq
  ring

/-- **The last level.** From `PreFold` on the last level, its folds, its queries against the
clear residual and the final check accept with probability at most `lastLevelError`. -/
theorem finalStage (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K) (hA : A.Correct) (n : ℕ)
    (b : ℕ → F) (initial : List K) (ext : ℕ) (l : Level) (hl : l.Ok) (hf : l.Fits)
    (hlanes : 1 ≤ l.lanes) (hQ : l.queries ≤ 2 ^ 64) (hext : l.logCols = ext) (e : ℕ)
    (st : LigState F K) (hpre : PreFold A n b initial l st.prev e st) :
    value (fun r => r = true) (recursiveRounds A n b initial ext [l] st) ≤
      ENNReal.ofReal (lastLevelError l) := by
  classical
  haveI : CharP K 2 := hA.charTwoK
  rw [recursiveRounds]
  refine (value_bind_le (foldRounds A l.lanes st.s st.q) _ _
    (fun out => PostFold A hA n b initial l hf l.lanes st.prev out.1
      { st with s := out.2.1, q := out.2.2, levels := st.levels ++ [out.1] })
    (ENNReal.ofReal l.queryError + ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F +
      ((({()} : Finset Unit).card : ℕ) : ℝ≥0∞) / Fintype.card F)) fun out hpost => ?_).trans ?_
  · obtain ⟨ch, s, q⟩ := out
    set st₁ : LigState F K := { st with s := s, q := q, levels := st.levels ++ [ch] }
    show value _ (Game.send (Fin (2 ^ (ext + 1)) → F) fun yr =>
      (queryAndBatch A l st₁.prev ch st₁.levels.length st₁ false).bind fun st' =>
        Game.ret (finalCheck A n b initial st'.levels st'.oods st'.cons ext yr st'.s)) ≤ _
    rw [value_send]
    refine iSup_le fun yr => ?_
    rw [value_bind_ret]
    let G : ℕ → K := fun y =>
      A.embed (yr ⟨(2 * y) % 2 ^ (ext + 1), Nat.mod_lt _ (Nat.two_pow_pos _)⟩) +
        A.u * A.embed (yr ⟨(2 * y + 1) % 2 ^ (ext + 1), Nat.mod_lt _ (Nat.two_pow_pos _)⟩)
    let D : Unit → K := fun _ => st₁.s - ∑ x ∈ Finset.range (2 ^ l.logCols),
      G x * resAt A n b initial st₁.levels st₁.oods st₁.cons l.logCols x
    refine value_queryAndBatch_le A hA l hf.2 hQ st₁.prev ch st₁.levels.length st₁ false {()} D
      (fun _ => G) (fun _ _ _ _ _ _ => rfl)
      (fun _ _ hD => query_bound_of_doom A hA n b initial l hf l.lanes st₁.prev ch st₁ hpost G
        (sub_eq_zero.1 hD)) _ ?_
    intro w α β st' h1 h2 h3 h4 _ _ hacc
    refine ⟨(), Finset.mem_singleton_self _, ?_⟩
    rw [finalCheck_eq_resAt, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq] at hacc
    have hacc : ∑ y ∈ Finset.range (2 ^ ext),
        resAt A n b initial st'.levels st'.oods st'.cons ext y * G y = st'.s := hacc.2
    rw [h2, h3, h4] at hacc
    simp only [resAt_cons_append] at hacc
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
      consTerm_new A st₁.levels ext ⟨l.logCols, positions A l.queries l.logLen w, List.ofFn α, β,
        st₁.levels.length⟩ rfl, add_mul, Finset.sum_add_distrib] at hacc
    rw [h1] at hacc
    have hswap : ∑ y ∈ Finset.range (2 ^ ext), A.embed β *
        ((positions A l.queries l.logLen w).zipIdx.map fun pj =>
          A.embed (eqList (List.ofFn α) pj.2) * A.embed (A.Xhat y ext (A.omega pj.1))).sum * G y =
        A.embed β * ((positions A l.queries l.logLen w).zipIdx.map fun pj =>
          A.embed (eqList (List.ofFn α) pj.2) * encK A ext G pj.1).sum := by
      rw [show (fun y => A.embed β * ((positions A l.queries l.logLen w).zipIdx.map fun pj =>
          A.embed (eqList (List.ofFn α) pj.2) * A.embed (A.Xhat y ext (A.omega pj.1))).sum * G y) =
          fun y => A.embed β * (G y * ((positions A l.queries l.logLen w).zipIdx.map fun pj =>
            A.embed (eqList (List.ofFn α) pj.2) * A.embed (A.Xhat y ext (A.omega pj.1))).sum) from
          funext fun y => by ring, ← Finset.mul_sum, sum_mul_list_sum]
      congr 2
      refine List.map_congr_left fun pj _ => ?_
      unfold encK
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun x _ => by ring
    rw [hswap] at hacc
    have hcv : consValue A (List.ofFn α) ch st₁.prev (positions A l.queries l.logLen w) =
        ((positions A l.queries l.logLen w).zipIdx.map fun pj =>
          A.embed (eqList (List.ofFn α) pj.2) * foldColumn A ch (fun ℓ => st₁.prev ℓ pj.1)).sum := rfl
    have hgap : ((positions A l.queries l.logLen w).zipIdx.map fun pj =>
        A.embed (eqList (List.ofFn α) pj.2) *
          ((foldColumn A ch fun ℓ => st₁.prev ℓ pj.1) - encK A l.logCols G pj.1)).sum =
        consValue A (List.ofFn α) ch st₁.prev (positions A l.queries l.logLen w) -
          ((positions A l.queries l.logLen w).zipIdx.map fun pj =>
            A.embed (eqList (List.ofFn α) pj.2) * encK A l.logCols G pj.1).sum := by
      rw [hcv, ← list_sum_map_sub]
      exact congrArg List.sum (List.map_congr_left fun pj _ => by ring)
    simp only [D, consGap]
    rw [hgap]
    rw [← hext] at hacc
    have hc : ∑ y ∈ Finset.range (2 ^ l.logCols),
        resAt A n b initial st₁.levels st₁.oods st₁.cons l.logCols y * G y =
      ∑ x ∈ Finset.range (2 ^ l.logCols),
        G x * resAt A n b initial st₁.levels st₁.oods st₁.cons l.logCols x :=
      Finset.sum_congr rfl fun y _ => mul_comm _ _
    rw [hc] at hacc
    linear_combination -hacc
  · exact add_le_add (phaseB hMCA A hA n b initial l hl hf st.prev e st hpre hlanes) le_rfl |>.trans
      (ofReal_lastLevel A hA l hl)

end FlockSoundness
