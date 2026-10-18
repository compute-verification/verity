import Proofs.Flock.Claims.KS.ZkLiveW

/-!
# One wait's bound from every table's bound, the escapes counted per round

Escapes that fit a weight counted per round (`Escapes.FitsR`) fit it as `Escapes.Fits` asks once their fork events are
merged by round (`Escapes.mergeR`: at each round one event, that some event there holds, of weight the round's count
times its events' total weight): the merged escapes have one event per round (`Escapes.FitsR.fits_mergeR`), and by
Cauchy–Schwarz their term is at least the escapes' (`Escapes.term_le_mergeR`). So `OneWait` follows from every table's
bound with the escapes counted per round (`Claims.Flock.TableBoundJR`, `Claims.Flock.oneWait_of_tableBoundJR`): the
conflict fork at round `0` gives escapes fitting per round (`Escapes.fitsR_addFork`), and their merge fits `OneWait`'s
weight. At tables with unopened positions it follows from `ExtendJ` (`Claims.Flock.tableBoundJR_of_extend`). The
deployed masked verifier is KS from every table's bound counted per round (`FlockDeploy.KS.zk_KS_of_tableBoundJR`),
so from the live sessions' bound off the table on a witness lane where some run accepts and `ExtendJ`
(`FlockDeploy.KS.zk_KS_of_liveAccWJ`).
-/

namespace Claims

/-! ## Fork events merged by round -/

section Merge

variable {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}

/-- **The escapes' fork events merged by round**: at each round one event, that some event at the round holds, of
weight the round's count of events times their total weight. -/
def Escapes.mergeR (e : Escapes H V P x) : Escapes H V P x where
  I := Fin (V.k x)
  c r := ((Finset.univ.filter fun i => e.j i = r).card : ℝ) * ∑ i ∈ Finset.univ.filter (fun i => e.j i = r), e.c i
  j r := r
  ev r ω s s' := ∃ i, e.j i = r ∧ e.ev i ω s s'
  L := e.L
  β := e.β
  ev₁ := e.ev₁
  c_nonneg _ := mul_nonneg (Nat.cast_nonneg _) (Finset.sum_nonneg fun i _ => e.c_nonneg i)
  β_nonneg := e.β_nonneg
  tied r ω s s' := by
    rintro ⟨i, rfl, h⟩
    exact e.tied i ω s s' h
  tied₁ := e.tied₁

/-- **Merged by round, escapes that fit `C` per round fit it** with one event per round. -/
theorem Escapes.FitsR.fits_mergeR {e : Escapes H V P x} {C : ℝ} (h : e.FitsR C) : e.mergeR.Fits C := by
  refine ⟨1, fun r => ?_, ?_, h.2⟩
  · show (Finset.univ.filter fun r' : Fin (V.k x) => r' = r).card ≤ 1
    exact Finset.card_le_one.2 fun a ha b hb => by
      rw [(Finset.mem_filter.1 ha).2, (Finset.mem_filter.1 hb).2]
  · show ((1 : ℕ) : ℝ) * ∑ r : Fin (V.k x), ((Finset.univ.filter fun i => e.j i = r).card : ℝ) *
        ∑ i ∈ Finset.univ.filter (fun i => e.j i = r), e.c i ≤ 2 * C
    rw [Nat.cast_one, one_mul]
    refine le_trans (le_of_eq ?_) h.1
    rw [← Finset.sum_fiberwise Finset.univ e.j
      (fun i => ((Finset.univ.filter fun i' => e.j i' = e.j i).card : ℝ) * e.c i)]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i hi => by rw [(Finset.mem_filter.1 hi).2]

/-- **Merging by round only raises the escapes' term**: at each round, by Cauchy–Schwarz,
`Σ_i √(c_i·Pr[ev_i]) ≤ √(n·Σ_i c_i·Pr[∃ i, ev_i])` over its `n` events. -/
theorem Escapes.term_le_mergeR [Fintype Coin] (e : Escapes H V P x) (ω : Ω) : e.term ω ≤ e.mergeR.term ω := by
  unfold Escapes.term
  refine add_le_add ?_ le_rfl
  show _ ≤ ∑ r : Fin (V.k x), Real.sqrt ((((Finset.univ.filter fun i => e.j i = r).card : ℝ) *
      ∑ i ∈ Finset.univ.filter (fun i => e.j i = r), e.c i) *
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ∃ i, e.j i = r ∧ e.ev i ω s.1 s.2)
  rw [← Finset.sum_fiberwise Finset.univ e.j]
  refine Finset.sum_le_sum fun r _ => ?_
  set F := Finset.univ.filter fun i => e.j i = r with hF
  set Q := Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ∃ i, e.j i = r ∧ e.ev i ω s.1 s.2 with hQ
  have hQ0 : 0 ≤ Q := Pr_nonneg _
  have hc0 : 0 ≤ ∑ i ∈ F, e.c i := Finset.sum_nonneg fun i _ => e.c_nonneg i
  calc ∑ i ∈ F, Real.sqrt (e.c i * Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => e.ev i ω s.1 s.2)
      ≤ ∑ i ∈ F, Real.sqrt 1 * Real.sqrt (e.c i) * Real.sqrt Q := Finset.sum_le_sum fun i hi => by
        rw [Real.sqrt_one, one_mul, ← Real.sqrt_mul (e.c_nonneg i)]
        exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
          (Pr_mono fun s h => ⟨i, (Finset.mem_filter.1 hi).2, h⟩) (e.c_nonneg i))
    _ = (∑ i ∈ F, Real.sqrt 1 * Real.sqrt (e.c i)) * Real.sqrt Q := by rw [Finset.sum_mul]
    _ ≤ (Real.sqrt (∑ _i ∈ F, (1 : ℝ)) * Real.sqrt (∑ i ∈ F, e.c i)) * Real.sqrt Q :=
        mul_le_mul_of_nonneg_right (Real.sum_sqrt_mul_sqrt_le F (fun _ => zero_le_one) fun i => e.c_nonneg i)
          (Real.sqrt_nonneg _)
    _ = Real.sqrt (((F.card : ℝ) * ∑ i ∈ F, e.c i) * Q) := by
        rw [Finset.sum_const, nsmul_eq_mul, mul_one, Real.sqrt_mul (mul_nonneg (Nat.cast_nonneg _) hc0),
          Real.sqrt_mul (Nat.cast_nonneg _)]

end Merge

end Claims

namespace Claims.Flock

open FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

/-! ## One wait, from every table's bound counted per round -/

section Tables

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **`TableBoundJ` with the escapes counted per round** (`Escapes.FitsForkR`). -/
def TableBoundJR (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x,
    e.FitsForkR (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (j : ℕ), j < nTab D x → ∀ C : Fin (N x j) → Col,
      (∀ p, Ok x j (C p)) → ¬ GoodTableAt D N x j C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x j)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen j) p.val = some u ∧ u ≠ C p

omit [Nonempty Coin] in
theorem tableBoundJR_of_accAtR {H : Msg → Msg} {D : Deploy Coin Col T} {N : Msg → ℕ → ℕ}
    {Ok : Msg → ℕ → Col → Prop} (h : ∀ x, TableBoundJAccAtR H D N Ok x) : TableBoundJR H D N Ok := by
  intro x Ω _ _ P
  obtain ⟨e, -, hf, hb⟩ := h x Ω P
  exact ⟨e, hf, hb⟩

/-- **Flock's one-wait bound, from every table's bound counted per round**: `oneWait_of_tableBoundJ`, the escapes
with the conflict fork merged by round (`Escapes.mergeR`). -/
theorem oneWait_of_tableBoundJR (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (hN : ∀ x, ((∑ j : Fin (nTab D x), N x j : ℕ) : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x)
    (hVk : ∀ x, 0 < D.V.k x) (hk : ∀ x, 1 ≤ D.k x ∨ NoAcc H D.V x)
    (hp : ∀ x, (∃ s, D.parse x = some s) ∨ NoAcc H D.V x) (Ok : Msg → ℕ → Col → Prop) (hz : ∀ x j, Ok x j 0)
    (hok : ColsOkJ H D N Ok) (hconf : ConflictTiedJ H D N) (htab : TableBoundJR H D N Ok) :
    OneWait H D := by
  classical
  intro x Ω _ _ P
  obtain ⟨e₀, hfit, hb⟩ := htab x Ω P
  set j₀ : Fin (D.V.k x) := ⟨0, hVk x⟩
  set ev : Ω → (Fin (D.V.k x) → Coin) → (Fin (D.V.k x) → Coin) → Prop :=
    fun ω r r' => ∃ p, ConflictAt (opJ H D N P x ω) (r, r') p with hev
  have tied : ∀ ω r r', ev ω r r' →
      TiedColl H ((run H P D.V x ω r).vLog ++ (run H P D.V x ω (forkCoins j₀ r r')).vLog) := by
    intro ω r r' ⟨p, h1, h2, h3⟩
    rw [Claims.Bridge.forkCoins_zero j₀ rfl]
    simp only [opJ] at h1 h2 h3
    split_ifs at h1 h2 h3 with ha ha'
    · obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
      obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 h2
      exact hconf x Ω P ω r r' p.1.val p.2.val u v ha ha' p.1.isLt p.2.isLt hu hv
        fun huv => h3 (by rw [hu, hv, huv])
    all_goals simp at h1 h2
  refine ⟨(e₀.addFork (D.k x) (Nat.cast_nonneg _) j₀ ev tied).mergeR,
    (e₀.fitsR_addFork (D.weight x) (D.k x) (Nat.cast_nonneg _) j₀ rfl ev tied hfit).fits_mergeR,
    fun ω hε => ?_⟩
  have hacc : ¬ NoAcc H D.V x := fun hno => by
    have : accPr H D.V P x ω = 0 := by
      unfold accPr
      rw [Pr_congr (q := fun _ => False) fun r => ⟨fun h => hno Ω P ω r h, False.elim⟩, Pr_false]
    linarith
  have hk1 : 1 ≤ D.k x := (hk x).resolve_right hacc
  obtain ⟨s, hs⟩ := (hp x).resolve_right hacc
  set Xt : List (Fin (D.V.k x) → Coin) → Prop := fun bs =>
    (∀ b ∈ bs, Acc H D.V P x ω b) → Extracts D x (seenOf H D.V P x ω bs) with hXt
  have hXb : ∀ bs, ¬ Xt bs → expect (ind (Acc H D.V P x ω)) (runGame (D.V.k x)) (runS _) ≤
      (D.εtable x + e₀.term ω) + expect (ind fun r => Acc H D.V P x ω r ∧
        ∃ p, Off (opJ H D N P x ω) (firstTable (opJ H D N P x ω) (fun _ => 0) bs) r p)
        (runGame (D.V.k x)) (runS _) := by
    intro bs hnX
    simp only [hXt, Classical.not_imp] at hnX
    obtain ⟨hall, hnot⟩ := hnX
    obtain ⟨j, hj, hnj⟩ : ∃ j < s.tables, ¬ ExtractsAt D x (seenOf H D.V P x ω bs) j := by
      by_contra hc
      push Not at hc
      exact hnot ⟨s, hs, hc⟩
    have hjn : j < nTab D x := (nTab_of_parse hs).symm ▸ hj
    have hCe : (fun p : Fin (N x j) => firstTable (opJ H D N P x ω) (fun _ => 0) bs ⟨⟨j, hjn⟩, p⟩) =
        fun p => (lookCol ((seenOf H D.V P x ω bs).flatMap fun v => D.cols v j) p.val).getD 0 :=
      funext fun p => firstTable_opJ H D N P x ω bs hall ⟨⟨j, hjn⟩, p⟩
    have hgood : ¬ GoodTableAt D N x j
        fun p => firstTable (opJ H D N P x ω) (fun _ => 0) bs ⟨⟨j, hjn⟩, p⟩ := fun hg =>
      hnj (extractsAt_of_goodAt D N x j _ (hCe ▸ hg))
    have hOk : ∀ p, Ok x j (firstTable (opJ H D N P x ω) (fun _ => 0) bs ⟨⟨j, hjn⟩, p⟩) := by
      intro p
      rw [congrFun hCe p]
      cases hl : lookCol ((seenOf H D.V P x ω bs).flatMap fun v => D.cols v j) p.val with
      | none => exact hz x j
      | some u =>
        rw [seenOf, List.flatMap_map] at hl
        obtain ⟨b, hb', hu⟩ := lookCol_flatMap_some (fun r => D.cols (run H P D.V x ω r).seen j) p.val u bs hl
        exact hok x Ω P ω b j p.val u (hall b hb') hjn p.isLt hu
    rw [← accPr_eq, expect_runGame, ← Claims.Bridge.Pr_eq_Ex]
    refine (hb ω j hjn _ hOk hgood).trans (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne⟩ => ?_))
    have e : opJ H D N P x ω r ⟨⟨j, hjn⟩, p⟩ = some u := by
      unfold opJ
      rw [ite_eq_left ha]
      exact hu
    refine ⟨ha, ⟨⟨j, hjn⟩, p⟩, ?_⟩
    show (opJ H D N P x ω r ⟨⟨j, hjn⟩, p⟩).isSome ∧ opJ H D N P x ω r ⟨⟨j, hjn⟩, p⟩ ≠ some _
    rw [e]
    exact ⟨rfl, fun h => hne (Option.some_injective _ h)⟩
  have hεc : 0 ≤ D.εtable x + e₀.term ω := add_nonneg (hεt x) (e₀.term_nonneg ω)
  have hj := joint_acc_le (opJ H D N P x ω) (fun _ => 0) (runGame (D.V.k x)) (runS _)
    (Acc H D.V P x ω) Xt _ hεc hXb (D.k x) hk1
  have hlhs : accPr H D.V P x ω * (accFail H D.V P x ω (D.k x) (Extracts D x) /
        accPr H D.V P x ω ^ D.k x) =
      expect (ind (Acc H D.V P x ω)) (runGame (D.V.k x)) (runS _) *
        expectAcc (runGame (D.V.k x)) (runS _) (Acc H D.V P x ω) (D.k x) (ind fun bs => ¬ Xt bs) := by
    rw [oneWait_lhs_eq]
    congr 1
    unfold expectAcc
    congr 1
    refine congrArg (expectN _ _ _) (funext fun bs => ?_)
    rw [prod_map_ind]
    by_cases hall : ∀ b ∈ bs, Acc H D.V P x ω b
    · congr 1
      unfold ind
      have : (¬ Xt bs) ↔ ¬ Extracts D x (seenOf H D.V P x ω bs) := by
        rw [hXt]
        exact not_congr ⟨fun h => h hall, fun h _ => h⟩
      beta_reduce
      split_ifs with h1 h2 h2 <;> first | rfl | (exfalso; tauto)
    · rw [ite_eq_right hall, zero_mul, zero_mul]
  have hE₂ : expect₂ (fun a a' => ind (fun y : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) =>
        ∃ p, ConflictAt (opJ H D N P x ω) y p) (a, a')) (runGame (D.V.k x)) (runS _) =
      Pr fun s : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) => ev ω s.1 s.2 := by
    unfold expect₂
    simp only [expect_runGame]
    rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  rw [hE₂, ← accPr_eq] at hj
  have hm := (e₀.addFork (D.k x) (Nat.cast_nonneg _) j₀ ev tied).term_le_mergeR ω
  rw [Escapes.term_addFork] at hm
  rw [hlhs, ← accPr_eq]
  have hrate : (Fintype.card (PosJ D N x) : ℝ) * accPr H D.V P x ω / (Real.exp 1 * D.k x) ≤
      D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω := by
    rw [Fintype.card_sigma, div_mul_eq_mul_div]
    simp only [Fintype.card_fin]
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (hN x) hε.le) (by positivity)
  linarith

omit [Nonempty Coin] in
/-- **Every table's bound counted per round at tables with unopened positions**: `tableBoundJ_of_extend`. -/
theorem tableBoundJR_of_extend (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (Ok : Msg → ℕ → Col → Prop) (hne : ∀ x j u, 0 < N x j → Ok x j u → u ≠ 0) (hok : ColsOkJ H D N Ok)
    (hext : ExtendJ D N Ok) (htab : TableBoundJR H D N Ok) : TableBoundJR H D N (orZeroJ Ok) := by
  intro x Ω _ _ P
  obtain ⟨e, hfit, hb⟩ := htab x Ω P
  refine ⟨e, hfit, fun ω j hj C hC hg => ?_⟩
  obtain ⟨C', hC', hkeep, hg'⟩ := hext x j hj C hC hg
  refine (hb ω j hj C' hC' hg').trans
    (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne'⟩ => ⟨ha, p, u, hu, ?_⟩))
  by_cases h0 : C p = 0
  · rw [h0]
    exact hne x j u (Nat.lt_of_le_of_lt (Nat.zero_le _) p.isLt) (hok x Ω P ω r j p.val u ha hj p.isLt hu)
  · rwa [← hkeep p h0]

end Tables

end Claims.Flock

namespace FlockDeploy.KS

open Flock

/-- **The deployed masked (`--zk`) verifier: KS**, at SHA-512, from every table's bound counted per round:
`zk_KS_of_tableBoundJ` at `oneWait_of_tableBoundJR`. -/
theorem zk_KS_of_tableBoundJR
    (htab : Claims.Flock.TableBoundJR sha512 (deployZk Readable.relation) (wordLenJ true)
      (Claims.Flock.orZeroJ (wideJ true))) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_of_oneWait_sha512 Readable.relation zk_splits
    (Claims.Flock.oneWait_of_tableBoundJR sha512 (deployZk Readable.relation) (wordLenJ true)
      (fun x => sum_wordLenJ_le true x _) (fun x => εtable_nonneg true x) (V_k_pos true) (k_one_zk _)
      (zk_parses _) _ (fun _ _ => .inl rfl) (Claims.Flock.colsOkJ_orZero _ _ _ _ (zk_colsOkJ sha512 _))
      (zk_conflictTiedJ _) htab)

/-- **Every masked table's bound counted per round at tables with unopened positions**, from the bound at full-width
tables and `ExtendJ`. -/
theorem zk_tableBoundJR_of_extend (rel : Readable.Parsed → Readable.Relation)
    (hext : Claims.Flock.ExtendJ (deployZk rel) (wordLenJ true) (wideJ true))
    (htab : Claims.Flock.TableBoundJR sha512 (deployZk rel) (wordLenJ true) (wideJ true)) :
    Claims.Flock.TableBoundJR sha512 (deployZk rel) (wordLenJ true) (Claims.Flock.orZeroJ (wideJ true)) :=
  Claims.Flock.tableBoundJR_of_extend _ _ _ _ (wideJ_ne_zero true) (zk_colsOkJ sha512 rel) hext htab

/-- **The deployed masked verifier: KS**, at SHA-512 at the deployed relation, at every number of tables, from the live
sessions' bound off the table on a witness lane at the statements some run accepts (`ZkLiveAccWJ`) and `ExtendJ` at
full-width columns. -/
theorem zk_KS_of_liveAccWJ (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveAccWJ x)
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_KS_of_tableBoundJR (zk_tableBoundJR_of_extend Readable.relation hext
    (Claims.Flock.tableBoundJR_of_accAtR fun x => zk_tableBoundJAccAtR hl x))

end FlockDeploy.KS
