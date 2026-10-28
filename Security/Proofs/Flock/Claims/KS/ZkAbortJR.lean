import Proofs.Flock.Claims.KS.ZkAbortR

/-!
# The recursion's KS over the masked record at every number of tables, from every table's bound counted per round

`ZkAbortR` at several tables, with no premise that the statement has one table. One wait over draws holds at every
table's word's positions (`Claims.Recursion.opDJ`, `Claims.Recursion.firstTable_opDJ`, `Claims.Recursion.oneWaitDJ`,
as `oneWait_of_tableBoundJ` over runs): at a draw list that does not extract, the first table that does not is not
`GoodTableAt` there. Every table's bound with escapes on accepted runs counted per round
(`Claims.Recursion.TableBoundJOnAccR`, at every statement `Flock.TableBoundJAccAtR`) gives it at the abort law
(`Claims.Recursion.TableBoundJAbortR`, `Claims.Recursion.tableBoundJAbortR_of_onAccR`), so one wait there with the
escapes and the conflict fork merged by round (`Claims.Recursion.oneWaitAbort_of_tableBoundJR`), so `Inner.KSAbort`
wherever the statement parses where some run accepts (`Claims.Recursion.KSAbort_of_tableBoundJOnAccR`). At the masked
record the claims read, every table's bound at full-width columns at the untrimmed record carries to the trimmed one
by padding back, `0` being a trimmed full-width column (`Claims.Recursion.tableBoundJOnAccR_mapCols`,
`FlockDeploy.KS.zk_tableBoundJOnAccR_trim`), so the record meets `Inner.KSAbort` and the recursion's KS over it holds
from that bound alone (`FlockDeploy.KS.zk_KSAbortJR`, `FlockDeploy.KS.zk_KSFlockJR`), so from a simulator of each
masked prover where some run accepts (`FlockDeploy.KS.zk_tableBoundJOnAccR_of_simWJ`,
`FlockDeploy.KS.zk_KSAbort_of_simWJ_tables`, `FlockDeploy.KS.zk_KSFlock_of_simWJ_tables`).
-/

namespace Claims.Recursion

open Claims Claims.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

/-! ## One wait over draws at every table -/

section OneWaitDJ

variable {Dr Coin Col T : Type} [Fintype Dr] [Nonempty Dr] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
  [DecidableEq Col] [Zero Col]

open Classical in
/-- **An accepted run's openings over draws** at the words' positions, its first column of table `j` at table `j`'s;
none on a rejected run. -/
noncomputable def opDJ (D : Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (x : Msg) (tr : Dr → Trace) (d : Dr)
    (p : PosJ D N x) : Option Col :=
  if (tr d).out = .acc then lookCol (D.cols (tr d).seen p.1) p.2.val else none

omit [Fintype Dr] [Nonempty Dr] [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **On accepted draws the extractor's table is the zero-filled words.** -/
theorem firstTable_opDJ (D : Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (x : Msg) (tr : Dr → Trace) (bs : List Dr)
    (hacc : ∀ b ∈ bs, AccD tr b) (p : PosJ D N x) :
    firstTable (opDJ D N x tr) (fun _ => 0) bs p =
      (lookCol ((bs.map fun d => (tr d).seen).flatMap fun v => D.cols v p.1) p.2.val).getD 0 := by
  unfold firstTable
  rw [List.flatMap_map,
    findSome?_lookCol _ (fun d => D.cols (tr d).seen p.1) p.2.val bs fun b hb => by
      unfold opDJ
      simp only [show (tr b).out = .acc from hacc b hb, ↓reduceIte]]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **One wait's bound over draws, from every table's bound**: `oneWaitD` at a statement of several tables, the
positions every table's word's (`PosJ`). -/
theorem oneWaitDJ (D : Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (x : Msg) (tr : Dr → Trace) {s : Flock.Stmt}
    (hs : D.parse x = some s) (Ok : Msg → ℕ → Col → Prop) (hz : ∀ j, Ok x j 0)
    (hok : ∀ d j q u, (tr d).out = .acc → j < nTab D x → q < N x j → lookCol (D.cols (tr d).seen j) q = some u →
      Ok x j u)
    (εc : ℝ) (hεc : 0 ≤ εc)
    (hb : ∀ j, j < nTab D x → ∀ C : Fin (N x j) → Col, (∀ p, Ok x j (C p)) → ¬ GoodTableAt D N x j C →
      accPrD tr ≤ εc + Pr fun d => (tr d).out = .acc ∧
        ∃ (p : Fin (N x j)) (u : Col), lookCol (D.cols (tr d).seen j) p.val = some u ∧ u ≠ C p)
    (k : ℕ) (hk : 1 ≤ k) :
    accPrD tr * (accFailD tr k (Extracts D x) / accPrD tr ^ k) ≤
      εc + (Fintype.card (PosJ D N x) : ℝ) * accPrD tr / (Real.exp 1 * k) +
        Real.sqrt (k * Pr fun s : Dr × Dr => ∃ p, ConflictAt (opDJ D N x tr) s p) := by
  classical
  set Xt : List Dr → Prop := fun bs =>
    (∀ b ∈ bs, AccD tr b) → Extracts D x (bs.map fun d => (tr d).seen) with hXt
  have hXb : ∀ bs, ¬ Xt bs → expect (ind (AccD tr)) (drawGame Dr) (drawS Dr) ≤
      εc + expect (ind fun d => AccD tr d ∧
        ∃ p, Off (opDJ D N x tr) (firstTable (opDJ D N x tr) (fun _ => 0) bs) d p) (drawGame Dr) (drawS Dr) := by
    intro bs hnX
    simp only [hXt, Classical.not_imp] at hnX
    obtain ⟨hall, hnot⟩ := hnX
    obtain ⟨j, hj, hnj⟩ : ∃ j < s.tables, ¬ ExtractsAt D x (bs.map fun d => (tr d).seen) j := by
      by_contra hc
      push Not at hc
      exact hnot ⟨s, hs, hc⟩
    have hjn : j < nTab D x := (nTab_of_parse hs).symm ▸ hj
    have hCe : (fun p : Fin (N x j) => firstTable (opDJ D N x tr) (fun _ => 0) bs ⟨⟨j, hjn⟩, p⟩) =
        fun p => (lookCol ((bs.map fun d => (tr d).seen).flatMap fun v => D.cols v j) p.val).getD 0 :=
      funext fun p => firstTable_opDJ D N x tr bs hall ⟨⟨j, hjn⟩, p⟩
    have hgood : ¬ GoodTableAt D N x j
        fun p => firstTable (opDJ D N x tr) (fun _ => 0) bs ⟨⟨j, hjn⟩, p⟩ := fun hg =>
      hnj (extractsAt_of_goodAt D N x j _ (hCe ▸ hg))
    have hOk : ∀ p, Ok x j (firstTable (opDJ D N x tr) (fun _ => 0) bs ⟨⟨j, hjn⟩, p⟩) := by
      intro p
      rw [congrFun hCe p]
      cases hl : lookCol ((bs.map fun d => (tr d).seen).flatMap fun v => D.cols v j) p.val with
      | none => exact hz j
      | some u =>
        rw [List.flatMap_map] at hl
        obtain ⟨b, hb', hu⟩ := lookCol_flatMap_some (fun d => D.cols (tr d).seen j) p.val u bs hl
        exact hok b j p.val u (hall b hb') hjn p.isLt hu
    rw [← accPrD_eq, expect_drawGame, ← Claims.Bridge.Pr_eq_Ex]
    refine (hb j hjn _ hOk hgood).trans (add_le_add le_rfl (Pr_mono fun d ⟨ha, p, u, hu, hne⟩ => ?_))
    have e : opDJ D N x tr d ⟨⟨j, hjn⟩, p⟩ = some u := by
      unfold opDJ
      rw [ite_eq_left ha]
      exact hu
    refine ⟨ha, ⟨⟨j, hjn⟩, p⟩, ?_⟩
    show (opDJ D N x tr d ⟨⟨j, hjn⟩, p⟩).isSome ∧ opDJ D N x tr d ⟨⟨j, hjn⟩, p⟩ ≠ some _
    rw [e]
    exact ⟨rfl, fun h => hne (Option.some_injective _ h)⟩
  have hj := joint_acc_le (opDJ D N x tr) (fun _ => 0) (drawGame Dr) (drawS Dr) (AccD tr) Xt εc hεc hXb k hk
  have hlhs : accPrD tr * (accFailD tr k (Extracts D x) / accPrD tr ^ k) =
      expect (ind (AccD tr)) (drawGame Dr) (drawS Dr) *
        expectAcc (drawGame Dr) (drawS Dr) (AccD tr) k (ind fun bs => ¬ Xt bs) := by
    rw [oneWaitD_lhs_eq]
    congr 1
    unfold expectAcc
    congr 1
    refine congrArg (expectN _ _ _) (funext fun bs => ?_)
    rw [prod_map_ind]
    by_cases hall : ∀ b ∈ bs, AccD tr b
    · congr 1
      unfold ind
      have : (¬ Xt bs) ↔ ¬ Extracts D x (bs.map fun d => (tr d).seen) := by
        rw [hXt]
        exact not_congr ⟨fun h => h hall, fun h _ => h⟩
      beta_reduce
      split_ifs with h1 h2 h2 <;> first | rfl | (exfalso; tauto)
    · rw [ite_eq_right hall, zero_mul, zero_mul]
  have hE₂ : expect₂ (fun a a' => ind (fun y : Dr × Dr => ∃ p, ConflictAt (opDJ D N x tr) y p) (a, a'))
      (drawGame Dr) (drawS Dr) = Pr fun s : Dr × Dr => ∃ p, ConflictAt (opDJ D N x tr) s p := by
    unfold expect₂
    simp only [expect_drawGame]
    rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  rw [hE₂, ← accPrD_eq] at hj
  rw [hlhs, ← accPrD_eq]
  exact hj

end OneWaitDJ

/-! ## The chain at the abort law at every table, counted per round -/

section TableBound

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **Every table's bound with escapes on accepted runs counted per round**: `Flock.TableBoundJAccAtR` at every
statement. -/
def TableBoundJOnAccR (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop) :
    Prop :=
  ∀ x, Flock.TableBoundJAccAtR H D N Ok x

/-- **Every table's bound at the abort law, counted per round** (`EscapesD.FitsForkR`). -/
def TableBoundJAbortR (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop) :
    Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
    (keep : Ω → (Fin (D.V.k x) → Coin) → A → Bool),
    ∃ e : EscapesD H (D.V.k x) (abortRun H P D.V x keep), e.FitsForkR (D.weight x) (D.k x) ∧
      ∀ (ω : Ω) (j : ℕ), j < nTab D x → ∀ C : Fin (N x j) → Col, (∀ p, Ok x j (C p)) → ¬ GoodTableAt D N x j C →
        accPrD (abortRun H P D.V x keep ω) ≤ D.εtable x + e.term ω +
          Pr fun d : (Fin (D.V.k x) → Coin) × A => (abortRun H P D.V x keep ω d).out = .acc ∧
            ∃ (p : Fin (N x j)) (u : Col),
              lookCol (D.cols (abortRun H P D.V x keep ω d).seen j) p.val = some u ∧ u ≠ C p

/-- **Every table's bound at the abort law counted per round, from every table's bound with escapes on accepted runs
counted per round**: `tableBoundθAbortR_of_onAccR` at each table. -/
theorem tableBoundJAbortR_of_onAccR (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (Ok : Msg → ℕ → Col → Prop) (decC : Msg → Msg → Coin) (htab : TableBoundJOnAccR H D N Ok)
    (hVk : ∀ x, 0 < D.V.k x) (hpc : CoinsShown H D.V decC) (hlast : AccLast H D.V) (hfree : DecideFree H D.V)
    (href : Refusable H D.V) : TableBoundJAbortR H D N Ok := by
  classical
  intro x Ω _ _ P A _ _ keep
  obtain ⟨m₀, hm₀⟩ := href x
  obtain ⟨e, ⟨hon, hon₁⟩, hfit, hb⟩ :=
    htab x (Ω × ((Fin (D.V.k x) → Coin) → Coin × A)) (refuser P x (D.V.k x) decC keep m₀)
  refine ⟨Escapes.toAbortD e, hfit, fun ω j hj C hC hg => ?_⟩
  exact accPrD_le_of_refuser H P D.V x decC (hVk x) hpc hlast hfree keep m₀ hm₀ e hon hon₁ (D.εtable x)
    (fun s => ∃ (p : Fin (N x j)) (u : Col), lookCol (D.cols s j) p.val = some u ∧ u ≠ C p) ω
    (fun g => hb (ω, g) j hj C hC hg)

/-- **Flock's one-wait bound at the abort law, from every table's bound there counted per round**: when the words
together are no longer than `N₀` and the statement parses wherever some run accepts, `oneWaitAbort_of_tableBoundR` at
every table's positions (`oneWaitDJ`). -/
theorem oneWaitAbort_of_tableBoundJR (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (hN : ∀ x, ((∑ j : Fin (nTab D x), N x j : ℕ) : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x)
    (hVk : ∀ x, 0 < D.V.k x) (hk : ∀ x, 1 ≤ D.k x ∨ NoAcc H D.V x)
    (hp : ∀ x, (∃ s, D.parse x = some s) ∨ NoAcc H D.V x) (Ok : Msg → ℕ → Col → Prop) (hz : ∀ x j, Ok x j 0)
    (hok : ColsOkJ H D N Ok) (hconf : ConflictTiedJ H D N) (htab : TableBoundJAbortR H D N Ok) :
    OneWaitAbort H D := by
  classical
  intro x Ω _ _ P A _ _ keep
  obtain ⟨e₀, hfit, hb⟩ := htab x Ω P A keep
  set j₀ : Fin (D.V.k x) := ⟨0, hVk x⟩
  set ev : Ω → (Fin (D.V.k x) → Coin) × A → (Fin (D.V.k x) → Coin) × A → Prop :=
    fun ω d d' => ∃ p, ConflictAt (opDJ D N x (abortRun H P D.V x keep ω)) (d, d') p with hev
  have tied : ∀ ω d d', ev ω d d' → TiedColl H ((abortRun H P D.V x keep ω d).vLog ++
      (abortRun H P D.V x keep ω (forkDraw j₀ d d')).vLog) := by
    intro ω d d' ⟨p, h1, h2, h3⟩
    have hfd : forkDraw j₀ d d' = d' := by
      show (forkCoins j₀ d.1 d'.1, d'.2) = d'
      rw [Claims.Bridge.forkCoins_zero j₀ rfl]
    rw [hfd]
    simp only [opDJ] at h1 h2 h3
    split_ifs at h1 h2 h3 with ha ha'
    · obtain ⟨hk1, hr1⟩ := abortRun_acc_iff.1 ha
      obtain ⟨hk2, hr2⟩ := abortRun_acc_iff.1 ha'
      obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
      obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 h2
      have hne : u ≠ v := fun huv => h3 (by rw [hu, hv, huv])
      rw [abortRun_of_keep hk1] at hu ⊢
      rw [abortRun_of_keep hk2] at hv ⊢
      exact hconf x Ω P ω d.1 d'.1 p.1.val p.2.val u v hr1 hr2 p.1.isLt p.2.isLt hu hv hne
    all_goals simp at h1 h2
  refine ⟨(e₀.addFork (D.k x) (Nat.cast_nonneg _) j₀ ev tied).mergeR,
    (e₀.fitsR_addFork (D.weight x) (D.k x) (Nat.cast_nonneg _) j₀ rfl ev tied hfit).fits_mergeR, fun ω hε => ?_⟩
  have hacc : ¬ NoAcc H D.V x := fun hno => by
    have : accPrD (abortRun H P D.V x keep ω) = 0 := by
      unfold accPrD
      rw [Pr_congr (q := fun _ => False) fun d =>
        ⟨fun h => hno Ω P ω d.1 (abortRun_acc_iff.1 h).2, False.elim⟩, Pr_false]
    linarith
  have hk1 : 1 ≤ D.k x := (hk x).resolve_right hacc
  obtain ⟨s, hs⟩ := (hp x).resolve_right hacc
  have hokD : ∀ d j q u, (abortRun H P D.V x keep ω d).out = .acc → j < nTab D x → q < N x j →
      lookCol (D.cols (abortRun H P D.V x keep ω d).seen j) q = some u → Ok x j u := by
    intro d j q u ha hj hq hu
    obtain ⟨hkeep, hr⟩ := abortRun_acc_iff.1 ha
    rw [abortRun_of_keep hkeep] at hu
    exact hok x Ω P ω d.1 j q u hr hj hq hu
  have hεc : 0 ≤ D.εtable x + e₀.term ω := add_nonneg (hεt x) (e₀.term_nonneg ω)
  have hj := oneWaitDJ D N x (abortRun H P D.V x keep ω) hs Ok (hz x) hokD _ hεc
    (fun j hj C hC hg => hb ω j hj C hC hg) (D.k x) hk1
  have hE : (Pr fun s : ((Fin (D.V.k x) → Coin) × A) × ((Fin (D.V.k x) → Coin) × A) =>
      ∃ p, ConflictAt (opDJ D N x (abortRun H P D.V x keep ω)) s p) =
      Pr fun s : ((Fin (D.V.k x) → Coin) × A) × ((Fin (D.V.k x) → Coin) × A) => ev ω s.1 s.2 := rfl
  rw [hE] at hj
  have hm := (e₀.addFork (D.k x) (Nat.cast_nonneg _) j₀ ev tied).term_le_mergeR ω
  rw [EscapesD.term_addFork] at hm
  have hrate : (Fintype.card (PosJ D N x) : ℝ) * accPrD (abortRun H P D.V x keep ω) / (Real.exp 1 * D.k x) ≤
      D.N₀ x / (Real.exp 1 * D.k x) * accPrD (abortRun H P D.V x keep ω) := by
    rw [Fintype.card_sigma, div_mul_eq_mul_div]
    simp only [Fintype.card_fin]
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (hN x) hε.le) (by positivity)
  linarith

/-- **Flock as the inner protocol meets `Inner.KSAbort`, from every table's bound counted per round**:
`KSAbort_of_tableBoundOnAccR` at every number of tables, the statement parsing wherever some run accepts. -/
theorem KSAbort_of_tableBoundJOnAccR (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (Ok : Msg → ℕ → Col → Prop) (decC : Msg → Msg → Coin)
    (hN : ∀ x, ((∑ j : Fin (nTab D x), N x j : ℕ) : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x)
    (hVk : ∀ x, 0 < D.V.k x) (hK : ∀ x, 0 < D.cap x) (hk : ∀ x, (8 ≤ D.k x ∧ Splits D x) ∨ NoAcc H D.V x)
    (hp : ∀ x, (∃ s, D.parse x = some s) ∨ NoAcc H D.V x) (hz : ∀ x j, Ok x j 0)
    (hok : ColsOkJ H D N Ok) (hconf : ConflictTiedJ H D N) (htab : TableBoundJOnAccR H D N Ok)
    (hpc : CoinsShown H D.V decC) (hlast : AccLast H D.V) (hfree : DecideFree H D.V) (href : Refusable H D.V)
    (decode : List (ℕ × Col) → List T) (hdec : decOK D decode) : (Inner.ofFlock D decode).KSAbort H :=
  KSAbort_of_oneWaitAbort H D hK hk hεt
    (oneWaitAbort_of_tableBoundJR H D N hN hεt hVk (fun x => (hk x).imp (fun h => by omega) id) hp Ok hz hok hconf
      (tableBoundJAbortR_of_onAccR H D N Ok decC htab hVk hpc hlast hfree href)) decode hdec

omit [Nonempty Coin] in
/-- **`TableBoundJOnAccR` through `f`, by padding back**, as `tableBoundJR_mapCols`. -/
theorem tableBoundJOnAccR_mapCols (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (f : Col → Col)
    (hf : f 0 = 0) (Ok : Msg → ℕ → Col → Prop) (hok : ColsOkJ H D N Ok)
    (hinj : ∀ x j u v, Ok x j u → Ok x j v → f u = f v → u = v) (htab : TableBoundJOnAccR H D N Ok) :
    TableBoundJOnAccR H (D.mapCols f) N (OkThroughJ f Ok) := by
  intro x Ω _ _ P
  obtain ⟨e, hon, hfit, hb⟩ := htab x Ω P
  refine ⟨e, hon, hfit, fun ω j hj C' hC' hg' => ?_⟩
  choose C hC hfC using hC'
  have hg : ¬ GoodTableAt D N x j C := fun h => hg' (by
    have := goodTableAt_mapCols D N x j f hf C h
    rwa [show (fun p => f (C p)) = C' from funext hfC] at this)
  refine (hb ω j hj C hC hg).trans (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne⟩ => ⟨ha, p, f u, ?_, ?_⟩))
  · show lookCol ((D.cols (run H P D.V x ω r).seen j).map fun q => (q.1, f q.2)) p.val = some (f u)
    rw [lookCol_map, hu]
    rfl
  · rw [← hfC p]
    exact fun he => hne (hinj x j u (C p) (hok x Ω P ω r j p.val u ha hj p.isLt hu) (hC p) he)

end TableBound

end Claims.Recursion

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **Every masked table's bound with escapes on accepted runs counted per round at the trimmed record**, at its
trimmed columns or `0`, from the bound at full-width tables at the untrimmed record (`zk_tableBoundJR_trim`). -/
theorem zk_tableBoundJOnAccR_trim (rel : Readable.Parsed → Readable.Relation)
    (htab : TableBoundJOnAccR sha512 (deployZk rel) (wordLenJ true) (wideJ true)) :
    TableBoundJOnAccR sha512 (deployZkTrim rel) (wordLenJ true)
      (Claims.Flock.orZeroJ (Claims.Flock.OkThroughJ trimCol (wideJ true))) := by
  have h := tableBoundJOnAccR_mapCols sha512 (deployZk rel) (wordLenJ true) trimCol rfl (wideJ true)
    (zk_colsOkJ sha512 rel) (wideJ_trimCol_inj true) htab
  intro x Ω _ _ P
  obtain ⟨e, hon, hfit, hb⟩ := h x Ω P
  refine ⟨e, hon, hfit, fun ω j hj C hC hg => hb ω j hj C (fun p => ?_) hg⟩
  rcases hC p with h0 | hp
  · rw [h0]
    exact okThroughJ_nil true x j (Nat.lt_of_le_of_lt (Nat.zero_le _) p.isLt)
  · exact hp

/-- **The masked (`--zk`) record the claims read meets `Inner.KSAbort`**, at SHA-512 at the deployed relation, at
every number of tables and every decoder within the radius, from every table's bound with escapes on accepted runs
counted per round over full-width tables at the untrimmed record. -/
theorem zk_KSAbortJR
    (htab : TableBoundJOnAccR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true))
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace)
    (hdec : decOK (deployZkTrim Readable.relation) decode) :
    (Inner.ofFlock (deployZkTrim Readable.relation) decode).KSAbort sha512 :=
  KSAbort_of_tableBoundJOnAccR sha512 (deployZkTrim Readable.relation) (wordLenJ true)
    (Claims.Flock.orZeroJ (Claims.Flock.OkThroughJ trimCol (wideJ true))) decCoin
    (fun x => sum_wordLenJ_le true x _) (fun x => εtable_nonneg true x) (V_k_pos true) (fun _ => cap_pos)
    (fun x => (k_range_zk Readable.relation x).elim (fun hk => (zk_splits_trim x).imp (fun h => ⟨hk, h⟩) id) .inr)
    (zk_parses Readable.relation) (fun _ _ => .inl rfl)
    (Claims.Flock.colsOkJ_orZero _ _ _ _
      (Claims.Flock.colsOkJ_mapCols _ _ _ _ _ (zk_colsOkJ sha512 Readable.relation)))
    (Claims.Flock.conflictTiedJ_mapCols _ _ _ _ (zk_conflictTiedJ Readable.relation))
    (zk_tableBoundJOnAccR_trim Readable.relation htab)
    (coinsShown_V sha512 true) (accLast_V_sha512 true) (decideFree_V sha512 true) (refusable_V sha512 true)
    decode hdec

/-- **KS of the recursion over the masked (`--zk`) record the claims read as the inner protocol**, at SHA-512 at the
deployed relation, at every number of tables: `zk_KSFlockR` with `Inner.KSAbort` from `zk_KSAbortJR`. -/
theorem zk_KSFlockJR (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (htab : TableBoundJOnAccR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true))
    (hRef : Refuses sha512 D) (hRounds : ∀ x, (deployZkTrim Readable.relation).V.k (xI D x) ≤ D.V.k x)
    (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hS : SessionAccepts D) (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployZkTrim Readable.relation) DO :=
  (KSFlockA_iff_KSFlock _ _ _ _).1 fun decI h => KS_of_flockKS sha512 D _ DO opening decCoin hF
    (zk_KSAbortJR htab decI h) hRef hRounds hN hε (hRd decI) hS (hR decI) (hV decI) (hT decI) (hL decI)
    (hO decI) (hAcc decI) (coinsShown_V sha512 true) hN0

/-- **Every masked table's bound with escapes on accepted runs counted per round**, at full-width columns, from a
simulator of each prover at the statements some run accepts (`zkXferWJ_of_simWJ`, `zk_tableBoundJAccAtR`). -/
theorem zk_tableBoundJOnAccR_of_simWJ (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimWJ x) :
    TableBoundJOnAccR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) :=
  zk_tableBoundJAccAtR fun x hno => zkLiveAccWJ_of_xferWJ hno (zkXferWJ_of_simWJ (hs x hno))

/-- **The masked record the claims read meets `Inner.KSAbort`** at the deployed relation, at every number of tables,
from a simulator of each prover at the statements some run accepts. -/
theorem zk_KSAbort_of_simWJ_tables (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimWJ x)
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace)
    (hdec : decOK (deployZkTrim Readable.relation) decode) :
    (Inner.ofFlock (deployZkTrim Readable.relation) decode).KSAbort sha512 :=
  zk_KSAbortJR (zk_tableBoundJOnAccR_of_simWJ hs) decode hdec

/-- **KS of the recursion over the masked record the claims read** at the deployed relation, at every number of
tables, from a simulator of each prover at the statements some run accepts (`zk_KSFlockJR`). -/
theorem zk_KSFlock_of_simWJ_tables (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimWJ x) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployZkTrim Readable.relation).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hS : SessionAccepts D) (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployZkTrim Readable.relation) DO :=
  zk_KSFlockJR D DO opening hF (zk_tableBoundJOnAccR_of_simWJ hs) hRef hRounds hN hε hRd hS hR hV hT hL hO hAcc hN0

end FlockDeploy.KS
