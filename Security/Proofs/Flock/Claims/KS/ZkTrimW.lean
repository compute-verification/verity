import Proofs.Flock.Claims.KS.ZkXferW
import Proofs.Flock.Claims.KS.Trim

/-!
# The masked record the claims read: KS at every number of tables, from the full-width bound counted per round

`Trim` at several tables. Every table's bound counted per round carries to the trimmed record by padding back
(`Flock.tableBoundJR_mapCols`, as `Flock.tableBoundθ_mapCols` at one table), with `ColsOkJ` and `ConflictTiedJ`
through the trim (`Flock.colsOkJ_mapCols`, `Flock.conflictTiedJ_mapCols`). Where table `j`'s word has a position, `[]`
is the trim of its zero column (`FlockDeploy.KS.okThroughJ_nil`), so an unopened position's `0` is a trimmed
full-width column and the bound needs no `ExtendJ` (`FlockDeploy.KS.zk_tableBoundJR_trim`). So `Flock.KS` at the
trimmed masked record follows from every table's bound counted per round at full-width columns at the untrimmed one
(`FlockDeploy.KS.zk_KS_trim_of_tableBoundJR`), so from the live sessions' bound off the table on a witness lane where
some run accepts (`FlockDeploy.KS.zk_KS_trim_of_liveAccWJ`), so from `ZkXferWJ` there
(`FlockDeploy.KS.zk_KS_trim_of_xferWJ`).
-/

namespace Claims.Flock

open FlockSoundness

section Tables

variable {Coin Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **A column through `f` of one `Ok` admits at table `j`.** -/
def OkThroughJ (f : Col → Col) (Ok : Msg → ℕ → Col → Prop) : Msg → ℕ → Col → Prop :=
  fun x j u => ∃ v, Ok x j v ∧ f v = u

omit [Fintype Coin] [DecidableEq Col] [Zero Col] in
/-- **An accepted run opens columns through `f` of ones `Ok` admits** at each table when it opens only columns `Ok`
admits. -/
theorem colsOkJ_mapCols (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (f : Col → Col)
    (Ok : Msg → ℕ → Col → Prop) (h : ColsOkJ H D N Ok) : ColsOkJ H (D.mapCols f) N (OkThroughJ f Ok) := by
  intro x Ω P ω r j q u ha hj hq hu
  rw [mapCols_cols, lookCol_map, Option.map_eq_some_iff] at hu
  obtain ⟨v, hv, rfl⟩ := hu
  exact ⟨v, h x Ω P ω r j q v ha hj hq hv, rfl⟩

omit [Fintype Coin] [DecidableEq Col] [Zero Col] in
/-- **Conflicts through `f` are conflicts** at each table: two columns with two images are two columns. -/
theorem conflictTiedJ_mapCols (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (f : Col → Col)
    (h : ConflictTiedJ H D N) : ConflictTiedJ H (D.mapCols f) N := by
  intro x Ω P ω r r' j q u v ha ha' hj hq hu hv huv
  rw [mapCols_cols, lookCol_map, Option.map_eq_some_iff] at hu hv
  obtain ⟨u₀, hu₀, rfl⟩ := hu
  obtain ⟨v₀, hv₀, rfl⟩ := hv
  exact h x Ω P ω r r' j q u₀ v₀ ha ha' hj hq hu₀ hv₀ fun he => huv (by rw [he])

omit [Fintype Coin] [Inhabited Coin] in
/-- **A table at table `j`'s word that extracts extracts through `f`**, when `f` keeps `0`. -/
theorem goodTableAt_mapCols (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (x : Msg) (j : ℕ) (f : Col → Col)
    (hf : f 0 = 0) (C : Fin (N x j) → Col) (h : GoodTableAt D N x j C) :
    GoodTableAt (D.mapCols f) N x j fun p => f (C p) := by
  obtain ⟨M, t, ht, hrel, hle, hag⟩ := h
  refine ⟨M, t, ht, hrel, hle, hag.trans ?_⟩
  have hsub : ((Finset.univ : Finset (Fin (D.len M))).filter fun i => D.enc M i.val = extTab C i.val) ⊆
      (Finset.univ : Finset (Fin (D.len M))).filter fun i =>
        (D.mapCols f).enc M i.val = extTab (fun p => f (C p)) i.val := by
    refine Finset.monotone_filter_right _ fun i _ hi => ?_
    show f (D.enc M i.val) = extTab (fun p => f (C p)) i.val
    rw [extTab_comp f hf, hi]
  exact_mod_cast Finset.card_le_card hsub

/-- **Every table's bound counted per round through `f`, by padding back**: at tables of columns through `f` of ones
`Ok` admits, from the bound at the tables they came from, with the same escapes, when `f` keeps `0`, accepted runs
open only columns `Ok` admits, and `f` is one to one on the columns `Ok` admits at each table. -/
theorem tableBoundJR_mapCols (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (f : Col → Col)
    (hf : f 0 = 0) (Ok : Msg → ℕ → Col → Prop) (hok : ColsOkJ H D N Ok)
    (hinj : ∀ x j u v, Ok x j u → Ok x j v → f u = f v → u = v) (htab : TableBoundJR H D N Ok) :
    TableBoundJR H (D.mapCols f) N (OkThroughJ f Ok) := by
  intro x Ω _ _ P
  obtain ⟨e, hfit, hb⟩ := htab x Ω P
  refine ⟨e, hfit, fun ω j hj C' hC' hg' => ?_⟩
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

end Tables

end Claims.Flock

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

namespace Gen

variable [S : ZkSetup]

/-- **`trimCol` is one to one on table `j`'s full-width columns.** -/
theorem wideJ_trimCol_inj (zk : Bool) (x : Claims.Msg) (j : ℕ) (u v : List 𝔽) (hu : wideJ zk x j u)
    (hv : wideJ zk x j v) (h : trimCol u = trimCol v) : u = v := by
  unfold wideJ at hu hv
  rcases hq : (tables zk x)[j]? with _ | ⟨a, st⟩
  · simp only [hq] at hu
  · simp only [hq] at hu hv
    exact trimCol_inj (hu.trans hv.symm) h

end Gen

/-- **`trimCol` is one to one on table `j`'s full-width columns.** -/
theorem wideJ_trimCol_inj (zk : Bool) (x : Claims.Msg) (j : ℕ) (u v : List 𝔽) (hu : wideJ zk x j u)
    (hv : wideJ zk x j v) (h : trimCol u = trimCol v) : u = v :=
  Gen.wideJ_trimCol_inj (S := S₀) (zk := zk) (x := x) (j := j) (u := u) (v := v) (hu := hu) (hv := hv) (h := h)

namespace Gen

variable [S : ZkSetup]

/-- **`[]` is a trimmed full-width column of table `j`** where its word has a position: its zero column's trim. -/
theorem okThroughJ_nil (zk : Bool) (x : Claims.Msg) (j : ℕ) (h : 0 < wordLenJ zk x j) :
    Claims.Flock.OkThroughJ trimCol (wideJ zk) x j [] := by
  unfold wordLenJ at h
  unfold Claims.Flock.OkThroughJ wideJ
  rcases hq : (tables zk x)[j]? with _ | ⟨a, st⟩
  · simp only [hq] at h
    exact absurd h (lt_irrefl 0)
  · simp only
    exact ⟨_, List.length_replicate, trimCol_replicate _⟩

end Gen

/-- **`[]` is a trimmed full-width column of table `j`** where its word has a position: its zero column's trim. -/
theorem okThroughJ_nil (zk : Bool) (x : Claims.Msg) (j : ℕ) (h : 0 < wordLenJ zk x j) :
    Claims.Flock.OkThroughJ trimCol (wideJ zk) x j [] :=
  Gen.okThroughJ_nil (S := S₀) (zk := zk) (x := x) (j := j) (h := h)

namespace Gen

variable [S : ZkSetup]

/-- **Every masked table's bound counted per round at the trimmed record**, at its trimmed columns or `0`, from the
bound at full-width tables at the untrimmed record: `0` is a trimmed full-width column at a position of table `j`'s
word. -/
theorem zk_tableBoundJR_trim (rel : Readable.Parsed → Readable.Relation)
    (htab : Claims.Flock.TableBoundJR sha512 (deployZk rel) (wordLenJ true) (wideJ true)) :
    Claims.Flock.TableBoundJR sha512 (deployZkTrim rel) (wordLenJ true)
      (Claims.Flock.orZeroJ (Claims.Flock.OkThroughJ trimCol (wideJ true))) := by
  have h := Claims.Flock.tableBoundJR_mapCols sha512 (deployZk rel) (wordLenJ true) trimCol rfl (wideJ true)
    (zk_colsOkJ sha512 rel) (wideJ_trimCol_inj true) htab
  intro x Ω _ _ P
  obtain ⟨e, hfit, hb⟩ := h x Ω P
  refine ⟨e, hfit, fun ω j hj C hC hg => hb ω j hj C (fun p => ?_) hg⟩
  rcases hC p with h0 | hp
  · rw [h0]
    exact okThroughJ_nil true x j (Nat.lt_of_le_of_lt (Nat.zero_le _) p.isLt)
  · exact hp

end Gen

/-- **Every masked table's bound counted per round at the trimmed record**, at its trimmed columns or `0`, from the
bound at full-width tables at the untrimmed record: `0` is a trimmed full-width column at a position of table `j`'s
word. -/
theorem zk_tableBoundJR_trim (rel : Readable.Parsed → Readable.Relation)
    (htab : Claims.Flock.TableBoundJR sha512 (deployZk rel) (wordLenJ true) (wideJ true)) :
    Claims.Flock.TableBoundJR sha512 (deployZkTrim rel) (wordLenJ true)
      (Claims.Flock.orZeroJ (Claims.Flock.OkThroughJ trimCol (wideJ true))) :=
  Gen.zk_tableBoundJR_trim (S := S₀) (rel := rel) (htab := htab)

namespace Gen

variable [S : ZkSetup]

/-- **The masked relation splits by table at the trimmed record wherever some run accepts.** -/
theorem zk_splits_trim (x : Claims.Msg) :
    Claims.Flock.Splits (deployZkTrim Readable.relation) x ∨
      Claims.Flock.NoAcc sha512 (deployZkTrim Readable.relation).V x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (V true) x
  · exact .inr hno
  · obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
    exact .inl (splits_of_inScope Claims.Flock.zkReadD (deployZkTrim Readable.relation) rfl rfl rfl hs hin)

end Gen

/-- **The masked relation splits by table at the trimmed record wherever some run accepts.** -/
theorem zk_splits_trim (x : Claims.Msg) :
    Claims.Flock.Splits (deployZkTrim Readable.relation) x ∨
      Claims.Flock.NoAcc sha512 (deployZkTrim Readable.relation).V x :=
  Gen.zk_splits_trim (S := S₀) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
every table's bound counted per round over full-width tables at the untrimmed record. -/
theorem zk_KS_trim_of_tableBoundJR
    (htab : Claims.Flock.TableBoundJR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Claims.Flock.KS_of_oneWait sha512 (deployZkTrim Readable.relation) (fun _ => cap_pos)
    (fun x => (k_range_zk Readable.relation x).elim (fun hk => (zk_splits_trim x).imp (fun h => ⟨hk, h⟩) id) .inr)
    (fun x => εtable_nonneg true x)
    (Claims.Flock.oneWait_of_tableBoundJR sha512 (deployZkTrim Readable.relation) (wordLenJ true)
      (fun x => sum_wordLenJ_le true x _) (fun x => εtable_nonneg true x) (V_k_pos true)
      (k_one_zk Readable.relation) (zk_parses Readable.relation) _ (fun _ _ => .inl rfl)
      (Claims.Flock.colsOkJ_orZero _ _ _ _
        (Claims.Flock.colsOkJ_mapCols _ _ _ _ _ (zk_colsOkJ sha512 Readable.relation)))
      (Claims.Flock.conflictTiedJ_mapCols _ _ _ _ (zk_conflictTiedJ Readable.relation))
      (zk_tableBoundJR_trim Readable.relation htab))

end Gen

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
every table's bound counted per round over full-width tables at the untrimmed record. -/
theorem zk_KS_trim_of_tableBoundJR
    (htab : Claims.Flock.TableBoundJR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Gen.zk_KS_trim_of_tableBoundJR (S := S₀) (htab := htab)

namespace Gen

variable [S : ZkSetup]

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the live sessions' bound off the table on a witness lane at the statements some run accepts (`ZkLiveAccWJ`). -/
theorem zk_KS_trim_of_liveAccWJ (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveAccWJ x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_trim_of_tableBoundJR (Claims.Flock.tableBoundJR_of_accAtR fun x => zk_tableBoundJAccAtR hl x)

end Gen

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the live sessions' bound off the table on a witness lane at the statements some run accepts (`ZkLiveAccWJ`). -/
theorem zk_KS_trim_of_liveAccWJ (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveAccWJ x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Gen.zk_KS_trim_of_liveAccWJ (S := S₀) (hl := hl)

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the live sessions' bound at the deployed decoders off the table on a witness lane at the statements some run accepts
(`ZkXferWJ`). -/
theorem zk_KS_trim_of_xferWJ (hx : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkXferWJ x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_trim_of_liveAccWJ fun x hno => zkLiveAccWJ_of_xferWJ hno (hx x hno)

namespace Gen

variable [S : ZkSetup]

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the live sessions' bound at the deployed decoders off the table on a witness lane at the statements some run accepts
(`ZkXferWJ`). -/
theorem zk_KS_trim_of_xferWJ (hx : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkXferWJ x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_trim_of_liveAccWJ fun x hno => zkLiveAccWJ_of_xferWJ hno (hx x hno)

end Gen

end FlockDeploy.KS
