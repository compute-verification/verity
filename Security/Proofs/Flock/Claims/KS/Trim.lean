import Proofs.Flock.Claims.KS.TableBound

/-!
# KS at the trimmed records, from the full-width obligations

Flock's claims read the deployed instances with their columns trimmed of trailing zeros (`FlockDeploy.deployPlainTrim`,
`FlockDeploy.deployZkTrim`: `Deploy.mapCols` at `FlockDeploy.trimCol`), so a position no rerun opened, `[]`, is the
zero column of every width. The three obligations of `Flock.oneWait_of_tableBound` carry over from the untrimmed
instance, where they are stated over full-width columns (`wide`):

* **The allowed columns** (`Flock.OkThrough`): a column through `f` of one `Ok` admits; at the records, a trimmed
  full-width column (`FlockDeploy.KS.trimWide`), `[]` among them (`FlockDeploy.KS.trimWide_nil`).
* `Flock.colsOk_mapCols`, `Flock.conflictTied_mapCols`: an accepted run's column through `f` is its column's, and two
  columns with two images are two columns.
* `Flock.tableBoundθ_mapCols`, by padding back: a table of trimmed full-width columns is the trim of one full-width
  table, which does not extract when the trimmed one does not (`Flock.goodTable_mapCols`), and a run that opens a
  column off the full-width table opens its trim off the trimmed one, as `trimCol` is one to one at one width
  (`FlockDeploy.trimCol_inj`).

So KS at the trimmed records at SHA-512 asks only the fixed-table bound over full-width tables at the untrimmed
instance (`FlockDeploy.KS.plain_KS_trim`, `FlockDeploy.KS.zk_KS_trim`), statement by statement at the accepted ones
(`FlockDeploy.KS.plain_KS_trim_of_acc`, `FlockDeploy.KS.zk_KS_trim_of_acc`).
-/

namespace Claims.Flock

open FlockSoundness

/-- **A column through `f` of one `Ok` admits.** -/
def OkThrough {Col : Type} (f : Col → Col) (Ok : Msg → Col → Prop) : Msg → Col → Prop :=
  fun x u => ∃ v, Ok x v ∧ f v = u

theorem lookCol_map {Col : Type} (f : Col → Col) (os : List (ℕ × Col)) (q : ℕ) :
    lookCol (os.map fun p => (p.1, f p.2)) q = (lookCol os q).map f := by
  induction os with
  | nil => rfl
  | cons a os ih =>
    unfold lookCol at ih ⊢
    rw [List.map_cons, List.find?_cons, List.find?_cons]
    by_cases h : a.1 = q
    · simp [h]
    · simp only [h, decide_false]
      exact ih

theorem mapCols_cols {Coin Col T : Type} (D : Deploy Coin Col T) (f : Col → Col) (s : Seen) :
    (D.mapCols f).cols s = (D.cols s).map fun p => (p.1, f p.2) := rfl

/-- **An accepted run opens columns through `f` of ones `Ok` admits** when it opens only columns `Ok` admits. -/
theorem colsOk_mapCols {Coin Col T : Type} [Inhabited Coin] [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T)
    (N : Msg → ℕ) (f : Col → Col) (Ok : Msg → Col → Prop) (h : ColsOk H D N Ok) :
    ColsOk H (D.mapCols f) N (OkThrough f Ok) := by
  intro x Ω P ω r q u ha hq hu
  rw [mapCols_cols, lookCol_map, Option.map_eq_some_iff] at hu
  obtain ⟨v, hv, rfl⟩ := hu
  exact ⟨v, h x Ω P ω r q v ha hq hv, rfl⟩

/-- **Conflicts through `f` are conflicts**: two columns with two images are two columns. -/
theorem conflictTied_mapCols {Coin Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin Col T)
    (N : Msg → ℕ) (f : Col → Col) (h : ConflictTied H D N) : ConflictTied H (D.mapCols f) N := by
  intro x Ω P ω r r' q u v ha ha' hq hu hv huv
  rw [mapCols_cols, lookCol_map, Option.map_eq_some_iff] at hu hv
  obtain ⟨u₀, hu₀, rfl⟩ := hu
  obtain ⟨v₀, hv₀, rfl⟩ := hv
  exact h x Ω P ω r r' q u₀ v₀ ha ha' hq hu₀ hv₀ fun he => huv (by rw [he])

theorem extTab_comp {Col : Type} [Zero Col] (f : Col → Col) (hf : f 0 = 0) {n : ℕ} (C : Fin n → Col) (q : ℕ) :
    extTab (fun p => f (C p)) q = f (extTab C q) := by
  unfold extTab
  split_ifs
  · rfl
  · exact hf.symm

/-- **A table that extracts extracts through `f`**, when `f` keeps `0`: a trace's code agrees with the table's image
wherever it agrees with the table. -/
theorem goodTable_mapCols {Coin Col T : Type} [DecidableEq Col] [Zero Col] (D : Deploy Coin Col T) (N : Msg → ℕ)
    (x : Msg) (f : Col → Col) (hf : f 0 = 0) (C : Fin (N x) → Col) (h : GoodTable D N x C) :
    GoodTable (D.mapCols f) N x fun p => f (C p) := by
  obtain ⟨M, s, hs, hrel, hle, hag⟩ := h
  refine ⟨M, s, hs, hrel, hle, hag.trans ?_⟩
  have hsub : ((Finset.univ : Finset (Fin (D.len M))).filter fun i => D.enc M i.val = extTab C i.val) ⊆
      (Finset.univ : Finset (Fin (D.len M))).filter fun i =>
        (D.mapCols f).enc M i.val = extTab (fun p => f (C p)) i.val := by
    refine Finset.monotone_filter_right _ fun i _ hi => ?_
    show f (D.enc M i.val) = extTab (fun p => f (C p)) i.val
    rw [extTab_comp f hf, hi]
  exact_mod_cast Finset.card_le_card hsub

/-- **The fixed-table bound through `f`, by padding back**: at a table of columns through `f` of ones `Ok` admits,
from the bound at the table they came from, when `f` keeps `0`, accepted runs open only columns `Ok` admits, and `f`
is one to one on them. -/
theorem tableBoundθ_mapCols {Coin Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (f : Col → Col) (hf : f 0 = 0) (Ok : Msg → Col → Prop)
    (hok : ColsOk H D N Ok) (hinj : ∀ x u v, Ok x u → Ok x v → f u = f v → u = v) (htab : TableBoundθ H D N Ok) :
    TableBoundθ H (D.mapCols f) N (OkThrough f Ok) := by
  intro x Ω _ _ P
  obtain ⟨e, hfit, hb⟩ := htab x Ω P
  refine ⟨e, hfit, fun ω C' hC' hg' => ?_⟩
  choose C hC hfC using hC'
  have hg : ¬ GoodTable D N x C := fun h => hg' (by
    have := goodTable_mapCols D N x f hf C h
    rwa [show (fun p => f (C p)) = C' from funext hfC] at this)
  refine (hb ω C hC hg).trans (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne⟩ => ⟨ha, p, f u, ?_, ?_⟩))
  · show lookCol ((D.cols (run H P D.V x ω r).seen).map fun q => (q.1, f q.2)) p.val = some (f u)
    rw [lookCol_map, hu]
    rfl
  · rw [← hfC p]
    exact fun he => hne (hinj x u (C p) (hok x Ω P ω r p.val u ha p.isLt hu) (hC p) he)

/-- **One wait's bound through `f`**, from the three obligations at `D` over the columns `Ok` admits: when `f` keeps
`0`, `Ok` admits a column `f` takes to `0`, and `f` is one to one on the columns `Ok` admits. -/
theorem oneWait_mapCols {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (f : Col → Col) (hf : f 0 = 0)
    (hN : ∀ x, (N x : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x) (hVk : ∀ x, 0 < D.V.k x)
    (hk : ∀ x, 1 ≤ D.k x ∨ NoAcc H D.V x) (Ok : Msg → Col → Prop) (hz : ∀ x, ∃ v, Ok x v ∧ f v = 0)
    (hinj : ∀ x u v, Ok x u → Ok x v → f u = f v → u = v) (hok : ColsOk H D N Ok) (hconf : ConflictTied H D N)
    (htab : TableBoundθ H D N Ok) : OneWait H (D.mapCols f) :=
  oneWait_of_tableBound H (D.mapCols f) N hN hεt hVk hk (OkThrough f Ok) hz (colsOk_mapCols H D N f Ok hok)
    (conflictTied_mapCols H D N f hconf) (tableBoundθ_mapCols H D N f hf Ok hok hinj htab)

end Claims.Flock

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **A trimmed full-width column**: a column of the code's width (`wide`) trimmed of its trailing zeros. -/
def trimWide (zk : Bool) : Claims.Msg → List 𝔽 → Prop := Claims.Flock.OkThrough trimCol (wide zk)

/-- **`[]` is a trimmed full-width column**: the zero column's trim. -/
theorem trimWide_nil (zk : Bool) (x : Claims.Msg) : trimWide zk x [] := by
  refine ⟨zeroCol zk x, rfl, ?_⟩
  unfold zeroCol
  split
  · exact trimCol_replicate _
  · rfl

theorem wide_trimCol_inj (zk : Bool) (x : Claims.Msg) (u v : List 𝔽) (hu : wide zk x u) (hv : wide zk x v)
    (h : trimCol u = trimCol v) : u = v :=
  trimCol_inj (hu.trans hv.symm) h

/-- **The plain record the claims read: one wait's bound** at SHA-512, from the fixed-table bound over full-width
tables at the untrimmed instance. -/
theorem plain_oneWait_trim (rel : Readable.Parsed → Readable.Relation)
    (htab : Claims.Flock.TableBoundθ sha512 (deployPlain rel) (wordLen false) (wide false)) :
    Claims.Flock.OneWait sha512 (deployPlainTrim rel) :=
  Claims.Flock.oneWait_mapCols sha512 (deployPlain rel) (wordLen false) trimCol rfl
    (fun x => (wordLen_eq_N₀ false x).le) (fun x => εtable_nonneg false x) (V_k_pos false) (k_one_plain rel)
    (wide false) (fun x => trimWide_nil false x) (wide_trimCol_inj false) (plain_colsOk sha512 rel)
    (plain_conflictTied rel) htab

/-- **The masked record the claims read: one wait's bound** at SHA-512, from the fixed-table bound over full-width
tables at the untrimmed instance. -/
theorem zk_oneWait_trim (rel : Readable.Parsed → Readable.Relation)
    (htab : Claims.Flock.TableBoundθ sha512 (deployZk rel) (wordLen true) (wide true)) :
    Claims.Flock.OneWait sha512 (deployZkTrim rel) :=
  Claims.Flock.oneWait_mapCols sha512 (deployZk rel) (wordLen true) trimCol rfl
    (fun x => (wordLen_eq_N₀ true x).le) (fun x => εtable_nonneg true x) (V_k_pos true) (k_one_zk rel)
    (wide true) (fun x => trimWide_nil true x) (wide_trimCol_inj true) (zk_colsOk sha512 rel)
    (zk_conflictTied rel) htab

/-- **The plain record the claims read: KS** at SHA-512, from the fixed-table bound over full-width tables at the
untrimmed instance. -/
theorem plain_KS_trim (rel : Readable.Parsed → Readable.Relation)
    (htab : Claims.Flock.TableBoundθ sha512 (deployPlain rel) (wordLen false) (wide false)) :
    Claims.Flock.KS sha512 (deployPlainTrim rel) :=
  Claims.Flock.KS_of_oneWait sha512 (deployPlainTrim rel) (fun _ => cap_pos) (k_range_plain rel)
    (fun x => εtable_nonneg false x) (plain_oneWait_trim rel htab)

/-- **The masked record the claims read: KS** at SHA-512, from the fixed-table bound over full-width tables at the
untrimmed instance. -/
theorem zk_KS_trim (rel : Readable.Parsed → Readable.Relation)
    (htab : Claims.Flock.TableBoundθ sha512 (deployZk rel) (wordLen true) (wide true)) :
    Claims.Flock.KS sha512 (deployZkTrim rel) :=
  Claims.Flock.KS_of_oneWait sha512 (deployZkTrim rel) (fun _ => cap_pos) (k_range_zk rel)
    (fun x => εtable_nonneg true x) (zk_oneWait_trim rel htab)

/-- **The plain record the claims read: KS** at SHA-512, from the fixed-table bound over full-width tables at each
statement some run accepts, which parses in scope and sets up. -/
theorem plain_KS_trim_of_acc (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s ins spec sts, ¬ Claims.Flock.NoAcc sha512 (V false) x → parse false rel x = some s →
      Claims.Flock.InScope false s → setupOf false x = some (ins, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployPlain rel) (wordLen false) (wide false) x) :
    Claims.Flock.KS sha512 (deployPlainTrim rel) :=
  plain_KS_trim rel (plain_tableBoundθ_of_acc rel _ h)

/-- **The masked record the claims read: KS** at SHA-512, from the fixed-table bound over full-width tables at each
statement some run accepts, which parses in scope and sets up. -/
theorem zk_KS_trim_of_acc (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s ins spec sts, ¬ Claims.Flock.NoAcc sha512 (V true) x → parse true rel x = some s →
      Claims.Flock.InScope true s → setupOf true x = some (ins, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployZk rel) (wordLen true) (wide true) x) :
    Claims.Flock.KS sha512 (deployZkTrim rel) :=
  zk_KS_trim rel (zk_tableBoundθ_of_acc rel _ h)

end FlockDeploy.KS
