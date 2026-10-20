import Proofs.Flock.Claims.KS.ZkFork0

/-!
# Flock's KS at the trimmed record and at the untrimmed one

A trimmed record (`FlockDeploy.deployZkTrim`, `FlockDeploy.deployPlainTrim`) is its untrimmed instance with `cols` and
`enc` through `FlockDeploy.trimCol` (`Deploy.mapCols`); every other field is the instance's, so `V`, `R`, `κ` and `γ`
are too. But `E` reads `cols` and `ListWithin` reads `enc`, so `Flock.KS` at the two records quantifies over different
decoders:

* **Untrimmed to trimmed holds.** `E` at `D.mapCols f` is `E` at `D` with its decoder after `f` on the opened columns
  (`Flock.TrimBridge.E_mapCols`), and a decoder within the radius at `D.mapCols f` is, after `f`, one within the radius
  at `D` when `f` keeps `0` (`Flock.TrimBridge.listWithin_comp`). So KS at `D` gives KS at `D.mapCols f` at the same
  `κ` and `γ` (`Flock.TrimBridge.KS_mapCols_of_KS`; at the records, `FlockDeploy.KS.TrimBridge.zk_KS_trim_of_KS` and
  `FlockDeploy.KS.TrimBridge.plain_KS_trim_of_KS`): the trimmed record is the weaker statement.
* **Trimmed to untrimmed does not follow from KS.** A position no rerun opened reads `[]` (`fill`), which equals no
  column but `[]` itself, while the trim of every zero column is `[]` (`Flock.TrimBridge.hole_agree`). At the untrimmed
  record a full-width trace agrees with the opened word at no unopened position, so `ListWithin` there admits decoders
  that miss a trace agreeing only through unopened zero columns, which the trimmed spec makes every decoder list, and
  KS at the untrimmed record asks extraction at those decoders too.
* **What closes it is a premise on the code, not an error term.** Both records' KS follow from the one bound at
  full-width tables (`Flock.TableBoundJR`): the trimmed directly, as `[]` is the trim of a full-width zero column, and
  the untrimmed with `Flock.ExtendJ` (a table that does not extract and has unopened positions extends to a full-width
  one that keeps its opened columns and does not extract), both at the same `κ` and `γ`
  (`Flock.TrimBridge.KS_both_of_tableBoundJR`). At the masked record: `FlockDeploy.KS.TrimBridge.zk_KS_both_of_tableBoundJR`,
  and at the premises `FlockDeploy.KS.zk_KS_of_kinds` takes, `FlockDeploy.KS.TrimBridge.zk_KS_both_of_kinds`.
-/

namespace Claims.Flock.TrimBridge

/-! ## The records' extractors and decoder specs -/

theorem firstWait_map {R W Op : Type} (k : ℕ) (opens : Claims.Seen → List Op) (g : Op → Op)
    (A : List Op → Option W) (check : W → Bool) (rss : List (List R)) :
    Claims.firstWait k (fun v => (opens v).map g) A check rss =
      Claims.firstWait k opens (fun os => A (os.map g)) check rss := by
  induction rss with
  | nil => rfl
  | cons rs rss ih =>
    simp only [Claims.firstWait]
    congr 1
    funext vs
    rw [← List.map_flatMap, ih]

theorem ERep_map {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] (V : Claims.Verifier Coin)
    (m k K : Claims.Msg → ℕ) (opens : Claims.Seen → List Op) (g : Op → Op) (A : Claims.Msg → List Op → Option W)
    (check : Claims.Msg → W → Bool) :
    Claims.ERep V m k K (fun v => (opens v).map g) A check =
      Claims.ERep V m k K opens (fun x os => A x (os.map g)) check := by
  unfold Claims.ERep
  simp only [firstWait_map]

theorem assemble_map {Col T : Type} (parse : Claims.Msg → Option Claims.Flock.Stmt)
    (table : Claims.Msg → ℕ → Option Claims.Flock.Stmt) (decode : List (ℕ × Col) → List T)
    (read : List T → Claims.Flock.Wit) (f : Col → Col) (x : Claims.Msg) (ops : List (ℕ → List (ℕ × Col))) :
    Claims.Flock.assemble parse table decode read x (ops.map fun o j => (o j).map fun p => (p.1, f p.2)) =
      Claims.Flock.assemble parse table (fun os => decode (os.map fun p => (p.1, f p.2))) read x ops := by
  have e : ∀ j, ((ops.map fun o j => (o j).map fun p => (p.1, f p.2)).flatMap fun o => o j) =
      (ops.flatMap fun o => o j).map fun p => (p.1, f p.2) := fun j => by
    rw [List.flatMap_map, List.map_flatMap]
  unfold Claims.Flock.assemble
  simp only [e]

/-- **`E` at `D.mapCols f` is `E` at `D` with its decoder after `f`** on the opened columns. -/
theorem E_mapCols {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (D : Claims.Flock.Deploy Coin Col T) (f : Col → Col) (decode : List (ℕ × Col) → List T) :
    Claims.Flock.E (D.mapCols f) decode = Claims.Flock.E D fun os => decode (os.map fun p => (p.1, f p.2)) := by
  show Claims.ERep D.V D.waits D.k D.cap (fun v => [D.cols v].map fun o j => (o j).map fun p => (p.1, f p.2))
      (Claims.Flock.assemble D.parse D.table decode D.read) (Claims.Flock.check D.parse) = _
  rw [ERep_map]
  unfold Claims.Flock.E
  simp only [assemble_map D.parse D.table decode D.read f]

theorem fill_map {Col : Type} [Zero Col] (f : Col → Col) (hf : f 0 = 0) (n : ℕ) (os : List (ℕ × Col))
    (i : Fin n) : Claims.Flock.fill n (os.map fun p => (p.1, f p.2)) i = f (Claims.Flock.fill n os i) := by
  have hp : ((fun p : ℕ × Col => p.1 == i.val) ∘ fun p : ℕ × Col => (p.1, f p.2)) = fun p => p.1 == i.val := rfl
  unfold Claims.Flock.fill
  rw [List.find?_map, hp]
  cases os.find? fun p => p.1 == i.val with
  | none => exact hf.symm
  | some p => rfl

/-- **A decoder within the radius at the code through `f` is, after `f`, one within the radius at the code**, when
`f` keeps `0`: a position where the code agrees with the opened word agrees through `f`. -/
theorem listWithin_comp {Col T : Type} [DecidableEq Col] [Zero Col] (len : T → ℕ) (enc : T → ℕ → Col)
    (θ : T → ℝ) (f : Col → Col) (hf : f 0 = 0) (decode : List (ℕ × Col) → List T)
    (h : Claims.Flock.ListWithin len (fun M q => f (enc M q)) θ decode) :
    Claims.Flock.ListWithin len enc θ fun os => decode (os.map fun p => (p.1, f p.2)) := by
  intro os M hag
  have hsub : ((Finset.univ : Finset (Fin (len M))).filter fun i => enc M i.val = Claims.Flock.fill (len M) os i) ⊆
      (Finset.univ : Finset (Fin (len M))).filter fun i =>
        f (enc M i.val) = Claims.Flock.fill (len M) (os.map fun p => (p.1, f p.2)) i := by
    refine Finset.monotone_filter_right _ fun i _ hi => ?_
    rw [fill_map f hf, hi]
  exact h _ M (hag.trans (by exact_mod_cast Finset.card_le_card hsub))

/-- **KS at a record gives KS at the record with its columns and code through `f`**, at the same `κ` and `γ`, when
`f` keeps `0`. -/
theorem KS_mapCols_of_KS {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (H : Claims.Msg → Claims.Msg) (D : Claims.Flock.Deploy Coin Col T) (f : Col → Col) (hf : f 0 = 0)
    (h : Claims.Flock.KS H D) : Claims.Flock.KS H (D.mapCols f) := by
  intro decode hlw
  rw [E_mapCols]
  exact h _ (listWithin_comp D.len D.enc D.θ f hf decode hlw)

/-! ## The unopened positions -/

theorem trimCol_eq_nil_iff {F : Type} [Zero F] [DecidableEq F] (u : List F) :
    FlockDeploy.trimCol u = [] ↔ ∀ a ∈ u, a = 0 := by
  induction u with
  | nil => exact ⟨fun _ a ha => (nomatch ha), fun _ => rfl⟩
  | cons a u ih =>
    rw [FlockDeploy.trimCol, List.forall_mem_cons]
    split_ifs with h
    · exact ⟨fun _ => ⟨h.1, ih.1 h.2⟩, fun _ => rfl⟩
    · exact ⟨fun h' => by simp at h', fun h' => absurd ⟨h'.1, ih.2 h'.2⟩ h⟩

/-- **An unopened position**: the opened word reads `[]` there, which a column equals only when it is `[]`, and which
a column's trim equals exactly when the column is zero. -/
theorem hole_agree {F : Type} [Zero F] [DecidableEq F] (n : ℕ) (i : Fin n) (u : List F) :
    (u = Claims.Flock.fill n ([] : List (ℕ × List F)) i ↔ u = []) ∧
      (FlockDeploy.trimCol u = Claims.Flock.fill n ([] : List (ℕ × List F)) i ↔ ∀ a ∈ u, a = 0) := by
  have h0 : Claims.Flock.fill n ([] : List (ℕ × List F)) i = [] := rfl
  rw [h0]
  exact ⟨Iff.rfl, trimCol_eq_nil_iff u⟩

/-! ## One premise, both records -/

/-- **KS at both records from the one bound at tables of `Ok` columns**: at `D.mapCols f` directly, `0` being `f` of an
`Ok` column on a nonempty word, and at `D` with `ExtendJ`, both at the same `κ` and `γ`. -/
theorem KS_both_of_tableBoundJR {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] (H : Claims.Msg → Claims.Msg) (D : Claims.Flock.Deploy Coin Col T)
    (N : Claims.Msg → ℕ → ℕ) (f : Col → Col) (hf : f 0 = 0) (Ok : Claims.Msg → ℕ → Col → Prop)
    (hN : ∀ x, ((∑ j : Fin (Claims.Flock.nTab D x), N x j : ℕ) : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x)
    (hVk : ∀ x, 0 < D.V.k x) (hK : ∀ x, 0 < D.cap x)
    (hk : ∀ x, (8 ≤ D.k x ∧ Claims.Flock.Splits D x) ∨ Claims.Flock.NoAcc H D.V x)
    (hp : ∀ x, (∃ s, D.parse x = some s) ∨ Claims.Flock.NoAcc H D.V x)
    (hz : ∀ x j, 0 < N x j → ∃ v, Ok x j v ∧ f v = 0)
    (hinj : ∀ x j u v, Ok x j u → Ok x j v → f u = f v → u = v)
    (hne : ∀ x j u, 0 < N x j → Ok x j u → u ≠ 0) (hok : Claims.Flock.ColsOkJ H D N Ok)
    (hconf : Claims.Flock.ConflictTiedJ H D N) (htab : Claims.Flock.TableBoundJR H D N Ok) :
    Claims.Flock.KS H (D.mapCols f) ∧ (Claims.Flock.ExtendJ D N Ok → Claims.Flock.KS H D) := by
  have hk1 : ∀ x, 1 ≤ D.k x ∨ Claims.Flock.NoAcc H D.V x :=
    fun x => (hk x).imp (fun h => le_trans (by norm_num) h.1) id
  have htrim : Claims.Flock.TableBoundJR H (D.mapCols f) N
      (Claims.Flock.orZeroJ (Claims.Flock.OkThroughJ f Ok)) := by
    have h := Claims.Flock.tableBoundJR_mapCols H D N f hf Ok hok hinj htab
    intro x Ω _ _ P
    obtain ⟨e, hfit, hb⟩ := h x Ω P
    refine ⟨e, hfit, fun ω j hj C hC hg => hb ω j hj C (fun p => ?_) hg⟩
    rcases hC p with h0 | hC'
    · rw [h0]
      exact hz x j (Nat.lt_of_le_of_lt (Nat.zero_le _) p.isLt)
    · exact hC'
  exact ⟨Claims.Flock.KS_of_oneWait H (D.mapCols f) hK hk hεt
      (Claims.Flock.oneWait_of_tableBoundJR H (D.mapCols f) N hN hεt hVk hk1 hp _ (fun _ _ => .inl rfl)
        (Claims.Flock.colsOkJ_orZero _ _ _ _ (Claims.Flock.colsOkJ_mapCols H D N f Ok hok))
        (Claims.Flock.conflictTiedJ_mapCols H D N f hconf) htrim),
    fun hext => Claims.Flock.KS_of_oneWait H D hK hk hεt
      (Claims.Flock.oneWait_of_tableBoundJR H D N hN hεt hVk hk1 hp _ (fun _ _ => .inl rfl)
        (Claims.Flock.colsOkJ_orZero _ _ _ _ hok) hconf
        (Claims.Flock.tableBoundJR_of_extend H D N Ok hne hok hext htab))⟩

end Claims.Flock.TrimBridge

namespace FlockDeploy.KS.TrimBridge

open Flock

/-- **KS at the untrimmed masked instance gives it at the masked record the claims read.** -/
theorem zk_KS_trim_of_KS (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (h : Claims.Flock.KS H (deployZk rel)) : Claims.Flock.KS H (deployZkTrim rel) :=
  Claims.Flock.TrimBridge.KS_mapCols_of_KS H (deployZk rel) trimCol rfl h

/-- **KS at the untrimmed plain instance gives it at the plain record the claims read.** -/
theorem plain_KS_trim_of_KS (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (h : Claims.Flock.KS H (deployPlain rel)) : Claims.Flock.KS H (deployPlainTrim rel) :=
  Claims.Flock.TrimBridge.KS_mapCols_of_KS H (deployPlain rel) trimCol rfl h

/-- **KS at both masked records from every table's bound counted per round over full-width tables**: at the trimmed
record directly, and at the untrimmed instance with `ExtendJ` at full-width columns, at the same `κ` and `γ`. -/
theorem zk_KS_both_of_tableBoundJR
    (htab : Claims.Flock.TableBoundJR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) ∧
      (Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true) →
        Claims.Flock.KS sha512 (deployZk Readable.relation)) :=
  ⟨zk_KS_trim_of_tableBoundJR htab, fun hext => zk_KS_of_tableBoundJR (zk_tableBoundJR_of_extend _ hext htab)⟩

/-- **KS at both masked records from `ZkXferWJ`** at the statements some run accepts, the untrimmed with `ExtendJ`. -/
theorem zk_KS_both_of_xferWJ (hx : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkXferWJ x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) ∧
      (Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true) →
        Claims.Flock.KS sha512 (deployZk Readable.relation)) :=
  ⟨zk_KS_trim_of_xferWJ hx, zk_KS_of_xferWJ hx⟩

/-- **KS at both masked records from the cap/`τ` and pads fork bounds and the view's pads clash**, the premises
`zk_KS_of_kinds` takes: the trimmed record's is `zk_KS_of_kinds`'s, and the untrimmed instance's needs `ExtendJ` at
full-width columns besides. -/
theorem zk_KS_both_of_kinds
    (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) ∧
      (Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true) →
        Claims.Flock.KS sha512 (deployZk Readable.relation)) :=
  zk_KS_both_of_xferWJ fun x hno => zkXferWJ_of_simWJ (zkSimWJ_of_parts (zkSimRunAt x) (zkSimAbortAt x)
    (zkSimForkLe_of_kinds (hct x hno) (hp x hno)) (zkViewClash_of_Q (zkViewClashQ₀ x) (zkViewClashQL x) (hpads x hno)))

/-- **KS at the untrimmed masked instance** at SHA-512 at the deployed relation, from `zk_KS_of_kinds`'s premises and
`ExtendJ` at full-width columns. -/
theorem zk_KS_untrim_of_kinds
    (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ)
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  (zk_KS_both_of_kinds hct hp hpads).2 hext

end FlockDeploy.KS.TrimBridge

namespace FlockDeploy.KS.Gen

open Flock

variable [S : ZkSetup]

/-- **KS at both masked records from `ZkXferWJ`** at the statements some run accepts, the untrimmed with `ExtendJ`. -/
theorem TrimBridge.zk_KS_both_of_xferWJ (hx : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkXferWJ x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) ∧
      (Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true) →
        Claims.Flock.KS sha512 (deployZk Readable.relation)) :=
  ⟨zk_KS_trim_of_xferWJ hx, zk_KS_of_xferWJ hx⟩

/-- **KS at both masked records from the cap/`τ` and pads fork bounds and the view's pads clash**, the premises
`zk_KS_of_kinds` takes: the trimmed record's is `zk_KS_of_kinds`'s, and the untrimmed instance's needs `ExtendJ` at
full-width columns besides. -/
theorem TrimBridge.zk_KS_both_of_kinds
    (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) ∧
      (Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true) →
        Claims.Flock.KS sha512 (deployZk Readable.relation)) :=
  zk_KS_both_of_xferWJ fun x hno => zkXferWJ_of_simWJ (zkSimWJ_of_parts (zkSimRunAt x) (zkSimAbortAt x)
    (zkSimForkLe_of_kinds (hct x hno) (hp x hno)) (zkViewClash_of_Q (zkViewClashQ₀ x) (zkViewClashQL x) (hpads x hno)))

/-- **KS at the untrimmed masked instance** at SHA-512 at the deployed relation, from `zk_KS_of_kinds`'s premises and
`ExtendJ` at full-width columns. -/
theorem TrimBridge.zk_KS_untrim_of_kinds
    (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ)
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  (zk_KS_both_of_kinds hct hp hpads).2 hext

end FlockDeploy.KS.Gen
