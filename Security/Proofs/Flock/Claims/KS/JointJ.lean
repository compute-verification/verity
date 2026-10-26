import Proofs.Flock.Claims.KS.Joint

/-!
# One wait's bound from every table's bound

`Joint` at a statement of several tables. The positions are every table's word's (`PosJ`: table `j`'s `N x j`
positions, for each of the statement's `nTab` tables), and an accepted run's opening at table `j`'s position `p` is
its first opened column of table `j` there (`opJ`, `firstTable_opJ`). The obligations are `Joint`'s at each table:

* `TableBoundJ`: at every statement and prover, escapes that fit the weight with one more fork event, such that at
  every tape, every table `j` of the statement and every table at its word's positions whose columns `Ok` admits and
  that does not extract table `j` (`GoodTableAt`), a run accepts with probability at most `ε_table + escapes(ω)`
  beyond that of accepting and opening a position of table `j`'s word off the table. The escapes are one family for
  all tables, and `ε_table` the tables' errors summed.
* `ColsOkJ`, `ConflictTiedJ`: `ColsOk` and `ConflictTied` at each table.

**One wait** (`oneWait_of_tableBoundJ`): `joint_acc_le` over `PosJ`, at a run list that does not extract, at the
first table that does not: its table is not `GoodTableAt` there. The rate is the words' lengths summed, at most `N₀`,
and the statement parses wherever some run accepts. At tables with unopened positions (`orZeroJ`), the bound follows
from the bound at tables of `Ok` columns and `ExtendJ` (`tableBoundJ_of_extend`), as in `Joint`.
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

section Tables

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The statement's tables**: `Stmt.tables` where it parses, none where it does not. -/
def nTab (D : Deploy Coin Col T) (x : Msg) : ℕ := ((D.parse x).map Stmt.tables).getD 0

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem nTab_of_parse {D : Deploy Coin Col T} {x : Msg} {s : Stmt} (hs : D.parse x = some s) :
    nTab D x = s.tables := by
  unfold nTab
  rw [hs]
  rfl

/-- **The words' positions**: table `j`'s word's `N x j`, for each table `j` of the statement. -/
abbrev PosJ (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (x : Msg) : Type := Σ j : Fin (nTab D x), Fin (N x j)

open Classical in
/-- **An accepted run's openings** at the words' positions, its first column of table `j` at table `j`'s; none on a
rejected run. -/
noncomputable def opJ (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (r : Fin (D.V.k x) → Coin) (p : PosJ D N x) : Option Col :=
  if (run H P D.V x ω r).out = .acc then lookCol (D.cols (run H P D.V x ω r).seen p.1) p.2.val else none

omit [DecidableEq Col] [Fintype Coin] [Nonempty Coin] in
/-- **On accepted reruns the extractor's table is the zero-filled words.** -/
theorem firstTable_opJ (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (bs : List (Fin (D.V.k x) → Coin)) (hacc : ∀ b ∈ bs, Acc H D.V P x ω b) (p : PosJ D N x) :
    firstTable (opJ H D N P x ω) (fun _ => 0) bs p =
      (lookCol ((seenOf H D.V P x ω bs).flatMap fun v => D.cols v p.1) p.2.val).getD 0 := by
  unfold firstTable
  rw [seenOf, List.flatMap_map,
    findSome?_lookCol _ (fun r => D.cols (run H P D.V x ω r).seen p.1) p.2.val bs fun b hb => by
      unfold opJ
      simp only [show (run H P D.V x ω b).out = .acc from hacc b hb, ↓reduceIte]]

/-- **A table at table `j`'s word's positions extracts**: a trace no longer than the word, that read alone is a
witness of table `j`'s own statement, agrees with it on at least `(1 − θ)·len` positions. -/
def GoodTableAt (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (x : Msg) (j : ℕ) (C : Fin (N x j) → Col) : Prop :=
  ∃ (M : T) (t : Stmt), D.table x j = some t ∧ RelS t (D.read [M]) ∧ D.len M ≤ N x j ∧
    (1 - D.θ M) * D.len M ≤ ((Finset.univ : Finset (Fin (D.len M))).filter fun i => D.enc M i.val = extTab C i.val).card

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem extractsAt_of_goodAt (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (x : Msg) (j : ℕ) (vs : List Seen)
    (h : GoodTableAt D N x j fun p => (lookCol (vs.flatMap fun v => D.cols v j) p.val).getD 0) :
    ExtractsAt D x vs j := by
  obtain ⟨M, t, ht, hrel, hle, hag⟩ := h
  have e : ((Finset.univ : Finset (Fin (D.len M))).filter fun i => D.enc M i.val =
        extTab (fun p : Fin (N x j) => (lookCol (vs.flatMap fun v => D.cols v j) p.val).getD 0) i.val) =
      (Finset.univ.filter fun i => D.enc M i.val = fill (D.len M) (vs.flatMap fun v => D.cols v j) i) :=
    Finset.filter_congr fun i _ => by rw [fill_eq_lookCol, extTab, dite_eq_left (lt_of_lt_of_le i.isLt hle)]
  exact ⟨M, t, ht, hrel, hag.trans_eq (by rw [e])⟩

/-- **Every table's fixed-table bound**, with one family of escapes: at every statement and prover, escapes that
fit the weight with one more fork event of weight `k`, such that at every tape, every table `j` of the statement and
every table at its word's positions whose columns `Ok` admits and which does not extract, a run accepts with
probability at most `ε_table + escapes(ω)` beyond that of accepting and opening a position of table `j`'s word off the
table. -/
def TableBoundJ (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x,
    e.FitsFork (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (j : ℕ), j < nTab D x → ∀ C : Fin (N x j) → Col,
      (∀ p, Ok x j (C p)) → ¬ GoodTableAt D N x j C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x j)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen j) p.val = some u ∧ u ≠ C p

/-- **An accepted run opens only columns `Ok` admits** at each table's word's positions. -/
def ColsOkJ (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (j q : ℕ) (u : Col),
    (run H P D.V x ω r).out = .acc → j < nTab D x → q < N x j →
      lookCol (D.cols (run H P D.V x ω r).seen j) q = some u → Ok x j u

/-- **Two accepted runs at one tape open each position of each table's word to one column**, or the verifier's calls
in the two runs collide. -/
def ConflictTiedJ (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r r' : Fin (D.V.k x) → Coin) (j q : ℕ) (u v : Col),
    (run H P D.V x ω r).out = .acc → (run H P D.V x ω r').out = .acc → j < nTab D x → q < N x j →
    lookCol (D.cols (run H P D.V x ω r).seen j) q = some u →
    lookCol (D.cols (run H P D.V x ω r').seen j) q = some v →
    u ≠ v → TiedColl H ((run H P D.V x ω r).vLog ++ (run H P D.V x ω r').vLog)

/-- **Two runs at one tape that open a position of some table's word to two columns collide**, with the second run
forked at round `0`: `ConflictTiedJ` at the runs' openings (`opJ`). -/
theorem opJ_tied (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (hconf : ConflictTiedJ H D N) {Ω : Type}
    (P : Prover Ω) (x : Msg) (j₀ : Fin (D.V.k x)) (hj₀ : j₀.val = 0) (ω : Ω) (r r' : Fin (D.V.k x) → Coin)
    (h : ∃ p, ConflictAt (opJ H D N P x ω) (r, r') p) :
    TiedColl H ((run H P D.V x ω r).vLog ++ (run H P D.V x ω (forkCoins j₀ r r')).vLog) := by
  classical
  obtain ⟨p, h1, h2, h3⟩ := h
  rw [Claims.Bridge.forkCoins_zero j₀ hj₀]
  simp only [opJ] at h1 h2 h3
  split_ifs at h1 h2 h3 with ha ha'
  · obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 h2
    exact hconf x Ω P ω r r' p.1.val p.2.val u v ha ha' p.1.isLt p.2.isLt hu hv
      fun huv => h3 (by rw [hu, hv, huv])
  all_goals simp at h1 h2

/-- **One wait's bound at a tape, from every table's bound there**: `oneWait_le_of_joint` over the words' positions
(`PosJ`), at the first table a list of accepted reruns does not extract; the escapes are `e₀` with the conflict at
round `0` (`opJ_tied`). -/
theorem oneWait_le_of_tableBoundJ (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (hN : ∀ x, ((∑ j : Fin (nTab D x), N x j : ℕ) : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x)
    (hVk : ∀ x, 0 < D.V.k x) (hk : ∀ x, 1 ≤ D.k x ∨ NoAcc H D.V x)
    (hp : ∀ x, (∃ s, D.parse x = some s) ∨ NoAcc H D.V x) (Ok : Msg → ℕ → Col → Prop) (hz : ∀ x j, Ok x j 0)
    (hok : ColsOkJ H D N Ok) (hconf : ConflictTiedJ H D N) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (e₀ : Escapes H D.V P x)
    (hb : ∀ (ω : Ω) (j : ℕ), j < nTab D x → ∀ C : Fin (N x j) → Col, (∀ p, Ok x j (C p)) → ¬ GoodTableAt D N x j C →
      accPr H D.V P x ω ≤ D.εtable x + e₀.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x j)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen j) p.val = some u ∧ u ≠ C p)
    (ω : Ω) (hε : 0 < accPr H D.V P x ω) :
    accPr H D.V P x ω * (accFail H D.V P x ω (D.k x) (Extracts D x) / accPr H D.V P x ω ^ D.k x) ≤
      D.εtable x + (e₀.addFork (D.k x) (Nat.cast_nonneg _) ⟨0, hVk x⟩
        (fun ω r r' => ∃ p, ConflictAt (opJ H D N P x ω) (r, r') p)
        (opJ_tied H D N hconf P x ⟨0, hVk x⟩ rfl)).term ω + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω := by
  classical
  have hacc : ¬ NoAcc H D.V x := fun hno => by
    have : accPr H D.V P x ω = 0 := by
      unfold accPr
      rw [Pr_congr (q := fun _ => False) fun r => ⟨fun h => hno Ω P ω r h, False.elim⟩, Pr_false]
    linarith
  obtain ⟨s, hs⟩ := (hp x).resolve_right hacc
  refine oneWait_le_of_joint H D P x (opJ H D N P x) ?_ (hεt x) e₀ ⟨0, hVk x⟩
    (opJ_tied H D N hconf P x ⟨0, hVk x⟩ rfl) ω ((hk x).resolve_right hacc) hε fun bs hall hnot => ?_
  · rw [Fintype.card_sigma]
    simp only [Fintype.card_fin]
    exact hN x
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
  refine (hb ω j hjn _ hOk hgood).trans (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne⟩ => ?_))
  have e : opJ H D N P x ω r ⟨⟨j, hjn⟩, p⟩ = some u := by
    unfold opJ
    rw [ite_eq_left ha]
    exact hu
  refine ⟨ha, ⟨⟨j, hjn⟩, p⟩, ?_⟩
  show (opJ H D N P x ω r ⟨⟨j, hjn⟩, p⟩).isSome ∧ opJ H D N P x ω r ⟨⟨j, hjn⟩, p⟩ ≠ some _
  rw [e]
  exact ⟨rfl, fun h => hne (Option.some_injective _ h)⟩

/-- **Flock's one-wait bound, from every table's bound**: when the words together are no longer than `N₀`,
`ε_table ≥ 0`, the verifier takes a round, `k ≥ 1` and the statement parses wherever it accepts some run, and `Ok`
admits `0` and every column an accepted run opens. -/
theorem oneWait_of_tableBoundJ (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (hN : ∀ x, ((∑ j : Fin (nTab D x), N x j : ℕ) : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x)
    (hVk : ∀ x, 0 < D.V.k x) (hk : ∀ x, 1 ≤ D.k x ∨ NoAcc H D.V x)
    (hp : ∀ x, (∃ s, D.parse x = some s) ∨ NoAcc H D.V x) (Ok : Msg → ℕ → Col → Prop) (hz : ∀ x j, Ok x j 0)
    (hok : ColsOkJ H D N Ok) (hconf : ConflictTiedJ H D N) (htab : TableBoundJ H D N Ok) :
    OneWait H D := by
  intro x Ω _ _ P
  obtain ⟨e₀, hfit, hb⟩ := htab x Ω P
  exact ⟨_, e₀.fits_addFork (D.weight x) (D.k x) (Nat.cast_nonneg _) ⟨0, hVk x⟩ rfl
    (fun ω r r' => ∃ p, ConflictAt (opJ H D N P x ω) (r, r') p) (opJ_tied H D N hconf P x ⟨0, hVk x⟩ rfl) hfit,
    oneWait_le_of_tableBoundJ H D N hN hεt hVk hk hp Ok hz hok hconf x P e₀ hb⟩

/-- **`Ok`, or a position no rerun opened** (`0`), at each table. -/
def orZeroJ (Ok : Msg → ℕ → Col → Prop) : Msg → ℕ → Col → Prop := fun x j u => u = 0 ∨ Ok x j u

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem colsOkJ_orZero (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop)
    (h : ColsOkJ H D N Ok) : ColsOkJ H D N (orZeroJ Ok) :=
  fun x Ω P ω r j q u ha hj hq hu => .inr (h x Ω P ω r j q u ha hj hq hu)

/-- **A table with unopened positions extends**, at each table of the statement: when it does not extract, some
table of `Ok` columns that keeps its opened ones does not extract either. -/
def ExtendJ (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop) : Prop :=
  ∀ (x : Msg) (j : ℕ), j < nTab D x → ∀ C : Fin (N x j) → Col, (∀ p, orZeroJ Ok x j (C p)) →
    ¬ GoodTableAt D N x j C →
    ∃ C' : Fin (N x j) → Col, (∀ p, Ok x j (C' p)) ∧ (∀ p, C p ≠ 0 → C' p = C p) ∧ ¬ GoodTableAt D N x j C'

omit [Nonempty Coin] in
/-- **Every table's bound at tables with unopened positions**, from the bound at tables of `Ok` columns and
`ExtendJ`, when accepted runs open only `Ok` columns and `Ok` admits no `0` on a nonempty word. -/
theorem tableBoundJ_of_extend (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (Ok : Msg → ℕ → Col → Prop) (hne : ∀ x j u, 0 < N x j → Ok x j u → u ≠ 0) (hok : ColsOkJ H D N Ok)
    (hext : ExtendJ D N Ok) (htab : TableBoundJ H D N Ok) : TableBoundJ H D N (orZeroJ Ok) := by
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
