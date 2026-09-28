import Proofs.Flock.Soundness.Discharge.Exec.Event
import Proofs.Flock.Soundness.Refine.Setup

/-!
# flock-verify's acceptance sets up each table with `Stmt.setupH`

`Accepts I t` sets up the session with `Stmt.setupTables`. Under a tag without hidden outputs
(`tags.hiddenOutputs = false`; with them it calls `Stmt.setupHidden`), each of its tables is `Stmt.setupH`'s statement
(`setupTables_ok`): for one table, at the inputs and the record's draw; for `count > 1`, under
`Tags.tableIdentity` and at a subset draw of the table's part. So every fact proved of `setupH`'s statements
(`setupH_wf`, `setupH_layout`, `placement_of_setupH`) holds of each table flock-verify accepts with
(`accepts_setupH`).

`setupH_public` records the two checks `setupH` makes on the public file that the layout's registered values read:
`HmRow.checkPublic st.c st.pub` passed, and `st.regions` is `HmRow.regions st.c st.pub`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine
open Lean (Json)

theorem list_mapM_ok_getElem {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' →
      l'.length = l.length ∧ ∀ (j : ℕ) (hj : j < l.length) (hj' : j < l'.length), f l[j] = .ok l'[j]
  | [], l', h => by
      simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
  | a :: l, l', h => by
      rw [List.mapM_cons] at h
      obtain ⟨b, hb, h⟩ := ebind_ok h
      obtain ⟨bs, hbs, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨hlen, hall⟩ := list_mapM_ok_getElem hbs
      refine ⟨by simp [hlen], fun j hj hj' => ?_⟩
      rcases j with _ | j
      · exact hb
      · exact hall j (by simpa using hj) (by simpa using hj')

theorem array_mapM_ok_getElem {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : ys.size = xs.size ∧ ∀ (j : ℕ) (hj : j < xs.size) (hj' : j < ys.size),
      f xs[j] = .ok ys[j] := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
      rw [hl] at h
      simp only [Functor.map, Except.map, Except.ok.injEq] at h
      subst h
      obtain ⟨hlen, hall⟩ := list_mapM_ok_getElem hl
      refine ⟨by simpa using hlen, fun j hj hj' => ?_⟩
      have := hall j (by simpa using hj) (by simpa using hj')
      simpa using this

theorem range_mapM_ok {β ε : Type} {f : ℕ → Except ε β} {k : ℕ} {ys : Array β} (h : (Array.range k).mapM f = .ok ys) :
    ys.size = k ∧ ∀ (j : ℕ) (hj : j < ys.size), f j = .ok ys[j] := by
  obtain ⟨hsz, hall⟩ := array_mapM_ok_getElem h
  rw [Array.size_range] at hsz
  refine ⟨hsz, fun j hj => ?_⟩
  have := hall j (by rw [Array.size_range]; omega) hj
  rwa [Array.getElem_range] at this

theorem mPts_of_any {sts : Array Stmt} {m : ℕ} (h : ¬ (sts.any fun x => x.mPts != m) = true) :
    ∀ (j : ℕ) (hj : j < sts.size), sts[j].mPts = m := by
  intro j hj
  by_contra hne
  exact h (Array.any_eq_true.2 ⟨j, hj, by simpa using hne⟩)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Stmt}

set_option maxHeartbeats 4000000 in
/-- A plain table's setup, `setupH` followed by the refusal of a `public_inputs` meta key, returns `setupH`'s statement. -/
theorem ok_of_refuse {x : V Stmt} {p : Stmt → Bool} {e : String} {s : Stmt}
    (h : (do let st ← x; if p st = true then throw e else pure st) = .ok s) : x = .ok s := by
  cases x with
  | error => cases h
  | ok a =>
    by_cases hp : p a = true
    · simp [bind, Except.bind, hp, throw, throwThe, MonadExceptOf.throw, Except.bind] at h
    · simpa [bind, Except.bind, hp, pure, Except.pure] using h

/-- **`setupTables` sets up each table with `setupH`** (§16.12) under a tag without hidden outputs (`hh`; with them
it calls `setupHidden`). For one table, its statement is `setupH`'s at the session's draw, and the session's spec is
the table's. For `count > 1`, there are `count` tables, and table `j` is `setupH`'s statement under
`Tags.tableIdentity count j` at a subset draw of its part. -/
theorem setupTables_ok (hh : tags.hiddenOutputs = false)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    (count ≤ 1 ∧ ∃ st, Stmt.setupH tags cf pf coins tables draw partition program = .ok st ∧ spec = st.spec ∧
        sts = #[st]) ∨
      (1 < count ∧ sts.size = count ∧ (∀ (j : ℕ) (hj : j < sts.size), ∃ d,
        Stmt.setupH { tags with identity := tags.tableIdentity count j } cf pf coins tables (some d) partition
          program = .ok sts[j]) ∧ (∀ (j : ℕ) (hj : j < sts.size), sts[j].mPts = spec.mPts) ∧ spec.points = 2) := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  have hn : ¬ tags.hiddenOutputs = true := by simp [hh]
  simp only [throw_bind', pure_bind, ite_eq_right hn] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  · exact Or.inl ⟨by assumption, _, ok_of_refuse (by assumption), rfl, rfl⟩
  all_goals
    obtain ⟨hsz, hall⟩ := range_mapM_ok (by assumption)
    exact Or.inr ⟨by omega, hsz, fun j hj => ⟨_, ok_of_refuse (hall j hj)⟩, mPts_of_any (by assumption), rfl⟩

/-- A table's tags in a session of `count` tables: the statement's for one, under `Tags.tableIdentity` for several. -/
def tagsAt (tags : Tags) (count j : ℕ) : Tags :=
  if count ≤ 1 then tags else { tags with identity := tags.tableIdentity count j }

theorem tagsAt_typed (tags : Tags) (count j : ℕ) : (tagsAt tags count j).typed = tags.typed := by
  unfold tagsAt; split_ifs <;> rfl

theorem tagsAt_hm96Rows (tags : Tags) (count j : ℕ) : (tagsAt tags count j).hm96Rows = tags.hm96Rows := by
  unfold tagsAt; split_ifs <;> rfl

/-- Every table `setupTables` returns is `setupH`'s statement, under the table's tags (`tagsAt`) and at some draw
(the session's, for one table). -/
theorem setupTables_setupH (hh : tags.hiddenOutputs = false)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    0 < sts.size ∧ ∀ (j : ℕ) (hj : j < sts.size), ∃ d,
      Stmt.setupH (tagsAt tags count j) cf pf coins tables d partition program = .ok sts[j] ∧
        (count ≤ 1 → d = draw) := by
  rcases setupTables_ok hh h with ⟨hc, st, hst, -, rfl⟩ | ⟨hc, hsz, hall, -⟩
  · refine ⟨by simp, fun j hj => ⟨draw, ?_, fun _ => rfl⟩⟩
    have hj0 : j = 0 := by simpa using hj
    subst hj0
    simpa [tagsAt, hc] using hst
  · refine ⟨by omega, fun j hj => ?_⟩
    obtain ⟨d, hd⟩ := hall j hj
    refine ⟨some d, ?_, fun h1 => absurd h1 (by omega)⟩
    simpa [tagsAt, show ¬ count ≤ 1 by omega] using hd

/-- The session's two link points have every table's length. -/
theorem setupTables_points (hh : tags.hiddenOutputs = false)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    spec.points = 2 ∧ ∀ (j : ℕ) (hj : j < sts.size), sts[j].mPts = spec.mPts := by
  rcases setupTables_ok hh h with ⟨-, st, -, rfl, rfl⟩ | ⟨-, -, -, hm, hp⟩
  · exact ⟨rfl, fun j hj => by
      have : j = 0 := by simpa using hj
      subst this
      rfl⟩
  · exact ⟨hp, hm⟩

/-- **flock-verify's acceptance sets up each table with `setupH`**: when `Accepts I t`, every table's statement is
`setupH`'s, under the table's tags and at some draw, the record's for a one-table session. -/
theorem accepts_setupH {I : Inputs} {t : Transcript} (hh : I.tags.hiddenOutputs = false) (h : Accepts I t) :
    ∃ spec sts, setupOf I t.record = .ok (spec, sts) ∧
      Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ d,
        Stmt.setupH (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables d I.partition I.program =
          .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record) := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  exact ⟨spec, sts, hs, hv, setupTables_setupH hh hs⟩

/-- **A one-table acceptance is one `setupH` statement's**: the session's spec is the table's, and `Flock.verify` ran on
`#[Setup.ofCircuit st]`, the form `verify_refines_ofCircuit_hm96` reads. -/
theorem accepts_one {I : Inputs} {t : Transcript} (hh : I.tags.hiddenOutputs = false) (hc : I.count ≤ 1)
    (h : Accepts I t) :
    ∃ st, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (recordDraw t.record) I.partition I.program =
        .ok st ∧ Flock.verify st.spec #[Setup.ofCircuit st] t.record t.proofs = .ok () := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  rcases setupTables_ok hh hs with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  · exact ⟨st, hst, by simpa using hv⟩
  · omega

set_option maxHeartbeats 4000000 in
/-- **`setupH`'s checks on the public file** (§16.2, §16.4): the population's (or the draw's) public file passed
`HmRow.checkPublic`, and the statement's regions are `HmRow.regions` of it. -/
theorem setupH_public {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    HmRow.checkPublic st.c st.pub = .ok () ∧ HmRow.regions st.c st.pub = .ok st.regions := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨by assumption, by assumption⟩

end FlockSoundness.Discharge.Exec
