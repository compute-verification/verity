import Proofs.Flock.Claims.Complete.ZkTables.Skeleton
import Proofs.Flock.Claims.Complete.Zk.Discharge
import Proofs.Flock.Claims.Scope.Fit

/-!
# The masked scope at several tables

* `setupTables_names`: a session of several tables names them `Tags.tableNames` and has at most `Tags.MAX_TABLES`.
* `Case.tables`: at a good case the session has `1 ≤ |tables| ≤ 64`, the spec names one table per statement, and
  every table is of `25 ≤ m ≤ 35` and commits with `hm96-sha512/v1`.
-/

open Flock FlockDeploy

namespace FlockComplete.ZkTables

open FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec

section Walk

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Stmt}

set_option maxHeartbeats 4000000 in
/-- **A session of several tables** names them `Tags.tableNames` and has at most `Tags.MAX_TABLES`. -/
theorem setupTables_names (hc : 1 < count)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    spec.tables = tags.tableNames count ∧ count ≤ Tags.MAX_TABLES := by
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
  have hn : ¬ count ≤ 1 := by omega
  simp only [throw_bind', pure_bind, ite_eq_right hn] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  all_goals exact ⟨rfl, by omega⟩

theorem tableNames_size (t : Tags) {count : ℕ} (h : 1 < count) : (t.tableNames count).size = count := by
  simp [Tags.tableNames, show ¬ count ≤ 1 by omega]

/-- **A session's tables**: at least one, at most `Tags.MAX_TABLES`, and the spec names one per statement. -/
theorem setupTables_size
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    0 < sts.size ∧ sts.size ≤ Tags.MAX_TABLES ∧ spec.tables.size = sts.size := by
  rcases HiddenExec.setupTables_okAny h with ⟨-, st, -, rfl, rfl⟩ | ⟨hc, hsz, -, -, -⟩
  · simp [Stmt.spec, Tags.MAX_TABLES]
  · obtain ⟨hn, hmax⟩ := setupTables_names hc h
    rw [hn, tableNames_size _ hc, hsz]
    exact ⟨by omega, hmax, rfl⟩

end Walk

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The masked scope at a good case**: one to `Tags.MAX_TABLES` tables, the spec names one per statement, and every
table is of `25 ≤ m ≤ 35` (the `--zk` scope's `mMin`, table 0's bound, and every table at table 0's `m`) and commits
with `hm96-sha512/v1` (the scope check's leaf scheme). -/
theorem Case.tables (c : Case H rel) :
    0 < c.sts.size ∧ c.sts.size ≤ Tags.MAX_TABLES ∧ c.spec.tables.size = c.sts.size ∧
      ∀ j (hj : j < c.sts.size), 25 ≤ c.sts[j].m ∧ c.sts[j].m ≤ 35 ∧ c.sts[j].merkle = MerkleScheme.hm96Sha512 := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  have hzk : p.zk = true := readOf_zk (x := c.x) (by
    unfold Readable.parseBytes at hp
    obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
    rw [hr, ← Readable.toRead_of_parse hrp])
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  have htab := Readable.tablesOf_of_parses (Readable.parses_of_parse hp)
  obtain ⟨hc, hsz⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hc' : ProvedScope.check p.tags p.zk p.I.circuitFile p.I.publicFile p.I.tables p.I.count false p.registered =
      .ok () := by
    have := hc
    unfold Readable.Read.check at this
    rw [hpi] at this
    exact this
  have hl := (FailClosed.check_ok hc').1.2.2.2.1
  obtain ⟨spec', hset'⟩ := Readable.setup_of_tablesOf htab
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [ht] at hset'
  have hpe : p.sts = c.sts := by
    have := hset'.symm.trans hset
    simp only [Except.ok.injEq, Prod.mk.injEq] at this
    exact this.2
  rw [hpe] at hsz htab
  have hset2 := hset
  rw [← ht] at hset2
  unfold Readable.Read.setup at hset2
  rw [hpi] at hset2
  obtain ⟨hpos, hmax, hnames⟩ := setupTables_size hset2
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset2
  obtain ⟨st0, h0, -, -, -, -, -, hm35, -⟩ := Readable.table0 htab hR.2.1 hpi
  have h0' : c.sts[0] = st0 := by
    rw [Array.getElem?_eq_getElem hpos] at h0
    exact Option.some.inj h0
  refine ⟨hpos, hmax, hnames, fun j hj => ⟨?_, ?_, ?_⟩⟩
  · have := (Readable.sizesOk_iff _ _).1 hsz c.sts[j].m (by
      simp only [Array.map_map, Array.mem_map]
      exact ⟨c.sts[j], Array.getElem_mem hj, rfl⟩)
    rw [hzk] at this
    simpa [ProvedScope.mMin] using this
  · have hf := (Readable.tables_fit htab hR.2.1 hpi hj).1
    have : c.sts[j].m = c.sts[0].m := hf
    rw [this, h0']
    exact hm35
  · obtain ⟨d, hd, -⟩ := hall j hj
    exact HiddenExec.setupOne_tagsAt_hm96 hl hd

end FlockComplete.ZkTables
