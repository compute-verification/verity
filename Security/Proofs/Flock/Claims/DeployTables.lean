import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Soundness.Discharge.Exec.Setup

/-!
# The deployed records' tables are the parsed statement's

Where a statement sets up, the claims' parser (`FlockDeploy.parse`) reads as many tables (`Claims.Flock.Stmt.tables`:
the table count, at least one) as the setup makes (`FlockDeploy.tables`), the tables whose columns the records read
(`FlockDeploy.runOf` at table `j`, `j` below the count). `Stmt.setupTables` makes one table at a count of at most one
and `count` tables above it, with or without hidden outputs and at any public input words (`setupTables_size`).
-/

namespace FlockDeploy

open Flock
open FlockSoundness.Refine FlockSoundness.Discharge.Exec

/-- **`setupTables` makes `max 1 count` tables.** -/
theorem setupTables_size {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray}
    {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ}
    {closure : Option (Std.HashMap ℕ (Array ℕ))} {publicInputs : Option ByteArray} {spec : SessionSpec}
    {sts : Array Stmt}
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure publicInputs =
      .ok (spec, sts)) :
    sts.size = max 1 count := by
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
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  all_goals
    first
    | (simp only [List.size_toArray, List.length_singleton]; omega)
    | (obtain ⟨hsz, -⟩ := range_mapM_ok (by assumption); omega)

/-- **A setup makes as many tables as the parsed statement counts.** -/
theorem setupOf_size_eq_tables {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {i : Ins} {spec : SessionSpec} {sts : Array Stmt} {s : Claims.Flock.Stmt}
    (h : setupOf zk x = some (i, spec, sts)) (hp : parse zk rel x = some s) : sts.size = s.tables := by
  obtain ⟨hi, hs⟩ := setupOf_some h
  have hr := Ins.setup_read hs
  have hform : s.form.tables = i.read.I.count := by
    unfold parse Readable.parseOf Readable.parseBytes readOf at hp
    rw [hi] at hp
    simp only [Option.map_some, Option.bind_some] at hp
    obtain ⟨p, hpp, rfl⟩ := Option.map_eq_some_iff.1 hp
    unfold Readable.Read.parse at hpp
    obtain ⟨sts', -, rfl⟩ := Option.map_eq_some_iff.1 hpp
    rfl
  have hsize : sts.size = max 1 i.read.I.count := setupTables_size hr
  rw [hsize, Claims.Flock.Stmt.tables, hform]

/-- **The records' tables are the parsed statement's**: where the statement sets up, the session's tables number
`Claims.Flock.Stmt.tables` of the parsed statement. -/
theorem tables_length {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {s : Claims.Flock.Stmt} (hp : parse zk rel x = some s) (hx : (setupOf zk x).isSome) :
    (tables zk x).length = s.tables := by
  obtain ⟨⟨i, spec, sts⟩, h⟩ := Option.isSome_iff_exists.1 hx
  unfold tables
  rw [h, List.length_map, Array.length_toList]
  exact setupOf_size_eq_tables h hp

end FlockDeploy
