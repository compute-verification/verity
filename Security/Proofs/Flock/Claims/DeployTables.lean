import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.Scope.Split
import Proofs.Flock.Soundness.Discharge.Exec.Setup

/-!
# The deployed records' tables are the parsed statement's

Where a statement sets up, the claims' parser (`FlockDeploy.parse`) reads as many tables (`Claims.Flock.Stmt.tables`:
the table count, at least one) as the setup makes (`FlockDeploy.tables`), the tables whose columns the records read
(`FlockDeploy.runOf` at table `j`, `j` below the count). `Stmt.setupTables` makes one table at a count of at most one
and `count` tables above it, with or without hidden outputs and at any public input words (`setupTables_size`).

Each table is its own statement (`FlockDeploy.table`: table `j` of the parsed statement alone), and the deployed
relation splits by table: at an in-scope statement a witness is in the relation exactly when each table's share is in
its own statement's (`relS_split`). So `assemble`, which takes each table's first candidate in that table's own
statement, returns only witnesses in the relation (`relS_of_assemble`), and at one table, whose own statement is the
statement, both records' `assemble` reads as before (`assemble_deployZk_one`, `assemble_deployPlain_one`).
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

namespace KS.Gen

variable [S : ZkSetup]

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

end KS.Gen

/-- **A setup makes as many tables as the parsed statement counts.** -/
theorem setupOf_size_eq_tables {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {i : Ins} {spec : SessionSpec} {sts : Array Stmt} {s : Claims.Flock.Stmt}
    (h : setupOf zk x = some (i, spec, sts)) (hp : parse zk rel x = some s) : sts.size = s.tables :=
  KS.Gen.setupOf_size_eq_tables (S := KS.S₀) (zk := zk) (rel := rel) (x := x) (i := i) (spec := spec) (sts := sts)
    (s := s) (h := h) (hp := hp)

namespace KS.Gen

variable [S : ZkSetup]

/-- **The records' tables are the parsed statement's**: where the statement sets up, the session's tables number
`Claims.Flock.Stmt.tables` of the parsed statement. -/
theorem tables_length {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {s : Claims.Flock.Stmt} (hp : parse zk rel x = some s) (hx : (setupOf zk x).isSome) :
    (tables zk x).length = s.tables := by
  obtain ⟨⟨i, spec, sts⟩, h⟩ := Option.isSome_iff_exists.1 hx
  unfold tables
  rw [h, List.length_map, Array.length_toList]
  exact setupOf_size_eq_tables h hp

end KS.Gen

/-- **The records' tables are the parsed statement's**: where the statement sets up, the session's tables number
`Claims.Flock.Stmt.tables` of the parsed statement. -/
theorem tables_length {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {s : Claims.Flock.Stmt} (hp : parse zk rel x = some s) (hx : (setupOf zk x).isSome) :
    (tables zk x).length = s.tables :=
  KS.Gen.tables_length (S := KS.S₀) (zk := zk) (rel := rel) (x := x) (s := s) (hp := hp) (hx := hx)

/-- **A statement the verifier parses has `max 1 count` tables.** -/
theorem size_of_parses {p : Readable.Parsed} (hp : Readable.Parses p) : p.sts.size = max 1 p.I.count := by
  obtain ⟨spec, hset⟩ := Readable.setup_of_tablesOf (Readable.tablesOf_of_parses hp)
  exact setupTables_size hset

namespace KS.Gen

variable [S : ZkSetup]

/-- **What the claims' parser accepts**: the readable form of a statement the verifier parses, whose tables number
`Claims.Flock.Stmt.tables`, and each table's own statement (`table`) is its table alone. -/
theorem parsed_of_parse {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {s : Claims.Flock.Stmt} (hs : parse zk rel x = some s) :
    ∃ p : Readable.Parsed, Readable.parseBytes (readOf zk) x = some p ∧ s = Readable.toReadable rel p ∧
      Readable.Parses p ∧ p.sts.size = s.tables ∧ ∀ j, table zk rel x j = some (Readable.toReadable rel (p.only j)) := by
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨r, -, hrp⟩ := Option.bind_eq_some_iff.1 hp
  have hP := Readable.parses_of_parse hrp
  refine ⟨p, hp, rfl, hP, (size_of_parses hP).trans rfl, fun j => ?_⟩
  simp only [table, Readable.tableOf, hp, Option.map_some]

end KS.Gen

/-- **What the claims' parser accepts**: the readable form of a statement the verifier parses, whose tables number
`Claims.Flock.Stmt.tables`, and each table's own statement (`table`) is its table alone. -/
theorem parsed_of_parse {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {s : Claims.Flock.Stmt} (hs : parse zk rel x = some s) :
    ∃ p : Readable.Parsed, Readable.parseBytes (readOf zk) x = some p ∧ s = Readable.toReadable rel p ∧
      Readable.Parses p ∧ p.sts.size = s.tables ∧ ∀ j, table zk rel x j = some (Readable.toReadable rel (p.only j)) :=
  KS.Gen.parsed_of_parse (S := KS.S₀) (zk := zk) (rel := rel) (x := x) (s := s) (hs := hs)

namespace KS.Gen

variable [S : ZkSetup]

/-- **The deployed relation splits by table**: at an in-scope statement the claims' parser accepts, a witness is in the
relation exactly when, for every table `j`, its share (`Readable.tableWit`: hidden values `a·J + j`) is in the relation
of table `j`'s own statement (`table`). -/
theorem relS_split {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse zk Readable.relation x = some s) (hin : Claims.Flock.InScope zk s) (w : Claims.Flock.Wit) :
    Claims.Flock.RelS s w ↔ ∀ j < s.tables, ∃ t, table zk Readable.relation x j = some t ∧
      Claims.Flock.RelS t (Readable.tableWit s.tables w j) := by
  obtain ⟨p, -, rfl, hP, hsz, htab⟩ := parsed_of_parse hs
  obtain ⟨-, -, -, -, -, ht, -, -, -, hpi, -⟩ := hin
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at ht hpi
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hpi])
  rw [Readable.relS_only hP ht hpi' w, ← hsz]
  simp only [htab, Option.some.injEq, exists_eq_left']

end KS.Gen

/-- **The deployed relation splits by table**: at an in-scope statement the claims' parser accepts, a witness is in the
relation exactly when, for every table `j`, its share (`Readable.tableWit`: hidden values `a·J + j`) is in the relation
of table `j`'s own statement (`table`). -/
theorem relS_split {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse zk Readable.relation x = some s) (hin : Claims.Flock.InScope zk s) (w : Claims.Flock.Wit) :
    Claims.Flock.RelS s w ↔ ∀ j < s.tables, ∃ t, table zk Readable.relation x j = some t ∧
      Claims.Flock.RelS t (Readable.tableWit s.tables w j) :=
  KS.Gen.relS_split (S := KS.S₀) (zk := zk) (x := x) (s := s) (hs := hs) (hin := hin) (w := w)

/-- **At one table, the table's own statement is the statement.** -/
theorem table_zero_of_tables_eq_one {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {s : Claims.Flock.Stmt} (hs : parse zk rel x = some s) (h1 : s.tables = 1) : table zk rel x 0 = some s := by
  obtain ⟨p, -, rfl, -, hsz, htab⟩ := parsed_of_parse hs
  have h1' : p.sts.size = 1 := hsz.trans h1
  have hst : #[p.sts[0]!] = p.sts := by
    apply Array.ext
    · simp [h1']
    · intro i hi1 _
      obtain rfl : i = 0 := by simpa using hi1
      simp [getElem!_pos p.sts 0 (by omega)]
  have hp1 : p.only 0 = p := by
    show { p with sts := #[p.sts[0]!] } = p
    rw [hst]
  rw [htab, hp1]

/-- **What `assemble` returns at the deployed parser is in the relation**, at an in-scope statement and a read step that
reads the tables' traces interleaved (`Claims.Flock.readTables`): each table's choice, read alone, is in its own
statement's relation, and read as one witness each table's values are its share (`Claims.Flock.readTables_getD`), so the
split (`relS_split`) gives the statement's. -/
theorem relS_of_assemble {zk : Bool} {Col T : Type} (r : T → Claims.Flock.Wit) (decode : List (ℕ × Col) → List T)
    {x : Claims.Msg} {opens : List (ℕ → List (ℕ × Col))} {s : Claims.Flock.Stmt} {w : Claims.Flock.Wit}
    (hs : parse zk Readable.relation x = some s) (hin : Claims.Flock.InScope zk s)
    (h : Claims.Flock.assemble (parse zk Readable.relation) (table zk Readable.relation) decode
      (Claims.Flock.readTables r) x opens = some w) :
    Claims.Flock.RelS s w := by
  obtain ⟨Ms, rfl, hlen, hall⟩ := Claims.Flock.assemble_eq_some hs h
  rw [relS_split hs hin]
  intro j hj
  obtain ⟨t, ht, -, hR⟩ := hall j (by omega)
  refine ⟨t, ht, (Readable.relS_congr t fun i => ?_).2 hR⟩
  rw [Readable.tableWit_getD hj, ← hlen, Claims.Flock.readTables_getD r Ms (by omega),
    Claims.Flock.readTables_singleton]

/-- **The masked record's `assemble` at one table reads as before**: the first decoded trace of table `0`'s columns whose
read is in the relation. -/
theorem assemble_deployZk_one (rel : Readable.Parsed → Readable.Relation)
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace) (x : Claims.Msg)
    (opens : List (ℕ → List (ℕ × List FlockLevel3.GF128))) {s : Claims.Flock.Stmt}
    (hs : (deployZk rel).parse x = some s) (h1 : s.tables = 1) :
    Claims.Flock.assemble (deployZk rel).parse (deployZk rel).table decode (deployZk rel).read x opens =
      ((decode (opens.flatMap fun o => o 0)).map Claims.Flock.zkReadD).find? fun w => decide (Claims.Flock.RelS s w) := by
  rw [Claims.Flock.assemble_of_tables_eq_one (deployZk rel).parse (deployZk rel).table decode (deployZk rel).read x opens
    s hs h1 (table_zero_of_tables_eq_one hs h1)]
  simp only [deployZk, Claims.Flock.readTables_singleton]

/-- **The plain record's `assemble` at one table reads as before**: the first decoded trace of table `0`'s columns whose
read is in the relation. -/
theorem assemble_deployPlain_one (rel : Readable.Parsed → Readable.Relation)
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (x : Claims.Msg)
    (opens : List (ℕ → List (ℕ × List FlockLevel3.GF128))) {s : Claims.Flock.Stmt}
    (hs : (deployPlain rel).parse x = some s) (h1 : s.tables = 1) :
    Claims.Flock.assemble (deployPlain rel).parse (deployPlain rel).table decode (deployPlain rel).read x opens =
      ((decode (opens.flatMap fun o => o 0)).map Claims.Flock.plainReadD).find? fun w =>
        decide (Claims.Flock.RelS s w) := by
  rw [Claims.Flock.assemble_of_tables_eq_one (deployPlain rel).parse (deployPlain rel).table decode
    (deployPlain rel).read x opens s hs h1 (table_zero_of_tables_eq_one hs h1)]
  simp only [deployPlain, Claims.Flock.readTables_singleton]

end FlockDeploy
