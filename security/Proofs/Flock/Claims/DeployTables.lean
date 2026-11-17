module

public import Proofs.Flock.Claims.DeployCode
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Soundness.Refine.Walk

/-! The deployed records' tables are the parsed statement's. -/

@[expose] public section

namespace FlockDeploy

open Flock
open FlockSoundness.Refine FlockSoundness.Discharge.Exec

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

theorem setupOf_size_eq_tables {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {i : Ins} {spec : SessionSpec} {sts : Array Stmt} {s : Claims.Flock.Stmt}
    (h : setupOf zk x = some (i, spec, sts)) (hp : parse zk rel x = some s) : sts.size = s.tables :=
  KS.Gen.setupOf_size_eq_tables (S := KS.S₀) (zk := zk) (rel := rel) (x := x) (i := i) (spec := spec) (sts := sts)
    (s := s) (h := h) (hp := hp)

namespace KS.Gen

variable [S : ZkSetup]

theorem tables_length {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {s : Claims.Flock.Stmt} (hp : parse zk rel x = some s) (hx : (setupOf zk x).isSome) :
    (tables zk x).length = s.tables := by
  obtain ⟨⟨i, spec, sts⟩, h⟩ := Option.isSome_iff_exists.1 hx
  unfold tables
  rw [h, List.length_map, Array.length_toList]
  exact setupOf_size_eq_tables h hp

end KS.Gen

theorem tables_length {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {s : Claims.Flock.Stmt} (hp : parse zk rel x = some s) (hx : (setupOf zk x).isSome) :
    (tables zk x).length = s.tables :=
  KS.Gen.tables_length (S := KS.S₀) (zk := zk) (rel := rel) (x := x) (s := s) (hp := hp) (hx := hx)

theorem size_of_parses {p : Readable.Parsed} (hp : Readable.Parses p) : p.sts.size = max 1 p.I.count := by
  obtain ⟨spec, hset⟩ := Readable.setup_of_tablesOf (Readable.tablesOf_of_parses hp)
  exact setupTables_size hset

namespace KS.Gen

variable [S : ZkSetup]

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

theorem parsed_of_parse {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {s : Claims.Flock.Stmt} (hs : parse zk rel x = some s) :
    ∃ p : Readable.Parsed, Readable.parseBytes (readOf zk) x = some p ∧ s = Readable.toReadable rel p ∧
      Readable.Parses p ∧ p.sts.size = s.tables ∧ ∀ j, table zk rel x j = some (Readable.toReadable rel (p.only j)) :=
  KS.Gen.parsed_of_parse (S := KS.S₀) (zk := zk) (rel := rel) (x := x) (s := s) (hs := hs)

namespace KS.Gen

variable [S : ZkSetup]

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

theorem relS_split {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse zk Readable.relation x = some s) (hin : Claims.Flock.InScope zk s) (w : Claims.Flock.Wit) :
    Claims.Flock.RelS s w ↔ ∀ j < s.tables, ∃ t, table zk Readable.relation x j = some t ∧
      Claims.Flock.RelS t (Readable.tableWit s.tables w j) :=
  KS.Gen.relS_split (S := KS.S₀) (zk := zk) (x := x) (s := s) (hs := hs) (hin := hin) (w := w)

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

theorem assemble_deployZk_one (rel : Readable.Parsed → Readable.Relation)
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace) (x : Claims.Msg)
    (opens : List (ℕ → List (ℕ × List FlockLevel3.GF128))) {s : Claims.Flock.Stmt}
    (hs : (deployZk rel).parse x = some s) (h1 : s.tables = 1) :
    Claims.Flock.assemble (deployZk rel).parse (deployZk rel).table decode (deployZk rel).read x opens =
      ((decode (opens.flatMap fun o => o 0)).map Claims.Flock.zkReadD).find? fun w => decide (Claims.Flock.RelS s w) := by
  rw [Claims.Flock.assemble_of_tables_eq_one (deployZk rel).parse (deployZk rel).table decode (deployZk rel).read x opens
    s hs h1 (table_zero_of_tables_eq_one hs h1)]
  simp only [deployZk, Claims.Flock.readTables_singleton]

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
