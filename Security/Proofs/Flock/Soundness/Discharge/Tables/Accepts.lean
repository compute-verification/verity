import Proofs.Flock.Soundness.Discharge.Exec.Setup

/-!
# flock-verify's tables at `count > 1`

`Stmt.setupTables` at `count > 1`, under a tag without hidden outputs (`setupTables_parts`): the tag allows several
tables and `count ≤ Tags.MAX_TABLES`; the draw, when there is one, is a subset draw, and the session's units are its
units (without one, the population's every unit); they split into `count` nonempty equal parts; and table `j` is
`setupH`'s statement under `Tags.tableIdentity count j` at the subset draw of part `j`, the consecutive units
`j·k … j·k + k − 1` with `k = units/count`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Tables

open Flock Refine Exec
open Lean (Json)

theorem splits_ok {units : Array ℕ} {count : ℕ} {e : String} {w : Unit}
    (h : (if (units.size == 0 || units.size % count != 0) = true then (throw e : V Unit) else pure ()) = .ok w) :
    0 < units.size ∧ units.size % count = 0 := by
  by_cases hc : (units.size == 0 || units.size % count != 0) = true
  · rw [ite_eq_left hc] at h
    exact absurd h (throw_ne_ok _ _)
  · simp only [Bool.or_eq_true, beq_iff_eq, bne_iff_ne, ne_eq, not_or, not_not] at hc
    omega

theorem draw_subset {draw : Option Json} {u u' : Draw.UnitDraw} {k : ℕ}
    (hd : draw.mapM Draw.UnitDraw.ofJson = .ok (some u)) (hlaw : u.law = .subset k) (huu : some u = some u') :
    (∀ d, draw = some d → ∃ v, Draw.UnitDraw.ofJson d = .ok v ∧ (∃ k', v.law = .subset k') ∧
      v.population = u'.population ∧ v.units = u'.units) ∧ (draw = none → u'.units = Array.range u'.population) := by
  cases huu
  cases draw with
  | none => simp [pure, Except.pure] at hd
  | some d =>
    refine ⟨fun d' hd' => ?_, fun h => by cases h⟩
    cases hd'
    rw [Option.mapM_some] at hd
    cases h1 : Draw.UnitDraw.ofJson d with
    | error e =>
      rw [h1] at hd
      cases hd
    | ok v =>
      rw [h1] at hd
      simp only [Functor.map, Except.map, Except.ok.injEq, Option.some.injEq] at hd
      subst hd
      exact ⟨_, rfl, ⟨k, hlaw⟩, rfl, rfl⟩

theorem draw_none {draw : Option Json} (hd : draw.mapM Draw.UnitDraw.ofJson = .ok none) : draw = none := by
  cases draw with
  | none => rfl
  | some d =>
    rw [Option.mapM_some] at hd
    cases h1 : Draw.UnitDraw.ofJson d with
    | error e =>
      rw [h1] at hd
      cases hd
    | ok v =>
      rw [h1] at hd
      simp [Functor.map, Except.map] at hd

theorem no_some {u : Draw.UnitDraw} (h : ∀ u' : Draw.UnitDraw, some u = some u' → False) : False := h u rfl

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Stmt}

set_option maxHeartbeats 4000000 in
/-- **`setupTables`'s tables at `count > 1`** (§16.12) under a tag without hidden outputs (`hh`). -/
theorem setupTables_parts (hh : tags.hiddenOutputs = false) (hc1 : 1 < count)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    tags.manyTables.isSome ∧ count ≤ Tags.MAX_TABLES ∧
      ∃ (population : ℕ) (units : Array ℕ),
        (∀ d, draw = some d → ∃ u, Draw.UnitDraw.ofJson d = .ok u ∧ (∃ k', u.law = .subset k') ∧
          u.population = population ∧ u.units = units) ∧
        (draw = none → units = Array.range population) ∧
        0 < units.size ∧ units.size % count = 0 ∧ sts.size = count ∧
        ∀ (j : ℕ) (hj : j < sts.size),
          Stmt.setupH { tags with identity := tags.tableIdentity count j } cf pf coins tables
            (some (Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson population),
              ("k", Lean.toJson (units.size / count)),
              ("units", Json.arr ((units.extract (j * (units.size / count))
                (j * (units.size / count) + units.size / count)).map Lean.toJson))]))
            partition program = .ok sts[j] := by
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
  all_goals try omega
  all_goals try contradiction
  all_goals try exact (no_some (by assumption)).elim
  all_goals
    obtain ⟨hsz, hall⟩ := range_mapM_ok (by assumption)
    obtain ⟨hpos, hmod⟩ := splits_ok (by assumption)
  · obtain ⟨hD1, hD2⟩ := draw_subset (by assumption) (by assumption) (by assumption)
    exact ⟨Option.isSome_iff_exists.2 ⟨_, by assumption⟩, by omega, _, _, hD1, hD2, hpos, hmod, hsz, fun j hj => ok_of_refuse (hall j hj)⟩
  · have hdn := draw_none (by assumption)
    exact ⟨Option.isSome_iff_exists.2 ⟨_, by assumption⟩, by omega, _, _, fun d hd => by simp [hdn] at hd,
      fun _ => rfl, hpos, hmod, hsz, fun j hj => ok_of_refuse (hall j hj)⟩

end FlockSoundness.Discharge.Tables
