import Proofs.Flock.Claims.Recursion.MoreSlots
import Proofs.Flock.C3.VStarList.Shape
import Proofs.Flock.C3.PubRows.Shape

/-!
# The chain's `acc` holdings are instances of their statement (`AccFits`)

`RecVStar.accReads` reads the `rec-acc` chain at a level's `acc`: at the table's level `l`, over `2 q` units
(`fast100` at the inner `m`), at R's level `l`, over `q` units (`fast100` at R's `m`). The verifier's list
(`RecVStar.statements`) holds the table's level `l` as V\*'s `recOpen l … (2 q)` (`statementsOf_eq`, at
`VStar.Algebra.Shape.sched`, which is `fast100` at the inner `m`), and R's level `l` as candidate 3's
`recOpenStacked l … q` (`candidate3Statements`, at `fast100` at R's `m`); no other statement reads `acc`. So every
unit a statement reads `acc` at is below its instances.
-/

namespace RecDeploy.MoreSlots

open Flock

/-! ## The verifier's list -/

/-- **The verifier's list**, as `FlockC3.VStarList.statements_ok`, at V\*'s shape with the inner `m`. -/
theorem statements_vsh {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    ∃ vsh A C K n, VStar.statementsOf vsh i.holo = .ok A ∧ RecVStar.candidate3Statements i = .ok C ∧
      VStar.levelCapStatements vsh = .ok K ∧ VStar.capStatement vsh = .ok (.cap 0 n) ∧
      ss = A ++ C ++ (K ++ #[.cap 0 n]) ∧ vsh.m = i.m := by
  unfold RecVStar.statements at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  split at h
  · cases h
  obtain ⟨ps, hps, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨gs, hgs, h⟩ := FlockC3.VStarList.eok_bind h
  split at h
  · cases h
  split at h
  · cases h
  obtain ⟨K, hK, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨t, ht, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨A, hA, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨C, hC, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨K0, hK0, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨n, rfl⟩ := FlockC3.VStarList.capStatement_ok hK0
  obtain ⟨u, hu, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u', -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u'', -, h⟩ := FlockC3.VStarList.eok_bind h
  cases h
  exact ⟨_, A, C, K, n, hA, hC, hK, hK0, Array.append_assoc, rfl⟩

/-- V\*'s schedule at a shape is `fast100`'s at its `m`. -/
theorem sched_fast100 {vsh : VStar.Algebra.Shape} {s : Schedule} (h : vsh.sched = .ok s) :
    fast100 vsh.m = some s := by
  unfold VStar.Algebra.Shape.sched at h
  cases hf : fast100 vsh.m with
  | none => rw [hf] at h; cases h
  | some s' =>
    rw [hf] at h
    dsimp only at h
    split at h
    · cases h
    split at h
    · cases h
    cases h
    rfl

/-- Candidate 3's statements are no table level's. -/
theorem recOpen_not_c3 {i : RecVStar.Inner} {C : Array VStar.Stmt} (hC : RecVStar.candidate3Statements i = .ok C)
    {l L H n : ℕ} : VStar.Stmt.recOpen l L H n ∉ C := by
  intro hm
  obtain ⟨hnone, -⟩ := FlockC3.VStarList.candidate3_ok hC
  cases hsp : i.sparse with
  | none =>
    rw [hnone hsp] at hm
    simp at hm
  | some sp =>
    obtain ⟨rest, hCe, hk⟩ := FlockC3.PubRows.candidate3_rest hC hsp
    rw [hCe] at hm
    rcases Array.mem_append.1 hm with hm | hm
    · simp at hm
    · rcases hk _ hm with ⟨_, _, _, _, he⟩ | ⟨_, _, _, _, _, _, he⟩ <;> cases he

/-- **Candidate 3's R levels**: `recOpenStacked l … n` is R's level `l` over its `q` openings, at `fast100` at R's
`m`. -/
theorem stacked_mem {i : RecVStar.Inner} {C : Array VStar.Stmt} (h : RecVStar.candidate3Statements i = .ok C)
    {l L H n : ℕ} (hm : VStar.Stmt.recOpenStacked l L H n ∈ C) :
    ∃ r sch, i.stacked = some r ∧ fast100 r.m = some sch ∧ n = (sch.levels[l]!).q := by
  unfold RecVStar.candidate3Statements at h
  cases hsp : i.sparse with
  | none =>
    cases hr : i.stacked with
    | none =>
      rw [hsp, hr] at h
      cases h
      simp at hm
    | some r =>
      rw [hsp, hr] at h
      cases h
  | some sp =>
    rw [hsp] at h
    dsimp only at h
    split at h
    · cases h
    cases hr : i.stacked with
    | none =>
      rw [hr] at h
      cases h
    | some r =>
      rw [hr] at h
      dsimp only at h
      split at h
      · cases h
      split at h
      · cases h
      split at h
      rotate_left
      · cases h
      rename_i sch hsch
      obtain ⟨x, hx, h⟩ := FlockC3.VStarList.eok_bind h
      cases h
      refine ⟨r, sch, rfl, hsch, ?_⟩
      rcases Array.mem_append.1 hm with hm | hm
      · rcases Array.mem_append.1 hm with hm | hm
        · simp at hm
        · obtain ⟨l', -, he⟩ := Array.mem_map.1 hm
          cases he
          rfl
      · obtain ⟨_, _, _, _, _, _, he⟩ := FlockC3.PubRows.openingParts_kind hx _ hm
        cases he

/-! ## What `accReads` reads -/

theorem accReads_recOpen {i : RecVStar.Inner} {sch : Schedule} (hf : fast100 i.m = some sch) (l L H n : ℕ) :
    ∃ Xs, RecVStar.accReads i (.recOpen l L H n) = .ok (some Xs) ∧ Xs.size = 2 * (sch.levels[l]!).q :=
  ⟨_, by unfold RecVStar.accReads; simp only [hf]; try rfl, by simp⟩

theorem accReads_stacked {i : RecVStar.Inner} {r : RecVStar.StackedShape} {sch sr : Schedule}
    (hf : fast100 i.m = some sch) (hr : i.stacked = some r) (hsr : fast100 r.m = some sr) (l L H n : ℕ) :
    ∃ Xs, RecVStar.accReads i (.recOpenStacked l L H n) = .ok (some Xs) ∧ Xs.size = (sr.levels[l]!).q :=
  ⟨_, by unfold RecVStar.accReads; simp only [hf, hr, hsr]; try rfl, by simp⟩

/-- Where a statement reads `acc` at unit `u`, `u` is one of `accReads`' units. -/
theorem lt_of_accRead {r : XR} {st : VStar.Stmt} {u : ℕ} {Xs : Array (Option ℕ)}
    (hX : RecVStar.accReads r.shape st = .ok (some Xs)) (hi : ∀ j nm pt, st ≠ .inner j nm pt)
    (h : (accRead r st "acc" u).isSome = true) : u < Xs.size := by
  have e : accRead r st "acc" u = (Xs[u]?).bind id := by
    cases st with
    | inner j nm pt => exact absurd rfl (hi j nm pt)
    | _ => unfold accRead; simp [hX, Except.toOption]
  rw [e] at h
  apply Classical.byContradiction
  intro hu
  rw [Array.getElem?_eq_none (by omega)] at h
  simp at h

theorem lt_recOpen {r : XR} {sch : Schedule} (hf : fast100 r.shape.m = some sch) {l L H n u : ℕ}
    (h : (accRead r (.recOpen l L H n) "acc" u).isSome = true) : u < 2 * (sch.levels[l]!).q := by
  obtain ⟨Xs, hX, hsz⟩ := accReads_recOpen hf l L H n
  have := lt_of_accRead hX (fun _ _ _ he => by cases he) h
  omega

theorem lt_stacked {r : XR} {R : RecVStar.StackedShape} {sch sr : Schedule} (hf : fast100 r.shape.m = some sch)
    (hR : r.shape.stacked = some R) (hsr : fast100 R.m = some sr) {l L H n u : ℕ}
    (h : (accRead r (.recOpenStacked l L H n) "acc" u).isSome = true) : u < (sr.levels[l]!).q := by
  obtain ⟨Xs, hX, hsz⟩ := accReads_stacked hf hR hsr l L H n
  have := lt_of_accRead hX (fun _ _ _ he => by cases he) h
  omega

/-! ## `AccFits` -/

/-- **`AccFits` holds**: a statement reads `acc` only at a table level (`recOpen l … (2 q)`, V\*'s, `2 q` units) or
an R level (`recOpenStacked l … q`, candidate 3's, `q` units), at the schedule its instances count. -/
theorem accFits : AccFits := by
  intro r ss s sv hss hsv u h
  have hmem : sv ∈ ss := Array.mem_of_getElem? hsv
  obtain ⟨vsh, A, C, K, N, hA, hC, hK, -, hse, hvm⟩ := statements_vsh hss
  subst hse
  obtain ⟨sc, sp, ps, hsc, -, -, hAe⟩ := FlockVBridge.Family.statementsOf_eq hA
  have hf : fast100 r.shape.m = some sc := by rw [← hvm]; exact sched_fast100 hsc
  have hcap : ∀ {st : VStar.Stmt}, st ∈ K ++ #[.cap 0 N] → ∃ l N', st = .cap l N' :=
    fun hs => FlockC3.VStarList.caps_cap hK N _ hs
  cases sv with
  | inner j nm pt =>
    unfold accRead at h
    have e1 : ("acc" : String) ≠ "acc0" := by decide
    have e2 : ("acc" : String) ≠ "acc1" := by decide
    simp [e1, e2] at h
  | recOpen l L H n =>
    have hA' : VStar.Stmt.recOpen l L H n ∈ A := by
      rcases Array.mem_append.1 hmem with h' | h'
      · rcases Array.mem_append.1 h' with h' | h'
        · exact h'
        · exact absurd h' (recOpen_not_c3 hC)
      · obtain ⟨_, _, he⟩ := hcap h'
        cases he
    rw [hAe] at hA'
    rcases Array.mem_append.1 hA' with h' | h'
    · obtain ⟨l', -, he⟩ := List.mem_map.1 (List.mem_toArray.1 h')
      cases he
      exact lt_recOpen hf h
    · obtain ⟨j, -, he⟩ := List.mem_map.1 (List.mem_toArray.1 h')
      cases he
  | recOpenStacked l L H n =>
    have hC' : VStar.Stmt.recOpenStacked l L H n ∈ C := by
      rcases Array.mem_append.1 hmem with h' | h'
      · rcases Array.mem_append.1 h' with h' | h'
        · rw [hAe] at h'
          rcases Array.mem_append.1 h' with h' | h'
          · obtain ⟨_, -, he⟩ := List.mem_map.1 (List.mem_toArray.1 h')
            cases he
          · obtain ⟨_, -, he⟩ := List.mem_map.1 (List.mem_toArray.1 h')
            cases he
        · exact h'
      · obtain ⟨_, _, he⟩ := hcap h'
        cases he
    obtain ⟨R, sr, hR, hsr, rfl⟩ := stacked_mem hC hC'
    exact lt_stacked hf hR hsr h
  | _ =>
    unfold accRead RecVStar.accReads at h
    simp [Except.toOption, pure, Except.pure] at h

end RecDeploy.MoreSlots
