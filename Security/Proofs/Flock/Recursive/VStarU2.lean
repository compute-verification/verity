import Proofs.Flock.Recursive.VStarStrata
import Proofs.Flock.Soundness.ExecDraw
import Proofs.Flock.Soundness.Discharge.ZkLink.DrawOk
import Proofs.Flock.Soundness.Discharge.Exec.Pts
import Proofs.Flock.Soundness.Discharge.Layout.Drawn

/-!
# Recursion: V*'s strata tile at the verifier's U2 check (gap 7)

`RecursiveAudit`'s `_hS` says each of V*'s laws has its executable strata `St` as its model's (`Law.ExecStrata`), and
`VLaw.execStrata_iff` splits that into `Law.Tiles St nv` and the model read off `St`. Here `Tiles` comes from what the
verifier checks: `UnitDraw.ofJson` decodes a stratified draw only once its strata pass U2 (`Flock.Draw.strataProblem`
finds nothing, and `stratumOfJson` gave each stratum non-empty ranges with `lo < hi`), and U2 sorts the strata's ranges
and checks they abut from `0` to the population. Each session of V*'s statement decodes its draw file at `S₀` (`y₀`), so
once the draw file is tied to `St` (`DrawFileZK`) the strata tile the population.

* `U2.tiles_of_u2`: strata that pass U2 tile `[0, pop)`. A unit below `pop` is in exactly one range of the sorted
  concatenation (`U2.hitsR_abuts`), so in exactly one stratum's units (`U2.total_mult`); U2's draw count `stratumK` is at
  most the stratum's size. `U2.ofJson_strata`: a draw `UnitDraw.ofJson` decodes under a stratified law has passed U2.
* `tiles_of_drawFile`, `VStmt.tiles_of_session`: the strata tile wherever the draw file tied to them decodes, so at
  every session. `VStmt.pop_of_session`: the population is the registered public file's count.
* `vstar_hS_of_sessions`: `_hS` from one session of each of V*'s statements. Left to the caller, as the verifier
  doesn't check them: the draw file tied to the strata (`DrawFileZK`, custody of the record's `unit_draw`), the law's
  population being V*'s units (`pop = nv`; `vstar_hS_of_sessions_reg` takes the registered file's count `nv` instead),
  and the model being the one read off the strata (`hσ`, `hk`).
* `VStmt.withExec`: the statement with its model read off its strata (`VLaw.ofExec`). It has the same live law, so the
  same sessions, and `vstar_hS_withExec`: at those statements `_hS` holds (`vstar_execStrata`), with `DrawFileZK` over
  `[0, nv)` the only premise left.
-/

open private Flock.Draw.stratumOfJson from Flock.Draw

namespace FlockSoundness.Discharge.Recursion.U2

open Finset FlockSoundness.Refine

/-! ## U2's loops -/

/-- The ranges `l` abut from `a` to `b`: the first starts at `a`, each next where the last ended, the last ends at `b`. -/
def Abuts : ℕ → List (ℕ × ℕ) → ℕ → Prop
  | a, [], b => a = b
  | a, r :: l, b => r.1 = a ∧ Abuts r.2 l b

section
variable {α : Type} (P : α → Prop)
  (f : α → Option (Option String) × Unit → Id (ForInStep (Option (Option String) × Unit)))
  (hf : ∀ a st, (P a ∧ f a st = pure (.yield (none, ()))) ∨ ∃ m, f a st = pure (.done (some (some m), ())))
include hf

theorem forIn_err : ∀ (l : List α) {r : Option String}, (forIn l (none, ()) f).run.fst = some r → ∃ m, r = some m
  | [], r, h => by simp at h
  | b :: l, r, h => by
      rw [List.forIn_cons] at h
      rcases hf b (none, ()) with ⟨-, he⟩ | ⟨m, he⟩
      · rw [he] at h
        exact forIn_err l h
      · rw [he] at h
        simp only [pure_bind, Id.run_pure, Option.some.injEq] at h
        exact ⟨m, h.symm⟩

theorem forIn_all_ok : ∀ (l : List α), (forIn l (none, ()) f).run.fst = none → ∀ a ∈ l, P a
  | [], _, a, ha => by simp at ha
  | b :: l, h, a, ha => by
      rw [List.forIn_cons] at h
      rcases hf b (none, ()) with ⟨hp, he⟩ | ⟨m, he⟩
      · rw [he] at h
        rcases List.mem_cons.1 ha with rfl | ha
        · exact hp
        · exact forIn_all_ok l h a ha
      · rw [he] at h
        simp at h

end

section
variable (msg : String)
  (f : ℕ × ℕ → Option (Option String) × ℕ → Id (ForInStep (Option (Option String) × ℕ)))
  (hf : ∀ x st, f x st =
    if (x.1 != st.2) = true then pure (.done (some (some msg), st.2)) else pure (.yield (none, x.2)))
include hf

theorem forIn_abut_err : ∀ (l : List (ℕ × ℕ)) (a : ℕ) {r : Option String},
    (forIn l (none, a) f).run.fst = some r → r = some msg
  | [], a, r, h => by simp at h
  | x :: l, a, r, h => by
      rw [List.forIn_cons, hf] at h
      by_cases hx : (x.1 != a) = true
      · rw [ite_eq_left hx] at h
        simp only [pure_bind, Id.run_pure, Option.some.injEq] at h
        exact h.symm
      · rw [ite_eq_right hx] at h
        exact forIn_abut_err l x.2 h

theorem forIn_abuts : ∀ (l : List (ℕ × ℕ)) (a : ℕ), (forIn l (none, a) f).run.fst = none →
    Abuts a l (forIn l (none, a) f).run.snd
  | [], a, _ => by simp [Abuts]
  | x :: l, a, h => by
      rw [List.forIn_cons, hf] at h ⊢
      by_cases hx : (x.1 != a) = true
      · rw [ite_eq_left hx] at h
        simp at h
      · rw [ite_eq_right hx] at h ⊢
        exact ⟨by simpa using hx, forIn_abuts l x.2 h⟩

end

/-- **What U2 checks**: when `strataProblem` finds nothing, `1 ≤ k ≤ pop`, each stratum draws `stratumK` of its size,
and the strata's ranges, sorted by their starts, abut from `0` to `pop`. -/
theorem strataProblem_none {k : ℕ} {strata : Array Flock.Draw.Stratum} {pop : ℕ}
    (h : Flock.Draw.strataProblem k strata pop = none) :
    1 ≤ k ∧ k ≤ pop ∧ (∀ i < strata.size, strata[i]!.k = Flock.Draw.stratumK k strata[i]!.size pop) ∧
      Abuts 0 ((strata.foldl (fun x1 x2 => x1 ++ x2.units) #[]).qsort fun a b => decide (a.1 < b.1)).toList pop := by
  unfold Flock.Draw.strataProblem at h
  split at h
  · simp at h
  · rename_i hg
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hg
    simp only [Id.run_bind, Std.Legacy.Range.forIn_eq_forIn_range'] at h
    split at h
    · rename_i r heq
      have hr : r = none := h
      subst hr
      obtain ⟨m, hm⟩ := forIn_err (P := fun _ => True) _ (fun a st => by
        split
        · exact .inr ⟨_, rfl⟩
        · split
          · exact .inr ⟨_, rfl⟩
          · exact .inl ⟨trivial, rfl⟩) _ heq
      cases hm
    · rename_i heq
      have hall := forIn_all_ok (P := fun i => strata[i]!.k = Flock.Draw.stratumK k strata[i]!.size pop) _
        (fun a st => by
          split
          · exact .inr ⟨_, rfl⟩
          · split
            · exact .inr ⟨_, rfl⟩
            · rename_i hB
              exact .inl ⟨by simpa using hB, rfl⟩) _ heq
      refine ⟨by omega, by omega, fun i hi => hall i (by simp [List.mem_range'] ; omega), ?_⟩
      rw [← Array.forIn_toList, Id.run_bind] at h
      split at h
      · rename_i r heq2
        have hr : r = none := h
        subst hr
        cases forIn_abut_err _ _ (fun x st => rfl) _ _ heq2
      · rename_i heq2
        have hc := forIn_abuts _ _ (fun x st => rfl) _ _ heq2
        split at h
        · rename_i hp
          rw [beq_iff_eq] at hp
          rw [hp] at hc
          exact hc
        · cases h

/-! ## The decoder -/

theorem mapM_list_ok {α β : Type} (f : α → Flock.V β) (P : β → Prop) (hf : ∀ a b, f a = .ok b → P b) :
    ∀ (l : List α) (r : List β), l.mapM f = .ok r → ∀ b ∈ r, P b
  | [], r, h => by
      simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst h
      simp
  | a :: l, r, h => by
      simp only [List.mapM_cons] at h
      obtain ⟨b, hb, h⟩ := ebind_ok h
      obtain ⟨bs, hbs, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact hf a x hb
      · exact mapM_list_ok f P hf l bs hbs x hx

theorem mapM_arr_ok {α β : Type} (f : α → Flock.V β) (P : β → Prop) (hf : ∀ a b, f a = .ok b → P b)
    (us : Array α) (r : Array β) (h : us.mapM f = .ok r) : ∀ b ∈ r, P b := by
  obtain ⟨l⟩ := us
  rw [List.mapM_toArray] at h
  cases hl : l.mapM f with
  | error e => rw [hl] at h; cases h
  | ok bs =>
      rw [hl] at h
      simp only [Functor.map, Except.map, Except.ok.injEq] at h
      subst h
      intro b hb
      exact mapM_list_ok f P hf l bs hl b (by simpa using hb)

/-- A stratum `stratumOfJson` decodes has ranges, each with `lo < hi`. -/
theorem stratumOfJson_ok {j : Lean.Json} {s : Flock.Draw.Stratum} (h : Flock.Draw.stratumOfJson j = .ok s) :
    s.units ≠ #[] ∧ ∀ r ∈ s.units, r.1 < r.2 := by
  unfold Flock.Draw.stratumOfJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    dsimp only
    refine ⟨fun he => ?_, mapM_arr_ok _ (fun r : ℕ × ℕ => r.1 < r.2) (fun a b h => ?_) _ _
      ‹Array.mapM _ _ = Except.ok _›⟩
    · subst he
      exact absurd rfl ‹¬(#[] : Array (ℕ × ℕ)).isEmpty = true›
    · try dsimp only at h
      repeat' walk_step
      all_goals
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        dsimp only
        omega

/-- **A stratified draw `UnitDraw.ofJson` decodes has passed U2 on its strata**: `strataProblem` finds nothing at its
population, and every stratum has ranges, each with `lo < hi`. -/
theorem ofJson_strata {j : Lean.Json} {D : Flock.Draw.UnitDraw} (h : Flock.Draw.UnitDraw.ofJson j = .ok D) {K : ℕ}
    {strata : Array Flock.Draw.Stratum} (hl : D.law = .stratified K strata) :
    Flock.Draw.strataProblem K strata D.population = none ∧
      ∀ s ∈ strata, s.units ≠ #[] ∧ ∀ r ∈ s.units, r.1 < r.2 := by
  unfold Flock.Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
  all_goals
    have hp := ‹(pure _ : Flock.V Flock.Draw.Law) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at hp
  all_goals
    simp only [reduceCtorEq, Flock.Draw.Law.stratified.injEq] at hl
  all_goals
    obtain ⟨rfl, rfl⟩ := hl
    obtain ⟨rfl, rfl⟩ := hp
    exact ⟨Option.eq_none_iff_forall_ne_some.2 ‹_›,
      mapM_arr_ok _ (fun s : Flock.Draw.Stratum => s.units ≠ #[] ∧ ∀ r ∈ s.units, r.1 < r.2)
        (fun _ _ hs => stratumOfJson_ok hs) _ _
        ‹Array.mapM _ _ = (Except.ok _ : Flock.V (Array Flock.Draw.Stratum))›⟩

/-! ## Counting the units -/

theorem qsort_perm {α : Type} (as : Array α) (lt : α → α → Bool) : (as.qsort lt).toList.Perm as.toList := by
  have h : (as.qsort lt).Perm as := by
    unfold Array.qsort
    split
    · exact Array.Perm.refl _
    · exact Audit.DrawExec.sort_perm lt _ _ _ _ _ _ _ rfl
  exact Array.perm_iff_toList_perm.1 h

/-- How many of the ranges `l` hold `u`. -/
def hitsR : List (ℕ × ℕ) → ℕ → ℕ
  | [], _ => 0
  | r :: l, u => (if r.1 ≤ u ∧ u < r.2 then 1 else 0) + hitsR l u

theorem hitsR_append (l₁ l₂ : List (ℕ × ℕ)) (u : ℕ) : hitsR (l₁ ++ l₂) u = hitsR l₁ u + hitsR l₂ u := by
  induction l₁ with
  | nil => simp [hitsR]
  | cons r l ih => simp only [List.cons_append, hitsR, ih, Nat.add_assoc]

theorem hitsR_perm {l₁ l₂ : List (ℕ × ℕ)} (h : l₁.Perm l₂) (u : ℕ) : hitsR l₁ u = hitsR l₂ u := by
  induction h with
  | nil => rfl
  | cons r _ ih => simp only [hitsR, ih]
  | swap r r' l => simp only [hitsR]; exact Nat.add_left_comm _ _ _
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

theorem hitsR_flatten (L : List Flock.Draw.Stratum) (u : ℕ) :
    hitsR (L.map fun s => s.units.toList).flatten u = (L.map fun s => hitsR s.units.toList u).sum := by
  induction L with
  | nil => simp [hitsR]
  | cons s L ih => simp only [List.map_cons, List.flatten_cons, List.sum_cons, hitsR_append, ih]

theorem hitsR_pos {r : ℕ × ℕ} {u : ℕ} (h1 : r.1 ≤ u) (h2 : u < r.2) :
    ∀ {l : List (ℕ × ℕ)}, r ∈ l → 1 ≤ hitsR l u
  | [], hr => by simp at hr
  | a :: l, hr => by
      simp only [hitsR]
      rcases List.mem_cons.1 hr with rfl | hr
      · rw [ite_eq_left ⟨h1, h2⟩]; omega
      · have := hitsR_pos h1 h2 hr; omega

/-- Non-empty ranges that abut from `a` to `b` hold each `u ∈ [a, b)` once and nothing else. -/
theorem hitsR_abuts : ∀ (l : List (ℕ × ℕ)) (a b : ℕ), Abuts a l b → (∀ r ∈ l, r.1 < r.2) →
    a ≤ b ∧ ∀ u, hitsR l u = if a ≤ u ∧ u < b then 1 else 0
  | [], a, b, h, _ => by
      have h' : a = b := h
      refine ⟨h'.le, fun u => ?_⟩
      simp only [hitsR]
      split_ifs <;> omega
  | r :: l, a, b, h, hr => by
      obtain ⟨h1, h2⟩ := h
      have hr1 := hr r (List.mem_cons_self ..)
      obtain ⟨hb, hl⟩ := hitsR_abuts l r.2 b h2 fun r' hr' => hr r' (List.mem_cons_of_mem _ hr')
      refine ⟨by omega, fun u => ?_⟩
      simp only [hitsR, hl u]
      split_ifs <;> omega

/-- How many of a stratum's indices `j` below its size have `u` as their unit. -/
def mult (s : Flock.Draw.Stratum) (u : ℕ) : ℕ := ∑ j ∈ range s.size, if s.unit j = u then 1 else 0

theorem mult_eq_card (s : Flock.Draw.Stratum) (u : ℕ) :
    mult s u = ((range s.size).filter (fun j => s.unit j = u)).card :=
  (Finset.card_filter _ _).symm

theorem sum_range_block (lo hi u : ℕ) :
    (∑ j ∈ range (hi - lo), if lo + j = u then 1 else 0) = if lo ≤ u ∧ u < hi then 1 else 0 := by
  split_ifs with h
  · rw [Finset.sum_eq_single (u - lo)]
    · rw [ite_eq_left (by omega)]
    · intro b _ hb
      rw [ite_eq_right (by omega)]
    · intro hn
      exact absurd (Finset.mem_range.2 (by omega)) hn
  · refine Finset.sum_eq_zero fun j hj => ite_eq_right fun he => h ⟨by omega, ?_⟩
    have := Finset.mem_range.1 hj
    omega

theorem unit_cons_lt (t : String) (k : ℕ) (a : ℕ × ℕ) (l : List (ℕ × ℕ)) {j : ℕ} (hj : j < a.2 - a.1) :
    (Flock.Draw.Stratum.mk t (a :: l).toArray k).unit j = a.1 + j := by
  rw [ZkLink.unit_cons, ite_eq_left hj]

theorem unit_cons_add (t : String) (k : ℕ) (a : ℕ × ℕ) (l : List (ℕ × ℕ)) (x : ℕ) :
    (Flock.Draw.Stratum.mk t (a :: l).toArray k).unit (a.2 - a.1 + x) = (Flock.Draw.Stratum.mk t l.toArray k).unit x := by
  rw [ZkLink.unit_cons, ite_eq_right (Nat.not_lt.2 (Nat.le_add_right _ _)), Nat.add_sub_cancel_left]

theorem mult_cons (t : String) (k : ℕ) (a : ℕ × ℕ) (l : List (ℕ × ℕ)) (u : ℕ) :
    mult ⟨t, (a :: l).toArray, k⟩ u = (if a.1 ≤ u ∧ u < a.2 then 1 else 0) + mult ⟨t, l.toArray, k⟩ u := by
  unfold mult
  rw [ZkLink.size_cons, Finset.sum_range_add, ← sum_range_block a.1 a.2 u]
  congr 1
  · exact Finset.sum_congr rfl fun j hj => by rw [unit_cons_lt t k a l (Finset.mem_range.1 hj)]
  · exact Finset.sum_congr rfl fun x _ => by rw [unit_cons_add]

theorem mult_list (t : String) (k u : ℕ) : ∀ l : List (ℕ × ℕ), mult ⟨t, l.toArray, k⟩ u = hitsR l u
  | [] => by simp [mult, hitsR, Flock.Draw.Stratum.size]
  | a :: l => by
      show _ = (if a.1 ≤ u ∧ u < a.2 then 1 else 0) + hitsR l u
      rw [mult_cons, mult_list t k u l]

theorem mult_eq_hitsR (s : Flock.Draw.Stratum) (u : ℕ) : mult s u = hitsR s.units.toList u := by
  obtain ⟨t, ⟨l⟩, k⟩ := s
  exact mult_list t k u l

theorem foldl_units (L : List Flock.Draw.Stratum) (acc : Array (ℕ × ℕ)) :
    (L.foldl (fun x1 x2 => x1 ++ x2.units) acc).toList = acc.toList ++ (L.map fun s => s.units.toList).flatten := by
  induction L generalizing acc with
  | nil => simp
  | cons s L ih => rw [List.foldl_cons, ih]; simp

variable {m : ℕ}

theorem concat_toList (St : Fin m → Flock.Draw.Stratum) :
    ((Array.ofFn St).foldl (fun x1 x2 => x1 ++ x2.units) #[]).toList =
      ((List.ofFn St).map fun s => s.units.toList).flatten := by
  rw [← Array.foldl_toList, foldl_units, Array.toList_ofFn]
  simp

theorem total_hitsR (St : Fin m → Flock.Draw.Stratum) (u : ℕ) :
    hitsR (((Array.ofFn St).foldl (fun x1 x2 => x1 ++ x2.units) #[]).qsort fun a b => decide (a.1 < b.1)).toList u =
      ∑ s, mult (St s) u := by
  rw [hitsR_perm (qsort_perm _ _) u, concat_toList, hitsR_flatten, List.map_ofFn, List.sum_ofFn]
  exact Finset.sum_congr rfl fun s _ => (mult_eq_hitsR (St s) u).symm

/-- **The strata pass U2**: `stratumOfJson`'s ranges (each stratum's non-empty, each with `lo < hi`), and
`strataProblem` finds nothing over `[0, pop)`. -/
def Ranges (K : ℕ) (St : Fin m → Flock.Draw.Stratum) (pop : ℕ) : Prop :=
  (∀ s, (St s).units ≠ #[] ∧ ∀ r ∈ (St s).units, r.1 < r.2) ∧ Flock.Draw.strataProblem K (Array.ofFn St) pop = none

/-- At strata that pass U2, each unit below `pop` is one of exactly one stratum's, once, and no other unit is any. -/
theorem total_mult {K pop : ℕ} {St : Fin m → Flock.Draw.Stratum} (hU : Ranges K St pop) (u : ℕ) :
    ∑ s, mult (St s) u = if u < pop then 1 else 0 := by
  obtain ⟨-, -, -, hc⟩ := strataProblem_none hU.2
  rw [← total_hitsR]
  obtain ⟨-, hl⟩ := hitsR_abuts _ 0 pop hc fun r hr => by
    have h := (qsort_perm _ _).mem_iff.1 hr
    rw [concat_toList] at h
    simp only [List.mem_flatten, List.mem_map, List.mem_ofFn] at h
    obtain ⟨_, ⟨_, ⟨i, rfl⟩, rfl⟩, hri⟩ := h
    exact (hU.1 i).2 r (by simpa using hri)
  rw [hl u]
  simp

theorem stratumK_le {K size pop : ℕ} (h1 : 1 ≤ size) (hK : K ≤ pop) (hp : 1 ≤ pop) :
    Flock.Draw.stratumK K size pop ≤ size := by
  unfold Flock.Draw.stratumK
  refine max_le h1 (Nat.le_of_lt_succ ((Nat.div_lt_iff_lt_mul (by omega)).2 ?_))
  nlinarith

/-- **Strata that pass U2 tile `[0, pop)`.** -/
theorem tiles_of_u2 {K pop : ℕ} {St : Fin m → Flock.Draw.Stratum} (hU : Ranges K St pop) : Audit.Law.Tiles St pop := by
  have htot := total_mult hU
  obtain ⟨hK1, hKp, hrule, -⟩ := strataProblem_none hU.2
  have hle : ∀ s u, mult (St s) u ≤ ∑ s', mult (St s') u := fun s u =>
    Finset.single_le_sum (f := fun s' => mult (St s') u) (fun _ _ => Nat.zero_le _) (Finset.mem_univ s)
  have hpos : ∀ s j, j < (St s).size → 1 ≤ mult (St s) ((St s).unit j) := fun s j hj => by
    rw [mult_eq_card, Finset.one_le_card]
    exact ⟨j, Finset.mem_filter.2 ⟨Finset.mem_range.2 hj, rfl⟩⟩
  refine ⟨fun s => ?_, fun s j hj => ?_, fun s i hi j hj hij => ?_, fun u => ?_⟩
  · have h0 : 0 < (St s).units.size := Array.size_pos_iff.2 (hU.1 s).1
    have hr := (hU.1 s).2 _ (Array.getElem_mem h0)
    have hc := hitsR_pos (le_refl _) hr (Array.mem_toList_iff.2 (Array.getElem_mem h0))
    rw [← mult_eq_hitsR] at hc
    have hsz : 1 ≤ (St s).size := by
      by_contra hn
      have hz : (St s).size = 0 := by omega
      simp [mult, hz] at hc
    have hk := hrule s.val (by simp)
    rw [getElem!_pos (Array.ofFn St) s.val (by simp), Array.getElem_ofFn] at hk
    rw [hk]
    exact stratumK_le hsz hKp (le_trans hK1 hKp)
  · have h1 := le_trans (hpos s j hj) (hle s _)
    rw [htot] at h1
    by_contra hn
    rw [ite_eq_right hn] at h1
    omega
  · by_contra hne
    have h2 : 2 ≤ mult (St s) ((St s).unit j) := by
      rw [mult_eq_card]
      calc 2 = ({i, j} : Finset ℕ).card := (Finset.card_pair hne).symm
        _ ≤ _ := Finset.card_le_card fun x hx => by
          simp only [Finset.mem_insert, Finset.mem_singleton] at hx
          rcases hx with rfl | rfl
          · exact Finset.mem_filter.2 ⟨Finset.mem_range.2 hi, hij⟩
          · exact Finset.mem_filter.2 ⟨Finset.mem_range.2 hj, rfl⟩
    have h3 := le_trans h2 (hle s _)
    rw [htot] at h3
    split_ifs at h3 <;> omega
  · have h1 := htot u.val
    rw [ite_eq_left u.isLt] at h1
    have hmem : ∀ s, 1 ≤ mult (St s) u.val ↔ u.val ∈ (range (St s).size).image (St s).unit := fun s => by
      rw [mult_eq_card, Finset.one_le_card]
      simp only [Finset.Nonempty, Finset.mem_filter, Finset.mem_range, Finset.mem_image]
    obtain ⟨s, hs⟩ : ∃ s, 1 ≤ mult (St s) u.val := by
      by_contra hn
      have : ∑ s, mult (St s) u.val = 0 := Finset.sum_eq_zero fun s _ => by
        by_contra h0
        exact hn ⟨s, by omega⟩
      omega
    refine ⟨s, (hmem s).1 hs, fun s' hs' => ?_⟩
    by_contra hne
    have h2 := Finset.sum_le_sum_of_subset (f := fun t => mult (St t) u.val) (Finset.subset_univ {s', s})
    rw [Finset.sum_pair hne] at h2
    have := (hmem s').2 hs'
    omega

end FlockSoundness.Discharge.Recursion.U2

namespace FlockSoundness.Discharge.Recursion

open Finset Audit

/-- **The strata tile `[0, pop)` wherever their draw file decodes**: when the draw file `dj` is the verifier's draw
under the stratified law on `St` over `[0, pop)` (`DrawFileZK`), any draw `dj T` that `UnitDraw.ofJson` decodes has
passed U2 on `St`. -/
theorem tiles_of_drawFile {n m : ℕ} {St : Fin m → Flock.Draw.Stratum} {K pop : ℕ} {dj : Finset (Fin n) → Lean.Json}
    (hdj : ZkLink.DrawFileZK dj (.stratified K (Array.ofFn St)) pop) {T : Finset (Fin n)} {D : Flock.Draw.UnitDraw}
    (hD : Flock.Draw.UnitDraw.ofJson (dj T) = .ok D) : Law.Tiles St pop := by
  obtain rfl := hdj T D hD
  obtain ⟨hp, hs⟩ := U2.ofJson_strata hD rfl
  exact U2.tiles_of_u2 ⟨fun s => hs (St s) (by simp), hp⟩

namespace VStmt

variable {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg') {m : ℕ} {St : Fin m → Flock.Draw.Stratum} {K pop : ℕ}

/-- **A session's strata tile**: a session decodes its draw file at `S₀` (`y₀`), so strata its draw file is tied to
tile `[0, pop)`. -/
theorem tiles_of_session (hdj : ZkLink.DrawFileZK z.dj (.stratified K (Array.ofFn St)) pop) : Law.Tiles St pop :=
  tiles_of_drawFile hdj z.y₀.x.hD

/-- **The population is the registered public file's count**: setupH takes the draw's population from the public
file it loads (`HmRow.drawn`), and that is the session's. -/
theorem pop_of_session (hdj : ZkLink.DrawFileZK z.dj (.stratified K (Array.ofFn St)) pop) : pop = z.pub.n := by
  obtain ⟨P, hP, hdr⟩ := FlockSoundness.Discharge.Layout.setupH_drawn z.y₀.x.h
  obtain ⟨D, hD, hpop, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec hdr
  rw [hdj _ D hD] at hpop
  have hc : Flock.HmRow.parse (ZkExec.zkTags z.I) z.I.circuitFile z.I.tables = .ok z.y₀.x.st.c := by
    have h := (Exec.setupH_facts z.y₀.x.h).1
    rwa [show (ZkExec.zkInputs z.I).tags.typed = false from (ZkExec.zkTags_typed z.I).trans z.ht,
      ite_eq_right Bool.false_ne_true] at h
  have hcc : z.y₀.x.st.c = Q.c := Except.ok.inj (hc.symm.trans z.hc)
  rw [hcc] at hP
  have hPP : P = z.pub := Except.ok.inj (hP.symm.trans z.hpub)
  rw [← hPP]
  exact hpop

end VStmt

/-- **`_hS` at one law from its strata's tiling**: strata that tile `[0, pop)`, with `pop` the law's units and its
model the one read off them, are its model's. -/
theorem VLaw.execStrata_of_tiles {nv : ℕ} (law : VLaw nv) (hT : Law.Tiles law.St law.pop) (hpop : law.pop = nv)
    (hσ : ∀ u, u.val ∈ (range (law.St (law.σs u)).size).image (law.St (law.σs u)).unit)
    (hk : ∀ s, law.ks s = (law.St s).k) : Law.ExecStrata law.σs law.ks law.St := by
  rw [hpop] at hT
  exact (VLaw.execStrata_iff law).2 ⟨hT, funext fun u => (hT.stratumOf_eq (hσ u)).symm, funext hk⟩

/-- **Gap 7 at the verifier's U2 check: `RecursiveAudit`'s `_hS`** from one session `z j` of each of V*'s statements
(at any history, `zs R ω t R₂`). The verifier checks the strata at every draw file it decodes; what it doesn't check,
and the caller supplies: the session's draw file is the verifier's draw under the law's own stratified law
(`hdj`, custody of the record's `unit_draw`); the law's population is V*'s units (`hpop`); and the law's model is the
one read off its strata (`hσ`: each unit's stratum holds it; `hk`: each stratum's draw count is its own). -/
theorem vstar_hS_of_sessions {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} (z : (j : Fin S) → (Vs j).Outer Reg')
    (hdj : ∀ j, ZkLink.DrawFileZK (z j).dj (.stratified (Vs j).law.Kd (Array.ofFn (Vs j).law.St)) (Vs j).law.pop)
    (hpop : ∀ j, (Vs j).law.pop = (Vs j).nv)
    (hσ : ∀ j u, u.val ∈ (range ((Vs j).law.St ((Vs j).law.σs u)).size).image ((Vs j).law.St ((Vs j).law.σs u)).unit)
    (hk : ∀ j s, (Vs j).law.ks s = ((Vs j).law.St s).k) :
    ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St := fun j =>
  (Vs j).law.execStrata_of_tiles (VStmt.tiles_of_session (z j) (hdj j)) (hpop j) (hσ j) (hk j)

/-- `vstar_hS_of_sessions` with `hpop` as the verifier sees it: the registered public file's count is V*'s units. -/
theorem vstar_hS_of_sessions_reg {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} (z : (j : Fin S) → (Vs j).Outer Reg')
    (hdj : ∀ j, ZkLink.DrawFileZK (z j).dj (.stratified (Vs j).law.Kd (Array.ofFn (Vs j).law.St)) (Vs j).law.pop)
    (hn : ∀ j, (z j).pub.n = (Vs j).nv)
    (hσ : ∀ j u, u.val ∈ (range ((Vs j).law.St ((Vs j).law.σs u)).size).image ((Vs j).law.St ((Vs j).law.σs u)).unit)
    (hk : ∀ j s, (Vs j).law.ks s = ((Vs j).law.St s).k) :
    ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St :=
  vstar_hS_of_sessions z hdj (fun j => (VStmt.pop_of_session (z j) (hdj j)).trans (hn j)) hσ hk

/-- **A law with its model read off its executable strata** (`VLaw.ofExec`), its live law unchanged. -/
noncomputable def VLaw.withExec {nv : ℕ} (law : VLaw nv) (hT : Law.Tiles law.St nv) : VLaw nv :=
  VLaw.ofExec law.St hT law.Kd law.pop law.Ω law.Lb law.src

theorem VLaw.withExec_live {nv : ℕ} (law : VLaw nv) (hT : Law.Tiles law.St nv) : (law.withExec hT).live = law.live :=
  rfl

/-- **A statement with its law's model read off its executable strata**: the same circuit and live law. -/
noncomputable def VStmt.withExec (Q : VStmt) (hT : Law.Tiles Q.law.St Q.nv) : VStmt :=
  { Q with law := Q.law.withExec hT }

/-- `withExec` keeps a statement's sessions. -/
theorem VStmt.withExec_outer (Q : VStmt) (hT : Law.Tiles Q.law.St Q.nv) (Reg' : Type) :
    (Q.withExec hT).Outer Reg' = Q.Outer Reg' :=
  rfl

/-- **Gap 7 at the read-off statements**: from one session of each of V*'s statements whose draw file is tied to its
strata over `[0, nv)` (`DrawFileZK`, the only premise), the strata tile, the statements with their models read off the
strata (`VStmt.withExec`) have the same sessions, and `_hS` holds at them (`vstar_execStrata`). -/
theorem vstar_hS_withExec {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} (z : (j : Fin S) → (Vs j).Outer Reg')
    (hdj : ∀ j, ZkLink.DrawFileZK (z j).dj (.stratified (Vs j).law.Kd (Array.ofFn (Vs j).law.St)) (Vs j).nv) :
    ∃ hT : ∀ j, Law.Tiles (Vs j).law.St (Vs j).nv,
      (∀ j, ((Vs j).withExec (hT j)).Outer Reg' = (Vs j).Outer Reg') ∧
      ∀ j, Law.ExecStrata ((Vs j).withExec (hT j)).law.σs ((Vs j).withExec (hT j)).law.ks
        ((Vs j).withExec (hT j)).law.St := by
  refine ⟨fun j => VStmt.tiles_of_session (z j) (hdj j), fun j => rfl, fun j => ?_⟩
  exact vstar_execStrata (Vs j).law.St (VStmt.tiles_of_session (z j) (hdj j)) (Vs j).law.Kd (Vs j).law.pop
    (Vs j).law.Ω (Vs j).law.Lb (Vs j).law.src

end FlockSoundness.Discharge.Recursion
