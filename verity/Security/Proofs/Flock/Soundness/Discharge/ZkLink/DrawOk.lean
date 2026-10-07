import Proofs.Flock.Soundness.Discharge.ZkLink.TablesAt
import Proofs.Flock.Soundness.Discharge.Exec.LawDraw

/-!
# The challenger's draw at every draw of the live law, at `--zk`

`zk_session_sound` takes `DrawOkZK I dj mPts` at each of `Law.execOS`'s draws: the draw file `dj S` proves `S`'s units,
and setupH, under the `--zk` tags, sets up at `mPts` link points. The first half is what the record carries, so it is
custody; the second is flock-verify's own arithmetic. This file splits them.

* `drawOf law pop S`: the verifier's own draw of `S`, as `UnitDraw.ofJson` decodes it: its law and population, `S`'s
  units ascending, no closure.
* `DrawFileZK dj law pop`: every draw file `dj S` that `UnitDraw.ofJson` decodes decodes to `drawOf law pop S`. It is a
  property of the file the challenger records, not of flock-verify (DrawWords' `hdj`, with the law).
* `drawOkZK_of_exec`: from `DrawFileZK` at the live law's own stratified law, `DrawOkZK` at every draw of `Law.execOS`,
  at `mPtsOf c.g (∑ s, ks s)`. At a run that succeeds the draw has `∑ s, ks s` units (`execOS_draw_card`); at a run that
  reads past the source's bytes the draw is every unit (`execOS_draw_card_or`), and a draw file of every unit that
  setupH accepts passes `UnitDraw.ofJson`'s per-stratum count, so every stratum draws all its units and
  `∑ s, ks s = n`.
-/

namespace FlockSoundness.Discharge.ZkLink

open Flock Exec ZkExec Finset FlockSoundness.Refine

theorem forIn_throw_ok {α : Type} (P : α → Bool) (msg : α → String) :
    ∀ (l : List α) (r : PUnit.{1}),
      forIn (m := V) l PUnit.unit
          (fun s _ => if P s = true then throw (msg s) else pure (ForInStep.yield PUnit.unit)) = .ok r →
        ∀ s ∈ l, P s = false
  | [], _, _, s, hs => by simp at hs
  | a :: l, r, h, s, hs => by
      rw [List.forIn_cons] at h
      by_cases ha : P a = true
      · rw [ite_eq_left ha, throw_bind'] at h
        exact absurd h (throw_ne_ok _ _)
      · rw [ite_eq_right ha] at h
        rcases List.mem_cons.1 hs with rfl | hs
        · simpa using ha
        · exact forIn_throw_ok P msg l r h s hs

theorem unit_cons (t : String) (k : ℕ) (a : ℕ × ℕ) (l : List (ℕ × ℕ)) (j : ℕ) :
    (Draw.Stratum.mk t (a :: l).toArray k).unit j =
      if j < a.2 - a.1 then a.1 + j else (Draw.Stratum.mk t l.toArray k).unit (j - (a.2 - a.1)) := by
  obtain ⟨lo, hi⟩ := a
  simp only [Draw.Stratum.unit, ← Array.forIn_toList, List.forIn_cons]
  split <;> rfl

theorem foldl_size_add (l : List (ℕ × ℕ)) (a : ℕ) :
    l.foldl (fun acc (p : ℕ × ℕ) => acc + (p.2 - p.1)) a = a + l.foldl (fun acc (p : ℕ × ℕ) => acc + (p.2 - p.1)) 0 := by
  induction l generalizing a with
  | nil => simp
  | cons p l ih => rw [List.foldl_cons, List.foldl_cons, ih, ih (0 + _)]; omega

theorem size_cons (t : String) (k : ℕ) (a : ℕ × ℕ) (l : List (ℕ × ℕ)) :
    (Draw.Stratum.mk t (a :: l).toArray k).size = (a.2 - a.1) + (Draw.Stratum.mk t l.toArray k).size := by
  obtain ⟨lo, hi⟩ := a
  simp only [Draw.Stratum.size, List.foldl_toArray', List.foldl_cons]
  rw [foldl_size_add]
  simp

theorem unit_mem_list (t : String) (k : ℕ) : ∀ (l : List (ℕ × ℕ)) (j : ℕ),
    j < (Draw.Stratum.mk t l.toArray k).size →
      ∃ p ∈ l, p.1 ≤ (Draw.Stratum.mk t l.toArray k).unit j ∧ (Draw.Stratum.mk t l.toArray k).unit j < p.2
  | [], j, hj => by simp [Draw.Stratum.size] at hj
  | a :: l, j, hj => by
      rw [size_cons] at hj
      rw [unit_cons]
      split
      · exact ⟨a, List.mem_cons_self .., by omega, by omega⟩
      · obtain ⟨p, hp, h1, h2⟩ := unit_mem_list t k l (j - (a.2 - a.1)) (by omega)
        exact ⟨p, List.mem_cons_of_mem _ hp, h1, h2⟩

/-- A stratum's `j`-th unit, `j` below its size, lies in one of its ranges, as `UnitDraw.ofJson`'s count tests it. -/
theorem unit_any (s : Draw.Stratum) {j : ℕ} (hj : j < s.size) :
    (s.units.any fun x => decide (x.1 ≤ s.unit j) && decide (s.unit j < x.2)) = true := by
  obtain ⟨t, ⟨l⟩, k⟩ := s
  obtain ⟨p, hp, h1, h2⟩ := unit_mem_list t k l j hj
  rw [Array.any_eq_true']
  exact ⟨p, by simpa using hp, by simp [h1, h2]⟩

theorem foldl_count_filter (p : ℕ → Bool) : ∀ (l : List ℕ) (a : ℕ),
    l.foldl (fun acc u => if p u = true then acc + 1 else acc) a = a + (l.filter p).length
  | [], a => by simp
  | u :: l, a => by
      rw [List.foldl_cons, foldl_count_filter p l, List.filter_cons]
      split <;> (simp_all; try omega)


set_option maxHeartbeats 2000000 in
/-- **`UnitDraw.ofJson`'s per-stratum count**: a draw it decodes under a stratified law holds exactly `k_s` of its units in
each stratum `s`'s ranges. -/
theorem ofJson_count {j : Lean.Json} {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson j = .ok D) {K : ℕ}
    {strata : Array Draw.Stratum} (hl : D.law = .stratified K strata) (s : Draw.Stratum) (hs : s ∈ strata) :
    D.units.foldl (fun acc u => if (s.units.any fun x => decide (x.1 ≤ u) && decide (u < x.2)) = true then acc + 1
      else acc) 0 = s.k := by
  unfold Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
  all_goals
    have hp := ‹(pure _ : V Draw.Law) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at hp
  all_goals
    simp only [reduceCtorEq, Draw.Law.stratified.injEq] at hl
  all_goals
    obtain ⟨rfl, rfl⟩ := hl
    have hf := ‹forIn _ _ _ = Except.ok _›
    rw [← Array.forIn_toList] at hf
    simpa using forIn_throw_ok _ _ _ _ hf s (by simpa using hs)


variable {n : ℕ}

/-- **The verifier's own draw of `S`** under `law` over `[0, pop)`, as `UnitDraw.ofJson` decodes it: `S`'s units,
ascending, and no closure. -/
def drawOf (law : Draw.Law) (pop : ℕ) (S : Finset (Fin n)) : Draw.UnitDraw :=
  { law, population := pop, units := ((S.image Fin.val).sort (· ≤ ·)).toArray }

/-- **The record's draw file is the verifier's draw** (custody of the record's `unit_draw`): at every draw `S`, a draw
file `dj S` that `UnitDraw.ofJson` decodes decodes to the verifier's own draw of `S` under `law` over `[0, pop)`. -/
def DrawFileZK (dj : Finset (Fin n) → Lean.Json) (law : Draw.Law) (pop : ℕ) : Prop :=
  ∀ S D, Draw.UnitDraw.ofJson (dj S) = .ok D → D = drawOf law pop S

theorem drawOf_proved (law : Draw.Law) (pop : ℕ) (S : Finset (Fin n)) :
    (drawOf law pop S).proved = (drawOf law pop S).units := by
  simp [Draw.UnitDraw.proved, drawOf]

theorem drawOf_mem (law : Draw.Law) (pop : ℕ) (S : Finset (Fin n)) (x : ℕ) :
    x ∈ (drawOf law pop S).units ↔ ∃ u ∈ S, u.val = x := by
  simp [drawOf]

theorem drawOf_size (law : Draw.Law) (pop : ℕ) (S : Finset (Fin n)) : (drawOf law pop S).units.size = S.card := by
  simp [drawOf, card_image_of_injective _ Fin.val_injective]

open Audit.Law in
/-- At a draw file of every unit that `UnitDraw.ofJson` decodes under the strata `St`, every stratum draws all its units. -/
theorem sum_ks_of_univ {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : ExecStrata σs ks St) (Kd pop : ℕ) {j : Lean.Json}
    (hj : Draw.UnitDraw.ofJson j = .ok (drawOf (.stratified Kd (Array.ofFn St)) pop (univ : Finset (Fin n)))) :
    ∑ s, ks s = n := by
  have hs : ∀ s, ks s = (stratum σs s).card := fun s => by
    refine le_antisymm (hks s) ?_
    have hc := ofJson_count hj rfl (St s) (Array.mem_ofFn.2 ⟨s, rfl⟩)
    simp only [drawOf, List.foldl_toArray'] at hc
    rw [foldl_count_filter, zero_add, (hS s).1] at hc
    rw [← hc, ← card_map Fin.valEmbedding, ← (hS s).2.2]
    refine le_trans (card_le_card fun x hx => ?_) (List.toFinset_card_le _)
    obtain ⟨i, hi, rfl⟩ := mem_image.1 hx
    have hi' : i < (St s).size := mem_range.1 hi
    have hv : (St s).unit i ∈ (stratum σs s).map Fin.valEmbedding := by
      rw [← (hS s).2.2]; exact mem_image_of_mem _ hi
    obtain ⟨v, -, hve⟩ := mem_map.1 hv
    rw [List.mem_toFinset, List.mem_filter]
    exact ⟨by simpa [Finset.mem_sort] using ⟨v, hve⟩, unit_any (St s) hi'⟩
  rw [Finset.sum_congr rfl fun s _ => hs s]
  simp only [stratum]
  rw [← card_eq_sum_card_fiberwise (fun _ _ => mem_univ _), card_univ, Fintype.card_fin]


/-- **The challenger's draw is `DrawOkZK` at every draw of the live law.** On untyped `--statement` tags whose circuit
file parses under the `--zk` tags to `c`, with the executable strata `St` describing `σs`'s, each drawing `ks s` of its
units: when the record's draw file is the verifier's draw under the live law's own stratified law (`DrawFileZK`),
`DrawOkZK` holds at `mPtsOf c.g (∑ s, ks s)` link points at every draw of `Law.execOS`, the run that reads past the
source's bytes (every unit) included. -/
theorem drawOkZK_of_exec (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Audit.Law.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : Audit.Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    DrawOkZK I dj (mPtsOf c.g (∑ s, ks s))
      ((Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) := by
  have hc' : (if (zkTags I).typed then HmRow.parseTyped (zkTags I) I.circuitFile I.tables
      else HmRow.parse (zkTags I) I.circuitFile I.tables) = .ok c := by
    rw [zkTags_typed, ht]; exact hc
  refine ⟨fun D hD u hu => ?_, fun st hst => ?_⟩
  · rw [hdj _ D hD, drawOf_proved, drawOf_mem]
    exact ⟨u, hu, rfl⟩
  · rw [setupH_mPts_of_proved hc' (fun D hD => by rw [hdj _ D hD, drawOf_proved, drawOf_size]) hst]
    rcases execOS_draw_card_or σs ks St hS Kd pop src ω with hcard | huniv
    · rw [hcard]
    · obtain ⟨ud, hud, -⟩ := setupH_mPts_drawn hst
      rw [hdj _ ud hud, huniv] at hud
      rw [huniv, card_univ, Fintype.card_fin, sum_ks_of_univ σs ks hks St hS Kd pop hud]

end FlockSoundness.Discharge.ZkLink
