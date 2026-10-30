import Proofs.Properties.PoubCheck.Shared

/-!
# The session link windows a draw (`SessionDrawsRecords`, `AcceptedDrawsOpenable`)

A run keeps, at any `g ≥ 1`, three facts of every drawn draw and accepted window:

* the draw is `draws` over its windows' plans, on the session's digest and coin `W + c`, and each window of it holds
  its share (`passesBefore`). At `g ≥ 2` that is `DrawLink` (`LinkInv`, `linkInv_step`): the close that draws it gives
  it (`drawIsSubset`), and every later event keeps it, since a drawn window keeps its draw and plans (`Frz`). At
  `g = 1` the draw is window `c`'s own (`DL1`, `Link1`, `link1_step`), the whole of it its share (`shareOf_all`).
* `DrawsIn`: each drawn window's draws are receipts of its plans (`close_drawsIn`, `drawsIn_of_link`), so its check
  reads only blocks below the store's extent at its draw (`draw_in_record`, `record_reads`).
* `AcceptNow`: an accepted window is drawn, and its share is empty or `checkWindow` of it passes on the session's
  appends now. Only `verify` and the close accept a window (`VK`, each event's `vk_*`): `verify` on `checkWindow`'s
  verdict (`acc_verify`), the close at `g ≥ 2` on an empty share (`acc_sclose`). A drawn window's record is kept
  (`record_frz`), and appends past its draw start above every block it reads (`firsts_big`), so its verdict holds on
  the appends that follow (`checkWindow_grow`).

Under `Inv`'s `sizes` a window's record has its plans as passes, so the draw's lengths and shares are the records'
(`lengthsOf_draw`, `passCount_draw`): `sessionDrawsRecords`. With `acceptDrawsOpenable` on the openings each window
passed with, `acceptedDrawsOpenable`.
-/

open PoubCheck PoubCheck.Session

namespace Properties.PoubCheck.Shared

/-! ## Accepts -/

/-- No window of `ws'` is accepted that was not in `ws`. -/
def VK (ws ws' : Array Win) : Prop := ∀ u : Nat, ws'[u]!.verdict = some .accept → ws[u]!.verdict = some .accept

theorem vk_refl (ws : Array Win) : VK ws ws := fun _ h => h

theorem vk_trans {a b c : Array Win} (h₁ : VK a b) (h₂ : VK b c) : VK a c := fun u h => h₁ u (h₂ u h)

theorem vk_modify (ws : Array Win) (u : Nat) (f : Win → Win)
    (hf : ∀ w, (f w).verdict = some .accept → w.verdict = some .accept) : VK ws (ws.modify u f) := by
  intro t h
  rw [getElem!_modify'] at h
  split at h
  · exact hf _ h
  · exact h

theorem vk_push (ws : Array Win) (w : Win) (hw : w.verdict = none) : VK ws (ws.push w) := by
  intro t h
  by_cases ht : t < ws.size
  · rw [getElem!_pos (ws.push w) t (by simp; omega), Array.getElem_push_lt] at h
    rw [getElem!_pos ws t ht]
    exact h
  · exfalso
    by_cases he : t = ws.size
    · subst he
      rw [getElem!_pos (ws.push w) ws.size (by simp), Array.getElem_push_eq, hw] at h
      cases h
    · rw [getElem!_neg (ws.push w) t (by simp; omega)] at h
      cases h

theorem vk_fail (s : State) (t : Nat) (f : Failure) (a : Option (Nat × Nat)) :
    VK s.windows (s.fail t f a).2.windows := by
  unfold State.fail
  dsimp only
  split
  · exact vk_modify _ _ _ (fun _ h => by first | cases h | simp at h)
  · exact vk_refl _

theorem vk_start {s : State} {now : Nat} {r : Result} {s' : State} (he : start s now = (r, s')) :
    VK s.windows s'.windows := by
  unfold start State.releaseCoin at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact vk_refl _
    | exact vk_fail _ _ _ _
    | (refine vk_push _ _ ?_; rfl)
    | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
    | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl)
    | (refine vk_trans ?_ (vk_fail _ _ _ _)
       first
         | exact vk_refl _
         | (refine vk_push _ _ ?_; rfl)
         | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
         | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl))
    | (split <;> first
        | exact vk_refl _
        | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
        | (refine vk_modify _ _ _ ?_; exact fun _ h => by first | cases h | simp at h))

theorem vk_boundary {s : State} {now : Nat} {r : Result} {s' : State} (he : boundary s now = (r, s')) :
    VK s.windows s'.windows := by
  unfold boundary State.releaseCoin at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact vk_refl _
    | exact vk_fail _ _ _ _
    | (refine vk_push _ _ ?_; rfl)
    | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
    | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl)
    | (refine vk_trans ?_ (vk_fail _ _ _ _)
       first
         | exact vk_refl _
         | (refine vk_push _ _ ?_; rfl)
         | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
         | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl))
    | (split <;> first
        | exact vk_refl _
        | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
        | (refine vk_modify _ _ _ ?_; exact fun _ h => by first | cases h | simp at h))

theorem vk_append {s : State} {root : ByteArray} {count : Int} {r : Result} {s' : State}
    (he : append s root count = .ok (r, s')) : VK s.windows s'.windows := by
  unfold append at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact vk_refl _
    | exact vk_fail _ _ _ _
    | (refine vk_push _ _ ?_; rfl)
    | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
    | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl)
    | (refine vk_trans ?_ (vk_fail _ _ _ _)
       first
         | exact vk_refl _
         | (refine vk_push _ _ ?_; rfl)
         | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
         | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl))
    | (split <;> first
        | exact vk_refl _
        | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
        | (refine vk_modify _ _ _ ?_; exact fun _ h => by first | cases h | simp at h))

theorem vk_release {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State}
    (he : release s runs withdrawn = (r, s')) : VK s.windows s'.windows := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact vk_refl _
    | exact vk_fail _ _ _ _
    | (refine vk_push _ _ ?_; rfl)
    | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
    | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl)
    | (refine vk_trans ?_ (vk_fail _ _ _ _)
       first
         | exact vk_refl _
         | (refine vk_push _ _ ?_; rfl)
         | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
         | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl))
    | (split <;> first
        | exact vk_refl _
        | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
        | (refine vk_modify _ _ _ ?_; exact fun _ h => by first | cases h | simp at h))

theorem vk_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {st : Option Int} {now : Nat} {r : Result}
    {s' : State} (he : receive s t i root runs change st now = (r, s')) : VK s.windows s'.windows := by
  unfold receive at he
  split at he
  · cases he; exact vk_refl _
  split at he
  · cases he; exact vk_refl _
  split at he
  · cases he; exact vk_fail _ _ _ _
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, VK s.windows s₂.windows → jp () s₂ = (r, s') → VK s.windows s'.windows := by
    intro s₂ k₂ hj
    dsimp only [jp] at hj
    repeat' split at hj
    all_goals (try dsimp only at hj)
    all_goals cases hj
    all_goals refine vk_trans k₂ ?_
    all_goals first
      | exact vk_refl _
      | exact vk_fail _ _ _ _
      | (refine vk_push _ _ ?_; rfl)
      | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
      | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl)
      | (refine vk_trans ?_ (vk_fail _ _ _ _)
         first
           | exact vk_refl _
           | (refine vk_push _ _ ?_; rfl)
           | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
           | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl))
      | (split <;> first
          | exact vk_refl _
          | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
          | (refine vk_modify _ _ _ ?_; exact fun _ h => by first | cases h | simp at h))
  split at he
  · exact key s₁ (by refine vk_modify _ _ _ ?_; exact fun _ h => h) he
  · split at he
    · cases he; exact vk_fail _ _ _ _
    · exact key s₀ (vk_refl _) he

theorem vk_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State}
    (he : close s t now = .ok (r, s')) : VK s.windows s'.windows := by
  unfold close State.fail at he
  dsimp only at he
  repeat' (split at he <;> try dsimp only at he)
  all_goals (try contradiction)
  all_goals cases he
  all_goals first
    | exact vk_refl _
    | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
    | (refine vk_trans (vk_modify _ _ _ ?_) (vk_modify _ _ _ ?_) <;>
        exact fun _ h => by first | exact h | cases h | simp at h)

/-- `verify` accepts only the window it checks, on `checkWindow`'s verdict on its draws. -/
theorem acc_verify {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State}
    (he : verify s t openings = (r, s')) :
    VK s.windows s'.windows ∨ ∃ d, s.windows[t.toNat]!.drawn = some d ∧
      checkWindow s.reg s.digest s.appends (s.record t.toNat) d openings = .accept ∧
      s' = { s with windows := s.windows.modify t.toNat ({· with verdict := some .accept}) } := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact Or.inl (vk_refl _)
    | exact Or.inl (vk_fail _ _ _ _)
    | exact Or.inr ⟨_, by assumption, by assumption, rfl⟩
    | (left; first
          | exact vk_refl _
          | exact vk_fail _ _ _ _
          | (refine vk_push _ _ ?_; rfl)
          | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
          | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl)
          | (refine vk_trans ?_ (vk_fail _ _ _ _)
             first
               | exact vk_refl _
               | (refine vk_push _ _ ?_; rfl)
               | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
               | (refine vk_trans (vk_modify _ _ _ ?_) (vk_push _ _ ?_) <;> first | exact fun _ h => h | rfl))
          | (split <;> first
              | exact vk_refl _
              | (refine vk_modify _ _ _ ?_; exact fun _ h => h)
              | (refine vk_modify _ _ _ ?_; exact fun _ h => by first | cases h | simp at h)))

theorem finalize_verdict (first m : Nat) (ws : Array Win) (v : Nat) :
    (finalize ((List.range m).map (first + ·)) ws)[v]!.verdict = ws[v]!.verdict := by
  rw [finalize_range]; split <;> rfl

/-- `Shared.close` accepts only a window of the draw whose share is empty. -/
theorem acc_sclose {g : Nat} {s : State} {t : Int} {now : Nat} {r : Result} {s' : State}
    (he : Shared.close g s t now = .ok (r, s')) (u : Nat) (h : s'.windows[u]!.verdict = some .accept) :
    s.windows[u]!.verdict = some .accept ∨ s'.windows[u]!.drawn = some #[] := by
  rcases sclose_spec he with ⟨-, rfl⟩ | ⟨v, f, -, rfl⟩ | ⟨d, -, -, -, -, -, -, -, -, rfl⟩
  · exact Or.inl h
  · left
    have h' := vk_fail { s with windows := finalize (grp g t.toNat) s.windows } v f none u h
    change (finalize (grp g t.toNat) s.windows)[u]!.verdict = some .accept at h'
    unfold grp at h'
    rwa [finalize_verdict] at h'
  · obtain ⟨-, -, h3⟩ := drawFold_range d now (t.toNat / g * g) 0 (finalize (grp g t.toNat) s.windows)
      (t.toNat + 1 - t.toNat / g * g)
    change (((grp g t.toNat).foldl (drawStep d now) (finalize (grp g t.toNat) s.windows, 0)).1)[u]!.verdict =
      some .accept at h
    show s.windows[u]!.verdict = some .accept ∨
      (((grp g t.toNat).foldl (drawStep d now) (finalize (grp g t.toNat) s.windows, 0)).1)[u]!.drawn = some #[]
    unfold grp at h3 h ⊢
    rw [h3 u] at h ⊢
    split at h
    · rename_i hc
      rw [ite_eq_left hc]
      simp only [drawW] at h ⊢
      split at h
      · rename_i hE
        right
        first
          | exact congrArg some (Array.isEmpty_iff.1 hE)
          | exact Array.isEmpty_iff.1 hE
          | simpa using hE
      · left
        rwa [finalize_verdict] at h
    · left
      rwa [finalize_verdict] at h

/-- At `g ≥ 2` a window newly accepted is drawn with an empty share, or `verify`'s on `checkWindow` of its draws. -/
theorem acc_sstep {g : Nat} (hg : 2 ≤ g) {s : State} {e : Event} {r : Result} {s' : State}
    (he : Shared.step g s e = .ok (r, s')) (u : Nat) (h : s'.windows[u]!.verdict = some .accept) :
    s.windows[u]!.verdict = some .accept ∨ s'.windows[u]!.drawn = some #[] ∨
      ∃ d openings, s.windows[u]!.drawn = some d ∧
        checkWindow s.reg s.digest s.appends (s.record u) d openings = .accept ∧
        s' = { s with windows := s.windows.modify u ({· with verdict := some .accept}) } := by
  rcases sstep_of hg he with ⟨t, now, rfl, hc⟩ | ⟨hn, hs⟩
  · rcases acc_sclose hc u h with h' | h'
    · exact Or.inl h'
    · exact Or.inr (Or.inl h')
  · cases e with
    | start now => injection hs with hs; exact Or.inl (vk_start hs u h)
    | boundary now => injection hs with hs; exact Or.inl (vk_boundary hs u h)
    | append root count => exact Or.inl (vk_append hs u h)
    | release runs withdrawn => injection hs with hs; exact Or.inl (vk_release hs u h)
    | receive t i root runs change st now => injection hs with hs; exact Or.inl (vk_receive hs u h)
    | close t now => exact (hn t now rfl).elim
    | verify t openings =>
      injection hs with hs
      rcases acc_verify hs with hv | ⟨d, hd, hc, rfl⟩
      · exact Or.inl (hv u h)
      · by_cases hu : t.toNat = u
        · subst hu
          exact Or.inr (Or.inr ⟨d, openings, hd, hc, rfl⟩)
        · left
          have h' : (s.windows.modify t.toNat ({· with verdict := some .accept}))[u]!.verdict = some .accept := h
          rw [getElem!_modify'] at h'
          split at h'
          · rename_i hc'; exact absurd hc'.1 hu
          · exact h'

/-- At `g = 1` a window newly accepted is `verify`'s, on `checkWindow` of its draws. -/
theorem acc_step1 {s : State} {e : Event} {r : Result} {s' : State} (he : Session.step s e = .ok (r, s')) (u : Nat)
    (h : s'.windows[u]!.verdict = some .accept) :
    s.windows[u]!.verdict = some .accept ∨
      ∃ d openings, s.windows[u]!.drawn = some d ∧
        checkWindow s.reg s.digest s.appends (s.record u) d openings = .accept ∧
        s' = { s with windows := s.windows.modify u ({· with verdict := some .accept}) } := by
  cases e with
  | start now => injection he with he; exact Or.inl (vk_start he u h)
  | boundary now => injection he with he; exact Or.inl (vk_boundary he u h)
  | append root count => exact Or.inl (vk_append he u h)
  | release runs withdrawn => injection he with he; exact Or.inl (vk_release he u h)
  | receive t i root runs change st now => injection he with he; exact Or.inl (vk_receive he u h)
  | close t now => exact Or.inl (vk_close he u h)
  | verify t openings =>
    injection he with he
    rcases acc_verify he with hv | ⟨d, hd, hc, rfl⟩
    · exact Or.inl (hv u h)
    · by_cases hu : t.toNat = u
      · subst hu
        exact Or.inr ⟨d, openings, hd, hc, rfl⟩
      · left
        have h' : (s.windows.modify t.toNat ({· with verdict := some .accept}))[u]!.verdict = some .accept := h
        rw [getElem!_modify'] at h'
        split at h'
        · rename_i hc'; exact absurd hc'.1 hu
        · exact h'


/-! ## What a drawn window keeps -/

theorem plans_frz {s s' : State} (hf : Frz s s') {u : Nat} (hu : (dr s.windows u).isSome) :
    s'.windows[u]!.plans = s.windows[u]!.plans ∧ dr s'.windows u = dr s.windows u := by
  have hd : s.windows[u]!.drawn.isSome := by rw [← dr_eq]; exact hu
  obtain ⟨e1, -, -, e4, -⟩ := (hf.1 u).1 hd
  exact ⟨e4, by rw [dr_eq, dr_eq, e1]⟩

theorem record_frz {s s' : State} (hp : Prefix s.windows) (hf : Frz s s') {u : Nat} (hu : (dr s.windows u).isSome) :
    s'.record u = s.record u := by
  obtain ⟨hk, hdig, -⟩ := hf
  have hd : s.windows[u]!.drawn.isSome := by rw [← dr_eq]; exact hu
  obtain ⟨-, e2, e3, e4, e5⟩ := (hk u).1 hd
  have hprev : (if u > 0 then s'.windows[u - 1]!.roots.back?.getD ByteArray.empty else s'.digest) =
      (if u > 0 then s.windows[u - 1]!.roots.back?.getD ByteArray.empty else s.digest) := by
    by_cases htp : u > 0
    · have hs : (dr s.windows (u - 1)).isSome := hp u (u - 1) (by omega) hu
      rw [dr_eq] at hs
      rw [ite_eq_left htp, ite_eq_left htp, ((hk (u - 1)).1 hs).2.2.1]
    · rw [ite_eq_right htp, ite_eq_right htp, hdig]
  unfold State.record
  simp only [e2, e3, e4, e5]
  rw [hprev]

theorem drawLengths_congr {g : Nat} (s s' : State) (hp : s'.params = s.params) (c : Nat)
    (h : ∀ u ∈ drawWindows g s.params.windows c, s'.windows[u]!.plans = s.windows[u]!.plans) :
    drawLengths g s' c = drawLengths g s c := by
  have key : ∀ (l : List Nat) (acc : Array Nat), (∀ u ∈ l, s'.windows[u]!.plans = s.windows[u]!.plans) →
      l.foldl (fun acc u => acc ++ s'.windows[u]!.plans.map (·.length)) acc =
        l.foldl (fun acc u => acc ++ s.windows[u]!.plans.map (·.length)) acc := by
    intro l
    induction l with
    | nil => intro acc _; rfl
    | cons a l ih =>
      intro acc hl
      simp only [List.foldl_cons]
      rw [hl a (by simp)]
      exact ih _ (fun u hu => hl u (by simp [hu]))
  unfold drawLengths
  rw [hp]
  exact key _ _ h

theorem passesBefore_congr (s s' : State) (first u : Nat)
    (h : ∀ j, j < u - first → s'.windows[first + j]!.plans = s.windows[first + j]!.plans) :
    passesBefore s' first u = passesBefore s first u := by
  unfold passesBefore
  congr 1
  apply List.map_congr_left
  intro j hj
  exact congrArg Array.size (h j (List.mem_range.1 hj))

theorem fold_plans {g : Nat} (s : State) (t : Int) (d : Array (Nat × Nat)) (now v : Nat) :
    (((grp g t.toNat).foldl (drawStep d now) (finalize (grp g t.toNat) s.windows, 0)).1)[v]!.plans =
      s.windows[v]!.plans := by
  obtain ⟨-, -, h3⟩ := drawFold_range d now (t.toNat / g * g) 0 (finalize (grp g t.toNat) s.windows)
    (t.toNat + 1 - t.toNat / g * g)
  unfold grp at h3 ⊢
  rw [h3 v]
  split
  · first | exact finalize_plans _ _ _ _ | (simp only [drawW]; exact finalize_plans _ _ _ _)
  · exact finalize_plans _ _ _ _

/-! ## Each draw is a receipt of its window -/

/-- Each drawn window's draws are receipts of its passes' plans. -/
def DrawsIn (ws : Array Win) : Prop :=
  ∀ (u : Nat) (d : Array (Nat × Nat)), ws[u]!.drawn = some d → ∀ k (hk : k < d.size),
    d[k].1 < ws[u]!.plans.size ∧ d[k].2 < ws[u]!.plans[d[k].1]!.length

theorem drawsIn_modify {ws : Array Win} (u : Nat) (f : Win → Win) (hf : ∀ w, (f w).drawn = w.drawn)
    (hp : ∀ w, (f w).plans = w.plans) (h : DrawsIn ws) : DrawsIn (ws.modify u f) := by
  intro v d hd k hk
  rw [Properties.PoubCheck.getElem!_modify'] at hd ⊢
  by_cases hc : u = v ∧ v < ws.size
  · rw [ite_eq_left hc] at hd ⊢
    rw [hf] at hd
    rw [hp]
    exact h v d hd k hk
  · rw [ite_eq_right hc] at hd ⊢
    exact h v d hd k hk

theorem drawsIn_draw {ws : Array Win} (u : Nat) (d : Array (Nat × Nat)) (f : Win → Win)
    (hf : ∀ w, (f w).drawn = some d) (hp : ∀ w, (f w).plans = w.plans) (h : DrawsIn ws)
    (hd : ∀ k (hk : k < d.size), d[k].1 < ws[u]!.plans.size ∧ d[k].2 < ws[u]!.plans[d[k].1]!.length) :
    DrawsIn (ws.modify u f) := by
  intro v d' hd' k hk
  rw [Properties.PoubCheck.getElem!_modify'] at hd' ⊢
  by_cases hc : u = v ∧ v < ws.size
  · rw [ite_eq_left hc] at hd' ⊢
    rw [hf] at hd'
    cases hd'
    rw [hp, ← hc.1]
    exact hd k hk
  · rw [ite_eq_right hc] at hd' ⊢
    exact h v d' hd' k hk

theorem plans_modify (ws : Array Win) (u v : Nat) (f : Win → Win) (hp : ∀ w, (f w).plans = w.plans) :
    (ws.modify u f)[v]!.plans = ws[v]!.plans := by
  rw [Properties.PoubCheck.getElem!_modify']
  split
  · exact hp _
  · rfl

theorem close_drawsIn {s : State} {t : Int} {now : Nat} {r : Result} {s₁ : State}
    (he : close s t now = .ok (r, s₁)) (h : DrawsIn s.windows) : DrawsIn s₁.windows := by
  unfold close State.fail at he
  dsimp only at he
  repeat' (split at he <;> try dsimp only at he)
  all_goals (try contradiction)
  all_goals cases he
  all_goals first
    | exact h
    | exact drawsIn_modify _ _ (fun _ => rfl) (fun _ => rfl) (drawsIn_modify _ _ (fun _ => rfl) (fun _ => rfl) h)
    | exact drawsIn_modify _ _ (fun _ => rfl) (fun _ => rfl) h
    | skip
  all_goals
    have hdr := ‹draws _ _ _ _ _ = some _›
    obtain ⟨_, _, -, -, -, hrange, -⟩ := Properties.PoubCheck.drawIsSubset _ _ _ _ _ _ hdr
    refine drawsIn_draw _ _ _ (fun _ => rfl) (fun _ => rfl)
      (drawsIn_modify _ _ (fun _ => rfl) (fun _ => rfl) h) fun k hk => ?_
    obtain ⟨hi, hp, -⟩ := hrange k hk
    have hpl : (s.windows.modify t.toNat ({· with final := true}))[t.toNat]!.plans = s.windows[t.toNat]!.plans :=
      plans_modify _ _ _ _ (fun _ => rfl)
    rw [hpl]
    simp only [Array.size_map] at hi
    refine ⟨hi, ?_⟩
    have e : ∀ i (hi : i < s.windows[t.toNat]!.plans.size),
        (s.windows[t.toNat]!.plans.map fun x : Plan => x.length)[i]! = s.windows[t.toNat]!.plans[i]!.length :=
      fun i hi => by
        rw [getElem!_pos (s.windows[t.toNat]!.plans.map fun x : Plan => x.length) i (by simpa using hi),
          getElem!_pos s.windows[t.toNat]!.plans i hi, Array.getElem_map]
    rw [e _ hi] at hp
    exact hp

theorem drawsIn_step {s : State} {e : Event} {r : Result} {s' : State} (h : FInv s) (hin : DrawsIn s.windows)
    (he : Session.step s e = .ok (r, s')) : DrawsIn s'.windows := by
  have hf := (finv_step h he).2.1
  have keep : Keep s.windows s'.windows → DrawsIn s'.windows := by
    intro hk u d hd k hkd
    have hd' : dr s'.windows u = some d := by rw [dr_eq]; exact hd
    rcases hk.2 u with e | ⟨-, e⟩
    · rw [e, dr_eq] at hd'
      obtain ⟨-, -, -, e4, -⟩ := (hf u).1 (by rw [hd']; rfl)
      rw [e4]
      exact hin u d hd' k hkd
    · rw [e] at hd'; cases hd'
  cases e with
  | start now => injection he with he; exact keep (keep_start he)
  | boundary now => injection he with he; exact keep (keep_boundary he)
  | append root count => exact keep (keep_append he)
  | release runs withdrawn => injection he with he; exact keep (keep_release he)
  | receive t i root runs change st now => injection he with he; exact keep (keep_receive he)
  | close t now => exact close_drawsIn he hin
  | verify t openings => injection he with he; exact keep (keep_verify he)

/-- A drawn window's draws are receipts of its record's passes. -/
theorem draw_in_record {s : State} (hI : Inv s) (hin : DrawsIn s.windows) {t : Nat} (ht : t < s.windows.size)
    {d : Array (Nat × Nat)} (hd : s.windows[t]!.drawn = some d) {k : Nat} (hk : k < d.size) :
    d[k]!.1 < (s.record t).passes.size ∧ d[k]!.2 < (s.record t).passes[d[k]!.1]!.plan.length := by
  have h' : InvOn _ _ _ _ _ _ _ _ _ _ := hI
  obtain ⟨h1, h2⟩ := hin t d hd k hk
  rw [getElem!_pos d k hk]
  have hsz : (s.record t).passes.size = s.windows[t]!.roots.size := by simp [State.record]
  have hpl : s.windows[t]!.plans.size = s.windows[t]!.roots.size := by
    rw [getElem!_pos s.windows t ht]; exact (h'.sizes t ht).1
  have hi : d[k].1 < (s.record t).passes.size := by rw [hsz, ← hpl]; exact h1
  refine ⟨hi, ?_⟩
  rw [(record_pass s t _ hi).1]
  exact h2



/-! ## The records -/

theorem sizes_all {s : State} (h : Inv s) (u : Nat) : s.windows[u]!.plans.size = s.windows[u]!.roots.size := by
  have h' : InvOn _ _ _ _ _ _ _ _ _ _ := h
  by_cases hu : u < s.windows.size
  · rw [getElem!_pos s.windows u hu]; exact (h'.sizes u hu).1
  · rw [getElem!_neg s.windows u hu]; rfl

theorem record_size (s : State) (u : Nat) : (s.record u).passes.size = s.windows[u]!.roots.size := by
  simp [State.record]

theorem lengths_record (s : State) (u : Nat) (h : s.windows[u]!.plans.size = s.windows[u]!.roots.size) :
    (s.record u).passes.map (·.plan.length) = s.windows[u]!.plans.map (·.length) := by
  unfold State.record
  apply Array.ext
  · simp [h]
  · intro i _ h2
    have hi : i < s.windows[u]!.plans.size := by simpa using h2
    simp only [Array.getElem_map, Array.getElem_range]
    rw [getElem!_pos s.windows[u]!.plans i hi]

theorem length_drawRecords {g : Nat} (s : State) (c : Nat) :
    (drawRecords g s c).length = min (c * g + g) s.params.windows - c * g := by
  simp [drawRecords, drawWindows]

theorem drawRecords_get {g : Nat} (s : State) (c k : Nat) (hk : k < (drawRecords g s c).length) :
    (drawRecords g s c)[k] = s.record (c * g + k) := by
  simp [drawRecords, drawWindows]

theorem foldl_append_eq (l : List Nat) (f : Nat → Array Nat) : ∀ acc : Array Nat,
    l.foldl (fun acc u => acc ++ f u) acc = acc ++ (l.flatMap fun u => (f u).toList).toArray := by
  induction l with
  | nil => intro acc; simp
  | cons a l ih =>
    intro acc
    rw [List.foldl_cons, ih]
    apply Array.ext'
    simp

theorem lengthsOf_draw {g : Nat} {s : State}
    (hs : ∀ u : Nat, s.windows[u]!.plans.size = s.windows[u]!.roots.size)
    (c : Nat) : lengthsOf (drawRecords g s c) = drawLengths g s c := by
  have key : ∀ L : List Nat, (L.map s.record).flatMap (fun w => (w.passes.map (·.plan.length)).toList) =
      L.flatMap (fun u => (s.windows[u]!.plans.map (·.length)).toList) := by
    intro L
    induction L with
    | nil => rfl
    | cons a l ih => simp only [List.map_cons, List.flatMap_cons, ih, lengths_record s a (hs a)]
  unfold lengthsOf drawRecords drawLengths
  rw [foldl_append_eq, key]
  all_goals simp

theorem passesBefore_succ (s : State) (first k : Nat) :
    passesBefore s first (first + (k + 1)) = passesBefore s first (first + k) + s.windows[first + k]!.plans.size := by
  simp [passesBefore, List.range_succ, List.sum_append]

theorem passCount_draw {g : Nat} {s : State}
    (hs : ∀ u : Nat, s.windows[u]!.plans.size = s.windows[u]!.roots.size)
    (c : Nat) : ∀ k, k ≤ (drawRecords g s c).length →
      passCount ((drawRecords g s c).take k) = passesBefore s (c * g) (c * g + k) := by
  intro k
  induction k with
  | zero => intro _; simp [passCount, passesBefore]
  | succ k ih =>
    intro hk
    have hk' : k < (drawRecords g s c).length := by omega
    have hk2 : k ≤ (drawRecords g s c).length := by omega
    rw [List.take_add_one, List.getElem?_eq_getElem hk', passesBefore_succ, ← ih hk2]
    simp only [Option.toList, passCount, List.map_append, List.sum_append, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, Nat.add_zero]
    rw [drawRecords_get s c k hk', record_size, hs]

/-! ## A drawn window's verdict on later appends -/

/-- A drawn window's `checkWindow` is the same on any appends that extend the session's: its draws read only blocks
below the store's extent, and the appends past it start at the extent or above. -/
theorem checkWindow_grow {s s' : State} (hI : Inv s) (hI' : Inv s') (hin : DrawsIn s.windows)
    (more : Array Append) (happ : s'.appends = s.appends ++ more) (hr : s'.reg = s.reg) {t : Nat}
    (ht : t < s.windows.size) {d : Array (Nat × Nat)} (hd : s.windows[t]!.drawn = some d) (o : Array Opening) :
    checkWindow s.reg s.digest s'.appends (s.record t) d o = checkWindow s.reg s.digest s.appends (s.record t) d o := by
  rw [happ, checkWindow_eq, checkWindow_eq]
  split
  · rfl
  congr 1
  apply findSome?_congr
  intro k hk
  rw [List.mem_range] at hk
  obtain ⟨hi, hp⟩ := draw_in_record hI hin ht hd hk
  unfold rejectAt
  rw [checkOpening_congr]
  apply appendOf_extend
  · rw [← happ]; exact firsts_mono hI'
  · intro k' hk' hk''
    have := firsts_big hI hI' more happ hr k' hk' (by rw [happ]; exact hk'')
    have hb := record_reads hI t ht _ _ hi hp
    rw [← happ]; omega

/-! ## A share's draws -/

theorem shareOf_spec {drawn : Array (Nat × Nat)} {base n : Nat} {x : Nat × Nat} (hx : x ∈ shareOf drawn base n) :
    ∃ k, ∃ hk : k < drawn.size, base ≤ drawn[k].1 ∧ drawn[k].1 < base + n ∧ x = (drawn[k].1 - base, drawn[k].2) := by
  unfold shareOf at hx
  rw [Array.mem_filterMap] at hx
  obtain ⟨y, hy, hf⟩ := hx
  obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hy
  refine ⟨k, hk, ?_⟩
  generalize drawn[k] = q at hf ⊢
  obtain ⟨a, b⟩ := q
  simp only at hf ⊢
  split at hf
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    injection hf with hf
    exact ⟨hc.1, hc.2, hf.symm⟩
  · cases hf

theorem lens_get (w : WindowRecord) (j : Nat) (hj : j < w.passes.size) :
    (lens w)[j]! = w.passes[j]!.plan.length := by
  have hj' : j < (lens w).length := by rw [length_lens]; exact hj
  rw [getElem!_pos (lens w) j hj', getElem!_pos w.passes j hj]
  simp [lens]

theorem getElem!_mid' (A B C : List Nat) (j : Nat) (hj : j < B.length) :
    (A ++ (B ++ C)).toArray[A.length + j]! = B[j]! := by
  rw [getElem!_mid A B C (A.length + j) (by omega) (by omega), getElem!_pos B j hj]
  first | simp only [Nat.add_sub_cancel_left] | (congr 1; omega) | simp

/-- Pass `j` of window `k` of draw `c` is the draw's pass `passesBefore + j`. -/
theorem drawLengths_mid {g : Nat} {s : State} (hs : ∀ u : Nat, s.windows[u]!.plans.size = s.windows[u]!.roots.size)
    (c k : Nat) (hk : k < (drawRecords g s c).length) (j : Nat) (hj : j < s.windows[c * g + k]!.plans.size) :
    (drawLengths g s c)[passesBefore s (c * g) (c * g + k) + j]! = s.windows[c * g + k]!.plans[j]!.length := by
  have hsplit := flatMap_split (drawRecords g s c) k hk
  have hA : (((drawRecords g s c).take k).flatMap lens).length = passesBefore s (c * g) (c * g + k) := by
    rw [← passCount_eq, passCount_draw hs c k (by omega)]
  have hrec : (drawRecords g s c)[k] = s.record (c * g + k) := drawRecords_get s c k hk
  have hpj : j < (s.record (c * g + k)).passes.size := by rw [record_size, ← hs]; exact hj
  have hl : drawLengths g s c = (((drawRecords g s c).take k).flatMap lens ++
      (lens (drawRecords g s c)[k] ++ ((drawRecords g s c).drop (k + 1)).flatMap lens)).toArray := by
    rw [← lengthsOf_draw hs c]; unfold lengthsOf; exact congrArg List.toArray hsplit
  rw [hl, ← hA, getElem!_mid' _ _ _ j (by rw [length_lens, hrec]; exact hpj), hrec, lens_get _ j hpj,
    (record_pass s (c * g + k) j hpj).1]

/-! ## At `g ≥ 2`: the draw's link -/

/-- Each drawn draw is `draws` over its windows' plans, on the session's digest and coin `W + c`, and each window of
it holds its share. -/
def DrawLink (g : Nat) (s : State) : Prop :=
  ∀ c, (dr s.windows (c * g)).isSome → ∃ drawn,
    draws s.digest c s.coins[s.params.windows + c]! s.params.samples (drawLengths g s c) = some drawn ∧
    ∀ u ∈ drawWindows g s.params.windows c,
      dr s.windows u = some (shareOf drawn (passesBefore s (c * g) u) s.windows[u]!.plans.size)

theorem drawLink_step {g : Nat} (hg : 2 ≤ g) {s : State} {e : Event} {r : Result} {s' : State}
    (hL : DrawLink g s) (hf : Frz s s') (hp : s'.params = s.params) (hc : s'.coins = s.coins)
    (hdig : s'.digest = s.digest) (he : Shared.step g s e = .ok (r, s')) : DrawLink g s' := by
  intro c hc'
  by_cases h0 : (dr s.windows (c * g)).isSome
  · obtain ⟨drawn, hdraws, hsh⟩ := hL c h0
    have hall : ∀ u ∈ drawWindows g s.params.windows c, (dr s.windows u).isSome := fun u hu => by
      rw [hsh u hu]; rfl
    have hpl : ∀ u ∈ drawWindows g s.params.windows c, s'.windows[u]!.plans = s.windows[u]!.plans :=
      fun u hu => (plans_frz hf (hall u hu)).1
    refine ⟨drawn, ?_, fun u hu => ?_⟩
    · rw [drawLengths_congr s s' hp c hpl, hdig, hc, hp]; exact hdraws
    · rw [hp] at hu
      obtain ⟨j0, hj0, rfl⟩ := List.mem_map.1 hu
      rw [List.mem_range] at hj0
      rw [(plans_frz hf (hall _ hu)).2, hsh _ hu, (plans_frz hf (hall _ hu)).1,
        passesBefore_congr s s' (c * g) (c * g + j0)
          (fun j hj => hpl _ (List.mem_map.2 ⟨j, List.mem_range.2 (by omega), rfl⟩))]
  · obtain ⟨t, now, rfl, hcl⟩ : ∃ t now, e = .close t now ∧ Shared.close g s t now = .ok (r, s') := by
      rcases sstep_of hg he with h | ⟨hn, hs⟩
      · exact h
      · exfalso; rw [keep_dr (keep_nonclose hn hs)] at hc'; exact h0 hc'
    rcases sclose_spec hcl with ⟨-, rfl⟩ | ⟨u, f, -, rfl⟩ | ⟨d, hrd, -, -, -, -, -, -, hdraws, hs'⟩
    · exact (h0 hc').elim
    · exfalso
      rw [dr_fail] at hc'
      unfold grp at hc'
      dsimp only at hc'
      rw [dr_finalize] at hc'
      exact h0 hc'
    · have hg0 : 0 < g := by omega
      have hr := (range_of_div hg0 (j := t.toNat)).2
      have hc2 : (dr ((grp g t.toNat).foldl (drawStep d now) (finalize (grp g t.toNat) s.windows, 0)).1
          (c * g)).isSome := by rw [hs'] at hc'; exact hc'
      rw [dr_drawn] at hc2
      split at hc2
      · rename_i hin
        have hcg : c = t.toNat / g := by
          have := div_of_range hin.1 (by omega : c * g < t.toNat / g * g + g)
          rwa [Nat.mul_div_cancel _ hg0] at this
        subst hcg hrd
        obtain ⟨-, -, -, -, -, -, -, hsh⟩ := drawIsSubset _ _ _ _ _ _ hcl
        have hpl : ∀ u : Nat, s'.windows[u]!.plans = s.windows[u]!.plans := fun u => by
          rw [hs']; exact fold_plans s t d now u
        refine ⟨d, ?_, fun u hu => ?_⟩
        · rw [drawLengths_congr s s' hp (t.toNat / g) (fun u _ => hpl u), hdig, hc, hp]
          exact hdraws
        · rw [hp] at hu
          rw [passesBefore_congr s s' _ _ (fun j _ => hpl _), hpl]
          exact hsh u hu
      · exact (h0 hc2).elim


/-- With each draw linked, each drawn window's draws are receipts of its plans: its share of its draw's values, each
a receipt of the draw's passes (`drawIsSubset`), falls in its own passes. -/
theorem drawsIn_of_link {g : Nat} (hg : 1 ≤ g) {s : State} (hI : Inv s) (hpre : Prefix s.windows)
    (hD : DP g s.params.windows s.windows) (hL : DrawLink g s) : DrawsIn s.windows := by
  have hs : ∀ u : Nat, s.windows[u]!.plans.size = s.windows[u]!.roots.size := sizes_all hI
  intro u d hd k hk
  have hdu : (dr s.windows u).isSome := by rw [dr_eq, hd]; rfl
  have hg0 : 0 < g := by omega
  obtain ⟨hlo, hhi⟩ := range_of_div hg0 (j := u)
  have huW : u < s.params.windows := hD.1 u hdu
  have hfirst : (dr s.windows (u / g * g)).isSome := by
    rcases Nat.eq_or_lt_of_le hlo with h | h
    · rw [h]; exact hdu
    · exact hpre u _ h hdu
  obtain ⟨drawn, hdraws, hsh⟩ := hL (u / g) hfirst
  have hmem : u ∈ drawWindows g s.params.windows (u / g) :=
    List.mem_map.2 ⟨u - u / g * g, List.mem_range.2 (by omega), by show u / g * g + (u - u / g * g) = u; omega⟩
  have hshu := hsh u hmem
  rw [dr_eq, hd] at hshu
  injection hshu with hshu
  subst hshu
  obtain ⟨m, hm, h1, h2, hx⟩ := shareOf_spec (Array.getElem_mem hk)
  rw [hx]
  obtain ⟨_, _, -, -, -, hrange, -⟩ := Properties.PoubCheck.drawIsSubset _ _ _ _ _ _ hdraws
  obtain ⟨-, hp, -⟩ := hrange m hm
  refine ⟨by omega, ?_⟩
  have hu : u / g * g + (u - u / g * g) = u := by omega
  have hk' : u - u / g * g < (drawRecords g s (u / g)).length := by rw [length_drawRecords]; omega
  have hj : drawn[m].1 - passesBefore s (u / g * g) u < s.windows[u / g * g + (u - u / g * g)]!.plans.size := by
    rw [hu]; omega
  have e := drawLengths_mid hs (u / g) (u - u / g * g) hk' _ hj
  rw [hu, show passesBefore s (u / g * g) u + (drawn[m].1 - passesBefore s (u / g * g) u) = drawn[m].1 by omega]
    at e
  rw [← e]
  exact hp

/-! ## Accepted windows -/

/-- Each accepted window is drawn, and its share is empty or passes `checkWindow` on the session's appends. -/
def AcceptNow (s : State) : Prop :=
  ∀ u, s.windows[u]!.verdict = some .accept → ∃ d, dr s.windows u = some d ∧
    (d = #[] ∨ ∃ openings : Array Opening, checkWindow s.reg s.digest s.appends (s.record u) d openings = .accept)

/-- A window accepted before an event that keeps its draw and record stays as `AcceptNow` has it. -/
theorem acceptNow_keep {s s' : State} (hI : Inv s) (hI' : Inv s') (hin : DrawsIn s.windows)
    (hpre : Prefix s.windows) (hf : Frz s s') (hr : s'.reg = s.reg) (hA : AcceptNow s) (u : Nat)
    (hu : s.windows[u]!.verdict = some .accept) :
    ∃ d, dr s'.windows u = some d ∧ (d = #[] ∨
      ∃ openings : Array Opening, checkWindow s'.reg s'.digest s'.appends (s'.record u) d openings = .accept) := by
  obtain ⟨d, hdr, hor⟩ := hA u hu
  have hsome : (dr s.windows u).isSome := by rw [hdr]; rfl
  refine ⟨d, by rw [(plans_frz hf hsome).2, hdr], ?_⟩
  rcases hor with h0 | ⟨o, hc⟩
  · exact Or.inl h0
  · obtain ⟨more, hm⟩ := hf.2.2
    have hus : u < s.windows.size := by
      rcases Nat.lt_or_ge u s.windows.size with h | h
      · exact h
      · rw [dr_none h] at hdr; cases hdr
    have hd : s.windows[u]!.drawn = some d := by rw [← dr_eq]; exact hdr
    refine Or.inr ⟨o, ?_⟩
    rw [hr, hf.2.1, record_frz hpre hf hsome, checkWindow_grow hI hI' hin more hm hr hus hd o]
    exact hc

theorem acceptNow_step {g : Nat} (hg : 2 ≤ g) {s : State} {e : Event} {r : Result} {s' : State} (hI : Inv s)
    (hI' : Inv s') (hin : DrawsIn s.windows) (hpre : Prefix s.windows) (hf : Frz s s') (hr : s'.reg = s.reg)
    (hA : AcceptNow s) (he : Shared.step g s e = .ok (r, s')) : AcceptNow s' := by
  intro u hu
  rcases acc_sstep hg he u hu with hold | hempty | ⟨d, o, hd, hc, rfl⟩
  · exact acceptNow_keep hI hI' hin hpre hf hr hA u hold
  · exact ⟨#[], by rw [dr_eq]; exact hempty, Or.inl rfl⟩
  · have hsome : (dr s.windows u).isSome := by rw [dr_eq, hd]; rfl
    refine ⟨d, ?_, Or.inr ⟨o, ?_⟩⟩
    · rw [dr_eq]
      show (s.windows.modify u ({· with verdict := some .accept}))[u]!.drawn = some d
      rw [getElem!_modify']
      split <;> exact hd
    · rw [record_frz hpre hf hsome]
      exact hc

theorem acceptNow_step1 {s : State} {e : Event} {r : Result} {s' : State} (hI : Inv s) (hI' : Inv s')
    (hin : DrawsIn s.windows) (hpre : Prefix s.windows) (hf : Frz s s') (hr : s'.reg = s.reg) (hA : AcceptNow s)
    (he : Session.step s e = .ok (r, s')) : AcceptNow s' := by
  intro u hu
  rcases acc_step1 he u hu with hold | ⟨d, o, hd, hc, rfl⟩
  · exact acceptNow_keep hI hI' hin hpre hf hr hA u hold
  · have hsome : (dr s.windows u).isSome := by rw [dr_eq, hd]; rfl
    refine ⟨d, ?_, Or.inr ⟨o, ?_⟩⟩
    · rw [dr_eq]
      show (s.windows.modify u ({· with verdict := some .accept}))[u]!.drawn = some d
      rw [getElem!_modify']
      split <;> exact hd
    · rw [record_frz hpre hf hsome]
      exact hc

/-! ## The invariants -/

/-- What a run from `State.init` keeps at `g ≥ 2`: `SF`, `Prefix`, what it started with, each draw's link, and its
accepted windows. -/
def LinkInv (g : Nat) (params : Params) (reg : Registration) (coins : Array ByteArray) (s : State) : Prop :=
  SF g s ∧ Prefix s.windows ∧ s.params = params ∧ s.reg = reg ∧ s.coins = coins ∧ s.digest = reg.digest ∧
    DrawLink g s ∧ AcceptNow s

theorem linkInv_step {g : Nat} (hg : 2 ≤ g) {params : Params} {reg : Registration} {coins : Array ByteArray}
    {s : State} {e : Event} {r : Result} {s' : State} (h : LinkInv g params reg coins s)
    (he : Shared.step g s e = .ok (r, s')) : LinkInv g params reg coins s' := by
  obtain ⟨hSF, hpre, hp, hr, hc, hdig, hL, hA⟩ := h
  obtain ⟨hSF', hf⟩ := sf_step hg hSF he
  obtain ⟨-, ⟨hp', hr', -⟩, hc', -, -⟩ := sstep hg hSF.1.1 he
  have hin : DrawsIn s.windows := drawsIn_of_link (by omega) hSF.1.1 hpre hSF.2 hL
  refine ⟨hSF', sstep_prefix hg hpre he, hp'.trans hp, hr'.trans hr, hc'.trans hc, hf.2.1.trans hdig, ?_, ?_⟩
  · exact drawLink_step hg hL hf hp' hc' hf.2.1 he
  · exact acceptNow_step hg hSF.1.1 hSF'.1.1 hin hpre hf hr' hA he

theorem linkInv_init (g : Nat) (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray)
    (notBeforeNs : Nat) (hm : 1 ≤ params.minPasses) :
    LinkInv g params reg coins (State.init params reg coins appendCoins notBeforeNs) := by
  refine ⟨⟨finv_init params reg coins appendCoins notBeforeNs hm, fun u h => by simp [dr, State.init] at h,
    fun u j h => by simp [dr, State.init] at h⟩, fun t j _ ht => by simp [dr, State.init] at ht, rfl, rfl, rfl, rfl,
    fun c h => by simp [dr, State.init] at h, fun u h => ?_⟩
  rw [getElem!_neg (State.init params reg coins appendCoins notBeforeNs).windows u (by simp [State.init])] at h
  cases h

/-- At `g = 1`, each drawn window's draws are `draws` over its own plans, on the session's digest and its coin. -/
def DL1 (digest : ByteArray) (coins : Array ByteArray) (W samples : Nat) (ws : Array Win) : Prop :=
  ∀ u d, dr ws u = some d → draws digest u coins[W + u]! samples (ws[u]!.plans.map (·.length)) = some d

theorem dl1_modify {digest : ByteArray} {coins : Array ByteArray} {W samples : Nat} {ws : Array Win} (u : Nat)
    (f : Win → Win) (hf : ∀ w, (f w).drawn = w.drawn) (hp : ∀ w, (f w).plans = w.plans)
    (h : DL1 digest coins W samples ws) : DL1 digest coins W samples (ws.modify u f) := by
  intro v d hd
  rw [dr_eq, getElem!_modify'] at hd
  rw [plans_modify _ _ _ _ hp]
  split at hd
  · rw [hf] at hd; exact h v d (by rw [dr_eq]; exact hd)
  · exact h v d (by rw [dr_eq]; exact hd)

theorem dl1_draw {digest : ByteArray} {coins : Array ByteArray} {W samples : Nat} {ws : Array Win} (u : Nat)
    (d : Array (Nat × Nat)) (f : Win → Win) (hf : ∀ w, (f w).drawn = some d) (hp : ∀ w, (f w).plans = w.plans)
    (h : DL1 digest coins W samples ws)
    (hd : draws digest u coins[W + u]! samples (ws[u]!.plans.map (·.length)) = some d) :
    DL1 digest coins W samples (ws.modify u f) := by
  intro v d' hd'
  rw [dr_eq, getElem!_modify'] at hd'
  rw [plans_modify _ _ _ _ hp]
  split at hd'
  · rename_i hc
    rw [hf] at hd'
    rw [← Option.some.inj hd', ← hc.1]
    exact hd
  · exact h v d' (by rw [dr_eq]; exact hd')

theorem close_dl1 {s : State} {t : Int} {now : Nat} {r : Result} {s₁ : State}
    (he : close s t now = .ok (r, s₁)) (h : DL1 s.digest s.coins s.params.windows s.params.samples s.windows) :
    DL1 s.digest s.coins s.params.windows s.params.samples s₁.windows := by
  unfold close State.fail at he
  dsimp only at he
  repeat' (split at he <;> try dsimp only at he)
  all_goals (try contradiction)
  all_goals cases he
  all_goals first
    | exact h
    | exact dl1_modify _ _ (fun _ => rfl) (fun _ => rfl) (dl1_modify _ _ (fun _ => rfl) (fun _ => rfl) h)
    | exact dl1_modify _ _ (fun _ => rfl) (fun _ => rfl) h
    | skip
  all_goals
    have hdr := ‹draws _ _ _ _ _ = some _›
    refine dl1_draw _ _ _ (fun _ => rfl) (fun _ => rfl) (dl1_modify _ _ (fun _ => rfl) (fun _ => rfl) h) ?_
    have hpl : (s.windows.modify t.toNat ({· with final := true}))[t.toNat]!.plans = s.windows[t.toNat]!.plans :=
      plans_modify _ _ _ _ (fun _ => rfl)
    rw [hpl]
    exact hdr

theorem dl1_step {s : State} {e : Event} {r : Result} {s' : State} (h : FInv s)
    (hL : DL1 s.digest s.coins s.params.windows s.params.samples s.windows) (he : Session.step s e = .ok (r, s')) :
    DL1 s'.digest s'.coins s'.params.windows s'.params.samples s'.windows := by
  have hf := (finv_step h he).2
  have hp : s'.params = s.params := (step_inv h.1 he).2.1
  rw [hf.2.1, step_coins_eq he, hp]
  have keep : Keep s.windows s'.windows → DL1 s.digest s.coins s.params.windows s.params.samples s'.windows := by
    intro hk u d hd
    rw [keep_dr hk] at hd
    rw [(plans_frz hf (by rw [hd]; rfl)).1]
    exact hL u d hd
  cases e with
  | start now => injection he with he; exact keep (keep_start he)
  | boundary now => injection he with he; exact keep (keep_boundary he)
  | append root count => exact keep (keep_append he)
  | release runs withdrawn => injection he with he; exact keep (keep_release he)
  | receive t i root runs change st now => injection he with he; exact keep (keep_receive he)
  | close t now => exact close_dl1 he hL
  | verify t openings => injection he with he; exact keep (keep_verify he)

/-- What a run from `State.init` keeps at `g = 1`: `FInv`, `Prefix`, what it started with, each window's draw, its
draws in its plans, and its accepted windows. -/
def Link1 (params : Params) (reg : Registration) (coins : Array ByteArray) (s : State) : Prop :=
  FInv s ∧ Prefix s.windows ∧ s.params = params ∧ s.reg = reg ∧ s.coins = coins ∧ s.digest = reg.digest ∧
    DL1 s.digest s.coins s.params.windows s.params.samples s.windows ∧ DrawsIn s.windows ∧ AcceptNow s

theorem link1_step {params : Params} {reg : Registration} {coins : Array ByteArray} {s : State} {e : Event}
    {r : Result} {s' : State} (h : Link1 params reg coins s) (he : Session.step s e = .ok (r, s')) :
    Link1 params reg coins s' := by
  obtain ⟨hF, hpre, hp, hr, hc, hdig, hL, hin, hA⟩ := h
  obtain ⟨hF', hf⟩ := finv_step hF he
  obtain ⟨-, hp', hr', -⟩ := step_inv hF.1 he
  have hc' := step_coins_eq he
  exact ⟨hF', prefix_step hpre (adv_step he), hp'.trans hp, hr'.trans hr, hc'.trans hc, hf.2.1.trans hdig,
    dl1_step hF hL he, drawsIn_step hF hin he, acceptNow_step1 hF.1 hF'.1 hin hpre hf hr' hA he⟩

theorem link1_init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (hm : 1 ≤ params.minPasses) : Link1 params reg coins (State.init params reg coins appendCoins notBeforeNs) := by
  refine ⟨finv_init params reg coins appendCoins notBeforeNs hm, fun t j _ ht => by simp [dr, State.init] at ht,
    rfl, rfl, rfl, rfl, fun u d h => by simp [dr, State.init] at h, fun u d hd => ?_, fun u h => ?_⟩
  · simp [State.init] at hd; cases hd
  · rw [getElem!_neg (State.init params reg coins appendCoins notBeforeNs).windows u (by simp [State.init])] at h
    cases h

/-! ## The link -/

/-- A draw linked at a state the run reached gives `SessionDrawsRecords`' conclusion. -/
theorem link_out {g : Nat} {params : Params} {reg : Registration} {coins : Array ByteArray} {s : State}
    (hI : Inv s) (hp : s.params = params) (hr : s.reg = reg) (hco : s.coins = coins) (hdig : s.digest = reg.digest)
    (hA : AcceptNow s) (c : Nat) (drawn : Array (Nat × Nat))
    (hdraws : draws s.digest c s.coins[s.params.windows + c]! s.params.samples (drawLengths g s c) = some drawn)
    (hsh : ∀ u ∈ drawWindows g s.params.windows c,
      dr s.windows u = some (shareOf drawn (passesBefore s (c * g) u) s.windows[u]!.plans.size)) :
    ∃ drawn, draws reg.digest c coins[params.windows + c]! params.samples (lengthsOf (drawRecords g s c)) =
        some drawn ∧
      ∀ k (hk : k < (drawRecords g s c).length),
        drawnOf s (c * g + k) =
          some (shareOf drawn (passCount ((drawRecords g s c).take k)) (drawRecords g s c)[k].passes.size) ∧
        (s.windows[c * g + k]!.verdict = some .accept →
          shareOf drawn (passCount ((drawRecords g s c).take k)) (drawRecords g s c)[k].passes.size = #[] ∨
          ∃ openings : Array Opening, checkWindow reg reg.digest s.appends (drawRecords g s c)[k]
            (shareOf drawn (passCount ((drawRecords g s c).take k)) (drawRecords g s c)[k].passes.size)
            openings = .accept) := by
  have hs : ∀ u : Nat, s.windows[u]!.plans.size = s.windows[u]!.roots.size := sizes_all hI
  refine ⟨drawn, by rw [lengthsOf_draw hs, ← hdig, ← hco, ← hp]; exact hdraws, fun k hk => ?_⟩
  have hlen := length_drawRecords (g := g) s c
  have hmem : c * g + k ∈ drawWindows g s.params.windows c :=
    List.mem_map.2 ⟨k, List.mem_range.2 (by omega), rfl⟩
  have hshk := hsh _ hmem
  have hbase : passCount ((drawRecords g s c).take k) = passesBefore s (c * g) (c * g + k) :=
    passCount_draw hs c k (by omega)
  have hn : (drawRecords g s c)[k].passes.size = s.windows[c * g + k]!.plans.size := by
    rw [drawRecords_get s c k hk, record_size, hs]
  rw [hbase, hn]
  refine ⟨hshk, fun hv => ?_⟩
  obtain ⟨d, hd, hor⟩ := hA _ hv
  rw [hshk] at hd
  injection hd with hd
  subst hd
  rw [drawRecords_get s c k hk, ← hdig, ← hr]
  exact hor

/-- **The session link** windows a draw, at any `g ≥ 1`. -/
theorem sessionDrawsRecords : SessionDrawsRecords := by
  intro g params reg coins appendCoins notBeforeNs events rs s h c hg hm hcW hc
  rcases Nat.lt_or_ge g 2 with hg2 | hg2
  · obtain rfl : g = 1 := by omega
    rw [run_le_one (Nat.le_refl 1)] at h
    obtain ⟨⟨hInv, -, -⟩, -, hp, hr, hco, hdig, hL, -, hA⟩ :=
      run_inv (Link1 params reg coins) (fun _ _ _ _ hx he => link1_step hx he) _ events rs s
        (link1_init params reg coins appendCoins notBeforeNs hm) h
    rw [Nat.mul_one] at hcW hc
    obtain ⟨d, hd⟩ := Option.isSome_iff_exists.1 hc
    have hW : c < s.params.windows := by rw [hp]; exact hcW
    have hdraws := hL c d hd
    obtain ⟨_, _, -, -, -, hrange, -⟩ := Properties.PoubCheck.drawIsSubset _ _ _ _ _ _ hdraws
    have hall : shareOf d 0 s.windows[c]!.plans.size = d :=
      shareOf_all d _ fun k hk => by simpa using (hrange k hk).1
    refine link_out hInv hp hr hco hdig hA c d (by rw [drawLengths_one hW]; exact hdraws) ?_
    intro u hu
    rw [drawWindows_one hW, List.mem_singleton] at hu
    rw [hu]
    have h0 : passesBefore s (c * 1) c = 0 := by simp [passesBefore]
    rw [h0, hall]
    exact hd
  · obtain ⟨⟨⟨hInv, -, -⟩, -⟩, -, hp, hr, hco, hdig, hL, hA⟩ :=
      srun_inv g (LinkInv g params reg coins) (fun _ _ _ _ hx he => linkInv_step hg2 hx he) _ events rs s
        (linkInv_init g params reg coins appendCoins notBeforeNs hm) h
    obtain ⟨drawn, hdraws, hsh⟩ := hL c hc
    exact link_out hInv hp hr hco hdig hA c drawn hdraws hsh

/-- **A draw's accepted windows open its values**: the session link, and `acceptDrawsOpenable` on the openings each
window passed with. -/
theorem acceptedDrawsOpenable : AcceptedDrawsOpenable := by
  intro g params reg coins appendCoins notBeforeNs events rs s h c hg hm hcW hc hall
  obtain ⟨drawn, hdraws, hk⟩ :=
    sessionDrawsRecords g params reg coins appendCoins notBeforeNs events rs s h c hg hm hcW hc
  obtain ⟨L, us, hs, hus⟩ := acceptDrawsOpenable reg s.appends c coins[params.windows + c]! params.samples
    (drawRecords g s c) drawn hdraws
  refine ⟨L, us, hs, ?_⟩
  have hex : ∀ k (hk' : k < (drawRecords g s c).length), ∃ o : Array Opening,
      shareOf drawn (passCount ((drawRecords g s c).take k)) (drawRecords g s c)[k].passes.size = #[] ∨
      checkWindow reg reg.digest s.appends (drawRecords g s c)[k]
        (shareOf drawn (passCount ((drawRecords g s c).take k)) (drawRecords g s c)[k].passes.size) o = .accept := by
    intro k hk'
    rcases (hk k hk').2 (hall k hk') with h0 | ⟨o, ho⟩
    · exact ⟨#[], Or.inl h0⟩
    · exact ⟨o, Or.inr ho⟩
  refine hus (fun k => if hk' : k < (drawRecords g s c).length then Classical.choose (hex k hk') else #[])
    (fun k hk' => ?_)
  try dsimp only
  rw [dite_eq_left hk']
  exact Classical.choose_spec (hex k hk')

end Properties.PoubCheck.Shared
