import Proofs.Properties.RemoteAudit.Shared

/-!
# The session link at `g ≥ 2` windows a draw (`SessionDrawsRecords`)

A run keeps, with `SF` and `Prefix`, two facts of every drawn draw (`LinkInv`, `linkInv_step`):

* `DrawLink`: the draw is `draws` over its windows' plans (`drawLengths`), on the session's digest and coin `W + c`,
  and each window of it holds its share (`passesBefore`). The close that draws it gives this (`drawIsSubset`), and
  every later event keeps it: a drawn window keeps its draw and plans (`Frz`), and so its share.
* `AcceptLink`: an accepted window is drawn, and its share is empty or `checkWindow` of it passed on a prefix of the
  appends, its record then its record now. Only `verify` and the close accept a window (`VK`, each event's
  `vk_*`): `verify` on `checkWindow`'s verdict (`acc_verify`), the close on an empty share (`acc_sclose`). A drawn
  window's record is kept (`record_frz`, as in `recordKeptAfterDraw`), and the appends only grow.

Under `Inv`'s `sizes` a window's record has its plans as passes, so the draw's lengths and shares are the records'
(`lengthsOf_draw`, `passCount_draw`): `sessionDrawsRecords`.
-/

open RemoteAudit RemoteAudit.Session

namespace Properties.RemoteAudit.Shared

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

/-! ## The invariant -/

/-- Each drawn draw is `draws` over its windows' plans, on the session's digest and coin `W + c`, and each window of
it holds its share. -/
def DrawLink (g : Nat) (s : State) : Prop :=
  ∀ c, (dr s.windows (c * g)).isSome → ∃ drawn,
    draws s.digest c s.coins[s.params.windows + c]! s.params.samples (drawLengths g s c) = some drawn ∧
    ∀ u ∈ drawWindows g s.params.windows c,
      dr s.windows u = some (shareOf drawn (passesBefore s (c * g) u) s.windows[u]!.plans.size)

/-- Each accepted window is drawn, and its share is empty or passed `checkWindow` on a prefix of the appends. -/
def AcceptLink (s : State) : Prop :=
  ∀ u, s.windows[u]!.verdict = some .accept → ∃ d, dr s.windows u = some d ∧
    (d = #[] ∨ ∃ (appends more : Array Append) (openings : Array Opening), s.appends = appends ++ more ∧
      checkWindow s.reg s.digest appends (s.record u) d openings = .accept)

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

theorem acceptLink_step {g : Nat} (hg : 2 ≤ g) {s : State} {e : Event} {r : Result} {s' : State}
    (hpre : Prefix s.windows) (hf : Frz s s') (hr : s'.reg = s.reg) (hA : AcceptLink s)
    (he : Shared.step g s e = .ok (r, s')) : AcceptLink s' := by
  intro u hu
  rcases acc_sstep hg he u hu with hold | hempty | ⟨d, o, hd, hc, rfl⟩
  · obtain ⟨d, hdr, hor⟩ := hA u hold
    have hsome : (dr s.windows u).isSome := by rw [hdr]; rfl
    refine ⟨d, by rw [(plans_frz hf hsome).2, hdr], ?_⟩
    rcases hor with h0 | ⟨A, more, o, hA', hc'⟩
    · exact Or.inl h0
    · obtain ⟨more', hm'⟩ := hf.2.2
      refine Or.inr ⟨A, more ++ more', o, by rw [hm', hA', Array.append_assoc], ?_⟩
      rw [hr, hf.2.1, record_frz hpre hf hsome]
      exact hc'
  · exact ⟨#[], by rw [dr_eq]; exact hempty, Or.inl rfl⟩
  · have hsome : (dr s.windows u).isSome := by rw [dr_eq, hd]; rfl
    refine ⟨d, ?_, Or.inr ⟨s.appends, #[], o, Array.append_empty.symm, ?_⟩⟩
    · rw [dr_eq]
      show (s.windows.modify u ({· with verdict := some .accept}))[u]!.drawn = some d
      rw [getElem!_modify']
      split <;> exact hd
    · rw [record_frz hpre hf hsome]
      exact hc

/-- What a run from `State.init` keeps at `g ≥ 2`: `SF`, `Prefix`, what it started with, and both links. -/
def LinkInv (g : Nat) (params : Params) (reg : Registration) (coins : Array ByteArray) (s : State) : Prop :=
  SF g s ∧ Prefix s.windows ∧ s.params = params ∧ s.reg = reg ∧ s.coins = coins ∧ s.digest = reg.digest ∧
    DrawLink g s ∧ AcceptLink s

theorem linkInv_step {g : Nat} (hg : 2 ≤ g) {params : Params} {reg : Registration} {coins : Array ByteArray}
    {s : State} {e : Event} {r : Result} {s' : State} (h : LinkInv g params reg coins s)
    (he : Shared.step g s e = .ok (r, s')) : LinkInv g params reg coins s' := by
  obtain ⟨hSF, hpre, hp, hr, hc, hdig, hL, hA⟩ := h
  obtain ⟨hSF', hf⟩ := sf_step hg hSF he
  obtain ⟨-, ⟨hp', hr', -⟩, hc', -, -⟩ := sstep hg hSF.1.1 he
  refine ⟨hSF', sstep_prefix hg hpre he, hp'.trans hp, hr'.trans hr, hc'.trans hc, hf.2.1.trans hdig, ?_, ?_⟩
  · exact drawLink_step hg hL hf hp' hc' hf.2.1 he
  · exact acceptLink_step hg hpre hf hr' hA he

theorem linkInv_init (g : Nat) (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray)
    (notBeforeNs : Nat) (hm : 1 ≤ params.minPasses) :
    LinkInv g params reg coins (State.init params reg coins appendCoins notBeforeNs) := by
  refine ⟨⟨finv_init params reg coins appendCoins notBeforeNs hm, fun u h => by simp [dr, State.init] at h,
    fun u j h => by simp [dr, State.init] at h⟩, fun t j _ ht => by simp [dr, State.init] at ht, rfl, rfl, rfl, rfl,
    fun c h => by simp [dr, State.init] at h, fun u h => ?_⟩
  rw [getElem!_neg (State.init params reg coins appendCoins notBeforeNs).windows u (by simp [State.init])] at h
  cases h

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

/-- **The session link** at `g ≥ 2` windows a draw. -/
theorem sessionDrawsRecords : SessionDrawsRecords := by
  intro g params reg coins appendCoins notBeforeNs events rs s h c hg hm hc
  obtain ⟨⟨⟨hInv, -, -⟩, -⟩, -, hp, hr, hco, hdig, hL, hA⟩ :=
    srun_inv g (LinkInv g params reg coins) (fun _ _ _ _ hx he => linkInv_step hg hx he) _ events rs s
      (linkInv_init g params reg coins appendCoins notBeforeNs hm) h
  have hs : ∀ u : Nat, s.windows[u]!.plans.size = s.windows[u]!.roots.size := sizes_all hInv
  obtain ⟨drawn, hdraws, hsh⟩ := hL c hc
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

end Properties.RemoteAudit.Shared
