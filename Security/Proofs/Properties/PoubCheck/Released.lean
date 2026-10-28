import Batteries.Tactic.OpenPrivate
import Proofs.Properties.PoubCheck.Session

/-!
# Released blocks stay out of every later plan (`ReleasedStayOut`)

`release` merges the accepted runs into `gone`, half-open intervals that, merged, still cover every block they covered
(`merge_spec`); every other event leaves `gone` as it is. `receive` refuses a plan with a run overlapping an interval
of `gone`, and each block a plan reads lies in one of its runs (`block_mem`), so a plan it accepts reads no block `gone`
covers (`avoid_of_check`). The invariant (`Rel R old`): the points `R` stay covered by `gone`, and every pass recorded
past the counts `old` reads none of them. It holds from `State.init` with no points, and from an accepted release on
with that release's blocks and the passes recorded before it.
-/

open PoubCheck PoubCheck.Session

open private PoubCheck.Session.merge PoubCheck.Session.live from PoubCheck.Session
open private PoubCheck.endsOf from PoubCheck.Plan

namespace Properties.PoubCheck

/-! ## Merging released runs -/

/-- `x` lies in one of the half-open intervals `gs`. -/
def Covered (gs : Array (Nat × Nat)) (x : Nat) : Prop := ∃ g ∈ gs, g.1 ≤ x ∧ x < g.2

/-- `merge`'s step: an interval starting where the last one ends extends it. -/
def mergeStep (out : Array (Nat × Nat)) (ab : Nat × Nat) : Array (Nat × Nat) :=
  match out.back? with
  | some (oa, ob) => if ob == ab.1 then out.pop.push (oa, ab.2) else out.push (ab.1, ab.2)
  | none => out.push (ab.1, ab.2)

theorem merge_eq (xs : Array (Nat × Nat)) :
    PoubCheck.Session.merge xs =
      (xs.qsort fun a b => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2)).foldl mergeStep #[] := rfl

theorem mergeStep_spec (out : Array (Nat × Nat)) (ab : Nat × Nat) (hout : ∀ g ∈ out, g.1 ≤ g.2) (hab : ab.1 ≤ ab.2) :
    (∀ g ∈ mergeStep out ab, g.1 ≤ g.2) ∧
      ∀ x, (Covered out x ∨ (ab.1 ≤ x ∧ x < ab.2)) → Covered (mergeStep out ab) x := by
  unfold mergeStep
  split
  · rename_i oa ob hb
    obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
    have hy : oa ≤ ob := hout (oa, ob) (by simp)
    split
    · rename_i he
      have he : ob = ab.1 := by simpa using he
      rw [Array.pop_push]
      refine ⟨fun g hg => ?_, fun x hx => ?_⟩
      · rcases Array.mem_push.1 hg with hg | rfl
        · exact hout g (by simp [hg])
        · dsimp only; omega
      · rcases hx with ⟨g, hg, h1, h2⟩ | ⟨h1, h2⟩
        · rcases Array.mem_push.1 hg with hg | rfl
          · exact ⟨g, by simp [hg], h1, h2⟩
          · exact ⟨(oa, ab.2), by simp, h1, by dsimp only at h2 ⊢; omega⟩
        · exact ⟨(oa, ab.2), by simp, by dsimp only; omega, h2⟩
    · refine ⟨fun g hg => ?_, fun x hx => ?_⟩
      · rcases Array.mem_push.1 hg with hg | rfl
        · exact hout g hg
        · exact hab
      · rcases hx with ⟨g, hg, h1, h2⟩ | ⟨h1, h2⟩
        · exact ⟨g, Array.mem_push.2 (Or.inl hg), h1, h2⟩
        · exact ⟨(ab.1, ab.2), by simp, h1, h2⟩
  · refine ⟨fun g hg => ?_, fun x hx => ?_⟩
    · rcases Array.mem_push.1 hg with hg | rfl
      · exact hout g hg
      · exact hab
    · rcases hx with ⟨g, hg, h1, h2⟩ | ⟨h1, h2⟩
      · exact ⟨g, Array.mem_push.2 (Or.inl hg), h1, h2⟩
      · exact ⟨(ab.1, ab.2), by simp, h1, h2⟩

theorem fold_spec (l : List (Nat × Nat)) :
    ∀ out : Array (Nat × Nat), (∀ g ∈ out, g.1 ≤ g.2) → (∀ g ∈ l, g.1 ≤ g.2) →
      (∀ g ∈ l.foldl mergeStep out, g.1 ≤ g.2) ∧
        ∀ x, (Covered out x ∨ ∃ g ∈ l, g.1 ≤ x ∧ x < g.2) → Covered (l.foldl mergeStep out) x := by
  induction l with
  | nil => intro out hout _; exact ⟨hout, fun x hx => by simpa using hx⟩
  | cons ab l ih =>
    intro out hout hl
    obtain ⟨hw, hc⟩ := mergeStep_spec out ab hout (hl ab (by simp))
    obtain ⟨hw', hc'⟩ := ih (mergeStep out ab) hw (fun g hg => hl g (by simp [hg]))
    refine ⟨hw', fun x hx => hc' x ?_⟩
    rcases hx with hx | ⟨g, hg, h1, h2⟩
    · exact Or.inl (hc x (Or.inl hx))
    · rcases List.mem_cons.1 hg with rfl | hg
      · exact Or.inl (hc x (Or.inr ⟨h1, h2⟩))
      · exact Or.inr ⟨g, hg, h1, h2⟩

/-- **Merged, the intervals still cover every block they covered.** -/
theorem merge_spec (xs : Array (Nat × Nat)) (hwf : ∀ g ∈ xs, g.1 ≤ g.2) :
    (∀ g ∈ PoubCheck.Session.merge xs, g.1 ≤ g.2) ∧
      ∀ x, Covered xs x → Covered (PoubCheck.Session.merge xs) x := by
  rw [merge_eq, ← Array.foldl_toList]
  have hperm := qsort_toList_perm xs (fun a b : Nat × Nat => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2))
  have hmem : ∀ g, g ∈ (xs.qsort fun a b : Nat × Nat => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2)).toList ↔ g ∈ xs :=
    fun g => by rw [hperm.mem_iff, Array.mem_toList_iff]
  obtain ⟨hw, hc⟩ := fold_spec _ #[] (by simp) (fun g hg => hwf g ((hmem g).1 hg))
  exact ⟨hw, fun x ⟨g, hg, h1, h2⟩ => hc x (Or.inr ⟨g, (hmem g).2 hg, h1, h2⟩)⟩

/-! ## The blocks a plan reads -/

/-- A plan's running ends are its counts' prefix sums. -/
def EndsOk (pl : Plan) : Prop := PrefixSums pl.ends (pl.runs.toList.map (·.2))

theorem make_spec {runs : Array (Int × Int)} {blocks : Nat} {pl : Plan} (h : Plan.make? runs blocks = some pl) :
    EndsOk pl ∧ ∀ r ∈ runs, 0 ≤ r.1 ∧ (r.1.toNat, r.2.toNat) ∈ pl.runs := by
  unfold Plan.make? at h
  simp only [Bool.not_eq_true', Option.bind_eq_bind, Option.bind_none] at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rename_i hin
  split at h
  · cases h
  split at h
  · cases h
  cases h
  refine ⟨endsOf_prefix _, fun r hr => ⟨?_, Array.mem_map.2 ⟨r, hr, rfl⟩⟩⟩
  have := Array.any_eq_false.1 (Bool.not_eq_true _ ▸ hin)
  obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hr
  have h' := this i hi
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at h'
  omega

theorem endsOk_whole (N : Nat) : EndsOk (Plan.whole N) :=
  ⟨rfl, fun i hi => by
    have : i = 0 := by simp [Plan.whole] at hi; omega
    subst this; simp [Plan.whole]⟩

/-- **Each block a plan reads lies in one of its runs.** -/
theorem block_mem {pl : Plan} (hP : EndsOk pl) {p : Nat} (hp : p < pl.length) :
    ∃ r ∈ pl.runs, r.1 ≤ pl.block p ∧ pl.block p < r.1 + r.2 := by
  obtain ⟨blocks, rs, ends⟩ := pl
  have hlen : Plan.length ⟨blocks, rs, ends⟩ = (rs.toList.map (·.2)).sum := back_getD_of_prefix hP
  rw [hlen] at hp
  obtain ⟨hr, hlo, hhi, hb⟩ := block_spec blocks rs ends hP p hp
  refine ⟨rs[bisectRight ends p], Array.getElem_mem hr, ?_⟩
  rw [hb, getElem!_pos rs _ hr]
  rw [getElem!_pos rs _ hr] at hhi
  omega

/-- **A plan whose runs overlap no interval of `gone` reads no block it covers.** -/
theorem avoid_of_check {R : Nat → Prop} {s : State} (hcov : ∀ x, R x → Covered s.gone x) {plan : Plan}
    (hP : EndsOk plan) (hchk : ¬(!s.gone.isEmpty && plan.runs.any fun x => s.isReleased x.fst x.snd) = true) :
    ∀ p, p < plan.length → ¬ R (plan.block p) := by
  intro p hp hR
  obtain ⟨g, hg, h1, h2⟩ := hcov _ hR
  obtain ⟨r, hr, h3, h4⟩ := block_mem hP hp
  apply hchk
  have hne : s.gone.isEmpty = false := by
    cases he : s.gone.isEmpty
    · rfl
    · rw [Array.isEmpty_iff.1 he] at hg; simp at hg
  rw [hne, Bool.not_false, Bool.true_and, Array.any_eq_true']
  refine ⟨r, hr, ?_⟩
  unfold State.isReleased
  rw [Array.any_eq_true']
  exact ⟨g, hg, by simp only [Bool.and_eq_true, decide_eq_true_eq]; omega⟩

theorem endsOk_changed {pl pl' : Plan} {added removed : Array (Int × Int)} {blocks : Nat}
    (h : pl.changed added removed blocks = some pl') : EndsOk pl' := by
  rcases changed_make pl added removed blocks with h0 | ⟨runs, h1⟩
  · rw [h0] at h; cases h
  · rw [h1] at h; exact (make_spec h).1

/-! ## The invariant -/

/-- The points `R` stay covered by `gone` (intervals each `first ≤ end`), the windows hold at least the passes `old`
counts, and every pass past those counts reads none of `R`. -/
structure RelOn (R : Nat → Prop) (old : Array Nat) (ws : Array Win) (gone : Array (Nat × Nat)) : Prop where
  wf : ∀ g ∈ gone, g.1 ≤ g.2
  covers : ∀ x, R x → Covered gone x
  grown : old.size ≤ ws.size
  kept : ∀ t (ht : t < old.size) (ht' : t < ws.size), old[t] ≤ ws[t].plans.size
  avoid : ∀ t (ht : t < ws.size) i (hi : i < ws[t].plans.size), (old.size ≤ t ∨ old[t]! ≤ i) →
    ∀ p, p < ws[t].plans[i].length → ¬ R (ws[t].plans[i].block p)

def Rel (R : Nat → Prop) (old : Array Nat) (s : State) : Prop := RelOn R old s.windows s.gone

theorem rel_init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat) :
    Rel (fun _ => False) #[] (State.init params reg coins appendCoins notBeforeNs) :=
  ⟨by simp [State.init], fun _ h => h.elim, by simp, fun t ht => by simp at ht, fun _ _ _ _ _ _ _ h => h⟩

/-- Windows with `ws`'s plans, one for one. -/
def SamePlans (ws ws' : Array Win) : Prop :=
  ws'.size = ws.size ∧ ∀ t (h : t < ws.size) (h' : t < ws'.size), ws'[t].plans = ws[t].plans

theorem samePlans_modify (ws : Array Win) (u : Nat) (f : Win → Win) (hf : ∀ w, (f w).plans = w.plans) :
    SamePlans ws (ws.modify u f) := by
  refine ⟨Array.size_modify, fun t h h' => ?_⟩
  rw [Array.getElem_modify]
  split
  · exact hf _
  · rfl

theorem relOn_same {R old ws gone} {ws' : Array Win} (h : RelOn R old ws gone) (hs : SamePlans ws ws') :
    RelOn R old ws' gone := by
  obtain ⟨hsz, hw⟩ := hs
  have e : ∀ t (h' : t < ws'.size), ws'[t].plans = ws[t].plans := fun t h' => hw t (by omega) h'
  refine ⟨h.wf, h.covers, by have := h.grown; omega, fun t ht ht' => ?_, fun t ht i hi hn => ?_⟩
  · rw [e t ht']; exact h.kept t ht (by omega)
  · have hi' : i < ws[t].plans.size := by rw [← e t ht]; exact hi
    have := h.avoid t (by omega) i hi' hn
    simp only [e t ht]; exact this

theorem relOn_push {R old ws gone} (h : RelOn R old ws gone) (w : Win) (hp : w.plans = #[]) :
    RelOn R old (ws.push w) gone := by
  have g : ∀ t (ht : t < (ws.push w).size), ¬ t < ws.size → (ws.push w)[t] = w := by
    intro t ht hn; simp [Array.getElem_push, hn]
  have k : ∀ t (ht : t < (ws.push w).size) (hl : t < ws.size), (ws.push w)[t] = ws[t] := by
    intro t ht hl; simp [Array.getElem_push, hl]
  have hg := h.grown
  refine ⟨h.wf, h.covers, by simp; omega, fun t ht ht' => ?_, fun t ht i hi hn => ?_⟩
  · simp only [k t ht' (by omega)]; exact h.kept t ht (by omega)
  · by_cases hl : t < ws.size
    · simp only [k t ht hl] at hi ⊢; exact h.avoid t hl i hi hn
    · simp only [g t ht hl, hp] at hi; simp at hi

theorem relOn_add {R old ws gone} {below now : Nat} (h : RelOn R old ws gone) (u : Nat) (root : ByteArray)
    (plan : Plan) (post : Nat) (st : Int) (head : Option ByteArray) (hav : ∀ p, p < plan.length → ¬ R (plan.block p)) :
    RelOn R old (ws.modify u (addPass root plan post st head below now)) gone := by
  have get : ∀ t (ht : t < (ws.modify u (addPass root plan post st head below now)).size),
      (ws.modify u (addPass root plan post st head below now))[t] =
        if u = t then addPass root plan post st head below now (ws[t]'(by simpa using ht)) else ws[t]'(by simpa using ht) :=
    fun t ht => Array.getElem_modify ht
  have hsz : (ws.modify u (addPass root plan post st head below now)).size = ws.size := Array.size_modify
  refine ⟨h.wf, h.covers, by rw [hsz]; exact h.grown, fun t ht ht' => ?_, fun t ht i hi hn => ?_⟩
  · rw [get t ht']
    have := h.kept t ht (by omega)
    split
    · simp only [addPass, Array.size_push]; omega
    · exact this
  · revert hi
    rw [get t ht]
    split
    · intro hi
      simp only [addPass, Array.getElem_push]
      split
      · exact h.avoid t (by omega) i (by assumption) hn
      · exact hav
    · intro hi; exact h.avoid t (by omega) i hi hn

theorem rel_fail {R old} {s : State} (h : Rel R old s) (t : Nat) (f : Failure) (at? : Option (Nat × Nat)) :
    Rel R old (s.fail t f at?).2 := by
  unfold State.fail
  dsimp only
  split
  · exact relOn_same h (samePlans_modify _ _ _ fun _ => rfl)
  · exact h

theorem rel_final {R old} {s : State} (h : Rel R old s) (u : Nat) :
    Rel R old { s with windows := s.windows.modify u ({· with final := true}) } :=
  relOn_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem rel_boundaryNs {R old} {s : State} (h : Rel R old s) (u now : Nat) :
    Rel R old { s with windows := s.windows.modify u ({· with boundaryNs := some now}) } :=
  relOn_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem rel_releaseCoin {R old} {s : State} (h : Rel R old s) (t now : Nat) : Rel R old (s.releaseCoin t now) :=
  relOn_push h _ rfl

theorem rel_draw {R old} {s : State} (h : Rel R old s) (u : Nat) (d : Array (Nat × Nat)) (now : Nat) :
    Rel R old { s with windows := s.windows.modify u ({· with drawn := some d, openedNs := some now}) } :=
  relOn_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem rel_accept {R old} {s : State} (h : Rel R old s) (u : Nat) :
    Rel R old { s with windows := s.windows.modify u ({· with verdict := some .accept}) } :=
  relOn_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem rel_merge {R old} {s : State} (h : Rel R old s) (ys : Array (Nat × Nat)) (hys : ∀ g ∈ ys, g.1 ≤ g.2)
    (n : Nat) : Rel R old { s with gone := PoubCheck.Session.merge (s.gone ++ ys), released := n } := by
  obtain ⟨hw, hc⟩ := merge_spec (s.gone ++ ys) fun g hg => (Array.mem_append.1 hg).elim (h.wf g) (hys g)
  refine ⟨hw, fun x hx => hc x ?_, h.grown, h.kept, h.avoid⟩
  obtain ⟨g, hg, h1, h2⟩ := h.covers x hx
  exact ⟨g, Array.mem_append.2 (Or.inl hg), h1, h2⟩

theorem wf_runs (pl : Plan) : ∀ g ∈ pl.runs.map (fun (f, c) => (f, f + c)), g.1 ≤ g.2 := by
  intro g hg
  obtain ⟨⟨f, c⟩, -, rfl⟩ := Array.mem_map.1 hg
  exact Nat.le_add_right f c

/-! ## The events -/

theorem rel_start {R old} {s : State} {now : Nat} {r : Result} {s' : State} (h : Rel R old s)
    (he : start s now = (r, s')) : Rel R old s' := by
  unfold start at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact rel_releaseCoin h _ _

theorem rel_boundary {R old} {s : State} {now : Nat} {r : Result} {s' : State} (h : Rel R old s)
    (he : boundary s now = (r, s')) : Rel R old s' := by
  unfold boundary at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact rel_releaseCoin (rel_boundaryNs h _ _) _ _
    | exact rel_boundaryNs h _ _

theorem rel_append {R old} {s : State} {root : ByteArray} {count : Int} {r : Result} {s' : State} (h : Rel R old s)
    (he : append s root count = .ok (r, s')) : Rel R old s' := by
  unfold append at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact rel_fail h _ _ _

theorem rel_release {R old} {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State}
    (h : Rel R old s) (he : release s runs withdrawn = (r, s')) : Rel R old s' := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact rel_fail h _ _ _
    | exact rel_merge h _ (wf_runs _) _

theorem rel_close {R old} {s : State} {t : Int} {now : Nat} {r : Result} {s' : State} (h : Rel R old s)
    (he : close s t now = .ok (r, s')) : Rel R old s' := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact rel_fail (rel_final h _) _ _ _
    | exact rel_draw (rel_final h _) _ _ _

theorem rel_verify {R old} {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State}
    (h : Rel R old s) (he : verify s t openings = (r, s')) : Rel R old s' := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact rel_fail h _ _ _
    | exact rel_accept h _

theorem rel_receive {R old} {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {step : Option Int} {now : Nat} {r : Result}
    {s' : State} (h : Rel R old s) (he : receive s t i root runs change step now = (r, s')) : Rel R old s' := by
  unfold receive at he
  split at he
  · cases he; exact h
  split at he
  · cases he; exact h
  split at he
  · cases he; exact rel_fail h _ _ _
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, Rel R old s₂ → jp () s₂ = (r, s') → Rel R old s' := by
    intro s₂ h₂ hj
    dsimp -zeta only [jp] at hj
    extract_lets t₂ N₂ ext₂ plan₂ s₃ below₂ head₂ jp₂ at hj
    have hW : ∀ plan, plan₂ = some plan → EndsOk plan := by
      intro plan hp
      dsimp only [plan₂, ext₂, N₂] at hp
      split at hp
      · cases hp
      · exact endsOk_changed hp
      · cases hp; exact endsOk_whole _
      · exact (make_spec hp).1
    have key₂ : jp₂ () = (r, s') → Rel R old s' := by
      intro hj
      dsimp only [jp₂] at hj
      generalize hq : plan₂ = q at hj
      repeat' split at hj
      all_goals (try dsimp only at hj)
      all_goals cases hj
      all_goals first
        | exact rel_fail h₂ _ _ _
        | exact relOn_add h₂ _ _ _ _ _ _
            (avoid_of_check h₂.covers (hW _ (by first | exact hq | exact hq.trans ‹_›)) ‹_›)
    split at hj
    · split at hj
      · cases hj; exact rel_fail h₂ _ _ _
      · exact key₂ hj
    · exact key₂ hj
  split at he
  · exact key s₁ (rel_final h _) he
  · split at he
    · cases he; exact rel_fail h _ _ _
    · exact key s₀ h he

theorem step_rel {R old} {s : State} {e : Event} {r : Result} {s' : State} (h : Rel R old s)
    (he : step s e = .ok (r, s')) : Rel R old s' := by
  cases e with
  | start now => injection he with he; exact rel_start h he
  | boundary now => injection he with he; exact rel_boundary h he
  | append root count => exact rel_append h he
  | release runs withdrawn => injection he with he; exact rel_release h he
  | receive t i root runs change st now => injection he with he; exact rel_receive h he
  | close t now => exact rel_close h he
  | verify t openings => injection he with he; exact rel_verify h he

theorem run_rel {R old} {s : State} {events : Array Event} {rs : Array Result} {s' : State} (h : Rel R old s)
    (he : run s events = .ok (rs, s')) : Rel R old s' :=
  run_inv (Rel R old) (fun _ _ _ _ hs he => step_rel hs he) _ _ _ _ h he

/-! ## The statement -/

theorem release_ok {s : State} {runs withdrawn : Array (Int × Int)} {s' : State}
    (he : release s runs withdrawn = (.ok, s')) (hwf : ∀ g ∈ s.gone, g.1 ≤ g.2) :
    s'.windows = s.windows ∧ (∀ g ∈ s'.gone, g.1 ≤ g.2) ∧
      ∀ x : Nat, (∃ r ∈ runs, r.1 ≤ (x : Int) ∧ (x : Int) < r.1 + r.2) → Covered s'.gone x := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  · cases he
    rename_i hl
    unfold PoubCheck.Session.live at hl
    cases hf : s.failed <;> simp [hf] at hl
  · cases he
  · cases he
  · rename_i pl hp _ _
    cases he
    have hw : ∀ g ∈ s.gone ++ pl.runs.map (fun (f, c) => (f, f + c)), g.1 ≤ g.2 :=
      fun g hg => (Array.mem_append.1 hg).elim (hwf g) (wf_runs pl g)
    refine ⟨rfl, (merge_spec _ hw).1, fun x ⟨r, hr, h1, h2⟩ => (merge_spec _ hw).2 x ?_⟩
    obtain ⟨h0, hm⟩ := (make_spec hp).2 r hr
    refine ⟨(r.1.toNat, r.1.toNat + r.2.toNat), Array.mem_append.2 (Or.inr (Array.mem_map.2 ⟨_, hm, rfl⟩)), ?_, ?_⟩
    · dsimp only; omega
    · dsimp only; omega
  · cases he

theorem releasedStayOut : ReleasedStayOut := by
  intro params reg coins appendCoins notBeforeNs events₁ events₂ runs withdrawn rs₁ rs s₁ s₂ s h₁ hrel h₂
  have r₁ : Rel (fun _ => False) #[] s₁ := run_rel (rel_init params reg coins appendCoins notBeforeNs) h₁
  injection hrel with hrel
  obtain ⟨hw, hwf, hcov⟩ := release_ok hrel r₁.wf
  have hsz : (s₁.windows.map (·.plans.size)).size = s₂.windows.size := by simp [hw]
  have hold : ∀ t (ht : t < s₁.windows.size), (s₁.windows.map (·.plans.size))[t]! = s₁.windows[t].plans.size :=
    fun t ht => by rw [getElem!_pos _ t (by simpa using ht), Array.getElem_map]
  have hpl : ∀ t (ht : t < s₂.windows.size), s₂.windows[t].plans.size = (s₁.windows[t]'(by rw [← hw]; exact ht)).plans.size :=
    fun t ht => by simp only [hw]
  have r₂ : Rel (fun x => ∃ r ∈ runs, r.1 ≤ (x : Int) ∧ (x : Int) < r.1 + r.2) (s₁.windows.map (·.plans.size)) s₂ := by
    refine ⟨hwf, hcov, by omega, fun t ht ht' => ?_, fun t ht i hi hn => ?_⟩
    · rw [Array.getElem_map, hpl t ht']
      exact Nat.le_refl _
    · exfalso
      rcases hn with hn | hn
      · omega
      · rw [hold t (by rw [← hw]; exact ht)] at hn
        have := hpl t ht
        omega
  have r := run_rel r₂ h₂
  intro rr hr t ht i hi hnew p hp hin
  refine r.avoid t ht i hi ?_ p hp ⟨rr, hr, hin⟩
  rcases hnew with hn | hn
  · left; simpa using hn
  · by_cases hl : t < s₁.windows.size
    · right
      rw [hold t hl]
      rw [getElem!_pos _ t hl] at hn
      exact hn
    · left; simp only [Array.size_map]; omega

end Properties.PoubCheck
