import Proofs.Properties.PoubCheck.Released

/-!
# Each append's blocks are read within `lifetime` windows (`LifetimeCapped`)

`receive` gives every append not yet joined the window it is filling (`joined`), so `joined` has one entry an append
received before the latest pass, each at most the window being filled, and nondecreasing. `staleBelow t` is the first
index of the first append that joined within `L` windows of `t` (the store's end if none did), and `receive` refuses a
plan with a run overlapping `[N, staleBelow t)`. The appends are the store's next indices in order (`Inv.firsts`), so a
block a plan reads past `N` lies in exactly one append, and the plan is refused unless that append joined within `L`
windows. The invariant (`Life`): every block past `N` a recorded pass in window `t` reads lies in an append `k` with
`joined[k] ≤ t < joined[k] + L`.
-/

open PoubCheck PoubCheck.Session

namespace Properties.PoubCheck

/-! ## The store's appends -/

theorem sum_take_le (c : List Nat) {i j : Nat} (h : i ≤ j) : (c.take i).sum ≤ (c.take j).sum := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [List.take_add, List.sum_append]
  exact Nat.le_add_right _ _

theorem sum_take_succ (c : List Nat) (k : Nat) (hk : k < c.length) : (c.take (k + 1)).sum = (c.take k).sum + c[k] := by
  rw [List.take_add, List.sum_append]
  congr 1
  rw [List.take_one, List.head?_drop, List.getElem?_eq_getElem hk]
  simp

theorem cover_list (c : List Nat) : ∀ x, x < c.sum → ∃ k, ∃ hk : k < c.length, (c.take k).sum ≤ x ∧ x < (c.take k).sum + c[k] := by
  induction c with
  | nil => intro x hx; simp at hx
  | cons a c ih =>
    intro x hx
    by_cases ha : x < a
    · exact ⟨0, by simp, by simp, by simpa using ha⟩
    · obtain ⟨k, hk, h1, h2⟩ := ih (x - a) (by simp at hx; omega)
      refine ⟨k + 1, by simp; omega, ?_, ?_⟩
      · simp only [List.take_succ_cons, List.sum_cons]; omega
      · simp only [List.take_succ_cons, List.sum_cons, List.getElem_cons_succ]; omega

/-- The appends' counts, in order. -/
abbrev counts (as : Array Append) : List Nat := as.toList.map (·.count)

theorem counts_take (as : Array Append) (k : Nat) : (as.toList.take k).map (·.count) = (counts as).take k :=
  List.map_take ..

theorem counts_get (as : Array Append) (k : Nat) (hk : k < as.size) :
    (counts as)[k]'(by simpa using hk) = as[k].count := by
  simp

/-- An earlier append ends at or before a later one starts. -/
theorem appends_ordered {N : Nat} {as : Array Append}
    (hf : ∀ k (hk : k < as.size), as[k].first = N + ((as.toList.take k).map (·.count)).sum)
    {k₁ k₂ : Nat} (h₁ : k₁ < as.size) (h₂ : k₂ < as.size) (hlt : k₁ < k₂) :
    as[k₁].first + as[k₁].count ≤ as[k₂].first := by
  rw [hf k₁ h₁, hf k₂ h₂, counts_take, counts_take, ← counts_get as k₁ h₁, Nat.add_assoc,
    ← sum_take_succ _ _ (by simpa using h₁)]
  exact Nat.add_le_add_left (sum_take_le _ hlt) _

/-- A block between `N` and the store's end lies in an append. -/
theorem appends_cover {N extent : Nat} {as : Array Append} (he : extent = N + ((as.toList.map (·.count)).sum))
    (hf : ∀ k (hk : k < as.size), as[k].first = N + ((as.toList.take k).map (·.count)).sum)
    {b : Nat} (h₁ : N ≤ b) (h₂ : b < extent) :
    ∃ k, ∃ hk : k < as.size, as[k].first ≤ b ∧ b < as[k].first + as[k].count := by
  obtain ⟨k, hk, hlo, hhi⟩ := cover_list (counts as) (b - N) (by simp only [counts]; omega)
  have hk' : k < as.size := by simpa using hk
  refine ⟨k, hk', ?_, ?_⟩
  · rw [hf k hk', counts_take]; omega
  · rw [hf k hk', counts_take, ← counts_get as k hk']; omega

/-- The append holding a block is the only one. -/
theorem appends_unique {N : Nat} {as : Array Append}
    (hf : ∀ k (hk : k < as.size), as[k].first = N + ((as.toList.take k).map (·.count)).sum)
    {b k₁ k₂ : Nat} (h₁ : k₁ < as.size) (h₂ : k₂ < as.size)
    (hb₁ : as[k₁].first ≤ b ∧ b < as[k₁].first + as[k₁].count) (hb₂ : as[k₂].first ≤ b ∧ b < as[k₂].first + as[k₂].count) :
    k₁ = k₂ := by
  rcases Nat.lt_trichotomy k₁ k₂ with hlt | heq | hlt
  · have := appends_ordered hf h₁ h₂ hlt; omega
  · exact heq
  · have := appends_ordered hf h₂ h₁ hlt; omega

theorem takeWhile_stop {α} (p : α → Bool) (l : List α) (k : Nat) (hk : (l.takeWhile p).length = k)
    (h : k < l.length) : p l[k] = false := by
  subst hk
  induction l with
  | nil => simp at h
  | cons a l ih =>
    by_cases ha : p a = true
    · simp only [List.takeWhile_cons, ha, ite_true, List.length_cons] at h ⊢
      simp only [List.getElem_cons_succ]
      exact ih (by omega)
    · simp only [List.takeWhile_cons, ha]
      simpa using ha

/-! ## The invariant -/

/-- `joined` has at most one entry an append, each at most the window being filled, nondecreasing; and every block
past `N` a recorded pass in window `t` reads lies in an append `k` that joined at or before `t` and within `L` windows
of it. -/
structure LifeOn (N L : Nat) (ws : Array Win) (appends : Array Append) (joined : Array Nat) (filling : Nat) : Prop where
  size : joined.size ≤ appends.size
  le : ∀ k (hk : k < joined.size), joined[k] ≤ filling
  mono : ∀ k₁ k₂ (h₁ : k₁ < joined.size) (h₂ : k₂ < joined.size), k₁ ≤ k₂ → joined[k₁] ≤ joined[k₂]
  life : ∀ t (ht : t < ws.size) i (hi : i < ws[t].plans.size) p, p < ws[t].plans[i].length →
    N ≤ ws[t].plans[i].block p → ∃ k, ∃ hk : k < joined.size, ∃ hk' : k < appends.size,
      appends[k].first ≤ ws[t].plans[i].block p ∧ ws[t].plans[i].block p < appends[k].first + appends[k].count ∧
      joined[k] ≤ t ∧ t < joined[k] + L

def Life (s : State) : Prop := LifeOn s.reg.blocks s.params.lifetime s.windows s.appends s.joined s.filling

theorem life_init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat) :
    Life (State.init params reg coins appendCoins notBeforeNs) :=
  ⟨by simp [State.init], fun k hk => by simp [State.init] at hk, fun k₁ _ h₁ => by simp [State.init] at h₁,
    fun t ht => by simp [State.init] at ht⟩

theorem lifeOn_same {N L ws appends joined filling} {ws' : Array Win} (h : LifeOn N L ws appends joined filling)
    (hs : SamePlans ws ws') : LifeOn N L ws' appends joined filling := by
  obtain ⟨hsz, hw⟩ := hs
  have e : ∀ t (h' : t < ws'.size), ws'[t].plans = ws[t].plans := fun t h' => hw t (by omega) h'
  refine ⟨h.size, h.le, h.mono, fun t ht i hi p hp hN => ?_⟩
  have hi' : i < ws[t].plans.size := by rw [← e t ht]; exact hi
  simp only [e t ht] at hp hN ⊢
  exact h.life t (by omega) i hi' p hp hN

theorem lifeOn_push {N L ws appends joined filling} (h : LifeOn N L ws appends joined filling) (w : Win)
    (hp : w.plans = #[]) : LifeOn N L (ws.push w) appends joined filling := by
  refine ⟨h.size, h.le, h.mono, fun t ht i hi => ?_⟩
  by_cases hl : t < ws.size
  · have k : (ws.push w)[t] = ws[t] := by simp [Array.getElem_push, hl]
    simp only [k] at hi ⊢
    exact h.life t hl i hi
  · have g : (ws.push w)[t] = w := by simp [Array.getElem_push, hl]
    simp only [g, hp] at hi; simp at hi

theorem lifeOn_append {N L ws appends joined filling} (h : LifeOn N L ws appends joined filling) (a : Append) :
    LifeOn N L ws (appends.push a) joined filling := by
  have hsz := h.size
  refine ⟨by simp; omega, h.le, h.mono, fun t ht i hi p hp hN => ?_⟩
  obtain ⟨k, hk, hk', h1, h2, h3, h4⟩ := h.life t ht i hi p hp hN
  refine ⟨k, hk, by simp; omega, ?_⟩
  simp only [Array.getElem_push_lt hk']
  exact ⟨h1, h2, h3, h4⟩

theorem lifeOn_filling {N L ws appends joined filling} (h : LifeOn N L ws appends joined filling) (f' : Nat)
    (hf : filling ≤ f') : LifeOn N L ws appends joined f' :=
  { h with le := fun k hk => Nat.le_trans (h.le k hk) hf }

theorem lifeOn_add {N L ws appends joined filling} {below now : Nat} (h : LifeOn N L ws appends joined filling)
    (root : ByteArray) (plan : Plan) (post : Nat) (st : Int) (head : Option ByteArray)
    (hav : ∀ p, p < plan.length → N ≤ plan.block p → ∃ k, ∃ hk : k < joined.size, ∃ hk' : k < appends.size,
      appends[k].first ≤ plan.block p ∧ plan.block p < appends[k].first + appends[k].count ∧
      joined[k] ≤ filling ∧ filling < joined[k] + L) :
    LifeOn N L (ws.modify filling (addPass root plan post st head below now)) appends joined filling := by
  have get : ∀ t (ht : t < (ws.modify filling (addPass root plan post st head below now)).size),
      (ws.modify filling (addPass root plan post st head below now))[t] =
        if filling = t then addPass root plan post st head below now (ws[t]'(by simpa using ht)) else ws[t]'(by simpa using ht) :=
    fun t ht => Array.getElem_modify ht
  have hsz : (ws.modify filling (addPass root plan post st head below now)).size = ws.size := Array.size_modify
  refine ⟨h.size, h.le, h.mono, fun t ht i hi => ?_⟩
  revert hi
  rw [get t ht]
  split
  · rename_i he
    subst he
    intro hi
    simp only [addPass, Array.getElem_push]
    split
    · exact h.life filling (by omega) i (by assumption)
    · exact hav
  · intro hi; exact h.life t (by omega) i hi

theorem life_fail {s : State} (h : Life s) (t : Nat) (f : Failure) (at? : Option (Nat × Nat)) :
    Life (s.fail t f at?).2 := by
  unfold State.fail
  dsimp only
  split
  · exact lifeOn_same h (samePlans_modify _ _ _ fun _ => rfl)
  · exact h

theorem life_final {s : State} (h : Life s) (u : Nat) :
    Life { s with windows := s.windows.modify u ({· with final := true}) } :=
  lifeOn_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem life_boundaryNs {s : State} (h : Life s) (u now : Nat) :
    Life { s with windows := s.windows.modify u ({· with boundaryNs := some now}) } :=
  lifeOn_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem life_releaseCoin {s : State} (h : Life s) (t now : Nat) : Life (s.releaseCoin t now) :=
  lifeOn_push h _ rfl

theorem life_draw {s : State} (h : Life s) (u : Nat) (d : Array (Nat × Nat)) (now : Nat) :
    Life { s with windows := s.windows.modify u ({· with drawn := some d, openedNs := some now}) } :=
  lifeOn_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem life_accept {s : State} (h : Life s) (u : Nat) :
    Life { s with windows := s.windows.modify u ({· with verdict := some .accept}) } :=
  lifeOn_same h (samePlans_modify _ _ _ fun _ => rfl)

/-- `receive` gives each append not yet joined the window being filled. -/
theorem life_join {s : State} (h : Life s) :
    Life { s with joined := s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling } ∧
      (s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling).size = s.appends.size := by
  have hsz : s.joined.size ≤ s.appends.size := h.size
  have get : ∀ k (hk : k < (s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling).size),
      (s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling)[k] =
        if hl : k < s.joined.size then s.joined[k] else s.filling := by
    intro k hk
    rw [Array.getElem_append]
    split
    · rfl
    · simp
  have hsz' : (s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling).size = s.appends.size := by
    simp; omega
  refine ⟨⟨by show (s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling).size ≤ s.appends.size; omega,
    fun k hk => ?_, fun k₁ k₂ h₁ h₂ hle => ?_, fun t ht i hi p hp hN => ?_⟩, hsz'⟩
  · show _ ≤ s.filling
    rw [get k hk]
    split
    · exact h.le k (by assumption)
    · exact Nat.le_refl _
  · show _ ≤ _
    rw [get k₁ h₁, get k₂ h₂]
    split <;> split
    · exact h.mono k₁ k₂ (by assumption) (by assumption) hle
    · exact h.le k₁ (by assumption)
    · omega
    · exact Nat.le_refl _
  · obtain ⟨k, hk, hk', h1, h2, h3, h4⟩ := h.life t ht i hi p hp hN
    have hk₂ : k < (s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling).size := by
      simp; omega
    refine ⟨k, hk₂, hk', h1, h2, ?_⟩
    show _ ≤ t ∧ t < _ + s.params.lifetime
    rw [get k hk₂, dite_eq_left hk]
    exact ⟨h3, h4⟩

/-- A plan `receive` lets past the lifetime check reads, past `N`, only blocks of appends that joined within `L` windows
of the window being filled. -/
theorem life_check {s : State} (hi : Inv s) (hl : Life s) (hJ : s.joined.size = s.appends.size) {plan : Plan}
    (hP : EndsOk plan) (hR : ReadsDistinct plan s.extent)
    (hchk : ¬(decide (s.staleBelow s.filling > s.reg.blocks) &&
      plan.runs.any fun x => decide (x.1 < s.staleBelow s.filling) && decide (x.1 + x.2 > s.reg.blocks)) = true) :
    ∀ p, p < plan.length → s.reg.blocks ≤ plan.block p → ∃ k, ∃ hk : k < s.joined.size, ∃ hk' : k < s.appends.size,
      s.appends[k].first ≤ plan.block p ∧ plan.block p < s.appends[k].first + s.appends[k].count ∧
      s.joined[k] ≤ s.filling ∧ s.filling < s.joined[k] + s.params.lifetime := by
  intro p hp hN
  have hb := hR.1 p hp
  obtain ⟨r, hr, hr1, hr2⟩ := block_mem hP hp
  have hge : s.staleBelow s.filling ≤ plan.block p := by
    refine Nat.le_of_not_lt fun hlt => hchk ?_
    simp only [Bool.and_eq_true, decide_eq_true_eq, Array.any_eq_true']
    exact ⟨by omega, r, hr, by omega, by omega⟩
  obtain ⟨k, hk', hk1, hk2⟩ := appends_cover hi.extentSum hi.firsts hN hb
  have hk : k < s.joined.size := by omega
  refine ⟨k, hk, hk', hk1, hk2, hl.le k hk, ?_⟩
  unfold State.staleBelow at hge
  extract_lets k₀ at hge
  by_cases h₀ : k₀ < s.appends.size
  · rw [Array.getElem?_eq_getElem h₀] at hge
    dsimp only at hge
    have hle : k₀ ≤ k := by
      refine Nat.le_of_not_lt fun hlt => ?_
      have := appends_ordered hi.firsts hk' h₀ hlt
      omega
    have hJ₀ : k₀ < s.joined.toList.length := by simp; omega
    have hstop := takeWhile_stop _ _ k₀ rfl hJ₀
    simp only [Array.getElem_toList, decide_eq_false_iff_not, Int.not_le] at hstop
    have := hl.mono k₀ k (by simpa using hJ₀) hk hle
    omega
  · rw [Array.getElem?_eq_none (by omega)] at hge
    dsimp only at hge
    omega

/-! ## The events -/

theorem life_start {s : State} {now : Nat} {r : Result} {s' : State} (h : Life s)
    (he : start s now = (r, s')) : Life s' := by
  unfold start at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact life_releaseCoin h _ _

theorem life_boundary {s : State} {now : Nat} {r : Result} {s' : State} (h : Life s)
    (he : boundary s now = (r, s')) : Life s' := by
  unfold boundary at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact life_releaseCoin (life_boundaryNs h _ _) _ _
    | exact life_boundaryNs h _ _

theorem life_append {s : State} {root : ByteArray} {count : Int} {r : Result} {s' : State} (h : Life s)
    (he : append s root count = .ok (r, s')) : Life s' := by
  unfold append at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact life_fail h _ _ _
    | exact lifeOn_append h _

theorem life_release {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State} (h : Life s)
    (he : release s runs withdrawn = (r, s')) : Life s' := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact life_fail h _ _ _

theorem life_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State} (h : Life s)
    (he : close s t now = .ok (r, s')) : Life s' := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact life_fail (life_final h _) _ _ _
    | exact life_draw (life_final h _) _ _ _

theorem life_verify {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State}
    (h : Life s) (he : verify s t openings = (r, s')) : Life s' := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact life_fail h _ _ _
    | exact life_accept h _

theorem life_advance {s : State} (h : Life s) {t i : Int}
    (hc : (t == ↑s.filling + 1 && i == 0 && decide (t < ↑s.windows.size)) = true) :
    Life { s with windows := s.windows.modify s.filling ({· with final := true}), filling := t.toNat } :=
  lifeOn_filling (life_final h s.filling) _ (by show s.filling ≤ t.toNat; simp at hc; omega)

theorem life_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {step : Option Int} {now : Nat} {r : Result}
    {s' : State} (hi : Inv s) (h : Life s) (he : receive s t i root runs change step now = (r, s')) : Life s' := by
  unfold receive at he
  split at he
  · cases he; exact h
  split at he
  · cases he; exact h
  split at he
  · cases he; exact life_fail h _ _ _
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, Inv s₂ → Life s₂ → jp () s₂ = (r, s') → Life s' := by
    intro s₂ i₂ h₂ hj
    dsimp -zeta only [jp] at hj
    extract_lets t₂ N₂ ext₂ plan₂ s₃ below₂ head₂ jp₂ at hj
    have hW : ∀ plan, plan₂ = some plan → EndsOk plan ∧ ReadsDistinct plan s₂.extent := by
      intro plan hp
      dsimp only [plan₂, ext₂, N₂] at hp
      split at hp
      · cases hp
      · exact ⟨endsOk_changed hp, changedReadsDistinct _ _ _ _ _ hp⟩
      · cases hp; exact ⟨endsOk_whole _, readsDistinct_mono (wholeReads _) i₂.store⟩
      · exact ⟨(make_spec hp).1, planReadsDistinct _ _ _ hp⟩
    obtain ⟨h₃, hJ⟩ := life_join h₂
    have key₂ : jp₂ () = (r, s') → Life s' := by
      intro hj
      dsimp only [jp₂] at hj
      generalize hq : plan₂ = q at hj
      repeat' split at hj
      all_goals (try dsimp only at hj)
      all_goals cases hj
      all_goals first
        | exact life_fail h₂ _ _ _
        | exact life_fail h₃ _ _ _
        | exact lifeOn_add h₃ _ _ _ _ _
            (life_check (s := s₃) i₂ h₃ hJ (hW _ (by first | exact hq | exact hq.trans ‹_›)).1
              (hW _ (by first | exact hq | exact hq.trans ‹_›)).2 ‹_›)
    split at hj
    · split at hj
      · cases hj; exact life_fail h₂ _ _ _
      · exact key₂ hj
    · exact key₂ hj
  split at he
  · exact key s₁ (inv_advance hi ‹_›) (life_advance h ‹_›) he
  · split at he
    · cases he; exact life_fail h _ _ _
    · exact key s₀ hi h he

theorem step_life {s : State} {e : Event} {r : Result} {s' : State} (hi : Inv s) (h : Life s)
    (he : step s e = .ok (r, s')) : Life s' := by
  cases e with
  | start now => injection he with he; exact life_start h he
  | boundary now => injection he with he; exact life_boundary h he
  | append root count => exact life_append h he
  | release runs withdrawn => injection he with he; exact life_release h he
  | receive t i root runs change st now => injection he with he; exact life_receive hi h he
  | close t now => exact life_close h he
  | verify t openings => injection he with he; exact life_verify h he

/-! ## The statement -/

theorem lifetimeCapped : LifetimeCapped := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun
  have hboth := run_inv (fun s => Inv s ∧ Life s) (fun _ _ _ _ ⟨hs, hl⟩ he => ⟨(step_inv hs he).1, step_life hs hl he⟩)
    _ _ _ _ ⟨inv_init .., life_init ..⟩ hrun
  obtain ⟨hi, hl⟩ := hboth
  obtain ⟨-, hp, hr, -⟩ := inv_run hrun
  intro k hk
  have hNk : s.reg.blocks ≤ s.appends[k].first := by rw [hi.firsts k hk]; exact Nat.le_add_right _ _
  by_cases hj : k < s.joined.size
  · refine ⟨s.joined[k], fun t ht i hi' p hpl h1 h2 => ?_⟩
    obtain ⟨k', hk', hk'', h3, h4, h5, h6⟩ := hl.life t ht i hi' p hpl (by omega)
    obtain rfl := appends_unique hi.firsts hk'' hk ⟨h3, h4⟩ ⟨h1, h2⟩
    rw [← hp]
    exact ⟨h5, h6⟩
  · refine ⟨0, fun t ht i hi' p hpl h1 h2 => ?_⟩
    obtain ⟨k', hk', hk'', h3, h4, -, -⟩ := hl.life t ht i hi' p hpl (by omega)
    obtain rfl := appends_unique hi.firsts hk'' hk ⟨h3, h4⟩ ⟨h1, h2⟩
    exact absurd hk' hj

end Properties.PoubCheck
