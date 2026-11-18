module

public import Proofs.Properties.Poub.Remote.Lifetime
public import Properties.Poub.Remote.Refresh
import Batteries.Tactic.OpenPrivate

/-!
# PoUB's verifier with refresh on: its properties, proved (`CoinAfterAppend`)

`step_appends`: an event either leaves the appends as they were and answers with no `appended` (`Quiet`), or is an
accepted append, which pushes one append at the store's extent with the next append coin and answers with that coin.
By induction over the events (`answered_fold`), every `appended` result is its event's append's, at its index among the
`appended` results.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open Poub Poub.Session

open private Poub.Session.live from Poub.Session

namespace Properties.Poub

/-! ## Each event -/

/-- An event that leaves the appends as they were and answers with no `appended`. -/
def Quiet (s s' : State) (r : Result) : Prop := s'.appends = s.appends ∧ ∀ first c, r ≠ .appended first c

theorem live_refused {s : State} {r : Result} (h : live s = some r) : ∀ first c, r ≠ .appended first c := by
  unfold live at h
  cases hf : s.failed with
  | none => rw [hf] at h; cases h
  | some f => rw [hf] at h; cases h; intro _ _ h; cases h

theorem quiet_start {s : State} {now : Nat} {r : Result} {s' : State} (he : start s now = (r, s')) : Quiet s s' r := by
  unfold start State.releaseCoin at he
  repeat' split at he
  all_goals cases he
  all_goals exact ⟨rfl, fun _ _ h => by cases h⟩

theorem quiet_boundary {s : State} {now : Nat} {r : Result} {s' : State} (he : boundary s now = (r, s')) :
    Quiet s s' r := by
  unfold boundary State.releaseCoin at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact ⟨rfl, live_refused ‹_›⟩
    | exact ⟨rfl, fun _ _ h => by cases h⟩

theorem quiet_release {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State}
    (he : release s runs withdrawn = (r, s')) : Quiet s s' r := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact ⟨rfl, live_refused ‹_›⟩
    | exact ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
    | exact ⟨rfl, fun _ _ h => by cases h⟩

theorem quiet_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State}
    (he : close s t now = .ok (r, s')) : Quiet s s' r := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact ⟨rfl, live_refused ‹_›⟩
    | exact ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
    | exact ⟨rfl, fun _ _ h => by cases h⟩

theorem quiet_verify {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State}
    (he : verify s t openings = (r, s')) : Quiet s s' r := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact ⟨rfl, live_refused ‹_›⟩
    | exact ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
    | exact ⟨rfl, fun _ _ h => by cases h⟩

theorem quiet_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {st : Option Int} {now : Nat} {r : Result}
    {s' : State} (he : receive s t i root runs change st now = (r, s')) : Quiet s s' r := by
  unfold receive at he
  split at he
  · cases he; exact ⟨rfl, live_refused ‹_›⟩
  split at he
  · cases he; exact ⟨rfl, fun _ _ h => by cases h⟩
  split at he
  · cases he; exact ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, s₂.appends = s.appends → jp () s₂ = (r, s') → Quiet s s' r := by
    intro s₂ e₂ hj
    dsimp only [jp] at hj
    repeat' split at hj
    all_goals (try dsimp only at hj)
    all_goals cases hj
    all_goals first
      | exact ⟨e₂, fun _ _ h => by cases h⟩
      | exact ⟨e₂, fun _ _ h => by simp [State.fail] at h⟩
  split at he
  · exact key s₁ rfl he
  · split at he
    · cases he; exact ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
    · exact key s₀ rfl he

/-- An accepted append pushes one append, at the store's extent, with the next append coin, and answers with it. -/
theorem append_step {s : State} {root : ByteArray} {count : Int} {origin : Origin} {r : Result} {s' : State}
    (he : append s root count origin = .ok (r, s')) :
    Quiet s s' r ∨ ∃ c, 1 ≤ count ∧ r = .appended s.extent c ∧ s.appendCoins[s.appendSampled]? = some c ∧
      s'.appends = s.appends.push { first := s.extent, count := count.toNat, root, coin := c, origin } := by
  unfold append at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact Or.inl ⟨rfl, live_refused ‹_›⟩
    | exact Or.inl ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
    | (refine Or.inr ⟨_, ?_, rfl, ‹_›, rfl⟩
       rcases Int.lt_or_ge count 1 with h | h
       · exfalso; simp_all
       · exact h)

theorem step_appends {s : State} {e : Event} {r : Result} {s' : State} (he : step s e = .ok (r, s')) :
    Quiet s s' r ∨ ∃ root count origin c, e = .append root count origin ∧ 1 ≤ count ∧ r = .appended s.extent c ∧
      s.appendCoins[s.appendSampled]? = some c ∧
      s'.appends = s.appends.push { first := s.extent, count := count.toNat, root, coin := c, origin } := by
  cases e with
  | start now => injection he with he; exact Or.inl (quiet_start he)
  | boundary now => injection he with he; exact Or.inl (quiet_boundary he)
  | append root count origin =>
    rcases append_step he with h | ⟨c, h1, h2, h3, h4⟩
    · exact Or.inl h
    · exact Or.inr ⟨root, count, origin, c, rfl, h1, h2, h3, h4⟩
  | release runs withdrawn => injection he with he; exact Or.inl (quiet_release he)
  | receive t i root runs change st now => injection he with he; exact Or.inl (quiet_receive he)
  | close t now => exact Or.inl (quiet_close he)
  | verify t openings => injection he with he; exact Or.inl (quiet_verify he)

/-! ## The run -/

theorem appendedOf_push (rs : Array Result) (r : Result) :
    appendedOf (rs.push r) = appendedOf rs ++ appendedOf #[r] := by
  unfold appendedOf
  rw [← Array.toList_inj]
  simp [List.filterMap_append]

theorem appendedOf_quiet {r : Result} (h : ∀ first c, r ≠ .appended first c) : appendedOf #[r] = #[] := by
  unfold appendedOf
  rw [← Array.toList_inj]
  cases r <;> simp_all

theorem appendedOf_one (first : Nat) (c : ByteArray) : appendedOf #[.appended first c] = #[(first, c)] := by
  unfold appendedOf
  rw [← Array.toList_inj]
  simp

theorem extract_push_le' (rs : Array Result) (r : Result) {e : Nat} (he : e ≤ rs.size) :
    (rs.push r).extract 0 e = rs.extract 0 e := by
  rw [Array.extract_push, ite_eq_left he]

/-- What `CoinAfterAppend` says of the results `rs` of the events `l`, reaching `s`, on the append coins `AC`. -/
def Answered (AC : Array ByteArray) (l : List Event) (rs : Array Result) (s : State) : Prop :=
  rs.size = l.length ∧ (appendedOf rs).size = s.appends.size ∧
  ∀ e first c, rs[e]? = some (.appended first c) →
    ∃ root count origin, l[e]? = some (.append root count origin) ∧
      ∃ hk : (appendedOf (rs.extract 0 e)).size < s.appends.size,
        s.appends[(appendedOf (rs.extract 0 e)).size].root = root ∧
        (s.appends[(appendedOf (rs.extract 0 e)).size].count : Int) = count ∧
        s.appends[(appendedOf (rs.extract 0 e)).size].origin = origin ∧
        s.appends[(appendedOf (rs.extract 0 e)).size].first = first ∧
        s.appends[(appendedOf (rs.extract 0 e)).size].coin = c ∧ AC[(appendedOf (rs.extract 0 e)).size]? = some c

theorem answered_step {AC : Array ByteArray} {l : List Event} {rs : Array Result} {s : State} {e : Event}
    {r : Result} {s' : State} (hi : Inv s) (hAC : s.appendCoins = AC) (h : Answered AC l rs s)
    (he : step s e = .ok (r, s')) : Answered AC (l ++ [e]) (rs.push r) s' := by
  obtain ⟨hsz, hcnt, hold⟩ := h
  have hlen : (rs.push r).size = (l ++ [e]).length := by simp [hsz]
  rcases step_appends he with ⟨ha, hq⟩ | ⟨root, count, origin, c, rfl, h1, rfl, hc, ha⟩
  · refine ⟨hlen, by rw [appendedOf_push, appendedOf_quiet hq, Array.append_empty, ha, hcnt], ?_⟩
    intro e' first c' he'
    rw [Array.getElem?_push] at he'
    split at he'
    · cases he'; exact absurd rfl (hq first c')
    · have hlt : e' < rs.size := by
        rcases Nat.lt_or_ge e' rs.size with hl | hl
        · exact hl
        · rw [Array.getElem?_eq_none (by omega)] at he'; cases he'
      obtain ⟨root, count, origin, hl, hk, h2⟩ := hold e' first c' he'
      rw [extract_push_le' rs r hlt.le, ha]
      exact ⟨root, count, origin, by rw [List.getElem?_append_left (by omega)]; exact hl, hk, h2⟩
  · have hk0 : (appendedOf rs).size = s.appends.size := hcnt
    refine ⟨hlen, by rw [appendedOf_push, appendedOf_one, ha]; simp [hcnt], ?_⟩
    intro e' first c' he'
    rw [Array.getElem?_push] at he'
    split at he'
    · rename_i hE
      subst hE
      cases he'
      rw [extract_push_le' rs _ (Nat.le_refl _), Array.extract_size, ha]
      have hsN : s.appends.size = s.appendSampled := hi.sampledN.1
      refine ⟨root, count, origin, by rw [List.getElem?_append_right (by omega)]; simp [hsz],
        by simp only [Array.size_push]; omega, ?_⟩
      simp only [hk0, Array.getElem_push_eq]
      refine ⟨rfl, by omega, rfl, rfl, rfl, ?_⟩
      rw [← hAC, hsN]; exact hc
    · have hlt : e' < rs.size := by
        rcases Nat.lt_or_ge e' rs.size with hl | hl
        · exact hl
        · rw [Array.getElem?_eq_none (by omega)] at he'; cases he'
      obtain ⟨root', count', origin', hl, hk, h2⟩ := hold e' first c' he'
      rw [extract_push_le' rs _ hlt.le, ha]
      refine ⟨root', count', origin', by rw [List.getElem?_append_left (by omega)]; exact hl,
        by simp only [Array.size_push]; omega, ?_⟩
      simp only [Array.getElem_push_lt hk]
      exact h2

theorem answered_run {AC : Array ByteArray} :
    ∀ (l pre : List Event) (acc : Array Result × State), Inv acc.2 → acc.2.appendCoins = AC →
      Answered AC pre acc.1 acc.2 →
      ∀ out, List.foldlM (fun (acc : Array Result × State) e => do
        let (r, s') ← step acc.2 e
        pure (acc.1.push r, s')) acc l = .ok out → Answered AC (pre ++ l) out.1 out.2 := by
  intro l
  induction l with
  | nil => intro pre acc _ _ h out ho; simp only [List.foldlM_nil] at ho; cases ho; simpa using h
  | cons e l ih =>
    intro pre acc hi hAC h out ho
    rw [List.foldlM_cons] at ho
    cases he : step acc.2 e with
    | error err => rw [he] at ho; cases ho
    | ok p =>
      rw [he] at ho
      obtain ⟨r, s''⟩ := p
      have ⟨hi', fp, -, fa⟩ := step_inv hi he
      have := ih (pre ++ [e]) (acc.1.push r, s'') hi' (fa.trans hAC) (answered_step hi hAC h he) out ho
      simpa using this

theorem coinAfterAppend : CoinAfterAppend := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun
  unfold run at hrun
  rw [← Array.foldlM_toList] at hrun
  have h := answered_run (AC := appendCoins) events.toList [] (#[], State.init params reg coins appendCoins notBeforeNs)
    (inv_init ..) rfl ⟨rfl, by simp [appendedOf, State.init], fun e _ _ h => by simp at h⟩ (rs, s) hrun
  obtain ⟨hsz, -, hold⟩ := h
  refine ⟨by simpa using hsz, fun e first c he => ?_⟩
  obtain ⟨root, count, origin, hl, hk, h2⟩ := hold e first c he
  exact ⟨root, count, origin, by simpa using hl, hk, h2⟩

end Properties.Poub
