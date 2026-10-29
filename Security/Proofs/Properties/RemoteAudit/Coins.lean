import Batteries.Tactic.OpenPrivate
import Properties.RemoteAudit.Coins
import Proofs.Properties.RemoteAudit.Draw
import Proofs.Properties.RemoteAudit.Session
import Proofs.Properties.RemoteAudit.Window

/-!
# When the verifier reads a window's openings coin (`AcceptDrawsOpenable`, `OpeningsUnreadUntilDrawn`)

`AcceptDrawsOpenable` is `acceptChecksOpenings` read through `drawIsSubset`: each drawn receipt is the sampler's value
at its index, and its opening passes `checkOpening`.

`OpeningsUnreadUntilDrawn`: every event commutes with replacing the session's coins (`C c s`), as long as the coins it
reads agree: `start` reads `coins[0]`, `boundary` window `t + 1`'s challenge (`t + 1 < W`), and `close t` the openings
coin `coins[W + t]`, only when it draws window `t` (`*_coins`, `step_coins`); the rest read none. No event changes
the coins (`step_coins_eq`). Each event keeps every window's draw, and only `close t` draws one, window `t`, with
every window before it drawn (`Adv`, `adv_step`): so the windows drawn are a prefix (`Prefix`), and a run that ends
with window `t` undrawn never drew a window from `t` on, so it read only coins below `W + t` (`fold_coins`).
-/

open RemoteAudit RemoteAudit.Session

open private RemoteAudit.Session.live from RemoteAudit.Session

namespace Properties.RemoteAudit

/-! ## The window -/

theorem acceptDrawsOpenable : AcceptDrawsOpenable := by
  intro reg appends w coin samples openings h
  obtain ⟨drawn, hd, hsz, hk⟩ := acceptChecksOpenings reg appends w coin samples openings h
  obtain ⟨L, us, hs, hsz', -, hrange, -⟩ := drawIsSubset _ _ _ _ _ _ hd
  refine ⟨L, us, hs, fun u hu => ?_⟩
  obtain ⟨k, hk', rfl⟩ := Array.mem_iff_getElem.1 hu
  have hkd : k < drawn.size := by omega
  obtain ⟨hi, hp, hidx⟩ := hrange k hkd
  rw [Array.size_map] at hi
  refine ⟨drawn[k].1, drawn[k].2, openings[k]!, hi, ?_, ?_, hk k hkd⟩
  · have e : (w.passes.map (·.plan.length))[drawn[k].1]! = w.passes[drawn[k].1]!.plan.length := by
      rw [getElem!_pos (w.passes.map (·.plan.length)) drawn[k].1 (by simpa using hi),
        getElem!_pos w.passes drawn[k].1 hi, Array.getElem_map]
    rw [e] at hp
    exact hp
  · rw [hidx, getElem!_pos us k hk']

/-! ## The session's coins -/

/-- The session `s` with the coins `c`. -/
abbrev C (c : Array ByteArray) (s : State) : State := { s with coins := c }

theorem live_C (c : Array ByteArray) (s : State) : RemoteAudit.Session.live (C c s) = RemoteAudit.Session.live s := rfl
theorem record_C (c : Array ByteArray) (s : State) (t : Nat) : (C c s).record t = s.record t := rfl
theorem deadline_C (c : Array ByteArray) (s : State) (t : Nat) : (C c s).deadline t = s.deadline t := rfl
theorem isReleased_C (c : Array ByteArray) (s : State) (a b : Nat) : (C c s).isReleased a b = s.isReleased a b := rfl
theorem staleBelow_C (c : Array ByteArray) (s : State) (t : Nat) : (C c s).staleBelow t = s.staleBelow t := rfl
theorem withdrawable_C (c : Array ByteArray) (s : State) (pl : Plan) (w : Array (Int × Int)) :
    (C c s).withdrawable pl w = s.withdrawable pl w := rfl


/-- `staleBelow` on the fields it reads, so a state's coins don't stand in the way of rewriting it. -/
def staleOf (params : Params) (appends : Array Append) (joined : Array Nat) (extent t : Nat) : Nat :=
  let k := (joined.toList.takeWhile fun (x : Nat) => (x : Int) ≤ (t : Int) - params.lifetime).length
  match appends[k]? with
  | some a => a.first
  | none => extent

theorem staleBelow_staleOf (s : State) (t : Nat) :
    s.staleBelow t = staleOf s.params s.appends s.joined s.extent t := by
  cases s; rfl

/-! ## Each event -/

theorem verify_coins (s : State) (c : Array ByteArray) (t : Int) (o : Array Opening) :
    verify (C c s) t o = ((verify s t o).1, C c (verify s t o).2) := by
  unfold verify State.fail
  simp only [Id.run, live_C, record_C]
  repeat' (split <;> try dsimp only)
  all_goals first | rfl | (simp only [*, ↓reduceIte]; rfl)

theorem release_coins (s : State) (c : Array ByteArray) (runs withdrawn : Array (Int × Int)) :
    release (C c s) runs withdrawn = ((release s runs withdrawn).1, C c (release s runs withdrawn).2) := by
  unfold release State.fail
  simp only [Id.run, live_C, isReleased_C, withdrawable_C]
  repeat' (split <;> try dsimp only)
  all_goals try rfl

theorem append_coins (s : State) (c : Array ByteArray) (root : ByteArray) (count : Int) :
    append (C c s) root count = (append s root count).map fun p => (p.1, C c p.2) := by
  unfold append State.fail
  simp only [live_C]
  repeat' (split <;> try dsimp only)
  all_goals try rfl

-- `receive` splits into many branches, each run twice (on `s` and on `C c s`)
set_option maxHeartbeats 2000000 in
theorem receive_coins {s : State} {c : Array ByteArray} {t i : Int} {root : ByteArray}
    {runs : Option (Array (Int × Int))} {change : Option (Array (Int × Int) × Array (Int × Int))} {st : Option Int}
    {now : Nat} {r₁ r₂ : Result} {s₁ s₂ : State}
    (h₁ : receive s t i root runs change st now = (r₁, s₁))
    (h₂ : receive (C c s) t i root runs change st now = (r₂, s₂)) : r₂ = r₁ ∧ s₂ = C c s₁ := by
  unfold receive at h₁ h₂
  rw [live_C] at h₂
  dsimp -zeta only [C] at h₂
  split at h₁ <;> split at h₂
  · cases h₁; cases h₂; simp_all
  · rename_i e₁ _ e₂; exact absurd e₁ (e₂ _)
  · rename_i e₁ _ _ e₂; exact absurd e₂ (e₁ _)
  split at h₁ <;> split at h₂
  · cases h₁; cases h₂; exact ⟨rfl, rfl⟩
  · contradiction
  · contradiction
  split at h₁ <;> split at h₂
  · cases h₁; cases h₂; exact ⟨rfl, rfl⟩
  · contradiction
  · contradiction
  change (have s := s; _) = _ at h₁
  extract_lets s₀ jp s₁' at h₁
  extract_lets s₀' jp' s₁'' at h₂
  have key : ∀ x : State, jp () x = (r₁, s₁) → jp' () (C c x) = (r₂, s₂) → r₂ = r₁ ∧ s₂ = C c s₁ := by
    intro x hx hx'
    dsimp only [jp] at hx
    dsimp only [jp', C] at hx'
    simp only [State.deadline, staleBelow_staleOf, State.isReleased] at hx hx'
    repeat' (split at hx <;> (try simp only [*] at hx'))
    all_goals (cases hx; cases hx'; exact ⟨rfl, rfl⟩)
  split at h₁ <;> split at h₂
  · exact key s₁' h₁ h₂
  · contradiction
  · contradiction
  · split at h₁ <;> split at h₂
    · cases h₁; cases h₂; exact ⟨rfl, rfl⟩
    · contradiction
    · contradiction
    · exact key s₀ h₁ h₂

theorem start_coins (s : State) (c : Array ByteArray) (now : Nat) (h0 : c[0]! = s.coins[0]!) :
    start (C c s) now = ((start s now).1, C c (start s now).2) := by
  unfold start State.releaseCoin
  repeat' (split <;> try dsimp only)
  all_goals try rfl
  all_goals simp only [h0]

theorem boundary_coins (s : State) (c : Array ByteArray) (now : Nat)
    (hc : ∀ k, k < s.params.windows → c[k]! = s.coins[k]!) :
    boundary (C c s) now = ((boundary s now).1, C c (boundary s now).2) := by
  unfold boundary State.releaseCoin
  simp only [Id.run, live_C]
  repeat' (split <;> try dsimp only)
  all_goals try rfl
  all_goals (rw [hc _ ‹_›]; rfl)

theorem close_coins {s : State} {c : Array ByteArray} {t : Int} {now : Nat} {r : Result} {s₁ : State}
    (h₁ : close s t now = .ok (r, s₁))
    (hc : ∀ t₀, (drawnOf s₁ t₀).isSome → c[s.params.windows + t₀]! = s.coins[s.params.windows + t₀]!) :
    close (C c s) t now = .ok (r, C c s₁) := by
  generalize hres : close (C c s) t now = res
  unfold close at h₁ hres
  simp only [live_C, deadline_C] at hres
  dsimp only at h₁ hres
  generalize s.deadline t.toNat = d at h₁ hres
  cases d <;> dsimp only at h₁ hres
  all_goals repeat' (split at h₁ <;> (try simp only [*, ↓reduceIte] at hres))
  all_goals (try (cases h₁ <;> cases hres <;> rfl))
  all_goals
    simp only [Bool.false_eq_true, ↓reduceIte] at hres
    have hb := ‹¬(decide (t < 0) || decide (t ≥ (s.windows.size : Int))) = true›
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hb
    have ht : t.toNat < s.windows.size := by omega
    cases h₁
    rw [hc t.toNat (by simp [drawnOf, ht, Array.getElem_modify]), ‹draws _ _ _ _ _ = some _›] at hres
    cases hres
    rfl

/-! ## Each window's draw -/

/-- Window `t`'s draws in `ws`. -/
def dr (ws : Array Win) (t : Nat) : Option (Array (Nat × Nat)) := ws[t]?.bind (·.drawn)

/-- Each window of `ws'` has the draw it had in `ws`, or is new and undrawn. -/

def Keep (ws ws' : Array Win) : Prop := ws.size ≤ ws'.size ∧ ∀ t, dr ws' t = dr ws t ∨ (ws.size ≤ t ∧ dr ws' t = none)

theorem keep_refl (ws : Array Win) : Keep ws ws := ⟨Nat.le_refl _, fun _ => Or.inl rfl⟩

theorem keep_trans {a b c : Array Win} (h₁ : Keep a b) (h₂ : Keep b c) : Keep a c := by
  obtain ⟨s1, k1⟩ := h₁
  obtain ⟨s2, k2⟩ := h₂
  refine ⟨by omega, fun t => ?_⟩
  rcases k2 t with e2 | ⟨l2, e2⟩ <;> rcases k1 t with e1 | ⟨l1, e1⟩
  · exact Or.inl (e2.trans e1)
  · exact Or.inr ⟨l1, e2.trans e1⟩
  · exact Or.inr ⟨by omega, e2⟩
  · exact Or.inr ⟨l1, e2⟩

theorem keep_modify (ws : Array Win) (u : Nat) (f : Win → Win) (hf : ∀ w, (f w).drawn = w.drawn) :
    Keep ws (ws.modify u f) := by
  refine ⟨by simp, fun t => Or.inl ?_⟩
  simp only [dr, Array.getElem?_modify]
  split
  · cases ws[t]? <;> simp [hf]
  · rfl

theorem keep_push (ws : Array Win) (w : Win) (hw : w.drawn = none) : Keep ws (ws.push w) := by
  refine ⟨by simp, fun t => ?_⟩
  simp only [dr, Array.getElem?_push]
  split
  · rename_i h; subst h; right; simp [hw]
  · left; rfl

theorem keep_fail (s : State) (t : Nat) (f : Failure) (a : Option (Nat × Nat)) : Keep s.windows (s.fail t f a).2.windows := by
  unfold State.fail
  dsimp only
  split
  · exact keep_modify _ _ _ (fun _ => rfl)
  · exact keep_refl _

theorem keep_start {s : State} {now : Nat} {r : Result} {s' : State} (he : start s now = (r, s')) :
    Keep s.windows s'.windows := by
  unfold start State.releaseCoin at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact keep_refl _
    | exact keep_fail _ _ _ _
    | (refine keep_push _ _ ?_; rfl)
    | (refine keep_modify _ _ _ ?_; intro; rfl)
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_push _ _ ?_) <;> (intros; rfl))
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_modify _ _ _ ?_) <;> (intros; rfl))
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_fail _ _ _ _); intro; rfl)

theorem keep_boundary {s : State} {now : Nat} {r : Result} {s' : State} (he : boundary s now = (r, s')) :
    Keep s.windows s'.windows := by
  unfold boundary State.releaseCoin at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact keep_refl _
    | exact keep_fail _ _ _ _
    | (refine keep_push _ _ ?_; rfl)
    | (refine keep_modify _ _ _ ?_; intro; rfl)
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_push _ _ ?_) <;> (intros; rfl))
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_modify _ _ _ ?_) <;> (intros; rfl))
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_fail _ _ _ _); intro; rfl)

theorem keep_append {s : State} {root : ByteArray} {count : Int} {r : Result} {s' : State}
    (he : append s root count = .ok (r, s')) : Keep s.windows s'.windows := by
  unfold append at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact keep_refl _
    | exact keep_fail _ _ _ _
    | (refine keep_push _ _ ?_; rfl)
    | (refine keep_modify _ _ _ ?_; intro; rfl)
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_push _ _ ?_) <;> (intros; rfl))
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_modify _ _ _ ?_) <;> (intros; rfl))
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_fail _ _ _ _); intro; rfl)

theorem keep_release {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State}
    (he : release s runs withdrawn = (r, s')) : Keep s.windows s'.windows := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact keep_refl _
    | exact keep_fail _ _ _ _
    | (refine keep_push _ _ ?_; rfl)
    | (refine keep_modify _ _ _ ?_; intro; rfl)
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_push _ _ ?_) <;> (intros; rfl))
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_modify _ _ _ ?_) <;> (intros; rfl))
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_fail _ _ _ _); intro; rfl)

theorem keep_verify {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State}
    (he : verify s t openings = (r, s')) : Keep s.windows s'.windows := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact keep_refl _
    | exact keep_fail _ _ _ _
    | (refine keep_push _ _ ?_; rfl)
    | (refine keep_modify _ _ _ ?_; intro; rfl)
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_push _ _ ?_) <;> (intros; rfl))
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_modify _ _ _ ?_) <;> (intros; rfl))
    | (refine keep_trans (keep_modify _ _ _ ?_) (keep_fail _ _ _ _); intro; rfl)

theorem keep_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {st : Option Int} {now : Nat} {r : Result}
    {s' : State} (he : receive s t i root runs change st now = (r, s')) : Keep s.windows s'.windows := by
  unfold receive at he
  split at he
  · cases he; exact keep_refl _
  split at he
  · cases he; exact keep_refl _
  split at he
  · cases he; exact keep_fail _ _ _ _
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, Keep s.windows s₂.windows → jp () s₂ = (r, s') → Keep s.windows s'.windows := by
    intro s₂ k₂ hj
    dsimp only [jp] at hj
    repeat' split at hj
    all_goals (try dsimp only at hj)
    all_goals cases hj
    all_goals first
      | exact keep_trans k₂ (keep_fail _ _ _ _)
      | (refine keep_trans k₂ (keep_modify _ _ _ ?_); intro; rfl)
  split at he
  · exact key s₁ (keep_modify _ _ _ (fun _ => rfl)) he
  · split at he
    · cases he; exact keep_fail _ _ _ _
    · exact key s₀ (keep_refl _) he

/-- Each window drawn in `ws` keeps its draw in `ws'`, and a window drawn in `ws'` but not in `ws` has every window
before it drawn in `ws`. -/
def Adv (ws ws' : Array Win) : Prop :=
  ws.size ≤ ws'.size ∧ (∀ t d, dr ws t = some d → dr ws' t = some d) ∧
    ∀ t, (dr ws' t).isSome → (dr ws t).isNone → ∀ j, j < t → (dr ws j).isSome

theorem dr_none {ws : Array Win} {t : Nat} (h : ws.size ≤ t) : dr ws t = none := by
  simp [dr, Array.getElem?_eq_none h]

theorem isSome_isNone {α : Type} {o : Option α} (h1 : o.isSome = true) (h2 : o.isNone = true) : False := by
  cases o <;> simp_all

theorem adv_of_keep {ws ws' : Array Win} (h : Keep ws ws') : Adv ws ws' := by
  obtain ⟨hs, k⟩ := h
  refine ⟨hs, fun t d hd => ?_, fun t h1 h2 => ?_⟩
  · rcases k t with e | ⟨l, -⟩
    · rw [e, hd]
    · rw [dr_none l] at hd; cases hd
  · rcases k t with e | ⟨-, e⟩
    · rw [e] at h1; exact (isSome_isNone h1 h2).elim
    · rw [e] at h1; cases h1

theorem adv_trans {a b c : Array Win} (h₁ : Adv a b) (h₂ : Keep b c) : Adv a c := by
  obtain ⟨s1, p1, n1⟩ := h₁
  obtain ⟨s2, k⟩ := h₂
  refine ⟨by omega, fun t d hd => ?_, fun t h1 h2 => ?_⟩
  · have := p1 t d hd
    rcases k t with e | ⟨l, -⟩
    · rw [e, this]
    · rw [dr_none (by omega)] at this; cases this
  · rcases k t with e | ⟨-, e⟩
    · rw [e] at h1; exact n1 t h1 h2
    · rw [e] at h1; cases h1

theorem dr_mm (ws : Array Win) (u : Nat) (f g : Win → Win) (d : Array (Nat × Nat)) (hg : ∀ w, (g w).drawn = some d)
    (t : Nat) : dr ((ws.modify u f).modify u g) t = if u = t ∧ t < ws.size then some d else dr ws t := by
  simp only [dr, Array.getElem?_modify]
  by_cases h : u = t
  · subst h
    simp only [true_and]
    by_cases hl : u < ws.size
    · simp [hg, hl]
    · simp [hl]
  · simp [h]

theorem adv_draw (ws : Array Win) (u : Nat) (f g : Win → Win) (d : Array (Nat × Nat))
    (hg : ∀ w, (g w).drawn = some d)
    (hu : ¬ws[u]!.drawn.isSome = true) (hord : ¬((ws.extract 0 u).any fun x => x.drawn.isNone) = true) :
    Adv ws ((ws.modify u f).modify u g) := by
  refine ⟨by simp, fun t d' hd => ?_, fun t h1 h2 => ?_⟩
  · rw [dr_mm _ _ _ _ _ hg]
    split
    · rename_i h; obtain ⟨rfl, hl⟩ := h
      simp only [dr, Array.getElem?_eq_getElem hl, Option.bind_some] at hd
      rw [getElem!_pos ws u hl, hd] at hu; exact absurd rfl hu
    · exact hd
  · rw [dr_mm _ _ _ _ _ hg] at h1
    split at h1
    · rename_i h; obtain ⟨rfl, hl⟩ := h
      intro j hj
      have hj' : j < (ws.extract 0 u).size := by simp; omega
      have hn := fun hh => hord (Array.any_eq_true.2 ⟨j, hj', hh⟩)
      simp only [Array.getElem_extract, Nat.zero_add] at hn
      simp only [dr, Array.getElem?_eq_getElem (by omega : j < ws.size), Option.bind_some]
      cases e : ws[j].drawn <;> simp_all
    · exact (isSome_isNone h1 h2).elim

theorem adv_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State}
    (he : close s t now = .ok (r, s')) : Adv s.windows s'.windows := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals (try (apply adv_of_keep; exact keep_refl _))
  all_goals (try (apply adv_of_keep; apply keep_trans _ (keep_fail _ _ _ _); refine keep_modify _ _ _ ?_; intro; rfl))
  all_goals
    show Adv s.windows ((s.windows.modify t.toNat _).modify t.toNat _)
    exact adv_draw _ _ _ _ _ (fun _ => rfl) ‹¬s.windows[t.toNat]!.drawn.isSome = true› ‹_›

theorem adv_step {s : State} {e : Event} {r : Result} {s' : State} (he : step s e = .ok (r, s')) :
    Adv s.windows s'.windows := by
  cases e with
  | start now => injection he with he; exact adv_of_keep (keep_start he)
  | boundary now => injection he with he; exact adv_of_keep (keep_boundary he)
  | append root count => exact adv_of_keep (keep_append he)
  | release runs withdrawn => injection he with he; exact adv_of_keep (keep_release he)
  | receive t i root runs change st now => injection he with he; exact adv_of_keep (keep_receive he)
  | close t now => exact adv_close he
  | verify t openings => injection he with he; exact adv_of_keep (keep_verify he)

/-! ## Runs -/

theorem step_coins {s : State} {e : Event} {r : Result} {s₁ : State} {c : Array ByteArray}
    (he : step s e = .ok (r, s₁)) (h0 : c[0]! = s.coins[0]!)
    (hW : ∀ k, k < s.params.windows → c[k]! = s.coins[k]!)
    (hd : ∀ t₀, (drawnOf s₁ t₀).isSome → c[s.params.windows + t₀]! = s.coins[s.params.windows + t₀]!) :
    step (C c s) e = .ok (r, C c s₁) := by
  cases e with
  | start now => injection he with he; show pure _ = _; rw [start_coins s c now h0, he]; rfl
  | boundary now => injection he with he; show pure _ = _; rw [boundary_coins s c now hW, he]; rfl
  | append root count =>
    show append (C c s) root count = _
    rw [append_coins, show append s root count = _ from he]; rfl
  | release runs withdrawn => injection he with he; show pure _ = _; rw [release_coins, he]; rfl
  | receive t i root runs change st now =>
    injection he with he
    show pure _ = _
    rcases h2 : receive (C c s) t i root runs change st now with ⟨r₂, s₂⟩
    obtain ⟨rfl, rfl⟩ := receive_coins he h2
    rfl
  | close t now => exact close_coins he hd
  | verify t openings => injection he with he; show pure _ = _; rw [verify_coins, he]; rfl

theorem step_coins_eq {s : State} {e : Event} {r : Result} {s₁ : State} (he : step s e = .ok (r, s₁)) :
    s₁.coins = s.coins := by
  have h := step_coins (c := s.coins) he rfl (fun _ _ => rfl) (fun _ _ => rfl)
  change step s e = _ at h
  rw [he] at h
  injection h with h
  injection h with _ h
  have := congrArg State.coins h
  exact this

theorem fold_inv (P : State → Prop) (hstep : ∀ s e r s', P s → step s e = .ok (r, s') → P s') :
    ∀ (l : List Event) (acc out : Array Result × State), P acc.2 →
      List.foldlM (fun (acc : Array Result × State) e => do
        let (r, s') ← step acc.2 e
        pure (acc.1.push r, s')) acc l = .ok out → P out.2 := by
  intro l
  induction l with
  | nil => intro acc out hacc h; simp only [List.foldlM_nil] at h; cases h; exact hacc
  | cons e l ih =>
    intro acc out hacc h
    rw [List.foldlM_cons] at h
    cases he : step acc.2 e with
    | error err => rw [he] at h; cases h
    | ok rs' =>
      rw [he] at h
      obtain ⟨r, s''⟩ := rs'
      exact ih _ out (hstep _ _ _ _ hacc he) h

def Prefix (ws : Array Win) : Prop := ∀ t j, j < t → (dr ws t).isSome → (dr ws j).isSome

theorem prefix_step {ws ws' : Array Win} (h : Prefix ws) (ha : Adv ws ws') : Prefix ws' := by
  obtain ⟨-, p, n⟩ := ha
  intro t j hj ht
  rcases hw : dr ws t with _ | d
  · obtain ⟨d', hd'⟩ := Option.isSome_iff_exists.1 (n t ht (by rw [hw]; rfl) j hj)
    rw [p j d' hd']; rfl
  · obtain ⟨d', hd'⟩ := Option.isSome_iff_exists.1 (h t j hj (by rw [hw]; rfl))
    rw [p j d' hd']; rfl

theorem fold_coins (T : Nat) (c : Array ByteArray) : ∀ (l : List Event) (acc out : Array Result × State),
    List.foldlM (fun (acc : Array Result × State) e => do
        let (r, s') ← step acc.2 e
        pure (acc.1.push r, s')) acc l = .ok out →
    Inv acc.2 → 0 < acc.2.params.windows →
    (∀ k, k < acc.2.params.windows + T → c[k]! = acc.2.coins[k]!) →
    (∀ t₀, (drawnOf out.2 t₀).isSome → t₀ < T) →
    List.foldlM (fun (acc : Array Result × State) e => do
        let (r, s') ← step acc.2 e
        pure (acc.1.push r, s')) (acc.1, C c acc.2) l = .ok (out.1, C c out.2) := by
  intro l
  induction l with
  | nil => intro acc out h _ _ _ _; simp only [List.foldlM_nil] at h ⊢; cases h; rfl
  | cons e l ih =>
    intro acc out h hinv h0 hag hT
    rw [List.foldlM_cons] at h ⊢
    cases he : step acc.2 e with
    | error err => rw [he] at h; cases h
    | ok rs' =>
      rw [he] at h
      obtain ⟨r, s'⟩ := rs'
      obtain ⟨hinv', hp, -, -⟩ := step_inv hinv he
      have hc := step_coins_eq he
      have hpers : ∀ t₀ d, drawnOf s' t₀ = some d → drawnOf out.2 t₀ = some d := fun t₀ d hd =>
        fold_inv (fun s'' => drawnOf s'' t₀ = some d)
          (fun _ _ _ _ hs he' => (adv_step he').2.1 t₀ d hs) l _ out hd h
      have hstep : step (C c acc.2) e = .ok (r, C c s') := by
        refine step_coins he (hag 0 (by omega)) (fun k hk => hag k (by omega)) (fun t₀ ht₀ => ?_)
        obtain ⟨d, hd⟩ := Option.isSome_iff_exists.1 ht₀
        have := hT t₀ (by rw [hpers t₀ d hd]; rfl)
        exact hag _ (by omega)
      rw [hstep]
      exact ih (acc.1.push r, s') out h hinv' (by rw [hp]; exact h0) (by rw [hp, hc]; exact hag) hT

theorem openingsUnreadUntilDrawn : OpeningsUnreadUntilDrawn := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun t coins' hW hnd hag
  have hpre : Prefix s.windows :=
    run_inv (fun s => Prefix s.windows) (fun _ _ _ _ hs he => prefix_step hs (adv_step he)) _ _ _ _
      (fun t j _ ht => by simp [dr, State.init] at ht) hrun
  have hlt : ∀ t₀, (drawnOf s t₀).isSome → t₀ < t := by
    intro t₀ h
    rcases Nat.lt_or_ge t₀ t with hl | hl
    · exact hl
    · rcases Nat.eq_or_lt_of_le hl with he | he
      · subst he; rw [hnd] at h; cases h
      · have := hpre t₀ t he h
        change (drawnOf s t).isSome = true at this
        rw [hnd] at this; cases this
  unfold run at hrun ⊢
  rw [← Array.foldlM_toList] at hrun ⊢
  exact fold_coins t coins' _ (#[], State.init params reg coins appendCoins notBeforeNs) (rs, s) hrun
    (inv_init ..) hW hag hlt

end Properties.RemoteAudit
