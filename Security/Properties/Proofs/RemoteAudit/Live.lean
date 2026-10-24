import Batteries.Tactic.OpenPrivate
import Properties.RemoteAudit.Live
import Properties.Proofs.RemoteAudit.Order

open RemoteAudit RemoteAudit.Session

open private RemoteAudit.Session.live from RemoteAudit.Session

namespace Properties.RemoteAudit.Live

/-! ## The session on other coins -/

/-- The session `s` with the coins `c` and the append coins `a`. -/
abbrev W (c a : Array ByteArray) (s : State) : State := { s with coins := c, appendCoins := a }

theorem live_W (c a : Array ByteArray) (s : State) : RemoteAudit.Session.live (W c a s) = RemoteAudit.Session.live s :=
  rfl

theorem record_W (c a : Array ByteArray) (s : State) (t : Nat) : (W c a s).record t = s.record t := rfl
theorem deadline_W (c a : Array ByteArray) (s : State) (t : Nat) : (W c a s).deadline t = s.deadline t := rfl
theorem isReleased_W (c a : Array ByteArray) (s : State) (x y : Nat) : (W c a s).isReleased x y = s.isReleased x y :=
  rfl
theorem withdrawable_W (c a : Array ByteArray) (s : State) (pl : Plan) (w : Array (Int × Int)) :
    (W c a s).withdrawable pl w = s.withdrawable pl w := rfl
theorem awaiting_W (c a : Array ByteArray) (s : State) (t : Int) : (W c a s).awaiting t = s.awaiting t := rfl

theorem verify_W (s : State) (c a : Array ByteArray) (t : Int) (o : Array Opening) :
    verify (W c a s) t o = ((verify s t o).1, W c a (verify s t o).2) := by
  unfold verify State.fail
  simp only [Id.run, live_W, record_W]
  repeat' (split <;> try dsimp only)
  all_goals try rfl

theorem verifyBy_W (s : State) (c a : Array ByteArray) (t : Int) (f : Array (Nat × Nat) → Verdict) :
    verifyBy (W c a s) t f = ((verifyBy s t f).1, W c a (verifyBy s t f).2) := by
  unfold verifyBy State.fail
  simp only [Id.run, live_W]
  repeat' (split <;> try dsimp only)
  all_goals try rfl

theorem release_W (s : State) (c a : Array ByteArray) (runs withdrawn : Array (Int × Int)) :
    release (W c a s) runs withdrawn = ((release s runs withdrawn).1, W c a (release s runs withdrawn).2) := by
  unfold release State.fail
  simp only [Id.run, live_W, isReleased_W, withdrawable_W]
  repeat' (split <;> try dsimp only)
  all_goals try rfl

theorem start_W {s : State} {c a : Array ByteArray} {now : Nat} {r : Result} {s' : State}
    (he : start s now = (r, s')) (h : ∀ k, r = .released (some k) → c[k]! = s.coins[k]!) :
    start (W c a s) now = (r, W c a s') := by
  unfold start State.releaseCoin at he ⊢
  repeat' (split at he <;> try dsimp only at he)
  all_goals cases he
  all_goals simp only [*, ↓reduceIte, Bool.false_eq_true]

theorem boundary_W {s : State} {c a : Array ByteArray} {now : Nat} {r : Result} {s' : State}
    (he : boundary s now = (r, s')) (h : ∀ k, r = .released (some k) → c[k]! = s.coins[k]!) :
    boundary (W c a s) now = (r, W c a s') := by
  unfold boundary State.releaseCoin at he ⊢
  simp only [Id.run, live_W] at he ⊢
  repeat' (split at he <;> try dsimp only at he)
  all_goals cases he
  all_goals (try simp only [*, ↓reduceIte])
  all_goals first | rfl | (rw [h _ rfl]; rfl)

theorem append_W {s : State} {c a : Array ByteArray} {root : ByteArray} {count : Int} {r : Result} {s' : State}
    (he : append s root count = .ok (r, s'))
    (h : ∀ x, s.appendCoins[s.appendDrawn]? = some x → a[s.appendDrawn]? = some x) :
    append (W c a s) root count = .ok (r, W c a s') := by
  unfold append State.fail at he ⊢
  simp only [live_W] at he ⊢
  repeat' (split at he <;> try dsimp only at he)
  all_goals cases he
  all_goals (try simp only [*, ↓reduceIte, Bool.false_eq_true])
  all_goals (try rfl)
  all_goals (rw [h _ ‹_›]; rfl)

theorem close_W {s : State} {c a : Array ByteArray} {t : Int} {now : Nat} {r : Result} {s₁ : State}
    (h₁ : close s t now = .ok (r, s₁))
    (hc : ∀ d, r = .drew d → c[s.params.windows + t.toNat]! = s.coins[s.params.windows + t.toNat]!) :
    close (W c a s) t now = .ok (r, W c a s₁) := by
  generalize hres : close (W c a s) t now = res
  unfold close at h₁ hres
  simp only [live_W, deadline_W] at hres
  dsimp only at h₁ hres
  generalize s.deadline t.toNat = d at h₁ hres
  cases d <;> dsimp only at h₁ hres
  all_goals repeat' (split at h₁ <;> (try simp only [*, ↓reduceIte] at hres))
  all_goals (try (cases h₁ <;> cases hres <;> rfl))
  all_goals
    simp only [Bool.false_eq_true, ↓reduceIte] at hres
    cases h₁
    rw [hc _ rfl, ‹draws _ _ _ _ _ = some _›] at hres
    cases hres
    rfl

-- `receive` splits into many branches, each run twice (on `s` and on `W c a s`)
set_option maxHeartbeats 2000000 in
theorem receive_W {s : State} {c a : Array ByteArray} {t i : Int} {root : ByteArray}
    {runs : Option (Array (Int × Int))} {change : Option (Array (Int × Int) × Array (Int × Int))} {st : Option Int}
    {now : Nat} {r₁ r₂ : Result} {s₁ s₂ : State}
    (h₁ : receive s t i root runs change st now = (r₁, s₁))
    (h₂ : receive (W c a s) t i root runs change st now = (r₂, s₂)) : r₂ = r₁ ∧ s₂ = W c a s₁ := by
  unfold receive at h₁ h₂
  rw [live_W] at h₂
  dsimp -zeta only [W] at h₂
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
  have key : ∀ x : State, jp () x = (r₁, s₁) → jp' () (W c a x) = (r₂, s₂) → r₂ = r₁ ∧ s₂ = W c a s₁ := by
    intro x hx hx'
    dsimp only [jp] at hx
    dsimp only [jp', W] at hx'
    simp only [State.deadline, Properties.RemoteAudit.staleBelow_staleOf, State.isReleased] at hx hx'
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

/-- An event commutes with replacing the coins and the append coins, as long as the ones it reads agree: the challenge
it releases, the openings coin it draws from, and the next append coin. -/
theorem step_W {s : State} {e : Event} {r : Result} {s₁ : State} {c a : Array ByteArray}
    (he : Session.step s e = .ok (r, s₁))
    (hs : ∀ k, (∃ now, e = .start now ∨ e = .boundary now) → r = .released (some k) → c[k]! = s.coins[k]!)
    (hd : ∀ t now d, e = .close t now → r = .drew d →
      c[s.params.windows + t.toNat]! = s.coins[s.params.windows + t.toNat]!)
    (ha : ∀ x, s.appendCoins[s.appendDrawn]? = some x → a[s.appendDrawn]? = some x) :
    Session.step (W c a s) e = .ok (r, W c a s₁) := by
  cases e with
  | start now =>
    injection he with he; show pure _ = _; rw [start_W he (fun k => hs k ⟨now, Or.inl rfl⟩)]; rfl
  | boundary now =>
    injection he with he; show pure _ = _; rw [boundary_W he (fun k => hs k ⟨now, Or.inr rfl⟩)]; rfl
  | append root count => exact append_W he ha
  | release runs withdrawn => injection he with he; show pure _ = _; rw [release_W, he]; rfl
  | receive t i root runs change st now =>
    injection he with he
    show pure _ = _
    rcases h2 : receive (W c a s) t i root runs change st now with ⟨r₂, s₂⟩
    obtain ⟨rfl, rfl⟩ := receive_W he h2
    rfl
  | close t now => exact close_W he (hd t now · rfl)
  | verify t openings => injection he with he; show pure _ = _; rw [verify_W, he]; rfl

/-- No event changes the coins or the append coins. -/
theorem step_keeps {s : State} {e : Event} {r : Result} {s₁ : State} (he : Session.step s e = .ok (r, s₁)) :
    s₁.coins = s.coins ∧ s₁.appendCoins = s.appendCoins := by
  have h := step_W (c := s.coins) (a := s.appendCoins) he (fun _ _ _ => rfl) (fun _ _ _ _ _ => rfl) (fun _ h => h)
  change Session.step s e = _ at h
  rw [he] at h
  injection h with h
  injection h with _ h
  have h1 := congrArg State.coins h
  have h2 := congrArg State.appendCoins h
  exact ⟨h1, h2⟩

theorem close_params {s : State} {t : Int} {now : Nat} {r : Result} {s₁ : State}
    (he : close s t now = .ok (r, s₁)) : s₁.params = s.params := by
  unfold close State.fail at he
  dsimp only at he
  repeat' (split at he <;> try dsimp only at he)
  all_goals (cases he; try rfl)

end Properties.RemoteAudit.Live
