import Batteries.Tactic.OpenPrivate
import Properties.RemoteAudit.Live
import Proofs.Properties.RemoteAudit.Order
import Proofs.Properties.RemoteAudit.SharedLink

/-!
# The verifier's process accepts what `Shared.run g` accepts (`liveIsRun`)

The recorded run is taken at `W c a s`, the process's session `s` with the coins `c` and the append coins `a`.

* Each event commutes with `W` (`step_W`, `sclose_W`, `sstep_W`) as long as the coins it reads agree: the challenge it
  releases, its draw's openings coin and the next append coin. No event changes the coins (`sstep_keeps`), and
  `withCoin` only fills a coin not yet sent or pushes an append coin (`withCoin_le`), so every coin the process holds
  stays, and `Agree c a` at the end holds at every request before it (`step_le`). The event's `unset` check says each
  coin it read is sent and of a coin's size (`unset_released`, `unset_drew`), so `Agree` gives it in `c` (`event_W`).
* A `verify`'s verdict is `checkWindowChunked`'s, which is `checkWindow`'s (`chunkedIsWhole`), at a point of the
  recorded run where the window awaits openings (`ChecksOK`). At its `settle`, later, the window still awaits them, and
  `checkWindow` gives the same verdict there (`verdict_kept`): its draws, record, registration and digest are kept
  (`Shared.recordKeptAfterDraw`, `srun_kept`), and the appends only grow past the blocks its draws read, which are
  receipts of its plans (`Shared.DrawsIn`, `Shared.checkWindow_grow`). So the process's `verifyBy` on that verdict is
  the recorded `verify` (`verifyBy_congr`).

`run_sim` puts them together by induction on the requests, from the end (`snoc_induction`), over `Shared.srunL g`.
-/

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
  all_goals first | rfl | (simp only [*, ↓reduceIte]; rfl)

theorem verifyBy_W (s : State) (c a : Array ByteArray) (t : Int) (f : Array (Nat × Nat) → Verdict) :
    verifyBy (W c a s) t f = ((verifyBy s t f).1, W c a (verifyBy s t f).2) := by
  unfold verifyBy State.fail
  simp only [Id.run, live_W]
  repeat' (split <;> try dsimp only)
  all_goals first | rfl | (simp only [*, ↓reduceIte]; rfl)

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

theorem withCoin_cases {g : Nat} {s : State} {e : Event} {c : ByteArray} {s₁ : State}
    (h : Verifier.withCoin g s e c = .ok s₁) :
    (∃ i x, s.coins[i]? = some x ∧ (x.isEmpty || x == c) = true ∧ s₁ = { s with coins := s.coins.set! i c }) ∨
      s₁ = { s with appendCoins := s.appendCoins.push c } := by
  unfold Verifier.withCoin at h
  simp only [bind, Except.bind] at h
  repeat' (split at h <;> try dsimp only at h)
  all_goals (try contradiction)
  all_goals first
    | (left; cases h; exact ⟨_, _, ‹_›, ‹_›, rfl⟩)
    | (right; cases h; rfl)

/-- What `s'` keeps of `s`'s coins: each of a coin's size, and each append coin. -/
def CoinsLE (s s' : State) : Prop :=
  (∀ (i : Nat) (x : ByteArray), s.coins[i]? = some x → x.size = Codec.COIN_BYTES → s'.coins[i]? = some x) ∧
    ∀ (k : Nat) (x : ByteArray), s.appendCoins[k]? = some x → s'.appendCoins[k]? = some x

theorem coinsLE_refl (s : State) : CoinsLE s s := ⟨fun _ _ h _ => h, fun _ _ h => h⟩

theorem coinsLE_trans {a b c : State} (h₁ : CoinsLE a b) (h₂ : CoinsLE b c) : CoinsLE a c :=
  ⟨fun i x h hs => h₂.1 i x (h₁.1 i x h hs) hs, fun k x h => h₂.2 k x (h₁.2 k x h)⟩

theorem coinsLE_of_eq {s s' : State} (h1 : s'.coins = s.coins) (h2 : s'.appendCoins = s.appendCoins) :
    CoinsLE s s' :=
  ⟨fun i x h _ => by rw [h1]; exact h, fun k x h => by rw [h2]; exact h⟩

theorem withCoin_W {g : Nat} {s : State} {e : Event} {c : ByteArray} {s₁ : State}
    (h : Verifier.withCoin g s e c = .ok s₁)
    (c' a' : Array ByteArray) : W c' a' s₁ = W c' a' s := by
  rcases withCoin_cases h with ⟨_, _, _, _, rfl⟩ | rfl <;> rfl

theorem withCoin_le {g : Nat} {s : State} {e : Event} {c : ByteArray} {s₁ : State}
    (h : Verifier.withCoin g s e c = .ok s₁) :
    CoinsLE s s₁ := by
  rcases withCoin_cases h with ⟨i, x, hx, hc, rfl⟩ | rfl
  · refine ⟨fun j y hy hs => ?_, fun k y hy => hy⟩
    show (s.coins.setIfInBounds i c)[j]? = some y
    rw [Array.getElem?_setIfInBounds]
    split
    · rename_i hij
      subst hij
      rw [hx] at hy
      cases hy
      have hne : x.isEmpty = false := by
        simp only [ByteArray.isEmpty, hs, Codec.COIN_BYTES]; decide
      rw [hne, Bool.false_or] at hc
      have hxc : x = c := eq_of_beq hc
      subst hxc
      have hi : i < s.coins.size := (Array.getElem?_eq_some_iff.1 hx).1
      simp [hi]
    · exact hy
  · refine ⟨fun j y hy _ => hy, fun k y hy => ?_⟩
    show (s.appendCoins.push c)[k]? = some y
    rw [Array.getElem?_push]
    split
    · rename_i hk; subst hk; simp at hy
    · exact hy

theorem sclose_W {g : Nat} {s : State} {c a : Array ByteArray} {t : Int} {now : Nat} {r : Result} {s₁ : State}
    (h₁ : Shared.close g s t now = .ok (r, s₁))
    (hc : (∃ d, r = .drew d) → c[s.params.windows + t.toNat / g]! = s.coins[s.params.windows + t.toNat / g]!) :
    Shared.close g (W c a s) t now = .ok (r, W c a s₁) := by
  generalize hres : Shared.close g (W c a s) t now = res
  unfold Shared.close at h₁ hres
  dsimp only [W] at hres
  dsimp only at h₁
  simp only [deadline_W] at hres
  split at h₁
  · rename_i f hf
    split at hres
    · rename_i f' hf'
      cases hf.symm.trans hf'
      cases h₁; cases hres; rfl
    · rename_i hf'; exact (hf' f hf).elim
  rename_i hf
  split at hres
  · rename_i f' hf'; exact (hf f' hf').elim
  split at h₁
  · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
  rename_i h; rw [ite_eq_right h] at hres
  split at h₁
  · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
  rename_i h; rw [ite_eq_right h] at hres
  split at h₁
  · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
  rename_i h; rw [ite_eq_right h] at hres
  split at h₁
  · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
  rename_i h; rw [ite_eq_right h] at hres
  cases hdl : s.deadline t.toNat <;> simp only [hdl] at h₁ hres
  all_goals
    split at h₁
    · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
    rename_i h; rw [ite_eq_right h] at hres
    split at h₁
    · rename_i u hu; rw [hu] at hres; cases h₁; cases hres; rfl
    rename_i hu
    rw [Shared.find_none' hu] at hres
    dsimp only at hres
    split at h₁
    · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
    rename_i h; rw [ite_eq_right h] at hres
    split at h₁
    · rename_i drawn hd
      cases h₁
      rw [hc ⟨_, rfl⟩, hd] at hres
      cases hres
      rfl
    · cases h₁

theorem sstep_nonclose {g : Nat} {s : State} {e : Event} (hn : ∀ t now, e ≠ .close t now) :
    Shared.step g s e = Session.step s e := by
  unfold Shared.step
  split
  · rfl
  · cases e <;> first | rfl | exact (hn _ _ rfl).elim

theorem sstep_close {g : Nat} (hg : 2 ≤ g) (s : State) (t : Int) (now : Nat) :
    Shared.step g s (.close t now) = Shared.close g s t now := by
  show (if g ≤ 1 then Session.step s (.close t now) else Shared.close g s t now) = _
  rw [ite_eq_right (by omega)]

theorem sstep_verify {g : Nat} (s : State) (t : Int) (o : Array Opening) :
    Shared.step g s (.verify t o) = Session.step s (.verify t o) :=
  sstep_nonclose fun _ _ h => by cases h

/-- `step_W` at `g` windows a draw: a close reads its draw's openings coin, `W + t / g`. -/
theorem sstep_W {g : Nat} (hg : 1 ≤ g) {s : State} {e : Event} {r : Result} {s₁ : State} {c a : Array ByteArray}
    (he : Shared.step g s e = .ok (r, s₁))
    (hs : ∀ k, (∃ now, e = .start now ∨ e = .boundary now) → r = .released (some k) → c[k]! = s.coins[k]!)
    (hd : ∀ t now d, e = .close t now → r = .drew d →
      c[s.params.windows + t.toNat / g]! = s.coins[s.params.windows + t.toNat / g]!)
    (ha : ∀ x, s.appendCoins[s.appendDrawn]? = some x → a[s.appendDrawn]? = some x) :
    Shared.step g (W c a s) e = .ok (r, W c a s₁) := by
  rcases Nat.lt_or_ge g 2 with hg2 | hg2
  · obtain rfl : g = 1 := by omega
    rw [RemoteAudit.Shared.step_one] at he ⊢
    exact step_W he hs (fun t now d he' hr => by simpa only [Nat.div_one] using hd t now d he' hr) ha
  · rcases Shared.sstep_of hg2 he with ⟨t, now, rfl, hc⟩ | ⟨hn, hs'⟩
    · rw [sstep_close hg2]
      exact sclose_W hc fun ⟨d, hr⟩ => hd t now d rfl hr
    · rw [sstep_nonclose hn]
      exact step_W hs' hs (fun t now d he' _ => (hn t now he').elim) ha

/-- No event changes the coins or the append coins. -/
theorem sstep_keeps {g : Nat} {s : State} {e : Event} {r : Result} {s₁ : State}
    (he : Shared.step g s e = .ok (r, s₁)) : s₁.coins = s.coins ∧ s₁.appendCoins = s.appendCoins := by
  rcases Nat.lt_or_ge g 2 with hg2 | hg2
  · rw [Shared.step_le_one (by omega)] at he
    exact step_keeps he
  · rcases Shared.sstep_of hg2 he with ⟨t, now, rfl, hc⟩ | ⟨-, hs⟩
    · rcases Shared.sclose_spec hc with ⟨-, rfl⟩ | ⟨u, f, -, rfl⟩ | ⟨d, -, -, -, -, -, -, -, -, rfl⟩ <;>
        exact ⟨rfl, rfl⟩
    · exact step_keeps hs

theorem sstep_close_params {g : Nat} {s : State} {t : Int} {now : Nat} {r : Result} {s₁ : State}
    (he : Shared.step g s (.close t now) = .ok (r, s₁)) : s₁.params = s.params := by
  rcases Nat.lt_or_ge g 2 with hg2 | hg2
  · rw [Shared.step_le_one (by omega)] at he
    exact close_params he
  · rw [sstep_close hg2] at he
    exact (Shared.sclose_dr he).2.2.2.1

theorem eventOn_ok {g : Nat} {s : State} {e : Event} {coin : Option ByteArray} {r : Result} {s₂ : State}
    (h : Verifier.eventOn 2 g s e coin = .ok (r, s₂)) :
    ∃ s₁, (∀ c a, W c a s₁ = W c a s) ∧ CoinsLE s s₁ ∧ Shared.step g s₁ e = .ok (r, s₂) ∧
      Verifier.unset g s₂ e (some r) = none := by
  unfold Verifier.eventOn at h
  simp only [bind, Except.bind, Flight.step_two] at h
  repeat' (split at h <;> try dsimp only at h)
  all_goals (try contradiction)
  all_goals
    rename_i v₁ hw _ v hstep _ hun
    obtain ⟨r', s'⟩ := v
    injection h with h
    injection h with h1 h2
    subst r s₂
    have hu : Verifier.unset g s' e (some r') = none := by
      cases hu : Verifier.unset g s' e (some r') with
      | none => rfl
      | some why => exact (hun why hu).elim
  · exact ⟨v₁, fun c a => withCoin_W hw c a, withCoin_le hw, hstep, hu⟩
  · injection hw with hw
    subst hw
    exact ⟨s, fun _ _ => rfl, coinsLE_refl s, hstep, hu⟩

theorem bang {xs : Array ByteArray} {k : Nat} {x : ByteArray} (h : xs[k]? = some x) : xs[k]! = x := by
  rw [getElem!_def, h]

theorem unset_released {g : Nat} {s : State} {e : Event} {k : Nat} (he : ∃ now, e = .start now ∨ e = .boundary now)
    (h : Verifier.unset g s e (some (.released (some k))) = none) :
    ∃ x, s.coins[k]? = some x ∧ x.size = Codec.COIN_BYTES := by
  obtain ⟨now, rfl | rfl⟩ := he <;>
  · unfold Verifier.unset at h
    cases hk : s.coins[k]? with
    | none => simp [hk] at h
    | some x => simp [hk] at h; exact ⟨x, rfl, h⟩

theorem unset_drew {g : Nat} {s : State} {t : Int} {now : Nat} {d : Array (Nat × Nat)}
    (h : Verifier.unset g s (.close t now) (some (.drew d)) = none) :
    ∃ x, s.coins[s.params.windows + t.toNat / g]? = some x ∧ x.size = Codec.COIN_BYTES := by
  unfold Verifier.unset at h
  cases hk : s.coins[s.params.windows + t.toNat / g]? with
  | none => simp [hk] at h
  | some x => simp [hk] at h; exact ⟨x, rfl, h⟩

/-! ## A window's verdict is kept -/

theorem awaiting_some {s : State} {t : Int} {w : WindowRecord} {d : Array (Nat × Nat)}
    (h : s.awaiting t = some (w, d)) :
    0 ≤ t ∧ t.toNat < s.windows.size ∧ s.windows[t.toNat]!.drawn = some d ∧ w = s.record t.toNat := by
  unfold State.awaiting at h
  simp only [Bool.or_eq_true, decide_eq_true_eq] at h
  split at h
  · cases h
  rename_i hc
  simp only [not_or] at hc
  obtain ⟨⟨-, h1⟩, h2⟩ := hc
  cases hd : s.windows[t.toNat]!.drawn with
  | none => simp [hd] at h
  | some d₀ =>
    simp only [hd] at h
    split at h
    · cases h
    · cases h
      exact ⟨by omega, by omega, rfl, rfl⟩

/-- What a run from `State.init` keeps from a point `sV` of it to a later one `s`: the invariant, each drawn window's
draws in its plans (`Shared.DrawsIn`), `Frz`, and the registration. -/
theorem srun_kept {g : Nat} (hg : 1 ≤ g) {params : Params} {reg : Registration} {c a : Array ByteArray} {nb : Nat}
    {es₀ es₂ : List Event} {rs₀ rs₂ : Array Result} {sV s : State} (hm : 1 ≤ params.minPasses)
    (h₀ : Shared.srunL g (State.init params reg c a nb) es₀ = .ok (rs₀, sV))
    (h₂ : Shared.srunL g sV es₂ = .ok (rs₂, s)) :
    Inv sV ∧ Shared.DrawsIn sV.windows ∧ Inv s ∧ Frz sV s ∧ s.reg = sV.reg := by
  rw [← Shared.srun_toArray] at h₀ h₂
  rcases Nat.lt_or_ge g 2 with hg2 | hg2
  · rw [Shared.run_le_one (by omega)] at h₀ h₂
    obtain ⟨hFV, -, -, -, -, -, -, hinV, -⟩ :=
      run_inv (Shared.Link1 params reg c) (fun _ _ _ _ hx he => Shared.link1_step hx he) _ _ _ _
        (Shared.link1_init params reg c a nb hm) h₀
    obtain ⟨hF, hf⟩ := run_frz _ _ _ _ hFV h₂
    have hr : s.reg = sV.reg := (run_inv (fun x => Inv x ∧ x.reg = sV.reg)
      (fun _ _ _ _ ⟨hx, e⟩ he => have ⟨hx', f⟩ := step_inv hx he; ⟨hx', f.2.1.trans e⟩) _ _ _ _ ⟨hFV.1, rfl⟩ h₂).2
    exact ⟨hFV.1, hinV, hF.1, hf, hr⟩
  · obtain ⟨hSF, hpre, -, -, -, -, hL, -⟩ :=
      Shared.srun_inv g (Shared.LinkInv g params reg c) (fun _ _ _ _ hx he => Shared.linkInv_step hg2 hx he)
        _ _ _ _ (Shared.linkInv_init g params reg c a nb hm) h₀
    obtain ⟨hS, hf⟩ := Shared.srun_frz hg2 _ _ _ _ hSF h₂
    have hr : s.reg = sV.reg := (Shared.srun_inv g (fun x => Inv x ∧ x.reg = sV.reg)
      (fun _ _ _ _ ⟨hx, e⟩ he => have ⟨hx', f, _⟩ := Shared.sstep hg2 hx he; ⟨hx', f.2.1.trans e⟩) _ _ _ _
        ⟨hSF.1.1, rfl⟩ h₂).2
    exact ⟨hSF.1.1, Shared.drawsIn_of_link hg hSF.1.1 hpre hSF.2 hL, hS.1.1, hf, hr⟩

/-- A window's verdict on its openings is the same at any later point of the session where it still awaits them: its
draws, its record, the registration and the digest are kept, and the appends only grow past the blocks it reads. -/
theorem verdict_kept {g : Nat} (hg : 1 ≤ g) {params : Params} {reg : Registration} {c a : Array ByteArray} {nb : Nat}
    {es₀ es₂ : List Event} {rs₀ rs₂ : Array Result} {sV s : State} (hm : 1 ≤ params.minPasses)
    (h₀ : Shared.srunL g (State.init params reg c a nb) es₀ = .ok (rs₀, sV))
    (h₂ : Shared.srunL g sV es₂ = .ok (rs₂, s))
    {t : Int} {w w' : WindowRecord} {d d' : Array (Nat × Nat)}
    (hv : sV.awaiting t = some (w, d)) (hs : s.awaiting t = some (w', d')) (o : Array Opening) :
    checkWindow s.reg s.digest s.appends w' d' o = checkWindow sV.reg sV.digest sV.appends w d o := by
  obtain ⟨-, ht, hd, rfl⟩ := awaiting_some hv
  obtain ⟨-, -, hd', rfl⟩ := awaiting_some hs
  obtain ⟨hIV, hinV, hI, hf, hr⟩ := srun_kept hg hm h₀ h₂
  have h₀' := h₀
  have h₂' := h₂
  rw [← Shared.srun_toArray] at h₀' h₂'
  have hdV : drawnOf sV t.toNat = some d := by
    show dr sV.windows t.toNat = some d
    rw [dr_eq]; exact hd
  obtain ⟨hds, hrec, -⟩ := Shared.recordKeptAfterDraw g params reg c a nb _ _ _ _ _ _ h₀' h₂' t.toNat d hm hdV
  have e : d' = d := by
    have : dr s.windows t.toNat = some d := hds
    rw [dr_eq, hd'] at this
    exact Option.some.inj this
  subst e
  obtain ⟨more, happ⟩ := hf.2.2
  rw [hrec, hr, hf.2.1, Shared.checkWindow_grow hIV hI hinV more happ hr ht hd o]

/-! ## The process's requests on the recorded run -/

/-- Coins `c` and append coins `a` agree with the ones `s` holds: each coin of a coin's size, and each append coin. -/
abbrev Agree (c a : Array ByteArray) (s : State) : Prop := CoinsLE s (W c a s)

theorem agree_of_le {c a : Array ByteArray} {s s' : State} (h : CoinsLE s s') (h' : Agree c a s') : Agree c a s :=
  coinsLE_trans h h'

/-- An event the process took on `s` is the same event on the recorded run, at `W c a s`, for coins that agree with
the ones the process holds after it. -/
theorem event_W {g : Nat} (hg : 1 ≤ g) {s : State} {e : Event} {coin : Option ByteArray} {r : Result} {s₂ : State}
    {c a : Array ByteArray} (h : Verifier.eventOn 2 g s e coin = .ok (r, s₂)) (hc : Agree c a s₂) :
    Shared.step g (W c a s) e = .ok (r, W c a s₂) ∧ CoinsLE s s₂ := by
  obtain ⟨s₁, hW, hle, hstep, hun⟩ := eventOn_ok h
  obtain ⟨h1, h2⟩ := sstep_keeps hstep
  refine ⟨?_, coinsLE_trans hle (coinsLE_of_eq h1 h2)⟩
  rw [← hW c a]
  refine sstep_W hg hstep (fun k he hr => ?_) (fun t now d he hr => ?_)
    (fun x hx => hc.2 _ x (by rw [h2]; exact hx))
  · subst hr
    obtain ⟨x, hx, hs⟩ := unset_released he hun
    rw [bang (hc.1 k x hx hs), ← h1, bang hx]
  · subst he hr
    obtain ⟨x, hx, hs⟩ := unset_drew hun
    rw [sstep_close_params hstep] at hx
    rw [bang (hc.1 _ x hx hs), ← h1, bang hx]

theorem verifyBy_keeps (s : State) (t : Int) (f : Array (Nat × Nat) → Verdict) :
    (verifyBy s t f).2.coins = s.coins ∧ (verifyBy s t f).2.appendCoins = s.appendCoins := by
  have h := verifyBy_W s s.coins s.appendCoins t f
  change verifyBy s t f = _ at h
  have h1 := congrArg (fun x : Result × State => x.2.coins) h
  have h2 := congrArg (fun x : Result × State => x.2.appendCoins) h
  exact ⟨h1, h2⟩

/-- `verifyBy` reads its check only on the draws of a window that awaits openings. -/
theorem verifyBy_congr {s : State} {t : Int} {f g : Array (Nat × Nat) → Verdict}
    (h : ∀ w d, s.awaiting t = some (w, d) → f d = g d) : verifyBy s t f = verifyBy s t g := by
  unfold verifyBy
  simp only [Id.run]
  split
  · rfl
  rename_i hl
  split
  · rfl
  rename_i hr
  split
  · rename_i d hd
    split
    · rfl
    rename_i hv
    split
    · rfl
    have hf : s.failed = none := by
      cases hs : s.failed with
      | none => rfl
      | some _ => exact (hl _ (by unfold RemoteAudit.Session.live; rw [hs]; rfl)).elim
    have ha : s.awaiting t = some (s.record t.toNat, d) := by
      unfold State.awaiting
      simp only [hf, Option.isSome_none, Bool.false_or]
      rw [ite_eq_right hr]
      simp only [hd, hv]
      rfl
    rw [h _ _ ha]
  · rfl

theorem srunL_snoc' {g : Nat} {s s₁ s₂ : State} {xs : List Event} {e : Event} {rs : Array Result} {r : Result}
    (h : Shared.srunL g s xs = .ok (rs, s₁)) (he : Shared.step g s₁ e = .ok (r, s₂)) :
    Shared.srunL g s (xs ++ [e]) = .ok (rs.push r, s₂) := by
  unfold Shared.srunL at h ⊢
  rw [List.foldlM_append, h]
  show List.foldlM (Shared.sAcc g) (rs, s₁) [e] = _
  rw [List.foldlM_cons, Shared.sAcc_ok (acc := (rs, s₁)) he]
  rfl

theorem snoc_induction {α : Type} {P : List α → Prop} (nil : P []) (snoc : ∀ l x, P l → P (l ++ [x])) (l : List α) :
    P l := by
  rw [← List.reverse_reverse l]
  induction l.reverse with
  | nil => exact nil
  | cons x l ih => rw [List.reverse_cons]; exact snoc _ _ ih

theorem foldlM_snoc {α β ε : Type} {f : β → α → Except ε β} {b : β} {l : List α} {x : α} {r : β}
    (h : (l ++ [x]).foldlM f b = .ok r) : ∃ m, l.foldlM f b = .ok m ∧ f m x = .ok r := by
  rw [List.foldlM_append] at h
  cases h₁ : l.foldlM f b with
  | error e => rw [h₁] at h; cases h
  | ok m =>
    refine ⟨m, rfl, ?_⟩
    rw [h₁] at h
    simp only [List.foldlM_cons, List.foldlM_nil, bind, Except.bind, pure, Except.pure] at h
    cases hf : f m x with
    | error e => simp [hf] at h
    | ok v => simp only [hf, Except.ok.injEq] at h; rw [h]

theorem run_nil (chunk g : Nat) (p : Proc) : run chunk g p [] = .ok ([], [], p) := rfl

theorem run_snoc {chunk g : Nat} {p p' : Proc} {qs : List Req} {q : Req} {answers : List Answer}
    {es : List Event} (h : run chunk g p (qs ++ [q]) = .ok (answers, es, p')) :
    ∃ ans₁ es₁ p₁ x e?, run chunk g p qs = .ok (ans₁, es₁, p₁) ∧ step chunk g p₁ q = .ok (x, e?, p') ∧
      answers = ans₁ ++ [x] ∧ es = es₁ ++ e?.toList := by
  unfold run at h ⊢
  obtain ⟨⟨ans₁, es₁, p₁⟩, h₁, h₂⟩ := foldlM_snoc h
  refine ⟨ans₁, es₁, p₁, ?_⟩
  simp only [bind, Except.bind] at h₂
  split at h₂
  · cases h₂
  rename_i v hs
  obtain ⟨x, e?, p₂⟩ := v
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h₂
  obtain ⟨rfl, rfl, rfl⟩ := h₂
  exact ⟨x, e?, h₁, hs, rfl, rfl⟩

/-- A request keeps the session's coins of a coin's size and its append coins. -/
theorem step_le {chunk g : Nat} {p p' : Proc} {q : Req} {x : Answer} {e? : Option Event}
    (h : step chunk g p q = .ok (x, e?, p')) : CoinsLE p.session p'.session := by
  cases q with
  | event e coin =>
    simp only [step, bind, Except.bind] at h
    split at h
    · cases h
    rename_i v hv
    obtain ⟨r, s⟩ := v
    cases h
    obtain ⟨s₁, -, hle, hstep, -⟩ := eventOn_ok hv
    obtain ⟨h1, h2⟩ := sstep_keeps hstep
    exact coinsLE_trans hle (coinsLE_of_eq h1 h2)
  | verify t id frames =>
    simp only [step] at h
    split at h
    · cases h
    split at h <;> (cases h; exact coinsLE_refl _)
  | settle id =>
    simp only [step] at h
    split at h
    · cases h
      obtain ⟨h1, h2⟩ := verifyBy_keeps p.session _ _
      exact coinsLE_of_eq h1 h2
    · cases h

/-- Each check the process holds was spawned where the recorded run, on `c` and `a`, had its window awaiting openings,
with the verdict `checkWindow` gives there. -/
def ChecksOK (g : Nat) (params : Params) (reg : Registration) (c a : Array ByteArray) (nb : Nat) (p : Proc)
    (es : List Event) : Prop :=
  ∀ ch ∈ p.checks, ∃ es₀ es₁ rs₀ sV, es = es₀ ++ es₁ ∧
    Shared.srunL g (State.init params reg c a nb) es₀ = .ok (rs₀, sV) ∧
    ∃ w d, sV.awaiting ch.t = some (w, d) ∧
      ch.verdict = checkWindow sV.reg sV.digest sV.appends w d (Wire.openingsOf ch.frames)

theorem checksOK_extend {g : Nat} {params : Params} {reg : Registration} {c a : Array ByteArray} {nb : Nat}
    {p p' : Proc} {es : List Event} (es' : List Event) (h : ChecksOK g params reg c a nb p es)
    (hp : ∀ ch ∈ p'.checks, ch ∈ p.checks) : ChecksOK g params reg c a nb p' (es ++ es') := by
  intro ch hch
  obtain ⟨es₀, es₁, rs₀, sV, rfl, h₀, hv⟩ := h ch (hp ch hch)
  exact ⟨es₀, es₁ ++ es', rs₀, sV, by rw [List.append_assoc], h₀, hv⟩

theorem results_snoc (ans : List Answer) (x : Answer) :
    ((ans ++ [x]).filterMap Answer.result?).toArray =
      (ans.filterMap Answer.result?).toArray ++ (x.result?.toList).toArray := by
  cases x <;> simp [Answer.result?]

/-- The process's run, on any coins that agree with the ones it holds at the end, is the recorded run: its results,
and its session but for the coins; and every check it holds is right. -/
theorem run_sim {chunk g : Nat} (hg : 1 ≤ g) {params : Params} {reg : Registration}
    {coins appendCoins : Array ByteArray} {nb : Nat} (hm : 1 ≤ params.minPasses) (qs : List Req) :
    ∀ {answers : List Answer} {es : List Event} {p : Proc},
      run chunk g { session := State.init params reg coins appendCoins nb } qs = .ok (answers, es, p) →
      ∀ {c a : Array ByteArray}, Agree c a p.session →
        Shared.srunL g (State.init params reg c a nb) es =
            .ok ((answers.filterMap Answer.result?).toArray, W c a p.session) ∧
          ChecksOK g params reg c a nb p es := by
  induction qs using snoc_induction with
  | nil =>
    intro answers es p h c a _
    rw [run_nil] at h
    cases h
    exact ⟨rfl, fun _ h => nomatch h⟩
  | snoc qs q ih =>
    intro answers es p' h c a hc
    obtain ⟨ans₁, es₁, p₁, x, e?, h₁, hs, rfl, rfl⟩ := run_snoc h
    obtain ⟨ihr, ihc⟩ := ih h₁ (agree_of_le (step_le hs) hc)
    cases q with
    | event e coin =>
      simp only [step, bind, Except.bind] at hs
      split at hs
      · cases hs
      rename_i v hv
      obtain ⟨r, s⟩ := v
      cases hs
      obtain ⟨hstep, -⟩ := event_W hg hv hc
      refine ⟨?_, checksOK_extend _ ihc fun _ h => h⟩
      rw [results_snoc]
      exact srunL_snoc' ihr hstep
    | verify t id frames =>
      simp only [step] at hs
      split at hs
      · cases hs
      split at hs
      · cases hs
        refine ⟨?_, checksOK_extend _ ihc fun _ h => h⟩
        rw [results_snoc]
        simpa [Answer.result?] using ihr
      · rename_i w d hw
        cases hs
        refine ⟨?_, ?_⟩
        · rw [results_snoc]
          simpa [Answer.result?] using ihr
        intro ch hch
        rcases List.mem_append.1 hch with hch | hch
        · obtain ⟨es₀, es₂, rs₀, sV, e, h₀, hv⟩ := ihc ch hch
          exact ⟨es₀, es₂, rs₀, sV, by simp [e], h₀, hv⟩
        · rw [List.mem_singleton] at hch
          subst hch
          refine ⟨es₁, [], _, W c a p₁.session, by simp, ihr, w, d, hw, ?_⟩
          exact chunkedIsWhole _ _ _ _ _ _ _
    | settle id =>
      simp only [step] at hs
      split at hs
      rotate_left
      · cases hs
      rename_i ch hch
      cases hs
      have hm' : ch ∈ p₁.checks := List.mem_of_find?_eq_some hch
      obtain ⟨es₀, es₂, rs₀, sV, e, h₀, wV, dV, hv, hverdict⟩ := ihc ch hm'
      have hstep : Shared.step g (W c a p₁.session) (.verify ch.t (Wire.openingsOf ch.frames)) =
          .ok ((verifyBy p₁.session ch.t fun _ => ch.verdict).1,
            W c a (verifyBy p₁.session ch.t fun _ => ch.verdict).2) := by
        rw [sstep_verify]
        show pure (verify _ _ _) = _
        rw [verify_eq_verifyBy, verifyBy_W]
        have hc : verifyBy p₁.session ch.t
            (checkWindow p₁.session.reg p₁.session.digest p₁.session.appends (p₁.session.record ch.t.toNat) ·
              (Wire.openingsOf ch.frames)) = verifyBy p₁.session ch.t fun _ => ch.verdict := by
          apply verifyBy_congr
          intro w d hw
          obtain ⟨-, -, -, hrec⟩ := awaiting_some hw
          rw [← hrec, hverdict]
          rw [e] at ihr
          obtain ⟨rs', s', rs₂, h₀', h₂⟩ := Shared.srunL_append ihr
          have hs' : (rs', s') = (rs₀, sV) := Shared.srunL_det h₀' h₀
          cases hs'
          exact verdict_kept hg hm h₀ h₂ hv (s := W c a p₁.session) hw _
        erw [hc]
        rfl
      refine ⟨?_, checksOK_extend _ ihc fun ch' h => (List.mem_filter.1 h).1⟩
      rw [results_snoc]
      exact srunL_snoc' ihr hstep

end Properties.RemoteAudit.Live

namespace Properties.RemoteAudit

open Live in
theorem liveIsRun : LiveIsRun := by
  intro chunk g params reg coins appendCoins nb qs answers es p h c a hg hm hc ha
  rw [Shared.srun_toArray]
  exact (run_sim hg hm qs h (c := c) (a := a) ⟨hc, ha⟩).1

end Properties.RemoteAudit
