import Batteries.Tactic.OpenPrivate
import Properties.RemoteAudit.Live
import Proofs.Properties.RemoteAudit.Order

/-!
# The verifier's process accepts what `Session.run` accepts (`liveIsRun`)

The recorded run is taken at `W c a s`, the process's session `s` with the coins `c` and the append coins `a`.

* Each event commutes with `W` (`step_W`) as long as the coins it reads agree: the challenge it releases, the openings
  coin it draws from and the next append coin. No event changes the coins (`step_keeps`), and `withCoin` only fills a
  coin not yet sent or pushes an append coin (`withCoin_le`), so every coin the process holds stays, and `Agree c a` at
  the end holds at every request before it (`step_le`). The event's `unset` check says each coin it read is sent and
  of a coin's size (`unset_released`, `unset_drew`), so `Agree` gives it in `c` (`event_W`).
* A `verify`'s verdict is `checkWindowChunked`'s, which is `checkWindow`'s (`chunkedIsWhole`), at a point of the
  recorded run where the window awaits openings (`ChecksOK`). At its `settle`, later, the window still awaits them, and
  `checkWindow` gives the same verdict there (`verdict_kept`): its draws, record, registration and digest are kept
  (`recordKeptAfterDraw`, `run_frz`), and the appends only grow past the blocks its draws read, which are receipts of
  its plans (`DrawsIn`). So the process's `verifyBy` on that verdict is the recorded `verify` (`verifyBy_congr`).

`run_sim` puts them together by induction on the requests, from the end (`snoc_induction`).
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

theorem eventOn_ok {s : State} {e : Event} {coin : Option ByteArray} {r : Result} {s₂ : State}
    (h : Verifier.eventOn 2 1 s e coin = .ok (r, s₂)) :
    ∃ s₁, (∀ c a, W c a s₁ = W c a s) ∧ CoinsLE s s₁ ∧ Session.step s₁ e = .ok (r, s₂) ∧
      Verifier.unset 1 s₂ e (some r) = none := by
  unfold Verifier.eventOn at h
  simp only [bind, Except.bind, Flight.step_two, Shared.step_one] at h
  repeat' (split at h <;> try dsimp only at h)
  all_goals (try contradiction)
  all_goals
    rename_i v₁ hw _ v hstep _ hun
    obtain ⟨r', s'⟩ := v
    injection h with h
    injection h with h1 h2
    subst r s₂
    have hu : Verifier.unset 1 s' e (some r') = none := by
      cases hu : Verifier.unset 1 s' e (some r') with
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

theorem unset_drew {s : State} {t : Int} {now : Nat} {d : Array (Nat × Nat)}
    (h : Verifier.unset 1 s (.close t now) (some (.drew d)) = none) :
    ∃ x, s.coins[s.params.windows + t.toNat]? = some x ∧ x.size = Codec.COIN_BYTES := by
  unfold Verifier.unset at h
  try simp only [Nat.div_one] at h
  cases hk : s.coins[s.params.windows + t.toNat]? with
  | none => simp [hk] at h
  | some x => simp [hk] at h; exact ⟨x, rfl, h⟩

/-! ## Each draw is a receipt of its window -/

/-- Each drawn window's draws are receipts of its passes' plans. -/
def DrawsIn (ws : Array Win) : Prop :=
  ∀ (u : Nat) (d : Array (Nat × Nat)), ws[u]!.drawn = some d → ∀ k (hk : k < d.size),
    d[k].1 < ws[u]!.plans.size ∧ d[k].2 < ws[u]!.plans[d[k].1]!.length

theorem drawsIn_modify {ws : Array Win} (u : Nat) (f : Win → Win) (hf : ∀ w, (f w).drawn = w.drawn)
    (hp : ∀ w, (f w).plans = w.plans) (h : DrawsIn ws) : DrawsIn (ws.modify u f) := by
  intro v d hd k hk
  rw [Properties.RemoteAudit.getElem!_modify'] at hd ⊢
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
  rw [Properties.RemoteAudit.getElem!_modify'] at hd' ⊢
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
  rw [Properties.RemoteAudit.getElem!_modify']
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
    obtain ⟨_, _, -, -, -, hrange, -⟩ := drawIsSubset _ _ _ _ _ _ hdr
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

theorem run_drawsIn {params : Params} {reg : Registration} {c a : Array ByteArray} {nb : Nat} {es : List Event}
    {rs : Array Result} {s : State} (hm : 1 ≤ params.minPasses)
    (h : runL (State.init params reg c a nb) es = .ok (rs, s)) : FInv s ∧ DrawsIn s.windows := by
  rw [← run_toArray] at h
  exact run_inv (fun s => FInv s ∧ DrawsIn s.windows)
    (fun _ _ _ _ ⟨hs, hd⟩ he => ⟨(finv_step hs he).1, drawsIn_step hs hd he⟩) _ _ _ _
    ⟨finv_init params reg c a nb hm, fun u d hd => by simp [State.init] at hd; cases hd⟩ h

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

/-- A window's verdict on its openings is the same at any later point of the session where it still awaits them: its
draws, its record, the registration and the digest are kept, and the appends only grow past the blocks it reads. -/
theorem verdict_kept {params : Params} {reg : Registration} {c a : Array ByteArray} {nb : Nat}
    {es₀ es₂ : List Event} {rs₀ rs₂ : Array Result} {sV s : State} (hm : 1 ≤ params.minPasses)
    (h₀ : runL (State.init params reg c a nb) es₀ = .ok (rs₀, sV)) (h₂ : runL sV es₂ = .ok (rs₂, s))
    {t : Int} {w w' : WindowRecord} {d d' : Array (Nat × Nat)}
    (hv : sV.awaiting t = some (w, d)) (hs : s.awaiting t = some (w', d')) (o : Array Opening) :
    checkWindow s.reg s.digest s.appends w' d' o = checkWindow sV.reg sV.digest sV.appends w d o := by
  obtain ⟨-, ht, hd, rfl⟩ := awaiting_some hv
  obtain ⟨-, -, hd', rfl⟩ := awaiting_some hs
  obtain ⟨hFV, hinV⟩ := run_drawsIn hm h₀
  have h₀' := h₀
  rw [← run_toArray] at h₀' h₂
  obtain ⟨hF, hk, hdig, more, happ⟩ := run_frz _ _ _ _ hFV h₂
  have hIV : Inv sV := hFV.1
  have hI : Inv s := hF.1
  have hr : s.reg = sV.reg := (run_inv (fun x => Inv x ∧ x.reg = sV.reg)
    (fun _ _ _ _ ⟨hx, e⟩ he => have ⟨hx', f⟩ := step_inv hx he; ⟨hx', f.2.1.trans e⟩) _ _ _ _ ⟨hIV, rfl⟩ h₂).2
  have hdV : drawnOf sV t.toNat = some d := by
    show dr sV.windows t.toNat = some d
    rw [dr_eq]; exact hd
  obtain ⟨hds, hrec, -⟩ := recordKeptAfterDraw params reg c a nb _ _ _ _ _ _ h₀' h₂ t.toNat d hm hdV
  have e : d' = d := by
    have : dr s.windows t.toNat = some d := hds
    rw [dr_eq, hd'] at this
    exact Option.some.inj this
  subst e
  rw [hrec, hr, hdig, happ, checkWindow_eq, checkWindow_eq]
  split
  · rfl
  congr 1
  apply findSome?_congr
  intro k hk
  rw [List.mem_range] at hk
  obtain ⟨hi, hp⟩ := draw_in_record hIV hinV ht hd hk
  unfold rejectAt
  rw [checkOpening_congr]
  apply appendOf_extend
  · rw [← happ]; exact firsts_mono hI
  · intro k' hk' hk''
    have := firsts_big hIV hI more happ hr k' hk' (by rw [happ]; exact hk'')
    have hb := record_reads hIV t.toNat ht _ _ hi hp
    rw [← happ]; omega

/-! ## The process's requests on the recorded run -/

/-- Coins `c` and append coins `a` agree with the ones `s` holds: each coin of a coin's size, and each append coin. -/
abbrev Agree (c a : Array ByteArray) (s : State) : Prop := CoinsLE s (W c a s)

theorem agree_of_le {c a : Array ByteArray} {s s' : State} (h : CoinsLE s s') (h' : Agree c a s') : Agree c a s :=
  coinsLE_trans h h'

/-- An event the process took on `s` is the same event on the recorded run, at `W c a s`, for coins that agree with
the ones the process holds after it. -/
theorem event_W {s : State} {e : Event} {coin : Option ByteArray} {r : Result} {s₂ : State} {c a : Array ByteArray}
    (h : Verifier.eventOn 2 1 s e coin = .ok (r, s₂)) (hc : Agree c a s₂) :
    Session.step (W c a s) e = .ok (r, W c a s₂) ∧ CoinsLE s s₂ := by
  obtain ⟨s₁, hW, hle, hstep, hun⟩ := eventOn_ok h
  obtain ⟨h1, h2⟩ := step_keeps hstep
  refine ⟨?_, coinsLE_trans hle (coinsLE_of_eq h1 h2)⟩
  rw [← hW c a]
  refine step_W hstep (fun k he hr => ?_) (fun t now d he hr => ?_) (fun x hx => hc.2 _ x (by rw [h2]; exact hx))
  · subst hr
    obtain ⟨x, hx, hs⟩ := unset_released he hun
    rw [bang (hc.1 k x hx hs), ← h1, bang hx]
  · subst he hr
    obtain ⟨x, hx, hs⟩ := unset_drew hun
    rw [close_params hstep] at hx
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

theorem runL_snoc' {s s₁ s₂ : State} {xs : List Event} {e : Event} {rs : Array Result} {r : Result}
    (h : runL s xs = .ok (rs, s₁)) (he : Session.step s₁ e = .ok (r, s₂)) :
    runL s (xs ++ [e]) = .ok (rs.push r, s₂) := by
  unfold runL at h ⊢
  rw [List.foldlM_append, h]
  show List.foldlM stepAcc (rs, s₁) [e] = _
  rw [List.foldlM_cons, stepAcc_ok he]
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

theorem run_nil (chunk : Nat) (p : Proc) : run chunk p [] = .ok ([], [], p) := rfl

theorem run_snoc {chunk : Nat} {p p' : Proc} {qs : List Req} {q : Req} {answers : List Answer}
    {es : List Event} (h : run chunk p (qs ++ [q]) = .ok (answers, es, p')) :
    ∃ ans₁ es₁ p₁ x e?, run chunk p qs = .ok (ans₁, es₁, p₁) ∧ step chunk p₁ q = .ok (x, e?, p') ∧
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
theorem step_le {chunk : Nat} {p p' : Proc} {q : Req} {x : Answer} {e? : Option Event}
    (h : step chunk p q = .ok (x, e?, p')) : CoinsLE p.session p'.session := by
  cases q with
  | event e coin =>
    simp only [step, bind, Except.bind] at h
    split at h
    · cases h
    rename_i v hv
    obtain ⟨r, s⟩ := v
    cases h
    obtain ⟨s₁, -, hle, hstep, -⟩ := eventOn_ok hv
    obtain ⟨h1, h2⟩ := step_keeps hstep
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
def ChecksOK (params : Params) (reg : Registration) (c a : Array ByteArray) (nb : Nat) (p : Proc) (es : List Event) :
    Prop :=
  ∀ ch ∈ p.checks, ∃ es₀ es₁ rs₀ sV, es = es₀ ++ es₁ ∧ runL (State.init params reg c a nb) es₀ = .ok (rs₀, sV) ∧
    ∃ w d, sV.awaiting ch.t = some (w, d) ∧
      ch.verdict = checkWindow sV.reg sV.digest sV.appends w d (Wire.openingsOf ch.frames)

theorem checksOK_extend {params : Params} {reg : Registration} {c a : Array ByteArray} {nb : Nat} {p p' : Proc}
    {es : List Event} (es' : List Event) (h : ChecksOK params reg c a nb p es) (hp : ∀ ch ∈ p'.checks, ch ∈ p.checks) :
    ChecksOK params reg c a nb p' (es ++ es') := by
  intro ch hch
  obtain ⟨es₀, es₁, rs₀, sV, rfl, h₀, hv⟩ := h ch (hp ch hch)
  exact ⟨es₀, es₁ ++ es', rs₀, sV, by rw [List.append_assoc], h₀, hv⟩

theorem results_snoc (ans : List Answer) (x : Answer) :
    ((ans ++ [x]).filterMap Answer.result?).toArray =
      (ans.filterMap Answer.result?).toArray ++ (x.result?.toList).toArray := by
  cases x <;> simp [Answer.result?]

/-- The process's run, on any coins that agree with the ones it holds at the end, is the recorded run: its results,
and its session but for the coins; and every check it holds is right. -/
theorem run_sim {chunk : Nat} {params : Params} {reg : Registration} {coins appendCoins : Array ByteArray} {nb : Nat}
    (hm : 1 ≤ params.minPasses) (qs : List Req) :
    ∀ {answers : List Answer} {es : List Event} {p : Proc},
      run chunk { session := State.init params reg coins appendCoins nb } qs = .ok (answers, es, p) →
      ∀ {c a : Array ByteArray}, Agree c a p.session →
        runL (State.init params reg c a nb) es = .ok ((answers.filterMap Answer.result?).toArray, W c a p.session) ∧
          ChecksOK params reg c a nb p es := by
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
      obtain ⟨hstep, -⟩ := event_W hv hc
      refine ⟨?_, checksOK_extend _ ihc fun _ h => h⟩
      rw [results_snoc]
      exact runL_snoc' ihr hstep
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
      have hstep : Session.step (W c a p₁.session) (.verify ch.t (Wire.openingsOf ch.frames)) =
          .ok ((verifyBy p₁.session ch.t fun _ => ch.verdict).1,
            W c a (verifyBy p₁.session ch.t fun _ => ch.verdict).2) := by
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
          obtain ⟨rs', s', rs₂, h₀', h₂⟩ := runL_append ihr
          have hs' : (rs', s') = (rs₀, sV) := runL_det h₀' h₀
          cases hs'
          exact verdict_kept hm h₀ h₂ hv (s := W c a p₁.session) hw _
        erw [hc]
        rfl
      refine ⟨?_, checksOK_extend _ ihc fun ch' h => (List.mem_filter.1 h).1⟩
      rw [results_snoc]
      exact runL_snoc' ihr hstep

end Properties.RemoteAudit.Live

namespace Properties.RemoteAudit

open Live in
theorem liveIsRun : LiveIsRun := by
  intro chunk params reg coins appendCoins nb qs answers es p h c a hm hc ha
  rw [run_toArray]
  exact (run_sim hm qs h (c := c) (a := a) ⟨hc, ha⟩).1

end Properties.RemoteAudit
