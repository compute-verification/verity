import Properties.RemoteAudit.P2Line
import Proofs.Properties.RemoteAudit.Session
import Proofs.Properties.RemoteAudit.Draw
import Proofs.Properties.RemoteAudit.P2Subprotocol
import Proofs.Accounting.RemoteHeld.Line
import Batteries.Tactic.OpenPrivate

/-!
# PoUB-P2's line of record: what its code keeps (`Properties.RemoteAudit.P2Line`)

What the line's argument reads of the P2 subprotocol's code, proved of it:

* **Every request keeps the session sound** (`p2_step`): the session's invariant (`Inv`), its drawn windows final and
  drawing `min(samples, U)` of their receipts (`DrawnSized`, which needs a pass a window, `minPasses_pos`), and the
  parameters, registration and append coins it was given (`Frame`). So every state a record's log reaches on the node
  is `Kept` (`kept_run`): the node's parameters, the held store standing in as its registration, and no append coin.
* **The check's replay is the record's run** (`run_of_replay`): the request read as the node's parameters, the log
  from the clock floor; and where the run takes every request, the state it reaches is its last (`reached_of_run`).
* **What an accepted record says** (`summary_facts`): one window for each of the node's, its registered blocks the
  store's units, and each window's passes those of the run's last state, below `RECEIPT_LIMIT` in all, drawing
  `min(s, U_t)` of them, each reading at most the store's units.
-/

open RemoteAudit RemoteAudit.Session

namespace Properties.RemoteAudit.P2Line.SecurityProofs

open Properties.RemoteAudit

/-- Every drawn window is final and drew `min(samples, U)` of its receipts. -/
def DrawnSized (s : State) : Prop :=
  ∀ t (ht : t < s.windows.size) d, s.windows[t].drawn = some d →
    s.windows[t].final = true ∧ d.size = min s.params.samples s.windows[t].receipts

theorem ds_modify {s : State} (h : DrawnSized s) (u : Nat) (f : Win → Win)
    (hf : ∀ w : Win, (f w).drawn = w.drawn ∧ (f w).final = w.final ∧ (f w).plans = w.plans) :
    DrawnSized { s with windows := s.windows.modify u f } := by
  intro t ht d hd
  simp only [Array.size_modify] at ht
  simp only [Array.getElem_modify] at hd ⊢
  split at hd
  · obtain ⟨e1, e2, e3⟩ := hf s.windows[t]
    rw [if_pos ‹_›, e2, Win.receipts, e3]
    rw [e1] at hd
    exact h t ht d hd
  · rw [if_neg ‹_›]
    exact h t ht d hd

theorem ds_fail {s : State} (h : DrawnSized s) (t : Nat) (f : Failure) (at? : Option (Nat × Nat)) :
    DrawnSized (s.fail t f at?).2 := by
  unfold State.fail
  dsimp only
  split
  · exact ds_modify h t _ fun _ => ⟨rfl, rfl, rfl⟩
  · exact h

theorem ds_releaseCoin {s : State} (h : DrawnSized s) (t now : Nat) : DrawnSized (s.releaseCoin t now) := by
  intro u hu d hd
  simp only [State.releaseCoin, Array.size_push] at hu hd ⊢
  rw [Array.getElem_push] at hd ⊢
  split at hd
  · rw [dif_pos ‹_›]; exact h u ‹_› d hd
  · cases hd

theorem ds_boundaryNs {s : State} (h : DrawnSized s) (u now : Nat) :
    DrawnSized { s with windows := s.windows.modify u ({· with boundaryNs := some now}) } :=
  ds_modify h u _ fun _ => ⟨rfl, rfl, rfl⟩

theorem ds_start {s : State} {now : Nat} {r : Result} {s' : State} (h : DrawnSized s) (he : start s now = (r, s')) :
    DrawnSized s' := by
  unfold start at he
  repeat' split at he
  all_goals cases he
  all_goals first | exact h | exact ds_releaseCoin h _ _

theorem ds_boundary {s : State} {now : Nat} {r : Result} {s' : State} (h : DrawnSized s)
    (he : boundary s now = (r, s')) : DrawnSized s' := by
  unfold boundary at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact ds_releaseCoin (ds_boundaryNs h _ _) _ _
    | exact ds_boundaryNs h _ _

theorem ds_final {s : State} (h : DrawnSized s) (u : Nat) :
    DrawnSized { s with windows := s.windows.modify u ({· with final := true}) } := by
  intro t ht d hd
  simp only [Array.size_modify] at ht
  simp only [Array.getElem_modify] at hd ⊢
  split at hd
  · rw [if_pos ‹_›]
    exact ⟨rfl, (h t ht d hd).2⟩
  · rw [if_neg ‹_›]
    exact h t ht d hd

theorem ds_draw {s : State} (h : DrawnSized s) (t : Nat) (d : Array (Nat × Nat)) (now : Nat)
    (hd : ∀ ht : t < s.windows.size, s.windows[t].final = true ∧ d.size = min s.params.samples s.windows[t].receipts) :
    DrawnSized { s with windows := s.windows.modify t ({· with drawn := some d, openedNs := some now}) } := by
  intro u hu e he
  simp only [Array.size_modify] at hu
  simp only [Array.getElem_modify] at he ⊢
  split at he
  · rename_i htu
    subst htu
    rw [if_pos rfl]
    cases he
    exact hd hu
  · rw [if_neg ‹_›]
    exact h u hu e he

theorem ds_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State} (h : DrawnSized s)
    (he : close s t now = .ok (r, s')) : DrawnSized s' := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact ds_fail (ds_final h _) _ _ _
    | skip
  all_goals
    rename_i hdr
    refine ds_draw (ds_final h _) _ _ _ fun ht => ?_
    simp only [Array.size_modify] at ht
    refine ⟨by rw [Array.getElem_modify_self], ?_⟩
    obtain ⟨L, us, -, hsz, hus, -⟩ := drawIsSubset _ _ _ _ _ _ hdr
    rw [hsz, hus, Array.getElem_modify_self, Win.receipts, getElem!_pos s.windows _ ht]

theorem ds_verifyBy {s : State} {t : Int} {check : Array (Nat × Nat) → Verdict} {r : Result} {s' : State}
    (h : DrawnSized s) (he : verifyBy s t check = (r, s')) : DrawnSized s' := by
  unfold verifyBy at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact ds_fail h _ _ _
    | exact ds_modify h _ _ fun _ => ⟨rfl, rfl, rfl⟩

theorem inv_verifyBy {s : State} {t : Int} {check : Array (Nat × Nat) → Verdict} {r : Result} {s' : State}
    (h : Inv s) (he : verifyBy s t check = (r, s')) : Inv s' ∧ Frame s s' := by
  unfold verifyBy at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact ⟨h, rfl, rfl, rfl⟩
    | exact ⟨inv_fail h _ _ _, rfl, rfl, rfl⟩
    | exact ⟨inv_accept h _ (drawn_of_get! _ _ _ (by assumption)), rfl, rfl, rfl⟩

theorem ds_untouched {s : State} (h : DrawnSized s) (u : Nat) (f : Win → Win) (hf : ∀ w, (f w).drawn = w.drawn)
    (hu : ∀ hu : u < s.windows.size, s.windows[u].drawn = none) :
    DrawnSized { s with windows := s.windows.modify u f } := by
  intro t ht d hd
  simp only [Array.size_modify] at ht
  simp only [Array.getElem_modify] at hd ⊢
  split at hd
  · rename_i htu
    subst htu
    rw [hf, hu ht] at hd
    cases hd
  · rw [if_neg ‹_›]
    exact h t ht d hd

/-- A root keeps drawn windows sized where the session takes a pass a window: the window it fills is never drawn, and
one it finalizes as it advances had at most its receipts drawn, none past them. -/
theorem ds_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {step : Option Int} {now : Nat} {r : Result}
    {s' : State} (hI : Inv s) (h : DrawnSized s) (hm : 1 ≤ s.params.minPasses)
    (he : receive s t i root runs change step now = (r, s')) : DrawnSized s' := by
  unfold receive at he
  split at he
  · cases he; exact h
  split at he
  · cases he; exact h
  split at he
  · cases he; exact ds_fail h _ _ _
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, DrawnSized s₂ → (∀ hf : s₂.filling < s₂.windows.size, s₂.windows[s₂.filling].drawn = none) →
      jp () s₂ = (r, s') → DrawnSized s' := by
    intro s₂ h₂ hf₂ hj
    dsimp -zeta only [jp] at hj
    extract_lets t₂ N₂ ext₂ plan₂ s₃ below₂ head₂ jp₂ at hj
    have key₂ : jp₂ () = (r, s') → DrawnSized s' := by
      intro hj
      dsimp only [jp₂] at hj
      generalize hq : plan₂ = q at hj
      repeat' split at hj
      all_goals cases hj
      all_goals first
        | exact ds_fail h₂ _ _ _
        | exact ds_untouched h₂ _ _ (fun _ => rfl) hf₂
    split at hj
    · split at hj
      · cases hj; exact ds_fail h₂ _ _ _
      · exact key₂ hj
    · exact key₂ hj
  split at he
  · rename_i hc
    refine key s₁ (ds_final h _) (fun hf => ?_) he
    simp only [s₁, s₀, Array.size_modify] at hf hc ⊢
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
    obtain ⟨⟨ht, -⟩, htl⟩ := hc
    have hne : s.filling ≠ t.toNat := by omega
    rw [Array.getElem_modify, if_neg hne]
    have h0 := hI.beyond t.toNat hf (by omega)
    cases hd : s.windows[t.toNat].drawn with
    | none => rfl
    | some d =>
      have := hI.few t.toNat hf (by rw [hd]; rfl)
      omega
  · split at he
    · cases he; exact ds_fail h _ _ _
    · rename_i hc₁ hc₂
      refine key s₀ h (fun hf => ?_) he
      simp only [s₀] at hf hc₂ ⊢
      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Bool.not_eq_true] at hc₂
      have hfin := hc₂.1.2
      rw [getElem!_pos s.windows _ hf] at hfin
      cases hd : s.windows[s.filling].drawn with
      | none => rfl
      | some d =>
        have := (h _ hf d hd).1
        rw [hfin] at this
        cases this

theorem withCoin_coins {g : Nat} {s : State} {e : Event} {c : ByteArray} {s₁ : State}
    (h : Verifier.withCoin g s e c = .ok s₁) (he : ∀ root n, e ≠ .append root n) :
    ∃ cs, s₁ = { s with coins := cs } := by
  unfold Verifier.withCoin at h
  simp only [bind, Except.bind] at h
  repeat' (split at h <;> try dsimp only at h)
  all_goals (try contradiction)
  all_goals first
    | (cases h; exact ⟨_, rfl⟩)
    | exact (he _ _ rfl).elim

/-- The P2 verifier's events (one window a coin, one coin a group) step the session from its state with some coins. -/
theorem eventOn_p2 {s : State} {e : Event} {coin : Option ByteArray} {r : Result} {s₂ : State}
    (h : Verifier.eventOn 2 1 s e coin = .ok (r, s₂)) (he : ∀ root n, e ≠ .append root n) :
    ∃ cs, Session.step { s with coins := cs } e = .ok (r, s₂) := by
  unfold Verifier.eventOn at h
  simp only [bind, Except.bind, Flight.step_two, Shared.step_one] at h
  repeat' (split at h <;> try dsimp only at h)
  all_goals (try contradiction)
  all_goals
    rename_i v₁ hw _ v hstep _ _
    obtain ⟨r', s'⟩ := v
    injection h with h
    injection h with h1 h2
    subst r s₂
  · obtain ⟨cs, rfl⟩ := withCoin_coins hw he
    exact ⟨cs, hstep⟩
  · injection hw with hw
    subst hw
    exact ⟨s.coins, hstep⟩

theorem fail_frame (s : State) (t : Nat) (f : Failure) (at? : Option (Nat × Nat)) : Frame s (s.fail t f at?).2 := by
  unfold State.fail
  dsimp only
  split <;> exact ⟨rfl, rfl, rfl⟩

theorem p2fail {s : State} (h : Inv s ∧ DrawnSized s) :
    (Inv (P2Session.fail s).2 ∧ DrawnSized (P2Session.fail s).2) ∧ Frame s (P2Session.fail s).2 := by
  unfold P2Session.fail
  split
  · exact ⟨⟨inv_fail h.1 _ _ _, ds_fail h.2 _ _ _⟩, fail_frame _ _ _ _⟩
  · exact ⟨h, rfl, rfl, rfl⟩

theorem ds_event {s : State} {e : Event} {r : Result} {s' : State} (h : DrawnSized s)
    (he : Session.step s e = .ok (r, s')) (hn : ∀ root n, e ≠ .append root n)
    (hr : ∀ runs withdrawn, e ≠ .release runs withdrawn) (hv : ∀ t o, e ≠ .verify t o)
    (hc : ∀ t i root runs change st now, e ≠ .receive t i root runs change st now) : DrawnSized s' := by
  cases e with
  | start now => injection he with he; exact ds_start h he
  | boundary now => injection he with he; exact ds_boundary h he
  | close t now => exact ds_close h he
  | append root n => exact (hn _ _ rfl).elim
  | release runs withdrawn => exact (hr _ _ rfl).elim
  | verify t o => exact (hv _ _ rfl).elim
  | receive t i root runs change st now => exact (hc _ _ _ _ _ _ _ rfl).elim

/-- Every request of a P2 log keeps the session's invariant and its drawn windows sized, with the parameters,
registration and append coins it was given. -/
theorem p2_step {held : P2Session.Held} {s s' : State} {op : P2Protocol.Op} (h : Inv s ∧ DrawnSized s)
    (hm : 1 ≤ s.params.minPasses) (he : P2Protocol.step held s op = .ok s') :
    (Inv s' ∧ DrawnSized s') ∧ Frame s s' := by
  cases op with
  | event e coin =>
    have key : ∀ (s₂ : State) (now : ℕ), (e = .start now ∨ e = .boundary now ∨ ∃ t, e = .close t now) →
        (do let (_, s) ← Verifier.eventOn 2 1 s e coin; pure s : Except String State) = .ok s₂ →
        (Inv s₂ ∧ DrawnSized s₂) ∧ Frame s s₂ := by
      intro s₂ now hev hx
      cases hq : Verifier.eventOn 2 1 s e coin with
      | error why => rw [hq] at hx; cases hx
      | ok v =>
        rw [hq] at hx
        obtain ⟨r, s₃⟩ := v
        cases hx
        have hn : ∀ root n, e ≠ .append root n := by
          intro root n hc; subst hc; rcases hev with h | h | ⟨_, h⟩ <;> cases h
        obtain ⟨cs, hst⟩ := eventOn_p2 hq hn
        have ⟨hI, hF⟩ := step_inv (s := { s with coins := cs }) h.1 hst
        refine ⟨⟨hI, ds_event (s := { s with coins := cs }) h.2 hst hn ?_ ?_ ?_⟩, hF⟩
        · intro _ _ hc; subst hc; rcases hev with h | h | ⟨_, h⟩ <;> cases h
        · intro _ _ hc; subst hc; rcases hev with h | h | ⟨_, h⟩ <;> cases h
        · intro _ _ _ _ _ _ _ hc; subst hc; rcases hev with h | h | ⟨_, h⟩ <;> cases h
    cases e with
    | start now => exact key s' now (Or.inl rfl) he
    | boundary now => exact key s' now (Or.inr (Or.inl rfl)) he
    | close t now => exact key s' now (Or.inr (Or.inr ⟨t, rfl⟩)) he
    | append => cases he
    | release => cases he
    | receive => cases he
    | verify => cases he
  | root t i now line =>
    injection he with he
    subst he
    unfold P2Session.receive
    split
    · exact p2fail h
    · generalize hx : Session.receive s _ _ _ _ _ _ _ = x
      obtain ⟨r, s₂⟩ := x
      have ⟨hI, hF⟩ := inv_receive h.1 hx
      exact ⟨⟨hI, ds_receive h.1 h.2 hm hx⟩, hF⟩
  | openings t line =>
    injection he with he
    subst he
    unfold P2Session.verify
    split
    · exact p2fail h
    · generalize hx : Session.verifyBy s _ _ = x
      obtain ⟨r, s₂⟩ := x
      have ⟨hI, hF⟩ := inv_verifyBy h.1 hx
      exact ⟨⟨hI, ds_verifyBy h.2 hx⟩, hF⟩
  | fail =>
    injection he with he
    subst he
    exact p2fail h

open private RemoteAudit.P2Session.positive from RemoteAudit.P2Session

theorem positive_pos {j : Lean.Json} {k : String} {n : Nat} (h : RemoteAudit.P2Session.positive j k = .ok n) :
    1 ≤ n := by
  unfold RemoteAudit.P2Session.positive at h
  simp only [bind, Except.bind] at h
  repeat' (split at h <;> try dsimp only at h)
  all_goals (try contradiction)
  rename_i hn
  injection h with h
  subst h
  omega

/-- A request that names a node's session has one or more passes a window. -/
theorem minPasses_pos {held : P2Session.Held} {r : Lean.Json} {params : Params}
    (h : P2Session.requestOf held r = .ok params) : 1 ≤ params.minPasses := by
  unfold P2Session.requestOf at h
  simp only [bind, Except.bind] at h
  repeat' (split at h <;> try dsimp only at h)
  all_goals (try contradiction)
  rename_i hp
  injection h with h
  subst h
  exact positive_pos hp

theorem forIn_foldlM {σ α : Type} (g : σ → α → Except String σ) (f : ℕ → σ → Except String (ForInStep σ))
    (get : ℕ → α) (hf : ∀ k r y, f k r = .ok y → ∃ x, y = .yield x ∧ g r (get k) = .ok x) :
    ∀ (l : List ℕ) (s s' : σ), forIn l s f = .ok s' → (l.map get).foldlM g s = .ok s' := by
  intro l
  induction l with
  | nil => intro s s' h; simpa using h
  | cons k l ih =>
    intro s s' h
    rw [List.forIn_cons] at h
    cases hk : f k s with
    | error e => rw [hk] at h; cases h
    | ok y =>
      obtain ⟨x, rfl, hx⟩ := hf k s y hk
      rw [hk] at h
      simp only [List.map_cons, List.foldlM_cons, hx]
      exact ih x s' h

/-- A replay that ran is the record's run: the request read as the node's parameters, the log from the clock floor. -/
theorem run_of_replay {p : P2Protocol.Params} {c : P2Protocol.Commitment} {e : P2Protocol.Evidence} {s : State}
    (h : P2Protocol.replay p c e = .ok s) :
    ∃ params, P2Session.requestOf c.held p.request = .ok params ∧
      Replay.run ⟨params, c.held⟩ ⟨p.notBeforeNs, p.request, e.log⟩ = .ok s := by
  unfold P2Protocol.replay at h
  cases hq : P2Session.requestOf c.held p.request with
  | error why => simp [hq, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  | ok params =>
    refine ⟨params, rfl, ?_⟩
    simp only [hq, bind, Except.bind, pure, Except.pure] at h
    split at h
    · cases h
    rename_i v hv
    injection h with h
    subst h
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hv
    have := forIn_foldlM (P2Protocol.step c.held) _ (fun k => e.log[k]!) (fun k r y hy => by
      split at hy
      · rename_i s' hs'
        cases hy
        exact ⟨s', rfl, hs'⟩
      · rename_i why hw
        simp [throw, throwThe, MonadExceptOf.throw] at hy) _ _ _ hv
    have hsz : [:e.log.size].size = e.log.size := by simp [Std.Legacy.Range.size]
    have hl : List.map (fun k => e.log[k]!) (List.range' 0 e.log.size 1) = e.log.toList := by
      apply List.ext_getElem
      · simp
      · intro i h1 h2
        simp only [List.getElem_map, List.getElem_range', Array.getElem_toList]
        simp at h1
        rw [getElem!_pos e.log _ (by omega)]
        simp
    rw [show [:e.log.size].start = 0 from rfl, show [:e.log.size].step = 1 from rfl, hsz, hl] at this
    unfold Replay.run Replay.init
    rw [← Array.foldlM_toList]
    exact this

/-- What every state a record's log reaches on the node keeps: the invariant, its drawn windows sized, the node's
parameters, the store standing in as its registration, and no append coin. -/
def Kept (node : Audited) (s : State) : Prop :=
  (Inv s ∧ DrawnSized s) ∧ s.params = node.params ∧ s.reg = P2Session.standIn node.held.reg ∧ s.appendCoins = #[]

theorem inv_p2init (held : P2Session.Held) (params : Params) (coins : Array ByteArray) (nb : Nat) :
    Inv (P2Session.init held params coins nb) := by
  have h := inv_init params (P2Session.standIn held.reg) coins #[] nb
  unfold Inv at h ⊢
  exact h

theorem kept_init (node : Audited) (r : Replay) : Kept node (r.init node) :=
  ⟨⟨inv_p2init _ _ _ _, fun t ht => by simp [Replay.init, P2Session.init, State.init] at ht⟩, rfl, rfl, rfl⟩

theorem kept_step {node : Audited} (hm : 1 ≤ node.params.minPasses) {s s' : State} {op : P2Protocol.Op}
    (h : Kept node s) (he : P2Protocol.step node.held s op = .ok s') : Kept node s' := by
  obtain ⟨hI, hp, hr, ha⟩ := h
  have ⟨hI', fp, fr, fa⟩ := p2_step hI (hp ▸ hm) he
  exact ⟨hI', fp.trans hp, fr.trans hr, fa.trans ha⟩

/-- Every state a record's log reaches on the node is `Kept`, where the node's parameters take a pass a window. -/
theorem kept_run {node : Audited} {r : Replay} {s : State} (hm : 1 ≤ node.params.minPasses) (h : r.run node = .ok s) :
    Kept node s := by
  unfold Replay.run at h
  rw [← Array.foldlM_toList] at h
  have key : ∀ (l : List P2Protocol.Op) (s₀ : State), Kept node s₀ →
      l.foldlM (P2Protocol.step node.held) s₀ = .ok s → Kept node s := by
    intro l
    induction l with
    | nil => intro s₀ h₀ h; cases h; exact h₀
    | cons op l ih =>
      intro s₀ h₀ h
      rw [List.foldlM_cons] at h
      cases hs : P2Protocol.step node.held s₀ op with
      | error => rw [hs] at h; cases h
      | ok s₁ =>
        rw [hs] at h
        exact ih s₁ (kept_step hm h₀ hs) h
  exact key _ _ (kept_init node r) h

theorem reach_of_fold (held : P2Session.Held) (l : List P2Protocol.Op) :
    ∀ (s s' : State), l.foldlM (P2Protocol.step held) s = .ok s' → l.foldlM (reachStep held) s = .ok s' := by
  induction l with
  | nil => intro s s' h; cases h; rfl
  | cons op l ih =>
    intro s s' h
    rw [List.foldlM_cons] at h ⊢
    cases hs : P2Protocol.step held s op with
    | error => rw [hs] at h; cases h
    | ok s₁ =>
      rw [hs] at h
      have : reachStep held s op = .ok s₁ := by simp only [reachStep, hs]
      rw [this]
      exact ih s₁ s' h

/-- Where the record's log runs every request, the state it reaches is its run's last. -/
theorem reached_of_run {node : Audited} {r : Replay} {s : State} (h : r.run node = .ok s) : r.reached node = s := by
  unfold Replay.run at h
  rw [← Array.foldlM_toList] at h
  unfold Replay.reached
  rw [← Array.foldlM_toList, reach_of_fold _ _ _ _ h]

theorem length_le_of_reads {pl : Plan} {B : Nat} (h : ReadsDistinct pl B) : pl.length ≤ B := by
  have := Fintype.card_le_of_injective (fun p : Fin pl.length => (⟨pl.block p, h.1 p p.2⟩ : Fin B))
    (fun a b hab => Fin.ext (h.2 a b a.2 b.2 (by simpa using congrArg Fin.val hab)))
  simpa using this

/-- Window `t` of a state's summary: its passes' lengths and its draws' count. -/
theorem summary_window {s : State} {sm : Summary} (h : s.summary = some sm) {t : Nat} (ht : t < s.windows.size) :
    (sm.windows[t]!).lengths = s.windows[t].plans.map (·.length) ∧
      (sm.windows[t]!).drawn = (s.windows[t].drawn.map (·.size)).getD 0 := by
  unfold State.summary at h
  split at h
  · cases h
  · simp only [Option.some.injEq] at h
    subst h
    rw [getElem!_pos _ t (by simpa using ht)]
    simp only [Array.getElem_map, Array.getElem_range]
    rw [getElem!_pos s.windows t ht]
    exact ⟨rfl, rfl⟩

/-- **The verifier's record of a session it accepts on the node**: the run's last state `s`, one window for each of the
node's, its registered blocks the store's units, and each window's passes those of `s`'s window, below
`RECEIPT_LIMIT` in all, drawing `min(s, U_t)` of them, each reading at most the store's units. -/
theorem summary_facts {node : Audited} {r : Replay} {sm : Summary} (hm : 1 ≤ node.params.minPasses)
    (hsm : r.summary node = some sm) :
    ∃ s, r.run node = .ok s ∧ s.outcome = some "accepted" ∧ s.summary = some sm ∧
      sm.windows.size = node.windows ∧ s.windows.size = node.windows ∧ sm.blocks = node.blocks ∧
      ∀ t (ht : t < s.windows.size),
        (sm.windows[t]!).lengths.toList.sum = s.windows[t].receipts ∧ s.windows[t].receipts < RECEIPT_LIMIT ∧
        (sm.windows[t]!).drawn = min node.params.samples s.windows[t].receipts ∧
        ∀ L ∈ (sm.windows[t]!).lengths, L ≤ node.blocks := by
  unfold Replay.summary at hsm
  split at hsm
  · rename_i s hrun
    obtain ⟨⟨hI, hD⟩, hp, hr, ha⟩ := kept_run hm hrun
    have hout := P2Subprotocol.outcome_of_summary hsm
    have hW : s.windows.size = node.windows := by
      rw [Line.SecurityProofs.windows_of_accepted hout, hp]; rfl
    obtain ⟨h1, h2⟩ := Line.SecurityProofs.summary_of hsm
    have hext : s.extent = s.reg.blocks := by
      have h0 : s.appends.size = 0 := by
        have := hI.drawnN
        rw [ha] at this
        simp at this
        omega
      rw [hI.extentSum, Array.eq_empty_of_size_eq_zero h0]
      simp
    have hacc : ∀ t (ht : t < s.windows.size), s.windows[t].verdict = some .accept := by
      intro t ht
      unfold State.outcome at hout
      split at hout
      · rename_i f _
        have := Option.some.inj hout
        cases hf : f.failure <;> simp [hf, _root_.RemoteAudit.Failure.name] at this
      · split at hout
        · rename_i hc
          simp only [Bool.and_eq_true, beq_iff_eq, Array.all_eq_true] at hc
          exact verdict_beq (hc.2 t ht)
        · cases hout
    refine ⟨s, hrun, hout, hsm, h1.trans hW, hW, by rw [h2, hr]; rfl, fun t ht => ?_⟩
    obtain ⟨hl, hd⟩ := summary_window hsm ht
    refine ⟨by rw [hl]; rfl, hI.receipts t ht, ?_, fun L hL => ?_⟩
    · obtain ⟨d, hd'⟩ := Option.isSome_iff_exists.1 (hI.drawn t ht (hacc t ht))
      rw [hd, hd', Option.map_some, Option.getD_some, (hD t ht d hd').2, hp]
    · rw [hl] at hL
      obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hL
      simp only [Array.size_map] at hi
      have := length_le_of_reads (hI.reads t ht i hi)
      rw [hext, hr] at this
      rw [Array.getElem_map]
      exact this
  · cases hsm

end Properties.RemoteAudit.P2Line.SecurityProofs
