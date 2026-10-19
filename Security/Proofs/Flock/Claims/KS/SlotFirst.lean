import Proofs.Flock.Claims.KS.SlotLead

/-!
# A slot's coin count past a game the strategy plays to its end first

A batch puts a position's lockstep beside the same games on both sides (`rightSame`, `batchLock`). Where the strategy
plays the game in front to its end first, after exactly `n` coins, or plays a part of the run whose every reached
outcome is `Dead` (`First`), a slot of the lockstep met after `j` coins once that game has ended is met after `n + j`:

* `First`: the strategy plays the first game of `interleave h g'` to its end before the second;
* `NoSlot.map`, `noSlot_rightSame_ret`: relabelling keeps no table node of the slot, and the same game beside a
  finished lockstep has none;
* `SlotAt.rightSame_first`: so a slot met after `j` coins once `h` has ended is met after `n + j` coins;
* `PreFirst`, `SlotAt.batchLock_first`: in a batch, the games in front of a position played to their ends first, in
  order, after `n` coins in all, put a slot of that position's lockstep met after `j` coins once they have ended at
  `n + j` coins.
-/

namespace FlockSoundness.Game.Lock

/-- **The strategy plays `h` to its end first**: at every choice of `interleave h g'` it advances `h`, which ends after
exactly `n` coins with `Q` at its outcome and the strategy after it (`done`), unless `g'` has ended (`last`), except
below a part of the run every reached outcome of which is `Dead` (`dead`). -/
inductive First {γ β : Type} (Dead : γ × β → Prop)
    (Q : (c : γ) → (g' : Game β) → Strategy (interleave (.ret c) g') → Prop) :
    ℕ → (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Prop
  | dead {n : ℕ} {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} :
      (∀ x, Reach _ σ x → Dead x) → First Dead Q n h g' σ
  | done {c : γ} {g' : Game β} {σ : Strategy (interleave (.ret c) g')} : Q c g' σ → First Dead Q 0 (.ret c) g' σ
  | last {n : ℕ} {h : Game γ} {b : β} {σ : Strategy (interleave h (.ret b))} : First Dead Q n h (.ret b) σ
  | coinCoin {n : ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game γ} {C₂ : Type} [Fintype C₂] [Nonempty C₂]
      {k₂ : C₂ → Game β} {τ : (c : C) → Strategy (interleave (k c) (.coin C₂ k₂))} :
      (∀ c, First Dead Q n (k c) (.coin C₂ k₂) (τ c)) → First Dead Q (n + 1) (.coin C k) (.coin C₂ k₂) ⟨true, τ⟩
  | coinSend {n : ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game γ} {M₂ : Type} {k₂ : M₂ → Game β}
      {τ : (c : C) → Strategy (interleave (k c) (.send M₂ k₂))} :
      (∀ c, First Dead Q n (k c) (.send M₂ k₂) (τ c)) → First Dead Q (n + 1) (.coin C k) (.send M₂ k₂) ⟨true, τ⟩
  | sendCoin {n : ℕ} {M : Type} {k : M → Game γ} {C₂ : Type} [Fintype C₂] [Nonempty C₂] {k₂ : C₂ → Game β} (m : M)
      {τ : Strategy (interleave (k m) (.coin C₂ k₂))} :
      First Dead Q n (k m) (.coin C₂ k₂) τ → First Dead Q n (.send M k) (.coin C₂ k₂) ⟨true, ⟨m, τ⟩⟩
  | sendSend {n : ℕ} {M : Type} {k : M → Game γ} {M₂ : Type} {k₂ : M₂ → Game β} (m : M)
      {τ : Strategy (interleave (k m) (.send M₂ k₂))} :
      First Dead Q n (k m) (.send M₂ k₂) τ → First Dead Q n (.send M k) (.send M₂ k₂) ⟨true, ⟨m, τ⟩⟩

section First

variable {T D I α β γ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}

theorem NoSlot.map {α₂ β₂ : Type} {R₂ : I → α₂ → β₂ → Prop} {slot : ℕ} (f : α → α₂) (f' : β → β₂)
    (h : ∀ i a b, R i a b → R₂ i (f a) (f' b)) {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'}
    (hL : NoSlot slot L) : NoSlot slot (Lock.map f f' h L) := by
  unfold Lock.map
  exact hL.bind fun i a b hr => .ret (h i a b hr)

/-- **The same game beside a finished lockstep has no table node.** -/
theorem noSlot_rightSame_ret {slot : ℕ} {i : I} {a : α} {b : β} (hr : R i a b) :
    ∀ h : Game γ, NoSlot slot (rightSame h (Lock.ret (T := T) (D := D) (get := get) hr))
  | .ret _ => NoSlot.map _ _ _ (.ret hr)
  | .send _ _ => NoSlot.map _ _ _ (noSlot_same _)
  | @Game.coin _ _ _ _ _ => NoSlot.map _ _ _ (noSlot_same _)

variable {plur : Chooser T D (γ × β)} {slot : ℕ} {Φ : D → γ × β → γ × β → ℝ} {Dead : γ × β → Prop}
  {Q : (c : γ) → (g' : Game β) → Strategy (interleave (.ret c) g') → Prop}

/-- **A slot met after `j` coins once `h` has ended is met after `n + j` coins** by a strategy playing `h` to its end
first, after `n` coins. -/
theorem SlotAt.rightSame_first (hΦ : ∀ d b b', Dead b → Dead b' → Φ d b b' = 0) {j n : ℕ} {h : Game γ}
    {g' : Game β} {σ : Strategy (interleave h g')} (hS : First Dead Q n h g' σ) :
    ∀ {i : I} {g : Game α} (L : Lock T D get R i g g'),
      (∀ c σ', Q c g' σ' → SlotAt plur slot Φ j (rightSame (.ret c) L) σ') →
        SlotAt plur slot Φ (n + j) (rightSame h L) σ := by
  induction hS with
  | dead hσ =>
      intro _ _ _ _
      exact .nul (forkE_of_reach plur slot hΦ _ _ hσ)
  | done hq =>
      intro _ _ L hQ
      rw [Nat.zero_add]
      exact hQ _ _ hq
  | last =>
      intro _ _ L _
      cases L with
      | ret hr => exact .of_noSlot (noSlot_rightSame_ret hr _) _
  | coinCoin _ ih =>
      intro _ _ L hQ
      cases L with
      | coin _ L =>
        rw [Nat.add_right_comm]
        simp only [rightSame, lockWithL]
        refine .send Bool ?_
        exact .coin _ fun c => ih c (.coin _ L) hQ
  | coinSend _ ih =>
      intro _ _ L hQ
      cases L with
      | send _ L =>
        rw [Nat.add_right_comm]
        simp only [rightSame, lockWithL]
        refine .send Bool ?_
        exact .coin _ fun c => ih c (.send _ L) hQ
      | table s upd hf hs hk L =>
        rw [Nat.add_right_comm]
        simp only [rightSame, lockWithL]
        refine .send Bool ?_
        exact .coin _ fun c => ih c (.table s upd hf hs hk L) hQ
  | sendCoin m _ ih =>
      intro _ _ L hQ
      cases L with
      | coin _ L =>
        simp only [rightSame, lockWithL]
        refine .send Bool ?_
        exact .send _ (ih (.coin _ L) hQ)
  | sendSend m _ ih =>
      intro _ _ L hQ
      cases L with
      | send _ L =>
        simp only [rightSame, lockWithL]
        refine .send Bool ?_
        exact .send _ (ih (.send _ L) hQ)
      | table s upd hf hs hk L =>
        simp only [rightSame, lockWithL]
        refine .send Bool ?_
        exact .send _ (ih (.table s upd hf hs hk L) hQ)

end First

/-- **The strategy plays the games in front of position `pre.length` to their ends, in order**, after `n` coins in
all, each by `First`; `Q cs σ'` holds at their outcomes `cs` and the strategy `σ'` on the rest of the batch after them,
and `Dead` reads the whole batch's outcomes. -/
def PreFirst {ζ : Type} (g : Game ζ) (post : List (Game ζ)) :
    (pre : List (Game ζ)) → (List ζ → Prop) → (List ζ → Strategy (batch (g :: post)) → Prop) → ℕ →
      Strategy (batch (pre ++ g :: post)) → Prop
  | [], _, Q, n, σ => n = 0 ∧ Q [] σ
  | h :: pre, Dead, Q, n, σ => ∃ n₀ n₁, n = n₀ + n₁ ∧
      First (fun x => Dead (x.1 :: x.2))
        (fun c g'' σ'' => ∃ e : g'' = batch (pre ++ g :: post),
          PreFirst g post pre (fun outs => Dead (c :: outs)) (fun cs => Q (c :: cs)) n₁
            (e ▸ Strategy.ofMap (fun b => (c, b)) g'' σ''))
        n₀ h (batch (pre ++ g :: post))
        (Strategy.ofMap (fun x : ζ × List ζ => x.1 :: x.2) (h.interleave (batch (pre ++ g :: post))) σ)

section Batch

variable {T D I ζ : Type} {get : I → ℕ → Option (T × D)} {R : I → ζ → ζ → Prop}

/-- **A slot of position `pre.length`'s lockstep met after `j` coins once the games in front have ended is met after
`n + j` coins** by a strategy playing them to their ends first, in order, after `n` coins (`PreFirst`), where it is met
after `j` at every chooser and every `Φ` that vanishes on two `Dead` outcomes. -/
theorem SlotAt.batchLock_first (d : ζ) {i : I} {g g' : Game ζ} (L : Lock T D get R i g g') (post : List (Game ζ))
    {slot j : ℕ} :
    ∀ (pre : List (Game ζ)) {plur : Chooser T D (List ζ)} {Φ : D → List ζ → List ζ → ℝ} {Dead : List ζ → Prop}
      {Q : List ζ → Strategy (batch (g' :: post)) → Prop} {n : ℕ} {σ : Strategy (batch (pre ++ g' :: post))},
      (∀ e b b', Dead b → Dead b' → Φ e b b' = 0) → PreFirst g' post pre Dead Q n σ →
      (∀ cs σ', Q cs σ' → ∀ (plur' : Chooser T D (List ζ)) (Φ' : D → List ζ → List ζ → ℝ),
        (∀ e b b', Dead (cs ++ b) → Dead (cs ++ b') → Φ' e b b' = 0) →
          SlotAt plur' slot Φ' j (batchLock d L post []) σ') →
      SlotAt plur slot Φ (n + j) (batchLock d L post pre) σ
  | [], plur, Φ, _, _, _, σ, hΦ, ⟨hn, hq⟩, hQ => by
      subst hn
      rw [Nat.zero_add]
      exact hQ [] σ hq plur Φ hΦ
  | h :: pre, plur, Φ, Dead, Q, _, σ, hΦ, ⟨n₀, n₁, hn, hF⟩, hQ => by
      subst hn
      have hs := SlotAt.rightSame_first (plur := plurMap (fun x : ζ × List ζ => x.1 :: x.2) plur)
        (Φ := fun e (b b' : ζ × List ζ) => Φ e (b.1 :: b.2) (b'.1 :: b'.2)) (fun e b b' hb hb' => hΦ e _ _ hb hb') hF
        (batchLock d L post pre) (j := n₁ + j) fun c σ' ⟨_, hp⟩ => by
          have hp' : PreFirst g' post pre (fun outs => Dead (c :: outs)) (fun cs => Q (c :: cs)) n₁
              (Strategy.ofMap (fun b => (c, b)) (batch (pre ++ g' :: post)) σ') := hp
          have key := SlotAt.map (R₂ := fun i (a b : ζ × List ζ) => a.1 = b.1 ∧
              R i (a.2.getD pre.length d) (b.2.getD pre.length d))
            (plur := plurMap (fun x : ζ × List ζ => x.1 :: x.2) plur)
            (Φ := fun e (x x' : ζ × List ζ) => Φ e (x.1 :: x.2) (x'.1 :: x'.2))
            (fun a => (c, a)) (fun b => (c, b)) (fun _ _ _ hr => ⟨rfl, hr⟩)
            (SlotAt.batchLock_first d L post pre (plur := plurMap (fun b => (c, b))
              (plurMap (fun x : ζ × List ζ => x.1 :: x.2) plur)) (Φ := fun e b b' => Φ e (c :: b) (c :: b'))
              (Dead := fun outs => Dead (c :: outs)) (Q := fun cs => Q (c :: cs))
              (fun e b b' hb hb' => hΦ e _ _ hb hb') hp' fun cs σ'' hq => hQ (c :: cs) σ'' hq)
          rw [Strategy.map_ofMap (fun b => (c, b)) (batch (pre ++ g' :: post)) σ'] at key
          exact key
      have key := SlotAt.map
        (R₂ := fun i (a b : List ζ) => R i (a.getD (h :: pre).length d) (b.getD (h :: pre).length d))
        (plur := plur) (Φ := Φ) (fun x : ζ × List ζ => x.1 :: x.2) (fun x : ζ × List ζ => x.1 :: x.2) (fun _ _ _ h' => h'.2) hs
      rw [Strategy.map_ofMap (fun x : ζ × List ζ => x.1 :: x.2) (h.interleave (batch (pre ++ g' :: post))) σ,
        ← Nat.add_assoc] at key
      exact key

end Batch

end FlockSoundness.Game.Lock
