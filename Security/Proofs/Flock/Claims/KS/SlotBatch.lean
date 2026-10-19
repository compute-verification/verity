import Proofs.Flock.Claims.KS.SlotPost

/-!
# A slot's coin count in a batch whose games pause and resume

`SlotAt.batchLock_first` and `SlotAtD.batchLock_nil` count a slot of one position's lockstep where the strategy plays
each game in front to its end before the next, and the lockstep's own game to its end before the games after it. A
strategy can instead play a game in front up to a point where only sends are left in it, move on, and send the rest of
it later; and play the games after the lockstep's at their sends at any time, and at their coins only once the
lockstep's game has flipped more coins than the slot is away. The count is the same:

* `SlotAtD.map`, `SlotAtD.mono`, `NoSlot.rightSame`, `SlotAtD.leftSame_ret`: relabelling keeps the chooser-free count,
  the same game beside a lockstep with no table node of the slot has none, and a finished game after the lockstep
  keeps its count;
* `FirstS`, `SlotAtD.rightSame_firstS`: the strategy advances the game in front, after exactly `n` coins, until `Q`
  holds; a slot met after `k` coins from there is met after `n + k`;
* `FollowR`, `SlotAtD.rightSame_followR`: the game in front moves only by sends while the rest follows `τ`; a slot met
  after `k` coins by `τ` is met after `k`;
* `FollowL`, `SlotAtD.leftSame_followL`: the lockstep's game follows `τ`, and the games after it move by sends, and by
  coins only once the lockstep's game has flipped more coins than the slot is away;
* `PreSched`, `SlotAtD.batchLock_sched`, `SlotAtD.batchLock_nil_followL`: in a batch, each game in front by `FirstS`
  and then `FollowR`, in order, after `n` coins in all, and the lockstep's game by `FollowL`, put a slot met after `k`
  coins by the lockstep's own strategy at `n + k`.
-/

namespace FlockSoundness.Game.Lock

section Map

variable {T D I α β γ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} {slot : ℕ}

theorem reach_map_map {β₂ : Type} (f : β → β₂) {x : β₂} :
    ∀ (g : Game β) (σ : Strategy g), Reach (g.map f) (Strategy.map f g σ) x → ∃ y, Reach g σ y ∧ x = f y
  | .ret b, _, h => ⟨b, rfl, h.symm⟩
  | .send _ k, σ, h => reach_map_map f (k σ.1) σ.2 h
  | @Game.coin _ _ _ _ k, σ, ⟨c, h⟩ => by
      obtain ⟨y, hy, rfl⟩ := reach_map_map f (k c) (σ c) h
      exact ⟨y, ⟨c, hy⟩, rfl⟩

/-- **The chooser-free count through a lockstep relabelled on both sides** is the count through the lockstep, with
`Dead` carried along. -/
theorem SlotAtD.map {α₂ β₂ : Type} {R₂ : I → α₂ → β₂ → Prop} {Dead : β → Prop} {Dead₂ : β₂ → Prop} (f : α → α₂)
    (f' : β → β₂) (h : ∀ i a b, R i a b → R₂ i (f a) (f' b)) (hD : ∀ b, Dead b → Dead₂ (f' b)) {j : ℕ} {i : I}
    {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} (hs : SlotAtD slot Dead j L σ') :
    SlotAtD slot Dead₂ j (Lock.map f f' h L) (Strategy.map f' g' σ') := by
  induction hs with
  | ret hr σ' => exact .ret (h _ _ _ hr) _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here σ' => exact .here _
  | @dead j i g g' L σ' hσ =>
      refine .dead fun x hx => ?_
      obtain ⟨y, hy, rfl⟩ := reach_map_map f' g' σ' hx
      exact hD y (hσ y hy)
  | noSlot h0 => exact .noSlot (h0.map f f' h)

theorem SlotAtD.mono {R' : I → α → β → Prop} {Dead : β → Prop} (h : ∀ i a b, R i a b → R' i a b) {j : ℕ} {i : I}
    {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} (hs : SlotAtD slot Dead j L σ') :
    SlotAtD slot Dead j (Lock.mono h L) σ' := by
  induction hs with
  | ret hr σ' => exact .ret (h _ _ _ hr) _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here σ' => exact .here _
  | dead hσ => exact .dead hσ
  | noSlot h0 => exact .noSlot (h0.mono h)

theorem NoSlot.lockWithL {R₂ : I → γ × α → γ × β → Prop} {adv : Game α → Game (γ × α)}
    {adv' : Game β → Game (γ × β)} {fin : α → Game (γ × α)} {fin' : β → Game (γ × β)}
    {hadv : ∀ {i : I} {g : Game α} {g' : Game β}, Lock T D get R i g g' → Lock T D get R₂ i (adv g) (adv' g')}
    {hfin : ∀ {i : I} {a : α} {b : β}, R i a b → Lock T D get R₂ i (fin a) (fin' b)}
    (hA : ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g'), NoSlot slot L → NoSlot slot (hadv L))
    (hF : ∀ {i : I} {a : α} {b : β} (hr : R i a b), NoSlot slot (hfin hr)) {i : I} {g : Game α} {g' : Game β}
    {L : Lock T D get R i g g'} (hL : NoSlot slot L) : NoSlot slot (Lock.lockWithL hadv hfin L) := by
  induction hL with
  | ret hr => exact hF hr
  | coin C hc ih =>
      simp only [Lock.lockWithL]
      refine .send Bool fun b => ?_
      cases b
      exacts [.coin C ih, hA _ (.coin C hc)]
  | send M hm ih =>
      simp only [Lock.lockWithL]
      refine .send Bool fun b => ?_
      cases b
      exacts [.send M ih, hA _ (.send M hm)]
  | table s hne ht ih =>
      simp only [Lock.lockWithL]
      refine .send Bool fun b => ?_
      cases b
      exacts [.table s hne ih, hA _ (.table s hne ht)]

/-- **The same game beside a lockstep with no table node of the slot has none**, the lockstep second. -/
theorem NoSlot.rightSame : ∀ (h : Game γ) {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'},
    NoSlot slot L → NoSlot slot (Lock.rightSame h L)
  | .ret _, _, _, _, _, hL => hL.map _ _ _
  | .send M k, _, _, _, _, hL =>
      NoSlot.lockWithL (hadv := fun L' => Lock.send M fun m => Lock.rightSame (k m) L')
        (hfin := fun hr => sameThen (.send M k) hr) (fun _ hL' => .send M fun m => NoSlot.rightSame (k m) hL')
        (fun _ => NoSlot.map _ _ _ (noSlot_same _)) hL
  | @Game.coin _ C _ _ k, _, _, _, _, hL =>
      NoSlot.lockWithL (hadv := fun L' => Lock.coin C fun c => Lock.rightSame (k c) L')
        (hfin := fun hr => sameThen (Game.coin C k) hr) (fun _ hL' => .coin C fun c => NoSlot.rightSame (k c) hL')
        (fun _ => NoSlot.map _ _ _ (noSlot_same _)) hL

/-- **A slot met after `j` coins through `L` is met after `j` through `leftSame L (.ret c)`**, read through the pairing
with `c`. -/
theorem SlotAtD.leftSame_ret {Dead : β → Prop} {Dead₂ : β × γ → Prop} (hD : ∀ b c, Dead b → Dead₂ (b, c)) {j : ℕ}
    {i : I} {g : Game α} {g' : Game β} (c : γ) (L : Lock T D get R i g g') (σ' : Strategy (interleave g' (.ret c)))
    (hs : SlotAtD slot Dead j L (Strategy.ofMap (·, c) g' (cast (congrArg Strategy (interleave_ret c g')) σ'))) :
    SlotAtD slot Dead₂ j (Lock.leftSame L (.ret c)) σ' := by
  have key := SlotAtD.map (R₂ := fun i a b => R i a.1 b.1 ∧ a.2 = b.2) (fun a => (a, c)) (fun b => (b, c))
    (fun _ _ _ hr => ⟨hr, rfl⟩) (fun b hb => hD b c hb) hs
  rw [Strategy.map_ofMap] at key
  cases L <;> exact key

end Map

/-- **The strategy advances `h` until `Q` holds, after exactly `n` coins**: at every choice of `interleave h g'` it
advances `h`, until `Q` holds at the rest of `h` and the strategy there (`stop`), unless `g'` has ended (`last`),
except below a part of the run every reached outcome of which is `Dead` (`dead`). -/
inductive FirstS {γ β : Type} (Dead : γ × β → Prop)
    (Q : (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Prop) :
    ℕ → (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Prop
  | dead {n : ℕ} {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} :
      (∀ x, Reach _ σ x → Dead x) → FirstS Dead Q n h g' σ
  | stop {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} : Q h g' σ → FirstS Dead Q 0 h g' σ
  | last {n : ℕ} {h : Game γ} {b : β} {σ : Strategy (interleave h (.ret b))} : FirstS Dead Q n h (.ret b) σ
  | coinCoin {n : ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game γ} {C₂ : Type} [Fintype C₂] [Nonempty C₂]
      {k₂ : C₂ → Game β} {τ : (c : C) → Strategy (interleave (k c) (.coin C₂ k₂))} :
      (∀ c, FirstS Dead Q n (k c) (.coin C₂ k₂) (τ c)) → FirstS Dead Q (n + 1) (.coin C k) (.coin C₂ k₂) ⟨true, τ⟩
  | coinSend {n : ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game γ} {M₂ : Type} {k₂ : M₂ → Game β}
      {τ : (c : C) → Strategy (interleave (k c) (.send M₂ k₂))} :
      (∀ c, FirstS Dead Q n (k c) (.send M₂ k₂) (τ c)) → FirstS Dead Q (n + 1) (.coin C k) (.send M₂ k₂) ⟨true, τ⟩
  | sendCoin {n : ℕ} {M : Type} {k : M → Game γ} {C₂ : Type} [Fintype C₂] [Nonempty C₂] {k₂ : C₂ → Game β} (m : M)
      {τ : Strategy (interleave (k m) (.coin C₂ k₂))} :
      FirstS Dead Q n (k m) (.coin C₂ k₂) τ → FirstS Dead Q n (.send M k) (.coin C₂ k₂) ⟨true, ⟨m, τ⟩⟩
  | sendSend {n : ℕ} {M : Type} {k : M → Game γ} {M₂ : Type} {k₂ : M₂ → Game β} (m : M)
      {τ : Strategy (interleave (k m) (.send M₂ k₂))} :
      FirstS Dead Q n (k m) (.send M₂ k₂) τ → FirstS Dead Q n (.send M k) (.send M₂ k₂) ⟨true, ⟨m, τ⟩⟩

section Right

variable {T D I α β γ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} {slot : ℕ}
  {Dead : γ × β → Prop}

/-- **A slot met after `k` coins from where `Q` holds is met after `n + k` coins** by a strategy advancing `h` until
`Q` holds, after `n` coins. -/
theorem SlotAtD.rightSame_firstS {Q : (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Prop} {k n : ℕ}
    {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} (hS : FirstS Dead Q n h g' σ) :
    ∀ {i : I} {g : Game α} (L : Lock T D get R i g g'),
      (∀ h' σ', Q h' g' σ' → SlotAtD slot Dead k (Lock.rightSame h' L) σ') →
        SlotAtD slot Dead (n + k) (Lock.rightSame h L) σ := by
  induction hS with
  | dead hσ =>
      intro _ _ _ _
      exact .dead hσ
  | stop hq =>
      intro _ _ L hQ
      rw [Nat.zero_add]
      exact hQ _ _ hq
  | last =>
      intro _ _ L _
      cases L with
      | ret hr => exact .noSlot (noSlot_rightSame_ret hr _)
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

end Right

/-- **The game in front moves only by sends, and the rest follows `τ`**: at every choice of `interleave h g'` the
strategy sends in `h` (`hsend…`), or advances `g'` as `τ` does (`icoin…`, `isend…`), until `h` has ended, where `τ` is
the strategy read through the pairing (`hret`), or `g'` has (`iret`), except below a part of the run every reached
outcome of which is `Dead` (`dead`). -/
inductive FollowR {γ β : Type} (Dead : γ × β → Prop) :
    (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Strategy g' → Prop
  | dead {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} {τ : Strategy g'} :
      (∀ x, Reach _ σ x → Dead x) → FollowR Dead h g' σ τ
  | hret {c : γ} {g' : Game β} (σ : Strategy (interleave (.ret c) g')) :
      FollowR Dead (.ret c) g' σ (Strategy.ofMap (fun b => (c, b)) g' σ)
  | iret {h : Game γ} {b : β} {σ : Strategy (interleave h (.ret b))} {τ : Strategy (.ret b)} :
      FollowR Dead h (.ret b) σ τ
  | hsendCoin {M : Type} {k : M → Game γ} {C₂ : Type} [Fintype C₂] [Nonempty C₂] {k₂ : C₂ → Game β} (m : M)
      {σ : Strategy (interleave (k m) (.coin C₂ k₂))} {τ : Strategy (.coin C₂ k₂)} :
      FollowR Dead (k m) (.coin C₂ k₂) σ τ → FollowR Dead (.send M k) (.coin C₂ k₂) ⟨true, ⟨m, σ⟩⟩ τ
  | hsendSend {M : Type} {k : M → Game γ} {M₂ : Type} {k₂ : M₂ → Game β} (m : M)
      {σ : Strategy (interleave (k m) (.send M₂ k₂))} {τ : Strategy (.send M₂ k₂)} :
      FollowR Dead (k m) (.send M₂ k₂) σ τ → FollowR Dead (.send M k) (.send M₂ k₂) ⟨true, ⟨m, σ⟩⟩ τ
  | icoinS {M : Type} {k : M → Game γ} {C₂ : Type} [Fintype C₂] [Nonempty C₂] {k₂ : C₂ → Game β}
      {σ : (c : C₂) → Strategy (interleave (.send M k) (k₂ c))} {τ : (c : C₂) → Strategy (k₂ c)} :
      (∀ c, FollowR Dead (.send M k) (k₂ c) (σ c) (τ c)) → FollowR Dead (.send M k) (.coin C₂ k₂) ⟨false, σ⟩ τ
  | icoinC {C : Type} [Fintype C] [Nonempty C] {k : C → Game γ} {C₂ : Type} [Fintype C₂] [Nonempty C₂]
      {k₂ : C₂ → Game β} {σ : (c : C₂) → Strategy (interleave (.coin C k) (k₂ c))} {τ : (c : C₂) → Strategy (k₂ c)} :
      (∀ c, FollowR Dead (.coin C k) (k₂ c) (σ c) (τ c)) → FollowR Dead (.coin C k) (.coin C₂ k₂) ⟨false, σ⟩ τ
  | isendS {M : Type} {k : M → Game γ} {M₂ : Type} {k₂ : M₂ → Game β} (m₂ : M₂)
      {σ : Strategy (interleave (.send M k) (k₂ m₂))} {τ : Strategy (k₂ m₂)} :
      FollowR Dead (.send M k) (k₂ m₂) σ τ → FollowR Dead (.send M k) (.send M₂ k₂) ⟨false, ⟨m₂, σ⟩⟩ ⟨m₂, τ⟩
  | isendC {C : Type} [Fintype C] [Nonempty C] {k : C → Game γ} {M₂ : Type} {k₂ : M₂ → Game β} (m₂ : M₂)
      {σ : Strategy (interleave (.coin C k) (k₂ m₂))} {τ : Strategy (k₂ m₂)} :
      FollowR Dead (.coin C k) (k₂ m₂) σ τ → FollowR Dead (.coin C k) (.send M₂ k₂) ⟨false, ⟨m₂, σ⟩⟩ ⟨m₂, τ⟩

/-- **A strategy whose game in front moves only by sends reaches only `Dead` outcomes or outcomes whose rest `τ`
reaches.** -/
theorem FollowR.reach {γ β : Type} {Dead : γ × β → Prop} {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')}
    {τ : Strategy g'} (hF : FollowR Dead h g' σ τ) : ∀ {x : γ × β}, Reach _ σ x → Dead x ∨ Reach g' τ x.2 := by
  induction hF with
  | dead hσ => exact fun hx => .inl (hσ _ hx)
  | @hret c g' σ =>
      intro x hx
      obtain ⟨y, hy, rfl⟩ := reach_map (fun b => (c, b)) g' σ hx
      exact .inr hy
  | @iret h b σ τ =>
      intro x hx
      obtain ⟨y, _, rfl⟩ := reach_map (·, b) h _ (reach_cast (interleave_ret b h) σ hx)
      exact .inr rfl
  | hsendCoin m _ ih => exact fun hx => ih hx
  | hsendSend m _ ih => exact fun hx => ih hx
  | icoinS _ ih =>
      rintro x ⟨c, hx⟩
      exact (ih c hx).imp_right fun h => ⟨c, h⟩
  | icoinC _ ih =>
      rintro x ⟨c, hx⟩
      exact (ih c hx).imp_right fun h => ⟨c, h⟩
  | isendS m₂ _ ih => exact fun hx => ih hx
  | isendC m₂ _ ih => exact fun hx => ih hx

/-- One coin off a budget of coins still to go up to the slot; `none` once the slot is behind. -/
def predS : Option ℕ → Option ℕ
  | some (n + 1) => some n
  | _ => none

/-- **The lockstep's game follows `τ`, and the games after it move by sends, and by coins only once the slot is
behind**: at every choice of `interleave g' h` the strategy advances `g'` as `τ` does (`gcoin…`, `gsend…`), one coin
off the budget `b` of coins still to go up to the slot at each coin, or sends in `h` (`hsend…`), or flips `h`'s coin
where the budget is spent (`hcoin…`), until `g'` has ended (`gret`) or `h` has, where `τ` is the strategy read through
the pairing (`hret`), except below a part of the run every reached outcome of which is `Dead` (`dead`). -/
inductive FollowL {β γ : Type} (Dead : β × γ → Prop) :
    Option ℕ → (g' : Game β) → (h : Game γ) → Strategy (interleave g' h) → Strategy g' → Prop
  | dead {b : Option ℕ} {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} {τ : Strategy g'} :
      (∀ x, Reach _ σ x → Dead x) → FollowL Dead b g' h σ τ
  | gret {b : Option ℕ} {a : β} {h : Game γ} {σ : Strategy (interleave (.ret a) h)} {τ : Strategy (.ret a)} :
      FollowL Dead b (.ret a) h σ τ
  | hret {b : Option ℕ} {g' : Game β} {c : γ} (σ : Strategy (interleave g' (.ret c))) :
      FollowL Dead b g' (.ret c) σ (Strategy.ofMap (·, c) g' (cast (congrArg Strategy (interleave_ret c g')) σ))
  | gcoinS {b : Option ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game β} {M₂ : Type} {k₂ : M₂ → Game γ}
      {σ : (c : C) → Strategy (interleave (k c) (.send M₂ k₂))} {τ : (c : C) → Strategy (k c)} :
      (∀ c, FollowL Dead (predS b) (k c) (.send M₂ k₂) (σ c) (τ c)) →
        FollowL Dead b (.coin C k) (.send M₂ k₂) ⟨true, σ⟩ τ
  | gcoinC {b : Option ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game β} {C₂ : Type} [Fintype C₂]
      [Nonempty C₂] {k₂ : C₂ → Game γ} {σ : (c : C) → Strategy (interleave (k c) (.coin C₂ k₂))}
      {τ : (c : C) → Strategy (k c)} :
      (∀ c, FollowL Dead (predS b) (k c) (.coin C₂ k₂) (σ c) (τ c)) →
        FollowL Dead b (.coin C k) (.coin C₂ k₂) ⟨true, σ⟩ τ
  | gsendS {b : Option ℕ} {M : Type} {k : M → Game β} {M₂ : Type} {k₂ : M₂ → Game γ} (m : M)
      {σ : Strategy (interleave (k m) (.send M₂ k₂))} {τ : Strategy (k m)} :
      FollowL Dead b (k m) (.send M₂ k₂) σ τ → FollowL Dead b (.send M k) (.send M₂ k₂) ⟨true, ⟨m, σ⟩⟩ ⟨m, τ⟩
  | gsendC {b : Option ℕ} {M : Type} {k : M → Game β} {C₂ : Type} [Fintype C₂] [Nonempty C₂] {k₂ : C₂ → Game γ}
      (m : M) {σ : Strategy (interleave (k m) (.coin C₂ k₂))} {τ : Strategy (k m)} :
      FollowL Dead b (k m) (.coin C₂ k₂) σ τ → FollowL Dead b (.send M k) (.coin C₂ k₂) ⟨true, ⟨m, σ⟩⟩ ⟨m, τ⟩
  | hsendS {b : Option ℕ} {M : Type} {k : M → Game β} {M₂ : Type} {k₂ : M₂ → Game γ} (m₂ : M₂)
      {σ : Strategy (interleave (.send M k) (k₂ m₂))} {τ : Strategy (.send M k)} :
      FollowL Dead b (.send M k) (k₂ m₂) σ τ → FollowL Dead b (.send M k) (.send M₂ k₂) ⟨false, ⟨m₂, σ⟩⟩ τ
  | hsendC {b : Option ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game β} {M₂ : Type} {k₂ : M₂ → Game γ}
      (m₂ : M₂) {σ : Strategy (interleave (.coin C k) (k₂ m₂))} {τ : Strategy (.coin C k)} :
      FollowL Dead b (.coin C k) (k₂ m₂) σ τ → FollowL Dead b (.coin C k) (.send M₂ k₂) ⟨false, ⟨m₂, σ⟩⟩ τ
  | hcoinS {M : Type} {k : M → Game β} {C₂ : Type} [Fintype C₂] [Nonempty C₂] {k₂ : C₂ → Game γ}
      {σ : (c : C₂) → Strategy (interleave (.send M k) (k₂ c))} {τ : Strategy (.send M k)} :
      (∀ c, FollowL Dead none (.send M k) (k₂ c) (σ c) τ) → FollowL Dead none (.send M k) (.coin C₂ k₂) ⟨false, σ⟩ τ
  | hcoinC {C : Type} [Fintype C] [Nonempty C] {k : C → Game β} {C₂ : Type} [Fintype C₂] [Nonempty C₂]
      {k₂ : C₂ → Game γ} {σ : (c : C₂) → Strategy (interleave (.coin C k) (k₂ c))} {τ : Strategy (.coin C k)} :
      (∀ c, FollowL Dead none (.coin C k) (k₂ c) (σ c) τ) → FollowL Dead none (.coin C k) (.coin C₂ k₂) ⟨false, σ⟩ τ

/-- **A strategy whose lockstep's game follows `τ` reaches only `Dead` outcomes or outcomes whose first part `τ`
reaches.** -/
theorem FollowL.reach {β γ : Type} {Dead : β × γ → Prop} {b : Option ℕ} {g' : Game β} {h : Game γ}
    {σ : Strategy (interleave g' h)} {τ : Strategy g'} (hF : FollowL Dead b g' h σ τ) :
    ∀ {x : β × γ}, Reach _ σ x → Dead x ∨ Reach g' τ x.1 := by
  induction hF with
  | dead hσ => exact fun hx => .inl (hσ _ hx)
  | @gret b a h σ τ =>
      intro x hx
      obtain ⟨y, _, rfl⟩ := reach_map (fun c => (a, c)) h σ hx
      exact .inr rfl
  | @hret b g' c σ =>
      intro x hx
      obtain ⟨y, hy, rfl⟩ := reach_map (·, c) g' _ (reach_cast (interleave_ret c g') σ hx)
      exact .inr hy
  | gcoinS _ ih =>
      rintro x ⟨c, hx⟩
      exact (ih c hx).imp_right fun h => ⟨c, h⟩
  | gcoinC _ ih =>
      rintro x ⟨c, hx⟩
      exact (ih c hx).imp_right fun h => ⟨c, h⟩
  | gsendS m _ ih => exact fun hx => ih hx
  | gsendC m _ ih => exact fun hx => ih hx
  | hsendS m₂ _ ih => exact fun hx => ih hx
  | hsendC m₂ _ ih => exact fun hx => ih hx
  | hcoinS _ ih =>
      rintro x ⟨c, hx⟩
      exact ih c hx
  | hcoinC _ ih =>
      rintro x ⟨c, hx⟩
      exact ih c hx

section Follow

variable {T D I α β γ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} {slot : ℕ}

/-- **A slot met after `k` coins by `τ` is met after `k` coins beside a game in front that moves only by sends.** -/
theorem SlotAtD.rightSame_followR {Dead : γ × β → Prop} {Dead' : β → Prop} (hD : ∀ x : γ × β, Dead' x.2 → Dead x)
    {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} {τ : Strategy g'} (hF : FollowR Dead h g' σ τ) :
    ∀ {k : ℕ} {i : I} {g : Game α} (L : Lock T D get R i g g'), SlotAtD slot Dead' k L τ →
      SlotAtD slot Dead k (Lock.rightSame h L) σ := by
  have hdead : ∀ {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} {τ : Strategy g'},
      FollowR Dead h g' σ τ → ∀ {k : ℕ} {i : I} {g : Game α} (L : Lock T D get R i g g'),
        (∀ b, Reach g' τ b → Dead' b) → SlotAtD slot Dead k (Lock.rightSame h L) σ :=
    fun hF _ _ _ _ hr => .dead fun x hx => (hF.reach hx).elim id fun h => hD x (hr _ h)
  have hno : ∀ {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} {k : ℕ} {i : I} {g : Game α}
      {L : Lock T D get R i g g'}, NoSlot slot L → SlotAtD slot Dead k (Lock.rightSame h L) σ :=
    fun h0 => .noSlot (h0.rightSame _)
  induction hF with
  | dead hσ => exact fun _ _ => .dead hσ
  | @hret c g' σ =>
      intro k i g L hL
      have key := SlotAtD.map (R₂ := fun i a b => a.1 = b.1 ∧ R i a.2 b.2) (fun a => (c, a)) (fun b => (c, b))
        (fun _ _ _ hr => ⟨rfl, hr⟩) (fun b hb => hD (c, b) hb) hL
      exact Strategy.map_ofMap (fun b => (c, b)) g' σ ▸ key
  | iret =>
      intro k i g L _
      cases L with
      | ret hr => exact .noSlot (noSlot_rightSame_ret hr _)
  | hsendCoin m _ ih =>
      intro k i g L hL
      cases L with
      | coin _ L =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .send _ (ih (.coin _ L) hL)
  | hsendSend m _ ih =>
      intro k i g L hL
      cases L with
      | send _ L =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .send _ (ih (.send _ L) hL)
      | table s upd hf hs hk L =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .send _ (ih (.table s upd hf hs hk L) hL)
  | icoinS hτ ih =>
      intro k i g L hL
      cases hL with
      | coin _ hL =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .coin _ fun c => ih c _ (hL c)
      | dead hr => exact hdead (.icoinS hτ) _ hr
      | noSlot h0 => exact hno h0
  | icoinC hτ ih =>
      intro k i g L hL
      cases hL with
      | coin _ hL =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .coin _ fun c => ih c _ (hL c)
      | dead hr => exact hdead (.icoinC hτ) _ hr
      | noSlot h0 => exact hno h0
  | isendS m₂ hτ ih =>
      intro k i g L hL
      cases hL with
      | send _ hL =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .send _ (ih _ hL)
      | other s hne hL =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .other _ hne fun t => ih _ (hL t)
      | here =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .here _
      | dead hr => exact hdead (.isendS m₂ hτ) _ hr
      | noSlot h0 => exact hno h0
  | isendC m₂ hτ ih =>
      intro k i g L hL
      cases hL with
      | send _ hL =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .send _ (ih _ hL)
      | other s hne hL =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .other _ hne fun t => ih _ (hL t)
      | here =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .here _
      | dead hr => exact hdead (.isendC m₂ hτ) _ hr
      | noSlot h0 => exact hno h0

/-- **A slot met after `k` coins by `τ` is met after `k` coins beside games after the lockstep** by a strategy whose
lockstep's game follows `τ`, with a budget of `k` coins up to the slot. -/
theorem SlotAtD.leftSame_followL {Dead : β × γ → Prop} {Dead' : β → Prop} (hD : ∀ x : β × γ, Dead' x.1 → Dead x)
    {b : Option ℕ} {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} {τ : Strategy g'}
    (hF : FollowL Dead b g' h σ τ) :
    ∀ {k : ℕ} {i : I} {g : Game α} (L : Lock T D get R i g g'), b = some k → SlotAtD slot Dead' k L τ →
      SlotAtD slot Dead k (Lock.leftSame L h) σ := by
  have hdead : ∀ {b : Option ℕ} {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} {τ : Strategy g'},
      FollowL Dead b g' h σ τ → ∀ {k : ℕ} {i : I} {g : Game α} (L : Lock T D get R i g g'),
        (∀ b, Reach g' τ b → Dead' b) → SlotAtD slot Dead k (Lock.leftSame L h) σ :=
    fun hF _ _ _ _ hr => .dead fun x hx => (hF.reach hx).elim id fun h => hD x (hr _ h)
  have hno : ∀ {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} {k : ℕ} {i : I} {g : Game α}
      {L : Lock T D get R i g g'}, NoSlot slot L → SlotAtD slot Dead k (Lock.leftSame L h) σ :=
    fun h0 => .noSlot (h0.leftSame _)
  induction hF with
  | dead hσ => exact fun _ _ _ => .dead hσ
  | gret =>
      intro k i g L _ _
      cases L with
      | ret hr => exact .noSlot (NoSlot.map _ _ _ (noSlot_same _))
  | @hret b g' c σ =>
      intro k i g L _ hL
      exact SlotAtD.leftSame_ret (fun b c hb => hD (b, c) hb) c L σ hL
  | gcoinS hτ ih =>
      intro k i g L hb hL
      cases hL with
      | coin _ hL =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .coin _ fun c => ih c _ (by rw [hb]; rfl) (hL c)
      | dead hr => exact hdead (.gcoinS hτ) _ hr
      | noSlot h0 => exact hno h0
  | gcoinC hτ ih =>
      intro k i g L hb hL
      cases hL with
      | coin _ hL =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .coin _ fun c => ih c _ (by rw [hb]; rfl) (hL c)
      | dead hr => exact hdead (.gcoinC hτ) _ hr
      | noSlot h0 => exact hno h0
  | gsendS m hτ ih =>
      intro k i g L hb hL
      cases hL with
      | send _ hL =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .send _ (ih _ hb hL)
      | other s hne hL =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .other _ hne fun t => ih _ hb (hL t)
      | here =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .here _
      | dead hr => exact hdead (.gsendS m hτ) _ hr
      | noSlot h0 => exact hno h0
  | gsendC m hτ ih =>
      intro k i g L hb hL
      cases hL with
      | send _ hL =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .send _ (ih _ hb hL)
      | other s hne hL =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .other _ hne fun t => ih _ hb (hL t)
      | here =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .here _
      | dead hr => exact hdead (.gsendC m hτ) _ hr
      | noSlot h0 => exact hno h0
  | hsendS m₂ _ ih =>
      intro k i g L hb hL
      cases L with
      | send _ L =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .send _ (ih (.send _ L) hb hL)
      | table s upd hf hs hk L =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .send _ (ih (.table s upd hf hs hk L) hb hL)
  | hsendC m₂ _ ih =>
      intro k i g L hb hL
      cases L with
      | coin _ L =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .send _ (ih (.coin _ L) hb hL)
  | hcoinS _ _ =>
      intro k i g L hb
      cases hb
  | hcoinC _ _ =>
      intro k i g L hb
      cases hb

end Follow

/-- **The strategy plays each game in front of position `pre.length` by `FirstS` and then `FollowR`, in order**, after
`n` coins in all; `Q Dead' σ'` holds at the strategy `σ'` on the rest of the batch where those games pause, `Dead'`
holding of the rest's outcomes with every outcome of the games in front `Dead`. -/
def PreSched {ζ : Type} (g : Game ζ) (post : List (Game ζ)) :
    (pre : List (Game ζ)) → (List ζ → Prop) → ((List ζ → Prop) → Strategy (batch (g :: post)) → Prop) → ℕ →
      Strategy (batch (pre ++ g :: post)) → Prop
  | [], Dead, Q, n, σ => n = 0 ∧ Q Dead σ
  | h :: pre, Dead, Q, n, σ => ∃ n₀ n₁, n = n₀ + n₁ ∧
      FirstS (fun x => Dead (x.1 :: x.2))
        (fun h' g'' σ'' => ∃ e : g'' = batch (pre ++ g :: post), ∃ τ : Strategy g'',
          FollowR (fun x => Dead (x.1 :: x.2)) h' g'' σ'' τ ∧
            PreSched g post pre (fun outs => ∀ c, Dead (c :: outs)) Q n₁ (e ▸ τ))
        n₀ h (batch (pre ++ g :: post))
        (Strategy.ofMap (fun x : ζ × List ζ => x.1 :: x.2) (h.interleave (batch (pre ++ g :: post))) σ)

section Batch

variable {T D I ζ : Type} {get : I → ℕ → Option (T × D)} {R : I → ζ → ζ → Prop}

/-- **A batch around a lockstep with no table node of the slot has none.** -/
theorem NoSlot.batchLock {slot : ℕ} (d : ζ) {i : I} {g g' : Game ζ} {L : Lock T D get R i g g'} (hL : NoSlot slot L)
    (post : List (Game ζ)) : ∀ pre : List (Game ζ), NoSlot slot (Lock.batchLock d L post pre)
  | [] => NoSlot.map _ _ _ (hL.leftSame _)
  | h :: pre => NoSlot.map _ _ _ (NoSlot.rightSame h (NoSlot.batchLock d hL post pre))

/-- **A slot of position `pre.length`'s lockstep met after `k` coins where the games in front pause is met after
`n + k` coins** by a strategy playing them by `PreSched`, after `n` coins. -/
theorem SlotAtD.batchLock_sched (d : ζ) {i : I} {g g' : Game ζ} (L : Lock T D get R i g g') (post : List (Game ζ))
    {slot k : ℕ} {Q : (List ζ → Prop) → Strategy (batch (g' :: post)) → Prop}
    (hQ : ∀ Dead σ', Q Dead σ' → SlotAtD slot Dead k (batchLock d L post []) σ') :
    ∀ (pre : List (Game ζ)) {Dead : List ζ → Prop} {n : ℕ} {σ : Strategy (batch (pre ++ g' :: post))},
      PreSched g' post pre Dead Q n σ → SlotAtD slot Dead (n + k) (batchLock d L post pre) σ
  | [], Dead, _, σ, ⟨hn, hq⟩ => by
      subst hn
      rw [Nat.zero_add]
      exact hQ Dead σ hq
  | h :: pre, Dead, _, σ, ⟨n₀, n₁, hn, hF⟩ => by
      subst hn
      have hs := SlotAtD.rightSame_firstS (k := n₁ + k) hF (batchLock d L post pre) fun h' σ' ⟨_, τ, hR, hP⟩ => by
        have hP' : PreSched g' post pre (fun outs => ∀ c, Dead (c :: outs)) Q n₁ τ := hP
        exact SlotAtD.rightSame_followR (Dead' := fun outs => ∀ c, Dead (c :: outs)) (fun x hx => hx x.1) hR _
          (SlotAtD.batchLock_sched d L post hQ pre hP')
      have key := SlotAtD.map
        (R₂ := fun i (a b : List ζ) => R i (a.getD (h :: pre).length d) (b.getD (h :: pre).length d))
        (Dead₂ := Dead) (fun x : ζ × List ζ => x.1 :: x.2) (fun x : ζ × List ζ => x.1 :: x.2)
        (fun _ _ _ h' => h'.2) (fun _ hx => hx) hs
      rw [Strategy.map_ofMap (fun x : ζ × List ζ => x.1 :: x.2) (h.interleave (batch (pre ++ g' :: post))) σ,
        ← Nat.add_assoc] at key
      exact key

/-- **A slot met after `k` coins by the lockstep's own strategy `τ` is met after `k` coins in the batch with nothing
in front** by a strategy whose lockstep's game follows `τ` (`FollowL`, a budget of `k` coins up to the slot). -/
theorem SlotAtD.batchLock_nil_followL (d : ζ) {i : I} {g g' : Game ζ} (L : Lock T D get R i g g')
    (post : List (Game ζ)) {slot k : ℕ} {Dead : List ζ → Prop} {σ' : Strategy (batch (g' :: post))}
    {τ : Strategy g'}
    (hF : FollowL (fun x : ζ × List ζ => Dead (x.1 :: x.2)) (some k) g' (batch post)
      (Strategy.ofMap (fun x : ζ × List ζ => x.1 :: x.2) (g'.interleave (batch post)) σ') τ)
    (hD : SlotAtD slot (fun x => ∀ rest, Dead (x :: rest)) k L τ) :
    SlotAtD slot Dead k (batchLock d L post []) σ' := by
  have hs := SlotAtD.leftSame_followL (fun x hx => hx x.2) hF L rfl hD
  have key := SlotAtD.map (R₂ := fun i a b => R i (a.getD 0 d) (b.getD 0 d)) (Dead₂ := Dead)
    (fun x : ζ × List ζ => x.1 :: x.2) (fun x : ζ × List ζ => x.1 :: x.2) (fun _ _ _ h => h.1) (fun _ hx => hx) hs
  exact Strategy.map_ofMap (fun x : ζ × List ζ => x.1 :: x.2) (g'.interleave (batch post)) σ' ▸ key

end Batch

end FlockSoundness.Game.Lock
