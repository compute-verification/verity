import Proofs.Flock.Claims.KS.SlotDist

/-!
# A slot's coin count through two interleaved locksteps, under a schedule that leads with the first

`Dist` fixes a slot's coin count inside one lockstep. Two locksteps interleaved (`interleaveLock`) meet a slot of either
after a count that depends on the choices the strategy makes at `interleave`'s nodes. Where the strategy advances the
first game at every choice until it ends, after exactly `n` coins (`Lead`), or plays a part of the run whose every
reached outcome is `Dead` (`Lead.dead`, `Reach`), a continuation with no table node of the slot after both:

* `forkE_of_reach`: the fork is `0` against a strategy reaching only `Dead` outcomes where `Φ` vanishes on two of them;
* `NoSlot.interleaveLock`, `Dist.embedLeft`, `Dist.embedRight`, `Dist.interleaveLock_ret`, `Dist.bind_noSlot`: no table
  node of the slot, and a distance, survive interleaving and a continuation with no such node;
* `SlotAt.lead_fst`: a slot of the first lockstep at distance `j` is met after `j` coins;
* `SlotAt.lead_snd`: a slot of the second at distance `j` is met after `n + j` coins.
-/

namespace FlockSoundness.Game.Lock

/-- **An outcome `σ` reaches in `g`**, at some coins. -/
def Reach {β : Type} : (g : Game β) → Strategy g → β → Prop
  | .ret b, _, b' => b = b'
  | .send _ k, σ, b' => Reach (k σ.1) σ.2 b'
  | @Game.coin _ _ _ _ k, σ, b' => ∃ c, Reach (k c) (σ c) b'

theorem expect_of_reach {β : Type} {f : β → ℝ} :
    ∀ (g : Game β) (σ : Strategy g), (∀ b, Reach g σ b → f b = 0) → expect f g σ = 0
  | .ret b, _, h => h b rfl
  | .send _ k, σ, h => expect_of_reach (k σ.1) σ.2 h
  | @Game.coin _ C _ _ k, σ, h => by
      simp only [expect]
      rw [Finset.sum_eq_zero fun c _ => expect_of_reach (k c) (σ c) fun b hb => h b ⟨c, hb⟩, zero_div]

/-- **Against a strategy reaching only `Dead` outcomes the fork is `0`**, where `Φ` vanishes on two `Dead` outcomes. -/
theorem forkE_of_reach {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) {Φ : D → β → β → ℝ} {Dead : β → Prop}
    (hΦ : ∀ d b b', Dead b → Dead b' → Φ d b b' = 0) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      (∀ b, Reach g' σ' b → Dead b) → forkE plur slot Φ L σ' = 0
  | _, _, _, .ret _, _, _ => rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', h => by
      simp only [forkE]
      rw [Finset.sum_eq_zero fun c _ => forkE_of_reach plur slot hΦ (L c) (σ' c) fun b hb => h b ⟨c, hb⟩,
        zero_div]
  | _, _, _, .send _ L, σ', h => forkE_of_reach plur slot hΦ (L σ'.1) σ'.2 h
  | _, _, _, .table s _ _ _ _ L, σ', h => by
      simp only [forkE]
      split_ifs
      · exact expect_of_reach _ _ fun b hb => expect_of_reach _ _ fun b' hb' => hΦ _ _ _ (h b hb) (h b' hb')
      · exact forkE_of_reach plur slot hΦ _ σ'.2 h

/-- **Sequencing a distance with a continuation that has no table node of the slot** keeps the distance. -/
theorem Dist.bind_noSlot {T D I α β α₂ β₂ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {R₂ : I → α₂ → β₂ → Prop} {f : α → Game α₂} {f' : β → Game β₂} {slot j : ℕ}
    {h : ∀ i a b, R i a b → Lock T D get R₂ i (f a) (f' b)} (hK : ∀ i a b (hr : R i a b), NoSlot slot (h i a b hr))
    {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (hL : Dist slot j L) :
    Dist slot j (Lock.bind h L) := by
  induction hL with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (h0.bind hK)

section Interleave

variable {T D α₁ β₁ α₂ β₂ I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (T × D)} {get₂ : I₂ → ℕ → Option (T × D)}
  {R₁ : I₁ → α₁ → β₁ → Prop} {R₂ : I₂ → α₂ → β₂ → Prop} [Inhabited T] [Inhabited D] (hd : Lock.Disjoint get₁ get₂)
  {slot : ℕ}

include hd

theorem NoSlot.embedLeft {i₂ : I₂} {a₂ : α₂} {b₂ : β₂} (h₂ : R₂ i₂ a₂ b₂) {i₁ : I₁} {g₁ : Game α₁}
    {g₁' : Game β₁} {L : Lock T D get₁ R₁ i₁ g₁ g₁'} (hL : NoSlot slot L) : NoSlot slot (embedLeft hd h₂ L) := by
  induction hL with
  | ret _ => exact .ret _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s hne _ ih => exact .table s hne ih

theorem NoSlot.embedRight {i₁ : I₁} {a₁ : α₁} {b₁ : β₁} (h₁ : R₁ i₁ a₁ b₁) {i₂ : I₂} {g₂ : Game α₂}
    {g₂' : Game β₂} {L : Lock T D get₂ R₂ i₂ g₂ g₂'} (hL : NoSlot slot L) : NoSlot slot (embedRight hd h₁ L) := by
  induction hL with
  | ret _ => exact .ret _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s hne _ ih => exact .table s hne ih

theorem Dist.embedLeft {i₂ : I₂} {a₂ : α₂} {b₂ : β₂} (h₂ : R₂ i₂ a₂ b₂) {j : ℕ} {i₁ : I₁} {g₁ : Game α₁}
    {g₁' : Game β₁} {L : Lock T D get₁ R₁ i₁ g₁ g₁'} (hL : Dist slot j L) : Dist slot j (embedLeft hd h₂ L) := by
  induction hL with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (h0.embedLeft hd h₂)

theorem Dist.embedRight {i₁ : I₁} {a₁ : α₁} {b₁ : β₁} (h₁ : R₁ i₁ a₁ b₁) {j : ℕ} {i₂ : I₂} {g₂ : Game α₂}
    {g₂' : Game β₂} {L : Lock T D get₂ R₂ i₂ g₂ g₂'} (hL : Dist slot j L) : Dist slot j (embedRight hd h₁ L) := by
  induction hL with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (h0.embedRight hd h₁)

/-- **Interleaving with a finished lockstep keeps a distance.** -/
theorem Dist.interleaveLock_ret {i₂ : I₂} {a₂ : α₂} {b₂ : β₂} (h₂ : R₂ i₂ a₂ b₂) {j : ℕ} {i₁ : I₁}
    {g₁ : Game α₁} {g₁' : Game β₁} {L : Lock T D get₁ R₁ i₁ g₁ g₁'} (hL : Dist slot j L) :
    Dist slot j (interleaveLock hd L (.ret h₂)) := by
  cases L with
  | ret _ => exact .noSlot (.ret _)
  | coin => exact hL.embedLeft hd h₂
  | send => exact hL.embedLeft hd h₂
  | table => exact hL.embedLeft hd h₂

theorem NoSlot.lockWith {i₁ : I₁} {adv₁ : Game α₂ → Game (α₁ × α₂)} {adv₁' : Game β₂ → Game (β₁ × β₂)}
    {fin₁ : α₂ → Game (α₁ × α₂)} {fin₁' : β₂ → Game (β₁ × β₂)}
    {hadv : ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂}, Lock T D get₂ R₂ i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (adv₁ g₂) (adv₁' g₂')}
    {hfin : ∀ {i₂ : I₂} {a₂ : α₂} {b₂ : β₂}, R₂ i₂ a₂ b₂ →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (fin₁ a₂) (fin₁' b₂)}
    (hA : ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂} (L : Lock T D get₂ R₂ i₂ g₂ g₂'), NoSlot slot L →
      NoSlot slot (hadv L))
    (hF : ∀ {i₂ : I₂} {a₂ : α₂} {b₂ : β₂} (h : R₂ i₂ a₂ b₂), NoSlot slot (hfin h))
    {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂} {L : Lock T D get₂ R₂ i₂ g₂ g₂'} (hL : NoSlot slot L) :
    NoSlot slot (lockWith hd hadv hfin L) := by
  induction hL with
  | ret h => exact hF h
  | coin C hc ih =>
      simp only [Lock.lockWith]
      refine .send Bool fun b => ?_
      cases b
      exacts [.coin C ih, hA _ (.coin C hc)]
  | send M hm ih =>
      simp only [Lock.lockWith]
      refine .send Bool fun b => ?_
      cases b
      exacts [.send M ih, hA _ (.send M hm)]
  | table s hne ht ih =>
      simp only [Lock.lockWith]
      refine .send Bool fun b => ?_
      cases b
      exacts [.table s hne ih, hA _ (.table s hne ht)]

/-- **Two locksteps with no table node of the slot interleave to none.** -/
theorem NoSlot.interleaveLock {i₁ : I₁} {g₁ : Game α₁} {g₁' : Game β₁} {L₁ : Lock T D get₁ R₁ i₁ g₁ g₁'}
    (h₁ : NoSlot slot L₁) : ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂} {L₂ : Lock T D get₂ R₂ i₂ g₂ g₂'},
      NoSlot slot L₂ → NoSlot slot (interleaveLock hd L₁ L₂) := by
  induction h₁ with
  | ret h => exact fun h₂ => h₂.embedRight hd h
  | coin C hc ih =>
      intro _ _ _ _ h₂
      refine NoSlot.lockWith hd ?_ ?_ h₂
      · intro _ _ _ _ h
        exact .coin C fun c => ih c h
      · intro _ _ _ h
        exact NoSlot.embedLeft hd h (.coin C hc)
  | send M hm ih =>
      intro _ _ _ _ h₂
      refine NoSlot.lockWith hd ?_ ?_ h₂
      · intro _ _ _ _ h
        exact .send M fun m => ih m h
      · intro _ _ _ h
        exact NoSlot.embedLeft hd h (.send M hm)
  | table s hne ht ih =>
      intro _ _ _ _ h₂
      refine NoSlot.lockWith hd ?_ ?_ h₂
      · intro _ _ _ _ h
        exact .table s hne fun t d => ih t d h
      · intro _ _ _ h
        exact NoSlot.embedLeft hd h (.table s hne ht)

end Interleave

/-- **The strategy leads with the first game**: at every choice of `interleave` it advances the first game, which ends
after exactly `n` coins (`done`), until the second has ended (`last`), except below a part of the run every reached
outcome of which is `Dead` (`dead`). -/
inductive Lead {α β γ : Type} (f : α × β → Game γ) (Dead : γ → Prop) :
    ℕ → (g₁ : Game α) → (g₂ : Game β) → Strategy ((interleave g₁ g₂).bind f) → Prop
  | dead {n : ℕ} {g₁ : Game α} {g₂ : Game β} {σ : Strategy ((interleave g₁ g₂).bind f)} :
      (∀ c, Reach _ σ c → Dead c) → Lead f Dead n g₁ g₂ σ
  | done {a : α} {g₂ : Game β} {σ : Strategy ((interleave (.ret a) g₂).bind f)} : Lead f Dead 0 (.ret a) g₂ σ
  | last {n : ℕ} {g₁ : Game α} {b : β} {σ : Strategy ((interleave g₁ (.ret b)).bind f)} :
      Lead f Dead n g₁ (.ret b) σ
  | coinCoin {n : ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game α} {C₂ : Type} [Fintype C₂] [Nonempty C₂]
      {k₂ : C₂ → Game β} {τ : (c : C) → Strategy ((interleave (k c) (.coin C₂ k₂)).bind f)} :
      (∀ c, Lead f Dead n (k c) (.coin C₂ k₂) (τ c)) → Lead f Dead (n + 1) (.coin C k) (.coin C₂ k₂) ⟨true, τ⟩
  | coinSend {n : ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game α} {M₂ : Type} {k₂ : M₂ → Game β}
      {τ : (c : C) → Strategy ((interleave (k c) (.send M₂ k₂)).bind f)} :
      (∀ c, Lead f Dead n (k c) (.send M₂ k₂) (τ c)) → Lead f Dead (n + 1) (.coin C k) (.send M₂ k₂) ⟨true, τ⟩
  | sendCoin {n : ℕ} {M : Type} {k : M → Game α} {C₂ : Type} [Fintype C₂] [Nonempty C₂] {k₂ : C₂ → Game β} (m : M)
      {τ : Strategy ((interleave (k m) (.coin C₂ k₂)).bind f)} :
      Lead f Dead n (k m) (.coin C₂ k₂) τ → Lead f Dead n (.send M k) (.coin C₂ k₂) ⟨true, ⟨m, τ⟩⟩
  | sendSend {n : ℕ} {M : Type} {k : M → Game α} {M₂ : Type} {k₂ : M₂ → Game β} (m : M)
      {τ : Strategy ((interleave (k m) (.send M₂ k₂)).bind f)} :
      Lead f Dead n (k m) (.send M₂ k₂) τ → Lead f Dead n (.send M k) (.send M₂ k₂) ⟨true, ⟨m, τ⟩⟩

section Lead

variable {T D α₁ β₁ α₂ β₂ I₁ I₂ γ γ' : Type} {get₁ : I₁ → ℕ → Option (T × D)}
  {get₂ : I₂ → ℕ → Option (T × D)} {R₁ : I₁ → α₁ → β₁ → Prop} {R₂ : I₂ → α₂ → β₂ → Prop} [Inhabited T]
  [Inhabited D] (hd : Lock.Disjoint get₁ get₂) {slot : ℕ} {plur : Chooser T D γ'} {Φ : D → γ' → γ' → ℝ}
  {Dead : γ' → Prop} {fo : α₁ × α₂ → Game γ} {f' : β₁ × β₂ → Game γ'} {R₃ : I₁ × I₂ → γ → γ' → Prop}
  {hK : ∀ i a b, RPair R₁ R₂ i a b → Lock T D (getPair get₁ get₂) R₃ i (fo a) (f' b)}

include hd

/-- **A slot of the first lockstep at distance `j` is met after `j` coins** by a strategy leading with it. -/
theorem SlotAt.lead_fst (hΦ : ∀ d b b', Dead b → Dead b' → Φ d b b' = 0)
    (hKn : ∀ i a b (hr : RPair R₁ R₂ i a b), NoSlot slot (hK i a b hr)) {n : ℕ} {g₁' : Game β₁} {g₂' : Game β₂}
    {σ : Strategy ((interleave g₁' g₂').bind f')} (hS : Lead f' Dead n g₁' g₂' σ) :
    ∀ {j : ℕ} {i₁ : I₁} {g₁ : Game α₁} (L₁ : Lock T D get₁ R₁ i₁ g₁ g₁') {i₂ : I₂} {g₂ : Game α₂}
      (L₂ : Lock T D get₂ R₂ i₂ g₂ g₂'), Dist slot j L₁ → NoSlot slot L₂ →
        SlotAt plur slot Φ j (Lock.bind hK (interleaveLock hd L₁ L₂)) σ := by
  induction hS with
  | dead hσ =>
      intro _ _ _ _ _ _ _ _ _
      exact .nul (forkE_of_reach plur slot hΦ _ _ hσ)
  | done =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ _ hN₂
      cases L₁ with
      | ret h₁ => exact .of_noSlot ((hN₂.embedRight hd h₁).bind hKn) _
  | last =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hD _
      cases L₂ with
      | ret h₂ => exact .of_dist ((hD.interleaveLock_ret hd h₂).bind_noSlot hKn) _
  | coinCoin _ ih =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hD hN₂
      cases L₁ with
      | coin _ L₁ =>
        cases hD with
        | coin _ hD =>
          cases L₂ with
          | coin _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .coin _ fun c => ih c (L₁ c) (.coin _ L₂) (hD c) hN₂
        | noSlot h0 => exact .of_noSlot ((h0.interleaveLock hd hN₂).bind hKn) _
  | coinSend _ ih =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hD hN₂
      cases L₁ with
      | coin _ L₁ =>
        cases hD with
        | coin _ hD =>
          cases L₂ with
          | send _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .coin _ fun c => ih c (L₁ c) (.send _ L₂) (hD c) hN₂
          | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .coin _ fun c => ih c (L₁ c) (.table s₂ upd₂ hf₂ hs₂ hk₂ L₂) (hD c) hN₂
        | noSlot h0 => exact .of_noSlot ((h0.interleaveLock hd hN₂).bind hKn) _
  | sendCoin m _ ih =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hD hN₂
      cases L₁ with
      | send _ L₁ =>
        cases hD with
        | send _ hD =>
          cases L₂ with
          | coin _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .send _ (ih (L₁ m) (.coin _ L₂) (hD m) hN₂)
        | noSlot h0 => exact .of_noSlot ((h0.interleaveLock hd hN₂).bind hKn) _
      | table s upd hf hs hk L₁ =>
        cases hD with
        | other _ hne hD =>
          cases L₂ with
          | coin _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .other _ hne (ih (L₁ _ m) (.coin _ L₂) (hD _ m) hN₂)
        | here =>
          cases L₂ with
          | coin _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .here _
        | noSlot h0 => exact .of_noSlot ((h0.interleaveLock hd hN₂).bind hKn) _
  | sendSend m _ ih =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hD hN₂
      cases L₁ with
      | send _ L₁ =>
        cases hD with
        | send _ hD =>
          cases L₂ with
          | send _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .send _ (ih (L₁ m) (.send _ L₂) (hD m) hN₂)
          | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .send _ (ih (L₁ m) (.table s₂ upd₂ hf₂ hs₂ hk₂ L₂) (hD m) hN₂)
        | noSlot h0 => exact .of_noSlot ((h0.interleaveLock hd hN₂).bind hKn) _
      | table s upd hf hs hk L₁ =>
        cases hD with
        | other _ hne hD =>
          cases L₂ with
          | send _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .other _ hne (ih (L₁ _ m) (.send _ L₂) (hD _ m) hN₂)
          | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .other _ hne (ih (L₁ _ m) (.table s₂ upd₂ hf₂ hs₂ hk₂ L₂) (hD _ m) hN₂)
        | here =>
          cases L₂ with
          | send _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .here _
          | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .here _
        | noSlot h0 => exact .of_noSlot ((h0.interleaveLock hd hN₂).bind hKn) _

/-- **A slot of the second lockstep at distance `j` is met after `n + j` coins** by a strategy leading with the first,
which ends after `n` coins. -/
theorem SlotAt.lead_snd (hΦ : ∀ d b b', Dead b → Dead b' → Φ d b b' = 0)
    (hKn : ∀ i a b (hr : RPair R₁ R₂ i a b), NoSlot slot (hK i a b hr)) {n : ℕ} {g₁' : Game β₁} {g₂' : Game β₂}
    {σ : Strategy ((interleave g₁' g₂').bind f')} (hS : Lead f' Dead n g₁' g₂' σ) :
    ∀ {j : ℕ} {i₁ : I₁} {g₁ : Game α₁} (L₁ : Lock T D get₁ R₁ i₁ g₁ g₁') {i₂ : I₂} {g₂ : Game α₂}
      (L₂ : Lock T D get₂ R₂ i₂ g₂ g₂'), NoSlot slot L₁ → Dist slot j L₂ →
        SlotAt plur slot Φ (n + j) (Lock.bind hK (interleaveLock hd L₁ L₂)) σ := by
  induction hS with
  | dead hσ =>
      intro _ _ _ _ _ _ _ _ _
      exact .nul (forkE_of_reach plur slot hΦ _ _ hσ)
  | done =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ _ hD
      cases L₁ with
      | ret h₁ =>
        rw [Nat.zero_add]
        exact .of_dist ((hD.embedRight hd h₁).bind_noSlot hKn) _
  | last =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hN₁ _
      cases L₂ with
      | ret h₂ => exact .of_noSlot ((hN₁.interleaveLock hd (.ret h₂)).bind hKn) _
  | coinCoin _ ih =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hN₁ hD
      rw [Nat.add_right_comm _ 1 j]
      cases L₁ with
      | coin _ L₁ =>
        cases hN₁ with
        | coin _ hN₁ =>
          cases L₂ with
          | coin _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .coin _ fun c => ih c (L₁ c) (.coin _ L₂) (hN₁ c) hD
  | coinSend _ ih =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hN₁ hD
      rw [Nat.add_right_comm _ 1 j]
      cases L₁ with
      | coin _ L₁ =>
        cases hN₁ with
        | coin _ hN₁ =>
          cases L₂ with
          | send _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .coin _ fun c => ih c (L₁ c) (.send _ L₂) (hN₁ c) hD
          | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .coin _ fun c => ih c (L₁ c) (.table s₂ upd₂ hf₂ hs₂ hk₂ L₂) (hN₁ c) hD
  | sendCoin m _ ih =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hN₁ hD
      cases L₁ with
      | send _ L₁ =>
        cases hN₁ with
        | send _ hN₁ =>
          cases L₂ with
          | coin _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .send _ (ih (L₁ m) (.coin _ L₂) (hN₁ m) hD)
      | table s upd hf hs hk L₁ =>
        cases hN₁ with
        | table _ hne hN₁ =>
          cases L₂ with
          | coin _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .other _ hne (ih (L₁ _ m) (.coin _ L₂) (hN₁ _ m) hD)
  | sendSend m _ ih =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hN₁ hD
      cases L₁ with
      | send _ L₁ =>
        cases hN₁ with
        | send _ hN₁ =>
          cases L₂ with
          | send _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .send _ (ih (L₁ m) (.send _ L₂) (hN₁ m) hD)
          | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .send _ (ih (L₁ m) (.table s₂ upd₂ hf₂ hs₂ hk₂ L₂) (hN₁ m) hD)
      | table s upd hf hs hk L₁ =>
        cases hN₁ with
        | table _ hne hN₁ =>
          cases L₂ with
          | send _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .other _ hne (ih (L₁ _ m) (.send _ L₂) (hN₁ _ m) hD)
          | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .other _ hne (ih (L₁ _ m) (.table s₂ upd₂ hf₂ hs₂ hk₂ L₂) (hN₁ _ m) hD)

end Lead

end FlockSoundness.Game.Lock
