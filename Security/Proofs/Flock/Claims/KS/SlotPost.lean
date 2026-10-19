import Proofs.Flock.Claims.KS.SlotFirst

/-!
# A slot's coin count in a lockstep the strategy plays first, beside the games after it

A batch puts the lockstep at its last position in front (`leftSame`, `batchLock d L post []`). A strategy that advances
that lockstep's game at every choice until it ends (`Leads`) plays the games after it only once the lockstep has ended,
so it meets a slot of the lockstep after the coins the lockstep's own strategy does. The lockstep's table nodes then
choose at the strategy on the whole batch, which is not a function of the lockstep's own, so the count is asked at
every table choice (`SlotAtD`):

* `SlotAtD`: every path meets the slot's table node after exactly `j` coins, at every table choice, or meets no such
  node, or reaches only `Dead` outcomes; `SlotAtD.slotAt`: then `SlotAt` holds at every chooser and every `Φ` that
  vanishes on two `Dead` outcomes;
* `NoSlot.lockWithSame`, `NoSlot.leftSame`: the same game beside a lockstep with no table node of the slot has none;
* `Leads`, `Leads.reach`: the strategy plays the lockstep's game first, as a strategy of that game does, and reaches
  only outcomes whose first part that strategy reaches;
* `SlotAtD.leftSame`, `SlotAtD.batchLock_nil`: so a slot met after `j` coins there is met after `j` coins in
  `leftSame L h`, and in the batch with nothing in front.
-/

namespace FlockSoundness.Game.Lock

/-- **The strategy reaches `slot`'s table node after exactly `j` coins at every table choice**: each path ends, or
passes sends and other table nodes, whatever table they choose, and meets that node with no coin left, or has no table
node of the slot, or reaches only `Dead` outcomes. -/
inductive SlotAtD {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} (slot : ℕ)
    (Dead : β → Prop) : ℕ → {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Strategy g' → Prop
  | ret {j : ℕ} {i : I} {a : α} {b : β} (h : R i a b) (σ' : Strategy (.ret b)) :
      SlotAtD slot Dead j (Lock.ret h) σ'
  | coin {j : ℕ} {i : I} (C : Type) [Fintype C] [Nonempty C] {k : C → Game α} {k' : C → Game β}
      {L : (c : C) → Lock T D get R i (k c) (k' c)} {σ' : Strategy (.coin C k')} :
      (∀ c, SlotAtD slot Dead j (L c) (σ' c)) → SlotAtD slot Dead (j + 1) (Lock.coin C L) σ'
  | send {j : ℕ} {i : I} (M : Type) {k : M → Game α} {k' : M → Game β}
      {L : (m : M) → Lock T D get R i (k m) (k' m)} {σ' : Strategy (.send M k')} :
      SlotAtD slot Dead j (L σ'.1) σ'.2 → SlotAtD slot Dead j (Lock.send M L) σ'
  | other {j : ℕ} {i : I} (s : ℕ) {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i s = none} {hs : ∀ t d, get (upd t d) s = some (t, d)}
      {hk : ∀ t d s', s' ≠ s → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} {σ' : Strategy (.send D k')}
      (hne : s ≠ slot) :
      (∀ t, SlotAtD slot Dead j (L t σ'.1) σ'.2) → SlotAtD slot Dead j (Lock.table s upd hf hs hk L) σ'
  | here {i : I} {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i slot = none} {hs : ∀ t d, get (upd t d) slot = some (t, d)}
      {hk : ∀ t d s', s' ≠ slot → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} (σ' : Strategy (.send D k')) :
      SlotAtD slot Dead 0 (Lock.table slot upd hf hs hk L) σ'
  | dead {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} :
      (∀ b, Reach g' σ' b → Dead b) → SlotAtD slot Dead j L σ'
  | noSlot {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} :
      NoSlot slot L → SlotAtD slot Dead j L σ'

/-- **The strategy plays `g'` first, as `σ₁` does**: at every choice of `interleave g' h` it advances `g'`, with
`σ₁`'s messages, until `g'` ends (`done`) or from the start where `h` has ended (`last`). -/
inductive Leads {β γ : Type} : (g' : Game β) → (h : Game γ) → Strategy g' → Strategy (interleave g' h) → Prop
  | done {b : β} {h : Game γ} {σ₁ : Strategy (.ret b)} {σ : Strategy (interleave (.ret b) h)} :
      Leads (.ret b) h σ₁ σ
  | last {g' : Game β} {c : γ} (σ : Strategy (interleave g' (.ret c))) :
      Leads g' (.ret c) (Strategy.ofMap (·, c) g' (cast (congrArg Strategy (interleave_ret c g')) σ)) σ
  | coinCoin {C : Type} [Fintype C] [Nonempty C] {k : C → Game β} {C₂ : Type} [Fintype C₂] [Nonempty C₂]
      {k₂ : C₂ → Game γ} {σ₁ : (c : C) → Strategy (k c)} {τ : (c : C) → Strategy (interleave (k c) (.coin C₂ k₂))} :
      (∀ c, Leads (k c) (.coin C₂ k₂) (σ₁ c) (τ c)) → Leads (.coin C k) (.coin C₂ k₂) σ₁ ⟨true, τ⟩
  | coinSend {C : Type} [Fintype C] [Nonempty C] {k : C → Game β} {M₂ : Type} {k₂ : M₂ → Game γ}
      {σ₁ : (c : C) → Strategy (k c)} {τ : (c : C) → Strategy (interleave (k c) (.send M₂ k₂))} :
      (∀ c, Leads (k c) (.send M₂ k₂) (σ₁ c) (τ c)) → Leads (.coin C k) (.send M₂ k₂) σ₁ ⟨true, τ⟩
  | sendCoin {M : Type} {k : M → Game β} {C₂ : Type} [Fintype C₂] [Nonempty C₂] {k₂ : C₂ → Game γ} (m : M)
      {σ₁ : Strategy (k m)} {τ : Strategy (interleave (k m) (.coin C₂ k₂))} :
      Leads (k m) (.coin C₂ k₂) σ₁ τ → Leads (.send M k) (.coin C₂ k₂) ⟨m, σ₁⟩ ⟨true, ⟨m, τ⟩⟩
  | sendSend {M : Type} {k : M → Game β} {M₂ : Type} {k₂ : M₂ → Game γ} (m : M)
      {σ₁ : Strategy (k m)} {τ : Strategy (interleave (k m) (.send M₂ k₂))} :
      Leads (k m) (.send M₂ k₂) σ₁ τ → Leads (.send M k) (.send M₂ k₂) ⟨m, σ₁⟩ ⟨true, ⟨m, τ⟩⟩

theorem reach_map {β β₂ : Type} (f : β → β₂) {x : β₂} :
    ∀ (g : Game β) (σ : Strategy (g.map f)), Reach (g.map f) σ x → ∃ y, Reach g (Strategy.ofMap f g σ) y ∧ x = f y
  | .ret b, _, h => ⟨b, rfl, h.symm⟩
  | .send _ k, σ, h => reach_map f (k σ.1) σ.2 h
  | @Game.coin _ _ _ _ k, σ, ⟨c, h⟩ => by
      obtain ⟨y, hy, rfl⟩ := reach_map f (k c) (σ c) h
      exact ⟨y, ⟨c, hy⟩, rfl⟩

theorem reach_cast {β : Type} {g₁ g₂ : Game β} (e : g₁ = g₂) (σ : Strategy g₁) {x : β}
    (h : Reach g₁ σ x) : Reach g₂ (cast (congrArg Strategy e) σ) x := by
  subst e
  exact h

/-- **A strategy playing `g'` first reaches only outcomes whose first part `σ₁` reaches.** -/
theorem Leads.reach {β γ : Type} {g' : Game β} {h : Game γ} {σ₁ : Strategy g'} {σ : Strategy (interleave g' h)}
    (hS : Leads g' h σ₁ σ) : ∀ {x : β × γ}, Reach _ σ x → Reach g' σ₁ x.1 := by
  induction hS with
  | @done b h _ σ =>
      intro x hx
      obtain ⟨y, _, rfl⟩ := reach_map (fun c => (b, c)) h σ hx
      rfl
  | @last g' c σ =>
      intro x hx
      obtain ⟨y, hy, rfl⟩ := reach_map (·, c) g' _ (reach_cast (interleave_ret c g') σ hx)
      exact hy
  | coinCoin _ ih =>
      rintro x ⟨c, hx⟩
      exact ⟨c, ih c hx⟩
  | coinSend _ ih =>
      rintro x ⟨c, hx⟩
      exact ⟨c, ih c hx⟩
  | sendCoin m _ ih => exact fun hx => ih hx
  | sendSend m _ ih => exact fun hx => ih hx

section Post

variable {T D I α β γ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} {slot : ℕ}

/-- **A chooser-free count gives the count at every chooser**, where `Φ` vanishes on two `Dead` outcomes. -/
theorem SlotAtD.slotAt {Dead : β → Prop} {plur : Chooser T D β} {Φ : D → β → β → ℝ}
    (hΦ : ∀ d b b', Dead b → Dead b' → Φ d b b' = 0) {j : ℕ} {i : I} {g : Game α} {g' : Game β}
    {L : Lock T D get R i g g'} {σ' : Strategy g'} (hs : SlotAtD slot Dead j L σ') : SlotAt plur slot Φ j L σ' := by
  induction hs with
  | ret hr σ' => exact .ret hr σ'
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne (ih _)
  | here σ' => exact .here σ'
  | dead hσ => exact .nul (forkE_of_reach plur slot hΦ _ _ hσ)
  | noSlot h0 => exact .of_noSlot h0 _

theorem NoSlot.lockWithSame {R₂ : I → α × γ → β × γ → Prop} {i : I} {adv : Game γ → Game (α × γ)}
    {adv' : Game γ → Game (β × γ)} {fin : γ → Game (α × γ)} {fin' : γ → Game (β × γ)}
    {hadv : ∀ h : Game γ, Lock T D get R₂ i (adv h) (adv' h)} {hfin : ∀ c : γ, Lock T D get R₂ i (fin c) (fin' c)}
    (hA : ∀ h, NoSlot slot (hadv h)) (hF : ∀ c, NoSlot slot (hfin c)) :
    ∀ h : Game γ, NoSlot slot (Lock.lockWithSame hadv hfin h)
  | .ret c => hF c
  | .send M k => by
      simp only [Lock.lockWithSame]
      refine .send Bool fun b => ?_
      cases b
      exacts [.send M fun m => NoSlot.lockWithSame hA hF (k m), hA _]
  | @Game.coin _ C _ _ k => by
      simp only [Lock.lockWithSame]
      refine .send Bool fun b => ?_
      cases b
      exacts [.coin C fun c => NoSlot.lockWithSame hA hF (k c), hA _]

/-- **The same game beside a lockstep with no table node of the slot has none.** -/
theorem NoSlot.leftSame {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (hL : NoSlot slot L) :
    ∀ h : Game γ, NoSlot slot (leftSame L h) := by
  induction hL with
  | ret hr => exact fun h => NoSlot.map _ _ _ (noSlot_same h)
  | coin C hc ih =>
      exact NoSlot.lockWithSame (fun h' => .coin C fun c => ih c h') fun _ => NoSlot.map _ _ _ (.coin C hc)
  | send M hm ih =>
      exact NoSlot.lockWithSame (fun h' => .send M fun m => ih m h') fun _ => NoSlot.map _ _ _ (.send M hm)
  | table s hne ht ih =>
      exact NoSlot.lockWithSame (fun h' => .table s hne fun t d => ih t d h')
        fun _ => NoSlot.map _ _ _ (.table s hne ht)

variable {Dead : β → Prop} {plur : Chooser T D (β × γ)} {Φ : D → β × γ → β × γ → ℝ}

/-- **A slot met after `j` coins at every table choice by `σ₁` is met after `j` coins beside `h`** by a strategy
playing the lockstep first as `σ₁` does, at every chooser and every `Φ` that vanishes where both first parts are
`Dead`. -/
theorem SlotAtD.leftSame (hΦ : ∀ d b b', Dead b.1 → Dead b'.1 → Φ d b b' = 0) {g' : Game β} {h : Game γ}
    {σ₁ : Strategy g'} {σ : Strategy (interleave g' h)} (hS : Leads g' h σ₁ σ) :
    ∀ {j : ℕ} {i : I} {g : Game α} (L : Lock T D get R i g g'), SlotAtD slot Dead j L σ₁ →
      SlotAt plur slot Φ j (Lock.leftSame L h) σ := by
  have hdead : ∀ {g' : Game β} {h : Game γ} {σ₁ : Strategy g'} {σ : Strategy (interleave g' h)}, Leads g' h σ₁ σ →
      ∀ {j : ℕ} {i : I} {g : Game α} (L : Lock T D get R i g g'), (∀ b, Reach g' σ₁ b → Dead b) →
        SlotAt plur slot Φ j (Lock.leftSame L h) σ :=
    fun hS _ _ _ L hr => .nul (forkE_of_reach plur slot hΦ _ _ fun x hx => hr _ (hS.reach hx))
  have hno : ∀ {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} {j : ℕ} {i : I} {g : Game α}
      {L : Lock T D get R i g g'}, NoSlot slot L → SlotAt plur slot Φ j (Lock.leftSame L h) σ :=
    fun h0 => .of_noSlot (h0.leftSame _) _
  induction hS with
  | @done b h _ _ =>
      intro _ _ _ L _
      cases L with
      | ret hr => exact .of_noSlot (NoSlot.map _ _ _ (noSlot_same h)) _
  | @last g' c σ =>
      intro _ _ _ L hD
      exact SlotAt.leftSame_ret c L σ (hD.slotAt fun d b b' hb hb' => hΦ d (b, c) (b', c) hb hb')
  | coinCoin hτ ih =>
      intro _ _ _ L hD
      cases hD with
      | coin _ hD =>
          simp only [Lock.leftSame, Lock.lockWithSame]
          refine .send Bool ?_
          exact .coin _ fun c => ih c _ (hD c)
      | dead hr => exact hdead (.coinCoin hτ) _ hr
      | noSlot h0 => exact hno h0
  | coinSend hτ ih =>
      intro _ _ _ L hD
      cases hD with
      | coin _ hD =>
          simp only [Lock.leftSame, Lock.lockWithSame]
          refine .send Bool ?_
          exact .coin _ fun c => ih c _ (hD c)
      | dead hr => exact hdead (.coinSend hτ) _ hr
      | noSlot h0 => exact hno h0
  | sendCoin m hτ ih =>
      intro _ _ _ L hD
      cases hD with
      | send _ hD =>
          simp only [Lock.leftSame, Lock.lockWithSame]
          refine .send Bool ?_
          exact .send _ (ih _ hD)
      | other s hne hD =>
          simp only [Lock.leftSame, Lock.lockWithSame]
          refine .send Bool ?_
          exact .other _ hne (ih _ (hD _))
      | here =>
          simp only [Lock.leftSame, Lock.lockWithSame]
          refine .send Bool ?_
          exact .here _
      | dead hr => exact hdead (.sendCoin m hτ) _ hr
      | noSlot h0 => exact hno h0
  | sendSend m hτ ih =>
      intro _ _ _ L hD
      cases hD with
      | send _ hD =>
          simp only [Lock.leftSame, Lock.lockWithSame]
          refine .send Bool ?_
          exact .send _ (ih _ hD)
      | other s hne hD =>
          simp only [Lock.leftSame, Lock.lockWithSame]
          refine .send Bool ?_
          exact .other _ hne (ih _ (hD _))
      | here =>
          simp only [Lock.leftSame, Lock.lockWithSame]
          refine .send Bool ?_
          exact .here _
      | dead hr => exact hdead (.sendSend m hτ) _ hr
      | noSlot h0 => exact hno h0

end Post

/-- **A slot met after `j` coins at every table choice by `σ₁` is met after `j` coins in the batch with nothing in
front** by a strategy playing the lockstep first as `σ₁` does, at every chooser and every `Φ` that vanishes on two
outcome lists whose heads are `Dead`. -/
theorem SlotAtD.batchLock_nil {T D I ζ : Type} {get : I → ℕ → Option (T × D)} {R : I → ζ → ζ → Prop} {slot j : ℕ}
    {Dead : ζ → Prop} {plur : Chooser T D (List ζ)} {Φ : D → List ζ → List ζ → ℝ}
    (hΦ : ∀ e x x' r r', Dead x → Dead x' → Φ e (x :: r) (x' :: r') = 0) (d : ζ) {i : I} {g g' : Game ζ}
    (L : Lock T D get R i g g') (post : List (Game ζ)) (σ₁ : Strategy g') (σ' : Strategy (batch (g' :: post)))
    (hS : Leads g' (batch post) σ₁
      (Strategy.ofMap (fun x : ζ × List ζ => x.1 :: x.2) (g'.interleave (batch post)) σ'))
    (hD : SlotAtD slot Dead j L σ₁) : SlotAt plur slot Φ j (batchLock d L post []) σ' := by
  have key := SlotAt.map (R₂ := fun i a b => R i (a.getD 0 d) (b.getD 0 d))
    (fun x : ζ × List ζ => x.1 :: x.2) (fun x : ζ × List ζ => x.1 :: x.2) (fun _ _ _ h => h.1)
    (hD.leftSame (plur := plurMap (fun x : ζ × List ζ => x.1 :: x.2) plur)
      (Φ := fun e b b' => Φ e (b.1 :: b.2) (b'.1 :: b'.2)) (fun e b b' hb hb' => hΦ e _ _ _ _ hb hb') hS L)
  exact Strategy.map_ofMap (fun x : ζ × List ζ => x.1 :: x.2) (g'.interleave (batch post)) σ' ▸ key

end FlockSoundness.Game.Lock
