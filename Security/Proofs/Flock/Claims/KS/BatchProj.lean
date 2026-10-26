import Proofs.Flock.Claims.KS.SlotBatch

/-!
# A game's strategy out of an interleaving's

`FollowR` and `FollowL` ask, once the slot is behind, that the run be dead wherever the followed strategy's is. The
followed strategy here is read off the interleaving's: `projR` is the rest's strategy out of a strategy of
`interleave h g'`, `projL` the first game's out of one of `interleave g' h`. Each plays the moves the interleaving's
strategy makes in its game; at the other game's coins it takes a branch below which it still reaches an outcome off
`Dead'`, if there is one (`liveAt`). So whatever the interleaving's strategy reaches off `Dead'` in that game, the
projection reaches something off `Dead'` too, and the `spent` case holds of every strategy:

* `projR`, `projR_live`, `followR_none`: the rest's strategy, and `FollowR` with the slot behind;
* `projL`, `projL_live`, `followL_none`, `reach_projL`: the first game's, `FollowL` with the slot behind, and what it
  reaches.
-/

namespace FlockSoundness.Game.Lock

variable {β γ : Type}

theorem reach_exists : ∀ (g : Game β) (σ : Strategy g), ∃ x, Reach g σ x
  | .ret b, _ => ⟨b, rfl⟩
  | .send _ k, σ => reach_exists (k σ.1) σ.2
  | @Game.coin _ C _ _ k, σ => by
      obtain ⟨x, hx⟩ := reach_exists (k (Classical.arbitrary C)) (σ _)
      exact ⟨x, _, hx⟩

theorem reach_ofMap_map {A : Type} (f : A → β) : ∀ (g : Game A) (σ : Strategy (g.map f)) (a : A),
    Reach g (Strategy.ofMap f g σ) a → Reach (g.map f) σ (f a)
  | .ret _, _, _, h => congrArg f h
  | .send _ k, σ, a, h => reach_ofMap_map f (k σ.1) σ.2 a h
  | @Game.coin _ _ _ _ k, σ, a, ⟨c, h⟩ => ⟨c, reach_ofMap_map f (k c) (σ c) a h⟩

/-- **A branch that stays live**: `f` at a coin below which it still reaches an outcome off `Dead'`, if there is
one. -/
noncomputable def liveAt (Dead' : β → Prop) {C : Type} [Nonempty C] {g : Game β} (f : C → Strategy g) : Strategy g :=
  f (Classical.epsilon fun c => ∃ y, Reach g (f c) y ∧ ¬ Dead' y)

theorem liveAt_live {Dead' : β → Prop} {C : Type} [Nonempty C] {g : Game β} (f : C → Strategy g) {c : C}
    (h : ∃ y, Reach g (f c) y ∧ ¬ Dead' y) : ∃ y, Reach g (liveAt Dead' f) y ∧ ¬ Dead' y :=
  Classical.epsilon_spec (p := fun c => ∃ y, Reach g (f c) y ∧ ¬ Dead' y) ⟨c, h⟩

section ProjR

variable (Dead' : β → Prop)

/-- **The rest's strategy once the game in front's next step is fixed**: the rest's moves as `σ` makes them, `advR`
where the game in front moves. -/
noncomputable def projRWith {adv : Game β → Game (γ × β)} {fin : β → Game (γ × β)}
    (advR : (g' : Game β) → Strategy (adv g') → Strategy g') :
    (g' : Game β) → Strategy (interleaveWith adv fin g') → Strategy g'
  | .ret _, _ => PUnit.unit
  | .send _ k₂, σ => match σ with
    | ⟨true, σ'⟩ => advR _ σ'
    | ⟨false, σ'⟩ => ⟨σ'.1, projRWith advR (k₂ σ'.1) σ'.2⟩
  | @Game.coin _ _ _ _ k₂, σ => match σ with
    | ⟨true, σ'⟩ => advR _ σ'
    | ⟨false, σ'⟩ => fun c => projRWith advR (k₂ c) (σ' c)

/-- **The rest's strategy out of a strategy of `interleave h g'`**: the rest's moves as `σ` makes them; at `h`'s coins
a branch below which the rest still reaches an outcome off `Dead'`, if any. -/
noncomputable def projR : (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Strategy g'
  | .ret a, g', σ => Strategy.ofMap (a, ·) g' σ
  | .send M k, g', σ => projRWith (adv := fun g₂ => .send M fun m => interleave (k m) g₂)
      (fin := fun b => (Game.send M k).map fun a => (a, b)) (fun g₂ σ' => projR (k σ'.1) g₂ σ'.2) g' σ
  | @Game.coin _ C _ iN k, g', σ => projRWith (adv := fun g₂ => @Game.coin _ C _ iN fun c => interleave (k c) g₂)
      (fin := fun b => (@Game.coin _ C _ iN k).map fun a => (a, b))
      (fun g₂ σ' => liveAt Dead' fun c => projR (k c) g₂ (σ' c)) g' σ

variable {Dead'}

theorem projRWith_live {adv : Game β → Game (γ × β)} {fin : β → Game (γ × β)}
    {advR : (g' : Game β) → Strategy (adv g') → Strategy g'}
    (hadv : ∀ g₂ σ' x, Reach (adv g₂) σ' x → ¬ Dead' x.2 → ∃ y, Reach g₂ (advR g₂ σ') y ∧ ¬ Dead' y)
    (hfin : ∀ b σ' x, Reach (fin b) σ' x → x.2 = b) :
    ∀ (g' : Game β) (σ : Strategy (interleaveWith adv fin g')) (x : γ × β), Reach _ σ x → ¬ Dead' x.2 →
      ∃ y, Reach g' (projRWith advR g' σ) y ∧ ¬ Dead' y
  | .ret b, σ, x, hx, hd => ⟨b, rfl, hfin b σ x hx ▸ hd⟩
  | .send _ _, ⟨true, σ'⟩, x, hx, hd => hadv _ σ' x hx hd
  | .send _ k₂, ⟨false, σ'⟩, x, hx, hd => projRWith_live hadv hfin (k₂ σ'.1) σ'.2 x hx hd
  | @Game.coin _ _ _ _ _, ⟨true, σ'⟩, x, hx, hd => hadv _ σ' x hx hd
  | @Game.coin _ _ _ _ k₂, ⟨false, σ'⟩, x, ⟨c, hx⟩, hd => by
      obtain ⟨y, hy, hy'⟩ := projRWith_live hadv hfin (k₂ c) (σ' c) x hx hd
      exact ⟨y, ⟨c, hy⟩, hy'⟩

theorem fin_reach {α₁ : Type} (g : Game α₁) (b : β) (σ' : Strategy (g.map fun a => (a, b))) (x : α₁ × β)
    (hx : Reach _ σ' x) : x.2 = b := by
  obtain ⟨y, -, rfl⟩ := reach_map _ g σ' hx
  rfl

/-- **What `σ` reaches off `Dead'` in the rest, `projR` reaches something off `Dead'` of.** -/
theorem projR_live : ∀ (h : Game γ) (g' : Game β) (σ : Strategy (interleave h g')) (x : γ × β), Reach _ σ x →
    ¬ Dead' x.2 → ∃ y, Reach g' (projR Dead' h g' σ) y ∧ ¬ Dead' y
  | .ret a, g', σ, x, hx, hd => by
      obtain ⟨y, hy, rfl⟩ := reach_map (a, ·) g' σ hx
      exact ⟨y, hy, hd⟩
  | .send M k, g', σ, x, hx, hd =>
      projRWith_live (adv := fun g₂ => .send M fun m => interleave (k m) g₂)
        (fun g₂ σ' x hx hd => projR_live (k σ'.1) g₂ σ'.2 x hx hd)
        (fun b σ' x hx => fin_reach (.send M k) b σ' x hx) g' σ x hx hd
  | @Game.coin _ C _ iN k, g', σ, x, hx, hd =>
      projRWith_live (adv := fun g₂ => @Game.coin _ C _ iN fun c => interleave (k c) g₂)
        (fun g₂ σ' x hx hd => by
          obtain ⟨c, hx⟩ := hx
          exact liveAt_live _ (projR_live (k c) g₂ (σ' c) x hx hd))
        (fun b σ' x hx => fin_reach (@Game.coin _ C _ iN k) b σ' x hx) g' σ x hx hd

/-- **With the slot behind, every strategy's rest follows its `projR`.** -/
theorem followR_none {Dead : γ × β → Prop} (hD : ∀ x : γ × β, Dead' x.2 → Dead x) (h : Game γ) (g' : Game β)
    (σ : Strategy (interleave h g')) : FollowR Dead Dead' none h g' σ (projR Dead' h g' σ) :=
  .spent fun hτ x hx => by
    by_contra hn
    obtain ⟨y, hy, hy'⟩ := projR_live h g' σ x hx fun hd => hn (hD x hd)
    exact hy' (hτ y hy)

end ProjR

section ProjL

variable (Dead' : β → Prop)

/-- **The first game's strategy once its next step is fixed** (`interleaveWith`): the first game's moves as `advL`
reads them, the strategy through the pairing once the second has ended (`finL`), and at the second's coins a branch
below which the first still reaches an outcome off `Dead'`, if any. -/
noncomputable def projLWith {g' : Game β} {adv : Game γ → Game (β × γ)} {fin : γ → Game (β × γ)}
    (advL : (h : Game γ) → Strategy (adv h) → Strategy g') (finL : (c : γ) → Strategy (fin c) → Strategy g') :
    (h : Game γ) → Strategy (interleaveWith adv fin h) → Strategy g'
  | .ret c, σ => finL c σ
  | .send _ k₂, σ => match σ with
    | ⟨true, σ'⟩ => advL _ σ'
    | ⟨false, σ'⟩ => projLWith advL finL (k₂ σ'.1) σ'.2
  | @Game.coin _ _ _ _ k₂, σ => match σ with
    | ⟨true, σ'⟩ => advL _ σ'
    | ⟨false, σ'⟩ => liveAt Dead' fun c => projLWith advL finL (k₂ c) (σ' c)

/-- **The first game's strategy out of a strategy of `interleave g' h`**: its moves as `σ` makes them; at `h`'s coins
a branch below which it still reaches an outcome off `Dead'`, if any. -/
noncomputable def projL : (g' : Game β) → (h : Game γ) → Strategy (interleave g' h) → Strategy g'
  | .ret _, _, _ => PUnit.unit
  | .send M k, h, σ => projLWith Dead' (g' := .send M k) (adv := fun h₂ => .send M fun m => interleave (k m) h₂)
      (fin := fun c => (Game.send M k).map fun a => (a, c)) (fun h₂ σ' => ⟨σ'.1, projL (k σ'.1) h₂ σ'.2⟩)
      (fun c σ' => Strategy.ofMap (·, c) (.send M k) σ') h σ
  | @Game.coin _ C _ iN k, h, σ => projLWith Dead' (g' := @Game.coin _ C _ iN k)
      (adv := fun h₂ => @Game.coin _ C _ iN fun c => interleave (k c) h₂)
      (fin := fun c => (@Game.coin _ C _ iN k).map fun a => (a, c)) (fun h₂ σ' => fun c => projL (k c) h₂ (σ' c))
      (fun c σ' => Strategy.ofMap (·, c) (@Game.coin _ C _ iN k) σ') h σ

variable {Dead'}

theorem projLWith_live {g' : Game β} {adv : Game γ → Game (β × γ)} {fin : γ → Game (β × γ)}
    {advL : (h : Game γ) → Strategy (adv h) → Strategy g'} {finL : (c : γ) → Strategy (fin c) → Strategy g'}
    (hadv : ∀ h₂ σ' x, Reach (adv h₂) σ' x → ¬ Dead' x.1 → ∃ y, Reach g' (advL h₂ σ') y ∧ ¬ Dead' y)
    (hfin : ∀ c σ' x, Reach (fin c) σ' x → ¬ Dead' x.1 → ∃ y, Reach g' (finL c σ') y ∧ ¬ Dead' y) :
    ∀ (h : Game γ) (σ : Strategy (interleaveWith adv fin h)) (x : β × γ), Reach _ σ x → ¬ Dead' x.1 →
      ∃ y, Reach g' (projLWith Dead' advL finL h σ) y ∧ ¬ Dead' y
  | .ret c, σ, x, hx, hd => hfin c σ x hx hd
  | .send _ _, ⟨true, σ'⟩, x, hx, hd => hadv _ σ' x hx hd
  | .send _ k₂, ⟨false, σ'⟩, x, hx, hd => projLWith_live hadv hfin (k₂ σ'.1) σ'.2 x hx hd
  | @Game.coin _ _ _ _ _, ⟨true, σ'⟩, x, hx, hd => hadv _ σ' x hx hd
  | @Game.coin _ _ _ _ k₂, ⟨false, σ'⟩, x, ⟨c, hx⟩, hd =>
      liveAt_live (fun c => projLWith Dead' advL finL (k₂ c) (σ' c)) (projLWith_live hadv hfin (k₂ c) (σ' c) x hx hd)

theorem finL_live {g' : Game β} (c : γ) (σ' : Strategy (g'.map fun a => (a, c))) (x : β × γ) (hx : Reach _ σ' x)
    (hd : ¬ Dead' x.1) : ∃ y, Reach g' (Strategy.ofMap (·, c) g' σ') y ∧ ¬ Dead' y := by
  obtain ⟨y, hy, rfl⟩ := reach_map _ g' σ' hx
  exact ⟨y, hy, hd⟩

/-- **What `σ` reaches off `Dead'` in the first game, `projL` reaches something off `Dead'` of.** -/
theorem projL_live : ∀ (g' : Game β) (h : Game γ) (σ : Strategy (interleave g' h)) (x : β × γ), Reach _ σ x →
    ¬ Dead' x.1 → ∃ y, Reach g' (projL Dead' g' h σ) y ∧ ¬ Dead' y
  | .ret a, h, σ, x, hx, hd => by
      obtain ⟨y, -, rfl⟩ := reach_map (a, ·) h σ hx
      exact ⟨a, rfl, hd⟩
  | .send M k, h, σ, x, hx, hd =>
      projLWith_live (adv := fun h₂ => .send M fun m => interleave (k m) h₂)
        (fun h₂ σ' x hx hd => projL_live (k σ'.1) h₂ σ'.2 x hx hd)
        (fun c σ' x hx hd => finL_live c σ' x hx hd) h σ x hx hd
  | @Game.coin _ C _ iN k, h, σ, x, hx, hd =>
      projLWith_live (adv := fun h₂ => @Game.coin _ C _ iN fun c => interleave (k c) h₂)
        (fun h₂ σ' x hx hd => by
          obtain ⟨c, hx⟩ := hx
          obtain ⟨y, hy, hy'⟩ := projL_live (k c) h₂ (σ' c) x hx hd
          exact ⟨y, ⟨c, hy⟩, hy'⟩)
        (fun c σ' x hx hd => finL_live c σ' x hx hd) h σ x hx hd

/-- **With the slot behind, every strategy's first game follows its `projL`.** -/
theorem followL_none {Dead : β × γ → Prop} (hD : ∀ x : β × γ, Dead' x.1 → Dead x) (g' : Game β) (h : Game γ)
    (σ : Strategy (interleave g' h)) : FollowL Dead Dead' none g' h σ (projL Dead' g' h σ) :=
  .spent fun hτ x hx => by
    by_contra hn
    obtain ⟨y, hy, hy'⟩ := projL_live g' h σ x hx fun hd => hn (hD x hd)
    exact hy' (hτ y hy)

theorem reach_projLWith {g' : Game β} {adv : Game γ → Game (β × γ)} {fin : γ → Game (β × γ)}
    {advL : (h : Game γ) → Strategy (adv h) → Strategy g'} {finL : (c : γ) → Strategy (fin c) → Strategy g'} {y : β}
    (hadv : ∀ h₂ σ', Reach g' (advL h₂ σ') y → ∃ x, Reach (adv h₂) σ' x ∧ x.1 = y)
    (hfin : ∀ c σ', Reach g' (finL c σ') y → ∃ x, Reach (fin c) σ' x ∧ x.1 = y) :
    ∀ (h : Game γ) (σ : Strategy (interleaveWith adv fin h)), Reach g' (projLWith Dead' advL finL h σ) y →
      ∃ x, Reach _ σ x ∧ x.1 = y
  | .ret c, σ, hy => hfin c σ hy
  | .send _ _, ⟨true, σ'⟩, hy => hadv _ σ' hy
  | .send _ k₂, ⟨false, σ'⟩, hy => reach_projLWith hadv hfin (k₂ σ'.1) σ'.2 hy
  | @Game.coin _ _ _ _ _, ⟨true, σ'⟩, hy => hadv _ σ' hy
  | @Game.coin _ C _ iN k₂, ⟨false, σ'⟩, hy => by
      obtain ⟨x, hx, rfl⟩ := reach_projLWith hadv hfin (k₂ (@Classical.epsilon C iN fun c =>
        ∃ y, Reach g' (projLWith Dead' advL finL (k₂ c) (σ' c)) y ∧ ¬ Dead' y)) (σ' _) hy
      exact ⟨x, ⟨_, hx⟩, rfl⟩

theorem reach_finL {g' : Game β} (c : γ) (σ' : Strategy (g'.map fun a => (a, c))) {y : β}
    (hy : Reach g' (Strategy.ofMap (·, c) g' σ') y) : ∃ x, Reach _ σ' x ∧ x.1 = y :=
  ⟨(y, c), reach_ofMap_map _ g' σ' y hy, rfl⟩

/-- **`projL` reaches only first parts of what `σ` reaches.** -/
theorem reach_projL : ∀ (g' : Game β) (h : Game γ) (σ : Strategy (interleave g' h)) {y : β},
    Reach g' (projL Dead' g' h σ) y → ∃ x, Reach _ σ x ∧ x.1 = y
  | .ret a, h, σ, y, hy => by
      obtain ⟨x, hx⟩ := reach_exists (interleave (.ret a) h) σ
      obtain ⟨z, -, rfl⟩ := reach_map (a, ·) h σ hx
      exact ⟨_, hx, hy⟩
  | .send M k, h, σ, y, hy =>
      reach_projLWith (adv := fun h₂ => .send M fun m => interleave (k m) h₂)
        (fun h₂ σ' hy => reach_projL (k σ'.1) h₂ σ'.2 hy) (fun c σ' hy => reach_finL c σ' hy) h σ hy
  | @Game.coin _ C _ iN k, h, σ, y, hy =>
      reach_projLWith (adv := fun h₂ => @Game.coin _ C _ iN fun c => interleave (k c) h₂)
        (fun h₂ σ' hy => by
          obtain ⟨c, hy⟩ := hy
          obtain ⟨x, hx, rfl⟩ := reach_projL (k c) h₂ (σ' c) hy
          exact ⟨x, ⟨c, hx⟩, rfl⟩)
        (fun c σ' hy => reach_finL c σ' hy) h σ hy

end ProjL

end FlockSoundness.Game.Lock
