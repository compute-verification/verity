import Proofs.Flock.Claims.KS.SlotPost
import Proofs.Flock.Soundness.Discharge.ZkExec.Shape

/-!
# A slot's distance on the paths that can end well

`Dist` asks every path of a lockstep to meet a slot after one number of coins. A game whose shape is fixed only on the
runs that can end well (`ExactOn`) has paths that end early, after fewer coins, and then meet the slot sooner. Where
every outcome past such a point is `Bad`, the distance can stop asking there:

* `DistOn slot Bad j L`: `Dist`, or every outcome of the game from there on is `Bad`; `SlotAtD.of_distOn`: then the
  chooser-free count holds against every strategy, where `Bad` outcomes are `Dead`; `Dist.distOn`,
  `DistOn.bind_noSlot`, `DistOn.embedLeft`, `DistOn.embedRight`, `DistOn.interleaveLock_ret`: a distance is one, and
  sequencing and pairing keep one;
* `DistOn.bind_on`: a lockstep with no table node of the slot whose game has an exact shape on the runs ending in `P`,
  then a continuation at distance `j`, is at distance `j` plus the shape's coins, the runs past a `¬ P` outcome `Bad`;
* `SlotAtD.lead_fst_on`, `SlotAtD.lead_snd_on`: `SlotAtD.lead_fst` and `SlotAtD.lead_snd` at such a distance, a `Bad`
  part's outcomes `Dead` after the continuation.
-/

namespace FlockSoundness.Game

theorem Out.of_interleaveWith {A B : Type} {g₁ : Game A} {adv : Game B → Game (A × B)}
    {fin : B → Game (A × B)} (hadv : ∀ g₂ c, Out (adv g₂) c → Out g₁ c.1 ∧ Out g₂ c.2)
    (hfin : ∀ b c, Out (fin b) c → Out g₁ c.1 ∧ c.2 = b) :
    ∀ {g₂ : Game B} {c : A × B}, Out (interleaveWith adv fin g₂) c → Out g₁ c.1 ∧ Out g₂ c.2
  | .ret b, c, h => ⟨(hfin b c h).1, by rw [(hfin b c h).2]; exact .ret b⟩
  | .send _ _, c, .send first h => by
      cases first
      · cases h with
        | send m h => exact ⟨(Out.of_interleaveWith hadv hfin h).1, .send m (Out.of_interleaveWith hadv hfin h).2⟩
      · exact hadv _ c h
  | @Game.coin _ _ _ _ _, c, .send first h => by
      cases first
      · cases h with
        | coin x h => exact ⟨(Out.of_interleaveWith hadv hfin h).1, .coin x (Out.of_interleaveWith hadv hfin h).2⟩
      · exact hadv _ c h

/-- **An outcome of two interleaved games pairs an outcome of each.** -/
theorem Out.of_interleave {A B : Type} :
    ∀ {g₁ : Game A} {g₂ : Game B} {c : A × B}, Out (interleave g₁ g₂) c → Out g₁ c.1 ∧ Out g₂ c.2
  | .ret a, _, _, h => by
      obtain ⟨b, hb, rfl⟩ := Out.of_map h
      exact ⟨.ret a, hb⟩
  | .send _ _, _, _, h =>
      Out.of_interleaveWith
        (fun _ _ h => by
          cases h with
          | send m h => exact ⟨.send m (Out.of_interleave h).1, (Out.of_interleave h).2⟩)
        (fun _ _ h => by obtain ⟨a, ha, rfl⟩ := Out.of_map h; exact ⟨ha, rfl⟩) h
  | @Game.coin _ _ _ _ _, _, _, h =>
      Out.of_interleaveWith
        (fun _ _ h => by
          cases h with
          | coin x h => exact ⟨.coin x (Out.of_interleave h).1, (Out.of_interleave h).2⟩)
        (fun _ _ h => by obtain ⟨a, ha, rfl⟩ := Out.of_map h; exact ⟨ha, rfl⟩) h

/-- **An outcome is the end of a run.** -/
theorem Out.playsT {β : Type} : ∀ {g : Game β} {b : β}, Out g b → ∃ ts, PlaysT g ts b
  | _, _, .ret b => ⟨[], .ret b⟩
  | _, _, .send m h => let ⟨_, hp⟩ := Out.playsT h; ⟨_, .send m hp⟩
  | _, _, .coin c h => let ⟨_, hp⟩ := Out.playsT h; ⟨_, .coin c hp⟩

namespace Lock

open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec

theorem Reach.out {β : Type} : ∀ {g : Game β} {σ : Strategy g} {b : β}, Reach g σ b → Out g b
  | .ret _, _, _, h => by cases h; exact .ret _
  | .send _ _, σ, _, h => .send σ.1 (Reach.out h)
  | @Game.coin _ _ _ _ _, _, _, ⟨c, h⟩ => .coin c (Reach.out h)

/-- **Every path meets `slot`'s table node after exactly `j` coins, or meets no such node, or reaches only `Bad`
outcomes** from some point on, whatever the messages and coins. -/
inductive DistOn {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} (slot : ℕ)
    (Bad : β → Prop) : ℕ → {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Prop
  | coin {j : ℕ} {i : I} (C : Type) [Fintype C] [Nonempty C] {k : C → Game α} {k' : C → Game β}
      {L : (c : C) → Lock T D get R i (k c) (k' c)} :
      (∀ c, DistOn slot Bad j (L c)) → DistOn slot Bad (j + 1) (Lock.coin C L)
  | send {j : ℕ} {i : I} (M : Type) {k : M → Game α} {k' : M → Game β}
      {L : (m : M) → Lock T D get R i (k m) (k' m)} :
      (∀ m, DistOn slot Bad j (L m)) → DistOn slot Bad j (Lock.send M L)
  | other {j : ℕ} {i : I} (s : ℕ) {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i s = none} {hs : ∀ t d, get (upd t d) s = some (t, d)}
      {hk : ∀ t d s', s' ≠ s → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} (hne : s ≠ slot) :
      (∀ t d, DistOn slot Bad j (L t d)) → DistOn slot Bad j (Lock.table s upd hf hs hk L)
  | here {i : I} {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i slot = none} {hs : ∀ t d, get (upd t d) slot = some (t, d)}
      {hk : ∀ t d s', s' ≠ slot → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} :
      DistOn slot Bad 0 (Lock.table slot upd hf hs hk L)
  | noSlot {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} :
      NoSlot slot L → DistOn slot Bad j L
  | bad {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} :
      (∀ b, Out g' b → Bad b) → DistOn slot Bad j L

section On

variable {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} {slot : ℕ} {Bad : β → Prop}

theorem Dist.distOn {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (h : Dist slot j L) :
    DistOn slot Bad j L := by
  induction h with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot h0

/-- **A distance on the good paths gives the chooser-free count against every strategy**, where `Bad` outcomes are
`Dead`. -/
theorem SlotAtD.of_distOn {Dead : β → Prop} (hB : ∀ b, Bad b → Dead b) {j : ℕ} {i : I} {g : Game α} {g' : Game β}
    {L : Lock T D get R i g g'} (h : DistOn slot Bad j L) (σ' : Strategy g') : SlotAtD slot Dead j L σ' := by
  induction h with
  | coin C _ ih => exact .coin C fun c => ih c (σ' c)
  | send M _ ih => exact .send M (ih _ _)
  | other s hne _ ih => exact .other s hne fun t => ih t _ _
  | here => exact .here σ'
  | noSlot h0 => exact .noSlot h0
  | bad hb => exact .dead fun b hr => hB b (hb b (Reach.out hr))

/-- **A continuation with no table node of the slot keeps a distance**, its outcomes after a `Bad` one `Bad'`. -/
theorem DistOn.bind_noSlot {α₂ β₂ : Type} {R₂ : I → α₂ → β₂ → Prop} {f : α → Game α₂} {f' : β → Game β₂}
    {Bad' : β₂ → Prop} (hB : ∀ b y, Bad b → Out (f' b) y → Bad' y) {j : ℕ}
    {h : ∀ i a b, R i a b → Lock T D get R₂ i (f a) (f' b)} (hK : ∀ i a b (hr : R i a b), NoSlot slot (h i a b hr))
    {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (hL : DistOn slot Bad j L) :
    DistOn slot Bad' j (Lock.bind h L) := by
  induction hL with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (h0.bind hK)
  | bad hb =>
      exact .bad fun y hy => by
        obtain ⟨b, hb', hy'⟩ := Out.of_bind hy
        exact hB b y (hb b hb') hy'

/-- **A distance after a shape exact on the runs ending in `P`, with no table node of the slot** adds the shape's
coins; the runs past a `¬ P` outcome reach only `Bad` ones. -/
theorem DistOn.bind_on {α₂ β₂ : Type} {R₂ : I → α₂ → β₂ → Prop} {f : α → Game α₂} {f' : β → Game β₂}
    {Bad₂ : β₂ → Prop} {P : β → Prop} {j : ℕ} {h : ∀ i a b, R i a b → Lock T D get R₂ i (f a) (f' b)}
    (hK : ∀ i a b (hr : R i a b), DistOn slot Bad₂ j (h i a b hr)) (hB : ∀ b y, ¬ P b → Out (f' b) y → Bad₂ y)
    {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (hL : NoSlot slot L) (σ : List Kind)
    (hσ : ExactOn g' σ P) : DistOn slot Bad₂ (j + coinsK σ) (Lock.bind h L) := by
  have hbad : ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (n : ℕ),
      (∀ b, Out g' b → ¬ P b) → DistOn slot Bad₂ n (Lock.bind h L) := fun _ _ hn =>
    .bad fun y hy => by
      obtain ⟨b, hb, hy'⟩ := Out.of_bind hy
      exact hB b y (hn b hb) hy'
  have hnot : ∀ {g' : Game β} {σ : List Kind}, ExactOn g' σ P →
      (∀ ts b, PlaysT g' ts b → ts.map kindOf ≠ σ) → ∀ b, Out g' b → ¬ P b := fun hσ hne b hb hp => by
    obtain ⟨ts, hts⟩ := hb.playsT
    exact hne ts b hts (hσ ts b hts hp)
  induction hL generalizing σ with
  | @ret _ a b hr =>
      by_cases hp : P b
      · have e := hσ [] b (.ret b) hp
        subst e
        exact hK _ _ _ hr
      · exact .bad fun y hy => hB b y hp hy
  | @coin _ C _ _ k k' L _ ih =>
      rcases σ with _ | ⟨_ | C', σ'⟩
      · exact hbad _ _ (hnot hσ fun ts b hp e => by cases hp; simp at e)
      · exact hbad _ _ (hnot hσ fun ts b hp e => by cases hp; simp [kindOf] at e)
      · have ht : ∀ c, ExactOn (k' c) σ' P := fun c ts b hp hb => by
          have e := hσ _ b (.coin c hp) hb
          simp only [List.map_cons, kindOf, List.cons.injEq] at e
          exact e.2
        simp only [coinsK, ← Nat.add_assoc]
        exact .coin C fun c => ih c σ' (ht c)
  | @send _ M k k' L _ ih =>
      rcases σ with _ | ⟨M' | _, σ'⟩
      · exact hbad _ _ (hnot hσ fun ts b hp e => by cases hp; simp at e)
      · have ht : ∀ m, ExactOn (k' m) σ' P := fun m ts b hp hb => by
          have e := hσ _ b (.send m hp) hb
          simp only [List.map_cons, kindOf, List.cons.injEq] at e
          exact e.2
        simp only [coinsK]
        exact .send M fun m => ih m σ' (ht m)
      · exact hbad _ _ (hnot hσ fun ts b hp e => by cases hp; simp [kindOf] at e)
  | @table _ s upd k k' hf hs hk L hne _ ih =>
      rcases σ with _ | ⟨M' | _, σ'⟩
      · exact hbad _ _ (hnot hσ fun ts b hp e => by cases hp; simp at e)
      · have ht : ∀ d, ExactOn (k' d) σ' P := fun d ts b hp hb => by
          have e := hσ _ b (.send d hp) hb
          simp only [List.map_cons, kindOf, List.cons.injEq] at e
          exact e.2
        simp only [coinsK]
        exact .other s hne fun t d => ih t d σ' (ht d)
      · exact hbad _ _ (hnot hσ fun ts b hp e => by cases hp; simp [kindOf] at e)

end On

section Interleave

variable {T D α₁ β₁ α₂ β₂ I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (T × D)} {get₂ : I₂ → ℕ → Option (T × D)}
  {R₁ : I₁ → α₁ → β₁ → Prop} {R₂ : I₂ → α₂ → β₂ → Prop} [Inhabited T] [Inhabited D] (hd : Lock.Disjoint get₁ get₂)
  {slot : ℕ}

include hd

theorem DistOn.embedLeft {Bad : β₁ → Prop} {Bad' : β₁ × β₂ → Prop} {i₂ : I₂} {a₂ : α₂} {b₂ : β₂}
    (h₂ : R₂ i₂ a₂ b₂) (hB : ∀ b, Bad b → Bad' (b, b₂)) {j : ℕ} {i₁ : I₁} {g₁ : Game α₁} {g₁' : Game β₁}
    {L : Lock T D get₁ R₁ i₁ g₁ g₁'} (hL : DistOn slot Bad j L) : DistOn slot Bad' j (embedLeft hd h₂ L) := by
  induction hL with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (h0.embedLeft hd h₂)
  | bad hb =>
      exact .bad fun y hy => by
        obtain ⟨b, hb', rfl⟩ := Out.of_map hy
        exact hB b (hb b hb')

theorem DistOn.embedRight {Bad : β₂ → Prop} {Bad' : β₁ × β₂ → Prop} {i₁ : I₁} {a₁ : α₁} {b₁ : β₁}
    (h₁ : R₁ i₁ a₁ b₁) (hB : ∀ b, Bad b → Bad' (b₁, b)) {j : ℕ} {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂}
    {L : Lock T D get₂ R₂ i₂ g₂ g₂'} (hL : DistOn slot Bad j L) : DistOn slot Bad' j (embedRight hd h₁ L) := by
  induction hL with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (h0.embedRight hd h₁)
  | bad hb =>
      exact .bad fun y hy => by
        obtain ⟨b, hb', rfl⟩ := Out.of_map hy
        exact hB b (hb b hb')

/-- **Interleaving with a finished lockstep keeps a distance on the good paths.** -/
theorem DistOn.interleaveLock_ret {Bad : β₁ → Prop} {Bad' : β₁ × β₂ → Prop} {i₂ : I₂} {a₂ : α₂} {b₂ : β₂}
    (h₂ : R₂ i₂ a₂ b₂) (hB : ∀ b, Bad b → Bad' (b, b₂)) {j : ℕ} {i₁ : I₁} {g₁ : Game α₁} {g₁' : Game β₁}
    {L : Lock T D get₁ R₁ i₁ g₁ g₁'} (hL : DistOn slot Bad j L) :
    DistOn slot Bad' j (interleaveLock hd L (.ret h₂)) := by
  cases L with
  | ret _ => exact .noSlot (.ret _)
  | coin => exact hL.embedLeft hd h₂ hB
  | send => exact hL.embedLeft hd h₂ hB
  | table => exact hL.embedLeft hd h₂ hB

end Interleave

section Lead

variable {T D α₁ β₁ α₂ β₂ I₁ I₂ γ γ' : Type} {get₁ : I₁ → ℕ → Option (T × D)}
  {get₂ : I₂ → ℕ → Option (T × D)} {R₁ : I₁ → α₁ → β₁ → Prop} {R₂ : I₂ → α₂ → β₂ → Prop} [Inhabited T]
  [Inhabited D] (hd : Lock.Disjoint get₁ get₂) {slot : ℕ} {Dead : γ' → Prop} {fo : α₁ × α₂ → Game γ}
  {f' : β₁ × β₂ → Game γ'} {R₃ : I₁ × I₂ → γ → γ' → Prop}
  {hK : ∀ i a b, RPair R₁ R₂ i a b → Lock T D (getPair get₁ get₂) R₃ i (fo a) (f' b)}

include hd

/-- **A slot of the first lockstep at distance `j` on its good paths is met after `j` coins at every table choice**
by a strategy leading with it, or the strategy reaches only `Dead` outcomes there, a `Bad` first part's outcomes `Dead`
(`SlotAtD.lead_fst` on `DistOn`). -/
theorem SlotAtD.lead_fst_on {Bad : β₁ → Prop} (hKn : ∀ i a b (hr : RPair R₁ R₂ i a b), NoSlot slot (hK i a b hr))
    (hB : ∀ b y, Bad b.1 → Out (f' b) y → Dead y) {n : ℕ}
    {g₁' : Game β₁} {g₂' : Game β₂} {σ : Strategy ((interleave g₁' g₂').bind f')} (hS : Lead f' Dead n g₁' g₂' σ) :
    ∀ {j : ℕ} {i₁ : I₁} {g₁ : Game α₁} (L₁ : Lock T D get₁ R₁ i₁ g₁ g₁') {i₂ : I₂} {g₂ : Game α₂}
      (L₂ : Lock T D get₂ R₂ i₂ g₂ g₂'), DistOn slot Bad j L₁ → NoSlot slot L₂ →
        SlotAtD slot Dead j (Lock.bind hK (interleaveLock hd L₁ L₂)) σ := by
  have hbad : ∀ {g₁' : Game β₁} {g₂' : Game β₂} {σ : Strategy ((interleave g₁' g₂').bind f')} {j : ℕ} {i : I₁ × I₂}
      {g : Game γ} {L : Lock T D (getPair get₁ get₂) R₃ i g ((interleave g₁' g₂').bind f')},
      (∀ b, Out g₁' b → Bad b) → SlotAtD slot Dead j L σ := fun hb =>
    .dead fun y hy => by
      obtain ⟨b, hb', hy'⟩ := Out.of_bind (Reach.out hy)
      exact hB b y (hb _ (Out.of_interleave hb').1) hy'
  induction hS with
  | dead hσ =>
      intro _ _ _ _ _ _ _ _ _
      exact .dead hσ
  | done =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ _ hN₂
      cases L₁ with
      | ret h₁ => exact .noSlot ((hN₂.embedRight hd h₁).bind hKn)
  | last =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hD _
      cases L₂ with
      | ret h₂ =>
        exact .of_distOn (Bad := Dead) (fun _ h => h)
          ((hD.interleaveLock_ret hd h₂ (Bad' := fun b => Bad b.1) fun _ h => h).bind_noSlot hB hKn) _
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
        | noSlot h0 => exact .noSlot ((h0.interleaveLock hd hN₂).bind hKn)
        | bad hb => exact hbad hb
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
        | noSlot h0 => exact .noSlot ((h0.interleaveLock hd hN₂).bind hKn)
        | bad hb => exact hbad hb
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
        | noSlot h0 => exact .noSlot ((h0.interleaveLock hd hN₂).bind hKn)
        | bad hb => exact hbad hb
      | table s upd hf hs hk L₁ =>
        cases hD with
        | other _ hne hD =>
          cases L₂ with
          | coin _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .other _ hne fun t => ih (L₁ t m) (.coin _ L₂) (hD t m) hN₂
        | here =>
          cases L₂ with
          | coin _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .here _
        | noSlot h0 => exact .noSlot ((h0.interleaveLock hd hN₂).bind hKn)
        | bad hb => exact hbad hb
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
        | noSlot h0 => exact .noSlot ((h0.interleaveLock hd hN₂).bind hKn)
        | bad hb => exact hbad hb
      | table s upd hf hs hk L₁ =>
        cases hD with
        | other _ hne hD =>
          cases L₂ with
          | send _ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .other _ hne fun t => ih (L₁ t m) (.send _ L₂) (hD t m) hN₂
          | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .other _ hne fun t => ih (L₁ t m) (.table s₂ upd₂ hf₂ hs₂ hk₂ L₂) (hD t m) hN₂
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
        | noSlot h0 => exact .noSlot ((h0.interleaveLock hd hN₂).bind hKn)
        | bad hb => exact hbad hb

/-- **A slot of the second lockstep at distance `j` on its good paths is met after `n + j` coins at every table
choice** by a strategy leading with the first, which ends after `n` coins, or the strategy reaches only `Dead` outcomes
there, a `Bad` second part's outcomes `Dead` (`SlotAtD.lead_snd` on `DistOn`). -/
theorem SlotAtD.lead_snd_on {Bad : β₂ → Prop} (hKn : ∀ i a b (hr : RPair R₁ R₂ i a b), NoSlot slot (hK i a b hr))
    (hB : ∀ b y, Bad b.2 → Out (f' b) y → Dead y) {n : ℕ}
    {g₁' : Game β₁} {g₂' : Game β₂} {σ : Strategy ((interleave g₁' g₂').bind f')} (hS : Lead f' Dead n g₁' g₂' σ) :
    ∀ {j : ℕ} {i₁ : I₁} {g₁ : Game α₁} (L₁ : Lock T D get₁ R₁ i₁ g₁ g₁') {i₂ : I₂} {g₂ : Game α₂}
      (L₂ : Lock T D get₂ R₂ i₂ g₂ g₂'), NoSlot slot L₁ → DistOn slot Bad j L₂ →
        SlotAtD slot Dead (n + j) (Lock.bind hK (interleaveLock hd L₁ L₂)) σ := by
  induction hS with
  | dead hσ =>
      intro _ _ _ _ _ _ _ _ _
      exact .dead hσ
  | done =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ _ hD
      cases L₁ with
      | ret h₁ =>
        rw [Nat.zero_add]
        exact .of_distOn (Bad := Dead) (fun _ h => h)
          ((hD.embedRight hd h₁ (Bad' := fun b => Bad b.2) fun _ h => h).bind_noSlot hB hKn) _
  | last =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ hN₁ _
      cases L₂ with
      | ret h₂ => exact .noSlot ((hN₁.interleaveLock hd (.ret h₂)).bind hKn)
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
            exact .other _ hne fun t => ih (L₁ t m) (.coin _ L₂) (hN₁ t m) hD
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
            exact .other _ hne fun t => ih (L₁ t m) (.send _ L₂) (hN₁ t m) hD
          | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
            simp only [Lock.interleaveLock, Lock.lockWith]
            refine .send Bool ?_
            exact .other _ hne fun t => ih (L₁ t m) (.table s₂ upd₂ hf₂ hs₂ hk₂ L₂) (hN₁ t m) hD

end Lead

end Lock

end FlockSoundness.Game
