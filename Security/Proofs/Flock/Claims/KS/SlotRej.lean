import Proofs.Flock.Claims.KS.SlotBatch
import Proofs.Flock.Soundness.Discharge.ZkExec.Shape

/-!
# A slot's coin count where the runs that miss it reject

`Dist` and `SlotAtD` fix a slot's coin count on every path. A lockstep that meets the slot after a fixed count only on
its runs that end well (`ExactOn`), and rejects on the others before the slot, has the count where it ends well and
only rejected outcomes elsewhere. Where `Φ` vanishes on two rejected outcomes that is enough for `SlotAt`:

* `Out.playsT`, `Out.of_interleave`, `Reach.out`: an outcome is played on some run; an outcome of two interleaved games
  is a pair of their outcomes; an outcome a strategy reaches is an outcome;
* `DistR`, `DistR.bind_on`: `Dist` with a `rej` case, every outcome rejected; a lockstep with no table node of the slot,
  exact on its runs that end well, followed by a continuation that rejects after the others, adds the shape's coins;
* `DistR.bind_noSlot`, `DistR.embedLeft`, `DistR.embedRight`, `DistR.interleaveLock_ret`: `Dist`'s, carrying `rej`;
* `SlotAtDR`, `SlotAtDR.slotAt`: `SlotAtD` with the same `rej` case, which gives `SlotAt` where `Φ` vanishes on two
  rejected outcomes;
* `SlotAtDR.lead_fst`, `SlotAtDR.lead_snd`, `SlotAtDR.batchLock_sched`, `SlotAtDR.batchLock_nil_followL`: the counts of
  `SlotPost` and `SlotBatch`, carrying `rej`.
-/

namespace FlockSoundness.Game

variable {α β : Type}

theorem Out.playsT {g : Game α} {b : α} (h : Out g b) : ∃ ts, PlaysT g ts b := by
  induction h with
  | ret b => exact ⟨[], .ret b⟩
  | send m _ ih =>
      obtain ⟨ts, h⟩ := ih
      exact ⟨_, .send m h⟩
  | coin c _ ih =>
      obtain ⟨ts, h⟩ := ih
      exact ⟨_, .coin c h⟩

theorem Out.of_interleaveWith {g₁ : Game α} {adv : Game β → Game (α × β)} {fin : β → Game (α × β)}
    (hadv : ∀ g₂ c, Out (adv g₂) c → Out g₁ c.1 ∧ Out g₂ c.2) (hfin : ∀ b c, Out (fin b) c → Out g₁ c.1 ∧ c.2 = b) :
    ∀ {g₂ : Game β} {c : α × β}, Out (interleaveWith adv fin g₂) c → Out g₁ c.1 ∧ Out g₂ c.2
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

/-- **An outcome of two interleaved games is a pair of their outcomes.** -/
theorem Out.of_interleave :
    ∀ {g₁ : Game α} {g₂ : Game β} {c : α × β}, Out (interleave g₁ g₂) c → Out g₁ c.1 ∧ Out g₂ c.2
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

end FlockSoundness.Game

namespace FlockSoundness.Game.Lock

open FlockSoundness.Discharge.Exec

theorem Reach.out {β : Type} : ∀ {g : Game β} {σ : Strategy g} {b : β}, Reach g σ b → Out g b
  | .ret _, _, _, h => by subst h; exact .ret _
  | .send _ k, σ, _, h => .send σ.1 (Reach.out (g := k σ.1) (σ := σ.2) h)
  | @Game.coin _ _ _ _ k, σ, _, ⟨c, h⟩ => .coin c (Reach.out (g := k c) (σ := σ c) h)

/-- **Every path meets `slot`'s table node after exactly `j` coins**, or meets no such node, or ends only in `Rej`
outcomes. -/
inductive DistR {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} (slot : ℕ)
    (Rej : β → Prop) : ℕ → {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Prop
  | coin {j : ℕ} {i : I} (C : Type) [Fintype C] [Nonempty C] {k : C → Game α} {k' : C → Game β}
      {L : (c : C) → Lock T D get R i (k c) (k' c)} :
      (∀ c, DistR slot Rej j (L c)) → DistR slot Rej (j + 1) (Lock.coin C L)
  | send {j : ℕ} {i : I} (M : Type) {k : M → Game α} {k' : M → Game β}
      {L : (m : M) → Lock T D get R i (k m) (k' m)} :
      (∀ m, DistR slot Rej j (L m)) → DistR slot Rej j (Lock.send M L)
  | other {j : ℕ} {i : I} (s : ℕ) {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i s = none} {hs : ∀ t d, get (upd t d) s = some (t, d)}
      {hk : ∀ t d s', s' ≠ s → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} (hne : s ≠ slot) :
      (∀ t d, DistR slot Rej j (L t d)) → DistR slot Rej j (Lock.table s upd hf hs hk L)
  | here {i : I} {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i slot = none} {hs : ∀ t d, get (upd t d) slot = some (t, d)}
      {hk : ∀ t d s', s' ≠ slot → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} :
      DistR slot Rej 0 (Lock.table slot upd hf hs hk L)
  | noSlot {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} :
      NoSlot slot L → DistR slot Rej j L
  | rej {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} :
      (∀ b, Out g' b → Rej b) → DistR slot Rej j L

theorem DistR.of_dist {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} {slot : ℕ}
    {Rej : β → Prop} {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (h : Dist slot j L) :
    DistR slot Rej j L := by
  induction h with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot h0

/-- **Sequencing a distance with a continuation that has no table node of the slot** keeps the distance, a rejected
outcome continuing to rejected ones. -/
theorem DistR.bind_noSlot {T D I α β α₂ β₂ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {R₂ : I → α₂ → β₂ → Prop} {f : α → Game α₂} {f' : β → Game β₂} {slot j : ℕ} {Rej : β → Prop}
    {Rej' : β₂ → Prop} {h : ∀ i a b, R i a b → Lock T D get R₂ i (f a) (f' b)}
    (hK : ∀ i a b (hr : R i a b), NoSlot slot (h i a b hr)) (hRj : ∀ b c, Rej b → Out (f' b) c → Rej' c)
    {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} (hL : DistR slot Rej j L) :
    DistR slot Rej' j (Lock.bind h L) := by
  induction hL with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (h0.bind hK)
  | rej hR =>
      refine .rej fun c hc => ?_
      obtain ⟨b, hb, hc⟩ := Out.of_bind hc
      exact hRj b c (hR b hb) hc

theorem exactOn_cases {β : Type} {g : Game β} {σ : List Kind} {P : β → Prop}
    (hσ : Discharge.ZkExec.ExactOn g σ P) :
    (¬ ∃ ts b, PlaysT g ts b ∧ P b) ∨ ∃ ts b, PlaysT g ts b ∧ P b ∧ ts.map kindOf = σ := by
  by_cases h : ∃ ts b, PlaysT g ts b ∧ P b
  · obtain ⟨ts, b, hp, hP⟩ := h
    exact .inr ⟨ts, b, hp, hP, hσ ts b hp hP⟩
  · exact .inl h

theorem out_bind_of_none {β γ : Type} {g' : Game β} {f' : β → Game γ} {P : β → Prop} {Rej : γ → Prop}
    (hR : ∀ b, ¬ P b → ∀ c, Out (f' b) c → Rej c) (hn : ¬ ∃ ts b, PlaysT g' ts b ∧ P b) :
    ∀ c, Out (g'.bind f') c → Rej c := by
  intro c hc
  obtain ⟨b, hb, hc⟩ := Out.of_bind hc
  obtain ⟨ts, hp⟩ := hb.playsT
  exact hR b (fun hP => hn ⟨ts, b, hp, hP⟩) c hc

/-- **A distance after a lockstep with no table node of the slot, exact on its runs that end well**, adds the shape's
coins, where the continuation after a run that ends badly rejects. -/
theorem DistR.bind_on {T D I α β α₂ β₂ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    {R₂ : I → α₂ → β₂ → Prop} {f : α → Game α₂} {f' : β → Game β₂} {slot j : ℕ} {P : β → Prop} {Rej : β₂ → Prop}
    {h : ∀ i a b, R i a b → Lock T D get R₂ i (f a) (f' b)}
    (hK : ∀ i a b (hr : R i a b), P b → DistR slot Rej j (h i a b hr))
    (hR : ∀ b, ¬ P b → ∀ c, Out (f' b) c → Rej c) {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'}
    (hL : NoSlot slot L) :
    ∀ σ : List Kind, Discharge.ZkExec.ExactOn g' σ P → DistR slot Rej (j + coinsK σ) (Lock.bind h L) := by
  induction hL with
  | @ret i a b hr =>
      intro σ hσ
      by_cases hb : P b
      · have e := hσ [] b (.ret b) hb
        subst e
        simp only [List.map_nil, coinsK, Nat.add_zero]
        exact hK _ _ _ hr hb
      · exact .rej (hR b hb)
  | coin C _ ih =>
      intro σ hσ
      rcases exactOn_cases hσ with hn | ⟨ts, b, hp, hP, e⟩
      · exact .rej (out_bind_of_none hR hn)
      · cases hp with
        | coin c' hp' =>
          subst e
          simp only [List.map_cons, kindOf, coinsK, ← Nat.add_assoc]
          exact .coin C fun c => ih c _ fun us b' hq hP' => congrArg List.tail (hσ _ _ (.coin c hq) hP')
  | send M _ ih =>
      intro σ hσ
      rcases exactOn_cases hσ with hn | ⟨ts, b, hp, hP, e⟩
      · exact .rej (out_bind_of_none hR hn)
      · cases hp with
        | send m' hp' =>
          subst e
          simp only [List.map_cons, kindOf, coinsK]
          exact .send M fun m => ih m _ fun us b' hq hP' => congrArg List.tail (hσ _ _ (.send m hq) hP')
  | table s hne _ ih =>
      intro σ hσ
      rcases exactOn_cases hσ with hn | ⟨ts, b, hp, hP, e⟩
      · exact .rej (out_bind_of_none hR hn)
      · cases hp with
        | send d' hp' =>
          subst e
          simp only [List.map_cons, kindOf, coinsK]
          exact .other s hne fun t d => ih t d _ fun us b' hq hP' => congrArg List.tail (hσ _ _ (.send d hq) hP')

section Interleave

variable {T D α₁ β₁ α₂ β₂ I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (T × D)} {get₂ : I₂ → ℕ → Option (T × D)}
  {R₁ : I₁ → α₁ → β₁ → Prop} {R₂ : I₂ → α₂ → β₂ → Prop} [Inhabited T] [Inhabited D] (hd : Lock.Disjoint get₁ get₂)
  {slot : ℕ}

include hd

theorem DistR.embedLeft {Rej : β₁ → Prop} {i₂ : I₂} {a₂ : α₂} {b₂ : β₂} (h₂ : R₂ i₂ a₂ b₂) {j : ℕ} {i₁ : I₁}
    {g₁ : Game α₁} {g₁' : Game β₁} {L : Lock T D get₁ R₁ i₁ g₁ g₁'} (hL : DistR slot Rej j L) :
    DistR slot (fun x : β₁ × β₂ => Rej x.1) j (embedLeft hd h₂ L) := by
  induction hL with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (h0.embedLeft hd h₂)
  | rej hR =>
      refine .rej fun x hx => ?_
      obtain ⟨y, hy, rfl⟩ := Out.of_map hx
      exact hR y hy

theorem DistR.embedRight {Rej : β₂ → Prop} {i₁ : I₁} {a₁ : α₁} {b₁ : β₁} (h₁ : R₁ i₁ a₁ b₁) {j : ℕ} {i₂ : I₂}
    {g₂ : Game α₂} {g₂' : Game β₂} {L : Lock T D get₂ R₂ i₂ g₂ g₂'} (hL : DistR slot Rej j L) :
    DistR slot (fun x : β₁ × β₂ => Rej x.2) j (embedRight hd h₁ L) := by
  induction hL with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (h0.embedRight hd h₁)
  | rej hR =>
      refine .rej fun x hx => ?_
      obtain ⟨y, hy, rfl⟩ := Out.of_map hx
      exact hR y hy

theorem DistR.interleaveLock_ret {Rej : β₁ → Prop} {i₂ : I₂} {a₂ : α₂} {b₂ : β₂} (h₂ : R₂ i₂ a₂ b₂) {j : ℕ}
    {i₁ : I₁} {g₁ : Game α₁} {g₁' : Game β₁} {L : Lock T D get₁ R₁ i₁ g₁ g₁'} (hL : DistR slot Rej j L) :
    DistR slot (fun x : β₁ × β₂ => Rej x.1) j (interleaveLock hd L (.ret h₂)) := by
  cases L with
  | ret _ => exact .noSlot (.ret _)
  | coin => exact hL.embedLeft hd h₂
  | send => exact hL.embedLeft hd h₂
  | table => exact hL.embedLeft hd h₂

end Interleave

/-- **The strategy reaches `slot`'s table node after exactly `j` coins at every table choice**, or meets no such node,
or reaches only `Dead` outcomes, or ends only in `Rej` outcomes (`SlotAtD` with a `rej` case). -/
inductive SlotAtDR {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} (slot : ℕ)
    (Dead Rej : β → Prop) :
    ℕ → {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Strategy g' → Prop
  | ret {j : ℕ} {i : I} {a : α} {b : β} (h : R i a b) (σ' : Strategy (.ret b)) :
      SlotAtDR slot Dead Rej j (Lock.ret h) σ'
  | coin {j : ℕ} {i : I} (C : Type) [Fintype C] [Nonempty C] {k : C → Game α} {k' : C → Game β}
      {L : (c : C) → Lock T D get R i (k c) (k' c)} {σ' : Strategy (.coin C k')} :
      (∀ c, SlotAtDR slot Dead Rej j (L c) (σ' c)) → SlotAtDR slot Dead Rej (j + 1) (Lock.coin C L) σ'
  | send {j : ℕ} {i : I} (M : Type) {k : M → Game α} {k' : M → Game β}
      {L : (m : M) → Lock T D get R i (k m) (k' m)} {σ' : Strategy (.send M k')} :
      SlotAtDR slot Dead Rej j (L σ'.1) σ'.2 → SlotAtDR slot Dead Rej j (Lock.send M L) σ'
  | other {j : ℕ} {i : I} (s : ℕ) {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i s = none} {hs : ∀ t d, get (upd t d) s = some (t, d)}
      {hk : ∀ t d s', s' ≠ s → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} {σ' : Strategy (.send D k')}
      (hne : s ≠ slot) :
      (∀ t, SlotAtDR slot Dead Rej j (L t σ'.1) σ'.2) → SlotAtDR slot Dead Rej j (Lock.table s upd hf hs hk L) σ'
  | here {i : I} {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i slot = none} {hs : ∀ t d, get (upd t d) slot = some (t, d)}
      {hk : ∀ t d s', s' ≠ slot → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} (σ' : Strategy (.send D k')) :
      SlotAtDR slot Dead Rej 0 (Lock.table slot upd hf hs hk L) σ'
  | dead {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} :
      (∀ b, Reach g' σ' b → Dead b) → SlotAtDR slot Dead Rej j L σ'
  | noSlot {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} :
      NoSlot slot L → SlotAtDR slot Dead Rej j L σ'
  | rej {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} :
      (∀ b, Out g' b → Rej b) → SlotAtDR slot Dead Rej j L σ'

section Post

variable {T D I α β γ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} {slot : ℕ}

/-- **A chooser-free count gives the count at every chooser**, where `Φ` vanishes on two `Dead` outcomes and on two
`Rej` outcomes. -/
theorem SlotAtDR.slotAt {Dead Rej : β → Prop} {plur : Chooser T D β} {Φ : D → β → β → ℝ}
    (hΦ : ∀ d b b', Dead b → Dead b' → Φ d b b' = 0) (hΦR : ∀ d b b', Rej b → Rej b' → Φ d b b' = 0) {j : ℕ} {i : I}
    {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} (hs : SlotAtDR slot Dead Rej j L σ') :
    SlotAt plur slot Φ j L σ' := by
  induction hs with
  | ret hr σ' => exact .ret hr σ'
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne (ih _)
  | here σ' => exact .here σ'
  | dead hσ => exact .nul (forkE_of_reach plur slot hΦ _ _ hσ)
  | noSlot h0 => exact .of_noSlot h0 _
  | rej hR => exact .nul (forkE_of_reach plur slot hΦR _ _ fun b hb => hR b (Reach.out hb))

theorem SlotAtDR.of_distR {Dead Rej : β → Prop} {j : ℕ} {i : I} {g : Game α} {g' : Game β}
    {L : Lock T D get R i g g'} (h : DistR slot Rej j L) (σ' : Strategy g') : SlotAtDR slot Dead Rej j L σ' := by
  induction h with
  | coin C _ ih => exact .coin C fun c => ih c (σ' c)
  | send M _ ih => exact .send M (ih _ _)
  | other s hne _ ih => exact .other s hne fun t => ih t _ _
  | here => exact .here σ'
  | noSlot h => exact .noSlot h
  | rej hR => exact .rej hR

theorem SlotAtDR.mapL {α₂ : Type} {R₂ : I → α₂ → β → Prop} {Dead Rej : β → Prop} (f : α → α₂)
    (h : ∀ i a b, R i a b → R₂ i (f a) b) {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'}
    {σ' : Strategy g'} (hs : SlotAtDR slot Dead Rej j L σ') : SlotAtDR slot Dead Rej j (Lock.mapL f h L) σ' := by
  induction hs with
  | ret hr σ' => exact .ret (h _ _ _ hr) σ'
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other s hne ih
  | here σ' => exact .here σ'
  | dead hσ => exact .dead hσ
  | noSlot h0 => exact .noSlot (h0.mapL f h)
  | rej hR => exact .rej hR

theorem SlotAtDR.map {α₂ β₂ : Type} {R₂ : I → α₂ → β₂ → Prop} {Dead Rej : β → Prop} {Dead₂ Rej₂ : β₂ → Prop}
    (f : α → α₂) (f' : β → β₂) (h : ∀ i a b, R i a b → R₂ i (f a) (f' b)) (hD : ∀ b, Dead b → Dead₂ (f' b))
    (hRj : ∀ b, Rej b → Rej₂ (f' b)) {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'}
    {σ' : Strategy g'} (hs : SlotAtDR slot Dead Rej j L σ') :
    SlotAtDR slot Dead₂ Rej₂ j (Lock.map f f' h L) (Strategy.map f' g' σ') := by
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
  | @rej j i g g' L σ' hR =>
      refine .rej fun x hx => ?_
      obtain ⟨y, hy, rfl⟩ := Out.of_map hx
      exact hRj y (hR y hy)

theorem SlotAtDR.leftSame_ret {Dead Rej : β → Prop} {Dead₂ Rej₂ : β × γ → Prop} (hD : ∀ b c, Dead b → Dead₂ (b, c))
    (hRj : ∀ b c, Rej b → Rej₂ (b, c)) {j : ℕ} {i : I} {g : Game α} {g' : Game β} (c : γ)
    (L : Lock T D get R i g g') (σ' : Strategy (interleave g' (.ret c)))
    (hs : SlotAtDR slot Dead Rej j L
      (Strategy.ofMap (·, c) g' (cast (congrArg Strategy (interleave_ret c g')) σ'))) :
    SlotAtDR slot Dead₂ Rej₂ j (Lock.leftSame L (.ret c)) σ' := by
  have key := SlotAtDR.map (R₂ := fun i a b => R i a.1 b.1 ∧ a.2 = b.2) (fun a => (a, c)) (fun b => (b, c))
    (fun _ _ _ hr => ⟨hr, rfl⟩) (fun b hb => hD b c hb) (fun b hb => hRj b c hb) hs
  rw [Strategy.map_ofMap] at key
  cases L <;> exact key

end Post

section Right

variable {T D I α β γ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} {slot : ℕ}
  {Dead Rej : γ × β → Prop}

theorem SlotAtDR.rightSame_firstS {Q : (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Prop} {k n : ℕ}
    {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} (hS : FirstS Dead Q n h g' σ) :
    ∀ {i : I} {g : Game α} (L : Lock T D get R i g g'),
      (∀ h' σ', Q h' g' σ' → SlotAtDR slot Dead Rej k (Lock.rightSame h' L) σ') →
        SlotAtDR slot Dead Rej (n + k) (Lock.rightSame h L) σ := by
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

section Follow

variable {T D I α β γ : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop} {slot : ℕ}

theorem SlotAtDR.rightSame_followR {Dead Rej : γ × β → Prop} {Dead' Rej' : β → Prop}
    (hRj : ∀ x : γ × β, Rej' x.2 → Rej x) {b : Option ℕ} {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')}
    {τ : Strategy g'} (hF : FollowR Dead Dead' b h g' σ τ) :
    ∀ {k : ℕ} {i : I} {g : Game α} (L : Lock T D get R i g g'), b = some k → SlotAtDR slot Dead' Rej' k L τ →
      SlotAtDR slot Dead Rej k (Lock.rightSame h L) σ := by
  have hdead : ∀ {b : Option ℕ} {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} {τ : Strategy g'},
      FollowR Dead Dead' b h g' σ τ → ∀ {k : ℕ} {i : I} {g : Game α} (L : Lock T D get R i g g'),
        (∀ b, Reach g' τ b → Dead' b) → SlotAtDR slot Dead Rej k (Lock.rightSame h L) σ :=
    fun hF _ _ _ _ hr => .dead (hF.dead_of hr)
  have hno : ∀ {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} {k : ℕ} {i : I} {g : Game α}
      {L : Lock T D get R i g g'}, NoSlot slot L → SlotAtDR slot Dead Rej k (Lock.rightSame h L) σ :=
    fun h0 => .noSlot (h0.rightSame _)
  have hrej : ∀ {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} {k : ℕ} {i : I} {g : Game α}
      (L : Lock T D get R i g g'), (∀ b, Out g' b → Rej' b) → SlotAtDR slot Dead Rej k (Lock.rightSame h L) σ :=
    fun _ hR => .rej fun x hx => hRj x (hR x.2 (Out.of_interleave hx).2)
  induction hF with
  | dead hσ => exact fun _ _ _ => .dead hσ
  | spent _ =>
      intro k i g L hb
      cases hb
  | icoinS hτ ih =>
      intro k i g L hb hL
      cases hL with
      | coin _ hL =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .coin _ fun c => ih c _ (by rw [hb]; rfl) (hL c)
      | dead hr => exact hdead (.icoinS hτ) _ hr
      | noSlot h0 => exact hno h0
      | rej hR => exact hrej _ hR
  | isendS m₂ hτ ih =>
      intro k i g L hb hL
      cases hL with
      | send _ hL =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .send _ (ih _ hb hL)
      | other s hne hL =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .other _ hne fun t => ih _ hb (hL t)
      | here =>
        simp only [Lock.rightSame, Lock.lockWithL]
        refine .send Bool ?_
        exact .here _
      | dead hr => exact hdead (.isendS m₂ hτ) _ hr
      | noSlot h0 => exact hno h0
      | rej hR => exact hrej _ hR

theorem SlotAtDR.leftSame_followL {Dead Rej : β × γ → Prop} {Dead' Rej' : β → Prop}
    (hD : ∀ x : β × γ, Dead' x.1 → Dead x) (hRj : ∀ x : β × γ, Rej' x.1 → Rej x) {b : Option ℕ} {g' : Game β}
    {h : Game γ} {σ : Strategy (interleave g' h)} {τ : Strategy g'} (hF : FollowL Dead Dead' b g' h σ τ) :
    ∀ {k : ℕ} {i : I} {g : Game α} (L : Lock T D get R i g g'), b = some k → SlotAtDR slot Dead' Rej' k L τ →
      SlotAtDR slot Dead Rej k (Lock.leftSame L h) σ := by
  have hdead : ∀ {b : Option ℕ} {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} {τ : Strategy g'},
      FollowL Dead Dead' b g' h σ τ → ∀ {k : ℕ} {i : I} {g : Game α} (L : Lock T D get R i g g'),
        (∀ b, Reach g' τ b → Dead' b) → SlotAtDR slot Dead Rej k (Lock.leftSame L h) σ :=
    fun hF _ _ _ _ hr => .dead (hF.dead_of hD hr)
  have hno : ∀ {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} {k : ℕ} {i : I} {g : Game α}
      {L : Lock T D get R i g g'}, NoSlot slot L → SlotAtDR slot Dead Rej k (Lock.leftSame L h) σ :=
    fun h0 => .noSlot (h0.leftSame _)
  have hrej : ∀ {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} {k : ℕ} {i : I} {g : Game α}
      (L : Lock T D get R i g g'), (∀ b, Out g' b → Rej' b) → SlotAtDR slot Dead Rej k (Lock.leftSame L h) σ :=
    fun _ hR => .rej fun x hx => hRj x (hR x.1 (Out.of_interleave hx).1)
  induction hF with
  | dead hσ => exact fun _ _ _ => .dead hσ
  | spent _ =>
      intro k i g L hb
      cases hb
  | gret =>
      intro k i g L _ _
      cases L with
      | ret hr => exact .noSlot (NoSlot.map _ _ _ (noSlot_same _))
  | @hret b g' c σ =>
      intro k i g L _ hL
      exact SlotAtDR.leftSame_ret (fun b c hb => hD (b, c) hb) (fun b c hb => hRj (b, c) hb) c L σ hL
  | gcoinS hτ ih =>
      intro k i g L hb hL
      cases hL with
      | coin _ hL =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .coin _ fun c => ih c _ (by rw [hb]; rfl) (hL c)
      | dead hr => exact hdead (.gcoinS hτ) _ hr
      | noSlot h0 => exact hno h0
      | rej hR => exact hrej _ hR
  | gcoinC hτ ih =>
      intro k i g L hb hL
      cases hL with
      | coin _ hL =>
        simp only [Lock.leftSame, Lock.lockWithSame]
        refine .send Bool ?_
        exact .coin _ fun c => ih c _ (by rw [hb]; rfl) (hL c)
      | dead hr => exact hdead (.gcoinC hτ) _ hr
      | noSlot h0 => exact hno h0
      | rej hR => exact hrej _ hR
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
      | rej hR => exact hrej _ hR
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
      | rej hR => exact hrej _ hR
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

end Follow

section Batch

variable {T D I ζ : Type} {get : I → ℕ → Option (T × D)} {R : I → ζ → ζ → Prop}

/-- **`SlotAtD.batchLock_sched` carrying `rej`**, the rejected outcomes those whose position `pre.length` is
rejected. -/
theorem SlotAtDR.batchLock_sched {P : ζ → Prop} (d : ζ) {i : I} {g g' : Game ζ} (L : Lock T D get R i g g')
    (post : List (Game ζ)) {slot k : ℕ} {Q : (List ζ → Prop) → Strategy (batch (g' :: post)) → Prop}
    (hQ : ∀ Dead σ', Q Dead σ' →
      SlotAtDR slot Dead (fun outs => P (outs.getD 0 d)) k (batchLock d L post []) σ') :
    ∀ (pre : List (Game ζ)) {Dead : List ζ → Prop} {n : ℕ} {σ : Strategy (batch (pre ++ g' :: post))},
      PreSched g' post k pre Dead Q n σ →
        SlotAtDR slot Dead (fun outs => P (outs.getD pre.length d)) (n + k) (batchLock d L post pre) σ
  | [], Dead, _, σ, ⟨hn, hq⟩ => by
      subst hn
      rw [Nat.zero_add]
      exact hQ Dead σ hq
  | h :: pre, Dead, _, σ, ⟨n₀, n₁, hn, hF⟩ => by
      subst hn
      have hs := SlotAtDR.rightSame_firstS (k := n₁ + k)
        (Rej := fun x : ζ × List ζ => P (x.2.getD pre.length d)) hF (batchLock d L post pre)
        fun h' σ' ⟨_, τ, hR, hP⟩ => by
          have hP' : PreSched g' post k pre (fun outs => ∀ c, Dead (c :: outs)) Q n₁ τ := hP
          exact SlotAtDR.rightSame_followR (Dead' := fun outs => ∀ c, Dead (c :: outs))
            (Rej' := fun outs => P (outs.getD pre.length d)) (fun _ hx => hx) hR _ rfl
            (SlotAtDR.batchLock_sched d L post hQ pre hP')
      have key := SlotAtDR.map
        (R₂ := fun i (a b : List ζ) => R i (a.getD (h :: pre).length d) (b.getD (h :: pre).length d))
        (Dead₂ := Dead) (Rej₂ := fun outs => P (outs.getD (h :: pre).length d))
        (fun x : ζ × List ζ => x.1 :: x.2) (fun x : ζ × List ζ => x.1 :: x.2)
        (fun _ _ _ h' => h'.2) (fun _ hx => hx) (fun _ hx => hx) hs
      rw [Strategy.map_ofMap (fun x : ζ × List ζ => x.1 :: x.2) (h.interleave (batch (pre ++ g' :: post))) σ,
        ← Nat.add_assoc] at key
      exact key

/-- **`SlotAtD.batchLock_nil_followL` carrying `rej`**, the rejected outcomes those whose head is rejected. -/
theorem SlotAtDR.batchLock_nil_followL {P : ζ → Prop} (d : ζ) {i : I} {g g' : Game ζ} (L : Lock T D get R i g g')
    (post : List (Game ζ)) {slot k : ℕ} {Dead : List ζ → Prop} {σ' : Strategy (batch (g' :: post))}
    {τ : Strategy g'}
    (hF : FollowL (fun x : ζ × List ζ => Dead (x.1 :: x.2)) (fun x => ∀ rest, Dead (x :: rest)) (some k) g'
      (batch post)
      (Strategy.ofMap (fun x : ζ × List ζ => x.1 :: x.2) (g'.interleave (batch post)) σ') τ)
    (hD : SlotAtDR slot (fun x => ∀ rest, Dead (x :: rest)) P k L τ) :
    SlotAtDR slot Dead (fun outs => P (outs.getD 0 d)) k (batchLock d L post []) σ' := by
  have hs := SlotAtDR.leftSame_followL (Rej := fun x : ζ × List ζ => P x.1) (fun x hx => hx x.2) (fun _ hx => hx)
    hF L rfl hD
  have key := SlotAtDR.map (R₂ := fun i a b => R i (a.getD 0 d) (b.getD 0 d)) (Dead₂ := Dead)
    (Rej₂ := fun outs => P (outs.getD 0 d)) (fun x : ζ × List ζ => x.1 :: x.2) (fun x : ζ × List ζ => x.1 :: x.2)
    (fun _ _ _ h => h.1) (fun _ hx => hx) (fun _ hx => hx) hs
  exact Strategy.map_ofMap (fun x : ζ × List ζ => x.1 :: x.2) (g'.interleave (batch post)) σ' ▸ key

end Batch

section Lead

variable {T D α₁ β₁ α₂ β₂ I₁ I₂ γ γ' : Type} {get₁ : I₁ → ℕ → Option (T × D)}
  {get₂ : I₂ → ℕ → Option (T × D)} {R₁ : I₁ → α₁ → β₁ → Prop} {R₂ : I₂ → α₂ → β₂ → Prop} [Inhabited T]
  [Inhabited D] (hd : Lock.Disjoint get₁ get₂) {slot : ℕ} {Dead Rej : γ' → Prop} {fo : α₁ × α₂ → Game γ}
  {f' : β₁ × β₂ → Game γ'} {R₃ : I₁ × I₂ → γ → γ' → Prop}
  {hK : ∀ i a b, RPair R₁ R₂ i a b → Lock T D (getPair get₁ get₂) R₃ i (fo a) (f' b)}

include hd

/-- **`SlotAtD.lead_fst` from a distance carrying `rej`**, a rejected outcome of the first lockstep continuing to
rejected ones. -/
theorem SlotAtDR.lead_fst {Rej₁ : β₁ → Prop} (hKn : ∀ i a b (hr : RPair R₁ R₂ i a b), NoSlot slot (hK i a b hr))
    (hRj : ∀ (x : β₁ × β₂) c, Rej₁ x.1 → Out (f' x) c → Rej c) {n : ℕ} {g₁' : Game β₁} {g₂' : Game β₂}
    {σ : Strategy ((interleave g₁' g₂').bind f')} (hS : Lead f' Dead n g₁' g₂' σ) :
    ∀ {j : ℕ} {i₁ : I₁} {g₁ : Game α₁} (L₁ : Lock T D get₁ R₁ i₁ g₁ g₁') {i₂ : I₂} {g₂ : Game α₂}
      (L₂ : Lock T D get₂ R₂ i₂ g₂ g₂'), DistR slot Rej₁ j L₁ → NoSlot slot L₂ →
        SlotAtDR slot Dead Rej j (Lock.bind hK (interleaveLock hd L₁ L₂)) σ := by
  have hrej : ∀ {g₁' : Game β₁} {g₂' : Game β₂} {σ : Strategy ((interleave g₁' g₂').bind f')} {j : ℕ} {i₁ : I₁}
      {g₁ : Game α₁} (L₁ : Lock T D get₁ R₁ i₁ g₁ g₁') {i₂ : I₂} {g₂ : Game α₂} (L₂ : Lock T D get₂ R₂ i₂ g₂ g₂'),
      (∀ b, Out g₁' b → Rej₁ b) → SlotAtDR slot Dead Rej j (Lock.bind hK (interleaveLock hd L₁ L₂)) σ :=
    fun {_ _ _ _ _ _} _ {_ _} _ hR => .rej fun c hc => by
      obtain ⟨x, hx, hc⟩ := Out.of_bind hc
      exact hRj x c (hR x.1 (Out.of_interleave hx).1) hc
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
      | ret h₂ => exact .of_distR ((hD.interleaveLock_ret hd h₂).bind_noSlot hKn hRj) _
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
        | rej hR => exact hrej _ _ hR
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
        | rej hR => exact hrej _ _ hR
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
        | rej hR => exact hrej _ _ hR
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
        | rej hR => exact hrej _ _ hR
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
        | rej hR => exact hrej _ _ hR
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
        | rej hR => exact hrej _ _ hR

/-- **`SlotAtD.lead_snd` from a distance carrying `rej`**, a rejected outcome of the second lockstep continuing to
rejected ones. -/
theorem SlotAtDR.lead_snd {Rej₂ : β₂ → Prop} (hKn : ∀ i a b (hr : RPair R₁ R₂ i a b), NoSlot slot (hK i a b hr))
    (hRj : ∀ (x : β₁ × β₂) c, Rej₂ x.2 → Out (f' x) c → Rej c) {n : ℕ} {g₁' : Game β₁} {g₂' : Game β₂}
    {σ : Strategy ((interleave g₁' g₂').bind f')} (hS : Lead f' Dead n g₁' g₂' σ) :
    ∀ {j : ℕ} {i₁ : I₁} {g₁ : Game α₁} (L₁ : Lock T D get₁ R₁ i₁ g₁ g₁') {i₂ : I₂} {g₂ : Game α₂}
      (L₂ : Lock T D get₂ R₂ i₂ g₂ g₂'), NoSlot slot L₁ → DistR slot Rej₂ j L₂ →
        SlotAtDR slot Dead Rej (n + j) (Lock.bind hK (interleaveLock hd L₁ L₂)) σ := by
  induction hS with
  | dead hσ =>
      intro _ _ _ _ _ _ _ _ _
      exact .dead hσ
  | done =>
      intro j i₁ g₁ L₁ i₂ g₂ L₂ _ hD
      cases L₁ with
      | ret h₁ =>
        rw [Nat.zero_add]
        exact .of_distR ((hD.embedRight hd h₁).bind_noSlot hKn hRj) _
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

end FlockSoundness.Game.Lock
