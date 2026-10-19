import Proofs.Flock.Claims.KS.SlotBatch
import Proofs.Flock.Soundness.Discharge.ZkLink.Push

/-!
# A schedule through the pushing chooser

The primed prover `pushStrat` is `Lock.oracleStrat pushC` along `pushSess`: it plays the compiled prover's moves and
sends the constant cap at every digest. The schedule predicates of `SlotBatch` and `SlotLead` read only a strategy's
choices at `interleave`'s nodes, the messages it sends and the outcomes it reaches, so each holds of the primed prover
where it holds of the compiled one:

* `pushed_upd`, `reach_push`, `reach_push'`, `ofMap_oracleStrat_map`: the pushing chooser keeps every cap pushed,
  reaches exactly the outcomes the compiled prover reaches, and commutes with relabelling;
* `firstS_push`, `followR_push`, `followL_push`, `preSched_push`: `Lock.FirstS`, `Lock.FollowR`, `Lock.FollowL` and
  `Lock.PreSched` through `Lock.rightSame`, `Lock.leftSame` and `batchPush`;
* `lead_push`: `Lock.Lead` through two interleaved locksteps and what follows them.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model

section Transport

variable {D I α : Type} {get : I → ℕ → Option (Cap D × D)}

/-- **A table node keeps every cap pushed** when the chooser sends the constant cap. -/
theorem pushed_upd {i : I} {slot : ℕ} {upd : Cap D → D → I} (hset : ∀ t d, get (upd t d) slot = some (t, d))
    (hkeep : ∀ t d s, s ≠ slot → get (upd t d) s = get i s) (hi : Pushed get i) (d : D) :
    Pushed get (upd (fun _ => d) d) := by
  intro s' c d' h
  by_cases hs : s' = slot
  · subst hs
    rw [hset] at h
    cases h
    rfl
  · rw [hkeep _ _ _ hs] at h
    exact hi s' c d' h

/-- **The primed prover reaches only outcomes the compiled prover reaches**, along a lockstep whose leaves relate
equal outputs while every cap is pushed. -/
theorem reach_push {R : I → α → α → Prop} (hR : ∀ i a b, R i a b → Pushed get i → a = b) {x : α} :
    ∀ {i : I} {g g' : Game α} (L : Lock (Cap D) D get R i g g') (σ' : Strategy g'), Pushed get i →
      Lock.Reach g (Lock.oracleStrat pushC L σ') x → Lock.Reach g' σ' x
  | _, _, _, .ret hr, _, hi, h => (hR _ _ _ hr hi).symm.trans h
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi, h => by
      obtain ⟨c, hc⟩ := h
      exact ⟨c, reach_push hR (L c) (σ' c) hi hc⟩
  | _, _, _, .send _ L, σ', hi, h => reach_push hR (L σ'.1) σ'.2 hi h
  | _, _, _, .table _ _ _ hs hk L, σ', hi, h => reach_push hR (L (fun _ => σ'.1) σ'.1) σ'.2 (pushed_upd hs hk hi σ'.1) h

/-- **The primed prover reaches every outcome the compiled prover reaches**, along a lockstep whose leaves relate
equal outputs while every cap is pushed. -/
theorem reach_push' {R : I → α → α → Prop} (hR : ∀ i a b, R i a b → Pushed get i → a = b) {x : α} :
    ∀ {i : I} {g g' : Game α} (L : Lock (Cap D) D get R i g g') (σ' : Strategy g'), Pushed get i →
      Lock.Reach g' σ' x → Lock.Reach g (Lock.oracleStrat pushC L σ') x
  | _, _, _, .ret hr, _, hi, h => (hR _ _ _ hr hi).trans h
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi, h => by
      obtain ⟨c, hc⟩ := h
      exact ⟨c, reach_push' hR (L c) (σ' c) hi hc⟩
  | _, _, _, .send _ L, σ', hi, h => reach_push' hR (L σ'.1) σ'.2 hi h
  | _, _, _, .table _ _ _ hs hk L, σ', hi, h =>
      reach_push' hR (L (fun _ => σ'.1) σ'.1) σ'.2 (pushed_upd hs hk hi σ'.1) h

/-- **The pushing chooser commutes with relabelling.** -/
theorem ofMap_oracleStrat_map {β α₂ β₂ : Type} {R : I → α → β → Prop} {R₂ : I → α₂ → β₂ → Prop} (f : α → α₂)
    (f' : β → β₂) (h : ∀ i a b, R i a b → R₂ i (f a) (f' b)) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock (Cap D) D get R i g g') (σ' : Strategy (g'.map f')),
      Strategy.ofMap f g (Lock.oracleStrat pushC (Lock.map f f' h L) σ') =
        Lock.oracleStrat pushC L (Strategy.ofMap f' g' σ')
  | _, _, _, .ret _, _ => rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' =>
      funext fun c => ofMap_oracleStrat_map f f' h (L c) (σ' c)
  | _, _, _, .send _ L, σ' => congrArg (Sigma.mk σ'.1) (ofMap_oracleStrat_map f f' h (L σ'.1) σ'.2)
  | _, _, _, .table _ _ _ _ _ L, σ' =>
      congrArg (Sigma.mk fun _ => σ'.1) (ofMap_oracleStrat_map f f' h (L (fun _ => σ'.1) σ'.1) σ'.2)

variable {γ : Type} {R : I → α → α → Prop}

/-- Unfold the primed prover one node into `Lock.rightSame` or `Lock.leftSame`. -/
local macro "push_unfold" : tactic =>
  `(tactic| simp only [Lock.rightSame, Lock.lockWithL, Lock.leftSame, Lock.lockWithSame, Lock.oracleStrat])

/-- **`FirstS` through the pushing chooser**: the game in front advances as the compiled prover advances it, the
lockstep beside it unmoved. -/
theorem firstS_push (hR : ∀ i a b, R i a b → Pushed get i → a = b) {Dead : γ × α → Prop}
    {Q Q' : (h : Game γ) → (g' : Game α) → Strategy (interleave h g') → Prop} {n : ℕ} {h : Game γ} {g' : Game α}
    {σ : Strategy (interleave h g')} (hF : Lock.FirstS Dead Q n h g' σ) :
    ∀ {i : I} {g : Game α} (L : Lock (Cap D) D get R i g g'), Pushed get i →
      (∀ h' σ', Q h' g' σ' → Q' h' g (Lock.oracleStrat pushC (Lock.rightSame h' L) σ')) →
        Lock.FirstS Dead Q' n h g (Lock.oracleStrat pushC (Lock.rightSame h L) σ) := by
  have hR' : ∀ i (a b : γ × α), a.1 = b.1 ∧ R i a.2 b.2 → Pushed get i → a = b :=
    fun i a b hr hp => Prod.ext hr.1 (hR i _ _ hr.2 hp)
  induction hF with
  | dead hσ =>
      intro i g L hi _
      exact .dead fun x hx => hσ x (reach_push hR' _ _ hi hx)
  | stop hq =>
      intro i g L hi hQ
      exact .stop (hQ _ _ hq)
  | last =>
      intro i g L hi _
      cases L with
      | ret hr => exact .last
  | coinCoin _ ih =>
      intro i g L hi hQ
      cases L with
      | coin _ L =>
        push_unfold
        exact .coinCoin fun c => ih c (.coin _ L) hi hQ
  | coinSend _ ih =>
      intro i g L hi hQ
      cases L with
      | send _ L =>
        push_unfold
        exact .coinSend fun c => ih c (.send _ L) hi hQ
      | table s upd hf hs hk L =>
        push_unfold
        exact .coinSend fun c => ih c (.table s upd hf hs hk L) hi hQ
  | sendCoin m _ ih =>
      intro i g L hi hQ
      cases L with
      | coin _ L =>
        push_unfold
        exact .sendCoin m (ih (.coin _ L) hi hQ)
  | sendSend m _ ih =>
      intro i g L hi hQ
      cases L with
      | send _ L =>
        push_unfold
        exact .sendSend m (ih (.send _ L) hi hQ)
      | table s upd hf hs hk L =>
        push_unfold
        exact .sendSend m (ih (.table s upd hf hs hk L) hi hQ)

/-- **`FollowR` through the pushing chooser**: the rest follows the primed prover of `τ`. -/
theorem followR_push (hR : ∀ i a b, R i a b → Pushed get i → a = b) {Dead : γ × α → Prop} {h : Game γ}
    {g' : Game α} {σ : Strategy (interleave h g')} {τ : Strategy g'} (hF : Lock.FollowR Dead h g' σ τ) :
    ∀ {i : I} {g : Game α} (L : Lock (Cap D) D get R i g g'), Pushed get i →
      Lock.FollowR Dead h g (Lock.oracleStrat pushC (Lock.rightSame h L) σ) (Lock.oracleStrat pushC L τ) := by
  have hR' : ∀ i (a b : γ × α), a.1 = b.1 ∧ R i a.2 b.2 → Pushed get i → a = b :=
    fun i a b hr hp => Prod.ext hr.1 (hR i _ _ hr.2 hp)
  induction hF with
  | dead hσ =>
      intro i g L hi
      exact .dead fun x hx => hσ x (reach_push hR' _ _ hi hx)
  | @hret c g' σ =>
      intro i g L hi
      have e : Strategy.ofMap (fun b => (c, b)) g (Lock.oracleStrat pushC (Lock.rightSame (.ret c) L) σ) =
          Lock.oracleStrat pushC L (Strategy.ofMap (fun b => (c, b)) g' σ) :=
        ofMap_oracleStrat_map _ _ _ L σ
      rw [← e]
      exact .hret _
  | iret =>
      intro i g L hi
      cases L with
      | ret hr => exact .iret
  | hsendCoin m _ ih =>
      intro i g L hi
      cases L with
      | coin _ L =>
        push_unfold
        exact .hsendCoin m (ih (.coin _ L) hi)
  | hsendSend m _ ih =>
      intro i g L hi
      cases L with
      | send _ L =>
        push_unfold
        exact .hsendSend m (ih (.send _ L) hi)
      | table s upd hf hs hk L =>
        push_unfold
        exact .hsendSend m (ih (.table s upd hf hs hk L) hi)
  | icoinS _ ih =>
      intro i g L hi
      cases L with
      | coin _ L =>
        push_unfold
        exact .icoinS fun c => ih c (L c) hi
  | icoinC _ ih =>
      intro i g L hi
      cases L with
      | coin _ L =>
        push_unfold
        exact .icoinC fun c => ih c (L c) hi
  | isendS m₂ _ ih =>
      intro i g L hi
      cases L with
      | send _ L =>
        push_unfold
        exact .isendS m₂ (ih (L m₂) hi)
      | table s upd hf hs hk L =>
        push_unfold
        exact .isendS (fun _ => m₂) (ih (L (fun _ => m₂) m₂) (pushed_upd hs hk hi m₂))
  | isendC m₂ _ ih =>
      intro i g L hi
      cases L with
      | send _ L =>
        push_unfold
        exact .isendC m₂ (ih (L m₂) hi)
      | table s upd hf hs hk L =>
        push_unfold
        exact .isendC (fun _ => m₂) (ih (L (fun _ => m₂) m₂) (pushed_upd hs hk hi m₂))

/-- **`FollowL` through the pushing chooser**: the lockstep's game follows the primed prover of `τ`. -/
theorem followL_push (hR : ∀ i a b, R i a b → Pushed get i → a = b) {Dead : α × γ → Prop} {Dead' : α → Prop}
    {b : Option ℕ} {g' : Game α} {h : Game γ} {σ : Strategy (interleave g' h)} {τ : Strategy g'}
    (hF : Lock.FollowL Dead Dead' b g' h σ τ) :
    ∀ {i : I} {g : Game α} (L : Lock (Cap D) D get R i g g'), Pushed get i →
      Lock.FollowL Dead Dead' b g h (Lock.oracleStrat pushC (Lock.leftSame L h) σ) (Lock.oracleStrat pushC L τ) := by
  have hR' : ∀ i (a b : α × γ), R i a.1 b.1 ∧ a.2 = b.2 → Pushed get i → a = b :=
    fun i a b hr hp => Prod.ext (hR i _ _ hr.1 hp) hr.2
  induction hF with
  | dead hσ =>
      intro i g L hi
      exact .dead fun x hx => hσ x (reach_push hR' _ _ hi hx)
  | spent hs =>
      intro i g L hi
      exact .spent fun hτ x hx => hs (fun y hy => hτ y (reach_push' hR L _ hi hy)) x (reach_push hR' _ _ hi hx)
  | gret =>
      intro i g L hi
      cases L with
      | ret hr => exact .gret
  | @hret b g' c σ =>
      intro i g L hi
      have e := ofMap_oracleStrat_map (R₂ := fun i a b => R i a.1 b.1 ∧ a.2 = b.2) (fun a => (a, c))
        (fun b => (b, c)) (fun _ _ _ hr => ⟨hr, rfl⟩) L (cast (congrArg Strategy (Lock.interleave_ret c g')) σ)
      have key := Lock.FollowL.hret (Dead := Dead) (Dead' := Dead') (b := b)
        (Lock.oracleStrat pushC (Lock.leftSame L (.ret c)) σ)
      cases L with
      | ret hr => exact .gret
      | coin _ L =>
        rw [← e]
        exact key
      | send _ L =>
        rw [← e]
        exact key
      | table s upd hf hs hk L =>
        rw [← e]
        exact key
  | gcoinS _ ih =>
      intro i g L hi
      cases L with
      | coin _ L =>
        push_unfold
        exact .gcoinS fun c => ih c (L c) hi
  | gcoinC _ ih =>
      intro i g L hi
      cases L with
      | coin _ L =>
        push_unfold
        exact .gcoinC fun c => ih c (L c) hi
  | gsendS m _ ih =>
      intro i g L hi
      cases L with
      | send _ L =>
        push_unfold
        exact .gsendS m (ih (L m) hi)
      | table s upd hf hs hk L =>
        push_unfold
        exact .gsendS (fun _ => m) (ih (L (fun _ => m) m) (pushed_upd hs hk hi m))
  | gsendC m _ ih =>
      intro i g L hi
      cases L with
      | send _ L =>
        push_unfold
        exact .gsendC m (ih (L m) hi)
      | table s upd hf hs hk L =>
        push_unfold
        exact .gsendC (fun _ => m) (ih (L (fun _ => m) m) (pushed_upd hs hk hi m))
  | hsendS m₂ _ ih =>
      intro i g L hi
      cases L with
      | send _ L =>
        push_unfold
        exact .hsendS m₂ (ih (.send _ L) hi)
      | table s upd hf hs hk L =>
        push_unfold
        exact .hsendS m₂ (ih (.table s upd hf hs hk L) hi)
  | hsendC m₂ _ ih =>
      intro i g L hi
      cases L with
      | coin _ L =>
        push_unfold
        exact .hsendC m₂ (ih (.coin _ L) hi)

/-- **`PreSched` through the pushing chooser** along `batchPush`: each game in front advances and then moves by
sends as the compiled prover's does, and where the games in front pause the rest is pushed. -/
theorem preSched_push {i : I} {g g' : Game α} (L : Lock (Cap D) D get (fun i a b => Pushed get i → a = b) i g g')
    (post : List (Game α)) (hi : Pushed get i) {Q : (List α → Prop) → Strategy (batch (g' :: post)) → Prop}
    {Q' : (List α → Prop) → Strategy (batch (g :: post)) → Prop}
    (hQ : ∀ Dead σ', Q Dead σ' → Q' Dead (Lock.oracleStrat pushC (batchPush (Pushed get) L post []) σ')) :
    ∀ (pre : List (Game α)) {Dead : List α → Prop} {n : ℕ} {σ : Strategy (batch (pre ++ g' :: post))},
      Lock.PreSched g' post pre Dead Q n σ →
        Lock.PreSched g post pre Dead Q' n (Lock.oracleStrat pushC (batchPush (Pushed get) L post pre) σ)
  | [], Dead, _, σ, ⟨hn, hq⟩ => ⟨hn, hQ Dead σ hq⟩
  | h :: pre, Dead, _, σ, ⟨n₀, n₁, hn, hF⟩ => by
      refine ⟨n₀, n₁, hn, ?_⟩
      have e : Strategy.ofMap (fun x : α × List α => x.1 :: x.2) (h.interleave (batch (pre ++ g :: post)))
          (Lock.oracleStrat pushC (batchPush (Pushed get) L post (h :: pre)) σ) =
          Lock.oracleStrat pushC (Lock.rightSame h (batchPush (Pushed get) L post pre))
            (Strategy.ofMap (fun x : α × List α => x.1 :: x.2) (h.interleave (batch (pre ++ g' :: post))) σ) :=
        ofMap_oracleStrat_map _ _ _ _ σ
      refine (congrArg (Lock.FirstS _ _ n₀ h _) e).mpr (firstS_push (fun _ _ _ hr hp => hr hp) hF _ hi ?_)
      rintro h' σ' ⟨e', τ, hR, hP⟩
      have hP' : Lock.PreSched g' post pre (fun outs => ∀ c, Dead (c :: outs)) Q n₁ τ := hP
      exact ⟨rfl, Lock.oracleStrat pushC (batchPush (Pushed get) L post pre) τ,
        followR_push (fun _ _ _ hr hp => hr hp) hR _ hi, preSched_push L post hi hQ pre hP'⟩

/-- Unfold the primed prover one node into two interleaved locksteps. -/
local macro "lead_unfold" : tactic =>
  `(tactic| simp only [Lock.interleaveLock, Lock.lockWith, Lock.bind, Lock.oracleStrat])

/-- **`Lead` through the pushing chooser**, along two interleaved locksteps and what follows them. -/
theorem lead_push [Inhabited D] {α₁ α₂ γ' I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (Cap D × D)}
    {get₂ : I₂ → ℕ → Option (Cap D × D)} {R₁ : I₁ → α₁ → α₁ → Prop} {R₂ : I₂ → α₂ → α₂ → Prop}
    (hd : Lock.Disjoint get₁ get₂) {f : α₁ × α₂ → Game γ'} {R₃ : I₁ × I₂ → γ' → γ' → Prop}
    (hR : ∀ i a b, R₃ i a b → Pushed (Lock.getPair get₁ get₂) i → a = b)
    {hK : ∀ i a b, Lock.RPair R₁ R₂ i a b → Lock (Cap D) D (Lock.getPair get₁ get₂) R₃ i (f a) (f b)}
    {Dead : γ' → Prop} {n : ℕ} {g₁' : Game α₁} {g₂' : Game α₂} {σ : Strategy ((interleave g₁' g₂').bind f)}
    (hS : Lock.Lead f Dead n g₁' g₂' σ) :
    ∀ {i₁ : I₁} {g₁ : Game α₁} (L₁ : Lock (Cap D) D get₁ R₁ i₁ g₁ g₁') {i₂ : I₂} {g₂ : Game α₂}
      (L₂ : Lock (Cap D) D get₂ R₂ i₂ g₂ g₂'), Pushed (Lock.getPair get₁ get₂) (i₁, i₂) →
        Lock.Lead f Dead n g₁ g₂ (Lock.oracleStrat pushC (Lock.bind hK (Lock.interleaveLock hd L₁ L₂)) σ) := by
  induction hS with
  | dead hσ =>
      intro i₁ g₁ L₁ i₂ g₂ L₂ hi
      exact .dead fun x hx => hσ x (reach_push hR _ _ hi hx)
  | done =>
      intro i₁ g₁ L₁ i₂ g₂ L₂ hi
      cases L₁ with
      | ret hr => exact .done
  | last =>
      intro i₁ g₁ L₁ i₂ g₂ L₂ hi
      cases L₂ with
      | ret hr => exact .last
  | coinCoin _ ih =>
      intro i₁ g₁ L₁ i₂ g₂ L₂ hi
      cases L₁ with
      | coin _ L₁ =>
        cases L₂ with
        | coin _ L₂ =>
          lead_unfold
          exact .coinCoin fun c => ih c (L₁ c) (.coin _ L₂) hi
  | coinSend _ ih =>
      intro i₁ g₁ L₁ i₂ g₂ L₂ hi
      cases L₁ with
      | coin _ L₁ =>
        cases L₂ with
        | send _ L₂ =>
          lead_unfold
          exact .coinSend fun c => ih c (L₁ c) (.send _ L₂) hi
        | table s upd hf hs hk L₂ =>
          lead_unfold
          exact .coinSend fun c => ih c (L₁ c) (.table s upd hf hs hk L₂) hi
  | sendCoin m _ ih =>
      intro i₁ g₁ L₁ i₂ g₂ L₂ hi
      cases L₂ with
      | coin _ L₂ =>
        cases L₁ with
        | send _ L₁ =>
          lead_unfold
          exact .sendCoin m (ih (L₁ m) (.coin _ L₂) hi)
        | table s upd hf hs hk L₁ =>
          lead_unfold
          exact .sendCoin (fun _ => m) (ih (L₁ (fun _ => m) m) (.coin _ L₂)
            (pushed_upd (Lock.getPair_set₁ hs _) (fun t d s h => Lock.getPair_keep₁ hk _ t d s h) hi m))
  | sendSend m _ ih =>
      intro i₁ g₁ L₁ i₂ g₂ L₂ hi
      cases L₁ with
      | send _ L₁ =>
        cases L₂ with
        | send _ L₂ =>
          lead_unfold
          exact .sendSend m (ih (L₁ m) (.send _ L₂) hi)
        | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
          lead_unfold
          exact .sendSend m (ih (L₁ m) (.table s₂ upd₂ hf₂ hs₂ hk₂ L₂) hi)
      | table s upd hf hs hk L₁ =>
        cases L₂ with
        | send _ L₂ =>
          lead_unfold
          exact .sendSend (fun _ => m) (ih (L₁ (fun _ => m) m) (.send _ L₂)
            (pushed_upd (Lock.getPair_set₁ hs _) (fun t d s h => Lock.getPair_keep₁ hk _ t d s h) hi m))
        | table s₂ upd₂ hf₂ hs₂ hk₂ L₂ =>
          lead_unfold
          exact .sendSend (fun _ => m) (ih (L₁ (fun _ => m) m) (.table s₂ upd₂ hf₂ hs₂ hk₂ L₂)
            (pushed_upd (Lock.getPair_set₁ hs _) (fun t d s h => Lock.getPair_keep₁ hk _ t d s h) hi m))

end Transport

end FlockSoundness.Discharge.ZkLink
