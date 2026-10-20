import Proofs.Flock.Claims.KS.SimLead
import Proofs.Flock.Claims.KS.BatchProj

/-!
# Paths of the model's messages, read at the games they move

The simulation of a session moves one rep of one table at a time, along a path of the model's messages through a
context (`Ctx`). The schedule predicates read the strategy at each game of the batch: through the pairing of a batch
position (`Strategy.ofMap`), at the rest of an interleaving (`Lock.projR`) and at its first game (`Lock.projL`). Each
reading of a strategy played along a path at its own game is the path played at the reading:

* `reach_playSteps`, `playSteps_lift_trans`, `playSteps_comp_lift`, `playSteps_congr_lift`: paths and contexts;
* `ofMap_bind_lift`: a path through `bind`'s context, read through the pairing;
* `projR_right_lift`, `projR_right_coin_send`, `projR_left_lift`: a path in the rest, or in the first game, read at
  the rest;
* `projL_ret`, `projL_left_lift`, `projL_left_coin`: a path in the first game, read at it;
* `lead_left_lift`, `lead_left_coin`: rep 0's messages and coins keep `Lock.Lead`, read at the table.
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Tables

section Paths

variable {α : Type}

theorem reach_playSteps : ∀ {g g' : Game α} (s : Steps g g') (ρ : Strategy g') {x : α},
    Lock.Reach g (playSteps s ρ) x ↔ Lock.Reach g' ρ x
  | _, _, .refl _, _, _ => Iff.rfl
  | _, _, .send _ s, ρ, _ => reach_playSteps s ρ

theorem playSteps_lift_trans {A : Type} {C : Game A → Game α} (c : Ctx C) :
    ∀ {g₁ g₂ g₃ : Game A} (s : Steps g₁ g₂) (s' : Steps g₂ g₃) (ρ : Strategy (C g₃)),
      playSteps (c.lift (s.trans s')) ρ = playSteps (c.lift s) (playSteps (c.lift s') ρ)
  | _, _, _, .refl _, _, _ => rfl
  | _, _, _, .send m s, s', ρ => by
      show playSteps ((c.send _ m).trans (c.lift (s.trans s'))) ρ =
        playSteps ((c.send _ m).trans (c.lift s)) (playSteps (c.lift s') ρ)
      rw [playSteps_trans, playSteps_trans, playSteps_lift_trans c s s' ρ]

theorem playSteps_comp_lift {A B : Type} {C : Game A → Game B} {D : Game B → Game α} (c : Ctx C) (d : Ctx D) :
    ∀ {g g' : Game A} (t : Steps g g') (ρ : Strategy (D (C g'))),
      playSteps ((c.comp d).lift t) ρ = playSteps (d.lift (c.lift t)) ρ
  | _, _, .refl _, _ => rfl
  | _, _, .send m t, ρ => by
      show playSteps ((d.lift (c.send _ m)).trans ((c.comp d).lift t)) ρ =
        playSteps (d.lift ((c.send _ m).trans (c.lift t))) ρ
      rw [playSteps_trans, playSteps_lift_trans, playSteps_comp_lift c d t ρ]

theorem playSteps_congr_lift {A : Type} {C C' : Game A → Game α} (c : Ctx C) (h : ∀ g, C g = C' g) :
    ∀ {g g' : Game A} (t : Steps g g') (ρ : Strategy (C' g')) (ρ' : Strategy (C g')), HEq ρ ρ' →
      HEq (playSteps ((c.congr h).lift t) ρ) (playSteps (c.lift t) ρ')
  | _, _, .refl _, _, _, hρ => hρ
  | _, _, .send m t, ρ, ρ', hρ => by
      show HEq (playSteps (((c.congr h).send _ m).trans ((c.congr h).lift t)) ρ)
        (playSteps ((c.send _ m).trans (c.lift t)) ρ')
      rw [playSteps_trans, playSteps_trans]
      exact playSteps_congr_send c h _ m _ _ (playSteps_congr_lift c h t ρ ρ' hρ)

/-- **A path through `bind`'s context**, read through the map, is the path. -/
theorem ofMap_bind_lift {A : Type} (f : A → α) :
    ∀ {g g' : Game A} (t : Steps g g') (ρ : Strategy (g'.map f)),
      Strategy.ofMap f g (playSteps ((Ctx.bind fun a => Game.ret (f a)).lift t) ρ) =
        playSteps t (Strategy.ofMap f g' ρ)
  | _, _, .refl _, _ => rfl
  | _, _, .send m t, ρ => congrArg (Sigma.mk m) (ofMap_bind_lift f t ρ)

end Paths

section Proj

variable {β γ : Type} (Dead' : β → Prop)

/-- **A path in the rest**, read at the rest. -/
theorem projR_right_lift : ∀ (h : Game γ) {g g' : Game β} (t : Steps g g') (ρ : Strategy (interleave h g')),
    Lock.projR Dead' h g (playSteps ((Ctx.right h).lift t) ρ) = playSteps t (Lock.projR Dead' h g' ρ)
  | _, _, _, .refl _, _ => rfl
  | .ret _, _, _, .send m t, ρ => congrArg (Sigma.mk m) (projR_right_lift (.ret _) t ρ)
  | .send _ _, _, _, .send m t, ρ => congrArg (Sigma.mk m) (projR_right_lift (.send _ _) t ρ)
  | @Game.coin _ _ _ _ _, _, _, .send m t, ρ => congrArg (Sigma.mk m) (projR_right_lift (Game.coin _ _) t ρ)

/-- **A coin in the rest beside a game at a send**, read at the rest. -/
theorem projR_right_coin_send {M : Type} (k : M → Game γ) {C : Type} [Fintype C] [Nonempty C] (k₂ : C → Game β)
    (ψ : (c : C) → Strategy (interleave (.send M k) (k₂ c))) :
    Lock.projR Dead' (.send M k) (.coin C k₂) (playSteps ((Ctx.right (.send M k)).coin k₂).steps ψ) =
      fun c => Lock.projR Dead' (.send M k) (k₂ c) (ψ c) :=
  rfl

/-- **A path in the first game**, read at the rest: the first game's moves are not the rest's. -/
theorem projR_left_lift : ∀ (g' : Game β) {h h' : Game γ} (t : Steps h h') (ρ : Strategy (interleave h' g')),
    Lock.projR Dead' h g' (playSteps ((Ctx.left g').lift t) ρ) = Lock.projR Dead' h' g' ρ
  | .ret _, _, _, _, _ => rfl
  | _, _, _, .refl _, _ => rfl
  | .send _ _, _, _, .send _ t, ρ => projR_left_lift (.send _ _) t ρ
  | @Game.coin _ _ _ _ _, _, _, .send _ t, ρ => projR_left_lift (Game.coin _ _) t ρ

/-- `projL` against an ended second game is the first game's strategy read through the pairing. -/
theorem projL_ret (c : γ) : ∀ (g : Game β) (σ : Strategy (interleave g (.ret c))),
    Lock.projL Dead' g (.ret c) σ = Strategy.ofMap (·, c) g (cast (congrArg Strategy (Lock.interleave_ret c g)) σ)
  | .ret _, _ => rfl
  | .send _ _, _ => rfl
  | @Game.coin _ _ _ _ _, _ => rfl

/-- **A path in the first game**, read at it. -/
theorem projL_left_lift : ∀ (h : Game γ) {g g' : Game β} (t : Steps g g') (ρ : Strategy (interleave g' h)),
    Lock.projL Dead' g h (playSteps ((Ctx.left h).lift t) ρ) = playSteps t (Lock.projL Dead' g' h ρ)
  | _, _, _, .refl _, _ => rfl
  | .send _ _, _, _, .send m t, ρ => congrArg (Sigma.mk m) (projL_left_lift (.send _ _) t ρ)
  | @Game.coin _ _ _ _ _, _, _, .send m t, ρ => congrArg (Sigma.mk m) (projL_left_lift (Game.coin _ _) t ρ)
  | .ret c, _, _, @Steps.send _ _ k _ m t, ρ => by
      refine congrArg (Sigma.mk m) ?_
      rw [← projL_left_lift (.ret c) t ρ, projL_ret]
      exact congrArg (Strategy.ofMap (·, c) (k m)) (eq_of_heq (((playSteps_trans _ _ ρ).symm ▸
        playSteps_ofEq_heq _ _ : HEq (playSteps ((Steps.ofEq (Plays.interleave_ret (k m) c).symm).trans
          ((Ctx.left (.ret c)).lift t)) ρ) (playSteps ((Ctx.left (.ret c)).lift t) ρ)).trans (cast_heq _ _).symm))

/-- **A coin in the first game**, read at it. -/
theorem projL_left_coin : ∀ (h : Game γ) {C : Type} [Fintype C] [Nonempty C] (k : C → Game β)
    (ψ : Strategy (.coin C ((Ctx.left h).coin k).next)),
    Lock.projL Dead' (.coin C k) h (playSteps ((Ctx.left h).coin k).steps ψ) =
      fun c => Lock.projL Dead' (k c) h (cast (congrArg Strategy (((Ctx.left h).coin k).eq c)) (ψ c))
  | .ret b, _, _, _, k, ψ => by
      funext c
      rw [projL_ret Dead' b (k c), cast_cast]
      exact congrArg (Strategy.ofMap (·, b) (k c)) (eq_of_heq (cast_heq _ (ψ c))).symm
  | .send _ _, _, _, _, _, _ => rfl
  | @Game.coin _ _ _ _ _, _, _, _, _, _ => rfl

end Proj

section Lead

variable {α β γ : Type} {f : α × β → Game γ} {D : γ → Prop}

/-- **Rep 0's messages keep the lead**, read at the table. -/
theorem lead_left_lift : ∀ (g₂ : Game β) {g g' : Game α} (t : Steps g g')
    (ρ : Strategy ((interleave g' g₂).bind f)) {n : ℕ}, Lock.Lead f D n g' g₂ ρ →
      Lock.Lead f D n g g₂ (playSteps ((Ctx.bind f).lift ((Ctx.left g₂).lift t)) ρ)
  | .ret _, _, _, _, _, _, _ => .last
  | _, _, _, .refl _, _, _, hL => hL
  | .send _ k₂, _, _, @Steps.send _ _ k _ m t, ρ, _, hL => by
      have e : playSteps ((Ctx.bind f).lift ((Ctx.left (Game.send _ k₂)).lift (Steps.send m t))) ρ =
          playSteps ((Ctx.bind f).lift ((Ctx.left (Game.send _ k₂)).send k m))
            (playSteps ((Ctx.bind f).lift ((Ctx.left (Game.send _ k₂)).lift t)) ρ) :=
        playSteps_lift_trans (Ctx.bind f) ((Ctx.left (Game.send _ k₂)).send k m) ((Ctx.left _).lift t) ρ
      rw [e]
      exact .sendSend m (lead_left_lift (.send _ k₂) t ρ hL)
  | @Game.coin _ _ _ _ k₂, _, _, @Steps.send _ _ k _ m t, ρ, _, hL => by
      have e : playSteps ((Ctx.bind f).lift ((Ctx.left (Game.coin _ k₂)).lift (Steps.send m t))) ρ =
          playSteps ((Ctx.bind f).lift ((Ctx.left (Game.coin _ k₂)).send k m))
            (playSteps ((Ctx.bind f).lift ((Ctx.left (Game.coin _ k₂)).lift t)) ρ) :=
        playSteps_lift_trans (Ctx.bind f) ((Ctx.left (Game.coin _ k₂)).send k m) ((Ctx.left _).lift t) ρ
      rw [e]
      exact .sendCoin m (lead_left_lift (Game.coin _ k₂) t ρ hL)

/-- **A coin of rep 0 keeps the lead** with one coin fewer, read at the table. -/
theorem lead_left_coin : ∀ (g₂ : Game β) {C : Type} [Fintype C] [Nonempty C] (k : C → Game α)
    (ψ : Strategy (.coin C (((Ctx.left g₂).comp (Ctx.bind f)).coin k).next)) {n : ℕ},
    (∀ x, Lock.Lead f D n (k x) g₂
      (cast (congrArg Strategy ((((Ctx.left g₂).comp (Ctx.bind f)).coin k).eq x)) (ψ x))) →
    Lock.Lead f D (n + 1) (.coin C k) g₂ (playSteps (((Ctx.left g₂).comp (Ctx.bind f)).coin k).steps ψ)
  | .ret _, _, _, _, _, _, _, _ => .last
  | .send _ _, _, _, _, _, _, _, hL => .coinSend hL
  | @Game.coin _ _ _ _ _, _, _, _, _, _, _, hL => .coinCoin hL

end Lead

end FlockDeploy.KS
