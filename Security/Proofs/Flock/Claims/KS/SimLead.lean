import Proofs.Flock.Claims.KS.SlotLead
import Proofs.Flock.Claims.Bridge.Returns

/-!
# The simulated strategy leads with rep 0

The simulation of a one-table session (`simRJ`) follows the live prover round by round. When the live prover submits
its next `r₀` rounds on rep 0 and then only rep 1's, the simulated strategy, read at its one table (`unB`), is to lead
with rep 0 for `r₀` coins (`Lock.Lead`), except below a part of the run whose every reached outcome is a dead
branch's, answering the openings with `nul`. This file has the steps that reading takes.

* `playSteps`: a strategy along a path of the model's messages;
* `unB`: a table's strategy out of its one-position batch's, and how it reads a step of rep 0 (`unB_ctxAt_send₁`,
  `unB_ctxAt_coin₁`);
* `reach_mapDead`: what a simulated strategy reaches, by the simulation's returns and dead branches.
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Tables Claims.Bridge

section Steps

variable {α : Type}

/-- **A strategy along a path of the model's messages**: the path's messages, then `σ`. -/
def playSteps : {g g' : Game α} → Steps g g' → Strategy g' → Strategy g
  | _, _, .refl _, σ => σ
  | _, _, .send m s, σ => ⟨m, playSteps s σ⟩

theorem playSteps_trans : ∀ {g₁ g₂ g₃ : Game α} (s : Steps g₁ g₂) (t : Steps g₂ g₃) (σ : Strategy g₃),
    playSteps (s.trans t) σ = playSteps s (playSteps t σ)
  | _, _, _, .refl _, _, _ => rfl
  | _, _, _, .send m s, t, σ => congrArg (Sigma.mk m) (playSteps_trans s t σ)

theorem playSteps_ofEq_heq : ∀ {g g' : Game α} (h : g = g') (σ : Strategy g'), HEq (playSteps (Steps.ofEq h) σ) σ
  | _, _, rfl, _ => HEq.rfl

theorem playSteps_lift_ofEq_heq {A : Type} {C : Game A → Game α} (c : Ctx C) {g g' : Game A} (h : g = g')
    (σ : Strategy (C g')) : HEq (playSteps (c.lift (Steps.ofEq h)) σ) σ := by
  subst h
  rfl

/-- A context up to pointwise equality sends what it sends. -/
theorem playSteps_congr_send {A : Type} {C C' : Game A → Game α} (c : Ctx C) (h : ∀ g, C g = C' g) {M : Type}
    (k : M → Game A) (m : M) (σ : Strategy (C' (k m))) (σ' : Strategy (C (k m))) (hσ : HEq σ σ') :
    HEq (playSteps ((c.congr h).send k m) σ) (playSteps (c.send k m) σ') := by
  obtain rfl : C = C' := funext h
  obtain rfl := eq_of_heq hσ
  show HEq (playSteps ((Steps.ofEq (h _).symm).trans ((c.send k m).trans (Steps.ofEq (h _)))) σ) _
  rw [playSteps_trans, playSteps_trans, eq_of_heq (playSteps_ofEq_heq (h (k m)) σ)]
  exact playSteps_ofEq_heq _ _

/-- A context up to pointwise equality takes the path to a coin it takes. -/
theorem playSteps_congr_coin {A : Type} {C C' : Game A → Game α} (c : Ctx C) (h : ∀ g, C g = C' g) {Cm : Type}
    [Fintype Cm] [Nonempty Cm] (k : Cm → Game A) (ψ : Strategy (.coin Cm (c.coin k).next)) :
    HEq (playSteps ((c.congr h).coin k).steps ψ) (playSteps (c.coin k).steps ψ) := by
  obtain rfl : C = C' := funext h
  show HEq (playSteps ((Steps.ofEq (h _).symm).trans (c.coin k).steps) ψ) _
  rw [playSteps_trans]
  exact playSteps_ofEq_heq _ _

end Steps

/-! ## A simulated strategy -/

section SimStrategy

variable {β α : Type} {pL : β → Prop} {pg : α → Prop}

theorem strategy_ofSteps {L : Game β} :
    ∀ {g g' : Game α} (s : Steps g g') (d : Sim pL pg L g') (σ : Strategy L),
      (Sim.ofSteps s d).strategy σ = playSteps s (d.strategy σ)
  | _, _, .refl _, _, _ => rfl
  | _, _, .send m s, d, σ => congrArg (Sigma.mk m) (strategy_ofSteps s d σ)

theorem mapDead_ofSteps (f : (g : Game α) → Strategy g → Strategy g) {L : Game β} :
    ∀ {g g' : Game α} (s : Steps g g') (d : Sim pL pg L g'),
      Sim.mapDead f (Sim.ofSteps s d) = Sim.ofSteps s (Sim.mapDead f d)
  | _, _, .refl _, _ => rfl
  | _, _, .send m s, d => congrArg (Sim.model m) (mapDead_ofSteps f s d)

theorem mapDead_castR (f : (g : Game α) → Strategy g → Strategy g) {L : Game β} {g g' : Game α} (h : g = g')
    (d : Sim pL pg L g) : Sim.mapDead f (Sim.castR h d) = Sim.castR h (Sim.mapDead f d) := by
  subst h
  rfl

theorem strategy_castR {L : Game β} {g g' : Game α} (h : g = g') (d : Sim pL pg L g) (σ : Strategy L) :
    (Sim.castR h d).strategy σ = cast (congrArg Strategy h) (d.strategy σ) := by
  subst h
  rfl

theorem all_ofSteps_inv {R : β → α → Prop} {D : (g : Game α) → Strategy g → Prop} {L : Game β} :
    ∀ {g g' : Game α} (s : Steps g g') (d : Sim pL pg L g'), Sim.All R D (Sim.ofSteps s d) → Sim.All R D d
  | _, _, .refl _, _, h => h
  | _, _, .send _ s, d, h => all_ofSteps_inv s d h

theorem all_castR_inv {R : β → α → Prop} {D : (g : Game α) → Strategy g → Prop} {L : Game β} {g g' : Game α}
    (h : g = g') (d : Sim pL pg L g) (hd : Sim.All R D (Sim.castR h d)) : Sim.All R D d := by
  subst h
  exact hd

theorem out_of_reach : ∀ {L : Game β} {s : Strategy L} {b : β}, Lock.Reach L s b → Out L b
  | .ret _, _, _, h => by cases h; exact .ret _
  | .send _ _, s, _, h => .send s.1 (out_of_reach h)
  | @Game.coin _ _ _ _ _, _, _, ⟨c, h⟩ => .coin c (out_of_reach h)

/-- **What a simulated strategy reaches**: a return's model outcome, whose live outcome the live strategy reaches, or
an outcome a dead branch's replaced strategy reaches. -/
theorem reach_mapDead (f : (g : Game α) → Strategy g → Strategy g) {R : β → α → Prop}
    {D : (g : Game α) → Strategy g → Prop} :
    ∀ {L : Game β} {g : Game α} (d : Sim pL pg L g), Sim.All R D d → ∀ (s : Strategy L) (a : α),
      Lock.Reach g ((Sim.mapDead f d).strategy s) a →
        (∃ b, Lock.Reach L s b ∧ R b a) ∨ ∃ (g' : Game α) (σ : Strategy g'), D g' σ ∧ Lock.Reach g' (f g' σ) a
  | _, _, @Sim.ret _ _ _ _ b _ _, hA, _, _, h => .inl ⟨b, rfl, h ▸ hA⟩
  | _, _, .model _ d, hA, s, a, h => reach_mapDead f d hA s a h
  | _, _, .live k, hA, s, a, h => reach_mapDead f (k s.1) (hA s.1) s.2 a h
  | _, _, .coin e k, hA, s, a, ⟨c, h⟩ =>
      (reach_mapDead f (k c) (hA c) (s (e c)) a h).imp (fun ⟨b, hb, hr⟩ => ⟨b, ⟨e c, hb⟩, hr⟩) id
  | _, g, .dead _ σ, hA, _, _, h => .inr ⟨g, σ, hA, h⟩

end SimStrategy

/-! ## A table's strategy out of its one-position batch's -/

section One

variable {A : Type}

/-- **A table's strategy out of its one-position batch's.** -/
noncomputable def unB (X : Game A) (σ : Strategy (batch [X])) : Strategy X :=
  Strategy.ofMap (·, ([] : List A)) _ (cast (congrArg Strategy (Lock.interleave_ret [] X))
    (Strategy.ofMap (fun x : A × List A => x.1 :: x.2) (X.interleave (.ret [])) σ))

theorem ofMap_heq {B : Type} (f : A → B) {g g' : Game A} (e : g = g') {σ : Strategy (g.map f)}
    {σ' : Strategy (g'.map f)} (h : HEq σ σ') : HEq (Strategy.ofMap f g σ) (Strategy.ofMap f g' σ') := by
  subst e
  obtain rfl := eq_of_heq h
  rfl

theorem unB_eq (X : Game A) (σ : Strategy (batch [X]))
    (ρ : Strategy ((X.map (·, ([] : List A))).map fun x : A × List A => x.1 :: x.2)) (h : HEq ρ σ) :
    Strategy.ofMap (·, ([] : List A)) X
      (Strategy.ofMap (fun x : A × List A => x.1 :: x.2) (X.map (·, ([] : List A))) ρ) = unB X σ :=
  congrArg (Strategy.ofMap (·, ([] : List A)) X) (eq_of_heq
    ((ofMap_heq (fun x : A × List A => x.1 :: x.2) (Lock.interleave_ret [] X).symm h).trans (cast_heq _ _).symm))

theorem reach_ofMap {B : Type} (f : A → B) : ∀ (g : Game A) (σ : Strategy (g.map f)) (a : A),
    Lock.Reach g (Strategy.ofMap f g σ) a → Lock.Reach (g.map f) σ (f a)
  | .ret _, _, _, h => congrArg f h
  | .send _ k, σ, a, h => reach_ofMap f (k σ.1) σ.2 a h
  | @Game.coin _ _ _ _ k, σ, a, ⟨c, h⟩ => ⟨c, reach_ofMap f (k c) (σ c) a h⟩

theorem reach_cast {g g' : Game A} (e : g = g') (σ : Strategy g) (a : A) :
    Lock.Reach g' (cast (congrArg Strategy e) σ) a ↔ Lock.Reach g σ a := by
  subst e
  rfl

theorem reach_unB (X : Game A) (σ : Strategy (batch [X])) (c : A) (h : Lock.Reach X (unB X σ) c) :
    Lock.Reach (batch [X]) σ [c] :=
  reach_ofMap (fun x : A × List A => x.1 :: x.2) (X.interleave (.ret [])) σ (c, [])
    ((reach_cast (Lock.interleave_ret [] X) _ _).1 (reach_ofMap (·, ([] : List A)) X _ c h))

theorem reach_answerWith {Op : Type} (nul : Op) : ∀ {g : Game A}, AllInh g → ∀ (σ : Strategy g) (a : A),
    Lock.Reach g (answerWith nul g σ) a → OutN nul g a
  | .ret _, _, _, _, h => by cases h; exact .ret _
  | .send M k, hg, σ, a, h => by
      by_cases hc : ∃ e : Op = M, Nonempty (Strategy (k (cast e nul)))
      · unfold answerWith at h
        rw [dite_eq_left_of_eq_true (eq_true hc)] at h
        exact .send (cast hc.fst nul) (fun _ => rfl) (reach_answerWith nul (hg.send_k _) _ a h)
      · unfold answerWith at h
        rw [dite_eq_right_of_eq_false (eq_false hc)] at h
        exact .send σ.1 (fun e => absurd ⟨e, (hg.send_k _).strategy⟩ hc) (reach_answerWith nul (hg.send_k _) _ a h)
  | @Game.coin _ _ _ _ k, hg, σ, a, ⟨c, h⟩ => .coin c (reach_answerWith nul (hg.coin_k c) (σ c) a h)

/-- **A message at the one position of a batch** is the table's message. -/
theorem unB_batchFn_send (f : Fin 1 → Game A) {M : Type} (k : M → Game A) (m : M)
    (σ : Strategy (batch (List.ofFn (Function.update f 0 (k m))))) :
    unB (.send M k) (playSteps ((Ctx.batchFn f 0).send k m) σ) = ⟨m, unB (k m) σ⟩ :=
  (congrArg (unB (.send M k)) (eq_of_heq (playSteps_congr_send
    (Ctx.batchAt ((List.ofFn f).take (0 : Fin 1)) ((List.ofFn f).drop ((0 : Fin 1) + 1)))
    (fun g => congrArg batch (ofFn_update f 0 g).symm) k m σ σ HEq.rfl))).trans
  (congrArg (Sigma.mk m) (unB_eq (k m) σ _ (playSteps_lift_ofEq_heq _ _ σ)))

/-- **A path at the one position of a batch** is the table's path. -/
theorem unB_batchFn_lift (f : Fin 1 → Game A) :
    ∀ {Y Y' : Game A} (t : Steps Y Y') (σ : Strategy (batch (List.ofFn (Function.update f 0 Y')))),
      unB Y (playSteps ((Ctx.batchFn f 0).lift t) σ) = playSteps t (unB Y' σ)
  | _, _, .refl _, _ => rfl
  | _, _, .send m t, σ =>
      (congrArg (unB _) (playSteps_trans ((Ctx.batchFn f 0).send _ m) ((Ctx.batchFn f 0).lift t) σ)).trans
        ((unB_batchFn_send f _ m _).trans (congrArg (Sigma.mk m) (unB_batchFn_lift f t σ)))

/-- **A coin at the one position of a batch** is the table's coin. -/
theorem unB_batchFn_coin (f : Fin 1 → Game A) {C : Type} [Fintype C] [Nonempty C] (k : C → Game A)
    (ψ : Strategy (.coin C ((Ctx.batchFn f 0).coin k).next)) :
    unB (.coin C k) (playSteps ((Ctx.batchFn f 0).coin k).steps ψ) =
      fun x => unB (k x) (cast (congrArg Strategy (((Ctx.batchFn f 0).coin k).eq x)) (ψ x)) :=
  (congrArg (unB (.coin C k)) (eq_of_heq (playSteps_congr_coin
    (Ctx.batchAt ((List.ofFn f).take (0 : Fin 1)) ((List.ofFn f).drop ((0 : Fin 1) + 1)))
    (fun g => congrArg batch (ofFn_update f 0 g).symm) k ψ))).trans
  (funext fun x => unB_eq (k x) _ (ψ x) (cast_heq _ _).symm)

end One

/-! ## Rep 0's steps, read at the table -/

section Rep0

variable {P OutT Cp Op : Type} [Nonempty Op] (out : Fin 1 → Cp → P × P → Op → OutT) (caps : Fin 1 → Cp)

/-- The two reps' positions of one table. -/
def twoReps (g₀ g₁ : Game P) : Fin 1 → Fin 2 → Game P := fun _ r => if r = 0 then g₀ else g₁

theorem twoReps_eq (gs : Fin 1 → Fin 2 → Game P) : gs = twoReps (gs 0 0) (gs 0 1) := by
  funext j r
  obtain rfl := Subsingleton.elim j 0
  fin_cases r <;> rfl

/-- **A message of rep 0 against rep 1 sending**: the table chooses rep 0, then the message. -/
theorem unB_ctxAt_send₁ (g₀ : Game P) {M₂ : Type} (k₂ : M₂ → Game P) {M : Type} (k : M → Game P) (m : M)
    (ρ : Strategy (posOf out caps (upd (twoReps g₀ (.send M₂ k₂)) 0 0 (k m)))) :
    unB (tabG out caps (upd (twoReps g₀ (.send M₂ k₂)) 0 0 (.send M k)) 0)
        (playSteps ((ctxAt out caps (twoReps g₀ (.send M₂ k₂)) 0 0).send k m) ρ) =
      (⟨true, ⟨m, unB (tabG out caps (upd (twoReps g₀ (.send M₂ k₂)) 0 0 (k m)) 0) ρ⟩⟩ :
        Strategy ((interleave (.send M k) (.send M₂ k₂)).bind (KgJ out 0 (caps 0)))) :=
  (congrArg (unB _) (eq_of_heq (playSteps_congr_send
    (((Ctx.left (.send M₂ k₂)).comp (Ctx.bind (KgJ out 0 (caps 0)))).comp
      (Ctx.batchFn (tabG out caps (twoReps g₀ (.send M₂ k₂))) 0))
    (posOf_upd₀ out caps (twoReps g₀ (.send M₂ k₂)) 0) k m ρ ρ HEq.rfl))).trans
  (unB_batchFn_lift (tabG out caps (twoReps g₀ (.send M₂ k₂)))
    ((Ctx.bind (KgJ out 0 (caps 0))).lift ((Ctx.left (.send M₂ k₂)).send k m)) ρ)

/-- **A message of rep 0 against rep 1 at a coin**: the table chooses rep 0, then the message. -/
theorem unB_ctxAt_send₂ (g₀ : Game P) {C₂ : Type} [Fintype C₂] [Nonempty C₂] (k₂ : C₂ → Game P) {M : Type}
    (k : M → Game P) (m : M) (ρ : Strategy (posOf out caps (upd (twoReps g₀ (.coin C₂ k₂)) 0 0 (k m)))) :
    unB (tabG out caps (upd (twoReps g₀ (.coin C₂ k₂)) 0 0 (.send M k)) 0)
        (playSteps ((ctxAt out caps (twoReps g₀ (.coin C₂ k₂)) 0 0).send k m) ρ) =
      (⟨true, ⟨m, unB (tabG out caps (upd (twoReps g₀ (.coin C₂ k₂)) 0 0 (k m)) 0) ρ⟩⟩ :
        Strategy ((interleave (.send M k) (.coin C₂ k₂)).bind (KgJ out 0 (caps 0)))) :=
  (congrArg (unB _) (eq_of_heq (playSteps_congr_send
    (((Ctx.left (.coin C₂ k₂)).comp (Ctx.bind (KgJ out 0 (caps 0)))).comp
      (Ctx.batchFn (tabG out caps (twoReps g₀ (.coin C₂ k₂))) 0))
    (posOf_upd₀ out caps (twoReps g₀ (.coin C₂ k₂)) 0) k m ρ ρ HEq.rfl))).trans
  (unB_batchFn_lift (tabG out caps (twoReps g₀ (.coin C₂ k₂)))
    ((Ctx.bind (KgJ out 0 (caps 0))).lift ((Ctx.left (.coin C₂ k₂)).send k m)) ρ)

/-- **A coin of rep 0**, read at the table: rep 0's path to the coin in the table, then the table's coin. -/
theorem unB_ctxAt_coin (gs : Fin 1 → Fin 2 → Game P) {C : Type} [Fintype C] [Nonempty C] (k : C → Game P)
    (ψ : Strategy (.coin C ((ctxAt out caps gs 0 0).coin k).next)) :
    unB (tabG out caps (upd gs 0 0 (.coin C k)) 0) (playSteps ((ctxAt out caps gs 0 0).coin k).steps ψ) =
      playSteps (((Ctx.left (gs 0 1)).comp (Ctx.bind (KgJ out 0 (caps 0)))).coin k).steps
        (unB (.coin C (((Ctx.left (gs 0 1)).comp (Ctx.bind (KgJ out 0 (caps 0)))).coin k).next)
          (playSteps ((Ctx.batchFn (tabG out caps gs) 0).coin
            (((Ctx.left (gs 0 1)).comp (Ctx.bind (KgJ out 0 (caps 0)))).coin k).next).steps ψ)) :=
  (congrArg (unB _) (eq_of_heq (playSteps_congr_coin
    (((Ctx.left (gs 0 1)).comp (Ctx.bind (KgJ out 0 (caps 0)))).comp (Ctx.batchFn (tabG out caps gs) 0))
    (posOf_upd₀ out caps gs 0) k ψ))).trans
  ((congrArg (unB _) (playSteps_trans
    ((Ctx.batchFn (tabG out caps gs) 0).lift (((Ctx.left (gs 0 1)).comp (Ctx.bind (KgJ out 0 (caps 0)))).coin k).steps)
    ((Ctx.batchFn (tabG out caps gs) 0).coin
      (((Ctx.left (gs 0 1)).comp (Ctx.bind (KgJ out 0 (caps 0)))).coin k).next).steps ψ)).trans
  (unB_batchFn_lift (tabG out caps gs) (((Ctx.left (gs 0 1)).comp (Ctx.bind (KgJ out 0 (caps 0)))).coin k).steps
    (playSteps ((Ctx.batchFn (tabG out caps gs) 0).coin
      (((Ctx.left (gs 0 1)).comp (Ctx.bind (KgJ out 0 (caps 0)))).coin k).next).steps ψ)))

/-- **A coin of rep 0 against rep 1 sending**: the table chooses rep 0, then the coin. -/
theorem unB_ctxAt_coin₁ (g₀ : Game P) {M₂ : Type} (k₂ : M₂ → Game P) {C : Type} [Fintype C] [Nonempty C]
    (k : C → Game P) (ψ : Strategy (.coin C ((ctxAt out caps (twoReps g₀ (.send M₂ k₂)) 0 0).coin k).next)) :
    unB (tabG out caps (upd (twoReps g₀ (.send M₂ k₂)) 0 0 (.coin C k)) 0)
        (playSteps ((ctxAt out caps (twoReps g₀ (.send M₂ k₂)) 0 0).coin k).steps ψ) =
      (⟨true, fun x => unB (tabG out caps (upd (twoReps g₀ (.send M₂ k₂)) 0 0 (k x)) 0)
        (cast (congrArg Strategy (((ctxAt out caps (twoReps g₀ (.send M₂ k₂)) 0 0).coin k).eq x)) (ψ x))⟩ :
        Strategy ((interleave (.coin C k) (.send M₂ k₂)).bind (KgJ out 0 (caps 0)))) :=
  (unB_ctxAt_coin out caps (twoReps g₀ (.send M₂ k₂)) k ψ).trans
    (congrArg (Sigma.mk true) ((unB_batchFn_coin (tabG out caps (twoReps g₀ (.send M₂ k₂)))
      (fun c => interleave (k c) (.send M₂ k₂) |>.bind (KgJ out 0 (caps 0))) ψ).trans
      (funext fun _ => congrArg (unB _) (eq_of_heq ((cast_heq _ _).trans (cast_heq _ _).symm)))))

/-- **A coin of rep 0 against rep 1 at a coin**: the table chooses rep 0, then the coin. -/
theorem unB_ctxAt_coin₂ (g₀ : Game P) {C₂ : Type} [Fintype C₂] [Nonempty C₂] (k₂ : C₂ → Game P) {C : Type}
    [Fintype C] [Nonempty C] (k : C → Game P)
    (ψ : Strategy (.coin C ((ctxAt out caps (twoReps g₀ (.coin C₂ k₂)) 0 0).coin k).next)) :
    unB (tabG out caps (upd (twoReps g₀ (.coin C₂ k₂)) 0 0 (.coin C k)) 0)
        (playSteps ((ctxAt out caps (twoReps g₀ (.coin C₂ k₂)) 0 0).coin k).steps ψ) =
      (⟨true, fun x => unB (tabG out caps (upd (twoReps g₀ (.coin C₂ k₂)) 0 0 (k x)) 0)
        (cast (congrArg Strategy (((ctxAt out caps (twoReps g₀ (.coin C₂ k₂)) 0 0).coin k).eq x)) (ψ x))⟩ :
        Strategy ((interleave (.coin C k) (.coin C₂ k₂)).bind (KgJ out 0 (caps 0)))) :=
  (unB_ctxAt_coin out caps (twoReps g₀ (.coin C₂ k₂)) k ψ).trans
    (congrArg (Sigma.mk true) ((unB_batchFn_coin (tabG out caps (twoReps g₀ (.coin C₂ k₂)))
      (fun c => interleave (k c) (.coin C₂ k₂) |>.bind (KgJ out 0 (caps 0))) ψ).trans
      (funext fun _ => congrArg (unB _) (eq_of_heq ((cast_heq _ _).trans (cast_heq _ _).symm)))))

end Rep0

end FlockDeploy.KS
