import Proofs.Flock.Claims.KS.SlotLead
import Proofs.Flock.Claims.Bridge.Returns

/-!
# The simulated strategy leads with rep 0

The simulation of a one-table session (`simRJ`) follows the live prover round by round. When the live prover submits
its next `r₀` rounds on rep 0 and then only rep 1's, the simulated strategy, read at its one table (`unB`), is to lead
with rep 0 for `r₀` coins (`Lock.Lead`), except below a part of the run whose every reached outcome is a dead
branch's, answering the openings with `nul`.

* `playSteps`: a strategy along a path of the model's messages;
* `unB`: a table's strategy out of its one-position batch's, and how it reads a step of rep 0 (`unB_ctxAt_send₁`,
  `unB_ctxAt_coin₁`);
* `reach_mapDead`: what a simulated strategy reaches, by the simulation's returns and dead branches;
* `lead_lift`, `lead_coin`: a lead through rep 0's messages and coins, read at the table;
* `roundJ_none`, `roundJ_some`, `coinJ_coin`: the simulation's round, unfolded;
* `SchedL`: the live prover submits its next `r₀` rounds on rep 0, then only rep 1's;
* `lead_simRJ`: so the simulated strategy, read at the table, leads with rep 0 for `r₀` coins.
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Tables Claims.Bridge

local notation "𝔽" => FlockLevel3.GF128

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

theorem cast_mapDead_castR (f : (g : Game α) → Strategy g → Strategy g) {L : Game β} {g g' : Game α} (h : g = g')
    (d : Sim pL pg L g') (σ : Strategy L) :
    cast (congrArg Strategy h) ((Sim.mapDead f (Sim.castR h.symm d)).strategy σ) = (Sim.mapDead f d).strategy σ := by
  subst h
  rfl

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

/-! ## Leading with rep 0, step by step -/

section Leads

variable {P OutT Cp Op : Type} [Nonempty Op] (out : Fin 1 → Cp → P × P → Op → OutT) (caps : Fin 1 → Cp) (nul : Op)
  {Dead : OutT → Prop}

theorem lead_of_eq {α β γ : Type} {f : α × β → Game γ} {D : γ → Prop} {n : ℕ} {g₁ : Game α} {g₂ : Game β}
    {σ σ' : Strategy ((interleave g₁ g₂).bind f)} (h : σ = σ') (hL : Lock.Lead f D n g₁ g₂ σ) :
    Lock.Lead f D n g₁ g₂ σ' :=
  h ▸ hL

omit [Nonempty Op] in
theorem dead_of_outN (hnul : ∀ cap pend, Dead (out 0 cap pend nul)) {caps' : Fin 1 → Cp}
    {gs' : Fin 1 → Fin 2 → Game P} {c : OutT} (h : OutN nul (posOf out caps' gs') [c]) : Dead c := by
  obtain ⟨pend, hp⟩ := OutN.of_posOf out nul caps' gs' c h 0
  rw [show c = out 0 (caps' 0) pend nul from hp]
  exact hnul _ _

/-- **A dead node answered with `nul`**: the table reaches only `Dead` outputs. -/
theorem lead_answerWith (hnul : ∀ cap pend, Dead (out 0 cap pend nul)) {gs : Fin 1 → Fin 2 → Game P}
    (h : ∀ j r, AllInh (gs j r)) (σ : Strategy (posOf out caps gs)) {n : ℕ} :
    Lock.Lead (KgJ out 0 (caps 0)) Dead n (gs 0 0) (gs 0 1)
      (unB (tabG out caps gs 0) (answerWith nul (posOf out caps gs) σ)) :=
  .dead fun c hc => dead_of_outN out nul hnul
    (reach_answerWith nul (allInh_posOf out caps h) σ [c] (reach_unB _ _ c hc))

/-- **A dead node of a simulation**, its strategy answered with `nul`. -/
theorem lead_of_dead (hnul : ∀ cap pend, Dead (out 0 cap pend nul)) {β : Type} {pL : β → Prop}
    {pg : List OutT → Prop} {L : Game β} {gs : Fin 1 → Fin 2 → Game P} (h : ∀ j r, AllInh (gs j r))
    (d : Sim pL pg L (posOf out caps gs)) (hd : ∃ hp σ, d = Sim.dead hp σ) (s : Strategy L) {n : ℕ} :
    Lock.Lead (KgJ out 0 (caps 0)) Dead n (gs 0 0) (gs 0 1)
      (unB (tabG out caps gs 0) ((Sim.mapDead (answerWith nul) d).strategy s)) := by
  obtain ⟨hp, σ, rfl⟩ := hd
  exact lead_answerWith out caps nul hnul h σ

/-- **No accepted live outcome below**: everything the table reaches is a dead branch's, answered with `nul`. -/
theorem lead_noAcc (hnul : ∀ cap pend, Dead (out 0 cap pend nul)) {β : Type} {pL : β → Prop}
    {pg : List OutT → Prop} {L : Game β} {gs : Fin 1 → Fin 2 → Game P} (d : Sim pL pg L (posOf out caps gs))
    (hA : Sim.All (fun b _ => pL b) (DeadAt out) d) (s : Strategy L) (hno : ∀ b, Lock.Reach L s b → ¬ pL b)
    {n : ℕ} :
    Lock.Lead (KgJ out 0 (caps 0)) Dead n (gs 0 0) (gs 0 1)
      (unB (tabG out caps gs 0) ((Sim.mapDead (answerWith nul) d).strategy s)) :=
  .dead fun c hc => by
    rcases reach_mapDead (answerWith nul) d hA s [c] (reach_unB _ _ c hc) with
      ⟨b, hb, hp⟩ | ⟨_, σ, ⟨caps', gs', hinh, rfl⟩, hr⟩
    · exact absurd hp (hno b hb)
    · exact dead_of_outN out nul hnul (reach_answerWith nul (allInh_posOf out caps' hinh) σ [c] hr)

/-- **Rep 0's messages**, read at the table: the table chooses rep 0 at each, until rep 1 has ended. -/
theorem lead_lift (g₀ : Game P) : ∀ (g₂ : Game P) {g g' : Game P} (t : Steps g g')
    (ρ : Strategy (posOf out caps (upd (twoReps g₀ g₂) 0 0 g'))) {n : ℕ},
    Lock.Lead (KgJ out 0 (caps 0)) Dead n g' g₂ (unB (tabG out caps (upd (twoReps g₀ g₂) 0 0 g') 0) ρ) →
    Lock.Lead (KgJ out 0 (caps 0)) Dead n g g₂
      (unB (tabG out caps (upd (twoReps g₀ g₂) 0 0 g) 0) (playSteps ((ctxAt out caps (twoReps g₀ g₂) 0 0).lift t) ρ))
  | .ret _, _, _, _, _, _, _ => .last
  | _, _, _, .refl _, _, _, hL => hL
  | .send _ k₂, _, _, @Steps.send _ _ k _ m t, ρ, _, hL =>
      lead_of_eq ((congrArg (unB (tabG out caps (upd (twoReps g₀ (.send _ k₂)) 0 0 (.send _ k)) 0))
          (playSteps_trans ((ctxAt out caps (twoReps g₀ (.send _ k₂)) 0 0).send k m)
            ((ctxAt out caps (twoReps g₀ (.send _ k₂)) 0 0).lift t) ρ)).trans
          (unB_ctxAt_send₁ out caps g₀ k₂ k m (playSteps ((ctxAt out caps (twoReps g₀ (.send _ k₂)) 0 0).lift t) ρ))).symm
        (.sendSend m (lead_lift g₀ _ t ρ hL))
  | @Game.coin _ _ _ _ k₂, _, _, @Steps.send _ _ k _ m t, ρ, _, hL =>
      lead_of_eq ((congrArg (unB (tabG out caps (upd (twoReps g₀ (.coin _ k₂)) 0 0 (.send _ k)) 0))
          (playSteps_trans ((ctxAt out caps (twoReps g₀ (.coin _ k₂)) 0 0).send k m)
            ((ctxAt out caps (twoReps g₀ (.coin _ k₂)) 0 0).lift t) ρ)).trans
          (unB_ctxAt_send₂ out caps g₀ k₂ k m (playSteps ((ctxAt out caps (twoReps g₀ (.coin _ k₂)) 0 0).lift t) ρ))).symm
        (.sendCoin m (lead_lift g₀ _ t ρ hL))

/-- **A coin of rep 0**, read at the table: the table chooses rep 0, then the coin, until rep 1 has ended. -/
theorem lead_coin (g₀ : Game P) : ∀ (g₂ : Game P) {C : Type} [Fintype C] [Nonempty C] (k : C → Game P)
    (ψ : Strategy (.coin C ((ctxAt out caps (twoReps g₀ g₂) 0 0).coin k).next)) {n : ℕ},
    (∀ x, Lock.Lead (KgJ out 0 (caps 0)) Dead n (k x) g₂ (unB (tabG out caps (upd (twoReps g₀ g₂) 0 0 (k x)) 0)
      (cast (congrArg Strategy (((ctxAt out caps (twoReps g₀ g₂) 0 0).coin k).eq x)) (ψ x)))) →
    Lock.Lead (KgJ out 0 (caps 0)) Dead (n + 1) (.coin C k) g₂
      (unB (tabG out caps (upd (twoReps g₀ g₂) 0 0 (.coin C k)) 0)
        (playSteps ((ctxAt out caps (twoReps g₀ g₂) 0 0).coin k).steps ψ))
  | .ret _, _, _, _, _, _, _, _ => .last
  | .send _ k₂, _, _, _, k, ψ, _, hL => lead_of_eq (unB_ctxAt_coin₁ out caps g₀ k₂ k ψ).symm (.coinSend hL)
  | @Game.coin _ _ _ _ k₂, _, _, _, k, ψ, _, hL => lead_of_eq (unB_ctxAt_coin₂ out caps g₀ k₂ k ψ).symm (.coinCoin hL)

end Leads

/-! ## A round of the simulation -/

section Eqns

variable {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
  (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
  (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
  (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (caps : Fin J → Cp) (rootB : Array ByteArray)
  (pts : Fin 2 → Fin mPts → 𝔽) {n : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {j : Fin J} {r : Fin 2}
  {b : ByteArray} {k : ℕ} {gs : Fin J → Fin 2 → Game P} (h : ∀ j r, AllInh (gs j r))
  (next : (c : Fin k → 𝔽) → (gs' : Fin J → Fin 2 → Game P) → (∀ j r, AllInh (gs' j r)) →
    InvJ out ok pL opn caps (roundL dg rootB pts n rs j r b k c) gs'
      (restJ dec one fin (addRoundJ rs j r (roundOf dg b k c))) →
    Sim pL (AllOk ok dflt) (roundL dg rootB pts n rs j r b k c) (posOf out caps gs'))

theorem roundJ_none
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) gs (restJ dec one fin rs))
    (hf : follow (gs j r) (dec j (rs j r).length b) = none) :
    ∃ hp σ, roundJ out ok dflt pL opn dg dec one fin caps rootB pts n rs j r b k gs h hI next = Sim.dead hp σ := by
  unfold roundJ
  split
  · exact ⟨prob_eq_zero_of_out (dead_follow dg dec one fin hf hI), _, rfl⟩
  · rename_i hf'
    exact absurd (hf.symm.trans hf') (by simp)

theorem roundJ_some
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) gs (restJ dec one fin rs))
    {g' : Game P} (hf : follow (gs j r) (dec j (rs j r).length b) = some g') :
    roundJ out ok dflt pL opn dg dec one fin caps rootB pts n rs j r b k gs h hI next =
      Sim.ofSteps ((Steps.ofEq (posOf_upd_self out caps gs j r).symm).trans
          ((ctxAt out caps gs j r).lift (stepsOfFollow _ _ hf)))
        (coinJ out ok dflt pL opn dg dec one fin caps rootB pts n rs j r b k gs h next g'
          (follow_allInh (h j r) hf) (inv_follow dg dec one fin hf hI)) := by
  unfold roundJ
  split
  · rename_i hf'
    exact absurd (hf'.symm.trans hf) (by simp)
  · rename_i g'' hf'
    obtain rfl : g'' = g' := Option.some.inj (hf'.symm.trans hf)
    rfl

theorem coinJ_coin {Cm : Type} [Fintype Cm] [Nonempty Cm] (kk : Cm → Game P) (hg : AllInh (.coin Cm kk))
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) (upd gs j r (.coin Cm kk))
      (upd (restJ dec one fin rs) j r (midJ dec one fin rs j r))) :
    (∃ hp σ, coinJ out ok dflt pL opn dg dec one fin caps rootB pts n rs j r b k gs h next (.coin Cm kk) hg hI =
        Sim.dead hp σ) ∨
      ∃ (e : Cm ≃ (Fin k → 𝔽)) (hI' : ∀ x, InvJ out ok pL opn caps (roundL dg rootB pts n rs j r b k (e x))
          (upd gs j r (kk x)) (restJ dec one fin (addRoundJ rs j r (roundOf dg b k (e x))))),
        coinJ out ok dflt pL opn dg dec one fin caps rootB pts n rs j r b k gs h next (.coin Cm kk) hg hI =
          Sim.ofSteps ((ctxAt out caps gs j r).coin kk).steps (Sim.coin e fun x =>
            Sim.castR (((ctxAt out caps gs j r).coin kk).eq x).symm
              (next (e x) (upd gs j r (kk x)) (allInh_upd h j r (hg.coin_k x)) (hI' x))) := by
  simp only [coinJ]
  split_ifs
  · rename_i hs hC
    exact .inr ⟨_, inv_coin dg dec one fin _ (.inl ⟨hs, hC, rfl⟩) hI, rfl⟩
  · rename_i hs hC
    exact .inl ⟨prob_eq_zero_of_out (dead_mid dg dec one fin (fun _ _ _ _ hk => by
      cases hk; unfold coinTy; rw [ite_eq_left hs]; exact hC) hI), _, rfl⟩
  · rename_i hs hC
    exact .inr ⟨_, inv_coin dg dec one fin _ (.inr ⟨hs, hC, rfl⟩) hI, rfl⟩
  · rename_i hs hC
    exact .inl ⟨prob_eq_zero_of_out (dead_mid dg dec one fin (fun _ _ _ _ hk => by
      cases hk; unfold coinTy; rw [ite_eq_right hs]; exact hC) hI), _, rfl⟩

end Eqns

/-! ## The simulated strategy leads with rep 0 -/

section Sched

variable {P OutT Cp Op : Type} [Nonempty Op] {mPts : ℕ} (out : Fin 1 → Cp → P × P → Op → OutT)
  (ok : Fin 1 → OutT → Prop) (dflt : OutT) (pL : LiveOutJ 1 mPts → Prop) (opn : Array ByteArray → Fin 1 → Op)
  (dg : ByteArray → ByteArray) (dec : Fin 1 → ℕ → ByteArray → List Msg) (one : Fin 1 → ℕ → Bool)
  (fin : Fin 1 → Fin 2 → Array ByteArray → List Msg) (caps : Fin 1 → Cp) (rootB : Array ByteArray)
  (pts : Fin 2 → Fin mPts → 𝔽) (nul : Op) {Dead : OutT → Prop}

/-- **The live strategy's next `r₀` rounds are rep 0's, and the rest rep 1's**: it does not stop, nor run out of
rounds, before rep 0's `r₀` are in. -/
def SchedL : (n : ℕ) → (rs : Fin 1 → Fin 2 → List Flock.Round) → ℕ →
    Strategy ((liveRoundsJ dg 1 n rs).bind (KLJ rootB pts)) → Prop
  | 0, _, r₀, _ => r₀ = 0
  | n + 1, rs, r₀, σ => match σ with
    | ⟨none, _⟩ => r₀ = 0
    | ⟨some s, τ⟩ => s.rep = (if r₀ = 0 then 1 else 0) ∧
        ∀ c, SchedL n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)) (r₀ - 1) (τ c)

/-- Under a schedule with no rep 0 rounds left, rep 0's stream ends as it is. -/
theorem reps_of_sched0 : ∀ (n : ℕ) (rs : Fin 1 → Fin 2 → List Flock.Round)
    (σ : Strategy ((liveRoundsJ dg 1 n rs).bind (KLJ rootB pts))), SchedL dg rootB pts n rs 0 σ →
      ∀ o, Lock.Reach _ σ o → o.reps 0 0 = rs 0 0
  | 0, _, _, _, _, h => by cases h; rfl
  | _ + 1, _, ⟨none, _⟩, _, _, h => by cases h; rfl
  | n + 1, rs, ⟨some s, τ⟩, hS, o, h => by
      obtain ⟨c, h⟩ := h
      exact (reps_of_sched0 n _ (τ c) (hS.2 c) o h).trans
        (addRoundJ_ne rs _ fun e => absurd (e.2.trans (hS.1.trans (ite_eq_left rfl))) (by decide))

omit [Nonempty Op] in
/-- **No accepted live outcome when rep 0 has not ended and no rep 0 round is left**: its tape is over. -/
theorem noAcc_of_sched0 (hfin : ∀ proofs, fin 0 0 proofs = []) {n : ℕ} {rs : Fin 1 → Fin 2 → List Flock.Round}
    {gs : Fin 1 → Fin 2 → Game P}
    (hI : InvJ out ok pL opn caps ((liveRoundsJ dg 1 n rs).bind (KLJ rootB pts)) gs (restJ dec one fin rs))
    {Pr : Strategy ((liveRoundsJ dg 1 n rs).bind (KLJ rootB pts))} (hS : SchedL dg rootB pts n rs 0 Pr)
    (hg : ∀ a, gs 0 0 ≠ .ret a) : ∀ o, Lock.Reach _ Pr o → ¬ pL o := by
  intro o ho hp
  obtain ⟨pend, hpl, -⟩ := hI o (out_of_reach ho) hp
  have h0 := hpl 0 0
  have hr : restJ dec one fin rs 0 0 o = [] := by
    simp only [restJ, reps_of_sched0 dg rootB pts n rs Pr hS o ho, List.drop_length, hfin, tapeFrom, List.map_nil,
      List.append_nil]
  rw [hr] at h0
  exact hg _ (PlaysT.nil_inv h0)

/-- **No rep 0 round left**: rep 0 has ended, or no accepted live outcome is below. -/
theorem lead_zero (hnul : ∀ cap pend, Dead (out 0 cap pend nul)) (hfin : ∀ proofs, fin 0 0 proofs = []) (n : ℕ)
    (rs : Fin 1 → Fin 2 → List Flock.Round) (gs : Fin 1 → Fin 2 → Game P) (h : ∀ j r, AllInh (gs j r))
    (hI : InvJ out ok pL opn caps ((liveRoundsJ dg 1 n rs).bind (KLJ rootB pts)) gs (restJ dec one fin rs))
    (Pr : Strategy ((liveRoundsJ dg 1 n rs).bind (KLJ rootB pts))) (hS : SchedL dg rootB pts n rs 0 Pr)
    (hA : Sim.All (fun b _ => pL b) (DeadAt out)
      (simRJ out ok dflt pL opn dg dec one fin caps rootB pts n rs gs h hI)) :
    Lock.Lead (KgJ out 0 (caps 0)) Dead 0 (gs 0 0) (gs 0 1) (unB (tabG out caps gs 0)
      ((Sim.mapDead (answerWith nul) (simRJ out ok dflt pL opn dg dec one fin caps rootB pts n rs gs h hI)).strategy
        Pr)) := by
  obtain ⟨g₁, g₂, rfl⟩ : ∃ g₁ g₂, gs = twoReps g₁ g₂ := ⟨_, _, twoReps_eq gs⟩
  cases g₁ with
  | ret _ => exact .done
  | send _ _ =>
      exact lead_noAcc out caps nul hnul _ hA Pr
        (noAcc_of_sched0 out ok pL opn dg dec one fin caps rootB pts hfin hI hS fun _ e => by simp [twoReps] at e)
  | coin _ _ =>
      exact lead_noAcc out caps nul hnul _ hA Pr
        (noAcc_of_sched0 out ok pL opn dg dec one fin caps rootB pts hfin hI hS fun _ e => by simp [twoReps] at e)

/-- **A round's coins on rep 0**, read at the table. -/
theorem lead_coinJ (hnul : ∀ cap pend, Dead (out 0 cap pend nul)) {n : ℕ} {rs : Fin 1 → Fin 2 → List Flock.Round}
    {b : ByteArray} {k : ℕ} (g₁ g₂ : Game P) (h : ∀ j r, AllInh (twoReps g₁ g₂ j r))
    (next : (c : Fin k → 𝔽) → (gs' : Fin 1 → Fin 2 → Game P) → (∀ j r, AllInh (gs' j r)) →
      InvJ out ok pL opn caps (roundL dg rootB pts n rs 0 0 b k c) gs'
        (restJ dec one fin (addRoundJ rs 0 0 (roundOf dg b k c))) →
      Sim pL (AllOk ok dflt) (roundL dg rootB pts n rs 0 0 b k c) (posOf out caps gs'))
    (τ : Strategy (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs 0 0 b k))) {r' : ℕ}
    (hnext : ∀ c (g : Game P) h' hI', Sim.All (fun b _ => pL b) (DeadAt out) (next c (upd (twoReps g₁ g₂) 0 0 g) h' hI') →
      Lock.Lead (KgJ out 0 (caps 0)) Dead r' g g₂ (unB (tabG out caps (upd (twoReps g₁ g₂) 0 0 g) 0)
        ((Sim.mapDead (answerWith nul) (next c (upd (twoReps g₁ g₂) 0 0 g) h' hI')).strategy (τ c)))) :
    ∀ (g' : Game P) (hg : AllInh g')
      (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs 0 0 b k))
        (upd (twoReps g₁ g₂) 0 0 g') (upd (restJ dec one fin rs) 0 0 (midJ dec one fin rs 0 0))),
      Sim.All (fun b _ => pL b) (DeadAt out)
        (coinJ out ok dflt pL opn dg dec one fin caps rootB pts n rs 0 0 b k (twoReps g₁ g₂) h next g' hg hI) →
      Lock.Lead (KgJ out 0 (caps 0)) Dead (r' + 1) g' g₂ (unB (tabG out caps (upd (twoReps g₁ g₂) 0 0 g') 0)
        ((Sim.mapDead (answerWith nul)
          (coinJ out ok dflt pL opn dg dec one fin caps rootB pts n rs 0 0 b k (twoReps g₁ g₂) h next g' hg hI)).strategy
            τ))
  | .ret _, hg, hI, _ => by
      refine lead_of_dead out caps nul hnul (allInh_upd h 0 0 hg) _ ?_ τ
      exact ⟨prob_eq_zero_of_out (dead_mid dg dec one fin (fun _ _ _ _ hk => by cases hk) hI), _, rfl⟩
  | .send _ _, hg, hI, _ => by
      refine lead_of_dead out caps nul hnul (allInh_upd h 0 0 hg) _ ?_ τ
      exact ⟨prob_eq_zero_of_out (dead_mid dg dec one fin (fun _ _ _ _ hk => by cases hk) hI), _, rfl⟩
  | @Game.coin _ Cm _ _ kk, hg, hI, hA => by
      rcases coinJ_coin out ok dflt pL opn dg dec one fin caps rootB pts h next kk hg hI with hd | ⟨e, hI', he⟩
      · exact lead_of_dead out caps nul hnul (allInh_upd h 0 0 hg) _ hd τ
      · rw [he] at hA
        refine lead_of_eq (congrArg (fun d : Sim pL (AllOk ok dflt) (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs 0 0 b k))
            (posOf out caps (upd (twoReps g₁ g₂) 0 0 (.coin Cm kk))) =>
          unB (tabG out caps (upd (twoReps g₁ g₂) 0 0 (.coin Cm kk)) 0)
            ((Sim.mapDead (answerWith nul) d).strategy τ)) he).symm ?_
        refine lead_of_eq (congrArg (unB (tabG out caps (upd (twoReps g₁ g₂) 0 0 (.coin Cm kk)) 0))
          ((congrArg (fun d => Sim.strategy d τ) (mapDead_ofSteps _ _ _)).trans (strategy_ofSteps _ _ τ))).symm ?_
        exact lead_coin out caps g₁ g₂ kk _ fun x => lead_of_eq
          (congrArg (unB (tabG out caps (upd (twoReps g₁ g₂) 0 0 (kk x)) 0)) (cast_mapDead_castR (answerWith nul)
            (((ctxAt out caps (twoReps g₁ g₂) 0 0).coin kk).eq x)
            (next (e x) (upd (twoReps g₁ g₂) 0 0 (kk x)) (allInh_upd h 0 0 (hg.coin_k x)) (hI' x)) (τ (e x)))).symm
          (hnext (e x) (kk x) _ (hI' x) (all_castR_inv _ _ (all_ofSteps_inv _ _ hA x)))

/-- **A round on rep 0**, read at the table: its messages, then its coins. -/
theorem lead_round (hnul : ∀ cap pend, Dead (out 0 cap pend nul)) {n : ℕ} {rs : Fin 1 → Fin 2 → List Flock.Round}
    {b : ByteArray} {k : ℕ} (g₁ g₂ : Game P) (h : ∀ j r, AllInh (twoReps g₁ g₂ j r))
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs 0 0 b k)) (twoReps g₁ g₂)
      (restJ dec one fin rs))
    (next : (c : Fin k → 𝔽) → (gs' : Fin 1 → Fin 2 → Game P) → (∀ j r, AllInh (gs' j r)) →
      InvJ out ok pL opn caps (roundL dg rootB pts n rs 0 0 b k c) gs'
        (restJ dec one fin (addRoundJ rs 0 0 (roundOf dg b k c))) →
      Sim pL (AllOk ok dflt) (roundL dg rootB pts n rs 0 0 b k c) (posOf out caps gs'))
    (τ : Strategy (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs 0 0 b k))) {r' : ℕ}
    (hnext : ∀ c (g : Game P) h' hI', Sim.All (fun b _ => pL b) (DeadAt out) (next c (upd (twoReps g₁ g₂) 0 0 g) h' hI') →
      Lock.Lead (KgJ out 0 (caps 0)) Dead r' g g₂ (unB (tabG out caps (upd (twoReps g₁ g₂) 0 0 g) 0)
        ((Sim.mapDead (answerWith nul) (next c (upd (twoReps g₁ g₂) 0 0 g) h' hI')).strategy (τ c))))
    (hA : Sim.All (fun b _ => pL b) (DeadAt out)
      (roundJ out ok dflt pL opn dg dec one fin caps rootB pts n rs 0 0 b k (twoReps g₁ g₂) h hI next)) :
    Lock.Lead (KgJ out 0 (caps 0)) Dead (r' + 1) g₁ g₂ (unB (tabG out caps (twoReps g₁ g₂) 0)
      ((Sim.mapDead (answerWith nul)
        (roundJ out ok dflt pL opn dg dec one fin caps rootB pts n rs 0 0 b k (twoReps g₁ g₂) h hI next)).strategy τ)) := by
  rcases hf : follow (twoReps g₁ g₂ 0 0) (dec 0 (rs 0 0).length b) with _ | g'
  · exact lead_of_dead out caps nul hnul h _ (roundJ_none out ok dflt pL opn dg dec one fin caps rootB pts h next hI hf) τ
  · have he := roundJ_some out ok dflt pL opn dg dec one fin caps rootB pts h next hI hf
    rw [he] at hA
    refine lead_of_eq (congrArg (fun d : Sim pL (AllOk ok dflt) (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs 0 0 b k))
        (posOf out caps (twoReps g₁ g₂)) =>
      unB (tabG out caps (twoReps g₁ g₂) 0) ((Sim.mapDead (answerWith nul) d).strategy τ)) he).symm ?_
    refine lead_of_eq (congrArg (unB (tabG out caps (twoReps g₁ g₂) 0))
      ((congrArg (fun d => Sim.strategy d τ) (mapDead_ofSteps _ _ _)).trans
        ((strategy_ofSteps _ _ τ).trans ((playSteps_trans _ _ _).trans
          (eq_of_heq (playSteps_ofEq_heq _ _)))))).symm ?_
    exact lead_lift out caps g₁ g₂ (stepsOfFollow _ _ hf) _
      (lead_coinJ out ok dflt pL opn dg dec one fin caps rootB pts nul hnul g₁ g₂ h next τ hnext g' _ _
        (all_ofSteps_inv _ _ hA))

/-- **The simulated strategy leads with rep 0** for `r₀` coins, read at the one table, with every dead branch
answered with `nul`, when the live strategy's next `r₀` rounds are rep 0's and the rest rep 1's (`SchedL`), and rep
0 sends no final message. -/
theorem lead_simRJ (hnul : ∀ cap pend, Dead (out 0 cap pend nul)) (hfin : ∀ proofs, fin 0 0 proofs = []) :
    ∀ (n : ℕ) (rs : Fin 1 → Fin 2 → List Flock.Round) (gs : Fin 1 → Fin 2 → Game P) (h : ∀ j r, AllInh (gs j r))
      (hI : InvJ out ok pL opn caps ((liveRoundsJ dg 1 n rs).bind (KLJ rootB pts)) gs (restJ dec one fin rs))
      (Pr : Strategy ((liveRoundsJ dg 1 n rs).bind (KLJ rootB pts))) (r₀ : ℕ), SchedL dg rootB pts n rs r₀ Pr →
      Sim.All (fun b _ => pL b) (DeadAt out) (simRJ out ok dflt pL opn dg dec one fin caps rootB pts n rs gs h hI) →
      Lock.Lead (KgJ out 0 (caps 0)) Dead r₀ (gs 0 0) (gs 0 1) (unB (tabG out caps gs 0)
        ((Sim.mapDead (answerWith nul) (simRJ out ok dflt pL opn dg dec one fin caps rootB pts n rs gs h hI)).strategy
          Pr)) := by
  intro n
  induction n with
  | zero =>
      intro rs gs h hI Pr r₀ hS hA
      obtain rfl : r₀ = 0 := hS
      exact lead_zero out ok dflt pL opn dg dec one fin caps rootB pts nul hnul hfin 0 rs gs h hI Pr rfl hA
  | succ n ih =>
      intro rs gs h hI Pr r₀ hS hA
      rcases r₀ with _ | r'
      · exact lead_zero out ok dflt pL opn dg dec one fin caps rootB pts nul hnul hfin (n + 1) rs gs h hI Pr hS hA
      rcases Pr with ⟨_ | ⟨tab, rep, b, k⟩, τ⟩
      · exact absurd hS (Nat.succ_ne_zero r')
      obtain rfl : tab = 0 := Subsingleton.elim _ _
      obtain ⟨hrep, hS'⟩ := hS
      obtain rfl : rep = 0 := by simpa using hrep
      obtain ⟨g₁, g₂, rfl⟩ : ∃ g₁ g₂, gs = twoReps g₁ g₂ := ⟨_, _, twoReps_eq gs⟩
      exact lead_round out ok dflt pL opn dg dec one fin caps rootB pts nul hnul g₁ g₂ h
        (hI.mono fun _ h => .send (some (⟨0, 0, b, k⟩ : SubmitJ 1)) h)
        (fun _ gs' h' hI' => simRJ out ok dflt pL opn dg dec one fin caps rootB pts n _ gs' h' hI') τ
        (fun c _ h' hI' hA' => ih _ _ h' hI' (τ c) r' (hS' c) hA') (hA (some ⟨0, 0, b, k⟩))

end Sched

end FlockDeploy.KS
