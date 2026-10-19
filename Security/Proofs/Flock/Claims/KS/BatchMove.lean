import Proofs.Flock.Claims.KS.BatchSched

/-!
# A round at one table, read at the batch's schedule

A round of the simulator at table `t` is a path of the model's messages in that table's game followed by one of its
coins, at position `t` of the batch (`Ctx.batchAt`). Each schedule predicate of `BatchSched` at the strategy before the
round follows from the same predicate, one coin further on, at the strategies after the round's coin:

* `headS_batchAt_nil_lift`, `headS_batchAt_cons_lift`, `headS_batchAt_nil_coin`, `headS_batchAt_cons_coin`: a path or
  a coin at a position of a batch, read at the batch's first game and the rest;
* `waitR_right_lift`, `waitR_right_coin`: a waiting game keeps waiting while the rest moves;
* `firstS_left_lift`, `firstS_left_coin`, `followLP_left_lift`, `followLP_left_coin`: the moving game's own moves;
* `paused_round`: the paused games before position `t`;
* `front_round`, `sess_round0`, `sess_round1`: a table in front, and the session's table at rep 0 and at rep 1.
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Tables
open FlockSoundness.Game.Lock (WaitR FollowLP Paused Front Sess consP headS deadT FirstS predS)

theorem playSteps_lift_send {α A : Type} {C : Game A → Game α} (c : Ctx C) {M : Type} (k : M → Game A) (m : M)
    {g' : Game A} (T : Steps (k m) g') (ψ : Strategy (C g')) :
    playSteps (c.lift (Steps.send m T)) ψ = playSteps (c.send k m) (playSteps (c.lift T) ψ) :=
  playSteps_trans _ _ _

section Decomp

variable {ζ : Type}

theorem headS_batchAt_nil_lift (B : List (Game ζ)) {X X' : Game ζ} (T : Steps X X') (ρ : Strategy (batch (X' :: B))) :
    headS X B (playSteps ((Ctx.batchAt [] B).lift T) ρ) = playSteps ((Ctx.left (batch B)).lift T) (headS X' B ρ) :=
  (congrArg (Strategy.ofMap consP _) (playSteps_comp_lift (Ctx.left (batch B))
    (Ctx.bind fun a : ζ × List ζ => Game.ret (consP a)) T ρ)).trans (ofMap_bind_lift consP _ ρ)

theorem headS_batchAt_cons_lift (h : Game ζ) (A B : List (Game ζ)) {X X' : Game ζ} (T : Steps X X')
    (ρ : Strategy (batch (h :: (A ++ X' :: B)))) :
    headS h (A ++ X :: B) (playSteps ((Ctx.batchAt (h :: A) B).lift T) ρ) =
      playSteps ((Ctx.right h).lift ((Ctx.batchAt A B).lift T)) (headS h (A ++ X' :: B) ρ) :=
  (congrArg (Strategy.ofMap consP _) (playSteps_comp_lift ((Ctx.batchAt A B).comp (Ctx.right h))
    (Ctx.bind fun a : ζ × List ζ => Game.ret (consP a)) T ρ)).trans
    ((ofMap_bind_lift consP _ ρ).trans (playSteps_comp_lift (Ctx.batchAt A B) (Ctx.right h) T _))

theorem headS_batchAt_nil_coin (B : List (Game ζ)) {C : Type} [Fintype C] [Nonempty C] (N : C → Game ζ)
    (ψ : Strategy (.coin C ((Ctx.batchAt [] B).coin N).next)) :
    headS (.coin C N) B (playSteps ((Ctx.batchAt [] B).coin N).steps ψ) =
      playSteps ((Ctx.left (batch B)).coin N).steps
        (Strategy.ofMap consP (.coin C ((Ctx.left (batch B)).coin N).next) ψ) :=
  (congrArg (Strategy.ofMap consP _) (playSteps_trans _ (Steps.refl _) ψ)).trans
    (ofMap_bind_lift consP ((Ctx.left (batch B)).coin N).steps ψ)

theorem headS_batchAt_cons_coin (h : Game ζ) (A B : List (Game ζ)) {C : Type} [Fintype C] [Nonempty C]
    (N : C → Game ζ) (ψ : Strategy (.coin C ((Ctx.batchAt (h :: A) B).coin N).next)) :
    headS h (A ++ .coin C N :: B) (playSteps ((Ctx.batchAt (h :: A) B).coin N).steps ψ) =
      playSteps ((Ctx.right h).lift ((Ctx.batchAt A B).coin N).steps)
        (playSteps ((Ctx.right h).coin ((Ctx.batchAt A B).coin N).next).steps
          (Strategy.ofMap consP (.coin C ((Ctx.right h).coin ((Ctx.batchAt A B).coin N).next).next) ψ)) :=
  (congrArg (Strategy.ofMap consP _) (playSteps_trans _ (Steps.refl _) ψ)).trans
    ((ofMap_bind_lift consP (((Ctx.batchAt A B).comp (Ctx.right h)).coin N).steps ψ).trans
      (playSteps_trans _ _ _))

/-- The pairing read through an equality of the paired game. -/
theorem ofMap_cast_consP {X : Game ζ} {G : Game (List ζ)} {G₁ : Game (ζ × List ζ)} (e₁ : G₁ = interleave X G)
    (e₃ : G₁.map consP = (interleave X G).map consP) (ψ : Strategy (G₁.map consP)) :
    Strategy.ofMap consP (interleave X G) (cast (congrArg Strategy e₃) ψ) =
      cast (congrArg Strategy e₁) (Strategy.ofMap consP G₁ ψ) := by
  subst e₁
  rfl

theorem headS_nil_coin_cast (B : List (Game ζ)) {C : Type} [Fintype C] [Nonempty C] (N : C → Game ζ)
    (ψ : Strategy (.coin C ((Ctx.batchAt [] B).coin N).next)) (x : C) :
    headS (N x) B (cast (congrArg Strategy (((Ctx.batchAt [] B).coin N).eq x)) (ψ x)) =
      cast (congrArg Strategy (((Ctx.left (batch B)).coin N).eq x))
        (Strategy.ofMap consP (((Ctx.left (batch B)).coin N).next x) (ψ x)) :=
  ofMap_cast_consP (((Ctx.left (batch B)).coin N).eq x) (((Ctx.batchAt [] B).coin N).eq x) (ψ x)

/-- A predicate of the rest's strategy read through equalities of the rest. -/
theorem tr_head {h : Game ζ} (P : (G : Game (List ζ)) → Strategy (interleave h G) → Prop)
    {G₁ : Game (ζ × List ζ)} {G₂ G₃ : Game (List ζ)} (e₁ : G₁ = interleave h G₂) (e₂ : G₂ = G₃)
    (e₃ : G₁.map consP = (interleave h G₃).map consP) (ψ : Strategy (G₁.map consP))
    (hP : P G₃ (Strategy.ofMap consP (interleave h G₃) (cast (congrArg Strategy e₃) ψ))) :
    P G₂ (cast (congrArg Strategy e₁) (Strategy.ofMap consP G₁ ψ)) := by
  subst e₂ e₁
  exact hP

theorem tr_projR {h : Game ζ} (D' : List ζ → Prop) {G₃ : Game (List ζ)} (Q : Strategy G₃ → Prop)
    {G₁ : Game (ζ × List ζ)} {G₂ : Game (List ζ)} (e₁ : G₁ = interleave h G₂) (e₂ : G₂ = G₃)
    (e₃ : G₁.map consP = (interleave h G₃).map consP) (ψ : Strategy (G₁.map consP))
    (hQ : Q (Lock.projR D' h G₃ (Strategy.ofMap consP (interleave h G₃) (cast (congrArg Strategy e₃) ψ)))) :
    Q (cast (congrArg Strategy e₂) (Lock.projR D' h G₂ (cast (congrArg Strategy e₁) (Strategy.ofMap consP G₁ ψ)))) := by
  subst e₂ e₁
  exact hQ

end Decomp

section Wait

variable {γ β : Type} {Dead : γ × β → Prop}

theorem projR_right_coin (D' : β → Prop) : ∀ (h : Game γ) {C : Type} [Fintype C] [Nonempty C] (k₂ : C → Game β)
    (ψ : Strategy (.coin C ((Ctx.right h).coin k₂).next)),
    Lock.projR D' h (.coin C k₂) (playSteps ((Ctx.right h).coin k₂).steps ψ) =
      fun c => Lock.projR D' h (k₂ c) (cast (congrArg Strategy (((Ctx.right h).coin k₂).eq c)) (ψ c))
  | .ret _, _, _, _, _, _ => rfl
  | .send _ _, _, _, _, _, _ => rfl
  | @Game.coin _ _ _ _ _, _, _, _, _, _ => rfl

theorem reach_right_coin : ∀ (h : Game γ) {C : Type} [Fintype C] [Nonempty C] (k₂ : C → Game β)
    (ψ : Strategy (.coin C ((Ctx.right h).coin k₂).next)) {x : γ × β},
    Lock.Reach _ (playSteps ((Ctx.right h).coin k₂).steps ψ) x →
      ∃ c, Lock.Reach _ (cast (congrArg Strategy (((Ctx.right h).coin k₂).eq c)) (ψ c)) x
  | .ret _, _, _, _, _, _, _, hx => hx
  | .send _ _, _, _, _, _, _, _, hx => hx
  | @Game.coin _ _ _ _ _, _, _, _, _, _, _, hx => hx

theorem waitR_right_lift_send {M : Type} (k : M → Game γ) {b : Option ℕ} :
    ∀ {g g' : Game β} (t : Steps g g') (σ : Strategy (interleave (.send M k) g')),
      WaitR Dead b (.send M k) g' σ → WaitR Dead b (.send M k) g (playSteps ((Ctx.right (.send M k)).lift t) σ)
  | _, _, .refl _, _, hW => hW
  | _, _, .send m t, σ, hW => by
      rw [playSteps_lift_send]
      exact .isend m (waitR_right_lift_send k t σ hW)

/-- **A waiting game keeps waiting along the rest's messages.** -/
theorem waitR_right_lift {b : Option ℕ} {h : Game γ} {g g' : Game β} (t : Steps g g')
    (σ : Strategy (interleave h g')) (hW : WaitR Dead b h g' σ) :
    WaitR Dead b h g (playSteps ((Ctx.right h).lift t) σ) := by
  cases hW with
  | dead hσ => exact .dead fun x hx => hσ x ((reach_playSteps _ σ).1 hx)
  | spent => exact waitR_right_lift_send _ t _ .spent
  | icoin hw => exact waitR_right_lift_send _ t _ (.icoin hw)
  | isend m hw => exact waitR_right_lift_send _ t _ (.isend m hw)

/-- A game that waits other than at a send is dead. -/
theorem waitR_reach_dead {b : Option ℕ} {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')}
    (hW : WaitR Dead b h g' σ) (hh : ∀ (M : Type) (k : M → Game γ), h ≠ .send M k) :
    ∀ x, Lock.Reach _ σ x → Dead x := by
  cases hW with
  | dead hσ => exact hσ
  | spent => exact absurd rfl (hh _ _)
  | icoin _ => exact absurd rfl (hh _ _)
  | isend _ _ => exact absurd rfl (hh _ _)

/-- **A waiting game keeps waiting at the rest's coin**, one coin off its budget. -/
theorem waitR_right_coin {b : Option ℕ} (h : Game γ) {C : Type} [Fintype C] [Nonempty C] (k₂ : C → Game β)
    (ψ : Strategy (.coin C ((Ctx.right h).coin k₂).next))
    (hW : ∀ c, WaitR Dead (predS b) h (k₂ c) (cast (congrArg Strategy (((Ctx.right h).coin k₂).eq c)) (ψ c))) :
    WaitR Dead b h (.coin C k₂) (playSteps ((Ctx.right h).coin k₂).steps ψ) := by
  cases h with
  | send M k => exact .icoin hW
  | ret a =>
      refine .dead fun x hx => ?_
      obtain ⟨c, hc⟩ := reach_right_coin _ k₂ ψ hx
      exact waitR_reach_dead (hW c) (fun _ _ e => by cases e) x hc
  | coin C' k =>
      refine .dead fun x hx => ?_
      obtain ⟨c, hc⟩ := reach_right_coin _ k₂ ψ hx
      exact waitR_reach_dead (hW c) (fun _ _ e => by cases e) x hc

end Wait

section Own

variable {γ β : Type} {Dead : γ × β → Prop}

theorem firstS_left_lift {Q : (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Prop} {n : ℕ} :
    ∀ (g' : Game β) {X X' : Game γ} (T : Steps X X') (ψ : Strategy (interleave X' g')),
      FirstS Dead Q n X' g' ψ → FirstS Dead Q n X g' (playSteps ((Ctx.left g').lift T) ψ)
  | _, _, _, .refl _, _, hS => hS
  | .ret _, _, _, .send _ _, _, _ => .last
  | .send _ _, _, _, .send m T, ψ, hS => by
      rw [playSteps_lift_send]
      exact .sendSend m (firstS_left_lift _ T ψ hS)
  | @Game.coin _ _ _ _ _, _, _, .send m T, ψ, hS => by
      rw [playSteps_lift_send]
      exact .sendCoin m (firstS_left_lift _ T ψ hS)

theorem firstS_left_coin {Q : (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Prop} {n : ℕ} :
    ∀ (g' : Game β) {C : Type} [Fintype C] [Nonempty C] (N : C → Game γ)
      (ψ : Strategy (.coin C ((Ctx.left g').coin N).next)),
      (∀ c, FirstS Dead Q n (N c) g' (cast (congrArg Strategy (((Ctx.left g').coin N).eq c)) (ψ c))) →
        FirstS Dead Q (n + 1) (.coin C N) g' (playSteps ((Ctx.left g').coin N).steps ψ)
  | .ret _, _, _, _, _, _, _ => .last
  | .send _ _, _, _, _, _, _, hS => .coinSend hS
  | @Game.coin _ _ _ _ _, _, _, _, _, _, hS => .coinCoin hS

theorem followLP_left_lift {b : Option ℕ} :
    ∀ (g' : Game β) {X X' : Game γ} (T : Steps X X') (ψ : Strategy (interleave X' g')),
      FollowLP Dead b X' g' ψ → FollowLP Dead b X g' (playSteps ((Ctx.left g').lift T) ψ)
  | _, _, _, .refl _, _, hF => hF
  | .ret _, _, _, .send _ _, _, _ => .hret
  | .send _ _, _, _, .send m T, ψ, hF => by
      rw [playSteps_lift_send]
      exact .gsendS m (followLP_left_lift _ T ψ hF)
  | @Game.coin _ _ _ _ _, _, _, .send m T, ψ, hF => by
      rw [playSteps_lift_send]
      exact .gsendC m (followLP_left_lift _ T ψ hF)

theorem followLP_left_coin {b : Option ℕ} :
    ∀ (g' : Game β) {C : Type} [Fintype C] [Nonempty C] (N : C → Game γ)
      (ψ : Strategy (.coin C ((Ctx.left g').coin N).next)),
      (∀ c, FollowLP Dead (predS b) (N c) g' (cast (congrArg Strategy (((Ctx.left g').coin N).eq c)) (ψ c))) →
        FollowLP Dead b (.coin C N) g' (playSteps ((Ctx.left g').coin N).steps ψ)
  | .ret _, _, _, _, _, _, _ => .hret
  | .send _ _, _, _, _, _, _, hF => .gcoinS hF
  | @Game.coin _ _ _ _ _, _, _, _, _, _, hF => .gcoinC hF

end Own

section Round

variable {ζ : Type}

/-- **The paused games before position `A.length` keep waiting through a round there**: a path `T` of the table's
messages to its coin `N`, the round's coin one off the budget. -/
theorem paused_round {b : Option ℕ} {Inner : (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop}
    {C : Type} [Fintype C] [Nonempty C] {X : Game ζ} {N : C → Game ζ} (T : Steps X (.coin C N)) (B : List (Game ζ))
    (Inner' : C → (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop)
    (hInner : ∀ D (ψ : Strategy (.coin C ((Ctx.batchAt [] B).coin N).next)),
      (∀ c, Inner' c D (N c :: B) (cast (congrArg Strategy (((Ctx.batchAt [] B).coin N).eq c)) (ψ c))) →
        Inner D (X :: B) (playSteps ((Ctx.batchAt [] B).lift T) (playSteps ((Ctx.batchAt [] B).coin N).steps ψ))) :
    ∀ (A : List (Game ζ)) {D : List ζ → Prop} (ψ : Strategy (.coin C ((Ctx.batchAt A B).coin N).next)),
      (∀ c, Paused (predS b) (Inner' c) A.length D (A ++ N c :: B)
        (cast (congrArg Strategy (((Ctx.batchAt A B).coin N).eq c)) (ψ c))) →
      Paused b Inner A.length D (A ++ X :: B)
        (playSteps ((Ctx.batchAt A B).lift T) (playSteps ((Ctx.batchAt A B).coin N).steps ψ))
  | [], D, ψ, hP => hInner D ψ hP
  | h :: A, D, ψ, hP => by
      have e₁ := headS_batchAt_cons_lift h A B T (playSteps ((Ctx.batchAt (h :: A) B).coin N).steps ψ)
      have e₂ := headS_batchAt_cons_coin h A B N ψ
      refine ⟨?_, ?_⟩
      · show WaitR (fun x => D (consP x)) b h (batch (A ++ X :: B)) (headS h (A ++ X :: B) _)
        rw [e₁, e₂]
        refine waitR_right_lift _ _ (waitR_right_lift _ _ (waitR_right_coin h _ _ fun c => ?_))
        exact tr_head (fun G σ => WaitR (fun x => D (consP x)) (predS b) h G σ)
          (((Ctx.right h).coin ((Ctx.batchAt A B).coin N).next).eq c) (((Ctx.batchAt A B).coin N).eq c)
          (((Ctx.batchAt (h :: A) B).coin N).eq c) (ψ c) (hP c).1
      · show Paused b Inner A.length (deadT D) (A ++ X :: B)
          (Lock.projR (deadT D) h (batch (A ++ X :: B)) (headS h (A ++ X :: B) _))
        rw [e₁, e₂, projR_right_lift, projR_right_lift, projR_right_coin]
        refine paused_round T B Inner' hInner A _ fun c => ?_
        exact tr_projR (deadT D) (Paused (predS b) (Inner' c) A.length (deadT D) (A ++ N c :: B))
          (((Ctx.right h).coin ((Ctx.batchAt A B).coin N).next).eq c) (((Ctx.batchAt A B).coin N).eq c)
          (((Ctx.batchAt (h :: A) B).coin N).eq c) (ψ c) (hP c).2

/-- **A table in front advances through a round of its own**, one coin off its count. -/
theorem front_round (k : ℕ) (S : (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop) (c : ℕ)
    (ns : List ℕ) {C : Type} [Fintype C] [Nonempty C] {X : Game ζ} {N : C → Game ζ} (T : Steps X (.coin C N))
    (B : List (Game ζ)) (D : List ζ → Prop) (ψ : Strategy (.coin C ((Ctx.batchAt [] B).coin N).next))
    (hF : ∀ x, Front k S (c :: ns) D (N x :: B) (cast (congrArg Strategy (((Ctx.batchAt [] B).coin N).eq x)) (ψ x))) :
    Front k S ((c + 1) :: ns) D (X :: B)
      (playSteps ((Ctx.batchAt [] B).lift T) (playSteps ((Ctx.batchAt [] B).coin N).steps ψ)) := by
  show FirstS _ _ (c + 1) X (batch B) (headS X B _)
  rw [headS_batchAt_nil_lift, headS_batchAt_nil_coin]
  refine firstS_left_lift _ T _ (firstS_left_coin _ N _ fun x => ?_)
  have h := hF x
  rw [show Front k S (c :: ns) D (N x :: B) _ =
    FirstS _ _ c (N x) (batch B) (headS (N x) B _) from rfl, headS_nil_coin_cast] at h
  exact h

theorem followLP_round {b : Option ℕ} {C : Type} [Fintype C] [Nonempty C] {X : Game ζ} {N : C → Game ζ}
    (T : Steps X (.coin C N)) (B : List (Game ζ)) (D : List ζ → Prop)
    (ψ : Strategy (.coin C ((Ctx.batchAt [] B).coin N).next))
    (hF : ∀ x, FollowLP (fun x => D (consP x)) (predS b) (N x) (batch B)
      (headS (N x) B (cast (congrArg Strategy (((Ctx.batchAt [] B).coin N).eq x)) (ψ x)))) :
    FollowLP (fun x => D (consP x)) b X (batch B)
      (headS X B (playSteps ((Ctx.batchAt [] B).lift T) (playSteps ((Ctx.batchAt [] B).coin N).steps ψ))) := by
  rw [headS_batchAt_nil_lift, headS_batchAt_nil_coin]
  refine followLP_left_lift _ T _ (followLP_left_coin _ N _ fun x => ?_)
  have h := hF x
  rw [headS_nil_coin_cast] at h
  exact h

/-- **The session's table moves rep 0 through a round**, one coin off its budget and off rep 0's lead. -/
theorem sess_round0 {α₁ β₁ : Type} {f : α₁ × β₁ → Game ζ} {b : Option ℕ} {nL : ℕ} {g₁ : Game α₁} (g₂ : Game β₁)
    {C : Type} [Fintype C] [Nonempty C] {kk : C → Game α₁} (s : Steps g₁ (.coin C kk)) (B : List (Game ζ))
    (D : List ζ → Prop)
    (ψ : Strategy (.coin C ((Ctx.batchAt [] B).coin (((Ctx.left g₂).comp (Ctx.bind f)).coin kk).next).next))
    (hS : ∀ x, Sess (predS b) nL f (kk x) g₂ D ((((Ctx.left g₂).comp (Ctx.bind f)).coin kk).next x :: B)
      (cast (congrArg Strategy (((Ctx.batchAt [] B).coin (((Ctx.left g₂).comp (Ctx.bind f)).coin kk).next).eq x))
        (ψ x))) :
    Sess b (nL + 1) f g₁ g₂ D ((interleave g₁ g₂).bind f :: B)
      (playSteps ((Ctx.batchAt [] B).lift ((((Ctx.left g₂).comp (Ctx.bind f)).lift s).trans
          (((Ctx.left g₂).comp (Ctx.bind f)).coin kk).steps))
        (playSteps ((Ctx.batchAt [] B).coin (((Ctx.left g₂).comp (Ctx.bind f)).coin kk).next).steps ψ)) := by
  refine ⟨followLP_round _ B D ψ fun x => (hS x).1, fun e => ?_⟩
  rw [cast_eq, headS_batchAt_nil_lift, headS_batchAt_nil_coin, projL_left_lift, playSteps_trans, playSteps_comp_lift,
    projL_left_coin]
  refine lead_left_lift g₂ s _ (lead_left_coin g₂ kk _ fun x => ?_)
  have h := (hS x).2 ((((Ctx.left g₂).comp (Ctx.bind f)).coin kk).eq x)
  rw [headS_nil_coin_cast] at h
  exact h

/-- **The session's table moves rep 1 through a round** once rep 0 has ended, one coin off its budget. -/
theorem sess_round1 {α₁ β₁ : Type} {f : α₁ × β₁ → Game ζ} {b : Option ℕ} {a : α₁} {g₂ : Game β₁}
    {C : Type} [Fintype C] [Nonempty C] {X : Game ζ} {N : C → Game ζ} (T : Steps X (.coin C N))
    (B : List (Game ζ)) (D : List ζ → Prop) (ψ : Strategy (.coin C ((Ctx.batchAt [] B).coin N).next))
    (hS : ∀ x, Sess (predS b) 0 f (.ret a) g₂ D (N x :: B)
      (cast (congrArg Strategy (((Ctx.batchAt [] B).coin N).eq x)) (ψ x))) :
    Sess b 0 f (.ret a) g₂ D (X :: B)
      (playSteps ((Ctx.batchAt [] B).lift T) (playSteps ((Ctx.batchAt [] B).coin N).steps ψ)) :=
  ⟨followLP_round T B D ψ fun x => (hS x).1, fun _ => .done⟩

end Round

end FlockDeploy.KS
