import Proofs.Flock.Soundness.Discharge.Tables.Sim

/-!
# A simulation's dead branches, answered with a fixed message

A simulation (`Refine.Sim`) plays, past a live branch on which no live run is accepted, any model strategy (`Sim.dead`).
`Sim.mapDead f` replaces each such strategy `σ` by `f σ`, and the transfer holds of the result (`Sim.prob_le_mapDead`),
since a dead branch's live probability is `0` whatever the model plays there. `answerWith nul` is the transformer that
answers every message of `nul`'s type with `nul` and plays `σ` elsewhere: at the salted session, empty openings, so a
dead branch opens nothing.

`Sim.All R D d` says what a simulation's nodes satisfy: `R` at each return, of the live and the model outcome, and `D`
at each dead branch, of its model game and strategy. When every return's model outcome in an event `Q` has its live
outcome in `P`, and no dead branch reaches `Q`, the model's `Q` against the simulated strategy is no likelier than the
live `P` (`Sim.prob_le_of_all`). A dead branch at the tables' positions answered with `nul` reaches only outcomes whose
every table's output is at the openings `nul` (`OutN`, `OutN.of_posOf`, `prob_posOf_answerWith`).
-/

namespace Claims.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Tables

open Classical in
/-- **Answer every message of `Op` with `nul`**: at a node sending `Op`, `nul` when the game goes on from it, and `σ`'s
message elsewhere. -/
noncomputable def answerWith {α Op : Type} (nul : Op) : (g : Game α) → Strategy g → Strategy g
  | .ret _, σ => σ
  | .send M k, σ =>
    (if h : ∃ e : Op = M, Nonempty (Strategy (k (cast e nul))) then
      ⟨cast h.fst nul, answerWith nul (k (cast h.fst nul)) (Classical.choice h.snd)⟩
    else ⟨σ.1, answerWith nul (k σ.1) σ.2⟩ : (m : M) × Strategy (k m))
  | @Game.coin _ _ _ _ k, σ => fun c => answerWith nul (k c) (σ c)

/-- **The dead branches' strategies replaced** by `f`'s. -/
def Sim.mapDead {β α : Type} {pL : β → Prop} {pg : α → Prop} (f : (g : Game α) → Strategy g → Strategy g) :
    {L : Game β} → {g : Game α} → Sim pL pg L g → Sim pL pg L g
  | _, _, .ret h => .ret h
  | _, _, .model m d => .model m (Sim.mapDead f d)
  | _, _, .live k => .live fun m => Sim.mapDead f (k m)
  | _, _, .coin e k => .coin e fun c => Sim.mapDead f (k c)
  | _, _, .dead h σ => .dead h (f _ σ)

/-- **The transfer with the dead branches replaced**: the live verdict is no likelier than the model's against the
simulated strategy. -/
theorem Sim.prob_le_mapDead {β α : Type} {pL : β → Prop} {pg : α → Prop} (f : (g : Game α) → Strategy g → Strategy g)
    {L : Game β} {g : Game α} (d : Sim pL pg L g) (s : Strategy L) :
    prob pL L s ≤ prob pg g ((Sim.mapDead f d).strategy s) :=
  Sim.prob_le _ s

/-! ## What a simulation's nodes say -/

/-- **Every node of a simulation**: `R` of each return's live and model outcomes, `D` of each dead branch's model
game and strategy. -/
def Sim.All {β α : Type} {pL : β → Prop} {pg : α → Prop} (R : β → α → Prop) (D : (g : Game α) → Strategy g → Prop) :
    {L : Game β} → {g : Game α} → Sim pL pg L g → Prop
  | _, _, @Sim.ret _ _ _ _ b a _ => R b a
  | _, _, .model _ d => Sim.All R D d
  | _, _, .live k => ∀ m, Sim.All R D (k m)
  | _, _, .coin _ k => ∀ c, Sim.All R D (k c)
  | _, g, .dead _ σ => D g σ

theorem Sim.All.mono {β α : Type} {pL : β → Prop} {pg : α → Prop} {R R' : β → α → Prop}
    {D D' : (g : Game α) → Strategy g → Prop} (hR : ∀ b a, R b a → R' b a) (hD : ∀ g σ, D g σ → D' g σ) :
    ∀ {L : Game β} {g : Game α} (d : Sim pL pg L g), Sim.All R D d → Sim.All R' D' d
  | _, _, @Sim.ret _ _ _ _ b a _, h => hR b a h
  | _, _, .model _ d, h => Sim.All.mono hR hD d h
  | _, _, .live k, h => fun m => Sim.All.mono hR hD (k m) (h m)
  | _, _, .coin _ k, h => fun c => Sim.All.mono hR hD (k c) (h c)
  | _, g, .dead _ σ, h => hD g σ h

/-- The dead branches replaced: the returns as they were, each dead branch's new strategy as `f` makes it. -/
theorem Sim.all_mapDead {β α : Type} {pL : β → Prop} {pg : α → Prop} (f : (g : Game α) → Strategy g → Strategy g)
    {R : β → α → Prop} {D D' : (g : Game α) → Strategy g → Prop} (hD : ∀ g σ, D g σ → D' g (f g σ)) :
    ∀ {L : Game β} {g : Game α} (d : Sim pL pg L g), Sim.All R D d → Sim.All R D' (Sim.mapDead f d)
  | _, _, .ret _, h => h
  | _, _, .model _ d, h => Sim.all_mapDead f hD d h
  | _, _, .live k, h => fun m => Sim.all_mapDead f hD (k m) (h m)
  | _, _, .coin _ k, h => fun c => Sim.all_mapDead f hD (k c) (h c)
  | _, g, .dead _ σ, h => hD g σ h

theorem Sim.all_ofSteps {β α : Type} {pL : β → Prop} {pg : α → Prop} {R : β → α → Prop}
    {D : (g : Game α) → Strategy g → Prop} {L : Game β} :
    ∀ {g g' : Game α} (s : Steps g g') (d : Sim pL pg L g'), Sim.All R D d → Sim.All R D (Sim.ofSteps s d)
  | _, _, .refl _, _, h => h
  | _, _, .send _ s, d, h => Sim.all_ofSteps s d h

theorem Sim.all_castR {β α : Type} {pL : β → Prop} {pg : α → Prop} {R : β → α → Prop}
    {D : (g : Game α) → Strategy g → Prop} {L : Game β} {g g' : Game α} (h : g = g') (d : Sim pL pg L g)
    (hd : Sim.All R D d) : Sim.All R D (Sim.castR h d) := by
  subst h
  exact hd

/-- **The transfer of a model event to a live one.** When every return's model outcome satisfying `Q` has its live
outcome satisfying `P`, and no dead branch's strategy reaches `Q`, the model's `Q` against the simulated strategy is
no likelier than the live `P`. -/
theorem Sim.prob_le_of_all {β α : Type} {pL : β → Prop} {pg : α → Prop} {P : β → Prop} {Q : α → Prop} :
    ∀ {L : Game β} {g : Game α} (d : Sim pL pg L g),
      Sim.All (fun b a => Q a → P b) (fun g σ => prob Q g σ = 0) d →
      ∀ s : Strategy L, prob Q g (d.strategy s) ≤ prob P L s
  | _, _, @Sim.ret _ _ _ _ b a _, h, _ => by
      classical
      have h' : Q a → P b := h
      simp only [prob]
      split_ifs with h₁ h₂ <;> simp_all
  | _, _, .model m d, h, s => by simp only [prob, Sim.strategy]; exact Sim.prob_le_of_all d h s
  | _, _, .live f, h, s => by simp only [prob, Sim.strategy]; exact Sim.prob_le_of_all (f s.1) (h s.1) s.2
  | _, _, @Sim.coin _ _ _ _ C C' iF iN iF' iN' k k' e f, h, s => by
      simp only [prob, Sim.strategy]
      rw [← e.sum_comp, @Fintype.card_congr C' C iF' iF e]
      exact Finset.sum_le_sum fun c' _ => mul_le_mul_right (Sim.prob_le_of_all (f c') (h c') (s (e c'))) _
  | _, _, .dead _ σ, h, _ => by
      have h' : prob Q _ σ = 0 := h
      simp only [Sim.strategy]
      rw [h']
      exact zero_le

/-! ## Outcomes when every message of one type is fixed -/

/-- `OutN nul g b`: `b` is an outcome of `g` on a path every message of `nul`'s type on which is `nul`. -/
inductive OutN {Op : Type} (nul : Op) : {β : Type} → Game β → β → Prop
  | ret {β : Type} (b : β) : OutN nul (.ret b) b
  | send {β M : Type} {k : M → Game β} (m : M) (hm : ∀ e : Op = M, m = cast e nul) {b : β} :
      OutN nul (k m) b → OutN nul (.send M k) b
  | coin {β C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β} (c : C) {b : β} :
      OutN nul (k c) b → OutN nul (@Game.coin β C iF iN k) b

/-- **`answerWith nul` reaches only `OutN nul` outcomes**: on a game every message type of which is inhabited, an
event no such outcome satisfies has probability `0` against it. -/
theorem prob_answerWith {α Op : Type} (nul : Op) {Q : α → Prop} :
    ∀ {g : Game α}, AllInh g → (∀ a, OutN nul g a → ¬ Q a) → ∀ σ : Strategy g, prob Q g (answerWith nul g σ) = 0
  | .ret a, _, h, _ => by classical simp only [prob]; simp [h a (.ret a)]
  | .send M k, hg, h, σ => by
      by_cases hc : ∃ e : Op = M, Nonempty (Strategy (k (cast e nul)))
      · unfold answerWith
        rw [dite_eq_left_of_eq_true (eq_true hc)]
        exact prob_answerWith nul (hg.send_k _) (fun a ha => h a (.send (cast hc.fst nul) (fun _ => rfl) ha)) _
      · unfold answerWith
        rw [dite_eq_right_of_eq_false (eq_false hc)]
        exact prob_answerWith nul (hg.send_k _)
          (fun a ha => h a (.send σ.1 (fun e => absurd ⟨e, (hg.send_k _).strategy⟩ hc) ha)) _
  | @Game.coin _ C _ _ k, hg, h, σ => by
      simp only [prob]
      exact Finset.sum_eq_zero fun c _ =>
        mul_eq_zero_of_right _ (prob_answerWith nul (hg.coin_k c) (fun a ha => h a (.coin c ha)) (σ c))

theorem OutN.of_bind {Op β γ : Type} {nul : Op} {f : β → Game γ} :
    ∀ {g : Game β} {c : γ}, OutN nul (g.bind f) c → ∃ b, OutN nul g b ∧ OutN nul (f b) c
  | .ret b, _, h => ⟨b, .ret b, h⟩
  | .send _ _, _, .send m hm h => let ⟨b, hb, hc⟩ := OutN.of_bind h; ⟨b, .send m hm hb, hc⟩
  | @Game.coin _ _ _ _ _, _, .coin x h => let ⟨b, hb, hc⟩ := OutN.of_bind h; ⟨b, .coin x hb, hc⟩

theorem OutN.of_map {Op β γ : Type} {nul : Op} {f : β → γ} {g : Game β} {c : γ} (h : OutN nul (g.map f) c) :
    ∃ b, OutN nul g b ∧ c = f b := by
  obtain ⟨b, hb, hc⟩ := OutN.of_bind h
  cases hc
  exact ⟨b, hb, rfl⟩

theorem OutN.of_interleaveWith {Op A B : Type} {nul : Op} {g₁ : Game A} {adv : Game B → Game (A × B)}
    {fin : B → Game (A × B)} (hadv : ∀ g₂ c, OutN nul (adv g₂) c → OutN nul g₁ c.1 ∧ OutN nul g₂ c.2)
    (hfin : ∀ b c, OutN nul (fin b) c → OutN nul g₁ c.1 ∧ c.2 = b) :
    ∀ {g₂ : Game B} {c : A × B}, OutN nul (interleaveWith adv fin g₂) c → OutN nul g₁ c.1 ∧ OutN nul g₂ c.2
  | .ret b, c, h => ⟨(hfin b c h).1, by rw [(hfin b c h).2]; exact .ret b⟩
  | .send _ _, c, .send first _ h => by
      cases first
      · cases h with
        | send m hm h =>
          exact ⟨(OutN.of_interleaveWith hadv hfin h).1, .send m hm (OutN.of_interleaveWith hadv hfin h).2⟩
      · exact hadv _ c h
  | @Game.coin _ _ _ _ _, c, .send first _ h => by
      cases first
      · cases h with
        | coin x h => exact ⟨(OutN.of_interleaveWith hadv hfin h).1, .coin x (OutN.of_interleaveWith hadv hfin h).2⟩
      · exact hadv _ c h

theorem OutN.of_interleave {Op A B : Type} {nul : Op} :
    ∀ {g₁ : Game A} {g₂ : Game B} {c : A × B}, OutN nul (interleave g₁ g₂) c → OutN nul g₁ c.1 ∧ OutN nul g₂ c.2
  | .ret a, _, _, h => by
      obtain ⟨b, hb, rfl⟩ := OutN.of_map h
      exact ⟨.ret a, hb⟩
  | .send _ _, _, _, h =>
      OutN.of_interleaveWith
        (fun _ _ h => by
          cases h with
          | send m hm h => exact ⟨.send m hm (OutN.of_interleave h).1, (OutN.of_interleave h).2⟩)
        (fun _ _ h => by obtain ⟨a, ha, rfl⟩ := OutN.of_map h; exact ⟨ha, rfl⟩) h
  | @Game.coin _ _ _ _ _, _, _, h =>
      OutN.of_interleaveWith
        (fun _ _ h => by
          cases h with
          | coin x h => exact ⟨.coin x (OutN.of_interleave h).1, (OutN.of_interleave h).2⟩)
        (fun _ _ h => by obtain ⟨a, ha, rfl⟩ := OutN.of_map h; exact ⟨ha, rfl⟩) h

theorem OutN.of_batch {Op A : Type} {nul : Op} :
    ∀ {gs : List (Game A)} {outs : List A}, OutN nul (batch gs) outs → List.Forall₂ (fun g o => OutN nul g o) gs outs
  | [], _, h => by cases h; exact .nil
  | _ :: _, _, h => by
      obtain ⟨x, hx, rfl⟩ := OutN.of_map h
      exact .cons (OutN.of_interleave hx).1 (OutN.of_batch (OutN.of_interleave hx).2)

theorem OutN.of_KgJ {P OutT Cp Op : Type} {J : ℕ} (out : Fin J → Cp → P × P → Op → OutT) {nul : Op} {j : Fin J}
    {cap : Cp} {pend : P × P} {o : OutT} (h : OutN nul (KgJ out j cap pend) o) : o = out j cap pend nul := by
  cases h with
  | send m hm hr =>
    cases hr
    rw [hm rfl]
    rfl

/-- **At the tables' positions**, an `OutN nul` outcome is every table's output at its openings `nul`. -/
theorem OutN.of_posOf {P OutT Cp Op : Type} {J : ℕ} (out : Fin J → Cp → P × P → Op → OutT) (nul : Op)
    (caps : Fin J → Cp) (gs : Fin J → Fin 2 → Game P) (dflt : OutT) {outs : List OutT}
    (h : OutN nul (posOf out caps gs) outs) (j : Fin J) :
    ∃ pend : P × P, outs.getD j dflt = out j (caps j) pend nul := by
  have hb := OutN.of_batch h
  have hl : J = outs.length := by simpa using hb.length_eq
  have hj := hb.get (i := j.val) (by simp) (hl ▸ j.isLt)
  simp only [List.get_eq_getElem, List.getElem_ofFn] at hj
  obtain ⟨pend, -, hk⟩ := OutN.of_bind hj
  refine ⟨pend, ?_⟩
  rw [List.getD_eq_getElem _ _ (hl ▸ j.isLt)]
  exact OutN.of_KgJ out hk

/-- **No dead branch at the tables' positions reaches `Q`** when it answers the openings with `nul`, for an event `Q`
no outcome with every table's openings `nul` satisfies. -/
theorem prob_posOf_answerWith {P OutT Cp Op : Type} [Nonempty Op] {J : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
    (nul : Op) (caps : Fin J → Cp) {gs : Fin J → Fin 2 → Game P} (hg : ∀ j r, AllInh (gs j r)) (dflt : OutT)
    {Q : List OutT → Prop} (hQ : ∀ outs, (∀ j : Fin J, ∃ pend, outs.getD j dflt = out j (caps j) pend nul) → ¬ Q outs)
    (σ : Strategy (posOf out caps gs)) : prob Q (posOf out caps gs) (answerWith nul _ σ) = 0 :=
  prob_answerWith nul (allInh_posOf out caps hg) (fun outs h => hQ outs (OutN.of_posOf out nul caps gs dflt h)) σ

end Claims.Bridge
