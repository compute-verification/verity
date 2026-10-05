import Pouw.Protocol.Draft.BatchSeq

/-!
# What a batch played one game after another induces (provisional)

Core-only lemmas, held here with `Pouw.Draft.Game` until both move to core's package. The strategy that plays a batch
one game after another (`Strategy.batchSeq`) induces, on the game at each position, only that game's own strategy
(`inducedAt_batchSeq`): it plays each game to its end before the next, so no other game's coin is ever drawn inside it.
-/

namespace Pouw.Draft.Game

open Verity Verity.Game

variable {α β : Type}

/-- Relabelling a strategy and reading it back gives it again. -/
theorem ofMap_toMap (f : α → β) :
    ∀ (g : Game α) (s : Strategy g), Strategy.ofMap f g (Strategy.toMap f g s) = s
  | .ret _, _ => rfl
  | .send _ k, ⟨m, s'⟩ => by
      show (⟨m, Strategy.ofMap f (k m) (Strategy.toMap f (k m) s')⟩ : Strategy (Game.send _ k)) = ⟨m, s'⟩
      rw [ofMap_toMap f (k m) s']
  | @Game.coin _ _ _ _ k, s => funext fun c => ofMap_toMap f (k c) (s c)

/-- One game, then the other, induces on the first only its own strategy. -/
theorem inducedL_seq :
    ∀ (g : Game α) (h : Game β) (sg : Strategy g) (sh : Strategy h) (s : Strategy g),
      InducedL g h (Strategy.seq g h sg sh) s → s = sg
  | .ret _, _, _, _, _, _ => rfl
  | .send M k, .ret b, sg, _, s, hs => by
      have e : s = Strategy.ofMap (fun a => (a, b)) (.send M k) (Strategy.toMap (fun a => (a, b)) (.send M k) sg) :=
        hs
      rw [e, ofMap_toMap]
  | .send M k, .send _ _, sg, sh, s, hs => by
      obtain ⟨s', rfl, hs'⟩ := hs
      obtain rfl := inducedL_seq (k sg.1) _ sg.2 sh s' hs'
      rfl
  | .send M k, @Game.coin _ _ _ _ _, sg, sh, s, hs => by
      obtain ⟨s', rfl, hs'⟩ := hs
      obtain rfl := inducedL_seq (k sg.1) _ sg.2 sh s' hs'
      rfl
  | @Game.coin _ C iF iN k, .ret b, sg, _, s, hs => by
      have e : s = Strategy.ofMap (fun a => (a, b)) (@Game.coin _ C iF iN k)
          (Strategy.toMap (fun a => (a, b)) (@Game.coin _ C iF iN k) sg) := hs
      rw [e, ofMap_toMap]
  | @Game.coin _ _ _ _ k, .send _ _, sg, sh, s, hs => funext fun c => inducedL_seq (k c) _ (sg c) sh (s c) (hs c)
  | @Game.coin _ _ _ _ k, @Game.coin _ _ _ _ _, sg, sh, s, hs =>
      funext fun c => inducedL_seq (k c) _ (sg c) sh (s c) (hs c)

/-- One game, then the other, induces on the second only its own strategy. -/
theorem inducedR_seq :
    ∀ (g : Game α) (h : Game β) (sg : Strategy g) (sh : Strategy h) (s : Strategy h),
      InducedR g h (Strategy.seq g h sg sh) s → s = sh
  | .ret a, h, _, sh, s, hs => by
      have e : s = Strategy.ofMap (fun b => (a, b)) h (Strategy.toMap (fun b => (a, b)) h sh) := hs
      rw [e, ofMap_toMap]
  | .send _ _, .ret _, _, _, _, _ => rfl
  | .send _ k, .send _ _, sg, sh, s, hs => inducedR_seq (k sg.1) _ sg.2 sh s hs
  | .send _ k, @Game.coin _ _ _ _ _, sg, sh, s, hs => inducedR_seq (k sg.1) _ sg.2 sh s hs
  | @Game.coin _ _ _ _ _, .ret _, _, _, _, _ => rfl
  | @Game.coin _ _ _ _ k, .send _ _, sg, sh, s, hs => by
      obtain ⟨c, hc⟩ := hs
      exact inducedR_seq (k c) _ (sg c) sh s hc
  | @Game.coin _ _ _ _ k, @Game.coin _ _ _ _ _, sg, sh, s, hs => by
      obtain ⟨c, hc⟩ := hs
      exact inducedR_seq (k c) _ (sg c) sh s hc

/-- **A batch played one game after another induces, on the game at each position, only that game's own strategy.** -/
theorem inducedAt_batchSeq (g : Game α) (post : List (Game α)) :
    ∀ (pre : List (Game α)) (ss : Strategies (pre ++ g :: post)) (s : Strategy g),
      InducedAt g post pre (Strategy.batchSeq _ ss) s → s = Strategies.get pre ss
  | [], ss, s, hs => by
      have hs' : InducedL g (batch post) (Strategy.ofMap (fun x : α × List α => x.1 :: x.2) _
          (Strategy.toMap (fun x : α × List α => x.1 :: x.2) _
            (Strategy.seq g (batch post) ss.1 (Strategy.batchSeq post ss.2)))) s := hs
      rw [ofMap_toMap] at hs'
      exact inducedL_seq g _ ss.1 _ s hs'
  | p :: pre, ss, s, hs => by
      obtain ⟨σ', hR, hA⟩ := hs
      have hR' : InducedR p (batch (pre ++ g :: post)) (Strategy.ofMap (fun x : α × List α => x.1 :: x.2) _
          (Strategy.toMap (fun x : α × List α => x.1 :: x.2) _
            (Strategy.seq p (batch (pre ++ g :: post)) ss.1 (Strategy.batchSeq _ ss.2)))) σ' := hR
      rw [ofMap_toMap] at hR'
      have e := inducedR_seq p _ ss.1 _ σ' hR'
      subst e
      exact inducedAt_batchSeq g post pre ss.2 s hA

theorem heq_snd {A B B' : Type} (h : B = B') {p : A × B} {p' : A × B'} (hp : HEq p p') : HEq p.2 p'.2 := by
  subst h
  cases hp
  rfl

/-- The strategy at a position of `Strategies.ofMap G hon ts` is `hon t` for the `t` there. -/
theorem get_ofMap {ι : Type} (G : ι → Game α) (hon : (t : ι) → Strategy (G t)) {g : Game α}
    {post : List (Game α)} :
    ∀ (pre : List (Game α)) (ts : List ι) (ss : Strategies (pre ++ g :: post)), pre ++ g :: post = ts.map G →
      HEq ss (Strategies.ofMap G hon ts) →
      ∃ (t : ι) (e : G t = g), Strategies.get pre ss = cast (congrArg Strategy e) (hon t)
  | [], [], _, hL, _ => absurd hL (by simp)
  | [], t :: ts, ss, hL, hss => by
      obtain ⟨rfl, rfl⟩ := List.cons.inj hL
      cases hss
      exact ⟨t, rfl, rfl⟩
  | _ :: _, [], _, hL, _ => absurd hL (by simp)
  | p :: pre, t :: ts, ss, hL, hss => by
      obtain ⟨rfl, hL'⟩ := List.cons.inj hL
      exact get_ofMap G hon pre ts ss.2 hL' (heq_snd (congrArg Strategies hL') hss)

/-- **The batch strategy's induced strategies, read through a cast**: if `σ` is the batch strategy of `L` played one
game after another, then wherever `σ`'s game is read as `batch (pre ++ g :: post)` with that list `L`, every strategy
`σ` induces on `g` is the one `ss` holds for it. -/
theorem inducedAt_of_heq {L : List (Game α)} (ss : Strategies L) {G : Game (List α)} (σ : Strategy G)
    (hσ : HEq σ (Strategy.batchSeq L ss)) (pre : List (Game α)) (g : Game α) (post : List (Game α))
    (e : batch (pre ++ g :: post) = G) (hL : pre ++ g :: post = L) (s : Strategy g)
    (hs : InducedAt g post pre (e ▸ σ) s) :
    ∃ ss' : Strategies (pre ++ g :: post), HEq ss' ss ∧ s = Strategies.get pre ss' := by
  subst hL
  subst e
  cases hσ
  exact ⟨ss, HEq.rfl, inducedAt_batchSeq g post pre ss s hs⟩

end Pouw.Draft.Game
