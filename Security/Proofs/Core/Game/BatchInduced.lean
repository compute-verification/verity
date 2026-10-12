import Proofs.Core.Game.Prob

/-!
# One game of a batch, for a strategy whose induced strategies are in a class

`value_batch_getD_le` bounds an event of one position's output in a batch by the event's `value` in that game alone, for
an optimal prover. A bound for a class of strategies (the query-bounded ones a computational soundness theorem speaks
of) needs it per strategy: a strategy for the batch, read at one position, does no better than some strategy it induces
on that position's game (`InducedAt`), the one that fixes each coin of the other games where the probability is largest.

* `avg_le_of_max`: an average is at most its largest term, which picks the other game's coin;
* `prob_interleave_fst_le_induced`, `prob_interleave_snd_le_induced`: either side of `interleave`;
* `prob_batch_getD_le_induced`, `prob_batch_getD_le_of_induced`: one position of a batch, by induction on the games
  before it.
-/

namespace Verity.Game

open scoped ENNReal

variable {α β : Type}

/-- An average is at most its largest term. -/
theorem avg_le_of_max {C : Type} [Fintype C] [Nonempty C] (f : C → ℝ≥0∞) : ∃ c, avg f ≤ f c := by
  obtain ⟨c, -, hc⟩ := Finset.exists_max_image Finset.univ f Finset.univ_nonempty
  exact ⟨c, (avg_mono fun c' => hc c' (Finset.mem_univ c')).trans (avg_const (f c)).le⟩

/-- An event already settled is no likelier in a game than at its end. -/
theorem prob_const_le_ret {γ : Type} (P : α → Prop) (a : α) (g : Game γ) (s : Strategy g) :
    prob (fun _ => P a) g s ≤ prob P (.ret a) PUnit.unit := by
  classical
  by_cases hP : P a
  · calc prob (fun _ => P a) g s ≤ 1 := prob_le_one _ _ _
      _ = prob P (.ret a) PUnit.unit := by simp [prob, hP]
  · have e : (fun _ : γ => P a) = fun _ => False := funext fun _ => propext ⟨hP, False.elim⟩
    rw [e, prob_false]
    exact zero_le

theorem inducedWithL_le {γ : Type} (g : Game γ) (Q : γ → Prop) (p : α × β → Prop)
    (adv₁ : Game β → Game (α × β)) (fin₁ : β → Game (α × β))
    (advR : Strategy g → (h : Game β) → Strategy (adv₁ h) → Prop)
    (finR : Strategy g → (b : β) → Strategy (fin₁ b) → Prop)
    (hadv : ∀ h σ', ∃ s, advR s h σ' ∧ prob p (adv₁ h) σ' ≤ prob Q g s)
    (hfin : ∀ b σ', ∃ s, finR s b σ' ∧ prob p (fin₁ b) σ' ≤ prob Q g s) :
    ∀ (h : Game β) (σ : Strategy (interleaveWith adv₁ fin₁ h)),
      ∃ s, InducedWithL adv₁ fin₁ (advR s) (finR s) h σ ∧ prob p (interleaveWith adv₁ fin₁ h) σ ≤ prob Q g s
  | .ret b, σ => hfin b σ
  | .send M k, ⟨true, σ'⟩ => hadv (.send M k) σ'
  | .send M k, ⟨false, σ'⟩ => inducedWithL_le g Q p adv₁ fin₁ advR finR hadv hfin (k σ'.1) σ'.2
  | @Game.coin _ C iF iN k, ⟨true, σ'⟩ => hadv (@Game.coin _ C iF iN k) σ'
  | @Game.coin _ C iF iN k, ⟨false, σ'⟩ => by
      choose s hs hle using fun c => inducedWithL_le g Q p adv₁ fin₁ advR finR hadv hfin (k c) (σ' c)
      obtain ⟨c, hc⟩ := avg_le_of_max fun c => prob p (interleaveWith adv₁ fin₁ (k c)) (σ' c)
      exact ⟨s c, ⟨c, hs c⟩, hc.trans (hle c)⟩

/-- **The first of two interleaved games.** An event of the first game's output is at most as likely, for a strategy
`σ` of the pair, as in the first game alone for some strategy that `σ` induces on it. -/
theorem prob_interleave_fst_le_induced (Q : α → Prop) :
    ∀ (g : Game α) (h : Game β) (σ : Strategy (interleave g h)),
      ∃ s, InducedL g h σ s ∧ prob (fun x => Q x.1) (interleave g h) σ ≤ prob Q g s
  | .ret a, h, σ => ⟨PUnit.unit, trivial,
      (prob_map (fun x => Q x.1) (fun b => (a, b)) h σ).trans_le (prob_const_le_ret Q a h _)⟩
  | .send M k, h, σ =>
      inducedWithL_le (.send M k) Q (fun x => Q x.1)
        (fun g₂ => .send M fun m => interleave (k m) g₂) (fun b => (Game.send M k).map fun a => (a, b))
        (fun s h' σ' => ∃ s', s = ⟨σ'.1, s'⟩ ∧ InducedL (k σ'.1) h' σ'.2 s')
        (fun s b σ' => s = Strategy.ofMap (fun a => (a, b)) (.send M k) σ')
        (fun h' σ' => by
          obtain ⟨s', hs', hle⟩ := prob_interleave_fst_le_induced Q (k σ'.1) h' σ'.2
          exact ⟨⟨σ'.1, s'⟩, ⟨s', rfl, hs'⟩, hle⟩)
        (fun b σ' => ⟨Strategy.ofMap (fun a => (a, b)) (.send M k) σ', rfl,
          (prob_map (fun x => Q x.1) (fun a => (a, b)) (.send M k) σ').le⟩) h σ
  | @Game.coin _ C iF iN k, h, σ =>
      inducedWithL_le (@Game.coin _ C iF iN k) Q (fun x => Q x.1)
        (fun g₂ => @Game.coin _ C iF iN fun c => interleave (k c) g₂)
        (fun b => (@Game.coin _ C iF iN k).map fun a => (a, b))
        (fun s h' σ' => ∀ c, InducedL (k c) h' (σ' c) (s c))
        (fun s b σ' => s = Strategy.ofMap (fun a => (a, b)) (@Game.coin _ C iF iN k) σ')
        (fun h' σ' => by
          choose s hs hle using fun c => prob_interleave_fst_le_induced Q (k c) h' (σ' c)
          exact ⟨s, hs, avg_mono hle⟩)
        (fun b σ' => ⟨Strategy.ofMap (fun a => (a, b)) (@Game.coin _ C iF iN k) σ', rfl,
          (prob_map (fun x => Q x.1) (fun a => (a, b)) (@Game.coin _ C iF iN k) σ').le⟩) h σ

theorem inducedWithR_le (Q : β → Prop) (p : α × β → Prop)
    (adv₁ : Game β → Game (α × β)) (fin₁ : β → Game (α × β))
    (advR : (h : Game β) → Strategy (adv₁ h) → Strategy h → Prop)
    (hadv : ∀ h σ', ∃ s, advR h σ' s ∧ prob p (adv₁ h) σ' ≤ prob Q h s)
    (hfin : ∀ b σ', prob p (fin₁ b) σ' ≤ prob Q (.ret b) PUnit.unit) :
    ∀ (h : Game β) (σ : Strategy (interleaveWith adv₁ fin₁ h)),
      ∃ s, InducedWithR adv₁ fin₁ advR h σ s ∧ prob p (interleaveWith adv₁ fin₁ h) σ ≤ prob Q h s
  | .ret b, σ => ⟨PUnit.unit, trivial, hfin b σ⟩
  | .send M k, ⟨true, σ'⟩ => hadv (.send M k) σ'
  | .send M k, ⟨false, σ'⟩ => by
      obtain ⟨s', hs', hle⟩ := inducedWithR_le Q p adv₁ fin₁ advR hadv hfin (k σ'.1) σ'.2
      exact ⟨⟨σ'.1, s'⟩, ⟨s', rfl, hs'⟩, hle⟩
  | @Game.coin _ C iF iN k, ⟨true, σ'⟩ => hadv (@Game.coin _ C iF iN k) σ'
  | @Game.coin _ C iF iN k, ⟨false, σ'⟩ => by
      choose s hs hle using fun c => inducedWithR_le Q p adv₁ fin₁ advR hadv hfin (k c) (σ' c)
      exact ⟨s, hs, avg_mono hle⟩

/-- **The second of two interleaved games.** An event of the second game's output is at most as likely, for a strategy
`σ` of the pair, as in the second game alone for some strategy that `σ` induces on it. -/
theorem prob_interleave_snd_le_induced (Q : β → Prop) :
    ∀ (g : Game α) (h : Game β) (σ : Strategy (interleave g h)),
      ∃ s, InducedR g h σ s ∧ prob (fun x => Q x.2) (interleave g h) σ ≤ prob Q h s
  | .ret a, h, σ => ⟨Strategy.ofMap (fun b => (a, b)) h σ, rfl,
      (prob_map (fun x => Q x.2) (fun b => (a, b)) h σ).le⟩
  | .send M k, h, σ =>
      inducedWithR_le Q (fun x => Q x.2)
        (fun g₂ => .send M fun m => interleave (k m) g₂) (fun b => (Game.send M k).map fun a => (a, b))
        (fun h' σ' s' => InducedR (k σ'.1) h' σ'.2 s')
        (fun h' σ' => prob_interleave_snd_le_induced Q (k σ'.1) h' σ'.2)
        (fun b σ' => (prob_map (fun x => Q x.2) (fun a => (a, b)) (.send M k) σ').trans_le
          (prob_const_le_ret Q b _ _)) h σ
  | @Game.coin _ C iF iN k, h, σ =>
      inducedWithR_le Q (fun x => Q x.2)
        (fun g₂ => @Game.coin _ C iF iN fun c => interleave (k c) g₂)
        (fun b => (@Game.coin _ C iF iN k).map fun a => (a, b))
        (fun h' σ' s' => ∃ c, InducedR (k c) h' (σ' c) s')
        (fun h' σ' => by
          choose s hs hle using fun c => prob_interleave_snd_le_induced Q (k c) h' (σ' c)
          obtain ⟨c, hc⟩ := avg_le_of_max fun c => prob Q h' (s c)
          exact ⟨s c, ⟨c, hs c⟩, (avg_mono hle).trans hc⟩)
        (fun b σ' => (prob_map (fun x => Q x.2) (fun a => (a, b)) (@Game.coin _ C iF iN k) σ').trans_le
          (prob_const_le_ret Q b _ _)) h σ

/-- **One position of a batch, for a fixed strategy.** An event of position `pre.length`'s output is at most as likely,
for a strategy `σ` of the batch, as in that position's game alone for some strategy that `σ` induces on it. -/
theorem prob_batch_getD_le_induced (Q : α → Prop) (d : α) (g : Game α) (post : List (Game α)) :
    ∀ (pre : List (Game α)) (σ : Strategy (batch (pre ++ g :: post))),
      ∃ s : Strategy g, InducedAt g post pre σ s ∧
        prob (fun outs => Q (outs.getD pre.length d)) (batch (pre ++ g :: post)) σ ≤ prob Q g s
  | [], σ => by
      obtain ⟨s, hs, hle⟩ := prob_interleave_fst_le_induced Q g (batch post)
        (Strategy.ofMap (fun x : α × List α => x.1 :: x.2) _ σ)
      refine ⟨s, hs, ?_⟩
      have e := prob_map (fun outs => Q (outs.getD ([] : List (Game α)).length d))
        (fun x : α × List α => x.1 :: x.2) (g.interleave (batch post)) σ
      exact e.trans_le ((prob_mono (fun x hx => by simpa using hx) _ _).trans hle)
  | h :: pre, σ => by
      obtain ⟨σ', hσ', hle⟩ := prob_interleave_snd_le_induced (fun outs => Q (outs.getD pre.length d)) h
        (batch (pre ++ g :: post)) (Strategy.ofMap (fun x : α × List α => x.1 :: x.2) _ σ)
      obtain ⟨s, hs, hle'⟩ := prob_batch_getD_le_induced Q d g post pre σ'
      refine ⟨s, ⟨σ', hσ', hs⟩, ?_⟩
      have e := prob_map (fun outs => Q (outs.getD (h :: pre).length d))
        (fun x : α × List α => x.1 :: x.2) (h.interleave (batch (pre ++ g :: post))) σ
      exact e.trans_le (((prob_mono (fun x hx => by simpa using hx) _ _).trans hle).trans hle')

/-- **A bound for a class of strategies holds in a batch.** If every strategy of `g` in `Cls` gives `Q` probability at
most `ε`, so does a strategy of the batch at `g`'s position whenever every strategy it induces on `g` is in `Cls`. -/
theorem prob_batch_getD_le_of_induced (Q : α → Prop) (d : α) (g : Game α) (post pre : List (Game α))
    (Cls : Strategy g → Prop) (ε : ℝ≥0∞) (hg : ∀ s, Cls s → prob Q g s ≤ ε)
    (σ : Strategy (batch (pre ++ g :: post))) (hσ : ∀ s, InducedAt g post pre σ s → Cls s) :
    prob (fun outs => Q (outs.getD pre.length d)) (batch (pre ++ g :: post)) σ ≤ ε := by
  obtain ⟨s, hs, hle⟩ := prob_batch_getD_le_induced Q d g post pre σ
  exact hle.trans (hg s (hσ s hs))

end Verity.Game
