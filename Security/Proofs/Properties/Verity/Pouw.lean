import Properties.Verity.Pouw
import Proofs.Core.Game.Prob

/-!
# PoUW's component bound from its named premises

`pouwComponentBound : Verity.PouwComponentBound`, by the union bound over the three parts of the event: a release
missing from the log or a break (the finder's, `FinderAnswers`), a drawn tile unbound in the extractor's transcript
(`ServiceSound`), and the rest, where the check's reading gives each window's salt and draw and the claim's failure is
a count below the credit floor (`CreditedPeriodRefines`, through `PublishedCredit`).
-/

namespace Verity

open Accounting Core
open scoped ENNReal

theorem pouwComponentBound : PouwComponentBound := by
  intro ι α _ d V R Λ δs δbind δks hsalt hreg hdraw hfind hserv hcred hW hδ θ F c σ
  classical
  unfold Vocab.error
  split_ifs with hcr
  · show _ ≤ V.delta θ .pouw (V.cost θ F c σ) + V.loss .pouw θ ((V.live θ F c).collides sha512Msg .pouw σ)
    let B := fun o : Outcome => ComponentAccepts .pouw o.sent ∧ (¬ V.Logged .pouw o ∨ R.brk θ F c σ o)
    let S := fun o : Outcome => R.AtCode θ F c σ o ∧ ∃ w < R.windows o.params, ∃ u ∈ R.drawn F o.sent w,
      u ∈ unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)
    let P := fun o : Outcome => R.AtCode θ F c σ o ∧
      (∀ w < R.windows o.params, (∃ r, R.salt o.sent w = some r ∧ r ∈ V.coinsRead .pouw o.sent) ∧
        ∃ r, R.draw o.sent w = some r ∧ r ∈ V.coinsRead .pouw o.sent ∧ r.answers = R.registration o.sent w ∧
          R.drawn F o.sent w = R.select F r.coins ∧
          ∀ u ∈ R.drawn F o.sent w, u ∉ unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)) ∧
      ((V.circuitOf F).size (V.computed θ F c σ o) : ℝ) < Pouw.creditGates Λ δs (R.windows θ) (R.liveSize θ)
    have hpt : ∀ o : Outcome, ComponentAccepts .pouw o.sent ∧ ¬ V.Holds θ F c σ .pouw o → B o ∨ (S o ∨ P o) := by
      intro o ho
      by_cases hb : ¬ V.Logged .pouw o ∨ R.brk θ F c σ o
      · exact Or.inl ⟨ho.1, hb⟩
      have hl : V.Logged .pouw o := not_not.mp fun h => hb (Or.inl h)
      have hat : R.AtCode θ F c σ o := ⟨ho.1, hl, fun h => hb (Or.inr h)⟩
      by_cases hs : ∃ w < R.windows o.params, ∃ u ∈ R.drawn F o.sent w,
          u ∈ unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)
      · exact Or.inr (Or.inl ⟨hat, hs⟩)
      refine Or.inr (Or.inr ⟨hat, fun w hw => ?_, ?_⟩)
      · obtain ⟨r, hr1, hr2⟩ := hsalt o.sent ho.1 w hw
        obtain ⟨q, hq1, hq2, hq3⟩ := hreg o.sent ho.1 w hw
        exact ⟨⟨r, hr1, hr2⟩, q, hq1, hq2, hq3, hdraw F o.sent w q hq1, fun u hu hun => hs ⟨w, hw, u, hu, hun⟩⟩
      · have hH : ¬ ((V.certOf θ).W ≤ ((V.circuitOf F).size (V.computed θ F c σ o) : ℝ)) := ho.2
        exact lt_of_lt_of_le (lt_of_not_ge hH) (hW θ)
    calc _ ≤ Game.prob (fun o => B o ∨ (S o ∨ P o)) (V.game θ F c) σ := Game.prob_mono hpt _ _
      _ ≤ Game.prob B (V.game θ F c) σ + (Game.prob S (V.game θ F c) σ + Game.prob P (V.game θ F c) σ) :=
          (Game.prob_or_le _ _ _ _).trans (add_le_add le_rfl (Game.prob_or_le _ _ _ _))
      _ ≤ V.loss .pouw θ ((V.live θ F c).collides sha512Msg .pouw σ) +
            (R.windows θ * δks + (ENNReal.ofReal δs + R.windows θ * δbind)) :=
          add_le_add (hfind θ F c σ hcr) (add_le_add (hserv θ F c σ) (hcred θ F c σ))
      _ = (ENNReal.ofReal δs + R.windows θ * (δbind + δks)) +
            V.loss .pouw θ ((V.live θ F c).collides sha512Msg .pouw σ) := by ring
      _ ≤ _ := add_le_add_right (hδ θ _) _
  · exact le_top

end Verity
