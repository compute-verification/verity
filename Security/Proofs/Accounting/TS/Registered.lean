import Models.Accounting.Guarantees.TSRegistered
import Proofs.Accounting.TS.Drawn
import Proofs.Accounting.TS.Rounds

/-!
# A padded window's records and its counting step

`RegisteredRightOff`, `EpochFormedRightOff` and `CountingSel` (note:20261018T2200Z-draft-padded-credit-rule, round 5).

`drawn_right_bound`: `DrawnRightOff`'s event ("in some round, binding fails or some drawn unit's proof is unsound",
`S·(δ_bind + δ_ks)` by the union bound), with what holds off it stated in full: an accepted proof of a drawn unit opens
its unit's committed values (not `MovedIn`), and they are right (not `Unsound`, `Pins`).

`RegisteredRightOff`, `EpochFormedRightOff`: that event. Off it, on an accepted run every drawn unit's proof is accepted
(`Of`), and every record unit is drawn (`AlwaysProved`: the draw names it at every draw coin), so it is right at its
committed values, and so is its record (`Computes`). A window's slot reading an epoch record is at the record's formed
row position (`DomainBound`), so its opening, which is its committed value, is the record's formed row.

`records_lift`: a record's failure read on the window's paths, the record's coins being coordinates of the window's
(a projection that carries the window's uniform coins to the record's), is at most the record's failure term. So
`RegisteredRightOff`'s and `EpochFormedRightOff`'s terms are `ProofsSoundOfSelected`'s `δ_reg` and `δ_epoch`.

`CountingSel`: off (I8)'s event an accepted window is accepted by the ideal game with
`Pub − d·N_w ≤ S·fixed`, so `S·fixed < Pub − N_w·Λ·ln(1/δ_s)` gives `Λ·ln(1/δ_s) < d`, which the budget's tail
(`budget_tail`) puts at `exp(−ln(1/δ_s)) = δ_s`.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Pouw.RelativeSecurity

/-- **`DrawnRightOff`'s event, with binding exposed**: outside an event of probability at most `S·(δ_bind + δ_ks)`,
every accepted proof of a drawn unit opens its unit's committed values, and they are right. -/
theorem drawn_right_bound {J Q Pos V : Type} [Nonempty V] {C B : Type} [Fintype C] [Fintype B] {S m s : ℕ}
    (pf : DrawnLayer C B S m s J Q Pos V) {δbind δks : ℝ}
    (hB : ∀ w, prob (fun x => pf.MovedIn x w) ≤ δbind) (hZ : pf.JointSound δks) (hPin : pf.Pins) :
    ∃ E : Paths (Paths C m × B) S → Prop, prob E ≤ S * (δbind + δks) ∧
      ∀ x, ¬ E x → ∀ w i, pf.accepted x w i →
        pf.opened x w i = pf.view x w (pf.drawnAt x w i) ∧
          pf.right x w (pf.drawnAt x w i) (pf.view x w (pf.drawnAt x w i)) := by
  refine ⟨fun x => ∃ w ∈ (Finset.univ : Finset (Fin S)), pf.MovedIn x w ∨ ∃ i, pf.Unsound x w i, ?_, ?_⟩
  · calc prob _ ≤ ∑ w : Fin S, prob (fun x => pf.MovedIn x w ∨ ∃ i, pf.Unsound x w i) := prob_exists_le _ _
      _ ≤ ∑ _w : Fin S, (δbind + δks) :=
          Finset.sum_le_sum fun w _ => (prob_or_le _ _).trans (add_le_add (hB w) (hZ w))
      _ = S * (δbind + δks) := by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  · intro x hE w i hacc
    have heq : pf.opened x w i = pf.view x w (pf.drawnAt x w i) := by
      by_contra hne
      exact hE ⟨w, Finset.mem_univ _, Or.inl ⟨i, hacc, hne⟩⟩
    refine ⟨heq, ?_⟩
    apply hPin
    rw [← heq]
    by_contra hr
    exact hE ⟨w, Finset.mem_univ _, Or.inr ⟨i, hacc, hr⟩⟩

/-- **Units in every draw are right** on a path where every drawn unit's proof is accepted and every accepted drawn
unit is right at its committed values. -/
theorem units_right {J Q Pos V : Type} [Nonempty V] {C B : Type} [Fintype C] [Fintype B] {S m s : ℕ}
    (pf : DrawnLayer C B S m s J Q Pos V) (units : Finset (Fin S × J))
    (hA : ∀ u ∈ units, ∀ b : B, ∃ i, pf.draw b i = u.2) (x : Paths (Paths C m × B) S)
    (hacc : ∀ w i, pf.accepted x w i)
    (hR : ∀ w i, pf.accepted x w i → pf.right x w (pf.drawnAt x w i) (pf.view x w (pf.drawnAt x w i))) :
    ∀ u ∈ units, pf.right x u.1 u.2 (pf.view x u.1 u.2) := by
  intro u hu
  obtain ⟨i, hi⟩ := hA u hu (coinAt x u.1).2
  have hd : pf.drawnAt x u.1 i = u.2 := hi
  have h := hR u.1 i (hacc u.1 i)
  rwa [hd] at h

theorem RegisteredRightOff : Accounting.Guarantees.TS.RegisteredRightOff := by
  intro J Q Pos V _ C B _ _ S m s rg δbind δks hA hOf hC hB hZ hPin
  obtain ⟨E, hE, hR⟩ := drawn_right_bound rg.toDrawnLayer hB hZ hPin
  exact ⟨E, hE, fun x hx hEx r =>
    hC x r (units_right rg.toDrawnLayer (rg.units r) (hA r) x (hOf x hx) fun w i h => (hR x hEx w i h).2)⟩

theorem EpochFormedRightOff : Accounting.Guarantees.TS.EpochFormedRightOff := by
  intro J Q Pos V _ C B _ _ S m s ep δbind δks hA hOf hD hC hB hZ hPin
  obtain ⟨E, hE, hR⟩ := drawn_right_bound ep.toDrawnLayer hB hZ hPin
  refine ⟨E, hE, fun x hx hEx => ?_⟩
  have hRight : ∀ e, ep.Right x e := fun e =>
    hC x e (units_right ep.toDrawnLayer (ep.units e) (hA e) x (hOf x hx) fun w i h => (hR x hEx w i h).2)
  refine ⟨hRight, fun w i k e hacc hre => ?_⟩
  have hop : ep.opened x w i k = ep.toDrawnLayer.view x w (ep.toDrawnLayer.drawnAt x w i) k :=
    congrFun (hR x hEx w i hacc).1 k
  have hp : ep.pos w (ep.toDrawnLayer.drawnAt x w i) k = ep.formedPos e := hD _ k e hre
  rw [hop]
  change ep.toDrawnLayer.committed x (ep.pos w (ep.toDrawnLayer.drawnAt x w i) k) = _
  rw [hp]
  exact hRight e

/-- **A record's failure on the window's paths**: if the window's coins carry the record's as coordinates (`proj`
carries the window's uniform coins to the record's), an accepted window's record is accepted, and a right record on the
projected coins is right on the window, then the window's accepted paths with a wrong record weigh at most the record's
failure term. -/
theorem records_lift {X Y : Type} [Fintype X] [Fintype Y] (proj : X → Y)
    (hproj : ∀ F : Y → Prop, prob (fun x => F (proj x)) = prob F) {acc R : X → Prop} {accY RY E : Y → Prop} {δ : ℝ}
    (hacc : ∀ x, acc x → accY (proj x)) (hR : ∀ x, RY (proj x) → R x) (hE : prob E ≤ δ)
    (hRY : ∀ y, accY y → ¬ E y → RY y) : prob (fun x => acc x ∧ ¬ R x) ≤ δ := by
  refine (prob_mono (q := fun x => E (proj x)) fun x hx => ?_).trans ((hproj E).le.trans hE)
  by_contra hEx
  exact hx.2 (hR x (hRY _ (hacc x hx.1) hEx))

theorem CountingSel : Accounting.Guarantees.TS.CountingSel := by
  intro Ω _ S K T L U P pub Nw Λ δs δ₁ hS hNw hΛ hδs hBud hCS
  unfold CreditSound at hCS
  obtain ⟨E, hE, hCS⟩ := hCS
  have hexp : Real.exp (-Real.log (1 / δs)) = δs := by
    rw [one_div, Real.log_inv, neg_neg, Real.exp_log hδs]
  have htail := budget_tail P.game Λ hΛ hBud (Real.log (1 / δs))
  have hSpos : (0 : ℝ) < S := by exact_mod_cast hS
  refine (prob_mono (q := fun x => (P.game.accepts x ∧ Λ * Real.log (1 / δs) ≤ P.game.deficiency x) ∨ E x)
    fun x hx => ?_).trans ((prob_or_le _ _).trans (add_le_add (htail.trans hexp.le) hE))
  obtain ⟨hacc, hlt⟩ := hx
  by_cases hEx : E x
  · exact Or.inr hEx
  · obtain ⟨hg, hfix⟩ := hCS x hacc hEx
    refine Or.inl ⟨hg, ?_⟩
    have h1 : pub x - P.game.deficiency x * Nw ≤ (P.run x).fixed * S := (div_le_iff₀ hSpos).mp hfix
    by_contra hcon
    have h2 : Nw * P.game.deficiency x < Nw * (Λ * Real.log (1 / δs)) :=
      mul_lt_mul_of_pos_left (not_le.mp hcon) hNw
    linarith

end Accounting.SecurityProofs.TS
