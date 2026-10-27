import Definitions.Accounting.Guarantees.PerRun
import Proofs.Accounting.Device
import Proofs.Pouw.Birthday

/-!
# PoUW's guarantee per run, from TT per coin

`Pouw.SecurityProofs.EndToEndPerCoin` bounds acceptance at the adversary's own cost with its finder's log collision-free;
the birthday bound (`Pouw.Proofs.pr_finder_collides_le`) charges the collision at `q_A(q_A − 1)/(2|R|)`, which gives
the per-run form with no hypothesis on the cost model. A run that realizes PoUW's game has the compute audit's per-run
guarantee, because the compute account's work covers the adversary's cost at every coin and the device's oracle rate
bounds its calls.
-/

namespace Accounting

open Pouw

variable {Q R S : Type}

namespace SecurityProofs

theorem mul_pred_le {a b : ℕ} (h : a ≤ b) : (a : ℝ) * (a - 1) ≤ b * (b - 1) := by
  rcases Nat.eq_zero_or_pos b with hb | hb
  · subst hb; simp [Nat.le_zero.1 h]
  · have h1 : (a : ℝ) ≤ b := by exact_mod_cast h
    have h2 : (1 : ℝ) ≤ b := by exact_mod_cast hb
    have h0 : (0 : ℝ) ≤ a := Nat.cast_nonneg a
    nlinarith

theorem PouwEndToEndPerRunPerCoin : Accounting.Guarantees.PouwEndToEndPerRunPerCoin := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ hΩ hWref hTT U hU A t εs hεs
  obtain ⟨hQ, _⟩ := id hTT
  let c := (1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout
  let E := fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
    AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
      CM.cost A.online x.1 U A.act (A.pre.eval x.1) x.2.1 < c ∧ ¬ Collides x.1 (finderQueries hQ A U x.1 x.2.1)
  let C := fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
    Collides x.1 (finderQueries hQ A U x.1 x.2.1)
  refine le_trans (pr_mono (P' := fun x => E x ∨ C x) fun x hx => ?_) ((pr_or_le E C).trans (add_le_add
    (Pouw.SecurityProofs.EndToEndPerCoin Q R S CM P D γ₀ Ωmax ε hQ hγ hΩ hWref hTT U hU A t εs hεs)
    ((pr_fst2_le fun y : (Q → R) × S => Collides y.1 (finderQueries hQ A U y.1 y.2)).trans
      (Pouw.Proofs.pr_finder_collides_le hQ A U))))
  by_cases hc : C x
  · exact Or.inr hc
  · exact Or.inl ⟨hx.1, hx.2, hc⟩

theorem ComputeOfPouwPerRunPerCoin : Accounting.Guarantees.ComputeOfPouwPerRunPerCoin := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ hΩ hWref hTT hmono U hU t εs hεs hne ι α ΩQ ΩS _ _ _ d r p real o
  obtain ⟨⟨_, _, tix⟩⟩ := hne
  have hA : CallsWithin (real.adv o) U (d.rate .oracle * (r.win .compute).len) := fun H s =>
    (real.calls_le (H, s, tix) o).trans (Schedule.ops_le (r.fits _) _ _ _)
  have hq : (real.adv o).queries U ≤ d.rate .oracle * (r.win .compute).len :=
    Finset.sup_le fun x _ => hA x.1 x.2
  refine le_trans (pr_mono fun x hx => ?_)
    ((PouwEndToEndPerRunPerCoin Q R S CM P D γ₀ Ωmax ε hγ hΩ hWref hTT U hU (real.adv o) t εs hεs).trans ?_)
  · exact ⟨(real.accepts_iff x o).1 hx.1, (real.cost_le x o).trans_lt hx.2⟩
  · refine add_le_add (add_le_add le_rfl (hmono _ hq)) (div_le_div_of_nonneg_right (mul_pred_le hq) (by positivity))

end SecurityProofs

end Accounting
