import Properties.Pouw.ServedFreed
import Proofs.Pouw.ServedWork
import Proofs.Pouw.OnsiteFreed

/-!
# Proof of `ServedFreed`

`Properties.Pouw.Served.ServedFreed` from `ServedWork` and `FreedOfFloor`:

* `outsourceServed_of_perProgram`: main's per-program `OutsourceWithinEgress` implies `OutsourceWithinEgressServed`.
  Each window program is replaced by its exchange-free one, which is a function of the program, so the replacement is
  fixed before the salt wherever the program is; its outputs are the program's, and its priced work is at most `β` a
  bit the program exchanged more.
* `servedFreed`: the outsourcing premise gives an exchange-free family of window programs with the same outputs, fixed
  before the salts, which `ServedWork` prices: off its event they spent at least `p · creditPrefix − Σ lam`, so the
  originals spent at least that less `β · ext`, and off the certifier's event `ext ≤ K`. `FreedOfFloor` turns the
  floor into freed compute, and the union bound adds the certifier's `δ`.
-/

namespace Properties.Pouw.Served.SecurityProofs

open Properties.Pouw.Served Properties.Pouw.PeriodJ10R
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The per-program price of an exchanged bit gives its period form at the served check**: each window program
replaced by its exchange-free one. -/
theorem outsourceServed_of_perProgram {C B I St : Type} (O : Type) {m n : ℕ} (hm : 0 < m) (G : Pouw.Dither.Accel)
    (μ : G.Res → ℝ) (β : ℝ) (inst : Paths (Paths C m × B) n → Fin n → I)
    (state : Paths (Paths C m × B) n → Fin n → St) (h : Pouw.Assumptions.OutsourceWithinEgress G μ β) :
    OutsourceWithinEgressServed O hm G μ β inst state := by
  intro g1 g2 p1 p2 work hFix
  choose F hF using fun f : G.Mc.Alg (St × C) O => h g1 g2 p1 p2 f
  refine ⟨fun x w => F (work x w), fun x z w => ?_, fun x w i => (hF _ i).2.1, fun x => ⟨fun w => (hF _ _).1, ?_⟩⟩
  · have e := hFix x z w
    simp only [Prod.mk.injEq] at e ⊢
    exact ⟨e.1, e.2.1, by rw [e.2.2]⟩
  · rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun w _ => (hF _ _).2.2

theorem servedFreed : ServedFreed := by
  intro ι K T L U C Ξ V I W St _ _ _ _ _ _ _ _ _ nB mB σ k hk sT unitOf Sc Cd n hn hN hr hM hKd ws opened enc place
    tile Wp Pl folded ρ δs δbind δks G μ Reg hon p lam κdeg B inst state work Pd cert β Kc δ hS hρ hρ1 hρN hwmax hlk
    hG hDD hPre hMoved hRBD hDraw hServ hSalt hLucks hUF g1 g2 p1 p2 hSAK hSt hReg hp hFix hRight hdual hC hβ hout
    himp
  obtain ⟨hμ, hneeds, hval, z', hfit', hslack⟩ := hdual
  obtain ⟨work', hFix', hext', hw'⟩ := hout g1 g2 p1 p2 work hFix
  have hRight' : FixedAreRight Pl (fun x w => Pouw.Assumptions.rightWords hon
      (fun t => (tileWords Sc Cd (tileCount σ sT) t : ℝ)) (inst x w) (saltCoin hr x w)
      (G.Mc.eval (work' x w) (state x w, saltCoin hr x w))) := fun x hx =>
    (hRight x hx).trans (le_of_eq (Finset.sum_congr rfl fun w _ => by simp only [(hw' x).1 w]))
  have hf := servedWork σ k hk sT unitOf Sc Cd hn hN hr hM hKd ws opened enc place tile Wp Pl folded ρ δs δbind δks
    G μ Reg hon p lam κdeg B inst state work' hS hρ hρ1 hρN hwmax hlk hG hDD hPre hMoved hRBD hDraw hServ hSalt
    hLucks hUF g1 g2 p1 p2 hSAK hSt hext' hReg hp hFix' hRight'
  have hi : prob (fun x => cert x ∧ Kc < ∑ w, G.ext (work x w) (state x w, saltCoin hr x w)) ≤ δ := himp
  refine le_trans (prob_mono fun x h => ?_) ((prob_or_le _ _).trans (add_le_add hf hi))
  obtain ⟨hacc, hcert, z, hfit, hnot⟩ := h
  by_cases hK : ∑ w, G.ext (work x w) (state x w, saltCoin hr x w) ≤ Kc
  · left
    refine ⟨hacc, not_le.mp fun hge => hnot ?_⟩
    have hc := (hw' x).2
    have hext : β * ∑ w, G.ext (work x w) (state x w, saltCoin hr x w) ≤ β * Kc := mul_le_mul_of_nonneg_left hK hβ
    have hX : ∑ r, μ r * (∑ w, G.use (work x w) (state x w, saltCoin hr x w) r) =
        ∑ w, G.cost μ (work x w) (state x w, saltCoin hr x w) := by
      simp only [Pouw.Dither.Accel.cost, Finset.mul_sum]
      exact Finset.sum_comm
    exact Pouw.SecurityProofs.Dither.FreedOfFloor G Pd 1 _ _ μ z z' hμ hneeds hval zero_le_one
      (div_nonneg (le_max_left _ _) hC.le) hfit' hslack hfit
      (Pouw.SecurityProofs.floor_of_max hC (by rw [hX]; linarith))
  · exact Or.inr ⟨hcert, not_le.mp hK⟩

end Properties.Pouw.Served.SecurityProofs
