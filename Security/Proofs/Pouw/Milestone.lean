import Proofs.Pouw.Theorem1
import Proofs.Pouw.Exact
import Proofs.Pouw.M1

/-!
# (d′) §3.3's requirement at the milestone, in budget form
-/

namespace Pouw.Proofs

open Finset

variable {Q R S : Type}

theorem complete_mono (P : Protocol Q R S) {D D' : Layout → Prop} (hD : ∀ L, D L → D' L) (h : P.Complete D') :
    P.Complete D := fun U hU => h U ⟨hU.1, hD _ hU.2.1, hU.2.2⟩

theorem m1Shape_inDefault (sh : Shape) (h : D1.M1Shape sh) : sh.InDefaultDomain := by
  obtain ⟨h1, -, h3, h4, h5⟩ := h
  unfold Shape.InDefaultDomain
  rw [h4, h5]
  exact ⟨lt_of_lt_of_le (by norm_num) h1, h3, by norm_num, le_rfl, by norm_num, by norm_num, le_rfl, by norm_num⟩

theorem m1Domain_shapes (L : Layout) (h : D1.m1Domain L) : L.ShapesIn Shape.InDefaultDomain :=
  fun u hu => m1Shape_inDefault _ (h u hu).1

/-- Every D1 instance with noise in `[−63, 63]` is complete on the milestone domain. -/
theorem d1_complete_m1Domain (P : D1.Params Q R S) (hN : P.NoiseRange) : P.toProtocol.Complete D1.m1Domain :=
  complete_mono _ m1Domain_shapes (d1Complete Q R S P hN)

/-- **(d′)** `D1.m1` with `W_ref ≤ Ω_max·W_mm`, `Ω_max ≤ 199/198`, `AdmitsRef` and TT(0.5%, ε) meets §3.3's
requirement at `γ = 1 − (199/200)/Ω_max ≤ 1%`. -/
theorem milestoneRequirement : GameStatements.MilestoneRequirement := by
  intro CM Wref qref Ωmax ε hΩ hΩ1 hW hAdm hTT
  exact ⟨⟨(D1.m1 Wref).toProtocol, qref, d1_complete_m1Domain _ (M1.noiseRange Wref), hAdm,
    gammaFromTT D1.M1Q D1.M1R Salt CM (D1.m1 Wref).toProtocol D1.m1Domain Params.γ0 Ωmax ε γ0_lt_one hΩ hW hTT⟩,
    (gammaBudget Ωmax hΩ).2 hΩ1⟩

end Pouw.Proofs
