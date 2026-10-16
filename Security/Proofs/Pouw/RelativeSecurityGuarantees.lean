import Mathlib
import Models.Pouw.Guarantees.RelativeSecurity
import Proofs.Pouw.RelativeSecurity

/-!
# Relative security's guarantees for the staged window: the proofs

`Pouw.SecurityProofs.RelativeSecurity.G : Pouw.Guarantees.RelativeSecurity.G` for each guarantee of
`Models.Pouw.Guarantees.RelativeSecurity`, from `Proofs.Pouw.RelativeSecurity`: Theorem S by the stages'
supermartingale at the luck ¾ (`RoundGame.expBound_of_staged`, `StagedPlay.game_staged`), and the floor by
`roundsHyp_of_staged` with the chain bound at factor 1 from Lemma G over F₂ (`chainBound_gProd`, `toF2`) and
`serviceFloor_of_rounds` at `D = 160 ln 2`. With overlapping stages, Theorem S by the spot check's `PathBound (3/4)`
(`RoundGame.pathBound_spot`, `SpotPlay.game_spot`), and the floor by `roundsHyp_of_overlap` with the same chain
bound on every stage's shape.
-/

namespace Pouw.SecurityProofs.RelativeSecurity

open Pouw.RelativeSecurity Pouw.Assumptions

theorem TheoremSStaged : Pouw.Guarantees.RelativeSecurity.TheoremSStaged := by
  intro C V A _ _ _ _ M R P hP D
  have hE := P.game.expBound_of_staged (by norm_num : (3 / 4 : ℝ) ≤ 1) (P.game_staged hP)
  have := P.game.prob_accepts_le (by norm_num : (0 : ℝ) ≤ 1 - 3 / 4) hE D
  convert this using 3
  ring

theorem StagedFloorK : Pouw.Guarantees.RelativeSecurity.StagedFloorK := by
  intro Mc S R St n L B n' e run δb δz κ Vmax hSt hκ hC hδb hδz hS hK hP
  have hKSt : RoundsHypKSt Mc S R St Vmax δb δz κ n L 7 B n' wRe fun π => AffProd π := ⟨e, fun Ω' _ M hcom r => by
    obtain ⟨hB, hZ, hacc⟩ := hS Ω' M hcom r
    exact ⟨(run Ω' M r).play, (run Ω' M r).Bind, (run Ω' M r).ZK, (run Ω' M r).V, hB, hZ, hacc,
      fun x => (hP Ω' M hcom r x).1, fun x =>
        ⟨(run Ω' M r).N x, (run Ω' M r).prog x, (run Ω' M r).pc x, (run Ω' M r).vq x, (run Ω' M r).word x,
          (hK Ω' M hcom r x).1, (hK Ω' M hcom r x).2.1, (hK Ω' M hcom r x).2.2, (hP Ω' M hcom r x).2⟩⟩⟩
  have hF₀ : 0 < (St : ℝ) * (κ * L * ((n * n' * B : ℕ) : ℝ)) := by
    rw [mul_assoc]; exact mul_pos (by exact_mod_cast hSt) (mul_pos hκ (by exact_mod_cast hC))
  have h := serviceFloor_of_rounds (roundsHyp_of_staged hKSt
    ((chainBound_gProd toF2 rfl toF2_wRe_zero).mono fun _ hπ => hπ.gProd) (fun _ hπ => hπ) hSt hκ.le)
    (by norm_num : (3 / 4 : ℝ) ≤ 1) hF₀ hδb hδz (160 * Real.log 2)
  have h40 : Real.exp (-((1 - 3 / 4) * (160 * Real.log 2))) = ((2 : ℝ) ^ 40)⁻¹ := by
    rw [show (1 - 3 / 4 : ℝ) * (160 * Real.log 2) = ((40 : ℕ) : ℝ) * Real.log 2 by push_cast; ring, Real.exp_neg,
      Real.exp_nat_mul, Real.exp_log two_pos]
  rwa [h40, mul_zero, sub_zero] at h

theorem TheoremSSpot : Pouw.Guarantees.RelativeSecurity.TheoremSSpot := by
  intro C V A _ _ _ _ R P hP D
  have hE := RoundGame.expBound_of_pathBound _ (P.game.pathBound_spot (P.game_spot hP))
  have := P.game.prob_accepts_le (by norm_num : (0 : ℝ) ≤ 1 - 3 / 4) hE D
  convert this using 3
  ring

theorem OverlapFloorK : Pouw.Guarantees.RelativeSecurity.OverlapFloorK := by
  intro Mc S R St L n B n' e run δb δz κ Vmax hκ hC hδb hδz hS hK hP
  have hKOv : RoundsHypKOv Mc S R St Vmax δb δz κ n L 7 B n' wRe := ⟨e, fun Ω' _ M hcom r => by
    obtain ⟨hB, hZ, hacc⟩ := hS Ω' M hcom r
    exact ⟨(run Ω' M r).play, (run Ω' M r).Bind, (run Ω' M r).ZK, (run Ω' M r).V, hB, hZ, hacc,
      fun x => (hP Ω' M hcom r x).1, fun x =>
        ⟨(run Ω' M r).N x, (run Ω' M r).prog x, (run Ω' M r).pc x, (run Ω' M r).vq x, (run Ω' M r).word x,
          (hK Ω' M hcom r x).1, (hK Ω' M hcom r x).2.1, (hK Ω' M hcom r x).2.2, (hP Ω' M hcom r x).2⟩⟩⟩
  have hF₀ : 0 < κ * L * (Fintype.card (OCells St n n' B) : ℝ) := by
    rw [mul_assoc, card_oCells]; exact mul_pos hκ (by exact_mod_cast hC)
  have h := serviceFloor_of_rounds (roundsHyp_of_overlap hKOv
    (fun _ => (chainBound_gProd toF2 rfl toF2_wRe_zero).mono fun _ hπ => hπ.gProd) hκ.le)
    (by norm_num : (3 / 4 : ℝ) ≤ 1) hF₀ hδb hδz (160 * Real.log 2)
  have h40 : Real.exp (-((1 - 3 / 4) * (160 * Real.log 2))) = ((2 : ℝ) ^ 40)⁻¹ := by
    rw [show (1 - 3 / 4 : ℝ) * (160 * Real.log 2) = ((40 : ℕ) : ℝ) * Real.log 2 by push_cast; ring, Real.exp_neg,
      Real.exp_nat_mul, Real.exp_log two_pos]
  rwa [h40, mul_zero, sub_zero, card_oCells] at h

end Pouw.SecurityProofs.RelativeSecurity
