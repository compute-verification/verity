import Mathlib
import Definitions.Pouw.Guarantees.RelativeSecurity
import Proofs.Pouw.RelativeSecurity
import Proofs.Pouw.RelativeSecurityFork

/-!
# Relative security's guarantees for the staged window: the proofs

`Pouw.SecurityProofs.RelativeSecurity.G : Pouw.Guarantees.RelativeSecurity.G` for each guarantee of
`Definitions.Pouw.Guarantees.RelativeSecurity`, from `Proofs.Pouw.RelativeSecurity`: Theorem S by the stages'
supermartingale at the luck ¾ (`RoundGame.expBound_of_staged`, `StagedPlay.game_staged`), and with overlapping stages
by the per-draw luck's `PassBound (3/4)` (`OvPlay.passBound_game`, `RoundGame.pathBound_of_passBound`). Each floor per
prover one adversary and one value of its coins at a time: for affine operands by `roundsSlice_of_staged` and
`roundsSlice_of_overlap` with the chain bound at factor 1 from Lemma G over F₂ (`chainBound_gProd`, `toF2`) at
`D = 160 ln 2`; for one-sided operands of any form by `roundsSlice_of_stagedP` and `roundsSlice_of_overlapP` (the
parity luck 427/500 in the product deficit, charging purity) at `toF2_wRe_zero` and `D = 274 ln 2`
(`ModelB.exp_parity_274`). Each slice's breaks are read at the adversary's fork finders and the slices averaged with
each capped at 1 (`serviceFloorPerProver_breakPr_of_slices`).
-/

namespace Pouw.SecurityProofs.RelativeSecurity

open Pouw.RelativeSecurity Pouw.Assumptions

theorem TheoremSStaged : Pouw.Guarantees.RelativeSecurity.TheoremSStaged := by
  intro C V A _ _ _ _ M R P hP D
  have hE := P.game.expBound_of_staged (by norm_num : (3 / 4 : ℝ) ≤ 1) (P.game_staged hP)
  have := P.game.prob_accepts_le (by norm_num : (0 : ℝ) ≤ 1 - 3 / 4) hE D
  convert this using 3
  ring

theorem TheoremSOverlap : Pouw.Guarantees.RelativeSecurity.TheoremSOverlap := by
  intro St κ ι _ _ _ A _ R P hP D
  have hE := RoundGame.expBound_of_pathBound _ (P.game.pathBound_of_passBound (P.passBound_game hP))
  have := P.game.prob_accepts_le (by norm_num : (0 : ℝ) ≤ 1 - 3 / 4) hE D
  convert this using 3
  ring

theorem StagedFloorKPerProver : Pouw.Guarantees.RelativeSecurity.StagedFloorKPerProver := by
  intro Mc S R St n L B n' e run sha kz δb βb δz βz κ Vmax hSt hκ hC _ _ _ _ hS hK hP
  have hF₀ : 0 < (St : ℝ) * (κ * L * ((n * n' * B : ℕ) : ℝ)) := by
    rw [mul_assoc]; exact mul_pos (by exact_mod_cast hSt) (mul_pos hκ (by exact_mod_cast hC))
  have h40 : Real.exp (-((1 - 3 / 4) * (160 * Real.log 2))) = ((2 : ℝ) ^ 40)⁻¹ := by
    rw [show (1 - 3 / 4 : ℝ) * (160 * Real.log 2) = ((40 : ℕ) : ℝ) * Real.log 2 by push_cast; ring, Real.exp_neg,
      Real.exp_nat_mul, Real.exp_log two_pos]
  have h := serviceFloorPerProver_breakPr_of_slices (p := 3 / 4) (sl := 0) (Vmax := Vmax) (δb := δb) (δz := δz) e sha
    (forkFinders S e sha 1) (forkFinders S e sha kz)
    (fun Ω' _ M hcom r => by
      obtain ⟨hB, hZ, hacc⟩ := hS Ω' M hcom r
      exact ⟨(run Ω' M r).Bind, (run Ω' M r).ZK, hB, hZ, roundsSlice_of_staged (wt := wRe) (Ok := fun π => AffProd π)
        (run Ω' M r).play _ _ _ _ (run Ω' M r).V hacc (fun x => (hP Ω' M hcom r x).1) (fun x =>
          ⟨(run Ω' M r).N x, (run Ω' M r).prog x, (run Ω' M r).pc x, (run Ω' M r).vq x, (run Ω' M r).word x,
            (hK Ω' M hcom r x).1, (hK Ω' M hcom r x).2.1, (hK Ω' M hcom r x).2.2, (hP Ω' M hcom r x).2⟩)
        ((chainBound_gProd toF2 rfl toF2_wRe_zero).mono fun _ hπ => hπ.gProd) (fun _ hπ => hπ) hSt hκ.le⟩)
    (by norm_num) hF₀ (160 * Real.log 2) h40.le
  rwa [mul_zero, sub_zero] at h

theorem OverlapFloorKPerProver : Pouw.Guarantees.RelativeSecurity.OverlapFloorKPerProver := by
  intro Mc S R St L n B n' e run sha kz δb βb δz βz κ Vmax hκ hC _ _ _ _ hS hK hP
  have hF₀ : 0 < κ * L * (Fintype.card (OCells St n n' B) : ℝ) := by
    rw [mul_assoc, card_oCells]; exact mul_pos hκ (by exact_mod_cast hC)
  have h40 : Real.exp (-((1 - 3 / 4) * (160 * Real.log 2))) = ((2 : ℝ) ^ 40)⁻¹ := by
    rw [show (1 - 3 / 4 : ℝ) * (160 * Real.log 2) = ((40 : ℕ) : ℝ) * Real.log 2 by push_cast; ring, Real.exp_neg,
      Real.exp_nat_mul, Real.exp_log two_pos]
  have h := serviceFloorPerProver_breakPr_of_slices (p := 3 / 4) (sl := 0) (Vmax := Vmax) (δb := δb) (δz := δz) e sha
    (forkFinders S e sha 1) (forkFinders S e sha kz)
    (fun Ω' _ M hcom r => by
      obtain ⟨hB, hZ, hacc⟩ := hS Ω' M hcom r
      exact ⟨(run Ω' M r).Bind, (run Ω' M r).ZK, hB, hZ, roundsSlice_of_overlap (wt := wRe)
        (run Ω' M r).play _ _ _ _ (run Ω' M r).V hacc (fun x => (hP Ω' M hcom r x).1) (fun x =>
          ⟨(run Ω' M r).N x, (run Ω' M r).prog x, (run Ω' M r).pc x, (run Ω' M r).vq x, (run Ω' M r).word x,
            (hK Ω' M hcom r x).1, (hK Ω' M hcom r x).2.1, (hK Ω' M hcom r x).2.2, (hP Ω' M hcom r x).2⟩)
        (fun _ => (chainBound_gProd toF2 rfl toF2_wRe_zero).mono fun _ hπ => hπ.gProd) hκ.le⟩)
    (by norm_num) hF₀ (160 * Real.log 2) h40.le
  rwa [mul_zero, sub_zero, card_oCells] at h

theorem StagedFloorKOneSidedPerProver : Pouw.Guarantees.RelativeSecurity.StagedFloorKOneSidedPerProver := by
  intro Mc S R St n L B n' e run sha kz δb βb δz βz κ Vmax hSt hκ hC _ _ _ _ hS hK hP
  have hF₀ : 0 < (St : ℝ) * (κ * L * ((n * n' * B : ℕ) : ℝ)) := by
    rw [mul_assoc]; exact mul_pos (by exact_mod_cast hSt) (mul_pos hκ (by exact_mod_cast hC))
  have h := serviceFloorPerProver_breakPr_of_slices (p := 427 / 500) (sl := 0) (Vmax := Vmax) (δb := δb) (δz := δz) e sha
    (forkFinders S e sha 1) (forkFinders S e sha kz) (fun Ω' _ M hcom r => by
      obtain ⟨hB, hZ, hacc⟩ := hS Ω' M hcom r
      exact ⟨(run Ω' M r).Bind, (run Ω' M r).ZK, hB, hZ, roundsSlice_of_stagedP (wt := wRe) (φ := toF2) (p₀ := 0)
        (run Ω' M r).play _ _ _ _ (run Ω' M r).V hacc (fun x => (hP Ω' M hcom r x).1) (fun x =>
          ⟨(run Ω' M r).N x, (run Ω' M r).prog x, (run Ω' M r).pc x, (run Ω' M r).vq x, (run Ω' M r).word x,
            (hK Ω' M hcom r x).1, (hK Ω' M hcom r x).2.1, (hK Ω' M hcom r x).2.2, (hP Ω' M hcom r x).2⟩)
        toF2_wRe_zero hSt hκ.le⟩)
    (by norm_num) hF₀ (274 * Real.log 2) ModelB.exp_parity_274
  rwa [mul_zero, sub_zero] at h

theorem OverlapFloorKOneSidedPerProver : Pouw.Guarantees.RelativeSecurity.OverlapFloorKOneSidedPerProver := by
  intro Mc S R St L n B n' e run sha kz δb βb δz βz κ Vmax hκ hC _ _ _ _ hS hK hP
  have hF₀ : 0 < κ * L * (Fintype.card (OCells St n n' B) : ℝ) := by
    rw [mul_assoc, card_oCells]; exact mul_pos hκ (by exact_mod_cast hC)
  have h := serviceFloorPerProver_breakPr_of_slices (p := 427 / 500) (sl := 0) (Vmax := Vmax) (δb := δb) (δz := δz) e sha
    (forkFinders S e sha 1) (forkFinders S e sha kz) (fun Ω' _ M hcom r => by
      obtain ⟨hB, hZ, hacc⟩ := hS Ω' M hcom r
      exact ⟨(run Ω' M r).Bind, (run Ω' M r).ZK, hB, hZ, roundsSlice_of_overlapP (wt := wRe) (φ := toF2) (p₀ := 0)
        (run Ω' M r).play _ _ _ _ (run Ω' M r).V hacc (fun x => (hP Ω' M hcom r x).1) (fun x =>
          ⟨(run Ω' M r).N x, (run Ω' M r).prog x, (run Ω' M r).pc x, (run Ω' M r).vq x, (run Ω' M r).word x,
            (hK Ω' M hcom r x).1, (hK Ω' M hcom r x).2.1, (hK Ω' M hcom r x).2.2, (hP Ω' M hcom r x).2⟩)
        toF2_wRe_zero hκ.le⟩)
    (by norm_num) hF₀ (274 * Real.log 2) ModelB.exp_parity_274
  rwa [mul_zero, sub_zero, card_oCells] at h

end Pouw.SecurityProofs.RelativeSecurity
