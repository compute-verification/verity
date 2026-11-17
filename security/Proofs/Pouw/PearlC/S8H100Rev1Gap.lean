module

public import Proofs.Pouw.PearlC.W1Best
public import Proofs.Pouw.PearlC.DeviceRev1Gap
public import Proofs.Pouw.PearlC.ServedH100Rev1Gap
public import Definitions.Pouw.Guarantees.PearlCS8

/-! The H100 headlines on the card's cost model, from the device model and the conjecture. -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

theorem ttOutTilePearlCH100Rev1_of_s8 {CM : CostModel Q R S} {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ}
    (hW1 : W1CompleteH100 salt sem ρ CM) (hTT : TTOutTilePearlCH100Rev1S8 salt sem ρ) :
    TTOutTilePearlCDevRev1 CM devH100Tensor sem ρ :=
  ttOutTile_of_dominatedWithin hW1.1 hW1.2 ((Pouw.PearlC.TTOutW1.ttOutTileW1_iff _ _ _ _ _ _ _).1 hTT)

theorem creditDevRev1_h100Tensor (m n k : ℕ) :
    creditDevRev1 devH100Tensor ⟨m, n, k⟩ =
      (m : ℚ) * k * n + m * k * n * (976 / 33) / 128 + m * k * 32 + 128 * m * n + 62 * m * k - 976 / 33 * m * n := by
  norm_num [creditDevRev1, firstAddDev, creditDev, devH100Tensor, devH100, Prices.h100Tensor, h100TensorMHz,
    h100SmMHz, Params.pi] <;> ring

theorem wrefDevRev1_h100Tensor (m n k : ℕ) :
    wrefDevRev1 devH100Tensor ⟨m, n, k⟩ = creditDevRev1 devH100Tensor ⟨m, n, k⟩ + 4 * m * k := by
  norm_num [wrefDevRev1, devH100Tensor, Prices.h100Tensor]

theorem wrefDevRev1_h100Tensor_nonneg (s : Shape) : 0 ≤ wrefDevRev1 devH100Tensor s := by
  obtain ⟨m, n, k⟩ := s
  rw [wrefDevRev1_h100Tensor, creditDevRev1_h100Tensor]
  have hm : (0 : ℚ) ≤ m := Nat.cast_nonneg m
  have hn : (0 : ℚ) ≤ n := Nat.cast_nonneg n
  have hk : (0 : ℚ) ≤ k := Nat.cast_nonneg k
  have h1 : 976 / 33 * (m : ℚ) * n ≤ 128 * m * n := by nlinarith [mul_nonneg hm hn]
  nlinarith [mul_nonneg (mul_nonneg hm hk) hn, mul_nonneg hm hk, mul_nonneg hm hn]

theorem wrefDevRev1_h100Tensor_llama32_1b_le :
    ∀ nk ∈ llama32_1bLinears, ∀ m : ℕ,
      wrefDevRev1 devH100Tensor ⟨m, nk.1, nk.2⟩ ≤
        1938800 / 1931027 * ((1 - 1 / 400) * creditDevRev1 devH100Tensor ⟨m, nk.1, nk.2⟩) := by
  intro nk hnk m
  have hm : (0 : ℚ) ≤ m := Nat.cast_nonneg m
  simp only [llama32_1bLinears, Finset.mem_insert, Finset.mem_singleton] at hnk
  rcases hnk with rfl | rfl | rfl | rfl <;>
    (dsimp only; rw [wrefDevRev1_h100Tensor, creditDevRev1_h100Tensor]; push_cast; nlinarith [hm])

theorem pearlCSampledH100TensorRev1_8192 (CM : CostModel Q R S) (sem : PearlCSem Q R S)
    (hTT : TTOutTilePearlCDevRev1 CM devH100Tensor sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 devH100Tensor sem (1 / 400)) (pearlCTilesDevRev1 devH100Tensor sem (1 / 400))
      (pearlCDomainDevAt devH100Tensor sh8192) ((216371 / 40211200 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1At CM devH100Tensor sem _ _ (301584 / 300713) _ (by norm_num)
    (wrefDevRev1_h100Tensor_nonneg _)
    (by rw [sh8192, wrefDevRev1_h100Tensor, creditDevRev1_h100Tensor]; norm_num) (by norm_num) hTT

theorem pearlCSampledH100TensorRev1_llama32_1b (CM : CostModel Q R S) (sem : PearlCSem Q R S)
    (hTT : TTOutTilePearlCDevRev1 CM devH100Tensor sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 devH100Tensor sem (1 / 400)) (pearlCTilesDevRev1 devH100Tensor sem (1 / 400))
      (pearlCDomainDevShapes devH100Tensor llama32_1bLinears) ((5040227 / 775520000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1Shapes CM devH100Tensor sem _ _ (1938800 / 1931027) _ (by norm_num)
    (fun _ _ _ => wrefDevRev1_h100Tensor_nonneg _) wrefDevRev1_h100Tensor_llama32_1b_le (by norm_num) hTT

theorem epsA_h100TensorRev1 :
    (216371 / 40211200 : ℚ) / (1 - 216371 / 40211200) = 216371 / 39994829 ∧
      (5040227 / 775520000 : ℚ) / (1 - 5040227 / 775520000) = 5040227 / 770479773 := by
  norm_num

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.SampledH100Rev1S8_8192 : Pouw.Guarantees.PearlC.SampledH100Rev1S8_8192 :=
  fun CM _ sem hW1 hTT =>
    Pouw.PearlC.pearlCSampledH100TensorRev1_8192 CM sem (Pouw.PearlC.ttOutTilePearlCH100Rev1_of_s8 hW1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledH100Rev1S8Llama32_1b :
    Pouw.Guarantees.PearlC.SampledH100Rev1S8Llama32_1b :=
  fun CM _ sem hW1 hTT =>
    Pouw.PearlC.pearlCSampledH100TensorRev1_llama32_1b CM sem (Pouw.PearlC.ttOutTilePearlCH100Rev1_of_s8 hW1 hTT)
