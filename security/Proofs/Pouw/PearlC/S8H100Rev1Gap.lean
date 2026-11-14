module

public import Proofs.Pouw.PearlC.W1Best
public import Proofs.Pouw.PearlC.DeviceRev1Gap
public import Proofs.Pouw.PearlC.ServedH100Rev1Gap
public import Definitions.Pouw.Guarantees.PearlCS8

/-!
# The H100 headlines on the card's cost model, from the device model and the conjecture

`ttOutTilePearlCH100Rev1_of_s8` is TT_OUT per audit tile at the card's cost model `CM` on `devH100Tensor`, the H100
record per second at the tensor clock, from its two named parts: W1's completeness for the card's credited work on the
H100 with INT8 (`W1CompleteH100`, a domination of `CM` by W1 on `W1Device.s8H100Best` within every bound on rev1's
audit tiles) and TT_OUT there (`TTOutTilePearlCH100Rev1S8`), by `Pouw.PearlC.ttOutTile_of_dominatedWithin`.

The gaps at `devH100Tensor`, `g = 1 − (399/400)²·credit/W_ref` per unit as at `devH100` (`ServedH100Rev1Gap`), with the
FP32 add at `976/33` and the forming `fs = 62`, `qa = 4`:

| shape | `W_ref/credit` | g | ε_a = g/(1 − g) | at `devH100` |
| --- | --- | --- | --- | --- |
| 8192³ | 56547/56525 | 216371/40211200 (0.53809%) | 216371/39994829 (0.54100%) | 0.53751% |
| qkv (3072, 2048) | 22151/22129 | 0.59820% | 0.60180% | 0.59683% |
| o (2048, 2048) | 14947/14925 | 0.64583% | 0.65002% | 0.64379% |
| gate_up (16384, 2048) | 115803/115781 | 0.51828% | 0.52098% | 0.51802% |
| down (2048, 8192) | 14541/14519 | 5040227/775520000 (0.64992%) | 5040227/770479773 (0.65417%) | 0.64766% |

Llama 3.2 1B's served layout takes the down projection's, the max of the four. The guarantees of
`Definitions.Pouw.Guarantees.PearlCS8` are the 8192³ gap (`pearlCSampledH100TensorRev1_8192`) and the served layout's
(`pearlCSampledH100TensorRev1_llama32_1b`) at that `CM`.
-/

@[expose] public section

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **TT_OUT per tile on the card's cost model** at the H100 record per second at the tensor clock, from W1's
completeness for its credited work on the H100 with INT8 and TT_OUT there. -/
theorem ttOutTilePearlCH100Rev1_of_s8 {CM : CostModel Q R S} {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ}
    (hW1 : W1CompleteH100 salt sem ρ CM) (hTT : TTOutTilePearlCH100Rev1S8 salt sem ρ) :
    TTOutTilePearlCDevRev1 CM devH100Tensor sem ρ :=
  ttOutTile_of_dominatedWithin hW1.1 hW1.2 ((Pouw.PearlC.TTOutW1.ttOutTileW1_iff _ _ _ _ _ _ _).1 hTT)

/-! ## The gaps at `devH100Tensor` -/

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

/-- Llama 3.2 1B's largest ratio `wrefDevRev1/creditDevRev1` at `devH100Tensor` is its down projection's,
`14541/14519`: `ω = (14541/14519)·(400/399) = 1938800/1931027` bounds every linear's `W_ref` against `(1 − 1/400)` of
its credit. -/
theorem wrefDevRev1_h100Tensor_llama32_1b_le :
    ∀ nk ∈ llama32_1bLinears, ∀ m : ℕ,
      wrefDevRev1 devH100Tensor ⟨m, nk.1, nk.2⟩ ≤
        1938800 / 1931027 * ((1 - 1 / 400) * creditDevRev1 devH100Tensor ⟨m, nk.1, nk.2⟩) := by
  intro nk hnk m
  have hm : (0 : ℚ) ≤ m := Nat.cast_nonneg m
  simp only [llama32_1bLinears, Finset.mem_insert, Finset.mem_singleton] at hnk
  rcases hnk with rfl | rfl | rfl | rfl <;>
    (dsimp only; rw [wrefDevRev1_h100Tensor, creditDevRev1_h100Tensor]; push_cast; nlinarith [hm])

/-- **The H100 headline per second at the tensor clock, under rev1 per audit tile, 8192³**:
`g = 216371/40211200` (0.53809%). -/
theorem pearlCSampledH100TensorRev1_8192 (CM : CostModel Q R S) (sem : PearlCSem Q R S)
    (hTT : TTOutTilePearlCDevRev1 CM devH100Tensor sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 devH100Tensor sem (1 / 400)) (pearlCTilesDevRev1 devH100Tensor sem (1 / 400))
      (pearlCDomainDevAt devH100Tensor sh8192) ((216371 / 40211200 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1At CM devH100Tensor sem _ _ (301584 / 300713) _ (by norm_num)
    (wrefDevRev1_h100Tensor_nonneg _)
    (by rw [sh8192, wrefDevRev1_h100Tensor, creditDevRev1_h100Tensor]; norm_num) (by norm_num) hTT

/-- **The H100 record per second at the tensor clock, under rev1 per audit tile, on Llama 3.2 1B's served layout**:
`g = 5040227/775520000` (0.64992%), the down projection's, the max of the four. -/
theorem pearlCSampledH100TensorRev1_llama32_1b (CM : CostModel Q R S) (sem : PearlCSem Q R S)
    (hTT : TTOutTilePearlCDevRev1 CM devH100Tensor sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 devH100Tensor sem (1 / 400)) (pearlCTilesDevRev1 devH100Tensor sem (1 / 400))
      (pearlCDomainDevShapes devH100Tensor llama32_1bLinears) ((5040227 / 775520000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1Shapes CM devH100Tensor sem _ _ (1938800 / 1931027) _ (by norm_num)
    (fun _ _ _ => wrefDevRev1_h100Tensor_nonneg _) wrefDevRev1_h100Tensor_llama32_1b_le (by norm_num) hTT

/-- **ε_a = g/(1 − g) at `devH100Tensor`**, 8192³ and Llama 3.2 1B's served layout: 0.54100% and 0.65417%. -/
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
