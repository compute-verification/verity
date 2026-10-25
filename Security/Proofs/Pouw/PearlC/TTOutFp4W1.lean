import Definitions.Pouw.Assumptions.PearlC.TTOutFp4
import Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1
import Proofs.Pouw.PearlC.TTOutW1

/-!
# TT_OUT-FP4 at W1 is TT_OUT-FP4 at the cost model `W1.w1Priced`

Each FP4 form at W1 (`Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1`) is the abstract form at
`CM := W1.w1Priced dv salt` (`ttOutFp4W1_iff`, …), since `w1Priced` meets `CostModel.Queries`
(`TTOutW1.w1Priced_queries`). The `_of_w1` lemmas give the forms the FP4 γ readers' reductions take, so each reader at
W1 is its reduction at that cost model.
-/

namespace Pouw.PearlC.TTOutFp4W1

open Pouw.PearlC.Assumptions Pouw.PearlC.TTOutW1

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

theorem ttOutFp4W1_iff (dv : W1.W1Device) (salt : S → List ℕ) (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) :
    TTOutFp4W1 dv salt d sem ρ ↔ TTOutFp4 (W1.w1Priced dv salt) d sem ρ :=
  ttOutW1_iff dv salt (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4 sem) (1 / 400) εPearlC

theorem ttOutTileFp4W1_iff (dv : W1.W1Device) (salt : S → List ℕ) (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) :
    TTOutTileFp4W1 dv salt d sem ρ ↔ TTOutTileFp4 (W1.w1Priced dv salt) d sem ρ :=
  ttOutTileW1_iff dv salt (pearlCProtocolFp4 d sem ρ) (pearlCTilesFp4 d sem ρ) (pearlCDomainFp4 sem) (1 / 400)
    εPearlC

theorem ttOutFp4Sm120W1_iff (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (ρ : ℚ) :
    TTOutFp4Sm120W1 sha fetch salt sem ρ ↔
      TTOutFp4Sm120 (W1.w1Priced (W1.W1Device.fp4Sm120 sha fetch) salt) sem ρ :=
  ttOutFp4W1_iff (W1.W1Device.fp4Sm120 sha fetch) salt devFp4Sm120 sem ρ

theorem ttOutTileFp4Sm120W1_iff (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (ρ : ℚ) :
    TTOutTileFp4Sm120W1 sha fetch salt sem ρ ↔
      TTOutTileFp4Sm120 (W1.w1Priced (W1.W1Device.fp4Sm120 sha fetch) salt) sem ρ :=
  ttOutTileFp4W1_iff (W1.W1Device.fp4Sm120 sha fetch) salt devFp4Sm120 sem ρ

theorem ttOutFp4_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : Fp4Device} {sem : Fp4Sem Q R S} {ρ : ℚ}
    (h : TTOutFp4W1 dv salt d sem ρ) : TTOutFp4 (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

theorem ttOutTileFp4_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : Fp4Device} {sem : Fp4Sem Q R S} {ρ : ℚ}
    (h : TTOutTileFp4W1 dv salt d sem ρ) : TTOutTileFp4 (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

theorem ttOutFp4Sm120_of_w1 {sha fetch : ℚ} {salt : S → List ℕ} {sem : Fp4Sem Q R S} {ρ : ℚ}
    (h : TTOutFp4Sm120W1 sha fetch salt sem ρ) :
    TTOutFp4Sm120 (W1.w1Priced (W1.W1Device.fp4Sm120 sha fetch) salt) sem ρ :=
  ⟨w1Priced_queries _ salt, h⟩

theorem ttOutTileFp4Sm120_of_w1 {sha fetch : ℚ} {salt : S → List ℕ} {sem : Fp4Sem Q R S} {ρ : ℚ}
    (h : TTOutTileFp4Sm120W1 sha fetch salt sem ρ) :
    TTOutTileFp4Sm120 (W1.w1Priced (W1.W1Device.fp4Sm120 sha fetch) salt) sem ρ :=
  ⟨w1Priced_queries _ salt, h⟩

end Pouw.PearlC.TTOutFp4W1
