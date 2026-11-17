module

public import Definitions.Pouw.PearlC.DeviceFp4Issue
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4Sm120W1

/-! Pearl-C's device records, their W1 settings, and the TT_OUT conjectures stated at them. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-- **`pearl-c-sm120` v2's record**: the same E4M3 step `sm120E4m3` without promotion (`G = 0`), at prices `pr`. -/
def devSm120v2 (pr : Prices) : PearlCDevice := devAt sm120E4m3 0 pr

/-- sm_120's W1 prices, issue-bound: an FP32 add 8, a BF16 MAC 2, credited forming 40 and A-only forming 1 per element. -/
def Prices.sm120 : Prices := ⟨8, 2, ⟨40, 1⟩⟩

/-- **sm_120's W1 prices in the measured loop**: an FP32 add `1047/125 = 8.376` per word, a BF16 MAC 2, the credited
forming 40 and the A-only forming 2 per element. `Costs` counts the forming in whole units, and rounding `qa` up from
1.047 raises `W_ref` and not the credit, so g here is above g at 1.047. -/
def Prices.sm120Loop : Prices := ⟨1047 / 125, 2, ⟨40, 2⟩⟩

end Pouw.PearlC

namespace Pouw.PearlC

/-- **sm_120's prices in FP4 units at the issue-bound FP32 price**: every FP32 operation at 8.00. -/
def Fp4Prices.sm120Issue : Fp4Prices := ⟨4, 5429 / 50, 18⟩

end Pouw.PearlC

namespace Pouw.PearlC.W1

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev

/-- sm_120's fragments with INT8: `sm120Frags` and S8 m16n8k32 with `.satfinite` (IMMA.16832.S8.S8.SAT). -/
def sm120S8Frags : List Frag := sm120Frags ++ [⟨.s8, 16, 8⟩]

/-- The H100 at prices `pr`: `hopperE4m3K32` and `hopperBf16K16`; no E2M1 fragment. -/
def W1Device.h100At (pr : W1Prices) : W1Device := ⟨pr, hopperE4m3K32, hopperBf16K16, sm120Nvf4, h100Frags⟩

/-- `devH100`'s setting (`Prices.h100`), without the card's integer tensor core, which `W1Device.s8H100` has. -/
def W1Device.h100 (sha fetch : ℚ) : W1Device := W1Device.h100At (W1Prices.h100 sha fetch)

/-- `devSm120v1 Prices.sm120` and `devSm120v2 Prices.sm120`'s setting: both run the same E4M3 step (`G` is the
program's choice, not the machine's), without the card's integer tensor core, which `W1Device.s8Sm120` has. -/
def W1Device.sm120 (sha fetch : ℚ) : W1Device := W1Device.sm120At (W1Prices.sm120 sha fetch)

/-- `devSm120v1 Prices.sm120Loop` and `devSm120v2 Prices.sm120Loop`'s setting, without the card's integer tensor core,
which `W1Device.s8Sm120` has. -/
def W1Device.sm120Loop (sha fetch : ℚ) : W1Device := W1Device.sm120At (W1Prices.sm120Loop sha fetch)

/-- **The RTX PRO 6000 (sm_120) with INT8** at prices `pr`: `sm120At pr`'s steps, and `sm120S8Frags`, its three
fragments and S8's. -/
def W1Device.s8Sm120At (pr : W1Prices) : W1Device := ⟨pr, sm120E4m3, hopperBf16K16, sm120Nvf4, sm120S8Frags⟩

/-- The INT8 setting, at `W1Prices.sm120`: S8 and E4M3 1, BF16 2 and NVFP4 0.5 per MAC, in E4M3 units. -/
def W1Device.s8Sm120 (sha fetch : ℚ) : W1Device := W1Device.s8Sm120At (W1Prices.sm120 sha fetch)

/-- The Hopper INT8 setting, at `W1Prices.h100`: S8 and E4M3 1, BF16 2 per MAC. -/
def W1Device.s8H100 (sha fetch : ℚ) : W1Device := W1Device.s8H100At (W1Prices.h100 sha fetch)

end Pouw.PearlC.W1

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **`tt-out/pearl-c-h100-rev1`**: rev1 at the H100 record, with `CM` the H100's W1 accounting, a `wgmma` priced by
its whole shape. -/
def TTOutPearlCH100Rev1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutPearlCDevRev1 CM devH100 sem ρ

/-- **`tt-out-tile/pearl-c-h100-rev1`**: rev1 per audit tile at the H100 record. -/
def TTOutTilePearlCH100Rev1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTilePearlCDevRev1 CM devH100 sem ρ

/-- **`tt-out/pearl-c-h100-rev1` at W1**: rev1 at the H100 record on `W1Device.h100 sha fetch`. -/
def TTOutPearlCH100Rev1W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutPearlCDevRev1W1 (W1.W1Device.h100 sha fetch) salt devH100 sem ρ

/-- **`tt-out-tile/pearl-c-h100-rev1` at W1**: rev1 per audit tile at the H100 record on `W1Device.h100 sha fetch`. -/
def TTOutTilePearlCH100Rev1W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ)
    (salt : S → List ℕ) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTilePearlCDevRev1W1 (W1.W1Device.h100 sha fetch) salt devH100 sem ρ

/-- **The named assumption `tt-out/fp4-sm120`: TT_OUT(1/400) for Pearl-C4 on sm_120**: `TTOutFp4` at `devFp4Sm120`,
with `CM` the sm_120 accounting in FP4 units. At W1 it is `TTOutFp4Sm120W1`. -/
def TTOutFp4Sm120 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : Fp4Sem Q R S)
    (ρ : ℚ) : Prop :=
  TTOutFp4 CM devFp4Sm120 sem ρ

/-- **`tt-out/fp4-sm120` per audit tile**: `TTOutTileFp4` at `devFp4Sm120`. It changes `TTOutFp4Sm120` as
`TTOutTilePearlC` changes `TTOutPearlC`: good tiles of any committed unit, credited rows per row, the bound per tile, and a
fixed credit of `(1 − ρ)·creditFp4` times the credited rows' share. -/
def TTOutTileFp4Sm120 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileFp4 CM devFp4Sm120 sem ρ

end Pouw.PearlC.Assumptions
