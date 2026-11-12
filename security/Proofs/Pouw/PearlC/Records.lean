module

public import Definitions.Pouw.PearlC.DeviceFp4Issue
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1

/-!
# Pearl-C's device records other than the served one

The H100's record (`devH100`, `Prices.h100`), `pearl-c-sm120` v2's (`devSm120v2`), the RTX PRO 6000's issue-bound prices
(`Prices.sm120`), Pearl-C4's record on sm_120 (`devFp4Sm120`, `Fp4Prices.sm120`, `Fp4Prices.sm120Issue`), their W1
settings (`W1Device.h100`, `.sm120`, `.fp4Sm120`, the S8 settings `.s8Sm120` and `.s8H100`, and their fragment lists),
and the TT_OUT conjectures stated at them. No listed statement reads one: the listed forms are stated over every device
record (`Definitions.Pouw.Guarantees.PearlCDev`), and these are the numbers the lemmas at device records
(`Proofs.Pouw.Lemmas.PearlCRecords`) and their proofs instantiate them at. The served record, `devSm120v1` at
`Prices.sm120Loop` on `W1Device.sm120Loop`, stays in `Definitions.Pouw.PearlC`, its prices generated from the card file.
-/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-- The H100's prices (Rounds 10–11): an FP32 add costs 32, a BF16 MAC 2, and §5's forming `Costs.adopted`. -/
def Prices.h100 : Prices := ⟨32, 2, Costs.adopted⟩

/-- The H100 record: `ctChain` promoted every 4 atoms, `PeelExact`'s U, `Skip`'s flags and P3's `SaltDead`. -/
def devH100 : PearlCDevice where
  G := 4
  prices := Prices.h100
  ticket := fun A B => ctChain 4 A B
  useful := fun A B FA FB beA beB => PeelExact.peelExactVal A B FA FB beA beB
  flags := fun A B => Skip.unitFlags A B
  saltDead := SaltDead.saltDead

/-- **`pearl-c-sm120` v2's record**: the same E4M3 step `sm120E4m3` without promotion (`G = 0`), at prices `pr`. The
ticket is the pure chain over all of k (`AccumProofs.chainAt_zero_eq_pure`), and the inert-work flags are the pure chain's
lone skips (`SkipP.pureFlagsP`, `FreezePPure.unitFlagsAt_zero_iff`). -/
def devSm120v2 (pr : Prices) : PearlCDevice := devAt sm120E4m3 0 pr

/-- sm_120's W1 prices, issue-bound: an FP32 add 8, a BF16 MAC 2, credited forming 40 and A-only forming 1 per element. -/
def Prices.sm120 : Prices := ⟨8, 2, ⟨40, 1⟩⟩

end Pouw.PearlC

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev

/-- **sm_120's prices in FP4 units** (theory §2–§4, node 2 at locked 2,100 MHz), with an FP8 unit worth 2 FP4 units,
the block scale on `lut256` (§6.6) and FADD at the measured `1047/125`
= 8.376, as `Prices.sm120Loop`. It is `Fp4Prices.sm120At (1047/125) Fp4ScalePath.lut256`:
- a BF16 MAC is 4 (2.000 FP8 units);
- `fs = 105299/1000 = 105.299`: the noise atom 64 (a dense k64 NVFP4 `mma.sync`, 64 lanes, at full shape); the block
  scale on `lut256` at its W1 issue prices, `7.5 + FADD/8 = 8.547` (half an `F2FP`, 1.25 `PRMT` and one `LDS` per block
  of 16, and the FMUL `amax·1/6`); and the cast, FMUL plus the E2M1 cvt's F2FP alone, at `2·8.376 + 2·8.0 = 32.752`;
- `qa = 9423/500 = 18.846`: the noisy amax (one FMNMX, `2·8.376`) and ρ's squared accumulate on every 8th element
  (`2·8.376/8`), both reading the row alone.

The price rule publishes the larger g, and g rises with the FADD price, so the record is at the in-loop 8.376, not the
issue-bound 8.00 (`fs = 104.5`, `qa = 18`). `creditFp4` and `wrefFp4` read this one `fs`. It was `2444/25 = 97.76`, with
the block scale at 1 and FADD at 8.38. -/
def Fp4Prices.sm120 : Fp4Prices := ⟨4, 105299 / 1000, 9423 / 500⟩

/-- **`devFp4Sm120`**, Pearl-C4 on sm_120 (`BLACKWELL_SM120_NVF4`, `Fp4.sm120Nvf4`: 4 groups of 16, UE4M3 per 16, 27
fraction bits, a 36-bit window, round toward zero, floor −174): NVFP4's forming, `C̃ = cTilde`, U by `peelFp4`,
`Fp4Skip`'s lone-skip flags, the sub-grid flags and `saltDead4`, at `Fp4Prices.sm120`. -/
def devFp4Sm120 : Fp4Device where
  prices := Fp4Prices.sm120
  form := fun α β x e F => formRow α β x e F
  ticket := fun A B => ticket4 sm120Nvf4 A B
  useful := fun A B FA FB beA beB i j => (peelFp4 sm120Nvf4 A B FA FB beA beB i j).bind wordQ
  flags := fun A B i j t => match A i, B j with
    | some a, some b => Fp4Skip.fp4PureFlags sm120Nvf4 0 a.1 b.1 a.2 b.2 t
    | _, _ => false
  subgrid := fun A B => subgridFlags4 sm120Nvf4 A B
  saltDead := saltDead4

/-- **sm_120's prices in FP4 units at the issue-bound FP32 price**: every FP32 operation at 8.00. -/
def Fp4Prices.sm120Issue : Fp4Prices := ⟨4, 5429 / 50, 18⟩

end Pouw.PearlC

namespace Pouw.PearlC.W1

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev

/-- **The H100's W1 prices** (`Dimension.H100.h100Nominal`, the datasheet): E4M3 1, BF16 2 and S8 1 per MAC (the H100
has no E2M1 fragment, so that price is never read); FP32 32; a cast 64 per instruction (32 per code); INT32 64.

S8's 1 (`nominalMac .s8`) is the data sheets' ratio, the same on the H100 and the H200 (one GH100; they differ in memory
and its bandwidth, which W1 does not price). NVIDIA's H100 Tensor Core GPU Architecture whitepaper V1.04, Table 1, H100
SXM5: "Peak INT8 Tensor Core" 1978.9 TOPS dense, equal to "Peak FP8 Tensor Core" 1978.9 TFLOPS (BF16 989.4). NVIDIA's
H200 Tensor Core GPU data sheet (2024), H200 SXM: "INT8 Tensor Core" and "FP8 Tensor Core" 3,958 each, with sparsity
(BF16 1,979). -/
def W1Prices.h100 (sha fetch : ℚ) : W1Prices where
  mac := fun | .e4m3 => 1 | .bf16 => 2 | .e2m1 => 1 / 2 | .s8 => 1
  fp32 := 32
  cvt := 64
  int32 := 64
  sha512 := sha
  fetch := fetch

/-- **sm_120's W1 prices in FP4 units** (an FP8 unit is 2, `Fp4Prices.sm120`): `sm120Loop` doubled, with `sha` and
`fetch` given in FP4 units. -/
def W1Prices.fp4Sm120 (sha fetch : ℚ) : W1Prices :=
  { (W1Prices.sm120Loop 0 0).scale 2 with sha512 := sha, fetch := fetch }

/-- sm_120's fragments with INT8: `sm120Frags` and S8 m16n8k32 with `.satfinite` (IMMA.16832.S8.S8.SAT). -/
def sm120S8Frags : List Frag := sm120Frags ++ [⟨.s8, 16, 8⟩]

/-- The H100's `wgmma` fragments at its nominal per-MAC prices: E4M3 m64nNk32 and BF16 m64nNk16 for `N = 64, 72, …, 256`. -/
def h100Frags : List Frag := (List.range 25).flatMap fun i => [⟨.e4m3, 64, 64 + 8 * i⟩, ⟨.bf16, 64, 64 + 8 * i⟩]

/-- The H100 at prices `pr`: `hopperE4m3K32` and `hopperBf16K16`; no E2M1 fragment. -/
def W1Device.h100At (pr : W1Prices) : W1Device := ⟨pr, hopperE4m3K32, hopperBf16K16, sm120Nvf4, h100Frags⟩

/-- `devH100`'s setting (`Prices.h100`). It omits the card's integer tensor core (`W1Device.s8H100` has it), pending
the restatement of its readers on an S8 setting. -/
def W1Device.h100 (sha fetch : ℚ) : W1Device := W1Device.h100At (W1Prices.h100 sha fetch)

/-- `devSm120v1 Prices.sm120` and `devSm120v2 Prices.sm120`'s setting: both run the same E4M3 step (`G` is the
program's choice, not the machine's). It omits the card's integer tensor core (`W1Device.s8Sm120` has it), pending
the restatement of its readers on an S8 setting. -/
def W1Device.sm120 (sha fetch : ℚ) : W1Device := W1Device.sm120At (W1Prices.sm120 sha fetch)

/-- `devFp4Sm120`'s setting, in FP4 units. It omits the card's integer tensor core (`W1Device.s8Sm120` has it),
pending the restatement of its readers on an S8 setting. -/
def W1Device.fp4Sm120 (sha fetch : ℚ) : W1Device := W1Device.sm120At (W1Prices.fp4Sm120 sha fetch)

/-- **The RTX PRO 6000 (sm_120) with INT8** at prices `pr`: `sm120At pr`'s steps, and `sm120S8Frags`, its three
fragments and S8's. -/
def W1Device.s8Sm120At (pr : W1Prices) : W1Device := ⟨pr, sm120E4m3, hopperBf16K16, sm120Nvf4, sm120S8Frags⟩

/-- The INT8 setting, at `W1Prices.sm120`: S8 and E4M3 1, BF16 2 and NVFP4 0.5 per MAC, in E4M3 units. -/
def W1Device.s8Sm120 (sha fetch : ℚ) : W1Device := W1Device.s8Sm120At (W1Prices.sm120 sha fetch)

/-- The H100's fragments with INT8: `h100Frags` and S8 m64nNk32 with `.satfinite` (`IGMMA.64xNx32.S8.S8.SAT`) for
`N = 64, 80, …, 256`, PTX's dense integer `wgmma` shapes at `N ≥ 64`, as `h100Frags`. Each output is
`GEMM_SM90_S8`'s dot (`verity.ml.circuits.ops.gemm`, `BlackwellS8ImmaDot32`): k32, clamped once. Only N = 8 is pinned:
`verity.ml.circuits.silicon.instructions` pins `sm90.wgmma.m64n8k32.s8` on an H200. The
wider N here follow PTX's `wgmma` semantics, the same dot at each output, and no run has measured them. -/
def h100S8Frags : List Frag := h100Frags ++ (List.range 13).map fun i => ⟨.s8, 64, 64 + 16 * i⟩

/-- **The H100 or the H200 (sm_90) with INT8** at prices `pr`: `h100At pr`'s steps, and `h100S8Frags`. -/
def W1Device.s8H100At (pr : W1Prices) : W1Device := ⟨pr, hopperE4m3K32, hopperBf16K16, sm120Nvf4, h100S8Frags⟩

/-- The Hopper INT8 setting, at `W1Prices.h100`: S8 and E4M3 1, BF16 2 per MAC. -/
def W1Device.s8H100 (sha fetch : ℚ) : W1Device := W1Device.s8H100At (W1Prices.h100 sha fetch)

end Pouw.PearlC.W1

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **`tt-out/pearl-c-h100-rev1`**: rev1 at the H100 record, the repair of `TTOutPearlC · Costs.adopted`, with `CM` the
H100's W1 accounting, a `wgmma` priced by its whole shape. -/
def TTOutPearlCH100Rev1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutPearlCDevRev1 CM devH100 sem ρ

/-- **`tt-out-tile/pearl-c-h100-rev1`**: rev1 per audit tile at the H100 record, the repair of
`TTOutTilePearlC · Costs.adopted`. -/
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

/-- **The named assumption `tt-out/fp4-sm120`: TT_OUT(1/400) for Pearl-C4 on sm_120** (`pearl-c-nvfp4-v0`, theory §2
and §5): `TTOutFp4` at `devFp4Sm120`, with `CM` the sm_120 accounting in FP4 units. What it says and carries is
`TTOutFp4`'s. At W1 it is `TTOutFp4Sm120W1`. -/
def TTOutFp4Sm120 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : Fp4Sem Q R S)
    (ρ : ℚ) : Prop :=
  TTOutFp4 CM devFp4Sm120 sem ρ

/-- **`tt-out/fp4-sm120` per audit tile**: `TTOutTileFp4` at `devFp4Sm120`. It changes `TTOutFp4Sm120` as
`TTOutTilePearlC` changes `TTOutPearlC`: good tiles of any committed unit, credited rows per row, the bound per tile, and a
fixed credit of `(1 − ρ)·creditFp4` times the credited rows' share. -/
def TTOutTileFp4Sm120 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileFp4 CM devFp4Sm120 sem ρ

/-- **`tt-out/fp4-sm120` at W1**: TT_OUT(1/400) for Pearl-C4 at `devFp4Sm120` on `W1Device.fp4Sm120 sha fetch`. -/
def TTOutFp4Sm120W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutFp4W1 (W1.W1Device.fp4Sm120 sha fetch) salt devFp4Sm120 sem ρ

/-- **`tt-out/fp4-sm120` per audit tile at W1**: `TTOutTileFp4W1` at `devFp4Sm120` on `W1Device.fp4Sm120 sha fetch`. -/
def TTOutTileFp4Sm120W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileFp4W1 (W1.W1Device.fp4Sm120 sha fetch) salt devFp4Sm120 sem ρ

end Pouw.PearlC.Assumptions
