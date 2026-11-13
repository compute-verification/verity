module

public import Definitions.Pouw.PearlC.DeviceFp4Issue
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1

/-!
# Pearl-C's device records

The served record's prices (`Prices.sm120Loop`) and W1 setting (`W1Device.sm120Loop`, `W1Prices.sm120Loop`), the H100's
record (`devH100`, `Prices.h100`), `pearl-c-sm120` v2's (`devSm120v2`), the RTX PRO 6000's issue-bound prices
(`Prices.sm120`, `W1Prices.sm120`), Pearl-C4's issue-bound prices (`Fp4Prices.sm120Issue`), the other W1 settings
(`W1Device.h100`, `.sm120`, `.fp4Sm120`, the S8 settings `.s8Sm120` and `.s8H100`, and their fragment lists), and the
TT_OUT conjectures stated at them. No listed statement reads one: the listed forms are stated over every device record
(`Definitions.Pouw.Guarantees.PearlCDev`, and `PearlCSampledByDeadline` from them), and these are the numbers the lemmas at
device records (`Proofs.Pouw.Lemmas.PearlCRecords`, `Proofs.Pouw.Lemmas.Deadline`) and their proofs instantiate them at.
`benchmarks/pouw/tests/test_price_twins_accounting.py` holds the served prices to the RTX PRO 6000's card file.
Pearl-C4's record on sm_120 (`devFp4Sm120`, `Fp4Prices.sm120`) stays in `Definitions.Pouw.PearlC.DeviceFp4`, since
`Pouw.Guarantees.PearlC.TileCheck4WordsCore` reads it.
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

/-- **sm_120's W1 prices in the measured loop**, the served record's (node 2 at locked 2,100 MHz): an FP32 add
`1047/125 = 8.376` per word (3 loop instructions per 64), a BF16 MAC 2, the credited forming 40 per element (the k32 noise
atom, 32, and the E4M3 cast at its cheapest implementation, 8.0, at both FP32 prices) and the A-only forming 2 per element
(one FFMA on every 8th element at 8.376, 1.047, rounded up). `Costs` counts the forming in whole units, and rounding `qa`
up raises `W_ref` and not the credit, so g here is above g at the measured 1.047 (0.37349% against 0.36218% for v2 at
the bound 1/1,000, 8192³). The add is exact. The gap's price rule computes g at this record and at the issue-bound
`Prices.sm120`, and publishes the larger. -/
def Prices.sm120Loop : Prices := ⟨1047 / 125, 2, ⟨40, 2⟩⟩

end Pouw.PearlC

namespace Pouw.PearlC

/-- **sm_120's prices in FP4 units at the issue-bound FP32 price**: every FP32 operation at 8.00. -/
def Fp4Prices.sm120Issue : Fp4Prices := ⟨4, 5429 / 50, 18⟩

end Pouw.PearlC

namespace Pouw.PearlC.W1

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev

/-- **sm_120's W1 prices**, issue-bound (node 2, locked 2,100 MHz):
E4M3 1, BF16 2 and NVFP4 0.5 per MAC; FP32 8; a cast 16 per instruction (8 per code); INT32 16.

S8 is 1 per MAC, from the data sheet; those runs did not measure it. NVIDIA's RTX PRO Blackwell GPU Architecture
whitepaper v1.1, Appendix A, Table 4 (the RTX PRO 6000 Blackwell Workstation Edition, the same 188-SM GB202; the source
`benchmarks/census/hardware.json` cites for `rtx-pro-6000-server/e4m3`) gives "Peak INT8 Tensor TOPS" 1007.6 dense,
equal to "Peak FP8 Tensor TFLOPS with FP32 Accumulate" 1007.6. The Server Edition's product page and data sheet give no
INT8 figure. The census measured the wrapping form (`m16n8k32.row.col.s32.s8.s8.s32`) at
1,023.7 MACs per SM per clock at 16 warps, against E4M3's 1,023.8 (`f32.e4m3.e4m3.f32`, 16 warps); the `.satfinite`
form's rate is not measured. `Pouw.TileBound.price` has int8 at 1 from locked-clock runs. -/
def W1Prices.sm120 (sha fetch : ℚ) : W1Prices where
  mac := fun | .e4m3 => 1 | .bf16 => 2 | .e2m1 => 1 / 2 | .s8 => 1
  fp32 := 8
  cvt := 16
  int32 := 16
  sha512 := sha
  fetch := fetch

/-- sm_120's W1 prices with FP32 at the measured loop's `1047/125 = 8.376`. -/
def W1Prices.sm120Loop (sha fetch : ℚ) : W1Prices := { W1Prices.sm120 sha fetch with fp32 := 1047 / 125 }

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

/-- `devSm120v1 Prices.sm120Loop` and `devSm120v2 Prices.sm120Loop`'s setting. It omits the card's integer tensor
core (`W1Device.s8Sm120` has it), pending the restatement of its readers on an S8 setting. -/
def W1Device.sm120Loop (sha fetch : ℚ) : W1Device := W1Device.sm120At (W1Prices.sm120Loop sha fetch)

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
