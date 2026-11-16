module

public import Definitions.Pouw.PearlC.W1

/-!
# sm_120's W1 prices and Pearl-C4's W1 setting (trusted; definitions only)

sm_120's W1 prices, issue-bound (`W1Prices.sm120`) and in the measured loop (`W1Prices.sm120Loop`), the same in FP4 units
(`W1Prices.fp4Sm120`), and `devFp4Sm120`'s W1 setting (`W1Device.fp4Sm120`), which Pearl-C4's listed statements
(`Definitions.Pouw.Guarantees.PearlC4`) read. The other W1 settings at these prices are in `Proofs.Pouw.PearlC.Records`.
-/

@[expose] public section

namespace Pouw.PearlC.W1

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev

/-- **sm_120's W1 prices**, issue-bound (node 2, locked 2,100 MHz):
E4M3 1, BF16 2 and NVFP4 0.5 per MAC; FP32 8; a cast 16 per instruction (8 per code); INT32 16. FP32's 8 is the default
reading (the card file's `price.fp32.sm`): the add at its pipe's issue ceiling, both pipes on the one SM clock.

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

/-- sm_120's W1 prices with FP32 at the measured loop's `1047/125 = 8.376`, the second reading (the card file's
`price.fp32_loop`): an add in a loop of 64 adds and 3 loop instructions at a locked 2,100 MHz, which holds for an add
issued in that loop. A statement at these prices rests on that reading. -/
def W1Prices.sm120Loop (sha fetch : ℚ) : W1Prices := { W1Prices.sm120 sha fetch with fp32 := 1047 / 125 }

/-- **sm_120's W1 prices in FP4 units** (an FP8 unit is 2, `Fp4Prices.sm120`): `sm120Loop` doubled, with `sha` and
`fetch` given in FP4 units. -/
def W1Prices.fp4Sm120 (sha fetch : ℚ) : W1Prices :=
  { (W1Prices.sm120Loop 0 0).scale 2 with sha512 := sha, fetch := fetch }

/-- `devFp4Sm120`'s setting, in FP4 units. It omits the card's integer tensor core (`W1Device.s8Sm120` has it),
pending the restatement of its readers on an S8 setting. -/
def W1Device.fp4Sm120 (sha fetch : ℚ) : W1Device := W1Device.sm120At (W1Prices.fp4Sm120 sha fetch)

end Pouw.PearlC.W1
