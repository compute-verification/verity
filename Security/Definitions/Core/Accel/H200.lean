import Definitions.Core.Accel.Card

/-!
# The H200 SXM as an accelerator (trusted)

The H200 SXM (GH100, sm_90a) at its data sheet's numbers, a tick one SM clock at f_max = 1980 MHz (H100 whitepaper
V1.04 Table 3, H100 SXM5: 'GPU Boost Clock for FP64 Tensor Core Ops, FP32 and FP64 non Tensor Core Ops: 1980 MHz',
footnote 2 'GPU Peak Clock and GPU Boost Clock are synonymous'; the driver's maximum on vy-dc-h200,
note:20261017T0452Z-draft-eps1-dc; `Accounting.TS.Sheets.H200.fmaxMHz`). Every per-SM figure is over the 132 SMs
(whitepaper Table 3, H100 SXM5). These are data-sheet placeholders until CALIB (CMP-115) measures them.

**Stores** (words of 32 bits): the register file, 64 Ki registers an SM ('Register File Size / SM 256 KB'); shared
memory with L1, 256 KB an SM ('Combined L1 Cache / Shared Memory 256 KB'; shared memory alone is configurable to
228 KB, so the pool is the larger); L2, 50 MB ('L2 Cache Size 50 MB'), taken as 50 MiB; HBM3e, 141 GB (H200 data
sheet, nvidia.com/en-us/data-center/h200, 'GPU Memory 141GB'), taken as 141 GiB, above the 143,771 MiB the driver
reports.

**Rates** (units a tick):
* tensor: 4096 s8 MACs an SM clock (`Accounting.TS.Sheets.H200` A3: the data sheet's 1,979 dense INT8 TOPS over
  2 · 132 SMs at the 1830 MHz tensor boost clock), so 540,672; a MAC of `e4m3` or `e5m2` uses 1 unit (the same
  1,979 TFLOPS), of `f16` or `bf16` 2 (989.5), of `tf32` 4 (494.5); `b1` has no peak in any document (`Open.b1`);
* `mma.sync`: its tensor rate apart from `wgmma`'s, which is the data sheet's (no document gives `mma.sync`'s on
  Hopper, where it runs below `wgmma`), is `Open.sync` (`scale`) until CALIB measures it;
* issue: 4 warp instructions an SM clock (whitepaper, 'H100 SM Architecture': four partitions, each a warp scheduler
  dispatching one warp instruction a clock), so 528;
* `alu` and `imad`: 64 lanes an SM clock each (CUDA C++ Programming Guide 12.9 §6.4.1 Table 7, CC 9.0: 32-bit integer
  add, compare, min, max, logic and shift 64; multiply and multiply-add 64), so 8448 each;
* shared memory: 32 banks of one 32-bit word a clock an SM (Programming Guide, 'Shared Memory', CC 9.0), so 4224;
* HBM: 4.8 TB/s (H200 data sheet, 'GPU Memory Bandwidth 4.8TB/s'), 4.8 · 10¹² / (4 · 1.98 · 10⁹) = 20000/33 words.

**Opcodes** (PTX ISA 9.0 §9.7.14–9.7.15): `mma.sync` at `m16n8` with both operands in registers, and `wgmma` at `m64`,
`n` a multiple of 8 up to 256, `A` in registers or shared memory and `B` in shared memory; `k` as PTX gives it per
format (`ks`); `s8`/`u8` either way, `e4m3`/`e5m2` either way (wgmma only: sm_90's `mma.sync` FP8 is two `f16` HMMAs
and FADDs, not a tensor instruction of its own), the other formats with themselves. Every lane function, and the
integer ones on the uniform datapath too (`uni`, a whole issue slot each: CUDA Binary Utilities §4.3 lists Hopper's
uniform opcodes, integer only, and no document gives their issue cost; measured at one slot beside an async `wgmma`,
1.08–1.19 SMSP cycles, RT-H200-EMP, run r20261019-064443-0ae6, and 0.984 beside a saturated lane stream, the census's
UNI line, DC-90; `[uniform_issue]` of the sm_90 sheet); copies of at most 4 words (a lane's 128-bit load or store),
and bulk copies of any length with shared memory at one end (TMA); atomics on shared memory, L2 and HBM.

**Captured dots** (`captured`): the integer path exactly, `silicon.integer.HOPPER_SM90_S8_K32` (the clamp or wrap at
D; `wgmma` m64n64k32, m64n128k32 and m64n256k32 `.s8`, all eight forms, about 10.3–10.6 M elements each with the
extreme and clamp tiles, 0 mismatches, run r20261026-010336-7755, #4512; `mma.sync` m16n8k32 `.s8` pinned by DPRE,
HASH-8); `bf16` at `k = 16`, `TC.HOPPER_BF16_M16N8K16` (`mma.sync`
and `wgmma`, `models.HOPPER_BF16_M16N8K16`, `HOPPER_BF16_WGMMA_K16`); `e4m3` and `e5m2` at `k = 32`, one group of 32 on
a 14-bit adder, floor −139 (`total_fp8.HOPPER_E4M3_WGMMA_K32`, lane QZ, H100, 144,543,744 words, 0 mismatches;
`models.HOPPER_E5M2_K32`, run r20260922-183837-940b, 25,554,432 elements, 0 mismatches; LIT: one group of 32 at
width 14), validated at a zero accumulator only (`zeroOnly`: at any other accumulator the FP8 dot is `Open.dot`).
Every other dot is `Open.dot`. `wgmma` `.s8` at m64n64k32, m64n128k32 and m64n256k32 runs at the data sheet's rate
(`scale` 1) and computes `intDot`.

**Where the instance is coarser than the card** (each lets a program do more): the pools of `Card`; a `wgmma` charged
one issue slot (it is a warpgroup's four warp instructions); `IMAD.WIDE` one `imad` lane (it takes two); FP32 lane
functions charged an issue slot's share only; the shuffle pipe (32 lanes a clock) omitted, a cross-lane move being a
register copy here; uniform registers pooled with the register file.

**Not yet in the instance** (each a gap in faithfulness, not a relaxation): DMMA (`f64` MMA, 128 MACs an SM clock,
32 units each), 2:4-sparse MMA (`.sp`, the same products at the same rate with a free gather of `A`), `f16`
accumulators, `HFMA2`/`HMUL2`, `DFMA`, conversions (`I2F`, `F2I`, `F2FP`), `MUFU`, `FMNMX`, `mul.hi.s32`, `BMSK`,
constant-bank operands, `redux`, `vote`, TMA multicast, and floating-point atomics.
-/

open scoped NNRat

namespace Verity.Accel.H200

/-- The words of each store. -/
def size : Store → ℕ
  | .reg => 132 * 65536
  | .smem => 132 * 65536
  | .l2 => 50 * 2 ^ 20 / 4
  | .dram => 141 * 2 ^ 30 / 4

/-- Each resource's units a tick. -/
def ρ : Res → ℚ≥0
  | .tensor => 132 * 4096
  | .issue => 132 * 4
  | .alu => 132 * 64
  | .imad => 132 * 64
  | .smem => 132 * 32
  | .dram => 20000 / 33

/-- The tensor units a MAC of each format uses. -/
def weight : Fmt → ℚ≥0
  | .s8 | .u8 | .e4m3 | .e5m2 => 1
  | .f16 | .bf16 => 2
  | .tf32 => 4
  | _ => 0

/-- PTX's `k` for each format (`mma.sync` and `wgmma`). -/
def ks : Fmt → List ℕ
  | .s8 | .u8 => [16, 32]
  | .e4m3 | .e5m2 => [32]
  | .f16 | .bf16 => [8, 16]
  | .tf32 => [4, 8]
  | .b1 => [128, 256]
  | _ => []

/-- Whether a format is FP8. -/
def fp8 (f : Fmt) : Bool := f = .e4m3 || f = .e5m2

/-- The formats an MMA pairs. -/
def pair (fa fb : Fmt) : Bool := intPair fa fb || (fp8 fa && fp8 fb) || fa = fb

/-- The opcodes the H200 has. -/
def Avail : Op Store → Prop
  | .mma fa fb _ sh sa sb =>
    pair fa fb ∧ sh.k ∈ ks fa ∧
      ((sh.m = 16 ∧ sh.n = 8 ∧ !fp8 fa ∧ sa = .reg ∧ sb = .reg) ∨
        (sh.m = 64 ∧ 8 ∣ sh.n ∧ 8 ≤ sh.n ∧ sh.n ≤ 256 ∧ (sa = .reg ∨ sa = .smem) ∧ sb = .smem))
  | .lane _ => True
  | .uni g => g.pipe ≠ .fma
  | .copy s t n => n ≤ 4 ∨ (s = .smem ∧ t ≠ .reg) ∨ (t = .smem ∧ s ≠ .reg)
  | .atom s _ => s ≠ .reg

/-- Hopper's FP8 pipeline (`models.HOPPER_E4M3_K32`, `HOPPER_E5M2_K32`). -/
def fp8Pipeline : TC.Pipeline := { groups := [32], width := 14, zeroExponent := -139 }

/-- The dots the H200's captures pin. -/
def captured (fa fb : Fmt) (sat : Bool) (k : ℕ) : Option Dot :=
  match intCaptured fa fb sat with
  | some d => some d
  | none =>
    if fa = .bf16 ∧ fb = .bf16 ∧ k = 16 then some (floatDot .bf16 TC.HOPPER_BF16_M16N8K16)
    else if fa = .e4m3 ∧ fb = .e4m3 ∧ k = 32 then some (floatDot .e4m3 fp8Pipeline)
    else if fa = .e5m2 ∧ fb = .e5m2 ∧ k = 32 then some (floatDot .e5m2 fp8Pipeline)
    else none

/-- **The H200 SXM's table.** A program's code is at most HBM's bytes over 16-byte instructions; the coarse device's
rates are the tensor core's s8 MACs, the 132 · 128 FP32 lanes, and HBM's bits a tick rounded up
(4.8 · 10¹² · 8 / 1.98 · 10⁹ = 19,393.9). -/
def card : Card where
  size := size
  size_pos := by intro s; cases s <;> decide
  ρ := ρ
  weight := weight
  Avail := Avail
  scale x sh := if sh.m = 16 then x.sync else 1
  captured := captured
  zeroOnly fa fb _ k := fp8 fa && fp8 fb && k == 32
  text := 141 * 2 ^ 30 / 16
  tc := 132 * 4096
  cuda := 132 * 128
  bw := 19394

/-- **The H200 SXM** as an accelerator, its open parameters `x`. -/
def machine {ι : Type} (x : Open) (cap : ι → ℕ) (oracle : ℕ) : Machine ι := card.machine x cap oracle

end Verity.Accel.H200
