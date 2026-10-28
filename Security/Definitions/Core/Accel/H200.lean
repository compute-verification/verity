import Definitions.Core.Accel.Card

/-!
# The H200 SXM as an accelerator (trusted)

The H200 SXM (GH100, sm_90a) at its data sheet's numbers, a tick one SM clock at f_max = 1980 MHz (H100 whitepaper
V1.04 Table 3, H100 SXM5: 'GPU Boost Clock for FP64 Tensor Core Ops, FP32 and FP64 non Tensor Core Ops: 1980 MHz',
footnote 2 'GPU Peak Clock and GPU Boost Clock are synonymous'; the driver's maximum on vy-dc-h200;
`Accounting.TS.Sheets.H200.fmaxMHz`). Every per-SM figure is over the 132 SMs (whitepaper Table 3, H100 SXM5).
These are data-sheet placeholders until the card's calibration runs measure them.

**Stores** (words of 32 bits): the register file, 64 Ki registers an SM ('Register File Size / SM 256 KB'), plus the
uniform registers (63 a warp at 64 warps, 4032 words), the predicates (7 bits a thread at 2048 threads, 448 words) and
the uniform predicates (7 bits a warp, 14 words), 4494 words an SM on top; shared
memory with L1, 256 KB an SM ('Combined L1 Cache / Shared Memory 256 KB'; shared memory alone is configurable to
228 KB, so the pool is the larger); L2, 50 MB ('L2 Cache Size 50 MB'), taken as 50 MiB; HBM3e, 141 GB (H200 data
sheet, nvidia.com/en-us/data-center/h200, 'GPU Memory 141GB'), taken as 141 GiB, above the 143,771 MiB the driver
reports.

**Rates** (units a tick):
* tensor: 4096 s8 MACs an SM clock (`Accounting.TS.Sheets.H200` A3: the data sheet's 1,979 dense INT8 TOPS over
  2 · 132 SMs at the 1830 MHz tensor boost clock), so 540,672 a tick at the tick's clock (`tensor`, the per-clock
  peak); `tensorLong` is the same 540,672, the tensor core at the 1980 MHz boost clock (θ never caps a rate below
  what the card might reach). The sheet quotes its 1,979 TOPS at 1830 MHz
  (whitepaper V1.04 Table 3's boost clock for the FP8, FP16, BF16, TF32 and INT8 tensor figures, against 1980 MHz for
  FP64 tensor, FP32 and FP64), and the card measured 1,942.7 TOPS at about 1797 MHz under saturated `wgmma`.
  Neither shows that no operand class sustains a higher clock, so the long-window cap at
  132 · 4096 · 1830/1980 = 499,712 waits on a calibration run locked at 1980 MHz with zero, constant and
  uniform operands, and returns if none sustains above 1830 MHz, except for DMMA, whose
  67 TFLOPS the sheet quotes at 1980 MHz: at 499,712 and weight 32 it would get 15,616 MACs a tick against the
  sheet's 16,896, which `tensorLong_sheet` (`Proofs/Core/Accel/Sheet.lean`) refuses; a MAC of `e4m3` or `e5m2`
  uses 1 unit (the same 1,979 TFLOPS), of `f16` or `bf16` 2 (989.5), of `tf32` 4 (494.5); `b1` has no peak in any
  document (`Open.b1`);
* `mma.sync`: its tensor rate apart from `wgmma`'s, which is the data sheet's (no document gives `mma.sync`'s on
  Hopper, where it runs below `wgmma`), is `Open.sync` (`scale`) until a calibration run measures it;
* issue: 4 warp instructions an SM clock (whitepaper, 'H100 SM Architecture': four partitions, each a warp scheduler
  dispatching one warp instruction a clock), so 528;
* `alu` and `imad`: 64 lanes an SM clock each (CUDA C++ Programming Guide 12.9 §6.4.1 Table 7, CC 9.0: 32-bit integer
  add, compare, min, max, logic and shift 64; multiply and multiply-add 64), so 8448 each. `IMAD.WIDE`
  (`Lane.madWide`) uses 2 `imad` lanes (Table 7's footnote 6: 32 for extended precision; the lane-pipe probe
  hasn't measured it yet) and `IMAD.HI` `Open.madHi`. The two pools overlap in full in the model,
  which is generous: the card's don't (an IMAD-balanced SHA-512 took 58.1 SM-clocks a compression against
  the 41.2 full overlap predicts), and a joint ALU + IMAD measurement for sm_90 is pending;
* `f64`: 64 lanes an SM clock (Programming Guide Table 7, CC 9.0: 64-bit floating-point add, multiply, multiply-add
  64), so 8448: the FP64 cores, 64 an SM apart from the tensor cores (whitepaper Table 3, 'FP64 Cores / SM (excl.
  Tensor)'); DMMA (`Opcode.dmma`) runs on the tensor cores (A100 whitepaper: the third-generation Tensor Cores accelerate
  'all data types including FP16, BF16, TF32, FP64, INT8'), 128 MACs an SM clock at 1980 MHz (whitepaper: FP64 Tensor
  Core 67 TFLOPS over 132 SMs at 1980 MHz is 128.2), so a DMMA MAC uses 32 units of `tensor` and of `tensorLong`, as
  measured (DMMA at 1/32 of s8 a MAC beside a saturated `wgmma`, `benchmarks/pouw/sass_census/
  diff_menu.py` ESCAPE), and DMMA beside the low-precision formats shares their one budget;
* shared memory: 32 banks of one 32-bit word a clock an SM (Programming Guide, 'Shared Memory', CC 9.0), so 4224;
* HBM: 4.8 TB/s (H200 data sheet, 'GPU Memory Bandwidth 4.8TB/s'), 4.8 · 10¹² / (4 · 1.98 · 10⁹) = 20000/33 words;
* the L2's reads (`l2`): no document gives them and no calibration run has measured them on this card (θ's
  `l2.read` row is open), so the instance takes the lane loads' issue bound, 4 · 32 · 4 = 512 words an SM, 67,584
  (535 TB/s, 75 × sm_120's measured 7.10). Until the H200 memory probe measures it, a bound over the model gets
  `c = 0.25` here, not the card's 0.286.

**Opcodes** (PTX ISA 9.0 §9.7.14–9.7.15): `mma.sync` at `m16n8` with both operands in registers, and `wgmma` at `m64`,
`n` a multiple of 8 up to 256, `A` in registers or shared memory and `B` in shared memory; `k` as PTX gives it per
format (`ks`); `s8`/`u8` either way, `e4m3`/`e5m2` either way (wgmma only: sm_90's `mma.sync` FP8 is two `f16` HMMAs
and FADDs, not a tensor instruction of its own), the other formats with themselves. Every lane function, and the
integer ones on the uniform datapath too (`uni`, a whole issue slot each: CUDA Binary Utilities §4.3 lists Hopper's
uniform opcodes, integer only, and no document gives their issue cost; measured at one slot beside an async `wgmma`,
1.08–1.19 SMSP cycles, and 0.984 beside a saturated lane stream, the census's
UNI line, DC-90; `[uniform_issue]` of the sm_90 sheet); copies of at most 4 words (a lane's 128-bit load or store),
and bulk copies of any length with shared memory at one end (TMA); atomics on shared memory, L2 and HBM; integer
reductions (`Lane.reduces`) of up to 8 registers into memory (`red`), and of any length from shared memory into L2 or
HBM (`cp.reduce.async.bulk`), priced as the copy.

**Captured dots** (`captured`): the integer path exactly, `silicon.integer.HOPPER_SM90_S8_K32` (the clamp or wrap at
D; `wgmma` m64n64k32, m64n128k32 and m64n256k32 `.s8`, all eight forms, about 10.3–10.6 M elements each with the
extreme and clamp tiles, 0 mismatches; `mma.sync` m16n8k32 `.s8` pinned by its own capture); `bf16` at `k = 16`,
`TC.HOPPER_BF16_M16N8K16` (`mma.sync`
and `wgmma`, `models.HOPPER_BF16_M16N8K16`, `HOPPER_BF16_WGMMA_K16`); `e4m3` and `e5m2` at `k = 32`, one group of 32 on
a 14-bit adder, floor −139 (`total_fp8.HOPPER_E4M3_WGMMA_K32`, H100, 144,543,744 words, 0 mismatches;
`models.HOPPER_E5M2_K32`, 25,554,432 elements, 0 mismatches; LIT: one group of 32 at
width 14), validated at a zero accumulator only (`zeroOnly`: at any other accumulator the FP8 dot is `Open.dot`).
Every other dot is `Open.dot`. `wgmma` `.s8` at m64n64k32, m64n128k32 and m64n256k32 runs at the data sheet's rate
(`scale` 1) and computes `intDot`.

**Where the instance is coarser than the card** (each lets a program do more): the pools of `Card`; a `wgmma` charged
one issue slot (it is a warpgroup's four warp instructions); `IMAD.WIDE` one `imad` lane (it takes two); FP32 lane
functions charged an issue slot's share only; uniform registers and predicates in the register store. A cross-lane
move is `SHFL` on the shuffle pipe (32 lanes a SM-clock, its own pipe beside the `alu`) or a register copy on the
`alu`.

**2:4 sparse MMAs** (`mma.sp`, `wgmma.mma_async.sp`, PTX ISA 9.0 §9.7.14.6 and §9.7.15.6): at the dense forms'
shapes and twice their `k`, at the same tensor units a real product (so twice the logical rate: the census's
`IMMA.SP.16864` row), their dots `Open.dotSp`.

**Over-approximated** (`Accel.Isa`'s docstring: an exact integer version at the unit's rate): DMMA (`Opcode.dmma`, at
`m8n8k4` and `m16n8k4`, `k8`, `k16`, PTX ISA 9.0 §9.7.14.5.14), the FP64 lanes (`Lane.dfma` on the `f64` pipe),
`HFMA2`, `HADD2`, `HMUL2` (`Lane.hfma2`, one lane on the FP32 pipe, an issue slot's share: Table 7, CC 9.0, gives
16-bit FP multiply-add 256 results, two a lane), the float reductions (`Opcode.red` by `iadd3` and `add2`, `red.v8`
included), and `redux.sync` (`Lane.redux`, a uniform instruction's slot).

**Not yet in the instance** (each a gap in faithfulness, not a relaxation): `f16` accumulators, conversions (`I2F`,
`F2I`, `F2FP`), `MUFU`, `FMNMX`, `mul.hi.s32`, `BMSK`, constant-bank operands, `vote`, TMA multicast, the 64-bit
and `inc`/`dec` reductions, and `mma.sync` FP8 (`m16n8k32` `.e4m3`/`.e5m2`, PTX sm_89 and later; no capture of its dot).
-/

open scoped NNRat

namespace Verity.Accel.H200

/-- The words of each store. -/
def size : Store → ℕ
  | .reg => 132 * (65536 + 4494)
  | .smem => 132 * 65536
  | .l2 => 50 * 2 ^ 20 / 4
  | .dram => 141 * 2 ^ 30 / 4

/-- Each resource's units a tick. -/
def ρ : Res → ℚ≥0
  | .tensor => 132 * 4096
  | .tensorLong => 132 * 4096
  | .issue => 132 * 4
  | .alu => 132 * 64
  | .imad => 132 * 64
  | .shfl => 132 * 32
  | .f64 => 132 * 64
  | .smem => 132 * 32
  | .dram => 20000 / 33
  | .l2 => 132 * 512

/-- The tensor units a MAC of each format uses. -/
def weight : Fmt → ℚ≥0
  | .s8 | .u8 | .e4m3 | .e5m2 => 1
  | .f16 | .bf16 => 2
  | .tf32 => 4
  | .f64 => 32
  | _ => 0

/-- PTX's `k` for each format, `mma.sync`'s (PTX ISA 9.0 §9.7.14); `wgmma` takes one of them, `wgK`. -/
def ks : Fmt → List ℕ
  | .s8 | .u8 => [16, 32]
  | .e4m3 | .e5m2 => [32]
  | .f16 | .bf16 => [8, 16]
  | .tf32 => [4, 8]
  | .b1 => [128, 256]
  | _ => []

/-- `wgmma`'s one `k` for each format (PTX ISA 9.0 §9.7.15: 32 for `s8`, `u8` and FP8, 16 for `f16` and `bf16`, 8
for `tf32`, 256 for `b1`), so its `k16` integer and `k8` half forms don't exist: on the card the only
`s8` `k16` form is `mma.sync` `IMMA.16816`. -/
def wgK : Fmt → ℕ
  | .s8 | .u8 | .e4m3 | .e5m2 => 32
  | .f16 | .bf16 => 16
  | .tf32 => 8
  | .b1 => 256
  | _ => 0

/-- Whether a format is FP8. -/
def fp8 (f : Fmt) : Bool := f = .e4m3 || f = .e5m2

/-- The formats an MMA pairs. -/
def pair (fa fb : Fmt) : Bool := intPair fa fb || (fp8 fa && fp8 fb) || fa = fb

/-- The opcodes the H200 has. -/
def Avail : Opcode Store → Prop
  | .mma fa fb _ sh sa sb =>
    pair fa fb ∧ sh.k ∈ ks fa ∧
      ((sh.m = 16 ∧ sh.n = 8 ∧ !fp8 fa ∧ sa = .reg ∧ sb = .reg) ∨
        (sh.m = 64 ∧ 8 ∣ sh.n ∧ 8 ≤ sh.n ∧ sh.n ≤ 256 ∧ (sa = .reg ∨ sa = .smem) ∧ sb = .smem ∧ sh.k = wgK fa))
  | .mmaSp fa fb _ sh sa sb =>
    pair fa fb ∧ fa ≠ .b1 ∧ sh.k ∈ (ks fa).map (2 * ·) ∧
      ((sh.m = 16 ∧ sh.n = 8 ∧ !fp8 fa ∧ sa = .reg ∧ sb = .reg) ∨
        (sh.m = 64 ∧ 8 ∣ sh.n ∧ 8 ≤ sh.n ∧ sh.n ≤ 256 ∧ (sa = .reg ∨ sa = .smem) ∧ sb = .smem ∧ sh.k = 2 * wgK fa))
  | .lane g => g.warp = false
  | .uni g => g.uniform = true
  | .copy s t n => n ≤ 4 ∨ (s = .smem ∧ t ≠ .reg) ∨ (t = .smem ∧ s ≠ .reg)
  | .bcast s n k => s ≠ .reg ∧ n ≤ 4 ∧ k ≤ 32
  | .atom s _ => s ≠ .reg
  | .red s t n g => g.reduces ∧ ((s = .reg ∧ t ≠ .reg ∧ n ≤ 8) ∨ (s = .smem ∧ (t = .l2 ∨ t = .dram)))
  | .dmma sh => (sh.m = 8 ∧ sh.n = 8 ∧ sh.k = 4) ∨ (sh.m = 16 ∧ sh.n = 8 ∧ (sh.k = 4 ∨ sh.k = 8 ∨ sh.k = 16))

/-- Hopper's FP8 pipeline (`models.HOPPER_E4M3_K32`, `HOPPER_E5M2_K32`). -/
def fp8Pipeline : TC.Pipeline := { groups := [32], width := 14, zeroExponent := -139 }

/-- The dots the H200's captures pin. The integer ones (`intCaptured`) ignore `k` and the shape, so the `k16`
forms inherit the `k32` captures: a sum of 16 products is exact (at most 2^18 in magnitude), and the only place a
clamp lands is `C + dot`, which the `k32` captures' clamp tiles pin. -/
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
(4.8 · 10¹² · 8 / 1.98 · 10⁹ = 19,393.9). θ's floors: `sync ≥ 1` (the tensor pipe's peak is the data sheet's, which
`wgmma` reaches, so no MMA form beats it; `mma.sync` measured at 65.6% of `wgmma` on every native form and the
`k16` INT8 form at 48.7%, a card's `sync` of about 1.52 and 2.05, so the floor is the sheet's
and generous), `b1 ≥ 1/8` (as measured on the H200; the A100's binary peak is 8 × its INT8 peak),
`madHi ≥ 2` (`IMAD.HI` at 32 lanes a SM-clock against `IMAD`'s 64, CUDA C++ Programming Guide §6.4.1 Table 7,
CC 9.0, footnote 6; measured at 30.6), and `slack ≥ 1` (a backlog of at least a tick: an
MMA completes after the tick that issues it, and `Rate` and `CodeFetch` only get weaker as `slack` grows, so this floor
adds nothing to `Assumptions.CardWithinAccel`). -/
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
  floors x := 1 ≤ x.sync ∧ 1 / 8 ≤ x.b1 ∧ 2 ≤ x.madHi ∧ 1 ≤ x.slack

/-- **The H200 SXM** as an accelerator, its open parameters `x`. -/
def machine {ι : Type} (x : Open) (cap : ι → ℕ) (oracle : ℕ) : Machine ι := card.machine x cap oracle

end Verity.Accel.H200
