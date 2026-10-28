import Definitions.Core.Accel.Card

/-!
# The RTX PRO 6000 Blackwell (sm_120) as an accelerator (trusted)

The RTX PRO 6000 Blackwell Server Edition (GB202, sm_120a) at its data sheet's numbers, a tick one SM clock at
f_max = 2430 MHz (the hardware's maximum, which the prover may set: SPEC's P1,
`benchmarks/pouw/machine_model/data/designs.toml`). Every per-SM figure is over 188 SMs (NVIDIA RTX PRO Blackwell GPU
Architecture whitepaper v1.1, Appendix A, Table 4, the same 188-SM GB202). These are data-sheet placeholders until
the card's calibration runs measure them.

**Stores** (32-bit words): the register file, 64 Ki registers an SM (whitepaper: a 256 KB register file an SM), plus
the uniform registers (63 a warp at 48 warps, 3024 words), the predicates (7 bits a thread at 1536 threads, 336 words)
and the uniform predicates (7 bits a warp, 11 words), 3371 words an SM on top;
shared memory with L1, 128 KB an SM (whitepaper: 128 KB of L1 and shared memory an SM); L2, 128 MB (whitepaper Table
4), taken as 128 MiB; GDDR7, 96 GB (product page, 'GPU Memory 96 GB GDDR7'), taken as 96 GiB.

**Rates** (units a tick):
* tensor: 1024 s8 MACs an SM clock (`IMMA.16832`, 4096 MACs in 16 tensor cycles of an SM sub-partition, the DS-120
  census; the whitepaper's 1,007.6 dense FP8 TFLOPS over 2 · 188 SMs at its
  2617 MHz boost is the same 1024), so 192,512; a MAC of `e4m3`, `e5m2` or `e2m1` (`kind::f8f6f4`, `QMMA.16832`) uses
  1 unit, of `nvf4` or `mxf4` (`OMMA.SF.16864`, 8192 MACs in 16 cycles; whitepaper FP4 2,015.2 dense) 1/2, of `f16` or
  `bf16` 2 (503.8), of `tf32` 4 (251.9); sm_120 has no native `b1` MMA (below); `tensorLong` is `tensor`, since the
  card has one clock for its tensor and lane figures;
* the `k16` integer `mma.sync` (`IMMA.16816`) runs at half the `k32` rate on the card (measured: 512 against 1024 MACs
  an SM clock); the instance prices it at the `k32` rate, which lets a program do more, and a per-shape θ row would
  tighten it;
* issue: 4 warp instructions an SM clock (four sub-partitions, one warp scheduler each), so 752;
* `alu` and `imad`: 64 lanes an SM clock each (the sm_120 census: `LOP3` and `IMAD` two cycles a warp instruction on a
  sub-partition; CUDA C++ Programming Guide §6.4.1 Table 7), so 12,032 each; `IMAD.WIDE` uses 2 `imad` lanes (Table 7's
  footnote 6, 32 for extended precision), and the pools overlap in full, generous as on the H200;
* `f64`: 2 lanes an SM clock (Programming Guide Table 7, CC 12.0: 64-bit floating-point multiply-add 2), so 376; DMMA
  (`Opcode.dmma`; PTX gives `mma` `.f64` to sm_80 and later) has no documented rate on this card, and it measured
  1.75 MACs an SM clock (`DMMA.8x8x4`): 32 tensor
  units a MAC, 32 MACs an SM clock, 18 × that, stays as a generous figure, since a measured rate bounds the card's
  from below only;
* shared memory: 32 banks of a 32-bit word a clock an SM (Programming Guide, 'Shared Memory'), so 6016;
* GDDR7: 1597 GB/s (product page, 'Memory Bandwidth 1597 GB/s'), 1597 · 10⁹ / (4 · 2.43 · 10⁹) = 39925/243 words;
* the L2's reads (`l2`): no document gives them; the census read at most 7.10 TB/s, 18.05 bytes an
  SM-clock at 2092 MHz (ROW_L2RD `l2_g256cg_line_q16_w16`). The instance takes 20 bytes an
  SM-clock, 5 words, so 940 (9.14 TB/s at the tick): 11% above the measurement even if the L2 kept pace with the SM
  clock. This is a measured cap, like `shfl`'s: the claim takes no L2 pattern to read faster.

**Opcodes** (PTX ISA 9.0 §9.7.14): `mma.sync` at `m16n8`, both operands in registers (sm_120 has no `wgmma`), `k` per
format (`ks`), no `b1` form (ptxas emulates every one, at 99–436 SMSP-cycles as measured, so it
is a program of the modelled instructions); `s8`/`u8` either way, `e4m3`, `e5m2` and `e2m1` in any pairing
(`kind::f8f6f4`), the other formats with themselves. Every lane function, and the integer ones on the uniform
datapath too (`uni`, a whole issue slot each: CUDA Binary Utilities §4.4 lists the uniform opcodes CC 10.0 to 12.0
share, and no document gives their issue cost; one slot as measured on sm_100,
on sm_90, and on this card, which measured 3.06 uniform warp instructions an SM clock against issue's 4;
whether sm_120's uniform datapath has functions beyond the
integer ones is a gap); copies of at most 4 words, and bulk copies with shared memory at one end (TMA); atomics on
shared memory, L2 and GDDR7; integer reductions as on the H200.

**Captured dots** (`captured`): the integer path exactly (`models.BLACKWELL_SM120_S8_M16N8K32`, `_WRAP`,
`_U8S8_M16N8K32`); `bf16` at `k = 16`, `TC.HOPPER_BF16_M16N8K16` (its docstring: `mma.sync.m16n8k16.bf16` on Hopper and
on sm_120); `e4m3` and `e5m2` with themselves at `k = 32`, one group of 32 and the accumulator on a 26-bit truncating
adder, floor −133 (`f8Pipeline`; `models.BLACKWELL_SM120_E4M3_M16N8K32`: `QMMA.16832.F32.E4M3.E4M3`, fitted on
6,434,816 elements, then two fresh-seed captures under total semantics, specials included, with 25,040,128 and
25,041,024 elements, each 0 mismatches;
`_E5M2_M16N8K32`: 6,434,816 elements, then 25,041,024, 0 mismatches; the
families include random nonzero accumulators over 2^±20, cancellation and near-overflow accumulators, so the dot is
pinned at every accumulator; the Pearl-C model's 99.96–99.997% compared the atom with the exact sum's FP32
truncation, not with this pipeline). `e2m1` with itself at `k = 32` is `Open.dot`: its two captures
(`_E2M1_M16N8K32`, 0 mismatches on finite accumulators) left NaN and
infinite accumulators record-only. The mixed FP8 pairs and the block-scaled FP4 dots
(`models.BLACKWELL_SM120_NVF4`, `_MXF4`, captured in Python, a block-scaled adder this module doesn't transcribe) and
every other dot are `Open.dot`.

**2:4 sparse MMAs** (`mma.sp`): at `m16n8` and twice the dense `k`, at the same tensor units a real product (measured:
`IMMA.SP` at 2,048 logical MACs an SM clock, twice the dense rate), their dots `Open.dotSp`; the block-scaled sparse forms (`kind::mxf4nvf4` at `k = 128`) are a gap.

**Coarser than the card**, and **not yet in the instance**: as for the H200 (`Accel.H200`), with FP6 (`e3m2`, `e2m3`)
and the block-scaled FP8 and FP6 forms (`kind::mxf8f6f4`) among the gaps. DMMA, the FP64 lanes, `HFMA2`, the float
reductions and `redux.sync` are over-approximated as on the H200 (`HFMA2`'s sm_120 rate is undocumented; the model
gives it 256 FP16 FMAs an SM clock, two a lane on the 128 FP32 lanes, and the census measured 125.9).
-/

open scoped NNRat

namespace Verity.Accel.Sm120

/-- The words of each store. -/
def size : Store → ℕ
  | .reg => 188 * (65536 + 3371)
  | .smem => 188 * 32768
  | .l2 => 128 * 2 ^ 20 / 4
  | .dram => 96 * 2 ^ 30 / 4

/-- Each resource's units a tick. -/
def ρ : Res → ℚ≥0
  | .tensor => 188 * 1024
  | .tensorLong => 188 * 1024
  | .issue => 188 * 4
  | .alu => 188 * 64
  | .imad => 188 * 64
  | .shfl => 188 * 32
  | .f64 => 188 * 2
  | .smem => 188 * 32
  | .dram => 39925 / 243
  | .l2 => 188 * 5

/-- The tensor units a MAC of each format uses. -/
def weight : Fmt → ℚ≥0
  | .s8 | .u8 | .e4m3 | .e5m2 | .e2m1 => 1
  | .nvf4 | .mxf4 => 1 / 2
  | .f16 | .bf16 => 2
  | .tf32 => 4
  | .b1 => 0
  | .f64 => 32

/-- PTX's `k` for each format at `m16n8`, none for `b1`: the card has no native `b1` MMA. -/
def ks : Fmt → List ℕ
  | .s8 | .u8 => [16, 32]
  | .e4m3 | .e5m2 | .e2m1 => [32]
  | .f16 | .bf16 => [8, 16]
  | .tf32 => [4, 8]
  | .nvf4 | .mxf4 => [64]
  | .b1 | .f64 => []

/-- Whether a format is one of `kind::f8f6f4`'s. -/
def f8f6f4 (f : Fmt) : Bool := f = .e4m3 || f = .e5m2 || f = .e2m1

/-- The formats an MMA pairs. -/
def pair (fa fb : Fmt) : Bool := intPair fa fb || (f8f6f4 fa && f8f6f4 fb) || fa = fb

/-- The opcodes the RTX PRO 6000 has. -/
def Avail : Opcode Store → Prop
  | .mma fa fb _ sh sa sb => pair fa fb ∧ sh.k ∈ ks fa ∧ sh.m = 16 ∧ sh.n = 8 ∧ sa = .reg ∧ sb = .reg
  | .mmaSp fa fb _ sh sa sb =>
    pair fa fb ∧ fa ≠ .nvf4 ∧ fa ≠ .mxf4 ∧ sh.k ∈ (ks fa).map (2 * ·) ∧ sh.m = 16 ∧ sh.n = 8 ∧ sa = .reg ∧ sb = .reg
  | .lane g => g.warp = false
  | .uni g => g.uniform = true
  | .copy s t n => n ≤ 4 ∨ (s = .smem ∧ t ≠ .reg) ∨ (t = .smem ∧ s ≠ .reg)
  | .bcast s n k => s ≠ .reg ∧ n ≤ 4 ∧ k ≤ 32
  | .atom s _ => s ≠ .reg
  | .red s t n g => g.reduces ∧ ((s = .reg ∧ t ≠ .reg ∧ n ≤ 8) ∨ (s = .smem ∧ (t = .l2 ∨ t = .dram)))
  | .dmma sh => (sh.m = 8 ∧ sh.n = 8 ∧ sh.k = 4) ∨ (sh.m = 16 ∧ sh.n = 8 ∧ (sh.k = 4 ∨ sh.k = 8 ∨ sh.k = 16))

/-- sm_120's FP8 and `kind::f8f6f4` E2M1 atom (`models.BLACKWELL_SM120_E4M3_M16N8K32`, `_E5M2_`, `_E2M1_`): pinned for
E4M3 and E5M2 (`captured`), E2M1's candidate. -/
def f8Pipeline : TC.Pipeline := { groups := [32], width := 26, zeroExponent := -133 }

/-- The dots the RTX PRO 6000's captures pin. The integer ones (`intCaptured`) ignore `k` and the shape, so the `k16`
forms inherit the `k32` captures: a sum of 16 products is exact (at most 2^18 in magnitude), and the only place a
clamp lands is `C + dot`, which the `k32` captures' clamp tiles pin. -/
def captured (fa fb : Fmt) (sat : Bool) (k : ℕ) : Option Dot :=
  match intCaptured fa fb sat with
  | some d => some d
  | none =>
    if fa = .bf16 ∧ fb = .bf16 ∧ k = 16 then some (floatDot .bf16 TC.HOPPER_BF16_M16N8K16)
    else if fa = .e4m3 ∧ fb = .e4m3 ∧ k = 32 then some (floatDot .e4m3 f8Pipeline)
    else if fa = .e5m2 ∧ fb = .e5m2 ∧ k = 32 then some (floatDot .e5m2 f8Pipeline)
    else none

/-- **The RTX PRO 6000's table.** A program's code is at most GDDR7's bytes over 16-byte instructions; the coarse
device's rates are the tensor core's s8 MACs, the 188 · 128 FP32 lanes, and GDDR7's bits a tick rounded up
(1597 · 10⁹ · 8 / 2.43 · 10⁹ = 5,257.6). θ's floor: `madHi ≥ 2` (`IMAD.HI` at 32 lanes a
SM-clock against `IMAD`'s 64, CUDA C++ Programming Guide §6.4.1 Table 7; not yet measured on this card: `SHA_ROT`
suggests about 16 lanes, which would make it 4, and the floor rises to the card's measurement once taken). `b1` needs
no floor: no `b1` MMA is available. -/
def card : Card where
  size := size
  size_pos := by intro s; cases s <;> decide
  ρ := ρ
  weight := weight
  Avail := Avail
  scale _ _ := 1
  captured := captured
  zeroOnly _ _ _ _ := false
  text := 96 * 2 ^ 30 / 16
  tc := 188 * 1024
  cuda := 188 * 128
  bw := 5258
  floors x := 2 ≤ x.madHi

/-- **The RTX PRO 6000 (sm_120)** as an accelerator, its open parameters `x`. -/
def machine {ι : Type} (x : Open) (cap : ι → ℕ) (oracle : ℕ) : Machine ι := card.machine x cap oracle

end Verity.Accel.Sm120
