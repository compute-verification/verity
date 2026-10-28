import Definitions.Accounting.TS.Sheet

/-!
# The B200's data sheet (generated)

The data sheet (`Accounting.TS.Sheet`) at the HGX B200 (sm_100a, 180 GB, 1000 W)'s vendor values: each constant with its
label (A1–A6), its value, its grade and its source. Each tensor peak has a lemma that the per-clock peak reproduces the
document's per-second figure at the clock that figure is quoted at. A3's MACs are device totals over the SMSPs, so a
warpgroup's `wgmma` or a single-thread `tcgen05.mma` counts its MACs once, spread over the SM's four SMSPs.

Scope: the sheet is one GPU's own SMs. Work a peer GPU on the node does over NVLink or NVSwitch and sends back (whole
tiles as 64-byte digests) is outside it: the sheet bounds what the credited GPU's SMs compute, not which GPU computed a
tile, and `Device.lean`'s side channels don't list NVLink peer offload.

`uv run machine-model --device sm_100 sheet --lean` prints this file from
`benchmarks/pouw/machine_model/data/devices/sm_100/sheet.toml`, and the machine model's tests hold it to that output.
-/

namespace Accounting.TS.Sheets.B200

noncomputable section

/-- The classes of the `DS-100` census (its kinds). -/
inductive Kind
  | ALU
  | ATOME32
  | ATOMS32
  | DMMA16816
  | DMMA1684
  | DMMA1688
  | DMMA884
  | FFMA
  | IMAD
  | IMMA16816_S8
  | IMMA16832_S8
  | LDS128
  | LDTM_16X128B_X8
  | LDTM_16X256B_X4
  | LDTM_16X64B_X16
  | LDTM_32X32B_X1
  | LDTM_32X32B_X128
  | LDTM_32X32B_X16
  | LDTM_32X32B_X32
  | LDTM_32X32B_X4
  | LDTM_32X32B_X64
  | NOROW
  | REDG32
  | REDUX
  | SHFL
  | STG128
  | STG32
  | STS128
  | STS32
  | UNI
  | UNLISTED
  | UTCHMMA_BF16_M128_N256
  | UTCHMMA_F16_M128_N256
  | UTCHMMA_TF32_M128_N256
  | UTCIMMA_M128_N128
  | UTCIMMA_M128_N256
  | UTCIMMA_M128_N256_SP
  | UTCIMMA_M128_N32
  | UTCIMMA_M128_N64
  | UTCIMMA_M64_N128
  | UTCIMMA_M64_N256
  | UTCIMMA_M64_N64
  | UTCOMMA_MXF4NVF4_M128_N256
  | UTCOMMA_MXF4_M128_N256
  | UTCQMMA_E2M1_M128_N256
  | UTCQMMA_E2M3_M128_N256
  | UTCQMMA_E3M2_M128_N256
  | UTCQMMA_E4M3_M128_N128
  | UTCQMMA_E4M3_M128_N256
  | UTCQMMA_E5M2_M128_N256
  | UTCQMMA_MXF8F6F4_E4M3_M128_N256
  | VIADD
  deriving DecidableEq

instance instFintypeKind : Fintype Kind :=
  Fintype.ofList [.ALU, .ATOME32, .ATOMS32, .DMMA16816, .DMMA1684, .DMMA1688, .DMMA884, .FFMA, .IMAD, .IMMA16816_S8,
    .IMMA16832_S8, .LDS128, .LDTM_16X128B_X8, .LDTM_16X256B_X4, .LDTM_16X64B_X16, .LDTM_32X32B_X1, .LDTM_32X32B_X128,
    .LDTM_32X32B_X16, .LDTM_32X32B_X32, .LDTM_32X32B_X4, .LDTM_32X32B_X64, .NOROW, .REDG32, .REDUX, .SHFL, .STG128,
    .STG32, .STS128, .STS32, .UNI, .UNLISTED, .UTCHMMA_BF16_M128_N256, .UTCHMMA_F16_M128_N256, .UTCHMMA_TF32_M128_N256,
    .UTCIMMA_M128_N128, .UTCIMMA_M128_N256, .UTCIMMA_M128_N256_SP, .UTCIMMA_M128_N32, .UTCIMMA_M128_N64,
    .UTCIMMA_M64_N128, .UTCIMMA_M64_N256, .UTCIMMA_M64_N64, .UTCOMMA_MXF4NVF4_M128_N256, .UTCOMMA_MXF4_M128_N256,
    .UTCQMMA_E2M1_M128_N256, .UTCQMMA_E2M3_M128_N256, .UTCQMMA_E3M2_M128_N256, .UTCQMMA_E4M3_M128_N128,
    .UTCQMMA_E4M3_M128_N256, .UTCQMMA_E5M2_M128_N256, .UTCQMMA_MXF8F6F4_E4M3_M128_N256, .VIADD]
    (by intro k; cases k <;> decide)

/-- The Nsight Compute pipes the `DS-100` classes issue to. -/
inductive Pipe
  | alu
  | imad
  | shfl
  | uniform
  deriving DecidableEq

instance instFintypePipe : Fintype Pipe :=
  Fintype.ofList [.alu, .imad, .shfl, .uniform] (by intro p; cases p <;> decide)

/-- The tensor operand types with a dense peak. -/
inductive Ty
  | s8
  | e4m3
  | e5m2
  | fp6
  | f16
  | bf16
  | tf32
  | fp4
  | fp4_f8f6f4
  | f64
  deriving DecidableEq

/-- A1, f_max: 1965 MHz. Grade: measured only (the driver's maximum). Source: no NVIDIA document read states a B200
clock (Blackwell Architecture Technical Brief V2.1, the Lenovo HGX B200 Product Guide, nvidia.com's HGX page); the
driver's maximum at default clocks on the deployed B200 is 1965 MHz (sm_100/table.toml, EPS1 DC), and the tensor and
clock probes saw at most 1957.1 MHz; the brief's 'FP64 Tensor Core 37 teraFLOPS' at the Guide's 64 FP64 results an SM
clock over 148 SMs puts the FP64 clock at 1927 to 1980 MHz. -/
def fmaxMHz : ℝ := 1965

/-- The SMs A3's per-SM peaks multiply: 148. Grade: measured only. Source: no NVIDIA document read gives the B200's SM
count (Technical Brief V2.1 Table 3 says only '16 Blackwell Die for 8 GPUs'); the device reports 148
(sm_100/table.toml). -/
def smCount : ℝ := 148

/-- The implied clock (no document states one), at which the s8, e4m3, e5m2, fp6, f16, bf16, tf32, fp4, fp4_f8f6f4
figures are quoted: 1854 MHz. Grade: none (derived: no document states it). Source: the clock at which every per-clock
peak below reproduces the Technical Brief's dense figure to within half its last digit, which holds from 1835 to 1855
MHz, and the Lenovo guide's FP16 2.25 too from 1852 MHz; 111 MHz under f_max, as the H200's tensor clock is 150 MHz
under its f_max (open premise B200-PEAK-PER-CLOCK: the per-clock peaks are measured, the clock they are quoted at is
nowhere). -/
def impliedMHz : ℝ := 1854

/-- The FP64 clock (no document states one), at which the f64 figure is quoted: 1965 MHz. Grade: none (derived: no
document states it). Source: f_max, the driver's maximum: the brief's 'FP64 Tensor Core 37 teraFLOPS' at the measured 64
MACs an SM clock over 148 SMs holds from 1927 to 1980 MHz, which the implied clock (1854 MHz, 35.1 teraFLOPS) misses;
the H200's FP64 tensor clock is its f_max too (H100 whitepaper V1.04 Table 3). -/
def fp64MHz : ℝ := 1965

/-- A3, s8: 8192 dense MACs an SM clock, the document's dense figure, 4500 × 10¹² a second, over 2 · 148 SMs at the
implied clock (no document states one), 1854 MHz (A1's f_max is 1965 MHz). Grade: data sheet per second; per clock
measured only. Source: NVIDIA Blackwell Architecture Technical Brief V2.1, Table 3 'System Specifications for HGX B300
and HGX B200', Per GPU Specs, HGX B200: 'INT8 Tensor Core 4.5 | 9 petaOPS' (Dense | Sparse); Lenovo ThinkSystem NVIDIA
HGX B200 180GB 1000W GPU Product Guide Table 2: '4.5 / 9 petaOPS'; measured UTCIMMA_M128_N256 512 SMSP-cycles (DC-100).
-/
def peakS8 : ℝ := 8192

/-- A3, e4m3: 8192 dense MACs an SM clock, the document's dense figure, 4500 × 10¹² a second, over 2 · 148 SMs at the
implied clock (no document states one), 1854 MHz (A1's f_max is 1965 MHz). Grade: data sheet per second; per clock
measured only. Source: Technical Brief V2.1 Table 3, HGX B200: 'FP8/FP6 Tensor Core 4.5 | 9 petaFLOPS'; measured
UTCQMMA_E4M3 lines (DC-100). -/
def peakE4m3 : ℝ := 8192

/-- A3, e5m2: 8192 dense MACs an SM clock, the document's dense figure, 4500 × 10¹² a second, over 2 · 148 SMs at the
implied clock (no document states one), 1854 MHz (A1's f_max is 1965 MHz). Grade: data sheet per second; per clock
derived at the implied clock. Source: Technical Brief V2.1 Table 3, HGX B200: 'FP8/FP6 Tensor Core 4.5 | 9 petaFLOPS',
for both FP8 formats (PTX ISA 9.4 tcgen05.mma .kind::f8f6f4: e4m3, e5m2). -/
def peakE5m2 : ℝ := 8192

/-- A3, fp6: 8192 dense MACs an SM clock, the document's dense figure, 4500 × 10¹² a second, over 2 · 148 SMs at the
implied clock (no document states one), 1854 MHz (A1's f_max is 1965 MHz). Grade: data sheet per second; per clock
derived at the implied clock; its classes unmeasured. Source: Technical Brief V2.1 Table 3, HGX B200: 'FP8/FP6 Tensor
Core 4.5 | 9 petaFLOPS' (PTX ISA 9.4 .kind::f8f6f4: e2m3, e3m2); the CUDA C++ Programming Guide 12.9 §18.9.1 lists no
FP6. -/
def peakFp6 : ℝ := 8192

/-- A3, f16: 4096 dense MACs an SM clock, the document's dense figure, 2200 × 10¹² a second, over 2 · 148 SMs at the
implied clock (no document states one), 1854 MHz (A1's f_max is 1965 MHz). Grade: data sheet per second; per clock
derived at the implied clock. Source: Technical Brief V2.1 Table 3, HGX B200: 'FP16/BF16 Tensor Core 2.2 | 4.5
petaFLOPS'; Lenovo Product Guide Table 2: '2.25 / 4.5'; measured UTCHMMA_F16_M128_N256 (DC-100). -/
def peakF16 : ℝ := 4096

/-- A3, bf16: 4096 dense MACs an SM clock, the document's dense figure, 2200 × 10¹² a second, over 2 · 148 SMs at the
implied clock (no document states one), 1854 MHz (A1's f_max is 1965 MHz). Grade: data sheet per second; per clock
derived at the implied clock. Source: Technical Brief V2.1 Table 3, HGX B200: 'FP16/BF16 Tensor Core 2.2 | 4.5
petaFLOPS'. -/
def peakBf16 : ℝ := 4096

/-- A3, tf32: 2048 dense MACs an SM clock, the document's dense figure, 1100 × 10¹² a second, over 2 · 148 SMs at the
implied clock (no document states one), 1854 MHz (A1's f_max is 1965 MHz). Grade: data sheet per second; per clock
derived at the implied clock. Source: Technical Brief V2.1 Table 3, HGX B200: 'TF32 Tensor Core 1.1 | 2.2 petaFLOPS';
Lenovo Product Guide Table 2: '1.1 / 2.2'. -/
def peakTf32 : ℝ := 2048

/-- A3, fp4: 16384 dense MACs an SM clock, the document's dense figure, 9000 × 10¹² a second, over 2 · 148 SMs at the
implied clock (no document states one), 1854 MHz (A1's f_max is 1965 MHz). Grade: data sheet per second; per clock
measured. Source: Technical Brief V2.1 Table 3, HGX B200: 'FP4 Tensor Core 9 | 18 petaFLOPS'; PTX ISA 9.4 tcgen05.mma
.kind::mxf4 and .kind::mxf4nvf4 take twice .kind::f8f6f4's K; the Programming Guide 12.9 §18.9.1 lists no FP4, and
DC-100's census measures no FP4 class; a probe timed kind::mxf4 (UTCOMMA) and kind::mxf4nvf4 (UTCOMMA.4X) alone at 16390
MACs an SM clock, 2.000 times kind::i8's 8195 at the same instruction rate (unpriced_fp4.cu). -/
def peakFp4 : ℝ := 16384

/-- A3, fp4_f8f6f4: 8192 dense MACs an SM clock, the document's dense figure, 4500 × 10¹² a second, over 2 · 148 SMs at
the implied clock (no document states one), 1854 MHz (A1's f_max is 1965 MHz). Grade: vendor documentation per kind
(CUTLASS), at the data sheet's FP8 figure; per clock measured. Source: NVIDIA CUTLASS, 'Blackwell SM100 GEMMs'
(media/docs/cpp/blackwell_functionality.md at 1b5b7d3062): 'tcgen05.mma(.sp).cta_group::[1|2].kind::f8f6f4 | 2x Hopper
Fp8 Tensor Core | Mixed precision MMA with A={f4,f6,f8} x B={f4,f6,f8}', against 'kind::mxf4.block_scale | 4x Hopper Fp8
Tensor Core'; so the kind's rate is the FP8 one, Technical Brief V2.1 Table 3, HGX B200: 'FP8/FP6 Tensor Core 4.5 | 9
petaFLOPS'; measured alone, kind::f8f6f4 with E2M1 operands at 8195 MACs an SM clock, E4M3's rate (unpriced_fp4.cu). On
sm_120, all five kind::f8f6f4 types run at one QMMA rate. -/
def peakFp4_f8f6f4 : ℝ := 8192

/-- A3, f64: 64 dense MACs an SM clock, the document's dense figure, 37 × 10¹² a second, over 2 · 148 SMs at the FP64
clock (no document states one), 1965 MHz (A1's f_max is 1965 MHz). Grade: data sheet per second; per clock measured.
Source: Technical Brief V2.1 Table 3, HGX B200: 'FP64 Tensor Core 37 teraFLOPS'; the Programming Guide 12.9 §6.4.1 Table
7, CC 10.0: 64 64-bit floating-point results a clock a multiprocessor; measured alone, DMMA.8x8x4 at 64 MACs an SM
clock, and beside a saturated tcgen05 kind::i8 form t + s = 0.939 to 0.952, so f64 shares the tensor pipe's time with
the other types (unpriced.cu). -/
def peakF64 : ℝ := 64

/-- A3: each type's dense MACs an SM clock. -/
def Ty.peak : Ty → ℝ
  | .s8 => peakS8
  | .e4m3 => peakE4m3
  | .e5m2 => peakE5m2
  | .fp6 => peakFp6
  | .f16 => peakF16
  | .bf16 => peakBf16
  | .tf32 => peakTf32
  | .fp4 => peakFp4
  | .fp4_f8f6f4 => peakFp4_f8f6f4
  | .f64 => peakF64

/-- A4, the alu pipe: 64 results an SM clock, 2 SMSP-cycles a warp instruction. Grade: Programming Guide and Nsight
Compute. Source: CUDA C++ Programming Guide 12.9 §6.4.1 Table 7, CC 10.0: 32-bit integer add 64, compare/minimum/maximum
64, bitwise AND/OR/XOR 64, shift 64 (results a clock a multiprocessor); Nsight Compute Kernel Profiling Guide §2.3.6:
alu executes 'most bit manipulation and logic instructions ... integer instructions, excluding IMAD and IMUL'. -/
def widthAlu : ℝ := 64

/-- A4, the imad pipe: 64 results an SM clock, 2 SMSP-cycles a warp instruction. Grade: Programming Guide. Source: CUDA
C++ Programming Guide 12.9 §6.4.1 Table 7, CC 10.0: '32-bit integer multiply, multiply-add, extended-precision
multiply-add' 64; Nsight Compute §2.3.6: IMUL and IMAD on FMA Heavy. -/
def widthImad : ℝ := 64

/-- A4, the shfl pipe: 32 results an SM clock, 4 SMSP-cycles a warp instruction. Grade: Programming Guide. Source: CUDA
C++ Programming Guide 12.9 §6.4.1 Table 7, CC 10.0: 'warp shuffle' 32; Nsight Compute §2.3.6: lsu issues shuffles. -/
def widthShfl : ℝ := 32

/-- A4, the uniform pipe: 65.3965 results an SM clock, 1.95729 SMSP-cycles a warp instruction. Grade: measured only (no
vendor document states the uniform datapath's rate). Source: Nsight Compute §2.3.6 defines the uniform pipe ('a scalar
unit') with no rate; the census's UP line (rows.csv): DC's uniform.cuh, the fastest form alone (uz_128x) at 0.5009
uniform instructions a SMSP-cycle, at a 2% margin 1.9572915 SMSP-cycles each (five runs); the width counts a uniform
instruction as a warp instruction's 32 results, as the pipe row reads a width. -/
def widthUniform : ℝ := 256000000 / 3914583

/-- A4: each pipe's results an SM clock. -/
def Pipe.width : Pipe → ℝ
  | .alu => widthAlu
  | .imad => widthImad
  | .shfl => widthShfl
  | .uniform => widthUniform

/-- Each class's tensor form: its operand type and its dense MACs a warp instruction, m · n · k over the warps that
issue one MMA (a `wgmma` warpgroup's 4, 1 for `mma.sync` and the single-thread `tcgen05.mma`), and for a 2:4 sparse
form, whose k is its logical k, half that (A3: twice the dense peak on logical k); `none` off the tensor pipe. -/
def Kind.form : Kind → Option (Ty × ℝ)
  | .ALU => none
  | .ATOME32 => none
  | .ATOMS32 => none
  | .DMMA16816 => some (.f64, 16 * 8 * 16 / 1)
  | .DMMA1684 => some (.f64, 16 * 8 * 4 / 1)
  | .DMMA1688 => some (.f64, 16 * 8 * 8 / 1)
  | .DMMA884 => some (.f64, 8 * 8 * 4 / 1)
  | .FFMA => none
  | .IMAD => none
  | .IMMA16816_S8 => some (.s8, 16 * 8 * 16 / 1)
  | .IMMA16832_S8 => some (.s8, 16 * 8 * 32 / 1)
  | .LDS128 => none
  | .LDTM_16X128B_X8 => none
  | .LDTM_16X256B_X4 => none
  | .LDTM_16X64B_X16 => none
  | .LDTM_32X32B_X1 => none
  | .LDTM_32X32B_X128 => none
  | .LDTM_32X32B_X16 => none
  | .LDTM_32X32B_X32 => none
  | .LDTM_32X32B_X4 => none
  | .LDTM_32X32B_X64 => none
  | .NOROW => none
  | .REDG32 => none
  | .REDUX => none
  | .SHFL => none
  | .STG128 => none
  | .STG32 => none
  | .STS128 => none
  | .STS32 => none
  | .UNI => none
  | .UNLISTED => none
  | .UTCHMMA_BF16_M128_N256 => some (.bf16, 128 * 256 * 16 / 1)
  | .UTCHMMA_F16_M128_N256 => some (.f16, 128 * 256 * 16 / 1)
  | .UTCHMMA_TF32_M128_N256 => some (.tf32, 128 * 256 * 8 / 1)
  | .UTCIMMA_M128_N128 => some (.s8, 128 * 128 * 32 / 1)
  | .UTCIMMA_M128_N256 => some (.s8, 128 * 256 * 32 / 1)
  | .UTCIMMA_M128_N256_SP => some (.s8, 128 * 256 * 64 / 2 / 1)
  | .UTCIMMA_M128_N32 => some (.s8, 128 * 32 * 32 / 1)
  | .UTCIMMA_M128_N64 => some (.s8, 128 * 64 * 32 / 1)
  | .UTCIMMA_M64_N128 => some (.s8, 64 * 128 * 32 / 1)
  | .UTCIMMA_M64_N256 => some (.s8, 64 * 256 * 32 / 1)
  | .UTCIMMA_M64_N64 => some (.s8, 64 * 64 * 32 / 1)
  | .UTCOMMA_MXF4NVF4_M128_N256 => some (.fp4, 128 * 256 * 64 / 1)
  | .UTCOMMA_MXF4_M128_N256 => some (.fp4, 128 * 256 * 64 / 1)
  | .UTCQMMA_E2M1_M128_N256 => some (.fp4_f8f6f4, 128 * 256 * 32 / 1)
  | .UTCQMMA_E2M3_M128_N256 => some (.fp6, 128 * 256 * 32 / 1)
  | .UTCQMMA_E3M2_M128_N256 => some (.fp6, 128 * 256 * 32 / 1)
  | .UTCQMMA_E4M3_M128_N128 => some (.e4m3, 128 * 128 * 32 / 1)
  | .UTCQMMA_E4M3_M128_N256 => some (.e4m3, 128 * 256 * 32 / 1)
  | .UTCQMMA_E5M2_M128_N256 => some (.e5m2, 128 * 256 * 32 / 1)
  | .UTCQMMA_MXF8F6F4_E4M3_M128_N256 => some (.e4m3, 128 * 256 * 32 / 1)
  | .VIADD => none

/-- The pipe each class issues to; `none` on the tensor pipe or issue only. -/
def Kind.pipe : Kind → Option Pipe
  | .ALU => some .alu
  | .ATOME32 => none
  | .ATOMS32 => none
  | .DMMA16816 => none
  | .DMMA1684 => none
  | .DMMA1688 => none
  | .DMMA884 => none
  | .FFMA => none
  | .IMAD => some .imad
  | .IMMA16816_S8 => none
  | .IMMA16832_S8 => none
  | .LDS128 => none
  | .LDTM_16X128B_X8 => none
  | .LDTM_16X256B_X4 => none
  | .LDTM_16X64B_X16 => none
  | .LDTM_32X32B_X1 => none
  | .LDTM_32X32B_X128 => none
  | .LDTM_32X32B_X16 => none
  | .LDTM_32X32B_X32 => none
  | .LDTM_32X32B_X4 => none
  | .LDTM_32X32B_X64 => none
  | .NOROW => none
  | .REDG32 => none
  | .REDUX => none
  | .SHFL => some .shfl
  | .STG128 => none
  | .STG32 => none
  | .STS128 => none
  | .STS32 => none
  | .UNI => some .uniform
  | .UNLISTED => none
  | .UTCHMMA_BF16_M128_N256 => none
  | .UTCHMMA_F16_M128_N256 => none
  | .UTCHMMA_TF32_M128_N256 => none
  | .UTCIMMA_M128_N128 => none
  | .UTCIMMA_M128_N256 => none
  | .UTCIMMA_M128_N256_SP => none
  | .UTCIMMA_M128_N32 => none
  | .UTCIMMA_M128_N64 => none
  | .UTCIMMA_M64_N128 => none
  | .UTCIMMA_M64_N256 => none
  | .UTCIMMA_M64_N64 => none
  | .UTCOMMA_MXF4NVF4_M128_N256 => none
  | .UTCOMMA_MXF4_M128_N256 => none
  | .UTCQMMA_E2M1_M128_N256 => none
  | .UTCQMMA_E2M3_M128_N256 => none
  | .UTCQMMA_E3M2_M128_N256 => none
  | .UTCQMMA_E4M3_M128_N128 => none
  | .UTCQMMA_E4M3_M128_N256 => none
  | .UTCQMMA_E5M2_M128_N256 => none
  | .UTCQMMA_MXF8F6F4_E4M3_M128_N256 => none
  | .VIADD => none

/-- A6's ω for each tensor class: the 32-bit words a warp instruction's accumulator fragment takes, m · n elements over
the warps that issue one MMA, an element one register and an f64 element two (DMMA, `mma.sync .f64`), a 2:4 sparse
form's as its dense twin's; `none` off the tensor pipe. Grade: PTX ISA. Source: PTX ISA 9.4, the accumulator (D)
fragments: mma.sync m16n8k32 .s32 four registers a lane (128 a warp instruction), m16n8k16 the same, .f64 m8n8k4 two f64
a lane (four registers, 128) and m16n8k4, k8 and k16 four (eight, 256); tcgen05.mma's accumulator is M × N 32-bit cells
of tensor memory (32,768 at M128 N256), which its one issuing thread's instruction writes. -/
def Kind.omega : Kind → Option ℝ
  | .ALU => none
  | .ATOME32 => none
  | .ATOMS32 => none
  | .DMMA16816 => some (16 * 8 * 2 / 1)
  | .DMMA1684 => some (16 * 8 * 2 / 1)
  | .DMMA1688 => some (16 * 8 * 2 / 1)
  | .DMMA884 => some (8 * 8 * 2 / 1)
  | .FFMA => none
  | .IMAD => none
  | .IMMA16816_S8 => some (16 * 8 / 1)
  | .IMMA16832_S8 => some (16 * 8 / 1)
  | .LDS128 => none
  | .LDTM_16X128B_X8 => none
  | .LDTM_16X256B_X4 => none
  | .LDTM_16X64B_X16 => none
  | .LDTM_32X32B_X1 => none
  | .LDTM_32X32B_X128 => none
  | .LDTM_32X32B_X16 => none
  | .LDTM_32X32B_X32 => none
  | .LDTM_32X32B_X4 => none
  | .LDTM_32X32B_X64 => none
  | .NOROW => none
  | .REDG32 => none
  | .REDUX => none
  | .SHFL => none
  | .STG128 => none
  | .STG32 => none
  | .STS128 => none
  | .STS32 => none
  | .UNI => none
  | .UNLISTED => none
  | .UTCHMMA_BF16_M128_N256 => some (128 * 256 / 1)
  | .UTCHMMA_F16_M128_N256 => some (128 * 256 / 1)
  | .UTCHMMA_TF32_M128_N256 => some (128 * 256 / 1)
  | .UTCIMMA_M128_N128 => some (128 * 128 / 1)
  | .UTCIMMA_M128_N256 => some (128 * 256 / 1)
  | .UTCIMMA_M128_N256_SP => some (128 * 256 / 1)
  | .UTCIMMA_M128_N32 => some (128 * 32 / 1)
  | .UTCIMMA_M128_N64 => some (128 * 64 / 1)
  | .UTCIMMA_M64_N128 => some (64 * 128 / 1)
  | .UTCIMMA_M64_N256 => some (64 * 256 / 1)
  | .UTCIMMA_M64_N64 => some (64 * 64 / 1)
  | .UTCOMMA_MXF4NVF4_M128_N256 => some (128 * 256 / 1)
  | .UTCOMMA_MXF4_M128_N256 => some (128 * 256 / 1)
  | .UTCQMMA_E2M1_M128_N256 => some (128 * 256 / 1)
  | .UTCQMMA_E2M3_M128_N256 => some (128 * 256 / 1)
  | .UTCQMMA_E3M2_M128_N256 => some (128 * 256 / 1)
  | .UTCQMMA_E4M3_M128_N128 => some (128 * 128 / 1)
  | .UTCQMMA_E4M3_M128_N256 => some (128 * 256 / 1)
  | .UTCQMMA_E5M2_M128_N256 => some (128 * 256 / 1)
  | .UTCQMMA_MXF8F6F4_E4M3_M128_N256 => some (128 * 256 / 1)
  | .VIADD => none

/-- **The B200's data sheet**: f_max in Hz, each class's MACs and its type's peak (`0` off the tensor pipe), its pipe,
each pipe's width, and no class on the shared-memory row (the census's shared-memory classes, ATOMS32, LDS128, STS128,
STS32, are issue only, so A5 is 0 for each). -/
def sheet : Sheet Kind Pipe where
  fmax := fmaxMHz * 10 ^ 6
  macs k := match k.form with
    | some (_, m) => m
    | none => 0
  peak k := match k.form with
    | some (t, _) => t.peak
    | none => 0
  pipe := Kind.pipe
  width := Pipe.width
  smem _ := 0

/-- The implied clock (no document states one) is under f_max. -/
theorem impliedMHz_lt_fmaxMHz : impliedMHz < fmaxMHz := by
  norm_num [impliedMHz, fmaxMHz]

/-- The FP64 clock (no document states one) is at most f_max. -/
theorem fp64MHz_le_fmaxMHz : fp64MHz ≤ fmaxMHz := by
  norm_num [fp64MHz, fmaxMHz]

/-- A3's s8 peak is consistent with the document's figure, not confirmed by it: 2 · 8192 MACs · 148 SMs at the implied
clock (no document states one), 1854 MHz, is its figure, dense 4500 ± 50, in 10¹² a second. No document states the
`implied` clock or the SM count (their grades above), so this holds the per-clock peak, the SMs and the clock to the
figure together, not each to a document. -/
theorem peakS8_figure :
    (4500 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakS8 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakS8 * smCount * (impliedMHz * 10 ^ 6) ≤ (4500 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakS8, smCount, impliedMHz]

/-- A3's e4m3 peak is consistent with the document's figure, not confirmed by it: 2 · 8192 MACs · 148 SMs at the implied
clock (no document states one), 1854 MHz, is its figure, dense 4500 ± 50, in 10¹² a second. No document states the
`implied` clock or the SM count (their grades above), so this holds the per-clock peak, the SMs and the clock to the
figure together, not each to a document. -/
theorem peakE4m3_figure :
    (4500 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakE4m3 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakE4m3 * smCount * (impliedMHz * 10 ^ 6) ≤ (4500 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakE4m3, smCount, impliedMHz]

/-- A3's e5m2 peak is consistent with the document's figure, not confirmed by it: 2 · 8192 MACs · 148 SMs at the implied
clock (no document states one), 1854 MHz, is its figure, dense 4500 ± 50, in 10¹² a second. No document states the
`implied` clock or the SM count (their grades above), so this holds the per-clock peak, the SMs and the clock to the
figure together, not each to a document. -/
theorem peakE5m2_figure :
    (4500 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakE5m2 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakE5m2 * smCount * (impliedMHz * 10 ^ 6) ≤ (4500 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakE5m2, smCount, impliedMHz]

/-- A3's fp6 peak is consistent with the document's figure, not confirmed by it: 2 · 8192 MACs · 148 SMs at the implied
clock (no document states one), 1854 MHz, is its figure, dense 4500 ± 50, in 10¹² a second. No document states the
`implied` clock or the SM count (their grades above), so this holds the per-clock peak, the SMs and the clock to the
figure together, not each to a document. -/
theorem peakFp6_figure :
    (4500 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakFp6 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakFp6 * smCount * (impliedMHz * 10 ^ 6) ≤ (4500 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakFp6, smCount, impliedMHz]

/-- A3's f16 peak is consistent with the document's figure, not confirmed by it: 2 · 4096 MACs · 148 SMs at the implied
clock (no document states one), 1854 MHz, is its figure, dense 2200 ± 50, in 10¹² a second. No document states the
`implied` clock or the SM count (their grades above), so this holds the per-clock peak, the SMs and the clock to the
figure together, not each to a document. -/
theorem peakF16_figure :
    (2200 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakF16 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakF16 * smCount * (impliedMHz * 10 ^ 6) ≤ (2200 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakF16, smCount, impliedMHz]

/-- A3's bf16 peak is consistent with the document's figure, not confirmed by it: 2 · 4096 MACs · 148 SMs at the implied
clock (no document states one), 1854 MHz, is its figure, dense 2200 ± 50, in 10¹² a second. No document states the
`implied` clock or the SM count (their grades above), so this holds the per-clock peak, the SMs and the clock to the
figure together, not each to a document. -/
theorem peakBf16_figure :
    (2200 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakBf16 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakBf16 * smCount * (impliedMHz * 10 ^ 6) ≤ (2200 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakBf16, smCount, impliedMHz]

/-- A3's tf32 peak is consistent with the document's figure, not confirmed by it: 2 · 2048 MACs · 148 SMs at the implied
clock (no document states one), 1854 MHz, is its figure, dense 1100 ± 50, in 10¹² a second. No document states the
`implied` clock or the SM count (their grades above), so this holds the per-clock peak, the SMs and the clock to the
figure together, not each to a document. -/
theorem peakTf32_figure :
    (1100 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakTf32 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakTf32 * smCount * (impliedMHz * 10 ^ 6) ≤ (1100 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakTf32, smCount, impliedMHz]

/-- A3's fp4 peak is consistent with the document's figure, not confirmed by it: 2 · 16384 MACs · 148 SMs at the implied
clock (no document states one), 1854 MHz, is its figure, dense 9000 ± 500, in 10¹² a second. No document states the
`implied` clock or the SM count (their grades above), so this holds the per-clock peak, the SMs and the clock to the
figure together, not each to a document. -/
theorem peakFp4_figure :
    (9000 - 500 : ℝ) * 10 ^ 12 ≤ 2 * peakFp4 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakFp4 * smCount * (impliedMHz * 10 ^ 6) ≤ (9000 + 500 : ℝ) * 10 ^ 12 := by
  norm_num [peakFp4, smCount, impliedMHz]

/-- A3's fp4_f8f6f4 peak is consistent with the document's figure, not confirmed by it: 2 · 8192 MACs · 148 SMs at the
implied clock (no document states one), 1854 MHz, is its figure, dense 4500 ± 50, in 10¹² a second. No document states
the `implied` clock or the SM count (their grades above), so this holds the per-clock peak, the SMs and the clock to the
figure together, not each to a document. -/
theorem peakFp4_f8f6f4_figure :
    (4500 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakFp4_f8f6f4 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakFp4_f8f6f4 * smCount * (impliedMHz * 10 ^ 6) ≤ (4500 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakFp4_f8f6f4, smCount, impliedMHz]

/-- A3's f64 peak is consistent with the document's figure, not confirmed by it: 2 · 64 MACs · 148 SMs at the FP64 clock
(no document states one), 1965 MHz, is its figure, dense 37 ± 0.5, in 10¹² a second. No document states the `fp64` clock
or the SM count (their grades above), so this holds the per-clock peak, the SMs and the clock to the figure together,
not each to a document. -/
theorem peakF64_figure :
    (37 - 1 / 2 : ℝ) * 10 ^ 12 ≤ 2 * peakF64 * smCount * (fp64MHz * 10 ^ 6) ∧
      2 * peakF64 * smCount * (fp64MHz * 10 ^ 6) ≤ (37 + 1 / 2 : ℝ) * 10 ^ 12 := by
  norm_num [peakF64, smCount, fp64MHz]

end

end Accounting.TS.Sheets.B200
