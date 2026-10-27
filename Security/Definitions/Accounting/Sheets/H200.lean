import Definitions.Accounting.TS.Sheet

/-!
# The H200's data sheet (generated)

The data-sheet machine (`Accounting.TS.Sheet.machine`) at the H200 SXM (sm_90a)'s vendor values: each constant with the
axiom that reads it, its value, its grade and its source, and the census of the machine model's `DS-90` vector, which
`machine_eq_census` proves the machine is, line for line. Each tensor peak has a lemma that the per-clock peak
reproduces the document's per-second figure at the clock that figure is quoted at. A3 reads `Run.n` as device totals
over the SMSPs, so a warpgroup's `wgmma` or a single-thread `tcgen05.mma` counts its MACs once, spread over the SM's
four SMSPs.

Scope: the machine is one GPU's own SMs. Work a peer GPU on the node does over NVLink or NVSwitch and sends back (whole
tiles as 64-byte digests) is outside it: the sheet bounds what the credited GPU's SMs compute, not which GPU computed a
tile, and `Device.lean`'s side channels don't list NVLink peer offload.

`uv run machine-model --device sm_90 sheet --lean` prints this file from
`benchmarks/pouw/machine_model/data/devices/sm_90/sheet.toml`, and the machine model's tests hold it to that output.
-/

namespace Accounting.TS.Sheets.H200

noncomputable section

/-- The classes of the `DS-90` census (its kinds). -/
inductive Kind
  | ALU
  | ATOME32
  | ATOMS32
  | BGMMA_B1_N256
  | BMMA168256_B1
  | DMMA16816
  | DMMA1684
  | DMMA1688
  | DMMA884
  | FFMA
  | HGMMA_BF16_N128
  | HGMMA_BF16_N256
  | HGMMA_BF16_N256_SP
  | HGMMA_F16_N256
  | HGMMA_F16_N256_SP
  | HGMMA_TF32_N256
  | HGMMA_TF32_N256_SP
  | HMMA16816_BF16
  | HMMA16816_F16
  | HMMA1688_TF32
  | IGMMA_S8_N128
  | IGMMA_S8_N16
  | IGMMA_S8_N192
  | IGMMA_S8_N256
  | IGMMA_S8_N256_RS
  | IGMMA_S8_N256_SP
  | IGMMA_S8_N32
  | IGMMA_S8_N64
  | IGMMA_S8_N8
  | IGMMA_U8S8_N256
  | IMAD
  | IMMA16816_S8
  | IMMA16832_S8
  | ISETP
  | LDS128
  | NOROW
  | QGMMA_E4M3_N128
  | QGMMA_E4M3_N256
  | QGMMA_E4M3_N256_SP
  | QGMMA_E5M2_N256
  | QGMMA_E5M2_N256_SP
  | REDG32
  | REDUX
  | SHFL
  | STG128
  | STG32
  | STS128
  | STS32
  | UNI
  | UNLISTED
  deriving DecidableEq

instance instFintypeKind : Fintype Kind :=
  Fintype.ofList [.ALU, .ATOME32, .ATOMS32, .BGMMA_B1_N256, .BMMA168256_B1, .DMMA16816, .DMMA1684, .DMMA1688, .DMMA884,
    .FFMA, .HGMMA_BF16_N128, .HGMMA_BF16_N256, .HGMMA_BF16_N256_SP, .HGMMA_F16_N256, .HGMMA_F16_N256_SP,
    .HGMMA_TF32_N256, .HGMMA_TF32_N256_SP, .HMMA16816_BF16, .HMMA16816_F16, .HMMA1688_TF32, .IGMMA_S8_N128,
    .IGMMA_S8_N16, .IGMMA_S8_N192, .IGMMA_S8_N256, .IGMMA_S8_N256_RS, .IGMMA_S8_N256_SP, .IGMMA_S8_N32, .IGMMA_S8_N64,
    .IGMMA_S8_N8, .IGMMA_U8S8_N256, .IMAD, .IMMA16816_S8, .IMMA16832_S8, .ISETP, .LDS128, .NOROW, .QGMMA_E4M3_N128,
    .QGMMA_E4M3_N256, .QGMMA_E4M3_N256_SP, .QGMMA_E5M2_N256, .QGMMA_E5M2_N256_SP, .REDG32, .REDUX, .SHFL, .STG128,
    .STG32, .STS128, .STS32, .UNI, .UNLISTED]
    (by intro k; cases k <;> decide)

/-- The Nsight Compute pipes the `DS-90` classes issue to. -/
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
  | f16
  | bf16
  | tf32
  | f64
  deriving DecidableEq

/-- A1, f_max: 1980 MHz. Grade: whitepaper (H100 SXM5), and measured. Source: H100 Tensor Core GPU Architecture
whitepaper V1.04, Table 3, H100 SXM5: 'GPU Boost Clock for FP64 Tensor Core Ops, FP32 and FP64 non Tensor Core Ops: 1980
MHz', footnote 2 'GPU Peak Clock and GPU Boost Clock are synonymous'; the H200 data sheet states no clock; the driver's
maximum on vy-dc-h200 is 1980 MHz (EPS1 DC). -/
def fmaxMHz : ℝ := 1980

/-- The SMs A3's per-SM peaks multiply: 132. Grade: whitepaper (H100 SXM5), and measured. Source: H100 whitepaper V1.04,
Table 3, H100 SXM5: 'SMs 132'; the H200 data sheet states none (its peaks equal H100 SXM5's); the device reports 132. -/
def smCount : ℝ := 132

/-- The tensor boost clock, at which the s8, e4m3, e5m2, f16, bf16, tf32 figures are quoted: 1830 MHz. Grade: whitepaper
(H100 SXM5). Source: H100 whitepaper V1.04, Table 3, H100 SXM5: 'GPU Boost Clock for FP8, FP16, BF16, TF32 Tensor Core
Ops: 1830 MHz'; the clock the tensor peaks are quoted at, 150 MHz under f_max. -/
def tensorMHz : ℝ := 1830

/-- The FP64 tensor boost clock, at which the f64 figure is quoted: 1980 MHz. Grade: whitepaper (H100 SXM5). Source:
H100 whitepaper V1.04, Table 3, H100 SXM5: 'GPU Boost Clock for FP64 Tensor Core Ops, FP32 and FP64 non Tensor Core Ops:
1980 MHz'; f_max itself. -/
def fp64MHz : ℝ := 1980

/-- A3, s8: 4096 dense MACs an SM clock, the document's dense figure, 1979 × 10¹² a second, over 2 · 132 SMs at the
tensor boost clock, 1830 MHz (A1's f_max is 1980 MHz). Grade: data sheet and whitepaper; per clock derived (figure over
SMs and the tensor clock). Source: H200 data sheet (nvidia.com/en-us/data-center/h200, Technical Specifications, H200
SXM): 'INT8 Tensor Core 3,958', footnote 2 'With sparsity', footnote 1 'Preliminary specifications'; H100 whitepaper
V1.04 Table 3 'Peak INT8 Tensor TOPS 1978.9/3957.8'; 'H100 SM Architecture': twice A100's MMA rate a SM on equivalent
data types. -/
def peakS8 : ℝ := 4096

/-- A3, e4m3: 4096 dense MACs an SM clock, the document's dense figure, 1979 × 10¹² a second, over 2 · 132 SMs at the
tensor boost clock, 1830 MHz (A1's f_max is 1980 MHz). Grade: data sheet and whitepaper; per clock derived. Source: H200
data sheet, H200 SXM: 'FP8 Tensor Core 3,958' with sparsity; H100 whitepaper V1.04 Table 3 'Peak FP8 Tensor TFLOPS with
FP32 Accumulate 1978.9/3957.8'. -/
def peakE4m3 : ℝ := 4096

/-- A3, e5m2: 4096 dense MACs an SM clock, the document's dense figure, 1979 × 10¹² a second, over 2 · 132 SMs at the
tensor boost clock, 1830 MHz (A1's f_max is 1980 MHz). Grade: data sheet and whitepaper; per clock derived. Source: H200
data sheet, H200 SXM: 'FP8 Tensor Core 3,958' with sparsity, for both FP8 formats (CUDA C++ Programming Guide 12.9
§18.8.1: FP8 'in either E4M3 or E5M2'). -/
def peakE5m2 : ℝ := 4096

/-- A3, f16: 2048 dense MACs an SM clock, the document's dense figure, 989.5 × 10¹² a second, over 2 · 132 SMs at the
tensor boost clock, 1830 MHz (A1's f_max is 1980 MHz). Grade: data sheet and whitepaper; per clock derived. Source: H200
data sheet, H200 SXM: 'FP16 Tensor Core 1,979' with sparsity; H100 whitepaper V1.04 Table 3 'Peak FP16 Tensor TFLOPS
with FP32 Accumulate 989.4/1978.9'. -/
def peakF16 : ℝ := 2048

/-- A3, bf16: 2048 dense MACs an SM clock, the document's dense figure, 989.5 × 10¹² a second, over 2 · 132 SMs at the
tensor boost clock, 1830 MHz (A1's f_max is 1980 MHz). Grade: data sheet and whitepaper; per clock derived. Source: H200
data sheet, H200 SXM: 'BFLOAT16 Tensor Core 1,979' with sparsity; H100 whitepaper V1.04 Table 3 'Peak BF16 Tensor TFLOPS
with FP32 Accumulate 989.4/1978.9'. -/
def peakBf16 : ℝ := 2048

/-- A3, tf32: 1024 dense MACs an SM clock, the document's dense figure, 494.5 × 10¹² a second, over 2 · 132 SMs at the
tensor boost clock, 1830 MHz (A1's f_max is 1980 MHz). Grade: data sheet and whitepaper; per clock derived. Source: H200
data sheet, H200 SXM: 'TF32 Tensor Core 989' with sparsity; H100 whitepaper V1.04 Table 3 'Peak TF32 Tensor TFLOPS
494.7/989.4'. -/
def peakTf32 : ℝ := 1024

/-- A3, f64: 128 dense MACs an SM clock, the document's dense figure, 67 × 10¹² a second, over 2 · 132 SMs at the FP64
tensor boost clock, 1980 MHz (A1's f_max is 1980 MHz). Grade: data sheet and whitepaper; per clock derived (figure over
SMs and the FP64 clock) and measured. Source: H200 data sheet, H200 SXM: 'FP64 Tensor Core 67 TFLOPS' (no sparsity
footnote); measured alone, DMMA.16x8x16 128 MACs an SM clock and DMMA.8x8x4 64, and beside a saturated wgmma s8 form t +
s = 0.935 to 0.958, so f64 shares the tensor pipe's time with the other types (unpriced90.cu). -/
def peakF64 : ℝ := 128

/-- A3: each type's dense MACs an SM clock. -/
def Ty.peak : Ty → ℝ
  | .s8 => peakS8
  | .e4m3 => peakE4m3
  | .e5m2 => peakE5m2
  | .f16 => peakF16
  | .bf16 => peakBf16
  | .tf32 => peakTf32
  | .f64 => peakF64

/-- A4, the alu pipe: 64 results an SM clock, 2 SMSP-cycles a warp instruction. Grade: Programming Guide and Nsight
Compute. Source: CUDA C++ Programming Guide 12.9 §6.4.1 Table 7, CC 9.0: 32-bit integer add 64, compare/minimum/maximum
64, bitwise AND/OR/XOR 64, shift 64 (results a clock a multiprocessor); Nsight Compute Kernel Profiling Guide §2.3.6:
alu executes 'most bit manipulation and logic instructions ... integer instructions, excluding IMAD and IMUL'. -/
def widthAlu : ℝ := 64

/-- A4, the imad pipe: 64 results an SM clock, 2 SMSP-cycles a warp instruction. Grade: Programming Guide. Source: CUDA
C++ Programming Guide 12.9 §6.4.1 Table 7, CC 9.0: '32-bit integer multiply, multiply-add, extended-precision
multiply-add' 64; Nsight Compute §2.3.6: IMUL and IMAD on FMA Heavy. -/
def widthImad : ℝ := 64

/-- A4, the shfl pipe: 32 results an SM clock, 4 SMSP-cycles a warp instruction. Grade: Programming Guide. Source: CUDA
C++ Programming Guide 12.9 §6.4.1 Table 7, CC 9.0: 'warp shuffle' 32; Nsight Compute §2.3.6: lsu issues shuffles. -/
def widthShfl : ℝ := 32

/-- A4, the uniform pipe: 65.2693 results an SM clock, 1.96111 SMSP-cycles a warp instruction. Grade: measured only (no
vendor document states the uniform datapath's rate). Source: Nsight Compute §2.3.6 defines the uniform pipe ('a scalar
unit') with no rate; the census's UP line (rows.csv): DC's uniform.cuh, the fastest of 11 forms alone (uz_128i) at
0.4999 uniform instructions a SMSP-cycle, at a 2% margin 1.96110716 SMSP-cycles each (five runs); the width counts a
uniform instruction as a warp instruction's 32 results, as the pipe row reads a width. -/
def widthUniform : ℝ := 3200000000 / 49027679

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
  | .BGMMA_B1_N256 => none
  | .BMMA168256_B1 => none
  | .DMMA16816 => some (.f64, 16 * 8 * 16 / 1)
  | .DMMA1684 => some (.f64, 16 * 8 * 4 / 1)
  | .DMMA1688 => some (.f64, 16 * 8 * 8 / 1)
  | .DMMA884 => some (.f64, 8 * 8 * 4 / 1)
  | .FFMA => none
  | .HGMMA_BF16_N128 => some (.bf16, 64 * 128 * 16 / 4)
  | .HGMMA_BF16_N256 => some (.bf16, 64 * 256 * 16 / 4)
  | .HGMMA_BF16_N256_SP => some (.bf16, 64 * 256 * 32 / 2 / 4)
  | .HGMMA_F16_N256 => some (.f16, 64 * 256 * 16 / 4)
  | .HGMMA_F16_N256_SP => some (.f16, 64 * 256 * 32 / 2 / 4)
  | .HGMMA_TF32_N256 => some (.tf32, 64 * 256 * 8 / 4)
  | .HGMMA_TF32_N256_SP => some (.tf32, 64 * 256 * 16 / 2 / 4)
  | .HMMA16816_BF16 => some (.bf16, 16 * 8 * 16 / 1)
  | .HMMA16816_F16 => some (.f16, 16 * 8 * 16 / 1)
  | .HMMA1688_TF32 => some (.tf32, 16 * 8 * 8 / 1)
  | .IGMMA_S8_N128 => some (.s8, 64 * 128 * 32 / 4)
  | .IGMMA_S8_N16 => some (.s8, 64 * 16 * 32 / 4)
  | .IGMMA_S8_N192 => some (.s8, 64 * 192 * 32 / 4)
  | .IGMMA_S8_N256 => some (.s8, 64 * 256 * 32 / 4)
  | .IGMMA_S8_N256_RS => some (.s8, 64 * 256 * 32 / 4)
  | .IGMMA_S8_N256_SP => some (.s8, 64 * 256 * 64 / 2 / 4)
  | .IGMMA_S8_N32 => some (.s8, 64 * 32 * 32 / 4)
  | .IGMMA_S8_N64 => some (.s8, 64 * 64 * 32 / 4)
  | .IGMMA_S8_N8 => some (.s8, 64 * 8 * 32 / 4)
  | .IGMMA_U8S8_N256 => some (.s8, 64 * 256 * 32 / 4)
  | .IMAD => none
  | .IMMA16816_S8 => some (.s8, 16 * 8 * 16 / 1)
  | .IMMA16832_S8 => some (.s8, 16 * 8 * 32 / 1)
  | .ISETP => none
  | .LDS128 => none
  | .NOROW => none
  | .QGMMA_E4M3_N128 => some (.e4m3, 64 * 128 * 32 / 4)
  | .QGMMA_E4M3_N256 => some (.e4m3, 64 * 256 * 32 / 4)
  | .QGMMA_E4M3_N256_SP => some (.e4m3, 64 * 256 * 64 / 2 / 4)
  | .QGMMA_E5M2_N256 => some (.e5m2, 64 * 256 * 32 / 4)
  | .QGMMA_E5M2_N256_SP => some (.e5m2, 64 * 256 * 64 / 2 / 4)
  | .REDG32 => none
  | .REDUX => none
  | .SHFL => none
  | .STG128 => none
  | .STG32 => none
  | .STS128 => none
  | .STS32 => none
  | .UNI => none
  | .UNLISTED => none

/-- The pipe each class issues to; `none` on the tensor pipe or issue only. -/
def Kind.pipe : Kind → Option Pipe
  | .ALU => some .alu
  | .ATOME32 => none
  | .ATOMS32 => none
  | .BGMMA_B1_N256 => none
  | .BMMA168256_B1 => none
  | .DMMA16816 => none
  | .DMMA1684 => none
  | .DMMA1688 => none
  | .DMMA884 => none
  | .FFMA => none
  | .HGMMA_BF16_N128 => none
  | .HGMMA_BF16_N256 => none
  | .HGMMA_BF16_N256_SP => none
  | .HGMMA_F16_N256 => none
  | .HGMMA_F16_N256_SP => none
  | .HGMMA_TF32_N256 => none
  | .HGMMA_TF32_N256_SP => none
  | .HMMA16816_BF16 => none
  | .HMMA16816_F16 => none
  | .HMMA1688_TF32 => none
  | .IGMMA_S8_N128 => none
  | .IGMMA_S8_N16 => none
  | .IGMMA_S8_N192 => none
  | .IGMMA_S8_N256 => none
  | .IGMMA_S8_N256_RS => none
  | .IGMMA_S8_N256_SP => none
  | .IGMMA_S8_N32 => none
  | .IGMMA_S8_N64 => none
  | .IGMMA_S8_N8 => none
  | .IGMMA_U8S8_N256 => none
  | .IMAD => some .imad
  | .IMMA16816_S8 => none
  | .IMMA16832_S8 => none
  | .ISETP => none
  | .LDS128 => none
  | .NOROW => none
  | .QGMMA_E4M3_N128 => none
  | .QGMMA_E4M3_N256 => none
  | .QGMMA_E4M3_N256_SP => none
  | .QGMMA_E5M2_N256 => none
  | .QGMMA_E5M2_N256_SP => none
  | .REDG32 => none
  | .REDUX => none
  | .SHFL => some .shfl
  | .STG128 => none
  | .STG32 => none
  | .STS128 => none
  | .STS32 => none
  | .UNI => some .uniform
  | .UNLISTED => none

/-- A6's ω for each tensor class (`SheetWords`, `accounting-sheet-words`): the 32-bit words a warp instruction's
accumulator fragment takes, m · n elements over the warps that issue one MMA, an element one register and an f64 element
two (DMMA, `mma.sync .f64`), a 2:4 sparse form's as its dense twin's; `none` off the tensor pipe. Grade: PTX ISA.
Source: PTX ISA 9.4, the accumulator (D) fragments: mma.sync m16n8k32 .s32 four registers a lane (128 a warp
instruction), m16n8k16 .f32 the same, .f64 m8n8k4 two f64 a lane (four registers, 128) and m16n8k4, k8 and k16 four
(eight, 256); wgmma m64nNk32 .s32 N/2 registers a thread of its warpgroup (4,096 a warp at N256). -/
def Kind.omega : Kind → Option ℝ
  | .ALU => none
  | .ATOME32 => none
  | .ATOMS32 => none
  | .BGMMA_B1_N256 => none
  | .BMMA168256_B1 => none
  | .DMMA16816 => some (16 * 8 * 2 / 1)
  | .DMMA1684 => some (16 * 8 * 2 / 1)
  | .DMMA1688 => some (16 * 8 * 2 / 1)
  | .DMMA884 => some (8 * 8 * 2 / 1)
  | .FFMA => none
  | .HGMMA_BF16_N128 => some (64 * 128 / 4)
  | .HGMMA_BF16_N256 => some (64 * 256 / 4)
  | .HGMMA_BF16_N256_SP => some (64 * 256 / 4)
  | .HGMMA_F16_N256 => some (64 * 256 / 4)
  | .HGMMA_F16_N256_SP => some (64 * 256 / 4)
  | .HGMMA_TF32_N256 => some (64 * 256 / 4)
  | .HGMMA_TF32_N256_SP => some (64 * 256 / 4)
  | .HMMA16816_BF16 => some (16 * 8 / 1)
  | .HMMA16816_F16 => some (16 * 8 / 1)
  | .HMMA1688_TF32 => some (16 * 8 / 1)
  | .IGMMA_S8_N128 => some (64 * 128 / 4)
  | .IGMMA_S8_N16 => some (64 * 16 / 4)
  | .IGMMA_S8_N192 => some (64 * 192 / 4)
  | .IGMMA_S8_N256 => some (64 * 256 / 4)
  | .IGMMA_S8_N256_RS => some (64 * 256 / 4)
  | .IGMMA_S8_N256_SP => some (64 * 256 / 4)
  | .IGMMA_S8_N32 => some (64 * 32 / 4)
  | .IGMMA_S8_N64 => some (64 * 64 / 4)
  | .IGMMA_S8_N8 => some (64 * 8 / 4)
  | .IGMMA_U8S8_N256 => some (64 * 256 / 4)
  | .IMAD => none
  | .IMMA16816_S8 => some (16 * 8 / 1)
  | .IMMA16832_S8 => some (16 * 8 / 1)
  | .ISETP => none
  | .LDS128 => none
  | .NOROW => none
  | .QGMMA_E4M3_N128 => some (64 * 128 / 4)
  | .QGMMA_E4M3_N256 => some (64 * 256 / 4)
  | .QGMMA_E4M3_N256_SP => some (64 * 256 / 4)
  | .QGMMA_E5M2_N256 => some (64 * 256 / 4)
  | .QGMMA_E5M2_N256_SP => some (64 * 256 / 4)
  | .REDG32 => none
  | .REDUX => none
  | .SHFL => none
  | .STG128 => none
  | .STG32 => none
  | .STS128 => none
  | .STS32 => none
  | .UNI => none
  | .UNLISTED => none

/-- **The H200's data sheet**: f_max in Hz, each class's MACs and its type's peak (`0` off the tensor pipe), its pipe,
each pipe's width, and no class on the shared-memory row (the census's shared-memory classes, ATOMS32, LDS128, STS128,
STS32, are issue only, so A5 bounds nothing here). -/
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

/-- `census/DS-90.csv`'s `DS_tensor` lines, SMSP-cycles a warp instruction; 0 for a class with none. -/
def tensorLine : Kind → ℝ
  | .ALU => 0
  | .ATOME32 => 0
  | .ATOMS32 => 0
  | .BGMMA_B1_N256 => 0
  | .BMMA168256_B1 => 0
  | .DMMA16816 => 64
  | .DMMA1684 => 16
  | .DMMA1688 => 32
  | .DMMA884 => 8
  | .FFMA => 0
  | .HGMMA_BF16_N128 => 64
  | .HGMMA_BF16_N256 => 128
  | .HGMMA_BF16_N256_SP => 128
  | .HGMMA_F16_N256 => 128
  | .HGMMA_F16_N256_SP => 128
  | .HGMMA_TF32_N256 => 128
  | .HGMMA_TF32_N256_SP => 128
  | .HMMA16816_BF16 => 4
  | .HMMA16816_F16 => 4
  | .HMMA1688_TF32 => 4
  | .IGMMA_S8_N128 => 64
  | .IGMMA_S8_N16 => 8
  | .IGMMA_S8_N192 => 96
  | .IGMMA_S8_N256 => 128
  | .IGMMA_S8_N256_RS => 128
  | .IGMMA_S8_N256_SP => 128
  | .IGMMA_S8_N32 => 16
  | .IGMMA_S8_N64 => 32
  | .IGMMA_S8_N8 => 4
  | .IGMMA_U8S8_N256 => 128
  | .IMAD => 0
  | .IMMA16816_S8 => 2
  | .IMMA16832_S8 => 4
  | .ISETP => 0
  | .LDS128 => 0
  | .NOROW => 0
  | .QGMMA_E4M3_N128 => 64
  | .QGMMA_E4M3_N256 => 128
  | .QGMMA_E4M3_N256_SP => 128
  | .QGMMA_E5M2_N256 => 128
  | .QGMMA_E5M2_N256_SP => 128
  | .REDG32 => 0
  | .REDUX => 0
  | .SHFL => 0
  | .STG128 => 0
  | .STG32 => 0
  | .STS128 => 0
  | .STS32 => 0
  | .UNI => 0
  | .UNLISTED => 0

/-- `census/DS-90.csv`'s `DS_alu` lines, SMSP-cycles a warp instruction; 0 for a class with none. -/
def aluLine : Kind → ℝ
  | .ALU => 2
  | .ATOME32 => 0
  | .ATOMS32 => 0
  | .BGMMA_B1_N256 => 0
  | .BMMA168256_B1 => 0
  | .DMMA16816 => 0
  | .DMMA1684 => 0
  | .DMMA1688 => 0
  | .DMMA884 => 0
  | .FFMA => 0
  | .HGMMA_BF16_N128 => 0
  | .HGMMA_BF16_N256 => 0
  | .HGMMA_BF16_N256_SP => 0
  | .HGMMA_F16_N256 => 0
  | .HGMMA_F16_N256_SP => 0
  | .HGMMA_TF32_N256 => 0
  | .HGMMA_TF32_N256_SP => 0
  | .HMMA16816_BF16 => 0
  | .HMMA16816_F16 => 0
  | .HMMA1688_TF32 => 0
  | .IGMMA_S8_N128 => 0
  | .IGMMA_S8_N16 => 0
  | .IGMMA_S8_N192 => 0
  | .IGMMA_S8_N256 => 0
  | .IGMMA_S8_N256_RS => 0
  | .IGMMA_S8_N256_SP => 0
  | .IGMMA_S8_N32 => 0
  | .IGMMA_S8_N64 => 0
  | .IGMMA_S8_N8 => 0
  | .IGMMA_U8S8_N256 => 0
  | .IMAD => 0
  | .IMMA16816_S8 => 0
  | .IMMA16832_S8 => 0
  | .ISETP => 0
  | .LDS128 => 0
  | .NOROW => 0
  | .QGMMA_E4M3_N128 => 0
  | .QGMMA_E4M3_N256 => 0
  | .QGMMA_E4M3_N256_SP => 0
  | .QGMMA_E5M2_N256 => 0
  | .QGMMA_E5M2_N256_SP => 0
  | .REDG32 => 0
  | .REDUX => 0
  | .SHFL => 0
  | .STG128 => 0
  | .STG32 => 0
  | .STS128 => 0
  | .STS32 => 0
  | .UNI => 0
  | .UNLISTED => 0

/-- `census/DS-90.csv`'s `DS_imad` lines, SMSP-cycles a warp instruction; 0 for a class with none. -/
def imadLine : Kind → ℝ
  | .ALU => 0
  | .ATOME32 => 0
  | .ATOMS32 => 0
  | .BGMMA_B1_N256 => 0
  | .BMMA168256_B1 => 0
  | .DMMA16816 => 0
  | .DMMA1684 => 0
  | .DMMA1688 => 0
  | .DMMA884 => 0
  | .FFMA => 0
  | .HGMMA_BF16_N128 => 0
  | .HGMMA_BF16_N256 => 0
  | .HGMMA_BF16_N256_SP => 0
  | .HGMMA_F16_N256 => 0
  | .HGMMA_F16_N256_SP => 0
  | .HGMMA_TF32_N256 => 0
  | .HGMMA_TF32_N256_SP => 0
  | .HMMA16816_BF16 => 0
  | .HMMA16816_F16 => 0
  | .HMMA1688_TF32 => 0
  | .IGMMA_S8_N128 => 0
  | .IGMMA_S8_N16 => 0
  | .IGMMA_S8_N192 => 0
  | .IGMMA_S8_N256 => 0
  | .IGMMA_S8_N256_RS => 0
  | .IGMMA_S8_N256_SP => 0
  | .IGMMA_S8_N32 => 0
  | .IGMMA_S8_N64 => 0
  | .IGMMA_S8_N8 => 0
  | .IGMMA_U8S8_N256 => 0
  | .IMAD => 2
  | .IMMA16816_S8 => 0
  | .IMMA16832_S8 => 0
  | .ISETP => 0
  | .LDS128 => 0
  | .NOROW => 0
  | .QGMMA_E4M3_N128 => 0
  | .QGMMA_E4M3_N256 => 0
  | .QGMMA_E4M3_N256_SP => 0
  | .QGMMA_E5M2_N256 => 0
  | .QGMMA_E5M2_N256_SP => 0
  | .REDG32 => 0
  | .REDUX => 0
  | .SHFL => 0
  | .STG128 => 0
  | .STG32 => 0
  | .STS128 => 0
  | .STS32 => 0
  | .UNI => 0
  | .UNLISTED => 0

/-- `census/DS-90.csv`'s `DS_shfl` lines, SMSP-cycles a warp instruction; 0 for a class with none. -/
def shflLine : Kind → ℝ
  | .ALU => 0
  | .ATOME32 => 0
  | .ATOMS32 => 0
  | .BGMMA_B1_N256 => 0
  | .BMMA168256_B1 => 0
  | .DMMA16816 => 0
  | .DMMA1684 => 0
  | .DMMA1688 => 0
  | .DMMA884 => 0
  | .FFMA => 0
  | .HGMMA_BF16_N128 => 0
  | .HGMMA_BF16_N256 => 0
  | .HGMMA_BF16_N256_SP => 0
  | .HGMMA_F16_N256 => 0
  | .HGMMA_F16_N256_SP => 0
  | .HGMMA_TF32_N256 => 0
  | .HGMMA_TF32_N256_SP => 0
  | .HMMA16816_BF16 => 0
  | .HMMA16816_F16 => 0
  | .HMMA1688_TF32 => 0
  | .IGMMA_S8_N128 => 0
  | .IGMMA_S8_N16 => 0
  | .IGMMA_S8_N192 => 0
  | .IGMMA_S8_N256 => 0
  | .IGMMA_S8_N256_RS => 0
  | .IGMMA_S8_N256_SP => 0
  | .IGMMA_S8_N32 => 0
  | .IGMMA_S8_N64 => 0
  | .IGMMA_S8_N8 => 0
  | .IGMMA_U8S8_N256 => 0
  | .IMAD => 0
  | .IMMA16816_S8 => 0
  | .IMMA16832_S8 => 0
  | .ISETP => 0
  | .LDS128 => 0
  | .NOROW => 0
  | .QGMMA_E4M3_N128 => 0
  | .QGMMA_E4M3_N256 => 0
  | .QGMMA_E4M3_N256_SP => 0
  | .QGMMA_E5M2_N256 => 0
  | .QGMMA_E5M2_N256_SP => 0
  | .REDG32 => 0
  | .REDUX => 0
  | .SHFL => 4
  | .STG128 => 0
  | .STG32 => 0
  | .STS128 => 0
  | .STS32 => 0
  | .UNI => 0
  | .UNLISTED => 0

/-- `census/DS-90.csv`'s `DS_uniform` lines, SMSP-cycles a warp instruction; 0 for a class with none. -/
def uniformLine : Kind → ℝ
  | .ALU => 0
  | .ATOME32 => 0
  | .ATOMS32 => 0
  | .BGMMA_B1_N256 => 0
  | .BMMA168256_B1 => 0
  | .DMMA16816 => 0
  | .DMMA1684 => 0
  | .DMMA1688 => 0
  | .DMMA884 => 0
  | .FFMA => 0
  | .HGMMA_BF16_N128 => 0
  | .HGMMA_BF16_N256 => 0
  | .HGMMA_BF16_N256_SP => 0
  | .HGMMA_F16_N256 => 0
  | .HGMMA_F16_N256_SP => 0
  | .HGMMA_TF32_N256 => 0
  | .HGMMA_TF32_N256_SP => 0
  | .HMMA16816_BF16 => 0
  | .HMMA16816_F16 => 0
  | .HMMA1688_TF32 => 0
  | .IGMMA_S8_N128 => 0
  | .IGMMA_S8_N16 => 0
  | .IGMMA_S8_N192 => 0
  | .IGMMA_S8_N256 => 0
  | .IGMMA_S8_N256_RS => 0
  | .IGMMA_S8_N256_SP => 0
  | .IGMMA_S8_N32 => 0
  | .IGMMA_S8_N64 => 0
  | .IGMMA_S8_N8 => 0
  | .IGMMA_U8S8_N256 => 0
  | .IMAD => 0
  | .IMMA16816_S8 => 0
  | .IMMA16832_S8 => 0
  | .ISETP => 0
  | .LDS128 => 0
  | .NOROW => 0
  | .QGMMA_E4M3_N128 => 0
  | .QGMMA_E4M3_N256 => 0
  | .QGMMA_E4M3_N256_SP => 0
  | .QGMMA_E5M2_N256 => 0
  | .QGMMA_E5M2_N256_SP => 0
  | .REDG32 => 0
  | .REDUX => 0
  | .SHFL => 0
  | .STG128 => 0
  | .STG32 => 0
  | .STS128 => 0
  | .STS32 => 0
  | .UNI => 49027679 / 25000000
  | .UNLISTED => 0

/-- **`DS-90`'s census** (`census/DS-90.csv`): SMSP-cycles a warp instruction of each class holds each row; every class
holds issue for 1, and a class with no line on a row holds it not at all. -/
def census : SheetRow Pipe → Kind → ℝ
  | .issue, _ => 1
  | .tensor, k => tensorLine k
  | .pipe .alu, k => aluLine k
  | .pipe .imad, k => imadLine k
  | .pipe .shfl, k => shflLine k
  | .pipe .uniform, k => uniformLine k
  | .smem, _ => 0

/-- The issue row: every class holds it for 1, on the sheet as in the census. A2 (a SMSP issues one warp instruction a
SMSP-cycle). Grade: Programming Guide. Source: CUDA C++ Programming Guide 12.9 §18.8.1 (CC 9.0): '4 warp schedulers ...
at every instruction issue time, each scheduler issues one instruction for one of its assigned warps'. A2u (a uniform
instruction, `UNI`, takes that slot too). Grade: measured only (no vendor document states it). Source: no vendor
document gives a uniform instruction's issue cost: Nsight Compute §2.3.6 defines the uniform pipe with no rate, and CUDA
Binary Utilities §4.3 lists the Hopper uniform opcodes (integer only); measured at saturation beside an async wgmma:
each uniform instruction costs 1.08 to 1.19 SMSP-cycles at issue 0.87 with no dual issue (dc/probes/sm90_emp.cu
emp_pack3), and 0.984 of a slot beside a saturated lane stream (the census's UNI issue line, DC-90). -/
theorem machine_issue (k : Kind) : sheet.machine.a .issue k = census .issue k := by
  simp [Sheet.machine, census]

-- the case split unfolds `Kind.form` in every class's case: its cost grows with the square of the class count
set_option maxHeartbeats 1000000 in
/-- The tensor row: each class's MACs over a quarter of its type's peak is its census line. -/
theorem machine_tensor (k : Kind) : sheet.machine.a .tensor k = census .tensor k := by
  cases k <;> norm_num [Sheet.machine, sheet, census, tensorLine, Kind.form, Ty.peak, peakS8, peakE4m3, peakE5m2,
      peakF16, peakBf16, peakTf32, peakF64]

/-- The alu pipe's row: 128 over its width for each class on it, and 0 for the rest. -/
theorem machine_alu (k : Kind) : sheet.machine.a (.pipe .alu) k = census (.pipe .alu) k := by
  cases k <;> norm_num [Sheet.machine, sheet, census, aluLine, Kind.pipe, Pipe.width, widthAlu]

/-- The imad pipe's row: 128 over its width for each class on it, and 0 for the rest. -/
theorem machine_imad (k : Kind) : sheet.machine.a (.pipe .imad) k = census (.pipe .imad) k := by
  cases k <;> norm_num [Sheet.machine, sheet, census, imadLine, Kind.pipe, Pipe.width, widthImad]

/-- The shfl pipe's row: 128 over its width for each class on it, and 0 for the rest. -/
theorem machine_shfl (k : Kind) : sheet.machine.a (.pipe .shfl) k = census (.pipe .shfl) k := by
  cases k <;> norm_num [Sheet.machine, sheet, census, shflLine, Kind.pipe, Pipe.width, widthShfl]

/-- The uniform pipe's row: 128 over its width for each class on it, and 0 for the rest. -/
theorem machine_uniform (k : Kind) : sheet.machine.a (.pipe .uniform) k = census (.pipe .uniform) k := by
  cases k <;> norm_num [Sheet.machine, sheet, census, uniformLine, Kind.pipe, Pipe.width, widthUniform]

/-- The shared-memory row: no class holds it. -/
theorem machine_smem (k : Kind) : sheet.machine.a .smem k = census .smem k := by
  simp [Sheet.machine, sheet, census]

/-- The sheet's machine is `DS-90`'s census, line for line. -/
theorem machine_eq_census : sheet.machine.a = census := by
  funext r k
  match r with
  | .issue => exact machine_issue k
  | .tensor => exact machine_tensor k
  | .pipe .alu => exact machine_alu k
  | .pipe .imad => exact machine_imad k
  | .pipe .shfl => exact machine_shfl k
  | .pipe .uniform => exact machine_uniform k
  | .smem => exact machine_smem k

/-- The tensor boost clock is under f_max. -/
theorem tensorMHz_lt_fmaxMHz : tensorMHz < fmaxMHz := by
  norm_num [tensorMHz, fmaxMHz]

/-- The FP64 tensor boost clock is at most f_max. -/
theorem fp64MHz_le_fmaxMHz : fp64MHz ≤ fmaxMHz := by
  norm_num [fp64MHz, fmaxMHz]

/-- A3's s8 peak reproduces the document: 2 · 4096 MACs · 132 SMs at the tensor boost clock, 1830 MHz, is its figure,
3958 with sparsity, dense 1979 ± 0.25, in 10¹² a second. -/
theorem peakS8_figure :
    (1979 - 1 / 4 : ℝ) * 10 ^ 12 ≤ 2 * peakS8 * smCount * (tensorMHz * 10 ^ 6) ∧
      2 * peakS8 * smCount * (tensorMHz * 10 ^ 6) ≤ (1979 + 1 / 4 : ℝ) * 10 ^ 12 := by
  norm_num [peakS8, smCount, tensorMHz]

/-- A3's e4m3 peak reproduces the document: 2 · 4096 MACs · 132 SMs at the tensor boost clock, 1830 MHz, is its figure,
3958 with sparsity, dense 1979 ± 0.25, in 10¹² a second. -/
theorem peakE4m3_figure :
    (1979 - 1 / 4 : ℝ) * 10 ^ 12 ≤ 2 * peakE4m3 * smCount * (tensorMHz * 10 ^ 6) ∧
      2 * peakE4m3 * smCount * (tensorMHz * 10 ^ 6) ≤ (1979 + 1 / 4 : ℝ) * 10 ^ 12 := by
  norm_num [peakE4m3, smCount, tensorMHz]

/-- A3's e5m2 peak reproduces the document: 2 · 4096 MACs · 132 SMs at the tensor boost clock, 1830 MHz, is its figure,
3958 with sparsity, dense 1979 ± 0.25, in 10¹² a second. -/
theorem peakE5m2_figure :
    (1979 - 1 / 4 : ℝ) * 10 ^ 12 ≤ 2 * peakE5m2 * smCount * (tensorMHz * 10 ^ 6) ∧
      2 * peakE5m2 * smCount * (tensorMHz * 10 ^ 6) ≤ (1979 + 1 / 4 : ℝ) * 10 ^ 12 := by
  norm_num [peakE5m2, smCount, tensorMHz]

/-- A3's f16 peak reproduces the document: 2 · 2048 MACs · 132 SMs at the tensor boost clock, 1830 MHz, is its figure,
1979 with sparsity, dense 989.5 ± 0.25, in 10¹² a second. -/
theorem peakF16_figure :
    (1979 / 2 - 1 / 4 : ℝ) * 10 ^ 12 ≤ 2 * peakF16 * smCount * (tensorMHz * 10 ^ 6) ∧
      2 * peakF16 * smCount * (tensorMHz * 10 ^ 6) ≤ (1979 / 2 + 1 / 4 : ℝ) * 10 ^ 12 := by
  norm_num [peakF16, smCount, tensorMHz]

/-- A3's bf16 peak reproduces the document: 2 · 2048 MACs · 132 SMs at the tensor boost clock, 1830 MHz, is its figure,
1979 with sparsity, dense 989.5 ± 0.25, in 10¹² a second. -/
theorem peakBf16_figure :
    (1979 / 2 - 1 / 4 : ℝ) * 10 ^ 12 ≤ 2 * peakBf16 * smCount * (tensorMHz * 10 ^ 6) ∧
      2 * peakBf16 * smCount * (tensorMHz * 10 ^ 6) ≤ (1979 / 2 + 1 / 4 : ℝ) * 10 ^ 12 := by
  norm_num [peakBf16, smCount, tensorMHz]

/-- A3's tf32 peak reproduces the document: 2 · 1024 MACs · 132 SMs at the tensor boost clock, 1830 MHz, is its figure,
989 with sparsity, dense 494.5 ± 0.25, in 10¹² a second. -/
theorem peakTf32_figure :
    (989 / 2 - 1 / 4 : ℝ) * 10 ^ 12 ≤ 2 * peakTf32 * smCount * (tensorMHz * 10 ^ 6) ∧
      2 * peakTf32 * smCount * (tensorMHz * 10 ^ 6) ≤ (989 / 2 + 1 / 4 : ℝ) * 10 ^ 12 := by
  norm_num [peakTf32, smCount, tensorMHz]

/-- A3's f64 peak reproduces the document: 2 · 128 MACs · 132 SMs at the FP64 tensor boost clock, 1980 MHz, is its
figure, dense 67 ± 0.5, in 10¹² a second. -/
theorem peakF64_figure :
    (67 - 1 / 2 : ℝ) * 10 ^ 12 ≤ 2 * peakF64 * smCount * (fp64MHz * 10 ^ 6) ∧
      2 * peakF64 * smCount * (fp64MHz * 10 ^ 6) ≤ (67 + 1 / 2 : ℝ) * 10 ^ 12 := by
  norm_num [peakF64, smCount, fp64MHz]

end

end Accounting.TS.Sheets.H200
