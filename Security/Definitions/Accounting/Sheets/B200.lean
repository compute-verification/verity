import Definitions.Accounting.TS.Sheet

/-!
# The B200's data sheet (generated)

The data-sheet machine (`Accounting.TS.Sheet.machine`) at the HGX B200 (sm_100a, 180 GB, 1000 W)'s vendor values: each
constant with the axiom that reads it, its value, its grade and its source, and the census of the machine model's
`DS-100` vector, which `machine_eq_census` proves the machine is, line for line. Each tensor peak has a lemma that the
per-clock peak reproduces the document's per-second figure at the clock that figure is quoted at. A3 reads `Run.n` as
device totals over the SMSPs, so a warpgroup's `wgmma` or a single-thread `tcgen05.mma` counts its MACs once, spread
over the SM's four SMSPs.

`uv run machine-model --device sm_100 sheet --lean` prints this file from
`benchmarks/pouw/machine_model/data/devices/sm_100/sheet.toml`, and the machine model's tests hold it to that output.
The grades, and where the sm_120 sheet's structure doesn't fit this device: note:20261019T0511Z-draft-mm-sheets.
-/

namespace Accounting.TS.Sheets.B200

noncomputable section

/-- The classes of the `DS-100` census (its kinds). -/
inductive Kind
  | ALU
  | FFMA
  | IMAD
  | IMMA16816_S8
  | IMMA16832_S8
  | LDTM_16X128B_X8
  | LDTM_16X256B_X4
  | LDTM_16X64B_X16
  | LDTM_32X32B_X1
  | LDTM_32X32B_X128
  | LDTM_32X32B_X16
  | LDTM_32X32B_X32
  | LDTM_32X32B_X4
  | LDTM_32X32B_X64
  | SHFL
  | UNI
  | UTCHMMA_F16_M128_N256
  | UTCIMMA_M128_N128
  | UTCIMMA_M128_N256
  | UTCIMMA_M128_N32
  | UTCIMMA_M128_N64
  | UTCIMMA_M64_N128
  | UTCIMMA_M64_N256
  | UTCIMMA_M64_N64
  | UTCQMMA_E4M3_M128_N128
  | UTCQMMA_E4M3_M128_N256
  deriving DecidableEq, Fintype

/-- The Nsight Compute pipes the `DS-100` classes issue to. -/
inductive Pipe
  | alu
  | imad
  | shfl
  deriving DecidableEq, Fintype

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
  deriving DecidableEq

/-- A1, f_max: 1965 MHz. Grade: measured only (the driver's maximum). Source: no NVIDIA document read states a B200
clock (Blackwell Architecture Technical Brief V2.1, the Lenovo HGX B200 Product Guide, nvidia.com's HGX page); the
driver's maximum at default clocks on vy-dc-b200 is 1965 MHz (sm_100/table.toml, EPS1 DC), RT-ISA and RT-CLOCK saw at
most 1957.1 MHz; the brief's 'FP64 Tensor Core 37 teraFLOPS' at the Guide's 64 FP64 results an SM clock over 148 SMs
puts the FP64 clock at 1927 to 1980 MHz. -/
def fmaxMHz : ℝ := 1965

/-- The SMs A3's per-SM peaks multiply: 148. Grade: measured only. Source: no NVIDIA document read gives the B200's SM
count (Technical Brief V2.1 Table 3 says only '16 Blackwell Die for 8 GPUs'); the device reports 148
(sm_100/table.toml). -/
def smCount : ℝ := 148

/-- The `implied` clock, at which the tensor figures are quoted: 1854 MHz. Grade: none (derived: no document states it).
Source: the clock at which every per-clock peak below reproduces the Technical Brief's dense figure to within half its
last digit, which holds from 1835 to 1855 MHz, and the Lenovo guide's FP16 2.25 too from 1852 MHz; 111 MHz under f_max,
as the H200's tensor clock is 150 MHz under its f_max (open premise B200-PEAK-PER-CLOCK: the per-clock peaks are
measured, the clock they are quoted at is nowhere). -/
def impliedMHz : ℝ := 1854

/-- A3, s8: 8192 dense MACs an SM clock. Grade: data sheet per second; per clock measured only. Source: NVIDIA Blackwell
Architecture Technical Brief V2.1, Table 3 'System Specifications for HGX B300 and HGX B200', Per GPU Specs, HGX B200:
'INT8 Tensor Core 4.5 | 9 petaOPS' (Dense | Sparse); Lenovo ThinkSystem NVIDIA HGX B200 180GB 1000W GPU Product Guide
Table 2: '4.5 / 9 petaOPS'; measured UTCIMMA_M128_N256 512 SMSP-cycles (DC-100). -/
def peakS8 : ℝ := 8192

/-- A3, e4m3: 8192 dense MACs an SM clock. Grade: data sheet per second; per clock measured only. Source: Technical
Brief V2.1 Table 3, HGX B200: 'FP8/FP6 Tensor Core 4.5 | 9 petaFLOPS'; measured UTCQMMA_E4M3 lines (DC-100). -/
def peakE4m3 : ℝ := 8192

/-- A3, e5m2: 8192 dense MACs an SM clock. Grade: data sheet per second; per clock derived at the implied clock. Source:
Technical Brief V2.1 Table 3, HGX B200: 'FP8/FP6 Tensor Core 4.5 | 9 petaFLOPS', for both FP8 formats (PTX ISA 9.4
tcgen05.mma .kind::f8f6f4: e4m3, e5m2). -/
def peakE5m2 : ℝ := 8192

/-- A3, fp6: 8192 dense MACs an SM clock. Grade: data sheet per second; per clock derived at the implied clock; no
class. Source: Technical Brief V2.1 Table 3, HGX B200: 'FP8/FP6 Tensor Core 4.5 | 9 petaFLOPS' (PTX ISA 9.4
.kind::f8f6f4: e2m3, e3m2); the CUDA C++ Programming Guide 12.9 §18.9.1 lists no FP6. -/
def peakFp6 : ℝ := 8192

/-- A3, f16: 4096 dense MACs an SM clock. Grade: data sheet per second; per clock derived at the implied clock. Source:
Technical Brief V2.1 Table 3, HGX B200: 'FP16/BF16 Tensor Core 2.2 | 4.5 petaFLOPS'; Lenovo Product Guide Table 2: '2.25
/ 4.5'; measured UTCHMMA_F16_M128_N256 (DC-100). -/
def peakF16 : ℝ := 4096

/-- A3, bf16: 4096 dense MACs an SM clock. Grade: data sheet per second; per clock derived at the implied clock. Source:
Technical Brief V2.1 Table 3, HGX B200: 'FP16/BF16 Tensor Core 2.2 | 4.5 petaFLOPS'. -/
def peakBf16 : ℝ := 4096

/-- A3, tf32: 2048 dense MACs an SM clock. Grade: data sheet per second; per clock derived at the implied clock. Source:
Technical Brief V2.1 Table 3, HGX B200: 'TF32 Tensor Core 1.1 | 2.2 petaFLOPS'; Lenovo Product Guide Table 2: '1.1 /
2.2'. -/
def peakTf32 : ℝ := 2048

/-- A3, fp4: 16384 dense MACs an SM clock. Grade: data sheet per second; per clock derived at the implied clock; no
class. Source: Technical Brief V2.1 Table 3, HGX B200: 'FP4 Tensor Core 9 | 18 petaFLOPS'; PTX ISA 9.4 tcgen05.mma
.kind::mxf4 and .kind::mxf4nvf4 take twice .kind::f8f6f4's K; the Programming Guide 12.9 §18.9.1 lists no FP4, and the
census no FP4 class. -/
def peakFp4 : ℝ := 16384

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

/-- A4, the alu pipe: 64 results an SM clock. Grade: Programming Guide and Nsight Compute. Source: CUDA C++ Programming
Guide 12.9 §6.4.1 Table 7, CC 10.0: 32-bit integer add 64, compare/minimum/maximum 64, bitwise AND/OR/XOR 64, shift 64
(results a clock a multiprocessor); Nsight Compute Kernel Profiling Guide §2.3.6: alu executes 'most bit manipulation
and logic instructions ... integer instructions, excluding IMAD and IMUL'. -/
def widthAlu : ℝ := 64

/-- A4, the imad pipe: 64 results an SM clock. Grade: Programming Guide. Source: CUDA C++ Programming Guide 12.9 §6.4.1
Table 7, CC 10.0: '32-bit integer multiply, multiply-add, extended-precision multiply-add' 64; Nsight Compute §2.3.6:
IMUL and IMAD on FMA Heavy. -/
def widthImad : ℝ := 64

/-- A4, the shfl pipe: 32 results an SM clock. Grade: Programming Guide. Source: CUDA C++ Programming Guide 12.9 §6.4.1
Table 7, CC 10.0: 'warp shuffle' 32; Nsight Compute §2.3.6: lsu issues shuffles. -/
def widthShfl : ℝ := 32

/-- A4: each pipe's results an SM clock. -/
def Pipe.width : Pipe → ℝ
  | .alu => widthAlu
  | .imad => widthImad
  | .shfl => widthShfl

/-- Each class's tensor form: its operand type and its dense MACs a warp instruction, m · n · k over the warps that
issue one MMA (a `wgmma` warpgroup's 4, 1 for `mma.sync` and the single-thread `tcgen05.mma`); `none` off the tensor
pipe. -/
def Kind.form : Kind → Option (Ty × ℝ)
  | .ALU => none
  | .FFMA => none
  | .IMAD => none
  | .IMMA16816_S8 => some (.s8, 16 * 8 * 16 / 1)
  | .IMMA16832_S8 => some (.s8, 16 * 8 * 32 / 1)
  | .LDTM_16X128B_X8 => none
  | .LDTM_16X256B_X4 => none
  | .LDTM_16X64B_X16 => none
  | .LDTM_32X32B_X1 => none
  | .LDTM_32X32B_X128 => none
  | .LDTM_32X32B_X16 => none
  | .LDTM_32X32B_X32 => none
  | .LDTM_32X32B_X4 => none
  | .LDTM_32X32B_X64 => none
  | .SHFL => none
  | .UNI => none
  | .UTCHMMA_F16_M128_N256 => some (.f16, 128 * 256 * 16 / 1)
  | .UTCIMMA_M128_N128 => some (.s8, 128 * 128 * 32 / 1)
  | .UTCIMMA_M128_N256 => some (.s8, 128 * 256 * 32 / 1)
  | .UTCIMMA_M128_N32 => some (.s8, 128 * 32 * 32 / 1)
  | .UTCIMMA_M128_N64 => some (.s8, 128 * 64 * 32 / 1)
  | .UTCIMMA_M64_N128 => some (.s8, 64 * 128 * 32 / 1)
  | .UTCIMMA_M64_N256 => some (.s8, 64 * 256 * 32 / 1)
  | .UTCIMMA_M64_N64 => some (.s8, 64 * 64 * 32 / 1)
  | .UTCQMMA_E4M3_M128_N128 => some (.e4m3, 128 * 128 * 32 / 1)
  | .UTCQMMA_E4M3_M128_N256 => some (.e4m3, 128 * 256 * 32 / 1)

/-- The pipe each class issues to; `none` on the tensor pipe or issue only. -/
def Kind.pipe : Kind → Option Pipe
  | .ALU => some .alu
  | .FFMA => none
  | .IMAD => some .imad
  | .IMMA16816_S8 => none
  | .IMMA16832_S8 => none
  | .LDTM_16X128B_X8 => none
  | .LDTM_16X256B_X4 => none
  | .LDTM_16X64B_X16 => none
  | .LDTM_32X32B_X1 => none
  | .LDTM_32X32B_X128 => none
  | .LDTM_32X32B_X16 => none
  | .LDTM_32X32B_X32 => none
  | .LDTM_32X32B_X4 => none
  | .LDTM_32X32B_X64 => none
  | .SHFL => some .shfl
  | .UNI => none
  | .UTCHMMA_F16_M128_N256 => none
  | .UTCIMMA_M128_N128 => none
  | .UTCIMMA_M128_N256 => none
  | .UTCIMMA_M128_N32 => none
  | .UTCIMMA_M128_N64 => none
  | .UTCIMMA_M64_N128 => none
  | .UTCIMMA_M64_N256 => none
  | .UTCIMMA_M64_N64 => none
  | .UTCQMMA_E4M3_M128_N128 => none
  | .UTCQMMA_E4M3_M128_N256 => none

/-- **The B200's data sheet**: f_max in Hz, each class's MACs and its type's peak (`0` off the tensor pipe), its pipe,
each pipe's width, and no shared-memory class (no LDS kind in the census, so A5 bounds nothing here). -/
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

/-- `census/DS-100.csv`'s `DS_tensor` lines, SMSP-cycles a warp instruction. -/
def tensorLine : Kind → ℝ
  | .IMMA16816_S8 => 1
  | .IMMA16832_S8 => 2
  | .UTCHMMA_F16_M128_N256 => 512
  | .UTCIMMA_M128_N128 => 256
  | .UTCIMMA_M128_N256 => 512
  | .UTCIMMA_M128_N32 => 64
  | .UTCIMMA_M128_N64 => 128
  | .UTCIMMA_M64_N128 => 128
  | .UTCIMMA_M64_N256 => 256
  | .UTCIMMA_M64_N64 => 64
  | .UTCQMMA_E4M3_M128_N128 => 256
  | .UTCQMMA_E4M3_M128_N256 => 512
  | _ => 0

/-- `census/DS-100.csv`'s pipe lines (`DS_<pipe>`), SMSP-cycles a warp instruction. -/
def pipeLine : Pipe → Kind → ℝ
  | .alu, .ALU => 2
  | .imad, .IMAD => 2
  | .shfl, .SHFL => 4
  | _, _ => 0

/-- **`DS-100`'s census** (`census/DS-100.csv`): SMSP-cycles a warp instruction of each class holds each row; every
class holds issue for 1, and a class with no line on a row holds it not at all. -/
def census : SheetRow Pipe → Kind → ℝ
  | .issue, _ => 1
  | .tensor, k => tensorLine k
  | .pipe p, k => pipeLine p k
  | .smem, _ => 0

/-- The sheet's machine is `DS-100`'s census, line for line. -/
theorem machine_eq_census : sheet.machine.a = census := by
  funext r k
  rcases r with _ | _ | p | _
  all_goals try cases p
  all_goals cases k <;> norm_num [Sheet.machine, sheet, census, tensorLine, pipeLine, Kind.form, Kind.pipe, Ty.peak,
      Pipe.width, peakS8, peakE4m3, peakE5m2, peakFp6, peakF16, peakBf16, peakTf32, peakFp4, widthAlu, widthImad,
      widthShfl]

/-- The `implied` clock is under f_max. -/
theorem impliedMHz_lt_fmaxMHz : impliedMHz < fmaxMHz := by
  norm_num [impliedMHz, fmaxMHz]

/-- A3's s8 peak reproduces the document: 2 · 8192 MACs · 148 SMs at the `implied` clock, 1854 MHz, is its figure, dense
4500 ± 50, in 10¹² a second. -/
theorem peakS8_figure :
    (4500 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakS8 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakS8 * smCount * (impliedMHz * 10 ^ 6) ≤ (4500 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakS8, smCount, impliedMHz]

/-- A3's e4m3 peak reproduces the document: 2 · 8192 MACs · 148 SMs at the `implied` clock, 1854 MHz, is its figure,
dense 4500 ± 50, in 10¹² a second. -/
theorem peakE4m3_figure :
    (4500 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakE4m3 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakE4m3 * smCount * (impliedMHz * 10 ^ 6) ≤ (4500 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakE4m3, smCount, impliedMHz]

/-- A3's e5m2 peak reproduces the document: 2 · 8192 MACs · 148 SMs at the `implied` clock, 1854 MHz, is its figure,
dense 4500 ± 50, in 10¹² a second. -/
theorem peakE5m2_figure :
    (4500 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakE5m2 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakE5m2 * smCount * (impliedMHz * 10 ^ 6) ≤ (4500 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakE5m2, smCount, impliedMHz]

/-- A3's fp6 peak reproduces the document: 2 · 8192 MACs · 148 SMs at the `implied` clock, 1854 MHz, is its figure,
dense 4500 ± 50, in 10¹² a second. -/
theorem peakFp6_figure :
    (4500 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakFp6 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakFp6 * smCount * (impliedMHz * 10 ^ 6) ≤ (4500 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakFp6, smCount, impliedMHz]

/-- A3's f16 peak reproduces the document: 2 · 4096 MACs · 148 SMs at the `implied` clock, 1854 MHz, is its figure,
dense 2200 ± 50, in 10¹² a second. -/
theorem peakF16_figure :
    (2200 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakF16 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakF16 * smCount * (impliedMHz * 10 ^ 6) ≤ (2200 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakF16, smCount, impliedMHz]

/-- A3's bf16 peak reproduces the document: 2 · 4096 MACs · 148 SMs at the `implied` clock, 1854 MHz, is its figure,
dense 2200 ± 50, in 10¹² a second. -/
theorem peakBf16_figure :
    (2200 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakBf16 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakBf16 * smCount * (impliedMHz * 10 ^ 6) ≤ (2200 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakBf16, smCount, impliedMHz]

/-- A3's tf32 peak reproduces the document: 2 · 2048 MACs · 148 SMs at the `implied` clock, 1854 MHz, is its figure,
dense 1100 ± 50, in 10¹² a second. -/
theorem peakTf32_figure :
    (1100 - 50 : ℝ) * 10 ^ 12 ≤ 2 * peakTf32 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakTf32 * smCount * (impliedMHz * 10 ^ 6) ≤ (1100 + 50 : ℝ) * 10 ^ 12 := by
  norm_num [peakTf32, smCount, impliedMHz]

/-- A3's fp4 peak reproduces the document: 2 · 16384 MACs · 148 SMs at the `implied` clock, 1854 MHz, is its figure,
dense 9000 ± 500, in 10¹² a second. -/
theorem peakFp4_figure :
    (9000 - 500 : ℝ) * 10 ^ 12 ≤ 2 * peakFp4 * smCount * (impliedMHz * 10 ^ 6) ∧
      2 * peakFp4 * smCount * (impliedMHz * 10 ^ 6) ≤ (9000 + 500 : ℝ) * 10 ^ 12 := by
  norm_num [peakFp4, smCount, impliedMHz]

end

end Accounting.TS.Sheets.B200
