import Definitions.Core.Accel.Machine

/-!
# The H200 SXM's numbers (trusted)

The H200 SXM (GH100, sm_90a): 132 SMs (H100 whitepaper V1.04 Table 3, H100 SXM5) at the clock's ceiling, 1980 MHz
(Table 3: 'GPU Boost Clock … 1980 MHz', footnote 2 'GPU Peak Clock and GPU Boost Clock are synonymous'; the driver's
maximum on the deployed node, note:20261017T0452Z-draft-eps1-dc). Data-sheet numbers until CALIB (CMP-115) measures them.

**Stores** (32-bit words): the register file, 64 Ki registers an SM, plus the uniform registers and the predicates
(4494 words an SM, RT-MODEL M4); shared memory with L1, 256 KB an SM; L2, 50 MiB; HBM3e, 141 GiB (H200 data sheet,
'GPU Memory 141GB').

**Rates**, a SM-clock each (CUDA C++ Programming Guide 12.9 §6.4.1 Table 7, CC 9.0, unless named):
* tensor: 4096 `s8` MACs (the data sheet's 1,979 dense INT8 TOPS over 2 · 132 SMs at 1830 MHz; taken at 1980 MHz,
  RT-MODEL M14), whichever instruction issues them (`mma.sync` measured at 65.6% of `wgmma`, CMP-115); a MAC of
  `e4m3` or `e5m2` uses 1 unit (the same peak), `f16` or `bf16` 2, `tf32` 4, FP64 (DMMA) 32 (67 TFLOPS over 132 SMs at
  1980 MHz), `b1` 1/8 (CALIB, CMP-115); the FP4 formats, which the card lacks, 1 (an op the card lacks only adds
  programs);
* issue: 4 warp instructions (four schedulers, one a clock each);
* `alu`, `imad`: 64 lanes each; `shfl` 32 (CALIB); `f64` 64; `sfu` 16 (reciprocal, square root, logarithm,
  exponential, sine and cosine);
* shared memory: 32 bank words; L2: 512 words read or written, a placeholder, neither measured nor a ceiling (the
  lane loads' issue bound; RM6, open until CALIB's ceiling with its argument replaces it). At 512 a table method that
  reads a word of L2 a MAC runs at 12.5% of the tensor's `s8` rate (run r20261027-010606-f6d5);
* HBM: 4.8 TB/s (data sheet), 1.2 · 10¹² words a second.
-/

open scoped NNRat

namespace Verity.Accel.H200

/-- The clock's ceiling, Hz. -/
def clock : ℚ≥0 := 1980 * 10 ^ 6

/-- **The H200 SXM's table.** -/
def card : Card where
  size
    | .reg => 132 * (65536 + 4494)
    | .smem => 132 * 65536
    | .l2 => 50 * 2 ^ 20 / 4
    | .dram => 141 * 2 ^ 30 / 4
  size_pos := by intro s; cases s <;> decide
  ρ
    | .tensor => 132 * 4096 * clock
    | .issue => 132 * 4 * clock
    | .alu | .imad | .f64 => 132 * 64 * clock
    | .shfl | .smem => 132 * 32 * clock
    | .sfu => 132 * 16 * clock
    | .l2 => 132 * 512 * clock
    | .dram => 12 * 10 ^ 11
  weight
    | .f16 | .bf16 => 2
    | .tf32 => 4
    | .f64 => 32
    | .b1 => 1 / 8
    | _ => 1
  exact := 31
  text := 141 * 2 ^ 30 / 16

end Verity.Accel.H200
