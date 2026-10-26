import Definitions.Core.Accel.Machine

/-!
# The RTX PRO 6000's numbers (trusted)

The RTX PRO 6000 Blackwell Server Edition (GB202, sm_120a): 188 SMs at the clock's ceiling, 2430 MHz (the hardware's
maximum, which the prover may set: note:20261017T0324Z-draft-eps1-spec). Data-sheet and CALIB numbers.

**Stores** (32-bit words): the register file, 64 Ki registers an SM, plus the uniform registers and the predicates
(3371 words an SM); shared memory with L1, 128 KB an SM; L2, 128 MiB; GDDR7, 96 GiB.

**Rates**, a SM-clock each (CUDA C++ Programming Guide 12.9 §6.4.1 Table 7, CC 12.0, unless named):
* tensor: 1024 `s8` MACs (`IMMA.16832`, 4096 MACs in 16 tensor cycles of a sub-partition, run r20261017-041103-d35d);
  a MAC of `e4m3`, `e5m2` or `e2m1` uses 1 unit, `nvf4` or `mxf4` 1/2, `f16` or `bf16` 2, `tf32` 4, FP64 (DMMA) 32
  (measured 1.75 MACs an SM-clock, so generous), `b1`, which the card lacks, 1;
* issue: 4 warp instructions; `alu`, `imad`: 64 lanes each (the DS-120 census); `shfl` 32 (CALIB); `f64` 2; `sfu` 16;
* shared memory: 32 bank words; L2's reads: 5 words (CALIB measured 18.6 bytes an SM-clock at 2092 MHz, run
  r20261017-065855-c8a9);
* GDDR7: 1597 GB/s (data sheet), 3.9925 · 10¹¹ words a second.
-/

open scoped NNRat

namespace Verity.Accel.Sm120

/-- The clock's ceiling, Hz. -/
def clock : ℚ≥0 := 2430 * 10 ^ 6

/-- **The RTX PRO 6000's table.** -/
def card : Card where
  size
    | .reg => 188 * (65536 + 3371)
    | .smem => 188 * 32768
    | .l2 => 128 * 2 ^ 20 / 4
    | .dram => 96 * 2 ^ 30 / 4
  size_pos := by intro s; cases s <;> decide
  ρ
    | .tensor => 188 * 1024 * clock
    | .issue => 188 * 4 * clock
    | .alu | .imad => 188 * 64 * clock
    | .shfl | .smem => 188 * 32 * clock
    | .f64 => 188 * 2 * clock
    | .sfu => 188 * 16 * clock
    | .l2 => 188 * 5 * clock
    | .dram => 39925 * 10 ^ 7
  weight
    | .nvf4 | .mxf4 => 1 / 2
    | .f16 | .bf16 => 2
    | .tf32 => 4
    | .f64 => 32
    | _ => 1
  exact := 31
  text := 96 * 2 ^ 30 / 16

end Verity.Accel.Sm120
