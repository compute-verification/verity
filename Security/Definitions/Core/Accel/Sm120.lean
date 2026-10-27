import Definitions.Core.Accel.Machine

/-!
# The RTX PRO 6000 Server Edition's numbers (trusted)

A table is a SKU's, at its firmware's maximum. This one is the RTX PRO 6000 Blackwell Server Edition (GB202, sm_120a,
188 SMs; board 900-2G153-0000-000, vBIOS 98.02.8D.00.01): 2430 MHz SM and 12481 MHz GDDR7. Its supported clocks top
at those two, its SM and memory clock offset ranges are [0, 0], and every lock, application clock and offset above
them is refused or clamped. Another SKU is another table: the Workstation Edition boosts to 2617 MHz with 1792 GB/s,
and NVIDIA's family headline (4 PFLOPS sparse FP4) is at about 2600 MHz. Which SKU and firmware a node's cards are is
part of the premise that reads the table (`Assumptions.CardWithinAccel`); nothing the protocol checks attests it (open:
an attestation of the SKU and the vBIOS). Data-sheet and measured numbers.

**Stores** (32-bit words): the register file, 64 Ki registers an SM, plus the uniform registers and the predicates
(3371 words an SM); shared memory with L1, 128 KB an SM; L2, 128 MiB; GDDR7, 96 GiB.

**Rates**, a SM-clock each (CUDA C++ Programming Guide 12.9 §6.4.1 Table 7, CC 12.0, unless named):
* tensor: 1024 `s8` MACs (`IMMA.16832`, measured at 4096 MACs in 16 tensor cycles of a sub-partition); a MAC of
  `e4m3`, `e5m2` or `e2m1` uses 1 unit, `nvf4` or `mxf4` 1/2, `f16` or `bf16` 2, `tf32` 4, FP64 (DMMA) 32 (measured
  1.75 MACs an SM-clock, so generous), `b1`, which the card lacks, 1;
* issue: 4 warp instructions; `alu`, `imad`: 64 lanes each (a SASS census); `shfl` 32 (measured); `f64` 2; `sfu` 16;
* shared memory: 32 bank words; L2: 5 words read or written, a measured value, not a ceiling (18.6 bytes an SM-clock
  of one read pattern at 2092 MHz; open until a ceiling with its argument replaces it);
* GDDR7: 12481 MHz at two transfers a clock on 512 bits, 1597.6 GB/s (the data sheet's 1597 GB/s),
  3.99392 · 10¹¹ words a second.
-/

open scoped NNRat

namespace Verity.Accel.Sm120

/-- The clock's ceiling, Hz. -/
def clock : ℚ≥0 := 2430 * 10 ^ 6

/-- **The RTX PRO 6000 Server Edition's table.** -/
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
    | .dram => 399392 * 10 ^ 6
  weight
    | .nvf4 | .mxf4 => 1 / 2
    | .f16 | .bf16 => 2
    | .tf32 => 4
    | .f64 => 32
    | _ => 1
  exact := 31
  text := 96 * 2 ^ 30 / 16

end Verity.Accel.Sm120
