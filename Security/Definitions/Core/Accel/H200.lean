import Definitions.Core.Accel.Machine

/-!
# The H200 SXM's numbers (trusted)

Each number is `verity/hardware/cards/h200-sxm-141gb.toml`'s, with its source and its mark (ceiling, measurement or
placeholder) there; `verity/hardware/tests/test_lean.py` holds this table to that file number by number.

A table is a SKU's, at its firmware's maximum. This one is the H200 SXM 141 GB (GH100, sm_90a; board
692-2G520-0280-001, vBIOS 96.00.A5.00.03, 700 W): 132 SMs (H100 whitepaper V1.04 Table 3, H100 SXM5) at 1980 MHz SM
and 3201 MHz HBM3e, the card's maximum and application clocks (Table 3: 'GPU Boost Clock … 1980 MHz', footnote 2 'GPU
Peak Clock and GPU Boost Clock are synonymous'). Its supported clocks top at those two, and its SM and memory clock
offset ranges are [0, 0]. Which SKU and firmware a node's cards are is part of the premise that reads the table
(`Assumptions.CardWithinAccel`); nothing the protocol checks attests it. Data-sheet numbers until measurements replace
them.

**Stores** (32-bit words): the register file, 64 Ki registers an SM, plus the uniform registers and the predicates
(4494 words an SM); shared memory with L1, 256 KB an SM; L2, 50 MiB; HBM3e, 141 GiB (H200 data sheet, 'GPU Memory
141GB').

**Rates**, a SM-clock each (CUDA C++ Programming Guide 12.9 §6.4.1 Table 7, CC 9.0, unless named):
* tensor: 4096 `s8` MACs (the data sheet's 1,979 dense INT8 TOPS over 2 · 132 SMs at 1830 MHz; taken at 1980 MHz),
  whichever instruction issues them (`mma.sync` measured at 65.6% of `wgmma`); a MAC of `e4m3` or `e5m2` uses 1 unit
  (the same peak), `f16` or `bf16` 2, `tf32` 4, FP64 (DMMA) 32 (67 TFLOPS over 132 SMs at 1980 MHz), `b1` 1/8
  (measured); the FP4 formats, which the card lacks, 1 (an op the card lacks only adds programs);
* issue: 4 warp instructions (four schedulers, one a clock each);
* `alu`, `imad`: 64 lanes each; `shfl` 32 (measured); `f64` 64; `sfu` 16 (reciprocal, square root, logarithm,
  exponential, sine and cosine);
* shared memory: 32 bank words; L2: 512 words read or written, a placeholder, neither measured nor a ceiling (the
  lane loads' issue bound; open until a measured ceiling with its argument replaces it). At 512 a table method that
  reads a word of L2 a MAC runs at 12.5% of the tensor's `s8` rate;
* HBM3e: 3201 MHz at two transfers a clock on 6144 bits (six stacks), 4.917 TB/s, above the data sheet's 4.8 TB/s;
  1.229184 · 10¹² words a second.
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
    | .dram => 1229184 * 10 ^ 6
  weight
    | .f16 | .bf16 => 2
    | .tf32 => 4
    | .f64 => 32
    | .b1 => 1 / 8
    | _ => 1
  exact := 31
  text := 141 * 2 ^ 30 / 16

end Verity.Accel.H200
