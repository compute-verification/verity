import Definitions.Core.Accel.Machine

/-!
# The H200 SXM's numbers (trusted)

Each number is `verity/hardware/cards/h200-sxm-141gb.toml`'s, with its source and its mark (ceiling, measurement or
placeholder) there; `verity/hardware/tests/test_lean.py` holds this table to that file number by number.

A table is a SKU's, at its firmware's maximum. This one is the H200 SXM 141 GB (GH100, sm_90a; board
692-2G520-0280-001, vBIOS 96.00.A5.00.03, 700 W): 132 SMs (H100 whitepaper V1.04 Table 3, H100 SXM5) at 1980 MHz SM
and 3201 MHz HBM3e, the card's maximum and application clocks (Table 3: 'GPU Boost Clock … 1980 MHz', footnote 2 'GPU
Peak Clock and GPU Boost Clock are synonymous'). Its supported clocks top at those two, and its SM and memory clock
offset ranges are [0, 0]. Each rate is at its clock's ceiling, not at a clock a loaded card holds: the tensor's at
1830 MHz (`tensorClock`), every other resource's at 1980 MHz (`clock`). Under its power limit a loaded card holds lower
clocks. Which SKU and firmware a node's cards are is part of the premise that reads the table
(`Assumptions.CardWithinAccel`); nothing the protocol checks attests it. Data-sheet numbers until measurements replace
them.

**The tensor's clock**, on two readings:
* 1830 MHz, the table's: the tensor boost clock the data sheet quotes its dense tensor peaks at on this firmware (its
  1,979 dense INT8 TOPS are 2 · 132 SMs · 4096 MACs at 1830 MHz);
* 1980 MHz: the card's maximum boost clock, which the driver reports and every other resource is taken at.

Each figure the tensor's clock moves, at 1830 MHz and then at 1980 MHz: the `s8` tensor rate, 9.894 · 10¹⁴ and
1.0705 · 10¹⁵ MACs a second; FP64's (DMMA) weight, 59/2 and 32 units a MAC; a table method reading a word of L2 a MAC,
13.5% and 12.5% of the `s8` rate; a route of 64 products a SM-clock (an `alu` lane a MAC, the texture path), 1.7% and
1.6%.

The tensor-duty sweep (`benchmarks/tensor_duty/`) is the 1830 MHz reading's falsification test: a run above
132 · 4096 · 1830 · 10⁶ = 9.894 · 10¹⁴ `s8` MACs a second over the host's wall clock refutes it. Throughput rests on
the host's wall clock alone; the GPU's own timers (clock64, the global timer, CUDA events) are diagnostics. Its run
art:f1bb35d70476, with the SM clock locked at 1980 MHz and power at the board's 700 W maximum, over zero, constant,
uniform and random operands (`s8` on `wgmma` and `mma.sync`, `e4m3` on `wgmma`) and tensor duty cycles from 50% to
100% (1% steps above 90%, the rest idle or FP32 work), read at most 988.59 · 10¹² `s8` MACs a second: 0.99915 of the
1830 MHz rate and 0.92346 of the 1980 MHz one, and under the 1830 MHz rate at the host stamps' millisecond resolution.
DMMA keeps the 1980 MHz the data sheet quotes its FP64 tensor peak at, through its weight.

**Stores** (32-bit words): the register file, 64 Ki registers an SM, plus the uniform registers and the predicates
(4494 words an SM); shared memory with L1, 256 KB an SM; L2, 50 MiB; HBM3e, 141 GiB (H200 data sheet, 'GPU Memory
141GB').

**Rates**, a SM-clock each (CUDA C++ Programming Guide 12.9 §6.4.1 Table 7, CC 9.0, unless named):
* tensor: 4096 `s8` MACs a clock of `tensorClock`, 989.4 · 10¹² a second (1070.5 · 10¹² at 1980 MHz; the data
  sheet's 1,979 dense INT8 TOPS over 2 · 132 SMs at 1830 MHz; measured 4096 a SM-clock), whichever instruction issues
  them (`mma.sync` measured at 65.6% of `wgmma`); a MAC of `e4m3` or `e5m2` uses 1 unit (the same peak), `f16` or
  `bf16` 2, `tf32` 4, FP64 (DMMA) 59/2 (67 TFLOPS over 132 SMs at the 1980 MHz FP64 boost clock is 128 MACs a clock of
  1980 MHz, 29.58 units a MAC at 1830 MHz, rounded down; 32 at 1980 MHz), `b1` 1/8 (measured); the FP4 formats, which
  the card lacks, 1 (an op the card lacks only adds programs);
* issue: 4 warp instructions (four schedulers, one a clock each);
* `alu`, `imad`: 64 lanes each; `shfl` 32 (measured); `f64` 64; `sfu` 16 (reciprocal, square root, logarithm,
  exponential, sine and cosine);
* shared memory: 32 bank words; L2: 512 words read or written, a placeholder, neither measured nor a ceiling (the
  lane loads' issue bound; open until a measured ceiling with its argument replaces it). At 512 a table method that
  reads a word of L2 a MAC runs at 13.5% of the tensor's `s8` rate (12.5% at 1980 MHz);
* HBM3e: 3201 MHz at two transfers a clock on 6144 bits (six stacks), 4.917 TB/s, above the data sheet's 4.8 TB/s;
  1.229184 · 10¹² words a second.
-/

open scoped NNRat

namespace Verity.Accel.H200

/-- The clock's ceiling, Hz: every resource's but the tensor's. -/
def clock : ℚ≥0 := 1980 * 10 ^ 6

/-- The tensor's clock's ceiling, Hz: the data sheet's tensor boost clock. At the second reading, `clock`, the tensor
would be 1.0705 · 10¹⁵ `s8` MACs a second; `benchmarks/tensor_duty/` is this one's falsification test. -/
def tensorClock : ℚ≥0 := 1830 * 10 ^ 6

/-- **The H200 SXM's table.** -/
def card : Card where
  size
    | .reg => 132 * (65536 + 4494)
    | .smem => 132 * 65536
    | .l2 => 50 * 2 ^ 20 / 4
    | .dram => 141 * 2 ^ 30 / 4
  size_pos := by intro s; cases s <;> decide
  ρ
    | .tensor => 132 * 4096 * tensorClock
    | .issue => 132 * 4 * clock
    | .alu | .imad | .f64 => 132 * 64 * clock
    | .shfl | .smem => 132 * 32 * clock
    | .sfu => 132 * 16 * clock
    | .l2 => 132 * 512 * clock
    | .dram => 1229184 * 10 ^ 6
  weight
    | .f16 | .bf16 => 2
    | .tf32 => 4
    | .f64 => 59 / 2
    | .b1 => 1 / 8
    | _ => 1
  exact := 31
  text := 141 * 2 ^ 30 / 16

end Verity.Accel.H200
