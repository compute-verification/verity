import Models.Pouw.Assumptions.Dimension.H100Assumptions

/-!
# `MH100`'s measured price table (Round 7; draft semantics)

**Draft semantics, pending review** (`internal/pouw-fp8/genuine-fp8-interface.md` §0 and §1.1; the measurements are
`internal/pouw/gpu-constants.md` §12.11).

`h100Measured` is the table that Round 7 measured on one H100 SXM (28 Sep 2026; driver 580.126.09, ptxas 12.4.131),
per 32-bit register written. Its unit is `Prices100`'s: one dense int8 MAC at the datasheet's 4,096 per SM per clock,
which m64n128k32 and m64n64k32 e4m3 reach to within 2 ppm. Each entry is the least price of its class that Round 7
measured, rounded down to a natural number. It is a measurement, not a floor, and since Round 8 it does not bound the
hardware from below (below).

`h32Measured` (`Pouw.Dimension.H100MeasuredProofs`) proves `H32 h100Measured` by `decide`, so a re-measured entry
changes only this definition. `DistinctWritesMeasured` (`Pouw.Dimension.H100Cite`) is the distinct-writes bound at
these prices, with no hypothesis on the tensor-core instructions' multiplicands (red team F18-2).

**Round 8 found part of this table to be loop overhead** (`internal/pouw/gpu-constants.md` §13, run
`r20260928-161529-c85c`). Unrolled, FADD's 33.5 is 32.0–32.1 and legacy k8's 34.0 is 32.00. DADD's 34.6 is 32.0,
half2's 40.1 is 36.0 and F2FP's 66.7 is 64.0.

**The largest gap is `int32`, and it is not loop overhead** (red team F18b-1). Two-source IADD measured 34.68 against
this table's 64, because ptxas now splits it across IMAD.IADD and IADD3, a lowering Round 7 did not see. Round 8 also
measured FADD at 32.29, DADD at 32.16 and half2 at 36.11 before any loop-free correction, all below this table's
entries. So a program's hardware cost can be below its `cost100` at this table, and bounds at it do not transfer to
the hardware. The table is kept as Round 7 measured it, so that no record that reads it moves. For the hardware, cite
`h100LoopFree` (Round 8's measurement) or `h100Floor` (the floor, except at `tc`), through `DistinctWritesLoopFree` or
`DistinctWritesFloor` (`Pouw.Dimension.H100Floor`).

**Caveats.**
* **Prices are additive work, not time.** Mixed kernels overlap in time. In Round 8's fixed `conc` (§13.6), `wgmma`
  keeps 98.3–100% of its rate beside IMAD, HFMA2 or F2FP chains (overlap 1.43–1.50), so such a kernel writes a
  register every 25.94–26.84 units. Beside FADD chains it keeps 37.2% (overlap 1.16). A bound on `cost100` at these
  prices bounds work. A bound on time needs a maximum over the tensor core and the CUDA-core pipes.
* **Co-issue of two-register writers** was not measured in Round 7. Round 8 measured it: mixes go below 32 per
  register, as low as 21.45 loop-free, and 21.33 at full dispatch (`H32`'s docstring).
* **Not measured on the H100:** the predicate and carry writers, and the uniform datapath.
-/

namespace Pouw.Dimension.H100

/-- **The measured H100 table** (Round 7, draft semantics), each entry rounded down. It is a measurement, not a floor,
and since Round 8 it does not bound the hardware from below (the module docstring):

* `mac`: dense `wgmma` at `N ≥ 64` runs at the datasheet rate, 32.00 per register for every `Shape100` shape (FP8 and
  int8 at k32, FP16 and BF16 at k16, TF32 at k8) and 64.0 with FP16 accumulate (32 per f16 word). That is
  `nominalMac`. m64n8k32 e4m3 → f32, C-2's atom shape, costs 144 per register; it is not its shape's least price, so it
  does not set the floor.
* `tc = 32`: b1 `wgmma` (m64n256k256 and.popc) at 32.00, the least of the other tensor-core instructions. Legacy
  m16n8k8 f16 → f32 costs 34.0, m16n8k16 s8 35.0, m16n8k16 f16 → f32 50.0, and sparse `wgmma` 36.0.
* `ffma = 33`: FADD and FMUL at 33.5 (FFMA 52.5).
* `int32 = 64`: IMAD.WIDE at 64.1 per register and IMAD at 64.3 (the rest 64.3–66.0).
* `fp64 = 34`: DADD and DMUL at 34.6 per 32-bit register (DFMA 47.5).
* `other = 40`: half2 (HADD2, HMUL2) at 40.1. bf16x2 costs 64, F2FP 66.7 per instruction, compare and select 65.1,
  other conversions 65.0, and memory-side reductions about 950 per word. -/
def h100Measured : Prices100 where
  mac := nominalMac
  tc := 32
  ffma := 33
  int32 := 64
  fp64 := 34
  other := 40

end Pouw.Dimension.H100
