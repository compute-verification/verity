import Models.Pouw.Assumptions.Dimension.H100Assumptions

/-!
# `MH100`'s loop-free price table (Round 8; draft semantics)

**Draft semantics, pending review** (`internal/pouw-fp8/genuine-fp8-interface.md` §0 and §1.1; the measurements are
`internal/pouw/gpu-constants.md` §13, run `r20260928-161529-c85c`).

`h100LoopFree` is the table that Round 8 measured on one H100 SXM, unrolled so that the loop's own instructions no
longer count, per 32-bit register written. Its unit is `Prices100`'s: one dense int8 MAC at the datasheet's 4,096 per
SM per clock. Each entry is the least loop-free price measured in its class, rounded down to a natural number. So no
entry exceeds its measurement, and a lower bound on `cost100` at these prices holds for programs of the measured
forms. Round 7's `h100Measured` is kept as it was measured.

**A measurement, not a floor** (red team F18b-2). `int32 = 34` is two-source IADD and `other = 36` is two-source
HADD2. The immediate forms (IADD a+K, HADD2 a+K) were never measured and could reach 32, the dispatch bound for a
writer of one register. The floor is `h100Floor` (`Pouw.Dimension.H100Floor`): this table with `int32 = other = 32`,
a floor except at `tc`, which both tables take from this board's legacy `mma.sync` (red team F18c-1). The hardware
citations are `DistinctWritesLoopFree` (these prices, for the measured forms) and `DistinctWritesFloor` (every scalar
form and every `wgmma` shape, and legacy tensor-core forms as measured).

`h32LoopFree` (`Pouw.Dimension.H100LoopFreeProofs`) proves `H32 h100LoopFree` by `decide`. It has no slack: `tc`,
`ffma` and `fp64` are exactly 32.

**Work, not time.** `H32` holds per instruction class, not per register in time. Writers on separate pipes co-issue:
1:1 DADD + FADD writes a register every 26.75 units with two register sources, and as often as every 21.45 loop-free
with one source and an immediate each, and 21.33 at full dispatch (the FP64 pipe's bound). So a bound on `cost100` at
these prices bounds additive-W1 work. A time bound needs a maximum over resources (`docs/pouw/w1-time-model.md`).
-/

namespace Pouw.Dimension.H100

/-- **The loop-free H100 table** (Round 8, draft semantics), each entry rounded down. It is a measurement, not a
floor: `int32` and `other` are the least measured forms, and the floor is `h100Floor`.

* `mac`: `nominalMac`. Dense `wgmma` at `N ≥ 64` runs at the datasheet rate (tensor alone at 4,096.0), 32.00 per
  register for every `Shape100` shape and 64 with FP16 accumulate.
* `tc = 32`: legacy `mma.sync` at 128 MMAs per loop costs 32.00 for m16n8k8 f16 or bf16 → f32, m16n8k4 tf32,
  m16n8k16 s8 and m16n8k128 b1. m8n8k16 s8 and m16n8k8 f16 → f16 cost 64.00, and m16n8k16 f16 → f32 48.06.
* `ffma = 32`: FADD, FMUL and two-source FFMA at 32.0–32.1. Three-source FFMA costs 51.8.
* `int32 = 34`: two-source IADD at 34.6 (ptxas splits it across IMAD.IADD and IADD3). IMAD.WIDE costs 64.0.
* `fp64 = 32`: DADD and DMUL at 32.0, the FP64 pipe's bound. DFMA costs 44.7.
* `other = 36`: HADD2, HMUL2 and two-source HFMA2 at 36.01. Three-source HFMA2 costs 60.39. The F2FP e4m3 pack costs
  64.0 per instruction, which is 32 per code, and the f16x2 route 64 per code. -/
def h100LoopFree : Prices100 where
  mac := nominalMac
  tc := 32
  ffma := 32
  int32 := 34
  fp64 := 32
  other := 36

end Pouw.Dimension.H100
