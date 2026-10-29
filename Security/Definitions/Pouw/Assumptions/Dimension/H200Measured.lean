import Definitions.Pouw.Assumptions.Dimension.H100Assumptions

/-!
# `MH100`'s price table measured on the H200 SXM (draft semantics)

**Draft semantics, pending review**.

`h200Measured` is the table measured on one H200 SXM (GH100, `sm_90a`, CUDA 13.0) by Round 8's loop-free method
(`benchmarks/pouw/dc/h32.py` over the `sm90_h32*.cu` probes), per 32-bit register written. Its unit is `Prices100`'s:
one dense int8 MAC at the data sheet's 4,096 per SM per clock. Each entry is the least price measured in its class,
rounded down to a natural number, so it is a measurement, not a floor. Every class sits at its bound, so the table is
`h100Floor` (`h200MeasuredFloor`), and `h32H200Measured` proves `H32 h200Measured` by `decide`, with no slack
(`Pouw.Dimension.H200MeasuredProofs`).

**What the H100's tables left unmeasured.** The immediate forms (IADD a+K, HADD2 a+K), m8n8k4 and sparse `mma.sp`
were measured, and none is below 32. On `sm_90a`, CUDA 13.0 has no tensor-pipe m8n8k4 with `f16` inputs: it calls a
scalar routine whose instructions each price in their own class. f64 m8n8k4 is DMMA.

**Work, not time.** As on the H100, writers on separate pipes co-issue below 32 per register (`H32`'s docstring), and
so does `wgmma` at its full rate beside IMAD and FFMA lanes. A bound on `cost100` at these prices bounds additive-W1
work. A time bound needs a maximum over resources.
-/

namespace Pouw.Dimension.H100

/-- **The measured H200 table** (draft semantics), each entry rounded down:

* `mac`: `nominalMac`. Dense `wgmma` at `N ≥ 64` runs at the data-sheet rate for every `Shape100` shape, 32 per
  register, and 64 with FP16 accumulate (32 per f16 word).
* `tc = 32`: b1 `wgmma` and legacy `mma.sync` m16n8k8 f16 or bf16 → f32, m16n8k4 tf32, m16n8k16 s8 and m16n8k128 b1.
* `ffma = 32`: FADD with an immediate, at dispatch.
* `int32 = 32`: IMAD beside LOP3, and IADD with an immediate, at dispatch.
* `fp64 = 32`: DADD and DMUL, the FP64 pipe's bound.
* `other = 32`: half2 HADD2 beside PRMT, and two-source HADD2, HMUL2 and HFMA2, which ptxas splits across two pipes. -/
def h200Measured : Prices100 where
  mac := nominalMac
  tc := 32
  ffma := 32
  int32 := 32
  fp64 := 32
  other := 32

end Pouw.Dimension.H100
