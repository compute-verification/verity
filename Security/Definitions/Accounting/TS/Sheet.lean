import Mathlib

/-!
# A device's data sheet (trusted)

The constants a device's vendor documents give on their own (data-sheet constants, no GPU self-attestation), over
instruction classes `K` and Nsight Compute pipes `P`: the maximum boost clock, each class's dense tensor MACs, its
operand type's dense peak, its pipe and its shared-memory bytes, and each pipe's results an SM clock.

`Accounting.TS.Sheets` gives them at each device's vendor values.
-/

namespace Accounting.TS

/-- **A device's data sheet** over classes `K` and pipes `P`: its maximum boost clock `fmax`; per class, the dense
tensor MACs `macs k` one warp instruction does (`0` off the tensor pipe; a sparse form at its dense equivalent; a
tensor form whose operand type the sheet doesn't rate has `macs = 0`, so it costs issue only), the sheet's dense MACs
an SM clock `peak k` for its operand type, the Nsight Compute pipe `pipe k` it issues to (`none`: issue only) and the
shared-memory bank bytes `smem k` it holds at least (`0` off shared memory); per pipe, the Programming Guide's results
an SM clock `width p`. -/
structure Sheet (K P : Type) where
  /-- The top of the declared vBIOS's clock table, which no SM clock on the node exceeds (`accounting-declared-device`):
  the data sheet's maximum boost clock (the H200's 1,980 MHz), or the driver's reported maximum where the sheet gives
  none for the board (sm_120's Server card, 2,430 MHz, measured as the table's top; the B200's 1,965 MHz).
  The H200's and B200's tables are unmeasured. -/
  fmax : ℝ
  macs : K → ℝ
  peak : K → ℝ
  pipe : K → Option P
  width : P → ℝ
  smem : K → ℝ

end Accounting.TS
