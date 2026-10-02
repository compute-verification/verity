import FlockSoundness.Discharge.Layout.Readers
import FlockSoundness.Discharge.Layout.Bytes
import FlockSoundness.Discharge.Layout.Region
import FlockSoundness.Discharge.Layout.HmRegion

/-!
# `lay` and `rs` of `Prog.flock_headline`, discharged

The layout's facts (`Binding.Layout.decode_row`, `Binding.HmRows.bc_registered`, `Binding.HmRows.other_registered`)
from the decoder's places and the executable's regions:
* `Readers`: the decoder's bits at fixed addresses (`decode_row_of_cols`, `ofBits`);
* `Bytes`: bytes from their bits (`bytesAt_eq`);
* `Region`: what a region of the executable's statement pins in a satisfying witness, and where (`region_read`,
  `inBlock_mkRegion`, `inBlock_lt`);
* `HmRegion`: the `b ‖ c` the statement pins at the `hm96` slots' outputs (`bc_pinned`, at `bcAddr`, the slot's output
  column by `bcAddr_eq_slotCol`), and setupH's regions (`setupH_regions`).
-/
