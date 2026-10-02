import FlockSoundness.Discharge.Layout.Readers
import FlockSoundness.Discharge.Layout.Bytes
import FlockSoundness.Discharge.Layout.Region
import FlockSoundness.Discharge.Layout.HmRegion
import FlockSoundness.Discharge.Layout.Pins
import FlockSoundness.Discharge.Layout.Drawn
import FlockSoundness.Discharge.Layout.QSort
import FlockSoundness.Discharge.Layout.Site
import FlockSoundness.Discharge.Layout.OfRows
import FlockSoundness.Discharge.Layout.KeyProg

/-!
# `lay` and `rs` of `Prog.flock_headline`, discharged

The layout's facts (`Binding.Layout.decode_row`, `Binding.HmRows.bc_registered`, `Binding.HmRows.other_registered`)
from the decoder's places and the executable's regions:
* `Readers`: the decoder's bits at fixed addresses (`decode_row_of_cols`, `ofBits`);
* `Bytes`: bytes from their bits (`bytesAt_eq`);
* `Region`: what a region of the executable's statement pins in a satisfying witness, and where (`region_read`,
  `inBlock_mkRegion`, `inBlock_lt`);
* `HmRegion`: the `b ‖ c` the statement pins at the `hm96` slots' outputs (`bc_pinned`, at `bcAddr`, the slot's output
  column by `bcAddr_eq_slotCol`), and setupH's regions (`setupH_regions`);
* `Pins`: `lay` and `rs` from fixed readers (`Readers`) and three facts of every satisfying witness: Δ's copies
  (`Readers.Copies`), the rows' pinned `b ‖ c` (`Readers.BcPinned`) and the other pinned values (`Readers.ValPinned`);
* `Drawn`: at a drawn statement, the pinned `b ‖ c` of a proved unit's `hm96` slot is the unit's in the registered
  population's public file (`bc_drawn`);
* `QSort`: every proved unit of a draw is in the population, with or without a closure (`proved_lt_population`, by
  `all_qsort`: `Array.qsort` keeps a property of every element);
* `Site`: at a drawn unit's site in its table's accepted statement, the population's `b ‖ c` at port `p`'s `hm96` output
  (`bc_site`) and, at each message-keyed unit input, its row's bit (`copy_site`);
* `OfRows`: readers whose commit strings are units' row ports and single gates (`Readers.ofRows`), with Δ's copies
  from the row bits alone (`ofRows_copies`), the rows' `b ‖ c` (`ofRows_bcPinned`) and the single gates' bits
  (`ofRows_valPinned`; 1 on the constant's gates and 0 on the zero's, `gate_pub`);
* `KeyProg`: those readers at [e2e-placed]'s places of the circuit's program on Δ's source keys (`Placed.keyProg`),
  with rows the message-keyed inputs (`keyRowOf`): Δ's copies (`keyProg_copies`), the rows' `b ‖ c`
  (`keyProg_bcPinned`), the single gates' bits (`keyProg_valPinned`), and `lay` and `rs` there (`keyLayout`,
  `keyRows`).
-/
