import Proofs.Flock.Soundness.Discharge.Layout.Readers
import Proofs.Flock.Soundness.Discharge.Layout.Bytes
import Proofs.Flock.Soundness.Discharge.Layout.Region
import Proofs.Flock.Soundness.Discharge.Layout.HmRegion
import Proofs.Flock.Soundness.Discharge.Layout.OutRegion
import Proofs.Flock.Soundness.Discharge.Layout.InRegion
import Proofs.Flock.Soundness.Discharge.Layout.InCover
import Proofs.Flock.Soundness.Discharge.Layout.Pins
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.Layout.SharedRows
import Proofs.Flock.Soundness.Discharge.Layout.QSort
import Proofs.Flock.Soundness.Discharge.Layout.Site
import Proofs.Flock.Soundness.Discharge.Layout.OutSite
import Proofs.Flock.Soundness.Discharge.Layout.OfRows
import Proofs.Flock.Soundness.Discharge.Layout.KeyProg
import Proofs.Flock.Soundness.Discharge.Layout.KeyProgP
import Proofs.Flock.Soundness.Discharge.Layout.OutShape
import Proofs.Flock.Soundness.Discharge.Layout.OutPin
import Proofs.Flock.Soundness.Discharge.Layout.RowEnc
import Proofs.Flock.Soundness.Discharge.Layout.KeysRow
import Proofs.Flock.Soundness.Discharge.Layout.Outs
import Proofs.Flock.Soundness.Discharge.Layout.OutsP
import Proofs.Flock.Soundness.Discharge.Layout.Keyed
import Proofs.Flock.Soundness.Discharge.Layout.Tables
import Proofs.Flock.Soundness.Discharge.Layout.Scope

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
* `OutRegion`: the returned outputs the statement pins at the unit slots' output columns, in an untyped circuit whose
  unit returns them (`out_pinned`, at `outAddr`; region "out"'s bytes `outRegion_bytes`, its packing `pack_bit`);
* `Pins`: `lay` and `rs` from fixed readers (`Readers`) and three facts of every satisfying witness: Δ's copies
  (`Readers.Copies`), the rows' pinned `b ‖ c` (`Readers.BcPinned`) and the other pinned values (`Readers.ValPinned`);
* `Drawn`: at a drawn statement, the pinned `b ‖ c` of a proved unit's `hm96` slot is the unit's in the registered
  population's public file (`bc_drawn`);
* `SharedRows`: the same with or without shared rows (`bc_drawnS`), from the shared-row population's row pieces;
* `QSort`: every proved unit of a draw is in the population, with or without a closure (`proved_lt_population`, by
  `all_qsort`: `Array.qsort` keeps a property of every element);
* `Site`: at a drawn unit's site in its table's accepted statement, the population's `b ‖ c` at port `p`'s `hm96` output
  (`bc_site`) and, at each message-keyed unit input, its row's bit (`copy_site`);
* `OutSite`: at a drawn statement, region "out"'s outputs at a proved unit's slot are the unit's in the registered
  population's public file (`outVal_drawn`, `out_drawn`), and at a drawn unit's site the bits at its template instance's
  output columns are those outputs' (`out_site`);
* `OfRows`: readers whose commit strings are units' row ports and single gates (`Readers.ofRows`), with Δ's copies
  from the row bits alone (`ofRows_copies`), the rows' `b ‖ c` (`ofRows_bcPinned`) and the single gates' bits
  (`ofRows_valPinned`; 1 on the constant's gates and 0 on the zero's, `gate_pub`);
* `KeyProg`: those readers at [e2e-placed]'s places of the circuit's program on Δ's source keys (`Placed.keyProg`),
  with rows the message-keyed inputs (`keyRowOf`): Δ's copies (`keyProg_copies`), the rows' `b ‖ c`
  (`keyProg_bcPinned`), the single gates' bits (`keyProg_valPinned`), and `lay` and `rs` there (`keyLayout`,
  `keyRows`);
* `KeyProgP`: the gates of `Placed.keyProgP` (keys with public inputs): row bits (`keyRowOfP`), public gates
  (`keyPubOf`), the registration's bit at a single gate (`keyGateP`), and `single_gateP` under `KeysRowP`;
* `OutShape`: the shape region "out" packs, from the verifier's checks in an untyped circuit whose unit returns the
  outputs (`outShape_of_parse`, by `Net.ofRows`'s `built_out` and `HmRow.check`'s `check_out`);
* `OutPin`: the units' committed gates at those places hold the population's returned outputs (`keyProg_hout`, with the
  registration's bit `outBitOf`), when they are the returned outputs' bits (`OutsOut`), and `rs` with them
  (`keyRowsOut`);
* `RowEnc`: the bytes `sha512/row/v1` hashes for a row (`rowEnc`, injective by `rowEnc_inj`; the layout's values carry
  their width, `BitVal`): at `keyProg`'s row port `q`, `rowPrefix words` and the row's bytes at the message columns
  (`keyRow_enc`, [e2e-hm]'s `port_commit` row), and `rs` with it (`keyRowsEnc`);
* `KeysRow`: `KeysRow` from the parse at row widths `16 · words` (`keysRow_of_parse`, `portRowW`), for a unit with only
  its leaf group whose rows read only port bits (`PortReads`, a circuit condition);
* `Outs`: the returned outputs' bits as the program's outputs (`outsOf`, with `OutsOut` by `outsOut_outsOf`: a unit's
  committed gate it computes is an output, `computed_committed_output`, since no unit reads a gate another computes,
  `keyProg_reads`);
* `OutsP`: `Outs` at `Placed.keyProgP` (`OutsOutP`, `outsOfP`, `outsOutP_outsOfP`, `keyProgP_reads`);
* `Keyed`: `lay` and `rs` at the production program's places (`keyedLayout`, `keyedRows`, `keyedRowsOfParse`,
  `keyedRowsOuts` at `outs := outsOf`), with [e2e-const]'s and [e2e-zero]'s constant and zero gates; `rs.Computes`
  there is [e2e-hm]'s `Discharge.Hm.keyedRowsOuts_computes`;
* `Tables`: a population without shared rows (`pop.tables = none`, which `bc_drawn` reads) under any tag
  without shared-row files (`loadPublic_tables`);
* `Scope`: `rs`'s circuit conditions (`h1`, `ho`, `PortReads`) as one check on the parsed circuit that runs
  (`scopeOk`, `scopeOk_spec`).
-/
