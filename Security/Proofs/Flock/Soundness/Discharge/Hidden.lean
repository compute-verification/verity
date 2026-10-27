import Proofs.Flock.Soundness.Discharge.Hidden.Defs
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.Hidden.SpecP
import Proofs.Flock.Soundness.Discharge.Hidden.PortsP
import Proofs.Flock.Soundness.Discharge.Hidden.DerivedP
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.Discharge.Hidden.Binding
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.Discharge.Hidden.Rows
import Proofs.Flock.Soundness.Discharge.Hidden.Layout
import Proofs.Flock.Soundness.Discharge.Hidden.LayoutP
import Proofs.Flock.Soundness.Discharge.Hidden.Flat
import Proofs.Flock.Soundness.Discharge.Hidden.Committed
import Proofs.Flock.Soundness.Discharge.Hidden.Theorems
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Theorems
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Fit
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegChain
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegShape
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegCommit
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegDelta
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegCopies
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegRowBc
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegV2
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegPort
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegTgt
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegFit
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegFitP
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SharedP
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.FitS

/-!
# Hidden outputs: `Stmt.setupHidden`'s soundness at the protocol as Lean defines it

* `Defs`: the hidden-output statement (`Spec`, `outPairs`, `outBytes`, `commitPair`) and the frozen statements;
* `Spec`: `setupHidden_spec'`, the walk of `Stmt.setupHidden` to its `Spec`;
* `PortsP`: the public leaves `HmIn.parse` accepted (`publicCol_at`), and the rows `HmIn.delta` keeps (`keepCol`,
  `deltaEntry_inDelta`);
* `DerivedP`: the public inputs the verifier derives itself (`derivedGroup_some`, `checkDerived_words`: instance `i`'s
  word `i mod G`);
* `Facts`: what is read off `Spec` and `HmOut.parse` (`setupHidden_hm96`, `setupHidden_wf`, `parse_untyped`, ...);
* `Binding`: the row commitment's extractor (`commitPair_spec'`) and `OutputsBinding` from `OutputsCommitted`;
* `Public`: what the statement takes from the public file (`public_exposes'`);
* `Delta`: Δ as `HmRow.delta`'s, then `outPairs` (`hidden_delta'`);
* `Rows`: an output row is a padded lone port or an exact run of ports (`checkRows_out`: `HmOut.checkRows`), and
  `sha512/row/v1` output rows are one per output port (`out_rows_v1`);
* `Layout`: an untyped statement's layout (`hidden_layout'`), `layout_of`'s with `outPairs` past its Δ;
* `LayoutP`: its layout at public inputs (`hidden_layoutP`), Δ without the pairs into a public leaf's bits;
* `Flat`: the flat twin (`HmRow.delta`'s Δ alone), which a satisfying witness also satisfies (`sat_flat`), and an output
  row's message bits as copies of `outSrc` (`msg_copy`);
* `Committed`: an output row's registered `b ‖ c` at a drawn unit is the commit string of its prefix and the output words
  its template instance computes (`outputs_committed'`);
* `Theorems`: one theorem per frozen statement;
* `Wide.Defs`, `Wide.Theorems`: `OutputsCommitted` and `OutputsBinding` widened to a tail that returns the outputs
  (`…Tail`), `sha512/row/v2` rows (`…V2`) and shared rows (`…Wide`);
* `Wide.Tail`: the Tail statements (`outputs_committed_tail'`, `outputs_binding_tail_of`);
* `Wide.RowCommit`: a row's commit string under any 128-byte prefix, with or without compressions (`port_commit_rowsP`,
  `port_commit_rows0`);
* `Wide.Copies`: `HmRow.delta`'s Δ for any rows (`delta_flat_all`) and its copies at a row's own columns (`…_filterAll`);
* `Wide.RowBc`: a row's `b ‖ c` under its own prefix for any rows (`row_bc_tgt`);
* `Wide.Fit`: output rows that fit (`OutRowsF`: unsegmented, their last bytes and SHA-512's padding in their `hm96`
  slot's block), at which `msg_copy` and `sat_flat` hold (`msg_copyF`, `sat_flatF`), and rows of one prefix kind
  (`RowsFit`, `prefixOf_fit`);
* `Wide.SegChain`: a `sha512/row-seg/v1` segment's chain, its compressions alone hashing its message under any 128-byte
  first block (`chain_digestP`);
* `Wide.SegShape`: a segmented row's compressions in closed form: where its chains start (`chainStart_some`,
  `chainStart_first`) and what each compression's message bit is (`msgBit_seg`, `msgBit_outer`);
* `Wide.SegCommit`: a segmented row's commit string from its chains: each segment's digest (`seg_digest`, when its
  bytes reach its last block, `SegFits`) and the outer chain over them (`seg_commit`), the bytes `ExplicitLeaf.rowSeg`
  binds;
* `Wide.SegDelta`: Δ for rows of every kind (`delta_flat_any`), a segmented row's pairs compression by compression
  (`segPairsOf`), and the rows' wires (`rowWires_any`);
* `Wide.SegCopies`: Δ's copies at a row's own columns for rows of every kind (`filter_rowsAny`), and `Wide.Copies`'
  copies at an unsegmented row's columns whatever the other rows (`Any.cv0_filterAll` … `Any.msgConst_filterAll`,
  `Any.msgData_filterAll`);
* `Wide.SegRowBc`: an unsegmented row's `b ‖ c` under its own prefix whatever the other rows (`Any.row_bc_tgt`);
* `Wide.SegV2`: the V2 statements with only the output rows unsegmented, an input row of any kind
  (`Any.outputs_committed_seg'`, `Any.outputs_binding_seg_of`);
* `Wide.SegPort`: a segmented row's `b ‖ c` at its columns, the commit string of `RowSeg.enc` of its bytes, from its
  slots' rows and the midstates, chain copies, constants and digest copies Δ writes (`Any.port_commit_seg`);
* `Wide.SegTgt`: those copies from Δ at a segmented row's columns (`Any.seg_copies_tgt`), so a segmented row's `b ‖ c`
  in a satisfying witness of the untyped statement (`Any.seg_bc_tgt`);
* `Wide.SegFit`: `Wide.Fit`'s output-row lemmas with only the output rows unsegmented (`Any.PadRow`, `Any.OutRowsS`), an
  input row of any kind (`Any.msg_copyF`, `Any.sat_flatF`);
* `Wide.SegFitP`: those at public inputs, off the unit input rows `HmIn.delta` drops (`unitIn_of_drop`,
  `Any.msg_copyFP`, `Any.sat_flatFP`);
* `Wide.FitS`: the rows the hidden-output headlines cover, each unsegmented (`Any.PadRow`) or a `sha512/row-seg/v1` row
  of any mix (`Any.RowsFitS`), and a row's `b ‖ c` under its own bytes (`Any.rowMsg`, `Any.row_bc_tgtS`);
* `Wide.V2`: the V2 statements (`outputs_committed_v2'`, `outputs_binding_v2_of`);
* `Wide.Shared`: a shared-row population's `b ‖ c` at a drawn unit (`bc_drawnW`), and the Wide statements
  (`outputs_committed_wide'`, `outputs_binding_wide_of`);
* `Wide.SharedP`: that `b ‖ c` at public inputs, the public file read under the public input words (`inRows`,
  `loadPublic_inRows`, `bc_drawnWP`);
* `Wide.Upv`: a unit per template instance from acceptance (`check_upv_pos`, `setupHidden_upv_pos`), since `HmRow.check`
  refuses `units_per_vu = 0`;
* `Wide.Block`: a template instance per block from acceptance (`check_g_pos`, `setupH_g_pos`), since `HmRow.check`
  refuses `vus_per_block = 0`;
* `Wide.Checked`: the six widened statements without `0 < st.c.upv` (`…Checked`), proved in `Wide.Theorems`
  (`…_checked`).
-/
