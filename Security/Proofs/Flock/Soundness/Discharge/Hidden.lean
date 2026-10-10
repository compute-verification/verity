import Proofs.Flock.Soundness.Discharge.Hidden.Defs
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.Discharge.Hidden.Binding
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.Discharge.Hidden.Rows
import Proofs.Flock.Soundness.Discharge.Hidden.Layout
import Proofs.Flock.Soundness.Discharge.Hidden.Flat
import Proofs.Flock.Soundness.Discharge.Hidden.Committed
import Proofs.Flock.Soundness.Discharge.Hidden.Theorems
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Theorems
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Fit

/-!
# Hidden outputs: `Stmt.setupHidden`'s soundness at the protocol as Lean defines it

* `Defs`: the hidden-output statement (`Spec`, `outPairs`, `outBytes`, `commitPair`) and the frozen statements;
* `Spec`: `setupHidden_spec'`, the walk of `Stmt.setupHidden` to its `Spec`;
* `Facts`: what is read off `Spec` and `HmOut.parse` (`setupHidden_hm96`, `setupHidden_wf`, `parse_untyped`, ...);
* `Binding`: the row commitment's extractor (`commitPair_spec'`) and `OutputsBinding` from `OutputsCommitted`;
* `Public`: what the statement takes from the public file (`public_exposes'`);
* `Delta`: Δ as `HmRow.delta`'s, then `outPairs` (`hidden_delta'`);
* `Rows`: an output row is a padded lone port or an exact run of ports (`checkRows_out`: `HmOut.checkRows`), and
  `sha512/row/v1` output rows are one per output port (`out_rows_v1`);
* `Layout`: an untyped statement's layout (`hidden_layout'`), `layout_of`'s with `outPairs` past its Δ;
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
* `Wide.V2`: the V2 statements (`outputs_committed_v2'`, `outputs_binding_v2_of`);
* `Wide.Shared`: a shared-row population's `b ‖ c` at a drawn unit (`bc_drawnW`), and the Wide statements
  (`outputs_committed_wide'`, `outputs_binding_wide_of`);
* `Wide.Upv`: a unit per template instance from acceptance (`check_upv_pos`, `setupHidden_upv_pos`), since `HmRow.check`
  refuses `units_per_vu = 0`;
* `Wide.Block`: a template instance per block from acceptance (`check_g_pos`, `setupH_g_pos`), since `HmRow.check`
  refuses `vus_per_block = 0`;
* `Wide.Checked`: the six widened statements without `0 < st.c.upv` (`…Checked`), proved in `Wide.Theorems`
  (`…_checked`).
-/
