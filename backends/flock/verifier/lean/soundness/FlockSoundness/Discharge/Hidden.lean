import FlockSoundness.Discharge.Hidden.Defs
import FlockSoundness.Discharge.Hidden.Spec
import FlockSoundness.Discharge.Hidden.Facts
import FlockSoundness.Discharge.Hidden.Binding
import FlockSoundness.Discharge.Hidden.Public
import FlockSoundness.Discharge.Hidden.Delta
import FlockSoundness.Discharge.Hidden.Rows
import FlockSoundness.Discharge.Hidden.Layout
import FlockSoundness.Discharge.Hidden.Flat
import FlockSoundness.Discharge.Hidden.Committed
import FlockSoundness.Discharge.Hidden.Theorems
import FlockSoundness.Discharge.Hidden.Wide.Theorems

/-!
# Hidden outputs: `Stmt.setupHidden`'s soundness at the protocol as Lean defines it

* `Defs`: the hidden-output statement (`Spec`, `outPairs`, `outBytes`, `commitPair`) and the frozen statements;
* `Spec`: `setupHidden_spec'`, the walk of `Stmt.setupHidden` to its `Spec`;
* `Facts`: what is read off `Spec` and `HmOut.parse` (`setupHidden_hm96`, `setupHidden_wf`, `parse_untyped`, ...);
* `Binding`: the row commitment's extractor (`commitPair_spec'`) and `OutputsBinding` from `OutputsCommitted`;
* `Public`: what the statement takes from the public file (`public_exposes'`);
* `Delta`: Δ as `HmRow.delta`'s, then `outPairs` (`hidden_delta'`);
* `Rows`: an output row's message bytes are its compressions' (`out_chunks`: `HmRow.rowPorts`' v2 rows, `HmOut.checkRows`);
* `Layout`: an untyped statement's layout (`hidden_layout'`), `layout_of`'s with `outPairs` past its Δ;
* `Flat`: the flat twin (`HmRow.delta`'s Δ alone), which a satisfying witness also satisfies (`sat_flat`), and an output
  row's message bits as copies of `outSrc` (`msg_copy`);
* `Committed`: an output row's registered `b ‖ c` at a drawn unit is the commit string of its prefix and the output words
  its VU computes (`outputs_committed'`);
* `Theorems`: one theorem per frozen statement;
* `Wide.Defs`, `Wide.Theorems`: `OutputsCommitted` and `OutputsBinding` widened to a tail that returns the outputs
  (`…Tail`), `sha512/row/v2` rows (`…V2`) and shared rows (`…Wide`);
* `Wide.Tail`: the Tail statements (`outputs_committed_tail'`, `outputs_binding_tail_of`);
* `Wide.RowCommit`: a row's commit string under any 128-byte prefix, with or without compressions (`port_commit_rowsP`,
  `port_commit_rows0`);
* `Wide.Copies`: `HmRow.delta`'s Δ for any rows (`delta_flat_all`) and its copies at a row's own columns (`…_filterAll`);
* `Wide.RowBc`: a row's `b ‖ c` under its own prefix for any rows (`row_bc_tgt`);
* `Wide.V2`: the V2 statements (`outputs_committed_v2'`, `outputs_binding_v2_of`);
* `Wide.Shared`: a shared-row population's `b ‖ c` at a drawn unit (`bc_drawnW`), and the Wide statements
  (`outputs_committed_wide'`, `outputs_binding_wide_of`);
* `Wide.Upv`: a unit per VU from acceptance (`check_upv_pos`, `setupHidden_upv_pos`), since `HmRow.check` refuses
  `units_per_vu = 0`;
* `Wide.Checked`: the six widened statements without `0 < st.c.upv` (`…Checked`), proved in `Wide.Theorems`
  (`…_checked`).
-/
