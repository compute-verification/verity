import FlockSoundness.Discharge.Hidden.Defs
import FlockSoundness.Discharge.Hidden.Spec
import FlockSoundness.Discharge.Hidden.Facts
import FlockSoundness.Discharge.Hidden.Binding
import FlockSoundness.Discharge.Hidden.Public
import FlockSoundness.Discharge.Hidden.Delta
import FlockSoundness.Discharge.Hidden.Rows
import FlockSoundness.Discharge.Hidden.Layout
import FlockSoundness.Discharge.Hidden.Flat
import FlockSoundness.Discharge.Hidden.Theorems

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
* `Theorems`: one theorem per frozen statement.
-/
