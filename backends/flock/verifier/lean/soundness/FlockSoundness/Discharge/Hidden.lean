import FlockSoundness.Discharge.Hidden.Defs
import FlockSoundness.Discharge.Hidden.Spec
import FlockSoundness.Discharge.Hidden.Facts
import FlockSoundness.Discharge.Hidden.Binding
import FlockSoundness.Discharge.Hidden.Public
import FlockSoundness.Discharge.Hidden.Theorems

/-!
# Hidden outputs: `Stmt.setupHidden`'s soundness at the protocol as Lean defines it

* `Defs`: the hidden-output statement (`Spec`, `outPairs`, `outBytes`, `commitPair`) and the frozen statements;
* `Spec`: `setupHidden_spec'`, the walk of `Stmt.setupHidden` to its `Spec`;
* `Facts`: what is read off `Spec` and `HmOut.parse` (`setupHidden_hm96`, `setupHidden_wf`, `parse_untyped`, ...);
* `Binding`: the row commitment's extractor (`commitPair_spec'`) and `OutputsBinding` from `OutputsCommitted`;
* `Public`: what the statement takes from the public file (`public_exposes'`);
* `Theorems`: one theorem per frozen statement.
-/
