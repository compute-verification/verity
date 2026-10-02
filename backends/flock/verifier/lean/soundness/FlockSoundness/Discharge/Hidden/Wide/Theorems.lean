import FlockSoundness.Discharge.Hidden.Wide.Defs
import FlockSoundness.Discharge.Hidden.Wide.Tail

/-!
# The widened hidden-output statements, one theorem each

* `outputs_committed_tail`, `outputs_binding_tail`: a unit or a tail that returns the outputs;
* `outputs_committed_v2`, `outputs_binding_v2`: `sha512/row/v1` or `sha512/row/v2` rows;
* `outputs_committed_wide`, `outputs_binding_wide`: a population file with or without shared rows.
-/

namespace FlockSoundness.Discharge.Hidden

theorem outputs_committed_tail : OutputsCommittedTail := outputs_committed_tail'

theorem outputs_binding_tail : OutputsBindingTail := outputs_binding_tail_of outputs_committed_tail'

theorem outputs_committed_v2 : OutputsCommittedV2 := sorry

theorem outputs_binding_v2 : OutputsBindingV2 := sorry

theorem outputs_committed_wide : OutputsCommittedWide := sorry

theorem outputs_binding_wide : OutputsBindingWide := sorry

end FlockSoundness.Discharge.Hidden
