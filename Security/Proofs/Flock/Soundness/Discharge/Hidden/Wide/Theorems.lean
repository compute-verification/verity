module

public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Defs
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Checked
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Shared
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Tail
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Upv
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.V2

/-!
# The widened hidden-output statements, one theorem each

* `outputs_committed_tail`, `outputs_binding_tail`: a unit or a tail that returns the outputs;
* `outputs_committed_v2`, `outputs_binding_v2`: `sha512/row/v1` or `sha512/row/v2` rows;
* `outputs_committed_wide`, `outputs_binding_wide`: a population file with or without shared rows;
* `…_checked`: each of them without `0 < st.c.upv`, which acceptance gives (`setupHidden_upv_pos`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Hidden

theorem outputs_committed_tail : OutputsCommittedTail := outputs_committed_tail'

theorem outputs_binding_tail : OutputsBindingTail := outputs_binding_tail_of outputs_committed_tail'

theorem outputs_committed_v2 : OutputsCommittedV2 := outputs_committed_v2'

theorem outputs_binding_v2 : OutputsBindingV2 := outputs_binding_v2_of outputs_committed_v2'

theorem outputs_committed_wide : OutputsCommittedWide := outputs_committed_wide'

theorem outputs_binding_wide : OutputsBindingWide := outputs_binding_wide_of outputs_committed_wide'

theorem outputs_committed_tail_checked : OutputsCommittedTailChecked := by
  intro _ _ _ _ _ _ _ _ _ ht hi hs _ hl hpt _ hD hDp
  exact outputs_committed_tail ht hi hs hl hpt hD hDp (setupHidden_upv_pos ht hi hs)

theorem outputs_binding_tail_checked : OutputsBindingTailChecked := by
  intro _ _ _ _ _ _ _ _ _ _ _ _ _ _ ht hi hs hs' _ hl hpt _ _ hD hDp hD' hDp'
  exact outputs_binding_tail ht hi hs hs' hl hpt hD hDp hD' hDp' (setupHidden_upv_pos ht hi hs)

theorem outputs_committed_v2_checked : OutputsCommittedV2Checked := by
  intro _ _ _ _ _ _ _ _ _ ht hi hs _ hl hpt _ hD hDp
  exact outputs_committed_v2 ht hi hs hl hpt hD hDp (setupHidden_upv_pos ht hi hs)

theorem outputs_binding_v2_checked : OutputsBindingV2Checked := by
  intro _ _ _ _ _ _ _ _ _ _ _ _ _ _ ht hi hs hs' _ hl hpt _ _ hD hDp hD' hDp'
  exact outputs_binding_v2 ht hi hs hs' hl hpt hD hDp hD' hDp' (setupHidden_upv_pos ht hi hs)

theorem outputs_committed_wide_checked : OutputsCommittedWideChecked := by
  intro _ _ _ _ _ _ _ _ _ ht hi hs _ hl _ hD hDp
  exact outputs_committed_wide ht hi hs hl hD hDp (setupHidden_upv_pos ht hi hs)

theorem outputs_binding_wide_checked : OutputsBindingWideChecked := by
  intro _ _ _ _ _ _ _ _ _ _ _ _ _ _ ht hi hs hs' _ hl _ _ hD hDp hD' hDp'
  exact outputs_binding_wide ht hi hs hs' hl hD hDp hD' hDp' (setupHidden_upv_pos ht hi hs)

end FlockSoundness.Discharge.Hidden
