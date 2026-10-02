import FlockSoundness.Discharge.Hidden.Defs

/-!
# The soundness of a hidden-output statement (`Stmt.setupHidden`)

The frozen statements of `Defs.lean`, one theorem each:
* `setupHidden_spec`: what `setupHidden` checked and produced (`Spec`);
* `hidden_delta`: its Δ, `HmRow.delta`'s then `outPairs`;
* `hidden_layout`: an untyped one has the layout the placement reads, so its block R1CS is `Layout.model`;
* `public_exposes`: it takes no output word from the public file, and its regions are every row's `b ‖ c`;
* `outputs_committed`: in a satisfying witness, an output row's registered `b ‖ c` at a drawn unit is the commit
  string of the row prefix and the output words its VU computes;
* `commitPair_spec`: the row commitment's extractor;
* `outputs_binding`: two accepted statements that put one registered unit at two VUs, whose output words differ, give a
  SHA-512 collision (`cr/sha-512`).
-/

namespace FlockSoundness.Discharge.Hidden

theorem setupHidden_spec : SetupHiddenSpec := by sorry

theorem hidden_delta : HiddenDelta := by sorry

theorem hidden_layout : HiddenLayout := by sorry

theorem public_exposes : PublicExposes := by sorry

theorem outputs_committed : OutputsCommitted := by sorry

theorem commitPair_spec : CommitPairSpec := by sorry

theorem outputs_binding : OutputsBinding := by sorry

end FlockSoundness.Discharge.Hidden
