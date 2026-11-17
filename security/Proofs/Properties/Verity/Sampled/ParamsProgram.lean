module

public import Spot.OneStage
public import Proofs.Flock.Soundness.Refine.Walk

/-! The verifier program v2's parameters name parses from some bytes (`paramsParsed_program`). -/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness.Refine in
theorem paramsParsed_program {j : Lean.Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsParsed j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program := by
  revert p
  show Ret (_root_.Spot.OneStage.paramsParsed j) fun p => ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program
  rw [_root_.Spot.OneStage.paramsParsed.eq_def]
  ret_walk_aux [_root_.Spot.Program.parse, _root_.Spot.OneStage.Params.mk] =>
    subst_vars
    exact ⟨_, by with_reducible assumption⟩

end Verity.Sampled
