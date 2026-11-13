module

public import Spot.OneStage
public import Proofs.Flock.Soundness.Refine.Walk

/-!
# The verifier program v2's parameters name parses from some bytes (`paramsParsed_program`)

`Spot.OneStage.paramsParsed` binds the verifier program to `Program.parse` of the canonical text of the parameters'
`program` and returns it as the parameters' `verifier.program`. `ret_walk` walks the block to each return, keeping the
parse's equation and the returned parameters' `let`; each of the block's join points is walked once.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness.Refine in
/-- **v2's parameters' verifier program is one some bytes parse to**: `paramsParsed` parses it from the canonical text
of the parameters' `program`. -/
theorem paramsParsed_program {j : Lean.Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsParsed j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program := by
  have hr : Ret (_root_.Spot.OneStage.paramsParsed j)
      (fun p => ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program) := by
    unfold _root_.Spot.OneStage.paramsParsed
    ret_walk [_root_.Spot.Program.parse, _root_.Spot.OneStage.Params.mk]
    all_goals
      subst_vars
      exact ⟨_, ‹_›⟩
  exact hr p h

end Verity.Sampled
