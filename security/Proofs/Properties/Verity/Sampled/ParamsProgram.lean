module

public import Spot.OneStage
public import Proofs.Flock.Soundness.Refine.Walk

/-!
# The verifier program v2's parameters name parses from some bytes (`paramsParsed_program`)

`Spot.OneStage.paramsParsed` binds the verifier program to `Program.parse` of the canonical text of the parameters'
`program` and returns it as the parameters' `verifier.program`. `rw [paramsParsed.eq_def]` keeps the block's join
points as `let`s, and `ret_walk_aux` walks each once to each return, keeping the parse's equation and the returned
parameters' `let`, each join point's body its own auxiliary theorem.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness.Refine in
/-- **v2's parameters' verifier program is one some bytes parse to**: `paramsParsed` parses it from the canonical text
of the parameters' `program`. -/
theorem paramsParsed_program {j : Lean.Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsParsed j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program := by
  revert p
  show Ret (_root_.Spot.OneStage.paramsParsed j) fun p => ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program
  rw [_root_.Spot.OneStage.paramsParsed.eq_def]
  ret_walk_aux [_root_.Spot.Program.parse, _root_.Spot.OneStage.Params.mk] =>
    subst_vars
    first
      | exact ⟨_, by with_reducible assumption⟩
      | exact ⟨_, ‹_›⟩

end Verity.Sampled
