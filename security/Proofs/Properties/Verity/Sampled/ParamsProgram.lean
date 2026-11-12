module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Properties.Verity.Sampled.Trees
import Proofs.Flock.Soundness.Refine.Walk

/-!
# v2's own program is its verifier's (`ParamsProgram`)

The parameters v2 reads (`ParamsAt`) hold the verifier's program once, as their `program` field, and read it twice:
`Spot.OneStage.paramsParsed` parses the field's canonical text into the verifier's program, and
`Spot.OneStageZk.paramsOfJson` keeps the same text as its own inputs' program. So that text parses to the verifier's
program (`params_program`): it is a descriptor of the verifier's program (`params_desc`).

* `paramsParsed_program`, `paramsOfJson_program`: the one-stage parameters' program is the parse of their `program`
  field's canonical text.
* `zkParams_program`: the `--zk` parameters' own program is that text.
* `params_program`, `params_desc`: at the parameters `ParamsAt` reads.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness.Refine (Ret)

/-- **`paramsParsed` parses its verifier's program from the `program` field's canonical text.** -/
theorem paramsParsed_program {j : Lean.Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsParsed j = .ok p) :
    ∃ s, Flock.canon (_root_.Spot.getD j "program") = .ok s ∧
      _root_.Spot.Program.parse s.toUTF8 = .ok p.verifier.program := by
  unfold _root_.Spot.OneStage.paramsParsed at h
  obtain ⟨-, -, h⟩ := Trees.bind_ok h
  obtain ⟨X, hX, h⟩ := Trees.bind_ok h
  obtain ⟨s, hs, h⟩ := Trees.bind_ok h
  obtain ⟨prog, hprog, h⟩ := Trees.bind_ok h
  have hXj : X = _root_.Spot.getD j "program" := by
    cases hj : _root_.Spot.getD j "program" <;> rw [hj] at hX ⊢ <;>
      first | exact (Except.ok.inj hX).symm | cases hX
  subst hXj
  refine ⟨s, hs, ?_⟩
  have hpp : p.verifier.program = prog := by
    revert p
    show Ret _ fun q : _root_.Spot.OneStage.Params => q.verifier.program = prog
    dsimp only
    ret_walk []
    all_goals rfl
  rw [hpp]
  exact hprog

/-- **The one-stage parameters' program is the parse of their `program` field's canonical text**
(`paramsParsed_program`; `paramsOfJson` only refuses a `bernoulli` design after it). -/
theorem paramsOfJson_program {j : Lean.Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsOfJson j = .ok p) :
    ∃ s, Flock.canon (_root_.Spot.getD j "program") = .ok s ∧
      _root_.Spot.Program.parse s.toUTF8 = .ok p.verifier.program := by
  revert p
  show Ret (_root_.Spot.OneStage.paramsOfJson j) fun q => ∃ s, Flock.canon (_root_.Spot.getD j "program") = .ok s ∧
    _root_.Spot.Program.parse s.toUTF8 = .ok q.verifier.program
  rw [_root_.Spot.OneStage.paramsOfJson.eq_def]
  ret_walk [_root_.Spot.OneStage.paramsParsed]
  all_goals exact paramsParsed_program ‹_root_.Spot.OneStage.paramsParsed _ = .ok _›

/-- **The `--zk` parameters' own program parses to their verifier's**: both are the `program` field's canonical text
on the parameters without their `proofs` (`paramsOfJson_program`). -/
theorem zkParams_program {j : Lean.Json} {p : _root_.Spot.OneStageZk.Params}
    (h : _root_.Spot.OneStageZk.paramsOfJson j = .ok p) :
    _root_.Spot.Program.parse p.own.program = .ok p.base.verifier.program := by
  revert p
  show Ret (_root_.Spot.OneStageZk.paramsOfJson j) fun q =>
    _root_.Spot.Program.parse q.own.program = .ok q.base.verifier.program
  rw [_root_.Spot.OneStageZk.paramsOfJson.eq_def]
  dsimp only
  ret_walk [_root_.Spot.OneStage.paramsOfJson, Flock.canon]
  all_goals
    have hc := ‹Flock.canon _ = .ok _›
    obtain ⟨s, hs, hp⟩ := paramsOfJson_program ‹_root_.Spot.OneStage.paramsOfJson _ = .ok _›
    try dsimp only
    rw [Except.ok.inj (hc.symm.trans hs)]
    exact hp

/-- **At the parameters v2 reads, its own program parses to its verifier's** (`zkParams_program`). -/
theorem params_program {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} (h : ParamsAt θ p) :
    _root_.Spot.Program.parse p.own.program = .ok p.base.verifier.program := by
  obtain ⟨-, -, -, -, hx⟩ := h
  obtain ⟨j, -, hj⟩ := Trees.bind_ok hx
  have key : Ret (VStar.Recursive.Sampled.paramsOfJson j) fun r => ∀ q, r = .inl q →
      _root_.Spot.Program.parse q.own.program = .ok q.base.verifier.program := by
    rw [VStar.Recursive.Sampled.paramsOfJson.eq_def]
    ret_walk [_root_.Spot.OneStageZk.paramsOfJson]
    all_goals
      intro q hq
      cases hq
      exact zkParams_program ‹_root_.Spot.OneStageZk.paramsOfJson _ = .ok _›
  exact key _ hj p rfl

/-- **The parameters' own program is a descriptor of their verifier's program**, the one a session's statement reads
(`ChecksAt.program`). -/
theorem params_desc {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} (h : ParamsAt θ p) :
    _root_.Spot.Program.parse ⟨(p.own.program.data.toList : Claims.Msg).toArray⟩ = .ok p.base.verifier.program :=
  params_program h

end Verity.Sampled
