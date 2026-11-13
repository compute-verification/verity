module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Properties.Verity.Sampled.Trees
import Proofs.Flock.Soundness.Refine.Walk
meta import Proofs.Flock.Soundness.Refine.Walk

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

open Lean Elab Tactic Meta in
/-- At the goal `Ret (let x := v; b) Q` for a `let` that is not a join point, the goal `Ret (b v) Q`. -/
meta def zetaStep (g : MVarId) : MetaM (Option MVarId) := g.withContext do
  let ty ← instantiateMVars (← g.getType)
  unless ty.isAppOfArity ``Ret 3 do return none
  let γ := ty.getArg! 0
  match (ty.getArg! 1).headBeta.consumeMData with
  | .letE _ t v b _ =>
      let vγ := mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) γ
      if (← forallTelescopeReducing t fun _ r => isDefEq r vγ) && v.consumeMData.isLambda then return none
      let g' ← mkFreshExprSyntheticOpaqueMVar (mkApp3 (mkConst ``Ret) γ (b.instantiate1 v) (ty.getArg! 2))
      g.assign g'
      return some g'.mvarId!
  | _ => return none

open Lean Elab Tactic Meta in
/-- `ret_walk`, substituting each `let` that is not a join point rather than generalizing it, for a postcondition that
reads a value the computation binds with `let`. Join points are walked once, as `ret_walk` walks them. -/
elab "ret_walk_zeta" "[" cs:ident,* "]" : tactic => do
  let mut keep := #[]
  for c in cs.getElems do keep := keep.push (← realizeGlobalConstNoOverloadWithInfo c)
  let mut todo ← getGoals
  let mut left := #[]
  while true do
    match todo with
    | [] => break
    | g :: rest =>
      match ← zetaStep g with
      | some g' => todo := g' :: rest
      | none =>
        match ← FlockSoundness.Refine.retStep keep g with
        | some new => todo := new ++ rest
        | none => left := left.push g; todo := rest
  setGoals left.toList

/-- **`paramsParsed` parses its verifier's program from the `program` field's canonical text.** -/
theorem paramsParsed_program {j : Lean.Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsParsed j = .ok p) :
    ∃ s, Flock.canon (_root_.Spot.getD j "program") = .ok s ∧
      _root_.Spot.Program.parse s.toUTF8 = .ok p.verifier.program := by
  unfold _root_.Spot.OneStage.paramsParsed at h
  obtain ⟨-, -, h₁⟩ := Trees.bind_ok h
  obtain ⟨x₀, hx₀, h₂⟩ := Trees.bind_ok h₁
  obtain ⟨s, hs, h₃⟩ := Trees.bind_ok h₂
  obtain ⟨prog, hprog, h₄⟩ := Trees.bind_ok h₃
  have hxj : x₀ = _root_.Spot.getD j "program" := by
    cases hj : _root_.Spot.getD j "program" <;> rw [hj] at hx₀ <;>
      first | exact (Except.ok.inj hx₀).symm | cases hx₀
  subst hxj
  refine ⟨s, hs, ?_⟩
  have hpp : p.verifier.program = prog := by
    clear * - h₄
    revert p
    show Ret _ fun q : _root_.Spot.OneStage.Params => q.verifier.program = prog
    ret_walk_zeta []
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
  obtain ⟨_, _, _, _, hx⟩ := h
  obtain ⟨j, -, hj⟩ := Trees.bind_ok hx
  have key : Ret (VStar.Recursive.Sampled.paramsOfJson j) fun r => ∀ q, r = .inl q →
      _root_.Spot.Program.parse q.own.program = .ok q.base.verifier.program := by
    rw [VStar.Recursive.Sampled.paramsOfJson.eq_def]
    ret_walk [_root_.Spot.OneStageZk.paramsOfJson]
    all_goals
      rename_i hq
      cases hq
      exact zkParams_program ‹_root_.Spot.OneStageZk.paramsOfJson _ = .ok _›
  exact key _ hj p rfl

/-- **The parameters' own program is a descriptor of their verifier's program**, the one a session's statement reads
(`ChecksAt.program`). -/
theorem params_desc {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} (h : ParamsAt θ p) :
    _root_.Spot.Program.parse ⟨(p.own.program.data.toList : Claims.Msg).toArray⟩ = .ok p.base.verifier.program :=
  params_program h

end Verity.Sampled
