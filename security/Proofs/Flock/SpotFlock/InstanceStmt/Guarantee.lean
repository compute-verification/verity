module

public import Proofs.Flock.SpotFlock.InstanceStmt.AcceptsFixture
public import Proofs.Flock.SpotFlock.Honest.Ref
public import Proofs.Flock.Claims.CompleteFails

/-! # Masked completeness at the fixture's statement, as a guarantee

`Flock.Guarantees.FixtureComplete` says, at one statement, `SpotFlock.InstanceStmt.x r` (the two-instance fixture, one
template over two rows with a hidden output) of any plain run `r`, that the masked Flock verifier accepts the honest masked
prover, run on the zero trace, with probability at least `1 − δ_c` over the verifier's coins at every tape, and that `δ_c`
there is at most `190 / 2^128`, so below 1. Its proof is `Flock.SecurityProofs.FixtureComplete`.

It is completeness at this one statement and no more: it is not soundness (it says nothing of what the verifier rejects),
and it is not completeness at every statement, which `FlockComplete.ZkTables.complete_masked_sha512` is, for every
statement in scope that has a witness in the relation. What this adds is that, for this statement, those premises hold
(the statement is in the masked scope, the zero trace is in the relation, the honest record fits its length word) and the
error is below 1, so the general theorem is not empty here.

Its one domain premise is `Run.Plain` (the run's three strings are plain JSON strings, under `2^20` characters in all, so
the public file's header parses back); `Flock.SecurityProofs.FixtureComplete_honest` is a plain run, and
`FixtureComplete_nontrivial` is the statement failing at a verifier that rejects every run.
-/

@[expose] public section

namespace Flock.Guarantees

/-- **The honest masked run at the fixture's statement is accepted with probability at least `1 − δ_c`.** For every plain run
`r` (`Run.Plain`) and every tape `ω` of the masked deployment (`deployZk Readable.relation`), the verifier accepts the
honest prover on the zero trace `Rel.wX r` at the statement `x r`, hashed with SHA-512, with probability at least
`1 − δ_c` over its coins, where `δ_c` (the deployment's completeness error at `x r`) is at most `190 / 2^128`, so below
`1`. This is completeness at one statement, the fixture's, for every plain run and tape: it is not soundness, and it is
not completeness at every statement, which `complete_masked_sha512` is. -/
def FixtureComplete : Prop :=
  ∀ r : SpotFlock.InstanceStmt.Run, r.Plain → ∀ ω : (FlockDeploy.deployZk _root_.Flock.Readable.relation).Tape,
    1 - (FlockDeploy.deployZk _root_.Flock.Readable.relation).δc FlockDeploy.sha512 (SpotFlock.InstanceStmt.x r) ≤
      Claims.Pr (fun c : Fin ((FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k (SpotFlock.InstanceStmt.x r)) →
          FlockDeploy.Coin =>
        (Claims.run FlockDeploy.sha512
          ((FlockDeploy.deployZk _root_.Flock.Readable.relation).P (SpotFlock.InstanceStmt.Rel.wX r))
          (FlockDeploy.deployZk _root_.Flock.Readable.relation).V (SpotFlock.InstanceStmt.x r) ω c).out = .acc) ∧
    (FlockDeploy.deployZk _root_.Flock.Readable.relation).δc FlockDeploy.sha512 (SpotFlock.InstanceStmt.x r) ≤
        190 / 2 ^ 128 ∧
      (FlockDeploy.deployZk _root_.Flock.Readable.relation).δc FlockDeploy.sha512 (SpotFlock.InstanceStmt.x r) < 1

end Flock.Guarantees

namespace Flock.SecurityProofs

/-- **`FixtureComplete`**: `complete_fixture` and `delta_fixture`. -/
theorem FixtureComplete : Flock.Guarantees.FixtureComplete :=
  fun _ hr ω => ⟨SpotFlock.InstanceStmt.complete_fixture hr ω, SpotFlock.InstanceStmt.delta_fixture hr⟩

/-- **`FixtureComplete`'s one premise holds**: the run `Honest.Ref.run₀` is plain. -/
theorem FixtureComplete_honest : SpotFlock.Honest.run₀.Plain := SpotFlock.Honest.run₀_plain

/-- **`FixtureComplete` fails at a verifier that rejects every run** (`rejV`), at the plain run `run₀` and a tape: no coin
accepts, so the probability is `0`, and `1 − δ_c` is positive because `δ_c < 1` (`delta_fixture`). -/
theorem FixtureComplete_nontrivial :
    ¬ (1 - (FlockDeploy.deployZk _root_.Flock.Readable.relation).δc FlockDeploy.sha512
          (SpotFlock.InstanceStmt.x SpotFlock.Honest.run₀) ≤
      Claims.Pr (fun c : Fin ((Claims.CompleteFails.rejV FlockDeploy.Coin).k
          (SpotFlock.InstanceStmt.x SpotFlock.Honest.run₀)) → FlockDeploy.Coin =>
        (Claims.run FlockDeploy.sha512
          ((FlockDeploy.deployZk _root_.Flock.Readable.relation).P
            (SpotFlock.InstanceStmt.Rel.wX SpotFlock.Honest.run₀))
          (Claims.CompleteFails.rejV FlockDeploy.Coin) (SpotFlock.InstanceStmt.x SpotFlock.Honest.run₀)
          (Classical.arbitrary (FlockDeploy.deployZk _root_.Flock.Readable.relation).Tape) c).out = .acc)) := by
  intro h
  have h0 : Claims.Pr (fun c : Fin ((Claims.CompleteFails.rejV FlockDeploy.Coin).k
          (SpotFlock.InstanceStmt.x SpotFlock.Honest.run₀)) → FlockDeploy.Coin =>
        (Claims.run FlockDeploy.sha512
          ((FlockDeploy.deployZk _root_.Flock.Readable.relation).P
            (SpotFlock.InstanceStmt.Rel.wX SpotFlock.Honest.run₀))
          (Claims.CompleteFails.rejV FlockDeploy.Coin) (SpotFlock.InstanceStmt.x SpotFlock.Honest.run₀)
          (Classical.arbitrary (FlockDeploy.deployZk _root_.Flock.Readable.relation).Tape) c).out = .acc) = 0 := by
    rw [Claims.Pr_congr (q := fun _ => False) fun c => by simp [Claims.CompleteFails.run_rejV]]
    exact Claims.Pr_false
  have hδ := (SpotFlock.InstanceStmt.delta_fixture SpotFlock.Honest.run₀_plain).2
  linarith

end Flock.SecurityProofs
