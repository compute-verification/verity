import Run.Rules
import RunModel.Assumptions

/-!
# The run's guarantees (trusted)

Two end-to-end statements about the code that runs (`Run`'s machines), and the design rules as one statement.

* `Soundness`: if the auditor's run accepts, the claim's error bound holds, against every developer (the boundary, the
  GPU rack) and network.
* `ZeroKnowledge`: the auditor's view of the whole run is simulatable from the leak, against every auditor, network and
  GPU rack, under the egress premise (the setting's: `Setting.zk`, `World.byGpu`).
* `Rules`: each party owns its state, refuses out-of-order and duplicate messages, gives coins only after their
  commitment, never repeats a salt, and lets nothing out of the boundary before its check.

`Soundness` is stated for an extraction and error terms, `ZeroKnowledge` for a simulation and the outer session's term;
a proof package on C-Flock's soundness package defines them for this run and proves both (`RunModelProofs`). Each
takes `cr/sha-512` for its own finders, charged at `Assumptions.collisions`, and `ZeroKnowledge` takes hm96 hiding.
-/

namespace RunModel.Guarantees

open Run
open scoped ENNReal

def Soundness (A : Agreement) (cl : Claim A) (K : Extraction A cl) (ε : ErrorTerms) : Prop :=
  LawDelta A →
    ∀ (fuel : ℕ) (dev : Adversary) (q : ℝ), (∀ f ∈ K.finders fuel dev, Assumptions.FinderCR f q) →
      (soundRun A dev fuel (World.init _)).toOuterMeasure {w | Violated A cl (K.committed fuel dev) w} ≤
        delta A + ε.total + Assumptions.collisions (K.finders fuel dev).length q

/-- The simulator's error: two hybrid steps per leaf the boundary salts, and the outer session's own. -/
noncomputable def zkError (A : Agreement) (δ₁ εouter : ℝ≥0∞) : ℝ≥0∞ := 2 * leaves A * δ₁ + εouter

def ZeroKnowledge (A : Agreement) (S : Simulation A) (εouter : ℝ≥0∞) : Prop :=
  ∀ δ₁ : ℝ≥0∞, Assumptions.Hm96Hiding δ₁ →
    ∀ (fuel : ℕ) (net gpu : Adversary) (q : ℝ), (∀ f ∈ S.finders fuel net, Assumptions.FinderCR f q) →
      tv (real A fuel net gpu) (ideal A fuel net gpu S.sim) ≤
        zkError A δ₁ εouter + Assumptions.collisions (S.finders fuel net).length q

def Rules : Prop :=
  (∀ S : Setting, Run.Rules.OwnsState S) ∧
  (∀ (C : Core) (self : Node), Machine.InOrder (C.seq self)) ∧
  (∀ A : Agreement,
    Run.Rules.CoinsAfterCommit (CoinServer.machine A) ∧ Run.Rules.SaltsFresh A ∧ Run.Rules.OnlyGateLeaves A ∧
      Run.Rules.GateChecks A ∧ Run.Rules.Egress A)

end RunModel.Guarantees
