import Proofs.Flock.Bridge.HiddenModel

/-!
# The toy's setup (`toy-setup`)

`EndToEndHiddenModel`'s nontrivial witness is at the toy of `Bridge.HiddenModel`: one unit, the draw file that reads no
draw, and the law that draws the unit. Two of its fields can't be built in Lean: a reference setup `y₀` is a statement
`Stmt.setupHidden` accepts on real circuit and public files, which the kernel doesn't evaluate, and a prover's finder
`B^τ`'s collision probability is SHA-512's. `ToySetupHJ` is the premise that both exist.
-/

namespace Flock.Assumptions

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Refine FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Composed
  FlockSoundness.Discharge.ZkHidden

/-- Premise (`toy-setup`) of `EndToEndHiddenModel`'s nontrivial witness. **The toy has a reference setup and a prover
whose finder rarely collides.** Some inputs `I` have a hidden-output table `verify --zk` sets up at table 0
(`DrawSetupZHJ`, at some link points, draw and part of it), and at the toy's tables over it (`planZHJ Bridge.toyDj y₀`)
the audit game under the law that draws the one unit has a prover `σ` whose finder `B^τ` (at a per-run time `t'` and a
verification `v`) has `J²·K·Pr[B^τ] ≤ 2^-200`, `K` the number of C-Flock's finders and `J` the plan's tables. Every
draw flock-verify `--zk` accepts gives the setup (`Flock.SecurityProofs.EndToEndHidden_refSetup`), so a run of
flock-verify on the deployed files checks it; the bound is `cr/sha-512`'s at a prover that finds no collision. -/
def ToySetupHJ : Prop :=
  ∃ (I : Inputs) (mPts : ℕ) (S₀ : Finset (Fin 1)) (d₀ : Lean.Json) (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (σ : Strategy (audit Bridge.toyLaw Unit (batchedSessionZC execArith hs512 (planZHJ Bridge.toyDj y₀))))
    (t' v : ℝ),
    (nTab I : ℝ) ^ 2 * Fintype.card (Bridge.PubIx (Reg := Unit) Bridge.toyDj y₀ Bridge.toyPartition
        (Bridge.toyBinding execArith (planZHJ Bridge.toyDj y₀) (tabZHJ Bridge.toyDj y₀) H512) Bridge.toyLaw) *
      (Bridge.pubFinder (Reg := Unit) Bridge.toyDj y₀ Bridge.toyPartition
        (Bridge.toyBinding execArith (planZHJ Bridge.toyDj y₀) (tabZHJ Bridge.toyDj y₀) H512) Bridge.kDep (2 ^ 192)
        (2 ^ 192) (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ)).Pr Claims.Sampling.sha512Msg ≤ 1 / 2 ^ 200

end Flock.Assumptions
