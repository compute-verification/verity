import FlockSoundness.ZK.GK.Theorem
import FlockSoundness.Discharge.ZkEveryCoin.Headline
import RunModel.Guarantees

/-!
# The run's zero knowledge, from hm96 hiding and C-Flock's Goldreich–Kahan simulator (plan; not built)

The simulator (`simulation`) runs the auditor's code (`net`) against a boundary of its own, which has no witness:

1. Every leaf the proxy and the ZK layer salt (`RunModel.leaves`: registration roots, inner rows, the masks' salts) is
   `hide` of a fixed value. `leaf_hybrids` moves the real view to that one leaf at a time, two hiding steps a leaf
   (`hiding_fun` both ways), within `2 * leaves A * δ₁`.
2. The send gate's outer messages are V*'s session on a dummy witness, and the simulator is GK on it (`ZK.GK.gk`, a
   randomized auditor through `gk_simulate_avg`): Lemma B and Lemma C at `2·|LeafPos|·2^-193` each, by
   `zk_session_view_all` at every coin vector, refused ones included; the coin tree's binding is
   `gk_simulate_coins_keyed`'s finder, `simulation`'s.
3. The leak fixes the frames' times, destinations and shapes, so the simulator emits its frames on the real schedule.
   The rack's events reach the auditor only through the boundary (`World.byGpu`), so the GPU strategy enters the view
   only through the leak, which is why `ideal` draws the leak from the real run.
-/

namespace RunModelProofs

open RunModel Run
open scoped ENNReal

/-- GK on the auditor's code with the boundary's dummy run, at completion threshold `t` and cap `M`, and its finder
(`bindCollKeyed`). -/
noncomputable def simulation (A : Agreement) (t M : ℕ) : Simulation A := sorry

/-- GK's terms beyond `cr/sha-512`'s: `3δ_C + 8t·δ_B + 3^-t + t/M`, with `t` and `M` the simulator's. -/
noncomputable def gkError (A : Agreement) (t M : ℕ) : ℝ≥0∞ := sorry

/-- The real run with every salted leaf a hash of a fixed value. -/
noncomputable def blanked (A : Agreement) (fuel : ℕ) (net gpu : Adversary) : PMF (View × Leak) := sorry

/-- Step 1. -/
theorem leaf_hybrids (A : Agreement) (δ₁ : ℝ≥0∞) (h : Assumptions.Hm96Hiding δ₁) (fuel : ℕ) (net gpu : Adversary) :
    tv (real A fuel net gpu) (blanked A fuel net gpu) ≤ 2 * leaves A * δ₁ := sorry

/-- Step 2. -/
theorem blanked_gk (A : Agreement) (t M : ℕ) (fuel : ℕ) (net gpu : Adversary) (q : ℝ)
    (hcr : ∀ f ∈ (simulation A t M).finders fuel net, Assumptions.FinderCR f q) :
    tv (blanked A fuel net gpu) (ideal A fuel net gpu (simulation A t M).sim) ≤
      gkError A t M + Assumptions.collisions ((simulation A t M).finders fuel net).length q := sorry

/-- The guarantee. -/
theorem zeroKnowledge (A : Agreement) (t M : ℕ) : Guarantees.ZeroKnowledge A (simulation A t M) (gkError A t M) := sorry

end RunModelProofs
