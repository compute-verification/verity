import RunModel.Protocol.Run

/-!
# What the auditor sees, and what a simulator gets (trusted)

The auditor's view is the network's (`World.netView`): every frame that crosses the boundary, either way, with its
tick, and every frame among the auditor's own parties, which in this setting are the adversary's. The simulator gets
the network's strategy (the auditor's code, which it may run on any input, so it may rewind) and the leak
(`World.leak`): when each frame left the boundary, to whom, and its shape. It never gets the GPU rack's strategy, the
developer's data, or the boundary's fresh bytes.

The leak is what zero knowledge concedes: the rack chooses when it sends, and a message the send gate refuses shows as
a frame that never leaves. It is a timing and abort channel, the network warden's to bound, not this model's.

A simulator that rewinds the auditor (C-Flock's Goldreich–Kahan, `ZK.GK.gk`) fails where the auditor opens a committed
coin two ways; that is a SHA-512 collision for a finder built from the auditor's code (`gk_simulate_coins_keyed`). A
`Simulation` names both, and as with an `Extraction` its definition is trusted text: a finder that hard-codes a
collision makes the statement vacuous.
-/

namespace RunModel

open Run
open scoped ENNReal

/-- Total variation distance. -/
noncomputable def tv {α : Type} (p q : PMF α) : ℝ≥0∞ := ⨆ E : Set α, p.toOuterMeasure E - q.toOuterMeasure E

abbrev View := List (ℕ × Frame)
abbrev Leak := List (ℕ × Node × ℕ × String)

abbrev Simulator := ℕ → Adversary → Leak → PMF View

structure Simulation (A : Agreement) where
  sim : Simulator
  /-- the finders built from the auditor's code that a conflicting coin opening reduces to -/
  finders : ℕ → Adversary → List Finder

/-- The auditor's view with the leak, in the real run of `fuel` ticks. -/
noncomputable def real (A : Agreement) (fuel : ℕ) (net gpu : Adversary) : PMF (View × Leak) :=
  (zkRun A net gpu fuel (World.init _)).map fun w => (w.netView, w.leak)

/-- The simulated view beside the real leak: the leak drawn as in the real run, the view from the simulator given it. -/
noncomputable def ideal (A : Agreement) (fuel : ℕ) (net gpu : Adversary) (Sim : Simulator) : PMF (View × Leak) :=
  ((real A fuel net gpu).map Prod.snd).bind fun l => (Sim fuel net l).map fun v => (v, l)

/-- The hm96 leaves whose salts the boundary draws: one per registration root, one per inner row, the ZK layer's. -/
def leaves (A : Agreement) : ℕ :=
  A.roots + A.batches * ((List.range A.rounds).map A.innerRows).sum + A.maskSalts

end RunModel
