import Definitions.Accounting.Compose
import Models.Accounting.Assumptions.TS

/-!
# The compiled circuit's named assumptions (trusted)

The premises the composed theorem (`Models.Accounting.Guarantees.Compose`, note:20261018T0346Z-draft-theorem-compose)
adds to the tensor–scalar model's and the data sheet's, each a `Prop` taken as a hypothesis, never an `axiom`. Both are
open.

* `TemplateNoise`: (H-μ)'s computational half, per template. What the coins can give is proved: a Program the Build
  admits keeps a noise injection's coins out of every view outside its cone (`Guarantees.CoinFlow.ViewIndependent`,
  `PointMass`), and fresh bits give the point mass (`Guarantees.HMu.CoinPointMass`). That a skipping cheater's view is
  such a view, that it hasn't computed the product its skipped value is part of, no distribution of coins supplies
  (`Guarantees.HMu`'s notes).
* `SeedGrind`: the hash-derived seeds' grinding term. It is a statement about SHA-512, so under the 8 Oct ruling it is
  owed as a random-oracle bound, frozen (no-function, hash) as `RemoteSessionConservation` is, and no guarantee may
  land resting on it as a premise.
-/

namespace Accounting.Assumptions.Compose

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

/-- `accounting-compose-template-noise` (computational (H-μ), per template; open). **A skipped unit bets on its own
fresh noise**: in every round, at each unit's stage, whatever the past, the stage's coin splits into the unit's
`bitsOf u` fresh noise bits and the rest, the cheater's decision to skip the unit and its committed receipt are
functions of the rest, the unit's skipped value is the bits' sum with nonzero integer weights (the other operand's
entries), and its honest receipt has fibre at most `fibreOf u` there. Rule (b) (`CoinFlow.Program.OneUse`) makes the
bits the unit's own; that the cheater's view doesn't read them is the claim. The weights being the other operand's
entries, `Scheme.bits` is a floor on a skipped value's nonzero-weight bits: a zero entry takes its bit out. -/
def TemplateNoise {κ K C D : Type} {M s n : ℕ} (Sc : Scheme κ K) (Cd : Declared κ)
    (W : StageWindow C (UnitOf Sc Cd) D M s n) : Prop :=
  ∀ u, windowNoise (stageOf Sc Cd) u (bitsOf Sc Cd u) (fibreOf Sc Cd u) W

/-- `accounting-compose-seed-grind` (random oracle, SHA-512; open, owed as a proof). **Grinding the seeds costs at most
`T·μ^m`**: the deployed play `Pd`, whose subcircuit seeds are SHA-512 of the operands' roots with their hiding salts
(decisions 101 and 102), is accepted below time `c` with probability at most the live play `Pl`'s, whose seeds are
coins released after the operands' commitments, plus `T·μ^m`: a re-roll of a root through its salts is one try, a try
lands a steer that needs `m` coordinates with probability at most `μ^m`, and the cheater makes at most `T` tries a
window (Daniel, 18 Oct 02:39Z). -/
def SeedGrind {Ωd Ωl : Type} [Fintype Ωd] [Fintype Ωl] {S : ℕ} {K T L U : Type} (Pd : Play Ωd S K T L U)
    (Pl : Play Ωl S K T L U) (c : ℝ) (Tq : ℕ) (μ : ℝ) (m : ℕ) : Prop :=
  prob (fun x => Pd.accepts x ∧ (Pd.run x).run.time < c) ≤
    prob (fun y => Pl.accepts y ∧ (Pl.run y).run.time < c) + Tq * μ ^ m

end Accounting.Assumptions.Compose
