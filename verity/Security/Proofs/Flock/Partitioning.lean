import Specs.Flock.Guarantees.Partitioning
import Proofs.Flock.Soundness.Audit.Partitioning

/-!
# Proofs of the partitioning guarantees

One theorem `Flock.SecurityProofs.Partitioning.G : Flock.Guarantees.Partitioning.G` per guarantee, which is the theorem
in `FlockSoundness.Audit.Partitioning.SecurityProofs` or, for the instances showing they are not vacuous (`Witness`),
`FlockSoundness.Audit.Partitioning.Witness` that proves the same statement under its old name.
-/

namespace Flock.SecurityProofs.Partitioning

open FlockSoundness.Audit.Partitioning

theorem ConverseExact : Flock.Guarantees.Partitioning.ConverseExact := SecurityProofs.ConverseExact
theorem ConverseAsymptotic : Flock.Guarantees.Partitioning.ConverseAsymptotic := SecurityProofs.ConverseAsymptotic
theorem Achievability : Flock.Guarantees.Partitioning.Achievability := SecurityProofs.Achievability
theorem GranularityOptimum : Flock.Guarantees.Partitioning.GranularityOptimum := SecurityProofs.GranularityOptimum
theorem GranularityChoice : Flock.Guarantees.Partitioning.GranularityChoice := SecurityProofs.GranularityChoice
theorem HeavyUnit : Flock.Guarantees.Partitioning.HeavyUnit := SecurityProofs.HeavyUnit
theorem UnequalDraws : Flock.Guarantees.Partitioning.UnequalDraws := SecurityProofs.UnequalDraws
theorem UnequalCost : Flock.Guarantees.Partitioning.UnequalCost := SecurityProofs.UnequalCost
theorem ExfiltrationOptimum : Flock.Guarantees.Partitioning.ExfiltrationOptimum := SecurityProofs.ExfiltrationOptimum
theorem RefinementRule : Flock.Guarantees.Partitioning.RefinementRule := SecurityProofs.RefinementRule
theorem RiskConcave : Flock.Guarantees.Partitioning.RiskConcave := SecurityProofs.RiskConcave
theorem Concentration : Flock.Guarantees.Partitioning.Concentration := SecurityProofs.Concentration
theorem NestedEscape : Flock.Guarantees.Partitioning.NestedEscape := SecurityProofs.NestedEscape

namespace Witness

theorem Converse : Flock.Guarantees.Partitioning.Witness.Converse := Witness.converse
theorem Achievability : Flock.Guarantees.Partitioning.Witness.Achievability := Witness.achievability
theorem Granularity : Flock.Guarantees.Partitioning.Witness.Granularity := Witness.granularity
theorem HeavyUnit : Flock.Guarantees.Partitioning.Witness.HeavyUnit := Witness.heavyUnit
theorem Unequal : Flock.Guarantees.Partitioning.Witness.Unequal := Witness.unequal
theorem Exfiltration : Flock.Guarantees.Partitioning.Witness.Exfiltration := Witness.exfiltration
theorem Refinement : Flock.Guarantees.Partitioning.Witness.Refinement := Witness.refinement
theorem RiskConcaveNeedsKTwo : Flock.Guarantees.Partitioning.Witness.RiskConcaveNeedsKTwo :=
  Witness.riskConcave_needs_k_two
theorem RiskConcaveNeedsSmallP : Flock.Guarantees.Partitioning.Witness.RiskConcaveNeedsSmallP :=
  Witness.riskConcave_needs_small_p
theorem Concentration : Flock.Guarantees.Partitioning.Witness.Concentration := Witness.concentration
theorem NestedEscape : Flock.Guarantees.Partitioning.Witness.NestedEscape := Witness.nestedEscape

end Witness

end Flock.SecurityProofs.Partitioning
