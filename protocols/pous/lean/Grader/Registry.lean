import Lean

/-!
# The grader's registry, compiled into the grader

A submission cannot change these at grading time: they are part of the grader binary, whose hash `grade.sh` checks
before and after the submission builds.
-/

open Lean

/-- The axioms a graded proof may depend on: Lean's three. Named assumptions are hypotheses of a pinned statement,
never axioms (Verity's policy), so this list does not grow. -/
def allowedAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

/-- The statements a submission may prove. Each must also be a `Prop`-valued definition in `Pous.Guarantees` of the
trusted environment. -/
def pinnedTargets : List Name := [
  `Pous.Guarantees.KFloor,
  `Pous.Guarantees.Theorem1ii,
  `Pous.Guarantees.Theorem1iiTimed,
  `Pous.Guarantees.Theorem1iTimed,
  `Pous.Guarantees.Theorem1iiSeq,
  `Pous.Guarantees.Theorem1iiiTimed,
  `Pous.Guarantees.FibreCount,
  `Pous.Guarantees.A1CompressionCard,
  `Pous.Guarantees.A1CompressionPr,
  `Pous.Guarantees.A2LemmaAM1,
  `Pous.Guarantees.A3Prime,
  `Pous.Guarantees.A3PrimeLam,
  `Pous.Guarantees.A5RabinGcd,
  `Pous.Guarantees.A6RederiveUntimed,
  `Pous.Guarantees.A6RederiveTimed,
  `Pous.Guarantees.A7Params,
  `Pous.Guarantees.M1Meets,
  `Pous.Guarantees.P3ConcreteMeets64,
  `Pous.Guarantees.P3ConcreteMeets14GB,
  `Pous.Guarantees.P3ConcreteMeets64D12,
  `Pous.Guarantees.BandChainMeets64,
  `Pous.Guarantees.BandChainMeets64D5,
  `Pous.Guarantees.BandChainMeets64D12,
  `Pous.Guarantees.BandMultiMeets14,
  `Pous.Guarantees.BandMultiMeets14K106,
  `Pous.Guarantees.BandMultiMeetsFamily,
  `Pous.Guarantees.ChainDenseMeets64,
  `Pous.Guarantees.BandMultiMeetsFamilyD2,
  `Pous.Guarantees.BandMultiMeetsFamilyD1,
  `Pous.Guarantees.BandMultiMeetsFamilyD0,
  `Pous.Guarantees.BandChainMeets64Rounds2,
  `Pous.Guarantees.BandChainMeets64Rounds1,
  `Pous.Guarantees.BandPubMeets64,
  `Pous.Guarantees.BandPubMeets14,
  `Pous.Guarantees.BandPubMeets14Global,
  `Pous.Guarantees.BandPubMeetsFamily,
  `Pous.Guarantees.BandPubMeetsFamilyD2,
  `Pous.Guarantees.BandPubMeetsFamilyD1,
  `Pous.Guarantees.ChainDenseMeets111,
  `Pous.Guarantees.BandGradedMeetsFamily,
  `Pous.Guarantees.DigestAU,
  `Pous.Guarantees.SecureErasureMeets,
  `Pous.Guarantees.SecureErasureMeetsFamily,
  `Pous.Guarantees.SecureErasureMeets96GiB,
  `Pous.Guarantees.SecureErasureMeetsDraw,
  `Pous.Guarantees.PageFillAuditSecure,
  `Pous.Guarantees.SecureErasurePagesFamily,
  `Pous.Guarantees.SecureErasurePages96112,
  `Pous.Guarantees.P2MeetsM1pB19Uncond,
  `Pous.Guarantees.P2MeetsM1pFamilyUncond,
  `Pous.Guarantees.P2MeetsM1pW8192Uncond,
  `Pous.Guarantees.P2SlackB19Uncond,
  `Pous.Guarantees.P2SlackFamilyUncond,
  `Pous.Guarantees.P2ErrorSplitB19Uncond,
  `Pous.Guarantees.B1IndepSet,
  `Pous.Guarantees.B2Stacking,
  `Pous.Guarantees.B3Chain,
  `Pous.Guarantees.FischClaim13,
  `Pous.Guarantees.FischClaim14,
  `Pous.Guarantees.B5ExPostFactoBounded,
  `Pous.Guarantees.B5ExPostFacto,
  `Pous.Guarantees.TransposeMixing,
  `Pous.Guarantees.ReverseNeedsAllChildren,
  `Pous.Guarantees.N10Refuted,
  `Pous.Guarantees.SampledAuditProd,
  `Pous.Guarantees.SampledAuditSeq,
  `Pous.Guarantees.SampledAuditSim,
  `Pous.Guarantees.StackedAuditSeq,
  `Pous.Guarantees.StackedAuditOnePercent,
  `Pous.Guarantees.DenseMeets]
