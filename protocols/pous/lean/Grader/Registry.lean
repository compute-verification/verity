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

/-- The statements a submission may prove. Each must also be a `Prop`-valued definition in `Pous.Pinned` of the
trusted environment. -/
def pinnedTargets : List Name := [
  `Pous.Pinned.KFloor,
  `Pous.Pinned.Theorem1ii,
  `Pous.Pinned.Theorem1iiTimed,
  `Pous.Pinned.Theorem1iTimed,
  `Pous.Pinned.Theorem1iiSeq,
  `Pous.Pinned.Theorem1iiiTimed,
  `Pous.Pinned.FibreCount,
  `Pous.Pinned.A1CompressionCard,
  `Pous.Pinned.A1CompressionPr,
  `Pous.Pinned.A2LemmaAM1,
  `Pous.Pinned.A3Prime,
  `Pous.Pinned.A3PrimeLam,
  `Pous.Pinned.A5RabinGcd,
  `Pous.Pinned.A6RederiveUntimed,
  `Pous.Pinned.A6RederiveTimed,
  `Pous.Pinned.A7Params,
  `Pous.Pinned.M1Meets,
  `Pous.Pinned.P3ConcreteMeets64,
  `Pous.Pinned.P3ConcreteMeets14GB,
  `Pous.Pinned.P3ConcreteMeets64D12,
  `Pous.Pinned.BandChainMeets64,
  `Pous.Pinned.BandChainMeets64D5,
  `Pous.Pinned.BandChainMeets64D12,
  `Pous.Pinned.BandMultiMeets14,
  `Pous.Pinned.BandMultiMeets14K106,
  `Pous.Pinned.BandMultiMeetsFamily,
  `Pous.Pinned.ChainDenseMeets64,
  `Pous.Pinned.BandPubMeets64,
  `Pous.Pinned.BandPubMeets14,
  `Pous.Pinned.BandPubMeets14Global,
  `Pous.Pinned.ChainDenseMeets111,
  `Pous.Pinned.P2MeetsM1pB19Uncond,
  `Pous.Pinned.P2MeetsM1pFamilyUncond,
  `Pous.Pinned.P2MeetsM1pW8192Uncond,
  `Pous.Pinned.P2SlackB19Uncond,
  `Pous.Pinned.P2SlackFamilyUncond,
  `Pous.Pinned.P2ErrorSplitB19Uncond,
  `Pous.Pinned.B1IndepSet,
  `Pous.Pinned.B2Stacking,
  `Pous.Pinned.B3Chain,
  `Pous.Pinned.FischClaim13,
  `Pous.Pinned.FischClaim14,
  `Pous.Pinned.B5ExPostFactoBounded,
  `Pous.Pinned.B5ExPostFacto,
  `Pous.Pinned.TransposeMixing,
  `Pous.Pinned.ReverseNeedsAllChildren,
  `Pous.Pinned.N10Refuted,
  `Pous.Pinned.SampledAuditProd,
  `Pous.Pinned.SampledAuditSeq,
  `Pous.Pinned.SampledAuditSim,
  `Pous.Pinned.StackedAuditSeq,
  `Pous.Pinned.StackedAuditOnePercent,
  `Pous.Pinned.DenseMeets]
