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

/-- The statements a submission may prove: the frozen statements of `Security/Definitions/Pous/Guarantees.lean` and the files
under `Security/Definitions/Pous/Guarantees/`, and six of the lemmas of `Security/Proofs/Pous/Statements.lean`, `KFloor` and
`Theorem1ii` (`check.sh`'s controls) and the open `A2LemmaAM1`, `A5RabinGcd`, `A7Params` and `M1Meets`. Each must also be
a `Prop`-valued definition in `Pous.Guarantees` of the trusted environment. -/
def pinnedTargets : List Name := [
  `Pous.Guarantees.KFloor,
  `Pous.Guarantees.Theorem1ii,
  `Pous.Guarantees.A2LemmaAM1,
  `Pous.Guarantees.A5RabinGcd,
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
  `Pous.Guarantees.ContinuousAudit,
  `Pous.Guarantees.BackgroundRebuildExcluded,
  `Pous.Guarantees.BandFamilyContinuousTimeAverage,
  `Pous.Guarantees.P2MeetsM1pB19Uncond,
  `Pous.Guarantees.P2MeetsM1pFamilyUncond,
  `Pous.Guarantees.P2MeetsM1pW8192Uncond,
  `Pous.Guarantees.P2SlackB19Uncond,
  `Pous.Guarantees.P2SlackFamilyUncond,
  `Pous.Guarantees.P2ErrorSplitB19Uncond,
  `Pous.Guarantees.DenseMeets]
