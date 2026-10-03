import NetTiming.Guarantees
import NetTiming.SecurityProofs.Occupancy
import NetTiming.SecurityProofs.Counting
import NetTiming.SecurityProofs.Advice
import NetTiming.SecurityProofs.Ingress
import NetTiming.SecurityProofs.Sanity

/-!
# Proofs of the network warden's guarantees

One theorem `NetTiming.SecurityProofs.G : NetTiming.Guarantees.G` per guarantee, each the theorem of the same statement
in `Occupancy` (Proposition 2), `Counting` (Theorem 1(b)), `Advice` (Theorem 3) or `Ingress` (the ingress bound), and
every lemma behind them. `Sanity` shows every named assumption satisfiable.
-/

namespace NetTiming.SecurityProofs

theorem CardFree : NetTiming.Guarantees.CardFree := NetTiming.card_free
theorem CardLeftPacked : NetTiming.Guarantees.CardLeftPacked := NetTiming.card_leftPacked
theorem FillInjective : NetTiming.Guarantees.FillInjective := @NetTiming.fill_injective
theorem ExistsFillOfLeftPacked : NetTiming.Guarantees.ExistsFillOfLeftPacked := @NetTiming.exists_fill_of_leftPacked
theorem CardCountsEqLe : NetTiming.Guarantees.CardCountsEqLe := NetTiming.card_counts_eq_le
theorem CardCountsLeLe : NetTiming.Guarantees.CardCountsLeLe := NetTiming.card_counts_le_le
theorem ReleasedSelf : NetTiming.Guarantees.ReleasedSelf := NetTiming.released_self
theorem SumReleasedLe : NetTiming.Guarantees.SumReleasedLe := NetTiming.sum_released_le
theorem EncardFifoGridsLeOfAvailable : NetTiming.Guarantees.EncardFifoGridsLeOfAvailable :=
  @NetTiming.encard_fifoGrids_le_of_available
theorem EncardFifoGrids : NetTiming.Guarantees.EncardFifoGrids := @NetTiming.encard_fifoGrids

theorem EncardLeProdOfBranching : NetTiming.Guarantees.EncardLeProdOfBranching := @NetTiming.encard_le_prod_of_branching
theorem EncardAcceptedRunsLe : NetTiming.Guarantees.EncardAcceptedRunsLe := @NetTiming.encard_acceptedRuns_le
theorem EncardObsDecodableLe : NetTiming.Guarantees.EncardObsDecodableLe := @NetTiming.encard_obsDecodable_le
theorem EncardDecodableLe : NetTiming.Guarantees.EncardDecodableLe := @NetTiming.encard_decodable_le
theorem NoFaultOfAccepted : NetTiming.Guarantees.NoFaultOfAccepted := @NetTiming.no_fault_of_accepted

theorem RecordedRowEqFill : NetTiming.Guarantees.RecordedRowEqFill := @NetTiming.recorded_row_eq_fill
theorem EncardAccSeqsLeAdvice : NetTiming.Guarantees.EncardAccSeqsLeAdvice := @NetTiming.encard_accSeqs_le_advice
theorem EncardAccSeqsLeSync : NetTiming.Guarantees.EncardAccSeqsLeSync := @NetTiming.encard_accSeqs_le_sync
theorem EncardAccSeqsLeFixedClock : NetTiming.Guarantees.EncardAccSeqsLeFixedClock :=
  @NetTiming.encard_accSeqs_le_fixedClock
theorem EncardAccSeqsLeBlocks : NetTiming.Guarantees.EncardAccSeqsLeBlocks := @NetTiming.encard_accSeqs_le_blocks
theorem EncardAccSeqsLeBlocksCounts : NetTiming.Guarantees.EncardAccSeqsLeBlocksCounts :=
  @NetTiming.encard_accSeqs_le_blocks_counts
theorem EncardDecodableLeBlocks : NetTiming.Guarantees.EncardDecodableLeBlocks :=
  @NetTiming.encard_decodable_le_blocks

theorem EncardIngressObsDecodableLe : NetTiming.Guarantees.EncardIngressObsDecodableLe :=
  @NetTiming.encard_ingress_obsDecodable_le
theorem EncardIngressDecodableLe : NetTiming.Guarantees.EncardIngressDecodableLe :=
  @NetTiming.encard_ingress_decodable_le
theorem EncardIngressObsDecodableLeContents : NetTiming.Guarantees.EncardIngressObsDecodableLeContents :=
  @NetTiming.encard_ingress_obsDecodable_le_contents
theorem EncardIngressDecodableLeContents : NetTiming.Guarantees.EncardIngressDecodableLeContents :=
  @NetTiming.encard_ingress_decodable_le_contents

end NetTiming.SecurityProofs
