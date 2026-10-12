import Models.NetworkCertifier.Guarantees
import Proofs.NetworkCertifier.Lemmas
import Proofs.NetworkCertifier.Occupancy
import Proofs.NetworkCertifier.Counting
import Proofs.NetworkCertifier.Advice
import Proofs.NetworkCertifier.Ingress
import Proofs.NetworkCertifier.Schedule
import Proofs.NetworkCertifier.PerWindow
import Proofs.NetworkCertifier.Sanity
import Proofs.NetworkCertifier.Code
import Proofs.NetworkCertifier.Accepted

/-!
# Proofs of the network certifier's guarantees

One theorem `NetTiming.SecurityProofs.G : NetTiming.Guarantees.G` per guarantee and per lemma (`Lemmas`), each the theorem of the same statement
in `Occupancy` (Proposition 2), `Counting` (Theorem 1(b)), `Advice` (Theorem 3), `Schedule` (Theorem 3 for the
constant-rate Program), `Ingress` (the ingress bound) or `PerWindow` (the last two over whole windows), and every lemma
behind them. `Sanity` shows every named assumption satisfiable. `Code` proves the certifier's program (`certifier_program`,
`NetworkCertifier`), which the difftest runs against the Python audit, equal to these definitions. `Accepted` proves
`security_properties`' `AcceptedRunsAreProgramRuns` through `Code`: a run the network certificate check accepts is a run
of the constant-rate Program under the declared advice, so the guarantees here count it.
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

theorem EncardAccSeqsLeConstantRate : NetTiming.Guarantees.EncardAccSeqsLeConstantRate :=
  fun T W ss frames op n D _ => NetTiming.encard_accSeqs_le_constantRate T W ss frames op n D
theorem EncardAccSeqsLeConstantRateFixedClock : NetTiming.Guarantees.EncardAccSeqsLeConstantRateFixedClock :=
  fun T W ss hfix frames op n D _ => NetTiming.encard_accSeqs_le_constantRate_fixedClock T W ss hfix frames op n D
theorem ExpectedRowCausal : NetTiming.Guarantees.ExpectedRowCausal := @NetTiming.expectedRow_causal
theorem EncardAccSeqsLeConstantRateWindows : NetTiming.Guarantees.EncardAccSeqsLeConstantRateWindows :=
  fun T W ss frames op n D m _ hm => NetTiming.encard_accSeqs_le_constantRate_windows T W ss frames op n D m hm
theorem EncardDecodableLeConstantRate : NetTiming.Guarantees.EncardDecodableLeConstantRate :=
  fun I T W ss frames n D m _ hm => NetTiming.encard_decodable_le_constantRate I T W ss frames n D m hm

theorem EncardIngressObsDecodableLe : NetTiming.Guarantees.EncardIngressObsDecodableLe :=
  @NetTiming.encard_ingress_obsDecodable_le
theorem EncardIngressDecodableLe : NetTiming.Guarantees.EncardIngressDecodableLe :=
  @NetTiming.encard_ingress_decodable_le
theorem EncardIngressObsDecodableLeContents : NetTiming.Guarantees.EncardIngressObsDecodableLeContents :=
  @NetTiming.encard_ingress_obsDecodable_le_contents
theorem EncardIngressDecodableLeContents : NetTiming.Guarantees.EncardIngressDecodableLeContents :=
  @NetTiming.encard_ingress_decodable_le_contents

theorem EncardAccSeqsLeConstantRatePerWindow : NetTiming.Guarantees.EncardAccSeqsLeConstantRatePerWindow :=
  fun T W ss frames op m D _ => NetTiming.encard_accSeqs_le_constantRate_perWindow T W ss frames op m D
theorem EncardDecodableLeConstantRatePerWindow : NetTiming.Guarantees.EncardDecodableLeConstantRatePerWindow :=
  fun I T W ss frames m D _ => NetTiming.encard_decodable_le_constantRate_perWindow I T W ss frames m D
theorem EncardIngressObsDecodableLePerWindow : NetTiming.Guarantees.EncardIngressObsDecodableLePerWindow :=
  @NetTiming.encard_ingress_obsDecodable_le_perWindow
theorem EncardIngressObsDecodableLeContentsPerWindow :
    NetTiming.Guarantees.EncardIngressObsDecodableLeContentsPerWindow :=
  @NetTiming.encard_ingress_obsDecodable_le_contents_perWindow

end NetTiming.SecurityProofs
