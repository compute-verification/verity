import Definitions.NetworkCertifier.Guarantees
import Definitions.NetworkCertifier.Guarantees.AdviceString
import Definitions.NetworkCertifier.Guarantees.AdviceV1
import Definitions.NetworkCertifier.Guarantees.WindowCharge
import Definitions.NetworkCertifier.Guarantees.Warden
import Definitions.NetworkCertifier.Guarantees.Horizon
import Definitions.NetworkCertifier.Guarantees.Period
import Definitions.NetworkCertifier.Guarantees.HorizonLeak
import Proofs.NetworkCertifier.Lemmas
import Proofs.NetworkCertifier.Occupancy
import Proofs.NetworkCertifier.Counting
import Proofs.NetworkCertifier.Advice
import Proofs.NetworkCertifier.Ingress
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.SampledAuditInstance
import Proofs.NetworkCertifier.SampledCharge
import Proofs.NetworkCertifier.SampledLeak
import Proofs.NetworkCertifier.SampledLeakReg
import Proofs.NetworkCertifier.SampledLeakRegPlain
import Proofs.NetworkCertifier.PeriodSum
import Proofs.NetworkCertifier.Horizon
import Proofs.NetworkCertifier.TapRows
import Proofs.NetworkCertifier.Period
import Proofs.NetworkCertifier.HorizonLeak
import Proofs.NetworkCertifier.Schedule
import Proofs.NetworkCertifier.PerWindow
import Proofs.NetworkCertifier.Sanity
import Proofs.NetworkCertifier.Code
import Proofs.NetworkCertifier.Accepted
import Proofs.NetworkCertifier.AdviceString
import Proofs.NetworkCertifier.AdviceV1
import Proofs.NetworkCertifier.WindowCharge
import Proofs.NetworkCertifier.Warden
import Proofs.NetworkCertifier.Tap
import Proofs.NetworkCertifier.RecordRows

/-!
# Proofs of the network certifier's guarantees

One theorem `NetTiming.SecurityProofs.G : NetTiming.Guarantees.G` per guarantee and per lemma (`Lemmas`), each the
theorem of the same statement in `Occupancy` (Proposition 2), `Counting` (Theorem 1(b)), `Advice` (Theorem 3),
`Schedule` (Theorem 3 for the constant-rate Program), `Ingress` (the ingress bound), `PerWindow` (the last two over
whole windows), `AdviceString` (the leak bound: the advice string's charge), `AdviceV1` (advice v1's charge counts, and
the leak bound with it), `WindowCharge` (advice v1's whole-window charge counts, and the leak bound with it),
`SampledCharge` (the leak bound under a sampled audit: the faulty units' charge) or `Warden`
(the warden's channel bound),
and every lemma behind them. `SampledLeak` composes `SampledCharge`'s count with `SampledAuditAccepted`, taken as a hypothesis until
PRF-144 proves it (`NetTiming.sampledLeak`), and `SampledLeakReg` moves its count to the registration's values
(`NetTiming.sampledLeak_registered`);
`SampledLeakRegPlain` states it at the deployed plain record (`NetTiming.sampledLeak_registered_plain_sha512`). `Sanity` shows every named assumption satisfiable. `Code` proves the certifier's program
(`certifier_program`, `NetworkCertifier`), which the difftest runs against the Python audit, equal to these definitions.
`Accepted` proves `security_properties`' `AcceptedRunsAreProgramRuns` through `Code`: a run the network certificate
check accepts is a run of the constant-rate Program under the declared advice, so the guarantees here count it.
`SampledAuditAccepted` proves that an accepted sampled audit is `Accepted` except at `ε` at `δ` over the selection.
`SampledAuditInstance` takes its `KS` hypothesis from `Claims.Sampling.sampling_assembled_of`, at the assembled records.
`TapRows` bounds the leak at the network certifier's rows (`TapProof.sampledLeak_tap`): an accepted audit's extracted
trace carries the network certifier's registered rows outside the faulty units, unless its openings collide with the
network certifier's; `Period` counts with it.
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

theorem EncardAccSeqsWithinLeConstantRate : NetTiming.Guarantees.EncardAccSeqsWithinLeConstantRate :=
  fun bits hbits b T ss frames op n D m _ hm =>
    NetTiming.encard_accSeqsWithin_le_constantRate bits hbits b T ss frames op n D m hm
theorem EncardDecodableWithinLeCompiled : NetTiming.Guarantees.EncardDecodableWithinLeCompiled :=
  fun I T ss bits hbits b compilerOf env hpre hdet n D m _ hm =>
    NetTiming.encard_decodableWithin_le_compiled I T ss bits hbits b compilerOf env hpre hdet n D m hm
theorem EncardDecodableWithinLeCompiledPerWindow : NetTiming.Guarantees.EncardDecodableWithinLeCompiledPerWindow :=
  fun I T ss bits hbits b compilerOf env hpre hdet D =>
    NetTiming.encard_decodableWithin_le_compiled_perWindow I T ss bits hbits b compilerOf env hpre hdet D

theorem EncardChargedLe : NetTiming.Guarantees.EncardChargedLe := NetTiming.AdviceV1.encard_charged_le
theorem EncardAccSeqsWithinLeConstantRateV1 : NetTiming.Guarantees.EncardAccSeqsWithinLeConstantRateV1 :=
  fun bk N b T ss frames op n D m _ hm => NetTiming.encard_accSeqsWithin_le_constantRate_v1 bk N b T ss frames op n D m hm
theorem EncardDecodableWithinLeCompiledV1 : NetTiming.Guarantees.EncardDecodableWithinLeCompiledV1 :=
  fun I T ss bk N b compilerOf env hpre hdet n D m _ hm =>
    NetTiming.encard_decodableWithin_le_compiled_v1 I T ss bk N b compilerOf env hpre hdet n D m hm
theorem EncardDecodableWithinLeCompiledPerWindowV1 : NetTiming.Guarantees.EncardDecodableWithinLeCompiledPerWindowV1 :=
  fun I T ss bk N b compilerOf env hpre hdet D =>
    NetTiming.encard_decodableWithin_le_compiled_perWindow_v1 I T ss bk N b compilerOf env hpre hdet D
theorem EncardChargedLeCounted : NetTiming.Guarantees.EncardChargedLeCounted :=
  fun bk N W c cc => NetTiming.AdviceV1.encard_chargedCounted_le bk N W c cc
theorem EncardAccSeqsWithinLeConstantRateV1Counted : NetTiming.Guarantees.EncardAccSeqsWithinLeConstantRateV1Counted :=
  fun bk N cc b T ss frames op n D m _ hm =>
    NetTiming.encard_accSeqsWithin_le_constantRate_v1Counted bk N cc b T ss frames op n D m hm
theorem EncardDecodableWithinLeCompiledV1Counted : NetTiming.Guarantees.EncardDecodableWithinLeCompiledV1Counted :=
  fun I T ss bk N cc b compilerOf env hpre hdet n D m _ hm =>
    NetTiming.encard_decodableWithin_le_compiled_v1Counted I T ss bk N cc b compilerOf env hpre hdet n D m hm
theorem EncardDecodableWithinLeCompiledPerWindowV1Counted :
    NetTiming.Guarantees.EncardDecodableWithinLeCompiledPerWindowV1Counted :=
  fun I T ss bk N cc b compilerOf env hpre hdet D =>
    NetTiming.encard_decodableWithin_le_compiled_perWindow_v1Counted I T ss bk N cc b compilerOf env hpre hdet D
theorem EncardChargedLeWhole : NetTiming.Guarantees.EncardChargedLeWhole := NetTiming.WindowCharge.encard_charged_le
theorem EncardAccSeqsWithinLeConstantRateV1Whole : NetTiming.Guarantees.EncardAccSeqsWithinLeConstantRateV1Whole :=
  fun bk N H O b T ss frames op n D m _ hm =>
    NetTiming.encard_accSeqsWithin_le_constantRate_v1Whole bk N H O b T ss frames op n D m hm
theorem EncardDecodableWithinLeCompiledV1Whole : NetTiming.Guarantees.EncardDecodableWithinLeCompiledV1Whole :=
  fun I T ss bk N H O b compilerOf env hpre hdet n D m _ hm =>
    NetTiming.encard_decodableWithin_le_compiled_v1Whole I T ss bk N H O b compilerOf env hpre hdet n D m hm
theorem EncardDecodableWithinLeCompiledPerWindowV1Whole :
    NetTiming.Guarantees.EncardDecodableWithinLeCompiledPerWindowV1Whole :=
  fun I T ss bk N H O b compilerOf env hpre hdet D =>
    NetTiming.encard_decodableWithin_le_compiled_perWindow_v1Whole I T ss bk N H O b compilerOf env hpre hdet D

theorem EncardAccSeqsSampledLe : NetTiming.Guarantees.EncardAccSeqsSampledLe := @NetTiming.encard_accSeqsSampled_le
theorem SampledLeakAccepted : NetTiming.Guarantees.SampledLeakAccepted :=
  fun hSA => NetTiming.sampledLeak_reg hSA
theorem SampledLeakRegistered : NetTiming.Guarantees.SampledLeakRegistered := by
  intro hSA _ _ _ _ _ _ _ _ H D hks hb L K S lay k δ decode hδ hdec reg ops β Dj hβ
  exact ⟨fun x hx hreg Ω _ _ P =>
      NetTiming.sampledLeak_registered hSA H D hks hb L K S lay k δ decode hδ hdec reg ops Dj x hx hreg Ω P,
    fun x hx s P hs hP op =>
      (NetTiming.sampledLeak_reg hSA H D hks L K S lay k δ decode hδ hdec reg β Dj hβ).2 x hx s P hs hP op _⟩

theorem WardenViewEqClientView : NetTiming.Guarantees.WardenViewEqClientView := @NetTiming.wardenView_eq_clientView
theorem EncardWardenViewsLe : NetTiming.Guarantees.EncardWardenViewsLe := @NetTiming.encard_wardenViews_le
theorem EncardWardenDecodableLe : NetTiming.Guarantees.EncardWardenDecodableLe :=
  @NetTiming.encard_wardenDecodable_le
theorem PeriodLeakAccepted : NetTiming.Guarantees.PeriodLeakAccepted := NetTiming.periodLeak
theorem HorizonWrongUnitsLe : NetTiming.Guarantees.HorizonWrongUnitsLe := NetTiming.horizonWrongUnits
theorem HorizonLeakAccepted : NetTiming.Guarantees.HorizonLeakAccepted := NetTiming.horizonLeak

end NetTiming.SecurityProofs
