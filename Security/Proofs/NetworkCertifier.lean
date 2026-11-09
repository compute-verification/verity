module

public import Definitions.NetworkCertifier.Guarantees
public import Definitions.NetworkCertifier.Guarantees.Pooled
public import Definitions.NetworkCertifier.Guarantees.Period
public import Definitions.NetworkCertifier.Guarantees.PooledLeak
public import Definitions.NetworkCertifier.Guarantees.Checked
public import Definitions.NetworkCertifier.Guarantees.Carried
public import Proofs.NetworkCertifier.Lemmas
public import Proofs.NetworkCertifier.Occupancy
public import Proofs.NetworkCertifier.Counting
public import Proofs.NetworkCertifier.Advice
public import Proofs.NetworkCertifier.Ingress
public import Proofs.NetworkCertifier.SampledAuditAccepted
public import Proofs.NetworkCertifier.SampledAuditInstance
public import Proofs.NetworkCertifier.SampledCharge
public import Proofs.NetworkCertifier.SampledLeak
public import Proofs.NetworkCertifier.SampledLeakReg
public import Proofs.NetworkCertifier.SampledLeakRegPlain
public import Proofs.NetworkCertifier.PeriodSum
public import Proofs.NetworkCertifier.Pooled
public import Proofs.NetworkCertifier.TapRows
public import Proofs.NetworkCertifier.Period
public import Proofs.NetworkCertifier.PooledLeak
public import Proofs.NetworkCertifier.Schedule
public import Proofs.NetworkCertifier.PerWindow
public import Proofs.NetworkCertifier.Sanity
public import Proofs.NetworkCertifier.Code
public import Proofs.NetworkCertifier.Accepted
public import Proofs.NetworkCertifier.Tap
public import Proofs.NetworkCertifier.RecordRows
public import Proofs.NetworkCertifier.Witnesses
public import Proofs.NetworkCertifier.Checked
public import Proofs.NetworkCertifier.WitnessesChecked
public import Proofs.NetworkCertifier.Carried
public import Proofs.NetworkCertifier.WitnessesCarried
public import Proofs.NetworkCertifier.PoolRoots
public import Definitions.NetworkCertifier.Guarantees.SampledCharge
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# Proofs of the network certifier's guarantees

One theorem `NetTiming.SecurityProofs.G : NetTiming.Guarantees.G` per guarantee and per lemma (`Lemmas`), each the
theorem of the same statement in `Occupancy` (Proposition 2), `Counting` (Theorem 1(b)), `Advice` (Theorem 3),
`Schedule` (Theorem 3 for the constant-rate Program), `Ingress` (the ingress bound), `PerWindow` (the last two over
whole windows) or `SampledCharge` (the leak bound under a sampled audit: the incorrect proof blocks' charge), and every lemma
behind them. `SampledLeak` composes `SampledCharge`'s count with `SampledAuditAccepted`, taken as a hypothesis
(`NetTiming.sampledLeak`), which the module `SampledAuditAccepted` proves, and `SampledLeakReg` moves its count to the
registration's values
(`NetTiming.sampledLeak_registered`);
`SampledLeakRegPlain` states it at the deployed plain record (`NetTiming.sampledLeak_registered_plain_sha512`). `Sanity` shows every named assumption satisfiable, and `Witnesses` gives each acceptance guarantee its honest and
nontrivial lemmas. `Code` proves the certifier's program
(`certifier_program`, `NetworkCertifier`), which the difftest runs against the Python audit, equal to these definitions.
`Accepted` proves `security_properties`' `AcceptedRunsAreProgramRuns` through `Code`: a run the network certificate
check accepts is a run of the constant-rate Program under the declared advice, so the guarantees here count it.
`SampledAuditAccepted` proves that an accepted sampled audit is `Accepted` except at `ε` at `δ` over the selection, at
any compiled sampled verifier where `Claims.SamplingGeneric.Compiled.KS` holds. `SampledAuditInstance` takes that
`Compiled.KS` from the compiler at Flock (`Claims.SamplingGeneric.flockKS`), at the assembled records.
`TapRows` bounds the leak at the network certifier's rows (`TapProof.sampledLeak_certifier`): an accepted audit's extracted
trace carries the network certifier's registered rows outside the incorrect proof blocks, unless its openings collide with the
network certifier's; `Period` counts with it. `Checked` proves the leak bound with checked inputs (`SampledLeakChecked`):
inputs a check pins cost no advice bits, and `WitnessesChecked` gives its nontrivial lemma.
`Carried` proves the leak bounds over a carried pool (`SampledLeakCarried`, `PooledLeakCarried`): one
run's relation at the rows its extracted trace holds, and the period's bound, where an opened row costs no advice bits,
since the chain fixes it unless two rows have one root, and the restarts serving picks cost `log₂` of the patterns it may
pick where every audit accepts. At the empty pool the period's bound is the bound with checked inputs over windows that
each hold a pool of their own (`pooledLeakChecked_of_carried`). `WitnessesCarried` gives their nontrivial lemmas and
`SampledLeakCarried`'s honest one. `PoolRoots` proves the period bound's premise on the opened rows' roots
(`CarriedRoots`) from each run's in-circuit check of its roots (`CheckedRoots`) and the record rule (`PoolChained`), as
the lemma `rootsOfChecked`.
-/

@[expose] public section

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

theorem EncardAccSeqsSampledLe : NetTiming.Guarantees.EncardAccSeqsSampledLe := @NetTiming.encard_accSeqsSampled_le
theorem SampledLeakAccepted : NetTiming.Guarantees.SampledLeakAccepted :=
  NetTiming.sampledLeak_reg NetTiming.SecurityProofs.SampledAuditAccepted
theorem SampledLeakRegistered : NetTiming.Guarantees.SampledLeakRegistered := by
  have hSA := @NetTiming.SecurityProofs.SampledAuditAccepted
  intro _ _ _ _ _ H D hks hb L K S lay k δ hδ reg ops β Dj hβ
  exact ⟨fun x hx hreg Ω _ _ P =>
      NetTiming.sampledLeak_registered hSA H D hks hb L K S lay k δ hδ reg ops Dj x hx hreg Ω P,
    fun x hx s P hs hP op =>
      (NetTiming.sampledLeak_reg hSA H D hks L K S lay k δ hδ reg β Dj hβ).2 x hx s P hs hP op _⟩

theorem PeriodLeakAccepted : NetTiming.Guarantees.PeriodLeakAccepted := NetTiming.periodLeak
theorem PooledIncorrectBlocksLe : NetTiming.Guarantees.PooledIncorrectBlocksLe := NetTiming.pooledIncorrectBlocks
theorem PooledLeakAccepted : NetTiming.Guarantees.PooledLeakAccepted := NetTiming.pooledLeak
theorem SampledLeakChecked : NetTiming.Guarantees.SampledLeakChecked := NetTiming.sampledLeakChecked
theorem SampledLeakCarried : NetTiming.Guarantees.SampledLeakCarried := NetTiming.sampledLeakCarried
theorem PooledLeakCarried : NetTiming.Guarantees.PooledLeakCarried := NetTiming.pooledLeakCarried

end NetTiming.SecurityProofs
