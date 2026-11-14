module

public import Definitions.NetworkCertifier.Guarantees.MonitoringPeriodFails
public import Proofs.NetworkCertifier.MonitoringPeriodCheckedChecks
public import Proofs.NetworkCertifier.MonitoringPeriodCheckedExclusion
public import Proofs.NetworkCertifier.MonitoringPeriodFails
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
import Mathlib.Order.CompletePartialOrder
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob
import Std.Tactic.BVDecide.Normalize.Bool

/-!
# Checks of the period statements with each failure charged: the factor is met, and the weights must sum to one

`MonitoringPeriodLeakFails`, `MonitoringPeriodLeakFailsAtMost` and `MonitoringPeriodLeakCharged` let the operator's moves and the registered values
follow the network certifier's fail cells, and pay for it: a factor `failPatterns bound j` on the count, or a charge
under weights on the number of failures (`KraftWeights`). These checks are at the monitoring period of one placed
window that `Proofs.NetworkCertifier.MonitoringPeriodCheckedExclusion` builds: the code `posL`, which rejects every run, the
record format `bitFmt` of one connection of one cell with no data, under the one schedule `oneSched`, with the secret
whether the server's end is late for its bucket (`bitArr`). There the connection fails exactly when the bit is set
(`bit_isSome`), the client reads the bit off the fail cell (`bit_dec`), and the window's record holds it (`bit_obs`).
The operator's ingress is whether the connection failed: it follows the fail pattern, so the network certifier
registers the window at both secrets (`bit_registers`) and both meet the good event (`bit_good`).

**The factor is met** (`MonitoringPeriodLeakFailsAtMost_factor_met`). Every premise of `MonitoringPeriodLeakFailsAtMost` holds there at
one connection of bound 1 and at most one failure, and the two secrets are counted: against `#Θ · windowsCount = 1`
the count fails, and the statement's factor there is 2 (`failsFactor_one`). So the count of `MonitoringPeriodLeakChecked`
does not hold at an operator that follows the fail pattern, and no smaller factor does at this period. It drops no
premise: it is a lemma about the factor, and no exclusion.

**The weights must sum to at most one** (`MonitoringPeriodLeakCharged_kraft_exclusion`). At the same period and the weights
`w_j = 1`, which sum to 2 over the failures of one connection (`not_kraftWeights_one`), every other premise of
`MonitoringPeriodLeakCharged` holds, each of the two secrets is charged at most `log2 1` (`bit_charge`), and both are counted
against `T = 1`. `MonitoringPeriodLeakCharged_nontrivial` is that count alone.

**The premises together** (`MonitoringPeriodLeakCharged_honest`). At the toy code over a monitoring period of no windows and no
connection, where `MonitoringPeriodLeakChecked_honest` has `MonitoringPeriodLeakChecked`'s premises, the weights `2^-(j+1)` sum to at most
one (`kraftWeights_halves`). The lemma shows that the premises hold together and nothing more: the toy code's
verifier rejects every run and the period has no window and no connection, so it shows nothing about a failure, a
window's check or a run that passes, and where a window is placed and a run is accepted every premise but the weights'
sum is still to be shown.

`posL` rejects every run, so no lemma at it is about a run that passes: each shows what the count does where the
premises hold. Its period has no checked input and one window, so none shows anything about a checked input or about
the windows before a window.
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling
open Claims.SamplingGeneric (Arg Compiled)
open Claims.Witness.SamplingGeneric (toyA notProg notStmt notPart posCm posStmt posX posTops posL)
open scoped NNRat

set_option linter.unusedVariables false

/-! ## The connection's failure at the bit -/

/-- The connection fails exactly when the bit is set: its end is late for bucket 0. -/
theorem bit_isSome (z : Bool) : (failCell oneSched (bitArr z) ()).isSome = z := by cases z <;> rfl

/-- The move of the operator that follows the fail pattern, at the bit: the bit. -/
theorem bit_hop (z : Bool) :
    (fun (_ : ℕ) (_ : List (Obs 1 1 1)) => [(failCell oneSched (bitArr z) ()).isSome]) 0 [] = [z] := by
  cases z <;> rfl

theorem failCount_unit_le (φ : Unit → Option ℕ) : failCount φ ≤ 1 := by
  unfold failCount
  exact (Finset.card_filter_le _ _).trans (by decide)

/-- The connection's fail cell is below the bound 1: its lane has one cell. -/
theorem bit_failsBelow (z : Bool) : FailsBelow (fun _ : Unit => 1) (failCell oneSched (bitArr z)) :=
  failsBelow_of_cells_le oneSched (bitArr z) (fun _ => 1) fun _ => le_rfl

/-- **The network certifier registers the window at the bit**, at the operator whose ingress is whether the connection
failed. -/
theorem bit_registers (z : Bool) (i : Fin 1) :
    CertifierRegisters id posCm notLay 1 (fun _ _ => [(failCell oneSched (bitArr z) ()).isSome]) (posStmt z)
      (posTops (posStmt z)) ((recordingCode bitFmt oneSched (bitArr z)).2 i) := by
  rw [bit_obs]
  exact pos_registers id z (fun _ _ => [(failCell oneSched (bitArr z) ()).isSome]) (bit_hop z)

/-- **The window is good at the bit's record**, at any registered values. -/
theorem bit_good (z : Bool) (ρ : ℕ → Bool) (i : Fin 1) :
    ExecutionPhaseGoodChecked id (posL Bool) 1 1 1 (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅) (fun _ _ => True ∧ True) 1
      (posStmt false) ρ posTops (posX z) ⟨[]⟩ ((recordingCode bitFmt oneSched (bitArr z)).2 i) := by
  rw [bit_obs]
  exact pos_good id z ρ

/-- The channel of the bit over the one registered parameter, as a finite set of parameters. -/
theorem bit_channel :
    MonitoringPeriodChannelReg bitFmt (fun (_ : Unit) s => s = oneSched) ((({()} : Finset Unit) : Set Unit))
      (Set.univ : Set Bool) (fun _ => oneSched) id (fun v => failedAt0 (v ()))
      fun p => (recordingCode bitFmt p.1 (bitArr p.2)).1 :=
  ⟨fun _ _ => ⟨rfl, (), by simp, rfl⟩, fun z _ => bit_dec z⟩

/-! ## The failures' factor is met -/

/-- One connection of bound 1 has one fail pattern at each number of failures it can have. -/
theorem failPatterns_unit (φ : Unit → Option ℕ) : failPatterns (fun _ : Unit => 1) (failCount φ) = 1 := by
  have h := failCount_unit_le φ
  generalize failCount φ = m at h
  rw [failPatterns_const, Fintype.card_unit, one_pow, mul_one]
  rcases (by omega : m = 0 ∨ m = 1) with rfl | rfl <;> rfl

/-- The factor of `MonitoringPeriodLeakFailsAtMost` at one connection of bound 1 and at most one failure is 2. -/
theorem failsFactor_one : ∑ i ∈ Finset.range (1 + 1), failPatterns (fun _ : Unit => 1) i = 2 :=
  (failPatterns_sum fun _ : Unit => 1).trans (by simp)

/-- **The failures' factor is met.** At `posL`, one window, the connection of `oneSched` under the host that runs
`recordingCode bitFmt`, and the operator whose ingress is whether the connection failed: the coins are a nonempty type
and `δ = 0` is nonnegative; `Compiled.KS`, the commitment's binding, the window's placement, its check's pinning,
`TrustedHost`, the read-back under the one registered parameter and the channel all hold; the network certifier
registers its observations at both secrets; the two secrets, each with at most one failure below the bound 1 and a
window that meets its good event, are counted against `#Θ · windowsCount = 1`, which fails; and the factor of
`MonitoringPeriodLeakFailsAtMost` there is 2. -/
theorem MonitoringPeriodLeakFailsAtMost_factor_met :
    Nonempty Bool ∧ (0 : ℝ) ≤ 0 ∧ (posL Bool).KS id ∧ (posL Bool).commit.Binds id ∧
      WindowsPlaced (posL Bool).partition 0 (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0) ∧
      CheckedUnique notProg notPart (⟨[.all]⟩ : Design) notLay 1 ∅ (fun _ => True ∧ True) ∧
      Verity.Assumptions.TrustedHost ()
        (fun (_ : Unit) (p : CertifierSchedule Unit × Bool) => recordingCode bitFmt p.1 (bitArr p.2))
        (fun p => recordingCode bitFmt p.1 (bitArr p.2)) ∧
      bitFmt.ReadsBy (fun (_ : Unit) => bitFmt.read oneSched) (fun _ s => s = oneSched) ∧
      MonitoringPeriodChannelReg bitFmt (fun (_ : Unit) s => s = oneSched) {()} (Set.univ : Set Bool) (fun _ => oneSched) id
        (fun v => failedAt0 (v ())) (fun p => (recordingCode bitFmt p.1 (bitArr p.2)).1) ∧
      (∀ (z : Bool) (i : Fin 1) (s : AuditStatement), (posL Bool).parse (posX z) = some s →
        CertifierRegisters id (posL Bool).commit notLay 1 (fun _ _ => [(failCell oneSched (bitArr z) ()).isSome]) s
          (posTops s) ((recordingCode bitFmt oneSched (bitArr z)).2 i)) ∧
      ¬ {z : Bool | z ∈ (Set.univ : Set Bool) ∧ failCount (failCell oneSched (bitArr z)) ≤ 1 ∧
            FailsBelow (fun _ : Unit => 1) (failCell oneSched (bitArr z)) ∧
            ∀ i : Fin 1, ExecutionPhaseGoodChecked id (posL Bool) 1 1 1 (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅)
              (fun _ _ => True ∧ True) 1 (posStmt false) (fun _ => false) posTops (posX z) ⟨[]⟩
              ((recordingCode bitFmt oneSched (bitArr z)).2 i)}.encard ≤
          ({()} : Set Unit).encard * ((windowsCount (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅ ∪ ∅) 1
            (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0) : ℕ) : ℕ∞) ∧
      ∑ i ∈ Finset.range (1 + 1), failPatterns (fun _ : Unit => 1) i = 2 := by
  refine ⟨⟨true⟩, le_rfl, posL_ks id, posCm_binds id, pos_placed, pos_checkedUnique, fun _ => rfl,
    bitFmt.readsBy_comp fun _ => oneSched, ⟨fun _ _ => ⟨rfl, (), Set.mem_singleton (), rfl⟩, fun z _ => bit_dec z⟩,
    fun z i s hs => ?_, fun hc => ?_, failsFactor_one⟩
  · obtain rfl : posStmt z = s := Option.some.inj ((pos_parse (Coin := Bool) z).symm.trans hs)
    exact bit_registers z i
  · have h2 := (Set.encard_pair (show true ≠ false by decide)).symm.le.trans
      ((Set.encard_le_encard (s := ({true, false} : Set Bool)) fun z _ =>
        Set.mem_ofPred.2 ⟨Set.mem_univ z, failCount_unit_le _, bit_failsBelow z, fun i => bit_good z _ i⟩).trans hc)
    exact absurd (h2.trans (le_of_eq pos_count)) two_not_le_one

/-! ## The weights must sum to at most one -/

/-- The weights `w_j = 1` on the failures of one connection sum to 2. -/
theorem not_kraftWeights_one : ¬ KraftWeights (fun _ => 1) (Fintype.card Unit) := by
  norm_num [KraftWeights, Fintype.card_unit, Finset.sum_range_succ]

/-- At the weights `w_j = 1` and `T = 1`, each secret's run is charged at most `log2 T`. -/
theorem bit_charge (z : Bool) :
    ((failPatterns (fun _ : Unit => 1) (failCount (failCell oneSched (bitArr z))) *
        (({()} : Finset Unit).card * windowsCount (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅ ∪ ∅) 1
          (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0)) : ℕ) : ℚ≥0) ≤
      (1 : ℚ≥0) * ((1 : ℕ) : ℚ≥0) := by
  rw [failPatterns_unit, pos_windowsCount]
  simp

/-- **`MonitoringPeriodLeakCharged`'s count fails at weights that sum to 2.** At `posL`, one window, the connection of `oneSched`
and the operator whose ingress is whether the connection failed, at the weights `w_j = 1` and `T = 1`: the two secrets,
each below the bound 1, charged at most `log2 1` (`bit_charge`) and with a window that meets its good event, are counted
against 1. -/
theorem MonitoringPeriodLeakCharged_nontrivial :
    ¬ {z : Bool | z ∈ (Set.univ : Set Bool) ∧ FailsBelow (fun _ : Unit => 1) (failCell oneSched (bitArr z)) ∧
          ((failPatterns (fun _ : Unit => 1) (failCount (failCell oneSched (bitArr z))) *
              (({()} : Finset Unit).card * windowsCount (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅ ∪ ∅) 1
                (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0)) : ℕ) : ℚ≥0) ≤
            (1 : ℚ≥0) * ((1 : ℕ) : ℚ≥0) ∧
          ∀ i : Fin 1, ExecutionPhaseGoodChecked id (posL Bool) 1 1 1 (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅)
            (fun _ _ => True ∧ True) 1 (posStmt false) (fun _ => false) posTops (posX z) ⟨[]⟩
            ((recordingCode bitFmt oneSched (bitArr z)).2 i)}.encard ≤ ((1 : ℕ) : ℕ∞) := fun hc => by
  have h2 := (Set.encard_pair (show true ≠ false by decide)).symm.le.trans
    ((Set.encard_le_encard (s := ({true, false} : Set Bool)) fun z _ =>
      Set.mem_ofPred.2 ⟨Set.mem_univ z, bit_failsBelow z, bit_charge z, fun i => bit_good z _ i⟩).trans hc)
  exact absurd (h2.trans (le_of_eq Nat.cast_one)) two_not_le_one

/-- **Without `KraftWeights`, `MonitoringPeriodLeakCharged`'s count fails.** At `posL`, one window, the connection of `oneSched`
under the host that runs `recordingCode bitFmt`, and the operator whose ingress is whether the connection failed: the
coins are a nonempty type and `δ = 0` is nonnegative; `Compiled.KS`, the commitment's binding, the window's placement,
its check's pinning, `TrustedHost`, the read-back under the one registered parameter and the channel all hold; the
network certifier registers its observations at both secrets; the weights `w_j = 1` sum to 2 over the failures of one
connection; and the two secrets, each charged at most `log2 1`, are counted against `T = 1`. -/
theorem MonitoringPeriodLeakCharged_kraft_exclusion :
    Nonempty Bool ∧ (0 : ℝ) ≤ 0 ∧ (posL Bool).KS id ∧ (posL Bool).commit.Binds id ∧
      WindowsPlaced (posL Bool).partition 0 (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0) ∧
      CheckedUnique notProg notPart (⟨[.all]⟩ : Design) notLay 1 ∅ (fun _ => True ∧ True) ∧
      Verity.Assumptions.TrustedHost ()
        (fun (_ : Unit) (p : CertifierSchedule Unit × Bool) => recordingCode bitFmt p.1 (bitArr p.2))
        (fun p => recordingCode bitFmt p.1 (bitArr p.2)) ∧
      bitFmt.ReadsBy (fun (_ : Unit) => bitFmt.read oneSched) (fun _ s => s = oneSched) ∧
      MonitoringPeriodChannelReg bitFmt (fun (_ : Unit) s => s = oneSched) ((({()} : Finset Unit) : Set Unit))
        (Set.univ : Set Bool) (fun _ => oneSched) id (fun v => failedAt0 (v ()))
        (fun p => (recordingCode bitFmt p.1 (bitArr p.2)).1) ∧
      (∀ (z : Bool) (i : Fin 1) (s : AuditStatement), (posL Bool).parse (posX z) = some s →
        CertifierRegisters id (posL Bool).commit notLay 1 (fun _ _ => [(failCell oneSched (bitArr z) ()).isSome]) s
          (posTops s) ((recordingCode bitFmt oneSched (bitArr z)).2 i)) ∧
      ¬ KraftWeights (fun _ => 1) (Fintype.card Unit) ∧
      ¬ {z : Bool | z ∈ (Set.univ : Set Bool) ∧ FailsBelow (fun _ : Unit => 1) (failCell oneSched (bitArr z)) ∧
            ((failPatterns (fun _ : Unit => 1) (failCount (failCell oneSched (bitArr z))) *
                (({()} : Finset Unit).card * windowsCount (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅ ∪ ∅) 1
                  (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0)) : ℕ) : ℚ≥0) ≤
              (1 : ℚ≥0) * ((1 : ℕ) : ℚ≥0) ∧
            ∀ i : Fin 1, ExecutionPhaseGoodChecked id (posL Bool) 1 1 1 (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅)
              (fun _ _ => True ∧ True) 1 (posStmt false) (fun _ => false) posTops (posX z) ⟨[]⟩
              ((recordingCode bitFmt oneSched (bitArr z)).2 i)}.encard ≤ ((1 : ℕ) : ℕ∞) := by
  refine ⟨⟨true⟩, le_rfl, posL_ks id, posCm_binds id, pos_placed, pos_checkedUnique, fun _ => rfl,
    bitFmt.readsBy_comp fun _ => oneSched, bit_channel, fun z i s hs => ?_, not_kraftWeights_one,
    MonitoringPeriodLeakCharged_nontrivial⟩
  obtain rfl : posStmt z = s := Option.some.inj ((pos_parse (Coin := Bool) z).symm.trans hs)
  exact bit_registers z i

/-! ## The premises together -/

theorem sum_halves (n : ℕ) : ∑ j ∈ Finset.range n, (2⁻¹ : ℚ≥0) ^ (j + 1) + 2⁻¹ ^ n = 1 := by
  induction n with
  | zero => simp
  | succ n ih =>
    have h2 : (2⁻¹ : ℚ≥0) ^ (n + 1) + 2⁻¹ ^ (n + 1) = 2⁻¹ ^ n := by
      rw [pow_succ, ← mul_add]
      norm_num
    rw [Finset.sum_range_succ, add_assoc, h2, ih]

/-- The weights `2^-(j+1)` on the number of failures sum to at most one, at every number of connections. -/
theorem kraftWeights_halves (N : ℕ) : KraftWeights (fun j => (2⁻¹ : ℚ≥0) ^ (j + 1)) N :=
  le_of_le_of_eq le_self_add (sum_halves (N + 1))

set_option synthInstance.maxSize 1024 in
/-- **`MonitoringPeriodLeakCharged`'s premises at the toy code**, over a monitoring period of no windows and no connection:
`MonitoringPeriodLeakChecked`'s premises there (`MonitoringPeriodLeakChecked_honest`), the channel over the one registered parameter as a
finite set, the network certifier's registration at every window (`CertifierRegisters`, which the charged form takes
as a premise), of which the period has none, and the weights `2^-(j+1)` on the number of failures, which sum to at
most one (`kraftWeights_halves`). The premises hold together there, and the lemma shows nothing more. The toy code
parses no statement and its verifier rejects every run, so no run passes there, and the period has no window and no
connection: it covers no run with a failure, and holds no window's placement, check or registration to anything. Where
a window is placed and a run is accepted, the window's placement (`WindowsPlaced`), its check's pinning
(`CheckedUnique`), its registration (`CertifierRegisters`), the read-back of a format with a connection
(`RecordFormat.ReadsBy`) and its channel (`MonitoringPeriodChannelReg`) are not shown here, with or without a failure; the
weights' sum alone is shown at every number of connections. -/
theorem MonitoringPeriodLeakCharged_honest :
    (SamplingDeploy.compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) SamplingDeploy.toyCode
        (fun _ _ _ => [])).KS FlockDeploy.sha512 ∧
      (SamplingDeploy.compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) SamplingDeploy.toyCode
        (fun _ _ _ => [])).commit.Binds FlockDeploy.sha512 ∧
      WindowsPlaced
        (SamplingDeploy.compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) SamplingDeploy.toyCode
          (fun _ _ _ => [])).partition 0 (Fin.elim0 : Fin 0 → Claims.Sampling.AuditStatement) Fin.elim0 Fin.elim0 ∧
      (∀ i : Fin 0, CheckedUnique ((Fin.elim0 : Fin 0 → Claims.Sampling.AuditStatement) i).prog
        ((Fin.elim0 : Fin 0 → Claims.Sampling.Partition) i)
        ((Fin.elim0 : Fin 0 → Claims.Sampling.AuditStatement) i).design trivLay 1 ∅ fun _ => True ∧ True) ∧
      noFormat.ReadsBy (fun (_ : Unit) _ c => c.elim) (fun _ _ => True) ∧
      MonitoringPeriodChannelReg noFormat (fun (_ : Unit) _ => True) (((Finset.univ : Finset Unit) : Set Unit))
        (Set.univ : Set Unit) (fun _ => ⟨0, 0, Empty.elim, Empty.elim⟩) (fun _ => ()) (fun _ => ())
        (fun p => (recordingCode noFormat p.1 fun c => c.elim).1) ∧
      (∀ (i : Fin 0) (s : Claims.Sampling.AuditStatement),
        CertifierRegisters FlockDeploy.sha512
          (SamplingDeploy.compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) SamplingDeploy.toyCode
            (fun _ _ _ => [])).commit trivLay 1 (fun _ _ => []) s []
          ((recordingCode noFormat ⟨0, 0, Empty.elim, Empty.elim⟩ fun c => c.elim).2 i)) ∧
      KraftWeights (fun j => (2⁻¹ : ℚ≥0) ^ (j + 1)) (Fintype.card Empty) :=
  ⟨MonitoringPeriodLeakChecked_honest.1, MonitoringPeriodLeakChecked_honest.2.1, MonitoringPeriodLeakChecked_honest.2.2.1,
    MonitoringPeriodLeakChecked_honest.2.2.2.1, MonitoringPeriodLeakChecked_honest.2.2.2.2.1,
    ⟨fun _ _ => ⟨trivial, (), Finset.mem_coe.2 (Finset.mem_univ _), trivial⟩, fun _ _ => rfl⟩, fun i _ => i.elim0,
    kraftWeights_halves _⟩

end NetTiming.Witness
