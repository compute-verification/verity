module

public import Definitions.NetworkCertifier.Tap
public import Definitions.NetworkCertifier.TapRows
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
# The named conditions of a monitoring period's statements

Definitions only. A statement over a monitoring period of sampled audits takes these conditions, each named once:

* each window's statement has its partition, and each of its possible incorrect sets commits a bounded number of bits
  (`WindowsPlaced`);
* the network certifier's record reads back under a registered parameter (`RecordFormat.ReadsBy`): the client's view
  is a function of the record and of a parameter fixed at registration, in place of the whole schedule, so a count
  over the records pays for the parameters and not for the schedules;
* a covert sender's channel over registered parameters (`PeriodChannelReg`): each secret's schedule is one the record
  format holds and a registered parameter allows, and a decoder recovers the secret from the client's view;
* the windows' count (`windowsCount`): what the windows' audits leave free, the product over the windows of each one's
  free inputs outside the pinned ones, its possible incorrect sets, their committed bits and the timing.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset

/-- **The windows' registrations**: each window's statement `sw i` has the partition `Pw i` under `partition`, and each
of its possible incorrect sets at `δ` commits at most `β i` bits. -/
def WindowsPlaced {n : ℕ} (partition : AuditStatement → Option Partition) (δ : ℝ) (sw : Fin n → AuditStatement)
    (Pw : Fin n → Partition) (β : Fin n → ℕ) : Prop :=
  (∀ i, partition (sw i) = some (Pw i)) ∧
    ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i

/-- **The record reads back under a registered parameter.** `rd θ` reads the client's cells from the record given the
parameter `θ` alone, and `allows θ s` says the schedule `s` is one of `θ`'s: on every schedule a parameter allows and
the format holds, `rd θ` of the record of the cells the network certifier's code sends is those cells. The format's own
`read` takes the whole schedule (`RecordFormat.read_record`): this at each schedule as its own parameter. -/
def RecordFormat.ReadsBy {κ Cell Ob Par : Type} (f : RecordFormat κ Cell Ob)
    (rd : Par → Ob → κ → List (ℕ × CellView Cell)) (allows : Par → CertifierSchedule κ → Prop) : Prop :=
  ∀ θ s, allows θ s → f.fits s → ∀ a, rd θ (f.record s a (certifierCode s a)) = certifierCode s a

/-- **A covert sender's channel over registered parameters**: the schedule the network certifier runs for each secret
`z` of `Zs` (`schedOf z`) is one the record format `f` holds and some registered parameter of `Θ` allows, and the
decoder `dec` recovers `z` from the cells the client receives on that schedule and the server behavior `srvOf z`
(`cells`). These are the count's own terms: a hypothesis on the sender's strategy (its secrets, schedules, server
behaviors and decoder) and on the registered parameters, nothing about the verifier's or the prover's code. -/
def PeriodChannelReg {Conn Cell Srv Ob Z Par : Type} (f : RecordFormat Conn Cell Ob)
    (allows : Par → CertifierSchedule Conn → Prop) (Θ : Set Par) (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn)
    (srvOf : Z → Srv) (dec : (Conn → List (ℕ × CellView Cell)) → Z)
    (cells : CertifierSchedule Conn × Srv → Conn → List (ℕ × CellView Cell)) : Prop :=
  (∀ z ∈ Zs, f.fits (schedOf z) ∧ ∃ θ ∈ Θ, allows θ (schedOf z)) ∧ ∀ z ∈ Zs, dec (cells (schedOf z, srvOf z)) = z

/-- **The windows' count**: the product over the windows of what each one's audit leaves free with the inputs `pin`
pinned, `2^#(freeInputs_i \ pin_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`: the free inputs outside the pinned ones, the
possible incorrect sets, their committed bits and the timing. -/
noncomputable def windowsCount {L S n : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ)
    (pin : AuditStatement → Finset ℕ) (Dj : ℕ) (sw : Fin n → AuditStatement) (Pw : Fin n → Partition)
    (β : Fin n → ℕ) : ℕ :=
  ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ pin (sw i)).card *
    (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i))

end NetTiming
