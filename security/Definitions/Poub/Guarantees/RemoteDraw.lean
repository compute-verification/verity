module

public import Definitions.Poub.Assumptions.RemoteDraw
public import Definitions.Poub.Assumptions.RemoteHeld
public import Definitions.Pous.Model.StreamedWipe
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

/-! The openings' pooled bound, the memory line's `LiveCoins` derived from the deployed opening sample, with `split_t`
not bounded. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.Remote

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sample
  ResourceAccounting.Remote.Sample.Assumptions

/-- **The openings' pooled bound.** On a session whose windows' sizes and correct receipts are fixed before their
openings coins (`Live`), whose accepted outcomes found every opened receipt correct or moved off the row its root holds
(`Refused`, at the receipts `R`) and count the trace's wrong receipts among the windows' (`Counted`), with every
window's sample on the first blocks `prf t` its coin gives and `m < 18446744073709551616` (2^64) receipts a window at
most, the probability that the session is accepted with more wrong receipts than the record's pooled bound at `δ` is at
most `δ + Σ_t s_t / 2^256`, plus the windows' distinguishers' advantages on `prf`, plus the receipts' binding term
(`R.binding`: some opened receipt moved off the row its root holds). -/
def PooledSample : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (D : Remote.Sample.Run n m k κ X)
    (T : Trace (SampleCoins n κ X)) (R : Receipts n κ X)
    (δ : ℝ), 0 ≤ δ → m < 18446744073709551616 → Live D → Refused D prf rest T R → Counted D T →
    T.Holds (δ + ((∑ t, k t : ℕ) : ℝ) / 2 ^ 256 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δ t)).toReal +
        (R.binding D prf rest).toReal) fun ω =>
      (T.incorrect ω : ℝ) ≤ D.pooled δ ω

/-- **Live coins, derived.** On a session meeting `PooledSample`'s premises, `m < 18446744073709551616` (2^64) among
them, `LiveCoins W` at `PooledSample`'s error holds for every `W` at least the record's pooled bound at `δ` on every
accepted outcome. -/
def LiveCoinsSampled : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (D : Remote.Sample.Run n m k κ X)
    (T : Trace (SampleCoins n κ X)) (R : Receipts n κ X)
    (W δ : ℝ), 0 ≤ δ → m < 18446744073709551616 → Live D → Refused D prf rest T R → Counted D T →
    (∀ ω, T.accepts ω → D.pooled δ ω ≤ W) →
    LiveCoins T W (δ + ((∑ t, k t : ℕ) : ℝ) / 2 ^ 256 +
      (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δ t)).toReal +
      (R.binding D prf rest).toReal)

/-- **The snapshot held at the record's pooled bound, under Identity.** `Held` on a trace over a session's coins, with
`LiveCoins` replaced by the sample's premises (`Live`, `Refused`, `Counted`) and `W` by the record's pooled bound at
`δc`, on windows of `m < 18446744073709551616` (2^64) receipts. -/
def HeldSampled : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (D : Remote.Sample.Run n m k κ X)
    (T : Trace (SampleCoins n κ X)) (R : Receipts n κ X) (S s₀ e : ℕ) (s E β Br Bd δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ δc → m < 18446744073709551616 →
    Identity T → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Causal T s₀ e →
    SketchBound T δs → Live D → Refused D prf rest T R → Counted D T →
    T.Holds (δs + (δc + ((∑ t, k t : ℕ) : ℝ) / 2 ^ 256 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δc t)).toReal +
        (R.binding D prf rest).toReal)) fun ω =>
      (T.planned ω : ℝ) - D.pooled δc ω - ((s / E + β) * ((e : ℝ) - s₀) + (Br + Bd)) - T.passes * S ≤ T.held ω

/-- **Held over time at the record's pooled bound, under Identity.** `HeldOverTime` with `LiveCoins` replaced by the
sample's premises and `W` by the record's pooled bound at `δc`, on windows of `m < 18446744073709551616` (2^64)
receipts. -/
def HeldOverTimeSampled : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (D : Remote.Sample.Run n m k κ X)
    (T : Trace (SampleCoins n κ X)) (R : Receipts n κ X)
    (K : Pace (SampleCoins n κ X)) (S : ℕ) (s E β Br Bd δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ δc → m < 18446744073709551616 →
    Identity T → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Paced T K → Covers T →
    SketchBound T δs → Live D → Refused D prf rest T R → Counted D T →
    T.Holds (δs + (δc + ((∑ t, k t : ℕ) : ℝ) / 2 ^ 256 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δc t)).toReal +
        (R.binding D prf rest).toReal)) fun ω =>
      ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω * ((s / E + β) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω) + (Br + Bd))
          - K.fill ω T.blocks (D.pooled δc ω)
        ≤ T.heldOver ω (K.start ω) (K.stop ω)

/-- **Held over time with the links charged, at the record's pooled bound** (no Identity). `HeldOverTimeLinks` with
`LiveCoins` replaced by the sample's premises and `W` by the record's pooled bound at `δc`, on windows of
`m < 18446744073709551616` (2^64) receipts. -/
def HeldOverTimeLinksSampled : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (D : Remote.Sample.Run n m k κ X)
    (T : Trace (SampleCoins n κ X)) (R : Receipts n κ X)
    (K : Pace (SampleCoins n κ X)) (S : ℕ) (s E β βl Br Bd Bl δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ βl → 0 ≤ δc → m < 18446744073709551616 →
    Links T βl Bl → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Paced T K → Covers T →
    SketchBound T δs → Live D → Refused D prf rest T R → Counted D T →
    T.Holds (δs + (δc + ((∑ t, k t : ℕ) : ℝ) / 2 ^ 256 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δc t)).toReal +
        (R.binding D prf rest).toReal)) fun ω =>
      ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω *
              ((s / E + β + βl) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω) + (Br + Bd + Bl))
          - K.fill ω T.blocks (D.pooled δc ω)
        ≤ T.heldOver ω (K.start ω) (K.stop ω)

end ResourceAccounting.Guarantees.Remote
