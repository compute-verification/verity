module

public import Proofs.Poub.RateAudit.Honest
public import Definitions.Poub.BandwidthAudit
public import Mathlib.Logic.Equiv.PartialEquiv
public import Std.Tactic.BVDecide.Normalize.Prop
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Topology.Sheaves.Init
import Proofs.Poub.RateAudit.Interactive
import Proofs.Pouw.Prob

/-! The joint guarantee's hypotheses and the bandwidth and rate audits' games hold on honest runs, so none of them is
vacuous. -/

@[expose] public section

namespace ResourceAccounting.Run

open Pouw

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (r : Run d α ΩW ΩQ ΩS)

theorem computeSound_ideal [Fintype ΩW] {p : Price} {W : ℝ} (h : ∀ ω, r.accepts .compute ω → W ≤ r.work p ω) :
    r.ComputeSound p W 0 := fun _ _ _ hT =>
  (pr_eq_zero fun x hx => absurd ((h _ hx.1).trans (hT x)) (not_le.2 hx.2)).le

theorem bandwidthSound_ideal [Fintype ΩQ] {Q : ℝ} (h : ∀ ω, r.accepts .bandwidth ω → Q ≤ r.read ω) :
    r.BandwidthSound Q 0 := fun _ _ _ _ hB =>
  (pr_eq_zero fun q hq => absurd ((h _ hq.1).trans (hB q)) (not_le.2 hq.2)).le

theorem capacitySound_ideal [Fintype ΩS] {S : ℝ} (h : ∀ ω, r.accepts .capacity ω → S * (r.win .capacity).len ≤ r.held ω) :
    r.CapacitySound S 0 := fun _ _ _ _ hB =>
  (pr_eq_zero fun s hs => absurd ((h _ hs.1).trans (hB s)) (not_le.2 hs.2)).le

end ResourceAccounting.Run

namespace ResourceAccounting.Sanity

open Bandwidth.Completeness (honest honest_accepts)

def honestDevice (N b : ℕ) : Device Empty := ⟨0, 0, fun _ => N, N * b, Empty.elim⟩

def honestRun (N b h s : ℕ) : Run (honestDevice N b) Unit Unit (BandwidthCoins N b h s) Unit where
  acct _ := ()
  win _ := ⟨0, 1⟩
  schedule _ _ := ⟨fun _ _ => 0, fun _ => N * b, fun _ i => i.elim, fun _ => 0, fun _ => 0⟩
  fits _ _ := ⟨fun _ => by simp [honestDevice], by simp [honestDevice], fun i => i.elim, by simp [honestDevice],
    by simp [honestDevice]⟩
  accepts _ ω := Bandwidth.Accepts (honest N b h) ω.2.1.1 ω.2.1.2.1 ω.2.1.2.2

theorem honestRun_realizesBandwidth (N b h s : ℕ) : Nonempty (RealizesBandwidth 0 (honestRun N b h s)) :=
  ⟨{ dev := fun _ => { honest N b h with
        bounded := fun st w => by
          obtain ⟨h1, h2, h3⟩ := (honest N b h).bounded st w
          exact ⟨by simpa using h1, h2, by simpa [honestRun, honestDevice] using h3⟩ }
     accepts_iff := fun _ _ _ => Iff.rfl
     ingress_le := fun _ y _ => by
      refine le_of_eq ((honest_accepts (s := s) y.1 y.2.1 y.2.2).2.trans ?_)
      simp [Run.read, Schedule.read, Window.ticks, honestRun] }⟩

def rateDevice (N b ρ : ℕ) : Device Unit := ⟨0, 0, fun o => if o = .oracle then N else 0, N * b, fun _ => ρ⟩

def tagPrice (cTag : ℕ) : Price := fun o => if o = .oracle then cTag else 0

def rateRun (N b h ρ s : ℕ) : Run (rateDevice N b ρ) Bool Unit (RateCoins N b h ρ s) Unit where
  acct r := match r with
    | .compute => false
    | _ => true
  win _ := ⟨0, 1⟩
  schedule _ _ := ⟨fun a o => if a ∧ o = .oracle then N else 0, fun a => if a then N * b else 0,
    fun a _ => if a then ρ else 0, fun _ => 0, fun _ => 0⟩
  fits _ _ := ⟨fun o => by cases o <;> simp [rateDevice], by simp [rateDevice], fun _ => by simp [rateDevice],
    by simp [rateDevice], by simp [rateDevice]⟩
  accepts _ ω := (Rate.Completeness.honest N b h ρ).Accepts ω.2.1.1 ω.2.1.2.1.1 ω.2.1.2.1.2 ω.2.1.2.2

theorem rateRun_realizesRate (N b h ρ s cTag cEnc : ℕ) :
    Nonempty (RealizesRate (tagPrice cTag) cTag cEnc (rateRun N b h ρ s)) :=
  ⟨{ dev := fun _ => { Rate.Completeness.honest N b h ρ with
        bounded := fun st w => by
          obtain ⟨h1, h2⟩ := (Rate.Completeness.honest N b h ρ).bounded st w
          exact ⟨by simpa [rateRun, rateDevice] using h1, by simpa [rateRun, rateDevice] using h2⟩ }
     accepts_iff := fun _ _ _ => Iff.rfl
     ingress_le := fun _ y _ => by
      refine le_of_eq ((Rate.Completeness.honest_accepts cTag cEnc y.1 y.2.1.1 y.2.1.2 y.2.2).2.2.1.trans ?_)
      simp [Run.read, Schedule.read, Window.ticks, rateRun]
     compute_le := fun _ y _ => by
      refine le_of_eq ((Rate.Completeness.honest_accepts cTag cEnc y.1 y.2.1.1 y.2.1.2 y.2.2).2.2.2.trans ?_)
      have he : ({true, false} : Finset Bool).erase false = {true} := by decide
      simp [Run.othersWork, Schedule.work, Schedule.ops, Window.ticks, rateRun, tagPrice, he, mul_comm] }⟩

theorem rateRun_realizesRateInteractive (N b h ρ s cTag cEnc : ℕ) :
    ∃ real : RealizesRateInteractive (tagPrice cTag) cTag cEnc (rateRun N b h ρ s),
      ∀ (x : Unit) (s' : Unit) E R O O', (real.dev (x, s')).side E R O = (real.dev (x, s')).side E R O' :=
  let ⟨real⟩ := rateRun_realizesRate N b h ρ s cTag cEnc
  ⟨real.toInteractive, real.toInteractive_side⟩

end ResourceAccounting.Sanity
