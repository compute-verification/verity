module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.Claims.Bridge.CommittedNil
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
# `KSLinkOn` at Hello-committed coins

`KSLinkOn` reads the composite's run at coins flipped after each message. Deployed, the coins are committed at Hello in
an hm96 coin tree and each opened with its round's answer, except where that answer is `[]`. `KSLinkOnC` is
`KSLinkOn`'s bound at that run, as `Claims.Recursion.KSLin2OnC` is `KSLin2On`'s: every hm96 tree whose leaves hide at
`δ₁ ≠ ∞` (`Hm96Hiding`) has a simulator `S` such that, for every prover that reads the root and the openings, the
committed run accepts at most as often as the composite's extractor `E*`, run on the prover with `S`'s root and openings
(`Bridge.simProverM`), recovers a witness of the composite's relation, plus `KSLinkOn`'s bound at that prover
(`linkRhs`) and `2 · k(x) · δ₁`. Like `KSLinkOn`, it is at the deployed decoders (`compDep`) and quantifies over none.
`ksLinkOnC_of_ksLinkOn` derives it from `KSLinkOn` (`Bridge.Pr_acc_runCM_le`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open scoped ENNReal

section Scoped

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- **`KSLinkOn`'s bound at the prover `P`**: `KSLin2`'s terms at the deployed decoders, with the gated link's binding
term (`linkPlan` at `verModel`), at `P` and the counts `M`, `R`. -/
noncomputable def linkRhs [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (M R : ℕ) : ℝ :=
  Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x +
    Real.sqrt (Claims.Recursion.γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x *
      PrB H (D.record Bd.IVStar).V P x) +
    (4 * (Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k (Claims.Recursion.xI (D.record Bd.IVStar) x) + 1) *
      PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode P x +
    4 * Cp.width (Claims.Recursion.xI (D.record Bd.IVStar) x) *
      (PrFCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode Cp P x M +
        1 / ((M : ℝ) + 1)) +
    4 * (PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode P x R +
      (((Inner.ofFlock Bd.deploy Bd.deploy.decode).k (Claims.Recursion.xI (D.record Bd.IVStar) x) : ℝ) + 1) /
        ((R : ℝ) + 1)) +
    βdev2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x *
      ((linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H
          ((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
          ((compDep D Bd DO).xI x) +
        1 / ((M : ℝ) + 1))

/-- **`KSLinkOn` at Hello-committed coins**, at a leaf hiding bound `δ₁ ≠ ∞`: at every statement `G` admits and every
hm96 coin tree whose leaves hide at `δ₁`, a simulator `S` of the tree such that, for every prover that reads its root
and openings (`Bridge.runCM`), the committed run's acceptance is at most the probability that the composite's extractor,
run on the prover with `S`'s root and openings (`Bridge.simProverM`), recovers a witness of the composite's relation,
plus `KSLinkOn`'s bound at that prover (`linkRhs`) and `2 · k(x) · δ₁`. -/
def KSLinkOnC [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (δ₁ : ℝ≥0∞) : Prop :=
  δ₁ ≠ ⊤ → ∀ (x : Msg), G x →
    ∀ (B Lf Cm T₀ : Type) [AddGroup B] [Fintype B] [Fintype Lf] [Nonempty Lf] [Fintype T₀] [Nonempty T₀]
      (Tr : Bridge.Hm96Tree Coin B Lf Cm T₀ ((compDep D Bd DO).V.k x)), Hm96Hiding Tr.My Tr.cs δ₁ →
    ∃ S : Bridge.OnlineSim, ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Bridge.CProver Ω) (M R : ℕ),
      Pr (fun p : Ω × (Fin ((compDep D Bd DO).V.k x) → Coin) × (Fin ((compDep D Bd DO).V.k x) → Lf) × T₀ =>
          (Bridge.runCM H P (compDep D Bd DO).V x Tr.tree p.1 p.2.1 p.2.2).out = .acc) ≤
        Pr (fun s : (Ω × S.Tape) × (Fin ((compDep D Bd DO).V.k x) → Coin) ×
              ((compDep D Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)).C x =>
            ((compDep D Bd DO).R (Proto.ofFlock Bd Bd.deploy.decode Cp)).ext x
              (((compDep D Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)).eval H
                (Bridge.simProverM S (compDep D Bd DO).V.k P) x s.1 s.2.1 s.2.2).1) +
          linkRhs H D Bd DO Cp x (Bridge.simProverM S (compDep D Bd DO).V.k P) M R +
          2 * (compDep D Bd DO).V.k x * δ₁.toReal

/-- **`KSLinkOn` gives `KSLinkOnC`** wherever the composite's verifier answers each round with its coin, through an
injective `ans`, or with nothing: `Bridge.Pr_acc_runCM_le` at the composite's extractor, with `PrBad` at the simulating
prover bounded by `KSLinkOn` there. -/
theorem ksLinkOnC_of_ksLinkOn [Zero Col] [Zero Col'] {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Claims.Flock.Deploy Coin Col' T'} {Cp : Caps Coin} {G : Msg → Prop} (h : KSLinkOn H D Bd DO Cp G)
    (ans : Coin → Msg) (hans : Function.Injective ans)
    (hV : ∀ x, Bridge.AnswersCoinOrNil H (compDep D Bd DO).V x ans) (δ₁ : ℝ≥0∞) :
    KSLinkOnC H D Bd DO Cp G δ₁ := by
  intro hδ x hx B Lf Cm T₀ _ _ _ _ _ _ Tr hH
  obtain ⟨S, hS⟩ := Bridge.Pr_acc_runCM_le H (compDep D Bd DO).V
    ((compDep D Bd DO).R (Proto.ofFlock Bd Bd.deploy.decode Cp))
    ((compDep D Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x Tr hH hδ ans hans (hV x)
  refine ⟨S, ?_⟩
  intro Ω _ _ P M R'
  have hA := hS Ω P
  have hK : PrBad H (compDep D Bd DO).V ((compDep D Bd DO).R (Proto.ofFlock Bd Bd.deploy.decode Cp))
      ((compDep D Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x
      (Bridge.simProverM S (compDep D Bd DO).V.k P) ≤
      linkRhs H D Bd DO Cp x (Bridge.simProverM S (compDep D Bd DO).V.k P) M R' :=
    h x hx (Ω × S.Tape) (Bridge.simProverM S (compDep D Bd DO).V.k P) M R'
  linarith

end Scoped

end Claims.RecGeneric
