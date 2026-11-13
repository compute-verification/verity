module

public import Definitions.Accounting.Guarantees.JointBridge.CommOfPeriodLeakAccepted
public import Definitions.NetworkCertifier.Guarantees.Period
public import Definitions.Accounting.NetWitness
public import Proofs.NetworkCertifier.Period
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
import Proofs.Accounting.PouwWork
import Proofs.Pouw.Prob
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# The network certifier's per-period guarantee from its period of sampled audits (proof)

`Accounting.SecurityProofs.Bridge.CommOfPeriodLeakAccepted`: `comm_of_periodLeak` at network accounting's proof of
`PeriodLeakAccepted` (`NetTiming.periodLeak`). On a run that realizes the network certifier's period of `n` sampled
audits (`Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted`), the network audit's `PeriodRun.Sound` holds at
`n · (δ + ε_ks + ε_bind + ε_tap)`. The network certifier accepting with the claim failing means some window fails
`WindowGood` (`comm_iff`); the union bound over the windows sums each window's failure, which is at most the sampled
verifier's `PrBad` at `RLeakCertifier` (`window_le`), which `PeriodLeakAccepted`'s per-window conjunct bounds by
`κ + √(γ·Pr[B]) + Pr[B_ext] + Pr[B_tap]`, at most `δ + ε_ks + ε_bind + ε_tap` at each window's prover.

The per-window conjunct of `PeriodLeakAccepted` reads neither the period's windows nor the network certifier device,
so it is read off at the period of no windows (`ksLin_of_periodLeak`). Its count of the secrets is what the claim at
`K` stands for, and the error doesn't use it. The network certifier's trust (`trusted-host`) is the realization's
`trusted`, which this proof doesn't read: `window_le` is the per-window bound its records give, and deriving it from
`PeriodLeakAccepted` at the deployed period and `trusted` is the follow-up.
-/

@[expose] public section

namespace Accounting.SecurityProofs.Bridge

open Claims Claims.Sampling NetTiming Accounting.JointBridge Finset

theorem pr_exists_le {β : Type} [Fintype β] {n : ℕ} (P : Fin n → β → Prop) :
    Pouw.pr (fun x => ∃ i, P i x) ≤ ∑ i, Pouw.pr (P i) := by
  classical
  unfold Pouw.pr Claims.Pr Claims.prUniform
  rw [← Finset.sum_div]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have hsub : (univ.filter fun x => ∃ i, P i x) ⊆ univ.biUnion (fun i => univ.filter (P i)) := fun x hx => by
    simp only [mem_filter, mem_biUnion, mem_univ, true_and] at hx ⊢
    exact hx
  have key : ((univ.filter fun x => ∃ i, P i x).card : ℝ) ≤ ∑ i, ((univ.filter (P i)).card : ℝ) := by
    exact_mod_cast (card_le_card hsub).trans card_biUnion_le
  convert key using 3
  exact Finset.filter_congr_decidable _ _ _

/-- `PeriodLeakAccepted`'s per-window conjunct, read off at the period of no windows. -/
theorem ksLin_of_periodLeak (hsrc : Guarantees.PeriodLeakAccepted) {Coin : Type} [Fintype Coin]
    [Nonempty Coin] [Inhabited Coin] {A : Claims.SamplingGeneric.Arg Coin} (H : Msg → Msg)
    (D : Claims.SamplingGeneric.Compiled Coin A) (hks : D.KS H) (L Kₒ S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg : AuditStatement → Finset ℕ) (Dj : ℕ) (tops : AuditStatement → List (Opening Bool)) (hb : D.commit.Binds H) :
    ∀ m, ScopeAcc D lay k m → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      PrBad H D.V
          (RLeakCertifier H D.commit D.parse D.partition L Kₒ S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
          D.E m P ≤
        D.κ δ m + Real.sqrt (D.γ m * PrB H D.V P m) +
          (Bext H D.V D.E Ω P m).Pr H + (Btap H D.V D.E D.commit D.parse tops Ω P m).Pr H := by
  let f : RecordFormat Empty Unit (Fin 0 → List (Obs L Kₒ S)) :=
    ⟨fun _ _ _ => Fin.elim0, fun _ _ c => c.elim, fun _ => True, fun _ _ _ => funext fun c => c.elim⟩
  exact (hsrc (n := 0) (Conn := Empty) (Cell := Unit) (Srv := Unit) (Dev := Unit) (Z := Unit) H D hks L Kₒ S lay k δ
    hδ reg Dj Fin.elim0 Fin.elim0 (fun i => i.elim0) Fin.elim0 Fin.elim0 Fin.elim0 (fun i => i.elim0)
    (fun _ c => c.elim) f () (fun _ p => recordingCode f p.1 (fun c => c.elim)) (fun _ => rfl) (fun c => c.elim) ∅
    (fun _ h => h.elim) (fun _ h => h.elim) ∅ (fun _ => ⟨0, 0, Empty.elim, Empty.elim⟩) (fun _ => ())
    (fun _ h => h.elim) (fun _ => ()) (fun _ h => h.elim) (fun _ => Fin.elim0) (fun _ => Fin.elim0) tops hb).1

/-- `Guarantees.Bridge.CommOfPeriodLeakAccepted` from `PeriodLeakAccepted`. -/
theorem comm_of_periodLeak (hsrc : Guarantees.PeriodLeakAccepted) {ι : Type} {d : Device ι} {α : Type}
    [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)]
    {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Claims.SamplingGeneric.Arg Coin}
    {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) {H : Msg → Msg}
    {D : Claims.SamplingGeneric.Compiled Coin A} {L Kₒ S : ℕ} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {δ : ℝ} {reg : AuditStatement → Finset ℕ}
    {Dj n : ℕ} (R : RealizesCommOfPeriodLeakAccepted r c.K H D L Kₒ S lay k δ reg Dj n) (hks : D.KS H)
    (hb : D.commit.Binds H) (hδ : 0 ≤ δ) (εks εbind εtap : ℝ)
    (herr : ∀ ξ i, D.κ δ (R.stmt ξ i) - δ +
      Real.sqrt (D.γ (R.stmt ξ i) * PrB H D.V (R.prover ξ i) (R.stmt ξ i)) ≤ εks)
    (hbind : ∀ ξ i, (Bext H D.V D.E (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ εbind)
    (htap : ∀ ξ i, (Btap H D.V D.E D.commit D.parse R.tops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤
      εtap) :
    r.Sound c p .network (n * (δ + εks + εbind + εtap)) := by
  intro ξ
  have hlin := ksLin_of_periodLeak hsrc H D hks L Kₒ S lay k δ hδ reg Dj R.tops hb
  calc Pouw.pr (fun x : Ω (r.own .network) =>
        r.accepts .network (Function.update ξ (r.own .network) x) ∧
          ¬ r.Holds c p .network (Function.update ξ (r.own .network) x))
      ≤ Pouw.pr (fun x : Ω (r.own .network) => ∃ i,
          r.accepts .network (Function.update ξ (r.own .network) x) ∧
            ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops
              (R.stmt (Function.update ξ (r.own .network) x) i) (R.wit (Function.update ξ (r.own .network) x) i)
              (R.obs (Function.update ξ (r.own .network) x) i)) := by
        refine SecurityProofs.PouwWork.pr_mono fun x ⟨ha, hn⟩ => ?_
        have hn' : ¬ r.commWithin c.K (Function.update ξ (r.own .network) x) := hn
        rw [R.comm_iff] at hn'
        obtain ⟨i, hi⟩ := not_forall.mp hn'
        exact ⟨i, ha, hi⟩
    _ ≤ ∑ i, Pouw.pr (fun x : Ω (r.own .network) =>
          r.accepts .network (Function.update ξ (r.own .network) x) ∧
            ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops
              (R.stmt (Function.update ξ (r.own .network) x) i) (R.wit (Function.update ξ (r.own .network) x) i)
              (R.obs (Function.update ξ (r.own .network) x) i)) := pr_exists_le _
    _ ≤ ∑ _i : Fin n, (δ + εks + εbind + εtap) := by
        refine Finset.sum_le_sum fun i _ => (R.window_le ξ i).trans ?_
        have h := hlin (R.stmt ξ i) (R.scope ξ i) (R.Tape ξ i) (R.prover ξ i)
        have he := herr ξ i
        have hx := hbind ξ i
        have ht := htap ξ i
        linarith
    _ = n * (δ + εks + εbind + εtap) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- **The network certifier's per-period guarantee**, at network accounting's proof of `PeriodLeakAccepted`. -/
theorem CommOfPeriodLeakAccepted : Guarantees.Bridge.CommOfPeriodLeakAccepted := by
  intro ι d α _ C I Ω _ _ Coin _ _ _ A r c p H D L Kₒ S lay k δ reg Dj n R
  exact comm_of_periodLeak NetTiming.periodLeak c p R

open NetWitness in
/-- **`CommOfPeriodLeakAccepted`'s conclusion fails at the verifier that accepts everything** (`acceptAllNet`): the
network audit accepts on every coin while the network certifier's claim fails on it, so the audit's guarantee fails at
error `½`. -/
theorem CommOfPeriodLeakAccepted_nontrivial :
    ¬ acceptAllNet.Sound ⟨0, 1, 0, 0, 0⟩ (fun _ => 1) .network (1 / 2) := by
  intro h
  have h1 := h fun _ => ()
  have hpr : Pouw.pr (fun x : Unit =>
      acceptAllNet.accepts .network (Function.update (fun _ => ()) (acceptAllNet.own .network) x) ∧
        ¬ acceptAllNet.Holds ⟨0, 1, 0, 0, 0⟩ (fun _ => 1) .network
          (Function.update (fun _ => ()) (acceptAllNet.own .network) x)) = 1 :=
    Pouw.pr_eq_one fun _ => ⟨trivial, fun h => h⟩
  rw [hpr] at h1
  norm_num at h1

end Accounting.SecurityProofs.Bridge
