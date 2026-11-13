module

public import Definitions.Accounting.Guarantees.JointBridge.CommOfPooledLeakAccepted
public import Definitions.NetworkCertifier.Guarantees.PooledLeak
public import Proofs.NetworkCertifier.PooledLeak
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
import Proofs.Pouw.Prob

/-!
# The network certifier's per-period guarantee, pooled over a monitoring period of sampled audits (proof)

`Accounting.SecurityProofs.Bridge.CommOfPooledLeakAccepted`: `comm_of_pooledLeak` at `PooledLeakAccepted`'s proof
(`NetTiming.pooledLeak`, which `NetTiming.SecurityProofs.PooledLeakAccepted` is). On a run that realizes the network
certifier's monitoring period of `m` sampled audits (`Accounting.JointBridge.RealizesCommOfPooledLeakAccepted`), under
the two inputs `hesc` and `hleak`, the network audit's `Sound` holds at `e^(−q·Fmax) + Σ e + Σ e'`.

At a value `ξ` of the other audits' coins, the network audit's own coins are the execution phases' coins (`coins`), so
the probability over them is the probability over the phases' coins (`Pouw.pr_equiv`). There the network audit
accepting with its claim failing means every verification phase's audit accepts (`accepts_imp`) while the period's
good event fails (`comm_iff`), which is the event `PooledLeakAccepted`'s first conjunct bounds, read at one secret:
the bound is on the period's coins and no secret enters it. Its `δ` is taken at `e^(−q·Fmax)`, the least it allows.

`PooledLeakAccepted` is taken as it is stated, so the proof also supplies what only its count reads, none of which the
realization holds: each execution phase's number of proof blocks (one past the largest index in its incorrect sets),
the bits a proof block commits (the most over those), the schedules (the run's, at the phases' coins) and the most
cells each gives a connection. The network certifier's trust (`trusted-host`) and its record format holding the run's
schedules are the realization's `trusted` and `fits`.
-/

@[expose] public section

namespace Accounting.SecurityProofs.Bridge

open Claims Claims.Sampling NetTiming Accounting.JointBridge

/-- `Guarantees.Bridge.CommOfPooledLeakAccepted` from `PooledLeakAccepted`. -/
theorem comm_of_pooledLeak (hsrc : NetTiming.Guarantees.PooledLeakAccepted) {ι : Type} {d : Device ι} {α : Type}
    [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)]
    {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) {L Kₒ S : ℕ} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {reg : AuditStatement → Finset ℕ} {Dj m Fmax : ℕ}
    (R : RealizesCommOfPooledLeakAccepted r c.K L Kₒ S lay k reg Dj m Fmax) (e e' : Fin m → ℝ) (he : ∀ i, 0 ≤ e i)
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hesc : PeriodEscapes Set.univ R.acc R.W q e)
    (hleak : PhaseLeaks lay k reg Dj R.sw R.Pw R.ρ R.op Set.univ R.acc R.W R.record e') :
    r.Sound c p .network (Real.exp (-(q * Fmax)) + ∑ i, e i + ∑ i, e' i) := by
  intro ξ
  -- What only `PooledLeakAccepted`'s count reads: proof blocks, their bits, and the cells of each connection.
  obtain ⟨n, hn⟩ : ∃ n : Fin m → ℕ, ∀ (ω : Fin m → R.Coin) (i : Fin m), R.W ξ ω i ⊆ Finset.range (n i) :=
    ⟨fun i => (Finset.univ.sup fun ω : Fin m → R.Coin => (R.W ξ ω i).sup id) + 1, fun ω i v hv =>
      Finset.mem_range.2 (Nat.lt_add_one_of_le ((Finset.le_sup (f := id) hv).trans
        (Finset.le_sup (f := fun ω : Fin m → R.Coin => (R.W ξ ω i).sup id) (Finset.mem_univ ω))))⟩
  obtain ⟨b, hb⟩ : ∃ b : ℕ, ∀ (i : Fin m) (v : ℕ), v < n i → faultyBits (R.sw i).prog (R.Pw i) {v} ≤ b :=
    ⟨Finset.univ.sup fun i : Fin m => (Finset.range (n i)).sup fun v => faultyBits (R.sw i).prog (R.Pw i) {v},
      fun i v hv => (Finset.le_sup (f := fun v => faultyBits (R.sw i).prog (R.Pw i) {v}) (Finset.mem_range.2 hv)).trans
        (Finset.le_sup (f := fun i : Fin m => (Finset.range (n i)).sup fun v =>
          faultyBits (R.sw i).prog (R.Pw i) {v}) (Finset.mem_univ i))⟩
  obtain ⟨cells, hcells⟩ : ∃ cells : R.Conn → ℕ, ∀ (ω : Fin m → R.Coin) (κ : R.Conn),
      (R.sched (Function.update ξ (r.own .network) (R.coins.symm ω))).cells κ ≤ cells κ :=
    ⟨fun κ => Finset.univ.sup fun ω : Fin m → R.Coin =>
        (R.sched (Function.update ξ (r.own .network) (R.coins.symm ω))).cells κ,
      fun ω κ => Finset.le_sup (f := fun ω : Fin m → R.Coin =>
        (R.sched (Function.update ξ (r.own .network) (R.coins.symm ω))).cells κ) (Finset.mem_univ ω)⟩
  have hF : Real.log (1 / Real.exp (-(q * Fmax))) ≤ q * Fmax :=
    le_of_eq (by rw [one_div, Real.exp_neg, inv_inv, Real.log_exp])
  -- `PooledLeakAccepted`'s first conjunct at one secret, on the execution phases' coins.
  have key : Pouw.pr (fun ω : Fin m → R.Coin => (∀ i, R.acc ξ i ω (ω i)) ∧
      ¬PooledGood lay k reg Dj R.sw R.Pw R.ρ R.op Fmax (R.W ξ ω)
        (R.run R.device (R.sched (Function.update ξ (r.own .network) (R.coins.symm ω)),
          R.srv (Function.update ξ (r.own .network) (R.coins.symm ω)))).2) ≤
      Real.exp (-(q * Fmax)) + ∑ i, e i + ∑ i, e' i :=
    (hsrc (Ω := R.Coin) (Z := Unit) lay k reg Dj R.sw R.Pw R.ρ R.op n b hb R.arr R.fmt R.device R.run R.trusted cells
      (Set.range fun ω : Fin m → R.Coin => R.sched (Function.update ξ (r.own .network) (R.coins.symm ω)))
      (fun s hs κ => by obtain ⟨ω, rfl⟩ := hs; exact hcells ω κ)
      (fun s hs => by obtain ⟨ω, rfl⟩ := hs; exact R.fits _)
      Set.univ (fun _ ω => R.sched (Function.update ξ (r.own .network) (R.coins.symm ω)))
      (fun _ ω => R.srv (Function.update ξ (r.own .network) (R.coins.symm ω)))
      (fun _ _ ω => ⟨ω, rfl⟩) (fun _ _ => ()) (fun _ _ _ => Subsingleton.elim _ _)
      (fun _ => R.acc ξ) (fun _ => R.acc_prefix ξ) (fun _ => R.W ξ) (fun _ => R.W_prefix ξ) (fun _ => hn)
      e e' he q hq0 hq1 (fun _ _ => hesc ξ (Set.mem_univ ξ)) (fun _ _ => hleak ξ (Set.mem_univ ξ))
      (Real.exp (-(q * Fmax))) (Real.exp_pos _) Fmax hF).1 () (Set.mem_univ ())
  -- The network audit's own coins are the execution phases' coins.
  refine le_trans (Pouw.pr_mono ?_) ((Pouw.pr_equiv R.coins _).trans_le key)
  intro x hx
  obtain ⟨ha, hn'⟩ := hx
  have ha' : r.accepts .network (Function.update ξ (r.own .network) (R.coins.symm (R.coins x))) := by
    rw [Equiv.symm_apply_apply]
    exact ha
  refine ⟨R.accepts_imp ξ (R.coins x) ha', fun hg => hn' ?_⟩
  have hc := (R.comm_iff ξ (R.coins x)).2 hg
  rw [Equiv.symm_apply_apply] at hc
  exact hc

/-- **The network certifier's per-period guarantee, pooled**, at `PooledLeakAccepted`'s proof. -/
theorem CommOfPooledLeakAccepted : Guarantees.Bridge.CommOfPooledLeakAccepted := by
  intro ι d α _ C I Ω _ _ r c p L Kₒ S lay k reg Dj m Fmax R e e' he q hq0 hq1 hesc hleak
  exact comm_of_pooledLeak NetTiming.pooledLeak c p R e e' he q hq0 hq1 hesc hleak

end Accounting.SecurityProofs.Bridge
