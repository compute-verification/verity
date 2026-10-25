import Definitions.Accounting.Guarantees.JointBridge.BandwidthOfAuditPerRunPeriod
import Definitions.Accounting.Guarantees.BandwidthAudit
import Proofs.Pouw.Prob
import Proofs.Accounting.BandwidthAudit

/-!
# Proof that PoUB's per-period guarantee is the bandwidth audit's

At a value `ξ` of the period's coins, the period is a joint run (`bandwidthRun`) whose bandwidth coins are PoUB's,
carried by the realization's equivalence, and whose other audits' coins are trivial: its schedule and bandwidth verdict
are the period's at `ξ` with PoUB's coordinate replaced. That run realizes the bandwidth audit's game
(`bandwidthRealizes`), since the period's game device doesn't depend on PoUB's own coins, so
`Accounting.Guarantees.BandwidthOfAuditPerRun` bounds its accept-while-short event. Uniform coins carried by an
equivalence keep their probabilities (`Pouw.pr_equiv`), so the same bound holds over PoUB's coordinate.
-/

namespace Accounting.SecurityProofs.Bridge

open Accounting

variable {G N b h s : ℕ} {ι : Type} [Fintype ι] {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
  {I : Type} [DecidableEq I] {Ω : I → Type} {r : PeriodRun d α C Ω}

/-- The joint run a period is at the coins `ξ`, with PoUB's coordinate drawn as the bandwidth audit's coins. -/
def bandwidthRun (real : RealizesBandwidthOfAuditPerRunPeriod G (N := N) (b := b) (h := h) (s := s) r)
    (ξ : (i : I) → Ω i) : Run d α Unit (BandwidthCoins N b h s) Unit where
  acct := r.acct
  win := fun _ => r.win
  schedule := fun ω => r.schedule (Function.update ξ (r.own .poub) (real.coins.symm ω.2.1))
  fits := fun _ => r.fits _
  accepts := fun _ ω => r.accepts .poub (Function.update ξ (r.own .poub) (real.coins.symm ω.2.1))

/-- The joint run a period is at the coins `ξ` realizes the bandwidth audit's game. -/
def bandwidthRealizes (real : RealizesBandwidthOfAuditPerRunPeriod G (N := N) (b := b) (h := h) (s := s) r)
    (ξ : (i : I) → Ω i) : RealizesBandwidth G (bandwidthRun real ξ) where
  dev := fun _ => real.dev ξ
  accepts_iff := fun _ y _ => by
    show r.accepts .poub (Function.update ξ (r.own .poub) (real.coins.symm y)) ↔
      Bandwidth.Accepts (real.dev ξ) y.1 y.2.1 y.2.2
    rw [real.accepts_iff, real.dev_update, Function.update_self, Equiv.apply_symm_apply]
  ingress_le := fun _ y _ => by
    show (real.dev ξ).ingress y.1 y.2.1 ≤ r.read (Function.update ξ (r.own .poub) (real.coins.symm y))
    have := real.ingress_le (Function.update ξ (r.own .poub) (real.coins.symm y))
    rwa [real.dev_update, Function.update_self, Equiv.apply_symm_apply] at this

/-- PoUB's per-period guarantee from the bandwidth audit's per-run guarantee. -/
theorem bandwidthOfAuditPerRunPeriod_of (hsrc : Accounting.Guarantees.BandwidthOfAuditPerRun) :
    Accounting.Guarantees.Bridge.BandwidthOfAuditPerRunPeriod := by
  intro N b h s G k lam ι α _ _ d C I _ Ω _ r c p hN real hQ ξ
  have key := hsrc N b h s G k lam ι α Unit Unit d (bandwidthRun real ξ) hN (bandwidthRealizes real ξ) () ()
  refine le_trans (Pouw.pr_mono ?_) ((Pouw.pr_equiv real.coins _).le.trans key)
  rintro x ⟨hacc, hshort⟩
  have hx : Function.update ξ (r.own .poub) (real.coins.symm (real.coins x)) = Function.update ξ (r.own .poub) x := by
    rw [Equiv.symm_apply_apply]
  refine ⟨?_, ?_⟩
  · show r.accepts .poub (Function.update ξ (r.own .poub) (real.coins.symm (real.coins x)))
    rw [hx]
    exact hacc
  · show ((r.read (Function.update ξ (r.own .poub) (real.coins.symm (real.coins x))) : ℕ) : ℝ) <
      ((k * b : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * r.win.len : ℕ) : ℝ) - ((G * b : ℕ) : ℝ) - lam
    rw [hx]
    have hlt : (r.read (Function.update ξ (r.own .poub) x) : ℝ) < c.Q := not_le.1 hshort
    linarith

/-- **PoUB's per-period guarantee is the bandwidth audit's**
(`Accounting.Guarantees.Bridge.BandwidthOfAuditPerRunPeriod`). -/
theorem BandwidthOfAuditPerRunPeriod : Accounting.Guarantees.Bridge.BandwidthOfAuditPerRunPeriod :=
  bandwidthOfAuditPerRunPeriod_of Accounting.SecurityProofs.BandwidthOfAuditPerRun

end Accounting.SecurityProofs.Bridge
